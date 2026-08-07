"""
A transformer block whose attention and feed-forward sub-layers are each wrapped in a
:class:`~olmo_core.nn.hyper_connections.HyperConnection`.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.exceptions import OLMoConfigurationError

from ..attention.base import SequenceMixerConfig
from ..attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..hyper_connections import HyperConnectionConfig
from ..layer_norm import LayerNormConfig
from .block import TransformerBlock

__all__ = ["HyperConnectionTransformerBlock"]


class HyperConnectionTransformerBlock(TransformerBlock):
    """
    A hyper-connected transformer block.

    The sub-layer ordering is OLMo-2's, the same as
    :class:`~olmo_core.nn.transformer.ReorderedNormTransformerBlock`: the norm is applied to the
    *output* of attention and of the feed-forward rather than to their input, so an ordinary
    block computes ``z = z + LN(f(z))``. Here each of the two ``+ LN(f(...))`` steps is instead
    routed through its own :class:`~olmo_core.nn.hyper_connections.HyperConnection`, which reads
    one branch input out of ``n`` residual streams, runs ``LN(f(...))`` on it exactly once, and
    writes the result back to every stream while mixing the streams among themselves.

    Input and output are both ``(batch_size, seq_len, n_streams, d_model)``. A
    ``(batch_size, seq_len, d_model)`` input is accepted and lifted into ``n`` identical
    streams, which is what lets a single block be tested on its own.

    With ``hyper_connection.init_noise_std = 0`` the block is numerically identical to
    ``ReorderedNormTransformerBlock`` at initialisation, in every stream.

    :param d_model: The model dimensionality.
    :param block_idx: The index of the block within the model.
    :param n_layers: The total number of blocks in the model.
    :param sequence_mixer: The sequence mixer (attention) config.
    :param feed_forward: The feed-forward config.
    :param layer_norm: The layer norm config for both sub-layer norms.
    :param hyper_connection: The hyper-connection config, applied to both sub-layers. Each
        sub-layer gets its own parameters.
    :param dropout: Dropout probability, applied to each branch output before it is written back
        to the streams.
    :param attention_residual_alpha: Must be left at 1.0; the write-out gate takes its place.
    :param feed_forward_residual_alpha: Must be left at 1.0; likewise.
    :param init_device: The device used when initializing parameters.
    :param cache: A shared buffer cache.

    :raises OLMoConfigurationError: If a residual alpha is set, since the hyper-connection's
        ``h_post`` gate, not a fixed scalar, controls how the branch output reaches the streams.
    """

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        layer_norm: LayerNormConfig,
        hyper_connection: Optional[HyperConnectionConfig] = None,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        if attention_residual_alpha != 1.0 or feed_forward_residual_alpha != 1.0:
            raise OLMoConfigurationError(
                "residual alphas are not supported by "
                f"{self.__class__.__name__}: the hyper-connection's learned 'h_post' gate is "
                "what scales a branch output on its way into each stream, and a second fixed "
                "scalar in front of it would only be absorbed into that gate"
            )

        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            layer_norm=layer_norm,
            dropout=0.0,
            init_device=init_device,
            cache=cache,
        )

        hc_config = hyper_connection if hyper_connection is not None else HyperConnectionConfig()
        self.hc_config = hc_config
        self.n_streams = hc_config.n_streams

        # `TransformerBlock` built two `ResidualStream`s that a hyper-connected block has no use
        # for. Drop them rather than leave dead modules that a hook or a TP plan could find.
        del self.attention_residual_stream
        del self.feed_forward_residual_stream

        self.attention_hc = hc_config.build(init_device=init_device)
        self.feed_forward_hc = hc_config.build(init_device=init_device)
        self.branch_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def num_routing_params(self) -> int:
        """
        The number of routing parameters this block adds over an ordinary block.

        :returns: Twice the per-sub-layer routing parameter count.
        """
        return 2 * self.hc_config.num_params()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the block on the residual streams.

        :param x: The input, shape ``(batch_size, seq_len, n_streams, d_model)``. A
            ``(batch_size, seq_len, d_model)`` input is expanded into ``n`` identical streams.
        :param loss_div_factor: Unused, accepted for interface compatibility.
        :param kwargs: Forwarded to the sequence mixer.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        """
        del loss_div_factor

        def attention_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(self.attention_norm(self.attention(h, **kwargs)))

        def feed_forward_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(self.feed_forward_norm(self.feed_forward(h)))

        streams = self.attention_hc(x, attention_branch)
        return self.feed_forward_hc(streams, feed_forward_branch)

    def apply_tp(
        self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False
    ):
        """
        :raises NotImplementedError: Always. Tensor parallelism over a four-dimensional residual
            stream needs its own placement plan for the routing parameters and for the stream
            dimension, and none has been written or tested yet.
        """
        del tp_mesh, input_layout, float8_enabled
        raise NotImplementedError(
            "tensor parallelism is not implemented for HyperConnectionTransformerBlock"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """
        :raises NotImplementedError: Always. Context parallelism has not been validated against
            the four-dimensional stream layout.
        """
        del cp_mesh, ring, uly
        raise NotImplementedError(
            "context parallelism is not implemented for HyperConnectionTransformerBlock"
        )
