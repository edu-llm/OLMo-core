"""
A transformer block whose attention and feed-forward sub-layers are each wrapped in a
:class:`~olmo_core.nn.hyper_connections.HyperConnection`.
"""

from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.exceptions import OLMoConfigurationError

from ..attention.base import SequenceMixerConfig
from ..attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..hyper_connections import HyperConnection, HyperConnectionConfig
from ..layer_norm import LayerNormConfig
from .block import TransformerBlock

__all__ = ["HyperConnectionBlockMixin", "HyperConnectionTransformerBlock"]


class HyperConnectionBlockMixin:
    """
    What every hyper-connected block has in common, whatever it wraps.

    A dense block and an MoE block differ in what their sub-layers are and in how many of them
    there are; they do not differ in how the hyper-connections are built, counted, or refused
    under tensor and context parallelism. That shared half lives here so that
    :class:`~olmo_core.nn.transformer.HyperConnectionTransformer` can recognise a hyper-connected
    block with one ``isinstance`` — the alternative is a growing tuple of block classes in
    ``_validate_block``, which is the kind of list that stops being complete.

    A plain mixin rather than a common base class, deliberately. The dense block extends
    :class:`~olmo_core.nn.transformer.TransformerBlock` and the MoE ones extend
    :class:`~olmo_core.nn.transformer.MoETransformerBlock`, which are siblings under
    ``TransformerBlockBase``; anything shared has to arrive from the side.

    Subclasses call :meth:`_init_hyper_connections` from their own ``__init__`` after the
    wrapped block is built, and are responsible for using ``self.hyper_connections`` in their
    ``forward``.
    """

    #: Set by :meth:`_init_hyper_connections`.
    hc_config: HyperConnectionConfig
    n_streams: int
    hc_names: Tuple[str, ...]
    branch_dropout: nn.Module

    def _init_hyper_connections(
        self,
        hc_config: Optional[HyperConnectionConfig],
        *,
        names: Tuple[str, ...],
        init_device: str = "cpu",
        dropout: float = 0.0,
    ) -> None:
        """
        Build one :class:`~olmo_core.nn.hyper_connections.HyperConnection` per wrapped sub-layer.

        :param hc_config: The hyper-connection config, or ``None`` for the default.
        :param names: The attribute name for each wrapped sub-layer's hyper-connection, in the
            order the block applies them. Each gets its own parameters.
        :param init_device: The device to allocate routing parameters on.
        :param dropout: Dropout applied to each branch output before it is written back.
        """
        assert isinstance(self, nn.Module)
        config = hc_config if hc_config is not None else HyperConnectionConfig()
        self.hc_config = config
        self.n_streams = config.n_streams
        self.hc_names = tuple(names)
        for name in self.hc_names:
            self.add_module(name, config.build(init_device=init_device))
        self.branch_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def hyper_connections(self) -> Tuple[HyperConnection, ...]:
        """
        Every :class:`~olmo_core.nn.hyper_connections.HyperConnection` this block owns, in the
        order the forward pass applies them.

        :returns: The hyper-connections.
        """
        assert isinstance(self, nn.Module)
        return tuple(cast(HyperConnection, getattr(self, name)) for name in self.hc_names)

    @property
    def num_routing_params(self) -> int:
        """
        The number of routing parameters this block adds over an ordinary block.

        Measured off the built modules rather than recomputed from the config, so that a block
        whose wrapped sub-layer count changes cannot quietly keep reporting the old number.

        :returns: The parameter count.
        """
        return sum(
            sum(p.numel() for p in hc.parameters(recurse=False)) for hc in self.hyper_connections
        )

    def compute_stream_metrics(self, reset: bool = True) -> Dict[str, Any]:
        """
        Every hyper-connection's diagnostics, keyed by which sub-layer it wraps.

        Separate from ``compute_metrics``, which on an MoE block already means the router's
        metrics. Merging the two here would make a dense block and an MoE block disagree about
        what the method returns.

        :param reset: Whether to clear the recorded values afterwards.

        :returns: A mapping from ``"<sub-layer>/<metric>"`` to (value, reduction).
        """
        out: Dict[str, Any] = {}
        for name, hc in zip(self.hc_names, self.hyper_connections):
            # `attention_hc` reads better as `attention` in a metrics panel, and the suffix
            # carries no information a reader of this list does not already have.
            label = name[: -len("_hc")] if name.endswith("_hc") else name
            for metric, value in hc.compute_metrics(reset=reset).items():
                out[f"{label}/{metric}"] = value
        return out

    def reset_stream_metrics(self) -> None:
        """
        Forget what the last forward pass measured, on every hyper-connection in this block.
        """
        for hc in self.hyper_connections:
            hc.reset_metrics()

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        *,
        input_layout: Optional[Placement] = None,
        float8_enabled: bool = False,
    ):
        """
        :raises NotImplementedError: Always. Tensor parallelism over a four-dimensional residual
            stream needs its own placement plan for the routing parameters and for the stream
            dimension, and none has been written or tested yet.
        """
        del tp_mesh, input_layout, float8_enabled
        raise NotImplementedError(
            f"tensor parallelism is not implemented for {type(self).__name__}"
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
            f"context parallelism is not implemented for {type(self).__name__}"
        )


class HyperConnectionTransformerBlock(HyperConnectionBlockMixin, TransformerBlock):
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

        # `TransformerBlock` built two `ResidualStream`s that a hyper-connected block has no use
        # for. Drop them rather than leave dead modules that a hook or a TP plan could find.
        del self.attention_residual_stream
        del self.feed_forward_residual_stream

        self._init_hyper_connections(
            hyper_connection,
            names=("attention_hc", "feed_forward_hc"),
            init_device=init_device,
            dropout=dropout,
        )

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
