"""
Mixture-of-experts transformer blocks whose residual adds are routed through
:class:`~olmo_core.nn.hyper_connections.HyperConnection` s.

Why this file exists separately from :mod:`~olmo_core.nn.transformer.hc_block`: a dense block
routes its residual through a :class:`~olmo_core.nn.residual_stream.ResidualStream` module, so
wrapping it was a substitution. The MoE blocks in :mod:`~olmo_core.nn.transformer.block` write
``h = x + dropout(...)`` inline, three times over five classes, so each one needs its own
wrapper. The wrapping is mechanical and the correctness standard is
``src/test/nn/hyper_connections_test.py``'s, which this matches: bit-exact baseline equivalence
at initialisation with the noise off, doubly stochastic mixers, shapes, gradients, save/resume,
and float32 routing under bfloat16.

**What is deliberately NOT wrapped, and the reason is the whole design of this file.** The MoE
sub-layer itself is untouched: the hyper-connection reads one ``(batch, seq, d_model)`` vector
out of the ``n`` streams, hands exactly that to ``feed_forward_moe``, and writes the result
back. The router, the dispatch, the expert MLPs, the load-balancing loss and the z-loss all see
the three-dimensional tensor they have always seen and are not modified in any way. Nothing
under ``src/olmo_core/nn/moe/`` is touched by this work.

**Expert parallelism raises rather than pretending.** See
:meth:`HyperConnectionMoEBlockMixin.apply_ep` for the argument; it is the same position
``hc_block.py`` already takes on tensor and context parallelism, for the same reason.
"""

from typing import Optional, Tuple, Union

import torch
from torch.distributed import DeviceMesh

from olmo_core.exceptions import OLMoConfigurationError

from ..attention.base import SequenceMixerConfig
from ..buffer_cache import BufferCache
from ..feed_forward import FeedForwardConfig
from ..hyper_connections import HyperConnection, HyperConnectionConfig
from ..layer_norm import LayerNormConfig
from ..moe import MoEConfig
from .block import (
    MoEHybridReorderedNormTransformerBlock,
    MoEHybridTransformerBlock,
    MoEHybridTransformerBlockBase,
    MoEReorderedNormTransformerBlock,
    MoETransformerBlock,
)
from .hc_block import HyperConnectionBlockMixin

__all__ = [
    "HyperConnectionMoEBlockMixin",
    "HyperConnectionMoETransformerBlock",
    "HyperConnectionMoEReorderedNormTransformerBlock",
    "HyperConnectionMoEHybridTransformerBlock",
    "HyperConnectionMoEHybridReorderedNormTransformerBlock",
]


class HyperConnectionMoEBlockMixin(HyperConnectionBlockMixin):
    """
    The half of a hyper-connected MoE block that does not depend on which MoE block it is.

    Adds three things to :class:`~olmo_core.nn.transformer.HyperConnectionBlockMixin`: the
    ``__init__`` that every MoE variant shares, the FSDP plan that keeps the router and the
    routing parameters in float32, and the expert-parallelism refusal.
    """

    #: Declared for the type checker; it comes from ``MoETransformerBlock``, which every
    #: concrete user of this mixin also inherits.
    feed_forward_moe: "torch.nn.Module"

    def _init_moe_hyper_connections(
        self,
        hyper_connection: Optional[HyperConnectionConfig],
        *,
        names: Tuple[str, ...],
        init_device: str,
        dropout: float,
        attention_residual_alpha: float,
        feed_forward_residual_alpha: float,
    ) -> None:
        """
        Validate the residual alphas and build the hyper-connections.

        :param hyper_connection: The hyper-connection config, applied to every wrapped
            sub-layer. Each gets its own parameters.
        :param names: One attribute name per wrapped sub-layer.
        :param init_device: The device to allocate routing parameters on.
        :param dropout: Dropout applied to a branch output before it is written back.
        :param attention_residual_alpha: Must be 1.0.
        :param feed_forward_residual_alpha: Must be 1.0.

        :raises OLMoConfigurationError: If either residual alpha is set.
        """
        if attention_residual_alpha != 1.0 or feed_forward_residual_alpha != 1.0:
            raise OLMoConfigurationError(
                f"residual alphas are not supported by {type(self).__name__}: the "
                "hyper-connection's learned 'h_post' gate is what scales a branch output on "
                "its way into each stream, and a second fixed scalar in front of it would only "
                "be absorbed into that gate"
            )
        self._init_hyper_connections(
            hyper_connection, names=names, init_device=init_device, dropout=dropout
        )

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: Optional[object] = None,
        **fsdp_kwargs,
    ):
        """
        Shard the block, holding the router and the routing parameters in float32.

        **THIS OVERRIDE IS LOAD-BEARING AND ITS ABSENCE WAS SILENT.**
        ``MoEHybridTransformerBlockBase.apply_fsdp`` shards ``feed_forward_moe.router`` under
        ``MixedPrecisionPolicy(param_dtype=torch.float32)`` and says why: the router's decision
        is a discrete argmax over small differences and rounding it changes which expert a token
        goes to. ``MoEReorderedNormTransformerBlock.apply_fsdp`` -- which is what the arms in
        ``docs/hc-ablation/EXPERIMENT-DESIGN.md`` actually run -- does not, so under
        ``param_dtype=bfloat16`` the router's weights are bfloat16 inside the forward and the
        ``.float()`` in the router upcasts numbers that have already been rounded.

        The same is true of every :class:`~olmo_core.nn.hyper_connections.HyperConnection`, and
        it is worse there. The whole claim of mHC is that ``H_res`` is doubly stochastic, and in
        bfloat16 the Sinkhorn fixed point is reached to about two decimal digits and the row and
        column sums drift far enough off 1 that the property stops holding. The module computes
        its routing in float32 and the test suite asserts that it does -- but it can only
        compute in float32 from whatever the parameter holds, and FSDP decides that. A
        bfloat16-sharded gate is a float32 computation over three decimal digits of input.

        Neither shows up as an error and neither shows up on a CPU, where FSDP is not applied.
        So both are sharded here under their own float32 policies, before the block root.

        :param dp_mesh: The data-parallel mesh.
        :param prefetch_factor: Accepted and ignored; the prefetch plans in ``block.py`` name
            sub-modules this method has already placed under their own policies.
        :param wrapping_strategy: Accepted and ignored, for the same reason.
        :param fsdp_kwargs: Forwarded to ``fully_shard`` for the block root.
        """
        del prefetch_factor, wrapping_strategy
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        from ..hyper_connections import HyperConnection

        float32 = MixedPrecisionPolicy(param_dtype=torch.float32)
        fully_shard(self.feed_forward_moe.router, mesh=dp_mesh, mp_policy=float32)
        for module in self.hyper_connections:
            if isinstance(module, HyperConnection) and any(module.parameters(recurse=False)):
                fully_shard(module, mesh=dp_mesh, mp_policy=float32)
        # The root last, and with the caller's own kwargs: everything already sharded above is
        # skipped by `fully_shard`, and the wrapping strategy is deliberately not honoured here
        # because the fine-grained plans in `block.py` name sub-modules by attribute and would
        # re-shard the two this method just placed.
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        """
        :raises NotImplementedError: Always.

        Expert parallelism is the one form of parallelism whose interaction with an ``n``-stream
        residual is not obviously safe, and the failure mode is quiet rather than loud, so this
        refuses rather than applying a plan written for a three-dimensional hidden state.

        The mechanics are the argument. ``MoEBase.apply_ep`` shards the experts over ``ep_mesh``
        and turns on the combined forward path in the hybrid blocks, which interleaves the
        expert all-to-all with attention and the dense MLP so that communication overlaps
        compute. That interleaving reads the block input ``x`` twice — once as the MoE branch's
        input and once as attention's — and writes three residual adds whose order is fixed by
        where the ``handle.wait()`` calls fall. A hyper-connection changes what each of those
        reads and writes means: the MoE branch's input is ``h_pre^T Z`` rather than ``x``, and
        each write goes through ``H_res``. Re-deriving the overlap under that contract is a real
        piece of work and none of it has been written, let alone run on hardware with more than
        one device.

        Refusing is cheap and correct here for the same reason it is in ``hc_block.py``: a plan
        that silently applied would produce a model that trains, reports a loss curve, and is
        computing something other than what the config says. There is no error anywhere in that
        failure, which is what makes it worth a hard refusal.

        Data parallelism (FSDP, HSDP, DDP) is inherited and unchanged, and is what these blocks
        are meant to be run under.
        """
        del ep_mesh, kwargs
        raise NotImplementedError(
            f"expert parallelism is not implemented for {type(self).__name__}. The "
            "hyper-connection changes what the block's residual reads and writes mean, and the "
            "combined forward path that expert parallelism turns on interleaves those reads and "
            "writes with the expert all-to-all. Run these blocks under data parallelism (FSDP, "
            "HSDP or DDP), which is unchanged, and see the method's docstring for what closing "
            "this would take."
        )


class HyperConnectionMoETransformerBlock(HyperConnectionMoEBlockMixin, MoETransformerBlock):
    """
    :class:`~olmo_core.nn.transformer.MoETransformerBlock` with both of its residual adds routed
    through hyper-connections.

    The unwrapped block computes ``h = x + attention(LN(x))`` then
    ``h + moe(LN(h))``. Here each ``+`` is instead a
    :class:`~olmo_core.nn.hyper_connections.HyperConnection`, which reads one branch input out
    of ``n`` streams, runs the sub-layer on it exactly once, and writes the result back to every
    stream while mixing the streams among themselves.

    Input and output are both ``(batch_size, seq_len, n_streams, d_model)``. A
    ``(batch_size, seq_len, d_model)`` input is accepted and lifted into ``n`` identical
    streams, which is what lets a single block be tested on its own.

    With ``hyper_connection.init_noise_std = 0`` this is numerically identical to
    :class:`~olmo_core.nn.transformer.MoETransformerBlock` at initialisation, in every stream.

    :param hyper_connection: The hyper-connection config, applied to both sub-layers. Each
        sub-layer gets its own parameters.
    :param attention_residual_alpha: Must be left at 1.0; the write-out gate takes its place.
    :param feed_forward_residual_alpha: Must be left at 1.0; likewise.

    See :class:`~olmo_core.nn.transformer.MoETransformerBlock` for the other parameters.

    :raises OLMoConfigurationError: If a residual alpha is set.
    """

    #: The wrapped sub-layers, in forward order.
    HC_NAMES = ("attention_hc", "feed_forward_moe_hc")

    #: Declared for the type checker; see the note on the hybrid base.
    attention_hc: HyperConnection
    feed_forward_moe_hc: HyperConnection

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        hyper_connection: Optional[HyperConnectionConfig] = None,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
            # The wrapped block's own dropout stays off: a hyper-connected block applies dropout
            # to the branch output before the write-out gate, which is the same position and one
            # module rather than two.
            dropout=0.0,
            init_device=init_device,
            cache=cache,
        )
        self._init_moe_hyper_connections(
            hyper_connection,
            names=self.HC_NAMES,
            init_device=init_device,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
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
        :param loss_div_factor: Forwarded to the MoE, which uses it to normalise its auxiliary
            losses.
        :param kwargs: Forwarded to the sequence mixer.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        """

        def attention_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(self.attention(self.attention_norm(h), **kwargs))

        def moe_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(
                self.feed_forward_moe(self.feed_forward_norm(h), loss_div_factor=loss_div_factor)
            )

        streams = self.attention_hc(x, attention_branch)
        return self.feed_forward_moe_hc(streams, moe_branch)


class HyperConnectionMoEReorderedNormTransformerBlock(
    HyperConnectionMoETransformerBlock, MoEReorderedNormTransformerBlock
):
    """
    :class:`~olmo_core.nn.transformer.MoEReorderedNormTransformerBlock` with both of its residual
    adds routed through hyper-connections.

    The OLMo-2 ordering: the norm is applied to the *output* of each sub-layer rather than to its
    input, so the unwrapped block computes ``h = x + LN(attention(x))`` then
    ``h + LN(moe(h))``. This is the ordering ``smallmoe``, ``small_hybrid_moe`` and
    ``olmoe_1B_7B`` all use, and so the one the ablation actually runs.

    See :class:`HyperConnectionMoETransformerBlock` for the parameters.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the block on the residual streams.

        :param x: The input, shape ``(batch_size, seq_len, n_streams, d_model)``, or
            ``(batch_size, seq_len, d_model)`` to be expanded.
        :param loss_div_factor: Forwarded to the MoE.
        :param kwargs: Forwarded to the sequence mixer.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        """

        def attention_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(self.attention_norm(self.attention(h, **kwargs)))

        def moe_branch(h: torch.Tensor) -> torch.Tensor:
            return self.branch_dropout(
                self.feed_forward_norm(self.feed_forward_moe(h, loss_div_factor=loss_div_factor))
            )

        streams = self.attention_hc(x, attention_branch)
        return self.feed_forward_moe_hc(streams, moe_branch)


class _HyperConnectionMoEHybridBlockBase(
    HyperConnectionMoEBlockMixin, MoEHybridTransformerBlockBase
):
    """
    The construction and the forward pass shared by the two hybrid variants.

    A hybrid block runs three sub-layers — attention, a dense feed-forward and the MoE — and the
    unwrapped version adds all three into one residual. Which of the two orderings is in play
    only changes where the norms sit, which is what the two subclasses supply.

    **The MoE branch reads the block input, not the post-attention state, and that is inherited
    rather than chosen.** ``MoEHybridTransformerBlockBase.forward`` computes
    ``sparse_forward(x) + dense_forward(x)``, so the sparse half is a function of ``x`` while the
    dense half's feed-forward is a function of the post-attention state. Three residual adds,
    two of which are sequential and one of which is parallel to both. The wrapping preserves
    exactly that graph: the MoE hyper-connection reads the *incoming* streams and the two dense
    hyper-connections chain, and the three write-outs compose in the same order the additions
    did. Straightening it into a chain would be a different block, which is not what an ablation
    of the residual mixer is allowed to change.
    """

    #: Forward order: attention, then the dense feed-forward, then the MoE, which reads the
    #: incoming streams rather than the post-attention ones.
    HC_NAMES = ("attention_hc", "feed_forward_hc", "feed_forward_moe_hc")

    #: Declared for the type checker; the modules themselves are registered by
    #: ``_init_hyper_connections`` from :data:`HC_NAMES`, which is the one place the names live.
    attention_hc: HyperConnection
    feed_forward_hc: HyperConnection
    feed_forward_moe_hc: HyperConnection

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        sequence_mixer: SequenceMixerConfig,
        feed_forward: FeedForwardConfig,
        feed_forward_moe: MoEConfig,
        layer_norm: LayerNormConfig,
        hyper_connection: Optional[HyperConnectionConfig] = None,
        dropout: float = 0.0,
        attention_residual_alpha: float = 1.0,
        feed_forward_residual_alpha: float = 1.0,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            sequence_mixer=sequence_mixer,
            feed_forward=feed_forward,
            feed_forward_moe=feed_forward_moe,
            layer_norm=layer_norm,
            dropout=0.0,
            init_device=init_device,
            cache=cache,
        )
        self._init_moe_hyper_connections(
            hyper_connection,
            names=self.HC_NAMES,
            init_device=init_device,
            dropout=dropout,
            attention_residual_alpha=attention_residual_alpha,
            feed_forward_residual_alpha=feed_forward_residual_alpha,
        )

    @property
    def use_combined_forward(self) -> bool:
        """
        Always ``False``.

        The combined forward is an expert-parallel optimisation and
        :meth:`HyperConnectionMoEBlockMixin.apply_ep` refuses expert parallelism, so it can never
        be reached; saying so here rather than leaving the inherited property to work it out
        keeps the two facts from drifting apart.

        :returns: ``False``.
        """
        return False

    @use_combined_forward.setter
    def use_combined_forward(self, should_use: bool):
        """
        :raises NotImplementedError: If asked to turn the combined forward on.
        """
        if should_use:
            raise NotImplementedError(
                f"the combined forward path is not implemented for {type(self).__name__}; it "
                "interleaves the residual adds with the expert all-to-all, and a "
                "hyper-connection changes what each of those adds means"
            )

    def combined_forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        :raises NotImplementedError: Always. See :meth:`use_combined_forward`.
        """
        del x, loss_div_factor, kwargs
        raise NotImplementedError(
            f"the combined forward path is not implemented for {type(self).__name__}"
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

        :param x: The input, shape ``(batch_size, seq_len, n_streams, d_model)``, or
            ``(batch_size, seq_len, d_model)`` to be expanded.
        :param loss_div_factor: Forwarded to the MoE.
        :param kwargs: Forwarded to the sequence mixer.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        """
        streams = self.attention_hc(x, lambda h: self._attention_branch(h, **kwargs))
        streams = self.feed_forward_hc(streams, self._feed_forward_branch)
        # Reads `x` and not `streams`, which is what the unwrapped block does: its MoE half is a
        # function of the block input while its dense feed-forward is a function of the
        # post-attention state. Passing `x` here and writing into `streams` below is the same
        # graph, expressed through the gates.
        moe_out = self.feed_forward_moe_hc.read_in(
            x if x.dim() == 4 else self.feed_forward_moe_hc.expand(x)
        )
        moe_out = self._moe_branch(moe_out, loss_div_factor=loss_div_factor)
        return self.feed_forward_moe_hc.write_out(streams, moe_out)

    def _attention_branch(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _feed_forward_branch(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _moe_branch(
        self, h: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        raise NotImplementedError


class HyperConnectionMoEHybridTransformerBlock(
    _HyperConnectionMoEHybridBlockBase, MoEHybridTransformerBlock
):
    """
    :class:`~olmo_core.nn.transformer.MoEHybridTransformerBlock` with all three of its residual
    adds routed through hyper-connections. Pre-norm ordering.

    See :class:`HyperConnectionMoETransformerBlock` for the parameters.
    """

    def _attention_branch(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.branch_dropout(self.attention(self.attention_norm(h), **kwargs))

    def _feed_forward_branch(self, h: torch.Tensor) -> torch.Tensor:
        return self.branch_dropout(self.feed_forward(self.feed_forward_norm(h)))

    def _moe_branch(
        self, h: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.branch_dropout(
            self.feed_forward_moe(self.feed_forward_moe_norm(h), loss_div_factor=loss_div_factor)
        )


class HyperConnectionMoEHybridReorderedNormTransformerBlock(
    _HyperConnectionMoEHybridBlockBase, MoEHybridReorderedNormTransformerBlock
):
    """
    :class:`~olmo_core.nn.transformer.MoEHybridReorderedNormTransformerBlock` with all three of
    its residual adds routed through hyper-connections. OLMo-2 ordering.

    See :class:`HyperConnectionMoETransformerBlock` for the parameters.
    """

    def _attention_branch(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.branch_dropout(self.attention_norm(self.attention(h, **kwargs)))

    def _feed_forward_branch(self, h: torch.Tensor) -> torch.Tensor:
        return self.branch_dropout(self.feed_forward_norm(self.feed_forward(h)))

    def _moe_branch(
        self, h: torch.Tensor, *, loss_div_factor: Optional[Union[torch.Tensor, float]] = None
    ) -> torch.Tensor:
        return self.branch_dropout(
            self.feed_forward_moe_norm(self.feed_forward_moe(h, loss_div_factor=loss_div_factor))
        )
