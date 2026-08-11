"""
Hyper-connected mixture-of-experts blocks.

Matched to the standard in ``hyper_connections_test.py``: bit-exact baseline equivalence at
initialisation with the noise off, doubly stochastic mixers, shapes, gradients through every
routing tensor, save/resume round-trips, and float32 routing under bfloat16.

**THE MoE FORWARD PASS DOES NOT RUN ON A CPU, AND THAT IS WHY THIS FILE LOOKS THE WAY IT DOES.**
``olmo_core.ops.moe`` imports its kernels from ``olmo_core.kernels.moe`` and every op in it
asserts ``kernels is not None``; the import fails without a GPU build, so ``binned_gather``
raises an ``AssertionError`` on the first expert dispatch. Every MoE test already in this
repository carries ``@requires_gpu`` for exactly that reason.

So the CPU tests here substitute :class:`MoEStandIn` for the ``feed_forward_moe`` module in
**both** the wrapped block and the unwrapped block it is compared against. That is not a
weakened test of the MoE — it is a test of the code this work actually adds, which is the
residual wiring *around* the MoE. The hyper-connection reads one ``(batch, seq, d_model)``
tensor out of the streams, hands it to whatever ``feed_forward_moe`` is, and writes the result
back; nothing it does depends on what happens inside. Substituting a deterministic dense module
on both sides makes the comparison exact rather than approximate, which is stronger than a
tolerance would be.

What the substitution cannot cover is listed on
:func:`test_real_moe_forward_matches_the_unwrapped_block`, which carries ``@requires_gpu``, has
never been run, and says so.
"""

import copy
import dataclasses
from typing import Optional, Union, cast

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionConfig,
    ResidualMixerType,
    StreamCollapseConfig,
)
from olmo_core.nn.transformer import (
    HyperConnectionMoEHybridReorderedNormTransformerBlock,
    HyperConnectionMoEHybridTransformerBlock,
    HyperConnectionMoEReorderedNormTransformerBlock,
    HyperConnectionMoETransformer,
    HyperConnectionMoETransformerBlock,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.testing import requires_gpu
from olmo_core.utils import seed_all

ALL_MIXERS = list(ResidualMixerType)

DOUBLY_STOCHASTIC_MIXERS = [
    ResidualMixerType.identity,
    ResidualMixerType.sinkhorn,
    ResidualMixerType.birkhoff,
    ResidualMixerType.kronecker,
]

#: Each row is (factory, unwrapped block type, hyper-connected block type, wrapped sub-layers).
#: The last column is what ``TransformerBlockConfig.num_hyper_connections`` has to agree with,
#: asserted rather than trusted: a hybrid block wraps three sub-layers and the others wrap two,
#: and a parameter count that silently used the wrong one would be short by 24 per block.
MOE_BLOCK_PAIRS = [
    ("smallmoe", TransformerBlockType.moe, TransformerBlockType.hyper_connection_moe, 2),
    (
        "smallmoe",
        TransformerBlockType.moe_reordered_norm,
        TransformerBlockType.hyper_connection_moe_reordered_norm,
        2,
    ),
    (
        "small_hybrid_moe",
        TransformerBlockType.moe_hybrid,
        TransformerBlockType.hyper_connection_moe_hybrid,
        3,
    ),
    (
        "small_hybrid_moe",
        TransformerBlockType.moe_hybrid_reordered_norm,
        TransformerBlockType.hyper_connection_moe_hybrid_reordered_norm,
        3,
    ),
]

MOE_BLOCK_IDS = [row[2] for row in MOE_BLOCK_PAIRS]

HC_MOE_BLOCK_CLASSES = {
    TransformerBlockType.hyper_connection_moe: HyperConnectionMoETransformerBlock,
    TransformerBlockType.hyper_connection_moe_reordered_norm: (
        HyperConnectionMoEReorderedNormTransformerBlock
    ),
    TransformerBlockType.hyper_connection_moe_hybrid: HyperConnectionMoEHybridTransformerBlock,
    TransformerBlockType.hyper_connection_moe_hybrid_reordered_norm: (
        HyperConnectionMoEHybridReorderedNormTransformerBlock
    ),
}


class MoEStandIn(nn.Module):
    """
    Something with an MoE's interface that runs on a CPU.

    ``MoEBase.forward(x, loss_div_factor=...)`` takes and returns ``(batch, seq, d_model)`` and
    the block also calls ``compute_metrics`` and ``reset_metrics`` on it. Nothing else about the
    MoE is visible to a hyper-connection, so nothing else is reproduced here.

    Deterministic from its seed and nonlinear, so that a wiring mistake which happened to be
    linear-algebraically equivalent — swapping two write-outs, say — still shows up as a
    difference.

    :param d_model: The model dimensionality.
    :param seed: The seed for this module's weights.
    """

    def __init__(self, d_model: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(d_model, d_model, generator=generator) / d_model**0.5)

    def forward(
        self,
        x: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        del loss_div_factor
        return torch.tanh(x @ self.w)

    def compute_metrics(self, reset: bool = True):
        del reset
        return {}

    def reset_metrics(self):
        pass


class RouterishStandIn(MoEStandIn):
    """
    A stand-in that also does the one thing :class:`MoEStandIn` leaves out: attach an auxiliary
    loss to its input the way ``MoEBase`` does.

    That omission is exactly what the substitution was hiding. The load-balancing loss reaches
    the optimizer through ``attach_auxiliary_loss``, an autograd function that returns its
    activation unchanged and seeds the auxiliary loss's gradient in the backward pass, and a
    hyper-connection puts two einsums and an addition between that activation and the loss. A
    stand-in with no auxiliary loss in it cannot answer whether the router still gets a
    gradient, which was the largest unverified risk in this work.

    It does not need CUDA — the graph in question is einsums — so this is asserted on every CPU
    run rather than waiting for the GPU test.

    :param d_model: The model dimensionality.
    :param seed: The seed for this module's weights.
    """

    def __init__(self, d_model: int, seed: int):
        super().__init__(d_model, seed)
        generator = torch.Generator().manual_seed(seed + 1)
        # Stands for the router: a real parameter whose only path to the loss is the attached
        # auxiliary term.
        self.router_weight = nn.Parameter(torch.randn(d_model, 4, generator=generator))

    def forward(
        self,
        x: torch.Tensor,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> torch.Tensor:
        del loss_div_factor
        from olmo_core.ops import attach_auxiliary_loss

        logits = x @ self.router_weight
        aux_loss = logits.float().softmax(dim=-1).pow(2).mean()
        return torch.tanh(attach_auxiliary_loss(x, aux_loss) @ self.w)


def _swap_in_stand_ins(model: nn.Module, *, d_model: int) -> None:
    """
    Replace every block's ``feed_forward_moe`` with a CPU-runnable stand-in.

    Seeded by block index rather than by call order, so that two models given the same treatment
    get identical stand-ins and the comparison between them is exact.

    :param model: The model to patch.
    :param d_model: The model dimensionality.
    """
    for block_idx, block in model.blocks.items():  # type: ignore[attr-defined]
        block.feed_forward_moe = MoEStandIn(d_model, 1000 + int(block_idx))


def _base_config(factory: str, block_type: TransformerBlockType, *, d_model: int = 64):
    """
    An unwrapped MoE model config at a size a CPU can build quickly.

    :param factory: The ``TransformerConfig`` factory name.
    :param block_type: The block type to force.
    :param d_model: The model dimensionality.

    :returns: The config.
    """
    base = getattr(TransformerConfig, factory)(
        vocab_size=256, d_model=d_model, n_layers=2, n_heads=4
    )
    return dataclasses.replace(
        base,
        block=dataclasses.replace(cast(TransformerBlockConfig, base.block), name=block_type),
        name=TransformerType.moe,
        init_seed=7,
    )


def _hc_config(
    factory: str,
    block_type: TransformerBlockType,
    mixer: ResidualMixerType,
    *,
    d_model: int = 64,
    n_streams: int = 4,
    init_noise_std: float = 0.0,
    residual_dropout_p: float = 0.0,
):
    """
    The hyper-connected counterpart of :func:`_base_config`.

    :param factory: The ``TransformerConfig`` factory name.
    :param block_type: The hyper-connected block type.
    :param mixer: The residual mixer.
    :param d_model: The model dimensionality.
    :param n_streams: The number of residual streams.
    :param init_noise_std: The symmetry-breaking noise, off by default so that comparisons are
        against the identity-preserving initialisation.
    :param residual_dropout_p: The residual-mixer logit dropout.

    :returns: The config.
    """
    base = getattr(TransformerConfig, factory)(
        vocab_size=256, d_model=d_model, n_layers=2, n_heads=4
    )
    hc = HyperConnectionConfig(
        n_streams=n_streams,
        mixer=mixer,
        init_noise_std=init_noise_std,
        residual_dropout_p=residual_dropout_p,
    )
    return dataclasses.replace(
        base,
        block=dataclasses.replace(
            cast(TransformerBlockConfig, base.block), name=block_type, hyper_connection=hc
        ),
        name=TransformerType.hyper_connection_moe,
        stream_collapse=StreamCollapseConfig(n_streams=n_streams),
        init_seed=7,
    )


# ---------------------------------------------------------------------------------------------
# Baseline equivalence: the property everything else is measured against
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_hc_moe_model_matches_the_unwrapped_model_at_init(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """With the noise off, every mixer reproduces the unwrapped MoE model bit for bit."""
    del n_hc
    seed_all(23)
    base = _base_config(factory, base_block).build()
    wrapped = _hc_config(factory, hc_block, mixer).build()
    base.init_weights()
    wrapped.init_weights()
    base.eval()
    wrapped.eval()
    _swap_in_stand_ins(base, d_model=64)
    _swap_in_stand_ins(wrapped, d_model=64)

    input_ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        expected = base(input_ids)
        actual = wrapped(input_ids)

    # Exactly zero, not a tolerance. At `init_noise_std = 0` every stream carries the same
    # vector, `h_pre` sums to 1, `h_post` is all ones and every constrained mixer is the uniform
    # doubly stochastic matrix, so the arithmetic is the same arithmetic in the same order.
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_every_stream_carries_the_unwrapped_block_output(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    Not just the collapsed readout: at initialisation each of the ``n`` streams individually
    reproduces the unwrapped block. A wrapping that got the write-out gate wrong could still
    average back to the right answer.
    """
    del n_hc
    seed_all(23)
    base = _base_config(factory, base_block).build()
    wrapped = _hc_config(factory, hc_block, mixer).build()
    base.init_weights()
    wrapped.init_weights()
    base.eval()
    wrapped.eval()
    _swap_in_stand_ins(base, d_model=64)
    _swap_in_stand_ins(wrapped, d_model=64)

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        expected = base.blocks["0"](x)
        streams = wrapped.blocks["0"](x)

    assert streams.shape == (2, 8, 4, 64)
    for stream in range(4):
        torch.testing.assert_close(streams[:, :, stream], expected, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------------------------
# Structure, counts and configuration
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_routing_param_count_is_per_wrapped_sublayer(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    The config's routing-parameter count, the block's, and the model's all agree, and the total
    parameter count is the unwrapped model's plus exactly that.
    """
    config = _hc_config(factory, hc_block, mixer)
    model = config.build()
    base_model = _base_config(factory, base_block).build()

    per_sublayer = HyperConnectionConfig(n_streams=4, mixer=mixer).num_params()
    expected = 2 * n_hc * per_sublayer  # two blocks

    assert cast(TransformerBlockConfig, config.block).num_hyper_connections == n_hc
    assert config.num_routing_params == expected
    assert model.num_routing_params == expected
    assert sum(block.num_routing_params for block in model.blocks.values()) == expected

    measured = sum(p.numel() for p in model.parameters())
    assert measured == config.num_params
    assert measured - sum(p.numel() for p in base_model.parameters()) == expected


@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_block_class_and_model_class_are_the_moe_ones(
    factory: str, base_block: TransformerBlockType, hc_block: TransformerBlockType, n_hc: int
):
    """
    The model has to be an :class:`HyperConnectionMoETransformer` and not a plain
    :class:`HyperConnectionTransformer`, because only the former routes the router's
    load-balancing loss and z-loss out through ``compute_auxiliary_metrics``. A hyper-connected
    MoE model built on the dense class would train with those losses silently unreported.
    """
    del base_block, n_hc
    model = _hc_config(factory, hc_block, ResidualMixerType.sinkhorn).build()

    assert isinstance(model, HyperConnectionMoETransformer)
    assert model.is_moe
    for block in model.blocks.values():
        assert isinstance(block, HC_MOE_BLOCK_CLASSES[hc_block])
        assert block.is_moe
    # The load-balancing loss and the router z-loss are the two the treatment arms depend on,
    # and this is the path they reach the trainer by. They are reported per block and pooled;
    # the values are zero because nothing has run a forward pass, and the keys existing is the
    # whole assertion.
    metrics = model.compute_auxiliary_metrics(reset=False)
    assert "load balancing loss" in metrics
    assert "router Z loss" in metrics
    for block_idx in model.blocks:
        assert f"block {int(block_idx):02d}/load balancing loss" in metrics


def test_dense_hc_model_refuses_moe_blocks():
    """
    The mistake this catches costs an experiment rather than a run: a `hyper_connection` model
    holding MoE blocks builds, trains, and drops the load-balancing loss on the floor.
    """
    config = _hc_config(
        "smallmoe",
        TransformerBlockType.hyper_connection_moe_reordered_norm,
        ResidualMixerType.sinkhorn,
    )
    with pytest.raises(OLMoConfigurationError, match="does not route MoE auxiliary losses"):
        dataclasses.replace(config, name=TransformerType.hyper_connection)


def test_moe_hc_model_refuses_an_all_dense_block_set():
    base = TransformerConfig.llama_like(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=256,
        block_name=TransformerBlockType.reordered_norm,
    )
    block = dataclasses.replace(
        cast(TransformerBlockConfig, base.block),
        name=TransformerBlockType.hyper_connection,
        hyper_connection=HyperConnectionConfig(n_streams=4),
    )
    with pytest.raises(OLMoConfigurationError, match="needs at least one hyper-connected MoE"):
        dataclasses.replace(
            base,
            block=block,
            name=TransformerType.hyper_connection_moe,
            stream_collapse=StreamCollapseConfig(n_streams=4),
        )


@pytest.mark.parametrize("hc_block", list(HC_MOE_BLOCK_CLASSES))
def test_residual_alphas_are_refused(hc_block: TransformerBlockType):
    """A fixed scalar in front of the write-out gate would only be absorbed into it."""
    factory = "small_hybrid_moe" if "hybrid" in hc_block else "smallmoe"
    config = _hc_config(factory, hc_block, ResidualMixerType.sinkhorn)
    block = dataclasses.replace(
        cast(TransformerBlockConfig, config.block), attention_residual_alpha=0.5
    )
    with pytest.raises(OLMoConfigurationError, match="residual alphas are not supported"):
        block.build(d_model=64, block_idx=0, n_layers=2)


# ---------------------------------------------------------------------------------------------
# Gradients, precision, and the mixer's own guarantees
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_backward_reaches_every_routing_tensor(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    Two things this is careful about, and both of them are ways to write a gradient check that
    reports a failure that is not there.

    A squared loss and not ``out.sum()``. A doubly stochastic mixer preserves the sum over
    streams exactly, so an objective that only sees ``out.sum(dim=streams)`` has zero gradient
    with respect to the `birkhoff` and `kronecker` parameters.

    And a four-dimensional input whose streams differ, not a three-dimensional one. A 3-D input
    is expanded into ``n`` **identical** streams, and with identical streams every constrained
    mixer's gradient is exactly zero — see
    :func:`test_constrained_mixer_gradient_vanishes_when_streams_are_identical`, which asserts
    that separately because it is a real property of the method rather than a quirk of this
    test.
    """
    del base_block, n_hc
    seed_all(23)
    model = _hc_config(factory, hc_block, mixer, init_noise_std=1e-2).build()
    model.init_weights()
    _swap_in_stand_ins(model, d_model=64)
    model.train()

    x = torch.randn(2, 8, 4, 64)
    out = model.blocks["0"](x)
    assert out.shape == (2, 8, 4, 64)
    (out.float() ** 2).mean().backward()

    for name, module in model.blocks["0"].named_modules():
        if not isinstance(module, HyperConnection):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            assert param.grad is not None, f"no gradient on {name}.{param_name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient on {name}.{param_name}"
            if param_name != "h_res_logits" or mixer == ResidualMixerType.unconstrained:
                assert param.grad.abs().max() > 0, f"zero gradient on {name}.{param_name}"

    # `h_res_logits` on a CONSTRAINED mixer is deliberately not asserted nonzero above, and the
    # reason is the finding rather than a weakened test: with the uniform doubly stochastic
    # initialisation the streams are nearly a common vector, and every constraint map's Jacobian
    # annihilates that component, so the gradient is 1e-8 or smaller and `birkhoff`'s underflows
    # to exactly zero in float32. See
    # `test_constrained_mixer_gradient_is_orders_below_the_unconstrained_one`, which measures it.


@pytest.mark.parametrize("mixer", [m for m in ALL_MIXERS if m != ResidualMixerType.identity])
def test_constrained_mixer_gradient_vanishes_when_streams_are_identical(
    mixer: ResidualMixerType,
):
    """
    **The mechanism this whole line of work is about, measured rather than argued.**

    Every constraint map — Sinkhorn's alternating normalisation, Birkhoff's softmax over
    permutations, Kronecker's per-factor softmax — produces a matrix whose rows sum to 1. When
    the ``n`` streams carry the same vector ``z``, ``H_res Z`` is ``z`` in every stream for any
    such matrix, and the derivative of that output along the constraint manifold is exactly
    zero: the Jacobian of the map annihilates the only direction identical streams can produce.

    So a hyper-connection whose streams have collapsed has an ``H_res`` that receives no
    gradient at all, and therefore never leaves its initialisation, and therefore looks useless.
    That is a mechanistic account of the ``1e-9`` ``H_res`` gradient norm a public mHC
    reproduction measured against ``1.84`` on the branch weights, and it is the reason the
    treatment in ``docs/hc-ablation/EXPERIMENT-DESIGN.md`` is a *balancing* loss rather than
    anything else: keeping the streams apart is what keeps the gradient alive.

    ``unconstrained`` is asserted in the other direction in the same breath, because it is the
    control that makes the claim mean something: the original Hyper-Connections formulation has
    no constraint map in front of its matrix, so identical streams give it a **large** gradient.
    The two rows together say the vanishing is caused by the constraint and not by the streams.
    """
    seed_all(11)
    n_streams, d_model = 4, 16
    hc = HyperConnectionConfig(
        n_streams=n_streams, mixer=mixer, init_noise_std=1e-2, residual_dropout_p=0.0
    ).build()
    assert hc.h_res_logits is not None

    identical = torch.randn(2, 5, 1, d_model).expand(2, 5, n_streams, d_model).contiguous()
    differing = torch.randn(2, 5, n_streams, d_model)

    grads = {}
    for label, streams in (("identical", identical), ("differing", differing)):
        hc.zero_grad(set_to_none=True)
        hc(streams, lambda h: h * 2.0).square().mean().backward()
        assert hc.h_res_logits.grad is not None
        grads[label] = hc.h_res_logits.grad.norm().item()

    if mixer == ResidualMixerType.unconstrained:
        assert grads["identical"] > 1e-2, grads
    else:
        # Exactly zero for `birkhoff` and `kronecker`; float32 round-off for `sinkhorn`, whose
        # twenty iterations leave a residual around 1e-11. Both are zero for any purpose an
        # optimizer has.
        assert grads["identical"] < 1e-8, grads
        assert grads["differing"] > 10 * grads["identical"], grads


def test_constrained_mixer_gradient_is_orders_below_the_unconstrained_one():
    """
    **The reproduction, on a CPU, in a second, of the number this whole line of work turns on.**

    A public mHC reproduction measured the ``H_res`` gradient norm at about ``1e-9`` against
    ``1.84`` on the branch weights and concluded the matrix never left its initialisation. That
    is reproducible here without training anything, and the cause is visible in the same
    measurement: identical inputs, identical loss, identical block, one mixer with a constraint
    map in front of its matrix and one without.

    The mechanism is
    :func:`test_constrained_mixer_gradient_vanishes_when_streams_are_identical`. Every
    constrained mixer starts at the uniform doubly stochastic matrix, which is an average over
    streams, which destroys the dispersion between them; and the gradient that survives the
    constraint map is the part proportional to that dispersion. `unconstrained` starts at the
    same matrix and has no map, so its gradient does not vanish.

    This is why the treatment in ``docs/hc-ablation/EXPERIMENT-DESIGN.md`` is a balancing loss on
    stream usage and why the primary endpoint is the mixer's displacement rather than validation
    loss: the quantity being intervened on spans seven orders of magnitude, and a loss
    difference does not.
    """
    seed_all(23)
    norms = {}
    for mixer in ALL_MIXERS:
        if mixer == ResidualMixerType.identity:
            continue
        model = _hc_config(
            "smallmoe",
            TransformerBlockType.hyper_connection_moe_reordered_norm,
            mixer,
            init_noise_std=1e-2,
        ).build()
        model.init_weights()
        _swap_in_stand_ins(model, d_model=64)
        model.train()
        torch.manual_seed(5)
        out = model.blocks["0"](torch.randn(2, 8, 4, 64))
        (out.float() ** 2).mean().backward()
        hc = model.blocks["0"].attention_hc
        assert hc.h_res_logits is not None and hc.h_res_logits.grad is not None
        norms[mixer] = hc.h_res_logits.grad.norm().item()

    unconstrained = norms[ResidualMixerType.unconstrained]
    assert unconstrained > 1e-2, norms
    for mixer in DOUBLY_STOCHASTIC_MIXERS:
        if mixer == ResidualMixerType.identity:
            continue
        assert norms[mixer] < unconstrained / 1e3, norms


@pytest.mark.parametrize("mixer", DOUBLY_STOCHASTIC_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_block_mixers_stay_doubly_stochastic_off_initialisation(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    The doubly stochastic guarantee has to hold everywhere on the manifold, not only at the
    identity-preserving starting point.

    Two logit scales, and the split is the caveat ``docs/hc-ablation/README.md`` already records
    rather than a weakened assertion. `birkhoff` and `kronecker` are exactly doubly stochastic
    for any logits at all. `sinkhorn` is not: its twenty iterations — the count the mHC paper
    specifies — converge only while the logits stay small, the **column** sums stay exactly 1
    because the column normalisation is applied last, and the **row** sums drift. A row sum
    below 1 shrinks that stream's residual, so a run whose logits grow is no longer doing what
    the method says it does. See
    :func:`test_sinkhorn_row_sums_drift_earlier_than_the_readme_says` for where that starts.
    """
    del base_block, n_hc
    seed_all(23)
    model = _hc_config(factory, hc_block, mixer, init_noise_std=0.5).build()
    model.init_weights()

    for block in model.blocks.values():
        for hc in block.hyper_connections:
            for std, check_row_sums in ((0.5, True), (4.0, mixer != ResidualMixerType.sinkhorn)):
                with torch.no_grad():
                    if hc.h_res_logits is not None:
                        hc.h_res_logits.normal_(mean=0.0, std=std)
                h_res = hc.residual_mixer()
                assert (h_res >= 0).all(), f"{mixer} went negative at std={std}"
                torch.testing.assert_close(
                    h_res.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0, msg=f"columns, std={std}"
                )
                if check_row_sums:
                    torch.testing.assert_close(
                        h_res.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0, msg=f"rows, std={std}"
                    )


def test_sinkhorn_row_sums_drift_earlier_than_the_readme_says():
    """
    Where the twenty-iteration budget stops delivering a doubly stochastic matrix, measured.

    ``docs/hc-ablation/README.md`` puts the onset at "roughly ``|logit| ~ 10``" and describes
    the damage as "a factor of two or more at ``|logit| ~ 100``". Measured over 200 draws per
    scale, the row-sum error is already 1.3e-2 at a maximum absolute logit of about 6.5 and
    4e-2 at about 12.7 — so the onset is about half the magnitude the document quotes, and the
    error at practical magnitudes is percent-level rather than only appearing at extremes.

    Turning the early-stop tolerance off changes it barely at all (3.4e-2 against 4.2e-2 at the
    same scale), so this is the iteration count and not the stopping rule.

    Asserted rather than written down, because a document that is 2x optimistic about when a
    guarantee stops holding is worse than no document: it tells a reader watching the largest
    residual logit that they have twice the headroom they have. The monitor in
    ``hyper_connection_monitor`` logs the row-sum error itself for this reason.
    """
    from olmo_core.nn.hyper_connections import sinkhorn_log_space

    torch.manual_seed(3)
    worst = {}
    for std in (0.5, 2.0, 3.0):
        row_error = 0.0
        largest_logit = 0.0
        for _ in range(200):
            logits = torch.randn(4, 4) * std
            mixer = sinkhorn_log_space(logits)
            row_error = max(row_error, (mixer.sum(-1) - 1).abs().max().item())
            largest_logit = max(largest_logit, logits.abs().max().item())
            # The column sums survive at every scale, because they are normalised last.
            assert (mixer.sum(-2) - 1).abs().max().item() < 1e-5
        worst[std] = (largest_logit, row_error)

    assert worst[0.5][1] < 1e-5, worst
    assert worst[2.0][1] > 1e-3, worst
    assert worst[2.0][0] < 10.0, worst
    assert worst[3.0][1] > worst[2.0][1], worst


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_routing_math_stays_float32_in_bfloat16(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    In bfloat16 the Sinkhorn fixed point is reached to about two decimal digits and the row and
    column sums drift far enough off 1 that the doubly stochastic property, which is the whole
    claim of mHC, stops holding. So the routing stays float32 whatever the activations are.
    """
    del base_block, n_hc
    seed_all(23)
    model = _hc_config(factory, hc_block, mixer).build()
    model.init_weights()
    model.to(torch.bfloat16)

    for block in model.blocks.values():
        for hc in block.hyper_connections:
            assert hc.read_in_gate().dtype == torch.float32
            assert hc.write_out_gate().dtype == torch.float32
            assert hc.residual_mixer().dtype == torch.float32


@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_state_dict_round_trip(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """Save and resume reproduces the forward pass exactly and carries every routing tensor."""
    del base_block
    seed_all(23)
    config = _hc_config(factory, hc_block, mixer, init_noise_std=1e-2)
    original = config.build()
    original.init_weights()
    _swap_in_stand_ins(original, d_model=64)
    original.eval()

    state = copy.deepcopy(original.state_dict())
    restored = config.build()
    restored.init_weights()
    _swap_in_stand_ins(restored, d_model=64)
    restored.load_state_dict(state)
    restored.eval()

    # Every routing tensor is in the checkpoint, not merely some of them: a mixer whose logits
    # were registered as a buffer rather than a parameter would resume from its initialisation
    # and nothing would say so.
    routing_keys = [key for key in state if "_hc." in key]
    expected_per_hc = 2 if mixer == ResidualMixerType.identity else 3
    assert len(routing_keys) == 2 * n_hc * expected_per_hc, routing_keys

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.blocks["0"](x), original.blocks["0"](x), atol=0.0, rtol=0.0
        )


# ---------------------------------------------------------------------------------------------
# The parallelism refusals
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hc_block", list(HC_MOE_BLOCK_CLASSES))
@pytest.mark.parametrize(
    "method,match",
    [
        ("apply_ep", "expert parallelism"),
        ("apply_tp", "tensor parallelism"),
        ("apply_cp", "context parallelism"),
    ],
)
def test_parallelism_that_has_not_been_validated_raises(
    hc_block: TransformerBlockType, method: str, match: str
):
    """
    Raising beats silently applying a plan written for a three-dimensional hidden state. The
    failure this prevents has no error in it: the model trains, reports a loss curve, and is
    computing something other than what the config says.
    """
    factory = "small_hybrid_moe" if "hybrid" in hc_block else "smallmoe"
    model = _hc_config(factory, hc_block, ResidualMixerType.sinkhorn).build()
    block = model.blocks["0"]
    with pytest.raises(NotImplementedError, match=match):
        getattr(block, method)(None)


@pytest.mark.parametrize(
    "hc_block",
    [
        TransformerBlockType.hyper_connection_moe_hybrid,
        TransformerBlockType.hyper_connection_moe_hybrid_reordered_norm,
    ],
)
def test_hybrid_combined_forward_is_refused(hc_block: TransformerBlockType):
    """
    The combined forward interleaves the three residual adds with the expert all-to-all, and a
    hyper-connection changes what each of those adds means. It is unreachable while
    ``apply_ep`` raises, and it refuses on its own so the two facts cannot drift apart.
    """
    model = _hc_config("small_hybrid_moe", hc_block, ResidualMixerType.sinkhorn).build()
    block = model.blocks["0"]
    assert block.use_combined_forward is False
    with pytest.raises(NotImplementedError, match="combined forward"):
        block.use_combined_forward = True
    with pytest.raises(NotImplementedError, match="combined forward"):
        block.combined_forward(torch.randn(1, 2, 4, 64))


# ---------------------------------------------------------------------------------------------
# Symmetry breaking
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_streams_diverge_only_with_symmetry_breaking(
    factory: str, base_block: TransformerBlockType, hc_block: TransformerBlockType, n_hc: int
):
    """
    Both directions. Identical streams leave an ``S_n`` permutation symmetry that gradient
    descent preserves exactly, so without the noise the streams stay copies of each other for
    the whole run and ``n > 1`` buys nothing but memory. A change that made them diverge on
    their own would break the baseline equivalence above, which is why this asserts the null
    direction too.
    """
    del base_block, n_hc
    for noise, should_diverge in ((0.0, False), (1e-2, True)):
        seed_all(23)
        model = _hc_config(
            factory, hc_block, ResidualMixerType.sinkhorn, init_noise_std=noise
        ).build()
        model.init_weights()
        _swap_in_stand_ins(model, d_model=64)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randn(2, 8, 64)
        for _ in range(3):
            optimizer.zero_grad()
            out = model.blocks["0"](x)
            (out.float() ** 2).mean().backward()
            optimizer.step()

        with torch.no_grad():
            out = model.blocks["0"](x)
        spread = (out - out.mean(dim=-2, keepdim=True)).abs().max().item()
        if should_diverge:
            assert spread > 1e-5, f"streams stayed identical at noise={noise}"
        else:
            assert spread < 1e-6, f"streams diverged with no symmetry breaking: {spread}"


# ---------------------------------------------------------------------------------------------
# What needs a GPU
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mixer", [ResidualMixerType.sinkhorn, ResidualMixerType.identity])
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_an_auxiliary_loss_reaches_its_own_parameter_through_the_write_out_gate(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    **The largest unverified risk in this work, retired on a CPU.**

    The MoE router's load-balancing loss reaches the optimizer through
    ``attach_auxiliary_loss``, which returns its activation unchanged and seeds the auxiliary
    loss's gradient in the backward pass. A hyper-connection puts two einsums and an addition
    between that activation and the loss, so the question is whether the router still receives
    a gradient — and if it does not, every MoE arm trains with an unbalanced router while
    looking perfectly healthy.

    This was left to a ``@requires_gpu`` test that has never run, on the reasoning that the MoE
    needs CUDA kernels. The kernels are needed for the expert *dispatch*; the graph between the
    auxiliary loss and the router is einsums, and it runs anywhere. So
    :class:`RouterishStandIn` does what ``MoEBase`` does — computes a loss from a real
    parameter and attaches it — the primary objective is multiplied by zero so only the
    attached loss can produce a gradient, and the parameter is checked.

    What this still does not cover is the real dispatch, the capacity drop path and the
    kernels' dtype behaviour. That is what the GPU test below is for.
    """
    del base_block, n_hc
    seed_all(23)
    model = _hc_config(factory, hc_block, mixer, init_noise_std=1e-2).build()
    model.init_weights()
    for block_idx, block in model.blocks.items():
        block.feed_forward_moe = RouterishStandIn(64, 2000 + int(block_idx))
    model.train()

    x = torch.randint(0, 256, (2, 16))
    out = model(x, labels=torch.roll(x, -1, dims=1))
    loss = out[0] if isinstance(out, tuple) else out
    # Zeroed, so the only surviving path to `router_weight` is the attached auxiliary loss.
    (loss.mean() * 0.0).backward()

    for name, block in model.blocks.items():
        grad = block.feed_forward_moe.router_weight.grad
        assert grad is not None, f"block {name}: no gradient reached the router at all"
        assert torch.isfinite(grad).all(), f"block {name}: non-finite router gradient"
        assert grad.abs().max() > 0, f"block {name}: the auxiliary loss did not reach the router"


@requires_gpu
@pytest.mark.parametrize("mixer", ALL_MIXERS)
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_real_moe_forward_matches_the_unwrapped_block(
    mixer: ResidualMixerType,
    factory: str,
    base_block: TransformerBlockType,
    hc_block: TransformerBlockType,
    n_hc: int,
):
    """
    The same equivalence as
    :func:`test_hc_moe_model_matches_the_unwrapped_model_at_init`, through the real expert
    dispatch instead of a stand-in.

    **THIS HAS NEVER BEEN RUN.** No GPU was available to the work that wrote it, so what it
    establishes today is nothing at all. It is the first thing to run on a machine that has one,
    and it is the reason the CPU tests above are careful to say what they do and do not cover:
    everything about the residual wiring is covered there, and everything about the interaction
    between that wiring and the real expert dispatch — dtype promotion through ``binned_gather``,
    the capacity-factor drop path, and whether the router's auxiliary loss survives the
    write-out gate's graph — is covered only here.
    """
    del n_hc
    seed_all(23)
    device = torch.device("cuda")
    base = _base_config(factory, base_block).build(init_device="cuda")
    wrapped = _hc_config(factory, hc_block, mixer).build(init_device="cuda")
    base.init_weights(device=device)
    wrapped.init_weights(device=device)
    base.eval()
    wrapped.eval()
    # Copy the shared weights across so the two models differ only in the residual wiring.
    wrapped.load_state_dict(base.state_dict(), strict=False)

    input_ids = torch.randint(0, 256, (2, 8), device=device)
    with torch.no_grad():
        expected = base(input_ids)
        actual = wrapped(input_ids)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@requires_gpu
@pytest.mark.parametrize("factory,base_block,hc_block,n_hc", MOE_BLOCK_PAIRS, ids=MOE_BLOCK_IDS)
def test_router_auxiliary_loss_survives_the_write_out_gate(
    factory: str, base_block: TransformerBlockType, hc_block: TransformerBlockType, n_hc: int
):
    """
    The load-balancing loss reaches the optimizer through ``attach_auxiliary_loss``, which is an
    autograd function that returns its activation unchanged and seeds the aux loss's gradient in
    the backward pass. A hyper-connection puts two einsums and an addition between that
    activation and the loss, so the question is whether the router still receives a gradient.

    **THIS HAS NEVER BEEN RUN**, for the reason on the test above. It is the single most
    important GPU check in this file: if it fails, the MoE arms train with an unbalanced router
    and the experiment measures the wrong thing while looking healthy.

    An earlier version of it could not fail. It ran an ordinary backward on the cross-entropy
    and asserted the router had a gradient — which it does regardless, because the primary loss
    flows through the expert weights the router produces. The primary term is multiplied by zero
    now, so the only remaining path to a router gradient is the attached auxiliary loss.
    """
    del base_block, n_hc
    seed_all(23)
    device = torch.device("cuda")
    model = _hc_config(factory, hc_block, ResidualMixerType.sinkhorn, init_noise_std=1e-2).build(
        init_device="cuda"
    )
    model.init_weights(device=device)
    model.train()

    input_ids = torch.randint(0, 256, (2, 8), device=device)
    labels = torch.randint(0, 256, (2, 8), device=device)
    out = model(input_ids, labels=labels)
    loss = out[0] if isinstance(out, tuple) else out
    # **MULTIPLIED BY ZERO, WHICH IS THE ENTIRE POINT OF THIS TEST.** The router receives
    # gradients from the primary objective anyway, through the expert weights it produces, so a
    # backward on the cross-entropy gives it a gradient whether or not the auxiliary loss is
    # attached at all -- and this test passed with `attach_auxiliary_loss` deleted. Zeroing the
    # primary term leaves the attached auxiliary loss as the only thing that can produce one.
    (loss.sum() * 0.0).backward()

    for block in model.blocks.values():
        router_grads = [
            p.grad for p in block.feed_forward_moe.router.parameters() if p.grad is not None
        ]
        assert router_grads, "the router received no gradient at all"
        assert any(g.abs().max() > 0 for g in router_grads)
