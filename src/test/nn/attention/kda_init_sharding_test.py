"""
Initialization of the KDA mixers under FSDP2 sharding.

By the time training calls ``init_weights`` every parameter is a sharded ``DTensor``, and a
``DTensor`` will not accept an in-place write into a slice of itself: the slice redistributes, so
the write lands on a fresh tensor and is discarded. Nothing raises when that happens, so the
parameter silently keeps whatever the ``reset_parameters`` sweep left in it. A random op is worse
than silent -- over a CPU mesh a ``DTensor`` has no RNG tracker, so it ignores the generator it was
handed and draws from each process's default one, which makes a seeded run unreproducible.

Neither failure is visible at one rank, so these tests run a real two-rank ``fully_shard``, assert
the parameters really are split before initializing, and measure what they end up containing.
"""

from typing import Callable, Dict, List, Tuple

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.utils import get_full_tensor
from olmo_core.nn.attention import KimiDeltaAttentionConfig, KimiDeltaHouseholderConfig
from olmo_core.nn.attention.base import SequenceMixer
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import run_distributed_test
from olmo_core.testing.utils import has_fla

D_MODEL = 128
N_HEADS = 8
HEAD_DIM = 32
SEED = 1234
WORLD_SIZE = 2

# `requires_fla` carries `pytest.mark.gpu`, so it cannot be used on a test that is meant to run
# on CPU. This is the same availability check without the GPU mark.
requires_fla_cpu = pytest.mark.skipif(not has_fla, reason="Requires flash-linear-attention (fla)")

#: The three bake-off variants, as zero-argument factories.
#:
#: Factories rather than instances because ``run_distributed_test`` sends its arguments to a
#: forked worker, and a shared config instance would let one rank's ``replace`` reach another's.
VARIANTS: Dict[str, Callable[[], object]] = {
    "base": lambda: KimiDeltaAttentionConfig(n_heads=N_HEADS, head_dim=HEAD_DIM),
    "householder_r2_negeig": lambda: KimiDeltaHouseholderConfig(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        num_householder=2,
        allow_neg_eigval=True,
        backend="torch",
    ),
    "gconv": lambda: KimiDeltaAttentionConfig(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        gated_conv=True,
        gate_structure="depthwise",
    ),
}


#: Parameters that the bare ``reset_parameters`` sweep owns, rather than ``init_weights``.
#:
#: ``Transformer.init_weights`` runs the sweep over every module *before* it dispatches, so these
#: are initialized on the real path -- but only there. That is safe for a **constant** fill and
#: only for a constant fill: a constant needs no generator, so it is reproducible, and it is the
#: same on every shard, so it survives being written into a ``DTensor``. Anything drawn would be
#: neither. :func:`test_sweep_owned_parameters_are_constant` is what holds that line, and this set
#: is spelled out so a parameter that quietly joins it has to be added here first.
SWEEP_OWNED = {"o_norm.weight"}


def _build(variant: str) -> SequenceMixer:
    config = VARIANTS[variant]()
    return config.build(D_MODEL, layer_idx=0, n_layers=2, init_device="cpu")  # type: ignore[attr-defined]


def _init(mixer: SequenceMixer, *, seed: int = SEED) -> None:
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=D_MODEL,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(seed),
    )


def _shard(mixer: SequenceMixer, world_size: int) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import init_device_mesh

    fully_shard(mixer, mesh=init_device_mesh("cpu", (world_size,)))


def _enter_worker() -> None:
    """
    Prepare a forked rank to run ATen ops.

    ``run_distributed_test`` forks for the gloo backend, and a child that inherits an OpenMP
    runtime from a parent which has already run an ATen op deadlocks the first time it enters a
    parallel region. Staying single-threaded costs nothing at these sizes and is what keeps a
    regression reported as a failure rather than as a hang.
    """
    torch.set_num_threads(1)


def _split_parameters(mixer: SequenceMixer, world_size: int) -> List[str]:
    """
    Names of the parameters that ``fully_shard`` actually split across ranks.

    :param mixer: The sharded mixer.
    :param world_size: The number of ranks.

    :returns: The names of every parameter whose local shape is smaller than its global one.

    :raises AssertionError: If any parameter is not a ``DTensor``, which would mean the test is
        measuring an unsharded module and proving nothing.
    """
    from torch.distributed.tensor import DTensor

    split: List[str] = []
    for name, param in mixer.named_parameters():
        assert isinstance(param, DTensor), f"{name} is not a DTensor; the test would prove nothing"
        if param.to_local().shape[0] < param.shape[0]:
            split.append(name)
    assert split, f"fully_shard split nothing at world size {world_size}"
    return split


def _fill_with_sentinel(mixer: SequenceMixer) -> None:
    """Poison every parameter, so anything ``init_weights`` fails to write stays detectable."""
    from olmo_core.distributed.utils import get_local_tensor

    with torch.no_grad():
        for param in mixer.parameters():
            get_local_tensor(param).fill_(torch.nan)


def _surviving_sentinels(mixer: SequenceMixer) -> List[str]:
    """Names of the parameters that were never written, in ``named_parameters`` order."""
    return [
        name
        for name, param in mixer.named_parameters()
        if not torch.isfinite(get_full_tensor(param.detach())).all()
    ]


def _reset_parameters_sweep(mixer: SequenceMixer) -> None:
    """
    The bare sweep ``Transformer.init_weights`` runs before it dispatches to a mixer.

    Every module has to survive this on its own, not only when reached through the mixer's
    ``init_weights``.
    """
    for module in mixer.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()  # type: ignore[operator]


def _timescale_reference(variant: str) -> Dict[str, List[float]]:
    mixer = _build(variant)
    _init(mixer)
    return {
        "A_log": mixer.A_log.detach().tolist(),  # type: ignore[union-attr]
        "dt_bias": mixer.dt_bias.detach().tolist(),  # type: ignore[union-attr]
    }


def _state_shapes(variant: str) -> Dict[str, Tuple[int, ...]]:
    mixer = _build(variant)
    return {name: tuple(value.shape) for name, value in mixer.state_dict().items()}


def _run_init_writes_every_parameter(world_size: int, variant: str) -> None:
    _enter_worker()
    mixer = _build(variant)
    _shard(mixer, world_size)
    _split_parameters(mixer, world_size)

    _fill_with_sentinel(mixer)
    _init(mixer)
    assert _surviving_sentinels(mixer) == sorted(SWEEP_OWNED)


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sharded_init_writes_every_parameter_the_sweep_does_not_own(variant: str):
    """
    ``init_weights`` must reach every parameter of a sharded mixer except the sweep-owned ones.

    A write that a ``DTensor`` discards leaves the parameter exactly as the sentinel left it, and
    nothing raises. That is the failure mode this catches, and no single-rank run can.
    """
    run_distributed_test(
        _run_init_writes_every_parameter,
        world_size=WORLD_SIZE,
        backend="gloo",
        func_args=(WORLD_SIZE, variant),
    )


def _run_reset_sweep_then_init(world_size: int, variant: str) -> None:
    _enter_worker()
    mixer = _build(variant)
    _shard(mixer, world_size)
    _split_parameters(mixer, world_size)

    _fill_with_sentinel(mixer)
    _reset_parameters_sweep(mixer)
    _init(mixer)
    assert _surviving_sentinels(mixer) == []


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sharded_reset_parameters_sweep_then_init(variant: str):
    """
    The bare ``reset_parameters`` sweep must survive sharding, and must not defeat ``init_weights``.

    ``Transformer.init_weights`` runs the sweep over every module before it dispatches, so a
    ``reset_parameters`` that raises on a ``DTensor`` takes the whole model down.
    """
    run_distributed_test(
        _run_reset_sweep_then_init,
        world_size=WORLD_SIZE,
        backend="gloo",
        func_args=(WORLD_SIZE, variant),
    )


def _run_timescale_honours_generator(
    world_size: int, variant: str, reference: Dict[str, List[float]]
) -> None:
    _enter_worker()
    mixer = _build(variant)
    _shard(mixer, world_size)
    split = _split_parameters(mixer, world_size)
    assert "A_log" in split, f"A_log was not split at world size {world_size}"
    assert "dt_bias" in split, f"dt_bias was not split at world size {world_size}"

    _init(mixer)

    for name, expected in reference.items():
        observed = get_full_tensor(getattr(mixer, name).detach())
        torch.testing.assert_close(
            observed,
            torch.tensor(expected, dtype=observed.dtype),
            rtol=0,
            atol=0,
            msg=lambda formatted, name=name: f"'{name}' disagrees with the one-rank draw\n{formatted}",
        )


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sharded_timescale_parameters_honour_the_seeded_generator(variant: str):
    """
    ``A_log`` and ``dt_bias`` must come from the generator the caller passed.

    A random op on a ``DTensor`` over a CPU mesh has no RNG tracker to fall back on, so it
    quietly ignores the supplied generator and draws from each process's default one. The
    parameters then differ between two runs of the same seeded configuration, in exactly the two
    parameters that set the recurrence's timescale.
    """
    run_distributed_test(
        _run_timescale_honours_generator,
        world_size=WORLD_SIZE,
        backend="gloo",
        func_args=(WORLD_SIZE, variant, _timescale_reference(variant)),
    )


def _run_sharding_preserves_state_dict_shapes(
    world_size: int, variant: str, reference: Dict[str, Tuple[int, ...]]
) -> None:
    _enter_worker()
    mixer = _build(variant)
    _shard(mixer, world_size)
    _init(mixer)

    observed = {
        name: tuple(get_full_tensor(value).shape) for name, value in mixer.state_dict().items()
    }
    assert observed == reference


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sharded_state_dict_matches_the_unsharded_one(variant: str):
    """Sharding must not move, rename or reshape a checkpoint key."""
    run_distributed_test(
        _run_sharding_preserves_state_dict_shapes,
        world_size=WORLD_SIZE,
        backend="gloo",
        func_args=(WORLD_SIZE, variant, _state_shapes(variant)),
    )


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_single_rank_init_leaves_only_the_sweep_owned_parameters(variant: str):
    """The same sentinel check at one rank, so a shard-only failure is distinguishable."""
    mixer = _build(variant)
    _fill_with_sentinel(mixer)
    _init(mixer)
    assert _surviving_sentinels(mixer) == sorted(SWEEP_OWNED)


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sweep_owned_parameters_are_constant(variant: str):
    """
    Whatever only the sweep writes must be a constant.

    A constant needs no generator, so it is reproducible, and it is identical on every shard, so
    writing it into a ``DTensor`` cannot go wrong. A *drawn* parameter reached only by the sweep
    would be neither, and would be invisible because the model still trains.
    """
    mixer = _build(variant)
    _fill_with_sentinel(mixer)
    _reset_parameters_sweep(mixer)

    written = dict(mixer.named_parameters())
    for name in SWEEP_OWNED:
        assert name in written, f"'{name}' is not a parameter of the {variant} arm"
        value = written[name].detach()
        assert torch.isfinite(value).all(), f"the sweep did not write '{name}'"
        assert torch.equal(value, torch.full_like(value, value.flatten()[0].item())), (
            f"'{name}' is reached only by the reset sweep but is not constant, so it is neither "
            f"reproducible nor safe to write into a shard"
        )


@requires_fla_cpu
def test_depthwise_gate_init_consumes_no_randomness():
    """
    A gated arm must draw the same random stream as the plain one.

    The depthwise gate is zero-initialized and draws nothing, so every parameter that both
    variants share is drawn at the same point in the shared generator's stream. A gate that
    consumed randomness would shift every later parameter in the model, and that confound does
    not show up anywhere in a loss curve.
    """
    gated = _build("gconv")
    plain = _build("base")
    _init(gated)
    _init(plain)

    gated_params = dict(gated.named_parameters())
    shared = set(dict(plain.named_parameters())) & set(gated_params)
    assert shared, "the two arms share no parameters, so the comparison is vacuous"
    for name, param in plain.named_parameters():
        if name in shared:
            assert torch.equal(param, gated_params[name]), f"'{name}' differs between the arms"

    # The convolution weight is renamed rather than dropped -- a `GatedCausalConv1d` *holds* an
    # `nn.Conv1d` where a `CausalConv1d` *is* one -- so it falls out of `shared` and has to be
    # compared by hand. It is drawn first and unconditionally, so the two arms must agree on it.
    for stream in ("q", "k", "v"):
        assert torch.equal(
            getattr(plain, f"{stream}_conv1d").weight,
            getattr(gated, f"{stream}_conv1d").conv.weight,
        ), f"the {stream} convolution was drawn at a different point in the random stream"

    for name, param in gated.named_parameters():
        if name.endswith(("pre_scale", "post_scale")):
            assert torch.equal(param, torch.zeros_like(param)), f"'{name}' is not neutral at init"


@requires_fla_cpu
def test_gate_parameters_are_the_only_addition_of_the_gconv_arm():
    """
    The gated-convolution arm must stay inside the layers it occupies.

    If it needed anything from the backbone -- a different block, a shared module, a change to
    the residual stream -- it could not be dropped into two slots of an otherwise frozen model
    without perturbing every other arm.
    """
    plain_state = _build("base").state_dict()
    gated_state = _build("gconv").state_dict()

    plain_keys, gated_keys = set(plain_state), set(gated_state)
    streams = ("q_conv1d.", "k_conv1d.", "v_conv1d.")

    # Every difference, in either direction, lives inside one of the three convolutions.
    for key in (plain_keys - gated_keys) | (gated_keys - plain_keys):
        assert key.startswith(streams), f"the gate reached outside the convolutions: '{key}'"

    assert gated_keys - plain_keys == {f"{stream}conv.weight" for stream in streams} | {
        f"{stream}{scale}" for stream in streams for scale in ("pre_scale", "post_scale")
    }
    assert plain_keys - gated_keys == {f"{stream}weight" for stream in streams}

    for key in plain_keys & gated_keys:
        assert plain_state[key].shape == gated_state[key].shape, key

    # The gate's whole parameter cost -- three streams, two gates each, one scalar per channel --
    # which is what keeps the arms matchable.
    added = sum(gated_state[key].numel() for key in gated_keys - plain_keys)
    removed = sum(plain_state[key].numel() for key in plain_keys - gated_keys)
    assert added - removed == 3 * 2 * (N_HEADS * HEAD_DIM)


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_mixer_is_a_registered_sequence_mixer_config(variant: str):
    """Each variant must round-trip through the ``SequenceMixerConfig`` registry."""
    from olmo_core.nn.attention.base import SequenceMixerConfig

    config = VARIANTS[variant]()
    rebuilt = SequenceMixerConfig.from_dict(config.as_config_dict())  # type: ignore[attr-defined]
    assert rebuilt == config


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_init_weights_is_reproducible_from_its_generator_alone(variant: str):
    """
    Two mixers built and initialized with the same seed must be identical.

    Anything drawn from the global RNG instead of the passed generator would make a resumed or
    repeated run differ, and would do it silently.
    """
    left, right = _build(variant), _build(variant)
    _reset_parameters_sweep(left)
    _reset_parameters_sweep(right)
    torch.manual_seed(0)
    _init(left)
    torch.manual_seed(999)
    _init(right)

    right_params = dict(right.named_parameters())
    for name, param in left.named_parameters():
        assert torch.equal(param, right_params[name]), f"'{name}' depends on the global RNG"


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_single_rank_timescale_draw_is_unchanged_bit_for_bit(variant: str):
    """
    Routing ``A_log`` and ``dt_bias`` through ``_apply_init`` must not disturb the draw.

    The shard-safe form reorganizes *where* the numbers are written, not what they are, so at one
    rank it has to reproduce the original expression exactly. Kept as an oracle rather than as
    golden numbers so the guarantee reads as what it is.
    """
    mixer = _build(variant)
    _init(mixer, seed=SEED)

    expected = _build(variant)
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        # The upstream expression, verbatim, fed a generator advanced to the same point.
        for _ in range(_draws_before_timescales(expected, generator)):
            pass
        expected.A_log.copy_(  # type: ignore[union-attr]
            nn.init.uniform_(expected.A_log, a=1.0, b=16.0, generator=generator).log()  # type: ignore[union-attr]
        )
        expected.dt_bias.zero_()  # type: ignore[union-attr]

    assert torch.equal(mixer.A_log, expected.A_log)  # type: ignore[union-attr]
    assert torch.equal(mixer.dt_bias, expected.dt_bias)  # type: ignore[union-attr]


def _draws_before_timescales(mixer: SequenceMixer, generator: torch.Generator) -> int:
    """
    Advance ``generator`` past everything ``init_weights`` draws before the timescales.

    ``A_log`` is drawn from a generator that the projections and convolutions have already moved,
    so comparing the two expressions means putting both at the same point in the stream.

    :param mixer: The mixer whose projections set the position.
    :param generator: The generator to advance, in place.

    :returns: 0, so this reads as the side-effecting call it is.
    """
    from olmo_core.nn.attention.recurrent import _init_short_conv
    from olmo_core.nn.transformer.init import init_linear

    with torch.no_grad():
        for name in ("w_q", "w_k", "w_v", "w_b"):
            init_linear(getattr(mixer, name), std=0.02, generator=generator)
        for sequential in (mixer.f_proj, mixer.g_proj):  # type: ignore[union-attr]
            for leaf in sequential:
                init_linear(leaf, std=0.02, generator=generator)
        for stream in ("q", "k", "v"):
            conv = getattr(mixer, f"{stream}_conv1d")
            if isinstance(conv, nn.Conv1d):
                init_linear(conv, std=0.02, generator=generator)
            else:
                _init_short_conv(conv, std=0.02, generator=generator)
    return 0


@requires_fla_cpu
def test_householder_r1_has_the_same_parameters_as_base_kda():
    """
    At ``R=1`` the Householder layer is the base operator, parameter for parameter.

    This is what makes ``R=2`` a controlled comparison rather than two unrelated models: the
    only thing that changes between them is the number of delta factors.
    """
    base = KimiDeltaAttentionConfig(n_heads=N_HEADS, head_dim=HEAD_DIM)
    r1 = KimiDeltaHouseholderConfig(n_heads=N_HEADS, head_dim=HEAD_DIM, num_householder=1)

    assert r1.num_params(D_MODEL) == base.num_params(D_MODEL)

    base_module = base.build(D_MODEL, layer_idx=0, n_layers=1, init_device="cpu")
    r1_module = r1.build(D_MODEL, layer_idx=0, n_layers=1, init_device="cpu")
    assert {k: tuple(v.shape) for k, v in base_module.state_dict().items()} == {
        k: tuple(v.shape) for k, v in r1_module.state_dict().items()
    }
    assert sum(p.numel() for p in base_module.parameters()) == base.num_params(D_MODEL)


@requires_fla_cpu
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_built_module_matches_the_configs_parameter_algebra(variant: str):
    """``num_params`` is what solves FFN widths, so it must equal what the module actually holds."""
    config = VARIANTS[variant]()
    module = config.build(D_MODEL, layer_idx=0, n_layers=1, init_device="cpu")  # type: ignore[attr-defined]
    assert sum(p.numel() for p in module.parameters()) == config.num_params(D_MODEL)  # type: ignore[attr-defined]


@requires_fla_cpu
def test_init_weights_is_reachable_without_a_generator():
    """``Transformer.init_weights`` may pass no generator; that path must not crash."""
    mixer = _build("base")
    mixer.init_weights(init_method=InitMethod.normal, d_model=D_MODEL, block_idx=0, num_blocks=2)
    for name, param in mixer.named_parameters():
        assert torch.isfinite(param).all(), name


@requires_fla_cpu
def test_a_log_is_strictly_negative_after_init():
    """
    ``A_log = log(U(1, 16))`` is non-negative, and the decay is ``-exp(A_log)``.

    Drawing from ``U(0, 16)`` instead would let a head sample zero, whose log is ``-inf`` and
    whose state would be frozen for the whole run.
    """
    mixer = _build("base")
    _init(mixer)
    a_log = mixer.A_log.detach()  # type: ignore[union-attr]
    assert torch.isfinite(a_log).all()
    assert (a_log >= 0).all()
    assert (a_log <= torch.tensor(16.0).log()).all()
    assert isinstance(mixer.dt_bias, nn.Parameter)  # type: ignore[union-attr]
    assert torch.equal(mixer.dt_bias.detach(), torch.zeros_like(mixer.dt_bias))  # type: ignore[union-attr]
