"""
Initialization of :class:`~olmo_core.nn.mamba3.Mamba3Mixer` under FSDP2 sharding.

Every parameter the mixer initializes is a sharded ``DTensor`` by the time training calls
``init_weights``, and a ``DTensor`` cannot be initialized by splitting it and writing into the
pieces: the split redistributes, so the pieces are fresh tensors and the in-place write lands
nowhere. Nothing raises when that happens -- the parameter simply keeps ``nn.Linear``'s default
init -- so the only way to hold the line is to run a real two-rank ``fully_shard`` and measure
what the parameters actually contain.
"""

import math
from typing import Optional

import pytest
import torch
import torch.nn as nn

from olmo_core.distributed.utils import get_full_tensor
from olmo_core.nn.mamba3 import Mamba3Mixer, Mamba3MixerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import run_distributed_test

D_MODEL = 128
SEED = 1234


def _mixer_config(*, fused: bool) -> Mamba3MixerConfig:
    """
    A mixer shaped like the production ``mamba-b3`` arm, narrow enough to shard on CPU.

    ``rotation_block_size=3`` and ``bc_bias=True`` match the arm; the widths are cut down so a
    two-rank run still splits every fused weight across ranks.
    """
    return Mamba3MixerConfig(
        n_heads=4,
        head_dim=32,
        d_state=96,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=3,
        bc_bias=True,
        fuse_input_projections=fused,
    )


def _build(fused: bool) -> Mamba3Mixer:
    return _mixer_config(fused=fused).build(D_MODEL, layer_idx=0, n_layers=2, init_device="cpu")


def _init(mixer: Mamba3Mixer, init_method: InitMethod = InitMethod.normal, seed: int = SEED):
    mixer.init_weights(
        init_method=init_method,
        d_model=D_MODEL,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(seed),
    )


def _logical_slices(mixer: Mamba3Mixer) -> dict[str, tuple[tuple[str, int], ...]]:
    """Map each fused weight to the (name, row count) of the projections packed into it."""
    inner = mixer.n_heads * mixer.head_dim
    bc_out = mixer.n_groups * mixer.mimo_rank * mixer.d_state
    theta_out = mixer.n_groups * mixer.n_rotation_blocks * mixer.angles_per_block
    return {
        "in_xz": (("in_x", inner), ("in_z", inner)),
        "in_bc": (("in_B", bc_out), ("in_C", bc_out)),
        "in_dynamics": (
            ("dt_proj", mixer.n_heads),
            ("lam_proj", mixer.n_heads),
            ("theta_proj", theta_out),
        ),
    }


def _projection_stds(mixer: Mamba3Mixer) -> dict[str, float]:
    """
    Sample standard deviation of every logical projection, fused or not.

    Reported per logical slice rather than per weight because the whole point of the fused
    layout is that its slices are drawn at different scales: ``theta_proj`` is ten times
    narrower than the rest, and a single number over the whole fused weight would average that
    away.
    """
    stds: dict[str, float] = {}
    if mixer.fuse_input_projections:
        for projection, slices in _logical_slices(mixer).items():
            weight = get_full_tensor(getattr(mixer, projection).weight.detach())
            sizes = [size for _, size in slices]
            for (name, _), chunk in zip(slices, weight.split(sizes, dim=0)):
                stds[name] = chunk.std().item()
    else:
        for name in ("in_x", "in_z", "in_B", "in_C", "dt_proj", "lam_proj", "theta_proj"):
            stds[name] = get_full_tensor(getattr(mixer, name).weight.detach()).std().item()
    return stds


def _assert_shard_is_real(weight: torch.Tensor, name: str, world_size: int) -> None:
    """Fail if `weight` is not actually split across ranks, so nothing passes by not sharding."""
    from torch.distributed.tensor import DTensor

    assert isinstance(weight, DTensor), f"{name} is not a DTensor; the test would prove nothing"
    assert weight.to_local().shape[0] < weight.shape[0], (
        f"{name} is a DTensor but not split along dim 0 "
        f"(local {tuple(weight.to_local().shape)}, full {tuple(weight.shape)}, "
        f"world size {world_size})"
    )


def _shard(mixer: Mamba3Mixer, world_size: int) -> None:
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


def _run_sharded_projection_init(world_size: int, fused: bool, reference: dict) -> None:
    _enter_worker()
    mixer = _build(fused=fused)
    _shard(mixer, world_size)

    checked = (
        ("in_xz", "in_bc", "in_dynamics")
        if fused
        else ("in_x", "in_z", "in_B", "in_C", "dt_proj", "lam_proj", "theta_proj")
    )
    for name in checked:
        _assert_shard_is_real(getattr(mixer, name).weight, name, world_size)

    _init(mixer)

    observed = _projection_stds(mixer)
    for name, expected in reference.items():
        assert observed[name] == pytest.approx(expected, rel=0.25), (name, observed, reference)
    # theta_proj is deliberately drawn ten times narrower than its neighbours, and it is the
    # slice a shard-blind init damages most: it sets the rotation angles, so widening it
    # shortens the memory horizon of the recurrence from step zero.
    assert observed["theta_proj"] < 0.2 * min(
        value for name, value in observed.items() if name != "theta_proj"
    )

    if fused:
        assert mixer.in_bc is not None and mixer.in_bc.bias is not None
        bias = get_full_tensor(mixer.in_bc.bias.detach())
        assert torch.equal(bias, torch.zeros_like(bias)), "fused B/C bias was not zeroed"


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_sharded_projections_give_every_slice_its_own_standard_deviation(fused: bool):
    reference_mixer = _build(fused=fused)
    _init(reference_mixer)
    reference = _projection_stds(reference_mixer)

    run_distributed_test(
        _run_sharded_projection_init,
        world_size=2,
        backend="gloo",
        func_args=(2, fused, reference),
    )


def _run_sharded_timescale_init(world_size: int, fused: bool, reference: dict) -> None:
    _enter_worker()
    mixer = _build(fused=fused)
    _shard(mixer, world_size)
    _assert_shard_is_real(mixer.A_log, "A_log", world_size)
    _assert_shard_is_real(mixer.dt_bias, "dt_bias", world_size)

    _init(mixer)

    for name, expected in reference.items():
        observed = get_full_tensor(getattr(mixer, name).detach())
        assert torch.equal(observed, torch.tensor(expected, dtype=observed.dtype)), (
            name,
            observed.tolist(),
            expected,
        )


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_sharded_timescale_parameters_honour_the_seeded_generator(fused: bool):
    """
    ``A_log`` and ``dt_bias`` must come from the generator the caller passed.

    A random op on a ``DTensor`` over a CPU mesh has no RNG tracker to fall back on, so it
    quietly ignores the supplied generator and draws from each process's default one. The
    parameters then differ between two runs of the same seeded configuration, which makes a
    resumed or repeated run unreproducible in exactly the two parameters that set the
    recurrence's timescale.
    """
    reference_mixer = _build(fused=fused)
    _init(reference_mixer)
    reference = {
        "A_log": reference_mixer.A_log.detach().tolist(),
        "dt_bias": reference_mixer.dt_bias.detach().tolist(),
    }

    run_distributed_test(
        _run_sharded_timescale_init,
        world_size=2,
        backend="gloo",
        func_args=(2, fused, reference),
    )


@torch.no_grad()
def _legacy_init_weights(
    mixer: Mamba3Mixer,
    *,
    init_method: InitMethod,
    d_model: int,
    block_idx: int,
    num_blocks: int,
    std: float = 0.02,
    generator: Optional[torch.Generator] = None,
) -> None:
    """
    ``Mamba3Mixer.init_weights`` as it stood before the sharding fix, copied verbatim.

    Kept as an oracle rather than a set of golden numbers so the single-rank guarantee reads as
    what it is: the fix reorganizes *where* the draws are written, and must not disturb the
    order or the size of a single draw.
    """
    from olmo_core.nn.transformer.init import init_linear

    if init_method == InitMethod.normalized:
        std = d_model**-0.5

    if mixer.fuse_input_projections:
        assert mixer.in_xz is not None
        assert mixer.in_bc is not None
        assert mixer.in_dynamics is not None

        def init_slice(weight, bias=None, *, slice_std: float = std) -> None:
            nn.init.trunc_normal_(
                weight,
                mean=0.0,
                std=slice_std,
                a=-3 * slice_std,
                b=3 * slice_std,
                generator=generator,
            )
            if bias is not None:
                nn.init.zeros_(bias)

        inner = mixer.n_heads * mixer.head_dim
        bc_out = mixer.n_groups * mixer.mimo_rank * mixer.d_state
        theta_out = mixer.n_groups * mixer.n_rotation_blocks * mixer.angles_per_block
        x_weight, z_weight = mixer.in_xz.weight.split((inner, inner), dim=0)
        init_slice(x_weight)
        init_slice(z_weight)
        b_weight, c_weight = mixer.in_bc.weight.split((bc_out, bc_out), dim=0)
        if mixer.in_bc.bias is None:
            b_bias = c_bias = None
        else:
            b_bias, c_bias = mixer.in_bc.bias.split((bc_out, bc_out), dim=0)
        init_slice(b_weight, b_bias)
        init_slice(c_weight, c_bias)
        dt_weight, lam_weight, theta_weight = mixer.in_dynamics.weight.split(
            (mixer.n_heads, mixer.n_heads, theta_out), dim=0
        )
        init_slice(dt_weight)
        init_slice(lam_weight)
        init_slice(theta_weight, slice_std=std * 0.1)
    else:
        for w in (mixer.in_x, mixer.in_z, mixer.in_B, mixer.in_C, mixer.dt_proj, mixer.lam_proj):
            init_linear(w, std=std, generator=generator)
        init_linear(mixer.theta_proj, std=std * 0.1, generator=generator)

    mixer.A_log.copy_(
        nn.init.uniform_(
            mixer.A_log, a=mixer.a_log_init_min, b=mixer.a_log_init_max, generator=generator
        ).log()
    )

    dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
    dt = torch.exp(
        nn.init.uniform_(mixer.dt_bias, generator=generator) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min),
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    mixer.dt_bias.copy_(inv_dt)

    mixer.o_norm_weight.fill_(1.0)
    if mixer.bc_norm_enabled:
        mixer.bc_norm_b.fill_(1.0)
        mixer.bc_norm_c.fill_(1.0)

    if init_method == InitMethod.llama:
        std = std / (2 * num_blocks) ** 0.5
    elif init_method == InitMethod.llama_depth:
        std = std / (2 * (block_idx + 1)) ** 0.5
    elif init_method == InitMethod.normalized:
        std = std / (2 * num_blocks) ** 0.5

    init_linear(mixer.out_proj, std=std, generator=generator)


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize(
    "init_method",
    [InitMethod.normal, InitMethod.llama, InitMethod.llama_depth, InitMethod.normalized],
)
def test_single_rank_init_is_unchanged_bit_for_bit(init_method: InitMethod, fused: bool):
    """A shard-safe init must draw the same numbers as before on one rank."""
    actual = _build(fused=fused)
    actual.init_weights(
        init_method=init_method,
        d_model=D_MODEL,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(SEED),
    )

    expected = _build(fused=fused)
    _legacy_init_weights(
        expected,
        init_method=init_method,
        d_model=D_MODEL,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(SEED),
    )

    actual_state, expected_state = actual.state_dict(), expected.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, value in actual_state.items():
        assert torch.equal(value, expected_state[name]), name
