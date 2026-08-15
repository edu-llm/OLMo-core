"""
GPU tests that the ``b=2`` and ``b=3`` ablation arms both run on the path they will train on.

Everything else about this comparison is checked on CPU or on ``meta``, which is enough for the
configuration contract and nothing at all for the kernel. The production settings --
``prefer_official_kernel=True``, ``ssd_backend="official_fast"``,
``rotation_scan_impl="quaternion"`` -- are all strict requests that raise rather than fall back, and
they were only ever exercised at ``b=3``: the August wave ran the treatment arm and never the
control. So the one configuration that has never executed is the baseline of the experiment.

These skip without a GPU and are the gate to run on the platform image before committing the wave.
"""

import pytest
import torch

from olmo_core.nn.mamba3.mamba3_ssd_api import (
    get_backend_counters,
    reset_backend_counters,
)
from olmo_core.nn.mamba3.mixer import Mamba3MixerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.testing import requires_gpu

D_MODEL = 1024

#: Exactly what `OLMo3-370M-mamba3-b-ablation.py` builds, minus the shell around it.
PRODUCTION_MIXER = dict(
    n_heads=32,
    head_dim=64,
    d_state=192,
    n_groups=1,
    mimo_rank=1,
    norm_eps=1e-6,
    bc_norm=True,
    bc_bias=False,
    dynamic_a=True,
    d_skip=True,
    norm_before_gate=True,
    bc_bias_after_norm=True,
    dt_scaled_rotation=True,
    rope_fraction=0.5,
    rotation_timescale="group_mean",
    rotation_scan_impl="quaternion",
    ssd_backend="official_fast",
    prefer_official_kernel=True,
)


def build_arm(block_size: int) -> torch.nn.Module:
    mixer = Mamba3MixerConfig(rotation_block_size=block_size, **PRODUCTION_MIXER).build(
        D_MODEL, layer_idx=0, n_layers=2, init_device="cuda"
    )
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=D_MODEL,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    return mixer


@requires_gpu
@pytest.mark.parametrize("block_size", [2, 3])
def test_both_arms_run_on_the_official_kernel(block_size):
    """
    Forward and backward at the production geometry, in bfloat16, on the fused path.

    ``official_fast`` is a strict request: dispatch raises rather than answering with the chunked
    form, so reaching the end of this test at all is the assertion. The counter check is the
    second half of it -- an arm that fell through to a slower backend without raising would train
    and would silently not be the thing that was benchmarked.
    """
    mixer = build_arm(block_size)
    x = torch.randn(2, 512, D_MODEL, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    reset_backend_counters()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = mixer(x)
    y.float().square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y.float()).all()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, f"b={block_size}: {name} received no gradient"
        assert torch.isfinite(
            parameter.grad.float()
        ).all(), f"b={block_size}: {name} gradient is not finite"

    counters = get_backend_counters()
    assert counters.get("official_fast", 0) > 0, f"b={block_size} did not reach the fused kernel"
    assert counters.get("chunked", 0) == 0, f"b={block_size} fell through to the chunked scan"


@requires_gpu
def test_the_quaternion_request_is_inert_at_b2_rather_than_fatal():
    """
    Both arms name the same scan so the config diff stays one field wide. At ``b=2`` that name
    describes nothing -- SO(2) prefix products collapse to a ``cumsum`` and the branch short-
    circuits before the implementation is consulted -- and it must stay harmless rather than
    become a strict request nothing can satisfy. Setting it per arm instead would buy nothing and
    cost the contract a second differing field.
    """
    mixer = build_arm(2)
    x = torch.randn(2, 256, D_MODEL, device="cuda", dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = mixer(x)

    assert torch.isfinite(y.float()).all()


@requires_gpu
def test_b3_is_the_slower_arm_so_it_bounds_the_runtime_request():
    """
    The wave's hours are sized off ``b=3``, on the reasoning that a non-commutative prefix product
    cannot be cheaper than the ``cumsum`` it replaces. That is the assumption the whole schedule
    rests on, so measure it rather than assume it.

    Deliberately loose: this asserts an ordering, not a ratio, because a ratio measured on one card
    at one shape would be quoted as if it were the wave's.
    """
    import time

    def median_step_seconds(block_size: int) -> float:
        mixer = build_arm(block_size)
        x = torch.randn(2, 1024, D_MODEL, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        timings = []
        for step in range(8):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = mixer(x)
            y.float().square().mean().backward()
            torch.cuda.synchronize()
            if step >= 3:  # discard warmup and any compile
                timings.append(time.perf_counter() - started)
            mixer.zero_grad(set_to_none=True)
            x.grad = None
        return sorted(timings)[len(timings) // 2]

    assert median_step_seconds(3) >= median_step_seconds(2)
