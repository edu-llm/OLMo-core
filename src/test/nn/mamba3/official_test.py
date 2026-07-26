"""
Tests for the official ``mamba-ssm`` Mamba-3 SISO Triton kernel path.

The upstream kernel is treated as correct and is never modified; what is under test here is
the *mapping* from this repo's tensor layout and discretization onto its signature. The
numerical oracle stays :func:`mamba3_ssd_reference`.

Every comparison is bf16-appropriate: ``mamba3_siso_combined`` hard-casts ``Q``/``K``/``V``/
``Trap``/``Angles`` to bfloat16 before dispatching, so the best achievable agreement with the
fp32 reference is a few bf16 ULPs (relative eps 2^-8 = 3.9e-3) accumulated over the scan. The
measured relative RMS deviation is ~4e-3 forward and ~4-8e-3 backward; the gates below sit at
2e-2/3e-2, i.e. 3-5x headroom, which is still two orders of magnitude tighter than the O(1)
error any genuine mapping mistake produces.
"""

import pytest
import torch

from olmo_core.nn.mamba3.mamba3_ssd_api import dispatch_mamba3_ssd, mamba3_ssd_reference
from olmo_core.nn.mamba3.mamba3_ssd_fast import mamba3_ssd_fast
from olmo_core.nn.mamba3.mamba3_ssd_official import (
    mamba3_ssd_official,
    official_mamba3_is_available,
)
from olmo_core.testing import requires_gpu

requires_official_mamba3 = pytest.mark.skipif(
    not official_mamba3_is_available(),
    reason="the official mamba-ssm Mamba-3 SISO kernel is not installed",
)


def _inputs(
    *,
    batch: int = 2,
    seq_len: int = 40,
    n_heads: int = 4,
    head_dim: int = 8,
    n_groups: int = 1,
    d_state: int = 8,
    block_size: int = 2,
    rank: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    torch.manual_seed(seed)
    n_blocks = d_state // block_size
    angles = block_size * (block_size - 1) // 2

    def r(*shape):
        return torch.randn(*shape, device=device)

    return dict(
        x=r(batch, seq_len, n_heads, head_dim),
        B=r(batch, seq_len, n_groups, rank, d_state),
        C=r(batch, seq_len, n_groups, rank, d_state),
        dt=torch.rand(batch, seq_len, n_heads, device=device) * 0.1 + 0.01,
        A=-torch.rand(n_heads, device=device) - 0.5,
        lam=torch.rand(batch, seq_len, n_heads, device=device),
        theta=r(batch, seq_len, n_groups, n_blocks, angles),
        heads_per_group=n_heads // n_groups,
        block_size=block_size,
    )


def _split(kwargs):
    kwargs = dict(kwargs)
    return kwargs, kwargs.pop("block_size"), kwargs.pop("heads_per_group")


def _assert_matches(actual, expected, *, rms: float, peak: float, msg: str = ""):
    """Compare on relative RMS and relative peak rather than elementwise assert_close.

    Elementwise ``rtol`` is the wrong instrument for a bf16 reduction: the output is a signed
    sum over the sequence, so individual entries pass through zero and any fixed ``rtol``
    becomes meaningless there while ``atol`` has to be tied to the tensor scale anyway.
    """
    actual, expected = actual.float(), expected.float()
    diff = actual - expected
    rel_rms = (diff.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()
    rel_peak = (diff.abs().max() / expected.abs().max()).item()
    assert rel_rms < rms, f"{msg} relative RMS {rel_rms:.5f} >= {rms}"
    assert rel_peak < peak, f"{msg} relative peak {rel_peak:.5f} >= {peak}"


# Two sequence lengths x two head configurations, plus a ragged length that is neither a
# multiple of the kernel's 64-wide chunk nor of the head count.
FORWARD_CONFIGS = [
    pytest.param(
        dict(batch=2, seq_len=40, n_heads=4, head_dim=8, n_groups=1, d_state=8), id="t40-h4g1"
    ),
    pytest.param(
        dict(batch=1, seq_len=128, n_heads=4, head_dim=16, n_groups=2, d_state=16), id="t128-h4g2"
    ),
    pytest.param(
        dict(batch=1, seq_len=65, n_heads=2, head_dim=32, n_groups=2, d_state=32), id="t65-h2g2"
    ),
]


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("cfg", FORWARD_CONFIGS)
def test_official_kernel_matches_reference_forward(cfg):
    """The official SISO kernel must reproduce the reference at ``block_size == 2``."""
    kwargs, block_size, heads_per_group = _split(_inputs(**cfg))

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_official(**kwargs, heads_per_group=heads_per_group, block_size=block_size)

    assert actual.shape == expected.shape
    _assert_matches(actual, expected, rms=2e-2, peak=3e-2, msg="forward")


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("seq_len", [1, 3, 17, 64, 65], ids=lambda t: f"t{t}")
def test_official_kernel_handles_ragged_sequence_lengths(seq_len: int):
    """Sequence lengths shorter than / not a multiple of the kernel chunk must still work."""
    kwargs, block_size, heads_per_group = _split(_inputs(batch=1, seq_len=seq_len))

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_official(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    _assert_matches(actual, expected, rms=2e-2, peak=3e-2, msg=f"forward t={seq_len}")


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("cfg", FORWARD_CONFIGS[:2])
def test_official_kernel_matches_reference_backward(cfg):
    """Backward must agree too: every input the mixer differentiates through."""
    kwargs, block_size, heads_per_group = _split(_inputs(**cfg))
    torch.manual_seed(1234)
    grad_out = torch.randn(kwargs["x"].shape, device="cuda")

    grads = {}
    for name in ("ref", "official"):
        args = {k: v.clone().requires_grad_(True) for k, v in kwargs.items()}
        if name == "ref":
            out = mamba3_ssd_reference(
                **args, heads_per_group=heads_per_group, block_size=block_size
            )
        else:
            out = mamba3_ssd_official(
                **args, heads_per_group=heads_per_group, block_size=block_size
            )
        (out.float() * grad_out).sum().backward()
        grads[name] = {k: v.grad for k, v in args.items()}

    for key, expected in grads["ref"].items():
        actual = grads["official"][key]
        assert actual is not None, f"official path produced no gradient for {key}"
        _assert_matches(actual, expected, rms=3e-2, peak=5e-2, msg=f"grad {key}")


@requires_gpu
@requires_official_mamba3
def test_dispatch_routes_block_size_2_to_the_official_kernel():
    """The routing must actually reach the kernel, not leave it sitting in the tree."""
    kwargs, block_size, heads_per_group = _split(_inputs())

    dispatched = dispatch_mamba3_ssd(
        **kwargs,
        heads_per_group=heads_per_group,
        block_size=block_size,
        prefer_official_kernel=True,
    )
    direct = mamba3_ssd_official(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert torch.equal(dispatched, direct), "dispatch did not take the official-kernel path"


@requires_gpu
@requires_official_mamba3
def test_dispatch_uses_the_official_kernel_under_bf16_autocast():
    """
    Autocast is the signal that reduced precision is acceptable, so it is what arms the
    default routing. The upstream kernel hard-casts to bf16 internally, so routing an fp32
    call into it would silently downgrade precision that the chunked path still delivers.

    The default is ``prefer_fast_rotation=True``, so the armed path is ``mamba3_ssd_fast`` -- the
    *same* upstream Triton kernel with a faster ``b >= 3`` rotation, not the chunked PyTorch form.
    """
    kwargs, block_size, heads_per_group = _split(_inputs())

    with torch.autocast("cuda", dtype=torch.bfloat16):
        auto = dispatch_mamba3_ssd(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
        direct = mamba3_ssd_fast(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert auto.dtype == torch.bfloat16, f"autocast ignored; got {auto.dtype}"
    assert torch.equal(auto, direct), "autocast dispatch did not take the fast-rotation kernel path"


@requires_gpu
@requires_official_mamba3
def test_dispatch_keeps_fp32_calls_on_the_chunked_path():
    """Without autocast the fp32 chunked form is more accurate, so it must stay the default."""
    from olmo_core.nn.mamba3.mamba3_ssd_chunked import mamba3_ssd_chunked

    kwargs, block_size, heads_per_group = _split(_inputs())

    dispatched = dispatch_mamba3_ssd(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    chunked = mamba3_ssd_chunked(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert torch.equal(dispatched, chunked), "fp32 dispatch left the chunked path"


@requires_gpu
@requires_official_mamba3
def test_dispatch_falls_back_to_chunked_for_mimo():
    """The upstream kernel is SISO only; ``mimo_rank > 1`` must not reach it."""
    from olmo_core.nn.mamba3.mamba3_ssd_chunked import mamba3_ssd_chunked

    kwargs, block_size, heads_per_group = _split(_inputs(rank=3))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        dispatched = dispatch_mamba3_ssd(
            **kwargs, heads_per_group=heads_per_group, block_size=block_size
        )
        chunked = mamba3_ssd_chunked(
            **kwargs, heads_per_group=heads_per_group, block_size=block_size
        )
    assert torch.equal(dispatched, chunked), "MIMO input reached the SISO kernel"


def test_dispatch_falls_back_to_chunked_on_cpu():
    """The kernel is CUDA-only; a CPU call must take the chunked path."""
    from olmo_core.nn.mamba3.mamba3_ssd_chunked import mamba3_ssd_chunked

    kwargs, block_size, heads_per_group = _split(_inputs(device="cpu"))

    dispatched = dispatch_mamba3_ssd(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    chunked = mamba3_ssd_chunked(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert torch.equal(dispatched, chunked), "CPU call did not fall back to the chunked path"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(device="cpu"), "CUDA"),
        (dict(device="cpu", rank=3), "mimo_rank"),
    ],
    ids=["cpu", "mimo"],
)
def test_dispatch_refuses_to_silently_downgrade_an_explicit_kernel_request(overrides, expected):
    """
    ``prefer_official_kernel=True`` must raise when the kernel cannot run, not fall back.

    A preference (``None``) may fall back; a request may not. Swallowing it meant a benchmark or
    a parity test could believe it had exercised the Triton kernel while measuring the chunked
    PyTorch path -- the failure mode is a wrong *conclusion*, with no error to notice.
    """
    kwargs, block_size, heads_per_group = _split(_inputs(**overrides))

    with pytest.raises(RuntimeError, match=expected):
        dispatch_mamba3_ssd(
            **kwargs,
            heads_per_group=heads_per_group,
            block_size=block_size,
            prefer_official_kernel=True,
        )


@requires_gpu
@requires_official_mamba3
def test_upstream_effective_angle_is_tanh_of_angles_times_pi_times_dt():
    """
    Pin the upstream angle parameterization, which is what forces ``Angles = 0``.

    ``angle_dt_fwd`` computes ``cumsum(tanh(Angles) * pi * DT) mod 2pi``, not ``cumsum(Angles *
    DT)`` as the docstring says. The exact inverse is therefore
    ``Angles = atanh(theta / (pi * dt))``, which only exists while ``|theta| < pi * dt``. This
    repo's ``theta`` is an unconstrained ``nn.Linear`` output and its rotation is shared across
    a whole group while ``DT`` is per head, so the native path cannot express it in general --
    hence the production mapping pre-rotates ``B``/``C`` and passes zero angles instead.

    This test exercises the one regime where the native path *is* expressible: one head per
    group and a ``theta`` constructed from the kernel's own parameterization.
    """
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    torch.manual_seed(7)
    batch, seq_len, n_heads, head_dim, d_state = 1, 128, 2, 16, 16
    n_groups = n_heads  # heads_per_group == 1: only then is a per-head DT compatible

    angle_rates = torch.randn(batch, seq_len, n_heads, d_state // 2, device="cuda") * 0.7
    dt = torch.rand(batch, seq_len, n_heads, device="cuda") * 0.1 + 0.01
    theta = torch.tanh(angle_rates) * torch.pi * dt.unsqueeze(-1)

    kwargs = dict(
        x=torch.randn(batch, seq_len, n_heads, head_dim, device="cuda"),
        B=torch.randn(batch, seq_len, n_groups, 1, d_state, device="cuda"),
        C=torch.randn(batch, seq_len, n_groups, 1, d_state, device="cuda"),
        dt=dt,
        A=-torch.rand(n_heads, device="cuda") - 0.5,
        lam=torch.rand(batch, seq_len, n_heads, device="cuda"),
        theta=theta.unsqueeze(-1),
    )
    expected = mamba3_ssd_reference(**kwargs, heads_per_group=1, block_size=2)

    actual = mamba3_siso_combined(
        kwargs["C"].squeeze(3).contiguous(),
        kwargs["B"].squeeze(3).contiguous(),
        kwargs["x"].contiguous(),
        (dt * kwargs["A"]).permute(0, 2, 1).contiguous(),
        dt.permute(0, 2, 1).contiguous(),
        torch.logit(kwargs["lam"], eps=1e-6).permute(0, 2, 1).contiguous(),
        torch.zeros(n_heads, d_state, device="cuda"),
        torch.zeros(n_heads, d_state, device="cuda"),
        angle_rates,
        chunk_size=64,
    )
    # Looser than the zero-angle path: `Angles` is rounded to bf16 and both `tanh` and the
    # cos/sin are PTX `.approx` instructions, so the rotation itself carries bf16 error.
    _assert_matches(actual, expected, rms=3e-2, peak=4e-2, msg="native-angle path")


# ---------------------------------------------------------------------------------------
# block_size >= 3: the same kernel, reached by pre-rotating B/C with the non-abelian SO(b)
# prefix product and handing the kernel zero angles.
# ---------------------------------------------------------------------------------------

BLOCKED_CONFIGS = [
    pytest.param(
        dict(batch=2, seq_len=40, n_heads=4, head_dim=8, n_groups=1, d_state=12, block_size=3),
        id="t40-h4g1-b3",
    ),
    pytest.param(
        dict(batch=1, seq_len=96, n_heads=4, head_dim=16, n_groups=2, d_state=24, block_size=3),
        id="t96-h4g2-b3",
    ),
    pytest.param(
        dict(batch=1, seq_len=65, n_heads=2, head_dim=16, n_groups=1, d_state=20, block_size=5),
        id="t65-h2g1-b5",
    ),
]


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("cfg", BLOCKED_CONFIGS)
def test_official_kernel_matches_reference_for_non_abelian_blocks(cfg):
    """
    ``SO(b >= 3)`` rides the same kernel: the rotation is pure ``B``/``C`` preprocessing, and
    with zero ``Angles`` the kernel is a plain scalar-decay SSD scan over whatever it is given.
    """
    kwargs, block_size, heads_per_group = _split(_inputs(**cfg))

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_official(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    _assert_matches(actual, expected, rms=2e-2, peak=3e-2, msg=f"forward b={block_size}")


@requires_gpu
@requires_official_mamba3
def test_official_kernel_matches_reference_backward_for_non_abelian_blocks():
    """Backward through the ``matrix_exp`` prefix product must survive the kernel too."""
    kwargs, block_size, heads_per_group = _split(
        _inputs(batch=2, seq_len=40, n_heads=4, head_dim=8, n_groups=1, d_state=12, block_size=3)
    )
    torch.manual_seed(4321)
    grad_out = torch.randn(kwargs["x"].shape, device="cuda")

    grads = {}
    for name in ("ref", "official"):
        args = {k: v.clone().requires_grad_(True) for k, v in kwargs.items()}
        if name == "ref":
            out = mamba3_ssd_reference(
                **args, heads_per_group=heads_per_group, block_size=block_size
            )
        else:
            out = mamba3_ssd_official(
                **args, heads_per_group=heads_per_group, block_size=block_size
            )
        (out.float() * grad_out).sum().backward()
        grads[name] = {k: v.grad for k, v in args.items()}

    for key, expected in grads["ref"].items():
        actual = grads["official"][key]
        assert actual is not None, f"official path produced no gradient for {key}"
        _assert_matches(actual, expected, rms=3e-2, peak=5e-2, msg=f"grad {key} (b=3)")


@requires_gpu
@requires_official_mamba3
def test_zero_angles_make_the_kernel_rotation_the_identity():
    """
    The load-bearing assumption behind the whole ``b >= 3`` reuse.

    ``angle_dt_fwd`` runs ``tanh``, a ``* pi * DT`` scaling and a ``mod 2pi`` before the
    cos/sin, any of which could have made zero non-neutral. It does not: with ``Angles = 0``
    the kernel must agree with the reference run at ``theta = 0``, i.e. apply no rotation of
    its own. The second half of the test keeps this from passing vacuously by confirming the
    rotation is not a no-op on this data in the first place.
    """
    kwargs, block_size, heads_per_group = _split(_inputs(seq_len=64))
    unrotated = dict(kwargs, theta=torch.zeros_like(kwargs["theta"]))

    expected = mamba3_ssd_reference(
        **unrotated, heads_per_group=heads_per_group, block_size=block_size
    )
    actual = mamba3_ssd_official(
        **unrotated, heads_per_group=heads_per_group, block_size=block_size
    )
    _assert_matches(actual, expected, rms=2e-2, peak=3e-2, msg="zero-angle identity")

    rotated = mamba3_ssd_reference(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    drift = ((rotated - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()
    assert drift > 0.5, f"rotation is nearly a no-op here ({drift:.3f}); the test has no teeth"


@requires_gpu
@requires_official_mamba3
def test_official_kernel_keeps_the_prefix_product_in_fp32_under_autocast():
    """
    Orthogonality drift of the ``b >= 3`` prefix product is ``O(T * eps)``: fp32 gives 6.4e-5
    at ``T = 1024`` but bf16 gives 2.7e-1, i.e. it stops being a rotation. Autocast intercepts
    at the *op* level, so the prefix product has to be wrapped in an ``autocast(enabled=False)``
    region -- casting the tensors to fp32 is not enough. A blocked run under autocast must
    therefore still track the fp32 reference at bf16 accuracy, not at rotation-is-broken
    accuracy.
    """
    kwargs, block_size, heads_per_group = _split(
        _inputs(batch=1, seq_len=256, n_heads=2, head_dim=8, n_groups=1, d_state=12, block_size=3)
    )

    expected = mamba3_ssd_reference(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = mamba3_ssd_official(
            **kwargs, heads_per_group=heads_per_group, block_size=block_size
        )
    assert actual.dtype == torch.bfloat16, f"autocast ignored; got {actual.dtype}"
    _assert_matches(actual, expected, rms=3e-2, peak=5e-2, msg="b=3 under autocast")


@requires_gpu
@requires_official_mamba3
def test_dispatch_routes_block_size_3_to_the_official_kernel():
    """``b >= 3`` must reach the kernel as well, not fall back to the chunked form.

    With the default ``prefer_fast_rotation=True`` this is the ``mamba3_ssd_fast`` path (Rodrigues +
    adaptive scan chunk over the same Triton kernel), which is exactly the route the ``b=3`` run
    depends on.
    """
    kwargs, block_size, heads_per_group = _split(_inputs(d_state=12, block_size=3))

    dispatched = dispatch_mamba3_ssd(
        **kwargs,
        heads_per_group=heads_per_group,
        block_size=block_size,
        prefer_official_kernel=True,
    )
    direct = mamba3_ssd_fast(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    assert torch.equal(dispatched, direct), "b=3 dispatch did not take the fast-rotation kernel path"


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("rotation_block_size", [2, 3], ids=["b2", "b3"])
def test_mixer_reaches_the_official_kernel_under_autocast(rotation_block_size: int):
    """
    End to end: a SISO mixer training in bf16 must actually land on the kernel.

    Everything above tests ``dispatch_mamba3_ssd`` directly; this is the one that would catch
    the routing being live in theory but unreachable from ``Mamba3Mixer`` in practice. The default
    ``prefer_fast_rotation=True`` reaches the kernel through ``mamba3_ssd_fast``, so this spies on
    the shared upstream Triton entry point (which both adapters call) rather than either adapter --
    the assertion is "reaches the Triton kernel, not the chunked PyTorch path".
    """
    from unittest.mock import patch

    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        mamba3_siso_combined as real_kernel,
    )

    from olmo_core.nn.mamba3 import Mamba3MixerConfig
    from olmo_core.nn.transformer.init import InitMethod

    torch.manual_seed(0)
    d_model = 32
    mixer = Mamba3MixerConfig(
        n_heads=2,
        head_dim=8,
        d_state=6 if rotation_block_size == 3 else 8,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=rotation_block_size,
    ).build(d_model, layer_idx=0, n_layers=2, init_device="cuda")
    mixer.init_weights(init_method=InitMethod.normal, d_model=d_model, block_idx=0, num_blocks=2)

    x = torch.randn(2, 16, d_model, device="cuda", requires_grad=True)
    spy = patch(
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined.mamba3_siso_combined",
        side_effect=real_kernel,
    )
    with spy as called, torch.autocast("cuda", dtype=torch.bfloat16):
        y = mixer(x)
    assert called.call_count == 1, "the mixer never reached the Triton kernel under autocast"

    y.float().pow(2).mean().backward()
    assert torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all()
