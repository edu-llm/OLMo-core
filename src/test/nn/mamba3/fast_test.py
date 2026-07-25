"""
Tests for :mod:`olmo_core.nn.mamba3.mamba3_ssd_fast`.

The claim under test is that the fast path computes the *same function* as
:mod:`mamba3_ssd_official`, only quicker -- so almost everything here is a parity assertion
against the untouched official path or against ``torch.matrix_exp``. The one place that is not a
parity check is the small-angle regime, which is where ``theta_proj`` initialises and where a
naive Rodrigues implementation produces NaN gradients on the first training step.
"""

import pytest
import torch

from olmo_core.nn.mamba3.mamba3_ssd_api import (
    _block_rotations,
    _cumulative_block_rotation,
    _rotate_bc_blocks,
    _skew_from_angles,
    dispatch_mamba3_ssd,
)
from olmo_core.nn.mamba3.mamba3_ssd_fast import (
    _rodrigues_so3,
    _rotate_bc_fused,
    fast_block_rotations,
    fast_cumulative_block_rotation,
    fast_mamba3_is_available,
    mamba3_ssd_fast,
)
from olmo_core.nn.mamba3.mamba3_ssd_official import mamba3_ssd_official
from olmo_core.testing import requires_gpu

requires_official_mamba3 = pytest.mark.skipif(
    not fast_mamba3_is_available(),
    reason="the official mamba-ssm Mamba-3 SISO kernel is not installed",
)


# ---------------------------------------------------------------------------------------
# Rodrigues == matrix_exp
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale", [0.0, 1e-8, 1e-4, 0.01, 1.0, 3.14159, 6.0], ids=lambda s: f"scale{s}"
)
def test_rodrigues_matches_matrix_exp(scale: float):
    """The closed form is an identity, not an approximation -- including at and near zero."""
    torch.manual_seed(0)
    theta = torch.randn(2, 64, 1, 8, 3, dtype=torch.float64) * scale

    expected = torch.matrix_exp(_skew_from_angles(theta, 3))
    actual = _rodrigues_so3(theta)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_rodrigues_matches_matrix_exp_at_exactly_zero():
    """All-zero angles must give exactly the identity, not a 0/0 NaN."""
    theta = torch.zeros(4, 3, dtype=torch.float64)
    rot = _rodrigues_so3(theta)

    assert torch.isfinite(rot).all()
    torch.testing.assert_close(rot, torch.eye(3, dtype=torch.float64).expand(4, 3, 3))


def test_rodrigues_gradient_is_finite_at_small_angles():
    """
    The failure this guards is a first-step NaN, not a wrong number.

    ``theta_proj`` initialises at ``std * 0.1``, so training *starts* in the small-angle regime.
    ``sin(phi)/phi`` at ``phi == 0`` is 0/0, and ``sqrt`` has an infinite derivative there, so an
    unclamped implementation emits NaN gradients even though the forward value looks fine and
    even though ``torch.where`` discards the offending branch.
    """
    for scale in (0.0, 1e-12, 1e-6, 1e-3):
        theta = (torch.randn(3, 5, 3, dtype=torch.float64) * scale).requires_grad_(True)
        _rodrigues_so3(theta).sum().backward()
        assert theta.grad is not None
        assert torch.isfinite(theta.grad).all(), f"non-finite gradient at scale {scale}"


def test_rodrigues_gradient_matches_matrix_exp():
    """Backward parity, not just forward: the training path differentiates through this."""
    torch.manual_seed(1)
    base = torch.randn(2, 16, 1, 4, 3, dtype=torch.float64) * 0.7
    grad_out = torch.randn(2, 16, 1, 4, 3, 3, dtype=torch.float64)

    grads = []
    for fn in (lambda t: torch.matrix_exp(_skew_from_angles(t, 3)), _rodrigues_so3):
        theta = base.clone().requires_grad_(True)
        (fn(theta) * grad_out).sum().backward()
        grads.append(theta.grad)

    torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("scale", [0.01, 1.0, 3.0])
def test_rodrigues_output_is_a_rotation(scale: float):
    """Orthogonal with determinant +1 -- the property the BIBO-stability argument rests on."""
    torch.manual_seed(2)
    rot = _rodrigues_so3(torch.randn(64, 3, dtype=torch.float64) * scale)

    eye = torch.eye(3, dtype=torch.float64).expand_as(rot)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), eye, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        torch.linalg.det(rot), torch.ones(64, dtype=torch.float64), rtol=0, atol=1e-12
    )


def test_rodrigues_survives_bfloat16_where_matrix_exp_does_not():
    """
    ``torch.matrix_exp`` accepts bfloat16 and returns silent NaN/Inf rather than raising, so the
    fp32 floor is *not* backed by a loud failure the way it is often assumed to be. The closed
    form degrades gracefully instead, which removes the hazard rather than relying on it.
    """
    torch.manual_seed(3)
    theta = (torch.randn(32, 3) * 0.5).bfloat16()

    assert not torch.isfinite(torch.matrix_exp(_skew_from_angles(theta, 3))).all()
    assert torch.isfinite(_rodrigues_so3(theta)).all()


def test_fast_block_rotations_delegates_for_other_block_sizes():
    """Only ``b == 3`` takes the closed form; every other ``b`` must be untouched."""
    torch.manual_seed(4)
    for block_size in (2, 4, 5):
        n_angles = block_size * (block_size - 1) // 2
        theta = torch.randn(2, 8, 1, 3, n_angles, dtype=torch.float64) * 0.5
        torch.testing.assert_close(
            fast_block_rotations(theta, block_size),
            _block_rotations(theta, block_size),
            rtol=0,
            atol=0,
        )


def test_fast_block_rotations_matches_reference_at_block_size_three():
    torch.manual_seed(5)
    theta = torch.randn(2, 32, 2, 4, 3, dtype=torch.float64) * 0.4
    torch.testing.assert_close(
        fast_block_rotations(theta, 3), _block_rotations(theta, 3), rtol=0, atol=1e-12
    )


# ---------------------------------------------------------------------------------------
# Prefix scan and the fused rotation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2, 8, 16, 64, 128, 512])
def test_prefix_scan_is_invariant_to_scan_chunk(chunk_size: int):
    """
    Lowering the scan chunk is a latency/arithmetic trade, never a numerical one.

    This is what licenses tuning ``_ROTATION_SCAN_CHUNK`` freely: the chunk only decides how the
    associative product is bracketed.
    """
    torch.manual_seed(6)
    rot = _block_rotations(torch.randn(2, 129, 1, 4, 3, dtype=torch.float64) * 0.2, 3)

    expected = _cumulative_block_rotation(rot, chunk_size=64)
    actual = fast_cumulative_block_rotation(rot, chunk_size=chunk_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_prefix_scan_equals_the_naive_ordered_product():
    """Guard the ordering convention itself: ``Q_t = R_t R_{t-1} ... R_1``, newest on the left."""
    torch.manual_seed(7)
    rot = _block_rotations(torch.randn(1, 17, 1, 2, 3, dtype=torch.float64) * 0.3, 3)

    scanned = fast_cumulative_block_rotation(rot, chunk_size=4)

    running = rot[:, 0]
    torch.testing.assert_close(scanned[:, 0], running, rtol=0, atol=1e-12)
    for t in range(1, rot.shape[1]):
        running = rot[:, t] @ running
        torch.testing.assert_close(scanned[:, t], running, rtol=0, atol=1e-11)


def test_fused_rotation_matches_two_separate_calls():
    """Concatenating B and C is a launch-count optimisation, not a change of contraction."""
    torch.manual_seed(8)
    B = torch.randn(2, 12, 2, 1, 12, dtype=torch.float64)
    C = torch.randn(2, 12, 2, 1, 12, dtype=torch.float64)
    rot = fast_cumulative_block_rotation(
        _block_rotations(torch.randn(2, 12, 2, 4, 3, dtype=torch.float64) * 0.3, 3)
    )

    fused_b, fused_c = _rotate_bc_fused(B, C, rot)

    torch.testing.assert_close(fused_b, _rotate_bc_blocks(B, rot), rtol=0, atol=1e-13)
    torch.testing.assert_close(fused_c, _rotate_bc_blocks(C, rot), rtol=0, atol=1e-13)


def test_fused_rotation_preserves_the_relative_transfer_identity():
    """
    The whole factorization rests on ``C~_t^T B~_s == C_t^T (R_t ... R_{s+1}) B_s``.

    Asserting it directly localizes a convention error to this one line, rather than letting it
    surface as a slightly-wrong loss five subsystems downstream.
    """
    torch.manual_seed(9)
    T, b, n_blocks = 9, 3, 2
    d_state = b * n_blocks
    B = torch.randn(1, T, 1, 1, d_state, dtype=torch.float64)
    C = torch.randn(1, T, 1, 1, d_state, dtype=torch.float64)
    per_step = fast_block_rotations(torch.randn(1, T, 1, n_blocks, 3, dtype=torch.float64), 3)

    rot_b, rot_c = _rotate_bc_fused(B, C, fast_cumulative_block_rotation(per_step, chunk_size=4))

    t, s = 7, 2
    intervening = per_step[0, s + 1, 0]
    for r in range(s + 2, t + 1):
        intervening = per_step[0, r, 0] @ intervening

    blocks_b = B[0, s, 0, 0].reshape(n_blocks, b)
    blocks_c = C[0, t, 0, 0].reshape(n_blocks, b)
    expected = torch.einsum("ki,kij,kj->", blocks_c, intervening, blocks_b)
    actual = (rot_c[0, t, 0, 0] * rot_b[0, s, 0, 0]).sum()

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


# ---------------------------------------------------------------------------------------
# End-to-end parity against the untouched official adapter (GPU)
# ---------------------------------------------------------------------------------------


def _inputs(
    *,
    batch: int = 2,
    seq_len: int = 40,
    n_heads: int = 4,
    head_dim: int = 8,
    n_groups: int = 1,
    d_state: int = 12,
    block_size: int = 3,
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
        B=r(batch, seq_len, n_groups, 1, d_state),
        C=r(batch, seq_len, n_groups, 1, d_state),
        dt=torch.rand(batch, seq_len, n_heads, device=device) * 0.1 + 0.01,
        A=-torch.rand(n_heads, device=device) - 0.5,
        lam=torch.rand(batch, seq_len, n_heads, device=device),
        theta=r(batch, seq_len, n_groups, n_blocks, angles),
    )


def _assert_matches(actual, expected, *, rms: float, peak: float, msg: str = ""):
    actual, expected = actual.float(), expected.float()
    diff = actual - expected
    rel_rms = (diff.pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()).item()
    rel_peak = (diff.abs().max() / expected.abs().max()).item()
    assert rel_rms < rms, f"{msg} relative RMS {rel_rms:.5f} >= {rms}"
    assert rel_peak < peak, f"{msg} relative peak {rel_peak:.5f} >= {peak}"


FAST_CONFIGS = [
    pytest.param(dict(d_state=12, block_size=3, n_groups=1), id="b3-g1"),
    pytest.param(dict(d_state=24, block_size=3, n_groups=2, seq_len=96), id="b3-g2-t96"),
    pytest.param(dict(d_state=16, block_size=2, n_groups=1), id="b2-g1"),
    pytest.param(dict(d_state=16, block_size=4, n_groups=1), id="b4-g1"),
]


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("cfg", FAST_CONFIGS)
def test_fast_matches_official_forward(cfg):
    """Same kernel, same inputs, faster rotation -- the output must not move."""
    kwargs = _inputs(**cfg)
    block_size = cfg["block_size"]
    heads_per_group = 4 // cfg["n_groups"]

    expected = mamba3_ssd_official(**kwargs, heads_per_group=heads_per_group, block_size=block_size)
    actual = mamba3_ssd_fast(
        **kwargs, heads_per_group=heads_per_group, block_size=block_size, selective_fp32=False
    )
    _assert_matches(actual, expected, rms=2e-3, peak=5e-3, msg=f"forward b={block_size}")


@requires_gpu
@requires_official_mamba3
@pytest.mark.parametrize("cfg", FAST_CONFIGS)
def test_fast_matches_official_backward(cfg):
    """Rodrigues replaces a ``matrix_exp`` backward; the gradients it produces must agree."""
    kwargs = _inputs(**cfg)
    block_size = cfg["block_size"]
    heads_per_group = 4 // cfg["n_groups"]
    torch.manual_seed(99)
    grad_out = torch.randn(kwargs["x"].shape, device="cuda")

    grads = {}
    for name, fn in (("official", mamba3_ssd_official), ("fast", mamba3_ssd_fast)):
        args = {k: v.clone().requires_grad_(True) for k, v in kwargs.items()}
        extra = {} if name == "official" else {"selective_fp32": False}
        out = fn(**args, heads_per_group=heads_per_group, block_size=block_size, **extra)
        (out.float() * grad_out).sum().backward()
        grads[name] = {k: v.grad for k, v in args.items()}

    for key, expected in grads["official"].items():
        actual = grads["fast"][key]
        assert actual is not None, f"fast path produced no gradient for {key}"
        _assert_matches(actual, expected, rms=5e-3, peak=2e-2, msg=f"grad {key} b={block_size}")


@requires_gpu
@requires_official_mamba3
def test_selective_fp32_stays_within_the_kernels_own_bf16_error():
    """
    Dropping the float32 floor on the *application* of the rotation must not cost more than the
    bf16 rounding ``mamba3_siso_combined`` performs on the very next line anyway.
    """
    kwargs = _inputs(d_state=24, block_size=3, n_groups=1, seq_len=128)

    expected = mamba3_ssd_official(**kwargs, heads_per_group=4, block_size=3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = mamba3_ssd_fast(**kwargs, heads_per_group=4, block_size=3, selective_fp32=True)

    _assert_matches(actual, expected, rms=2e-2, peak=3e-2, msg="selective fp32")


@requires_gpu
@requires_official_mamba3
def test_rotation_scan_chunk_does_not_change_the_output():
    kwargs = _inputs(d_state=12, block_size=3, seq_len=65)

    outputs = [
        mamba3_ssd_fast(
            **kwargs,
            heads_per_group=4,
            block_size=3,
            selective_fp32=False,
            rotation_scan_chunk=chunk,
        )
        for chunk in (2, 8, 64, 256)
    ]
    for other in outputs[1:]:
        _assert_matches(other, outputs[0], rms=1e-4, peak=1e-3, msg="scan chunk")


@requires_gpu
@requires_official_mamba3
def test_dispatch_prefers_the_fast_rotation_by_default():
    """
    A flag that never reaches the kernel is the failure mode this whole path exists to avoid, so
    assert the routing directly rather than trusting the default.
    """
    kwargs = _inputs(d_state=12, block_size=3)

    dispatched = dispatch_mamba3_ssd(
        **kwargs, heads_per_group=4, block_size=3, prefer_official_kernel=True
    )
    direct = mamba3_ssd_fast(**kwargs, heads_per_group=4, block_size=3)
    assert torch.equal(dispatched, direct), "dispatch did not take the fast-rotation path"


@requires_gpu
@requires_official_mamba3
def test_dispatch_can_still_reach_the_untouched_official_path():
    """``prefer_fast_rotation=False`` must land on ``mamba3_ssd_official`` exactly."""
    kwargs = _inputs(d_state=12, block_size=3)

    dispatched = dispatch_mamba3_ssd(
        **kwargs,
        heads_per_group=4,
        block_size=3,
        prefer_official_kernel=True,
        prefer_fast_rotation=False,
    )
    direct = mamba3_ssd_official(**kwargs, heads_per_group=4, block_size=3)
    assert torch.equal(dispatched, direct), "prefer_fast_rotation=False did not reach official"
