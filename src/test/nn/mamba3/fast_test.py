"""
Tests for :mod:`olmo_core.nn.mamba3.mamba3_ssd_fast`.

The claim under test is that the fast path computes the *same function* as
:mod:`mamba3_ssd_official`, only quicker -- so almost everything here is a parity assertion
against the untouched official path or against ``torch.matrix_exp``. The one place that is not a
parity check is the small-angle regime, which is where ``theta_proj`` initialises and where a
naive Rodrigues implementation produces NaN gradients on the first training step.
"""

import importlib

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
    _ROTATION_SCAN_IMPL,
    _adaptive_scan_chunk,
    _angles_to_quaternion,
    _fast_rotate_bc_pair,
    _quaternion_pointwise_combine,
    _quaternion_to_matrix,
    _rodrigues_so3,
    _rotate_bc_fused,
    _so3_affine_combine,
    _so3_pointwise_combine,
    associative_autograd_cumulative_block_rotation,
    associative_cumulative_block_rotation,
    fast_block_rotations,
    fast_cumulative_block_rotation,
    fast_mamba3_is_available,
    mamba3_ssd_fast,
    quaternion_cumulative_block_rotation,
)
from olmo_core.nn.mamba3.mamba3_ssd_official import mamba3_ssd_official
from olmo_core.testing import requires_gpu

requires_official_mamba3 = pytest.mark.skipif(
    not fast_mamba3_is_available(),
    reason="the official mamba-ssm Mamba-3 SISO kernel is not installed",
)

# The package `__init__` re-exports the `mamba3_ssd_fast` *function* under the same name as the
# module that defines it, so `import ...mamba3_ssd_fast as m` binds the function. Go through
# `importlib` to reach the module itself, which is what the gate tests monkeypatch.
fast_mod = importlib.import_module("olmo_core.nn.mamba3.mamba3_ssd_fast")


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

    This is what licenses tuning the scan chunk freely: the chunk only decides how the
    associative product is bracketed.
    """
    torch.manual_seed(6)
    rot = _block_rotations(torch.randn(2, 129, 1, 4, 3, dtype=torch.float64) * 0.2, 3)

    expected = _cumulative_block_rotation(rot, chunk_size=64)
    actual = fast_cumulative_block_rotation(rot, chunk_size=chunk_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize(
    "seq_len, expected",
    [
        (128, 8),  # below the floor: 128 // 128 == 1, clamped up to the minimum
        (512, 8),  # 512 // 128 == 4, still clamped up to 8
        (1024, 8),  # 1024 // 128 == 8, exactly the floor
        (2048, 16),  # measured optimum at seq 2048
        (4096, 32),  # measured optimum at the production sequence length
        (8192, 64),  # 8192 // 128 == 64, exactly the ceiling
        (65536, 64),  # far above: clamped to the ceiling
        (1, 8),  # degenerate length still yields a valid, floored chunk
    ],
)
def test_adaptive_scan_chunk_tracks_sequence_length(seq_len: int, expected: int):
    """
    The scan chunk is a launch/arithmetic trade whose optimum grows with ``T``.

    A single fixed chunk is wrong at both ends: too small pays needless Hillis-Steele levels at
    long ``T`` (measured 13.9 ms at chunk 8 vs 11.1 ms at chunk 32 for the b=3 scan at T=4096),
    too large pays a long dependent product at short ``T``. This pins the ``~T/128`` rule and its
    clamp so a regression to a constant is caught. Numerics are unaffected -- that is guaranteed
    separately by ``test_prefix_scan_is_invariant_to_scan_chunk``.
    """
    assert _adaptive_scan_chunk(seq_len) == expected


def test_adaptive_scan_chunk_is_monotonic_non_decreasing():
    """Coarser batching at longer sequences is the whole point; it must never invert."""
    chunks = [_adaptive_scan_chunk(t) for t in (1, 128, 512, 1024, 2048, 4096, 8192, 16384)]
    assert chunks == sorted(chunks)


@pytest.mark.skipif(
    _ROTATION_SCAN_IMPL != "chunked",
    reason="spies on the chunked path, which every other MAMBA3_ROTATION_SCAN_IMPL bypasses",
)
def test_default_scan_uses_the_adaptive_chunk(monkeypatch):
    """
    The default path must actually consult the adaptive rule, not a pinned constant.

    Numerics are invariant to the chunk, so a value regression cannot be caught by an output
    assertion -- it has to be observed at the point the chunk is chosen. This spies on the one
    call that receives it.
    """
    import olmo_core.nn.mamba3.mamba3_ssd_api as api

    seen = {}
    original = api._cumulative_block_rotation

    def spy(rot, chunk_size=64):
        seen["chunk"] = chunk_size
        return original(rot, chunk_size=chunk_size)

    monkeypatch.setattr(api, "_cumulative_block_rotation", spy)

    rot = _block_rotations(torch.randn(1, 4096, 1, 4, 3, dtype=torch.float64) * 0.1, 3)
    fast_cumulative_block_rotation(rot)  # no chunk_size -> adaptive from seq_len

    assert seen["chunk"] == _adaptive_scan_chunk(4096) == 32


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


# ---------------------------------------------------------------------------------------
# associative_scan prefix product
# ---------------------------------------------------------------------------------------


def _as_leaves(rot: torch.Tensor):
    """Split a ``(..., 3, 3)`` rotation into the 9 elementwise leaves the combine consumes."""
    return tuple(rot[..., i, j] for i in range(3) for j in range(3))


def test_so3_pointwise_combine_matches_matmul():
    """
    The 9-leaf form must be exactly ``b @ a`` -- newest on the left.

    This is the whole reason the scan can use ``combine_mode="pointwise"``: a 3x3 product written
    over 9 separate tensors is elementwise, so Inductor can emit a real ``tl.associative_scan``
    instead of the generic fallback. Getting the operand order backwards here would silently
    reverse the rotation, which no shape or dtype check would catch.
    """
    torch.manual_seed(10)
    a = _block_rotations(torch.randn(2, 5, 1, 4, 3, dtype=torch.float64) * 0.4, 3)
    b = _block_rotations(torch.randn(2, 5, 1, 4, 3, dtype=torch.float64) * 0.4, 3)

    combined = _so3_pointwise_combine(_as_leaves(a), _as_leaves(b))
    got = torch.stack(combined, dim=-1).unflatten(-1, (3, 3))

    torch.testing.assert_close(got, b @ a, rtol=0, atol=1e-13)


@pytest.mark.parametrize("seq_len", [1, 2, 17, 64, 129])
def test_associative_scan_matches_cumulative_block_rotation(seq_len: int):
    """The associative_scan path must compute the same prefix product as the chunked one."""
    torch.manual_seed(13)
    rot = _block_rotations(torch.randn(2, seq_len, 1, 4, 3, dtype=torch.float64) * 0.3, 3)

    expected = _cumulative_block_rotation(rot, chunk_size=8)
    actual = associative_cumulative_block_rotation(rot)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_associative_scan_output_stays_in_so3():
    """
    Orthogonality is the property the whole NC^1 claim rests on.

    A scan that drifts off ``SO(3)`` would not crash; it would quietly weaken the b=3 arm, so this
    asserts the group membership directly rather than trusting the parity check above.
    """
    torch.manual_seed(12)
    rot = _block_rotations(torch.randn(2, 64, 1, 4, 3, dtype=torch.float64) * 0.5, 3)

    q = associative_cumulative_block_rotation(rot)

    eye = torch.eye(3, dtype=q.dtype).expand_as(q)
    torch.testing.assert_close(q @ q.transpose(-1, -2), eye, rtol=0, atol=1e-10)
    torch.testing.assert_close(
        torch.linalg.det(q), torch.ones_like(q[..., 0, 0]), rtol=0, atol=1e-10
    )


def test_associative_scan_gradients_match_the_chunked_path():
    """
    Backward parity, not just forward.

    ``associative_scan`` grew autograd separately from its forward, so the gradient is the part
    most likely to be subtly wrong. The weighting below is deliberately non-symmetric so the
    gradient actually probes the ordering of the product rather than cancelling it out.
    """
    torch.manual_seed(11)
    theta = torch.randn(2, 24, 1, 4, 3, dtype=torch.float64) * 0.3

    def grad_through(scan_fn):
        t = theta.clone().requires_grad_(True)
        out = scan_fn(fast_block_rotations(t, 3))
        weight = torch.arange(out.numel(), dtype=out.dtype).reshape(out.shape) * 1e-3
        (out * weight).sum().backward()
        assert t.grad is not None
        return t.grad

    expected = grad_through(lambda r: _cumulative_block_rotation(r, chunk_size=8))
    actual = grad_through(associative_cumulative_block_rotation)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_associative_scan_refuses_block_sizes_other_than_three():
    """
    The 9-leaf combine is written out for 3x3 only.

    Slicing a 4x4 rotation down to its top-left 3x3 would still return orthogonal-looking output
    and would not crash -- it would just be the wrong rotation. Refuse loudly instead.
    """
    rot = _block_rotations(torch.randn(1, 8, 1, 2, 6, dtype=torch.float64) * 0.3, 4)

    with pytest.raises(ValueError, match="block_size 3"):
        associative_cumulative_block_rotation(rot)


def test_block_size_four_still_matches_the_chunked_product():
    """b=4 must keep the chunked path and stay correct in *either* MAMBA3_ROTATION_SCAN_IMPL mode."""
    torch.manual_seed(14)
    rot = _block_rotations(torch.randn(1, 33, 1, 2, 6, dtype=torch.float64) * 0.3, 4)

    torch.testing.assert_close(
        fast_cumulative_block_rotation(rot),
        _cumulative_block_rotation(rot, chunk_size=8),
        rtol=0,
        atol=1e-11,
    )


# ---------------------------------------------------------------------------------------
# associative_scan wrapped in an analytic-backward autograd.Function
# ---------------------------------------------------------------------------------------


def _naive_ordered_product(rot: torch.Tensor) -> torch.Tensor:
    """``Q_t = R_t R_{t-1} ... R_1`` as a Python loop -- the ordering oracle, autograd-friendly."""
    running = rot[:, 0]
    out = [running]
    for t in range(1, rot.shape[1]):
        running = rot[:, t] @ running
        out.append(running)
    return torch.stack(out, dim=1)


def _grad_through_scan(scan_fn, theta: torch.Tensor, *, weight_seed: int) -> torch.Tensor:
    """
    ``d/dtheta`` of a scalar built from a scan's output, under a fixed non-symmetric weighting.

    The weight is drawn from its own generator rather than the global RNG so that every arm sees
    the *same* weight no matter how much randomness the arm itself consumes -- otherwise a
    gradient mismatch could be a weighting mismatch. It is dense random rather than uniform
    because a symmetric weighting lets the two orderings ``R_t ... R_1`` and ``R_1 ... R_t``
    produce the same gradient, which would make this test blind to the exact error it exists
    to catch.
    """
    t = theta.clone().requires_grad_(True)
    out = scan_fn(fast_block_rotations(t, 3))
    gen = torch.Generator().manual_seed(weight_seed)
    weight = torch.randn(out.shape, dtype=out.dtype, generator=gen)
    (out * weight).sum().backward()
    assert t.grad is not None
    return t.grad


@pytest.mark.parametrize("seq_len", [1, 2, 3, 17, 64, 129])
def test_autograd_scan_forward_matches_the_chunked_product(seq_len: int):
    """
    Forward must be bit-for-bit the same *function*, since only the backward is being replaced.

    Checked against both the chunked path and the naive ordered loop: the first catches a change
    in value, the second pins the ``newest on the left`` ordering independently of the path that
    also has to get it right.
    """
    torch.manual_seed(20)
    rot = _block_rotations(torch.randn(2, seq_len, 1, 4, 3, dtype=torch.float64) * 0.3, 3)

    actual = associative_autograd_cumulative_block_rotation(rot)

    torch.testing.assert_close(
        actual, _cumulative_block_rotation(rot, chunk_size=8), rtol=0, atol=1e-11
    )
    torch.testing.assert_close(actual, _naive_ordered_product(rot), rtol=0, atol=1e-11)


def test_autograd_scan_forward_matches_the_plain_associative_scan():
    """Wrapping the scan in an autograd.Function must not perturb the value it computes."""
    torch.manual_seed(21)
    rot = _block_rotations(torch.randn(2, 48, 1, 4, 3, dtype=torch.float64) * 0.3, 3)

    torch.testing.assert_close(
        associative_autograd_cumulative_block_rotation(rot),
        associative_cumulative_block_rotation(rot),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 3, 17, 64, 129])
def test_autograd_scan_gradients_match_the_chunked_path(seq_len: int):
    """
    The point of the exercise: the hand-written analytic backward must agree with autograd.

    ``associative_scan``'s own backward returns NaN on CUDA/fp32, so this path bypasses it with
    ``dL/dR_i = M_i Q_{i-1}^T``, ``M_i = G_i + R_{i+1}^T M_{i+1}``. That derivation is checked
    here against the gradient the chunked path's autograd produces for the same scalar.
    """
    torch.manual_seed(22)
    theta = torch.randn(2, seq_len, 1, 4, 3, dtype=torch.float64) * 0.3

    expected = _grad_through_scan(
        lambda r: _cumulative_block_rotation(r, chunk_size=8), theta, weight_seed=seq_len
    )
    actual = _grad_through_scan(
        associative_autograd_cumulative_block_rotation, theta, weight_seed=seq_len
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_autograd_scan_gradients_match_the_naive_ordered_product():
    """
    Second opinion on the backward, against the loop rather than the chunked path.

    The chunked path and the analytic backward could in principle share an ordering mistake --
    both were written against the same convention. The naive loop is the independent statement
    of that convention, so agreeing with *its* autograd is the stronger claim.
    """
    torch.manual_seed(23)
    theta = torch.randn(1, 13, 1, 2, 3, dtype=torch.float64) * 0.4

    expected = _grad_through_scan(_naive_ordered_product, theta, weight_seed=5)
    actual = _grad_through_scan(
        associative_autograd_cumulative_block_rotation, theta, weight_seed=5
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_autograd_scan_passes_gradcheck():
    """
    Finite differences against the analytic backward, with no other implementation in the loop.

    ``gradcheck`` perturbs ``rot`` off the ``SO(3)`` manifold, which is exactly right here: the
    derivation uses only associativity of the matrix product and the transpose of a product, never
    orthogonality, so the backward has to be correct for arbitrary matrices too.
    """
    torch.manual_seed(24)
    rot = _block_rotations(torch.randn(1, 6, 1, 2, 3, dtype=torch.float64) * 0.3, 3)
    rot = rot.clone().requires_grad_(True)

    assert torch.autograd.gradcheck(
        associative_autograd_cumulative_block_rotation, (rot,), atol=1e-9, rtol=1e-7
    )


@pytest.mark.parametrize("scale", [0.0, 1e-4, 1e-8], ids=lambda s: f"scale{s}")
def test_autograd_scan_gradients_are_finite_at_tiny_angles(scale: float):
    """
    The init regime, which is where the NaN this replaces actually shows up.

    ``theta_proj`` starts at ``std * 0.1``, so every rotation is near-identity on step one. A
    backward that reconstructed ``Q_{i-1}`` by inverting ``Q_i`` would be at its worst here; the
    recurrence used instead contains no division at all, so tiny and exactly-zero angles are
    ordinary inputs rather than a special case.
    """
    torch.manual_seed(25)
    theta = torch.randn(2, 32, 1, 4, 3, dtype=torch.float64) * scale

    actual = _grad_through_scan(
        associative_autograd_cumulative_block_rotation, theta, weight_seed=7
    )
    expected = _grad_through_scan(
        lambda r: _cumulative_block_rotation(r, chunk_size=8), theta, weight_seed=7
    )

    assert torch.isfinite(actual).all(), f"non-finite gradient at scale {scale}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def test_autograd_scan_gradient_is_finite_and_accurate_in_float32():
    """
    The production dtype for the prefix product is float32, not float64.

    The reported NaN is fp32-specific, so the finiteness claim has to be made in fp32 too. The
    accuracy claim is made against a float64 run of the *same* scan rather than against a fixed
    tolerance, and it is only required to be no worse than what the chunked path loses over the
    same 128-step product -- fp32 accumulation is the floor here, and the point is that this path
    does not sit below it.

    Note the objective has to be weighted: ``sum(Q * Q)`` is ``tr(Q Q^T) == 3`` for any rotation,
    so its gradient is identically zero and it would call any implementation finite.
    """
    torch.manual_seed(26)
    base = torch.randn(2, 128, 1, 4, 3) * 0.1
    gen = torch.Generator().manual_seed(27)
    weight = torch.randn(2, 128, 1, 4, 3, 3, generator=gen)

    def grad(scan_fn, dtype):
        theta = base.clone().to(dtype).requires_grad_(True)
        out = scan_fn(fast_block_rotations(theta, 3))
        (out * weight.to(dtype)).sum().backward()
        assert theta.grad is not None
        return theta.grad

    truth = grad(associative_autograd_cumulative_block_rotation, torch.float64)
    actual = grad(associative_autograd_cumulative_block_rotation, torch.float32)
    chunked = grad(lambda r: _cumulative_block_rotation(r, chunk_size=8), torch.float32)

    assert torch.isfinite(actual).all()
    assert (actual.double() - truth).abs().max() <= 4 * (chunked.double() - truth).abs().max()


def test_autograd_scan_compiles_without_a_graph_break():
    """
    A graph break here would silently cost the entire speedup, and nothing else would notice.

    The only reason to route through ``associative_scan`` at all is that Inductor lowers it to one
    fused ``tl.associative_scan``; if Dynamo cannot trace the ``autograd.Function`` it falls back
    to eager, where the scan is a Python tree reduction and strictly *slower* than the chunked
    form it replaced. The output would still be correct, so every other test in this file would
    still pass. ``fullgraph=True`` turns that silent regression into a failure.

    ``backend="eager"`` keeps this to Dynamo tracing -- which is where a break would happen --
    without paying for Inductor codegen, since the generated kernel is not what is under test.
    """
    torch.manual_seed(33)
    base = torch.randn(1, 16, 1, 2, 3) * 0.1
    gen = torch.Generator().manual_seed(34)
    weight = torch.randn(1, 16, 1, 2, 3, 3, generator=gen)

    def step(t):
        out = associative_autograd_cumulative_block_rotation(fast_block_rotations(t, 3))
        return (out * weight).sum()

    def grad_of(fn):
        theta = base.clone().requires_grad_(True)
        fn(theta).backward()
        assert theta.grad is not None
        return theta.grad

    torch._dynamo.reset()
    compiled = grad_of(torch.compile(step, fullgraph=True, backend="eager"))

    assert torch.isfinite(compiled).all()
    torch.testing.assert_close(compiled, grad_of(step), rtol=0, atol=1e-5)


def test_autograd_scan_output_stays_in_so3():
    """Orthogonality with determinant +1 -- the property the NC^1 claim rests on."""
    torch.manual_seed(27)
    rot = _block_rotations(torch.randn(2, 64, 1, 4, 3, dtype=torch.float64) * 0.5, 3)

    q = associative_autograd_cumulative_block_rotation(rot)

    eye = torch.eye(3, dtype=q.dtype).expand_as(q)
    torch.testing.assert_close(q @ q.transpose(-1, -2), eye, rtol=0, atol=1e-10)
    torch.testing.assert_close(
        torch.linalg.det(q), torch.ones_like(q[..., 0, 0]), rtol=0, atol=1e-10
    )


def test_autograd_scan_refuses_block_sizes_other_than_three():
    """
    A 4x4 rotation must not be silently truncated to its top-left 3x3.

    That truncation returns orthogonal-looking output and passes every shape and dtype check, so
    the only thing standing between it and a quietly wrong ``b=4`` model is this refusal.
    """
    rot = _block_rotations(torch.randn(1, 8, 1, 2, 6, dtype=torch.float64) * 0.3, 4)

    with pytest.raises(ValueError, match="block_size 3"):
        associative_autograd_cumulative_block_rotation(rot)


def test_so3_affine_combine_matches_the_affine_composition():
    """
    ``(A1, b1) . (A2, b2) = (A1 A2, A1 b2 + b1)`` -- the backward's associative operator.

    Written over 18 elementwise leaves for the same reason the forward combine is written over 9:
    that is what lets Inductor emit a real ``tl.associative_scan``. Getting the operand order
    backwards here reverses the reverse-recurrence and produces a plausible but wrong gradient.
    """
    torch.manual_seed(28)
    shape = (2, 5, 1, 4, 3, 3)
    a1 = torch.randn(*shape, dtype=torch.float64)
    b1 = torch.randn(*shape, dtype=torch.float64)
    a2 = torch.randn(*shape, dtype=torch.float64)
    b2 = torch.randn(*shape, dtype=torch.float64)

    def leaves(a, b):
        return _as_leaves(a) + _as_leaves(b)

    # `_so3_affine_combine(older, newer)` applies `newer` on the outside, matching the forward's
    # `_so3_pointwise_combine(a, b) -> b @ a`.
    combined = _so3_affine_combine(leaves(a2, b2), leaves(a1, b1))
    got_a = torch.stack(combined[:9], dim=-1).unflatten(-1, (3, 3))
    got_b = torch.stack(combined[9:], dim=-1).unflatten(-1, (3, 3))

    torch.testing.assert_close(got_a, a1 @ a2, rtol=0, atol=1e-13)
    torch.testing.assert_close(got_b, a1 @ b2 + b1, rtol=0, atol=1e-13)


@pytest.mark.parametrize("impl", ["chunked", "associative", "associative_autograd"])
def test_scan_impl_gate_routes_block_size_three(monkeypatch, impl: str):
    """
    Every gate value must reach a different implementation and agree on the answer.

    The gate is the revert switch, so a value that silently lands on the wrong implementation is
    the failure that would make an incident unrecoverable in the time available.
    """
    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", impl)
    torch.manual_seed(29)
    rot = _block_rotations(torch.randn(2, 40, 1, 4, 3, dtype=torch.float64) * 0.3, 3)

    torch.testing.assert_close(
        fast_cumulative_block_rotation(rot),
        _cumulative_block_rotation(rot, chunk_size=8),
        rtol=0,
        atol=1e-11,
    )


@pytest.mark.parametrize("impl", ["chunked", "associative", "associative_autograd"])
def test_block_size_four_falls_back_to_chunked_under_every_gate(monkeypatch, impl: str):
    """
    ``b=4`` has no 3x3 combine to use, so every gate value must route it to the chunked path.

    Both forward *and* backward: a fallback that only covered the forward would leave ``b=4``
    differentiating through the wrong thing.
    """
    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", impl)
    torch.manual_seed(30)
    base = torch.randn(1, 33, 1, 2, 6, dtype=torch.float64) * 0.3

    grads = []
    for scan_fn in (fast_cumulative_block_rotation, lambda r: _cumulative_block_rotation(r, 8)):
        theta = base.clone().requires_grad_(True)
        out = scan_fn(_block_rotations(theta, 4))
        gen = torch.Generator().manual_seed(31)
        (out * torch.randn(out.shape, dtype=out.dtype, generator=gen)).sum().backward()
        assert theta.grad is not None
        grads.append((out.detach(), theta.grad))

    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=0, atol=1e-11)
    torch.testing.assert_close(grads[0][1], grads[1][1], rtol=0, atol=1e-11)


def test_autograd_scan_returns_no_gradient_when_none_is_wanted():
    """An input that does not require grad must not force one to be materialised."""
    torch.manual_seed(32)
    rot = _block_rotations(torch.randn(1, 9, 1, 2, 3, dtype=torch.float64) * 0.3, 3)

    out = associative_autograd_cumulative_block_rotation(rot)

    assert not out.requires_grad


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
# quaternion prefix product (MAMBA3_ROTATION_SCAN_IMPL=quaternion)
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale", [0.0, 1e-8, 1e-4, 0.01, 1.0, 3.14159, 6.0], ids=lambda s: f"scale{s}"
)
def test_quaternion_roundtrip_matches_fast_block_rotations(scale: float):
    """
    Pin the quaternion<->matrix convention against the matrix path, at and near zero included.

    This is the test that fixes the axis/sign mapping: whatever makes
    ``_quaternion_to_matrix(_angles_to_quaternion(theta))`` reproduce ``fast_block_rotations``
    (Rodrigues, i.e. ``matrix_exp`` of the skew) is the correct convention, so the round trip is
    asserted directly rather than a hand-derived sign being trusted.
    """
    torch.manual_seed(40)
    theta = torch.randn(2, 64, 1, 8, 3, dtype=torch.float64) * scale

    actual = _quaternion_to_matrix(_angles_to_quaternion(theta))
    expected = fast_block_rotations(theta, 3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e-4, 0.01, 1.0, 3.0, 6.0], ids=lambda s: f"scale{s}")
def test_angles_to_quaternion_is_unit_norm(scale: float):
    """Every per-step quaternion must live on the unit sphere, including the small-angle Taylor."""
    torch.manual_seed(41)
    theta = torch.randn(3, 16, 1, 4, 3, dtype=torch.float64) * scale

    norm_sq = _angles_to_quaternion(theta).pow(2).sum(-1)

    torch.testing.assert_close(norm_sq, torch.ones_like(norm_sq), rtol=0, atol=1e-12)


def test_quaternion_combine_matches_matrix_compose():
    """
    The Hamilton product must compose rotations newest-left, matching the matrix path's ``b @ a``.

    ``_quaternion_pointwise_combine(a, b)`` has to be the quaternion for ``R(b) @ R(a)`` -- the
    newer operand ``b`` on the left -- so that the scan reproduces ``Q_t = R_t R_{t-1} ... R_1``.
    Getting the operand order backwards would silently reverse the rotation.
    """
    torch.manual_seed(42)
    qa = _angles_to_quaternion(torch.randn(2, 5, 1, 4, 3, dtype=torch.float64) * 0.4)
    qb = _angles_to_quaternion(torch.randn(2, 5, 1, 4, 3, dtype=torch.float64) * 0.4)

    a_leaves = tuple(qa[..., i] for i in range(4))
    b_leaves = tuple(qb[..., i] for i in range(4))
    combined = _quaternion_pointwise_combine(a_leaves, b_leaves)
    got = _quaternion_to_matrix(torch.stack(combined, dim=-1))

    expected = _quaternion_to_matrix(qb) @ _quaternion_to_matrix(qa)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("scale", [0.01, 1.0, 3.0])
def test_quaternion_to_matrix_is_in_so3(scale: float):
    """``_quaternion_to_matrix`` of a unit quaternion is orthogonal with determinant +1."""
    torch.manual_seed(43)
    q = _angles_to_quaternion(torch.randn(64, 3, dtype=torch.float64) * scale)

    rot = _quaternion_to_matrix(q)

    eye = torch.eye(3, dtype=torch.float64).expand_as(rot)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), eye, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        torch.linalg.det(rot), torch.ones(64, dtype=torch.float64), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("seq_len", [1, 2, 17, 64, 129])
def test_quaternion_cumulative_matches_chunked(seq_len: int):
    """The quaternion prefix product must compute the same rotation as the chunked matrix path."""
    torch.manual_seed(44)
    theta = torch.randn(2, seq_len, 1, 4, 3, dtype=torch.float64) * 0.3

    expected = _cumulative_block_rotation(_block_rotations(theta, 3), chunk_size=8)
    actual = quaternion_cumulative_block_rotation(theta, 3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-11)


def _grad_of_theta_build(build, theta: torch.Tensor, *, weight_seed: int) -> torch.Tensor:
    """``d/dtheta`` of a non-symmetric weighting of a ``theta -> (..., 3, 3)`` builder's output."""
    t = theta.clone().requires_grad_(True)
    out = build(t)
    gen = torch.Generator().manual_seed(weight_seed)
    weight = torch.randn(out.shape, dtype=out.dtype, generator=gen)
    (out * weight).sum().backward()
    assert t.grad is not None
    return t.grad


def test_quaternion_cumulative_gradients_match_the_chunked_path():
    """
    Backward parity against the chunked path, under a dense non-symmetric weighting.

    The forward is quaternion-compose + quaternion->matrix, all standard differentiable ops, so
    ``associative_scan``'s own autograd is expected to suffice. The weighting is dense random so
    the gradient probes the ordering of the product rather than cancelling it out.
    """
    torch.manual_seed(45)
    theta = torch.randn(2, 24, 1, 4, 3, dtype=torch.float64) * 0.3

    expected = _grad_of_theta_build(
        lambda t: _cumulative_block_rotation(fast_block_rotations(t, 3), chunk_size=8),
        theta,
        weight_seed=46,
    )
    actual = _grad_of_theta_build(
        lambda t: quaternion_cumulative_block_rotation(t, 3), theta, weight_seed=46
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("scale", [0.0, 1e-4, 0.1], ids=lambda s: f"scale{s}")
def test_quaternion_cumulative_gradients_are_finite_at_init_scale(scale: float):
    """
    The init regime, where ``theta_proj`` starts (``std * 0.1``) and a naive small-angle handling
    produces first-step NaN gradients.

    ``_angles_to_quaternion`` clamps before the sqrt and Taylors both half-angle coefficients, so
    tiny and exactly-zero angles are ordinary inputs. Both finiteness and parity with the chunked
    path are asserted.
    """
    torch.manual_seed(47)
    theta = torch.randn(2, 32, 1, 4, 3, dtype=torch.float64) * scale

    actual = _grad_of_theta_build(
        lambda t: quaternion_cumulative_block_rotation(t, 3), theta, weight_seed=48
    )
    expected = _grad_of_theta_build(
        lambda t: _cumulative_block_rotation(fast_block_rotations(t, 3), chunk_size=8),
        theta,
        weight_seed=48,
    )

    assert torch.isfinite(actual).all(), f"non-finite gradient at scale {scale}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_quaternion_cumulative_refuses_block_sizes_other_than_three():
    """
    The Hamilton combine and the quaternion<->matrix maps are written for ``SO(3)`` only.

    A larger block silently handled would truncate the rotation, so refuse it outright -- the same
    guard the matrix scans carry -- and let the dispatch route ``b != 3`` to the chunked path.
    """
    theta = torch.randn(1, 8, 1, 2, 6, dtype=torch.float64) * 0.3  # b=4 angles per block

    with pytest.raises(ValueError, match="block_size 3"):
        quaternion_cumulative_block_rotation(theta, 4)


def test_quaternion_gate_falls_back_to_chunked_for_block_size_four(monkeypatch):
    """
    Under ``MAMBA3_ROTATION_SCAN_IMPL=quaternion`` the dispatch must still route ``b=4`` to the
    chunked path, bit-for-bit identical to any other gate value.

    An earlier rotation optimisation truncated a 4x4 to its top-left 3x3 and only the b=4 mixer
    tests caught it; this pins the fallback at the dispatch itself.
    """
    torch.manual_seed(49)
    B = torch.randn(1, 33, 1, 1, 16, dtype=torch.float64)
    C = torch.randn(1, 33, 1, 1, 16, dtype=torch.float64)
    theta = torch.randn(1, 33, 1, 4, 6, dtype=torch.float64) * 0.3  # d_state=16, n_blocks=4, b=4

    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", "quaternion")
    b_quat, c_quat = _fast_rotate_bc_pair(B, C, theta, 4, None)
    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", "chunked")
    b_chunked, c_chunked = _fast_rotate_bc_pair(B, C, theta, 4, None)

    torch.testing.assert_close(b_quat, b_chunked, rtol=0, atol=0)
    torch.testing.assert_close(c_quat, c_chunked, rtol=0, atol=0)


def test_quaternion_gate_matches_chunked_for_block_size_three(monkeypatch):
    """
    The whole point of the gate: at ``b=3`` the quaternion dispatch must apply the *same* rotation
    to ``B`` and ``C`` as the chunked default, through the shared ``_rotate_bc_fused``.

    ``_fast_rotate_bc_pair`` runs the prefix product in float32 (``theta.float()``) whatever
    ``B``/``C`` carry, and the quaternion scan is a different arithmetic path from the chunked
    matrix product, so agreement here is at the float32 floor (~1e-6) rather than the 1e-11 that
    ``test_quaternion_cumulative_matches_chunked`` pins in float64. A convention/ordering error
    would move the output by O(1) and is still caught comfortably.
    """
    torch.manual_seed(52)
    B = torch.randn(2, 40, 1, 1, 12, dtype=torch.float64)
    C = torch.randn(2, 40, 1, 1, 12, dtype=torch.float64)
    theta = torch.randn(2, 40, 1, 4, 3, dtype=torch.float64) * 0.3  # d_state=12, n_blocks=4, b=3

    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", "quaternion")
    b_quat, c_quat = _fast_rotate_bc_pair(B, C, theta, 3, None)
    monkeypatch.setattr(fast_mod, "_ROTATION_SCAN_IMPL", "chunked")
    b_chunked, c_chunked = _fast_rotate_bc_pair(B, C, theta, 3, None)

    torch.testing.assert_close(b_quat, b_chunked, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c_quat, c_chunked, rtol=1e-4, atol=1e-5)


def test_quaternion_scan_compiles_without_a_graph_break():
    """
    A graph break here silently costs the speedup and nothing else notices -- ``fullgraph=True``
    turns that into a failure.

    Mirrors ``test_autograd_scan_compiles_without_a_graph_break``: ``backend="eager"`` keeps this
    to Dynamo tracing (where a break would happen) without paying for Inductor codegen. On CPU the
    scan runs ``combine_mode="generic"``; ``pointwise`` codegen is CUDA-only and untestable here.
    """
    torch.manual_seed(50)
    base = torch.randn(1, 16, 1, 2, 3) * 0.1
    gen = torch.Generator().manual_seed(51)
    weight = torch.randn(1, 16, 1, 2, 3, 3, generator=gen)

    def step(t):
        out = quaternion_cumulative_block_rotation(t, 3)
        return (out * weight).sum()

    def grad_of(fn):
        theta = base.clone().requires_grad_(True)
        fn(theta).backward()
        assert theta.grad is not None
        return theta.grad

    torch._dynamo.reset()
    compiled = grad_of(torch.compile(step, fullgraph=True, backend="eager"))

    assert torch.isfinite(compiled).all()
    torch.testing.assert_close(compiled, grad_of(step), rtol=0, atol=1e-5)


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
