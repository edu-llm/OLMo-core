"""
Tests for the ``b x b`` orthogonal block transitions of the Mamba-3 mixer.

These cover the transition *algebra* only -- no training. That separation is deliberate: if a
state-tracking run fails, these tests say whether the group structure is wrong or whether the
optimizer simply could not reach it.
"""

import math

import pytest
import torch

from olmo_core.nn.mamba3.mamba3_ssd_api import (
    _block_rotations,
    _cumulative_block_rotation,
    _rotate_bc,
    _rotate_bc_blocks,
    _skew_from_angles,
    mamba3_ssd_reference,
)

GOLDEN_RATIO = (1 + 5**0.5) / 2


def _angles_from_axis_angle(axis: list[float], angle: float) -> torch.Tensor:
    """
    Express an ``so(3)`` axis-angle rotation in this module's angle convention.

    :func:`_skew_from_angles` lays angles out as
    ``S = [[0, t01, t02], [-t01, 0, t12], [-t02, -t12, 0]]``, whereas the textbook axis-angle
    generator is ``K(w) = [[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]]``. Matching the two gives
    ``(t01, t02, t12) = (-w3, w2, -w1)``.
    """
    a = torch.tensor(axis, dtype=torch.float64)
    w = angle * a / a.norm()
    return torch.tensor([-w[2], w[1], -w[0]], dtype=torch.float64)


def _sequential_prefix_product(rot: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix product over dim 1 by an explicit loop, newest-left."""
    acc = rot[:, 0]
    out = [acc]
    for t in range(1, rot.shape[1]):
        acc = rot[:, t] @ acc
        out.append(acc)
    return torch.stack(out, dim=1)


def test_skew_from_angles_is_skew_symmetric():
    """The generator must live in ``so(b)``, otherwise ``matrix_exp`` leaves ``SO(b)``."""
    torch.manual_seed(0)
    for b in (2, 3, 4, 5, 8):
        skew = _skew_from_angles(torch.randn(3, b * (b - 1) // 2), b)
        assert skew.shape == (3, b, b)
        torch.testing.assert_close(skew, -skew.transpose(-1, -2))


def test_skew_from_angles_rejects_wrong_angle_count():
    with pytest.raises(ValueError, match="expected 3 angles"):
        _skew_from_angles(torch.randn(2, 4), 3)


@pytest.mark.parametrize("n_groups", [1, 2], ids=["g1", "g2"])
@pytest.mark.parametrize("mimo_rank", [1, 3], ids=["siso", "mimo3"])
@pytest.mark.parametrize("d_state", [4, 8], ids=["n4", "n8"])
def test_skew_exp_degenerates_to_rope_at_block_size_2(n_groups: int, mimo_rank: int, d_state: int):
    """
    At ``b == 2`` the blocked path must reproduce the legacy 2x2 RoPE path.

    This is the backward-compatibility gate for the whole change: ``exp`` of
    ``[[0, theta], [-theta, 0]]`` is ``R(-theta)``, and the transpose applied when rotating
    ``B``/``C`` turns it back into the ``R(+theta_cumulative)`` of :func:`_rotate_bc`. A sign
    error anywhere in that chain shows up here as an O(1) disagreement, not a small one.

    Tolerance: the two routes reach the same rotation by different arithmetic -- ``cos``/``sin``
    of an accumulated angle versus a chain of ``matrix_exp`` outputs multiplied together -- so
    they agree only to fp32 rounding accumulated over the sequence. Measured worst deviation is
    ~1e-5 on values of order 4, i.e. a few ULPs; a genuine convention error is O(1).
    """
    torch.manual_seed(0)
    batch, seq_len = 2, 13
    bc = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    theta = torch.randn(batch, seq_len, n_groups, d_state // 2)

    legacy = _rotate_bc(bc, torch.cumsum(theta, dim=1))
    blocked = _rotate_bc_blocks(
        bc, _cumulative_block_rotation(_block_rotations(theta.unsqueeze(-1), 2))
    )
    torch.testing.assert_close(blocked, legacy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_size", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("seq_len", [1, 7, 64, 129])
@pytest.mark.parametrize("chunk_size", [4, 64], ids=["chunk4", "chunk64"])
def test_cumulative_block_rotation_matches_sequential_product(
    block_size: int, seq_len: int, chunk_size: int
):
    """
    The chunked Hillis-Steele scan must equal a plain loop of matmuls.

    Non-power-of-two lengths (7, 129) exercise the identity padding, and ``chunk_size=4`` forces
    several scan levels at short sequence lengths where the default 64 would take the
    single-chunk shortcut.
    """
    torch.manual_seed(0)
    batch, n_groups, n_blocks = 2, 2, 3
    theta = torch.randn(batch, seq_len, n_groups, n_blocks, block_size * (block_size - 1) // 2)
    rot = _block_rotations(theta, block_size)

    scanned = _cumulative_block_rotation(rot, chunk_size=chunk_size)
    assert scanned.shape == rot.shape
    torch.testing.assert_close(scanned, _sequential_prefix_product(rot), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_size", [2, 3, 4, 5, 8])
def test_block_rotations_are_orthogonal(block_size: int):
    """
    Per-step and accumulated transitions must stay in ``SO(b)``.

    Orthogonality is load-bearing twice over: it is what makes ``Q_s^-1 = Q_s^T`` (so the
    factorization that leaves the scan untouched is valid at all), and it is what bounds the
    transition norm at exactly ``alpha < 1`` so the layer stays BIBO-stable.

    Drift of the accumulated product grows as ``O(T * eps)``, *not* ``O(sqrt(T) * eps)`` -- the
    rounding errors accumulate coherently rather than as a random walk. Measured at ``T=4096``
    in fp32: ``4.7e-4`` at ``b=2`` rising to ``1.4e-3`` at ``b=8``, i.e. a factor of 1-3 times
    ``T * eps``, growing slowly with the block size as the per-entry dot products lengthen. The
    same run in fp64 lands at ``4e-13``, confirming this is pure floating-point accumulation
    rather than an algorithmic defect. The chunked scan and a plain sequential loop drift
    identically, so the scan structure is not the source and re-chunking will not help; the
    lever, if drift ever matters, is periodic Newton-Schulz re-orthogonalization.

    The bound below is ``10 * T * eps``, roughly 3x headroom over the worst measured case while
    remaining orders of magnitude below the O(1) deviation a genuine bug would produce.
    """
    torch.manual_seed(0)
    seq_len = 4096
    theta = torch.randn(1, seq_len, 1, 2, block_size * (block_size - 1) // 2)
    rot = _block_rotations(theta, block_size)
    eye = torch.eye(block_size)

    torch.testing.assert_close(rot.transpose(-1, -2) @ rot, eye.expand_as(rot), rtol=0, atol=1e-5)
    # Rotations, not reflections: matrix_exp of a skew matrix always has det +1.
    torch.testing.assert_close(
        torch.linalg.det(rot), torch.ones_like(rot[..., 0, 0]), rtol=0, atol=1e-5
    )

    scanned = _cumulative_block_rotation(rot)
    drift = (scanned.transpose(-1, -2) @ scanned - eye).abs().max()
    budget = 10 * seq_len * torch.finfo(torch.float32).eps
    assert drift < budget, f"orthogonality drift {drift:.2e} exceeds {budget:.2e} at T={seq_len}"


@pytest.mark.parametrize("block_size", [2, 3, 4])
def test_transition_norm_is_contractive(block_size: int):
    """
    The composed transition ``alpha * Q`` must have spectral norm ``<= 1``.

    ``Q`` is orthogonal so it contributes exactly 1, leaving the decay ``alpha = exp(dt * A)``
    with ``A < 0`` and ``dt > 0`` to do the contracting. Widening the block must not trade this
    away.
    """
    torch.manual_seed(0)
    seq_len = 32
    theta = torch.randn(1, seq_len, 1, 2, block_size * (block_size - 1) // 2)
    rot = _block_rotations(theta, block_size)
    dt = torch.rand(1, seq_len, 1, 1) * 0.1 + 0.01
    alpha = torch.exp(dt * -torch.rand(1))

    transition = alpha.unsqueeze(-1) * rot
    norms = torch.linalg.matrix_norm(transition, ord=2)
    assert norms.max() <= 1.0 + 1e-5
    # And the accumulated transition over the window is contractive too.
    accumulated = _sequential_prefix_product(transition)
    assert torch.linalg.matrix_norm(accumulated, ord=2).max() <= 1.0 + 1e-5


def test_pi_rotations_are_representable():
    """
    Guard against a Cayley-transform regression.

    ``Q = (I - S)(I + S)^-1`` is a cheaper map onto (most of) ``SO(b)``, but it can never
    produce a rotation with a ``-1`` eigenvalue -- precisely the 15 order-2 elements of ``A_5``.
    A model parameterized that way provably cannot represent ``A_5`` and would fail state
    tracking with no visible symptom, so this asserts the ``matrix_exp`` route reaches an exact
    ``pi`` rotation and that such a rotation is indeed outside Cayley's image.
    """
    axis = [1.0, 2.0, -0.5]
    rot = _block_rotations(_angles_from_axis_angle(axis, math.pi), 3)

    # A pi rotation in SO(3) has trace 1 + 2*cos(pi) = -1 and squares to the identity.
    assert rot.trace().item() == pytest.approx(-1.0, abs=1e-6)
    torch.testing.assert_close(rot @ rot, torch.eye(3, dtype=rot.dtype), rtol=0, atol=1e-6)

    # It rotates a vector orthogonal to the axis to its negation.
    axis_t = torch.tensor(axis, dtype=torch.float64)
    perp = torch.linalg.cross(axis_t, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(rot @ perp, -perp, rtol=0, atol=1e-6)

    # I + Q is singular, which is exactly the condition Cayley cannot invert.
    assert torch.linalg.det(torch.eye(3, dtype=rot.dtype) + rot).abs().item() < 1e-9


def test_a5_generators_are_representable_and_close():
    """
    The parameterization must express ``A_5`` itself, not merely something non-abelian.

    Barrington's theorem makes the word problem of any non-solvable group NC^1-complete, and
    ``A_5`` is the smallest such group; ``SO(3)`` contains it as the icosahedral rotation group.
    This builds the standard ``(2, 3, 5)`` generating pair -- a 5-fold rotation about a vertex
    axis and a 2-fold rotation about an edge axis -- and closes the monoid under the same
    ``matmul`` the scan uses. Exactly 60 distinct elements is the certificate that the group is
    ``A_5`` and not a solvable subgroup.

    Run in float64: distinguishing 60 group elements by numerical equality needs more headroom
    than fp32 gives once words get long.
    """
    five_fold = _block_rotations(
        _angles_from_axis_angle([0.0, 1.0, GOLDEN_RATIO], 2 * math.pi / 5), 3
    )
    two_fold = _block_rotations(_angles_from_axis_angle([0.0, 0.0, 1.0], math.pi), 3)
    identity = torch.eye(3, dtype=torch.float64)

    def order_of(m: torch.Tensor) -> int:
        power = m.clone()
        for k in range(1, 16):
            if torch.allclose(power, identity, atol=1e-9):
                return k
            power = power @ m
        raise AssertionError("element has order > 15, so it is not an icosahedral rotation")

    # The (2, 3, 5) triangle-group presentation of A_5.
    assert order_of(five_fold) == 5
    assert order_of(two_fold) == 2
    assert order_of(five_fold @ two_fold) == 3

    elements = [identity]
    frontier = [identity]
    while frontier:
        discovered = []
        for elem in frontier:
            for gen in (five_fold, two_fold):
                candidate = gen @ elem
                if not any(torch.allclose(candidate, seen, atol=1e-7) for seen in elements):
                    elements.append(candidate)
                    discovered.append(candidate)
        frontier = discovered
        assert len(elements) <= 60, "closure exceeded |A_5|, generators are not icosahedral"

    assert len(elements) == 60


def _naive_block_matrix_scan(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int,
) -> torch.Tensor:
    """
    Independent oracle: apply the block rotation to the *state* instead of folding it into
    ``B``/``C``.

    This is the un-factorized model. Where :func:`mamba3_ssd_reference` pre-rotates ``B`` and
    ``C`` and then runs a scalar-transition scan, this materializes the per-step block-diagonal
    transition and runs ``h_t = alpha_t R_t h_{t-1} + gamma_t v_t + beta_t R_t v_{t-1}`` with
    unrotated ``B``/``C``. Note the trapezoidal ``v_{t-1}`` term is rotated too -- it enters the
    state one step late and so rides along with the same transition.
    """
    batch, seq_len, n_heads, head_dim = x.shape
    x, B, C, dt, A, lam, theta = (t.float() for t in (x, B, C, dt, A, lam, theta))
    rank, d_state = B.shape[3], B.shape[4]
    n_blocks = d_state // block_size

    rot = _block_rotations(theta, block_size)
    if heads_per_group != 1:
        rot = rot.repeat_interleave(heads_per_group, dim=2)
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)

    alpha = torch.exp(dt * A)
    gamma = lam * dt
    beta = (1.0 - lam) * dt * alpha

    blocked = (batch, n_heads, rank, n_blocks, block_size, head_dim)
    h = x.new_zeros(blocked)
    v_prev = torch.zeros_like(h)
    outputs = []
    for t in range(seq_len):
        v_t = (B[:, t].unsqueeze(-1) * x[:, t].unsqueeze(2).unsqueeze(2)).reshape(blocked)
        rot_t = rot[:, t]
        a_t = alpha[:, t].view(batch, n_heads, 1, 1, 1, 1)
        g_t = gamma[:, t].view(batch, n_heads, 1, 1, 1, 1)
        b_t = beta[:, t].view(batch, n_heads, 1, 1, 1, 1)
        h = (
            a_t * torch.einsum("bhkij,bhrkjp->bhrkip", rot_t, h)
            + g_t * v_t
            + b_t * torch.einsum("bhkij,bhrkjp->bhrkip", rot_t, v_prev)
        )
        readout = C[:, t].reshape(batch, n_heads, rank, n_blocks, block_size)
        outputs.append((readout.unsqueeze(-1) * h).sum(dim=(2, 3, 4)))
        v_prev = v_t

    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("block_size", [2, 3, 4])
@pytest.mark.parametrize("n_groups", [1, 2], ids=["g1", "g2"])
def test_ssd_reference_blocked_matches_naive_matrix_scan(block_size: int, n_groups: int):
    """
    The factorized scan must equal the un-factorized matrix scan.

    This is the load-bearing test of the whole change. The Mamba-3 paper derives the RoPE trick
    under an explicit ``R_i R_j = R_j R_i`` assumption, and the claim here is that the
    assumption is unnecessary: ``R_t ... R_{s+1} = Q_t Q_s^-1`` needs only associativity, plus
    orthogonality to make the inverse a transpose. If that reasoning is wrong, the reference and
    this oracle disagree at ``b >= 3`` while still agreeing at ``b == 2`` -- which is why the
    ``b == 2`` case is parameterized alongside rather than trusted on its own.

    Tolerance: both sides are fp32 and differ in operation order (pre-rotated ``B``/``C`` and a
    scalar scan versus a rotated state), so error accumulates over the ``seq_len * rank *
    d_state`` terms of the contraction. At these sizes that is ~1e-5 relative; a broken
    factorization is O(1).
    """
    torch.manual_seed(0)
    batch, seq_len, n_heads, head_dim = 2, 10, 4, 4
    d_state, mimo_rank = 4 * block_size, 2
    heads_per_group = n_heads // n_groups
    n_blocks = d_state // block_size

    x = torch.randn(batch, seq_len, n_heads, head_dim)
    B = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    C = torch.randn(batch, seq_len, n_groups, mimo_rank, d_state)
    dt = torch.rand(batch, seq_len, n_heads) * 0.1 + 0.01
    A = -torch.rand(n_heads) - 0.5
    lam = torch.rand(batch, seq_len, n_heads)
    # Large angles, so the rotations genuinely fail to commute; near-zero angles would let an
    # abelian implementation pass.
    theta = torch.randn(batch, seq_len, n_groups, n_blocks, block_size * (block_size - 1) // 2)

    y = mamba3_ssd_reference(
        x, B, C, dt, A, lam, theta, heads_per_group=heads_per_group, block_size=block_size
    )
    expected = _naive_block_matrix_scan(
        x, B, C, dt, A, lam, theta, heads_per_group=heads_per_group, block_size=block_size
    )
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-5)


def test_block_size_3_transitions_do_not_commute():
    """
    Sanity check on the premise: at ``b >= 3`` the sampled transitions must actually fail to
    commute, otherwise every hardness claim above is vacuous.
    """
    torch.manual_seed(0)
    rot = _block_rotations(torch.randn(2, 4, 1, 1, 3), 3)
    commutator = rot[:, 0] @ rot[:, 1] - rot[:, 1] @ rot[:, 0]
    assert commutator.abs().max() > 0.1

    # b == 2 is abelian by construction, and must stay that way.
    rot2 = _block_rotations(torch.randn(2, 4, 1, 1, 1), 2)
    commutator2 = rot2[:, 0] @ rot2[:, 1] - rot2[:, 1] @ rot2[:, 0]
    assert commutator2.abs().max() < 1e-6


def test_ssd_reference_rejects_legacy_theta_shape_above_block_size_2():
    """A 4-D theta is only meaningful at b == 2; anything else must fail loudly, not broadcast."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 2, 4)
    B = torch.randn(1, 4, 1, 1, 6)
    C = torch.randn(1, 4, 1, 1, 6)
    dt = torch.rand(1, 4, 2) * 0.1
    A = -torch.rand(2)
    lam = torch.rand(1, 4, 2)
    with pytest.raises(ValueError, match="theta must be 5-D"):
        mamba3_ssd_reference(
            x, B, C, dt, A, lam, torch.randn(1, 4, 1, 3), heads_per_group=2, block_size=3
        )
