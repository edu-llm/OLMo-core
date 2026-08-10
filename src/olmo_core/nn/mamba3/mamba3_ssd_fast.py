"""
Performance-tuned copy of :mod:`olmo_core.nn.mamba3.mamba3_ssd_official`, specialised for the
``block_size >= 3`` (NC^1) configuration.

:mod:`mamba3_ssd_official` remains the correctness oracle: its math is untouched, and it differs
only in routing its kernel call through the shared ``_mamba3_siso_combined_eager`` wrapper. This
module calls the *same unmodified* upstream Triton kernel
(``mamba_ssm.ops.triton.mamba3.mamba3_siso_combined``); nothing in ``mamba_ssm`` is forked.

Why the kernel itself needs no change
-------------------------------------
With ``Angles = 0`` the upstream kernel degenerates to a pure per-head scalar-decay SSD scan,
which is exactly the right computation and is already well tuned (it even carries an explicit
Blackwell branch, ``mamba3_siso_fwd.py:19-31``). At ``b == 2`` that leaves almost nothing on the
table. At ``b >= 3`` it leaves a *lot*, because the SO(b) rotation is PyTorch preprocessing that
runs identically whichever scan follows it, so the kernel cannot touch it. Measured per layer at
1B Mamba dims, seq 512: ``b=2`` 2.57 ms against ``b=3`` 8.76 ms. The 6.2 ms difference -- about
70% of the ``b=3`` mixer -- is entirely rotation preprocessing.

So the speedups here are all in that preprocessing:

1. **Rodrigues instead of** ``torch.matrix_exp`` **at** ``b == 3`` (:func:`_rodrigues_so3`).
   ``matrix_exp`` is Pade scaling-and-squaring plus an LU solve, and its backward needs a second
   exponential of a ``2b x 2b`` block matrix. For ``so(3)`` there is an exact closed form.
   Measured: 5.7x forward, **9.0x forward+backward**, agreeing with ``matrix_exp`` to 6.7e-16 in
   float64. This is an identity, not an approximation.
2. **A sequence-length-adaptive prefix-product chunk** (:func:`_adaptive_scan_chunk`). The scan
   cost that matters on GPU is *dependent kernel launches* traded against per-launch arithmetic,
   and the balance point moves with ``T``: the upstream default of 64 pays 63 sequential matmuls
   plus few Hillis-Steele levels, far too fine-grained at short ``T`` and too coarse at long ``T``.
   The ``~T/128`` rule (chunk 8 at ``T<=1024`` up to 64 at ``T>=8192``) tracks the measured
   optimum -- this is the ``(d)`` "batch the block-diagonal matmuls harder" lever.
3. **A four-value quaternion pointwise scan with analytic backward** at ``b == 3``. The custom
   backward reduces the reverse adjoint to one reverse cumsum, avoiding the prototype pointwise
   scan's one-copy-per-timestep OOM. The quaternion is applied directly to ``B``/``C`` instead of
   materializing every cumulative 3x3 matrix.
4. **One fused rotation einsum for** ``B`` **and** ``C`` on the matrix fallback paths, since both
   are rotated by the same ``Q^T``.
5. **A selective fp32 floor** (``selective_fp32``). The float32 requirement is on the *prefix
   product*, where orthogonality drift accumulates as ``O(T * eps)``. Applying the resulting
   rotation to ``B``/``C`` in float32 is wasted work, because ``mamba3_siso_combined`` hard-casts
   ``Q``/``K`` to bfloat16 on the next line anyway. This keeps the floor where it earns its
   keep and drops it where it does not.

Rodrigues also removes a latent hazard rather than just being faster: ``torch.matrix_exp``
accepts bfloat16 and returns silent ``NaN``/``Inf`` rather than raising, so the "it crashes
loudly" guard assumed elsewhere does not exist. The closed form stays finite in bfloat16.

An alternate scan: :func:`mamba3_ssd_simple_gla`
------------------------------------------------
Everything above optimises the preprocessing in front of ``mamba3_siso_combined``. The scan
itself has a different problem, which no amount of preprocessing touches: its grid is
``(nheads, batch)``, so the 370M arm's geometry puts 32 thread blocks on an A100's 108 SMs and
each of them walks 64 chunks in sequence. :func:`mamba3_ssd_simple_gla` computes the same
recurrence through ``fla``'s ``chunk_simple_gla``, whose grid also spans chunk tiles.

It is a *second backend*, not a replacement: nothing selects it by default, both are reachable
by name through :data:`~olmo_core.nn.mamba3.mamba3_ssd_api.SSD_BACKENDS`, and they share this
module's rotation, so the choice is only which scan runs behind it.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from .mamba3_ssd_api import (
    _block_rotations,
    _mamba3_siso_combined_eager,
    _rotate_bc,
    kernel_padded_width,
)

__all__ = [
    "fast_mamba3_is_available",
    "mamba3_ssd_fast",
    "mamba3_ssd_simple_gla",
    "simple_gla_is_available",
    "fast_block_rotations",
    "fast_cumulative_block_rotation",
    "associative_cumulative_block_rotation",
    "associative_autograd_cumulative_block_rotation",
    "quaternion_cumulative_block_rotation",
    "ROTATION_SCAN_IMPLS",
    "resolve_rotation_scan_impl",
]

# Copied verbatim from `mamba3_ssd_official`; see its module docstring for the derivation of
# each one. They are duplicated rather than imported so that module stays untouched. The
# power-of-two padding rule itself lives in `kernel_padded_width` (imported above), the single
# source shared with the official adapter and the mixer's diagnostics.
_ANGLE_WIDTH = 2
_LOGIT_EPS = 1e-6

# Sequential-product chunk for the prefix scan. Cost is ``chunk - 1`` dependent matmuls plus
# ``log2(T / chunk)`` Hillis-Steele levels -- a latency/arithmetic trade whose optimum *grows*
# with the sequence length, in the opposite direction from the SSD chunk size. A single fixed
# chunk is wrong at both ends, so :func:`_adaptive_scan_chunk` scales it instead. The rule tracks
# ``~T/128`` across a B200-absent GPU sweep (b=3 scan at T=4096: 13.9 ms at chunk 8 -> 11.1 ms at
# chunk 32; chunk 16 fastest at T=2048), clamped to a sane band; treat it as a tuned default, not
# a proven optimum.
_ROTATION_SCAN_CHUNK_MIN = 8
_ROTATION_SCAN_CHUNK_MAX = 64
_ROTATION_SCAN_TARGET_DIVISOR = 128
# Retuning escape hatch, read once at import so the traced region sees a plain int constant rather
# than an `os.environ` lookup that could graph-break. That sweep predates two things which move the
# optimum: it ran eager, and it ran off-B200. Once the preprocessing compiles, Inductor makes the
# per-level arithmetic cheap but cannot remove a data dependency, so the binding cost becomes the
# ``chunk - 1`` chain and the optimum shifts *down*. The override bypasses the clamp so the
# fully-parallel end stays reachable. Safe to sweep blind: the chunk only re-brackets an associative
# product and never changes the result (``test_prefix_scan_is_invariant_to_scan_chunk``); it costs
# Hillis-Steele memory as it shrinks, which is the only thing to watch.
_ROTATION_SCAN_CHUNK_ENV = "MAMBA3_ROTATION_SCAN_CHUNK"
_ROTATION_SCAN_CHUNK_OVERRIDE: Optional[int] = (
    int(os.environ[_ROTATION_SCAN_CHUNK_ENV]) if os.environ.get(_ROTATION_SCAN_CHUNK_ENV) else None
)

# Which prefix-scan implementation the b>=3 path uses when the caller does not name one. This is
# the *default*, not the setting: every entry point takes an explicit `rotation_scan_impl` that
# outranks it (`resolve_rotation_scan_impl`), so the choice can live in a config field a checkpoint
# records and a log line reports. It used to be reachable only from here, which made it invisible
# to both -- and the fallback is silent, so a relaunch in a fresh shell that lost the export
# trained at chunked's 33,468 tok/s instead of quaternion's 75,040 with nothing raising.
#
# "chunked" is the hand-rolled sequential-product-plus-Hillis-Steele form; "associative" routes
# through `torch.associative_scan`,
# which Inductor can lower to a single fused `tl.associative_scan` Triton kernel instead of the
# ~`chunk - 1 + log2(T / chunk)` dependent kernel launches the chunked form costs. Default stays
# "chunked" so this is opt-in and revertible by unsetting one variable. Validated at import rather
# than silently falling back, because a typo here would quietly cost the speedup it was set to buy.
#
# "associative_autograd" is the same forward with `associative_scan`'s *own* autograd taken out of
# the loop -- see `associative_autograd_cumulative_block_rotation`. Plain "associative" is kept
# reachable because its forward is the one that has been run end-to-end on real data; when both
# work, "associative_autograd" is the one to use.
#
# "quaternion" (b == 3 only) carries the SO(3) prefix product as a 4-value unit quaternion instead
# of a 9-value 3x3 matrix -- see `quaternion_cumulative_block_rotation`. The motivation is
# `combine_mode="pointwise"`: the 9-leaf matrix carry overflows what `tl.associative_scan` keeps in
# registers and OOMs (1152 GiB), while a 4-leaf carry may fit. That register question is CUDA-only,
# so this path is built to be correct and measurable; every non-b==3 call falls back to chunked.
_ROTATION_SCAN_IMPL_ENV = "MAMBA3_ROTATION_SCAN_IMPL"
_ROTATION_SCAN_IMPL = os.environ.get(_ROTATION_SCAN_IMPL_ENV, "chunked").strip().lower()

#: Every prefix-scan implementation the ``b >= 3`` rotation can be pointed at. Public because it
#: is the valid set for :func:`resolve_rotation_scan_impl`, for the ``rotation_scan_impl`` config
#: field, and for the training script's ``--rotation-scan-impl`` flag, none of which should carry
#: their own copy of the list.
ROTATION_SCAN_IMPLS = ("chunked", "associative", "associative_autograd", "quaternion")

# Pinned, not selected per-device. The docs call ``pointwise`` the more efficient mode, and for a
# 9-leaf 3x3 combine that is inverted. Measured on a B200 at the production shape
# (32, 4096, 1, 64, 3), fp32, compiled fwd+bwd: generic **61.2 ms** vs the chunked path's 181.2 ms
# (2.96x), while pointwise is **837.5 ms** -- 4.6x *slower* than chunked -- and eagerly OOMs trying
# to allocate 1152 GiB. Pointwise is also the mode whose backward returns NaN in training. The
# 18-leaf affine combine in the backward is larger still, so it wants generic even more.
_ROTATION_SCAN_COMBINE_MODE = "generic"
if _ROTATION_SCAN_IMPL not in ROTATION_SCAN_IMPLS:
    raise ValueError(
        f"{_ROTATION_SCAN_IMPL_ENV} must be one of {ROTATION_SCAN_IMPLS}, "
        f"got {_ROTATION_SCAN_IMPL!r}"
    )


def resolve_rotation_scan_impl(impl: Optional[str]) -> str:
    """
    Canonicalise an explicit scan-implementation request, falling back to the module default.

    ``None`` means "whatever ``MAMBA3_ROTATION_SCAN_IMPL`` asked for", which is what every caller
    predating the explicit parameter passes and is why the default path is unchanged. Anything
    else outranks the environment. That ordering is the point of the function: the scan was
    previously selectable *only* by an environment variable read once at import, so it never
    entered the saved checkpoint config and was never logged, and a resumed run in a fresh shell
    that forgot the export silently fell back to ``chunked`` -- 33,468 tok/s against
    ``quaternion``'s 75,040 on the 370M hybrid, a 2.2x loss with no error anywhere.

    Case and surrounding whitespace are normalised because both sources are hand-typed: a shell
    export and a CLI flag.

    :param impl: One of :data:`ROTATION_SCAN_IMPLS`, or ``None`` for the module default.

    :returns: The canonical lower-case name.

    :raises ValueError: If ``impl`` names no known implementation. Rejected rather than defaulted
        for the same reason the resolution order exists: a typo that quietly selected ``chunked``
        would hand back precisely the slow run the caller was trying to avoid.
    """
    if impl is None:
        return _ROTATION_SCAN_IMPL
    normalised = impl.strip().lower()
    if normalised not in ROTATION_SCAN_IMPLS:
        raise ValueError(
            f"rotation_scan_impl (also settable as {_ROTATION_SCAN_IMPL_ENV}) must be one of "
            f"{ROTATION_SCAN_IMPLS}, or None to take the {_ROTATION_SCAN_IMPL_ENV} default; "
            f"got {impl!r}"
        )
    return normalised


def _adaptive_scan_chunk(seq_len: int) -> int:
    """
    Pick the prefix-scan chunk for a sequence of length ``seq_len``.

    The chunk decides only how the associative product is bracketed, never the result (guaranteed
    by ``test_prefix_scan_is_invariant_to_scan_chunk``), so it is free to tune purely for speed. A
    longer sequence wants a coarser chunk -- fewer, larger batched block-diagonal matmuls in place
    of more Hillis-Steele levels -- which is the ``(d)`` "batch harder" lever measured for ``b=3``.
    ``~T/128`` clamped to ``[8, 64]`` matches the swept optimum and degrades gracefully off it.

    :param seq_len: The sequence length the scan will run over.

    :returns: A chunk length in ``[8, 64]``, or the unclamped ``MAMBA3_ROTATION_SCAN_CHUNK``
        override when that environment variable is set.
    """
    if _ROTATION_SCAN_CHUNK_OVERRIDE is not None:
        return _ROTATION_SCAN_CHUNK_OVERRIDE
    target = seq_len // _ROTATION_SCAN_TARGET_DIVISOR
    return max(_ROTATION_SCAN_CHUNK_MIN, min(_ROTATION_SCAN_CHUNK_MAX, target))


# Below this squared angle the ``sin(phi)/phi`` and ``(1-cos(phi))/phi^2`` coefficients are
# evaluated by their Taylor series instead. ``theta_proj`` initialises at ``std * 0.1``, so the
# near-zero regime is where training *starts* -- getting it wrong would produce NaN gradients on
# step one. At ``phi^2 = 1e-6`` the truncated terms are ``O(phi^6 / 5040) ~ 1e-22``, far below
# float32 resolution.
_SMALL_ANGLE_SQ = 1e-6


def fast_mamba3_is_available() -> bool:
    """Whether the official Mamba-3 SISO Triton entry point can be imported."""
    from .mamba3_ssd_official import official_mamba3_is_available

    return official_mamba3_is_available()


def simple_gla_is_available() -> bool:
    """
    Whether ``fla``'s chunked simple-GLA Triton entry point can be imported.

    Only a genuinely absent top-level ``fla`` returns ``False`` -- the same policy, for the same
    reason, as :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.has_mamba3`. Broken transitive
    dependencies, a missing ``simple_gla`` API, and binary/ABI import errors are re-raised with
    their original diagnostics rather than reported as an optional-kernel absence.

    Misclassifying those was quiet in both directions: the strict-request refusal in
    :func:`_require_simple_gla` blamed "an installed flash-linear-attention" for a package that
    was installed, and ``requires_simple_gla`` skipped the ``simple_gla`` vs ``official_fast``
    parity tests -- this backend's only correctness gate -- leaving CI green having checked
    nothing.
    """
    try:
        from fla.ops.simple_gla import chunk_simple_gla  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "fla":
            return False
        raise
    return True


def _rodrigues_so3(theta: torch.Tensor) -> torch.Tensor:
    """
    Closed-form ``exp`` of the ``3 x 3`` skew-symmetric matrix built from three angles.

    Rodrigues' rotation formula, ``exp(S) = I + (sin(phi)/phi) S + ((1-cos(phi))/phi^2) S^2``
    with ``phi = ||theta||``. The identity holds for any labelling of the three angles onto the
    strict upper triangle, because ``phi`` is the norm of the axis vector and permuting or
    negating its components does not change that norm -- so this matches
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api._block_rotations` exactly without needing to
    reproduce its sign convention.

    :param theta: Angles of shape ``(..., 3)``.

    :returns: Rotation matrices of shape ``(..., 3, 3)``, orthogonal to machine precision.
    """
    if theta.shape[-1] != 3:
        raise ValueError(f"Rodrigues needs exactly 3 angles, got {theta.shape[-1]}")

    t1, t2, t3 = theta[..., 0], theta[..., 1], theta[..., 2]
    zero = torch.zeros_like(t1)
    # Matches `_skew_from_angles(theta, 3)`: angles fill the strict upper triangle in row-major
    # order and are mirrored with the opposite sign below it.
    skew = torch.stack(
        [
            torch.stack([zero, t1, t2], dim=-1),
            torch.stack([-t1, zero, t3], dim=-1),
            torch.stack([-t2, -t3, zero], dim=-1),
        ],
        dim=-2,
    )

    phi_sq = (theta * theta).sum(-1)
    # Clamp before the sqrt, not after: `sqrt` has an infinite derivative at 0, so an unclamped
    # `phi` would emit NaN gradients for any block whose angles are all exactly zero even though
    # `torch.where` discards its value. Both branches below must be finite everywhere.
    phi = torch.sqrt(phi_sq.clamp_min(_SMALL_ANGLE_SQ))

    small = phi_sq < _SMALL_ANGLE_SQ
    sin_over_phi = torch.where(
        small, 1.0 - phi_sq / 6.0 + phi_sq * phi_sq / 120.0, torch.sin(phi) / phi
    )
    one_minus_cos_over_phi_sq = torch.where(
        small,
        0.5 - phi_sq / 24.0 + phi_sq * phi_sq / 720.0,
        (1.0 - torch.cos(phi)) / (phi * phi),
    )

    eye = torch.eye(3, dtype=theta.dtype, device=theta.device).expand_as(skew)
    a = sin_over_phi.unsqueeze(-1).unsqueeze(-1)
    b = one_minus_cos_over_phi_sq.unsqueeze(-1).unsqueeze(-1)
    return eye + a * skew + b * (skew @ skew)


def fast_block_rotations(theta: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Map per-step angles to per-step rotations in ``SO(b)``, taking the closed form when there is
    one.

    Numerically identical to
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api._block_rotations`; only ``b == 3`` takes a
    different route to the same matrix. Every other ``b`` falls through to ``matrix_exp``, which
    stays surjective onto ``SO(b)`` and so keeps every element of ``A_5`` representable.

    ``b == 4`` is the only other block size in wide use, and it does have a closed form --
    ``SO(4)`` factors as a pair of unit quaternions -- but it is not implemented here, so it
    pays ``matrix_exp`` and its expensive Frechet-derivative backward. Measured at 1B Mamba dims
    and seq 512, the rotation costs 20 ms forward+backward at ``b=3`` against 129 ms at ``b=4``.
    Implementing the quaternion pair would close most of that gap; until then ``b=3`` is the
    cheap non-solvable block, and it is not less expressive for the purpose -- ``A_5`` already
    lives in ``SO(3)``.

    :param theta: Angles of shape ``(..., b*(b-1)//2)``.
    :param block_size: The rotation block size ``b``.
    """
    if block_size == 3:
        return _rodrigues_so3(theta)
    return _block_rotations(theta, block_size)


def fast_cumulative_block_rotation(
    rot: torch.Tensor, chunk_size: Optional[int] = None, *, scan_impl: Optional[str] = None
) -> torch.Tensor:
    """
    Inclusive prefix product ``Q_t = R_t R_{t-1} ... R_1`` over the sequence axis.

    Same algorithm as
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api._cumulative_block_rotation` -- a sequential
    product within each chunk, Hillis-Steele across chunk boundaries -- but with ``chunk_size``
    reachable by the caller instead of pinned at 64, and defaulted per sequence length by
    :func:`_adaptive_scan_chunk` instead of a constant.

    :param rot: Per-step rotations of shape ``(batch, seq_len, n_groups, n_blocks, b, b)``.
    :param chunk_size: Sequential-product chunk length. ``None`` picks it from the sequence
        length via :func:`_adaptive_scan_chunk`.
    :param scan_impl: Which of :data:`ROTATION_SCAN_IMPLS` to run; ``None`` takes the
        ``MAMBA3_ROTATION_SCAN_IMPL`` default. ``"quaternion"`` is not reachable from here --
        it consumes angles, not the pre-built matrices this function receives, so
        :func:`_fast_rotate_bc_pair` branches to it one level up and anything arriving here
        under that name gets the chunked path.

    :returns: Inclusive prefix products, same shape as ``rot``.
    """
    from .mamba3_ssd_api import _cumulative_block_rotation

    impl = resolve_rotation_scan_impl(scan_impl)

    # Only b == 3 has a 9-leaf pointwise form here; every other block size keeps the chunked path.
    # That fallback is load-bearing rather than a nicety: an earlier attempt sliced a 4x4 rotation
    # down to its top-left 3x3, which is still orthogonal-looking output and was caught only by the
    # b=4 mixer tests. Both scan variants refuse b != 3 outright for the same reason.
    if rot.shape[-1] == 3:
        if impl == "associative_autograd":
            return associative_autograd_cumulative_block_rotation(rot)
        if impl == "associative":
            return associative_cumulative_block_rotation(rot)
    if chunk_size is None:
        chunk_size = _adaptive_scan_chunk(rot.shape[1])
    return _cumulative_block_rotation(rot, chunk_size=chunk_size)


def _to_leaves(mat: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Split a ``(..., 3, 3)`` tensor into the 9 contiguous elementwise leaves the combines consume.

    One definition of the row-major leaf order shared by every scan here, because the forward and
    the backward have to agree on it and a disagreement would be a silently transposed rotation.
    """
    return tuple(mat[..., i, j].contiguous() for i in range(3) for j in range(3))


def _from_leaves(leaves) -> torch.Tensor:
    """Inverse of :func:`_to_leaves`: reassemble 9 elementwise leaves into ``(..., 3, 3)``."""
    return torch.stack(tuple(leaves), dim=-1).unflatten(-1, (3, 3))


def _so3_pointwise_combine(a, b):
    """
    Compose two rotations held as 9 elementwise tensors, returning ``b @ a``.

    Nine separate leaves rather than a ``(..., 3, 3)`` tensor keeps every output a sum of products of
    scalars, which is what ``associative_scan`` demands of a ``pointwise`` combine. It nonetheless
    runs under ``generic``: a 9-value carry overflows what ``tl.associative_scan`` keeps in registers,
    measured at 837.5 ms against generic's 61.2 ms -- see :data:`_ROTATION_SCAN_COMBINE_MODE`.
    ``b @ a`` (not ``a @ b``) keeps the newest rotation on the left, matching
    ``Q_t = R_t R_{t-1} ... R_1`` -- see ``test_prefix_scan_equals_the_naive_ordered_product``.
    """
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = a
    b00, b01, b02, b10, b11, b12, b20, b21, b22 = b
    return (
        b00 * a00 + b01 * a10 + b02 * a20,
        b00 * a01 + b01 * a11 + b02 * a21,
        b00 * a02 + b01 * a12 + b02 * a22,
        b10 * a00 + b11 * a10 + b12 * a20,
        b10 * a01 + b11 * a11 + b12 * a21,
        b10 * a02 + b11 * a12 + b12 * a22,
        b20 * a00 + b21 * a10 + b22 * a20,
        b20 * a01 + b21 * a11 + b22 * a21,
        b20 * a02 + b21 * a12 + b22 * a22,
    )


def associative_cumulative_block_rotation(rot: torch.Tensor) -> torch.Tensor:
    """
    Inclusive prefix product ``Q_t = R_t R_{t-1} ... R_1`` via :func:`torch.associative_scan`.

    Computes exactly what :func:`fast_cumulative_block_rotation` does, but as one tree reduction of
    depth ``log2(T)`` that Inductor fuses into a single kernel, rather than ``chunk - 1`` dependent
    matmuls plus ``log2(T / chunk)`` Hillis-Steele levels each round-tripping through global memory.
    That dependency chain -- not arithmetic -- is what leaves the b=3 arm latency-bound.

    Uses ``combine_mode="generic"`` -- see :data:`_ROTATION_SCAN_COMBINE_MODE` for the measurement
    that rules out ``pointwise`` despite the docs preferring it.

    :param rot: Per-step rotations of shape ``(batch, seq_len, n_groups, n_blocks, 3, 3)``.

    :returns: Inclusive prefix products, same shape as ``rot``.
    """
    from torch._higher_order_ops.associative_scan import associative_scan

    if rot.shape[-1] != 3:
        # The combine is written out for 3x3. Silently slicing a larger block would truncate the
        # rotation and still return plausible-looking orthogonal-ish output, so refuse instead.
        raise ValueError(
            f"associative_cumulative_block_rotation only supports block_size 3, "
            f"got {rot.shape[-1]}; use the chunked path for other block sizes"
        )
    if rot.shape[1] == 1:
        # A length-1 scan is the identity, and the scan op has no work to bracket.
        return rot

    leaves = _to_leaves(rot)
    scanned = associative_scan(
        _so3_pointwise_combine, leaves, dim=1, combine_mode=_ROTATION_SCAN_COMBINE_MODE
    )
    return _from_leaves(scanned)


def _so3_affine_combine(x, y):
    """
    Compose two affine maps ``M -> A M + b`` held as 18 elementwise leaves (9 for ``A``, 9 for
    ``b``), returning ``y after x``.

    ``(A_y, b_y) . (A_x, b_x) = (A_y A_x, A_y b_x + b_y)``, which is associative because matrix
    multiplication is. ``x`` is the lower-index (older) operand and ``y`` the newer, matching
    :func:`_so3_pointwise_combine`'s ``b @ a``; both halves reuse that function so the two scans
    cannot drift apart on operand order.

    Eighteen separate leaves rather than two ``(..., 3, 3)`` tensors, mirroring the forward's nine:
    every output is a sum of products of scalars, which keeps both ``combine_mode`` options open.
    ``generic`` is the one actually taken -- see :data:`_ROTATION_SCAN_COMBINE_MODE` -- and this
    combine is the larger of the two, so it wants ``generic`` by an even wider margin.
    """
    a_x, b_x = x[:9], x[9:]
    a_y, b_y = y[:9], y[9:]
    a_out = _so3_pointwise_combine(a_x, a_y)  # A_y @ A_x
    ab = _so3_pointwise_combine(b_x, a_y)  # A_y @ b_x
    return a_out + tuple(ab[k] + b_y[k] for k in range(9))


def _prefix_rotation_backward(
    rot: torch.Tensor, q: torch.Tensor, grad_q: torch.Tensor
) -> torch.Tensor:
    """
    Analytic gradient of ``Q_t = R_t R_{t-1} ... R_1`` with respect to ``R``.

    With ``Q_t = S_{t,i+1} R_i Q_{i-1}`` and ``S_{t,i+1} = R_t ... R_{i+1}``, perturbing ``R_i``
    gives ``dQ_t = S_{t,i+1} dR_i Q_{i-1}`` for every ``t >= i``, so for ``G_t = dL/dQ_t``::

        dL = sum_{t>=i} tr(G_t^T S_{t,i+1} dR_i Q_{i-1})   =>   dL/dR_i = M_i Q_{i-1}^T
        M_i = sum_{t>=i} S_{t,i+1}^T G_t = G_i + R_{i+1}^T M_{i+1}

    the last step because ``S_{t,i+1}^T = R_{i+1}^T S_{t,i+2}^T``. Note what this derivation does
    *not* use: orthogonality. Only associativity and ``(XY)^T = Y^T X^T``, so the result is the
    true gradient even where the scan has drifted off ``SO(3)`` -- and ``gradcheck`` can therefore
    probe it with arbitrary matrices.

    ``M`` is a reverse first-order linear recurrence, which is itself associative-scannable as the
    affine pairs of :func:`_so3_affine_combine`. So the backward is one scan of the same depth as
    the forward plus one batched matmul, and -- the property that matters here -- it contains no
    division and no inverse. A backward that instead recovered ``Q_{i-1}`` as ``R_i^{-1} Q_i``
    would be at its least stable exactly at initialisation, where ``theta_proj`` starts at
    ``std * 0.1`` and every ``R`` is near-identity; this form treats that regime as ordinary.

    :param rot: Per-step rotations ``R``, shape ``(batch, seq_len, n_groups, n_blocks, 3, 3)``.
    :param q: The forward's prefix products ``Q``, same shape.
    :param grad_q: Incoming gradient ``G``, same shape.

    :returns: ``dL/dR``, same shape.
    """
    from torch._higher_order_ops.associative_scan import associative_scan

    # A_i = R_{i+1}^T, zero at the final step: M_{T+1} is zero, so A_T multiplies nothing and never
    # reaches the answer. Zero rather than identity so an off-by-one here fails loudly.
    rot_t = rot.transpose(-1, -2)
    shifted = torch.cat([rot_t[:, 1:], torch.zeros_like(rot_t[:, :1])], dim=1)

    # Reverse the sequence so the reverse recurrence becomes an ordinary forward inclusive scan.
    # `flip` rather than `associative_scan(..., reverse=True)`: the reverse lowering is a second
    # prototype code path, and the whole reason this function exists is that the first one
    # miscompiled its backward. Two elementwise copies is a cheap price for staying on the exact
    # scan configuration the forward already validates.
    leaves = _to_leaves(shifted.flip(1)) + _to_leaves(grad_q.flip(1))
    scanned = associative_scan(
        _so3_affine_combine, leaves, dim=1, combine_mode=_ROTATION_SCAN_COMBINE_MODE
    )
    m = _from_leaves(scanned[9:]).flip(1)

    # dL/dR_i = M_i Q_{i-1}^T with Q_0 = I, so the first step is just M_0 and no identity block
    # has to be materialised to hold its place.
    return torch.cat([m[:, :1], m[:, 1:] @ q[:, :-1].transpose(-1, -2)], dim=1)


class _AssociativePrefixRotation(torch.autograd.Function):
    """
    ``associative_scan`` forward bolted to the analytic backward above.

    ``torch.associative_scan`` is a prototype whose documentation warns about miscompiles, and its
    backward is observed to return NaN on CUDA/fp32 with ``combine_mode="pointwise"`` while its
    forward reproduces the chunked path's CE loss exactly on a real training step. Wrapping it here
    keeps the forward that works and replaces the autograd that does not, so neither direction goes
    through ``associative_scan``'s own differentiation rule.

    ``rot`` and ``Q`` are both saved. That is still cheaper than the chunked path, which stores its
    ``chunk`` sequential intermediates plus ``log2(T / chunk)`` Hillis-Steele levels, and ``Q`` is
    the op's own output so it is usually alive anyway.
    """

    @staticmethod
    def forward(ctx, rot: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q = associative_cumulative_block_rotation(rot)
        ctx.save_for_backward(rot, q)
        return q

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q: torch.Tensor):  # type: ignore[override]
        # `once_differentiable` is deliberate: a second-order backward would have to differentiate
        # the scan again, which is the broken path. Better to raise than to hand back the NaN this
        # class was written to avoid.
        if not ctx.needs_input_grad[0]:
            return None
        rot, q = ctx.saved_tensors
        return _prefix_rotation_backward(rot, q, grad_q.contiguous())


def associative_autograd_cumulative_block_rotation(rot: torch.Tensor) -> torch.Tensor:
    """
    Inclusive prefix product ``Q_t = R_t R_{t-1} ... R_1`` with a fast scan in *both* directions.

    Same value as :func:`associative_cumulative_block_rotation` -- bit-for-bit, it calls it -- but
    the gradient comes from :func:`_prefix_rotation_backward` instead of from
    ``associative_scan``'s autograd. Forward and backward are then each one tree reduction of
    depth ``log2(T)``, against the ``chunk - 1`` dependent matmuls plus ``log2(T / chunk)``
    Hillis-Steele levels the chunked form pays in each direction.

    Selected by ``MAMBA3_ROTATION_SCAN_IMPL=associative_autograd``; unset the variable to return to
    the chunked default.

    :param rot: Per-step rotations of shape ``(batch, seq_len, n_groups, n_blocks, 3, 3)``.

    :returns: Inclusive prefix products, same shape as ``rot``.
    """
    if rot.shape[-1] != 3:
        # Same refusal as the sibling: silently slicing a larger block would truncate the rotation
        # and still return plausible-looking orthogonal-ish output.
        raise ValueError(
            f"associative_autograd_cumulative_block_rotation only supports block_size 3, "
            f"got {rot.shape[-1]}; use the chunked path for other block sizes"
        )
    if rot.shape[1] == 1:
        # A length-1 scan is the identity in both directions, and returning `rot` itself keeps that
        # differentiable without entering the scan op at all.
        return rot
    return _AssociativePrefixRotation.apply(rot)


def _angles_to_quaternion(theta: torch.Tensor) -> torch.Tensor:
    """
    Map three so(3) angles per block to the unit quaternion ``(w, x, y, z)`` of ``exp(skew)``.

    The skew is the same one :func:`_rodrigues_so3` builds,
    ``[[0, t1, t2], [-t1, 0, t3], [-t2, -t3, 0]]``, which is ``hat(v)`` for ``v = (-t3, t2, -t1)``
    under the standard hat map ``hat(v) = [[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]]``. The rotation is
    then ``exp(hat(v))``, an active rotation of angle ``phi = ||theta||`` about ``v/phi``, whose
    unit quaternion is ``(cos(phi/2), sin(phi/2)/phi * v)``. That axis/sign mapping is pinned by
    ``test_quaternion_roundtrip_matches_fast_block_rotations``, which requires
    ``_quaternion_to_matrix(_angles_to_quaternion(theta)) == fast_block_rotations(theta, 3)``.

    Small-angle handling mirrors :func:`_rodrigues_so3` exactly, because this is the init regime
    (``theta_proj`` starts at ``std * 0.1``): clamp before the sqrt so the discarded large-angle
    branch cannot inject a NaN gradient at all-zero angles, and evaluate both half-angle
    coefficients by their Taylor series below ``_SMALL_ANGLE_SQ``. At exactly zero this yields the
    identity quaternion ``(1, 0, 0, 0)``.

    :param theta: Angles of shape ``(..., 3)``.

    :returns: Unit quaternions of shape ``(..., 4)`` in ``(w, x, y, z)`` order.
    """
    if theta.shape[-1] != 3:
        raise ValueError(f"quaternion so(3) needs exactly 3 angles, got {theta.shape[-1]}")

    t1, t2, t3 = theta[..., 0], theta[..., 1], theta[..., 2]

    phi_sq = (theta * theta).sum(-1)
    # Clamp before the sqrt, not after: `sqrt` has an infinite derivative at 0, so an unclamped
    # `phi` would emit NaN gradients for all-zero-angle blocks even though `torch.where` discards
    # its value. Both branches below must be finite everywhere -- see `_rodrigues_so3`.
    phi = torch.sqrt(phi_sq.clamp_min(_SMALL_ANGLE_SQ))
    half = phi / 2.0

    small = phi_sq < _SMALL_ANGLE_SQ
    # w = cos(phi/2); the axis coefficient is sin(phi/2)/phi. Both are even in phi, so they have
    # exact Taylor series in phi_sq with truncation ~1e-23 at the threshold, far below float32
    # resolution, and finite gradients through phi_sq == 0.
    w = torch.where(small, 1.0 - phi_sq / 8.0 + phi_sq * phi_sq / 384.0, torch.cos(half))
    sin_half_over_phi = torch.where(
        small,
        0.5 - phi_sq / 48.0 + phi_sq * phi_sq / 3840.0,
        torch.sin(half) / phi,
    )

    # Vector part = (sin(phi/2)/phi) * v with v = (-t3, t2, -t1); see the docstring for why.
    x = -sin_half_over_phi * t3
    y = sin_half_over_phi * t2
    z = -sin_half_over_phi * t1
    return torch.stack([w, x, y, z], dim=-1)


def _quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion ``(w, x, y, z)`` to its ``3 x 3`` rotation matrix.

    The standard Hamilton (active) form, so that ``_quaternion_to_matrix(b_quat_x_a)`` equals
    ``R(b) @ R(a)`` and the round trip through :func:`_angles_to_quaternion` reproduces
    :func:`_rodrigues_so3`.

    This is where the double cover stops mattering: every entry is a *quadratic* in
    ``(w, x, y, z)``, so ``R(q) == R(-q)``. The Hamilton product of unit quaternions can walk to
    either sheet of the cover across the scan, but once a matrix is emitted the sign is gone, so no
    hemisphere / sign-flip / canonicalisation logic is needed anywhere in this path.

    :param q: Unit quaternions of shape ``(..., 4)``.

    :returns: Rotation matrices of shape ``(..., 3, 3)``.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _quaternion_pointwise_combine(a, b):
    """
    Compose two rotations held as 4 elementwise quaternion leaves, returning ``b ⊗ a``.

    Four separate leaves rather than a ``(..., 4)`` tensor keeps every output a sum of products of
    scalars, which is what ``associative_scan`` demands of a ``pointwise`` combine -- and a 4-value
    carry is the whole reason to try this: it may fit the registers a 9-value 3x3 carry overflowed.
    ``b ⊗ a`` (not ``a ⊗ b``) keeps the newest rotation on the left, matching
    :func:`_so3_pointwise_combine`'s ``b @ a`` and ``Q_t = R_t R_{t-1} ... R_1``, because
    ``R(b ⊗ a) == R(b) @ R(a)``.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        bw * aw - bx * ax - by * ay - bz * az,
        bw * ax + bx * aw + by * az - bz * ay,
        bw * ay - bx * az + by * aw + bz * ax,
        bw * az + bx * ay - by * ax + bz * aw,
    )


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``left ⊗ right`` for tensors ending in ``(w, x, y, z)``."""
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate, with no unit-norm assumption."""
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


class _QuaternionPrefix(torch.autograd.Function):
    """
    Pointwise quaternion prefix product with an analytic linear-memory backward.

    The prototype autograd for ``associative_scan(..., combine_mode="pointwise")`` saves one
    full scan level per token and OOMs at the production shape. For prefixes
    ``p_t = q_t ⊗ ... ⊗ q_0``, the reverse adjoint recurrence has the closed form

    ``a_t = p_t / |p_t|² ⊗ sum_{k>=t}(conj(p_k) ⊗ g_k)``.

    A reverse cumsum therefore replaces the scan operator's O(T)-copy backward while retaining
    its faster pointwise forward. The final input gradient is
    ``dq_t = a_t ⊗ conj(p_{t-1})``.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor) -> torch.Tensor:
        prefix = _quaternion_prefix_forward(q)
        ctx.save_for_backward(prefix)
        return prefix

    @staticmethod
    def backward(ctx, grad_prefix: torch.Tensor) -> tuple[torch.Tensor]:
        (prefix,) = ctx.saved_tensors
        return (_quaternion_prefix_backward(prefix, grad_prefix),)


def _quaternion_prefix_forward(q: torch.Tensor) -> torch.Tensor:
    """Pointwise quaternion prefix forward shared by both custom-autograd boundaries."""
    if q.shape[1] == 1:
        return q

    from torch._higher_order_ops.associative_scan import associative_scan

    leaves = tuple(q[..., i].contiguous() for i in range(4))
    combine_mode = "pointwise" if q.device.type in ("cuda", "xpu") else "generic"
    scanned = associative_scan(
        _quaternion_pointwise_combine,
        leaves,
        dim=1,
        combine_mode=combine_mode,
    )
    return torch.stack(scanned, dim=-1)


def _quaternion_prefix_backward(prefix: torch.Tensor, grad_prefix: torch.Tensor) -> torch.Tensor:
    """Analytic reverse adjoint for the newest-left quaternion prefix product."""
    weighted = _quaternion_multiply(
        _quaternion_conjugate(prefix),
        grad_prefix,
    )
    suffix_sum = torch.flip(
        torch.cumsum(torch.flip(weighted, dims=(1,)), dim=1),
        dims=(1,),
    )
    norm_sq = prefix.square().sum(dim=-1, keepdim=True)
    adjoint = _quaternion_multiply(prefix / norm_sq, suffix_sum)

    identity = torch.zeros_like(prefix[:, :1])
    identity[..., 0] = 1
    previous = torch.cat((identity, prefix[:, :-1]), dim=1)
    return _quaternion_multiply(adjoint, _quaternion_conjugate(previous))


def _quaternion_prefix(q: torch.Tensor) -> torch.Tensor:
    """Inclusive newest-left quaternion prefix product over sequence dimension 1."""
    if q.shape[-1] != 4:
        raise ValueError(f"expected quaternions with width 4, got {q.shape[-1]}")
    if q.shape[1] < 1:
        raise ValueError("quaternion prefix scan requires a non-empty sequence")
    if q.shape[1] == 1:
        return q
    return _QuaternionPrefix.apply(q)


def quaternion_cumulative_block_rotation(theta: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Inclusive prefix product ``Q_t = R_t R_{t-1} ... R_1`` for ``b == 3``, carried as quaternions.

    Builds a unit quaternion per step straight from the *angles* (never from a pre-built matrix),
    runs the pointwise four-leaf prefix scan through :class:`_QuaternionPrefix`, then converts the
    cumulative quaternion to a matrix for this compatibility API. The production B/C path consumes
    the prefix quaternion directly and skips this conversion.

    :class:`_QuaternionPrefix` supplies the analytic backward that makes the pointwise forward
    trainable at production sequence lengths; the prototype scan backward otherwise materializes
    one full copy per timestep and OOMs.

    :param theta: Per-step angles of shape ``(batch, seq_len, n_groups, n_blocks, 3)``.
    :param block_size: The rotation block size; must be 3.

    :returns: Inclusive prefix products of shape ``(batch, seq_len, n_groups, n_blocks, 3, 3)``.
    """
    if block_size != 3:
        # Only the b == 3 quaternion form is written out. A larger block silently handled would
        # truncate the rotation and still return orthogonal-looking output, so refuse it -- exactly
        # as the matrix scans do -- and let the dispatch route b != 3 to the chunked path.
        raise ValueError(
            f"quaternion_cumulative_block_rotation only supports block_size 3, "
            f"got {block_size}; use the chunked path for other block sizes"
        )

    return _quaternion_to_matrix(_quaternion_prefix(_angles_to_quaternion(theta)))


def _quaternion_rotate(q: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply unit-quaternion rotations directly to 3-vectors without building matrices."""
    q_vector = q[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vectors, dim=-1)
    return vectors + q[..., :1] * twice_cross + torch.linalg.cross(q_vector, twice_cross, dim=-1)


def _quaternion_rotate_backward(
    q: torch.Tensor,
    vectors: torch.Tensor,
    grad_rotated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic gradients of :func:`_quaternion_rotate` without a unit-norm approximation."""
    w, u = q[..., :1], q[..., 1:]
    u_dot_v = (u * vectors).sum(dim=-1, keepdim=True)
    grad_dot_u = (grad_rotated * u).sum(dim=-1, keepdim=True)
    grad_dot_v = (grad_rotated * vectors).sum(dim=-1, keepdim=True)

    grad_w = 2.0 * (grad_rotated * torch.linalg.cross(u, vectors, dim=-1)).sum(dim=-1, keepdim=True)
    grad_u = 2.0 * w * torch.linalg.cross(vectors, grad_rotated, dim=-1)
    grad_u = grad_u + 2.0 * (grad_rotated * u_dot_v + vectors * grad_dot_u - 2.0 * u * grad_dot_v)
    grad_q = torch.cat((grad_w, grad_u), dim=-1)
    grad_vectors = _quaternion_rotate(_quaternion_conjugate(q), grad_rotated)
    return grad_q, grad_vectors


def _angles_to_quaternion_backward(
    theta: torch.Tensor,
    grad_q: torch.Tensor,
) -> torch.Tensor:
    """Analytic gradient of :func:`_angles_to_quaternion`, including its small-angle branch."""
    t1, t2, t3 = theta[..., 0], theta[..., 1], theta[..., 2]
    grad_w, grad_x, grad_y, grad_z = grad_q.unbind(dim=-1)
    phi_sq = (theta * theta).sum(dim=-1)
    phi = torch.sqrt(phi_sq.clamp_min(_SMALL_ANGLE_SQ))
    half = phi / 2.0
    small = phi_sq < _SMALL_ANGLE_SQ

    axis_scale = torch.where(
        small,
        0.5 - phi_sq / 48.0 + phi_sq * phi_sq / 3840.0,
        torch.sin(half) / phi,
    )
    w_prime = torch.where(
        small,
        -0.125 + phi_sq / 192.0,
        -torch.sin(half) / (4.0 * phi),
    )
    axis_scale_prime = torch.where(
        small,
        -1.0 / 48.0 + phi_sq / 1920.0,
        (phi * torch.cos(half) - 2.0 * torch.sin(half)) / (4.0 * phi * phi * phi),
    )

    axis_dot_grad = -t3 * grad_x + t2 * grad_y - t1 * grad_z
    radial = 2.0 * (grad_w * w_prime + axis_dot_grad * axis_scale_prime)
    return torch.stack(
        (
            radial * t1 - axis_scale * grad_z,
            radial * t2 + axis_scale * grad_y,
            radial * t3 - axis_scale * grad_x,
        ),
        dim=-1,
    )


def _rotate_bc_quaternion(
    B: torch.Tensor,
    C: torch.Tensor,
    cumulative_q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply cumulative ``Q_t^T`` to B/C directly from prefix quaternions."""
    d_state = B.shape[-1]
    rank = B.shape[-2]
    block_size = 3
    if d_state % block_size != 0:
        raise ValueError(f"d_state ({d_state}) must be divisible by quaternion block size 3")

    both = torch.cat((B, C), dim=-2)
    blocks = both.reshape(*both.shape[:-1], d_state // block_size, block_size)
    # B/C use Q^T. For a unit quaternion, conjugation represents the inverse rotation.
    inverse_q = _quaternion_conjugate(cumulative_q).unsqueeze(-3)
    rotated = _quaternion_rotate(inverse_q, blocks)
    rotated = rotated.reshape(*both.shape[:-1], d_state)
    return rotated[..., :rank, :], rotated[..., rank:, :]


def _fused_quaternion_rotate_bc_forward(
    B: torch.Tensor,
    C: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared arithmetic for the training autograd boundary and eager evaluation path."""
    d_state = B.shape[-1]
    rank = B.shape[-2]
    if d_state % 3 != 0:
        raise ValueError(f"d_state ({d_state}) must be divisible by quaternion block size 3")

    both = torch.cat((B, C), dim=-2)
    vectors = both.reshape(*both.shape[:-1], d_state // 3, 3)
    prefix = _quaternion_prefix_forward(_angles_to_quaternion(theta))
    inverse_q = _quaternion_conjugate(prefix.to(B.dtype)).unsqueeze(-3)
    rotated = _quaternion_rotate(inverse_q, vectors)
    rotated = rotated.reshape(*both.shape[:-1], d_state)
    return rotated[..., :rank, :], rotated[..., rank:, :], prefix


class _FusedQuaternionRotateBC(torch.autograd.Function):
    """Fuse angle conversion, quaternion prefix, and direct B/C rotation."""

    @staticmethod
    def forward(
        ctx,
        B: torch.Tensor,
        C: torch.Tensor,
        theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotated_B, rotated_C, prefix = _fused_quaternion_rotate_bc_forward(B, C, theta)

        # Save input references rather than ``vectors``: the latter aliases ``both``, so retaining
        # it pins the full materialized B/C concatenation until backward. B and C are already live
        # autograd inputs; rebuilding the cheap reshape/concatenation lets compiled forward release
        # (or fuse away) its largest temporary while keeping the expensive quaternion prefix.
        ctx.save_for_backward(B, C, theta, prefix)
        return rotated_B, rotated_C

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        grad_B: torch.Tensor,
        grad_C: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, theta, prefix = ctx.saved_tensors
        rank = B.shape[-2]
        d_state = grad_B.shape[-1]
        both = torch.cat((B, C), dim=-2)
        vectors = both.reshape(*both.shape[:-1], d_state // 3, 3)
        grad_rotated = torch.cat((grad_B, grad_C), dim=-2)
        grad_rotated = grad_rotated.reshape(*grad_rotated.shape[:-1], d_state // 3, 3)

        inverse_q = _quaternion_conjugate(prefix.to(vectors.dtype)).unsqueeze(-3)
        grad_inverse_q, grad_vectors = _quaternion_rotate_backward(
            inverse_q,
            vectors,
            grad_rotated,
        )
        grad_prefix = _quaternion_conjugate(grad_inverse_q.sum(dim=-3)).to(prefix.dtype)
        grad_step_q = _quaternion_prefix_backward(prefix, grad_prefix)
        grad_theta = _angles_to_quaternion_backward(theta, grad_step_q)

        grad_both = grad_vectors.reshape(*grad_B.shape[:-2], 2 * rank, d_state)
        return grad_both[..., :rank, :], grad_both[..., rank:, :], grad_theta


@torch.compiler.disable
def _fused_quaternion_rotate_bc_eval(
    B: torch.Tensor,
    C: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the quaternion scan eagerly so Inductor never lowers ``associative_scan``."""
    rotated_B, rotated_C, _ = _fused_quaternion_rotate_bc_forward(B, C, theta)
    return rotated_B, rotated_C


def _fused_quaternion_rotate_bc(
    B: torch.Tensor,
    C: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run complete b=3 quaternion preprocessing under one analytic-autograd boundary."""
    if not torch.is_grad_enabled():
        # ``torch.compile`` traces a separate no-grad graph for held-out evaluation. Inductor
        # cannot lower the pointwise associative scan in that graph because its symbolic batch
        # size is lifted into the higher-order op. Keep the hot training graph unchanged and
        # cross an eager graph break only for evaluation, which never needs the custom backward.
        return _fused_quaternion_rotate_bc_eval(B, C, theta)
    return _FusedQuaternionRotateBC.apply(B, C, theta)


def _rotate_bc_fused(
    B: torch.Tensor, C: torch.Tensor, cumulative_rot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply ``Q_t^T`` to ``B`` and ``C`` in a single einsum.

    ``B`` and ``C`` are rotated by the same ``Q^T``, so concatenating them along the rank axis
    halves the launch count of this step and gives the matmul twice the work to amortise over.
    The contraction is character-for-character the one in
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api._rotate_bc_blocks`, including the transpose that
    makes ``C~_t^T B~_s`` recover ``R_t ... R_{s+1}``.
    """
    d_state = B.shape[-1]
    rank = B.shape[-2]
    block_size = cumulative_rot.shape[-1]

    both = torch.cat((B, C), dim=-2)
    blocks = both.reshape(*both.shape[:-1], d_state // block_size, block_size)
    rotated = torch.einsum("btgkji,btgrkj->btgrki", cumulative_rot, blocks)
    rotated = rotated.reshape(*both.shape[:-1], d_state)
    return rotated[..., :rank, :], rotated[..., rank:, :]


def _fast_rotate_bc_pair(
    B: torch.Tensor,
    C: torch.Tensor,
    theta: torch.Tensor,
    block_size: int,
    chunk_size: Optional[int],
    *,
    scan_impl: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the cumulative rotation to ``B`` and ``C``.

    The prefix product runs in at least float32 regardless of what ``B``/``C`` are carrying: its
    orthogonality drift is ``O(T * eps)``, which bfloat16 cannot survive (~27% error by T=1024).
    The *application* of that rotation then follows the dtype of ``B``/``C``, which is the
    selective floor -- one rounding, identical to the one the kernel performs on the next line.
    A floor rather than a cast: float64 inputs keep float64 here, which is what lets the scan
    implementations be compared against each other at 1e-8 instead of at float32's ~1e-6, where
    a genuine ordering bug and ordinary rounding are indistinguishable. Every production dtype
    (bfloat16, float16, float32) floors to float32 exactly as before.

    :param scan_impl: Which of :data:`ROTATION_SCAN_IMPLS` to run; ``None`` takes the
        ``MAMBA3_ROTATION_SCAN_IMPL`` default.
    """
    impl = resolve_rotation_scan_impl(scan_impl)
    theta = theta.to(torch.promote_types(theta.dtype, torch.float32))
    if block_size == 2:
        theta_cumulative = torch.cumsum(theta.squeeze(-1) if theta.dim() == 5 else theta, dim=1)
        theta_cumulative = theta_cumulative.to(B.dtype)
        return _rotate_bc(B, theta_cumulative), _rotate_bc(C, theta_cumulative)

    if theta.dim() != 5:
        raise ValueError(
            f"theta must be 5-D (batch, seq_len, n_groups, n_blocks, angles_per_block) "
            f"for block_size={block_size}, got shape {tuple(theta.shape)}"
        )
    if impl == "quaternion" and block_size == 3:
        return _fused_quaternion_rotate_bc(B, C, theta)

    cumulative_rot = fast_cumulative_block_rotation(
        fast_block_rotations(theta, block_size),
        chunk_size=chunk_size,
        scan_impl=impl,
    )
    return _rotate_bc_fused(B, C, cumulative_rot.to(B.dtype))


def mamba3_ssd_fast(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int = 2,
    chunk_size: int = 64,
    rotation_scan_chunk: Optional[int] = None,
    rotation_scan_impl: Optional[str] = None,
    selective_fp32: bool = True,
) -> torch.Tensor:
    """
    Run the Mamba-3 recurrence through the official SISO Triton kernel with a faster ``b >= 3``
    rotation.

    Drop-in for :func:`~olmo_core.nn.mamba3.mamba3_ssd_official.mamba3_ssd_official`: same
    arguments, same semantics, same upstream kernel. See
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.mamba3_ssd_reference` for the argument contract.

    :param chunk_size: Kernel chunk length, passed straight through to the upstream kernel.
    :param rotation_scan_chunk: Sequential-product chunk for the ``b >= 3`` prefix scan. This is
        a *different* knob from ``chunk_size`` and much smaller; ``None`` (the default) picks it
        per sequence length via :func:`_adaptive_scan_chunk`.
    :param rotation_scan_impl: Which of :data:`ROTATION_SCAN_IMPLS` computes the ``b >= 3``
        prefix product. ``None`` (the default) takes the ``MAMBA3_ROTATION_SCAN_IMPL`` default,
        which is what makes this addition a no-op for existing callers. Pass it explicitly to
        get the choice into a config a checkpoint saves and a log line reports -- see
        :func:`resolve_rotation_scan_impl` for what the environment-only version cost.
    :param selective_fp32: Keep the prefix product in float32 but apply the resulting rotation to
        ``B``/``C`` in the kernel's own dtype. ``False`` applies it in float32, matching
        ``mamba3_ssd_official`` to float32 precision at the cost of a wasted upcast.
    """
    # Validate the public option before probing an optional runtime dependency, so a malformed
    # config reports its own actionable error in CPU-only dry runs.
    scan_impl = resolve_rotation_scan_impl(rotation_scan_impl)
    if not fast_mamba3_is_available():
        raise RuntimeError("the official mamba-ssm Mamba-3 SISO kernel is not installed")

    batch, seq_len, n_heads, head_dim = x.shape
    n_groups, rank, d_state = B.shape[2], B.shape[3], B.shape[4]

    if rank != 1:
        raise ValueError(f"the official SISO kernel needs mimo_rank == 1, got {rank}")
    if n_groups * heads_per_group != n_heads:
        raise ValueError(
            f"n_groups ({n_groups}) * heads_per_group ({heads_per_group}) must equal n_heads "
            f"({n_heads})"
        )
    if d_state % block_size != 0:
        raise ValueError(f"d_state ({d_state}) must be divisible by block_size ({block_size})")
    device_type = x.device.type
    autocast_on = torch.is_autocast_enabled(device_type)
    out_dtype = torch.get_autocast_dtype(device_type) if autocast_on else x.dtype

    d_state_padded = kernel_padded_width(d_state)
    head_dim_padded = kernel_padded_width(head_dim)

    # The kernel casts Q/K/V to bfloat16 itself, so there is no accuracy to protect by rotating
    # in float32 -- only the prefix product needs the floor, and `_fast_rotate_bc_pair` keeps it
    # there unconditionally.
    bc_dtype = out_dtype if selective_fp32 else torch.float32

    # Autocast intercepts at the *op* level, so casting the tensors is not enough: the rotation
    # and the discretization coefficients have to be computed with autocast off or they run in
    # bfloat16 regardless.
    with torch.autocast(device_type=device_type, enabled=False):
        key, query = _fast_rotate_bc_pair(
            B.to(bc_dtype),
            C.to(bc_dtype),
            theta,
            block_size,
            rotation_scan_chunk,
            scan_impl=scan_impl,
        )
        # Index the rank axis away rather than `squeeze`: rank is pinned at 1 above.
        key, query = key[:, :, :, 0], query[:, :, :, 0]

        # The kernel wants (batch, nheads, seqlen) for the per-head scalars.
        a_dt = (dt.float() * A.float()).permute(0, 2, 1)
        delta = dt.float().permute(0, 2, 1)
        trap = torch.logit(lam.float(), eps=_LOGIT_EPS).permute(0, 2, 1)

        if d_state_padded != d_state:
            key = F.pad(key, (0, d_state_padded - d_state))
            query = F.pad(query, (0, d_state_padded - d_state))
        value = x if selective_fp32 else x.float()
        if head_dim_padded != head_dim:
            value = F.pad(value, (0, head_dim_padded - head_dim))

    # The reference has no B/C bias, and the kernel adds these *before* its rotation, so they
    # have to be zero rather than merely folded into B/C.
    q_bias = torch.zeros(n_heads, d_state_padded, device=x.device, dtype=torch.float32)
    k_bias = torch.zeros(n_heads, d_state_padded, device=x.device, dtype=torch.float32)
    angles = torch.zeros(
        batch, seq_len, n_heads, _ANGLE_WIDTH, device=x.device, dtype=torch.float32
    )

    y = _mamba3_siso_combined_eager(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        a_dt.contiguous(),
        delta.contiguous(),
        trap.contiguous(),
        q_bias,
        k_bias,
        angles,
        chunk_size=chunk_size,
    )

    if head_dim_padded != head_dim:
        y = y[..., :head_dim]
    return y.to(out_dtype)


# `chunk_simple_gla` accepts the dtypes its Triton kernels are compiled for. float64 reaches this
# path only from a test that forgot `selective_fp32`, and the resulting Triton error names none of
# the arguments the caller actually set.
_SIMPLE_GLA_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _require_simple_gla(x: torch.Tensor, B: torch.Tensor) -> None:
    """Refuse a ``simple_gla`` call this host cannot run, rather than answering with another."""
    if x.is_cuda and B.shape[3] == 1 and simple_gla_is_available():
        return
    raise RuntimeError(
        "the simple_gla backend cannot run this call: it needs CUDA (got "
        f"{x.device.type}), mimo_rank == 1 (got {B.shape[3]}), and an installed "
        "flash-linear-attention. Naming a backend is a strict request, so this raises rather "
        "than quietly running official_fast or the chunked form."
    )


def _simple_gla_trapezoidal_terms(
    dt: torch.Tensor, A: torch.Tensor, lam: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the exponential-trapezoidal discretization into simple-GLA's three per-head scalars.

    Unrolling the reference recurrence and reindexing the trapezoidal term by one step (the same
    derivation :mod:`~olmo_core.nn.mamba3.mamba3_ssd_official` writes out for the upstream
    kernel) gives::

        y_t = sum_{s<t} exp(L_t - L_s) scale_s <q_t, k_s> x_s  +  gamma_t <q_t, k_t> x_t
        gamma_s = lam_s dt_s          scale_s = gamma_s + (1 - lam_{s+1}) dt_{s+1}

    ``fla``'s simple GLA computes ``S_t = exp(g_t) S_{t-1} + k_t (x) v_t``, ``o_t = q_t S_t``,
    i.e. ``o_t = sum_{s<=t} exp(L_t - L_s) <q_t, k_s> v_s`` once ``g_t = dt_t A``. So ``scale_s``
    folds into the values and the whole difference is the diagonal, where the scan carries
    ``scale_t`` and the reference wants ``gamma_t``.

    Nothing here divides. That is the reason this backend uses ``chunk_simple_gla`` rather than
    the pinned package's own ``mamba_chunk_scan_combined``: that kernel drives the decay and the
    input scale from one ``dt``, so the only fold available is ``x * (scale / dt)``, and ``dt``
    is a ``softplus`` output that reaches exactly ``0.0`` in float32 (see
    ``test_trapezoidal_fold_needs_no_division_by_dt``).

    :param dt: Discretization step, shape ``(batch, seq_len, n_heads)``.
    :param A: Per-head log-decay, shape ``(n_heads,)``.
    :param lam: Trapezoidal mixing coefficient, shape ``(batch, seq_len, n_heads)``.

    :returns: ``(g, v_scale, diag_scale)``, each float32 and shaped like ``dt``. ``g`` is the
        per-step log decay, ``v_scale`` multiplies the values, and ``diag_scale`` weights the
        ``s == t`` correction that turns ``scale_t`` back into ``gamma_t``.
    """
    dt = dt.float()
    lam = lam.float()
    trailing = (1.0 - lam) * dt
    # The last position has no successor and the reference reads it through gamma alone, so the
    # shifted term is zero there -- which the same zero-fill also makes true of the correction.
    carried = F.pad(trailing[:, 1:], (0, 0, 0, 1))
    return dt * A.float(), lam * dt + carried, -carried


def _simple_gla_operands(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int = 2,
    rotation_scan_chunk: Optional[int] = None,
    scan_impl: Optional[str] = None,
    bc_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build everything :func:`mamba3_ssd_simple_gla` hands to ``chunk_simple_gla``, plus the
    diagonal correction added to its output.

    Pure PyTorch and device-agnostic on purpose: the trapezoidal fold is the only genuinely new
    algebra in this backend, and keeping it here lets a CPU test check it against
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.mamba3_ssd_reference` without a GPU or a Triton
    kernel. The rotation is the *same* ``_fast_rotate_bc_pair`` the official-kernel path uses, so
    ``rotation_scan_impl`` -- quaternion included -- behaves identically on both backends.

    :param bc_dtype: Dtype the rotation is applied in and the kernel operands are built in;
        ``None`` follows ``x``.

    :returns: ``(query, key, value, g, correction)``. ``query``/``key`` are per-head
        ``(batch, seq_len, n_heads, d_state)``, ``value`` is
        ``(batch, seq_len, n_heads, head_dim)``, ``g`` is float32
        ``(batch, seq_len, n_heads)``, and ``correction`` is float32 and shaped like the output.
    """
    n_heads = x.shape[2]
    n_groups, rank, d_state = B.shape[2], B.shape[3], B.shape[4]

    if rank != 1:
        raise ValueError(f"the simple_gla backend needs mimo_rank == 1, got {rank}")
    if n_groups * heads_per_group != n_heads:
        raise ValueError(
            f"n_groups ({n_groups}) * heads_per_group ({heads_per_group}) must equal n_heads "
            f"({n_heads})"
        )
    if d_state % block_size != 0:
        raise ValueError(f"d_state ({d_state}) must be divisible by block_size ({block_size})")

    bc_dtype = x.dtype if bc_dtype is None else bc_dtype

    # Autocast intercepts at the op level, so the rotation and the discretization have to be
    # computed with it off or they run in bfloat16 whatever dtype they were handed.
    with torch.autocast(device_type=x.device.type, enabled=False):
        key, query = _fast_rotate_bc_pair(
            B.to(bc_dtype),
            C.to(bc_dtype),
            theta,
            block_size,
            rotation_scan_chunk,
            scan_impl=scan_impl,
        )
        # Index the rank axis away rather than `squeeze`: rank is pinned at 1 above.
        key, query = key[:, :, :, 0], query[:, :, :, 0]
        # Take the diagonal before broadcasting groups to heads -- it is a group quantity, and
        # <q_t, k_t> from the rounded operands is exactly what the scan puts on its own diagonal.
        diagonal = (query.float() * key.float()).sum(-1)
        if heads_per_group != 1:
            key = key.repeat_interleave(heads_per_group, dim=2)
            query = query.repeat_interleave(heads_per_group, dim=2)
            diagonal = diagonal.repeat_interleave(heads_per_group, dim=2)

        g, v_scale, diag_scale = _simple_gla_trapezoidal_terms(dt, A, lam)
        # Scale in float32 and round once. Rounding `v_scale` first instead would put a second
        # bf16 rounding straight onto the recurrence's input weight, which the official path
        # never pays -- it hands the kernel `delta` and `trap` in float32 and forms the
        # trapezoidal coefficients internally.
        x_fp32 = x.float()
        value = (x_fp32 * v_scale.unsqueeze(-1)).to(bc_dtype)
        correction = (diag_scale * diagonal).unsqueeze(-1) * x_fp32

    return query, key, value, g, correction


def mamba3_ssd_simple_gla(
    x: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    lam: torch.Tensor,
    theta: torch.Tensor,
    *,
    heads_per_group: int,
    block_size: int = 2,
    rotation_scan_chunk: Optional[int] = None,
    rotation_scan_impl: Optional[str] = None,
    selective_fp32: bool = True,
) -> torch.Tensor:
    """
    Run the Mamba-3 recurrence through ``fla``'s chunked simple-GLA kernel.

    Drop-in for :func:`mamba3_ssd_fast`: same arguments, same semantics, different scan. See
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.mamba3_ssd_reference` for the argument contract
    and :func:`_simple_gla_trapezoidal_terms` for the fold that makes the substitution exact.

    The motivation is occupancy, not arithmetic. ``mamba3_siso_combined`` grids over
    ``(nheads, batch)``, which at the 370M arm's geometry is 32 thread blocks against an A100's
    108 SMs, each running a 64-iteration serial chunk loop; ``chunk_simple_gla`` grids over chunk
    tiles as well. It also masks the state dimension, so ``d_state=96`` runs unpadded instead of
    at the power-of-two width :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.kernel_padded_width`
    forces on the official path. Against that, ``B``/``C`` have to be broadcast to heads because
    the kernel has no GQA, which the official path gets for free.

    Whether any of that is a whole-model win is a measurement, not a claim: this backend is
    opt-in and nothing selects it by default.

    :param rotation_scan_chunk: Sequential-product chunk for the ``b >= 3`` prefix scan; ``None``
        picks it per sequence length via :func:`_adaptive_scan_chunk`.
    :param rotation_scan_impl: Which of :data:`ROTATION_SCAN_IMPLS` computes the ``b >= 3``
        prefix product. Reaches the same :func:`_fast_rotate_bc_pair` as the official path.
    :param selective_fp32: Keep the prefix product in float32 but build the kernel operands in
        the ambient dtype. ``False`` builds them in float32.

    :raises RuntimeError: If CUDA, ``mimo_rank == 1`` or ``flash-linear-attention`` is missing.
        A named backend is a strict request and is never silently downgraded.
    """
    # Validate the public option before probing an optional runtime dependency, so a malformed
    # config reports its own actionable error in CPU-only dry runs.
    scan_impl = resolve_rotation_scan_impl(rotation_scan_impl)
    _require_simple_gla(x, B)

    device_type = x.device.type
    autocast_on = torch.is_autocast_enabled(device_type)
    out_dtype = torch.get_autocast_dtype(device_type) if autocast_on else x.dtype
    kernel_dtype = out_dtype if selective_fp32 else torch.float32
    if kernel_dtype not in _SIMPLE_GLA_DTYPES:
        raise ValueError(
            f"the simple_gla backend runs in {_SIMPLE_GLA_DTYPES}, got {kernel_dtype}; pass "
            "selective_fp32=False to run a float64 call in float32"
        )

    query, key, value, g, correction = _simple_gla_operands(
        x,
        B,
        C,
        dt,
        A,
        lam,
        theta,
        heads_per_group=heads_per_group,
        block_size=block_size,
        rotation_scan_chunk=rotation_scan_chunk,
        scan_impl=scan_impl,
        bc_dtype=kernel_dtype,
    )

    # Imported here rather than at module scope: `fla` is an optional dependency, and resolving
    # the symbol per call is what lets the tests patch it.
    from fla.ops.simple_gla import chunk_simple_gla

    # `scale` defaults to `K ** -0.5` inside fla, which this recurrence does not want.
    output, _ = chunk_simple_gla(q=query, k=key, v=value, g=g, scale=1.0)
    return (output.float() + correction).to(out_dtype)


def fast_rotation_speedup_note() -> str:
    """Human-readable summary of what this module changes, for diagnostics and logging."""
    return (
        f"mamba3_ssd_fast: Rodrigues so(3) closed form at b==3, adaptive prefix-product scan "
        f"chunk (~T/128 in [{_ROTATION_SCAN_CHUNK_MIN}, {_ROTATION_SCAN_CHUNK_MAX}] vs a fixed "
        f"64), fused B/C rotation einsum, selective fp32 floor"
    )
