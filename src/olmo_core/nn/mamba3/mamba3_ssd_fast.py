"""
Performance-tuned copy of :mod:`olmo_core.nn.mamba3.mamba3_ssd_official`, specialised for the
``block_size >= 3`` (NC^1) configuration.

:mod:`mamba3_ssd_official` is left byte-identical and remains the correctness oracle. This
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
3. **One fused rotation einsum for** ``B`` **and** ``C`` instead of two, since both are rotated
   by the same ``Q^T``.
4. **A selective fp32 floor** (``selective_fp32``). The float32 requirement is on the *prefix
   product*, where orthogonality drift accumulates as ``O(T * eps)``. Applying the resulting
   rotation to ``B``/``C`` in float32 is wasted work, because ``mamba3_siso_combined`` hard-casts
   ``Q``/``K`` to bfloat16 on the next line anyway. This keeps the floor where it earns its
   keep and drops it where it does not.

Rodrigues also removes a latent hazard rather than just being faster: ``torch.matrix_exp``
accepts bfloat16 and returns silent ``NaN``/``Inf`` rather than raising, so the "it crashes
loudly" guard assumed elsewhere does not exist. The closed form stays finite in bfloat16.
"""

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
    "fast_block_rotations",
    "fast_cumulative_block_rotation",
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


def _adaptive_scan_chunk(seq_len: int) -> int:
    """
    Pick the prefix-scan chunk for a sequence of length ``seq_len``.

    The chunk decides only how the associative product is bracketed, never the result (guaranteed
    by ``test_prefix_scan_is_invariant_to_scan_chunk``), so it is free to tune purely for speed. A
    longer sequence wants a coarser chunk -- fewer, larger batched block-diagonal matmuls in place
    of more Hillis-Steele levels -- which is the ``(d)`` "batch harder" lever measured for ``b=3``.
    ``~T/128`` clamped to ``[8, 64]`` matches the swept optimum and degrades gracefully off it.

    :param seq_len: The sequence length the scan will run over.

    :returns: A chunk length in ``[8, 64]``.
    """
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
    rot: torch.Tensor, chunk_size: Optional[int] = None
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

    :returns: Inclusive prefix products, same shape as ``rot``.
    """
    from .mamba3_ssd_api import _cumulative_block_rotation

    if chunk_size is None:
        chunk_size = _adaptive_scan_chunk(rot.shape[1])
    return _cumulative_block_rotation(rot, chunk_size=chunk_size)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the cumulative rotation to ``B`` and ``C``.

    The prefix product runs in float32 regardless of what ``B``/``C`` are carrying: its
    orthogonality drift is ``O(T * eps)``, which bfloat16 cannot survive (~27% error by T=1024).
    The *application* of that rotation then follows the dtype of ``B``/``C``, which is the
    selective floor -- one rounding, identical to the one the kernel performs on the next line.
    """
    theta = theta.float()
    if block_size == 2:
        theta_cumulative = torch.cumsum(theta.squeeze(-1) if theta.dim() == 5 else theta, dim=1)
        theta_cumulative = theta_cumulative.to(B.dtype)
        return _rotate_bc(B, theta_cumulative), _rotate_bc(C, theta_cumulative)

    if theta.dim() != 5:
        raise ValueError(
            f"theta must be 5-D (batch, seq_len, n_groups, n_blocks, angles_per_block) "
            f"for block_size={block_size}, got shape {tuple(theta.shape)}"
        )
    cumulative_rot = fast_cumulative_block_rotation(
        fast_block_rotations(theta, block_size), chunk_size=chunk_size
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
    :param selective_fp32: Keep the prefix product in float32 but apply the resulting rotation to
        ``B``/``C`` in the kernel's own dtype. ``False`` applies it in float32, matching
        ``mamba3_ssd_official`` to float32 precision at the cost of a wasted upcast.
    """
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
            B.to(bc_dtype), C.to(bc_dtype), theta, block_size, rotation_scan_chunk
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


def fast_rotation_speedup_note() -> str:
    """Human-readable summary of what this module changes, for diagnostics and logging."""
    return (
        f"mamba3_ssd_fast: Rodrigues so(3) closed form at b==3, adaptive prefix-product scan "
        f"chunk (~T/128 in [{_ROTATION_SCAN_CHUNK_MIN}, {_ROTATION_SCAN_CHUNK_MAX}] vs a fixed "
        f"64), fused B/C rotation einsum, selective fp32 floor"
    )
