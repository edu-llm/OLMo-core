"""
Adapter onto the official ``mamba-ssm`` Mamba-3 SISO Triton kernel.

The upstream kernel (``mamba_ssm.ops.triton.mamba3.mamba3_siso_combined``) is treated as
correct and is never modified. Everything here is the mapping from this repo's tensor layout
and discretization onto its signature.

Deriving the mapping
--------------------
Reading ``mamba3_siso_fwd`` the kernel computes, per head, with ``L_t = sum_{r<=t} ADT_r``::

    y_t = sum_{s<t} exp(L_t - L_s) scale_s <q~_t, k~_s> V_s  +  gamma_t <q~_t, k~_t> V_t
    gamma_s = DT_s sigmoid(Trap_s)
    scale_s = gamma_s + DT_{s+1} (1 - sigmoid(Trap_{s+1}))

with ``q~``/``k~`` the rotary-rotated ``Q + Q_bias`` / ``K + K_bias``. Unrolling the
reference recurrence ``h_t = alpha_t h_{t-1} + gamma_t v_t + beta_t v_{t-1}`` and reindexing
the trapezoidal term by one step gives exactly the same expression with
``gamma_s = lam_s dt_s`` and ``dt_{s+1}(1 - lam_{s+1})``: the ``alpha_{s+1}`` inside
``beta_{s+1} = (1 - lam_{s+1}) dt_{s+1} alpha_{s+1}`` cancels against the extra step of decay
picked up by shifting ``exp(L_t - L_{s+1})`` to ``exp(L_t - L_s)``. So the two agree term by
term under::

    Q <- C (pre-rotated)   K <- B (pre-rotated)   V <- x
    ADT <- dt * A          DT <- dt               Trap <- logit(lam)
    Q_bias = K_bias = 0    D = Z = None           Angles = 0

``Trap`` is a *pre-sigmoid logit*, not the mixing coefficient itself -- the kernel applies
``sigmoid`` to it. The biases must be zero because the reference has no ``B``/``C`` bias, and
the kernel adds them before the rotation, so a nonzero bias would also be rotated.

Why ``Angles`` must be zero
---------------------------
The docstring upstream says it computes ``cumsum(Angles * DT)``, but ``angle_dt_fwd``
actually computes ``cumsum(tanh(Angles) * pi * DT) mod 2pi``. The exact inverse is therefore
``Angles = atanh(theta / (pi * dt))``, which exists only while ``|theta| < pi * dt``. This
repo's ``theta`` is an unconstrained ``nn.Linear`` output, and its rotation is shared by a
whole ``(B, C)`` group while ``DT`` is per head, so the native path cannot express it in
general. Passing ``Angles = 0`` makes the internal rotation the identity (``tanh(0) = 0``,
and the ``mod 2pi`` fixes zero), which degenerates the kernel into a pure scalar-decay SSD
scan over whatever ``Q``/``K`` it is handed. The rotation is then applied here, by the same
helpers the chunked path uses, which is also what lets ``block_size >= 3`` reuse the kernel:
the non-abelian ``SO(b)`` prefix product is pure ``B``/``C`` preprocessing and the scan never
sees anything but a per-head scalar decay.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

from .mamba3_ssd_api import (
    _block_rotations,
    _cumulative_block_rotation,
    _mamba3_siso_combined_eager,
    _rotate_bc,
    _rotate_bc_blocks,
    apply_partial_rotation,
    has_mamba3,
    kernel_padded_width,
)

__all__ = ["official_mamba3_is_available", "mamba3_ssd_official"]

# The kernel's `tl.dot` reductions need a contraction dimension of at least 16 (a narrower
# `headdim_qk` fails to compile), and TMA wants a power-of-two block, so both head dimensions
# are zero-padded up to a power-of-two floor by `kernel_padded_width` (imported above, the single
# source of the padding rule). Padding is exactly neutral: a zero column of `B` never enters the
# state and the matching zero column of `C` never reads it, and a zero channel of `V` produces a
# zero output channel that is sliced back off.

# `Angles` only has to be even and no wider than `headdim_qk // 2`; anything beyond it is left
# unrotated, which for zero angles is the same identity. Two is the smallest legal width.
_ANGLE_WIDTH = 2

# `lam` reaches the mixer as `sigmoid(...)` so it is open on (0, 1), but the tests feed
# `torch.rand`, which can return an exact 0 and send `logit` to -inf.
_LOGIT_EPS = 1e-6


@lru_cache(maxsize=1)
def official_mamba3_is_available() -> bool:
    """Whether the official Mamba-3 SISO Triton entry point can be imported."""
    if not has_mamba3():
        return False
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (  # noqa: F401
            mamba3_siso_combined,
        )
    except Exception:
        return False
    return True


def _rotate_bc_pair(
    B: torch.Tensor, C: torch.Tensor, theta: torch.Tensor, block_size: int
) -> tuple:
    """Apply the cumulative rotation to ``B`` and ``C``, matching the reference exactly.

    Caller must already have autocast disabled: the ``b >= 3`` prefix product is a chain of
    ``T`` matmuls whose orthogonality drift is ``O(T * eps)``, which bf16 cannot survive.
    """
    theta = theta.float()

    def _rotate(B_in: torch.Tensor, C_in: torch.Tensor, angles: torch.Tensor):
        if block_size == 2:
            cumulative = torch.cumsum(angles.squeeze(-1) if angles.dim() == 5 else angles, dim=1)
            return _rotate_bc(B_in, cumulative), _rotate_bc(C_in, cumulative)

        if angles.dim() != 5:
            raise ValueError(
                f"theta must be 5-D (batch, seq_len, n_groups, n_blocks, angles_per_block) "
                f"for block_size={block_size}, got shape {tuple(angles.shape)}"
            )
        cumulative_rot = _cumulative_block_rotation(_block_rotations(angles, block_size))
        return _rotate_bc_blocks(B_in, cumulative_rot), _rotate_bc_blocks(C_in, cumulative_rot)

    # ``theta`` may cover only the leading blocks; the identity tail is left as it arrived.
    return apply_partial_rotation(B, C, theta, block_size, _rotate)


def mamba3_ssd_official(
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
) -> torch.Tensor:
    """
    Run the Mamba-3 recurrence through the official SISO Triton kernel.

    Arguments and semantics match :func:`~olmo_core.nn.mamba3.mamba3_ssd_api
    .mamba3_ssd_reference` exactly; see its docstring. SISO only, so ``B``/``C`` must have
    ``rank == 1``.

    The kernel hard-casts ``Q``/``K``/``V``/``Trap`` to bfloat16 internally, so this path is
    bf16-accurate no matter what dtype it is handed. The returned dtype still follows the
    ambient autocast (or the input dtype) for interface parity with the other two entry
    points; it does not imply fp32 accuracy.

    :param chunk_size: Kernel chunk length. ``64`` is the value upstream tunes for, and is
        unrelated to the chunked PyTorch form's much larger chunk.
    """
    if not official_mamba3_is_available():
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

    # Autocast intercepts at the *op* level, so casting the tensors to fp32 is not enough: the
    # rotation and the discretization coefficients have to be computed with autocast off or
    # they run in bf16 anyway. See `mamba3_ssd_chunked` for the same guard and why the b >= 3
    # prefix product in particular cannot survive bf16.
    with torch.autocast(device_type=device_type, enabled=False):
        key, query = _rotate_bc_pair(B.float(), C.float(), theta, block_size)
        # Index the rank axis away rather than `squeeze`: rank is pinned at 1 above.
        key, query = key[:, :, :, 0], query[:, :, :, 0]

        # The kernel wants (batch, nheads, seqlen) for the per-head scalars.
        a_dt = (dt.float() * A.float()).permute(0, 2, 1)
        delta = dt.float().permute(0, 2, 1)
        trap = torch.logit(lam.float(), eps=_LOGIT_EPS).permute(0, 2, 1)

        if d_state_padded != d_state:
            key = F.pad(key, (0, d_state_padded - d_state))
            query = F.pad(query, (0, d_state_padded - d_state))
        value = x.float()
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
