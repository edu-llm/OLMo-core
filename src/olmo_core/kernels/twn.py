"""
Fused TWN ternarization.

The reference implementation in :mod:`olmo_core.nn.quantization` expresses the quantizer as a
sequence of whole-tensor PyTorch ops. That is the right way to *define* it and the wrong way to
run it: it casts the weight to float32 (doubling every byte moved), materializes ``abs``, a
boolean mask and ``absw * mask`` as separate full-size temporaries, and then recomputes the cast,
the sign and the comparison a second time to emit the result. Measured on an RTX 5050, that runs
at 2.33 G elements/s against a ~42 G elements/s bandwidth ceiling.

This kernel does the same arithmetic in three streaming reads and one write, keeping the row
statistics in registers. The reduction still accumulates in float32, which
:func:`~olmo_core.nn.quantization.twn_threshold_and_scale` documents as load-bearing rather than
cosmetic: a bf16 sum over a 1024-wide row moves ``delta`` far enough to change which weights
survive, and which weights survive is the quantizer's identity.

Results are not bitwise identical to the reference. Triton's sequential accumulation and torch's
pairwise ``mean`` differ in the last bits of ``delta``, which can flip a weight sitting within a
float32 epsilon of the threshold. On real weights such ties do not occur; the tests assert an
exact ternary pattern match and a float32-tolerance match on the values.
"""

import math
from typing import Optional, Tuple

import torch

__all__ = ["fused_twn_available", "fused_twn_quantize"]

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only where triton is absent
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _twn_quantize_kernel(
        w_ptr,
        out_ptr,
        n_reduce,
        n_trail,
        delta_factor,
        BLOCK_R: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        """
        Ternarize one ``(n_reduce, BLOCK_B)`` slab of a weight viewed as ``(A, R, B)``.

        Program ``(a, b_tile)`` owns the rows selected by that trailing tile and reduces over
        ``R``. Laying the tile across the trailing axis is what keeps the loads coalesced when
        the reduced axis is strided, which is the case for four of the six stacked expert
        weights; when the reduced axis is already innermost the caller passes ``BLOCK_B = 1``
        and the tile runs along ``R`` instead.
        """
        pid_a = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        valid_b = offs_b < n_trail
        base = pid_a * n_reduce * n_trail

        # Pass 1: mean|W| over the reduced axis gives the threshold.
        abs_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for r0 in tl.range(0, n_reduce, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            tile = (offs_r[:, None] < n_reduce) & valid_b[None, :]
            w = tl.load(
                w_ptr + base + offs_r[:, None] * n_trail + offs_b[None, :],
                mask=tile,
                other=0.0,
            ).to(tl.float32)
            abs_sum += tl.sum(tl.abs(w), axis=0)
        delta = delta_factor * abs_sum / n_reduce

        # Pass 2: alpha is the mean magnitude of the weights that clear the threshold.
        surviving_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
        surviving_count = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for r0 in tl.range(0, n_reduce, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            tile = (offs_r[:, None] < n_reduce) & valid_b[None, :]
            w = tl.load(
                w_ptr + base + offs_r[:, None] * n_trail + offs_b[None, :],
                mask=tile,
                other=0.0,
            ).to(tl.float32)
            magnitude = tl.abs(w)
            keeps = magnitude > delta[None, :]
            surviving_sum += tl.sum(tl.where(keeps, magnitude, 0.0), axis=0)
            surviving_count += tl.sum(tl.where(keeps, 1.0, 0.0), axis=0)
        # An all-zero row has delta = 0, clears nothing, and must yield alpha = 0 rather than
        # a division by zero.
        alpha = surviving_sum / tl.maximum(surviving_count, 1.0)

        # Pass 3: emit alpha * sign(W) * 1[|W| > delta].
        for r0 in tl.range(0, n_reduce, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            tile = (offs_r[:, None] < n_reduce) & valid_b[None, :]
            offsets = base + offs_r[:, None] * n_trail + offs_b[None, :]
            w = tl.load(w_ptr + offsets, mask=tile, other=0.0).to(tl.float32)
            magnitude = tl.abs(w)
            signed = tl.where(w > 0, alpha[None, :], -alpha[None, :])
            tl.store(
                out_ptr + offsets,
                tl.where(magnitude > delta[None, :], signed, 0.0),
                mask=tile,
            )


def fused_twn_available(w: torch.Tensor) -> bool:
    """
    Whether the fused kernel can run on ``w``.

    :param w: The latent weight.

    :returns: ``True`` when Triton is importable and the tensor is on CUDA.
    """
    return _HAS_TRITON and w.is_cuda


def _block_shape(n_trail: int) -> Tuple[int, int]:
    """
    Choose a tile that keeps the loads coalesced.

    With the reduced axis innermost (``n_trail == 1``) consecutive lanes must run along the
    reduction; otherwise they must run along the trailing axis, which is the contiguous one.
    """
    if n_trail == 1:
        return 1024, 1
    block_b = min(128, 1 << int(math.ceil(math.log2(n_trail))))
    return max(1, 2048 // block_b), block_b


def fused_twn_quantize(
    w: torch.Tensor, *, in_dim: int, delta_factor: Optional[float] = None
) -> Optional[torch.Tensor]:
    """
    Ternarize ``w`` with the TWN rule in a single fused kernel.

    Matches :func:`olmo_core.nn.quantization.twn_quantize` up to float32 reduction order.

    :param w: The latent weight.
    :param in_dim: The input-feature axis to reduce over. May be negative; callers routinely
        pass ``-1`` for a 2-D :class:`torch.nn.Linear` weight.
    :param delta_factor: The threshold constant, defaulting to TWN's ``0.7``.

    :returns: The quantized weight, or ``None`` if the fused path does not apply, in which case
        the caller should fall back to the reference implementation.
    """
    if not fused_twn_available(w):
        return None

    from ..nn.quantization import TWN_DELTA_FACTOR

    if delta_factor is None:
        delta_factor = TWN_DELTA_FACTOR
    source = w.detach().contiguous()
    axis = in_dim % source.ndim
    n_reduce = source.shape[axis]
    n_lead = math.prod(source.shape[:axis])
    n_trail = math.prod(source.shape[axis + 1 :])
    if n_reduce == 0 or n_lead == 0 or n_trail == 0:
        return torch.zeros_like(source)

    out = torch.empty_like(source)
    block_r, block_b = _block_shape(n_trail)
    _twn_quantize_kernel[(n_lead, triton.cdiv(n_trail, block_b))](
        source.view(n_lead, n_reduce, n_trail),
        out.view(n_lead, n_reduce, n_trail),
        n_reduce,
        n_trail,
        delta_factor,
        BLOCK_R=block_r,
        BLOCK_B=block_b,
    )
    return out
