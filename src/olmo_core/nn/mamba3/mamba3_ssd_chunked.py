"""
Chunkwise-parallel (SSD) form of the Mamba-3 recurrence.

:func:`mamba3_ssd_reference` walks the sequence one timestep at a time, which costs
``O(seq_len)`` dependent kernel launches and retains one state tensor per timestep for
autograd. This module computes the same function as a small number of matrix multiplies.

The identity that makes this possible is that the state transition is a per-head **scalar**
``alpha_t = exp(dt_t * A)`` -- the rotation is folded into ``B``/``C`` beforehand, never into
the state -- so the transfer from step ``s`` to step ``t`` is ``exp(L_t - L_s)`` with
``L_t = sum_{r<=t} log alpha_r``. Unrolling ``h_t = alpha_t h_{t-1} + gamma_t v_t + beta_t
v_{t-1}`` with ``v_t = B_t (outer) x_t`` and reading out with ``C_t`` gives

.. code-block::

    y_t = sum_{s<=t} exp(L_t - L_s) gamma_s (C_t . B_s)     x_s
        + sum_{s<=t} exp(L_t - L_s) beta_s  (C_t . B_{s-1}) x_{s-1}

i.e. two independent scalar-decay SSD passes over the same ``alpha`` and ``C``, the second
reading a right-shifted ``(B, x)``. The trapezoidal term therefore costs a shift, not a
sequential dependency.

Because the rotation is pure preprocessing, this form is agnostic to ``block_size``: it works
unchanged for the abelian ``SO(2)`` default and for non-abelian ``SO(b >= 3)`` blocks.
"""

from typing import Optional

import torch

from .mamba3_ssd_api import _rotate_bc

# The rotation is identical in both fast paths, so it is imported rather than duplicated. No
# cycle: `mamba3_ssd_fast` imports only from `mamba3_ssd_api`, and `mamba3_ssd_api` imports both
# this module and that one function-locally.
from .mamba3_ssd_fast import (
    _rotate_bc_fused,
    fast_block_rotations,
    fast_cumulative_block_rotation,
)

__all__ = ["mamba3_ssd_chunked"]


def _shift_right(t: torch.Tensor) -> torch.Tensor:
    """Shift along the sequence dim, inserting zeros at ``t == 0`` (``v_{-1} == 0``)."""
    return torch.cat((torch.zeros_like(t[:, :1]), t[:, :-1]), dim=1)


def _ssd_scalar_decay(
    log_alpha: torch.Tensor,
    coef: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Compute ``y_t = sum_{s<=t} exp(L_t - L_s) * coef_s * (C_t . B_s) * x_s`` chunkwise.

    :param log_alpha: Per-step log decay (non-positive), shape ``(batch, seq_len, n_heads)``.
    :param coef: Per-step input scaling, shape ``(batch, seq_len, n_heads)``.
    :param B: State-input projection, shape ``(batch, seq_len, n_heads, d_flat)``.
    :param C: State-output projection, shape ``(batch, seq_len, n_heads, d_flat)``.
    :param x: Values, shape ``(batch, seq_len, n_heads, head_dim)``.
    :param chunk_size: Chunk length ``Q``. Larger trades more intra-chunk work for fewer
        sequential steps.

    :returns: Shape ``(batch, seq_len, n_heads, head_dim)``.
    """
    batch, seq_len, n_heads, d_flat = B.shape
    head_dim = x.shape[-1]

    # Never pad past the sequence: a chunk larger than the input is pure wasted work.
    chunk_size = min(chunk_size, max(seq_len, 1))
    n_chunks = (seq_len + chunk_size - 1) // chunk_size
    pad = n_chunks * chunk_size - seq_len
    if pad:
        # log_alpha pads with 0 (decay 1) and coef with 0, so padding contributes nothing to
        # the output and passes the carried state through untouched.
        log_alpha = torch.nn.functional.pad(log_alpha, (0, 0, 0, pad))
        coef = torch.nn.functional.pad(coef, (0, 0, 0, pad))
        B = torch.nn.functional.pad(B, (0, 0, 0, 0, 0, pad))
        C = torch.nn.functional.pad(C, (0, 0, 0, 0, 0, pad))
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))

    shape = (batch, n_chunks, chunk_size)
    log_alpha = log_alpha.view(*shape, n_heads)
    coef = coef.view(*shape, n_heads)
    B = B.view(*shape, n_heads, d_flat)
    C = C.view(*shape, n_heads, d_flat)
    x = x.view(*shape, n_heads, head_dim)

    # Inclusive within-chunk cumulative log decay, laid out head-major for the masks below.
    cum = torch.cumsum(log_alpha, dim=2).permute(0, 1, 3, 2)  # (b, nc, H, Q)

    # Intra-chunk: a causal, decay-weighted quadratic form -- one (Q x Q) matmul per chunk.
    scores = torch.einsum("bcqhk,bcshk->bchqs", C, B)
    decay = cum.unsqueeze(-1) - cum.unsqueeze(-2)  # [..., t, s] = L_t - L_s
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=B.device).tril()
    # Mask BEFORE the exponential, not after. Above the diagonal ``L_t - L_s`` is positive and
    # grows with |A| * dt * chunk_size, so at realistic decay (|A| ~ 16) it overflows to inf.
    # Discarding those with a post-hoc ``where`` hides it in the forward but leaves the
    # backward computing ``0 * inf`` -> NaN. ``exp(-inf)`` is exactly 0 with a zero gradient.
    weight = torch.exp(decay.masked_fill(~causal, float("-inf")))
    weight = weight * coef.permute(0, 1, 3, 2).unsqueeze(-2)  # scale by coef_s
    y = torch.einsum("bchqs,bcshp->bcqhp", scores * weight, x)

    # Each chunk's own contribution to the state carried at its final position.
    to_end = torch.exp(cum[..., -1:] - cum).permute(0, 1, 3, 2) * coef  # (b, nc, Q, H)
    states = torch.einsum("bcshk,bcshp->bchkp", B * to_end.unsqueeze(-1), x)

    # Sequential only across chunks: seq_len / chunk_size steps instead of seq_len.
    chunk_decay = torch.exp(cum[..., -1])  # (b, nc, H)
    # The carry is a recurrence across the whole sequence, so it accumulates in fp32 even when
    # the matmuls feeding it are reduced precision -- the same reason recurrent state is the
    # one thing not to quantize.
    carry = torch.zeros((batch, n_heads, d_flat, head_dim), dtype=torch.float32, device=B.device)
    from_start = torch.exp(cum).permute(0, 1, 3, 2)  # (b, nc, Q, H)
    inter = []
    for c in range(n_chunks):
        inter.append(
            torch.einsum("bqhk,bhkp->bqhp", C[:, c] * from_start[:, c].unsqueeze(-1), carry)
        )
        carry = chunk_decay[:, c].unsqueeze(-1).unsqueeze(-1) * carry + states[:, c].float()
    y = y + torch.stack(inter, dim=1).to(y.dtype)

    y = y.reshape(batch, n_chunks * chunk_size, n_heads, head_dim)
    return y[:, :seq_len] if pad else y


def mamba3_ssd_chunked(
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
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Chunkwise-parallel equivalent of :func:`mamba3_ssd_reference`.

    Arguments and semantics match the reference exactly; see its docstring. The only addition
    is ``chunk_size``.

    :param chunk_size: Sequence chunk length ``Q``, clamped to ``seq_len``. Cost is
        ``O(seq_len * Q)`` intra-chunk and ``O(seq_len / Q)`` sequential steps, so ``Q`` trades
        arithmetic against latency. The default was measured at the 1B preset's Mamba
        dimensions: ``256`` beat ``64`` by 1.7-4.6x across sequence lengths 512-4096.

    :returns: The SSM output, shape ``(batch, seq_len, n_heads, head_dim)``.
    """
    n_heads = x.shape[2]
    device_type = x.device.type

    orig_dtype = x.dtype
    autocast_on = torch.is_autocast_enabled(device_type)
    out_dtype = torch.get_autocast_dtype(device_type) if autocast_on else orig_dtype

    # Autocast intercepts at the *op* level, so casting the tensors to fp32 is not enough --
    # the matmuls inside the prefix product would still run in bf16. Orthogonality drift there
    # is O(T * eps), which at bf16 reaches ~27% by T=1024 and stops being a rotation at all.
    # The decay exponentials want the fp32 dynamic range for the same reason. The Q x Q
    # einsums below are ordinary matmuls and are deliberately left under ambient autocast.
    with torch.autocast(device_type=device_type, enabled=False):
        x, B, C = x.float(), B.float(), C.float()
        dt, A, lam, theta = dt.float(), A.float(), lam.float(), theta.float()
        B, C = _rotate_and_broadcast(B, C, theta, block_size, heads_per_group, n_heads)

        batch, seq_len = x.shape[0], x.shape[1]
        B = B.reshape(batch, seq_len, n_heads, -1)
        C = C.reshape(batch, seq_len, n_heads, -1)

        log_alpha = dt * A
        gamma = lam * dt
        beta = (1.0 - lam) * dt * torch.exp(log_alpha)

    # gamma reads (B_s, x_s); the trapezoidal beta term reads the previous step's pair.
    y = _ssd_scalar_decay(log_alpha, gamma, B, C, x, chunk_size)
    y = y + _ssd_scalar_decay(log_alpha, beta, _shift_right(B), C, _shift_right(x), chunk_size)
    return y.to(out_dtype)


def _rotate_and_broadcast(
    B: torch.Tensor,
    C: torch.Tensor,
    theta: torch.Tensor,
    block_size: int,
    heads_per_group: int,
    n_heads: int,
):
    """
    Apply the cumulative rotation to ``B``/``C`` and broadcast groups to heads.

    Shares the ``b >= 3`` rotation with the official-kernel adapter rather than reimplementing
    it: the closed-form ``so(3)`` exponential and the shorter prefix-product scan are properties
    of the rotation, not of whichever scan consumes it. Before this the chunked path paid
    ``matrix_exp`` and a 64-long sequential scan while the official path did not, so every fp32
    eval, CPU run and ``mimo_rank > 1`` call silently gave up the speedup.
    """
    if block_size == 2:
        theta_cumulative = torch.cumsum(theta.squeeze(-1) if theta.dim() == 5 else theta, dim=1)
        B = _rotate_bc(B, theta_cumulative)
        C = _rotate_bc(C, theta_cumulative)
    else:
        if theta.dim() != 5:
            raise ValueError(
                f"theta must be 5-D (batch, seq_len, n_groups, n_blocks, angles_per_block) "
                f"for block_size={block_size}, got shape {tuple(theta.shape)}"
            )
        cumulative_rot = fast_cumulative_block_rotation(
            fast_block_rotations(theta, block_size)
        )
        B, C = _rotate_bc_fused(B, C, cumulative_rot)

    if heads_per_group != 1:
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)
    assert B.shape[2] == n_heads
    return B, C


def chunked_is_available(_: Optional[str] = None) -> bool:
    """The chunked path is pure PyTorch, so it is always available."""
    return True
