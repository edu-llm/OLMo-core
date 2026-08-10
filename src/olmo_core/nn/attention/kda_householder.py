"""
Triton kernels (forward **and** backward) for **KDA with** ``R`` **Householder (DeltaProduct)
factors**.

The recurrence
--------------
Let ``R = num_householder``. For every token ``t in [0, T)``, with state ``S`` of shape
``[K, V]`` per (batch, head):

1. **KDA per-channel decay** -- applied *once* per token, broadcast over the ``V`` axis::

       S <- S * exp(g[:, t])[..., None]          # g[:, t] has shape [B, H, K]

2. **DeltaProduct** -- ``R`` successive rank-1 delta / Householder updates, reading the
   *interleaved* ``k`` / ``v`` / ``beta`` tensors at positions ``t * R + r``::

       for r in 0 .. R - 1:
           kr, vr, br = k[:, t * R + r], v[:, t * R + r], beta[:, t * R + r]
           u = br * (vr - kr @ S)                # [B, H, V]
           S <- S + kr[..., None] * u[..., None, :]

3. **Readout**::

       o[:, t] = (S * q[:, t][..., None]).sum(-2)

Step 1 happens **once per token**, not once per Householder factor: the ``R`` factors of a token
share a single decay. This matches the convention of ``fla.ops.gated_delta_product``, which
materialises the gate as ``[g_t, 0, 0, ..., 0]`` along the interleaved axis before taking a chunk
local cumsum.

Shapes
------
=========  =======================  ============================================================
tensor     shape                    notes
=========  =======================  ============================================================
``q``      ``[B, T, H, K]``         one query per token
``k``      ``[B, T * R, H, K]``     interleaved along time
``v``      ``[B, T * R, H, V]``     interleaved along time
``beta``   ``[B, T * R, H]``        interleaved along time
``g``      ``[B, T, H, K]``         **log-space, per-channel**, one per token
``o``      ``[B, T, H, V]``
=========  =======================  ============================================================

Prior art, and the contribution
-------------------------------
* **DeltaProduct** -- Siems et al., *DeltaProduct: Improving State-Tracking in Linear RNNs via
  Householder Products*, `arXiv:2502.10297 <https://arxiv.org/abs/2502.10297>`_. Supplies the
  ``R``-Householder-per-token generalisation of DeltaNet and, in ``flash-linear-attention``, the
  interleaved-token layout reused here.
* **Kimi Linear / KDA** -- *Kimi Linear: An Expressive, Efficient Attention Architecture* (Kimi
  Delta Attention). Supplies the **per-channel** (diagonal, shape ``[..., K]``) forget gate, a
  strict generalisation of the per-head scalar gate of Gated DeltaNet.

**No kernel in flash-linear-attention combines a per-channel gate with** ``R > 1`` **Householder
factors -- that combination is the contribution of this file.** Concretely, in fla-core 0.4.1:

* ``fla.ops.gated_delta_product.chunk_gated_delta_product`` supports ``R > 1`` but guards
  ``assert g.shape == (B, T, H)`` -- a per-*head* scalar gate only.
* ``fla.ops.kda.chunk_kda`` supports a per-channel ``g`` of shape ``[B, T, H, K]`` but has no
  notion of ``num_householder``; it is hard-wired to ``R == 1``.
* ``fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`` accepts a per-channel ``gk`` but
  likewise has no ``num_householder``.

Implementation status
---------------------
This is a **simple fused-recurrent (sequential-over-time) Triton kernel**, *not* a chunked
WY-representation kernel. Each program owns one ``[BK, BV]`` tile of the state in registers and
walks the time axis, so the kernel body is a near-transcription of the already-validated reference
recurrence in ``probes/naive_kda_householder.py``. It is expected to be materially slower than
fla's chunked kernels; the goal of this milestone is mechanism validation, not throughput.

Because the state is carried explicitly across time steps, **no gate cumsum is required**: the
kernel consumes the raw per-token ``g`` and applies ``exp(g)`` once per step. (fla's chunked
kernels instead need ``chunk_local_cumsum`` to build within-chunk cumulative decays. That helper
*does* accept a 4-D ``[B, T, H, K]`` gate -- it dispatches to
``chunk_local_cumsum_vector`` for ``g.ndim == 4`` -- but it is not needed here.)

**Both paths train.** The Triton backward is the default and is ~400x faster than the reference at
``B2/T8192/R4``. The pure-PyTorch backend (``backend="torch"``, dispatching to
:func:`olmo_core.nn.attention.kda_householder_torch.kda_householder_torch`) is retained as the
*reference*: it is built from ordinary differentiable torch ops (a Python loop over ``T`` and
``R``), so autograd supplies the backward pass -- no gradient is hand-derived anywhere. It is
verified to agree with the naive oracle bit-exactly in ``float64`` for ``R in {1, 2, 3}`` and to
pass ``torch.autograd.gradcheck`` in ``float64``, and the Triton gradients are tested against it.

Keep the torch backend: it is the only path that runs on **CPU** (the triton import is GPU-only),
the only one that supports **float64** (hence the only one ``gradcheck`` can validate), and the
only one that is **twice differentiable** (the Triton backward is
:func:`~torch.autograd.function.once_differentiable`, since its gradients are kernel outputs
outside the autograd graph).

Structural attribution
----------------------
Pointer arithmetic, heuristics and grid layout are adapted from
``fla/ops/gated_delta_rule/fused_recurrent.py`` (``fused_recurrent_gated_delta_rule_fwd_kernel``;
Songlin Yang & Yu Zhang, 2023-2025, MIT licence). The ``R``-interleaved indexing of
``k`` / ``v`` / ``beta`` and the argument-validation guards are adapted from
``fla/ops/gated_delta_product/chunk.py``. No fla code is imported at runtime.
"""

from typing import Literal, Optional, Tuple

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

__all__ = ["chunk_kda_householder"]


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def kda_householder_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    R: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Sequential-over-time KDA + ``R``-Householder forward kernel.

    Grid is ``(cdiv(V, BV), N * H)``; each program owns a ``[BK, BV]`` tile of the recurrent
    state in registers and iterates over the time axis.

    ``BK`` must cover the entire ``K`` axis (i.e. ``BK == next_power_of_2(K)``) because the delta
    update ``kr @ S`` contracts over ``K``. The ``V`` axis *is* safely splittable: for a fixed
    ``v``-channel, ``u = beta * (v - kr @ S)`` touches only that column of ``S``.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n.to(tl.int64) * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # One entry per token: q, g, o.
    p_q = q + (bos * H + i_h) * K + o_k
    p_g = g + (bos * H + i_h) * K + o_k
    p_o = o + (bos * H + i_h) * V + o_v
    # R entries per token (interleaved along time): k, v, beta.
    p_k = k + (bos * R * H + i_h) * K + o_k
    p_v = v + (bos * R * H + i_h) * V + o_v
    p_beta = beta + bos * R * H + i_h

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for _ in range(0, T):
        # 1. KDA per-channel decay, applied once per token.
        #    Out-of-range k-lanes load g = 0 -> exp(0) = 1, which is harmless because the
        #    corresponding rows of b_h are identically zero (b_k is masked to 0 as well).
        b_g = tl.load(p_g, mask=mask_k, other=0.0).to(tl.float32)
        b_h = b_h * tl.exp(b_g)[:, None]

        # 2. R successive rank-1 delta / Householder updates. The inner loop is fully unrolled
        #    (R is a constexpr) and offsets from the token base pointer, so only the outer,
        #    per-token pointer bump is loop-carried.
        for i_r in tl.static_range(R):
            b_k = tl.load(p_k + i_r * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i_r * H * V, mask=mask_v, other=0.0).to(tl.float32)
            b_beta = tl.load(p_beta + i_r * H).to(tl.float32)
            # [BV] = beta * (v - k @ S), contracting the (fully covered) K axis.
            b_u = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_u[None, :]

        # 3. Readout.
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32) * scale
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_g += H * K
        p_o += H * V
        p_k += R * H * K
        p_v += R * H * V
        p_beta += R * H

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def kda_householder_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    num_householder: int,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch :func:`kda_householder_fwd_kernel`.

    All tensors must already be contiguous; ``g`` must already be ``float32``.

    :param q: Queries, ``[B, T, H, K]``.
    :param k: Keys, ``[B, T * R, H, K]``.
    :param v: Values, ``[B, T * R, H, V]``.
    :param g: Log-space per-channel decay, ``[B, T, H, K]``, ``float32``.
    :param beta: Delta-rule step sizes, ``[B, T * R, H]``.
    :param scale: Query scaling factor.
    :param num_householder: Number of Householder factors ``R`` per token.
    :param initial_state: Optional initial state, ``[N, H, K, V]``.
    :param output_final_state: Whether to return the final state.
    :param cu_seqlens: Optional cumulative sequence lengths, ``[N + 1]``, in *token* units
        (i.e. **not** multiplied by ``R``).
    :returns: ``(o, final_state)``; ``o`` has dtype ``v.dtype``, ``final_state`` is ``float32``
        or ``None``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # BK must span all of K: the delta update contracts over the K axis.
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    # `b_h` is [BK, BV] float32 and lives in registers. fla's analogous fused-recurrent kernel
    # hard-codes num_warps=1; at BK >= 128 that is ~32+ registers per thread for the state alone,
    # so widen the warp count to keep it off the spill path. Correctness-neutral: `tl.sum` reduces
    # across the whole block regardless of warp count.
    num_warps = 1 if BK <= 64 else (2 if BK <= 128 else 4)

    # Note: `o` has T (not T * R) rows, so it cannot be `torch.empty_like(v)`.
    o = torch.empty(B, T, H, V, dtype=v.dtype, device=v.device)
    final_state = q.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NV, N * H)
    kda_householder_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        R=num_householder,
        num_warps=num_warps,
        num_stages=3,
    )
    return o, final_state


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def kda_householder_bwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    do,
    h0,
    hs,
    dq_p,
    dk_p,
    dv,
    dg_p,
    db_p,
    dh0,
    cu_seqlens,
    scale,
    T,
    s_dq,
    s_dk,
    s_db,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    R: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Sequential-over-time KDA + ``R``-Householder backward kernel.

    A transcription of ``probes/bwd_emulator.py``, whose flat-offset arithmetic was verified in
    ``float64`` against a ``gradcheck``-validated reference across seven shape/varlen cases.

    Grid is ``(cdiv(V, BV), N * H)``, mirroring :func:`kda_householder_fwd_kernel`. Each program
    makes two passes over the time axis: pass 1 recomputes the forward, persisting the pre-decay
    state of every token to ``hs``; pass 2 walks time in reverse carrying a ``[BK, BV]`` gradient
    tile.

    The forward is *recomputed* rather than inverted: undoing a rank-1 delta update requires
    dividing by ``|k|^2 - 1/beta``, which is singular at ``beta == 1`` with L2-normalised keys --
    an interior point of ``beta = 2 * sigmoid(.) in (0, 2)``.

    ``dq``, ``dk``, ``dg`` and ``dbeta`` all contract over ``V``, so each program produces only a
    *partial* result; each writes into its own ``i_v`` slot of a ``[NV, ...]`` buffer, which the
    host sums. Deterministic, unlike ``tl.atomic_add``. ``dv`` and ``dh0`` need no reduction --
    distinct ``i_v`` programs own disjoint ``V`` columns.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_n.to(tl.int64) * T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # ---- pass 1: forward recompute, persisting each token's pre-decay state ---------------------
    p_g = g + (bos * H + i_h) * K + o_k
    p_k = k + (bos * R * H + i_h) * K + o_k
    p_v = v + (bos * R * H + i_h) * V + o_v
    p_beta = beta + bos * R * H + i_h
    p_hs = hs + (bos * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for _ in range(0, T):
        tl.store(p_hs, b_h, mask=mask_h)
        b_g = tl.load(p_g, mask=mask_k, other=0.0).to(tl.float32)
        b_h = b_h * tl.exp(b_g)[:, None]
        for i_r in tl.static_range(R):
            b_k = tl.load(p_k + i_r * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i_r * H * V, mask=mask_v, other=0.0).to(tl.float32)
            b_beta = tl.load(p_beta + i_r * H).to(tl.float32)
            b_u = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_u[None, :]
        p_g += H * K
        p_k += R * H * K
        p_v += R * H * V
        p_beta += R * H
        p_hs += H * K * V

    # Pass 1 stores 'hs'; pass 2 loads it back. Triton is free to choose different register
    # layouts for a store and a load of the same block shape, in which case the lane that wrote a
    # given address is not the lane that reads it -- a cross-warp race with no program-order
    # guarantee. This is invisible at BK <= 64 (num_warps == 1, so the warp executes in lockstep)
    # and only becomes reachable at head_dim >= 128, which no test currently covers. The barrier
    # executes once per program, not per time step, so it does not show up in the benchmark.
    tl.debug_barrier()

    # ---- pass 2: reverse time -------------------------------------------------------------------
    # Rewind to the last token; every pointer below is decremented at the end of each iteration.
    last = T - 1
    p_q = q + ((bos + last) * H + i_h) * K + o_k
    p_g = g + ((bos + last) * H + i_h) * K + o_k
    p_do = do + ((bos + last) * H + i_h) * V + o_v
    p_k = k + (((bos + last) * R) * H + i_h) * K + o_k
    p_v = v + (((bos + last) * R) * H + i_h) * V + o_v
    p_beta = beta + ((bos + last) * R) * H + i_h
    p_hs = hs + ((bos + last) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
    p_dq = dq_p + i_v.to(tl.int64) * s_dq + ((bos + last) * H + i_h) * K + o_k
    p_dg = dg_p + i_v.to(tl.int64) * s_dq + ((bos + last) * H + i_h) * K + o_k
    p_dk = dk_p + i_v.to(tl.int64) * s_dk + (((bos + last) * R) * H + i_h) * K + o_k
    p_dv = dv + (((bos + last) * R) * H + i_h) * V + o_v
    p_db = db_p + i_v.to(tl.int64) * s_db + ((bos + last) * R) * H + i_h

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(0, T):
        b_g = tl.load(p_g, mask=mask_k, other=0.0).to(tl.float32)
        b_a = tl.exp(b_g)

        # Re-materialise S^(0) from the persisted pre-decay state. S^(0) = S_{t-1} * exp(g_t).
        b_s0 = tl.load(p_hs, mask=mask_h, other=0.0).to(tl.float32) * b_a[:, None]

        # Walk forward through the R updates to reach S^(R), keeping the running state. The
        # reverse sweep below re-walks from b_s0 for each r (O(R^2) arithmetic, two live tiles)
        # rather than holding R + 1 tiles, which would spill at BK = 128.
        b_sr = b_s0
        for i_r in tl.static_range(R):
            b_k = tl.load(p_k + i_r * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i_r * H * V, mask=mask_v, other=0.0).to(tl.float32)
            b_beta = tl.load(p_beta + i_r * H).to(tl.float32)
            b_u = b_beta * (b_v - tl.sum(b_sr * b_k[:, None], 0))
            b_sr += b_k[:, None] * b_u[None, :]

        # Readout: o_t = (q_t * scale) @ S^(R), after all R updates.
        b_do = tl.load(p_do, mask=mask_v, other=0.0).to(tl.float32)
        b_dq = scale * tl.sum(b_sr * b_do[None, :], 1)
        tl.store(p_dq, b_dq, mask=mask_k)
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_dh += (b_q * scale)[:, None] * b_do[None, :]

        # Reverse the R-chain. inner[i_r] is recomputed by re-walking i_r steps from b_s0.
        for i_rr in tl.static_range(R):
            i_r = R - 1 - i_rr
            b_inner = b_s0
            for j in tl.static_range(R):
                if j < i_r:
                    b_kj = tl.load(p_k + j * H * K, mask=mask_k, other=0.0).to(tl.float32)
                    b_vj = tl.load(p_v + j * H * V, mask=mask_v, other=0.0).to(tl.float32)
                    b_bj = tl.load(p_beta + j * H).to(tl.float32)
                    b_uj = b_bj * (b_vj - tl.sum(b_inner * b_kj[:, None], 0))
                    b_inner += b_kj[:, None] * b_uj[None, :]

            b_k = tl.load(p_k + i_r * H * K, mask=mask_k, other=0.0).to(tl.float32)
            b_v = tl.load(p_v + i_r * H * V, mask=mask_v, other=0.0).to(tl.float32)
            b_beta = tl.load(p_beta + i_r * H).to(tl.float32)
            b_resid = b_v - tl.sum(b_inner * b_k[:, None], 0)
            b_u = b_beta * b_resid
            b_du = tl.sum(b_k[:, None] * b_dh, 0)
            tl.store(p_db + i_r * H, tl.sum(b_du * b_resid))
            tl.store(p_dv + i_r * H * V, (b_beta * b_du).to(p_dv.dtype.element_ty), mask=mask_v)
            b_dw = -b_beta * b_du
            # k_i appears twice -- in outer(k_i, u) and inside w = k_i @ S^(r). Both terms
            # contract V, so dk is a partial as well; easy to overlook next to dv, which is not.
            b_dk = tl.sum(b_dh * b_u[None, :], 1) + tl.sum(b_inner * b_dw[None, :], 1)
            tl.store(p_dk + i_r * H * K, b_dk, mask=mask_k)
            b_dh += b_k[:, None] * b_dw[None, :]

        # dg uses b_dh *after* the R-chain has been walked back but *before* the decay hand-off.
        tl.store(p_dg, tl.sum(b_dh * b_s0, 1), mask=mask_k)
        b_dh = b_dh * b_a[:, None]

        p_q -= H * K
        p_g -= H * K
        p_do -= H * V
        p_k -= R * H * K
        p_v -= R * H * V
        p_beta -= R * H
        p_hs -= H * K * V
        p_dq -= H * K
        p_dg -= H * K
        p_dk -= R * H * K
        p_dv -= R * H * V
        p_db -= R * H

    # b_dh at loop exit is dh0, already past the final decay hand-off. Only stored when there is
    # an initial state to receive it -- the host leaves 'dh0' unallocated (None) otherwise.
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh.to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh, mask=mask_h)


def kda_householder_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    num_householder: int,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    """Host wrapper for :func:`kda_householder_bwd_kernel`.

    Allocates the ``[NV, ...]`` partial buffers for the four ``V``-contracting gradients, launches
    the grid, and reduces over the ``NV`` axis. The reduction is a plain ``sum``, so results are
    bit-identical run to run (no atomics).

    :param scale: Query scaling factor, as applied in the forward.
    :param initial_state: Optional initial state ``[N, H, K, V]``.
    :param cu_seqlens: Optional cumulative sequence lengths ``[N + 1]`` in token units.

    :returns: ``(dq, dk, dv, dg, dbeta, dh0)``; ``dh0`` is ``None`` when ``initial_state`` is
        ``None``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    R = num_householder
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    num_warps = 1 if BK <= 64 else (2 if BK <= 128 else 4)

    fp32 = torch.float32
    dev = q.device
    # Workspace: the state entering each token, before its decay. Indexed by the flat token
    # index (bos + t), so varlen sequences cannot alias.
    #
    # WARNING: this is O(B * T * H * K * V) float32 -- the *entire* per-token state history, K*V
    # floats per token per head. It is proportional to sequence length and is the dominant memory
    # cost of the backward. At B4/T4096/H16/K64/V64 it is 6.0 GiB; at B8/T4096/H16/K128/V128 it is
    # 48 GiB and will OOM a single GPU. The forward gives no warning of this -- the layer forwards
    # fine at production length and then OOMs on the first '.backward()'. Chunking the two passes
    # over time so that only a window of 'hs' is live would bound it; that is not done here
    # because the probe models this op was added for are far below the cliff.
    hs = torch.empty(B * T * H * K * V, dtype=fp32, device=dev)

    s_dq = B * T * H * K
    s_dk = B * T * R * H * K
    s_db = B * T * R * H
    dq_p = torch.zeros(NV * s_dq, dtype=fp32, device=dev)
    dg_p = torch.zeros(NV * s_dq, dtype=fp32, device=dev)
    dk_p = torch.zeros(NV * s_dk, dtype=fp32, device=dev)
    db_p = torch.zeros(NV * s_db, dtype=fp32, device=dev)
    dv = torch.zeros_like(v, dtype=v.dtype)
    # Only materialised when there is an initial state to receive the gradient. The kernel's
    # 'dh0' stores are guarded by USE_INITIAL_STATE, so the buffer is untouched otherwise.
    dh0 = torch.zeros(N, H, K, V, dtype=fp32, device=dev) if initial_state is not None else None

    grid = (NV, N * H)
    kda_householder_bwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        do,
        initial_state,
        hs,
        dq_p,
        dk_p,
        dv,
        dg_p,
        db_p,
        dh0,
        cu_seqlens,
        scale,
        T,
        s_dq,
        s_dk,
        s_db,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        R=R,
        num_warps=num_warps,
    )

    # Collapse the NV axis of the four partials.
    dq = dq_p.view(NV, -1).sum(0).view(B, T, H, K).to(q.dtype)
    dg = dg_p.view(NV, -1).sum(0).view(B, T, H, K).to(g.dtype)
    dk = dk_p.view(NV, -1).sum(0).view(B, T * R, H, K).to(k.dtype)
    dbeta = db_p.view(NV, -1).sum(0).view(B, T * R, H).to(beta.dtype)
    return (
        dq,
        dk,
        dv,
        dg,
        dbeta,
        (dh0.to(initial_state.dtype) if initial_state is not None else None),
    )


class ChunkKDAHouseholderFunction(torch.autograd.Function):
    """Autograd wrapper for the fused-recurrent forward and backward kernels."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        num_householder: int,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the forward recurrence. See :func:`chunk_kda_householder` for semantics."""
        o, final_state = kda_householder_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            num_householder=num_householder,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, g, beta, initial_state)
        ctx.scale = scale
        ctx.num_householder = num_householder
        ctx.cu_seqlens = cu_seqlens
        # The kernel propagates only 'do'; it has no path for a cotangent on the final state.
        # Mark it structurally rather than inspecting 'dht' in the backward: a value test
        # ('dht.abs().any()') passes silently whenever the cotangent happens to be all-zero even
        # though the final state is genuinely in the graph -- e.g. a masked TBPTT carry-over --
        # and returns gradients that quietly omit that path. It also forces a device-to-host sync
        # on every backward step. Marking is checked by autograd at graph-construction time.
        if final_state is not None:
            ctx.mark_non_differentiable(final_state)
        return o.to(q.dtype), final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):  # type: ignore[override]
        """Backward pass via :func:`kda_householder_bwd`.

        Marked :func:`~torch.autograd.function.once_differentiable`: the returned gradients are
        freshly-allocated kernel outputs that live *outside* the autograd graph, so without this
        decorator a double-backward consumer (gradient penalties, Hessian-vector products,
        ``create_graph=True``) would silently receive zero for every second-order term instead of
        an error. Use ``backend="torch"`` if you genuinely need to differentiate twice.
        """
        del dht  # 'final_state' is marked non-differentiable in the forward.
        q, k, v, g, beta, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dbeta, dh0 = kda_householder_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            do=do.contiguous(),
            num_householder=ctx.num_householder,
            scale=ctx.scale,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
        )
        # Match forward's parameter order: q, k, v, g, beta, scale, num_householder,
        # initial_state, output_final_state, cu_seqlens.
        return dq, dk, dv, dg, dbeta, None, None, dh0, None, None


@torch.compiler.disable
def chunk_kda_householder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    backend: Literal["triton", "torch"] = "triton",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA per-channel gated delta rule with ``R = num_householder`` Householder factors per token.

    Numerically equivalent to ``probes/naive_kda_householder.naive_recurrent_kda_householder``,
    which is the validated oracle for this op (bit-exact against
    ``fla.ops.kda.naive.naive_recurrent_kda`` at ``R == 1``, and matching
    ``fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product`` to float64 ulp when
    ``g`` is constant along ``K``).

    Despite the ``chunk_`` prefix -- kept so the signature matches its fla counterparts and a
    future chunked implementation can be swapped in -- the current backend is a sequential
    fused-recurrent Triton kernel. See the module docstring.

    .. warning::
        The Triton backward allocates an ``O(B * T * H * K * V)`` float32 workspace (the full
        per-token state history). This is proportional to sequence length and can OOM at
        production shapes even when the forward fits -- see :func:`kda_householder_bwd`.

    :param q: Queries of shape ``[B, T, H, K]``. Must **not** be ``float32``; use ``bfloat16``.
    :param k: Keys of shape ``[B, T * R, H, K]``, interleaved along the time axis.
    :param v: Values of shape ``[B, T * R, H, V]``, interleaved along the time axis.
    :param g: Log-space **per-channel** forget gate of shape ``[B, T, H, K]``, one entry per
        token (*not* interleaved). Cast internally to ``float32``. Expected to be ``<= 0``.
    :param beta: Delta-rule step sizes of shape ``[B, T * R, H]``, interleaved along time.
    :param num_householder: Number of Householder / delta factors ``R`` applied per token.
    :param scale: Query scaling factor. Defaults to ``K ** -0.5``.
    :param cu_seqlens: Cumulative sequence lengths of shape ``[N + 1]`` for variable-length
        batches, in **token** units (the kernel scales them by ``R`` internally for the
        interleaved tensors). Requires ``q.shape[0] == 1``.
    :param initial_state: Optional initial state of shape ``[N, H, K, V]``.
    :param output_final_state: Whether to also return the final state.
    :param backend: Which implementation to dispatch to. ``"triton"`` (default) is the fast,
        GPU-only fused-recurrent kernel; both its forward and backward are validated, and it is
        what you want for training. ``"torch"`` is the slow reference recurrence
        (:func:`~olmo_core.nn.attention.kda_householder_torch.kda_householder_torch`); it is the
        only path that runs on CPU, supports ``float64``, and is twice differentiable.
    :returns: ``(o, final_state)`` where ``o`` has shape ``[B, T, H, V]`` and dtype ``q.dtype``,
        and ``final_state`` has shape ``[N, H, K, V]`` in ``float32`` if ``output_final_state``
        else ``None``.
    :raises AssertionError: on a dtype or shape violation.
    :raises ValueError: if ``cu_seqlens`` is given with a batch size other than 1, if the
        number of initial states disagrees with the number of sequences, or if ``backend`` is not
        one of ``"triton"`` / ``"torch"``.
    """
    if backend not in ("triton", "torch"):
        raise ValueError(f"backend must be 'triton' or 'torch', got {backend!r}")
    if backend == "triton":
        # Mirrors `fla/ops/gated_delta_product/chunk.py`: the kernel accumulates in float32 but the
        # inputs are expected to be bf16, and a float32 input silently doubles memory traffic.
        # Scoped to the Triton backend: the torch backend has no such memory concern and is
        # deliberately usable in float32/float64 (the latter is what makes `gradcheck` meaningful).
        assert (
            q.dtype != torch.float32
        ), "ChunkKDAHouseholderFunction does not support float32. Please use bfloat16."

    R = num_householder
    assert R >= 1, f"num_householder must be >= 1, got {R}"
    assert q.ndim == 4, f"expected q to be 4-D [B, T, H, K], got {tuple(q.shape)}"
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T * R, H, K), f"expected k {(B, T * R, H, K)}, got {tuple(k.shape)}"
    assert v.shape == (B, T * R, H, V), f"expected v {(B, T * R, H, V)}, got {tuple(v.shape)}"
    assert beta.shape == (B, T * R, H), f"expected beta {(B, T * R, H)}, got {tuple(beta.shape)}"
    assert g.shape == (B, T, H, K), f"expected g {(B, T, H, K)}, got {tuple(g.shape)}"
    assert K <= 256, f"only key headdim <= 256 is supported, got {K}"

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using "
                f"`cu_seqlens`. Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input "
                f"sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
        assert cu_seqlens.ndim == 1, f"expected cu_seqlens to be 1-D, got {tuple(cu_seqlens.shape)}"
        # Workspaces are sized off T but every store is indexed off `bos + t` from cu_seqlens.
        # If the two disagree the kernel writes out of bounds (silently, with no mask to stop it)
        # or leaves trailing tokens holding uninitialised `torch.empty` memory.
        assert int(cu_seqlens[0]) == 0 and int(cu_seqlens[-1]) == q.shape[1], (
            f"cu_seqlens must span exactly T={q.shape[1]} tokens starting at 0, got "
            f"[{int(cu_seqlens[0])}, {int(cu_seqlens[-1])}]"
        )
        assert bool((cu_seqlens[1:] >= cu_seqlens[:-1]).all()), "cu_seqlens must be non-decreasing"

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if initial_state is not None:
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"expected initial_state {(N, H, K, V)}, got {tuple(initial_state.shape)}"

    if scale is None:
        scale = K**-0.5

    if backend == "torch":
        # Differentiable path. Deliberately does *not* force `g` to float32 or the inputs to be
        # contiguous: there is no pointer arithmetic to satisfy, and casting `g` would both break
        # a float64 `gradcheck` and insert a needless node in the autograd graph. The
        # implementation promotes internally against float32, so low-precision inputs still
        # accumulate in float32 exactly as the kernel does.
        return kda_householder_torch(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_householder=R,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    # The kernel does raw pointer arithmetic, so contiguity is mandatory (cf. fla's
    # `input_guard`). `g` additionally has to be float32: it is the one tensor whose precision
    # compounds multiplicatively over the whole sequence.
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    g = g.contiguous().to(torch.float32)
    if initial_state is not None:
        initial_state = initial_state.contiguous().to(torch.float32)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()

    return ChunkKDAHouseholderFunction.apply(  # type: ignore[return-value]
        q,
        k,
        v,
        g,
        beta,
        scale,
        R,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
