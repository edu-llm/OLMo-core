"""PyTorch *emulator* of the flat-pointer arithmetic of a KDA + ``R``-Householder **backward**
Triton kernel.

There is deliberately **no Triton in this file**. It is a blueprint: a plain-``torch`` replay of the
exact flat offsets, boundary masks, tile shapes and loop order the real kernel will use, so the
indexing can be executed and checked on CPU before a single line of Triton is written.

What is emulated
----------------
* Grid ``(NV, N * H)``, iterated as an explicit double Python loop over program ids, with
  ``i_n, i_h = i_nh // H, i_nh % H`` exactly as in
  :func:`olmo_core.nn.attention.kda_householder.kda_householder_fwd_kernel`.
* ``BK = next_power_of_2(K)`` spans the whole ``K`` axis (every reduction over ``K`` -- ``k @ S``,
  ``du = k @ dS`` -- is therefore exact inside one program). ``BV`` splits the ``V`` axis.
* All tensors are ``reshape(-1)``-flattened and every access goes through a 1-D/2-D **integer
  offset tensor** plus a boundary mask, i.e. ``tl.load(ptr + off, mask=..., other=0.0)`` and
  ``tl.store(ptr + off, val, mask=...)``. No multi-dimensional indexing shortcuts.
* Two passes per program, both over the same offsets:
  1. **forward recompute**, storing the state *entering* each token (before its decay) into a
     ``[total_tokens, H, K, V]`` fp32 workspace ``hs``;
  2. **reverse-time** pass carrying a ``[BK, BV]`` gradient tile ``b_dh``.

  Pass 1 is unavoidable: the forward kernel only persists the *final* state, and recovering the
  per-token state by running the recurrence backwards would require inverting a rank-1 delta
  update (``u_r = (k_r . S^(r+1) - v_r) / (|k_r|^2 - 1/beta_r)``), which is singular for
  ``beta_r = 1 / |k_r|^2``. The workspace costs ``total_tokens * H * K * V`` floats; a production
  kernel would checkpoint every ``C`` tokens and recompute within the chunk instead.

The V-reduction (the crux)
--------------------------
``dq``, ``dk``, ``dg`` and ``dbeta`` all contract over ``V``, so a program that owns only the
``i_v``-th ``BV``-slice of ``V`` can compute only a **partial** result for them. (Note that ``dk``
contracts over ``V`` too -- both of its terms end in ``.sum(-1)`` over the ``V`` axis. It is easy to
mistake it for a ``dv``-like per-tile output.) This emulator uses approach **(c)**: each program
writes its partial into its own ``i_v`` slot of a flat ``[NV, ...]`` buffer, and the ``NV`` axis is
summed *after* the grid loop. Deliberately **not** ``atomic_add``: fp32 atomics accumulate in
nondeterministic order, which would make a GPU test flaky at tight tolerances.

``dv`` and ``dh0`` need no reduction: distinct ``i_v`` programs own distinct ``V`` columns, so they
are stored straight into the output with ``mask_v`` / ``mask_h``.

Numerics
--------
Everything runs in ``float64``, so that any disagreement with
``probes/manual_backward_check.manual_backward`` (verified to float64 round-off) is an **indexing**
bug and not a precision artefact. The real kernel carries the same tiles in fp32 registers.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

__all__ = ["emulate_bwd_kernel"]


def _next_power_of_2(n: int) -> int:
    """Smallest power of two ``>= n`` (stand-in for ``triton.next_power_of_2``).

    :param n: A positive integer.

    :returns: The smallest power of two greater than or equal to ``n``.
    """
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _load(flat: torch.Tensor, off: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Emulate ``tl.load(ptr + off, mask=mask, other=0.0)`` on a flattened tensor.

    Masked-off lanes are never dereferenced (their offset is clamped to 0 first, because unlike a
    Triton predicated load a torch gather would raise on an out-of-range index) and read as zero.

    :param flat: 1-D view of the tensor being read.
    :param off: Integer offsets, same shape as the tile.
    :param mask: Boolean mask, same shape as ``off``.

    :returns: The gathered tile, zero where ``mask`` is ``False``.
    """
    safe = torch.where(mask, off, torch.zeros_like(off))
    return torch.where(mask, flat[safe], flat.new_zeros(()))


def _store(flat: torch.Tensor, off: torch.Tensor, val: torch.Tensor, mask: torch.Tensor) -> None:
    """Emulate ``tl.store(ptr + off, val, mask=mask)`` on a flattened tensor.

    :param flat: 1-D view of the tensor being written (modified in place).
    :param off: Integer offsets, same shape as the tile.
    :param val: Values to store, same shape as ``off``.
    :param mask: Boolean mask, same shape as ``off``.
    """
    sel = mask.reshape(-1)
    flat[off.reshape(-1)[sel]] = val.reshape(-1)[sel]


def emulate_bwd_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    num_householder: int,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    BV: int = 8,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay the backward kernel's pointer arithmetic in plain ``torch``.

    :param q: Queries, ``[B, T, H, K]`` (un-scaled; ``scale`` is applied internally).
    :param k: Keys, ``[B, T * R, H, K]``, interleaved along time.
    :param v: Values, ``[B, T * R, H, V]``, interleaved along time.
    :param g: Raw log-space per-channel decay, ``[B, T, H, K]`` (no cumsum).
    :param beta: Delta-rule step sizes, ``[B, T * R, H]``, interleaved along time.
    :param do: Upstream gradient of the output, ``[B, T, H, V]``.
    :param num_householder: Number of Householder / delta factors ``R`` per token.
    :param scale: Query scaling factor. Defaults to ``K ** -0.5``.
    :param initial_state: Optional initial state, ``[N, H, K, V]``.
    :param BV: Tile width along ``V``; the real kernel uses ``min(8, next_power_of_2(V))``. Exposed
        so tests can force ``NV > 1`` and partial trailing tiles.
    :param cu_seqlens: Optional cumulative sequence lengths, ``[N + 1]``, in **token** units
        (scaled by ``R`` internally for the interleaved tensors). Requires ``B == 1``.

    :returns: ``(dq, dk, dv, dg, dbeta, dh0)`` in ``float64``, shaped like the corresponding
        inputs; ``dh0`` is ``[N, H, K, V]`` (the zero-state gradient if ``initial_state`` is
        ``None``).
    """
    R = num_householder
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    dt = torch.float64

    # --- launch parameters, mirroring `kda_householder_fwd` -------------------------------------
    BK = _next_power_of_2(K)  # must span all of K: `k @ S` contracts over K
    NV = (V + BV - 1) // BV  # == triton.cdiv(V, BV)
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    NT = B * T  # tokens in the flattened (B * T) axis; == cu_seqlens[-1] when varlen (B == 1)

    # --- everything the kernel sees is a flat 1-D array of elements ------------------------------
    qf, kf, vf, gf, bf, dof = (x.to(dt).contiguous().reshape(-1) for x in (q, k, v, g, beta, do))
    h0f = None if initial_state is None else initial_state.to(dt).contiguous().reshape(-1)

    # Per-`i_v` partial buffers for the four V-contracting gradients. One flat allocation each,
    # strided by `s_*` along the leading NV axis; summed over NV after the grid loop.
    s_dq = NT * H * K  # also the dg stride
    s_dk = NT * R * H * K
    s_db = NT * R * H
    dq_p = torch.zeros(NV * s_dq, dtype=dt)
    dg_p = torch.zeros(NV * s_dq, dtype=dt)
    dk_p = torch.zeros(NV * s_dk, dtype=dt)
    db_p = torch.zeros(NV * s_db, dtype=dt)
    # No NV axis: distinct i_v programs own distinct V columns.
    dvf = torch.zeros(NT * R * H * V, dtype=dt)
    dh0f = torch.zeros(N * H * K * V, dtype=dt)
    # fp32 workspace holding the state *entering* each token, laid out [NT, H, K, V].
    hs = torch.zeros(NT * H * K * V, dtype=dt)

    for i_v in range(NV):  # grid axis 0
        for i_nh in range(N * H):  # grid axis 1
            i_n, i_h = i_nh // H, i_nh % H

            if cu_seqlens is not None:
                bos = int(cu_seqlens[i_n])
                eos = int(cu_seqlens[i_n + 1])
                T_n = eos - bos
            else:
                bos = i_n * T
                T_n = T

            o_k = torch.arange(BK)
            o_v = i_v * BV + torch.arange(BV)
            mask_k = o_k < K
            mask_v = o_v < V
            mask_h = mask_k[:, None] & mask_v[None, :]

            # ---- pass 1: forward recompute, saving the pre-decay state of every token ----------
            b_h = torch.zeros(BK, BV, dtype=dt)
            if h0f is not None:
                b_h = b_h + _load(h0f, i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask_h)
            for t in range(T_n):
                o_hs = ((bos + t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                _store(hs, o_hs, b_h, mask_h)
                b_g = _load(gf, ((bos + t) * H + i_h) * K + o_k, mask_k)
                b_h = b_h * b_g.exp()[:, None]  # masked lanes load g = 0 -> exp = 1, rows are 0
                for i_r in range(R):
                    o_ki = (((bos + t) * R + i_r) * H + i_h) * K + o_k
                    o_vi = (((bos + t) * R + i_r) * H + i_h) * V + o_v
                    o_bi = ((bos + t) * R + i_r) * H + i_h
                    b_k = _load(kf, o_ki, mask_k)
                    b_v = _load(vf, o_vi, mask_v)
                    b_beta = bf[o_bi]
                    b_u = b_beta * (b_v - (b_h * b_k[:, None]).sum(0))
                    b_h = b_h + b_k[:, None] * b_u[None, :]

            # ---- pass 2: reverse time, carrying the [BK, BV] gradient tile ---------------------
            b_dh = torch.zeros(BK, BV, dtype=dt)
            for t in reversed(range(T_n)):
                o_hs = ((bos + t) * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_g = _load(gf, ((bos + t) * H + i_h) * K + o_k, mask_k)
                b_a = b_g.exp()

                # Re-materialise the R + 1 intra-token states S^(0) .. S^(R) from the saved
                # pre-decay state. R is a constexpr in the kernel, so this list is R + 1 unrolled
                # register tiles; a register-starved variant can re-walk from inner[0] per r at
                # O(R^2) arithmetic and only two live tiles.
                inner: List[torch.Tensor] = [_load(hs, o_hs, mask_h) * b_a[:, None]]
                for i_r in range(R):
                    o_ki = (((bos + t) * R + i_r) * H + i_h) * K + o_k
                    o_vi = (((bos + t) * R + i_r) * H + i_h) * V + o_v
                    o_bi = ((bos + t) * R + i_r) * H + i_h
                    b_k = _load(kf, o_ki, mask_k)
                    b_v = _load(vf, o_vi, mask_v)
                    b_u = bf[o_bi] * (b_v - (inner[i_r] * b_k[:, None]).sum(0))
                    inner.append(inner[i_r] + b_k[:, None] * b_u[None, :])

                # Readout o_t = (q_t * scale) @ S^(R) happens after all R updates.
                b_do = _load(dof, ((bos + t) * H + i_h) * V + o_v, mask_v)
                # dq contracts V -> PARTIAL, into slot i_v of dq_p.
                b_dq = scale * (inner[R] * b_do[None, :]).sum(1)
                _store(dq_p, i_v * s_dq + ((bos + t) * H + i_h) * K + o_k, b_dq, mask_k)
                b_q = _load(qf, ((bos + t) * H + i_h) * K + o_k, mask_k)
                b_dh = b_dh + (b_q * scale)[:, None] * b_do[None, :]

                # The R updates form a chain in S; walk them in reverse r order.
                for i_r in reversed(range(R)):
                    o_ki = (((bos + t) * R + i_r) * H + i_h) * K + o_k
                    o_vi = (((bos + t) * R + i_r) * H + i_h) * V + o_v
                    o_bi = ((bos + t) * R + i_r) * H + i_h
                    b_k = _load(kf, o_ki, mask_k)
                    b_v = _load(vf, o_vi, mask_v)
                    b_beta = bf[o_bi]
                    b_resid = b_v - (inner[i_r] * b_k[:, None]).sum(0)  # [BV]
                    b_u = b_beta * b_resid
                    b_du = (b_k[:, None] * b_dh).sum(0)  # [BV]; contracts K, fully covered by BK
                    # dbeta contracts V -> PARTIAL, into slot i_v of db_p.
                    db_p[i_v * s_db + o_bi] = (b_du * b_resid).sum()
                    # dv needs no reduction: this program owns these V columns outright.
                    _store(dvf, o_vi, b_beta * b_du, mask_v)
                    b_dw = -b_beta * b_du  # grad w.r.t. w = k_i @ S^(r)
                    # k_i appears twice: in outer(k_i, u) and inside w. BOTH terms contract V,
                    # so dk is a PARTIAL as well -- easy to overlook.
                    b_dk = (b_dh * b_u[None, :]).sum(1) + (inner[i_r] * b_dw[None, :]).sum(1)
                    _store(dk_p, i_v * s_dk + o_ki, b_dk, mask_k)
                    b_dh = b_dh + b_k[:, None] * b_dw[None, :]  # now grad w.r.t. S^(r)

                # S^(0) = S_{t-1} * exp(g_t); inner[0] is exactly that product.
                # dg contracts V -> PARTIAL, into slot i_v of dg_p.
                _store(
                    dg_p,
                    i_v * s_dq + ((bos + t) * H + i_h) * K + o_k,
                    (b_dh * inner[0]).sum(1),
                    mask_k,
                )
                b_dh = b_dh * b_a[:, None]

            # dh0 needs no reduction either: one program per (i_nh, V-slice).
            _store(dh0f, i_nh * K * V + o_k[:, None] * V + o_v[None, :], b_dh, mask_h)

    # --- THE V-REDUCTION: collapse the NV axis of the four partial buffers ----------------------
    dq = dq_p.reshape(NV, s_dq).sum(0).reshape(B, T, H, K)
    dg = dg_p.reshape(NV, s_dq).sum(0).reshape(B, T, H, K)
    dk = dk_p.reshape(NV, s_dk).sum(0).reshape(B, T * R, H, K)
    dbeta = db_p.reshape(NV, s_db).sum(0).reshape(B, T * R, H)
    dv = dvf.reshape(B, T * R, H, V)
    dh0 = dh0f.reshape(N, H, K, V)
    return dq, dk, dv, dg, dbeta, dh0


# ---------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------

_NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")

_CASES = (
    # label, B, T, H, K, V, R, BV, use_h0, cu_seqlens
    ("dense R=1", 2, 5, 2, 4, 4, 1, 4, False, None),
    ("dense R=2", 2, 5, 2, 4, 8, 2, 4, False, None),
    ("dense R=3", 2, 6, 1, 8, 8, 3, 8, False, None),
    ("ragged K", 2, 5, 2, 5, 4, 2, 4, False, None),
    ("ragged V", 2, 5, 2, 4, 5, 2, 4, False, None),
    ("initial_state", 2, 5, 2, 4, 4, 2, 4, True, None),
    ("varlen", 1, 7, 2, 4, 4, 2, 4, False, (0, 3, 7)),
)


def _manual_backward():  # type: ignore[no-untyped-def]
    """Import ``manual_backward`` from the sibling probe, extending ``sys.path`` if needed.

    :returns: The :func:`probes.manual_backward_check.manual_backward` function.
    """
    for p in ("/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src", str(Path(__file__).parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from manual_backward_check import manual_backward

    return manual_backward


def _reference(q, k, v, g, beta, do, R, h0, cu_seqlens):  # type: ignore[no-untyped-def]
    """Ground truth from :func:`probes.manual_backward_check.manual_backward`.

    ``manual_backward`` has no ``cu_seqlens`` argument, so for the varlen case it is called once
    per sequence on the corresponding slices (``bos:eos`` for token-rate tensors, ``bos * R:eos *
    R`` for the interleaved ones) and the results are concatenated.
    """
    manual_backward = _manual_backward()
    if cu_seqlens is None:
        return manual_backward(q, k, v, g, beta, do, R, initial_state=h0)
    bounds = [int(x) for x in cu_seqlens.tolist()]
    parts = [
        manual_backward(
            q[:, bounds[n] : bounds[n + 1]],
            k[:, bounds[n] * R : bounds[n + 1] * R],
            v[:, bounds[n] * R : bounds[n + 1] * R],
            g[:, bounds[n] : bounds[n + 1]],
            beta[:, bounds[n] * R : bounds[n + 1] * R],
            do[:, bounds[n] : bounds[n + 1]],
            R,
            initial_state=None if h0 is None else h0[n : n + 1],
        )
        for n in range(len(bounds) - 1)
    ]
    dims = (1, 1, 1, 1, 1, 0)  # dq,dk,dv,dg,dbeta concat along time; dh0 along the sequence axis
    return tuple(torch.cat([p[i] for p in parts], dim=d) for i, d in enumerate(dims))


def _run_case(label, B, T, H, K, V, R, BV, use_h0, cu, seed=0):  # type: ignore[no-untyped-def]
    """Build one case, emulate it, and compare all six gradients against the reference.

    :returns: ``True`` if every gradient matches to ``< 1e-10``.
    """
    torch.manual_seed(seed)
    dt = torch.float64
    q = torch.randn(B, T, H, K, dtype=dt)
    k = torch.randn(B, T * R, H, K, dtype=dt)
    v = torch.randn(B, T * R, H, V, dtype=dt)
    g = -torch.rand(B, T, H, K, dtype=dt)
    beta = torch.rand(B, T * R, H, dtype=dt)
    do = torch.randn(B, T, H, V, dtype=dt)
    cu_seqlens = None if cu is None else torch.tensor(cu, dtype=torch.int32)
    N = B if cu_seqlens is None else len(cu) - 1
    h0 = torch.randn(N, H, K, V, dtype=dt) if use_h0 else None

    got = emulate_bwd_kernel(
        q, k, v, g, beta, do, R, initial_state=h0, BV=BV, cu_seqlens=cu_seqlens
    )
    ref = _reference(q, k, v, g, beta, do, R, h0, cu_seqlens)

    diffs = [(a - b).abs().max().item() for a, b in zip(got, ref)]
    ok = all(d < 1e-10 for d in diffs)
    cells = "  ".join(f"{n}={d:.2e}" for n, d in zip(_NAMES, diffs))
    nv = (V + BV - 1) // BV
    tag = f"{label} (BK={_next_power_of_2(K)},BV={BV},NV={nv})"
    print(f"{tag:<40} {cells}   {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for i, case in enumerate(_CASES):
        all_ok &= _run_case(*case, seed=i + 1)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
