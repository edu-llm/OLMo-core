"""Reference (naive) recurrence for KDA with multiple Householder / delta-product factors.

This is the cross of two fla-core 0.4.1 reference kernels:

* ``fla.ops.kda.naive.naive_recurrent_kda``                     -> per-channel log-space decay ``g``
* ``fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product``
                                                                -> ``R`` delta updates per token

Run directly to execute the two equivalence checks::

    python probes/naive_kda_householder.py
"""

from __future__ import annotations

import torch

__all__ = ["naive_recurrent_kda_householder"]


def _recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    scale: float,
    initial_state: torch.Tensor | None,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared recurrence body, parameterised by the internal accumulation dtype.

    Split out purely so the test block can re-run the *identical* code path in ``float64`` to
    separate mathematical error from ``float32`` round-off. The public entry point always uses
    ``float32``, matching the fla naive-kernel convention (``x.to(torch.float)``).

    :param q: Queries, shape ``[B, T, H, K]``.
    :param k: Keys, shape ``[B, T * R, H, K]``.
    :param v: Values, shape ``[B, T * R, H, V]``.
    :param g: Log-space per-channel decay, shape ``[B, T, H, K]``.
    :param beta: Delta-rule step sizes, shape ``[B, T * R, H]``.
    :param num_householder: Number of Householder/delta factors ``R`` applied per token.
    :param scale: Query scaling factor.
    :param initial_state: Optional initial state, shape ``[B, H, K, V]``.
    :param compute_dtype: Floating dtype used for all internal arithmetic.
    :returns: ``(o, S)`` in ``compute_dtype``, shapes ``[B, T, H, V]`` and ``[B, H, K, V]``.
    """
    R = num_householder
    B, T, H, K, V = *q.shape, v.shape[-1]

    q, k, v, g, beta = (x.to(compute_dtype) for x in (q, k, v, g, beta))
    q = q * scale

    S = torch.zeros(B, H, K, V, dtype=compute_dtype, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(compute_dtype)
    o = torch.zeros(B, T, H, V, dtype=compute_dtype, device=q.device)
    for i in range(T):
        # 1. KDA: a single per-channel decay of the state for this token.
        S = S * g[:, i][..., None].exp()
        # 2. DeltaProduct: R successive rank-1 delta updates.
        for j in range(R):
            k_ij, v_ij, b_ij = k[:, i * R + j], v[:, i * R + j], beta[:, i * R + j]
            S = S + torch.einsum(
                "b h k, b h v -> b h k v",
                b_ij[..., None] * k_ij,
                v_ij - (k_ij[..., None] * S).sum(-2),
            )
        # 3. Readout.
        o[:, i] = torch.einsum("b h k, b h k v -> b h v", q[:, i], S)
    return o, S


def naive_recurrent_kda_householder(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int = 1,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference recurrence: KDA per-channel decay + R Householder factors per token.

    Per token ``t`` (with ``R = num_householder``):

    1. decay the state once, per-channel along ``K``: ``S <- S * exp(g[:, t])``
    2. for ``r`` in ``0..R-1`` apply one delta / Householder update built from
       ``k[:, t * R + r]``, ``v[:, t * R + r]``, ``beta[:, t * R + r]``
    3. read out ``o[:, t] = (S * q[:, t, ..., None]).sum(-2)``

    All arithmetic is performed in ``float32`` (matching the fla naive kernels) and the output is
    cast back to ``v.dtype``.

    :param q: Queries, shape ``[B, T, H, K]``.
    :param k: Keys, shape ``[B, T * R, H, K]``, interleaved along the time axis.
    :param v: Values, shape ``[B, T * R, H, V]``, interleaved along the time axis.
    :param g: Log-space per-channel decay, shape ``[B, T, H, K]``, applied once per token.
    :param beta: Delta-rule step sizes, shape ``[B, T * R, H]``, interleaved along time.
    :param num_householder: Number of Householder/delta factors ``R`` applied per token.
    :param scale: Query scaling. Defaults to ``K ** -0.5``.
    :param initial_state: Optional initial state, shape ``[B, H, K, V]``.
    :param output_final_state: If ``True`` also return the final state, else return ``None``.
    :returns: ``(o, S)`` with ``o`` of shape ``[B, T, H, V]`` and dtype ``v.dtype``, and ``S`` the
        final state ``[B, H, K, V]`` in ``float32`` (or ``None``).
    """
    dtype = v.dtype
    R = num_householder
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    assert k.shape == (B, T * R, H, K), f"expected k {(B, T * R, H, K)}, got {tuple(k.shape)}"
    assert v.shape == (B, T * R, H, V), f"expected v {(B, T * R, H, V)}, got {tuple(v.shape)}"
    assert beta.shape == (B, T * R, H), f"expected beta {(B, T * R, H)}, got {tuple(beta.shape)}"
    assert g.shape == (B, T, H, K), f"expected g {(B, T, H, K)}, got {tuple(g.shape)}"

    o, S = _recurrence(q, k, v, g, beta, R, scale, initial_state, torch.float32)
    if not output_final_state:
        S = None
    return o.to(dtype), S


# ---------------------------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------------------------

def _kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Verbatim copy of ``naive_recurrent_kda`` from fla-core 0.4.1.

    Source: ``/tmp/fc/fla/ops/kda/naive.py`` lines 7-36 (module ``fla.ops.kda.naive``).
    Copied rather than imported because ``fla`` is not installed in this environment.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def _gdp_reference(q, k, v, g, beta, scale, cu_seqlens,
                   initial_state=None, output_final_state=False,
                   num_householder=1, _dtype=torch.float32):
    """Copy of ``naive_recurrent_gated_delta_product`` from fla-core 0.4.1.

    Source: ``/tmp/fc/fla/ops/gated_delta_product/naive.py`` lines 4-36
    (module ``fla.ops.gated_delta_product.naive``).
    Copied rather than imported because ``fla`` is not installed in this environment.

    The body is verbatim except for the ``_dtype`` hook, which replaces the three hard-coded
    ``float32`` sites so the diagnostic can re-run the reference in ``float64``. Calling it with
    the default ``_dtype=torch.float32`` reproduces the upstream function exactly.

    Two unmodified quirks of the upstream reference, both load-bearing for the checks below:

    * ``scale`` is accepted but never applied to ``q`` -- it behaves as ``scale == 1.0``.
    * ``g`` is the one input never passed through ``.float()``. With ``float64`` ``g`` the state
      ``h`` gets type-promoted to ``float64``, but the ``float32`` ``o`` buffer truncates every
      readout, so the returned tensor carries ``float32`` round-off regardless of input dtype.
    """
    q_original_dtype = q.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T*num_householder, H, K)
    assert v.shape == (B, T*num_householder, H, V)
    assert beta.shape == (B, T*num_householder, H)
    if g is not None:
        assert g.shape == (B, T, H)
    q, k, v, beta = map(lambda x: x.to(_dtype), (q, k, v, beta))  # upstream: x.float()

    h = torch.zeros(B, H, K, V, dtype=_dtype, device=q.device)
    if initial_state is not None:
        h = initial_state

    o = torch.zeros(B, T, H, V, dtype=_dtype, device=q.device)

    for i in range(T):
        if g is not None:
            h = h * g[:, i, :].exp()[..., None, None]
        # multiple state transition
        for j in range(num_householder):
            k_ij = k[:, i*num_householder+j, :, :]
            v_ij = v[:, i*num_householder+j, :, :]
            beta_ij = beta[:, i*num_householder+j, :]
            h = h + (v_ij - (h * k_ij[..., None]).sum(-2)).unsqueeze(-2) * k_ij[..., None] * beta_ij[..., None, None]
        # memory readout
        q_i = q[:, i, :, :]
        o_i = (h * q_i[..., None]).sum(-2)
        o[:, i] = o_i
    return o.to(q_original_dtype), h


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    """:returns: max absolute difference between ``a`` and ``b``, computed in ``float64``."""
    return (a.double() - b.double()).abs().max().item()


def _main() -> None:
    torch.manual_seed(0)
    B, T, H, K, V = 2, 8, 2, 4, 4
    dt = torch.float64
    atol = 1e-10
    failures: list[str] = []

    # ----------------------------------------------------------------------------- CHECK 1
    # R = 1 must reproduce naive_recurrent_kda exactly.
    q = torch.randn(B, T, H, K, dtype=dt)
    k = torch.randn(B, T, H, K, dtype=dt)
    v = torch.randn(B, T, H, V, dtype=dt)
    beta = torch.rand(B, T, H, dtype=dt)
    g = -torch.rand(B, T, H, K, dtype=dt).exp()  # log-space decay, per-channel, g <= 0

    o_mine, _ = naive_recurrent_kda_householder(q, k, v, g, beta, num_householder=1)
    o_ref, _ = _kda_reference(q, k, v, g, beta)
    d1 = _maxdiff(o_mine, o_ref)
    p1 = torch.allclose(o_mine, o_ref, atol=atol, rtol=0.0)
    if not p1:
        failures.append(f"CHECK 1 (R=1 vs KDA): max|diff| = {d1:.6e} > atol {atol:.0e}")
    print(f"CHECK 1  R=1 vs naive_recurrent_kda   : max|diff| = {d1:.6e}   "
          f"{'PASS' if p1 else 'FAIL'}")

    # ----------------------------------------------------------------------------- CHECK 2
    # g constant along K (per-head) must reproduce naive_recurrent_gated_delta_product.
    # The GDP reference never applies `scale` to q, so both sides are compared at scale = 1.0.
    d2: dict[int, float] = {}
    d2_f64: dict[int, float] = {}
    for R in (2, 3):
        q = torch.randn(B, T, H, K, dtype=dt)
        k = torch.randn(B, T * R, H, K, dtype=dt)
        v = torch.randn(B, T * R, H, V, dtype=dt)
        beta = torch.rand(B, T * R, H, dtype=dt)
        g_head = -torch.rand(B, T, H, dtype=dt).exp()               # [B, T, H]
        g_chan = g_head[..., None].expand(B, T, H, K).contiguous()  # [B, T, H, K]

        o_mine, _ = naive_recurrent_kda_householder(
            q, k, v, g_chan, beta, num_householder=R, scale=1.0
        )
        o_ref, _ = _gdp_reference(
            q, k, v, g_head, beta, scale=1.0, cu_seqlens=None, num_householder=R
        )
        d = _maxdiff(o_mine, o_ref)
        p = torch.allclose(o_mine, o_ref.to(o_mine.dtype), atol=atol, rtol=0.0)
        d2[R] = d
        if not p:
            failures.append(f"CHECK 2 (R={R} vs GDP): max|diff| = {d:.6e} > atol {atol:.0e}")
        print(f"CHECK 2  R={R} vs gated_delta_product : max|diff| = {d:.6e}   "
              f"{'PASS' if p else 'FAIL'}")

        # Diagnostic: identical comparison with BOTH sides run in float64. Same code paths,
        # only the accumulation dtype changes. Isolates math error from float32 round-off.
        o_mine64, _ = _recurrence(q, k, v, g_chan, beta, R, 1.0, None, torch.float64)
        o_ref64, _ = _gdp_reference(
            q, k, v, g_head, beta, scale=1.0, cu_seqlens=None,
            num_householder=R, _dtype=torch.float64,
        )
        d2_f64[R] = _maxdiff(o_mine64, o_ref64)

    # ----------------------------------------------------------------------------- SUMMARY
    print()
    print("float64-accumulation diagnostic (same code paths, float32 round-off removed):")
    for R in (2, 3):
        ulp32 = torch.finfo(torch.float32).eps
        print(f"  R={R}: max|diff| = {d2_f64[R]:.6e}   "
              f"(float32 comparison was {d2[R]:.6e}; 1 float32 ulp at |o|~15 is ~{15 * ulp32:.1e})")

    print()
    print(f"SUMMARY  check1 = {d1:.6e}   check2_R2 = {d2[2]:.6e}   check2_R3 = {d2[3]:.6e}")
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print("RESULT: ALL PASS")


if __name__ == "__main__":
    _main()
