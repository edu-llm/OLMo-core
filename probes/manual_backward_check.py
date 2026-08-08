"""Hand-derived reverse-time backward pass for the KDA + ``R``-Householder recurrence.

The forward being differentiated is :func:`olmo_core.nn.attention.kda_householder_torch`
(per token ``t``, with ``R = num_householder`` and state ``S`` of shape ``[K, V]``)::

    S <- S * exp(g[:, t])[..., None]              # per-channel decay, ONCE per token
    for r in 0 .. R - 1:
        i = t * R + r
        u = beta[:, i] * (v[:, i] - k[:, i] @ S)  # reads the CURRENT S
        S <- S + outer(k[:, i], u)
    o[:, t] = (q[:, t] * scale) @ S               # readout AFTER all R updates

:func:`manual_backward` reproduces the autograd gradients with an explicit reverse-time loop; the
``__main__`` block checks it against autograd in ``float64``.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path("/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src")))

from olmo_core.nn.attention.kda_householder_torch import (  # noqa: E402
    kda_householder_torch,
)

__all__ = ["manual_backward"]


def manual_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    num_householder: int,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Compute gradients of ``(o * do).sum()`` by an explicit reverse-time recurrence.

    :param q: Queries of shape ``[B, T, H, K]`` (un-scaled; ``scale`` is applied internally).
    :param k: Keys of shape ``[B, T * R, H, K]``, interleaved along time.
    :param v: Values of shape ``[B, T * R, H, V]``, interleaved along time.
    :param g: Raw log-space per-channel decay of shape ``[B, T, H, K]`` (no cumsum).
    :param beta: Delta-rule step sizes of shape ``[B, T * R, H]``, interleaved along time.
    :param do: Upstream gradient of the output, shape ``[B, T, H, V]``.
    :param num_householder: Number of Householder / delta factors ``R`` per token.
    :param scale: Query scaling factor. Defaults to ``K ** -0.5``.
    :param initial_state: Optional initial state of shape ``[B, H, K, V]``.

    :returns: ``(dq, dk, dv, dg, dbeta, dh0)`` with the shapes of the corresponding inputs;
        ``dh0`` has shape ``[B, H, K, V]`` and is the gradient w.r.t. ``initial_state`` (a
        zero-state gradient if ``initial_state`` is ``None``).
    """
    R = num_householder
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    dt = torch.float64
    q, k, v, g, beta, do = (x.to(dt) for x in (q, k, v, g, beta, do))
    a = g.exp()  # [B, T, H, K] per-channel decay factor

    # ---- forward, retaining the state *entering* each token (before its decay) ----
    S = (
        torch.zeros(B, H, K, V, dtype=dt, device=q.device)
        if initial_state is None
        else initial_state.to(dt)
    )
    pre: List[torch.Tensor] = []
    for t in range(T):
        pre.append(S)
        S = S * a[:, t][..., None]
        for r in range(R):
            i = t * R + r
            u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * S).sum(-2))
            S = S + k[:, i][..., None] * u[..., None, :]

    # ---- reverse-time loop ----
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dg, dbeta = torch.zeros_like(g), torch.zeros_like(beta)
    dS = torch.zeros(B, H, K, V, dtype=dt, device=q.device)
    for t in reversed(range(T)):
        # Re-materialise the R + 1 intra-token states S^(0) .. S^(R).
        inner: List[torch.Tensor] = [pre[t] * a[:, t][..., None]]
        for r in range(R):
            i = t * R + r
            u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * inner[r]).sum(-2))
            inner.append(inner[r] + k[:, i][..., None] * u[..., None, :])

        # Readout o_t = (q_t * scale) @ S^(R) happens *after* the updates.
        dq[:, t] = scale * (inner[R] * do[:, t][..., None, :]).sum(-1)
        dS = dS + (q[:, t] * scale)[..., None] * do[:, t][..., None, :]

        # The R updates form a chain in S; walk them in reverse r order.
        for r in reversed(range(R)):
            i = t * R + r
            k_i, b_i = k[:, i], beta[:, i]
            resid = v[:, i] - (k_i[..., None] * inner[r]).sum(-2)  # [B, H, V]
            u = b_i[..., None] * resid
            du = (k_i[..., None] * dS).sum(-2)  # [B, H, V]
            dbeta[:, i] = (du * resid).sum(-1)
            dv[:, i] = b_i[..., None] * du
            dw = -b_i[..., None] * du  # grad w.r.t. w = k_i @ S^(r)
            # k_i appears twice: in outer(k_i, u) and inside w = k_i @ S^(r).
            dk[:, i] = (dS * u[..., None, :]).sum(-1) + (inner[r] * dw[..., None, :]).sum(-1)
            dS = dS + k_i[..., None] * dw[..., None, :]  # now grad w.r.t. S^(r)

        # S^(0) = S_{t-1} * exp(g_t): dS is the whole token's accumulated effect.
        dg[:, t] = (dS * pre[t] * a[:, t][..., None]).sum(-1)
        dS = dS * a[:, t][..., None]

    return dq, dk, dv, dg, dbeta, dS


def _check(R: int, use_h0: bool, seed: int = 0) -> bool:
    """Compare :func:`manual_backward` against autograd for one configuration.

    :param R: Number of Householder factors.
    :param use_h0: Whether to pass a non-zero ``initial_state``.
    :param seed: RNG seed.

    :returns: ``True`` if every compared gradient matches to ``< 1e-10``.
    """
    torch.manual_seed(seed)
    B, T, H, K, V = 2, 5, 2, 3, 4
    kw = dict(dtype=torch.float64, requires_grad=True)
    q = torch.randn(B, T, H, K, **kw)
    k = torch.randn(B, T * R, H, K, **kw)
    v = torch.randn(B, T * R, H, V, **kw)
    g = (-torch.rand(B, T, H, K, dtype=torch.float64)).requires_grad_(True)
    beta = torch.rand(B, T * R, H, dtype=torch.float64).requires_grad_(True)
    h0 = torch.randn(B, H, K, V, **kw) if use_h0 else None
    do = torch.randn(B, T, H, V, dtype=torch.float64)

    o, _ = kda_householder_torch(q, k, v, g, beta, num_householder=R, initial_state=h0)
    (o * do).sum().backward()
    ref = [q.grad, k.grad, v.grad, g.grad, beta.grad, None if h0 is None else h0.grad]
    got = manual_backward(q, k, v, g, beta, do, R, initial_state=h0)

    names = ["dq", "dk", "dv", "dg", "dbeta", "dh0"]
    ok = True
    cells = []
    for name, a, b in zip(names, got, ref):
        if b is None:
            cells.append(f"{name}=n/a")
            continue
        diff = (a - b).abs().max().item()
        ok = ok and diff < 1e-10
        cells.append(f"{name}={diff:.3e}")
    label = f"R={R}" + (" +initial_state" if use_h0 else "")
    print(f"{label:<20} {'  '.join(cells)}   {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for r in (1, 2, 3):
        all_ok &= _check(r, use_h0=False, seed=r)
    all_ok &= _check(2, use_h0=True, seed=17)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
