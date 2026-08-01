"""Adversarial numerics audit harness for the KDA+Householder fwd/bwd.

Faithful CPU re-implementation of the *kernel's* arithmetic order (per (b,h), fp32 accumulate
on bf16-rounded inputs), parameterised by accumulate dtype so fp32-vs-fp64 drift is measurable.
"""

import sys
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src")


def fwd(q, k, v, g, beta, R, scale, dt, h0=None, track=False):
    """Forward recurrence in accumulate dtype `dt`. Inputs are cast to dt on entry.

    Returns (o, S, stats) where stats is a list of ||S||_F per token if track.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.to(dt) for x in (q, k, v, g, beta))
    S = torch.zeros(B, H, K, V, dtype=dt) if h0 is None else h0.to(dt)
    outs = []
    norms = []
    for t in range(T):
        S = S * g[:, t][..., None].exp()
        for r in range(R):
            i = t * R + r
            kk, vv, bb = k[:, i], v[:, i], beta[:, i]
            u = bb[..., None] * (vv - (kk[..., None] * S).sum(-2))
            S = S + kk[..., None] * u[..., None, :]
        outs.append(((q[:, t] * scale)[..., None] * S).sum(-2))
        if track:
            norms.append(S.reshape(B * H, -1).norm(dim=-1))
    o = torch.stack(outs, 1) if outs else q.new_zeros(B, 0, H, V)
    return o, S, norms


def bwd(q, k, v, g, beta, do, R, scale, dt, h0=None):
    """Hand-derived reverse-time backward in accumulate dtype `dt` (mirrors the kernel)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta, do = (x.to(dt) for x in (q, k, v, g, beta, do))
    a = g.exp()
    S = torch.zeros(B, H, K, V, dtype=dt) if h0 is None else h0.to(dt)
    pre: List[torch.Tensor] = []
    for t in range(T):
        pre.append(S)
        S = S * a[:, t][..., None]
        for r in range(R):
            i = t * R + r
            u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * S).sum(-2))
            S = S + k[:, i][..., None] * u[..., None, :]
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    dg, dbeta = torch.zeros_like(g), torch.zeros_like(beta)
    dS = torch.zeros(B, H, K, V, dtype=dt)
    for t in reversed(range(T)):
        inner = [pre[t] * a[:, t][..., None]]
        for r in range(R):
            i = t * R + r
            u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * inner[r]).sum(-2))
            inner.append(inner[r] + k[:, i][..., None] * u[..., None, :])
        dq[:, t] = scale * (inner[R] * do[:, t][..., None, :]).sum(-1)
        dS = dS + (q[:, t] * scale)[..., None] * do[:, t][..., None, :]
        for r in reversed(range(R)):
            i = t * R + r
            k_i, b_i = k[:, i], beta[:, i]
            resid = v[:, i] - (k_i[..., None] * inner[r]).sum(-2)
            u = b_i[..., None] * resid
            du = (k_i[..., None] * dS).sum(-2)
            dbeta[:, i] = (du * resid).sum(-1)
            dv[:, i] = b_i[..., None] * du
            dw = -b_i[..., None] * du
            dk[:, i] = (dS * u[..., None, :]).sum(-1) + (inner[r] * dw[..., None, :]).sum(-1)
            dS = dS + k_i[..., None] * dw[..., None, :]
        dg[:, t] = (dS * pre[t] * a[:, t][..., None]).sum(-1)
        dS = dS * a[:, t][..., None]
    return dq, dk, dv, dg, dbeta, dS


def make(B, T, H, K, V, R, seed=0, beta_mode="sigmoid", g_mode="logsigmoid", bf16=True):
    """Build inputs. beta_mode in {sigmoid, sigmoid2, zero, two, one}."""
    gen = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32)

    q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    raw = rnd(B, T * R, H)
    if beta_mode == "sigmoid":
        beta = raw.sigmoid()
    elif beta_mode == "sigmoid2":
        beta = raw.sigmoid() * 2.0
    elif beta_mode == "zero":
        beta = torch.zeros(B, T * R, H)
    elif beta_mode == "two":
        beta = torch.full((B, T * R, H), 2.0)
    elif beta_mode == "one":
        beta = torch.ones(B, T * R, H)
    else:
        raise ValueError(beta_mode)
    if g_mode == "logsigmoid":
        g = torch.nn.functional.logsigmoid(rnd(B, T, H, K))
    elif g_mode == "zero":
        g = torch.zeros(B, T, H, K)
    elif g_mode == "tiny":  # realistic KDA: g = -exp(A_log)*softplus(.) ~ -1e-3..-1e-1
        g = -torch.nn.functional.softplus(rnd(B, T, H, K)) * 0.01
    elif g_mode == "verynegative":
        g = torch.full((B, T, H, K), -200.0)
    else:
        raise ValueError(g_mode)
    do = rnd(B, T, H, V)
    if bf16:
        q, k, v, beta, do = (x.to(torch.bfloat16) for x in (q, k, v, beta, do))
    return q, k, v, g, beta, do
