"""EXP6: is the O(R^2) re-walk bit-identical to storing R+1 tiles?

Replays BOTH strategies in fp32 with the exact op order of the kernel (pass-2 forward walk for
`b_sr`, then a separate re-walk from `b_s0` for each `i_r`) and compares bit patterns.
Also compares dk/dbeta computed from re-walked vs stored `inner`.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")

NAMES = ["dq", "dk", "dv", "dg", "dbeta"]


def bwd_variant(q, k, v, g, beta, do, R, scale, dt, store_tiles: bool):
    """store_tiles=True: keep inner[0..R] (emulator strategy).
    store_tiles=False: re-walk from S^(0) for each r (kernel strategy)."""
    B, T, H, K = q.shape
    q, k, v, g, beta, do = (x.to(dt) for x in (q, k, v, g, beta, do))
    V = v.shape[-1]
    a = g.exp()
    S = torch.zeros(B, H, K, V, dtype=dt)
    pre = []
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

    def walk(s0, upto, t):
        s = s0
        for j in range(upto):
            i = t * R + j
            u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * s).sum(-2))
            s = s + k[:, i][..., None] * u[..., None, :]
        return s

    for t in reversed(range(T)):
        s0 = pre[t] * a[:, t][..., None]
        if store_tiles:
            inner = [s0]
            for r in range(R):
                inner.append(walk(inner[r], 1, 0) if False else None)
            inner = [s0]
            for r in range(R):
                i = t * R + r
                u = beta[:, i][..., None] * (v[:, i] - (k[:, i][..., None] * inner[r]).sum(-2))
                inner.append(inner[r] + k[:, i][..., None] * u[..., None, :])
            get = lambda r: inner[r]  # noqa: E731
            sR = inner[R]
        else:
            sR = walk(s0, R, t)  # the kernel's pass-2 forward walk for dq
            get = lambda r: walk(s0, r, t)  # noqa: E731  O(R^2) re-walk
        dq[:, t] = scale * (sR * do[:, t][..., None, :]).sum(-1)
        dS = dS + (q[:, t] * scale)[..., None] * do[:, t][..., None, :]
        for r in reversed(range(R)):
            i = t * R + r
            inr = get(r)
            k_i, b_i = k[:, i], beta[:, i]
            resid = v[:, i] - (k_i[..., None] * inr).sum(-2)
            u = b_i[..., None] * resid
            du = (k_i[..., None] * dS).sum(-2)
            dbeta[:, i] = (du * resid).sum(-1)
            dv[:, i] = b_i[..., None] * du
            dw = -b_i[..., None] * du
            dk[:, i] = (dS * u[..., None, :]).sum(-1) + (inr * dw[..., None, :]).sum(-1)
            dS = dS + k_i[..., None] * dw[..., None, :]
        dg[:, t] = (dS * pre[t] * a[:, t][..., None]).sum(-1)
        dS = dS * a[:, t][..., None]
    return dq, dk, dv, dg, dbeta, dS


print("=" * 100)
print("EXP6  re-walk (kernel) vs stored-tiles (emulator/reference), IDENTICAL fp32 arithmetic")
print("      bit-exact means the O(R^2) restructuring introduces ZERO extra error")
print("=" * 100)
for R in (1, 2, 3, 4, 8):
    gen = torch.Generator().manual_seed(42)
    B, T, H, K, V = 2, 32, 2, 64, 64
    q = torch.nn.functional.normalize(
        torch.randn(B, T, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(B, T * R, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T * R, H, V, generator=gen).to(torch.bfloat16)
    beta = (torch.randn(B, T * R, H, generator=gen).sigmoid() * 2.0).to(torch.bfloat16)
    g = -torch.rand(B, T, H, K, generator=gen)
    do = torch.randn(B, T, H, V, generator=gen).to(torch.bfloat16)
    a32 = bwd_variant(q, k, v, g, beta, do, R, K**-0.5, torch.float32, store_tiles=False)
    b32 = bwd_variant(q, k, v, g, beta, do, R, K**-0.5, torch.float32, store_tiles=True)
    bitex = all(torch.equal(x, y) for x, y in zip(a32, b32))
    diffs = [(x - y).abs().max().item() for x, y in zip(a32[:5], b32[:5])]
    print(f"  R={R}  bit-identical={bitex}   " + "  ".join(
        f"{n}={d:.1e}" for n, d in zip(NAMES, diffs)))
