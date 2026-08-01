"""EXP2: MUTATION TESTING. Inject plausible transcription slips into the kernel's backward math
and ask: would the acceptance test (atol 2e-2, T=64, g=-rand, beta=rand) catch it?

Run in the exact regime of probes/gpu_bwd_accept.py and of the realistic KDA-init regime.
"""

import sys
from typing import List

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src")

NAMES = ["dq", "dk", "dv", "dg", "dbeta"]
ATOL = 2e-2


def bwd_mut(q, k, v, g, beta, do, R, scale, mut="none", dt=torch.float64):
    """Reverse-time backward with an optional injected mutation."""
    B, T, H, K = q.shape
    q, k, v, g, beta, do = (x.to(dt) for x in (q, k, v, g, beta, do))
    V = v.shape[-1]
    a = g.exp()
    S = torch.zeros(B, H, K, V, dtype=dt)
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

        src = inner[0] if mut == "M5_dq_uses_S0" else inner[R]
        dq[:, t] = scale * (src * do[:, t][..., None, :]).sum(-1)
        dS = dS + (q[:, t] * scale)[..., None] * do[:, t][..., None, :]

        for r in reversed(range(R)):
            i = t * R + r
            k_i, b_i = k[:, i], beta[:, i]
            ridx = min(r + 1, R) if mut == "M6_rewalk_offby1" else r
            resid = v[:, i] - (k_i[..., None] * inner[ridx]).sum(-2)
            u = b_i[..., None] * resid
            du = (k_i[..., None] * dS).sum(-2)
            dbeta[:, i] = (du * (u if mut == "M4_dbeta_uses_u" else resid)).sum(-1)
            dv[:, i] = du if mut == "M7_dv_no_beta" else b_i[..., None] * du
            dw = -b_i[..., None] * du
            t1 = (dS * u[..., None, :]).sum(-1)
            t2 = (inner[ridx] * dw[..., None, :]).sum(-1)
            dk[:, i] = t1 if mut == "M3_dk_drop_term2" else t1 + t2
            dS = dS + k_i[..., None] * (du[..., None, :] if mut == "M8_dh_wrong_sign" else dw[..., None, :])

        if mut == "M2_dg_uses_pre":
            dg[:, t] = (dS * pre[t]).sum(-1)
        elif mut == "M1_dg_after_decay":
            dg[:, t] = ((dS * a[:, t][..., None]) * pre[t] * a[:, t][..., None]).sum(-1)
        else:
            dg[:, t] = (dS * pre[t] * a[:, t][..., None]).sum(-1)
        dS = dS * a[:, t][..., None]
    return dq, dk, dv, dg, dbeta, dS


MUTS = [
    "none",
    "M1_dg_after_decay",
    "M2_dg_uses_pre",
    "M3_dk_drop_term2",
    "M4_dbeta_uses_u",
    "M5_dq_uses_S0",
    "M6_rewalk_offby1",
    "M7_dv_no_beta",
    "M8_dh_wrong_sign",
]


def build(regime, B=2, T=64, H=2, K=64, V=64, R=2, seed=7):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32)

    q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16)
    v = rnd(B, T * R, H, V).to(torch.bfloat16)
    if regime == "accept":
        beta = torch.rand(B, T * R, H, generator=gen).to(torch.bfloat16)
        g = -torch.rand(B, T, H, K, generator=gen)
    elif regime == "real":  # KDA init: A_log = log U(1,16), dt_bias = 0, allow_neg_eigval=False
        beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16)
        A = torch.rand(H, generator=gen) * 15 + 1.0
        g = -A.view(1, 1, H, 1) * torch.nn.functional.softplus(rnd(B, T, H, K) * 0.02)
    elif regime == "negeig":  # allow_neg_eigval=True -> beta in (0, 2)
        beta = (rnd(B, T * R, H).sigmoid() * 2.0).to(torch.bfloat16)
        A = torch.rand(H, generator=gen) * 15 + 1.0
        g = -A.view(1, 1, H, 1) * torch.nn.functional.softplus(rnd(B, T, H, K) * 0.02)
    else:
        raise ValueError(regime)
    do = rnd(B, T, H, V).to(torch.bfloat16)
    return q, k, v, g, beta, do


for regime in ("accept", "real", "negeig"):
    q, k, v, g, beta, do = build(regime)
    R, K = 2, q.shape[-1]
    scale = K**-0.5
    ref = bwd_mut(q, k, v, g, beta, do, R, scale, "none")
    print("=" * 104)
    print(f"REGIME {regime}   (B2 T64 H2 K64 V64 R2)   acceptance criterion: max|diff| < {ATOL}")
    print(f"  |ref| max: " + "  ".join(f"{n}={t.abs().max():.3e}" for n, t in zip(NAMES, ref[:5])))
    print("=" * 104)
    print(f"  {'mutation':22s} " + "".join(f"{n:>11s}" for n in NAMES) + "   VERDICT")
    for m in MUTS[1:]:
        got = bwd_mut(q, k, v, g, beta, do, R, scale, m)
        diffs = [(a - b).abs().max().item() for a, b in zip(got[:5], ref[:5])]
        caught = any(d >= ATOL for d in diffs)
        print(
            f"  {m:22s} "
            + "".join(f"{d:>11.2e}" for d in diffs)
            + f"   {'CAUGHT' if caught else '*** PASSES (silent corruption) ***'}"
        )
    print()
