"""EXP4: does fp32 state accumulation drift over long T?

Runs the *kernel's* arithmetic order at fp32 and at fp64 on the SAME bf16-rounded inputs and
reports relative drift of o, S_T and every gradient as T grows. This isolates accumulate
precision (the only thing the kernel could do differently) from input quantisation.
"""

import sys
import time

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_numerics import bwd, fwd  # noqa: E402

NAMES = ["dq", "dk", "dv", "dg", "dbeta"]


def build(T, regime, B=1, H=1, K=64, V=64, R=2, seed=3):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32)

    q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16)
    v = rnd(B, T * R, H, V).to(torch.bfloat16)
    if regime == "accept":
        beta = torch.rand(B, T * R, H, generator=gen).to(torch.bfloat16)
        g = -torch.rand(B, T, H, K, generator=gen)
    elif regime == "real":
        beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16)
        A = torch.rand(H, generator=gen) * 15 + 1.0
        g = -A.view(1, 1, H, 1) * torch.nn.functional.softplus(rnd(B, T, H, K) * 0.02)
    elif regime == "nodecay":
        beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16)
        g = torch.zeros(B, T, H, K)
    do = rnd(B, T, H, V).to(torch.bfloat16)
    return q, k, v, g, beta, do


def rel(a, b):
    d = (a.double() - b.double()).abs().max().item()
    s = b.double().abs().max().item()
    return d, d / max(s, 1e-300)


print("=" * 112)
print("EXP4  fp32-accumulate vs fp64-accumulate drift, identical bf16 inputs (B1 H1 K64 V64 R2)")
print("      absolute max|diff| and relative-to-fp64-max. bf16 round-trip of o is ~4e-3 for scale.")
print("=" * 112)
for regime in ("accept", "real", "nodecay"):
    print(f"\n regime={regime}")
    print(f"   {'T':>6s} {'|S_T|F':>11s} {'o abs':>10s} {'o rel':>9s} "
          + "".join(f"{n + ' rel':>11s}" for n in NAMES) + f"{'wall_s':>8s}")
    for T in (64, 256, 1024, 2048):
        q, k, v, g, beta, do = build(T, regime)
        t0 = time.time()
        o32, S32, _ = fwd(q, k, v, g, beta, 2, 64**-0.5, torch.float32)
        o64, S64, _ = fwd(q, k, v, g, beta, 2, 64**-0.5, torch.float64)
        gr32 = bwd(q, k, v, g, beta, do, 2, 64**-0.5, torch.float32)
        gr64 = bwd(q, k, v, g, beta, do, 2, 64**-0.5, torch.float64)
        oa, orl = rel(o32, o64)
        cells = "".join(f"{rel(a, b)[1]:>11.2e}" for a, b in zip(gr32[:5], gr64[:5]))
        print(f"   {T:>6d} {S64.reshape(-1).norm().item():>11.3e} {oa:>10.2e} {orl:>9.2e} "
              f"{cells}{time.time() - t0:>8.1f}")
