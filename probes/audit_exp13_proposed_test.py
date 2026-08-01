"""EXP13: validate the PROPOSED test criterion.

Replace the flat `atol=2e-2` with a PER-GRADIENT RELATIVE criterion in the realistic-gate regime:

    err = |got - ref|.max() / (|ref|.max() + eps)   must be < 2e-2

and re-run the full mutation battery. A good criterion catches every mutation in every regime.
Also confirm the criterion does NOT false-positive on genuine fp32-vs-fp64 kernel error.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_exp2_mutation import MUTS, NAMES, build, bwd_mut  # noqa: E402
from audit_numerics import bwd  # noqa: E402

REL = 2e-2


def relerr(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


print("=" * 108)
print("EXP13a  PROPOSED criterion: per-gradient RELATIVE error < 2e-2, in 3 regimes")
print("=" * 108)
regimes = ("accept", "real", "negeig")
refs = {}
inputs = {}
for r in regimes:
    inputs[r] = build(r)
    q, k, v, g, beta, do = inputs[r]
    refs[r] = bwd_mut(q, k, v, g, beta, do, 2, q.shape[-1] ** -0.5, "none")

print(f"  {'mutation':22s}" + "".join(f"{r:>28s}" for r in regimes))
print(f"  {'':22s}" + "".join(f"{'flat 2e-2':>13s}{'REL 2e-2':>15s}" for _ in regimes))
for m in MUTS[1:]:
    row = f"  {m:22s}"
    for r in regimes:
        q, k, v, g, beta, do = inputs[r]
        got = bwd_mut(q, k, v, g, beta, do, 2, q.shape[-1] ** -0.5, m)
        flat = any((a - b).abs().max().item() >= 2e-2 for a, b in zip(got[:5], refs[r][:5]))
        rl = any(relerr(a, b) >= REL for a, b in zip(got[:5], refs[r][:5]))
        row += f"{'CAUGHT' if flat else 'MISS':>13s}{'CAUGHT' if rl else 'MISS':>15s}"
    print(row)

print("\n" + "=" * 108)
print("EXP13b  false-positive check: real fp32-accumulate error under the RELATIVE criterion")
print("        (T swept to 2048 -- the criterion must have margin at LM sequence lengths)")
print("=" * 108)
print(f"  {'regime':10s}{'T':>7s}" + "".join(f"{n + ' rel':>12s}" for n in NAMES) + f"{'worst':>11s}{'margin':>9s}")
for regime in regimes:
    for T in (64, 512, 2048):
        gen = torch.Generator().manual_seed(11)

        def rnd(*s):
            return torch.randn(*s, generator=gen, dtype=torch.float32)

        B, H, K, V, R = 1, 1, 64, 64, 2
        q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16)
        v = rnd(B, T * R, H, V).to(torch.bfloat16)
        do = rnd(B, T, H, V).to(torch.bfloat16)
        if regime == "accept":
            beta = torch.rand(B, T * R, H, generator=gen).to(torch.bfloat16)
            g = -torch.rand(B, T, H, K, generator=gen)
        else:
            mul = 2.0 if regime == "negeig" else 1.0
            beta = (rnd(B, T * R, H).sigmoid() * mul).to(torch.bfloat16)
            A = torch.rand(H, generator=gen) * 15 + 1.0
            g = -A.view(1, 1, H, 1) * torch.nn.functional.softplus(rnd(B, T, H, K) * 0.02)
        a = bwd(q, k, v, g, beta, do, R, K**-0.5, torch.float32)
        b = bwd(q, k, v, g, beta, do, R, K**-0.5, torch.float64)
        es = [relerr(x.double(), y) for x, y in zip(a[:5], b[:5])]
        w = max(es)
        print(f"  {regime:10s}{T:>7d}" + "".join(f"{e:>12.2e}" for e in es)
              + f"{w:>11.2e}{REL / w:>8.0f}x")
