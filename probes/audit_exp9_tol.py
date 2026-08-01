"""EXP9: how much margin does atol=2e-2 actually have?

(a) absolute fp32-vs-fp64 accumulate drift per gradient at the acceptance shape -> the tightest
    tolerance the kernel could pass with.
(b) NV-partition sensitivity in fp32: does splitting V into NV=8 partials and summing change the
    answer vs a single tile? (the determinism claim's blind spot: reproducible != partition-
    independent)
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_exp2_mutation import NAMES, build  # noqa: E402
from audit_numerics import bwd  # noqa: E402

print("=" * 106)
print("EXP9a  absolute fp32-vs-fp64 drift at the ACCEPTANCE shape (B2 T64 H2 K64 V64 R2)")
print("       -> the acceptance atol=2e-2 has this much headroom over real kernel error")
print("=" * 106)
print(f"  {'regime':10s}" + "".join(f"{n:>12s}" for n in NAMES) + f"{'atol needed':>14s}{'headroom':>11s}")
for regime in ("accept", "real", "negeig"):
    q, k, v, g, beta, do = build(regime)
    K = q.shape[-1]
    a = bwd(q, k, v, g, beta, do, 2, K**-0.5, torch.float32)
    b = bwd(q, k, v, g, beta, do, 2, K**-0.5, torch.float64)
    ds = [(x.double() - y).abs().max().item() for x, y in zip(a[:5], b[:5])]
    worst = max(ds)
    print(f"  {regime:10s}" + "".join(f"{d:>12.2e}" for d in ds)
          + f"{worst:>14.2e}{2e-2 / worst:>10.0f}x")

print("\n" + "=" * 106)
print("EXP9b  NV-partition sensitivity: fp32 partials summed over NV vs one monolithic fp32 tile")
print("       (bwd() here is monolithic; the partitioned variant splits V into BV=8 chunks)")
print("=" * 106)


def bwd_partitioned(q, k, v, g, beta, do, R, scale, BV, dt=torch.float32):
    """Run bwd on each BV-slice of V independently, then sum the V-contracting partials."""
    V = v.shape[-1]
    NV = -(-V // BV)
    acc = None
    for i in range(NV):
        sl = slice(i * BV, min((i + 1) * BV, V))
        gs = bwd(q, k, v[..., sl], g, beta, do[..., sl], R, scale, dt)
        if acc is None:
            acc = [gs[0].clone(), gs[1].clone(), None, gs[3].clone(), gs[4].clone()]
            dvs = [gs[2]]
        else:
            for j, idx in ((0, 0), (1, 1), (3, 3), (4, 4)):
                acc[j] += gs[idx]
            dvs.append(gs[2])
    acc[2] = torch.cat(dvs, dim=-1)
    return acc


for regime in ("accept", "real", "negeig"):
    q, k, v, g, beta, do = build(regime)
    K = q.shape[-1]
    mono = bwd(q, k, v, g, beta, do, 2, K**-0.5, torch.float32)
    part = bwd_partitioned(q, k, v, g, beta, do, 2, K**-0.5, BV=8)
    f64 = bwd(q, k, v, g, beta, do, 2, K**-0.5, torch.float64)
    print(f"  {regime}")
    for n, m, p, r in zip(NAMES, mono[:5], part, f64[:5]):
        print(f"    {n:6s} |mono-part|={(m - p).abs().max():.2e}   "
              f"|mono-f64|={(m.double() - r).abs().max():.2e}   "
              f"|part-f64|={(p.double() - r).abs().max():.2e}")
