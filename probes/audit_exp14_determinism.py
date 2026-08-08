"""EXP14: how strong is the determinism check in gpu_bwd_accept.py?

The host returns dq/dk/dbeta cast to bf16 (q/k/beta.dtype) and dv as bf16. `torch.equal` on bf16
can only see differences >= 1 bf16 ULP (~0.4% relative). Reduction-order nondeterminism at fp32
is ~1e-7 relative -- 4 orders of magnitude below what a bf16 comparison can resolve.

So: would the determinism test still print True if the kernel used tl.atomic_add?
Simulate by perturbing the fp32 partials with reduction-order-scale noise, casting, comparing.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_exp2_mutation import build  # noqa: E402
from audit_numerics import bwd  # noqa: E402

print("=" * 104)
print("EXP14  can `torch.equal` on the RETURNED (bf16-cast) gradients detect fp32 nondeterminism?")
print("=" * 104)
q, k, v, g, beta, do = build("real")
K = q.shape[-1]
ref = bwd(q, k, v, g, beta, do, 2, K**-0.5, torch.float32)
NAMES = ["dq", "dk", "dv", "dg", "dbeta"]
# dtype each gradient is cast to on return (see kda_householder_bwd lines 560-563)
RET = {"dq": torch.bfloat16, "dk": torch.bfloat16, "dv": torch.bfloat16,
       "dg": torch.float32, "dbeta": torch.bfloat16}

gen = torch.Generator().manual_seed(9)
print(f"  {'grad':7s}{'return dtype':>14s}{'1 ULP rel':>12s}"
      + "".join(f"{'noise ' + s:>16s}" for s in ("1e-7", "1e-5", "1e-3")))
for n, t in zip(NAMES, ref[:5]):
    dt = RET[n]
    base = t.to(dt)
    ulp = (torch.finfo(dt).eps)
    cells = ""
    for noise in (1e-7, 1e-5, 1e-3):
        pert = (t * (1 + noise * torch.randn(t.shape, generator=gen))).to(dt)
        same = torch.equal(base, pert)
        frac = (base != pert).float().mean().item()
        cells += f"{('EQUAL' if same else f'diff {frac:.1%}'):>16s}"
    print(f"  {n:7s}{str(dt).replace('torch.', ''):>14s}{ulp:>12.2e}{cells}")

print("\n  interpretation: 'EQUAL' means the determinism assertion would PASS even though the")
print("  kernel produced a different fp32 answer. bf16 eps = 7.8e-3, so any reduction-order")
print("  difference below ~0.4% relative is invisible to `torch.equal` on the returned tensors.")
print("  Only `dg` (returned fp32) is a genuine determinism probe.")

print("\n" + "=" * 104)
print("EXP14b  is `torch.sum` over the NV axis order-dependent at all? (partition-order test)")
print("=" * 104)
M = 8192
for NV in (2, 8, 16):
    gen2 = torch.Generator().manual_seed(4)
    p = torch.randn(NV, M, generator=gen2) * torch.logspace(-3, 3, NV).unsqueeze(1)
    s_fwd = p.sum(0)
    s_rev = p.flip(0).sum(0)
    s_pair = p.reshape(NV, M).sum(0)
    print(f"  NV={NV:<3d} sum(0) vs sum(0) of reversed rows: max|diff|={(s_fwd - s_rev).abs().max():.3e}"
          f"   sum(0) repeatable: {torch.equal(s_fwd, s_pair)}")
print("  -> torch.sum(0) is repeatable for a fixed shape/device (the claim holds), but the VALUE")
print("     depends on the NV partition order, so it is not partition-invariant. Fine for")
print("     determinism; means BV/BK retuning changes bits.")
