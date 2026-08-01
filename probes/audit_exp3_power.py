"""EXP3: NULL POWER of the flat atol=2e-2 criterion.

For each gradient, ask the strongest possible question: if the kernel returned ZERO for that
gradient (or a 2x-scaled version of it), would `probes/gpu_bwd_accept.py` still print PASS?
A 'PASS' here means the test has literally no power over that output.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_exp2_mutation import NAMES, build, bwd_mut  # noqa: E402

ATOL = 2e-2

print("=" * 100)
print("Null-power of the flat atol=2e-2 acceptance criterion (B2 T64 H2 K64 V64 R2)")
print("  'ZERO passes'  = returning all-zeros for that gradient satisfies max|diff| < 2e-2")
print("  'x2 passes'    = returning 2x the correct gradient satisfies max|diff| < 2e-2")
print("  'x1.01 passes' = a 1% systematic scale error satisfies max|diff| < 2e-2")
print("=" * 100)
for regime, note in [
    ("accept", "gpu_bwd_accept.py's own regime: g = -rand(), beta = rand()"),
    ("real", "realistic KDA init: A_log=logU(1,16), dt_bias=0, beta=sigmoid"),
    ("negeig", "realistic + allow_neg_eigval=True: beta = 2*sigmoid in (0,2)"),
]:
    q, k, v, g, beta, do = build(regime)
    K = q.shape[-1]
    ref = bwd_mut(q, k, v, g, beta, do, 2, K**-0.5, "none")
    print(f"\n {regime}: {note}")
    print(f"   {'grad':7s} {'|ref|max':>11s} {'ZERO':>10s} {'x2':>10s} {'x1.01':>10s}")
    for n, t in zip(NAMES, ref[:5]):
        m = t.abs().max().item()
        print(
            f"   {n:7s} {m:>11.3e} "
            f"{('PASS' if m < ATOL else 'caught'):>10s} "
            f"{('PASS' if m < ATOL else 'caught'):>10s} "
            f"{('PASS' if 0.01 * m < ATOL else 'caught'):>10s}"
        )
