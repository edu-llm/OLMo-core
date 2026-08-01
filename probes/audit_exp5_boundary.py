"""EXP5: degenerate & boundary inputs through the KERNEL-FAITHFUL emulator (bwd_emulator),
checked against manual_backward (the gradcheck-validated reference).

Every case is a shape/regime the 260-test suite and gpu_bwd_accept.py never touch.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src")

from bwd_emulator import emulate_bwd_kernel  # noqa: E402
from manual_backward_check import manual_backward  # noqa: E402

NAMES = ("dq", "dk", "dv", "dg", "dbeta", "dh0")

CASES = [
    # label, B,T,H,K,V,R, BV, beta_mode, g_mode, h0
    ("T=1  R=1", 1, 1, 1, 4, 4, 1, 4, "rand", "neg", False),
    ("T=1  R=8", 1, 1, 1, 4, 4, 8, 4, "rand", "neg", False),
    ("T=2  R=8", 1, 2, 1, 8, 8, 8, 8, "rand", "neg", False),
    ("R=8  T=5", 2, 5, 2, 8, 8, 8, 4, "rand", "neg", False),
    ("beta=0 exact", 2, 5, 2, 4, 4, 2, 4, "zero", "neg", False),
    ("beta=1 exact", 2, 5, 2, 4, 4, 2, 4, "one", "neg", False),
    ("beta=2 exact", 2, 5, 2, 4, 4, 2, 4, "two", "neg", False),
    ("beta=2 +l2norm k", 2, 5, 2, 4, 4, 2, 4, "two_l2", "neg", False),
    ("g=0 (no decay)", 2, 5, 2, 4, 4, 2, 4, "rand", "zero", False),
    ("g=-200 underflow", 2, 5, 2, 4, 4, 2, 4, "rand", "vneg", False),
    ("g=-745 to 0.0", 2, 5, 2, 4, 4, 2, 4, "rand", "denorm", False),
    ("K=3 (BK=4)", 2, 5, 2, 3, 4, 2, 4, "rand", "neg", False),
    ("K=5 (BK=8)", 2, 5, 2, 5, 4, 2, 4, "rand", "neg", False),
    ("K=33 (BK=64)", 1, 4, 1, 33, 8, 2, 8, "rand", "neg", False),
    ("V=1 (BV=1)", 2, 5, 2, 4, 1, 2, 1, "rand", "neg", False),
    ("V=3 < BV=4", 2, 5, 2, 4, 3, 2, 4, "rand", "neg", False),
    ("V=9 NV=2 partial", 2, 5, 2, 4, 9, 2, 8, "rand", "neg", False),
    ("K=33 V=9 R=3", 1, 4, 2, 33, 9, 3, 8, "rand", "neg", True),
    ("varlen empty seq", 1, 6, 2, 4, 4, 2, 4, "rand", "neg", False),
    ("varlen len-1 seqs", 1, 4, 2, 4, 4, 2, 4, "rand", "neg", False),
]
VARLEN = {"varlen empty seq": (0, 3, 3, 6), "varlen len-1 seqs": (0, 1, 2, 3, 4)}


def build(B, T, H, K, V, R, beta_mode, g_mode, use_h0, N, seed):
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64

    def rnd(*s):
        return torch.randn(*s, generator=gen, dtype=dt)

    q = rnd(B, T, H, K)
    k = rnd(B, T * R, H, K)
    if beta_mode == "two_l2":
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    if beta_mode == "rand":
        beta = torch.rand(B, T * R, H, generator=gen, dtype=dt)
    elif beta_mode == "zero":
        beta = torch.zeros(B, T * R, H, dtype=dt)
    elif beta_mode == "one":
        beta = torch.ones(B, T * R, H, dtype=dt)
    else:
        beta = torch.full((B, T * R, H), 2.0, dtype=dt)
    if g_mode == "neg":
        g = -torch.rand(B, T, H, K, generator=gen, dtype=dt)
    elif g_mode == "zero":
        g = torch.zeros(B, T, H, K, dtype=dt)
    elif g_mode == "vneg":
        g = torch.full((B, T, H, K), -200.0, dtype=dt)
    else:
        g = torch.full((B, T, H, K), -800.0, dtype=dt)  # exp -> 0.0 exactly
    do = rnd(B, T, H, V)
    h0 = rnd(N, H, K, V) if use_h0 else None
    return q, k, v, g, beta, do, h0


def ref_varlen(q, k, v, g, beta, do, R, h0, cu):
    bounds = [int(x) for x in cu]
    parts = []
    for n in range(len(bounds) - 1):
        b, e = bounds[n], bounds[n + 1]
        if e == b:  # empty sequence: zero grads, zero dh0
            B, _, H, K = q.shape
            V = v.shape[-1]
            parts.append((
                q.new_zeros(B, 0, H, K), k.new_zeros(B, 0, H, K), v.new_zeros(B, 0, H, V),
                g.new_zeros(B, 0, H, K), beta.new_zeros(B, 0, H), q.new_zeros(1, H, K, V),
            ))
            continue
        parts.append(manual_backward(
            q[:, b:e], k[:, b * R:e * R], v[:, b * R:e * R], g[:, b:e],
            beta[:, b * R:e * R], do[:, b:e], R,
            initial_state=None if h0 is None else h0[n:n + 1],
        ))
    dims = (1, 1, 1, 1, 1, 0)
    return tuple(torch.cat([p[i] for p in parts], dim=d) for i, d in enumerate(dims))


print("=" * 118)
print("EXP5  degenerate/boundary inputs: bwd_emulator (kernel-faithful indexing) vs manual_backward")
print("      all float64; PASS threshold 1e-10; 'nonfinite' flags NaN/Inf that would ship silently")
print("=" * 118)
print(f"  {'case':20s}" + "".join(f"{n:>11s}" for n in NAMES) + "  verdict")
nfail = 0
for i, (label, B, T, H, K, V, R, BV, bm, gm, h0f) in enumerate(CASES):
    cu = VARLEN.get(label)
    cu_t = None if cu is None else torch.tensor(cu, dtype=torch.int32)
    N = B if cu is None else len(cu) - 1
    q, k, v, g, beta, do, h0 = build(B, T, H, K, V, R, bm, gm, h0f, N, seed=100 + i)
    try:
        got = emulate_bwd_kernel(q, k, v, g, beta, do, R, initial_state=h0, BV=BV, cu_seqlens=cu_t)
    except Exception as e:
        print(f"  {label:20s}  *** RAISED {type(e).__name__}: {str(e)[:60]}")
        nfail += 1
        continue
    ref = ref_varlen(q, k, v, g, beta, do, R, h0, cu) if cu else \
        manual_backward(q, k, v, g, beta, do, R, initial_state=h0)
    diffs, nf = [], []
    for a, b in zip(got, ref):
        fa, fb = torch.isfinite(a).all().item(), torch.isfinite(b).all().item()
        nf.append(not fa)
        if fa and fb:
            s = b.abs().max().item()
            diffs.append((a - b).abs().max().item() / max(s, 1.0))
        else:
            diffs.append(float("nan"))
    ok = all((d == d and d < 1e-10) for d in diffs)
    flag = "  NONFINITE:" + ",".join(n for n, x in zip(NAMES, nf) if x) if any(nf) else ""
    nfail += (not ok)
    print(f"  {label:20s}" + "".join(f"{d:>11.2e}" for d in diffs)
          + f"  {'PASS' if ok else 'FAIL'}{flag}")
print(f"\n  {nfail} case(s) failed / errored out of {len(CASES)}")
