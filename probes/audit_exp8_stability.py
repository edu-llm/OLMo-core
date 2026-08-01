"""EXP8: does the recurrence stay bounded at beta -> 2 with bf16-rounded L2-normalized k?

(I - beta k k^T) with beta = 2 and ||k|| = 1 EXACTLY is a reflection (norm-preserving).
bf16 rounding makes ||k|| != 1, so the eigenvalue is 1 - beta*||k||^2 whose magnitude can exceed 1.
Measure the realised ||k||^2 spectrum and the state-norm growth over long T at g = 0 (no decay),
plus whether fp32 overflows to Inf.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_numerics import fwd  # noqa: E402

print("=" * 100)
print("EXP8a  realised ||k||^2 after L2-normalize-in-fp32 then cast to bf16 (K=64)")
print("=" * 100)
gen = torch.Generator().manual_seed(0)
k32 = torch.nn.functional.normalize(torch.randn(200000, 64, generator=gen), p=2, dim=-1)
for dt, lbl in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
    n2 = k32.to(dt).float().pow(2).sum(-1)
    print(f"  {lbl}: ||k||^2  min={n2.min():.6f}  max={n2.max():.6f}  mean={n2.mean():.6f}  "
          f"frac>1: {(n2 > 1).float().mean():.3f}")
    print(f"        eigenvalue |1 - 2*||k||^2|  max={(1 - 2 * n2).abs().max():.6f}  "
          f"frac>1: {((1 - 2 * n2).abs() > 1).float().mean():.3f}")

print("\n" + "=" * 100)
print("EXP8b  state-norm growth, g = 0 (no decay), beta = 2 exactly, k L2-normed then bf16")
print("       R factors per token -> T*R reflections total")
print("=" * 100)
K = V = 64
for R in (1, 4):
    for T in (256, 1024, 4096):
        gen = torch.Generator().manual_seed(5)
        q = torch.nn.functional.normalize(
            torch.randn(1, T, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(
            torch.randn(1, T * R, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
        v = torch.randn(1, T * R, 1, V, generator=gen).to(torch.bfloat16)
        beta = torch.full((1, T * R, 1), 2.0).to(torch.bfloat16)
        g = torch.zeros(1, T, 1, K)
        o, S, norms = fwd(q, k, v, g, beta, R, K**-0.5, torch.float32, track=True)
        ns = torch.stack(norms).reshape(-1)
        print(f"  R={R} T={T:5d}  ||S||F: t0={ns[0]:.3e} t=T/4={ns[T // 4]:.3e} "
              f"t=T/2={ns[T // 2]:.3e} tEnd={ns[-1]:.3e}   |o|max={o.abs().max():.3e}  "
              f"finite={torch.isfinite(S).all().item()}")

print("\n" + "=" * 100)
print("EXP8c  same but beta = 2*sigmoid(randn) (realistic allow_neg_eigval=True)")
print("=" * 100)
for R in (1, 4):
    for T in (1024, 4096):
        gen = torch.Generator().manual_seed(5)
        q = torch.nn.functional.normalize(
            torch.randn(1, T, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(
            torch.randn(1, T * R, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
        v = torch.randn(1, T * R, 1, V, generator=gen).to(torch.bfloat16)
        beta = (torch.randn(1, T * R, 1, generator=gen).sigmoid() * 2).to(torch.bfloat16)
        g = torch.zeros(1, T, 1, K)
        o, S, norms = fwd(q, k, v, g, beta, R, K**-0.5, torch.float32, track=True)
        ns = torch.stack(norms).reshape(-1)
        print(f"  R={R} T={T:5d}  ||S||F: t0={ns[0]:.3e} tEnd={ns[-1]:.3e}  "
              f"|o|max={o.abs().max():.3e}  finite={torch.isfinite(S).all().item()}")

print("\n" + "=" * 100)
print("EXP8d  can fp32 state overflow to Inf/NaN with no error? beta=2, g=0, v scaled up")
print("=" * 100)
for vscale in (1.0, 1e2, 1e4):
    T, R = 512, 4
    gen = torch.Generator().manual_seed(5)
    q = torch.nn.functional.normalize(
        torch.randn(1, T, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(1, T * R, 1, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
    v = (torch.randn(1, T * R, 1, V, generator=gen) * vscale).to(torch.bfloat16)
    beta = torch.full((1, T * R, 1), 2.0).to(torch.bfloat16)
    g = torch.zeros(1, T, 1, K)
    o, S, _ = fwd(q, k, v, g, beta, R, K**-0.5, torch.float32)
    print(f"  v*{vscale:.0e}: ||S||F={S.reshape(-1).norm():.3e}  |o|max={o.abs().max():.3e}  "
          f"o finite={torch.isfinite(o).all().item()}  S finite={torch.isfinite(S).all().item()}")

print("\n" + "=" * 100)
print("EXP8e  NaN/Inf pass-through: does a single Inf/NaN input silently produce NaN output?")
print("=" * 100)
T, R = 8, 2
gen = torch.Generator().manual_seed(1)
base = lambda: (  # noqa: E731
    torch.nn.functional.normalize(torch.randn(1, T, 1, 8, generator=gen), p=2, dim=-1),
    torch.nn.functional.normalize(torch.randn(1, T * R, 1, 8, generator=gen), p=2, dim=-1),
    torch.randn(1, T * R, 1, 8, generator=gen),
    -torch.rand(1, T, 1, 8, generator=gen),
    torch.rand(1, T * R, 1, generator=gen),
)
for name, mut in [
    ("g = +100 (positive gate)", lambda t: t[3].fill_(100.0)),
    ("g = +800 (exp -> inf)", lambda t: t[3].fill_(800.0)),
    ("v has one inf", lambda t: t[2].view(-1).__setitem__(3, float("inf"))),
    ("beta has one nan", lambda t: t[4].view(-1).__setitem__(1, float("nan"))),
    ("k all zeros", lambda t: t[1].zero_()),
]:
    ts = list(base())
    mut(ts)
    o, S, _ = fwd(ts[0], ts[1], ts[2], ts[3], ts[4], R, 8**-0.5, torch.float32)
    print(f"  {name:26s} o finite={torch.isfinite(o).all().item():<5}  "
          f"o nan_frac={torch.isnan(o).float().mean():.3f}  |o|max={o.abs().max():.3e}")
