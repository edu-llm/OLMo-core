"""EXP11: unguarded cu_seqlens. The kernel sizes every workspace off `T` but indexes off
`cu_seqlens`. If cu_seqlens[-1] != T the two disagree; if cu_seqlens[-1] > T the kernel writes
past the end of hs / dq_p / dk_p / db_p (an out-of-bounds heap write, no bounds check anywhere).

(a) compute the exact OOB element count for a plausible mis-specified cu_seqlens
(b) show the torch backend does something DIFFERENT (silently) for cu_seqlens[-1] < T
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/OLMo-core/src")
sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")

from bwd_emulator import emulate_bwd_kernel  # noqa: E402
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch  # noqa: E402

print("=" * 100)
print("EXP11a  OOB accounting: hs is sized B*T*H*K*V but indexed by (bos + t) from cu_seqlens")
print("=" * 100)
for T, cu_last, H, K, V, R in [(64, 64, 2, 64, 64, 2), (64, 80, 2, 64, 64, 2), (8192, 8200, 8, 64, 64, 4)]:
    hs_sz = T * H * K * V
    hs_max_written = cu_last * H * K * V - 1
    dq_sz = T * H * K
    dq_max = cu_last * H * K - 1
    over = max(0, hs_max_written + 1 - hs_sz)
    print(f"  T={T:<6d} cu_seqlens[-1]={cu_last:<6d} -> hs alloc={hs_sz:>12d} elems, "
          f"max write idx={hs_max_written:>12d}  OOB fp32 words={over:>10d} ({over * 4 / 2**20:.1f} MiB)")
    print(f"         {'':>0s}dq_p alloc={dq_sz} per-NV slot, max write idx={dq_max}, "
          f"OOB={max(0, dq_max + 1 - dq_sz)}  -> also corrupts the NV=1.. slots of dq_p/dg_p")
print("  NOTE: no assert anywhere checks cu_seqlens[-1] == T, monotonicity, or cu_seqlens[0] == 0.")

print("\n" + "=" * 100)
print("EXP11b  cu_seqlens[-1] < T: the two backends silently DISAGREE (no error from either)")
print("=" * 100)
B, T, H, K, V, R = 1, 8, 1, 4, 4, 2
dt = torch.float64
gen = torch.Generator().manual_seed(0)
q = torch.randn(B, T, H, K, generator=gen, dtype=dt)
k = torch.randn(B, T * R, H, K, generator=gen, dtype=dt)
v = torch.randn(B, T * R, H, V, generator=gen, dtype=dt)
g = -torch.rand(B, T, H, K, generator=gen, dtype=dt)
beta = torch.rand(B, T * R, H, generator=gen, dtype=dt)
do = torch.randn(B, T, H, V, generator=gen, dtype=dt)
cu = torch.tensor([0, 3, 5], dtype=torch.int32)  # covers only 5 of 8 tokens

for t_ in (q, k, v, g, beta):
    t_.requires_grad_(True)
try:
    o_t, _ = kda_householder_torch(q, k, v, g, beta, num_householder=R, cu_seqlens=cu)
    print(f"  torch backend : returned o.shape={tuple(o_t.shape)}  (T={T} in, "
          f"{o_t.shape[1]} out -> SILENTLY DROPS {T - o_t.shape[1]} tokens)")
except Exception as e:
    print(f"  torch backend : raised {type(e).__name__}: {e}")

got = emulate_bwd_kernel(q.detach(), k.detach(), v.detach(), g.detach(), beta.detach(), do, R,
                        BV=4, cu_seqlens=cu)
print(f"  kernel bwd    : dq.shape={tuple(got[0].shape)} (full T), "
      f"tokens 5..7 grad = {got[0][:, 5:].abs().max().item():.3e} (zeros, never visited)")
print("  -> torch returns a SHORTER o; triton writes a full-length o with the tail untouched")
print("     (torch.empty in fwd => tail of `o` is uninitialised garbage, not zeros)")

print("\n" + "=" * 100)
print("EXP11c  `o` in the forward is torch.empty(): unvisited rows ship uninitialised")
print("=" * 100)
print("  kda_householder_fwd: o = torch.empty(B, T, H, V, ...)  <- line 256")
print("  kernel loop:         for _ in range(0, T)  where T = eos - bos for varlen")
print("  => any token index in [cu_seqlens[-1], T) is NEVER stored, so `o` keeps whatever")
print("     was in the reused caching-allocator block. No NaN, no error, just stale numbers.")
