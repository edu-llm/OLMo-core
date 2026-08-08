"""Isolate which stage of the Triton backward diverges."""
import torch
from olmo_core.nn.attention.kda_householder import kda_householder_fwd, kda_householder_bwd
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

torch.manual_seed(0)
import sys
B, T, H, K, V, R = [int(x) for x in sys.argv[1:7]]
dev = "cuda"
q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, T*R, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, T*R, H, V, device=dev, dtype=torch.bfloat16)
g = -torch.rand(B, T, H, K, device=dev, dtype=torch.float32)
beta = torch.rand(B, T*R, H, device=dev, dtype=torch.bfloat16)
do = torch.randn(B, T, H, V, device=dev, dtype=torch.bfloat16)
scale = K ** -0.5

# 1. forward agreement (already known good)
o_tri, _ = kda_householder_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                               num_householder=R, initial_state=None,
                               output_final_state=False, cu_seqlens=None)
o_ref, _ = kda_householder_torch(q, k, v, g, beta, num_householder=R, scale=scale)
print(f"B{B} T{T} H{H} K{K} V{V} R{R}: forward diff={(o_tri.float()-o_ref.float()).abs().max().item():.2e}")

# 2. backward
out = kda_householder_bwd(q=q, k=k, v=v, g=g, beta=beta, do=do,
                          num_householder=R, scale=scale,
                          initial_state=None, cu_seqlens=None)
names = ["dq", "dk", "dv", "dg", "dbeta"]

# 3. reference grads for scale
ins = [x.clone().detach().requires_grad_() for x in (q, k, v, g, beta)]
o, _ = kda_householder_torch(*ins, num_householder=R, scale=scale)
o.backward(do)
worst=0
for n, got, ref in zip(names, out, [x.grad for x in ins]):
    d=(got.float()-ref.float()).abs().max().item(); worst=max(worst,d)
    print(f"    {n}={d:.2e}", end="")
print(f"   -> {'PASS' if worst<2e-2 else 'FAIL'}")
