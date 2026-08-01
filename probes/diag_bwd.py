"""Isolate which stage of the Triton backward diverges."""
import torch
from olmo_core.nn.attention.kda_householder import kda_householder_fwd, kda_householder_bwd
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

torch.manual_seed(0)
B, T, H, K, V, R = 1, 4, 1, 8, 8, 1
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
print(f"forward  max|diff| = {(o_tri.float()-o_ref.float()).abs().max().item():.3e}")

# 2. backward
out = kda_householder_bwd(q=q, k=k, v=v, g=g, beta=beta, do=do,
                          num_householder=R, scale=scale,
                          initial_state=None, cu_seqlens=None)
names = ["dq", "dk", "dv", "dg", "dbeta", "dh0"]
for n, t_ in zip(names, out):
    if t_ is None:
        print(f"  {n}: None"); continue
    f = t_.float()
    print(f"  {n}: max|{f.abs().max().item():.3e}|  nan={bool(f.isnan().any())}  "
          f"inf={bool(f.isinf().any())}  allzero={bool((f==0).all())}")

# 3. reference grads for scale
ins = [x.clone().detach().requires_grad_() for x in (q, k, v, g, beta)]
o, _ = kda_householder_torch(*ins, num_householder=R, scale=scale)
o.backward(do)
print("\nreference magnitudes:")
for n, x in zip(names[:5], ins):
    print(f"  {n}: max|{x.grad.float().abs().max().item():.3e}|")
