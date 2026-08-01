"""Did adding the backward regress the FORWARD kernel? Test fwd in isolation."""
import torch
from olmo_core.nn.attention.kda_householder import kda_householder_fwd
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

for (B, T, H, K, V, R) in [(1,4,1,8,8,1), (1,64,1,8,8,1), (1,4,1,64,64,1),
                            (1,64,1,64,64,1), (1,64,1,64,64,2)]:
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T*R, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T*R, H, V, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, K, device=dev, dtype=torch.float32)
    beta = torch.rand(B, T*R, H, device=dev, dtype=torch.bfloat16)
    s = K ** -0.5
    o1, _ = kda_householder_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=s, num_householder=R,
                                initial_state=None, output_final_state=False, cu_seqlens=None)
    o2, _ = kda_householder_torch(q, k, v, g, beta, num_householder=R, scale=s)
    d = (o1.float() - o2.float()).abs().max().item()
    print(f"  B{B} T{T:3d} H{H} K{K:3d} V{V:3d} R{R}: fwd diff = {d:.3e}  {'OK' if d < 2e-2 else 'REGRESSED'}")
