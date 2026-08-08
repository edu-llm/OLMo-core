"""EXP12: is the untested `dg` actually MATERIAL for training?

`dg` is tiny in the realistic regime (|dg|max ~1.5e-3 << atol 2e-2), so the acceptance test has
ZERO power over it. But dg feeds A_log and f_proj through
    g = -exp(A_log) * softplus(f_proj(x) + dt_bias)
so the chain rule multiplies by exp(A_log) in U(1,16). Measure the resulting parameter gradients
and compare to the *other* parameter gradients in the same layer -- if dA_log / d(f_proj) are the
same order as dW_q etc., an entirely-wrong dg silently trains a wrong gate.
"""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_numerics import bwd  # noqa: E402

torch.manual_seed(0)
B, T, H, K, V, R = 2, 64, 4, 64, 64, 2
D = 256  # d_model

# --- build the realistic gate exactly as recurrent.py does -----------------------------------
A_log = (torch.rand(H) * 15 + 1.0).log().requires_grad_(True)  # log U(1,16)
dt_bias = torch.zeros(H * K).requires_grad_(True)
f1 = torch.nn.Linear(D, V, bias=False)
f2 = torch.nn.Linear(V, H * K, bias=False)
for m in (f1, f2):
    torch.nn.init.normal_(m.weight, std=0.02)
x = torch.randn(B, T, D)

g = -A_log.exp().unsqueeze(-1) * torch.nn.functional.softplus(
    f2(f1(x)).view(B, T, H, K) + dt_bias.view(H, K)
)
print("=" * 100)
print("EXP12  materiality of dg (untested by the acceptance criterion) at layer scale")
print("=" * 100)
print(f"  realistic g: min={g.min():.4f} max={g.max():.4f} mean={g.mean():.4f}")
print(f"  exp(g):      min={g.exp().min():.3e} max={g.exp().max():.4f}")

# --- forward/backward the recurrence to get dg -------------------------------------------------
gen = torch.Generator().manual_seed(1)
q = torch.nn.functional.normalize(
    torch.randn(B, T, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
k = torch.nn.functional.normalize(
    torch.randn(B, T * R, H, K, generator=gen), p=2, dim=-1).to(torch.bfloat16)
v = torch.randn(B, T * R, H, V, generator=gen).to(torch.bfloat16)
beta = torch.randn(B, T * R, H, generator=gen).sigmoid().to(torch.bfloat16)
do = torch.randn(B, T, H, V, generator=gen).to(torch.bfloat16)

grads = bwd(q, k, v, g.detach(), beta, do, R, K**-0.5, torch.float64)
dg = grads[3]
print(f"\n  |dg|max        = {dg.abs().max():.4e}   <-- acceptance atol is 2e-2, "
      f"{2e-2 / dg.abs().max().item():.0f}x LARGER")
print(f"  |dg|rms        = {dg.pow(2).mean().sqrt():.4e}")

# --- propagate dg into the gate parameters ------------------------------------------------------
g.backward(dg.float())
print("\n  resulting gate-parameter gradients (from the CORRECT dg):")
for n, p in [("A_log", A_log), ("dt_bias", dt_bias), ("f_proj.0.W", f1.weight), ("f_proj.1.W", f2.weight)]:
    print(f"    {n:12s} |grad|max={p.grad.abs().max():.4e}  |grad|rms={p.grad.pow(2).mean().sqrt():.4e}")

# --- what if dg were ENTIRELY ZERO (which the atol=2e-2 test cannot detect)? --------------------
print("\n  If the kernel returned dg == 0 (passes the acceptance test in the 'real' regime):")
print("    every gate parameter above gets EXACTLY ZERO gradient -> A_log, dt_bias and both")
print("    f_proj layers never train. The forget gate stays frozen at initialization for the")
print("    entire run, and the loss curve still descends (q/k/v/beta/out still train).")

# --- relative scale vs a same-layer parameter that IS covered ----------------------------------
w_q = torch.nn.Linear(D, H * K, bias=False)
torch.nn.init.normal_(w_q.weight, std=0.02)
xq = x.detach().clone()
qq = w_q(xq).view(B, T, H, K)
qq.backward(grads[0].float())
print(f"\n  for scale, w_q (fed by dq, which IS covered): |grad|max="
      f"{w_q.weight.grad.abs().max():.4e}")
print(f"  ratio |f_proj.1.W grad| / |w_q grad| = "
      f"{f2.weight.grad.abs().max().item() / w_q.weight.grad.abs().max().item():.3f}")

# --- how small must the gate be before dg falls under atol? ------------------------------------
print("\n" + "=" * 100)
print("  |dg|max vs the flat atol=2e-2 as a function of the decay strength")
print("=" * 100)
print(f"  {'exp(A_log) scale':>18s} {'exp(g) mean':>13s} {'|dg|max':>11s}  {'atol/|dg|':>10s}  covered?")
for a_scale in (0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    gg = -a_scale * torch.nn.functional.softplus(
        f2(f1(x)).view(B, T, H, K).detach() + dt_bias.detach().view(H, K)
    )
    d = bwd(q, k, v, gg, beta, do, R, K**-0.5, torch.float64)[3]
    m = d.abs().max().item()
    print(f"  {a_scale:>18.2f} {gg.exp().mean():>13.4f} {m:>11.3e}  {2e-2 / m:>10.1f}x  "
          f"{'yes' if m > 2e-2 else 'NO -- untestable'}")
