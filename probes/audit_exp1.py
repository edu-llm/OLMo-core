"""EXP1: how easy is the acceptance test? Gradient magnitudes vs the flat atol=2e-2,
and what regime of input space the tests actually cover."""

import sys

import torch

sys.path.insert(0, "/Users/ericwu/Developer/Capstone_LLM/probes")
from audit_numerics import bwd, fwd, make  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=True)


def gm(*ts):
    return "  ".join(f"{n}={t.abs().max().item():.3e}" for n, t in ts)


print("=" * 92)
print("EXP1a  gradient magnitudes in the ACCEPTANCE-TEST regime (fp64 ground truth)")
print("       accept test: g=-rand(), beta=rand(), q/k L2-normed, T=64, atol=2e-2 flat")
print("=" * 92)
B, T, H, K, V, R = 2, 64, 2, 64, 64, 2
scale = K**-0.5

for label, gmode, bmode in [
    ("accept  (g=-rand)", "uniform_neg", "sigmoid"),
    ("unittest(g=logsig)", "logsigmoid", "sigmoid"),
    ("REAL-init(g~-0.7..-11)", "kda_init", "sigmoid"),
    ("RETAIN  (g~-0.01)", "tiny", "sigmoid"),
    ("NO-DECAY(g=0)", "zero", "sigmoid"),
]:
    gen = torch.Generator().manual_seed(7)

    def rnd(*s):
        return torch.randn(*s, generator=gen, dtype=torch.float32)

    q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16)
    k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16)
    v = rnd(B, T * R, H, V).to(torch.bfloat16)
    beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16)
    if gmode == "uniform_neg":
        g = -torch.rand(B, T, H, K, generator=gen)
    elif gmode == "logsigmoid":
        g = torch.nn.functional.logsigmoid(rnd(B, T, H, K))
    elif gmode == "kda_init":
        A = torch.rand(H, generator=gen) * 15 + 1.0  # exp(A_log), U(1,16)
        g = -A.view(1, 1, H, 1) * torch.nn.functional.softplus(rnd(B, T, H, K) * 0.02)
    elif gmode == "tiny":
        g = -torch.nn.functional.softplus(rnd(B, T, H, K)) * 0.01
    else:
        g = torch.zeros(B, T, H, K)
    do = rnd(B, T, H, V).to(torch.bfloat16)

    o64, S64, norms = fwd(q, k, v, g, beta, R, scale, torch.float64, track=True)
    grads = bwd(q, k, v, g, beta, do, R, scale, torch.float64)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    print(f"\n {label}")
    print(f"   exp(g): mean={g.exp().mean():.4f}  min={g.exp().min():.3e}  max={g.exp().max():.4f}")
    print(f"   |o|max={o64.abs().max():.3e}   ||S_T||_F={S64.reshape(-1).norm():.4e}"
          f"   ||S||_F t=0/T/2/T-1: {norms[0].max():.3e} {norms[T//2].max():.3e} {norms[-1].max():.3e}")
    print("   " + gm(*zip(names, grads[:5])))
    print("   atol 2e-2 as a RELATIVE bound: " + "  ".join(
        f"{n}={2e-2 / max(t.abs().max().item(), 1e-30) * 100:.2f}%" for n, t in zip(names, grads[:5])
    ))
