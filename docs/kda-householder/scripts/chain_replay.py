"""Re-run every level of the claimed 6-level verification chain and report the ACTUAL metric.

Reports metric name, tolerance, backend under test, and the measured number for each level.
"""
import json, os, subprocess, sys
import torch
import torch.nn.functional as F

PROBES = os.environ["KDA_PROBES_DIR"]
SRC = os.environ["OLMO_SRC"]
sys.path.insert(0, PROBES)
sys.path.insert(0, SRC)
OUT = {}


def run_probe(name, args=()):
    """Run a probe as a subprocess; return (rc, stdout+stderr)."""
    r = subprocess.run([sys.executable, os.path.join(PROBES, name), *args],
                       capture_output=True, text=True, timeout=3000,
                       env={**os.environ, "PYTHONPATH": f"{SRC}:{PROBES}"})
    return r.returncode, (r.stdout + r.stderr)


# ---------- LEVEL 1: naive oracle vs fla KDA at R=1 ----------
print("=" * 100); print("LEVEL 1: probes/naive_kda_householder.py vs fla's KDA at R=1"); print("=" * 100, flush=True)
rc, out = run_probe("naive_kda_householder.py")
print(out[-4000:], flush=True)
OUT["level1"] = dict(rc=rc, tail=out[-2500:])

# ---------- LEVEL 2: fp64 gradcheck on the TORCH backend ----------
print("=" * 100); print("LEVEL 2: kda_householder_torch.py fp64 gradcheck (TORCH backend)"); print("=" * 100, flush=True)
from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

def mk64(B, T, H, K, V, R, seed=0, rg=True):
    gen = torch.Generator().manual_seed(seed)
    def rnd(*s): return torch.randn(*s, generator=gen, dtype=torch.float64)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    g = F.logsigmoid(rnd(B, T, H, K))
    ts = (q, k, v, g, beta)
    return tuple(t.detach().clone().requires_grad_(rg) for t in ts) if rg else ts

l2 = {}
for R in (1, 2, 3):
    B, T, H, K, V = 1, 4, 2, 4, 4
    inp = mk64(B, T, H, K, V, R, seed=R)
    fn = lambda *a, R=R: kda_householder_torch(*a, num_householder=R)[0]
    try:
        ok = torch.autograd.gradcheck(fn, inp, eps=1e-6, atol=1e-8, rtol=1e-5)
        l2[f"gradcheck_R{R}"] = f"PASS={ok}"
    except Exception as e:
        l2[f"gradcheck_R{R}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"  gradcheck (1st order) R={R}: {l2[f'gradcheck_R{R}']}", flush=True)

# varlen gradcheck
try:
    B, T, H, K, V, R = 1, 6, 2, 4, 4, 2
    inp = mk64(B, T, H, K, V, R, seed=99)
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)
    fn = lambda *a: kda_householder_torch(*a, num_householder=R, cu_seqlens=cu)[0]
    ok = torch.autograd.gradcheck(fn, inp, eps=1e-6, atol=1e-8, rtol=1e-5)
    l2["gradcheck_varlen"] = f"PASS={ok}"
except Exception as e:
    l2["gradcheck_varlen"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
print(f"  gradcheck varlen: {l2['gradcheck_varlen']}", flush=True)

# 2nd order: gradgradcheck on the TORCH backend
for R in (1, 2):
    try:
        B, T, H, K, V = 1, 3, 1, 3, 3
        inp = mk64(B, T, H, K, V, R, seed=R + 50)
        fn = lambda *a, R=R: kda_householder_torch(*a, num_householder=R)[0]
        ok = torch.autograd.gradgradcheck(fn, inp, eps=1e-6, atol=1e-6, rtol=1e-4)
        l2[f"gradgradcheck_R{R}"] = f"PASS={ok}"
    except Exception as e:
        l2[f"gradgradcheck_R{R}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"  gradgradcheck (2nd order, torch backend) R={R}: {l2[f'gradgradcheck_R{R}']}", flush=True)

# 2nd order on the TRITON backend -> must RAISE (once_differentiable)
if torch.cuda.is_available():
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder
    B, T, H, K, V, R = 1, 8, 2, 64, 64, 1
    dev = "cuda"; gen = torch.Generator(device=dev).manual_seed(11)
    def rnd(*s): return torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)
    g = (-F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_()
    beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16).requires_grad_()
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    v = rnd(B, T * R, H, V).to(torch.bfloat16).requires_grad_()
    o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R, backend="triton")
    (gq,) = torch.autograd.grad(o.sum(), [q], create_graph=True)
    try:
        torch.autograd.grad(gq.sum(), [q])
        l2["triton_2nd_order"] = "NO RAISE (would silently give zeros) <-- BAD"
    except RuntimeError as e:
        l2["triton_2nd_order"] = f"RAISES RuntimeError (correct): {str(e)[:120]}"
    print(f"  TRITON backend 2nd order: {l2['triton_2nd_order']}", flush=True)
OUT["level2"] = l2

# ---------- LEVEL 3 ----------
print("=" * 100); print("LEVEL 3: probes/manual_backward_check.py vs level-2 autograd"); print("=" * 100, flush=True)
rc, out = run_probe("manual_backward_check.py")
print(out[-3000:], flush=True)
OUT["level3"] = dict(rc=rc, tail=out[-2500:])

# ---------- LEVEL 4 ----------
print("=" * 100); print("LEVEL 4: probes/bwd_emulator.py vs level 3"); print("=" * 100, flush=True)
rc, out = run_probe("bwd_emulator.py")
print(out[-4000:], flush=True)
OUT["level4"] = dict(rc=rc, tail=out[-3000:])

# ---------- LEVEL 6 ----------
print("=" * 100); print("LEVEL 6: probes/gpu_bwd_accept.py vs level 2 on GPU"); print("=" * 100, flush=True)
rc, out = run_probe("gpu_bwd_accept.py")
print(out[-3000:], flush=True)
OUT["level6"] = dict(rc=rc, tail=out[-2500:])

# ---------- audit probes ----------
for nm in ("audit_exp2_mutation.py", "audit_exp6_rewalk.py", "audit_exp7_mem.py", "audit_exp14_determinism.py"):
    print("=" * 100); print(f"AUDIT PROBE: {nm}"); print("=" * 100, flush=True)
    rc, out = run_probe(nm)
    print(out[-6000:], flush=True)
    OUT[nm] = dict(rc=rc, tail=out[-4000:])

with open(os.environ["OUT_JSON"], "w") as f:
    json.dump(OUT, f, indent=1)
print("\nWROTE", os.environ["OUT_JSON"], flush=True)
