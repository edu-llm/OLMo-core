"""Re-measure the KDA-Householder perf claims with proper warmup + sync + repeats.

Also benchmarks the FAIR baselines: fla.ops.kda.chunk_kda (R=1, production chunked kernel)
and fla.ops.gated_delta_product.chunk_gated_delta_product (R>1, per-HEAD gate).
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ["OLMO_SRC"])
from olmo_core.nn.attention.kda_householder import chunk_kda_householder

DEV = "cuda"
ROWS = []


def mk(B, T, H, K, V, R, seed=0, per_head_gate=False):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    def rnd(*s):
        return torch.randn(*s, generator=gen, device=DEV, dtype=torch.float32)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_()
    v = rnd(B, T * R, H, V).to(torch.bfloat16).requires_grad_()
    beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16).requires_grad_()
    if per_head_gate:
        g = (-F.softplus(rnd(B, T, H) * 0.02)).requires_grad_()
    else:
        g = (-F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_()
    do = rnd(B, T, H, V).to(torch.bfloat16)
    return q, k, v, g, beta, do


def timeit(fn, warmup, iters, label):
    """fwd+bwd timing with warmup, sync, and peak memory measured on the timed region."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"  {label:58s} {dt*1000:11.2f} ms  peak {mem:6.3f} GiB  (warmup={warmup} iters={iters})", flush=True)
        return dt * 1000, mem, "ok"
    except Exception as e:
        print(f"  {label:58s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None, None, f"{type(e).__name__}: {str(e)[:90]}"
    finally:
        torch.cuda.empty_cache()


def bench_ours(B, T, H, K, V, R, backend, warmup, iters):
    q, k, v, g, beta, do = mk(B, T, H, K, V, R)
    leaves = [q, k, v, g, beta]
    def step():
        o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R, backend=backend)
        torch.autograd.grad(o, leaves, grad_outputs=do)
    return timeit(step, warmup, iters, f"ours[{backend}] B{B} T{T} H{H} K{K} V{V} R{R}")


def bench_fla_kda(B, T, H, K, V, warmup, iters):
    from fla.ops.kda import chunk_kda
    q, k, v, g, beta, do = mk(B, T, H, K, V, 1)
    leaves = [q, k, v, g, beta]
    def step():
        o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
                         use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False)
        torch.autograd.grad(o, leaves, grad_outputs=do)
    return timeit(step, warmup, iters, f"fla.chunk_kda B{B} T{T} H{H} K{K} V{V} R1")


def bench_fla_gdp(B, T, H, K, V, R, warmup, iters):
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    q, k, v, g, beta, do = mk(B, T, H, K, V, R, per_head_gate=True)
    leaves = [q, k, v, g, beta]
    def step():
        o, _ = chunk_gated_delta_product(q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
                                         num_householder=R, use_qk_l2norm_in_kernel=False)
        torch.autograd.grad(o, leaves, grad_outputs=do)
    return timeit(step, warmup, iters, f"fla.chunk_gated_delta_product(PER-HEAD g) B{B} T{T} H{H} K{K} V{V} R{R}")


p = torch.cuda.get_device_properties(0)
print(f"GPU {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count} mem={p.total_memory/2**30:.1f} GiB")
print(f"torch {torch.__version__}  triton {__import__('triton').__version__}", flush=True)
import fla
print(f"fla {getattr(fla,'__version__','?')}\n", flush=True)

STACK = os.environ.get("STACK_LABEL", "?")


def rec(cfg, impl, ms, mem, status, note):
    ROWS.append(dict(stack=STACK, cfg=cfg, impl=impl, ms=ms, gib=mem, status=status, note=note))


# ---- C: the two claimed configs. bench_bwd.py used H=4 K=64 V=64. ----
print("=== CLAIMED CONFIGS (H=4 K=64 V=64, matching probes/bench_bwd.py) ===", flush=True)
for (B, T, R) in [(4, 2048, 4), (2, 8192, 4)]:
    cfg = f"B{B} T{T} H4 K64 V64 R{R}"
    ms, mem, st = bench_ours(B, T, 4, 64, 64, R, "triton", 2, 5)
    rec(cfg, "ours-triton", ms, mem, st, "fused-recurrent Triton, fwd+bwd")
    n_it = 2 if T >= 8192 else 3
    ms2, mem2, st2 = bench_ours(B, T, 4, 64, 64, R, "torch", 1, n_it)
    rec(cfg, "ours-torch-ref", ms2, mem2, st2, "OUR OWN naive per-timestep Python loop reference")

# ---- C: the FAIR baseline. R=1, same shapes, ours vs fla's production chunked KDA ----
print("\n=== FAIR BASELINE: R=1, ours vs fla.ops.kda.chunk_kda (production chunked) ===", flush=True)
for (B, T, H, K, V) in [(4, 2048, 4, 64, 64), (2, 8192, 4, 64, 64), (2, 2048, 8, 64, 64)]:
    cfg = f"B{B} T{T} H{H} K{K} V{V} R1"
    ms, mem, st = bench_ours(B, T, H, K, V, 1, "triton", 2, 5)
    rec(cfg, "ours-triton", ms, mem, st, "fused-recurrent Triton (ours)")
    ms2, mem2, st2 = bench_fla_kda(B, T, H, K, V, 2, 5)
    rec(cfg, "fla.chunk_kda", ms2, mem2, st2, "fla production CHUNKED KDA kernel, per-channel gate, R=1 only")

# ---- C: closest R>1 production kernel (different op: per-HEAD gate) ----
print("\n=== NEAREST R>1 PRODUCTION KERNEL (different op: per-HEAD gate) ===", flush=True)
for (B, T, H, K, V, R) in [(4, 2048, 4, 64, 64, 4), (2, 8192, 4, 64, 64, 4)]:
    cfg = f"B{B} T{T} H{H} K{K} V{V} R{R}"
    ms2, mem2, st2 = bench_fla_gdp(B, T, H, K, V, R, 2, 5)
    rec(cfg, "fla.chunk_gated_delta_product", ms2, mem2, st2,
        "DIFFERENT OP: per-HEAD scalar gate (not per-channel). Bound only.")

# ---- E: occupancy / batch scaling ----
print("\n=== OCCUPANCY: batch scaling at the probe shape (grid = cdiv(V,BV)*B*H) ===", flush=True)
BV = min(8, 64)
for B in (1, 2, 4, 8, 16, 32):
    H = 8
    progs = (64 + BV - 1) // BV * B * H
    ms, mem, st = bench_ours(B, 512, H, 64, 64, 2, "triton", 2, 5)
    rec(f"B{B} T512 H{H} K64 V64 R2", "ours-triton", ms, mem, st,
        f"occupancy scan: grid={progs} programs on {p.multi_processor_count} SMs")

# ---- E: hs workspace empirical check ----
print("\n=== hs WORKSPACE: measured peak vs O(B*T*H*K*V)*4 formula ===", flush=True)
for (B, T, H, K, V, R) in [(4, 4096, 16, 64, 64, 2), (2, 2048, 8, 64, 64, 4), (1, 1024, 8, 128, 128, 2)]:
    hs = B * T * H * K * V * 4 / 2**30
    ms, mem, st = bench_ours(B, T, H, K, V, R, "triton", 1, 2)
    print(f"      -> hs formula = {hs:.3f} GiB ; measured peak = {mem if mem else -1:.3f} GiB", flush=True)
    rec(f"B{B} T{T} H{H} K{K} V{V} R{R}", "ours-triton", ms, mem, st,
        f"hs formula B*T*H*K*V*4 = {hs:.3f} GiB")

with open(os.environ["OUT_JSON"], "w") as f:
    json.dump(ROWS, f, indent=1)
print("\nWROTE", os.environ["OUT_JSON"], flush=True)
