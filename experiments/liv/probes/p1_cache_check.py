"""Independent test of the claim that the P1 latency result was an L2-cache artifact.

THE CLAIM (from a reassessment subagent): p1_launch_bench.py holds <=40 MiB of gate weights and
replays a CUDA graph with nothing evicting them, so every timed replay was L2-resident on a card
whose L2 is ~96 MiB. Real decode reads far more than L2 per token and is genuinely HBM-bound.
Cache residency is the regime where byte savings buy least -- so, the claim goes, the benchmark
was constructed such that P1's only benefit could not appear, and at a realistic working set the
factorized arms flip from losing to winning.

WHY THIS NEEDS INDEPENDENT VERIFICATION: it reverses a frozen decision ("P1's latency claim is
dead", recorded in HANDOFF.md and the design doc) on the strength of two jobs that ran 7s and 16s
and left no output files on disk. If true it is the most consequential finding in the project. If
false, acting on it re-animates a dead claim. Either way I should measure it myself.

METHOD: the honest version of the comparison. Instead of one stack of 10 layers (40 MiB), hold N
independent copies of the gate stack and cycle through them, so consecutive timed iterations touch
different weights and the aggregate working set exceeds L2. Kernel count per timed step is held
IDENTICAL across working-set sizes -- only residency changes. That isolates cache effects from
launch effects, which is exactly what the original benchmark could not do.

Reports achieved bandwidth so residency is visible directly: an in-cache run reports far above
HBM peak, an HBM-bound run reports at or below it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
import torch.nn as nn

D = 1024
N_LIV = 10
GATES = 2
BYTES_PER_PARAM = 2
WARMUP, ITERS = 20, 100


class Dense(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.b = nn.Linear(d, d, bias=False)
        self.c = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.b(x), self.c(x)


class LowRankFused(nn.Module):
    """One shared d->2r down-projection, two r->d up-projections."""

    def __init__(self, d=D, r=128):
        super().__init__()
        self.down = nn.Linear(d, 2 * r, bias=False)
        self.bu, self.cu = nn.Linear(r, d, bias=False), nn.Linear(r, d, bias=False)
        self.r = r

    def forward(self, x):
        h = self.down(x)
        return self.bu(h[..., : self.r]), self.cu(h[..., self.r :])


class Grouped(nn.Module):
    def __init__(self, d=D, g=4):
        super().__init__()
        self.g, self.bs = g, d // g
        self.wb = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs**0.5)
        self.wc = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs**0.5)

    def forward(self, x):
        v = x.reshape(-1, self.g, self.bs).transpose(0, 1)
        return (
            torch.bmm(v, self.wb).transpose(0, 1).reshape(x.shape),
            torch.bmm(v, self.wc).transpose(0, 1).reshape(x.shape),
        )


ARMS = {
    "dense": lambda: Dense(),
    "lowrank_fused_r128": lambda: LowRankFused(r=128),
    "grouped_g4": lambda: Grouped(g=4),
}


def build_replicas(ctor, n_replicas: int, dev) -> list:
    """n_replicas independent stacks of N_LIV layers each."""
    return [
        nn.ModuleList([ctor().to(dev).to(torch.bfloat16).eval() for _ in range(N_LIV)])
        for _ in range(n_replicas)
    ]


def time_cycling(replicas: list, x, graphed: bool) -> float:
    """
    Median us for ONE stack traversal, while cycling across replicas.

    Kernel count per timed step is one stack's worth regardless of n_replicas -- only the
    aggregate working set changes. CUDA graphs cannot be used across a Python-level cycle, so
    when graphed=True we capture one graph per replica and replay them round-robin.
    """
    def step(i):
        for m in replicas[i]:
            m(x)

    n = len(replicas)
    for i in range(WARMUP):
        step(i % n)
    torch.cuda.synchronize()

    if graphed:
        graphs = []
        for i in range(n):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    step(i)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step(i)
            graphs.append(g)
        torch.cuda.synchronize()
        run = lambda i: graphs[i].replay()  # noqa: E731
    else:
        run = step

    samples = []
    for it in range(ITERS):
        i = it % n
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        run(i)
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p1_cache_check.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs CUDA -- run on FarmShare via sbatch", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 0)
    print(f"device: {props.name}")
    print(f"  L2 cache: {l2 / 2**20:.1f} MiB   (claim under test: ~96 MiB)")
    print(f"  SMs: {props.multi_processor_count}, total mem {props.total_memory / 2**30:.1f} GiB")
    print(f"\ngeometry d={D}, {N_LIV} layers x {GATES} gates, bf16, batch=1\n")

    x = torch.randn(1, D, device=dev, dtype=torch.bfloat16)
    per_stack = {
        "dense": 2 * D * D * N_LIV * BYTES_PER_PARAM,
        "lowrank_fused_r128": (D * 256 + 2 * 128 * D) * N_LIV * BYTES_PER_PARAM,
        "grouped_g4": 2 * (D * D // 4) * N_LIV * BYTES_PER_PARAM,
    }

    rows = []
    # Replica counts chosen so dense spans from well inside L2 to well past it.
    for n_rep in (1, 4, 24):
        ws = per_stack["dense"] * n_rep / 2**20
        print(f"--- {n_rep} replica(s): dense working set {ws:.0f} MiB "
              f"({'IN' if ws < l2 / 2**20 else 'PAST'} L2) ---")
        base = None
        for name, ctor in ARMS.items():
            reps = build_replicas(ctor, n_rep, dev)
            with torch.no_grad():
                us = time_cycling(reps, x, graphed=True)
            mib = per_stack[name] / 2**20
            gbs = per_stack[name] / (us * 1e-6) / 1e9
            if name == "dense":
                base = us
            delta = (base - us) / base * 100
            rows.append({
                "n_replicas": n_rep, "arm": name, "us": us,
                "stack_mib": mib, "achieved_gbs": gbs, "vs_dense_pct": delta,
                "working_set_mib": per_stack[name] * n_rep / 2**20,
            })
            tag = "" if name == "dense" else f"  {delta:+6.2f}% vs dense"
            print(f"  {name:<20} {us:8.1f}us  {mib:6.2f} MiB/stack  "
                  f"{gbs:6.0f} GB/s{tag}")
            del reps
            torch.cuda.empty_cache()
        print()

    print("=== VERDICT ===")
    for n_rep in (1, 4, 24):
        lr = next(r for r in rows if r["n_replicas"] == n_rep and r["arm"].startswith("lowrank"))
        ws = next(r for r in rows if r["n_replicas"] == n_rep and r["arm"] == "dense")
        print(f"  dense working set {ws['working_set_mib']:>6.0f} MiB -> "
              f"lowrank_fused_r128 {lr['vs_dense_pct']:+6.2f}% vs dense")
    small = next(r for r in rows if r["n_replicas"] == 1 and r["arm"].startswith("lowrank"))
    big = next(r for r in rows if r["n_replicas"] == 24 and r["arm"].startswith("lowrank"))
    print()
    if small["vs_dense_pct"] < 0 < big["vs_dense_pct"]:
        print("  CLAIM SUPPORTED: low-rank loses in-cache and WINS past L2. The original")
        print("  benchmark's working set was unrepresentative and its sign is not trustworthy.")
    elif big["vs_dense_pct"] < 0:
        print("  CLAIM NOT SUPPORTED: low-rank still loses past L2. The original result stands;")
        print("  the cache-artifact explanation does not rescue P1's latency claim.")
    else:
        print("  MIXED: see the table. Do not summarize this as a clean flip either way.")

    with open(args.out, "w") as f:
        json.dump({"device": props.name, "l2_bytes": l2, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
