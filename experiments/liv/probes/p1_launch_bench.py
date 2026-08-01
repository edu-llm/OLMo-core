"""P1 gate-factorization latency microbenchmark. THE pre-training gate for P1.

Run on FarmShare (NVIDIA L40S, sm_89). This benchmark decides whether P1's latency claim is
real at the frozen 350M geometry. It can kill the claim for the price of one 25-minute job.

WHY IT IS NOT OPTIONAL, AND WHY THE DETAILS MATTER
  Factorizing a gate d->d into d->r->d reduces weight bytes read per decode step from d^2 to
  2dr, but replaces one GEMM with two -- so it adds one kernel launch per gate. At the frozen
  geometry (d=1024, 10 LIV layers, 2 gates each) that is 20 extra launches per decoded token.

  Analytic breakeven, recomputed for THIS geometry and THIS card:

    card          r=128 saving   breakeven/launch
    A100 (0.75x)     27.0 us         1.35 us
    L40S (0.75x)     48.5 us         2.43 us

  against a typical un-graphed launch cost of 5-10 us. So WITHOUT CUDA graphs P1 measures
  ~4% SLOWER and the benchmark reports the WRONG SIGN.

  NOTE the design doc's widely-quoted "4.72 us breakeven" was computed at d=2048 (the 1.2B
  geometry). At d=1024 the byte saving is 4x smaller, so breakeven falls to ~1.35 us (A100).
  This is the single most important correction in this file.

  Two mitigations decide the outcome, so both are measured as first-class arms:
    (a) FUSED gates: compute B and C in ONE d->2r projection. Halves extra launches 20->10,
        which doubles the breakeven. Free win, no modelling change.
    (b) CUDA GRAPHS: collapses launch cost to ~0.5-1.5 us.
  Analytically, fused + graphed WINS at d=1024; separate + un-graphed LOSES. Confirm on metal.

ARMS
  dense            2 x (d->d)                      -- the control (stock LIV)
  lowrank_sep      2 x (d->r->d)                   -- naive P1
  lowrank_fused    1 x (d->2r) then 2 x (r->d)     -- P1 + mitigation (a)
  grouped          2 x block-diagonal(g blocks)    -- STRONGEST SYSTEMS COMPETITOR. In STAR's
                   search space and NOT selected for the gated-conv featurizer; grouped
                   matmuls are better supported than skinny ones. If grouped beats lowrank,
                   P1's systems story is gone regardless of quality results.

Each arm runs BOTH un-graphed and CUDA-graphed. Reports the sign explicitly so the result
cannot be misread.

  srun -p gpu --gres=gpu:1 -c 8 --mem=48G -t 00:25:00 \
      ../venv/bin/python p1_launch_bench.py
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
RANKS = (128, 256, 512)
GROUPS = (2, 4)
WARMUP, ITERS = 50, 300


class Dense(nn.Module):
    """Stock LIV: two independent full-width gate projections."""

    def __init__(self, d=D):
        super().__init__()
        self.b = nn.Linear(d, d, bias=False)
        self.c = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.b(x), self.c(x)


class LowRankSep(nn.Module):
    """Naive P1: each gate factorized independently -> 2 extra launches per layer."""

    def __init__(self, d=D, r=128):
        super().__init__()
        self.bd, self.bu = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        self.cd, self.cu = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)

    def forward(self, x):
        return self.bu(self.bd(x)), self.cu(self.cd(x))


class LowRankFused(nn.Module):
    """P1 + mitigation (a): ONE d->2r down-projection shared by both gates.

    Same parameter count and same weight bytes as LowRankSep, but one fewer launch per
    layer, which doubles the analytic breakeven. Strictly dominates the separate version.
    """

    def __init__(self, d=D, r=128):
        super().__init__()
        self.down = nn.Linear(d, 2 * r, bias=False)
        self.bu, self.cu = nn.Linear(r, d, bias=False), nn.Linear(r, d, bias=False)
        self.r = r

    def forward(self, x):
        h = self.down(x)
        return self.bu(h[..., :self.r]), self.cu(h[..., self.r:])


class Grouped(nn.Module):
    """Block-diagonal gates via bmm. Params = d^2/g, and grouped matmuls map well to
    tensor cores -- unlike skinny r=128 GEMMs, which FLAR-SVD measured as SLOWER than
    baseline on mobile despite 2x fewer params."""

    def __init__(self, d=D, g=2):
        super().__init__()
        self.g, self.bs = g, d // g
        self.wb = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs ** 0.5)
        self.wc = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs ** 0.5)

    def forward(self, x):
        v = x.reshape(-1, self.g, self.bs).transpose(0, 1)      # (g, B, bs)
        return (torch.bmm(v, self.wb).transpose(0, 1).reshape(x.shape),
                torch.bmm(v, self.wc).transpose(0, 1).reshape(x.shape))


def bytes_per_token(arm: str, r: int | None, g: int | None) -> int:
    """bf16 weight bytes read per decode token across all LIV gate projections.

    Inner dict is PARAMS per layer; multiplied by N_LIV and 2 (bf16) at the end.
    Note lowrank_fused has the SAME param count as lowrank_sep -- one d->2r down-projection
    (d*2r) plus two r->d up-projections (2*r*d) equals two independent d->r->d chains. So
    fused strictly dominates: identical bytes, one fewer launch per layer.
    """
    per_layer = {
        "dense": 2 * D * D,
        "lowrank_sep": 2 * (2 * D * r) if r else 0,
        "lowrank_fused": (D * 2 * r) + (2 * r * D) if r else 0,
        "grouped": 2 * (D * D // g) if g else 0,
    }[arm]
    return per_layer * N_LIV * 2


def time_stack(stack, x, graphed: bool) -> float:
    """Median us for one decode step through all N_LIV layers' gates."""
    def step():
        for m in stack:
            m(x)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()

    if graphed:
        # Capture on a side stream, per CUDA graph requirements.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            step()
        torch.cuda.synchronize()
        run = gr.replay
    else:
        run = step

    samples = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        run()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) * 1000.0)   # ms -> us
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p1_bench_results.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: needs CUDA. This benchmark is meaningless without CUDA graphs, and "
              "graphs need a real device. Run on FarmShare: "
              "srun -p gpu --gres=gpu:1 -c 8 --mem=48G -t 00:25:00", file=sys.stderr)
        return 1

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}")
    print(f"geometry: d={D}, {N_LIV} LIV layers x {GATES} gates, bf16, batch=1 (decode)\n")

    x = torch.randn(1, D, device=dev, dtype=torch.bfloat16)

    arms: list[tuple[str, int | None, int | None, callable]] = [
        ("dense", None, None, lambda: Dense()),
    ]
    for r in RANKS:
        arms.append((f"lowrank_sep", r, None, lambda r=r: LowRankSep(r=r)))
        arms.append((f"lowrank_fused", r, None, lambda r=r: LowRankFused(r=r)))
    for g in GROUPS:
        arms.append((f"grouped", None, g, lambda g=g: Grouped(g=g)))

    results = []
    for arm, r, g, ctor in arms:
        stack = [ctor().to(dev).to(torch.bfloat16).eval() for _ in range(N_LIV)]
        with torch.no_grad():
            row = {
                "arm": arm, "r": r, "g": g,
                "bytes_per_token": bytes_per_token(arm, r, g),
                "eager_us": time_stack(stack, x, graphed=False),
                "graph_us": time_stack(stack, x, graphed=True),
            }
        results.append(row)
        tag = f"{arm}" + (f" r={r}" if r else "") + (f" g={g}" if g else "")
        print(f"  {tag:<22} eager {row['eager_us']:7.1f}us   graphed {row['graph_us']:7.1f}us   "
              f"{row['bytes_per_token']/2**20:5.2f} MiB/tok")
        del stack
        torch.cuda.empty_cache()

    base = next(r_ for r_ in results if r_["arm"] == "dense")
    print(f"\n{'arm':<22} {'eager':>16} {'graphed':>16}   verdict(graphed)")
    for row in results:
        if row is base:
            continue
        de = base["eager_us"] - row["eager_us"]
        dg = base["graph_us"] - row["graph_us"]
        row["speedup_graphed_pct"] = dg / base["graph_us"] * 100
        tag = f"{row['arm']}" + (f" r={row['r']}" if row['r'] else "") + \
              (f" g={row['g']}" if row['g'] else "")
        print(f"{tag:<22} {de:+8.1f}us({de/base['eager_us']*100:+5.1f}%) "
              f"{dg:+8.1f}us({dg/base['graph_us']*100:+5.1f}%)   "
              f"{'FASTER' if dg > 0 else 'SLOWER than stock LIV'}")

    winners = [r_ for r_ in results
               if r_ is not base and r_.get("speedup_graphed_pct", 0) > 0]
    print("\n=== GATE DECISION ===")
    if not winners:
        print("NO arm beats dense even WITH CUDA graphs -> DROP P1's latency claim now.")
        print("P1 may still be defensible as a PARAMETER-EFFICIENCY claim, but not a speed one.")
    else:
        best = max(winners, key=lambda r_: r_["speedup_graphed_pct"])
        print(f"best: {best['arm']} r={best['r']} g={best['g']} "
              f"{best['speedup_graphed_pct']:+.2f}% graphed")
        print("Report GRAPHED numbers as headline; eager numbers as the cautionary footnote.")
        if best["arm"] == "grouped":
            print("WARNING: grouped wins. It is in STAR's search space and was NOT selected, "
                  "so P1's novelty argument weakens -- reconsider before training.")

    with open(args.out, "w") as f:
        json.dump({"device": name, "d": D, "n_liv": N_LIV, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
