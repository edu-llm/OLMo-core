"""Verification pass for p1_launch_bench.py. Three things the first run left open.

1. REPLICATION. The +20.3% grouped win came from one job on a shared node (14 other jobs
   running). Repeat 3 independent trials and report spread. A 20% claim that moves under
   contention is not a claim.

2. KERNEL COUNTS, measured not assumed. The first analysis assumed 20/30/40 launches from
   the module structure. Count them with the profiler so the "time tracks kernel count, not
   bytes" conclusion rests on data.

3. THE TWO ISO-BYTE CONTROLS, which are the cleanest comparisons available:
     10 MiB: grouped g=4      vs lowrank_fused r=128
     40 MiB: dense            vs lowrank_fused r=512
   Same bytes read, different factorization. Any time difference is the structural penalty
   of the factorization itself, with the bandwidth effect held constant. This is what
   isolates cause; the headline table conflates bytes and structure.
"""

from __future__ import annotations

import json
import statistics
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))
from p1_launch_bench import (  # noqa: E402
    D,
    N_LIV,
    Dense,
    Grouped,
    LowRankFused,
    LowRankSep,
    bytes_per_token,
    time_stack,
)

TRIALS = 3
ARMS = [
    ("dense", None, None, lambda: Dense()),
    ("lowrank_fused", 128, None, lambda: LowRankFused(r=128)),
    ("lowrank_fused", 512, None, lambda: LowRankFused(r=512)),
    ("lowrank_sep", 128, None, lambda: LowRankSep(r=128)),
    ("grouped", None, 2, lambda: Grouped(g=2)),
    ("grouped", None, 4, lambda: Grouped(g=4)),
]


def count_kernels(stack, x) -> int:
    """Measured CUDA kernel launches for one decode step through all layers."""
    with torch.no_grad():
        for _ in range(10):                       # warm up autotuners
            for m in stack:
                m(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for m in stack:
                m(x)
            torch.cuda.synchronize()
    return sum(int(e.count) for e in prof.key_averages()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0)


def main() -> int:
    if not torch.cuda.is_available():
        print("needs CUDA", file=sys.stderr)
        return 1
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}   trials={TRIALS}\n")
    x = torch.randn(1, D, device=dev, dtype=torch.bfloat16)

    out = {}
    for arm, r, g, ctor in ARMS:
        tag = arm + (f" r={r}" if r else "") + (f" g={g}" if g else "")
        stack = [ctor().to(dev).to(torch.bfloat16).eval() for _ in range(N_LIV)]
        with torch.no_grad():
            trials = [time_stack(stack, x, graphed=True) for _ in range(TRIALS)]
        nk = count_kernels(stack, x)
        mb = bytes_per_token(arm, r, g) / 2**20
        out[tag] = {"graphed_us": trials, "median": statistics.median(trials),
                    "spread_pct": (max(trials) - min(trials)) / statistics.median(trials) * 100,
                    "kernels": nk, "mib": mb}
        print(f"  {tag:<22} {statistics.median(trials):7.1f}us  "
              f"spread {out[tag]['spread_pct']:4.1f}%  kernels {nk:>3}  {mb:5.2f} MiB  "
              f"{mb/statistics.median(trials)*1e6/1024:6.0f} GB/s")
        del stack
        torch.cuda.empty_cache()

    print("\n=== ISO-BYTE CONTROLS (bytes held constant; difference = factorization cost) ===")
    for bytes_mib, a, b in ((10.0, "grouped g=4", "lowrank_fused r=128"),
                            (40.0, "dense", "lowrank_fused r=512")):
        ta, tb = out[a]["median"], out[b]["median"]
        assert abs(out[a]["mib"] - bytes_mib) < 0.01 and abs(out[b]["mib"] - bytes_mib) < 0.01, \
            f"iso-byte assumption broken for {bytes_mib} MiB"
        print(f"  at {bytes_mib:5.1f} MiB:  {a:<20} {ta:6.1f}us   vs  {b:<20} {tb:6.1f}us"
              f"   -> {a} is {(tb-ta)/tb*100:+.1f}% faster")

    d = out["dense"]
    print("\n=== VERDICT ===")
    for tag, v in out.items():
        if tag == "dense":
            continue
        delta = (d["median"] - v["median"]) / d["median"] * 100
        print(f"  {tag:<22} {delta:+6.1f}% vs dense   "
              f"{'FASTER' if delta > 0 else 'SLOWER'}")
    with open("p1_verify_results.json", "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "arms": out}, f, indent=2)
    print("\nwrote p1_verify_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
