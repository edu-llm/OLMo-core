"""Is P1 slower because factorizing is bad, or because the benchmark fit in L2 cache?

L40S L2 = 96 MiB (MEASURED, job 1671407). The original probe (p1_launch_bench.py) holds at
most 40 MiB of gate weights and replays a CUDA graph 300x with nothing evicting them, so
every timed replay is L2-resident. Real batch-1 decode of the 350M model reads ~709 MB of
weights per token -- 7.4x the L2 -- so it is genuinely HBM-bound.

A cache-resident test is the single regime in which byte savings buy the LEAST time, which
biases against P1 specifically.

METHOD. The first attempt used an explicit L2 flush (zero a 192 MiB buffer between replays).
That was wrong: it left dirty lines whose writeback bled into the timed region and added a
near-uniform +26 us to every arm, changing nothing. Instead, simply scale the number of
stacked layers so the working set genuinely exceeds L2 -- which is also what the real model
does. All arms use the SAME layer count at each rung, so kernel-count ratios stay 20/30/40
exactly as in the original probe; only the residency regime changes.

If P1's -8.2% was a cache artifact, the sign should move as the working set crosses 96 MiB.
"""
import json
import statistics

import torch
import torch.nn as nn

D = 1024
TARGET_MIB = [40, 320, 960]   # 40 = original (L2-resident); 960 ~= real 709 MB decode
WARMUP, ITERS, TRIALS = 20, 60, 3


class Dense(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.b = nn.Linear(d, d, bias=False)
        self.c = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.b(x), self.c(x)


class LowRankFused(nn.Module):
    def __init__(self, d=D, r=128):
        super().__init__()
        self.down = nn.Linear(d, 2 * r, bias=False)
        self.bu = nn.Linear(r, d, bias=False)
        self.cu = nn.Linear(r, d, bias=False)
        self.r = r

    def forward(self, x):
        h = self.down(x)
        return self.bu(h[..., :self.r]), self.cu(h[..., self.r:])


class LowRankSep(nn.Module):
    def __init__(self, d=D, r=128):
        super().__init__()
        self.bd = nn.Linear(d, r, bias=False)
        self.bu = nn.Linear(r, d, bias=False)
        self.cd = nn.Linear(d, r, bias=False)
        self.cu = nn.Linear(r, d, bias=False)

    def forward(self, x):
        return self.bu(self.bd(x)), self.cu(self.cd(x))


class Grouped(nn.Module):
    def __init__(self, d=D, g=4):
        super().__init__()
        self.g, self.bs = g, d // g
        self.wb = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs ** 0.5)
        self.wc = nn.Parameter(torch.randn(g, self.bs, self.bs) / self.bs ** 0.5)

    def forward(self, x):
        v = x.reshape(-1, self.g, self.bs).transpose(0, 1)
        return (torch.bmm(v, self.wb).transpose(0, 1).reshape(x.shape),
                torch.bmm(v, self.wc).transpose(0, 1).reshape(x.shape))


# (tag, ctor, weight bytes per layer instance, bf16)
ARMS = [
    ("dense", Dense, 2 * D * D * 2),
    ("lowrank_fused r=128", lambda: LowRankFused(r=128), 4 * D * 128 * 2),
    ("lowrank_sep r=128", lambda: LowRankSep(r=128), 4 * D * 128 * 2),
    ("grouped g=4", lambda: Grouped(g=4), 2 * (D * D // 4) * 2),
]


def timeit(stack, x):
    def step():
        for m in stack:
            m(x)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
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

    out = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        gr.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) * 1000.0)
    return statistics.median(out)


def main():
    p = torch.cuda.get_device_properties(0)
    print("%s  L2=%.0f MiB\n" % (p.name, p.L2_cache_size / 2 ** 20))
    x = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
    res = {}

    for target in TARGET_MIB:
        regime = "L2-RESIDENT" if target < 96 else "EXCEEDS L2"
        print("--- working set target %d MiB (%s) ---" % (target, regime))
        # dense sets the layer count; every arm uses the same count so the
        # kernel-count ratio stays 20/30/40 as in the original probe.
        n = max(1, int(target * 2 ** 20 // (2 * D * D * 2)))
        row = {}
        for tag, ctor, bpl in ARMS:
            stack = [ctor().cuda().to(torch.bfloat16).eval() for _ in range(n)]
            mib = n * bpl / 2 ** 20
            with torch.no_grad():
                t = statistics.median([timeit(stack, x) for _ in range(TRIALS)])
            row[tag] = {"us": t, "mib": mib, "layers": n, "GBs": mib * 2 ** 20 / t / 1e3}
            print("  %-22s n=%-4d %7.1f MiB  %8.2fus  %7.1f GB/s"
                  % (tag, n, mib, t, mib * 2 ** 20 / t / 1e3))
            del stack
            torch.cuda.empty_cache()

        d = row["dense"]["us"]
        for tag, v in row.items():
            if tag == "dense":
                continue
            v["vs_dense_pct"] = (d - v["us"]) / d * 100
            print("      -> %-20s %+7.2f%% vs dense" % (tag, v["vs_dense_pct"]))
        res[str(target)] = row
        print()

    with open("/scratch/users/ericrcwu/liv/p1_scaled_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote p1_scaled_results.json")


main()
