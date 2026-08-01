"""Does the P1 microbenchmark measure HBM bandwidth, or L2 hits?

L40S L2 = 96 MiB (measured). The original benchmark holds at most 40 MiB of weights and
replays a CUDA graph 300x with nothing evicting them -> every timed replay is L2-resident.
Real batch-1 decode reads ~709 MB/token, so it is genuinely HBM-bound.

This reruns the same arms with an L2 flush (write 192 MiB) before each timed replay.
The flush is OUTSIDE the timed region; CUDA events bracket only the graph replay.
"""
import json, statistics, torch, torch.nn as nn

D, N_LIV, WARMUP, ITERS, TRIALS = 1024, 10, 50, 200, 3

class Dense(nn.Module):
    def __init__(s, d=D):
        super().__init__(); s.b = nn.Linear(d, d, bias=False); s.c = nn.Linear(d, d, bias=False)
    def forward(s, x): return s.b(x), s.c(x)

class LowRankSep(nn.Module):
    def __init__(s, d=D, r=128):
        super().__init__()
        s.bd, s.bu = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        s.cd, s.cu = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
    def forward(s, x): return s.bu(s.bd(x)), s.cu(s.cd(x))

class LowRankFused(nn.Module):
    def __init__(s, d=D, r=128):
        super().__init__()
        s.down = nn.Linear(d, 2*r, bias=False)
        s.bu, s.cu = nn.Linear(r, d, bias=False), nn.Linear(r, d, bias=False); s.r = r
    def forward(s, x):
        h = s.down(x); return s.bu(h[..., :s.r]), s.cu(h[..., s.r:])

class Grouped(nn.Module):
    def __init__(s, d=D, g=2):
        super().__init__(); s.g, s.bs = g, d//g
        s.wb = nn.Parameter(torch.randn(g, s.bs, s.bs)/s.bs**0.5)
        s.wc = nn.Parameter(torch.randn(g, s.bs, s.bs)/s.bs**0.5)
    def forward(s, x):
        v = x.reshape(-1, s.g, s.bs).transpose(0, 1)
        return (torch.bmm(v, s.wb).transpose(0,1).reshape(x.shape),
                torch.bmm(v, s.wc).transpose(0,1).reshape(x.shape))

MIB = {"dense":40.0, "lowrank_fused r=128":10.0, "lowrank_fused r=512":40.0,
       "lowrank_sep r=128":10.0, "grouped g=2":20.0, "grouped g=4":10.0}
ARMS = [("dense", Dense), ("lowrank_fused r=128", lambda: LowRankFused(r=128)),
        ("lowrank_fused r=512", lambda: LowRankFused(r=512)),
        ("lowrank_sep r=128", lambda: LowRankSep(r=128)),
        ("grouped g=2", lambda: Grouped(g=2)), ("grouped g=4", lambda: Grouped(g=4))]

def bench(stack, x, flush_buf):
    def step():
        for m in stack: m(x)
    for _ in range(WARMUP): step()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): step()
    torch.cuda.synchronize()
    out = {}
    for mode in ("hot", "flushed"):
        samples = []
        for _ in range(ITERS):
            if mode == "flushed": flush_buf.zero_()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); gr.replay(); e1.record()
            torch.cuda.synchronize()
            samples.append(e0.elapsed_time(e1)*1000.0)
        out[mode] = statistics.median(samples)
    return out

def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  L2={p.L2_cache_size/2**20:.0f} MiB")
    x = torch.randn(1, D, device="cuda", dtype=torch.bfloat16)
    flush_buf = torch.empty(192*1024*1024//4, dtype=torch.float32, device="cuda")
    res = {}
    for tag, ctor in ARMS:
        stack = [ctor().cuda().to(torch.bfloat16).eval() for _ in range(N_LIV)]
        with torch.no_grad():
            trials = [bench(stack, x, flush_buf) for _ in range(TRIALS)]
        hot = statistics.median([t["hot"] for t in trials])
        fl  = statistics.median([t["flushed"] for t in trials])
        mib = MIB[tag]
        res[tag] = {"hot_us": hot, "flushed_us": fl, "mib": mib,
                    "hot_GBs": mib*2**20/hot/1e3, "flushed_GBs": mib*2**20/fl/1e3}
        print(f"  {tag:<22} hot {hot:7.2f}us ({mib*2**20/hot/1e3:6.1f} GB/s)   "
              f"flushed {fl:7.2f}us ({mib*2**20/fl/1e3:6.1f} GB/s)   delta {fl-hot:+7.2f}us")
        del stack; torch.cuda.empty_cache()
    print("\n=== vs dense ===")
    for mode in ("hot","flushed"):
        d = res["dense"][mode+"_us"]
        print(f"  [{mode}]")
        for tag,v in res.items():
            if tag=="dense": continue
            print(f"    {tag:<22} {(d-v[mode+_us])/d*100:+6.2f}%")
    json.dump(res, open("/scratch/users/ericrcwu/liv/p1_l2flush_results.json","w"), indent=2)
    print("\nwrote p1_l2flush_results.json")

main()
