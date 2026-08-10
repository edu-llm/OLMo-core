#!/usr/bin/env python3
"""
Benchmark harness for the multiply-free ternary MoE decode kernel.

**UNRUN as of authoring. This file contains NO measured numbers and makes NO speedup claim.**

Run only after ``mulfree_correctness.py`` passes. A speed number on an incorrect kernel is worse
than no number.

    TRITON_F32_DEFAULT=ieee PYTHONPATH=$MINE/src $V/bin/python \\
        $MINE/src/scripts/maple/mulfree_bench.py --rung M7B --l2-multiple 4.0

Design, against the four ways this project has previously reported a wrong sign
------------------------------------------------------------------------------
**1. The cache trap (D-107 -- a physically impossible 4.77x, built on cache residency).**
The bank is tiled with ``Tensor.repeat`` -- which **copies**, it is tile not ``expand`` -- until the
resident footprint is at least ``--l2-multiple`` times the device's *measured* L2, and the multiple
is **printed**. Successive dispatch replicas index physically distinct regions, so there is no
wraparound reuse. And a **deliberately-failing control** runs the identical kernel with
``BANK_STRIDE=0``, which re-reads one small region: that arm is *known* to be cache-resident, so if
it does **not** report inflated bandwidth, the guard itself is broken and every other number in the
run is void. **A guard that cannot fail is not a guard** (D-019); this one is demonstrated, not
asserted.

**2. The launch-count trap (D-097 -- ~25 us of dispatch swallowing ~1 us of memory traffic, which
masked the cache signal itself).** All ``R`` replicas go in **one** dispatch, so per-launch overhead
amortises ``R``-fold. Dispatch overhead is *separately* measured by an empty-grid kernel rather than
assumed from D-101's 23.39 us, and both are printed.

**3. Slack (D-101 -- a lane threw out its own passing result because its control had ~2x of slack,
making the claim unfalsifiable).** The primary A/B here needs no cross-dtype control: it is
multiply-free vs multiply-accumulate over the **same kernel, same grid, same bytes**, so byte count
is invariant by construction rather than by argument. What slack *would* invalidate is the
``%-of-achievable`` figure, so the run **refuses to treat that figure as a bandwidth statement**
unless the ternary arm clears ``--min-baseline-frac`` (default 0.85) of measured achievable. A
kernel at 45% has ~2x of headroom for any effect to hide in, and D-101 already showed a tuned
gathered ternary kernel reaching 99.76% of achievable on this device class -- so a materially lower
number here indicts *this* kernel's tuning, and any ratio taken against it would be measuring
tuning, not arithmetic.

**No bf16 arm is implemented here, deliberately.** D-101 already measured the honest
ternary-vs-bf16 expert-path win (7.85x of the nominal 8x, L40S) with a shape-matched control, and
re-measuring it would only risk contradicting a better-controlled result. This harness answers the
one question that record does not: does removing the multiply change anything?

**4. An unreachable denominator (D-101 -- the L40S datasheet 864 GB/s is not reachable; a pure
vectorized stream read tops at 749.7).** Achievable bandwidth is **measured here** by a stream
control in this same process, and it is the denominator. The datasheet figure is reported in
parentheses and clearly labelled as not the denominator to judge by.

The primary comparison is multiply-free vs multiply-accumulate over the **same kernel**, same grid,
same bytes. Since the two arms are required to be **bitwise identical** (see the correctness
script's T2), any timing difference is *not* arithmetic saved -- it is register or select pressure.
**The honest prior is that they tie.** Reporting a tie is a valid result.
"""

import argparse
import os
import statistics
import sys
from typing import Callable, Dict, List

import torch

from olmo_core.kernels.ternary_moe import (
    MEASURED_US_PER_LAUNCH_L40S,
    TernaryExportSpec,
    fused_gathered_w2_combine,
    fused_gathered_w13_swiglu,
    pack_expert_bank,
)


def assert_ieee_env() -> None:
    v = os.environ.get("TRITON_F32_DEFAULT")
    if v != "ieee":
        print(
            f"FATAL: TRITON_F32_DEFAULT is {v!r}, must be 'ieee'. Asserted inside the job because "
            "an export in a wrapper that never reaches the srun payload looks like success.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"env: TRITON_F32_DEFAULT={v}")


def timed(fn: Callable[[], None], n: int, warmup: int) -> Dict[str, float]:
    """
    Median wall time over a fixed window after a fixed discard. Median, never mean.

    ``warmup`` is discarded entirely -- ``torch.compile`` and Triton JIT can take minutes on the
    first call, and the first touch of a fresh buffer is a page-fault path, not a bandwidth path.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / 1e3)
    samples.sort()
    return {
        "median_s": statistics.median(samples),
        "p10_s": samples[max(0, int(0.10 * len(samples)) - 1)],
        "p90_s": samples[min(len(samples) - 1, int(0.90 * len(samples)))],
        "n": len(samples),
    }


def measure_achievable_bandwidth(dev: str, mib: int, n: int, warmup: int) -> float:
    """
    Measure achievable read bandwidth with a pure vectorized stream. **This is the denominator.**

    D-101 measured 749.7 GB/s on L40S against an 864 datasheet figure and ruled that every
    ``%-of-peak`` in this project had been quoted against a ceiling that cannot be hit. Rather than
    inherit that constant, this measures it in-process on whatever device is present -- which is
    also the only way the number is valid on a device that is not an L40S.
    """
    n_el = (mib * 2**20) // 2
    buf = torch.randn(n_el, device=dev, dtype=torch.float16)
    out = torch.zeros(1024, device=dev, dtype=torch.float32)

    def run() -> None:
        out[0] = buf.view(-1, 1024).sum(dim=0)[0]

    t = timed(run, n, warmup)
    return (n_el * 2) / t["median_s"] / 1e9


def measure_launch_overhead(dev: str, n: int, warmup: int) -> float:
    """
    Measure per-launch dispatch overhead in-process rather than inheriting D-101's 23.39 us.

    Launches a trivial 1-program kernel ``n_launch`` times and divides. Inheriting a constant from
    another device would make the launch-count arithmetic unfalsifiable on this one.
    """
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _noop(p):
        tl.store(p + tl.program_id(0), 1)

    p = torch.zeros(1, device=dev, dtype=torch.int32)
    n_launch = 200

    def run() -> None:
        for _ in range(n_launch):
            _noop[(1,)](p)

    t = timed(run, n, warmup)
    return t["median_s"] / n_launch * 1e6


def bench(opts) -> int:
    assert_ieee_env()
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device -- this needs a GPU node via srun, not the login node.")
        return 2
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    l2 = props.L2_cache_size
    print(f"device: {props.name} sm_{props.major}{props.minor}  L2 = {l2/2**20:.2f} MiB (measured)")
    print(
        "TRANSFERABILITY CAVEAT: FarmShare is L40S, sm_89. The eduLLM training target is A100, "
        "sm_80, and A100's grouped-GEMM fast path is gated on sm90/sm100 so it behaves differently "
        "again. NOTHING timed here is an A100 number and it must not be quoted as one. D-101's own "
        "rpp/num_warps optimum was expected to move across devices."
    )

    spec = TernaryExportSpec.from_rung(opts.rung)
    print(
        f"\nrung {spec.rung}: d={spec.d_model} L={spec.n_layers} E={spec.num_experts} "
        f"f_e={spec.expert_hidden} k={spec.top_k}"
    )
    print(f"  {spec.verify_against_transformer_config()}")

    # ---------------- the formulae, printed so every number below is auditable ----------------
    d, fe, E, k = spec.d_model, spec.expert_hidden, spec.num_experts, spec.top_k
    kb13, kb2 = -(-d // 4), -(-fe // 4)
    print("\nFORMULAE (printed so the numbers are auditable, per maple/CLAUDE.md):")
    print(f"  w13 codes/expert  = 2*f_e*ceil(d/4)   = 2*{fe}*{kb13} = {2*fe*kb13} B")
    print(f"  w2  codes/expert  = d*ceil(f_e/4)     = {d}*{kb2} = {d*kb2} B")
    print(f"  alpha/expert      = (2*f_e + d)*4 B   = {(2*fe+d)*4} B")
    print(f"  touched bytes/step (k={k} experts) = k*(codes+alpha) + x + h + y")
    print("  resident bytes    = E*(codes+alpha)*n_replicas   <- the CACHE-RELEVANT figure")
    print("  achieved GB/s     = touched_bytes / median_s / 1e9")
    print("  MACs/step         = k*(2*f_e*d + d*f_e);  conventional FLOPs = 2*MACs")
    print("  mul-free muls     = k*(2*f_e + 2*d)   <- alpha once per (expert, output element)")
    print(f"  arithmetic mix    = {spec.arith_mix()}")

    # ---------------- calibrations, measured not inherited ----------------
    achievable = measure_achievable_bandwidth(dev, opts.stream_mib, opts.calib_iters, opts.warmup)
    print(
        f"\nCALIBRATION 1 -- achievable read bandwidth, measured in-process: {achievable:.1f} GB/s"
    )
    print(
        "  This is THE DENOMINATOR. A datasheet figure is not reachable (D-101: 749.7 measured "
        "vs 864 datasheet on L40S, GDDR6+ECC)."
    )
    us_launch = measure_launch_overhead(dev, opts.calib_iters, opts.warmup)
    print(f"CALIBRATION 2 -- per-launch dispatch overhead, measured in-process: {us_launch:.2f} us")
    print(
        f"  (D-101 measured {MEASURED_US_PER_LAUNCH_L40S} us on L40S. Measured here rather than "
        f"inherited so the launch arithmetic is falsifiable on THIS device.)"
    )

    # ---------------- size the bank past L2 ----------------
    probe = pack_expert_bank(
        torch.randn(E, d, fe, device=dev),
        torch.randn(E, fe, d, device=dev),
        torch.randn(E, d, fe, device=dev),
    )
    one = probe.resident_bytes
    del probe
    torch.cuda.empty_cache()
    n_rep = max(1, -(-int(opts.l2_multiple * l2) // one))
    print(
        f"\nWORKING SET: one bank = {one/2**20:.2f} MiB; L2 = {l2/2**20:.2f} MiB "
        f"({one/l2:.2f}x L2). Tiling n_replicas={n_rep} -> "
        f"{one*n_rep/2**20:.2f} MiB = {one*n_rep/l2:.2f}x L2 (target >= {opts.l2_multiple}x)."
    )
    if one * n_rep / l2 < opts.l2_multiple:
        print("  WARNING: could not reach the target multiple.")

    w1 = torch.randn(E, d, fe, device=dev)
    w2 = torch.randn(E, fe, d, device=dev)
    w3 = torch.randn(E, d, fe, device=dev)
    bank = pack_expert_bank(w1, w2, w3, n_replicas=n_rep)
    del w1, w2, w3
    torch.cuda.empty_cache()
    print(f"  resident = {bank.resident_bytes/2**20:.2f} MiB = {bank.resident_bytes/l2:.2f}x L2")

    R = n_rep
    x = torch.randn(R, d, device=dev, dtype=torch.float32)
    idx = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(R)]).to(torch.int32)
    rw = torch.rand(R, k, device=dev)
    rw = (rw / rw.sum(-1, keepdim=True)).to(torch.float32)
    h = torch.empty((R, k, fe), device=dev, dtype=torch.float32)
    y = torch.empty((R, d), device=dev, dtype=torch.float32)

    # touched bytes for the two-launch pair, counted per dispatch (R steps)
    codes_b = k * (2 * fe * kb13 + d * kb2)
    alpha_b = k * (2 * fe + d) * 4
    x_b, h_b, y_b = d * 4, k * fe * 4, d * 4
    touched = R * (codes_b + alpha_b + x_b + 2 * h_b + y_b)
    print(
        f"  touched bytes/dispatch = {R} * ({codes_b} + {alpha_b} + {x_b} + 2*{h_b} + {y_b}) "
        f"= {touched/2**20:.2f} MiB"
    )

    rows: Dict[str, Dict[str, float]] = {}

    def two_launch(mf: bool, stride: bool = True):
        def run() -> None:
            fused_gathered_w13_swiglu(
                x,
                bank,
                idx,
                multiply_free=mf,
                rows_per_prog=opts.rpp,
                num_warps=opts.num_warps,
                bank_stride_replicas=stride,
                out=h,
            )
            fused_gathered_w2_combine(
                h,
                bank,
                idx,
                rw,
                multiply_free=mf,
                rows_per_prog=opts.rpp,
                num_warps=opts.num_warps,
                bank_stride_replicas=stride,
                out=y,
            )

        return run

    for label, mf in (("mulfree", True), ("mul-acc (control)", False)):
        t = timed(two_launch(mf), opts.iters, opts.warmup)
        gbps = touched / t["median_s"] / 1e9
        rows[label] = {
            "median_us": t["median_s"] * 1e6,
            "p10_us": t["p10_s"] * 1e6,
            "p90_us": t["p90_s"] * 1e6,
            "gbps": gbps,
            "frac_achievable": gbps / achievable,
            "n": t["n"],
        }

    # ---------------- the deliberately-failing cache-trap control ----------------
    t_cached = timed(two_launch(True, stride=False), opts.iters, opts.warmup)
    gbps_cached = touched / t_cached["median_s"] / 1e9
    inflation = gbps_cached / rows["mulfree"]["gbps"]
    rows["mulfree CACHED (must inflate)"] = {
        "median_us": t_cached["median_s"] * 1e6,
        "p10_us": t_cached["p10_s"] * 1e6,
        "p90_us": t_cached["p90_s"] * 1e6,
        "gbps": gbps_cached,
        "frac_achievable": gbps_cached / achievable,
        "n": t_cached["n"],
    }

    print("\n" + "=" * 96)
    print(
        f"{'arm':34s} {'median us':>11s} {'p10':>9s} {'p90':>9s} {'GB/s':>9s} "
        f"{'%achv':>7s} {'%sheet':>7s}"
    )
    for name, r in rows.items():
        print(
            f"{name:34s} {r['median_us']:11.2f} {r['p10_us']:9.2f} {r['p90_us']:9.2f} "
            f"{r['gbps']:9.1f} {100*r['frac_achievable']:6.2f}% "
            f"{100*r['gbps']/864.0:6.2f}%"
        )
    print("=" * 96)
    print(
        "  %achv is against the MEASURED achievable ceiling above. %sheet is against L40S's "
        "864 GB/s datasheet figure and is NOT the denominator to judge by (D-101)."
    )

    # ---------------- verdicts ----------------
    print("\nGUARD -- cache-trap control (this guard is REQUIRED to be able to fail):")
    print(
        f"  BANK_STRIDE=0 re-reads one region: {gbps_cached:.1f} GB/s vs "
        f"{rows['mulfree']['gbps']:.1f} GB/s strided = {inflation:.3f}x inflation"
    )
    if inflation < opts.min_cache_inflation:
        print(
            f"  *** GUARD FAILED: inflation {inflation:.3f}x < {opts.min_cache_inflation}x. A "
            f"known-cache-resident arm did NOT read faster, so this harness cannot distinguish "
            f"cache from HBM and EVERY bandwidth number above is VOID. Do not quote them. ***"
        )
    else:
        print(
            f"  Guard is live: a known-cached arm inflates {inflation:.3f}x, so the guard can "
            f"fail. The strided numbers are HBM-forced."
        )

    ratio = rows["mul-acc (control)"]["median_us"] / rows["mulfree"]["median_us"]
    print("\nPRIMARY RESULT -- multiply-free vs multiply-accumulate, same kernel/grid/bytes:")
    print(
        f"  mulfree {rows['mulfree']['median_us']:.2f} us vs mul-acc "
        f"{rows['mul-acc (control)']['median_us']:.2f} us  ->  {ratio:.4f}x"
    )
    band = (rows["mulfree"]["p90_us"] - rows["mulfree"]["p10_us"]) / rows["mulfree"]["median_us"]
    if abs(ratio - 1.0) <= band:
        print(
            f"  VERDICT: **TIE** -- the difference ({abs(ratio-1)*100:.2f}%) is inside the "
            f"mulfree arm's own p10-p90 spread ({band*100:.2f}%). This is the EXPECTED result. "
            f"The two arms are bitwise identical (correctness T2), so there was no arithmetic to "
            f"save: removing a multiply frees no issue slot when fma and add are the same rate."
        )
    else:
        print(
            f"  VERDICT: a {abs(ratio-1)*100:.2f}% difference exceeds the p10-p90 spread "
            f"({band*100:.2f}%). Since the arms are bitwise identical, this is register/select "
            f"pressure or scheduling -- NOT saved arithmetic. Read T6's PTX histogram before "
            f"attributing it to anything."
        )

    print("\nSLACK CHECK (D-101: a control with 2x of slack makes a claim unfalsifiable):")
    fa = rows["mulfree"]["frac_achievable"]
    if fa < opts.min_baseline_frac:
        print(
            f"  Ternary arm is at {100*fa:.2f}% of achievable, below the "
            f"{100*opts.min_baseline_frac:.0f}% floor. It is NOT bandwidth-bound, so its "
            f"%-of-achievable is not a bandwidth statement and no ratio against any other arm "
            f"should be quoted from this run. Tune rpp/num_warps first."
        )
    else:
        print(f"  Ternary arm at {100*fa:.2f}% of achievable -- bandwidth-bound, little slack.")
    print(
        "  NOTE: D-101 already measured a tuned gathered ternary kernel at 99.76% of "
        "achievable on this device class. If this arm lands materially below that, the gap is "
        "THIS kernel's tuning, not a property of multiply-free arithmetic."
    )

    # ---------------- launch accounting ----------------
    lc = spec.launch_counts()
    print(f"\nLAUNCH ACCOUNTING for {spec.rung} (expert path, per token):")
    print(f"  naive  = L*k*3 = {spec.n_layers}*{k}*3 = {lc['naive_launches']:.0f} launches")
    print(
        f"  fused  = L*2   = {spec.n_layers}*2   = {lc['fused_launches']:.0f} launches "
        f"-> {lc['reduction_vs_naive']:.1f}x fewer"
    )
    print(
        f"  at this device's MEASURED {us_launch:.2f} us/launch: "
        f"{lc['naive_launches']*us_launch:.0f} us -> {lc['fused_launches']*us_launch:.0f} us "
        f"of dispatch/token"
    )
    print(
        f"  dispatch-only ceiling: {1e6/(lc['naive_launches']*us_launch):.0f} tok/s -> "
        f"{1e6/(lc['fused_launches']*us_launch):.0f} tok/s"
    )
    print(
        "  CAVEAT: CUDA-graph capture makes a decode step ONE replay, which SUBSUMES most of "
        "this. Quote the reduction against the naive UNGRAPHED path only. What fusion still buys "
        "under a graph is fewer graph nodes and one fewer HBM round-trip for the gate/up "
        "intermediate -- real, but much smaller than the raw launch ratio."
    )

    print(
        "\nNOT MEASURED HERE, stated rather than implied: no tokens/s headline (the expert path "
        "is a minority of bytes/token -- lm_head dominates at 58.4% at M20, D-104); no prefill "
        "(variable-M, genuinely the grouped-GEMM problem); no A100 or sm_90 number; no "
        "end-to-end decode loop; no ncu counters, so any mechanism statement is timing plus "
        "arithmetic, not hardware counters."
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rung", default="M7B", choices=["R0", "R1", "R2", "R3", "E8", "M20", "M7B"])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=50, help="discarded; >=50 per project rule")
    p.add_argument("--calib-iters", type=int, default=50)
    p.add_argument("--stream-mib", type=int, default=512)
    p.add_argument("--l2-multiple", type=float, default=4.0)
    p.add_argument("--min-cache-inflation", type=float, default=1.15)
    p.add_argument("--min-baseline-frac", type=float, default=0.85)
    p.add_argument("--rpp", type=int, default=8, help="D-101's L40S optimum; expected to move")
    p.add_argument("--num-warps", type=int, default=2)
    opts = p.parse_args()
    if opts.warmup < 50:
        print("FATAL: >=50 warmup iterations are required by maple/CLAUDE.md.", file=sys.stderr)
        return 2
    return bench(opts)


if __name__ == "__main__":
    raise SystemExit(main())
