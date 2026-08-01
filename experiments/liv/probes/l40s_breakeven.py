"""Recompute the P1 low-rank-gate launch-overhead breakeven for FarmShare's actual GPU.

The design doc quotes 4.72 us/launch (A100) and 2.19 us (H100). FarmShare has neither --
KDA/HANDOFF.md records the gpu partition as NVIDIA L40S 46 GB, sm_89. Bandwidth differs by
~2.2x from A100, so the breakeven differs too and the microbenchmark's pass/fail line moves.

Model (same as the A100/H100 derivation):
  Factorizing one gate d->d into d->r->d changes the WEIGHT BYTES read per decode step from
  d^2 to 2dr, but adds one extra kernel launch (two GEMMs instead of one).
  Per LIV layer there are 2 gates, and the 350M geometry has 10 LIV layers,
  so the variant adds 2*10 = 20 launches per decoded token.

  saving_s   = (bytes_dense - bytes_lowrank) / effective_bandwidth
  overhead_s = n_extra_launches * launch_us

  breakeven launch_us = saving_us / n_extra_launches

Above that launch cost, P1 is a net LATENCY LOSS even though it reads fewer bytes.
"""

# --- geometry (frozen 350M, per docs/liv-kda-gqa-sub500m-experiment.md) -------------------
D = 1024          # hidden size
N_LIV = 10        # LIV layers (16 total = 10 conv + 6 GQA)
GATES_PER_LIV = 2 # B (pre-gate) and C (post-gate)
BYTES_PER_PARAM = 2  # bf16

N_EXTRA_LAUNCHES = N_LIV * GATES_PER_LIV  # 20

# --- hardware -----------------------------------------------------------------------------
# Peak HBM bandwidth, GB/s (vendor spec). Achieved fraction for skinny GEMV-like decode
# matmuls is well below peak; 0.75 is the usual charitable figure and is what the
# A100 number in the design doc assumed, so we keep it for comparability.
GPUS = {
    "A100-40GB (SXM)": 1555.0,
    "A100-80GB (SXM)": 2039.0,
    "H100-80GB (SXM)": 3350.0,
    "L40S-46GB":        864.0,   # <-- FarmShare
}
ACHIEVED_FRAC = 0.75


def gate_bytes(d: int, r: int | None) -> int:
    """Weight bytes for ONE gate projection, per decode step."""
    if r is None:
        return d * d * BYTES_PER_PARAM          # dense d->d
    return 2 * d * r * BYTES_PER_PARAM          # d->r->d


def analyse(gpu: str, bw_gbs: float, ranks=(128, 256, 512)) -> None:
    eff_bw = bw_gbs * 1e9 * ACHIEVED_FRAC       # bytes/s
    dense_total = gate_bytes(D, None) * N_EXTRA_LAUNCHES

    print(f"\n{gpu}   peak {bw_gbs:.0f} GB/s, assumed achieved {eff_bw/1e9:.0f} GB/s")
    print(f"  {'r':>5} {'bytes saved':>13} {'saving':>10} {'breakeven':>11}  {'verdict at 5/10us':>22}")
    for r in ranks:
        lowrank_total = gate_bytes(D, r) * N_EXTRA_LAUNCHES
        saved = dense_total - lowrank_total
        saving_us = saved / eff_bw * 1e6
        breakeven_us = saving_us / N_EXTRA_LAUNCHES

        def verdict(launch_us: float) -> str:
            net = saving_us - N_EXTRA_LAUNCHES * launch_us
            return f"{'WIN ' if net > 0 else 'LOSS'} {net:+.0f}us"

        if saved <= 0:
            print(f"  {r:>5} {saved:>13,} {'--':>10} {'never':>11}   r >= d/2, no saving")
            continue
        print(f"  {r:>5} {saved:>13,} {saving_us:>9.1f}us {breakeven_us:>10.2f}us"
              f"   {verdict(5.0):>10} / {verdict(10.0)}")


if __name__ == "__main__":
    print(f"P1 breakeven: d={D}, {N_LIV} LIV layers x {GATES_PER_LIV} gates "
          f"= {N_EXTRA_LAUNCHES} extra launches/token, bf16")
    print(f"Dense gate weight traffic/token: "
          f"{gate_bytes(D, None) * N_EXTRA_LAUNCHES / 2**20:.2f} MiB")
    for gpu, bw in GPUS.items():
        analyse(gpu, bw)
    print("\nNOTE: breakeven scales with 1/bandwidth -- SLOWER cards are MORE forgiving,")
    print("because the byte saving buys more time while launch cost stays fixed.")
