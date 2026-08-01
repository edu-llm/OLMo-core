"""Schedule + cost model for the P1 quality pilot: FarmShare vs AWS.

All prices verified from the live AWS Pricing / Spot APIs in sbsandbox, us-east-1, 2026-08-01.
Throughput is the one input still being measured (FarmShare job 1671574), so every schedule is
reported across a RANGE rather than a single number, and the range is stated wherever a
conclusion depends on it.
"""

L0_PARAMS = 338_886_400
L0_NONEMB = L0_PARAMS - 50_304 * 1024
CORPUS = 1_200_000_000

# (label, gpus, $/hr, per-GPU rel. speed vs one L40S, note)
# Relative speed: L40S 864 GB/s HBM, 362 dense BF16 TFLOPS. A100-40G 1,555 GB/s / 312 TFLOPS
# -> ~1.5x on a memory-bound 350M model. H100 3,350 GB/s / ~990 TFLOPS -> ~2.5x.
OPTIONS = [
    ("FarmShare L40S (free, 4 GPU cap)",      4, 0.00,  1.0, "queue-shared, 48h walltime"),
    ("g6e.xlarge  1x L40S  on-dem",           1, 1.861, 1.0, "spot == on-demand, no discount"),
    ("g6e.12xlarge 4x L40S on-dem",           4, 10.4926, 1.0, "$2.62/GPU-hr"),
    ("g6e.12xlarge 4x L40S SPOT us-east-1a",  4, 3.8588, 1.0, "$0.96/GPU-hr - best L40S price"),
    ("g6e.48xlarge 8x L40S SPOT us-east-1d",  8, 13.5897, 1.0, "$1.70/GPU-hr"),
    ("p4d.24xlarge 8x A100-40G SPOT 1a",      8, 8.9580, 1.5, "$1.12/GPU-hr, 1.8x HBM of L40S"),
    ("p5.48xlarge 8x H100 SPOT",              8, 21.0363, 2.5, "$2.63/GPU-hr, fastest wall-clock"),
]

# Candidate pilot designs. The HANDOFF figure of "~2 GPU-hours" is included to show what it buys.
DESIGNS = [
    ("HANDOFF as written: 4 arms x 4 seeds, 2 GPU-hr total", 4, 4, None, 2.0),
    ("Minimal:  3 arms x 2 seeds x 300M tok",  3, 2, 300_000_000, None),
    ("Lean:     3 arms x 3 seeds x 500M tok",  3, 3, 500_000_000, None),
    ("Full:     4 arms x 4 seeds x 1B tok",    4, 4, 1_000_000_000, None),
]


def fmt_hours(h: float) -> str:
    if h < 1:
        return "{:.0f} min".format(h * 60)
    if h < 48:
        return "{:.1f} h".format(h)
    return "{:.1f} d".format(h / 24)


def main() -> None:
    print("=" * 100)
    print("PART 1 -- what does the HANDOFF's '~2 GPU-hours' actually buy?")
    print("=" * 100)
    for tps in (5_000, 15_000, 30_000):
        tok = 2.0 * 3600 * tps
        per_run = tok / 16
        print("  at {:>6,} tok/s: {:>13,.0f} tok total = {:>11,.0f} tok/run over 16 runs"
              "  = {:.3f} tok/param".format(tps, tok, per_run, per_run / L0_PARAMS))
    print()
    print("  Chinchilla-optimal for 338.9M params is 20 tok/param = {:,} tokens PER RUN.".format(
        20 * L0_PARAMS))
    print("  So '2 GPU-hours' is 3-4 ORDERS OF MAGNITUDE short. At 0.02 tok/param every arm is")
    print("  still in the initial loss drop and the comparison is pure seed noise.")
    print("  NOTE the corpus itself is only {:,} tokens = {:.1f} tok/param -> Chinchilla".format(
        CORPUS, CORPUS / L0_PARAMS))
    print("  for a single run would need ~5.7 epochs. The pilot is well under 1 epoch either way.")
    print()

    print("=" * 100)
    print("PART 2 -- GPU-hours per design (hardware-independent), at 3 throughput assumptions")
    print("=" * 100)
    print("{:<48}{:>10}{:>14}{:>13}{:>13}".format(
        "design", "runs", "tok total", "GPU-h @15k", "GPU-h @30k"))
    for label, arms, seeds, tok_per_run, fixed_gpu_h in DESIGNS:
        runs = arms * seeds
        if fixed_gpu_h is not None:
            print("{:<48}{:>10}{:>14}{:>13.1f}{:>13.1f}".format(
                label, runs, "(fixed)", fixed_gpu_h, fixed_gpu_h))
            continue
        total = runs * tok_per_run
        print("{:<48}{:>10}{:>14,}{:>13.0f}{:>13.0f}".format(
            label, runs, total, total / 15_000 / 3600, total / 30_000 / 3600))
    print()

    print("=" * 100)
    print("PART 3 -- wall clock and cost, at 15,000 tok/s per L40S (the middle assumption)")
    print("=" * 100)
    TPS = 15_000
    for label, arms, seeds, tok_per_run, fixed in DESIGNS:
        if fixed is not None:
            continue
        runs = arms * seeds
        total = runs * tok_per_run
        print()
        print("--- {}  ({} runs, {:,} tokens) ---".format(label, runs, total))
        print("    {:<42}{:>12}{:>12}{:>10}".format("platform", "wall clock", "cost", "$/run"))
        for name, gpus, hr, rel, note in OPTIONS:
            gpu_h = total / (TPS * rel) / 3600
            # One run per GPU: embarrassingly parallel, no distributed overhead.
            waves = -(-runs // gpus)
            wall = waves * (tok_per_run / (TPS * rel) / 3600)
            cost = hr * wall if gpus else 0.0
            print("    {:<42}{:>12}{:>12}{:>10}".format(
                name, fmt_hours(wall),
                "free" if hr == 0 else "${:,.0f}".format(cost),
                "-" if hr == 0 else "${:,.0f}".format(cost / runs)))
            del gpu_h, note
    print()
    print("=" * 100)
    print("PART 4 -- price facts worth knowing")
    print("=" * 100)
    print("  * g6e.xlarge SPOT == ON-DEMAND ($1.861). Single-GPU L40S spot has no discount at all.")
    print("  * g6e.12xlarge SPOT is AZ-dependent and the spread is huge: us-east-1a $3.86,")
    print("    1b $6.53, 1c $8.60, 1d $9.89. Pinning us-east-1a is a 2.6x saving vs 1d.")
    print("  * p4d.24xlarge SPOT is also AZ-split: 1a $8.96 vs 1b/1c/1d $18.6-19.2 (2.1x).")
    print("  * Cheapest L40S anywhere: g6e.12xlarge spot 1a = $0.96/GPU-hr, HALF g6e.xlarge on-dem.")
    print("  * Cheapest fast GPU: p4d spot 1a = $1.12/GPU-hr for A100-40G (1.8x the HBM of L40S).")
    print("  * Spot interruption is the risk these prices buy; needs checkpoint/resume to be safe.")


if __name__ == "__main__":
    main()
