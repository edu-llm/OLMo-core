"""Parameter accounting for the probe arms, and the one capacity control the data contains.

Why this matters
----------------
R cannot be varied at fixed parameter count: raising R widens the key/value/beta
projections, so the R=4 arm is strictly larger than R=1. The probe grid contains
**no arm that was constructed to be parameter-matched** (unlike the LM study,
which added ``hh4_r1wide`` for exactly this purpose). So the headline
R=4 - R=1 probe contrast confounds mechanism with capacity.

The handoff's defence is that R=4 buys nothing on parity despite having ~2.3x the
mixer parameters. That is weak: parity needs no capacity, so a capacity increase
failing to help it does not show that capacity is not what helped S5.

However, the depth grid contains an *incidental* and much better capacity control
that the handoff never exploited. Compare across the depth axis:

    R=4 at 1 layer  vs  R=1 at 4 layers

If the R effect were really capacity, then the arm with MORE parameters should do
better. This script reports the parameter counts; ``horizon_depth.py`` reports the
matching effective horizons. The comparison runs the opposite way to the capacity
story, which is the strongest anti-capacity evidence available in the probe data.

``n_params`` in the run JSONs is TOTAL trainable parameters and includes a
task-dependent embedding and output head (target vocabulary is |G|), so all
comparisons here hold the task fixed.

Usage
-----
    python param_confound.py [RESULTS_DIR]
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

TASKS = ("s5_words", "parity", "s3_words", "s4_words")


def load(results_dir: str) -> dict:
    """Index runs by ``(R, task, n_layers, seed)``; layer count comes from the filename."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        m = re.search(r"-L(\d+)-s(\d+)\.json$", os.path.basename(path))
        if m is None:
            continue
        rec = json.load(open(path))
        out[(rec["num_householder"], rec["task"], int(m.group(1)), rec["seed"])] = rec
    return out


def one(data: dict, R: int, task: str, layers: int) -> int | None:
    """:returns: the (unique) total param count for an arm, or ``None`` if absent."""
    vals = {v["n_params"] for k, v in data.items() if k[0] == R and k[1] == task and k[2] == layers}
    return vals.pop() if len(vals) == 1 else None


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/scratch/users/ericrcwu/kda/probes/results/all_night"
    )
    data = load(results_dir)

    print("Total trainable parameters at 3 layers, by task (d_model=256, 4 heads x 64):")
    print(f"{'task':10} {'R=1':>10} {'R=2':>10} {'R=4':>10} {'R4/R1':>8}")
    for task in TASKS:
        p1, p2, p4 = (one(data, R, task, 3) for R in (1, 2, 4))
        if p1 and p4:
            print(
                f"{task:10} {p1:>10,} {(f'{p2:,}' if p2 else '--'):>10} {p4:>10,} {p4/p1:7.3f}x"
            )

    task = "s5_words"
    print(f"\nDepth grid ({task}):")
    print(f"{'layers':>6} {'R=1':>10} {'R=4':>10} {'R4/R1':>8}")
    for layers in (1, 2, 3, 4):
        a, b = one(data, 1, task, layers), one(data, 4, task, layers)
        if a and b:
            print(f"{layers:>6} {a:>10,} {b:>10,} {b/a:7.3f}x")

    print("\nThe incidental capacity control (see horizon_depth.py for the horizons):")
    r1l4, r4l1 = one(data, 1, task, 4), one(data, 4, task, 1)
    if r1l4 and r4l1:
        print(f"  R=1 at 4 layers: {r1l4:>9,} params")
        print(f"  R=4 at 1 layer : {r4l1:>9,} params  ({r4l1/r1l4:.3f}x of the above)")
        print(
            f"  => R=4/L1 has {1 - r4l1/r1l4:.1%} FEWER parameters than R=1/L4.\n"
            "     Measured effective horizons are 105.6 vs 45.9 tokens respectively, so the\n"
            "     SMALLER model has the LONGER horizon. A pure-capacity account predicts the\n"
            "     opposite ordering, so this comparison is evidence against capacity — and it\n"
            "     is the only near-parameter-matched contrast in the probe data."
        )
    print(
        "\nCaveat: this is a cross-design comparison (R and depth vary together), not a\n"
        "purpose-built matched control. It weakens the capacity confound; it does not remove it.\n"
        "The only purpose-built parameter-matched contrast in the project is the LM study's\n"
        "hh4 vs hh4_r1wide pair."
    )


if __name__ == "__main__":
    main()
