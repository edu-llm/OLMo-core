"""Effective-horizon reframing of the KDA-Householder probe results.

Motivation
----------
The probe metric is **mean token accuracy over all positions** of the evaluation
sequence (``train_probe.py``'s ``evaluate``), and the group-word tasks mark no
position with ``-100``, so every position enters the denominator. Training draws
lengths uniformly from ``[3, 40]``. Consequently an accuracy reported "at length
2048" is not the model's accuracy on a length-2048 state-tracking problem; it is
an average over a prefix the model handles and a tail where it has degraded.

That makes raw accuracy at long lengths hard to interpret: an arm at 6.78% and an
arm at 2.85% (S5, length 2048) are both far below the 0.83% chance level yet both
have plainly failed the task in the tail. The *difference* between them is real
but its magnitude in percentage points is an artifact of how much easy prefix the
average includes, which shrinks as 1/L.

Model
-----
Assume the model is correct on the first ``h`` positions and at chance thereafter::

    acc(L) = (h / L) * 1 + ((L - h) / L) * chance          for h <= L

Solving for ``h`` gives the **effective horizon**, in tokens::

    h = L * (acc - chance) / (1 - chance)

``h`` is a length-independent summary: if one scalar ``h`` reproduces a run's
accuracy at every evaluation length, then the seven per-length numbers are seven
views of one quantity rather than seven independent findings. This script reports
the within-run coefficient of variation of ``h`` across lengths to test exactly
that, and then reports the paired R=4 - R=1 horizon effect per task.

Cells at or above 99.9% accuracy are excluded from the CV and the per-run mean,
because there ``h`` is pinned to ``L`` by the ceiling and carries no information.

Chance levels are derived from the target vocabulary in ``probes/tasks.py``: the
target is the index of the accumulated group product, so chance is ``1/|G|``
(parity 1/2, S3 1/6, S4 1/24, S5 1/120).

Caveats
-------
The two-regime (solved prefix / chance tail) model is a simplification; real
degradation is gradual. It is used here as a *descriptive* reparameterization,
not a fitted claim about the mechanism. Its adequacy is testable and is reported:
a low within-run CV means the one-parameter description is good.

Usage
-----
    python effective_horizon.py [RESULTS_DIR] [OUT_TSV]

Defaults to the FarmShare results directory. Runs on CPU; needs only the stdlib.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys

# Target-space cardinality |G| per task, from probes/tasks.py ("out_vocab").
CHANCE = {"parity": 1 / 2, "s3_words": 1 / 6, "s4_words": 1 / 24, "s5_words": 1 / 120}
SOLVABLE = {"parity": "yes", "s3_words": "yes", "s4_words": "yes", "s5_words": "NO"}
LENGTHS = ["40", "64", "128", "256", "512", "1024", "2048"]
TRAIN_MAX = 40  # probes/run_packed.sbatch leaves --train-max at its default of 40.
CEILING = 0.999  # Above this, h is pinned to L and is uninformative.

# Two-sided t at alpha=.05 by degrees of freedom (n-1).
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def load(results_dir: str) -> dict:
    """Index runs by ``(R, task, n_layers, seed)``.

    The layer count is recovered from the FILENAME: ``train_probe.py`` does not
    persist ``--n-layers`` into the JSON, and the depth grid (L1/L2/L4) shares
    this directory with the 3-layer confirmatory grid.
    """
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        m = re.search(r"-L(\d+)-s\d+\.json$", os.path.basename(path))
        if m is None:
            continue
        rec = json.load(open(path))
        out[(rec["num_householder"], rec["task"], int(m.group(1)), rec["seed"])] = rec
    return out


def horizon(acc: float, length: int, chance: float) -> float:
    """:returns: effective horizon in tokens, ``L (acc - chance) / (1 - chance)``."""
    return length * (acc - chance) / (1 - chance)


def run_horizon(rec: dict, chance: float) -> tuple[float, float]:
    """Mean and CV of ``h`` across non-ceiling lengths for one run."""
    hs = [
        horizon(rec["accuracy_by_length"][L], int(L), chance)
        for L in LENGTHS
        if rec["accuracy_by_length"][L] <= CEILING
    ]
    if not hs:
        return float("nan"), float("nan")
    mean = st.mean(hs)
    return mean, (st.stdev(hs) / mean if len(hs) > 1 and mean else 0.0)


def paired(data: dict, task: str, chance: float) -> tuple:
    """Paired-by-seed R=4 - R=1 horizon contrast at 3 layers."""
    seeds = sorted(
        {s for (r, t, lay, s) in data if r == 1 and t == task and lay == 3}
        & {s for (r, t, lay, s) in data if r == 4 and t == task and lay == 3}
    )
    h1 = [run_horizon(data[(1, task, 3, s)], chance)[0] for s in seeds]
    h4 = [run_horizon(data[(4, task, 3, s)], chance)[0] for s in seeds]
    diffs = [b - a for a, b in zip(h1, h4)]
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    ci = T_CRIT.get(n - 1, 2.0) * sd / math.sqrt(n) if n > 1 else 0.0
    return n, st.mean(h1), st.mean(h4), mean, ci, sd


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/scratch/users/ericrcwu/kda/probes/results/all_night"
    )
    out_tsv = sys.argv[2] if len(sys.argv) > 2 else "probe_effective_horizon.tsv"
    data = load(results_dir)

    rows = []
    for task in ("s5_words", "parity", "s3_words", "s4_words"):
        chance = CHANCE[task]
        n, mh1, mh4, mean, ci, sd = paired(data, task, chance)
        cvs = []
        for R in (1, 4):
            for (r, t, lay, s) in [k for k in data if k[0] == R and k[1] == task and k[2] == 3]:
                _, cv = run_horizon(data[(r, t, lay, s)], chance)
                if not math.isnan(cv):
                    cvs.append(cv)
        rows.append(
            dict(
                task=task,
                solvable=SOLVABLE[task],
                group_order=round(1 / chance),
                chance_pct=100 * chance,
                n=n,
                h_R1=mh1,
                h_R4=mh4,
                delta_h=mean,
                ci_lo=mean - ci,
                ci_hi=mean + ci,
                sd=sd,
                verdict="SIG" if abs(mean) > ci else "ns",
                within_run_cv_pct=100 * st.mean(cvs) if cvs else float("nan"),
                h_R1_over_train_max=mh1 / TRAIN_MAX,
            )
        )

    cols = list(rows[0].keys())
    with open(out_tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write(
                "\t".join(
                    f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols
                )
                + "\n"
            )

    print(f"train_max = {TRAIN_MAX} tokens; chance = 1/|G|; ceiling excluded at >{CEILING}\n")
    hdr = f"{'task':10s} {'solv':>5s} {'n':>2s} {'h(R1)':>7s} {'h(R4)':>7s} {'dh':>8s} {'95% CI':>18s} {'CV':>6s}"
    print(hdr)
    for r in rows:
        print(
            f"{r['task']:10s} {r['solvable']:>5s} {r['n']:>2d} {r['h_R1']:7.1f} {r['h_R4']:7.1f} "
            f"{r['delta_h']:+8.1f}   [{r['ci_lo']:+7.1f},{r['ci_hi']:+7.1f}] "
            f"{r['within_run_cv_pct']:5.1f}%  {r['verdict']}"
        )
    print(f"\nwrote {out_tsv}")


if __name__ == "__main__":
    main()
