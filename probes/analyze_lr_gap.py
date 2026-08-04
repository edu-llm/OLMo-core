"""Ask whether a learning rate exists that closes the strict arms' in-distribution gap.

The 48-cell regime-arity sweep left one thing unexplained. Under the strict beta regime the
probe reaches ~0.0000 *training* loss by step 1000 and still scores only 68-89% on held-out
sequences of length 40 -- a length inside the training range of 3..40. The reflection arms
score 100% (A5) and 93-99% (S5) on the same bank. That is not underfitting and it is not a
failure to converge: it is a generalization gap on data drawn from the training distribution.

Two explanations survive that observation, and they are not distinguishable from the sweep:

1. **Optimization.** The strict parameterization is harder to optimize, and the rate the sweep
   used (1e-3 peak, OneCycle) lands it in a worse basin. A different rate would close the gap.
2. **Expressivity or inductive bias.** Strict beta cannot represent the solution as cleanly, so
   no rate closes the gap and the deficit is a property of the model class.

This module reads an LR sweep over the strict arms and reports the in-distribution accuracy at
each rate, against the reflection arms' accuracy as the target to be reached. If the best rate
still falls short of the reflection arms, (1) is ruled out to the resolution of the grid --
which is the useful direction, because it is the explanation that would otherwise undercut
every claim the sweep makes.

Usage::

    python probes/analyze_lr_gap.py <dir-of-records> [--length 40] [--json out.json]
    python probes/analyze_lr_gap.py <dir> --reference <dir-of-48-cell-sweep>

Why this is not a flag on ``analyze_regime_arity.py``
-----------------------------------------------------
That module measures an interaction across a 2x2 square and needs all four arms present. This
one varies a *continuous* treatment on two arms and has no square at all. Bolting a second
design onto the first would mean every table there grew a dimension that is absent from every
record it was written for.

:raises SystemExit: If the directory holds no usable records, or holds only one learning rate.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Optional

#: The evaluation length that is inside the training range. The probe trains on lengths
#: sampled uniformly from ``[--train-min, --train-max]`` = ``[3, 40]``, so 40 is the longest
#: length the model has actually seen and the only evaluation point that is not extrapolation.
#: A gap here cannot be explained by length generalization, which is what makes it worth
#: isolating from the rest of the curve.
DEFAULT_LENGTH = 40

#: Arms whose in-distribution accuracy is the target. They are not swept: both already score at
#: or near ceiling at ``DEFAULT_LENGTH``, so there is no gap for a learning rate to close and a
#: sweep over them would spend cells to confirm 100% four more times.
REFERENCE_ARMS = ("R1-refl", "Reflection")


def mean(v: list[float]) -> float:
    """Arithmetic mean."""
    return sum(v) / len(v)


def sd(v: list[float]) -> float:
    """Sample standard deviation (n-1). ``nan`` for fewer than two values."""
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def load(directory: str) -> list[dict]:
    """Load every completed, provenanced record under ``directory``.

    Filters match ``analyze_regime_arity.load`` on purpose: a record whose ``outcome`` is not
    ``completed``, or whose ``probe_source_revision`` is absent or ``"unknown"``, is dropped
    rather than averaged in.

    :param directory: Directory holding the records; searched recursively.
    :returns: The usable records.
    :raises SystemExit: If none are usable.
    """
    out: list[dict] = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue
        if d.get("outcome") != "completed":
            continue
        if d.get("probe_source_revision") in (None, "unknown"):
            continue
        out.append(d)
    if not out:
        raise SystemExit(f"no usable records under {directory}")
    return out


def accuracy_at(record: dict, length: int) -> Optional[float]:
    """Percent accuracy at ``length``, or ``None`` if the record has no entry for it."""
    v = record.get("accuracy_by_length", {}).get(str(length))
    return None if v is None else 100.0 * v


def fit_quality(record: dict) -> Optional[float]:
    """The run's tail-mean training loss, the honest read on whether it fit the training set.

    Prefers ``loss_summary.tail_mean`` and falls back to the minimum sampled loss after warmup
    for records written before that field existed. Deliberately does NOT fall back to the last
    entry of ``loss_trace``: that is one minibatch at the final step, where OneCycle has driven
    the rate to ~4e-9 and the weights are frozen, so a high value there is a hard batch rather
    than a model that failed to converge. Reading it as convergence is exactly the error that
    produced a false 'unconverged' finding on this probe.

    :param record: One result record.
    :returns: The tail-mean loss, or ``None`` if neither field is available.
    """
    summary = record.get("loss_summary") or {}
    if "tail_mean" in summary:
        return summary["tail_mean"]
    trace = record.get("loss_trace") or []
    return min((v for _, v in trace[2:]), default=None)


def main() -> None:
    """Report in-distribution accuracy by learning rate, against the reference arms."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", help="Directory holding the LR sweep's records.")
    p.add_argument(
        "--reference",
        default=None,
        help=(
            "Directory holding records for the reflection arms, if they are not in the sweep "
            "directory. The 48-cell regime-arity sweep is the natural source."
        ),
    )
    p.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="Evaluation length.")
    p.add_argument("--json", default=None, help="Also write the full result here.")
    opts = p.parse_args()

    records = load(opts.directory)
    rates = sorted({r["lr"] for r in records if r.get("lr") is not None})
    print(f"# {len(records)} records; learning rates {rates}")
    if len(rates) < 2:
        raise SystemExit(
            f"this directory holds {len(rates)} learning rate(s); an LR sweep needs at least "
            f"two. Records written before the 'lr' field existed do not carry one."
        )

    reference = load(opts.reference) if opts.reference else records
    tasks = sorted({r["task"] for r in records})
    swept_arms = sorted({r["arm"] for r in records if r["arm"] not in REFERENCE_ARMS})

    out: dict = {"length": opts.length, "rates": rates, "tasks": {}}

    for task in tasks:
        print(f"\n## {task}: accuracy at L={opts.length} (inside the training range 3..40)")
        print(f"{'arm':12s} {'lr':>8} {'n':>3} {'acc%':>8} {'sd':>7} {'fit loss':>9}")
        out["tasks"][task] = {"swept": {}, "reference": {}}

        for arm in swept_arms:
            for rate in rates:
                cells = [
                    r
                    for r in records
                    if r["task"] == task and r["arm"] == arm and r.get("lr") == rate
                ]
                accs = [a for a in (accuracy_at(c, opts.length) for c in cells) if a is not None]
                if not accs:
                    continue
                losses = [x for x in (fit_quality(c) for c in cells) if x is not None]
                print(
                    f"{arm:12s} {rate:>8.0e} {len(accs):>3} {mean(accs):>8.2f} "
                    f"{sd(accs):>7.2f} {mean(losses) if losses else float('nan'):>9.4f}"
                )
                out["tasks"][task]["swept"].setdefault(arm, {})[str(rate)] = {
                    "n": len(accs),
                    "mean": mean(accs),
                    "sd": sd(accs),
                    "fit_loss": mean(losses) if losses else None,
                    "bundles": sorted(c.get("bundle_id") for c in cells),
                }

        # The bar. Printed in the same table so the comparison does not require holding a
        # number from another run in your head.
        best_by_arm: dict[str, float] = {}
        for arm in swept_arms:
            per_rate = out["tasks"][task]["swept"].get(arm, {})
            if per_rate:
                best_by_arm[arm] = max(v["mean"] for v in per_rate.values())
        for arm in REFERENCE_ARMS:
            cells = [r for r in reference if r["task"] == task and r["arm"] == arm]
            accs = [a for a in (accuracy_at(c, opts.length) for c in cells) if a is not None]
            if not accs:
                continue
            print(f"{arm:12s} {'(ref)':>8} {len(accs):>3} {mean(accs):>8.2f} {sd(accs):>7.2f}")
            out["tasks"][task]["reference"][arm] = {
                "n": len(accs),
                "mean": mean(accs),
                "sd": sd(accs),
            }

        target = out["tasks"][task]["reference"]
        if target and best_by_arm:
            bar = max(v["mean"] for v in target.values())
            print(f"#   reference ceiling: {bar:.2f}%")
            for arm, best in sorted(best_by_arm.items()):
                shortfall = bar - best
                verdict = "CLOSED" if shortfall <= 1.0 else f"still short by {shortfall:.2f}pp"
                print(f"#   {arm:12s} best over all rates: {best:.2f}% -> {verdict}")
            out["tasks"][task]["ceiling"] = bar
            out["tasks"][task]["best_by_arm"] = best_by_arm

    print("\n## READING THIS")
    print(f"# L={opts.length} is inside the training range, so a deficit here is NOT length")
    print("# generalization. 'fit loss' is the tail-mean training loss: if it is ~0 while")
    print("# accuracy is well below the reference, the run fit the training set and still")
    print("# did not generalize, and no learning rate is going to fix that by fitting harder.")
    print("# A rate that closes the gap supports the optimization explanation; a grid where")
    print("# every rate falls short rules it out to the resolution of the grid, and leaves")
    print("# the model class itself as the explanation.")

    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n# wrote {opts.json}")


if __name__ == "__main__":
    main()
