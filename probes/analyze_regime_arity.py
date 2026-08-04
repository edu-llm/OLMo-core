"""Aggregate the (beta regime x R) x task sweep and test the interaction.

Reads the records a fan-out sweep leaves under one prefix and reports, per task and
evaluation length, the R effect within each beta regime and the **interaction** between
them. The interaction is the quantity the sweep exists to measure: whether the benefit of
more Householder factors depends on the beta range, and therefore on whether a reflection
(determinant -1) is reachable at all.

Usage::

    python probes/analyze_regime_arity.py <dir-holding-records> [--json out.json]

Why this exists rather than ``docs/dp2-kda/evidence/review-sigma/analyze_sigma.py``
----------------------------------------------------------------------------------
That script parses the arm and bundle out of the *filename* (``<prefix>_<arm>_b<id>.json``)
and carries no task axis at all, because every record it was written for came from one task.
A two-task sweep collides in its ``recs[(arm, bundle)]`` dictionary: the S5 record silently
overwrites the A5 record for the same arm and bundle, and nothing reports it.

This module keys on the fields *inside* each record instead, so the filenames a fan-out
happens to produce do not matter and a missing task axis cannot alias two cells together.

The provenance and completion filters are kept identical to ``analyze_sigma.py`` on purpose:
a record whose ``outcome`` is not ``completed``, or whose ``probe_source_revision`` is absent
or ``"unknown"``, is rejected rather than averaged in. A results file with no traceable source
looks like data and is not.

:raises SystemExit: If the directory holds no usable records.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Optional

#: The four cells of the square, as ``arm -> (R, beta regime)``. Spelled out rather than read
#: from ``train_probe.ARMS`` so this module can run anywhere the records were copied to,
#: without importing the harness or its torch dependency.
SQUARE = {
    "R1": (1, "strict"),
    "DP2-strict": (2, "strict"),
    "R1-refl": (1, "reflection"),
    "Reflection": (2, "reflection"),
}

#: The same square with each R=1 cell replaced by its parameter-matched control. Selected by
#: ``--square matched``. The R=2 arms are shared with :data:`SQUARE` deliberately: the control
#: is defined as "R=1 carrying the R=2 arm's parameter count", so the arm it is matched to has
#: to be the same object in both squares or the match means nothing.
SQUARE_MATCHED = {
    "R1-P": (1, "strict"),
    "DP2-strict": (2, "strict"),
    "R1-refl-P": (1, "reflection"),
    "Reflection": (2, "reflection"),
}

SQUARES = {"raw": SQUARE, "matched": SQUARE_MATCHED}

#: One-sided 95% / two-sided 90% Student t, by degrees of freedom. Only the entries a real
#: sweep can produce are tabulated; anything absent falls through to :func:`tcrit95`'s guard
#: rather than a silently-too-small value. ``analyze_sigma.py``'s fallback of
#: ``1.645 + 1/df`` is deliberately not copied: it under-covers for several df in range.
T_CRIT_95 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
    9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
    16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 21: 1.721, 22: 1.717,
    23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699,
    30: 1.697,
}


def tcrit95(df: int) -> float:
    """One-sided 95% critical value of Student's t.

    :param df: Degrees of freedom.
    :returns: The critical value.
    :raises SystemExit: If ``df`` is outside the tabulated range, rather than returning an
        approximation that silently under-covers.
    """
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df > 30:
        return 1.645  # asymptotic; conservative direction for df>30
    raise SystemExit(f"no tabulated t critical value for df={df}; refusing to approximate")


def mean(v: list[float]) -> float:
    """Arithmetic mean."""
    return sum(v) / len(v)


def sd(v: list[float]) -> float:
    """Sample standard deviation (n-1). ``nan`` for fewer than two values."""
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def load(directory: str, lr: Optional[float] = None) -> tuple[dict, list[tuple[str, str]]]:
    """Load every record under ``directory``, keyed by ``(arm, task, bundle)``.

    Recurses, because a fan-out writes each cell into its own ``cell-<index>/`` prefix.

    ``lr`` selects a single learning rate when the directory holds a sweep over several. It is
    required in that case rather than defaulted, because the alternative -- picking one, or
    pooling them -- would report an average over a treatment as though it were a measurement
    of one condition. Records predating the ``lr`` field are treated as an unnamed single rate
    and pass through untouched.

    :param directory: Directory holding the records.
    :param lr: Keep only records at this learning rate.
    :returns: ``(records, rejected)``.
    :raises SystemExit: If a key appears twice, which would mean two cells wrote the same
        (arm, task, bundle) and one is about to be discarded unnoticed.
    """
    records: dict[tuple[str, str, int], dict] = {}
    rejected: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        with open(path) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError as exc:
                rejected.append((os.path.basename(path), f"unparseable: {exc}"))
                continue
        name = os.path.relpath(path, directory)
        if d.get("outcome") != "completed":
            rejected.append((name, f"outcome={d.get('outcome')}"))
            continue
        if d.get("probe_source_revision") in (None, "unknown"):
            rejected.append((name, f"prov={d.get('probe_source_revision')}"))
            continue
        if lr is not None and d.get("lr") is not None and float(d["lr"]) != lr:
            rejected.append((name, f"lr={d['lr']} != {lr}"))
            continue
        arm, task, bundle = d.get("arm"), d.get("task"), d.get("bundle_id")
        if arm is None or task is None or bundle is None:
            rejected.append((name, f"arm={arm} task={task} bundle={bundle}"))
            continue
        key = (arm, task, int(bundle))
        if key in records:
            # Two records for one cell. If they differ in learning rate this is a sweep the
            # caller forgot to slice, and saying so is more useful than the generic message:
            # the fix is a flag, not a re-run.
            previous = records[key].get("lr")
            current = d.get("lr")
            if previous != current:
                raise SystemExit(
                    f"two records claim {key} at different learning rates ({previous} and "
                    f"{current}). This directory holds an LR sweep; pass --lr to select one. "
                    f"Pooling them would average over a treatment."
                )
            raise SystemExit(
                f"two records claim {key}: {name} duplicates an earlier one. A sweep should "
                f"produce each cell once; averaging silently over a duplicate would weight it "
                f"twice."
            )
        records[key] = d
    if not records:
        raise SystemExit(f"no usable records under {directory} (rejected {len(rejected)})")
    return records, rejected


def denest(acc_by_length: dict[int, float], lengths: list[int]) -> dict[tuple[int, int], float]:
    """Convert prefix-averaged accuracies into disjoint position bands.

    THE REPORTED METRIC IS AN AVERAGE OVER POSITIONS 1..L, NOT AN ACCURACY *AT* L. The group
    tasks mask nothing (``train_probe.py:427``), so every evaluation length's number already
    contains every shorter length's positions. The five lengths are nested, and reading them
    as five independent points overstates long-length performance by exactly the weight of the
    short prefix that is carried along -- which at L=512 is most of it.

    A model that is perfect below the training cutoff and at chance above it therefore traces
    a smooth decay with no long-range ability whatsoever. That curve is arithmetic, and
    mistaking it for extrapolation is the error this function exists to prevent.

    Band accuracy follows from the definition of the average::

        A(L) * L = A(L_prev) * L_prev + band * (L - L_prev)

    :param acc_by_length: Prefix-averaged accuracy at each evaluation length.
    :param lengths: Evaluation lengths, ascending. The first is its own band's upper edge.
    :returns: Mapping ``(lo, hi) -> accuracy over positions lo..hi``, ``lo`` inclusive.
    """
    bands: dict[tuple[int, int], float] = {}
    previous_length = 0
    previous_mass = 0.0
    for length in lengths:
        if length not in acc_by_length:
            continue
        mass = acc_by_length[length] * length
        width = length - previous_length
        if width > 0:
            bands[(previous_length + 1, length)] = (mass - previous_mass) / width
        previous_length, previous_mass = length, mass
    return bands


def paired(a: list[float], b: list[float]) -> dict:
    """Paired difference statistics for ``a - b``.

    :param a: First arm's per-bundle values.
    :param b: Second arm's per-bundle values, in the same bundle order.
    :returns: mean, sd, se, t, dz, one-sided 95% lower bound, and n.
    """
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    m, s = mean(diffs), sd(diffs)
    se = s / math.sqrt(n) if n > 1 and s == s else float("nan")
    return {
        "n": n,
        "mean": m,
        "sd": s,
        "se": se,
        "t": m / se if se and se == se and se != 0 else float("nan"),
        "dz": m / s if s == s and s != 0 else float("nan"),
        "l95": m - tcrit95(n - 1) * se if se == se else float("nan"),
        "diffs": diffs,
    }


def main() -> None:
    """Aggregate the sweep and print the per-task interaction table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", help="Directory holding the sweep's records.")
    p.add_argument("--json", default=None, help="Also write the full result here.")
    p.add_argument(
        "--square",
        default="raw",
        choices=sorted(SQUARES),
        help=(
            "'raw' uses R1/R1-refl as the R=1 cells; 'matched' uses the parameter-matched "
            "controls R1-P/R1-refl-P instead."
        ),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Select one learning rate when the directory holds an LR sweep.",
    )
    opts = p.parse_args()
    square = SQUARES[opts.square]

    records, rejected = load(opts.directory, lr=opts.lr)
    print(f"# loaded {len(records)} records; rejected {len(rejected)}")
    for name, why in rejected[:10]:
        print(f"#   REJECTED {name}: {why}")
    print(f"# square={opts.square} ({', '.join(sorted(square))})")
    if opts.lr is not None:
        print(f"# lr={opts.lr}")

    arms = sorted({k[0] for k in records})
    tasks = sorted({k[1] for k in records})
    bundles = sorted({k[2] for k in records})
    print(f"# arms={arms}")
    print(f"# tasks={tasks}")
    print(f"# bundles={bundles}")

    # Completeness is judged against the selected square, not against every arm present. A
    # directory may legitimately hold both squares -- the matched controls are pointless
    # without the R=2 arms they are matched to -- and cross-producting all arms would report
    # the arms of the *other* square as missing cells of this one.
    missing = [
        (a, t, b) for a in square for t in tasks for b in bundles if (a, t, b) not in records
    ]
    if missing:
        print(f"# INCOMPLETE: {len(missing)} cells absent, e.g. {missing[:5]}")

    # Integrity. The claim to check is that the arms share one evaluation bank *within* each
    # bundle -- that is what makes the contrast paired and what lets n=6 resolve effects far
    # smaller than the between-seed spread. Counting distinct banks across the whole task is
    # the wrong test and reads as a failure on a healthy sweep: the bank is seeded from the
    # bundle's eval stream, so a 6-bundle sweep is *supposed* to have 6 distinct banks.
    print("\n## INTEGRITY")
    for task in tasks:
        revs = {d.get("probe_source_revision") for (a, t, b), d in records.items() if t == task}
        unpaired = []
        for bundle in bundles:
            banks = {
                d["eval_bank_sha256"]
                for (a, t, b), d in records.items()
                if t == task and b == bundle and "eval_bank_sha256" in d
            }
            if len(banks) > 1:
                unpaired.append(bundle)
        collisions = sum(
            d.get("eval_collisions") or 0 for (a, t, b), d in records.items() if t == task
        )
        status = "PAIRED" if not unpaired else f"NOT PAIRED in bundles {unpaired}"
        print(
            f"{task:10s} {status}  train/eval_collisions={collisions}  revisions={sorted(revs)}"
        )
        if unpaired:
            print(
                "#   A bundle whose arms saw different evaluation instances is not a paired "
                "comparison; its contrast carries evaluation-sampling noise the design was "
                "built to cancel."
            )

    lengths = sorted(
        {int(k) for d in records.values() for k in d.get("accuracy_by_length", {})}
    )

    # The two R=1 cells of the selected square, and the R=2 arm each is contrasted against.
    r1_strict = next(a for a, (r, g) in square.items() if r == 1 and g == "strict")
    r2_strict = next(a for a, (r, g) in square.items() if r == 2 and g == "strict")
    r1_refl = next(a for a, (r, g) in square.items() if r == 1 and g == "reflection")
    r2_refl = next(a for a, (r, g) in square.items() if r == 2 and g == "reflection")

    def acc(arm: str, task: str, bundle: int, length: int) -> Optional[float]:
        d = records.get((arm, task, bundle))
        if d is None:
            return None
        v = d.get("accuracy_by_length", {}).get(str(length))
        return None if v is None else 100.0 * v

    def band_acc(arm: str, task: str, bundle: int, band: tuple[int, int]) -> Optional[float]:
        d = records.get((arm, task, bundle))
        if d is None:
            return None
        raw = {int(k): v for k, v in d.get("accuracy_by_length", {}).items()}
        bands = denest(raw, sorted(raw))
        v = bands.get(band)
        return None if v is None else 100.0 * v

    out: dict = {
        "square": opts.square,
        "lr": opts.lr,
        "tasks": {},
        "rejected": rejected,
        "missing": [list(m) for m in missing],
    }

    # Capacity ledger. When the matched square is selected, the whole point is that the two
    # arms of each contrast have the same parameter count -- so report whether they actually
    # do rather than trusting that the arm name implies it.
    print("\n## CAPACITY")
    for arm in sorted(square):
        counts = {
            d.get("param_ledger", {}).get("non_embedding")
            for (a, t, b), d in records.items()
            if a == arm
        }
        counts.discard(None)
        print(f"{arm:12s} non_embedding={sorted(counts) if len(counts) != 1 else counts.pop()}")
    for label, one, two in (("strict", r1_strict, r2_strict), ("reflection", r1_refl, r2_refl)):
        def one_count(arm: str) -> Optional[int]:
            for (a, t, b), d in records.items():
                if a == arm:
                    return d.get("param_ledger", {}).get("non_embedding")
            return None

        lo, hi = one_count(one), one_count(two)
        if lo is not None and hi is not None:
            delta = 100.0 * (hi - lo) / lo
            verdict = "MATCHED" if abs(delta) <= 0.5 else "NOT matched"
            print(f"#   {label:10s} {one} vs {two}: {delta:+.2f}% -> {verdict}")

    for task in tasks:
        print(f"\n## {task}: R effect within each beta regime, and the interaction")
        print(
            f"{'L':>5} {'n':>3} {'R_strict':>9} {'R_refl':>9} {'interaction':>12} "
            f"{'se':>7} {'t':>7} {'L95':>8}"
        )
        out["tasks"][task] = {}
        for length in lengths:
            common = [
                b
                for b in bundles
                if all(acc(a, task, b, length) is not None for a in square)
            ]
            if len(common) < 2:
                continue
            # Within-regime R effect, paired by bundle.
            strict = paired(
                [acc(r2_strict, task, b, length) for b in common],
                [acc(r1_strict, task, b, length) for b in common],
            )
            refl = paired(
                [acc(r2_refl, task, b, length) for b in common],
                [acc(r1_refl, task, b, length) for b in common],
            )
            # The interaction, paired at the bundle level. Differencing per bundle first --
            # rather than subtracting the two means and combining their SEs -- keeps the
            # pairing that the shared eval bank buys. Treating the two contrasts as
            # independent would inflate the standard error and understate the effect.
            inter = paired(refl["diffs"], strict["diffs"])
            print(
                f"{length:>5} {len(common):>3} {strict['mean']:>+9.2f} {refl['mean']:>+9.2f} "
                f"{inter['mean']:>+12.2f} {inter['se']:>7.2f} {inter['t']:>7.2f} "
                f"{inter['l95']:>+8.2f}"
            )
            out["tasks"][task][length] = {
                "bundles": common,
                "r_effect_strict": strict,
                "r_effect_reflection": refl,
                "interaction": inter,
            }

        # The same contrast on disjoint position bands. This is the primary table: the one
        # above shares positions across every row, so its rows cannot be read as five
        # independent measurements and its long-length entries are dominated by the short
        # prefix they contain. Both are printed so the size of the artifact is visible.
        print(f"\n## {task}: DE-NESTED into disjoint position bands (primary)")
        print(
            f"{'band':>12} {'n':>3} {'R_strict':>9} {'R_refl':>9} {'interaction':>12} "
            f"{'se':>7} {'t':>7} {'L95':>8}"
        )
        all_bands: list[tuple[int, int]] = []
        previous = 0
        for length in lengths:
            all_bands.append((previous + 1, length))
            previous = length
        out["tasks"][task]["bands"] = {}
        for band in all_bands:
            common = [
                b for b in bundles if all(band_acc(a, task, b, band) is not None for a in square)
            ]
            if len(common) < 2:
                continue
            strict = paired(
                [band_acc(r2_strict, task, b, band) for b in common],
                [band_acc(r1_strict, task, b, band) for b in common],
            )
            refl = paired(
                [band_acc(r2_refl, task, b, band) for b in common],
                [band_acc(r1_refl, task, b, band) for b in common],
            )
            inter = paired(refl["diffs"], strict["diffs"])
            label = f"{band[0]}-{band[1]}"
            print(
                f"{label:>12} {len(common):>3} {strict['mean']:>+9.2f} {refl['mean']:>+9.2f} "
                f"{inter['mean']:>+12.2f} {inter['se']:>7.2f} {inter['t']:>7.2f} "
                f"{inter['l95']:>+8.2f}"
            )
            out["tasks"][task]["bands"][label] = {
                "bundles": common,
                "r_effect_strict": strict,
                "r_effect_reflection": refl,
                "interaction": inter,
            }

    print("\n## READING THIS")
    print(f"# R_strict  : {r2_strict} - {r1_strict}  (R=2 vs R=1, beta in (0,1))")
    print(f"# R_refl    : {r2_refl} - {r1_refl}  (R=2 vs R=1, beta in (0,2))")
    print("# interaction: R_refl - R_strict, paired per bundle.")
    print("# The banded table is primary. The prefix table's rows are nested -- each length's")
    print("# number already contains every shorter length's positions -- so a model that is")
    print("# perfect below the training cutoff and at chance above traces a smooth decay")
    print("# there while having no long-range ability at all.")
    print("# A positive interaction means extra factors buy more when a reflection is")
    print("# reachable. Comparing it across a5_words and s5_words separates the parity")
    print("# obstruction from group order: both groups are non-solvable, but only S5's")
    print("# generators include an odd permutation.")

    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n# wrote {opts.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
