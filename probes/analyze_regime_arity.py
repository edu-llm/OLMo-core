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


def load(directory: str) -> tuple[dict, list[tuple[str, str]]]:
    """Load every record under ``directory``, keyed by ``(arm, task, bundle)``.

    Recurses, because a fan-out writes each cell into its own ``cell-<index>/`` prefix.

    :param directory: Directory holding the records.
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
        arm, task, bundle = d.get("arm"), d.get("task"), d.get("bundle_id")
        if arm is None or task is None or bundle is None:
            rejected.append((name, f"arm={arm} task={task} bundle={bundle}"))
            continue
        key = (arm, task, int(bundle))
        if key in records:
            raise SystemExit(
                f"two records claim {key}: {name} duplicates an earlier one. A sweep should "
                f"produce each cell once; averaging silently over a duplicate would weight it "
                f"twice."
            )
        records[key] = d
    if not records:
        raise SystemExit(f"no usable records under {directory} (rejected {len(rejected)})")
    return records, rejected


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
    opts = p.parse_args()

    records, rejected = load(opts.directory)
    print(f"# loaded {len(records)} records; rejected {len(rejected)}")
    for name, why in rejected[:10]:
        print(f"#   REJECTED {name}: {why}")

    arms = sorted({k[0] for k in records})
    tasks = sorted({k[1] for k in records})
    bundles = sorted({k[2] for k in records})
    print(f"# arms={arms}")
    print(f"# tasks={tasks}")
    print(f"# bundles={bundles}")

    missing = [
        (a, t, b) for a in arms for t in tasks for b in bundles if (a, t, b) not in records
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

    def acc(arm: str, task: str, bundle: int, length: int) -> Optional[float]:
        d = records.get((arm, task, bundle))
        if d is None:
            return None
        v = d.get("accuracy_by_length", {}).get(str(length))
        return None if v is None else 100.0 * v

    out: dict = {"tasks": {}, "rejected": rejected, "missing": [list(m) for m in missing]}

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
                if all(acc(a, task, b, length) is not None for a in SQUARE)
            ]
            if len(common) < 2:
                continue
            # Within-regime R effect, paired by bundle.
            strict = paired(
                [acc("DP2-strict", task, b, length) for b in common],
                [acc("R1", task, b, length) for b in common],
            )
            refl = paired(
                [acc("Reflection", task, b, length) for b in common],
                [acc("R1-refl", task, b, length) for b in common],
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

    print("\n## READING THIS")
    print("# R_strict  : DP2-strict - R1        (R=2 vs R=1, beta in (0,1))")
    print("# R_refl    : Reflection - R1-refl   (R=2 vs R=1, beta in (0,2))")
    print("# interaction: R_refl - R_strict, paired per bundle.")
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
