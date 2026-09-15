#!/usr/bin/env python3
"""Does an arm's advantage concentrate on leaked items? (v4 section 5.3)

This is the decision statistic. Everything upstream exists to produce a
trustworthy matched-item set; this asks whether it matters.

For an arm a, a benchmark b and a matched subset S:

    delta_i     = bpb_i(arm a) - bpb_i(control)      paired, same seed
    D_matched   = mean of delta_i over i in S
    D_unmatched = mean of delta_i over i not in S
    T(a, b, S)  = D_matched - D_unmatched

Arm and control at a matched seed share initialization and RNG stream, so
delta_i is a paired difference and far less noisy than raw bpb. If the arm
effect is the same size on matched and unmatched items, contamination is not
driving it -- whatever the absolute contamination level turns out to be.

Two things this deliberately does NOT do:

  - It does not bound contamination's effect in advance. That apparatus (a
    1-bit ceiling, total-variation distances over exposure distributions,
    closure thresholds) was dropped: the runs are happening, so the effect is
    measured rather than bounded, and the bound's primary form was invalid
    anyway -- a ceiling bounds |d_arm - d_ctrl| <= 1, not d_arm = d_ctrl.
  - It does not correct for precision. Under this test a false positive in
    the matched set DILUTES T toward zero, so junk is the conservative
    direction. Recall is what matters, which is why the index is loose.

Subsets are nested -- S_gold within S_gold_or_stem within S_any -- which buys
a free validity check: if the matched set is real leakage, T should be
strongest on S_gold and weaken monotonically as the subset loosens. It also
supports the decile-conditioned form, which is the sharper test for arms that
over-expose particular deciles.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("arm_effect")

FIELD_STEM, FIELD_GOLD = 0, 1
SUBSETS = ("S_gold", "S_gold_or_stem", "S_any")


def read_per_item(path: str | Path) -> dict[tuple[str, int], float]:
    """(label, doc_id) -> gold bpb, from a per-item capture file."""
    out: dict[tuple[str, int], float] = {}
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["label"], int(rec["doc_id"]))
            if key in out:
                raise RuntimeError(f"duplicate per-item row for {key}")
            out[key] = float(rec["gold_bpb"])
    if not out:
        raise RuntimeError(f"{path}: no per-item rows")
    return out


def subset_membership(
    matched: dict[int, set[int]], n_items: int
) -> dict[str, set[int]]:
    """Nested matched subsets from provenance-tagged hits.

    `matched` maps item row -> set of field ids it was matched on.
    """
    gold = {row for row, fields in matched.items() if FIELD_GOLD in fields}
    stem = {row for row, fields in matched.items() if FIELD_STEM in fields}
    any_ = set(matched)
    subsets = {"S_gold": gold, "S_gold_or_stem": gold | stem, "S_any": any_}
    # Nesting is what makes the monotonicity check meaningful, so assert it.
    if not subsets["S_gold"] <= subsets["S_gold_or_stem"] <= subsets["S_any"]:
        raise RuntimeError("matched subsets are not nested")
    if any(row >= n_items or row < 0 for row in any_):
        raise RuntimeError("a matched item row is outside the item inventory")
    return subsets


def paired_deltas(
    arm: dict[tuple[str, int], float], control: dict[tuple[str, int], float]
) -> dict[tuple[str, int], float]:
    """arm minus control, per item, refusing a partial overlap."""
    if set(arm) != set(control):
        missing = len(set(control) - set(arm))
        extra = len(set(arm) - set(control))
        raise RuntimeError(
            f"arm and control scored different item sets "
            f"({missing} missing from arm, {extra} extra); a paired "
            f"difference is not defined"
        )
    return {key: arm[key] - control[key] for key in arm}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _inverse_variance_macro(usable: list[tuple[str, dict]]) -> float | None:
    """Macro over benchmarks, weighted by effective sample size.

    The variance of a difference of means goes as
    sigma^2 * (1/n_matched + 1/n_clean), so the effective sample size
    n_eff = 1 / (1/n_matched + 1/n_clean) is the inverse-variance weight up
    to the common sigma^2. Weighting by it is the statistically correct macro
    and needs no minimum-count threshold: a benchmark with one matched item
    earns a weight near 1 against another's several hundred, rather than an
    equal vote.
    """
    num = 0.0
    den = 0.0
    for _bench, stats in usable:
        n_m = int(stats["n_matched"])
        n_c = int(stats["n_clean_baseline"])
        if n_m <= 0 or n_c <= 0:
            continue
        weight = 1.0 / (1.0 / n_m + 1.0 / n_c)
        num += weight * float(stats["T"])
        den += weight
    return num / den if den else None


def compute_T(
    deltas: dict[tuple[str, int], float],
    rows_by_key: dict[tuple[str, int], int],
    subset: set[int],
    keys: list[tuple[str, int]],
    clean: set[int],
) -> dict[str, object]:
    """T for one subset, always against the SAME clean baseline.

    `clean` is the set of items matched by nothing, and every subset is
    compared against it. Using each subset's own complement instead puts
    contaminated items into the comparison group: with a synthetic -0.05
    injected on all 1,126 matched items of the real scan, the complement of
    S_gold still held the other 942, and T came out at +0.025 -- the wrong
    SIGN, not merely attenuated. That also destroys the nested-subset
    monotonicity check, which cannot weaken monotonically when each
    subset's baseline is polluted by the subsets above it.
    """
    matched = [deltas[k] for k in keys if rows_by_key[k] in subset]
    baseline = [deltas[k] for k in keys if rows_by_key[k] in clean]
    d_m, d_u = _mean(matched), _mean(baseline)
    return {
        "n_matched": len(matched),
        "n_clean_baseline": len(baseline),
        "D_matched": d_m,
        "D_clean": d_u,
        "T": None if d_m is None or d_u is None else d_m - d_u,
    }


def bootstrap_ci(
    deltas: dict[tuple[str, int], float],
    rows_by_key: dict[tuple[str, int], int],
    subset: set[int],
    keys: list[tuple[str, int]],
    clean: set[int],
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile CI for T, resampling items -- the unit of the statistic."""
    matched = [deltas[k] for k in keys if rows_by_key[k] in subset]
    unmatched = [deltas[k] for k in keys if rows_by_key[k] in clean]
    if not matched or not unmatched:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(draws):
        m = sum(matched[rng.randrange(len(matched))] for _ in matched) / len(matched)
        u = sum(unmatched[rng.randrange(len(unmatched))] for _ in unmatched) / len(
            unmatched
        )
        stats.append(m - u)
    stats.sort()
    lo = stats[max(0, int(math.floor(0.025 * len(stats))))]
    hi = stats[min(len(stats) - 1, int(math.ceil(0.975 * len(stats))) - 1)]
    return lo, hi


def benchmark_of(label: str) -> str:
    if label.startswith("mmlu_"):
        return "mmlu"
    return label[: -len("_rc_5shot_bpb")].rsplit("_", 1)[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-per-item", required=True)
    parser.add_argument("--control-per-item", required=True)
    parser.add_argument("--items", required=True, help="eval_items.jsonl.gz")
    parser.add_argument("--report", required=True, help="aggregate_contamination output")
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    arm = read_per_item(args.arm_per_item)
    control = read_per_item(args.control_per_item)
    deltas = paired_deltas(arm, control)

    # Item row order is the dump's line order, which is what hit rows index.
    rows_by_key: dict[tuple[str, int], int] = {}
    with gzip.open(args.items, "rt", encoding="utf-8") as fh:
        for row, line in enumerate(fh):
            rec = json.loads(line)
            rows_by_key[(rec["label"], int(rec["doc_id"]))] = row
    n_items = len(rows_by_key)

    unknown = [k for k in deltas if k not in rows_by_key]
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} scored items are absent from the item dump "
            f"(first: {unknown[0]}); the join is wrong and T would be "
            f"computed over the wrong subset"
        )

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if "matched_rows" not in report:
        raise RuntimeError(
            "the aggregate report carries no `matched_rows`, so there is no "
            "item-level matched set to condition on; rerun "
            "aggregate_contamination.py"
        )
    matched: dict[int, set[int]] = defaultdict(set)
    for row_str, fields in report["matched_rows"].items():
        matched[int(row_str)] = {int(f) for f in fields}

    subsets = subset_membership(matched, n_items)
    # Items matched by nothing. Every subset is scored against this one
    # baseline rather than its own complement (see compute_T).
    all_matched = subsets["S_any"]
    clean = {rows_by_key[k] for k in deltas if rows_by_key[k] not in all_matched}
    keys = sorted(deltas)
    by_bench: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        by_bench[benchmark_of(key[0])].append(key)

    out: dict[str, object] = {"arm": args.arm_name, "n_items_scored": len(keys)}
    for name in SUBSETS:
        subset = subsets[name]
        per_bench = {}
        for bench, bench_keys in sorted(by_bench.items()):
            stats = compute_T(deltas, rows_by_key, subset, bench_keys, clean)
            ci = bootstrap_ci(
                deltas, rows_by_key, subset, bench_keys, clean, args.bootstrap_draws
            )
            stats["ci95"] = list(ci) if ci else None
            per_bench[bench] = stats
        usable = [
            (b, s) for b, s in per_bench.items() if s["T"] is not None
        ]
        macro_flat = _mean([s["T"] for _b, s in usable])
        macro_weighted = _inverse_variance_macro(usable)
        out[name] = {
            "subset_size": len(subset),
            "clean_baseline_size": len(clean),
            "per_benchmark": per_bench,
            # PRIMARY. Weighted by effective sample size, which is
            # inverse-variance weighting for a difference of means.
            "macro_T": macro_weighted,
            # Reported for comparison only. Equal weighting is right for the
            # ENDPOINT, where every benchmark has thousands of items, and
            # wrong for T, whose matched subset per benchmark ranges from 0
            # to several hundred. On a synthetic -0.05 injected across the
            # real scan's 1,126 matched items, socialiqa contributed a single
            # item at T=+0.69 -- pure noise at full weight -- and flipped the
            # flat macro to +0.024 while the weighted one read -0.059.
            "macro_T_equal_weight": macro_flat,
            "benchmarks_contributing": len(usable),
            "smallest_matched_count": min((s["n_matched"] for _b, s in usable), default=0),
        }
        log.info(
            "%-16s |S|=%6d  macro T=%s (equal-weight %s)  over %d/%d benchmarks, min n=%d",
            name,
            len(subset),
            "n/a" if macro_weighted is None else f"{macro_weighted:+.6f}",
            "n/a" if macro_flat is None else f"{macro_flat:+.6f}",
            len(usable),
            len(per_bench),
            min((s["n_matched"] for _b, s in usable), default=0),
        )

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
