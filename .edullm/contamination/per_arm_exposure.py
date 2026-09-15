#!/usr/bin/env python3
"""Per-arm realized contaminated exposure (v4 section 1's headline deliverable).

A per-decile contamination profile only matters to the extent that some arm
over-exposes the affected deciles. This combines the two:

    exposure(arm) = sum over deciles of  visit_share(arm, d) * matched_rate(d)

where `visit_share` is exact -- per-decile visits follow from the pacing
schedule alone, not from the permutation, so they are seed-invariant -- and
`matched_rate`
is the per-decile matched-item rate from the scan. Reported per arm and as a
ratio against control, per benchmark and macro-averaged over the ten
benchmarks the way the endpoint weights them.

What this is and is not. It is a descriptive statement of how much
contaminated material each arm actually saw. It is NOT a bound on the effect
on bpb -- that apparatus was dropped in v4 for reasons recorded there, and
the effect is measured directly by `arm_effect_on_matched_items.py` once
checkpoints exist. Its use here is to say which arms have any exposure
differential worth conditioning on, and by how much.

Section 1 predicts, and this should confirm, that the three equal-budget arms
come out at exactly 1.000x control while only the quadratic and warmup family
depart from it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("per_arm_exposure")

METRICS = ("mtld", "flesch", "compression_ratio")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="aggregate_contamination output")
    parser.add_argument(
        "--visits", required=True, help="JSON: arm -> per-decile chunk-visit counts"
    )
    parser.add_argument("--metric", default="mtld", choices=METRICS)
    parser.add_argument("--field", default="stem", choices=("stem", "gold"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    visits = json.loads(Path(args.visits).read_text(encoding="utf-8"))

    tab = report["cross_tab"][args.metric]["benchmarks"]
    margin = report["margin"]
    key = f"{args.field}_found"
    denom_key = f"{args.field}_assessable"

    # Per-decile matched-item rate per benchmark, over the assessable set.
    rates: dict[str, list[float]] = {}
    for bench, cells in tab.items():
        denom = margin[bench][denom_key]
        if not denom:
            continue  # NOT ASSESSABLE under this field; contributes nothing
        rates[bench] = [c[key] / denom for c in cells]
    if not rates:
        raise RuntimeError(f"no benchmark is assessable under field {args.field!r}")

    control = visits["control"]
    if len(control) != 10:
        raise RuntimeError("visit table must have ten deciles")

    out: dict[str, object] = {
        "metric": args.metric,
        "field": args.field,
        "benchmarks_contributing": sorted(rates),
        "arms": {},
    }
    log.info("metric=%s field=%s benchmarks=%d", args.metric, args.field, len(rates))
    log.info("%-30s %14s %10s", "arm", "macro exposure", "vs control")

    baseline = None
    for arm, counts in visits.items():
        if len(counts) != 10:
            continue
        # Normalize to visit SHARES so the result reads as an expected
        # matched-item rate per chunk-visit rather than a raw count times a
        # rate, whose absolute magnitude means nothing.
        total = sum(counts)
        share = [c / total for c in counts]
        per_bench = {
            bench: sum(share[d] * rate[d] for d in range(10)) for bench, rate in rates.items()
        }
        macro = sum(per_bench.values()) / len(per_bench)
        if arm == "control":
            baseline = macro
        out["arms"][arm] = {"per_benchmark": per_bench, "macro_exposure": macro}  # type: ignore[index]

    if not baseline:
        raise RuntimeError("control produced zero exposure; cannot form a ratio")
    for arm, block in out["arms"].items():  # type: ignore[union-attr]
        block["ratio_vs_control"] = block["macro_exposure"] / baseline
        log.info(
            "%-30s %14.8f %9.4fx",
            arm,
            block["macro_exposure"],
            block["ratio_vs_control"],
        )

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
