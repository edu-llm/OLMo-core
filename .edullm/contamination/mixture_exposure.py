#!/usr/bin/env python3
"""Per-arm contaminated exposure for a domain-weighting experiment.

The curriculum experiment needed exposure weighted by per-decile visit share.
A domain-weighting experiment manipulates the domain mixture directly, so the
weighting is the mixture itself:

    exposure(arm) = sum over domains of  weight(arm, d) * rate(d)

Two rates are reported because they answer different questions:

  - `matched_span_word_rate` -- the fraction of an arm's training WORDS that
    sit inside a verified eval-item match. This is the contaminated-text
    density the arm actually trains on.
  - `matched_document_rate` -- the fraction of its DOCUMENTS carrying a match.
    Reported for comparability with published figures; it is length-biased and
    on the curriculum corpus it overstated the span rate by 154x.

Why this matters more here than for curriculum. Contamination varies by
domain far more than it varies along a difficulty axis: on the curriculum
corpus the span rate spans 305.8x between wiki and algebraic-stack. A
domain-weighting experiment moves weight across exactly that axis, so its arms
can differ in contaminated exposure by a large factor by construction --
whereas the curriculum arms differ by at most 1.25x, and the three carrying
its headline by exactly 1.0000x.

Note what this does and does not establish. It is the exposure differential,
not an effect on bpb. Whether contamination moved the endpoint is answered by
`arm_effect_on_matched_items.py`, which needs per-item bpb from the runs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("mixture_exposure")

RATE_KEYS = ("matched_span_word_rate", "matched_document_rate")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", required=True, help="per_domain_*.json")
    parser.add_argument("--mixtures", required=True, help="validation_mixtures_*.json")
    parser.add_argument("--baseline", default="natural", help="run_name or tag to index against")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    per_domain = json.loads(Path(args.per_domain).read_text(encoding="utf-8"))["domains"]
    mix = json.loads(Path(args.mixtures).read_text(encoding="utf-8"))
    order = mix["domain_order"]

    missing = [d for d in order if d not in per_domain]
    if missing:
        raise RuntimeError(
            f"no measured rate for {missing}; every domain in the mixture must "
            f"have been scanned or the exposure is computed over a subset "
            f"without saying so"
        )

    rates = {key: {d: float(per_domain[d][key]) for d in order} for key in RATE_KEYS}

    arms: dict[str, dict] = {}
    for m in mix["mixtures"]:
        weights = m.get("weights")
        if not weights:
            continue
        if len(weights) != len(order):
            raise RuntimeError(f"{m['run_name']}: {len(weights)} weights for {len(order)} domains")
        total = sum(weights)
        # Published mixtures are rounded, so some sum to 0.9999 rather than 1.
        # Normalize, because an arm's realized mixture is proportional to its
        # weights -- but record the deviation instead of absorbing it silently,
        # and refuse anything far enough off to be an error rather than
        # rounding.
        if abs(total - 1.0) > 1e-2:
            raise RuntimeError(
                f"{m['run_name']}: weights sum to {total}, too far from 1 to be rounding"
            )
        normalized = [w / total for w in weights]
        entry: dict[str, object] = {
            "id": m["id"],
            "tag": m.get("tag"),
            "weights": dict(zip(order, normalized)),
            "published_weight_sum": total,
        }
        for key in RATE_KEYS:
            entry[key] = sum(w * rates[key][d] for d, w in zip(order, normalized))
        arms[m["run_name"]] = entry

    base = None
    for name, entry in arms.items():
        if name == args.baseline or entry.get("tag") == args.baseline:
            base = entry
            break
    if base is None:
        raise RuntimeError(f"baseline {args.baseline!r} not among {sorted(arms)}")

    for entry in arms.values():
        for key in RATE_KEYS:
            entry[f"{key}_vs_baseline"] = entry[key] / base[key] if base[key] else None

    log.info("baseline: %s", args.baseline)
    log.info(
        "%-18s %-14s %13s %8s %13s %8s",
        "arm",
        "tag",
        "span rate",
        "vs base",
        "doc rate",
        "vs base",
    )
    ordered = sorted(arms.items(), key=lambda kv: kv[1]["matched_span_word_rate"])
    for name, e in ordered:
        log.info(
            "%-18s %-14s %13.4e %7.3fx %13.4e %7.3fx",
            name,
            str(e["tag"])[:14],
            e["matched_span_word_rate"],
            e["matched_span_word_rate_vs_baseline"],
            e["matched_document_rate"],
            e["matched_document_rate_vs_baseline"],
        )

    spans = [e["matched_span_word_rate"] for e in arms.values()]
    spread = max(spans) / min(spans)
    log.info("")
    log.info("span-rate spread across arms: %.2fx", spread)
    log.info(
        "  lowest  %s  highest %s",
        ordered[0][0],
        ordered[-1][0],
    )

    report = {
        "domain_order": order,
        "per_domain_rates": rates,
        "baseline": args.baseline,
        "arms": arms,
        "span_rate_spread_across_arms": spread,
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
