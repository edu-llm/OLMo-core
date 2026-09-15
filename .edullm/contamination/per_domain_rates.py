#!/usr/bin/env python3
"""Per-domain contaminated-item rates.

The curriculum experiment needed contamination profiled along the difficulty
axis. A domain-weighting experiment manipulates the domain mixture directly,
so it needs the profile along the DOMAIN axis instead -- and that axis has far
more spread: matched-document rates run from wiki at ~0.14% down to starcoder
at ~0.002%, a factor of ~70. Two arms that weight domains differently can
therefore differ in contaminated exposure by a large factor by construction,
which is not true of the curriculum arms (at most 1.25x, and exactly 1.0000x
for the three that carry its headline).

Reports, per domain and per benchmark, the number of DISTINCT eval items whose
text was found in that domain, over the assessable item set. Item units, not
tokens, because the endpoint is a per-item macro-average.

Also reports, per domain, the matched-span word rate, so a mixture can be
weighted either by item count or by contaminated text volume.

Reuses the same gates as `aggregate_contamination.py`: no rate is emitted
until every domain has a DONE sentinel, spikes are removed by row identity,
and gate-dropped documents are excluded from the join.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("per_domain_rates")

MIN_WIDTH = 8
FIELD_STEM, FIELD_GOLD = 0, 1


def benchmark_of(label: str) -> str:
    if label.startswith("mmlu_"):
        return "mmlu"
    return label[: -len("_rc_5shot_bpb")].rsplit("_", 1)[0]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--decile-map", required=True, help="for the gate-drop join")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import aggregate_contamination as agg

    hits_dir = Path(args.hits)
    sentinels = agg.check_sentinels(hits_dir)
    benches, stem_ok, gold_ok = agg.load_items(Path(args.items))
    decile = dict(np.load(args.decile_map))
    lookup = agg.build_doc_lookup(decile)
    domain_index = {d: i for i, d in enumerate(agg.DOMAINS)}

    bench_rows: dict[str, list[int]] = defaultdict(list)
    for row, bench in enumerate(benches):
        bench_rows[bench].append(row)
    bench_set = {b: set(rows) for b, rows in bench_rows.items()}

    def denom(bench: str, field: int) -> int:
        ok = stem_ok if field == FIELD_STEM else gold_ok
        return int(sum(1 for r in bench_rows[bench] if ok[r]))

    # domain -> field -> set of matched item rows
    found: dict[str, dict[int, set[int]]] = {
        d: {FIELD_STEM: set(), FIELD_GOLD: set()} for d in agg.DOMAINS
    }
    span_words: dict[str, int] = dict.fromkeys(agg.DOMAINS, 0)
    matched_docs: dict[str, int] = dict.fromkeys(agg.DOMAINS, 0)
    spikes = 0

    for domain in agg.DOMAINS:
        with gzip.open(hits_dir / f"hits_{domain}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["domain"] == "__spike__":
                    spikes += 1
                    continue
                did = domain_index[rec["domain"]]
                table = lookup[did]
                source_doc = int(rec["source_doc"])
                row = int(table[source_doc]) if source_doc < table.shape[0] else -1
                if row < 0:
                    continue  # gate-dropped; not part of the trained corpus
                matched_docs[domain] += 1
                span_words[domain] += int(rec["matched_words"])
                for item_row, field in rec["items"]:
                    found[domain][int(field)].add(int(item_row))

    total_spikes = sum(s["spikes_injected"] for s in sentinels.values())
    if spikes != total_spikes:
        raise RuntimeError(f"removed {spikes} spike rows, expected {total_spikes}")

    bench_names = sorted(bench_rows)
    report: dict[str, object] = {"domains": {}, "benchmarks": bench_names}

    log.info(
        "%-18s %10s %9s %11s %11s %11s",
        "domain",
        "docs",
        "matched",
        "doc rate",
        "span rate",
        "items(stem)",
    )
    for domain in agg.DOMAINS:
        s = sentinels[domain]
        per_bench = {}
        for bench in bench_names:
            entry = {}
            for field, name in ((FIELD_STEM, "stem"), (FIELD_GOLD, "gold")):
                d = denom(bench, field)
                hit = len(found[domain][field] & bench_set[bench])
                entry[f"{name}_found"] = hit
                entry[f"{name}_assessable"] = d
                entry[f"{name}_rate"] = hit / d if d else None
            per_bench[bench] = entry
        macro_stem = [
            e["stem_rate"] for e in per_bench.values() if e["stem_rate"] is not None
        ]
        macro_gold = [
            e["gold_rate"] for e in per_bench.values() if e["gold_rate"] is not None
        ]
        report["domains"][domain] = {  # type: ignore[index]
            "docs_scanned": s["docs_scanned"],
            "words_scanned": s["words_scanned"],
            "matched_documents": matched_docs[domain],
            "matched_document_rate": matched_docs[domain] / max(s["docs_scanned"], 1),
            "matched_span_words": span_words[domain],
            "matched_span_word_rate": span_words[domain] / max(s["words_scanned"], 1),
            "distinct_items_stem": len(found[domain][FIELD_STEM]),
            "distinct_items_gold": len(found[domain][FIELD_GOLD]),
            "per_benchmark": per_bench,
            "macro_stem_rate": sum(macro_stem) / len(macro_stem) if macro_stem else None,
            "macro_gold_rate": sum(macro_gold) / len(macro_gold) if macro_gold else None,
        }
        log.info(
            "%-18s %10d %9d %10.4f%% %11.3e %11d",
            domain,
            s["docs_scanned"],
            matched_docs[domain],
            100 * matched_docs[domain] / max(s["docs_scanned"], 1),
            span_words[domain] / max(s["words_scanned"], 1),
            len(found[domain][FIELD_STEM]),
        )

    rates = [
        b["matched_span_word_rate"]
        for b in report["domains"].values()  # type: ignore[union-attr]
        if b["matched_span_word_rate"] > 0
    ]
    if rates:
        spread = max(rates) / min(rates)
        report["span_rate_spread"] = spread
        log.info("")
        log.info("matched-span word rate spread across domains: %.1fx", spread)
    doc_rates = [
        b["matched_document_rate"]
        for b in report["domains"].values()  # type: ignore[union-attr]
        if b["matched_document_rate"] > 0
    ]
    if doc_rates:
        report["doc_rate_spread"] = max(doc_rates) / min(doc_rates)
        log.info("matched-document rate spread across domains:  %.1fx", report["doc_rate_spread"])

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
