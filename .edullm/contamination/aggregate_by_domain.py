#!/usr/bin/env python3
"""Per-domain contamination rates for an arbitrary corpus.

`aggregate_contamination.py` is specific to the curriculum corpus: it joins
against a per-metric decile map, excludes quality-gate-dropped documents, and
asserts that corpus's exact document count. The domain-weighting and
token-selection corpora have none of that -- no difficulty axis, no gate -- so
this is the same measurement reduced to the domain axis alone.

Why the domain axis is the right one here. A domain-weighting experiment
manipulates the mixture directly, and contamination varies enormously by
domain: on the curriculum corpus the matched-span word rate spans 305.8x
between wiki and algebraic-stack. Two arms with different mixtures therefore
differ in contaminated exposure by a large factor by construction, which is
not true of the curriculum arms (at most 1.25x, and exactly 1.0000x for the
three carrying its headline).

Keeps the gates that still apply: one DONE sentinel per domain, spikes removed
by row identity with the count asserted, and rates over the ASSESSABLE item
set per field so NOT ASSESSABLE items are never scored as clean.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("aggregate_by_domain")

MIN_WIDTH = 8
FIELD_STEM, FIELD_GOLD = 0, 1


def benchmark_of(label: str) -> str:
    if label.startswith("mmlu_"):
        return "mmlu"
    return label[: -len("_rc_5shot_bpb")].rsplit("_", 1)[0]


def load_items(path: Path) -> tuple[list[str], list[bool], list[bool]]:
    benches: list[str] = []
    stem_ok: list[bool] = []
    gold_ok: list[bool] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            benches.append(benchmark_of(rec["label"]))
            stem_ok.append(int(rec["stem_words"]) >= MIN_WIDTH)
            gold_ok.append(int(rec["gold_words"]) >= MIN_WIDTH)
    return benches, stem_ok, gold_ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--domains", required=True, help="comma-separated")
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    hits_dir = Path(args.hits)

    sentinels: dict[str, dict] = {}
    for domain in domains:
        path = hits_dir / f"DONE_{domain}.json"
        if not path.exists():
            raise RuntimeError(
                f"missing DONE sentinel for {domain}: the scan did not complete, "
                f"and aggregating now would under-report every rate"
            )
        s = json.loads(path.read_text(encoding="utf-8"))
        if not s.get("canary_ok"):
            raise RuntimeError(f"{domain}: canary did not pass")
        if s["spikes_recovered"] != s["spikes_injected"]:
            raise RuntimeError(f"{domain}: positive controls not fully recovered")
        sentinels[domain] = s

    benches, stem_ok, gold_ok = load_items(Path(args.items))
    bench_rows: dict[str, list[int]] = defaultdict(list)
    for row, bench in enumerate(benches):
        bench_rows[bench].append(row)
    bench_set = {b: set(rows) for b, rows in bench_rows.items()}
    bench_names = sorted(bench_rows)

    def denom(bench: str, field: int) -> int:
        ok = stem_ok if field == FIELD_STEM else gold_ok
        return sum(1 for r in bench_rows[bench] if ok[r])

    found: dict[str, dict[int, set[int]]] = {
        d: {FIELD_STEM: set(), FIELD_GOLD: set()} for d in domains
    }
    span_words = dict.fromkeys(domains, 0)
    doc_words = dict.fromkeys(domains, 0)
    matched_docs = dict.fromkeys(domains, 0)
    spikes = 0

    for domain in domains:
        with gzip.open(hits_dir / f"hits_{domain}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["domain"] == "__spike__":
                    spikes += 1
                    continue
                matched_docs[domain] += 1
                span_words[domain] += int(rec["matched_words"])
                doc_words[domain] += int(rec["n_words"])
                for item_row, field in rec["items"]:
                    found[domain][int(field)].add(int(item_row))

    expected_spikes = sum(s["spikes_injected"] for s in sentinels.values())
    if spikes != expected_spikes:
        raise RuntimeError(f"removed {spikes} spike rows, expected {expected_spikes}")

    report: dict[str, object] = {
        "corpus": args.corpus_name,
        "domains": {},
        "benchmarks": bench_names,
        "n_items": len(benches),
        "totals": {},
    }

    log.info("corpus: %s", args.corpus_name)
    log.info(
        "%-18s %10s %9s %10s %11s %8s %8s",
        "domain",
        "docs",
        "matched",
        "doc rate",
        "span rate",
        "it(stem)",
        "it(gold)",
    )
    all_stem: set[int] = set()
    all_gold: set[int] = set()
    tot_docs = tot_words = tot_matched = tot_span = 0
    for domain in domains:
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
        macro_stem = [e["stem_rate"] for e in per_bench.values() if e["stem_rate"] is not None]
        macro_gold = [e["gold_rate"] for e in per_bench.values() if e["gold_rate"] is not None]
        report["domains"][domain] = {  # type: ignore[index]
            "docs_scanned": s["docs_scanned"],
            "words_scanned": s["words_scanned"],
            "matched_documents": matched_docs[domain],
            "matched_document_rate": matched_docs[domain] / max(s["docs_scanned"], 1),
            "matched_span_words": span_words[domain],
            "matched_span_word_rate": span_words[domain] / max(s["words_scanned"], 1),
            "tokens_in_matched_docs_rate": doc_words[domain] / max(s["words_scanned"], 1),
            "distinct_items_stem": len(found[domain][FIELD_STEM]),
            "distinct_items_gold": len(found[domain][FIELD_GOLD]),
            "macro_stem_rate": sum(macro_stem) / len(macro_stem) if macro_stem else None,
            "macro_gold_rate": sum(macro_gold) / len(macro_gold) if macro_gold else None,
            "per_benchmark": per_bench,
        }
        all_stem |= found[domain][FIELD_STEM]
        all_gold |= found[domain][FIELD_GOLD]
        tot_docs += s["docs_scanned"]
        tot_words += s["words_scanned"]
        tot_matched += matched_docs[domain]
        tot_span += span_words[domain]
        log.info(
            "%-18s %10d %9d %9.4f%% %11.3e %8d %8d",
            domain,
            s["docs_scanned"],
            matched_docs[domain],
            100 * matched_docs[domain] / max(s["docs_scanned"], 1),
            span_words[domain] / max(s["words_scanned"], 1),
            len(found[domain][FIELD_STEM]),
            len(found[domain][FIELD_GOLD]),
        )

    corpus_bench = {}
    for bench in bench_names:
        entry = {}
        for field, name, allset in (
            (FIELD_STEM, "stem", all_stem),
            (FIELD_GOLD, "gold", all_gold),
        ):
            d = denom(bench, field)
            hit = len(allset & bench_set[bench])
            entry[f"{name}_found"] = hit
            entry[f"{name}_assessable"] = d
            entry[f"{name}_rate"] = hit / d if d else None
        corpus_bench[bench] = entry
    ms = [e["stem_rate"] for e in corpus_bench.values() if e["stem_rate"] is not None]
    mg = [e["gold_rate"] for e in corpus_bench.values() if e["gold_rate"] is not None]
    report["totals"] = {
        "docs_scanned": tot_docs,
        "words_scanned": tot_words,
        "matched_documents": tot_matched,
        "matched_document_rate": tot_matched / max(tot_docs, 1),
        "matched_span_words": tot_span,
        "matched_span_word_rate": tot_span / max(tot_words, 1),
        "distinct_items_stem": len(all_stem),
        "distinct_items_gold": len(all_gold),
        "distinct_items_any": len(all_stem | all_gold),
        "macro_stem_rate": sum(ms) / len(ms) if ms else None,
        "macro_gold_rate": sum(mg) / len(mg) if mg else None,
        "per_benchmark": corpus_bench,
    }

    span_rates = [
        b["matched_span_word_rate"]
        for b in report["domains"].values()  # type: ignore[union-attr]
        if b["matched_span_word_rate"] > 0
    ]
    if len(span_rates) > 1:
        report["span_rate_spread"] = max(span_rates) / min(span_rates)

    log.info("")
    log.info(
        "TOTAL  docs=%d  words=%.2fB  matched_docs=%d (%.4f%%)  span_rate=%.3e",
        tot_docs,
        tot_words / 1e9,
        tot_matched,
        100 * tot_matched / max(tot_docs, 1),
        tot_span / max(tot_words, 1),
    )
    log.info(
        "distinct items: stem=%d gold=%d any=%d of %d",
        len(all_stem),
        len(all_gold),
        len(all_stem | all_gold),
        len(benches),
    )
    if ms:
        log.info("macro stem rate %.4f%%   macro gold rate %.4f%%", 100 * sum(ms) / len(ms),
                 100 * sum(mg) / len(mg) if mg else float("nan"))
    if "span_rate_spread" in report:
        log.info("span-rate spread across domains: %.1fx", report["span_rate_spread"])

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
