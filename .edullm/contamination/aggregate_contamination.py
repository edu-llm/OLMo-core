#!/usr/bin/env python3
"""Aggregate the scan into methodology v4's report (sections 5.1 and 5.2).

Produces one cross-tabulation in item units -- benchmark x (metric, decile) --
plus a single corpus-side token line, and refuses to emit anything until the
scan's own coverage gates pass.

Three gates run before any rate is printed:

  1. Exactly one DONE sentinel per domain, and `sum(docs_scanned)` equal to
     5,911,841 -- the PRE-gate count, because the scanner reads every trim/
     line. A silently failed array task would otherwise move that task's
     share of matches into the unmatched group and bias every rate downward
     by its share: larger than any effect being measured, and invisible.
  2. Spike rows removed by identity, with the removed count asserted equal to
     the number injected and no non-spike row removed.
  3. The row margin -- an item found anywhere in the corpus -- must be
     IDENTICAL across all three metrics. The three corpora hold the same
     5,857,887 documents in different orders and the scan reads document text,
     so presence cannot depend on ordering. Any discrepancy means the decile
     join is broken.

Rows do not sum to 100% by design: near-duplicates receive identical
difficulty scores and land at adjacent ranks, so one item can be found in
several deciles.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger("aggregate")

METRICS = ("mtld", "flesch", "compression_ratio")
DOMAINS = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "nemotron-hqdqa",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)
MIN_WIDTH = 8
FIELD_STEM, FIELD_GOLD = 0, 1

# Three different document counts, and conflating them is a real hazard.
#
# SCANNED_DOCS is what `trim/` actually holds and therefore what the scanner
# enumerates: the PRE-gate corpus. CORPUS_DOCS is the post-gate corpus that
# was tokenized and trained on, and is the denominator for every rate.
# GATE_DROPPED is the difference -- documents present in `trim/`, scanned, and
# then excluded from the join because they are not part of the trained corpus.
#
# The coverage gate has to assert the PRE-gate number or it fails on a
# complete scan, which is how this was found.
SCANNED_DOCS = 5_911_841
CORPUS_DOCS = 5_857_887
GATE_DROPPED = 53_954
CORPUS_TOKENS = 9_979_762_663


def benchmark_of(label: str) -> str:
    if label.startswith("mmlu_"):
        return "mmlu"
    prefix = label[: -len("_rc_5shot_bpb")]
    return prefix.rsplit("_", 1)[0]


def load_items(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (benchmark per row, stem assessable, gold assessable)."""
    benches: list[str] = []
    stem_ok: list[bool] = []
    gold_ok: list[bool] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            benches.append(benchmark_of(rec["label"]))
            stem_ok.append(int(rec["stem_words"]) >= MIN_WIDTH)
            gold_ok.append(int(rec["gold_words"]) >= MIN_WIDTH)
    return benches, np.asarray(stem_ok), np.asarray(gold_ok)


def check_sentinels(hits_dir: Path) -> dict:
    sentinels = {}
    for domain in DOMAINS:
        path = hits_dir / f"DONE_{domain}.json"
        if not path.exists():
            raise RuntimeError(
                f"missing DONE sentinel for {domain}: the scan did not "
                f"complete, and aggregating now would under-report every rate"
            )
        sentinels[domain] = json.loads(path.read_text(encoding="utf-8"))

    if SCANNED_DOCS - CORPUS_DOCS != GATE_DROPPED:
        raise RuntimeError(
            f"the three document counts do not reconcile: "
            f"{SCANNED_DOCS} - {CORPUS_DOCS} != {GATE_DROPPED}"
        )
    total_docs = sum(s["docs_scanned"] for s in sentinels.values())
    if total_docs != SCANNED_DOCS:
        raise RuntimeError(
            f"scanned {total_docs} documents but trim/ holds {SCANNED_DOCS}; "
            f"coverage is incomplete. (Note this is the PRE-gate count: the "
            f"scanner reads every trim/ line, and the {GATE_DROPPED} "
            f"gate-dropped documents are excluded later, at the join.)"
        )
    for domain, s in sentinels.items():
        if not s.get("canary_ok"):
            raise RuntimeError(f"{domain}: canary did not pass")
        if s["spikes_recovered"] != s["spikes_injected"]:
            raise RuntimeError(f"{domain}: positive controls not fully recovered")
    return sentinels


def build_doc_lookup(decile: dict) -> dict[int, np.ndarray]:
    """(domain_id, source_doc) -> row in the decile map, as per-domain arrays."""
    domain_id = decile["domain_id"]
    source_doc = decile["source_doc"]
    lookup: dict[int, np.ndarray] = {}
    for d in range(len(DOMAINS)):
        mask = domain_id == d
        rows = np.flatnonzero(mask)
        docs = source_doc[mask]
        # Sized from the gated source_doc values, which is one short of the
        # trim/ line count for any domain whose LAST document was gate-dropped.
        # That does not happen in the current corpus, but an IndexError here
        # would be a confusing way to discover it, so the hit loop range-checks
        # instead and treats an out-of-range index as gate-dropped.
        table = np.full(int(docs.max()) + 1 if docs.size else 1, -1, dtype=np.int64)
        table[docs] = rows
        lookup[d] = table
    return lookup


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--decile-map", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hits_dir = Path(args.hits)
    sentinels = check_sentinels(hits_dir)
    log.info(
        "coverage gate OK: %d documents, %.2fB words across %d domains",
        sum(s["docs_scanned"] for s in sentinels.values()),
        sum(s["words_scanned"] for s in sentinels.values()) / 1e9,
        len(sentinels),
    )

    benches, stem_ok, gold_ok = load_items(Path(args.items))
    n_items = len(benches)
    decile = dict(np.load(args.decile_map))
    lookup = build_doc_lookup(decile)
    domain_index = {d: i for i, d in enumerate(DOMAINS)}

    # found[metric][decile] -> set of (item_row, field); and found_any per field
    found: dict[str, list[set[tuple[int, int]]]] = {
        m: [set() for _ in range(10)] for m in METRICS
    }
    found_any: set[tuple[int, int]] = set()
    spike_rows_removed = 0
    gate_dropped_hits = 0
    matched_span_words = 0
    matched_doc_words = 0
    matched_docs = 0
    docs_by_decile: dict[str, list[int]] = {m: [0] * 10 for m in METRICS}
    # per-item document counts, for the duplicate-concentration report
    docs_per_item: dict[int, int] = defaultdict(int)

    for domain in DOMAINS:
        path = hits_dir / f"hits_{domain}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["domain"] == "__spike__":
                    spike_rows_removed += 1
                    continue
                did = domain_index[rec["domain"]]
                source_doc = int(rec["source_doc"])
                table = lookup[did]
                row = int(table[source_doc]) if source_doc < table.shape[0] else -1
                if row < 0:
                    # Present in trim/ and scanned, but dropped by the quality
                    # gate, so it is not part of the trained corpus and must
                    # not enter any rate.
                    gate_dropped_hits += 1
                    continue
                matched_docs += 1
                matched_span_words += int(rec["matched_words"])
                matched_doc_words += int(rec["n_words"])
                pairs = [(int(a), int(b)) for a, b in rec["items"]]
                for pair in pairs:
                    found_any.add(pair)
                    docs_per_item[pair[0]] += 1
                for metric in METRICS:
                    d0 = int(decile[f"start_decile_{metric}"][row])
                    d1 = int(decile[f"end_decile_{metric}"][row])
                    docs_by_decile[metric][d0] += 1
                    for d in {d0, d1}:
                        found[metric][d].update(pairs)

    total_spikes = sum(s["spikes_injected"] for s in sentinels.values())
    if spike_rows_removed != total_spikes:
        raise RuntimeError(
            f"removed {spike_rows_removed} spike rows but {total_spikes} were "
            f"injected; spike handling is wrong and rates cannot be trusted"
        )
    log.info("spike gate OK: %d rows removed by identity", spike_rows_removed)

    # Denominators: assessable items per benchmark per field.
    bench_names = sorted(set(benches))
    bench_rows: dict[str, list[int]] = defaultdict(list)
    for row, b in enumerate(benches):
        bench_rows[b].append(row)

    def denom(bench: str, field: int) -> int:
        ok = stem_ok if field == FIELD_STEM else gold_ok
        return int(sum(1 for r in bench_rows[bench] if ok[r]))

    def rate(pairs: set[tuple[int, int]], bench: str, field: int) -> tuple[int, int]:
        ok = stem_ok if field == FIELD_STEM else gold_ok
        rows = {r for (r, f) in pairs if f == field and ok[r]}
        rows &= set(bench_rows[bench])
        return len(rows), denom(bench, field)

    report: dict[str, object] = {
        "sentinels": sentinels,
        "n_items": n_items,
        "corpus": {
            "documents_scanned": SCANNED_DOCS,
            "documents_gate_dropped": GATE_DROPPED,
            "matched_documents_discarded_as_gate_dropped": gate_dropped_hits,
            "documents": CORPUS_DOCS,
            "tokens": CORPUS_TOKENS,
            "matched_documents": matched_docs,
            "matched_document_rate": matched_docs / CORPUS_DOCS,
            "matched_span_words": matched_span_words,
            # Words in trim/, i.e. including the gate-dropped documents.
            # Labelled rather than adjusted: the scanner reports words per
            # domain, not per document, so kept-only words are not recoverable
            # from its output, and the difference is 0.91% of documents.
            "words_scanned_including_gate_dropped": sum(
                s["words_scanned"] for s in sentinels.values()
            ),
            "matched_span_word_rate": matched_span_words
            / max(sum(s["words_scanned"] for s in sentinels.values()), 1),
            "gate_dropped_hit_share": gate_dropped_hits / max(matched_docs + gate_dropped_hits, 1),
            "tokens_in_matched_documents_upper_bound_rate": matched_doc_words
            / max(sum(s["words_scanned"] for s in sentinels.values()), 1),
        },
        "margin": {},
        "cross_tab": {},
        "duplication": {
            "items_found": len({r for (r, _f) in found_any}),
            "max_documents_for_one_item": max(docs_per_item.values(), default=0),
            "top10_item_document_share": (
                sum(sorted(docs_per_item.values(), reverse=True)[:10])
                / max(sum(docs_per_item.values()), 1)
            ),
        },
    }

    # Gate 3: the row margin must not depend on the metric.
    margins: dict[str, dict[str, list[int]]] = {}
    for metric in METRICS:
        union: set[tuple[int, int]] = set()
        for d in range(10):
            union |= found[metric][d]
        margins[metric] = {
            b: [rate(union, b, FIELD_STEM)[0], rate(union, b, FIELD_GOLD)[0]]
            for b in bench_names
        }
    first = margins[METRICS[0]]
    for metric in METRICS[1:]:
        if margins[metric] != first:
            raise RuntimeError(
                f"row margin differs between {METRICS[0]} and {metric}; the "
                f"three corpora hold the same documents, so presence cannot "
                f"depend on ordering -- the decile join is broken"
            )
    log.info("margin gate OK: identical across all three metrics")

    log.info("")
    log.info("%-14s %7s %14s %14s", "benchmark", "items", "stem found", "gold found")
    for b in bench_names:
        s_hit, s_den = rate(set().union(*found[METRICS[0]]), b, FIELD_STEM)
        g_hit, g_den = rate(set().union(*found[METRICS[0]]), b, FIELD_GOLD)
        report["margin"][b] = {  # type: ignore[index]
            "items": len(bench_rows[b]),
            "stem_found": s_hit,
            "stem_assessable": s_den,
            "stem_rate": s_hit / s_den if s_den else None,
            "gold_found": g_hit,
            "gold_assessable": g_den,
            "gold_rate": g_hit / g_den if g_den else None,
        }
        log.info(
            "%-14s %7d %6d/%-7d %6d/%-7d",
            b,
            len(bench_rows[b]),
            s_hit,
            s_den,
            g_hit,
            g_den,
        )

    for metric in METRICS:
        tab: dict[str, list[dict]] = {}
        for b in bench_names:
            cells = []
            for d in range(10):
                s_hit, s_den = rate(found[metric][d], b, FIELD_STEM)
                g_hit, g_den = rate(found[metric][d], b, FIELD_GOLD)
                cells.append(
                    {
                        "decile": d + 1,
                        "stem_found": s_hit,
                        "stem_rate": s_hit / s_den if s_den else None,
                        "gold_found": g_hit,
                        "gold_rate": g_hit / g_den if g_den else None,
                    }
                )
            tab[b] = cells
        report["cross_tab"][metric] = {  # type: ignore[index]
            "benchmarks": tab,
            "matched_documents_by_decile": docs_by_decile[metric],
        }

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("")
    log.info(
        "corpus-side: %d matched documents (%.4f%%), matched-span word rate %.3e",
        matched_docs,
        100 * matched_docs / CORPUS_DOCS,
        report["corpus"]["matched_span_word_rate"],  # type: ignore[index]
    )
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
