"""End-to-end exercise of the contamination aggregator on synthetic inputs.

The aggregator runs once, on data that took an hour of cluster time to
produce, and its three gates are the only thing standing between a silently
incomplete scan and a published rate. Debugging it against the real hits is
the wrong time to find out that a gate is inverted, so it is driven here
against fixtures small enough to reason about.

The gates under test:

  1. Coverage -- exactly one DONE sentinel per domain, and docs_scanned
     summing to the PRE-gate trim/ line count, because the scanner reads
     every line. A silently failed array task would move its share of matches
     into the unmatched group. Asserting the post-gate count here instead
     fails on a complete scan, which is how that bug was found.
  2. Spikes -- removed by identity, with the count asserted.
  3. Margin -- an item found anywhere must be found identically under all
     three metrics, because the three corpora hold the same documents in
     different orders.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CONTAM = Path(__file__).resolve().parents[1] / "contamination"
sys.path.insert(0, str(CONTAM))

import aggregate_contamination as agg  # noqa: E402

DOMAINS = agg.DOMAINS


def test_benchmark_mapping_collapses_mmlu_and_keeps_splits_together() -> None:
    """The endpoint is a macro over 10 benchmarks, so all eight MMLU labels
    and both splits of ARC must land on one benchmark each."""
    assert agg.benchmark_of("mmlu_stem_val_rc_5shot_bpb") == "mmlu"
    assert agg.benchmark_of("mmlu_humanities_test_rc_5shot_bpb") == "mmlu"
    assert agg.benchmark_of("arc_challenge_val_rc_5shot_bpb") == "arc_challenge"
    assert agg.benchmark_of("arc_challenge_test_rc_5shot_bpb") == "arc_challenge"
    assert agg.benchmark_of("boolq_val_rc_5shot_bpb") == "boolq"
    labels = [
        "arc_challenge_val_rc_5shot_bpb",
        "arc_challenge_test_rc_5shot_bpb",
        "arc_easy_val_rc_5shot_bpb",
        "arc_easy_test_rc_5shot_bpb",
        "boolq_val_rc_5shot_bpb",
        "csqa_val_rc_5shot_bpb",
        "hellaswag_val_rc_5shot_bpb",
        "openbookqa_val_rc_5shot_bpb",
        "openbookqa_test_rc_5shot_bpb",
        "piqa_val_rc_5shot_bpb",
        "socialiqa_val_rc_5shot_bpb",
        "winogrande_val_rc_5shot_bpb",
        "mmlu_stem_val_rc_5shot_bpb",
    ]
    assert len({agg.benchmark_of(x) for x in labels}) == 10


def _write_sentinels(hits: Path, per_domain_docs: dict[str, int], spikes: int = 2) -> None:
    hits.mkdir(parents=True, exist_ok=True)
    for domain, docs in per_domain_docs.items():
        (hits / f"DONE_{domain}.json").write_text(
            json.dumps(
                {
                    "domain": domain,
                    "docs_scanned": docs,
                    "words_scanned": docs * 10,
                    "hit_rows": 0,
                    "spikes_injected": spikes,
                    "spikes_recovered": spikes,
                    "elapsed_s": 1.0,
                    "canary_ok": True,
                }
            ),
            encoding="utf-8",
        )


def _even_split(total: int) -> dict[str, int]:
    base, extra = divmod(total, len(DOMAINS))
    return {d: base + (1 if i < extra else 0) for i, d in enumerate(DOMAINS)}


def test_missing_sentinel_is_refused(tmp_path: Path) -> None:
    hits = tmp_path / "hits"
    split = _even_split(agg.SCANNED_DOCS)
    split.pop("wiki")
    _write_sentinels(hits, split)
    with pytest.raises(RuntimeError, match="missing DONE sentinel for wiki"):
        agg.check_sentinels(hits)


def test_short_coverage_is_refused(tmp_path: Path) -> None:
    """The failure this gate exists for: every task reports success but one
    scanned fewer documents than it should have."""
    hits = tmp_path / "hits"
    split = _even_split(agg.SCANNED_DOCS)
    split["dclm"] -= 1000
    _write_sentinels(hits, split)
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        agg.check_sentinels(hits)


def test_unrecovered_spikes_are_refused(tmp_path: Path) -> None:
    hits = tmp_path / "hits"
    _write_sentinels(hits, _even_split(agg.SCANNED_DOCS))
    path = hits / "DONE_wiki.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["spikes_recovered"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="positive controls not fully recovered"):
        agg.check_sentinels(hits)


def test_failed_canary_is_refused(tmp_path: Path) -> None:
    hits = tmp_path / "hits"
    _write_sentinels(hits, _even_split(agg.SCANNED_DOCS))
    path = hits / "DONE_dclm.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canary_ok"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canary did not pass"):
        agg.check_sentinels(hits)


def test_complete_coverage_passes(tmp_path: Path) -> None:
    hits = tmp_path / "hits"
    _write_sentinels(hits, _even_split(agg.SCANNED_DOCS))
    sentinels = agg.check_sentinels(hits)
    assert sum(s["docs_scanned"] for s in sentinels.values()) == agg.SCANNED_DOCS


def _tiny_corpus(tmp_path: Path, hit_records: list[dict]) -> tuple[Path, Path, Path]:
    """Build a 4-document, 2-item fixture and return (hits, items, decile map)."""
    hits = tmp_path / "hits"
    _write_sentinels(hits, _even_split(agg.SCANNED_DOCS), spikes=1)
    for domain in DOMAINS:
        rows = [r for r in hit_records if r["domain"] == domain]
        # One spike row per domain, matching spikes_injected=1.
        rows = [
            {
                "domain": "__spike__",
                "source_doc": 0,
                "n_words": 20,
                "matched_words": 0,
                "items": [[0, 0]],
            }
        ] + rows
        with gzip.open(hits / f"hits_{domain}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    items = tmp_path / "items.jsonl.gz"
    with gzip.open(items, "wt", encoding="utf-8") as fh:
        for row, label in enumerate(
            ["hellaswag_val_rc_5shot_bpb", "boolq_val_rc_5shot_bpb"]
        ):
            fh.write(
                json.dumps(
                    {
                        "label": label,
                        "doc_id": row,
                        "stem_words": 30,
                        # BoolQ's gold is yes/no, i.e. NOT ASSESSABLE.
                        "gold_words": 30 if row == 0 else 1,
                    }
                )
                + "\n"
            )

    # Four documents, all in domain "wiki" (domain_id 7), spread over deciles.
    dm = tmp_path / "decile_map.npz"
    arrays = {
        "domain_id": np.array([7, 7, 7, 7], dtype=np.int8),
        "source_doc": np.array([0, 1, 2, 3], dtype=np.int64),
        "n_tokens": np.array([100, 100, 100, 100], dtype=np.int64),
    }
    for metric in agg.METRICS:
        arrays[f"start_decile_{metric}"] = np.array([0, 1, 2, 3], dtype=np.int8)
        arrays[f"end_decile_{metric}"] = np.array([0, 1, 2, 3], dtype=np.int8)
    np.savez_compressed(dm, **arrays)
    return hits, items, dm


def test_end_to_end_on_a_tiny_fixture(tmp_path: Path) -> None:
    """A hellaswag item found in one wiki document must show up in exactly one
    decile cell, with the margin identical across metrics."""
    records = [
        {
            "domain": "wiki",
            "source_doc": 1,
            "n_words": 500,
            "matched_words": 13,
            "items": [[0, 0], [0, 1]],
        }
    ]
    hits, items, dm = _tiny_corpus(tmp_path, records)
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CONTAM / "aggregate_contamination.py"),
            "--hits",
            str(hits),
            "--items",
            str(items),
            "--decile-map",
            str(dm),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))

    assert report["n_items"] == 2
    assert report["corpus"]["matched_documents"] == 1
    assert report["corpus"]["matched_span_words"] == 13
    # hellaswag row 0 is assessable on both sides and was found.
    assert report["margin"]["hellaswag"]["stem_found"] == 1
    assert report["margin"]["hellaswag"]["gold_found"] == 1
    # boolq's gold is NOT ASSESSABLE, so its denominator is zero and its rate
    # is None rather than 0.0 -- reporting 0% from an invalid instrument is
    # the defect this whole design exists to avoid.
    assert report["margin"]["boolq"]["gold_assessable"] == 0
    assert report["margin"]["boolq"]["gold_rate"] is None
    # The document sits in decile 2 (index 1), so that is the only cell.
    cells = report["cross_tab"]["mtld"]["benchmarks"]["hellaswag"]
    assert [c["stem_found"] for c in cells] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_spike_count_mismatch_is_refused(tmp_path: Path) -> None:
    """If spike rows are miscounted, rates cannot be trusted -- subtracting
    them by count rather than identity would floor every cell at zero."""
    hits, items, dm = _tiny_corpus(tmp_path, [])
    # Claim two spikes were injected while only one row carries the marker.
    for domain in DOMAINS:
        path = hits / f"DONE_{domain}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["spikes_injected"] = 2
        payload["spikes_recovered"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CONTAM / "aggregate_contamination.py"),
            "--hits",
            str(hits),
            "--items",
            str(items),
            "--decile-map",
            str(dm),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "spike rows" in result.stdout + result.stderr


def test_the_three_document_counts_reconcile() -> None:
    """trim/ holds the pre-gate corpus; the trained corpus is smaller by
    exactly the gate's drop count. Conflating them is what broke the coverage
    gate, so the relationship is asserted rather than assumed."""
    assert agg.SCANNED_DOCS - agg.CORPUS_DOCS == agg.GATE_DROPPED
    assert agg.SCANNED_DOCS == 5_911_841
    assert agg.CORPUS_DOCS == 5_857_887
    assert agg.GATE_DROPPED == 53_954


def test_a_hit_on_a_gate_dropped_document_is_discarded(tmp_path: Path) -> None:
    """A document can be present in trim/, be scanned, match an item, and
    still not belong to the trained corpus. Counting it would inflate every
    rate against a denominator it is not part of."""
    records = [
        {
            "domain": "wiki",
            # Beyond the 4 documents the fixture's decile map knows about.
            "source_doc": 99,
            "n_words": 500,
            "matched_words": 13,
            "items": [[0, 0]],
        }
    ]
    hits, items, dm = _tiny_corpus(tmp_path, records)
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CONTAM / "aggregate_contamination.py"),
            "--hits", str(hits), "--items", str(items),
            "--decile-map", str(dm), "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["corpus"]["matched_documents"] == 0
    assert report["corpus"]["matched_documents_discarded_as_gate_dropped"] == 1
    assert report["margin"]["hellaswag"]["stem_found"] == 0
