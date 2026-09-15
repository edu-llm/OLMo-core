#!/usr/bin/env python3
"""Reduction and serialization of per-item gold bpb.

Kept free of torch, olmo_core and ai2-olmo on purpose. The evaluator that
calls this imports all three, so nothing in that module can be tested on a
laptop or on a CPU-only node -- and an untested reduction is exactly the kind
of thing that silently records the wrong numbers. These functions are pure
stdlib so they run in the ordinary test suite.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

# Relative tolerance when checking the recomputed mean against the harness's
# own scalar. These two numbers are computed in DIFFERENT PRECISIONS and can
# never agree exactly: `ICLMetric` accumulates in float32 torch tensors, while
# this module sums Python floats in float64.
#
# Calibrated against a real run rather than guessed. The first execution of
# this code (smoke 1725050, step 0) produced
# 3.6052631578947367 against 3.6052494049072266 on arc_easy_val -- a relative
# difference of 3.8e-6 over 570 items, which the previous 1e-6 tolerance
# rejected. float32 has eps ~1.19e-7, so pairwise summation over n items
# drifts on the order of log2(n) * eps, i.e. ~1e-6 at n=570 and ~1.7e-6 at
# n=15,573 (MMLU) -- with the constant factor easily pushing that an order of
# magnitude higher.
#
# Note which side is the less accurate one. The per-item values are exact --
# each is a float32 read out of the metric and widened -- so the drift lives
# entirely in the harness's float32 SUMMATION, and this module's float64 mean
# is the better estimate of the two. The guard is therefore checking
# agreement, not correcting toward the harness.
#
# The size of the discrepancy is itself diagnostic: one missing or extra item
# would move a 570-item mean by ~5e-3, three orders of magnitude above what
# was seen, so the observed drift cannot be a capture error.
#
# 1e-4 leaves one to two orders of magnitude of headroom above float noise
# while still catching every failure this guard exists for. Those are all
# gross: reading the wrong label's rows, keeping distractors instead of the
# gold continuation, or a truncated capture each move the mean by percent,
# not by parts per million.
AGGREGATE_REL_TOL = 1e-4


def reduce_per_item(
    raw_rows: dict[str, list[tuple[int, int, float]]],
    labels: dict[str, float],
) -> tuple[dict[str, dict[int, float]], int]:
    """Reduce captured rows exactly as ``ICLMetric.compute()`` does, and verify.

    ``compute()`` builds ``loglikelihood_dict[doc_id][cont_id]``, takes the
    gold continuation per document, and averages. Reproducing that here and
    requiring the result to equal the harness's own scalar is what makes the
    capture trustworthy: if it reads the wrong state, or the gold continuation
    is not the lowest ``cont_id``, this raises instead of silently recording
    numbers that do not correspond to the reported endpoint.

    :param raw_rows: per label, the gathered ``(doc_id, cont_id, value)``
        triples from every rank.
    :param labels: the harness's own per-label aggregate, used as the oracle.

    :returns: the per-document gold values, and the number of duplicate
        ``(doc_id, cont_id)`` rows. Duplicates come from
        ``DistributedSampler`` padding to divisibility (audit finding 1e);
        ``ICLMetric`` collapses them by dict assignment so they do not bias
        the aggregate, but the count is worth reporting.

    :raises RuntimeError: if a label captured no rows, or if the recomputed
        aggregate disagrees with the harness.
    """
    per_item: dict[str, dict[int, float]] = {}
    padding_duplicates = 0
    for label, rows in raw_rows.items():
        by_doc: dict[int, dict[int, float]] = {}
        for doc_id, cont_id, value in rows:
            conts = by_doc.setdefault(int(doc_id), {})
            if int(cont_id) in conts:
                padding_duplicates += 1
            conts[int(cont_id)] = float(value)
        gold = {doc_id: conts[min(conts)] for doc_id, conts in by_doc.items()}
        if not gold:
            raise RuntimeError(f"{label}: captured no per-item rows")
        recomputed = sum(gold.values()) / len(gold)
        expected = float(labels[label])
        if abs(recomputed - expected) > AGGREGATE_REL_TOL * max(1.0, abs(expected)):
            raise RuntimeError(
                f"{label}: per-item capture does not reproduce the reported "
                f"aggregate ({recomputed!r} vs {expected!r}, relative difference "
                f"{abs(recomputed - expected) / max(1.0, abs(expected)):.2e} "
                f"exceeds {AGGREGATE_REL_TOL:.0e} over {len(gold)} items); the "
                f"captured values do not correspond to the endpoint and must "
                f"not be used"
            )
        per_item[label] = gold
    return per_item, padding_duplicates


def write_per_item(path: str | Path, per_item: dict[str, dict[int, float]], step: int) -> None:
    """Write one gzipped JSONL row per ``(label, doc_id)`` of gold bpb.

    Written at every permanent checkpoint because it cannot be reconstructed
    afterwards. It is what allows the endpoint to be recomputed on a clean
    item subset, and what shows whether an arm's effect concentrates on
    contamination-matched items rather than being spread across the suite.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        for label in sorted(per_item):
            for doc_id in sorted(per_item[label]):
                handle.write(
                    json.dumps(
                        {
                            "step": int(step),
                            "label": label,
                            "doc_id": int(doc_id),
                            "gold_bpb": float(per_item[label][doc_id]),
                        }
                    )
                    + "\n"
                )


def read_per_item(path: str | Path) -> list[dict]:
    """Read back a per-item file. Used by tests and by downstream analysis."""
    rows: list[dict] = []
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
