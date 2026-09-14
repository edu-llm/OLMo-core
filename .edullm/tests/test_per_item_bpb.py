"""Per-item gold bpb capture: reduction, verification, and serialization.

The endpoint is a macro-average, so a contamination-driven arm effect and a
genuine curriculum effect look identical at the aggregate. Per-item bpb is what
separates them -- it lets the endpoint be recomputed on a clean item subset and
shows whether an arm's advantage concentrates on matched items. It cannot be
reconstructed after a run, so it is captured at every permanent checkpoint.

The values already exist: `ICLMetric.update()` stores
`(doc_id, cont_id, log_likelihood)` where for a `*_bpb` label the third value
is already bits-per-byte, and `compute()` keeps the gold continuation per
document and averages the rest away. These tests cover the reduction that
mirrors `compute()` and the guard that proves the capture reproduces the
harness's own number.

Stdlib only, so they run everywhere -- the evaluator module itself imports
torch, olmo_core and ai2-olmo and cannot be imported on a laptop or a CPU node.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK_LOSS = Path(__file__).resolve().parents[1] / "task_loss"
if str(TASK_LOSS) not in sys.path:
    sys.path.insert(0, str(TASK_LOSS))

import per_item as pi  # noqa: E402


def test_reduction_matches_the_harness_aggregate() -> None:
    rows = {"lbl": [(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0)]}
    captured, duplicates = pi.reduce_per_item(rows, {"lbl": 2.0})
    assert captured["lbl"] == {0: 1.0, 1: 2.0, 2: 3.0}
    assert duplicates == 0


def test_only_the_gold_continuation_is_kept() -> None:
    """`compute()` takes the lowest cont_id per document; the distractors
    must not enter the mean, or the captured values would not be the
    endpoint's quantity."""
    rows = {"lbl": [(0, 0, 1.0), (0, 1, 9.0), (0, 2, 9.0), (1, 0, 3.0), (1, 1, 9.0)]}
    captured, _ = pi.reduce_per_item(rows, {"lbl": 2.0})
    assert captured["lbl"] == {0: 1.0, 1: 3.0}


def test_distributed_sampler_padding_is_counted_not_double_counted() -> None:
    """The sampler pads to divisibility, so an item can land on two ranks.
    ICLMetric collapses those by dict assignment, so the aggregate is
    unaffected -- but the count quantifies audit finding 1e."""
    rows = {"lbl": [(0, 0, 1.0), (1, 0, 3.0), (0, 0, 1.0), (1, 0, 3.0)]}
    captured, duplicates = pi.reduce_per_item(rows, {"lbl": 2.0})
    assert captured["lbl"] == {0: 1.0, 1: 3.0}
    assert duplicates == 2


def test_a_capture_that_disagrees_with_the_harness_is_refused() -> None:
    """The whole point of the guard: if the capture reads the wrong state,
    it must fail loudly rather than record numbers that look plausible but
    do not correspond to the reported endpoint."""
    rows = {"lbl": [(0, 0, 1.0), (1, 0, 2.0)]}
    with pytest.raises(RuntimeError, match="does not reproduce the reported"):
        pi.reduce_per_item(rows, {"lbl": 0.5})


def test_capturing_nothing_is_an_error_not_an_empty_result() -> None:
    """An empty capture would serialize as a clean, complete-looking file."""
    with pytest.raises(RuntimeError, match="captured no per-item rows"):
        pi.reduce_per_item({"lbl": []}, {"lbl": 1.0})


def test_round_trip_through_the_written_file(tmp_path: Path) -> None:
    per_item = {"b_label": {5: 2.5, 1: 1.5}, "a_label": {0: 0.5}}
    path = tmp_path / "nested" / "step125_task_loss_per_item.jsonl.gz"
    pi.write_per_item(path, per_item, step=125)

    rows = pi.read_per_item(path)
    assert len(rows) == 3
    assert all(row["step"] == 125 for row in rows)
    # Deterministic ordering by (label, doc_id) so two runs of the same
    # checkpoint produce byte-identical files.
    assert [(r["label"], r["doc_id"]) for r in rows] == [
        ("a_label", 0),
        ("b_label", 1),
        ("b_label", 5),
    ]
    recovered = {}
    for row in rows:
        recovered.setdefault(row["label"], {})[row["doc_id"]] = row["gold_bpb"]
    assert recovered == per_item


def test_float_tolerance_is_relative_not_absolute() -> None:
    """Accumulating ~40k floats reorders differently across ranks, so the
    guard must tolerate float noise while still catching a real mismatch."""
    rows = {"lbl": [(i, 0, 1.0) for i in range(1000)]}
    pi.reduce_per_item(rows, {"lbl": 1.0 + 1e-9})
    with pytest.raises(RuntimeError):
        pi.reduce_per_item(rows, {"lbl": 1.01})
