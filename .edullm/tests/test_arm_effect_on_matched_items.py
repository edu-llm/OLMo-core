"""The decision statistic: does an arm's effect concentrate on leaked items?

T = mean(delta_i over matched) - mean(delta_i over unmatched), where
delta_i = bpb_i(arm) - bpb_i(control) at a matched seed.

This is the only thing in the pipeline that answers the actual question, and
it cannot be run until checkpoints exist -- which means it cannot be debugged
then either, because by that point the runs are the expensive part. So its
arithmetic and its refusals are pinned here on constructed data where the
right answer is known by hand.

Stdlib only.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

CONTAM = Path(__file__).resolve().parents[1] / "contamination"
if str(CONTAM) not in sys.path:
    sys.path.insert(0, str(CONTAM))

import arm_effect_on_matched_items as ae  # noqa: E402

LABEL = "hellaswag_val_rc_5shot_bpb"


def _keys(n: int) -> list[tuple[str, int]]:
    return [(LABEL, i) for i in range(n)]


def test_T_is_zero_when_the_effect_is_uniform() -> None:
    """The null this test exists to detect: an arm that improves every item
    equally is not contamination-driven, whatever the contamination rate."""
    keys = _keys(10)
    deltas = dict.fromkeys(keys, -0.01)
    rows = {k: i for i, k in enumerate(keys)}
    stats = ae.compute_T(deltas, rows, {0, 1, 2}, keys)
    assert stats["n_matched"] == 3
    assert stats["n_unmatched"] == 7
    assert stats["T"] == pytest.approx(0.0)


def test_T_is_negative_when_matched_items_improve_more() -> None:
    """The signature of contamination driving the effect: gold bpb falls
    further on items the corpus contains."""
    keys = _keys(10)
    deltas = {k: (-0.10 if i < 3 else -0.01) for i, k in enumerate(keys)}
    rows = {k: i for i, k in enumerate(keys)}
    stats = ae.compute_T(deltas, rows, {0, 1, 2}, keys)
    assert stats["T"] == pytest.approx(-0.09)


def test_a_false_positive_in_the_matched_set_dilutes_T_toward_zero() -> None:
    """Why the index is deliberately loose: junk in the matched set is the
    conservative direction. Here item 3 is not really leaked, and including
    it shrinks the measured effect rather than inflating it."""
    keys = _keys(10)
    deltas = {k: (-0.10 if i < 3 else -0.01) for i, k in enumerate(keys)}
    rows = {k: i for i, k in enumerate(keys)}
    clean = ae.compute_T(deltas, rows, {0, 1, 2}, keys)["T"]
    with_fp = ae.compute_T(deltas, rows, {0, 1, 2, 3}, keys)["T"]
    assert clean is not None and with_fp is not None
    assert abs(with_fp) < abs(clean)


def test_T_is_none_when_a_side_is_empty() -> None:
    """A benchmark with no matched items yields no statistic, not a zero.
    Reporting 0.0 there would read as evidence of no effect."""
    keys = _keys(5)
    deltas = dict.fromkeys(keys, -0.01)
    rows = {k: i for i, k in enumerate(keys)}
    assert ae.compute_T(deltas, rows, set(), keys)["T"] is None
    assert ae.compute_T(deltas, rows, {0, 1, 2, 3, 4}, keys)["T"] is None


def test_paired_deltas_refuse_a_partial_overlap() -> None:
    """A paired difference over different item sets is not defined, and
    silently intersecting would compute T on an unannounced subset."""
    arm = {(LABEL, 0): 1.0, (LABEL, 1): 1.0}
    control = {(LABEL, 0): 1.1}
    with pytest.raises(RuntimeError, match="different item sets"):
        ae.paired_deltas(arm, control)


def test_paired_deltas_subtract_in_the_right_direction() -> None:
    """Lower bpb is better, so an improving arm must give a NEGATIVE delta."""
    arm = {(LABEL, 0): 0.90}
    control = {(LABEL, 0): 1.00}
    assert ae.paired_deltas(arm, control)[(LABEL, 0)] == pytest.approx(-0.10)


def test_subsets_are_nested_and_field_provenance_drives_them() -> None:
    matched = {0: {ae.FIELD_GOLD}, 1: {ae.FIELD_STEM}, 2: {ae.FIELD_STEM, ae.FIELD_GOLD}}
    subsets = ae.subset_membership(matched, n_items=10)
    assert subsets["S_gold"] == {0, 2}
    assert subsets["S_gold_or_stem"] == {0, 1, 2}
    assert subsets["S_any"] == {0, 1, 2}
    assert subsets["S_gold"] <= subsets["S_gold_or_stem"] <= subsets["S_any"]


def test_a_matched_row_outside_the_inventory_is_refused() -> None:
    with pytest.raises(RuntimeError, match="outside the item inventory"):
        ae.subset_membership({99: {ae.FIELD_GOLD}}, n_items=10)


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    keys = _keys(200)
    deltas = {k: (-0.10 if i < 50 else -0.01) for i, k in enumerate(keys)}
    rows = {k: i for i, k in enumerate(keys)}
    subset = set(range(50))
    point = ae.compute_T(deltas, rows, subset, keys)["T"]
    lo, hi = ae.bootstrap_ci(deltas, rows, subset, keys, draws=500, seed=1)
    assert lo <= point <= hi


def test_bootstrap_ci_is_none_without_both_sides() -> None:
    keys = _keys(5)
    deltas = dict.fromkeys(keys, -0.01)
    rows = {k: i for i, k in enumerate(keys)}
    assert ae.bootstrap_ci(deltas, rows, set(), keys, draws=10) is None


def test_duplicate_per_item_rows_are_refused(tmp_path: Path) -> None:
    """One row per (label, doc_id) is the invariant the join depends on."""
    path = tmp_path / "per_item.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for _ in range(2):
            fh.write(
                json.dumps({"step": 1, "label": LABEL, "doc_id": 0, "gold_bpb": 1.0}) + "\n"
            )
    with pytest.raises(RuntimeError, match="duplicate per-item row"):
        ae.read_per_item(path)


def test_empty_per_item_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "per_item.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8"):
        pass
    with pytest.raises(RuntimeError, match="no per-item rows"):
        ae.read_per_item(path)


def test_benchmark_mapping_matches_the_endpoint_grouping() -> None:
    assert ae.benchmark_of("mmlu_other_test_rc_5shot_bpb") == "mmlu"
    assert ae.benchmark_of("arc_easy_val_rc_5shot_bpb") == "arc_easy"
    assert ae.benchmark_of("winogrande_val_rc_5shot_bpb") == "winogrande"
