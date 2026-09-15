"""The section 4b comparison: did a run score the items the scan indexed?

Section 4b originally only *recorded* identity -- item counts, a doc_id
digest, and text hashes for a sample -- and nothing compared them. That is
the weaker half: an off-by-one join between a contamination hit and a
per-item bpb number produces entirely plausible numbers and no error, which
is this project's characteristic failure.

These tests drive each disagreement the comparison must refuse. They matter
because the comparison runs once, against data that cost GPU hours, and a
verifier that passes on mismatched inputs is worse than none.

Stdlib only.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

CONTAM = Path(__file__).resolve().parents[1] / "contamination"
TASK_LOSS = Path(__file__).resolve().parents[1] / "task_loss"
for extra in (CONTAM, TASK_LOSS):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import item_identity as ii  # noqa: E402
import verify_item_identity as v  # noqa: E402

LABEL = "arc_easy_val_rc_5shot_bpb"


def _dump(n: int = 5, label: str = LABEL) -> dict[str, dict[int, dict[str, str]]]:
    return {
        label: {
            i: {
                "full_context_sha256": ii.sha256(f"ctx {i}"),
                "gold_sha256": ii.sha256(f"gold {i}"),
            }
            for i in range(n)
        }
    }


def _identity(
    dump: dict[str, dict[int, dict[str, str]]], sample: list[int] | None = None
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label, items in dump.items():
        picked = sample if sample is not None else sorted(items)
        out[label] = {
            "n_docs": len(items),
            "doc_id_digest": v.doc_id_digest(list(items)),
            "sampled": {
                str(d): {
                    "full_context_sha256": items[d]["full_context_sha256"],
                    "gold_sha256": items[d]["gold_sha256"],
                }
                for d in picked
            },
        }
    return out


def _fp() -> str:
    return ii.normalizer_fingerprint()


def test_matching_sides_verify() -> None:
    dump = _dump()
    summary = v.verify_identity(dump, _identity(dump), payload_fingerprint=_fp())
    assert summary["labels"] == 1
    assert summary["items_total"] == 5
    assert summary["items_hash_checked"] == 5


def test_a_sampled_subset_is_enough() -> None:
    """The evaluator hashes only ~64 spaced doc_ids per label, so the
    comparison must accept a subset while still checking the full count and
    digest."""
    dump = _dump(n=100)
    identity = _identity(dump, sample=[0, 25, 50, 99])
    summary = v.verify_identity(dump, identity, payload_fingerprint=_fp())
    assert summary["items_total"] == 100
    assert summary["items_hash_checked"] == 4


def test_a_differing_item_count_is_refused() -> None:
    dump = _dump(n=5)
    identity = _identity(dump)
    identity[LABEL]["n_docs"] = 6
    with pytest.raises(v.IdentityMismatch, match="scored 6 items, the dump holds 5"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())


def test_a_reordered_item_set_is_refused_even_at_the_same_count() -> None:
    """The failure a count check cannot see, and the one that makes a
    positional join pair the wrong rows."""
    dump = _dump(n=5)
    identity = _identity(dump)
    identity[LABEL]["doc_id_digest"] = v.doc_id_digest([0, 1, 2, 3, 99])
    with pytest.raises(v.IdentityMismatch, match="doc_id digest differs"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())


def test_changed_underlying_text_is_refused() -> None:
    """doc_ids can line up perfectly while the oe-eval text behind them has
    changed. Only the hashes catch it."""
    dump = _dump(n=5)
    identity = _identity(dump)
    identity[LABEL]["sampled"]["2"]["gold_sha256"] = ii.sha256("something else")
    with pytest.raises(v.IdentityMismatch, match="doc_id 2: gold_sha256 differs"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())

    identity = _identity(dump)
    identity[LABEL]["sampled"]["3"]["full_context_sha256"] = ii.sha256("other ctx")
    with pytest.raises(v.IdentityMismatch, match="doc_id 3: full_context_sha256 differs"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())


def test_a_scored_doc_id_absent_from_the_dump_is_refused() -> None:
    dump = _dump(n=5)
    identity = _identity(dump)
    identity[LABEL]["sampled"]["77"] = identity[LABEL]["sampled"]["0"]
    identity[LABEL]["n_docs"] = 5
    with pytest.raises(v.IdentityMismatch, match="scored doc_id 77"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())


def test_label_set_differences_are_refused() -> None:
    dump = _dump()
    identity = _identity(dump)
    identity["boolq_val_rc_5shot_bpb"] = identity[LABEL]
    with pytest.raises(v.IdentityMismatch, match="label sets differ"):
        v.verify_identity(dump, identity, payload_fingerprint=_fp())


def test_a_missing_fingerprint_is_refused() -> None:
    """Every recorded hash is post-normalization, so an unpinned normalizer
    makes them uncomparable rather than merely unverified."""
    dump = _dump()
    with pytest.raises(v.IdentityMismatch, match="no normalizer_fingerprint"):
        v.verify_identity(dump, _identity(dump), payload_fingerprint=None)


def test_a_differing_fingerprint_is_refused() -> None:
    dump = _dump()
    with pytest.raises(v.IdentityMismatch, match="normalizer fingerprint differs"):
        v.verify_identity(dump, _identity(dump), payload_fingerprint="deadbeef")


def test_duplicate_doc_ids_in_the_dump_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for _ in range(2):
            fh.write(
                json.dumps(
                    {
                        "label": LABEL,
                        "doc_id": 0,
                        "full_context_sha256": "a",
                        "gold_sha256": "b",
                    }
                )
                + "\n"
            )
    with pytest.raises(v.IdentityMismatch, match="duplicate doc_id 0"):
        v.load_dump(path)


def test_an_empty_dump_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8"):
        pass
    with pytest.raises(v.IdentityMismatch, match="dump is empty"):
        v.load_dump(path)


def test_digest_construction_matches_the_evaluator() -> None:
    """The evaluator builds the digest as sha256 of the comma-joined sorted
    doc_ids. If these two drift apart every comparison fails for a reason
    that has nothing to do with the data."""
    assert v.doc_id_digest([2, 0, 1]) == ii.sha256("0,1,2")
    assert v.doc_id_digest([10, 9]) == ii.sha256("9,10")
