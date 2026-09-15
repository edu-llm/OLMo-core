"""Canonical eval-item identity, shared by the scan and the evaluator.

The contamination scan searches the corpus for strings derived from eval items;
the evaluator records a hash of the items it scored. If those disagree, a
contamination hit cannot be joined to a per-item bpb number -- and a join that
is off by one yields entirely plausible numbers. So there is one normalizer and
both sides import it, and these tests pin its behavior.

Stdlib only, so they run where the evaluator itself cannot be imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TASK_LOSS = Path(__file__).resolve().parents[1] / "task_loss"
if str(TASK_LOSS) not in sys.path:
    sys.path.insert(0, str(TASK_LOSS))

import item_identity as ii  # noqa: E402


def test_normalization_is_the_documented_pipeline() -> None:
    assert ii.normalize("The Quick--Brown FOX, jumps!") == "the quick brown fox jumps"
    assert ii.normalize("a  b\tc\nd") == "a b c d"
    assert ii.normalize("dog's") == "dog s"
    assert ii.normalize("") == ""


def test_nfkc_folding_is_applied() -> None:
    """A ligature and a full-width digit must fold, or the index and the
    corpus disagree on text that is visually identical."""
    assert ii.normalize("ﬁne") == "fine"
    assert ii.normalize("１２") == "12"


def test_normalizer_fingerprint_is_stable_and_behavioral() -> None:
    assert ii.normalizer_fingerprint() == ii.normalizer_fingerprint()
    assert len(ii.normalizer_fingerprint()) == 64


def test_common_prefix() -> None:
    assert ii.common_prefix(["abcdef", "abcxyz"]) == "abc"
    assert ii.common_prefix(["abc"]) == "abc"
    assert ii.common_prefix([]) == ""
    assert ii.common_prefix(["abc", "xyz"]) == ""


def test_shared_few_shot_preamble_is_stripped() -> None:
    """The five exemplars are identical across a label, so they show up as a
    long common prefix. Leaving them in would make every item in the label
    match on any exemplar text."""
    preamble = "here are five worked examples of the task for you to imitate "
    contexts = [preamble + "first real question", preamble + "second real question"]
    stems, found = ii.strip_preamble(contexts)
    assert found == preamble
    assert stems == ["first real question", "second real question"]


def test_short_accidental_overlap_is_not_treated_as_a_preamble() -> None:
    """Two stems can share a few opening words by chance. Stripping that
    would silently truncate the stems the index is built from."""
    contexts = ["what is the capital of France", "what is the capital of Peru"]
    stems, found = ii.strip_preamble(contexts)
    assert found == ""
    assert stems == contexts


def test_identity_record_shape() -> None:
    rec = ii.item_identity("What is 2+2?", " four")
    assert rec["stem_words"] == 4
    assert rec["gold_words"] == 1
    assert rec["stem_sha256"] == ii.sha256("what is 2 2")
    assert rec["gold_sha256"] == ii.sha256("four")


def test_identities_for_label_strips_once_across_the_label() -> None:
    preamble = "five exemplars appear here before every single question asked "
    contexts = [preamble + "alpha question", preamble + "beta question"]
    golds = ["first gold", "second gold"]
    records, found = ii.identities_for_label(contexts, golds)
    assert found == preamble
    assert [r["stem_sha256"] for r in records] == [
        ii.sha256("alpha question"),
        ii.sha256("beta question"),
    ]


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="contexts against"):
        ii.identities_for_label(["a"], ["b", "c"])


def test_hashes_are_computed_on_normalized_text_not_raw() -> None:
    """The whole point: the scan normalizes corpus text, so the recorded hash
    must be of the normalized item or the two can never match."""
    a = ii.item_identity("The CAT sat!", "on the MAT")
    b = ii.item_identity("the cat sat", "on the mat")
    assert a["stem_sha256"] == b["stem_sha256"]
    assert a["gold_sha256"] == b["gold_sha256"]
