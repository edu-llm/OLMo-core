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


def test_subject_wise_exemplars_are_stripped() -> None:
    """MMLU draws its five exemplars per SUBJECT, so no prefix is shared by
    every item in the label and `strip_preamble` finds nothing -- leaving ~95%
    of each stem as exemplar text. Sorting puts same-subject items adjacent,
    so the prefix shared across a run of them is exactly the exemplar block.

    Each subject needs more than MIN_SHARERS items, which is the realistic
    case: MMLU subjects hold dozens.
    """
    astro = "astronomy question why is mars red answer oxidized minerals question "
    bio = "biology question what is a cell answer the basic unit of life question "
    tails = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf"]
    contexts = [astro + t for t in tails] + [bio + t for t in tails]
    # The label-wide prefix is empty, so the old path strips nothing.
    unchanged, preamble = ii.strip_preamble(contexts)
    assert preamble == ""
    assert unchanged == contexts

    stems, stripped = ii.strip_shared_prefixes(contexts)
    assert stems == tails + tails
    assert all(count >= ii.MIN_WIDTH for count in stripped)


def test_an_overlap_shared_by_too_few_items_is_left_alone() -> None:
    """Two questions opening with the same words is not an exemplar block.
    Stripping it would shorten the stem the index is built from, and could
    push an item under the assessability floor, for no gain."""
    shared = "these are nine words shared by exactly two items here "
    contexts = [shared + "one", shared + "two"] + [
        f"wholly distinct context number {i} with no shared opening at all"
        for i in range(8)
    ]
    stems, stripped = ii.strip_shared_prefixes(contexts)
    assert stems[0] == contexts[0]
    assert stems[1] == contexts[1]
    assert stripped[0] == 0 and stripped[1] == 0


def test_shared_prefix_cuts_land_on_word_boundaries() -> None:
    """A cut inside a word would leave a fragment that matches nothing and
    silently shorten the indexed stem."""
    shared = "the first eight words here are common to all of these items "
    contexts = [shared + "alpha beta", shared + "alphabet soup"] + [
        shared + f"other tail {i}" for i in range(6)
    ]
    stems, _ = ii.strip_shared_prefixes(contexts)
    # "alpha" is a prefix of "alphabet", so a naive character LCP would cut
    # mid-word and leave "bet soup".
    assert stems[0] == "alpha beta"
    assert stems[1] == "alphabet soup"


def test_a_short_shared_run_is_not_stripped() -> None:
    contexts = ["what is the capital of France", "what is the capital of Peru"]
    stems, stripped = ii.strip_shared_prefixes(contexts)
    assert stems == contexts
    assert stripped == [0, 0]


def test_a_lone_item_keeps_its_whole_context() -> None:
    contexts = ["a single item has no neighbour to share a prefix with"]
    stems, stripped = ii.strip_shared_prefixes(contexts)
    assert stems == contexts
    assert stripped == [0]


def test_shared_prefix_stripping_preserves_input_order() -> None:
    shared = "these nine words are shared between the adjacent items here "
    contexts = ["zulu unique tail"] + [shared + str(i) for i in range(7)]
    stems, _ = ii.strip_shared_prefixes(contexts)
    assert stems[0] == "zulu unique tail"
    assert stems[1] == "0"
    assert stems[7] == "6"


def test_empty_input_is_handled() -> None:
    assert ii.strip_shared_prefixes([]) == ([], [])
