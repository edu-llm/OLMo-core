"""The fact-block mask, tested as algebra rather than through a model.

`mask_alignment_test.py` checks the tokenized arrays and skips until they exist. These
tests need neither a model nor a corpus, so they run on every commit and pin the one
behaviour the whole experiment rests on: which positions the split arm scores.

If these pass and the arms still train identically, the fault is downstream — in how
the mask reaches the loss — not in the boundary logic.
"""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from train_module import DerivedMaskTrainModule  # noqa: E402

SEP = [7, 8]
EOS = 9


def supervised(ids, sep=SEP, eos=EOS):
    """Call the real method, bypassing __init__ so no model is needed."""
    m = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    m._sep = torch.tensor(sep, dtype=torch.long)
    m.eos_token_id = eos
    x = torch.tensor([ids], dtype=torch.long)
    return [int(v) for v in DerivedMaskTrainModule.supervised_mask(m, x)[0]]


def test_single_document_masks_only_the_fact_block():
    #        facts  separator  target
    ids = [1, 2, 3, 7, 8, 4, 5, 6]
    assert supervised(ids) == [0, 0, 0, 0, 0, 1, 1, 1]


def test_separator_itself_is_not_supervised():
    """The separator is prompt scaffolding, not derivation. Supervision starts after."""
    assert supervised([1, 7, 8, 2])[1:3] == [0, 0]


def test_two_packed_documents_each_get_their_own_block():
    ids = [1, 2, 7, 8, 3, 4, EOS, 5, 7, 8, 6, EOS]
    assert supervised(ids) == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


def test_three_packed_documents():
    ids = [1, 7, 8, 2, EOS, 3, 7, 8, 4, EOS, 5, 7, 8, 6, EOS]
    assert supervised(ids) == [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


def test_a_document_without_a_separator_is_never_supervised():
    """The safe direction: the split arm may lose supervision on a proof, but it must
    never gain supervision on a fact."""
    assert sum(supervised([1, 2, 3, 4, 5])) == 0


def test_a_missing_separator_does_not_leak_into_the_next_document():
    # first document has no separator; the second one does
    ids = [1, 2, EOS, 3, 7, 8, 4, EOS]
    got = supervised(ids)
    assert got[:3] == [0, 0, 0], "unterminated block must not become supervised"
    assert got[6] == 1, "the following document still finds its own boundary"


def test_a_second_separator_inside_the_target_changes_nothing():
    """Only the first separator per document opens supervision; a later one is inert."""
    ids = [1, 7, 8, 2, 7, 8, 3]
    assert supervised(ids) == [0, 0, 0, 1, 1, 1, 1]


def test_dense_and_split_differ_exactly_on_the_fact_block():
    ids = [1, 2, 7, 8, 3, 4]
    mask = supervised(ids)
    assert sum(mask) < len(mask), "split must mask something"
    assert sum(1 for v in mask if v == 0) == 4, "and only the block plus separator"


def test_empty_separator_is_rejected_at_construction():
    with pytest.raises(ValueError, match="separator_ids is empty"):
        DerivedMaskTrainModule.__init__(
            DerivedMaskTrainModule.__new__(DerivedMaskTrainModule),
            arm="split",
            separator_ids=[],
            eos_token_id=EOS,
            pad_token_id=0,
        )


def test_unknown_arm_is_rejected_at_construction():
    with pytest.raises(ValueError, match="arm must be"):
        DerivedMaskTrainModule.__init__(
            DerivedMaskTrainModule.__new__(DerivedMaskTrainModule),
            arm="both",
            separator_ids=SEP,
            eos_token_id=EOS,
            pad_token_id=0,
        )


def test_separator_longer_than_the_window_supervises_nothing():
    """Rather than indexing out of bounds. tokenize_corpus refuses this case up front."""
    assert sum(supervised([1, 2], sep=[7, 8, 9, 10])) == 0


# Env-driven, not a path from my laptop: this file ships in the container image, where
# that directory does not exist. Absent the variable the test skips rather than lying.
VENDORED = Path(os.environ.get("P3_QWEN_TOKENIZER_DIR", "")) if os.environ.get(
    "P3_QWEN_TOKENIZER_DIR"
) else Path("tokenizers/qwen25-vendored")


@pytest.mark.skipif(not VENDORED.exists(), reason="vendored tokenizer not present")
def test_the_search_string_survives_bpe_but_the_full_separator_does_not():
    """The regression that nearly shipped.

    `\\n---\\nGOAL ` is what the corpus contains, but it is NOT what can be searched
    for. BPE merges the trailing space rightward into the goal's first word and the
    leading newline leftward into the fact block, so the full run appears in 0.30% of
    documents — and 0% of metamath, prf2, enigma and isabelle. Searching for it would
    have left the split arm unable to find any boundary, supervising every token and
    quietly becoming a second dense arm with a plausible loss curve.

    The three-token core survives because neither of its edges touches variable text.
    """
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(VENDORED))
    full = tok("\n---\nGOAL ", add_special_tokens=False)["input_ids"]
    core = tok("---\nGOAL", add_special_tokens=False)["input_ids"]
    assert core == [10952, 15513, 969], core
    assert full[1:4] == core, "the core must be a contiguous slice of the full run"

    # a fact block ending in ')' and a goal opening with '|-' is the metamath shape,
    # and it is where both merges fire
    doc = tok("|- ( ph -> ps )\n---\nGOAL |- ph\n1 ax-mp |- ph",
              add_special_tokens=False)["input_ids"]

    def runs(hay, needle):
        n = len(needle)
        return [i for i in range(len(hay) - n + 1) if hay[i : i + n] == needle]

    assert runs(doc, full) == [], "if this ever passes, re-check SEPARATOR_SEARCH"
    assert len(runs(doc, core)) == 1, "the core must appear exactly once"
