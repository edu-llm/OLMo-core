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

import train_module as p3_train_module  # noqa: E402
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.train.train_module import TransformerTrainModule  # noqa: E402
from train_module import (  # noqa: E402
    DerivedMaskTrainModule,
    FixedDivisorTransformerTrainModule,
)

SEP = [7, 8]
EOS = 9


@pytest.mark.parametrize(
    ("dp_world_size", "expected"),
    [(1, 262_144.0), (8, 32_768.0)],
)
def test_fixed_global_divisor_becomes_the_rank_local_divisor(
    monkeypatch, dp_world_size, expected
):
    """FSDP averages rank gradients, so each rank must normalize its own batch."""

    monkeypatch.setattr(
        TransformerTrainModule,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        p3_train_module,
        "_data_parallel_world_size",
        lambda _module: dp_world_size,
    )

    module = FixedDivisorTransformerTrainModule(fixed_loss_div_factor=262_144)

    assert module.global_fixed_loss_div_factor == 262_144.0
    assert module.fixed_loss_div_factor == expected


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


def test_qwen_eos_is_supervised_but_repeated_eos_padding_is_not():
    """Qwen uses id 151643 for both EOS and pad.

    Masking every token equal to pad_token_id also masks every genuine proof-ending
    EOS, so neither arm learns to stop. In packed rows, padding starts after the
    final real EOS and is therefore the repeated-EOS tail.
    """
    m = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    m.eos_token_id = EOS
    m.pad_token_id = EOS
    ids = torch.tensor([[1, 7, 8, 2, EOS, 3, 7, 8, 4, EOS, EOS, EOS]])
    got = DerivedMaskTrainModule.padding_mask(m, ids)[0].tolist()
    assert got == [False] * 10 + [True, True]


def test_distinct_pad_id_masks_every_occurrence():
    m = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    m.eos_token_id = EOS
    m.pad_token_id = 0
    ids = torch.tensor([[1, 7, 8, 2, EOS, 0, 0]])
    assert DerivedMaskTrainModule.padding_mask(m, ids)[0].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def label_supervision(ids, arm, sep=SEP, eos=EOS, pad=EOS):
    m = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    m._sep = torch.tensor(sep, dtype=torch.long)
    m.eos_token_id = eos
    m.pad_token_id = pad
    m.arm = arm
    x = torch.tensor([ids], dtype=torch.long)
    return [int(v) for v in DerivedMaskTrainModule.label_supervision_mask(m, x)[0]]


def test_split_label_mask_is_shifted_to_prediction_targets():
    """`labels[i]` predicts `input_ids[i+1]`; mask the target, not input i."""
    # tokens: fact, separator, separator, goal, EOS, pad, pad
    ids = [1, 7, 8, 2, EOS, EOS, EOS]
    # Score first goal token (label at final separator) and real EOS (label at goal).
    assert label_supervision(ids, "split") == [0, 0, 1, 1, 0, 0, 0]


def test_packed_split_label_mask_scores_each_goal_and_real_eos():
    ids = [1, 7, 8, 2, EOS, 3, 7, 8, 4, EOS, EOS]
    assert label_supervision(ids, "split") == [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]


def test_dense_label_mask_scores_all_real_targets_but_not_padding():
    ids = [1, 7, 8, 2, EOS, 3, 7, 8, 4, EOS, EOS]
    # Every real token after the first is a target, including both genuine EOS.
    # The EOS at position 10 is padding, so position 9 must not predict it.
    assert label_supervision(ids, "dense") == [1] * 9 + [0, 0]


def test_split_masks_local_assumptions_while_dense_scores_them():
    # tokens: fact, local-header, local-$e, separator, separator, goal, EOS, pad
    ids = [1, 2, 3, 7, 8, 4, EOS, EOS]
    split = label_supervision(ids, "split")
    dense = label_supervision(ids, "dense")

    # Labels 0-2 predict the pre-separator prompt tokens, including both local-
    # assumption tokens. Both arms receive those tokens as input, but only dense
    # receives loss on predicting them.
    assert split[:3] == [0, 0, 0]
    assert dense[:3] == [1, 1, 1]
    # Both arms score the first goal token and the genuine proof-ending EOS.
    assert split[4:6] == dense[4:6] == [1, 1]


def test_model_forward_masks_shifted_labels_and_records_true_fraction(monkeypatch):
    module = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    module._sep = torch.tensor(SEP, dtype=torch.long)
    module.eos_token_id = EOS
    module.pad_token_id = EOS
    module.arm = "split"
    recorded = {}
    forwarded = {}

    def record_metric(name, value, _reduce):
        recorded[name] = float(value)

    def fake_parent(_self, input_ids, labels=None, **kwargs):
        forwarded["labels"] = labels.clone()
        return None, None, None, None

    module.record_metric = record_metric
    monkeypatch.setattr(
        FixedDivisorTransformerTrainModule, "model_forward", fake_parent
    )
    ids = torch.tensor([[1, 7, 8, 2, EOS, EOS, EOS]])
    shifted = get_labels({"input_ids": ids})
    DerivedMaskTrainModule.model_forward(
        module, ids, labels=shifted, loss_div_factor=7
    )

    assert forwarded["labels"].tolist() == [
        [-100, -100, 2, EOS, -100, -100, -100]
    ]
    assert recorded["train/supervised token fraction"] == pytest.approx(2 / 7)


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


def test_real_constructor_keeps_separator_on_the_train_modules_device(monkeypatch):
    """TrainModule is not nn.Module, so register_buffer is unavailable.

    Keep this as a constructor test rather than another ``__new__`` algebra test:
    the latter let an AttributeError survive every local gate until a paid GPU job.
    """

    def fake_parent_init(self, *args, **kwargs):
        del args, kwargs
        self.device = torch.device("cpu")

    monkeypatch.setattr(
        FixedDivisorTransformerTrainModule,
        "__init__",
        fake_parent_init,
    )
    module = DerivedMaskTrainModule(
        arm="split",
        separator_ids=SEP,
        eos_token_id=EOS,
        pad_token_id=EOS,
        fixed_loss_div_factor=8,
    )

    assert module._sep.tolist() == SEP
    assert module._sep.device == module.device


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
