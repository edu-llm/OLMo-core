"""
Scoring a reasoning endpoint, checked against models whose answers we choose.

The scorer takes a `forward` callable, so every branch is reachable from a stub: a model that is always
right, always wrong, always degenerate, or wrong in a way that only an off-by-one would call right. A
real model would rarely produce any of those cleanly, which is exactly why the stub is the test.

The counts are the point. This programme has four uninterpretable nulls behind it (PRD 1) and each would
have been visible as a count that did not add up, or a score under its own floor.
"""

from typing import Any, Dict

import numpy as np
import pytest
from factcrowd.corpus import render as R
from factcrowd.corpus import tasks as T
from factcrowd.corpus import values as V
from factcrowd.corpus import vocab as Vo
from factcrowd.measure import reasoning
from factcrowd.measure.endpoints import EndpointAccumulator, EndpointResult

from olmo_core.exceptions import OLMoConfigurationError

DOMAIN = ("<facts>", "<mano>", "<compare>")


def never_called(batch):
    """A correctly-shaped forward for the guard tests, which refuse before reaching it."""
    raise AssertionError("forward should not have been reached")


def mano(split="eval", seed=1238):
    """A real Mano task, so the items and answers are the production ones."""
    literals = R.literal_words_of(R.BIOS_TEMPLATES)
    task_words = T.all_required_words((T.ManoTask, T.CompareTask))
    schema = V.bios_schema(reserved=tuple(literals) + tuple(task_words) + Vo.SPECIALS + DOMAIN)
    vocabulary = Vo.Vocabulary.build(
        schema.schema, literal_words=tuple(literals) + tuple(task_words), domain_tokens=DOMAIN
    )
    return T.ManoTask(vocabulary, domain_token="<mano>", length=10, seed=seed, split=split)


def forward_from(task, chooser, n_items):
    """
    A stub forward that returns logits naming whatever `chooser(item)` says, and a known loss.

    Loss is 1 nat at every position, so a one-token answer must come out at exactly `1/ln2` bits -- which
    catches a scorer that summed the wrong span or averaged when it should have summed.
    """
    vocab = task.vocabulary.size

    def forward(batch):
        ce = np.ones(batch.shape, dtype=np.float64)
        logits = np.zeros((*batch.shape, vocab), dtype=np.float64)
        for row in range(batch.shape[0]):
            item = task.item(_index_of(task, batch[row], n_items))
            for offset, word in enumerate(chooser(item)):
                position = item.answer_start + offset
                logits[row, position - 1, task.vocabulary.id_of(word)] = 10.0
        return ce, logits

    return forward


def _index_of(task, row, n_items):
    """Recover which item a row is, so the stub can answer per item."""
    target = row.tobytes()
    for index in range(n_items):
        if task.item(index).tokens.tobytes() == target:
            return index
    raise AssertionError("row is not one of the scored items")


# --- the result type ---------------------------------------------------------------------------------


def test_the_result_reports_counts_a_floor_and_the_subtraction_between_them():
    result = EndpointResult(
        name="mano",
        n_total=200,
        n_correct=60,
        n_degenerate=12,
        n_unparseable=0,
        answer_ce_bits=2.5,
        floor=0.0464,
    )
    assert result.accuracy == pytest.approx(0.30)
    assert result.degenerate_rate == pytest.approx(0.06)
    assert result.above_floor == pytest.approx(100 * (0.30 - 0.0464))
    assert result.headroom == pytest.approx(100 * (1 - 0.0464))
    assert result.summary()["above_floor_pp"] == pytest.approx(25.36, abs=0.01)


def test_a_score_below_its_own_floor_is_visible_rather_than_hidden():
    """
    The failure that killed a previous deduction eval: it scored under 0.500 on a two-way task and
    reported the number anyway. Here the subtraction is a field, so it is negative and obvious.
    """
    result = EndpointResult(
        name="compare",
        n_total=100,
        n_correct=2,
        n_degenerate=1,
        n_unparseable=0,
        answer_ce_bits=9.0,
        floor=0.20,
    )
    assert result.above_floor < 0
    assert float(result.summary()["above_floor_pp"]) < 0  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(n_total=0), "scored 0 items"),
        (dict(n_correct=101), r"outside \[0, 100\]"),
        (dict(n_correct=-1), r"outside \[0, 100\]"),
        (dict(n_unparseable=101), r"outside \[0, 100\]"),
        (dict(floor=1.5), "not a fraction"),
    ],
)
def test_an_impossible_result_is_refused(kwargs, match):
    """A count above its denominator means items were dropped or double-counted."""
    base: Dict[str, Any] = dict(
        name="x",
        n_total=100,
        n_correct=10,
        n_degenerate=5,
        n_unparseable=0,
        answer_ce_bits=1.0,
        floor=0.05,
    )
    with pytest.raises(OLMoConfigurationError, match=match):
        EndpointResult(**{**base, **kwargs})


def test_a_degenerate_answer_can_also_be_correct():
    """
    Degenerate is not a subset of wrong, and treating it as one would understate accuracy.

    The best constant answer is right some of the time -- that is what the floor *is*.
    """
    acc = EndpointAccumulator("mano", floor=0.0464, degenerate_answer=("<n2>",))
    acc.add(predicted=("<n2>",), expected=("<n2>",), ce_bits=1.0)  # both
    acc.add(predicted=("<n2>",), expected=("<n5>",), ce_bits=1.0)  # degenerate only
    acc.add(predicted=("<n7>",), expected=("<n7>",), ce_bits=1.0)  # correct only
    result = acc.result()
    assert (result.n_total, result.n_correct, result.n_degenerate) == (3, 2, 2)


def test_an_unparseable_item_counts_but_is_never_correct():
    acc = EndpointAccumulator("x", floor=0.0, degenerate_answer=None)
    acc.add(predicted=("a",), expected=("a",), ce_bits=1.0, parseable=False)
    result = acc.result()
    assert (result.n_total, result.n_correct, result.n_unparseable) == (1, 0, 1)


# --- the scorer --------------------------------------------------------------------------------------


def test_a_perfect_model_scores_one_and_a_wrong_one_scores_zero():
    task = mano()
    n = 24
    perfect = reasoning.score_reasoning(
        task,
        forward_from(task, lambda item: item.answer, n),
        n_items=n,
        batch_size=8,
        floor=0.0464,
        degenerate_answer=("<n2>",),
    )
    assert perfect.accuracy == 1.0 and perfect.n_total == n

    def always_wrong(item):
        wrong = "<n0>" if item.answer[0] != "<n0>" else "<n1>"
        return (wrong,)

    zero = reasoning.score_reasoning(
        task,
        forward_from(task, always_wrong, n),
        n_items=n,
        batch_size=8,
        floor=0.0464,
        degenerate_answer=("<n2>",),
    )
    assert zero.accuracy == 0.0
    assert zero.above_floor < 0  # below its own floor, and visibly so


def test_the_answer_ce_is_the_answer_span_and_nothing_else():
    """
    The stub returns 1 nat everywhere, so a single-token answer must be exactly 1/ln2 bits.

    Summing the whole sequence instead would give 24x that; averaging over the sequence would give 1/24.
    """
    task = mano()
    n = 8
    result = reasoning.score_reasoning(
        task,
        forward_from(task, lambda item: item.answer, n),
        n_items=n,
        batch_size=4,
        floor=0.0464,
        degenerate_answer=("<n2>",),
    )
    assert result.answer_ce_bits == pytest.approx(1.0 / np.log(2), rel=1e-9)


def test_a_model_that_predicts_one_position_early_scores_zero():
    """
    The off-by-one, from the scorer's side.

    A stub that writes its logits one position late is a model answering the token *after* the answer. If
    the scorer read the wrong offset it would grade that as correct.
    """
    task = mano()
    n = 8
    vocab = task.vocabulary.size

    def shifted(batch):
        ce = np.ones(batch.shape)
        logits = np.zeros((*batch.shape, vocab))
        for row in range(batch.shape[0]):
            item = task.item(_index_of(task, batch[row], n))
            # answer_start rather than answer_start - 1: one position too late.
            logits[row, item.answer_start, task.vocabulary.id_of(item.answer[0])] = 10.0
        return ce, logits

    result = reasoning.score_reasoning(
        task, shifted, n_items=n, batch_size=4, floor=0.0464, degenerate_answer=("<n2>",)
    )
    assert result.accuracy == 0.0


def test_scoring_the_training_split_is_refused():
    """
    The guard that keeps a leaked eval from being reported as reasoning.

    Item keys are domain-separated by split, so a train-split task simply is not the held-out set -- and
    the refusal says which split it was handed rather than scoring it quietly.
    """
    with pytest.raises(OLMoConfigurationError, match="'train' split"):
        reasoning.score_reasoning(mano(split="train"), never_called, n_items=4)


def test_a_forward_that_returns_the_wrong_shape_is_refused():
    task = mano()

    def bad_ce(batch):
        return np.ones((batch.shape[0], batch.shape[1] + 1)), np.zeros(
            (*batch.shape, task.vocabulary.size)
        )

    with pytest.raises(OLMoConfigurationError, match="ce_loss of shape"):
        reasoning.score_reasoning(
            task, bad_ce, n_items=4, batch_size=4, floor=0.0, degenerate_answer=None
        )

    def bad_logits(batch):
        return np.ones(batch.shape), np.zeros((batch.shape[0], 3, task.vocabulary.size))

    with pytest.raises(OLMoConfigurationError, match="logits of shape"):
        reasoning.score_reasoning(
            task, bad_logits, n_items=4, batch_size=4, floor=0.0, degenerate_answer=None
        )


def test_n_items_must_be_positive():
    with pytest.raises(OLMoConfigurationError, match="n_items must be positive"):
        reasoning.score_reasoning(mano(), never_called, n_items=0)


def test_the_same_items_are_scored_at_every_checkpoint():
    """
    The frozen evaluation set: the first n items of the eval split, so a trend across checkpoints is not
    confounded by asking different questions each time.
    """
    a, b = mano(), mano()
    assert [a.item(i).tokens.tobytes() for i in range(50)] == [
        b.item(i).tokens.tobytes() for i in range(50)
    ]


def test_the_measured_floor_is_used_when_none_is_supplied():
    """The floor is measured from the task, never assumed -- and it lands near the uniform 1/23."""
    task = mano()
    n = 8
    result = reasoning.score_reasoning(
        task,
        forward_from(task, lambda item: item.answer, n),
        n_items=n,
        batch_size=8,
        floor_sample=3_000,
    )
    assert 1 / T.MANO_MODULUS - 0.01 < float(result.floor) < 0.06


def test_an_argmax_into_the_vocabulary_padding_is_unparseable_not_a_crash():
    """
    The output layer is wider than the vocabulary, and an untrained model will use the gap.

    `padded_size()` rounds up to a multiple of 128 for the matmul, leaving ids with no word behind them --
    65 on the entropy axis, 31 on the count axis. Indexing the word list without checking raised
    `IndexError` and took the whole scoring job down, which is a worse failure than the one it hid. It is
    also the only way `n_unparseable` becomes non-zero on these endpoints, which is why the field exists.

    Found by scoring the entropy axis after the count axis had passed: the count axis had simply been
    lucky about where its argmax landed.
    """
    task = mano()
    n = 8
    padded = task.vocabulary.padded_size()
    assert padded > task.vocabulary.size, "no padding, so this test would prove nothing"

    def into_the_padding(batch):
        ce = np.ones(batch.shape)
        logits = np.zeros((*batch.shape, padded))
        for row in range(batch.shape[0]):
            item = task.item(_index_of(task, batch[row], n))
            logits[row, item.answer_start - 1, padded - 1] = 10.0  # an id with no word
        return ce, logits

    result = reasoning.score_reasoning(
        task,
        into_the_padding,
        n_items=n,
        batch_size=4,
        floor=0.0464,
        degenerate_answer=("<n2>",),
    )
    assert result.n_unparseable == n
    assert result.n_correct == 0
    assert result.unparseable_rate == 1.0
    # The CE is still recorded, because the answer span's loss is well defined either way.
    assert result.answer_ce_bits > 0


def test_the_answer_ce_is_read_from_the_right_position_under_varying_loss():
    """
    PRD 8.3's continuous endpoint, pinned against a shift.

    An adversarial pass shifted the CE span one earlier and the suite passed, because the stub returned a
    constant nat at every position. `answer_ce_bits` is the quantity that moves before accuracy does, so
    an offset in it is an offset in the only signal a flat grid would show.
    """
    task = mano()
    n = 4
    vocab = task.vocabulary.size
    width = task.tokens_per_item

    def varying(batch):
        # ce[t] = t, so the answer's own position is the only one that gives its expected value.
        ce = np.tile(np.arange(width, dtype=np.float64), (batch.shape[0], 1))
        logits = np.zeros((*batch.shape, vocab))
        for row in range(batch.shape[0]):
            item = task.item(_index_of(task, batch[row], n))
            logits[row, item.answer_start - 1, task.vocabulary.id_of(item.answer[0])] = 10.0
        return ce, logits

    result = reasoning.score_reasoning(
        task, varying, n_items=n, batch_size=n, floor=0.0464, degenerate_answer=("<n2>",)
    )
    # Every item has the same width, so the answer sits at the same position in all of them.
    expected_nats = float(task.item(0).answer_start - 1)
    assert result.answer_ce_bits == pytest.approx(expected_nats / np.log(2), rel=1e-9)
    assert result.accuracy == 1.0
