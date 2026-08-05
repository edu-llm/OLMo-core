"""
Recall: generation, recognition, and the chance level each is judged against.

This module had **no test at all** until an adversarial pass noticed, and three defects were living in
the gap -- each of which inflated apparent difficulty or deflated the stated chance, and one of which
reported an essentially untrained model as 4.15x above chance on the headline row.

The loss and logits are supplied directly here rather than through a model, so each defect has a test
that fails on the old behaviour and passes on the new.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from factcrowd.corpus import build as B
from factcrowd.measure import recall
from factcrowd.measure.recall import RecallResult

from olmo_core.exceptions import OLMoConfigurationError


def real_corpus(tmp_path, cell_id="smoke_13m_reason"):
    from factcrowd import cells as C

    cell = C.load_cell(f"src/scripts/train/factcrowd/configs/cells/smoke/{cell_id}.yaml")
    resolved = cell.resolve()
    return resolved, B.BuiltCorpus(resolved, tmp_path, split="eval", with_streams=False)


def test_the_chance_level_is_the_whole_value_not_one_position():
    """
    Recognition needs every position right, so chance is the product over positions.

    A four-word value from four four-word pools is one chance in 256, not one in sixteen -- and the
    shipped code took one over the *candidate count*, which is a third answer again.
    """
    one = (np.arange(12),)
    assert recall._chance_of(one) == pytest.approx(1 / 12)
    four = tuple(np.arange(4) for _ in range(4))
    assert recall._chance_of(four) == pytest.approx(1 / 256)
    assert recall._chance_of(()) == 0.0
    assert recall._chance_of((np.arange(0),)) == 0.0


def test_recognition_restricts_each_position_to_its_own_pool():
    """
    A word that only pool 2 contains must not beat the truth at position 1.

    The corpus never puts pool 2's words in position 1, so counting that as a failure is scoring a
    question nobody asked. The shipped code concatenated the pools and took one argmax over the union.
    """
    tokens = np.array([0, 10, 20, 30], dtype=np.int64)
    logits = np.zeros((4, 40))

    class Span:
        start, end = 1, 3

    # Truth at each position; an intruder from the *other* position's pool scores higher.
    position_one = np.array([10, 11, 12])
    position_two = np.array([20, 21, 22])
    logits[0, 21] = 5.0  # predicts position 1 -> 21, which belongs to position 2's pool
    logits[0, 10] = 1.0
    logits[1, 20] = 5.0  # predicts position 2 correctly

    generated, recognised = recall._score_span(logits, tokens, Span(), (position_one, position_two))
    assert generated is False  # unrestricted argmax picked 21
    assert recognised is True  # within position 1's own pool, 10 wins


def test_unreachable_words_do_not_compete():
    """
    On the entropy axis a pool holds the sweep's 256-word union while only a few are ever assigned.

    The rest are never trained and their embeddings are still at init, so letting them compete measures
    initialisation noise rather than recall. `_candidates_per_position` slices to `active_size`.
    """
    tmp = __import__("tempfile").mkdtemp()
    from pathlib import Path

    _, corpus = real_corpus(Path(tmp), cell_id="smoke_13m_entropy")
    pools = {pool.name: pool for pool in corpus.corpus_schema.schema.attributes}
    spec = corpus.corpus_schema.values[0]
    candidates = recall._candidates_per_position(corpus, spec.pool_names)

    assert len(candidates) == len(spec.pool_names)
    for name, allowed in zip(spec.pool_names, candidates):
        pool = pools[name]
        assert allowed.size == pool.active_size < len(pool), (name, allowed.size, len(pool))
        # And the reachable ids are the pool's own prefix, matching how the table samples.
        expected = np.asarray(corpus.vocabulary.pool_token_ids[name])[: pool.active_size]
        np.testing.assert_array_equal(allowed, expected)


def test_the_pooled_chance_is_the_mean_of_the_per_attribute_chances():
    """
    `mean(1/n)`, not `1/mean(n)`. Those differ, and the second understated bioS's pooled chance by 3.8x
    -- which reported an untrained model as 4.15x above chance when it was at chance.
    """
    parts = [
        RecallResult(attribute="a", n_probed=10, n_generated=0, n_recognised=1, chance=1 / 200),
        RecallResult(attribute="b", n_probed=10, n_generated=0, n_recognised=1, chance=1 / 12),
    ]
    correct = float(np.mean([p.chance for p in parts]))
    wrong = 1.0 / float(np.mean([1 / p.chance for p in parts]))
    assert correct == pytest.approx((1 / 200 + 1 / 12) / 2)
    assert wrong < correct / 2, "the two must be far enough apart for this test to mean anything"


def test_generation_and_recognition_are_different_questions():
    """
    A model can fail to produce a value and still pick it out of its pool, which is the distinction the
    pair exists to draw -- "the fact is absent" against "the fact is there but not retrievable".
    """
    tokens = np.array([0, 8], dtype=np.int64)
    logits = np.zeros((2, 20))

    class Span:
        start, end = 1, 2

    # The truth is deliberately *not* the first candidate, and a decoy sits one position later -- so a
    # read from `logits[position]` rather than `logits[position - 1]` picks the wrong word instead of
    # accidentally landing on the truth. Without that, an off-by-one here passed unnoticed.
    logits[0, 15] = 9.0  # unrestricted winner, outside the pool
    logits[0, 8] = 3.0  # best within the pool, at the correct predictor position
    logits[0, 7] = 1.0
    logits[1, 7] = 9.0  # what a shifted read would choose
    generated, recognised = recall._score_span(logits, tokens, Span(), (np.array([7, 8]),))
    assert generated is False and recognised is True


def test_a_result_with_impossible_counts_is_refused():
    with pytest.raises(OLMoConfigurationError, match="nothing was probed"):
        RecallResult(attribute="a", n_probed=0, n_generated=0, n_recognised=0, chance=0.5)
    with pytest.raises(OLMoConfigurationError, match=r"outside \[0, 4\]"):
        RecallResult(attribute="a", n_probed=4, n_generated=5, n_recognised=0, chance=0.5)


def test_the_control_has_no_facts_to_recall():
    """The reasoning-only control has no entity table, so recall is empty rather than zero."""
    from pathlib import Path

    tmp = Path(__import__("tempfile").mkdtemp())
    resolved, corpus = real_corpus(tmp, cell_id="smoke_13m_ctrl")

    loaded = SimpleNamespace(corpus=corpus, resolved=resolved)
    assert corpus.renderer is None
    assert recall.score_recall(loaded, lambda batch: None) == ()
