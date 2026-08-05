"""
The achieved-bit estimator: what the model stored, against what the corpus demanded.

Two things make this measurement easy to get wrong in ways that look fine. It must **sum** over value
tokens, because a mean is independent of how many facts the corpus holds -- the swept quantity. And it
must charge the *value* tokens, which means trusting the renderer's spans rather than re-deriving
positions; the arithmetic that turns a span into loss positions lives in `measure.spans` and is tested
there against a manual cross-entropy.

The capacity assertion is the honesty check. Achieved bits above Physics 3.3's ceiling is not a striking
result, it is a broken measurement -- most likely a biography reading its neighbour because packed
documents are not masked from each other.
"""

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
from factcrowd.corpus import render as R
from factcrowd.corpus import values as V
from factcrowd.ladder import rho
from factcrowd.measure import bits

from olmo_core.exceptions import OLMoConfigurationError

PRIOR = 47.5916
PARAMS = 12_595_456


def test_stored_bits_are_the_prior_minus_what_the_model_still_needs_told():
    """
    The estimator's definition, on numbers chosen so the arithmetic is checkable by hand.

    A residual of 20 bits against a 47.59-bit prior means the model supplies 27.59 of them.
    """
    a = bits.achieved_bits(
        [20.0, 20.0],
        n_entities_total=1_000,
        prior_bits_per_entity=PRIOR,
        non_embedding_params=PARAMS,
    )
    assert a.residual_bits_per_entity == pytest.approx(20.0)
    assert a.stored_bits_per_entity == pytest.approx(PRIOR - 20.0)
    assert a.stored_bits_total == pytest.approx((PRIOR - 20.0) * 1_000)
    assert a.achieved_per_param == pytest.approx((PRIOR - 20.0) * 1_000 / PARAMS)


def test_a_model_worse_than_the_prior_stores_zero_rather_than_negative_bits():
    """
    Early in training the residual exceeds the prior, which is not negative storage.

    Observed for real: a 20-step checkpoint scored a residual of 83-91 bits against a 47.59-bit prior,
    because an untrained model spreads probability over the whole vocabulary rather than over the pool.
    Reporting that as -36 bits stored would be nonsense that propagates into every plot.
    """
    a = bits.achieved_bits(
        [90.0, 85.0], n_entities_total=10, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    # The headline clamps once, in aggregate -- Allen-Zhu's estimator.
    assert a.stored_bits_per_entity == 0.0
    assert a.achieved_per_param == 0.0
    # The distribution stays *signed*, so "worse than uniform about this entity" is visible rather
    # than floored. Clamping per entity and then averaging is a different, upward-biased quantity,
    # and publishing both a clamped distribution and a clamped mean is how two disagreeing numbers
    # got into one summary.
    assert all(value < 0.0 for value in a.per_entity_bits)


def test_the_dataset_bound_is_enforced_and_the_published_estimate_only_warns():
    """
    Two bounds, and only one of them is a theorem.

    ``achieved <= demanded`` is a fact about the dataset -- a model cannot supply more information than
    its data holds -- so a violation is a measurement fault and stops the run.

    Physics 3.3's ~2 bits/parameter is an empirical measurement at 1,000 exposures, not a theorem, and
    three of the six entropy cells *demand* 2.2 to 4.2 bits/param. An earlier revision raised there, which
    would have aborted scoring on its own primary axis and censored the quantity M0 exists to discover.
    """
    demanded = rho.demanded_bits(1_000, PRIOR, name_space=160_000_000)
    honest = bits.achieved_bits(
        [20.0], n_entities_total=1_000, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    honest.check_against_demand(demanded)
    assert honest.capacity_warning() is None

    # Storing more than the corpus contains violates a theorem, so it stops the run.
    absurd = bits.achieved_bits(
        [0.0], n_entities_total=1_000, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    with pytest.raises(OLMoConfigurationError, match="the corpus contains"):
        absurd.check_against_demand(demanded * 0.5)

    # Passing the published estimate warns and carries on.
    dense = bits.achieved_bits(
        [0.0], n_entities_total=1_000_000, prior_bits_per_entity=PRIOR, non_embedding_params=10_000
    )
    assert dense.achieved_per_param > rho.R_E_MAX
    warning = dense.capacity_warning()
    assert warning is not None and "not impossible" in warning
    dense.check_against_demand(rho.demanded_bits(1_000_000, PRIOR, name_space=160_000_000))


def test_the_per_entity_distribution_is_reported_not_just_the_mean():
    """
    PRD 8.1 asks for the distribution, because a mean hides a corpus where a few entities are memorised
    and the rest are not -- which is a different finding from uniform partial storage.
    """
    residuals = [10.0, 20.0, 30.0, 40.0]
    a = bits.achieved_bits(
        residuals, n_entities_total=4, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    summary = a.summary()
    assert summary["signed_bits_median"] == pytest.approx(PRIOR - 25.0, abs=0.01)
    assert float(summary["signed_bits_p10"]) < float(summary["signed_bits_median"])  # type: ignore[arg-type]
    assert float(summary["signed_bits_median"]) < float(summary["signed_bits_p90"])  # type: ignore[arg-type]
    assert float(summary["signed_bits_sd"]) > 0  # type: ignore[arg-type]
    assert summary["bits_is_upper_bound"] is True


def test_value_bits_charge_the_span_and_nothing_around_it():
    """
    The spans come from the renderer, and the loss positions come from `measure.spans`.

    Loss is 1 nat everywhere here, so a two-token span must be exactly 2/ln2 bits. Charging the whole
    sequence, or the span shifted by one, gives a different answer.
    """
    ce = np.ones((2, 20))
    out = bits.value_bits_of_batch(ce, [[(3, 5), (10, 11)]] * 2)
    assert out == [pytest.approx(3.0 / np.log(2))] * 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(value_bits_per_entity=[]), "no per-entity value bits"),
        (dict(n_entities_total=1), "out of"),
        (dict(prior_bits_per_entity=-1.0), "must be positive"),
        (dict(non_embedding_params=0), "must be positive"),
    ],
)
def test_an_impossible_measurement_is_refused(kwargs, match):
    base: Dict[str, Any] = dict(
        value_bits_per_entity=[10.0, 11.0],
        n_entities_total=100,
        prior_bits_per_entity=PRIOR,
        non_embedding_params=PARAMS,
    )
    with pytest.raises(OLMoConfigurationError, match=match):
        bits.achieved_bits(**{**base, **kwargs})


def test_demanded_and_achieved_are_quoted_on_one_definition():
    """
    The comparison reuses `rho.demanded_bits` rather than restating the formula.

    The two halves drifted once already -- two PRD tables quoted the axis on different definitions of the
    name term, which made a slope subtraction between them meaningless.
    """
    a = bits.achieved_bits(
        [20.0], n_entities_total=100_000, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    got = bits.demanded_vs_achieved(
        a, n_entities=100_000, bits_per_entity=PRIOR, name_space=160_000_000
    )
    expected = rho.demanded_bits(100_000, PRIOR, name_space=160_000_000) / PARAMS
    assert got["demanded_bits_per_param"] == pytest.approx(expected)
    assert got["achieved_bits_per_param"] == pytest.approx(a.achieved_per_param)
    assert got["achieved_over_demanded"] == pytest.approx(a.achieved_per_param / expected)


def test_a_span_shifted_by_one_gives_a_different_answer_under_varying_loss():
    """
    The test that the constant-loss version could not be.

    An adversarial pass mutated `bits.value_bits_of_batch` to charge `(start+1, end+1)` and the whole
    suite still passed, because the fixture used `ce = np.ones(...)` -- under constant loss a shift is
    arithmetically invisible. With a varying row it is not, and this is the only thing standing between a
    correct bit count and one that charges each value token's cost to its neighbour.
    """
    ce = np.arange(20, dtype=np.float64).reshape(1, 20)  # ce[t] = t, so every shift is visible
    spans = [[(3, 5), (10, 12)]]
    # Tokens 3,4 cost ce[2],ce[3] = 2+3; tokens 10,11 cost ce[9],ce[10] = 9+10. Total 24 nats.
    assert bits.value_bits_of_batch(ce, spans) == [pytest.approx(24.0 / np.log(2))]
    # Shifted by one in either direction is a different number, which is what makes the test bite.
    assert bits.value_bits_of_batch(ce, [[(4, 6), (11, 13)]]) != bits.value_bits_of_batch(ce, spans)
    assert bits.value_bits_of_batch(ce, [[(2, 4), (9, 11)]]) != bits.value_bits_of_batch(ce, spans)


def test_the_prior_comes_from_the_schema_rather_than_a_constant():
    """
    Doubling the prior in the estimator passed the whole suite, because nothing tied it to a schema.

    The prior is `sum(log2(pool))` over exactly the attributes whose spans are charged -- 47.5916 for
    bioS, 48.0 for the entropy axis at b=8 (six attributes of eight bits). Pinning both catches a prior
    that has drifted from the corpus it is supposed to describe.
    """
    bios = V.bios_schema(reserved=tuple(R.literal_words_of(R.BIOS_TEMPLATES)))
    assert bios.schema.bits_per_entity == pytest.approx(47.5916, abs=1e-4)
    entropy = V.entropy_schema(8)
    assert entropy.schema.bits_per_entity == pytest.approx(48.0)
    # And the count of charged spans equals the count of attributes the prior sums over.
    assert len(bios.values) == len(bios.schema.attributes)
    assert len(entropy.values) * V.ENTROPY_WORDS_PER_VALUE == len(entropy.schema.attributes)


def test_score_checkpoint_charges_the_schemas_own_prior_over_the_renderers_own_spans():
    """
    The driver, not just the arithmetic.

    An adversarial pass doubled the prior inside `score_checkpoint` and the whole suite passed, because
    nothing exercised the driver -- only the pure functions beneath it. The prior it charges must be the
    schema's `bits_per_entity`, and the spans must be the renderer's own, or the achieved-bits axis is
    describing a corpus nobody trained on.

    A stub forward returns zero loss everywhere, so the residual is zero and every entity stores exactly
    the prior. That makes the prior readable straight off the result.
    """
    import tempfile
    from pathlib import Path

    from factcrowd import cells as C
    from factcrowd.corpus import build as B

    cell = C.load_cell("src/scripts/train/factcrowd/configs/cells/smoke/smoke_13m_reason.yaml")
    resolved = cell.resolve()
    with tempfile.TemporaryDirectory() as raw:
        corpus = B.BuiltCorpus(resolved, Path(raw), split="eval", with_streams=False)

        loaded = SimpleNamespace(corpus=corpus, resolved=resolved, cell=cell)

        def perfect(batch):
            # Zero loss: the model needs to be told nothing, so stored == prior exactly.
            return np.zeros(batch.shape), np.zeros((*batch.shape, corpus.vocabulary.size))

        achieved = bits.score_checkpoint(loaded, perfect, n_entities=16, batch_size=8)

    prior = corpus.corpus_schema.schema.bits_per_entity
    assert achieved is not None
    assert achieved.prior_bits_per_entity == pytest.approx(prior)
    assert achieved.residual_bits_per_entity == pytest.approx(0.0)
    # A perfect model stores the whole prior for every entity, so the headline is the prior itself.
    assert achieved.stored_bits_per_entity == pytest.approx(prior)
    assert achieved.n_entities_sampled == 16
    assert achieved.n_entities_total == resolved.n_entities
    assert achieved.is_upper_bound is True
