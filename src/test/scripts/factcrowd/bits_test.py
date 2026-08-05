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

from typing import Any, Dict

import numpy as np
import pytest
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
    assert a.stored_bits_per_entity == 0.0
    assert a.achieved_per_param == 0.0
    assert all(value == 0.0 for value in a.per_entity_bits)


def test_the_capacity_ceiling_is_asserted_and_says_what_to_check():
    """
    R <= R_max, and a violation is reported as a fault rather than a finding.

    The message has to name the likely cause, because "achieved 40 bits/param" reads as a discovery to
    anyone who has not just written the estimator.
    """
    honest = bits.achieved_bits(
        [20.0], n_entities_total=1_000, prior_bits_per_entity=PRIOR, non_embedding_params=PARAMS
    )
    honest.check_against_capacity()  # ~0.002 bits/param, nowhere near the ceiling

    absurd = bits.achieved_bits(
        [0.0],
        n_entities_total=10_000_000,
        prior_bits_per_entity=PRIOR,
        non_embedding_params=1_000,
    )
    with pytest.raises(OLMoConfigurationError, match="exceeds the .* ceiling"):
        absurd.check_against_capacity()
    with pytest.raises(OLMoConfigurationError, match="measurement fault"):
        absurd.check_against_capacity(r_max=rho.R_E_MAX)


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
    assert summary["stored_bits_median"] == pytest.approx(PRIOR - 25.0, abs=0.01)
    assert float(summary["stored_bits_p10"]) < float(summary["stored_bits_median"])  # type: ignore[arg-type]
    assert float(summary["stored_bits_median"]) < float(summary["stored_bits_p90"])  # type: ignore[arg-type]
    assert float(summary["stored_bits_sd"]) > 0  # type: ignore[arg-type]
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
