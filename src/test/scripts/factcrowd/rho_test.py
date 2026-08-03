"""
What the :math:`\\rho` arithmetic guarantees, and what it refuses.

The refusals carry most of the weight here. :math:`\\rho` is the experiment's independent variable,
and the failure that matters is not a crash but a cell that runs happily at a :math:`\\rho` other
than the one on its label -- so the tests that assert something *raises* are the ones protecting
the result.
"""

import math

import pytest
from factcrowd.ladder import rho as R

from olmo_core.exceptions import OLMoConfigurationError

# The ladder from PRD.md section 7.1: non-embedding parameters at d_model 256/384/576/768,
# depth 12. Recomputed from a built model by sizes.py; hard-coded here so a drift in either
# shows up as a test failure rather than as two modules quietly agreeing on a new number.
LADDER = {
    "13M": 12_595_456,
    "28M": 28_330_368,
    "64M": 63_729_216,
    "113M": 113_283_840,
}

GRID_RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)

# Shorthand for the degenerate-input cases at the bottom, which need one valid model size
# and one valid bits-per-entity to vary a single argument away from.
_P = LADDER["28M"]
_BITS = R.BIOS_BITS_PER_ENTITY


def test_bios_schema_bits_match_the_published_value():
    """
    A declared set of pools reproduces Physics 3.3's 47.6 bits/person.

    Physics 3.3 publishes the total, not the factorisation, so the pools below are our
    reconstruction of it: four categorical attributes at 200/300/100/263 choices plus a birth date
    over 12 months x 28 days x 400 years. That comes to 47.592 bits, which pins the published
    figure to within 0.01 of a bit -- close enough that this is the schema they used, and in any
    case it is the schema *we* use, which is what makes our bit-counts comparable to theirs.

    The number to distrust is 47.6 as a constant. Bits per entity is computed from whatever pools
    are actually in use, and this test exists so that a change to those pools shows up as a broken
    comparison rather than as a silently different x-axis.

    .. note::
        These pool sizes move to ``factcrowd.corpus.entities`` as the default schema once that
        module lands, and this test should import them from there rather than restate them.
    """
    categorical_pools = [200, 300, 100, 263]
    birth_date_choices = 12 * 28 * 400  # month x day x year-range
    bits = sum(math.log2(n) for n in categorical_pools) + math.log2(birth_date_choices)

    assert bits == pytest.approx(R.BIOS_BITS_PER_ENTITY, abs=0.01)


@pytest.mark.parametrize("params", LADDER.values(), ids=list(LADDER))
@pytest.mark.parametrize("rho", GRID_RHOS)
def test_solve_and_rho_of_round_trip(params, rho):
    """
    Every grid cell round-trips: solve to an entity count, invert, land back on the same rho.

    Integer rounding of ``n_entities`` is the only loss, and at grid sizes it is far under the 1%
    tolerance :func:`factcrowd.ladder.rho.check` allows.
    """
    size = R.solve(params, rho, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
    realised = R.rho_of(params, size.n_entities, bits_per_entity=R.BIOS_BITS_PER_ENTITY)

    assert realised == pytest.approx(rho, rel=1e-4)
    assert size.achieved_rho == pytest.approx(realised)
    assert size.achieved_rho == pytest.approx(rho, rel=1e-4)


@pytest.mark.parametrize("params", LADDER.values(), ids=list(LADDER))
def test_entity_count_is_proportional_to_rho(params):
    """
    Doubling rho doubles the entity count, which is what makes the sweep a sweep of one variable.
    """
    counts = [
        R.solve(params, rho, bits_per_entity=R.BIOS_BITS_PER_ENTITY).n_entities for rho in GRID_RHOS
    ]
    for smaller, larger in zip(counts, counts[1:]):
        assert larger == pytest.approx(2 * smaller, rel=1e-4)


def test_fact_tokens_follow_entities_exposures_and_length():
    """``fact_tokens`` is exactly the product, so a budget can be read off it."""
    size = R.solve(LADDER["28M"], 1.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
    assert size.fact_tokens == size.n_entities * R.EXPOSURES * R.TOKENS_PER_BIO


def test_the_ladder_lands_where_the_prd_says():
    """
    The PRD's published entity counts and token budgets are reproducible from this module.

    If this fails, either the ladder moved or the arithmetic did, and section 7.1 of the PRD is
    now wrong -- which matters because the compute budget is derived from these token counts.
    """
    expected = {  # row -> (n_entities, fact tokens in billions), PRD section 7.1
        "13M": (318_000, 6.35),
        "28M": (714_000, 14.28),
        "64M": (1_607_000, 32.13),
        "113M": (2_856_000, 57.12),
    }
    for row, (n_entities, fact_tokens_b) in expected.items():
        size = R.solve(LADDER[row], 1.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
        assert size.n_entities == pytest.approx(n_entities, rel=0.005), row
        assert size.fact_tokens / 1e9 == pytest.approx(fact_tokens_b, rel=0.005), row


def test_check_passes_on_a_consistent_cell():
    """The happy path, so the failure cases below are known to be about disagreement."""
    size = R.solve(LADDER["64M"], 2.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
    R.check(
        LADDER["64M"], 2.0, size.n_entities, bits_per_entity=R.BIOS_BITS_PER_ENTITY, label="d576"
    )


@pytest.mark.parametrize("wrong_factor", [1.02, 0.98, 2.0, 0.5])
def test_check_raises_when_rho_and_entity_count_disagree(wrong_factor):
    """
    A cell whose entity count was chosen independently of its rho is refused.

    Includes the 2% cases, not just the 2x ones: a cell 2% off its label still lands at the wrong
    x on the trend plot, and 2% is the sort of error a hand-edited config produces.
    """
    size = R.solve(LADDER["28M"], 1.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
    with pytest.raises(OLMoConfigurationError, match="disagree"):
        R.check(
            LADDER["28M"],
            1.0,
            round(size.n_entities * wrong_factor),
            bits_per_entity=R.BIOS_BITS_PER_ENTITY,
        )


def test_check_error_names_the_cell_and_both_numbers():
    """
    The message has to be actionable, because it fires in front of somebody editing a config.
    """
    with pytest.raises(OLMoConfigurationError) as excinfo:
        R.check(
            LADDER["13M"], 1.0, 999_999, bits_per_entity=R.BIOS_BITS_PER_ENTITY, label="d256_rho1"
        )
    message = str(excinfo.value)
    assert "d256_rho1" in message
    assert "999,999" in message
    assert "solve()" in message


def test_capacity_excludes_embeddings_by_taking_the_count_it_is_given():
    """
    Capacity is whatever parameter count the caller passes, so the exclusion is the caller's job.

    Asserted because it is the one place the design could silently go wrong without raising: a
    total-parameter count at d=256 would overstate capacity by 65%, and every entity count with it.
    """
    non_embedding = LADDER["13M"]
    with_embeddings = non_embedding + 32_000 * 256
    assert R.capacity_bits(with_embeddings, R.R_E_AT_200_EXPOSURES) > 1.6 * R.capacity_bits(
        non_embedding, R.R_E_AT_200_EXPOSURES
    )


def test_r_e_resolves_only_for_the_declared_exposure_count():
    """
    Changing exposures without restating capacity per parameter is refused, not guessed.

    Capacity per parameter is a function of exposures, so silently reusing 1.2 at, say, 50
    exposures would put every cell at an unknown rho -- the exact confound a previous sweep hit.
    """
    assert R.resolve_r_e(R.EXPOSURES) == R.R_E_AT_200_EXPOSURES
    assert R.resolve_r_e(50, 0.4) == 0.4

    with pytest.raises(OLMoConfigurationError, match="no declared capacity constant"):
        R.resolve_r_e(50)


def test_r_e_error_offers_the_loglinear_alternative():
    """The refusal above is only useful if it tells you what to pass instead."""
    with pytest.raises(OLMoConfigurationError) as excinfo:
        R.solve(LADDER["28M"], 1.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY, exposures=1000)
    assert "r_e_loglinear(1000)" in str(excinfo.value)


def test_loglinear_interpolation_hits_the_published_anchors():
    """
    1.0 bits/param at 100 exposures and 2.0 at 1000, per Physics 3.3.

    At 200 it returns 1.301 against the programme's declared 1.2. Asserted so the 8% disagreement
    documented on ``R_E_AT_200_EXPOSURES`` stays a known quantity rather than a surprise.
    """
    assert R.r_e_loglinear(100) == pytest.approx(1.0)
    assert R.r_e_loglinear(1000) == pytest.approx(2.0)
    assert R.r_e_loglinear(200) == pytest.approx(1.301, abs=0.001)
    assert R.R_E_BAND[0] <= R.R_E_AT_200_EXPOSURES <= R.R_E_BAND[1]


def test_r_e_choice_moves_entity_counts_but_not_the_shape_of_the_sweep():
    """
    The capacity constant scales every cell together, so it cannot manufacture or hide a trend.

    This is the claim that lets us proceed on an interpolated constant while reporting against
    achieved R(F): the ratios between cells are invariant to it, and only the axis labels move.
    """
    ratios = {}
    for r_e in (R.R_E_BAND[0], R.R_E_AT_200_EXPOSURES, R.R_E_BAND[1]):
        counts = [
            R.solve(LADDER["28M"], rho, bits_per_entity=R.BIOS_BITS_PER_ENTITY, r_e=r_e).n_entities
            for rho in GRID_RHOS
        ]
        ratios[r_e] = [c / counts[0] for c in counts]

    # 1e-4 rather than exact: n_entities is an integer, so each ratio carries a rounding residual
    # of order 1/n. Agreement to 0.01% across the band is the claim, and it is not a tight fit.
    reference = ratios[R.R_E_AT_200_EXPOSURES]
    for r_e, observed in ratios.items():
        assert observed == pytest.approx(reference, rel=1e-4), r_e


def test_achieved_r_is_measured_bits_per_parameter():
    """
    The measured x-axis. A model at the 2 bits/param ceiling reads 2.0 regardless of its label.
    """
    params = LADDER["64M"]
    assert R.achieved_r(2.0 * params, params) == pytest.approx(2.0)
    assert R.achieved_r(0.0, params) == 0.0


def test_achieved_r_can_fall_short_of_demanded_rho():
    """
    Oversubscription is the whole experiment: demanded 4.0 bits/param, stored 2.0, so recall halves.

    Encodes prediction P2's mechanism as an arithmetic relation, so a bit-counter that returned
    demanded bits instead of achieved ones would fail here rather than in the analysis.
    """
    params = LADDER["28M"]
    demanded = R.solve(params, 4.0, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
    demanded_r = R.demanded_bits(demanded.n_entities, R.BIOS_BITS_PER_ENTITY) / params
    assert demanded_r == pytest.approx(4.0 * R.R_E_AT_200_EXPOSURES, rel=1e-3)

    stored = R.achieved_r(2.0 * params, params)
    assert stored < demanded_r
    assert stored / demanded_r == pytest.approx(0.417, abs=0.01)


@pytest.mark.parametrize(
    "bad_call, match",
    [
        (lambda: R.solve(_P, 0.0, bits_per_entity=_BITS), "'rho' must be positive"),
        (lambda: R.solve(_P, -1.0, bits_per_entity=_BITS), "'rho' must be positive"),
        (lambda: R.solve(_P, 1.0, bits_per_entity=0.0), "'bits_per_entity' must be positive"),
        (lambda: R.solve(_P, 1.0, bits_per_entity=-1.0), "'bits_per_entity' must be positive"),
        (
            lambda: R.solve(_P, 1.0, bits_per_entity=_BITS, tokens_per_bio=0),
            "'tokens_per_bio' must be positive",
        ),
        (
            lambda: R.solve(_P, 1.0, bits_per_entity=_BITS, exposures=0, r_e=1.2),
            "'exposures' must be positive",
        ),
        (lambda: R.solve(0, 1.0, bits_per_entity=_BITS), "'non_embedding_params' must be positive"),
    ],
    ids=[
        "rho-zero",
        "rho-negative",
        "bits-zero",
        "bits-negative",
        "tokens-per-bio-zero",
        "exposures-zero",
        "params-zero",
    ],
)
def test_solve_refuses_degenerate_inputs(bad_call, match):
    """
    Each of these would otherwise produce a corpus of zero or negative size.

    Written as explicit calls rather than a splatted kwargs dict so that each case type-checks as
    the signature it is actually exercising.
    """
    with pytest.raises(OLMoConfigurationError, match=match):
        bad_call()


def test_solve_refuses_a_cell_too_small_to_be_a_corpus():
    """A rho small enough to round to zero entities is a config error, not an empty corpus."""
    with pytest.raises(OLMoConfigurationError, match="too small to be a corpus"):
        R.solve(LADDER["13M"], 1e-9, bits_per_entity=R.BIOS_BITS_PER_ENTITY)
