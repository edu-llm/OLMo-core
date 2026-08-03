"""
What the demand arithmetic guarantees, and what it refuses.

The refusals carry most of the weight. Demanded bits per parameter is the experiment's independent
variable, and the failure that matters is not a crash but a cell that runs happily at a demand other
than the one on its label -- so the tests that assert something *raises* are the ones protecting the
result.

Three properties here were bugs in an earlier revision and are now pinned: demand includes the name
term and so is nonlinear in N; capacity is reported on both parameter bases because they diverge
monotonically with model size; and R_E appears only in the interpretive transform, never in placing
a cell.
"""

import math

import pytest
from factcrowd.ladder import rho as R

from olmo_core.exceptions import OLMoConfigurationError

# The ladder from PRD.md section 7.1: non-embedding parameters at d_model 256/384/576/768, depth 12,
# and the total with a tied 32k embedding table. Restated rather than imported so a drift in either
# module shows up as a failure rather than as two modules agreeing on a new number.
LADDER = {
    "13M": (12_595_456, 12_595_456 + 32_000 * 256),
    "28M": (28_330_368, 28_330_368 + 32_000 * 384),
    "64M": (63_729_216, 63_729_216 + 32_000 * 576),
    "113M": (113_283_840, 113_283_840 + 32_000 * 768),
}

NAME_SPACE = 400 * 400 * 1000
"""The default schema's name universe: 160M distinct names."""

GRID_DEMANDS = (0.30, 0.60, 1.20, 2.40, 4.80)
"""Demand levels standing in for rho = 0.25/0.5/1/2/4 at R_E = 1.2."""

_P = LADDER["28M"][0]
_BITS = R.BIOS_BITS_PER_ENTITY


# --- the name term, which an earlier revision omitted ---------------------------------------------


def test_name_term_is_zero_when_every_available_name_is_used():
    """``log2(N0/N)`` vanishes at N = N0: if you use every name, which ones exist is not news."""
    assert R.name_bits(1000, 1000) == pytest.approx(0.0)


def test_name_term_grows_as_the_selection_gets_sparser():
    """A name drawn from a wider universe is more surprising, which is the right shape."""
    dense = R.name_bits(1_000_000, 2_000_000)
    sparse = R.name_bits(1_000_000, 1_000_000_000)
    assert sparse > dense > 0


def test_name_term_matches_the_magnitudes_the_prd_quotes():
    """
    +16% of attribute demand at 714k entities and +10% at 6.4M, against a 160M name space.

    These are the figures section 3 uses to argue the term bends the trend rather than shifting it,
    so they are checked rather than asserted in prose.
    """
    for n_entities, expected_share in ((714_331, 0.164), (6_430_000, 0.097)):
        names = R.name_bits(n_entities, NAME_SPACE)
        attributes = n_entities * _BITS
        assert names / attributes == pytest.approx(expected_share, abs=0.003), n_entities


def test_the_name_term_shrinks_as_a_share_as_the_corpus_grows():
    """
    Which is why it changes the *shape* of the trend: it is not a constant offset.

    A term that shrank proportionally would only relabel the axis. This one does not, so omitting it
    distorts the slope being measured.
    """
    shares = []
    for n_entities in (100_000, 714_331, 3_000_000, 6_430_000):
        names = R.name_bits(n_entities, NAME_SPACE)
        shares.append(names / (n_entities * _BITS))
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > 1.3 * shares[-1]


def test_omitting_the_name_space_drops_the_term():
    """The attribute-only figure is still reachable, for reproducing a published number."""
    with_names = R.demanded_bits(714_331, _BITS, name_space=NAME_SPACE)
    without = R.demanded_bits(714_331, _BITS)
    assert without == pytest.approx(714_331 * _BITS)
    assert with_names > without


def test_name_term_refuses_more_entities_than_names():
    """Beyond the name space two entities share a key and the corpus contradicts itself."""
    with pytest.raises(OLMoConfigurationError, match="exceeds a name space"):
        R.name_bits(1001, 1000)


# --- both parameter bases -------------------------------------------------------------------------


def test_the_two_parameter_bases_diverge_monotonically_with_model_size():
    """
    1.65x at 13M falling to 1.22x at 113M, so a cross-size comparison must say which basis it is.

    This is the finding that makes reporting both non-negotiable: read one basis as the other and the
    error itself looks like a trend across model size.
    """
    ratios = []
    for label, (non_emb, total) in LADDER.items():
        d = R.demand(
            714_331,
            bits_per_entity=_BITS,
            non_embedding_params=non_emb,
            total_params=total,
            name_space=NAME_SPACE,
        )
        ratios.append(d.per_non_embedding_param / d.per_total_param)

    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] == pytest.approx(1.650, abs=0.01)
    assert ratios[-1] == pytest.approx(1.217, abs=0.01)


def test_demand_decomposes_into_attribute_and_name_bits():
    """The parts sum to the whole, so a reader can see how much of demand is which."""
    d = R.demand(
        714_331,
        bits_per_entity=_BITS,
        non_embedding_params=LADDER["28M"][0],
        total_params=LADDER["28M"][1],
        name_space=NAME_SPACE,
    )
    assert d.attribute_bits + d.name_bits == pytest.approx(d.bits)
    assert d.per_non_embedding_param == pytest.approx(d.bits / LADDER["28M"][0])
    assert d.per_total_param == pytest.approx(d.bits / LADDER["28M"][1])


def test_demand_refuses_parameter_counts_from_different_models():
    """A total below a non-embedding count cannot describe one model, and would invert the bases."""
    with pytest.raises(OLMoConfigurationError, match="cannot describe the same model"):
        R.demand(
            1000,
            bits_per_entity=_BITS,
            non_embedding_params=20_000_000,
            total_params=10_000_000,
        )


# --- solve: nonlinear, so it bisects --------------------------------------------------------------


@pytest.mark.parametrize("label", list(LADDER))
@pytest.mark.parametrize("target", GRID_DEMANDS)
def test_solve_round_trips_through_the_name_term(label, target):
    """
    Every grid cell round-trips: solve to an entity count, invert, land back on the same demand.

    With the name term the mapping is nonlinear, so this is a real check on the bisection rather
    than on a division.
    """
    non_emb = LADDER[label][0]
    size = R.solve(non_emb, target, bits_per_entity=_BITS, name_space=NAME_SPACE)
    realised = R.demand_per_param(
        size.n_entities, non_emb, bits_per_entity=_BITS, name_space=NAME_SPACE
    )

    assert realised == pytest.approx(target, rel=1e-5)
    assert size.achieved_demand_per_param == pytest.approx(realised)


def test_solve_finds_the_closest_integer_not_merely_a_bracketing_one():
    """
    The bisection lands on the smallest N at or above target; the neighbour below is often closer.

    Asserted because an off-by-one here is invisible at grid scale and wrong at small scale, and the
    smoke-run configs are small.
    """
    for target in (0.05, 0.3, 1.2, 4.8):
        size = R.solve(_P, target, bits_per_entity=_BITS, name_space=NAME_SPACE)
        here = abs(
            R.demand_per_param(size.n_entities, _P, bits_per_entity=_BITS, name_space=NAME_SPACE)
            - target
        )
        for neighbour in (size.n_entities - 1, size.n_entities + 1):
            if neighbour < 1:
                continue
            there = abs(
                R.demand_per_param(neighbour, _P, bits_per_entity=_BITS, name_space=NAME_SPACE)
                - target
            )
            assert here <= there, (target, size.n_entities, neighbour)


def test_solve_without_the_name_term_is_a_plain_division():
    """The linear path still works, and agrees with hand arithmetic."""
    size = R.solve(_P, 1.2, bits_per_entity=_BITS)
    assert size.n_entities == round(1.2 * _P / _BITS)


def test_the_name_term_lowers_the_entity_count_for_a_given_demand():
    """
    Because each entity now carries more bits, fewer are needed -- by 8-23%, shrinking as demand
    rises.

    Direction matters: the term *adds* demand at fixed N, so at fixed demand it *removes* entities. A
    sign error here would move every cell the wrong way.
    """
    with_names = R.solve(_P, 1.2, bits_per_entity=_BITS, name_space=NAME_SPACE).n_entities
    without = R.solve(_P, 1.2, bits_per_entity=_BITS).n_entities
    assert with_names < without
    assert 0.80 < with_names / without < 0.95


def test_demand_is_monotone_in_entity_count_which_is_what_makes_bisection_valid():
    """
    The bisection assumes strict monotonicity over ``[1, name_space]``. Checked over six decades.

    The derivative is ``log2(N0/N) - 1/ln2 + bits_per_entity``, positive for any sane bits/entity,
    but a schema with very few bits per entity and a vast name space could break it -- so the
    property is tested rather than trusted.
    """
    previous = 0.0
    for exponent in range(0, 8):
        n_entities = 10**exponent
        bits = R.demanded_bits(n_entities, _BITS, name_space=NAME_SPACE)
        assert bits > previous, n_entities
        previous = bits


def test_solve_refuses_a_demand_the_name_space_cannot_carry():
    """
    An unreachable target is a config error, not a silently clamped corpus.

    Fires when the demand needs more entities than there are names, which is the ceiling the name
    pools impose.
    """
    with pytest.raises(OLMoConfigurationError, match="more than the"):
        R.solve(_P, 10_000.0, bits_per_entity=_BITS, name_space=1000)


def test_fact_tokens_follow_entities_exposures_and_length():
    """``fact_tokens`` is exactly the product, so a budget can be read off it."""
    size = R.solve(_P, 1.2, bits_per_entity=_BITS, name_space=NAME_SPACE)
    assert size.fact_tokens == size.n_entities * R.EXPOSURES * R.TOKENS_PER_BIO


def test_entity_count_is_close_to_proportional_in_demand():
    """
    Doubling demand roughly doubles the entity count -- the name term makes it slightly super-linear.

    Not exactly 2x, and that is the point: the deviation is the nonlinearity the term introduces, and
    it is bounded rather than absent.
    """
    counts = [
        R.solve(_P, d, bits_per_entity=_BITS, name_space=NAME_SPACE).n_entities
        for d in GRID_DEMANDS
    ]
    for smaller, larger in zip(counts, counts[1:]):
        assert 2.0 < larger / smaller < 2.2


# --- check ----------------------------------------------------------------------------------------


def test_check_passes_on_a_consistent_cell():
    """The happy path, so the failure cases below are known to be about disagreement."""
    size = R.solve(LADDER["64M"][0], 2.4, bits_per_entity=_BITS, name_space=NAME_SPACE)
    R.check(
        LADDER["64M"][0],
        2.4,
        size.n_entities,
        bits_per_entity=_BITS,
        name_space=NAME_SPACE,
        label="d576",
    )


@pytest.mark.parametrize("wrong_factor", [1.02, 0.98, 2.0, 0.5])
def test_check_raises_when_demand_and_entity_count_disagree(wrong_factor):
    """
    Includes the 2% cases, not just the 2x ones: 2% is what a hand-edited config produces, and it
    still lands the cell at the wrong x.
    """
    size = R.solve(_P, 1.2, bits_per_entity=_BITS, name_space=NAME_SPACE)
    with pytest.raises(OLMoConfigurationError, match="disagree"):
        R.check(
            _P,
            1.2,
            round(size.n_entities * wrong_factor),
            bits_per_entity=_BITS,
            name_space=NAME_SPACE,
        )


def test_check_would_pass_a_cell_that_omitted_the_name_term_only_if_told_to():
    """
    The name term is part of the contract: a count derived without it fails a check made with it.

    This is the guard that stops the two halves of the pipeline disagreeing about what a cell means.
    """
    without = R.solve(_P, 1.2, bits_per_entity=_BITS).n_entities
    with pytest.raises(OLMoConfigurationError, match="disagree"):
        R.check(_P, 1.2, without, bits_per_entity=_BITS, name_space=NAME_SPACE)
    R.check(_P, 1.2, without, bits_per_entity=_BITS)


def test_check_error_names_the_cell_and_both_numbers():
    """The message fires in front of somebody editing a config, so it has to be actionable."""
    with pytest.raises(OLMoConfigurationError) as excinfo:
        R.check(
            LADDER["13M"][0],
            1.2,
            999_999,
            bits_per_entity=_BITS,
            name_space=NAME_SPACE,
            label="d256_b8",
        )
    message = str(excinfo.value)
    assert "d256_b8" in message
    assert "999,999" in message
    assert "solve()" in message


# --- R_E is interpretation only -------------------------------------------------------------------


def test_rho_is_a_presentation_transform_on_the_demand():
    """
    rho = demand / R_E, and nothing places a cell by it. Getting R_E wrong relabels, never bends.

    Asserted as the ratio invariance: across the whole R_E band the *relative* positions of the grid
    cells are identical, which is why an interpolated constant is tolerable at all.
    """
    for r_e in (R.R_E_BAND[0], R.R_E_AT_200_EXPOSURES, R.R_E_BAND[1]):
        rhos = [R.rho_from_demand(d, r_e) for d in GRID_DEMANDS]
        ratios = [x / rhos[0] for x in rhos]
        assert ratios == pytest.approx([d / GRID_DEMANDS[0] for d in GRID_DEMANDS])


def test_the_bios_anchor_lands_on_rho_one():
    """
    A demand of 1.2 bits/param reads as rho = 1 at the declared constant, which is the whole point
    of the scale.
    """
    assert R.rho_from_demand(1.2) == pytest.approx(1.0)
    assert R.rho_from_demand(1.2, 1.0) == pytest.approx(1.2)


def test_r_e_band_brackets_the_declared_value_and_is_wide():
    """
    The band is 0.9-1.4, widened at the bottom for the SwiGLU capacity penalty.

    A 1.56x span, which is the honest uncertainty on where the knee sits.
    """
    low, high = R.R_E_BAND
    assert low <= R.R_E_AT_200_EXPOSURES <= high
    assert high / low > 1.5


def test_r_e_resolves_only_for_the_declared_exposure_count():
    """Changing exposures without restating capacity per parameter is refused, not guessed."""
    assert R.resolve_r_e(R.EXPOSURES) == R.R_E_AT_200_EXPOSURES
    assert R.resolve_r_e(50, 0.4) == 0.4
    with pytest.raises(OLMoConfigurationError, match="no declared capacity constant"):
        R.resolve_r_e(50)


def test_r_e_error_offers_the_loglinear_alternative():
    """The refusal is only useful if it says what to pass instead."""
    with pytest.raises(OLMoConfigurationError) as excinfo:
        R.resolve_r_e(1000)
    assert "r_e_loglinear(1000)" in str(excinfo.value)


def test_loglinear_interpolation_hits_the_published_anchors():
    """1.0 at 100 exposures and 2.0 at 1000, and 1.301 at 200 against the declared 1.2."""
    assert R.r_e_loglinear(100) == pytest.approx(1.0)
    assert R.r_e_loglinear(1000) == pytest.approx(2.0)
    assert R.r_e_loglinear(200) == pytest.approx(1.301, abs=0.001)


# --- achieved R(F) --------------------------------------------------------------------------------


def test_achieved_r_is_measured_bits_per_parameter():
    """The measured x-axis. A model at 2 bits/param reads 2.0 whatever its label says."""
    params = LADDER["64M"][0]
    assert R.achieved_r(2.0 * params, params) == pytest.approx(2.0)
    assert R.achieved_r(0.0, params) == 0.0


def test_achieved_r_can_fall_short_of_demand_which_is_the_experiment():
    """
    Demanded 4.8 bits/param, stored 2.0, so recall is capacity over demand.

    Encodes P2's mechanism as an arithmetic relation, so a bit-counter returning demanded bits
    instead of achieved ones would fail here rather than in the analysis.
    """
    params = _P
    demanded = 4.8
    stored = R.achieved_r(2.0 * params, params)
    assert stored < demanded
    assert stored / demanded == pytest.approx(0.417, abs=0.01)


def test_the_bios_schema_bits_reproduce_the_published_value():
    """
    Four categorical pools at 200/300/100/263 plus a birth date over 12 x 28 x 400 gives 47.592.

    Physics 3.3 publishes the total, not the factorisation, so this is our reconstruction -- but it
    pins their figure to 0.01 of a bit, and it is the schema we use, which is what makes the
    bit-counts comparable. Note this is the attribute half only; the name term is separate.
    """
    categorical = [200, 300, 100, 263]
    birth_date = 12 * 28 * 400
    bits = sum(math.log2(n) for n in categorical) + math.log2(birth_date)
    assert bits == pytest.approx(R.BIOS_BITS_PER_ENTITY, abs=0.01)


# --- degenerate inputs ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_call, match",
    [
        (lambda: R.solve(_P, 0.0, bits_per_entity=_BITS), "'demand_bits_per_param' must be"),
        (lambda: R.solve(_P, -1.0, bits_per_entity=_BITS), "'demand_bits_per_param' must be"),
        (lambda: R.solve(_P, 1.2, bits_per_entity=0.0), "'bits_per_entity' must be"),
        (lambda: R.solve(0, 1.2, bits_per_entity=_BITS), "'non_embedding_params' must be"),
        (
            lambda: R.solve(_P, 1.2, bits_per_entity=_BITS, exposures=0),
            "'exposures' must be",
        ),
        (
            lambda: R.solve(_P, 1.2, bits_per_entity=_BITS, tokens_per_bio=0),
            "'tokens_per_bio' must be",
        ),
        (lambda: R.capacity_bits(0, 1.2), "'params' must be"),
        (lambda: R.capacity_bits(100, 0.0), "'r_e' must be"),
        (lambda: R.demanded_bits(0, _BITS), "'n_entities' must be"),
        (lambda: R.achieved_r(-1.0, 100), "must not be negative"),
        (lambda: R.rho_from_demand(1.2, 0.0), "'r_e' must be"),
    ],
    ids=[
        "demand-zero",
        "demand-negative",
        "bits-zero",
        "params-zero",
        "exposures-zero",
        "tokens-per-bio-zero",
        "capacity-params-zero",
        "capacity-r_e-zero",
        "entities-zero",
        "achieved-negative",
        "rho-r_e-zero",
    ],
)
def test_degenerate_inputs_are_refused(bad_call, match):
    """Each of these would otherwise produce a corpus or an axis of zero or negative size."""
    with pytest.raises(OLMoConfigurationError, match=match):
        bad_call()


def test_solve_refuses_a_cell_too_small_to_be_a_corpus():
    """A demand small enough to round to zero entities is a config error, not an empty corpus."""
    with pytest.raises(OLMoConfigurationError, match="too small to be a corpus"):
        R.solve(LADDER["13M"][0], 1e-9, bits_per_entity=_BITS)
