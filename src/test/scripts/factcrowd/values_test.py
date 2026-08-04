"""
What the value schemes guarantee: exact bits, iso-token rendering, and globally disjoint words.

The iso-token property is the one the experiment's identification rests on. The entropy axis exists
because the count axis sweeps demand and token count together, so if the entropy axis leaked *any*
dependence of token count on ``b`` it would inherit the confound it was built to remove -- and it
would do so silently, because the corpus would still be valid and the bits would still be exact.

Disjointness is a correctness property rather than tidiness. Under a word-level vocabulary a word
shared between the ``university`` and ``employer`` pools makes that surface form ambiguous about
which fact it states, so an eval item would have a defensible wrong answer.
"""

import math

import pytest
from factcrowd.corpus import values as V

from olmo_core.exceptions import OLMoConfigurationError

ENTROPY_LEVELS = (0, 4, 8, 16, 24, 32)
"""The sweep from PRD.md section 3.1: demand 0 to 4.84 bits/param at fixed tokens."""


def all_words(corpus_schema: V.CorpusSchema) -> list:
    """Every word in every pool, attributes and names alike."""
    pools = list(corpus_schema.schema.attributes) + list(corpus_schema.schema.names)
    return [word for pool in pools for word in pool.values]


# --- the iso-token property ------------------------------------------------------------------------


@pytest.mark.parametrize("bits", ENTROPY_LEVELS)
def test_the_entropy_axis_renders_the_same_number_of_words_at_every_demand(bits):
    """
    Token count is invariant in b. This is the axis's defining property and its whole justification.

    Six attributes of four words each, at every b from 0 to 32 -- so tokens, steps, schedule position
    and mixture ratio are held fixed while demand sweeps over a 0-to-4.84 bits/param range.
    """
    schema = V.entropy_schema(bits)
    assert schema.words_per_entity == V.ENTROPY_ATTRIBUTES * V.ENTROPY_WORDS_PER_VALUE == 24
    assert all(spec.words_per_value == 4 for spec in schema.values)


def test_bits_per_entity_is_exactly_six_b():
    """``6b``, exactly, with no estimator anywhere -- pools are powers of two by construction."""
    for bits in ENTROPY_LEVELS:
        assert V.entropy_schema(bits).bits_per_entity == pytest.approx(6.0 * bits)
        assert V.bits_per_entity_for(bits) == 6.0 * bits


def test_the_entropy_midpoint_anchors_to_bios():
    """
    b=8 gives 48 bits/entity against bioS's 47.592 -- a 0.9% match.

    This is what lets the entropy axis be read against Physics 3.3 at all: its middle cell is the
    literature's corpus to within a rounding.
    """
    entropy = V.entropy_schema(8).bits_per_entity
    bios = V.bios_schema().bits_per_entity
    assert entropy == pytest.approx(48.0)
    assert abs(entropy - bios) / bios < 0.01


def test_b_zero_is_a_legitimate_point_not_a_degenerate_one():
    """
    Pools of one value: zero demanded bits at unchanged tokens, steps and ratio.

    The anchor that makes the sweep's intercept measurable. Every entity shares one value tuple,
    which is the manipulation rather than a broken schema.
    """
    schema = V.entropy_schema(0)
    assert schema.bits_per_entity == 0.0
    assert schema.words_per_entity == 24
    # One *reachable* value per pool, out of a union pool shared with every other cell in the sweep.
    # The pool is not shrunk to one word: doing that made the vocabulary a function of the treatment,
    # so b=0 and b=32 differed by 8.1% in parameters and 4.2x in softmax width.
    assert all(pool.active_size == 1 for pool in schema.schema.attributes)
    assert all(
        len(pool) == 2 ** (V.ENTROPY_VOCABULARY_BITS // 4) for pool in schema.schema.attributes
    )
    # Names still vary, because entities must remain distinguishable keys.
    assert all(len(pool) > 1 for pool in schema.schema.names)


def test_pool_size_is_a_power_of_two_at_every_level():
    """
    The *active* size is ``2^(b/4)``, so bits per pool is an integer and the arithmetic stays exact.

    The pool itself is the sweep's union and identical everywhere, which is what holds the model fixed
    while the information content varies.
    """
    union = 2 ** (V.ENTROPY_VOCABULARY_BITS // 4)
    for bits in ENTROPY_LEVELS:
        for pool in V.entropy_schema(bits).schema.attributes:
            assert pool.active_size == 2 ** (bits // 4)
            assert len(pool) == union
            assert pool.bits == float(bits // 4)


# --- bioS ------------------------------------------------------------------------------------------


def test_bios_reproduces_the_published_bits_per_entity():
    """47.592 from seven pools, pinning Physics 3.3's published 47.6 to 0.01 of a bit."""
    assert V.bios_schema().bits_per_entity == pytest.approx(47.592, abs=0.001)
    assert V.bios_bits_per_entity() == pytest.approx(47.592, abs=0.001)


def test_the_bios_helper_agrees_with_the_built_schema():
    """
    The cheap path and the real path must not diverge -- config validation uses the cheap one.

    If they disagreed, a cell would be validated against one bits/entity and generated against
    another.
    """
    assert V.bios_bits_per_entity() == pytest.approx(V.bios_schema().bits_per_entity)


def test_the_birth_date_is_decomposed_and_carries_the_same_bits():
    """
    Three pools of 12/28/400 carry 17.036 bits, identical to one 134,400-value pool.

    Decomposed because 134,400 word types at d_model=256 is a 34.4M embedding table against a 12.6M
    model -- nearly three times the thing whose capacity is being measured.
    """
    date_bits = sum(
        math.log2(size) for name, size in V.BIOS_POOL_SIZES if name.startswith("birth_")
    ) - math.log2(
        200
    )  # exclude birth_city, which also starts with birth_
    assert date_bits == pytest.approx(math.log2(12 * 28 * 400), abs=1e-9)
    assert date_bits == pytest.approx(17.036, abs=0.001)


def test_bios_has_one_word_per_attribute():
    """The count axis renders one word per field, so tokens/bio tracks the template not the pools."""
    schema = V.bios_schema()
    assert schema.words_per_entity == 7
    assert all(spec.words_per_value == 1 for spec in schema.values)


# --- disjointness and determinism -----------------------------------------------------------------


@pytest.mark.parametrize("bits", ENTROPY_LEVELS)
def test_every_word_is_unique_across_every_pool_on_the_entropy_axis(bits):
    """A word in two pools makes that surface form ambiguous about which fact it states."""
    words = all_words(V.entropy_schema(bits))
    assert len(words) == len(set(words))


def test_every_word_is_unique_across_every_pool_on_the_count_axis():
    """Same requirement for bioS, and it includes the name pools."""
    words = all_words(V.bios_schema())
    assert len(words) == len(set(words))


def test_names_do_not_collide_with_attribute_values():
    """
    A name that is also a university is the same ambiguity, so names share the one allocation.

    Checked separately because it would be easy to allocate names from their own cursor and
    reintroduce exactly this.
    """
    schema = V.bios_schema()
    attribute_words = {w for pool in schema.schema.attributes for w in pool.values}
    name_words = {w for pool in schema.schema.names for w in pool.values}
    assert not (attribute_words & name_words)


def test_allocation_is_deterministic_across_calls():
    """Reproducibility from ``(schema, seed)`` is what we publish instead of token shards."""
    assert V.bios_schema().schema.fingerprint() == V.bios_schema().schema.fingerprint()
    assert V.entropy_schema(16).schema.fingerprint() == V.entropy_schema(16).schema.fingerprint()


def test_different_demand_levels_are_different_schemas():
    """Otherwise the fingerprint could not refuse a table built at the wrong b."""
    assert V.entropy_schema(8).schema.fingerprint() != V.entropy_schema(16).schema.fingerprint()


def test_consecutive_words_do_not_look_like_each_other():
    """
    The coprime stride exists because a naive enumeration produced "Pouss Poust Pout Pouth".

    The enumeration's fastest-varying digit is the coda, so walking it in order hands adjacent
    allocations words differing only in their last letters. Asserted on shared prefixes: fewer than a
    fifth of consecutive pairs may share their first three characters.
    """
    words = V.allocate_words([("p", 400)])["p"]
    shared = sum(1 for a, b in zip(words, words[1:]) if a[:3] == b[:3])
    assert shared < len(words) // 5, shared


def test_words_are_pronounceable_and_capitalised():
    """
    Generated rather than placeholders like ``employer_017``, which tokenize unlike real text.

    Both the token budget and the BPE depend on the strings looking like words.
    """
    for word in V.allocate_words([("p", 50)])["p"]:
        assert word[0].isupper()
        assert word.isalpha()
        assert 3 <= len(word) <= 12
        assert any(vowel in word.lower() for vowel in "aeiou")


def test_allocation_serves_pools_in_a_declaration_order_independent_way():
    """
    The result depends on the set of pools, not the order they were listed.

    So two callers building the same schema differently still get the same corpus.
    """
    forward = V.allocate_words([("a", 10), ("b", 20), ("c", 30)])
    backward = V.allocate_words([("c", 30), ("b", 20), ("a", 10)])
    assert forward == backward


# --- the guards -----------------------------------------------------------------------------------


def test_a_pool_not_used_by_any_attribute_is_refused():
    """
    Its bits would count toward bits_per_entity while no rendered text carried them.

    That puts every cell above its true demand, which is the single most consequential silent error
    available in this module.
    """
    schema = V.bios_schema()
    with pytest.raises(OLMoConfigurationError, match="every pool exactly once"):
        V.CorpusSchema(schema=schema.schema, values=schema.values[:3])


def test_a_pool_used_by_two_attributes_is_refused():
    """Its bits would be counted once and rendered twice."""
    schema = V.bios_schema()
    doubled = schema.values + (V.ValueSpec(name="again", pool_names=("major",)),)
    with pytest.raises(OLMoConfigurationError, match="every pool exactly once"):
        V.CorpusSchema(schema=schema.schema, values=doubled)


def test_an_attribute_composing_no_pools_is_refused():
    """It would render nothing and carry no value."""
    with pytest.raises(OLMoConfigurationError, match="composes zero pools"):
        V.ValueSpec(name="empty", pool_names=())


@pytest.mark.parametrize("bad_bits", [6, 10, 30])
def test_bits_not_divisible_by_the_word_count_are_refused_with_the_nearest_usable_values(bad_bits):
    """
    A non-power-of-two pool would make bits per attribute irrational and the arithmetic inexact.

    The message names the two nearest usable values, because this fires in front of somebody editing
    a config.
    """
    with pytest.raises(OLMoConfigurationError, match="must be divisible") as excinfo:
        V.entropy_schema(bad_bits)
    assert "nearest usable values" in str(excinfo.value)


def test_a_demand_needing_a_huge_vocabulary_is_refused():
    """
    Two refusals, and both matter.

    A level above the sweep's union is refused by name, because a cell whose pools were sized for itself
    would carry a different model from the rest of the sweep -- the confound the union exists to remove.
    And a union so wide that the embedding table dwarfs the model is refused with the way out: more
    words per value spreads the same bits over smaller pools.
    """
    with pytest.raises(OLMoConfigurationError, match="above the union vocabulary"):
        V.entropy_schema(64)
    with pytest.raises(OLMoConfigurationError, match="larger than the model") as excinfo:
        V.entropy_schema(64, vocabulary_bits=64)
    assert "words_per_value" in str(excinfo.value)


def test_negative_bits_are_refused():
    """There is no such thing as negative demand."""
    with pytest.raises(OLMoConfigurationError, match="must not be negative"):
        V.entropy_schema(-4)


def test_duplicate_and_degenerate_pool_specs_are_refused():
    """Two pools of one name would silently share an allocation."""
    with pytest.raises(OLMoConfigurationError, match="must be unique"):
        V.allocate_words([("a", 10), ("a", 20)])
    with pytest.raises(OLMoConfigurationError, match="positive size"):
        V.allocate_words([("a", 0)])


def test_the_wrong_number_of_name_pool_sizes_is_refused():
    """Names are first/middle/last; a different count would change the name universe silently."""
    with pytest.raises(OLMoConfigurationError, match="three name pool sizes"):
        V.bios_schema(name_pool_sizes=(400, 400))


# --- the link to the demand arithmetic ------------------------------------------------------------


def test_the_name_universe_is_the_product_of_the_name_pools():
    """
    ``N0`` feeds the demand formula's name term, so it is load-bearing rather than a mere ceiling.

    160M clears the largest cell's 6.43M entities by 25x.
    """
    schema = V.bios_schema()
    assert schema.schema.name_space == 400 * 400 * 1000
    assert schema.schema.name_space / 6_430_000 > 24  # 24.9x headroom


def test_the_entropy_sweep_reaches_the_demand_levels_the_prd_quotes():
    """
    b in {0,4,8,16,24,32} at 28M with N fixed gives 0 to 4.84 bits/param, i.e. rho 0 to 4.03.

    Checked against the demand arithmetic rather than restated, so the two modules cannot drift.
    """
    from factcrowd.ladder import rho

    non_embedding = 28_330_368
    n_entities = 714_331
    name_space = V.bios_schema().schema.name_space
    attribute_only = (0.00, 0.61, 1.21, 2.42, 3.63, 4.84)

    for bits, want in zip(ENTROPY_LEVELS, attribute_only):
        got = rho.demand_per_param(
            n_entities, non_embedding, bits_per_entity=V.bits_per_entity_for(bits), name_space=None
        )
        assert got == pytest.approx(want, abs=0.01), bits

        # With the name term the whole sweep shifts by one constant, because N is fixed here. That
        # is what distinguishes this axis from the count axis, where a varying N bends the curve.
        with_names = rho.demand_per_param(
            n_entities,
            non_embedding,
            bits_per_entity=V.bits_per_entity_for(bits),
            name_space=name_space,
        )
        # 0.233 rather than 0.197: the name term is the exact log2 C(N0, N), and the proxy is
        # 15.2% low at this entity count.
        assert with_names - got == pytest.approx(0.233, abs=0.001), bits


def test_the_b_zero_cell_is_not_zero_demand():
    """
    Its attribute bits are zero, but distinct names still carry the name term: 0.233 bits/param.

    So the sweep's intercept is the name floor rather than the origin, and a plot drawn through zero
    would misplace every fitted line. Zero attribute bits is legal input to the demand arithmetic for
    exactly this cell and is refused by ``solve``, which divides by it.
    """
    from factcrowd.ladder import rho

    schema = V.entropy_schema(0)
    assert schema.bits_per_entity == 0.0

    demand = rho.demand(
        714_331,
        bits_per_entity=schema.bits_per_entity,
        non_embedding_params=28_330_368,
        total_params=28_330_368 + 32_000 * 384,
        name_space=schema.schema.name_space,
    )
    assert demand.attribute_bits == 0.0
    assert demand.name_bits > 0
    assert demand.per_non_embedding_param == pytest.approx(0.233, abs=0.001)

    with pytest.raises(OLMoConfigurationError, match="'bits_per_entity' must be positive"):
        rho.solve(28_330_368, 1.2, bits_per_entity=0.0, name_space=None)
