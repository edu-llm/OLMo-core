"""
What the entity table guarantees: exact bits, unique names, and a prefix property across cells.

The prefix property is the one that would be easiest to lose and hardest to notice. Every cell must
describe the *same* 25k probe people for related-reasoning accuracy to be comparable across the
ladder, and that only holds because entity ``i``'s attributes do not depend on ``n_entities``. A
single ``(N, K)`` block draw would break it silently -- the corpus would still be valid, the bits
would still be exact, and P4 would be measuring a different population in every cell.
"""

import math

import numpy as np
import pytest
from factcrowd.corpus import entities as E

from olmo_core.exceptions import OLMoConfigurationError

SEED = 1234


def small_schema() -> E.Schema:
    """A schema small enough to enumerate, so tests can check exhaustively rather than by sample."""
    return E.Schema(
        attributes=(
            E.AttributePool("colour", ("red", "green", "blue", "grey")),  # 2 bits
            E.AttributePool("shape", ("disc", "cone")),  # 1 bit
        ),
        names=(
            E.AttributePool("first_name", tuple(f"f{i}" for i in range(5))),
            E.AttributePool("last_name", tuple(f"l{i}" for i in range(7))),
        ),
    )


# --- bits are arithmetic, never an estimate -------------------------------------------------------


def test_pool_bits_are_exactly_log2_of_size():
    """No estimator anywhere: a pool of 2^k values carries exactly k bits."""
    assert E.AttributePool("p", tuple(str(i) for i in range(256))).bits == 8.0
    assert E.AttributePool("p", ("a", "b", "c")).bits == pytest.approx(math.log2(3))


def test_bits_per_entity_sums_attribute_pools_only():
    """4 values plus 2 values is 3 bits, and the two name pools contribute nothing."""
    assert small_schema().bits_per_entity == pytest.approx(3.0)


def test_names_are_excluded_from_bits_because_a_name_is_a_key():
    """
    Including name pools would inflate demanded bits and put every cell below its labelled rho.

    Physics 3.3's 47.6 bits/person ignores names for this reason, so excluding them is what keeps
    our bit-counts comparable to theirs.
    """
    schema = small_schema()
    if_names_counted = schema.bits_per_entity + sum(p.bits for p in schema.names)
    assert if_names_counted > schema.bits_per_entity
    assert schema.bits_per_entity == pytest.approx(3.0)


def test_bios_schema_reproduces_the_published_bits_per_entity():
    """
    The default schema lands on 47.6 bits/person, which is what makes the comparison legitimate.

    Asserted to 0.01 bits against the value :mod:`factcrowd.ladder.rho` documents.
    """
    from factcrowd.ladder import rho

    assert E.bios_schema().bits_per_entity == pytest.approx(rho.BIOS_BITS_PER_ENTITY, abs=0.01)


def test_total_bits_is_entities_times_bits_each():
    """Demanded bits is a product, recomputed here from the table rather than trusted."""
    table = E.EntityTable.build(small_schema(), 30, SEED)
    assert table.total_bits == pytest.approx(30 * 3.0)


@pytest.mark.parametrize("bad_values", [(), ("only",), ("dup", "dup")])
def test_pool_refuses_sizes_that_break_the_bit_arithmetic(bad_values):
    """
    Empty and single-value pools carry no bits; duplicates make log2(len) an overstatement.

    Each would leave ``bits_per_entity`` describing something the corpus does not contain.
    """
    with pytest.raises(OLMoConfigurationError):
        E.AttributePool("p", bad_values)


def test_schema_refuses_missing_name_pools():
    """Without a name there is no key to attach a fact to, so recall has nothing to ask about."""
    with pytest.raises(OLMoConfigurationError, match="needs at least one name pool"):
        E.Schema(attributes=(E.AttributePool("a", ("x", "y")),), names=())


def test_schema_refuses_a_name_that_is_also_an_attribute():
    """A pool serving as both key and value would be counted in bits and used for lookup."""
    shared = E.AttributePool("city", ("x", "y"))
    with pytest.raises(OLMoConfigurationError, match="must be distinct"):
        E.Schema(attributes=(shared,), names=(shared,))


# --- the prefix property that makes cells comparable ---------------------------------------------


@pytest.mark.parametrize("small, large", [(10, 100), (100, 5_000), (1_000, 20_000)])
def test_a_smaller_table_is_a_prefix_of_a_larger_one(small, large):
    """
    Entity i has the same attributes at every N, which is what makes the probe subset comparable.

    This is the invariant a single ``(N, K)`` block draw would silently break.
    """
    schema = E.bios_schema()
    lo = E.EntityTable.build(schema, small, SEED)
    hi = E.EntityTable.build(schema, large, SEED)

    np.testing.assert_array_equal(lo.attributes, hi.attributes[:small])
    np.testing.assert_array_equal(lo.name_indices, hi.name_indices[:small])


def test_the_probe_subset_is_the_same_people_in_every_cell():
    """
    Related-reasoning accuracy is compared across cells, so it must ask about one population.

    Sizes here bracket the real ladder: the smallest cell is ~79k entities and the largest ~6.4M.
    """
    schema = E.bios_schema()
    reference = E.EntityTable.build(schema, 79_000, SEED)
    reference_probe = [reference.attribute_values(int(i)) for i in reference.probe_ids[:200]]

    for n_entities in (100_000, 714_000):
        table = E.EntityTable.build(schema, n_entities, SEED)
        np.testing.assert_array_equal(table.probe_ids, reference.probe_ids)
        assert [table.attribute_values(int(i)) for i in table.probe_ids[:200]] == reference_probe


def test_probe_subset_is_capped_by_a_table_smaller_than_it():
    """A table below the probe size yields what it has rather than indices that do not exist."""
    table = E.EntityTable.build(small_schema(), 20, SEED)
    assert len(table.probe_ids) == 20
    assert int(table.probe_ids.max()) < table.n_entities


def test_probe_size_fits_the_smallest_cell_in_the_grid():
    """
    The 13M row at rho=0.25 is ~79k entities, so 25k leaves a non-probe comparison group.

    If the probe ever exceeded the smallest cell, probe-vs-non-probe recall -- the check on whether
    QA mentions change storage -- would have nothing to compare against.
    """
    from factcrowd.ladder import rho

    schema = E.bios_schema()
    smallest = rho.solve(
        12_595_456,
        0.30,
        bits_per_entity=schema.bits_per_entity,
        name_space=schema.name_space,
    ).n_entities
    # 64,180 with the name term included, against 79,397 without it -- the name term removes 19% of
    # the entities at fixed demand, so the non-probe group is 39k rather than the 54k an
    # attribute-only count would suggest. Still comfortably above the n >= 2,000 the eval needs.
    assert smallest == pytest.approx(64_180, rel=0.01)
    assert E.PROBE_SIZE < smallest
    assert smallest - E.PROBE_SIZE > 30_000


# --- determinism and uniqueness -------------------------------------------------------------------


def test_the_same_seed_gives_the_same_table():
    """Reproducibility from ``(schema, n_entities, seed)`` is what we publish instead of shards."""
    a = E.EntityTable.build(E.bios_schema(), 5_000, SEED)
    b = E.EntityTable.build(E.bios_schema(), 5_000, SEED)
    np.testing.assert_array_equal(a.attributes, b.attributes)
    np.testing.assert_array_equal(a.name_indices, b.name_indices)


def test_a_different_seed_gives_a_different_table():
    """Otherwise the seed is decoration and M5's extra seeds would re-run one experiment."""
    a = E.EntityTable.build(E.bios_schema(), 5_000, SEED)
    b = E.EntityTable.build(E.bios_schema(), 5_000, SEED + 1)
    assert not np.array_equal(a.attributes, b.attributes)


@pytest.mark.parametrize("n_entities", [2, 100, 10_000, 60_000])
def test_names_are_unique_by_construction(n_entities):
    """
    Two entities sharing a name would make the corpus assert contradictory facts about one key.

    Checked exhaustively rather than by sampling, because the bijection either holds for every id
    or is not a bijection.
    """
    table = E.EntityTable.build(E.bios_schema(), n_entities, SEED)
    rows = {tuple(int(v) for v in row) for row in table.name_indices}
    assert len(rows) == n_entities


def test_names_stay_unique_up_to_the_ceiling_of_a_tiny_name_space():
    """
    The bijection must not degrade near capacity, which is where rejection sampling would.

    A 5 x 7 name space holds exactly 35 entities, so 35 distinct names is the strongest form of
    this claim.
    """
    table = E.EntityTable.build(small_schema(), 35, SEED)
    rows = {tuple(int(v) for v in row) for row in table.name_indices}
    assert len(rows) == 35


def test_building_more_entities_than_names_is_refused():
    """Silently colliding names would corrupt every fact about the colliding keys."""
    with pytest.raises(OLMoConfigurationError, match="distinct names"):
        E.EntityTable.build(small_schema(), 36, SEED)


def test_name_codes_refuse_an_id_past_the_name_space():
    """The lower-level guard, so a caller bypassing build() still cannot collide."""
    with pytest.raises(OLMoConfigurationError, match="does not fit a name space"):
        E.name_codes(np.array([0, 35], dtype=np.uint64), name_space=35, seed=SEED)


def test_name_codes_are_a_permutation_of_the_whole_space():
    """A bijection, so every name is reachable and none is favoured."""
    codes = E.name_codes(np.arange(35, dtype=np.uint64), name_space=35, seed=SEED)
    assert sorted(int(c) for c in codes) == list(range(35))


def test_consecutive_entities_do_not_get_consecutive_names():
    """
    The multiplier is chosen to spread adjacent ids, so a rendered corpus does not read as a list.

    Not a statistical claim -- the codes are a permutation and have structure -- just enough spread
    that neighbouring entities are not neighbouring names.
    """
    codes = E.name_codes(np.arange(1000, dtype=np.uint64), name_space=160_000_000, seed=SEED)
    gaps = np.abs(np.diff(codes.astype(np.int64)))
    assert gaps.min() > 1000


def test_build_refuses_a_non_positive_entity_count():
    """An empty fact slice is a config error, not a cell with rho=0."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        E.EntityTable.build(small_schema(), 0, SEED)


# --- values, round-tripping, and the schema guard --------------------------------------------------


def test_attribute_values_and_name_parts_resolve_to_pool_strings():
    """The indices point where they claim to, so rendering and eval read the same values."""
    schema = small_schema()
    table = E.EntityTable.build(schema, 30, SEED)

    values = table.attribute_values(7)
    assert values[0] in schema.attributes[0].values
    assert values[1] in schema.attributes[1].values

    parts = table.name_parts(7)
    assert parts[0] in schema.names[0].values
    assert parts[1] in schema.names[1].values


def test_save_and_load_round_trip(tmp_path):
    """The table we publish must read back as the table we generated."""
    schema = E.bios_schema()
    original = E.EntityTable.build(schema, 3_000, SEED)
    path = tmp_path / "tables" / "d576_rho1.npz"
    original.save(path)

    loaded = E.EntityTable.load(path, schema)
    np.testing.assert_array_equal(loaded.attributes, original.attributes)
    np.testing.assert_array_equal(loaded.name_indices, original.name_indices)
    assert loaded.seed == original.seed
    assert loaded.bits_per_entity == pytest.approx(original.bits_per_entity)


def test_load_refuses_a_table_built_against_different_vocabulary(tmp_path):
    """
    Indices are meaningless without the pools they point into, and a mismatch is silent otherwise.

    Same pool *sizes*, so bits per entity is identical and no arithmetic check would catch it --
    only the fingerprint does. Without this the corpus would assert different facts about every
    entity while every number in the run stayed plausible.
    """
    original_schema = E.bios_schema()
    table = E.EntityTable.build(original_schema, 100, SEED)
    path = tmp_path / "t.npz"
    table.save(path)

    renamed = list(original_schema.attributes)
    renamed[0] = E.AttributePool(
        "birth_city", tuple(f"other_city_{i}" for i in range(len(renamed[0])))
    )
    other_schema = E.Schema(attributes=tuple(renamed), names=original_schema.names)
    assert other_schema.bits_per_entity == pytest.approx(original_schema.bits_per_entity)

    with pytest.raises(OLMoConfigurationError, match="different schema"):
        E.EntityTable.load(path, other_schema)


def test_schema_fingerprint_tracks_values_not_just_sizes():
    """Two schemas with equal bit counts and different vocabulary are different schemas."""
    base = E.bios_schema()
    swapped = list(base.attributes)
    swapped[1] = E.AttributePool("university", tuple(f"u{i}" for i in range(len(swapped[1]))))
    other = E.Schema(attributes=tuple(swapped), names=base.names)

    assert base.bits_per_entity == pytest.approx(other.bits_per_entity)
    assert base.fingerprint() != other.fingerprint()


# --- the placeholder vocabulary, and the guard on replacing it ------------------------------------


def test_default_vocabulary_is_placeholders_and_says_so():
    """
    A reminder in test form: bit accounting is already exact, token counts are not.

    ``employer_017`` and a real employer name do not tokenize alike, and both the ~100 tokens/bio
    budget and the 32k BPE depend on the real strings. This asserts the gap still exists so that
    closing it is a visible change rather than an assumed one.
    """
    schema = E.bios_schema()
    assert schema.attributes[0].values[0].startswith("birth_city_")


def test_real_vocabulary_can_be_supplied_at_matching_sizes():
    """The replacement path works, and bits per entity is unchanged by it."""
    placeholder = E.bios_schema()
    real = [
        E.AttributePool(pool.name, tuple(f"Real {pool.name} {i}" for i in range(len(pool))))
        for pool in placeholder.attributes
    ]
    schema = E.bios_schema(vocabulary=real)

    assert schema.bits_per_entity == pytest.approx(placeholder.bits_per_entity)
    assert schema.attributes[0].values[0].startswith("Real birth_city")


def test_supplying_a_wrong_pool_size_is_refused():
    """Changing a pool size changes bits per entity, which moves the x-axis of every plot."""
    placeholder = E.bios_schema()
    truncated = [
        E.AttributePool(
            pool.name, pool.values[: len(pool) - 1] if pool.name == "major" else pool.values
        )
        for pool in placeholder.attributes
    ]
    with pytest.raises(OLMoConfigurationError, match="must have 100 values"):
        E.bios_schema(vocabulary=truncated)


def test_supplying_the_wrong_pool_names_is_refused():
    """A vocabulary missing an attribute would leave a placeholder silently in the corpus."""
    placeholder = E.bios_schema()
    with pytest.raises(OLMoConfigurationError, match="must supply exactly"):
        E.bios_schema(vocabulary=list(placeholder.attributes)[:2])


def test_name_space_is_the_product_of_the_name_pools():
    """The ceiling on n_entities, and the default clears the largest cell by 25x."""
    assert small_schema().name_space == 35
    assert E.bios_schema().name_space == 400 * 400 * 1000
    assert E.bios_schema().name_space > 6_430_000
