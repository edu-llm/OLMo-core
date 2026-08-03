"""
What the entity table guarantees: exact bits, a pseudorandom name set, and a prefix property.

Three of these were found missing by an adversarial pass and are the reason this file is shaped the
way it is.

**The name set has to look random, not merely be unique.** The demand formula charges
``N·log2(N0/N)`` for knowing which names exist, which is the entropy of a random N-subset. An affine
``(a·i + b) mod N0`` map is bijective, so an earlier version passed every uniqueness test -- but its
image has exactly three distinct sorted gaps and is described by three numbers, so the formula
overstated it by 200,000x on the experiment's own independent variable.

**Nothing checked that the corpus realises the bits it claims.** Sampling a pool as
``integers(0, len(pool)//2)`` leaves ``bits_per_entity`` reporting 47.592 for a corpus carrying
42.586, and drawing every column from one stream collapses the joint entropy the sum assumes. Both
passed the previous suite.

**The prefix property's stated villain was innocent.** A ``(N, K)`` block draw is prefix-stable --
numpy fills row-major -- so the tests here pin a ``(K, N)`` draw, which is not.
"""

import math

import numpy as np
import pytest
from factcrowd.corpus import entities as E
from factcrowd.corpus import values as V

from olmo_core.exceptions import OLMoConfigurationError

SEED = 1234
NAME_SPACE = 400 * 400 * 1000


def bios() -> E.Schema:
    """The bioS schema, which lives in :mod:`factcrowd.corpus.values`."""
    return V.bios_schema().schema


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


def plug_in_mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    """Plug-in mutual information in bits. Biased upward, which is fine for a contrast."""
    joint = np.zeros((int(left.max()) + 1, int(right.max()) + 1), dtype=np.int64)
    np.add.at(joint, (left.astype(np.int64), right.astype(np.int64)), 1)
    total = joint.sum()
    probability = joint / total
    marginal_left = probability.sum(axis=1, keepdims=True)
    marginal_right = probability.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = probability * np.log2(probability / (marginal_left * marginal_right))
    return float(np.nansum(terms))


# --- the name set must look like a random subset ---------------------------------------------------


def test_the_name_set_does_not_have_the_three_gaps_of_an_affine_map():
    """
    The test that pins the fix. An affine map's image has exactly three distinct sorted gaps.

    By the three-distance theorem, ``{(a·i + b) mod N0}`` sorted has gaps taking at most three
    values -- so the set is described by ``(N0, N, b)``, at most ``log2(N0) = 27`` bits, while
    ``N·log2(N0/N)`` charged 29.8 Mbit for it. A keyed pseudorandom permutation gives the
    geometrically-distributed gaps the formula assumes.
    """
    codes = np.sort(
        E.name_codes(np.arange(200_000, dtype=np.uint64), name_space=NAME_SPACE, seed=SEED).astype(
            np.int64
        )
    )
    distinct_gaps = len(np.unique(np.diff(codes)))
    assert distinct_gaps > 1_000, distinct_gaps


def test_the_gap_distribution_is_close_to_geometric():
    """
    The positive form of the check above: sampling N of N0 uniformly gives mean gap N0/N.

    An affine map gives a mean gap of N0/N too -- that is forced -- so the mean is not the signal.
    The variance is: a geometric distribution has standard deviation close to its mean, while three
    distinct gap values cannot.
    """
    n_entities = 200_000
    codes = np.sort(
        E.name_codes(
            np.arange(n_entities, dtype=np.uint64), name_space=NAME_SPACE, seed=SEED
        ).astype(np.int64)
    )
    gaps = np.diff(codes)
    expected_mean = NAME_SPACE / n_entities

    assert gaps.mean() == pytest.approx(expected_mean, rel=0.05)
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.15)


def test_the_permutation_is_a_bijection_over_whole_small_name_spaces():
    """
    Uniqueness is structural -- a Feistel round is invertible whatever its round function does.

    Checked exhaustively over the entire space for sizes that are prime, a power of two, highly
    composite and awkward, because cycle walking is where a bijection would most plausibly be lost.
    """
    for name_space in (2, 3, 35, 64, 97, 100, 256, 720, 1_000, 1_024, 5_041):
        codes = E.name_codes(
            np.arange(name_space, dtype=np.uint64), name_space=name_space, seed=SEED
        )
        assert sorted(int(c) for c in codes) == list(range(name_space)), name_space


def test_names_are_unique_at_grid_scale():
    """Two entities sharing a name would make the corpus contradict itself about one key."""
    for n_entities in (2, 1_000, 200_000):
        table = E.EntityTable.build(bios(), n_entities, SEED)
        rows = {tuple(int(v) for v in row) for row in table.name_indices}
        assert len(rows) == n_entities, n_entities


def test_names_stay_unique_right_up_to_a_tiny_name_space_ceiling():
    """A 5 x 7 space holds exactly 35 entities, the strongest form of the claim."""
    table = E.EntityTable.build(small_schema(), 35, SEED)
    rows = {tuple(int(v) for v in row) for row in table.name_indices}
    assert len(rows) == 35


def test_a_different_seed_changes_the_names_and_not_only_the_attributes():
    """
    A fixed permutation key would give every seed the same name assignment.

    The previous suite compared only ``attributes`` between seeds, so dropping the seed from the name
    key passed -- and M5's extra seeds would have re-used one name assignment.
    """
    a = E.EntityTable.build(bios(), 5_000, SEED)
    b = E.EntityTable.build(bios(), 5_000, SEED + 1)
    assert not np.array_equal(a.attributes, b.attributes)
    assert not np.array_equal(a.name_indices, b.name_indices)


def test_the_mixed_radix_decomposition_covers_every_combination_exactly_once():
    """Every name triple is reachable and none is favoured, checked by enumeration."""
    schema = small_schema()  # 5 x 7 = 35
    table = E.EntityTable.build(schema, 35, SEED)
    rows = {tuple(int(v) for v in row) for row in table.name_indices}
    assert rows == {(first, last) for first in range(5) for last in range(7)}


# --- the corpus must realise the bits it claims -----------------------------------------------------


def test_every_pool_value_is_actually_sampled():
    """
    ``bits_per_entity`` assumes the full pool is in play. Nothing checked that it is.

    Sampling ``integers(0, len(pool)//2)`` leaves the property reporting 47.592 bits for a corpus
    carrying 42.586 -- a 10.5% overstatement of the experiment's independent variable, with a green
    suite. 200k rows covers the largest pool here (400 values) many times over.
    """
    schema = bios()
    table = E.EntityTable.build(schema, 200_000, SEED)
    for column, pool in enumerate(schema.attributes):
        observed = set(np.unique(table.attributes[:, column]).tolist())
        assert observed == set(range(len(pool))), (pool.name, len(observed), len(pool))


def test_attribute_columns_are_independent():
    """
    Summing ``log2(len(pool))`` over pools assumes the columns are independent.

    Drawing every column from one ``SeedSequence`` makes them near-deterministic in each other and
    collapses the joint entropy the sum asserts. Plug-in mutual information is biased upward at these
    pool sizes, so this is a contrast rather than a zero test: independent columns read about 0.2
    bits here, the shared-stream version read 7.3 of a 7.6-bit maximum.
    """
    schema = bios()
    table = E.EntityTable.build(schema, 200_000, SEED)
    for left in range(len(schema.attributes)):
        for right in range(left + 1, len(schema.attributes)):
            information = plug_in_mutual_information(
                table.attributes[:, left], table.attributes[:, right]
            )
            assert information < 1.0, (left, right, information)


def test_the_sampled_marginals_are_close_to_uniform():
    """
    Uniform sampling is what makes attribute demand exactly ``N x bits_per_entity``.

    A skewed draw would leave demand an unknown smaller number, which is the failure mode that made
    a Zipf arm uncomputable.
    """
    schema = bios()
    table = E.EntityTable.build(schema, 200_000, SEED)
    for column, pool in enumerate(schema.attributes):
        counts = np.bincount(table.attributes[:, column], minlength=len(pool))
        expected = 200_000 / len(pool)
        assert counts.max() < 1.4 * expected, pool.name
        assert counts.min() > 0.6 * expected, pool.name


def test_a_golden_row_is_pinned():
    """
    A regression pin: nothing else asserts *which* values a seed produces.

    Any change to the streams, the spawn order or the column mapping moves this, which is what makes
    a silent reshuffle visible.
    """
    table = E.EntityTable.build(bios(), 1_000, SEED)
    assert [int(v) for v in table.attributes[0]] == [85, 213, 28, 96, 9, 3, 37]
    assert [int(v) for v in table.name_indices[0]] == [151, 196, 114]
    assert table.attribute_values(0) == (
        "Soagh",
        "Slayng",
        "Slealn",
        "Kneng",
        "Grieft",
        "Ceell",
        "Coosp",
    )
    assert " ".join(table.name_parts(0)) == "Sowt Yeemp Snisp"


def test_bits_per_entity_sums_attribute_pools_only():
    """4 values plus 2 values is 3 bits, and the two name pools contribute nothing."""
    assert small_schema().bits_per_entity == pytest.approx(3.0)


def test_attribute_bits_is_not_the_corpus_demand():
    """
    The trap this property used to be. It omits the name term, by 8.9% to 18.8% depending on N.

    Both this and ``rho.demanded_bits`` once carried the docstring "fact bits this corpus makes
    available", so the first consumer to reach for the one on the table it already held would have
    placed the cell 9-19% off its x.
    """
    from factcrowd.ladder import rho

    table = E.EntityTable.build(bios(), 200_000, SEED)
    demand = rho.demanded_bits(
        table.n_entities, table.bits_per_entity, name_space=table.schema.name_space
    )
    assert table.attribute_bits < demand
    assert table.attribute_bits / demand < 0.95


@pytest.mark.parametrize("bad_values", [(), ("only",), ("dup", "dup")])
def test_pool_refuses_sizes_that_break_the_bit_arithmetic(bad_values):
    """Empty and singleton pools carry no bits; duplicates make log2(len) an overstatement."""
    with pytest.raises(OLMoConfigurationError):
        E.AttributePool("p", bad_values)


def test_a_singleton_pool_is_allowed_only_when_asked_for():
    """The entropy axis's b=0 cell is the one legitimate use."""
    with pytest.raises(OLMoConfigurationError):
        E.AttributePool("p", ("only",))
    assert E.AttributePool("p", ("only",), allow_singleton=True).bits == 0.0


# --- the prefix property ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "small, large", [(10, 100), (63, 65), (4_095, 4_097), (65_535, 65_537), (1_000, 200_000)]
)
def test_a_smaller_table_is_a_prefix_of_a_larger_one(small, large):
    """
    Entity i has the same attributes at every N, which makes the probe subset comparable.

    Sizes straddle powers of two because that is where a chunked generator would most plausibly
    break.
    """
    lo = E.EntityTable.build(bios(), small, SEED)
    hi = E.EntityTable.build(bios(), large, SEED)
    np.testing.assert_array_equal(lo.attributes, hi.attributes[:small])
    np.testing.assert_array_equal(lo.name_indices, hi.name_indices[:small])


def test_adding_an_attribute_leaves_the_existing_columns_unmoved():
    """
    The property the column-wise draw actually buys, as against a block draw.

    A ``(N, K)`` block draw is prefix-stable in N -- numpy fills row-major -- so it is not what the
    column-wise design protects against. This is.
    """
    schema = bios()
    extended = E.Schema(
        attributes=schema.attributes + (E.AttributePool("extra", ("x", "y", "z")),),
        names=schema.names,
    )
    base = E.EntityTable.build(schema, 5_000, SEED)
    grown = E.EntityTable.build(extended, 5_000, SEED)
    np.testing.assert_array_equal(base.attributes, grown.attributes[:, : len(schema.attributes)])


def test_the_probe_subset_is_the_same_people_in_every_cell():
    """Related-reasoning accuracy is compared across cells, so it must ask about one population."""
    schema = bios()
    reference = E.EntityTable.build(schema, 64_180, SEED)
    sample = [reference.attribute_values(int(i)) for i in reference.probe_ids[:200]]
    for n_entities in (100_000, 611_184):
        table = E.EntityTable.build(schema, n_entities, SEED)
        np.testing.assert_array_equal(table.probe_ids, reference.probe_ids)
        assert [table.attribute_values(int(i)) for i in table.probe_ids[:200]] == sample


def test_probe_size_fits_the_smallest_cell_in_the_grid():
    """
    13M at a demand of 0.30 bits/param is 64,180 entities with the name term, so 39k are non-probe.

    Without the name term it would read 79,397 and 54k. The docstring quotes the former.
    """
    from factcrowd.ladder import rho

    schema = bios()
    smallest = rho.solve(
        12_595_456, 0.30, bits_per_entity=schema.bits_per_entity, name_space=schema.name_space
    ).n_entities
    assert smallest == pytest.approx(64_180, rel=0.01)
    assert E.PROBE_SIZE < smallest
    assert smallest - E.PROBE_SIZE > 30_000


def test_probe_subset_is_capped_by_a_table_smaller_than_it():
    """A table below the probe size yields what it has rather than indices that do not exist."""
    table = E.EntityTable.build(small_schema(), 20, SEED)
    assert len(table.probe_ids) == 20
    assert int(table.probe_ids.max()) < table.n_entities


# --- guards on the inputs --------------------------------------------------------------------------


def test_name_codes_refuses_a_float_id_array():
    """
    ``astype(uint64)`` truncates, so 0.0, 0.4 and 0.9 would all become one name.

    The guard used to check only ``max()``, so this passed and produced three ids sharing a name.
    """
    with pytest.raises(OLMoConfigurationError, match="must be an integer array"):
        E.name_codes(np.array([0.0, 0.4, 0.9]), name_space=1000, seed=SEED)


def test_name_codes_refuses_a_negative_id():
    """A negative id wraps to a huge uint64 rather than raising, so ``max()`` alone missed it."""
    with pytest.raises(OLMoConfigurationError, match="must be non-negative"):
        E.name_codes(np.array([-3, 0, 1], dtype=np.int64), name_space=1000, seed=SEED)


def test_name_codes_refuses_an_id_past_the_name_space():
    """The lower-level guard, so a caller bypassing build() still cannot collide."""
    with pytest.raises(OLMoConfigurationError, match="does not fit a name space"):
        E.name_codes(np.array([0, 35], dtype=np.uint64), name_space=35, seed=SEED)


def test_name_codes_refuses_a_name_space_beyond_safe_uint64_arithmetic():
    """
    The name space is a demand knob, so somebody will widen it, and numpy wraps without warning.

    An earlier affine implementation lost uniqueness above about 4.6e12 and reverted silently to
    luck -- which is exactly what its docstring said it was not.
    """
    with pytest.raises(OLMoConfigurationError, match="ceiling"):
        E.name_codes(np.array([0], dtype=np.uint64), name_space=2**63, seed=SEED)


@pytest.mark.parametrize("bad_seed", [-1, -1000])
def test_a_negative_seed_is_refused_rather_than_reaching_numpy(bad_seed):
    """It would otherwise escape as a raw ``ValueError`` from ``SeedSequence``."""
    with pytest.raises(OLMoConfigurationError, match="must not be negative"):
        E.EntityTable.build(small_schema(), 10, bad_seed)


def test_build_refuses_a_non_positive_entity_count():
    """An empty fact slice is a config error, not a cell with zero demand."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        E.EntityTable.build(small_schema(), 0, SEED)


def test_building_more_entities_than_names_is_refused():
    """Silently colliding names would corrupt every fact about the colliding keys."""
    with pytest.raises(OLMoConfigurationError, match="distinct names"):
        E.EntityTable.build(small_schema(), 36, SEED)


def test_schema_refuses_missing_name_pools():
    """Without a name there is no key to attach a fact to, so recall has nothing to ask about."""
    with pytest.raises(OLMoConfigurationError, match="needs at least one name pool"):
        E.Schema(attributes=(E.AttributePool("a", ("x", "y")),), names=())


def test_schema_refuses_a_name_that_is_also_an_attribute():
    """A pool serving as both key and value would be counted in bits and used for lookup."""
    shared = E.AttributePool("city", ("x", "y"))
    with pytest.raises(OLMoConfigurationError, match="must be distinct"):
        E.Schema(attributes=(shared,), names=(shared,))


# --- the fingerprint, and save / load --------------------------------------------------------------


def test_the_fingerprint_is_length_framed_so_a_pool_name_cannot_impersonate_a_header():
    """
    Without framing, a pool called ``a:2:x\\x00y\\x00attribute:b`` collided with a two-pool schema.

    ``load`` then accepted a table carrying twice the bits the schema declared. Contrived, but the
    guarantee is stated absolutely.
    """
    honest = E.Schema(
        attributes=(E.AttributePool("a", ("x", "y")), E.AttributePool("b", ("p", "q"))),
        names=(E.AttributePool("n", ("1", "2")),),
    )
    impersonator = E.Schema(
        attributes=(E.AttributePool("a:2:x\x00y\x00attribute:b", ("p", "q")),),
        names=(E.AttributePool("n", ("1", "2")),),
    )
    assert honest.bits_per_entity != impersonator.bits_per_entity
    assert honest.fingerprint() != impersonator.fingerprint()


def test_fingerprint_tracks_values_not_just_sizes():
    """Two schemas with equal bit counts and different vocabulary are different schemas."""
    base = bios()
    swapped = list(base.attributes)
    swapped[1] = E.AttributePool("university", tuple(f"u{i}" for i in range(len(swapped[1]))))
    other = E.Schema(attributes=tuple(swapped), names=base.names)
    assert base.bits_per_entity == pytest.approx(other.bits_per_entity)
    assert base.fingerprint() != other.fingerprint()


def test_save_and_load_round_trip(tmp_path):
    """The table we publish must read back as the table we generated."""
    schema = bios()
    original = E.EntityTable.build(schema, 3_000, SEED)
    original.save(tmp_path / "tables" / "d576_b8")

    loaded = E.EntityTable.load(tmp_path / "tables" / "d576_b8", schema)
    np.testing.assert_array_equal(loaded.attributes, original.attributes)
    np.testing.assert_array_equal(loaded.name_indices, original.name_indices)
    assert loaded.seed == original.seed
    assert loaded.bits_per_entity == pytest.approx(original.bits_per_entity)


@pytest.mark.parametrize("name", ["plain", "with.dots", "d576_b8.npy", "trailing."])
def test_save_and_load_agree_on_the_path_whatever_it_is_called(tmp_path, name):
    """
    ``np.savez`` appended ``.npz`` while ``np.load`` did not, so ``save(p); load(p)`` failed.

    Every test happened to use a ``.npz`` name, so the suite never saw it -- and a cell id or a
    ``--table-path`` without the suffix wrote a file nothing would read.
    """
    schema = small_schema()
    E.EntityTable.build(schema, 20, SEED).save(tmp_path / name)
    assert E.EntityTable.load(tmp_path / name, schema).n_entities == 20


def test_load_memory_maps_by_default(tmp_path):
    """
    The claim the class makes about dataloader workers, asserted rather than assumed.

    ``np.load`` silently ignores ``mmap_mode`` for a zip archive, so the previous ``.npz`` layout
    memory-mapped nothing and eight workers would each have held their own 206 MB.
    """
    schema = bios()
    E.EntityTable.build(schema, 2_000, SEED).save(tmp_path / "t")

    mapped = E.EntityTable.load(tmp_path / "t", schema, mmap=True)
    assert isinstance(mapped.attributes, np.memmap)
    assert isinstance(mapped.name_indices, np.memmap)

    in_memory = E.EntityTable.load(tmp_path / "t", schema, mmap=False)
    assert not isinstance(in_memory.attributes, np.memmap)


def test_load_refuses_a_table_built_against_different_vocabulary(tmp_path):
    """
    Same pool sizes means identical bits per entity, so no arithmetic check would catch it.

    Only the fingerprint does, and without it every fact would differ while every number in the run
    stayed plausible.
    """
    original_schema = bios()
    E.EntityTable.build(original_schema, 100, SEED).save(tmp_path / "t")

    renamed = list(original_schema.attributes)
    renamed[0] = E.AttributePool(
        renamed[0].name, tuple(f"other_{i}" for i in range(len(renamed[0])))
    )
    other = E.Schema(attributes=tuple(renamed), names=original_schema.names)
    assert other.bits_per_entity == pytest.approx(original_schema.bits_per_entity)

    with pytest.raises(OLMoConfigurationError, match="different schema"):
        E.EntityTable.load(tmp_path / "t", other)


def test_load_refuses_a_table_with_the_wrong_number_of_columns(tmp_path):
    """
    A truncated or half-written table used to load clean and fail hours later inside a worker.

    The fingerprint cannot see it, because the fingerprint describes the schema and not the arrays.
    """
    schema = bios()
    directory = tmp_path / "t"
    E.EntityTable.build(schema, 100, SEED).save(directory)
    np.save(directory / "attributes.npy", np.zeros((100, 2), dtype=np.uint32))

    with pytest.raises(OLMoConfigurationError, match="declares 7 attribute pools"):
        E.EntityTable.load(directory, schema)


def test_load_refuses_an_index_that_points_outside_its_pool(tmp_path):
    """Corruption that keeps the shape still produces an IndexError deep in a dataloader."""
    schema = bios()
    directory = tmp_path / "t"
    table = E.EntityTable.build(schema, 100, SEED)
    table.save(directory)
    corrupt = np.array(table.attributes)
    corrupt[0, 0] = 10_000
    np.save(directory / "attributes.npy", corrupt)

    with pytest.raises(OLMoConfigurationError, match="which has only"):
        E.EntityTable.load(directory, schema)


def test_load_refuses_a_directory_that_is_not_a_table(tmp_path):
    """The message says a table is a directory, because it used to be a file."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(OLMoConfigurationError, match="is missing"):
        E.EntityTable.load(tmp_path / "empty", small_schema())


# --- values resolve, and determinism ---------------------------------------------------------------


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


def test_the_same_seed_gives_the_same_table():
    """Reproducibility from ``(schema, n_entities, seed)`` is what we publish instead of shards."""
    a = E.EntityTable.build(bios(), 5_000, SEED)
    b = E.EntityTable.build(bios(), 5_000, SEED)
    np.testing.assert_array_equal(a.attributes, b.attributes)
    np.testing.assert_array_equal(a.name_indices, b.name_indices)


def test_name_space_is_the_product_of_the_name_pools():
    """The ceiling on n_entities, and it feeds the demand formula's name term."""
    assert small_schema().name_space == 35
    assert bios().name_space == NAME_SPACE
    assert bios().name_space / 6_430_000 > 24


def test_bios_bits_reproduce_the_published_value():
    """47.592 from seven pools, pinning Physics 3.3's 47.6 to 0.01 of a bit."""
    from factcrowd.ladder import rho

    assert bios().bits_per_entity == pytest.approx(rho.BIOS_BITS_PER_ENTITY, abs=0.01)
    assert bios().bits_per_entity == pytest.approx(47.591624, abs=1e-5)
    assert sum(math.log2(len(p)) for p in bios().attributes) == pytest.approx(47.591624, abs=1e-5)
