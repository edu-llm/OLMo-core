"""
The entity table: N people, each a tuple of attribute values drawn from closed declared pools.

Everything about the fact slice is downstream of this module, and one property is what makes the
experiment measurable: **bits per entity is arithmetic, never an estimate.** Pools are closed and
their sizes are declared, so an entity carries exactly ``sum(log2(len(pool)))`` bits by
construction. There is no entropy estimator anywhere in the pipeline and no place for one to be
wrong.

Three design points that are easy to get backwards.

**Names are keys, not values.** Physics 3.3's 47.6 bits/person ignores names, because a name is the
handle you look a fact up by rather than a fact to be stored. So names are excluded from
:attr:`Schema.bits_per_entity`, and instead carry a requirement the attributes do not: they must be
**unique**, or two entities collide and the corpus asserts contradictory facts about one key.
:func:`name_codes` gets uniqueness by construction rather than by rejection sampling.

**An entity's attributes do not depend on how many entities there are.** Cell d576_rho2 and cell
d576_rho4 differ only in N, and at the same seed entity 12,345 has the same attributes in both.
That is what lets the 25k probe subset (:attr:`EntityTable.probe_ids`) be the *same* 25k people in
every cell, which is what makes related-reasoning accuracy comparable across the ladder. It is also
why attributes are drawn per column from independent streams rather than as one block: a single
``(N, K)`` draw would make every entity's values a function of N.

**Uniform sampling is an instrument decision.** Attribute values are drawn uniformly from their
pools, so demanded bits really are ``N x bits_per_entity``. Under a skewed draw they would not be,
and rho -- the experiment's independent variable -- would stop being computable.
"""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "AttributePool",
    "Schema",
    "EntityTable",
    "bios_schema",
    "name_codes",
    "PROBE_SIZE",
]


PROBE_SIZE = 25_000
"""
Entities covered by the related-reasoning slice, the same ones in every cell.

Constant across the grid so per-entity coverage does not vary with N -- if the slice covered every
entity instead, coverage would swing 20x across the ladder and confound P4. 25k fits inside the
smallest cell (13M at rho=0.25 has ~79k entities), leaves a ~54k non-probe comparison group there,
and stays far above the n>=2,000 the eval needs.
"""

_INDEX_DTYPE = np.uint32
"""Pool indices are stored as uint32: pools are far under 2^32 and the table is memory-mapped."""


@dataclass(frozen=True)
class AttributePool:
    """
    One closed pool of values, and the exact number of bits an index into it carries.

    :param name: Attribute name, used in rendering and in the bit-counter's span labels.
    :param values: The closed set of values. Order is meaningful: an entity stores an index.
    :param allow_singleton: Permit a pool of exactly one value, which carries zero bits. Off by
        default because a singleton pool is nearly always a mistake; the exception is the entropy
        axis's ``b=0`` cell, where every entity sharing one value tuple *is* the manipulation.
    """

    name: str
    values: Tuple[str, ...]
    allow_singleton: bool = False

    def __post_init__(self) -> None:
        floor = 1 if self.allow_singleton else 2
        if len(self.values) < floor:
            raise OLMoConfigurationError(
                f"pool '{self.name}' has {len(self.values)} values; a pool of one carries no bits "
                f"and a pool of none cannot be sampled. Pass allow_singleton=True if a "
                f"zero-bit pool is intended -- the entropy axis's b=0 cell is the one place it is."
            )
        if len(set(self.values)) != len(self.values):
            raise OLMoConfigurationError(
                f"pool '{self.name}' has duplicate values, so an index into it is ambiguous and "
                f"log2(len(values)) overstates the bits it carries"
            )

    @property
    def bits(self) -> float:
        """Bits carried by one index into this pool: ``log2(len(values))``, exactly."""
        return math.log2(len(self.values))

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Schema:
    """
    The attribute pools an entity is built from, plus the name pools it is keyed by.

    :param attributes: Pools whose values are facts to be stored. These set
        :attr:`bits_per_entity`.
    :param names: Pools the entity's name is assembled from, e.g. first / middle / last. Excluded
        from :attr:`bits_per_entity` -- see the module docstring.
    """

    attributes: Tuple[AttributePool, ...]
    names: Tuple[AttributePool, ...]

    def __post_init__(self) -> None:
        if not self.attributes:
            raise OLMoConfigurationError("a schema needs at least one attribute pool")
        if not self.names:
            raise OLMoConfigurationError(
                "a schema needs at least one name pool: entities are looked up by name, and "
                "without one there is no key to attach a fact to"
            )
        duplicates = {p.name for p in self.attributes} & {p.name for p in self.names}
        if duplicates:
            raise OLMoConfigurationError(
                f"pool names must be distinct across attributes and names, got {sorted(duplicates)}"
            )

    @property
    def bits_per_entity(self) -> float:
        """
        Bits one entity carries, summed over attribute pools. Exact by construction.

        Name pools are excluded: a name is the key a fact is retrieved by, not a fact. Including
        them would inflate demanded bits and put every cell at a lower true rho than its label.
        """
        return sum(pool.bits for pool in self.attributes)

    @property
    def name_space(self) -> int:
        """How many distinct names the name pools can express, i.e. the ceiling on ``n_entities``."""
        return math.prod(len(pool) for pool in self.names)

    def fingerprint(self) -> str:
        """
        A stable digest of the pools, for provenance and for refusing a mismatched saved table.

        Covers pool names, sizes, and values in order, because a table generated against different
        vocabulary is a different table even at the same seed and the same bit count.
        """
        digest = hashlib.sha256()
        for kind, pools in (("attribute", self.attributes), ("name", self.names)):
            for pool in pools:
                digest.update(f"{kind}:{pool.name}:{len(pool)}:".encode())
                for value in pool.values:
                    digest.update(f"{value}\x00".encode())
        return digest.hexdigest()


def name_codes(entity_ids: np.ndarray, *, name_space: int, seed: int) -> np.ndarray:
    """
    Map entity ids to distinct points in the name space, by construction rather than by luck.

    ``id -> (a * id + b) mod name_space`` with ``a`` coprime to ``name_space`` is a bijection, so
    distinct ids get distinct codes with no collision check and no rejection loop. ``a`` is taken
    near ``name_space / phi`` so that consecutive ids land far apart, which keeps the generated
    names from looking obviously sequential.

    The codes are a permutation, not a random sample, and the difference is worth stating: adjacent
    ids differ by exactly ``a`` in code space, so the name stream has structure a statistical test
    would find. That is acceptable because names are keys -- each is used for ~200 exposures of one
    entity, and nothing in the measurement depends on their distribution, only on their uniqueness.

    :param entity_ids: Entity ids to map. Must all be below ``name_space``.
    :param name_space: Size of the name space, from :attr:`Schema.name_space`.
    :param seed: Seed for ``b``, so different runs get different names for the same id.

    :returns: An array of codes, same shape as ``entity_ids``, each below ``name_space``.

    :raises OLMoConfigurationError: If any id is at or above ``name_space``, i.e. there are more
        entities than distinct names available.
    """
    if name_space < 2:
        raise OLMoConfigurationError(f"'name_space' must be at least 2, got {name_space}")
    if entity_ids.size and int(entity_ids.max()) >= name_space:
        raise OLMoConfigurationError(
            f"entity id {int(entity_ids.max()):,} does not fit a name space of {name_space:,}. "
            f"Two entities would share a name and the corpus would assert contradictory facts "
            f"about one key; widen the name pools."
        )

    multiplier = int(name_space / ((1 + 5**0.5) / 2)) | 1
    while math.gcd(multiplier, name_space) != 1:
        multiplier += 2
    offset = int(np.random.default_rng(seed).integers(0, name_space))

    codes = (entity_ids.astype(np.uint64) * np.uint64(multiplier) + np.uint64(offset)) % np.uint64(
        name_space
    )
    return codes.astype(np.uint64)


@dataclass(frozen=True)
class EntityTable:
    """
    N entities x K attributes drawn from closed pools. Bits are exact by construction.

    Attributes and name codes are stored as index arrays rather than strings: at the largest cell
    that is ~206 MB against many gigabytes of text, and it memory-maps, so dataloader workers share
    one copy instead of each holding its own.

    Build with :meth:`build`; do not construct directly unless you are loading.
    """

    schema: Schema
    attributes: np.ndarray
    """``(n_entities, len(schema.attributes))`` of pool indices."""

    name_indices: np.ndarray
    """``(n_entities, len(schema.names))`` of indices into the name pools. Rows are distinct."""

    seed: int

    @property
    def n_entities(self) -> int:
        """Number of entities in the table."""
        return int(self.attributes.shape[0])

    @property
    def bits_per_entity(self) -> float:
        """Exact bits per entity, from :attr:`Schema.bits_per_entity`."""
        return self.schema.bits_per_entity

    @property
    def total_bits(self) -> float:
        """Fact bits this corpus makes available: ``n_entities * bits_per_entity``."""
        return self.n_entities * self.bits_per_entity

    @property
    def probe_ids(self) -> np.ndarray:
        """
        The related-reasoning probe subset: the first :data:`PROBE_SIZE` entity ids.

        The first ids rather than a random draw, because every cell must cover the *same* people
        for related-reasoning accuracy to be comparable across the ladder, and every cell contains
        ids ``0..PROBE_SIZE`` by construction while a random subset of N would not.
        """
        return np.arange(min(PROBE_SIZE, self.n_entities), dtype=_INDEX_DTYPE)

    @classmethod
    def build(cls, schema: Schema, n_entities: int, seed: int) -> "EntityTable":
        """
        Generate a table deterministically from ``(schema, n_entities, seed)``.

        Entity ``i``'s attributes do not depend on ``n_entities``: each attribute column is drawn
        from its own stream, so a table of 1M entities is a prefix of a table of 6M at the same
        seed. That is what lets the probe subset be the same people in every cell.

        :param schema: Attribute and name pools.
        :param n_entities: How many entities to generate.
        :param seed: Root seed. Per-column streams are spawned from it.

        :returns: The table.

        :raises OLMoConfigurationError: If ``n_entities`` is not positive or exceeds
            :attr:`Schema.name_space`.
        """
        if n_entities <= 0:
            raise OLMoConfigurationError(f"'n_entities' must be positive, got {n_entities}")
        if n_entities > schema.name_space:
            raise OLMoConfigurationError(
                f"{n_entities:,} entities exceeds the {schema.name_space:,} distinct names the "
                f"name pools can express, so names would collide. Widen the name pools."
            )

        # One SeedSequence child per attribute column. Drawing column-wise is what makes entity
        # i's values independent of n_entities; a single (N, K) block draw would not.
        streams = np.random.SeedSequence(seed).spawn(len(schema.attributes))
        attributes = np.empty((n_entities, len(schema.attributes)), dtype=_INDEX_DTYPE)
        for column, (pool, stream) in enumerate(zip(schema.attributes, streams)):
            attributes[:, column] = np.random.default_rng(stream).integers(
                0, len(pool), size=n_entities, dtype=_INDEX_DTYPE
            )

        entity_ids = np.arange(n_entities, dtype=np.uint64)
        codes = name_codes(entity_ids, name_space=schema.name_space, seed=seed)
        name_indices = np.empty((n_entities, len(schema.names)), dtype=_INDEX_DTYPE)
        remaining = codes
        for column, pool in enumerate(reversed(schema.names)):  # mixed radix, least significant
            name_indices[:, len(schema.names) - 1 - column] = remaining % np.uint64(len(pool))
            remaining = remaining // np.uint64(len(pool))

        return cls(schema=schema, attributes=attributes, name_indices=name_indices, seed=seed)

    def attribute_values(self, entity_id: int) -> Tuple[str, ...]:
        """
        The attribute values of one entity, as strings.

        :param entity_id: Which entity.

        :returns: One value per attribute pool, in schema order.
        """
        row = self.attributes[entity_id]
        return tuple(pool.values[int(row[i])] for i, pool in enumerate(self.schema.attributes))

    def name_parts(self, entity_id: int) -> Tuple[str, ...]:
        """
        The name parts of one entity, as strings.

        :param entity_id: Which entity.

        :returns: One value per name pool, in schema order.
        """
        row = self.name_indices[entity_id]
        return tuple(pool.values[int(row[i])] for i, pool in enumerate(self.schema.names))

    def save(self, path: PathOrStr) -> None:
        """
        Write the index arrays and the schema fingerprint to a single ``.npz``.

        The pools themselves are not written: they come from the schema, and a table is only
        meaningful against the schema that made it. :meth:`load` checks the fingerprint so a table
        can never be read against different vocabulary.

        :param path: Destination file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(path),
            attributes=self.attributes,
            name_indices=self.name_indices,
            seed=np.asarray(self.seed),
            schema_fingerprint=np.asarray(self.schema.fingerprint()),
        )

    @classmethod
    def load(cls, path: PathOrStr, schema: Schema, *, mmap: bool = True) -> "EntityTable":
        """
        Read a table written by :meth:`save`, refusing a schema it was not built against.

        :param path: Source file.
        :param schema: The schema the table must have been built against.
        :param mmap: Memory-map the arrays instead of reading them into memory. Leave on: eight
            dataloader workers reading 206 MB each is 1.6 GB that does not need to exist.

        :returns: The table.

        :raises OLMoConfigurationError: If the saved fingerprint does not match ``schema``.
        """
        loaded = np.load(Path(path), mmap_mode="r" if mmap else None)
        saved_fingerprint = str(loaded["schema_fingerprint"])
        if saved_fingerprint != schema.fingerprint():
            raise OLMoConfigurationError(
                f"the table at {path} was built against a different schema "
                f"(fingerprint {saved_fingerprint[:12]} vs {schema.fingerprint()[:12]}). Its "
                f"indices point into other pools, so reading it would silently assert different "
                f"facts about every entity."
            )
        return cls(
            schema=schema,
            attributes=loaded["attributes"],
            name_indices=loaded["name_indices"],
            seed=int(loaded["seed"]),
        )


def _placeholder_pool(name: str, size: int, prefix: str) -> AttributePool:
    """Build a pool of ``size`` distinct placeholder values. See :func:`bios_schema`."""
    width = len(str(size - 1))
    return AttributePool(name=name, values=tuple(f"{prefix}{i:0{width}d}" for i in range(size)))


def bios_schema(
    *,
    name_pool_sizes: Sequence[int] = (400, 400, 1000),
    vocabulary: Optional[Sequence[AttributePool]] = None,
) -> Schema:
    """
    The bioS schema: pool sizes chosen so bits per entity reproduces Physics 3.3's 47.6.

    Four categorical attributes at 200 / 300 / 100 / 263 choices plus a birth date over
    12 x 28 x 400 = 134,400, giving 47.592 bits. Physics 3.3 publishes the total rather than the
    factorisation, so this is our reconstruction of it -- but it pins their figure to within 0.01
    of a bit, and in any case it is the schema *we* use, which is what makes the bit-counts
    comparable.

    .. warning::
        **The default vocabulary is placeholders, and it must be replaced before M1.** Bit
        accounting depends only on pool *sizes*, so it is already exact -- but two things that
        matter depend on the actual strings: tokens per biography (the budget in
        :mod:`factcrowd.ladder.rho` assumes ~100) and the 32k BPE trained on the rendered corpus.
        ``employer_017`` and ``Princeton University`` do not tokenize alike. Pass ``vocabulary``
        with real city, university, major and employer names to fix it; the sizes must match.

    :param name_pool_sizes: Sizes of the first / middle / last name pools. The product is the
        ceiling on ``n_entities``; the default 400 x 400 x 1000 gives 160M, well above the 6.43M of
        the largest cell.
    :param vocabulary: Real attribute pools to use instead of the placeholders. Must have the same
        names and sizes as the defaults.

    :returns: The schema.

    :raises OLMoConfigurationError: If ``vocabulary`` does not match the required names and sizes.
    """
    required = (("birth_city", 200), ("university", 300), ("major", 100), ("employer", 263))
    birth_date_size = 12 * 28 * 400

    if vocabulary is None:
        attributes = tuple(_placeholder_pool(name, size, f"{name}_") for name, size in required) + (
            _placeholder_pool("birth_date", birth_date_size, "date_"),
        )
    else:
        supplied = {pool.name: pool for pool in vocabulary}
        expected = dict(required + (("birth_date", birth_date_size),))
        if set(supplied) != set(expected):
            raise OLMoConfigurationError(
                f"'vocabulary' must supply exactly {sorted(expected)}, got {sorted(supplied)}"
            )
        for pool_name, size in expected.items():
            if len(supplied[pool_name]) != size:
                raise OLMoConfigurationError(
                    f"pool '{pool_name}' must have {size} values to keep bits per entity at "
                    f"{47.6}, got {len(supplied[pool_name])}. Changing a pool size changes the "
                    f"x-axis of every plot."
                )
        attributes = tuple(supplied[pool_name] for pool_name, _ in required) + (
            supplied["birth_date"],
        )

    names = tuple(
        _placeholder_pool(pool_name, size, f"{pool_name}_")
        for pool_name, size in zip(("first_name", "middle_name", "last_name"), name_pool_sizes)
    )
    return Schema(attributes=attributes, names=names)
