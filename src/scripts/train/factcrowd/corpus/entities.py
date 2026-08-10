"""
The entity table: N people, each a tuple of attribute values drawn from closed declared pools.

Everything about the fact slice is downstream of this module, and one property is what makes the
experiment measurable: **bits per entity is arithmetic, never an estimate.** Pools are closed and
their sizes are declared, so an entity's attribute values carry exactly ``sum(log2(len(pool)))`` bits
by construction. There is no entropy estimator in the pipeline and no place for one to be wrong.

Three design points that are easy to get backwards.

**Names are keys, not values -- but they still carry information.** Physics 3.3 excludes names from
the per-entity attribute count, because a name is the handle you look a fact up by. It does *not*
exclude them from demand: its bioS formula is ``N·[log2(N0/N) + log2(S0)]``, and the first term is
the cost of knowing *which* N names out of N0 exist. So :attr:`Schema.bits_per_entity` covers
attributes only, and the name term lives in :func:`factcrowd.ladder.rho.demanded_bits`. Names carry a
requirement attributes do not: they must be **unique**, or two entities collide and the corpus
asserts contradictory facts about one key.

**That name term is only honest if the name set is actually pseudorandom, and getting this wrong once
already cost 200,000x.** ``N·log2(N0/N)`` is the entropy of a uniformly random N-subset. An earlier
version of :func:`name_codes` used an affine map ``(a·i + b) mod N0``, which is a bijection -- so
names were unique -- but the resulting *set* is a Beatty set whose sorted gaps take exactly three
distinct values (the three-distance theorem), fully described by ``(N0, N, b)``: at most
``log2(N0) = 27`` bits. The formula was charging 29.8 Mbit for a set worth 27 bits, on the
experiment's own independent variable. :func:`name_codes` now uses a seed-keyed pseudorandom
permutation instead, so the subset is computationally indistinguishable from a random one and the
formula describes what the corpus contains.

**An entity's attributes do not depend on how many entities there are.** At the same seed, entity
12,345 has the same attributes in a 1M table and a 6M table. That is what lets the 25k probe subset
(:attr:`EntityTable.probe_ids`) be the *same* 25k people in every experimental cell, which is what
makes related-reasoning accuracy comparable across the ladder. Attributes are therefore drawn one
column at a time from independent streams. Note that a single ``(N, K)`` block draw would *not* break
this -- numpy fills row-major, so element ``(i, j)`` sits at flat index ``i·K + j`` regardless of N.
What the column-wise draw buys is that **adding an attribute leaves the existing columns unmoved**,
and that any draw whose consumption order depends on N (a ``(K, N)`` draw, for one) is structurally
impossible here.

**Uniform sampling is an instrument decision.** Attribute values are drawn uniformly from their
pools, so attribute demand really is ``N x bits_per_entity``. Under a skewed draw it would not be,
and demand -- the experiment's independent variable -- would stop being computable.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "AttributePool",
    "Schema",
    "EntityTable",
    "name_codes",
    "PROBE_SIZE",
]


PROBE_SIZE = 25_000
"""
Entities covered by the related-reasoning slice, the same ones in every cell.

Constant across the grid so per-entity coverage does not vary with N -- if the slice covered every
entity instead, coverage would swing 20x across the ladder and confound P4. It must fit the smallest
cell, which is 13M at a demand of 0.30 bits/param: **64,180 entities** once the name term is counted,
against 79,397 on an attribute-only count. So 25k leaves a **39k** non-probe comparison group there,
still far above the n >= 2,000 the eval needs.
"""

_INDEX_DTYPE = np.uint32
"""Pool indices are stored as uint32: pools are far under 2^32 and the arrays are memory-mapped."""

_MAX_NAME_SPACE = 2**62
"""
Ceiling on the name universe, below the uint64 range :func:`name_codes` computes in.

Not a limit anyone will meet -- the largest cell needs 6.43M names -- but the name space is a demand
knob (widening it raises demand at fixed entity count), so somebody will turn it, and silent uint64
wraparound is not an acceptable way to find the edge.
"""

_GOLDEN64 = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_FEISTEL_ROUNDS = 4


def _splitmix64(value: np.ndarray) -> np.ndarray:
    """
    The splitmix64 finaliser, vectorised over a uint64 array.

    Chosen over ``np.random.Generator`` deliberately: constructing a Generator costs 7.9 microseconds
    against 0.28 for this, which caps a dataloader worker below the throughput a training node needs
    before any rendering happens. Pure in its input, so it is safe to call per entity.

    :param value: uint64 array to mix.

    :returns: The mixed uint64 array.
    """
    with np.errstate(over="ignore"):
        mixed = value + _GOLDEN64
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _MIX_A
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _MIX_B
        return mixed ^ (mixed >> np.uint64(31))


def _feistel(value: np.ndarray, *, half_bits: int, key: np.uint64) -> np.ndarray:
    """
    A balanced Feistel network over ``[0, 2^(2*half_bits))``, which is a permutation by construction.

    Four rounds of :func:`_splitmix64`. Being a permutation is structural -- a Feistel round is
    invertible whatever the round function does -- so uniqueness does not depend on the mixer's
    quality, only the *pseudorandomness* of the resulting subset does.

    :param value: uint64 array of points in the domain.
    :param half_bits: Half the domain width in bits.
    :param key: Seed-derived key.

    :returns: The permuted array.
    """
    mask = np.uint64((1 << half_bits) - 1)
    shift = np.uint64(half_bits)
    left = value >> shift
    right = value & mask
    for round_index in range(_FEISTEL_ROUNDS):
        with np.errstate(over="ignore"):
            salted = right + key + np.uint64(round_index + 1) * _GOLDEN64
        left, right = right, left ^ (_splitmix64(salted) & mask)
    return (left << shift) | right


def name_codes(entity_ids: np.ndarray, *, name_space: int, seed: int) -> np.ndarray:
    """
    Map entity ids to distinct points in the name space, pseudorandomly and by construction.

    A seed-keyed Feistel network is a permutation of ``[0, 2^(2h))`` for the smallest ``h`` with
    ``4^h >= name_space``; points landing outside ``[0, name_space)`` are re-permuted until they land
    inside (*cycle walking*, which terminates because iterating a permutation from a point traverses
    the cycle containing it, and costs ``4^h / name_space <= 4`` iterations in expectation). The
    result is a bijection on ``[0, name_space)`` **and** a subset indistinguishable from a random one.

    Both halves matter and an earlier version had only the first. An affine ``(a·i + b) mod N0`` map
    is equally bijective, but its image has exactly three distinct sorted gaps and is described by
    ``(N0, N, b)`` -- so ``N·log2(N0/N)`` overstated its information content by five orders of
    magnitude. Since that term is part of the experiment's independent variable, uniqueness alone was
    not enough: the construction has to justify the formula applied to it.

    :param entity_ids: Entity ids to map. Must be a non-negative integer array below ``name_space``.
    :param name_space: Size of the name space, from :attr:`Schema.name_space`.
    :param seed: Seed for the permutation key.

    :returns: A uint64 array of distinct codes, same shape as ``entity_ids``, each below
        ``name_space``.

    :raises OLMoConfigurationError: If ``name_space`` is out of range, if ``entity_ids`` is not a
        non-negative integer array, or if any id is at or above ``name_space`` -- which would mean
        more entities than distinct names.
    """
    if name_space < 2:
        raise OLMoConfigurationError(f"'name_space' must be at least 2, got {name_space}")
    if name_space > _MAX_NAME_SPACE:
        raise OLMoConfigurationError(
            f"'name_space' is {name_space:,}, above the {_MAX_NAME_SPACE:,} ceiling this module "
            f"computes safely in uint64. Beyond it numpy wraps silently and uniqueness stops being "
            f"structural."
        )
    if not np.issubdtype(entity_ids.dtype, np.integer):
        raise OLMoConfigurationError(
            f"'entity_ids' must be an integer array, got dtype {entity_ids.dtype}. A float array "
            f"casts to uint64 by truncation, so 0.4 and 0.9 would collide on one name."
        )
    if entity_ids.size:
        lowest, highest = int(entity_ids.min()), int(entity_ids.max())
        if lowest < 0:
            raise OLMoConfigurationError(
                f"'entity_ids' must be non-negative, got {lowest}. A negative id wraps to a huge "
                f"uint64 rather than raising."
            )
        if highest >= name_space:
            raise OLMoConfigurationError(
                f"entity id {highest:,} does not fit a name space of {name_space:,}. Two entities "
                f"would share a name and the corpus would assert contradictory facts about one "
                f"key; widen the name pools."
            )
    if seed < 0:
        raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")

    half_bits = max(1, (max(1, (name_space - 1).bit_length()) + 1) // 2)
    key = _splitmix64(np.array([seed], dtype=np.uint64))[0]
    codes = entity_ids.astype(np.uint64)
    limit = np.uint64(name_space)

    codes = _feistel(codes, half_bits=half_bits, key=key)
    outside = codes >= limit
    # Bounded by the cycle length, but in practice a handful of passes; the guard is against a bug
    # turning this into a spin rather than against the mathematics.
    for _ in range(1024):
        if not outside.any():
            break
        codes[outside] = _feistel(codes[outside], half_bits=half_bits, key=key)
        outside = codes >= limit
    else:  # pragma: no cover - unreachable for a genuine permutation
        raise OLMoConfigurationError(
            f"cycle walking did not converge for name_space={name_space:,}; the permutation is not "
            f"a permutation, which is a bug in _feistel rather than a configuration error"
        )
    return codes


@dataclass(frozen=True)
class AttributePool:
    """
    One closed pool of values, and the exact number of bits an index into it carries.

    :param name: Attribute name, used in rendering and in the bit-counter's span labels.
    :param values: The closed set of values. Order is meaningful: an entity stores an index.
    :param allow_singleton: Permit a pool of exactly one *active* value, which carries zero bits. Off by
        default because a singleton pool is nearly always a mistake; the exception is the entropy
        axis's ``b=0`` cell, where every entity sharing one value tuple *is* the manipulation.
    :param active: How many leading values entities may actually be assigned, defaulting to all of
        them. The rest stay in the vocabulary and are never sampled.

        This exists to hold the *model* fixed while the information content varies. The entropy axis
        used to size each pool at ``2**(b/4)``, which made the vocabulary a function of the treatment:
        1,920 padded tokens at b=0 against 8,064 at b=32, so the high-entropy cell carried **8.1% more
        parameters and a 4.2x larger softmax** than the cell it was being compared with. Those biases
        run opposite ways -- extra parameters can hide crowding, a bigger softmax can manufacture a
        reasoning decline -- so the axis could not identify anything. Now every cell shares one union
        pool and differs only in how much of it is reachable.
    """

    name: str
    values: Tuple[str, ...]
    allow_singleton: bool = False
    active: Optional[int] = None

    def __post_init__(self) -> None:
        if self.active is not None and not 0 < self.active <= len(self.values):
            raise OLMoConfigurationError(
                f"pool '{self.name}' declares {self.active} active values out of "
                f"{len(self.values)}; it must be positive and no larger than the pool"
            )
        floor = 1 if self.allow_singleton else 2
        if self.active_size < floor:
            raise OLMoConfigurationError(
                f"pool '{self.name}' has {self.active_size} active values; a pool of one carries no "
                f"bits and a pool of none cannot be sampled. Pass allow_singleton=True if a zero-bit "
                f"pool is intended -- the entropy axis's b=0 cell is the one place it is."
            )
        if len(set(self.values)) != len(self.values):
            raise OLMoConfigurationError(
                f"pool '{self.name}' has duplicate values, so an index into it is ambiguous and "
                f"log2(len(values)) overstates the bits it carries"
            )

    @property
    def active_size(self) -> int:
        """How many values entities are actually assigned. The whole pool unless ``active`` is set."""
        return len(self.values) if self.active is None else self.active

    @property
    def bits(self) -> float:
        """
        Bits carried by one index into this pool: ``log2(active_size)``, exactly.

        The *active* size, not the vocabulary size. An unreachable value carries no information about
        an entity, so counting it would overstate demand -- and on the entropy axis every cell would
        then report the union's bits rather than its own.
        """
        return math.log2(self.active_size)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class Schema:
    """
    The attribute pools an entity is built from, plus the name pools it is keyed by.

    :param attributes: Pools whose values are facts to be stored. These set
        :attr:`bits_per_entity`.
    :param names: Pools the entity's name is assembled from, e.g. first / middle / last.
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
        if self.name_space > _MAX_NAME_SPACE:
            raise OLMoConfigurationError(
                f"the name pools express {self.name_space:,} names, above the "
                f"{_MAX_NAME_SPACE:,} ceiling name_codes computes safely in"
            )

    @property
    def bits_per_entity(self) -> float:
        """
        Bits one entity's **attribute values** carry. Exact by construction.

        Name pools are excluded, matching Physics 3.3's per-entity figure. This is *not* the whole
        demand: the ``N·log2(N0/N)`` cost of knowing which names exist is added by
        :func:`factcrowd.ladder.rho.demanded_bits`, which is the function to ask for demand.
        """
        return sum(pool.bits for pool in self.attributes)

    @property
    def name_space(self) -> int:
        """How many distinct names the name pools can express, i.e. the ceiling on ``n_entities``."""
        return math.prod(len(pool) for pool in self.names)

    def fingerprint(self) -> str:
        """
        A stable digest of the pools, for provenance and for refusing a mismatched saved table.

        Every field is length-framed. Without framing a pool *name* could impersonate the next
        pool's header -- a pool called ``a:2:x\\x00y\\x00attribute:b`` collided with a two-pool
        schema carrying twice its bits, and :meth:`EntityTable.load` accepted the wrong table.
        Contrived, but the guarantee is stated absolutely, so it should hold absolutely.

        :returns: A hex digest covering pool kinds, names, sizes and values in order.
        """
        digest = hashlib.sha256()

        def field(raw: bytes) -> None:
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)

        field(b"factcrowd.Schema.v2")
        field(len(self.attributes).to_bytes(8, "big"))
        field(len(self.names).to_bytes(8, "big"))
        for kind, pools in (("attribute", self.attributes), ("name", self.names)):
            for pool in pools:
                field(kind.encode())
                field(pool.name.encode())
                field(len(pool).to_bytes(8, "big"))
                field(pool.active_size.to_bytes(8, "big"))
                for value in pool.values:
                    field(value.encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class EntityTable:
    """
    N entities x K attribute pools of indices. Attribute bits are exact by construction.

    Indices rather than strings: at the largest cell that is ~206 MB against many gigabytes of text,
    and :meth:`load` memory-maps them, so dataloader workers share one copy of the pages instead of
    each holding its own.

    Build with :meth:`build`; construct directly only when loading.
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
        """Exact attribute bits per entity, from :attr:`Schema.bits_per_entity`."""
        return self.schema.bits_per_entity

    @property
    def attribute_bits(self) -> float:
        """
        Attribute bits this corpus carries: ``n_entities * bits_per_entity``.

        **Not the corpus's demand.** Demand also includes the ``N·log2(N0/N)`` name term, which is
        8.9% to 18.8% more depending on N, so a cell placed by this number alone sits at the wrong x.
        Call :func:`factcrowd.ladder.rho.demanded_bits` or :func:`factcrowd.ladder.rho.demand` for
        demand. This property exists for the bit-counter, which measures attribute value tokens and
        needs the matching denominator.
        """
        return self.n_entities * self.bits_per_entity

    @property
    def probe_ids(self) -> np.ndarray:
        """
        The related-reasoning probe subset: the first :data:`PROBE_SIZE` entity ids.

        The first ids rather than a random draw, because every cell must cover the *same* people for
        related-reasoning accuracy to be comparable across the ladder, and every cell contains ids
        ``0..PROBE_SIZE`` by construction while a random subset of N would not.
        """
        return np.arange(min(PROBE_SIZE, self.n_entities), dtype=_INDEX_DTYPE)

    def probe_ids_for(self, split: str) -> np.ndarray:
        """
        The ``<compare>`` entity pool for one split, and the two are **disjoint**.

        THIS IS THE FIX FOR A LEAK THAT MADE THE ENDPOINT UNUSABLE. ``<compare>``'s answer is the earlier
        person's birth-*year value*, so every training item states ``min(year(A), year(B)) = Y`` -- and an
        entity's own year is exactly the maximum answer over the items it appears in, whenever it is the
        earlier one even once. Measured on the first campaign's corpus: 97.1% of years recovered from 400k
        of the 2.63M items, and **99.7% eval accuracy from those alone, with no biographies and no model**.

        Reducing mentions per entity does not fix it. At 1.6 mentions 58% of years are still recovered,
        because one item in which an entity is the earlier one reveals that entity's year outright. The
        pools have to be disjoint instead: no training item mentions an entity the evaluation asks about,
        so triangulation has nothing to trade on and the model must retrieve the years from the biographies
        it read. That is the thing the endpoint was always supposed to measure.

        Even split, train first, so the eval pool is the same people in every cell for the same reason
        :attr:`probe_ids` gives the first ids.

        :param split: ``"train"`` or ``"eval"``.

        :returns: The ids.

        :raises OLMoConfigurationError: On an unknown split, or a table too small to halve.
        """
        pool = self.probe_ids
        if pool.size < 4:
            raise OLMoConfigurationError(
                f"a table of {self.n_entities} entities cannot supply two disjoint <compare> pools"
            )
        half = pool.size // 2
        if split == "train":
            return pool[:half]
        if split == "eval":
            return pool[half : 2 * half]
        raise OLMoConfigurationError(f"unknown split {split!r}; expected 'train' or 'eval'")

    @classmethod
    def build(cls, schema: Schema, n_entities: int, seed: int) -> "EntityTable":
        """
        Generate a table deterministically from ``(schema, n_entities, seed)``.

        Entity ``i``'s attributes do not depend on ``n_entities``, so a 1M table is a prefix of a 6M
        table at the same seed -- which is what lets the probe subset be the same people in every
        cell. Each attribute column is drawn from its own spawned stream, so adding a column also
        leaves the existing ones unmoved.

        :param schema: Attribute and name pools.
        :param n_entities: How many entities to generate.
        :param seed: Root seed. Per-column streams are spawned from it.

        :returns: The table.

        :raises OLMoConfigurationError: If ``n_entities`` or ``seed`` is out of range, or if
            ``n_entities`` exceeds :attr:`Schema.name_space`.
        """
        if n_entities <= 0:
            raise OLMoConfigurationError(f"'n_entities' must be positive, got {n_entities}")
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")
        if n_entities > schema.name_space:
            raise OLMoConfigurationError(
                f"{n_entities:,} entities exceeds the {schema.name_space:,} distinct names the "
                f"name pools can express, so names would collide. Widen the name pools."
            )

        streams = np.random.SeedSequence(seed).spawn(len(schema.attributes))
        attributes = np.empty((n_entities, len(schema.attributes)), dtype=_INDEX_DTYPE)
        for column, (pool, stream) in enumerate(zip(schema.attributes, streams)):
            attributes[:, column] = np.random.default_rng(stream).integers(
                0, pool.active_size, size=n_entities, dtype=_INDEX_DTYPE
            )

        codes = name_codes(
            np.arange(n_entities, dtype=np.uint64), name_space=schema.name_space, seed=seed
        )
        name_indices = np.empty((n_entities, len(schema.names)), dtype=_INDEX_DTYPE)
        remaining = codes
        for offset, pool in enumerate(reversed(schema.names)):  # mixed radix, least significant
            name_indices[:, len(schema.names) - 1 - offset] = remaining % np.uint64(len(pool))
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
        Write the table as a directory of ``.npy`` arrays plus a JSON sidecar.

        A directory of plain ``.npy`` rather than one ``.npz``, for two reasons that both bit. ``npz``
        is a zip archive, so ``np.load`` **silently ignores** ``mmap_mode`` on it -- the memory
        sharing this class claims would not have happened, and eight workers would each have held
        their own 206 MB. And ``np.savez`` appends ``.npz`` to the path while ``np.load`` does not, so
        ``save(p)`` followed by ``load(p)`` failed for any ``p`` without the suffix.

        The pools are not written: a table is only meaningful against the schema that made it, and
        :meth:`load` checks the fingerprint.

        :param path: Destination directory. Created if absent.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "attributes.npy", self.attributes)
        np.save(directory / "name_indices.npy", self.name_indices)
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "format": "factcrowd.EntityTable.v2",
                    "seed": self.seed,
                    "n_entities": self.n_entities,
                    "schema_fingerprint": self.schema.fingerprint(),
                    "bits_per_entity": self.bits_per_entity,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path: PathOrStr, schema: Schema, *, mmap: bool = True) -> "EntityTable":
        """
        Read a table written by :meth:`save`, refusing anything that does not match ``schema``.

        Four checks, because the fingerprint alone let a truncated or corrupt table load clean and
        fail hours later inside a dataloader worker: the schema fingerprint, both array widths, and
        that every index actually points into its pool.

        :param path: Source directory.
        :param schema: The schema the table must have been built against.
        :param mmap: Memory-map the arrays. Leave on: eight workers reading 206 MB each is 1.6 GB
            that does not need to exist.

        :returns: The table.

        :raises OLMoConfigurationError: If the fingerprint, the array shapes or the index ranges
            disagree with ``schema``.
        """
        directory = Path(path)
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise OLMoConfigurationError(
                f"no entity table at {directory}: {meta_path.name} is missing. A table is a "
                f"directory written by save(), not a single file."
            )
        meta = json.loads(meta_path.read_text())

        saved_fingerprint = str(meta.get("schema_fingerprint", ""))
        if saved_fingerprint != schema.fingerprint():
            raise OLMoConfigurationError(
                f"the table at {directory} was built against a different schema (fingerprint "
                f"{saved_fingerprint[:12]} vs {schema.fingerprint()[:12]}). Its indices point into "
                f"other pools, so reading it would silently assert different facts about every "
                f"entity."
            )

        mode = "r" if mmap else None
        attributes = np.load(directory / "attributes.npy", mmap_mode=mode)
        name_indices = np.load(directory / "name_indices.npy", mmap_mode=mode)

        for label, array, pools in (
            ("attributes", attributes, schema.attributes),
            ("name_indices", name_indices, schema.names),
        ):
            if array.ndim != 2 or array.shape[1] != len(pools):
                raise OLMoConfigurationError(
                    f"the table at {directory} has {label} of shape {array.shape}, but the schema "
                    f"declares {len(pools)} {label.rstrip('s')} pools. A truncated or half-written "
                    f"table would otherwise load clean and fail inside a dataloader worker."
                )
            for column, pool in enumerate(pools):
                limit = pool.active_size if label == "attributes" else len(pool)
                if array.shape[0] and int(array[:, column].max()) >= limit:
                    raise OLMoConfigurationError(
                        f"the table at {directory} has an index of "
                        f"{int(array[:, column].max()):,} in {label} column {column} "
                        f"('{pool.name}'), which has only {len(pool):,} values"
                    )
        if attributes.shape[0] != name_indices.shape[0]:
            raise OLMoConfigurationError(
                f"the table at {directory} has {attributes.shape[0]:,} attribute rows against "
                f"{name_indices.shape[0]:,} name rows"
            )

        return cls(
            schema=schema,
            attributes=attributes,
            name_indices=name_indices,
            seed=int(meta["seed"]),
        )
