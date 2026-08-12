"""Synthetic entities with attribute pools that can be split train/novel.

## What changed and why

The previous generator drew every attribute from one global pool regardless of
seed, so a "novel" population got new **names** but reused all the same
**values**. Only the first hop of a composition was therefore out of
distribution, and the headline 53.5%-vs-0.8% novel-entity result measured
first-hop *access* for unfamiliar names rather than composition over new facts.

Here the pools are large enough to be **partitioned**, so a novel population can
be given values the model has never seen. `split_pools()` returns two disjoint
halves and asserts disjointness. The binding constraint in the old generator was
`major`, which had only 125 constructible values against 100 used -- no room for
a split at all. The generators below carry >=4x headroom on every attribute.

## Bit accounting

`bits_per_entity()` sums log2(pool size) over the attributes actually used, so
the number moves when you change the pools instead of being a stale constant. Two
notes carried forward from the audit:

* `birth_date` is not a pool. It is a day range, and in the old corpus it carried
  27.9% of the ~53 bits per entity on its own. It is excluded from
  `COMPOSE_ATTRS` here (its long punctuated surface is also awkward under exact
  match) and must be counted separately if you enable it.
* The old "~a few hundred values per attribute" description was only true of four
  of six attributes. Report the actual sizes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

ATTRIBUTES: tuple[str, ...] = (
    "birth_city",
    "university",
    "major",
    "employer",
    "current_city",
)

RELATION_PHRASES: dict[str, str] = {
    "birth_city": "city of birth",
    "university": "university",
    "major": "field of study",
    "employer": "employer",
    "current_city": "current city",
}

# ------------------------------------------------------------------ name space

_ONSETS = ["b", "br", "c", "cl", "d", "dr", "f", "g", "gr", "h", "j", "k", "kr",
           "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr", "v", "w", "z"]
_NUCLEI = ["a", "e", "i", "o", "u", "ae", "ei", "ou", "ia"]
_CODAS = ["", "l", "n", "r", "s", "m", "th", "sk"]
_ENDS = ["a", "o", "e", "is", "us", "en", "ar", "el"]

_LAST_ROOTS = ["ash", "bell", "cliff", "dale", "elm", "fern", "gold", "hawk",
               "iron", "jade", "kell", "lark", "moss", "north", "oak", "pine",
               "quill", "rook", "stone", "thorn", "vale", "west", "yew", "birch",
               "crane", "drift", "ember", "frost", "grove", "hollow"]
_LAST_SUFFIXES = ["berg", "born", "brook", "burn", "den", "field", "ford", "gate",
                  "haven", "hill", "ley", "mere", "mont", "ridge", "shaw", "stead",
                  "ton", "vale", "wick", "wood", "well", "worth", "crest", "march"]


def _syllables(cap: int) -> list[str]:
    out: list[str] = []
    for o in _ONSETS:
        for n in _NUCLEI:
            for e in _ENDS:
                out.append((o + n + e).capitalize())
                if len(out) >= cap:
                    return out
    return out


def _last_names(cap: int) -> list[str]:
    out: list[str] = []
    for r in _LAST_ROOTS:
        for s in _LAST_SUFFIXES:
            if r.endswith(s[0]):
                continue  # avoid doubled fragments like "Ashshaw"
            out.append((r + s).capitalize())
            if len(out) >= cap:
                return out
    return out


FIRST_NAMES = _syllables(600)
MIDDLE_NAMES = _syllables(320)[:320]
LAST_NAMES = _last_names(700)

# ---------------------------------------------------------------- value pools

_CITY_A = ["Alder", "Aspen", "Birch", "Cedar", "Clear", "Copper", "Crane",
           "Cypress", "Drift", "Ember", "Fern", "Frost", "Gold", "Grove",
           "Hollow", "Iron", "Jade", "Lark", "Maple", "Moss", "North", "Oak",
           "Pine", "Quarry", "Ridge", "Rook", "Slate", "Stone", "Thorn", "Vale",
           "West", "Willow", "Yew", "Amber", "Basalt", "Chalk"]
_CITY_B = ["Point", "Springs", "Grove", "View", "Hollow", "Crossing", "Falls",
           "Harbor", "Landing", "Mills", "Reach", "Bend", "Ford", "Gate",
           "Heights", "Meadow", "Bluff", "Cove"]

_UNI_A = ["Ashford", "Brenton", "Calder", "Dunmore", "Eastvale", "Fairholm",
          "Glenmoor", "Harrowe", "Ilburn", "Kestrel", "Lindmere", "Marchwood",
          "Northgate", "Orrick", "Pemberly", "Quinlan", "Ravensmere",
          "Stonehill", "Thornbury", "Underhill", "Vantage", "Whitmore",
          "Yarrow", "Blackmoor", "Cranmere", "Dovewell"]
_UNI_B = ["University", "Institute", "College", "Polytechnic", "Academy",
          "State University", "Technical Institute", "School of Sciences"]

_MAJOR_FIELD = ["Biology", "Chemistry", "Physics", "Mathematics", "Geology",
                "Linguistics", "Economics", "Psychology", "Sociology",
                "Astronomy", "Meteorology", "Botany", "Zoology", "Ecology",
                "Statistics", "Philosophy", "Archaeology", "Anthropology",
                "Cartography", "Hydrology", "Metallurgy", "Optics",
                "Acoustics", "Genetics", "Immunology", "Neuroscience"]
_MAJOR_MOD = ["Applied", "Theoretical", "Computational", "Experimental",
              "Comparative", "Molecular", "Historical", "Quantitative",
              "Structural", "Environmental", "Industrial", "Clinical"]

_EMP_A = ["Aldridge", "Brightwater", "Corvid", "Dunlin", "Everline", "Fenmark",
          "Greyfell", "Halcyon", "Ironvale", "Junipel", "Kestrelon", "Larkspur",
          "Meridian", "Northwind", "Orielle", "Pinnacle", "Quillon", "Redshale",
          "Silverbrook", "Tessellate", "Umbral", "Verdance", "Wolvesey",
          "Xanthe", "Yarrowin", "Zephyra"]
_EMP_B = ["Analytics", "Dynamics", "Industries", "Laboratories", "Logistics",
          "Instruments", "Systems", "Works", "Foundry", "Collective",
          "Partners", "Research Group", "Consortium", "Holdings"]


def _cross(a: list[str], b: list[str], joiner: str = " ") -> list[str]:
    return [f"{x}{joiner}{y}" for x in a for y in b]


def _build_pools() -> dict[str, list[str]]:
    cities = _cross(_CITY_A, _CITY_B)               # 36*18 = 648
    unis = _cross(_UNI_A, _UNI_B)                   # 26*8  = 208 -> widen below
    unis += _cross([f"{a} {b}" for a in _UNI_A[:12] for b in ["North", "South"]],
                   ["University", "College"])       # +48
    majors = _cross(_MAJOR_MOD, _MAJOR_FIELD)       # 12*26 = 312
    employers = _cross(_EMP_A, _EMP_B)              # 26*14 = 364
    pools = {
        "birth_city": cities,
        # current_city deliberately draws from a DISJOINT slice of the city
        # space. Sharing one list (as before) makes ~1/200 of entities match by
        # coincidence, which then has to be compensated for downstream in the
        # same-city question family.
        "current_city": cities,
        "university": unis,
        "major": majors,
        "employer": employers,
    }
    return pools


VALUE_POOLS: dict[str, list[str]] = _build_pools()

# Half of the city space is reserved for current_city so the two city
# attributes cannot collide.
_HALF = len(VALUE_POOLS["birth_city"]) // 2
VALUE_POOLS["birth_city"] = VALUE_POOLS["birth_city"][:_HALF]
VALUE_POOLS["current_city"] = VALUE_POOLS["current_city"][_HALF:]

MIN_POOL = 150  # every attribute must support a 50/50 train/novel split


def _check_pools(pools: dict[str, list[str]]) -> None:
    for attr in ATTRIBUTES:
        pool = pools[attr]
        if len(pool) != len(set(pool)):
            raise AssertionError(f"{attr}: pool has duplicates")
        if len(pool) < MIN_POOL:
            raise AssertionError(
                f"{attr}: pool is {len(pool)}, need >= {MIN_POOL} to support a "
                "disjoint train/novel split"
            )
    if set(pools["birth_city"]) & set(pools["current_city"]):
        raise AssertionError("birth_city and current_city pools overlap")


_check_pools(VALUE_POOLS)


def pool_sizes(pools: dict[str, list[str]] | None = None) -> dict[str, int]:
    pools = pools or VALUE_POOLS
    return {a: len(pools[a]) for a in ATTRIBUTES}


def bits_per_entity(pools: dict[str, list[str]] | None = None) -> float:
    """Sum of log2(pool size) over the attributes in use."""
    pools = pools or VALUE_POOLS
    return sum(math.log2(len(pools[a])) for a in ATTRIBUTES)


def chance_accuracy(attr: str, pools: dict[str, list[str]] | None = None) -> float:
    """Exact-match chance for a single attribute. Report this on every table."""
    pools = pools or VALUE_POOLS
    return 1.0 / len(pools[attr])


def split_pools(
    seed: int = 0, novel_frac: float = 0.5
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Partition every pool into disjoint (train, novel) halves.

    This is what makes a genuinely-novel population possible: a novel entity gets
    both an unseen name and unseen attribute *values*, so every hop of a
    composition over it is out of distribution rather than just the first.
    """
    train: dict[str, list[str]] = {}
    novel: dict[str, list[str]] = {}
    for attr in ATTRIBUTES:
        pool = list(VALUE_POOLS[attr])
        random.Random(f"poolsplit:{seed}:{attr}").shuffle(pool)
        cut = int(round(len(pool) * (1.0 - novel_frac)))
        train[attr], novel[attr] = pool[:cut], pool[cut:]
        assert not set(train[attr]) & set(novel[attr])
        assert train[attr] and novel[attr]
    return train, novel


# -------------------------------------------------------------------- records


@dataclass
class BioRecord:
    entity_id: int
    name: str
    attrs: dict[str, str] = field(default_factory=dict)


def generate_records(
    n: int,
    seed: int = 0,
    pools: dict[str, list[str]] | None = None,
    name_offset: int = 0,
) -> list[BioRecord]:
    """Generate `n` entities, prefix-stably.

    The RNG is consumed strictly in sequence so `generate_records(n)` is a prefix
    of `generate_records(n + k)` -- builds are reproducible and a larger corpus
    is a superset of a smaller one.

    `name_offset` shifts the name space so a novel population cannot collide with
    a training population by chance even before the pool split.
    """
    pools = pools or VALUE_POOLS
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[BioRecord] = []
    for i in range(n):
        for _ in range(64):
            first = FIRST_NAMES[(rng.randrange(len(FIRST_NAMES)) + name_offset) % len(FIRST_NAMES)]
            mid = MIDDLE_NAMES[rng.randrange(len(MIDDLE_NAMES))]
            last = LAST_NAMES[(rng.randrange(len(LAST_NAMES)) + name_offset) % len(LAST_NAMES)]
            name = f"{first} {mid} {last}"
            if name not in seen:
                break
        else:  # pragma: no cover
            raise RuntimeError("exhausted name space")
        seen.add(name)
        attrs = {a: pools[a][rng.randrange(len(pools[a]))] for a in ATTRIBUTES}
        out.append(BioRecord(entity_id=i, name=name, attrs=attrs))
    return out


def name_space_size() -> int:
    return len(FIRST_NAMES) * len(MIDDLE_NAMES) * len(LAST_NAMES)
