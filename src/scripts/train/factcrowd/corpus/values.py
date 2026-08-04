"""
How an attribute value is composed out of pools, which is what makes the entropy axis possible.

An attribute's value is one or more **words**, each drawn from its own closed pool. That single idea
covers both of the experiment's axes without a second representation:

* **The count axis (bioS).** Seven attributes, one word each, pools of 200 / 300 / 100 / 263 for the
  categorical fields and 12 / 28 / 400 for a decomposed birth date. Bits per entity is 47.592, which
  pins Physics 3.3's published 47.6.
* **The entropy axis.** Six attributes, **four** words each, every pool the same size ``2^(b/4)``. So
  bits per attribute is exactly ``b`` and bits per entity is ``6b`` -- while the token count is
  *invariant in b*, because the number of words never changes. That invariance is the whole point:
  it holds tokens, steps, schedule position, mixture ratio and cumulative weight decay fixed while
  demand sweeps, which the count axis cannot do (PRD.md §3.1).

The two agree where it matters. At ``b = 8`` -- four words from pools of 4 -- bits per entity is 48
against bioS's 47.592, a 0.9% match, so the entropy axis anchors to the literature at its midpoint.

**Why the birth date is three attributes rather than one.** A single 12x28x400 = 134,400-value pool
carries the same 17.036 bits, and it is the natural reading of "one attribute." But it would need
134,400 word types in the vocabulary, and at ``d_model=256`` that embedding table is 34.4M parameters
against a 12.6M model -- the table would be nearly three times the thing whose capacity we are
measuring. Decomposed, it is 440 word types and the bits are identical. This supersedes
:func:`factcrowd.corpus.entities.bios_schema`, which enumerated the dates.

**Words are generated, pronounceable and unique.** Not placeholders like ``employer_017``: those
tokenize unlike real text, and both the token budget and the BPE depend on the strings. Generated
syllable words ("Bellmont", "Carwick") read as English, tokenize as one word each under a BPE trained
on this corpus, and -- unlike a real-world word list -- come in any quantity we need with no
licensing question and no frequency skew. The review's point that a 32-bit value must stay *natural*
is satisfied by composition: four ordinary-looking words, not one random string.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from olmo_core.exceptions import OLMoConfigurationError

from .entities import AttributePool, Schema

__all__ = [
    "ValueSpec",
    "CorpusSchema",
    "ENTROPY_ATTRIBUTES",
    "ENTROPY_VOCABULARY_BITS",
    "ENTROPY_WORDS_PER_VALUE",
    "allocate_words",
    "REAL_WORDS_BY_POOL",
    "bios_schema",
    "entropy_schema",
    "bits_per_entity_for",
    "bios_bits_per_entity",
    "BIOS_POOL_SIZES",
    "DEFAULT_NAME_POOL_SIZES",
    "NAME_SPACE",
]


ENTROPY_VOCABULARY_BITS = 32
"""
The entropy sweep's maximum ``bits_per_attribute``, which sizes the union vocabulary for every cell.

Every cell in a sweep must pass the same value, or the axis varies the model alongside the information
content it is supposed to isolate. A sweep-level constant rather than a per-cell field for that reason.
"""

ENTROPY_ATTRIBUTES = 6
"""
Attributes on the entropy axis, chosen so bits per entity is ``6b`` and ``b=8`` lands on bioS.

Six rather than bioS's seven because ``6 x 8 = 48`` is the closest clean multiple to 47.592. The
count is fixed across the sweep; only the pool size varies.
"""

ENTROPY_WORDS_PER_VALUE = 4
"""
Words per attribute value on the entropy axis, held **constant** across the sweep.

This is the invariant that makes the axis iso-token. Varying the number of words instead of the pool
size would sweep bits and tokens together, which is the confound the axis exists to remove. Four is
the smallest count that reaches 32 bits per attribute from pools small enough to keep the vocabulary
modest (``2^(32/4) = 256`` words per pool).
"""

_CONSONANT_ONSETS = (
    "b",
    "br",
    "c",
    "ch",
    "cl",
    "cr",
    "d",
    "dr",
    "f",
    "fl",
    "fr",
    "g",
    "gl",
    "gr",
    "h",
    "j",
    "k",
    "kn",
    "l",
    "m",
    "n",
    "p",
    "ph",
    "pl",
    "pr",
    "qu",
    "r",
    "s",
    "sh",
    "sk",
    "sl",
    "sm",
    "sn",
    "sp",
    "st",
    "sw",
    "t",
    "th",
    "tr",
    "tw",
    "v",
    "w",
    "wh",
    "y",
    "z",
)
_VOWELS = ("a", "e", "i", "o", "u", "ai", "ea", "ee", "ie", "oa", "oo", "ou", "ow", "ay")
_CODAS = (
    "b",
    "ck",
    "d",
    "ft",
    "g",
    "gh",
    "ld",
    "ll",
    "lm",
    "ln",
    "m",
    "mp",
    "n",
    "nd",
    "ng",
    "nt",
    "p",
    "rd",
    "rk",
    "rn",
    "rt",
    "sh",
    "sk",
    "sp",
    "ss",
    "st",
    "t",
    "th",
    "x",
    "zz",
)


_SYLLABLE_SPACE = len(_CONSONANT_ONSETS) * len(_VOWELS) * len(_CODAS)
"""45 onsets x 14 vowels x 30 codas = 18,900 single-syllable words before a second is appended."""


def _syllable(index: int) -> str:
    """The ``index``-th syllable in a fixed enumeration order."""
    index %= _SYLLABLE_SPACE
    coda = _CODAS[index % len(_CODAS)]
    index //= len(_CODAS)
    vowel = _VOWELS[index % len(_VOWELS)]
    index //= len(_VOWELS)
    onset = _CONSONANT_ONSETS[index % len(_CONSONANT_ONSETS)]
    return onset + vowel + coda


def _stride(index: int) -> str:
    """
    Walk the syllable space in a coprime stride so consecutive allocations look unrelated.

    A naive ``index -> _syllable(index)`` hands adjacent blocks near-identical words, because the
    enumeration's fastest-varying digit is the coda: "Pouss Poust Pout Pouth" was a real output of
    that. 7919 is coprime to 18,900, so the stride is a permutation of the space and consecutive
    words differ in onset and vowel as well.
    """
    word = _syllable(index * 7919)
    if index >= _SYLLABLE_SPACE:  # exhausted one syllable, append a second
        word += _syllable((index // _SYLLABLE_SPACE) * 7919)
    return word.capitalize()


# --- real vocabulary, mixed with generated words ---------------------------------------------------
#
# Single words only: the corpus is tokenised word-by-word, so "New York" would be two tokens and would
# break the fixed one-word-per-pool-value accounting. Real words are preferred over generated ones and
# the pools are topped up from the syllable generator, which gives text that reads naturally where it
# can and stays exactly as countable where it cannot. Any real word that collides with a template
# literal or another pool is skipped rather than deduplicated, so disjointness still holds.

REAL_CITIES: Tuple[str, ...] = (
    "Boston",
    "Denver",
    "Bristol",
    "Leeds",
    "Oslo",
    "Bergen",
    "Turin",
    "Naples",
    "Lyon",
    "Nantes",
    "Ghent",
    "Bruges",
    "Utrecht",
    "Aarhus",
    "Malmo",
    "Uppsala",
    "Tampere",
    "Tallinn",
    "Riga",
    "Vilnius",
    "Krakow",
    "Gdansk",
    "Brno",
    "Graz",
    "Basel",
    "Lucerne",
    "Verona",
    "Padua",
    "Bologna",
    "Perugia",
    "Cordoba",
    "Granada",
    "Valencia",
    "Porto",
    "Coimbra",
    "Salzburg",
    "Bonn",
    "Mainz",
    "Freiburg",
    "Kassel",
    "Erfurt",
    "Rostock",
    "Aalborg",
    "Odense",
    "Galway",
    "Limerick",
    "Dundee",
    "Aberdeen",
    "Norwich",
    "Exeter",
    "Durham",
    "Lancaster",
    "Preston",
    "Swansea",
    "Newport",
    "Hobart",
    "Geelong",
    "Dunedin",
    "Nelson",
    "Rotorua",
    "Halifax",
    "Regina",
    "Kelowna",
    "Sudbury",
    "Kingston",
    "Guelph",
    "Moncton",
    "Spokane",
    "Boise",
    "Fresno",
    "Tucson",
    "Wichita",
    "Omaha",
    "Madison",
    "Duluth",
    "Peoria",
    "Akron",
    "Toledo",
    "Dayton",
    "Trenton",
    "Hartford",
    "Worcester",
)

REAL_UNIVERSITY_NAMES: Tuple[str, ...] = (
    "Princeton",
    "Cornell",
    "Rutgers",
    "Purdue",
    "Emory",
    "Tufts",
    "Brandeis",
    "Villanova",
    "Fordham",
    "Marquette",
    "Creighton",
    "Gonzaga",
    "Baylor",
    "Bucknell",
    "Colgate",
    "Haverford",
    "Swarthmore",
    "Oberlin",
    "Carleton",
    "Grinnell",
    "Kenyon",
    "Bowdoin",
    "Amherst",
    "Wesleyan",
    "Vassar",
    "Skidmore",
    "Bard",
    "Hampshire",
    "Middlebury",
    "Dartmouth",
    "Georgetown",
    "Loyola",
    "DePaul",
    "Xavier",
    "Duquesne",
    "Drexel",
    "Lehigh",
    "Clarkson",
    "Rensselaer",
    "Stevens",
    "Worcester",
    "Rochester",
    "Syracuse",
    "Binghamton",
    "Buffalo",
    "Albany",
    "Towson",
    "Salisbury",
    "Radford",
    "Longwood",
    "Elon",
    "Furman",
    "Mercer",
    "Samford",
    "Stetson",
    "Rollins",
    "Bradley",
    "Butler",
    "Valparaiso",
    "Wabash",
    "Denison",
    "Wooster",
    "Otterbein",
    "Hiram",
    "Ashland",
)

REAL_MAJORS: Tuple[str, ...] = (
    "Physics",
    "Chemistry",
    "Biology",
    "Mathematics",
    "Statistics",
    "Economics",
    "History",
    "Philosophy",
    "Linguistics",
    "Anthropology",
    "Archaeology",
    "Geology",
    "Geography",
    "Sociology",
    "Psychology",
    "Astronomy",
    "Botany",
    "Zoology",
    "Genetics",
    "Ecology",
    "Neuroscience",
    "Immunology",
    "Pharmacology",
    "Physiology",
    "Biochemistry",
    "Metallurgy",
    "Hydrology",
    "Meteorology",
    "Oceanography",
    "Cartography",
    "Musicology",
    "Theology",
    "Jurisprudence",
    "Criminology",
    "Demography",
    "Epidemiology",
    "Entomology",
    "Ornithology",
    "Mycology",
    "Palaeontology",
    "Seismology",
    "Volcanology",
    "Cryptography",
    "Robotics",
    "Optics",
    "Acoustics",
    "Thermodynamics",
    "Aeronautics",
    "Horticulture",
    "Agronomy",
    "Forestry",
    "Architecture",
)

REAL_EMPLOYERS: Tuple[str, ...] = (
    "Siemens",
    "Bosch",
    "Ericsson",
    "Nokia",
    "Philips",
    "Alstom",
    "Thales",
    "Safran",
    "Valeo",
    "Michelin",
    "Renault",
    "Peugeot",
    "Fiat",
    "Iveco",
    "Leonardo",
    "Indra",
    "Repsol",
    "Endesa",
    "Enel",
    "Eni",
    "Statoil",
    "Equinor",
    "Orsted",
    "Vestas",
    "Novozymes",
    "Lundbeck",
    "Genmab",
    "Alfa",
    "Atlas",
    "Sandvik",
    "Skanska",
    "Volvo",
    "Scania",
    "Husqvarna",
    "Electrolux",
    "Assa",
    "Kone",
    "Wartsila",
    "Metso",
    "Outotec",
    "Fortum",
    "Neste",
    "Stora",
    "Fiskars",
    "Amer",
    "Konica",
    "Ricoh",
    "Omron",
    "Kyocera",
    "Murata",
    "Nidec",
    "Makita",
    "Yamaha",
    "Bridgestone",
    "Denso",
)

REAL_FIRST_NAMES: Tuple[str, ...] = (
    "Amara",
    "Bettina",
    "Caius",
    "Dagmar",
    "Eamon",
    "Fenella",
    "Gudrun",
    "Halvard",
    "Ingrid",
    "Jasper",
    "Katinka",
    "Lorcan",
    "Marisol",
    "Nils",
    "Ottoline",
    "Piet",
    "Quentin",
    "Rosalind",
    "Soren",
    "Tamsin",
    "Ulrik",
    "Verity",
    "Wilhelm",
    "Xenia",
    "Yusuf",
    "Zora",
    "Anselm",
    "Beatrix",
    "Casimir",
    "Delphine",
    "Emrys",
    "Fiora",
    "Gaspard",
    "Henrike",
    "Isolde",
    "Joris",
    "Kasimir",
    "Liesel",
    "Matthias",
    "Nadia",
    "Oleg",
    "Perrine",
    "Rasmus",
    "Sigrid",
    "Thibault",
    "Ursula",
    "Vidar",
    "Wanda",
    "Yannick",
    "Zelda",
    "Aurelio",
    "Bianca",
    "Cosimo",
    "Dorotea",
    "Elias",
    "Freya",
    "Gunnar",
    "Helena",
    "Ivar",
    "Jolanta",
    "Klaus",
    "Ludmila",
    "Mikkel",
    "Noor",
    "Osvaldo",
)

REAL_LAST_NAMES: Tuple[str, ...] = (
    "Aaltonen",
    "Bergqvist",
    "Castellan",
    "Dahlgren",
    "Engebretsen",
    "Fabbri",
    "Gustavsson",
    "Halvorsen",
    "Ivanovic",
    "Jakobsen",
    "Kaufmann",
    "Lindqvist",
    "Marchetti",
    "Nordstrom",
    "Olesen",
    "Pedersen",
    "Quaglia",
    "Rasmussen",
    "Sandberg",
    "Thorvaldsen",
    "Ullmann",
    "Vasquez",
    "Wennberg",
    "Ylonen",
    "Zetterberg",
    "Andersson",
    "Brekke",
    "Cavalli",
    "Dziedzic",
    "Eklund",
    "Forsberg",
    "Grimaldi",
    "Hedlund",
    "Isaksen",
    "Jansen",
    "Kowalczyk",
    "Lehtinen",
    "Moretti",
    "Nyberg",
    "Ostberg",
    "Pirlo",
    "Rinaldi",
    "Sjoberg",
    "Tarantino",
    "Uusitalo",
    "Virtanen",
    "Wallenberg",
    "Zieliński",
    "Alvarado",
    "Bittner",
    "Cerny",
    "Dvorak",
    "Eichmann",
    "Farkas",
    "Gerhardt",
    "Horvath",
    "Ingebrigtsen",
    "Janssens",
    "Klimenko",
    "Lombardi",
    "Mazzola",
    "Novotny",
    "Oravec",
    "Pekkarinen",
    "Reinhardt",
    "Salonen",
    "Tikhonov",
    "Urbanek",
    "Vodnik",
)

REAL_WORDS_BY_POOL: Dict[str, Tuple[str, ...]] = {
    "birth_city": REAL_CITIES,
    "university": REAL_UNIVERSITY_NAMES,
    "major": REAL_MAJORS,
    "employer": REAL_EMPLOYERS,
    "first_name": REAL_FIRST_NAMES,
    "middle_name": REAL_FIRST_NAMES,
    "last_name": REAL_LAST_NAMES,
}
"""
Real words to prefer per pool, topped up from the generator.

``middle_name`` reuses the first-name list on purpose -- middle names are first names -- and the
allocator's disjointness pass means the two pools still receive different words from it.

The date pools get no real words: month, day and year indices are rendered as generated words rather
than "March" or "1974", because a real calendar would let a model exploit ordinal structure between
values and the demand formula assumes the pool is unstructured.
"""


def allocate_words(
    pool_sizes: Sequence[Tuple[str, int]], *, reserved: Iterable[str] = ()
) -> Dict[str, Tuple[str, ...]]:
    """
    Hand every pool a block of words that no other pool shares.

    **Disjointness across pools is a correctness requirement, not tidiness.** With a word-level
    vocabulary, a word appearing in both the ``university`` and ``employer`` pools makes that surface
    form ambiguous about which fact it states, so an eval item asking "where did X study" has a
    defensible wrong answer and recall is measured against a moving target. Allocating from one
    cursor makes a collision impossible rather than unlikely.

    Pools are served in sorted-name order so the result depends on the *set* of pools and not on the
    order they were declared. Adding or resizing a pool therefore reshuffles the words of pools after
    it alphabetically -- which is a real cost, accepted because
    :meth:`~factcrowd.corpus.entities.Schema.fingerprint` detects exactly that and refuses a table
    built against the old vocabulary.

    :param pool_sizes: ``(name, size)`` pairs. Names must be unique.
    :param reserved: Words the allocator must not hand out -- template literals, special tokens and
        domain tokens. A generated syllable can coincide with an ordinary English word ("Born" was the
        first real collision), and a word serving as both a template literal and a pool value is
        ambiguous between prose and a fact. Passing them here is what makes the schema fingerprint
        depend on the template set, which is correct: change the templates and the pools really are
        different.

    :returns: Pool name to its distinct word block.

    :raises OLMoConfigurationError: If a name repeats or a size is not positive.
    """
    names = [name for name, _ in pool_sizes]
    if len(names) != len(set(names)):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise OLMoConfigurationError(f"pool names must be unique, got duplicates {duplicated}")
    for name, size in pool_sizes:
        if size <= 0:
            raise OLMoConfigurationError(f"pool '{name}' must have a positive size, got {size}")

    allocated: Dict[str, Tuple[str, ...]] = {}
    taken = set(reserved)
    cursor = 0
    for name, size in sorted(pool_sizes):
        block: List[str] = []
        # Real words first, where this pool has any, so the text reads naturally as far as the curated
        # lists reach; the generator supplies the rest. A real word colliding with a template literal
        # or an earlier pool is skipped rather than renamed, which keeps disjointness structural.
        for word in REAL_WORDS_BY_POOL.get(name, ()):
            if len(block) >= size:
                break
            if word in taken:
                continue
            taken.add(word)
            block.append(word)
        while len(block) < size:
            word = _stride(cursor)
            cursor += 1
            if word in taken:
                continue
            taken.add(word)
            block.append(word)
        allocated[name] = tuple(block)
    return allocated


@dataclass(frozen=True)
class ValueSpec:
    """
    One attribute, and the pools whose words compose its value.

    :param name: Attribute name, used in templates and in the bit-counter's span labels.
    :param pool_names: The pools contributing one word each, in order. Length is the words per value.
    """

    name: str
    pool_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pool_names:
            raise OLMoConfigurationError(
                f"attribute '{self.name}' composes zero pools, so it carries no value"
            )

    @property
    def words_per_value(self) -> int:
        """How many words this attribute's value renders to. Fixed for a given schema."""
        return len(self.pool_names)


@dataclass(frozen=True)
class CorpusSchema:
    """
    A :class:`~factcrowd.corpus.entities.Schema` plus the attribute grouping over its pools.

    The entity table stores one index per *pool*; this says which pools group into which attribute.
    Keeping the two separate is what lets one table representation serve both axes.

    :param schema: The pools themselves, which is what sets bits per entity.
    :param values: The attribute grouping over ``schema.attributes``.
    """

    schema: Schema
    values: Tuple[ValueSpec, ...]

    def __post_init__(self) -> None:
        declared = [pool.name for pool in self.schema.attributes]
        used: List[str] = []
        for spec in self.values:
            used.extend(spec.pool_names)

        if sorted(used) != sorted(declared):
            missing = sorted(set(declared) - set(used))
            extra = sorted(set(used) - set(declared))
            raise OLMoConfigurationError(
                f"the attribute grouping must use every pool exactly once. "
                f"Unused pools: {missing or 'none'}; unknown pools: {extra or 'none'}. "
                f"An unused pool contributes bits to bits_per_entity that no rendered text carries, "
                f"which would put every cell above its true demand."
            )
        if len(used) != len(set(used)):
            duplicated = sorted({name for name in used if used.count(name) > 1})
            raise OLMoConfigurationError(
                f"pools {duplicated} appear in more than one attribute, so their bits would be "
                f"counted once and rendered twice"
            )

    @property
    def bits_per_entity(self) -> float:
        """Exact bits per entity, from the pools. Identical to ``schema.bits_per_entity``."""
        return self.schema.bits_per_entity

    @property
    def words_per_entity(self) -> int:
        """Total attribute words in one rendered biography, excluding the name and template text."""
        return sum(spec.words_per_value for spec in self.values)

    @property
    def pool_index(self) -> Dict[str, int]:
        """Pool name to its column in :attr:`~factcrowd.corpus.entities.EntityTable.attributes`."""
        return {pool.name: i for i, pool in enumerate(self.schema.attributes)}


DEFAULT_NAME_POOL_SIZES: Tuple[int, ...] = (400, 400, 1000)
"""
First / middle / last name pool sizes, and so the name universe.

Load-bearing rather than a mere ceiling: the product appears in the demand formula's name term
(:func:`factcrowd.ladder.rho.name_bits`), so widening the pools raises demand at fixed entity count.
160M clears the largest remaining cell's 2.9M entities by 55x.
"""

NAME_SPACE = DEFAULT_NAME_POOL_SIZES[0] * DEFAULT_NAME_POOL_SIZES[1] * DEFAULT_NAME_POOL_SIZES[2]
"""The default name universe, 160,000,000. What a cell's demand is computed against."""

_NAME_POOLS = ("first_name", "middle_name", "last_name")


def _build_pools(
    attribute_sizes: Sequence[Tuple[str, int]],
    name_pool_sizes: Sequence[int],
    *,
    allow_singleton: bool = False,
    reserved: Iterable[str] = (),
    active: Optional[int] = None,
) -> Tuple[Tuple[AttributePool, ...], Tuple[AttributePool, ...]]:
    """
    Build the attribute and name pools together, so all of them draw from one disjoint allocation.

    Names are allocated alongside attribute values rather than separately: a name colliding with a
    university would be the same ambiguity as two attributes colliding.
    """
    if len(name_pool_sizes) != 3:
        raise OLMoConfigurationError(f"expected three name pool sizes, got {len(name_pool_sizes)}")
    every = list(attribute_sizes) + list(zip(_NAME_POOLS, name_pool_sizes))
    words = allocate_words(every, reserved=reserved)
    attributes = tuple(
        AttributePool(name=name, values=words[name], allow_singleton=allow_singleton, active=active)
        for name, _ in attribute_sizes
    )
    names = tuple(AttributePool(name=name, values=words[name]) for name in _NAME_POOLS)
    return attributes, names


BIOS_POOL_SIZES: Tuple[Tuple[str, int], ...] = (
    ("birth_city", 200),
    ("university", 300),
    ("major", 100),
    ("employer", 263),
    ("birth_month", 12),
    ("birth_day", 28),
    ("birth_year", 400),
)
"""
The bioS pools, with the birth date decomposed into month / day / year.

Sums to 47.592 bits: 7.644 + 8.229 + 6.644 + 8.039 for the categorical fields and 3.585 + 4.807 +
8.644 for the date. Physics 3.3 publishes the 47.6 total rather than the factorisation, so this is
our reconstruction of it -- but it agrees to 0.01 of a bit, and it is the schema we use, which is
what makes the comparison legitimate.
"""


def bios_schema(
    *, name_pool_sizes: Sequence[int] = DEFAULT_NAME_POOL_SIZES, reserved: Iterable[str] = ()
) -> CorpusSchema:
    """
    The bioS schema for the count axis: seven attributes, one word each.

    :param name_pool_sizes: First / middle / last name pool sizes. The product is the name universe
        ``N0``, which is now load-bearing -- it appears in the demand formula's name term
        (:func:`factcrowd.ladder.rho.name_bits`), so widening it raises demand at fixed entity count.
        The default 400 x 400 x 1000 gives 160M, clearing the largest cell's 6.43M entities by 25x.

    :returns: The schema and its attribute grouping.
    """
    attributes, names = _build_pools(BIOS_POOL_SIZES, name_pool_sizes, reserved=reserved)
    values = tuple(ValueSpec(name=name, pool_names=(name,)) for name, _ in BIOS_POOL_SIZES)
    return CorpusSchema(schema=Schema(attributes=attributes, names=names), values=values)


def entropy_schema(
    bits_per_attribute: int,
    *,
    n_attributes: int = ENTROPY_ATTRIBUTES,
    words_per_value: int = ENTROPY_WORDS_PER_VALUE,
    name_pool_sizes: Sequence[int] = DEFAULT_NAME_POOL_SIZES,
    reserved: Iterable[str] = (),
    vocabulary_bits: int = ENTROPY_VOCABULARY_BITS,
) -> CorpusSchema:
    """
    The entropy-axis schema: ``n_attributes`` attributes of ``words_per_value`` words each.

    Every pool has size ``2^(bits_per_attribute / words_per_value)``, so bits per attribute is
    exactly ``bits_per_attribute`` and the rendered token count does not depend on it. That is the
    axis's defining property.

    ``bits_per_attribute = 0`` is a legitimate point, not a degenerate one: pools of size 1 mean
    every entity shares one value tuple, so demand is zero while tokens, steps, schedule and mixture
    ratio are unchanged. It is the anchor that makes the sweep's intercept measurable. Note that a
    pool of size 1 is below what :class:`~factcrowd.corpus.entities.AttributePool` normally accepts,
    which is deliberate -- see the ``allow_singleton`` note below.

    :param bits_per_attribute: Bits each attribute carries. Must be divisible by ``words_per_value``
        so the per-pool size is a power of two.
    :param n_attributes: How many attributes. Fixed across a sweep.
    :param words_per_value: Words per attribute value. Fixed across a sweep -- varying it would
        sweep tokens along with bits.
    :param name_pool_sizes: See :func:`bios_schema`.
    :param reserved: Words the pools must avoid. See :func:`allocate_words`.
    :param vocabulary_bits: The **sweep's maximum** ``bits_per_attribute``, which sizes every pool.
        Shared by every cell so the vocabulary, the padded size, the parameter count and the token ids
        are identical across the sweep; only the reachable prefix of each pool changes. Defaults to
        :data:`ENTROPY_VOCABULARY_BITS`.

    :returns: The schema and its attribute grouping.

    :raises OLMoConfigurationError: If ``bits_per_attribute`` is negative, not divisible by
        ``words_per_value``, above ``vocabulary_bits``, or if the union pool exceeds a sane budget.
    """
    if bits_per_attribute < 0:
        raise OLMoConfigurationError(
            f"'bits_per_attribute' must not be negative, got {bits_per_attribute}"
        )
    if words_per_value <= 0:
        raise OLMoConfigurationError(f"'words_per_value' must be positive, got {words_per_value}")
    if bits_per_attribute % words_per_value != 0:
        raise OLMoConfigurationError(
            f"'bits_per_attribute' ({bits_per_attribute}) must be divisible by 'words_per_value' "
            f"({words_per_value}) so each pool is a power of two. The nearest usable values are "
            f"{words_per_value * (bits_per_attribute // words_per_value)} and "
            f"{words_per_value * (bits_per_attribute // words_per_value + 1)}."
        )

    active_size = 2 ** (bits_per_attribute // words_per_value)
    # The pool is sized by the *union* over the sweep, not by this cell. Every entropy cell therefore
    # shares one vocabulary, one padded size, one parameter count and one set of token ids, and differs
    # only in how many of each pool's words any entity can be assigned. Sizing pools per cell made the
    # vocabulary a function of the treatment -- 1,920 padded tokens at b=0 against 8,064 at b=32, i.e.
    # 8.1% more parameters and a 4.2x larger softmax on the high-entropy arm -- which is a confound
    # running opposite ways in the two things this axis measures.
    pool_size = 2 ** (vocabulary_bits // words_per_value)
    if active_size > pool_size:
        raise OLMoConfigurationError(
            f"bits_per_attribute={bits_per_attribute} needs {active_size:,} values per pool, above "
            f"the union vocabulary's {pool_size:,} at vocabulary_bits={vocabulary_bits}. Raise "
            f"vocabulary_bits to the sweep's maximum -- every cell in a sweep must share one."
        )
    total_words = n_attributes * words_per_value * pool_size
    if total_words > 65_536:
        raise OLMoConfigurationError(
            f"bits_per_attribute={bits_per_attribute} at {words_per_value} words per value needs "
            f"pools of {pool_size:,} words, {total_words:,} word types in total, which would make "
            f"the embedding table larger than the model whose capacity is being measured. Raise "
            f"'words_per_value' to spread the same bits over more, smaller pools."
        )

    attribute_sizes: List[Tuple[str, int]] = []
    values: List[ValueSpec] = []
    for attribute in range(n_attributes):
        pool_names = tuple(f"attr{attribute}_w{word}" for word in range(words_per_value))
        attribute_sizes.extend((name, pool_size) for name in pool_names)
        values.append(ValueSpec(name=f"attr{attribute}", pool_names=pool_names))

    attributes, names = _build_pools(
        attribute_sizes,
        name_pool_sizes,
        allow_singleton=True,
        reserved=reserved,
        active=active_size,
    )
    return CorpusSchema(schema=Schema(attributes=attributes, names=names), values=tuple(values))


def bits_per_entity_for(
    bits_per_attribute: int, *, n_attributes: int = ENTROPY_ATTRIBUTES
) -> float:
    """
    Bits per entity on the entropy axis, without building a schema.

    For budget arithmetic and config validation. ``n_attributes * bits_per_attribute``, exactly.

    :param bits_per_attribute: Bits each attribute carries.
    :param n_attributes: How many attributes.

    :returns: Bits per entity.
    """
    return float(n_attributes * bits_per_attribute)


def bios_bits_per_entity() -> float:
    """
    Bits per entity for the bioS schema, from :data:`BIOS_POOL_SIZES` alone.

    Cheaper than building the schema, and used by config validation to check a cell's demand without
    generating any word lists.

    :returns: 47.592, to the precision of the pool sizes.
    """
    return sum(math.log2(size) for _, size in BIOS_POOL_SIZES)
