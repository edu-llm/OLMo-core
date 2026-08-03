"""
Turning an entity into a biography, at a few million tokens a second, with exact value spans.

Three properties matter more than the prose, and all three are structural rather than tested-for.

**Biographies vary in length, from about 21 tokens to about 152.** An earlier version forced every
template to one length, which made packing pure arithmetic -- but it also forced every biography into
the same terse shape, and measured 1.90 bits per token where the design calls for prose at 0.48 and
names compact records at 3.12. Four length bands now span the range, so the corpus is prose rather
than a record dump.

Variable length moves packing out of this module. :class:`~factcrowd.corpus.source.BioTokenSource`
answers ``get_token_range`` and OLMo-core's ``ConcatAndChunkInstanceSource`` does the chunking, which
is machinery that already exists and should not be rewritten. What that costs is a token-offset index,
and the trick is that it need not be per-document: a cumulative count every few hundred biographies
is a few tens of megabytes rather than the ten-plus gigabytes a per-document table would be.

**Value spans are exact by construction.** :meth:`Renderer.render_into` returns the token range each
attribute value occupies, because it *wrote* those tokens at those offsets. The bit-counter needs to
sum loss over exactly the value tokens, and every alternative -- string matching, re-tokenising,
aligning after the fact -- is a way for the eval prompt and the rendered prose to disagree about
where a value starts. That disagreement is a bug this programme has already shipped once, in a corpus
that keyed facts to a canonical name while the prose used a random variant.

**No ``np.random.Generator`` is constructed per biography.** Building one costs 7.9 microseconds
against 0.28 for :func:`splitmix64`, which caps a dataloader worker at about 13M tokens/s before any
rendering happens -- below what an eight-GPU node consumes. ``RandomInstanceSource`` constructs one
per *instance*, which is safe at its granularity and fatal at ours. Randomness here is a pure
function of ``(seed, entity_id, exposure)``, which is also what makes the stream reproducible from a
seed rather than from a saved shard.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .entities import EntityTable
from .values import CorpusSchema
from .vocab import Vocabulary

__all__ = [
    "NAME_PLACEHOLDER",
    "Template",
    "ValueSpan",
    "Renderer",
    "splitmix64",
    "BIOS_TEMPLATES",
    "entropy_templates",
    "literal_words_of",
]


NAME_PLACEHOLDER = "name"
"""The placeholder every template must contain: the entity's key."""

_GOLDEN64 = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_TOKEN_DTYPE = np.uint32


def splitmix64(value: np.ndarray) -> np.ndarray:
    """
    The splitmix64 finaliser, vectorised over a uint64 array. Pure, and ~28x cheaper than a Generator.

    :param value: uint64 array to mix.

    :returns: The mixed uint64 array.
    """
    with np.errstate(over="ignore"):
        mixed = value + _GOLDEN64
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _MIX_A
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _MIX_B
        return mixed ^ (mixed >> np.uint64(31))


class ValueSpan(NamedTuple):
    """
    Where one attribute's value sits in a rendered biography.

    Half-open, and relative to the start of the biography rather than of the instance, so a span
    survives being packed at a different offset.
    """

    attribute: str
    """The attribute name, matching a :class:`~factcrowd.corpus.values.ValueSpec`."""

    start: int
    """First token of the value, relative to the biography."""

    end: int
    """One past the last token of the value."""


@dataclass(frozen=True)
class Template:
    """
    One biography phrasing: a sequence of literal words and ``{placeholder}`` slots.

    :param parts: Words in order. A part written ``{like_this}`` is a slot naming either
        :data:`NAME_PLACEHOLDER` or an attribute of the schema; anything else is a literal word.
    """

    parts: Tuple[str, ...]

    @property
    def slots(self) -> Tuple[str, ...]:
        """The placeholder names, in order of appearance."""
        return tuple(part[1:-1] for part in self.parts if self._is_slot(part))

    @property
    def literals(self) -> Tuple[str, ...]:
        """The literal words, in order of appearance."""
        return tuple(part for part in self.parts if not self._is_slot(part))

    @staticmethod
    def _is_slot(part: str) -> bool:
        """Whether a part is a ``{placeholder}`` rather than a literal."""
        return len(part) > 2 and part.startswith("{") and part.endswith("}")


@dataclass(frozen=True)
class _Compiled:
    """
    A template flattened into a skeleton buffer plus vectorised scatter plans for its slots.

    The plans exist for throughput. Writing each value word with a scalar assignment ran at 1.65M
    tokens/s single-threaded, below the ~2.4M a worker needs to keep an eight-GPU node fed. Rendering
    is now the skeleton copy plus two gather-scatter pairs -- five vectorised operations per
    biography, whatever the schema's shape.
    """

    skeleton: np.ndarray
    """Token ids with literals in place and zeros where slot values go."""

    length: int
    """Tokens this template renders to, including the domain token, BOS and the trailing EOS."""

    spans: Tuple[ValueSpan, ...]
    """Where each attribute's value lands, relative to the start of the biography."""

    attribute_dest: np.ndarray
    """Positions in the biography that attribute-value words are written to."""

    attribute_source: np.ndarray
    """For each of those, the flat-token-table base of the pool it draws from."""

    attribute_column: np.ndarray
    """For each of those, which column of ``table.attributes`` holds the pool index."""

    name_dest: np.ndarray
    """Positions in the biography that name words are written to."""

    name_source: np.ndarray
    """Flat-token-table bases for the name pools."""

    name_column: np.ndarray
    """Columns of ``table.name_indices``."""


class Renderer:
    """
    Renders biographies from an entity table, one fixed-length block per biography.

    :param table: The entity table. Its schema must match ``corpus_schema``.
    :param corpus_schema: The pools and the attribute grouping over them.
    :param vocabulary: Must contain every pool value, every template literal and the domain token.
    :param templates: At least ``min_templates`` phrasings, all rendering to the same length.
    :param domain_token: Prepended to every biography. Mandatory -- see PRD.md §3.4.
    :param seed: Seeds template choice. Independent of the table's own seed so a phrasing set can be
        resampled without regenerating facts.
    :param min_templates: Refuse a smaller template set. Physics 3.3 found diverse rendering does not
        hurt capacity and may help, and our own single-template corpus answered one question at 83%
        under one phrasing and 1.3% under another -- it had stored a pattern, not a fact.

    :raises OLMoConfigurationError: If the templates disagree on length, if any template omits the
        name or an attribute, if there are too few of them, or if the vocabulary is missing a word.
    """

    def __init__(
        self,
        table: EntityTable,
        corpus_schema: CorpusSchema,
        vocabulary: Vocabulary,
        templates: Sequence[Template],
        *,
        domain_token: str,
        seed: int = 0,
        min_templates: int = 20,
    ) -> None:
        if len(templates) < min_templates:
            raise OLMoConfigurationError(
                f"{len(templates)} templates, below the {min_templates} minimum. A single phrasing "
                f"lets the model store a pattern-to-value association instead of a fact: our own "
                f"corpus answered the same question at 83% under one phrasing and 1.3% under "
                f"another. Pass min_templates=1 only for a deliberate single-template control."
            )
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")

        self._table = table
        self._schema = corpus_schema
        self._vocabulary = vocabulary
        self._seed = seed
        self._domain_token = domain_token

        if table.schema.fingerprint() != corpus_schema.schema.fingerprint():
            raise OLMoConfigurationError(
                "the entity table was built against a different schema than the corpus schema, so "
                "its indices point into other pools"
            )

        self._attribute_widths = {spec.name: spec.words_per_value for spec in corpus_schema.values}
        self._name_width = len(corpus_schema.schema.names)
        self._pool_columns = corpus_schema.pool_index
        self._value_pools: Dict[str, Tuple[str, ...]] = {
            spec.name: spec.pool_names for spec in corpus_schema.values
        }
        self._name_pool_names = tuple(pool.name for pool in corpus_schema.schema.names)

        # One concatenated token table over every pool, so a value lookup is a single gather rather
        # than a dict access plus an index. The bases are what a compiled template stores.
        self._flat_base: Dict[str, int] = {}
        flat_pieces: List[np.ndarray] = []
        cursor = 0
        for pool in tuple(corpus_schema.schema.attributes) + tuple(corpus_schema.schema.names):
            self._flat_base[pool.name] = cursor
            flat_pieces.append(vocabulary.pool_token_ids[pool.name])
            cursor += len(pool)
        self._flat_ids = np.concatenate(flat_pieces)

        self._compiled = tuple(self._compile(template) for template in templates)
        self._template_lengths = np.array(
            [compiled.length for compiled in self._compiled], dtype=np.int64
        )
        self._max_length = int(self._template_lengths.max())

    @property
    def n_table_entities(self) -> int:
        """Entities in the underlying table -- the ceiling on what a slice may draw from."""
        return self._table.n_entities

    def fingerprint(self) -> str:
        """
        A digest of everything that determines the rendered text.

        Covers the schema (and so the vocabulary of every pool), the vocabulary's id assignment, the
        template set and the template-choice seed. Two renderers with the same fingerprint produce
        byte-identical biographies, which is what lets a corpus be reproduced from a config.

        :returns: A hex digest.
        """
        digest = hashlib.sha256()
        fields = [
            "factcrowd.Renderer.v2",
            self._schema.schema.fingerprint(),
            self._vocabulary.fingerprint(),
            self._domain_token,
            str(self._seed),
            str(self._table.seed),
        ]
        for template in self._compiled:
            fields.append(",".join(str(int(token)) for token in template.skeleton))
        for field in fields:
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    @property
    def template_lengths(self) -> np.ndarray:
        """Tokens each template renders to, indexed by template."""
        return self._template_lengths

    @property
    def max_tokens_per_bio(self) -> int:
        """The longest template's length, which is what a render buffer must accommodate."""
        return self._max_length

    @property
    def mean_tokens_per_bio(self) -> float:
        """
        Mean tokens per biography, for budget arithmetic.

        A plain mean over templates, which is the expectation because template choice is uniform.
        Exact per-cell token counts come from :attr:`factcrowd.corpus.source.BioTokenSource.num_tokens`,
        which sums the real lengths rather than estimating them.
        """
        return float(self._template_lengths.mean())

    @property
    def bits_per_token(self) -> float:
        """
        Attribute bits per rendered token, the density the design has an opinion about.

        PRD.md §3.3 chose prose over compact records, putting prose near 0.48 bits/token and records
        near 3.12. Worth reading off a built renderer rather than assuming, since it is a function of
        the templates and moved by a factor of four when they were rewritten.
        """
        return self._schema.bits_per_entity / self.mean_tokens_per_bio

    @property
    def n_templates(self) -> int:
        """How many phrasings are in play."""
        return len(self._compiled)

    def _compile(self, template: Template) -> _Compiled:
        """Flatten one template into a skeleton and its slot offsets."""
        known = set(self._attribute_widths) | {NAME_PLACEHOLDER}
        missing = known - set(template.slots)
        if missing:
            raise OLMoConfigurationError(
                f"template {' '.join(template.parts)!r} omits {sorted(missing)}. Every biography must "
                f"state every attribute, or entities differ in how many facts they assert and "
                f"exposures stop being comparable."
            )
        unknown = set(template.slots) - known
        if unknown:
            raise OLMoConfigurationError(
                f"template {' '.join(template.parts)!r} names unknown slots {sorted(unknown)}; the "
                f"schema declares {sorted(known)}"
            )
        repeated = [slot for slot in set(template.slots) if template.slots.count(slot) > 1]
        if repeated:
            raise OLMoConfigurationError(
                f"template {' '.join(template.parts)!r} repeats {sorted(repeated)}. A value stated "
                f"twice is exposed twice, so the exposure count would not be 200."
            )

        pieces: List[np.ndarray] = [
            self._vocabulary.encode([self._domain_token, self._vocabulary.words[2]])
        ]
        spans: List[ValueSpan] = []
        attribute_plan: List[Tuple[int, int, int]] = []
        name_plan: List[Tuple[int, int, int]] = []
        position = len(pieces[0])
        for part in template.parts:
            if Template._is_slot(part):
                slot = part[1:-1]
                if slot == NAME_PLACEHOLDER:
                    for word, pool_name in enumerate(self._name_pool_names):
                        name_plan.append((position + word, self._flat_base[pool_name], word))
                    width = self._name_width
                else:
                    for word, pool_name in enumerate(self._value_pools[slot]):
                        attribute_plan.append(
                            (
                                position + word,
                                self._flat_base[pool_name],
                                self._pool_columns[pool_name],
                            )
                        )
                    width = self._attribute_widths[slot]
                    spans.append(ValueSpan(slot, position, position + width))
                pieces.append(np.zeros(width, dtype=_TOKEN_DTYPE))
                position += width
            else:
                pieces.append(self._vocabulary.encode([part]))
                position += 1
        pieces.append(np.array([self._vocabulary.eos_id], dtype=_TOKEN_DTYPE))

        def columns(plan: List[Tuple[int, int, int]], index: int) -> np.ndarray:
            return np.array([row[index] for row in plan], dtype=np.int64)

        skeleton = np.concatenate(pieces)
        return _Compiled(
            skeleton=skeleton,
            length=int(skeleton.size),
            spans=tuple(spans),
            attribute_dest=columns(attribute_plan, 0),
            attribute_source=columns(attribute_plan, 1),
            attribute_column=columns(attribute_plan, 2),
            name_dest=columns(name_plan, 0),
            name_source=columns(name_plan, 1),
            name_column=columns(name_plan, 2),
        )

    def template_indices(self, entity_ids: np.ndarray, exposures: np.ndarray) -> np.ndarray:
        """
        Which phrasing each ``(entity, exposure)`` pair uses, vectorised.

        Depends on both, so one entity's 200 exposures spread across the template set rather than
        repeating one phrasing -- which is the point of having a set.

        Vectorised because the scalar form built a one-element array per biography, and at ~1
        microsecond of numpy call overhead that alone capped a worker near 2M tokens/s.

        :param entity_ids: Entity ids.
        :param exposures: Exposure indices, same shape.

        :returns: Indices into the template set.
        """
        key = (
            (entity_ids.astype(np.uint64) << np.uint64(20))
            ^ exposures.astype(np.uint64)
            ^ np.uint64(self._seed)
        )
        return (splitmix64(key) % np.uint64(self.n_templates)).astype(np.int64)

    def template_index(self, entity_id: int, exposure: int) -> int:
        """
        Which phrasing one ``(entity, exposure)`` pair uses. See :meth:`template_indices`.

        :param entity_id: Which entity.
        :param exposure: Which exposure, ``0 <= exposure < exposures``.

        :returns: An index into the template set.
        """
        return int(
            self.template_indices(
                np.array([entity_id], dtype=np.uint64), np.array([exposure], dtype=np.uint64)
            )[0]
        )

    def render_run(
        self, out: np.ndarray, entity_ids: np.ndarray, exposures: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[Tuple[ValueSpan, ...], ...]]:
        """
        Render biographies back to back into ``out``, and report how long each turned out.

        The throughput path. Template choice is vectorised once for the whole run, and each biography
        is then a skeleton copy plus two gather-scatter pairs on a *view*, so nothing in the loop
        allocates.

        :param out: Destination buffer, uint32. Must hold the sum of the run's lengths; use
            :meth:`lengths_of` to size it, or ``len(entity_ids) * max_tokens_per_bio`` for a bound.
        :param entity_ids: One entity per biography.
        :param exposures: One exposure index per biography, same shape.

        :returns: The length of each biography, and its value spans with offsets relative to that
            biography's own start.

        :raises OLMoConfigurationError: If the shapes disagree or the buffer is too small.
        """
        if entity_ids.shape != exposures.shape:
            raise OLMoConfigurationError(
                f"'entity_ids' and 'exposures' must have the same shape, got "
                f"{entity_ids.shape} and {exposures.shape}"
            )
        indices = self.template_indices(entity_ids, exposures)
        lengths = self._template_lengths[indices]
        needed = int(lengths.sum())
        if out.dtype != _TOKEN_DTYPE or out.size < needed:
            raise OLMoConfigurationError(
                f"'out' must be {_TOKEN_DTYPE} with at least {needed} elements, got "
                f"{out.dtype} of {out.size}"
            )

        flat = self._flat_ids
        attributes = self._table.attributes
        names = self._table.name_indices
        spans: List[Tuple[ValueSpan, ...]] = []
        cursor = 0
        for position in range(entity_ids.size):
            compiled = self._compiled[indices[position]]
            block = out[cursor : cursor + compiled.length]
            block[:] = compiled.skeleton
            row = attributes[entity_ids[position]]
            block[compiled.attribute_dest] = flat[
                compiled.attribute_source + row[compiled.attribute_column]
            ]
            name_row = names[entity_ids[position]]
            block[compiled.name_dest] = flat[compiled.name_source + name_row[compiled.name_column]]
            spans.append(compiled.spans)
            cursor += compiled.length
        return lengths, tuple(spans)

    def lengths_of(self, entity_ids: np.ndarray, exposures: np.ndarray) -> np.ndarray:
        """
        How long each biography will be, without rendering it.

        This is what makes a token-offset index cheap to build: lengths come from the template choice
        alone, so a whole cell's length profile is one vectorised pass with no string or buffer work.

        :param entity_ids: One entity per biography.
        :param exposures: One exposure index per biography, same shape.

        :returns: The lengths, as int64.
        """
        return self._template_lengths[self.template_indices(entity_ids, exposures)]

    def render_into(
        self, out: np.ndarray, offset: int, entity_id: int, exposure: int
    ) -> Tuple[int, Tuple[ValueSpan, ...]]:
        """
        Render one biography into ``out`` at ``offset``.

        :param out: Destination token buffer, uint32.
        :param offset: Where in ``out`` this biography starts.
        :param entity_id: Which entity.
        :param exposure: Which exposure.

        :returns: How many tokens were written, and one :class:`ValueSpan` per attribute with offsets
            relative to ``offset``.

        :raises OLMoConfigurationError: If ``out`` is the wrong dtype or too small.
        """
        if out.dtype != _TOKEN_DTYPE:
            raise OLMoConfigurationError(f"'out' must be {_TOKEN_DTYPE}, got {out.dtype}")
        compiled = self._compiled[self.template_index(entity_id, exposure)]
        end = offset + compiled.length
        if offset < 0 or end > out.size:
            raise OLMoConfigurationError(
                f"a biography of {compiled.length} tokens does not fit at offset {offset} in a "
                f"buffer of {out.size}"
            )
        lengths, spans = self.render_run(
            out[offset:end],
            np.array([entity_id], dtype=np.uint64),
            np.array([exposure], dtype=np.uint64),
        )
        return int(lengths[0]), spans[0]

    def render(self, entity_id: int, exposure: int) -> Tuple[np.ndarray, Tuple[ValueSpan, ...]]:
        """
        Render one biography into a fresh buffer. Convenience for tests and inspection.

        :param entity_id: Which entity.
        :param exposure: Which exposure.

        :returns: The token ids, trimmed to the biography's own length, and its value spans.
        """
        compiled = self._compiled[self.template_index(entity_id, exposure)]
        out = np.empty(compiled.length, dtype=_TOKEN_DTYPE)
        _, spans = self.render_into(out, 0, entity_id, exposure)
        return out, spans

    def text(self, entity_id: int, exposure: int) -> str:
        """
        The rendered biography as a string, for eyeballing a sample.

        :param entity_id: Which entity.
        :param exposure: Which exposure.

        :returns: Space-joined words.
        """
        token_ids, _ = self.render(entity_id, exposure)
        return " ".join(self._vocabulary.decode(token_ids))


def _template(text: str) -> Template:
    """Parse a whitespace-separated template string."""
    return Template(parts=tuple(text.split()))


_BIOS_TEMPLATE_TEXT: Tuple[str, ...] = (
    # --- short band, ~12 literal words -------------------------------------------------------------
    "{name} was born in {birth_city} on {birth_month} {birth_day} {birth_year} , studied {major} at {university} , joined {employer} .",
    "{name} , born {birth_month} {birth_day} {birth_year} in {birth_city} , read {major} at {university} for {employer} .",
    "Born in {birth_city} on {birth_month} {birth_day} {birth_year} , {name} took {major} at {university} , then {employer} .",
    "{name} of {birth_city} , born {birth_month} {birth_day} {birth_year} , studied {major} at {university} , now with {employer} .",
    "{name} , a {major} graduate of {university} , was born {birth_month} {birth_day} {birth_year} in {birth_city} ; employer {employer} .",
    "{name} hails from {birth_city} , born {birth_month} {birth_day} {birth_year} , trained in {major} at {university} , serves {employer} .",
    "{name} , born {birth_month} {birth_day} {birth_year} , grew up in {birth_city} , studied {major} at {university} , joined {employer} .",
    "{name} left {birth_city} , birthplace since {birth_month} {birth_day} {birth_year} , to read {major} at {university} for {employer} .",
    # --- medium band, ~30 literal words ------------------------------------------------------------
    "The public record for {name} is unusually complete . It gives a birthplace of {birth_city} and a date of birth of {birth_month} {birth_day} {birth_year} . It lists a degree in {major} taken at {university} , and current employment at {employer} .",
    "Colleagues who have worked with {name} tend to mention the same few details first . The birthplace is {birth_city} . The date of birth is {birth_month} {birth_day} {birth_year} . The degree is {major} , awarded by {university} . The employer is {employer} .",
    "{name} is one of those people whose biography reads as a straight line . Born in {birth_city} on {birth_month} {birth_day} {birth_year} , they went on to study {major} at {university} , and have worked at {employer} ever since without any obvious detour .",
    "There is little mystery about {name} . The town of {birth_city} claims the birth , which took place on {birth_month} {birth_day} {birth_year} . The subject was {major} and the institution was {university} . The current post is at {employer} , and has been for some years .",
    "Anyone drawing up a short profile of {name} would begin with {birth_city} , the birthplace , and with {birth_month} {birth_day} {birth_year} , the date . They would then note {major} as the field of study , {university} as the school , and {employer} as the employer .",
    "{name} , whose file is often cited as a model of clarity , was born in {birth_city} . The date given is {birth_month} {birth_day} {birth_year} . The course of study was {major} , pursued at {university} . The employer , listed without qualification , is {employer} .",
    "Ask about {name} and you will hear the same four things . First , the birthplace : {birth_city} . Second , the date : {birth_month} {birth_day} {birth_year} . Third , the training : {major} at {university} . Fourth , the position : {employer} .",
    "The biography of {name} is short enough to state in a paragraph . Birthplace {birth_city} , date {birth_month} {birth_day} {birth_year} , field {major} , school {university} , employer {employer} . Nothing in it has ever been disputed by anyone who looked .",
    "{name} appears in the register with every field filled in . Under birthplace it says {birth_city} . Under date of birth it says {birth_month} {birth_day} {birth_year} . Under degree it says {major} , and under institution {university} . Under employer it says {employer} .",
    "A profile of {name} , assembled from the usual sources , agrees on all the particulars . The birth was in {birth_city} , on {birth_month} {birth_day} {birth_year} . The degree was in {major} , from {university} . The employment , present and continuing , is with {employer} .",
    # --- long band, ~60 literal words --------------------------------------------------------------
    "Among the many biographies collected in this volume , the entry for {name} is one of the more straightforward , and it is worth setting out in full because the details are so rarely in doubt . The birthplace is given as {birth_city} , a town that appears in the record without further comment . The date of birth is {birth_month} {birth_day} {birth_year} . The field of study is {major} , and the institution at which it was studied is {university} . The employer , as of the most recent revision of the record , is {employer} .",
    "It is worth pausing over the entry for {name} , not because anything in it is surprising , but because so few entries are this complete . We are told the birthplace without hedging : {birth_city} . We are told the date , which is {birth_month} {birth_day} {birth_year} , and which has never been amended . We are told what was studied , namely {major} , and where , namely {university} . And we are told the employer , {employer} , with no note of any earlier position .",
    "The compilers of this register were careful , and their care shows most clearly in an entry like the one for {name} . Every field is populated and none contradicts another . For birthplace they wrote {birth_city} . For date of birth they wrote {birth_month} {birth_day} {birth_year} . For course of study they wrote {major} , and for the awarding institution they wrote {university} . For employer , finally , they wrote {employer} , and there the entry ends .",
    "Readers who work through these biographies in order will find that the entry for {name} sets the pattern the rest follow . It opens with a birthplace , {birth_city} , and gives it without qualification . It follows with a date of birth , {birth_month} {birth_day} {birth_year} , stated as plainly . It then names a field of study , {major} , and the institution where that study took place , {university} . It closes , as the others do , with an employer : {employer} .",
    "What is striking about the record kept for {name} is how little of it has ever needed correction . The birthplace has always read {birth_city} . The date of birth has always read {birth_month} {birth_day} {birth_year} . The degree has always been listed as {major} and the institution as {university} , and no revision has touched either . The employer has been {employer} for as long as the file has been open , and the entry gives no sign of that changing .",
    "This entry concerns {name} , and like the others in the series it confines itself to what can be established . On the matter of birthplace it says {birth_city} , and nothing more . On the date of birth it says {birth_month} {birth_day} {birth_year} . On education it records a degree in {major} obtained at {university} . On employment it records a single position , held at {employer} , and it declines to speculate about anything further .",
    "The file on {name} has been consulted often enough that its contents are widely known , and it is short . A birthplace is given , and that birthplace is {birth_city} . A date of birth is given , and that date is {birth_month} {birth_day} {birth_year} . A field of study is given , {major} , together with the institution , {university} . An employer is given , {employer} . Nothing else in the file is presented as established fact .",
    "Of the several accounts that survive concerning {name} , the one reproduced here is the one the archive treats as authoritative , and it is plain in its statements . It places the birth in {birth_city} . It dates that birth to {birth_month} {birth_day} {birth_year} . It names {major} as the field studied and {university} as the place of study . It names {employer} as the employer , and it makes no claim beyond these .",
    # --- very long band, ~90 literal words ---------------------------------------------------------
    "The following entry is reproduced without abridgement , since abridgement is exactly what has caused trouble with entries of this kind in the past , and the details concerning {name} are worth having in full . On the question of birthplace the record is unambiguous and gives {birth_city} , with no alternative reading offered anywhere in the margins . On the question of the date of birth it gives {birth_month} {birth_day} {birth_year} , a date that has survived every subsequent revision of the register without amendment . On education , the record states that the field of study was {major} , and that the institution at which that study was undertaken was {university} ; both are given without qualification . On employment , the record names {employer} , and it names no other , which the compilers took to mean that no other was ever held .",
    "Anyone who has spent time with this register will know that entries vary a great deal in how much they are willing to assert , and that the entry for {name} sits at the confident end of that range . It asserts a birthplace , and the birthplace it asserts is {birth_city} . It asserts a date of birth , and that date is {birth_month} {birth_day} {birth_year} , written out in the same form the register uses throughout . It asserts a field of study , which is {major} , and an institution at which that field was studied , which is {university} . And it asserts an employer , {employer} , in the present tense , which the compilers reserved for positions they had confirmed rather than merely inferred .",
    "It has become customary , in works of this sort , to preface each entry with a note about its reliability , and the note attached to the entry for {name} is unusually short : the compilers found nothing to query . The birthplace stands as {birth_city} . The date of birth stands as {birth_month} {birth_day} {birth_year} . The field of study is recorded as {major} and the institution as {university} , and the two are recorded together , as the register requires , so that neither can be read apart from the other . The employer is recorded as {employer} . Readers who wish to check any of this will find the underlying sources listed elsewhere , and will find that they agree .",
    "The entry that follows is longer than most , and the length is a function of the register's conventions rather than of anything remarkable about {name} . Where a birthplace is known with confidence the register states it plainly , and here it states {birth_city} . Where a date of birth is known it is given in full , and here it is given as {birth_month} {birth_day} {birth_year} . Where a course of study can be established the register names both the field and the institution , and here it names {major} and {university} respectively . Where an employer is on record it is named without further comment , and here it is {employer} . The remainder of the entry , omitted , concerns matters the compilers were unwilling to state as fact .",
    "One learns , reading through these files , to distrust any entry that says too much , and to trust the ones that say little and say it clearly ; the entry for {name} belongs to the second kind . It offers a birthplace : {birth_city} . It offers a date of birth : {birth_month} {birth_day} {birth_year} . It offers a field of study , {major} , and the institution where that study was done , {university} , and it offers them in that order because the register's form requires it . It offers an employer , {employer} . It offers nothing else at all , and the compilers were explicit that this was a decision rather than a gap in their sources .",
    "The register's editors were in the habit of appending a remark to entries they considered settled , and the entry for {name} carries such a remark , which is why it is reproduced here in preference to the shorter summary . The birthplace given is {birth_city} , and it is given without any of the qualifications the editors used elsewhere . The date of birth given is {birth_month} {birth_day} {birth_year} . The field of study is {major} , and it was studied at {university} ; the editors noted that both had been confirmed against a second source . The employer is {employer} , and the editors let that stand as the final line of the entry , which was their way of indicating that they regarded the matter as closed .",
)

BIOS_TEMPLATES: Tuple[Template, ...] = tuple(_template(text) for text in _BIOS_TEMPLATE_TEXT)
"""
Twenty-two bioS phrasings, all with the same literal word count so they render to one length.

Equal length is checked by :class:`Renderer`; the count here is deliberately a couple above the
twenty-template floor so one can be dropped without tripping it.
"""


def entropy_templates(n_attributes: int, words_per_value: int) -> Tuple[Template, ...]:
    """
    Templates for the entropy axis, generated because its attributes are positional.

    The count axis has meaningful field names to write prose around; the entropy axis has
    ``attr0..attr5``, so its phrasings are built from a rotating set of connectives instead. They read
    less naturally than :data:`BIOS_TEMPLATES` -- but the review's naturalness requirement was about
    the *values* (four ordinary-looking words rather than one random string), and that is satisfied by
    :func:`~factcrowd.corpus.values.entropy_schema`.

    All generated templates share a literal count by construction, since they differ only by which
    connective goes where.

    :param n_attributes: How many attributes the schema has.
    :param words_per_value: Unused in the text, but part of the signature so a caller cannot forget
        that value width is fixed across the sweep.

    :returns: One template per rotation, at least twenty.

    :raises OLMoConfigurationError: If ``n_attributes`` is not positive.
    """
    if n_attributes <= 0:
        raise OLMoConfigurationError(f"'n_attributes' must be positive, got {n_attributes}")
    del words_per_value  # width is a property of the schema, not of the phrasing

    connectives = (
        "records",
        "lists",
        "shows",
        "gives",
        "notes",
        "holds",
        "carries",
        "states",
        "reports",
        "marks",
        "keeps",
        "names",
    )
    templates: List[Template] = []
    for rotation in range(max(20, len(connectives))):
        parts: List[str] = ["{" + NAME_PLACEHOLDER + "}"]
        for attribute in range(n_attributes):
            parts.append(connectives[(rotation + attribute) % len(connectives)])
            parts.append("{attr" + str(attribute) + "}")
            parts.append(";" if attribute < n_attributes - 1 else ".")
        templates.append(Template(parts=tuple(parts)))
    return tuple(templates)


def literal_words_of(templates: Sequence[Template]) -> Tuple[str, ...]:
    """
    Every literal word a template set uses, deduplicated, for :meth:`Vocabulary.build`.

    :param templates: The templates.

    :returns: The distinct literal words, sorted for determinism.
    """
    return tuple(sorted({word for template in templates for word in template.literals}))
