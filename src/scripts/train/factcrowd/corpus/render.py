"""
Turning an entity into a biography, at a few million tokens a second, with exact value spans.

Three properties matter more than the prose, and all three are structural rather than tested-for.

**Every biography is the same number of tokens.** Templates are checked at build time to render to an
identical length, so biography ``i`` always occupies tokens ``[i·L, (i+1)·L)`` of the stream. That is
what makes packing arithmetic: an :class:`~olmo_core.data.composable.InstanceSource` can answer
"which biographies are in instance ``idx``" in constant time, with no prefix-sum index over 1.29
billion documents and no padding. The alternative -- variable-length biographies -- costs either 80%
padding at ``sequence_length=512`` or a 10-20 GB offset table, which would contradict the whole
never-materialise design.

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

import math
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
    "bios_per_instance",
    "instance_count",
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
        lengths = {len(compiled.skeleton) for compiled in self._compiled}
        if len(lengths) != 1:
            by_length: Dict[int, List[int]] = {}
            for index, compiled in enumerate(self._compiled):
                by_length.setdefault(len(compiled.skeleton), []).append(index)
            raise OLMoConfigurationError(
                f"templates must all render to the same token count, got {sorted(by_length)} "
                f"(template indices by length: "
                f"{ {length: idx[:4] for length, idx in sorted(by_length.items())} }). Equal length "
                f"is what makes biography i occupy tokens [i*L, (i+1)*L) and so lets packing be "
                f"arithmetic; unequal lengths cost either heavy padding or a prefix-sum index over "
                f"every document. Add or remove literal words until they agree."
            )
        self._tokens_per_bio = lengths.pop()

    @property
    def tokens_per_bio(self) -> int:
        """Tokens in every rendered biography, including the domain token and the trailing EOS."""
        return self._tokens_per_bio

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

        return _Compiled(
            skeleton=np.concatenate(pieces),
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

    def render_instance(
        self, out: np.ndarray, entity_ids: np.ndarray, exposures: np.ndarray
    ) -> Tuple[Tuple[ValueSpan, ...], ...]:
        """
        Render a whole instance: one biography per element of ``entity_ids``, back to back.

        The throughput path. Template choice is vectorised once for the instance, and each biography
        is then a skeleton copy plus two gather-scatter pairs on a *view* -- so nothing in the loop
        allocates.

        :param out: Destination buffer, uint32, at least ``len(entity_ids) * tokens_per_bio`` long.
        :param entity_ids: One entity per biography.
        :param exposures: One exposure index per biography, same shape.

        :returns: The value spans of each biography, offsets relative to that biography's start.

        :raises OLMoConfigurationError: If the shapes disagree or the buffer is too small.
        """
        if entity_ids.shape != exposures.shape:
            raise OLMoConfigurationError(
                f"'entity_ids' and 'exposures' must have the same shape, got "
                f"{entity_ids.shape} and {exposures.shape}"
            )
        needed = entity_ids.size * self._tokens_per_bio
        if out.dtype != _TOKEN_DTYPE or out.size < needed:
            raise OLMoConfigurationError(
                f"'out' must be {_TOKEN_DTYPE} with at least {needed} elements, got "
                f"{out.dtype} of {out.size}"
            )

        indices = self.template_indices(entity_ids, exposures)
        flat = self._flat_ids
        attributes = self._table.attributes
        names = self._table.name_indices
        width = self._tokens_per_bio
        spans: List[Tuple[ValueSpan, ...]] = []
        for position in range(entity_ids.size):
            compiled = self._compiled[indices[position]]
            start = position * width
            block = out[start : start + width]
            block[:] = compiled.skeleton
            row = attributes[entity_ids[position]]
            block[compiled.attribute_dest] = flat[
                compiled.attribute_source + row[compiled.attribute_column]
            ]
            name_row = names[entity_ids[position]]
            block[compiled.name_dest] = flat[compiled.name_source + name_row[compiled.name_column]]
            spans.append(compiled.spans)
        return tuple(spans)

    def render_into(
        self, out: np.ndarray, offset: int, entity_id: int, exposure: int
    ) -> Tuple[ValueSpan, ...]:
        """
        Render one biography into ``out`` at ``offset``, and say where its values landed.

        Writes exactly :attr:`tokens_per_bio` tokens. The hot path is one buffer copy plus one slice
        assignment per slot, with no allocation, which is what keeps a worker above the throughput a
        training node needs.

        :param out: Destination token buffer, uint32.
        :param offset: Where in ``out`` this biography starts.
        :param entity_id: Which entity.
        :param exposure: Which exposure.

        :returns: One :class:`ValueSpan` per attribute, with offsets relative to ``offset``.

        :raises OLMoConfigurationError: If ``out`` is too small or the wrong dtype.
        """
        if out.dtype != _TOKEN_DTYPE:
            raise OLMoConfigurationError(f"'out' must be {_TOKEN_DTYPE}, got {out.dtype}")
        end = offset + self._tokens_per_bio
        if offset < 0 or end > out.size:
            raise OLMoConfigurationError(
                f"a biography of {self._tokens_per_bio} tokens does not fit at offset {offset} in a "
                f"buffer of {out.size}"
            )

        compiled = self._compiled[self.template_index(entity_id, exposure)]
        out[offset:end] = compiled.skeleton

        flat = self._flat_ids
        attribute_row = self._table.attributes[entity_id]
        out[offset + compiled.attribute_dest] = flat[
            compiled.attribute_source + attribute_row[compiled.attribute_column]
        ]
        name_row = self._table.name_indices[entity_id]
        out[offset + compiled.name_dest] = flat[
            compiled.name_source + name_row[compiled.name_column]
        ]
        return compiled.spans

    def render(self, entity_id: int, exposure: int) -> Tuple[np.ndarray, Tuple[ValueSpan, ...]]:
        """
        Render one biography into a fresh buffer. Convenience for tests and inspection.

        :param entity_id: Which entity.
        :param exposure: Which exposure.

        :returns: The token ids and the value spans.
        """
        out = np.empty(self._tokens_per_bio, dtype=_TOKEN_DTYPE)
        spans = self.render_into(out, 0, entity_id, exposure)
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
    "{name} was born in {birth_city} on {birth_month} {birth_day} {birth_year} , and later studied {major} at {university} before joining {employer} .",
    "{name} , born on {birth_month} {birth_day} {birth_year} in {birth_city} , read {major} at {university} and now works for {employer} .",
    "Born in {birth_city} on {birth_month} {birth_day} {birth_year} , {name} went on to study {major} at {university} and joined {employer} .",
    "{name} grew up in {birth_city} , born {birth_month} {birth_day} {birth_year} , took {major} at {university} and works at {employer} .",
    "A native of {birth_city} , {name} was born {birth_month} {birth_day} {birth_year} , studied {major} at {university} and joined {employer} .",
    "{name} was born on {birth_month} {birth_day} {birth_year} in {birth_city} ; they studied {major} at {university} and work for {employer} .",
    "From {birth_city} , where they were born on {birth_month} {birth_day} {birth_year} , {name} studied {major} at {university} for {employer} .",
    "{name} , graduate in {major} from {university} , was born in {birth_city} on {birth_month} {birth_day} {birth_year} , joined {employer} .",
    "{name} completed {major} at {university} after being born in {birth_city} on {birth_month} {birth_day} {birth_year} , and works for {employer} .",
    "{name} of {birth_city} was born on {birth_month} {birth_day} {birth_year} , earned {major} at {university} , and is with {employer} .",
    "{name} was raised in {birth_city} , born {birth_month} {birth_day} {birth_year} , studied {major} at {university} , now at {employer} .",
    "{name} , born {birth_month} {birth_day} {birth_year} , comes from {birth_city} , studied {major} at {university} , and joined {employer} .",
    "{name} took a degree in {major} at {university} ; born in {birth_city} on {birth_month} {birth_day} {birth_year} , joined {employer} .",
    "{name} , born in {birth_city} , dated {birth_month} {birth_day} {birth_year} , majored in {major} at {university} and joined {employer} .",
    "{name} hails from {birth_city} , was born {birth_month} {birth_day} {birth_year} , studied {major} at {university} , and joined {employer} .",
    "{name} , born on {birth_month} {birth_day} {birth_year} , is from {birth_city} , studied {major} at {university} , joined {employer} .",
    "{name} left {birth_city} , where they were born {birth_month} {birth_day} {birth_year} , to read {major} at {university} for {employer} .",
    "{name} was born in {birth_city} , on {birth_month} {birth_day} {birth_year} , and read {major} at {university} before joining {employer} .",
    "{name} , originally from {birth_city} , born {birth_month} {birth_day} {birth_year} , studied {major} at {university} , and joined {employer} .",
    "{name} , born {birth_month} {birth_day} {birth_year} , spent childhood in {birth_city} , studied {major} at {university} , joined {employer} .",
    "{name} studied {major} at {university} , having been born in {birth_city} on {birth_month} {birth_day} {birth_year} , and joined {employer} .",
    "{name} , born {birth_month} {birth_day} {birth_year} in {birth_city} , holds a degree in {major} from {university} and serves {employer} .",
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


def bios_per_instance(tokens_per_bio: int, sequence_length: int) -> int:
    """
    How many whole biographies fit an instance, which is what makes packing arithmetic.

    :param tokens_per_bio: From :attr:`Renderer.tokens_per_bio`.
    :param sequence_length: The instance length.

    :returns: The number of whole biographies per instance.

    :raises OLMoConfigurationError: If not even one fits, or if the waste exceeds a fifth of the
        instance -- at which point a different sequence length is cheaper than the padding.
    """
    if tokens_per_bio <= 0 or sequence_length <= 0:
        raise OLMoConfigurationError(
            f"'tokens_per_bio' and 'sequence_length' must be positive, got "
            f"{tokens_per_bio} and {sequence_length}"
        )
    count = sequence_length // tokens_per_bio
    if count < 1:
        raise OLMoConfigurationError(
            f"a biography is {tokens_per_bio} tokens and the sequence length is {sequence_length}, "
            f"so not one fits. Shorten the templates or raise the sequence length."
        )
    waste = sequence_length - count * tokens_per_bio
    if waste > sequence_length // 5:
        raise OLMoConfigurationError(
            f"{count} biographies of {tokens_per_bio} tokens leave {waste} of {sequence_length} "
            f"unused ({100 * waste / sequence_length:.0f}%), which is paid for in FLOPs at every "
            f"step. Choose a sequence length near a multiple of {tokens_per_bio} -- "
            f"{count * tokens_per_bio} or {(count + 1) * tokens_per_bio} would waste nothing."
        )
    return count


def instance_count(n_entities: int, exposures: int, bios_per_instance_count: int) -> int:
    """
    How many instances the fact slice yields.

    :param n_entities: Entities in the table.
    :param exposures: Exposures per entity.
    :param bios_per_instance_count: From :func:`bios_per_instance`.

    :returns: The instance count, rounding down: a partial final instance is dropped rather than
        padded, which costs at most one instance and keeps every instance identical in shape.
    """
    return math.floor(n_entities * exposures / bios_per_instance_count)
