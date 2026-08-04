"""
The reasoning slices: generated per example, fixed width, with the answer span known by construction.

Three properties, each answering a specific way this programme has failed before.

**Fixed width per task.** Unlike biographies, a reasoning item renders to a constant token count, so
item ``i`` occupies tokens ``[i·W, (i+1)·W)`` and the stream needs no offset index at all. That is not
a coincidence of the tasks chosen -- it is enforced, because it makes the arithmetic exact and the
mixture's absolute token counts exact with it.

**No memorizable content.** Every item is regenerated from its index, and the item space is far larger
than any slice drawn from it, so the same expression essentially never recurs. That matters because a
reasoning slice carrying reusable facts would compete for the capacity the experiment is measuring --
which is the whole reason FLD's per-example regeneration was called a feature rather than a
limitation.

**The answer span is returned by the code that writes it.** Four reasoning endpoints in this programme
produced uninterpretable nulls, and the two that were parser failures -- iGSM graded on a single
integer with the derivation discarded, a deduction eval scoring *below* its own floor because
truncation parsed as wrong -- would both have been caught by knowing exactly which tokens carry the
answer. Here that is arithmetic, not a regex.

**What is here and what is not.** ``mano`` is the primary endpoint: mod-23 mental arithmetic with no
chain of thought, at expression length 10 rather than 13, because at 13 it sits about a point above its
own degenerate policy at our model sizes while at 10 it moves 18 points across the ladder at fixed
depth 12. ``compare`` is the related-reasoning slice, over the same entities the fact slice describes.
Brevo1, Reasoning Core and the gate battery of PRD.md §8.6 are not built yet; the framework here is
what they plug into.
"""

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .entities import EntityTable
from .render import splitmix64
from .values import CorpusSchema
from .vocab import Vocabulary

__all__ = [
    "MANO_MODULUS",
    "TaskItem",
    "ReasoningTask",
    "ManoTask",
    "CompareTask",
    "TaskStream",
    "mano_words",
    "compare_words",
]


MANO_MODULUS = 23
"""
The modulus for Mano. Physics 4.1 uses 23, and matching it is what makes their numbers comparable.

Residues get their own tokens rather than being spelled as digits, so the task carries no tokenizer
risk: an endpoint whose difficulty depends on how a BPE splits "17" is measuring the tokenizer.
"""

_TOKEN_DTYPE = np.uint32


def mano_words() -> Tuple[str, ...]:
    """
    The vocabulary Mano needs: one token per residue, the two operators, and an equals sign.

    :returns: The words, in a fixed order.
    """
    return tuple(f"<n{value}>" for value in range(MANO_MODULUS)) + ("<plus>", "<times>", "<equals>")


def compare_words() -> Tuple[str, ...]:
    """
    The vocabulary the comparison task needs beyond the entity pools.

    :returns: The words, in a fixed order.
    """
    return ("Between", "and", "the", "earlier", "birth", "year", "?", "Answer", ":")


class TaskItem:
    """
    One rendered reasoning item.

    :param tokens: The full item, exactly ``tokens_per_item`` long.
    :param answer_start: First token of the answer, relative to the item.
    :param answer_end: One past the last answer token.
    :param answer: The answer as words, for scoring.
    """

    __slots__ = ("tokens", "answer_start", "answer_end", "answer")

    def __init__(
        self,
        tokens: np.ndarray,
        answer_start: int,
        answer_end: int,
        answer: Tuple[str, ...],
    ) -> None:
        self.tokens = tokens
        self.answer_start = answer_start
        self.answer_end = answer_end
        self.answer = answer

    @property
    def prompt_tokens(self) -> np.ndarray:
        """Everything before the answer -- what a closed-book eval conditions on."""
        return self.tokens[: self.answer_start]

    @property
    def answer_tokens(self) -> np.ndarray:
        """The answer's tokens, which is what a scorer compares and what the CE metric sums over."""
        return self.tokens[self.answer_start : self.answer_end]


class ReasoningTask(ABC):
    """
    A generated reasoning slice: fixed width, regenerated per index, answer span known.

    Subclasses declare their vocabulary needs through :meth:`required_words` so
    :class:`~factcrowd.corpus.vocab.Vocabulary` can reserve them, and implement :meth:`item`.
    """

    #: Slice name, used as a config key and a metric prefix.
    name: str = ""

    #: Set by every subclass. Declared here because :meth:`degenerate_baseline` decodes items to search
    #: copy policies, which needs the words rather than the ids.
    _vocabulary: "Vocabulary"

    @property
    @abstractmethod
    def tokens_per_item(self) -> int:
        """
        Tokens every item renders to. Constant, which is what makes the stream O(1).

        :returns: The width.
        """

    @property
    @abstractmethod
    def domain_token(self) -> str:
        """The token prepended to every item. Mandatory on every segment of the mixture."""

    @staticmethod
    @abstractmethod
    def required_words() -> Tuple[str, ...]:
        """Words this task needs in the vocabulary beyond the specials and the entity pools."""

    @abstractmethod
    def item(self, index: int) -> TaskItem:
        """
        Render one item.

        :param index: Which item. Items are a pure function of this and the task's seed.

        :returns: The item.
        """

    @abstractmethod
    def fingerprint(self) -> str:
        """A digest of everything that determines the items this task generates."""

    def render_run(self, out: np.ndarray, indices: np.ndarray) -> Tuple[TaskItem, ...]:
        """
        Render items back to back into ``out``.

        The default walks :meth:`item`; a task with a vectorisable body may override it.

        :param out: Destination buffer, uint32, at least ``len(indices) * tokens_per_item`` long.
        :param indices: Which items.

        :returns: The items, in order.

        :raises OLMoConfigurationError: If the buffer is the wrong dtype or too small.
        """
        width = self.tokens_per_item
        needed = int(indices.size) * width
        if out.dtype != _TOKEN_DTYPE or out.size < needed:
            raise OLMoConfigurationError(
                f"'out' must be {_TOKEN_DTYPE} with at least {needed} elements, got "
                f"{out.dtype} of {out.size}"
            )
        items: List[TaskItem] = []
        for position, index in enumerate(indices):
            rendered = self.item(int(index))
            out[position * width : (position + 1) * width] = rendered.tokens
            items.append(rendered)
        return tuple(items)

    def degenerate_baseline(self, sample: int = 20_000) -> Tuple[str, float]:
        """
        The strongest fact-free policy and how often it is right, **measured rather than assumed**.

        Every endpoint declares its degenerate baseline, because an endpoint whose score matches its
        baseline carries no signal -- and because assuming one is how a previous eval in this programme
        came to report a figure below its own floor.

        Two policy families are searched, not one:

        - **constant**: always emit the same answer. The obvious family, and for Mano the binding one.
        - **copy**: always emit the span sitting at a fixed offset in the prompt. Requires no facts, no
          ordering and no arithmetic -- an induction head suffices.

        Searching only constants is a mistake that costs an endpoint its floor. ``<compare>``'s answer is
        a copy of one of the two names in its own prompt, so "always name the first person" is right
        **50%** of the time while the best constant name is right 0.02% -- a factor of 1,400. An endpoint
        whose floor is quoted as 0% when it is really 50% has half the dynamic range its admission gate
        assumes, and any score under 50% is below its own floor.

        :param sample: How many items to draw.

        :returns: A label for the winning policy, and its accuracy.
        """
        width = self.tokens_per_item
        constant: Dict[Tuple[str, ...], int] = {}
        # One counter per prompt offset a copy policy could read an answer-width span from.
        copies: Dict[int, int] = {}
        for index in range(sample):
            item = self.item(index)
            answer = item.answer
            constant[answer] = constant.get(answer, 0) + 1
            span = item.answer_end - item.answer_start
            words = self._vocabulary.decode(item.tokens)
            for offset in range(0, width - span + 1):
                if offset == item.answer_start:
                    continue  # reading the answer itself is not a policy
                if tuple(words[offset : offset + span]) == answer:
                    copies[offset] = copies.get(offset, 0) + 1

        best_constant = max(constant, key=lambda key: constant[key])
        label, count = f"constant:{' '.join(best_constant)}", constant[best_constant]
        if copies:
            # A copy policy wins only if it beats the best constant by more than sampling noise. The
            # maximum over ~20 offsets is upward-biased -- each is right about 1/23 of the time on Mano,
            # so the best of them exceeds the constant by two standard errors for nothing, and the floor
            # would drift up with the number of offsets searched rather than with the task. Three
            # standard errors of the constant rate is the bar; the real thing this exists to catch beat
            # it by a factor of 1,400.
            best_offset = max(copies, key=lambda key: copies[key])
            rate = count / sample
            margin = 3.0 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / sample)
            if copies[best_offset] / sample > rate + margin:
                label, count = f"copy@{best_offset}", copies[best_offset]
        return label, count / sample

    def degenerate_answer(self, sample: int = 20_000) -> Tuple[Tuple[str, ...], float]:
        """
        The best *constant* answer and how often it is right.

        Kept because the constant family is worth reporting on its own, but it is **not** the endpoint's
        floor -- see :meth:`degenerate_baseline`, which is.

        :param sample: How many items to draw.

        :returns: The most frequent answer, and its frequency.
        """
        counts: Dict[Tuple[str, ...], int] = {}
        for index in range(sample):
            answer = self.item(index).answer
            counts[answer] = counts.get(answer, 0) + 1
        best = max(counts, key=lambda key: counts[key])
        return best, counts[best] / sample


class ManoTask(ReasoningTask):
    """
    Mod-23 mental arithmetic with no chain of thought: Physics 4.1's Mano.

    An expression of ``length`` residues joined by ``+`` and ``*``, then ``=``, then the answer -- one
    token, evaluated left to right modulo 23. No intermediate steps appear, so the model has to hold
    them, which is what makes the task parameter-sensitive rather than a copying exercise.

    :param vocabulary: Must contain :meth:`required_words` and the domain token.
    :param domain_token: Prepended to every item.
    :param length: Residues in the expression. **10, not 13.** At 13 the task sits about a point above
        its own degenerate policy at 13M-28M, failing the 20-80% admission band; at 10 Physics 4.1
        reports 47.8 to 66.0 from scratch at our exact 12 layers, moving 18.2 points across the
        parameter range.
    :param seed: Seeds expression generation.

    :raises OLMoConfigurationError: If ``length`` is below two or the vocabulary is missing a word.
    """

    name = "mano"

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        domain_token: str,
        length: int = 10,
        seed: int = 0,
    ) -> None:
        if length < 2:
            raise OLMoConfigurationError(f"'length' must be at least 2, got {length}")
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")
        self._vocabulary = vocabulary
        self._domain_token = domain_token
        self._length = length
        self._seed = seed

        self._residue_ids = np.array(
            [vocabulary.id_of(f"<n{value}>") for value in range(MANO_MODULUS)], dtype=_TOKEN_DTYPE
        )
        self._operator_ids = np.array(
            [vocabulary.id_of("<plus>"), vocabulary.id_of("<times>")], dtype=_TOKEN_DTYPE
        )
        self._prefix = vocabulary.encode([domain_token, vocabulary.words[2]])
        self._equals = np.array([vocabulary.id_of("<equals>")], dtype=_TOKEN_DTYPE)
        self._eos = np.array([vocabulary.eos_id], dtype=_TOKEN_DTYPE)
        # domain + bos + length residues + (length - 1) operators + equals + answer + eos
        self._width = 2 + length + (length - 1) + 1 + 1 + 1

    @property
    def tokens_per_item(self) -> int:
        return self._width

    @property
    def domain_token(self) -> str:
        return self._domain_token

    @property
    def length(self) -> int:
        """Residues in the expression."""
        return self._length

    @staticmethod
    def required_words() -> Tuple[str, ...]:
        return mano_words()

    def item(self, index: int) -> TaskItem:
        # One mix per item gives 64 bits to spend; the expression needs `length` residues and
        # `length - 1` operator bits, which fits comfortably for any length worth training on.
        draw = int(
            splitmix64(np.array([np.uint64(index) ^ np.uint64(self._seed)], dtype=np.uint64))[0]
        )
        residues: List[int] = []
        operators: List[int] = []
        state = draw
        for position in range(self._length):
            if position:
                operators.append(state & 1)
                state >>= 1
            residues.append(state % MANO_MODULUS)
            state //= MANO_MODULUS
            if state == 0:  # exhausted the draw; re-mix rather than repeat a residue pattern
                # XOR rather than add: `draw + position + 1` overflows np.uint64 when draw is within
                # `length` of 2**64, which raises rather than wrapping.
                state = int(
                    splitmix64(
                        np.array([np.uint64(draw) ^ np.uint64(position + 1)], dtype=np.uint64)
                    )[0]
                )

        # Zero is excluded as a multiplicand, because multiplication by it is an absorbing state: with
        # a free choice of operand the answer is 0 far more often than any other residue, and the best
        # constant policy scores 8.3% instead of the 6.8% Physics 4.1 reports. An endpoint whose floor
        # is higher than the paper's is a weaker instrument.
        #
        # The replacement is *redrawn*, not computed from the zero it replaces. The first version wrote
        # `1 + (residues[position + 1] + position) % 22`, and inside a branch that only fires when that
        # term is zero, which makes it the constant `1 + position` -- so operator position 0 always got
        # <n1>, position 4 always <n5>, doubling the mass on one position-specific operand. The floor
        # survived it; the uniformity the comment claimed did not.
        if operators:
            spare = int(
                splitmix64(np.array([np.uint64(draw) ^ np.uint64(0x9E3779B9)], dtype=np.uint64))[0]
            )
            for position, operator in enumerate(operators):
                if operator == 1 and residues[position + 1] == 0:
                    residues[position + 1] = 1 + spare % (MANO_MODULUS - 1)
                    spare //= MANO_MODULUS - 1
                    if spare == 0:
                        spare = int(
                            splitmix64(np.array([np.uint64(draw + position + 7)], dtype=np.uint64))[
                                0
                            ]
                        )

        total = residues[0]
        for operator, residue in zip(operators, residues[1:]):
            total = (
                (total + residue) % MANO_MODULUS
                if operator == 0
                else (total * residue) % MANO_MODULUS
            )

        tokens = np.empty(self._width, dtype=_TOKEN_DTYPE)
        tokens[:2] = self._prefix
        cursor = 2
        for position, residue in enumerate(residues):
            if position:
                tokens[cursor] = self._operator_ids[operators[position - 1]]
                cursor += 1
            tokens[cursor] = self._residue_ids[residue]
            cursor += 1
        tokens[cursor] = self._equals[0]
        cursor += 1
        answer_start = cursor
        tokens[cursor] = self._residue_ids[total]
        cursor += 1
        tokens[cursor] = self._eos[0]

        return TaskItem(tokens, answer_start, answer_start + 1, (f"<n{total}>",))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for field in (
            "factcrowd.ManoTask.v1",
            self._vocabulary.fingerprint(),
            self._domain_token,
            str(self._length),
            str(self._seed),
            str(MANO_MODULUS),
        ):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


class CompareTask(ReasoningTask):
    """
    Related reasoning: which of two people was born earlier.

    Two-fact by construction -- the model must retrieve both birth years and order them -- and
    verifiable without a parser, because the answer is one of the two names it was given. Restricted to
    the probe subset, so the population is the same 25k people in every cell and P4 compares like with
    like across the ladder.

    Chosen over composition because composition over these entities is ambiguous: "the person whose
    university is U" has many answers unless uniqueness is enforced, and enforcing it changes the
    attribute distribution and so the demand. Comparison needs no such surgery. The composition and
    aggregation forms PRD.md §3.3 also names are follow-ons, not replacements.

    :param table: The entity table, which must be the one the fact slice uses.
    :param corpus_schema: Its schema, for the birth-year column.
    :param vocabulary: Must contain :meth:`required_words`, the name pools and the domain token.
    :param domain_token: Prepended to every item.
    :param probe_ids: The entities this slice may ask about.
    :param order_attribute: Which attribute orders the comparison. Must be a single-pool attribute
        whose pool index is meaningfully ordinal.
    :param seed: Seeds pair selection.

    :raises OLMoConfigurationError: If the attribute is absent or not single-pool, or the probe subset
        is too small to draw a pair from.
    """

    name = "compare"

    def __init__(
        self,
        table: EntityTable,
        corpus_schema: CorpusSchema,
        vocabulary: Vocabulary,
        *,
        domain_token: str,
        probe_ids: np.ndarray,
        order_attribute: str = "birth_year",
        seed: int = 0,
    ) -> None:
        if probe_ids.size < 2:
            raise OLMoConfigurationError(
                f"the comparison task needs at least two entities to compare, got {probe_ids.size}"
            )
        specs = {spec.name: spec for spec in corpus_schema.values}
        if order_attribute not in specs:
            raise OLMoConfigurationError(
                f"'{order_attribute}' is not an attribute of this schema; it declares "
                f"{sorted(specs)}"
            )
        if len(specs[order_attribute].pool_names) != 1:
            raise OLMoConfigurationError(
                f"'{order_attribute}' composes {len(specs[order_attribute].pool_names)} pools, so its "
                f"index is not a single ordinal value to compare"
            )

        self._table = table
        self._vocabulary = vocabulary
        self._domain_token = domain_token
        self._probe_ids = probe_ids.astype(np.int64)
        self._seed = seed
        self._order_pool_name = specs[order_attribute].pool_names[0]
        self._order_column = corpus_schema.pool_index[self._order_pool_name]
        self._order_attribute = order_attribute
        self._order_pool = {pool.name: pool for pool in corpus_schema.schema.attributes}[
            self._order_pool_name
        ]
        self._name_pools = tuple(pool.name for pool in corpus_schema.schema.names)
        self._name_width = len(self._name_pools)

        # The answer is the earlier person's *year*, not their name, so the question asks for a year.
        # A name answer is a span of the prompt, and "copy the first name" then scores 50% -- half the
        # endpoint's range for free, on a task that needs no facts at all. Asking for the value puts the
        # answer outside the prompt, where no copy policy can reach it: the measured floor falls from
        # 50.2% to 0.7%. The composition under test is unchanged -- recall both years, compare, emit the
        # smaller -- and it is now the only policy that works.
        prompt = ["Between"]
        prompt_after_first = ["and"]
        tail = ["the", "earlier", "birth", "year", "?", "Answer", ":"]
        self._prefix = vocabulary.encode([domain_token, vocabulary.words[2], *prompt])
        self._joiner = vocabulary.encode(prompt_after_first)
        self._tail = vocabulary.encode(tail)
        self._eos = np.array([vocabulary.eos_id], dtype=_TOKEN_DTYPE)
        self._width = (
            self._prefix.size
            + self._name_width
            + self._joiner.size
            + self._name_width
            + self._tail.size
            + 1  # the answer: the earlier person's ordinal value, one word
            + 1  # eos
        )

    @property
    def tokens_per_item(self) -> int:
        return self._width

    @property
    def domain_token(self) -> str:
        return self._domain_token

    @staticmethod
    def required_words() -> Tuple[str, ...]:
        return compare_words()

    def _name_ids(self, entity_id: int) -> np.ndarray:
        """Token ids of one entity's name, from the pool lookup tables."""
        row = self._table.name_indices[entity_id]
        pool_ids = self._vocabulary.pool_token_ids
        return np.array(
            [pool_ids[pool][row[position]] for position, pool in enumerate(self._name_pools)],
            dtype=_TOKEN_DTYPE,
        )

    def _order_ids(self, entity_id: int) -> np.ndarray:
        """Token ids of one entity's ordinal attribute value -- the answer."""
        index = int(self._table.attributes[entity_id][self._order_column])
        return np.array(
            [self._vocabulary.pool_token_ids[self._order_pool_name][index]], dtype=_TOKEN_DTYPE
        )

    def _order_value(self, entity_id: int) -> Tuple[str, ...]:
        """The words of one entity's ordinal attribute value."""
        index = int(self._table.attributes[entity_id][self._order_column])
        return (self._order_pool.values[index],)

    def item(self, index: int) -> TaskItem:
        draw = splitmix64(
            np.array([np.uint64(index) ^ (np.uint64(self._seed) << np.uint64(32))], dtype=np.uint64)
        )[0]
        count = self._probe_ids.size
        first = self._probe_ids[int(draw % np.uint64(count))]
        # A second, distinct entity: offset by a nonzero amount modulo the count, which cannot collide.
        offset = 1 + int((draw >> np.uint64(20)) % np.uint64(count - 1))
        second = self._probe_ids[(int(draw % np.uint64(count)) + offset) % count]

        first_order = int(self._table.attributes[first][self._order_column])
        second_order = int(self._table.attributes[second][self._order_column])
        if first_order == second_order:
            # A tie has no correct answer, so break it by entity id -- deterministic, and rare enough
            # that it cannot bias the task, but it must not be left to chance.
            earlier = first if first < second else second
        else:
            earlier = first if first_order < second_order else second

        tokens = np.empty(self._width, dtype=_TOKEN_DTYPE)
        cursor = 0
        for piece in (
            self._prefix,
            self._name_ids(first),
            self._joiner,
            self._name_ids(second),
            self._tail,
        ):
            tokens[cursor : cursor + piece.size] = piece
            cursor += piece.size
        answer_start = cursor
        answer_ids = self._order_ids(earlier)
        tokens[cursor : cursor + answer_ids.size] = answer_ids
        cursor += answer_ids.size
        tokens[cursor] = self._eos[0]

        return TaskItem(
            tokens,
            answer_start,
            answer_start + answer_ids.size,
            self._order_value(earlier),
        )

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for field in (
            "factcrowd.CompareTask.v1",
            self._vocabulary.fingerprint(),
            self._table.schema.fingerprint(),
            self._domain_token,
            self._order_attribute,
            # The probe ids themselves and the table's own seed, not just the subset's size. Without
            # them two tasks over different entities -- or the same entities with different attribute
            # values -- shared a fingerprint, and that digest keys the cached instance index and is what
            # a "checkpoint versus corpus" audit compares.
            hashlib.sha256(self._probe_ids.tobytes()).hexdigest(),
            str(self._table.seed),
            str(self._seed),
        ):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


class TaskStream:
    """
    A reasoning slice as a contiguous token stream.

    Items are fixed width, so ``locate`` is a division and no offset index is needed -- which is why
    this is a fraction of the size of :class:`~factcrowd.corpus.stream.BioStream`.

    :param task: The task to draw from.
    :param num_tokens: How many tokens the slice should hold. Rounded **down** to a whole number of
        items, because a truncated item has a truncated answer and a truncated answer is exactly the
        failure that made a previous eval score below its own floor.
    :param label: Optional label for source visualisations.

    :raises OLMoConfigurationError: If ``num_tokens`` is smaller than one item.
    """

    def __init__(
        self, task: ReasoningTask, *, num_tokens: int, label: Optional[str] = None
    ) -> None:
        width = task.tokens_per_item
        if num_tokens < width:
            raise OLMoConfigurationError(
                f"a slice of {num_tokens:,} tokens cannot hold one {task.name} item of {width} "
                f"tokens"
            )
        self._task = task
        self._n_items = num_tokens // width
        self._num_tokens = self._n_items * width
        self._label = label

    @property
    def task(self) -> ReasoningTask:
        """The underlying task."""
        return self._task

    @property
    def num_tokens(self) -> int:
        """
        A whole number of items, so the *stream* never ends mid-item.

        That is a weaker guarantee than it looks, and weaker than this line used to claim. Rounding down
        protects the end of the stream only; the trainer asks
        :class:`~olmo_core.data.composable.ConcatAndChunkInstanceSource` for 512-token windows, and
        neither 24 nor 19 divides 512, so **3.1% of mano items and 3.5% of compare items are cut by an
        instance boundary** -- some of them mid-answer, leaving an instance that opens with an answer and
        no question. The streams are byte-identical across cells so the cuts are identical too, which
        makes this a uniform tax rather than a confound, but two things follow: ``answer_start`` is valid
        only *before* chunking, so an eval must locate answers itself rather than trust it; and a
        sequence length of 504 (or per-item padding with a label mask) would remove the tax entirely.
        """
        return self._num_tokens

    @property
    def n_items(self) -> int:
        """Items in the slice."""
        return self._n_items

    def tokens(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        The tokens at ``[start_idx, end_idx)``.

        :param start_idx: First token, inclusive.
        :param end_idx: One past the last token.

        :returns: Exactly ``end_idx - start_idx`` tokens.

        :raises OLMoConfigurationError: If the range is empty, inverted or out of bounds.
        """
        if end_idx <= start_idx:
            raise OLMoConfigurationError(
                f"token range [{start_idx}, {end_idx}) is empty or inverted"
            )
        if start_idx < 0 or end_idx > self._num_tokens:
            raise OLMoConfigurationError(
                f"token range [{start_idx}, {end_idx}) is out of bounds for a slice of "
                f"{self._num_tokens:,} tokens"
            )
        width = self._task.tokens_per_item
        first = start_idx // width
        last = (end_idx - 1) // width
        buffer = np.empty((last - first + 1) * width, dtype=_TOKEN_DTYPE)
        self._task.render_run(buffer, np.arange(first, last + 1))
        offset = start_idx - first * width
        return buffer[offset : offset + (end_idx - start_idx)]

    def fingerprint(self) -> str:
        """A digest of the task and the slice size."""
        digest = hashlib.sha256()
        for field in ("factcrowd.TaskStream.v1", self._task.fingerprint(), str(self._num_tokens)):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


def all_required_words(task_types: Sequence[Type["ReasoningTask"]]) -> Tuple[str, ...]:
    """
    Every word a set of task *classes* needs, deduplicated and ordered.

    Classes rather than instances, and the ordering is forced: a task cannot be constructed until the
    vocabulary contains its tokens, and the vocabulary cannot be built until it knows which tokens to
    reserve. So the words come off the class, before either exists.

    :param task_types: The task classes, e.g. ``(ManoTask, CompareTask)``.

    :returns: The distinct words.
    """
    seen: List[str] = []
    for task_type in task_types:
        for word in task_type.required_words():
            if word not in seen:
                seen.append(word)
    return tuple(seen)
