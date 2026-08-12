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
from typing import Dict, Final, List, Optional, Sequence, Tuple, Type

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .entities import EntityTable
from .render import splitmix64
from .values import CorpusSchema
from .vocab import PAD, Vocabulary

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

_MAX_INSTANCE_TOKENS = 512
"""
The instance length every cell trains at, repeated here as a bound rather than imported.

`cells.CellSpec.sequence_length` is the authority and defaults to this. Duplicated because an item wider
than an instance is a corpus-layer error that should be refused when the task is built, and importing the
cell layer from the corpus layer to learn one integer would invert the dependency.
"""


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


SPLITS: Tuple[str, ...] = ("train", "eval")
"""
The generation splits. A task belongs to exactly one and cannot produce the other's items.

The split is part of the key, not a convention, because a convention was not enough. The first version
of this file mixed ``index ^ seed``, which makes the item set a *function of neither alone*: two seeds
differing by 15 produce the same set of items, permuted, so ``item(i)`` under seed 1241 equals
``item(i ^ 15)`` under seed 1238. Verified at 2,000/2,000. An evaluation set drawn that way is 100%
leaked from training however the seed is chosen, and nothing about it looks wrong.
"""

_SPLIT_TAGS: Dict[str, int] = {
    "train": 0x5F3759DF00000001,
    "eval": 0xA5A5A5A500000002,
}


def item_key(*, class_tag: int, split: str, seed: int, index: int) -> int:
    """
    The 64-bit draw for one item, keyed by ``(class, split, seed, index)``.

    Two rounds, and the structure is the point. The seed and split are mixed into a *key* first, then
    combined with a separately mixed index. That makes the dependence on the seed non-translational: two
    seeds no longer index the same sequence at an offset, so an eval item lands inside a training range
    of N items only with probability about ``N / 2**64``. A single-round ``index ^ seed`` gives set
    equality instead, which is the defect this replaces.

    :param class_tag: Distinguishes task types, so two tasks cannot share a stream.
    :param split: A member of :data:`SPLITS`.
    :param seed: The task seed.
    :param index: The item index.

    :returns: The mixed draw.

    :raises OLMoConfigurationError: If the split is unknown or the index is negative.
    """
    if split not in _SPLIT_TAGS:
        raise OLMoConfigurationError(f"unknown split {split!r}; expected one of {SPLITS}")
    if index < 0:
        raise OLMoConfigurationError(f"item index must not be negative, got {index}")
    mask = (1 << 64) - 1
    key = int(
        splitmix64(
            np.array(
                [np.uint64((int(seed) ^ _SPLIT_TAGS[split] ^ int(class_tag)) & mask)],
                dtype=np.uint64,
            )
        )[0]
    )
    mixed_index = int(splitmix64(np.array([np.uint64((index + 1) & mask)], dtype=np.uint64))[0])
    return int(splitmix64(np.array([np.uint64((key ^ mixed_index) & mask)], dtype=np.uint64))[0])


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


def resolve_padding(pad_to: Optional[int], natural: int, *, what: str) -> int:
    """
    Validate an instance-alignment width and return the padded item width.

    **Why any of this exists.** The trainer concatenates a reasoning slice and chunks it into
    :data:`_MAX_INSTANCE_TOKENS`-token instances, so an item whose width does not *divide* that is cut by a
    boundary. Over one period there are ``w/gcd(w, 512)`` boundaries and only those landing on an item start
    are harmless, which works out at 3.12% of 24-token ``<mano>`` items and 52% of 266-token in-context ones.

    A cut item is worse than a missing one: it opens mid-expression, or loses the answer, or keeps an answer
    whose question is in the previous instance.

    **The second reason, and the one that made this necessary rather than tidy.** A depth sweep holds the
    token budget fixed while item width grows with depth, so the arms get *different item counts* -- 125M at
    length 2 against 41.7M at length 10, a factor of three. "Accuracy falls with depth" then conflates the
    task getting harder with the arm getting a third of the examples. Padding every depth to one width makes
    items, tokens and steps all equal across the sweep and leaves depth as the only thing that moves. It
    costs no compute at all: the budget is in tokens either way, so padding trades item count for
    comparability rather than buying it with FLOPs.

    :param pad_to: The requested width, or ``None`` for no padding.
    :param natural: The item's unpadded width.
    :param what: Names the task, for the message.

    :returns: The width an item occupies, which is ``natural`` when ``pad_to`` is ``None``.

    :raises OLMoConfigurationError: If the width is below ``natural`` or does not divide the instance.
    """
    if pad_to is None:
        return natural
    if pad_to < natural:
        raise OLMoConfigurationError(
            f"'pad_to' is {pad_to} but a {what} item is already {natural} tokens"
        )
    if _MAX_INSTANCE_TOKENS % pad_to:
        options = [n for n in (8, 16, 32, 64, 128, 256, 512) if n >= natural]
        raise OLMoConfigurationError(
            f"'pad_to' must divide the {_MAX_INSTANCE_TOKENS}-token instance, and {pad_to} does not. "
            f"Padding to a width that does not divide it moves the cut rate without removing it, which is "
            f"the worst of both -- try one of {options}."
        )
    return pad_to


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

    #: Set by every subclass. Part of the item key, so a task cannot produce another split's items.
    _split: str

    def structure_fingerprint(self) -> str:
        """
        A digest of everything that determines an item's *shape*, excluding the split and the seed.

        :meth:`fingerprint` bakes in both, which is right for identifying a stream but useless for
        checking a rebuild: measurement generates the ``eval`` split, so it can never reproduce the
        ``train`` digest a run recorded. That left the two things most worth checking unchecked -- a
        changed expression length or domain token passes the schema and vocabulary digests untouched while
        changing every item scored.

        :returns: The digest.
        """
        digest = hashlib.sha256()
        for field in (
            "factcrowd.ReasoningTask.structure.v1",
            type(self).__name__,
            self.name,
            self.domain_token,
            str(self.tokens_per_item),
            "\u0000".join(self.required_words()),
        ):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    @property
    def split(self) -> str:
        """Which generation split this task produces, ``"train"`` or ``"eval"``."""
        return self._split

    @property
    def vocabulary(self) -> "Vocabulary":
        """
        The vocabulary this task's ids index into.

        Public so a scorer can decode a prediction without being handed a second vocabulary that might
        not be the one the task was built against.
        """
        return self._vocabulary

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
        # SELECT ON ONE HALF, SCORE ON THE OTHER. Taking the best offset's rate on the sample that chose
        # it reports a maximum, and a maximum over W offsets is biased upward by roughly
        # `2.5 * sqrt(p(1-p)/n)`. At W ~ 20 that is a rounding error, which is why a flat three-standard-
        # error bar sufficed here for as long as every task was narrow. `InContextManoTask` searches ~240
        # offsets and inflated its floor by 0.6pp of pure selection noise -- and a bias that grows with
        # `tokens_per_item` makes two endpoints' floors incomparable, which is exactly what the
        # in-context and memorised variants need from each other. Splitting the sample removes it
        # outright, and needs no multiplicity correction to argue about.
        pick = sample // 2
        constant: Dict[Tuple[str, ...], int] = {}
        held_constant: Dict[Tuple[str, ...], int] = {}
        # One counter per prompt offset a copy policy could read an answer-width span from.
        copies: Dict[int, int] = {}
        held_copies: Dict[int, int] = {}
        for index in range(sample):
            selecting = index < pick
            item = self.item(index)
            answer = item.answer
            target = constant if selecting else held_constant
            target[answer] = target.get(answer, 0) + 1
            span = item.answer_end - item.answer_start
            words = self._vocabulary.decode(item.tokens)
            hits = copies if selecting else held_copies
            for offset in range(0, width - span + 1):
                if offset == item.answer_start:
                    continue  # reading the answer itself is not a policy
                if tuple(words[offset : offset + span]) == answer:
                    hits[offset] = hits.get(offset, 0) + 1

        held = sample - pick
        best_constant = max(constant, key=lambda key: constant[key])
        rate = held_constant.get(best_constant, 0) / held
        label = f"constant:{' '.join(best_constant)}"
        if copies:
            # A copy policy wins only if it beats the best constant by more than sampling noise on the
            # held-out half. Three standard errors, now measuring only noise rather than noise plus a
            # selection bias that scaled with the task's width. The real thing this exists to catch --
            # `<compare>`'s copy-the-first-name policy -- beat the constant by a factor of 1,400.
            best_offset = max(copies, key=lambda key: copies[key])
            copy_rate = held_copies.get(best_offset, 0) / held
            margin = 3.0 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / held)
            if copy_rate > rate + margin:
                label, rate = f"copy@{best_offset}", copy_rate
        return label, rate

    def degenerate_answer(self, sample: int = 20_000) -> Tuple[Tuple[str, ...], float]:
        """
        The best *constant* answer and how often it is right.

        Kept because the constant family is worth reporting on its own, but it is **not** the endpoint's
        floor -- see :meth:`degenerate_baseline`, which is.

        **Selected and scored on disjoint halves, like** :meth:`degenerate_baseline`. Both estimators have
        to agree on their footing or the invariant between them fails: an in-sample constant rate can
        exceed a held-out best-of-both-families rate, which reads as "the constant family beat the search
        that contains it" and is an artefact of mixing the two. A test asserts the ordering.

        :param sample: How many items to draw. Half select the constant and half score it, so the
            reported rate carries the standard error of ``sample / 2``.

        :returns: The most frequent answer, and its held-out frequency.
        """
        pick = sample // 2
        counts: Dict[Tuple[str, ...], int] = {}
        held: Dict[Tuple[str, ...], int] = {}
        for index in range(sample):
            answer = self.item(index).answer
            target = counts if index < pick else held
            target[answer] = target.get(answer, 0) + 1
        best = max(counts, key=lambda key: counts[key])
        return best, held.get(best, 0) / (sample - pick)


_MANO_SPLIT_ATTEMPTS: Final = 64
"""
How many redraws a content-disjoint split may take before giving up.

Half the draws land in the wanted half, so two attempts is the mean and 64 is 2**-64 territory. Bounded
rather than unbounded because a future generator with a degenerate content hash would otherwise hang a
data loader rather than fail.
"""


def _content_half(residues: Sequence[int], operators: Sequence[int]) -> int:
    """
    Which split an expression belongs to, from its content alone.

    Keyed on the expression rather than on an index, because two different indices routinely encode the
    same expression -- which is exactly how index-disjoint splits leak content.

    :param residues: The operands.
    :param operators: The operator bits.

    :returns: ``0`` for the train half, ``1`` for the eval half.
    """
    mixed = np.uint64(0x9E3779B97F4A7C15)
    for value in list(residues) + list(operators):
        mixed = np.uint64(
            splitmix64(np.array([mixed ^ np.uint64(int(value) + 1)], dtype=np.uint64))[0]
        )
    return int(mixed & np.uint64(1))


class ManoTask(ReasoningTask):
    """
    Mod-23 mental arithmetic with no chain of thought, in the spirit of Physics 4.1's Mano.

    .. warning::
        **This is not the published Mano, and that paper's accuracies do not transfer.** Upstream builds
        recursive prefix trees over ``+``, ``-`` and ``x``, trains at variable lengths and evaluates at
        exact length. This builds a flat infix expression of ten *operands* over ``+`` and ``x`` only,
        evaluated left to right, at one fixed length, with zero excluded as a multiplicand. The
        47.8-66.0 from-scratch range in PRD 8.3 describes a harder generator; this task needs its own
        calibration, and the floor below is measured rather than cited for that reason. The class name is
        a naming debt to be paid before anything is written up.

    An expression of ``length`` residues joined by ``+`` and ``*``, then ``=``, then the answer -- one
    token, evaluated left to right modulo 23. No intermediate steps appear, so the model has to hold
    them, which is what makes the task parameter-sensitive rather than a copying exercise.

    :param vocabulary: Must contain :meth:`required_words` and the domain token.
    :param domain_token: Prepended to every item.
    :param length: Residues in the expression. **10, not 13.** At 13 the task sits about a point above
        its own degenerate policy at 13M-28M, failing the 20-80% admission band; at 10 Physics 4.1
        reports 47.8 to 66.0 from scratch at our exact 12 layers, moving 18.2 points across the
        parameter range.
    :param pad_to: Pad each item to this width, which must divide the 512-token instance. **Wanted on any
        config that sweeps ``length``**, and not for the reason it looks like: a sweep holds the token
        budget fixed while an item's width grows with depth, so length 2 gets 125M items and length 10 gets
        41.7M -- a factor of three, in the one experiment whose job is to compare depths. Padding every
        depth to 32 gives all of them 31.25M items, equal steps, equal tokens and a zero cut rate, and costs
        no compute because the budget was in tokens all along. See :func:`resolve_padding`.
    :param seed: Seeds expression generation.

    :raises OLMoConfigurationError: If ``length`` is below two, the vocabulary is missing a word, or
        ``pad_to`` is below the natural width or does not divide the instance.
    """

    name = "mano"

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        domain_token: str,
        length: int = 10,
        pad_to: Optional[int] = None,
        seed: int = 0,
        split: str = "train",
    ) -> None:
        if length < 2:
            raise OLMoConfigurationError(f"'length' must be at least 2, got {length}")
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")
        self._vocabulary = vocabulary
        self._domain_token = domain_token
        self._length = length
        if split not in SPLITS:
            raise OLMoConfigurationError(f"unknown split {split!r}; expected one of {SPLITS}")
        self._seed = seed
        self._split = split

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
        self._natural_width = 2 + length + (length - 1) + 1 + 1 + 1
        self._pad_to = pad_to
        self._width = resolve_padding(pad_to, self._natural_width, what="mano")
        self._pad_id = _TOKEN_DTYPE(vocabulary.id_of(PAD))

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
        # CONTENT-DISJOINT SPLITS, NOT INDEX-DISJOINT ONES, AND THE DIFFERENCE DECIDES WHAT THE ENDPOINT
        # MEASURES. `item_key` gives train and eval different index streams, which guarantees different
        # *items* and guarantees nothing about different *expressions*. The space is
        # `23**length * 2**(length-1)`: 1,058 at length 2, and a 1.0B-token budget buys 125M items, so
        # every expression appears about 118,000 times and 100% of the eval set is trained on verbatim.
        # Measured overlap from a 60,000-item sample: 100% at L2, 72% at L3, and at the full stream L4 is
        # exhausted too (37 items per expression). A depth sweep over those lengths measures lookup.
        #
        # So the *content* is hashed and assigned to a half, and a draw landing in the wrong half is
        # redrawn. Two attempts on average, bounded below. An eval expression is then one the training
        # stream never contained at any length, and the model has to compute it.
        draw = item_key(
            class_tag=0x4D414E4F, split=self._split, seed=self._seed, index=index  # 'MANO'
        )
        wanted = 0 if self._split == "train" else 1
        for attempt in range(_MANO_SPLIT_ATTEMPTS):
            residues, operators = self._expression(draw)
            if _content_half(residues, operators) == wanted:
                break
            draw = int(
                splitmix64(
                    np.array([np.uint64(draw) ^ np.uint64(0xC2B2AE3D + attempt)], dtype=np.uint64)
                )[0]
            )
        else:  # pragma: no cover - 2**-64 territory
            raise OLMoConfigurationError(
                f"could not draw a {self._split!r}-half expression for index {index} in "
                f"{_MANO_SPLIT_ATTEMPTS} attempts"
            )
        return self._assemble(residues, operators)

    def _expression(self, draw: int) -> Tuple[List[int], List[int]]:
        """
        The residues and operators one draw encodes.

        Split out of :meth:`item` so the content can be generated, hashed and rejected before any tokens
        are laid down -- which is what makes a content-disjoint split possible at all.

        :param draw: A 64-bit mix.

        :returns: The residues and the operator bits.
        """
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

        return residues, operators

    def _assemble(self, residues: List[int], operators: List[int]) -> TaskItem:
        """
        Lay one expression out as tokens and compute its answer.

        :param residues: The operands.
        :param operators: The operator bits, ``0`` for ``+`` and ``1`` for ``x``.

        :returns: The item.
        """
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
        # After the eos, so the answer's offset is unchanged. `PAD` and not `EOS`: the document-boundary
        # machinery sizes its per-instance array by counting eos occurrences.
        if cursor + 1 < self._width:
            tokens[cursor + 1 :] = self._pad_id

        return TaskItem(tokens, answer_start, answer_start + 1, (f"<n{total}>",))

    @property
    def natural_width(self) -> int:
        """Tokens an item needs before any instance-alignment padding."""
        return self._natural_width

    @property
    def padding(self) -> int:
        """Padding tokens per item; zero when unpadded or already aligned."""
        return self._width - self._natural_width

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for field in (
            "factcrowd.ManoTask.v1",
            self._vocabulary.fingerprint(),
            self._domain_token,
            str(self._length),
            str(self._pad_to),
            str(self._seed),
            self._split,
            str(MANO_MODULUS),
        ):
            raw = field.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()


IN_CONTEXT_ALPHABET = 10
"""
Symbols the in-context variant composes over.

Bounded by the prompt, not by taste. Stating one binary operator row-major with a row label costs
``k * (2 + k)`` tokens, so two operators plus the expression and framing is ``2 * k**2 + 4 * k + 2 * L + 6``:
**266 tokens at k=10 and length 10**, against a 512-token instance.

**The binding constraint is alignment, not fitting.** An item is cut by an instance boundary unless its
width *divides* 512, so what a k buys is the set of depths one aligned ``pad_to`` covers:

===  ======  ==============  ==========================
k    floor   width           aligned depths
===  ======  ==============  ==========================
6    17.82%  ``102 + 2L``    2-13 at ``pad_to=128``
8    13.18%  ``166 + 2L``    2-45 at ``pad_to=256``
10   10.45%  ``246 + 2L``    2-5 at ``pad_to=256``
12    8.65%  ``342 + 2L``    2-85 at ``pad_to=512``, 32% padding
16    6.43%  ``578 + 2L``    none -- over the instance, refused
===  ======  ==============  ==========================

Ten because it gives the lowest floor that still pads cheaply: 2.3% of tokens at worst, against 18.8% at
k=9 and 32.4% at k=12. It buys only four depths, which is tight for a depth response, and the recorded
fallback is k=6 -- twelve depths for a floor 7.4 points higher. The floor is **not** ``1 / k``, which an
earlier revision of this docstring claimed; see :class:`InContextManoTask`.
"""


class InContextManoTask(ReasoningTask):
    """
    Composition over operator tables **given in the prompt**, freshly randomised per item.

    .. note::
        **This exists to remove a confound that :class:`ManoTask` can only measure.** Mod-23 Mano needs
        its operator tables in the weights: two 23x23 tables and a unary map is 1,058 entries at
        ``log2(23)`` bits, so 4,786 bits -- 0.0042% of the 114.3 Mbit demanded at ``b=32``, which sounds
        negligible until you remember the design runs about 4x oversubscribed, where the marginal thing is
        exactly what gets evicted. A decline under fact load is then equally well explained by "the facts
        evicted the arithmetic tables", which is knowledge-versus-knowledge and not this project's claim.

        Here the tables are **in the prompt and new every item**, so there is nothing to memorise: a model
        that had stored every table it ever saw still could not answer, because this item's table is one it
        has never seen. A decline therefore cannot be table eviction.

        The honest cost is that the construct changes. This is **read-then-compute**, not
        compute-from-memory, so it says nothing about stored procedural knowledge -- and it is closer to
        the chained-inference-over-context claim that motivated the project in the first place.
        :class:`ManoTask` is kept as the secondary endpoint for the other reading.

    Layout, at ``alphabet`` = k: the domain token, ``bos``, then each operator's table as a header token
    followed by k rows of ``label <equals> cell_0 ... cell_{k-1}``, then the expression as k-ary residues
    joined by the two operator tokens, then ``<equals>``, the answer, ``eos``. Every item is the same
    width, which is what keeps the stream O(1).

    **The answer is exactly uniform, so the constant floor is exactly ``1 / k``.** Composition is left to
    right, ``total <- table[op][total][operand]``, and each operand is drawn uniformly and independently; a
    row of a uniformly random table is k iid uniform values, so a uniform draw from it is uniform whatever
    ``total`` was. Measured entropy is 3.3216 bits of the 3.3219 a uniform ten-way answer has.

    **The floor is not ``1 / k``, though, and an earlier revision of this docstring said it was.** A
    fixed-offset copy is right whenever the cell it reads happens to equal the answer, and once in
    ``2 * k**2`` items that offset *is* the answer's own cell -- so the best copy policy scores
    ``1/(2k**2) + (1 - 1/(2k**2))/k``, which is **10.45%** at k=10 rather than 10.0%. Small, real, and
    exactly the kind of thing :meth:`degenerate_baseline` exists to find by measurement instead of
    argument. It found something else at the same time: searching ~240 offsets inflated the reported floor
    another 0.6pp by selection alone, which is why that method now selects and scores on disjoint halves.

    **Splits need no rejection sampling here**, unlike :class:`ManoTask`. Content disjointness is what the
    fresh table buys: train and eval draw different index streams, so they draw different tables, and two
    items sharing an expression do not share an answer unless they also share ``2 * k**2`` table cells.
    The ``split`` still enters the item key, so a task cannot produce the other split's items.

    :param vocabulary: Must contain :meth:`required_words` and the domain token.
    :param domain_token: Prepended to every item.
    :param length: Operands in the expression.
    :param alphabet: Symbols to compose over. See :data:`IN_CONTEXT_ALPHABET`.
    :param pad_to: Pad each item's tail to this width, which must divide the 512-token instance. **Wanted
        on every production config**, because the trainer chunks the concatenated slice into 512-token
        windows and an item whose width does not divide 512 is cut by a boundary: 3.1% of 24-token
        ``<mano>`` items, but **52%** of 266-token in-context ones, most of them losing the answer or part
        of the table. At k=10 the natural widths are 250, 252, 254 and 256 at lengths 2 to 5, so
        ``pad_to=256`` aligns that whole ladder for at most 2.3% of tokens. Lengths above 5 need
        ``pad_to=512`` and cost 48%, which is why the in-context ladder is short.
    :param seed: Seeds table and expression generation.
    :param split: Which generation split.

    :raises OLMoConfigurationError: If ``length`` is below two, the alphabet is outside
        ``2 .. MANO_MODULUS``, an item would not fit a 512-token instance, or a word is missing.
    """

    name = "ctxmano"

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        domain_token: str,
        length: int = 10,
        alphabet: int = IN_CONTEXT_ALPHABET,
        pad_to: Optional[int] = None,
        seed: int = 0,
        split: str = "train",
    ) -> None:
        if length < 2:
            raise OLMoConfigurationError(f"'length' must be at least 2, got {length}")
        if not 2 <= alphabet <= MANO_MODULUS:
            raise OLMoConfigurationError(
                f"'alphabet' must be between 2 and {MANO_MODULUS}, got {alphabet}; the symbols are the "
                f"residue tokens, so there are no more of them than the modulus"
            )
        if seed < 0:
            raise OLMoConfigurationError(f"'seed' must not be negative, got {seed}")
        if split not in SPLITS:
            raise OLMoConfigurationError(f"unknown split {split!r}; expected one of {SPLITS}")

        self._vocabulary = vocabulary
        self._domain_token = domain_token
        self._length = length
        self._alphabet = alphabet
        self._seed = seed
        self._split = split

        self._residue_ids = np.array(
            [vocabulary.id_of(f"<n{value}>") for value in range(alphabet)], dtype=_TOKEN_DTYPE
        )
        self._operator_ids = np.array(
            [vocabulary.id_of("<plus>"), vocabulary.id_of("<times>")], dtype=_TOKEN_DTYPE
        )
        self._prefix = vocabulary.encode([domain_token, vocabulary.words[2]])
        self._equals = np.array([vocabulary.id_of("<equals>")], dtype=_TOKEN_DTYPE)
        self._eos = np.array([vocabulary.eos_id], dtype=_TOKEN_DTYPE)

        # Per operator: the operator token, then k rows of [row label, <equals>, k cells].
        self._block = 1 + alphabet * (2 + alphabet)
        self._table_width = 2 * self._block
        # domain + bos + tables + length operands + (length - 1) operators + equals + answer + eos
        self._width = 2 + self._table_width + length + (length - 1) + 3
        if self._width > _MAX_INSTANCE_TOKENS:
            raise OLMoConfigurationError(
                f"an item is {self._width} tokens at alphabet={alphabet} and length={length}, over the "
                f"{_MAX_INSTANCE_TOKENS}-token instance; the table alone is {self._table_width}. Reduce "
                f"'alphabet' -- it costs 2 * k * (2 + k) tokens against 2 per unit of 'length'"
            )
        # INSTANCE ALIGNMENT, AND FOR THIS TASK IT IS NOT OPTIONAL. The trainer concatenates a slice and
        # chunks it into 512-token windows, so an item is cut by a boundary unless its width divides 512:
        # over one period there are `lcm(w, 512)/512` boundaries and only those landing on an item start
        # are harmless. At w=24 that is 3.1% of `<mano>` items, which is the figure recorded on
        # `TaskStream.num_tokens`. At w=266 it is **52%** -- half the items truncated, most of them losing
        # the answer or part of the table they were supposed to read. Padding the tail to a divisor of 512
        # costs a few percent of tokens and takes the cut rate to zero.
        self._natural_width = self._width
        self._pad_to = pad_to
        self._width = resolve_padding(pad_to, self._natural_width, what="in-context mano")
        self._pad_id = _TOKEN_DTYPE(vocabulary.id_of(PAD))
        # Cells to draw per item. Named because the fingerprint and the draw both need it.
        self._cells = 2 * alphabet * alphabet

    @property
    def tokens_per_item(self) -> int:
        return self._width

    @property
    def domain_token(self) -> str:
        return self._domain_token

    @property
    def length(self) -> int:
        """Operands in the expression."""
        return self._length

    @property
    def alphabet(self) -> int:
        """Symbols composed over."""
        return self._alphabet

    @property
    def table_tokens(self) -> int:
        """Tokens the two tables occupy, which is what bounds ``length``."""
        return self._table_width

    @staticmethod
    def required_words() -> Tuple[str, ...]:
        # The full mod-23 word list, not the k the alphabet uses. Deliberate: it keeps the vocabulary --
        # and so the softmax width and the parameter count -- identical to `ManoTask`'s, which is what
        # lets a confirmatory in-context row and a secondary memorised row be compared at all.
        return mano_words()

    def _draw(self, index: int) -> np.ndarray:
        """
        The uniform stream one item is built from.

        :param index: Which item.

        :returns: ``cells + 2 * length`` values, enough for the tables, the operands and the operators.
        """
        base = item_key(
            class_tag=0x43544D4E, split=self._split, seed=self._seed, index=index  # 'CTMN'
        )
        wanted = self._cells + 2 * self._length
        # One splitmix64 per output rather than a strided walk of a single 64-bit draw: the tables need
        # ~200 independent values and a 64-bit state carries about 19 of them at k=10.
        keys = (np.uint64(base) + np.arange(wanted, dtype=np.uint64)) * np.uint64(
            0x9E3779B97F4A7C15
        )
        return splitmix64(keys)

    def item(self, index: int) -> TaskItem:
        raw = self._draw(index)
        alphabet = np.uint64(self._alphabet)
        cells = (raw[: self._cells] % alphabet).astype(np.int64)
        tail = raw[self._cells :]
        operands = (tail[: self._length] % alphabet).astype(np.int64)
        operators = (tail[self._length :][: self._length - 1] % np.uint64(2)).astype(np.int64)

        tokens = np.empty(self._width, dtype=_TOKEN_DTYPE)
        tokens[:2] = self._prefix
        cursor = 2
        for operator in range(2):
            tokens[cursor] = self._operator_ids[operator]
            cursor += 1
            for left in range(self._alphabet):
                tokens[cursor] = self._residue_ids[left]
                tokens[cursor + 1] = self._equals[0]
                cursor += 2
                start = (operator * self._alphabet + left) * self._alphabet
                row = cells[start : start + self._alphabet]
                tokens[cursor : cursor + self._alphabet] = self._residue_ids[row]
                cursor += self._alphabet

        total = int(operands[0])
        tokens[cursor] = self._residue_ids[total]
        cursor += 1
        for position in range(1, self._length):
            operator = int(operators[position - 1])
            operand = int(operands[position])
            tokens[cursor] = self._operator_ids[operator]
            tokens[cursor + 1] = self._residue_ids[operand]
            cursor += 2
            total = int(cells[(operator * self._alphabet + total) * self._alphabet + operand])

        tokens[cursor] = self._equals[0]
        cursor += 1
        answer_start = cursor
        tokens[cursor] = self._residue_ids[total]
        tokens[cursor + 1] = self._eos[0]
        # After the eos, so the answer's offset is unchanged and every scorer indexing `answer_start`
        # keeps working. `PAD` rather than `EOS`: the document-boundary machinery counts eos occurrences
        # to size its per-instance array, and padding that shares that id inflates the count.
        if cursor + 2 < self._width:
            tokens[cursor + 2 :] = self._pad_id
        return TaskItem(tokens, answer_start, answer_start + 1, (f"<n{total}>",))

    @property
    def natural_width(self) -> int:
        """Tokens an item needs before any instance-alignment padding."""
        return self._natural_width

    @property
    def padding(self) -> int:
        """Padding tokens per item, zero when unaligned or already aligned."""
        return self._width - self._natural_width

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for field in (
            "factcrowd.InContextManoTask.v1",
            str(self._pad_to),
            self._vocabulary.fingerprint(),
            self._domain_token,
            str(self._length),
            str(self._alphabet),
            str(self._seed),
            self._split,
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
        split: str = "train",
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
        if split not in SPLITS:
            raise OLMoConfigurationError(f"unknown split {split!r}; expected one of {SPLITS}")
        self._seed = seed
        self._split = split
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
        draw = np.uint64(
            item_key(
                class_tag=0x434D5052, split=self._split, seed=self._seed, index=index  # 'CMPR'
            )
        )
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
            self._split,
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
        No sequence length divides both widths -- 504 is 24x21 but not a multiple of 19 -- so the fix is
        padding both tasks to 32 tokens with a label mask.
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
