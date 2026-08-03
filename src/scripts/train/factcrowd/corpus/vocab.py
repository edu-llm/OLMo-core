"""
The vocabulary: every word the synthetic corpus can contain, and its token id.

A synthetic corpus over closed pools has a **closed vocabulary**, so it can be enumerated rather than
learned. That is worth taking advantage of, for three reasons beyond convenience.

**It makes the value-token spans exact by construction.** The bit-counter must sum loss over exactly
the tokens carrying an attribute value. With a word-level vocabulary a value is one token per word, so
a span is arithmetic -- no alignment heuristic, no post-hoc recovery, and no way for the eval prompt
and the rendered prose to disagree about where a value starts.

**It removes the tokenizer as a variable while the pipeline is being built.** A BPE trained on the
mixture is the real plan (PRD.md §6.3), and it changes token counts but nothing else about the
experiment. Swapping it in behind :class:`Vocabulary` is a contained change; discovering a corpus bug
through a BPE is not.

**It is small.** The bioS schema's pools plus templates and specials come to about 3,300 word types.
At ``d_model=256`` a tied 3,328-row embedding table is 0.85M parameters against 12.6M non-embedding,
so the table is 6% of the model rather than the 39% a 32k vocabulary costs -- which matters because
:math:`\\rho` is defined against non-embedding capacity and a reader will look at that ratio.

**What this is not.** It is not a tokenizer for natural text: it has no subword units, no unknown-token
handling, and it will refuse a word it has never seen. That refusal is deliberate. A synthetic corpus
producing an out-of-vocabulary word means the renderer and the schema disagree, which is a bug to
raise rather than to encode as ``<unk>``.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .entities import Schema

__all__ = [
    "PAD",
    "EOS",
    "BOS",
    "SPECIALS",
    "Vocabulary",
]


PAD = "<pad>"
EOS = "<eos>"
BOS = "<bos>"

SPECIALS: Tuple[str, ...] = (PAD, EOS, BOS)
"""
Reserved ids 0, 1, 2, in that order.

``PAD`` is id 0 so a zero-filled buffer decodes as padding rather than as content. ``PAD`` and ``EOS``
are **distinct**: OLMo-core's document-boundary machinery counts EOS occurrences to size its
per-instance document array, and a corpus where padding and EOS share an id inflates that count until
it explodes.
"""

_TOKEN_DTYPE = np.uint32
"""Token ids as uint32, matching the eduLLM standard's ``.u32le.bin`` shard format."""


@dataclass(frozen=True)
class Vocabulary:
    """
    A closed word-to-id mapping, plus the per-pool lookup tables the renderer needs.

    Build with :meth:`build`, which is the only way to get the id ordering right.

    :param words: Every word, in id order. ``words[i]`` has id ``i``.
    :param pool_token_ids: Pool name to an array mapping a pool index to its token id. This is what
        turns an entity's stored indices into tokens with no string handling at train time.
    """

    words: Tuple[str, ...]
    pool_token_ids: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if len(set(self.words)) != len(self.words):
            duplicated = sorted({w for w in self.words if self.words.count(w) > 1})
            raise OLMoConfigurationError(
                f"the vocabulary has duplicate words {duplicated[:5]}, so a word would have two ids"
            )
        for index, special in enumerate(SPECIALS):
            if self.words[index] != special:
                raise OLMoConfigurationError(
                    f"id {index} must be {special!r}, got {self.words[index]!r}. The special ids are "
                    f"fixed because a zero-filled buffer must decode as padding."
                )

    @property
    def size(self) -> int:
        """Number of distinct token ids."""
        return len(self.words)

    @property
    def pad_id(self) -> int:
        """Token id of :data:`PAD`. Always 0."""
        return 0

    @property
    def eos_id(self) -> int:
        """Token id of :data:`EOS`. Always 1."""
        return 1

    @property
    def bos_id(self) -> int:
        """Token id of :data:`BOS`. Always 2."""
        return 2

    def padded_size(self, multiple_of: int = 128) -> int:
        """
        :attr:`size` rounded up to a multiple, for embedding-matrix efficiency.

        The unused rows never appear in the data, so they cost parameters and no correctness. Pass the
        result to the model config and the real :attr:`size` to the renderer.

        :param multiple_of: Rounding quantum. 128 suits tensor cores.

        :returns: The padded vocabulary size.

        :raises OLMoConfigurationError: If ``multiple_of`` is not positive.
        """
        if multiple_of <= 0:
            raise OLMoConfigurationError(f"'multiple_of' must be positive, got {multiple_of}")
        return multiple_of * ((self.size + multiple_of - 1) // multiple_of)

    def id_of(self, word: str) -> int:
        """
        The token id of one word.

        :param word: The word to look up.

        :returns: Its token id.

        :raises OLMoConfigurationError: If the word is not in the vocabulary. Deliberately a refusal
            rather than an unknown token: a synthetic corpus emitting an unseen word means the
            renderer and the schema disagree.
        """
        try:
            return self._index[word]
        except KeyError:
            raise OLMoConfigurationError(
                f"{word!r} is not in the vocabulary. This corpus has a closed vocabulary, so an "
                f"unseen word means the renderer and the schema disagree rather than that the text "
                f"is unusual."
            ) from None

    def encode(self, words: Sequence[str]) -> np.ndarray:
        """
        Encode a word sequence to token ids.

        :param words: The words.

        :returns: A uint32 array of ids.

        :raises OLMoConfigurationError: If any word is absent.
        """
        return np.fromiter(
            (self.id_of(word) for word in words), dtype=_TOKEN_DTYPE, count=len(words)
        )

    def decode(self, token_ids: Iterable[int]) -> Tuple[str, ...]:
        """
        Decode token ids back to words, for tests and for eyeballing a sample.

        :param token_ids: The ids.

        :returns: The words.

        :raises OLMoConfigurationError: If an id is out of range.
        """
        decoded = []
        for token_id in token_ids:
            index = int(token_id)
            if not 0 <= index < self.size:
                raise OLMoConfigurationError(
                    f"token id {index} is outside the vocabulary of {self.size}"
                )
            decoded.append(self.words[index])
        return tuple(decoded)

    def fingerprint(self) -> str:
        """
        A length-framed digest of the word list in id order.

        Framed for the same reason :meth:`~factcrowd.corpus.entities.Schema.fingerprint` is: without
        it a word containing the separator could impersonate a boundary.

        :returns: A hex digest.
        """
        digest = hashlib.sha256()
        digest.update(len(self.words).to_bytes(8, "big"))
        for word in self.words:
            raw = word.encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    @property
    def _index(self) -> Dict[str, int]:
        """Word to id. Rebuilt per access on a frozen dataclass, so callers use :meth:`encode`."""
        return {word: index for index, word in enumerate(self.words)}

    @classmethod
    def build(
        cls,
        schema: Schema,
        *,
        literal_words: Sequence[str] = (),
        domain_tokens: Sequence[str] = (),
    ) -> "Vocabulary":
        """
        Enumerate the vocabulary of a schema, plus template literals and domain tokens.

        Ids are assigned in a fixed order -- specials, domain tokens, literals, then pools in schema
        order -- so the mapping depends on the schema and the template set and nothing else. Two
        callers building from the same inputs get the same ids, which is what lets a checkpoint and a
        corpus be checked against each other by fingerprint.

        :param schema: The pools. Every pool value becomes a word.
        :param literal_words: Words appearing in templates, e.g. ``was``, ``born``, ``.``.
        :param domain_tokens: One per corpus slice, e.g. ``<facts>``, ``<mano>``. Mandatory in the
            mixture (PRD.md §3.4), so they are reserved here rather than discovered later.

        :returns: The vocabulary.

        :raises OLMoConfigurationError: If a word appears in more than one role -- a pool value that
            is also a template literal would make that token ambiguous between prose and a fact.
        """
        words: list = list(SPECIALS)
        seen = set(words)

        for group_name, group in (
            ("domain_tokens", domain_tokens),
            ("literal_words", literal_words),
        ):
            for word in group:
                if word in seen:
                    raise OLMoConfigurationError(
                        f"{word!r} in {group_name} is already a word. A token serving two roles is "
                        f"ambiguous between prose and a fact, which makes recall unmeasurable."
                    )
                seen.add(word)
                words.append(word)

        pool_token_ids: Dict[str, np.ndarray] = {}
        for pool in tuple(schema.attributes) + tuple(schema.names):
            ids = np.empty(len(pool), dtype=_TOKEN_DTYPE)
            for pool_index, value in enumerate(pool.values):
                if value in seen:
                    raise OLMoConfigurationError(
                        f"pool value {value!r} in '{pool.name}' is already a word. Pools must be "
                        f"disjoint from each other and from the template literals -- see "
                        f"factcrowd.corpus.values.allocate_words."
                    )
                seen.add(value)
                ids[pool_index] = len(words)
                words.append(value)
            pool_token_ids[pool.name] = ids

        return cls(words=tuple(words), pool_token_ids=pool_token_ids)
