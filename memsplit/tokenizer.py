"""Tokenizer with frozen special-token ids, and a dependency-free fallback.

Two invariants matter for this project and both are enforced here.

**Segment-independent encoding.** `encode_segments` encodes each segment on its
own so no BPE merge can straddle a mask boundary. If a merge crossed the edge
between "...is" and " Paris", one token would be partly supervised and partly
masked, and the mask would no longer mean what the design says it means.

**Both arms share one vocabulary.** The control tokens are in-vocabulary for the
dense arm too, even though it never emits them. That is what makes a cross-arm
KL well defined on a shared support -- see docs/METHODOLOGY.md. Do not "save"
embedding parameters by giving the dense arm a smaller vocab.

The fallback tokenizer exists so the corpus builders, maskers and scorers are
testable on a machine with no `tiktoken` wheel. It is byte-level, so it is
trivially segment-independent, which makes it a *stricter* test of mask-boundary
logic than BPE rather than a weaker one. It is not for training.
"""

from __future__ import annotations

import os
from functools import lru_cache

DB_START = "<|db_start|>"
DB_RETRIEVE = "<|db_retrieve|>"
DB_END = "<|db_end|>"
DB_FAIL = "<|db_fail|>"
EOT = "<|eot|>"

# Frozen ids. Changing these invalidates every existing .bin, so they are
# asserted rather than derived.
SPECIAL_IDS = {
    DB_START: 50257,
    DB_RETRIEVE: 50258,
    DB_END: 50259,
    EOT: 50260,
    DB_FAIL: 50261,
}
VOCAB_SIZE = 50304  # padded; identical for both arms, deliberately


class _TokBase:
    VOCAB_SIZE = VOCAB_SIZE

    def __init__(self) -> None:
        for name, tid in SPECIAL_IDS.items():
            attr = name.strip("<|>").upper()
            setattr(self, attr, tid)
        self.SPECIALS = dict(SPECIAL_IDS)

    def encode(self, text: str) -> list[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def encode_segments(
        self, segments: list[tuple[str, bool]]
    ) -> tuple[list[int], list[int]]:
        """Encode `(text, masked)` segments to (ids, loss_mask).

        `loss_mask[i] == 1` means loss is ON for token i. Each segment is
        encoded independently, so a mask edge always lands on a token boundary.
        """
        ids: list[int] = []
        mask: list[int] = []
        for text, masked in segments:
            if not text:
                continue
            seg = self.encode(text)
            ids.extend(seg)
            mask.extend([0 if masked else 1] * len(seg))
        return ids, mask


class ByteTok(_TokBase):
    """Byte-level fallback: one id per UTF-8 byte, specials above 256.

    Deterministic, dependency-free, and reversible. Used by the test suite and
    by corpus-shape validation; never used to train a reported model.
    """

    name = "byte-fallback"

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            if text[i] == "<":
                for lit, tid in SPECIAL_IDS.items():
                    if text.startswith(lit, i):
                        out.append(tid)
                        i += len(lit)
                        matched = True
                        break
            if not matched:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids: list[int]) -> str:
        rev = {v: k for k, v in SPECIAL_IDS.items()}
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            if i in rev:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(rev[i])
            elif i < 256:
                buf.append(i)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


class TiktokenTok(_TokBase):
    """Production tokenizer: gpt2 BPE plus the five control tokens."""

    name = "gpt2+control"

    def __init__(self) -> None:
        super().__init__()
        import tiktoken

        base = tiktoken.get_encoding("gpt2")
        self._enc = tiktoken.Encoding(
            name="gpt2-memsplit",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **SPECIAL_IDS},
        )
        assert self._enc.n_vocab <= VOCAB_SIZE, self._enc.n_vocab
        self._allowed = set(SPECIAL_IDS)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special=self._allowed)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


@lru_cache(maxsize=None)
def get_tok(prefer: str | None = None):
    """Return the production tokenizer, or the fallback if unavailable.

    `MEMSPLIT_TOKENIZER=byte` forces the fallback. Anything that writes a
    trainable corpus must assert `tok.name == "gpt2+control"`; see
    `require_production_tokenizer`.
    """
    choice = prefer or os.environ.get("MEMSPLIT_TOKENIZER", "auto")
    if choice == "byte":
        return ByteTok()
    if os.environ.get("TIKTOKEN_CACHE_DIR") is None:
        cache = os.path.join(os.getcwd(), ".tiktoken_cache")
        if os.path.isdir(cache):
            os.environ["TIKTOKEN_CACHE_DIR"] = cache
    try:
        return TiktokenTok()
    except Exception:
        if choice == "gpt2":
            raise
        return ByteTok()


def require_production_tokenizer(tok) -> None:
    """Refuse to write a trainable artifact with the fallback tokenizer."""
    if tok.name != "gpt2+control":
        raise RuntimeError(
            "refusing to write a trainable corpus with the "
            f"{tok.name!r} tokenizer; install tiktoken or pass --allow-fallback "
            "for shape-only validation"
        )
