"""Co-LMLM special tokens, matching the released ``lil-lab/CoLMLM-*`` tokenizer exactly.

The released tokenizer adds three tokens on top of the SmolLM2 vocabulary (49,152), giving a
final vocab of 49,155. They are appended in this order, so the opening token gets the first new
id, etc.:

    <FACT>    (49152)  opening / retrieval trigger; its hidden state is the retrieval query
    </FACT>   (49153)  closing; appended mechanically after the (retrieved) fact span
    <FACT-q>  (49154)  query marker appended to a question; its hidden state is the question query
"""

from dataclasses import dataclass

#: Opening fact token. At inference, emitting this triggers a retrieval; at training, its position's
#: hidden state is the document-side ("fact") query for the contrastive loss.
FACT_OPEN = "<FACT>"

#: Closing fact token, appended mechanically after the fact span. Not optimized for generation.
FACT_CLOSE = "</FACT>"

#: Query marker appended to a synthesized question. Its position's hidden state is the
#: question-side query for the contrastive loss.
FACT_QUERY = "<FACT-q>"

#: The three Co-LMLM tokens, in the order they are appended to the base vocabulary.
SPECIAL_TOKENS = (FACT_OPEN, FACT_CLOSE, FACT_QUERY)

NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

#: Size of the SmolLM2 tokenizer vocabulary before the Co-LMLM tokens are added.
SMOLLM2_BASE_VOCAB_SIZE = 49152


@dataclass(frozen=True)
class SpecialTokenIds:
    """The token ids assigned to the three Co-LMLM special tokens."""

    fact_open: int
    fact_close: int
    fact_query: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.fact_open, self.fact_close, self.fact_query)


def special_token_ids(base_vocab_size: int = SMOLLM2_BASE_VOCAB_SIZE) -> SpecialTokenIds:
    """Return the ids the Co-LMLM tokens receive when appended after ``base_vocab_size``.

    This matches the released checkpoint when ``base_vocab_size == 49152``: ``<FACT>`` -> 49152,
    ``</FACT>`` -> 49153, ``<FACT-q>`` -> 49154.
    """
    return SpecialTokenIds(
        fact_open=base_vocab_size,
        fact_close=base_vocab_size + 1,
        fact_query=base_vocab_size + 2,
    )


def colmlm_vocab_size(base_vocab_size: int = SMOLLM2_BASE_VOCAB_SIZE) -> int:
    """Vocab size after adding the Co-LMLM tokens (49,155 for the SmolLM2 tokenizer)."""
    return base_vocab_size + NUM_SPECIAL_TOKENS
