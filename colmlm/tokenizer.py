"""SmolLM2 tokenizer config for OLMo-core.

The Co-LMLM corpus is tokenized with ``HuggingFaceTB/SmolLM2-135M`` (see the corpus ``meta.json``).
Its vocab is 49,152 (already a multiple of 128, so no embedding padding is needed) and its
BOS/EOS is ``<|endoftext|>`` = id 0.
"""

from olmo_core.data import TokenizerConfig

SMOLLM2_VOCAB_SIZE = 49152
SMOLLM2_EOS_ID = 0


def smollm2_tokenizer_config() -> TokenizerConfig:
    """OLMo-core :class:`TokenizerConfig` matching the SmolLM2 tokenizer."""
    return TokenizerConfig(
        vocab_size=SMOLLM2_VOCAB_SIZE,
        eos_token_id=SMOLLM2_EOS_ID,
        bos_token_id=SMOLLM2_EOS_ID,
        pad_token_id=SMOLLM2_EOS_ID,
        identifier="HuggingFaceTB/SmolLM2-135M",
    )
