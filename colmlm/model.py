"""Co-LMLM model configs: the SmolLM2 backbone with the three Co-LMLM tokens added to the vocab.

The Co-LMLM model is architecturally identical to SmolLM2 (see
:meth:`olmo_core.nn.transformer.TransformerConfig.smollm2_135M`); the only difference is that the
vocabulary is 3 larger to hold ``<FACT>``, ``</FACT>``, ``<FACT-q>``. Everything that makes it
"Co-LMLM" lives in the training loss and inference-time retrieval, not in the model parameters.
"""

from typing import Optional

from olmo_core.nn.transformer import TransformerConfig

from .special_tokens import SMOLLM2_BASE_VOCAB_SIZE, colmlm_vocab_size


def _resolve_vocab(base_vocab_size: int, pad_to_multiple_of: Optional[int]) -> int:
    vocab = colmlm_vocab_size(base_vocab_size)
    if pad_to_multiple_of:
        vocab = ((vocab + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    return vocab


def colmlm_smollm2_135M(
    base_vocab_size: int = SMOLLM2_BASE_VOCAB_SIZE,
    *,
    pad_to_multiple_of: Optional[int] = None,
    **kwargs,
) -> TransformerConfig:
    """The 135M Co-LMLM model config: SmolLM2-135M with the 3 Co-LMLM tokens added.

    :param base_vocab_size: Size of the underlying tokenizer's vocab before the Co-LMLM tokens
        (49,152 for the SmolLM2 tokenizer, which is what the paper uses).
    :param pad_to_multiple_of: Optionally pad the vocab up to a multiple of this value (e.g. 128
        for faster matmuls). Defaults to ``None`` to match the released checkpoint exactly
        (vocab 49,155). The Co-LMLM token ids are unaffected by padding since they are appended
        directly after the base vocab.
    """
    return TransformerConfig.smollm2_135M(
        vocab_size=_resolve_vocab(base_vocab_size, pad_to_multiple_of), **kwargs
    )


def colmlm_smollm2_360M(
    base_vocab_size: int = SMOLLM2_BASE_VOCAB_SIZE,
    *,
    pad_to_multiple_of: Optional[int] = None,
    **kwargs,
) -> TransformerConfig:
    """The 360M Co-LMLM model config: SmolLM2-360M with the 3 Co-LMLM tokens added.

    Reproduces the released ``lil-lab/CoLMLM-360M-FW`` architecture (vocab 49,155) when
    ``base_vocab_size=49152`` and ``pad_to_multiple_of=None``.
    """
    return TransformerConfig.smollm2_360M(
        vocab_size=_resolve_vocab(base_vocab_size, pad_to_multiple_of), **kwargs
    )
