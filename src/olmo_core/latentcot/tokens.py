"""
Special tokens and tokenizer loading for the latent-CoT experiments (PRD Phase 2.1).

We reuse the dolma2 vocabulary (so the base ``olmo2_370M`` embeddings apply) and add
our control tokens at the top of the *padded* vocab region. The model embedding is
sized to ``padded_vocab_size`` (100352) while the real vocab ends at 100278, so ids
100278..100351 are unused — our four control tokens live there and therefore never
collide with a real token and need no embedding resize.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, List

from olmo_core.data import TokenizerConfig

if TYPE_CHECKING:
    from tokenizers import Tokenizer

__all__ = [
    "TOKENIZER_CONFIG",
    "VOCAB_SIZE",
    "PADDED_VOCAB_SIZE",
    "BOT",
    "EOT",
    "DISTILL",
    "THOUGHT",
    "SPECIAL_TOKENS",
    "load_tokenizer",
    "encode",
]

TOKENIZER_CONFIG = TokenizerConfig.dolma2()
VOCAB_SIZE = TOKENIZER_CONFIG.vocab_size  # 100278 (real tokens: ids 0..100277)
PADDED_VOCAB_SIZE = TOKENIZER_CONFIG.padded_vocab_size()  # 100352 (model embedding rows)

# Control tokens, placed in the unused padded region (>= VOCAB_SIZE).
BOT = PADDED_VOCAB_SIZE - 1  # begin-of-thought (start of the reasoning region)
EOT = PADDED_VOCAB_SIZE - 2  # end-of-thought (end of the reasoning region)
DISTILL = PADDED_VOCAB_SIZE - 3  # alignment token; hidden state distilled teacher->student
THOUGHT = PADDED_VOCAB_SIZE - 4  # placeholder for a student latent slot (embedding overwritten)
SPECIAL_TOKENS = {"BOT": BOT, "EOT": EOT, "DISTILL": DISTILL, "THOUGHT": THOUGHT}


@lru_cache(maxsize=1)
def load_tokenizer() -> "Tokenizer":
    """
    Load the dolma2 tokenizer (cached). Downloads ``tokenizer.json`` from the Hub on
    first use, falling back to an already-cached copy when the Hub is unreachable.

    That fallback is what makes this usable inside a training container. Every encode path
    reaches here, so with no egress to huggingface.co the run dies at the first
    ``LatentCotDataset`` access — before step 1, having already paid for the GPU. The eduLLM
    image pre-warms this cache at build time, so ``local_files_only`` finds the file.

    :raises ImportError: If the ``tokenizers`` package is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "The latent-CoT tokenization needs the `tokenizers` package "
            "(a transitive dep of the `transformers` extra). Install with: "
            "`uv pip install tokenizers`."
        ) from e
    identifier = str(TOKENIZER_CONFIG.identifier)
    try:
        path = hf_hub_download(identifier, "tokenizer.json")
    except Exception:  # pragma: no cover - network dependent
        # Offline / no egress: use the baked cache rather than failing the run.
        path = hf_hub_download(identifier, "tokenizer.json", local_files_only=True)
    return Tokenizer.from_file(path)


def encode(text: str) -> List[int]:
    """Encode text to real dolma2 token ids (no special tokens added)."""
    return load_tokenizer().encode(text, add_special_tokens=False).ids


def decode(ids: List[int]) -> str:
    """Decode real dolma2 token ids back to text (control tokens must be stripped first)."""
    return load_tokenizer().decode(ids)
