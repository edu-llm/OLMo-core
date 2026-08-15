"""Shared OLMo-core model config for curriculum arms.

Stock OLMo2 370M architecture via ``TransformerConfig.olmo2_370M``.
This is the same dense model used by the already-run ``curriculum`` W&B arms:
``d_model=1024``, 16 layers, 16 heads, reordered RMSNorm, QK-norm, RoPE
theta 500,000. Vocabulary stays Dolma2-padded 100,352 so the sealed
regmix curriculum data remains valid.
"""

from __future__ import annotations

import os
from typing import Any

from olmo_core.nn.transformer import TransformerConfig

VOCAB_SIZE = 100_352
N_LAYERS = 16
N_HEADS = 16
MODEL_IDENTITY = "TransformerConfig.olmo2_370M"


def build_model_config(vocab_size: int = VOCAB_SIZE, **kwargs: Any) -> TransformerConfig:
    """Return the fixed stock OLMo2-370M config used by every arm and eval."""
    if os.environ.get("EDULLM_BENCH_FLASH") == "1":
        kwargs.setdefault("use_flash", True)
    if os.environ.get("EDULLM_BENCH_FUSED_OPS") == "1":
        kwargs.setdefault("fused_ops", True)
    return TransformerConfig.olmo2_370M(
        vocab_size=vocab_size,
        n_layers=kwargs.pop("n_layers", N_LAYERS),
        n_heads=kwargs.pop("n_heads", N_HEADS),
        **kwargs,
    )
