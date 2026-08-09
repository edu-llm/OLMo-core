"""Shared OLMo-core model config for curriculum arms.

Stock OLMoE-1B-7B architecture via ``TransformerConfig.olmoe_1B_7B``:
``d_model=2048``, 16 layers, 16 heads, dropless MoE with 64 experts / top-8
and expert hidden ``0.5 * d_model`` (1024), reordered RMSNorm, QK-norm,
RoPE theta 500,000. Vocabulary stays Dolma2-padded 100,352 so the sealed
regmix curriculum data remains valid.
"""

from __future__ import annotations

import os
from typing import Any

from olmo_core.nn.transformer import TransformerConfig

VOCAB_SIZE = 100_352
D_MODEL = 2048
N_LAYERS = 16
N_HEADS = 16
NUM_EXPERTS = 64
TOP_K = 8
EXPERT_HIDDEN_SIZE = 1024  # 0.5 * d_model
ROPE_THETA = 500_000
MODEL_IDENTITY = (
    "TransformerConfig.olmoe_1B_7B("
    "d_model=2048,n_layers=16,n_heads=16,num_experts=64,top_k=8,expert_hidden_size=1024)"
)


def build_model_config(vocab_size: int = VOCAB_SIZE, **kwargs: Any) -> TransformerConfig:
    """Return the fixed stock OLMoE-1B-7B config used by every arm and eval."""
    if os.environ.get("EDULLM_BENCH_FLASH") == "1":
        kwargs.setdefault("use_flash", True)
    if os.environ.get("EDULLM_BENCH_FUSED_OPS") == "1":
        kwargs.setdefault("fused_ops", True)
    return TransformerConfig.olmoe_1B_7B(
        vocab_size=vocab_size,
        d_model=kwargs.pop("d_model", D_MODEL),
        n_layers=kwargs.pop("n_layers", N_LAYERS),
        n_heads=kwargs.pop("n_heads", N_HEADS),
        **kwargs,
    )
