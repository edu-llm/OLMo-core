"""Shared OLMo-core model config for curriculum 370M MoE arms.

Configuration A: same backbone as dense ``olmo2_370M`` (d=1024, 16 layers,
16 heads, reordered norm) with an 8-expert top-2 MoE MLP whose active width
matches the dense gated-SiLU 4096 MLP (``top_k * expert_hidden = 4096``).
"""

from __future__ import annotations

from typing import Any

from olmo_core.nn.transformer import TransformerConfig

VOCAB_SIZE = 100_352
D_MODEL = 1024
N_LAYERS = 16
N_HEADS = 16
NUM_EXPERTS = 8
TOP_K = 2
EXPERT_HIDDEN_SIZE = 2048
ROPE_THETA = 500_000
MODEL_IDENTITY = (
    "TransformerConfig.llama_like_moe("
    "d_model=1024,n_layers=16,n_heads=16,num_experts=8,top_k=2,expert_hidden_size=2048)"
)


def build_model_config(vocab_size: int = VOCAB_SIZE, **kwargs: Any) -> TransformerConfig:
    """Return the fixed MoE config used by every curriculum arm and task-loss eval."""
    return TransformerConfig.llama_like_moe(
        d_model=kwargs.pop("d_model", D_MODEL),
        vocab_size=vocab_size,
        n_layers=kwargs.pop("n_layers", N_LAYERS),
        n_heads=kwargs.pop("n_heads", N_HEADS),
        num_experts=kwargs.pop("num_experts", NUM_EXPERTS),
        top_k=kwargs.pop("top_k", TOP_K),
        expert_hidden_size=kwargs.pop("expert_hidden_size", EXPERT_HIDDEN_SIZE),
        reordered_norm=kwargs.pop("reordered_norm", True),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", ROPE_THETA),
        layer_norm_eps=kwargs.pop("layer_norm_eps", 1e-6),
        **kwargs,
    )
