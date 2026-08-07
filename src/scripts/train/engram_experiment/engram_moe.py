"""Engram memory-MoE arm for the sealed 10B-token experiment."""

from typing import Optional, Sequence

from olmo_core.nn.memory import EngramConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from scripts.train.engram_experiment import common
from scripts.train.engram_experiment.base_moe import (
    EXPERT_HIDDEN_SIZE,
    MODEL_DIM,
    NUM_HEADS,
    NUM_LAYERS,
)

NUM_EXPERTS = 51
EMBEDDING_DIM = 32
TABLE_SIZES = (58_193, 58_313)

# Exact TransformerConfig accounting for the dimensions above.
EXPECTED_TOTAL_PARAMETERS = 392_372_864
EXPECTED_ACTIVE_PARAMETERS = 114_414_208


def build_model_config() -> TransformerConfig:
    """Build the MoE model with two fully copied Engram block overrides."""

    model = TransformerConfig.smallmoe(
        vocab_size=100_352,
        d_model=MODEL_DIM,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
    )
    assert model.block.feed_forward_moe is not None
    model.block.feed_forward_moe.num_experts = NUM_EXPERTS
    model.block.feed_forward_moe.hidden_size = EXPERT_HIDDEN_SIZE

    model.block_overrides = {}
    for layer_idx in (2, model.n_layers // 2):
        block = model.block.copy()
        block.name = TransformerBlockType.moe_engram_reordered_norm
        block.memory = EngramConfig(
            orders=(2, 3),
            num_hash_heads=8,
            table_sizes=TABLE_SIZES,
            embedding_dim=EMBEDDING_DIM,
            vocab_size=100_352,
            # Keep the compression hook enabled; without a committed Dolma2 map this
            # scaffold intentionally uses EngramConfig's documented identity fallback.
            tokenizer_compression=True,
        )
        model.block_overrides[layer_idx] = block

    return model


def build_config() -> common.ExperimentConfig:
    """Build the shared local-only experiment config with memory optimization."""

    return common.build_experiment_config(
        build_model_config(),
        with_memory_optimizer=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> common.ExperimentConfig:
    """Build locally by default; train only for the explicit ``train`` command."""

    return common.dispatch(
        build_model_config,
        with_memory_optimizer=True,
        argv=argv,
        prog=__file__,
    )


if __name__ == "__main__":
    main()
