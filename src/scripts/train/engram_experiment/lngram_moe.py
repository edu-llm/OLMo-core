"""Lngram memory-MoE arm for the sealed 10B-token Engram experiment."""

from typing import Optional, Sequence

from olmo_core.nn.memory import LngramConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from scripts.train.engram_experiment import base_moe, common

NUM_EXPERTS = 51
MEMORY_DIM = 61

# Exact accounting produced by TransformerConfig for the dimensions below.
EXPECTED_TOTAL_PARAMETERS = 392_198_016
EXPECTED_ACTIVE_PARAMETERS = 122_942_208
# Lngram's discretization and shared key/value projections are dense, so its active count
# is 8.15% above the base even though sparse table rows and routed experts stay near-isometric.
ACTIVE_PARAMETER_BUMP_REASON = "dense discretization and shared key/value projections"


def build_model_config() -> TransformerConfig:
    """Build the MoE model with two fully copied Lngram block overrides."""

    model = TransformerConfig.smallmoe(
        vocab_size=100_352,
        d_model=base_moe.MODEL_DIM,
        n_layers=base_moe.NUM_LAYERS,
        n_heads=base_moe.NUM_HEADS,
    )
    assert model.block.feed_forward_moe is not None
    model.block.feed_forward_moe.num_experts = NUM_EXPERTS
    model.block.feed_forward_moe.hidden_size = base_moe.EXPERT_HIDDEN_SIZE
    model.checkpoint_revision = common.EXPERIMENT_REVISION

    # Paper layers 2 and 6 expressed as zero-based OLMo block indices.
    override_indices = (1, model.n_layers // 2 - 1)
    model.block_overrides = {}
    for layer_idx in override_indices:
        block = model.block.copy()
        block.name = TransformerBlockType.moe_lngram_reordered_norm
        block.memory = LngramConfig(
            orders=(2, 3),
            bits_per_route=4,
            memory_dim=MEMORY_DIM,
            conv_dilation=3,
            require_triton=True,
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
