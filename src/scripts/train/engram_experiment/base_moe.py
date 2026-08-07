"""Base reordered-norm MoE arm for the sealed 10B-token Engram experiment."""

from typing import Optional, Sequence

from olmo_core.nn.transformer import TransformerConfig
from scripts.train.engram_experiment import common

# Ordinary MoE dimensions tuned to put the arm near 400M total and 100M active parameters.
MODEL_DIM = 384
NUM_LAYERS = 12
NUM_HEADS = 6
NUM_EXPERTS = 64
EXPERT_HIDDEN_SIZE = 336

# Exact accounting produced by TransformerConfig for the dimensions above.
EXPECTED_TOTAL_PARAMETERS = 392_373_120
EXPECTED_ACTIVE_PARAMETERS = 113_681_280


def build_model_config() -> TransformerConfig:
    """Build the reordered-norm MoE model without block or memory overrides."""

    model = TransformerConfig.smallmoe(
        vocab_size=100_352,
        d_model=MODEL_DIM,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
    )
    assert model.block.feed_forward_moe is not None
    model.block.feed_forward_moe.num_experts = NUM_EXPERTS
    model.block.feed_forward_moe.hidden_size = EXPERT_HIDDEN_SIZE
    return model


def build_config() -> common.ExperimentConfig:
    """Build the shared local-only experiment configuration."""

    return common.build_experiment_config(build_model_config())


def main(argv: Optional[Sequence[str]] = None) -> common.ExperimentConfig:
    """Build locally by default; train only for the explicit ``train`` command."""

    return common.dispatch(build_model_config, argv=argv, prog=__file__)


if __name__ == "__main__":
    main()
