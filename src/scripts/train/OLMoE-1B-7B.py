"""
Train a ~7B OLMoE-style model with 32 routed experts and top-4 routing for
100B tokens.

This mirrors the ``olmoe_7b_32x4`` recipe in ``.edullm/train_on_corpus.py``,
which is the entrypoint an eduLLM run actually uses. Keep the two in step: a
model that differs between them is two experiments wearing one name.

Run this script without any arguments to see usage info.
"""

from functools import partial

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CommonComponents, build_config, main
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, CometCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH
TRAINING_TOKENS = 100_000_000_000

MODEL_DIM = 2048
NUM_LAYERS = 16
NUM_HEADS = 16
NUM_ROUTED_EXPERTS = 32
TOP_K = 4
ROUTED_EXPERT_HIDDEN_SIZE = 2048
# OLMo-core represents shared experts as one always-on MLP, so ``None`` is none
# of them. Shared experts are a screened arm rather than part of the base, and
# the arm sets this to twice a routed expert's width for two of them.
SHARED_EXPERT_HIDDEN_SIZE = None


def build_model_config(common: CommonComponents) -> TransformerConfig:
    return TransformerConfig.llama_like_moe(
        vocab_size=common.tokenizer.padded_vocab_size(),
        d_model=MODEL_DIM,
        n_layers=NUM_LAYERS,
        n_heads=NUM_HEADS,
        num_experts=NUM_ROUTED_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=ROUTED_EXPERT_HIDDEN_SIZE,
        shared_expert_hidden_size=SHARED_EXPERT_HIDDEN_SIZE,
        dropless=True,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )


def build_train_module_config(common: CommonComponents) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=2 * common.max_sequence_length,
        max_sequence_length=common.max_sequence_length,
        optim=AdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=2000),
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    return (
        TrainerConfig(
            save_folder=common.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            max_duration=Duration.tokens(TRAINING_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=10_000,
                ephemeral_save_interval=1000,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=common.run_name,
                workspace="ai2",
                project="OLMo-core-7B",
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=common.run_name,
                entity="ai2-llm",
                project="OLMo-core-7B",
                enabled=False,
                cancel_check_interval=10,
            ),
        )
    )


if __name__ == "__main__":
    config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
    )
    main(config_builder=config_builder)
