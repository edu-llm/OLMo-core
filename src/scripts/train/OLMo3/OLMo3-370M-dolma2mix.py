"""
Pre-train a 370M OLMo-3 model on the edu-llm dolma2 source mixture (AWS / torchrun).

This is a self-contained ``torchrun`` entrypoint (no Beaker / Gantry). It is modelled on
``src/examples/llm/train.py`` but wired to:

  * the ``olmo3_370M`` architecture (371M ladder rung: d_model=1024, 16 layers, 16 heads;
    OLMo-3 sliding-window attention, defaulting to the flash_2 backend);
  * the **dolma2** tokenizer (must match the pre-tokenized data);
  * the data team's dolma2 source mixture, loaded from a ``SourceMixtureList`` YAML
    (default: ``s3://edullm-datasets/olmo-150b-dolma2/configs/equal-weighting-config.yaml``);
  * S3 checkpoints, auto-resumed from ``--save-folder`` (cadence set by ``--checkpoint-tokens``).

Launch with torchrun, e.g. on an 8-GPU node:

    torchrun --standalone --nproc-per-node=8 src/scripts/train/OLMo3/OLMo3-370M-dolma2mix.py my-run \\
        --save-folder=s3://<bucket>/<run> \\
        --work-dir=/mnt/nvme/olmo-work

Validate the config on CPU first (no GPUs, no training):

    python src/scripts/train/OLMo3/OLMo3-370M-dolma2mix.py my-run --dry-run

Recipe provenance: the LR and global batch size follow the OLMo ladder formulas (same as
``allenai/OLMo-ladder`` and OLMo-core's ``estimate_lr``); at seq-len 4096 this yields
lr ~= 7.8e-4 and a global batch of 192 sequences (786,432 tokens). Override any of them from the
CLI; the defaults are the proven 370M ladder recipe, not guesses.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198
    load_path: Optional[str] = None


# --- Defaults (the proven 370M ladder recipe at seq-len 4096) --------------------------------
DEFAULT_DATA_CONFIG = "s3://edullm-datasets/olmo-150b-dolma2/configs/equal-weighting-config.yaml"
DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 192 * DEFAULT_SEQUENCE_LENGTH  # 786,432 tokens (192 sequences)
DEFAULT_RANK_MICROBATCH_SIZE = 4 * DEFAULT_SEQUENCE_LENGTH  # 16,384 tokens/rank; raise on B200
DEFAULT_TOKEN_BUDGET = 10_000_000_000  # train on a 10B slice of the mix (~1.35x Chinchilla)
DEFAULT_CHECKPOINT_TOKENS = 2_500_000_000  # ~4 checkpoints over a 10B run
DEFAULT_WARMUP_STEPS = 500  # ~= model_params / tokens_per_step for this recipe
DEFAULT_SEED = 6198


def ladder_lr(model_params: int) -> float:
    """OLMo ladder LR formula, with the /4 seq-4096 adjustment used by the 370M ladder run."""
    base = 0.0047 * (model_params / 108_000_000) ** (-1 / 3)
    return base / 4  # seq-len 4096 adjustment (matches allenai/OLMo-ladder 370M)


def build_config(opts, overrides: List[str]):
    tokenizer_config = TokenizerConfig.dolma2()

    # --- Model -------------------------------------------------------------------------------
    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    model_kwargs = {}
    if opts.attn_backend is not None:
        # OLMo-3 factories default to the flash_2 backend; override for other GPUs (e.g. flash_4 on B200).
        model_kwargs["attn_backend"] = AttentionBackendName(opts.attn_backend)
    model_config = factory(vocab_size=tokenizer_config.padded_vocab_size(), **model_kwargs)

    # --- Data: the dolma2 source mixture -----------------------------------------------------
    # `SourceMixtureList.from_yaml` accepts local paths or s3:// (via cached_path). `requested_tokens`
    # is how much of the mix to actually draw; keep it equal to the training token budget.
    source_list = SourceMixtureList.from_yaml(opts.data_config)
    src_mix = SourceMixtureDatasetConfig(
        source_list=source_list,
        requested_tokens=opts.token_budget,
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
    )
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix,
        tokenizer=tokenizer_config,
        sequence_length=opts.sequence_length,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
        num_workers=8,
    )

    # --- Optimization ------------------------------------------------------------------------
    lr = opts.lr if opts.lr is not None else ladder_lr(model_config.num_non_embedding_params)
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=opts.warmup_steps),
    )

    # --- Trainer -----------------------------------------------------------------------------
    # Checkpoint cadence is in steps; convert the requested per-checkpoint token count.
    tokens_per_step = opts.global_batch_size
    save_interval_steps = max(1, round(opts.checkpoint_tokens / tokens_per_step))
    ephemeral_save_interval = max(1, save_interval_steps // 4)

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(opts.token_budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=save_interval_steps,
                ephemeral_save_interval=ephemeral_save_interval,
                # Synchronous checkpointing: async needs a separate CPU process group that this
                # DLAMI's torch build fails to resolve in DCP gather_object ("Group ... is not
                # registered"). Sync save uses the default (registered) group and is fine here.
                save_async=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                entity="eduLLM",
                project="pretraining",
                enabled=opts.wandb,
                cancel_check_interval=10,
            ),
        )
    )

    # --- Comparable evals (for the data-mix trial) -------------------------------------------
    # To decide which data config is "better" you MUST evaluate both runs on the SAME yardstick.
    # Train loss across different mixes is NOT comparable. Pass a common held-out set via
    # --eval-data (same for both runs) and/or downstream tasks via --eval-tasks.
    if opts.eval_data:
        eval_paths = [p.strip() for p in opts.eval_data.split(",") if p.strip()]
        trainer_config = trainer_config.with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=eval_paths,
                    metadata=[{"label": "heldout-val"} for _ in eval_paths],
                    sequence_length=opts.sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=opts.eval_interval,
                eval_on_finish=True,
            ),
        )
    if opts.eval_tasks:
        tasks = [t.strip() for t in opts.eval_tasks.split(",") if t.strip()]
        trainer_config = trainer_config.with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks,
                tokenizer=tokenizer_config,
                eval_interval=opts.eval_interval,
                eval_on_finish=True,
            ),
        )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=opts.seed,
        load_path=opts.load_path,
    ).merge(overrides)

    return config


def train(config):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Auto-resume: if a checkpoint already exists in save_folder (e.g. from a previous B200
    # session synced to S3), pick up where we left off. Otherwise optionally warm-start from
    # --load-path (e.g. a pretraining checkpoint for continued pretraining).
    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(f"No checkpoint in save folder; loading from {config.load_path}")
        trainer.load_checkpoint(config.load_path, load_trainer_state=False)

    trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Pre-train a 370M OLMo-3 model on the dolma2 source mixture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_name", type=str, help="Name of the run (used for W&B + checkpoint dir)."
    )
    parser.add_argument("--model-factory", type=str, default="olmo3_370M")
    parser.add_argument(
        "--attn-backend",
        type=str,
        default=None,
        help="Override the OLMo-3 attention backend (e.g. flash_4 on B200). "
        "Default: the factory default (flash_2).",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=DEFAULT_DATA_CONFIG,
        help="SourceMixtureList YAML (local path or s3://).",
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=DEFAULT_GLOBAL_BATCH_SIZE,
        help="Global batch size IN TOKENS (must be a multiple of sequence-length).",
    )
    parser.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=DEFAULT_RANK_MICROBATCH_SIZE,
        help="Per-rank microbatch IN TOKENS. Raise on B200 (183GB) for throughput.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Total tokens to train on (also caps how much of the mix is drawn).",
    )
    parser.add_argument(
        "--checkpoint-tokens",
        type=int,
        default=DEFAULT_CHECKPOINT_TOKENS,
        help="Save a full checkpoint roughly every this many tokens.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override LR. Default: OLMo ladder formula (~7.8e-4 for 370M @ seq4096).",
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Comma-separated held-out .npy paths/globs for a COMMON validation set "
        "(use the SAME value for both trial runs to compare data configs fairly).",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        default=None,
        help="Comma-separated downstream task names (e.g. hellaswag,arc_easy). "
        "Note: near-random at ~70M/1B tokens; held-out val loss is more sensitive.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Steps between evals (also evaluated once at the end).",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default=None,
        help="Where to write checkpoints. Must be an s3:///gs:// path for real runs "
        "so checkpoints survive the (ephemeral) box.",
    )
    parser.add_argument(
        "--allow-local-save",
        action="store_true",
        help="Permit a non-remote --save-folder for real training (NOT durable; discouraged).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Local scratch dir for dataset index/cache (e.g. /mnt/nvme/olmo-work).",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Optional warm-start checkpoint if save-folder is empty.",
    )
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases logging (default: on; needs WANDB_API_KEY).",
    )
    parser.add_argument(
        "--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and print the config, then exit."
    )
    opts, overrides = parser.parse_known_args()

    if opts.work_dir is None:
        opts.work_dir = "/tmp/olmo-dataset-cache"
    if opts.global_batch_size % opts.sequence_length != 0:
        parser.error("--global-batch-size must be a multiple of --sequence-length")

    # Enforce durable (S3/GS) checkpoints for real training runs. The box is ephemeral, so a local
    # save folder would be lost. --dry-run and --allow-local-save opt out.
    remote_prefixes = ("s3://", "gs://", "http://", "https://")
    if opts.save_folder is None:
        opts.save_folder = f"/tmp/{opts.run_name}"
    if (
        not opts.dry_run
        and not opts.allow_local_save
        and not str(opts.save_folder).startswith(remote_prefixes)
    ):
        parser.error(
            "checkpoints must be saved to S3: pass --save-folder s3://<bucket>/<run> "
            "(or --allow-local-save to override for a throwaway local run)."
        )
    return opts, overrides


def main():
    opts, overrides = parse_args()

    if opts.dry_run:
        # Build without touching distributed/GPU so it runs on a laptop.
        config = build_config(opts, overrides)
        rich.print(config)
        print("\nDry run OK — config built. Remove --dry-run and launch under torchrun to train.")
        return

    prepare_training_environment()
    try:
        config = build_config(opts, overrides)
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
