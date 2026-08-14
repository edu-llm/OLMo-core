#!/usr/bin/env python3
"""Train the no-curriculum OLMo2-370M RegMix control with OLMo-ladder hyperparameters."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_validation import (  # noqa: E402
    EVAL_SCRIPT,
    FinalValidationConfigError,
    ResolvedRegMix,
    platform_values,
    resolve_regmix,
    run_training,
    validation_steps,
)
from final_validation_wandb import FinalValidationEvalCallback  # noqa: E402

ARM_NAME = "olmo-ladder-control"
MODEL_NAME = "olmo2_370M"
DATASET_ID = "pretrain/regmix-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 256 * 1_024
WORLD_SIZE = 8
RANK_MICROBATCH_TOKENS = 16 * 1_024
TARGET_TOKENS = 10_000_000_000
TOTAL_STEPS = TARGET_TOKENS // GLOBAL_BATCH_TOKENS
TRAIN_TOKENS = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
PEAK_LR = 7.78548e-4
TERMINAL_LR_RATIO = 0.1
WARMUP_FRACTION = 0.005
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
SEED = 12_536
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-validation"
DEFAULT_WANDB_GROUP = "hpo-validation-olmo2-370m-ladder-control"


def build_experiment_config(
    corpus: ResolvedRegMix,
    *,
    save_folder: str,
    length_steps: int = TOTAL_STEPS,
    work_dir: str = "/tmp/olmo-core/olmo-ladder-control",
    environ: Mapping[str, str] = os.environ,
) -> ExperimentConfig:
    """Build the fixed no-curriculum 370M control for eight A100s."""

    if length_steps < VALIDATION_POINTS - 1:
        raise FinalValidationConfigError(
            f"length_steps must be at least {VALIDATION_POINTS - 1} for endpoint evaluations"
        )
    tokenizer = TokenizerConfig.dolma2()
    checkpoints = validation_steps(length_steps, points=VALIDATION_POINTS)
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}
    run_name = environ.get("WANDB_NAME") or environ.get("EDULLM_RUN_ID") or ARM_NAME
    project = environ.get("EDULLM_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)

    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        dtype=corpus.dtype,
        work_dir=work_dir,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_TOKENS,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=PEAK_LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
            fused=True,
        ),
        scheduler=CosWithWarmup(
            warmup_fraction=WARMUP_FRACTION,
            alpha_f=TERMINAL_LR_RATIO,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=MAX_GRAD_NORM,
    )
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            work_dir=work_dir,
            max_duration=Duration.steps(length_steps),
            metrics_collect_interval=5,
            cancel_check_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
                fixed_steps=checkpoints[1:],
                ephemeral_save_interval=None,
                pre_train_checkpoint=not skip_pre_train,
                save_async=True,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project=project,
                group=environ.get("WANDB_RUN_GROUP", DEFAULT_WANDB_GROUP),
                enabled=bool(project),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            FinalValidationEvalCallback(
                vector_name=ARM_NAME,
                total_steps=length_steps,
                checkpoint_steps=checkpoints,
                save_folder=save_folder,
                run_name=run_name,
                work_dir=environ.get(
                    "EDULLM_EVAL_WORK_DIR", str(Path(work_dir) / "task-loss-eval")
                ),
                eval_script=EVAL_SCRIPT,
                nproc=WORLD_SIZE,
            ),
        )
    )
    return ExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=GLOBAL_BATCH_TOKENS,
            seed=SEED,
            num_workers=4,
        ),
        train_module=train_module,
        trainer=trainer,
        init_seed=SEED,
    )


def scientific_identity(config: ExperimentConfig) -> dict[str, Any]:
    """Return the fixed experiment identity persisted beside checkpoints and in W&B."""

    total_steps = int(config.trainer.max_duration.value)
    return {
        "schema_version": 1,
        "arm": ARM_NAME,
        "control": "no_curriculum",
        "model": MODEL_NAME,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "world_size": WORLD_SIZE,
        "budget_tokens_requested": TARGET_TOKENS,
        "train_tokens": total_steps * GLOBAL_BATCH_TOKENS,
        "total_steps": total_steps,
        "checkpoint_steps": validation_steps(total_steps, points=VALIDATION_POINTS),
        "optimizer": {
            "name": "AdamW",
            "lr": PEAK_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "embedding_weight_decay": 0.0,
        },
        "scheduler": {
            "name": "cos_with_warmup",
            "warmup_fraction": WARMUP_FRACTION,
            "terminal_lr_ratio": TERMINAL_LR_RATIO,
            "terminal_lr": PEAK_LR * TERMINAL_LR_RATIO,
        },
        "max_grad_norm": MAX_GRAD_NORM,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "seed": SEED,
    }


def torchrun_command(length_steps: int | None = None) -> list[str]:
    """Build the fixed eight-rank launch command."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--train-worker",
    ]
    if length_steps is not None:
        command.extend(["--length-steps", str(length_steps)])
    return command


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], ResolvedRegMix] = resolve_regmix,
) -> int:
    """Launch or execute the fixed control arm."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length-steps",
        type=int,
        help="smoke-only duration override; production omits this for the full 10B budget",
    )
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker:
            os.execv(sys.executable, torchrun_command(args.length_steps))
        if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
            raise FinalValidationConfigError(f"worker requires WORLD_SIZE={WORLD_SIZE}")
        os.environ["WANDB_NAME"] = f"{run_id}-{ARM_NAME}"
        config = build_experiment_config(
            resolver(),
            save_folder=checkpoint_dir,
            length_steps=args.length_steps or TOTAL_STEPS,
            environ=os.environ,
        )
        run_training(config, scientific_identity(config))
    except FinalValidationConfigError as exc:
        print(f"[olmo-ladder-control] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
