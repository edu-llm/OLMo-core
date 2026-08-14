#!/usr/bin/env python3
"""Train 370M RegMix batch-ablation arms with a 256 Ki W&B step axis."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
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
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback
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
from final_validation_wandb import (  # noqa: E402
    FinalValidationEvalCallback,
    ScaledWandBCallback,
)

MODEL_NAME = "olmo2_370M"
DATASET_ID = "pretrain/regmix-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
SEQUENCE_LENGTH = 2_048
WORLD_SIZE = 8
RANK_MICROBATCH_TOKENS = 16 * 1_024
TARGET_TOKENS = 10_000_000_000
WANDB_AXIS_TOKENS = 256 * 1_024
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-ladder"
DEFAULT_WANDB_GROUP = "hpo-ladder-batch-ablation"
LADDER_PEAK_LR = 7.78548e-4
LADDER_TERMINAL_LR_RATIO = 0.1
LADDER_WARMUP_FRACTION = 0.005
LADDER_SEED = 12_536
MIXLAW_PEAK_LR = 4e-4
MIXLAW_WARMUP_STEPS = 24
MIXLAW_SEED = 6_199
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class BatchArm:
    """One fixed 370M / ~10B RegMix batch-ablation contract."""

    name: str
    global_batch_tokens: int
    optimizer: str
    peak_lr: float
    seed: int
    warmup_fraction: float | None = None
    warmup_steps: int | None = None

    def __post_init__(self) -> None:
        if self.global_batch_tokens <= 0 or self.global_batch_tokens % WANDB_AXIS_TOKENS:
            raise FinalValidationConfigError(
                f"{self.name} global batch {self.global_batch_tokens} must be a positive "
                f"multiple of the {WANDB_AXIS_TOKENS}-token W&B axis"
            )
        if (self.global_batch_tokens // WORLD_SIZE) % RANK_MICROBATCH_TOKENS:
            raise FinalValidationConfigError(
                f"{self.name} rank batch is not a multiple of {RANK_MICROBATCH_TOKENS}"
            )
        if self.optimizer not in {"adamw", "skip_step_adamw"}:
            raise FinalValidationConfigError(f"{self.name} has unknown optimizer {self.optimizer}")
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise FinalValidationConfigError(
                f"{self.name} must set exactly one of warmup_fraction or warmup_steps"
            )

    @property
    def wandb_step_multiplier(self) -> int:
        return self.global_batch_tokens // WANDB_AXIS_TOKENS

    @property
    def total_steps(self) -> int:
        return TARGET_TOKENS // self.global_batch_tokens

    @property
    def train_tokens(self) -> int:
        return self.total_steps * self.global_batch_tokens

    @property
    def wandb_logged_steps(self) -> int:
        return self.total_steps * self.wandb_step_multiplier


ARMS: dict[str, BatchArm] = {
    "ladder-768ki": BatchArm(
        name="ladder-768ki",
        global_batch_tokens=768 * 1_024,
        optimizer="adamw",
        peak_lr=LADDER_PEAK_LR,
        warmup_fraction=LADDER_WARMUP_FRACTION,
        seed=LADDER_SEED,
    ),
    "ladder-4mi": BatchArm(
        name="ladder-4mi",
        global_batch_tokens=4 * 1_024 * 1_024,
        optimizer="adamw",
        peak_lr=LADDER_PEAK_LR,
        warmup_fraction=LADDER_WARMUP_FRACTION,
        seed=LADDER_SEED,
    ),
    "mixlaw-regmix-4mi": BatchArm(
        name="mixlaw-regmix-4mi",
        global_batch_tokens=4 * 1_024 * 1_024,
        optimizer="skip_step_adamw",
        peak_lr=MIXLAW_PEAK_LR,
        warmup_steps=MIXLAW_WARMUP_STEPS,
        seed=MIXLAW_SEED,
    ),
}


def _optimizer_config(arm: BatchArm) -> AdamWConfig | SkipStepAdamWConfig:
    group_overrides = [OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})]
    if arm.optimizer == "adamw":
        return AdamWConfig(
            lr=arm.peak_lr,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            group_overrides=group_overrides,
            fused=True,
        )
    return SkipStepAdamWConfig(
        lr=arm.peak_lr,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        group_overrides=group_overrides,
        foreach=True,
    )


def _scheduler_config(arm: BatchArm) -> CosWithWarmup:
    if arm.warmup_fraction is not None:
        return CosWithWarmup(warmup_fraction=arm.warmup_fraction, alpha_f=LADDER_TERMINAL_LR_RATIO)
    assert arm.warmup_steps is not None
    return CosWithWarmup(warmup=arm.warmup_steps, alpha_f=LADDER_TERMINAL_LR_RATIO)


def build_experiment_config(
    arm: BatchArm,
    corpus: ResolvedRegMix,
    *,
    save_folder: str,
    length_steps: int | None = None,
    work_dir: str | None = None,
    environ: Mapping[str, str] = os.environ,
) -> ExperimentConfig:
    """Build one fixed no-curriculum 370M batch-ablation arm for eight A100s."""

    total_steps = arm.total_steps if length_steps is None else int(length_steps)
    if total_steps < VALIDATION_POINTS - 1:
        raise FinalValidationConfigError(
            f"length_steps must be at least {VALIDATION_POINTS - 1} for endpoint evaluations"
        )
    tokenizer = TokenizerConfig.dolma2()
    checkpoints = validation_steps(total_steps, points=VALIDATION_POINTS)
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}
    run_name = environ.get("WANDB_NAME") or environ.get("EDULLM_RUN_ID") or arm.name
    project = environ.get("EDULLM_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    work_dir = work_dir or f"/tmp/olmo-core/{arm.name}"

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
        optim=_optimizer_config(arm),
        scheduler=_scheduler_config(arm),
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
            max_duration=Duration.steps(total_steps),
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
            ScaledWandBCallback(
                name=run_name,
                project=project,
                group=environ.get("WANDB_RUN_GROUP", DEFAULT_WANDB_GROUP),
                enabled=bool(project),
                cancel_check_interval=10,
                step_multiplier=arm.wandb_step_multiplier,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            FinalValidationEvalCallback(
                vector_name=arm.name,
                total_steps=total_steps,
                checkpoint_steps=checkpoints,
                save_folder=save_folder,
                run_name=run_name,
                work_dir=environ.get(
                    "EDULLM_EVAL_WORK_DIR", str(Path(work_dir) / "task-loss-eval")
                ),
                eval_script=EVAL_SCRIPT,
                nproc=WORLD_SIZE,
                wandb_step_multiplier=arm.wandb_step_multiplier,
            ),
        )
    )
    return ExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=arm.global_batch_tokens,
            seed=arm.seed,
            num_workers=4,
        ),
        train_module=train_module,
        trainer=trainer,
        init_seed=arm.seed,
    )


def scientific_identity(arm: BatchArm, config: ExperimentConfig) -> dict[str, Any]:
    """Return the fixed experiment identity persisted beside checkpoints and in W&B."""

    total_steps = int(config.trainer.max_duration.value)
    scheduler = {
        "name": "cos_with_warmup",
        "terminal_lr_ratio": LADDER_TERMINAL_LR_RATIO,
        "terminal_lr": arm.peak_lr * LADDER_TERMINAL_LR_RATIO,
    }
    if arm.warmup_fraction is not None:
        scheduler["warmup_fraction"] = arm.warmup_fraction
    else:
        scheduler["warmup_steps"] = arm.warmup_steps
    return {
        "schema_version": 1,
        "arm": arm.name,
        "control": "no_curriculum",
        "model": MODEL_NAME,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": arm.global_batch_tokens,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "world_size": WORLD_SIZE,
        "budget_tokens_requested": TARGET_TOKENS,
        "train_tokens": total_steps * arm.global_batch_tokens,
        "total_steps": total_steps,
        "wandb_axis_tokens": WANDB_AXIS_TOKENS,
        "wandb_step_multiplier": arm.wandb_step_multiplier,
        "wandb_logged_steps": total_steps * arm.wandb_step_multiplier,
        "checkpoint_steps": validation_steps(total_steps, points=VALIDATION_POINTS),
        "optimizer": {
            "name": "AdamW" if arm.optimizer == "adamw" else "SkipStepAdamW",
            "lr": arm.peak_lr,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "embedding_weight_decay": 0.0,
        },
        "scheduler": scheduler,
        "max_grad_norm": MAX_GRAD_NORM,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "seed": arm.seed,
    }


def torchrun_command(arm_name: str, length_steps: int | None = None) -> list[str]:
    """Build the fixed eight-rank launch command."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--train-worker",
        "--arm",
        arm_name,
    ]
    if length_steps is not None:
        command.extend(["--length-steps", str(length_steps)])
    return command


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], ResolvedRegMix] = resolve_regmix,
) -> int:
    """Launch or execute one fixed batch-ablation arm."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument(
        "--length-steps",
        type=int,
        help="smoke-only duration override; production omits this for the full 10B budget",
    )
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        arm = ARMS[args.arm]
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker:
            os.execv(sys.executable, torchrun_command(arm.name, args.length_steps))
        if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
            raise FinalValidationConfigError(f"worker requires WORLD_SIZE={WORLD_SIZE}")
        os.environ["WANDB_NAME"] = f"{run_id}-{arm.name}"
        config = build_experiment_config(
            arm,
            resolver(),
            save_folder=checkpoint_dir,
            length_steps=args.length_steps,
            environ=os.environ,
        )
        run_training(config, scientific_identity(arm, config))
    except FinalValidationConfigError as exc:
        print(f"[hpo-ladder-batch] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
