#!/usr/bin/env python3
"""Run arm 9 with the curriculum HPO winner at the hpo-moe 128 Ki contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO_DIR", "/workspace/OLMo-core"))
for path in (REPO / "src", REPO / ".edullm", REPO / ".edullm" / "runpod"):
    sys.path.insert(0, str(path))

import curriculum_loader  # noqa: E402
import curriculum_pacing  # noqa: E402
import entrypoint as runpod_entrypoint  # noqa: E402
from olmo_core.optim import SchedulerUnits, WSD  # noqa: E402

c = runpod_entrypoint.curriculum

SOURCE_COMMIT = "92a9ed9d7b983bcb607b331532373f029e7966ad"
ORIGINAL_GLOBAL_BATCH = 4_194_304
ORIGINAL_TOTAL_STEPS = 2_384
ORIGINAL_CHECKPOINT_INTERVAL = 125
TARGET_TOKENS = ORIGINAL_GLOBAL_BATCH * ORIGINAL_TOTAL_STEPS
GLOBAL_BATCH_TOKENS = 262_144
TOTAL_STEPS = TARGET_TOKENS // GLOBAL_BATCH_TOKENS
PACING_STEP_SCALE = ORIGINAL_GLOBAL_BATCH // GLOBAL_BATCH_TOKENS
WANDB_STEP_SCALE = 2

HPS = {
    "lr": 6.550313e-4,
    "weight_decay": 0.01156898,
    "beta2_gap": 0.02109002,
    "eps": 1.419878e-7,
    "warmup_fraction": 0.07599508,
    "decay_fraction": 0.17565491,
    "terminal_lr_ratio": 0.02764273,
    "max_grad_norm": 1.24295101,
}

if TARGET_TOKENS % GLOBAL_BATCH_TOKENS:
    raise SystemExit("128 Ki global batch does not divide the production token budget")
if GLOBAL_BATCH_TOKENS % (8 * c.SEQUENCE_LENGTH):
    raise SystemExit("global batch must divide evenly across eight sequence-aligned ranks")

c.GLOBAL_BATCH_TOKENS = GLOBAL_BATCH_TOKENS
c.TOTAL_STEPS = TOTAL_STEPS
c.CHECKPOINT_INTERVAL = (
    ORIGINAL_CHECKPOINT_INTERVAL * ORIGINAL_GLOBAL_BATCH // GLOBAL_BATCH_TOKENS
)
c.PEAK_LR = HPS["lr"]
c.WARMUP_STEPS = round(TOTAL_STEPS * HPS["warmup_fraction"])
c.LR_ALPHA_F = HPS["terminal_lr_ratio"]
c.WANDB_PROJECT_NAME = "hpo-moe"
c.WANDB_PROJECT_NAMES = frozenset(("hpo-moe",))

_original_train_module_config = c.train_module_config


def optimized_train_module_config(rank_microbatch_tokens: int = c.RANK_MICROBATCH_TOKENS):
    config = _original_train_module_config(rank_microbatch_tokens)
    config.optim.lr = HPS["lr"]
    config.optim.weight_decay = HPS["weight_decay"]
    config.optim.betas = (0.9, 1.0 - HPS["beta2_gap"])
    config.optim.eps = HPS["eps"]
    config.scheduler = WSD(
        units=SchedulerUnits.tokens,
        warmup_fraction=HPS["warmup_fraction"],
        decay_fraction=HPS["decay_fraction"],
        decay_min_lr=HPS["terminal_lr_ratio"] * HPS["lr"],
    )
    config.max_grad_norm = HPS["max_grad_norm"]
    return config


c.train_module_config = optimized_train_module_config

# Preserve the original arm-9 token boundaries after reducing the optimizer-step batch.
_original_pool_for_step = curriculum_loader.pool_for_step


def token_aligned_pool_for_step(step: int, size: int, pacing: str):
    reference_step = int(step) // PACING_STEP_SCALE
    return _original_pool_for_step(reference_step, size, pacing)


curriculum_loader.pool_for_step = token_aligned_pool_for_step

_original_identity = c.scientific_identity


def optimized_identity(**kwargs):
    identity = _original_identity(**kwargs)
    original_boundaries = curriculum_pacing.WARMUP_QUADRATIC_SEGMENT_BOUNDARIES
    identity.update(
        {
            "experiment": "curriculum-quadratic-mtld-hpo-winner-256ki",
            "source_commit": SOURCE_COMMIT,
            "hpo_probe": {
                "arm": "curriculum_quadratic_mtld",
                "run_id": "7f74348409e054561b12348d0f5a815b",
                "trial_id": "t4_0",
                "hyperparameters": HPS,
            },
            "optimizer": {
                "name": "SkipStepAdamW",
                "lr": HPS["lr"],
                "weight_decay": HPS["weight_decay"],
                "betas": [0.9, 1.0 - HPS["beta2_gap"]],
                "eps": HPS["eps"],
                "max_grad_norm": HPS["max_grad_norm"],
            },
            "scheduler": {
                "name": "WSD",
                "units": "tokens",
                "warmup_fraction": HPS["warmup_fraction"],
                "decay_fraction": HPS["decay_fraction"],
                "terminal_lr_ratio": HPS["terminal_lr_ratio"],
            },
            "target_tokens": TARGET_TOKENS,
            "actual_training_tokens": TOTAL_STEPS * GLOBAL_BATCH_TOKENS,
            "optimizer_step_count": TOTAL_STEPS,
            "wandb_total_steps": TOTAL_STEPS * WANDB_STEP_SCALE,
            "curriculum_reference_global_batch_tokens": ORIGINAL_GLOBAL_BATCH,
            "curriculum_pacing_step_scale": PACING_STEP_SCALE,
            "curriculum_pacing_boundaries": [
                step * PACING_STEP_SCALE for step in original_boundaries
            ],
            "wandb_step_scale": WANDB_STEP_SCALE,
            "wandb_checkpoint_steps": [
                step * WANDB_STEP_SCALE for step in c.checkpoint_steps(TOTAL_STEPS)
            ],
        }
    )
    return identity


c.scientific_identity = optimized_identity

_original_wandb_callback = c.WandBCallback


class ScaledStepWandBCallback(_original_wandb_callback):
    def log_metrics(self, step: int, metrics: dict[str, float]):
        return super().log_metrics(step * WANDB_STEP_SCALE, metrics)


c.WandBCallback = ScaledStepWandBCallback

_original_wandb_log_eval = c.wandb_artifacts.wandb_log_eval


def scaled_wandb_log_eval(run, payload, *, step, **kwargs):
    return _original_wandb_log_eval(
        run,
        payload,
        step=int(step) * WANDB_STEP_SCALE,
        **kwargs,
    )


c.wandb_artifacts.wandb_log_eval = scaled_wandb_log_eval

_original_checkpointer_options = c.checkpoint_contract.checkpointer_kwargs_for_ladder


def optimized_checkpointer_options(total_steps, interval=c.CHECKPOINT_INTERVAL, **kwargs):
    del interval
    options = _original_checkpointer_options(
        total_steps,
        interval=c.CHECKPOINT_INTERVAL,
        **kwargs,
    )
    options["max_checkpoints"] = 1
    return options


c.checkpoint_contract.checkpointer_kwargs_for_ladder = optimized_checkpointer_options

if os.environ.get("HPO_ARM9_INSPECT") == "1":
    config = optimized_train_module_config(32_768)
    print(
        json.dumps(
            {
                "global_batch_tokens": c.GLOBAL_BATCH_TOKENS,
                "rank_microbatch_tokens": config.rank_microbatch_size,
                "total_steps": c.TOTAL_STEPS,
                "target_tokens": TARGET_TOKENS,
                "checkpoint_interval": c.CHECKPOINT_INTERVAL,
                "wandb_step_scale": WANDB_STEP_SCALE,
                "pacing_boundaries": [
                    step * PACING_STEP_SCALE
                    for step in curriculum_pacing.WARMUP_QUADRATIC_SEGMENT_BOUNDARIES
                ],
                "project": c.WANDB_PROJECT_NAME,
                "optimizer": {
                    "lr": config.optim.lr,
                    "weight_decay": config.optim.weight_decay,
                    "betas": config.optim.betas,
                    "eps": config.optim.eps,
                    "max_grad_norm": config.max_grad_norm,
                },
                "scheduler": {
                    "name": type(config.scheduler).__name__,
                    "units": str(config.scheduler.units),
                    "warmup_fraction": config.scheduler.warmup_fraction,
                    "decay_fraction": config.scheduler.decay_fraction,
                    "decay_min_lr": config.scheduler.decay_min_lr,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0)

raise SystemExit(runpod_entrypoint.main())
