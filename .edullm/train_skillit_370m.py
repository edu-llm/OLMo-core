#!/usr/bin/env python3
"""Train one immutable OLMo2-370M Skill-It arm."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_fs_local_rank, get_rank, get_world_size
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
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
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from production_contract.checkpoint import (
    assert_resume_fingerprint,
    checkpointer_kwargs_for_ladder,
    is_permanent_checkpoint_step,
    read_last_durable_step,
    write_run_fingerprint,
)
from production_contract.task_loss import TaskLossEvalCallback, validate_task_loss_result
from skillit_controller import SkillItController
from skillit_loader import (
    SEED,
    SEQUENCE_LENGTH,
    TOTAL_STEPS,
    WeightedDomainDataLoader,
    resolve_domain_datasets,
)
from skillit_math import RECIPE, arm_by_index, initial_weights

RANK_MICROBATCH_TOKENS = 32_768
CHECKPOINT_INTERVAL = 125


class ResumeAwareTaskLossEvalCallback(TaskLossEvalCallback):
    """Do not re-finalize step 0 when an explicit durable resume is loaded."""

    def pre_train(self) -> None:
        if int(self.step) == 0:
            super().pre_train()
            return
        step = int(self.step)
        if not is_permanent_checkpoint_step(step, self.total_steps, self.interval):
            raise RuntimeError(f"resume step {step} is not a permanent checkpoint")
        marker = read_last_durable_step(self.progress_dir or self.save_folder)
        if marker is None or int(marker["last_durable_step"]) != step:
            raise RuntimeError(f"resume step {step} lacks an exact durable-step marker")
        payload = validate_task_loss_result(self.results_dir / f"step{step}_task_loss.json")
        if int(payload.get("step", -1)) != step:
            raise RuntimeError(f"resume task-loss payload is stale for step {step}")
        self._completed.add(step)


def build_model_config() -> TransformerConfig:
    tokenizer = TokenizerConfig.dolma2()
    return TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size())


def build_train_module_config() -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_TOKENS,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts={"weight_decay": 0.0},
                )
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=24, alpha_f=0.1),
    )


def build_trainer_config(
    *,
    arm_index: int,
    run_name: str,
    save_folder: str,
    progress_dir: str,
    task_loss_dir: str,
    eval_script: str,
    task_loss_nproc: int,
    wandb_entity: Optional[str],
    wandb_mode: str,
    production: bool,
) -> TrainerConfig:
    arm = arm_by_index(arm_index)
    checkpointer_kwargs = checkpointer_kwargs_for_ladder(
        TOTAL_STEPS, CHECKPOINT_INTERVAL, save_async=False
    )
    # The shared task-loss callback finalizes in post_step. Save the true final
    # checkpoint in post_train_batch so it exists before final-step evaluation.
    checkpointer_kwargs["fixed_steps"] = [
        *checkpointer_kwargs["fixed_steps"],
        TOTAL_STEPS,
    ]
    config = TrainerConfig(
        save_folder=save_folder,
        save_overwrite=False,
        metrics_collect_interval=5,
        cancel_check_interval=5,
        max_duration=Duration.steps(TOTAL_STEPS),
    )
    config = config.with_callback(
        "checkpointer",
        # Explicitly no ephemeral saves: every retained state is scientifically durable.
        CheckpointerCallback(**checkpointer_kwargs),
    )
    config = config.with_callback(
        "wandb",
        WandBCallback(
            name=run_name,
            group=run_name,
            entity=wandb_entity,
            project=arm.wandb_project,
            enabled=wandb_mode != "disabled",
            config={
                "arm_index": arm.index,
                "arm_id": arm.arm_id,
                "a_mode": arm.a_mode,
                "methodology": RECIPE,
            },
        ),
    )
    config = config.with_callback("config_saver", ConfigSaverCallback())
    config = config.with_callback(
        "task_loss",
        ResumeAwareTaskLossEvalCallback(
            total_steps=TOTAL_STEPS,
            save_folder=save_folder,
            run_name=run_name,
            results_dir=task_loss_dir,
            eval_script=eval_script,
            interval=CHECKPOINT_INTERVAL,
            arm=arm.arm_id,
            progress_dir=progress_dir,
            method=f"skillit-{arm.a_mode}",
            task_loss_nproc=task_loss_nproc,
            production=production,
            wandb_mode=wandb_mode,
        ),
    )
    config = config.with_callback(
        "skillit",
        SkillItController(
            arm_id=arm.arm_id,
            a_mode=arm.a_mode,
            progress_dir=progress_dir,
            task_loss_dir=task_loss_dir,
            production=production,
            wandb_mode=wandb_mode,
        ),
    )
    return config


def _run_identity(arm_index: int) -> dict[str, Any]:
    arm = arm_by_index(arm_index)
    return {
        "schema": 1,
        "experiment": "skillit-370m",
        "arm_index": arm.index,
        "arm_id": arm.arm_id,
        "a_mode": arm.a_mode,
        "recipe": RECIPE,
    }


def validate_trainer_assembly(
    trainer: Any,
    loader: WeightedDomainDataLoader,
    *,
    production: bool,
) -> None:
    """Fail before training if the concrete loader/callback graph is incomplete."""
    if trainer.data_loader is not loader:
        raise RuntimeError("trainer is not attached to the weighted Skill-It loader")
    task_loss = trainer.callbacks.get("task_loss")
    controller = trainer.callbacks.get("skillit")
    if not isinstance(task_loss, ResumeAwareTaskLossEvalCallback):
        raise RuntimeError("trainer lacks the resume-aware 20-label callback")
    if not isinstance(controller, SkillItController):
        raise RuntimeError("trainer lacks the checkpoint-gated Skill-It controller")
    if task_loss.trainer is not trainer or controller.trainer is not trainer:
        raise RuntimeError("Skill-It callbacks are not attached to this trainer")
    if task_loss.priority <= controller.priority:
        raise RuntimeError("task-loss callback must run before the Skill-It controller")
    if production and loader.dp_world_size != 8:
        raise RuntimeError(
            f"production Skill-It requires exactly 8 data-parallel ranks, got "
            f"{loader.dp_world_size}"
        )
    expected_evaluator = Path(__file__).with_name("eval_task_loss_olmo_core.py").resolve()
    if task_loss.eval_script.resolve() != expected_evaluator:
        raise RuntimeError(
            f"production callback must use branch-local evaluator {expected_evaluator}"
        )


def fit_with_resume(trainer: Any, args: argparse.Namespace, identity: dict[str, Any]) -> None:
    """Perform the explicit fingerprint/resume handoff and enter ``Trainer.fit``."""
    if get_rank() == 0:
        if args.resume:
            assert_resume_fingerprint(args.load_path or args.save_folder, identity)
        write_run_fingerprint(args.save_folder, identity)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if args.resume:
        loaded = trainer.maybe_load_checkpoint(
            str(args.load_path) if args.load_path else str(args.save_folder)
        )
        if not loaded:
            raise RuntimeError("resume requested but no checkpoint was found")
    elif args.load_path:
        raise RuntimeError("--load-path is only valid with --resume")
    trainer.fit()


def train(args: argparse.Namespace) -> None:
    arm = arm_by_index(args.arm_index)
    production = not args.allow_local_only
    if production and args.wandb_mode != "online":
        raise SystemExit("production Skill-It runs require --wandb-mode=online")
    if not args.task_loss_evaluator.is_file():
        raise SystemExit(f"missing task-loss evaluator: {args.task_loss_evaluator}")
    branch_evaluator = Path(__file__).with_name("eval_task_loss_olmo_core.py").resolve()
    if args.task_loss_evaluator.resolve() != branch_evaluator:
        raise SystemExit(f"Skill-It requires its branch-local evaluator: {branch_evaluator}")

    seed_all(SEED)
    model = build_model_config().build(init_device="meta")
    train_module = build_train_module_config().build(model)
    datasets = resolve_domain_datasets(Path(args.work_dir) / "dataset-cache")
    loader = WeightedDomainDataLoader(
        datasets,
        work_dir=Path(args.work_dir) / "loader",
        weights=initial_weights(),
        dp_world_size=get_world_size(train_module.dp_process_group),
        dp_rank=get_rank(train_module.dp_process_group),
        fs_local_rank=get_fs_local_rank(),
    )
    trainer_config = build_trainer_config(
        arm_index=arm.index,
        run_name=args.run_name,
        save_folder=str(args.save_folder),
        progress_dir=str(args.progress_dir),
        task_loss_dir=str(args.task_loss_dir),
        eval_script=str(args.task_loss_evaluator),
        task_loss_nproc=get_world_size(train_module.dp_process_group),
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        production=production,
    )
    trainer = trainer_config.build(train_module, loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = {
        "model": build_model_config().as_config_dict(),
        "train_module": build_train_module_config().as_config_dict(),
        "trainer": trainer_config.as_config_dict(),
        "methodology": RECIPE,
    }

    validate_trainer_assembly(trainer, loader, production=production)
    fit_with_resume(trainer, args, _run_identity(arm.index))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--save-folder", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--task-loss-dir", type=Path, required=True)
    parser.add_argument(
        "--task-loss-evaluator",
        type=Path,
        default=Path(__file__).with_name("eval_task_loss_olmo_core.py"),
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--allow-local-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-path", type=Path)
    args = parser.parse_args(argv)
    if args.arm_index == 0 and "deriv" in args.run_name.lower():
        parser.error("arm-index 0 is probe but run name says deriv")
    if args.arm_index == 1 and "probe" in args.run_name.lower():
        parser.error("arm-index 1 is deriv but run name says probe")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    prepare_training_environment()
    try:
        train(args)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
