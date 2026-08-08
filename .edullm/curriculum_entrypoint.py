#!/usr/bin/env python3
"""Run one README-faithful curriculum arm on OLMo-core's public APIs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import TransformerDataParallelWrappingStrategy
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

from curriculum_data import (
    PublishedInputError,
    ResolvedInput,
    load_order,
    resolve_published_inputs,
    stage_input,
)
from curriculum_loader import CurriculumDataLoader, ParentChunkDataset
from curriculum_model import MODEL_IDENTITY, build_model_config
from curriculum_pacing import DIFFICULTY_METRICS, ORDER_GROUPS, PACING_NAMES
from curriculum_ema import EMA_WANDB_STEP, build_ema_checkpoint, finalize_ema_production
from production_contract import checkpoint as checkpoint_contract
from production_contract import task_loss
from production_contract import wandb_artifacts

RECIPE_PATH = Path(__file__).with_name("curriculum_recipe.json")
PACKAGED_TASK_LOSS_SCRIPT = Path(__file__).with_name("task_loss") / "eval_task_loss_olmo_core.py"
PACKAGED_LADDER_CONFIG = Path(__file__).with_name("task_loss") / "ladder_base_config.yaml"
PARENT_DATASET_ID = "pretrain/regmix-10b"
PARENT_VERSION = "v1"
PARENT_MANIFEST_SHA256 = "a24992f53dc4a900bacf8fa571d77e343fd28ffa9054c14b93d54204b0a38cb4"
ORDER_DATASET_ID = "curriculum/regmix-370m"
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_TOKENS = 4_194_304
RANK_MICROBATCH_TOKENS = 32_768
TOTAL_STEPS = 2384
SEED = 42
PEAK_LR = 4e-4
WARMUP_STEPS = 24
LR_ALPHA_F = 1.0
CHECKPOINT_INTERVAL = 125
EMA_STEPS = (2000, 2125, 2250, 2384)
EMA_ALPHA = 0.8
WANDB_PROJECT_NAME = "curriculum"
WANDB_PROJECT_EXT_NAME = "curriculum-ext"
WANDB_PROJECT_NAMES = frozenset((WANDB_PROJECT_NAME, WANDB_PROJECT_EXT_NAME))


def _wandb_project_name() -> str:
    return (
        os.environ.get("EDULLM_WANDB_PROJECT")
        or os.environ.get("WANDB_PROJECT")
        or WANDB_PROJECT_NAME
    )


def _reset_olmo_world_mesh() -> None:
    try:
        import olmo_core.distributed.parallel as parallel

        if getattr(parallel, "_WORLD_MESH", None) is not None:
            parallel._WORLD_MESH = None  # type: ignore[attr-defined]
    except Exception:
        pass


class CurriculumConfigError(RuntimeError):
    """The selected arm or runtime violates the fixed experiment contract."""


@dataclass(frozen=True)
class Arm:
    index: int
    name: str
    pacing: str
    metric: str | None
    order_group: str | None

    @property
    def wandb_project(self) -> str:
        return WANDB_PROJECT_NAME


def load_recipe(path: Path = RECIPE_PATH) -> tuple[Arm, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent") or {}
    if parent != {
        "dataset_id": PARENT_DATASET_ID,
        "version": PARENT_VERSION,
        "manifest_sha256": PARENT_MANIFEST_SHA256,
    }:
        raise CurriculumConfigError("recipe parent pin differs from the approved parent")
    training = payload.get("training") or {}
    fixed = {
        "steps": TOTAL_STEPS,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "seed": SEED,
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "alpha_f": LR_ALPHA_F,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "ema_steps": list(EMA_STEPS),
        "ema_alpha": EMA_ALPHA,
        "ema_wandb_step": EMA_WANDB_STEP,
    }
    if training != fixed:
        raise CurriculumConfigError("recipe training fields differ from the approved recipe")
    arms = tuple(
        Arm(
            index=int(item["index"]),
            name=str(item["name"]),
            pacing=str(item["pacing"]),
            metric=item.get("metric"),
            order_group=item.get("order_group"),
        )
        for item in payload.get("arms") or []
    )
    expected_arms = (
        (0, "linear10-flesch", "linear_n10", "flesch", "flesch"),
        (1, "linear10-mtld", "linear_n10", "mtld", "mtld"),
        (2, "linear10-learn", "linear_n10", "learnability", "learnability"),
        (3, "warmup-flesch", "warmup_1000", "flesch", "flesch"),
        (4, "interleave-flesch", "interleave_i10_linear", "flesch", "flesch"),
        (5, "control", "control", None, None),
        (6, "quadratic10-mtld", "quadratic_n10", "mtld", "mtld"),
    )
    if tuple(
        (arm.index, arm.name, arm.pacing, arm.metric, arm.order_group) for arm in arms
    ) != expected_arms:
        raise CurriculumConfigError("recipe arms differ from the approved seven-arm matrix")
    for arm in arms:
        if arm.pacing not in PACING_NAMES:
            raise CurriculumConfigError(f"{arm.name}: unknown pacing {arm.pacing}")
        if arm.pacing == "control":
            if arm.metric is not None or arm.order_group is not None:
                raise CurriculumConfigError("control must not select a metric or order")
        elif arm.metric not in DIFFICULTY_METRICS or arm.order_group != ORDER_GROUPS[arm.metric]:
            raise CurriculumConfigError(f"{arm.name}: metric/order mapping is invalid")
    return arms


ARMS = load_recipe()


def production_steps(length_tokens: int | None) -> int:
    if length_tokens is None:
        return TOTAL_STEPS
    if length_tokens <= 0 or length_tokens % GLOBAL_BATCH_TOKENS:
        raise CurriculumConfigError(
            f"length tokens must be a positive multiple of {GLOBAL_BATCH_TOKENS}"
        )
    steps = length_tokens // GLOBAL_BATCH_TOKENS
    if steps > TOTAL_STEPS:
        raise CurriculumConfigError("benchmark override cannot exceed the production budget")
    return steps


def checkpoint_steps(total_steps: int) -> list[int]:
    return checkpoint_contract.permanent_checkpoint_steps(total_steps, CHECKPOINT_INTERVAL)


def train_module_config(
    rank_microbatch_tokens: int = RANK_MICROBATCH_TOKENS,
) -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=int(rank_microbatch_tokens),
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=PEAK_LR,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS, alpha_f=LR_ALPHA_F),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_train_module(
    rank_microbatch_tokens: int = RANK_MICROBATCH_TOKENS,
) -> Any:
    model = build_model_config().build(init_device="meta")
    return train_module_config(rank_microbatch_tokens).build(model)


def _broadcast_failure(error: str | None) -> None:
    errors = [error]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(errors, src=0)
    if errors[0] is not None:
        raise checkpoint_contract.CheckpointContractError(errors[0])


def assert_distributed_runtime(expected_world_size: int) -> None:
    """Verify torchrun supplied one process for every requested local GPU."""
    world_size = get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != expected_world_size or local_world_size != expected_world_size:
        raise CurriculumConfigError(
            f"torchrun topology differs from --nproc={expected_world_size}: "
            f"WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    visible_devices = torch.cuda.device_count()
    if visible_devices < local_world_size:
        raise CurriculumConfigError(
            f"torchrun started {local_world_size} local ranks but only "
            f"{visible_devices} CUDA devices are visible"
        )


class CurriculumCheckpointCallback(Callback):
    """Checkpoint → all-rank eval/reload → awaited W&B artifacts → marker."""

    priority = 0

    def __init__(
        self,
        *,
        arm: Arm,
        total_steps: int,
        save_folder: Path,
        progress_dir: Path,
        task_loss_dir: Path,
        eval_script: Path | None,
        task_loss_nproc: int,
        production: bool,
        wandb_mode: str,
        run_name: str,
        fingerprint_path: Path,
        module_builder: Callable[[], Any] = build_train_module,
    ) -> None:
        self.arm = arm
        self.total_steps = int(total_steps)
        self.save_folder = save_folder
        self.progress_dir = progress_dir
        self.task_loss_dir = task_loss_dir
        self.eval_script = eval_script
        self.task_loss_nproc = int(task_loss_nproc)
        self.production = bool(production)
        self.wandb_mode = wandb_mode
        self.run_name = run_name
        self.fingerprint_path = fingerprint_path
        self.module_builder = module_builder
        self._completed: set[int] = set()
        self._ema_completed = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_steps": sorted(self._completed),
            "ema_completed": self._ema_completed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._completed = {int(step) for step in state_dict.get("completed_steps", [])}
        self._ema_completed = bool(state_dict.get("ema_completed", False))

    def _release(self) -> None:
        old_module = self.trainer.train_module
        old_module._trainer = None
        self.trainer.train_module = None  # type: ignore[assignment]
        del old_module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reload(self, checkpoint: Path) -> Any:
        _reset_olmo_world_mesh()
        restored = self.module_builder()
        self.trainer.train_module = restored
        restored._attach_trainer(self.trainer)
        self.trainer.load_checkpoint(checkpoint, load_trainer_state=True, load_optim_state=True)
        return restored

    def _evaluate(self, checkpoint: Path, output: Path, name: str) -> dict[str, Any] | None:
        if get_rank() != 0:
            return None
        task_loss.trigger_task_loss_eval(
            checkpoint,
            run_name=name,
            out_path=output,
            eval_script=self.eval_script,
            nproc=self.task_loss_nproc,
        )
        return task_loss.validate_task_loss_result(output)

    @staticmethod
    def _already_evaluated(_checkpoint: Path, *, out_path: Path, **_kwargs: Any) -> None:
        task_loss.validate_task_loss_result(out_path)

    def _method_name(self) -> str:
        return (
            "plain_ce"
            if self.arm.pacing == "control"
            else f"curriculum:{self.arm.pacing}"
        )

    def _finalize_ema(self) -> None:
        """Build the post-hoc EMA, eval it, and publish it as the final model.

        Intermediate permanent steps never upload a model artifact. After the
        true final training step, this merges EMA_STEPS, runs the full 20-label
        task-loss suite on the merged weights, and uploads that EMA checkpoint
        plus its eval to the same W&B run at step EMA_WANDB_STEP (2385).
        """
        if self._ema_completed:
            return
        if self.eval_script is None:
            if self.production:
                raise checkpoint_contract.CheckpointContractError(
                    "production runs require automatic EMA finalization with task loss"
                )
            return
        if not all(step in self._completed for step in EMA_STEPS):
            return

        ema_dir = self.save_folder / "step2384-ema"
        if get_rank() == 0:
            build_ema_checkpoint(
                self.save_folder,
                arm=self.arm.name,
                output_dir=ema_dir,
                overwrite=True,
            )
        barrier()

        self._release()
        barrier()

        failure: str | None = None
        if get_rank() == 0:
            try:
                finalize_ema_production(
                    checkpoints_root=self.save_folder,
                    arm=self.arm.name,
                    run_name=self.run_name,
                    task_loss_dir=self.task_loss_dir,
                    eval_script=self.eval_script,
                    task_loss_nproc=self.task_loss_nproc,
                    progress_dir=self.progress_dir,
                    fingerprint_path=self.fingerprint_path,
                    wandb_run=wandb_artifacts.wandb_run_from_trainer(self.trainer),
                    wandb_mode=self.wandb_mode,
                    production=self.production,
                    method=self._method_name(),
                    ema_dir=ema_dir,
                    evaluate=self._evaluate,
                )
                self._ema_completed = True
            except BaseException as exc:  # noqa: BLE001
                failure = f"EMA finalization failed: {type(exc).__name__}: {exc}"
        _broadcast_failure(failure)
        barrier()

    def _finalize(self, step: int) -> None:
        step = int(step)
        if step in self._completed or step not in checkpoint_steps(self.total_steps):
            return
        checkpoint = self.save_folder / f"step{step}"
        output = self.task_loss_dir / f"step{step}_task_loss.json"
        payload: dict[str, Any] | None = None
        if self.eval_script is not None:
            _, payload = task_loss.pause_eval_reload_distributed(
                checkpoint,
                output,
                f"{self.run_name}-step{step}",
                evaluate=self._evaluate,
                release_train_state=self._release,
                reload_train_state=lambda: self._reload(checkpoint),
                strict=True,
            )
        elif self.production:
            raise checkpoint_contract.CheckpointContractError(
                "production checkpoints require the synchronous 20-label evaluator"
            )

        failure: str | None = None
        if get_rank() == 0:
            try:
                checkpoint_contract.finalize_permanent_checkpoint(
                    arm=self.arm.name,
                    checkpoint_dir=checkpoint,
                    step=step,
                    run_name=self.run_name,
                    task_loss_dir=self.task_loss_dir,
                    task_loss_enabled=payload is not None,
                    eval_script=self.eval_script,
                    task_loss_nproc=self.task_loss_nproc,
                    progress_dir=self.progress_dir,
                    fingerprint_path=self.fingerprint_path,
                    method=self._method_name(),
                    wandb_run=wandb_artifacts.wandb_run_from_trainer(self.trainer),
                    wandb_mode=self.wandb_mode,
                    production=self.production,
                    upload_checkpoint=False,
                    run_evaluator=self._already_evaluated,
                )
            except BaseException as exc:  # noqa: BLE001
                failure = f"permanent checkpoint step {step} failed: {type(exc).__name__}: {exc}"
        _broadcast_failure(failure)
        self._completed.add(step)
        barrier()
        if step == self.total_steps:
            self._finalize_ema()

    def pre_train(self) -> None:
        self._finalize(0)

    def post_train_batch(self) -> None:
        self._finalize(self.step)

    def post_train(self) -> None:
        self._finalize(self.step)


def _input_from_identity(identity: Mapping[str, Any]) -> ResolvedInput:
    return ResolvedInput(
        dataset_id=str(identity["dataset_id"]),
        version=str(identity["version"]),
        group=str(identity["group"]),
        profile=str(identity["profile"]),
        manifest_sha256=str(identity["manifest_sha256"]),
        paths=tuple(identity["paths"]),
        numpy_dtype=str(identity["numpy_dtype"]),
        header_bytes=int(identity["header_bytes"]),
    )


def resolve_and_stage(
    *,
    arm: Arm,
    order_version: str | None,
    cache_dir: Path,
) -> tuple[ResolvedInput, tuple[Path, ...], ResolvedInput | None, tuple[Path, ...]]:
    sidecar = cache_dir / f"resolved-{arm.name}.json"
    if get_rank() == 0:
        parent, order = resolve_published_inputs(
            parent_dataset_id=PARENT_DATASET_ID,
            parent_version=PARENT_VERSION,
            parent_manifest_sha256=PARENT_MANIFEST_SHA256,
            order_dataset_id=None if arm.pacing == "control" else ORDER_DATASET_ID,
            order_version=None if arm.pacing == "control" else order_version,
            order_group=arm.order_group,
        )
        parent_paths = stage_input(parent, cache_dir)
        order_paths = stage_input(order, cache_dir) if order is not None else ()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {
                    "parent": parent.identity,
                    "parent_paths": [str(path) for path in parent_paths],
                    "order": order.identity if order is not None else None,
                    "order_paths": [str(path) for path in order_paths],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    barrier()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    parent = _input_from_identity(payload["parent"])
    order = _input_from_identity(payload["order"]) if payload["order"] else None
    return (
        parent,
        tuple(Path(path) for path in payload["parent_paths"]),
        order,
        tuple(Path(path) for path in payload["order_paths"]),
    )


def scientific_identity(
    *,
    arm: Arm,
    total_steps: int,
    rank_microbatch_tokens: int,
    parent: ResolvedInput,
    order: ResolvedInput | None,
) -> dict[str, Any]:
    return {
        "family": "curriculum",
        "arm": arm.name,
        "pacing": arm.pacing,
        "difficulty_metric": arm.metric,
        "parent": parent.identity,
        "order": order.identity if order is not None else None,
        "model": MODEL_IDENTITY,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": int(rank_microbatch_tokens),
        "total_steps": int(total_steps),
        "seed": SEED,
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "alpha_f": LR_ALPHA_F,
        "checkpoint_steps": checkpoint_steps(total_steps),
        "ema_steps": list(EMA_STEPS),
        "ema_alpha": EMA_ALPHA,
        "ema_wandb_step": EMA_WANDB_STEP,
    }


def _validate_runtime(args: argparse.Namespace, arm: Arm) -> tuple[Path, Path, Path]:
    project = _wandb_project_name()
    if project not in WANDB_PROJECT_NAMES:
        raise CurriculumConfigError(
            f"W&B project must be one of {sorted(WANDB_PROJECT_NAMES)!r}, got {project!r}"
        )
    if os.environ.get("EDULLM_DATASET_ID") not in (None, "", PARENT_DATASET_ID):
        raise CurriculumConfigError(f"platform dataset must be {PARENT_DATASET_ID}")
    if os.environ.get("EDULLM_DATASET_VERSION") not in (None, "", PARENT_VERSION):
        raise CurriculumConfigError(f"platform dataset version must be {PARENT_VERSION}")
    if args.fresh == bool(args.load_path):
        raise CurriculumConfigError("choose exactly one of --fresh or --load-path")
    production = not args.local_smoke
    if production and args.wandb_mode != "online":
        raise CurriculumConfigError("production requires --wandb-mode online")
    if production and not os.environ.get("WANDB_API_KEY"):
        raise CurriculumConfigError("production requires WANDB_API_KEY")
    if production and args.no_task_loss:
        raise CurriculumConfigError("production cannot disable task-loss evaluation")
    if production and not args.task_loss_eval_script.is_file():
        raise CurriculumConfigError(
            f"packaged task-loss evaluator not found: {args.task_loss_eval_script}"
        )
    if production and not args.ladder_base_config.is_file():
        raise CurriculumConfigError(f"packaged ladder config not found: {args.ladder_base_config}")
    if args.local_smoke and args.task_loss_eval_script is None:
        args.no_task_loss = True

    run_root = Path(args.run_dir)
    save_folder = Path(args.save_folder or run_root / "checkpoints")
    progress_dir = Path(args.progress_dir or run_root / "progress")
    cache_dir = Path(args.cache_dir or run_root / "cache")
    for path in (save_folder, progress_dir, cache_dir):
        if "://" in str(path):
            raise CurriculumConfigError("curriculum runtime paths must be job-local scratch")
        path.mkdir(parents=True, exist_ok=True)
    return save_folder, progress_dir, cache_dir


def run_worker(args: argparse.Namespace) -> None:
    arm = ARMS[args.arm_index]
    save_folder, progress_dir, cache_dir = _validate_runtime(args, arm)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["TASK_LOSS_EVAL_SCRIPT"] = str(args.task_loss_eval_script)
    os.environ["LADDER_BASE_CONFIG"] = str(args.ladder_base_config)
    if not args.local_smoke:
        assert_distributed_runtime(args.nproc)
    total_steps = production_steps(args.length_tokens)
    if args.device_batch_size <= 0:
        raise CurriculumConfigError("--device-batch-size must be positive")
    task_loss_nproc = args.task_loss_nproc or args.nproc
    if task_loss_nproc <= 0:
        raise CurriculumConfigError("--task-loss-nproc must be positive")
    rank_microbatch_tokens = args.device_batch_size * SEQUENCE_LENGTH
    parent, parent_paths, order, order_paths = resolve_and_stage(
        arm=arm, order_version=args.curriculum_version, cache_dir=cache_dir
    )
    dataset = ParentChunkDataset(
        parent_paths, sequence_length=SEQUENCE_LENGTH, dtype=parent.numpy_dtype
    )
    ranked = load_order(order_paths, order.numpy_dtype) if order is not None else None

    train_module = build_train_module(rank_microbatch_tokens)
    world_size = get_world_size(train_module.dp_process_group)
    if GLOBAL_BATCH_TOKENS % (world_size * rank_microbatch_tokens):
        raise CurriculumConfigError(
            f"global batch {GLOBAL_BATCH_TOKENS} is not divisible by "
            f"world_size*rank_microbatch_tokens "
            f"({world_size}*{rank_microbatch_tokens})"
        )
    loader = CurriculumDataLoader(
        dataset,
        ranked_chunk_indices=ranked,
        pacing=arm.pacing,
        difficulty_metric=arm.metric,
        seed=SEED,
        total_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        work_dir=cache_dir / "loader",
        parent_identity=parent.identity,
        order_identity=order.identity if order is not None else None,
        pad_token_id=100_277,
        vocab_size=100_352,
        dp_world_size=world_size,
        dp_rank=get_rank(train_module.dp_process_group),
        fs_local_rank=get_fs_local_rank(),
    )
    identity = scientific_identity(
        arm=arm,
        total_steps=total_steps,
        rank_microbatch_tokens=rank_microbatch_tokens,
        parent=parent,
        order=order,
    )
    if get_rank() == 0:
        fingerprint = checkpoint_contract.write_run_fingerprint(
            progress_dir / "current_fingerprint", identity
        )
        (progress_dir / "run_identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    barrier()
    fingerprint = (
        progress_dir / "current_fingerprint" / checkpoint_contract.RUN_FINGERPRINT_FILENAME
    )

    eval_script = None if args.no_task_loss else args.task_loss_eval_script
    if eval_script is not None and not eval_script.is_file():
        raise CurriculumConfigError(f"task-loss evaluator not found: {eval_script}")
    task_loss_dir = progress_dir / "task_loss_results"
    task_loss_dir.mkdir(parents=True, exist_ok=True)
    callback = CurriculumCheckpointCallback(
        arm=arm,
        total_steps=total_steps,
        save_folder=save_folder,
        progress_dir=progress_dir,
        task_loss_dir=task_loss_dir,
        eval_script=eval_script,
        task_loss_nproc=task_loss_nproc,
        production=not args.local_smoke,
        wandb_mode=args.wandb_mode,
        run_name=os.environ.get("EDULLM_RUN_ID", arm.name),
        fingerprint_path=fingerprint,
        module_builder=lambda: build_train_module(rank_microbatch_tokens),
    )
    checkpointer_options = checkpoint_contract.checkpointer_kwargs_for_ladder(total_steps)
    # The contract callback finalizes from post_train_batch. Materialize the true
    # final checkpoint in the same hook before it runs; CheckpointerCallback's
    # post_train fallback would otherwise be one hook too late.
    checkpointer_options["fixed_steps"] = [
        *checkpointer_options["fixed_steps"],
        total_steps,
    ]
    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            work_dir=str(Path(args.run_dir) / "trainer"),
            max_duration=Duration.steps(total_steps),
            metrics_collect_interval=5,
            cancel_check_interval=10,
            save_overwrite=False,
        )
        .with_callback("checkpointer", CheckpointerCallback(**checkpointer_options))
        .with_callback(
            "wandb",
            WandBCallback(
                name=os.environ.get("WANDB_NAME")
                or f"{os.environ.get('EDULLM_RUN_ID', arm.name)}-{arm.name}",
                project=_wandb_project_name(),
                group=os.environ.get("WANDB_RUN_GROUP"),
                enabled=not args.local_smoke,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("curriculum_contract", callback)
    )
    trainer = trainer_config.build(train_module, loader)
    config_saver = trainer.callbacks["config_saver"]
    assert isinstance(config_saver, ConfigSaverCallback)
    config_saver.config = identity

    if args.load_path:
        load_path = args.load_path
        if load_path.startswith("wandb-artifact://"):
            load_path = str(
                wandb_artifacts.restore_checkpoint_artifact(
                    load_path.removeprefix("wandb-artifact://"), save_folder
                )
            )
        checkpoint_contract.assert_resume_fingerprint(load_path, identity)
        trainer.load_checkpoint(load_path, load_trainer_state=True, load_optim_state=True)
    elif any(save_folder.glob("step*")):
        raise CurriculumConfigError(
            "--fresh requires an empty checkpoint directory; refusing scratch leftovers"
        )
    if get_rank() == 0:
        (save_folder / checkpoint_contract.RUN_FINGERPRINT_FILENAME).write_bytes(
            fingerprint.read_bytes()
        )
    barrier()
    trainer.fit()


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--arm-index", type=int, choices=range(len(ARMS)), required=True)
    out.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    out.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "1")))
    out.add_argument("--length-tokens", type=int)
    out.add_argument(
        "--device-batch-size",
        type=int,
        default=RANK_MICROBATCH_TOKENS // SEQUENCE_LENGTH,
        help="sequences per rank microbatch (default: 32)",
    )
    out.add_argument("--curriculum-version")
    out.add_argument("--run-dir", default=os.environ.get("RUN_DIR", "/tmp/curriculum"))
    out.add_argument("--save-folder")
    out.add_argument("--progress-dir")
    out.add_argument("--cache-dir")
    recovery = out.add_mutually_exclusive_group(required=True)
    recovery.add_argument("--fresh", action="store_true")
    recovery.add_argument("--load-path")
    out.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    out.add_argument("--local-smoke", action="store_true")
    out.add_argument(
        "--task-loss-eval-script",
        type=Path,
        default=PACKAGED_TASK_LOSS_SCRIPT,
    )
    out.add_argument(
        "--ladder-base-config",
        type=Path,
        default=PACKAGED_LADDER_CONFIG,
    )
    out.add_argument("--task-loss-nproc", type=int)
    out.add_argument("--no-task-loss", action="store_true")
    return out


def torchrun_command(args: argparse.Namespace) -> list[str]:
    forwarded = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.nproc}",
        "--",
        str(Path(__file__).resolve()),
        "--train-worker",
        "--arm-index",
        str(args.arm_index),
        "--nproc",
        str(args.nproc),
        "--device-batch-size",
        str(args.device_batch_size),
        "--run-dir",
        args.run_dir,
        "--wandb-mode",
        args.wandb_mode,
    ]
    for name in (
        "length_tokens",
        "curriculum_version",
        "save_folder",
        "progress_dir",
        "cache_dir",
        "load_path",
        "task_loss_eval_script",
        "ladder_base_config",
        "task_loss_nproc",
    ):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend([f"--{name.replace('_', '-')}", str(value)])
    for enabled, flag in (
        (args.fresh, "--fresh"),
        (args.local_smoke, "--local-smoke"),
        (args.no_task_loss, "--no-task-loss"),
    ):
        if enabled:
            forwarded.append(flag)
    return forwarded


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.train_worker:
        if args.nproc <= 0:
            raise CurriculumConfigError("--nproc must be positive")
        os.execv(sys.executable, torchrun_command(args))
    try:
        prepare_training_environment(seed=SEED, shared_filesystem=False)
        run_worker(args)
    except (CurriculumConfigError, PublishedInputError) as exc:
        print(f"[curriculum] {exc}", file=sys.stderr)
        return 2
    finally:
        if dist.is_available() and dist.is_initialized():
            teardown_training_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
