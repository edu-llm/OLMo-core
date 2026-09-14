#!/usr/bin/env python3
"""Run one curriculum arm on OLMo-core's public APIs.

Rewritten from scratch for `curriculum-new` (no code copied from the
edullm repo). Differences from the prior `edullm/curriculum-370m`
implementation, all driven by the five-reviewer audit (plan section 1):

- The reported model is the true final checkpoint, produced after a
  post-curriculum anneal phase (plan 6a). No post-hoc averaging of any
  kind: averaging across checkpoints would mix in weights from before an
  arm's hardest decile had ever been presented, an arm-dependent bias as
  large as the effect being measured.
- Plain `AdamWConfig`, not `SkipStepAdamW`. The skip-step guard's rolling
  loss window is not checkpointed and is lost across every forced restart,
  and curriculum decile boundaries are themselves large, arm-dependent loss
  jumps that the guard would selectively suppress (finding 1c).
- No adaptive-bucket or domain-mixture pacing. Neither module was ever
  version-controlled in the prior branch, and neither is part of the frozen
  arm set (plan 6b).
- Local-scratch inputs only. No AWS, no S3, no `edullm_data`, no platform
  dataset gate (plan 3a).
- `pacing`, `metric`, `lr_schedule`, and `seed` are independent CLI
  dimensions rather than entries in a frozen, hard-to-extend arm-index
  matrix -- the audit found the old matrix could not express every
  (pacing, metric) combination the design needed under any single metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import torch
import torch.distributed as dist

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from olmo_core.float8 import AOFloat8LinearConfig, Float8Config
from olmo_core.nn.transformer import TransformerDataParallelWrappingStrategy
from olmo_core.optim import AdamWConfig, ConstantWithWarmup, CosWithWarmup, OptimGroupOverride
from olmo_core.optim.scheduler import Scheduler, SequentialScheduler
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
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)

from curriculum_data import PublishedInputError, ResolvedInput, resolve_local_parent
from curriculum_loader import CurriculumDataLoader, ParentChunkDataset
from curriculum_model import MODEL_IDENTITY, build_model_config
from curriculum_pacing import (
    CURRICULUM_STEPS,
    DIFFICULTY_METRIC_DEFINITIONS,
    DIFFICULTY_METRICS,
    PACING_NAMES,
    TOTAL_STEPS,
)
from production_contract import checkpoint as checkpoint_contract
from production_contract import task_loss
from production_contract import wandb_artifacts

PACKAGED_TASK_LOSS_SCRIPT = Path(__file__).with_name("task_loss") / "eval_task_loss_olmo_core.py"
PACKAGED_LADDER_CONFIG = Path(__file__).with_name("task_loss") / "ladder_base_config.yaml"

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_TOKENS = 4_194_304
log = logging.getLogger(__name__)

RANK_MICROBATCH_TOKENS = 16_384  # 8 sequences/rank, verified to fit 4xL40S (plan freeze contract)
PEAK_LR = 4e-4
WARMUP_STEPS = 24
LR_SCHEDULES = ("constant", "cosine")
COSINE_ALPHA_F = 0.1
CHECKPOINT_INTERVAL = 125
WANDB_PROJECT_NAME = "curriculum-new"
# The pre-campaign smoke runs the real W&B path -- that is the only way to
# verify that eval and metrics artifacts upload while model weights do not
# (see ALLOW_MODEL_ARTIFACT_UPLOAD) -- but a 250-step shakedown must not sit
# in the campaign's project, where it could later be mistaken for data. The
# allowlist exists to stop runs landing in the wrong project, so the smoke
# gets a named member rather than a silent override.
WANDB_SMOKE_PROJECT_NAME = "curriculum-new-smoke"
WANDB_PROJECT_NAMES = frozenset((WANDB_PROJECT_NAME, WANDB_SMOKE_PROJECT_NAME))
CHECKPOINT_RESTART_REQUEST = "restart_after_checkpoint.json"


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
    """One (pacing, metric, lr_schedule) combination.

    Replaces the prior frozen 16-entry arm-index matrix (plan finding: that
    matrix could not express every (pacing, metric) combination the design
    needed under a single metric, and any addition required editing a
    byte-exact-checked tuple in two files). `pacing`, `metric`, and
    `lr_schedule` are validated independently instead.
    """

    pacing: str
    metric: str | None
    lr_schedule: str

    def __post_init__(self) -> None:
        if self.pacing not in PACING_NAMES:
            raise CurriculumConfigError(f"unknown pacing {self.pacing!r}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise CurriculumConfigError(f"unknown lr_schedule {self.lr_schedule!r}")
        if self.pacing == "control":
            if self.metric is not None:
                raise CurriculumConfigError("control's --metric is only used to pick a parent")
        elif self.metric not in DIFFICULTY_METRICS:
            raise CurriculumConfigError(f"unknown difficulty metric {self.metric!r}")

    @property
    def name(self) -> str:
        metric_part = f"-{self.metric}" if self.metric else ""
        return f"{self.pacing}{metric_part}-{self.lr_schedule}"

    @property
    def wandb_project(self) -> str:
        return WANDB_PROJECT_NAME


@Scheduler.register("curriculum_sqrt_anneal")
@dataclass
class SqrtAnnealScheduler(Scheduler):
    """1 - sqrt(r) decay from `initial_lr` to `end_lr`.

    Matches the WSD decay shape in Luo et al. (arXiv:2511.18903, ICLR 2026):
    eta(t) = eta_0*(1-sqrt(r)) + eta_T*sqrt(r), r = current/t_max. Used as
    the second stage of a `SequentialScheduler` for the post-curriculum
    anneal (plan 6a/6c): the anneal always starts from whatever LR the
    curriculum phase ended at -- `SequentialScheduler` makes that
    continuous by construction (the next stage's `initial_lr` is the
    previous stage's final LR), so there is never a jump at the
    curriculum/anneal boundary.
    """

    end_lr: float = 0.0

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if t_max <= 0:
            return self.end_lr
        clamped = min(max(current, 0), t_max)
        r = sqrt(clamped / t_max)
        return initial_lr * (1 - r) + self.end_lr * r


def build_scheduler(lr_schedule: str) -> SequentialScheduler:
    """Compose the curriculum-phase schedule with the post-curriculum anneal.

    Both conditions run the identical data protocol (CURRICULUM_STEPS of
    curriculum, then ANNEAL_STEPS of uniform-pool 1-sqrt anneal to 0); only
    the LR curve during the curriculum phase differs (plan 6c):

    - "constant": ConstantWithWarmup (peak LR held after a 24-step warmup)
      through CURRICULUM_STEPS, then anneal from peak to 0.
    - "cosine": CosWithWarmup(alpha_f=0.1) decaying peak -> 0.1*peak over
      CURRICULUM_STEPS, then anneal from 0.1*peak to 0.

    `SequentialScheduler.schedulers_max=[CURRICULUM_STEPS]` means the first
    scheduler runs for exactly CURRICULUM_STEPS and the second (the anneal)
    is assumed to run until the end of training -- i.e. exactly
    ANNEAL_STEPS, since the trainer's total duration is TOTAL_STEPS.
    """
    if lr_schedule == "constant":
        phase_one: Scheduler = ConstantWithWarmup(warmup=WARMUP_STEPS)
    elif lr_schedule == "cosine":
        phase_one = CosWithWarmup(warmup=WARMUP_STEPS, alpha_f=COSINE_ALPHA_F)
    else:
        raise CurriculumConfigError(f"unknown lr_schedule {lr_schedule!r}")
    return SequentialScheduler(
        schedulers=[phase_one, SqrtAnnealScheduler(end_lr=0.0)],
        schedulers_max=[CURRICULUM_STEPS],
    )


def checkpoint_steps(total_steps: int) -> list[int]:
    return checkpoint_contract.permanent_checkpoint_steps(total_steps, CHECKPOINT_INTERVAL)


def numerics_identity() -> dict[str, Any]:
    """Resolve every EDULLM_BENCH_* knob that changes the arithmetic.

    Single source of truth for both :func:`train_module_config` and the run
    fingerprint (plan finding 1f). These variables alter parallelism
    strategy, gradient-reduction dtype, expert parallelism, prefetch depth
    and fp8 linears; none of them entered `scientific_identity` before, so
    two runs could be recorded as identical while computing differently.

    Resolved values are returned rather than raw env strings, so an unset
    variable and an explicitly-default one produce the same fingerprint.
    """
    return {
        "dp_name": (
            "fsdp" if os.environ.get("EDULLM_BENCH_DP") == "fsdp" else "hsdp"
        ),
        "reduce_dtype": (
            "bfloat16" if os.environ.get("EDULLM_BENCH_REDUCE_BF16") == "1" else "float32"
        ),
        "param_dtype": "bfloat16",
        "ep_degree": int(os.environ.get("EDULLM_BENCH_EP_DEGREE", "0")),
        "prefetch_factor": int(os.environ.get("EDULLM_BENCH_PREFETCH", "0")),
        "float8": os.environ.get("EDULLM_BENCH_FLOAT8") == "1",
        "compile_model": True,
        "z_loss_multiplier": 1e-5,
        "max_grad_norm": 1.0,
        "attn_backend": os.environ.get("OLMO_ATTN_BACKEND", "torch"),
        "flash_attention": os.environ.get("OLMO_FLASH_ATTENTION", "0") == "1",
        "fused_loss": os.environ.get("OLMO_FUSED_LOSS", "0") == "1",
    }


def train_module_config(
    lr_schedule: str,
    rank_microbatch_tokens: int = RANK_MICROBATCH_TOKENS,
) -> TransformerTrainModuleConfig:
    numerics = numerics_identity()
    dp_name = (
        DataParallelType.fsdp
        if numerics["dp_name"] == "fsdp"
        else DataParallelType.hsdp
    )
    reduce_dtype = (
        DType.bfloat16 if numerics["reduce_dtype"] == "bfloat16" else DType.float32
    )
    ep_degree = int(numerics["ep_degree"])
    dp_options: dict[str, Any] = {}
    if ep_degree and dp_name == DataParallelType.hsdp:
        dp_options["num_replicas"] = 1
    return TransformerTrainModuleConfig(
        rank_microbatch_size=int(rank_microbatch_tokens),
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=PEAK_LR,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=build_scheduler(lr_schedule),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=dp_name,
            param_dtype=DType.bfloat16,
            reduce_dtype=reduce_dtype,
            prefetch_factor=int(numerics["prefetch_factor"]),
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            **dp_options,
        ),
        ep_config=(
            TransformerExpertParallelConfig(degree=ep_degree) if ep_degree else None
        ),
        float8_config=(
            Float8Config(ao=AOFloat8LinearConfig.recommended())
            if numerics["float8"]
            else Float8Config(enabled=False)
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_train_module(lr_schedule: str, rank_microbatch_tokens: int = RANK_MICROBATCH_TOKENS) -> Any:
    model = build_model_config().build(init_device="meta")
    return train_module_config(lr_schedule, rank_microbatch_tokens).build(model)


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
    """Checkpoint -> all-rank eval/reload -> awaited W&B artifacts -> marker.

    No post-hoc finalization step: the true final checkpoint (reached after
    the post-curriculum anneal) is evaluated and published like every other
    permanent checkpoint, nothing more (see the module docstring).
    """

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
        module_builder: Callable[[], Any],
        prune_older: bool = True,
    ) -> None:
        self.prune_older = bool(prune_older)
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
        self._load_durable_state()

    def _durable_state_path(self) -> Path:
        return self.progress_dir / "checkpoint_callback_state.json"

    def _load_durable_state(self) -> None:
        # Mirrors _completed into a plain file under progress_dir so that a
        # hard-stop/resume cycle (one occurs at every non-final permanent
        # checkpoint, to clear compiled CUDA state after the synchronous
        # evaluator) reconstructs it exactly, even though a checkpoint's
        # embedded state_dict always lags one step behind the callback's own
        # in-memory state at the moment it is written.
        path = self._durable_state_path()
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self._completed |= {int(step) for step in payload.get("completed_steps", [])}

    def _persist_durable_state(self) -> None:
        if get_rank() != 0:
            return
        path = self._durable_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"completed_steps": sorted(self._completed)}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)

    def state_dict(self) -> dict[str, Any]:
        return {"completed_steps": sorted(self._completed)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._completed |= {int(step) for step in state_dict.get("completed_steps", [])}
        self._load_durable_state()

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
        return "plain_ce" if self.arm.pacing == "control" else f"curriculum:{self.arm.pacing}"

    def _finalize(self, step: int) -> None:
        if not self.production and self.eval_script is None:
            return
        step = int(step)
        if step not in checkpoint_steps(self.total_steps):
            return
        if step in self._completed:
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
                    # Never upload model weights; see
                    # ALLOW_MODEL_ARTIFACT_UPLOAD in wandb_artifacts.py.
                    # Checkpoints stay on FarmShare scratch, which is this
                    # campaign's system of record. Eval and metrics artifacts
                    # (small) still upload.
                    upload_checkpoint=False,
                    prune_older=self.prune_older,
                    run_evaluator=self._already_evaluated,
                )
            except BaseException as exc:  # noqa: BLE001
                failure = f"permanent checkpoint step {step} failed: {type(exc).__name__}: {exc}"
        _broadcast_failure(failure)
        self._completed.add(step)
        self._persist_durable_state()
        barrier()
        if 0 < step < self.total_steps:
            if get_rank() == 0:
                request = self.progress_dir / CHECKPOINT_RESTART_REQUEST
                request.parent.mkdir(parents=True, exist_ok=True)
                temporary = request.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "durable_step": step,
                            "reason": "clear_cuda_state_after_task_loss_eval",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, request)
            barrier()
            # Reloading after the synchronous evaluator leaves compiled CUDA
            # state resident. End this process cleanly at the durable
            # boundary; the FarmShare supervisor loop resumes from the kept
            # checkpoint in a fresh CUDA process.
            self.trainer.hard_stop = Duration.steps(step)

    def pre_train(self) -> None:
        self._finalize(0)

    def post_train_batch(self) -> None:
        self._finalize(self.step)

    def post_train(self) -> None:
        self._finalize(self.step)


def resolve_parent(*, metric_for_parent: str, manifest_path: Path, cache_dir: Path) -> ResolvedInput:
    """Resolve (and lightly verify) the locally-staged parent for this run.

    Every rank resolves independently -- there is no network call and no
    download to coordinate, unlike the removed S3 path, so no rank-0-then-
    barrier dance is needed here. A sidecar is still written for humans
    inspecting a run directory, not for cross-rank coordination.
    """
    parent = resolve_local_parent(manifest_path=manifest_path, metric=metric_for_parent)
    if get_rank() == 0:
        sidecar = cache_dir / f"resolved-{metric_for_parent}.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps({"parent": parent.identity}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return parent


def scientific_identity(
    *,
    arm: Arm,
    seed: int,
    total_steps: int,
    rank_microbatch_tokens: int,
    parent: ResolvedInput,
    task_loss_nproc: int,
) -> dict[str, Any]:
    return {
        "family": "curriculum-new",
        "arm": arm.name,
        "pacing": arm.pacing,
        "difficulty_metric": arm.metric,
        "lr_schedule": arm.lr_schedule,
        "parent": parent.identity,
        "model": MODEL_IDENTITY,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": int(rank_microbatch_tokens),
        "total_steps": int(total_steps),
        "curriculum_steps": CURRICULUM_STEPS,
        "seed": int(seed),
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "cosine_alpha_f": COSINE_ALPHA_F,
        "checkpoint_steps": checkpoint_steps(total_steps),
        "optimizer": "AdamW",
        # Plan 1f: every numerics-affecting value, resolved. Changing any of
        # these changes the arithmetic, so resume refuses and two runs that
        # differ here can never be pooled by accident.
        "numerics": numerics_identity(),
        # Plan 1e: DistributedSampler pads to divisibility, so bpb from a
        # 4-rank eval is not comparable to bpb from an 8-rank eval.
        "task_loss_nproc": int(task_loss_nproc),
        # Plan section 2: the ratified definition of the difficulty metric
        # this arm is ordered by. Recorded so a rescore or a changed formula
        # cannot be substituted silently -- the fingerprint moves, resume
        # refuses, and runs either side of the change are never pooled.
        # Only this arm's metric is recorded; control carries None.
        "difficulty_metric_definition": (
            DIFFICULTY_METRIC_DEFINITIONS[arm.metric] if arm.metric else None
        ),
    }


def _validate_runtime(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    project = _wandb_project_name()
    if project not in WANDB_PROJECT_NAMES:
        raise CurriculumConfigError(
            f"W&B project must be one of {sorted(WANDB_PROJECT_NAMES)!r}, got {project!r}"
        )
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
    # FarmShare home carries a 48 GB quota, which one 370M run's checkpoints
    # will exhaust on its own. Runtime paths must be on scratch. Local smoke
    # runs are exempt: they write a few MB into a temp dir, which on some
    # platforms legitimately lives under the user's home.
    try:
        home = Path.home().resolve()
    except (RuntimeError, OSError):  # no resolvable home; nothing to guard
        home = None
    for path in (save_folder, progress_dir, cache_dir):
        if "://" in str(path):
            raise CurriculumConfigError("curriculum runtime paths must be job-local scratch")
        if home is not None and not args.local_smoke:
            resolved = path.resolve()
            if resolved == home or home in resolved.parents:
                raise CurriculumConfigError(
                    f"curriculum runtime path {resolved} is under $HOME ({home}); "
                    f"FarmShare home has a 48 GB quota that one run will exhaust. "
                    f"Use /scratch/users/$USER/..."
                )
        path.mkdir(parents=True, exist_ok=True)
    return save_folder, progress_dir, cache_dir


def run_worker(args: argparse.Namespace) -> None:
    arm = Arm(pacing=args.pacing, metric=args.metric, lr_schedule=args.lr_schedule)
    save_folder, progress_dir, cache_dir = _validate_runtime(args)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["TASK_LOSS_EVAL_SCRIPT"] = str(args.task_loss_eval_script)
    os.environ["LADDER_BASE_CONFIG"] = str(args.ladder_base_config)
    if not args.local_smoke:
        assert_distributed_runtime(args.nproc)
    total_steps = TOTAL_STEPS if args.length_tokens is None else args.length_tokens // GLOBAL_BATCH_TOKENS
    if args.device_batch_size <= 0:
        raise CurriculumConfigError("--device-batch-size must be positive")
    task_loss_nproc = args.task_loss_nproc or args.nproc
    if task_loss_nproc <= 0:
        raise CurriculumConfigError("--task-loss-nproc must be positive")
    rank_microbatch_tokens = args.device_batch_size * SEQUENCE_LENGTH

    metric_for_parent = arm.metric or args.control_metric
    if metric_for_parent is None:
        raise CurriculumConfigError("control requires --control-metric to select a parent")
    parent = resolve_parent(
        metric_for_parent=metric_for_parent,
        manifest_path=Path(args.parent_manifest),
        cache_dir=cache_dir,
    )
    dataset = ParentChunkDataset(
        [Path(path) for path in parent.paths],
        sequence_length=SEQUENCE_LENGTH,
        dtype=parent.numpy_dtype,
    )
    if len(dataset) != parent.chunk_count:
        raise CurriculumConfigError(
            f"parent manifest claims {parent.chunk_count} chunks, dataset has {len(dataset)}"
        )
    # The corpus rebuild re-chunks in difficulty order (plan section 5), so
    # chunk index already equals difficulty rank -- the "order" is simply
    # the identity permutation, generated in memory rather than read from a
    # separate file.
    ranked = None if arm.pacing == "control" else np.arange(len(dataset), dtype=np.int64)

    train_module = build_train_module(arm.lr_schedule, rank_microbatch_tokens)
    world_size = get_world_size(train_module.dp_process_group)
    if GLOBAL_BATCH_TOKENS % (world_size * rank_microbatch_tokens):
        raise CurriculumConfigError(
            f"global batch {GLOBAL_BATCH_TOKENS} is not divisible by "
            f"world_size*rank_microbatch_tokens ({world_size}*{rank_microbatch_tokens})"
        )
    loader = CurriculumDataLoader(
        dataset,
        ranked_chunk_indices=ranked,
        pacing=arm.pacing,
        difficulty_metric=arm.metric,
        seed=args.seed,
        total_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        work_dir=cache_dir / "loader",
        parent_identity=parent.identity,
        order_identity=None if arm.pacing == "control" else {"kind": "identity_permutation"},
        pad_token_id=100_277,
        vocab_size=100_352,
        dp_world_size=world_size,
        dp_rank=get_rank(train_module.dp_process_group),
        fs_local_rank=get_fs_local_rank(),
    )
    identity = scientific_identity(
        arm=arm,
        seed=args.seed,
        total_steps=total_steps,
        rank_microbatch_tokens=rank_microbatch_tokens,
        parent=parent,
        task_loss_nproc=task_loss_nproc,
    )
    if get_rank() == 0:
        checkpoint_contract.write_run_fingerprint(progress_dir / "current_fingerprint", identity)
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
    prune_older = args.keep_checkpoints != "all"
    if not prune_older:
        log.warning(
            "--keep-checkpoints=all: retaining the full %d-step permanent ladder on "
            "scratch. Nothing is pruned, and model artifacts are never uploaded, so "
            "scratch is the only copy.",
            len(checkpoint_contract.permanent_checkpoint_steps(total_steps, CHECKPOINT_INTERVAL)),
        )
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
        module_builder=lambda: build_train_module(arm.lr_schedule, rank_microbatch_tokens),
        prune_older=prune_older,
    )
    checkpointer_options = checkpoint_contract.checkpointer_kwargs_for_ladder(total_steps)
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
        .with_callback(
            "checkpointer",
            CheckpointerCallback(enabled=not args.local_smoke, **checkpointer_options),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=os.environ.get("WANDB_NAME") or f"{os.environ.get('EDULLM_RUN_ID', arm.name)}",
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
    out.add_argument("--pacing", choices=PACING_NAMES, required=True)
    out.add_argument("--metric", choices=DIFFICULTY_METRICS, default=None)
    out.add_argument(
        "--control-metric",
        choices=DIFFICULTY_METRICS,
        default=None,
        help="for --pacing control only: which metric's materialized parent to train on",
    )
    out.add_argument("--lr-schedule", choices=LR_SCHEDULES, required=True)
    out.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "ONE seed, driving both data order and model initialization -- "
            "ratified, not an oversight. Combined with the pacing tag being "
            "absent from the step seed (plan 1g), two arms at the same seed "
            "share their data stream and their initial weights, which is what "
            "common random numbers requires and is worth more at this budget "
            "than an extra replicate. Do not split this into separate data "
            "and init seeds: it is part of the frozen contract, and a split "
            "would allow two arms to differ in init unnoticed."
        ),
    )
    out.add_argument(
        "--parent-manifest",
        type=Path,
        required=True,
        help="path to the corpus-rebuild manifest for the metric this run trains on",
    )
    out.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    out.add_argument("--nproc", type=int, default=int(os.environ.get("NPROC", "1")))
    out.add_argument("--length-tokens", type=int)
    out.add_argument(
        "--device-batch-size",
        type=int,
        default=RANK_MICROBATCH_TOKENS // SEQUENCE_LENGTH,
        help="sequences per rank microbatch (default: 8, verified to fit 4xL40S)",
    )
    out.add_argument("--run-dir", default=os.environ.get("RUN_DIR", "/tmp/curriculum"))
    out.add_argument("--save-folder")
    out.add_argument("--progress-dir")
    out.add_argument("--cache-dir")
    recovery = out.add_mutually_exclusive_group(required=True)
    recovery.add_argument("--fresh", action="store_true")
    recovery.add_argument("--load-path")
    out.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    out.add_argument("--local-smoke", action="store_true")
    out.add_argument("--task-loss-eval-script", type=Path, default=PACKAGED_TASK_LOSS_SCRIPT)
    out.add_argument("--ladder-base-config", type=Path, default=PACKAGED_LADDER_CONFIG)
    out.add_argument("--task-loss-nproc", type=int)
    out.add_argument("--no-task-loss", action="store_true")
    out.add_argument(
        "--keep-checkpoints",
        choices=("latest", "all"),
        default="latest",
        help=(
            "checkpoint retention. 'latest' (default) keeps only the newest "
            "durable checkpoint, which is all a resume needs. 'all' keeps the "
            "whole permanent ladder -- required if intermediate checkpoints are "
            "themselves data (e.g. per-document loss trajectories), since a "
            "pruned checkpoint is gone and scratch is the only copy."
        ),
    )
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
        "--pacing",
        args.pacing,
        "--lr-schedule",
        args.lr_schedule,
        "--seed",
        str(args.seed),
        "--parent-manifest",
        str(args.parent_manifest),
        "--nproc",
        str(args.nproc),
        "--device-batch-size",
        str(args.device_batch_size),
        "--run-dir",
        args.run_dir,
        "--wandb-mode",
        args.wandb_mode,
    ]
    if args.metric is not None:
        forwarded.extend(["--metric", args.metric])
    if args.control_metric is not None:
        forwarded.extend(["--control-metric", args.control_metric])
    for name in (
        "length_tokens",
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
        prepare_training_environment(seed=args.seed, shared_filesystem=False)
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
