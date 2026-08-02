"""Synchronous 20-label task-loss evaluation and pause/reload coordination."""

from __future__ import annotations

import gc
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, TypeVar

from .checkpoint import (
    DEFAULT_CHECKPOINT_INTERVAL,
    CheckpointContractError,
    finalize_permanent_checkpoint,
    is_permanent_checkpoint_step,
)

log = logging.getLogger(__name__)

TASK_LOSS_RAW_LABELS = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
)
_TASK_LOSS_RAW_LABEL_SET = frozenset(TASK_LOSS_RAW_LABELS)

_DIST_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "PET_NPROC_PER_NODE",
    "PET_NNODES",
    "PET_NODE_RANK",
    "PET_MASTER_ADDR",
    "PET_MASTER_PORT",
)


class TaskLossContractError(CheckpointContractError):
    """The required synchronous task-loss suite did not complete."""


def _task_loss_label_values(payload: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in ("task_loss_bpb", "labels"):
        source = payload.get(field) or {}
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[str(key)] = float(value)
    return values


def task_loss_payload_complete(payload: Mapping[str, Any]) -> bool:
    """Return true only when all exact 20 raw BPB labels are numeric."""
    return _TASK_LOSS_RAW_LABEL_SET.issubset(_task_loss_label_values(payload))


def task_loss_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    labels = _task_loss_label_values(payload)
    metrics = {f"eval/bpb/{key}": value for key, value in labels.items()}
    if task_loss_payload_complete(payload):
        metrics["eval/macro_bpb"] = sum(labels[label] for label in TASK_LOSS_RAW_LABELS) / len(
            TASK_LOSS_RAW_LABELS
        )
    if isinstance(payload.get("macro_mean_accuracy"), (int, float)):
        metrics["eval/macro_acc"] = float(payload["macro_mean_accuracy"])
    return metrics


def validate_task_loss_result(path: str | Path) -> dict[str, Any]:
    result = Path(path)
    if not result.is_file():
        raise TaskLossContractError(f"task-loss result was not written: {result}")
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskLossContractError(f"invalid task-loss result: {result}") from exc
    if not isinstance(payload, dict) or not task_loss_payload_complete(payload):
        raise TaskLossContractError(f"out-of-contract partial task-loss result: {result}")
    return payload


def _eval_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in _DIST_ENV_KEYS:
        environment.pop(key, None)
    eval_devices = os.environ.get("TASK_LOSS_CUDA_VISIBLE_DEVICES")
    if eval_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = eval_devices
    return environment


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def trigger_task_loss_eval(
    checkpoint_dir: str | Path,
    *,
    run_name: str,
    out_path: str | Path,
    eval_script: Optional[str | Path],
    nproc: Optional[int] = None,
    python_executable: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
) -> None:
    """Run the evaluator synchronously and require a complete output file."""
    checkpoint = Path(checkpoint_dir)
    script = Path(eval_script) if eval_script is not None else None
    if script is None:
        configured = os.environ.get("TASK_LOSS_EVAL_SCRIPT", "").strip()
        script = Path(configured) if configured else None
    if script is None or not script.is_file():
        raise TaskLossContractError(
            "task-loss eval script not found; pass eval_script or set TASK_LOSS_EVAL_SCRIPT"
        )
    ready = (
        (checkpoint / "state.pt").is_file()
        or (checkpoint / "model_eval.pt").is_file()
        or (checkpoint / "model_and_optim" / ".metadata").is_file()
    )
    if not ready:
        raise TaskLossContractError(
            f"checkpoint is not materialized for task-loss evaluation: {checkpoint}"
        )

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    process_count = max(
        1,
        int(
            nproc
            if nproc is not None
            else os.environ.get("TASK_LOSS_NPROC", os.environ.get("NPROC", "1"))
        ),
    )
    python = python_executable or sys.executable
    evaluator_args = [
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(output),
        "--run-name",
        str(run_name),
        "--format",
        "auto",
        *(extra_args or []),
    ]
    if process_count > 1:
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={process_count}",
            f"--master_port={_free_port()}",
            *evaluator_args,
        ]
    else:
        command = [python, *evaluator_args]
    completed = subprocess.run(command, check=False, env=_eval_environment())
    if completed.returncode != 0:
        raise TaskLossContractError(
            f"task-loss eval exited {completed.returncode} for {checkpoint}"
        )
    validate_task_loss_result(output)


class _Distributed(Protocol):
    def is_initialized(self) -> bool: ...

    def barrier(self) -> None: ...


_T = TypeVar("_T")


def pause_eval_reload_distributed(
    checkpoint: str | Path,
    out_path: str | Path,
    run_name: str,
    *,
    evaluate: Callable[[Path, Path, str], Optional[dict[str, Any]]],
    release_train_state: Optional[Callable[[], None]],
    reload_train_state: Callable[[], _T],
    strict: bool = True,
    dist_module: Optional[_Distributed] = None,
    empty_device_cache: Optional[Callable[[], None]] = None,
) -> tuple[_T, Optional[dict[str, Any]]]:
    """Release state, evaluate in lockstep, and always reload before returning.

    Every rank must call this helper. Evaluation errors are re-raised only after
    the training state has been rebuilt, preventing a soft-failure path from
    continuing with released model or optimizer state.
    """
    if dist_module is None:
        import torch
        import torch.distributed as torch_dist

        dist_module = torch_dist
        empty_device_cache = empty_device_cache or torch.cuda.empty_cache
    if not dist_module.is_initialized():
        raise TaskLossContractError(
            "pause_eval_reload_distributed requires an initialized process group"
        )

    def clear_memory() -> None:
        gc.collect()
        if empty_device_cache is not None:
            try:
                empty_device_cache()
            except Exception:
                log.debug("device cache cleanup failed", exc_info=True)

    dist_module.barrier()
    if release_train_state is not None:
        release_train_state()
    clear_memory()
    dist_module.barrier()

    payload: Optional[dict[str, Any]] = None
    eval_error: Optional[Exception] = None
    try:
        payload = evaluate(Path(checkpoint), Path(out_path), run_name)
    except Exception as exc:
        eval_error = exc
        log.exception("task-loss evaluation failed; restoring training state")
    finally:
        clear_memory()
        dist_module.barrier()

    shared_error = str(eval_error) if eval_error is not None else None
    get_world_size = getattr(dist_module, "get_world_size", None)
    all_gather_object = getattr(dist_module, "all_gather_object", None)
    if callable(get_world_size) and callable(all_gather_object):
        gathered_errors: list[Optional[str]] = [None] * int(get_world_size())
        all_gather_object(gathered_errors, shared_error)
        shared_error = next((error for error in gathered_errors if error), None)

    restored = reload_train_state()
    dist_module.barrier()
    if shared_error is not None:
        if strict:
            raise TaskLossContractError(
                f"task-loss evaluation failed for {checkpoint}; training state was restored"
            ) from eval_error
        log.error(
            "task-loss evaluation failed for %s; continuing because strict=False",
            checkpoint,
        )
    return restored, payload


try:  # pragma: no cover - import behavior depends on the active environment.
    from olmo_core.train.callbacks import Callback

    _HAS_OLMO_CORE = True
except Exception:  # pragma: no cover
    Callback = object  # type: ignore[assignment,misc]
    _HAS_OLMO_CORE = False


class TaskLossEvalCallback(Callback if _HAS_OLMO_CORE else object):  # type: ignore[misc]
    """Finalize each permanent checkpoint while every training rank waits."""

    priority = 0

    def __init__(
        self,
        *,
        total_steps: int,
        save_folder: str | Path,
        run_name: str,
        results_dir: str | Path,
        eval_script: str | Path,
        interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        arm: Optional[str] = None,
        progress_dir: Optional[str | Path] = None,
        method: Optional[str] = None,
        task_loss_nproc: Optional[int] = None,
        production: bool = False,
        wandb_mode: Optional[str] = None,
        checkpoint_wait_seconds: int = 3600,
    ) -> None:
        if not _HAS_OLMO_CORE:
            raise ImportError("olmo_core is required for TaskLossEvalCallback")
        super().__init__()  # type: ignore[misc]
        self.total_steps = int(total_steps)
        self.save_folder = Path(save_folder)
        self.run_name = str(run_name)
        self.results_dir = Path(results_dir)
        self.eval_script = Path(eval_script)
        self.interval = int(interval)
        self.arm = str(arm or method or self.save_folder.name)
        self.progress_dir = Path(progress_dir) if progress_dir else None
        self.method = str(method) if method else self.arm
        self.task_loss_nproc = int(task_loss_nproc) if task_loss_nproc is not None else None
        self.production = bool(production)
        self.wandb_mode = wandb_mode
        self.checkpoint_wait_seconds = int(checkpoint_wait_seconds)
        self._completed: set[int] = set()

    @staticmethod
    def _distributed() -> tuple[Any, int]:
        import torch.distributed as dist

        initialized = dist.is_available() and dist.is_initialized()
        return dist, dist.get_rank() if initialized else 0

    def _wait_for_checkpoint(self, checkpoint: Path) -> None:
        deadline = time.monotonic() + self.checkpoint_wait_seconds
        while time.monotonic() < deadline:
            if (
                (checkpoint / "state.pt").is_file()
                or (checkpoint / "model_eval.pt").is_file()
                or (checkpoint / "model_and_optim" / ".metadata").is_file()
            ):
                return
            time.sleep(1)
        raise TaskLossContractError(f"timed out waiting for permanent checkpoint {checkpoint}")

    def _maybe_finalize(self, step: int) -> None:
        step = int(step)
        if step in self._completed or not is_permanent_checkpoint_step(
            step, self.total_steps, self.interval
        ):
            return

        dist, rank = self._distributed()
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()
        failure: list[Optional[str]] = [None]
        if rank == 0:
            try:
                checkpoint = self.save_folder / f"step{step}"
                self._wait_for_checkpoint(checkpoint)
                from .wandb_artifacts import wandb_run_from_trainer

                finalize_permanent_checkpoint(
                    arm=self.arm,
                    checkpoint_dir=checkpoint,
                    step=step,
                    run_name=self.run_name,
                    task_loss_dir=self.results_dir,
                    task_loss_enabled=True,
                    eval_script=self.eval_script,
                    task_loss_nproc=self.task_loss_nproc,
                    progress_dir=self.progress_dir,
                    fingerprint_path=self.save_folder / "run_fingerprint.json",
                    method=self.method,
                    wandb_run=wandb_run_from_trainer(self.trainer),
                    wandb_mode=self.wandb_mode,
                    production=self.production,
                    upload_checkpoint=step == self.total_steps,
                )
            except BaseException as exc:  # noqa: BLE001
                failure[0] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(failure, src=0)
        if failure[0] is not None:
            raise TaskLossContractError(f"permanent checkpoint step {step} failed: {failure[0]}")
        self._completed.add(step)
        if distributed:
            dist.barrier()

    def pre_train(self) -> None:  # pragma: no cover - requires an OLMo trainer.
        self._maybe_finalize(0)

    def post_step(self) -> None:  # pragma: no cover
        self._maybe_finalize(int(self.step))

    def post_train(self) -> None:  # pragma: no cover
        self._maybe_finalize(int(self.step))
