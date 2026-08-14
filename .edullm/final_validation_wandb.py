"""Synchronous task-loss evaluation for the 370M/10B final validations."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch.distributed as dist

from olmo_core import io
from olmo_core.train.callbacks import Callback, WandBCallback

TASK_LABELS = (
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


class FinalValidationContractError(RuntimeError):
    """A required checkpoint evaluation or W&B upload did not complete."""


@dataclass
class ScaledWandBCallback(WandBCallback):
    """Log trainer metrics on a token-aligned W&B step axis."""

    step_multiplier: int = 1
    """Multiply native optimizer steps so each W&B step represents a fixed token count."""

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        multiplier = int(self.step_multiplier)
        if multiplier < 1:
            raise ValueError(f"step_multiplier must be >= 1, got {multiplier}")
        scaled = dict(metrics)
        if multiplier != 1 and "checkpoint/step" in scaled:
            scaled["checkpoint/step"] = float(scaled["checkpoint/step"]) * multiplier
        super().log_metrics(step * multiplier, scaled)


def validate_eval(path: Path) -> dict[str, Any]:
    """Load an evaluator result and require all 20 numeric BPB labels."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalValidationContractError(f"invalid task-loss result: {path}") from exc
    labels = payload.get("labels") if isinstance(payload, Mapping) else None
    if not isinstance(labels, Mapping):
        raise FinalValidationContractError(f"task-loss result has no labels mapping: {path}")
    missing = [
        label
        for label in TASK_LABELS
        if isinstance(labels.get(label), bool) or not isinstance(labels.get(label), (int, float))
    ]
    if missing:
        raise FinalValidationContractError(
            f"task-loss result is missing {len(missing)} required labels: {missing[0]}"
        )
    return payload


def eval_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Convert a complete task-loss result into W&B metrics."""

    labels = payload["labels"]
    metrics = {f"eval/bpb/{label}": float(labels[label]) for label in TASK_LABELS}
    metrics["eval/macro_bpb"] = sum(metrics.values()) / len(TASK_LABELS)
    return metrics


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _eval_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in _DIST_ENV_KEYS:
        environment.pop(key, None)
    environment["WANDB_DISABLED"] = "1"
    environment["WANDB_MODE"] = "disabled"
    return environment


def _wandb_run(trainer: Any) -> Any | None:
    callbacks = getattr(trainer, "callbacks", None) or {}
    for callback in callbacks.values() if isinstance(callbacks, Mapping) else ():
        for attribute in ("run", "_run", "wandb_run"):
            run = getattr(callback, attribute, None)
            if run is not None:
                return run
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run


def _wait_for_upload(logged: Any, description: str) -> None:
    wait = getattr(logged, "wait", None)
    if not callable(wait):
        raise FinalValidationContractError(
            f"required W&B {description} upload returned no waitable handle"
        )
    try:
        wait()
    except Exception as exc:  # noqa: BLE001
        raise FinalValidationContractError(
            f"required W&B {description} upload did not complete"
        ) from exc


def _artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "final-validation"


def upload_eval(
    run: Any,
    payload: Mapping[str, Any],
    path: Path,
    step: int,
    *,
    run_name: str,
    vector_name: str,
) -> None:
    """Log task metrics and the complete JSON result."""

    import wandb

    run.log(eval_metrics(payload), step=step)
    artifact = wandb.Artifact(
        name=f"{_artifact_name(run_name)}-task-loss-step{step:07d}",
        type="eval",
        metadata={"step": step, "vector": vector_name},
    )
    artifact.add_file(str(path), name=path.name)
    _wait_for_upload(
        run.log_artifact(artifact, aliases=[f"step-{step:07d}"]),
        f"task-loss step {step}",
    )


def upload_final_checkpoint(
    run: Any,
    checkpoint: Path,
    *,
    step: int,
    run_name: str,
    vector_name: str,
) -> None:
    """Publish only the final model checkpoint to W&B."""

    import wandb

    artifact = wandb.Artifact(
        name=f"{_artifact_name(run_name)}-checkpoint",
        type="model",
        metadata={"step": step, "vector": vector_name, "final": True},
    )
    artifact.add_dir(str(checkpoint))
    _wait_for_upload(
        run.log_artifact(
            artifact,
            aliases=["latest", "final", f"step-{step:07d}"],
        ),
        f"final checkpoint step {step}",
    )


class FinalValidationEvalCallback(Callback):
    """Evaluate each point in the fixed ladder and upload the final checkpoint."""

    priority = 0

    def __init__(
        self,
        *,
        vector_name: str,
        total_steps: int,
        checkpoint_steps: Sequence[int],
        save_folder: str,
        run_name: str,
        work_dir: str | Path,
        eval_script: str | Path,
        nproc: int = 8,
        checkpoint_wait_seconds: int = 3_600,
        wandb_step_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.vector_name = str(vector_name)
        self.total_steps = int(total_steps)
        self.wandb_step_multiplier = int(wandb_step_multiplier)
        if self.wandb_step_multiplier < 1:
            raise ValueError(
                f"wandb_step_multiplier must be >= 1, got {self.wandb_step_multiplier}"
            )
        self.checkpoint_steps = tuple(int(step) for step in checkpoint_steps)
        if (
            not self.checkpoint_steps
            or self.checkpoint_steps[0] != 0
            or self.checkpoint_steps[-1] != self.total_steps
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
        ):
            raise ValueError("checkpoint_steps must be sorted, unique, and endpoint-inclusive")
        self.save_folder = str(save_folder).rstrip("/")
        self.run_name = str(run_name)
        self.work_dir = Path(work_dir)
        self.eval_script = Path(eval_script)
        self.nproc = int(nproc)
        self.checkpoint_wait_seconds = int(checkpoint_wait_seconds)
        self._completed: set[int] = set()

    def state_dict(self) -> dict[str, Any]:
        return {"completed_steps": sorted(self._completed)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._completed = {int(step) for step in state_dict.get("completed_steps", [])}

    def _wait_for_checkpoint(self, source: str) -> None:
        candidates = (
            io.join_path(source, "state.pt"),
            io.join_path(source, "model_eval.pt"),
            io.join_path(source, "model_and_optim", ".metadata"),
        )
        deadline = time.monotonic() + self.checkpoint_wait_seconds
        while time.monotonic() < deadline:
            if any(io.file_exists(path) for path in candidates):
                return
            time.sleep(1)
        raise FinalValidationContractError(f"timed out waiting for checkpoint {source}")

    @staticmethod
    def _stage_checkpoint(source: str, target: Path) -> None:
        if io.is_url(source):
            target.mkdir(parents=True)
            io.copy_dir(source, target, save_overwrite=True)
        else:
            shutil.copytree(Path(source), target)

    def _evaluate(self, checkpoint: Path, output: Path, step: int) -> dict[str, Any]:
        if not self.eval_script.is_file():
            raise FinalValidationContractError(f"evaluation script not found: {self.eval_script}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={self.nproc}",
            f"--master_port={_free_port()}",
            str(self.eval_script),
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
            "--run-name",
            f"{self.run_name}-step{step}",
            "--format",
            "auto",
        ]
        completed = subprocess.run(command, check=False, env=_eval_environment())
        if completed.returncode != 0:
            raise FinalValidationContractError(
                f"task-loss evaluation exited {completed.returncode} at step {step}"
            )
        return validate_eval(output)

    def _finalize_rank_zero(self, step: int) -> None:
        run = _wandb_run(self.trainer)
        if run is None:
            raise FinalValidationContractError("required W&B run is not active")
        source = io.join_path(self.save_folder, f"step{step}")
        self._wait_for_checkpoint(source)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        output = self.work_dir / "task-loss" / f"step{step}_task_loss.json"
        with tempfile.TemporaryDirectory(prefix=f"step{step}-", dir=self.work_dir) as temporary:
            checkpoint = Path(temporary) / f"step{step}"
            self._stage_checkpoint(source, checkpoint)
            payload = self._evaluate(checkpoint, output, step)
            wandb_step = step * self.wandb_step_multiplier
            upload_eval(
                run,
                payload,
                output,
                wandb_step,
                run_name=self.run_name,
                vector_name=self.vector_name,
            )
            if step == self.total_steps:
                upload_final_checkpoint(
                    run,
                    checkpoint,
                    step=wandb_step,
                    run_name=self.run_name,
                    vector_name=self.vector_name,
                )

    def _maybe_finalize(self, step: int) -> None:
        step = int(step)
        if step in self._completed or step not in self.checkpoint_steps:
            return
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()
        failure: list[str | None] = [None]
        rank = dist.get_rank() if distributed else 0
        if rank == 0:
            try:
                self._finalize_rank_zero(step)
            except BaseException as exc:  # noqa: BLE001
                failure[0] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(failure, src=0)
        if failure[0] is not None:
            raise FinalValidationContractError(
                f"checkpoint step {step} failed task-loss durability: {failure[0]}"
            )
        self._completed.add(step)
        if distributed:
            dist.barrier()

    def pre_train(self) -> None:
        self._maybe_finalize(0)

    def post_step(self) -> None:
        self._maybe_finalize(self.step)

    def post_train(self) -> None:
        self._maybe_finalize(self.total_steps)
