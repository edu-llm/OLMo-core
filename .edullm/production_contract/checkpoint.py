"""Permanent-checkpoint ladder and durability contract.

The ordering is strict: materialize the checkpoint, complete the 20-label
task-loss suite, upload required W&B artifacts, then advance the local durable
step marker. Production online runs fail before advancing the marker if any
required operation fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

DEFAULT_CHECKPOINT_INTERVAL = 125
RUN_FINGERPRINT_FILENAME = "run_fingerprint.json"
LAST_DURABLE_STEP_FILENAME = "last_durable_step.json"
FINGERPRINT_SCHEMA_VERSION = 2


class CheckpointContractError(RuntimeError):
    """A permanent checkpoint failed validation, evaluation, or durability."""


def permanent_checkpoint_steps(total_steps: int, interval: int = 125) -> list[int]:
    """Return the permanent ladder: init, interval grid, and true final step.

    A last on-grid step strictly less than the final step is omitted when it is
    less than one interval from the final step.
    """
    total_steps = int(total_steps)
    interval = int(interval)
    if total_steps < 0:
        raise ValueError(f"total_steps must be >= 0, got {total_steps}")
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    if total_steps == 0:
        return [0]

    steps = {0, total_steps}
    last_grid = (total_steps // interval) * interval
    steps.update(range(interval, last_grid + 1, interval))
    if 0 < last_grid < total_steps and total_steps - last_grid < interval:
        steps.discard(last_grid)
    return sorted(steps)


def is_permanent_checkpoint_step(
    step: int, total_steps: int, interval: int = DEFAULT_CHECKPOINT_INTERVAL
) -> bool:
    return int(step) in permanent_checkpoint_steps(total_steps, interval)


def checkpointer_kwargs_for_ladder(
    total_steps: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    *,
    save_async: bool = False,
) -> dict[str, Any]:
    """Return OLMo-core ``CheckpointerCallback`` kwargs for this ladder."""
    final = int(total_steps)
    fixed_steps = [
        step for step in permanent_checkpoint_steps(final, interval) if step not in (0, final)
    ]
    return {
        "save_interval": None,
        "fixed_steps": fixed_steps,
        "ephemeral_save_interval": None,
        "pre_train_checkpoint": True,
        "save_async": bool(save_async),
        "max_checkpoints": None,
    }


def make_run_fingerprint(identity: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.loads(json.dumps(dict(identity), sort_keys=True))
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "identity": canonical,
        "identity_sha256": digest,
    }


def write_run_fingerprint(save_folder: str | Path, identity: Mapping[str, Any]) -> Path:
    root = Path(save_folder)
    root.mkdir(parents=True, exist_ok=True)
    target = root / RUN_FINGERPRINT_FILENAME
    payload = make_run_fingerprint(identity)
    _atomic_json_write(target, payload)
    return target


def read_run_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / RUN_FINGERPRINT_FILENAME
    if not source.is_file():
        raise CheckpointContractError(f"missing {RUN_FINGERPRINT_FILENAME}: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise CheckpointContractError(f"out-of-contract fingerprint (expected schema 2): {source}")
    if payload != make_run_fingerprint(payload.get("identity") or {}):
        raise CheckpointContractError(f"invalid or corrupted run fingerprint: {source}")
    return payload


def assert_resume_fingerprint(checkpoint_dir: str | Path, identity: Mapping[str, Any]) -> None:
    checkpoint = Path(checkpoint_dir)
    candidates = (
        checkpoint / RUN_FINGERPRINT_FILENAME,
        checkpoint.parent / RUN_FINGERPRINT_FILENAME,
    )
    prior_path = next((path for path in candidates if path.is_file()), None)
    if prior_path is None:
        raise CheckpointContractError(
            f"{checkpoint} is an out-of-contract checkpoint: no {RUN_FINGERPRINT_FILENAME}"
        )
    prior = read_run_fingerprint(prior_path)
    current = make_run_fingerprint(identity)
    if prior != current:
        old_identity = prior.get("identity") or {}
        new_identity = current["identity"]
        differing = sorted(
            key
            for key in set(old_identity) | set(new_identity)
            if old_identity.get(key) != new_identity.get(key)
        )
        raise CheckpointContractError(
            f"refusing resume with changed scientific identity (differing fields: {differing})"
        )


def copy_fingerprint_into_checkpoint(
    fingerprint_path: str | Path, checkpoint_dir: str | Path
) -> Path:
    source = Path(fingerprint_path)
    read_run_fingerprint(source)
    destination = Path(checkpoint_dir) / RUN_FINGERPRINT_FILENAME
    destination.write_bytes(source.read_bytes())
    return destination


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_last_durable_step(metadata_dir: str | Path) -> Optional[dict[str, Any]]:
    path = Path(metadata_dir) / LAST_DURABLE_STEP_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("last_durable_step"), int)
        or int(payload["last_durable_step"]) < 0
    ):
        raise CheckpointContractError(f"invalid durable-step metadata: {path}")
    return payload


def write_last_durable_step(
    metadata_dir: str | Path,
    step: int,
    *,
    checkpoint_artifact: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically advance local metadata after required uploads complete."""
    step = int(step)
    if step < 0:
        raise ValueError("last durable step must be non-negative")
    directory = Path(metadata_dir)
    target = directory / LAST_DURABLE_STEP_FILENAME
    current = read_last_durable_step(directory)
    if current is not None and step < int(current["last_durable_step"]):
        raise CheckpointContractError(
            "refusing to move last durable step backward "
            f"({current['last_durable_step']} -> {step})"
        )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "last_durable_step": step,
        "durability": "local_scratch+wandb",
    }
    if checkpoint_artifact:
        payload["checkpoint_artifact"] = str(checkpoint_artifact)
    if extra:
        payload.update(
            {
                str(key): value
                for key, value in extra.items()
                if key not in {"schema_version", "last_durable_step", "durability"}
            }
        )
    _atomic_json_write(target, payload)
    return target


def assert_checkpoint_materialized(checkpoint_dir: str | Path) -> Path:
    checkpoint = Path(checkpoint_dir)
    ready = (
        (checkpoint / "state.pt").is_file()
        or (checkpoint / "model_eval.pt").is_file()
        or (checkpoint / "model_and_optim" / ".metadata").is_file()
    )
    if not ready:
        raise CheckpointContractError(
            f"permanent checkpoint is not fully materialized: {checkpoint}"
        )
    return checkpoint


def finalize_permanent_checkpoint(
    *,
    arm: str,
    checkpoint_dir: str | Path,
    step: int,
    run_name: str,
    task_loss_dir: str | Path,
    task_loss_enabled: bool,
    eval_script: Optional[str | Path] = None,
    task_loss_nproc: Optional[int] = None,
    progress_dir: Optional[str | Path] = None,
    fingerprint_path: Optional[str | Path] = None,
    method: Optional[str] = None,
    wandb_run: Any | None = None,
    wandb_mode: Optional[str] = None,
    production: bool = False,
    run_evaluator: Optional[Callable[..., Any]] = None,
) -> Optional[dict[str, Any]]:
    """Evaluate and durably publish one already-materialized checkpoint."""
    from . import task_loss
    from . import wandb_artifacts as artifacts

    checkpoint = assert_checkpoint_materialized(checkpoint_dir)
    output = Path(task_loss_dir) / f"step{int(step)}_task_loss.json"
    strict_upload = artifacts.production_online(production=production, mode=wandb_mode)
    artifacts.require_wandb_for_production(wandb_run, production=production, mode=wandb_mode)
    if strict_upload and not task_loss_enabled:
        raise CheckpointContractError(
            "refusing to mark a production checkpoint durable without the "
            "complete 20-label task-loss suite"
        )

    payload: Optional[dict[str, Any]] = None
    if task_loss_enabled:
        evaluator = run_evaluator or task_loss.trigger_task_loss_eval
        evaluator(
            checkpoint,
            run_name=run_name,
            out_path=output,
            eval_script=eval_script,
            nproc=task_loss_nproc,
        )
        payload = task_loss.validate_task_loss_result(output)

    if fingerprint_path is not None:
        copy_fingerprint_into_checkpoint(fingerprint_path, checkpoint)

    artifact_ref = artifacts.checkpoint_artifact_ref(
        run_name=run_name,
        project=str(getattr(wandb_run, "project", "") or "edullm"),
        entity=str(getattr(wandb_run, "entity", "") or "") or None,
        alias=f"step-{int(step):07d}",
    )
    artifacts.wandb_log_checkpoint(
        wandb_run,
        checkpoint,
        step=int(step),
        extra_meta={"arm": arm, "method": method},
        strict=strict_upload,
        run_name=run_name,
    )
    if payload is not None:
        artifacts.wandb_log_eval(
            wandb_run,
            payload,
            step=int(step),
            eval_path=output,
            strict=strict_upload,
        )
    if progress_dir is not None:
        artifacts.wandb_log_directory_artifact(
            wandb_run,
            progress_dir,
            name=f"{run_name}-progress",
            artifact_type="metrics",
            strict=strict_upload,
        )
    if task_loss_enabled:
        artifacts.wandb_log_directory_artifact(
            wandb_run,
            task_loss_dir,
            name=f"{run_name}-task-loss",
            artifact_type="eval",
            strict=strict_upload,
        )

    marker_dir = Path(progress_dir) if progress_dir is not None else checkpoint.parent
    write_last_durable_step(
        marker_dir,
        int(step),
        checkpoint_artifact=artifact_ref if wandb_run is not None else None,
        extra={
            "run_name": run_name,
            "task_loss_complete": bool(task_loss_enabled),
            "task_loss_result": str(output) if task_loss_enabled else None,
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        },
    )
    return payload
