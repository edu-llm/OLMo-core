"""Fail-closed W&B artifact helpers for eduLLM production runs."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .task_loss import task_loss_metrics

log = logging.getLogger(__name__)

DEFAULT_WANDB_MODE = "online"

try:
    import wandb as _wandb
except ImportError:  # pragma: no cover - W&B is optional for local tests.
    _wandb = None  # type: ignore[assignment]


class WandbArtifactError(RuntimeError):
    """A required W&B artifact upload or restore failed."""


def production_online(*, production: bool, mode: Optional[str] = None) -> bool:
    resolved = (mode or os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE)).strip().lower()
    return bool(production) and resolved == "online"


def require_wandb_for_production(
    run: Any | None, *, production: bool, mode: Optional[str] = None
) -> None:
    if production_online(production=production, mode=mode) and run is None:
        raise WandbArtifactError(
            "production online runs require an initialized W&B run; "
            "checkpoint durability is fail-closed"
        )


def _artifact_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return slug or "edullm"


def checkpoint_artifact_name(run_name: str) -> str:
    return f"{_artifact_slug(run_name)}-checkpoint"


def checkpoint_artifact_ref(
    *,
    run_name: str,
    project: str,
    entity: Optional[str] = None,
    alias: str = "latest",
) -> str:
    prefix = f"{entity}/" if entity else ""
    return f"{prefix}{project}/{checkpoint_artifact_name(run_name)}:{_artifact_slug(alias)}"


def _wait_for_upload(logged: Any, *, description: str, strict: bool) -> None:
    if not strict:
        return
    if logged is None:
        raise WandbArtifactError(f"required W&B {description} upload returned no handle")
    wait = getattr(logged, "wait", None)
    if not callable(wait):
        raise WandbArtifactError(f"required W&B {description} upload cannot be confirmed")
    try:
        wait()
    except Exception as exc:  # noqa: BLE001
        raise WandbArtifactError(f"required W&B {description} upload did not complete") from exc


def wandb_log_checkpoint(
    run: Any | None,
    checkpoint_dir: str | Path,
    *,
    step: int,
    tokens_seen: Optional[int] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    run_name: Optional[str] = None,
) -> Any | None:
    if run is None:
        if strict:
            raise WandbArtifactError("required W&B checkpoint upload has no active run")
        return None
    if _wandb is None:
        if strict:
            raise WandbArtifactError("wandb is unavailable for checkpoint upload")
        return None
    source = Path(checkpoint_dir)
    if not source.is_dir():
        if strict:
            raise WandbArtifactError(f"checkpoint artifact source does not exist: {source}")
        return None

    metadata: dict[str, Any] = {"step": int(step)}
    if tokens_seen is not None:
        metadata["tokens_seen"] = int(tokens_seen)
    if extra_meta:
        metadata.update({key: value for key, value in extra_meta.items() if value is not None})
    artifact = _wandb.Artifact(
        name=checkpoint_artifact_name(run_name or str(getattr(run, "name", "") or "edullm")),
        type="model",
        metadata=metadata,
    )
    artifact.add_dir(str(source))
    aliases = ["latest", f"step-{int(step):07d}"]
    try:
        logged = run.log_artifact(artifact, aliases=aliases)
        _wait_for_upload(logged, description="checkpoint", strict=strict)
    except WandbArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise WandbArtifactError(
                f"required W&B checkpoint upload failed for step {int(step)}"
            ) from exc
        log.warning("W&B checkpoint upload failed for step %s: %s", step, exc)
        return None
    return logged


def wandb_log_eval(
    run: Any | None,
    payload: Mapping[str, Any],
    *,
    step: int,
    eval_path: Optional[str | Path] = None,
    strict: bool = False,
) -> Any | None:
    if run is None:
        if strict:
            raise WandbArtifactError("required W&B eval upload has no active run")
        return None
    run.log(task_loss_metrics(payload), step=int(step))
    if _wandb is None or eval_path is None:
        if strict:
            raise WandbArtifactError("wandb or eval artifact path is unavailable")
        return None
    source = Path(eval_path)
    if not source.is_file():
        if strict:
            raise WandbArtifactError(f"eval artifact source does not exist: {source}")
        return None
    artifact = _wandb.Artifact(
        name=f"eval-step{int(step):07d}",
        type="eval",
    )
    artifact.add_file(str(source), name=source.name)
    try:
        logged = run.log_artifact(artifact, aliases=[f"step-{int(step):07d}"])
        _wait_for_upload(logged, description="eval", strict=strict)
    except WandbArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise WandbArtifactError(
                f"required W&B eval upload failed for step {int(step)}"
            ) from exc
        log.warning("W&B eval upload failed for step %s: %s", step, exc)
        return None
    return logged


def wandb_log_directory_artifact(
    run: Any | None,
    path: str | Path,
    *,
    name: str,
    artifact_type: str,
    aliases: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> Any | None:
    source = Path(path)
    if run is None or _wandb is None:
        if strict:
            raise WandbArtifactError(f"required W&B {artifact_type} upload has no active run")
        return None
    if not source.exists():
        if strict:
            raise WandbArtifactError(f"artifact source does not exist: {source}")
        return None
    artifact = _wandb.Artifact(
        name=_artifact_slug(name),
        type=artifact_type,
    )
    if source.is_dir():
        artifact.add_dir(str(source))
    else:
        artifact.add_file(str(source), name=source.name)
    try:
        logged = run.log_artifact(artifact, aliases=list(aliases or ["latest"]))
        _wait_for_upload(logged, description=artifact_type, strict=strict)
    except WandbArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise WandbArtifactError(f"required W&B {artifact_type} upload failed") from exc
        log.warning("W&B %s upload failed: %s", artifact_type, exc)
        return None
    return logged


def restore_checkpoint_artifact(
    artifact_ref: str,
    save_folder: str | Path,
    *,
    api: Any | None = None,
    require_fingerprint: bool = True,
) -> Path:
    """Restore into an absent ``stepN`` directory, refusing ambiguous overwrite."""
    if _wandb is None and api is None:
        raise WandbArtifactError("wandb is required to restore a checkpoint")
    reference = str(artifact_ref).strip()
    if not reference:
        raise WandbArtifactError("a non-empty W&B artifact reference is required")
    root = Path(save_folder)
    root.mkdir(parents=True, exist_ok=True)
    client = api or _wandb.Api()
    try:
        artifact = client.artifact(reference, type="model")
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        step = int(metadata["step"])
        target = root / f"step{step}"
        if target.exists():
            raise WandbArtifactError(
                f"refusing to overwrite existing checkpoint during restore: {target}"
            )
        with tempfile.TemporaryDirectory(prefix=".wandb-restore-", dir=str(root)) as temporary:
            downloaded = Path(artifact.download(root=temporary))
            shutil.copytree(downloaded, target)
    except WandbArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise WandbArtifactError(
            f"failed to restore W&B checkpoint artifact {reference!r}"
        ) from exc
    fingerprint = target / "run_fingerprint.json"
    if require_fingerprint and not fingerprint.is_file():
        shutil.rmtree(target)
        raise WandbArtifactError(f"W&B artifact {reference!r} is missing run_fingerprint.json")
    if fingerprint.is_file():
        shutil.copy2(fingerprint, root / "run_fingerprint.json")
    return target


def wandb_run_from_trainer(trainer: Any) -> Any | None:
    callbacks = getattr(trainer, "callbacks", None) or {}
    if isinstance(callbacks, Mapping):
        candidates = callbacks.values()
    else:
        candidates = ()
    for callback in candidates:
        for attribute in ("run", "_run", "wandb_run"):
            run = getattr(callback, attribute, None)
            if run is not None:
                return run
    return getattr(_wandb, "run", None) if _wandb is not None else None
