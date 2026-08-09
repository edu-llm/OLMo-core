"""Mirror ephemeral HPO artifacts and stats to the preregistered W&B project."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

__all__ = [
    "HPO_PROBE_PROJECT",
    "HpoProbeSession",
    "collect_controller_metrics",
    "durable_storage_roots",
    "storage_path_is_durable",
]

HPO_PROBE_PROJECT = "hpo-probe"
WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"


def durable_storage_roots(
    *,
    checkpoint_dir: Optional[str] = None,
    extra_roots: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Return filesystem prefixes that count as durable (typically S3-backed mounts)."""

    roots: list[str] = []
    for value in (
        os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        checkpoint_dir or "",
        *(extra_roots or ()),
    ):
        normalized = str(value).strip()
        if normalized:
            roots.append(os.path.normpath(normalized.rstrip("/")))
    # Preserve order while dropping duplicates.
    return tuple(dict.fromkeys(roots))


def storage_path_is_durable(path: str | Path, *, durable_roots: Sequence[str]) -> bool:
    """Return whether ``path`` is already persisted on durable storage."""

    from olmo_core.io import is_url

    text = str(path)
    if is_url(text):
        return True
    normalized = os.path.normpath(text)
    for root in durable_roots:
        if not root:
            continue
        try:
            if os.path.commonpath([normalized, root]) == root:
                return True
        except ValueError:
            continue
    return False


def collect_controller_metrics(controller, *, step: int) -> dict[str, float | int]:
    """Extract scalar controller stats suitable for ``wandb.log``."""

    state = controller.state()
    metrics: dict[str, float | int] = {
        "hpo/step": step,
        "hpo/tokens_charged": int(state.tokens_charged),
        "hpo/accelerator_seconds": float(state.accelerator_seconds_charged),
        "hpo/num_trials": len(state.trials),
        "hpo/log_events": len(controller.log),
    }
    top_candidates = getattr(controller, "top_candidates", None)
    if callable(top_candidates):
        try:
            best = top_candidates(1)
        except RuntimeError:
            best = []
        if best:
            metrics["hpo/best_search_validation_ce"] = float(best[0][2])
    return metrics


def _flatten_summary(prefix: str, value: Any, out: MutableMapping[str, Any]) -> None:
    if is_dataclass(value):
        _flatten_summary(prefix, asdict(value), out)
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            _flatten_summary(child, nested, out)
        return
    if isinstance(value, (list, tuple)):
        out[prefix] = value
        return
    out[prefix] = value


def study_result_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a study-result payload for ``run.summary``."""

    summary: dict[str, Any] = {}
    _flatten_summary("", dict(payload), summary)
    return summary


class HpoProbeSession:
    """Log non-durable HPO stats and artifacts to the ``hpo-probe`` W&B project."""

    def __init__(
        self,
        *,
        run_id: str,
        job_type: str,
        durable_roots: Sequence[str],
        arm: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        if WANDB_API_KEY_ENV_VAR not in os.environ:
            raise RuntimeError(
                f"missing env var {WANDB_API_KEY_ENV_VAR!r}; ephemeral HPO artifacts must be "
                f"mirrored to W&B project {HPO_PROBE_PROJECT!r}"
            )
        import wandb

        os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
        run_name = run_id if arm is None else f"{run_id}-{arm}"
        run_tags = list(tags or ())
        if arm is not None and arm not in run_tags:
            run_tags.append(arm)
        self._wandb = wandb
        self._run = wandb.init(
            project=HPO_PROBE_PROJECT,
            name=run_name,
            group=run_id,
            job_type=job_type,
            tags=run_tags or None,
            config=dict(config or {}),
        )
        self._durable_roots = tuple(durable_roots)
        self._step = 0
        self._finished = False
        self._mirrored_paths: set[str] = set()

    @classmethod
    def open(
        cls,
        *,
        run_id: str,
        job_type: str,
        durable_roots: Sequence[str],
        arm: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> "HpoProbeSession":
        """Create a probe session or fail closed when W&B is unavailable."""

        return cls(
            run_id=run_id,
            job_type=job_type,
            durable_roots=durable_roots,
            arm=arm,
            config=config,
            tags=tags,
        )

    def log_controller(self, controller) -> None:
        """Record the latest controller accounting snapshot."""

        metrics = collect_controller_metrics(controller, step=self._step)
        self._wandb.log(metrics, step=self._step)
        self._step += 1

    def record_study_result(self, payload: Mapping[str, Any], path: str | Path) -> None:
        """Mirror the winner artifact and its scalar summary."""

        self._run.summary.update(study_result_summary(payload))
        self._mirror_file(path, artifact_name="study-result", artifact_type="hpo-study-result")

    def record_proxy_cohort(
        self,
        payload: Mapping[str, Any],
        *,
        output_path: str | Path,
    ) -> None:
        """Mirror proxy-cohort metrics and any non-durable evidence file."""

        metrics = payload.get("metrics", {})
        if isinstance(metrics, Mapping):
            summary = {f"proxy/{key}": value for key, value in metrics.items()}
            summary["proxy/decision"] = payload.get("decision")
            self._run.summary.update(summary)
        self._mirror_file(
            output_path,
            artifact_name="proxy-evidence",
            artifact_type="hpo-proxy-evidence",
        )

    def mirror_ephemeral_path(
        self,
        path: str | Path,
        *,
        artifact_name: str,
        artifact_type: str,
    ) -> None:
        """Upload a single local file when it is not already on durable storage."""

        self._mirror_file(path, artifact_name=artifact_name, artifact_type=artifact_type)

    def mirror_ephemeral_directory(
        self,
        path: str | Path,
        *,
        artifact_name: str,
        artifact_type: str,
    ) -> None:
        """Upload a local directory tree when it is not already on durable storage."""

        directory = Path(path)
        if not directory.exists():
            return
        normalized = os.path.normpath(str(directory))
        if storage_path_is_durable(normalized, durable_roots=self._durable_roots):
            return
        if normalized in self._mirrored_paths:
            return
        artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_dir(normalized)
        self._run.log_artifact(artifact)
        self._mirrored_paths.add(normalized)

    def close(self, *, exit_code: int = 0) -> None:
        """Finish the W&B run once."""

        if self._finished:
            return
        self._wandb.finish(exit_code=exit_code, quiet=True)
        self._finished = True

    def _mirror_file(self, path: str | Path, *, artifact_name: str, artifact_type: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            return
        normalized = os.path.normpath(str(file_path))
        if storage_path_is_durable(normalized, durable_roots=self._durable_roots):
            return
        if normalized in self._mirrored_paths:
            return
        artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(normalized)
        self._run.log_artifact(artifact)
        self._mirrored_paths.add(normalized)
