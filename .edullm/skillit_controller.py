"""Checkpoint-gated Skill-It controller for the two 370M arms."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional

import numpy as np
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, is_distributed
from olmo_core.train.callbacks import Callback

from production_contract.task_loss import validate_task_loss_result
from production_contract.wandb_artifacts import (
    production_online,
    wandb_log_directory_artifact,
    wandb_run_from_trainer,
)
from skillit_loader import WeightedDomainDataLoader
from skillit_math import (
    DOMAINS,
    ETA,
    FAMILIES,
    UPDATE_STEPS,
    W,
    adjacency,
    family_losses,
    update_weights,
)


class SkillItControllerError(RuntimeError):
    """A checkpoint-gated update was incomplete, stale, or inconsistent."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SkillItControllerError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise SkillItControllerError(f"{path}:{line_number}: record is not an object")
        output.append(record)
    return output


@dataclass
class SkillItController(Callback):
    """Apply exact Skill-It math only after the checkpoint's strict task-loss suite."""

    priority: ClassVar[int] = -1

    arm_id: str
    a_mode: str
    progress_dir: str
    task_loss_dir: str
    production: bool = True
    wandb_mode: str = "online"
    update_steps: tuple[int, ...] = UPDATE_STEPS
    _applied_steps: set[int] = field(default_factory=set)

    @property
    def loader(self) -> WeightedDomainDataLoader:
        loader = self.trainer.data_loader
        if not isinstance(loader, WeightedDomainDataLoader):
            raise SkillItControllerError("SkillItController requires WeightedDomainDataLoader")
        return loader

    @property
    def progress(self) -> Path:
        return Path(self.progress_dir)

    @property
    def task_losses(self) -> Path:
        return Path(self.task_loss_dir)

    @property
    def updates_jsonl(self) -> Path:
        return self.progress / "skillit_updates.jsonl"

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "arm_id": self.arm_id,
            "a_mode": self.a_mode,
            "applied_steps": sorted(self._applied_steps),
            "weights": self.loader.weights_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("schema") != 1:
            raise SkillItControllerError("unsupported controller checkpoint schema")
        if state_dict.get("arm_id") != self.arm_id or state_dict.get("a_mode") != self.a_mode:
            raise SkillItControllerError("controller checkpoint belongs to another arm")
        self._applied_steps = {int(step) for step in state_dict.get("applied_steps", [])}
        if "weights" in state_dict:
            self.loader.set_weights(state_dict["weights"])

    def post_checkpoint_loaded(self, path: str) -> None:
        del path
        records = [
            record
            for record in _records(self.updates_jsonl)
            if int(record.get("step", -1)) <= int(self.step)
        ]
        if not records:
            return
        latest = max(records, key=lambda record: int(record["step"]))
        self._validate_record(latest, int(latest["step"]))
        if int(latest["step"]) in self.update_steps:
            self._applied_steps.add(int(latest["step"]))

    def pre_train(self) -> None:
        if get_rank() == 0:
            self.progress.mkdir(parents=True, exist_ok=True)
        if is_distributed():
            dist.barrier()
        if self.step == 0:
            self._write_baseline_once()
        if self.step in self.update_steps and self.step not in self._applied_steps:
            self._apply_update(self.step)

    def post_step(self) -> None:
        if self.step in self.update_steps and self.step not in self._applied_steps:
            self._apply_update(self.step)
        for domain, value in self.loader.weights_dict().items():
            self.trainer.record_metric(f"skillit/weight/{domain}", value)

    def _write_baseline_once(self) -> None:
        existing = [record for record in _records(self.updates_jsonl) if record.get("step") == 0]
        if existing:
            return
        weights = self.loader.weights
        matrix = adjacency(self.a_mode, weights)
        record = self._record(
            step=0,
            matrix=matrix,
            p_before=weights,
            p_after=weights,
            losses=None,
            note="baseline RegMix weights; no Skill-It update",
        )
        self._persist_rank0(record)

    def _apply_update(self, step: int) -> None:
        record: Optional[dict[str, Any]] = None
        failure: Optional[str] = None
        if get_rank() == 0:
            try:
                existing = [
                    item
                    for item in _records(self.updates_jsonl)
                    if int(item.get("step", -1)) == int(step)
                ]
                if existing:
                    record = existing[-1]
                    self._validate_record(record, step)
                else:
                    result_path = self.task_losses / f"step{int(step)}_task_loss.json"
                    payload = validate_task_loss_result(result_path)
                    if int(payload.get("step", -1)) != int(step):
                        raise SkillItControllerError(
                            f"{result_path}: stale task-loss step {payload.get('step')!r}"
                        )
                    losses_by_family = family_losses(payload)
                    p_before = self.loader.weights
                    matrix = adjacency(self.a_mode, p_before)
                    p_after = update_weights(
                        matrix,
                        [losses_by_family[family] for family in FAMILIES],
                        eta=ETA,
                        w=W,
                    )
                    record = self._record(
                        step=step,
                        matrix=matrix,
                        p_before=p_before,
                        p_after=p_after,
                        losses=losses_by_family,
                    )
                    self._persist_rank0(record)
                strict = production_online(production=self.production, mode=self.wandb_mode)
                wandb_log_directory_artifact(
                    wandb_run_from_trainer(self.trainer),
                    self.progress,
                    name=f"skillit-{self.arm_id}-state",
                    artifact_type="method-state",
                    aliases=["latest", f"step-{int(step):07d}"],
                    strict=strict,
                )
            except BaseException as exc:  # noqa: BLE001
                failure = f"{type(exc).__name__}: {exc}"

        message: list[Any] = [record, failure]
        if is_distributed():
            dist.broadcast_object_list(message, src=0)
        record, failure = message
        if failure is not None:
            raise SkillItControllerError(
                f"Skill-It update step {step} failed after checkpoint evaluation: {failure}"
            )
        if record is None:
            raise SkillItControllerError(f"Skill-It update step {step} produced no state")
        self.loader.set_weights(record["p_after"])
        self._applied_steps.add(int(step))

    def _validate_record(self, record: Mapping[str, Any], step: int) -> None:
        if (
            int(record.get("step", -1)) != int(step)
            or record.get("arm_id") != self.arm_id
            or record.get("a_mode") != self.a_mode
            or tuple(record.get("domain_order") or ()) != DOMAINS
            or tuple(record.get("family_order") or ()) != FAMILIES
        ):
            raise SkillItControllerError(
                f"existing Skill-It update record is inconsistent at step {step}"
            )
        matrix = np.asarray(record.get("A"), dtype=np.float64)
        if matrix.shape != (len(DOMAINS), len(FAMILIES)) or not np.isfinite(matrix).all():
            raise SkillItControllerError(f"existing A is invalid at step {step}")
        self.loader.set_weights(record["p_after"])

    def _record(
        self,
        *,
        step: int,
        matrix: np.ndarray,
        p_before: np.ndarray,
        p_after: np.ndarray,
        losses: Optional[Mapping[str, float]],
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step": int(step),
            "arm_id": self.arm_id,
            "a_mode": self.a_mode,
            "eta": ETA,
            "w": W,
            "domain_order": list(DOMAINS),
            "family_order": list(FAMILIES),
            "A": np.asarray(matrix, dtype=np.float64).tolist(),
            "p_before": {domain: float(p_before[i]) for i, domain in enumerate(DOMAINS)},
            "p_after": {domain: float(p_after[i]) for i, domain in enumerate(DOMAINS)},
        }
        if losses is not None:
            record["losses"] = {family: float(losses[family]) for family in FAMILIES}
        if self.a_mode == "derivative":
            record["r"] = dict(record["p_before"])
        if note:
            record["note"] = note
        return record

    def _persist_rank0(self, record: Mapping[str, Any]) -> None:
        if get_rank() != 0:
            return
        step = int(record["step"])
        updates_dir = self.progress / "skillit_updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        a_payload = {
            "step": step,
            "arm_id": self.arm_id,
            "a_mode": self.a_mode,
            "domain_order": list(DOMAINS),
            "family_order": list(FAMILIES),
            "A": record["A"],
        }
        weights_payload = {
            key: record[key] for key in ("step", "arm_id", "a_mode", "p_before", "p_after")
        }
        for optional in ("losses", "r", "note"):
            if optional in record:
                weights_payload[optional] = record[optional]
        _atomic_json(updates_dir / f"step{step}_A.json", a_payload)
        _atomic_json(updates_dir / f"step{step}_weights.json", weights_payload)
        with self.updates_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


__all__ = ["SkillItController", "SkillItControllerError"]
