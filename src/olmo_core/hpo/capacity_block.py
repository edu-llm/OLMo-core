"""Capacity-block orchestration for independent single-node HPO trials."""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import math
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .types import WorkerObservation

PLATFORM_REPOSITORY = "edu-llm/platform"
STATUS_WORKFLOW = "block-status.yml"
RUN_WORKFLOW = "block-run.yml"
LOGS_WORKFLOW = "block-logs.yml"
OBSERVATION_MARKER = "EDULLM_HPO_OBSERVATION="


class WorkflowGateway(Protocol):
    """Minimal GitHub workflow interface used by the backend."""

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> str:
        """Dispatch a workflow and return its GitHub run id."""

    def wait(self, run_id: str) -> str:
        """Wait for a workflow and return its complete log."""


@dataclass(frozen=True)
class CapacityBlockConfig:
    """Configuration shared by all capacity-block waves."""

    branch: str
    checkpoint_root: str
    repository: str = "edu-llm/OLMo-core"
    platform_repository: str = PLATFORM_REPOSITORY
    max_workers: int = 8
    worker_world_size: int = 8
    wandb_project: str = "hpo-probe"
    reservation_id: str = ""
    region: str = "us-east-2"
    outputs_bucket: str = "edullm-block-outputs-us-east-2"
    poll_interval_seconds: float = 30.0
    observation_sync_attempts: int = 4

    def __post_init__(self) -> None:
        if not self.branch or not self.repository:
            raise ValueError("capacity-block branch and repository must be non-empty")
        if not self.checkpoint_root.startswith("s3://"):
            raise ValueError("capacity-block checkpoints must use a shared s3:// root")
        if self.max_workers < 1:
            raise ValueError("capacity-block max_workers must be positive")
        if self.worker_world_size != 8:
            raise ValueError("capacity-block HPO requires one eight-GPU worker per node")
        if self.poll_interval_seconds < 0:
            raise ValueError("capacity-block poll interval cannot be negative")
        if self.observation_sync_attempts < 1:
            raise ValueError("capacity-block observation sync attempts must be positive")


@dataclass(frozen=True)
class CapacityTrial:
    """One controller allocation ready for remote execution."""

    trial_id: str
    decision_id: int
    target_fidelity: int
    payload: Mapping[str, Any]


def parse_idle_nodes(log_text: str) -> list[int]:
    """Extract only explicitly ``IDLE`` nodes from a block-status workflow log."""

    reading_pattern = re.compile(r"\bnode\s+(\d+)\s{2,}.*$", re.MULTILINE)
    idle_pattern = re.compile(r"\bnode\s+(\d+)\s{2,}.*\bIDLE\s*$", re.MULTILINE)
    if not reading_pattern.search(log_text):
        raise RuntimeError("block-status workflow emitted no node readings")
    idle = [int(match.group(1)) for match in idle_pattern.finditer(log_text)]
    if len(idle) != len(set(idle)):
        raise RuntimeError("block-status workflow emitted a duplicate IDLE node")
    return sorted(idle)


def _parse_observation(log_text: str) -> Mapping[str, Any]:
    payloads: list[Mapping[str, Any]] = []
    for line in log_text.splitlines():
        marker = line.find(OBSERVATION_MARKER)
        if marker < 0:
            continue
        raw = line[marker + len(OBSERVATION_MARKER) :].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("capacity-block log contains a malformed observation") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("capacity-block observation must be a JSON object")
        payloads.append(payload)
    if len(payloads) != 1:
        raise RuntimeError(
            f"capacity-block log contains {len(payloads)} durable observations; expected one"
        )
    return payloads[0]


class GhWorkflowGateway:
    """GitHub CLI implementation with fail-closed workflow-run correlation."""

    def __init__(
        self,
        *,
        repository: str = PLATFORM_REPOSITORY,
        ref: str = "main",
        runner: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
        dispatch_timeout_seconds: float = 60.0,
    ) -> None:
        self.repository = repository
        self.ref = ref
        self._runner = runner
        self._sleep = sleep
        self._dispatch_timeout_seconds = dispatch_timeout_seconds
        self._dispatch_lock = threading.Lock()
        self._actor_login: str | None = None

    def _run(self, argv: Sequence[str]) -> str:
        completed = self._runner(
            list(argv),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{' '.join(argv[:3])} failed ({completed.returncode}): "
                f"{completed.stderr.strip()[-4000:]}"
            )
        return str(completed.stdout)

    def _actor(self) -> str:
        if self._actor_login is None:
            login = self._run(["gh", "api", "user", "--jq", ".login"]).strip()
            if not login:
                raise RuntimeError("GitHub did not identify the authenticated actor")
            self._actor_login = login
        return self._actor_login

    def _run_ids(self, workflow: str, *, actor: str) -> set[int]:
        raw = self._run(
            [
                "gh",
                "run",
                "list",
                "-R",
                self.repository,
                "--workflow",
                workflow,
                "--event",
                "workflow_dispatch",
                "--user",
                actor,
                "--limit",
                "100",
                "--json",
                "databaseId",
            ]
        )
        try:
            values = json.loads(raw)
            return {int(value["databaseId"]) for value in values}
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("GitHub returned malformed workflow run metadata") from exc

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> str:
        # workflow_dispatch has no response body containing the run id. Serialize this short
        # correlation window so concurrent node launches cannot claim each other's run.
        with self._dispatch_lock:
            actor = self._actor()
            previous = self._run_ids(workflow, actor=actor)
            argv = [
                "gh",
                "workflow",
                "run",
                workflow,
                "--ref",
                self.ref,
                "-R",
                self.repository,
            ]
            for key, value in sorted(inputs.items()):
                argv.extend(["-f", f"{key}={value}"])
            self._run(argv)
            deadline = time.monotonic() + self._dispatch_timeout_seconds
            while True:
                created = self._run_ids(workflow, actor=actor) - previous
                if len(created) == 1:
                    return str(created.pop())
                if len(created) > 1:
                    raise RuntimeError(
                        f"could not safely correlate concurrent {workflow} workflow runs"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"GitHub did not expose the dispatched {workflow} run")
                self._sleep(1.0)

    def wait(self, run_id: str) -> str:
        watched = self._runner(
            ["gh", "run", "watch", run_id, "-R", self.repository, "--exit-status"],
            text=True,
            capture_output=True,
            check=False,
        )
        logs = self._run(["gh", "run", "view", run_id, "-R", self.repository, "--log"])
        if watched.returncode != 0:
            raise RuntimeError(
                f"GitHub workflow run {run_id} failed ({watched.returncode}): "
                f"{logs.strip()[-8000:]}"
            )
        return logs


class CapacityBlockBackend:
    """Dispatch controller allocations as concurrent one-node workflow runs."""

    def __init__(
        self,
        config: CapacityBlockConfig,
        gateway: WorkflowGateway,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self.gateway = gateway
        self._sleep = sleep
        self._clock = clock

    def discover_idle_nodes(self) -> list[int]:
        inputs = {"region": self.config.region}
        if self.config.reservation_id:
            inputs["reservation_id"] = self.config.reservation_id
        run_id = self.gateway.dispatch(STATUS_WORKFLOW, inputs)
        return parse_idle_nodes(self.gateway.wait(run_id))

    def worker_count(self, idle_nodes: Sequence[int]) -> int:
        return min(self.config.max_workers, len(idle_nodes))

    def wait_for_idle_nodes(
        self,
        *,
        required: int = 1,
        heartbeat: Callable[[], None] | None = None,
    ) -> list[int]:
        if required < 1:
            raise ValueError("required capacity must be positive")
        while True:
            idle = self.discover_idle_nodes()
            if len(idle) >= required:
                return idle
            if heartbeat is not None:
                heartbeat()
            self._sleep(self.config.poll_interval_seconds)

    def _run_name(self, trial: CapacityTrial) -> str:
        raw = f"hpo-{trial.decision_id}-{trial.trial_id}"
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", raw)
        if len(safe) <= 64:
            return safe
        digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:12]
        return f"{safe[:51]}-{digest}"

    def _observation_path(self, trial: CapacityTrial) -> str:
        return (
            f"{self.config.checkpoint_root.rstrip('/')}/observations/"
            f"decision-{trial.decision_id}-{trial.trial_id}.json"
        )

    def _command(self, run_id: str, trial: CapacityTrial) -> str:
        payload_root = str(trial.payload.get("checkpoint_root", ""))
        if payload_root != self.config.checkpoint_root:
            raise ValueError("segment payload checkpoint root does not match capacity backend")
        encoded = base64.urlsafe_b64encode(
            json.dumps(trial.payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        checkpoint_dir = f"{self.config.checkpoint_root.rstrip('/')}/trials/{trial.trial_id}"
        argv = [
            "python",
            ".edullm/hpo_on_corpus.py",
            run_id,
            "--run-segment",
            f"--worker-world-size={self.config.worker_world_size}",
            f"--trial-id={trial.trial_id}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--hard-stop-tokens={trial.target_fidelity}",
            "--param-dtype=bfloat16",
            f"--segment-spec-payload={encoded}",
            f"--observation-path={self._observation_path(trial)}",
        ]
        command = shlex.join(argv)
        forbidden = (
            "torchrun",
            "torch.distributed",
            "--moe-shard-degree",
            "--moe-num-replicas",
        )
        if any(value in command for value in forbidden):
            raise ValueError(
                "capacity-block worker command contains a forbidden launcher or mesh flag"
            )
        return command

    def _run_inputs(self, run_id: str, node: int, trial: CapacityTrial) -> dict[str, str]:
        inputs = {
            "node": str(node),
            "branch": self.config.branch,
            "run_name": self._run_name(trial),
            "repository": self.config.repository,
            "processes": "all",
            "wandb_project": self.config.wandb_project,
            "region": self.config.region,
            "command": self._command(run_id, trial),
        }
        if self.config.reservation_id:
            inputs["reservation_id"] = self.config.reservation_id
        return inputs

    def _logs_inputs(self, node: int, trial: CapacityTrial) -> dict[str, str]:
        inputs = {
            "node": str(node),
            "run_name": self._run_name(trial),
            "lines": "400",
            "region": self.config.region,
            "outputs_bucket": self.config.outputs_bucket,
        }
        if self.config.reservation_id:
            inputs["reservation_id"] = self.config.reservation_id
        return inputs

    def _wait_many(self, run_ids: Sequence[str]) -> list[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(run_ids)) as executor:
            futures = [executor.submit(self.gateway.wait, run_id) for run_id in run_ids]
            return [future.result() for future in futures]

    def _collect_log(self, node: int, trial: CapacityTrial) -> str:
        latest = ""
        for attempt in range(self.config.observation_sync_attempts):
            workflow = self.gateway.dispatch(LOGS_WORKFLOW, self._logs_inputs(node, trial))
            latest = self.gateway.wait(workflow)
            lowered = latest.lower()
            if (
                OBSERVATION_MARKER in latest
                or "out of memory" in lowered
                or "outofmemoryerror" in lowered
            ):
                return latest
            if attempt + 1 < self.config.observation_sync_attempts:
                self._sleep(self.config.poll_interval_seconds)
        return latest

    def _result(
        self,
        trial: CapacityTrial,
        log_text: str,
        *,
        accelerator_seconds: float,
    ) -> WorkerObservation:
        try:
            payload = _parse_observation(log_text)
        except RuntimeError:
            lowered = log_text.lower()
            if "out of memory" not in lowered and "outofmemoryerror" not in lowered:
                raise
            return WorkerObservation(
                trial_id=trial.trial_id,
                tokens=trial.target_fidelity,
                heldout_ce=float("nan"),
                train_ce_history=(float("nan"),),
                grad_norm_history=(float("nan"),),
                activation_ratio=None,
                numeric_failure=True,
                checkpoint_ref=None,
                accelerator_seconds=accelerator_seconds,
            )
        if str(payload.get("trial_id")) != trial.trial_id:
            raise RuntimeError("capacity-block observation belongs to a different trial")
        heldout = payload.get("heldout_ce")
        heldout_ce = float("nan") if heldout is None else float(heldout)
        return WorkerObservation(
            trial_id=trial.trial_id,
            tokens=int(payload["tokens"]),
            heldout_ce=heldout_ce,
            train_ce_history=tuple(float(value) for value in payload["train_ce_history"]),
            grad_norm_history=tuple(float(value) for value in payload["grad_norm_history"]),
            activation_ratio=(
                None
                if payload.get("activation_ratio") is None
                else float(payload["activation_ratio"])
            ),
            numeric_failure=bool(payload["numeric_failure"]),
            checkpoint_ref=payload.get("checkpoint_ref"),
            accelerator_seconds=accelerator_seconds,
        )

    def run(
        self,
        trials: Sequence[CapacityTrial],
        *,
        run_id: str = "hpo-run",
        idle_nodes: Sequence[int] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> list[WorkerObservation]:
        if not trials:
            return []
        available = (
            tuple(idle_nodes)
            if idle_nodes is not None
            else self.wait_for_idle_nodes(heartbeat=heartbeat)
        )
        count = self.worker_count(available)
        if count == 0:
            available = self.wait_for_idle_nodes(heartbeat=heartbeat)
            count = self.worker_count(available)
        if len(trials) > count:
            completed: list[WorkerObservation] = []
            remaining = list(trials)
            while remaining:
                count = self.worker_count(available)
                wave = remaining[:count]
                completed.extend(
                    self.run(
                        wave,
                        run_id=run_id,
                        idle_nodes=available,
                        heartbeat=heartbeat,
                    )
                )
                del remaining[:count]
                if remaining:
                    available = self.wait_for_idle_nodes(heartbeat=heartbeat)
            return completed
        selected = available[: len(trials)]
        started_at: dict[str, float] = {}
        run_workflows: list[str] = []
        for node, trial in zip(selected, trials):
            started_at[trial.trial_id] = self._clock()
            run_workflows.append(
                self.gateway.dispatch(RUN_WORKFLOW, self._run_inputs(run_id, node, trial))
            )

        # All runs are dispatched before any one startup workflow is awaited. Per-node
        # concurrency groups then allow them to claim distinct nodes in parallel.
        self._wait_many(run_workflows)
        while True:
            now_idle = set(self.discover_idle_nodes())
            if set(selected).issubset(now_idle):
                break
            if heartbeat is not None:
                heartbeat()
            self._sleep(self.config.poll_interval_seconds)

        finished_at = {trial.trial_id: self._clock() for trial in trials}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(trials)) as executor:
            log_futures = [
                executor.submit(self._collect_log, node, trial)
                for node, trial in zip(selected, trials)
            ]
            logs = [future.result() for future in log_futures]
        results = []
        for trial, log_text in zip(trials, logs):
            elapsed = max(0.0, finished_at[trial.trial_id] - started_at[trial.trial_id])
            accelerator_seconds = elapsed * self.config.worker_world_size
            if not math.isfinite(accelerator_seconds):
                raise RuntimeError("capacity-block trial duration was not finite")
            results.append(
                self._result(
                    trial,
                    log_text,
                    accelerator_seconds=accelerator_seconds,
                )
            )
        return results
