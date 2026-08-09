"""
One-trial-per-GPU OLMo worker: launch topology, checkpoint namespaces, segment scheduling,
resume guards, and bounded diagnostics.

OLMo has no in-process pause API, so freeze/thaw is *checkpoint + process exit + a freshly built
trainer that loads that checkpoint*. The controller runs CPU-only and never initializes one
global process group; it normally spawns an isolated subprocess per segment with ``WORLD_SIZE=1``.
An explicitly marked finalist continuation may instead use one multi-rank FSDP subprocess. This
module owns the pieces of that decision:

- :func:`world_size_one_env` / :func:`finalist_distributed_env` -- isolated worker launch
  environments.
- :func:`assert_worker_topology` -- a guard that permits multiple ranks only for the explicit
  finalist continuation.
- :func:`trial_checkpoint_dir` and friends -- per-trial checkpoint namespaces so two trials
  never collide on a flat ``step{N}`` directory, and latest-checkpoint lookup scoped to one
  trial only.
- :func:`next_absolute_hard_stop` -- refreshes ``hard_stop`` to a fresh absolute token ceiling
  strictly above the loaded token count before every segment, so a stale stop can never produce
  a zero-step segment while the full LR horizon is preserved.
- :func:`build_hpo_scheduler` / :class:`TrialConfigArtifact` / :func:`reconstruct_scheduler` --
  a token-unit, fraction-based schedule reconstructed from a hash-verified immutable artifact.
- :func:`assert_resume_batch_size` -- fail closed if a resumed global batch size differs.
- :class:`HpoDiagnosticsCallback` -- a stateful :class:`~olmo_core.train.callbacks.Callback`
  that emits bounded telemetry (held-out/train CE, grad summaries, numeric-failure flag, token
  progress) for BTTackler and FT-PFN.

Importing this module pulls in ``torch``/``olmo_core.train`` because the callback subclasses the
real :class:`Callback`; the controller-side pure logic lives elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from ..io import join_path, list_directory
from ..optim.scheduler import WSD, SchedulerUnits
from ..train.callbacks import Callback, EvaluatorCallback, LMEvaluatorCallbackConfig
from ..train.common import Duration
from .objective import EvaluatorGate
from .proxy import FrozenLayerProxy
from .types import WorkerObservation

__all__ = [
    "SegmentComplete",
    "SegmentSpec",
    "execute_segment",
    "BatchSizeMismatch",
    "world_size_one_env",
    "finalist_distributed_env",
    "assert_single_process_topology",
    "assert_worker_topology",
    "should_emit_worker_result",
    "trial_namespace",
    "trial_checkpoint_dir",
    "controller_dir",
    "latest_step_dir",
    "next_absolute_hard_stop",
    "build_hpo_scheduler",
    "configure_hpo_experiment",
    "validate_umup_model",
    "assert_resume_batch_size",
    "TrialConfigArtifact",
    "reconstruct_scheduler",
    "WorkerConfig",
    "HpoDiagnosticsCallback",
    "ReconcileResult",
    "reconcile_trial",
]


class SegmentComplete(Exception):
    """Raised when a lineage has already reached its target token horizon."""


class BatchSizeMismatch(Exception):
    """Raised when a resumed global batch size differs from the lineage config."""


@dataclass(frozen=True)
class SegmentSpec:
    """Runtime invariants for one allocation-boundary training segment."""

    trial_id: str
    target_tokens: int
    hard_stop_tokens: int
    lineage_global_batch_size: int
    transition: Optional[Mapping[str, Any]] = None
    search_validation_callback: str = "search_validation"


def execute_segment(
    trainer,
    *,
    diagnostics: "HpoDiagnosticsCallback",
    spec: SegmentSpec,
    actual_global_batch_size: int,
) -> WorkerObservation:
    """Load, train, save, and return one typed allocation-boundary observation."""
    assert_resume_batch_size(
        spec.lineage_global_batch_size,
        actual_global_batch_size,
    )
    transition = spec.transition
    fresh_parameters = None
    if transition is not None and transition.get("weight_policy") == "shrink_perturb":
        fresh_parameters = [
            parameter.detach().clone() for parameter in trainer.train_module.model.parameters()
        ]

    loaded = trainer.maybe_load_checkpoint()
    load_path = getattr(trainer, "load_path", None)
    if not loaded and load_path is not None:
        trainer.maybe_load_checkpoint(
            load_path,
            load_trainer_state=getattr(trainer, "load_trainer_state", None),
            load_optim_state=getattr(trainer, "load_optim_state", None),
        )
    if fresh_parameters is not None:
        import torch

        assert transition is not None
        scale = float(transition["weight_scale"])
        if not 0.0 <= scale <= 1.0:
            raise ValueError("shrink-perturb weight_scale must be in [0, 1]")
        with torch.no_grad():
            parameters = list(trainer.train_module.model.parameters())
            if len(parameters) != len(fresh_parameters):
                raise RuntimeError("model parameter structure changed during checkpoint load")
            for parameter, fresh in zip(parameters, fresh_parameters):
                parameter.mul_(scale).add_(fresh, alpha=1.0 - scale)
    loaded_tokens = int(trainer.global_train_tokens_seen)
    if loaded_tokens == spec.hard_stop_tokens:
        snapshot = diagnostics.snapshot()
        if (
            snapshot["heldout_ce"] is None or snapshot["heldout_tokens_seen"] != loaded_tokens
        ) and not snapshot["numeric_failure"]:
            evaluator = getattr(trainer, "callbacks", {}).get(spec.search_validation_callback)
            if (
                evaluator is None
                or not callable(getattr(evaluator, "perform_eval", None))
                or not bool(getattr(evaluator, "eval_on_finish", False))
            ):
                raise RuntimeError("completed checkpoint requires its configured search evaluator")
            evaluator.perform_eval()
            log_metrics = getattr(trainer, "_log_metrics", None)
            if callable(log_metrics):
                log_metrics()
            join_bookkeeping = getattr(trainer, "_join_bookkeeping_ops", None)
            if callable(join_bookkeeping):
                join_bookkeeping()
            snapshot = diagnostics.snapshot()
        heldout_ce = snapshot["heldout_ce"]
        if (
            heldout_ce is None
            and snapshot["numeric_failure"]
            and snapshot["heldout_tokens_seen"] == loaded_tokens
        ):
            heldout_ce = float("nan")
        elif heldout_ce is None or snapshot["heldout_tokens_seen"] != loaded_tokens:
            raise RuntimeError("completed checkpoint lacks fresh search-validation evidence")
        checkpointer_callback = getattr(trainer, "callbacks", {}).get("checkpointer")
        checkpoint_ref = (
            ""
            if checkpointer_callback is None
            else str(
                getattr(
                    checkpointer_callback,
                    "_latest_checkpoint_path",
                    "",
                )
            )
        )
        if not checkpoint_ref:
            if loaded:
                checkpoint_ref = str(trainer.checkpointer.latest_checkpoint(trainer.save_folder))
            else:
                checkpoint_ref = str(getattr(trainer, "load_path", "") or trainer.save_folder)
        return WorkerObservation(
            trial_id=spec.trial_id,
            tokens=loaded_tokens,
            heldout_ce=float(heldout_ce),
            train_ce_history=tuple(float(value) for value in snapshot["train_ce_history"]),
            grad_norm_history=tuple(float(value) for value in snapshot["grad_norm_history"]),
            activation_ratio=snapshot.get("activation_ratio"),
            numeric_failure=bool(snapshot["numeric_failure"]),
            checkpoint_ref=checkpoint_ref,
        )
    if loaded_tokens >= spec.target_tokens:
        raise SegmentComplete(
            f"trial {spec.trial_id} already reached {loaded_tokens} tokens "
            f"(target {spec.target_tokens})"
        )
    if not loaded_tokens < spec.hard_stop_tokens <= spec.target_tokens:
        raise ValueError(
            f"hard_stop_tokens must be in ({loaded_tokens}, {spec.target_tokens}], "
            f"got {spec.hard_stop_tokens}"
        )
    segment_tokens = spec.hard_stop_tokens - loaded_tokens
    if segment_tokens % actual_global_batch_size != 0:
        raise ValueError(
            f"segment token delta {segment_tokens} must align to global batch size "
            f"{actual_global_batch_size}"
        )
    trainer.hard_stop = Duration.tokens(spec.hard_stop_tokens)
    trainer.fit()
    completed_tokens = int(trainer.global_train_tokens_seen)
    if completed_tokens != spec.hard_stop_tokens:
        raise RuntimeError(
            f"segment stopped at {completed_tokens}, expected {spec.hard_stop_tokens}"
        )
    checkpointer_callback = getattr(trainer, "callbacks", {}).get("checkpointer")
    checkpoint_ref = (
        ""
        if checkpointer_callback is None
        else str(getattr(checkpointer_callback, "_latest_checkpoint_path", ""))
    )
    if not checkpoint_ref:
        checkpoint_ref = str(trainer.save_checkpoint())
    snapshot = diagnostics.snapshot()
    heldout_ce = snapshot["heldout_ce"]
    if heldout_ce is None:
        if snapshot["numeric_failure"]:
            heldout_ce = float("nan")
        else:
            raise RuntimeError("segment completed without search-validation CE")
    elif snapshot["heldout_tokens_seen"] != completed_tokens:
        raise RuntimeError(
            "segment completed without fresh search-validation CE at its token boundary"
        )
    return WorkerObservation(
        trial_id=spec.trial_id,
        tokens=completed_tokens,
        heldout_ce=float(heldout_ce),
        train_ce_history=tuple(float(value) for value in snapshot["train_ce_history"]),
        grad_norm_history=tuple(float(value) for value in snapshot["grad_norm_history"]),
        activation_ratio=snapshot.get("activation_ratio"),
        numeric_failure=bool(snapshot["numeric_failure"]),
        checkpoint_ref=checkpoint_ref,
    )


# Environment variables that indicate an outer (torchrun/torchelastic) launcher.
_TORCHRUN_MARKERS = (
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "OMP_NUM_THREADS_SET_BY_TORCHRUN",
)
_FINALIST_CONTINUATION_ENV = "EDULLM_FINALIST_CONTINUATION"
_FINALIST_WORLD_SIZE_ENV = "EDULLM_FINALIST_WORLD_SIZE"


def world_size_one_env(
    gpu: int, master_port: int, base_env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Build the environment for an isolated single-GPU trial subprocess.

    Any inherited torchrun/torchelastic markers are stripped so a nested launch cannot make the
    child believe it is one rank of a larger job.
    """
    env: Dict[str, str] = dict(base_env if base_env is not None else os.environ)
    for marker in _TORCHRUN_MARKERS:
        env.pop(marker, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    return env


def finalist_distributed_env(
    gpu_ids: List[int],
    *,
    world_size: int,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Build the parent environment for an explicitly opted-in finalist ``torchrun``.

    The launcher itself supplies rank and rendezvous variables. The marker here is intentionally
    separate from those generic variables so ordinary HPO workers remain fail-closed at one rank.
    """
    if world_size <= 1:
        raise ValueError("finalist distributed world size must be greater than one")
    if len(gpu_ids) != world_size or len(set(gpu_ids)) != world_size:
        raise ValueError("finalist distributed launch requires one distinct GPU per rank")
    env: Dict[str, str] = dict(base_env if base_env is not None else os.environ)
    for name in (
        *_TORCHRUN_MARKERS,
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpu_ids),
            _FINALIST_CONTINUATION_ENV: "1",
            _FINALIST_WORLD_SIZE_ENV: str(world_size),
        }
    )
    return env


def assert_single_process_topology(env: Mapping[str, str]) -> None:
    """Fail closed unless ``env`` describes a single-process, non-torchrun launch."""
    world_size = env.get("WORLD_SIZE", "1")
    if world_size != "1":
        raise RuntimeError(
            f"HPO trial workers must run at WORLD_SIZE=1, found WORLD_SIZE={world_size}. "
            "Do not launch the controller under `torchrun --nproc-per-node=N`."
        )
    present = [m for m in _TORCHRUN_MARKERS if m in env]
    if present:
        raise RuntimeError(
            f"detected outer launcher markers {present}; the controller must be CPU-only and "
            "spawn isolated single-process trial workers itself"
        )


def assert_worker_topology(env: Mapping[str, str]) -> None:
    """Permit distributed topology only for an explicitly marked finalist continuation."""
    if env.get(_FINALIST_CONTINUATION_ENV) != "1":
        assert_single_process_topology(env)
        return

    try:
        expected_world_size = int(env[_FINALIST_WORLD_SIZE_ENV])
        world_size = int(env["WORLD_SIZE"])
        rank = int(env["RANK"])
        local_rank = int(env["LOCAL_RANK"])
        local_world_size = int(env["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("finalist distributed worker is missing valid rank metadata") from exc
    if expected_world_size <= 1 or world_size != expected_world_size:
        raise RuntimeError(
            f"finalist distributed WORLD_SIZE={world_size} does not match explicit "
            f"expected world size {expected_world_size}"
        )
    if local_world_size != expected_world_size:
        raise RuntimeError(
            f"single-node finalist requires LOCAL_WORLD_SIZE={expected_world_size}, "
            f"found {local_world_size}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError("finalist distributed rank metadata is out of range")
    visible_devices = [device for device in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if device]
    if len(visible_devices) != expected_world_size:
        raise RuntimeError(
            "finalist distributed worker requires one CUDA_VISIBLE_DEVICES entry per rank"
        )
    if "TORCHELASTIC_RUN_ID" not in env:
        raise RuntimeError("finalist distributed worker must be launched through torchrun")


def should_emit_worker_result(env: Mapping[str, str]) -> bool:
    """Return whether this worker rank owns the controller-facing observation JSON."""
    return env.get(_FINALIST_CONTINUATION_ENV) != "1" or env.get("RANK", "0") == "0"


# --- Checkpoint namespaces (scoped strictly to one trial) ---


def trial_namespace(root: str, trial_id: str) -> str:
    return str(join_path(root, "trials", trial_id))


def trial_checkpoint_dir(root: str, trial_id: str, step: int) -> str:
    return str(join_path(trial_namespace(root, trial_id), f"step{step}"))


def controller_dir(root: str) -> str:
    return str(join_path(root, "controller"))


def latest_step_dir(root: str, trial_id: str) -> Optional[str]:
    """The highest ``step{N}`` directory *within one trial namespace*, or ``None``.

    Deliberately never scans the shared run root or other trials' namespaces.
    """
    ns = trial_namespace(root, trial_id)
    try:
        children = list(list_directory(ns, include_files=False))
    except FileNotFoundError:
        return None
    best_step = -1
    best_dir: Optional[str] = None
    for path in children:
        name = os.path.basename(path)
        m = re.fullmatch(r"step(\d+)", name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_dir = str(path)
    return best_dir


# --- Segment scheduling ---


def next_absolute_hard_stop(loaded_tokens: int, target_tokens: int, quantum: int) -> Duration:
    """A fresh absolute token ceiling for the next segment, strictly above ``loaded_tokens``.

    :raises SegmentComplete: if the lineage has already reached its target.
    """
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    if loaded_tokens >= target_tokens:
        raise SegmentComplete(
            f"lineage already at {loaded_tokens} >= target {target_tokens} tokens"
        )
    ceiling = min(loaded_tokens + quantum, target_tokens)
    assert ceiling > loaded_tokens  # guaranteed by the guard above
    return Duration.tokens(ceiling)


def build_hpo_scheduler(realized_hps: Mapping[str, float]) -> WSD:
    """A token-based WSD schedule with fraction warmup/decay and a terminal LR ratio.

    Peak LR lives on the optimizer; the scheduler scales from it, so ``terminal_lr_ratio`` maps
    to ``decay_min_lr = terminal_lr_ratio * peak_lr``.
    """
    peak_lr = float(realized_hps["lr"])
    return WSD(
        units=SchedulerUnits.tokens,
        warmup_fraction=float(realized_hps["warmup_fraction"]),
        decay_fraction=float(realized_hps["decay_fraction"]),
        decay_min_lr=float(realized_hps["terminal_lr_ratio"]) * peak_lr,
    )


def assert_resume_batch_size(
    lineage_global_batch_size: int, resumed_global_batch_size: int
) -> None:
    """Fail closed if the resumed global batch size differs from the lineage config."""
    if lineage_global_batch_size != resumed_global_batch_size:
        raise BatchSizeMismatch(
            f"resumed global_batch_size {resumed_global_batch_size} != lineage "
            f"{lineage_global_batch_size}; optimizer/data-loader state is not transferable"
        )


@dataclass(frozen=True)
class TrialConfigArtifact:
    """An immutable, content-hashed record of a lineage's configuration.

    On resume the scheduler is rebuilt from this artifact (after verifying its hash) rather than
    from whatever the current controller defaults happen to be.
    """

    payload: Dict[str, Any]

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reconstruct_scheduler(artifact: TrialConfigArtifact, *, expected_hash: str) -> WSD:
    """Rebuild the schedule from a hash-verified artifact. Fails closed on a hash mismatch."""
    actual = artifact.content_hash
    if actual != expected_hash:
        raise ValueError(
            f"trial config artifact hash mismatch: got {actual}, expected {expected_hash}"
        )
    return build_hpo_scheduler(artifact.payload["realized_hps"])


@dataclass(frozen=True)
class ReconcileResult:
    """The outcome of reconciling recorded controller state with on-disk checkpoints."""

    trial_id: str
    resume_dir: Optional[str]
    resume_tokens: int
    dropped_incomplete: bool


def reconcile_trial(
    root: str,
    trial_id: str,
    *,
    recorded_tokens: int,
    step_to_tokens: Callable[[int], int],
    is_complete: Callable[[str], bool],
) -> ReconcileResult:
    """Reconcile a trial's recorded fidelity against its latest *complete* checkpoint.

    A crash between an allocation event and a completed checkpoint save leaves a torn newest
    directory. On retry we ignore any incomplete checkpoint and resume from the latest complete
    one, reporting whether a torn save was dropped so the controller can re-dispatch the segment.
    """
    ns = trial_namespace(root, trial_id)
    try:
        children = list(list_directory(ns, include_files=False))
    except FileNotFoundError:
        return ReconcileResult(trial_id, None, 0, False)

    complete_steps: List[int] = []
    step_dirs: List[str] = []
    for path in children:
        name = os.path.basename(path)
        m = re.fullmatch(r"step(\d+)", name)
        if not m:
            continue
        step_dirs.append(str(path))
        if is_complete(str(path)):
            complete_steps.append(int(m.group(1)))

    if not complete_steps:
        return ReconcileResult(
            trial_id,
            None,
            0,
            dropped_incomplete=bool(step_dirs and recorded_tokens > 0),
        )

    best = max(complete_steps)
    resume_dir = str(join_path(ns, f"step{best}"))
    resume_tokens = int(step_to_tokens(best))
    dropped = resume_tokens < recorded_tokens
    return ReconcileResult(trial_id, resume_dir, resume_tokens, dropped)


@dataclass
class WorkerConfig:
    """Everything one trial segment needs, plus the held-out evaluator gate."""

    trial_id: str
    gpu: int
    target_tokens: int
    quantum: int
    global_batch_size: int
    realized_hps: Dict[str, float]
    checkpoint_root: str
    evaluator_gate: EvaluatorGate

    def assert_evaluator_ready(self, available: Set[str]) -> None:
        """Fail closed unless the search-validation evaluator is present at a segment boundary."""
        self.evaluator_gate.require_ready(available)

    def config_artifact(self) -> TrialConfigArtifact:
        return TrialConfigArtifact(
            payload={
                "realized_hps": dict(self.realized_hps),
                "global_batch_size": self.global_batch_size,
                "target_tokens": self.target_tokens,
            }
        )

    def segment_spec(
        self,
        hard_stop_tokens: int,
        *,
        transition: Optional[Mapping[str, Any]] = None,
    ) -> SegmentSpec:
        return SegmentSpec(
            trial_id=self.trial_id,
            target_tokens=self.target_tokens,
            hard_stop_tokens=hard_stop_tokens,
            lineage_global_batch_size=self.global_batch_size,
            transition=transition,
            search_validation_callback=self.evaluator_gate.search_validation,
        )


@dataclass
class HpoDiagnosticsCallback(Callback):
    """Bounded per-segment telemetry for BTTackler and FT-PFN.

    Aggregates a *bounded* history rather than copying every parameter to CPU each batch. The
    controller reads :meth:`snapshot` at the segment boundary and binds a BTT verdict to
    :meth:`observation_hash`.
    """

    heldout_metric: str = "eval/search_validation/val/CE loss"
    grad_norm_metric: str = "optim/total grad norm"
    train_ce_metric: str = "train/CE loss"
    activation_ratio_metric: str = "hpo/activation effective support"
    max_history: int = 256

    heldout_ce: Optional[float] = None
    heldout_tokens_seen: Optional[int] = None
    train_ce: Optional[float] = None
    activation_ratio: Optional[float] = None
    tokens_seen: int = 0
    numeric_failure: bool = False
    grad_norm_history: List[float] = field(default_factory=list)
    train_ce_history: List[float] = field(default_factory=list)
    checkpoint_saved: bool = False

    def _push(self, buf: List[float], value: float) -> None:
        buf.append(value)
        if len(buf) > self.max_history:
            del buf[0 : len(buf) - self.max_history]

    def _record_failure_if_nonfinite(self, value: float) -> bool:
        import math

        if not math.isfinite(value):
            self.numeric_failure = True
            return True
        return False

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        del step
        if self._trainer is not None:
            self.tokens_seen = int(self.trainer.global_train_tokens_seen)
        if self.heldout_metric in metrics:
            ce = float(metrics[self.heldout_metric])
            if self._record_failure_if_nonfinite(ce):
                self.heldout_ce = None
                self.heldout_tokens_seen = self.tokens_seen
            else:
                self.heldout_ce = ce
                self.heldout_tokens_seen = self.tokens_seen
        if self.train_ce_metric in metrics:
            ce = float(metrics[self.train_ce_metric])
            if not self._record_failure_if_nonfinite(ce):
                self.train_ce = ce
                self._push(self.train_ce_history, ce)
        if self.grad_norm_metric in metrics:
            gn = float(metrics[self.grad_norm_metric])
            if not self._record_failure_if_nonfinite(gn):
                self._push(self.grad_norm_history, gn)
        if self.activation_ratio_metric in metrics:
            ratio = float(metrics[self.activation_ratio_metric])
            if not self._record_failure_if_nonfinite(ratio):
                if not 0.0 <= ratio <= 1.0:
                    raise ValueError("activation effective support must be in [0, 1]")
                self.activation_ratio = ratio

    def post_checkpoint_saved(self, path) -> None:  # noqa: ANN001 - matches base signature
        del path
        self.checkpoint_saved = True

    def on_error(self, exc: BaseException) -> None:
        del exc
        self.numeric_failure = True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "heldout_ce": self.heldout_ce,
            "heldout_tokens_seen": self.heldout_tokens_seen,
            "train_ce": self.train_ce,
            "activation_ratio": self.activation_ratio,
            "tokens_seen": self.tokens_seen,
            "numeric_failure": self.numeric_failure,
            "grad_norm_history": list(self.grad_norm_history),
            "train_ce_history": list(self.train_ce_history),
            "checkpoint_saved": self.checkpoint_saved,
        }

    def observation_hash(self) -> str:
        canonical = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def state_dict(self) -> Dict[str, Any]:
        state = self.snapshot()
        # post_checkpoint_saved() runs after Trainer serializes callback state, so this process-
        # local completion signal cannot truthfully describe the checkpoint containing it.
        state.pop("checkpoint_saved")
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.heldout_ce = state_dict.get("heldout_ce")
        self.heldout_tokens_seen = state_dict.get("heldout_tokens_seen")
        self.train_ce = state_dict.get("train_ce")
        self.activation_ratio = state_dict.get("activation_ratio")
        self.tokens_seen = int(state_dict.get("tokens_seen", 0))
        self.numeric_failure = bool(state_dict.get("numeric_failure", False))
        self.grad_norm_history = list(state_dict.get("grad_norm_history", []))
        self.train_ce_history = list(state_dict.get("train_ce_history", []))
        self.checkpoint_saved = False


def configure_hpo_experiment(
    config,
    *,
    worker: WorkerConfig,
    hard_stop_tokens: int,
    heldout_metric: str,
    fidelity: Optional[Mapping[str, Any]] = None,
    model_parameterization: Optional[Mapping[str, Any]] = None,
) -> HpoDiagnosticsCallback:
    """Apply model parameterization and training fidelity as independent dimensions."""
    callbacks = config.trainer.callbacks
    worker.assert_evaluator_ready(set(callbacks))
    if worker.evaluator_gate.untouched in callbacks:
        raise RuntimeError("untouched final evaluator must not run inside HPO search segments")
    search_evaluator = callbacks[worker.evaluator_gate.search_validation]
    if not isinstance(search_evaluator, (EvaluatorCallback, LMEvaluatorCallbackConfig)) or not bool(
        search_evaluator.eval_on_finish
    ):
        raise RuntimeError(
            "search-validation callback must be an OLMo evaluator with eval_on_finish=True"
        )
    config.data_loader.global_batch_size = worker.global_batch_size
    if not 0 < hard_stop_tokens <= worker.target_tokens:
        raise ValueError("hard_stop_tokens must be positive and at most target_tokens")
    model_parameterization = (
        {"kind": "standard"} if model_parameterization is None else dict(model_parameterization)
    )
    parameterization_kind = model_parameterization.get("kind", "standard")
    if parameterization_kind == "umup":
        from .umup import UMuPAdamWConfig, require_official_umup_forward

        require_official_umup_forward()
        if getattr(config, "umup_backend", None) != "unit-scaling" or not bool(
            getattr(config, "umup_parity_validated", False)
        ):
            raise RuntimeError(
                "u-muP parameterization requires the official unit-scaling configurator "
                "with validated parameter-count parity"
            )
        metadata = getattr(config, "umup_metadata", None)
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("proxy_depth") != config.model.n_layers
        ):
            raise RuntimeError("u-muP parameterization metadata is missing or inconsistent")
        if not isinstance(config.train_module.optim, UMuPAdamWConfig):
            raise RuntimeError("u-muP parameterization requires official scaled AdamW groups")
    elif parameterization_kind != "standard":
        raise ValueError(f"unknown model parameterization kind: {parameterization_kind}")

    fidelity = {"kind": "exact"} if fidelity is None else dict(fidelity)
    fidelity_kind = fidelity.get("kind", "exact")
    if fidelity_kind == "frozen_layer":
        proxy = FrozenLayerProxy(
            n_layers=int(config.model.n_layers),
            train_last_k=int(fidelity["train_last_k"]),
        )
        existing_patterns = list(config.model.freeze_params or [])
        config.model.freeze_params = list(
            dict.fromkeys(existing_patterns + proxy.freeze_patterns())
        )
    elif fidelity_kind != "exact":
        raise ValueError(f"unknown HPO fidelity kind: {fidelity_kind}")

    config.trainer.save_folder = trial_namespace(worker.checkpoint_root, worker.trial_id)
    config.trainer.max_duration = Duration.tokens(worker.target_tokens)
    config.trainer.hard_stop = Duration.tokens(hard_stop_tokens)

    hps = worker.realized_hps
    optim = config.train_module.optim
    optim.lr = float(hps["lr"])
    optim.weight_decay = float(hps["weight_decay"])
    optim.eps = float(hps["eps"])
    beta1 = float(optim.betas[0])
    beta2 = 1.0 - float(hps["beta2_gap"])
    if not 0.0 < beta2 < 1.0:
        raise ValueError("beta2_gap must produce beta2 in (0, 1)")
    optim.betas = (beta1, beta2)
    config.train_module.scheduler = build_hpo_scheduler(hps)
    config.train_module.max_grad_norm = float(hps["max_grad_norm"])

    checkpointer = callbacks.get("checkpointer")
    if checkpointer is None:
        raise RuntimeError("HPO experiment requires the OLMo checkpointer callback")
    checkpointer.save_interval = None
    checkpointer.ephemeral_save_interval = None
    checkpointer.fixed_steps = None
    checkpointer.max_checkpoints = None
    checkpointer.save_async = False

    if "hpo_diagnostics" in callbacks:
        raise RuntimeError("HPO diagnostics callback is already configured")
    diagnostics = HpoDiagnosticsCallback(heldout_metric=heldout_metric)
    callbacks["hpo_diagnostics"] = diagnostics
    return diagnostics


def validate_umup_model(model) -> None:
    """Require official unit-scaled execution and metadata on every trainable parameter."""
    from unit_scaling.parameter import has_parameter_data

    from .umup import UMUP_EXECUTION_BACKEND

    if getattr(model, "_umup_execution_backend", None) != UMUP_EXECUTION_BACKEND:
        raise RuntimeError("u-muP model has no verified official unit-scaled execution backend")
    untagged = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not has_parameter_data(parameter)
    ]
    if untagged:
        preview = ", ".join(untagged[:5])
        raise RuntimeError(
            f"u-muP model contains parameters without official unit_scaling metadata: {preview}"
        )
