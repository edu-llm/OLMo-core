"""
Typed contracts shared across the HPO controller.

Everything here is pure ``numpy`` + standard library so the controller's core logic can be
unit-tested without ``torch`` or the optional ``hpo`` third-party packages. Higher layers
(FT-PFN adapter, IPBT, Centaur, workers) build on these types.

The two load-bearing abstractions are:

- :class:`SearchSpace` -- an ordered, fixed set of :class:`SearchDim` that maps a realized
  hyperparameter dict to/from a unit ``[0, 1]`` vector. FT-PFN requires unit-scaled inputs and
  a stable dimension ordering, so the space is the single source of truth for both.
- :class:`Allocation` -- the one typed action the controller ever emits. It carries a
  monotonic ``decision_id`` and everything needed to launch, resume, replay, or audit a
  segment. It serializes to a JSON-safe dict for the append-only event log.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifacts import FTPFN_MAX_HP_DIMS

__all__ = [
    "ActionKind",
    "ProposalSource",
    "BTTVerdictKind",
    "BTTDisposition",
    "TrialStatus",
    "SearchDim",
    "SearchSpace",
    "CurvePoint",
    "WorkerObservation",
    "Verdict",
    "Allocation",
]


class _StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class ActionKind(_StrEnum):
    """The only two things the allocator (ifBO) may grant compute for."""

    START = "start"
    """Begin a brand-new configuration at the minimum fidelity."""

    RESUME = "resume"
    """Continue an existing, eligible trial by exactly one fidelity quantum."""


class ProposalSource(_StrEnum):
    """Where a *new configuration* originated. Only meaningful for ``START`` actions."""

    RANDOM = "random"
    IFBO = "ifbo"
    CMA = "cma"
    LLM = "llm"
    IPBT_META = "ipbt_meta"


class BTTVerdictKind(_StrEnum):
    """BTTackler's per-trial verdict at a fidelity boundary."""

    HEALTHY = "healthy"
    """No indicator fired; the trial stays resumable."""

    DEGRADED = "degraded"
    """A pathology fired; request a checkpoint-boundary recycle (not a global restart)."""

    SATURATED = "saturated"
    """NMG (no more gain): training is *sufficient*. Pause/complete but keep as an incumbent."""

    FATAL = "fatal"
    """Non-finite training. Retire the trial irreversibly."""


class BTTDisposition(_StrEnum):
    """Operational action requested by an evidence-bound BTT verdict."""

    CONTINUE = "continue"
    RECYCLE = "recycle"
    STOP = "stop"
    COMPLETE = "complete"
    RETIRE = "retire"


class TrialStatus(_StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass(frozen=True)
class SearchDim:
    """A single hyperparameter axis with a monotone map to/from ``[0, 1]``.

    :param name: Stable identifier; also the key in a realized-HP dict.
    :param low: Lower bound (inclusive), mapped to unit ``0``.
    :param high: Upper bound (inclusive), mapped to unit ``1``.
    :param log: If ``True``, interpolate geometrically (both bounds must be > 0).
    """

    name: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SearchDim.name must be non-empty")
        if not (math.isfinite(self.low) and math.isfinite(self.high)):
            raise ValueError(f"SearchDim '{self.name}': bounds must be finite")
        if not (self.high > self.low):
            raise ValueError(
                f"SearchDim '{self.name}': require high > low, got {self.low}, {self.high}"
            )
        if self.log and (self.low <= 0.0 or self.high <= 0.0):
            raise ValueError(f"SearchDim '{self.name}': log axis requires positive bounds")

    def to_unit(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"SearchDim '{self.name}': value must be finite")
        if value < self.low or value > self.high:
            raise ValueError(
                f"SearchDim '{self.name}': value {value} outside [{self.low}, {self.high}]"
            )
        if self.log:
            return math.log(value / self.low) / math.log(self.high / self.low)
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        u = float(u)
        if not math.isfinite(u):
            raise ValueError(f"SearchDim '{self.name}': unit coordinate must be finite")
        if u < 0.0 or u > 1.0:
            raise ValueError(f"SearchDim '{self.name}': unit coordinate {u} outside [0, 1]")
        if self.log:
            return self.low * (self.high / self.low) ** u
        return self.low + u * (self.high - self.low)


@dataclass(frozen=True)
class SearchSpace:
    """An ordered, fixed collection of :class:`SearchDim`.

    The ordering is the FT-PFN feature ordering and must never change within a study.
    """

    dims: Tuple[SearchDim, ...]

    def __post_init__(self) -> None:
        if not self.dims:
            raise ValueError("SearchSpace requires at least one dimension")
        names = [d.name for d in self.dims]
        if len(set(names)) != len(names):
            raise ValueError(f"SearchSpace has duplicate dimension names: {names}")
        if len(self.dims) > FTPFN_MAX_HP_DIMS:
            raise ValueError(
                f"SearchSpace has {len(self.dims)} dims but FT-PFN accepts at most "
                f"{FTPFN_MAX_HP_DIMS}"
            )

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def to_unit(self, hps: Mapping[str, float]) -> np.ndarray:
        missing = [d.name for d in self.dims if d.name not in hps]
        if missing:
            raise ValueError(f"SearchSpace.to_unit missing keys: {missing}")
        return np.array([d.to_unit(hps[d.name]) for d in self.dims], dtype=np.float64)

    def from_unit(self, unit: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(unit, dtype=np.float64)
        if arr.shape != (self.ndim,):
            raise ValueError(
                f"SearchSpace.from_unit expected shape ({self.ndim},), got {arr.shape}"
            )
        return {d.name: d.from_unit(float(arr[i])) for i, d in enumerate(self.dims)}


@dataclass(frozen=True)
class CurvePoint:
    """One observed point on a trial's learning curve.

    :param tokens: Absolute tokens seen at this observation (the fidelity coordinate).
    :param ce: Held-out cross-entropy (lower is better).
    :param y: Optional pre-normalized objective in ``[0, 1]`` (higher is better).
    """

    tokens: int
    ce: float
    y: Optional[float] = None


@dataclass(frozen=True)
class WorkerObservation:
    """One completed segment's measured result and bounded BTT telemetry.

    Workers measure and checkpoint; they do not make scheduling decisions. ``heldout_ce`` may
    be non-finite only when ``numeric_failure`` is true, in which case it is retained as fatal
    evidence but excluded from the FT-PFN learning curve.
    """

    trial_id: str
    tokens: int
    heldout_ce: float
    train_ce_history: Tuple[float, ...]
    grad_norm_history: Tuple[float, ...]
    activation_ratio: Optional[float]
    numeric_failure: bool
    checkpoint_ref: Optional[str]
    accelerator_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("WorkerObservation.trial_id must be non-empty")
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens <= 0:
            raise ValueError("WorkerObservation.tokens must be a positive integer")
        if not self.numeric_failure and not math.isfinite(self.heldout_ce):
            raise ValueError("non-finite heldout_ce requires numeric_failure=True")
        if not math.isfinite(self.accelerator_seconds) or self.accelerator_seconds < 0.0:
            raise ValueError("accelerator_seconds must be finite and non-negative")


@dataclass(frozen=True)
class Verdict:
    """A BTTackler diagnosis bound to the exact observation it was computed from.

    Binding a verdict to ``(trial_id, completed_fidelity, observation_hash)`` lets the
    controller reject stale/out-of-order verdicts rather than acting on the wrong evidence.
    """

    kind: BTTVerdictKind
    indicators: Tuple[str, ...]
    trial_id: str
    completed_fidelity: int
    observation_hash: str
    profile_version: str
    spared_by_reserve: bool = False
    """True when a would-be termination was continued by the late-bloomer reserve."""
    protected_by_peer_rank: bool = False
    """True when same-fidelity top-trial protection overrode a would-be degradation."""
    disposition: Optional[BTTDisposition] = None

    def __post_init__(self) -> None:
        if self.disposition is None:
            default = {
                BTTVerdictKind.HEALTHY: BTTDisposition.CONTINUE,
                BTTVerdictKind.DEGRADED: BTTDisposition.RECYCLE,
                BTTVerdictKind.SATURATED: BTTDisposition.COMPLETE,
                BTTVerdictKind.FATAL: BTTDisposition.RETIRE,
            }[self.kind]
            object.__setattr__(self, "disposition", default)

    @property
    def binding_key(self) -> Tuple[str, int, str]:
        return (self.trial_id, self.completed_fidelity, self.observation_hash)

    def is_eligible_for_resume(self) -> bool:
        """Only a healthy trial may be resumed as-is."""
        return self.kind is BTTVerdictKind.HEALTHY and self.disposition is BTTDisposition.CONTINUE

    def is_incumbent_candidate(self) -> bool:
        """Healthy and saturated (NMG) trials remain valid best-candidate contenders."""
        return self.kind in (BTTVerdictKind.HEALTHY, BTTVerdictKind.SATURATED)


@dataclass(frozen=True)
class Allocation:
    """The single typed action the controller emits, with everything needed to replay it.

    :param decision_id: Monotonically increasing controller decision counter (replay order).
    :param kind: ``START`` (new config) or ``RESUME`` (continue a trial).
    :param trial_id: The trial this action creates or continues.
    :param parent_trial_id: For ``RESUME``, the trial/segment id being continued.
    :param unit_config: The configuration as a unit ``[0, 1]`` vector (FT-PFN ordering).
    :param realized_hps: The decoded, evaluated hyperparameters.
    :param current_fidelity: Absolute tokens already trained for this lineage at emit time.
    :param target_fidelity: Absolute token ceiling this segment must reach.
    :param checkpoint_ref: Checkpoint to resume from (``None`` for a fresh start).
    :param horizon: The sampled MFPI-random forecast horizon (scoring only).
    :param threshold: The MFPI improvement threshold ``T`` used for ranking.
    :param mfpi_score: The winning candidate's probability-of-improvement score.
    :param tie_break: Deterministic tie-break key for reproducible selection.
    :param source: Where a ``START`` configuration originated.
    :param verdict_id: The BTT verdict hash consulted for a ``RESUME`` eligibility check.
    """

    decision_id: int
    kind: ActionKind
    trial_id: str
    parent_trial_id: Optional[str]
    unit_config: Tuple[float, ...]
    realized_hps: Dict[str, float]
    current_fidelity: int
    target_fidelity: int
    checkpoint_ref: Optional[str]
    horizon: int
    threshold: float
    mfpi_score: float
    tie_break: Tuple[Any, ...]
    source: ProposalSource
    verdict_id: Optional[str] = None
    batch_id: int = 0
    transition: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["source"] = self.source.value
        d["unit_config"] = list(self.unit_config)
        d["tie_break"] = list(self.tie_break)
        d["realized_hps"] = {k: float(v) for k, v in self.realized_hps.items()}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Allocation":
        return cls(
            decision_id=int(d["decision_id"]),
            kind=ActionKind(d["kind"]),
            trial_id=str(d["trial_id"]),
            parent_trial_id=d["parent_trial_id"],
            unit_config=tuple(float(x) for x in d["unit_config"]),
            realized_hps={k: float(v) for k, v in d["realized_hps"].items()},
            current_fidelity=int(d["current_fidelity"]),
            target_fidelity=int(d["target_fidelity"]),
            checkpoint_ref=d["checkpoint_ref"],
            horizon=int(d["horizon"]),
            threshold=float(d["threshold"]),
            mfpi_score=float(d["mfpi_score"]),
            tie_break=tuple(d["tie_break"]),
            source=ProposalSource(d["source"]),
            verdict_id=d.get("verdict_id"),
            batch_id=int(d.get("batch_id", 0)),
            transition=None if d.get("transition") is None else dict(d["transition"]),
        )
