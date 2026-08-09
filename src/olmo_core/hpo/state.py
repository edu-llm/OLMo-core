"""
Event-sourced controller state with deterministic replay.

The controller never mutates state in place as its source of truth. Instead it appends
immutable :class:`Event` records to an :class:`EventLog`, and *derives* the current
:class:`ControllerState` by folding that log with :func:`replay`. This gives three properties
the plan requires:

- **Deterministic replay / crash recovery.** Re-reading the log reconstructs byte-identical
  state, so a preempted job resumes exactly where it left off.
- **Incremental budget accounting.** Each allocation charges only ``target - current`` tokens,
  fixing the cumulative-fidelity over-count found in the NePS reference runtime.
- **Auditability.** Every decision, observation, and verdict is a durable line.

Pure standard library (``json`` only) so it is testable without ``torch``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .types import (
    ActionKind,
    Allocation,
    BTTDisposition,
    BTTVerdictKind,
    TrialStatus,
    Verdict,
)

__all__ = [
    "EventKind",
    "Event",
    "EventLog",
    "TrialRecord",
    "ControllerState",
    "observation_hash",
    "replay",
]


class EventKind(str, Enum):
    ALLOCATION = "allocation"
    OBSERVATION = "observation"
    VERDICT = "verdict"
    STATUS = "status"
    CONTROLLER_SNAPSHOT = "controller_snapshot"
    ADVISOR = "advisor"
    IPBT_TRANSITION = "ipbt_transition"
    FINAL_EVALUATION = "final_evaluation"


@dataclass(frozen=True)
class Event:
    """One immutable, sequenced entry in the controller log."""

    seq: int
    kind: EventKind
    payload: Dict[str, Any]

    def to_json_line(self) -> str:
        value = {"seq": self.seq, "kind": self.kind.value, "payload": self.payload}
        return json.dumps(_encode_json_value(value), allow_nan=False)

    @classmethod
    def from_json_line(cls, line: str) -> "Event":
        d = _decode_json_value(json.loads(line))
        return cls(seq=int(d["seq"]), kind=EventKind(d["kind"]), payload=d["payload"])


_ENCODED_TYPE_TAG = "__hpo_encoded_type__"
_ENCODED_VALUE_TAG = "__hpo_encoded_value__"


def _encode_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            name = "nan"
        elif value > 0:
            name = "positive_infinity"
        else:
            name = "negative_infinity"
        return {
            _ENCODED_TYPE_TAG: "nonfinite_float",
            _ENCODED_VALUE_TAG: name,
        }
    if isinstance(value, dict):
        return {key: _encode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_json_value(item) for item in value]
    return value


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        if (
            set(value) == {_ENCODED_TYPE_TAG, _ENCODED_VALUE_TAG}
            and value[_ENCODED_TYPE_TAG] == "nonfinite_float"
        ):
            return {
                "nan": float("nan"),
                "positive_infinity": float("inf"),
                "negative_infinity": -float("inf"),
            }[value[_ENCODED_VALUE_TAG]]
        return {key: _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    return value


class EventLog:
    """Append-only, strictly monotonically sequenced list of events."""

    def __init__(self) -> None:
        self._events: List[Event] = []

    @property
    def events(self) -> List[Event]:
        return copy.deepcopy(self._events)

    def __iter__(self):
        return iter(copy.deepcopy(self._events))

    def __len__(self) -> int:
        return len(self._events)

    def append(self, event: Event) -> None:
        expected = 0 if not self._events else self._events[-1].seq + 1
        if event.seq != expected:
            raise ValueError(
                f"EventLog is append-only and monotonic: expected seq {expected}, got {event.seq}"
            )
        self._events.append(copy.deepcopy(event))

    def to_jsonl(self) -> str:
        return "".join(e.to_json_line() + "\n" for e in self._events)

    @classmethod
    def from_jsonl(cls, text: str) -> "EventLog":
        log = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            log.append(Event.from_json_line(line))
        return log


@dataclass
class TrialRecord:
    """Derived per-trial state (never authored directly; produced by :func:`replay`)."""

    trial_id: str
    parent_trial_id: Optional[str] = None
    lineage_id: Optional[str] = None
    parent_lineage_id: Optional[str] = None
    status: Optional[TrialStatus] = None
    current_fidelity: int = 0
    pending_target_fidelity: Optional[int] = None
    curve: List[List[float]] = field(default_factory=list)  # [[tokens, ce], ...]
    latest_verdict: Optional[Verdict] = None
    latest_observation_fidelity: Optional[int] = None
    latest_observation_hash: Optional[str] = None
    latest_checkpoint_ref: Optional[str] = None


def observation_hash(
    trial_id: str,
    tokens: int,
    ce: float,
    *,
    train_ce_history: Sequence[float] = (),
    grad_norm_history: Sequence[float] = (),
    activation_ratio: Optional[float] = None,
    numeric_failure: bool = False,
) -> str:
    """Return a stable hash for one observation, including non-finite failure evidence."""
    payload = {
        "trial_id": trial_id,
        "tokens": int(tokens),
        "ce": ce,
        "train_ce_history": list(train_ce_history),
        "grad_norm_history": list(grad_norm_history),
        "activation_ratio": activation_ratio,
        "numeric_failure": bool(numeric_failure),
    }
    value = json.dumps(
        _encode_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class ControllerState:
    """The folded view of the event log."""

    next_decision_id: int = 0
    tokens_charged: int = 0
    accelerator_seconds_charged: float = 0.0
    trials: Dict[str, TrialRecord] = field(default_factory=dict)
    controller_snapshot: Optional[Dict[str, Any]] = None
    advisor_events: List[Dict[str, Any]] = field(default_factory=list)
    ipbt_transitions: List[Dict[str, Any]] = field(default_factory=list)
    final_evaluations: List[Dict[str, Any]] = field(default_factory=list)

    def _trial(self, trial_id: str) -> TrialRecord:
        rec = self.trials.get(trial_id)
        if rec is None:
            rec = TrialRecord(trial_id=trial_id)
            self.trials[trial_id] = rec
        return rec

    def apply(self, event: Event) -> None:
        if event.kind is EventKind.ALLOCATION:
            self._apply_allocation(event.payload)
        elif event.kind is EventKind.OBSERVATION:
            self._apply_observation(event.payload)
        elif event.kind is EventKind.VERDICT:
            self._apply_verdict(event.payload)
        elif event.kind is EventKind.STATUS:
            self._apply_status(event.payload)
        elif event.kind is EventKind.CONTROLLER_SNAPSHOT:
            self.controller_snapshot = copy.deepcopy(event.payload)
        elif event.kind is EventKind.ADVISOR:
            self.advisor_events.append(copy.deepcopy(event.payload))
        elif event.kind is EventKind.IPBT_TRANSITION:
            self.ipbt_transitions.append(copy.deepcopy(event.payload))
        elif event.kind is EventKind.FINAL_EVALUATION:
            self.final_evaluations.append(copy.deepcopy(event.payload))
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown event kind: {event.kind}")

    def _apply_allocation(self, payload: Mapping[str, Any]) -> None:
        alloc = Allocation.from_dict(payload)
        if alloc.decision_id != self.next_decision_id:
            raise ValueError(
                f"expected decision_id {self.next_decision_id}, got {alloc.decision_id}"
            )
        # Incremental (not cumulative) budget charge.
        charge = alloc.target_fidelity - alloc.current_fidelity
        if charge <= 0:
            raise ValueError(
                f"allocation {alloc.decision_id} target {alloc.target_fidelity} not above current "
                f"{alloc.current_fidelity}"
            )

        if alloc.kind is ActionKind.START:
            if alloc.trial_id in self.trials:
                raise ValueError(f"START reuses existing trial_id {alloc.trial_id}")
            if alloc.parent_trial_id is None:
                if alloc.current_fidelity != 0:
                    raise ValueError("fresh START must begin at fidelity 0")
            else:
                parent = self.trials.get(alloc.parent_trial_id)
                if parent is None:
                    raise ValueError(
                        f"inherited START references unknown parent {alloc.parent_trial_id}"
                    )
                if parent.current_fidelity != alloc.current_fidelity:
                    raise ValueError(
                        "inherited START must begin at its parent's completed fidelity"
                    )
            rec = self._trial(alloc.trial_id)
            rec.parent_trial_id = alloc.parent_trial_id
            transition = alloc.transition or {}
            rec.lineage_id = str(transition.get("lineage_id", alloc.trial_id))
            parent_lineage_id = transition.get("parent_lineage_id")
            rec.parent_lineage_id = None if parent_lineage_id is None else str(parent_lineage_id)
            rec.current_fidelity = alloc.current_fidelity
        else:
            resume_record = self.trials.get(alloc.trial_id)
            if resume_record is None:
                raise ValueError(f"RESUME references unknown trial_id {alloc.trial_id}")
            rec = resume_record
            if rec.pending_target_fidelity is not None or rec.status is TrialStatus.RUNNING:
                raise ValueError(f"trial {alloc.trial_id} already has pending work")
            if rec.current_fidelity != alloc.current_fidelity:
                raise ValueError(
                    f"trial {alloc.trial_id} completed fidelity {rec.current_fidelity}, "
                    f"allocation claims {alloc.current_fidelity}"
                )
            if rec.latest_verdict is None or not rec.latest_verdict.is_eligible_for_resume():
                raise ValueError(f"trial {alloc.trial_id} is not eligible to resume")
            if (
                alloc.verdict_id is not None
                and alloc.verdict_id != rec.latest_verdict.observation_hash
            ):
                raise ValueError(f"trial {alloc.trial_id} resume uses a stale verdict")

        self.tokens_charged += charge
        self.next_decision_id += 1
        rec.pending_target_fidelity = alloc.target_fidelity
        rec.status = TrialStatus.RUNNING

    def _apply_observation(self, payload: Mapping[str, Any]) -> None:
        trial_id = str(payload["trial_id"])
        rec = self.trials.get(trial_id)
        if rec is None:
            raise ValueError(f"observation references unknown trial_id {trial_id}")
        tokens = int(payload["tokens"])
        if rec.pending_target_fidelity is None:
            raise ValueError(f"trial {trial_id} has no pending allocation")
        if tokens != rec.pending_target_fidelity:
            raise ValueError(
                f"trial {trial_id} observation fidelity {tokens} does not match pending target "
                f"{rec.pending_target_fidelity}"
            )
        ce = float(payload["ce"])
        computed_hash = observation_hash(
            trial_id,
            tokens,
            ce,
            train_ce_history=tuple(payload.get("train_ce_history", ())),
            grad_norm_history=tuple(payload.get("grad_norm_history", ())),
            activation_ratio=payload.get("activation_ratio"),
            numeric_failure=bool(payload.get("numeric_failure", False)),
        )
        supplied_hash = payload.get("observation_hash")
        if supplied_hash is not None and str(supplied_hash) != computed_hash:
            raise ValueError(f"observation hash mismatch for trial {trial_id}")
        rec.current_fidelity = tokens
        rec.pending_target_fidelity = None
        rec.latest_observation_fidelity = tokens
        rec.latest_observation_hash = computed_hash
        rec.latest_checkpoint_ref = payload.get("checkpoint_ref")
        accelerator_seconds = float(payload.get("accelerator_seconds", 0.0))
        if not math.isfinite(accelerator_seconds) or accelerator_seconds < 0.0:
            raise ValueError("observation accelerator_seconds must be finite and non-negative")
        self.accelerator_seconds_charged += accelerator_seconds
        if math.isfinite(ce):
            rec.curve.append([tokens, ce])

    def _apply_verdict(self, payload: Mapping[str, Any]) -> None:
        verdict = Verdict(
            kind=BTTVerdictKind(payload["kind"]),
            indicators=tuple(payload.get("indicators", ())),
            trial_id=str(payload["trial_id"]),
            completed_fidelity=int(payload["completed_fidelity"]),
            observation_hash=str(payload["observation_hash"]),
            profile_version=str(payload["profile_version"]),
            spared_by_reserve=bool(payload.get("spared_by_reserve", False)),
            protected_by_peer_rank=bool(payload.get("protected_by_peer_rank", False)),
            disposition=(
                None
                if payload.get("disposition") is None
                else BTTDisposition(payload["disposition"])
            ),
        )
        rec = self.trials.get(verdict.trial_id)
        if rec is None:
            raise ValueError(f"verdict references unknown trial_id {verdict.trial_id}")
        if (
            rec.latest_observation_fidelity != verdict.completed_fidelity
            or rec.latest_observation_hash != verdict.observation_hash
        ):
            raise ValueError(
                f"verdict for {verdict.trial_id} is not bound to its latest observation"
            )
        was_retired = rec.status is TrialStatus.RETIRED
        rec.latest_verdict = verdict
        if was_retired:
            return
        if verdict.kind is BTTVerdictKind.FATAL or verdict.disposition is BTTDisposition.STOP:
            rec.status = TrialStatus.RETIRED
        elif verdict.kind is BTTVerdictKind.SATURATED:
            rec.status = TrialStatus.COMPLETED
        elif verdict.kind is BTTVerdictKind.DEGRADED:
            rec.status = TrialStatus.PAUSED
        else:
            rec.status = TrialStatus.PAUSED

    def _apply_status(self, payload: Mapping[str, Any]) -> None:
        rec = self._trial(str(payload["trial_id"]))
        rec.status = TrialStatus(payload["status"])

    def snapshot(self) -> Dict[str, Any]:
        """A JSON-safe, order-independent view for equality checks and forensics."""
        return {
            "next_decision_id": self.next_decision_id,
            "tokens_charged": self.tokens_charged,
            "accelerator_seconds_charged": self.accelerator_seconds_charged,
            "controller_snapshot": copy.deepcopy(self.controller_snapshot),
            "advisor_events": copy.deepcopy(self.advisor_events),
            "ipbt_transitions": copy.deepcopy(self.ipbt_transitions),
            "final_evaluations": copy.deepcopy(self.final_evaluations),
            "trials": {
                tid: {
                    "parent_trial_id": rec.parent_trial_id,
                    "lineage_id": rec.lineage_id,
                    "parent_lineage_id": rec.parent_lineage_id,
                    "status": None if rec.status is None else rec.status.value,
                    "current_fidelity": rec.current_fidelity,
                    "pending_target_fidelity": rec.pending_target_fidelity,
                    "curve": rec.curve,
                    "latest_observation_fidelity": rec.latest_observation_fidelity,
                    "latest_observation_hash": rec.latest_observation_hash,
                    "latest_checkpoint_ref": rec.latest_checkpoint_ref,
                    "latest_verdict": None
                    if rec.latest_verdict is None
                    else {
                        "kind": rec.latest_verdict.kind.value,
                        "spared_by_reserve": rec.latest_verdict.spared_by_reserve,
                        "protected_by_peer_rank": rec.latest_verdict.protected_by_peer_rank,
                        "disposition": (
                            None
                            if rec.latest_verdict.disposition is None
                            else rec.latest_verdict.disposition.value
                        ),
                    },
                }
                for tid, rec in sorted(self.trials.items())
            },
        }


def replay(events: List[Event]) -> ControllerState:
    """Fold an event list into a fresh :class:`ControllerState`."""
    state = ControllerState()
    for event in events:
        state.apply(event)
    return state
