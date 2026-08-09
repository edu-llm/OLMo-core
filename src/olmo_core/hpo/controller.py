"""
The single-owner typed-action arbiter that composes the HPO stack.

:class:`HpoController` is the one place decisions are made, and it enforces the plan's ownership
contract:

- **ifBO is the sole allocator.** Every action is either a ``START`` or a ``RESUME`` granted one
  fixed token quantum via MFPI-random (:mod:`olmo_core.hpo.ifbo`). Nothing else spends compute.
- **BTTackler is the sole eligibility gate.** Only trials whose latest verdict is ``HEALTHY`` are
  offered as continuation candidates; a low MFPI score can pause a trial but never terminally
  discards it.
- **Every decision is event-sourced.** Actions, observations, and verdicts are appended to an
  :class:`~olmo_core.hpo.state.EventLog` with monotonic ``decision_id``s, so the controller
  replays deterministically and recovers from a crash by re-reading the log.
- **Order-independence.** Worker results are sorted before they are folded in, so the derived
  state is identical no matter what order eight workers happen to finish in.

This module is pure ``numpy`` + standard library; it drives real GPU work only through the
``simulate`` callback (a synthetic objective in tests, the OLMo worker subprocess in production).
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .bttackler import BTTDiagnoser, BTTObservation
from .centaur import (
    AdvisorUnavailable,
    AskLedger,
    CentaurOverlay,
    LegalAction,
    LLMAdvisor,
    PendingAsk,
    build_advisor_state,
)
from .ftpfn import ObservedCurve, Posterior
from .ifbo import Candidate, IfBOCandidateGenerator, MFPIRandom, observed_f_best
from .ipbt import IPBTController, Member, WeightPolicy
from .objective import CENormalizer
from .state import ControllerState, Event, EventKind, EventLog, observation_hash, replay
from .types import (
    ActionKind,
    Allocation,
    BTTVerdictKind,
    CurvePoint,
    ProposalSource,
    SearchSpace,
    TrialStatus,
    WorkerObservation,
)

__all__ = ["PopulationRestartMode", "ControllerConfig", "HpoController"]


class PopulationRestartMode(str, Enum):
    IPBT_REFERENCE = "ipbt_reference"
    BTT_AGGREGATE = "btt_aggregate"


@dataclass
class ControllerConfig:
    target_tokens: int
    quantum: int
    n_fidelity_bins: int
    worker_count: int
    budget_tokens: int
    checkpoint_root: str
    seed: int = 0
    failure_penalty: float = 0.0
    restart_mode: PopulationRestartMode = PopulationRestartMode.IPBT_REFERENCE
    btt_restart_fraction: float = 0.5
    ensure_full_fidelity: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.btt_restart_fraction <= 1.0:
            raise ValueError("btt_restart_fraction must be in (0, 1]")
        if self.budget_tokens < self.target_tokens:
            raise ValueError("budget_tokens must fund at least one full-fidelity trial")


class _Proposer(Protocol):  # structural type for RandomProposer / CMAESProposer
    def ask(self, n: int) -> List[Tuple[float, ...]]:
        ...  # pragma: no cover


def _optional_state_dict(proposer: Optional[_Proposer]) -> Any:
    if proposer is None:
        return None
    method = getattr(proposer, "state_dict", None)
    return method() if callable(method) else None


def _optional_load_state_dict(proposer: Optional[_Proposer], state: Any) -> None:
    if proposer is None or state is None:
        return
    method = getattr(proposer, "load_state_dict", None)
    if not callable(method):
        raise ValueError("proposer snapshot exists but proposer cannot restore it")
    method(state)


class HpoController:
    def __init__(
        self,
        search_space: SearchSpace,
        normalizer: CENormalizer,
        posterior: Posterior,
        proposer: _Proposer,
        btt: BTTDiagnoser,
        config: ControllerConfig,
        *,
        centaur: Optional[CentaurOverlay] = None,
        advisor: Optional[LLMAdvisor] = None,
        ipbt: Optional[IPBTController] = None,
        ipbt_meta_proposer: Optional[_Proposer] = None,
        cma_replacement_proposer: Optional[_Proposer] = None,
        action_centaur: Optional[CentaurOverlay] = None,
        action_advisor: Optional[LLMAdvisor] = None,
    ) -> None:
        self.search_space = search_space
        self.normalizer = normalizer
        self.posterior = posterior
        self.proposer = proposer
        self.btt = btt
        self.config = config
        self.centaur = centaur
        self.advisor = advisor
        if (centaur is None) != (advisor is None):
            raise ValueError("centaur and advisor must be provided together")
        if (ipbt is None) != (ipbt_meta_proposer is None):
            raise ValueError("ipbt and ipbt_meta_proposer must be provided together")
        if (action_centaur is None) != (action_advisor is None):
            raise ValueError("action_centaur and action_advisor must be provided together")
        self.ipbt = ipbt
        self.ipbt_meta_proposer = ipbt_meta_proposer
        self._is_cma = hasattr(proposer, "population_size") and callable(
            getattr(proposer, "tell", None)
        )
        self._ask_ledger = AskLedger() if self._is_cma else None
        self.cma_replacement_proposer = cma_replacement_proposer
        self.action_centaur = action_centaur
        self.action_advisor = action_advisor

        self.log = EventLog()
        self.mfpi = MFPIRandom(
            posterior,
            n_fidelity_bins=config.n_fidelity_bins,
            target_tokens=config.target_tokens,
            normalizer=normalizer,
        )
        self._rng = np.random.default_rng(config.seed)
        self._trial_counter = 0
        self._next_curve_id = 1
        self._curve_ids: Dict[str, int] = {}
        self._configs: Dict[str, Tuple[float, ...]] = {}
        self._batch_counter = 0
        self._pending_batches: Dict[int, set[str]] = {}
        self._result_buffer: Dict[int, Dict[str, WorkerObservation]] = {}
        self._new_candidate_pool: Dict[str, Candidate] = {}
        self._candidate_ask_ids: Dict[str, int] = {}
        self._trial_ask_ids: Dict[str, int] = {}
        self._proposal_counter = 0
        self._action_proposal_counter = 0
        self._ipbt_metadata: Dict[str, Dict[str, Any]] = {}
        self._ipbt_planned_signatures: set[str] = set()
        self._ipbt_last_transition_fidelity = 0

    # --- derived state ---

    def state(self) -> ControllerState:
        return replay(self.log.events)

    def log_snapshot(self):
        return self.state().snapshot()

    def pending_allocations(self) -> List[Allocation]:
        """Reconstruct outstanding allocation grants for retry redispatch."""
        state = self.state()
        pending_ids = {
            trial_id
            for trial_id, record in state.trials.items()
            if record.pending_target_fidelity is not None
        }
        latest: Dict[str, Allocation] = {}
        for event in self.log:
            if event.kind is EventKind.ALLOCATION:
                allocation = Allocation.from_dict(event.payload)
                if allocation.trial_id in pending_ids:
                    latest[allocation.trial_id] = allocation
        return sorted(latest.values(), key=lambda allocation: allocation.decision_id)

    def final_evaluation_completed(self) -> bool:
        return bool(self.state().final_evaluations)

    def record_final_evaluation(self, payload: Dict[str, Any]) -> None:
        if self.final_evaluation_completed():
            raise ValueError("untouched final evaluation is already recorded")
        self._append(EventKind.FINAL_EVALUATION, copy.deepcopy(payload))
        self._snapshot_controller()

    def _append(self, kind: EventKind, payload: dict) -> None:
        seq = 0 if len(self.log) == 0 else self.log.events[-1].seq + 1
        self.log.append(Event(seq=seq, kind=kind, payload=payload))

    def _snapshot_controller(self) -> None:
        proposer_state = None
        state_dict = getattr(self.proposer, "state_dict", None)
        if callable(state_dict):
            proposer_state = state_dict()
        self._append(
            EventKind.CONTROLLER_SNAPSHOT,
            {
                "rng_state": copy.deepcopy(self._rng.bit_generator.state),
                "proposer_state": copy.deepcopy(proposer_state),
                "trial_counter": self._trial_counter,
                "next_curve_id": self._next_curve_id,
                "curve_ids": dict(self._curve_ids),
                "configs": {trial_id: list(config) for trial_id, config in self._configs.items()},
                "batch_counter": self._batch_counter,
                "pending_batches": {
                    str(batch_id): sorted(trial_ids)
                    for batch_id, trial_ids in self._pending_batches.items()
                },
                "new_candidate_pool": {
                    key: {
                        "curve_id": candidate.curve_id,
                        "unit_config": list(candidate.unit_config),
                        "base_tokens": candidate.base_tokens,
                        "is_continuation": candidate.is_continuation,
                        "source": candidate.source.value,
                    }
                    for key, candidate in self._new_candidate_pool.items()
                },
                "candidate_ask_ids": dict(self._candidate_ask_ids),
                "trial_ask_ids": dict(self._trial_ask_ids),
                "proposal_counter": self._proposal_counter,
                "action_proposal_counter": self._action_proposal_counter,
                "ask_ledger": (None if self._ask_ledger is None else self._ask_ledger.state_dict()),
                "ipbt_metadata": copy.deepcopy(self._ipbt_metadata),
                "ipbt_planned_signatures": sorted(self._ipbt_planned_signatures),
                "ipbt_last_transition_fidelity": self._ipbt_last_transition_fidelity,
                "ipbt_state": None if self.ipbt is None else self.ipbt.state_dict(),
                "ipbt_meta_proposer_state": _optional_state_dict(self.ipbt_meta_proposer),
                "cma_replacement_proposer_state": _optional_state_dict(
                    self.cma_replacement_proposer
                ),
            },
        )

    def restore_log(self, jsonl: str) -> None:
        """Restore the event log and all controller-local deterministic state."""
        restored = EventLog.from_jsonl(jsonl)
        state = replay(restored.events)
        self.log = restored
        snapshot = state.controller_snapshot
        if snapshot is not None:
            snapshot_seq = max(
                event.seq
                for event in restored
                if event.kind is EventKind.CONTROLLER_SNAPSHOT
            )
            self._rng.bit_generator.state = snapshot["rng_state"]
            proposer_state = snapshot.get("proposer_state")
            load_state_dict = getattr(self.proposer, "load_state_dict", None)
            if proposer_state is not None:
                if not callable(load_state_dict):
                    raise ValueError("proposer snapshot exists but proposer cannot restore it")
                load_state_dict(proposer_state)
            self._trial_counter = int(snapshot["trial_counter"])
            self._next_curve_id = int(snapshot["next_curve_id"])
            self._curve_ids = {str(key): int(value) for key, value in snapshot["curve_ids"].items()}
            self._configs = {
                str(key): tuple(float(value) for value in config)
                for key, config in snapshot["configs"].items()
            }
            self._batch_counter = int(snapshot.get("batch_counter", 0))
            self._pending_batches = {
                int(batch_id): set(trial_ids)
                for batch_id, trial_ids in snapshot.get("pending_batches", {}).items()
            }
            self._result_buffer = {}
            self._new_candidate_pool = {
                str(key): Candidate(
                    key=str(key),
                    curve_id=int(value["curve_id"]),
                    unit_config=tuple(float(x) for x in value["unit_config"]),
                    base_tokens=int(value["base_tokens"]),
                    is_continuation=bool(value["is_continuation"]),
                    source=ProposalSource(value["source"]),
                )
                for key, value in snapshot.get("new_candidate_pool", {}).items()
            }
            self._candidate_ask_ids = {
                str(key): int(value) for key, value in snapshot.get("candidate_ask_ids", {}).items()
            }
            self._trial_ask_ids = {
                str(key): int(value) for key, value in snapshot.get("trial_ask_ids", {}).items()
            }
            self._proposal_counter = int(snapshot.get("proposal_counter", 0))
            self._action_proposal_counter = int(snapshot.get("action_proposal_counter", 0))
            if self._ask_ledger is not None and snapshot.get("ask_ledger") is not None:
                self._ask_ledger.load_state_dict(snapshot["ask_ledger"])
            self._ipbt_metadata = copy.deepcopy(snapshot.get("ipbt_metadata", {}))
            self._ipbt_planned_signatures = set(snapshot.get("ipbt_planned_signatures", []))
            self._ipbt_last_transition_fidelity = int(
                snapshot.get("ipbt_last_transition_fidelity", 0)
            )
            if self.ipbt is not None and snapshot.get("ipbt_state") is not None:
                self.ipbt.load_state_dict(snapshot["ipbt_state"])
            _optional_load_state_dict(
                self.ipbt_meta_proposer,
                snapshot.get("ipbt_meta_proposer_state"),
            )
            _optional_load_state_dict(
                self.cma_replacement_proposer,
                snapshot.get("cma_replacement_proposer_state"),
            )
            # Allocation events are appended as a round is assembled. If a later
            # slot fails before the round-end snapshot, those grants are durable
            # but absent from the restored controller-local bookkeeping.
            for event in restored:
                if event.seq <= snapshot_seq or event.kind is not EventKind.ALLOCATION:
                    continue
                allocation = Allocation.from_dict(event.payload)
                if allocation.kind is ActionKind.START and allocation.trial_id not in self._configs:
                    self._configs[allocation.trial_id] = allocation.unit_config
                    self._curve_ids[allocation.trial_id] = self._next_curve_id
                    self._next_curve_id += 1
                    self._trial_counter += 1
                self._batch_counter = max(self._batch_counter, allocation.batch_id + 1)
                if self.action_centaur is not None:
                    self._action_proposal_counter += 1

            # Rebuild this from the authoritative event-sourced state even when
            # a snapshot exists, since that snapshot may predate trailing grants.
            pending_ids = {
                trial_id
                for trial_id, record in state.trials.items()
                if record.pending_target_fidelity is not None
            }
            latest_pending: Dict[str, Allocation] = {}
            for event in restored:
                if event.kind is EventKind.ALLOCATION:
                    allocation = Allocation.from_dict(event.payload)
                    if allocation.trial_id in pending_ids:
                        latest_pending[allocation.trial_id] = allocation
            self._pending_batches = {}
            for allocation in latest_pending.values():
                self._pending_batches.setdefault(allocation.batch_id, set()).add(
                    allocation.trial_id
                )
            return

        # Backward-compatible deterministic reconstruction for logs predating snapshots.
        self._trial_counter = 0
        self._next_curve_id = 1
        self._curve_ids = {}
        self._configs = {}
        self._batch_counter = 0
        self._pending_batches = {}
        self._result_buffer = {}
        self._new_candidate_pool = {}
        self._candidate_ask_ids = {}
        self._trial_ask_ids = {}
        self._proposal_counter = 0
        self._action_proposal_counter = 0
        self._ipbt_metadata = {}
        self._ipbt_planned_signatures = set()
        self._ipbt_last_transition_fidelity = 0
        for event in restored:
            if event.kind is not EventKind.ALLOCATION:
                continue
            allocation = Allocation.from_dict(event.payload)
            if allocation.kind is ActionKind.START and allocation.trial_id not in self._configs:
                self._configs[allocation.trial_id] = allocation.unit_config
                self._curve_ids[allocation.trial_id] = self._next_curve_id
                self._next_curve_id += 1
                self._trial_counter += 1
            self._batch_counter = max(self._batch_counter, allocation.batch_id + 1)
            self._pending_batches.setdefault(allocation.batch_id, set()).add(allocation.trial_id)
        for trial_id, record in state.trials.items():
            if record.pending_target_fidelity is None:
                for trial_ids in self._pending_batches.values():
                    trial_ids.discard(trial_id)
        self._pending_batches = {
            batch_id: trial_ids
            for batch_id, trial_ids in self._pending_batches.items()
            if trial_ids
        }

    def _observed(self, state: ControllerState) -> List[ObservedCurve]:
        curves: List[ObservedCurve] = []
        for tid, rec in state.trials.items():
            if not rec.curve or tid not in self._curve_ids:
                continue
            if rec.latest_verdict is not None and rec.latest_verdict.kind is BTTVerdictKind.FATAL:
                continue
            points = tuple(CurvePoint(int(tok), float(ce)) for tok, ce in rec.curve)
            curves.append(ObservedCurve(self._curve_ids[tid], self._configs[tid], points))
        return curves

    def _btt_cohort(self, completed_fidelity: int) -> Tuple[List[BTTObservation], Dict[str, float]]:
        observations: Dict[str, BTTObservation] = {}
        scores: Dict[str, float] = {}
        for event in self.log:
            if event.kind is not EventKind.OBSERVATION:
                continue
            payload = event.payload
            if int(payload["tokens"]) != completed_fidelity:
                continue
            trial_id = str(payload["trial_id"])
            ce = float(payload["ce"])
            observations[trial_id] = BTTObservation(
                trial_id=trial_id,
                completed_fidelity=completed_fidelity,
                observation_hash=str(payload["observation_hash"]),
                grad_norm_history=tuple(payload.get("grad_norm_history", ())),
                loss_history=tuple(payload.get("train_ce_history", ())),
                activation_ratio=payload.get("activation_ratio"),
                non_finite=bool(payload.get("numeric_failure", False)),
            )
            scores[trial_id] = 0.0 if not np.isfinite(ce) else self.normalizer.to_y(ce)
        return list(observations.values()), scores

    def _ifbo_meta_configs(
        self,
        members: Sequence[Member],
        count: int,
    ) -> List[Tuple[float, ...]]:
        assert isinstance(self.ipbt_meta_proposer, IfBOCandidateGenerator)
        incumbent_member = max(members, key=lambda member: (member.score, member.member_id))
        slate_size = max(count * 4, count + 1)
        observed = self._observed(self.state())
        represented = {curve.unit_config for curve in observed}
        configs: List[Tuple[float, ...]] = []
        while len(configs) < slate_size:
            generated = self.ipbt_meta_proposer.ask(
                slate_size - len(configs),
                incumbent=incumbent_member.unit_config,
            )
            configs.extend(config for config in generated if config not in represented)
        candidates = [
            Candidate(
                key=f"ipbt-ifbo-slate:{index}",
                curve_id=0,
                unit_config=config,
                base_tokens=incumbent_member.fidelity,
                is_continuation=False,
                source=ProposalSource.IFBO,
            )
            for index, config in enumerate(configs)
        ]
        selections = self.mfpi.select_batch(
            observed,
            candidates,
            count=count,
            rng=self._rng,
            f_best=observed_f_best(observed, self.normalizer),
        )
        return [candidates[selection.chosen_index].unit_config for selection in selections]

    def _eligible_continuations(self, state: ControllerState) -> List[Candidate]:
        initialization_fidelity: Optional[int] = None
        if (
            self.ipbt is not None
            and self.ipbt.config.initial_oversample is not None
            and not self._ipbt_planned_signatures
        ):
            completed_initial = sum(
                record.current_fidelity > 0
                and record.status not in (TrialStatus.RETIRED, TrialStatus.FAILED)
                for record in state.trials.values()
            )
            if completed_initial < self.ipbt.config.initial_oversample:
                return []
            paused_fidelities = [
                record.current_fidelity
                for record in state.trials.values()
                if record.current_fidelity > 0 and record.status is TrialStatus.PAUSED
            ]
            if paused_fidelities:
                initialization_fidelity = min(paused_fidelities)
        cands: List[Candidate] = []
        for tid, rec in sorted(state.trials.items()):
            if tid not in self._configs:
                continue
            if rec.current_fidelity >= self.config.target_tokens:
                continue
            if (
                initialization_fidelity is not None
                and rec.current_fidelity != initialization_fidelity
            ):
                continue
            if rec.status is not TrialStatus.PAUSED or rec.pending_target_fidelity is not None:
                continue
            verdict = rec.latest_verdict
            if verdict is None or not verdict.is_eligible_for_resume():
                continue
            if rec.latest_checkpoint_ref is None:
                continue
            cands.append(
                Candidate(
                    key=tid,
                    curve_id=self._curve_ids[tid],
                    unit_config=self._configs[tid],
                    base_tokens=rec.current_fidelity,
                    is_continuation=True,
                    source=ProposalSource.CMA,
                )
            )
        return cands

    def _preserves_full_fidelity_reserve(
        self,
        state: ControllerState,
        candidate: Candidate,
    ) -> bool:
        """
        Return whether granting ``candidate`` still funds one full-fidelity result.

        The ordinary acquisition policy may keep preferring broad first-rung exploration until
        the global token budget is exhausted. Reserve the cheapest path from an eligible or
        pending trial to the target so a comparison run cannot finish without a reportable
        winner. Near the end of the budget this naturally forces successive quanta onto the
        furthest-progressed viable trial.
        """

        projected_fidelity = 0
        for record in state.trials.values():
            if record.pending_target_fidelity is not None:
                projected_fidelity = max(projected_fidelity, record.pending_target_fidelity)
                continue
            if record.current_fidelity >= self.config.target_tokens:
                return True
            verdict = record.latest_verdict
            if (
                record.status is TrialStatus.PAUSED
                and verdict is not None
                and verdict.is_eligible_for_resume()
            ):
                projected_fidelity = max(projected_fidelity, record.current_fidelity)

        if candidate.is_continuation:
            current_fidelity = state.trials[candidate.key].current_fidelity
            candidate_target = min(
                current_fidelity + self.config.quantum,
                self.config.target_tokens,
            )
        else:
            transition = self._ipbt_metadata.get(candidate.key)
            if transition is None:
                current_fidelity = 0
                candidate_target = min(self.config.quantum, self.config.target_tokens)
            else:
                current_fidelity = int(transition["current_fidelity"])
                candidate_target = int(transition["target_fidelity"])

        charge = candidate_target - current_fidelity
        projected_fidelity = max(projected_fidelity, candidate_target)
        completion_reserve = self.config.target_tokens - projected_fidelity
        return (
            state.tokens_charged + charge + completion_reserve
            <= self.config.budget_tokens
        )

    def _full_fidelity_rescue_candidate(self, state: ControllerState) -> Optional[Candidate]:
        """Continue the best saturated incumbent when it is the only funded completion path."""
        if not self.config.ensure_full_fidelity:
            return None
        if any(
            record.current_fidelity >= self.config.target_tokens
            for record in state.trials.values()
        ):
            return None
        viable = []
        for trial_id, record in state.trials.items():
            verdict = record.latest_verdict
            if (
                trial_id not in self._configs
                or record.pending_target_fidelity is not None
                or record.current_fidelity <= 0
                or record.latest_checkpoint_ref is None
                or verdict is None
                or verdict.kind is not BTTVerdictKind.SATURATED
            ):
                continue
            completion_cost = self.config.target_tokens - record.current_fidelity
            if state.tokens_charged + completion_cost > self.config.budget_tokens:
                continue
            viable.append((record.current_fidelity, record.curve[-1][1], trial_id))
        if not viable:
            return None
        _, _, trial_id = min(viable, key=lambda item: (-item[0], item[1], item[2]))
        return Candidate(
            key=trial_id,
            curve_id=self._curve_ids[trial_id],
            unit_config=self._configs[trial_id],
            base_tokens=state.trials[trial_id].current_fidelity,
            is_continuation=True,
            source=ProposalSource.IFBO,
        )

    def _maybe_plan_ipbt(self) -> None:
        if self.ipbt is None or self.ipbt_meta_proposer is None:
            return
        state = self.state()
        by_fidelity: Dict[int, List[Member]] = {}
        for trial_id, record in state.trials.items():
            if (
                trial_id not in self._configs
                or record.pending_target_fidelity is not None
                or record.current_fidelity >= self.config.target_tokens
                or record.latest_verdict is None
            ):
                continue
            values = [ce for tokens, ce in record.curve if tokens == record.current_fidelity]
            if self.config.restart_mode is PopulationRestartMode.IPBT_REFERENCE and (
                record.latest_checkpoint_ref is None
                or not record.latest_verdict.is_incumbent_candidate()
                or not values
            ):
                continue
            by_fidelity.setdefault(record.current_fidelity, []).append(
                Member(
                    member_id=trial_id,
                    lineage_id=record.lineage_id or trial_id,
                    unit_config=self._configs[trial_id],
                    score=(self.normalizer.to_y(values[-1]) if values else -1.0),
                    fidelity=record.current_fidelity,
                    checkpoint_ref=record.latest_checkpoint_ref or "",
                    optimizer_state_valid=(
                        record.latest_checkpoint_ref is not None
                        and record.latest_verdict.kind is not BTTVerdictKind.FATAL
                    ),
                    comparison_stratum=(
                        "inherited" if record.parent_trial_id is not None else "from_scratch"
                    ),
                )
            )

        population_size = self.ipbt.config.population_size
        for fidelity in sorted(by_fidelity, reverse=True):
            members = sorted(by_fidelity[fidelity], key=lambda member: member.member_id)
            initial_oversample = (
                self.ipbt.config.initial_oversample if not self._ipbt_planned_signatures else None
            )
            expected_size = initial_oversample or population_size
            if len(members) != expected_size:
                continue
            if fidelity - self._ipbt_last_transition_fidelity < self.ipbt.restart_tracker.interval:
                continue
            if initial_oversample is not None:
                oversampled_members = members
                members = self.ipbt.select_initial_population(oversampled_members)
                selected_ids = {member.member_id for member in members}
                for rejected in oversampled_members:
                    if rejected.member_id not in selected_ids:
                        self._append(
                            EventKind.STATUS,
                            {
                                "trial_id": rejected.member_id,
                                "status": TrialStatus.RETIRED.value,
                                "reason": "ipbt_initial_oversample_rejected",
                            },
                        )
            signature = hashlib.sha256(
                repr((fidelity, tuple(member.member_id for member in members))).encode()
            ).hexdigest()
            if signature in self._ipbt_planned_signatures:
                return
            bo_configs = (
                self._ifbo_meta_configs(members, population_size)
                if isinstance(self.ipbt_meta_proposer, IfBOCandidateGenerator)
                else self.ipbt_meta_proposer.ask(population_size)
            )
            restart_evidence: Dict[str, Any]
            if self.config.restart_mode is PopulationRestartMode.BTT_AGGREGATE:
                bad_ids = []
                for member in members:
                    verdict = state.trials[member.member_id].latest_verdict
                    assert verdict is not None
                    if verdict.kind in (
                        BTTVerdictKind.DEGRADED,
                        BTTVerdictKind.FATAL,
                        BTTVerdictKind.SATURATED,
                    ):
                        bad_ids.append(member.member_id)
                bad_fraction = len(bad_ids) / len(members)
                restart_requested = bad_fraction >= self.config.btt_restart_fraction
                restart_evidence = {
                    "mode": PopulationRestartMode.BTT_AGGREGATE.value,
                    "bad_trial_ids": bad_ids,
                    "bad_fraction": bad_fraction,
                    "threshold": self.config.btt_restart_fraction,
                }
                if restart_requested:
                    self.ipbt.restart_tracker.interval *= 2
            else:
                restart_requested = self.ipbt.restart_tracker.update(
                    max(member.score for member in members)
                )
                restart_evidence = {
                    "mode": PopulationRestartMode.IPBT_REFERENCE.value,
                    "bad_trial_ids": [],
                    "bad_fraction": 0.0,
                    "threshold": None,
                }
            if restart_requested:
                has_safe_donor = False
                for member in members:
                    verdict = state.trials[member.member_id].latest_verdict
                    assert verdict is not None
                    if verdict.kind is not BTTVerdictKind.FATAL and member.checkpoint_ref:
                        has_safe_donor = True
                        break
                restart_plan = (
                    self.ipbt.restart_population(
                        members,
                        rng=self._rng,
                        bo_configs=bo_configs,
                    )
                    if has_safe_donor
                    else self.ipbt.fresh_restart_population(
                        members,
                        rng=self._rng,
                        bo_configs=bo_configs,
                    )
                )
                transition_kind = "restart"
                kept_ids: List[str] = []
                transition_descendants = restart_plan.copies + restart_plan.descendants
            else:
                generation_plan = self.ipbt.plan_generation(
                    members,
                    rng=self._rng,
                    bo_configs=bo_configs,
                )
                transition_kind = "generation"
                kept_ids = [member.member_id for member in generation_plan.kept]
                transition_descendants = generation_plan.descendants
            descendants_payload: List[Dict[str, Any]] = []
            by_member_id = {member.member_id: member for member in members}
            for descendant in transition_descendants:
                self._append(
                    EventKind.STATUS,
                    {
                        "trial_id": descendant.slot_id,
                        "status": TrialStatus.RETIRED.value,
                        "reason": "ipbt_replaced",
                    },
                )
                donor = None if descendant.donor_id is None else by_member_id[descendant.donor_id]
                current_fidelity = 0 if donor is None else donor.fidelity
                target_fidelity = min(
                    current_fidelity + self.config.quantum,
                    self.config.target_tokens,
                )
                if target_fidelity <= current_fidelity:
                    continue
                key = f"ipbt:{signature}:{descendant.slot_id}"
                self._new_candidate_pool[key] = Candidate(
                    key=key,
                    curve_id=(
                        self._curve_ids[donor.member_id]
                        if donor is not None and descendant.weight_policy is WeightPolicy.PURE_COPY
                        else 0
                    ),
                    unit_config=descendant.unit_config,
                    base_tokens=current_fidelity,
                    is_continuation=False,
                    source=ProposalSource.IPBT_META,
                )
                metadata: Dict[str, Any] = {
                    "transition_signature": signature,
                    "transition_kind": transition_kind,
                    "lineage_id": descendant.lineage_id,
                    "parent_lineage_id": descendant.parent_lineage_id,
                    "parent_trial_id": descendant.donor_id,
                    "checkpoint_ref": None if donor is None else donor.checkpoint_ref,
                    "current_fidelity": current_fidelity,
                    "target_fidelity": target_fidelity,
                    "weight_policy": descendant.weight_policy.value,
                    "optimizer_reset": descendant.optimizer_reset,
                    "weight_scale": descendant.weight_scale,
                    "schedule_age_tokens": descendant.schedule_age_tokens,
                    "hp_source": descendant.hp_source.value,
                }
                self._ipbt_metadata[key] = metadata
                descendants_payload.append(
                    {
                        "slot_id": descendant.slot_id,
                        "unit_config": list(descendant.unit_config),
                        **metadata,
                    }
                )
            self._append(
                EventKind.IPBT_TRANSITION,
                {
                    "signature": signature,
                    "transition_kind": transition_kind,
                    "completed_fidelity": fidelity,
                    "restart_evidence": restart_evidence,
                    "kept": kept_ids,
                    "descendants": descendants_payload,
                },
            )
            self._ipbt_planned_signatures.add(signature)
            self._ipbt_last_transition_fidelity = fidelity
            return

    def _new_candidates(
        self,
        n: int,
        *,
        incumbent: Optional[Tuple[float, ...]] = None,
    ) -> List[Candidate]:
        if self._new_candidate_pool:
            return list(self._new_candidate_pool.values())
        raw_asks: List[Optional[PendingAsk]]
        mapped_ask_ids = set(self._candidate_ask_ids.values()) | set(self._trial_ask_ids.values())
        retry_asks = (
            []
            if self._ask_ledger is None
            else [
                ask
                for ask in self._ask_ledger.pending_asks(stratum="from_scratch")
                if ask.ask_id not in mapped_ask_ids
            ]
        )
        if retry_asks:
            raw_asks = list(retry_asks)
            raw_configs = [ask.unit_config for ask in retry_asks]
        else:
            if self._ask_ledger is not None and self._ask_ledger.has_open_generation(
                stratum="from_scratch"
            ):
                return []
            ask_count = int(getattr(self.proposer, "population_size")) if self._is_cma else n
            proposed = (
                self.proposer.ask(ask_count, incumbent=incumbent)
                if isinstance(self.proposer, IfBOCandidateGenerator)
                else self.proposer.ask(ask_count)
            )
            raw_configs = [tuple(float(x) for x in cfg) for cfg in proposed]
            if self._ask_ledger is not None:
                raw_asks = list(self._ask_ledger.register(raw_configs, stratum="from_scratch"))
            else:
                raw_asks = [None] * len(raw_configs)
        evaluated_configs: List[Tuple[float, ...]] = []
        sources: List[ProposalSource] = []
        asks: List[Optional[PendingAsk]] = []
        for cma_config, raw_ask in zip(raw_configs, raw_asks):
            config = cma_config
            source = (
                ProposalSource.CMA
                if self._is_cma
                else getattr(
                    self.proposer,
                    "proposal_source",
                    ProposalSource.RANDOM,
                )
            )
            if self.centaur is not None:
                assert self.advisor is not None
                state_method = getattr(self.proposer, "state", None)
                cma_state: Dict[str, Any] = state_method() if callable(state_method) else {}
                controller_state = self.state()
                ranked = sorted(
                    (
                        {
                            "trial_id": trial_id,
                            "ce": record.curve[-1][1],
                            "fidelity": record.current_fidelity,
                        }
                        for trial_id, record in controller_state.trials.items()
                        if record.curve
                    ),
                    key=lambda item: (item["ce"], item["trial_id"]),
                )
                advisor_state = build_advisor_state(
                    cma_mean=cma_state.get("mean", [0.5] * self.search_space.ndim),
                    cma_sigma=float(cma_state.get("sigma", 0.0)),
                    cma_cov=cma_state.get(
                        "covariance",
                        np.eye(self.search_space.ndim).tolist(),
                    ),
                    cma_proposal=cma_config,
                    ifbo_action={"kind": "allocation_pending"},
                    ifbo_alternatives=[],
                    population_lineages=[
                        {
                            "trial_id": trial_id,
                            "lineage_id": record.lineage_id,
                            "status": (None if record.status is None else record.status.value),
                            "fidelity": record.current_fidelity,
                        }
                        for trial_id, record in sorted(controller_state.trials.items())
                    ],
                    btt_evidence=[
                        {
                            "trial_id": trial_id,
                            "kind": record.latest_verdict.kind.value,
                            "indicators": list(record.latest_verdict.indicators),
                        }
                        for trial_id, record in sorted(controller_state.trials.items())
                        if record.latest_verdict is not None
                    ],
                    incumbent={} if not ranked else ranked[0],
                    top_five=ranked[:5],
                    recent_decisions=[
                        event.payload
                        for event in self.log.events[-20:]
                        if event.kind is EventKind.ALLOCATION
                    ][-10:],
                    bounds=[[0.0, 1.0] for _ in range(self.search_space.ndim)],
                    remaining_budget=(self.config.budget_tokens - controller_state.tokens_charged),
                    action_schema={
                        "kinds": ["start"],
                        "config_dimensions": self.search_space.ndim,
                    },
                )
                advisor_state["proposal_id"] = self._proposal_counter
                try:
                    config, source, record = self.centaur.propose(
                        proposal_id=self._proposal_counter,
                        cma_config=cma_config,
                        advisor=self.advisor,
                        state=advisor_state,
                    )
                except AdvisorUnavailable as error:
                    record = error.record
                    self._append(
                        EventKind.ADVISOR,
                        {
                            "proposal_id": self._proposal_counter,
                            "prompt_state": (
                                advisor_state if record is None else record.prompt_state
                            ),
                            "response": (
                                None
                                if record is None
                                else {
                                    "action": record.response.action,
                                    "raw_text": record.response.raw_text,
                                    "model": record.response.model,
                                    "version": record.response.version,
                                    "latency_ms": record.response.latency_ms,
                                }
                            ),
                            "error": str(error),
                        },
                    )
                    self._proposal_counter += 1
                    self._snapshot_controller()
                    raise
                if record is not None:
                    self._append(
                        EventKind.ADVISOR,
                        {
                            "proposal_id": self._proposal_counter,
                            "prompt_state": record.prompt_state,
                            "response": {
                                "action": record.response.action,
                                "raw_text": record.response.raw_text,
                                "model": record.response.model,
                                "version": record.response.version,
                                "latency_ms": record.response.latency_ms,
                            },
                        },
                    )
            active_ask = raw_ask
            if self._ask_ledger is not None and raw_ask is not None and config != cma_config:
                active_ask = self._ask_ledger.replace_pending(raw_ask.ask_id, config)
            evaluated_configs.append(config)
            sources.append(source)
            asks.append(active_ask)
            self._proposal_counter += 1

        for index, (config, source, ask) in enumerate(zip(evaluated_configs, sources, asks)):
            key = (
                f"new:ask:{ask.ask_id}"
                if ask is not None
                else "new:"
                + hashlib.sha256(f"{self._proposal_counter}:{index}:{config}".encode()).hexdigest()[
                    :12
                ]
            )
            candidate = Candidate(
                key=key,
                curve_id=0,
                unit_config=config,
                base_tokens=min(self.config.quantum, self.config.target_tokens),
                is_continuation=False,
                source=source,
            )
            self._new_candidate_pool[key] = candidate
            if ask is not None:
                self._candidate_ask_ids[key] = ask.ask_id
        return list(self._new_candidate_pool.values())

    def _replace_censored_cma_ask(self, ask_id: int) -> None:
        if self._ask_ledger is None or self.cma_replacement_proposer is None:
            raise RuntimeError("censored CMA ask requires a configured replacement proposer")
        config = tuple(float(value) for value in self.cma_replacement_proposer.ask(1)[0])
        replacement = self._ask_ledger.replace_censored(ask_id, config)
        key = f"new:ask:{replacement.ask_id}"
        self._new_candidate_pool[key] = Candidate(
            key=key,
            curve_id=0,
            unit_config=config,
            base_tokens=min(self.config.quantum, self.config.target_tokens),
            is_continuation=False,
            source=ProposalSource.CMA,
        )
        self._candidate_ask_ids[key] = replacement.ask_id

    def _decide_action(
        self,
        *,
        default_action: Dict[str, Any],
        state: Dict[str, Any],
    ):
        assert self.action_centaur is not None
        assert self.action_advisor is not None
        try:
            return self.action_centaur.decide(
                proposal_id=self._action_proposal_counter,
                default_action=default_action,
                advisor=self.action_advisor,
                expected_dim=self.search_space.ndim,
                state=state,
            )
        except AdvisorUnavailable as error:
            record = error.record
            self._append(
                EventKind.ADVISOR,
                {
                    "scope": "multi_action",
                    "proposal_id": self._action_proposal_counter,
                    "prompt_state": (state if record is None else record.prompt_state),
                    "response": (
                        None
                        if record is None
                        else {
                            "action": record.response.action,
                            "raw_text": record.response.raw_text,
                            "model": record.response.model,
                            "version": record.response.version,
                            "latency_ms": record.response.latency_ms,
                        }
                    ),
                    "error": str(error),
                },
            )
            self._snapshot_controller()
            raise

    def _resolve_advisor_action(
        self,
        action: Dict[str, Any],
        candidates: Sequence[Candidate],
    ) -> Candidate:
        action_kind = LegalAction(action["kind"])
        if action_kind is LegalAction.RESUME:
            matching = [
                candidate
                for candidate in candidates
                if candidate.is_continuation and candidate.key == action["trial_id"]
            ]
            if len(matching) != 1:
                raise ValueError("advisor selected an ineligible resume")
            return matching[0]
        if action_kind is LegalAction.START:
            return Candidate(
                key=f"llm-action:{self._action_proposal_counter}",
                curve_id=0,
                unit_config=tuple(float(x) for x in action["unit_config"]),
                base_tokens=min(
                    self.config.quantum,
                    self.config.target_tokens,
                ),
                is_continuation=False,
                source=ProposalSource.LLM,
            )
        if action_kind is LegalAction.IPBT_EXPLOIT:
            target_slot_id = action["target_slot_id"]
            matching = [
                candidate
                for candidate in candidates
                if candidate.source is ProposalSource.IPBT_META
                and candidate.key.endswith(f":{target_slot_id}")
                and self._ipbt_metadata.get(candidate.key, {}).get("parent_trial_id")
                == action["donor_id"]
            ]
            if len(matching) != 1:
                raise ValueError("advisor selected an unavailable IPBT exploit")
            return matching[0]
        restart_id = action["restart_id"]
        target_slot_id = action["target_slot_id"]
        matching = [
            candidate
            for candidate in candidates
            if candidate.source is ProposalSource.IPBT_META
            and self._ipbt_metadata.get(candidate.key, {}).get("transition_signature") == restart_id
            and candidate.key.endswith(f":{target_slot_id}")
            and self._ipbt_metadata.get(candidate.key, {}).get("transition_kind") == "restart"
        ]
        if len(matching) != 1:
            raise ValueError("advisor selected an unavailable IPBT restart")
        return matching[0]

    # --- decision round ---

    def propose_round(self) -> List[Allocation]:
        pending_count = sum(len(trial_ids) for trial_ids in self._pending_batches.values())
        available_workers = self.config.worker_count - pending_count
        if available_workers <= 0:
            return []
        allocations: List[Allocation] = []
        batch_id = self._batch_counter
        fantasy_context = self._observed(self.state())
        for slot in range(available_workers):
            rescue_key: Optional[str] = None
            state = self.state()
            f_best = observed_f_best(fantasy_context, self.normalizer)
            incumbent_trial = min(
                (
                    (trial_id, record.curve[-1][1])
                    for trial_id, record in state.trials.items()
                    if trial_id in self._configs
                    and record.curve
                    and (
                        record.latest_verdict is None
                        or record.latest_verdict.kind is not BTTVerdictKind.FATAL
                    )
                ),
                key=lambda item: (item[1], item[0]),
                default=None,
            )
            incumbent = None if incumbent_trial is None else self._configs[incumbent_trial[0]]
            suppress_new_candidates = (
                self.ipbt is not None
                and self.ipbt.config.initial_oversample is not None
                and not self._ipbt_planned_signatures
                and sum(
                    record.current_fidelity > 0
                    and record.status not in (TrialStatus.RETIRED, TrialStatus.FAILED)
                    for record in state.trials.values()
                )
                >= self.ipbt.config.initial_oversample
            )
            while True:
                new_candidates = (
                    []
                    if suppress_new_candidates
                    else self._new_candidates(
                        available_workers - slot,
                        incumbent=incumbent,
                    )
                )
                represented_configs = {curve.unit_config for curve in fantasy_context}
                duplicate_keys = {
                    candidate.key
                    for candidate in new_candidates
                    if candidate.curve_id <= 0 and candidate.unit_config in represented_configs
                }
                if not duplicate_keys:
                    break
                for key in duplicate_keys:
                    self._new_candidate_pool.pop(key, None)
                    self._ipbt_metadata.pop(key, None)
                    self._candidate_ask_ids.pop(key, None)
            candidates = [
                candidate
                for candidate in self._eligible_continuations(state) + new_candidates
                if self._preserves_full_fidelity_reserve(state, candidate)
            ]
            if not candidates:
                rescue = self._full_fidelity_rescue_candidate(state)
                if rescue is not None:
                    candidates = [rescue]
                    rescue_key = rescue.key
            if not candidates:
                break
            sel = self.mfpi.select_batch(
                fantasy_context,
                candidates,
                count=1,
                rng=self._rng,
                f_best=f_best,
            )[0]
            cand = candidates[sel.chosen_index]
            ifbo_selected_cand = cand
            if self.action_centaur is not None:
                assert self.action_advisor is not None
                default_action: Dict[str, Any] = (
                    {"kind": LegalAction.RESUME.value, "trial_id": cand.key}
                    if cand.is_continuation
                    else {
                        "kind": LegalAction.START.value,
                        "unit_config": list(cand.unit_config),
                    }
                )
                action, record = self._decide_action(
                    default_action=default_action,
                    state={
                        "proposal_id": self._action_proposal_counter,
                        "default_action": default_action,
                        "ifbo_action": {
                            **default_action,
                            "candidate_key": cand.key,
                            "mfpi_score": sel.mfpi_score,
                            "horizon": sel.horizon,
                            "threshold": sel.threshold,
                        },
                        "ifbo_alternatives": sorted(
                            [
                                {
                                    "candidate_key": candidates[index].key,
                                    "unit_config": list(candidates[index].unit_config),
                                    "is_continuation": candidates[index].is_continuation,
                                    "mfpi_score": float(score),
                                }
                                for index, score in zip(sel.score_indices, sel.scores)
                                if index != sel.chosen_index
                            ],
                            key=lambda item: (
                                -item["mfpi_score"],
                                item["candidate_key"],
                            ),
                        ),
                        "eligible_resumes": [
                            candidate.key for candidate in candidates if candidate.is_continuation
                        ],
                        "ipbt_actions": [
                            {
                                "candidate_key": candidate.key,
                                "metadata": self._ipbt_metadata.get(candidate.key),
                            }
                            for candidate in candidates
                            if candidate.source is ProposalSource.IPBT_META
                        ],
                        "population_lineages": [
                            {
                                "trial_id": trial_id,
                                "lineage_id": record.lineage_id,
                                "status": (None if record.status is None else record.status.value),
                                "fidelity": record.current_fidelity,
                            }
                            for trial_id, record in sorted(state.trials.items())
                        ],
                        "btt_evidence": [
                            {
                                "trial_id": trial_id,
                                "kind": record.latest_verdict.kind.value,
                                "indicators": list(record.latest_verdict.indicators),
                                "disposition": (
                                    None
                                    if record.latest_verdict.disposition is None
                                    else record.latest_verdict.disposition.value
                                ),
                            }
                            for trial_id, record in sorted(state.trials.items())
                            if record.latest_verdict is not None
                        ],
                        "remaining_budget": (
                            self.config.budget_tokens - self.state().tokens_charged
                        ),
                    },
                )
                if record is not None:
                    self._append(
                        EventKind.ADVISOR,
                        {
                            "scope": "multi_action",
                            "proposal_id": self._action_proposal_counter,
                            "prompt_state": record.prompt_state,
                            "response": {
                                "action": record.response.action,
                                "raw_text": record.response.raw_text,
                                "model": record.response.model,
                                "version": record.response.version,
                                "latency_ms": record.response.latency_ms,
                            },
                        },
                    )
                    try:
                        cand = self._resolve_advisor_action(action, candidates)
                    except ValueError as error:
                        self._append(
                            EventKind.ADVISOR,
                            {
                                "scope": "multi_action",
                                "proposal_id": self._action_proposal_counter,
                                "prompt_state": record.prompt_state,
                                "response": {
                                    "action": record.response.action,
                                    "raw_text": record.response.raw_text,
                                    "model": record.response.model,
                                    "version": record.response.version,
                                    "latency_ms": record.response.latency_ms,
                                },
                                "error": str(error),
                            },
                        )
                        self._snapshot_controller()
                        raise
            state = self.state()  # re-derive so budget/decision-id reflect prior appends

            if cand.is_continuation:
                rec = state.trials[cand.key]
                current = rec.current_fidelity
                target = min(current + self.config.quantum, self.config.target_tokens)
                if target <= current:
                    continue
                verdict = rec.latest_verdict
                alloc = Allocation(
                    decision_id=state.next_decision_id,
                    kind=ActionKind.RESUME,
                    trial_id=cand.key,
                    parent_trial_id=None,
                    unit_config=cand.unit_config,
                    realized_hps=self.search_space.from_unit(cand.unit_config),
                    current_fidelity=current,
                    target_fidelity=target,
                    checkpoint_ref=rec.latest_checkpoint_ref,
                    horizon=sel.horizon,
                    threshold=sel.threshold,
                    mfpi_score=sel.mfpi_score,
                    tie_break=sel.tie_break,
                    source=ProposalSource.IFBO,
                    verdict_id=None if verdict is None else verdict.observation_hash,
                    batch_id=batch_id,
                    transition=(
                        {"full_fidelity_rescue": True}
                        if cand.key == rescue_key
                        else None
                    ),
                )
            else:
                trial_id = f"t{self._trial_counter}_0"
                self._trial_counter += 1
                self._curve_ids[trial_id] = self._next_curve_id
                self._next_curve_id += 1
                self._configs[trial_id] = cand.unit_config
                transition = self._ipbt_metadata.get(cand.key)
                current_fidelity = 0 if transition is None else int(transition["current_fidelity"])
                target_fidelity = (
                    min(self.config.quantum, self.config.target_tokens)
                    if transition is None
                    else int(transition["target_fidelity"])
                )
                alloc = Allocation(
                    decision_id=state.next_decision_id,
                    kind=ActionKind.START,
                    trial_id=trial_id,
                    parent_trial_id=(None if transition is None else transition["parent_trial_id"]),
                    unit_config=cand.unit_config,
                    realized_hps=self.search_space.from_unit(cand.unit_config),
                    current_fidelity=current_fidelity,
                    target_fidelity=target_fidelity,
                    checkpoint_ref=(None if transition is None else transition["checkpoint_ref"]),
                    horizon=sel.horizon,
                    threshold=sel.threshold,
                    mfpi_score=sel.mfpi_score,
                    tie_break=sel.tie_break,
                    source=cand.source,
                    batch_id=batch_id,
                    transition=transition,
                )
            charge = alloc.target_fidelity - alloc.current_fidelity
            if state.tokens_charged + charge > self.config.budget_tokens:
                continue
            self._append(EventKind.ALLOCATION, alloc.to_dict())
            if batch_id not in self._pending_batches:
                self._pending_batches[batch_id] = set()
                self._batch_counter += 1
            self._pending_batches[batch_id].add(alloc.trial_id)
            if alloc.kind is ActionKind.START:
                consumed_keys = {cand.key, ifbo_selected_cand.key}
                for consumed_key in consumed_keys:
                    self._new_candidate_pool.pop(consumed_key, None)
                    self._ipbt_metadata.pop(consumed_key, None)
                ask_id = None
                for consumed_key in consumed_keys:
                    candidate_ask_id = self._candidate_ask_ids.pop(consumed_key, None)
                    if candidate_ask_id is not None:
                        ask_id = candidate_ask_id
                if ask_id is not None:
                    self._trial_ask_ids[alloc.trial_id] = ask_id
            allocations.append(alloc)
            if self.action_centaur is not None:
                self._action_proposal_counter += 1
            fantasy_context.append(
                ObservedCurve(
                    curve_id=self._curve_ids[alloc.trial_id],
                    unit_config=alloc.unit_config,
                    points=(
                        CurvePoint(
                            tokens=alloc.target_fidelity,
                            ce=self.normalizer.ce_for_y(f_best),
                        ),
                    ),
                )
            )
            # Make every successfully assembled grant recoverable even if a
            # later slot in this same proposal round raises.
            self._snapshot_controller()
        self._snapshot_controller()
        return allocations

    # --- result ingestion ---

    def ingest(self, results: Sequence[WorkerObservation]) -> None:
        """Buffer asynchronous results and fold only complete allocation batches."""
        for result in results:
            matching = [
                batch_id
                for batch_id, trial_ids in self._pending_batches.items()
                if result.trial_id in trial_ids
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"worker result for {result.trial_id} does not identify one pending batch"
                )
            batch_id = matching[0]
            buffer = self._result_buffer.setdefault(batch_id, {})
            if result.trial_id in buffer:
                raise ValueError(f"duplicate worker result for {result.trial_id}")
            buffer[result.trial_id] = result

        while self._pending_batches:
            batch_id = min(self._pending_batches)
            expected = self._pending_batches[batch_id]
            buffered = self._result_buffer.get(batch_id, {})
            if set(buffered) != expected:
                break
            self._ingest_complete_batch(list(buffered.values()))
            del self._result_buffer[batch_id]
            del self._pending_batches[batch_id]
            self._snapshot_controller()
            self.retry_pending_tell()

    def _ingest_complete_batch(self, results: Sequence[WorkerObservation]) -> None:
        sorted_results = sorted(results, key=lambda item: (item.trial_id, item.tokens))
        for result in sorted_results:
            trial_id = result.trial_id
            tokens = result.tokens
            ce = float(result.heldout_ce)
            obs_hash = observation_hash(
                trial_id,
                tokens,
                ce,
                train_ce_history=result.train_ce_history,
                grad_norm_history=result.grad_norm_history,
                activation_ratio=result.activation_ratio,
                numeric_failure=result.numeric_failure,
            )
            self._append(
                EventKind.OBSERVATION,
                {
                    "trial_id": trial_id,
                    "tokens": tokens,
                    "ce": ce,
                    "train_ce_history": list(result.train_ce_history),
                    "grad_norm_history": list(result.grad_norm_history),
                    "activation_ratio": result.activation_ratio,
                    "numeric_failure": result.numeric_failure,
                    "observation_hash": obs_hash,
                    "checkpoint_ref": result.checkpoint_ref,
                    "accelerator_seconds": result.accelerator_seconds,
                },
            )

        verdicts = {}
        for fidelity in sorted({result.tokens for result in sorted_results}):
            cohort, scores = self._btt_cohort(fidelity)
            verdicts.update(self.btt.diagnose_cohort(cohort, scores=scores))
        state_before_verdicts = self.state()
        for trial_id, verdict in sorted(verdicts.items()):
            record = state_before_verdicts.trials.get(trial_id)
            if (
                record is None
                or record.pending_target_fidelity is not None
                or record.latest_observation_fidelity != verdict.completed_fidelity
                or record.latest_observation_hash != verdict.observation_hash
            ):
                continue
            assert verdict.disposition is not None
            self._append(
                EventKind.VERDICT,
                {
                    "kind": verdict.kind.value,
                    "indicators": list(verdict.indicators),
                    "trial_id": verdict.trial_id,
                    "completed_fidelity": verdict.completed_fidelity,
                    "observation_hash": verdict.observation_hash,
                    "profile_version": verdict.profile_version,
                    "spared_by_reserve": verdict.spared_by_reserve,
                    "protected_by_peer_rank": verdict.protected_by_peer_rank,
                    "disposition": verdict.disposition.value,
                },
            )
        for result in sorted_results:
            trial_id = result.trial_id
            tokens = result.tokens
            ce = float(result.heldout_ce)
            verdict = verdicts[trial_id]
            ask_id = self._trial_ask_ids.get(trial_id)
            if ask_id is not None and self._ask_ledger is not None:
                if result.numeric_failure or verdict.kind is BTTVerdictKind.FATAL:
                    self._ask_ledger.fail(ask_id, penalty=self.config.failure_penalty)
                elif tokens == self.config.target_tokens and verdict.is_incumbent_candidate():
                    self._ask_ledger.resolve(
                        ask_id,
                        score=self.normalizer.to_ftpfn_y(ce),
                        evaluated_config=self._configs[trial_id],
                    )
                elif not verdict.is_eligible_for_resume():
                    self._ask_ledger.censor(ask_id)
                    self._replace_censored_cma_ask(ask_id)

        self._maybe_plan_ipbt()

    def retry_pending_tell(self) -> None:
        """Retry any complete, resolved CMA generation without replaying observations."""
        if self._ask_ledger is None or not self._ask_ledger.can_collect_tell(
            stratum="from_scratch", allow_inherited=False
        ):
            return
        solutions = self._ask_ledger.collect_tell(
            stratum="from_scratch",
            allow_inherited=False,
            consume=False,
        )
        tell = getattr(self.proposer, "tell")
        tell(solutions)
        self._ask_ledger.mark_told(stratum="from_scratch", allow_inherited=False)
        self._snapshot_controller()

    # --- driver + reporting ---

    def run(self, rounds: int, simulate: Callable[[Sequence[float], float], float]) -> None:
        for _ in range(rounds):
            allocations = self.propose_round()
            if not allocations:
                break
            results = []
            for allocation in allocations:
                ce = simulate(
                    allocation.unit_config,
                    allocation.target_fidelity / self.config.target_tokens,
                )
                results.append(
                    WorkerObservation(
                        trial_id=allocation.trial_id,
                        tokens=allocation.target_fidelity,
                        heldout_ce=ce,
                        train_ce_history=(ce,),
                        grad_norm_history=(1.0,),
                        activation_ratio=0.5,
                        numeric_failure=not np.isfinite(ce),
                        checkpoint_ref=f"synthetic://{allocation.trial_id}/{allocation.target_fidelity}",
                    )
                )
            self.ingest(results)

    def top_candidates(self, limit: int = 5) -> List[Tuple[str, Tuple[float, ...], float]]:
        """Return the best full-fidelity candidates in ascending CE order."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        state = self.state()
        candidates: List[Tuple[str, Tuple[float, ...], float]] = []
        for tid, rec in state.trials.items():
            if tid not in self._configs or not rec.curve:
                continue
            if rec.latest_verdict is not None and not rec.latest_verdict.is_incumbent_candidate():
                continue
            if rec.current_fidelity != self.config.target_tokens:
                continue
            final_values = [ce for tokens, ce in rec.curve if tokens == self.config.target_tokens]
            if not final_values:
                continue
            final_ce = final_values[-1]
            candidates.append((tid, self._configs[tid], final_ce))
        candidates.sort(key=lambda candidate: (candidate[2], candidate[0]))
        return candidates[:limit]

    def best(self) -> Tuple[str, Tuple[float, ...], float]:
        """The incumbent: the observed point with the lowest CE (highest objective)."""

        candidates = self.top_candidates(1)
        if not candidates:
            raise RuntimeError("no completed observations to report a best from")
        return candidates[0]
