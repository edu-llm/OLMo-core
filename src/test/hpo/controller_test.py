import numpy as np
import pytest

from olmo_core.hpo.bttackler import BTTConfig, BTTDiagnoser
from olmo_core.hpo.centaur import AdvisorResponse, CentaurOverlay
from olmo_core.hpo.config import SearchDimConfig, SearchSpaceConfig
from olmo_core.hpo.controller import (
    ControllerConfig,
    HpoController,
    PopulationRestartMode,
)
from olmo_core.hpo.ifbo import IfBOCandidateGenerator
from olmo_core.hpo.ipbt import IPBTConfig, IPBTController
from olmo_core.hpo.objective import CENormalizer
from olmo_core.hpo.simulate import OracleFTPFN, RandomProposer, SyntheticObjective
from olmo_core.hpo.state import EventKind
from olmo_core.hpo.types import (
    ActionKind,
    Allocation,
    BTTVerdictKind,
    ProposalSource,
    TrialStatus,
    Verdict,
    WorkerObservation,
)


def build(
    seed=0,
    budget=1000,
    workers=4,
    rounds_target=8,
    quantum=2,
    btt=None,
    proposer=None,
    centaur=None,
    advisor=None,
    ipbt=None,
    ipbt_meta_proposer=None,
    cma_replacement_proposer=None,
    action_centaur=None,
    action_advisor=None,
    restart_mode=PopulationRestartMode.IPBT_REFERENCE,
    btt_restart_fraction=0.5,
    ensure_full_fidelity=False,
):
    space = SearchSpaceConfig(
        dims=[SearchDimConfig("x", 0.0, 1.0), SearchDimConfig("y", 0.0, 1.0)]
    ).build()
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    obj = SyntheticObjective(optimum=(0.5, 0.5))
    post = OracleFTPFN(obj, norm)
    proposer = proposer or RandomProposer(ndim=2, seed=seed)
    btt = btt or BTTDiagnoser(BTTConfig(min_fidelity=1, nmg_min_rel_improve=1e-9))
    cfg = ControllerConfig(
        target_tokens=rounds_target,
        quantum=quantum,
        n_fidelity_bins=4,
        worker_count=workers,
        budget_tokens=budget,
        checkpoint_root="/tmp/run",
        seed=seed,
        restart_mode=restart_mode,
        btt_restart_fraction=btt_restart_fraction,
        ensure_full_fidelity=ensure_full_fidelity,
    )
    return (
        HpoController(
            space,
            norm,
            post,
            proposer,
            btt,
            cfg,
            centaur=centaur,
            advisor=advisor,
            ipbt=ipbt,
            ipbt_meta_proposer=ipbt_meta_proposer,
            cma_replacement_proposer=cma_replacement_proposer,
            action_centaur=action_centaur,
            action_advisor=action_advisor,
        ),
        obj,
    )


def _alloc_events(controller):
    return [
        Allocation.from_dict(e.payload) for e in controller.log if e.kind is EventKind.ALLOCATION
    ]


def _worker_observation(allocation, ce):
    return WorkerObservation(
        trial_id=allocation.trial_id,
        tokens=allocation.target_fidelity,
        heldout_ce=ce,
        train_ce_history=(ce,),
        grad_norm_history=(1.0,),
        activation_ratio=0.5,
        numeric_failure=not np.isfinite(ce),
        checkpoint_ref=f"/ckpt/{allocation.trial_id}/step1",
    )


def test_decision_ids_are_contiguous_single_owner_sequence():
    c, obj = build()
    c.run(rounds=3, simulate=obj.ce)
    ids = [a.decision_id for a in _alloc_events(c)]
    assert ids == list(range(len(ids)))
    assert len(ids) > 0


def test_every_allocation_charges_exactly_one_quantum_and_respects_budget():
    c, obj = build(budget=10)
    c.run(rounds=20, simulate=obj.ce)
    state = c.state()
    allocs = _alloc_events(c)
    # Every non-terminal grant charges one quantum; the terminal grant may be smaller.
    for a in allocs:
        assert 0 < a.target_fidelity - a.current_fidelity <= 2
    assert state.tokens_charged <= 10  # never exceeds the cap


def test_budget_reserves_a_path_to_one_full_fidelity_result():
    c, obj = build(budget=10, workers=4, rounds_target=8, quantum=2)
    c.run(rounds=20, simulate=obj.ce)
    state = c.state()
    assert state.tokens_charged == 10
    assert any(record.current_fidelity == 8 for record in state.trials.values())


def test_full_fidelity_reserve_continues_a_saturated_incumbent():
    controller, _ = build(
        budget=4,
        workers=1,
        rounds_target=4,
        quantum=2,
        ensure_full_fidelity=True,
    )
    first = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=first.trial_id,
                tokens=first.target_fidelity,
                heldout_ce=3.0,
                train_ce_history=(3.0, 3.01),
                grad_norm_history=(1.0, 1.0),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/saturated/step1",
            )
        ]
    )
    assert controller.state().trials[first.trial_id].latest_verdict.kind is BTTVerdictKind.SATURATED

    rescue = controller.propose_round()
    assert len(rescue) == 1
    assert rescue[0].kind is ActionKind.RESUME
    assert rescue[0].transition == {"full_fidelity_rescue": True}
    assert rescue[0].target_fidelity == 4
    assert controller.state().tokens_charged == 4


def test_budget_must_fund_one_full_fidelity_result():
    with pytest.raises(ValueError, match="full-fidelity"):
        build(budget=7, rounds_target=8)


def test_resume_only_targets_currently_eligible_trials():
    c, obj = build(seed=5)
    c.run(rounds=6, simulate=obj.ce)
    # Replay the log tracking each trial's latest verdict; a RESUME may only follow HEALTHY.
    latest = {}
    for e in c.log:
        if e.kind is EventKind.VERDICT:
            latest[e.payload["trial_id"]] = BTTVerdictKind(e.payload["kind"])
        elif e.kind is EventKind.ALLOCATION:
            a = Allocation.from_dict(e.payload)
            if a.kind is ActionKind.RESUME:
                assert latest.get(a.trial_id) is BTTVerdictKind.HEALTHY


def test_run_is_deterministic_for_a_seed():
    a, obj = build(seed=1)
    a.run(rounds=4, simulate=obj.ce)
    b, obj2 = build(seed=1)
    b.run(rounds=4, simulate=obj2.ce)
    assert a.log_snapshot() == b.log_snapshot()


def test_state_is_invariant_to_worker_completion_order():
    c1, obj = build(seed=2)
    allocs = c1.propose_round()
    results = [_worker_observation(a, obj.ce(a.unit_config, a.target_fidelity / 8)) for a in allocs]
    c1.ingest(results)

    c2, obj2 = build(seed=2)
    c2.propose_round()  # identical proposals for the same seed
    c2.ingest(list(reversed(results)))  # results arrive in the opposite order

    assert c1.log_snapshot() == c2.log_snapshot()


def test_controller_optimizes_toward_the_hidden_optimum():
    c, obj = build(seed=3, budget=400, workers=4)
    c.run(rounds=10, simulate=obj.ce)
    trial_id, unit_config, best_ce = c.best()
    # A working controller drives at least one promising config to low CE.
    assert best_ce < 4.5
    # And the best config it found is nearer the optimum than the unit-cube corner baseline.
    dist = np.linalg.norm(np.array(unit_config) - np.array([0.5, 0.5]))
    assert dist < np.linalg.norm(np.array([0.0, 0.0]) - np.array([0.5, 0.5]))


def test_propose_round_respects_existing_pending_capacity():
    controller, objective = build(workers=1)
    first = controller.propose_round()[0]
    assert first.kind is ActionKind.START
    assert controller.propose_round() == []
    controller.ingest(
        [
            _worker_observation(
                first,
                objective.ce(first.unit_config, first.target_fidelity / 8),
            )
        ]
    )
    assert len(controller.propose_round()) == 1


def test_terminal_allocation_respects_budget_and_target_cap():
    controller, _ = build(workers=1, rounds_target=2, quantum=3, budget=2)
    allocation = controller.propose_round()[0]
    assert allocation.current_fidelity == 0
    assert allocation.target_fidelity == 2
    assert controller.state().tokens_charged == 2


def test_ingest_forwards_worker_diagnostics_to_btt():
    class RecordingDiagnoser:
        def __init__(self):
            self.observation = None

        def diagnose_cohort(self, observations, *, scores):
            self.observation = observations[0]
            verdict = Verdict(
                kind=BTTVerdictKind.HEALTHY,
                indicators=(),
                trial_id=self.observation.trial_id,
                completed_fidelity=self.observation.completed_fidelity,
                observation_hash=self.observation.observation_hash,
                profile_version="test",
            )
            return {self.observation.trial_id: verdict}

    diagnoser = RecordingDiagnoser()
    controller, _ = build(workers=1, btt=diagnoser)
    allocation = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=allocation.target_fidelity,
                heldout_ce=3.5,
                train_ce_history=(4.0, 3.7),
                grad_norm_history=(1.0, 8.0),
                activation_ratio=0.05,
                numeric_failure=False,
                checkpoint_ref="/ckpt/step1",
            )
        ]
    )
    assert diagnoser.observation.loss_history == (4.0, 3.7)
    assert diagnoser.observation.grad_norm_history == (1.0, 8.0)
    assert diagnoser.observation.activation_ratio == 0.05
    assert controller.state().trials[allocation.trial_id].latest_checkpoint_ref == "/ckpt/step1"


def test_fatal_observation_is_censored_from_posterior():
    controller, _ = build(workers=1)
    allocation = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=allocation.target_fidelity,
                heldout_ce=float("nan"),
                train_ce_history=(float("nan"),),
                grad_norm_history=(float("nan"),),
                activation_ratio=None,
                numeric_failure=True,
                checkpoint_ref=None,
            )
        ]
    )
    assert controller.state().trials[allocation.trial_id].status is TrialStatus.RETIRED
    next_allocation = controller.propose_round()[0]
    assert next_allocation.kind is ActionKind.START
    assert next_allocation.trial_id != allocation.trial_id


def test_finite_fatal_observation_is_censored_from_posterior():
    class FatalDiagnoser(BTTDiagnoser):
        def diagnose_cohort(self, observations, *, scores):
            return {
                observation.trial_id: Verdict(
                    kind=BTTVerdictKind.FATAL,
                    indicators=("AGV",),
                    trial_id=observation.trial_id,
                    completed_fidelity=observation.completed_fidelity,
                    observation_hash=observation.observation_hash,
                    profile_version="test",
                )
                for observation in observations
            }

    controller, _ = build(workers=1, btt=FatalDiagnoser(BTTConfig(min_fidelity=1)))
    allocation = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=allocation.target_fidelity,
                heldout_ce=3.5,
                train_ce_history=(4.0, 3.5),
                grad_norm_history=(1.0, 100.0),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/fatal",
            )
        ]
    )
    assert controller._observed(controller.state()) == []


def test_fatal_curve_is_not_ifbo_incumbent():
    class RecordingIfBO(IfBOCandidateGenerator):
        def __init__(self):
            super().__init__(ndim=2, seed=3)
            self.incumbents = []

        def ask(self, n, *, incumbent=None):
            self.incumbents.append(incumbent)
            return super().ask(n, incumbent=incumbent)

    class FatalDiagnoser(BTTDiagnoser):
        def diagnose_cohort(self, observations, *, scores):
            return {
                observation.trial_id: Verdict(
                    kind=BTTVerdictKind.FATAL,
                    indicators=("AGV",),
                    trial_id=observation.trial_id,
                    completed_fidelity=observation.completed_fidelity,
                    observation_hash=observation.observation_hash,
                    profile_version="test",
                )
                for observation in observations
            }

    proposer = RecordingIfBO()
    controller, _ = build(
        workers=1,
        proposer=proposer,
        btt=FatalDiagnoser(BTTConfig(min_fidelity=1)),
    )
    allocation = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=allocation.target_fidelity,
                heldout_ce=3.5,
                train_ce_history=(4.0, 3.5),
                grad_norm_history=(1.0, 100.0),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/fatal",
            )
        ]
    )
    controller.propose_round()
    assert proposer.incumbents == [None, None]


def test_best_requires_comparable_final_fidelity():
    controller, _ = build(workers=1, rounds_target=4, quantum=2)
    allocation = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=2,
                heldout_ce=2.0,
                train_ce_history=(2.0,),
                grad_norm_history=(1.0,),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/step1",
            )
        ]
    )
    with pytest.raises(RuntimeError):
        controller.best()


def test_restore_reconstructs_rng_proposer_curve_and_trial_state():
    original, objective = build(seed=11, workers=2)
    allocations = original.propose_round()
    original.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(
                    allocation.unit_config,
                    allocation.target_fidelity / original.config.target_tokens,
                ),
            )
            for allocation in allocations
        ]
    )
    serialized = original.log.to_jsonl()

    restored, _ = build(seed=11, workers=2)
    restored.restore_log(serialized)
    expected = [allocation.to_dict() for allocation in original.propose_round()]
    actual = [allocation.to_dict() for allocation in restored.propose_round()]
    assert actual == expected
    assert len({allocation["trial_id"] for allocation in actual}) == len(actual)


def test_completion_permutation_preserves_exact_event_log():
    first, objective = build(seed=13, workers=2)
    first_allocations = first.propose_round()
    first_results = [
        _worker_observation(
            allocation,
            objective.ce(
                allocation.unit_config,
                allocation.target_fidelity / first.config.target_tokens,
            ),
        )
        for allocation in first_allocations
    ]
    first.ingest([first_results[0]])
    assert all(record.curve == [] for record in first.state().trials.values())
    first.ingest([first_results[1]])

    second, _ = build(seed=13, workers=2)
    second.propose_round()
    second.ingest([first_results[1]])
    second.ingest([first_results[0]])
    assert second.log.to_jsonl() == first.log.to_jsonl()


def test_cross_batch_completion_permutation_preserves_event_log():
    class TwoAtATimeProposer:
        def __init__(self):
            self.offset = 0

        def ask(self, n):
            configs = [
                (0.1 + self.offset * 0.1, 0.2),
                (0.3 + self.offset * 0.1, 0.4),
            ]
            self.offset += 1
            return configs

    def make():
        return build(workers=4, proposer=TwoAtATimeProposer(), seed=17)

    first, objective = make()
    first_batch = first.propose_round()
    second_batch = first.propose_round()
    first_results = [
        _worker_observation(
            allocation,
            objective.ce(
                allocation.unit_config,
                allocation.target_fidelity / first.config.target_tokens,
            ),
        )
        for allocation in first_batch
    ]
    second_results = [
        _worker_observation(
            allocation,
            objective.ce(
                allocation.unit_config,
                allocation.target_fidelity / first.config.target_tokens,
            ),
        )
        for allocation in second_batch
    ]
    first.ingest(second_results)
    first.ingest(first_results)

    second, _ = make()
    second.propose_round()
    second.propose_round()
    second.ingest(first_results)
    second.ingest(second_results)
    assert first.log.to_jsonl() == second.log.to_jsonl()


def test_cma_generation_is_asked_once_and_told_at_final_fidelity():
    class RecordingCMA:
        population_size = 2

        def __init__(self):
            self.ask_count = 0
            self.tells = []

        def ask(self, n):
            self.ask_count += 1
            assert n == self.population_size
            return [(0.1, 0.2), (0.8, 0.9)]

        def tell(self, solutions):
            self.tells.append(solutions)

        def state(self):
            return {
                "mean": [0.5, 0.5],
                "sigma": 0.2,
                "covariance": [[1.0, 0.0], [0.0, 1.0]],
                "generation": 0,
                "population_size": 2,
            }

    proposer = RecordingCMA()
    controller, objective = build(
        workers=2,
        rounds_target=2,
        quantum=2,
        proposer=proposer,
    )
    allocations = controller.propose_round()
    assert proposer.ask_count == 1
    assert {allocation.source for allocation in allocations} == {ProposalSource.CMA}
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 1.0),
            )
            for allocation in allocations
        ]
    )
    assert proposer.ask_count == 1
    assert len(proposer.tells) == 1
    assert [tuple(config) for config, _ in proposer.tells[0]] == [
        allocation.unit_config for allocation in allocations
    ]


def test_failed_controller_tell_retries_without_reingesting_observations():
    class FlakyCMA:
        population_size = 1

        def __init__(self):
            self.tell_calls = 0

        def ask(self, n):
            return [(0.1, 0.2)]

        def tell(self, solutions):
            self.tell_calls += 1
            if self.tell_calls == 1:
                raise RuntimeError("transient tell failure")

        def state(self):
            return {}

    proposer = FlakyCMA()
    controller, objective = build(
        workers=1,
        rounds_target=2,
        quantum=2,
        proposer=proposer,
    )
    allocation = controller.propose_round()[0]
    with pytest.raises(RuntimeError, match="transient tell failure"):
        controller.ingest(
            [
                _worker_observation(
                    allocation,
                    objective.ce(allocation.unit_config, 1.0),
                )
            ]
        )
    assert controller.pending_allocations() == []
    controller.retry_pending_tell()
    assert proposer.tell_calls == 2
    assert {ask.status.value for ask in controller._ask_ledger._asks.values()} == {"told"}


def test_centaur_ratio_one_evaluates_and_records_llm_overrides():
    class RecordingCMA:
        population_size = 2

        def ask(self, n):
            return [(0.1, 0.2), (0.8, 0.9)]

        def tell(self, solutions):
            pass

        def state(self):
            return {
                "mean": [0.5, 0.5],
                "sigma": 0.2,
                "covariance": [[1.0, 0.0], [0.0, 1.0]],
                "generation": 0,
                "population_size": 2,
            }

    class Advisor:
        def advise(self, state):
            proposal_id = state["proposal_id"]
            return AdvisorResponse(
                action={"kind": "start", "unit_config": [0.3 + proposal_id * 0.1, 0.4]},
                raw_text="{}",
                model="test-model",
                version="v1",
                latency_ms=1.0,
            )

    controller, _ = build(
        workers=2,
        rounds_target=2,
        quantum=2,
        proposer=RecordingCMA(),
        centaur=CentaurOverlay(warmup=0, ratio=1.0),
        advisor=Advisor(),
    )
    allocations = controller.propose_round()
    assert {allocation.unit_config for allocation in allocations} == {(0.3, 0.4), (0.4, 0.4)}
    assert {allocation.source for allocation in allocations} == {ProposalSource.LLM}
    assert sum(event.kind is EventKind.ADVISOR for event in controller.log) == 2
    asks = sorted(controller._ask_ledger._asks.values(), key=lambda ask: ask.ask_id)
    assert [ask.unit_config for ask in asks[:2]] == [(0.1, 0.2), (0.8, 0.9)]
    assert all(ask.status.value == "replaced" for ask in asks[:2])
    assert {ask.replaces_ask_id for ask in asks[2:]} == {0, 1}


def test_failed_centaur_turn_is_event_sourced_before_rethrow():
    class RecordingCMA:
        population_size = 1

        def ask(self, n):
            return [(0.1, 0.2)]

        def tell(self, solutions):
            pass

        def state(self):
            return {"mean": [0.5, 0.5]}

    class InvalidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action={"kind": "resume", "trial_id": "missing"},
                raw_text='{"kind":"resume"}',
                model="test",
                version="v1",
                latency_ms=1.0,
            )

    controller, _ = build(
        workers=1,
        proposer=RecordingCMA(),
        centaur=CentaurOverlay(warmup=0, ratio=1.0),
        advisor=InvalidAdvisor(),
    )
    with pytest.raises(Exception):
        controller.propose_round()
    assert any(event.kind is EventKind.ADVISOR for event in controller.log)
    assert controller.log.events[-1].kind is EventKind.CONTROLLER_SNAPSHOT

    class ValidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action={"kind": "start", "unit_config": [0.3, 0.4]},
                raw_text="{}",
                model="test",
                version="v1",
                latency_ms=1.0,
            )

    controller.advisor = ValidAdvisor()
    retry = controller.propose_round()
    assert len(retry) == 1
    assert retry[0].unit_config == (0.3, 0.4)


def test_failed_multi_action_sol_turn_is_logged_and_retryable():
    class InvalidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action={"kind": "start", "unit_config": [0.5]},
                raw_text="{}",
                model="gpt-5.6-sol",
                version="v1",
                latency_ms=1.0,
            )

    controller, _ = build(
        workers=1,
        proposer=IfBOCandidateGenerator(ndim=2, seed=7),
        action_centaur=CentaurOverlay(warmup=0, ratio=1.0),
        action_advisor=InvalidAdvisor(),
    )
    with pytest.raises(Exception):
        controller.propose_round()
    failure = [
        event
        for event in controller.log
        if event.kind is EventKind.ADVISOR and event.payload.get("scope") == "multi_action"
    ]
    assert failure and "error" in failure[-1].payload
    assert controller._action_proposal_counter == 0

    class ValidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action=state["default_action"],
                raw_text="{}",
                model="gpt-5.6-sol",
                version="v1",
                latency_ms=1.0,
            )

    controller.action_advisor = ValidAdvisor()
    assert len(controller.propose_round()) == 1


def test_partial_batch_failure_keeps_earlier_grants_recoverable():
    class FailScheduledTurn:
        def advise(self, state):
            raise RuntimeError("scheduled turn failed")

    controller, objective = build(
        workers=2,
        proposer=IfBOCandidateGenerator(ndim=2, seed=7),
        action_centaur=CentaurOverlay(warmup=0, ratio=0.5),
        action_advisor=FailScheduledTurn(),
    )
    # Reproduce a real recovery log where a valid snapshot predates the
    # allocations granted before a later slot fails.
    controller._snapshot_controller()
    with pytest.raises(Exception):
        controller.propose_round()
    pending = controller.pending_allocations()
    assert len(pending) == 1
    assert pending[0].trial_id in next(iter(controller._pending_batches.values()))

    restored, _ = build(
        workers=2,
        proposer=IfBOCandidateGenerator(ndim=2, seed=7),
        action_centaur=CentaurOverlay(warmup=0, ratio=0.5),
        action_advisor=FailScheduledTurn(),
    )
    restored.restore_log(controller.log.to_jsonl())
    restored.ingest(
        [
            _worker_observation(
                pending[0],
                objective.ce(
                    pending[0].unit_config,
                    pending[0].target_fidelity / restored.config.target_tokens,
                ),
            )
        ]
    )
    assert restored.pending_allocations() == []


def test_controller_supplies_complete_centaur_advisor_state():
    captured = {}

    class RecordingCMA:
        population_size = 1

        def ask(self, n):
            return [(0.1, 0.2)]

        def tell(self, solutions):
            pass

        def state(self):
            return {
                "mean": [0.5, 0.5],
                "sigma": 0.2,
                "covariance": [[1.0, 0.0], [0.0, 1.0]],
                "generation": 0,
                "population_size": 1,
            }

    class Advisor:
        def advise(self, state):
            if not captured:
                captured.update(state)
            return AdvisorResponse(
                action={"kind": "start", "unit_config": [0.3, 0.4]},
                raw_text="{}",
                model="test",
                version="v1",
                latency_ms=1.0,
            )

    controller, _ = build(
        workers=2,
        proposer=RecordingCMA(),
        centaur=CentaurOverlay(warmup=0, ratio=1.0),
        advisor=Advisor(),
    )
    controller.propose_round()
    required = {
        "cma_mean",
        "cma_sigma",
        "cma_cov",
        "cma_proposal",
        "ifbo_action",
        "ifbo_alternatives",
        "population_lineages",
        "btt_evidence",
        "incumbent",
        "top_five",
        "recent_decisions",
        "bounds",
        "remaining_budget",
        "action_schema",
        "proposal_id",
    }
    assert required <= set(captured)


def test_multi_action_advisor_can_override_allocation_to_resume():
    controller, objective = build(workers=1)
    first = controller.propose_round()[0]
    controller.ingest(
        [
            _worker_observation(
                first,
                objective.ce(first.unit_config, first.target_fidelity / 8),
            )
        ]
    )

    captured = {}

    class ResumeAdvisor:
        def advise(self, state):
            captured.update(state)
            return AdvisorResponse(
                action={"kind": "resume", "trial_id": first.trial_id},
                raw_text='{"kind":"resume"}',
                model="test",
                version="v1",
                latency_ms=1.0,
            )

    controller.action_centaur = CentaurOverlay(warmup=0, ratio=1.0)
    controller.action_advisor = ResumeAdvisor()
    allocation = controller.propose_round()[0]
    assert allocation.kind is ActionKind.RESUME
    assert allocation.trial_id == first.trial_id
    assert any(
        event.kind is EventKind.ADVISOR and event.payload.get("scope") == "multi_action"
        for event in controller.log
    )
    assert captured["population_lineages"]
    assert captured["btt_evidence"]


def test_sol_receives_actual_ifbo_action_and_ranked_alternatives():
    captured = {}

    class Advisor:
        def advise(self, state):
            if not captured:
                captured.update(state)
            return AdvisorResponse(
                action=state["default_action"],
                raw_text="{}",
                model="gpt-5.6-sol",
                version="v1",
                latency_ms=1.0,
            )

    controller, _ = build(
        workers=2,
        proposer=IfBOCandidateGenerator(ndim=2, seed=7),
        action_centaur=CentaurOverlay(warmup=0, ratio=1.0),
        action_advisor=Advisor(),
    )
    controller.propose_round()
    assert captured["ifbo_action"]["candidate_key"]
    assert captured["ifbo_action"]["mfpi_score"] >= 0.0
    assert captured["ifbo_action"]["horizon"] >= 1
    assert captured["ifbo_action"]["threshold"] >= 0.0
    assert captured["ifbo_alternatives"]


def test_controller_emits_ipbt_transition_at_population_boundary():
    ipbt = IPBTController(
        IPBTConfig(
            population_size=4,
            top_quantile=0.25,
            bottom_quantile=0.25,
            update_interval_init=2,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
        )
    )
    controller, objective = build(
        workers=4,
        rounds_target=8,
        quantum=2,
        ipbt=ipbt,
        ipbt_meta_proposer=RandomProposer(ndim=2, seed=99),
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(
                    allocation.unit_config,
                    allocation.target_fidelity / controller.config.target_tokens,
                ),
            )
            for allocation in allocations
        ]
    )
    transitions = [event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION]
    assert len(transitions) == 1
    assert transitions[0].payload["completed_fidelity"] == 2
    assert transitions[0].payload["descendants"]
    next_allocations = controller.propose_round()
    assert any(allocation.source is ProposalSource.IPBT_META for allocation in next_allocations)


def test_ipbt_meta_bo_configs_are_selected_by_ftpfn_ifbo(monkeypatch):
    controller, objective = build(
        workers=4,
        ipbt=IPBTController(
            IPBTConfig(
                population_size=4,
                top_quantile=0.25,
                bottom_quantile=0.25,
                update_interval_init=2,
            )
        ),
        ipbt_meta_proposer=IfBOCandidateGenerator(ndim=2, seed=99),
    )
    allocations = controller.propose_round()
    calls = []
    original = controller.mfpi.select_batch

    def recording_select(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(controller.mfpi, "select_batch", recording_select)
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 0.25),
            )
            for allocation in allocations
        ]
    )
    assert calls
    assert max(calls) >= controller.ipbt.config.population_size * 2


def test_ipbt_transition_removes_replaced_slot_from_resume_candidates():
    ipbt = IPBTController(
        IPBTConfig(
            population_size=4,
            top_quantile=0.25,
            bottom_quantile=0.25,
            update_interval_init=2,
        )
    )
    controller, objective = build(
        workers=4,
        rounds_target=8,
        quantum=2,
        ipbt=ipbt,
        ipbt_meta_proposer=RandomProposer(ndim=2, seed=99),
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(allocation, objective.ce(allocation.unit_config, 0.25))
            for allocation in allocations
        ]
    )
    transition = next(event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION)
    replaced_slots = {descendant["slot_id"] for descendant in transition.payload["descendants"]}
    eligible = {
        candidate.key for candidate in controller._eligible_continuations(controller.state())
    }
    assert replaced_slots.isdisjoint(eligible)


def test_controller_applies_declared_ipbt_initial_oversampling():
    ipbt = IPBTController(
        IPBTConfig(
            population_size=4,
            initial_oversample=8,
            top_quantile=0.25,
            bottom_quantile=0.25,
            update_interval_init=2,
        )
    )
    controller, objective = build(
        workers=4,
        rounds_target=8,
        quantum=2,
        ipbt=ipbt,
        ipbt_meta_proposer=RandomProposer(ndim=2, seed=99),
    )
    first = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(allocation, objective.ce(allocation.unit_config, 0.25))
            for allocation in first
        ]
    )
    second = controller.propose_round()
    assert all(allocation.kind is ActionKind.START for allocation in second)
    controller.ingest(
        [
            _worker_observation(allocation, objective.ce(allocation.unit_config, 0.25))
            for allocation in second
        ]
    )
    transition = next(event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION)
    assert len(transition.payload["kept"]) + len(transition.payload["descendants"]) == 4
    selected = set(transition.payload["kept"]) | {
        descendant["slot_id"] for descendant in transition.payload["descendants"]
    }
    rejected = {allocation.trial_id for allocation in first + second} - selected
    state = controller.state()
    assert rejected
    assert all(state.trials[trial_id].status is TrialStatus.RETIRED for trial_id in rejected)


def test_oversampling_waits_for_ipbt_update_interval():
    controller, objective = build(
        workers=4,
        rounds_target=12,
        quantum=2,
        ipbt=IPBTController(
            IPBTConfig(
                population_size=4,
                initial_oversample=8,
                top_quantile=0.25,
                bottom_quantile=0.25,
                update_interval_init=4,
            )
        ),
        ipbt_meta_proposer=IfBOCandidateGenerator(ndim=2, seed=99),
    )
    for _ in range(4):
        allocations = controller.propose_round()
        controller.ingest(
            [
                _worker_observation(
                    allocation,
                    objective.ce(
                        allocation.unit_config,
                        allocation.target_fidelity / controller.config.target_tokens,
                    ),
                )
                for allocation in allocations
            ]
        )
    assert any(event.kind is EventKind.IPBT_TRANSITION for event in controller.log)


def test_controller_applies_same_fidelity_btt_protection_as_a_cohort():
    diagnoser = BTTDiagnoser(
        BTTConfig(
            min_fidelity=1,
            same_fidelity_top_fraction=0.5,
            nmg_min_rel_improve=1e-9,
        )
    )
    controller, _ = build(workers=2, btt=diagnoser)
    allocations = controller.propose_round()
    results = [
        WorkerObservation(
            trial_id=allocation.trial_id,
            tokens=allocation.target_fidelity,
            heldout_ce=ce,
            train_ce_history=(4.0, 4.0, 4.0, 4.0),
            grad_norm_history=(1.0, 1.0, 1.0, 1.0),
            activation_ratio=0.5,
            numeric_failure=False,
            checkpoint_ref=f"/ckpt/{allocation.trial_id}",
        )
        for allocation, ce in zip(allocations, (3.0, 5.0))
    ]
    controller.ingest(results)
    state = controller.state()
    best_trial = results[0].trial_id
    worst_trial = results[1].trial_id
    assert state.trials[best_trial].latest_verdict.protected_by_peer_rank is True
    assert state.trials[best_trial].latest_verdict.kind is BTTVerdictKind.HEALTHY
    assert state.trials[worst_trial].latest_verdict.kind is BTTVerdictKind.DEGRADED


@pytest.mark.parametrize("first_ce,second_ce", [(3.0, 5.0), (5.0, 3.0)])
def test_same_fidelity_protection_spans_worker_batches(first_ce, second_ce):
    diagnoser = BTTDiagnoser(
        BTTConfig(
            min_fidelity=1,
            same_fidelity_top_fraction=0.5,
            nmg_min_rel_improve=1e-9,
        )
    )
    controller, _ = build(workers=1, btt=diagnoser)
    first = controller.propose_round()[0]
    controller.ingest(
        [
            WorkerObservation(
                trial_id=first.trial_id,
                tokens=first.target_fidelity,
                heldout_ce=first_ce,
                train_ce_history=(4.0, 4.0),
                grad_norm_history=(1.0, 1.0),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/first",
            )
        ]
    )
    controller._append(
        EventKind.STATUS,
        {"trial_id": first.trial_id, "status": TrialStatus.RETIRED.value},
    )
    second = controller.propose_round()[0]
    assert second.trial_id != first.trial_id
    controller.ingest(
        [
            WorkerObservation(
                trial_id=second.trial_id,
                tokens=second.target_fidelity,
                heldout_ce=second_ce,
                train_ce_history=(4.0, 4.0),
                grad_norm_history=(1.0, 1.0),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/second",
            )
        ]
    )
    state = controller.state()
    best, worst = (first, second) if first_ce < second_ce else (second, first)
    assert state.trials[best.trial_id].latest_verdict.protected_by_peer_rank is True
    assert state.trials[worst.trial_id].latest_verdict.kind is BTTVerdictKind.DEGRADED


def test_restore_preserves_ipbt_and_meta_proposer_state():
    first_ipbt = IPBTController(IPBTConfig(population_size=4))
    first_meta = RandomProposer(ndim=2, seed=99)
    first_ipbt._used_lineage_ids.add("Lnew0")
    first_ipbt._new_lineage_id()
    first_meta.ask(1)
    first, _ = build(
        workers=4,
        ipbt=first_ipbt,
        ipbt_meta_proposer=first_meta,
    )
    first.propose_round()

    second_ipbt = IPBTController(IPBTConfig(population_size=4))
    second_meta = RandomProposer(ndim=2, seed=99)
    restored, _ = build(
        workers=4,
        ipbt=second_ipbt,
        ipbt_meta_proposer=second_meta,
    )
    restored.restore_log(first.log.to_jsonl())
    assert restored.ipbt._new_lineage_id() == first.ipbt._new_lineage_id()
    assert restored.ipbt_meta_proposer.ask(1) == first.ipbt_meta_proposer.ask(1)


def test_btt_censored_cma_trial_creates_same_generation_replacement():
    class RecordingCMA:
        population_size = 2

        def ask(self, n):
            return [(0.1, 0.2), (0.8, 0.9)]

        def tell(self, solutions):
            pass

        def state(self):
            return {}

    class OneCutDiagnoser(BTTDiagnoser):
        def diagnose_cohort(self, observations, *, scores):
            verdicts = super().diagnose_cohort(observations, scores=scores)
            first = sorted(verdicts)[0]
            observation = next(item for item in observations if item.trial_id == first)
            verdicts[first] = Verdict(
                kind=BTTVerdictKind.DEGRADED,
                indicators=("PLC",),
                trial_id=first,
                completed_fidelity=observation.completed_fidelity,
                observation_hash=observation.observation_hash,
                profile_version="test",
            )
            return verdicts

    controller, objective = build(
        workers=2,
        rounds_target=4,
        quantum=2,
        proposer=RecordingCMA(),
        btt=OneCutDiagnoser(BTTConfig(min_fidelity=1)),
        cma_replacement_proposer=RandomProposer(ndim=2, seed=123),
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 0.5),
            )
            for allocation in allocations
        ]
    )
    assert controller._new_candidate_pool
    statuses = {ask.status.value for ask in controller._ask_ledger._asks.values()}
    assert "replaced" in statuses
    assert "pending" in statuses


def test_ipbt_restart_tracker_switches_boundary_to_restart(monkeypatch):
    ipbt = IPBTController(
        IPBTConfig(
            population_size=2,
            top_quantile=0.5,
            bottom_quantile=0.5,
            update_interval_init=2,
        )
    )
    called = {"restart": False}
    original_restart = ipbt.restart_population

    def restart(*args, **kwargs):
        called["restart"] = True
        return original_restart(*args, **kwargs)

    monkeypatch.setattr(ipbt.restart_tracker, "update", lambda best_score: True)
    monkeypatch.setattr(ipbt, "restart_population", restart)
    controller, objective = build(
        workers=2,
        quantum=2,
        rounds_target=8,
        ipbt=ipbt,
        ipbt_meta_proposer=RandomProposer(ndim=2, seed=77),
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 0.25),
            )
            for allocation in allocations
        ]
    )
    assert called["restart"] is True
    transition = next(event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION)
    assert transition.payload["transition_kind"] == "restart"


def test_brainlift_btt_aggregate_replaces_ipbt_restart_tracker(monkeypatch):
    class HalfBadDiagnoser(BTTDiagnoser):
        def diagnose_cohort(self, observations, *, scores):
            verdicts = super().diagnose_cohort(observations, scores=scores)
            for observation in sorted(observations, key=lambda item: item.trial_id)[:2]:
                verdicts[observation.trial_id] = Verdict(
                    kind=BTTVerdictKind.DEGRADED,
                    indicators=("PLC",),
                    trial_id=observation.trial_id,
                    completed_fidelity=observation.completed_fidelity,
                    observation_hash=observation.observation_hash,
                    profile_version="test",
                )
            return verdicts

    ipbt = IPBTController(
        IPBTConfig(
            population_size=4,
            top_quantile=0.25,
            bottom_quantile=0.25,
            update_interval_init=2,
        )
    )
    monkeypatch.setattr(
        ipbt.restart_tracker,
        "update",
        lambda score: (_ for _ in ()).throw(
            AssertionError("reference restart tracker must not run")
        ),
    )
    controller, objective = build(
        workers=4,
        btt=HalfBadDiagnoser(BTTConfig(min_fidelity=1)),
        ipbt=ipbt,
        ipbt_meta_proposer=IfBOCandidateGenerator(ndim=2, seed=99),
        restart_mode=PopulationRestartMode.BTT_AGGREGATE,
        btt_restart_fraction=0.5,
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 0.25),
            )
            for allocation in allocations
        ]
    )
    transition = next(event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION)
    assert transition.payload["transition_kind"] == "restart"
    assert transition.payload["restart_evidence"]["bad_fraction"] == 0.5


def test_brainlift_all_fatal_population_restarts_entirely_fresh():
    class AllFatalDiagnoser(BTTDiagnoser):
        def diagnose_cohort(self, observations, *, scores):
            return {
                observation.trial_id: Verdict(
                    kind=BTTVerdictKind.FATAL,
                    indicators=("AGV",),
                    trial_id=observation.trial_id,
                    completed_fidelity=observation.completed_fidelity,
                    observation_hash=observation.observation_hash,
                    profile_version="test",
                )
                for observation in observations
            }

    controller, objective = build(
        workers=4,
        btt=AllFatalDiagnoser(BTTConfig(min_fidelity=1)),
        ipbt=IPBTController(
            IPBTConfig(
                population_size=4,
                top_quantile=0.25,
                bottom_quantile=0.25,
                update_interval_init=2,
            )
        ),
        ipbt_meta_proposer=IfBOCandidateGenerator(ndim=2, seed=99),
        restart_mode=PopulationRestartMode.BTT_AGGREGATE,
        btt_restart_fraction=0.5,
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(
                allocation,
                objective.ce(allocation.unit_config, 0.25),
            )
            for allocation in allocations
        ]
    )
    transition = next(event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION)
    assert len(transition.payload["descendants"]) == 4
    assert {descendant["weight_policy"] for descendant in transition.payload["descendants"]} == {
        "fresh_reset"
    }
    assert all(
        descendant["checkpoint_ref"] is None for descendant in transition.payload["descendants"]
    )


def test_initial_oversample_refills_after_all_trials_fail_before_transition():
    controller, _ = build(
        workers=4,
        rounds_target=4,
        quantum=2,
        btt=BTTDiagnoser(BTTConfig(min_fidelity=2)),
        ipbt=IPBTController(
            IPBTConfig(
                population_size=4,
                initial_oversample=4,
                top_quantile=0.25,
                bottom_quantile=0.25,
                update_interval_init=4,
            )
        ),
        ipbt_meta_proposer=IfBOCandidateGenerator(ndim=2, seed=99),
        restart_mode=PopulationRestartMode.BTT_AGGREGATE,
    )
    allocations = controller.propose_round()
    controller.ingest(
        [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=allocation.target_fidelity,
                heldout_ce=float("nan"),
                train_ce_history=(float("nan"),),
                grad_norm_history=(float("nan"),),
                activation_ratio=None,
                numeric_failure=True,
                checkpoint_ref=None,
            )
            for allocation in allocations
        ]
    )
    replacements = controller.propose_round()
    assert len(replacements) == 4
    assert all(allocation.kind is ActionKind.START for allocation in replacements)


def test_doubled_ipbt_interval_elapses_from_last_transition(monkeypatch):
    ipbt = IPBTController(
        IPBTConfig(
            population_size=2,
            top_quantile=0.5,
            bottom_quantile=0.5,
            update_interval_init=2,
        )
    )

    def fire_and_double(best_score):
        ipbt.restart_tracker.interval = 4
        return True

    monkeypatch.setattr(ipbt.restart_tracker, "update", fire_and_double)
    controller, objective = build(
        workers=2,
        quantum=2,
        rounds_target=8,
        ipbt=ipbt,
        ipbt_meta_proposer=RandomProposer(ndim=2, seed=77),
    )
    first = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(allocation, objective.ce(allocation.unit_config, 0.25))
            for allocation in first
        ]
    )
    assert len([event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION]) == 1
    second = controller.propose_round()
    controller.ingest(
        [
            _worker_observation(allocation, objective.ce(allocation.unit_config, 0.5))
            for allocation in second
        ]
    )
    assert len([event for event in controller.log if event.kind is EventKind.IPBT_TRANSITION]) == 1
