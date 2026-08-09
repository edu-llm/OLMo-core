import sys
import types

import numpy as np
import pytest

from olmo_core.hpo.centaur import (
    AdvisorResponse,
    AdvisorUnavailable,
    AskLedger,
    AskStatus,
    CentaurOverlay,
    LegalAction,
    build_advisor_state,
    consumes_cma_ask,
    should_llm_intervene,
    validate_action,
    validate_start_config,
)
from olmo_core.hpo.openai_advisor import OpenAICompatibleAdvisor
from olmo_core.hpo.types import ProposalSource


def test_ask_ledger_assigns_monotonic_ids_and_guarantees_vector_identity():
    ledger = AskLedger()
    asks = ledger.register([(0.1, 0.2), (0.3, 0.4)], stratum="from_scratch")
    assert [a.ask_id for a in asks] == [0, 1]
    # Resolving requires the evaluated vector to be byte-for-byte the asked vector.
    ledger.resolve(0, score=0.8, evaluated_config=(0.1, 0.2))
    with pytest.raises(ValueError):
        ledger.resolve(1, score=0.7, evaluated_config=(0.3, 0.9))  # tampered vector
    assert ledger.get(0).status is AskStatus.RESOLVED


def test_tell_uses_only_resolved_from_scratch_anchors():
    ledger = AskLedger()
    ledger.register([(0.1,), (0.2,)], stratum="from_scratch")
    ledger.register([(0.9,)], stratum="inherited")
    ledger.resolve(0, score=0.8, evaluated_config=(0.1,))
    ledger.fail(1, penalty=0.0)  # numeric/OOM failure -> preregistered penalty
    ledger.resolve(2, score=0.95, evaluated_config=(0.9,))  # inherited stratum
    tell = ledger.collect_tell(stratum="from_scratch", allow_inherited=False)
    assert [(tuple(x), s) for x, s in tell] == [((0.1,), 0.8), ((0.2,), 0.0)]
    assert ledger.collect_tell(stratum="from_scratch", allow_inherited=False) == []


def test_ask_ledger_transitions_are_terminal_and_censored_generation_blocks_tell():
    ledger = AskLedger()
    ledger.register([(0.1,), (0.2,)], stratum="from_scratch")
    ledger.censor(0)
    ledger.resolve(1, score=0.8, evaluated_config=(0.2,))
    with pytest.raises(ValueError):
        ledger.resolve(0, score=0.9, evaluated_config=(0.1,))
    with pytest.raises(ValueError):
        ledger.collect_tell(stratum="from_scratch", allow_inherited=False)


def test_censored_ask_can_be_replaced_without_becoming_a_bad_objective():
    ledger = AskLedger()
    ledger.register([(0.1,), (0.2,)], stratum="from_scratch")
    ledger.censor(0)
    replacement = ledger.replace_censored(0, (0.3,))
    ledger.resolve(replacement.ask_id, score=0.7, evaluated_config=(0.3,))
    ledger.resolve(1, score=0.8, evaluated_config=(0.2,))
    tell = ledger.collect_tell(stratum="from_scratch", allow_inherited=False)
    assert tell == [((0.2,), 0.8), ((0.3,), 0.7)]
    assert all(config != (0.1,) for config, _ in tell)


def test_prepared_tell_is_retryable_until_committed():
    ledger = AskLedger()
    ledger.register([(0.1,), (0.2,)], stratum="from_scratch")
    ledger.resolve(0, score=0.8, evaluated_config=(0.1,))
    ledger.resolve(1, score=0.7, evaluated_config=(0.2,))
    first = ledger.collect_tell(stratum="from_scratch", allow_inherited=False, consume=False)
    second = ledger.collect_tell(stratum="from_scratch", allow_inherited=False, consume=False)
    assert second == first
    ledger.mark_told(stratum="from_scratch", allow_inherited=False)
    assert ledger.collect_tell(stratum="from_scratch", allow_inherited=False) == []


def test_ask_ledger_preserves_vector_bytes_and_rejects_nonfinite():
    ledger = AskLedger()
    ledger.register([np.array([-0.0], dtype=np.float32)], stratum="from_scratch")
    with pytest.raises(ValueError):
        ledger.resolve(0, score=0.8, evaluated_config=np.array([0.0], dtype=np.float32))
    with pytest.raises(ValueError):
        ledger.register([np.array([float("nan")])], stratum="from_scratch")


def test_resume_does_not_consume_a_cma_ask():
    assert consumes_cma_ask(LegalAction.START) is True
    assert consumes_cma_ask(LegalAction.RESUME) is False
    assert consumes_cma_ask(LegalAction.IPBT_EXPLOIT) is False


def test_llm_intervention_schedule_is_deterministic_and_hits_ratio():
    # No interventions during warmup.
    assert not any(should_llm_intervene(i, warmup=10, ratio=0.3) for i in range(10))
    # Over 100 post-warmup decisions at ratio 0.3, exactly 30 are LLM turns.
    count = sum(should_llm_intervene(i, warmup=0, ratio=0.3) for i in range(100))
    assert count == 30
    # r = 0 means CMA-only.
    assert sum(should_llm_intervene(i, warmup=0, ratio=0.0) for i in range(100)) == 0
    assert sum(should_llm_intervene(i, warmup=0, ratio=0.58) for i in range(100)) == 58
    for ratio in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            should_llm_intervene(0, warmup=0, ratio=ratio)


def test_openai_compatible_advisor_uses_structured_multi_action_output():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                model="gpt-5.6-sol",
                system_fingerprint="sol-v1",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='{"kind":"resume","trial_id":"trial-1"}'
                        )
                    )
                ],
            )

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    response = OpenAICompatibleAdvisor(client=client).advise(
        {"default_action": {"kind": "resume", "trial_id": "trial-1"}}
    )
    assert response.action == {"kind": "resume", "trial_id": "trial-1"}
    assert response.model == "gpt-5.6-sol"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["temperature"] == 0


def test_validate_start_config_rejects_non_finite_and_out_of_bounds():
    validate_start_config((0.0, 0.5, 1.0))  # ok
    with pytest.raises(ValueError):
        validate_start_config((0.0, 1.2))
    with pytest.raises(ValueError):
        validate_start_config((float("nan"), 0.5))


def test_validate_action_rejects_illegal_kind():
    validate_action({"kind": "resume", "trial_id": "t1"})
    validate_action({"kind": "start", "unit_config": [0.1, 0.2]})
    with pytest.raises(ValueError):
        validate_action({"kind": "delete_everything"})
    with pytest.raises(ValueError):
        validate_action({"kind": "start", "unit_config": [1.5]})  # out of bounds
    with pytest.raises(ValueError):
        validate_action({"kind": "resume", "trial_id": "t1", "unit_config": [0.5]})
    with pytest.raises(ValueError):
        validate_action({"kind": "ipbt_exploit"})
    with pytest.raises(ValueError):
        validate_action({"kind": "ipbt_restart", "restart_id": "r1"})
    validate_action(
        {
            "kind": "ipbt_restart",
            "restart_id": "r1",
            "target_slot_id": "slot-1",
        }
    )
    with pytest.raises(ValueError):
        validate_action({"kind": "start", "unit_config": [0.5]}, expected_dim=2)


def test_build_advisor_state_includes_cma_mean_explicitly():
    state = build_advisor_state(
        cma_mean=[0.1, 0.2],
        cma_sigma=0.3,
        cma_cov=[[1.0, 0.0], [0.0, 1.0]],
        cma_proposal=[0.15, 0.25],
        ifbo_action={"kind": "resume", "trial_id": "t1"},
        ifbo_alternatives=[],
        population_lineages=[],
        btt_evidence=[],
        incumbent={"trial_id": "t1", "y": 0.9},
        top_five=[],
        recent_decisions=[],
        bounds=[[0, 1], [0, 1]],
        remaining_budget=1000,
        action_schema={"kinds": ["start", "resume"]},
    )
    # The pinned upstream extracts but never prompts the mean; ours must include it.
    assert state["cma_mean"] == [0.1, 0.2]
    assert "action_schema" in state


class _GoodAdvisor:
    def advise(self, state):
        return AdvisorResponse(
            action={"kind": "start", "unit_config": [0.5, 0.5]},
            raw_text="{}",
            model="frontier-x",
            version="2026-01-01",
            latency_ms=12.0,
        )


class _BrokenAdvisor:
    def advise(self, state):
        raise TimeoutError("provider 503")


def test_overlay_returns_llm_action_when_scheduled():
    overlay = CentaurOverlay(warmup=0, ratio=1.0)
    vec, source, record = overlay.propose(
        proposal_id=0, cma_config=(0.1, 0.2), advisor=_GoodAdvisor(), state={"cma_mean": [0.1, 0.2]}
    )
    assert source is ProposalSource.LLM
    assert tuple(vec) == (0.5, 0.5)
    assert record.model == "frontier-x" and record.latency_ms == 12.0


def test_overlay_uses_cma_when_not_scheduled():
    overlay = CentaurOverlay(warmup=100, ratio=0.3)
    vec, source, record = overlay.propose(
        proposal_id=0, cma_config=(0.1, 0.2), advisor=_GoodAdvisor(), state={}
    )
    assert source is ProposalSource.CMA
    assert tuple(vec) == (0.1, 0.2)
    assert record is None


def test_overlay_fails_loud_on_advisor_error_never_silent_cma():
    overlay = CentaurOverlay(warmup=0, ratio=1.0)
    with pytest.raises(AdvisorUnavailable):
        overlay.propose(proposal_id=0, cma_config=(0.1, 0.2), advisor=_BrokenAdvisor(), state={})


def test_overlay_schema_failure_is_audited_and_fails_loud():
    class InvalidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action={"kind": "resume", "trial_id": "t1"},
                raw_text='{"kind":"resume"}',
                model="frontier-x",
                version="v1",
                latency_ms=1.0,
            )

    with pytest.raises(AdvisorUnavailable) as exc:
        CentaurOverlay(warmup=0, ratio=1.0).propose(
            proposal_id=0,
            cma_config=(0.1, 0.2),
            advisor=InvalidAdvisor(),
            state={"cma_mean": [0.1, 0.2]},
        )
    assert exc.value.record is not None
    assert exc.value.record.response.raw_text == '{"kind":"resume"}'


def test_multi_action_start_validates_expected_dimension():
    class InvalidAdvisor:
        def advise(self, state):
            return AdvisorResponse(
                action={"kind": "start", "unit_config": [0.5]},
                raw_text="{}",
                model="gpt-5.6-sol",
                version="v1",
                latency_ms=1.0,
            )

    with pytest.raises(AdvisorUnavailable) as exc:
        CentaurOverlay(warmup=0, ratio=1.0).decide(
            proposal_id=0,
            default_action={"kind": "start", "unit_config": [0.1, 0.2]},
            advisor=InvalidAdvisor(),
            state={},
            expected_dim=2,
        )
    assert exc.value.record is not None


def test_advisor_record_is_detached_from_mutable_inputs():
    state = {"nested": {"value": 1}}
    response = _GoodAdvisor().advise(state)

    class SameResponseAdvisor:
        def advise(self, prompt):
            return response

    _, _, record = CentaurOverlay(warmup=0, ratio=1.0).propose(
        proposal_id=0,
        cma_config=(0.1, 0.2),
        advisor=SameResponseAdvisor(),
        state=state,
    )
    state["nested"]["value"] = 2
    response.action["unit_config"][0] = 0.9
    assert record.prompt_state["nested"]["value"] == 1
    assert record.response.action["unit_config"][0] == 0.5


def test_cma_proposer_bounds_maximization_sign_and_public_state(monkeypatch):
    calls = {}

    class FakeCMA:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.mean = kwargs["mean"]
            self.population_size = kwargs["population_size"]
            self.generation = 0
            self._sigma = kwargs["sigma"]
            self._C = np.eye(len(self.mean))

        def ask(self):
            return np.array([0.25, 0.75])

        def tell(self, solutions):
            calls["tell"] = solutions

    module = types.ModuleType("cmaes")
    module.CMA = FakeCMA
    monkeypatch.setitem(sys.modules, "cmaes", module)
    from olmo_core.hpo.centaur import CMAESProposer

    proposer = CMAESProposer(dim=2, seed=0, population_size=2)
    batch = proposer.ask(2)
    proposer.tell([(batch[0], 0.9), (batch[1], 0.2)])
    assert np.array_equal(calls["init"]["bounds"], np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert [objective for _, objective in calls["tell"]] == [-0.9, -0.2]
    assert proposer.state() == {
        "mean": [0.5, 0.5],
        "sigma": 0.2,
        "covariance": [[1.0, 0.0], [0.0, 1.0]],
        "generation": 0,
        "population_size": 2,
    }


def test_real_cma_proposer_ask_tell_roundtrips():
    pytest.importorskip("cmaes")
    from olmo_core.hpo.centaur import CMAESProposer

    proposer = CMAESProposer(dim=3, seed=0, population_size=4)
    batch = proposer.ask(4)
    assert len(batch) == 4
    assert all(len(x) == 3 for x in batch)
    proposer.tell([(x, float(np.sum(x))) for x in batch])  # must not raise


def test_real_cma_state_round_trip_replays_rng():
    pytest.importorskip("cmaes")
    from olmo_core.hpo.centaur import CMAESProposer

    original = CMAESProposer(dim=2, seed=7, population_size=4)
    original.ask(4)
    state = original.state_dict()
    restored = CMAESProposer(dim=2, seed=999, population_size=4)
    restored.load_state_dict(state)
    assert restored.ask(4) == original.ask(4)
