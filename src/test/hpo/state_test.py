import json
from dataclasses import replace

import pytest

from olmo_core.hpo.state import Event, EventKind, EventLog, observation_hash, replay
from olmo_core.hpo.types import (
    ActionKind,
    Allocation,
    BTTDisposition,
    BTTVerdictKind,
    ProposalSource,
    TrialStatus,
    Verdict,
)


def _alloc(decision_id, trial_id, kind, current, target, parent=None):
    return Allocation(
        decision_id=decision_id,
        kind=kind,
        trial_id=trial_id,
        parent_trial_id=parent,
        unit_config=(0.5,),
        realized_hps={"lr": 1e-3},
        current_fidelity=current,
        target_fidelity=target,
        checkpoint_ref=None,
        horizon=1,
        threshold=0.1,
        mfpi_score=0.5,
        tie_break=(0.5, decision_id),
        source=ProposalSource.RANDOM,
    )


def test_event_log_enforces_monotonic_sequence():
    log = EventLog()
    log.append(Event(seq=0, kind=EventKind.ALLOCATION, payload={}))
    log.append(Event(seq=1, kind=EventKind.OBSERVATION, payload={}))
    with pytest.raises(ValueError):
        log.append(Event(seq=1, kind=EventKind.OBSERVATION, payload={}))  # duplicate
    with pytest.raises(ValueError):
        log.append(Event(seq=0, kind=EventKind.OBSERVATION, payload={}))  # backwards


def test_event_log_jsonl_round_trip():
    log = EventLog()
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": 3.5},
        )
    )
    text = log.to_jsonl()
    assert text.count("\n") == 2  # one line per event, trailing newline
    reloaded = EventLog.from_jsonl(text)
    assert [e.seq for e in reloaded] == [0, 1]
    assert reloaded.events[0].kind is EventKind.ALLOCATION


def test_appended_event_payload_is_immutable():
    payload = _alloc(0, "t0_0", ActionKind.START, 0, 10).to_dict()
    log = EventLog()
    log.append(Event(seq=0, kind=EventKind.ALLOCATION, payload=payload))
    payload["target_fidelity"] = 99
    exposed = log.events[0]
    exposed.payload["target_fidelity"] = 77
    assert replay(log.events).tokens_charged == 10


def test_nonfinite_observation_round_trips_as_strict_json():
    log = EventLog()
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": float("nan")},
        )
    )
    text = log.to_jsonl()
    assert "NaN" not in text
    for line in text.splitlines():
        json.loads(line, parse_constant=lambda value: pytest.fail(f"non-strict JSON: {value}"))
    reloaded = EventLog.from_jsonl(text)
    assert reloaded.to_jsonl() == text
    assert replay(reloaded.events).trials["t0_0"].curve == []


def test_reserved_nonfinite_tag_dictionary_round_trips():
    payload = {"nested": {"__hpo_nonfinite_float__": "nan"}}
    event = Event(seq=0, kind=EventKind.ADVISOR, payload=payload)
    restored = Event.from_json_line(event.to_json_line())
    assert restored.payload == payload


def test_budget_is_charged_incrementally_not_cumulatively():
    log = EventLog()
    h1 = observation_hash("t0_0", 1024, 3.5)
    # Start t0 to 1024, then resume it to 3072. Correct resume consumes 1024 then 2048 = 3072.
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": 3.5, "observation_hash": h1},
        )
    )
    log.append(
        Event(
            seq=2,
            kind=EventKind.VERDICT,
            payload={
                "kind": BTTVerdictKind.HEALTHY.value,
                "indicators": [],
                "trial_id": "t0_0",
                "completed_fidelity": 1024,
                "observation_hash": h1,
                "profile_version": "btt-v1",
            },
        )
    )
    log.append(
        Event(
            seq=3,
            kind=EventKind.ALLOCATION,
            payload=_alloc(1, "t0_0", ActionKind.RESUME, 1024, 3072).to_dict(),
        )
    )
    state = replay(log.events)
    assert state.tokens_charged == 3072  # not 1024 + 3072


def test_next_decision_id_tracks_allocation_count():
    log = EventLog()
    assert replay(log.events).next_decision_id == 0
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": 3.5},
        )
    )
    log.append(
        Event(
            seq=2,
            kind=EventKind.ALLOCATION,
            payload=_alloc(1, "t1_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    assert replay(log.events).next_decision_id == 2


def test_verdict_and_status_are_folded_into_state():
    log = EventLog()
    fatal_hash = observation_hash("t0_0", 1024, float("nan"))
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={
                "trial_id": "t0_0",
                "tokens": 1024,
                "ce": float("nan"),
                "observation_hash": fatal_hash,
            },
        )
    )
    verdict = Verdict(
        kind=BTTVerdictKind.FATAL,
        indicators=("AGV",),
        trial_id="t0_0",
        completed_fidelity=1024,
        observation_hash=fatal_hash,
        profile_version="btt-v1",
    )
    log.append(
        Event(
            seq=2,
            kind=EventKind.VERDICT,
            payload={
                "kind": verdict.kind.value,
                "indicators": list(verdict.indicators),
                "trial_id": verdict.trial_id,
                "completed_fidelity": verdict.completed_fidelity,
                "observation_hash": verdict.observation_hash,
                "profile_version": verdict.profile_version,
            },
        )
    )
    state = replay(log.events)
    assert state.trials["t0_0"].latest_verdict.kind is BTTVerdictKind.FATAL
    assert state.trials["t0_0"].status is not None


def test_replay_is_deterministic_across_serialization():
    log = EventLog()
    h1 = observation_hash("t0_0", 1024, 3.5)
    h2 = observation_hash("t0_0", 2048, 3.1)
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": 3.5, "observation_hash": h1},
        )
    )
    log.append(
        Event(
            seq=2,
            kind=EventKind.VERDICT,
            payload={
                "kind": BTTVerdictKind.HEALTHY.value,
                "indicators": [],
                "trial_id": "t0_0",
                "completed_fidelity": 1024,
                "observation_hash": h1,
                "profile_version": "btt-v1",
            },
        )
    )
    log.append(
        Event(
            seq=3,
            kind=EventKind.ALLOCATION,
            payload=_alloc(1, "t0_0", ActionKind.RESUME, 1024, 2048).to_dict(),
        )
    )
    log.append(
        Event(
            seq=4,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 2048, "ce": 3.1, "observation_hash": h2},
        )
    )

    snap_a = replay(log.events).snapshot()
    reloaded = EventLog.from_jsonl(json.dumps and log.to_jsonl())  # ensure json module used
    snap_b = replay(reloaded.events).snapshot()
    assert snap_a == snap_b
    assert snap_a["trials"]["t0_0"]["curve"] == [[1024, 3.5], [2048, 3.1]]


def test_allocation_tracks_pending_and_completed_fidelity_separately():
    log = EventLog()
    h1 = observation_hash("t0_0", 1024, 3.5)
    log.append(
        Event(
            seq=0,
            kind=EventKind.ALLOCATION,
            payload=_alloc(0, "t0_0", ActionKind.START, 0, 1024).to_dict(),
        )
    )
    pending = replay(log.events).trials["t0_0"]
    assert pending.current_fidelity == 0
    assert pending.pending_target_fidelity == 1024
    log.append(
        Event(
            seq=1,
            kind=EventKind.OBSERVATION,
            payload={"trial_id": "t0_0", "tokens": 1024, "ce": 3.5, "observation_hash": h1},
        )
    )
    complete = replay(log.events).trials["t0_0"]
    assert complete.current_fidelity == 1024
    assert complete.pending_target_fidelity is None


def test_replay_rejects_duplicate_decision_and_mismatched_resume_frontier():
    duplicate = EventLog()
    duplicate.append(
        Event(0, EventKind.ALLOCATION, _alloc(0, "t0", ActionKind.START, 0, 10).to_dict())
    )
    duplicate.append(
        Event(1, EventKind.ALLOCATION, _alloc(0, "t1", ActionKind.START, 0, 10).to_dict())
    )
    with pytest.raises(ValueError):
        replay(duplicate.events)

    mismatch = EventLog()
    mismatch.append(
        Event(0, EventKind.ALLOCATION, _alloc(0, "t0", ActionKind.START, 0, 10).to_dict())
    )
    mismatch_hash = observation_hash("t0", 10, 3.5)
    mismatch.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {"trial_id": "t0", "tokens": 10, "ce": 3.5, "observation_hash": mismatch_hash},
        )
    )
    mismatch.append(
        Event(
            2,
            EventKind.VERDICT,
            {
                "kind": BTTVerdictKind.HEALTHY.value,
                "indicators": [],
                "trial_id": "t0",
                "completed_fidelity": 10,
                "observation_hash": mismatch_hash,
                "profile_version": "btt-v1",
            },
        )
    )
    mismatch.append(
        Event(3, EventKind.ALLOCATION, _alloc(1, "t0", ActionKind.RESUME, 0, 20).to_dict())
    )
    with pytest.raises(ValueError):
        replay(mismatch.events)


def test_replay_rejects_verdict_not_bound_to_latest_observation():
    log = EventLog()
    latest_hash = observation_hash("t0", 20, 3.5)
    log.append(Event(0, EventKind.ALLOCATION, _alloc(0, "t0", ActionKind.START, 0, 20).to_dict()))
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {"trial_id": "t0", "tokens": 20, "ce": 3.5, "observation_hash": latest_hash},
        )
    )
    log.append(
        Event(
            2,
            EventKind.VERDICT,
            {
                "kind": BTTVerdictKind.FATAL.value,
                "indicators": ["AGV"],
                "trial_id": "t0",
                "completed_fidelity": 10,
                "observation_hash": "stale",
                "profile_version": "btt-v1",
            },
        )
    )
    with pytest.raises(ValueError):
        replay(log.events)


def test_replay_rejects_observation_hash_mismatch():
    log = EventLog()
    log.append(
        Event(
            0,
            EventKind.ALLOCATION,
            _alloc(0, "t0", ActionKind.START, 0, 10).to_dict(),
        )
    )
    valid_hash = observation_hash("t0", 10, 3.5)
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {
                "trial_id": "t0",
                "tokens": 10,
                "ce": 9.9,
                "observation_hash": valid_hash,
            },
        )
    )
    with pytest.raises(ValueError):
        replay(log.events)


def test_observation_hash_binds_btt_telemetry():
    normal = observation_hash(
        "t0",
        10,
        3.5,
        train_ce_history=(4.0, 3.5),
        grad_norm_history=(1.0, 1.1),
        activation_ratio=0.5,
        numeric_failure=False,
    )
    exploding = observation_hash(
        "t0",
        10,
        3.5,
        train_ce_history=(4.0, 3.5),
        grad_norm_history=(1.0, 100.0),
        activation_ratio=0.5,
        numeric_failure=False,
    )
    assert exploding != normal


def test_verdict_spared_by_reserve_survives_replay():
    log = EventLog()
    obs_hash = observation_hash("t0", 10, 3.5)
    log.append(Event(0, EventKind.ALLOCATION, _alloc(0, "t0", ActionKind.START, 0, 10).to_dict()))
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {"trial_id": "t0", "tokens": 10, "ce": 3.5, "observation_hash": obs_hash},
        )
    )
    log.append(
        Event(
            2,
            EventKind.VERDICT,
            {
                "kind": BTTVerdictKind.HEALTHY.value,
                "indicators": ["PLC"],
                "trial_id": "t0",
                "completed_fidelity": 10,
                "observation_hash": obs_hash,
                "profile_version": "btt-v1",
                "spared_by_reserve": True,
            },
        )
    )
    assert replay(log.events).trials["t0"].latest_verdict.spared_by_reserve is True


def test_verdict_disposition_survives_replay():
    log = EventLog()
    log.append(Event(0, EventKind.ALLOCATION, _alloc(0, "t0", ActionKind.START, 0, 10).to_dict()))
    obs_hash = observation_hash("t0", 10, 4.0)
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {"trial_id": "t0", "tokens": 10, "ce": 4.0, "observation_hash": obs_hash},
        )
    )
    log.append(
        Event(
            2,
            EventKind.VERDICT,
            {
                "kind": BTTVerdictKind.DEGRADED.value,
                "indicators": ["PLC"],
                "trial_id": "t0",
                "completed_fidelity": 10,
                "observation_hash": obs_hash,
                "profile_version": "btt-paper-v1",
                "disposition": BTTDisposition.STOP.value,
            },
        )
    )
    assert replay(log.events).trials["t0"].latest_verdict.disposition is BTTDisposition.STOP


def test_paper_stop_is_terminal_while_adapted_recycle_is_paused():
    def status_for(disposition):
        log = EventLog()
        log.append(
            Event(
                0,
                EventKind.ALLOCATION,
                _alloc(0, "t0", ActionKind.START, 0, 10).to_dict(),
            )
        )
        obs_hash = observation_hash("t0", 10, 4.0)
        log.append(
            Event(
                1,
                EventKind.OBSERVATION,
                {
                    "trial_id": "t0",
                    "tokens": 10,
                    "ce": 4.0,
                    "observation_hash": obs_hash,
                },
            )
        )
        log.append(
            Event(
                2,
                EventKind.VERDICT,
                {
                    "kind": BTTVerdictKind.DEGRADED.value,
                    "indicators": ["PLC"],
                    "trial_id": "t0",
                    "completed_fidelity": 10,
                    "observation_hash": obs_hash,
                    "profile_version": "test",
                    "disposition": disposition.value,
                },
            )
        )
        return replay(log.events).trials["t0"].status

    assert status_for(BTTDisposition.STOP) is TrialStatus.RETIRED
    assert status_for(BTTDisposition.RECYCLE) is TrialStatus.PAUSED


def test_inherited_start_preserves_transition_lineage():
    parent = _alloc(0, "parent", ActionKind.START, 0, 10)
    child = replace(
        _alloc(1, "child", ActionKind.START, 10, 20, parent="parent"),
        transition={
            "lineage_id": "L-child",
            "parent_lineage_id": "L-parent",
        },
    )
    log = EventLog()
    log.append(Event(0, EventKind.ALLOCATION, parent.to_dict()))
    parent_hash = observation_hash("parent", 10, 4.0)
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {
                "trial_id": "parent",
                "tokens": 10,
                "ce": 4.0,
                "observation_hash": parent_hash,
            },
        )
    )
    log.append(Event(2, EventKind.ALLOCATION, child.to_dict()))
    record = replay(log.events).trials["child"]
    assert record.lineage_id == "L-child"
    assert record.parent_lineage_id == "L-parent"


def test_inherited_start_may_use_historical_parent_checkpoint():
    parent = _alloc(0, "parent", ActionKind.START, 0, 20)
    child = replace(
        _alloc(1, "child", ActionKind.START, 10, 20, parent="parent"),
        checkpoint_ref="/checkpoints/parent/step10",
    )
    log = EventLog()
    log.append(Event(0, EventKind.ALLOCATION, parent.to_dict()))
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {
                "trial_id": "parent",
                "tokens": 20,
                "ce": 3.0,
                "observation_hash": observation_hash("parent", 20, 3.0),
            },
        )
    )
    log.append(Event(2, EventKind.ALLOCATION, child.to_dict()))

    state = replay(log.events)
    assert state.trials["child"].current_fidelity == 10


def test_inherited_start_rejects_future_parent_fidelity():
    parent = _alloc(0, "parent", ActionKind.START, 0, 10)
    child = _alloc(1, "child", ActionKind.START, 20, 30, parent="parent")
    log = EventLog()
    log.append(Event(0, EventKind.ALLOCATION, parent.to_dict()))
    log.append(
        Event(
            1,
            EventKind.OBSERVATION,
            {
                "trial_id": "parent",
                "tokens": 10,
                "ce": 4.0,
                "observation_hash": observation_hash("parent", 10, 4.0),
            },
        )
    )
    log.append(Event(2, EventKind.ALLOCATION, child.to_dict()))

    with pytest.raises(ValueError, match="cannot begin beyond"):
        replay(log.events)
