import math

import numpy as np
import pytest

from olmo_core.hpo.types import (
    ActionKind,
    Allocation,
    BTTVerdictKind,
    CurvePoint,
    ProposalSource,
    SearchDim,
    SearchSpace,
    TrialStatus,
    Verdict,
)


def _example_space() -> SearchSpace:
    return SearchSpace(
        (
            SearchDim("lr", 1e-4, 1e-2, log=True),
            SearchDim("weight_decay", 0.0, 0.3, log=False),
            SearchDim("beta2_gap", 1e-3, 1e-1, log=True),  # log10(1-beta2)
        )
    )


def test_search_space_unit_round_trip_linear_and_log():
    space = _example_space()
    hps = {"lr": 3e-3, "weight_decay": 0.12, "beta2_gap": 5e-2}
    unit = space.to_unit(hps)
    assert unit.shape == (3,)
    assert np.all(unit >= 0.0) and np.all(unit <= 1.0)
    back = space.from_unit(unit)
    for k, v in hps.items():
        assert back[k] == pytest.approx(v, rel=1e-9, abs=1e-12)


def test_search_space_endpoints_map_to_zero_and_one():
    space = _example_space()
    lo = space.to_unit({"lr": 1e-4, "weight_decay": 0.0, "beta2_gap": 1e-3})
    hi = space.to_unit({"lr": 1e-2, "weight_decay": 0.3, "beta2_gap": 1e-1})
    assert lo == pytest.approx(np.zeros(3), abs=1e-12)
    assert hi == pytest.approx(np.ones(3), abs=1e-12)


def test_search_space_rejects_more_than_ten_dims():
    dims = tuple(SearchDim(f"h{i}", 0.0, 1.0) for i in range(11))
    with pytest.raises(ValueError):
        SearchSpace(dims)


def test_from_unit_rejects_out_of_range_without_silent_clip():
    space = _example_space()
    with pytest.raises(ValueError):
        space.from_unit(np.array([1.2, 0.5, 0.5]))
    with pytest.raises(ValueError):
        space.from_unit(np.array([-0.01, 0.5, 0.5]))


def test_to_unit_rejects_out_of_bounds_value():
    space = _example_space()
    with pytest.raises(ValueError):
        space.to_unit({"lr": 1.0, "weight_decay": 0.1, "beta2_gap": 1e-2})


def test_search_dim_rejects_nonfinite_bounds_and_coordinates():
    for low, high in (
        (float("nan"), 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (-float("inf"), 1.0),
    ):
        with pytest.raises(ValueError):
            SearchDim("x", low, high)

    dim = SearchDim("x", 0.0, 1.0)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            dim.to_unit(value)
        with pytest.raises(ValueError):
            dim.from_unit(value)


def test_enum_values_are_stable_strings():
    assert ActionKind.START == "start"
    assert ActionKind.RESUME == "resume"
    assert {v.value for v in BTTVerdictKind} == {"healthy", "degraded", "saturated", "fatal"}
    assert ProposalSource.RANDOM == "random"
    assert TrialStatus.RUNNING == "running"


def test_verdict_binding_key_is_trial_fidelity_observation():
    v = Verdict(
        kind=BTTVerdictKind.DEGRADED,
        indicators=("EAG",),
        trial_id="t3",
        completed_fidelity=2048,
        observation_hash="abc123",
        profile_version="btt-v1",
    )
    assert v.binding_key == ("t3", 2048, "abc123")
    assert v.is_eligible_for_resume() is False
    assert (
        Verdict(
            kind=BTTVerdictKind.HEALTHY,
            indicators=(),
            trial_id="t3",
            completed_fidelity=2048,
            observation_hash="abc123",
            profile_version="btt-v1",
        ).is_eligible_for_resume()
        is True
    )
    # SATURATED (NMG) is not a degraded failure: it remains an eligible incumbent.
    assert (
        Verdict(
            kind=BTTVerdictKind.SATURATED,
            indicators=("NMG",),
            trial_id="t3",
            completed_fidelity=2048,
            observation_hash="abc123",
            profile_version="btt-v1",
        ).is_incumbent_candidate()
        is True
    )


def test_allocation_json_round_trip_preserves_fields():
    space = _example_space()
    hps = {"lr": 3e-3, "weight_decay": 0.12, "beta2_gap": 5e-2}
    alloc = Allocation(
        decision_id=7,
        kind=ActionKind.START,
        trial_id="t7_0",
        parent_trial_id=None,
        unit_config=tuple(space.to_unit(hps).tolist()),
        realized_hps=hps,
        current_fidelity=0,
        target_fidelity=1024,
        checkpoint_ref=None,
        horizon=3,
        threshold=0.42,
        mfpi_score=0.87,
        tie_break=(0.87, 7),
        source=ProposalSource.CMA,
        verdict_id=None,
    )
    d = alloc.to_dict()
    # Must be JSON-safe (only primitive containers).
    import json

    restored = Allocation.from_dict(json.loads(json.dumps(d)))
    assert restored == alloc
    assert restored.kind is ActionKind.START
    assert restored.source is ProposalSource.CMA


def test_curve_point_normalized_optional():
    p = CurvePoint(tokens=1024, ce=3.5)
    assert p.y is None
    p2 = CurvePoint(tokens=1024, ce=3.5, y=0.6)
    assert math.isclose(p2.y, 0.6)
