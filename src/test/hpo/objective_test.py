import pytest

from olmo_core.hpo.objective import CENormalizer, EvaluatorGate


def test_endpoints_map_to_zero_and_one_and_are_monotonic():
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    assert norm.to_y(6.0) == pytest.approx(0.0)
    assert norm.to_y(2.0) == pytest.approx(1.0)
    # Lower CE -> strictly higher y (higher is better).
    assert norm.to_y(3.0) > norm.to_y(4.0)


def test_requires_best_below_worst():
    with pytest.raises(ValueError):
        CENormalizer(ce_at_zero=2.0, ce_at_one=6.0)  # best must be < worst


def test_unclamped_to_y_can_exit_unit_interval():
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    assert norm.to_y(7.0) < 0.0  # worse than calibration worst
    assert norm.to_y(1.0) > 1.0  # better than calibration best


def test_ftpfn_projection_rejects_out_of_range_without_clipping():
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    assert norm.to_ftpfn_y(4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        norm.to_ftpfn_y(7.0)  # would be < 0; reject, don't clip (avoids tie pile-ups)
    with pytest.raises(ValueError):
        norm.to_ftpfn_y(1.0)  # would be > 1


def test_rejects_non_finite_ce():
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            norm.to_y(bad)


def test_from_calibration_fits_endpoints_from_disjoint_runs():
    ces = [2.1, 3.4, 5.9, 4.0]
    norm = CENormalizer.from_calibration(ces)
    assert norm.ce_at_one == pytest.approx(min(ces))
    assert norm.ce_at_zero == pytest.approx(max(ces))
    assert norm.to_ftpfn_y(min(ces)) == pytest.approx(1.0)
    assert norm.to_ftpfn_y(max(ces)) == pytest.approx(0.0)


def test_from_calibration_rejects_negative_margin():
    with pytest.raises(ValueError):
        CENormalizer.from_calibration([2.0, 6.0], margin=-0.25)


def test_evaluator_gate_fails_closed_when_search_validation_absent():
    gate = EvaluatorGate(search_validation="ppl-val", untouched="ppl-holdout")
    gate.require_ready({"ppl-val", "downstream"})  # ok
    with pytest.raises(RuntimeError):
        gate.require_ready({"downstream"})  # search-validation missing -> fail closed


def test_evaluator_gate_requires_disjoint_splits():
    with pytest.raises(ValueError):
        EvaluatorGate(search_validation="same", untouched="same")


def test_ce_for_y_is_the_inverse_of_to_y():
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    for y in (0.0, 0.25, 0.5, 1.0):
        assert norm.to_y(norm.ce_for_y(y)) == pytest.approx(y)
    # Used to synthesize fantasy points for pending workers.
    assert norm.ce_for_y(0.5) == pytest.approx(4.0)
