"""Tests for the training-dose adjustment pre-registered on 2026-08-10.

Three things are under test and they are different kinds of thing.

* **The mechanism**, driven through the real ``SkipStepAdamW`` rather than restated. The whole
  amendment rests on "a declined step performs no update and does not increment the Adam step
  counter", and that sentence is worth an assertion against the optimizer the tranche runs
  rather than against a reading of it.
* **The frozen constants**, re-derived from the cell table published in ``hyper-connections.md``
  before this code existed. A pre-registered constant that nothing recomputes is a constant that
  drifts.
* **The rule**, including the property that makes it safe to freeze a slope this imprecise: it
  can withhold a claim and it can never create one.
"""

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dose_adjustment as dose  # noqa: E402

from olmo_core.optim import SkipStepAdamW  # noqa: E402

# ---------------------------------------------------------------------------------------
# The mechanism, against the optimizer the tranche actually runs.
# ---------------------------------------------------------------------------------------


def _drive(optim: SkipStepAdamW, parameter: torch.Tensor, loss: float, grad_norm: float) -> None:
    """One step, with the two signals set the way the train module sets them."""
    optim.latest_loss = torch.tensor(loss)
    optim.latest_grad_norm = torch.tensor(grad_norm)
    parameter.grad = torch.ones_like(parameter)
    optim.step()


@pytest.mark.parametrize("foreach", [False, True])
def test_a_declined_step_moves_no_parameter_no_moment_and_no_step_counter(foreach: bool):
    """
    THE FACT THE WHOLE DOSE ADJUSTMENT RESTS ON. ``get_step_factor`` returns 0.0 and that factor
    multiplies the decoupled weight decay, both moment updates, the parameter update and the
    increment of the Adam step counter. So a declined step is a complete no-op on the optimizer
    while the trainer's global step, the cosine schedule and the data loader advance anyway --
    which is what makes the amount of training an arm receives a function of how often it
    declines.

    Both kernel paths, because ``SkipStepAdamWConfig`` defaults to ``foreach=True`` and the
    non-foreach path is the one that is easiest to read.
    """
    parameter = torch.nn.Parameter(torch.ones(4))
    optim = SkipStepAdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.5,
        rolling_interval_length=16,
        sigma_factor=6,
        foreach=foreach,
    )

    for i in range(20):
        _drive(optim, parameter, loss=2.5 + 0.001 * (i % 3), grad_norm=0.15 + 0.001 * (i % 3))

    state = optim.state[parameter]
    assert float(optim.step_skipped) == 0.0, "the flat prefix should all have been taken"
    taken = float(state["step"])
    assert taken == 20.0, "every step of a flat series is applied, so the counter is the count"

    before = (
        parameter.detach().clone(),
        state["exp_avg"].detach().clone(),
        state["exp_avg_sq"].detach().clone(),
        state["step"].detach().clone(),
    )

    _drive(optim, parameter, loss=500.0, grad_norm=500.0)

    assert float(optim.step_skipped) == 1.0, "a 500-sigma excursion has to be declined"
    torch.testing.assert_close(parameter.detach(), before[0])
    torch.testing.assert_close(state["exp_avg"].detach(), before[1])
    torch.testing.assert_close(state["exp_avg_sq"].detach(), before[2])
    torch.testing.assert_close(state["step"].detach(), before[3])
    assert float(state["step"]) == taken, "a declined step must not increment the Adam counter"


@pytest.mark.parametrize("foreach", [False, True])
def test_an_accepted_step_moves_all_four_so_the_test_above_is_not_vacuous(foreach: bool):
    """
    The complement. If the assertions above passed because nothing ever moves, they say nothing.
    """
    parameter = torch.nn.Parameter(torch.ones(4))
    optim = SkipStepAdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.5,
        rolling_interval_length=16,
        sigma_factor=6,
        foreach=foreach,
    )
    for i in range(20):
        _drive(optim, parameter, loss=2.5 + 0.001 * (i % 3), grad_norm=0.15 + 0.001 * (i % 3))

    state = optim.state[parameter]
    before = (
        parameter.detach().clone(),
        state["exp_avg"].detach().clone(),
        state["exp_avg_sq"].detach().clone(),
        float(state["step"]),
    )

    _drive(optim, parameter, loss=2.5, grad_norm=0.15)

    assert float(optim.step_skipped) == 0.0
    assert not torch.allclose(parameter.detach(), before[0])
    assert not torch.allclose(state["exp_avg"].detach(), before[1])
    assert not torch.allclose(state["exp_avg_sq"].detach(), before[2])
    assert float(state["step"]) == before[3] + 1.0


def test_the_threshold_is_run_relative_so_the_count_is_not_monotone_in_instability():
    """
    WHY A DECLINE COUNT IS NOT A STABILITY MEASUREMENT, which is the second half of the finding.

    ``get_step_factor`` compares each step against the mean and standard deviation of *that
    run's own* previous window. A run whose gradient norms are uniformly ten times larger but
    just as smooth raises its own bar by the same factor and declines the same steps. So an arm
    can be far more unstable in absolute terms and decline no more often, and the dose
    adjustment reads the count as a training-duration variable rather than as evidence about
    stability.
    """
    counts = []
    for scale in (1.0, 10.0):
        parameter = torch.nn.Parameter(torch.ones(4))
        optim = SkipStepAdamW(
            [parameter], lr=0.1, rolling_interval_length=16, sigma_factor=6, foreach=False
        )
        declined = 0
        for i in range(60):
            _drive(
                optim,
                parameter,
                loss=2.5 + 0.001 * (i % 7),
                grad_norm=scale * (0.15 + 0.001 * (i % 7)),
            )
            declined += int(float(optim.step_skipped))
        counts.append(declined)

    assert counts[0] == counts[1], (
        "a uniform rescale of the gradient norms changed the decline count, so this test no "
        f"longer demonstrates what it claims: {counts}"
    )


def test_a_declined_step_poisons_its_own_detector_so_declines_cannot_cluster():
    """
    WHY ``Delta n`` ACCUMULATES AS A RATE AND NOT AS A BURST, which is what the amendment models.

    The setters append the step's loss and gradient norm to the rolling window *before*
    ``step()`` consults it, and nothing removes the value again when the step is declined. So a
    spike stays in the window that judges the following ``rolling_interval_length`` steps and
    lifts the very mean and standard deviation that declining it again would require.

    The consequence is not a mild one. An isolated spike is declined exactly once and identical
    spikes immediately after it are accepted, which is why the amendment declines to treat the
    count as a measurement of the missing dose: the same instability yields a different count
    depending on where in the window it lands.
    """
    parameter = torch.nn.Parameter(torch.ones(4))
    optim = SkipStepAdamW(
        [parameter], lr=0.1, rolling_interval_length=16, sigma_factor=6, foreach=False
    )
    for i in range(20):
        _drive(optim, parameter, loss=2.5 + 0.001 * (i % 3), grad_norm=0.15 + 0.001 * (i % 3))

    verdicts = []
    for _ in range(3):
        _drive(optim, parameter, loss=2.5, grad_norm=50.0)
        verdicts.append(bool(float(optim.step_skipped)))

    assert verdicts == [True, False, False], (
        "three identical gradient-norm spikes were expected to cost exactly one declined step, "
        "because the first one enters the window and raises the bar for the other two. Got "
        f"{verdicts}. If this has changed, the amendment's claim that declines are anti-clustered "
        "no longer holds and the way it models the dose difference needs revisiting."
    )


# ---------------------------------------------------------------------------------------
# The frozen constants.
# ---------------------------------------------------------------------------------------


def test_the_frozen_slope_is_the_one_the_published_cell_table_implies():
    """
    Re-derived from ``hyper-connections.md``'s cell-by-cell table, which was published before
    this module existed and cannot have been chosen to suit it.
    """
    point, low, high = dose.slope_from_clean_cells()
    assert point == pytest.approx(dose.PER_DECLINE_NATS, rel=1e-4)
    assert low == pytest.approx(dose.PER_DECLINE_NATS_LOW, rel=1e-4)
    assert high == pytest.approx(dose.PER_DECLINE_NATS_HIGH, rel=1e-4)


def test_the_interval_crosses_zero_and_the_high_end_is_the_operative_one():
    """
    The three-cell mean movement does not clear zero at df = 2, so the point estimate is not a
    measurement of the slope so much as a bound on it. Recorded as an assertion because the rule
    quotes the top of the interval, and quoting the top of an interval is only defensible while
    the interval is this wide.
    """
    assert dose.PER_DECLINE_NATS_LOW < 0.0 < dose.PER_DECLINE_NATS < dose.PER_DECLINE_NATS_HIGH
    assert dose.PER_DECLINE_NATS_HIGH / dose.PER_DECLINE_NATS > 2.5


def test_the_critical_decline_gap_is_the_number_the_design_has_to_live_with():
    """
    At the top of the interval the dose alone spans the whole pre-registered gate at a
    seventeen-step difference in declined counts, against the forty-nine the adversarial review
    quotes from the point estimate. Seventeen is the number the design has to survive: the
    amended baseline's own cells declined 10 to 20 steps, so a spread of that size is inside
    what one arm already produced.
    """
    assert dose.CRITICAL_DECLINE_GAP == pytest.approx(16.8, abs=0.2)
    assert dose.GATE_NATS / dose.PER_DECLINE_NATS == pytest.approx(48.7, abs=0.5)


def test_the_nats_conversion_is_the_one_the_rest_of_the_tranche_uses():
    """A second copy of a constant is a constant that will disagree with the first one."""
    from noise_floor import NATS_PER_BPB

    assert dose.NATS_PER_BPB == pytest.approx(NATS_PER_BPB, rel=1e-12)


def test_the_slope_refuses_to_be_quoted_without_an_interval():
    with pytest.raises(ValueError):
        dose.slope_from_clean_cells(movement_bpb=(0.0003,), declines=(18,))
    with pytest.raises(ValueError):
        dose.slope_from_clean_cells(movement_bpb=(0.0003, 0.0004), declines=(18,))


# ---------------------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------------------


def _check(delta_nats: float, gate_nats: float, treated, compared, **kwargs) -> dose.DoseCheck:
    return dose.dose_check(
        name="H1",
        treatment="faithful",
        comparator="baseline",
        delta_nats=delta_nats,
        gate_nats=gate_nats,
        declined_treatment=treated,
        declined_comparator=compared,
        **kwargs,
    )


def test_a_dose_that_opposes_the_claim_is_reported_and_never_penalised():
    """
    The treatment declined 60 steps more than the comparator, so it was trained *less*, and it
    improved on the comparator anyway. The dose cannot have manufactured that; it can only have
    hidden some of it. The claim stands and the report says the estimate is conservative.
    """
    check = _check(-0.0100, 0.0026, treated=[80] * 5, compared=[20] * 5)

    assert check.delta_declines == pytest.approx(60.0)
    assert not check.dose_favours_the_claim
    assert check.clears_gate_unadjusted and check.survives_the_dose
    assert "conservative" in check.verdict


def test_a_dose_that_favours_the_claim_and_is_large_enough_withholds_it():
    """
    The treatment declined 40 fewer steps -- it was trained more -- and the contrast is 0.0040
    nats. At the top of the interval that dose is worth 0.0062, which swallows the contrast
    whole. The gate is cleared and the claim is still not made.
    """
    check = _check(-0.0040, 0.0026, treated=[10] * 5, compared=[50] * 5)

    assert check.delta_declines == pytest.approx(-40.0)
    assert check.dose_favours_the_claim
    assert check.clears_gate_unadjusted
    assert not check.survives_the_dose
    assert "DOSE-LIMITED" in check.verdict


def test_a_dose_that_favours_the_claim_but_is_small_leaves_it_standing():
    """Three declined steps of difference is worth 0.0005 nats and cannot reach the gate."""
    check = _check(-0.0100, 0.0026, treated=[15] * 5, compared=[18] * 5)

    assert check.dose_favours_the_claim
    assert check.survives_the_dose
    assert "survives the dose" in check.verdict


def test_equal_decline_counts_change_nothing():
    check = _check(-0.0100, 0.0026, treated=[17] * 5, compared=[17] * 5)

    assert check.delta_declines == 0.0
    assert check.dose_nats == 0.0
    assert not check.dose_favours_the_claim
    assert check.survives_the_dose
    assert check.adjusted_delta_nats == pytest.approx(check.delta_nats)


@pytest.mark.parametrize("delta_nats", [-0.0025, -0.0010, 0.0, 0.0010, 0.0025])
@pytest.mark.parametrize("delta_declines", [-200, -40, -3, 0, 3, 40, 200])
def test_the_rule_can_only_withhold_a_claim_and_never_create_one(
    delta_nats: float, delta_declines: int
):
    """
    THE PROPERTY THAT MAKES IT SAFE TO FREEZE A SLOPE WITH THIS MUCH UNCERTAINTY IN IT. Nothing
    the dose block does can turn a contrast that failed the gate into one that passed. If it
    could, the slope would be a knob that moves results, and a knob estimated on three points
    that do not clear zero is the last thing a pre-registration should hand anybody.
    """
    check = _check(delta_nats, 0.0026, treated=[20 + delta_declines] * 5, compared=[20] * 5)

    assert check.survives_the_dose <= check.clears_gate_unadjusted
    if not check.clears_gate_unadjusted:
        assert not check.survives_the_dose


def test_a_missing_declined_count_refuses_rather_than_defaulting_to_zero():
    """
    An absent ``stability/steps skipped`` key is missing data. Reading it as zero would report a
    dose difference of zero for the arm whose instrumentation failed, which is the arm least
    entitled to the benefit of the doubt.
    """
    with pytest.raises(ValueError, match="missing data|is missing"):
        _check(-0.0100, 0.0026, treated=[10, None, 12, 13, 14], compared=[20] * 5)
    with pytest.raises(ValueError, match="missing data|is missing"):
        _check(-0.0100, 0.0026, treated=[10] * 5, compared=[20, 20, None, 20, 20])


def test_the_adjusted_point_estimate_is_the_contrast_minus_the_dose():
    check = _check(-0.0100, 0.0026, treated=[10] * 5, compared=[40] * 5)

    assert check.delta_declines == pytest.approx(-30.0)
    assert check.dose_nats == pytest.approx(-30.0 * dose.PER_DECLINE_NATS)
    assert check.adjusted_delta_nats == pytest.approx(-0.0100 + 30.0 * dose.PER_DECLINE_NATS)
    # Adjusting towards zero, because the treatment trained more and some of its advantage is
    # bought with the extra updates rather than with the mechanism.
    assert abs(check.adjusted_delta_nats) < abs(check.delta_nats)


def test_the_band_brackets_the_contrast_by_the_largest_dose_the_interval_allows():
    check = _check(-0.0100, 0.0026, treated=[10] * 5, compared=[40] * 5)

    width = abs(30.0 * dose.PER_DECLINE_NATS_HIGH)
    assert check.band_nats[0] == pytest.approx(-0.0100 - width)
    assert check.band_nats[1] == pytest.approx(-0.0100 + width)
    assert check.band_nats[0] < check.delta_nats < check.band_nats[1]


def test_the_rendered_block_names_its_date_and_every_hypothesis_it_checked():
    """
    The report is what a reader sees, and a pre-registered adjustment that is not visible in it
    is one nobody can check was applied.
    """
    checks = [
        _check(-0.0100, 0.0026, treated=[10] * 5, compared=[40] * 5),
        dose.dose_check(
            name="H2a",
            treatment="faithful",
            comparator="output-only",
            delta_nats=-0.0005,
            gate_nats=0.0026,
            declined_treatment=[12] * 5,
            declined_comparator=[13] * 5,
        ),
    ]
    text = dose.render(checks)

    assert "2026-08-10" in text
    assert "H1" in text and "H2a" in text
    assert "declined step" in text
    assert dose.render([]) == ""


def test_a_positive_contrast_with_a_dose_pushing_the_same_way_is_also_withheld():
    """
    The rule is about the direction the hypothesis predicts, not about the sign of the number.
    An arm predicted to improve the loss that instead comes back significantly *worse* is a
    result too, and a dose difference that could have manufactured it gets the same treatment --
    here the treatment declined 60 steps more, was trained less, and looks worse for it.
    """
    check = _check(0.0040, 0.0026, treated=[80] * 5, compared=[20] * 5, predicted_sign=1)

    assert check.dose_favours_the_claim
    assert check.clears_gate_unadjusted
    assert not check.survives_the_dose


def test_unequal_cell_counts_use_arm_means_rather_than_refusing():
    """
    A lost cell is reported elsewhere and makes the whole read provisional; it does not also
    have to stop the dose check, which reads arm means and is defined at unequal n.
    """
    check = _check(-0.0100, 0.0026, treated=[10, 12, 14], compared=[20] * 5)

    assert check.delta_declines == pytest.approx(12.0 - 20.0)
    assert math.isfinite(check.dose_nats)


# ---------------------------------------------------------------------------------------
# Multiplicity, which is the other half of the same amendment: funding arm 4 took the family
# from three comparisons to five, and the gate is uncorrected by design.
# ---------------------------------------------------------------------------------------


def test_holm_is_reported_beside_the_gate_and_does_not_replace_it():
    """
    Holm-Bonferroni over the primary rows. Checked against the definition rather than against a
    library, and checked for the two properties a step-down procedure has to have: it is at
    least as large as the raw value, and it is monotone in rank so two adjusted values cannot
    cross.
    """
    raw = {"H1": 0.001, "H1a": 0.02, "H1b": 0.30, "H2a": 0.04, "H5": 0.60}
    adjusted = dose.holm_adjust(raw)

    assert set(adjusted) == set(raw)
    # (n - rank) * p, in ascending order of p, with a running maximum.
    assert adjusted["H1"] == pytest.approx(5 * 0.001)
    assert adjusted["H1a"] == pytest.approx(4 * 0.02)
    assert adjusted["H2a"] == pytest.approx(max(4 * 0.02, 3 * 0.04))
    assert adjusted["H1b"] == pytest.approx(max(3 * 0.04, 2 * 0.30))
    assert adjusted["H5"] == pytest.approx(0.60)

    for name, value in adjusted.items():
        assert value >= raw[name], "an adjustment that lowers a p-value is not a correction"
        assert value <= 1.0

    ranked = sorted(raw, key=lambda k: raw[k])
    values = [adjusted[name] for name in ranked]
    assert values == sorted(values), "Holm has to be monotone in rank"


def test_holm_on_an_empty_or_single_family_is_the_identity():
    assert dose.holm_adjust({}) == {}
    assert dose.holm_adjust({"H1": 0.03}) == pytest.approx({"H1": 0.03})
