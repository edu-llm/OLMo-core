"""
The admission gates, exercised on endpoints whose numbers we choose.

PRD 8.6 decides whether a score may be read at all, so the tests that carry weight are the ones where a
gate could plausibly say yes and has to say no: an endpoint pinned to its own floor, an untrained model
beating its baseline, a positive control that never moves -- and, the failure the whole section exists
to prevent, a gate whose evidence was never produced. Every one of those returns a verdict rather than
raising, so each is a fixture and an assertion rather than a paragraph in a design document.

PRD 13 asks for "a passing *and* a failing fixture for **each** of G1-G8". These are those, plus the
boundary values, where inclusive-versus-exclusive is a decision and is stated as one.
"""

import math
from typing import Any, Dict

import pytest
from factcrowd.measure import gates
from factcrowd.measure.endpoints import EndpointResult

from olmo_core.exceptions import OLMoConfigurationError

MANO_FLOOR = 0.0464
"""PRD 8.3's measured degenerate floor for ``<mano>`` at L=10, so the fixtures sit on real numbers."""


def endpoint(accuracy, *, floor=0.05, n=10_000, unparseable=0.0, name="mano"):
    """An :class:`EndpointResult` with the accuracy, floor and unparseable rate a case needs."""
    return EndpointResult(
        name=name,
        n_total=n,
        n_correct=round(accuracy * n),
        n_degenerate=0,
        n_unparseable=round(unparseable * n),
        answer_ce_bits=2.5,
        floor=floor,
    )


def full_evidence(**overrides):
    """
    Everything :func:`gates.run_gates` needs from an endpoint that passes all seven.

    Built from PRD 8.3's Mano figures where they exist -- 47.8 at L=10 against 19.4 at L=13, +18.2pp
    across the parameter ladder -- so a test that breaks one gate has to say which number it changed.
    """
    evidence = dict(
        result=endpoint(0.478, floor=MANO_FLOOR),
        depth_scores={10: 0.478, 13: 0.194},
        random_init_result=endpoint(0.047, floor=MANO_FLOOR),
        premise_ablated_result=endpoint(0.047, floor=MANO_FLOOR),
        achievable_ceiling=0.52,
        scores_by_params={13_000_000: 0.400, 28_000_000: 0.478, 64_000_000: 0.582},
        replicates=[endpoint(0.478), endpoint(0.4785), endpoint(0.4790)],
        dilution_scores={100: 0.478, 95: 0.4775, 90: 0.4740, 80: 0.4630, 60: 0.4380},
    )
    evidence.update(overrides)
    return evidence


def evidence_of(result):
    """A gate's evidence pairs as a mapping, for asserting on numbers the detail only prose-ifies."""
    return dict(result.evidence)


# --- the verdict type --------------------------------------------------------------------------------


def test_a_verdict_carries_the_number_the_limit_and_everything_else_it_computed():
    """
    A log line has to be actionable on its own. Whoever reads "G7 failed" at 3am should not have to
    open the PRD to learn which number was over which limit.
    """
    verdict = gates.g7_resolution([endpoint(0.50), endpoint(0.52), endpoint(0.54)])
    summary = verdict.summary()
    assert summary["gate"] == "G7" and summary["passed"] is False
    assert summary["value"] == pytest.approx(2.0) and summary["threshold"] == pytest.approx(0.65)
    assert float(summary["g7_mde_pp"]) > 2.0  # type: ignore[arg-type]
    assert "0.65pp limit" in str(summary["detail"])


def test_a_verdict_is_hashable_so_collecting_verdicts_cannot_fail_later():
    """
    The reason evidence is a tuple of pairs and not a dict: a frozen dataclass holding a dict looks
    fine until the first ``hash()``, which is somewhere else entirely and much later.
    """
    verdict = gates.g6_capacity_responsive({13_000_000: 0.40, 64_000_000: 0.58})
    assert len({verdict, verdict}) == 1


# --- the resolution arithmetic -----------------------------------------------------------------------


def test_the_mde_reproduces_the_sigma_prd_8_6_derives():
    """
    PRD 8.6: "a one-seed 5-point trend needs run-level sigma <= 0.63pp to see 2pp at 80% power". If
    this drifts, G7's threshold stops meaning what the section says it means.
    """
    assert gates.minimum_detectable_effect(0.636) == pytest.approx(2.0, abs=0.01)


def test_the_gate_threshold_is_looser_than_the_power_calculation_and_says_so():
    """
    0.65pp is PRD 8.6's table; 0.636pp is PRD 8.6's arithmetic. Publishing the MDE beside sigma is
    what keeps that rounding visible instead of buried in a constant.
    """
    assert gates.minimum_detectable_effect(gates.SIGMA_MAX_PP) > gates.TARGET_EFFECT_PP


@pytest.mark.parametrize(
    "kwargs,match",
    [(dict(n_points=2), "at least 3 points"), (dict(sigma_pp=-1.0), "non-negative")],
)
def test_an_mde_that_cannot_be_computed_raises(kwargs, match):
    with pytest.raises(OLMoConfigurationError, match=match):
        arguments: Dict[str, Any] = {"sigma_pp": 0.5, **kwargs}
        gates.minimum_detectable_effect(**arguments)


# --- G1: dynamic range -------------------------------------------------------------------------------


def test_g1_admits_an_endpoint_in_band_with_a_working_difficulty_dial():
    verdict = gates.g1_dynamic_range(
        endpoint(0.478, floor=MANO_FLOOR), depth_scores={10: 0.478, 13: 0.194}
    )
    assert verdict.passed
    assert evidence_of(verdict)["position_pct"] == pytest.approx(45.3, abs=0.1)
    assert evidence_of(verdict)["depth_spread_pp"] == pytest.approx(28.4, abs=0.1)


def test_g1_refuses_an_endpoint_pinned_to_its_own_floor():
    """
    PRD 8.3's worked example, and the reason Mano runs at L=10. At L=13 the task scores 8.2 against a
    6.80% best-constant policy -- 1.5% of its range -- so a decline has nowhere to fall and a null
    would be a property of the task. The retune happened because this gate was applied first.
    """
    verdict = gates.g1_dynamic_range(
        endpoint(0.082, floor=0.068), depth_scores={10: 0.478, 13: 0.194}
    )
    assert not verdict.passed
    assert "1.4pp above its own 6.8pp floor" in verdict.detail


def test_g1_refuses_a_saturated_endpoint():
    """The other edge: at 96% of range a decline is read as a ceiling rather than as an effect."""
    verdict = gates.g1_dynamic_range(endpoint(0.96, floor=0.05), depth_scores={6: 0.60, 10: 0.45})
    assert not verdict.passed and "saturated" in verdict.detail


def test_g1_band_is_closed_at_both_edges():
    """
    **The convention: 20% and 80% of range are admitted, and only past them is refused.**

    Both edges are round numbers chosen to be legible, not measured cliffs, so a verdict that turned on
    the last bit of a float would be false precision. A floor of 5% puts exactly 20.0% of range at 24pp
    and exactly 80.0% at 81pp, with no rounding in either.
    """
    dial = {6: 0.60, 10: 0.45}
    assert gates.g1_dynamic_range(endpoint(0.24, floor=0.05), depth_scores=dial).passed
    assert gates.g1_dynamic_range(endpoint(0.81, floor=0.05), depth_scores=dial).passed
    assert not gates.g1_dynamic_range(endpoint(0.23, floor=0.05), depth_scores=dial).passed
    assert not gates.g1_dynamic_range(endpoint(0.82, floor=0.05), depth_scores=dial).passed


def test_g1_depth_requirement_is_closed_at_fifteen_points():
    """``>=15pp`` is how PRD 8.6 writes it, so exactly 15pp of dial passes and 14pp does not."""
    result = endpoint(0.478, floor=MANO_FLOOR)
    assert gates.g1_dynamic_range(result, depth_scores={6: 0.60, 10: 0.45}).passed
    assert not gates.g1_dynamic_range(result, depth_scores={6: 0.59, 10: 0.45}).passed


def test_g1_refuses_a_dial_that_does_not_move_the_instrument():
    """
    An endpoint that ignores task difficulty cannot be read as responding to fact load either: a flat
    grid would then be a property of the instrument, which is PRD 1's whole complaint.
    """
    verdict = gates.g1_dynamic_range(
        endpoint(0.478, floor=MANO_FLOOR), depth_scores={6: 0.48, 10: 0.478, 13: 0.475}
    )
    assert not verdict.passed and "does not respond to task difficulty" in verdict.detail


def test_g1_does_not_require_the_dial_to_point_a_particular_way():
    """
    The gate reads the spread and records the direction without gating on it. PRD 8.5 deleted this
    programme's monotonicity rule for firing on 98.3% of rows by chance and pushing type-I error from
    5.0% to 16.7%; a gate that demanded a direction would be that rule under another name.
    """
    verdict = gates.g1_dynamic_range(
        endpoint(0.478, floor=MANO_FLOOR), depth_scores={6: 0.30, 10: 0.478, 13: 0.60}
    )
    assert verdict.passed
    assert evidence_of(verdict)["deepest_minus_shallowest_pp"] == pytest.approx(30.0, abs=0.01)


def test_g1_without_a_depth_sweep_reports_no_evidence_rather_than_passing():
    """
    The rule the section turns on. An in-band endpoint whose dial was never measured is not admitted
    for the half that was checked -- silently passing on absent evidence is the failure mode itself.
    """
    verdict = gates.g1_dynamic_range(endpoint(0.478, floor=MANO_FLOOR), depth_scores={10: 0.478})
    assert not verdict.passed
    assert verdict.detail.startswith(gates.NO_EVIDENCE)
    assert evidence_of(verdict)["position_pct"] == pytest.approx(45.3, abs=0.1)


def test_g1_refuses_a_task_whose_floor_answers_every_item():
    """A floor of 100% leaves no range to sit inside, and that is a broken task rather than a score."""
    verdict = gates.g1_dynamic_range(endpoint(1.0, floor=1.0), depth_scores={10: 1.0, 13: 0.194})
    assert not verdict.passed and "no range to sit inside" in verdict.detail


# --- G2: the label-permuted control ------------------------------------------------------------------


def test_g2_admits_a_random_init_model_that_lands_at_its_floor():
    verdict = gates.g2_label_permuted(endpoint(0.047, floor=MANO_FLOOR))
    assert verdict.passed and evidence_of(verdict)["above_floor_pp"] < 0.1


def test_g2_refuses_a_random_init_model_that_beats_its_own_floor():
    """
    PRD 8.3's `<compare>` bug, replayed with its own numbers. "Always name the first person" is a copy
    of the prompt, so it scores 50.2% where the best *constant* name scores 0.02% -- and an endpoint
    whose floor is quoted as the constant one has half the range its admission gate assumes.

    An untrained model landing 50pp above the quoted floor is how that gets found before a grid runs:
    the network cannot do the task, so the margin is the instrument.
    """
    verdict = gates.g2_label_permuted(endpoint(0.502, floor=0.0002, name="compare"))
    assert not verdict.passed
    assert "50.18pp above the measured floor" in verdict.detail


def test_g2_tolerance_narrows_as_the_eval_set_grows():
    """
    "At the floor" is a statistical claim, so the band is binomial rather than a fixed epsilon. The
    same 0.4pp excess is noise on 10,000 items and a finding on the 30,000 PRD 8.5 freezes -- and on a
    million it is not arguable.
    """
    assert gates.g2_label_permuted(endpoint(0.0504, floor=MANO_FLOOR, n=10_000)).passed
    assert not gates.g2_label_permuted(endpoint(0.0504, floor=MANO_FLOOR, n=1_000_000)).passed


def test_g2_without_a_random_init_checkpoint_reports_no_evidence_rather_than_passing():
    """
    The gate our design should pass trivially is exactly the one most tempting to skip. It is cheap --
    one forward pass over the eval set, no training -- so "not run" stays a refusal.
    """
    verdict = gates.g2_label_permuted(None)
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)


# --- G3: the premise-ablated probe -------------------------------------------------------------------


def test_g3_admits_an_ablation_that_destroys_the_score():
    verdict = gates.g3_premise_ablated(
        endpoint(0.478, floor=MANO_FLOOR), endpoint(0.047, floor=MANO_FLOOR)
    )
    assert verdict.passed and evidence_of(verdict)["drop_pp"] == pytest.approx(43.1, abs=0.1)


def test_g3_refuses_a_probe_that_still_answers_without_the_premise():
    """
    PRD 1's two-hop failure: composition scored 2.3x the product of its parts because it was reading
    fact access, not composing. An endpoint that survives having its premise deleted is measuring
    something, but not the thing on the label.
    """
    verdict = gates.g3_premise_ablated(
        endpoint(0.478, floor=MANO_FLOOR), endpoint(0.300, floor=MANO_FLOOR)
    )
    assert not verdict.passed and "still scores" in verdict.detail


def test_g3_refuses_an_ablation_that_only_dents_the_score():
    """
    Both halves are needed. Here the ablated arm does sit at its floor -- but so nearly does the
    intact one, so there was never a signal for a fact load to crowd out.
    """
    verdict = gates.g3_premise_ablated(
        endpoint(0.100, floor=MANO_FLOOR), endpoint(0.047, floor=MANO_FLOOR)
    )
    assert not verdict.passed and "costs only 5.3pp" in verdict.detail


def test_g3_compares_floor_corrected_scores_so_a_floor_shift_is_not_the_collapse():
    """
    Removing the premise changes the task, and the task's own degenerate baseline with it -- a shorter
    prompt has fewer spans to copy. Here raw accuracy falls only 10pp, which a naive difference would
    refuse, while the floor-corrected fall is the full 45pp and the ablated arm is exactly at its
    floor. Differencing raw accuracies would book the floor shift as the effect, in either direction.
    """
    verdict = gates.g3_premise_ablated(endpoint(0.50, floor=0.05), endpoint(0.40, floor=0.40))
    assert verdict.passed
    assert evidence_of(verdict)["drop_pp"] == pytest.approx(45.0, abs=0.01)


def test_g3_without_an_ablated_probe_reports_no_evidence_rather_than_passing():
    verdict = gates.g3_premise_ablated(endpoint(0.478, floor=MANO_FLOOR), None)
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)


# --- G4: headroom against the achievable ceiling -----------------------------------------------------


def test_g4_admits_a_cell_with_a_measured_ceiling_above_it():
    verdict = gates.g4_headroom(endpoint(0.478, floor=MANO_FLOOR), 0.52)
    assert verdict.passed


def test_g4_reports_the_range_that_exists_rather_than_the_oracle_one():
    """
    The whole point of the gate. ``EndpointResult.headroom`` says 95.4pp because it measures to a
    nominal 100; the b=0 arm says the endpoint tops out at 52%, so the range an effect can occupy is
    47.4pp. Both numbers are carried, and the smaller one is the honest one.
    """
    verdict = gates.g4_headroom(endpoint(0.478, floor=MANO_FLOOR), endpoint(0.52, floor=MANO_FLOOR))
    evidence = evidence_of(verdict)
    assert evidence["oracle_headroom_pp"] == pytest.approx(95.36)
    assert evidence["achievable_range_pp"] == pytest.approx(47.36)


def test_g4_refuses_a_cell_that_out_scores_the_arm_carrying_no_facts():
    """
    The check no other gate can make. A cell storing facts cannot beat the arm that stores none, so
    this says the ceiling is mis-measured -- most likely the two arms differ in reasoning-token
    exposure, which PRD 3.4 holds constant in absolute tokens precisely to keep them comparable.
    """
    verdict = gates.g4_headroom(endpoint(0.478, floor=MANO_FLOOR), 0.40)
    assert not verdict.passed and "mis-measured" in verdict.detail


def test_g4_refuses_a_control_arm_that_barely_beats_the_floor():
    """
    If the reasoning-only arm -- every parameter available, no facts to store -- reaches 8% on a task
    whose floor is 4.64%, the whole instrument is 3.4pp wide and the 2pp effect is most of it. That is
    a dead endpoint, and against a nominal 100 it would look like it had 95pp of room.
    """
    verdict = gates.g4_headroom(endpoint(0.07, floor=MANO_FLOOR), 0.08)
    assert not verdict.passed and "achievable range is 3.4pp" in verdict.detail


def test_g4_without_the_b0_arm_reports_no_evidence_rather_than_passing():
    verdict = gates.g4_headroom(endpoint(0.478, floor=MANO_FLOOR), None)
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)


# --- G6: capacity responsiveness ---------------------------------------------------------------------


def test_g6_admits_an_endpoint_that_moves_with_parameter_count():
    """PRD 8.3's ladder: +18.2pp across 13M-64M at fixed depth 12, which is what makes Mano usable."""
    verdict = gates.g6_capacity_responsive(
        {13_000_000: 0.400, 28_000_000: 0.478, 64_000_000: 0.582}
    )
    assert verdict.passed
    assert evidence_of(verdict)["rise_pp"] == pytest.approx(18.2, abs=0.05)


def test_g6_refuses_an_endpoint_flat_in_parameters():
    """
    "An endpoint flat in parameters cannot detect a capacity effect by construction" (PRD 8.6). It
    would return the same null whatever the fact load did, and PRD 8.4 had to withdraw the belief that
    reasoning is flat in width once this was measured rather than assumed.
    """
    verdict = gates.g6_capacity_responsive({13_000_000: 0.400, 64_000_000: 0.405})
    assert (
        not verdict.passed and "cannot detect a capacity effect by construction" in verdict.detail
    )


def test_g6_does_not_require_monotonicity():
    """
    The rise is read end to end and the shape between is ignored. Allen-Zhu's own single-seed grid
    violates parameter-order monotonicity by a median 27.1pp (PRD 8.6), and PRD 8.5 deleted this
    programme's monotonicity rule because it shrank the variance estimate to 0.57 sigma -- which makes
    the pre-registered equivalence test *falsely* declare equivalence. A monotonicity gate here would
    reinstate that by the back door.
    """
    verdict = gates.g6_capacity_responsive(
        {13_000_000: 0.400, 28_000_000: 0.330, 64_000_000: 0.582}
    )
    assert verdict.passed


def test_g6_refuses_an_endpoint_that_falls_with_parameter_count():
    """A negative rise is not a response, it is a broken ladder -- the direction is not free here."""
    verdict = gates.g6_capacity_responsive({13_000_000: 0.500, 64_000_000: 0.400})
    assert not verdict.passed and evidence_of(verdict)["rise_pp"] < 0


def test_g6_without_a_parameter_sweep_reports_no_evidence_rather_than_passing():
    verdict = gates.g6_capacity_responsive({13_000_000: 0.400})
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)


# --- G7: resolution ----------------------------------------------------------------------------------


def test_g7_admits_three_replicates_inside_the_resolution_limit():
    verdict = gates.g7_resolution([endpoint(0.478), endpoint(0.4785), endpoint(0.4790)])
    assert verdict.passed
    assert evidence_of(verdict)["sigma_pp"] == pytest.approx(0.05, abs=0.001)


def test_g7_publishes_sigma_and_the_mde_whether_it_passes_or_fails():
    """
    PRD 8.6 asks for both to be published, and PRD 8.5 keys the seed count to a measured sigma -- so
    the number is what re-scopes the design on a failure. Withholding it on the failing path would
    withhold it exactly when it is needed.
    """
    for replicates in (
        [endpoint(0.478), endpoint(0.4785), endpoint(0.4790)],
        [endpoint(0.470), endpoint(0.478), endpoint(0.490)],
    ):
        evidence = evidence_of(gates.g7_resolution(replicates))
        assert evidence["sigma_pp"] > 0 and evidence["mde_pp"] > 0


def test_g7_refuses_fewer_than_three_replicates():
    """
    ``k >= 3`` is PRD 8.6's. Two runs give a sigma on one degree of freedom, and PRD 8.5 shows what
    that costs: with m=4 a "SD <= 1pp" gate passes 27.9% of the time when the truth is 1.5pp.
    """
    verdict = gates.g7_resolution([endpoint(0.478), endpoint(0.4785)])
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)


def test_g7_refuses_a_sigma_that_cannot_resolve_the_effect():
    """
    The gate revision 1 did not have. The design was 8-50x short of the sigma it needed and the old
    gate, which only asked whether the endpoint responded, saw nothing wrong (PRD 8.6).
    """
    verdict = gates.g7_resolution([endpoint(0.50), endpoint(0.52), endpoint(0.54)])
    assert not verdict.passed
    assert evidence_of(verdict)["mde_pp"] > gates.TARGET_EFFECT_PP


def test_g7_sigma_limit_is_closed():
    """
    **The convention: sigma exactly at the limit passes**, which is how PRD 8.6 writes it (``<=``).

    Asserted by feeding the gate its own measured sigma as the threshold, because three replicates
    0.65pp apart do not produce exactly 0.65 in binary -- and a test that pretended otherwise would be
    testing float representation rather than the convention.
    """
    replicates = [endpoint(0.5000), endpoint(0.5065), endpoint(0.5130)]
    measured = gates.g7_resolution(replicates).value
    assert measured == pytest.approx(0.65, abs=1e-9)
    assert gates.g7_resolution(replicates, sigma_max_pp=measured).passed
    assert not gates.g7_resolution(replicates, sigma_max_pp=math.nextafter(measured, 0.0)).passed


def test_g7_unparseable_cap_is_closed_at_five_percent():
    """**The convention: exactly 5% passes**, again because PRD 8.6 writes ``<=``. 500 of 10,000."""
    tight = [endpoint(0.478), endpoint(0.4785), endpoint(0.4790, unparseable=0.05)]
    loose = [endpoint(0.478), endpoint(0.4785), endpoint(0.4790, unparseable=0.0501)]
    assert gates.g7_resolution(tight).passed
    assert not gates.g7_resolution(loose).passed


def test_g7_reads_the_worst_replicate_rather_than_the_pool():
    """
    One replicate whose scorer broke is a broken replicate. Pooled across three clean runs its 12%
    would average to 4% and pass, so the cap is applied to the worst run.
    """
    verdict = gates.g7_resolution(
        [endpoint(0.478), endpoint(0.4785), endpoint(0.4790, unparseable=0.12)]
    )
    assert not verdict.passed and "12.0% of items" in verdict.detail


def test_g7_cannot_check_unparseable_from_bare_accuracies_and_says_so():
    """
    Half a gate is not a pass. Bare floats carry a sigma but no unparseable count, so the verdict is a
    refusal naming what is missing -- with the sigma and MDE it *did* compute still published.
    """
    verdict = gates.g7_resolution([endpoint(0.478), 0.4785, endpoint(0.4790)])
    assert not verdict.passed and verdict.detail.startswith(gates.NO_EVIDENCE)
    assert evidence_of(verdict)["sigma_pp"] > 0


# --- G8: the calibrated positive control -------------------------------------------------------------


def test_g8_admits_a_ladder_that_brackets_the_dose_worth_two_points():
    """
    Calibrated means bracketed: under the target at the gentlest dose and over it at the strongest, so
    the dose worth 2pp can be named rather than merely bounded.
    """
    verdict = gates.g8_calibrated_positive_control(
        {100: 0.478, 95: 0.4775, 90: 0.4740, 80: 0.4630, 60: 0.4380}
    )
    assert verdict.passed
    assert 60 < evidence_of(verdict)["dose_pct_at_target"] < 90


def test_g8_refuses_a_ladder_that_never_reaches_the_target():
    """
    Dilution is a treatment known to hurt. If cutting 40% of the reasoning tokens does not move the
    endpoint 2pp, nothing will, and PRD 3.3 has the sharper worry: Physics 3.3's Result 11 says
    non-knowledge-dense competitors do not interfere, which would make a null predicted regardless of
    crowding. This gate is what separates those two readings.
    """
    verdict = gates.g8_calibrated_positive_control(
        {100: 0.478, 95: 0.4779, 90: 0.4778, 80: 0.4777, 60: 0.4776}
    )
    assert not verdict.passed and "A treatment known to hurt does not move it" in verdict.detail


def test_g8_refuses_a_ladder_too_coarse_to_name_the_dose():
    """
    The other way to fail a calibration: if the 95% arm already costs 5pp, the ladder steps straight
    over the target and the dose is unnamed. "Reasoning fell 2pp" is only interpretable once it can be
    converted into an amount of reasoning exposure.
    """
    verdict = gates.g8_calibrated_positive_control(
        {100: 0.50, 95: 0.45, 90: 0.44, 80: 0.43, 60: 0.40}
    )
    assert not verdict.passed and "brackets nothing" in verdict.detail


def test_g8_with_a_missing_dose_reports_no_evidence_rather_than_passing():
    """A partial ladder cannot be interpolated, and the refusal names which arm is still owed."""
    verdict = gates.g8_calibrated_positive_control({100: 0.478, 95: 0.4775, 90: 0.4740, 80: 0.4630})
    assert not verdict.passed
    assert verdict.detail.startswith(gates.NO_EVIDENCE) and "60%" in verdict.detail


# --- units -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: gates.g1_dynamic_range(endpoint(0.478), depth_scores={10: 47.8, 13: 19.4}),
        lambda: gates.g4_headroom(endpoint(0.478), 52.0),
        lambda: gates.g6_capacity_responsive({13_000_000: 40.0, 64_000_000: 58.2}),
        lambda: gates.g7_resolution([47.8, 47.85, 47.9]),
        lambda: gates.g8_calibrated_positive_control(
            {100: 47.8, 95: 47.75, 90: 47.4, 80: 46.3, 60: 43.8}
        ),
    ],
)
def test_an_accuracy_in_points_rather_than_as_a_fraction_raises(call):
    """
    PRD 8.3 quotes Mano as 47.8, ``EndpointResult.accuracy`` reports 0.478, and a gate handed the
    former would compare 4,780 against a 20% band and admit it. That is a unit error, not a finding,
    so it raises rather than becoming a verdict -- the one thing in this module that does.
    """
    with pytest.raises(OLMoConfigurationError, match=r"0.478, not 47.8"):
        call()


# --- the registry ------------------------------------------------------------------------------------


def test_there_is_no_g5():
    """
    PRD 8.6's table skips G5, and so does this. Renumbering would make a gate name in a log mean
    something different from the row it cites; inventing a G5 would send every future reader looking
    for a requirement that does not exist.
    """
    assert gates.GATES == ("G1", "G2", "G3", "G4", "G6", "G7", "G8")


def test_run_gates_returns_one_verdict_per_gate_in_the_prd_s_order():
    verdicts = gates.run_gates(**full_evidence())
    assert tuple(one.gate for one in verdicts) == gates.GATES


def test_run_gates_on_a_bare_result_refuses_every_gate_that_needs_evidence():
    """
    The shape of an early M0 report: seven refusals, each naming an arm still to run. Not one of them
    is a pass, which is the property this whole module exists to hold.
    """
    verdicts = gates.run_gates(endpoint(0.478, floor=MANO_FLOOR))
    assert not any(one.passed for one in verdicts)
    assert all(one.detail.startswith(gates.NO_EVIDENCE) for one in verdicts)


def test_a_fully_evidenced_endpoint_is_admitted():
    """The passing fixture for the table as a whole: every gate green, and `require_all` silent."""
    verdicts = gates.run_gates(**full_evidence())
    assert all(one.passed for one in verdicts), [one.detail for one in verdicts if not one.passed]
    gates.require_all(verdicts, endpoint="mano")  # returns None; raising is the failure mode


def test_require_all_names_every_failure_at_once():
    """
    Failures here are fixed by scheduling arms, so a caller that learns about one per run learns it
    six times. The message lists all of them, with each gate's own reason attached.
    """
    verdicts = gates.run_gates(
        **full_evidence(
            achievable_ceiling=None,
            scores_by_params={13_000_000: 0.400, 64_000_000: 0.405},
            dilution_scores={100: 0.478, 95: 0.4779, 90: 0.4778, 80: 0.4777, 60: 0.4776},
        )
    )
    with pytest.raises(OLMoConfigurationError) as caught:
        gates.require_all(verdicts, endpoint="mano")
    message = str(caught.value)
    assert "3 of 7 refused it" in message
    for gate in ("G4", "G6", "G8"):
        assert f"  {gate}: " in message
    for gate in ("G1", "G2", "G3", "G7"):
        assert f"  {gate}: " not in message


def test_require_all_treats_a_gate_that_never_ran_as_a_failure():
    """
    The last way to smuggle an endpoint through: run the four gates whose arms exist and hand those
    in. An unrun gate is not a passed one, so the missing names are listed like any other refusal.
    """
    verdicts = gates.run_gates(**full_evidence())
    with pytest.raises(OLMoConfigurationError) as caught:
        gates.require_all([one for one in verdicts if one.gate not in ("G7", "G8")])
    message = str(caught.value)
    assert "2 of 7 refused it" in message
    assert "G7: no evidence: the gate was never run" in message
    assert "G8: no evidence: the gate was never run" in message
