"""
Turning scored runs into a gate report, and the ways that could quietly go wrong.

The module exists because nothing produced a report. ``score_run`` consumed one and ``gates`` defined
them, so a finished grid arrived labelled ``confirmatory=False`` with hand-written JSON as the only
remedy -- which is asserting the gates passed, not measuring it.

Everything here is about *recognition*: which run is the ladder, which is the ceiling, which is the cell
under test. A wrong assignment produces a report that reads as evidence and is not, so each mistake the
assembler could make has a test that names it.
"""

from dataclasses import replace
from typing import Optional, Sequence

import pytest
from factcrowd import cells as C
from factcrowd.measure import evidence as E
from factcrowd.measure import gates
from factcrowd.measure.checkpoint import CheckpointRef
from factcrowd.measure.collect import ScoredCheckpoint
from factcrowd.measure.endpoints import EndpointResult

from olmo_core.exceptions import OLMoConfigurationError

CONFIG_ROOT = "src/scripts/train/factcrowd/configs/cells"
MEAN = 69.21875


def endpoint(accuracy: float, *, name: str = "mano", floor: float = 0.0459) -> EndpointResult:
    """An endpoint result at a given accuracy, on the 30,000-item frozen eval set."""
    return EndpointResult(
        name=name,
        n_total=30_000,
        n_correct=round(accuracy * 30_000),
        n_degenerate=0,
        n_unparseable=0,
        answer_ce_bits=1.0,
        floor=floor,
    )


def scored(
    cell: C.CellSpec,
    accuracy: Optional[float],
    *,
    step: int = 3_814,
    name: str = "mano",
    floor: Optional[float] = None,
) -> ScoredCheckpoint:
    """
    One scored checkpoint of a cell, carrying the resolved record the assembler reads.

    The floor follows the endpoint unless overridden, because passing mano's 4.59% for a ``ctxmano`` result
    makes G2 refuse a perfectly good untrained checkpoint -- a score is only "at the floor" relative to its
    own endpoint's floor, and the two differ by 6pp here.
    """
    resolved = cell.resolve()
    if floor is None:
        floor = 0.1045 if name == "ctxmano" else 0.0459
    results: Sequence[EndpointResult] = (
        () if accuracy is None else (endpoint(accuracy, name=name, floor=floor),)
    )
    return ScoredCheckpoint(
        ref=CheckpointRef(step=step, path=f"/tmp/{cell.qualified_id}/step{step}"),
        cell=cell.to_dict(),
        resolved=resolved.summary(MEAN),
        endpoints=results,
    )


def ladder(accuracies) -> list:
    """The committed 13M ladder, scored at the given accuracies in dose order."""
    arms = C.dilution_ladder_cells("13M")
    return [scored(cell, acc) for cell, acc in zip(arms, accuracies)]


# --- recognising the ladder ------------------------------------------------------------------------


def test_the_ladder_is_recognised_by_dose_and_keyed_by_percent_retained():
    """
    G8 keys its ladder by percent of reasoning tokens retained, so the assembler has to produce that.

    Keying by anything else -- token count, cell index, order encountered -- gives G8 a mapping whose
    keys never match ``DILUTION_DOSES_PCT``, and it reports a missing ladder while the ladder sits right
    there in the data.
    """
    assignment = E.assign_roles(ladder([0.50, 0.495, 0.49, 0.48, 0.46]), endpoint="mano")
    assert assignment.dilution == {100: 0.50, 95: 0.495, 90: 0.49, 80: 0.48, 60: 0.46}
    assert set(assignment.dilution) == set(gates.DILUTION_DOSES_PCT)


def test_only_the_last_checkpoint_of_a_run_reaches_a_gate():
    """
    Intermediate checkpoints are the trajectory, not the result.

    A cell saves ten of them. Letting an earlier one win would put a partially-trained model into a
    statement about what the design can resolve -- and since the ladder's arms have *different* step
    counts (fewer reasoning tokens is fewer steps), "the last one" cannot be a fixed step number.
    """
    arms = C.dilution_ladder_cells("13M")
    entries = [
        scored(arms[0], 0.10, step=100),
        scored(arms[0], 0.50, step=3_814),  # the real endpoint
        scored(arms[0], 0.30, step=1_000),
    ]
    assignment = E.assign_roles(entries, endpoint="mano")
    assert assignment.dilution == {100: 0.50}


def test_two_rows_of_ladder_are_refused_rather_than_interleaved():
    """
    Doses collide across rows: 13M's 80% arm and 28M's 80% arm are one key.

    Whichever was seen last would win, silently, and the resulting curve would be two widths' dose
    responses spliced together. G8 reads one curve.
    """
    mixed = ladder([0.50, 0.49, 0.48, 0.47, 0.46]) + [
        scored(cell, 0.60) for cell in C.dilution_ladder_cells("28M")
    ]
    with pytest.raises(OLMoConfigurationError, match="spans rows"):
        E.assign_roles(mixed, endpoint="mano")


def test_a_ladder_arm_with_no_score_for_the_endpoint_is_reported_not_dropped():
    """
    A silently short ladder is worse than a refused one: G8 would interpolate over a gap.

    G8 does refuse a partial ladder, but its message names the doses, not the reason they are absent.
    The note carries that, because "the 80% arm has no mano score" and "the 80% arm was never run" call
    for different actions.
    """
    arms = C.dilution_ladder_cells("13M")
    entries = [scored(cell, None if cell.cell_id.endswith("80") else 0.5) for cell in arms]
    assignment = E.assign_roles(entries, endpoint="mano")
    assert assignment.dilution is not None and 80 not in assignment.dilution
    assert any("13m_dil80" in note and "no 'mano' score" in note for note in assignment.notes)


# --- the controls ----------------------------------------------------------------------------------


def test_the_controls_supply_the_ceiling_and_the_width_sweep():
    """
    G4 wants the b=0 arm at the row under test; G6 wants accuracy against width at fixed depth.

    Both are the reasoning-only controls, which the committed grid already has one of per row -- so two
    of the seven gates were feedable from data the first run produces anyway, and were not being fed.
    """
    controls = [
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml"), 0.52),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/28m_ctrl.yaml"), 0.61),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/64m_ctrl.yaml"), 0.68),
    ]
    assignment = E.assign_roles(controls + ladder([0.5] * 5), endpoint="mano", row="13M")
    assert assignment.ceiling == 0.52  # the 13M control, because 13M is the row under test
    assert assignment.by_params is not None and len(assignment.by_params) == 3
    # Keyed by non-embedding parameters, ascending with width.
    keys = sorted(assignment.by_params)
    assert [assignment.by_params[k] for k in keys] == [0.52, 0.61, 0.68]


def test_sigma_comes_from_one_configurations_replicates_not_from_several_pooled():
    """
    G7 is a statement about run-to-run sigma at one configuration.

    Pooling two cells' replicates would mix the seed spread with the treatment difference and report the
    sum as sigma -- which runs in the permissive direction for G7 (a wider sigma is easier to pass
    a 'can we resolve this' gate against, and then the grid is under-powered in a way the gate blessed).
    """
    control = C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml")
    other = C.load_cell(f"{CONFIG_ROOT}/count/13m_d1p2.yaml")
    entries = [
        scored(replace(control, replicate=r), acc) for r, acc in enumerate((0.520, 0.526, 0.518))
    ] + [scored(replace(other, replicate=r), acc) for r, acc in enumerate((0.40, 0.47))]
    assignment = E.assign_roles(entries, endpoint="mano", row="13M")
    # The most-replicated cell wins, and the other is named rather than silently ignored.
    assert assignment.replicates is not None
    assert [r.accuracy for r in assignment.replicates] == [0.520, 0.526, 0.518]
    assert any("3 replicates of 13m_ctrl" in note for note in assignment.notes)
    assert any("13m_d1p2" in note and "not used" in note for note in assignment.notes)

    # Whole results, not bare accuracies. G7 also caps the unparseable rate of its *worst* replicate,
    # and a float carries no such count -- half the gate would have had nothing to fail on. Its docstring
    # says so explicitly and the first version of this assembler passed floats anyway.
    assert all(isinstance(r, EndpointResult) for r in assignment.replicates)
    verdict = gates.g7_resolution(assignment.replicates)
    assert "sigma" in verdict.detail.lower() or "σ" in verdict.detail

    # And a replicate whose scorer broke is caught rather than averaged away by three clean runs.
    broken = list(assignment.replicates)
    broken[1] = replace(broken[1], n_unparseable=int(0.4 * broken[1].n_total))
    assert not gates.g7_resolution(broken).passed


# --- the cell under test ---------------------------------------------------------------------------


def test_the_endpoint_under_test_is_the_rows_highest_demand_cell():
    """
    The gates ask whether the endpoint can resolve an effect where the design most needs it to.

    Reading the control instead would ask it about the easiest cell in the row, which is the one place a
    resolution gate is least informative.
    """
    row = [
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml"), 0.52),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_d0p3.yaml"), 0.50),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_d4p8.yaml"), 0.44),
    ]
    assignment = E.assign_roles(row, endpoint="mano", row="13M")
    assert assignment.result is not None
    assert assignment.result.accuracy == pytest.approx(0.44, abs=1e-4)
    assert any("13m_d4p8" in note for note in assignment.notes)


def test_a_gate_run_with_no_confirmatory_cell_falls_back_to_the_reference_arm():
    """
    During M0 nothing on the confirmatory grid has run yet.

    Refusing to report at all then would make the report unwritable exactly when it is most useful --
    the checklist of what is owed. The 100% arm is the same width and the same endpoint, so it stands in.

    The controls are present here on purpose. A control is not a cell under test -- it is the ceiling
    arm -- so with only controls and a ladder in the data the fallback must still fire. Selecting by
    highest demand alone would not catch this, because every cell here has demand 0.0 and one of the
    controls would win by tie-break and be reported as the cell under test.
    """
    entries = ladder([0.50, 0.49, 0.48, 0.47, 0.46]) + [
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml"), 0.52),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/28m_ctrl.yaml"), 0.61),
    ]
    assignment = E.assign_roles(entries, endpoint="mano")
    assert assignment.result is not None
    assert assignment.result.accuracy == pytest.approx(0.50, abs=1e-4)
    assert any("100% arm" in note for note in assignment.notes)
    assert not any("13m_ctrl" in note and "under test" in note for note in assignment.notes)


def test_a_ladder_arm_is_never_mistaken_for_the_cell_under_test():
    """
    A ladder arm is a control with a reduced dose, not a treatment cell.

    Both have ``demand_bits_per_param`` 0.0 on the default ladder, so an assembler that picked the
    highest-demand cell without excluding ladder arms would pick one of them and then run the gates on
    a cell the ladder is meant to be calibrating.
    """
    entries = ladder([0.50, 0.49, 0.48, 0.47, 0.46]) + [
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_d1p2.yaml"), 0.41)
    ]
    assignment = E.assign_roles(entries, endpoint="mano", row="13M")
    assert assignment.result is not None
    assert assignment.result.accuracy == pytest.approx(0.41, abs=1e-4)


# --- the report -----------------------------------------------------------------------------------


def test_the_assembled_report_passes_nothing_it_has_no_evidence_for():
    """
    The property that makes an early report safe to write.

    Four gates are feedable from configs that exist; G1, G2 and G3 need corpus and task variants that are
    not built. They must come back refused, and the report must not pass -- an empty or partial report
    that read as passing would admit the whole grid on the strength of a file.
    """
    entries = ladder([0.50, 0.499, 0.494, 0.485, 0.472]) + [
        scored(C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml"), 0.52),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/28m_ctrl.yaml"), 0.61),
        scored(C.load_cell(f"{CONFIG_ROOT}/count/64m_ctrl.yaml"), 0.68),
    ]
    report, assignment = E.assemble(entries, endpoint="mano", row="13M", commit="deadbee")
    assert report.version == gates.GATE_REPORT_VERSION
    assert report.endpoint == "mano" and report.commit == "deadbee"
    assert len(report.results) == len(gates.GATES)
    # G8 had a well-formed ladder, so it is not among the refusals.
    assert "G8" not in report.failures
    # The three unbuilt gates are.
    assert {"G1", "G2", "G3"} <= set(report.failures)
    assert not report.passed
    assert assignment.dilution is not None and assignment.ceiling == 0.52


def test_a_report_about_an_endpoint_nothing_measured_is_refused():
    """
    An empty report is a file that admits rows. It has to be impossible to write one by accident.
    """
    entries = ladder([0.5] * 5)
    with pytest.raises(OLMoConfigurationError, match="no scored checkpoint carries endpoint"):
        E.assemble(entries, endpoint="brevo1")


def test_the_pattern_matches_exactly_what_the_generator_writes():
    """
    Recognition is by cell id, so the pattern and the generator cannot be allowed to drift.
    """
    for cell in C.dilution_ladder_cells("13M"):
        match = E.DILUTION_CELL_PATTERN.match(cell.cell_id)
        assert match is not None, cell.cell_id
        assert match.group("row") == "13m"
        assert int(match.group("dose")) in gates.DILUTION_DOSES_PCT
    # And it does not sweep up the rest of the grid.
    for name in ("13m_ctrl", "13m_d4p8", "28m_b8", "13m_dilution", "13m_dil"):
        assert E.DILUTION_CELL_PATTERN.match(name) is None


def test_the_width_sweep_and_the_ceiling_average_over_their_replicates():
    """
    G6 asks whether accuracy rises with width; one seed per width can invert that.

    The sigma block runs three replicates of exactly these control cells, so the mean is free -- and
    reading `r0` alone was letting seed noise decide the ordering of a gate whose whole content is an
    ordering. Constructed here so the r0 values run *backwards* while the means run forwards.
    """
    thirteen = C.load_cell(f"{CONFIG_ROOT}/count/13m_ctrl.yaml")
    twenty_eight = C.load_cell(f"{CONFIG_ROOT}/count/28m_ctrl.yaml")
    entries = [
        # 13M's r0 is high and 28M's r0 is low, so a single-seed read sees accuracy *falling* with width.
        scored(replace(thirteen, replicate=0), 0.60),
        scored(replace(thirteen, replicate=1), 0.40),
        scored(replace(thirteen, replicate=2), 0.41),  # mean 0.47
        scored(replace(twenty_eight, replicate=0), 0.50),
        scored(replace(twenty_eight, replicate=1), 0.62),
        scored(replace(twenty_eight, replicate=2), 0.62),  # mean 0.58
    ]
    assignment = E.assign_roles(entries, endpoint="mano", row="13M")
    assert assignment.by_params is not None
    keys = sorted(assignment.by_params)
    means = [assignment.by_params[k] for k in keys]
    assert means == [pytest.approx(0.47, abs=1e-9), pytest.approx(0.58, abs=1e-9)]
    assert means[0] < means[1]  # rises with width, which r0 alone would have denied

    # G4's ceiling is the row's control, averaged the same way rather than taken from one seed.
    assert assignment.ceiling == pytest.approx(0.47, abs=1e-9)
    assert any("3 replicate(s) each" in note for note in assignment.notes)
    assert any("mean of 3" in note for note in assignment.notes)


# --- G1 and G2: the two gates that had evidence all along and no way to reach it ------------------


def calibration(variant: str, accuracies) -> list:
    """The committed 28M calibration sweep for one variant, scored at the given accuracies."""
    arms = C.mano_calibration_cells(("28M",), variant=variant, architecture="entropy")
    name = "ctxmano" if variant == "in_context" else "mano"
    return [scored(cell, acc, name=name) for cell, acc in zip(arms, accuracies)]


def test_the_calibration_sweep_supplies_g1s_depth_curve():
    """
    **G1 was unreachable for the silliest of reasons.** ``run_gates`` has taken ``depth_scores`` since it
    was written and nothing ever supplied it, so a fully scored calibration sweep would have left the gate
    reporting "no evidence" however many arms sat in the CSV.

    Keyed by depth, and the depth comes off the cell id, so the ids one function generates and the pattern
    that reads them cannot drift without a test failing.
    """
    rows = calibration("in_context", (0.62, 0.48, 0.35, 0.24))
    assignment = E.assign_roles(rows, endpoint="ctxmano")
    assert assignment.depths == {2: 0.62, 3: 0.48, 4: 0.35, 5: 0.24}
    assert any("G1: depth sweep" in note for note in assignment.notes)

    memorised = E.assign_roles(calibration("memorised", (0.3,) * 7), endpoint="mano")
    assert sorted(memorised.depths or {}) == list(C.MANO_CALIBRATION_LENGTHS)


def test_the_two_variants_depth_curves_are_never_mixed():
    """
    They are different instruments -- a 10.45% floor against 4.35%, 256-token items against 24 -- so one
    curve built from both would be two curves averaged, and G1's spread would be measuring the difference
    between the tasks rather than between depths.
    """
    both = calibration("in_context", (0.62, 0.48, 0.35, 0.24)) + calibration(
        "memorised", (0.30, 0.22, 0.14, 0.09, 0.07, 0.06, 0.05)
    )
    assert E.assign_roles(both, endpoint="ctxmano").depths == {2: 0.62, 3: 0.48, 4: 0.35, 5: 0.24}
    assert sorted(E.assign_roles(both, endpoint="mano").depths or {}) == list(
        C.MANO_CALIBRATION_LENGTHS
    )
    # An endpoint with no depth sweep of its own gets none, rather than borrowing another's.
    assert E.assign_roles(both, endpoint="compare").depths is None
    assert E.assign_roles(both, endpoint="mano_table").depths is None


def test_a_depth_sweep_spanning_two_rows_is_refused():
    """Two widths interleaved by depth would let a wider row's success stand in for the treatment's."""
    spanning = calibration("in_context", (0.62, 0.48, 0.35, 0.24)) + [
        scored(cell, 0.9, name="ctxmano")
        for cell in C.mano_calibration_cells(
            ("113M",), variant="in_context", architecture="entropy"
        )
    ]
    with pytest.raises(OLMoConfigurationError, match="depth sweep spans rows"):
        E.assign_roles(spanning, endpoint="ctxmano")


def test_a_calibration_arm_is_evidence_and_never_the_cell_under_test():
    """It carries no facts, so admitting on one would admit a demand-0 run as though it were a treatment."""
    rows = calibration("in_context", (0.62, 0.48, 0.35, 0.24))
    assert E.assign_roles(rows, endpoint="ctxmano").result is None


def test_the_step_zero_checkpoint_supplies_g2():
    """
    **G2 asks for an untrained model, and every run has written one at step 0 all along.**
    ``CheckpointerCallback.pre_train_checkpoint`` writes it. The reason it never reached the gate is that
    the assembler reads the *last* checkpoint per cell, which discards step 0 wherever a run trained.
    """
    cell = C.dilution_ladder_cells("13M")[0]
    untrained = scored(cell, 0.0461, step=0)  # a hair off the 4.59% floor, as a random model is
    trained = scored(cell, 0.35, step=3_814)

    assignment = E.assign_roles([untrained, trained], endpoint="mano")
    assert assignment.random_init is not None
    assert assignment.random_init.accuracy == pytest.approx(0.0461, abs=1e-4)
    # And the trained one is still what the other gates read.
    assert assignment.dilution == {100: pytest.approx(0.35, abs=1e-4)}
    assert any("G2: untrained checkpoint" in note for note in assignment.notes)

    without = E.assign_roles([trained], endpoint="mano")
    assert without.random_init is None
    assert any("G2: no step-0 checkpoint" in note for note in without.notes)


def test_g1_and_g2_reach_the_report_rather_than_stopping_at_the_assignment():
    """
    The assignment is not the deliverable; the report is. A field populated here and dropped on the way to
    ``run_gates`` would look correct in every test above and change nothing about admission.
    """
    rows = calibration("in_context", (0.62, 0.48, 0.35, 0.24))
    treatment = C.CellSpec(
        cell_id="28m_b8",
        row="28M",
        sweep="entropy",
        bits_per_attribute=8,
        n_entities=100_000,
        reasoning_tokens=C.REASONING_TOKENS,
        mano_variant="in_context",
        ctxmano_length=4,
    )
    rows += [
        # 10.5% is the in-context floor, which is where an untrained network lands.
        scored(treatment, 0.105, step=0, name="ctxmano"),
        scored(treatment, 0.55, name="ctxmano"),
    ]

    report, assignment = E.assemble(rows, endpoint="ctxmano", commit="cafe")
    verdicts = {one.gate: one for one in report.results}
    assert tuple(verdicts) == gates.GATES
    # Both now refuse or pass ON THE MERITS -- the string that must not appear is "no evidence".
    assert "no evidence" not in verdicts["G1"].detail, verdicts["G1"].detail
    assert "no evidence" not in verdicts["G2"].detail, verdicts["G2"].detail
    assert verdicts["G1"].passed, verdicts["G1"].detail  # 55% is in band with a 15pp+ spread
    assert verdicts["G2"].passed, verdicts["G2"].detail  # 10.5% is its floor
    # G3 is the one that genuinely has no evidence, and it must still say so.
    assert "no evidence" in verdicts["G3"].detail
    assert assignment.depths and assignment.random_init is not None
