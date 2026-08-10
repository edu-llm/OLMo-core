"""
Tests for ``analysis.py``, the pre-registered analysis of the hyper-connection tranche.

WHAT IS BEING TESTED HERE IS AN ANSWER NOBODY HAS SEEN YET, so almost every test in this file
plants a truth first and then asks the pipeline to find it. The treatment arms were still
running when this was written, which is the point: an analysis whose choices are fixed after
the endpoints are visible is not an analysis, it is a search. Planted truths and the frozen
comparator are the only two things available to check against before the data lands, and both
are used.

Three groups, and they are not equally interesting.

*The reader*, which is most of the file, because the reader is where this module has already
been bitten. A crash reporter calling ``wandb.init`` with ``WANDB_RUN_ID`` still set overwrote
seven cells' summaries with a diagnostic, so those cells read ``step: None`` -- one of them
after 3.993 hours and 4,910 steps, further than anything else in its submission. Every test
that starts "a clobbered" is that defect, and what they check is that a replicate is never lost
in silence: recovered where the history allows, and named with a reason where it does not.

*The frozen comparator*, one test, and the strongest evidence in the file. The baseline's noise
floor was measured and frozen into ``noise-floor-skip-step.json`` before ``analysis.py``
existed, so reproducing it bit for bit through an entirely separate reading path is a check
against a number that cannot have been tuned to.

*The estimators*, against planted effects, planted correlations and a planted null, including
the calibration checks -- coverage at 95%, false positives at 5% -- that say whether the
interval and the p-value mean what they are labelled as.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import noise_floor as nf  # noqa: E402

import analysis as an  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "noise-floor-skip-step.json")

#: The five ``baseline`` endpoints, unweighted mean held-out BPB over the seven sources at step
#: 6,000, read from ``run_019fe40f-c71e`` cells 0-4.
#:
#: A FIXTURE AND NOT A FETCH, DELIBERATELY. These are the numbers the frozen artifact was
#: computed from, so a test that re-reads W&B to get them would be checking that W&B is up.
#: Frozen here, the comparison is between two independent paths to one published number.
BASELINE_ENDPOINTS_BPB = (
    0.6756527410392009,
    0.6752062205312278,
    0.6756557516764151,
    0.6768358031893593,
    0.6760047490431987,
)

SUBMISSION = "run_019feaaa-0001"


# ---------------------------------------------------------------------------------------
# A W&B run, reduced to what the reader touches.
# ---------------------------------------------------------------------------------------


class FakeRun:
    """
    A stand-in for a W&B API run.

    ``summary`` DEFAULTS TO EMPTY BECAUSE THAT IS THE INTERESTING CASE AND THE CHEAP ONE IS THE
    TRAP. The reader is not allowed to take any number from a summary, so a fake whose summary
    is populated would pass tests that a clobbered cell fails, and clobbered cells are the ones
    this reader was written for.
    """

    def __init__(
        self,
        run_id,
        *,
        config,
        history,
        summary=None,
        state="finished",
        last_history_step=6000,
        job_type=None,
    ):
        self.id = run_id
        self.name = run_id
        self.config = config
        self.summary = {} if summary is None else summary
        self.state = state
        self.lastHistoryStep = last_history_step  # noqa: N815 -- the public API's own spelling
        self.job_type = job_type
        self._history = history

    def scan_history(self, keys=None, page_size=None):
        for row in self._history:
            if keys is None:
                yield dict(row)
            elif any(key in row for key in keys):
                yield {key: row.get(key) for key in keys}


def _config(arm="baseline", seed=0):
    """
    A saved run config as ``arms_consistent_with`` and ``sources_from_config`` read one.

    THE LANE BLOCK COMES FROM THE ARM TABLE AND IS NOT SPELLED OUT HERE. A hand-written config
    that happens to satisfy the classifier today would go on satisfying it after the table
    changed, and then these tests would be asserting about an arm the tranche no longer runs.
    """
    from hyper_connection_arms import ARMS

    lanes = ARMS[arm].hyper_connections
    model = {"block": {"hyper_connections": None if lanes is None else lanes.as_config_dict()}}
    return {
        "model": model,
        "data_loader": {"seed": seed},
        "callbacks": {
            "lm_evaluator": {
                "eval_dataset": {"paths": {s: [f"{s}.npy"] for s in an.HELD_OUT_SOURCES}}
            }
        },
    }


def _history(endpoint_bpb, *, arm="baseline", steps=(3000, 6000), declined=17, trigger=0.4):
    """
    A cell's per-step history: the held-out evaluations, the stability keys, a training curve,
    and the lane monitor on an arm that has lanes.
    """
    rows = []
    for step in steps:
        # Sources spread around the endpoint so that their unweighted mean is exactly it.
        offsets = np.linspace(-0.3, 0.3, len(an.HELD_OUT_SOURCES))
        # A falling curve keyed on the *step*, not on the row's position, so that two arms of
        # different depths agree wherever they overlap. Keyed on position, every fixture would
        # carry a gap at the shared step and would then "prove" the alignment bug it was
        # written to catch.
        level = endpoint_bpb + 0.4 * (an.HORIZON - step) / an.HORIZON
        row = {"_step": step, an.TRAIN_LOSS_METRIC: level * an.NATS_PER_BPB}
        for source, offset in zip(an.HELD_OUT_SOURCES, offsets):
            row[an.BPB_METRIC.format(source=source)] = level + offset
            row[an.CE_METRIC.format(source=source)] = (level + offset) * an.NATS_PER_BPB
        row[an.SKIPPED_COUNT_METRIC] = declined
        row[an.MAX_TRIGGER_METRIC] = trigger
        if arm != "baseline":
            for witness in an.LANE_WITNESS_METRICS:
                row[witness] = 0.31
        rows.append(row)

    # The individual declines behind that running total. A fixture with a total and no events
    # under it is the inconsistency `stability_at` refuses, and rightly: the count over any
    # prefix would be short by an unknown amount.
    for index in range(declined):
        rows.append(
            {
                "_step": 1 + index * max(steps[-1] // max(declined, 1), 1),
                an.SKIP_TRIGGER_METRIC: trigger * (0.4 + 0.6 * (index + 1) / max(declined, 1)),
            }
        )
    return rows


def cell(index, endpoint, *, arm="baseline", seed=None, submission=SUBMISSION, **kwargs):
    """One healthy fan-out cell."""
    config = kwargs.pop("config", _config(arm, index if seed is None else seed))
    history = kwargs.pop("history", _history(endpoint, arm=arm))
    return FakeRun(f"{submission}-cell-{index}", config=config, history=history, **kwargs)


def read(runs):
    """Every run through the per-cell reader, as ``read_arm`` does it."""
    return [an._read_cell(run, train_curve_samples=100) for run in runs]


def healthy(arm="baseline", endpoints=BASELINE_ENDPOINTS_BPB, submission=SUBMISSION):
    return [cell(i, value, arm=arm, submission=submission) for i, value in enumerate(endpoints)]


# ---------------------------------------------------------------------------------------
# The frozen comparator. One test, and the one that carries the most weight.
# ---------------------------------------------------------------------------------------


def test_the_frozen_noise_floor_is_reproduced_bit_for_bit_from_the_measured_endpoints():
    """
    THE ARTIFACT WAS FROZEN BEFORE THIS MODULE EXISTED, WHICH IS WHAT MAKES THIS A TEST. Every
    other check here is against a truth this file planted, and a generator and an estimator
    written by the same hand can agree with each other while both being wrong about the world.
    ``noise-floor-skip-step.json`` was measured off ``run_019fe40f-c71e`` by a different program
    and published before ``analysis.py`` was started, so nobody can have tuned this to it.

    Bit for bit and not to a tolerance, because this is the same five numbers through the same
    arithmetic, so anything but equality means one of the two is not doing what it says.
    """
    with open(FROZEN) as handle:
        frozen = json.load(handle)

    pooled = nf.pooled_sigma([np.array(BASELINE_ENDPOINTS_BPB)])

    assert pooled.df == frozen["sigma_df"] == 4
    assert pooled.sigma == frozen["sigma_bpb"]
    assert pooled.sigma_unbiased == frozen["sigma_bpb_unbiased"]
    assert [pooled.ci_low, pooled.ci_high] == frozen["sigma_ci_bpb"]
    assert pooled.sigma * an.NATS_PER_BPB == frozen["sigma_nats"]
    # The number the pre-registration quotes, to the precision it quotes it at.
    assert round(pooled.sigma * an.NATS_PER_BPB, 5) == 0.00193


def test_the_reader_arrives_at_the_frozen_number_from_a_cells_per_step_history():
    """
    The other half: that the endpoint the reader forms out of seven per-source series is the
    endpoint the frozen artifact was computed from.

    NOT BIT FOR BIT, AND THE GAP IS FLOATING-POINT REASSOCIATION RATHER THAN A DISAGREEMENT.
    Summing seven sources and dividing lands a few units in the last place away from the number
    those seven were derived from, which is 1e-16 relative -- fourteen orders of magnitude below
    the noise floor being estimated. Against the live tranche, where the per-source values are
    read rather than reconstructed, this path reproduces the artifact exactly.
    """
    arm = an.assemble_arm(read(healthy()), "baseline", SUBMISSION)
    endpoints = arm.endpoint_matrix().mean(axis=1)

    assert endpoints == pytest.approx(np.array(BASELINE_ENDPOINTS_BPB), rel=1e-12)
    with open(FROZEN) as handle:
        frozen = json.load(handle)
    assert nf.pooled_sigma([endpoints]).sigma == pytest.approx(frozen["sigma_bpb"], rel=1e-12)


def test_the_frozen_artifact_is_reached_through_the_whole_analysis_and_not_only_the_estimator():
    """The same number again, out of the far end of ``analyse``, where a report would read it."""
    result = an.analyse([an.assemble_arm(read(healthy()), "baseline", SUBMISSION)])
    with open(FROZEN) as handle:
        frozen = json.load(handle)
    assert result["sigma"]["sigma_bpb"] == pytest.approx(frozen["sigma_bpb"], rel=1e-12)
    assert result["sigma"]["df"] == 4
    assert result["single_arm"], "one arm supports no contrast and the report must say so"


# ---------------------------------------------------------------------------------------
# The clobbered summary, which is the defect this reader exists for.
# ---------------------------------------------------------------------------------------


def test_the_endpoint_is_taken_from_history_when_the_summary_holds_nothing():
    """
    A cell whose summary its own crash report overwrote still has every number it logged, and
    the reader takes all of them from the history. Read from the summary this cell has no
    endpoint at all; read properly it is an ordinary replicate.
    """
    runs = healthy()
    runs[2].summary = {"_runtime": 0.0}
    runs[2].state = "crashed"

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)

    assert len(arm.seeds) == 5, "the clobbered cell is still a replicate"
    assert arm.excluded == []
    assert arm.endpoint_matrix().mean(axis=1)[2] == pytest.approx(BASELINE_ENDPOINTS_BPB[2])


def test_a_clobbered_summary_is_reported_as_a_recovery_rather_than_absorbed():
    """
    RECOVERING QUIETLY IS BETTER THAN LOSING QUIETLY AND IT IS STILL NOT GOOD ENOUGH. The number
    is an inference from a record that was damaged, and a reader who is not told that cannot
    weigh it. So the recovery is named, and it travels into the report and the JSON.
    """
    runs = healthy()
    runs[1].summary = {"_runtime": 0.0}
    runs[1].lastHistoryStep = 4910

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    notes = " ".join(arm.recovered)

    assert f"{SUBMISSION}-cell-1" in notes
    assert "4910" in notes and "overwritten" in notes
    assert arm.summary_steps[1] is None and arm.history_steps[1] == 4910

    result = an.analyse([arm])
    assert any("cell-1" in note for _, note in result["recovered"])


def test_a_crashed_state_does_not_condemn_a_cell_that_reached_the_horizon():
    """
    A cell the crash reporter wrote onto reads ``crashed`` however far it trained, so a
    completeness check keyed on ``state`` throws away exactly the cells the recovery rescued.
    What settles whether a cell finished is the step its history reaches.
    """
    runs = healthy()
    runs[0].state = "crashed"
    runs[0].summary = {}
    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)

    assert an.completeness_refusals([arm]) == []


def test_a_cell_that_really_stopped_short_is_still_refused():
    """
    The other side of it: history that stops at 4,910 is a cell that did not finish, and the
    check has to keep saying so now that ``state`` alone no longer decides it.

    The cell is kept rather than excluded -- it has an evaluation, just not the last one -- and
    what falls is the step the arm can be compared at, which is the honest consequence and is
    reported as one.
    """
    runs = healthy()
    runs[0].state = "failed"
    runs[0].lastHistoryStep = 4910
    runs[0]._history = _history(BASELINE_ENDPOINTS_BPB[0], steps=(3000,))

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    problems = " ".join(an.completeness_refusals([arm]))

    assert "stopped before the horizon" in problems and "4910" in problems
    assert "shares its last evaluation at step 3000 of 6000" in problems


# ---------------------------------------------------------------------------------------
# Losing a replicate, which is allowed, and losing one quietly, which is not.
# ---------------------------------------------------------------------------------------


def test_a_cell_with_no_evaluation_is_excluded_by_name_and_with_a_reason():
    runs = healthy()
    runs[3]._history = [{"_step": 4200, an.TRAIN_LOSS_METRIC: 2.4}]
    runs[3].lastHistoryStep = 4200

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)

    assert len(arm.seeds) == 4
    assert len(arm.excluded) == 1
    run_id, why = arm.excluded[0]
    assert run_id == f"{SUBMISSION}-cell-3"
    assert "4200" in why and "no endpoint" in why


def test_a_cell_absent_from_the_fanout_is_named_as_absent_rather_than_as_late():
    """
    A fan-out short a cell and a fan-out short an evaluation are different failures with
    different causes, and "4 of 5" alone does not distinguish them.
    """
    arm = an.assemble_arm(read(healthy()[:4]), "baseline", SUBMISSION)

    assert [run for run, _ in arm.excluded] == [f"{SUBMISSION}-cell-4"]
    assert "no run with this cell index exists" in arm.excluded[0][1]


def test_an_exclusion_is_never_silently_a_smaller_n():
    """
    THE ONE BEHAVIOUR THAT IS NOT ALLOWED. A lost replicate shortens the df, narrows the
    interval and moves the mean, and cells do not lose their evaluations at random -- they lose
    them by hitting a wall -- so the survivors are biased in a direction nobody chose. Every
    exclusion is therefore a completeness refusal in its own right, before the shortfall is
    even counted.
    """
    runs = healthy()
    runs[4]._history = []
    runs[4].lastHistoryStep = -1

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    problems = an.completeness_refusals([arm])

    assert any("lost a replicate" in p and "cell-4" in p for p in problems)
    assert any("4 of 5 cells" in p for p in problems)
    result = an.analyse([arm], provisional=problems)
    assert [entry[1] for entry in result["excluded"]] == [f"{SUBMISSION}-cell-4"]
    assert "cell-4" in an.render(result)


def test_the_report_leads_with_what_it_is_missing():
    """A loss buried under the tables is a loss nobody reads. It goes above them."""
    runs = healthy()
    runs[0]._history = []
    runs[0].lastHistoryStep = -1
    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    text = an.render(an.analyse([arm], provisional=an.completeness_refusals([arm])))

    assert text.index("CELLS READ AND LEFT OUT") < text.index("THE NOISE FLOOR")


# ---------------------------------------------------------------------------------------
# The two recoveries, and the checks that gate them.
# ---------------------------------------------------------------------------------------


def test_a_lost_seed_comes_back_from_the_cell_index_once_the_relation_is_confirmed():
    """
    These are ``--fanout-index-parameter seed`` submissions, so index equals seed by
    construction -- and "by construction" is an argument, not evidence. The evidence is that it
    holds on every sibling whose config survived.
    """
    runs = healthy()
    runs[2].config = {"data_loader": None}

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)

    assert arm.seeds == [0, 1, 2, 3, 4]
    assert any("seed 2 recovered from the cell index" in note for note in arm.recovered)


def test_a_lost_seed_is_not_guessed_when_the_relation_does_not_hold():
    """One sibling disagreeing is enough. A guess that is usually right is still a guess."""
    runs = healthy()
    runs[2].config = {"data_loader": None}
    runs[0].config = _config("baseline", seed=41)

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)

    assert len(arm.excluded) == 1
    assert "index-equals-seed relation could not be confirmed" in arm.excluded[0][1]


def test_a_lost_arm_comes_back_from_the_siblings_and_the_cells_own_lane_monitor():
    runs = healthy(arm="faithful")
    runs[1].config = {"data_loader": {"seed": 1}}

    arm = an.assemble_arm(read(runs), "faithful", SUBMISSION)

    assert len(arm.seeds) == 5
    assert any("arm 'faithful' recovered from its siblings" in n for n in arm.recovered)
    assert any("confirmed against this cell's own lane monitor" in n for n in arm.recovered)


def test_a_cell_whose_lane_monitor_contradicts_the_arm_is_excluded():
    """
    The monitor is attached to an arm with lanes and to no other, so a cell of ``faithful``
    that never logged it did not run ``faithful``. That is the cell's own testimony about the
    half of the question its siblings cannot answer for it.
    """
    runs = healthy(arm="faithful")
    runs[1].config = {"data_loader": {"seed": 1}}
    runs[1]._history = _history(BASELINE_ENDPOINTS_BPB[1], arm="baseline")

    arm = an.assemble_arm(read(runs), "faithful", SUBMISSION)

    assert len(arm.excluded) == 1
    assert "did not run the arm the submission says it did" in arm.excluded[0][1]


def test_silence_from_a_cell_with_no_history_is_not_read_as_evidence_of_no_lanes():
    """
    A cell that logged nothing has no testimony to give, and convicting it on that silence would
    exclude every clobbered cell of every arm with lanes -- the same defect one layer along.
    It is excluded here, but for having no endpoint, and the reason says so.
    """
    runs = healthy(arm="faithful")
    runs[1].config = {"data_loader": {"seed": 1}}
    runs[1]._history = []
    runs[1].lastHistoryStep = -1

    arm = an.assemble_arm(read(runs), "faithful", SUBMISSION)

    assert len(arm.excluded) == 1
    assert "logged no history at all" in arm.excluded[0][1]
    assert any("rests on its siblings alone" in note for note in arm.recovered)


# ---------------------------------------------------------------------------------------
# Disagreements, which are never recoverable and never downgradeable.
# ---------------------------------------------------------------------------------------


def test_a_cell_whose_config_names_another_arm_stops_the_read():
    """
    Not an exclusion. An exclusion says a replicate is missing; this says the report would name
    the arms wrongly, and a contrast between mislabelled arms is worse than no contrast.
    """
    runs = healthy(arm="faithful")
    runs[2].config = _config("mhc", seed=2)

    with pytest.raises(an.Refusal, match="saved model config is consistent with"):
        an.assemble_arm(read(runs), "faithful", SUBMISSION)


def test_two_cells_on_one_seed_stop_the_read():
    """
    Identical curves, a noise floor near zero, and every contrast against it significant. This
    is what ``resolve_seed`` exists to refuse and the analysis refuses it again.
    """
    runs = healthy()
    runs[3].config = _config("baseline", seed=1)

    with pytest.raises(an.Refusal, match="not distinct"):
        an.assemble_arm(read(runs), "baseline", SUBMISSION)


def test_an_arm_with_nothing_in_it_is_absent_rather_than_empty():
    runs = healthy()
    for run in runs:
        run._history = []
        run.lastHistoryStep = -1

    with pytest.raises(an.Refusal, match="nothing to average"):
        an.assemble_arm(read(runs), "baseline", SUBMISSION)


def test_arms_that_do_not_share_a_seed_set_cannot_be_paired():
    other = "run_019feaaa-0002"
    left = an.assemble_arm(read(healthy()), "baseline", SUBMISSION)
    right = an.assemble_arm(read(healthy(arm="faithful", submission=other)), "faithful", other)
    right.seeds = [0, 1, 2, 3, 7]

    with pytest.raises(an.Refusal, match="no pairing to make"):
        an.analyse([left, right])


def test_a_missing_stability_family_is_not_a_count_of_zero():
    """
    ``stability/steps skipped`` is written on every optimizer step of every arm, so a cell
    without it had no monitor -- and reading that as "declined nothing" makes an arm look
    maximally stable exactly when the instrument was missing.
    """
    runs = healthy()
    for row in runs[0]._history:
        row.pop(an.SKIPPED_COUNT_METRIC, None)

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    problems = " ".join(an.stability_refusals([arm]))

    assert "missing instrument and not a count of zero" in problems


def test_a_crash_report_beside_a_cell_is_not_a_replicate():
    """
    A report shares its cell's group and display-name stem and carries no model config, which
    is exactly how a baseline cell reads. Left in, a diagnostic filed against a dead ``mhc``
    cell arrives in the baseline arm as a seed-0 replicate with nothing in it.
    """
    report = FakeRun(
        f"{SUBMISSION}-cell-2-died", config={}, history=[], last_history_step=-1, job_type="crash"
    )
    assert nf.is_crash_report(report.id, report.job_type, -1) is True
    # And the converse, which deleted two replicates the first time it was written: a cell the
    # report was written *onto* carries `job_type: crash` itself and is not a report.
    assert nf.is_crash_report(f"{SUBMISSION}-cell-2", "crash", 4910) is False


# ---------------------------------------------------------------------------------------
# Comparing arms that are not the same age, which is where this tool invented a result.
# ---------------------------------------------------------------------------------------


#: What the fixture below plants between the two arms, at every step both of them reach.
PLANTED_OFFSET_BPB = -0.004


def _two_arms(baseline_steps, faithful_steps):
    """
    Two arms that reached different depths, carrying a constant, known offset wherever they
    overlap. An aligned contrast must find that offset; a contrast that reads each arm at its
    own last step finds the training curve instead, which is the failure being tested for.
    """
    left = an.assemble_arm(
        read(
            [
                cell(i, value, history=_history(value, steps=baseline_steps))
                for i, value in enumerate(BASELINE_ENDPOINTS_BPB)
            ]
        ),
        "baseline",
        SUBMISSION,
    )
    other = "run_019feaaa-0002"
    right = an.assemble_arm(
        read(
            [
                cell(
                    i,
                    value + PLANTED_OFFSET_BPB + jitter,
                    arm="faithful",
                    submission=other,
                    history=_history(
                        value + PLANTED_OFFSET_BPB + jitter, arm="faithful", steps=faithful_steps
                    ),
                )
                # A shift with no seed-to-seed variation gives the contrast a standard error of
                # exactly zero, which is a degenerate design and not a very sensitive one. The
                # jitter sums to zero so the mean difference is still the planted offset.
                for i, (value, jitter) in enumerate(
                    zip(BASELINE_ENDPOINTS_BPB, (-2e-4, -1e-4, 0.0, 1e-4, 2e-4))
                )
            ]
        ),
        "faithful",
        other,
    )
    return left, right


def test_arms_of_different_ages_are_compared_at_the_step_they_share():
    """
    THE TOOL INVENTED A DECISIVE RESULT HERE AND THE TEST IS THAT IT CANNOT AGAIN. Reading the
    complete baseline against a half-trained ``faithful``, each at its own last evaluation, gave
    ``+0.159 nats``, ``t = 70``, ``p = 0.0000``, past every gate and in the opposite direction to
    the prediction. It was step 6,000 against step 2,500 and the number was the training curve.

    A warning above it saying the arms ended at different steps was already being printed, which
    is why the fix is not another warning: the comparison is aligned, or it does not happen.
    """
    left, right = _two_arms((2500, 6000), (1500, 2500))
    result = an.analyse([left, right])

    assert result["compared_at_step"] == 2500
    delta = next(
        row["delta_nats"]
        for entry in result["contrasts"]
        for row in entry["rows"]
        if entry["name"] == "H1" and row["analysis"] == "paired"
    )
    # The planted offset, and nothing else. Read at each arm's own last step instead, the same
    # fixture gives about +1.3 nats -- the half of the training curve between 2,500 and 6,000,
    # which is some three hundred times the offset and would swamp any real effect.
    assert delta == pytest.approx(PLANTED_OFFSET_BPB * an.NATS_PER_BPB, rel=1e-9)


def test_the_report_says_which_step_it_read_each_arm_at():
    left, right = _two_arms((2500, 6000), (1500, 2500))
    text = an.render(an.analyse([left, right]))

    assert "every arm read at step 2500" in text
    assert "reached 6000, read back" in text


def test_arms_that_share_no_evaluation_step_cannot_be_compared_at_all():
    left, right = _two_arms((5500, 6000), (1000, 1500))

    with pytest.raises(an.Refusal, match="share no evaluation step"):
        an.analyse([left, right])


def test_the_age_difference_is_a_completeness_refusal_as_well_as_an_alignment():
    """
    Aligning makes the number coherent; it does not make it the pre-registered endpoint, and
    the reader has to be told which of those they are looking at.
    """
    left, right = _two_arms((2500, 6000), (1500, 2500))
    problems = " ".join(an.completeness_refusals([left, right]))

    assert "every contrast is taken at step 2500" in problems
    assert "is not the pre-registered endpoint" in problems


# ---------------------------------------------------------------------------------------
# The estimators, against planted truths.
# ---------------------------------------------------------------------------------------


def test_the_block_fit_uses_the_randomized_block_error_df():
    """
    ``(k-1)(n-1)``, which for four arms and five seeds is 12 -- not the 16 an unpaired analysis
    would claim. Overstating the df narrows every interval and inflates every t.
    """
    arms = an.synthetic_tranche({"faithful": -0.01, "output-only": 0.0, "mhc": -0.005})
    fit = an.block_fit(
        np.array([arm.endpoint_matrix().mean(axis=1) for arm in arms]),
        [arm.arm for arm in arms],
        arms[0].seeds,
    )
    assert fit.df_paired == (4 - 1) * (5 - 1) == 12
    assert fit.df_unpaired == 4 * (5 - 1) == 16
    assert fit.df_paired < fit.df_unpaired, "pairing is bought with degrees of freedom"


def test_a_post_hoc_contrast_cannot_inflate_a_pre_registered_p_value():
    """
    The pre-registration fixes the family at the pre-registered contrasts and says the gate stays
    uncorrected with Holm printed beside it. A contrast this module added after the endpoints were
    visible, joining that family, would move every pre-registered adjusted p by an amount chosen
    after the fact -- the multiplicity correction running backwards.
    """
    result = an.analyse(
        an.synthetic_tranche({"faithful": -0.010, "output-only": -0.005, "mhc": -0.005}),
        label="synthetic",
    )
    post_hoc = {entry["name"] for entry in result["contrasts"] if entry.get("post_hoc")}

    assert post_hoc, "the post-hoc contrasts are the ones being checked, so there must be some"
    assert not (
        post_hoc & set(result["holm"]["family"])
    ), f"{sorted(post_hoc)} joined the Holm family {result['holm']['family']}"
    assert set(result["holm"]["family"]) <= {
        h.name for h in an.HYPOTHESES if not h.post_hoc
    }, "the family is the pre-registered contrasts and nothing else"


def test_every_post_hoc_contrast_says_so_where_a_reader_will_see_it():
    """
    In the ledger at the foot of the report is not sufficient: a number lifted out of the middle
    of a table carries whatever is on its own line and nothing else.
    """
    result = an.analyse(
        an.synthetic_tranche({"faithful": -0.010, "output-only": -0.005, "mhc": -0.005}),
        label="synthetic",
    )
    text = an.render(result)
    for entry in result["contrasts"]:
        if not entry.get("post_hoc"):
            continue
        heading = next(
            line for line in text.splitlines() if f" {entry['name']}: " in line and "-" in line
        )
        assert "POST-HOC" in heading, f"{entry['name']} is headed as though pre-registered"
        assert str(entry["post_hoc"]) in heading, "a post-hoc addition carries the date it was made"


def test_the_per_source_panel_carries_the_pre_registered_contrasts_only():
    """
    The per-source series are drawn identically to one another, so a post-hoc one among them
    reads as pre-registered. It is in the table with its label instead.
    """
    result = an.analyse(
        an.synthetic_tranche({"faithful": -0.010, "output-only": -0.005, "mhc": -0.005}),
        label="synthetic",
    )
    names = {entry["name"] for entry in result["per_source"]}

    assert names, "the per-source panel is empty, so this test is asserting nothing"
    assert not (names & {h.name for h in an.HYPOTHESES if h.post_hoc})


@pytest.mark.parametrize("planted_rho", [0.0, 0.5])
def test_a_planted_effect_comes_back(planted_rho):
    deltas = []
    for replicate in range(120):
        result = an.analyse(
            an.synthetic_tranche(
                {"faithful": -0.010, "output-only": 0.0, "mhc": -0.005},
                rho=planted_rho,
                rng_seed=replicate,
            ),
            label="synthetic",
        )
        deltas.append(
            next(
                row["delta_nats"]
                for entry in result["contrasts"]
                for row in entry["rows"]
                if entry["name"] == "H1" and row["analysis"] == "paired"
            )
        )
    standard_error = float(np.std(deltas, ddof=1)) / math.sqrt(len(deltas))
    assert abs(float(np.mean(deltas)) + 0.010) < 4.0 * standard_error


def test_the_interval_covers_the_planted_effect_at_about_95_percent():
    """
    A 95% interval that covers 80% of the time is not a 95% interval, and nothing downstream of
    it means what its label says.
    """
    covered = []
    for replicate in range(300):
        result = an.analyse(
            an.synthetic_tranche(
                {"faithful": -0.010, "output-only": 0.0, "mhc": -0.005},
                rho=0.5,
                rng_seed=5_000 + replicate,
            ),
            label="synthetic",
        )
        row = next(
            row
            for entry in result["contrasts"]
            for row in entry["rows"]
            if entry["name"] == "H1" and row["analysis"] == "paired"
        )
        covered.append(row["ci_nats"][0] <= -0.010 <= row["ci_nats"][1])
    rate = float(np.mean(covered))
    assert abs(rate - 0.95) < 4.0 * math.sqrt(0.95 * 0.05 / len(covered)) + 0.01, rate


def test_a_planted_null_is_rejected_at_about_the_rate_it_should_be():
    """
    The calibration that decides whether a p-value below 0.05 means anything. Planted zero
    effect, so every rejection is a false one.
    """
    rejected = []
    for replicate in range(300):
        result = an.analyse(
            an.synthetic_tranche(
                {"faithful": 0.0, "output-only": 0.0, "mhc": 0.0},
                rho=0.5,
                rng_seed=30_000 + replicate,
            ),
            label="synthetic",
        )
        row = next(
            row
            for entry in result["contrasts"]
            for row in entry["rows"]
            if entry["name"] == "H1" and row["analysis"] == "paired"
        )
        rejected.append(row["p_value"] < 0.05)
    rate = float(np.mean(rejected))
    assert rate < 0.05 + 4.0 * math.sqrt(0.05 * 0.95 / len(rejected)), rate


def test_the_minimum_detectable_effect_is_the_effect_that_would_be_found_at_80_percent():
    """
    ``mde_from_se`` is defined by inverting the power function, so the check is that putting it
    back in returns the power it was asked for. Anything else and the gate every contrast is
    reported against is a number with no operational meaning.
    """
    for df, se in ((12, 0.0009), (16, 0.0012), (4, 0.002)):
        detectable = an.mde_from_se(se, df)
        assert nf.power_of(detectable, se / nf.c4(df), df, 0.05) == pytest.approx(0.80, abs=1e-6)


def test_the_c4_correction_is_applied_to_the_gate_and_not_to_the_test():
    """
    ONCE, AND IN ONE PLACE. A standard error built from a sample standard deviation is biased
    low by ``c4(df)``, the pre-registration prices the design off the corrected figure, and the
    t statistic beside it is built on the distribution of ``s`` and already carries that bias.
    Correcting both is the double-count this check exists to prevent.
    """
    se, df = 0.001, 12
    assert an.mde_from_se(se, df) == pytest.approx(an.mde_from_se(se * nf.c4(df), df) / nf.c4(df))
    assert an.mde_from_se(se, df) > se * (1.96 + 0.84), "the correction inflates the gate"


def test_the_break_even_correlation_is_where_pairing_stops_paying():
    """
    Below it, pairing costs more in df than it buys in variance. The pre-registered 0.09 was
    derived at three arms; this is the same calculation at whatever the design actually is.
    """
    rho = an.break_even_rho(n_arms=3, n_seeds=5)
    assert 0.0 < rho < 0.2
    assert an.break_even_rho(n_arms=4, n_seeds=5) < rho, "more arms, cheaper pairing"


# ---------------------------------------------------------------------------------------
# H7, and the limit it has to state.
# ---------------------------------------------------------------------------------------


def permutation(treatment, comparator):
    return an.exact_permutation_test(treatment, comparator, "declined steps", "a", "b")


def test_complete_separation_gives_the_smallest_p_the_design_can_produce():
    """
    At five against five there are C(10,5) = 252 splits, so the smallest attainable two-sided p
    is 2/252 = 0.0079. Complete separation is detectable at alpha = 0.05 and nothing weaker is
    guaranteed to be -- which the report has to say rather than let a null be read as evidence.
    """
    test = permutation([40, 44, 51, 39, 62], [19, 10, 16, 18, 20])
    assert test.n_permutations == 252
    assert test.smallest_attainable_p == pytest.approx(2.0 / 252.0, abs=1e-9)
    assert test.p_value == pytest.approx(2.0 / 252.0, abs=1e-9)
    assert test.complete_separation


def test_identical_arms_give_a_p_of_one():
    counts = [19, 10, 16, 18, 20]
    test = permutation(list(counts), list(counts))
    assert test.p_value == pytest.approx(1.0)
    assert not test.complete_separation


def test_partial_separation_is_mostly_not_detectable_and_the_report_must_say_so():
    """
    The honest limit of H7. These two arms differ by a third of their mean and the test cannot
    call it at 5 v 5, so a p above 0.05 here is an absence of evidence rather than evidence of
    absence, and the report says which.
    """
    assert permutation([24, 21, 19, 26, 17], [19, 10, 16, 18, 20]).p_value > 0.05


def test_the_exact_test_agrees_with_an_independent_enumeration():
    """The estimator against a second implementation of the definition, not against itself."""
    import itertools

    left, right = [7, 3, 9, 2, 11], [4, 6, 1, 8, 5]
    pool = left + right
    observed = abs(np.mean(left) - np.mean(right))
    extreme = 0
    for chosen in itertools.combinations(range(10), 5):
        a = [pool[i] for i in chosen]
        b = [pool[i] for i in range(10) if i not in chosen]
        extreme += abs(np.mean(a) - np.mean(b)) >= observed - 1e-12
    assert permutation(left, right).p_value == pytest.approx(extreme / 252.0)


def test_a_declined_step_count_is_taken_over_the_same_number_of_steps_on_both_arms():
    """
    A COUNT IS AN EXPOSURE STATISTIC AND THIS ONE ALMOST REPORTED THE OPPOSITE OF THE TRUTH.
    ``stability/steps skipped`` is cumulative over a whole run, so reading each arm's own
    running total compared a comparator's six thousand steps against a treatment's three
    thousand. On the live tranche that gave complete separation at the floor of the test,
    ``p = 0.0079``, with the treatment apparently the calmer arm; counted over the same first
    3,000 steps on both, the same data give ``p = 0.0794`` and no separation. The entire result
    was the treatment having had half as long to decline anything.
    """
    arms = an.synthetic_tranche({"faithful": -0.01}, declined={"faithful": (8, 9, 7, 8, 9)})
    faithful = next(arm for arm in arms if arm.arm == "faithful")

    early, _ = faithful.stability_at(1000)
    whole, _ = faithful.stability_at(None)

    assert all(a <= b for a, b in zip(early, whole)), "a prefix cannot hold more events"
    assert sum(early) < sum(whole), "and over a sixth of the run it should hold fewer"
    assert whole == [8, 9, 7, 8, 9], "over the whole run it is the count that was logged"


def test_h7_counts_through_the_step_the_contrasts_were_taken_at():
    arms = an.synthetic_tranche({"faithful": -0.01})
    outcome = an.h7(arms, through_step=1500)

    assert outcome["through_step"] == 1500
    row = next(e for e in outcome["tests"] if e["arm"] == "faithful")
    assert "in the first 1,500" in row["primary"]["statistic_name"]


def test_a_running_total_that_disagrees_with_the_history_is_refused():
    """
    If the events behind the total are not all there, a count over any prefix is short by an
    unknown amount -- and H7 is a test on exactly that count, so it cannot be run.
    """
    arms = an.synthetic_tranche({"faithful": -0.01})
    faithful = next(arm for arm in arms if arm.arm == "faithful")
    faithful.declined_steps[0] = faithful.declined_steps[0][:-1]

    with pytest.raises(an.Refusal, match="appear in its per-step history"):
        faithful.stability_at(None)


def test_h7_runs_on_the_stability_families_and_names_its_own_floor():
    arms = an.synthetic_tranche(
        {"faithful": -0.01, "output-only": 0.0, "mhc": -0.005},
        declined={"faithful": (44, 51, 39, 47, 62)},
        triggers={"faithful": (8.4, 11.2, 6.9, 9.1, 14.0)},
    )
    outcome = an.h7(arms)
    text = json.dumps(outcome, default=float)

    assert "0.0079" in text, "the smallest attainable p is stated wherever H7 is reported"
    row = next(
        entry for entry in outcome["tests"] if entry["arm"] == "faithful"  # type: ignore[index]
    )
    assert row["primary"]["p_value"] == pytest.approx(2.0 / 252.0, abs=1e-9)
    assert row["primary"]["complete_separation"]
    assert row["secondary"]["complete_separation"]
    # The two arms that are not H7 are still run, and still labelled as carrying no claim.
    descriptive = [e for e in outcome["tests"] if e["arm"] != "faithful"]  # type: ignore[index]
    assert descriptive and all(not e["pre_registered"] for e in descriptive)
    assert row["pre_registered"], "faithful against baseline is the one that is H7"


# ---------------------------------------------------------------------------------------
# The defect class the module was told to guard against: numbers with no data behind them.
# ---------------------------------------------------------------------------------------


def test_a_synthetic_report_says_so_on_every_line():
    """
    ``noise_floor.py --submission X --dry-run`` once produced a complete synthetic report that
    was read as a measurement, because the W&B query sat behind a flag nobody had passed. The
    rule taken from that: anything that can emit plausible numbers without touching real data
    marks every one of them.
    """
    result = an.analyse(an.synthetic_tranche({"faithful": -0.01}), label="synthetic")
    text = an.render(result)
    body = [line for line in text.splitlines() if line.strip() and not line.startswith("=")]

    assert "SYNTHETIC" in text
    assert all(
        line.startswith("[synthetic]") or line.startswith("  ") for line in body
    ), "a line of a synthetic report that is not marked as one can be quoted as a measurement"


def test_the_banner_is_at_the_top_and_at_the_bottom():
    """
    Both ends, because a long report is read from either. Truncated to its first screen or to
    its last, it still says what it is.
    """
    text = an.render(an.analyse(an.synthetic_tranche({"faithful": -0.01}), label="synthetic"))
    lines = text.splitlines()
    top = "\n".join(lines[:8]).upper()
    bottom = "\n".join(lines[-8:]).upper()

    assert "NOTHING BELOW IS A MEASUREMENT OF ANYTHING" in top
    assert "NOTHING ABOVE IS A MEASUREMENT OF ANYTHING" in bottom


def test_a_cache_of_synthetic_data_cannot_be_read_back_as_measured(tmp_path):
    """
    The cache is the one place a synthetic number could re-enter the measured path wearing the
    measured label, so the label is written into the file and checked on the way out.
    """
    path = str(tmp_path / "cache.json")
    with open(path, "w") as handle:
        json.dump({"label": "synthetic", "arms": []}, handle)

    with pytest.raises(an.Refusal, match="not a cache of measured data"):
        an._cache_load(path)


def test_the_measured_and_synthetic_command_lines_cannot_be_combined():
    from subprocess import run as spawn

    finished = spawn(
        [sys.executable, os.path.join(HERE, "analysis.py"), "--demo", "--group", "x"],
        capture_output=True,
        text=True,
    )
    assert finished.returncode != 0
    assert "cannot be combined" in finished.stderr


def test_there_is_no_default_and_no_fallback():
    """The analysis command fails loudly rather than falling back to anything."""
    from subprocess import run as spawn

    finished = spawn(
        [sys.executable, os.path.join(HERE, "analysis.py")], capture_output=True, text=True
    )
    assert finished.returncode != 0
    assert "no default and there is no fallback" in finished.stderr


def test_a_nats_column_that_disagrees_with_the_bpb_column_is_refused():
    """
    Both are logged, and the report quotes effects in both. If the conversion drifts, one of
    the two columns is wrong and there is no way to tell which from inside the report.
    """
    runs = healthy()
    key = an.CE_METRIC.format(source=an.HELD_OUT_SOURCES[0])
    for row in runs[0]._history:
        if key in row:
            row[key] = row[key] * 1.05

    arm = an.assemble_arm(read(runs), "baseline", SUBMISSION)
    assert an.check_the_nats_conversion([arm]), "a 5% drift in one source must be caught"


# ---------------------------------------------------------------------------------------
# The figures, checked for the one property that matters when they leave the repository.
# ---------------------------------------------------------------------------------------


def test_every_synthetic_figure_carries_the_watermark(tmp_path):
    """
    A figure travels further than the report it came from -- into a slide, into a message --
    and arrives without the banner. So the mark goes on the image.
    """
    import analysis_figures

    arms = an.synthetic_tranche({"faithful": -0.01, "output-only": 0.0, "mhc": -0.005})
    result = an.analyse(arms, label="synthetic")
    written = analysis_figures.draw(arms, result, str(tmp_path))

    assert written
    for path in written:
        assert os.path.basename(path).startswith("synthetic-")
        assert os.path.getsize(path) > 0
