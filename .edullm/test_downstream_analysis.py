"""
Tests for ``downstream_analysis.py``, the pre-registered analysis of the downstream scoring job.

EVERY NUMBER THIS FILE CHECKS AGAINST IS EITHER PLANTED OR FROZEN, BECAUSE WHEN IT WAS WRITTEN
THERE WAS NO THIRD OPTION. The scoring job had not been submitted and no document of
``edullm.hyper-connections.downstream.v1`` existed anywhere, so there was nothing to tune an
estimator to even by accident. That is the point of the sequencing and this file is where it is
cashed: the slope estimator is asked to recover a truth chosen by a random number generator, the
interval is asked to cover at the rate it claims, and the arm contrasts are asked to find an
effect that was written into the fixture before the estimator saw it.

Five groups.

*The reader*, which is the largest, because this is where the module can be wrong in ways that
still produce a number. A truncated score is a number. A headline averaged over two groups
instead of three is a number. Twenty-four cells is a number. Each of those is a test that the
answer is a **refusal** and that the refusal names what is wrong.

*The slope*, against planted truths of one and of 0.35 -- the two cases the pre-registration
turns on, because one is the coupled world and materially below one is the decoupling the whole
job exists to detect. Recovery, coverage and false-positive rate are all checked.

*The two levels*, because the pooled slope over twenty-five cells blends a between-arm and a
within-arm relationship, and the withholding rule that fires when they differ is pre-registered.
Planted both ways: arms on one line, and arms deliberately off it.

*The declarations*, which are the part a reader has to be able to trust was made in advance. The
``declared_underpowered`` flag on every hypothesis is re-derived from the frozen in-loop
endpoints, so a flag that has been edited to suit an outcome fails here.

*The document*, because a pre-registration that lives only in code is not one. The section in
``hyper-connections.md`` is parsed and checked to say what this module does.
"""

import copy
import json
import math
import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import downstream_analysis as da  # noqa: E402
import score_checkpoints as sc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "hyper-connections.md")

#: The in-loop endpoint of all twenty-five cells, unweighted mean held-out BPB over the seven
#: sources at step 6,000, as ``analysis.py`` froze them on 2026-08-12.
#:
#: A FIXTURE AND NOT A FETCH, for the reason ``test_analysis.py`` gives about its own copy of the
#: baseline five: these are the numbers the in-loop report was published against, so a test that
#: re-read W&B to get them would be checking that W&B is up. Frozen here, the x-axis of every
#: test below is the same x-axis the measured run will use.
FROZEN_IN_LOOP = {
    "baseline": (
        0.6756527410392009,
        0.6752062205312278,
        0.6756557516764151,
        0.6768358031893593,
        0.6760047490431987,
    ),
    "faithful": (
        0.6620785896370645,
        0.6619471108090087,
        0.659254697948324,
        0.6622963178451698,
        0.6610237161185776,
    ),
    "output-only": (
        0.6637584338082599,
        0.6635457968027293,
        0.662235981449714,
        0.665641625018725,
        0.6651504126765343,
    ),
    "no-output-init": (
        0.6646951989536051,
        0.6638543172273952,
        0.6608141811446051,
        0.6604911075146911,
        0.6665350477404611,
    ),
    "mhc": (
        0.6637147526879641,
        0.6647114080093209,
        0.6634783047677517,
        0.6635713925951882,
        0.6646226641012937,
    ),
}

FROZEN_CELLS = {
    (arm, seed): value for arm, row in FROZEN_IN_LOOP.items() for seed, value in enumerate(row)
}


def planted(slope=1.0, residual_sd=0.002, seed=0, in_loop=None, arm_offsets=None):
    """
    Twenty-five cells with a planted slope, on the frozen in-loop x-axis by default.

    :param slope: The truth.
    :param residual_sd: Downstream scatter around the line.
    :param seed: RNG seed.
    :param in_loop: Override the x-axis.
    :param arm_offsets: Per-arm vertical displacement off the common line, for the withholding
        test. ``None`` puts every arm on it.

    :returns: ``(cells, endpoints)``.
    """
    documents, endpoints = da.synthetic_documents(
        slope=slope, residual_sd=residual_sd, seed=seed, in_loop=in_loop or FROZEN_CELLS
    )
    if arm_offsets:
        for document in documents:
            shift = arm_offsets.get(document["arm"], 0.0)
            for entry in document["tasks"].values():
                entry["metrics"][da.PRIMARY_METRIC] += shift
            results = [
                sc.TaskResult(
                    label=label,
                    group=entry["group"],
                    metrics={da.PRIMARY_METRIC: entry["metrics"][da.PRIMARY_METRIC]},
                )
                for label, entry in document["tasks"].items()
            ]
            document["downstream"][da.PRIMARY_METRIC] = sc.aggregate(results, da.PRIMARY_METRIC)
    cells = [da.cell_from_document(d, f"planted-{i}.json") for i, d in enumerate(documents)]
    da.attach_in_loop(cells, endpoints)
    return cells, endpoints


def write_documents(directory, documents):
    """
    Put documents on disk under the names the scoring job writes.

    :param directory: Where.
    :param documents: What.

    :returns: The directory.
    """
    for document in documents:
        name = f"downstream-{document['arm']}-seed{document['seed']}-step{document['step']}.json"
        with open(os.path.join(directory, name), "w") as handle:
            json.dump(document, handle)
    return str(directory)


# ---------------------------------------------------------------------------------------
# The reader, and every case where a smaller analysis would have produced a number.
# ---------------------------------------------------------------------------------------


def test_the_schema_is_taken_from_the_writer_and_not_restated():
    assert da.INPUT_SCHEMA == sc.OUTPUT_SCHEMA
    assert da.PRIMARY_METRIC == sc.PRIMARY_METRIC
    assert da.HEADLINE_GROUPS == sc.HEADLINE_GROUPS
    assert da.FINAL_STEP == sc.FINAL_STEP
    assert len(da.EXPECTED_CELLS) == 25


def test_a_whole_tranche_reads():
    cells, _ = planted()
    assert len(cells) == 25
    assert not da.completeness_refusals(cells)
    assert not da.instrument_refusals(cells)


def test_a_short_tranche_names_the_missing_cells_and_does_not_count_them():
    cells, _ = planted()
    short = [c for c in cells if (c.arm, c.seed) not in {("mhc", 4), ("faithful", 1)}]
    complaints = da.completeness_refusals(short)
    assert complaints
    assert "mhc seed 4" in complaints[0]
    assert "faithful seed 1" in complaints[0]


def test_two_documents_for_one_cell_are_refused():
    cells, _ = planted()
    complaints = da.completeness_refusals(list(cells) + [cells[0]])
    assert any("2 documents" in c for c in complaints)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "NVIDIA A100-SXM4-80GB"),
        ("param_dtype", "float32"),
        ("torch", "2.9.0"),
        ("tokenizer", "/somewhere/else/tokenizer.json"),
        ("suite_version", "h2b-rc-2026-09-a"),
    ],
)
def test_twenty_five_numbers_off_two_instruments_are_refused(field, value):
    documents, endpoints = da.synthetic_documents(in_loop=FROZEN_CELLS)
    documents[7][field] = value
    cells = [da.cell_from_document(d, f"p-{i}.json") for i, d in enumerate(documents)]
    complaints = da.instrument_refusals(cells)
    assert complaints and field in complaints[0]


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda d: d.update(schema="edullm.something.else.v1"), "schema"),
        (lambda d: d.update(truncated=True), "truncated"),
        (lambda d: d.update(step=5500), "step"),
        (lambda d: d.update(arm_number=99), "arm number"),
        (lambda d: d.update(seed=17), "not a cell of the tranche"),
        (lambda d: d.update(arm="decay-everything"), "not a cell of the tranche"),
    ],
)
def test_a_document_that_would_still_have_produced_a_number_is_refused(mutate, expected):
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    document = copy.deepcopy(documents[0])
    mutate(document)
    with pytest.raises(da.Refusal, match=expected):
        da.cell_from_document(document, "p.json")


def test_a_headline_over_fewer_groups_is_a_different_quantity_and_is_refused():
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    document = copy.deepcopy(documents[0])
    aggregate = document["downstream"][da.PRIMARY_METRIC]
    aggregate["headline_groups"] = ["olmes", "mmlu"]
    with pytest.raises(da.Refusal, match="headline from groups"):
        da.cell_from_document(document, "p.json")


def test_a_missing_task_is_refused_because_it_moves_the_headline_silently():
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    document = copy.deepcopy(documents[0])
    del document["tasks"]["hellaswag_rc_5shot"]["metrics"][da.PRIMARY_METRIC]
    with pytest.raises(da.Refusal, match="hellaswag_rc_5shot"):
        da.cell_from_document(document, "p.json")


def test_a_null_headline_is_refused_rather_than_read_as_zero():
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    document = copy.deepcopy(documents[0])
    document["downstream"][da.PRIMARY_METRIC]["headline"] = None
    with pytest.raises(da.Refusal, match="null or non-finite headline"):
        da.cell_from_document(document, "p.json")


def test_an_s3_prefix_is_refused_rather_than_opened():
    with pytest.raises(da.Refusal, match="AGENTS.md"):
        da.read_documents("s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/whatever/")


def test_a_directory_with_nothing_in_it_says_so(tmp_path):
    with pytest.raises(da.Refusal, match="downstream-\\*.json"):
        da.read_documents(str(tmp_path))


def test_documents_round_trip_through_disk_under_the_names_the_job_writes(tmp_path):
    documents, endpoints = da.synthetic_documents(in_loop=FROZEN_CELLS)
    where = write_documents(tmp_path, documents)
    read = da.read_documents(where)
    assert len(read) == 25
    cells = [da.cell_from_document(d, p) for p, d in read]
    da.attach_in_loop(cells, endpoints)
    assert not da.completeness_refusals(cells)


def test_a_cell_with_no_in_loop_endpoint_is_refused_rather_than_dropped():
    cells, endpoints = planted()
    partial = {k: v for k, v in endpoints.items() if k != ("mhc", 2)}
    with pytest.raises(da.Refusal, match="mhc seed 2"):
        da.attach_in_loop(cells, partial)


# ---------------------------------------------------------------------------------------
# The in-loop artifact, which is the fixed axis of the primary.
# ---------------------------------------------------------------------------------------


def _artifact(**overrides):
    body = {
        "label": "measured",
        "provisional": [],
        "compared_at_step": da.FINAL_STEP,
        "generated": "2026-08-12",
        "arms": [
            {"arm": arm, "seeds": list(range(5)), "endpoint_bpb": list(values)}
            for arm, values in FROZEN_IN_LOOP.items()
        ],
    }
    body.update(overrides)
    return body


def test_the_frozen_in_loop_artifact_reads(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(_artifact()))
    endpoints, provenance = da.in_loop_from_artifact(str(path))
    assert endpoints == FROZEN_CELLS
    assert "2026-08-12" in provenance


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"label": "synthetic"}, "measured"),
        ({"provisional": ["one cell never reached the horizon"]}, "provisional"),
        ({"compared_at_step": 3000}, "two different models"),
    ],
)
def test_an_in_loop_axis_that_is_not_the_frozen_one_is_refused(tmp_path, overrides, expected):
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(_artifact(**overrides)))
    with pytest.raises(da.Refusal, match=expected):
        da.in_loop_from_artifact(str(path))


# ---------------------------------------------------------------------------------------
# The slope. The primary, against planted truths.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("truth", [1.0, 0.35, 0.0, 1.4])
def test_the_slope_estimator_recovers_what_was_planted(truth):
    """
    The two cases the pre-registration turns on are 1.0 and materially below it. Zero and above
    one are here so that a bug which pulls every answer towards one -- which is the bug that
    would be hardest to see, because it agrees with the null -- cannot pass.
    """
    recovered = [
        da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t").slope
        for cells, _ in (planted(slope=truth, residual_sd=0.003, seed=s) for s in range(300))
    ]
    assert abs(float(np.mean(recovered)) - truth) < 0.04


def test_the_slope_interval_covers_at_the_rate_it_claims():
    truth = 0.6
    covered = 0
    trials = 600
    for s in range(trials):
        cells, _ = planted(slope=truth, residual_sd=0.004, seed=5_000 + s)
        fit = da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")
        covered += int(fit.ci[0] <= truth <= fit.ci[1])
    rate = covered / trials
    # Three binomial standard errors of 0.95 at this many trials.
    assert abs(rate - 0.95) < 3 * math.sqrt(0.95 * 0.05 / trials) + 0.01


def test_the_test_against_one_rejects_at_five_percent_when_the_truth_is_one():
    rejections = 0
    trials = 600
    for s in range(trials):
        cells, _ = planted(slope=1.0, residual_sd=0.004, seed=9_000 + s)
        fit = da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")
        against_one = next(t for t in fit.tests if t.null == 1.0)
        rejections += int(against_one.p_value < 0.05)
    rate = rejections / trials
    assert abs(rate - 0.05) < 3 * math.sqrt(0.05 * 0.95 / trials) + 0.01


def test_a_planted_decoupling_is_found_and_a_planted_coupling_is_not():
    """
    The two headline outcomes, at a downstream scatter the design can work at. A slope of 0.35
    should clear the gate against one; a slope of one should not.
    """
    decoupled, _ = planted(slope=0.35, residual_sd=0.002, seed=3)
    coupled, _ = planted(slope=1.0, residual_sd=0.002, seed=3)
    for cells, expected in ((decoupled, True), (coupled, False)):
        fit = da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")
        against_one = next(t for t in fit.tests if t.null == 1.0)
        assert against_one.clears_gate is expected


def test_the_slope_has_the_degrees_of_freedom_the_design_has():
    cells, _ = planted()
    xs = [c.in_loop_bpb for c in cells]
    ys = [c.downstream_bpb for c in cells]
    arms = [c.arm for c in cells]
    assert da.regress(xs, ys, "pooled").df == 23
    assert da.arm_mean_slope(xs, ys, arms).df == 3
    assert da.within_arm_slope(xs, ys, arms).df == 19


def test_the_x_axis_leverage_is_almost_entirely_between_arms():
    """
    The arithmetic the pre-registration rests on when it says the within-arm row is the weak one:
    94% of the spread in the in-loop endpoint is between arms, so a within-arm slope is about
    four times less sensitive at the same residual scatter.
    """
    xs = np.asarray([FROZEN_CELLS[cell] for cell in da.EXPECTED_CELLS])
    arms = [arm for arm, _ in da.EXPECTED_CELLS]
    arm_mean = {a: xs[[i for i, n in enumerate(arms) if n == a]].mean() for a in set(arms)}
    total = float(((xs - xs.mean()) ** 2).sum())
    within = float(sum((x - arm_mean[a]) ** 2 for x, a in zip(xs, arms)))
    assert 0.93 < (total - within) / total < 0.95
    assert 3.8 < math.sqrt(total / within) < 4.2
    # And the same numbers come out of the fits the report prints, which is the thing that
    # matters: `s_xx` on each fit is the leverage that fit's standard error is divided by.
    cells, _ = planted(residual_sd=0.003, seed=1)
    ys = [c.downstream_bpb for c in cells]
    assert da.regress(list(xs), ys, "pooled").s_xx == pytest.approx(total)
    assert da.within_arm_slope(list(xs), ys, arms).s_xx == pytest.approx(within)


def test_the_c4_correction_enters_the_mde_and_not_the_interval():
    """
    Applied once, in ``mde_from_se``, and deliberately not to the t statistic or the interval,
    which are built on the distribution of s and already carry the bias. Counting it twice is
    the mistake the in-loop module documents at length.
    """
    cells, _ = planted()
    fit = da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")
    test = fit.tests[0]
    assert test.five_percent == pytest.approx(stats.t.ppf(0.975, fit.df) * fit.se_slope)
    assert test.gate == pytest.approx(2.0 * fit.se_slope)
    assert test.mde > test.five_percent  # 80% power costs more than significance alone


def test_the_power_table_is_a_function_of_the_unknown_and_is_monotone():
    cells, _ = planted()
    fit = da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")
    rows = da.slope_power_table(fit.s_xx, (0.001, 0.004, 0.012), fit.df)
    assert [r["se_slope"] for r in rows] == sorted(r["se_slope"] for r in rows)
    assert [r["power_against_zero"] for r in rows] == sorted(
        (r["power_against_zero"] for r in rows), reverse=True
    )
    # The claim the pre-registration makes: gross decoupling survives a much noisier instrument
    # than an attenuation to a half does.
    assert rows[-1]["power_against_zero"] > rows[-1]["power_against_half"]


def test_a_tranche_with_no_x_spread_is_refused_rather_than_dividing_by_zero():
    flat = {cell: 0.66 for cell in da.EXPECTED_CELLS}
    cells, _ = planted(in_loop=flat)
    with pytest.raises(da.Refusal, match="no x spread"):
        da.regress([c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t")


# ---------------------------------------------------------------------------------------
# The two levels, and the withholding rule.
# ---------------------------------------------------------------------------------------


def test_arms_planted_on_one_line_do_not_trip_the_withholding_rule():
    fired = 0
    for s in range(200):
        cells, _ = planted(slope=0.8, residual_sd=0.003, seed=20_000 + s)
        check = da.one_line_check(
            [c.in_loop_bpb for c in cells],
            [c.downstream_bpb for c in cells],
            [c.arm for c in cells],
        )
        fired += int(check.rejects)
    # A 5% test under the null. Three binomial standard errors.
    assert fired / 200 < 0.05 + 3 * math.sqrt(0.05 * 0.95 / 200) + 0.02


def test_arms_planted_off_the_line_do_trip_it_and_the_primary_is_withheld():
    cells, _ = planted(
        slope=1.0,
        residual_sd=0.001,
        seed=4,
        arm_offsets={"faithful": 0.02, "mhc": -0.02, "output-only": 0.015},
    )
    check = da.one_line_check(
        [c.in_loop_bpb for c in cells],
        [c.downstream_bpb for c in cells],
        [c.arm for c in cells],
    )
    assert check.rejects
    result = da.analyse(cells)
    assert result["regression"]["withheld"] is True
    assert result["regression"]["reported_fit"] == "arm means"
    assert "anti-conservative" in result["regression"]["withholding_rule"]


def test_the_withholding_rule_can_never_create_a_claim():
    """
    It moves which fit is *reported* and never touches a point estimate. Both fits are in the
    artifact either way, so nothing is withdrawn -- only demoted.
    """
    cells, _ = planted(slope=0.5, residual_sd=0.002, seed=6)
    result = da.analyse(cells)
    assert set(result["regression"]) >= {"pooled", "arm_means", "within_arm", "one_line_check"}


def test_the_arm_mean_fit_pays_about_fifty_percent_in_interval_width_for_its_df():
    """
    The conservative companion costs ``t(0.975, 3) / t(0.975, 23) = 1.54`` and buys immunity to
    the clustering assumption. Checked as a median over draws because a df = 3 estimate of the
    residual scatter is itself volatile.
    """
    ratios = []
    for s in range(200):
        cells, _ = planted(slope=1.0, residual_sd=0.003, seed=30_000 + s)
        xs = [c.in_loop_bpb for c in cells]
        ys = [c.downstream_bpb for c in cells]
        arms = [c.arm for c in cells]
        pooled = da.regress(xs, ys, "pooled")
        between = da.arm_mean_slope(xs, ys, arms)
        ratios.append((between.ci[1] - between.ci[0]) / (pooled.ci[1] - pooled.ci[0]))
    assert 1.3 < float(np.median(ratios)) < 1.9


# ---------------------------------------------------------------------------------------
# The arm contrasts, and the declarations made before the data.
# ---------------------------------------------------------------------------------------


def test_the_underpowered_declarations_are_what_the_frozen_in_loop_arithmetic_gives():
    """
    THE TEST THAT MAKES THE DECLARATION A DERIVATION. Every ``declared_underpowered`` flag is
    re-computed from the frozen in-loop endpoints, so a flag edited to suit an outcome fails
    here rather than being taken on trust.
    """
    means = {arm: float(np.mean(values)) for arm, values in FROZEN_IN_LOOP.items()}
    for hypothesis in da.HYPOTHESES:
        effect = means[hypothesis.treatment] - means[hypothesis.comparator]
        assert da.declares_underpowered(effect) == hypothesis.declared_underpowered, hypothesis.name


def test_exactly_the_three_arm_versus_arm_rows_are_declared_underpowered():
    declared = {h.name for h in da.HYPOTHESES if h.declared_underpowered}
    assert declared == {"D2b-i", "D5", "D1b"}
    powered = {h.name for h in da.HYPOTHESES if not h.declared_underpowered}
    assert powered == {"D1", "D1a", "D2b-ii"}
    # And every powered one is a treatment against the baseline, which is the structural reason.
    assert all(h.comparator == "baseline" for h in da.HYPOTHESES if not h.declared_underpowered)


def test_the_sigma_ceilings_separate_the_two_groups_by_a_factor_of_four():
    means = {arm: float(np.mean(values)) for arm, values in FROZEN_IN_LOOP.items()}
    ceilings = {
        h.name: da.sigma_ceiling(abs(means[h.treatment] - means[h.comparator]))
        for h in da.HYPOTHESES
    }
    powered = [ceilings[n] for n in ("D1", "D1a", "D2b-ii")]
    starved = [ceilings[n] for n in ("D2b-i", "D5", "D1b")]
    # Nothing lands in the gap, which is why the threshold can sit in the middle of it.
    assert min(powered) / max(starved) > 4.0
    assert max(starved) <= da.IN_LOOP_POOLED_SIGMA_BPB
    assert min(powered) > 3.5 * da.IN_LOOP_POOLED_SIGMA_BPB
    # Roughly ten to twelve times the in-loop baseline floor, which is the ratio the plan quotes.
    assert 9 < min(powered) / da.IN_LOOP_BASELINE_SIGMA_BPB
    assert max(powered) / da.IN_LOOP_BASELINE_SIGMA_BPB < 13
    assert da.UNDERPOWERED_BELOW_BPB > max(starved)
    assert da.UNDERPOWERED_BELOW_BPB < min(powered)


def test_the_sigma_ceilings_reproduce_the_table_in_the_pre_registration():
    """
    The document quotes six ceilings and this asserts the code gives them, so the two cannot
    drift. Priced as a standalone two-arm comparison at df = 8, which is the conservative read
    and the one Bartlett's in-loop rejection points at.
    """
    for effect, expected in (
        (0.0146, 0.0072),
        (0.0126, 0.0062),
        (0.0118, 0.0058),
        (0.0028, 0.0014),
        (0.0027, 0.0013),
        (0.0020, 0.0010),
    ):
        assert da.sigma_ceiling(effect) == pytest.approx(expected, abs=5e-5)


def _plan_table(header_fragment):
    """
    One markdown table out of the pre-registration section, as lists of cell strings.

    THE SAME DEVICE ``test_noise_floor.py`` USES ON THE MDE TABLE, AND FOR THE SAME REASON: a
    number quoted in the plan and a number the estimator gives are two things that agree until
    one of them is edited. Parsing the document back out means the pair cannot drift silently.

    :param header_fragment: A string that appears in the table's header row.

    :returns: The body rows, each a list of cells.
    """
    section = _plan_text().split("## The downstream pre-registration of 2026-08-12")[1]
    section = section.split("\n## ")[0]
    lines = section.splitlines()
    start = next(i for i, line in enumerate(lines) if header_fragment in line and "|" in line)
    rows = []
    for line in lines[start + 2 :]:
        if not line.strip().startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _number(text):
    """
    A float out of a document cell, tolerating the typography the plan is written in.

    :param text: The cell.

    :returns: The float.
    """
    cleaned = (
        text.replace("**", "")
        .replace("\u2212", "-")
        .replace("\u2264", "")
        .replace("\u00d7", "")
        .strip()
    )
    return float(cleaned)


def test_the_plans_sigma_ceiling_table_is_what_the_code_gives():
    rows = _plan_table("downstream \u03c3 it needs")
    assert len(rows) == 6
    means = {arm: float(np.mean(values)) for arm, values in FROZEN_IN_LOOP.items()}
    by_name = {h.name: h for h in da.HYPOTHESES}
    for row in rows:
        name = row[0].replace("**", "")
        hypothesis = by_name[name]
        effect = means[hypothesis.treatment] - means[hypothesis.comparator]
        assert _number(row[2]) == pytest.approx(effect, abs=5e-5), name
        assert _number(row[3]) == pytest.approx(da.sigma_ceiling(abs(effect)), abs=5e-5), name
        assert _number(row[4]) == pytest.approx(
            da.sigma_ceiling(abs(effect)) / da.IN_LOOP_POOLED_SIGMA_BPB, abs=0.05
        ), name
        declared = "under-powered" in row[5]
        assert declared == hypothesis.declared_underpowered, name


def test_the_plans_slope_power_table_is_what_the_code_gives():
    rows = _plan_table("95% half-width")
    assert len(rows) == 7
    xs = np.asarray([FROZEN_CELLS[cell] for cell in da.EXPECTED_CELLS])
    s_xx = float(((xs - xs.mean()) ** 2).sum())
    computed = da.slope_power_table(s_xx, [_number(row[0]) for row in rows], 23)
    for row, expected in zip(rows, computed):
        assert _number(row[1]) == pytest.approx(expected["se_slope"], abs=5e-4)
        assert _number(row[2]) == pytest.approx(expected["half_width"], abs=5e-3)
        assert _number(row[3]) == pytest.approx(expected["power_against_zero"], abs=5e-3)
        assert _number(row[4]) == pytest.approx(expected["power_against_half"], abs=5e-3)


def test_the_plans_seeds_needed_table_is_what_the_code_gives():
    rows = _plan_table("seeds/arm at")
    assert len(rows) == 6
    means = {arm: float(np.mean(values)) for arm, values in FROZEN_IN_LOOP.items()}
    by_name = {h.name: h for h in da.HYPOTHESES}
    for row in rows:
        hypothesis = by_name[row[0].replace("**", "")]
        effect = abs(means[hypothesis.treatment] - means[hypothesis.comparator])
        for column, sigma in enumerate((0.002, 0.004, 0.006, 0.008), start=1):
            assert int(row[column]) == da.seeds_needed(effect, sigma), (row[0], sigma)


def test_the_plans_frozen_constants_are_the_ones_the_code_carries():
    section = _plan_text().split("## The downstream pre-registration of 2026-08-12")[1]
    section = section.split("\n## ")[0]
    assert f"{da.IN_LOOP_BASELINE_SIGMA_BPB:.5f}" in section
    assert f"{da.IN_LOOP_POOLED_SIGMA_BPB:.5f}" in section
    xs = np.asarray([FROZEN_CELLS[cell] for cell in da.EXPECTED_CELLS])
    assert f"{float(xs.max() - xs.min()):.4f}" in section
    assert f"{float(((xs - xs.mean()) ** 2).sum()):.3e}".replace("e-04", "e-04") in section


def test_a_planted_arm_effect_is_recovered_by_the_primary_row():
    planted_gap = 0.010
    cells, _ = planted(slope=1.0, residual_sd=0.001, seed=8)
    for cell in cells:
        if cell.arm == "faithful":
            cell.downstream_bpb -= planted_gap
    result = da.analyse(cells)
    entry = next(e for e in result["contrasts"] if e["name"] == "D1")
    row = next(r for r in entry["rows"] if r["primary"])
    # The in-loop gap is already about -0.0146 and the plant adds to it.
    expected = -0.0146 - planted_gap
    assert row["delta_bpb"] == pytest.approx(expected, abs=0.004)
    assert row["ci_bpb"][0] < expected < row["ci_bpb"][1]


def test_the_contrast_carries_no_nats_column():
    """
    ``analysis.py`` reports nats because 4.57 bytes per token is one constant across the seven
    held-out sources. The downstream suite is thirteen other texts and the constant is neither
    4.57 nor one number, so a nats column here would be in no unit at all and would be quoted.
    """
    fields = set(da.DownstreamContrast.__dataclass_fields__)
    assert not any("nats" in name for name in fields)
    assert "delta_bpb" in fields


def test_bartlett_rejecting_makes_welch_primary_on_every_row():
    cells, _ = planted(slope=1.0, residual_sd=0.001, seed=9)
    rng = np.random.default_rng(0)
    for cell in cells:
        if cell.arm == "mhc":
            cell.downstream_bpb += float(rng.normal(0.0, 0.05))
    result = da.analyse(cells)
    assert result["sigma"]["bartlett"]["rejects"] is True
    for entry in result["contrasts"]:
        primary = [r for r in entry["rows"] if r["primary"]]
        assert len(primary) == 1
        assert primary[0]["analysis"] == "welch"
        assert entry["bartlett_forced_welch"] is True


def test_the_pairing_blocks_on_the_seed_every_arm_shares():
    cells, _ = planted()
    result = da.analyse(cells)
    assert "init_seed" in result["pairing"]["blocked_on"]
    assert result["pairing"]["block_fit"]["df_paired"] == 16
    assert result["pairing"]["block_fit"]["df_unpaired"] == 20


def test_the_noise_floor_is_the_baseline_at_df_four_and_the_pooled_one_beside_it():
    cells, _ = planted()
    result = da.analyse(cells)
    assert result["sigma"]["baseline_only"]["df"] == 4
    assert result["sigma"]["pooled"]["df"] == 20
    assert (
        result["sigma"]["baseline_only"]["sigma_bpb_unbiased"]
        > result["sigma"]["baseline_only"]["sigma_bpb"]
    )


def test_seeds_needed_grows_with_sigma_and_shrinks_with_the_effect():
    assert da.seeds_needed(0.0146, 0.002) < da.seeds_needed(0.0146, 0.006)
    assert da.seeds_needed(0.0146, 0.004) < da.seeds_needed(0.0028, 0.004)
    assert da.seeds_needed(0.0020, 0.020, cap=50) is None


def test_the_underpowered_rows_say_a_null_is_uninformative():
    cells, _ = planted()
    text = da.render(da.analyse(cells))
    for name in ("D2b-i", "D5", "D1b"):
        assert name in text
    assert text.count("A null on this row is UNINFORMATIVE") == 3
    assert "UNDER-POWERED, DECLARED IN ADVANCE" in text


# ---------------------------------------------------------------------------------------
# The families, and the discipline over thirteen tasks.
# ---------------------------------------------------------------------------------------


def test_the_holm_family_is_the_six_arm_contrasts_and_nothing_else():
    cells, _ = planted()
    result = da.analyse(cells)
    assert sorted(result["holm"]["family"]) == ["D1", "D1a", "D1b", "D2b-i", "D2b-ii", "D5"]
    assert result["families"]["primary"] == ["the slope of downstream on in-loop, against 1"]
    assert not da.POST_HOC


def test_every_headline_task_carries_a_holm_adjusted_p_and_the_canary_does_not():
    cells, _ = planted()
    profile = da.per_task_slopes(cells)
    assert profile["family_size"] == 12
    for row in profile["rows"]:
        if row["in_headline"]:
            assert row["holm_adjusted_p"] is not None
            assert row["holm_adjusted_p"] >= row["p_against_one"]
        else:
            assert row["holm_adjusted_p"] is None


def test_the_thirteen_tasks_are_the_suite_and_twelve_of_them_are_the_headline():
    cells, _ = planted()
    profile = da.per_task_slopes(cells)
    assert len(profile["rows"]) == 13
    assert sum(r["in_headline"] for r in profile["rows"]) == 12
    assert [r["task"] for r in profile["rows"]] == [t.label for t in sc.SUITE_H2B]


def test_holm_makes_a_single_lucky_task_stop_looking_significant():
    """
    The whole point of the discipline: twelve raw p-values against one adjusted one. Planted with
    a common slope of one, so nothing should survive adjustment.
    """
    surviving = 0
    for s in range(60):
        cells, _ = planted(slope=1.0, residual_sd=0.004, seed=40_000 + s)
        profile = da.per_task_slopes(cells)
        surviving += any(
            r["holm_adjusted_p"] is not None and r["holm_adjusted_p"] < 0.05
            for r in profile["rows"]
        )
    assert surviving <= 6  # well under the ~12 x 5% a raw read would give


@pytest.mark.parametrize(
    "accuracy,at_chance",
    [
        (0.090, True),  # a third of a cell's own sampling noise below chance
        (0.100, True),
        (0.112, True),  # distinguishable by a z test over 2,500 items, and still unusable
        (0.350, False),  # a model that can do the format, which would need the metric revisiting
    ],
)
def test_the_canary_is_read_against_chance_and_is_never_an_outcome(accuracy, at_chance):
    """
    Set directly rather than drawn, because a draw at chance is outside two sigma one time in
    twenty and a test that fails one run in twenty teaches nobody anything.

    0.112 is the case that fixes which reading ``at_chance`` is. Over twenty-five cells the
    standard error of the mean is 0.006, so a z test calls 0.112 distinguishable from chance --
    and a five-against-five contrast on an accuracy of 0.112 still has nothing to divide by,
    which is the question the metric decision turns on.
    """
    cells, _ = planted()
    for cell in cells:
        cell.canary_accuracy = accuracy
    reading = da.canary_reading(cells)
    assert reading["available"] is True
    assert reading["chance"] == 0.10
    assert reading["at_chance"] is at_chance
    assert "never an outcome" in reading["note"]


def test_a_canary_that_reported_no_accuracy_says_so_rather_than_reading_as_zero():
    cells, _ = planted()
    for cell in cells:
        cell.canary_accuracy = None
    assert da.canary_reading(cells)["available"] is False


# ---------------------------------------------------------------------------------------
# H2b's two halves.
# ---------------------------------------------------------------------------------------


def test_h2b_has_two_halves_and_only_one_is_an_arm_ordering():
    ordering = next(h for h in da.HYPOTHESES if h.name == "D2b-i")
    sign = next(h for h in da.HYPOTHESES if h.name == "D2b-ii")
    assert ordering.declared_underpowered is True
    assert sign.declared_underpowered is False
    assert sign.comparator == "baseline"
    assert sign.predicted_sign == +1  # the published result is a DEGRADATION


@pytest.mark.parametrize(
    "arm3_shift,arm2_shift,expected",
    [
        (0.030, 0.0, "reproduced"),
        (0.0, 0.0, "not reproduced"),
        (0.030, 0.030, "both degrade"),
        (0.0, 0.030, "inverted"),
    ],
)
def test_the_published_degradation_verdict_reads_the_conjunction(arm3_shift, arm2_shift, expected):
    cells, _ = planted(slope=1.0, residual_sd=0.001, seed=12)
    for cell in cells:
        if cell.arm == "output-only":
            cell.downstream_bpb += arm3_shift + 0.0118
        if cell.arm == "faithful":
            cell.downstream_bpb += arm2_shift + 0.0146
    result = da.analyse(cells)
    assert result["published_degradation"]["verdict"] == expected


def test_only_the_sign_of_the_published_effect_transfers():
    cells, _ = planted()
    verdict = da.analyse(cells)["published_degradation"]
    assert "CLIMB accuracy" in verdict["comparability"]
    assert "No magnitude comparison" in verdict["comparability"]


# ---------------------------------------------------------------------------------------
# The synthetic path, which the measured one cannot reach.
# ---------------------------------------------------------------------------------------


def test_the_synthetic_report_is_stamped_on_every_line_a_reader_could_quote():
    cells, _ = planted(slope=0.4)
    text = da.render(da.analyse(cells, label="synthetic"))
    assert "SYNTHETIC" in text
    assert "no decision may be taken on it" in text


def test_a_provisional_read_is_stamped_and_drops_the_pairing_rather_than_inventing_one():
    """
    A hole anywhere in the grid means there is no block, and the honest response is the unpaired
    and Welch rows with the reason printed -- not a pairing over whichever seeds happen to be
    shared, which would change which cells each contrast is built from and still print.
    """
    cells, _ = planted()
    short = [c for c in cells if not (c.arm == "mhc" and c.seed == 4)]
    result = da.analyse(short, provisional=["mhc seed 4 has no document"])
    text = da.render(result)
    assert "PROVISIONAL" in text
    assert "mhc seed 4" in text
    assert result["pairing"]["available"] is False
    assert result["pairing"]["block_fit"] is None
    assert "NO PAIRING" in text
    for entry in result["contrasts"]:
        analyses = {row["analysis"] for row in entry["rows"]}
        assert analyses == {"unpaired", "welch"}
        assert sum(row["primary"] for row in entry["rows"]) == 1


def test_the_measured_path_refuses_a_partial_set_by_default(tmp_path, capsys):
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    write_documents(tmp_path, documents[:-1])
    artifact = tmp_path / "analysis.json"
    artifact.write_text(json.dumps(_artifact()))
    argv = sys.argv
    sys.argv = [
        "downstream_analysis.py",
        "--documents",
        str(tmp_path),
        "--in-loop",
        str(artifact),
    ]
    try:
        assert da.main() == 1
    finally:
        sys.argv = argv
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err
    assert "mhc seed 4" in captured.err
    assert "will not run on a partial set" in captured.err


def test_the_measured_path_needs_exactly_one_in_loop_source(tmp_path):
    documents, _ = da.synthetic_documents(in_loop=FROZEN_CELLS)
    write_documents(tmp_path, documents)
    argv = sys.argv
    sys.argv = ["downstream_analysis.py", "--documents", str(tmp_path)]
    try:
        assert da.main() == 2
    finally:
        sys.argv = argv


def test_the_measured_path_runs_end_to_end_on_documents_and_the_frozen_axis(tmp_path, capsys):
    documents, _ = da.synthetic_documents(
        slope=0.7, residual_sd=0.002, in_loop=FROZEN_CELLS, seed=2
    )
    write_documents(tmp_path, documents)
    artifact = tmp_path / "analysis.json"
    artifact.write_text(json.dumps(_artifact()))
    out = tmp_path / "out"
    argv = sys.argv
    sys.argv = [
        "downstream_analysis.py",
        "--documents",
        str(tmp_path),
        "--in-loop",
        str(artifact),
        "--out",
        str(out),
    ]
    try:
        assert da.main() == 0
    finally:
        sys.argv = argv
    capsys.readouterr()
    written = json.loads((out / "downstream.json").read_text())
    assert written["label"] == "measured"
    assert written["pre_registered_on"] == da.PRE_REGISTERED_ON
    assert written["regression"]["pooled"]["n"] == 25
    assert (out / "downstream.txt").exists()


def test_the_self_test_passes():
    assert da.self_test(replicates=40) == 0


# ---------------------------------------------------------------------------------------
# The document, because a pre-registration that lives only in code is not one.
# ---------------------------------------------------------------------------------------


def _plan_text():
    with open(PLAN) as handle:
        return handle.read()


def test_the_plan_carries_the_downstream_pre_registration_and_dates_it():
    text = _plan_text()
    assert "## The downstream pre-registration of 2026-08-12" in text
    assert da.PRE_REGISTERED_ON in text


def test_the_plan_says_it_was_written_before_the_job_was_submitted():
    section = _plan_text().split("## The downstream pre-registration of 2026-08-12")[1]
    section = section.split("\n## ")[0]
    lowered = section.lower()
    assert "before the scoring job was submitted" in lowered
    assert "no downstream data existed" in lowered


def test_the_plan_names_the_module_that_implements_it():
    section = _plan_text().split("## The downstream pre-registration of 2026-08-12")[1]
    section = section.split("\n## ")[0]
    assert "downstream_analysis.py" in section


def test_the_plan_declares_the_three_underpowered_contrasts_by_name():
    section = _plan_text().split("## The downstream pre-registration of 2026-08-12")[1]
    section = section.split("\n## ")[0]
    for name in ("D2b-i", "D5", "D1b"):
        assert name in section
    assert "under-powered" in section.lower()
