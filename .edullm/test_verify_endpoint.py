#!/usr/bin/env python3
"""Tests for ``verify_endpoint.py``.

Two kinds. The first exercise the statistics against inputs whose answers can be worked out by
hand, so that a wrong pooled sigma fails here rather than in a report. The second read the
frozen document and assert the properties the verification actually turned on -- that the seeds
are matched, that nothing is averaged across horizons, that the units constant is the one every
cell was launched with. Neither kind reaches the network.

The frozen document is a snapshot of live runs and the ``mhc`` arm was still training when it
was taken, so nothing here asserts a value for that arm. What is asserted is that the code
*refuses* to put it in a step-6,000 contrast.
"""

import math
import os
import statistics
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import verify_endpoint as ve  # noqa: E402


@pytest.mark.parametrize(
    "df, expected",
    [(1, 0.7979), (2, 0.8862), (4, 0.9400), (8, 0.9693), (12, 0.9794), (29, 0.9914)],
)
def test_c4_matches_the_published_table(df, expected):
    """The published table is indexed by sample size; ``c4`` here is indexed by df, so n - 1."""
    assert ve.c4(df) == pytest.approx(expected, abs=5e-5)


def test_c4_is_the_wrong_constant_if_the_seed_count_is_used_instead_of_the_df():
    """The trap the report has to avoid: a df-12 pool is not corrected by a df-4 factor."""
    pooled = 0.00355719
    assert pooled / ve.c4(12) == pytest.approx(0.003632, abs=1e-6)
    assert pooled / ve.c4(4) == pytest.approx(0.003784, abs=1e-6)


def test_nats_conversion_is_ln2_times_bytes_per_token():
    assert ve.NATS_PER_BPB == pytest.approx(math.log(2.0) * 4.57)
    assert ve.to_nats(1.0) == pytest.approx(3.167682615159, abs=1e-9)
    assert ve.to_nats(0.5) == pytest.approx(ve.to_nats(1.0) / 2)


def test_implied_bytes_per_token_inverts_the_trainer():
    for bpt in (2.0, 4.57, 8.125):
        bpb = 3.0 / (math.log(2.0) * bpt)
        assert ve.implied_bytes_per_token(3.0, bpb) == pytest.approx(bpt)


def test_unweighted_endpoint_is_unweighted_and_refuses_a_short_table():
    assert ve.unweighted_endpoint({s: 0.25 for s in ve.SOURCES}) == pytest.approx(0.25)
    graded = {s: float(i) for i, s in enumerate(ve.SOURCES)}
    assert ve.unweighted_endpoint(graded) == pytest.approx(3.0)
    with pytest.raises(KeyError):
        ve.unweighted_endpoint({s: 1.0 for s in ve.SOURCES[:-1]})


def test_pooled_sigma_on_two_identical_groups_is_the_within_group_sd():
    sigma, df = ve.pooled_sigma([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])
    assert sigma == pytest.approx(1.0)
    assert df == 4


def test_pooled_sigma_ignores_a_singleton_arm_rather_than_dividing_by_zero():
    sigma, df = ve.pooled_sigma([[1.0, 2.0, 3.0], [7.0]])
    assert sigma == pytest.approx(1.0)
    assert df == 2


def test_blocked_residual_of_a_perfectly_additive_table_is_zero():
    table = {"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 4.0, 5.0, 6.0], "c": [0.0, 1.0, 2.0, 3.0]}
    sigma, df = ve.blocked_residual_sigma(table)
    assert sigma == pytest.approx(0.0, abs=1e-12)
    assert df == (3 - 1) * (4 - 1)


def test_blocked_residual_needs_a_complete_table():
    with pytest.raises(ValueError):
        ve.blocked_residual_sigma({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]})


def test_blocked_residual_equals_the_pooled_sigma_when_the_seeds_carry_no_signal():
    """With no seed main effect the blocking buys nothing and must not pretend otherwise."""
    table = {"a": [0.0, 1.0, -1.0, 2.0, -2.0], "b": [5.0, 4.0, 6.0, 3.0, 7.0]}
    pooled, _ = ve.pooled_sigma(list(table.values()))
    blocked, _ = ve.blocked_residual_sigma(table)
    assert blocked > pooled  # the seed df is spent for nothing, so the residual is larger


def test_seed_effect_f_is_large_when_the_seeds_are_the_whole_story():
    table = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.1, 2.1, 3.1, 4.1, 5.1]}
    f, p = ve.seed_effect_f(table)
    assert f > 100 and p < 0.001


def test_intervals_all_share_the_point_estimate_and_only_the_width_differs():
    table = {
        "t": [1.0, 1.2, 0.9, 1.1, 1.3],
        "c": [2.0, 2.1, 2.3, 1.9, 2.2],
        "o": [1.5, 1.6, 1.4, 1.7, 1.55],
    }
    got = ve.intervals(table, "t", "c")
    assert [iv.method for iv in got] == ["pooled", "blocked", "paired", "welch"]
    point = statistics.fmean(table["t"]) - statistics.fmean(table["c"])
    for iv in got:
        assert iv.point == pytest.approx(point)
        assert iv.lo < iv.point < iv.hi
        assert iv.half_width > 0


def test_interval_excludes():
    iv = ve.Interval("x", -0.046, 0.004, 0.003, 8)
    assert iv.excludes(0.0)
    assert iv.excludes(-0.030)
    assert not iv.excludes(-0.046)


def test_slope_per_doubling_is_positive_when_the_loss_falls():
    steps = [3000, 4000, 6000]
    values = [3.0, 3.0 - math.log2(4 / 3), 3.0 - 1.0]
    assert ve.slope_per_doubling(steps, values) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def doc():
    if not os.path.exists(ve.FROZEN):
        pytest.skip(f"{ve.FROZEN} has not been fetched")
    return ve.load(ve.FROZEN)


def test_every_arm_has_five_cells_and_the_same_five_seeds(doc):
    """The claim rests on five distinct seeds per arm. Read it off each cell's own config."""
    by_arm = {}
    for cell in doc["cells"].values():
        by_arm.setdefault(cell["arm"], []).append((cell["init_seed"], cell["data_loader_seed"]))
    assert set(by_arm) == set(ve.SUBMISSIONS)
    for arm, seeds in by_arm.items():
        assert len(seeds) == ve.N_SEEDS, arm
        assert len(set(seeds)) == ve.N_SEEDS, f"{arm} has a repeated seed: {seeds}"
    assert len({frozenset(s) for s in by_arm.values()}) == 1, "arms are not seed-matched"


def test_every_cell_agrees_on_the_bytes_per_token_constant(doc):
    for run_id, cell in doc["cells"].items():
        for step, entry in cell["evals"].items():
            got = ve.implied_bytes_per_token(entry["dclm CE loss"], entry["dclm"])
            assert got == pytest.approx(ve.BYTES_PER_TOKEN, abs=1e-6), f"{run_id} at {step}"


def test_the_three_claimed_arms_are_complete_and_mhc_is_not(doc):
    assert ve.complete_arms(doc) == ["baseline", "faithful", "output-only"]


def test_a_partial_arm_cannot_enter_a_contrast_at_the_horizon(doc):
    """Mixing horizons is the failure mode; the table must raise rather than substitute."""
    with pytest.raises(ValueError, match="no eval at step 6000"):
        ve.endpoint_table(doc, ["mhc"], ve.FINAL_STEP)
    partial = ve.endpoint_table(doc, ["mhc"], 3500)
    assert len(partial["mhc"]) == ve.N_SEEDS


def test_the_endpoint_table_is_ordered_by_cell_index(doc):
    table = ve.endpoint_table(doc, ["baseline"], ve.FINAL_STEP)
    direct = []
    for index in range(ve.N_SEEDS):
        cell = doc["cells"][f"{ve.SUBMISSIONS['baseline']}-cell-{index}"]
        direct.append(ve.to_nats(ve.unweighted_endpoint(cell["evals"][str(ve.FINAL_STEP)])))
    assert table["baseline"] == pytest.approx(direct)


def test_the_frozen_baseline_sigma_still_matches_the_noise_floor_artifact(doc):
    """``noise-floor-skip-step.json`` was frozen off these same five cells. It must agree."""
    import json

    with open(os.path.join(_HERE, "noise-floor-skip-step.json")) as handle:
        floor = json.load(handle)
    table = ve.endpoint_table(doc, ["baseline"], ve.FINAL_STEP)
    assert statistics.stdev(table["baseline"]) == pytest.approx(floor["sigma_nats"], rel=1e-6)
    assert floor["sigma_nats_unbiased"] == pytest.approx(floor["sigma_nats"] / ve.c4(4), rel=1e-6)


def test_h1_replicates_and_no_estimator_puts_zero_or_the_published_value_inside(doc):
    """The headline. The point estimate is one number; the interval depends on the model."""
    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    got = ve.intervals(table, "faithful", "baseline")
    for iv in got:
        assert iv.point == pytest.approx(-0.04609, abs=5e-6)
        assert iv.excludes(0.0)
        assert iv.excludes(-0.030), f"{iv.method} admits the published 1B value"
    widths = {iv.method: iv.half_width for iv in got}
    assert widths["blocked"] < widths["pooled"], "blocking should be the narrowest here"
    assert widths["pooled"] / widths["blocked"] == pytest.approx(1.346, abs=0.01)


def test_the_quoted_pooled_sigma_does_not_generate_the_quoted_interval(doc):
    """
    The one arithmetic disagreement, pinned so it cannot be lost.

    The report quotes sigma 0.00356 at df 12 and an interval of half-width 0.00365. Those are
    not the same estimator: 0.00356 at df 12 gives 0.00490, and 0.00365 is the arm-plus-seed
    residual at df 8. Both are defensible; quoting one and printing the other is not.
    """
    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    pooled = next(iv for iv in ve.intervals(table, "faithful", "baseline") if iv.method == "pooled")
    blocked = next(
        iv for iv in ve.intervals(table, "faithful", "baseline") if iv.method == "blocked"
    )
    assert pooled.sigma == pytest.approx(0.00356, abs=5e-6)
    assert pooled.df == 12
    assert pooled.half_width == pytest.approx(0.004902, abs=5e-6)
    assert blocked.sigma == pytest.approx(0.002497, abs=5e-6)
    assert blocked.df == 8
    assert blocked.half_width == pytest.approx(0.003642, abs=5e-6)


def test_the_treatment_arms_are_noisier_but_not_provably_so(doc):
    """A treatment that raises seed variance would be a finding. At five seeds it is not one."""
    from scipy import stats as sps

    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    base = statistics.variance(table["baseline"])
    for arm in ("faithful", "output-only"):
        ratio = statistics.variance(table[arm]) / base
        assert 4.0 < ratio < 5.5
        p = 2 * min(sps.f.sf(ratio, 4, 4), sps.f.cdf(ratio, 4, 4))
        assert p > 0.05, f"{arm} variance inflation is significant after all: p={p}"
    assert sps.bartlett(*[table[a] for a in ve.complete_arms(doc)]).pvalue > 0.05


def test_the_arms_improve_at_indistinguishable_rates_over_the_last_doubling(doc):
    from scipy import stats as sps

    arms = ve.complete_arms(doc)
    window = [s for s in range(3000, ve.FINAL_STEP + 1, ve.EVAL_INTERVAL)]
    per_arm = {
        a: [
            ve.slope_per_doubling(window, [ve.endpoint_table(doc, [a], s)[a][i] for s in window])
            for i in range(ve.N_SEEDS)
        ]
        for a in arms
    }
    assert sps.f_oneway(*[per_arm[a] for a in arms]).pvalue > 0.05
    for arm in arms:
        assert statistics.fmean(per_arm[arm]) == pytest.approx(0.155, abs=0.002)


def test_the_intervention_is_behind_at_equal_wall_clock(doc):
    """Un-claimed and load-bearing: the gain costs more compute than it returns."""
    rates = {}
    for arm in ("baseline", "faithful"):
        v = [ve.step_seconds(c) for c in doc["cells"].values() if c["arm"] == arm]
        rates[arm] = statistics.fmean([x for x in v if x is not None])
    assert rates["faithful"] / rates["baseline"] == pytest.approx(1.72, abs=0.02)
    reached = ve.FINAL_STEP * rates["baseline"] / rates["faithful"]
    assert 3000 < reached < 3500
    base = statistics.fmean(ve.endpoint_table(doc, ["baseline"], ve.FINAL_STEP)["baseline"])
    for step in (3000, 3500):
        ahead = statistics.fmean(ve.endpoint_table(doc, ["faithful"], step)["faithful"]) - base
        assert ahead > 0, f"faithful@{step} is already ahead of baseline@6000"


def test_the_effect_is_in_every_source_and_no_single_source_carries_it(doc):
    base_arm = [c for c in doc["cells"].values() if c["arm"] == "baseline"]
    hc_arm = [c for c in doc["cells"].values() if c["arm"] == "faithful"]
    relative = {}
    for source in ve.SOURCES:
        b = statistics.fmean([c["evals"][str(ve.FINAL_STEP)][source] for c in base_arm])
        f = statistics.fmean([c["evals"][str(ve.FINAL_STEP)][source] for c in hc_arm])
        assert f < b, source
        relative[source] = (f - b) / b
    assert max(relative.values()) < -0.017
    assert min(relative.values()) > -0.028
    total = ve.to_nats(statistics.fmean(list(relative.values())))
    assert total < 0
