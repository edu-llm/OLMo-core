#!/usr/bin/env python3
"""Tests for ``verify_endpoint.py``.

Two kinds. The first exercise the statistics against inputs whose answers can be worked out by
hand, so that a wrong pooled sigma fails here rather than in a report. The second read the
frozen document and assert the properties the verification actually turned on -- that the seeds
are matched, that nothing is averaged across horizons, that the units constant is the one every
cell was launched with. Neither kind reaches the network.

The frozen document was refrozen on 2026-08-12 with all five arms finished at step 6,000, so
the tests that used to assert that an unfinished ``mhc`` was kept out of a contrast now assert
that property against a document truncated on purpose. Nothing here reaches a network, and the
refusal is still the thing being tested; it just no longer has a live incomplete arm to test on.
"""

import copy
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


def test_every_arm_of_the_tranche_is_complete_at_the_horizon(doc):
    """All five arms landed on 2026-08-12, so the horizon is shared and nothing is truncated."""
    assert ve.complete_arms(doc) == [
        "baseline",
        "faithful",
        "output-only",
        "no-output-init",
        "mhc",
    ]
    for arm in ve.complete_arms(doc):
        assert len(ve.endpoint_table(doc, [arm], ve.FINAL_STEP)[arm]) == ve.N_SEEDS


def test_a_partial_arm_cannot_enter_a_contrast_at_the_horizon(doc):
    """
    Mixing horizons is the failure mode; the table must raise rather than substitute.

    Every real arm now reaches 6,000, so the refusal is exercised against a copy of the document
    with one arm's last evaluation removed. That is the situation the guard exists for and it
    must not need a live unfinished arm to stay tested.
    """
    truncated = copy.deepcopy(doc)
    for name, cell in truncated["cells"].items():
        if cell["arm"] == "mhc":
            cell["evals"].pop(str(ve.FINAL_STEP), None)
    with pytest.raises(ValueError, match="no eval at step 6000"):
        ve.endpoint_table(truncated, ["mhc"], ve.FINAL_STEP)
    assert "mhc" not in ve.complete_arms(truncated)
    earlier = ve.endpoint_table(truncated, ["mhc"], 3500)
    assert len(earlier["mhc"]) == ve.N_SEEDS


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
    # At three arms this ratio was 1.346. Arm 4 landed with 4.25x the baseline's seed variance,
    # which inflates the pooled sigma without inflating the blocked residual by as much, so the
    # two estimators moved together and the gap between them closed to 9%.
    assert widths["pooled"] / widths["blocked"] == pytest.approx(1.092, abs=0.01)


def test_h1_decomposes_into_a_mechanism_that_replicates_and_a_rescale_that_does_not(doc):
    """
    H1a and H1b, by the route that shares no code with ``analysis.py``.

    H1 bundles hyper-connections with the sqrt(n) output-module rescale, so on its own it cannot
    say which one it measured. Arm 4 is the mechanism without the rescale, and the two contrasts
    that separate them have to agree with the primary analysis to the digit or one of them is
    wrong. H1a carries 86.5% of H1 and clears its own interval; H1b spans zero. That is the whole
    of the attribution and it is asserted here rather than only computed there.
    """
    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    h1 = ve.intervals(table, "faithful", "baseline")[0].point
    h1a = ve.intervals(table, "no-output-init", "baseline")
    h1b = ve.intervals(table, "faithful", "no-output-init")

    for iv in h1a:
        assert iv.point == pytest.approx(-0.039891, abs=5e-6)
        assert iv.excludes(0.0), f"{iv.method} cannot tell the mechanism from nothing"
    assert h1a[0].point / h1 == pytest.approx(0.865, abs=0.002)

    for iv in h1b:
        assert iv.point == pytest.approx(-0.006202, abs=5e-6)
    # The decomposition is exact by construction, and a rounding error in either arm breaks it.
    assert h1a[0].point + h1b[0].point == pytest.approx(h1, abs=1e-12)
    # Welch is the primary estimator once Bartlett rejects, and it is the one that spans zero.
    welch = next(iv for iv in h1b if iv.method == "welch")
    assert not welch.excludes(0.0), "H1b must not read as a resolved effect; its interval spans 0"
    assert welch.df == pytest.approx(5.78, abs=0.01)


def test_the_quoted_pooled_sigma_does_not_generate_the_quoted_interval(doc):
    """
    The one arithmetic disagreement, pinned so it cannot be lost.

    The pooled sigma is over arms and the blocked one is the arm-plus-seed residual with the seed
    effect taken out, so they are different numbers and the interval built from one cannot be
    reconstructed from the other. Quoting one at the top of a report and printing the other
    beside the estimate is the error this pins, and it is a property of the two estimators rather
    than of any particular tranche -- so it is asserted as the relation, and the values are
    asserted beside it for whatever the tranche currently is.
    """
    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    pooled = next(iv for iv in ve.intervals(table, "faithful", "baseline") if iv.method == "pooled")
    blocked = next(
        iv for iv in ve.intervals(table, "faithful", "baseline") if iv.method == "blocked"
    )
    assert pooled.sigma != pytest.approx(blocked.sigma, rel=1e-3)
    assert pooled.half_width != pytest.approx(blocked.half_width, rel=1e-3)
    assert pooled.df > blocked.df
    # Five arms at five seeds: k(n-1) = 20 pooled against (k-1)(n-1) = 16 blocked.
    assert pooled.sigma == pytest.approx(0.004663, abs=5e-6)
    assert pooled.df == 20
    assert pooled.half_width == pytest.approx(0.006152, abs=5e-6)
    assert blocked.sigma == pytest.approx(0.004203, abs=5e-6)
    assert blocked.df == 16
    assert blocked.half_width == pytest.approx(0.005635, abs=5e-6)


def test_only_arm_four_raises_seed_variance_and_it_is_the_whole_of_the_bartlett_rejection(doc):
    """
    Where the variance story stands at five arms, which is not where it stood at three.

    The claim that hyper-connections raise seed variance was withdrawn on 2026-08-10 and stays
    withdrawn: ``faithful`` and ``output-only`` sit at twice the baseline's sd and neither is
    significant at five seeds. What is new is ``no-output-init``, at 4.25x the baseline's sd and
    p = 0.016, which is the entire reason Bartlett now rejects -- drop that one arm and the
    remaining four are homogeneous at p = 0.26.

    This matters beyond bookkeeping. Bartlett rejecting is what commits the pre-registered plan
    to Welch everywhere, and arm 4's spread is what inflates H1b's MDE to 0.0144 nats, which is
    why H1b cannot resolve an effect its own point estimate puts at 0.0062.
    """
    from scipy import stats as sps

    table = ve.endpoint_table(doc, ve.complete_arms(doc), ve.FINAL_STEP)
    base = statistics.variance(table["baseline"])

    def two_sided(arm):
        ratio = statistics.variance(table[arm]) / base
        return ratio, 2 * min(sps.f.sf(ratio, 4, 4), sps.f.cdf(ratio, 4, 4))

    for arm in ("faithful", "output-only"):
        ratio, p = two_sided(arm)
        assert 4.0 < ratio < 5.5
        assert p > 0.05, f"{arm} variance inflation is significant after all: p={p}"
    ratio, p = two_sided("no-output-init")
    assert ratio == pytest.approx(18.04, abs=0.1)
    assert p < 0.05, "arm 4's spread is the one that is significant; it must not go unremarked"
    _, p_mhc = two_sided("mhc")
    assert p_mhc > 0.5, "mhc is as tight as the baseline, so the spread is not about lanes"

    assert sps.bartlett(*[table[a] for a in ve.complete_arms(doc)]).pvalue < 0.05
    without = [a for a in ve.complete_arms(doc) if a != "no-output-init"]
    assert sps.bartlett(*[table[a] for a in without]).pvalue > 0.05


def test_every_arm_but_mhc_improves_at_the_baseline_rate_over_the_last_doubling(doc):
    """
    Whether an arm's advantage is a level shift or a change of slope, at five arms.

    At three arms every slope was indistinguishable and 'parallel slopes' held across the family.
    It no longer does: the omnibus F rejects at p = 0.0007, and it is entirely ``mhc``, which
    improves at 0.1516 nats per doubling against the baseline's 0.1553 (p = 0.006). So mHC is not
    only behind HC at the horizon, it is falling further behind the baseline as training goes on.

    The three arms H1 and H1b turn on are all still indistinguishable from the baseline, so the
    level-shift reading survives exactly where it is used and fails where it is not.
    """
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
    assert sps.f_oneway(*[per_arm[a] for a in arms]).pvalue < 0.05
    assert statistics.fmean(per_arm["mhc"]) == pytest.approx(0.1516, abs=0.0005)
    assert sps.ttest_ind(per_arm["mhc"], per_arm["baseline"], equal_var=False).pvalue < 0.01
    for arm in ("faithful", "output-only", "no-output-init"):
        p = sps.ttest_ind(per_arm[arm], per_arm["baseline"], equal_var=False).pvalue
        assert p > 0.05, f"{arm} no longer improves at the baseline's rate: p={p}"
    for arm in arms:
        assert statistics.fmean(per_arm[arm]) == pytest.approx(0.154, abs=0.003)


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
