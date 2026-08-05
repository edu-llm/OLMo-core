"""Tests for sigma.py.

TESTING DISCIPLINE, from the repo memory ``test-must-call-not-recompute``: a test that re-derives
the code's own formula passes when the code changes. So every load-bearing test here pins against an
**externally derived number** -- one published in ``KDA/HANDOFF.md`` or ``R3-statistics.md`` by
someone who was not this code -- and the ``test_negative_control_*`` tests demonstrate that the
guard can actually fail by mutating an input rather than the source.

The three external anchors:

* ``KDA/HANDOFF.md:452-457``  -- 184 seeds/arm (normal approx), rho=0.5, 10 pp, sigma_within=48.4 pp
* ``R3-statistics.md`` F8     -- the full required-n table: 1110/279/126, 556/141/64, 1473/370/166,
                                 738/186/84 (exact noncentral-t)
* ``R3-statistics.md`` F5e    -- priced miss rates 0.949 / 0.930 / 0.629 / 0.944 / 0.860
* ``R3-statistics.md`` F3     -- sign-test null probabilities 0.1875 (>=4 of 5) and 0.0312 (5 of 5)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigma as S  # noqa: E402


# ======================================================================================
# THE MANDATED VALIDATION: does the power code reproduce 184?
# ======================================================================================


def test_reproduces_repo_published_184():
    """
    ``KDA/HANDOFF.md:452-457`` publishes **184 seeds/arm** for rho=0.5, MDE 10 pp,
    sigma_within=48.4 pp. R3's closed form: ``(2.80159 * 48.4 / 10) ** 2 = 183.87``.

    This is the validation that the power code is right. It is pinned to a number derived
    independently of this file.
    """
    v = S.validate_against_repo_184()
    assert v["s_delta_pp"] == pytest.approx(48.4), "rho=0.5 must give s_delta == sigma exactly"
    assert v["z_sum"] == pytest.approx(2.8015852, abs=1e-6)
    assert v["closed_form"] == pytest.approx(183.865, abs=0.01)
    assert v["rel_err_closed_form"] < 0.0008, "closed form must match 184 to better than 0.08%"
    assert v["normal_n"] == 184, f"normal-approx n must be the repo's 184, got {v['normal_n']}"
    assert v["matches_repo_184"] is True


def test_exact_nct_gives_186_and_the_df_penalty_is_two_seeds():
    """
    R3 F8's table publishes **186** in the (sigma=48.4, rho=0.5, 10 pp) cell, two more than the
    normal approximation's 184. The gap IS the t-distribution's df penalty, and it must point in the
    direction where the approximation under-states n -- otherwise the fallback would make an
    underpowered design look adequate.
    """
    if not S.HAVE_SCIPY:
        pytest.skip("scipy unavailable; exact path not exercised")
    v = S.validate_against_repo_184()
    assert v["exact_n"] == 186, f"exact noncentral-t must give 186, got {v['exact_n']}"
    assert v["matches_r3_186"] is True
    assert v["df_penalty_seeds"] == 2
    assert v["exact_n"] > v["normal_n"], "the normal approximation must UNDER-state required n"
    assert v["method"] == "exact noncentral-t"


def test_reproduces_r3_f8_required_n_table():
    """
    R3 F8's full required-n table (exact noncentral-t, paired, 80 % power, alpha=0.05 two-sided)::

        sigma  rho  s_delta    5 pp   10 pp   15 pp
        42.0   0.0   59.40    1,110     279     126
        42.0   0.5   42.00      556     141      64
        48.4   0.0   68.45    1,473     370     166
        48.4   0.5   48.40      738     186      84

    Pinned to R3's derivation, not to ours.
    """
    if not S.HAVE_SCIPY:
        pytest.skip("scipy unavailable; the table is defined on the exact path")
    expected = {
        (42.0, 0.0): (59.397, 1110, 279, 126),
        (42.0, 0.5): (42.000, 556, 141, 64),
        (48.4, 0.0): (68.448, 1473, 370, 166),
        (48.4, 0.5): (48.400, 738, 186, 84),
    }
    rows = S.required_n_table()
    assert len(rows) == 4
    for row in rows:
        sd_exp, n5, n10, n15 = expected[(row["sigma"], row["rho"])]
        assert row["s_delta"] == pytest.approx(sd_exp, abs=1e-3)
        assert row["n_5pp"] == n5, f"sigma={row['sigma']} rho={row['rho']} 5pp"
        assert row["n_10pp"] == n10, f"sigma={row['sigma']} rho={row['rho']} 10pp"
        assert row["n_15pp"] == n15, f"sigma={row['sigma']} rho={row['rho']} 15pp"
        assert row["method"] == "exact noncentral-t"


def test_exp2_shortfall_is_two_orders_of_magnitude():
    """R3 F8's headline: the proposal plans 5 seeds to detect 5 pp; it needs ~556-738."""
    for sigma, expected in ((42.0, 556), (48.4, 738)):
        n = S.required_n(S.s_delta_from_sigma(sigma, 0.5), 5.0)["n"]
        assert n == expected
        assert n / 5 > 100, "the shortfall against n=5 must be >100x"


def test_negative_control_wrong_z_sum_breaks_the_184_check():
    """
    The guard must be ABLE to fail. Mutate the inputs (not the source) so the required-n moves, and
    assert the 184 check no longer holds. A test that has never failed is not known to work.
    """
    # Right answer at the published anchor.
    assert S._required_n_normal(48.4, 10.0, alpha=0.05, power=0.80, sided=2) == 184
    # One-sided alpha (the classic error) drops it well below 184.
    one_sided = S._required_n_normal(48.4, 10.0, alpha=0.05, power=0.80, sided=1)
    assert one_sided < 184, "a one-sided alpha must give a SMALLER n; if not, the code ignores it"
    assert one_sided == 145
    # 90 % power raises it above 184.
    assert S._required_n_normal(48.4, 10.0, alpha=0.05, power=0.90, sided=2) > 184
    # Forgetting the sqrt(2(1-rho)) factor -- using sigma where s_delta belongs at rho=0 -- halves n.
    assert S._required_n_normal(S.s_delta_from_sigma(48.4, 0.0), 10.0) == 368
    assert 368 != 184


# ======================================================================================
# s_delta identity and pooled sigma
# ======================================================================================


def test_s_delta_identity():
    """``s_delta = sigma*sqrt(2(1-rho))``: equals sigma at rho=0.5, sqrt(2)*sigma at rho=0."""
    assert S.s_delta_from_sigma(48.4, 0.5) == pytest.approx(48.4)
    assert S.s_delta_from_sigma(42.0, 0.0) == pytest.approx(42.0 * math.sqrt(2))
    assert S.s_delta_from_sigma(42.0, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        S.s_delta_from_sigma(42.0, 1.5)


def test_pooled_sigma_is_df_weighted_not_a_mean_of_sds():
    """
    Pooling must weight by df. A mean of SDs is biased low, and with unequal seed counts the two
    answers differ -- so this test would fail if the implementation averaged SDs.
    """
    cells = [
        _cell(sigma_acc=0.10, n=3),
        _cell(sigma_acc=0.50, n=11),
    ]
    got = S.pooled_sigma(cells)["pooled_sigma"]
    df_weighted = math.sqrt((2 * 0.10**2 + 10 * 0.50**2) / 12)
    assert got == pytest.approx(df_weighted)
    assert got != pytest.approx((0.10 + 0.50) / 2), "must not be a plain mean of SDs"
    assert S.pooled_sigma(cells)["total_df"] == 12


def test_pooled_sigma_skips_single_seed_cells():
    """A cell with n=1 has no measured spread; treating its 0.0 as an SD would bias the pool down."""
    cells = [_cell(sigma_acc=0.40, n=6), _cell(sigma_acc=float("nan"), n=1)]
    r = S.pooled_sigma(cells)
    assert r["pooled_sigma"] == pytest.approx(0.40)
    assert r["n_cells"] == 1


def _cell(*, sigma_acc: float, n: int) -> S.CellEndpoints:
    return S.CellEndpoints(
        arm="static", topology="allliv", kernel_size=3, config="N512_D64", num_pairs=64,
        n_seeds=n, floor=1 / 64, success_rate=0.0, median_accuracy=0.2, median_over_floor=12.8,
        mean_accuracy=0.2, sigma_accuracy=sigma_acc, mean_nll_query=3.0, sigma_nll_query=0.5,
        per_seed_accuracy=tuple([0.2] * n), per_seed_nll=tuple([3.0] * n),
        per_seed_seeds=tuple(range(n)), bimodal=False, n_mid=n,
    )


# ======================================================================================
# Sign-test operating characteristics (R3 F3 / F8)
# ======================================================================================


def test_sign_test_floors_are_0p1875_and_0p0312():
    """
    R3 F3's table: ``>=4 of 5`` has exact null P = 6/32 = 0.1875 and CANNOT reach alpha=0.05;
    ``5 of 5`` gives 1/32 = 0.03125 and can.
    """
    four = S.sign_criterion_oc(4, 5, s_delta=42.0)
    assert four["p_null"] == pytest.approx(0.1875)
    assert four["p_null"] == pytest.approx(6 / 32)
    assert four["can_reach_alpha_05"] is False

    five = S.sign_criterion_oc(5, 5, s_delta=42.0)
    assert five["p_null"] == pytest.approx(0.03125)
    assert five["p_null"] == pytest.approx(1 / 32)
    assert five["can_reach_alpha_05"] is True

    # F3 also prices the n=3 and n=2 floors.
    assert S.sign_criterion_oc(3, 3, s_delta=42.0)["p_null"] == pytest.approx(0.125)
    assert S.sign_criterion_oc(2, 2, s_delta=42.0)["p_null"] == pytest.approx(0.25)


def test_four_of_five_has_24_to_38_percent_power_at_the_hypothesized_effect():
    """
    R3 F8's measured operating characteristics of the proposal's own criterion::

        true effect  s_delta  P(one seed +)  P(>=4 of 5) = power
            8 pp      42.0        0.576           0.296
           12 pp      42.0        0.612           0.359
            5 pp      48.4        0.541           0.243
           15 pp      48.4        0.622           0.375

    24-38 % power while firing 18.75 % of the time under the null: a likelihood ratio of ~1.5-2:1.
    """
    for s_delta, effect, p_one, power in (
        (42.0, 8.0, 0.576, 0.296),
        (42.0, 12.0, 0.612, 0.359),
        (48.4, 5.0, 0.541, 0.243),
        (48.4, 15.0, 0.622, 0.375),
    ):
        oc = S.sign_criterion_oc(4, 5, s_delta=s_delta, effects=(effect,))
        row = oc["rows"][0]
        assert row["p_one_seed_positive"] == pytest.approx(p_one, abs=0.001)
        assert row["power"] == pytest.approx(power, abs=0.001)
        assert 0.24 <= row["power"] <= 0.38 or effect == 15.0


def test_negative_control_a_pass_criterion_with_no_power_is_visible_as_such():
    """Under the null, power must equal the null firing rate exactly -- not more."""
    oc = S.sign_criterion_oc(4, 5, s_delta=42.0, effects=(0.0,))
    assert oc["rows"][0]["power"] == pytest.approx(oc["p_null"])
    assert oc["rows"][0]["p_one_seed_positive"] == pytest.approx(0.5)


# ======================================================================================
# Priced miss rates (R3 F5e)
# ======================================================================================


def test_reproduces_r3_f5e_priced_miss_rates():
    """
    R3 F5e, alpha=0.05, two-sided, rho=0.5 (Exp-2 rows at sigma=42.0, Exp-4 rows at 48.4)::

        clause                                   n   regression   P(missed)
        Exp-2 (3) "control avg not down >2 pts"   5      2 pp        0.949
        "                                        5     10 pp        0.930
        "                                        5     40 pp        0.629
        Exp-4 (5) "downstream averages"          3     10 pp        0.944
        "                                        3     40 pp        0.860
    """
    expected = [0.949, 0.930, 0.629, 0.944, 0.860]
    rows = S.miss_rate_table()
    assert len(rows) == 5
    for row, want in zip(rows, expected):
        assert row["p_missed"] == pytest.approx(want, abs=0.001), row


def test_a_forty_point_regression_passes_63_percent_of_the_time():
    """
    The finding, stated as an assertion: **a true 40-point regression on the control tasks passes
    Exp-2's n=5 gate 63 % of the time.** A clause that misses a 40-point regression that often is
    decoration -- either buy the seeds or delete it and say the study is silent on it.
    """
    p = S.price_miss_rate(5, 42.0, 40.0)
    assert p == pytest.approx(0.629, abs=0.001)
    assert p > 0.5, "the clause is more likely to miss than to catch a 40 pp regression"
    # And buying seeds is the only fix: the miss rate must fall monotonically in n.
    ns = [5, 20, 50, 100]
    misses = [S.price_miss_rate(n, 42.0, 40.0) for n in ns]
    assert misses == sorted(misses, reverse=True)
    assert misses[-1] < 0.01


def test_memorize_regression_is_preregistered_at_6p1_points():
    """SPEC Sec 4.6: 0.856 static -> 0.795 dynamic = 6.1 points down. Expect it; it is not a bug."""
    assert S.MEMORIZE_STATIC == 0.856
    assert S.MEMORIZE_DYNAMIC == 0.795
    assert S.MEMORIZE_EXPECTED_REGRESSION_PP == pytest.approx(6.1, abs=0.01)
    # And a >2-point clause at n=5 would miss a regression of exactly this size ~93 % of the time.
    assert S.price_miss_rate(5, 42.0, S.MEMORIZE_EXPECTED_REGRESSION_PP) > 0.9


# ======================================================================================
# Paired stats, CIs, verdicts
# ======================================================================================


def _rec(arm: str, seed: int, acc: float, nll: float = 3.0) -> S.SeedRecord:
    return S.SeedRecord(
        arm=arm, topology="allliv", kernel_size=3, config="N512_D64", seed=seed,
        accuracy=acc, nll_query=nll, num_pairs=64,
    )


def test_paired_stats_backs_out_the_realized_rho():
    """
    rho must be BACKED OUT of the measured s_delta, not assumed. With two perfectly correlated arms
    (a constant offset) s_delta = 0 and rho = 1.
    """
    a = [_rec("dynamic", i, v) for i, v in enumerate([0.10, 0.30, 0.50, 0.70])]
    b = [_rec("static", i, v - 0.05) for i, v in enumerate([0.10, 0.30, 0.50, 0.70])]
    st = S.paired_stats(a, b)
    assert st.n_pairs == 4
    assert st.unit == "pp" and st.scale == 100.0, "accuracy must be reported in percentage points"
    assert st.mean_delta == pytest.approx(5.0), "0.05 fraction == 5.0 pp"
    assert st.s_delta == pytest.approx(0.0, abs=1e-9)
    assert st.rho_realized == pytest.approx(1.0)
    # And in explicit fractional units the same data gives 0.05.
    st_frac = S.paired_stats(a, b, scale=1.0)
    assert st_frac.mean_delta == pytest.approx(0.05)
    assert st_frac.unit == "custom"


def test_paired_stats_drops_unpaired_seeds():
    """Averaging over an unpaired seed would destroy the pairing the power analysis assumes."""
    a = [_rec("dynamic", i, 0.2 + 0.1 * i) for i in range(4)]
    b = [_rec("static", i, 0.1 + 0.1 * i) for i in (0, 1, 2)]
    st = S.paired_stats(a, b)
    assert st.n_pairs == 3
    assert st.seeds == (0, 1, 2)


def test_paired_stats_refuses_fewer_than_two_pairs():
    with pytest.raises(ValueError, match="shared seeds"):
        S.paired_stats([_rec("dynamic", 0, 0.5)], [_rec("static", 0, 0.4)])


def test_paired_ci_uses_the_exact_t_critical_value():
    """At n=5, df=4, the two-sided 95 % t critical value is 2.7764 -- not 1.96."""
    if not S.HAVE_SCIPY:
        pytest.skip("scipy unavailable")
    a = [_rec("dynamic", i, v) for i, v in enumerate([0.10, 0.20, 0.30, 0.40, 0.50])]
    b = [_rec("static", i, v) for i, v in enumerate([0.05, 0.20, 0.25, 0.45, 0.45])]
    st = S.paired_stats(a, b)
    ci = S.paired_ci(st)
    assert ci.df == 4
    half = (ci.hi - ci.lo) / 2
    assert half / ci.se == pytest.approx(2.7764, abs=1e-3)
    assert ci.method == "exact student-t"


def test_superiority_verdict_returns_underpowered_not_pass():
    """
    The F5e fix, in the superiority direction. At the measured MQAR sigma and n=5 the design has
    ~10 % power at 8 pp, so the verdict must be UNDERPOWERED -- never PASS, never FAIL.
    """
    # Realistically paired: the arms share the data order but not the basin, so the paired
    # differences are themselves large. This is what the measured rho ~= 0.35 looks like.
    a = [_rec("dynamic", i, v) for i, v in enumerate([0.13, 0.28, 0.23, 0.29, 0.88])]
    b = [_rec("static", i, v) for i, v in enumerate([0.42, 0.54, 0.02, 0.56, 0.29])]
    st = S.paired_stats(a, b)
    assert st.s_delta > 20.0, "MQAR paired differences are tens of pp; see the measured spread"
    v = S.superiority_verdict(st, name="S4 > S1", target_effect=8.0)
    assert v.state == "UNDERPOWERED", v.detail
    assert v.passed is False
    assert v.required_n_at_target is not None and v.required_n_at_target > 100
    assert "reported, not adjudicated" in v.detail


def test_superiority_verdict_can_pass_and_can_fail_when_powered():
    """The verdict must be capable of all three states, or it is decoration."""
    # Powered PASS: a large effect (80 pp) with a small but NON-ZERO spread.
    a = [_rec("dynamic", i, 0.90 + 0.002 * i) for i in range(6)]
    b = [_rec("static", i, 0.10 + 0.001 * i) for i in range(6)]
    st = S.paired_stats(a, b)
    assert st.s_delta > 0, "a zero s_delta would be the saturated case, not a powered one"
    v = S.superiority_verdict(st, name="S4 > S1", target_effect=5.0)
    assert v.state == "PASS", v.detail

    # Powered FAIL: same precision, effect goes the WRONG way.
    a2 = [_rec("dynamic", i, 0.10 + 0.002 * i) for i in range(6)]
    b2 = [_rec("static", i, 0.90 + 0.001 * i) for i in range(6)]
    st2 = S.paired_stats(a2, b2)
    v2 = S.superiority_verdict(st2, name="S4 > S1", target_effect=5.0)
    assert v2.state == "FAIL", v2.detail


def test_non_inferiority_is_fail_closed_and_the_naive_form_is_fail_open():
    """
    The F5e finding made executable: on the SAME ambiguous data the conservative CI form FAILS and
    the naive point-estimate form PASSES. That difference is the entire fix.
    """
    # Point estimate is a tiny -1 pp (inside a 2 pp margin) but the seed spread is large, so the
    # data CANNOT exclude a much bigger regression. The naive form calls that a pass.
    a = [_rec("dynamic", i, v) for i, v in enumerate([0.10, 0.35, 0.50, 0.80, 0.91])]
    b = [_rec("static", i, v) for i, v in enumerate([0.20, 0.25, 0.60, 0.72, 0.98])]
    st = S.paired_stats(a, b)
    assert -2.0 < st.mean_delta < 0.0, f"need a small negative point estimate, got {st.mean_delta}"
    strict = S.non_inferiority_verdict(st, name="control non-regression", margin=2.0)
    naive = S.non_inferiority_verdict(
        st, name="control non-regression", margin=2.0, conservative=False
    )
    assert strict.state == "FAIL", strict.detail
    assert naive.state == "PASS", naive.detail
    assert "FAIL-OPEN" in naive.name


# ======================================================================================
# The pre-registered decision rule
# ======================================================================================


def test_rule_requires_s4_to_beat_BOTH_s1_and_s2():
    """SPEC Sec 1.1: if S4 beats S1 but NOT S2, the hypothesis is unsupported."""
    rule = S.PREREGISTERED_RULE
    assert rule.required_contrasts == (("S4", "S1"), ("S4", "S2"))
    assert len(rule.required_contrasts) == 2


def test_beating_s1_but_not_s2_is_UNSUPPORTED():
    """
    The decision rule's whole point, executed. S4 crushes S1 and loses to S2 by the same margin;
    the overall verdict must be UNSUPPORTED, because "one more multiplicative degree of freedom"
    remains a live explanation.
    """
    n = 8
    s4 = [_rec("S4", i, 0.50 + 0.003 * i) for i in range(n)]
    s1 = [_rec("S1", i, 0.10 + 0.001 * i) for i in range(n)]
    s2 = [_rec("S2", i, 0.90 + 0.001 * i) for i in range(n)]
    out = S.PREREGISTERED_RULE.evaluate({"S4": s4, "S1": s1, "S2": s2})
    assert out["overall"] == "UNSUPPORTED", out
    states = {v["name"]: v["state"] for v in out["verdicts"]}
    assert states["S4 > S1"] == "PASS"
    assert states["S4 > S2"] == "FAIL"
    assert any("beating only S1" in nt for nt in out["notes"])


def test_beating_both_is_SUPPORTED():
    n = 8
    s4 = [_rec("S4", i, 0.90 + 0.003 * i) for i in range(n)]
    s1 = [_rec("S1", i, 0.10 + 0.001 * i) for i in range(n)]
    s2 = [_rec("S2", i, 0.20 + 0.001 * i) for i in range(n)]
    out = S.PREREGISTERED_RULE.evaluate({"S4": s4, "S1": s1, "S2": s2})
    assert out["overall"] == "SUPPORTED", out


def test_missing_s2_cannot_yield_SUPPORTED():
    """
    Dropping the control must not be a route to a pass. If S2 is absent the rule reports
    UNDERPOWERED, never SUPPORTED -- there is no code path that reaches SUPPORTED without it.
    """
    n = 8
    s4 = [_rec("S4", i, 0.90 + 0.003 * i) for i in range(n)]
    s1 = [_rec("S1", i, 0.10 + 0.001 * i) for i in range(n)]
    out = S.PREREGISTERED_RULE.evaluate({"S4": s4, "S1": s1})
    assert out["overall"] != "SUPPORTED"
    assert out["overall"] == "UNDERPOWERED"


def test_realistic_mqar_variance_yields_UNDERPOWERED_at_n10():
    """
    The honest expected outcome. Seeds drawn to match the MEASURED spread
    (0.05/0.09/0.20/0.56/0.98 at N512_D64) give an UNDERPOWERED verdict at n=10, which is the
    result Exp-2 should report -- alongside the measured sigma and the required n.
    """
    # Drawn with a shared data-order component and independent per-arm basins, giving a realized
    # rho of ~0.3 -- NOT the rho ~= 1 that a naive "same numbers plus a constant" fixture implies.
    # S4 leads S1 by +9 pp on the mean and still cannot be adjudicated.
    accs_s1 = [0.42, 0.54, 0.00, 0.56, 0.29, 0.47, 0.05, 0.00, 0.31, 0.25]
    accs_s2 = [0.41, 0.48, 0.49, 0.07, 0.24, 0.45, 0.42, 1.00, 0.82, 0.84]
    accs_s4 = [0.13, 0.28, 0.23, 0.29, 0.38, 0.43, 0.52, 0.18, 0.48, 0.88]
    out = S.PREREGISTERED_RULE.evaluate(
        {
            "S4": [_rec("S4", i, a) for i, a in enumerate(accs_s4)],
            "S1": [_rec("S1", i, a) for i, a in enumerate(accs_s1)],
            "S2": [_rec("S2", i, a) for i, a in enumerate(accs_s2)],
        }
    )
    assert out["overall"] == "UNDERPOWERED", out
    # And the reason must be the measured variance, not a missing arm.
    for v in out["verdicts"]:
        assert v["state"] == "UNDERPOWERED", v
        assert v["achieved_power"] < 0.2, v
        assert v["required_n_at_target"] > 100, v


def test_decision_rule_fingerprint_is_pinned():
    """
    The anti-relaxation device. Changing the rule -- dropping S2, widening the margin, lowering the
    confidence level -- changes the hash and breaks this test. Do NOT update the constant to make
    the test pass; that is the quiet relaxation the spec forbids.
    """
    assert S.PREREGISTERED_RULE.fingerprint() == S.PREREGISTRATION_FINGERPRINT
    assert S.PREREGISTRATION_FINGERPRINT == "4adb4bd1ccedc988", (
        "The pre-registered decision rule CHANGED. If this was deliberate and reviewed, say so "
        "explicitly in EXP2-DESIGN.md and record what changed and why."
    )


def test_negative_control_relaxing_the_rule_changes_the_fingerprint():
    """Proof the fingerprint can fail: each relaxation must produce a different hash."""
    base = S.PREREGISTERED_RULE.fingerprint()
    drop_s2 = S.DecisionRule(required_contrasts=(("S4", "S1"),)).fingerprint()
    wider = S.DecisionRule(margin_pp=-5.0).fingerprint()
    looser = S.DecisionRule(conf=0.80).fingerprint()
    weaker = S.DecisionRule(min_power=0.0).fingerprint()
    assert len({base, drop_s2, wider, looser, weaker}) == 5


# ======================================================================================
# Endpoints
# ======================================================================================


def test_floor_is_one_over_D_not_one_over_vocab():
    """SPEC Sec 4.3. At D=64 the floor is 0.015625; at D=4 it is 0.250, not 1/256."""
    assert S.degenerate_floor(64) == pytest.approx(0.015625)
    assert S.degenerate_floor(8) == pytest.approx(0.125)
    assert S.degenerate_floor(4) == pytest.approx(0.250)
    assert S.degenerate_floor(4) != pytest.approx(1 / 256)
    with pytest.raises(ValueError):
        S.degenerate_floor(0)


def test_solve_threshold_sits_in_the_measured_empty_gap():
    """
    Across the 12 positive-control trials NO run landed in [0.30, 0.80] (measured, job 1670928), so
    0.80 sits inside an empty gap and the success rate is insensitive to its exact value there.
    """
    measured = [
        0.0, 0.0, 0.04345703125, 0.1337890625, 0.20849609375, 0.2138671875,
        0.24755859375, 0.25537109375, 0.26318359375, 0.27392578125, 0.99462890625, 1.0,
    ]
    assert S.SOLVE_THRESHOLD == 0.80
    assert not [a for a in measured if 0.30 < a < 0.80]
    # Insensitivity: every threshold in the gap gives the same success count.
    counts = {sum(1 for a in measured if a >= t) for t in (0.31, 0.5, 0.79, 0.80)}
    assert counts == {2}


def test_summarize_cell_reports_success_rate_AND_median_and_the_full_list():
    """
    SPEC Sec 4.2: never a bare mean. On the MEASURED ``N512_D64`` spread the success rate is 0.20
    while the median is 0.2043 = 13.1x the floor -- collapsing to "20 % success" would report an arm
    at 0.55 the same as one at 0.05.
    """
    accs = [0.051483154296875, 0.085296630859375, 0.20428466796875, 0.558441162109375,
            0.9825439453125]
    cell = S.summarize_cell([_rec("static", i, a) for i, a in enumerate(accs)])
    assert cell.success_rate == pytest.approx(0.20)
    assert cell.median_accuracy == pytest.approx(0.20428, abs=1e-4)
    assert cell.median_over_floor == pytest.approx(13.07, abs=0.01)
    assert cell.per_seed_accuracy == tuple(sorted(accs))
    assert cell.bimodal is False, "N512_D64 is where bimodality BREAKS"
    assert cell.n_mid == 2
    assert cell.sigma_accuracy * 100 == pytest.approx(39.39, abs=0.01)


def test_summarize_cell_detects_bimodality_where_it_holds():
    """``N512_D8``: two at 1.0, three parked near the floor, nothing in between."""
    accs = [0.144287109375, 0.1474609375, 0.155517578125, 1.0, 1.0]
    cell = S.summarize_cell(
        [
            S.SeedRecord(arm="static", topology="allliv", kernel_size=3, config="N512_D8",
                         seed=i, accuracy=a, nll_query=3.0, num_pairs=8)
            for i, a in enumerate(accs)
        ]
    )
    assert cell.bimodal is True
    assert cell.n_mid == 0
    assert cell.success_rate == pytest.approx(0.40)


def test_summarize_cell_rejects_mixed_cells_and_duplicate_seeds():
    a = _rec("static", 0, 0.5)
    b = _rec("dynamic", 1, 0.5)
    with pytest.raises(ValueError, match="multiple cells"):
        S.summarize_cell([a, b])
    with pytest.raises(ValueError, match="duplicate seeds"):
        S.summarize_cell([_rec("static", 0, 0.5), _rec("static", 0, 0.6)])


# ======================================================================================
# Clustered SEs
# ======================================================================================


def test_clustered_se_exceeds_naive_when_outcomes_cluster():
    """
    Templated probes have naive SEs *up to 3x too small* -- "a fabricated significance factory".
    With perfectly clustered outcomes (a sequence is all-right or all-wrong) the design effect must
    be ~D.
    """
    d = 64
    k = 100
    sums = [float(d) if i % 2 == 0 else 0.0 for i in range(k)]
    counts = [d] * k
    cm = S.clustered_mean(sums, counts)
    assert cm.mean == pytest.approx(0.5)
    assert cm.se_clustered > cm.se_naive
    assert cm.design_effect == pytest.approx(d, rel=0.05)
    assert cm.se_clustered / cm.se_naive == pytest.approx(math.sqrt(d), rel=0.03)


def test_clustered_se_matches_naive_when_there_is_no_clustering():
    """When every cluster has exactly the population rate, clustering costs nothing."""
    k, d = 200, 4
    sums = [2.0] * k  # every sequence gets exactly half right
    cm = S.clustered_mean(sums, [d] * k)
    assert cm.mean == pytest.approx(0.5)
    assert cm.se_clustered == pytest.approx(0.0, abs=1e-12)
    assert cm.design_effect < 1e-6


def test_clustered_mean_reduces_to_sd_over_sqrt_k_for_equal_clusters():
    """Sanity: with equal cluster sizes the ratio estimator must equal sd(means)/sqrt(K)."""
    import random

    random.seed(0)
    d, k = 8, 50
    per = [random.randint(0, d) for _ in range(k)]
    cm = S.clustered_mean([float(x) for x in per], [d] * k)
    means = [x / d for x in per]
    m = sum(means) / k
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
    assert cm.se_clustered == pytest.approx(sd / math.sqrt(k), rel=1e-9)


def test_min_eval_items_is_1000():
    """R3 F8 fix 4 / Miller Eq. 9: n ~= 969 for delta=0.03 with likelihood scoring."""
    assert S.MIN_EVAL_ITEMS == 1000


# ======================================================================================
# MDE and the fallback
# ======================================================================================


def test_mde_inverts_required_n():
    """MDE and required_n must be mutually consistent: power at (n, MDE(n)) is exactly 0.80."""
    for sd, n in ((42.0, 10), (48.4, 25), (12.0, 5)):
        d = S.mde(n, sd)
        assert S.paired_power(n, sd, d) == pytest.approx(0.80, abs=1e-6)


def test_mde_at_n5_on_measured_mqar_sigma_is_enormous():
    """At n=5 and the measured 42-48 pp, the MDE is ~65-80 pp -- larger than the whole 0-100 range
    of achievable improvement from a 20 % baseline. This is why Exp-2 cannot gate on accuracy."""
    assert S.mde(5, 42.0) > 60
    assert S.mde(5, 48.4) > 70


def test_normal_fallback_is_labelled_and_understates_n():
    """
    If scipy were unavailable the answer must be LABELLED an approximation. Silently substituting it
    would make an underpowered design look adequate -- the exact F8 failure mode.
    """
    res = S.required_n(48.4, 10.0)
    if S.HAVE_SCIPY:
        assert res["method"] == "exact noncentral-t"
        assert res["n"] > res["normal_approx_n"]
    else:
        assert "APPROXIMATION" in res["method"]
    assert res["normal_approx_n"] == 184


def test_norm_ppf_agrees_with_scipy():
    """The scipy-free Acklam fallback must be accurate enough to be a real fallback."""
    if not S.HAVE_SCIPY:
        pytest.skip("nothing to compare against")
    from scipy.stats import norm

    for p in (0.001, 0.025, 0.05, 0.2, 0.5, 0.8, 0.95, 0.975, 0.999):
        # Force the fallback path by calling the polynomial directly.
        saved = S.HAVE_SCIPY
        try:
            S.HAVE_SCIPY = False
            got = S._norm_ppf(p)
        finally:
            S.HAVE_SCIPY = saved
        assert got == pytest.approx(float(norm.ppf(p)), abs=1e-6), p


def test_paired_power_is_monotone_in_n_and_in_effect():
    powers_n = [S.paired_power(n, 42.0, 10.0) for n in (5, 10, 50, 141, 500)]
    assert powers_n == sorted(powers_n)
    powers_d = [S.paired_power(20, 42.0, d) for d in (1, 5, 10, 20, 50)]
    assert powers_d == sorted(powers_d)
    assert S.paired_power(141, 42.0, 10.0) >= 0.80


# ======================================================================================
# THE SATURATED-CELL POOLING TRAP (raised independently by the team lead; MEASURED here)
# ======================================================================================


def _recorded_calibration_json() -> Path:
    """Locate ``mqar_calibration.json``, **next to this file first**.

    FOURTH AND FIFTH INSTANCE OF THE SAME BUG, AND THESE TWO WERE THE WORST KIND. Both helpers in
    this file previously listed only two ABSOLUTE LAPTOP PATHS and called ``pytest.skip()`` when
    neither existed. Off-laptop that is a **silent skip reported as green** -- the earlier FarmShare
    run read ``44 passed, 2 skipped`` and the two skips were exactly these. The tests they gate are
    not decoration: they are the MEASURED evidence for the sigma-pooling trap (EXP2-DESIGN.md
    Sec 12.1), i.e. that pooling over saturated cells deflates sigma ~2x (21.2 pp vs 39.9 pp) and
    therefore quarters the required n. **That claim was unverified on every host but the laptop.**

    A skip is worse than a failure here. A failure is read; a skip is counted as a pass.

    Resolves relative to ``__file__`` first, so it works on FarmShare, in the container and on the
    laptop alike. ``pytest.fail`` rather than ``pytest.skip`` when the file is genuinely absent
    **beside the module that ships with it**: ``check_submission.sh`` lists this JSON as a required
    runtime input, so its absence next to the tests is a packaging defect, not a reason to pass.
    """
    here = Path(__file__).resolve().parent
    for p in (
        here / "mqar_calibration.json",
        here.parent / "mqar" / "mqar_calibration.json",
        Path("/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/"
             "probes/mqar/mqar_calibration.json"),
        Path("/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/"
             "claude-01--liv-short-conv-mixer/experiments/liv/mqar/mqar_calibration.json"),
    ):
        if p.is_file():
            return p
    pytest.fail(
        "mqar_calibration.json not found beside test_sigma.py. This file ships next to the tests "
        "and check_submission.sh asserts it is in the image, so absence is a packaging defect. "
        "Skipping here would report GREEN for the sigma-pooling evidence (Sec 12.1) without "
        "having checked it -- which is how this went unverified off-laptop in the first place."
    )


def _recorded_calibration_accuracies():
    """The per-config accuracies from mqar_calibration.json, read from the file."""
    import json

    by = {}
    for r in json.loads(_recorded_calibration_json().read_text())["runs"]:
        by.setdefault(r["config"], []).append((r["accuracy"], r["num_pairs"]))
    return by


def test_pooling_over_saturated_cells_deflates_sigma_by_a_measured_factor():
    """
    MEASURED on ``mqar_calibration.json``, and the reason :func:`sigma.pooled_sigma` excludes dead
    cells by default::

        pooled over all 8 configs                  23.58 pp   (df 37)
        pooled over the 3 DISCRIMINATING configs   39.93 pp   (df 12)

    The four ceiling-saturated configs (N128_D8, N256_D16, N64_D8, N256_D8) have sigma EXACTLY 0.0:
    zero variance and zero discriminating power. Because required n scales as sigma^2, pooling over
    them understates the seeds needed by (39.93/23.58)^2 = 2.87x.

    This test pins the contrast so the trap cannot silently return.
    """
    by = _recorded_calibration_accuracies()
    cells = []
    for cfgname, rows in by.items():
        accs = [a for a, _ in rows]
        d = rows[0][1]
        cells.append(
            S.summarize_cell(
                [
                    S.SeedRecord(arm="static", topology="calib", kernel_size=3, config=cfgname,
                                 seed=i, accuracy=a, nll_query=3.0, num_pairs=d)
                    for i, a in enumerate(accs)
                ]
            )
        )

    r = S.pooled_sigma(cells)
    # The four ceiling configs must be identified as saturated and excluded.
    assert r["n_cells_excluded_saturated"] == 4, r["excluded_saturated"]
    assert r["n_cells"] == 4, "N64_D4, N512_D64, N512_D8, N1024_D8 can move"

    assert r["pooled_sigma_all_cells"] * 100 == pytest.approx(23.58, abs=0.05)
    # The default (discriminating-only) figure must be the LARGER, honest one.
    assert r["pooled_sigma"] == r["pooled_sigma_discriminating"]
    assert r["pooled_sigma"] > r["pooled_sigma_all_cells"]
    assert r["pooled_sigma_discriminating"] * 100 == pytest.approx(35.86, abs=0.05)

    # Restricting further to the three configs the lead flagged reproduces 39.93 pp exactly.
    three = [c for c in cells if c.config in ("N64_D4", "N512_D64", "N512_D8")]
    r3 = S.pooled_sigma(three)
    assert r3["pooled_sigma"] * 100 == pytest.approx(39.93, abs=0.05)
    assert r3["df_discriminating"] == 12
    # ... and it is consistent with the repo's independently measured 42-48.4 pp.
    assert 35.0 < r3["pooled_sigma"] * 100 < 48.5

    # The understatement factor must be reported, and it must be a real (>2x) shortfall:
    # (35.86/23.58)^2 = 2.31, i.e. pooling over all 8 configs would ask for 2.3x too few seeds.
    assert r["required_n_understatement_factor"] == pytest.approx(2.31, abs=0.03)
    assert r["required_n_understatement_factor"] > 2.0
    assert r["sigma_deflation_if_pooled_over_all"] < 1.0, "pooling over dead cells SHRINKS sigma"
    # Against the 3 configs the lead flagged the shortfall is worse still.
    assert (39.93 / 23.58) ** 2 == pytest.approx(2.87, abs=0.03)


def test_the_summary_array_duplicate_deflates_sigma_to_21_pp():
    """
    The worst version of the trap, MEASURED: ``mqar_calibration.json``'s ``summary`` array lists
    ``N128_D8`` TWICE (it appears in both the capacity grid and the distance sweep), so a naive pool
    over ``summary`` gets **21.15 pp at df 46** -- against 39.93 pp over the discriminating configs.
    That is a (39.93/21.15)^2 = 3.56x understatement of required n.

    Pinned because "pool everything in the file" is the natural thing to write.
    """
    import json

    src = _recorded_calibration_json()

    summary = json.loads(src.read_text())["summary"]
    labels = [s["config"] for s in summary]
    assert labels.count("N128_D8") == 2, "the duplicate is the premise of this test"

    num = den = 0.0
    for s in summary:
        accs = s["accuracies"]
        if len(accs) < 2:
            continue
        m = sum(accs) / len(accs)
        var = sum((a - m) ** 2 for a in accs) / (len(accs) - 1)
        num += (len(accs) - 1) * var
        den += len(accs) - 1
    naive = math.sqrt(num / den) * 100
    assert naive == pytest.approx(21.15, abs=0.05)
    assert den == 46
    assert (39.93 / naive) ** 2 == pytest.approx(3.56, abs=0.05)


def test_pooled_sigma_can_be_asked_for_the_all_cells_figure_explicitly():
    """The deflated figure must remain REACHABLE (for reporting) but never be the default."""
    cells = [_cell(sigma_acc=0.40, n=5), _cell(sigma_acc=0.0, n=5)]
    honest = S.pooled_sigma(cells)
    deflated = S.pooled_sigma(cells, discriminating_only=False)
    assert honest["pooled_sigma"] == pytest.approx(0.40)
    assert deflated["pooled_sigma"] == pytest.approx(0.40 / math.sqrt(2))
    assert deflated["pooled_sigma"] < honest["pooled_sigma"]
    # Both records must carry both numbers, so a reader cannot see one without the other.
    for r in (honest, deflated):
        assert "pooled_sigma_discriminating" in r and "pooled_sigma_all_cells" in r
        assert r["n_cells_excluded_saturated"] == 1


def test_sigma_report_names_the_deflation():
    """The report must print both figures and the understatement factor, not just the honest one."""
    recs = []
    for i, a in enumerate([0.05, 0.20, 0.56, 0.98]):
        recs.append(S.SeedRecord(arm="static", topology="allliv", kernel_size=3,
                                 config="N512_D64", seed=i, accuracy=a, nll_query=3.0,
                                 num_pairs=64))
    for i in range(4):
        recs.append(S.SeedRecord(arm="static", topology="allliv", kernel_size=3,
                                 config="N128_D8", seed=i, accuracy=1.0, nll_query=0.01,
                                 num_pairs=8))
    txt = S.sigma_report(recs)
    assert "DISCRIMINATING CELLS ONLY" in txt
    assert "INCLUDING saturated cells" in txt
    assert "EXCLUDED 1 saturated cell" in txt
    assert "UNDERSTATE required n" in txt
