"""
What the trend test guarantees, and what it refuses.

The headline of this experiment is one slope and an interval around it, and the design expects the
slope to be zero. That makes the failures worth guarding against unusual ones: not a crash, but a
standard error that comes out too small, a margin read in the wrong unit, or a one-sided rule quoted
as if it were an equivalence test. Each of those produces a publishable-looking number that is wrong,
and none of them is visible by inspection of the output.

So four properties carry most of the weight here. The point estimate is the same whether you average
per-seed slopes or run one regression over every cell -- pinned, because it is what makes the
*standard error* the only thing the choice of inferential unit changes, and the naive standard error
is 2.83x too small on the design PRD 3.1 specifies. The margin is 2pp across the sweep and not 2pp per
bit. TOST and non-inferiority disagree about an arm that improved, which is the entire reason both
exist (PRD 16.5). And the Student-t implementation, which has no library behind it, is checked against
published table values and against the closed forms at df 1, 2 and 3.
"""

import math
from dataclasses import replace
from typing import List

import numpy as np
import pytest
from factcrowd.analysis import trend as T
from factcrowd.measure.endpoints import EndpointResult

from olmo_core.exceptions import OLMoConfigurationError

ENTROPY_DEMANDS = (0.173, 0.691, 1.209, 2.244, 3.280, 4.315)
"""
The entropy sweep at 28M from PRD 3.1: demanded bits per parameter at b = 0/4/8/16/24/32.

Restated rather than computed through ``ladder.rho``, so that a drift in the sweep's placement shows up
here as a change to think about rather than as two modules agreeing with each other.
"""

SPAN = ENTROPY_DEMANDS[-1] - ENTROPY_DEMANDS[0]
"""4.142 bits/param end to end -- the ``-4B`` in PRD 8.5's ``D = -4B``."""

_MEAN_DEMAND = sum(ENTROPY_DEMANDS) / len(ENTROPY_DEMANDS)


def line(replicate: int, slope_pp: float, *, level: float = 0.44) -> T.SeedBlock:
    """
    A noiseless replicate whose slope is exactly ``slope_pp`` pp per bit/param.

    Pivoted about the mean demand so that ``level`` is the block's mean accuracy whatever the slope,
    which keeps every accuracy a fraction and keeps the level and the slope independent.
    """
    return T.SeedBlock(
        replicate=replicate,
        demands=ENTROPY_DEMANDS,
        accuracies=tuple(level + slope_pp * (x - _MEAN_DEMAND) / 100.0 for x in ENTROPY_DEMANDS),
    )


def lines(*slopes_pp: float) -> List[T.SeedBlock]:
    """One noiseless block per slope, numbered from zero."""
    return [line(replicate, slope) for replicate, slope in enumerate(slopes_pp)]


def endpoint(accuracy: float, *, n_total: int = 30_000, floor: float = 0.25) -> EndpointResult:
    """An :class:`EndpointResult` at a given accuracy, on PRD 8.5's 30,000-item frozen eval set."""
    return EndpointResult(
        name="mano",
        n_total=n_total,
        n_correct=round(accuracy * n_total),
        n_degenerate=0,
        n_unparseable=0,
        answer_ce_bits=1.0,
        floor=floor,
    )


# --- the slope, which is the estimate everything else is built on ----------------------------------


def test_a_noiseless_line_returns_its_own_slope_exactly():
    """
    ``y = a + b*x`` with no noise recovers ``b`` to the bit, not to a tolerance.

    Chosen so every intermediate value is an exact binary fraction: accuracies stepping by 0.125 over
    integer demands make the cross-products exact, so a tolerance here would only be hiding an error.
    A slope wrong by a constant factor -- ``ddof`` in the wrong place, the intercept subtracted twice --
    still looks like a plausible number and would be invisible against ``approx``.
    """
    block = T.SeedBlock(
        replicate=0, demands=(0.0, 1.0, 2.0, 3.0), accuracies=(0.5, 0.375, 0.25, 0.125)
    )
    assert T.seed_slope(block) == -12.5


def test_the_slope_is_in_percentage_points_and_the_conversion_happens_once():
    """
    A slope of 0.01 accuracy per bit is 1pp per bit, and the factor of 100 is applied exactly once.

    This is the module's only unit conversion, and it sits between two quantities that both look
    reasonable: drop it and a 2pp margin is compared against a number 100x too small, so every sweep
    declares equivalence. Apply it twice and nothing ever does.
    """
    block = T.SeedBlock(replicate=0, demands=(0.0, 1.0, 2.0), accuracies=(0.40, 0.41, 0.42))
    assert T.seed_slope(block) == pytest.approx(1.0, rel=1e-12)


def test_the_slope_does_not_see_the_replicate_s_own_level():
    """
    Shifting one replicate's accuracies by a constant leaves its slope untouched.

    This is what makes a set of blocks a paired design rather than six numbers: a seed that trained to
    a uniformly better model contributes the same trend, so between-seed level differences -- which
    PRD 16.5 says are the whole reason replicates now vary ``init_seed`` -- cannot leak into the
    estimate.
    """
    assert T.seed_slope(line(0, -0.5, level=0.30)) == pytest.approx(
        T.seed_slope(line(0, -0.5, level=0.60))
    )


def test_the_slope_is_least_squares_and_not_the_difference_between_the_ends():
    """
    A sweep whose ends agree is not a flat sweep, and the fit has to say so.

    ``(y_last - y_first) / (x_last - x_first)`` satisfies every other test in this section and discards
    the four interior cells of a six-cell sweep. Here it reports exactly zero -- the flattest result
    the experiment can produce, and the one it expects -- on data whose least-squares slope is
    +1.4pp per bit. Order-invariance is asserted alongside, because an endpoint difference also
    silently assumes the cells arrive sorted.
    """
    demands = (0.0, 1.0, 2.0, 3.0)
    accuracies = (0.40, 0.20, 0.34, 0.40)
    block = T.SeedBlock(replicate=0, demands=demands, accuracies=accuracies)
    endpoint_difference = 100.0 * (accuracies[-1] - accuracies[0]) / (demands[-1] - demands[0])

    fitted = T.seed_slope(block)
    assert endpoint_difference == 0.0
    assert fitted == pytest.approx(1.4, rel=1e-9)

    shuffled = T.SeedBlock(replicate=0, demands=demands[::-1], accuracies=accuracies[::-1])
    assert T.seed_slope(shuffled) == pytest.approx(fitted)


# --- the inferential unit, which is the point of the module -----------------------------------------


def test_averaging_seed_slopes_gives_the_same_estimate_as_one_pooled_regression():
    """
    On a balanced design the two estimators agree to floating point, so only the SE is at stake.

    Worth pinning because it is the load-bearing half of the argument for blocks. If averaging slopes
    also moved the point estimate, choosing it over PRD 8.5's pooled regression would be a change of
    answer that needed defending on its own terms. It is not: the mean of per-seed slopes *is* the
    pooled OLS slope when every block shares its x values, and the disagreement is confined to how
    many independent observations were counted.
    """
    blocks = lines(-0.8, -0.1, 0.4)
    xs = np.array([x for block in blocks for x in block.demands])
    ys = np.array([100.0 * a for block in blocks for a in block.accuracies])
    centred = xs - xs.mean()
    pooled_slope = float(centred @ (ys - ys.mean()) / (centred @ centred))

    assert T.pooled_trend(blocks).slope_mean == pytest.approx(pooled_slope, abs=1e-12)


def test_counting_cells_as_observations_would_shrink_the_interval_past_the_margin():
    """
    The defect the module exists to avoid, demonstrated on the design PRD 3.1 specifies.

    Three replicates over six cells, each replicate perfectly linear at -0.5, 0.0 and +0.5pp per bit.
    Treating the eighteen cells as eighteen independent observations gives df 16 and a standard error
    smaller by 2.83x, and its 90% interval is [-0.74, +0.74]pp -- comfortably inside a 2pp margin, so
    it declares equivalence. The blocked test, which counts the three things that were actually
    randomised, returns [-3.49, +3.49]pp and declares nothing. Same data, same point estimate of zero,
    opposite headline.
    """
    blocks = lines(-0.5, 0.0, 0.5)
    honest = T.tost(blocks)

    xs = np.array([x for block in blocks for x in block.demands])
    ys = np.array([100.0 * a for block in blocks for a in block.accuracies])
    centred = xs - xs.mean()
    slope = float(centred @ (ys - ys.mean()) / (centred @ centred))
    residuals = ys - (ys.mean() - slope * xs.mean()) - slope * xs
    naive_df = len(xs) - 2
    naive_se = math.sqrt(float(residuals @ residuals) / naive_df / float(centred @ centred))
    naive_half_width = T._t_quantile(0.95, naive_df) * naive_se * honest.span

    assert T.pooled_trend(blocks).slope_se == pytest.approx(2.8284 * naive_se, rel=1e-3)
    assert naive_half_width == pytest.approx(0.738, abs=0.005)
    assert naive_half_width < T.EQUIVALENCE_MARGIN_PP  # the naive route would call this equivalent
    assert honest.interval == pytest.approx((-3.491, 3.491), abs=0.005)
    assert not honest.equivalent


def test_one_replicate_is_refused_rather_than_reported():
    """
    A single seed has no between-seed variance, and inventing one is how the prior nulls happened.

    PRD 8.5 traces four uninterpretable nulls to a seed term that was never added: the "0.5pp seed SD"
    was eval sampling noise. A one-block call has to raise rather than return a zero SD, which would
    reproduce exactly that error with a confident interval attached.
    """
    with pytest.raises(OLMoConfigurationError, match="at least 2 replicates"):
        T.pooled_trend(lines(-0.5))


def test_the_between_seed_sd_is_the_sample_sd_of_the_slopes():
    """
    ``ddof=1``, so three replicates divide by two.

    The population form is 18% smaller at n=3, and a standard error 18% too small narrows every
    interval this experiment reports -- in the direction that makes an equivalence claim easier.
    """
    trend = T.pooled_trend(lines(-1.0, 0.0, 1.0))
    assert trend.slope_sd == pytest.approx(1.0)
    assert trend.slope_se == pytest.approx(1.0 / math.sqrt(3))
    assert trend.df == 2
    assert trend.n_blocks == 3


# --- the degenerate case: replicates that agreed exactly --------------------------------------------


def test_identical_blocks_have_no_spread_and_do_not_divide_by_zero():
    """
    Zero variance is a limit, not a crash: the t is +/-inf and the p-value is 0.

    Reached by any synthetic dataset and by an unlucky real one, and the tempting fixes are both
    wrong -- an epsilon in the denominator invents a scale, and raising would make the module unusable
    for exactly the smoke data it is developed against.
    """
    trend = T.pooled_trend(lines(-1.5, -1.5, -1.5))
    assert trend.slope_sd == 0.0
    assert trend.slope_se == 0.0
    assert trend.t_statistic == -math.inf
    assert trend.p_value == 0.0
    assert trend.confidence_interval() == (pytest.approx(-1.5), pytest.approx(-1.5))


def test_identical_blocks_at_zero_slope_give_no_evidence_in_either_direction():
    """
    Zero over zero is the other limit, and it has to be ``t = 0``, not ``nan``.

    A NaN would propagate into the incomplete beta and come back as a NaN p-value, which compares
    false against every threshold -- so an equivalence test would quietly report "not equivalent" for
    the most equivalent dataset expressible.
    """
    trend = T.pooled_trend(lines(0.0, 0.0, 0.0))
    assert trend.t_statistic == 0.0
    assert trend.p_value == 1.0
    assert not math.isnan(trend.p_value)


def test_a_zero_sd_advertises_perfect_resolution_and_that_is_a_warning():
    """
    An MDE of 0 is the arithmetic being honest about an input that is not.

    Documented rather than clamped, because the clamp would need a scale nobody has measured. PRD 8.5
    records the same failure from the other side: the deleted monotonicity re-run rule shrank the
    variance estimate to 0.57 sigma, and a shrunken variance makes an equivalence test falsely declare
    equivalence. A reported MDE of zero means the variance estimate is not usable, not that the
    instrument is perfect.
    """
    assert T.minimum_detectable_effect(0.0, 3) == 0.0
    assert T.pooled_trend(lines(2.0, 2.0, 2.0)).summary()["mde_pp_per_bit"] == 0.0


# --- the Student-t implementation, which has no library behind it -----------------------------------


@pytest.mark.parametrize(
    "probability, df, expected",
    [
        (0.975, 1, 12.706205),
        (0.975, 2, 4.302653),
        (0.975, 3, 3.182446),
        (0.95, 4, 2.131847),
        (0.995, 30, 2.749996),
        (0.975, 100, 1.983972),
    ],
)
def test_the_quantiles_match_published_table_values(probability, df, expected):
    """
    Six entries from a standard t table, to six decimals.

    The confidence intervals are quoted against a 2pp margin, so a quantile good to 1e-4 -- which is
    what the usual rational approximations manage in the tails -- would move a verdict. The bisection
    has to be exact to the CDF's own precision and this is the check that it is.
    """
    assert T._t_quantile(probability, df) == pytest.approx(expected, abs=5e-7)


def test_the_quantile_is_symmetric_about_zero():
    """The t distribution is symmetric, so the two-sided interval is symmetric; asserted, not assumed."""
    for df in (1, 2, 5, 30):
        assert T._t_quantile(0.025, df) == pytest.approx(-T._t_quantile(0.975, df))
    assert T._t_quantile(0.5, 7) == 0.0


def test_the_p_value_matches_the_closed_form_at_one_degree_of_freedom():
    """
    At df=1 the t distribution is Cauchy, whose tail is ``1 - (2/pi)*arctan|t|`` -- no beta function.

    Two replicates with slopes 12.5 and 25 pp/bit: mean 18.75, sample SD 12.5/sqrt(2) = 8.8388,
    SE 6.25, so t is exactly 3.0. The Cauchy tail gives ``1 - (2/pi)*arctan(3) = 0.204833``, and the
    continued fraction has to reproduce it. Every value here is an exact binary fraction, so the only
    approximation being tested is the special function.
    """
    trend = T.pooled_trend([_step_block(0, 0.125), _step_block(1, 0.25)])
    assert trend.slope_mean == 18.75
    assert trend.t_statistic == pytest.approx(3.0, rel=1e-12)

    by_hand = 1.0 - (2.0 / math.pi) * math.atan(3.0)
    assert by_hand == pytest.approx(0.204833, abs=1e-6)
    assert trend.p_value == pytest.approx(by_hand, abs=1e-12)


def test_the_p_value_matches_the_closed_form_at_two_degrees_of_freedom():
    """
    At df=2 the CDF is elementary: ``P(|T| >= t) = 1 - t/sqrt(2 + t^2)``.

    Three replicates at 12.5, 25 and 18.75 pp/bit: mean 18.75, SD 6.25, SE 6.25/sqrt(3), so
    ``t = 3*sqrt(3) = 5.196152`` and ``p = 1 - 5.196152/sqrt(29) = 0.035099``. This is the df the real
    design runs at -- three replicates, two degrees of freedom -- so it is the value most worth being
    able to check by hand.
    """
    trend = T.pooled_trend([_step_block(0, 0.125), _step_block(1, 0.25), _step_block(2, 0.1875)])
    assert trend.df == 2
    assert trend.t_statistic == pytest.approx(3.0 * math.sqrt(3.0), rel=1e-12)

    t = trend.t_statistic
    by_hand = 1.0 - t / math.sqrt(2.0 + t * t)
    assert by_hand == pytest.approx(0.035099, abs=1e-6)
    assert trend.p_value == pytest.approx(by_hand, abs=1e-12)


def test_the_p_value_matches_the_closed_form_at_three_degrees_of_freedom():
    """
    At df=3, ``p = 1 - (2/pi)*(arctan(t/sqrt3) + sqrt3*t/(3 + t^2))``.

    A third independent formula, because the two easy cases are both special enough that an
    implementation could satisfy them by accident -- df=1 needs no beta function at all and df=2 has a
    rational CDF. df=3 mixes an arctangent with a rational term and pins the general path.
    """
    blocks = [_step_block(r, step) for r, step in enumerate((0.125, 0.25, 0.1875, 0.1875))]
    trend = T.pooled_trend(blocks)
    assert trend.df == 3

    t = trend.t_statistic
    by_hand = 1.0 - (2.0 / math.pi) * (
        math.atan(t / math.sqrt(3.0)) + math.sqrt(3.0) * t / (3.0 + t * t)
    )
    assert by_hand == pytest.approx(0.005208, abs=1e-6)
    assert trend.p_value == pytest.approx(by_hand, abs=1e-12)


def _step_block(replicate: int, step: float) -> T.SeedBlock:
    """A three-cell block whose accuracies step by ``step``, giving a slope of ``100*step`` pp/bit."""
    return T.SeedBlock(
        replicate=replicate,
        demands=(0.0, 1.0, 2.0),
        accuracies=(0.25, 0.25 + step, 0.25 + 2.0 * step),
    )


def test_the_interval_and_the_p_value_agree_about_zero():
    """
    A ``1 - alpha`` interval excludes zero exactly when the two-sided p is below alpha.

    They are computed from different directions -- one through the incomplete beta, one through a
    bisection of its inverse -- so agreement across a range of effect sizes is a real cross-check on
    both, and a disagreement is how a figure and its caption come to contradict each other.
    """
    for slope in (0.0, 0.05, 0.2, 0.5, 2.0):
        blocks = lines(slope - 0.3, slope, slope + 0.3)
        trend = T.pooled_trend(blocks)
        low, high = trend.confidence_interval(0.95)
        assert (low > 0.0 or high < 0.0) == (trend.p_value < 0.05), slope


def test_the_interval_defaults_to_ninety_percent():
    """
    PRD 8.5 says to report the 90% CI, because that is the interval TOST's two 5% tests correspond to.

    Quoting a 95% interval beside a TOST verdict would put two alphas in one sentence, and the wider
    interval would look like the more cautious claim while resting on the same rejections.
    """
    trend = T.pooled_trend(lines(-1.0, 0.0, 1.0))
    default = trend.confidence_interval()
    assert default == pytest.approx(trend.confidence_interval(0.90))

    wider = trend.confidence_interval(0.95)
    assert wider[0] < default[0] and wider[1] > default[1]
    assert default[1] - default[0] == pytest.approx(2.0 * T._t_quantile(0.95, 2) * trend.slope_se)


# --- equivalence against non-inferiority: the distinction PRD 16.5 insisted on ----------------------


def test_tost_accepts_an_effect_that_is_genuinely_zero():
    """
    Flat replicates with a small spread are declared equivalent, which is the result P3 predicts.

    The positive control for the whole module: if this failed, every real null would be reported as
    "not established" and the experiment could not return its expected answer.
    """
    result = T.tost(lines(-0.05, 0.0, 0.05))
    assert result.equivalent
    assert result.effect_pp == pytest.approx(0.0, abs=1e-9)
    assert result.p_lower < 0.05 and result.p_upper < 0.05
    assert (
        -T.EQUIVALENCE_MARGIN_PP
        < result.interval[0]
        <= result.interval[1]
        < T.EQUIVALENCE_MARGIN_PP
    )


def test_tost_rejects_a_decline_far_larger_than_the_margin():
    """
    A 4pp decline across the sweep is not equivalence within 2pp, however tight the seeds agree.

    The failure this guards is a TOST that rejects on precision alone: with a small enough SE both
    one-sided tests reject whatever the estimate is, unless the estimate itself is compared to the
    margin. Here the SE is tiny and the verdict must still be "not equivalent".
    """
    result = T.tost(lines(-1.0, -0.97, -1.03))
    assert result.effect_pp == pytest.approx(-4.142, abs=0.01)
    assert not result.equivalent
    assert result.p_upper < 0.05  # improvements of 2pp or more are excluded ...
    assert result.p_lower > 0.05  # ... and declines of 2pp or more are not


def test_non_inferiority_accepts_a_large_improvement_that_tost_rejects():
    """
    The contrast that is the reason both functions exist, on an arm that gained 20pp.

    Revision 1 pre-registered the one-sided rule and read it as flatness. An arm improving by 20
    points across the sweep passes it -- correctly, since it excludes declines -- and is about as far
    from flat as this instrument can measure. PRD 16.5: the rule "is non-inferiority and is now called
    that". Reading one verdict as the other is the error, and here the two verdicts are opposite.
    """
    improving = lines(4.9, 5.0, 5.1)
    equivalence = T.tost(improving)
    safety = T.non_inferiority(improving)

    assert equivalence.effect_pp == pytest.approx(20.71, abs=0.01)
    assert not equivalence.equivalent
    assert equivalence.p_upper > 0.05  # nothing bounds the improvement
    assert safety.non_inferior
    assert safety.p_value < 1e-3
    assert safety.lower_bound > 0.0


def test_non_inferiority_and_tost_both_reject_a_large_decline():
    """
    They differ only about improvements, so on the outcome the experiment is looking for they agree.

    Stated because the previous test could be satisfied by a non-inferiority test that accepted
    everything. It has to reject the one thing it is for.
    """
    declining = lines(-1.05, -1.0, -0.95)
    assert not T.tost(declining).equivalent
    assert not T.non_inferiority(declining).non_inferior


def test_the_tost_verdict_is_exactly_the_interval_inside_the_margin():
    """
    Both formulations of TOST are implemented -- two p-values and one interval -- and they must agree.

    PRD 8.5 asks for the interval to be reported and the tests to decide, so a dataset where the
    published interval sits inside the margin while the verdict says otherwise would be a figure
    contradicting its own caption. Checked across margins that put the boundary on either side.
    """
    for slopes in ((-0.05, 0.0, 0.05), (-0.6, -0.5, -0.4), (0.9, 1.0, 1.1), (-2.0, 0.0, 2.0)):
        for margin in (0.5, 1.0, 2.0, 5.0):
            result = T.tost(lines(*slopes), margin_pp=margin)
            inside = -margin < result.interval[0] and result.interval[1] < margin
            assert inside == result.equivalent, (slopes, margin)


def test_the_margin_is_read_across_the_sweep_and_not_per_bit():
    """
    A slope of -0.6pp per bit is 2.5pp end to end, and 2.5pp is outside a 2pp margin.

    PRD 8.5 defines the effect as ``D = -4B``: the margin is quoted on the change across the whole
    entropy axis, which spans 4.142 bits/param. Compared per bit the same slope looks like a third of
    the margin, so the sweep would be declared equivalent while costing more accuracy than the margin
    allows -- the error is a factor of 4.142 and it runs in the permissive direction.
    """
    result = T.tost(lines(-0.61, -0.60, -0.59))
    assert abs(result.effect_pp / SPAN) < T.EQUIVALENCE_MARGIN_PP  # per bit it looks fine
    assert result.effect_pp == pytest.approx(-2.485, abs=0.01)
    assert not result.equivalent


def test_the_span_comes_from_the_cells_and_can_be_stated_instead():
    """
    The default is the range actually swept; an explicit span rescales the same slope.

    Passing it is how two arms that stopped at different cells get compared on one x-axis, which PRD
    3.1 requires ("both sweeps must be plotted on the same x-axis definition") for the count-minus-
    entropy subtraction to mean anything.
    """
    blocks = lines(-0.5, -0.5, -0.5)
    assert T.tost(blocks).span == pytest.approx(SPAN)
    assert T.tost(blocks).effect_pp == pytest.approx(-0.5 * SPAN)
    assert T.tost(blocks, span=1.0).effect_pp == pytest.approx(-0.5)
    assert T.non_inferiority(blocks, span=2.0).effect_pp == pytest.approx(-1.0)


def test_replicates_that_swept_different_ranges_are_refused():
    """
    Two blocks measured over different ranges have effects in different units.

    Averaging them silently rescales one, and the result is a number in no unit at all quoted against
    a margin in pp. Refused, with the two ranges in the message and a pointer at the override.
    """
    short = T.SeedBlock(replicate=0, demands=(0.2, 1.0, 2.0), accuracies=(0.44, 0.43, 0.42))
    long = T.SeedBlock(replicate=1, demands=(0.2, 2.0, 4.3), accuracies=(0.44, 0.43, 0.42))
    # Refused on the grid itself, and **stating a span no longer opens that door**. An audit found blocks
    # over different treatment grids being pooled whenever a span was named; two grids are two
    # experiments, and a paired design's whole claim is that its blocks differ only in seed.
    # Without a span, the range mismatch is caught first: an effect "across the sweep" is not one
    # quantity when the sweeps differ.
    with pytest.raises(OLMoConfigurationError, match="swept different ranges"):
        T.tost([short, long])
    # And naming a span no longer opens the door -- the grid itself is refused behind it.
    with pytest.raises(OLMoConfigurationError, match="one treatment grid"):
        T.tost([short, long], span=SPAN)

    # On a valid design, span still controls how a slope becomes an end-to-end effect.
    matched = [
        T.SeedBlock(replicate=r, demands=(0.2, 1.0, 2.0), accuracies=(0.44, 0.43 + 0.001 * r, 0.42))
        for r in range(3)
    ]
    assert T.tost(matched, span=SPAN).span == pytest.approx(SPAN)


def test_the_published_sentences_say_what_is_excluded_rather_than_no_effect():
    """
    PRD 8.5: report the interval and say "declines greater than X pp are excluded", never "no effect".

    The verdict strings are what a caller will paste into a report, so the phrasing is part of the
    contract. Non-inferiority's has to name itself, because being quoted as equivalence is the
    specific failure PRD 16.5 corrected.
    """
    equivalence = T.tost(lines(-0.05, 0.0, 0.05)).verdict
    assert "excluded" in equivalence and "no effect" not in equivalence
    assert "in either direction" in equivalence

    safety = T.non_inferiority(lines(4.9, 5.0, 5.1)).verdict
    assert "declines greater than" in safety
    assert "not equivalence" in safety
    assert "improvements are not bounded" in safety


def test_a_failed_equivalence_test_does_not_claim_an_effect():
    """
    Wide intervals and real effects produce the same boolean, and the sentence must not confuse them.

    "Not equivalent" from three noisy seeds means the interval is too wide to place, which is a
    statement about the design and not about the model -- and is why the MDE is published beside it.
    """
    verdict = T.tost(lines(-3.0, 0.0, 3.0)).verdict
    assert "not established" in verdict
    assert "leaves the margin" in verdict


@pytest.mark.parametrize("test", [T.tost, T.non_inferiority])
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"margin_pp": 0.0}, "'margin_pp' must be positive"),
        ({"margin_pp": -2.0}, "'margin_pp' must be positive"),
        ({"alpha": 0.5}, "'alpha' must be in"),
        ({"alpha": 0.0}, "'alpha' must be in"),
        ({"span": 0.0}, "'span' must be positive"),
    ],
)
def test_both_margin_tests_refuse_the_same_bad_settings(test, kwargs, match):
    """
    One validation path for both, so they cannot drift into accepting different inputs.

    ``alpha >= 0.5`` is the one worth naming: at that level both one-sided tests can reject on an
    estimate that lies outside the margin, so TOST would declare equivalence for an effect it has just
    measured as too large.
    """
    with pytest.raises(OLMoConfigurationError, match=match):
        test(lines(-0.5, 0.0, 0.5), **kwargs)


# --- what a block will not accept --------------------------------------------------------------------


def test_unpaired_demands_and_accuracies_are_refused():
    """
    A missing cell would otherwise be silently dropped or zip itself against the wrong demand.

    Either way the block still returns a slope, which is the failure mode this whole file is about: a
    plausible number from data that no longer describes the runs.
    """
    with pytest.raises(OLMoConfigurationError, match="unpaired"):
        T.SeedBlock(replicate=0, demands=(0.2, 1.0, 2.0), accuracies=(0.44, 0.43))


def test_two_cells_are_not_a_trend():
    """
    Two points fit a line exactly, with no residual freedom and no way to see curvature.

    The entropy axis runs six cells, so this only fires on a truncated results file -- which is
    precisely when a two-point "slope" would be quoted as if the other four had been measured.
    """
    with pytest.raises(OLMoConfigurationError, match="at least 3 cells"):
        T.SeedBlock(replicate=0, demands=(0.2, 4.3), accuracies=(0.44, 0.40))


def test_a_block_with_no_spread_in_demand_is_refused():
    """
    Every cell at one demand is a division by zero, and the message has to say which failure it is.

    A NaN slope would flow all the way to a NaN p-value that compares false against every threshold,
    so the block that was misconfigured would be reported as "not equivalent" rather than as broken.
    """
    with pytest.raises(OLMoConfigurationError, match="no independent variable"):
        T.SeedBlock(replicate=0, demands=(1.2, 1.2, 1.2), accuracies=(0.44, 0.43, 0.42))


@pytest.mark.parametrize("bad", [44.0, 1.5, -0.01, float("nan")])
def test_accuracies_must_be_fractions_and_not_percentage_points(bad):
    """
    ``44.0`` for 44% is the trap: it survives every other check and multiplies the slope by 100.

    A slope 100x too large is still a number a reader would believe, and against a 2pp margin it turns
    every null into a finding. NaN is included because it fails the same range test -- and because a
    NaN that reaches the incomplete beta comes back as a NaN p-value rather than as an error.
    """
    with pytest.raises(OLMoConfigurationError, match="not a fraction"):
        T.SeedBlock(replicate=0, demands=(0.2, 1.0, 2.0), accuracies=(0.44, 0.43, bad))


def test_a_non_finite_demand_is_refused():
    """
    An infinite demand means the cell was never placed, and it poisons the fit rather than skewing it.

    The mean demand becomes infinite, every centred x becomes ``-inf`` or ``nan``, and the slope comes
    back NaN -- which then compares false against every threshold downstream and reports the broken
    block as "not equivalent" instead of as broken.
    """
    with pytest.raises(OLMoConfigurationError, match="not finite"):
        T.SeedBlock(replicate=0, demands=(0.2, 1.0, math.inf), accuracies=(0.44, 0.43, 0.42))


# --- building a block from what the scorer returned --------------------------------------------------


def test_a_block_reads_accuracies_off_endpoint_results():
    """
    The scorer's own numbers go in unmodified, as fractions, with the demand supplied per cell.

    The alternative -- a caller writing ``result.n_correct / result.n_total`` at each site -- is how a
    denominator comes to be the wrong one, which PRD 8's endpoint contract exists to prevent.
    """
    observations = [(x, endpoint(0.44 - 0.005 * x)) for x in ENTROPY_DEMANDS]
    block = T.SeedBlock.from_endpoints(2, observations)

    assert block.replicate == 2
    assert block.demands == pytest.approx(ENTROPY_DEMANDS)
    assert block.accuracies[0] == pytest.approx(observations[0][1].accuracy)
    assert T.seed_slope(block) == pytest.approx(-0.5, abs=0.02)


def test_a_floor_shared_by_every_cell_cancels_out_of_the_slope():
    """
    Which is why the slope may be taken on raw accuracy rather than on above-floor.

    A slope's cell weights sum to zero, so any constant common to the cells drops out exactly. This is
    the same mechanism PRD 8.5's first fix buys with one frozen eval set -- shared item difficulty
    cancels -- and it is worth pinning because it is the reason the two possible dependent variables
    give the same answer, so a future change from one to the other cannot move a result.
    """
    results = [endpoint(0.44 - 0.005 * x, floor=0.31) for x in ENTROPY_DEMANDS]
    block = T.SeedBlock.from_endpoints(0, list(zip(ENTROPY_DEMANDS, results)))

    # The same regression on above-floor scores, which is the other defensible dependent variable and
    # is already in pp. Asserting against it is the real check; asserting that two floors give the same
    # slope would only be observing that from_endpoints does not read the floor.
    xs = np.array(ENTROPY_DEMANDS)
    ys = np.array([result.above_floor for result in results])
    centred = xs - xs.mean()
    above_floor_slope = float(centred @ (ys - ys.mean()) / (centred @ centred))

    assert results[0].floor == 0.31  # shared, nonzero, and worth 31 points of accuracy
    assert T.seed_slope(block) == pytest.approx(above_floor_slope)


def test_a_block_refuses_cells_that_were_not_scored_the_same_way():
    """
    A floor, an endpoint or an eval set that changed mid-sweep enters the slope as if it were an effect.

    Only quantities *shared* by the cells cancel. An instrument that moved between cells is
    indistinguishable, in the slope, from the treatment -- and 30,000 items against 2,000 is exactly
    the change PRD 8.5's first fix made, so a results file spanning that change is a live possibility.
    """
    good = (ENTROPY_DEMANDS[0], endpoint(0.44))
    with pytest.raises(OLMoConfigurationError, match="mixes endpoints"):
        T.SeedBlock.from_endpoints(
            0, [good, (1.2, EndpointResult("brevo1", 30_000, 100, 0, 0, 1.0, 0.25))]
        )
    with pytest.raises(OLMoConfigurationError, match="frozen eval set was not shared"):
        T.SeedBlock.from_endpoints(0, [good, (1.2, endpoint(0.44, n_total=2_000))])
    with pytest.raises(OLMoConfigurationError, match="the instrument changed"):
        T.SeedBlock.from_endpoints(0, [good, (1.2, endpoint(0.44, floor=0.5))])
    with pytest.raises(OLMoConfigurationError, match="no cells"):
        T.SeedBlock.from_endpoints(0, [])


# --- the minimum detectable effect, which PRD 8.6's G7 requires published ---------------------------


def test_the_mde_is_linear_in_the_standard_deviation():
    """
    Twice the seed noise is twice the smallest effect that could have been seen.

    The relationship a reader will assume when scaling a published MDE to their own sigma, so it is
    worth pinning rather than leaving to the reader's inspection of the formula.
    """
    assert T.minimum_detectable_effect(2.0, 5) == pytest.approx(
        2.0 * T.minimum_detectable_effect(1.0, 5)
    )
    assert T.minimum_detectable_effect(0.5, 5) == pytest.approx(
        0.5 * T.minimum_detectable_effect(1.0, 5)
    )


def test_the_mde_falls_faster_than_one_over_root_n_at_small_n():
    """
    Doubling the replicates buys more than sqrt(2), because the t quantiles shrink with df as well.

    Worth stating because the planning arithmetic is usually done with the normal approximation, which
    predicts exactly sqrt(2) and is the same conflation PRD 8.5 caught: "the docs paired a t threshold
    with a normal statistic". At n=3 the extra df is worth far more than the extra sample.
    """
    ratio_small = T.minimum_detectable_effect(1.0, 6) / T.minimum_detectable_effect(1.0, 3)
    assert ratio_small < 1.0 / math.sqrt(2.0)
    # 0.4395 with exact noncentral-t power; the central-t approximation this replaced gave 0.460.
    assert ratio_small == pytest.approx(0.4395, abs=0.005)  # against 0.707 from the normal form

    # And it converges to the normal answer once df stops mattering.
    ratio_large = T.minimum_detectable_effect(1.0, 2000) / T.minimum_detectable_effect(1.0, 1000)
    assert ratio_large == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-3)


def test_the_mde_is_monotone_in_the_replicates_and_in_the_power_demanded():
    """
    More seeds resolve more; more power and more confidence both cost resolution.

    Three directions that a sign error in the quantile sum would reverse, and a reversed MDE would be
    reported beside a null as evidence the design was better than it was.
    """
    assert [T.minimum_detectable_effect(1.0, n) for n in (2, 3, 4, 8, 16)] == sorted(
        [T.minimum_detectable_effect(1.0, n) for n in (2, 3, 4, 8, 16)], reverse=True
    )
    assert T.minimum_detectable_effect(1.0, 4, power=0.95) > T.minimum_detectable_effect(1.0, 4)
    assert T.minimum_detectable_effect(1.0, 4, alpha=0.01) > T.minimum_detectable_effect(1.0, 4)


def test_the_mde_of_the_planned_design_is_far_above_the_margin_it_is_meant_to_test():
    """
    Three replicates and a 1pp-per-bit seed SD resolve 13.5pp across the sweep, against a 2pp margin.

    This is PRD 8.6's finding in one assertion: "the design is 8-50x short and the old gate would not
    have noticed". Pinned with real numbers so that publishing the MDE beside a null is a statement
    with content -- and so that anyone who improves sigma or the seed count can see the figure move.
    """
    per_bit = T.minimum_detectable_effect(1.0, 3)
    # 3.2640, not the 3.0965 the central-t approximation gave: that formula treats the alternative as a
    # central t and was ~5.4% optimistic at k=3, in the direction that flatters an under-powered design.
    assert per_bit == pytest.approx(3.2640, abs=1e-3)
    assert per_bit * SPAN == pytest.approx(13.52, abs=0.01)
    assert per_bit * SPAN > 6.0 * T.EQUIVALENCE_MARGIN_PP


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"slope_sd": -1.0, "n_blocks": 3}, "must not be negative"),
        ({"slope_sd": 1.0, "n_blocks": 1}, "at least 2 replicates"),
        ({"slope_sd": 1.0, "n_blocks": 3, "alpha": 1.0}, "'alpha' must be in"),
        ({"slope_sd": 1.0, "n_blocks": 3, "power": 0.0}, "'power' must be in"),
    ],
)
def test_the_mde_refuses_inputs_it_cannot_answer_for(kwargs, match):
    """A single replicate has no variance estimate, so it has no resolution to report either."""
    with pytest.raises(OLMoConfigurationError, match=match):
        T.minimum_detectable_effect(**kwargs)


# --- what gets logged --------------------------------------------------------------------------------


def test_the_trend_summary_publishes_sigma_and_the_mde():
    """
    PRD 8.6's G7 requires both beside any result, and a summary that omitted them would be the default.

    All four of this programme's prior nulls are consistent with an unmeasured sigma (PRD 8.5), so a
    logged row without one is a row that cannot be re-read later.
    """
    summary = T.pooled_trend(lines(-0.6, -0.5, -0.4)).summary()
    assert summary["slope_sd_pp_per_bit"] == pytest.approx(0.1)
    # Exact small-sample power; 0.3097 was the central-t approximation.
    assert summary["mde_pp_per_bit"] == pytest.approx(0.32640, abs=1e-4)
    assert summary["n_blocks"] == 3 and summary["df"] == 2
    assert float(summary["ci90_low_pp_per_bit"]) < float(  # type: ignore[arg-type]
        summary["slope_mean_pp_per_bit"]  # type: ignore[arg-type]
    )
    assert all(isinstance(value, (int, float, str, bool)) for value in summary.values())


def test_the_two_verdict_summaries_name_the_test_they_came_from():
    """
    A collected results file will hold both, and they differ only in fields a reader has to notice.

    ``equivalent`` and ``non_inferior`` are both booleans about the same slope, so the ``test`` key is
    what stops a plot from labelling one as the other -- the confusion PRD 16.5 found in the
    pre-registration itself.
    """
    blocks = lines(-0.05, 0.0, 0.05)
    equivalence, safety = T.tost(blocks).summary(), T.non_inferiority(blocks).summary()

    assert equivalence["test"] == "tost" and "equivalent" in equivalence
    assert safety["test"] == "non_inferiority" and "non_inferior" in safety
    assert "equivalent" not in safety
    assert equivalence["span_bits_per_param"] == pytest.approx(round(SPAN, 6))
    assert all(isinstance(value, (int, float, str, bool)) for value in safety.values())


def test_the_equivalence_interval_is_the_trend_interval_scaled_by_the_span():
    """
    One interval, reported in two units, so a figure in pp per bit and a verdict in pp cannot disagree.

    They are computed in different functions from the same quantile, and the scaling is the only
    difference; if that stopped being true the caption and the axis would drift apart silently.
    """
    blocks = lines(-0.6, -0.5, -0.4)
    trend_low, trend_high = T.pooled_trend(blocks).confidence_interval(0.90)
    equivalence = T.tost(blocks, alpha=0.05)

    assert equivalence.interval[0] == pytest.approx(trend_low * SPAN)
    assert equivalence.interval[1] == pytest.approx(trend_high * SPAN)
    assert equivalence.p_value == max(equivalence.p_lower, equivalence.p_upper)


def test_three_readings_of_one_seed_are_not_three_replicates():
    """
    The between-seed variance has to come from between seeds.

    An adversarial pass handed `pooled_trend` three blocks all labelled `replicate=0`. It returned a
    confident interval built from a variance no seed had produced -- exactly the failure mode the
    per-seed slope design exists to avoid, arriving through the door marked "paired".
    """
    with pytest.raises(OLMoConfigurationError, match=r"replicate ids \[0\] appear more than once"):
        T.check_blocks([line(0, -0.5), line(0, -0.4), line(1, -0.3)])
    T.check_blocks(lines(-0.5, -0.4, -0.3))  # distinct ids are fine


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", "compare"),
        ("row", "28m"),
        ("sweep", "count"),
        ("step", 12_098),
        ("eval_items", 10_000),
    ],
)
def test_blocks_differing_in_anything_but_the_seed_are_refused(field, value):
    """
    "Differs only in initialisation and data order" is a claim about the design, and it is not visible
    in the numbers.

    Two blocks read at different checkpoint steps, or on differently-sized eval sets, or from different
    model rows, will pool into a perfectly plausible slope. So the identity travels with the block and is
    compared, rather than assumed because the caller grouped them.
    """
    good = line(0, -0.5)
    odd = replace(line(1, -0.4), **{field: value})
    with pytest.raises(OLMoConfigurationError, match=f"has {field}="):
        T.check_blocks([good, odd])


def test_a_trend_through_a_subset_cannot_be_reported_as_the_pre_registered_one():
    """
    Six entropy cells were pre-registered; three of them are a different, weaker claim.

    Nothing in the arithmetic objects to a three-point fit, so the level count is stated by the caller
    and checked here.
    """
    blocks = lines(-0.5, -0.4)
    T.check_blocks(blocks, required_levels=len(ENTROPY_DEMANDS))
    with pytest.raises(OLMoConfigurationError, match="not the 3 the pre-registered design states"):
        T.check_blocks(blocks, required_levels=3)


def test_the_mde_uses_exact_small_sample_power_not_the_central_t_approximation():
    """
    The old formula was optimistic in the direction that makes an under-powered design look adequate.

    `t(1-a/2, df) + t(power, df)` over sqrt(n) treats the alternative as a *central* t. At k=3 that returns
    3.0965 x SD where exact 80% power needs 3.264 x SD -- about 5.4% optimistic, and every MDE this project
    published for a three-seed design was understated by that much.

    Integrated rather than approximated: a noncentral t is `(Z + lambda) / sqrt(V/df)`, so the power is an
    expectation over a chi-square of a normal tail and needs nothing beyond `erf`.
    """
    exact = {3: 3.264, 4: 2.128, 6: 1.435}
    for blocks, expected in exact.items():
        assert T.minimum_detectable_effect(1.0, blocks) == pytest.approx(expected, abs=0.005)

    # The old approximation, reproduced, so the direction of the error is pinned rather than described.
    df = 2
    central = (T._t_quantile(0.975, df) + T._t_quantile(0.80, df)) / math.sqrt(3)
    assert central == pytest.approx(3.0965, abs=0.001)
    assert T.minimum_detectable_effect(1.0, 3) > central

    # Monotone in replicates, and linear in the SD it is expressed in.
    values = [T.minimum_detectable_effect(1.0, k) for k in (3, 4, 6, 8, 12)]
    assert values == sorted(values, reverse=True)
    assert T.minimum_detectable_effect(2.0, 6) == pytest.approx(
        2 * T.minimum_detectable_effect(1.0, 6)
    )


def test_three_seeds_only_resolve_the_two_point_margin_at_a_sub_point_sigma():
    """
    The number that decides how many seeds phase 2 needs, stated where it cannot be lost.

    PRD 8.5's margin is 2pp on the end-to-end effect. At three replicates the MDE is 3.264 x SD, so three
    seeds reach 80% power only if the true between-seed SD is at or under 0.61pp -- and phase 1's sigma was
    measured on models pinned at a constant-policy floor, which says nothing about a live endpoint.
    """
    margin = 2.0
    ceiling = margin / T.minimum_detectable_effect(1.0, 3)
    assert ceiling == pytest.approx(0.613, abs=0.01)
    # Six seeds are adequate at a full point of SD, which is why phase 2 targets six.
    assert T.minimum_detectable_effect(1.0, 6) < margin
    assert T.minimum_detectable_effect(1.0, 4) > margin
