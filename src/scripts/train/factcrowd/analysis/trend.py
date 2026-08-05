"""
The pre-registered trend test: does reasoning fall as demanded fact bits per parameter rises?

One number carries the headline (PRD 14, P3) and the design expects it to be a null, so what this
module has to get right is not the slope -- four lines of algebra -- but the standard error and the
shape of the test built around it. Four decisions, all of them consequential.

**The inferential unit is the per-seed slope, not the cell.** Each replicate contributes exactly one
number, the OLS slope of accuracy on demand across its own cells, and the headline test is a
one-sample t over those. The alternative -- one regression over every cell with replicates as rows --
is what PRD 8.5's second fix asks for, and it is right about the *point estimate*: on a balanced
design the mean of the per-seed slopes and the pooled OLS slope are the same number to the last bit,
which :func:`seed_slope`'s tests pin. It is wrong about the standard error. Cells within a replicate
share an initialisation, a data order and -- by PRD 8.5's first fix -- one frozen eval set, so they
are not independent draws; counting six correlated cells as six observations divides the SE by the
wrong square root and reports a narrow interval for a wide one. PRD 16.5 reaches the same place from
the other side: replicates that differed only in the corpus seed never were replicates, and a shared
eval set lowers measurement noise without creating any. So the pooling here is over blocks, and
``n`` is the number of trained-model replicates -- a number that will be 3.

**The margin is accuracy points across the sweep, not per bit.** PRD 8.5 writes the effect as
``D = -4B``: the slope times the ~4.14 bits/param the entropy axis spans (0.173 to 4.315, PRD 3.1).
"Within 2pp" therefore means 2pp end to end. Read as 2pp *per bit* the same margin is eight times too
loose and would declare equivalence for a sweep that cost eight accuracy points. :func:`tost` and
:func:`non_inferiority` scale the slope by the swept span before comparing to the margin, and refuse
blocks that do not share a span, because effects measured over different ranges are not in the same
units and their mean means nothing.

**Equivalence and non-inferiority are different claims, and both are here.** The rule revision 1
pre-registered -- reject ``H0: D >= 2pp``, one-sided -- bounds only the decline. An arm that gained
ten points passes it comfortably, which makes it a safety claim ("crowding is not costing us two
points") and not the flatness claim P3 makes. PRD 16.5 lists the correction among the things accepted
and not yet done: "a TOST interval instead of the one-sided rule, which is non-inferiority and is now
called that." Both are implemented, neither is reachable by accident from the other, and :func:`tost`
is the one P3 refers to.

**No scipy.** It is not in the image and not in ``pyproject.toml``, and a dependency added for two
special functions would have to survive every rebuild of a training image for the sake of an analysis
that runs on a laptop. The Student-t CDF and its inverse are eighty lines here, checked against
published table values and against the closed forms at ``df`` 1, 2 and 3 -- which is more verification
than an import would have received.

Accuracies enter as fractions, because that is what :class:`~factcrowd.measure.endpoints.EndpointResult`
reports; everything this module returns is in **percentage points**, because that is the unit the
margin, the MDE and every figure in PRD 8 are quoted in. The conversion happens once, in
:func:`seed_slope`.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..measure.endpoints import EndpointResult

__all__ = [
    "EQUIVALENCE_MARGIN_PP",
    "check_blocks",
    "SeedBlock",
    "TrendResult",
    "EquivalenceResult",
    "NonInferiorityResult",
    "seed_slope",
    "pooled_trend",
    "tost",
    "non_inferiority",
    "minimum_detectable_effect",
]


_DEGENERATE_SE_PP = 1e-12
"""
Below this, an observed between-seed standard error is treated as no information rather than as perfect
precision.

An endpoint that is dead (``n_correct == 0`` in every cell of every replicate) or saturated produces
*identical* integer counts across seeds, so the between-seed SD is exactly zero, every t is infinite, and
both one-sided nulls are rejected with p = 0. The pre-registered headline then comes out maximally strong
from an instrument that measured nothing -- the failure PRD 1 lists four times. A frozen shared evaluation
set makes identical counts more likely, not less.

A real effect with identical seeds is still correctly rejected, so this blocks only the flat case, and it
blocks it by withholding the verdict rather than by hiding the numbers: both p-values and the interval are
still reported, and the verdict string says the variance estimate is degenerate.
"""

EQUIVALENCE_MARGIN_PP = 2.0
"""
The pre-registered margin: accuracy points of change across the whole swept range.

PRD 14 sizes the effect to power for at ~2pp, from ``arXiv:2505.18091`` -- the paper that already
found crowding and attributed it to capacity. PRD 8.5 states the null the same way, ``H0: D >= 2.0pp``
with ``D = -4B``. It is a margin on the *end-to-end* change and not on the slope; see the module
docstring for what reading it per bit would cost.
"""

_MIN_POINTS_PER_BLOCK = 3
"""
Cells a replicate must contribute before its slope is a trend rather than a difference.

Two points fit a line exactly, with no residual degrees of freedom and no way to see curvature, so a
two-point "slope" is a rescaled difference of two cells wearing a regression's clothes. The entropy
axis runs six cells (PRD 3.1), so this floor is never binding on the real design -- it is here to
catch a truncated results file.
"""


# --- the Student-t distribution, which we do not get to import ------------------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """
    Lentz's continued fraction for the incomplete beta, the standard Numerical Recipes ``betacf``.

    :param a: First shape parameter.
    :param b: Second shape parameter.
    :param x: Point in ``(0, 1)``, already known to be on the fast-converging side.

    :returns: The continued fraction's value.
    """
    tiny = 1e-300  # Lentz's guard: a zero denominator would otherwise take the recurrence to inf
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < 3e-16:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """
    The regularised incomplete beta ``I_x(a, b)``.

    :param a: First shape parameter.
    :param b: Second shape parameter.
    :param x: Point in ``[0, 1]``.

    :returns: ``I_x(a, b)``, in ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    # The fraction converges fast only on one side of the mean; past it, use the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a) rather than iterating a slowly converging series.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _t_two_sided_p(t: float, df: int) -> float:
    """
    ``P(|T| >= |t|)`` for Student's t on ``df`` degrees of freedom.

    :param t: The t statistic. ``±inf`` is accepted and returns 0, which is the limit and is what a
        zero standard error produces; see :func:`pooled_trend`.
    :param df: Degrees of freedom, at least 1.

    :returns: The two-sided tail area.

    :raises OLMoConfigurationError: If ``df`` is below 1 or ``t`` is NaN.
    """
    if df < 1:
        raise OLMoConfigurationError(f"the t distribution needs df >= 1, got {df}")
    if math.isnan(t):
        raise OLMoConfigurationError(
            "t statistic is NaN, which is a defect upstream, not a p-value"
        )
    if math.isinf(t):
        return 0.0
    return _incomplete_beta(0.5 * df, 0.5, df / (df + t * t))


def _t_survival(t: float, df: int) -> float:
    """
    ``P(T > t)``: the upper-tail area, which is what the one-sided tests need.

    :param t: The t statistic.
    :param df: Degrees of freedom.

    :returns: The upper-tail area.
    """
    tail = 0.5 * _t_two_sided_p(t, df)
    return tail if t > 0 else 1.0 - tail


def _t_quantile(p: float, df: int) -> float:
    """
    The inverse CDF: the ``t`` with ``P(T <= t) == p``.

    Found by bisection on :func:`_t_survival` rather than by a rational approximation, because a
    published approximation is accurate to about 1e-4 in the tails and the confidence intervals here
    are reported to two decimals of a two-point margin. Bisection costs microseconds and is exact to
    the CDF's own precision.

    :param p: Probability in ``(0, 1)``.
    :param df: Degrees of freedom, at least 1.

    :returns: The quantile.

    :raises OLMoConfigurationError: If ``p`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < p < 1.0:
        raise OLMoConfigurationError(f"a quantile needs a probability in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    sign = 1.0 if p > 0.5 else -1.0
    tail = min(p, 1.0 - p)
    low, high = 0.0, 2.0
    while _t_survival(high, df) > tail and high < 1e13:
        high *= 2.0
    for _ in range(400):
        middle = 0.5 * (low + high)
        if _t_survival(middle, df) > tail:
            low = middle
        else:
            high = middle
        if high - low < 1e-13 * max(1.0, high):
            break
    return sign * 0.5 * (low + high)


def _t_statistic(difference: float, standard_error: float) -> float:
    """
    ``difference / standard_error``, taking the limit when the replicates agreed exactly.

    Every test in this module is the same statistic recentred on a different null, so the zero-SE
    convention is written once: a nonzero difference over no spread is an infinitely strong rejection,
    and zero over zero is no evidence either way. Both are limits of the ratio and both feed
    :func:`_t_two_sided_p`, which accepts infinities. See :func:`pooled_trend` for why an observed zero
    is a warning rather than a triumph.

    :param difference: Estimate minus the null value.
    :param standard_error: Standard error of the estimate, never negative.

    :returns: The t statistic, possibly ``±inf``.
    """
    if standard_error > 0.0:
        return difference / standard_error
    if difference == 0.0:
        return 0.0
    return math.copysign(math.inf, difference)


# --- one replicate's observations -----------------------------------------------------------------


@dataclass(frozen=True)
class SeedBlock:
    """
    One replicate's cells: the demands it was run at and the accuracy each one reached.

    A block is the experiment's independent unit. PRD 16.5 is precise about what makes it one: a
    replicate varies ``TransformerConfig.init_seed`` and the data order while the corpus, the
    reasoning items and their volumes stay fixed, so the cells inside a block are a paired set over
    one fixed set of facts and the blocks are the things that were randomised.

    :param replicate: Which replicate this is -- :attr:`factcrowd.cells.CellSpec.replicate`. Carried
        so a slope can be traced back to the runs that produced it.
    :param demands: Demanded fact bits per parameter, one per cell, on the non-embedding basis and
        **including the name term** (PRD 3.1: both sweeps must be plotted on the same x definition).
    :param endpoint: Which endpoint these accuracies came from. Blocks that disagree are refused: a slope
        pooled across two endpoints is not a slope of either.
    :param row: Ladder row. Slopes are not exchangeable across widths (PRD P5), so a pooled trend runs
        per row.
    :param sweep: ``"count"`` or ``"entropy"``. The two are different experiments (PRD 3.1) and must not
        be pooled.
    :param step: The checkpoint these came from. A trend mixing checkpoints is measuring training
        progress, not demand.
    :param eval_items: How many held-out items each accuracy is over. Differing ``n`` means differing
        measurement noise, which the paired design assumes away.
    :param accuracies: Reasoning accuracy per cell, as **fractions** in ``[0, 1]`` --
        :attr:`~factcrowd.measure.endpoints.EndpointResult.accuracy`, not a percentage.
    """

    replicate: int
    demands: Tuple[float, ...]
    accuracies: Tuple[float, ...]
    endpoint: str = ""
    row: str = ""
    sweep: str = ""
    step: int = -1
    eval_items: int = -1

    def __post_init__(self) -> None:
        if len(self.demands) != len(self.accuracies):
            raise OLMoConfigurationError(
                f"replicate {self.replicate}: {len(self.demands)} demands against "
                f"{len(self.accuracies)} accuracies, so at least one cell is unpaired"
            )
        if len(self.demands) < _MIN_POINTS_PER_BLOCK:
            raise OLMoConfigurationError(
                f"replicate {self.replicate}: a slope needs at least {_MIN_POINTS_PER_BLOCK} cells, "
                f"got {len(self.demands)}"
            )
        if not all(math.isfinite(x) for x in self.demands):
            raise OLMoConfigurationError(
                f"replicate {self.replicate}: a demand is not finite, so its cell was never placed"
            )
        if max(self.demands) - min(self.demands) <= 0.0:
            raise OLMoConfigurationError(
                f"replicate {self.replicate}: every cell sits at demand {self.demands[0]}, so there "
                f"is no independent variable and the slope would divide by zero"
            )
        for accuracy in self.accuracies:
            # The trap this catches is 44.0 for 44%. It would survive every other check here and come
            # out as a slope 100x too large, which is well inside the range a reader would believe.
            if not 0.0 <= accuracy <= 1.0:
                raise OLMoConfigurationError(
                    f"replicate {self.replicate}: accuracy {accuracy} is not a fraction in [0, 1] "
                    f"-- these are fractions, not percentage points"
                )

    @classmethod
    def from_endpoints(
        cls, replicate: int, observations: Sequence[Tuple[float, EndpointResult]]
    ) -> "SeedBlock":
        """
        Build a block from what a scorer returned, one ``(demand, result)`` pair per cell.

        The refusals here are the reason to use this instead of the constructor. A slope's cell
        weights sum to zero, so anything **shared** by every cell in a block -- the eval set's item
        difficulty, the endpoint's floor -- cancels out of the slope exactly, which is the mechanism
        PRD 8.5's first fix buys with one frozen checksummed eval set. Anything that *varies* between
        cells does not cancel and enters the slope indistinguishably from a treatment effect. So a
        block whose cells were scored on differently sized sets, or against different floors, is
        rejected rather than averaged.

        :param replicate: Which replicate these runs are.
        :param observations: ``(demanded bits per parameter, endpoint score)`` per cell.

        :returns: The block.

        :raises OLMoConfigurationError: If the cells disagree about the endpoint, the floor or the
            size of the eval set, or if :class:`SeedBlock`'s own validation fails.
        """
        if not observations:
            raise OLMoConfigurationError(f"replicate {replicate}: no cells to build a block from")
        first = observations[0][1]
        for _, result in observations[1:]:
            if result.name != first.name:
                raise OLMoConfigurationError(
                    f"replicate {replicate}: mixes endpoints '{first.name}' and '{result.name}', "
                    f"and a slope across two endpoints measures neither"
                )
            if result.n_total != first.n_total:
                raise OLMoConfigurationError(
                    f"replicate {replicate}, endpoint '{first.name}': {result.n_total:,} items "
                    f"against {first.n_total:,} elsewhere, so the frozen eval set was not shared "
                    f"and item difficulty no longer cancels out of the slope"
                )
            if abs(result.floor - first.floor) > 1e-12:
                raise OLMoConfigurationError(
                    f"replicate {replicate}, endpoint '{first.name}': floor {result.floor} against "
                    f"{first.floor} elsewhere, so the instrument changed across the sweep"
                )
        return cls(
            replicate=replicate,
            demands=tuple(float(demand) for demand, _ in observations),
            accuracies=tuple(result.accuracy for _, result in observations),
        )

    @property
    def span(self) -> float:
        """Bits per parameter between the block's lowest and highest cell -- the swept range."""
        return max(self.demands) - min(self.demands)


def seed_slope(block: SeedBlock) -> float:
    """
    One replicate's OLS slope: percentage points of accuracy per demanded bit per parameter.

    The unit conversion lives here and nowhere else. Accuracies arrive as fractions and leave as
    percentage points, so no caller multiplies by 100 a second time.

    Written as the centred cross-product rather than through ``np.polyfit``, which solves a
    least-squares system through a Vandermonde matrix and returns coefficients in an order that is
    easy to read backwards. For a straight line the closed form is exact and obviously the thing it
    claims to be.

    :param block: One replicate's cells.

    :returns: The slope in pp per bit/param. Negative is crowding: accuracy falling as the corpus
        demands more of the model's capacity.
    """
    x = np.asarray(block.demands, dtype=np.float64)
    y = 100.0 * np.asarray(block.accuracies, dtype=np.float64)
    centred = x - x.mean()
    return float(centred @ (y - y.mean()) / (centred @ centred))


# --- the headline test ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendResult:
    """
    The one-sample t-test across per-seed slopes.

    :param n_blocks: Replicates, i.e. independent observations. Not the number of cells; see the
        module docstring for why the difference is the whole point.
    :param slope_mean: Mean per-seed slope, in pp per bit/param.
    :param slope_sd: Between-seed standard deviation of the slope, ``ddof=1``. This is the sigma PRD
        8.6's G7 requires published alongside any null.
    :param slope_se: Standard error of :attr:`slope_mean`, ``slope_sd / sqrt(n_blocks)``.
    :param t_statistic: ``slope_mean / slope_se``. Infinite when the seeds agreed exactly; see
        :func:`pooled_trend`.
    :param df: ``n_blocks - 1``.
    """

    n_blocks: int
    slope_mean: float
    slope_sd: float
    slope_se: float
    t_statistic: float
    df: int

    def __post_init__(self) -> None:
        if self.n_blocks < 2:
            raise OLMoConfigurationError(
                f"a between-seed test needs at least 2 replicates, got {self.n_blocks}"
            )
        if self.df != self.n_blocks - 1:
            raise OLMoConfigurationError(
                f"df is {self.df} for {self.n_blocks} blocks, but the between-seed test spends "
                f"exactly one degree of freedom on the mean"
            )
        if self.slope_sd < 0.0:
            raise OLMoConfigurationError(f"slope_sd is {self.slope_sd}, which is not a deviation")

    @property
    def p_value(self) -> float:
        """
        Two-sided p against ``H0: slope == 0``.

        **A large p is not a null result.** It is the reason :func:`tost` exists: report it beside
        :meth:`confidence_interval` and :func:`minimum_detectable_effect`, never alone.
        """
        return _t_two_sided_p(self.t_statistic, self.df)

    def confidence_interval(self, level: float = 0.90) -> Tuple[float, float]:
        """
        Confidence interval on the mean slope, in pp per bit/param.

        90% by default, matching PRD 8.5's instruction to report the 90% CI on ``D``: it is the
        interval that corresponds to the two one-sided 5% tests in :func:`tost`, so quoting a 95%
        interval beside a TOST verdict would be quoting two different alphas.

        :param level: Coverage, e.g. ``0.90``.

        :returns: ``(low, high)``.

        :raises OLMoConfigurationError: If ``level`` is not strictly inside ``(0, 1)``.
        """
        if not 0.0 < level < 1.0:
            raise OLMoConfigurationError(f"'level' must be in (0, 1), got {level}")
        half_width = _t_quantile(0.5 + 0.5 * level, self.df) * self.slope_se
        return (self.slope_mean - half_width, self.slope_mean + half_width)

    def summary(self) -> Dict[str, object]:
        """
        A flat mapping for logging, carrying what PRD 8.6's G7 requires a null to be published with.

        :returns: The fields, the p-value, the 90% interval, and the MDE at the conventional 5% and
            80% -- all in pp per bit/param.
        """
        low, high = self.confidence_interval()
        return {
            "n_blocks": self.n_blocks,
            "slope_mean_pp_per_bit": round(self.slope_mean, 6),
            "slope_sd_pp_per_bit": round(self.slope_sd, 6),
            "slope_se_pp_per_bit": round(self.slope_se, 6),
            "t_statistic": round(self.t_statistic, 6),
            "df": self.df,
            "p_value": round(self.p_value, 6),
            "ci90_low_pp_per_bit": round(low, 6),
            "ci90_high_pp_per_bit": round(high, 6),
            "mde_pp_per_bit": round(minimum_detectable_effect(self.slope_sd, self.n_blocks), 6),
        }


def check_blocks(blocks: Sequence[SeedBlock], *, required_levels: int = 0) -> None:
    """
    Refuse a set of blocks that is not a paired design over one treatment grid.

    Every check here corresponds to something the old code accepted. An adversarial pass fed it three
    blocks all labelled ``replicate=0`` -- three readings of one seed reported as three replicates, so the
    between-seed variance was noise about nothing -- and two blocks swept over *different* demand grids,
    which is two experiments averaged into one slope.

    The design's whole claim is that a block differs from its neighbours **only** in initialisation and
    data order. That is unverifiable from the numbers alone, so the identity travels with the block and is
    checked here.

    :param blocks: The blocks.
    :param required_levels: If set, the number of treatment levels each block must carry. The entropy
        sweep is six cells; a trend fitted through three of them is a different, weaker claim and should
        not be able to masquerade as the pre-registered one.

    :raises OLMoConfigurationError: On duplicate replicates, a mismatched grid, or mismatched identity.
    """
    if not blocks:
        raise OLMoConfigurationError("no blocks were given")

    replicates = [block.replicate for block in blocks]
    if len(set(replicates)) != len(replicates):
        duplicated = sorted({r for r in replicates if replicates.count(r) > 1})
        raise OLMoConfigurationError(
            f"replicate ids {duplicated} appear more than once. Repeated readings of one seed are not "
            f"replicates, and averaging them reports a between-seed variance that no seed produced."
        )

    reference = blocks[0]
    for block in blocks[1:]:
        if len(block.demands) != len(reference.demands) or any(
            abs(a - b) > 1e-6 for a, b in zip(block.demands, reference.demands)
        ):
            raise OLMoConfigurationError(
                f"replicate {block.replicate} was run at demands {block.demands} but replicate "
                f"{reference.replicate} at {reference.demands}. A paired design needs one treatment "
                f"grid; two grids with the same span are still two experiments."
            )
        for field in ("endpoint", "row", "sweep", "step", "eval_items"):
            here, there = getattr(block, field), getattr(reference, field)
            if here != there:
                raise OLMoConfigurationError(
                    f"replicate {block.replicate} has {field}={here!r} but replicate "
                    f"{reference.replicate} has {field}={there!r}. Blocks that differ in anything but "
                    f"initialisation and data order are not replicates of one another."
                )

    if required_levels and len(reference.demands) != required_levels:
        raise OLMoConfigurationError(
            f"each block carries {len(reference.demands)} treatment levels, not the {required_levels} "
            f"the pre-registered design states. A trend through a subset is a weaker claim and has to be "
            f"labelled as one rather than reported in its place."
        )


def pooled_trend(blocks: Sequence[SeedBlock]) -> TrendResult:
    """
    Pool the replicates: a one-sample t-test over their per-seed slopes.

    Pooling happens over *blocks*, not over cells. On a balanced design this returns exactly the point
    estimate a single regression over all cells would return, and a standard error that is larger and
    correct -- the module docstring gives the argument and a test pins the equality.

    A note on the degenerate case, because it will occur in tests and could occur in a two-seed pilot:
    when every replicate returns the same slope the sample SD is 0, the SE is 0, and the t statistic is
    ``±inf`` (or 0 when the common slope is itself 0). That is the correct limit and is reported rather
    than smoothed, but **an observed SD of zero is not evidence of zero seed variance** -- with three
    replicates it is a coincidence with real probability, and it drives
    :func:`minimum_detectable_effect` to 0, which would advertise infinite resolution. PRD 8.5 flags
    the same failure mode from the other direction: the deleted monotonicity re-run rule shrank the
    variance estimate to 0.57 sigma, and a shrunken variance makes an equivalence test *falsely*
    declare equivalence.

    :param blocks: One per replicate. At least two, because one seed says nothing about seed variance.

    :returns: The :class:`TrendResult`.

    :raises OLMoConfigurationError: If fewer than two blocks are given.
    """
    check_blocks(blocks)
    if len(blocks) < 2:
        raise OLMoConfigurationError(
            f"a between-seed t-test needs at least 2 replicates, got {len(blocks)} -- one seed "
            f"measures the model that was trained, not the distribution it was drawn from"
        )
    slopes = np.array([seed_slope(block) for block in blocks], dtype=np.float64)
    n_blocks = len(slopes)
    mean = float(slopes.mean())
    sd = float(slopes.std(ddof=1))
    se = sd / math.sqrt(n_blocks)
    return TrendResult(
        n_blocks=n_blocks,
        slope_mean=mean,
        slope_sd=sd,
        slope_se=se,
        t_statistic=_t_statistic(mean, se),
        df=n_blocks - 1,
    )


# --- equivalence, and the weaker claim it is often confused with ------------------------------------


def _resolve_span(blocks: Sequence[SeedBlock], span: Optional[float]) -> float:
    """
    The bits/param range the margin is quoted over.

    :param blocks: The replicates.
    :param span: An explicit span, or ``None`` to read it off the blocks.

    :returns: The span in bits per parameter.

    :raises OLMoConfigurationError: If ``span`` is not positive, or if the blocks disagree about
        theirs -- two replicates swept over different ranges have effects in different units, and
        averaging them silently rescales one of the two.
    """
    if span is not None:
        if not span > 0.0:
            raise OLMoConfigurationError(f"'span' must be positive, got {span}")
        return span
    spans = [block.span for block in blocks]
    if max(spans) - min(spans) > 1e-9 * max(spans):
        raise OLMoConfigurationError(
            f"replicates swept different ranges ({min(spans):.4f} to {max(spans):.4f} bits/param), "
            f"so an effect 'across the sweep' is not one quantity -- pass 'span' to say which range "
            f"the margin refers to"
        )
    return spans[0]


def _effect(
    blocks: Sequence[SeedBlock], margin_pp: float, alpha: float, span: Optional[float]
) -> Tuple[TrendResult, float, float, float]:
    """
    Shared setup for the two margin tests: validate, fit, and rescale the slope to the sweep's ends.

    :param blocks: The replicates.
    :param margin_pp: The margin, validated here so both tests refuse the same inputs.
    :param alpha: The level, validated here for the same reason.
    :param span: An explicit span, or ``None``.

    :returns: ``(trend, effect_pp, se_pp, span)``.

    :raises OLMoConfigurationError: If the margin is not positive or alpha is outside ``(0, 0.5)``.
    """
    if not margin_pp > 0.0:
        raise OLMoConfigurationError(f"'margin_pp' must be positive, got {margin_pp}")
    # At alpha >= 0.5 the two one-sided tests can both reject on an estimate outside the margin, so
    # TOST would declare equivalence for an effect it has just measured as too large.
    if not 0.0 < alpha < 0.5:
        raise OLMoConfigurationError(f"'alpha' must be in (0, 0.5), got {alpha}")
    resolved_span = _resolve_span(blocks, span)
    trend = pooled_trend(blocks)
    return trend, trend.slope_mean * resolved_span, trend.slope_se * resolved_span, resolved_span


@dataclass(frozen=True)
class EquivalenceResult:
    """
    A two-one-sided-test verdict: the effect is bounded on **both** sides.

    This is what P3 claims and what :func:`tost` produces. Contrast
    :class:`NonInferiorityResult`, which bounds the decline only.

    :param n_blocks: Replicates.
    :param df: ``n_blocks - 1``.
    :param span: Bits per parameter between the sweep's ends, which is what turns a slope into an
        end-to-end effect.
    :param effect_pp: Estimated change in accuracy across the whole sweep, in percentage points.
        Signed as a **change**: negative is a decline. PRD 8.5's ``D`` is its negation.
    :param se_pp: Standard error of :attr:`effect_pp`, from the between-seed SD.
    :param margin_pp: The equivalence margin, applied as ``±margin_pp``.
    :param alpha: Level of each one-sided test.
    :param p_lower: One-sided p against ``H0: effect <= -margin_pp`` -- "the decline is at least the
        margin". Small means declines that large are excluded.
    :param p_upper: One-sided p against ``H0: effect >= +margin_pp``. Small means improvements that
        large are excluded. This is the half a one-sided rule leaves out.
    :param interval: The ``1 - 2*alpha`` confidence interval on :attr:`effect_pp`. TOST's verdict is
        exactly "this interval is inside the margin", and a test asserts the two agree.
    :param equivalent: Whether both one-sided tests reject at :attr:`alpha`.
    """

    n_blocks: int
    df: int
    span: float
    effect_pp: float
    se_pp: float
    margin_pp: float
    alpha: float
    p_lower: float
    p_upper: float
    interval: Tuple[float, float]
    equivalent: bool

    @property
    def p_value(self) -> float:
        """The TOST p-value: the larger of the two one-sided p's, since both must reject."""
        return max(self.p_lower, self.p_upper)

    @property
    def verdict(self) -> str:
        """
        The sentence to publish, phrased the way PRD 8.5 requires: what is excluded, never "no effect".
        """
        low, high = self.interval
        confidence = 100.0 * (1.0 - 2.0 * self.alpha)
        if self.se_pp <= _DEGENERATE_SE_PP:
            return (
                f"no verdict: the between-seed standard error is {self.se_pp:.2e}pp, so the seeds "
                f"produced identical scores and this measured no variance. A dead or saturated endpoint "
                f"looks exactly like this (n={self.n_blocks})"
            )
        if self.equivalent:
            return (
                f"changes larger than {self.margin_pp:.2f}pp in either direction are excluded across "
                f"{self.span:.3f} bits/param ({confidence:.0f}% CI "
                f"[{low:+.2f}, {high:+.2f}]pp, n={self.n_blocks})"
            )
        return (
            f"equivalence within {self.margin_pp:.2f}pp is not established: the {confidence:.0f}% CI "
            f"[{low:+.2f}, {high:+.2f}]pp leaves the margin (n={self.n_blocks})"
        )

    def summary(self) -> Dict[str, object]:
        """
        A flat mapping for logging.

        :returns: Every field plus the TOST p-value and the published sentence.
        """
        return {
            "test": "tost",
            "n_blocks": self.n_blocks,
            "df": self.df,
            "span_bits_per_param": round(self.span, 6),
            "effect_pp": round(self.effect_pp, 6),
            "se_pp": round(self.se_pp, 6),
            "margin_pp": self.margin_pp,
            "alpha": self.alpha,
            "p_lower": round(self.p_lower, 6),
            "p_upper": round(self.p_upper, 6),
            "p_value": round(self.p_value, 6),
            "interval_low_pp": round(self.interval[0], 6),
            "interval_high_pp": round(self.interval[1], 6),
            "equivalent": self.equivalent,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class NonInferiorityResult:
    """
    A one-sided verdict: the **decline** is bounded and the other direction is not tested.

    Not equivalence, and named so it cannot be quoted as equivalence. An arm that improved by ten
    points is non-inferior at any margin and is not flat, so this result licenses "declines greater
    than the margin are excluded" and nothing else. PRD 16.5 records the correction: the rule revision
    1 pre-registered as an equivalence test "is non-inferiority and is now called that."

    :param n_blocks: Replicates.
    :param df: ``n_blocks - 1``.
    :param span: Bits per parameter between the sweep's ends.
    :param effect_pp: Estimated change in accuracy across the whole sweep, in pp. Negative is a
        decline.
    :param se_pp: Standard error of :attr:`effect_pp`.
    :param margin_pp: The non-inferiority margin. The null is ``effect <= -margin_pp``.
    :param alpha: Level of the single one-sided test.
    :param p_value: One-sided p against ``H0: effect <= -margin_pp``.
    :param lower_bound: One-sided ``1 - alpha`` lower confidence bound on :attr:`effect_pp`. The
        bound is one-sided on purpose: there is no upper end, because none was tested.
    :param non_inferior: Whether the test rejects at :attr:`alpha`.
    """

    n_blocks: int
    df: int
    span: float
    effect_pp: float
    se_pp: float
    margin_pp: float
    alpha: float
    p_value: float
    lower_bound: float
    non_inferior: bool

    @property
    def verdict(self) -> str:
        """The sentence to publish, including the half of the parameter space left untested."""
        confidence = 100.0 * (1.0 - self.alpha)
        if self.se_pp <= _DEGENERATE_SE_PP:
            return (
                f"no verdict: the between-seed standard error is {self.se_pp:.2e}pp, so the seeds "
                f"produced identical scores and this measured no variance (n={self.n_blocks})"
            )
        if self.non_inferior:
            return (
                f"declines greater than {self.margin_pp:.2f}pp are excluded across "
                f"{self.span:.3f} bits/param ({confidence:.0f}% one-sided lower bound "
                f"{self.lower_bound:+.2f}pp, n={self.n_blocks}); improvements are not bounded, so "
                f"this is non-inferiority and not equivalence"
            )
        return (
            f"a decline of {self.margin_pp:.2f}pp or more is not excluded ({confidence:.0f}% "
            f"one-sided lower bound {self.lower_bound:+.2f}pp, n={self.n_blocks})"
        )

    def summary(self) -> Dict[str, object]:
        """
        A flat mapping for logging.

        :returns: Every field plus the published sentence.
        """
        return {
            "test": "non_inferiority",
            "n_blocks": self.n_blocks,
            "df": self.df,
            "span_bits_per_param": round(self.span, 6),
            "effect_pp": round(self.effect_pp, 6),
            "se_pp": round(self.se_pp, 6),
            "margin_pp": self.margin_pp,
            "alpha": self.alpha,
            "p_value": round(self.p_value, 6),
            "lower_bound_pp": round(self.lower_bound, 6),
            "non_inferior": self.non_inferior,
        }


def tost(
    blocks: Sequence[SeedBlock],
    *,
    margin_pp: float = EQUIVALENCE_MARGIN_PP,
    alpha: float = 0.05,
    span: Optional[float] = None,
) -> EquivalenceResult:
    """
    Two one-sided tests for equivalence: is the end-to-end change inside ``±margin_pp``?

    **This is the test P3 refers to and the one to quote for a flat result.** It rejects two nulls at
    once -- that the sweep costs at least ``margin_pp`` and that it gains at least ``margin_pp`` -- so
    passing it bounds the effect on both sides. Its verdict is identical to "the ``1 - 2*alpha``
    interval lies inside the margin", which is the interval PRD 8.5 asks to be reported and is
    returned in :attr:`EquivalenceResult.interval`.

    Failing it is not evidence of an effect, only of an interval too wide to place;
    :func:`minimum_detectable_effect` is what says how wide.

    :param blocks: One per replicate, at least two.
    :param margin_pp: Equivalence margin in accuracy points **across the whole sweep**, not per bit.
        Defaults to :data:`EQUIVALENCE_MARGIN_PP`.
    :param alpha: Level of each one-sided test. 0.05 gives the 90% interval PRD 8.5 asks for.
    :param span: Bits per parameter the margin is quoted over. Defaults to the range the blocks
        actually swept, which is the honest reading; pass it explicitly to hold the x-axis fixed while
        comparing arms that stopped at different cells.

    :returns: The :class:`EquivalenceResult`.

    :raises OLMoConfigurationError: If there are fewer than two blocks, if ``margin_pp`` is not
        positive, if ``alpha`` is outside ``(0, 0.5)``, or if the blocks swept different ranges.
    """
    trend, effect, standard_error, resolved_span = _effect(blocks, margin_pp, alpha, span)
    # Each null sits at one end of the margin, so each t is the estimate's distance from that end in
    # standard errors -- pooled_trend's statistic, recentred twice.
    p_lower = _t_survival(_t_statistic(effect + margin_pp, standard_error), trend.df)
    # The lower tail by symmetry rather than as 1 - survival: against H0 at +margin the statistic is
    # far negative exactly when the result is most convincing, and that is where subtracting from one
    # would cancel away the significant digits.
    p_upper = _t_survival(_t_statistic(margin_pp - effect, standard_error), trend.df)
    half_width = _t_quantile(1.0 - alpha, trend.df) * standard_error
    return EquivalenceResult(
        n_blocks=trend.n_blocks,
        df=trend.df,
        span=resolved_span,
        effect_pp=effect,
        se_pp=standard_error,
        margin_pp=margin_pp,
        alpha=alpha,
        p_lower=p_lower,
        p_upper=p_upper,
        interval=(effect - half_width, effect + half_width),
        equivalent=bool(p_lower < alpha and p_upper < alpha and standard_error > _DEGENERATE_SE_PP),
    )


def non_inferiority(
    blocks: Sequence[SeedBlock],
    *,
    margin_pp: float = EQUIVALENCE_MARGIN_PP,
    alpha: float = 0.05,
    span: Optional[float] = None,
) -> NonInferiorityResult:
    """
    The weaker, one-sided claim: are declines of ``margin_pp`` or more excluded?

    **Not an equivalence test.** It is revision 1's pre-registered rule, kept because "storing facts
    does not cost us two points of reasoning" is a real and useful claim, and renamed because PRD 16.5
    found it being read as flatness. It tests ``H0: effect <= -margin_pp`` and leaves the upper half of
    the parameter space alone: an arm that *gained* twenty points passes here and fails :func:`tost`,
    and that contrast is the entire reason both functions exist. Use :func:`tost` for P3.

    :param blocks: One per replicate, at least two.
    :param margin_pp: Non-inferiority margin in accuracy points across the whole sweep. Defaults to
        :data:`EQUIVALENCE_MARGIN_PP`.
    :param alpha: Level of the single one-sided test.
    :param span: Bits per parameter the margin is quoted over. See :func:`tost`.

    :returns: The :class:`NonInferiorityResult`.

    :raises OLMoConfigurationError: On the same conditions as :func:`tost`.
    """
    trend, effect, standard_error, resolved_span = _effect(blocks, margin_pp, alpha, span)
    # Deliberately only the lower half of tost()'s pair of tests. Nothing here looks at +margin_pp.
    p_value = _t_survival(_t_statistic(effect + margin_pp, standard_error), trend.df)
    return NonInferiorityResult(
        n_blocks=trend.n_blocks,
        df=trend.df,
        span=resolved_span,
        effect_pp=effect,
        se_pp=standard_error,
        margin_pp=margin_pp,
        alpha=alpha,
        p_value=p_value,
        lower_bound=effect - _t_quantile(1.0 - alpha, trend.df) * standard_error,
        non_inferior=bool((p_value < alpha) and standard_error > _DEGENERATE_SE_PP),
    )


def minimum_detectable_effect(
    slope_sd: float, n_blocks: int, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """
    The smallest effect this many replicates could have found -- what a null has to be reported with.

    PRD 8.6's G7 requires sigma and the MDE published beside any result, and PRD 8.5 is the reason:
    all four of this programme's prior nulls are consistent with an unmeasured sigma, and a null
    without an MDE cannot be told apart from a measurement that never had the resolution to see the
    effect it was looking for.

    Uses ``(t_{1-alpha/2,df} + t_{power,df}) * sd / sqrt(n)``, the shifted-t approximation to the exact
    noncentral-t calculation. Two notes on that. It is the *t* quantile in both terms: PRD 8.5's first
    error was pairing a t threshold with a normal statistic, which reported 79% power where the exact
    figure was 77.2%. And the approximation remains mildly optimistic against exact noncentral-t at
    small df -- by well under a point of power at ``df=2`` -- so treat a returned MDE as a lower bound
    on what would really be needed.

    :param slope_sd: Between-replicate standard deviation, in whatever unit the answer is wanted in.
        :attr:`TrendResult.slope_sd` gives pp per bit/param; multiply it by the sweep's span first to
        get an MDE in pp across the sweep, comparable with :data:`EQUIVALENCE_MARGIN_PP`.
    :param n_blocks: Replicates, at least two.
    :param alpha: Two-sided significance level.
    :param power: Target power.

    :returns: The minimum detectable effect, in the units of ``slope_sd``.

    :raises OLMoConfigurationError: If the SD is negative, there are fewer than two blocks, or alpha
        or power is outside ``(0, 1)``.
    """
    if slope_sd < 0.0:
        raise OLMoConfigurationError(f"'slope_sd' must not be negative, got {slope_sd}")
    if n_blocks < 2:
        raise OLMoConfigurationError(
            f"an MDE needs at least 2 replicates, got {n_blocks} -- with one there is no variance "
            f"estimate and therefore no resolution to report"
        )
    if not 0.0 < alpha < 1.0:
        raise OLMoConfigurationError(f"'alpha' must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise OLMoConfigurationError(f"'power' must be in (0, 1), got {power}")
    df = n_blocks - 1
    quantile_sum = _t_quantile(1.0 - 0.5 * alpha, df) + _t_quantile(power, df)
    return quantile_sum * slope_sd / math.sqrt(n_blocks)
