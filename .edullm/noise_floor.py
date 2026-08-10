#!/usr/bin/env python3
"""Measure the two numbers stage 2 is gated on, and fix the arithmetic that reads them.

A SIBLING TO ``wandb_panels.py`` RATHER THAN AN EXTENSION OF IT, BECAUSE THEY ANSWER
DIFFERENT QUESTIONS. That module asks whether a *key* arrived and builds a chart over the ones
that did; nothing in it estimates anything. This one asks what the numbers in those keys are
worth, and every function in it is an estimator with a degrees-of-freedom count attached.
Keeping them apart means the estimators are importable and testable without W&B, a report
cannot fail because a variance did not converge, and neither file grows a second purpose.

WHAT THE MODULE IS FOR. Stage 1 of the tranche is five seeds of ``baseline`` and no treatment
arm. Two numbers have to come out of it, and both have to be frozen in the repository before
stage 2 is submitted, because either one estimated after a treatment arm is visible is a
researcher degree of freedom rather than a measurement:

``sigma-hat``       the across-seed standard deviation of the primary endpoint -- held-out
                    bits-per-byte -- at the final step, with a confidence interval and its
                    df stated. Five seeds of one arm is df = 4 and not df = 5, and the
                    interval that df carries is wide enough that quoting sigma-hat without
                    it would be quoting a rumour.

``per-source w``    inverse-variance weights over the seven held-out sources. Per-source seed
                    sigma spans an order of magnitude on DataDecide (arXiv 2504.11393), with
                    code-type sources several times noisier than web text, so the unweighted
                    mean the pre-registration currently reads is inefficient by construction.

A third estimator, ``rho-hat``, cannot be computed from the baseline at all -- pairing is
across arms and there is only one arm -- so it is written and tested here against synthetic
data with a known truth, and pre-committed, so that stage 2 lands on an estimator nobody chose
after seeing it.

WHAT IS DELIBERATELY NOT HERE: GENERALIZED LEAST SQUARES. The efficient weight vector over
seven correlated sources is proportional to ``Sigma^-1 1``, and a 7x7 covariance estimated
from five seeds has rank 4. It is singular, its inverse does not exist, and every ridge or
shrinkage that would make one exist is a knob set after the fact. So the weights here come
from the *diagonal* only -- see :func:`inverse_variance_weights` -- and the cross-source
covariance is used exactly once, in :func:`realised_sigma`, to *report* what the resulting
fixed weight vector actually achieves rather than to choose it.

    python .edullm/noise_floor.py --self-test
    python .edullm/noise_floor.py --dry-run --group hyper-connections-370m
    python .edullm/noise_floor.py --group hyper-connections-370m --freeze .edullm/noise-floor.json
    python .edullm/noise_floor.py --mde-table

Needs ``scipy``, which is in the ``dev`` extra beside ``matplotlib`` and for the same reason:
this runs on a laptop against W&B and never inside a container. The training image installs
``.[wandb]`` and not ``.[all]``, so nothing here reaches a run.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: The primary endpoint, per source. Bits-per-byte and not cross-entropy because the
#: pre-registration names BPB as the endpoint, and because ``BitsPerByteCallback`` writes it
#: beside every CE key so no conversion happens on this side of the wire.
PRIMARY_METRIC = "eval/lm/{source}/BPB"

#: The seven sources ``regmix-10b-v1`` declares a validation shard for. Read off a run's own
#: config where one is available -- see :func:`sources_from_config`, which derives them with
#: ``hyper_connection_arms.source_label`` from the shard paths the run actually evaluated on --
#: and used as the fallback only when no run has landed yet.
HELD_OUT_SOURCES: Tuple[str, ...] = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)

#: What the pre-registration quotes every threshold at, in nats of held-out cross-entropy.
#: A planning value from the literature and not a measurement of this configuration; the whole
#: point of stage 1 is to replace it. Every MDE scales linearly with it.
PLANNING_SIGMA_NATS = 0.010

#: Divide a figure in nats by this for the same figure in bits-per-byte:
#: ``BPB = CE_nats / (bytes_per_token * ln 2)``. Identical across arms, so it moves the level
#: of every number here and no comparison between two of them.
NATS_PER_BPB = 4.57 * math.log(2)


# ---------------------------------------------------------------------------------------
# (a) sigma-hat: the noise floor, with the df that governs its interval.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SigmaEstimate:
    """A standard deviation, the df behind it, and what that df costs in certainty."""

    sigma: float
    """The point estimate."""

    df: int
    """
    Degrees of freedom. For ``g`` arms of ``n`` seeds this is ``g * (n - 1)``, so five seeds
    of one arm is 4. The number is carried rather than recomputed at the call site because it
    is the quantity every interval and every MDE downstream is governed by, and it is the one
    the pre-registration got wrong in its paired table.
    """

    ci_low: float
    ci_high: float
    """
    The chi-square interval on sigma: ``sigma * sqrt(df / chi2_{1-a/2,df})`` to
    ``sigma * sqrt(df / chi2_{a/2,df})``. Asymmetric, and much wider than intuition allows at
    small df -- see :func:`variance_interval_span`.
    """

    confidence: float
    n_observations: int
    n_groups: int

    @property
    def span(self) -> float:
        """End to end, as a multiplicative factor. The honest width of the noise floor."""
        return self.ci_high / self.ci_low

    @property
    def sigma_unbiased(self) -> float:
        """
        The same estimate with the small-sample bias in a standard deviation taken out.

        A SECOND WAY THIS DESIGN IS OPTIMISTIC BY A FEW PERCENT, AND IT IS THE SAME SHAPE OF
        MISTAKE AS THE DF ONE. ``s`` is unbiased for the *variance* and not for the standard
        deviation: ``E[s] = c4(df) sigma``, and ``c4(4) = 0.9400``. So the sample standard
        deviation of five seeds understates sigma by 6% on average, every minimum detectable
        effect is linear in sigma, and quoting the raw ``s`` therefore quotes an MDE 6% smaller
        than the design really has. Dividing by ``c4`` costs nothing and is what
        :func:`mde_from` uses.

        The correction belongs to the *point estimate* and not to the inference. The t
        machinery in :func:`power_of` is built on the distribution of ``s`` itself and already
        carries this; correcting the input to a t test as well would count it twice.
        """
        return self.sigma / c4(self.df)

    @property
    def standard_error_of_a_mean(self) -> float:
        """``sigma / sqrt(n)`` for one group of the size that was measured."""
        return self.sigma / math.sqrt(self.n_observations / max(self.n_groups, 1))


def c4(df: int) -> float:
    """
    The bias factor of a standard deviation, ``E[s] / sigma``, at a given df.

    ``sqrt(2/df) * Gamma((df+1)/2) / Gamma(df/2)``. It is 0.7979 at df = 1, 0.9400 at df = 4,
    0.9594 at df = 6, 0.9794 at df = 12, and approaches 1 from below.

    :param df: Degrees of freedom of the variance estimate.

    :returns: The multiplicative bias, always below one.

    :raises ValueError: If the df is below one.
    """
    if df < 1:
        raise ValueError(f"c4 needs at least one degree of freedom, got {df}")
    return math.sqrt(2.0 / df) * math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))


def variance_interval_span(df: int, confidence: float = 0.95) -> float:
    """
    How wide the interval on a standard deviation is at a given df, as a factor.

    The number that decides whether a measured noise floor is a floor or a rumour, and it is
    brutal at the small end: df = 2 spans a factor of 12.1, df = 4 spans 4.8 and df = 6 spans
    3.4. Five seeds of one arm buy the middle one, which is why stage 1 is five and not three.

    :param df: Degrees of freedom of the variance estimate.
    :param confidence: Two-sided coverage.

    :returns: ``ci_high / ci_low`` for sigma.
    """
    alpha = 1.0 - confidence
    lo = math.sqrt(df / stats.chi2.ppf(1.0 - alpha / 2.0, df))
    hi = math.sqrt(df / stats.chi2.ppf(alpha / 2.0, df))
    return hi / lo


def pooled_sigma(groups: Sequence[Sequence[float]], confidence: float = 0.95) -> SigmaEstimate:
    """
    The pooled within-group standard deviation, with its chi-square interval.

    Pooled rather than averaged: the sums of squares are added and divided by the summed df,
    so a group of five and a group of three contribute in proportion to what they know. With
    a single group this is the ordinary sample standard deviation and the df is ``n - 1``.

    :param groups: One sequence of observations per group. Groups of fewer than two
        observations carry no information about spread and are skipped rather than erroring,
        so a partially-landed fan-out still returns something.
    :param confidence: Two-sided coverage for the interval.

    :returns: The estimate.

    :raises ValueError: If no group has two or more observations, which is the case where
        there is no variance estimate at all rather than a bad one.
    """
    total_ss = 0.0
    df = 0
    n_observations = 0
    n_groups = 0
    for group in groups:
        values = np.asarray(list(group), dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        total_ss += float(((values - values.mean()) ** 2).sum())
        df += values.size - 1
        n_observations += int(values.size)
        n_groups += 1

    if df == 0:
        raise ValueError(
            "no group has two or more finite observations, so there is no variance estimate "
            "to make. This is what a fan-out that has not landed looks like."
        )

    sigma = math.sqrt(total_ss / df)
    alpha = 1.0 - confidence
    return SigmaEstimate(
        sigma=sigma,
        df=df,
        ci_low=sigma * math.sqrt(df / stats.chi2.ppf(1.0 - alpha / 2.0, df)),
        ci_high=sigma * math.sqrt(df / stats.chi2.ppf(alpha / 2.0, df)),
        confidence=confidence,
        n_observations=n_observations,
        n_groups=n_groups,
    )


@dataclass(frozen=True)
class SigmaTrajectory:
    """Sigma at a sequence of checkpoints, and whether it has stopped moving."""

    steps: Tuple[int, ...]
    sigma: Tuple[float, ...]
    ci_low: Tuple[float, ...]
    ci_high: Tuple[float, ...]
    df: int
    reference_step: int
    """The checkpoint the final one is compared against, nearest to half the horizon."""

    ratio: float
    """``sigma(final) / sigma(reference)``. Above 1 means it is still growing."""

    ratio_ci: Tuple[float, float]
    """
    Percentile interval from a bootstrap over *seeds*, not over steps. The sigmas at two
    checkpoints of the same five runs are not independent -- they are five trajectories read
    twice -- so resampling the seeds is the only resampling that respects the dependence.
    At five seeds the bootstrap is coarse and the interval should be read as an order of
    magnitude on the uncertainty rather than as a calibrated 95%.
    """

    settled: bool
    """True when the ratio interval contains 1. A weak test, and labelled as one."""


def sigma_trajectory(
    values: np.ndarray,
    steps: Sequence[int],
    confidence: float = 0.95,
    bootstrap: int = 4000,
    rng_seed: int = 0,
) -> SigmaTrajectory:
    """
    Sigma at each checkpoint, so the horizon question can be answered rather than assumed.

    WHY THIS IS NOT A COSMETIC PANEL. The tranche traded token horizon for replicates on the
    strength of ``sigma ~ D^-0.172``, which says sigma is still falling at 4.72B tokens. If
    the measured trajectory is flat by step 6,000 the trade was better than argued; if sigma
    is still *rising* at the final step then the endpoint is being read while the runs are
    still separating, and every MDE in the plan is quoted against a quantity that has not
    stopped moving. Both readings change how stage 2 is interpreted, and neither is visible
    from the final step alone.

    :param values: Shape ``(n_seeds, n_steps)``. Row order must be the same seed at every
        step, because the bootstrap resamples rows.
    :param steps: The checkpoint step of each column.
    :param confidence: Coverage for both the per-step chi-square intervals and the bootstrap.
    :param bootstrap: Bootstrap replicates over seeds.
    :param rng_seed: Fixed so the reported interval is reproducible.

    :returns: The trajectory.

    :raises ValueError: If the array is not two-dimensional, has fewer than two seeds, or
        does not have one step per column.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected (n_seeds, n_steps), got shape {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("a trajectory of standard deviations needs at least two seeds")
    if values.shape[1] != len(steps):
        raise ValueError(f"{values.shape[1]} columns against {len(steps)} steps")

    per_step = [pooled_sigma([values[:, j]], confidence) for j in range(values.shape[1])]
    order = list(steps)
    final = len(order) - 1
    target = order[final] / 2.0
    reference = min(range(len(order)), key=lambda j: abs(order[j] - target))
    if reference == final and final > 0:
        reference = final - 1

    def ratio_of(rows: np.ndarray) -> float:
        sub = values[rows, :]
        top = float(sub[:, final].std(ddof=1))
        bottom = float(sub[:, reference].std(ddof=1))
        return top / bottom if bottom > 0 else float("nan")

    rng = np.random.default_rng(rng_seed)
    n_seeds = values.shape[0]
    draws = [ratio_of(rng.integers(0, n_seeds, n_seeds)) for _ in range(bootstrap)]
    finite = np.asarray([d for d in draws if math.isfinite(d)], dtype=float)
    alpha = 1.0 - confidence
    if finite.size:
        lo = float(np.quantile(finite, alpha / 2.0))
        hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    else:
        lo, hi = float("nan"), float("nan")

    ratio = ratio_of(np.arange(n_seeds))
    return SigmaTrajectory(
        steps=tuple(int(s) for s in order),
        sigma=tuple(e.sigma for e in per_step),
        ci_low=tuple(e.ci_low for e in per_step),
        ci_high=tuple(e.ci_high for e in per_step),
        df=per_step[final].df,
        reference_step=int(order[reference]),
        ratio=ratio,
        ratio_ci=(lo, hi),
        settled=bool(lo <= 1.0 <= hi),
    )


# ---------------------------------------------------------------------------------------
# (b) per-source inverse-variance weights, from the diagonal only.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceWeights:
    """A weight vector over the held-out sources, and what it is worth."""

    sources: Tuple[str, ...]
    sigma: Tuple[float, ...]
    """Per-source across-seed sigma, each on ``n_seeds - 1`` df."""

    weights: Tuple[float, ...]
    """Normalized to sum to one, so the composite stays on the endpoint's own scale."""

    scheme: str
    """``"strata"`` or ``"inverse-variance"``. See :func:`inverse_variance_weights`."""

    strata: Tuple[int, ...]
    """Which variance stratum each source landed in. All zeros under inverse-variance."""

    df_per_weight: Tuple[int, ...]
    """
    How many df each source's weight rests on. This is the whole argument for strata: under
    inverse-variance every weight rests on the ``n - 1`` df of one source, and under two
    strata it rests on the pooled df of its group.
    """

    n_seeds: int
    unweighted_sigma: float
    weighted_sigma: float
    variance_reduction: float
    """
    ``(unweighted_sigma / weighted_sigma) ** 2``, measured on these five seeds. OPTIMISTIC:
    the weights were derived from the same five, so this is an in-sample number. It is the
    right thing to report anyway, beside the cross-validated figure, because when the frozen
    vector is applied to stage 2 it is a constant and the in-sample bias is gone -- what
    remains is whether it is a *good* constant, which is what the next field measures.
    """

    cross_validated_variance_reduction: float
    """
    The same ratio, leave-one-seed-out. Each seed's composite is formed with weights fitted
    on the other four and compared with their mean, which is the arrangement stage 2 will
    actually be in: a fixed vector meeting data it did not see.
    """

    diagonal_variance_reduction: float
    """
    What the reduction would be if the sources were independent, ``sum(w^2 s^2)`` against
    ``sum(s^2)/m^2``. Reported beside the realised figure because the gap between them is the
    cross-source covariance, and that gap is the reason a diagonal weighting is a compromise
    rather than the answer: a shared seed effect that moves every source together cannot be
    weighted away.
    """


def per_source_sigma(values: np.ndarray) -> np.ndarray:
    """
    Across-seed standard deviation of each source.

    :param values: Shape ``(n_seeds, n_sources)``.

    :returns: One sigma per source, each on ``n_seeds - 1`` df.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"expected (n_seeds >= 2, n_sources), got shape {values.shape}")
    return values.std(axis=0, ddof=1)


def realised_sigma(values: np.ndarray, weights: np.ndarray) -> float:
    """
    The across-seed sigma of the weighted composite, formed directly.

    THE ONLY PLACE THE CROSS-SOURCE COVARIANCE IS USED, AND IT IS USED TO REPORT AND NEVER TO
    CHOOSE. Forming the composite per seed and taking its sample standard deviation gives
    ``sqrt(w' Sigma w)`` without ever writing ``Sigma`` down, let alone inverting it. That is
    what makes this legitimate at df = 4 where GLS is not: a quadratic form in a fixed vector
    is estimable from five numbers, and ``Sigma^-1`` is not estimable at all.

    :param values: Shape ``(n_seeds, n_sources)``.
    :param weights: One weight per source. Not required to sum to one.

    :returns: The standard deviation across seeds of ``values @ weights``.
    """
    composite = np.asarray(values, dtype=float) @ np.asarray(weights, dtype=float)
    return float(composite.std(ddof=1))


def _weights_from_sigma(sigma: np.ndarray, scheme: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn per-source sigmas into weights and the stratum labels behind them.

    :returns: ``(weights summing to one, stratum label per source)``.
    """
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("every per-source sigma must be finite and positive to invert it")

    if scheme == "inverse-variance":
        strata = np.zeros(sigma.size, dtype=int)
        precision = 1.0 / sigma**2
        return precision / precision.sum(), strata

    if scheme != "strata":
        raise ValueError(f"unknown weighting scheme {scheme!r}")

    order = np.argsort(sigma)
    logs = np.log(sigma[order])
    smallest = 2 if logs.size >= 4 else 1
    cut = min(
        range(smallest, logs.size - smallest + 1),
        key=lambda c: float(
            ((logs[:c] - logs[:c].mean()) ** 2).sum() + ((logs[c:] - logs[c:].mean()) ** 2).sum()
        ),
    )
    strata = np.zeros(sigma.size, dtype=int)
    strata[order[cut:]] = 1

    precision = np.empty_like(sigma)
    for label in (0, 1):
        members = strata == label
        if not members.any():
            continue
        precision[members] = 1.0 / float((sigma[members] ** 2).mean())
    return precision / precision.sum(), strata


def _cross_validated_sigma(values: np.ndarray, scheme: Optional[str]) -> float:
    """
    Leave-one-seed-out sigma of a composite.

    With ``scheme`` set, the weights are refitted on the four retained seeds for every held-out
    seed, so the number includes the cost of having estimated them. With ``scheme`` None the
    weights are uniform and nothing is fitted, which is the comparison the reduction is taken
    against.

    The held-out seed's composite is compared with the mean of the four retained composites,
    and that residual has variance ``sigma^2 (1 + 1/(n-1))`` under independence, so the mean
    square is divided by that factor to come back to sigma.
    """
    values = np.asarray(values, dtype=float)
    n_seeds, n_sources = values.shape
    residuals = []
    for i in range(n_seeds):
        retained = np.delete(values, i, axis=0)
        if scheme is None:
            weights = np.full(n_sources, 1.0 / n_sources)
        else:
            weights, _ = _weights_from_sigma(per_source_sigma(retained), scheme)
        composite_retained = retained @ weights
        residuals.append(float(values[i] @ weights - composite_retained.mean()))
    inflation = 1.0 + 1.0 / (n_seeds - 1)
    return math.sqrt(float(np.mean(np.square(residuals))) / inflation)


def inverse_variance_weights(
    values: np.ndarray,
    sources: Sequence[str],
    scheme: str = "strata",
) -> SourceWeights:
    """
    Weights over the held-out sources, from the per-source variances and nothing else.

    WHICH SCHEME, AND WHY IT IS THE STRATIFIED ONE. Both are offered and the default is
    ``strata``. Plain inverse-variance sets ``w_j`` proportional to ``1 / s_j^2`` where each
    ``s_j`` carries ``n - 1 = 4`` df, and a weight that is itself that noisy stops behaving
    like a weight: the estimator becomes a random-weight average whose variance exceeds the
    fixed-weight optimum it is aiming at, and at df = 4 the reciprocal of a variance is a
    heavy-tailed object -- ``E[1/s^2]`` is twice ``1/sigma^2``. Splitting the seven sources
    into a low-variance and a high-variance group at the largest gap in ``log s`` and pooling
    within each takes the df behind every weight from 4 to the group's total, which is where
    the estimated-weight penalty stops mattering. It also matches what the literature actually
    claims: DataDecide reports a *separation between source types*, code against web text,
    not seven individually resolved variances.

    The split point minimizes the within-group sum of squares of ``log s`` over the candidate
    cuts of the sorted sources, and **no stratum is allowed fewer than two members**. Both
    halves of that matter. The largest-gap rule the obvious implementation reaches for ignores
    how many sources sit either side of it, and the minimum-scatter rule on its own will still
    isolate a single far outlier -- putting one source in a stratum by itself, back on the
    df = 4 the strata exist to escape, and doing it precisely for the source whose variance is
    least well determined. Requiring two takes the worst-case df behind any weight from 4 to 8.
    The cost is that a genuinely singular source shares a stratum with its nearest neighbour and
    is weighted a little too heavily; that is a bounded error in a weight, against an unbounded
    one in the estimate of the weight.

    ``inverse-variance`` is kept because it is the thing a reader expects to see, and because
    when the two agree the choice was not load-bearing. When they disagree they disagree in one
    direction: the in-sample reduction under ``inverse-variance`` is spectacular and its
    leave-one-seed-out reduction is not, which is what an over-fitted weight vector looks like
    from the outside. Whichever is used, it is the frozen vector that goes into stage 2 and the
    choice is recorded in :attr:`SourceWeights.scheme`.

    NOT GLS, AND NOT BY OVERSIGHT. The efficient vector is ``Sigma^-1 1 / (1' Sigma^-1 1)``,
    and a 7x7 covariance from five seeds has rank 4. There is no inverse. What the covariance
    is used for is :attr:`SourceWeights.variance_reduction`, which reports what this vector
    achieves against the shared seed effect rather than pretending the sources are
    independent.

    :param values: Shape ``(n_seeds, n_sources)``, per-source endpoint at one step.
    :param sources: Source names, in the column order of ``values``.
    :param scheme: ``"strata"`` or ``"inverse-variance"``.

    :returns: The weights and what they buy.

    :raises ValueError: If the shape and the source names disagree, if fewer than two seeds
        are supplied, or if a per-source sigma is zero and cannot be inverted.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(sources):
        raise ValueError(f"shape {values.shape} against {len(sources)} sources")
    n_seeds, n_sources = values.shape

    sigma = per_source_sigma(values)
    weights, strata = _weights_from_sigma(sigma, scheme)
    uniform = np.full(n_sources, 1.0 / n_sources)

    df_per_weight = []
    for j in range(n_sources):
        members = int((strata == strata[j]).sum()) if scheme == "strata" else 1
        df_per_weight.append(members * (n_seeds - 1))

    unweighted = realised_sigma(values, uniform)
    weighted = realised_sigma(values, weights)
    diagonal_unweighted = math.sqrt(float(np.sum(uniform**2 * sigma**2)))
    diagonal_weighted = math.sqrt(float(np.sum(weights**2 * sigma**2)))

    cv_unweighted = _cross_validated_sigma(values, None)
    cv_weighted = _cross_validated_sigma(values, scheme)

    return SourceWeights(
        sources=tuple(sources),
        sigma=tuple(float(s) for s in sigma),
        weights=tuple(float(w) for w in weights),
        scheme=scheme,
        strata=tuple(int(s) for s in strata),
        df_per_weight=tuple(df_per_weight),
        n_seeds=n_seeds,
        unweighted_sigma=unweighted,
        weighted_sigma=weighted,
        variance_reduction=(unweighted / weighted) ** 2 if weighted > 0 else float("nan"),
        cross_validated_variance_reduction=(
            (cv_unweighted / cv_weighted) ** 2 if cv_weighted > 0 else float("nan")
        ),
        diagonal_variance_reduction=(
            (diagonal_unweighted / diagonal_weighted) ** 2
            if diagonal_weighted > 0
            else float("nan")
        ),
    )


# ---------------------------------------------------------------------------------------
# (c) rho-hat, the paired-analysis correlation. Written now, used in stage 2.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedCorrelation:
    """The within-seed correlation between two arms, by two routes that should agree."""

    n_pairs: int
    rho_pearson: float
    """Ordinary correlation across the shared seeds. df = n - 2 for its own test."""

    rho_variance_components: float
    """
    ``1 - s_delta^2 / (2 s_pooled^2)``. The same quantity read out of the paired difference
    rather than out of the cross-product. Reported beside the Pearson estimate because they
    are algebraically equal only when the two arms have exactly equal sample variances; a
    visible gap between them is a visible failure of the homoscedasticity the pooled sigma
    rests on. It is a ratio of two estimates and Jensen pushes it low, more so than the
    Pearson figure, so neither of these is the number the analysis runs on.
    """

    ci_low: float
    ci_high: float
    """
    Fisher-z interval on the Pearson estimate, at ``1 / sqrt(n - 3)``. At five pairs that is
    ``1 / sqrt(2)`` and the interval covers most of the line. It is reported so that nobody
    reads a point estimate of rho as a measurement.
    """

    sigma_delta: float
    """
    Standard deviation of the paired differences, on ``n - 1`` df, and THE QUANTITY THE
    PAIRED ANALYSIS ACTUALLY RUNS ON. It is estimated directly, ``s_delta^2`` is unbiased for
    ``2 sigma^2 (1 - rho)`` exactly, and the test of the paired mean needs nothing else. Both
    ``rho`` fields above are functions of it and of the pooled sigma, and both inherit a
    small-sample downward bias from being ratios -- the self-test measures 0.65 against a
    planted 0.70 at five pairs. So rho-hat is *reported*, because it is what makes the
    variance reduction legible and what the pre-registration's table is indexed by, and
    ``sigma_delta`` is what is *used*. Pre-committing to the ratio instead would import a
    bias the design does not have to carry.
    """

    sigma_pooled: float
    """Pooled within-arm sigma over both arms, on ``2(n - 1)`` df."""

    paired_se: float
    unpaired_se: float
    """
    The standard error of the contrast each way. Their ratio is what pairing buys before the
    df it costs is accounted for, which :func:`mde` does account for.
    """


def paired_correlation(
    arm_a: Sequence[float],
    arm_b: Sequence[float],
    confidence: float = 0.95,
) -> PairedCorrelation:
    """
    Estimate the within-seed correlation the paired analysis exploits.

    NOT COMPUTABLE FROM STAGE 1, WHICH IS WHY IT IS WRITTEN BEFORE STAGE 1 FINISHES. Pairing
    is across arms at a shared seed, so rho needs two arms and the baseline is one. Committing
    the estimator now means the number that decides whether the primary analysis is paired or
    unpaired comes out of a function written before anybody could see which answer they
    preferred. It is tested against synthetic data with a known rho in
    ``test_noise_floor.py``.

    WHAT THE PAIRING REMOVES, WHICH IS LESS THAN IT LOOKS. ``build_config`` adds ``--seed`` to
    the data seed identically on every arm, so arm ``a`` seed ``k`` and arm ``b`` seed ``k``
    stream the same documents in the same order. It adds the same offset to the two init
    seeds, but two arms with different parameter shapes draw from that generator in different
    orders, so initialization does *not* match across arms. What pairing removes is the data
    order and nothing else, and that is exactly why rho has to be measured rather than
    assumed.

    :param arm_a: One arm's endpoint, one value per seed.
    :param arm_b: The other arm's endpoint, in the same seed order.
    :param confidence: Coverage for the Fisher-z interval.

    :returns: The estimate.

    :raises ValueError: If the two arms have different lengths or fewer than three shared
        seeds, which is the point below which the correlation has no interval at all.
    """
    a = np.asarray(list(arm_a), dtype=float)
    b = np.asarray(list(arm_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"{a.size} seeds against {b.size}; pairing needs the same seeds")
    if a.size < 3:
        raise ValueError("a correlation over fewer than three pairs is not an estimate")

    n = int(a.size)
    rho_pearson = float(np.corrcoef(a, b)[0, 1])
    pooled = pooled_sigma([a, b])
    delta = a - b
    sigma_delta = float(delta.std(ddof=1))
    rho_vc = 1.0 - sigma_delta**2 / (2.0 * pooled.sigma**2)

    if n > 3 and abs(rho_pearson) < 1.0:
        z = math.atanh(rho_pearson)
        half = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0) / math.sqrt(n - 3)
        ci_low, ci_high = math.tanh(z - half), math.tanh(z + half)
    else:
        ci_low, ci_high = -1.0, 1.0

    return PairedCorrelation(
        n_pairs=n,
        rho_pearson=rho_pearson,
        rho_variance_components=float(rho_vc),
        ci_low=ci_low,
        ci_high=ci_high,
        sigma_delta=sigma_delta,
        sigma_pooled=pooled.sigma,
        paired_se=sigma_delta / math.sqrt(n),
        unpaired_se=pooled.sigma * math.sqrt(2.0 / n),
    )


# ---------------------------------------------------------------------------------------
# The df convention, and the minimum detectable effects that rest on it.
# ---------------------------------------------------------------------------------------


def error_df(n_arms: int, n_seeds: int, paired: bool) -> int:
    """
    The error degrees of freedom of the design, which is the thing the plan had wrong.

    A PAIRED ANALYSIS OF ``k`` ARMS ACROSS ``n`` SHARED SEEDS IS A RANDOMIZED COMPLETE BLOCK
    DESIGN, AND THE BLOCK EATS ``n - 1`` OF THE DF. The total is ``kn - 1``; the arm main
    effect takes ``k - 1``, the seed main effect takes ``n - 1``, and what is left for error
    is ``(k - 1)(n - 1)``. The pre-registration's paired MDE table used ``k(n - 1)`` -- the
    *unpaired* count, which is right for a design with no block term and wrong the moment a
    seed effect is removed. At three arms and three seeds that is 6 against a true 4, and
    every paired MDE in that table was about 11% optimistic as a result. Removing the seed
    effect is not free, and the free-looking version of it was the arithmetic error.

    The unpaired count is ``N - k = k(n - 1)``, which is the pooled within-arm df and is what
    :func:`pooled_sigma` returns for ``k`` arms of ``n`` seeds.

    :param n_arms: Arms sharing the pooled variance estimate.
    :param n_seeds: Seeds per arm. Shared across arms in the paired case.
    :param paired: Whether the seed effect is removed as a block.

    :returns: Error degrees of freedom.

    :raises ValueError: If either count is below two, where the design has no error df.
    """
    if n_arms < 2 or n_seeds < 2:
        raise ValueError(f"{n_arms} arms x {n_seeds} seeds has no error df to speak of")
    return (n_arms - 1) * (n_seeds - 1) if paired else n_arms * (n_seeds - 1)


def contrast_se(sigma: float, n_seeds: int, rho: float = 0.0, paired: bool = True) -> float:
    """
    The standard error of a difference between two arm means.

    Unpaired it is ``sigma sqrt(2/n)``. Paired it is ``sigma sqrt(2(1 - rho)/n)``, which is
    the same expression with the within-seed component of the variance removed -- the paired
    difference has standard deviation ``sigma sqrt(2(1 - rho))`` per pair, and there are ``n``
    of them.

    :param sigma: Per-run standard deviation of the endpoint.
    :param n_seeds: Seeds per arm.
    :param rho: Within-seed correlation across arms. Ignored when ``paired`` is False.
    :param paired: Whether to credit the pairing.

    :returns: The standard error.
    """
    if not paired:
        return sigma * math.sqrt(2.0 / n_seeds)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    return sigma * math.sqrt(2.0 * (1.0 - rho) / n_seeds)


def _noncentral_t_cdf(t: float, df: int, ncp: float) -> float:
    """
    The noncentral t distribution function, computed so that it never returns a NaN.

    ``scipy.stats.nct`` IS NOT USABLE HERE AND THE FAILURE IS NOT AT THE EXTREMES. At the small
    df this design has it returns NaN at scattered ordinary arguments -- df = 2 and a
    noncentrality of 7.750 gives NaN while 7.756 is fine -- so a power curve read off it has
    holes in it at no particular place. A root find walking through one of those holes does not
    raise; it converges on the hole. That is how a minimum detectable effect comes back 0.6%
    wrong with every appearance of having converged, and it is exactly the class of silent
    error this module exists to remove from the analysis.

    So the integral is taken directly. Conditioning on the chi variable in the denominator,

        P(T <= t) = E_U[ Phi(t sqrt(U/df) - ncp) ],  U ~ chi-square(df)

    which is an integral of a bounded smooth function against a density. The scipy value is
    still used wherever it is finite, and not only because it is faster:
    **where scipy answers at all it is the more accurate of the two.** Checked against an
    8,192-node rule, scipy sits within 3e-9 and the 512-node rule below within 2.3e-7, so the
    sub-1e-6 disagreement between them is this function's own truncation error and not
    scipy's. That is worth stating because the obvious reading of a fallback is that it is the
    reference, and here it is the approximation -- which is fine, since 2.3e-7 of power is
    about 1e-8 nats on a minimum detectable effect, four orders below the last digit anything
    here prints, but it is not fine to have backwards.
    ``test_the_quadrature_agrees_with_scipy_wherever_scipy_is_finite`` holds the agreement,
    and it is what makes the fallback a repair of scipy rather than a second opinion about it.

    :param t: The quantile.
    :param df: Degrees of freedom.
    :param ncp: Noncentrality.

    :returns: The distribution function at ``t``.
    """
    value = float(stats.nct.cdf(t, df, ncp))
    if math.isfinite(value):
        return value

    # Integrated against the chi-square *probability* rather than against its density: with
    # p = F(u) the integral runs over [0, 1] and the integrand is a bounded monotone function
    # of p, where integrating the density directly is over a half-line and quad calls it
    # divergent at df = 1.
    nodes, weights = _quadrature(df)
    integrand = stats.norm.cdf(t * np.sqrt(nodes / df) - ncp)
    return float(min(max(float(weights @ integrand), 0.0), 1.0))


@lru_cache(maxsize=32)
def _quadrature(df: int, order: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes on the chi-square quantile scale, cached per df.

    Fixed nodes rather than adaptive quadrature so that the power function is a deterministic
    smooth function of its arguments. An adaptive rule changes its subdivision as the arguments
    move, which puts steps of the order of its own tolerance into a curve that a root find is
    walking along -- a smaller version of the defect this whole function exists to route around.

    :param df: Degrees of freedom.
    :param order: Nodes. 512 holds agreement with scipy to under 1e-6 across df 1 to 20 and
        noncentralities out to 12, and costs about 60 microseconds after the first call.
        Raising it is a bad trade and was measured rather than assumed: 2,048 nodes cut the
        truncation error from 2.3e-7 to 1.4e-8 but cost 3.5 seconds to build per distinct df
        against 0.45 for 512, because building the rule is quadratic in the node count while
        the error it buys down is already four orders below anything printed.

    :returns: ``(chi-square quantiles at the nodes, weights)``.
    """
    x, w = np.polynomial.legendre.leggauss(order)
    probabilities = 0.5 * (x + 1.0)
    return stats.chi2.ppf(probabilities, df), 0.5 * w


def power_of(delta: float, se: float, df: int, alpha: float = 0.05) -> float:
    """
    Two-sided power of a t test at a true effect, by exact noncentral t.

    Not the normal approximation. With ``sigma`` estimated the statistic is a t and the
    alternative is a *noncentral* t, and the gap between the two matters at the df this design
    has: at df = 8 the normal approximation understates the detectable effect by around 7%,
    which is the same order as the df error this function exists to price.

    :param delta: The true effect.
    :param se: Standard error of the contrast.
    :param df: Error degrees of freedom of the variance estimate behind ``se``.
    :param alpha: Two-sided significance level.

    :returns: Power, in ``[0, 1]``.
    """
    critical = stats.t.ppf(1.0 - alpha / 2.0, df)
    ncp = delta / se
    return 1.0 - _noncentral_t_cdf(critical, df, ncp) + _noncentral_t_cdf(-critical, df, ncp)


def mde(
    sigma: float,
    n_seeds: int,
    n_arms: int = 3,
    rho: float = 0.0,
    paired: bool = True,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    The smallest effect this design detects, by exact noncentral t.

    :param sigma: Per-run standard deviation of the endpoint, in the endpoint's own units.
    :param n_seeds: Seeds per arm.
    :param n_arms: Arms sharing the variance estimate, which sets the error df.
    :param rho: Within-seed correlation, credited only when ``paired``.
    :param paired: Whether the analysis blocks on seed.
    :param alpha: Two-sided significance level.
    :param power: Target power.

    :returns: The minimum detectable effect, in the units of ``sigma``.
    """
    se = contrast_se(sigma, n_seeds, rho, paired)
    df = error_df(n_arms, n_seeds, paired)

    # Bracket from the normal approximation rather than from a round number. The noncentral t
    # answer is above it and within a factor of two of it at every df this design reaches, and
    # a bracket set by orders of magnitude walks the solver into the region where scipy's nct
    # underflows.
    approximate = se * (stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    upper = max(approximate * 4.0, se)
    while power_of(upper, se, df, alpha) < power:
        upper *= 2.0

    return float(
        optimize.brentq(
            lambda delta: power_of(delta, se, df, alpha) - power,
            1e-12,
            upper,
            xtol=1e-14,
        )
    )


def mde_from(
    estimate: SigmaEstimate,
    n_seeds: int,
    n_arms: int = 3,
    rho: float = 0.0,
    paired: bool = True,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    The MDE at a *measured* sigma, with the standard deviation's own bias taken out.

    The wrapper exists so that the correction is applied wherever a measurement feeds an MDE
    and nowhere else. Passing the raw ``estimate.sigma`` to :func:`mde` would report a design
    6% more sensitive than it is at df = 4, for the reason in
    :attr:`SigmaEstimate.sigma_unbiased`.

    :param estimate: A measured sigma with its df.

    :returns: The minimum detectable effect, in the units of the estimate.
    """
    return mde(estimate.sigma_unbiased, n_seeds, n_arms, rho, paired, alpha, power)


@dataclass(frozen=True)
class MDERow:
    """One row of the paired MDE table: a correlation, and what it buys at each seed count."""

    rho: float
    by_seeds: Dict[int, float] = field(default_factory=dict)


def mde_table(
    sigma: float = PLANNING_SIGMA_NATS,
    rhos: Sequence[float] = (0.0, 0.3, 0.5, 0.7),
    seed_counts: Sequence[int] = (3, 4, 5),
    n_arms: int = 3,
    alpha: float = 0.05,
    power: float = 0.80,
) -> List[MDERow]:
    """
    The paired MDE table, recomputed with the randomized-block df.

    :param sigma: Planning or measured per-run sigma.
    :param rhos: Within-seed correlations to tabulate.
    :param seed_counts: Seeds per arm, one column each.
    :param n_arms: Arms in the design, which sets ``(k-1)(n-1)``.
    :param alpha: Two-sided significance level.
    :param power: Target power.

    :returns: One row per correlation.
    """
    return [
        MDERow(
            rho=rho,
            by_seeds={n: mde(sigma, n, n_arms, rho, True, alpha, power) for n in seed_counts},
        )
        for rho in rhos
    ]


def render_mde_table(
    sigma: float = PLANNING_SIGMA_NATS,
    rhos: Sequence[float] = (0.0, 0.3, 0.5, 0.7),
    seed_counts: Sequence[int] = (3, 4, 5),
    n_arms: int = 3,
) -> str:
    """
    The same table as the markdown the pre-registration carries, so the two cannot drift.

    :returns: A markdown table, with the error df in each column header.
    """
    header = ["| rho |"]
    rule = ["| --- |"]
    for n in seed_counts:
        header.append(f" {n} pairs (df {error_df(n_arms, n, True)}) |")
        rule.append(" --- |")
    lines = ["".join(header), "".join(rule)]
    for row in mde_table(sigma, rhos, seed_counts, n_arms):
        cells = "".join(f" {row.by_seeds[n]:.3f} |" for n in seed_counts)
        lines.append(f"| {row.rho:.1f} |{cells}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Reading the runs.
# ---------------------------------------------------------------------------------------


#: How a fan-out cell's W&B run id ends, matched exactly as ``tranche_watch`` matches it.
CELL_SUFFIX = re.compile(r"^(?P<submission>.+)-cell-(?P<index>\d+)$")

#: What ``train_on_corpus.leave_the_reason_in_wandb`` appends to a cell's id when it files a
#: crash report beside it. Spelled out here as well as there, for the reason ``stage_gate``
#: keeps its own copy of the source list: the image that runs the entry point installs neither
#: scipy nor this file, so nothing in ``.edullm/`` imports across that line for a constant.
CRASH_REPORT_SUFFIX = "-died"

#: ``job_type`` on a crash report, which is the same fact stated where a reader holding only
#: the run object can see it.
CRASH_REPORT_JOB_TYPE = "crash"


def is_crash_report(run_id: str, job_type: Optional[str] = None, history_step: int = -1) -> bool:
    """
    Whether a run is a crash report rather than a cell that trained.

    A crash report shares the group and the display-name stem of the cell it is about, so a
    query that selects a submission selects it too -- and a report carries no model config,
    which makes :func:`_arm_of` read it as ``baseline``. Left in, the diagnostic filed against
    a dead ``mhc`` cell arrives in the baseline arm as a seed-0 replicate with nothing in it.

    ``history_step`` IS NOT DEFENSIVENESS AND LEAVING IT OUT DROPS TWO REPLICATES. The seven
    clobbered cells carry ``job_type: crash`` themselves, because the report was written *onto*
    them rather than beside them -- that is the defect. Filtering on ``job_type`` alone
    therefore deletes exactly the runs this recovery exists to rescue, and it deletes them
    silently, which is the same failure again one layer up. A run that logged a step trained,
    whatever its metadata was overwritten to say, so the history settles it.

    :param run_id: The W&B run id.
    :param job_type: The run's ``job_type``, where the caller has it.
    :param history_step: The last step the run's history holds, from
        :func:`steps_from_summary_and_history`. Anything at zero or above is a run that
        trained and is never a report.

    :returns: Whether to leave it out of an arm.
    """
    if history_step >= 0:
        return False
    return run_id.endswith(CRASH_REPORT_SUFFIX) or job_type == CRASH_REPORT_JOB_TYPE


def steps_from_summary_and_history(run) -> Tuple[Optional[int], int]:
    """
    What a run's summary says its last step was, and what its history actually holds.

    THESE DISAGREE ON SEVEN RUNS AND THE DISAGREEMENT IS THE WHOLE POINT. A crash report that
    re-initialised W&B under the cell's own id replaced the summary and left the history
    alone, so those cells read ``_step: None, _runtime: 0.0`` -- which is also what a cell
    that never started reads. One of them had trained 3.993 hours to step 4,910.

    ``lastHistoryStep`` is the recovery and it is a field rather than a scan: W&B tracks it
    beside the history and the public API hands it over with the run. On an intact run it
    equals the summary's ``_step``, which is what makes it safe to prefer.

    :param run: A W&B API run.

    :returns: ``(the summary's step or None, the history's last step or -1)``.
    """
    raw_summary = run.summary.get("_step")
    summary_step = int(raw_summary) if isinstance(raw_summary, (int, float)) else None
    raw_history = getattr(run, "lastHistoryStep", None)
    history_step = int(raw_history) if isinstance(raw_history, (int, float)) else -1
    return summary_step, history_step


@dataclass
class SeedSeries:
    """One run of one arm, reduced to what the estimators need."""

    run_name: str
    seed: int
    arm: str
    state: str
    last_step: int
    """The furthest step this run reached, from its history where its summary has lost it."""

    per_source: Dict[int, Dict[str, float]] = field(default_factory=dict)
    """Evaluation step -> source -> bits-per-byte."""

    run_id: str = ""
    """
    The W&B run id, which is the only thing that identifies a cell. Recorded so the frozen
    artifact names the exact five runs it was computed from rather than a group that will
    keep acquiring members after the freeze.
    """

    summary_step: Optional[int] = None
    """
    What the run's own summary claims, or None where it carries nothing. Kept apart from
    :attr:`last_step` so that "the summary was overwritten" stays a fact the report can state
    rather than a difference that quietly disappears into a recovered number.
    """

    history_step: int = -1
    """The last step the per-step history holds, which a crash report cannot reach."""

    @property
    def summary_was_clobbered(self) -> bool:
        """Whether this run's summary lost a record its history still has."""
        return self.summary_step is None and self.history_step >= 0


def belongs_to_submission(run_id: str, submission: Optional[str]) -> bool:
    """
    Whether one run is a cell of the named submission.

    MATCHED ON THE RUN ID AND NEVER ON THE DISPLAY NAME, for the reason ``tranche_watch``
    gives: the cells of a fan-out share a display name and differ only in the id's
    ``-cell-<index>`` suffix, and the name is not even stable, since the cancelled L40S cells
    were all renamed to ``...-died`` after the fact. The prefix is compared against the id
    with that suffix removed, so naming a submission selects all of its cells and none of
    anything else's.

    :param run_id: The W&B run id.
    :param submission: A platform run id, or a unique prefix of one. None or empty accepts
        every run, which is the behaviour from before this argument existed.

    :returns: Whether the run belongs.
    """
    if not submission:
        return True
    match = CELL_SUFFIX.match(run_id)
    stem = match.group("submission") if match else run_id
    return stem.startswith(submission)


def contributing(series: Sequence["SeedSeries"]) -> List["SeedSeries"]:
    """
    The runs that carry an evaluation, which are the only ones any estimator can use.

    A run that logged none contributes an empty step set and would take the intersection in
    :func:`aligned_matrix` to nothing, which is what a group holding two dead runs from an
    earlier submission looks like. Defined once and used both there and by the caller that
    records which runs the frozen numbers came from, so the artifact cannot name a different
    set from the one that was measured.

    :param series: One entry per run.

    :returns: The subset with at least one evaluation, in the order given.
    """
    return [entry for entry in series if entry.per_source]


def excluded(series: Sequence["SeedSeries"]) -> List[Tuple["SeedSeries", str]]:
    """
    The runs :func:`contributing` drops, each with the reason it was dropped.

    DROPPING A REPLICATE QUIETLY IS THE FAILURE THIS EXISTS TO END, AND IT IS WORSE THAN A
    WRONG NUMBER BECAUSE IT LOOKS LIKE A RIGHT ONE. ``contributing`` is a filter, and a filter
    with no complement reports four cells where five were submitted as though four were what
    was asked for: n falls, the df falls with it, the interval narrows, and the report says
    only that a cell "has not landed". It is also not a random four. Cells lose their
    evaluations by hitting a wall or dying, so the ones that leave are the slow ones and the
    unlucky ones, and an arm mean over the survivors is biased in a direction nobody chose.

    So every run that goes is named here with a reason, and :func:`provisional_reasons` turns
    that into something ``--freeze`` refuses to write.

    :param series: One entry per run, as :func:`read_seed_series` returned them.

    :returns: ``(run, reason)`` for each run carrying no evaluation, in the order given.
    """
    out: List[Tuple[SeedSeries, str]] = []
    for entry in series:
        if entry.per_source:
            continue
        if entry.history_step >= 0:
            out.append(
                (
                    entry,
                    f"reached step {entry.history_step} but logged no held-out evaluation this "
                    "read can find, so it has an endpoint nobody can compare",
                )
            )
        else:
            out.append((entry, "logged no history at all -- queued, or died before its first step"))
    return out


def sources_from_config(config: Mapping) -> Tuple[str, ...]:
    """
    The held-out sources a run actually evaluated on, from its own saved config.

    Derived with ``hyper_connection_arms.source_label`` over the evaluation shard paths rather
    than from the metadata labels beside them, because that function is what put the labels
    there in the first place and reading them back through it is what would catch the two
    disagreeing. Falls back to the labels, and then to :data:`HELD_OUT_SOURCES`, so a run with
    an unexpected config shape degrades to a warning rather than an exception.

    :param config: A W&B run config.

    :returns: Source names, sorted.
    """
    from hyper_connection_arms import source_label

    try:
        held_out = config["trainer"]["callbacks"]["held_out"]["eval_dataset"]
    except (KeyError, TypeError):
        return HELD_OUT_SOURCES

    paths = held_out.get("paths") or []
    labelled = sorted({source_label(p) for p in paths}) if paths else []
    if labelled:
        return tuple(labelled)

    metadata = held_out.get("metadata") or []
    from_metadata = sorted({m.get("label", "") for m in metadata if m.get("label")})
    return tuple(from_metadata) if from_metadata else HELD_OUT_SOURCES


def read_seed_series(
    entity: str,
    project: str,
    group: str,
    arm: str = "baseline",
    metric: str = PRIMARY_METRIC,
    submission: Optional[str] = None,
) -> Tuple[List[SeedSeries], Tuple[str, ...]]:
    """
    Pull every seed of one arm out of W&B, with its per-source endpoint at every evaluation.

    The seed is read off ``data_loader.seed``, which ``build_config`` sets to
    ``--data-seed + --seed`` with the flag defaulting to zero, so it is the replicate index
    itself. ``init_seed`` moves with it and is reported alongside as a cross-check that the
    fan-out really did resolve five different cells rather than running one five times -- the
    failure ``resolve_seed`` exists to refuse, and the one that would report a noise floor of
    exactly zero.

    A GROUP IS NOT AN EXPERIMENTAL UNIT, WHICH IS WHY ``submission`` EXISTS. Every attempt at
    this arm shares one experiment slug, so ``hyper-connections-370m`` holds the five live A100
    cells, the three cancelled L40S cells, two submissions that died before logging a step, and
    the probes. Reading the group whole returns eight baseline entries carrying seeds
    ``0, 0, 1, 1, 2, 3, 3, 4``, which trips the distinct-seed refusal in :func:`main` -- and
    that refusal is right to fire, because the duplicates are real. What it is diagnosing is the
    query and not the fan-out. Naming the submission is how the pre-registered choice of *which*
    baseline is the comparator gets executed rather than left to whatever else shares the slug;
    it is not a filter chosen after seeing a number, and the id it selects is recorded in the
    frozen artifact.

    :param entity: W&B entity.
    :param project: W&B project.
    :param group: The experiment slug the runs are grouped under.
    :param arm: Which arm to collect. Matched against the run's own config.
    :param metric: A format string with a ``{source}`` field.
    :param submission: A platform run id, or a unique prefix of one, restricting the read to
        the cells of that one submission. None reads the whole group.

    :returns: ``(one series per run, the source names)``.
    """
    import wandb

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=100)[:100])

    series: List[SeedSeries] = []
    sources: Tuple[str, ...] = HELD_OUT_SOURCES
    for run in runs:
        if not belongs_to_submission(run.id, submission):
            continue
        summary_step, history_step = steps_from_summary_and_history(run)
        # BEFORE THE ARM TEST, BECAUSE A CRASH REPORT WOULD PASS IT. It carries no model
        # config, and a config without a hyper-connection block is how `_arm_of` spells
        # `baseline` -- so a report filed against a dead `mhc` cell would join the baseline
        # arm as a seed-0 replicate with nothing in it.
        if is_crash_report(run.id, run.job_type, history_step):
            continue
        config = run.config or {}
        if _arm_of(config) != arm:
            continue
        sources = sources_from_config(config)
        keys = [metric.format(source=s) for s in sources]
        entry = SeedSeries(
            run_name=run.name,
            seed=int(((config.get("data_loader") or {}).get("seed")) or 0),
            arm=arm,
            state=run.state,
            # The history's answer wherever the summary has none. A summary that was
            # overwritten says None here and a run that never stepped says None too, and the
            # difference between those two is the whole of `summary_was_clobbered`.
            last_step=summary_step if summary_step is not None else history_step,
            run_id=run.id,
            summary_step=summary_step,
            history_step=history_step,
        )
        for row in run.scan_history(keys=["_step", *keys]):
            step = row.get("_step")
            values = {s: row.get(metric.format(source=s)) for s in sources}
            if step is None or any(v is None for v in values.values()):
                continue
            entry.per_source[int(step)] = {s: float(v) for s, v in values.items()}
        series.append(entry)

    series.sort(key=lambda s: s.seed)
    return series, sources


def _arm_of(config: Mapping) -> str:
    """
    Which arm a run's config describes, from the model rather than from a label.

    No ``--arm`` string is saved anywhere in the config, so it is recovered from the
    hyper-connection block: absent is ``baseline``, ``mode`` of ``output`` is ``output-only``,
    and anything else with lanes is read as ``faithful``. Coarse on purpose -- this only has to
    separate the three funded arms, and separating them on the model is what makes it
    impossible for a mislabelled run to be counted into the wrong noise floor.
    """
    try:
        hc = config["model"]["block"].get("hyper_connections")
    except (KeyError, TypeError, AttributeError):
        return "baseline"
    if not hc:
        return "baseline"
    return "output-only" if str(hc.get("mode")) == "output" else "faithful"


def aligned_matrix(
    series: Sequence[SeedSeries],
    sources: Sequence[str],
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
    """
    Stack the runs into ``(n_seeds, n_steps, n_sources)`` over the steps they all reached.

    Only steps every run evaluated at are kept, because a sigma over a step that three of five
    seeds reached is a sigma over three seeds wearing the label of five. Runs that logged no
    evaluation at all are dropped first rather than emptying the intersection.

    :param series: One entry per run.
    :param sources: Column order.

    :returns: ``(values, steps, seeds)``.
    """
    usable = contributing(series)
    if not usable:
        return np.zeros((0, 0, len(sources))), (), ()
    common = set(usable[0].per_source)
    for entry in usable[1:]:
        common &= set(entry.per_source)
    steps = tuple(sorted(common))
    if not steps:
        return np.zeros((0, 0, len(sources))), (), ()
    values = np.asarray(
        [[[entry.per_source[st][s] for s in sources] for st in steps] for entry in usable],
        dtype=float,
    )
    return values, steps, tuple(entry.seed for entry in usable)


# ---------------------------------------------------------------------------------------
# Synthetic data, so every estimator has a known truth to be tested against.
# ---------------------------------------------------------------------------------------


def synthetic_baseline(
    n_seeds: int = 5,
    sources: Sequence[str] = HELD_OUT_SOURCES,
    sigma_by_source: Optional[Mapping[str, float]] = None,
    shared_fraction: float = 0.5,
    steps: Sequence[int] = tuple(range(500, 6001, 500)),
    settling: float = 0.0,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[int, ...]]:
    """
    A baseline arm with a known noise structure, for ``--dry-run`` and for the tests.

    Built as a shared per-seed effect plus independent per-source noise, because that is the
    structure the real thing has: a seed that draws a lucky shuffle is lucky on every source
    at once, which is what puts a floor under how much any diagonal weighting can buy.
    Per-source sigma spans the order of magnitude DataDecide reports, with the code-type
    sources at the noisy end.

    :param n_seeds: Replicates.
    :param sources: Column names.
    :param sigma_by_source: Total per-source sigma. Defaults to a DataDecide-shaped spread.
    :param shared_fraction: How much of the *quietest* source's variance the common seed
        effect carries. The loading is the same absolute size on every source rather than
        proportional to that source's own sigma, which is the pessimistic reading and the
        right one to build a tool against: a seed effect that scales with each source's noise
        can be weighted away along with it, and one that does not is a floor no diagonal
        weighting reaches. Zero makes the sources independent, which is the only case where a
        diagonal weighting is exactly the right answer.
    :param steps: Evaluation checkpoints.
    :param settling: How much larger sigma is at the first checkpoint than at the last, as a
        multiplier on the whole trajectory's log range. Zero is a settled noise floor.
    :param rng_seed: Reproducibility.

    :returns: ``(values of shape (n_seeds, n_steps, n_sources), sources, steps)``.
    """
    default = {
        "dclm": 0.0035,
        "wiki": 0.0042,
        "pes2o": 0.0048,
        "arxiv": 0.0060,
        "open-web-math": 0.0180,
        "algebraic-stack": 0.0240,
        "starcoder": 0.0300,
    }
    sigma_by_source = dict(default if sigma_by_source is None else sigma_by_source)
    sigma = np.asarray([sigma_by_source[s] for s in sources], dtype=float)

    rng = np.random.default_rng(rng_seed)
    steps = tuple(steps)
    scale = np.asarray(
        [1.0 + settling * (1.0 - i / max(len(steps) - 1, 1)) for i in range(len(steps))]
    )

    shared = rng.standard_normal((n_seeds, len(steps)))
    private = rng.standard_normal((n_seeds, len(steps), len(sources)))
    level = np.linspace(1.20, 0.85, len(steps))[None, :, None]

    loading = math.sqrt(shared_fraction) * float(sigma.min())
    own_sigma = np.sqrt(np.maximum(sigma**2 - loading**2, 0.0))
    common = loading * shared[:, :, None]
    own = private * own_sigma[None, None, :]
    values = level + (common + own) * scale[None, :, None]
    return values, tuple(sources), steps


def synthetic_pair(
    n_seeds: int = 5,
    sigma: float = 0.010,
    rho: float = 0.5,
    effect: float = 0.0,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two arms sharing seeds at a known within-seed correlation, for testing :func:`paired_correlation`.

    :param n_seeds: Shared seeds.
    :param sigma: Per-run standard deviation, the same in both arms.
    :param rho: The truth the estimator has to recover.
    :param effect: A constant added to the second arm.
    :param rng_seed: Reproducibility.

    :returns: ``(arm_a, arm_b)``, aligned by seed.
    """
    rng = np.random.default_rng(rng_seed)
    shared = rng.standard_normal(n_seeds)
    a = sigma * (math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * rng.standard_normal(n_seeds))
    b = sigma * (math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * rng.standard_normal(n_seeds))
    return a, b + effect


# ---------------------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------------------


def _format_sigma(estimate: SigmaEstimate, unit: str) -> str:
    return (
        f"{estimate.sigma:.5f} {unit}  "
        f"[{estimate.ci_low:.5f}, {estimate.ci_high:.5f}] at {estimate.confidence:.0%}, "
        f"df = {estimate.df}, span x{estimate.span:.1f}"
    )


def provisional_reasons(
    n_seeds: int,
    final_step: int,
    horizon: int,
    expected_seeds: int = 5,
    exclusions: Sequence[str] = (),
) -> List[str]:
    """
    Everything that makes a reading provisional rather than the frozen noise floor.

    THE ONE FAILURE THIS TOOL COULD CAUSE ON ITS OWN IS BEING BELIEVED TOO EARLY. Ten minutes
    after the fan-out is admitted every cell has logged exactly one evaluation -- the
    ``eval_on_startup`` pass over an untrained model -- and a sigma over those is a real
    number, computed correctly, that measures initialization scatter and has nothing to do
    with the endpoint. It would be an entirely plausible thing to paste into the
    pre-registration. So the conditions under which a reading is not yet the answer are listed
    here and printed at the top of the report rather than left to whoever is reading it.

    :param n_seeds: Runs contributing to the estimate.
    :param final_step: The last step every one of them evaluated at.
    :param horizon: The step count the tranche was submitted for.
    :param expected_seeds: How many cells the fan-out has.
    :param exclusions: One sentence per run that was read and then left out, from
        :func:`excluded`. Each is a reason on its own, so a reading missing a replicate cannot
        be frozen without somebody having read why.

    :returns: One sentence per reason, empty when the reading stands.
    """
    reasons = [f"a cell was read and excluded: {reason}" for reason in exclusions]
    if n_seeds < expected_seeds:
        reasons.append(
            f"{n_seeds} of {expected_seeds} cells have landed, so the df is "
            f"{max(n_seeds - 1, 0)} and not {expected_seeds - 1}"
        )
    if final_step <= 0:
        reasons.append(
            "the only shared evaluation is the eval_on_startup pass at step 0, which scores "
            "an untrained model and measures initialization scatter, not the endpoint"
        )
    elif final_step < horizon:
        reasons.append(
            f"the last shared evaluation is step {final_step} of {horizon}, so this is the "
            "noise floor of a partial run"
        )
    return reasons


def report(
    values: np.ndarray,
    sources: Sequence[str],
    steps: Sequence[int],
    seeds: Sequence[int],
    scheme: str = "strata",
    label: str = "measured",
    horizon: int = 6000,
    *,
    exclusions: Sequence[str] = (),
) -> Dict[str, object]:
    """
    Print the noise-floor table and return the numbers that get frozen.

    :param values: Shape ``(n_seeds, n_steps, n_sources)``.
    :param sources: Column names.
    :param steps: Evaluation checkpoints.
    :param seeds: Replicate indices, for the record.
    :param scheme: Which weighting scheme to freeze.
    :param label: ``measured`` or ``synthetic``, printed on every line that came from it.
    :param horizon: The submitted step count, for :func:`provisional_reasons`.
    :param exclusions: Runs that were read and left out, from :func:`excluded`. Carried into
        the banner and into the frozen dict, so a number computed without a replicate says so
        wherever it is read.

    :returns: A JSON-serializable dict of the frozen numbers.
    """
    n_seeds, n_steps, _ = values.shape
    final = values[:, -1, :]
    unweighted = final.mean(axis=1)

    print(
        f"[{label}] {n_seeds} seed(s) {list(seeds)}, {n_steps} checkpoint(s), "
        f"{len(sources)} source(s), final step {steps[-1]}"
    )
    reasons = provisional_reasons(n_seeds, int(steps[-1]), horizon, exclusions=exclusions)
    if reasons:
        print()
        print("PROVISIONAL. This is not the frozen noise floor, because:")
        for reason in reasons:
            print(f"  - {reason}")
    print()

    print("(a) sigma-hat on the primary endpoint, held-out BPB, unweighted mean over sources")
    endpoint = pooled_sigma([unweighted])
    print(f"    {_format_sigma(endpoint, 'BPB')}")
    print(f"    = {endpoint.sigma * NATS_PER_BPB:.5f} nats of held-out CE")
    print(
        f"    bias-corrected point estimate {endpoint.sigma_unbiased:.5f} BPB "
        f"(s / c4({endpoint.df}) = s / {c4(endpoint.df):.4f}); every MDE below uses this one"
    )
    print(f"    SE of a {n_seeds}-seed arm mean: {endpoint.standard_error_of_a_mean:.5f} BPB")
    print()

    print("    sigma against step, to see whether the noise floor has settled")
    trajectory = sigma_trajectory(values.mean(axis=2), steps)
    for step, sig, lo, hi in zip(
        trajectory.steps, trajectory.sigma, trajectory.ci_low, trajectory.ci_high
    ):
        print(f"      step {step:>6}  sigma {sig:.5f}  [{lo:.5f}, {hi:.5f}]")
    if n_steps < 2:
        print("      one checkpoint only; there is no trajectory to read yet.")
    else:
        verdict = (
            "cannot be distinguished from settled -- the interval covers 1"
            if trajectory.settled
            else "STILL MOVING at the final step"
        )
        print(
            f"      sigma(final)/sigma(step {trajectory.reference_step}) = {trajectory.ratio:.2f} "
            f"[{trajectory.ratio_ci[0]:.2f}, {trajectory.ratio_ci[1]:.2f}] -> {verdict}"
        )
    print()

    print("(b) per-source sigma and the inverse-variance weights derived from it")
    weights = inverse_variance_weights(final, sources, scheme)
    alternative = inverse_variance_weights(
        final, sources, "inverse-variance" if scheme == "strata" else "strata"
    )
    print(f"    scheme: {weights.scheme}")
    print(f"    {'source':<18}{'sigma':>10}{'stratum':>9}{'weight':>9}{'df':>5}")
    for j, source in enumerate(weights.sources):
        print(
            f"    {source:<18}{weights.sigma[j]:>10.5f}{weights.strata[j]:>9}"
            f"{weights.weights[j]:>9.4f}{weights.df_per_weight[j]:>5}"
        )
    print(f"    unweighted composite sigma {weights.unweighted_sigma:.5f} BPB")
    print(f"    weighted   composite sigma {weights.weighted_sigma:.5f} BPB")
    print(
        f"    variance reduction: {weights.variance_reduction:.2f}x in sample, "
        f"{weights.cross_validated_variance_reduction:.2f}x leave-one-seed-out, "
        f"{weights.diagonal_variance_reduction:.2f}x if the sources were independent"
    )
    print(
        f"    the other scheme ({alternative.scheme}) would give "
        f"{alternative.variance_reduction:.2f}x in sample and "
        f"{alternative.cross_validated_variance_reduction:.2f}x leave-one-seed-out"
    )
    print()

    print("(c) rho-hat: not computable from one arm. The estimator is committed and tested;")
    print("    stage 2 supplies the second arm. Sanity check on synthetic data at rho = 0.5:")
    a, b = synthetic_pair(n_seeds=max(n_seeds, 5), rho=0.5, rng_seed=7)
    check = paired_correlation(a, b)
    print(
        f"      rho_pearson {check.rho_pearson:+.3f}, rho_vc {check.rho_variance_components:+.3f}, "
        f"95% CI [{check.ci_low:+.3f}, {check.ci_high:+.3f}] over {check.n_pairs} pairs"
    )
    print()

    print(
        "    minimum detectable effect at the measured sigma, 3 arms x "
        f"{n_seeds} seeds, exact noncentral t, two-sided alpha 0.05, 80% power"
    )
    sigma_nats = endpoint.sigma_unbiased * NATS_PER_BPB
    weighted = pooled_sigma([final @ np.asarray(weights.weights)])
    weighted_nats = weighted.sigma_unbiased * NATS_PER_BPB
    for name, s in (("unweighted", sigma_nats), (f"{weights.scheme}-weighted", weighted_nats)):
        unpaired = mde(s, n_seeds, 3, 0.0, False)
        print(
            f"      {name:<20} unpaired (df {error_df(3, n_seeds, False):>2}) {unpaired:.4f} nats"
        )
        for rho in (0.0, 0.3, 0.5, 0.7):
            value = mde(s, n_seeds, 3, rho, True)
            print(
                f"      {'':<20} paired rho={rho:.1f} (df {error_df(3, n_seeds, True):>2}) "
                f"{value:.4f} nats"
            )

    # REPEATED AT THE BOTTOM BECAUSE THE BANNER AT THE TOP SCROLLS OFF. A reader arrives at
    # this report for the MDE table, which is the last thing in it, and a warning sixty lines
    # above the number being read is a warning that has already been missed once.
    if label != "measured":
        print()
        print(
            f"    ^ ALL {label.upper()}. Nothing above was measured; it is the generator's "
            "planted truth, and its per-source spread is a DataDecide-shaped fiction with the "
            "code sources at the noisy end. Pass --group to read W&B."
        )

    return {
        "label": label,
        "provisional": reasons,
        "excluded": list(exclusions),
        "n_seeds": n_seeds,
        "seeds": list(seeds),
        "final_step": int(steps[-1]),
        "sources": list(weights.sources),
        "sigma_bpb": endpoint.sigma,
        "sigma_bpb_unbiased": endpoint.sigma_unbiased,
        "sigma_nats": endpoint.sigma * NATS_PER_BPB,
        "sigma_nats_unbiased": sigma_nats,
        "sigma_df": endpoint.df,
        "sigma_ci_bpb": [endpoint.ci_low, endpoint.ci_high],
        "sigma_trajectory": {
            "steps": list(trajectory.steps),
            "sigma": list(trajectory.sigma),
            "ratio": trajectory.ratio,
            "ratio_ci": list(trajectory.ratio_ci),
            "settled": trajectory.settled,
        },
        "weights": {k: v for k, v in asdict(weights).items()},
    }


def self_test(replicates: int = 3000) -> int:
    """
    Check every estimator against a truth it was not shown, and say so out loud.

    THE POINT IS THAT THIS RUNS BEFORE THE DATA DOES. Each of these is a claim the tool makes
    about numbers nobody has seen yet, and an estimator that cannot recover a planted answer
    from synthetic data will not be corrected by real data -- it will be believed. Averaged
    over replicates rather than checked once, because at five seeds a single draw of any of
    these quantities is wide enough to pass or fail by luck.

    :param replicates: Synthetic datasets per check. The Monte Carlo tolerances below are
        widened as ``sqrt(3000 / replicates)`` so that a cheaper run is not a stricter test.

    :returns: A process exit status. Non-zero if any check misses.
    """
    failures = 0

    def check(name: str, got: float, want: float, tolerance: float) -> None:
        nonlocal failures
        ok = abs(got - want) <= tolerance
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<52} {got:+.4f} against {want:+.4f}")

    def check_mean(name: str, sample: Sequence[float], want: float, sigmas: float = 4.0) -> None:
        """
        Check a Monte Carlo mean against its own standard error rather than a hand-set number.

        A fixed tolerance is a claim about how many replicates were run, and it goes stale the
        moment somebody changes that argument -- becoming a *stricter* test on a smaller sample,
        which is backwards. Four standard errors of the mean is the same statement at every
        replicate count.
        """
        values = np.asarray(list(sample), dtype=float)
        error = float(values.std(ddof=1)) / math.sqrt(values.size)
        check(name, float(values.mean()), want, sigmas * error)

    print("estimator self-test, synthetic data with a planted truth")
    print()

    truth = 0.010
    estimates = [
        pooled_sigma([synthetic_pair(5, truth, 0.0, 0.0, r)[0]]) for r in range(replicates)
    ]
    check_mean(
        "pooled variance is unbiased, 5 seeds",
        [(e.sigma / truth) ** 2 for e in estimates],
        1.0,
    )
    check_mean(
        "the raw sigma is biased low by c4(4), as documented",
        [e.sigma / truth for e in estimates],
        c4(4),
    )
    check_mean(
        "sigma_unbiased takes that bias out",
        [e.sigma_unbiased / truth for e in estimates],
        1.0,
    )
    check_mean(
        "chi-square interval coverage at df = 4",
        [1.0 if e.ci_low <= truth <= e.ci_high else 0.0 for e in estimates],
        0.95,
    )

    # A Pearson r at five pairs is biased towards zero -- 0.70 comes back as 0.65 here -- so it
    # is checked against the planted value with a band wide enough to hold that bias rather
    # than against a closed form for it, which is a thing this module has no reason to need.
    # What the paired analysis consumes is sigma_delta, checked on the line below, and that one
    # is unbiased exactly and is held to its own Monte Carlo error.
    recovered = []
    for planted in (0.0, 0.3, 0.7):
        pairs = [
            paired_correlation(*synthetic_pair(5, truth, planted, 0.0, r))
            for r in range(replicates)
        ]
        mean_rho = float(np.mean([p.rho_pearson for p in pairs]))
        recovered.append(mean_rho)
        check(f"rho-hat (Pearson) at a planted rho of {planted:.1f}", mean_rho, planted, 0.06)
        check_mean(
            f"sigma_delta^2 against 2 sigma^2 (1 - rho) at {planted:.1f}",
            [p.sigma_delta**2 / (2.0 * truth**2 * (1.0 - planted)) for p in pairs],
            1.0,
        )
    check(
        "rho-hat is monotone in the planted correlation",
        1.0 if recovered == sorted(recovered) else 0.0,
        1.0,
        0.0,
    )
    check(
        "and biased towards zero rather than away from it",
        1.0 if all(r <= p + 1e-9 for r, p in zip(recovered[1:], (0.3, 0.7))) else 0.0,
        1.0,
        0.0,
    )

    print()
    print("  the df convention, which is what Deliverable 2 corrected")
    check("error df, 3 arms x 3 seeds, paired", error_df(3, 3, True), 4, 0)
    check("error df, 3 arms x 3 seeds, unpaired", error_df(3, 3, False), 6, 0)
    check("error df, 3 arms x 5 seeds, paired", error_df(3, 5, True), 8, 0)
    check("error df, 3 arms x 5 seeds, unpaired", error_df(3, 5, False), 12, 0)

    print()
    print("  the MDE solve inverts its own power function")
    for n, rho in ((3, 0.0), (5, 0.5), (5, 0.0)):
        effect = mde(truth, n, 3, rho, True)
        check(
            f"power at the {n}-seed rho={rho:.1f} paired MDE",
            power_of(effect, contrast_se(truth, n, rho, True), error_df(3, n, True)),
            0.80,
            1e-6,
        )

    print()
    print("  variance-interval spans the pre-registration quotes")
    check("span at df = 2", variance_interval_span(2), 12.1, 0.1)
    check("span at df = 4", variance_interval_span(4), 4.8, 0.1)
    check("span at df = 6", variance_interval_span(6), 3.4, 0.1)
    check("c4 at df = 4", c4(4), 0.9400, 0.0005)

    print()
    print("  weights recover the planted per-source ordering")
    values, sources, _ = synthetic_baseline(n_seeds=5, shared_fraction=0.0, rng_seed=3)
    weights = inverse_variance_weights(values[:, -1, :], sources, "strata")
    quiet = weights.weights[sources.index("dclm")]
    loud = weights.weights[sources.index("starcoder")]
    check("dclm outweighs starcoder", 1.0 if quiet > loud else 0.0, 1.0, 0.0)
    check("weights sum to one", float(sum(weights.weights)), 1.0, 1e-12)

    print()
    print("no misses." if not failures else f"{failures} check(s) missed.")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", help="The experiment slug the baseline seeds are grouped under.")
    parser.add_argument("--arm", default="baseline")
    parser.add_argument(
        "--submission",
        help="Restrict the read to the cells of one platform run id. The slug holds every "
        "attempt at this arm, including cancelled ones with the same seeds, so a freeze "
        "should name the submission it is freezing.",
    )
    parser.add_argument(
        "--scheme",
        default="strata",
        choices=("strata", "inverse-variance"),
        help="Which weighting to freeze. See inverse_variance_weights for why strata is the "
        "default at df = 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never require complete data. Reads whatever W&B has, reports coverage, and "
        "falls back to synthetic data with a known truth when nothing has landed yet -- so "
        "this can be exercised the hour the fan-out is submitted rather than the day it "
        "finishes. Every synthetic number is labelled as one.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check every estimator against a planted truth and exit. Reaches no network.",
    )
    parser.add_argument("--mde-table", action="store_true", help="Print the corrected MDE table.")
    parser.add_argument(
        "--sigma", type=float, default=PLANNING_SIGMA_NATS, help="Sigma for --mde-table, in nats."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6000,
        help="The step count the tranche was submitted for. A reading short of it is labelled "
        "provisional and cannot be frozen.",
    )
    parser.add_argument(
        "--freeze",
        help="Write the frozen numbers to this path as JSON. Refuses anything provisional.",
    )
    opts = parser.parse_args()

    if opts.self_test:
        return self_test()

    # --submission NARROWS A READ, AND THERE IS NOTHING TO NARROW WITHOUT --group. Naming one
    # without the other used to be accepted in silence and fall through to the synthetic
    # fallback below, which is a complete report -- per-source sigma, weights, variance
    # reductions, MDEs -- computed off a generator whose planted spread puts the two code
    # sources an order of magnitude above the rest. It says `[synthetic]` on every block and
    # it was read as a measurement of the submission that had been named, which cost about
    # twelve hours and came within one commit of freezing a weighting that deleted the code
    # sources on the strength of noise that was never in the data. The refusal is one line and
    # it is the whole of the fix.
    if opts.submission and not opts.group:
        parser.error(
            f"--submission {opts.submission} selects cells within a group, and no --group was "
            "given, so nothing would have been read. Add --group, or drop --submission to run "
            "the synthetic self-check."
        )

    if opts.mde_table:
        print(
            f"3 arms, paired, exact noncentral t, two-sided alpha 0.05, 80% power, "
            f"sigma = {opts.sigma:.3f} nats"
        )
        print(render_mde_table(opts.sigma))
        print()
        for n in (3, 4, 5):
            print(
                f"unpaired {n} v {n} (df {error_df(3, n, False)}): "
                f"{mde(opts.sigma, n, 3, 0.0, False):.3f} nats"
            )
        if not (opts.group or opts.dry_run):
            return 0

    values: Optional[np.ndarray] = None
    sources: Tuple[str, ...] = HELD_OUT_SOURCES
    steps: Tuple[int, ...] = ()
    seeds: Tuple[int, ...] = ()
    contributors: List[str] = []
    exclusions: List[str] = []
    label = "measured"

    if opts.group:
        series, sources = read_seed_series(
            opts.entity, opts.project, opts.group, opts.arm, submission=opts.submission
        )
        print(
            f"{opts.entity}/{opts.project}  group={opts.group}  arm={opts.arm}"
            + (f"  submission={opts.submission}" if opts.submission else "  (whole group)")
        )
        if not series:
            print(f"no runs of arm '{opts.arm}' in this group yet.")
        for entry in series:
            recovered = "  [step recovered from history]" if entry.summary_was_clobbered else ""
            print(
                f"  {entry.run_id or entry.run_name}  seed {entry.seed}  {entry.state}  "
                f"last step {entry.last_step}  {len(entry.per_source)} evaluation(s){recovered}"
            )

        # SAID SEPARATELY FROM THE LINE ABOVE, BECAUSE THE RECOVERY IS THE INTERESTING PART
        # AND A SUFFIX ON A DENSE LINE IS NOT READ. A cell whose summary was overwritten by
        # its own crash report reads `step None` in W&B and in anything built on summaries,
        # which is also what a cell that never started reads -- and one of these was reported
        # to the researcher as exactly that when it was the furthest-progressed cell in its
        # arm. The endpoints below come from history and are unaffected; what needed saying is
        # that the two records disagree and which one was believed.
        clobbered = [e for e in series if e.summary_was_clobbered]
        if clobbered:
            print()
            print(
                f"  {len(clobbered)} cell(s) have a summary that has lost its step. Recovered "
                "from history; the endpoints below never came from a summary:"
            )
            for entry in clobbered:
                print(
                    f"    {entry.run_id}  summary says step None, history says "
                    f"{entry.history_step}  ({len(entry.per_source)} evaluation(s) intact)"
                )
            print()

        left_out = excluded(series)
        exclusions = [
            f"{entry.run_id or entry.run_name} (seed {entry.seed}) {reason}"
            for entry, reason in left_out
        ]
        for entry, reason in left_out:
            print(f"  EXCLUDED  {entry.run_id or entry.run_name}  seed {entry.seed}: {reason}")

        contributors = [e.run_id or e.run_name for e in contributing(series)]
        candidate, steps, seeds = aligned_matrix(series, sources)
        if len(set(seeds)) != len(seeds):
            print()
            print(
                f"REFUSING TO ESTIMATE ANYTHING. Seeds {list(seeds)} are not distinct, so at "
                "least two cells of the fan-out ran the same replicate. That is the failure "
                "resolve_seed exists to refuse: identical curves, a measured noise floor of "
                "zero or near it, and every later arm significant against it."
            )
            return 1
        if candidate.size and candidate.shape[0] >= 2 and candidate.shape[1] >= 1:
            values = candidate
        else:
            print(
                "not enough landed to estimate anything: an across-seed sigma needs two or "
                "more runs sharing at least one evaluation step."
            )
            if not opts.dry_run:
                return 1
        print()

    if values is None:
        if not opts.dry_run:
            parser.error("give --group, or --dry-run to run against synthetic data")
        print("falling back to SYNTHETIC data with a known truth. Nothing below is measured.")
        print()
        synthetic, sources, steps = synthetic_baseline()
        values, seeds, label = synthetic, tuple(range(synthetic.shape[0])), "synthetic"

    frozen = report(
        values, sources, steps, seeds, opts.scheme, label, opts.horizon, exclusions=exclusions
    )
    frozen["entity"] = opts.entity
    frozen["project"] = opts.project
    frozen["group"] = opts.group
    frozen["arm"] = opts.arm
    frozen["submission"] = opts.submission
    frozen["runs"] = contributors
    frozen["metric"] = PRIMARY_METRIC
    frozen["horizon"] = opts.horizon

    if opts.freeze:
        if label != "measured" or frozen["provisional"]:
            print()
            print(
                f"refusing to freeze {label} numbers to {opts.freeze}. The point of freezing "
                "is that stage 2 is read against a floor nobody could revise after seeing it, "
                "and a provisional floor would be revised."
            )
            return 1
        with open(opts.freeze, "w") as handle:
            json.dump(frozen, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print()
        print(f"froze the noise floor to {opts.freeze}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
