#!/usr/bin/env python3
"""
Turn the mixer bake-off's per-cell summary JSONs into a production recommendation.

THE ANALYSIS IS PRE-REGISTERED. ``docs/mixer-bakeoff/PREREGISTRATION.md`` and
``docs/mixer-bakeoff/seeds.json`` are the contract; this file implements it and nothing else.
Where it deviates it says so out loud, in the report, in a section called DEVIATIONS -- an
undeclared deviation from a pre-registration is the same thing as not having one.

Run it::

    python scripts/analyse_bakeoff.py results/            # a directory of per-cell JSONs
    python scripts/analyse_bakeoff.py a.json b.json ...   # explicit files
    ls results/*.json | python scripts/analyse_bakeoff.py # paths on stdin
    cat cell.log | python scripts/analyse_bakeoff.py      # raw trainer logs on stdin

It writes ``bakeoff_results.json`` (machine-readable) and prints a markdown report (the
deliverable) to stdout.

NOTHING HERE TOUCHES AWS AND NOTHING HERE IS COMPUTATIONAL. It is arithmetic over a few
hundred numbers. ``--print-fetch-command s3://...`` prints the ``aws s3 cp`` line to run
*elsewhere*; it does not run it. That separation is deliberate.

NO SCIPY. The statistics below (regularized incomplete beta and gamma, Student t, F, chi^2,
the *exact* non-central t survival function, and the equicorrelated max-|t| integral that
gives Dunnett's critical value) are implemented here in the standard library, because the
environment this has to run in does not reliably have scipy. Every one of them is validated
in ``src/test/scripts/analyse_bakeoff_test.py`` against numbers that were pre-registered
before this file existed.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------------
# Frozen constants. Every one of these is quoted from the pre-registration or from the source
# it names, and every one of them is checkable against that source rather than believed.
# --------------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEEDS_JSON = REPO_ROOT / "docs" / "mixer-bakeoff" / "seeds.json"
DEFAULT_ARMS_SOURCE = REPO_ROOT / "src" / "olmo_core" / "nn" / "transformer" / "core6_arms.py"

#: Dolma2 vocabulary. ``PREREGISTRATION.md`` s3: ``first_loss`` should be ~= ln(vocab) = 11.52.
VOCAB_SIZE = 100352
LN_VOCAB = math.log(VOCAB_SIZE)

#: Init sanity band for ``first_loss``: ln(vocab) +/- 0.5. An init that is not here did not
#: start from a uniform distribution over the vocabulary, which is the one check in this
#: project's history that has ever caught uninitialised weights.
FIRST_LOSS_BAND: Tuple[float, float] = (11.016, 12.016)

#: Plausibility band for the primary endpoint, in nats. An ABSOLUTE-MAGNITUDE gate, not an
#: existence check -- ``val_ce is not None`` passes for a diverged run that landed on 11.5 and
#: for a leaked eval that landed on 0.4.
#:
#: Upper edge 11.016 = ln(vocab) - 0.5: a model that is not at least half a nat better than
#: uniform did not converge. Lower edge 1.5: a 390M model on 1.0B tokens of held-out dolma2
#: cannot reach 1.5 nats, so a number below it means the denominator or the partition is
#: wrong, not that the model is good.
VAL_CE_BAND: Tuple[float, float] = (1.5, 11.016)

#: A cell this far in nats from the median of the admissible cells is WARNED ABOUT, not
#: excluded. At an expected sigma of ~0.01 nats a 0.5-nat gap is ~50 sigma and is almost
#: certainly a different run -- but the pre-registration does not pre-commit an outlier rule,
#: and inventing an exclusion criterion after seeing the data is exactly what a
#: pre-registration exists to prevent. So: flagged loudly, left in, Eric decides.
OUTLIER_WARN_NATS = 0.5

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_CONTROL = "KDA_BASE"

#: Pre-registration s5.1: the in-tree measurement of how much a low-token-budget architecture
#: comparison overstates. Quoted in the report because a CE magnitude from this run is an
#: upper bound on the production effect, not the production effect.
TPP = 2.6
GDN_EDGE_AT_1B = 0.0103
GDN_EDGE_AT_15B = 0.0059

#: Pre-registration s5: the two available sigma estimates, differing 5.5x. Reported alongside
#: the *measured* sigma so the reader can see which one run 1 actually landed on.
SIGMA_OPTIMISTIC = 0.0019
SIGMA_PESSIMISTIC = 0.0105

#: Pre-registration s5.1 / s5: the literature CE gap these arms are being hunted for.
LITERATURE_CE_GAP = (0.010, 0.030)

#: Keys ``summarise()`` writes that this script consumes. Anything else a cell carries is
#: reported under ``unconsumed_keys`` rather than silently dropped -- ``summarise()`` grew
#: throughput and memory keys mid-project, and a key ignored without saying so is a key
#: nobody knows was ignored.
CONSUMED_KEYS = frozenset(
    {
        "run_id",
        "arm",
        "data_seed",
        "init_seed",
        "parameters",
        "steps",
        "tokens_trained",
        "world_size",
        "gpu",
        "torch",
        "cuda",
        "seconds",
        "first_loss",
        "last_loss",
        "val_ce",
        "val_tokens",
        "val_tokens_present",
        "val_tokens_declared",
        "val_shards",
        "val_nll_sum",
        # The speed half of the record, as committed at 51c1a60.
        "throughput_tok_s_steady",
        "throughput_tok_s_steady_per_device",
        "throughput_tok_s_whole_run",
        "throughput_tok_s_whole_run_per_device",
        "throughput_tok_s_all_steps",
        "steps_measured",
        "steady_state_steps",
        "warmup_steps_excluded",
        "tokens_in_steady_window",
        "step_time_s_p50",
        "step_time_s_p90",
        "steady_window_seconds",
        "training_seconds_excluding_startup",
        "mfu_pct",
        "mfu_basis",
        "device_peak_bf16_flops",
        "flops_per_token",
        # The memory half.
        "peak_memory_gib",
        "peak_memory_reserved_gib",
        "peak_memory_source",
        "peak_memory_samples",
        # Upstream SpeedMonitorCallback's independent measurement, kept for cross-check.
        "tps_device_avg",
        "tps_device_last",
        "tps_total_avg",
        "tps_naive_wall_clock",
        "production_decision",
        "sliced_eval",
        "checkpoint_uri",
        "wandb_project",
        "wandb_url",
        "dataset_id",
        "dataset_version",
        "image_digest",
    }
)

#: Alias tables, most-preferred first. ``throughput_tok_s_steady`` is the committed name and
#: the figure to rank arms on; ``tps_total_avg`` is upstream ``SpeedMonitorCallback``'s
#: independent measurement of nearly the same thing over a window starting after step 1
#: rather than after the warmup cutoff, kept as a fallback and as a cross-check.
#:
#: WHICH KEY SUPPLIED EACH NUMBER IS RECORDED. A ratio whose numerator came from one
#: definition on one arm and another definition on another arm is worse than no ratio.
THROUGHPUT_TOTAL_STEADY_KEYS = ("throughput_tok_s_steady", "tps_total_avg")
THROUGHPUT_DEVICE_STEADY_KEYS = ("throughput_tok_s_steady_per_device", "tps_device_avg")
THROUGHPUT_WHOLE_RUN_KEYS = ("throughput_tok_s_whole_run", "tps_naive_wall_clock")
PEAK_MEMORY_KEYS = ("peak_memory_gib",)
STEP_TIME_MEDIAN_KEYS = ("step_time_s_p50",)
STEP_TIME_P90_KEYS = ("step_time_s_p90",)

#: ``memory_report()``'s three sources. Only the first is a whole-run peak.
#:
#: ``final_step_only`` is A LOWER BOUND WEARING THE NAME OF A PEAK -- the GPU monitor resets
#: the counters every step, so it is the last step's peak and nothing more. It looks exactly
#: like the real figure in the value. Comparing an arm measured one way against an arm
#: measured the other is comparing two different quantities, so a mixed set is a loud finding
#: rather than a footnote.
MEMORY_SOURCE_TRUSTED = "per_step_running_max"
MEMORY_SOURCE_LOWER_BOUND = "final_step_only"
MEMORY_SOURCE_UNAVAILABLE = "unavailable"

#: ``.edullm/train_core6_arm.py``'s warmup cutoff at the frozen commit. A steady-state figure
#: computed over very few post-warmup steps is not the same measurement as one over hundreds,
#: and this project measured the steady figure itself moving 1.5x between a 20-step and a
#: 200-step run. Below this many steady steps the number is flagged.
MIN_STEADY_STEPS = 100

#: Arm-major cell index -> arm, transcribed from ``.edullm/run-bakeoff.yaml``. Used ONLY to
#: cross-check the arm each JSON reports against the slot it was launched in. It is never used
#: to infer an arm: a disagreement is a finding, not something to reconcile.
CELL_INDEX_TO_ARM: Dict[int, str] = {
    i: arm
    for i, arm in enumerate(
        ["KDA_BASE"] * 3
        + ["KDA_NOACT"] * 3
        + ["KDA_GCONV"] * 3
        + ["GDN2"] * 3
        + ["KDA_R1"] * 3
        + ["KDA_R2"] * 3
    )
}


# --------------------------------------------------------------------------------------------
# Special functions. Standard library only.
# --------------------------------------------------------------------------------------------

_TINY = 1e-300
_EPS = 3e-16


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float, itmax: int = 400) -> float:
    """Continued fraction for the incomplete beta (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)``, the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def regularized_gamma_p(a: float, x: float) -> float:
    """``P(a, x)``, the regularized lower incomplete gamma function."""
    if x < 0.0 or a <= 0.0:
        raise ValueError(f"regularized_gamma_p domain error: a={a}, x={x}")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        # Series representation.
        ap = a
        total = 1.0 / a
        term = total
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q, then complement.
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_sf(x: float, df: float) -> float:
    """Upper tail of the chi-squared distribution."""
    if x <= 0.0:
        return 1.0
    return 1.0 - regularized_gamma_p(df / 2.0, x / 2.0)


def _bisect(fn, lo: float, hi: float, target: float, iters: int = 200) -> float:
    """Bisection on a monotone ``fn`` for ``fn(x) == target``. No derivative, no surprises."""
    flo = fn(lo) - target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid) - target
        if fmid == 0.0:
            return mid
        if (fmid > 0.0) == (flo > 0.0):
            lo = mid
            flo = fmid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def chi2_ppf(p: float, df: float) -> float:
    """Quantile of the chi-squared distribution."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"chi2_ppf needs 0 < p < 1, got {p}")
    hi = max(1.0, df)
    while chi2_sf(hi, df) > 1.0 - p:
        hi *= 2.0
        if hi > 1e12:
            break
    return _bisect(lambda x: 1.0 - chi2_sf(x, df), 0.0, hi, p)


def student_t_sf(t: float, df: float) -> float:
    """Upper tail of Student's t."""
    x = df / (df + t * t)
    half = 0.5 * regularized_incomplete_beta(x, df / 2.0, 0.5)
    return half if t > 0.0 else 1.0 - half


def student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"student_t_ppf needs 0 < p < 1, got {p}")
    return _bisect(lambda t: 1.0 - student_t_sf(t, df), -1e3, 1e3, p)


def f_sf(f_stat: float, df1: float, df2: float) -> float:
    """Upper tail of the F distribution."""
    if f_stat <= 0.0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    return regularized_incomplete_beta(x, df2 / 2.0, df1 / 2.0)


def noncentral_t_cdf(t: float, df: float, ncp: float, max_terms: int = 2000) -> float:
    """
    CDF of the non-central t distribution -- the EXACT one, not the normal approximation.

    Lenth (1989) / AS 243. The normal approximation is 2.2x too optimistic at n=3 in this
    project's own measurement and is not used anywhere.
    """
    if t < 0.0:
        return 1.0 - noncentral_t_cdf(-t, df, -ncp)
    x = t * t / (t * t + df)
    if x <= 0.0:
        return normal_cdf(-ncp)
    lam = 0.5 * ncp * ncp
    if lam == 0.0:
        # Central case: falls out of the same identity with a single term.
        return normal_cdf(0.0) + 0.5 * regularized_incomplete_beta(x, 0.5, df / 2.0)
    log_lam = math.log(lam)
    half_df = df / 2.0
    total = 0.0
    for j in range(max_terms):
        log_pj = -lam + j * log_lam - math.lgamma(j + 1.0)
        log_qj = -lam + j * log_lam - math.lgamma(j + 1.5)
        pj = math.exp(log_pj)
        qj = math.exp(log_qj) * ncp / math.sqrt(2.0)
        term = pj * regularized_incomplete_beta(x, j + 0.5, half_df)
        term += qj * regularized_incomplete_beta(x, j + 1.0, half_df)
        total += term
        # Stop only once past the mode of the Poisson weights, or an early tiny term ends it.
        if j > lam + 10 and abs(term) < 1e-15 * max(1.0, abs(total)):
            break
    return min(1.0, max(0.0, normal_cdf(-ncp) + 0.5 * total))


def noncentral_t_sf(t: float, df: float, ncp: float) -> float:
    """
    DOMINANT-TAIL survival function of the non-central t. This is the power estimator the
    pre-registration commits to.

    The naive two-tail form ``sf(c) + cdf(-c)`` suffers catastrophic cancellation, and the
    lower tail is negligible for any ncp large enough to matter. Dominant tail only.
    """
    return 1.0 - noncentral_t_cdf(t, df, ncp)


# --------------------------------------------------------------------------------------------
# Gauss-Legendre quadrature and the equicorrelated max-|t| integral (Dunnett).
# --------------------------------------------------------------------------------------------

_GL_CACHE: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {}


def gauss_legendre_nodes(n: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Nodes and weights for n-point Gauss-Legendre quadrature on [-1, 1]."""
    if n in _GL_CACHE:
        return _GL_CACHE[n]
    nodes: List[float] = []
    weights: List[float] = []
    for i in range(1, n + 1):
        # Chebyshev starting guess, then Newton on the Legendre polynomial.
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, 0.0
            for j in range(1, n + 1):
                p0, p1 = ((2.0 * j - 1.0) * x * p0 - (j - 1.0) * p1) / j, p0
            dp = n * (x * p0 - p1) / (x * x - 1.0)
            dx = -p0 / dp
            x += dx
            if abs(dx) < 1e-15:
                break
        p0, p1 = 1.0, 0.0
        for j in range(1, n + 1):
            p0, p1 = ((2.0 * j - 1.0) * x * p0 - (j - 1.0) * p1) / j, p0
        dp = n * (x * p0 - p1) / (x * x - 1.0)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    result = (tuple(nodes), tuple(weights))
    _GL_CACHE[n] = result
    return result


def _scaled_nodes(n: int, lo: float, hi: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    nodes, weights = gauss_legendre_nodes(n)
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    return tuple(mid + half * x for x in nodes), tuple(half * w for w in weights)


def _chi_scale_log_pdf(u: float, df: float) -> float:
    """log density of U = S/sigma where ``df * U^2 ~ chi^2_df``."""
    return (
        math.log(2.0)
        + (df / 2.0) * math.log(df / 2.0)
        - math.lgamma(df / 2.0)
        + (df - 1.0) * math.log(u)
        - df * u * u / 2.0
    )


def _chi_scale_quadrature(df: float, n_nodes: int) -> Tuple[List[float], List[float]]:
    """
    Nodes and weights for integrating against the density of ``U = S/sigma``.

    THE INTEGRATION WINDOW MUST FOLLOW THE MODE, WHICH CONCENTRATES AS df GROWS. U has mean
    ~1 and sd ~1/sqrt(2 df), so a fixed window that is right at df = 12 is thousands of
    standard deviations wide at df = 1e5 and a fixed node count cannot resolve the spike. A
    first version of this function used a fixed ``[0, 1 + 16/sqrt(2 df)]`` and returned 0.84
    for a critical value whose true answer is 1.96 -- a plausible-looking number, silently
    wrong, in a regime the k = 1 reduction check would not have visited.

    Split into panels so the resolution scales with the window rather than with df.
    """
    width = 14.0 / math.sqrt(2.0 * df)
    lo = max(1e-12, 1.0 - width)
    hi = 1.0 + width
    n_panels = 4
    nodes: List[float] = []
    weights: List[float] = []
    edges = [lo + (hi - lo) * i / n_panels for i in range(n_panels + 1)]
    for left, right in zip(edges[:-1], edges[1:]):
        panel_nodes, panel_weights = _scaled_nodes(n_nodes, left, right)
        nodes.extend(panel_nodes)
        weights.extend(panel_weights)
    return nodes, weights


def chi_scale_mass(df: float, n_nodes: int = 64) -> float:
    """
    Total probability mass the quadrature window captures. MUST be ~1.

    This is the guard that makes a mis-resolved integral loud instead of plausible. It is
    called on every critical-value computation, and it is fireable: the window that produced
    the 0.84 bug captures mass 0.36, which this rejects.
    """
    nodes, weights = _chi_scale_quadrature(df, n_nodes)
    return sum(w * math.exp(_chi_scale_log_pdf(u, df)) for u, w in zip(nodes, weights))


def equicorrelated_max_abs_t_cdf(
    c: float, k: int, df: float, rho: float = 0.5, n_nodes: int = 64
) -> float:
    """
    ``P(max_i |T_i| <= c)`` for k equicorrelated t variates sharing one variance estimate.

    This is the Dunnett two-sided probability. With a shared control and equal n, rho = 1/2
    exactly. rho = 0 gives the studentized maximum modulus, which is the unequal-variance
    fallback's critical value.

    Represent ``W_i = sqrt(rho) Z + sqrt(1-rho) X_i`` with Z, X_i iid standard normal, and
    ``T_i = W_i / U`` with ``df U^2 ~ chi^2_df``. Then integrate out U and Z by Gauss-Legendre
    quadrature.
    """
    if c <= 0.0:
        return 0.0
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    u_nodes, u_weights = _chi_scale_quadrature(df, n_nodes)
    sr = math.sqrt(rho)
    s1r = math.sqrt(1.0 - rho)
    if rho == 0.0:
        z_nodes: Tuple[float, ...] = (0.0,)
        z_weights: Tuple[float, ...] = (1.0,)
    else:
        z_nodes, z_weights = _scaled_nodes(n_nodes, -8.5, 8.5)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    total = 0.0
    for u, wu in zip(u_nodes, u_weights):
        y = c * u
        if rho == 0.0:
            inner = (2.0 * normal_cdf(y) - 1.0) ** k
        else:
            inner = 0.0
            for z, wz in zip(z_nodes, z_weights):
                phi = inv_sqrt_2pi * math.exp(-0.5 * z * z)
                hi = (y - sr * z) / s1r
                lo = (-y - sr * z) / s1r
                prob = normal_cdf(hi) - normal_cdf(lo)
                if prob <= 0.0:
                    continue
                inner += wz * phi * prob**k
        total += wu * math.exp(_chi_scale_log_pdf(u, df)) * inner
    return min(1.0, max(0.0, total))


_CRIT_CACHE: Dict[Tuple[int, float, float, float, int], float] = {}


def dunnett_two_sided_critical_value(
    k: int, df: float, alpha: float = DEFAULT_ALPHA, rho: float = 0.5, n_nodes: int = 64
) -> float:
    """
    Two-sided Dunnett critical value by Gaussian quadrature over the max-|t| integral.

    Memoised: the bisection evaluates a double quadrature ~50 times and the same (k, df, rho)
    recurs across the contrast table, the validation and the MDE. Pure function of its
    arguments, so the cache cannot go stale.
    """
    key = (k, float(df), float(alpha), float(rho), n_nodes)
    if key not in _CRIT_CACHE:
        _CRIT_CACHE[key] = _bisect(
            lambda c: equicorrelated_max_abs_t_cdf(c, k, df, rho, n_nodes),
            0.1,
            20.0,
            1.0 - alpha,
            iters=80,
        )
    return _CRIT_CACHE[key]


def studentized_max_modulus_critical_value(
    k: int, df: float, alpha: float = DEFAULT_ALPHA, n_nodes: int = 64
) -> float:
    """SMM critical value: the rho = 0 case, used by the unequal-variance fallback."""
    return dunnett_two_sided_critical_value(k, df, alpha, rho=0.0, n_nodes=n_nodes)


def validate_dunnett_critical_value(
    k: int, df: float, alpha: float = DEFAULT_ALPHA, rho: float = 0.5
) -> Dict[str, Any]:
    """
    Compute the Dunnett critical value and CHECK IT TWO WAYS, because a quadrature that
    silently under-resolves returns a plausible number.

    Check 1 -- quadrature refinement: 48 nodes vs 96 nodes must agree. If they do not, the
    integral has not converged and the number is not usable.

    Check 2 -- the k = 1 reduction: with a single comparison the max-|t| is just |t|, so the
    critical value must equal the ordinary two-sided Student t quantile at the same df. That
    quantile is computed by a completely different route (inverse regularized incomplete
    beta), so agreement is evidence about the integral and not about itself.

    Check 3 -- the integration window captures ~all the probability mass. Checks 1 and 2 can
    both pass on a window that has drifted off the mode, because a coarse and a fine
    quadrature of the same wrong window agree with each other. This one is what caught a
    fixed-width window returning 0.84 for a 1.96 critical value.
    """
    coarse = dunnett_two_sided_critical_value(k, df, alpha, rho, n_nodes=48)
    fine = dunnett_two_sided_critical_value(k, df, alpha, rho, n_nodes=96)
    k1_quadrature = dunnett_two_sided_critical_value(1, df, alpha, rho, n_nodes=96)
    k1_student = student_t_ppf(1.0 - alpha / 2.0, df)
    mass = chi_scale_mass(df, 96)
    return {
        "method": "Gaussian quadrature (Gauss-Legendre) over the equicorrelated max-|t| integral",
        "k": k,
        "df": df,
        "alpha": alpha,
        "rho": rho,
        "critical_value": fine,
        "check_1_quadrature_refinement": {
            "n_nodes_48": coarse,
            "n_nodes_96": fine,
            "abs_diff": abs(coarse - fine),
            "passed": abs(coarse - fine) < 1e-6,
        },
        "check_2_k1_reduces_to_student_t": {
            "quadrature_at_k1": k1_quadrature,
            "student_t_ppf": k1_student,
            "abs_diff": abs(k1_quadrature - k1_student),
            "passed": abs(k1_quadrature - k1_student) < 1e-6,
        },
        "check_3_integration_window_captures_the_mass": {
            "mass": mass,
            "passed": abs(mass - 1.0) < 1e-6,
        },
    }


# --------------------------------------------------------------------------------------------
# Power / MDE. Exact non-central t, dominant tail only.
# --------------------------------------------------------------------------------------------


def contrast_standard_error(sigma: float, n_treat: int, n_control: int) -> float:
    """SE of a difference of two arm means at a common sigma."""
    if n_treat < 1 or n_control < 1:
        raise ValueError(f"n must be >= 1, got n_treat={n_treat}, n_control={n_control}")
    return sigma * math.sqrt(1.0 / n_treat + 1.0 / n_control)


def power_for_effect(
    effect: float, sigma: float, n_treat: int, n_control: int, crit: float, df: float
) -> float:
    """Power to detect ``effect`` at critical value ``crit``. Exact non-central t."""
    se = contrast_standard_error(sigma, n_treat, n_control)
    if se <= 0.0:
        raise ValueError(f"standard error must be positive, got {se}")
    return noncentral_t_sf(crit, df, abs(effect) / se)


def mde_for_power(
    sigma: float,
    n_treat: int,
    n_control: int,
    crit: float,
    df: float,
    power: float = DEFAULT_POWER,
) -> float:
    """
    The minimum detectable effect, in the endpoint's own units, at the requested power.

    Exact non-central t, dominant tail only, inverted by bisection on the non-centrality.
    """
    se = contrast_standard_error(sigma, n_treat, n_control)
    ncp = _bisect(lambda d: noncentral_t_sf(crit, df, d), 0.0, 60.0, power, iters=120)
    return ncp * se


def sigma_chi2_interval(df: float, alpha: float = DEFAULT_ALPHA) -> Tuple[float, float]:
    """
    Multiplicative confidence interval on a pooled sigma estimate: ``sigma_hat * [lo, hi]``.

    At df = 12 this is [0.717, 1.651] -- a factor-2.3 bracket. Reported because the pooled
    sigma is a pre-registered deliverable of run 1 and quoting it without its own uncertainty
    would replace a 5.5x guess with a false precision.
    """
    lo = math.sqrt(df / chi2_ppf(1.0 - alpha / 2.0, df))
    hi = math.sqrt(df / chi2_ppf(alpha / 2.0, df))
    return lo, hi


# --------------------------------------------------------------------------------------------
# Group statistics.
# --------------------------------------------------------------------------------------------


def mean_sd_n(values: Sequence[float]) -> Dict[str, Any]:
    """
    Mean, sample sd (ddof=1) and n.

    ``sd`` IS ``None`` AT n < 2, NEVER 0.0. A single observation has no variance estimate and
    reporting 0.0 there is how one number gets read as a mean with tight error bars.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "values": []}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n >= 2 else None
    return {"n": n, "mean": mean, "sd": sd, "values": list(values)}


def pooled_variance(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """
    Pooled within-group variance and its df.

    ``df = sum_i (n_i - 1)``. A group with n < 2 contributes NOTHING -- not zero variance,
    nothing -- and is named in ``groups_without_variance`` so a shrinking df is visible.
    """
    ss = 0.0
    df = 0
    contributing = 0
    for values in groups:
        n = len(values)
        if n < 2:
            continue
        m = statistics.fmean(values)
        ss += sum((v - m) ** 2 for v in values)
        df += n - 1
        contributing += 1
    if df == 0:
        return {"variance": None, "sd": None, "df": 0, "contributing_groups": 0}
    var = ss / df
    return {
        "variance": var,
        "sd": math.sqrt(var),
        "df": df,
        "contributing_groups": contributing,
    }


def one_way_anova(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """
    Pooled-variance one-way ANOVA. Pre-registration s4.3: this, not independent pairwise
    t-tests, which throw away 4/5 of the df.
    """
    usable = [list(g) for g in groups if len(g) >= 1]
    k = len(usable)
    total_n = sum(len(g) for g in usable)
    if k < 2 or total_n - k < 1:
        return {
            "computable": False,
            "reason": f"need >= 2 groups and error df >= 1; got k={k}, N={total_n}",
            "f": None,
            "df_between": None,
            "df_within": None,
            "p": None,
        }
    grand = statistics.fmean([v for g in usable for v in g])
    ss_between = sum(len(g) * (statistics.fmean(g) - grand) ** 2 for g in usable)
    ss_within = sum(sum((v - statistics.fmean(g)) ** 2 for v in g) for g in usable)
    df_between = k - 1
    df_within = total_n - k
    ms_within = ss_within / df_within
    if ms_within <= 0.0:
        return {
            "computable": False,
            "reason": "within-group mean square is zero; F is undefined",
            "f": None,
            "df_between": df_between,
            "df_within": df_within,
            "p": None,
        }
    f_stat = (ss_between / df_between) / ms_within
    return {
        "computable": True,
        "reason": None,
        "f": f_stat,
        "df_between": df_between,
        "df_within": df_within,
        "ms_within": ms_within,
        "p": f_sf(f_stat, df_between, df_within),
    }


def levene_median(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """
    Levene's test, median-centred (Brown-Forsythe). Pre-registration s4.5: THE decision test
    for variance homogeneity.
    """
    usable = [list(g) for g in groups if len(g) >= 2]
    if len(usable) < 2:
        return {
            "computable": False,
            "reason": f"need >= 2 groups with n >= 2; got {len(usable)}",
            "statistic": None,
            "p": None,
        }
    deviations = [[abs(v - statistics.median(g)) for v in g] for g in usable]
    result = one_way_anova(deviations)
    return {
        "computable": result["computable"],
        "reason": result["reason"],
        "statistic": result["f"],
        "df_between": result["df_between"],
        "df_within": result["df_within"],
        "p": result["p"],
    }


def bartlett(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Bartlett's test. Pre-registration s4.5: reported alongside Levene, not deciding."""
    usable = [list(g) for g in groups if len(g) >= 2]
    k = len(usable)
    if k < 2:
        return {
            "computable": False,
            "reason": f"need >= 2 groups with n >= 2; got {k}",
            "statistic": None,
            "p": None,
        }
    variances = [statistics.variance(g) for g in usable]
    if any(v <= 0.0 for v in variances):
        return {
            "computable": False,
            "reason": "a group has zero sample variance; the log is undefined",
            "statistic": None,
            "p": None,
        }
    ns = [len(g) for g in usable]
    total_n = sum(ns)
    pooled = sum((n - 1) * v for n, v in zip(ns, variances)) / (total_n - k)
    numerator = (total_n - k) * math.log(pooled) - sum(
        (n - 1) * math.log(v) for n, v in zip(ns, variances)
    )
    correction = 1.0 + (sum(1.0 / (n - 1) for n in ns) - 1.0 / (total_n - k)) / (3.0 * (k - 1))
    stat = numerator / correction
    return {
        "computable": True,
        "reason": None,
        "statistic": stat,
        "df": k - 1,
        "p": chi2_sf(stat, k - 1),
    }


def welch_anova(groups: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Welch's heteroscedastic one-way ANOVA. The pre-registered fallback when Levene rejects."""
    usable = [list(g) for g in groups if len(g) >= 2]
    k = len(usable)
    if k < 2:
        return {"computable": False, "reason": f"need >= 2 groups with n >= 2; got {k}"}
    variances = [statistics.variance(g) for g in usable]
    if any(v <= 0.0 for v in variances):
        return {"computable": False, "reason": "a group has zero sample variance"}
    ns = [len(g) for g in usable]
    means = [statistics.fmean(g) for g in usable]
    w = [n / v for n, v in zip(ns, variances)]
    sw = sum(w)
    grand = sum(wi * m for wi, m in zip(w, means)) / sw
    a = sum(wi * (m - grand) ** 2 for wi, m in zip(w, means)) / (k - 1)
    lam = sum((1.0 - wi / sw) ** 2 / (n - 1) for wi, n in zip(w, ns))
    b = 1.0 + (2.0 * (k - 2) / (k * k - 1.0)) * lam
    f_stat = a / b
    df2 = (k * k - 1.0) / (3.0 * lam)
    return {
        "computable": True,
        "reason": None,
        "f": f_stat,
        "df_between": k - 1,
        "df_within": df2,
        "p": f_sf(f_stat, k - 1, df2),
    }


def dunnett_contrasts(
    arm_values: Dict[str, List[float]],
    control: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """
    Every arm against the shared control, pooled variance, Dunnett two-sided correction.

    Pre-registration s4.4 and s4.6: effect, Dunnett-adjusted CI and n for every contrast --
    never a bare p-value, never "n.s." as though it meant "no effect".
    """
    if control not in arm_values:
        return {"computable": False, "reason": f"control arm {control!r} has no admissible cells"}
    control_values = arm_values[control]
    treatments = [a for a in arm_values if a != control]
    if len(control_values) < 2:
        return {
            "computable": False,
            "reason": (
                f"control arm {control} has n={len(control_values)}; a contrast against a "
                "control with no variance estimate is not a contrast"
            ),
        }
    if not treatments:
        return {"computable": False, "reason": "no treatment arms with admissible cells"}
    pooled = pooled_variance([arm_values[a] for a in arm_values])
    if pooled["df"] < 1 or pooled["sd"] is None:
        return {"computable": False, "reason": "pooled variance has no df"}
    sd_p = pooled["sd"]
    df = float(pooled["df"])
    k = len(treatments)
    n0 = len(control_values)
    ns = [len(arm_values[a]) for a in treatments]
    balanced = len(set(ns)) == 1 and ns[0] == n0
    if balanced:
        rho = 0.5
        rho_note = "exact (balanced design, shared control)"
    else:
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append(
                    1.0 / math.sqrt((1.0 + n0 / ns[i]) * (1.0 + n0 / ns[j]))
                )
        rho = statistics.fmean(pairs) if pairs else 0.5
        rho_note = (
            f"APPROXIMATE: design is unbalanced (n = {ns} vs control n = {n0}), so the "
            f"correlations are not all equal; the mean correlation {rho:.4f} is used in the "
            "equicorrelated integral. The critical value is approximate for this design."
        )
    validation = validate_dunnett_critical_value(k, df, alpha, rho)
    crit = validation["critical_value"]
    control_mean = statistics.fmean(control_values)
    rows = []
    for arm in treatments:
        values = arm_values[arm]
        n = len(values)
        est = statistics.fmean(values) - control_mean
        se = sd_p * math.sqrt(1.0 / n + 1.0 / n0)
        t_stat = est / se if se > 0.0 else None
        half = crit * se
        adj_p = None
        if t_stat is not None:
            adj_p = 1.0 - equicorrelated_max_abs_t_cdf(abs(t_stat), k, df, rho, n_nodes=64)
        rows.append(
            {
                "arm": arm,
                "n": n,
                "n_control": n0,
                "estimate": est,
                "se": se,
                "t": t_stat,
                "ci_low": est - half,
                "ci_high": est + half,
                "ci_half_width": half,
                "dunnett_adjusted_p": adj_p,
                "excludes_zero": (est - half > 0.0) or (est + half < 0.0),
            }
        )
    return {
        "computable": True,
        "reason": None,
        "control": control,
        "k": k,
        "df": df,
        "pooled_sd": sd_p,
        "alpha": alpha,
        "rho": rho,
        "rho_note": rho_note,
        "critical_value": crit,
        "critical_value_validation": validation,
        "contrasts": rows,
    }


def welch_t3_contrasts(
    arm_values: Dict[str, List[float]],
    control: str,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """
    The unequal-variance fallback: each arm against the control on a Welch-Satterthwaite SE
    and df, with a studentized-maximum-modulus critical value over the k comparisons.

    DECLARED DEVIATION. The pre-registration names "Games-Howell". Games-Howell is the
    ALL-PAIRS procedure and its critical value is the studentized RANGE. These comparisons are
    k arms against ONE control, whose unequal-variance analogue is Dunnett's T3, which uses
    the studentized maximum modulus -- the rho = 0 case of the same integral that gives the
    Dunnett critical value above. T3 is used here, and the substitution is reported in the
    DEVIATIONS section rather than made quietly.
    """
    if control not in arm_values or len(arm_values[control]) < 2:
        return {"computable": False, "reason": f"control arm {control!r} has n < 2"}
    treatments = [a for a in arm_values if a != control and len(arm_values[a]) >= 2]
    if not treatments:
        return {"computable": False, "reason": "no treatment arms with n >= 2"}
    c_values = arm_values[control]
    n0, m0, v0 = len(c_values), statistics.fmean(c_values), statistics.variance(c_values)
    if v0 <= 0.0:
        return {"computable": False, "reason": "control has zero sample variance"}
    k = len(treatments)
    rows = []
    for arm in treatments:
        values = arm_values[arm]
        n, m, v = len(values), statistics.fmean(values), statistics.variance(values)
        if v <= 0.0:
            rows.append({"arm": arm, "n": n, "computable": False, "reason": "zero variance"})
            continue
        se = math.sqrt(v / n + v0 / n0)
        df = (v / n + v0 / n0) ** 2 / ((v / n) ** 2 / (n - 1) + (v0 / n0) ** 2 / (n0 - 1))
        crit = studentized_max_modulus_critical_value(k, df, alpha)
        est = m - m0
        half = crit * se
        rows.append(
            {
                "arm": arm,
                "n": n,
                "computable": True,
                "estimate": est,
                "se": se,
                "welch_df": df,
                "critical_value": crit,
                "ci_low": est - half,
                "ci_high": est + half,
                "excludes_zero": (est - half > 0.0) or (est + half < 0.0),
            }
        )
    return {
        "computable": True,
        "reason": None,
        "procedure": "Dunnett T3 (studentized maximum modulus), substituted for Games-Howell",
        "k": k,
        "contrasts": rows,
    }


def exploratory_pairwise(
    arm_values: Dict[str, List[float]],
    left: str,
    right: str,
    pooled_sd: Optional[float],
    df: Optional[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """
    One arm-vs-arm contrast, UNCORRECTED and declared EXPLORATORY.

    Pre-registration s4.4: comparisons that are not against the control are exploratory. The
    one that matters here is ``KDA_GCONV - KDA_NOACT``, which is the contrast that isolates
    the gate -- with ``gate_structure="depthwise"`` the depthwise pre-gate is a SiLU with a
    learnable slope, so ``KDA_GCONV - KDA_BASE`` does not isolate anything.
    """
    for arm in (left, right):
        if arm not in arm_values or len(arm_values[arm]) < 1:
            return {"computable": False, "reason": f"{arm} has no admissible cells"}
    if pooled_sd is None or df is None or df < 1:
        return {"computable": False, "reason": "no pooled variance available"}
    lv, rv = arm_values[left], arm_values[right]
    est = statistics.fmean(lv) - statistics.fmean(rv)
    se = pooled_sd * math.sqrt(1.0 / len(lv) + 1.0 / len(rv))
    crit = student_t_ppf(1.0 - alpha / 2.0, df)
    half = crit * se
    return {
        "computable": True,
        "reason": None,
        "label": f"{left} - {right}",
        "status": "EXPLORATORY, uncorrected",
        "estimate": est,
        "se": se,
        "ci_low": est - half,
        "ci_high": est + half,
        "n_left": len(lv),
        "n_right": len(rv),
        "p_uncorrected": 2.0 * student_t_sf(abs(est / se), df) if se > 0.0 else None,
    }


def ratio_to_control(
    arm_values: Dict[str, List[float]], control: str
) -> Dict[str, Dict[str, Any]]:
    """
    Ratio of each arm's mean to the control's mean -- the number a production decision turns
    on. ``None`` where either side is missing; never 1.0 as a stand-in.
    """
    out: Dict[str, Dict[str, Any]] = {}
    control_values = arm_values.get(control) or []
    control_mean = statistics.fmean(control_values) if control_values else None
    for arm, values in arm_values.items():
        stats = mean_sd_n(values)
        ratio = None
        if control_mean is not None and control_mean != 0.0 and stats["mean"] is not None:
            ratio = stats["mean"] / control_mean
        out[arm] = {
            "n": stats["n"],
            "mean": stats["mean"],
            "sd": stats["sd"],
            "values": stats["values"],
            "ratio_to_control": ratio,
            "ratio_unavailable_reason": (
                None
                if ratio is not None
                else ("control mean unavailable" if control_mean is None else "arm mean unavailable")
            ),
        }
    return out


# --------------------------------------------------------------------------------------------
# Loading cells.
# --------------------------------------------------------------------------------------------


@dataclass
class Finding:
    code: str
    severity: str  # "hard_error" | "excluded" | "warning" | "info"
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class Cell:
    cell_id: str
    source: str
    raw: Dict[str, Any]
    arm: Optional[str] = None
    replicate: Optional[int] = None
    cell_index: Optional[int] = None
    findings: List[Finding] = field(default_factory=list)
    admissible: bool = True

    def add(self, code: str, severity: str, message: str) -> None:
        self.findings.append(Finding(code, severity, message))
        if severity in ("excluded", "hard_error"):
            self.admissible = False

    def reasons(self, severity: str) -> List[str]:
        return [f.message for f in self.findings if f.severity == severity]


def metric_state(raw: Dict[str, Any], key: str) -> Tuple[str, Optional[float]]:
    """
    Read a numeric field and say EXACTLY what was found.

    Returns one of ``absent`` (the key is not there), ``null`` (present and explicitly null),
    ``nonnumeric``, ``nan`` (present and NaN or +/-inf -- a divergence, NOT a missing value),
    or ``ok``. Nothing here ever turns a missing metric into 0.0.
    """
    if key not in raw:
        return "absent", None
    value = raw[key]
    if value is None:
        return "null", None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "nonnumeric", None
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        return "nan", None
    return "ok", fv


def first_present_metric(
    raw: Dict[str, Any], keys: Sequence[str]
) -> Tuple[Optional[str], str, Optional[float]]:
    """First of ``keys`` that yields a real number, plus which key it was."""
    last_state = "absent"
    for key in keys:
        state, value = metric_state(raw, key)
        if state == "ok":
            return key, state, value
        if state != "absent":
            last_state = state
    return None, last_state, None


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Pull every top-level JSON object out of a blob of text.

    The platform reads results back out of the LOG STREAM, so a "results file" may well be a
    log with the summary embedded in it. Parsing the whole file as JSON first and falling back
    to scanning is what makes ``cat cell.log | analyse_bakeoff.py`` work.
    """
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        idx = end
    return found


def looks_like_a_cell(obj: Dict[str, Any]) -> bool:
    """A cell summary is the object that carries the arm and the endpoint."""
    return "arm" in obj and ("val_ce" in obj or "steps" in obj)


def cell_index_from_path(source: str) -> Optional[int]:
    """
    Recover the Batch array index from a ``cell-<N>/`` path component.

    Results land under a per-cell prefix keyed on ``$AWS_BATCH_JOB_ARRAY_INDEX``. That index
    is used ONLY to cross-check the arm the JSON reports against the slot it was launched in.
    It is never used to infer an arm -- an off-by-one in the launcher's arm array is exactly
    the failure this catches, and inferring from the directory would hide it.
    """
    for part in Path(source).parts:
        lowered = part.lower()
        if lowered.startswith("cell-") or lowered.startswith("cell_"):
            tail = part[5:]
            if tail.isdigit():
                return int(tail)
    return None


def load_cells_from_text(text: str, source: str) -> List[Cell]:
    cells: List[Cell] = []
    candidates: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        candidates = [parsed]
    elif isinstance(parsed, list):
        candidates = [o for o in parsed if isinstance(o, dict)]
    else:
        candidates = extract_json_objects(text)
    index = cell_index_from_path(source)
    for obj in candidates:
        if not looks_like_a_cell(obj):
            continue
        cell_id = str(obj.get("run_id") or Path(source).stem)
        if index is not None:
            cell_id = f"cell-{index}/{cell_id}"
        cells.append(Cell(cell_id=cell_id, source=source, raw=obj, cell_index=index))
    return cells


def gather_input_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*") if q.is_file() and q.suffix in
                                (".json", ".log", ".txt", ".jsonl")))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"no such file or directory: {item}")
    return paths


def load_cells(inputs: Sequence[str], stdin_text: Optional[str] = None) -> Tuple[List[Cell], List[str]]:
    """Load every cell summary reachable from ``inputs`` and/or stdin."""
    notes: List[str] = []
    cells: List[Cell] = []
    paths = gather_input_paths(inputs)
    if stdin_text is not None and stdin_text.strip():
        stripped = stdin_text.strip()
        looks_like_paths = "{" not in stripped
        if looks_like_paths:
            extra = [line.strip() for line in stripped.splitlines() if line.strip()]
            notes.append(f"read {len(extra)} path(s) from stdin")
            paths.extend(gather_input_paths(extra))
        else:
            found = load_cells_from_text(stripped, "<stdin>")
            notes.append(f"read {len(found)} cell summary object(s) from stdin")
            cells.extend(found)
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            notes.append(f"UNREADABLE {path}: {exc}")
            continue
        found = load_cells_from_text(text, str(path))
        if not found:
            notes.append(f"no cell summary found in {path}")
        cells.extend(found)
    return cells, notes


# --------------------------------------------------------------------------------------------
# The seed schedule and the parameter ledger, read from their own sources.
# --------------------------------------------------------------------------------------------


def load_seed_schedule(path: Path) -> Dict[str, Any]:
    """
    Load ``seeds.json``. If it is missing, EVERY seed-derived check reports UNCHECKED rather
    than passing. A frozen contract that cannot be found is not a satisfied contract.
    """
    if not path.exists():
        return {
            "available": False,
            "reason": f"{path} does not exist; every seed-schedule check is UNCHECKED",
            "cells": {},
            "arms": [],
            "locked": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "frozen":
        status = data.get("status")
        return {
            "available": False,
            "reason": f"{path} has status={status!r}, not 'frozen'; checks are UNCHECKED",
            "cells": {},
            "arms": [],
            "locked": {},
        }
    by_seed: Dict[Tuple[int, int], Dict[str, Any]] = {}
    arms: List[str] = []
    for entry in data.get("schedule", []):
        if 1 not in entry.get("in_run", []):
            continue
        arm = entry["arm"]
        arms.append(arm)
        for cell in entry.get("cells", []):
            by_seed[(int(cell["data_seed"]), int(cell["init_seed"]))] = {
                "arm": arm,
                "replicate": int(cell["replicate"]),
            }
    return {
        "available": True,
        "reason": None,
        "path": str(path),
        "cells": by_seed,
        "arms": arms,
        "locked": data.get("run_1", {}).get("locked_run_parameters", {}),
        "n_expected_cells": data.get("run_1", {}).get("cells"),
        "dunnett_k": data.get("run_1", {}).get("dunnett_k"),
    }


def load_arm_param_targets(path: Path) -> Dict[str, Any]:
    """
    Read ``L0_PARAM_TARGET`` and ``ARM_L0_DELTA`` out of ``core6_arms.py`` BY PARSING IT.

    Importing the module would pull in torch, which does not run on this machine and has no
    business being loaded to read two integer literals. ``ast`` reads the declared numbers
    from the file that declares them, so a change to that file changes this analysis rather
    than being silently out of date.
    """
    if not path.exists():
        return {
            "available": False,
            "reason": f"{path} does not exist; the parameter-match audit is UNCHECKED",
            "target": None,
            "deltas": {},
        }
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ints: Dict[str, int] = {}
    deltas_node: Optional[ast.Dict] = None
    for node in tree.body:
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        for target in targets:
            if not isinstance(target, ast.Name) or value is None:
                continue
            if target.id == "ARM_L0_DELTA" and isinstance(value, ast.Dict):
                deltas_node = value
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(literal, int) and not isinstance(literal, bool):
                ints[target.id] = literal
    if deltas_node is None or "L0_PARAM_TARGET" not in ints:
        return {
            "available": False,
            "reason": f"could not find L0_PARAM_TARGET and ARM_L0_DELTA in {path}",
            "target": None,
            "deltas": {},
        }
    deltas: Dict[str, int] = {}
    for key_node, value_node in zip(deltas_node.keys, deltas_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        name = key_node.value
        if isinstance(value_node, ast.Name):
            if value_node.id in ints:
                deltas[name] = ints[value_node.id]
            continue
        try:
            literal = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            continue
        if isinstance(literal, int) and not isinstance(literal, bool):
            deltas[name] = literal
    return {
        "available": True,
        "reason": None,
        "path": str(path),
        "target": ints["L0_PARAM_TARGET"],
        "deltas": deltas,
    }


def expected_parameters(arm: str, ledger: Dict[str, Any]) -> Optional[int]:
    """``L0_PARAM_TARGET + ARM_L0_DELTA[arm]``, or None if the ledger cannot say."""
    if not ledger.get("available"):
        return None
    delta = ledger["deltas"].get(arm)
    if delta is None:
        return None
    return int(ledger["target"]) + int(delta)


# --------------------------------------------------------------------------------------------
# Admissibility. STEP 0, before any pooling.
# --------------------------------------------------------------------------------------------


def canonicalise_arm(name: Optional[str], alias_map: Dict[str, str]) -> Optional[str]:
    """Map a placeholder arm name (``kda-paper``) onto its ARMS key (``KDA_BASE``)."""
    if name is None:
        return None
    if name in alias_map.values():
        return name
    return alias_map.get(name, name)


PLACEHOLDER_TO_ARM = {
    "kda-paper": "KDA_BASE",
    "kda-noact": "KDA_NOACT",
    "kda-gated-conv": "KDA_GCONV",
    "gdn2": "GDN2",
    "kda-hh-r1": "KDA_R1",
    "kda-hh-r2": "KDA_R2",
    "kda-k3": "KDA_K3",
}


def check_admissibility(
    cell: Cell,
    *,
    schedule: Dict[str, Any],
    ledger: Dict[str, Any],
    expected_steps: Optional[int],
    val_ce_band: Tuple[float, float] = VAL_CE_BAND,
    first_loss_band: Tuple[float, float] = FIRST_LOSS_BAND,
) -> None:
    """
    Run every admissibility check on one cell, recording a Finding for each outcome.

    Pre-registration s4.1: admissibility is step 0 and it runs BEFORE pooling. Every exclusion
    lands in ``cell.findings`` with a reason, and the caller declares them with a count.
    """
    raw = cell.raw

    cell.arm = canonicalise_arm(raw.get("arm"), PLACEHOLDER_TO_ARM)
    if cell.arm is None:
        cell.add("no_arm", "excluded", "the summary carries no `arm` field")

    # --- the launcher's slot vs what the run says it was ---
    if cell.cell_index is not None:
        slot_arm = CELL_INDEX_TO_ARM.get(cell.cell_index)
        if slot_arm is None:
            cell.add(
                "cell_index_out_of_range",
                "hard_error",
                f"the path says cell-{cell.cell_index}, which is outside the 18-cell arm-major "
                "layout in .edullm/run-bakeoff.yaml",
            )
        elif cell.arm is not None and slot_arm != cell.arm:
            cell.add(
                "cell_index_arm_disagreement",
                "hard_error",
                f"the path says cell-{cell.cell_index}, which run-bakeoff.yaml's arm-major "
                f"layout assigns to {slot_arm}, but the summary reports arm={cell.arm}. The "
                "launcher's arm array and the run's own record disagree; one of them is "
                "wrong and this is not reconcilable from here.",
            )

    # --- parameter-match audit: a HARD ERROR, because it invalidates the comparison ---
    state, params = metric_state(raw, "parameters")
    if state != "ok":
        cell.add(
            "parameters_missing",
            "excluded",
            f"`parameters` is {state}; the parameter-match audit cannot run and an arm whose "
            "size was never checked is not a declared arm",
        )
    elif cell.arm is not None:
        want = expected_parameters(cell.arm, ledger)
        if want is None:
            cell.add(
                "parameters_unchecked",
                "warning",
                f"no ARM_L0_DELTA entry for {cell.arm}; parameter count {int(params):,} is "
                "UNCHECKED, not confirmed",
            )
        elif int(params) != want:
            cell.add(
                "parameter_mismatch",
                "hard_error",
                f"reported {int(params):,} parameters, ARM_L0_DELTA declares {want:,} "
                f"(difference {int(params) - want:+,}). THE ARM THAT RAN IS NOT THE ARM THAT "
                "WAS DECLARED; this invalidates the arm's comparison.",
            )

    # --- seed schedule: which (arm, replicate) is this, and is it on the frozen schedule ---
    ds_state, data_seed = metric_state(raw, "data_seed")
    is_state, init_seed = metric_state(raw, "init_seed")
    if not schedule.get("available"):
        cell.add(
            "seed_schedule_unchecked",
            "warning",
            f"seed schedule UNCHECKED: {schedule.get('reason')}",
        )
    elif ds_state != "ok" or is_state != "ok":
        cell.add(
            "seeds_missing",
            "excluded",
            f"data_seed is {ds_state} and init_seed is {is_state}; a cell that cannot say "
            "which data order and which initialisation produced it cannot be placed on the "
            "frozen schedule",
        )
    else:
        key = (int(data_seed), int(init_seed))
        entry = schedule["cells"].get(key)
        if entry is None:
            cell.add(
                "seed_pair_off_schedule",
                "hard_error",
                f"(data_seed={int(data_seed)}, init_seed={int(init_seed)}) is not on the "
                "frozen run-1 schedule in seeds.json",
            )
        else:
            cell.replicate = entry["replicate"]
            if cell.arm is not None and entry["arm"] != cell.arm:
                cell.add(
                    "seed_pair_wrong_arm",
                    "hard_error",
                    f"the summary says arm={cell.arm} but seeds.json binds "
                    f"(data_seed={int(data_seed)}, init_seed={int(init_seed)}) to "
                    f"{entry['arm']} replicate {entry['replicate']}",
                )

    # --- the primary endpoint must be a real number in a plausible band ---
    ce_state, val_ce = metric_state(raw, "val_ce")
    if ce_state == "nan":
        cell.add("val_ce_nan", "excluded", "`val_ce` is NaN or infinite: the run diverged")
    elif ce_state != "ok":
        cell.add(
            "val_ce_missing",
            "excluded",
            f"`val_ce` is {ce_state}; there is no primary endpoint to analyse",
        )
    elif not val_ce_band[0] <= val_ce <= val_ce_band[1]:
        cell.add(
            "val_ce_out_of_band",
            "excluded",
            f"`val_ce` = {val_ce:.4f} nats is outside the declared plausibility band "
            f"[{val_ce_band[0]}, {val_ce_band[1]}] (ln(vocab) = {LN_VOCAB:.4f}); the run did "
            "not converge, or the held-out denominator is wrong",
        )

    # --- init sanity: first_loss ~ ln(vocab) ---
    fl_state, first_loss = metric_state(raw, "first_loss")
    if fl_state == "absent":
        cell.add(
            "first_loss_absent",
            "info",
            "no `first_loss` key; the init sanity check is UNCHECKED for this cell",
        )
    elif fl_state != "ok":
        cell.add(
            "first_loss_unusable",
            "warning",
            f"`first_loss` is {fl_state}; the init sanity check is UNCHECKED, not passed",
        )
    elif not first_loss_band[0] <= first_loss <= first_loss_band[1]:
        cell.add(
            "first_loss_out_of_band",
            "excluded",
            f"`first_loss` = {first_loss:.4f} is outside ln(vocab) +/- 0.5 = "
            f"[{first_loss_band[0]}, {first_loss_band[1]}]; this run did not start from a "
            "uniform distribution over the vocabulary",
        )

    # --- the held-out denominator: exact, no tolerance ---
    present_state, present = metric_state(raw, "val_tokens_present")
    declared_state, declared = metric_state(raw, "val_tokens_declared")
    if present_state == "ok" and declared_state == "ok":
        if int(present) != int(declared):
            cell.add(
                "val_tokens_mismatch",
                "excluded",
                f"val_tokens_present = {int(present):,} but val_tokens_declared = "
                f"{int(declared):,} ({int(present) - int(declared):+,}); the CE was not "
                "computed over the whole declared partition",
            )
    elif ce_state == "ok":
        cell.add(
            "val_denominator_unchecked",
            "warning",
            f"val_tokens_present is {present_state} and val_tokens_declared is "
            f"{declared_state}; the denominator that makes this CE comparable is UNCHECKED",
        )

    # --- did it reach its declared step count ---
    st_state, steps = metric_state(raw, "steps")
    if expected_steps is None:
        cell.add(
            "steps_unchecked",
            "warning",
            "no declared step count available; the run-length check is UNCHECKED",
        )
    elif st_state != "ok":
        cell.add(
            "steps_missing",
            "excluded",
            f"`steps` is {st_state}; a run that cannot say how far it got cannot be shown to "
            "have finished",
        )
    elif int(steps) != int(expected_steps):
        cell.add(
            "steps_mismatch",
            "excluded",
            f"reached step {int(steps):,}, declared {int(expected_steps):,}; a short cell "
            "consumed a prefix of the data stream, not the same stream",
        )

    # --- co-primary quality: is the throughput window long enough to mean anything ---
    # A WARNING, NOT AN EXCLUSION. It says nothing about the CE endpoint, and excluding a CE
    # observation because a speed measurement was short would throw away the primary endpoint
    # to protect a co-primary one.
    ss_state, steady_steps = metric_state(raw, "steady_state_steps")
    if ss_state == "ok" and steady_steps < MIN_STEADY_STEPS:
        cell.add(
            "steady_window_short",
            "warning",
            f"steady-state throughput was computed over only {int(steady_steps)} step(s) "
            f"(fewer than {MIN_STEADY_STEPS}); this project measured the steady figure itself "
            "moving 1.5x between a 20-step and a 200-step run, so this cell's THROUGHPUT is "
            "not comparable with a full-length cell's. Its CE is unaffected.",
        )

    # --- co-primary quality: which memory read is this ---
    source = raw.get("peak_memory_source")
    if source == MEMORY_SOURCE_LOWER_BOUND:
        cell.add(
            "peak_memory_lower_bound",
            "warning",
            "peak_memory_source = 'final_step_only': this is the LAST STEP'S peak, not the "
            "whole run's, because the GPU monitor resets the counters every step. It is a "
            "LOWER BOUND wearing the name of a peak and is not comparable with a "
            "'per_step_running_max' figure from another arm.",
        )
    elif source == MEMORY_SOURCE_UNAVAILABLE:
        cell.add(
            "peak_memory_unavailable",
            "warning",
            "peak_memory_source = 'unavailable': no CUDA, so there is no memory measurement "
            "for this cell. Absent, not zero.",
        )


def flag_gross_outliers(cells: Sequence[Cell], threshold: float = OUTLIER_WARN_NATS) -> List[str]:
    """
    WARN about a cell that sits absurdly far from the rest. Does not exclude it.

    At an expected sigma of ~0.01 nats a half-nat gap is ~50 sigma. But the pre-registration
    does not pre-commit an outlier rule, and an exclusion criterion chosen after seeing the
    data is the failure a pre-registration exists to prevent. So this is loud and advisory.
    """
    values = []
    for cell in cells:
        if not cell.admissible:
            continue
        state, value = metric_state(cell.raw, "val_ce")
        if state == "ok":
            values.append((cell, value))
    if len(values) < 3:
        return []
    median = statistics.median(v for _, v in values)
    return [
        f"{cell.cell_id} ({cell.arm}): val_ce = {value:.4f} is {abs(value - median):.4f} nats "
        f"from the median of the admissible cells ({median:.4f}). NOT excluded -- no outlier "
        "rule was pre-registered -- but at an expected sigma of ~0.01 nats this is far too "
        "large to be sampling noise and should be looked at before the number is believed."
        for cell, value in values
        if abs(value - median) > threshold
    ]


def deduplicate(cells: Sequence[Cell]) -> Tuple[List[Cell], List[str]]:
    """
    One result per (arm, data_seed, init_seed). A duplicate would inflate n silently.

    The later source wins, and every drop is named.
    """
    seen: Dict[Tuple[Any, Any, Any], Cell] = {}
    unkeyed: List[Cell] = []
    notes: List[str] = []
    for cell in cells:
        _, ds = metric_state(cell.raw, "data_seed")
        _, iseed = metric_state(cell.raw, "init_seed")
        if ds is None or iseed is None or cell.arm is None:
            unkeyed.append(cell)
            continue
        key = (cell.arm, int(ds), int(iseed))
        if key in seen:
            notes.append(
                f"DUPLICATE for {key[0]} (data_seed={key[1]}, init_seed={key[2]}): keeping "
                f"{cell.source}, dropping {seen[key].source}"
            )
        seen[key] = cell
    return list(seen.values()) + unkeyed, notes


# --------------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------------


def collect_metric(
    cells: Sequence[Cell], keys: Sequence[str]
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    Gather one metric per arm, recording which key supplied it and which cells could not.

    A cell with no value is listed in ``missing`` -- it never becomes a 0.0 and it never
    inherits a neighbour's number.
    """
    by_arm: Dict[str, List[float]] = {}
    keys_used: Dict[str, int] = {}
    missing: List[Dict[str, str]] = []
    for cell in cells:
        if cell.arm is None:
            continue
        key, state, value = first_present_metric(cell.raw, keys)
        if value is None:
            missing.append({"cell": cell.cell_id, "arm": cell.arm, "state": state})
            by_arm.setdefault(cell.arm, [])
            continue
        keys_used[key] = keys_used.get(key, 0) + 1
        by_arm.setdefault(cell.arm, []).append(value)
    return by_arm, {"keys_used": keys_used, "missing": missing}


def _fmt(value: Optional[float], spec: str = ".4f", dash: str = "--") -> str:
    if value is None:
        return dash
    return format(value, spec)


def build_recommendation(
    *,
    control: str,
    arm_order: Sequence[str],
    ce_by_arm: Dict[str, List[float]],
    contrasts: Dict[str, Any],
    mde: Optional[float],
    budget: Dict[str, Any],
    throughput: Dict[str, Dict[str, Any]],
    throughput_is_steady: bool,
    throughput_available: bool,
    memory: Dict[str, Dict[str, Any]],
    memory_available: bool,
) -> Dict[str, Any]:
    """
    Synthesise the recommendation. This is the point of the whole script.

    The rule, in order:

    1. An arm whose Dunnett CI excludes zero AND whose |estimate| clears the MDE is CE-resolved.
       Resolved-better arms are preferred; resolved-worse arms are struck out whatever their
       speed.
    2. If nothing is CE-resolved, THE CE RANKING IS NOT RESOLVED AT n = 3 and the decision
       falls back to throughput and memory -- and SAYS SO. It does not quietly rank on a
       difference it cannot see.
    3. If throughput is unavailable too, the recommendation is the control, because "no
       evidence to move" is a reason to stay put, not a reason to pick the smallest number.
    """
    resolved_better: List[Dict[str, Any]] = []
    resolved_worse: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    rows = contrasts.get("contrasts", []) if contrasts.get("computable") else []
    for row in rows:
        clears = None if mde is None else abs(row["estimate"]) > mde
        entry = dict(row)
        entry["clears_mde"] = clears
        if row["excludes_zero"] and clears:
            (resolved_better if row["estimate"] < 0 else resolved_worse).append(entry)
        else:
            unresolved.append(row["arm"])

    ce_resolved = bool(resolved_better or resolved_worse)
    struck = {r["arm"] for r in resolved_worse}
    eligible = [a for a in arm_order if a in ce_by_arm and a not in struck]

    basis: str
    choice: Optional[str]
    rationale: List[str] = []

    if not contrasts.get("computable"):
        basis = "no CE contrasts were computable"
        rationale.append(
            f"CE contrasts could not be computed: {contrasts.get('reason')}. The CE endpoint "
            "contributes nothing to this recommendation."
        )
    if ce_resolved:
        rationale.append(
            "At least one arm's CE difference from the control clears the MDE and its "
            "Dunnett-adjusted CI excludes zero, so CE is a usable axis here."
        )
    elif contrasts.get("computable"):
        rationale.append(
            "NO arm's CE difference from the control clears the MDE with a Dunnett-adjusted "
            "CI that excludes zero. THE CE RANKING IS NOT RESOLVED AT n = 3. The arms are not "
            "ranked on CE below, and the recommendation falls back to throughput and memory. "
            "That fallback is a choice forced by the design's power, and it is stated rather "
            "than hidden behind an ordering of point estimates."
        )

    def throughput_of(arm: str) -> Optional[float]:
        entry = throughput.get(arm)
        return None if entry is None else entry.get("mean")

    def memory_of(arm: str) -> Optional[float]:
        entry = memory.get(arm)
        return None if entry is None else entry.get("mean")

    if resolved_better:
        best_ce = min(resolved_better, key=lambda r: r["estimate"])
        choice = best_ce["arm"]
        basis = "CE (resolved): the arm with the largest CE improvement that clears the MDE"
        rationale.append(
            f"{choice} improves CE by {abs(best_ce['estimate']):.4f} nats "
            f"(Dunnett CI [{best_ce['ci_low']:.4f}, {best_ce['ci_high']:.4f}]), which clears "
            f"the MDE of {_fmt(mde)} nats."
        )
        tput = throughput_of(choice)
        base_tput = throughput_of(control)
        if tput is not None and base_tput:
            rationale.append(
                f"Its steady throughput is {tput / base_tput:.3f}x the control's. Check that "
                "this is a price worth paying before committing."
            )
    elif throughput_available and any(throughput_of(a) is not None for a in eligible):
        ranked = [(a, throughput_of(a)) for a in eligible if throughput_of(a) is not None]
        ranked.sort(key=lambda p: p[1], reverse=True)
        choice = ranked[0][0]
        basis = (
            "throughput and memory (CE unresolved): the fastest arm not shown to be worse on CE"
        )
        base_tput = throughput_of(control)
        for arm, value in ranked:
            ratio = None if not base_tput else value / base_tput
            mem = memory_of(arm)
            base_mem = memory_of(control)
            mem_ratio = None if not base_mem or mem is None else mem / base_mem
            rationale.append(
                f"{arm}: throughput {value:,.0f} tok/s ({_fmt(ratio, '.3f')}x control), "
                f"peak memory {_fmt(mem, '.2f')} GiB ({_fmt(mem_ratio, '.3f')}x control)."
            )
        if not throughput_is_steady:
            rationale.append(
                "WARNING: the throughput figure above is the WHOLE-RUN wall-clock number, not "
                "the steady-state one. It charges process start, dataset open, FSDP wrap and "
                "the first-step compile against the hardware. This project measured whole-run "
                "wall clock reading 3.1x LOW on a short probe, and it penalises bigger shapes "
                "hardest -- which is exactly the direction that would bias this decision. Do "
                "not commit on this number; get tps_device_avg / tps_total_avg."
            )
    else:
        choice = control if control in ce_by_arm else None
        basis = "default to the control: no endpoint resolved a difference"
        rationale.append(
            "Neither CE nor throughput resolved a difference between the arms, so there is no "
            "measured reason to move off the control. That is a statement about this study's "
            "power, not about the arms."
        )

    if struck:
        rationale.append(
            "Struck from consideration on CE (significantly WORSE than the control, clearing "
            f"the MDE): {', '.join(sorted(struck))}. Speed does not buy this back."
        )

    tpp = budget.get("realised_tpp")
    tokens = budget.get("realised_tokens_per_cell")
    n_params = budget.get("parameters")
    if tpp is not None:
        tpp_line = (
            f"TPP IS {tpp:.1f} -- {tokens:,} tokens per cell for a {n_params:,}-parameter "
            "model, computed from what the cells actually report, not from the plan. That is "
            "BELOW the academic literature cluster of 3-20, and further below this project's "
            "own 1B flagship at TPP 27-44."
        )
        if budget.get("steps_differ_from_plan"):
            tpp_line += (
                f" NOTE the budget moved after pre-registration: seeds.json planned "
                f"{budget['steps_planned_in_seeds_json']:,} steps, the cells ran "
                f"{budget['steps_used_for_admissibility']:,} (planned TPP was {TPP}). The cut "
                "makes every CE magnitude below MORE inflated, not less."
            )
    else:
        tpp_line = (
            "TPP COULD NOT BE COMPUTED (no cell reported both tokens_trained and parameters). "
            f"The plan was {TPP}, which is already below the literature cluster of 3-20."
        )
    caveats = [
        tpp_line,
        "ARCHITECTURE EFFECTS MEASURED AT LOW TOKEN BUDGETS SYSTEMATICALLY OVERSTATE. Measured "
        f"in-tree: GDN's edge over baseline shrank {GDN_EDGE_AT_1B} nats @1B -> "
        f"{GDN_EDGE_AT_15B} @15B, roughly halved for a 15x budget increase. So the CE result "
        "here is DIRECTIONAL WITH INFLATED MAGNITUDES: treat every CE number below as an "
        "UPPER BOUND on the production effect, never as the effect itself.",
        "THROUGHPUT AND PEAK MEMORY ARE BUDGET-INDEPENDENT AND FULLY VALID. They do not "
        "inflate at a short budget the way CE does. At this TPP they are very likely to be "
        "what carries the recommendation, and they carry it soundly.",
        "A NULL IS A BOUND, NOT EQUIVALENCE. A non-significant CE contrast licenses exactly one "
        "claim: the difference is smaller than the Dunnett-adjusted CI half-width. Equivalence "
        "needs a pre-declared margin and a TOST; no margin was declared, so no equivalence "
        "claim may be made from this design.",
        "Say it that way when this is presented: the speed and memory conclusions are STRONG, "
        "the CE conclusions are INDICATIVE.",
        "A 1.0B-token run structurally CANNOT see any effect that only emerges late in "
        "training, or any long-horizon recall advantage. That is a limitation, not a null.",
        f"The literature CE gap between these mixer families is ~{LITERATURE_CE_GAP[0]}-"
        f"{LITERATURE_CE_GAP[1]} nats, which may sit below this design's MDE. The "
        "pre-registration committed to that as an expected outcome before the run.",
    ]
    if not memory_available:
        caveats.append(
            "PEAK MEMORY IS UNAVAILABLE for some or all cells and is therefore not part of "
            "this recommendation. Note that summarise() writes peak_memory_gib = 0.0 when "
            "torch.cuda is unavailable, so a 0.0 is a CPU run, not a measurement; this script "
            "treats it as missing."
        )

    return {
        "choice": choice,
        "basis": basis,
        "ce_resolved": ce_resolved,
        "mde_nats": mde,
        "resolved_better": resolved_better,
        "resolved_worse": resolved_worse,
        "unresolved_arms": unresolved,
        "rationale": rationale,
        "caveats": caveats,
    }


def analyse(
    cells: Sequence[Cell],
    *,
    schedule: Dict[str, Any],
    ledger: Dict[str, Any],
    control: str = DEFAULT_CONTROL,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    load_notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the whole pre-registered analysis over already-loaded cells."""
    locked = schedule.get("locked") or {}

    # THE STEP BUDGET IS READ FROM THE CELLS, NOT FROM seeds.json.
    #
    # The budget is a submit-time decision and it HAS moved under this analysis once already
    # (1,907 steps -> 1,144, i.e. 999,817,216 -> 599,785,472 tokens/cell, forced by A100
    # capacity). seeds.json records what was frozen at planning time, so pinning to it would
    # have excluded all 18 cells as "short runs" and reported a clean, confident, empty study.
    #
    # What actually matters for comparability is not which budget was used but that EVERY CELL
    # USED THE SAME ONE -- a cell on a different budget consumed a different prefix of the
    # stream and is not paired. So: take the modal step count as the realised budget, hold
    # every cell to it, and report the disagreement loudly if there is one.
    observed_steps: Dict[int, int] = {}
    for cell in cells:
        state, value = metric_state(cell.raw, "steps")
        if state == "ok":
            observed_steps[int(value)] = observed_steps.get(int(value), 0) + 1
    if observed_steps:
        expected_steps: Optional[int] = max(observed_steps.items(), key=lambda kv: kv[1])[0]
        steps_basis = (
            f"the modal `steps` across the {sum(observed_steps.values())} cell(s) present"
        )
    else:
        expected_steps = locked.get("steps")
        steps_basis = "seeds.json's locked step count (no cell reported a step count)"
    budget_split = len(observed_steps) > 1
    planned_steps = locked.get("steps")
    budget_differs_from_plan = (
        planned_steps is not None
        and expected_steps is not None
        and int(planned_steps) != int(expected_steps)
    )

    for cell in cells:
        check_admissibility(
            cell, schedule=schedule, ledger=ledger, expected_steps=expected_steps
        )
    deduped, dedup_notes = deduplicate(cells)

    admissible = [c for c in deduped if c.admissible]
    excluded = [c for c in deduped if not c.admissible]
    hard_errors = [c for c in deduped if any(f.severity == "hard_error" for f in c.findings)]

    arm_order = list(schedule.get("arms") or [])
    for cell in deduped:
        if cell.arm is not None and cell.arm not in arm_order:
            arm_order.append(cell.arm)

    # --- coverage: what is here and what is not ---
    expected_cells = schedule.get("cells") or {}
    observed_keys = set()
    for cell in deduped:
        _, ds = metric_state(cell.raw, "data_seed")
        _, iseed = metric_state(cell.raw, "init_seed")
        if ds is not None and iseed is not None:
            observed_keys.add((int(ds), int(iseed)))
    missing_cells = [
        {"arm": v["arm"], "replicate": v["replicate"], "data_seed": k[0], "init_seed": k[1]}
        for k, v in sorted(expected_cells.items(), key=lambda kv: (kv[1]["arm"], kv[1]["replicate"]))
        if k not in observed_keys
    ]

    # --- the primary endpoint ---
    ce_by_arm: Dict[str, List[float]] = {a: [] for a in arm_order}
    ce_cells: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arm_order}
    for cell in admissible:
        state, value = metric_state(cell.raw, "val_ce")
        if cell.arm is None or state != "ok":
            continue
        ce_by_arm.setdefault(cell.arm, []).append(value)
        ce_cells.setdefault(cell.arm, []).append(
            {"cell": cell.cell_id, "replicate": cell.replicate, "val_ce": value}
        )
    ce_by_arm = {a: v for a, v in ce_by_arm.items() if v}

    arms_without_variance = [a for a, v in ce_by_arm.items() if len(v) < 2]
    arms_absent = [a for a in arm_order if a not in ce_by_arm]

    # PER-ARM n IS THE FIRST THING THE READER NEEDS AT 10:00. Arms complete at different times
    # and the arm-major fan-out means a partial run has whole arms at n = 0 while others are
    # at n = 3. Which arms can carry a variance estimate is the question that decides what the
    # rest of the report is even allowed to say.
    arm_census = []
    for arm in arm_order:
        n = len(ce_by_arm.get(arm, []))
        n_found = sum(1 for c in deduped if c.arm == arm)
        n_excluded = sum(1 for c in excluded if c.arm == arm)
        if n == 0:
            status = "NO DATA -- absent from every table below"
        elif n == 1:
            status = "n=1: NO VARIANCE ESTIMATE. One number, not a mean. No sd, no df, no CI."
        elif n == 2:
            status = "n=2: reduced power; contributes 1 df to the pooled sigma"
        else:
            status = "complete"
        arm_census.append(
            {
                "arm": arm,
                "n_admissible": n,
                "n_found": n_found,
                "n_excluded": n_excluded,
                "n_expected": 3,
                "contributes_variance": n >= 2,
                "df_contributed": max(0, n - 1),
                "status": status,
            }
        )

    per_arm = {a: mean_sd_n(v) for a, v in ce_by_arm.items()}
    pooled = pooled_variance(list(ce_by_arm.values()))
    anova = one_way_anova(list(ce_by_arm.values()))
    levene = levene_median(list(ce_by_arm.values()))
    bart = bartlett(list(ce_by_arm.values()))

    homogeneity_rejected = bool(
        levene.get("computable") and levene.get("p") is not None and levene["p"] < alpha
    )
    contrasts = dunnett_contrasts(ce_by_arm, control, alpha=alpha)
    fallback = None
    if homogeneity_rejected:
        fallback = {
            "welch_anova": welch_anova(list(ce_by_arm.values())),
            "t3_contrasts": welch_t3_contrasts(ce_by_arm, control, alpha=alpha),
            "per_arm_sd": {a: per_arm[a]["sd"] for a in per_arm},
        }

    # --- sigma and MDE ---
    sigma_hat = pooled["sd"]
    sigma_interval = None
    if sigma_hat is not None and pooled["df"] >= 1:
        lo, hi = sigma_chi2_interval(float(pooled["df"]), alpha)
        sigma_interval = {
            "multiplier_low": lo,
            "multiplier_high": hi,
            "sigma_low": sigma_hat * lo,
            "sigma_high": sigma_hat * hi,
            "df": pooled["df"],
        }

    mde_measured = None
    mde_table: List[Dict[str, Any]] = []
    if contrasts.get("computable"):
        crit = contrasts["critical_value"]
        df = contrasts["df"]
        n_ctrl = len(ce_by_arm[control])
        n_typ = max((len(v) for a, v in ce_by_arm.items() if a != control), default=n_ctrl)
        for label, sigma in (
            ("MEASURED (pooled, this run)", sigma_hat),
            ("literature optimistic", SIGMA_OPTIMISTIC),
            ("literature pessimistic", SIGMA_PESSIMISTIC),
        ):
            if sigma is None:
                mde_table.append({"label": label, "sigma": None, "mde": None})
                continue
            value = mde_for_power(sigma, n_typ, n_ctrl, crit, df, power)
            mde_table.append(
                {
                    "label": label,
                    "sigma": sigma,
                    "se_of_difference": contrast_standard_error(sigma, n_typ, n_ctrl),
                    "mde": value,
                }
            )
            if label.startswith("MEASURED"):
                mde_measured = value

    # --- the gate contrast, exploratory ---
    gate = exploratory_pairwise(
        ce_by_arm, "KDA_GCONV", "KDA_NOACT", pooled["sd"], float(pooled["df"]), alpha=alpha
    )

    # --- co-primary endpoints ---
    tput_total_by_arm, tput_total_meta = collect_metric(admissible, THROUGHPUT_TOTAL_STEADY_KEYS)
    tput_dev_by_arm, tput_dev_meta = collect_metric(admissible, THROUGHPUT_DEVICE_STEADY_KEYS)
    tput_whole_by_arm, tput_whole_meta = collect_metric(admissible, THROUGHPUT_WHOLE_RUN_KEYS)

    # Peak memory. `peak_memory_source` says WHICH read each figure is and the three are not
    # interchangeable: only `per_step_running_max` is a whole-run peak. A 0.0 or a null is an
    # absent measurement, never a claim that the arm is free.
    #
    # GATED ON ``peak_memory_source``. `peak_memory_gib` CHANGED MEANING WHILE KEEPING ITS
    # NAME: it is now a true per-step running max. `final_step_only` is the naive post-fit
    # read, which is the LAST STEP'S peak because the GPU monitor resets the counters every
    # step -- a lower bound that must never size hardware. Only `per_step_running_max` feeds
    # the headline table; the lower bounds are kept in a separate, labelled table.
    mem_by_arm: Dict[str, List[float]] = {}
    mem_lower_bound_by_arm: Dict[str, List[float]] = {}
    mem_missing: List[Dict[str, str]] = []
    mem_sources: Dict[str, int] = {}
    for cell in admissible:
        if cell.arm is None:
            continue
        _, state, value = first_present_metric(cell.raw, PEAK_MEMORY_KEYS)
        mem_by_arm.setdefault(cell.arm, [])
        src = cell.raw.get("peak_memory_source")
        if isinstance(src, str):
            mem_sources[src] = mem_sources.get(src, 0) + 1
        if value is None:
            mem_missing.append(
                {"cell": cell.cell_id, "arm": cell.arm, "state": state, "source": str(src)}
            )
            continue
        if value <= 0.0:
            mem_missing.append(
                {
                    "cell": cell.cell_id,
                    "arm": cell.arm,
                    "state": "zero_is_not_a_measurement",
                    "source": str(src),
                }
            )
            continue
        if src == MEMORY_SOURCE_TRUSTED:
            mem_by_arm[cell.arm].append(value)
        elif src == MEMORY_SOURCE_LOWER_BOUND:
            mem_lower_bound_by_arm.setdefault(cell.arm, []).append(value)
            mem_missing.append(
                {
                    "cell": cell.cell_id,
                    "arm": cell.arm,
                    "state": "final_step_only_is_a_lower_bound_not_a_peak",
                    "source": str(src),
                }
            )
        else:
            # An unrecognised source. Absent, not assumed good -- a new source name added
            # upstream must be reviewed before its numbers size a card.
            mem_missing.append(
                {
                    "cell": cell.cell_id,
                    "arm": cell.arm,
                    "state": f"unrecognised peak_memory_source {src!r}",
                    "source": str(src),
                }
            )
    mem_sources_mixed = len(mem_sources) > 1

    # UPSTREAM'S AVERAGE VS OURS, PER CELL. `tps_device_avg` is SpeedMonitorCallback's own
    # measurement over a window that starts after step 1; `throughput_tok_s_steady_per_device`
    # starts after 50. Upstream's should therefore sit slightly BELOW ours. A large gap means
    # compilation or allocator growth leaked into upstream's average -- which is the exact
    # failure the 50-step cutoff exists for, so the gap is surfaced rather than the column
    # being dropped.
    tps_gap: List[Dict[str, Any]] = []
    for cell in admissible:
        ours_state, ours = metric_state(cell.raw, "throughput_tok_s_steady_per_device")
        theirs_state, theirs = metric_state(cell.raw, "tps_device_avg")
        if ours_state != "ok" or theirs_state != "ok" or ours <= 0.0:
            continue
        ratio = theirs / ours
        tps_gap.append(
            {
                "cell": cell.cell_id,
                "arm": cell.arm,
                "tps_device_avg": theirs,
                "throughput_tok_s_steady_per_device": ours,
                "ratio_upstream_over_ours": ratio,
                # Upstream excludes only step 1, so it should be at or just below ours. More
                # than 15% below means the compile leaked into its window; ABOVE ours at all is
                # backwards and means one of the two is not measuring what it says.
                "suspicious": ratio < 0.85 or ratio > 1.02,
            }
        )
    tps_gap_suspicious = [g for g in tps_gap if g["suspicious"]]

    steady_total_available = bool(any(v for v in tput_total_by_arm.values()))
    steady_device_available = bool(any(v for v in tput_dev_by_arm.values()))
    whole_available = bool(any(v for v in tput_whole_by_arm.values()))
    throughput_is_steady = steady_total_available or steady_device_available

    if steady_total_available:
        headline_tput = ratio_to_control(tput_total_by_arm, control)
        headline_keys = tput_total_meta["keys_used"]
        headline_source = "steady-state, total across devices"
    elif steady_device_available:
        headline_tput = ratio_to_control(tput_dev_by_arm, control)
        headline_keys = tput_dev_meta["keys_used"]
        headline_source = "steady-state, per device"
    elif whole_available:
        headline_tput = ratio_to_control(tput_whole_by_arm, control)
        headline_keys = tput_whole_meta["keys_used"]
        headline_source = "WHOLE-RUN WALL CLOCK -- startup-contaminated, see the warning"
    else:
        headline_tput = {}
        headline_keys = {}
        headline_source = "unavailable"
    if headline_keys:
        headline_source += " (from " + ", ".join(f"`{k}` x{v}" for k, v in headline_keys.items()) + ")"
    # A ratio whose numerator came from two different keys across arms is comparing two
    # definitions, so say so rather than printing a clean-looking multiple.
    headline_mixed_keys = len(headline_keys) > 1

    memory_stats = ratio_to_control({a: v for a, v in mem_by_arm.items() if v}, control)

    # --- what did the trainer emit that we did not consume ---
    unconsumed: Dict[str, int] = {}
    for cell in deduped:
        for key in cell.raw:
            if key not in CONSUMED_KEYS:
                unconsumed[key] = unconsumed.get(key, 0) + 1
    trainer_reported = {
        cell.cell_id: cell.raw["production_decision"]
        for cell in deduped
        if isinstance(cell.raw.get("production_decision"), (dict, list, str))
    }

    # --- the realised token budget, and the TPP that follows from it ---
    tokens_seen: Dict[int, int] = {}
    for cell in deduped:
        state, value = metric_state(cell.raw, "tokens_trained")
        if state == "ok":
            tokens_seen[int(value)] = tokens_seen.get(int(value), 0) + 1
    realised_tokens = (
        max(tokens_seen.items(), key=lambda kv: kv[1])[0] if tokens_seen else None
    )
    params_seen = [
        int(v)
        for v in (metric_state(c.raw, "parameters")[1] for c in deduped)
        if v is not None
    ]
    n_params = max(set(params_seen), key=params_seen.count) if params_seen else None
    realised_tpp = (
        realised_tokens / n_params if realised_tokens and n_params else None
    )
    budget = {
        "steps_used_for_admissibility": expected_steps,
        "steps_basis": steps_basis,
        "steps_observed": observed_steps,
        "steps_disagree_across_cells": budget_split,
        "steps_planned_in_seeds_json": planned_steps,
        "steps_differ_from_plan": budget_differs_from_plan,
        "realised_tokens_per_cell": realised_tokens,
        "tokens_observed": tokens_seen,
        "parameters": n_params,
        "realised_tpp": realised_tpp,
        "planned_tpp": TPP,
    }

    recommendation = build_recommendation(
        control=control,
        arm_order=arm_order,
        ce_by_arm=ce_by_arm,
        contrasts=contrasts,
        mde=mde_measured,
        budget=budget,
        throughput=headline_tput,
        throughput_is_steady=throughput_is_steady,
        throughput_available=bool(headline_tput),
        memory=memory_stats,
        memory_available=bool(memory_stats),
    )

    deviations = [
        "The unequal-variance fallback uses Dunnett's T3 (studentized maximum modulus) rather "
        "than Games-Howell (studentized range). Games-Howell is the ALL-PAIRS procedure; these "
        "comparisons are k arms against ONE control, whose unequal-variance analogue is T3. "
        "The two differ only in the critical-value distribution, and T3 is the correct family "
        "here. This is the only substantive departure from the pre-registered plan.",
        "The pre-registration's saturation rule (s4.2 -- never pool sigma over cells where the "
        "endpoint cannot move) is implemented as FAIL-OPEN: the CE endpoint has no ceiling in "
        "this design, no cell is dropped from the sigma pool on saturation grounds, and the "
        "count of such drops is reported as zero rather than the rule being silently skipped.",
        "Sliced-eval / W&B trajectory (the pre-registered SECONDARY endpoint) is not analysed "
        "here: sliced_eval is null unless --slice-mask-uri was set, and the W&B series is not "
        "in the per-cell JSON. Secondary means secondary; it moves no gate.",
        "The cross-run drift check (s7) and run 2 are out of scope for this script, which "
        "analyses run 1 only.",
    ]

    return {
        "schema_version": 1,
        "control": control,
        "alpha": alpha,
        "power_target": power,
        "sources": {
            "seed_schedule": schedule.get("path") or schedule.get("reason"),
            "seed_schedule_available": schedule.get("available", False),
            "arm_param_ledger": ledger.get("path") or ledger.get("reason"),
            "arm_param_ledger_available": ledger.get("available", False),
            "load_notes": list(load_notes or []),
            "dedup_notes": dedup_notes,
        },
        "realised_budget": budget,
        "coverage": {
            "cells_found": len(deduped),
            "cells_expected": schedule.get("n_expected_cells"),
            "cells_admissible": len(admissible),
            "cells_excluded": len(excluded),
            "arm_census": arm_census,
            "missing_cells": missing_cells,
            "partial": bool(missing_cells)
            or (
                schedule.get("n_expected_cells") is not None
                and len(deduped) < schedule["n_expected_cells"]
            ),
        },
        "hard_errors": [
            {
                "cell": c.cell_id,
                "arm": c.arm,
                "source": c.source,
                "reasons": c.reasons("hard_error"),
            }
            for c in hard_errors
        ],
        "admissibility": {
            "excluded_count": len(excluded),
            "excluded_cells": [
                {
                    "cell": c.cell_id,
                    "arm": c.arm,
                    "source": c.source,
                    "reasons": c.reasons("excluded") + c.reasons("hard_error"),
                }
                for c in excluded
            ],
            "warnings": [
                {"cell": c.cell_id, "arm": c.arm, "warnings": c.reasons("warning")}
                for c in deduped
                if c.reasons("warning")
            ],
            "outlier_warnings": flag_gross_outliers(admissible),
            "saturation_excluded_count": 0,
            "saturation_note": (
                "Pre-registration s4.2, fail-open: the CE endpoint has no ceiling in this "
                "design, so no cell is dropped from the sigma pool on saturation grounds."
            ),
            "arms_with_no_variance_estimate": arms_without_variance,
            "arms_with_no_admissible_cells": arms_absent,
        },
        "primary_endpoint": {
            "metric": "val_ce (nats, held out)",
            "per_arm": per_arm,
            "per_cell": ce_cells,
            "pooled_sigma": {
                "sigma": sigma_hat,
                "df": pooled["df"],
                "contributing_arms": pooled["contributing_groups"],
                "chi2_interval": sigma_interval,
                "literature_optimistic": SIGMA_OPTIMISTIC,
                "literature_pessimistic": SIGMA_PESSIMISTIC,
            },
            "anova": anova,
            "homogeneity": {
                "levene_median_centred": levene,
                "bartlett": bart,
                "decision_test": "Levene (median-centred)",
                "rejected_at_alpha": homogeneity_rejected,
                "fallback_engaged": homogeneity_rejected,
                "power_caveat": (
                    "At n = 3 per arm Levene and Bartlett have almost no power. Failing to "
                    "reject is weak evidence of homogeneity, not a demonstration of it."
                ),
            },
            "dunnett": contrasts,
            "welch_fallback": fallback,
            "mde": {"target_power": power, "table": mde_table, "measured_mde": mde_measured},
            "exploratory_gate_contrast": gate,
        },
        "co_primary_endpoints": {
            "throughput_headline": {
                "source": headline_source,
                "is_steady_state": throughput_is_steady,
                "keys_used": headline_keys,
                "mixed_keys": headline_mixed_keys,
                "per_arm": headline_tput,
            },
            "throughput_total_steady": {
                "per_arm": ratio_to_control(
                    {a: v for a, v in tput_total_by_arm.items() if v}, control
                ),
                "meta": tput_total_meta,
            },
            "throughput_per_device_steady": {
                "per_arm": ratio_to_control(
                    {a: v for a, v in tput_dev_by_arm.items() if v}, control
                ),
                "meta": tput_dev_meta,
            },
            "throughput_whole_run": {
                "per_arm": ratio_to_control(
                    {a: v for a, v in tput_whole_by_arm.items() if v}, control
                ),
                "meta": tput_whole_meta,
                "warning": (
                    "NOT AN ENDPOINT. Whole-run wall clock charges process start, dataset "
                    "open, FSDP wrap and the first-step compile against the hardware. This "
                    "project measured it reading 3.1x LOW on a short probe, and it penalises "
                    "bigger shapes hardest."
                ),
            },
            "peak_memory_gib": {
                "per_arm": memory_stats,
                "missing": mem_missing,
                "sources": mem_sources,
                "sources_mixed": mem_sources_mixed,
                "gated_on_source": MEMORY_SOURCE_TRUSTED,
                "lower_bound_only_per_arm": ratio_to_control(
                    {a: v for a, v in mem_lower_bound_by_arm.items() if v}, control
                ),
                "note": (
                    "The headline table admits ONLY peak_memory_source == "
                    f"'{MEMORY_SOURCE_TRUSTED}'. peak_memory_gib changed meaning while keeping "
                    "its name: it is now a true per-step running max. 'final_step_only' is the "
                    "LAST STEP'S peak (the GPU monitor resets the counters every step) -- a "
                    "lower bound wearing the name of a peak, which must not size hardware, so "
                    "it is tabled separately. A null or a zero is missing, never a value."
                ),
                "mixed_source_warning": (
                    "MEMORY FIGURES COME FROM MORE THAN ONE SOURCE: "
                    + ", ".join(f"{k} x{v}" for k, v in sorted(mem_sources.items()))
                    + ". These are different quantities and are NOT pooled: only "
                    f"'{MEMORY_SOURCE_TRUSTED}' cells are in the headline table."
                )
                if mem_sources_mixed
                else None,
            },
            "throughput_measurement_cross_check": {
                "note": (
                    "tps_device_avg is upstream SpeedMonitorCallback's average over a window "
                    "starting after step 1; throughput_tok_s_steady_per_device starts after "
                    f"{50}. Upstream's should sit slightly BELOW ours. A large gap means "
                    "compilation or allocator growth leaked into upstream's average, which is "
                    "the exact failure the cutoff exists for."
                ),
                "per_cell": tps_gap,
                "suspicious": tps_gap_suspicious,
            },
            "step_time_median_s": ratio_to_control(
                {a: v for a, v in collect_metric(admissible, STEP_TIME_MEDIAN_KEYS)[0].items() if v},
                control,
            ),
            "step_time_p90_s": ratio_to_control(
                {a: v for a, v in collect_metric(admissible, STEP_TIME_P90_KEYS)[0].items() if v},
                control,
            ),
        },
        "recommendation": recommendation,
        "deviations_from_preregistration": deviations,
        "trainer_reported_production_decision": trainer_reported,
        "unconsumed_keys": unconsumed,
    }


# --------------------------------------------------------------------------------------------
# Markdown rendering. This is what Eric reads.
# --------------------------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    out: List[str] = []
    add = out.append
    control = report["control"]
    cov = report["coverage"]

    add("# Mixer bake-off -- run 1 analysis")
    add("")
    add(
        f"**{cov['cells_admissible']} admissible cells** out of {cov['cells_found']} found; "
        f"{cov['cells_expected'] if cov['cells_expected'] is not None else '?'} expected. "
        f"Control arm: `{control}`. alpha = {report['alpha']}, target power = "
        f"{report['power_target']}."
    )
    add("")

    # --- hard errors first, loud ---
    if report["hard_errors"]:
        add("## HARD ERRORS -- read these before anything else")
        add("")
        add(
            "A hard error means the arm that ran is not the arm that was declared. The "
            "affected arm's comparison is INVALID; it is not a noisy number, it is the wrong "
            "number."
        )
        add("")
        for err in report["hard_errors"]:
            add(f"- **{err['cell']}** (`{err['arm']}`, {err['source']})")
            for reason in err["reasons"]:
                add(f"  - {reason}")
        add("")

    # --- the realised budget, before any number that depends on it ---
    bud = report["realised_budget"]
    if bud["steps_disagree_across_cells"]:
        add("## THE CELLS DID NOT ALL RUN THE SAME BUDGET")
        add("")
        add(
            "`steps` takes more than one value across these cells: "
            + ", ".join(f"{k:,} steps x{v}" for k, v in sorted(bud["steps_observed"].items()))
            + ". A cell on a different budget consumed a DIFFERENT PREFIX of the data stream "
            "and is not paired with the others. The modal budget "
            f"({bud['steps_used_for_admissibility']:,}) is treated as correct and the rest are "
            "excluded below, but this needs explaining before any of it is believed."
        )
        add("")
    if bud["steps_differ_from_plan"]:
        add(
            f"> **Budget changed after pre-registration.** `seeds.json` froze "
            f"{bud['steps_planned_in_seeds_json']:,} steps; these cells ran "
            f"{bud['steps_used_for_admissibility']:,}"
            + (
                f" ({bud['realised_tokens_per_cell']:,} tokens/cell"
                + (
                    f", TPP {bud['realised_tpp']:.1f}"
                    if bud["realised_tpp"] is not None
                    else ""
                )
                + ")"
                if bud["realised_tokens_per_cell"]
                else ""
            )
            + f". Admissibility is held to the realised budget ({bud['steps_basis']}), not to "
            "the plan -- pinning to the plan would have excluded every cell as a short run and "
            "reported a confident empty study. **A shorter budget makes the CE magnitudes MORE "
            "inflated, not less.**"
        )
        add("")

    # --- per-arm census: the first thing to read on a partial run ---
    add("## 0. Where each arm stands")
    add("")
    add("| arm | n admissible | of expected | excluded | df it contributes | status |")
    add("|---|---:|---:|---:|---:|---|")
    for row in cov["arm_census"]:
        add(
            f"| `{row['arm']}` | **{row['n_admissible']}** | {row['n_expected']} | "
            f"{row['n_excluded']} | {row['df_contributed']} | {row['status']} |"
        )
    add("")
    n_one = [r["arm"] for r in cov["arm_census"] if r["n_admissible"] == 1]
    n_zero = [r["arm"] for r in cov["arm_census"] if r["n_admissible"] == 0]
    if n_one:
        add(
            "> **"
            + ", ".join(f"`{a}`" for a in n_one)
            + " has exactly ONE admissible cell.** Its `val_ce` below is a single "
            "observation. It is not a mean, it has no sd, it contributes no df, and it gets "
            "no confidence interval. Do not compare it to anything as though it did."
        )
        add("")
    if n_zero:
        add(
            "> **"
            + ", ".join(f"`{a}`" for a in n_zero)
            + " has NO admissible cells** and is absent from every table below. Absent is "
            "not the same as tied."
        )
        add("")

    # --- coverage / partial input ---
    if cov["partial"]:
        add("## PARTIAL INPUT")
        add("")
        add(
            f"{len(cov['missing_cells'])} of the frozen 18-cell schedule are NOT in this "
            "input. Everything below is computed on what is present, and the power, the "
            "pooled sigma and every MDE shrink accordingly. This is not the full analysis."
        )
        add("")
        add("| arm | replicate | data_seed | init_seed |")
        add("|---|---:|---:|---:|")
        for cell in cov["missing_cells"]:
            add(
                f"| `{cell['arm']}` | {cell['replicate']} | {cell['data_seed']} | "
                f"{cell['init_seed']} |"
            )
        add("")

    # --- admissibility ---
    adm = report["admissibility"]
    add("## 1. Admissibility (step 0, before any pooling)")
    add("")
    add(
        f"**{adm['excluded_count']} cell(s) excluded.** "
        + (
            "Every exclusion is named below with its reason."
            if adm["excluded_count"]
            else "No cell was excluded."
        )
    )
    add("")
    if adm["excluded_cells"]:
        add("| cell | arm | why |")
        add("|---|---|---|")
        for cell in adm["excluded_cells"]:
            reasons = "; ".join(cell["reasons"])
            add(f"| `{cell['cell']}` | `{cell['arm']}` | {reasons} |")
        add("")
    add(f"Saturation-excluded from the sigma pool: {adm['saturation_excluded_count']}. "
        f"{adm['saturation_note']}")
    add("")
    if adm["arms_with_no_variance_estimate"]:
        add(
            "> **ARMS WITH FEWER THAN 2 ADMISSIBLE CELLS: "
            + ", ".join(f"`{a}`" for a in adm["arms_with_no_variance_estimate"])
            + ".** These arms CANNOT contribute a variance estimate. Their mean below is a "
            "single number, not a mean of replicates, and it carries no error bar. It is "
            "excluded from the pooled sigma."
        )
        add("")
    if adm["arms_with_no_admissible_cells"]:
        add(
            "> **ARMS WITH NO ADMISSIBLE CELLS AT ALL: "
            + ", ".join(f"`{a}`" for a in adm["arms_with_no_admissible_cells"])
            + ".** These arms are absent from every table below."
        )
        add("")
    if adm["outlier_warnings"]:
        add("**Outlier warnings (flagged, NOT excluded):**")
        add("")
        for warning in adm["outlier_warnings"]:
            add(f"- {warning}")
        add("")
    if adm["warnings"]:
        add("<details><summary>Per-cell warnings (checks that are UNCHECKED, not passed)</summary>")
        add("")
        for entry in adm["warnings"]:
            for warning in entry["warnings"]:
                add(f"- `{entry['cell']}` (`{entry['arm']}`): {warning}")
        add("")
        add("</details>")
        add("")

    # --- primary endpoint ---
    prim = report["primary_endpoint"]
    add("## 2. Primary endpoint -- held-out `val_ce` (nats)")
    add("")
    add("| arm | n | mean | sd | per-cell values |")
    add("|---|---:|---:|---:|---|")
    for arm, stats in prim["per_arm"].items():
        values = ", ".join(f"{v:.4f}" for v in stats["values"])
        if stats["n"] < 2:
            # NEVER a mean-with-error-bars for a single observation.
            add(
                f"| `{arm}` | {stats['n']} | _{_fmt(stats['mean'])} (single obs, NOT a mean)_ | "
                f"n/a | {values} |"
            )
            continue
        add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'])} | {stats['sd']:.5f} | {values} |")
    add("")

    sigma = prim["pooled_sigma"]
    add("### 2.1 Pooled within-arm sigma -- a pre-registered deliverable in its own right")
    add("")
    if sigma["sigma"] is None:
        add("Not computable: no arm has 2 or more admissible cells.")
    else:
        add(
            f"**sigma_hat = {sigma['sigma']:.5f} nats** at df = {sigma['df']} "
            f"({sigma['contributing_arms']} arms contributing)."
        )
        interval = sigma["chi2_interval"]
        if interval:
            # THE BRACKET WIDTH IS COMPUTED FROM THE REALISED df, not quoted from the
            # pre-registration's df = 12 case. On a partial run df is smaller and the bracket
            # is wider; printing the full-run "factor 2.3" there would understate the
            # uncertainty on the one number run 2 gets sized from.
            factor = interval["multiplier_high"] / interval["multiplier_low"]
            add("")
            add(
                f"chi-squared interval at df = {interval['df']}: "
                f"sigma_hat x [{interval['multiplier_low']:.3f}, "
                f"{interval['multiplier_high']:.3f}] = "
                f"[{interval['sigma_low']:.5f}, {interval['sigma_high']:.5f}] nats -- a "
                f"**factor-{factor:.1f} bracket at df = {interval['df']}**"
                + (
                    f" (the pre-registration's full-run df = 12 case is factor 2.3; this run "
                    f"has only df = {interval['df']}, so the bracket is wider)"
                    if interval["df"] < 12
                    else ""
                )
                + ". This narrows the 5.5x guess, it does not close it. **Run 2 must be sized "
                "from this number, not from either literature estimate.**"
            )
        add("")
        add(
            f"For reference the two prior estimates were {sigma['literature_optimistic']} "
            f"(optimistic) and {sigma['literature_pessimistic']} (pessimistic) nats."
        )
    add("")

    add("### 2.2 Variance homogeneity")
    add("")
    hom = prim["homogeneity"]
    lev, bar = hom["levene_median_centred"], hom["bartlett"]
    if lev["computable"]:
        add(
            f"- **Levene (median-centred), THE decision test**: W = {lev['statistic']:.4f}, "
            f"df = ({lev['df_between']}, {lev['df_within']}), p = {lev['p']:.4f}"
        )
    else:
        add(f"- **Levene**: not computable -- {lev['reason']}")
    if bar["computable"]:
        add(
            f"- Bartlett (reported, not deciding): chi2 = {bar['statistic']:.4f}, "
            f"df = {bar['df']}, p = {bar['p']:.4f}"
        )
    else:
        add(f"- Bartlett: not computable -- {bar['reason']}")
    add("")
    if hom["rejected_at_alpha"]:
        add(
            "**Levene REJECTS at alpha = "
            f"{report['alpha']}. The pre-registered fallback is engaged: Welch's ANOVA, no "
            "pooling, per-arm sigma, and unequal-variance contrasts against the control.**"
        )
        fb = report["primary_endpoint"]["welch_fallback"]
        if fb and fb["welch_anova"].get("computable"):
            wa = fb["welch_anova"]
            add("")
            add(
                f"Welch's ANOVA: F = {wa['f']:.4f}, df = ({wa['df_between']}, "
                f"{wa['df_within']:.2f}), p = {wa['p']:.4f}"
            )
        if fb and fb["t3_contrasts"].get("computable"):
            add("")
            add(f"Unequal-variance contrasts vs `{control}` ({fb['t3_contrasts']['procedure']}):")
            add("")
            add("| arm | n | estimate | Welch df | crit | 95% CI | excludes 0 |")
            add("|---|---:|---:|---:|---:|---|:--:|")
            for row in fb["t3_contrasts"]["contrasts"]:
                if not row.get("computable"):
                    add(f"| `{row['arm']}` | {row['n']} | -- | -- | -- | {row['reason']} | -- |")
                    continue
                add(
                    f"| `{row['arm']}` | {row['n']} | {row['estimate']:+.5f} | "
                    f"{row['welch_df']:.2f} | {row['critical_value']:.3f} | "
                    f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | "
                    f"{'YES' if row['excludes_zero'] else 'no'} |"
                )
    else:
        add(
            "Levene does not reject; the pooled analysis stands. "
            + hom["power_caveat"]
        )
    add("")

    add("### 2.3 Pooled-variance one-way ANOVA")
    add("")
    anova = prim["anova"]
    if anova["computable"]:
        add(
            f"F({anova['df_between']}, {anova['df_within']}) = {anova['f']:.4f}, "
            f"p = {anova['p']:.4f}. Error df = sum over arms of (n_i - 1) = "
            f"{anova['df_within']}."
        )
    else:
        add(f"Not computable: {anova['reason']}")
    add("")

    add(f"### 2.4 Dunnett contrasts vs `{control}` (two-sided)")
    add("")
    dun = prim["dunnett"]
    mde = prim["mde"]["measured_mde"]
    if not dun.get("computable"):
        add(f"Not computable: {dun['reason']}")
    else:
        val = dun["critical_value_validation"]
        add(
            f"Critical value **{dun['critical_value']:.4f}** at k = {dun['k']}, "
            f"df = {dun['df']:.0f}, rho = {dun['rho']:.3f}. Method: {val['method']}."
        )
        add("")
        add(
            "Sanity-checked two ways: (1) quadrature refinement 48 vs 96 nodes -> "
            f"{val['check_1_quadrature_refinement']['abs_diff']:.2e} "
            f"({'PASS' if val['check_1_quadrature_refinement']['passed'] else 'FAIL'}); "
            "(2) the k = 1 reduction, where max-|t| is just |t| and the critical value must "
            "equal the Student t quantile computed by a different route (inverse regularized "
            f"incomplete beta) -> {val['check_2_k1_reduces_to_student_t']['abs_diff']:.2e} "
            f"({'PASS' if val['check_2_k1_reduces_to_student_t']['passed'] else 'FAIL'})."
        )
        if dun["rho"] != 0.5:
            add("")
            add(f"> {dun['rho_note']}")
        add("")
        add("| arm | n | estimate (nats) | Dunnett 95% CI | half-width | clears MDE? | adj. p |")
        add("|---|---:|---:|---|---:|:--:|---:|")
        for row in dun["contrasts"]:
            clears = "n/a" if mde is None else ("YES" if abs(row["estimate"]) > mde else "no")
            add(
                f"| `{row['arm']}` | {row['n']} | {row['estimate']:+.5f} | "
                f"[{row['ci_low']:+.5f}, {row['ci_high']:+.5f}] | {row['ci_half_width']:.5f} | "
                f"{clears} | {_fmt(row['dunnett_adjusted_p'], '.4f')} |"
            )
        add("")
        add(
            "Negative estimate = better than control (lower CE). A contrast is only a "
            "detected difference if its CI excludes zero AND it clears the MDE. **Where the "
            "CI includes zero, the honest statement is the BOUND: the difference is smaller "
            "than the half-width in that row.**"
        )
    add("")

    add("### 2.5 MDE at 80% power -- exact non-central t, dominant tail only")
    add("")
    if prim["mde"]["table"]:
        add("| sigma basis | sigma (nats) | SE(difference) | MDE (nats) |")
        add("|---|---:|---:|---:|")
        for row in prim["mde"]["table"]:
            add(
                f"| {row['label']} | {_fmt(row['sigma'], '.5f')} | "
                f"{_fmt(row.get('se_of_difference'), '.6f')} | {_fmt(row['mde'], '.5f')} |"
            )
        add("")
        add(
            "The normal approximation is 2.2x too optimistic at n = 3 and is not used "
            "anywhere here."
        )
    else:
        add("Not computable (no usable contrasts).")
    add("")

    gate = prim["exploratory_gate_contrast"]
    add("### 2.6 The gate contrast -- `KDA_GCONV - KDA_NOACT` (EXPLORATORY)")
    add("")
    add(
        "With `gate_structure=\"depthwise\"`, `gated_conv_activation=None` does NOT mean "
        "activation-free -- the depthwise pre-gate is exactly a SiLU with a learnable "
        "per-channel slope. So the contrast that isolates the gate is `KDA_GCONV - "
        "KDA_NOACT`, never `KDA_GCONV - KDA_BASE`. It is not against the control, so per the "
        "pre-registration it is EXPLORATORY and uncorrected."
    )
    add("")
    if gate.get("computable"):
        add(
            f"Estimate {gate['estimate']:+.5f} nats, uncorrected 95% CI "
            f"[{gate['ci_low']:+.5f}, {gate['ci_high']:+.5f}], n = {gate['n_left']} vs "
            f"{gate['n_right']}, uncorrected p = {_fmt(gate['p_uncorrected'], '.4f')}."
        )
    else:
        add(f"Not computable: {gate['reason']}")
    add("")

    # --- co-primary ---
    co = report["co_primary_endpoints"]
    add("## 3. Co-primary endpoints -- throughput and peak memory")
    add("")
    head = co["throughput_headline"]
    add(f"Headline throughput source: **{head['source']}**.")
    add("")
    if head["mixed_keys"]:
        add(
            "> **THE HEADLINE THROUGHPUT CAME FROM MORE THAN ONE KEY across these cells.** "
            "`throughput_tok_s_steady` excludes 50 warmup steps; `tps_total_avg` is upstream "
            "SpeedMonitorCallback's average over a window starting after step 1 only. They are "
            "not the same measurement, and a ratio built from both is comparing two "
            "definitions. Treat the ratios below as unreliable until every cell reports the "
            "same key."
        )
        add("")
    if not head["is_steady_state"] and head["per_arm"]:
        add(
            "> **WARNING -- THIS IS THE WHOLE-RUN, STARTUP-CONTAMINATED FIGURE.** No "
            "steady-state throughput key (`tps_device_avg` / `tps_total_avg`) is present in "
            "these summaries. Whole-run wall clock charges process start, dataset open, FSDP "
            "wrap and the first-step compile against the hardware; this project measured it "
            "reading **3.1x LOW** on a short probe, and it penalises bigger shapes hardest -- "
            "which is precisely the direction that would bias a mixer choice. Do not make a "
            "production decision on this row."
        )
        add("")
    if head["per_arm"]:
        add("| arm | n | mean tok/s | sd | ratio to control |")
        add("|---|---:|---:|---:|---:|")
        for arm, stats in head["per_arm"].items():
            sd = "n/a (n<2)" if stats["sd"] is None else f"{stats['sd']:,.0f}"
            ratio = (
                stats["ratio_unavailable_reason"]
                if stats["ratio_to_control"] is None
                else f"{stats['ratio_to_control']:.3f}x"
            )
            add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'], ',.0f')} | {sd} | {ratio} |")
        add("")
    else:
        add(
            "**No throughput data at all.** Neither `tps_total_avg`, `tps_device_avg`, "
            "`throughput_tok_s_steady` nor `tps_naive_wall_clock` is present in any admissible "
            "cell. Throughput contributes nothing to the recommendation below, and that is "
            "recorded as absent rather than treated as equal."
        )
        add("")

    dev = co["throughput_per_device_steady"]["per_arm"]
    if dev:
        add("Per-device steady-state (`tps_device_avg`):")
        add("")
        add("| arm | n | mean tok/s/device | sd | ratio to control |")
        add("|---|---:|---:|---:|---:|")
        for arm, stats in dev.items():
            sd = "n/a (n<2)" if stats["sd"] is None else f"{stats['sd']:,.0f}"
            ratio = (
                "--" if stats["ratio_to_control"] is None else f"{stats['ratio_to_control']:.3f}x"
            )
            add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'], ',.0f')} | {sd} | {ratio} |")
        add("")

    mem = co["peak_memory_gib"]["per_arm"]
    add("### 3.1 Peak memory (GiB, rank 0)")
    add("")
    if co["peak_memory_gib"]["mixed_source_warning"]:
        add("> **" + co["peak_memory_gib"]["mixed_source_warning"] + "**")
        add("")
    elif co["peak_memory_gib"]["sources"]:
        add(
            "Source: "
            + ", ".join(f"`{k}` x{v}" for k, v in sorted(co["peak_memory_gib"]["sources"].items()))
            + "."
        )
        add("")
    if mem:
        add("| arm | n | mean GiB | sd | ratio to control |")
        add("|---|---:|---:|---:|---:|")
        for arm, stats in mem.items():
            sd = "n/a (n<2)" if stats["sd"] is None else f"{stats['sd']:.3f}"
            ratio = (
                "--" if stats["ratio_to_control"] is None else f"{stats['ratio_to_control']:.3f}x"
            )
            add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'], '.2f')} | {sd} | {ratio} |")
        add("")
    else:
        add("**No usable peak-memory data.** " + co["peak_memory_gib"]["note"])
        add("")
    lower = co["peak_memory_gib"]["lower_bound_only_per_arm"]
    if lower:
        add(
            "**Lower-bound-only cells (`final_step_only`), tabled separately because they are "
            "not the same quantity and MUST NOT size hardware:**"
        )
        add("")
        add("| arm | n | mean GiB (LOWER BOUND) | sd |")
        add("|---|---:|---:|---:|")
        for arm, stats in lower.items():
            sd = "n/a (n<2)" if stats["sd"] is None else f"{stats['sd']:.3f}"
            add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'], '.2f')} | {sd} |")
        add("")
    if co["peak_memory_gib"]["missing"]:
        add(
            "Cells not in the headline memory table: "
            + ", ".join(f"`{m['cell']}` ({m['state']})" for m in co["peak_memory_gib"]["missing"])
        )
        add("")

    cross = co["throughput_measurement_cross_check"]
    if cross["suspicious"]:
        add("### 3.2 Throughput cross-check -- DISAGREEMENT BETWEEN THE TWO MEASUREMENTS")
        add("")
        add(cross["note"])
        add("")
        add("| cell | arm | tps_device_avg | steady_per_device | upstream/ours |")
        add("|---|---|---:|---:|---:|")
        for row in cross["suspicious"]:
            add(
                f"| `{row['cell']}` | `{row['arm']}` | "
                f"{row['tps_device_avg']:,.0f} | "
                f"{row['throughput_tok_s_steady_per_device']:,.0f} | "
                f"**{row['ratio_upstream_over_ours']:.3f}** |"
            )
        add("")
        add(
            "A ratio well below 1 means compilation or allocator growth leaked into upstream's "
            "average. A ratio above 1 is backwards and means one of the two is not measuring "
            "what its name says. Either way this cell's throughput needs a look before it "
            "carries a production decision."
        )
        add("")
    elif cross["per_cell"]:
        add(
            f"Throughput cross-check: `tps_device_avg` sits below "
            f"`throughput_tok_s_steady_per_device` on all {len(cross['per_cell'])} cell(s) "
            "with both, as expected. No compilation leak into upstream's window."
        )
        add("")

    for label, key in (
        ("Median step time (s)", "step_time_median_s"),
        ("p90 step time (s)", "step_time_p90_s"),
    ):
        table = co.get(key) or {}
        if not table:
            continue
        add(f"### {label}")
        add("")
        add("| arm | n | mean | sd | ratio to control |")
        add("|---|---:|---:|---:|---:|")
        for arm, stats in table.items():
            sd = "n/a (n<2)" if stats["sd"] is None else f"{stats['sd']:.4f}"
            ratio = (
                "--" if stats["ratio_to_control"] is None else f"{stats['ratio_to_control']:.3f}x"
            )
            add(f"| `{arm}` | {stats['n']} | {_fmt(stats['mean'], '.4f')} | {sd} | {ratio} |")
        add("")

    # --- recommendation ---
    rec = report["recommendation"]
    add("## 4. RECOMMENDATION")
    add("")
    add(f"### Use `{rec['choice']}`" if rec["choice"] else "### No arm can be recommended")
    add("")
    add(f"**Basis: {rec['basis']}.**")
    add("")
    add(
        f"CE resolved at this n? **{'YES' if rec['ce_resolved'] else 'NO'}** "
        f"(MDE = {_fmt(rec['mde_nats'], '.5f')} nats at the measured sigma)."
    )
    add("")
    for line in rec["rationale"]:
        add(f"- {line}")
    add("")
    add("### Caveats that travel with this recommendation")
    add("")
    for caveat in rec["caveats"]:
        add(f"- {caveat}")
    add("")

    if report["trainer_reported_production_decision"]:
        add("### Production-decision objects reported by the trainer itself")
        add("")
        add(
            "Passed through for visibility. They did NOT feed the recommendation above -- this "
            "script's decision is derived from the endpoints, not from a field in the input."
        )
        add("")
        add("```json")
        add(json.dumps(report["trainer_reported_production_decision"], indent=2, sort_keys=True))
        add("```")
        add("")

    add("## 5. Deviations from the pre-registration")
    add("")
    for dev_note in report["deviations_from_preregistration"]:
        add(f"- {dev_note}")
    add("")

    if report["unconsumed_keys"]:
        add("## 6. Keys present in the summaries that this analysis did not consume")
        add("")
        add(
            "Listed so nobody has to guess whether a new field was read. If one of these is "
            "meant to be an endpoint, it is not one yet."
        )
        add("")
        for key, count in sorted(report["unconsumed_keys"].items()):
            add(f"- `{key}` (in {count} cell(s))")
        add("")

    src = report["sources"]
    add("## 7. Provenance")
    add("")
    add(f"- Seed schedule: `{src['seed_schedule']}` (available: {src['seed_schedule_available']})")
    add(
        f"- Parameter ledger (`ARM_L0_DELTA`): `{src['arm_param_ledger']}` "
        f"(available: {src['arm_param_ledger_available']})"
    )
    for note in src["load_notes"] + src["dedup_notes"]:
        add(f"- {note}")
    add("")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------------------------


def fetch_command(prefix: str) -> str:
    """
    Print -- and only print -- the command that would pull the per-cell summaries down.

    THIS SCRIPT DOES NOT TOUCH AWS. Fetching is a separate, deliberate step run by a human
    with credentials; the analysis takes local files.
    """
    prefix = prefix.rstrip("/")
    return (
        "# Run this yourself; analyse_bakeoff.py does not touch AWS.\n"
        f"aws s3 cp --recursive {prefix}/ ./bakeoff-results/ --exclude '*' --include '*.json'\n"
        "python scripts/analyse_bakeoff.py ./bakeoff-results/\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse the mixer bake-off per-cell summaries per the pre-registration.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="per-cell JSON files, log files, or directories containing them",
    )
    parser.add_argument(
        "--out",
        default="bakeoff_results.json",
        help="where to write the machine-readable results (default: bakeoff_results.json)",
    )
    parser.add_argument("--seeds", default=str(DEFAULT_SEEDS_JSON), help="path to seeds.json")
    parser.add_argument(
        "--arms-source", default=str(DEFAULT_ARMS_SOURCE), help="path to core6_arms.py"
    )
    parser.add_argument("--control", default=DEFAULT_CONTROL, help="the shared control arm")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument(
        "--print-fetch-command",
        metavar="S3_PREFIX",
        default=None,
        help="print the aws s3 cp command for a results prefix and exit; touches nothing",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="skip writing the machine-readable results file"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.print_fetch_command:
        sys.stdout.write(fetch_command(args.print_fetch_command))
        return 0

    stdin_text = None
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
    if not args.inputs and not (stdin_text and stdin_text.strip()):
        sys.stderr.write(
            "no input: pass files or a directory, or pipe summaries / paths on stdin\n"
        )
        return 2

    cells, notes = load_cells(args.inputs, stdin_text)
    if not cells:
        sys.stderr.write(
            "no cell summaries found in the input. Notes:\n  " + "\n  ".join(notes) + "\n"
        )
        return 2

    schedule = load_seed_schedule(Path(args.seeds))
    ledger = load_arm_param_targets(Path(args.arms_source))
    report = analyse(
        cells,
        schedule=schedule,
        ledger=ledger,
        control=args.control,
        alpha=args.alpha,
        power=args.power,
        load_notes=notes,
    )

    if not args.no_json:
        out_path = Path(args.out)
        if out_path.parent != Path(""):
            os.makedirs(out_path.parent, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        sys.stderr.write(f"wrote {out_path}\n")

    sys.stdout.write(render_markdown(report) + "\n")

    # Exit 1 on a hard error so a wrapper cannot mistake an invalidated arm for a clean run.
    return 1 if report["hard_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
