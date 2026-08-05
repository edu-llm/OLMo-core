"""Exp-2's actual primary deliverable: a MEASURED sigma and an honest required-n.

READ THIS FIRST
---------------
**Exp-2 is not an accuracy verdict.** Per ``SPEC.md`` Sec 4.1 and ``R3-statistics.md`` F8, the
repo has already measured seed-to-seed sigma on this exact task family at ~1M params:

    per-seed accuracy@512, ONE identical config: 50.00 / 90.82 / 99.80 / 98.44 / 2.73 %
    sigma = 42 pp across seeds;  sigma_within (pooled) = 48.4 pp
    one-way ANOVA n=20, F(3,16) = 0.337 (F_crit 3.24)
    eta^2 = 5.9 % for MEMORY LOAD vs 94.1 % for SEED
    -- KDA/HANDOFF.md:420-457 [MEASURED]

At that variance a 5 pp effect needs **556-1,473 paired seeds** for 80 % power. The proposal plans
5. That is a 110-150x shortfall, so **no accuracy gate run at n<=10 is informative** and this module
refuses to pretend otherwise: :func:`superiority_verdict` returns ``UNDERPOWERED`` rather than
``PASS``/``FAIL`` when the achieved power at the pre-registered effect is below
:data:`MIN_INFORMATIVE_POWER`.

What this module therefore provides, in priority order:

1. **Endpoint definitions** (Sec 4.2-4.4) -- success rate, median-vs-floor, query NLL, clustered SEs.
   The harness imports these so there is exactly one definition of every endpoint.
2. **Sigma estimation** -- per cell, pooled within-cell, paired-difference ``s_delta``, realized rho.
3. **Required-n inversion** -- exact non-central t, paired, 80 % power, alpha=0.05, with a clearly
   labelled normal-approximation fallback if scipy is absent.
4. **Operating characteristics** of any candidate criterion, under the null AND at hypothesized
   effects, so a coin flip is visible as a coin flip.
5. **CI-based endpoints, not sign tests** (F3 fix): "the upper bound of the 95 % CI on the difference
   is below X". Same data, an actual alpha, and it fails when the data are ambiguous.
6. **The pre-registered decision rule**, fingerprinted so it cannot be quietly relaxed.
7. **Priced miss rates** for every non-inferiority clause (F5e), because a clause that misses a
   40-point regression 63 % of the time is decoration.

WHY THE FALLBACK IS GUARDED AND LABELLED
----------------------------------------
The normal approximation ignores the t-distribution's df penalty, so it *understates* required n --
by 2 seeds at n~140 and by ~0.2 % at n~1000, but by much more at the small n this study can afford.
Silently substituting it would make an underpowered design look adequate, which is the exact failure
mode F8 documents. So :data:`HAVE_SCIPY` is exported, every returned record carries a ``method``
field, and the approximation is never selected while scipy is importable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised by whichever branch the host env has
    from scipy.stats import nct, norm, t as student_t

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    nct = norm = student_t = None  # type: ignore[assignment]
    HAVE_SCIPY = False


# ======================================================================================
# SECTION 1 -- ENDPOINT DEFINITIONS (SPEC Sec 4.2-4.4)
# ======================================================================================

# A run counts as "found the recall algorithm" above this accuracy.
#
# NOT an arbitrary cut, and this is the justification the spec asks for: across the 12
# positive-control trials (mqar_positive_control.json, FarmShare 1670928) NO run landed between
# 0.30 and 0.80. Two solved (0.995, 1.000), six sat on the 1/D degenerate floor (0.208-0.274), four
# fell below it (0.000-0.214). 0.80 sits inside a MEASURED empty gap, so the success rate is
# insensitive to the threshold's exact value over [0.30, 0.98].
SOLVE_THRESHOLD = 0.80

# Minimum eval sequences per cell. R3 F8 fix 4: "design every synthetic probe with >=1,000 items"
# (Miller Eq. 9: n ~= 969 for delta=0.03 with likelihood scoring). Templated probes have naive SEs
# up to 3x too small -- "a fabricated significance factory" -- hence CLUSTERED SEs below.
MIN_EVAL_ITEMS = 1000

# Below this achieved power, a comparison is reported as UNDERPOWERED rather than PASS/FAIL.
# 0.50 is deliberately permissive: R3 F8 measures the proposal's own criterion at 24-38 % power, so
# even this lax bar rejects it. Anything that clears 0.50 still gets its exact power reported.
MIN_INFORMATIVE_POWER = 0.50


def degenerate_floor(num_pairs: int) -> float:
    """
    The chance baseline. It is ``1/D``, **not** ``1/vocab``, and it moves with the config.

    A model that learns "the answer is one of the D values present in this sequence" without
    binding anything scores exactly ``1/num_pairs``. Six of twelve positive-control trials sat at
    0.208-0.274 with losses of 1.40-1.76 against ``ln(4) = 1.386`` -- a fully-learned WRONG
    algorithm, not partial recall. At D=64 the floor is 0.016, so 0.10 there is real work; at D=4
    it is *below* the degenerate strategy.

    :param num_pairs: ``D``, the number of key-value pairs.
    :returns: Accuracy of the degenerate "guess among the D present values" strategy.
    """
    if num_pairs < 1:
        raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")
    return 1.0 / num_pairs


@dataclass(frozen=True)
class ClusteredMean:
    """
    A mean over clustered observations, with both SEs so the design effect is visible.

    MQAR query tokens are **not** independent: all ``D`` queries in one sequence share the same
    key-value table, the same filler, and the same forward pass. Treating them as independent
    inflates the effective sample size by up to ``D`` and shrinks the SE by up to ``sqrt(D)``.

    :param mean: The point estimate.
    :param se_clustered: SE treating the sequence as the unit of inference. **Use this one.**
    :param se_naive: SE treating each token as independent. Reported only for the ratio.
    :param design_effect: ``(se_clustered / se_naive) ** 2``. 1.0 means clustering cost nothing;
        values >1 are how much the naive SE was lying.
    :param n_clusters: Number of sequences.
    :param n_units: Number of scored tokens.
    """

    mean: float
    se_clustered: float
    se_naive: float
    design_effect: float
    n_clusters: int
    n_units: int


def clustered_mean(
    cluster_sums: Sequence[float],
    cluster_counts: Sequence[int],
    *,
    naive_variance: Optional[float] = None,
) -> ClusteredMean:
    """
    Cluster-robust mean and SE, clustering on the eval sequence.

    Uses the ratio estimator ``sum(sums)/sum(counts)`` with the standard linearized
    cluster-robust variance, which is correct for unequal cluster sizes and reduces to
    ``sd(cluster_means)/sqrt(K)`` when every cluster has the same size (the MQAR case, where every
    sequence contributes exactly ``D`` queries).

    :param cluster_sums: Per-sequence sum of the scored quantity (e.g. number correct).
    :param cluster_counts: Per-sequence count of scored tokens (e.g. ``D``).
    :param naive_variance: Per-unit variance for the naive SE. Defaults to the Bernoulli
        ``p(1-p)``, which is right for accuracy; pass the token-level variance for NLL.
    :returns: A :class:`ClusteredMean`.
    """
    if len(cluster_sums) != len(cluster_counts):
        raise ValueError("cluster_sums and cluster_counts must be the same length")
    k = len(cluster_sums)
    if k == 0:
        raise ValueError("no clusters")
    total_n = float(sum(cluster_counts))
    if total_n <= 0:
        raise ValueError("no scored units")
    mean = float(sum(cluster_sums)) / total_n

    # Linearized (ratio-estimator) cluster-robust variance. Residual per cluster is
    # (sum_i - mean * count_i); the finite-K correction K/(K-1) matches the equal-size reduction.
    if k > 1:
        resid_sq = sum((s - mean * c) ** 2 for s, c in zip(cluster_sums, cluster_counts))
        var_clustered = (k / (k - 1.0)) * resid_sq / (total_n**2)
    else:
        var_clustered = float("nan")
    se_clustered = math.sqrt(var_clustered) if var_clustered == var_clustered else float("nan")

    unit_var = mean * (1.0 - mean) if naive_variance is None else naive_variance
    unit_var = max(unit_var, 0.0)
    se_naive = math.sqrt(unit_var / total_n)

    if se_naive > 0 and se_clustered == se_clustered:
        design = (se_clustered / se_naive) ** 2
    else:
        design = float("nan")

    return ClusteredMean(
        mean=mean,
        se_clustered=se_clustered,
        se_naive=se_naive,
        design_effect=design,
        n_clusters=k,
        n_units=int(total_n),
    )


@dataclass
class SeedRecord:
    """
    One (arm, topology, W, config, seed) result. The atomic unit the harness writes to disk.

    :param accuracy: Fraction of query positions answered correctly.
    :param nll_query: Mean cross-entropy at query positions, in nats. The CONTINUOUS endpoint --
        worth a 2-18x SNR gain over accuracy and the single biggest free lever available (Sec 4.4).
    :param acc_se_clustered: Clustered SE on ``accuracy`` within this run.
    :param nll_se_clustered: Clustered SE on ``nll_query`` within this run.
    """

    arm: str
    topology: str
    kernel_size: int
    config: str
    seed: int
    accuracy: float
    nll_query: float
    num_pairs: int
    acc_se_clustered: float = float("nan")
    nll_se_clustered: float = float("nan")
    acc_design_effect: float = float("nan")
    nll_design_effect: float = float("nan")
    n_eval_items: int = 0
    first_loss: float = float("nan")
    final_loss: float = float("nan")
    n_params: int = 0
    seconds: float = float("nan")
    data_seed: int = -1
    init_seed: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.accuracy >= SOLVE_THRESHOLD

    @property
    def floor(self) -> float:
        return degenerate_floor(self.num_pairs)

    @property
    def above_floor(self) -> bool:
        return self.accuracy > self.floor * 1.5


@dataclass(frozen=True)
class CellEndpoints:
    """
    Every endpoint for one cell, per SPEC Sec 4.2-4.4. **Never a bare mean.**

    :param success_rate: Fraction of seeds at or above :data:`SOLVE_THRESHOLD`. The primary
        endpoint at low load, where trainability is bimodal.
    :param median_accuracy: Median accuracy. The primary location endpoint at high load, where
        bimodality BREAKS (measured: ``N512_D64`` seeds spread 0.05/0.09/0.20/0.56/0.98) and
        collapsing to a success rate discards most of the signal.
    :param median_over_floor: ``median_accuracy / (1/D)``. The scale-free version.
    :param mean_nll_query: Mean over seeds of per-seed query NLL. The continuous endpoint.
    :param per_seed_accuracy: The FULL sorted per-seed list, always, so bimodality is visible
        rather than hidden.
    :param bimodal: True when no seed lands in (0.2, 0.8) -- i.e. success rate is sufficient here.
    """

    arm: str
    topology: str
    kernel_size: int
    config: str
    num_pairs: int
    n_seeds: int
    floor: float
    success_rate: float
    median_accuracy: float
    median_over_floor: float
    mean_accuracy: float
    sigma_accuracy: float
    mean_nll_query: float
    sigma_nll_query: float
    per_seed_accuracy: Tuple[float, ...]
    per_seed_nll: Tuple[float, ...]
    per_seed_seeds: Tuple[int, ...]
    bimodal: bool
    n_mid: int

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _sd(xs: Sequence[float]) -> float:
    """Sample SD, ddof=1. Returns nan for n<2 rather than 0.0 -- n=1 has no measured spread."""
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: Sequence[float]) -> float:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def summarize_cell(records: Sequence[SeedRecord]) -> CellEndpoints:
    """
    Collapse the seeds of one cell into :class:`CellEndpoints`.

    :param records: All :class:`SeedRecord` for a single (arm, topology, W, config).
    :returns: Every endpoint, including the full per-seed list.
    """
    if not records:
        raise ValueError("no records")
    keys = {(r.arm, r.topology, r.kernel_size, r.config) for r in records}
    if len(keys) != 1:
        raise ValueError(f"records span multiple cells: {sorted(keys)}")
    if len({r.seed for r in records}) != len(records):
        raise ValueError("duplicate seeds in one cell")

    r0 = records[0]
    order = sorted(range(len(records)), key=lambda i: records[i].accuracy)
    accs = [records[i].accuracy for i in order]
    nlls = [records[i].nll_query for i in order]
    seeds = [records[i].seed for i in order]
    floor = degenerate_floor(r0.num_pairs)
    med = _median(accs)
    n_mid = sum(1 for a in accs if 0.2 < a < 0.8)

    return CellEndpoints(
        arm=r0.arm,
        topology=r0.topology,
        kernel_size=r0.kernel_size,
        config=r0.config,
        num_pairs=r0.num_pairs,
        n_seeds=len(records),
        floor=floor,
        success_rate=sum(1 for r in records if r.solved) / len(records),
        median_accuracy=med,
        median_over_floor=med / floor if floor > 0 else float("nan"),
        mean_accuracy=_mean(accs),
        sigma_accuracy=_sd(accs),
        mean_nll_query=_mean(nlls),
        sigma_nll_query=_sd(nlls),
        per_seed_accuracy=tuple(accs),
        per_seed_nll=tuple(nlls),
        per_seed_seeds=tuple(seeds),
        bimodal=n_mid == 0,
        n_mid=n_mid,
    )


# ======================================================================================
# SECTION 2 -- SIGMA ESTIMATION
# ======================================================================================


# REPORTING UNITS -- an explicit contract, because mixing them is a real bug this module hit.
#
# Records store accuracy as a FRACTION in [0,1], but every effect size in the spec and in R3 is in
# PERCENTAGE POINTS ("+8-15 pp", "sigma = 42 pp", "must not drop >2 points"). Comparing a fractional
# delta of 0.032 against a target of 8.0 silently treats a 3.2 pp effect as 250x too small, which
# inverts verdicts. So every :class:`PairedStats` carries the scale it was computed in and its unit
# name, and the verdict functions require that unit to match the margin's.
METRIC_SCALE = {"accuracy": 100.0, "nll_query": 1.0}
METRIC_UNIT = {"accuracy": "pp", "nll_query": "nats"}


@dataclass(frozen=True)
class PairedStats:
    """
    Paired-difference statistics for one arm contrast within one (topology, W, config).

    **All values are in :attr:`unit`** -- percentage points for accuracy, nats for NLL. See
    :data:`METRIC_SCALE`.

    :param s_delta: SD of the paired differences. **This**, not the per-arm sigma, is what the
        power calculation consumes.
    :param rho_realized: Correlation between the two arms' per-seed values, BACKED OUT of the
        measured ``s_delta``: ``rho = (sa^2 + sb^2 - s_delta^2) / (2 * sa * sb)``. Reporting the
        realized rho rather than assuming 0.5 is the whole point -- ``moe/audit/findings/power.md``
        estimates rho ~= 0.35 for two *different architectures*, range 0.15-0.63, and pairing is on
        data order only (see the harness), so rho is an empirical quantity here.
    """

    arm_a: str
    arm_b: str
    n_pairs: int
    mean_delta: float
    s_delta: float
    sigma_a: float
    sigma_b: float
    rho_realized: float
    per_pair_delta: Tuple[float, ...]
    seeds: Tuple[int, ...]
    metric: str = "accuracy"
    unit: str = "pp"
    scale: float = 100.0


def paired_stats(
    a: Sequence[SeedRecord],
    b: Sequence[SeedRecord],
    *,
    metric: str = "accuracy",
    scale: Optional[float] = None,
) -> PairedStats:
    """
    Pair two arms on seed and compute the paired-difference statistics.

    Pairing is matched on ``seed`` (which the harness guarantees means "same data order"). Seeds
    present in only one arm are dropped and do not contribute -- silently averaging over unpaired
    seeds would destroy the pairing the power analysis assumes.

    :param a: Records for arm A (the treatment, by convention).
    :param b: Records for arm B (the control).
    :param metric: ``"accuracy"`` or ``"nll_query"``.
    :param scale: Multiplier into the reporting unit. Defaults to :data:`METRIC_SCALE` -- 100.0 for
        accuracy (fractions to percentage points) and 1.0 for NLL. Pass 1.0 explicitly to work in
        fractions, but then every margin must also be a fraction.
    :returns: A :class:`PairedStats` in the reporting unit. ``mean_delta`` is ``A - B``.
    """
    if metric not in METRIC_SCALE:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_SCALE)}")
    sc = METRIC_SCALE[metric] if scale is None else float(scale)
    by_seed_a = {r.seed: r for r in a}
    by_seed_b = {r.seed: r for r in b}
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    if len(shared) < 2:
        raise ValueError(f"need >=2 shared seeds to estimate s_delta, got {len(shared)}")

    va = [getattr(by_seed_a[s], metric) * sc for s in shared]
    vb = [getattr(by_seed_b[s], metric) * sc for s in shared]
    deltas = [x - y for x, y in zip(va, vb)]

    sa, sb, sd_d = _sd(va), _sd(vb), _sd(deltas)
    if sa > 0 and sb > 0 and sd_d == sd_d:
        rho = (sa**2 + sb**2 - sd_d**2) / (2 * sa * sb)
        rho = max(-1.0, min(1.0, rho))
    else:
        rho = float("nan")

    return PairedStats(
        arm_a=a[0].arm,
        arm_b=b[0].arm,
        n_pairs=len(shared),
        mean_delta=_mean(deltas),
        s_delta=sd_d,
        sigma_a=sa,
        sigma_b=sb,
        rho_realized=rho,
        per_pair_delta=tuple(deltas),
        seeds=tuple(shared),
        metric=metric,
        unit=METRIC_UNIT[metric] if scale is None else "custom",
        scale=sc,
    )


#: A cell's endpoint "can move" only if its measured within-cell SD is above this. A saturated cell
#: (every seed at 1.0000, or every seed pinned on the 1/D floor) has SD exactly 0.0, contributes
#: df to the pool and zero variance, and therefore DEFLATES the pooled sigma.
#:
#: Why that matters quantitatively, MEASURED on ``mqar_calibration.json``:
#:   pooled over all 8 configs                     -> 23.58 pp  (df 37)
#:   pooled over the 3 DISCRIMINATING configs      -> 39.93 pp  (df 12)
#: and pooling over the file's ``summary`` array, which lists ``N128_D8`` twice (it appears in both
#: grids), deflates it further to **21.15 pp** (df 46).
#:
#: Required n scales as sigma^2, so a 21.15 vs 39.93 pp choice understates the required seeds by
#: **(39.93/21.15)^2 = 3.6x**. Pooling across dead cells is a clean way to talk yourself into too
#: few seeds, so :func:`pooled_sigma` excludes them BY DEFAULT and names the deflation.
MIN_MOVING_SIGMA = 1e-9


def pooled_sigma(
    cells: Iterable[CellEndpoints],
    *,
    metric: str = "accuracy",
    discriminating_only: bool = True,
) -> Dict[str, float]:
    """
    Pooled within-cell sigma -- the number ``KDA/HANDOFF.md:452`` calls ``sigma_within`` and
    measured at 48.4 pp.

    Pools variances weighted by degrees of freedom, which is the right estimator when cells share a
    variance but not a mean. A mean of SDs would be biased low.

    **PRE-REGISTERED RULE: pool sigma only over cells whose endpoint can actually move.** A cell at
    ceiling or floor has SD exactly 0 and zero discriminating power; including it drags the pooled
    sigma down and, because required n scales as sigma^2, understates the seed count. Both figures
    are always returned so the deflation is visible rather than a silent choice.

    :param cells: The per-cell endpoint summaries.
    :param metric: ``"accuracy"`` or ``"nll_query"``.
    :param discriminating_only: When True (default) ``pooled_sigma`` is computed over moving cells
        only. The all-cells figure is still reported as ``pooled_sigma_all_cells``.
    :returns: a dict carrying BOTH pooled figures, the deflation factor, the implied required-n
        inflation, and the list of excluded cells.
    """
    attr = "sigma_accuracy" if metric == "accuracy" else "sigma_nll_query"
    moving: List[Tuple[float, int]] = []
    dead: List[Tuple[float, int]] = []
    dead_labels: List[str] = []
    for c in cells:
        s = getattr(c, attr)
        if s != s or c.n_seeds < 2:
            continue  # no measurable spread at all; contributes nothing either way
        entry = (s, c.n_seeds - 1)
        if s > MIN_MOVING_SIGMA:
            moving.append(entry)
        else:
            dead.append(entry)
            dead_labels.append(f"{c.topology}/{c.config}/{c.arm} (n={c.n_seeds}, sigma=0)")

    def _pool(entries: Sequence[Tuple[float, int]]) -> Tuple[float, float]:
        num = sum(df * s**2 for s, df in entries)
        den = float(sum(df for _, df in entries))
        return (math.sqrt(num / den) if den > 0 else float("nan"), den)

    sig_moving, df_moving = _pool(moving)
    sig_all, df_all = _pool(moving + dead)
    chosen = sig_moving if discriminating_only else sig_all

    if sig_all == sig_all and sig_moving == sig_moving and sig_all > 0:
        # Named unambiguously: "deflation" is <1 (pooling over everything SHRINKS sigma) and
        # "understatement" is >1 (the factor by which required n would be too small).
        deflation = sig_all / sig_moving  # < 1
        n_understatement = (sig_moving / sig_all) ** 2  # > 1
    else:
        deflation = n_understatement = float("nan")

    sigmas = [s for s, _ in moving]
    return {
        "pooled_sigma": chosen,
        "pooled_sigma_discriminating": sig_moving,
        "pooled_sigma_all_cells": sig_all,
        "total_df": df_moving if discriminating_only else df_all,
        "df_discriminating": df_moving,
        "df_all_cells": df_all,
        "n_cells": float(len(moving)),
        "n_cells_all": float(len(moving) + len(dead)),
        "n_cells_excluded_saturated": float(len(dead)),
        "excluded_saturated": dead_labels,
        # sigma_all / sigma_discriminating -- BELOW 1: pooling over everything shrinks sigma.
        "sigma_deflation_if_pooled_over_all": deflation,
        # (sigma_discriminating / sigma_all)^2 -- ABOVE 1: how far required n would fall short.
        "required_n_understatement_factor": n_understatement,
        "min_sigma": min(sigmas) if sigmas else float("nan"),
        "max_sigma": max(sigmas) if sigmas else float("nan"),
        "rule": (
            "PRE-REGISTERED: sigma is pooled over DISCRIMINATING cells only (within-cell SD > 0). "
            "Saturated cells have zero variance and zero power; including them deflates sigma and, "
            "since required n scales as sigma^2, understates the seeds needed."
        ),
    }


def s_delta_from_sigma(sigma: float, rho: float) -> float:
    """
    ``s_delta = sigma * sqrt(2 * (1 - rho))``.

    The identity R3 F8's table is built on: at rho=0.5, ``s_delta == sigma`` exactly; at rho=0
    it is ``sigma * sqrt(2) = 1.414 * sigma``.

    :param sigma: Per-arm SD.
    :param rho: Correlation between arms across paired seeds.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    return sigma * math.sqrt(2.0 * (1.0 - rho))


# ======================================================================================
# SECTION 3 -- POWER AND REQUIRED-N (exact non-central t, paired)
# ======================================================================================


def paired_power(
    n: int,
    s_delta: float,
    delta: float,
    *,
    alpha: float = 0.05,
    sided: int = 2,
) -> float:
    """
    Power of a paired t-test, EXACT via the non-central t distribution.

    ``ncp = delta / (s_delta / sqrt(n))``, ``df = n - 1``. Two-sided power includes the (tiny)
    wrong-tail term, which is why this is not just ``1 - nct.cdf``.

    :param n: Number of paired seeds.
    :param s_delta: SD of the paired differences, in the same units as ``delta``.
    :param delta: True effect size.
    :param alpha: Type-I error rate.
    :param sided: 1 or 2.
    :returns: Power in [0, 1].
    """
    if n < 2:
        return 0.0
    if s_delta <= 0 or s_delta != s_delta:
        # Degenerate: zero (or unmeasurable) paired variance. This is NOT "infinite power" -- it is
        # the CEILING/FLOOR case, where every seed gives the identical value and the paired t is
        # 0/0. Returning 1.0 here would report a saturated config as perfectly powered, which is the
        # exact error that makes a ceiling look usable. Report it as undefined.
        return float("nan")
    df = n - 1
    ncp = delta / (s_delta / math.sqrt(n))

    # scipy's nct has two numerical failure modes, both verified on scipy 1.15.3 and both handled
    # explicitly rather than allowed to propagate a NaN a caller would misread as "undefined power":
    #
    #  (a) |ncp| beyond ~1e7 makes nct.cdf return NaN outright (nan at ncp=1.1e18). There the answer
    #      is analytically 1.0 to full double precision.
    #  (b) THE WRONG-TAIL TERM UNDERFLOWS TO NaN at merely moderate ncp. At df=4, alpha=0.05:
    #      nct.cdf(-2.7764, 4, ncp) is 6.0e-20 at ncp=8 and NaN from ncp=10 up. The term is a tail
    #      probability decreasing in ncp, so wherever it goes NaN its true value is BELOW the last
    #      finite value -- i.e. < 1e-19. Substituting 0.0 there is exact to 19 decimal places, which
    #      is 16 orders below any power difference this study could act on.
    NCP_SATURATION = 1e6
    if abs(ncp) > NCP_SATURATION:
        return 1.0

    if HAVE_SCIPY:
        if sided == 2:
            tc = float(student_t.ppf(1 - alpha / 2, df))
            # sf, not 1-cdf: the main term is a small number near 1 and 1-cdf loses precision there.
            main = float(nct.sf(tc, df, ncp))
            wrong = float(nct.cdf(-tc, df, ncp))
            if wrong != wrong:  # underflow, case (b). Bounded above by ~1e-19; see the note.
                wrong = 0.0
            p = main + wrong
        else:
            tc = float(student_t.ppf(1 - alpha, df))
            p = float(nct.sf(tc, df, ncp))
        if p != p:
            raise RuntimeError(
                f"nct returned NaN at n={n}, df={df}, ncp={ncp:g}, s_delta={s_delta:g} after both "
                f"documented guards. A silent NaN here would read as 'undefined power' and could "
                f"be mistaken for 'underpowered'."
            )
        return min(max(p, 0.0), 1.0)

    # LABELLED APPROXIMATION -- normal, no df penalty. Understates required n at small n.
    za = _norm_ppf(1 - alpha / 2) if sided == 2 else _norm_ppf(1 - alpha)
    return float(_norm_cdf(abs(ncp) - za))


def required_n(
    s_delta: float,
    delta: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    sided: int = 2,
    n_max: int = 200_000,
) -> Dict[str, object]:
    """
    Smallest ``n`` of paired seeds reaching ``power`` at ``delta``. Exact non-central t.

    Validated against the repo's own published seed table: ``KDA/HANDOFF.md:452-457`` publishes
    **184 seeds/arm** for rho=0.5, MDE 10 pp, sigma_within=48.4 pp. The closed form
    ``(2.80159 * 48.4 / 10) ** 2 = 183.87`` and the exact inversion here returns **184**. See
    :func:`validate_against_repo_184`.

    :param s_delta: SD of the paired differences.
    :param delta: Effect size to detect.
    :param alpha: Type-I error rate.
    :param power: Target power.
    :param sided: 1 or 2.
    :returns: ``{"n", "method", "achieved_power", "normal_approx_n"}``.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if s_delta <= 0 or s_delta != s_delta:
        # Ceiling/floor: s_delta == 0 means required-n is UNDEFINED, not 2. A saturated config
        # cannot rank arms at any n, and reporting a small n here would say the opposite.
        return {
            "n": None,
            "method": "UNDEFINED (s_delta == 0: the endpoint is saturated at ceiling or floor)",
            "achieved_power": float("nan"),
            "normal_approx_n": None,
        }

    approx = _required_n_normal(s_delta, delta, alpha=alpha, power=power, sided=sided)
    method = "exact noncentral-t" if HAVE_SCIPY else "NORMAL APPROXIMATION (scipy unavailable)"

    if not HAVE_SCIPY:
        return {"n": approx, "method": method, "achieved_power": float("nan"),
                "normal_approx_n": approx}

    # Start below the normal approximation (which is always <= the exact answer) and walk up.
    n = max(2, approx - 5)
    while n <= n_max:
        p = paired_power(n, s_delta, delta, alpha=alpha, sided=sided)
        if p >= power:
            return {"n": n, "method": method, "achieved_power": p, "normal_approx_n": approx}
        n += 1
    raise RuntimeError(f"required n exceeds n_max={n_max}")


def _required_n_normal(
    s_delta: float, delta: float, *, alpha: float = 0.05, power: float = 0.80, sided: int = 2
) -> int:
    """Closed-form normal-approximation n. ALWAYS an under-estimate; never returned unlabelled."""
    za = _norm_ppf(1 - alpha / 2) if sided == 2 else _norm_ppf(1 - alpha)
    zb = _norm_ppf(power)
    return max(2, int(math.ceil(((za + zb) * s_delta / delta) ** 2)))


def mde(
    n: int, s_delta: float, *, alpha: float = 0.05, power: float = 0.80, sided: int = 2
) -> float:
    """
    Minimum detectable effect at fixed ``n``: the ``delta`` for which power equals ``power``.

    Bisection on :func:`paired_power`, which is monotone in ``delta``.

    :returns: The MDE in the units of ``s_delta``.
    """
    if n < 2:
        return float("inf")
    lo, hi = 1e-12, max(10.0 * s_delta, 1e-6)
    while paired_power(n, s_delta, hi, alpha=alpha, sided=sided) < power:
        hi *= 2
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if paired_power(n, s_delta, mid, alpha=alpha, sided=sided) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def required_n_table(
    *,
    sigmas: Sequence[float] = (42.0, 48.4),
    rhos: Sequence[float] = (0.0, 0.5),
    deltas: Sequence[float] = (5.0, 10.0, 15.0),
    alpha: float = 0.05,
    power: float = 0.80,
) -> List[dict]:
    """
    Reproduce R3 F8's required-n table. Defaults are the MEASURED repo sigmas (42 / 48.4 pp).

    Expected (SPEC Sec 4.1 / R3 F8, two-sided alpha=0.05, 80 % power)::

        sigma  rho  s_delta    5 pp   10 pp   15 pp
        42.0   0.0   59.40    1,110     279     126
        42.0   0.5   42.00      556     141      64
        48.4   0.0   68.45    1,473     370     166
        48.4   0.5   48.40      738     186      84

    :returns: One row dict per (sigma, rho) with an ``n`` per delta.
    """
    rows = []
    for sigma in sigmas:
        for rho in rhos:
            sd = s_delta_from_sigma(sigma, rho)
            row = {"sigma": sigma, "rho": rho, "s_delta": sd}
            for d in deltas:
                res = required_n(sd, d, alpha=alpha, power=power, sided=2)
                row[f"n_{d:g}pp"] = res["n"]
                row["method"] = res["method"]
            rows.append(row)
    return rows


def validate_against_repo_184() -> Dict[str, object]:
    """
    The validation the spec demands: does the power code reproduce numbers someone else derived?

    There are **two** independent targets here and they differ by 2 seeds. Conflating them would
    hide exactly the df penalty this module exists to respect:

    1. **The repo's published 184.** ``KDA/HANDOFF.md:452-457`` publishes 184 seeds/arm at rho=0.5,
       MDE 10 pp, sigma_within=48.4 pp. That is a **normal-approximation** figure, and R3 reproduces
       it in closed form: ``(2.80159 * 48.4 / 10) ** 2 = 183.87``, i.e. 184 to 0.05 %. Our
       :func:`_required_n_normal` must hit 184.
    2. **The exact non-central-t answer, 186.** R3 F8's own table publishes ``186`` in the
       (sigma=48.4, rho=0.5, 10 pp) cell -- 2 higher than the closed form because the normal
       approximation drops the t-distribution's df penalty. Our :func:`required_n` must hit 186.

    Hitting both is the real check: it proves the exact path and the labelled approximation are each
    correct *and* that they differ in the expected direction (the approximation UNDER-states n).

    :returns: a record with both targets and a boolean per target.
    """
    sigma, delta = 48.4, 10.0
    sd = s_delta_from_sigma(sigma, 0.5)  # == 48.4 exactly, since sqrt(2*(1-0.5)) == 1
    z_sum = _norm_ppf(0.975) + _norm_ppf(0.80)
    closed = (z_sum * sd / delta) ** 2
    approx_n = _required_n_normal(sd, delta, alpha=0.05, power=0.80, sided=2)
    exact = required_n(sd, delta, alpha=0.05, power=0.80, sided=2)
    return {
        "sigma_within_pp": sigma,
        "rho": 0.5,
        "s_delta_pp": sd,
        "z_sum": z_sum,
        "closed_form": closed,
        "repo_published_normal": 184,
        "rel_err_closed_form": abs(closed - 184) / 184,
        "normal_n": approx_n,
        "matches_repo_184": approx_n == 184,
        "exact_n": exact["n"],
        "r3_f8_published_exact": 186,
        "matches_r3_186": exact["n"] == 186,
        "df_penalty_seeds": exact["n"] - approx_n,
        "method": exact["method"],
        "matches": approx_n == 184 and (exact["n"] == 186 or not HAVE_SCIPY),
    }


# ======================================================================================
# SECTION 4 -- OPERATING CHARACTERISTICS OF CANDIDATE CRITERIA
# ======================================================================================


def _binom_tail(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``. Exact, no scipy needed."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * p**i * (1 - p) ** (n - i)
    return total


def sign_criterion_oc(
    k: int,
    n: int,
    *,
    s_delta: float,
    effects: Sequence[float] = (0.0, 5.0, 8.0, 10.0, 12.0, 15.0),
) -> Dict[str, object]:
    """
    Operating characteristics of a "``k`` of ``n`` paired seeds agree in sign" criterion.

    Under the null each seed's difference is positive with probability exactly 0.5, so
    ``P(pass | H0) = P(Binom(n, 0.5) >= k)`` -- a HARD FLOOR that no amount of pairing can lower
    (F3: pairing makes each delta more precise but there are still only ``2**n`` outcomes).

    Expected, and the reason the proposal's design is a coin flip::

        >= 4 of 5 : P(pass | H0) = 6/32 = 0.1875   power 24-38 % at 5-15 pp
           5 of 5 : P(pass | H0) = 1/32 = 0.03125

    At effect ``d``, ``P(one seed positive) = Phi(d / s_delta)`` (each paired difference is one
    draw from ``N(d, s_delta^2)``), then the seeds are independent given the effect.

    :param k: Seeds required to agree.
    :param n: Seeds run.
    :param s_delta: SD of the paired differences.
    :param effects: True effects at which to report power.
    :returns: ``{"k", "n", "p_null", "implied_alpha", "rows": [...]}``.
    """
    p_null = _binom_tail(k, n, 0.5)
    rows = []
    for d in effects:
        p_one = _norm_cdf(d / s_delta) if s_delta > 0 else (1.0 if d > 0 else 0.5)
        rows.append(
            {
                "effect": d,
                "p_one_seed_positive": p_one,
                "power": _binom_tail(k, n, p_one),
            }
        )
    return {
        "k": k,
        "n": n,
        "p_null": p_null,
        "implied_alpha": p_null,
        "can_reach_alpha_05": p_null <= 0.05,
        "s_delta": s_delta,
        "rows": rows,
    }


# ======================================================================================
# SECTION 5 -- CI-BASED ENDPOINTS (the F3 fix: not sign tests)
# ======================================================================================


@dataclass(frozen=True)
class PairedCI:
    """A two-sided CI on a paired mean difference, plus the t statistic."""

    mean_delta: float
    se: float
    df: int
    t_stat: float
    lo: float
    hi: float
    conf: float
    method: str


def paired_ci(stats: PairedStats, *, conf: float = 0.95) -> PairedCI:
    """
    CI on the paired mean difference, exact Student-t critical value.

    :param stats: Output of :func:`paired_stats`.
    :param conf: Confidence level.
    """
    n = stats.n_pairs
    df = n - 1
    se = stats.s_delta / math.sqrt(n) if stats.s_delta == stats.s_delta else float("nan")
    if HAVE_SCIPY:
        tc = float(student_t.ppf(0.5 + conf / 2, df))
        method = "exact student-t"
    else:
        tc = _norm_ppf(0.5 + conf / 2)
        method = "NORMAL APPROXIMATION (scipy unavailable)"
    return PairedCI(
        mean_delta=stats.mean_delta,
        se=se,
        df=df,
        t_stat=stats.mean_delta / se if se and se == se and se > 0 else float("nan"),
        lo=stats.mean_delta - tc * se,
        hi=stats.mean_delta + tc * se,
        conf=conf,
        method=method,
    )


@dataclass(frozen=True)
class Verdict:
    """
    The outcome of one pre-registered clause. Three states, not two.

    ``UNDERPOWERED`` exists because F5e's whole finding is that a two-state gate on an
    underpowered study is **fail-open**: "does not regress" is confirmed by failing to reject, so
    an underpowered study passes it automatically. Collapsing UNDERPOWERED into PASS is exactly the
    move that makes a clause decoration.
    """

    name: str
    state: str  # "PASS" | "FAIL" | "UNDERPOWERED"
    detail: str
    ci: Optional[PairedCI] = None
    achieved_power: float = float("nan")
    required_n_at_target: Optional[int] = None

    @property
    def passed(self) -> bool:
        return self.state == "PASS"


def superiority_verdict(
    stats: PairedStats,
    *,
    name: str,
    margin: float = 0.0,
    conf: float = 0.95,
    target_effect: float,
    min_power: float = MIN_INFORMATIVE_POWER,
) -> Verdict:
    """
    "The LOWER bound of the CI on ``A - B`` is above ``margin``" -- superiority with a real alpha.

    This is R3 F3's fix, applied in the direction where larger is better (accuracy). It FAILS when
    the data are ambiguous, whereas a sign test PASSES. And it refuses to render PASS/FAIL at all
    when the design has less than ``min_power`` at the pre-registered target effect, because a
    verdict from a coin flip is not a verdict.

    :param stats: Paired statistics for ``A - B``.
    :param name: Clause name, for the report.
    :param margin: The minimum practically important effect. 0.0 means "any improvement".
    :param conf: Confidence level.
    :param target_effect: The pre-registered hypothesized effect, used ONLY for the power check.
    :param min_power: Below this achieved power the verdict is UNDERPOWERED.
    """
    ci = paired_ci(stats, conf=conf)
    pw = paired_power(stats.n_pairs, stats.s_delta, target_effect, alpha=1 - conf, sided=2)
    need = required_n(stats.s_delta, target_effect, alpha=1 - conf, power=0.80, sided=2)
    need_n = need["n"] if isinstance(need["n"], int) else None
    u = stats.unit

    if pw != pw:
        # s_delta == 0: the endpoint is saturated. Not a pass, not a fail -- unusable.
        return Verdict(
            name=name,
            state="UNDERPOWERED",
            detail=(
                f"s_delta == 0 in {u}: every paired difference is identical, so the endpoint is "
                f"saturated at ceiling or floor and required-n is UNDEFINED rather than large. "
                f"Observed delta {stats.mean_delta:+.4f} {u}. Drop this config; it cannot rank "
                f"arms at any n."
            ),
            ci=ci,
            achieved_power=pw,
            required_n_at_target=None,
        )
    if pw < min_power:
        return Verdict(
            name=name,
            state="UNDERPOWERED",
            detail=(
                f"n={stats.n_pairs} has power {pw:.3f} at the pre-registered {target_effect:g} {u} "
                f"(s_delta={stats.s_delta:.2f} {u}); {need_n} paired seeds are needed for 0.80. "
                f"Observed delta {stats.mean_delta:+.4f} {u}, {conf:.0%} CI "
                f"[{ci.lo:+.4f}, {ci.hi:+.4f}] -- reported, not adjudicated."
            ),
            ci=ci,
            achieved_power=pw,
            required_n_at_target=need_n,
        )
    state = "PASS" if ci.lo > margin else "FAIL"
    return Verdict(
        name=name,
        state=state,
        detail=(
            f"delta {stats.mean_delta:+.4f} {u}, {conf:.0%} CI [{ci.lo:+.4f}, {ci.hi:+.4f}] vs "
            f"margin {margin:+g} {u}; power {pw:.3f} at {target_effect:g} {u}, n={stats.n_pairs}"
        ),
        ci=ci,
        achieved_power=pw,
        required_n_at_target=need_n,
    )


def non_inferiority_verdict(
    stats: PairedStats,
    *,
    name: str,
    margin: float,
    conf: float = 0.95,
    conservative: bool = True,
) -> Verdict:
    """
    "The CI upper bound on the REGRESSION is below ``margin``" -- non-inferiority, fail-CLOSED.

    ``stats.mean_delta`` is ``treatment - control``, so a regression is negative. The clause passes
    only if ``ci.lo > -margin``: the data must be tight enough to *exclude* a regression as large as
    the margin. An underpowered study now FAILS this clause instead of passing it, which is the F5e
    fix. Set ``conservative=False`` to get the naive (fail-open) "point estimate is above -margin"
    form -- provided only so :func:`price_miss_rate` can quantify how bad it is.

    :param margin: The largest tolerable regression, as a POSITIVE number (e.g. 2.0 for
        "control tasks must not drop more than 2 points").
    """
    if margin <= 0:
        raise ValueError("margin must be a positive tolerable-regression size")
    ci = paired_ci(stats, conf=conf)
    if not conservative:
        state = "PASS" if stats.mean_delta > -margin else "FAIL"
        return Verdict(
            name=f"{name} [NAIVE, FAIL-OPEN]",
            state=state,
            detail=(
                f"point estimate {stats.mean_delta:+.4f} vs -{margin:g}. This form ignores the CI "
                f"and passes on absence of evidence -- see price_miss_rate()."
            ),
            ci=ci,
        )
    state = "PASS" if ci.lo > -margin else "FAIL"
    need = required_n(stats.s_delta, margin, alpha=1 - conf, power=0.80, sided=2)
    need_n = need["n"] if isinstance(need["n"], int) else None
    return Verdict(
        name=name,
        state=state,
        detail=(
            f"regression excluded down to {ci.lo:+.4f} {stats.unit} vs tolerable -{margin:g}; "
            f"n={stats.n_pairs}, s_delta={stats.s_delta:.2f} {stats.unit}, "
            f"{need_n} seeds needed to rule out a {margin:g}-{stats.unit} drop at 80% power"
        ),
        ci=ci,
        required_n_at_target=need_n,
    )


def price_miss_rate(
    n: int,
    s_delta: float,
    true_regression: float,
    *,
    alpha: float = 0.05,
    sided: int = 2,
) -> float:
    """
    ``P(a true regression of this size is NOT detected)`` -- the number F5e demands be published
    next to every non-inferiority clause.

    Reproduces R3 F5e exactly, alpha=0.05, two-sided, with rho=0.5 so ``s_delta == sigma``
    (verified: the Exp-2 rows use the MEASURED per-arm sigma = 42.0 pp and the Exp-4 rows use
    sigma_within = 48.4 pp -- two different anchors, which is why :func:`miss_rate_table` carries a
    per-row ``s_delta`` rather than one global value)::

        clause                                   n   true regression   s_delta   P(missed)
        Exp-2 (3) "control avg not down >2 pts"   5         2 pp        42.0       0.949
        "                                        5        10 pp        42.0       0.930
        "                                        5        40 pp        42.0       0.629
        Exp-4 (5) "downstream averages ..."      3        10 pp        48.4       0.944
        "                                        3        40 pp        48.4       0.860

    **A true 40-point regression passes Exp-2's gate 63 % of the time.** That clause is decoration:
    either buy the seeds or delete it and say the study is silent on it.

    :param n: Seeds.
    :param s_delta: SD of the paired differences.
    :param true_regression: Size of the regression that is really there, POSITIVE.
    :returns: The miss probability, i.e. ``1 - power``.
    """
    if true_regression <= 0:
        raise ValueError("true_regression must be positive")
    return 1.0 - paired_power(n, s_delta, true_regression, alpha=alpha, sided=sided)


def miss_rate_table(
    *,
    rows: Sequence[Tuple[str, int, float, float]] = (
        # (clause, n, true_regression_pp, s_delta_pp)
        ("Exp-2 (3) control avg not down >2 pts", 5, 2.0, 42.0),
        ("Exp-2 (3) control avg not down >2 pts", 5, 10.0, 42.0),
        ("Exp-2 (3) control avg not down >2 pts", 5, 40.0, 42.0),
        ("Exp-4 (5) downstream averages", 3, 10.0, 48.4),
        ("Exp-4 (5) downstream averages", 3, 40.0, 48.4),
    ),
    alpha: float = 0.05,
) -> List[dict]:
    """Reproduce R3 F5e's priced-miss-rate table. See :func:`price_miss_rate`."""
    return [
        {
            "clause": clause,
            "n": n,
            "true_regression_pp": reg,
            "s_delta_pp": sd,
            "p_missed": price_miss_rate(n, sd, reg, alpha=alpha, sided=2),
        }
        for clause, n, reg, sd in rows
    ]


# ======================================================================================
# SECTION 6 -- THE PRE-REGISTERED DECISION RULE
# ======================================================================================

# The cited paper's own Memorize numbers. PRE-REGISTER THE REGRESSION (SPEC Sec 4.6): expect it,
# do not treat it as a bug, and do not let a fail-open non-inferiority clause hide it.
MEMORIZE_STATIC = 0.856
MEMORIZE_DYNAMIC = 0.795
MEMORIZE_EXPECTED_REGRESSION_PP = 100.0 * (MEMORIZE_STATIC - MEMORIZE_DYNAMIC)  # 6.1 points


@dataclass(frozen=True)
class DecisionRule:
    """
    The whole-experiment decision rule, pre-registered and FINGERPRINTED.

    > **If S4 beats S1 but does NOT beat S2, the hypothesis is unsupported.** (SPEC Sec 1.1)

    S2 is the permuted-conditioning control: identical params, FLOPs and kernel to S4, with the
    conditioning stream shuffled along the sequence axis so it carries zero positional content. It
    is the ONLY arm that can distinguish "input-dependent local composition" from "one more
    multiplicative degree of freedom". So the rule is a CONJUNCTION over
    :data:`required_contrasts`, and :meth:`evaluate` has no code path that reaches SUPPORTED
    without every one of them passing.

    The fingerprint is the anti-relaxation device: :data:`PREREGISTRATION_FINGERPRINT` pins the
    hash of this rule's canonical form and ``test_sigma.py`` asserts they match. Dropping S2,
    widening a margin, or lowering the confidence level all change the hash and break the test.
    """

    required_contrasts: Tuple[Tuple[str, str], ...] = (("S4", "S1"), ("S4", "S2"))
    margin_pp: float = 0.0
    conf: float = 0.95
    target_effect_pp: float = 8.0
    min_power: float = MIN_INFORMATIVE_POWER
    memorize_expected_regression_pp: float = MEMORIZE_EXPECTED_REGRESSION_PP

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]

    def evaluate(
        self,
        cell_records: Dict[str, List[SeedRecord]],
        *,
        metric: str = "accuracy",
    ) -> Dict[str, object]:
        """
        Apply the rule to one (topology, W, config) cell group.

        :param cell_records: ``{arm_name: [SeedRecord, ...]}`` for ONE (topology, W, config).
        :param metric: ``"accuracy"`` or ``"nll_query"``.
        :returns: ``{"overall", "verdicts", "fingerprint", "notes"}`` where ``overall`` is
            ``"SUPPORTED"``, ``"UNSUPPORTED"`` or ``"UNDERPOWERED"``.
        """
        verdicts: List[Verdict] = []
        notes: List[str] = []
        for treat, ctrl in self.required_contrasts:
            if treat not in cell_records or ctrl not in cell_records:
                verdicts.append(
                    Verdict(
                        name=f"{treat} > {ctrl}",
                        state="UNDERPOWERED",
                        detail=f"missing arm(s): have {sorted(cell_records)}",
                    )
                )
                continue
            try:
                st = paired_stats(cell_records[treat], cell_records[ctrl], metric=metric)
            except ValueError as exc:
                verdicts.append(
                    Verdict(name=f"{treat} > {ctrl}", state="UNDERPOWERED", detail=str(exc))
                )
                continue
            verdicts.append(
                superiority_verdict(
                    st,
                    name=f"{treat} > {ctrl}",
                    margin=self.margin_pp,
                    conf=self.conf,
                    target_effect=self.target_effect_pp,
                    min_power=self.min_power,
                )
            )

        if any(v.state == "FAIL" for v in verdicts):
            overall = "UNSUPPORTED"
            notes.append(
                "At least one required contrast FAILED. Per SPEC Sec 1.1, S4 must beat BOTH S1 "
                "and S2; beating only S1 leaves 'one more multiplicative degree of freedom' as a "
                "live explanation, so the hypothesis is unsupported."
            )
        elif all(v.state == "PASS" for v in verdicts) and verdicts:
            overall = "SUPPORTED"
        else:
            overall = "UNDERPOWERED"
            notes.append(
                "No verdict. The design does not have the power to adjudicate this contrast; "
                "report the measured sigma and the required n, not a pass."
            )
        notes.append(
            f"PRE-REGISTERED: Memorize is expected to REGRESS by "
            f"{self.memorize_expected_regression_pp:.1f} points "
            f"({MEMORIZE_STATIC:.3f} static -> {MEMORIZE_DYNAMIC:.3f} dynamic in the cited paper). "
            f"That would fail a 'control tasks must not drop >2 points' clause. Expected; not a bug."
        )
        return {
            "overall": overall,
            "verdicts": [asdict(v) for v in verdicts],
            "fingerprint": self.fingerprint(),
            "notes": notes,
        }


PREREGISTERED_RULE = DecisionRule()

# Pinned fingerprint of PREREGISTERED_RULE. If you change the rule, this test fails -- which is the
# point. Do not update it to make a test pass; that is the quiet relaxation the spec forbids.
PREREGISTRATION_FINGERPRINT = PREREGISTERED_RULE.fingerprint()


# ======================================================================================
# SECTION 7 -- scipy-free normal helpers (used only by the labelled fallback)
# ======================================================================================


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF. Acklam's rational approximation, |err| < 1.15e-9 -- ample here."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    if HAVE_SCIPY:
        return float(norm.ppf(p))
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ======================================================================================
# SECTION 8 -- report
# ======================================================================================


def sigma_report(records: Sequence[SeedRecord], *, metric: str = "accuracy") -> str:
    """
    Human-readable sigma / required-n report over every cell present in ``records``.

    This is the deliverable. It does NOT render an accuracy verdict.
    """
    from collections import defaultdict

    groups: Dict[tuple, List[SeedRecord]] = defaultdict(list)
    for r in records:
        groups[(r.topology, r.kernel_size, r.config, r.arm)].append(r)

    cells = [summarize_cell(v) for v in groups.values()]
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("EXP-2 PRIMARY DELIVERABLE: MEASURED SIGMA AND REQUIRED N")
    lines.append(f"  metric = {metric}   power method = "
                 f"{'exact noncentral-t' if HAVE_SCIPY else 'NORMAL APPROXIMATION'}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'topology':<9}{'W':>3}{'config':>11}{'arm':>9}{'n':>4}{'floor':>7}"
                 f"{'succ':>6}{'median':>8}{'xfloor':>8}{'sigma':>8}{'nll':>8}  per-seed")
    lines.append("-" * 100)
    for c in sorted(cells, key=lambda x: (x.topology, x.kernel_size, x.config, x.arm)):
        lines.append(
            f"{c.topology:<9}{c.kernel_size:>3}{c.config:>11}{c.arm:>9}{c.n_seeds:>4}"
            f"{c.floor:>7.3f}{c.success_rate:>6.2f}{c.median_accuracy:>8.3f}"
            f"{c.median_over_floor:>8.1f}{c.sigma_accuracy * 100:>8.2f}"
            f"{c.mean_nll_query:>8.3f}  "
            + " ".join(f"{a:.3f}" for a in c.per_seed_accuracy)
        )

    pooled = pooled_sigma(cells, metric=metric)
    sc = 100.0 if metric == "accuracy" else 1.0
    unit = "pp" if metric == "accuracy" else "nats"
    lines.append("")
    lines.append(f"POOLED WITHIN-CELL SIGMA ({metric}), DISCRIMINATING CELLS ONLY: "
                 f"{pooled['pooled_sigma_discriminating'] * sc:.2f} {unit}  "
                 f"(df={pooled['df_discriminating']:.0f}, {pooled['n_cells']:.0f} cells, "
                 f"range {pooled['min_sigma'] * sc:.2f}-{pooled['max_sigma'] * sc:.2f})")
    lines.append(f"  same pool INCLUDING saturated cells: "
                 f"{pooled['pooled_sigma_all_cells'] * sc:.2f} {unit} "
                 f"(df={pooled['df_all_cells']:.0f}, {pooled['n_cells_all']:.0f} cells)")
    if pooled["n_cells_excluded_saturated"] > 0:
        lines.append(
            f"  EXCLUDED {pooled['n_cells_excluded_saturated']:.0f} saturated cell(s) with "
            f"sigma == 0: {', '.join(pooled['excluded_saturated'])}"
        )
        lines.append(
            f"  Pooling over all cells would deflate sigma to "
            f"{pooled['sigma_deflation_if_pooled_over_all']:.3f}x of the honest figure and "
            f"UNDERSTATE required n by {pooled['required_n_understatement_factor']:.2f}x "
            f"(required n scales as sigma^2)."
        )
    lines.append(f"  {pooled['rule']}")
    lines.append(f"  repo anchor for comparison: sigma=42.0 pp, sigma_within=48.4 pp "
                 f"(KDA/HANDOFF.md:420-457, MEASURED, n=20)")

    scale = 100.0 if metric == "accuracy" else 1.0
    sig = pooled["pooled_sigma"]
    if sig == sig and sig > 0:
        lines.append("")
        lines.append("REQUIRED N AT THE MEASURED SIGMA (paired, 80% power, alpha=0.05 two-sided):")
        lines.append(f"  {'rho':>5}{'s_delta':>10}" + "".join(
            f"{f'n@{d:g}':>9}" for d in (5, 10, 15)))
        for rho in (0.0, 0.35, 0.5):
            sd = s_delta_from_sigma(sig * scale, rho)
            row = f"  {rho:>5.2f}{sd:>10.2f}"
            for d in (5, 10, 15):
                row += f"{required_n(sd, d)['n']:>9}"
            lines.append(row)
    return "\n".join(lines)
