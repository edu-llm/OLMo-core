#!/usr/bin/env python3
"""Independent re-derivation of every headline probe statistic for the KDA + Householder write-up.

This script is the reproduction path shipped with the write-up. It reads ONLY the raw probe
JSON artifacts and recomputes every number from scratch. It deliberately does not import, reuse,
or trust the original ``analyze_*.py`` scripts; where this script and those scripts disagree, the
disagreement is reported rather than reconciled.

Design decisions that matter for correctness
--------------------------------------------
1. ``n_layers`` is NOT stored in the probe JSON records. It survives only in the filename as
   ``-L<n>-``. The depth x R grid (L1/L2/L4) shares one flat directory with the 3-layer
   confirmatory grid, so every load path here parses L from the filename and filters on it
   explicitly. Failing to do so silently pools 1- and 2-layer runs into 3-layer cells.
2. All tests are paired-by-seed, df = n - 1, two-sided alpha = 0.05. Exact p-values are computed
   from the Student t distribution via the regularized incomplete beta function (implemented
   below, since scipy is not available in the probe venv). The original scripts used a hardcoded
   critical-value table; ``verify_t_crit_table`` checks that table against these exact quantiles.
3. Degenerate cells. When every seed yields the identical paired difference, sd = 0 and the t
   statistic is undefined (0/0 or x/0). The original ``report()`` returns ``dz = inf`` and the
   verdict ``SIG`` whenever sd == 0 and mean != 0. That is not a valid test -- it is a division by
   an estimated zero variance from a finite sample. This script labels those cells
   ``DEGENERATE_SD0`` and reports them separately instead of calling them significant.
4. Multiplicity. Every test performed is appended to a global registry with a family label, so
   Holm-Bonferroni can be applied within family and globally. Nothing in the original analysis
   corrected for multiplicity.
5. Non-independence. The 7 eval lengths are evaluations of the SAME trained model per seed, so the
   per-length tests are not 7 independent findings. The cross-length correlation of per-seed
   effects is computed and emitted so the write-up can state this quantitatively.

Determinism: all globs are sorted, all seed lists are sorted, no RNG is used.

Usage::

    python verify_claims.py [--results-root DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

#: The seven eval lengths present in the ``all_night`` grids.
LENGTHS_7 = ["40", "64", "128", "256", "512", "1024", "2048"]
#: The five eval lengths present in every other results family (backend equivalence, KDA-vs-GDN,
#: the P3 depth ladder). These runs predate the 1024/2048 additions.
LENGTHS_5 = ["40", "64", "128", "256", "512"]

ALPHA = 0.05

#: The hardcoded two-sided t critical values used by the original analysis scripts, reproduced
#: verbatim so they can be checked against exact quantiles.
AUTHOR_T_CRIT = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    14: 2.145,
    15: 2.131,
}

#: Tolerance (in percentage points) within which a recomputed value is considered to match a
#: claimed value that was printed to two decimal places.
MATCH_TOL = 0.005

# ---------------------------------------------------------------------------------------------
# Statistics primitives (no scipy in the probe venv)
# ---------------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function (Lentz's method).

    :returns: The continued fraction evaluated at ``x``.
    """
    tiny, eps, itmax = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function :math:`I_x(a, b)`.

    :returns: ``I_x(a, b)`` in [0, 1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_pvalue_two_sided(t: float, df: int) -> float:
    """Exact two-sided p-value for a Student t statistic.

    :param t: The t statistic.
    :param df: Degrees of freedom.

    :returns: ``P(|T| >= |t|)``.
    """
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return betai(0.5 * df, 0.5, df / (df + t * t))


def t_crit_two_sided(df: int, alpha: float = ALPHA) -> float:
    """Exact two-sided t critical value, obtained by bisecting :func:`t_pvalue_two_sided`.

    :returns: ``t*`` such that ``P(|T| >= t*) == alpha``.
    """
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_pvalue_two_sided(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _sym_eigenvalues(mat: Sequence[Sequence[float]], iters: int = 2000) -> List[float]:
    """Eigenvalues of a small real symmetric matrix via the cyclic Jacobi rotation method.

    Used for the effective-number-of-independent-tests statistic. Implemented here so the script
    has no numpy/scipy dependency.

    :returns: The eigenvalues, unordered.
    """
    n = len(mat)
    a = [[float(mat[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(iters):
        off = max(
            ((abs(a[i][j]), i, j) for i in range(n) for j in range(i + 1, n)),
            default=(0.0, 0, 0),
        )
        if off[0] < 1e-14:
            break
        _, p, q = off
        if a[p][p] == a[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * a[p][q], a[p][p] - a[q][q])
        c, s = math.cos(theta), math.sin(theta)
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p], a[k][q] = c * akp + s * akq, -s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k], a[q][k] = c * apk + s * aqk, -s * apk + c * aqk
    return [a[i][i] for i in range(n)]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """:returns: Pearson correlation of two equal-length samples, or nan if either is constant."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


# ---------------------------------------------------------------------------------------------
# Test registry: every hypothesis test performed lands here, for the multiplicity analysis.
# ---------------------------------------------------------------------------------------------

#: (family, test_id, p_value, verdict, degenerate_flag)
TEST_REGISTRY: List[Tuple[str, str, float, str, bool]] = []


class Result:
    """A one-sample (paired-difference) t-test result."""

    def __init__(self, diffs: Sequence[float]):
        self.n = len(diffs)
        self.diffs = list(diffs)
        self.mean = st.mean(diffs) if self.n >= 1 else float("nan")
        self.sd = st.stdev(diffs) if self.n >= 2 else float("nan")
        if self.n < 2:
            self.degenerate, self.se, self.t, self.p = (
                True,
                float("nan"),
                float("nan"),
                float("nan"),
            )
            self.ci_lo = self.ci_hi = float("nan")
            self.dz = float("nan")
            self.verdict = "INSUFFICIENT_N"
            return
        if self.sd == 0.0:
            # Zero sample variance. The t statistic is undefined; do not manufacture one.
            self.degenerate = True
            self.se = 0.0
            self.t = float("nan")
            self.ci_lo = self.ci_hi = self.mean
            self.dz = float("nan")
            if self.mean == 0.0:
                # Literally identical in every seed: no difference at all.
                self.p = 1.0
                self.verdict = "ns_IDENTICAL"
            else:
                self.p = float("nan")
                self.verdict = "DEGENERATE_SD0"
            return
        self.degenerate = False
        self.se = self.sd / math.sqrt(self.n)
        self.t = self.mean / self.se
        self.p = t_pvalue_two_sided(self.t, self.n - 1)
        half = t_crit_two_sided(self.n - 1) * self.se
        self.ci_lo, self.ci_hi = self.mean - half, self.mean + half
        self.dz = self.mean / self.sd
        self.verdict = "SIG" if self.p < ALPHA else "ns"

    def author_verdict(self) -> str:
        """Reproduce the ORIGINAL scripts' verdict, including their sd == 0 shortcut.

        :returns: ``SIG`` or ``ns`` exactly as ``analyze_*.py`` would have printed it.
        """
        if self.n < 2:
            return "n/a"
        if self.sd == 0.0:
            return "SIG" if self.mean != 0.0 else "ns"
        half = AUTHOR_T_CRIT.get(self.n - 1, 2.0) * self.sd / math.sqrt(self.n)
        return "SIG" if abs(self.mean) > half else "ns"

    def register(self, family: str, test_id: str) -> "Result":
        """Record this test in the global multiplicity registry.

        :returns: ``self``, for chaining.
        """
        TEST_REGISTRY.append((family, test_id, self.p, self.verdict, self.degenerate))
        return self


def fmt(x: float, nd: int = 4) -> str:
    """:returns: A stable, locale-independent string for a float (``NA`` for nan)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def pfmt(p: float) -> str:
    """:returns: A p-value string with enough precision to be useful, or ``NA``/``<1e-12``."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NA"
    if p < 1e-12:
        return "<1e-12"
    return f"{p:.3e}"


def status(claimed: Optional[float], got: float, tol: float = MATCH_TOL) -> str:
    """Compare a claimed headline number to the recomputed one.

    :returns: ``EXACT`` (agrees to printed precision), ``CLOSE``, ``MISMATCH``, or ``NO_CLAIM``.
    """
    if claimed is None:
        return "NO_CLAIM"
    if got is None or (isinstance(got, float) and math.isnan(got)):
        return "MISMATCH"
    d = abs(claimed - got)
    if d <= tol:
        return "EXACT"
    if d <= 0.05:
        return "CLOSE"
    return "MISMATCH"


# ---------------------------------------------------------------------------------------------
# Claimed values, transcribed from the handoff (HANDOFF.md) for automated comparison.
# ---------------------------------------------------------------------------------------------

CLAIMED_S5 = {
    1: dict(zip(LENGTHS_7, [91.93, 64.34, 32.87, 16.81, 8.66, 4.84, 2.85])),
    2: dict(zip(LENGTHS_7, [99.37, 86.53, 46.75, 23.67, 12.23, 6.48, 3.68])),
    4: dict(zip(LENGTHS_7, [100.00, 99.96, 88.60, 48.12, 24.51, 12.79, 6.78])),
}

# length -> (s5 effect, parity effect, interaction). len40 is ABSENT from the handoff table even
# though the prose claims significance "at all seven lengths"; encoded as None to flag the gap.
CLAIMED_INTERACTION = {
    "40": (None, None, None),
    "64": (35.63, -0.00, 35.63),
    "128": (55.74, -2.42, 58.15),
    "256": (31.31, -2.63, 33.95),
    "512": (15.85, -2.10, 17.95),
    "1024": (7.96, -0.79, 8.75),
    "2048": (3.93, -1.15, 5.09),
}
CLAIMED_INTERACTION_CI = {"128": (48.2, 68.1)}
CLAIMED_S5_CI = {"128": (47.5, 63.9)}

# task -> (acc@2048 at R=1, R4-R1 @128, R4-R1 @2048)
CLAIMED_SOLV = {
    "parity": (54.55, -2.42, -1.15),
    "s3_words": (21.96, 4.57, 0.78),
    "s4_words": (13.05, 4.77, -0.99),
    "s5_words": (2.85, 55.74, 3.93),
}

# length -> (L1 effect, L2 effect, L4 effect, substitution contrast)
CLAIMED_DEPTH = {
    "40": (61.69, 24.95, 3.37, 58.32),
    "64": (75.97, 46.28, 25.64, 50.32),
    "128": (68.10, 52.80, 48.48, 19.62),
    "256": (None, None, None, None),
    "512": (18.59, 14.48, 13.42, 5.17),
    "1024": (None, None, None, None),
    "2048": (4.70, 3.59, 3.31, 1.39),
}

CLAIMED_LADDER = {1: 39.0, 2: 72.7, 4: 97.6, 6: 99.0}

CLAIMED_BEQ_PER_SEED = {
    0: {"40": 0.00, "64": 0.00, "128": -4.94, "256": -3.70, "512": -2.51, "mean": -2.79},
    1: {"40": 0.00, "64": 0.00, "128": -5.75, "256": -5.19, "512": -1.72, "mean": -3.17},
    2: {"40": 0.00, "64": 0.00, "128": 1.64, "256": 1.38, "512": 1.76, "mean": 1.19},
    3: {"40": 0.00, "64": 0.00, "128": -4.99, "256": -3.52, "512": -1.62, "mean": -2.53},
    4: {"40": 0.00, "64": -0.29, "128": -8.54, "256": -4.28, "512": -2.24, "mean": -3.84},
    5: {"40": 0.00, "64": 0.00, "128": 1.45, "256": 2.17, "512": 1.34, "mean": 1.24},
    6: {"40": 0.00, "64": 0.00, "128": -2.84, "256": -2.95, "512": -2.38, "mean": -2.04},
    7: {"40": 0.00, "64": 0.00, "128": 3.59, "256": 2.65, "512": 1.71, "mean": 1.99},
}
CLAIMED_BEQ_POOLED = (-1.24, -3.18, 0.70)
CLAIMED_BEQ_SIGN = (5, 8, 0.73)

# KDA-vs-GDN claims, as (task, length, claimed pp, claimed verdict).
#
# SIGN CONVENTION. The handoff writes "parity@512 -7.43pp SIG against KDA" and "mod_arith@40
# -1.30pp SIG against KDA". Recomputation shows -7.4314 and -1.3021 are exactly KDA minus GDN,
# so the handoff's numbers are KDA - GDN and the phrase "against KDA" merely glosses the negative
# sign. The claims are therefore compared against the KDA - GDN column.
CLAIMED_KDA_GDN = [
    ("s5_words", None, 2.01, "ns"),
    ("parity", "512", -7.43, "SIG"),
    ("mod_arith", "40", -1.30, "SIG"),
]

# ---------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------


def load_all_night(root: str) -> Dict[Tuple[int, str, int, int], dict]:
    """Load the ``all_night`` grids, keyed by ``(R, task, n_layers, seed)``.

    ``n_layers`` is parsed from the filename because the JSON does not contain it.

    :returns: Mapping from ``(num_householder, task, n_layers, seed)`` to the raw record.
    """
    out: Dict[Tuple[int, str, int, int], dict] = {}
    pattern = os.path.join(root, "all_night", "*.json")
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)
        m = re.search(r"-L(\d+)-s(\d+)\.json$", base)
        if m is None:
            raise ValueError(f"unparseable all_night filename (no -L<n>-s<k>): {base}")
        layers, seed_from_name = int(m.group(1)), int(m.group(2))
        rec = json.load(open(f))
        if rec["seed"] != seed_from_name:
            raise ValueError(f"seed mismatch filename vs record in {base}")
        key = (rec["num_householder"], rec["task"], layers, rec["seed"])
        if key in out:
            raise ValueError(f"duplicate cell {key} at {base}")
        rec["_file"] = base
        out[key] = rec
    return out


def load_flat(root: str, prefix: str) -> Dict[int, dict]:
    """Load a flat ``results/<prefix>-s<seed>.json`` family.

    :returns: Mapping from seed to raw record.
    """
    out: Dict[int, dict] = {}
    for f in sorted(glob.glob(os.path.join(root, f"{prefix}-s*.json"))):
        m = re.search(r"-s(\d+)\.json$", os.path.basename(f))
        if m is None:
            continue
        rec = json.load(open(f))
        rec["_file"] = os.path.basename(f)
        out[int(m.group(1))] = rec
    return out


def load_beq(root: str) -> Dict[int, Dict[str, dict]]:
    """Load the backend-equivalence family, keyed by ``seed`` then ``backend``.

    :returns: Mapping from seed to ``{"triton": rec, "torch": rec}``.
    """
    out: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(root, "beq", "*.json"))):
        rec = json.load(open(f))
        rec["_file"] = os.path.basename(f)
        out[rec["seed"]][rec["backend"]] = rec
    return dict(out)


def acc(rec: dict, L: str) -> float:
    """:returns: Accuracy in percentage points at eval length ``L``."""
    return 100.0 * rec["accuracy_by_length"][L]


def paired_diff(
    D: dict, key_hi, key_lo, L: str, seeds: Sequence[int]
) -> Tuple[List[float], List[int]]:
    """Per-seed paired difference (hi minus lo) in pp at one eval length.

    ``key_hi``/``key_lo`` are callables mapping a seed to a dict key.

    :returns: ``(differences, seeds actually used)``.
    """
    used = [s for s in sorted(seeds) if key_hi(s) in D and key_lo(s) in D]
    diffs = [acc(D[key_hi(s)], L) - acc(D[key_lo(s)], L) for s in used]
    return diffs, used


# ---------------------------------------------------------------------------------------------
# Section 0: T_CRIT audit
# ---------------------------------------------------------------------------------------------


def verify_t_crit_table(out_dir: str) -> List[str]:
    """Check the original scripts' hardcoded critical values against exact t quantiles.

    :returns: Human-readable log lines.
    """
    rows = ["df\tauthor_t_crit\texact_t_crit\tabs_error\tstatus"]
    log = []
    for df in sorted(AUTHOR_T_CRIT):
        exact = t_crit_two_sided(df)
        err = abs(AUTHOR_T_CRIT[df] - exact)
        stat = "OK" if err < 5e-4 else "OFF"
        rows.append(f"{df}\t{AUTHOR_T_CRIT[df]:.3f}\t{exact:.6f}\t{err:.2e}\t{stat}")
        if stat != "OK":
            log.append(f"T_CRIT df={df}: author {AUTHOR_T_CRIT[df]} vs exact {exact:.6f}")
    # The original tables are also INCOMPLETE: df 12, 13 are missing, and analyze_solv.py stops at
    # df=8. A missing df silently falls back to 2.0, which is anti-conservative for small n.
    missing = [d for d in range(1, 16) if d not in AUTHOR_T_CRIT]
    rows.append(f"#missing_df_in_author_table\t{','.join(map(str, missing))}\t\t\tFALLBACK_2.0")
    write_tsv(out_dir, "probe_t_crit_audit.tsv", rows)
    log.append(
        f"T_CRIT: {len(AUTHOR_T_CRIT)} entries checked, all within 5e-4 of exact"
        if not log
        else "T_CRIT has discrepancies (see above)"
    )
    log.append(f"T_CRIT missing df in author table (fall back to 2.0): {missing}")
    return log


def write_tsv(out_dir: str, name: str, rows: Sequence[str]) -> str:
    """Write a TSV file.

    :returns: The absolute path written.
    """
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


# ---------------------------------------------------------------------------------------------
# Section 1: S5 accuracy table (n=8, L3)
# ---------------------------------------------------------------------------------------------


def do_s5_table(D: dict, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the S5 R x length mean-accuracy table and the R=2/R=4 vs R=1 tests."""
    rows = ["R\tlength\tn\tmean_acc_pp\tsd_pp\tmin_pp\tmax_pp\tclaimed_pp\tabs_delta\tstatus"]
    seeds = range(8)
    for r in (1, 2, 4):
        have = [s for s in seeds if (r, "s5_words", 3, s) in D]
        for L in LENGTHS_7:
            vs = [acc(D[(r, "s5_words", 3, s)], L) for s in have]
            m = st.mean(vs)
            c = CLAIMED_S5[r][L]
            stat = status(c, m)
            rows.append(
                f"{r}\t{L}\t{len(vs)}\t{fmt(m)}\t{fmt(st.stdev(vs) if len(vs)>1 else 0.0)}\t"
                f"{fmt(min(vs))}\t{fmt(max(vs))}\t{fmt(c,2)}\t{fmt(abs(c-m))}\t{stat}"
            )
            claims.append((f"S5_acc_R{r}_len{L}", c, m, stat))
    write_tsv(out_dir, "probe_s5_table.tsv", rows)

    # "R=2 and R=4 are both SIG at every length."
    rows2 = ["contrast\tlength\tn\tmean_pp\tsd_pp\tci_lo\tci_hi\tdz\tt\tp\tverdict\tauthor_verdict"]
    for r in (2, 4):
        for L in LENGTHS_7:
            d, used = paired_diff(
                D, lambda s, r=r: (r, "s5_words", 3, s), lambda s: (1, "s5_words", 3, s), L, seeds
            )
            res = Result(d).register("F_S5_R_effects", f"S5_R{r}vs1_len{L}")
            rows2.append(
                f"R{r}-R1\t{L}\t{res.n}\t{fmt(res.mean)}\t{fmt(res.sd)}\t{fmt(res.ci_lo)}\t"
                f"{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t{pfmt(res.p)}\t{res.verdict}\t"
                f"{res.author_verdict()}"
            )
            claims.append((f"S5_R{r}vs1_len{L}_verdict", None, res.mean, res.verdict))
    write_tsv(out_dir, "probe_s5_r_effects.tsv", rows2)


# ---------------------------------------------------------------------------------------------
# Section 2: the headline interaction
# ---------------------------------------------------------------------------------------------


def do_interaction(D: dict, out_dir: str, claims: List[tuple]) -> None:
    """Recompute (R4-R1 on S5) - (R4-R1 on parity), paired by seed, at all SEVEN lengths."""
    rows = [
        "length\tn\ts5_effect\ts5_p\ts5_verdict\tparity_effect\tparity_sd\tparity_p\t"
        "parity_verdict\tinteraction\tci_lo\tci_hi\tdz\tt\tp\tverdict\tauthor_verdict\t"
        "claimed_interaction\tabs_delta\tstatus"
    ]
    seeds = range(8)
    per_length_effects: Dict[str, List[float]] = {}
    for L in LENGTHS_7:
        d5, s5used = paired_diff(
            D, lambda s: (4, "s5_words", 3, s), lambda s: (1, "s5_words", 3, s), L, seeds
        )
        dp, spused = paired_diff(
            D, lambda s: (4, "parity", 3, s), lambda s: (1, "parity", 3, s), L, seeds
        )
        r5 = Result(d5).register("F_S5_side", f"S5_R4vs1_len{L}")
        rp = Result(dp).register("F_parity_side", f"parity_R4vs1_len{L}")
        common = sorted(set(s5used) & set(spused))
        m5, mp = dict(zip(s5used, d5)), dict(zip(spused, dp))
        inter = [m5[s] - mp[s] for s in common]
        per_length_effects[L] = inter
        ri = Result(inter).register("F_interaction", f"interaction_len{L}")
        c5, cp, ci_ = CLAIMED_INTERACTION[L]
        stat = status(ci_, ri.mean)
        rows.append(
            f"{L}\t{ri.n}\t{fmt(r5.mean)}\t{pfmt(r5.p)}\t{r5.verdict}\t{fmt(rp.mean)}\t"
            f"{fmt(rp.sd)}\t{pfmt(rp.p)}\t{rp.verdict}\t{fmt(ri.mean)}\t{fmt(ri.ci_lo)}\t"
            f"{fmt(ri.ci_hi)}\t{fmt(ri.dz)}\t{fmt(ri.t)}\t{pfmt(ri.p)}\t{ri.verdict}\t"
            f"{ri.author_verdict()}\t{fmt(ci_,2) if ci_ is not None else 'NOT_IN_HANDOFF'}\t"
            f"{fmt(abs(ci_-ri.mean)) if ci_ is not None else 'NA'}\t{stat}"
        )
        claims.append((f"interaction_len{L}", ci_, ri.mean, stat))
        claims.append((f"S5side_len{L}", c5, r5.mean, status(c5, r5.mean)))
        claims.append((f"parityside_len{L}", cp, rp.mean, status(cp, rp.mean)))
        if L in CLAIMED_INTERACTION_CI:
            lo, hi = CLAIMED_INTERACTION_CI[L]
            claims.append((f"interaction_len{L}_ci_lo", lo, ri.ci_lo, status(lo, ri.ci_lo, 0.05)))
            claims.append((f"interaction_len{L}_ci_hi", hi, ri.ci_hi, status(hi, ri.ci_hi, 0.05)))
        if L in CLAIMED_S5_CI:
            lo, hi = CLAIMED_S5_CI[L]
            claims.append((f"S5side_len{L}_ci_lo", lo, r5.ci_lo, status(lo, r5.ci_lo, 0.05)))
            claims.append((f"S5side_len{L}_ci_hi", hi, r5.ci_hi, status(hi, r5.ci_hi, 0.05)))
    write_tsv(out_dir, "probe_interaction.tsv", rows)

    # Non-independence: correlate the per-seed interaction effect across lengths, and do the same
    # for the S5-side effect (which is what "SIG at all seven lengths" is actually about).
    s5_effects: Dict[str, List[float]] = {}
    for L in LENGTHS_7:
        d5, s5used = paired_diff(
            D, lambda s: (4, "s5_words", 3, s), lambda s: (1, "s5_words", 3, s), L, seeds
        )
        s5_effects[L] = d5
    rows3 = ["contrast\tlength_a\tlength_b\tn\tpearson_r"]
    for label, series in (("interaction", per_length_effects), ("S5_R4vs1", s5_effects)):
        for i, a in enumerate(LENGTHS_7):
            for b in LENGTHS_7[i + 1 :]:
                rows3.append(
                    f"{label}\t{a}\t{b}\t{len(series[a])}\t" f"{fmt(pearson(series[a], series[b]))}"
                )
    rows3.append("#--- effective number of independent tests across the 7 lengths ---")
    rows3.append("contrast\tmean_abs_r\tmean_r\teig_spectrum\tn_eff_cheverud\tn_eff_participation")
    for label, series in (("interaction", per_length_effects), ("S5_R4vs1", s5_effects)):
        k = len(LENGTHS_7)
        corr = [[pearson(series[a], series[b]) for b in LENGTHS_7] for a in LENGTHS_7]
        offdiag = [corr[i][j] for i in range(k) for j in range(k) if i != j]
        mean_r = st.mean(offdiag)
        mean_abs = st.mean([abs(x) for x in offdiag])
        eigs = sorted(_sym_eigenvalues(corr), reverse=True)
        # Cheverud/Nyholt: n_eff = 1 + (k-1)*(1 - var(eig)/k).
        var_eig = st.pvariance(eigs)
        n_eff_chev = 1.0 + (k - 1) * (1.0 - var_eig / k)
        # Participation ratio of the eigenvalue spectrum: (sum eig)^2 / sum(eig^2).
        n_eff_part = sum(eigs) ** 2 / sum(e * e for e in eigs)
        rows3.append(
            f"{label}\t{fmt(mean_abs)}\t{fmt(mean_r)}\t"
            f"{'|'.join(fmt(e, 3) for e in eigs)}\t{fmt(n_eff_chev, 2)}\t{fmt(n_eff_part, 2)}"
        )
    write_tsv(out_dir, "probe_length_dependence.tsv", rows3)


# ---------------------------------------------------------------------------------------------
# Section 3: solvability control -- ALL SEVEN lengths for every task
# ---------------------------------------------------------------------------------------------


def do_solvability(D: dict, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the R4-R1 effect for parity/S3/S4/S5 at every length, checking the universal."""
    tasks = [
        ("parity", "yes"),
        ("s3_words", "yes"),
        ("s4_words", "yes"),
        ("s5_words", "NO"),
    ]
    rows = [
        "task\tsolvable\tlength\tn\tr1_mean_acc_pp\tr4_mean_acc_pp\teffect_pp\tsd_pp\tci_lo\t"
        "ci_hi\tdz\tt\tp\tverdict\tauthor_verdict\tseeds"
    ]
    for task, solv in tasks:
        seeds_all = sorted(s for s in range(8) if (4, task, 3, s) in D and (1, task, 3, s) in D)
        for L in LENGTHS_7:
            d, used = paired_diff(
                D, lambda s, t=task: (4, t, 3, s), lambda s, t=task: (1, t, 3, s), L, seeds_all
            )
            res = Result(d).register("F_solvability", f"{task}_R4vs1_len{L}")
            r1m = st.mean([acc(D[(1, task, 3, s)], L) for s in used])
            r4m = st.mean([acc(D[(4, task, 3, s)], L) for s in used])
            rows.append(
                f"{task}\t{solv}\t{L}\t{res.n}\t{fmt(r1m)}\t{fmt(r4m)}\t{fmt(res.mean)}\t"
                f"{fmt(res.sd)}\t{fmt(res.ci_lo)}\t{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t"
                f"{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t"
                f"{','.join(map(str,used))}"
            )
            if task in CLAIMED_SOLV:
                _, c128, c2048 = CLAIMED_SOLV[task]
                if L == "128":
                    claims.append((f"solv_{task}_eff@128", c128, res.mean, status(c128, res.mean)))
                if L == "2048":
                    claims.append(
                        (f"solv_{task}_eff@2048", c2048, res.mean, status(c2048, res.mean))
                    )
        # Baseline difficulty at R=1, acc@2048.
        c_acc, _, _ = CLAIMED_SOLV[task]
        got = st.mean([acc(D[(1, task, 3, s)], "2048") for s in seeds_all])
        claims.append((f"solv_{task}_R1_acc@2048", c_acc, got, status(c_acc, got)))
    write_tsv(out_dir, "probe_solvability.tsv", rows)


# ---------------------------------------------------------------------------------------------
# Section 4: depth x R and the substitution contrast
# ---------------------------------------------------------------------------------------------


def do_depth_r(D: dict, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the depth x R grid and the (L=1)-(L=4) substitution contrast at all 7 lengths.

    Also computes the L=3 arm, which the original ``analyze_depthr.py`` omits entirely even though
    L=3 s5_words data exists (n=8 in the confirmatory grid). Both an n=5 seed-matched L3 arm and
    the full n=8 L3 arm are reported so the depth ladder can be read without a hole at L=3.
    """
    depth_seeds = range(5)
    effects: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(dict)
    acc_rows = ["R\tL\tn\tlength\tmean_acc_pp\tsd_pp\tseeds\tn_params"]
    eff_rows = [
        "L\tlength\tn\teffect_pp\tsd_pp\tci_lo\tci_hi\tdz\tt\tp\tverdict\tauthor_verdict\t"
        "claimed_pp\tabs_delta\tstatus"
    ]
    for layers in (1, 2, 3, 4):
        seeds = depth_seeds if layers != 3 else range(8)
        for r in (1, 4):
            have = [s for s in seeds if (r, "s5_words", layers, s) in D]
            if not have:
                continue
            npar = sorted({D[(r, "s5_words", layers, s)]["n_params"] for s in have})
            for L in LENGTHS_7:
                vs = [acc(D[(r, "s5_words", layers, s)], L) for s in have]
                acc_rows.append(
                    f"{r}\t{layers}\t{len(vs)}\t{L}\t{fmt(st.mean(vs))}\t"
                    f"{fmt(st.stdev(vs) if len(vs)>1 else 0.0)}\t{','.join(map(str,have))}\t"
                    f"{'|'.join(map(str,npar))}"
                )
        # Paired R4-R1 at this depth. For L=3 use the seed-matched n=5 subset so the substitution
        # ladder is comparable, and additionally report the full n=8.
        subsets = [(depth_seeds, "")] if layers != 3 else [(range(5), "_n5"), (range(8), "_n8")]
        for seedset, suffix in subsets:
            d, used = paired_diff(
                D,
                lambda s, ly=layers: (4, "s5_words", ly, s),
                lambda s, ly=layers: (1, "s5_words", ly, s),
                LENGTHS_7[0],
                seedset,
            )
            if len(used) < 2:
                continue
            for L in LENGTHS_7:
                d, used = paired_diff(
                    D,
                    lambda s, ly=layers: (4, "s5_words", ly, s),
                    lambda s, ly=layers: (1, "s5_words", ly, s),
                    L,
                    seedset,
                )
                res = Result(d).register("F_depth_R_effects", f"depth_L{layers}{suffix}_len{L}")
                if suffix in ("", "_n5"):
                    effects[L][layers] = dict(zip(used, d))
                c = CLAIMED_DEPTH[L][{1: 0, 2: 1, 4: 2}.get(layers, 99)] if layers != 3 else None
                stat = status(c, res.mean)
                eff_rows.append(
                    f"{layers}{suffix}\t{L}\t{res.n}\t{fmt(res.mean)}\t{fmt(res.sd)}\t"
                    f"{fmt(res.ci_lo)}\t{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t"
                    f"{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t"
                    f"{fmt(c,2) if c is not None else 'NOT_IN_HANDOFF'}\t"
                    f"{fmt(abs(c-res.mean)) if c is not None else 'NA'}\t{stat}"
                )
                if c is not None:
                    claims.append((f"depth_L{layers}_eff_len{L}", c, res.mean, stat))
    write_tsv(out_dir, "probe_depth_r_accuracy.tsv", acc_rows)

    # Substitution contrast at all seven lengths.
    rows = [
        "length\tn\tL1_effect\tL2_effect\tL3_effect_n5\tL4_effect\tsubstitution_L1_minus_L4\t"
        "sd\tci_lo\tci_hi\tdz\tt\tp\tverdict\tauthor_verdict\tclaimed_subst\tabs_delta\tstatus\t"
        "pearson_r_L1_vs_L4_effect\tseeds"
    ]
    for L in LENGTHS_7:
        if 1 not in effects[L] or 4 not in effects[L]:
            continue
        common = sorted(set(effects[L][1]) & set(effects[L][4]))
        diff = [effects[L][1][s] - effects[L][4][s] for s in common]
        res = Result(diff).register("F_substitution", f"substitution_len{L}")
        e = {k: st.mean(list(v.values())) for k, v in effects[L].items()}
        corr = pearson([effects[L][1][s] for s in common], [effects[L][4][s] for s in common])
        c = CLAIMED_DEPTH[L][3]
        stat = status(c, res.mean)
        rows.append(
            f"{L}\t{res.n}\t{fmt(e.get(1, float('nan')))}\t{fmt(e.get(2, float('nan')))}\t"
            f"{fmt(e.get(3, float('nan')))}\t{fmt(e.get(4, float('nan')))}\t{fmt(res.mean)}\t"
            f"{fmt(res.sd)}\t{fmt(res.ci_lo)}\t{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t"
            f"{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t"
            f"{fmt(c,2) if c is not None else 'NOT_IN_HANDOFF'}\t"
            f"{fmt(abs(c-res.mean)) if c is not None else 'NA'}\t{stat}\t{fmt(corr)}\t"
            f"{','.join(map(str,common))}"
        )
        if c is not None:
            claims.append((f"substitution_len{L}", c, res.mean, stat))
    write_tsv(out_dir, "probe_depth_r.tsv", rows)


# ---------------------------------------------------------------------------------------------
# Section 5: P3 depth ladder (mixer=kda, a DIFFERENT arm)
# ---------------------------------------------------------------------------------------------


def do_depth_ladder(root: str, D: dict, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the P3 depth ladder from the ``depth<N>-kda-s5_words-s*.json`` family.

    This family is a different arm from the R-grid: mixer ``kda`` (not ``kda_hh``), only five eval
    lengths, n=3 seeds, and parameter counts that grow with depth. All of that is emitted so the
    write-up can decide whether it is comparable.
    """
    rows = [
        "n_layers\tmixer\tn\tn_params\tlength\tmean_acc_pp\tsd_pp\tper_seed_pp\t"
        "claimed_pp_at_len40\tabs_delta\tstatus"
    ]
    for layers in (1, 2, 4, 6):
        fam = load_flat(root, f"depth{layers}-kda-s5_words")
        seeds = sorted(fam)
        npar = sorted({fam[s]["n_params"] for s in seeds})
        mixers = sorted({fam[s]["mixer"] for s in seeds})
        rs = sorted({fam[s]["num_householder"] for s in seeds})
        for L in LENGTHS_5:
            vs = [acc(fam[s], L) for s in seeds]
            m = st.mean(vs)
            c = CLAIMED_LADDER[layers] if L == "40" else None
            stat = status(c, m, 0.05)
            rows.append(
                f"{layers}\t{'|'.join(mixers)}+R{'|'.join(map(str,rs))}\t{len(vs)}\t"
                f"{'|'.join(map(str,npar))}\t{L}\t{fmt(m)}\t"
                f"{fmt(st.stdev(vs) if len(vs)>1 else 0.0)}\t{'|'.join(fmt(v,2) for v in vs)}\t"
                f"{fmt(c,2) if c is not None else 'NA'}\t"
                f"{fmt(abs(c-m)) if c is not None else 'NA'}\t{stat}"
            )
            if c is not None:
                claims.append((f"P3_ladder_L{layers}_acc@40", c, m, stat))
    # Comparability check: the kda_hh R=1 arm at the same depths, same task, from all_night.
    rows.append("#--- comparison arm: kda_hh R=1 (all_night), same task/depths ---")
    for layers in (1, 2, 4):
        have = [s for s in range(5) if (1, "s5_words", layers, s) in D]
        if not have:
            continue
        npar = sorted({D[(1, "s5_words", layers, s)]["n_params"] for s in have})
        m = st.mean([acc(D[(1, "s5_words", layers, s)], "40") for s in have])
        rows.append(
            f"{layers}\tkda_hh+R1\t{len(have)}\t{'|'.join(map(str,npar))}\t40\t{fmt(m)}\tNA\tNA\t"
            f"NA\tNA\tCOMPARISON_ARM"
        )
    write_tsv(out_dir, "probe_depth_ladder.tsv", rows)


# ---------------------------------------------------------------------------------------------
# Section 6: KDA vs GDN
# ---------------------------------------------------------------------------------------------


def do_kda_vs_gdn(root: str, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the KDA-vs-GDN comparisons on s5_words, parity, and mod_arith.

    Sign convention: the handoff reports GDN minus KDA (negative = GDN worse than KDA is NOT the
    reading; the handoff says "against KDA", i.e. negative means KDA loses). Both directions are
    emitted to remove the ambiguity.
    """
    families = [
        ("s5_words", "kda-s5_words", "gdn-s5_words"),
        ("parity", "kda-parity", "gdn-parity"),
        ("mod_arith", "p46-kda-mod_arith-L3", "p46-gdn-mod_arith-L3"),
    ]
    rows = [
        "task\tlength\tn\tkda_mean_pp\tgdn_mean_pp\tkda_minus_gdn\tgdn_minus_kda\tsd\tci_lo\t"
        "ci_hi\tdz\tt\tp\tverdict\tauthor_verdict\tseeds"
    ]
    means_by_task: Dict[str, Dict[str, float]] = {}
    for task, kpre, gpre in families:
        K, G = load_flat(root, kpre), load_flat(root, gpre)
        seeds = sorted(set(K) & set(G))
        means_by_task[task] = {}
        for L in LENGTHS_5:
            d = [acc(K[s], L) - acc(G[s], L) for s in seeds]
            res = Result(d).register("F_kda_vs_gdn", f"{task}_KDAvsGDN_len{L}")
            means_by_task[task][L] = res.mean  # KDA - GDN, matching the handoff's sign
            rows.append(
                f"{task}\t{L}\t{res.n}\t{fmt(st.mean([acc(K[s],L) for s in seeds]))}\t"
                f"{fmt(st.mean([acc(G[s],L) for s in seeds]))}\t{fmt(res.mean)}\t{fmt(-res.mean)}\t"
                f"{fmt(res.sd)}\t{fmt(res.ci_lo)}\t{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t"
                f"{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t{','.join(map(str,seeds))}"
            )
        # Also a per-seed pooled mean across lengths (one observation per seed), which is how the
        # "+2.01pp" style single number is most plausibly obtained.
        per_seed = [st.mean([acc(K[s], L) - acc(G[s], L) for L in LENGTHS_5]) for s in seeds]
        res = Result(per_seed).register("F_kda_vs_gdn", f"{task}_KDAvsGDN_pooled")
        rows.append(
            f"{task}\tPOOLED_5len\t{res.n}\tNA\tNA\t{fmt(res.mean)}\t{fmt(-res.mean)}\t"
            f"{fmt(res.sd)}\t{fmt(res.ci_lo)}\t{fmt(res.ci_hi)}\t{fmt(res.dz)}\t{fmt(res.t)}\t"
            f"{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t{','.join(map(str,seeds))}"
        )
        means_by_task[task]["POOLED"] = res.mean

    # ---- Provenance of the handoff's "+2.01pp" S5 figure -------------------------------------
    # No aggregation of the intended flat family (kda-s5_words-s*.json vs gdn-s5_words-s*.json)
    # produces +2.01. It is reproduced only by ``probes/analyze8.py``, which globs
    # ``results/*.json`` UNSORTED and keys records by (mixer, task, seed). Nine keys collide,
    # because depth<N>-kda-s5_words-s<k>.json and p46-kda-s5_words-L3-s<k>.json share the mixer,
    # task and seed of the intended kda-s5_words-s<k>.json while being different architectures.
    # Last-write-wins on an unsorted glob therefore substitutes depth-grid runs (including a
    # 2-layer, 696712-param model) into the 8-seed 3-layer comparison. Reproduced here.
    contam: Dict[Tuple[str, str, int], dict] = {}
    contam_prov: Dict[Tuple[str, str, int], str] = {}
    for f in glob.glob(os.path.join(root, "*.json")):  # deliberately UNSORTED, as analyze8.py is
        rec = json.load(open(f))
        key = (rec["mixer"], rec["task"], rec["seed"])
        contam[key] = rec
        contam_prov[key] = os.path.basename(f)
    rows2 = [
        "issue\tmixer\ttask\tseed\tfile_analyze8_actually_loads\tn_params\tintended_file\t"
        "intended_n_params\tsame_architecture"
    ]
    for s in sorted({k[2] for k in contam if k[:2] == ("kda", "s5_words")}):
        key = ("kda", "s5_words", s)
        got = contam_prov[key]
        intended = f"kda-s5_words-s{s}.json"
        ip = json.load(open(os.path.join(root, intended)))["n_params"]
        rows2.append(
            f"analyze8_key_collision\tkda\ts5_words\t{s}\t{got}\t{contam[key]['n_params']}\t"
            f"{intended}\t{ip}\t{contam[key]['n_params'] == ip}"
        )
    rows2.append("#--- effect of the collision on the headline KDA-vs-GDN S5 number ---")
    rows2.append("length\tcontaminated_kda_minus_gdn\tclean_kda_minus_gdn\tabs_shift")
    Kc, Gc = load_flat(root, "kda-s5_words"), load_flat(root, "gdn-s5_words")
    cseeds = sorted(set(Kc) & set(Gc))
    for L in LENGTHS_5:
        cont = st.mean(
            [
                acc(contam[("kda", "s5_words", s)], L) - acc(contam[("gdn", "s5_words", s)], L)
                for s in cseeds
            ]
        )
        clean = st.mean([acc(Kc[s], L) - acc(Gc[s], L) for s in cseeds])
        rows2.append(f"{L}\t{fmt(cont)}\t{fmt(clean)}\t{fmt(abs(cont-clean))}")
    rows2.append(
        "#NOTE\tthe handoff's '+2.01pp' matches the CONTAMINATED len128 cell (2.066), not any "
        "clean aggregate; the clean len128 effect is +3.46pp and is SIG, not ns"
    )
    write_tsv(out_dir, "probe_kda_vs_gdn_provenance.tsv", rows2)
    write_tsv(out_dir, "probe_kda_vs_gdn.tsv", rows)
    for task, L, c, _v in CLAIMED_KDA_GDN:
        if L is None:
            # s5_words "+2.01pp": no clean aggregate matches. Report the closest clean aggregate
            # (which is len256, +1.99) and flag that the figure traces to analyze8.py's
            # contaminated len128 cell instead.
            best = min(means_by_task[task].items(), key=lambda kv: abs(abs(kv[1]) - abs(c)))
            claims.append(
                (
                    f"kdagdn_{task}_2.01pp_NO_CLEAN_AGGREGATE_MATCHES" f"(closest_clean={best[0]})",
                    c,
                    best[1],
                    status(c, best[1], 0.05),
                )
            )
        else:
            got = means_by_task[task][L]
            claims.append((f"kdagdn_{task}@{L}", c, got, status(c, got, 0.05)))


# ---------------------------------------------------------------------------------------------
# Section 7: backend equivalence
# ---------------------------------------------------------------------------------------------


def do_backend_equiv(root: str, out_dir: str, claims: List[tuple]) -> None:
    """Recompute the triton-vs-torch backend equivalence analysis, including the per-seed table."""
    runs = load_beq(root)
    paired_seeds = sorted(s for s in runs if len(runs[s]) == 2)
    diffs: Dict[str, List[float]] = {L: [] for L in LENGTHS_5}
    rows = [
        "scope\tseed\tlength\ttriton_pp\ttorch_pp\ttriton_minus_torch\tclaimed\tabs_delta\tstatus"
    ]
    for s in paired_seeds:
        at, ao = runs[s]["triton"], runs[s]["torch"]
        for L in LENGTHS_5:
            d = acc(at, L) - acc(ao, L)
            diffs[L].append(d)
            c = CLAIMED_BEQ_PER_SEED.get(s, {}).get(L)
            stat = status(c, d)
            rows.append(
                f"per_seed\t{s}\t{L}\t{fmt(acc(at,L))}\t{fmt(acc(ao,L))}\t{fmt(d)}\t"
                f"{fmt(c,2) if c is not None else 'NA'}\t"
                f"{fmt(abs(c-d)) if c is not None else 'NA'}\t{stat}"
            )
            if c is not None:
                claims.append((f"beq_s{s}_len{L}", c, d, stat))
    # Non-ceiling lengths, defined exactly as the original script defines them.
    nonceil = [L for L in LENGTHS_5 if any(abs(x) > 1e-9 for x in diffs[L])]
    per_seed = [st.mean([diffs[L][i] for L in nonceil]) for i in range(len(paired_seeds))]
    for i, s in enumerate(paired_seeds):
        c = CLAIMED_BEQ_PER_SEED.get(s, {}).get("mean")
        stat = status(c, per_seed[i])
        rows.append(
            f"per_seed_mean_nonceiling\t{s}\t{'|'.join(nonceil)}\tNA\tNA\t{fmt(per_seed[i])}\t"
            f"{fmt(c,2) if c is not None else 'NA'}\t"
            f"{fmt(abs(c-per_seed[i])) if c is not None else 'NA'}\t{stat}"
        )
        if c is not None:
            claims.append((f"beq_s{s}_mean_nonceiling", c, per_seed[i], stat))
    rows.append("#--- per-length tests (n = seeds) ---")
    rows.append("scope\tlength\tn\tmean_pp\tsd\tci_lo\tci_hi\tt\tp\tverdict\tauthor_verdict\tnote")
    for L in LENGTHS_5:
        res = Result(diffs[L]).register("F_backend", f"beq_len{L}")
        note = (
            "all_seeds_identical"
            if res.sd == 0 and res.mean == 0
            else ("ZERO_VARIANCE_NONZERO_MEAN" if res.sd == 0 else "")
        )
        rows.append(
            f"per_length\t{L}\t{res.n}\t{fmt(res.mean)}\t{fmt(res.sd)}\t{fmt(res.ci_lo)}\t"
            f"{fmt(res.ci_hi)}\t{fmt(res.t)}\t{pfmt(res.p)}\t{res.verdict}\t"
            f"{res.author_verdict()}\t{note}"
        )
    # Pooled test (one mean per seed over non-ceiling lengths) + sign test.
    res = Result(per_seed).register("F_backend", "beq_pooled_nonceiling")
    cm, clo, chi = CLAIMED_BEQ_POOLED
    rows.append("#--- pooled ---")
    rows.append(
        f"pooled\t{'|'.join(nonceil)}\t{res.n}\t{fmt(res.mean)}\t{fmt(res.sd)}\t{fmt(res.ci_lo)}\t"
        f"{fmt(res.ci_hi)}\t{fmt(res.t)}\t{pfmt(res.p)}\t{res.verdict}\t{res.author_verdict()}\t"
        f"claimed_{cm}_[{clo},{chi}]"
    )
    claims.append(("beq_pooled_mean", cm, res.mean, status(cm, res.mean)))
    claims.append(("beq_pooled_ci_lo", clo, res.ci_lo, status(clo, res.ci_lo, 0.05)))
    claims.append(("beq_pooled_ci_hi", chi, res.ci_hi, status(chi, res.ci_hi, 0.05)))
    n_neg = sum(1 for x in per_seed if x < 0)
    n = len(per_seed)
    k = min(n_neg, n - n_neg)
    p_sign = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    rows.append("#--- sign test ---")
    rows.append(
        f"sign_test\tNA\t{n}\tn_negative={n_neg}\tNA\tNA\tNA\tNA\t{p_sign:.4f}\tNA\tNA\t"
        f"claimed_{CLAIMED_BEQ_SIGN[0]}/{CLAIMED_BEQ_SIGN[1]}_p={CLAIMED_BEQ_SIGN[2]}"
    )
    claims.append(
        (
            "beq_sign_n_negative",
            float(CLAIMED_BEQ_SIGN[0]),
            float(n_neg),
            status(float(CLAIMED_BEQ_SIGN[0]), float(n_neg)),
        )
    )
    claims.append(
        ("beq_sign_p", CLAIMED_BEQ_SIGN[2], p_sign, status(CLAIMED_BEQ_SIGN[2], p_sign, 0.005))
    )
    # Cluster structure claim: five seeds at -2..-4pp, three at +1.2..+2pp.
    lo_cluster = sorted(x for x in per_seed if x < 0)
    hi_cluster = sorted(x for x in per_seed if x >= 0)
    rows.append("#--- cluster structure ---")
    rows.append(
        f"clusters\tNA\t{n}\tnegative_n={len(lo_cluster)}\trange=[{fmt(min(lo_cluster),2)},"
        f"{fmt(max(lo_cluster),2)}]\tpositive_n={len(hi_cluster)}\t"
        f"range=[{fmt(min(hi_cluster),2)},{fmt(max(hi_cluster),2)}]\tNA\tNA\tNA\tNA\t"
        f"claimed_5_at_-2..-4_and_3_at_+1.2..+2"
    )
    write_tsv(out_dir, "probe_backend_equiv.tsv", rows)
    TEST_REGISTRY.append(
        ("F_backend", "beq_sign_test", p_sign, "SIG" if p_sign < ALPHA else "ns", False)
    )


# ---------------------------------------------------------------------------------------------
# Section 8: multiplicity and ceiling audits
# ---------------------------------------------------------------------------------------------


def holm(pvals: Sequence[Tuple[str, float]], alpha: float = ALPHA) -> Dict[str, bool]:
    """Holm-Bonferroni step-down procedure.

    Degenerate tests (p is nan) are treated as NOT rejected, since no valid p exists.

    :returns: Mapping test_id -> rejected at family-wise ``alpha``.
    """
    usable = [(tid, p) for tid, p in pvals if p is not None and not math.isnan(p)]
    m = len(usable)
    out = {tid: False for tid, _ in pvals}
    for i, (tid, p) in enumerate(sorted(usable, key=lambda kv: kv[1])):
        if p <= alpha / (m - i):
            out[tid] = True
        else:
            break
    return out


def do_multiplicity(out_dir: str) -> None:
    """Emit the multiplicity analysis: per-family Holm, global Holm, and global Bonferroni."""
    by_family: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for fam, tid, p, _v, _d in TEST_REGISTRY:
        by_family[fam].append((tid, p))
    # De-duplicate: the solvability family re-runs the identical parity and S5 R4-vs-1 tests that
    # the interaction section already performed. Count each distinct test once for global control.
    unique: Dict[str, float] = {}
    for _fam, tid, p, _v, _d in TEST_REGISTRY:
        canon = tid.replace("S5_R4vs1_", "s5_words_R4vs1_").replace(
            "parity_R4vs1_", "parity_R4vs1_"
        )
        unique.setdefault(canon, p)
    global_holm = holm(list(unique.items()))
    n_global = len([p for p in unique.values() if p is not None and not math.isnan(p)])
    bonf_alpha = ALPHA / n_global

    rows = [
        "family\ttest_id\tp\tuncorrected\tholm_within_family\tholm_global\t"
        "bonferroni_global\tdegenerate"
    ]
    for fam in sorted(by_family):
        fam_holm = holm(by_family[fam])
        for tid, p in by_family[fam]:
            deg = any(t == tid and d for _f, t, _p, _v, d in TEST_REGISTRY)
            canon = tid.replace("S5_R4vs1_", "s5_words_R4vs1_")
            unc = (
                "REJECT"
                if (p is not None and not math.isnan(p) and p < ALPHA)
                else ("no" if p is not None and not math.isnan(p) else "NA")
            )
            rows.append(
                f"{fam}\t{tid}\t{pfmt(p)}\t{unc}\t"
                f"{'REJECT' if fam_holm.get(tid) else 'no'}\t"
                f"{'REJECT' if global_holm.get(canon) else 'no'}\t"
                f"{'REJECT' if (p is not None and not math.isnan(p) and p < bonf_alpha) else 'no'}\t"
                f"{deg}"
            )
    rows.append(f"#total_tests_registered\t{len(TEST_REGISTRY)}")
    rows.append(f"#unique_tests\t{len(unique)}")
    rows.append(f"#unique_tests_with_valid_p\t{n_global}")
    rows.append(f"#global_bonferroni_alpha\t{bonf_alpha:.3e}")
    for fam in sorted(by_family):
        valid = [p for _t, p in by_family[fam] if p is not None and not math.isnan(p)]
        rows.append(
            f"#family_size\t{fam}\t{len(by_family[fam])}\tvalid_p={len(valid)}\t"
            f"bonferroni_alpha={ALPHA/max(len(valid),1):.3e}"
        )
    write_tsv(out_dir, "probe_multiplicity.tsv", rows)


def do_ceiling_audit(D: dict, root: str, out_dir: str) -> None:
    """Enumerate every cell at or near ceiling, and every zero-variance paired difference."""
    rows = ["scope\tcell\tn\tmean_acc_pp\tn_seeds_at_100\tall_at_ceiling"]
    # Ceiling cells in the all_night grid.
    keys = sorted({(r, t, ly) for (r, t, ly, _s) in D})
    for r, t, ly in keys:
        have = sorted(s for (rr, tt, ll, s) in D if (rr, tt, ll) == (r, t, ly))
        for L in LENGTHS_7:
            vs = [acc(D[(r, t, ly, s)], L) for s in have]
            n100 = sum(1 for v in vs if v >= 100.0 - 1e-9)
            if n100 == 0:
                continue
            rows.append(
                f"all_night\tR{r}-{t}-L{ly}-len{L}\t{len(vs)}\t{fmt(st.mean(vs))}\t{n100}\t"
                f"{n100 == len(vs)}"
            )
    rows.append("#--- zero-variance paired differences (sd == 0) across every test performed ---")
    rows.append("scope\ttest_id\tn\tmean_pp\tsd\tauthor_verdict\tthis_script_verdict")
    for fam, tid, p, verdict, deg in TEST_REGISTRY:
        if verdict in ("DEGENERATE_SD0", "ns_IDENTICAL"):
            rows.append(
                f"{fam}\t{tid}\tNA\tNA\t0.0\t"
                f"{'SIG' if verdict=='DEGENERATE_SD0' else 'ns'}\t{verdict}"
            )
    write_tsv(out_dir, "probe_ceiling_audit.tsv", rows)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main() -> None:
    """Recompute everything and write all TSVs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="/scratch/users/ericrcwu/kda/probes/results")
    ap.add_argument("--out-dir", default="/scratch/users/ericrcwu/agent-runs/dp2-kda-p0/writeup")
    args = ap.parse_args()
    root, out_dir = args.results_root, args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log: List[str] = []
    log += verify_t_crit_table(out_dir)

    D = load_all_night(root)
    log.append(f"all_night records loaded: {len(D)}")
    comp: Dict[Tuple[int, str, int], int] = defaultdict(int)
    for r, t, ly, _s in D:
        comp[(r, t, ly)] += 1
    prov = ["num_householder\ttask\tn_layers\tn_seeds"]
    for k in sorted(comp):
        prov.append(f"{k[0]}\t{k[1]}\t{k[2]}\t{comp[k]}")
    prov.append(f"#total\t{len(D)}")
    write_tsv(out_dir, "probe_provenance.tsv", prov)

    claims: List[tuple] = []
    do_s5_table(D, out_dir, claims)
    do_interaction(D, out_dir, claims)
    do_solvability(D, out_dir, claims)
    do_depth_r(D, out_dir, claims)
    do_depth_ladder(root, D, out_dir, claims)
    do_kda_vs_gdn(root, out_dir, claims)
    do_backend_equiv(root, out_dir, claims)
    do_multiplicity(out_dir)
    do_ceiling_audit(D, root, out_dir)

    rows = ["claim_id\tclaimed\trecomputed\tabs_delta\tstatus"]
    for cid, c, got, stat in claims:
        rows.append(
            f"{cid}\t{fmt(c,2) if isinstance(c,(int,float)) else 'NOT_IN_HANDOFF'}\t"
            f"{fmt(got) if isinstance(got,(int,float)) else got}\t"
            f"{fmt(abs(c-got)) if isinstance(c,(int,float)) and isinstance(got,(int,float)) else 'NA'}\t"
            f"{stat}"
        )
    counts = defaultdict(int)
    for _cid, _c, _g, stat in claims:
        counts[stat] += 1
    for k in sorted(counts):
        rows.append(f"#status_count\t{k}\t{counts[k]}")
    write_tsv(out_dir, "probe_claims_check.tsv", rows)

    print("\n".join(log))
    print(f"\nclaim comparisons: {dict(counts)}")
    print(f"tests registered: {len(TEST_REGISTRY)}")
    print(f"outputs in {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".tsv"):
            print("  ", f)


if __name__ == "__main__":
    main()
