"""Final analysis of the KDA-Householder language-model grid (Slurm job 1662404).

WHAT THIS SCRIPT IS
-------------------
The reproduction path for every LM number in the write-up. It reads the 13 result JSONs
produced by ``lm/train_lm.py`` and emits three TSVs (arms, contrasts, power). It is pure
CPU, deterministic, and depends only on the standard library plus nothing else -- the
t-distribution is implemented here (via the regularized incomplete beta function) because
scipy is absent from the analysis venv.

Run::

    python analyze_lm_final.py --results-dir /scratch/users/ericrcwu/kda/lm/results/lm \
                               --out-dir     /scratch/users/ericrcwu/agent-runs/dp2-kda-p0/writeup/lm

THE DESIGN BEING ANALYSED
-------------------------
Three arms, because R cannot be varied at fixed parameter count (R widens ``w_k``/``w_v``/
``w_b`` and the k/v convolutions):

======================  ===  =======  ==================
arm                     R    d_model  non-embed params
======================  ===  =======  ==================
``hh1``                 1    512      52.05M  (baseline)
``hh4``                 4    512      71.22M  (+36.8%; the mechanism)
``hh4_r1wide``          1    616      71.67M  (+0.6% vs hh4; capacity control)
======================  ===  =======  ==================

Pre-registered decision rule (fixed before data, on the DEGRADATION endpoint):

* hh4 beats **both** hh1 and r1wide  => the effect is **R**
* hh4 beats hh1 but ties r1wide      => the effect is **capacity**; R contributes nothing
* hh4 ~ hh1                          => **no effect at this scale**

TWO CORRECTIONS THIS SCRIPT MAKES TO THE PRE-REGISTERED ANALYSIS
----------------------------------------------------------------
1. ``tokens_seen`` in every JSON is **wrong by exactly 4x**. ``train_lm.py`` computes it as
   ``steps * batch * seq_len`` and omits ``args.accum``, while the training loop provably
   consumes ``accum`` micro-batches per optimiser step. The runs are at 1.042B tokens
   (~20 tok/param), not the 260.5M the JSONs report. Both are carried in the output so the
   discrepancy is auditable rather than silently patched.

2. The degradation statistic ``loss(L) - loss(2048)`` is **not** a within-run length effect.
   ``evaluate()`` builds a fresh ``FlatWindowLoader`` per length with seed
   ``seed * 7919 + L``, so the windows scored at 2048 and at L are *disjoint random draws*
   from the validation stream, and only 32 windows are scored per length. The between-seed
   scatter in raw val loss (sd ~= 0.09 nats) is therefore dominated by which validation
   windows happened to be drawn, not by the training seed.

   That same fact is what rescues the analysis: the eval seed depends only on ``(seed, L)``,
   **not on the arm**, so at a given seed all three arms are scored on the *identical*
   windows at every length. Every arm contrast is paired on eval windows and the shared
   draw noise cancels exactly. Concretely, the degradation contrast is a
   difference-in-differences,

       [loss_A(L) - loss_A(2048)] - [loss_B(L) - loss_B(2048)]
         = [loss_A(L) - loss_B(L)] - [loss_A(2048) - loss_B(2048)],

   in which each bracket is computed on one common window set. So the *arm contrast* is
   valid and tight even though the *per-arm* degradation is confounded. This script reports
   per-arm degradation for completeness but draws inferences only from the contrasts, and
   quantifies the shared-draw fraction explicitly (``--verbose`` prints the between-arm
   correlation across seeds).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------
# Experimental constants, taken from lm/run_lm_grid.sbatch. They are not recorded in the
# result JSONs, so they are pinned here and asserted where possible.
# ---------------------------------------------------------------------------------------
TRAIN_LEN = 2048
LENGTHS = [2048, 4096, 8192, 16384]
ACCUM = 4  # --accum 4 ; NOT reflected in the JSON's tokens_seen field
MICRO_BATCH = 4  # --batch 4
EVAL_BATCH = 1  # --eval-batch 1
EVAL_BATCHES = 32  # --eval-batches 32  => 32 scored windows per length
ALL_SEEDS = [0, 1, 2, 3, 4]

# arm -> (R, d_model). d_model is cross-checked against the JSON.
ARM_META: Dict[str, Tuple[int, int]] = {
    "hh1": (1, 512),
    "hh4": (4, 512),
    "hh4_r1wide": (1, 616),
}
ARM_ORDER = ["hh1", "hh4", "hh4_r1wide"]

# The three contrasts. (high, low, label, what_it_isolates)
CONTRASTS = [
    ("hh4", "hh1", "hh4-hh1", "R + 37% capacity (CONFOUNDED)"),
    ("hh4", "hh4_r1wide", "hh4-r1wide", "R at matched capacity (THE REAL TEST)"),
    ("hh4_r1wide", "hh1", "r1wide-hh1", "capacity alone, no R"),
]


# =======================================================================================
# Statistics. Implemented locally: the analysis venv has numpy but no scipy, and a
# shipped reproduction path should not depend on either.
# =======================================================================================
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT, EPS, FPMIN = 500, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_p_two_sided(t: float, df: int) -> float:
    """Two-sided p-value for Student's t: ``I_{df/(df+t^2)}(df/2, 1/2)``."""
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_cdf(t: float, df: int) -> float:
    """CDF of Student's t."""
    p = t_p_two_sided(t, df)
    return 1.0 - 0.5 * p if t > 0 else 0.5 * p


def t_ppf(p: float, df: int) -> float:
    """Inverse CDF of Student's t, by bisection. Adequate and deterministic."""
    lo, hi = -1.0e3, 1.0e3
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Paired:
    """One-sample t-test on paired differences, with CI and Cohen's dz."""

    def __init__(self, diffs: Sequence[float], seeds: Sequence[int], alpha: float = 0.05):
        self.diffs = list(diffs)
        self.seeds = list(seeds)
        self.n = len(self.diffs)
        self.alpha = alpha
        self.mean = st.mean(self.diffs) if self.n else float("nan")
        self.sd = st.stdev(self.diffs) if self.n > 1 else float("nan")
        self.df = self.n - 1
        if self.n > 1 and self.sd > 0:
            self.sem = self.sd / math.sqrt(self.n)
            self.t = self.mean / self.sem
            self.p = t_p_two_sided(self.t, self.df)
            self.tcrit = t_ppf(1.0 - alpha / 2.0, self.df)
            self.ci = self.tcrit * self.sem
            self.dz = self.mean / self.sd
        else:
            self.sem = self.t = self.p = self.tcrit = self.ci = self.dz = float("nan")

    @property
    def lo(self) -> float:
        return self.mean - self.ci

    @property
    def hi(self) -> float:
        return self.mean + self.ci

    @property
    def sig(self) -> bool:
        return self.p == self.p and self.p < self.alpha

    def verdict(self, lower_is_better: bool = True) -> str:
        """Human verdict. ``lower_is_better`` because the endpoint is a loss."""
        if self.n < 2 or self.p != self.p:
            return "insufficient-n"
        if not self.sig:
            return "ns"
        better = "high-arm-better" if (self.mean < 0) == lower_is_better else "low-arm-better"
        return f"SIG({better})"


def mde(sd: float, n: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect (same units as ``sd``) for a paired t-test.

    Uses the standard t-shift approximation ``delta = sd * (t_{1-a/2,n-1} + t_{power,n-1})
    / sqrt(n)``, which is mildly conservative relative to exact noncentral-t power.
    """
    if n < 2 or sd != sd:
        return float("nan")
    df = n - 1
    return sd * (t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df)) / math.sqrt(n)


def n_for_power(
    effect: float, sd: float, power: float = 0.80, alpha: float = 0.05, cap: int = 200000
) -> Optional[int]:
    """Smallest n giving ``power`` to detect ``effect`` given paired ``sd``."""
    if sd != sd or sd <= 0 or effect == 0 or effect != effect:
        return None
    dz = abs(effect) / sd
    for n in range(2, cap + 1):
        df = n - 1
        if dz * math.sqrt(n) >= t_ppf(1.0 - alpha / 2.0, df) + t_ppf(power, df):
            return n
    return None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; used to show the eval draw is shared across arms."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


# =======================================================================================
# Loading
# =======================================================================================
def load(results_dir: str) -> Dict[Tuple[str, int], dict]:
    """Load ``<arm>-s<seed>.json``, keyed by ``(arm, seed)``.

    The arm is taken from the FILENAME, not from the record's ``arm`` field. That field is
    unreliable: the r1wide arm is launched as ``--arm hh1 --d-model 616``, so its JSONs
    record ``"arm": "hh1"``. The filename is the only correct label. This is checked and
    reported rather than silently tolerated.
    """
    out: Dict[Tuple[str, int], dict] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        m = re.match(r"(.+)-s(\d+)\.json$", os.path.basename(path))
        if m is None:
            continue
        arm, seed = m.group(1), int(m.group(2))
        if arm not in ARM_META:
            continue
        with open(path) as fh:
            rec = json.load(fh)
        rec["_path"] = path
        rec["_arm"] = arm
        rec["_label_mismatch"] = rec.get("arm") != arm
        # Sanity: the geometry in the record must match the arm definition.
        exp_r, exp_d = ARM_META[arm]
        assert rec["d_model"] == exp_d, f"{path}: d_model {rec['d_model']} != {exp_d}"
        rec["_R"] = exp_r
        # Token accounting: the JSON field omits --accum.
        rec["_tokens_reported"] = rec["tokens_seen"]
        rec["_tokens_actual"] = rec["steps"] * MICRO_BATCH * ACCUM * rec["train_seq_len"]
        assert rec["_tokens_reported"] == rec["steps"] * MICRO_BATCH * rec["train_seq_len"], (
            f"{path}: tokens_seen is not steps*batch*seq_len; the 4x accum correction "
            f"assumed here may not apply."
        )
        out[(arm, seed)] = rec
    return out


def val(rec: dict, L: int) -> float:
    return rec["val_loss_by_length"][str(L)]


def degradation(rec: dict, L: int) -> float:
    """``loss(L) - loss(2048)``. Positive = worse at the longer length."""
    return val(rec, L) - val(rec, TRAIN_LEN)


# =======================================================================================
# Reporting
# =======================================================================================
def write_tsv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join("" if c is None else str(c) for c in row) + "\n")
    print(f"  wrote {path}  ({len(rows)} rows)")


def f4(x: float) -> str:
    return "" if x != x else f"{x:.4f}"


def f6(x: float) -> str:
    return "" if x != x else f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/scratch/users/ericrcwu/kda/lm/results/lm")
    ap.add_argument(
        "--out-dir", default="/scratch/users/ericrcwu/agent-runs/dp2-kda-p0/writeup/lm"
    )
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    D = load(args.results_dir)
    arms = [a for a in ARM_ORDER if any(k[0] == a for k in D)]
    seeds_of = {a: sorted(s for (arm, s) in D if arm == a) for a in arms}

    print("=" * 88)
    print("KDA-HOUSEHOLDER LM GRID -- FINAL ANALYSIS (Slurm job 1662404)")
    print("=" * 88)
    print(f"\n{len(D)} runs loaded from {args.results_dir}")
    for a in arms:
        print(f"  {a:>12}  n={len(seeds_of[a])}  seeds={seeds_of[a]}")
    missing = [(a, s) for a in ARM_ORDER for s in ALL_SEEDS if (a, s) not in D]
    print(f"  MISSING (of 15 planned): {missing}")

    bad = [f"{r['_arm']}-s{k[1]} (field={r['arm']!r})" for k, r in D.items() if r["_label_mismatch"]]
    print(
        f"\nJSON 'arm' field mismatches filename for {len(bad)} runs (expected for r1wide, "
        f"which is launched as --arm hh1 --d-model 616):"
    )
    for b in bad:
        print(f"    {b}")

    # ---------------------------------------------------------------------------------
    # Token budget
    # ---------------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print("TOKEN BUDGET -- the JSON field is wrong by exactly 4x")
    print("-" * 88)
    r0 = D[(arms[0], seeds_of[arms[0]][0])]
    print(
        f"  train_lm.py:349  tokens_seen = steps * batch * seq_len   (omits --accum)\n"
        f"  train_lm.py:301  micro_iter  = loader.batch_iter(steps * accum, ...)\n"
        f"  train_lm.py:304  for _ in range(accum): x, y = next(micro_iter)  <- accum DOES run\n"
        f"  reported : {r0['_tokens_reported']:,} tokens  ({r0['_tokens_reported']/1e6:.1f}M)\n"
        f"  actual   : {r0['_tokens_actual']:,} tokens  ({r0['_tokens_actual']/1e9:.3f}B)"
    )
    for a in arms:
        ne = D[(a, seeds_of[a][0])]["n_params_nonembed"]
        rec = D[(a, seeds_of[a][0])]
        print(
            f"    {a:>12}: {ne/1e6:6.2f}M non-embed -> "
            f"{rec['_tokens_actual']/ne:5.2f} tok/param actual "
            f"(vs {rec['_tokens_reported']/ne:4.2f} if the JSON were right)"
        )
    print("  Throughput cross-check (independent of the code reading):")
    for a in arms:
        ws = [D[(a, s)]["wall_seconds"] for s in seeds_of[a]]
        sps = st.mean(ws) / r0["steps"]
        tok_per_s_4 = MICRO_BATCH * ACCUM * TRAIN_LEN / sps
        tok_per_s_1 = MICRO_BATCH * TRAIN_LEN / sps
        print(
            f"    {a:>12}: {sps:.3f} s/opt-step -> {tok_per_s_4:8.0f} tok/s if accum=4, "
            f"{tok_per_s_1:7.0f} tok/s if accum=1"
        )

    # ---------------------------------------------------------------------------------
    # Eval geometry -- why the degradation metric needs care
    # ---------------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print("EVAL GEOMETRY -- loss(L) and loss(2048) are DISJOINT random draws")
    print("-" * 88)
    print(
        f"  evaluate() builds FlatWindowLoader(val, L, seed=seed*7919 + L) per length:\n"
        f"    * the window offsets differ per L  -> cross-L comparison within a run is NOT\n"
        f"      the same tokens; it is two independent samples of the val stream\n"
        f"    * the seed does NOT depend on the arm -> at a fixed (seed, L) all arms are\n"
        f"      scored on IDENTICAL windows, so arm contrasts are paired and draw noise cancels\n"
        f"    * only {EVAL_BATCHES} windows (eval_batch={EVAL_BATCH}) are scored per length:"
    )
    for L in LENGTHS:
        print(f"        L={L:>5}: {EVAL_BATCHES*EVAL_BATCH*L:>9,} tokens scored")
    print("  => tokens scored grows 8x from L=2048 to L=16384; the number of independent")
    print("     windows (32) does not. The effective n for a per-arm loss is ~32 windows.")

    print("\n  Evidence that between-seed scatter is the shared eval draw, not the training seed:")
    common_all = sorted(set.intersection(*[set(seeds_of[a]) for a in arms]))
    for L in LENGTHS:
        parts = []
        for a in arms:
            vs = [val(D[(a, s)], L) for s in common_all]
            parts.append(f"sd({a})={st.stdev(vs):.4f}")
        rs = []
        for hi, lo, lab, _ in CONTRASTS:
            if hi in arms and lo in arms:
                r = pearson(
                    [val(D[(hi, s)], L) for s in common_all],
                    [val(D[(lo, s)], L) for s in common_all],
                )
                rs.append(f"r({lab})={r:+.4f}")
        print(f"    L={L:>5}  " + "  ".join(parts) + "   " + "  ".join(rs))
    print("    Near-unit correlations => the arms rise and fall together across seeds because")
    print("    they share the eval windows. The PAIRED contrast removes essentially all of it.")

    # ---------------------------------------------------------------------------------
    # lm_arms.tsv  (long format: one row per arm x length)
    # ---------------------------------------------------------------------------------
    print("\n" + "-" * 88)
    print("PER-ARM SUMMARY")
    print("-" * 88)
    print(f"{'arm':>12} {'R':>2} {'dmod':>5} {'n':>2} " + " ".join(f"{L:>16}" for L in LENGTHS))
    arm_rows: List[List[object]] = []
    for a in arms:
        ss = seeds_of[a]
        rec0 = D[(a, ss[0])]
        cells = []
        for L in LENGTHS:
            vs = [val(D[(a, s)], L) for s in ss]
            cells.append(f"{st.mean(vs):8.4f}+-{st.stdev(vs):.4f}")
            dg = [degradation(D[(a, s)], L) for s in ss] if L != TRAIN_LEN else []
            arm_rows.append(
                [
                    a,
                    rec0["_R"],
                    rec0["d_model"],
                    rec0["n_params"],
                    rec0["n_params_nonembed"],
                    len(ss),
                    ",".join(map(str, ss)),
                    L,
                    f6(st.mean(vs)),
                    f6(st.stdev(vs)),
                    f6(st.stdev(vs) / math.sqrt(len(vs))),
                    f6(st.mean(dg)) if dg else "",
                    f6(st.stdev(dg)) if len(dg) > 1 else "",
                    EVAL_BATCHES * EVAL_BATCH * L,
                    f6(st.mean([D[(a, s)]["final_train_loss"] for s in ss])),
                    rec0["_tokens_reported"],
                    rec0["_tokens_actual"],
                    f"{rec0['_tokens_actual']/rec0['n_params_nonembed']:.2f}",
                    f"{st.mean([D[(a, s)]['wall_seconds'] for s in ss]):.1f}",
                ]
            )
        print(f"{a:>12} {rec0['_R']:>2} {rec0['d_model']:>5} {len(ss):>2} " + " ".join(cells))

    print(f"\n{'arm':>12} DEGRADATION loss(L)-loss(2048), per-arm (CONFOUNDED, see above)")
    for a in arms:
        ss = seeds_of[a]
        cells = []
        for L in LENGTHS[1:]:
            dg = [degradation(D[(a, s)], L) for s in ss]
            cells.append(f"{st.mean(dg):+8.4f}+-{st.stdev(dg):.4f}")
        print(f"{a:>12} " + " ".join(cells))

    print(f"\n{'arm':>12} {'final_train_loss':>18} {'val@2048':>10}  train-val")
    for a in arms:
        ss = seeds_of[a]
        tr = st.mean([D[(a, s)]["final_train_loss"] for s in ss])
        va = st.mean([val(D[(a, s)], TRAIN_LEN) for s in ss])
        print(f"{a:>12} {tr:>18.4f} {va:>10.4f}  {tr-va:+.4f}")

    write_tsv(
        os.path.join(args.out_dir, "lm_arms.tsv"),
        [
            "arm", "R", "d_model", "n_params", "n_params_nonembed", "n", "seeds",
            "eval_len", "val_loss_mean", "val_loss_sd", "val_loss_sem",
            "degradation_mean", "degradation_sd", "tokens_scored_at_len",
            "final_train_loss_mean", "tokens_seen_reported_json", "tokens_seen_actual",
            "tokens_per_nonembed_param", "wall_seconds_mean",
        ],
        arm_rows,
    )

    # ---------------------------------------------------------------------------------
    # Contrasts
    # ---------------------------------------------------------------------------------
    contrast_rows: List[List[object]] = []
    power_rows: List[List[object]] = []
    results: Dict[Tuple[str, str, int], Paired] = {}

    def add(endpoint: str, label: str, isolates: str, L: int, tt: Paired) -> None:
        results[(endpoint, label, L)] = tt
        contrast_rows.append(
            [
                label, isolates, endpoint, L, tt.n, ",".join(map(str, tt.seeds)),
                f6(tt.mean), f6(tt.sd), f6(tt.sem), f6(tt.lo), f6(tt.hi),
                f4(tt.t), tt.df, f6(tt.p), f4(tt.dz), tt.verdict(),
            ]
        )

    print("\n" + "=" * 88)
    print("PRIMARY (pre-registered): DEGRADATION contrasts  =  difference-in-differences")
    print("  negative => the high arm degrades LESS with length, i.e. extrapolates better")
    print("=" * 88)
    for hi, lo, label, isolates in CONTRASTS:
        if hi not in arms or lo not in arms:
            continue
        print(f"\n  {label:>12}   ({isolates})")
        for L in LENGTHS[1:]:
            ss = sorted(set(seeds_of[hi]) & set(seeds_of[lo]))
            tt = Paired([degradation(D[(hi, s)], L) - degradation(D[(lo, s)], L) for s in ss], ss)
            add("degradation", label, isolates, L, tt)
            print(
                f"    L={L:>5} n={tt.n} seeds={ss}  {tt.mean:+9.5f}  "
                f"95%CI [{tt.lo:+8.5f},{tt.hi:+8.5f}]  sd={tt.sd:.5f}  "
                f"t({tt.df})={tt.t:+7.3f}  p={tt.p:.4f}  dz={tt.dz:+6.2f}  {tt.verdict()}"
            )

    print("\n" + "=" * 88)
    print("SECONDARY (pre-declared underpowered): VAL LOSS at each length")
    print("  negative => the high arm has LOWER loss, i.e. is BETTER")
    print("=" * 88)
    for hi, lo, label, isolates in CONTRASTS:
        if hi not in arms or lo not in arms:
            continue
        print(f"\n  {label:>12}   ({isolates})")
        for L in LENGTHS:
            ss = sorted(set(seeds_of[hi]) & set(seeds_of[lo]))
            tt = Paired([val(D[(hi, s)], L) - val(D[(lo, s)], L) for s in ss], ss)
            add("val_loss", label, isolates, L, tt)
            star = "  <-- strictly parameter-matched" if (label == "hh4-r1wide" and L == TRAIN_LEN) else ""
            print(
                f"    L={L:>5} n={tt.n} seeds={ss}  {tt.mean:+9.5f}  "
                f"95%CI [{tt.lo:+8.5f},{tt.hi:+8.5f}]  sd={tt.sd:.5f}  "
                f"t({tt.df})={tt.t:+7.3f}  p={tt.p:.4f}  dz={tt.dz:+6.2f}  {tt.verdict()}{star}"
            )

    print("\n" + "=" * 88)
    print("TERTIARY: final TRAIN loss contrasts (same optimiser trajectory, paired by seed)")
    print("=" * 88)
    for hi, lo, label, isolates in CONTRASTS:
        if hi not in arms or lo not in arms:
            continue
        ss = sorted(set(seeds_of[hi]) & set(seeds_of[lo]))
        tt = Paired([D[(hi, s)]["final_train_loss"] - D[(lo, s)]["final_train_loss"] for s in ss], ss)
        add("final_train_loss", label, isolates, TRAIN_LEN, tt)
        print(
            f"  {label:>12} n={tt.n}  {tt.mean:+9.5f}  95%CI [{tt.lo:+8.5f},{tt.hi:+8.5f}]  "
            f"t({tt.df})={tt.t:+7.3f}  p={tt.p:.5f}  dz={tt.dz:+6.2f}  {tt.verdict()}"
        )

    # Pooled length effect: is there a length effect at all, across all runs?
    print("\n" + "=" * 88)
    print("IS THERE A LENGTH EFFECT AT ALL?  degradation pooled over every run (all arms)")
    print("  handoff predicted 1-5 nats of degradation from 2048 -> 16384")
    print("=" * 88)
    for L in LENGTHS[1:]:
        dg = [degradation(rec, L) for rec in D.values()]
        tt = Paired(dg, [k[1] for k in D])
        add("degradation_pooled_all_arms", "pooled", "length effect, arms pooled", L, tt)
        print(
            f"  L={L:>5} n={tt.n}  mean {tt.mean:+8.5f}  95%CI [{tt.lo:+8.5f},{tt.hi:+8.5f}]  "
            f"sd={tt.sd:.5f}  range [{min(dg):+.4f},{max(dg):+.4f}]  p={tt.p:.4f}  {tt.verdict()}"
        )
    print("  Observed |degradation| is ~0.03-0.06 nats: 1-2 ORDERS OF MAGNITUDE below the")
    print("  1-5 nats the power analysis assumed, and inconsistent in sign across seeds.")

    write_tsv(
        os.path.join(args.out_dir, "lm_contrasts.tsv"),
        [
            "contrast", "isolates", "endpoint", "eval_len", "n", "seeds_used",
            "paired_mean_diff", "sd_of_diff", "sem", "ci95_lo", "ci95_hi",
            "t", "df", "p_two_sided", "cohens_dz", "verdict",
        ],
        contrast_rows,
    )

    # ---------------------------------------------------------------------------------
    # Power
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("POWER -- computed from the OBSERVED paired sds")
    print("=" * 88)
    probe_effects = [0.0053, 0.0100, 0.0500, 1.0000]
    header = (
        f"{'endpoint':>26} {'contrast':>12} {'L':>6} {'sd_paired':>10} {'obs_eff':>9} "
        f"{'MDE@4':>8} {'MDE@5':>8} " + " ".join(f"n@{e:g}".rjust(8) for e in probe_effects)
    )
    print(header)
    for (endpoint, label, L), tt in results.items():
        if endpoint == "degradation_pooled_all_arms" or tt.n < 2 or tt.sd != tt.sd:
            continue
        ns = [n_for_power(e, tt.sd) for e in probe_effects]
        n_obs = n_for_power(tt.mean, tt.sd)
        print(
            f"{endpoint:>26} {label:>12} {L:>6} {tt.sd:>10.5f} {tt.mean:>+9.5f} "
            f"{mde(tt.sd,4):>8.5f} {mde(tt.sd,5):>8.5f} "
            + " ".join((str(n) if n else ">2e5").rjust(8) for n in ns)
        )
        power_rows.append(
            [
                endpoint, label, L, tt.n, f6(tt.sd), f6(tt.mean), f4(tt.dz),
                f6(mde(tt.sd, 4)), f6(mde(tt.sd, 5)), f6(mde(tt.sd, 10)),
                n_obs if n_obs else ">200000",
                *[(n if n else ">200000") for n in ns],
                "achieved" if tt.sig else "not-achieved",
            ]
        )

    # The handoff's "n~43 seeds for 0.0053 nats" claim, checked against the real sds.
    print("\n  CHECK of the handoff claim 'the 0.0053-nat gap needs n~43 seeds':")
    print(f"    n=43 at 80% power implies a paired sd of ~{0.0053/ (t_ppf(0.975,42)+t_ppf(0.8,42)) * math.sqrt(43):.5f} nats.")
    for label in ("hh4-hh1", "hh4-r1wide", "r1wide-hh1"):
        tt = results.get(("val_loss", label, TRAIN_LEN))
        if tt:
            n = n_for_power(0.0053, tt.sd)
            print(
                f"    observed paired sd for {label:>11} @2048 = {tt.sd:.5f} nats "
                f"-> n needed for 0.0053 nats = {n}"
            )
    # unpaired (marginal) sd, which is what the handoff appears to have used
    for a in arms:
        vs = [val(D[(a, s)], TRAIN_LEN) for s in seeds_of[a]]
        n = n_for_power(0.0053, st.stdev(vs))
        print(
            f"    marginal (UNPAIRED) sd for {a:>11} @2048 = {st.stdev(vs):.5f} nats "
            f"-> n = {n}   <- the unpaired route is what inflates the estimate"
        )

    write_tsv(
        os.path.join(args.out_dir, "lm_power.tsv"),
        [
            "endpoint", "contrast", "eval_len", "n_current", "observed_sd_paired",
            "observed_effect", "observed_dz", "mde80_at_n4", "mde80_at_n5", "mde80_at_n10",
            "n_for_observed_effect",
            *[f"n_for_{e:g}_nats" for e in probe_effects],
            "significance_at_current_n",
        ],
        power_rows,
    )

    # ---------------------------------------------------------------------------------
    # Verdict under the pre-registered rule
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("VERDICT UNDER THE PRE-REGISTERED DECISION RULE")
    print("=" * 88)

    def branch(endpoint: str, L: int) -> str:
        a = results.get((endpoint, "hh4-hh1", L))
        b = results.get((endpoint, "hh4-r1wide", L))
        if a is None or b is None:
            return "indeterminate (missing contrast)"
        hh4_beats_hh1 = a.sig and a.mean < 0
        hh4_beats_r1w = b.sig and b.mean < 0
        r1w_beats_hh4 = b.sig and b.mean > 0
        if hh4_beats_hh1 and hh4_beats_r1w:
            return "BRANCH 1: the effect is R"
        if hh4_beats_hh1 and r1w_beats_hh4:
            return "BRANCH 2+ : capacity, and R is ACTIVELY HARMFUL at matched params"
        if hh4_beats_hh1 and not b.sig:
            return "BRANCH 2: capacity; R contributes nothing"
        if not a.sig:
            return "BRANCH 3: no detectable effect at this scale"
        return "indeterminate"

    for L in LENGTHS[1:]:
        print(f"  PRIMARY   degradation @L={L:<6}: {branch('degradation', L)}")
    print(f"  SECONDARY val_loss   @L={TRAIN_LEN:<6}: {branch('val_loss', TRAIN_LEN)}")
    print(f"  TERTIARY  train_loss           : {branch('final_train_loss', TRAIN_LEN)}")

    print("\n  Equivalence bounds on the PRIMARY (what the tight null actually licenses):")
    for L in LENGTHS[1:]:
        tt = results[("degradation", "hh4-r1wide", L)]
        print(
            f"    R's differential length-extrapolation benefit at L={L:>5} is bounded to "
            f"[{tt.lo:+.5f}, {tt.hi:+.5f}] nats (95% CI, n={tt.n})"
        )

    # ---------------------------------------------------------------------------------
    # Variance decomposition. This is the load-bearing methodological result: it shows
    # WHICH variance component pairing removes, and therefore which endpoint is broken
    # and why. Between-seed marginal sd and paired-contrast sd have different fixes
    # (more eval windows vs more seeds), and conflating them misdiagnoses the study.
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("VARIANCE DECOMPOSITION -- what pairing removes, and what it does not")
    print("=" * 88)
    print(
        f"  {'quantity':>44} {'sd':>9} {'vs marginal':>12}\n"
        f"  {'-'*44} {'-'*9} {'-'*12}"
    )
    marg = st.mean(
        [st.stdev([val(D[(a, s)], TRAIN_LEN) for s in seeds_of[a]]) for a in arms]
    )
    print(f"  {'per-arm val loss @2048, across seeds (marginal)':>44} {marg:>9.5f} {1.0:>11.2f}x")
    for L in LENGTHS[1:]:
        dgs = st.mean([st.stdev([degradation(D[(a, s)], L) for s in seeds_of[a]]) for a in arms])
        print(
            f"  {f'per-arm degradation @{L}, across seeds':>44} {dgs:>9.5f} "
            f"{dgs/marg:>11.2f}x"
        )
    vl = st.mean([results[("val_loss", c[2], TRAIN_LEN)].sd for c in CONTRASTS])
    print(f"  {'PAIRED val-loss contrast @2048':>44} {vl:>9.5f} {vl/marg:>11.2f}x")
    for L in LENGTHS[1:]:
        dc = st.mean([results[("degradation", c[2], L)].sd for c in CONTRASTS])
        print(f"  {f'PAIRED degradation contrast @{L}':>44} {dc:>9.5f} {dc/marg:>11.2f}x")
    print(
        f"\n  Pairing on seed shrinks the sd by ~{marg/vl:.0f}x on val loss and by "
        f"~{marg/st.mean([results[('degradation', c[2], 16384)].sd for c in CONTRASTS]):.0f}x on "
        f"degradation, because the eval-window draw is a function of (seed, L) ONLY -- not of\n"
        f"  the arm -- so it is a shared additive nuisance that differences out exactly.\n"
        f"  CONSEQUENCE: the ARM CONTRASTS are precise (MDE ~0.01-0.02 nats at n=4). The study\n"
        f"  is NOT noise-limited in the arm comparison. What is broken is the PHENOMENON the\n"
        f"  primary endpoint was supposed to measure -- see below."
    )

    # ---------------------------------------------------------------------------------
    # Is the primary endpoint VACUOUS (rather than merely underpowered)?
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("IS THE PRIMARY ENDPOINT VACUOUS?")
    print("=" * 88)
    pooled16 = results[("degradation_pooled_all_arms", "pooled", 16384)]
    print(
        f"  The endpoint is 'which arm resists length degradation better'. That presupposes\n"
        f"  degradation exists. Pooled over all 13 runs, 2048 -> 16384 degradation is\n"
        f"  {pooled16.mean:+.4f} nats (95% CI [{pooled16.lo:+.4f},{pooled16.hi:+.4f}]) -- i.e. loss\n"
        f"  DECREASES at longer evaluation lengths, versus the +1 to +5 nats assumed.\n"
        f"  Sign of degradation, per run, at L=16384:"
    )
    neg = sum(1 for rec in D.values() if degradation(rec, 16384) < 0)
    print(f"    negative (loss improves) in {neg}/{len(D)} runs; positive in {len(D)-neg}/{len(D)}")
    print(
        "\n  THREE reasons degradation is <= 0 here, none of which the power analysis anticipated:\n"
        "   1. NO POSITIONAL ENCODING ANYWHERE. train_lm.py has no RoPE/ALiBi/learned positions,\n"
        "      and neither does the KDA/Householder layer. Position enters only via the causal\n"
        "      recurrence and the depthwise conv. The +1..+5 nat prediction is imported from\n"
        "      softmax transformers with a positional-extrapolation cliff; there is no cliff here.\n"
        "   2. FIXED-SIZE RECURRENT STATE. Memory/compute per token is O(1) in L, so nothing\n"
        "      structurally breaks when L grows past the training length.\n"
        "   3. COLD-START DILUTION, and this one biases the metric negative by construction.\n"
        "      The reported loss is the unweighted MEAN over every position in the window. The\n"
        "      first tokens of a window are predicted from an empty recurrent state and cost more\n"
        "      nats; call that prefix c. Those expensive tokens are a fraction c/L of the average,\n"
        "      so the SAME model scores lower simply because the penalty is diluted 8x from\n"
        "      L=2048 to L=16384. With c on the order of a few hundred tokens this is a shift of\n"
        "      the observed size and sign. loss(L)-loss(2048) therefore mixes cold-start dilution\n"
        "      with any true extrapolation effect and CANNOT separate them. It is not measurable\n"
        "      from the saved artefacts: train_lm.py saves no checkpoints and logs only the\n"
        "      window-mean, so per-position losses are unrecoverable without retraining.\n"
        "  => The primary endpoint is CONFOUNDED AND VACUOUS, not merely underpowered: there is\n"
        "     no degradation for R to mitigate, and the metric's sign is pre-biased negative."
    )

    # ---------------------------------------------------------------------------------
    # The confound in the capacity control itself.
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("CONFOUND IN THE 'PARAMETER-MATCHED' CONTROL -- matched on NON-EMBED only")
    print("=" * 88)
    h4, rw = D[("hh4", 0)], D[("hh4_r1wide", 0)]
    e4 = h4["n_params"] - h4["n_params_nonembed"]
    ew = rw["n_params"] - rw["n_params_nonembed"]
    print(
        f"  {'':>14}{'non-embed':>12}{'embed(tied)':>14}{'TOTAL':>13}\n"
        f"  {'hh4':>14}{h4['n_params_nonembed']:>12,}{e4:>14,}{h4['n_params']:>13,}\n"
        f"  {'hh4_r1wide':>14}{rw['n_params_nonembed']:>12,}{ew:>14,}{rw['n_params']:>13,}\n"
        f"  {'delta':>14}{rw['n_params_nonembed']-h4['n_params_nonembed']:>+12,}"
        f"{ew-e4:>+14,}{rw['n_params']-h4['n_params']:>+13,}\n"
        f"  {'delta %':>14}{100*(rw['n_params_nonembed']/h4['n_params_nonembed']-1):>+11.2f}%"
        f"{100*(ew/e4-1):>+13.2f}%{100*(rw['n_params']/h4['n_params']-1):>+12.2f}%\n"
        f"\n  Non-embed params match to {100*(rw['n_params_nonembed']/h4['n_params_nonembed']-1):+.2f}%"
        f" as designed, BUT r1wide carries {(ew-e4)/1e6:+.2f}M\n"
        f"  ({100*(ew/e4-1):+.1f}%) more embedding parameters, because the tied embedding/head is\n"
        f"  vocab x d_model and d_model went 512 -> 616. Total params differ by "
        f"{100*(rw['n_params']/h4['n_params']-1):+.1f}%.\n"
        f"  Since the head is TIED to the embedding, r1wide also gets a wider OUTPUT projection.\n"
        f"  => r1wide's win over hh4 is 'width (incl. a 20% larger tied embedding/head) beats R',\n"
        f"     NOT the cleaner 'mixer width beats R'. This is a first-order caveat on the single\n"
        f"     strongest positive finding in the study and must be stated in the write-up."
    )

    # ---------------------------------------------------------------------------------
    # Compute-normalised view: R=4 is dominated on every axis.
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("COMPUTE-NORMALISED VIEW -- cost of R at equal steps/tokens")
    print("=" * 88)
    base = st.mean([D[("hh1", s)]["wall_seconds"] for s in seeds_of["hh1"]])
    print(f"  {'arm':>12} {'s/step':>8} {'wall(h)':>9} {'vs hh1':>8} {'val@2048':>10}")
    for a in arms:
        w = st.mean([D[(a, s)]["wall_seconds"] for s in seeds_of[a]])
        v = st.mean([val(D[(a, s)], TRAIN_LEN) for s in seeds_of[a]])
        print(f"  {a:>12} {w/r0['steps']:>8.3f} {w/3600:>9.2f} {w/base:>7.2f}x {v:>10.4f}")
    wh4 = st.mean([D[("hh4", s)]["wall_seconds"] for s in seeds_of["hh4"]])
    wrw = st.mean([D[("hh4_r1wide", s)]["wall_seconds"] for s in seeds_of["hh4_r1wide"]])
    print(
        f"\n  At matched non-embed params and matched tokens, R=4 costs {wh4/wrw:.2f}x the wall-clock\n"
        f"  of the R=1 control AND is {results[('val_loss','hh4-r1wide',TRAIN_LEN)].mean:+.4f} nats\n"
        f"  WORSE. R=4 is therefore dominated on both axes; no compute-matched reweighting can\n"
        f"  rescue it at this scale."
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
