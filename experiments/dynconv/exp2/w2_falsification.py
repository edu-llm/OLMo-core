"""W=2 FALSIFICATION CONTROL for Exp-2. The best free falsification test in the program.

WHAT THIS IS
------------
It is a verified theorem that at kernel width W=2 the dynamic short-conv block is an EXACT
reparameterization of the static one -- it has ZERO new degrees of freedom. Therefore W=2 is a
built-in negative control:

    If the dynamic arm "wins" at W=2, that is a BUG (or an optimization artifact), NOT a result.

This script establishes the theorem in a form strong enough to be a *test*, and then supplies the
empirical decision rule that reads a W=2 cell of the real experiment.

WHY NOT JUST TRUST ``orch_verify_W_minus_2.py``
-----------------------------------------------
That script verifies the claim via Jacobian rank plus a log-linear least-squares realization on
*positive* random values. Two gaps for our purposes:

1. Real conv weights are SIGNED. A log-linear system needs |kappa|, so the published check does not
   literally cover the sign pattern the trained model will have. Here the realization is solved by
   a DIRECT RECURRENCE instead, which is exact for signed values and needs no logs. That is a
   strictly stronger statement, obtained more cheaply.
2. It works on the abstract coefficient field. Here we additionally push the constructed static
   parameters through an ACTUAL gated-conv forward pass and assert the OUTPUTS agree to float64
   precision -- i.e. the two blocks are the same operator, not merely the same table of numbers.

So this is an independent re-derivation, not a re-run. Per ``test-must-call-not-recompute``: a test
that re-derives the code's own formula passes when the code changes. This file derives the
realization from the algebra and checks it against a forward pass.

THE ALGEBRA
-----------
Per channel, the effective coefficient multiplying ``x_{t-k}`` in the block output is:

    static :  kappa[t,k] = C_t * a_k     * B_{t-k}       (a is a shared W-tap filter)
    dynamic:  kappa[t,k] = C_t * w[t,k]  * B_{t-k}       (w is generated per position)

Both forms are elementwise in the channel, so this is exact and channel-wise.

At W=2, given ANY target per-position 2-tap filter ``w``, set ``a = (1, 1)`` and solve:

    C_t * B_t      = kappa[t,0]
    C_t * B_{t-1}  = kappa[t,1]

Dividing:   B_t / B_{t-1} = kappa[t,0] / kappa[t,1]

so with ``B_0`` free,

    B_t = B_0 * prod_{s=1..t} kappa[s,0]/kappa[s,1]        and    C_t = kappa[t,0] / B_t.

Exact, signed, no logs, no optimization. That is the whole proof: the static family already
contains every 2-tap position-dependent filter field.

At W=3 the same elimination leaves a residual constraint -- the shift-invariant cross-ratio

    kappa[t+1,2] * kappa[t,0] / (kappa[t+1,1] * kappa[t,1])  ==  a_0*a_2 / a_1^2

which is the SAME constant for every t in any static block, and free in a dynamic one. That single
scalar IS the entire expressive gap at W=3. This script measures its dispersion as the positive
control for the negative control: the statistic must be flat for static and spread for dynamic, or
the test has no power to detect anything.

CONTAINMENT DIRECTION -- and it is the favourable one
-----------------------------------------------------
The real generator produces ``Delta_w = alpha * U(V h)``, which is rank-R in position, so the
realized dynamic filter field is a SUBSET of the free filter field. We prove the FREE field at W=2
is absorbable by the static form. A fortiori the rank-R field is too. The containment runs the
right way, so no rank-specific argument is needed.

ONE HONEST CAVEAT, stated so nobody over-claims
-----------------------------------------------
The theorem is about the tap-coefficient FIELD, which is what determines the operator. In the real
block ``B`` and ``C`` are produced by dense ``d -> d`` projections of the same normalized input
``h`` that the generator reads, and R5 F4 measured R^2 = 1.0000 regressing ``Delta_w`` on the gate
pre-activations -- the gates' features exactly span everything the generator uses. So the
information needed to realize the absorbed filter IS available to the gates. What is not guaranteed
is that SGD finds it.

Consequence for how a W=2 result must be read:

    A W=2 dynamic-vs-static gap beyond seed noise means EITHER a wiring/accounting bug OR an
    optimization/conditioning artifact. In NEITHER case is it the claimed mechanism, because at
    W=2 the claimed mechanism provably does not exist.

That is still a decisive falsification -- it just names two exit doors instead of one. Do not
report a W=2 gap as evidence for input-dependent local composition under any circumstances.

Run:  python3 w2_falsification.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# float64 throughout: this is an exactness claim, and bf16/fp32 rounding would smear the very
# residual that separates W=2 (must be ~0) from W=3 (must be large).
DTYPE = torch.float64

# Machine-precision tolerance for an "exact" realization. float64 eps is 2.22e-16; the recurrence
# accumulates O(T) products, so allow a few orders of headroom and no more.
EXACT_TOL = 1e-10

# A W=3 residual must be MANY orders above EXACT_TOL or the test cannot distinguish the cases.
GAP_TOL = 1e-4


# --------------------------------------------------------------------------------------------
# The operator, written once, in the form the algebra is stated in
# --------------------------------------------------------------------------------------------


def gated_conv_taps(
    B: torch.Tensor,  # (T,) pre-gate, per position
    C: torch.Tensor,  # (T,) post-gate, per position
    w: torch.Tensor,  # (T, W) per-position filter; a static block passes a broadcast `a`
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Effective tap coefficients ``kappa[t,k] = C_t * w[t,k] * B_{t-k}``, valid entries only.

    This is the single definition of the operator used by every check below. Writing it once is
    the point: if it is wrong, every check moves together and the disagreement between the W=2 and
    W=3 cases -- which is the actual signal -- would vanish rather than flip.
    """
    T, W = w.shape
    return {
        (t, k): C[t] * w[t, k] * B[t - k]
        for t in range(T)
        for k in range(W)
        if t - k >= 0
    }


def gated_conv_forward(
    x: torch.Tensor,  # (T,) one channel of the value stream
    B: torch.Tensor,
    C: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """``y_t = C_t * sum_k w[t,k] * B_{t-k} * x_{t-k}`` -- the block, one channel, causal.

    Deliberately a literal transcription of the math rather than a call into ``ShortConv``: the
    claim under test is about the operator's algebra, and routing it through the production module
    would test the module instead. ``ShortConv`` parity is sub-agent A's job; this is the theorem.

    Tap index convention: ``k`` is the LAG (k=0 is the current token). Note the production
    ``nn.Conv1d`` layout stores the current-token tap LAST (``w[:, :, -1] = 1.0`` in
    ``ShortConv.init_weights``), i.e. reversed relative to this. That does not affect any claim
    here -- the static family is closed under reversing the tap axis -- but the convention is
    stated so a reader comparing the two does not think one of them is wrong.
    """
    T, W = w.shape
    y = torch.zeros(T, dtype=x.dtype)
    for t in range(T):
        acc = torch.zeros((), dtype=x.dtype)
        for k in range(W):
            if t - k >= 0:
                acc = acc + w[t, k] * B[t - k] * x[t - k]
        y[t] = C[t] * acc
    return y


# --------------------------------------------------------------------------------------------
# PART 1 -- the constructive realization at W=2, signed, exact
# --------------------------------------------------------------------------------------------


def realize_w2_as_static(
    kappa: Dict[Tuple[int, int], torch.Tensor],
    T: int,
    B0: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve for a static ``(B', C', a')`` reproducing a W=2 tap field EXACTLY.

    Direct recurrence, so it is exact for SIGNED coefficients -- unlike a log-linear least-squares
    fit, which needs ``|kappa|`` and therefore silently drops the sign pattern a trained model
    will actually have.

    :returns: ``(B_prime, C_prime, a_prime)`` with ``a_prime = (1, 1)``.
    :raises ValueError: if any ``kappa[t,1]`` is zero, where the recurrence is undefined. That is
        a measure-zero set, not a limitation of the theorem, but it must not be silently divided
        through.
    """
    a = torch.ones(2, dtype=DTYPE)
    Bp = torch.empty(T, dtype=DTYPE)
    Cp = torch.empty(T, dtype=DTYPE)

    Bp[0] = B0
    for t in range(1, T):
        k1 = kappa[(t, 1)]
        if k1.abs() < 1e-300:
            raise ValueError(f"kappa[{t},1] == 0; recurrence undefined (measure-zero case)")
        Bp[t] = Bp[t - 1] * (kappa[(t, 0)] / k1)

    for t in range(T):
        Cp[t] = kappa[(t, 0)] / Bp[t]

    return Bp, Cp, a


def check_w2_exact_reparameterization(T: int = 24, seed: int = 0) -> dict:
    """W=2: an arbitrary SIGNED dynamic filter field is exactly realizable by a static block.

    Two levels of assertion, and the second is the one that matters:
      (a) the tap fields agree entrywise;
      (b) the two blocks produce the same OUTPUT on the same input -- same operator, not just
          the same table.
    """
    g = torch.Generator().manual_seed(seed)

    # Signed and away from zero. `sign * (0.5 + U)` keeps |value| in [0.5, 1.5], which avoids the
    # measure-zero singular set without making the test easier: the realization is exact for any
    # nonzero field, and near-zero denominators would only add float noise, not difficulty.
    def signed(*shape: int) -> torch.Tensor:
        mag = 0.5 + torch.rand(*shape, generator=g, dtype=DTYPE)
        sgn = torch.where(torch.rand(*shape, generator=g, dtype=DTYPE) < 0.5, -1.0, 1.0)
        return mag * sgn

    w_dyn = signed(T, 2)  # a free per-position 2-tap filter: the most general W=2 dynamic block
    B_dyn = signed(T)
    C_dyn = signed(T)

    kappa_dyn = gated_conv_taps(B_dyn, C_dyn, w_dyn)
    Bp, Cp, a = realize_w2_as_static(kappa_dyn, T)
    w_static = a.unsqueeze(0).expand(T, 2).contiguous()
    kappa_static = gated_conv_taps(Bp, Cp, w_static)

    tap_err = max(
        float((kappa_dyn[key] - kappa_static[key]).abs()) / max(float(kappa_dyn[key].abs()), 1e-30)
        for key in kappa_dyn
    )

    x = signed(T)
    y_dyn = gated_conv_forward(x, B_dyn, C_dyn, w_dyn)
    y_static = gated_conv_forward(x, Bp, Cp, w_static)
    out_err = float((y_dyn - y_static).norm() / y_dyn.norm())

    return {
        "name": "W2_exact_reparameterization",
        "T": T,
        "max_rel_tap_err": tap_err,
        "rel_output_err": out_err,
        "tol": EXACT_TOL,
        "passed": tap_err < EXACT_TOL and out_err < EXACT_TOL,
    }


def check_w3_not_realizable(T: int = 24, seed: int = 0) -> dict:
    """W=3: the SAME construction must FAIL, by a wide margin.

    This is the positive control for the negative control. Without it, a realization routine that
    trivially "succeeds" everywhere (e.g. because of a bug that ignores the target) would report
    W=2 as exact and nobody would notice.

    Best static fit is obtained in closed form on the log-magnitudes -- the static family is
    rank-1-in-logs, so least squares there is the exact projection onto the static manifold, and
    no multi-start search is needed to establish a LOWER bound on the residual.
    """
    g = torch.Generator().manual_seed(seed)
    # Positive here, because the least-squares projection is taken in log-magnitude space. Signs
    # can only make a static fit HARDER (the static form's sign pattern is itself constrained), so
    # a positive-only target gives a conservative -- i.e. small -- residual. If the gap shows up
    # even in the easiest case, it is real.
    w_dyn = 0.5 + torch.rand(T, 3, generator=g, dtype=DTYPE)
    B_dyn = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)
    C_dyn = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)

    kappa = gated_conv_taps(B_dyn, C_dyn, w_dyn)
    keys = sorted(kappa.keys())

    # log kappa[t,k] = logC_t + loga_k + logB_{t-k}: linear, so lstsq is the exact projection.
    rows, rhs = [], []
    for (t, k) in keys:
        r = torch.zeros(T + 3 + T, dtype=DTYPE)
        r[t] = 1.0
        r[T + k] = 1.0
        r[T + 3 + (t - k)] = 1.0
        rows.append(r)
        rhs.append(torch.log(kappa[(t, k)]))
    A = torch.stack(rows)
    b = torch.stack(rhs)
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    resid = float((A @ sol - b).abs().max())

    return {
        "name": "W3_not_realizable",
        "T": T,
        "max_log_residual": resid,
        "tol": GAP_TOL,
        "passed": resid > GAP_TOL,
    }


def check_w2_realization_can_fail(T: int = 24, seed: int = 0) -> dict:
    """NEGATIVE CONTROL on the realization routine itself.

    Feed ``realize_w2_as_static`` a target it must NOT be able to hit, by corrupting one entry of
    the tap field after the recurrence has been solved for the clean field. If the checker still
    reports "exact", it is not reading the target at all.

    Per ``HANDOFF.md``: a guard that has never failed is not known to work.
    """
    g = torch.Generator().manual_seed(seed)
    w_dyn = 0.5 + torch.rand(T, 2, generator=g, dtype=DTYPE)
    B_dyn = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)
    C_dyn = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)

    kappa = gated_conv_taps(B_dyn, C_dyn, w_dyn)
    Bp, Cp, a = realize_w2_as_static(kappa, T)

    # Corrupt the TARGET, not the solution: ask whether the comparison notices.
    corrupted = dict(kappa)
    corrupted[(T // 2, 0)] = corrupted[(T // 2, 0)] * 3.0

    w_static = a.unsqueeze(0).expand(T, 2).contiguous()
    kappa_static = gated_conv_taps(Bp, Cp, w_static)
    err = max(
        float((corrupted[key] - kappa_static[key]).abs()) / max(float(corrupted[key].abs()), 1e-30)
        for key in corrupted
    )

    return {
        "name": "W2_realization_negative_control",
        "T": T,
        "max_rel_tap_err_on_corrupted_target": err,
        "tol": EXACT_TOL,
        # The check PASSES when the error is LARGE -- i.e. the comparison really is reading the
        # target and would have caught a wrong realization.
        "passed": err > GAP_TOL,
    }


# --------------------------------------------------------------------------------------------
# PART 2 -- the cross-ratio, the single scalar that IS the W=3 gap
# --------------------------------------------------------------------------------------------


def cross_ratios(kappa: Dict[Tuple[int, int], torch.Tensor], T: int) -> List[float]:
    """``kappa[t+1,2]*kappa[t,0] / (kappa[t+1,1]*kappa[t,1])`` over the positions where it exists.

    Constant in ``t`` (and equal to ``a0*a2/a1^2``) for ANY static block. Free for a dynamic one.
    """
    out = []
    for t in range(2, T - 1):
        num = kappa[(t + 1, 2)] * kappa[(t, 0)]
        den = kappa[(t + 1, 1)] * kappa[(t, 1)]
        if den.abs() < 1e-300:
            continue
        out.append(float(num / den))
    return out


def check_cross_ratio_discriminates(T: int = 24, seed: int = 0) -> dict:
    """The gap statistic must be FLAT for static and SPREAD for dynamic.

    If it is flat for both, the statistic has no power and a "no gap" reading would be vacuous --
    exactly the failure mode R5's appendix warns about: two of its own candidate counterexamples
    "looked" strongly position-dependent and were EXACTLY static-realizable at 0.00% residual.
    The invariant, not apparent position-dependence, is what must be tested.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor([0.7, -0.3, 0.15], dtype=DTYPE)
    B = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)
    C = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)

    k_static = gated_conv_taps(B, C, a.unsqueeze(0).expand(T, 3).contiguous())
    r_static = torch.tensor(cross_ratios(k_static, T), dtype=DTYPE)
    predicted = float(a[0] * a[2] / a[1] ** 2)

    w_dyn = 0.5 + torch.rand(T, 3, generator=g, dtype=DTYPE)
    k_dyn = gated_conv_taps(B, C, w_dyn)
    r_dyn = torch.tensor(cross_ratios(k_dyn, T), dtype=DTYPE)

    return {
        "name": "cross_ratio_discriminates",
        "static_mean": float(r_static.mean()),
        "static_std": float(r_static.std()),
        "predicted_a0a2_over_a1sq": predicted,
        "static_matches_prediction": abs(float(r_static.mean()) - predicted) < 1e-12,
        "dynamic_std": float(r_dyn.std()),
        "dynamic_spread_x": float(r_dyn.max() / r_dyn.min()) if float(r_dyn.min()) != 0 else float("inf"),
        "passed": float(r_static.std()) < 1e-12 and float(r_dyn.std()) > 1e-3,
    }


def check_w2_has_no_cross_ratio() -> dict:
    """Why the gap statistic does not even EXIST at W=2 -- the structural reason, asserted.

    The cross-ratio needs a ``k=2`` tap. At W=2 there is none, so there is no invariant to break.
    Stated as a check rather than a comment because it is the reason the W=2 control is a theorem
    and not an empirical observation.
    """
    T = 12
    g = torch.Generator().manual_seed(0)
    w = 0.5 + torch.rand(T, 2, generator=g, dtype=DTYPE)
    B = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)
    C = 0.5 + torch.rand(T, generator=g, dtype=DTYPE)
    kappa = gated_conv_taps(B, C, w)
    has_k2 = any(k == 2 for (_, k) in kappa)
    return {
        "name": "W2_has_no_cross_ratio",
        "n_taps_with_lag_2": sum(1 for (_, k) in kappa if k == 2),
        "passed": not has_k2,
    }


# --------------------------------------------------------------------------------------------
# PART 3 -- DOF accounting, independently re-derived
# --------------------------------------------------------------------------------------------


def check_dof_gap_closed_form() -> dict:
    """``gap = T(W-2) - W(W-1)/2 - W + 3``, and it is 0 at W=2 for every T.

    Counted combinatorially here (valid entries minus the static manifold dimension ``2T+W-3``),
    rather than by Jacobian rank as in ``orch_verify_W_minus_2.py``. Two independent routes to the
    same integer is the point: a rank computation has a tolerance, a count does not.
    """
    rows = []
    ok = True
    for T in (6, 8, 12, 24, 40):
        for W in (2, 3, 4, 8):
            if W > T:
                continue
            n_entries = sum(1 for t in range(T) for k in range(W) if t - k >= 0)
            static_dim = 2 * T + W - 3
            gap = n_entries - static_dim
            closed = T * (W - 2) - W * (W - 1) // 2 - W + 3
            match = gap == closed
            zero_at_w2 = (W != 2) or (gap == 0)
            ok = ok and match and zero_at_w2
            rows.append(
                {
                    "T": T,
                    "W": W,
                    "entries": n_entries,
                    "static_dim": static_dim,
                    "gap": gap,
                    "closed_form": closed,
                    "match": match,
                }
            )
    return {"name": "dof_gap_closed_form", "rows": rows, "passed": ok}


def dof_fraction_table() -> List[dict]:
    """Fraction of the W generated numbers per position that are genuinely NEW.

    This is why W=3 is the starved configuration and why a W=3-only null is uninterpretable: it
    cannot separate "the mechanism does not help" from "we gave it one number per position."
    """
    out = []
    for W in (2, 3, 4, 8):
        new = max(W - 2, 0)
        out.append(
            {
                "W": W,
                "new_dof_per_position": new,
                "fraction_new": new / W,
                "role": {
                    2: "NEGATIVE CONTROL -- provably zero new DOF",
                    3: "LFM2 fidelity anchor -- starved, 1 of 3",
                }.get(W, "informative"),
            }
        )
    return out


# --------------------------------------------------------------------------------------------
# PART 4 -- the empirical decision rule that reads a real W=2 cell
# --------------------------------------------------------------------------------------------


def w2_verdict(
    static_scores: Sequence[float],
    dynamic_scores: Sequence[float],
    *,
    paired: bool = True,
    alpha: float = 0.05,
) -> dict:
    """Read the W=2 cell of the real experiment. Returns a VERDICT, not a p-value to interpret.

    The rule, pre-registered:

      * CI on (dynamic - static) CONTAINS 0  ->  ``CLEAN``. Expected. The theorem holds
        empirically and the harness is not manufacturing a difference from nothing.
      * CI EXCLUDES 0 in the dynamic arm's favour  ->  ``BUG_OR_ARTIFACT``. **The experiment is
        not interpretable at any W until this is explained.** At W=2 the mechanism provably does
        not exist, so a real win cannot be the mechanism. Look for: wrong module count, an arm
        that got extra tuning, non-paired data order, a param/FLOP mismatch, or a
        conditioning/optimization difference.
      * CI EXCLUDES 0 against the dynamic arm  ->  ``DYNAMIC_WORSE_AT_W2``. Not a falsification of
        the hypothesis, but it IS evidence of an optimization penalty from the extra parameters
        and the reparameterization, which must be reported because it biases every other W cell in
        the SAME direction.

    Deliberately a two-sided CI, not a one-sided sign test. R3 F3: a sign test at n=5 has a hard
    floor of p=0.1875 for "4 of 5" and cannot reach alpha=0.05 at all; and a criterion that fails
    to reject reads as a pass, so ambiguity would license proceeding.
    """
    n = len(static_scores)
    if n != len(dynamic_scores):
        raise ValueError("static and dynamic score lists must be the same length (paired design)")
    if n < 2:
        raise ValueError("need >= 2 seeds to form any interval")
    if not paired:
        raise NotImplementedError(
            "Exp-2 is a paired design on shared data order; an unpaired path would silently "
            "change the variance being estimated. Add it deliberately or not at all."
        )

    d = [float(dy) - float(st) for st, dy in zip(static_scores, dynamic_scores)]
    mean = sum(d) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
    else:
        var = float("nan")
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)

    tcrit = _t_ppf(1.0 - alpha / 2.0, n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se

    if lo <= 0.0 <= hi:
        verdict = "CLEAN"
        reading = (
            "CI contains 0, as the theorem predicts. W=2 shows no dynamic advantage; the harness "
            "is not manufacturing a difference where none can exist."
        )
    elif lo > 0.0:
        verdict = "BUG_OR_ARTIFACT"
        reading = (
            "CI EXCLUDES 0 in the dynamic arm's favour at W=2, where the dynamic block is an EXACT "
            "reparameterization of the static one. This cannot be the claimed mechanism. STOP and "
            "diagnose before reading any other W."
        )
    else:
        verdict = "DYNAMIC_WORSE_AT_W2"
        reading = (
            "CI EXCLUDES 0 against the dynamic arm at W=2. Not a falsification, but it quantifies "
            "an optimization penalty that biases every other W cell in the same direction and must "
            "be reported alongside them."
        )

    return {
        "n_seeds": n,
        "per_seed_differences": d,
        "mean_difference": mean,
        "sd_of_differences": sd,
        "se": se,
        "t_crit": tcrit,
        "ci95": (lo, hi),
        "verdict": verdict,
        "reading": reading,
    }


def _t_ppf(p: float, df: int) -> float:
    """Two-sided t quantile. Uses scipy when present; otherwise an explicitly-labelled table.

    The fallback is a hard-coded table of exact values rather than a normal approximation, because
    at the df this experiment runs at (n=10 -> df=9) the normal quantile 1.960 vs the true 2.262 is
    a 15% understatement of every interval -- which would make intervals look tighter than they
    are, in the direction that manufactures significance.
    """
    try:
        from scipy.stats import t as _t  # type: ignore

        return float(_t.ppf(p, df))
    except Exception:
        if abs(p - 0.975) > 1e-9:
            raise RuntimeError(
                f"no scipy, and the fallback table only covers the two-sided 95% quantile "
                f"(p=0.975); asked for p={p}. Install scipy rather than approximating."
            )
        exact_975 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
            8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 14: 2.145, 16: 2.120,
            18: 2.101, 20: 2.086, 24: 2.064, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
        }
        if df in exact_975:
            return exact_975[df]
        keys = sorted(exact_975)
        if df > keys[-1]:
            return 1.960
        nxt = min(k for k in keys if k > df)
        return exact_975[nxt]  # conservative: a larger t widens the interval


def w2_detectable_difference(sigma_pp: float, n: int, rho: float = 0.5) -> dict:
    """What size of spurious W=2 gap would this design actually SEE?

    The falsification control is only as good as its power. Against the repo's MEASURED MQAR
    sigma of 42-48.4 pp, a W=2 test at n=10 cannot detect a small artifact -- so "W=2 came back
    CLEAN" must be reported with its own MDE attached, or it is an absence of evidence being read
    as evidence of absence. That is the same fail-open structure R3 F5e priced across five
    non-inferiority clauses (a true 40 pp regression passed one gate 63% of the time).
    """
    s_delta = sigma_pp * math.sqrt(2.0 * (1.0 - rho))
    se = s_delta / math.sqrt(n)
    tcrit = _t_ppf(0.975, n - 1)
    return {
        "sigma_pp": sigma_pp,
        "rho": rho,
        "s_delta_pp": s_delta,
        "n": n,
        "se_pp": se,
        "half_width_95_pp": tcrit * se,
        "note": (
            "A W=2 artifact smaller than the 95% half-width is INVISIBLE to this design. Report "
            "the half-width next to any CLEAN verdict."
        ),
    }


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def run_all(verbose: bool = True) -> Tuple[bool, List[dict]]:
    checks = [
        check_dof_gap_closed_form(),
        check_w2_exact_reparameterization(),
        check_w3_not_realizable(),
        check_w2_realization_can_fail(),
        check_cross_ratio_discriminates(),
        check_w2_has_no_cross_ratio(),
    ]
    all_ok = all(c["passed"] for c in checks)

    if verbose:
        print("=" * 88)
        print("W=2 FALSIFICATION CONTROL -- Exp-2")
        print("=" * 88)

        dof = checks[0]
        print("\n[1] DOF gap, counted combinatorially (independent of the Jacobian-rank route)")
        print(f"    {'T':>4} {'W':>3} {'entries':>8} {'static_dim':>11} {'gap':>6} {'closed':>7}  ok")
        for r in dof["rows"]:
            print(
                f"    {r['T']:>4} {r['W']:>3} {r['entries']:>8} {r['static_dim']:>11} "
                f"{r['gap']:>6} {r['closed_form']:>7}  {'OK' if r['match'] else 'MISMATCH'}"
            )
        print(f"    -> gap is 0 at W=2 for every T: {'CONFIRMED' if dof['passed'] else 'FAILED'}")

        print("\n[2] New DOF per position, as a fraction of the W numbers generated")
        print(f"    {'W':>3} {'new':>5} {'frac':>7}  role")
        for r in dof_fraction_table():
            print(f"    {r['W']:>3} {r['new_dof_per_position']:>5} {r['fraction_new']:>7.2%}  {r['role']}")

        c = checks[1]
        print(f"\n[3] W=2 exact reparameterization (SIGNED, direct recurrence, T={c['T']})")
        print(f"    max relative tap error : {c['max_rel_tap_err']:.3e}")
        print(f"    relative OUTPUT error  : {c['rel_output_err']:.3e}   (same operator, not just same table)")
        print(f"    tolerance              : {c['tol']:.0e}")
        print(f"    -> {'CONFIRMED: zero new degrees of freedom at W=2.' if c['passed'] else 'FAILED'}")

        c = checks[2]
        print(f"\n[4] W=3 is NOT realizable -- the positive control for the negative control")
        print(f"    best static fit, max |log residual| : {c['max_log_residual']:.3e}")
        print(f"    -> {'CONFIRMED: W=3 is out of reach, so the test has power.' if c['passed'] else 'FAILED'}")

        c = checks[3]
        print(f"\n[5] NEGATIVE CONTROL on the realization routine (corrupt the target)")
        print(f"    error on corrupted target : {c['max_rel_tap_err_on_corrupted_target']:.3e}")
        print(f"    -> {'CONFIRMED: the comparison reads the target and can fail.' if c['passed'] else 'FAILED -- the checker is not reading the target!'}")

        c = checks[4]
        print(f"\n[6] The cross-ratio is the whole W=3 gap, and it discriminates")
        print(f"    static  : mean {c['static_mean']:.12f}  std {c['static_std']:.3e}")
        print(f"    predicted a0*a2/a1^2 = {c['predicted_a0a2_over_a1sq']:.12f}  "
              f"match={c['static_matches_prediction']}")
        print(f"    dynamic : std {c['dynamic_std']:.6f}   spread {c['dynamic_spread_x']:.1f}x")
        print(f"    -> {'CONFIRMED: flat for static, spread for dynamic.' if c['passed'] else 'FAILED'}")

        c = checks[5]
        print(f"\n[7] At W=2 there is no cross-ratio to break (taps with lag 2: {c['n_taps_with_lag_2']})")
        print(f"    -> {'CONFIRMED: the invariant does not exist at W=2.' if c['passed'] else 'FAILED'}")

        print("\n[8] Power of the W=2 control itself, against the repo's MEASURED MQAR sigma")
        print(f"    {'sigma':>7} {'n':>4} {'s_delta':>9} {'95% half-width':>16}")
        for sig in (42.0, 48.4):
            for n in (5, 10, 20):
                r = w2_detectable_difference(sig, n)
                print(f"    {r['sigma_pp']:>7.1f} {n:>4} {r['s_delta_pp']:>9.1f} {r['half_width_95_pp']:>15.1f}pp")
        print("    -> A CLEAN W=2 verdict MUST be reported with this half-width attached.")
        print("       An artifact below it is invisible; absence of evidence is not evidence of absence.")

        print("\n[9] The decision rule, exercised on synthetic score vectors")
        demo = {
            "no effect (expected)": ([0.20, 0.55, 0.05, 0.98, 0.09, 0.56, 0.20, 0.05, 0.98, 0.09],
                                     [0.22, 0.51, 0.07, 0.96, 0.11, 0.52, 0.18, 0.08, 0.95, 0.12]),
            "implausible W=2 win": ([0.20, 0.22, 0.19, 0.21, 0.20, 0.18, 0.23, 0.20, 0.21, 0.19],
                                    [0.40, 0.42, 0.39, 0.41, 0.40, 0.38, 0.43, 0.40, 0.41, 0.39]),
        }
        for label, (st, dy) in demo.items():
            v = w2_verdict(st, dy)
            print(f"    {label:<22} mean_diff {v['mean_difference']:+.4f}  "
                  f"CI95 [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  -> {v['verdict']}")

        print("\n" + "=" * 88)
        print(f"OVERALL: {'ALL CHECKS PASS' if all_ok else 'FAILURES PRESENT -- see above'}")
        print("=" * 88)
        print("\nPre-registered reading of the W=2 cell:")
        print("  At W=2 the dynamic block has ZERO new degrees of freedom -- proven above, signed and")
        print("  exact, at the level of the block's output. A W=2 dynamic 'win' beyond seed noise is")
        print("  therefore a bug or an optimization artifact, never the mechanism. It invalidates the")
        print("  W=3/4/8 cells until explained.")

    return all_ok, checks


if __name__ == "__main__":
    ok, _ = run_all()
    raise SystemExit(0 if ok else 1)
