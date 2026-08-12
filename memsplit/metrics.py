"""Divergence, noise-floor, and compute-to-threshold measurement.

## Divergence

The previous write-up reported "per-ability KL between arms on value tokens: 6.7
nats". Two problems. First, `grep -r kl_div` over that codebase returns nothing --
**no KL was ever computed**; the number is a difference of gold-token
cross-entropies, which is not a divergence between predictive distributions.
Second, 6.7 nats exceeds the Jensen-Shannon ceiling of ln 2 = 0.693 by an order of
magnitude, so as a *KL* it is a statement about near-disjointness whose magnitude
depends on the softmax floor rather than on anything about the model.

So: `jsd` is primary (symmetric, bounded, well defined without absolute
continuity), both KL directions are reported alongside with the direction stated,
and everything is returned as a **distribution over positions** rather than a
mean. A mean hides the finding: drift concentrates in a small fraction of
positions, and offsetting per-position differences cancel.

What is *not* a problem here, contrary to an early worry: the two arms share one
50,304-dimensional vocabulary with all five control tokens in-vocabulary for both,
so the distributions live on the same support and no renormalisation or optimal
transport is needed.

## The H2 noise floor -- read this before claiming H2

Published KL between two runs of the *same* model differing only in seed is
~0.1 bits/byte. The previous write-up's H2 evidence was "<= 0.08 nats outside fact
positions", which is plausibly **at or below that floor** -- i.e. absence of
evidence rather than evidence of absence. `seed_floor_report` exists to make the
comparison explicit: measure arm-vs-arm against seed-vs-seed on the same slices,
and if they are indistinguishable, say so. It is still a fine result, but a
bounded one.

## Compute to threshold

`compute_to_threshold` interpolates the first crossing and returns tokens, FLOPs
and optimizer steps, because a claim that survives on only one of those axes is
not a compute claim. It also refuses to report a crossing that is not bracketed,
which is the specific failure in "10-15x fewer tokens": the split arm's first
evaluated point was already at 99.8%, so its convergence was never resolved and
the ratio is a censored lower bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2.0)


# ------------------------------------------------------------------ divergence


def _safe(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, None)
    return p / p.sum(axis=-1, keepdims=True)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p||q) per row, in nats. Unbounded; state the direction when reporting."""
    p, q = _safe(p), _safe(q)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


def jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence per row, in nats. Bounded in [0, ln 2].

    Preferred as the headline number: symmetric, so no direction to justify, and
    bounded, so one pathological position cannot dominate an aggregate.
    """
    p, q = _safe(p), _safe(q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def rank_of(probs: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    """1-indexed rank of `token_ids` under `probs`. Immune to unboundedness."""
    chosen = np.take_along_axis(probs, token_ids[:, None], axis=-1)
    return (probs > chosen).sum(axis=-1) + 1


def rank_shift_classes(ranks: np.ndarray) -> dict:
    """Unshifted / marginal / shifted, using the established thresholds.

    rank 1 = unshifted, 2-3 = marginal, >3 = shifted. Reported as a companion to
    divergence because it is interpretable and cannot blow up.
    """
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    unshifted = int((ranks == 1).sum())
    marginal = int(((ranks > 1) & (ranks <= 3)).sum())
    shifted = int((ranks > 3).sum())
    return {
        "n": n,
        "unshifted": unshifted / n,
        "marginal": marginal / n,
        "shifted": shifted / n,
        "unshifted_or_marginal": (unshifted + marginal) / n,
    }


def divergence_report(
    p: np.ndarray,
    q: np.ndarray,
    roles: list[str] | None = None,
    gold_ids: np.ndarray | None = None,
) -> dict:
    """Full divergence readout: distributions, not means, stratified by role.

    `p`, `q` are (n_positions, vocab) probability arrays from the two arms on the
    same teacher-forced stream. Returns per-role percentiles for JSD and both KL
    directions, plus the JSD ceiling so a saturated value is obvious.
    """
    j = jsd(p, q)
    k_pq, k_qp = kl(p, q), kl(q, p)

    def _pack(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        out = {"n": int(mask.sum())}
        for name, arr in (("jsd", j), ("kl_p_q", k_pq), ("kl_q_p", k_qp)):
            v = arr[mask]
            out[name] = {
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)),
                "max": float(v.max()),
            }
        out["jsd_saturated_frac"] = float((j[mask] > 0.95 * LN2).mean())
        return out

    report = {
        "jsd_ceiling_nats": LN2,
        "all": _pack(np.ones(len(j), dtype=bool)),
    }
    if roles is not None:
        roles_arr = np.asarray(roles)
        for role in sorted(set(roles)):
            report[f"role:{role}"] = _pack(roles_arr == role)
    if gold_ids is not None:
        report["rank_shift_q_under_p"] = rank_shift_classes(rank_of(_safe(p), gold_ids))
    return report


def nats_to_bits_per_byte(nats: float, n_bytes: int, n_tokens: int) -> float:
    """Convert a per-token nat figure to bits per byte.

    Needed to place a divergence on the published interpretive scale, where
    across-seed KL is ~0.1, fine-tuning ~0.40, and random same-type model pairs
    ~0.95 bits/byte. Reporting nats alone leaves the reader with no yardstick.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")
    return nats * n_tokens / (LN2 * n_bytes)


@dataclass
class SeedFloor:
    arm_vs_arm: float
    seed_vs_seed: float
    ratio: float
    distinguishable: bool


def seed_floor_report(
    arm_vs_arm: float, seed_vs_seed_values: list[float], k: float = 2.0
) -> SeedFloor:
    """Is the arm difference bigger than the same-arm seed difference?

    If not, an H2-style "the arms differ by almost nothing outside fact positions"
    claim is unsupported -- not wrong, but bounded by the measurement floor, and it
    must be reported that way.
    """
    if not seed_vs_seed_values:
        raise ValueError("need at least one same-arm seed pair")
    floor = float(np.mean(seed_vs_seed_values))
    spread = float(np.std(seed_vs_seed_values)) if len(seed_vs_seed_values) > 1 else 0.0
    return SeedFloor(
        arm_vs_arm=arm_vs_arm,
        seed_vs_seed=floor,
        ratio=arm_vs_arm / floor if floor else float("inf"),
        distinguishable=arm_vs_arm > floor + k * spread,
    )


def checkpoint_noise_floor(values: list[float]) -> dict:
    """Run-to-run noise estimated from the last N checkpoints of ONE run.

    The relative standard deviation of a run's final checkpoints predicts
    init-seed noise, data-order noise and whole-run checkpoint noise (R^2 = 0.82,
    0.86, 0.95 respectively). So every run already carries a noise estimate and
    there is no excuse for an unquantified n=1 comparison. Use ~30 checkpoints.
    """
    if len(values) < 3:
        raise ValueError("need >= 3 checkpoints")
    v = np.asarray(values, dtype=np.float64)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    return {
        "n_checkpoints": len(values),
        "mean": mean,
        "sd": sd,
        "relative_sd": sd / mean if mean else float("nan"),
        "range": float(v.max() - v.min()),
        "implied_mde_pp_at_80_power": 2.8 * sd * 100.0,
    }


# ----------------------------------------------------------- compute accounting


@dataclass
class Crossing:
    threshold: float
    bracketed: bool
    tokens: float | None
    steps: float | None
    flops: float | None
    note: str = ""


def compute_to_threshold(
    steps: list[int],
    accuracy: list[float],
    threshold: float,
    tokens_per_step: float,
    flops_per_token: float,
) -> Crossing:
    """First crossing of `threshold`, on all three axes, or an honest refusal.

    Refuses when the first evaluated point is already above threshold: the
    crossing is then unbracketed and any ratio built on it is a censored lower
    bound, not a measurement. That is precisely what happened to the "10-15x
    fewer tokens" claim -- the split arm's first evaluated step was already at
    99.8%, so its convergence point was never resolved. Evaluate on a log-spaced
    grid from step 1 until the crossing is bracketed.
    """
    if len(steps) != len(accuracy) or not steps:
        raise ValueError("steps and accuracy must be parallel and non-empty")
    order = np.argsort(steps)
    s = np.asarray(steps, dtype=np.float64)[order]
    a = np.asarray(accuracy, dtype=np.float64)[order]

    if a[0] >= threshold:
        return Crossing(
            threshold, False, None, None, None,
            note=(
                f"first evaluated step ({int(s[0])}) is already at {a[0]:.3f} "
                f">= {threshold}; crossing is unbracketed and any ratio from it "
                "is a censored lower bound. Evaluate earlier (log-spaced from 1)."
            ),
        )
    idx = np.where(a >= threshold)[0]
    if len(idx) == 0:
        return Crossing(
            threshold, False, None, None, None,
            note=f"threshold {threshold} never reached (max {a.max():.3f})",
        )
    i = int(idx[0])
    s0, s1, a0, a1 = s[i - 1], s[i], a[i - 1], a[i]
    frac = 0.0 if a1 == a0 else (threshold - a0) / (a1 - a0)
    step_at = s0 + frac * (s1 - s0)
    tokens = step_at * tokens_per_step
    return Crossing(
        threshold, True, tokens, step_at, tokens * flops_per_token,
        note=f"interpolated between steps {int(s0)} and {int(s1)}",
    )


def threshold_ratio(a: Crossing, b: Crossing) -> dict:
    """Ratio of two crossings on every axis, refusing if either is unbracketed."""
    if not (a.bracketed and b.bracketed):
        return {
            "usable": False,
            "note": "at least one crossing is unbracketed; ratio would be a bound",
            "a_note": a.note,
            "b_note": b.note,
        }
    return {
        "usable": True,
        "threshold": a.threshold,
        "tokens_ratio": a.tokens / b.tokens,
        "steps_ratio": a.steps / b.steps,
        "flops_ratio": a.flops / b.flops,
    }


def inference_overhead(
    n_answer_tokens_dense: float,
    n_answer_tokens_split: float,
    prefill_mfu: float = 0.5,
    decode_mfu: float = 0.01,
) -> dict:
    """Honest inference cost, including the utilisation asymmetry.

    Training and prefill run at ~40-60% MFU; sequential generation runs at ~1%.
    The split arm's overhead is *generated* tokens -- the worst-utilisation kind --
    while its saving is in training compute. A pure FLOP-matched comparison
    therefore flatters it by up to ~50x on the overhead tokens, which is why this
    returns a cost ratio as well as a token ratio.
    """
    tok_ratio = n_answer_tokens_split / n_answer_tokens_dense
    cost_ratio = tok_ratio * (prefill_mfu / decode_mfu) ** 0.0  # tokens are decode
    # Both arms decode, so the MFU factor cancels in the *ratio* -- but it does
    # not cancel when comparing a training saving against an inference cost.
    return {
        "answer_token_ratio": tok_ratio,
        "decode_cost_ratio": cost_ratio,
        "train_to_inference_mfu_penalty": prefill_mfu / decode_mfu,
        "note": (
            "A training-FLOP saving and an inference-token cost are not "
            "interchangeable: the same FLOP bought at ~1% MFU during decode costs "
            f"~{prefill_mfu / decode_mfu:.0f}x more than one spent in training."
        ),
    }
