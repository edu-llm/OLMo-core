"""
What this tranche can and cannot detect, computed rather than asserted.

Every threshold in ``docs/hc-ablation/EXPERIMENT-DESIGN.md`` scales linearly with the seed
standard deviation, and the whole point of running stage 1 (``.edullm/run.hc-baseline.yaml``)
alone is to replace the planning value with a measured one. So the design document quotes this
script and this script takes ``--sigma``, rather than the numbers being typed into prose where
they would go stale the moment the baseline reports.

    python src/scripts/ablations/hc_power.py
    python src/scripts/ablations/hc_power.py --sigma 0.0212 --seeds 5
    python src/scripts/ablations/hc_power.py --json

Two things it deliberately does NOT do.

It does not compute exact noncentral-t power. The minimum detectable effect here is the
textbook normal-theory expression with central-t quantiles substituted,
``MDE = (t_{1-alpha/2,df} + t_{1-beta,df}) * SE``, which is very slightly conservative (it
overstates the MDE by roughly 1-3% at these degrees of freedom) and needs no SciPy. Being
conservative in that direction is the safe one for a design document: it never claims the
experiment can see something it cannot.

It does not know sigma. Nothing in this repository does. ``PLANNING_SIGMA`` below is an
extrapolation from a published measurement on a different model family at a different scale,
carried with its whole derivation so that a reader can reject it, and every number this prints
before stage 1 reports is an estimate of an estimate.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DATADECIDE_SIGMA_EXPONENT",
    "PLANNING_SIGMA",
    "Contrast",
    "minimum_detectable_effect",
    "scale_sigma_with_horizon",
    "standard_error",
    "t_quantile",
]

#: Ai2's DataDecide (arXiv 2504.11393) regressed seed sigma against token count over the
#: 3B-30B window with model-size fixed effects and found ``sigma ~ D^-0.172``, bootstrap CI
#: [0.088, 0.306]. This is the single exponent the whole design rests on: a run costs in
#: proportion to ``D``, so a fixed budget buys ``n = C/D`` runs and the standard error of an
#: arm mean goes as ``D^(0.5 - 0.172) = D^+0.328``. Positive across the entire interval, so
#: horizon bought with replicate money makes the experiment less sensitive.
DATADECIDE_SIGMA_EXPONENT = 0.172

#: The reference point the planning sigma is extrapolated from: 0.010 nats at 4.72e9 tokens on
#: a 370M dense model, which is the middle of the 0.008-0.012 range this team's earlier
#: pre-registration (`38b66591`) used and the value its gate was written against.
REFERENCE_SIGMA = 0.010
REFERENCE_TOKENS = 4.72e9

#: An extra factor on top of the horizon extrapolation, for two differences the extrapolation
#: does not cover: the model is smaller (about 190M active parameters against 370M dense, and
#: seed sigma rises as models shrink), and it is a mixture of experts, whose router is a
#: discrete decision made freshly at every step and is an additional source of run-to-run
#: variance that a dense model does not have. 1.35 is a guess. It is written as its own factor
#: rather than folded into a single number so that it can be argued with separately, and so
#: that stage 1 replaces the product rather than hiding a disagreement inside it.
SMALL_MOE_SIGMA_FACTOR = 1.35


def scale_sigma_with_horizon(
    sigma: float, *, from_tokens: float, to_tokens: float, exponent: float = DATADECIDE_SIGMA_EXPONENT
) -> float:
    """
    Move a seed sigma from one token budget to another along ``sigma ~ D^-exponent``.

    :param sigma: The known sigma.
    :param from_tokens: The token budget it was measured at.
    :param to_tokens: The token budget wanted.
    :param exponent: The DataDecide exponent.

    :returns: The extrapolated sigma.
    """
    return sigma * (to_tokens / from_tokens) ** (-exponent)


#: 786M tokens is 3,000 steps of 262,144, which is what one twelve-hour cell is sized for.
TRANCHE_TOKENS = 786_432_000

PLANNING_SIGMA = round(
    scale_sigma_with_horizon(
        REFERENCE_SIGMA, from_tokens=REFERENCE_TOKENS, to_tokens=TRANCHE_TOKENS
    )
    * SMALL_MOE_SIGMA_FACTOR,
    5,
)


# ---------------------------------------------------------------------------------------------
# Student's t, without SciPy
# ---------------------------------------------------------------------------------------------


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, *, iterations: int = 300, tiny: float = 1e-30) -> float:
    """
    The continued fraction for the incomplete beta function, by the modified Lentz method.

    :param a: First shape parameter.
    :param b: Second shape parameter.
    :param x: The argument, in ``[0, 1]``.
    :param iterations: The iteration cap.
    :param tiny: The floor that keeps a zero denominator from ending the recursion.

    :returns: The continued fraction's value.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-16:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """
    The regularized incomplete beta function ``I_x(a, b)``.

    :param a: First shape parameter.
    :param b: Second shape parameter.
    :param x: The argument, in ``[0, 1]``.

    :returns: ``I_x(a, b)``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        b * math.log(1.0 - x) + a * math.log(x) - _log_beta(b, a)
    ) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """
    The CDF of Student's t.

    :param t: The quantile.
    :param df: The degrees of freedom.

    :returns: ``P(T <= t)``.
    """
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """
    The inverse CDF of Student's t, by bisection on :func:`t_cdf`.

    Bisection rather than a closed-form approximation because this is called a few dozen times
    in the life of the process and correctness is worth more here than speed: an approximation
    that is 2% off at df = 4 moves every threshold in the design document by 2%.

    :param p: The probability, strictly between 0 and 1.
    :param df: The degrees of freedom, at least 1.

    :returns: The value ``t`` with ``P(T <= t) = p``.

    :raises ValueError: If ``p`` is not in ``(0, 1)`` or ``df`` is below 1.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    low, high = -1e3, 1e3
    for _ in range(200):
        middle = 0.5 * (low + high)
        if t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Contrast:
    """
    One pre-registered comparison, as the linear combination of arm means that defines it.

    :param name: The hypothesis label.
    :param weights: The coefficient on each arm, in ``arms`` order. A simple difference is
        ``(1, -1)``; the 2x2 interaction is ``(1, -1, -1, 1)``.
    :param reads: What a result on it would mean.
    """

    name: str
    weights: Tuple[float, ...]
    reads: str

    @property
    def variance_factor(self) -> float:
        """
        ``sum(w_i^2)``, which multiplies ``sigma^2 / n`` to give the contrast's variance.

        :returns: The factor. 2 for a simple difference, 4 for a 2x2 interaction.
        """
        return sum(weight * weight for weight in self.weights)


def standard_error(sigma: float, *, seeds: int, contrast: Contrast) -> float:
    """
    The standard error of a contrast estimated from balanced arms.

    :param sigma: The within-arm (seed-to-seed) standard deviation.
    :param seeds: The number of seeds per arm.
    :param contrast: The contrast.

    :returns: The standard error, in the units of ``sigma``.
    """
    return sigma * math.sqrt(contrast.variance_factor / seeds)


def minimum_detectable_effect(
    sigma: float,
    *,
    seeds: int,
    contrast: Contrast,
    df: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    The smallest effect this design detects at the stated power.

    :param sigma: The within-arm standard deviation.
    :param seeds: Seeds per arm.
    :param contrast: The contrast.
    :param df: Degrees of freedom in the pooled variance estimate.
    :param alpha: Two-sided significance level.
    :param power: The power wanted.

    :returns: The minimum detectable effect, in the units of ``sigma``.
    """
    return (t_quantile(1.0 - alpha / 2.0, df) + t_quantile(power, df)) * standard_error(
        sigma, seeds=seeds, contrast=contrast
    )


#: The four arms of the 2x2, in the order every ``Contrast.weights`` tuple below is written in.
ARMS: Tuple[str, ...] = (
    "mhc_moe",  # learned Sinkhorn mixer, stream balancing OFF -- the reference
    "mhc_moe_balanced",  # learned Sinkhorn mixer, stream balancing ON -- the treatment
    "mhc_moe_identity",  # H_res pinned to I, stream balancing OFF
    "mhc_moe_identity_balanced",  # H_res pinned to I, stream balancing ON
)

CONTRASTS: Tuple[Contrast, ...] = (
    Contrast(
        name="H1  balancing, at learned mixing",
        weights=(-1.0, 1.0, 0.0, 0.0),
        reads="the one isolated change. Does turning stream balancing on help a learned mixer?",
    ),
    Contrast(
        name="H2  learned mixing, unbalanced",
        weights=(1.0, 0.0, -1.0, 0.0),
        reads=(
            "the literature's finding, re-measured here. Alimaskina et al. and the mHC "
            "finetuning paper both report this at or below zero."
        ),
    ),
    Contrast(
        name="H3  learned mixing, balanced",
        weights=(0.0, 1.0, 0.0, -1.0),
        reads="whether mixing earns its keep once the streams are not collapsed.",
    ),
    Contrast(
        name="H4  the interaction (H3 - H2)",
        weights=(1.0, -1.0, -1.0, 1.0),
        reads=(
            "THE MECHANISM CLAIM. Balancing should help a LEARNED mixer more than it helps a "
            "pinned one; anything else and balancing is a generic regulariser."
        ),
    ),
    Contrast(
        name="H5  balancing, at pinned mixing",
        weights=(0.0, 0.0, -1.0, 1.0),
        reads=(
            "the generic-regulariser control. H_res = I has no mixing to rescue, so a gain "
            "here is balancing doing something else."
        ),
    ),
)


def table(sigma: float, *, seeds: int, arms: int = 4, alpha: float = 0.05, power: float = 0.80):
    """
    Every pre-registered contrast's standard error and minimum detectable effect.

    :param sigma: The within-arm standard deviation.
    :param seeds: Seeds per arm.
    :param arms: How many arms the pooled variance is estimated from.
    :param alpha: Two-sided significance level.
    :param power: The power wanted.

    :returns: A list of dicts, one per contrast.
    """
    df = arms * (seeds - 1)
    rows = []
    for contrast in CONTRASTS:
        rows.append(
            {
                "hypothesis": contrast.name,
                "se": standard_error(sigma, seeds=seeds, contrast=contrast),
                "mde": minimum_detectable_effect(
                    sigma, seeds=seeds, contrast=contrast, df=df, alpha=alpha, power=power
                ),
                "gate_2se": 2.0 * standard_error(sigma, seeds=seeds, contrast=contrast),
                "reads": contrast.reads,
            }
        )
    return rows, df


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the CLI.

    :param argv: Arguments, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=PLANNING_SIGMA,
        help="the within-arm seed standard deviation, in nats. THE DEFAULT IS AN "
        "EXTRAPOLATION AND NOT A MEASUREMENT; replace it with stage 1's number.",
    )
    parser.add_argument("--seeds", type=int, default=5, help="seeds per arm")
    parser.add_argument("--arms", type=int, default=4, help="arms the pooled sigma comes from")
    parser.add_argument("--alpha", type=float, default=0.05, help="two-sided significance level")
    parser.add_argument("--power", type=float, default=0.80, help="the power wanted")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also print what other seed counts would buy, which is the budget question",
    )
    parser.add_argument("--json", action="store_true", help="print one JSON document instead")
    args = parser.parse_args(argv)

    rows, df = table(
        args.sigma, seeds=args.seeds, arms=args.arms, alpha=args.alpha, power=args.power
    )
    sweep: Dict[int, Dict[str, float]] = {}
    if args.sweep:
        for seeds in (3, 4, 5, 6, 8, 10):
            swept, _ = table(
                args.sigma, seeds=seeds, arms=args.arms, alpha=args.alpha, power=args.power
            )
            sweep[seeds] = {row["hypothesis"].split()[0]: row["mde"] for row in swept}

    if args.json:
        print(
            json.dumps(
                {
                    "sigma": args.sigma,
                    "sigma_is_measured": args.sigma != PLANNING_SIGMA,
                    "planning_sigma": PLANNING_SIGMA,
                    "seeds_per_arm": args.seeds,
                    "arms": args.arms,
                    "pooled_df": df,
                    "alpha": args.alpha,
                    "power": args.power,
                    "contrasts": rows,
                    "sweep": sweep,
                },
                indent=2,
            )
        )
        return 0

    measured = "MEASURED" if args.sigma != PLANNING_SIGMA else "PLANNING ESTIMATE, NOT MEASURED"
    print(
        f"\nsigma = {args.sigma:.5f} nats  [{measured}]\n"
        f"{args.seeds} seeds x {args.arms} arms, pooled df = {df}, "
        f"two-sided alpha = {args.alpha}, power = {args.power}\n"
    )
    print(f"{'hypothesis':<36}{'SE':>9}{'2*SE gate':>12}{'MDE':>9}")
    print("-" * 66)
    for row in rows:
        print(
            f"{row['hypothesis']:<36}{row['se']:>9.4f}{row['gate_2se']:>12.4f}{row['mde']:>9.4f}"
        )
    print("\nWhat each one reads:")
    for row in rows:
        print(f"  {row['hypothesis'].split()[0]:<5}{row['reads']}")

    if sweep:
        print(f"\nMDE against seeds per arm, at sigma = {args.sigma:.5f}:")
        keys = sorted({key for values in sweep.values() for key in values})
        print(f"{'seeds':>6}" + "".join(f"{key:>9}" for key in keys))
        for seeds in sorted(sweep):
            print(
                f"{seeds:>6}" + "".join(f"{sweep[seeds].get(key, 0.0):>9.4f}" for key in keys)
            )
        print(
            "\nEvery column falls as 1/sqrt(seeds) and the money is linear in seeds, so the\n"
            "returns diminish; the design document argues where to stop."
        )

    print(
        "\nThe MDE is the smallest TRUE effect this design would detect at the stated power.\n"
        "An effect smaller than it is not ruled out by a null result -- it is unmeasured.\n"
        "See docs/hc-ablation/EXPERIMENT-DESIGN.md for what effect sizes the literature\n"
        "predicts and whether these numbers reach them. They do not, for the loss endpoint,\n"
        "and that is why the loss endpoint is not the primary one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
