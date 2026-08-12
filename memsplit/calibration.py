"""Endpoint calibration. Runs BEFORE the grid, and refuses endpoints.

## The failure this exists to stop

Seven endpoints in a row failed on instrumentation rather than on science: iGSM,
deduction, reasoning-gym, two-hop, KQA-Pro, a Wikidata chain eval, and finally a
mental-arithmetic endpoint whose 18 confirmatory cells all landed in 4.13-4.61%
against a 4.695% best-constant floor -- **zero cells above floor**, with eleven
scoring *exactly* 1342/30000. Its own verdict: "crowding is untested rather than
refuted", and "an endpoint with no dynamic range has none at any replication", so
more seeds would not have helped.

The bracketing gate had in fact been implemented, and it *did* correctly refuse
the endpoint. But it ran **after 32 cells had trained**, because a module-level
constant made the depth sweep inexpressible. So the rule is not "add a gate":

    The calibration must be cheaper than the grid, and it must run first.

Concretely that means: depth is a parameter (it is, see `nhop`), and the gate
needs no trained model. `calibrate_endpoint` scores an **untrained** model and the
**best-constant policy** through the *production* parser at every depth, and
returns a verdict. It costs one forward pass per item, not a training run.

## What "usable" means

Four conditions, all of which the previous endpoints failed at least one of:

1. **Dynamic range.** The ceiling reachable by a competent solver, minus the
   best-constant floor, must exceed `min_range_pp`. An endpoint whose floor is
   4.695% and whose observed spread is 0.5pp cannot register any effect.
2. **Untrained model at floor.** An untrained model must not beat the floor. If it
   does, the items leak their answers.
3. **Parseable.** The untrained model's output must be parseable at some
   non-trivial rate under the production parser, or accuracy is measuring format
   compliance rather than content.
4. **Depth-graded.** Accuracy must fall with depth for an oracle-with-noise
   solver. A flat curve means the endpoint does not measure serial reasoning --
   which is the paper's own first stated criterion for a reasoning benchmark, and
   was never once exercised.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from memsplit.scorers import best_constant_accuracy, score_items


@dataclass
class DepthCalibration:
    depth: int
    n_items: int
    chance: float
    best_constant: float
    best_constant_label: str
    untrained_accuracy: float
    untrained_unparseable: float
    oracle_accuracy: float
    oracle_noisy_accuracy: float
    oracle_noisy_expected: float
    oracle_noisy_se: float
    dynamic_range_pp: float
    degenerate: dict[str, float] = field(default_factory=dict)


@dataclass
class Verdict:
    usable: bool
    reasons: list[str] = field(default_factory=list)
    per_depth: list[DepthCalibration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "usable": self.usable,
            "reasons": self.reasons,
            "per_depth": [asdict(d) for d in self.per_depth],
        }


def oracle_generation(item, per_hop: float, rng: random.Random) -> str:
    """What a solver with per-hop reliability `per_hop` would emit.

    Used to establish the reachable ceiling and, at per_hop < 1, to check the
    endpoint is depth-graded. Each of the `n_lookups_expected` steps succeeds
    independently, so the emitted answer is correct with probability
    per_hop**n_lookups -- i.e. this simulates the p**n null directly, which is the
    curve the real depth results must be read against.
    """
    n = item.meta["n_lookups_expected"]
    ok = all(rng.random() < per_hop for _ in range(n))
    answer = item.answer if ok else "__wrong__"
    return f" ...\nAnswer: {answer}"


def untrained_generation(item, vocab_words: list[str], rng: random.Random) -> str:
    """A degenerate but format-compliant generation.

    Deliberately emits a well-formed `Answer:` line with a *random in-pool value*.
    That is the strongest degenerate policy available without training, and it is
    what catches answer leakage: if items can be solved from surface cues, this
    scores above floor.
    """
    return f" ...\nAnswer: {rng.choice(vocab_words)}"


def default_degenerate_policies(vocab_words: list[str]) -> dict:
    """The degenerate policies every endpoint must be checked against.

    Pluggable on purpose. A random-from-pool guesser cannot exploit an
    *item-specific* cue, so it cannot detect leakage on its own -- and leakage is
    exactly the failure where an item is answerable from surface features. Pass
    your own adversarial policies here; the strongest one you can think of is the
    floor your result has to clear. Two were found the hard way in the previous
    programme: a comparison task whose answer *was* one of the stated attributes
    (99.7% recoverable with no biographies at all), and expression spaces small
    enough that 100% of an eval set appeared verbatim in training.
    """
    return {
        "random_from_pool": lambda item, rng: (
            f" ...\nAnswer: {rng.choice(vocab_words)}"
        ),
        "constant": lambda item, rng: f" ...\nAnswer: {vocab_words[0]}",
        "empty": lambda item, rng: " ...",
    }


def calibrate_depth(
    items: list,
    vocab_words: list[str],
    chance: float,
    mode: str = "answer_tag_exact",
    per_hop_noisy: float = 0.93,
    seed: int = 0,
    degenerate_policies: dict | None = None,
) -> DepthCalibration:
    """Calibrate one depth stratum. No trained model required."""
    if not items:
        raise ValueError("no items to calibrate")
    depth = items[0].meta["depth"]
    golds = [it.answer for it in items]
    n = len(items)

    policies = (
        degenerate_policies
        if degenerate_policies is not None
        else default_degenerate_policies(vocab_words)
    )
    degen: dict[str, float] = {}
    unparse: dict[str, float] = {}
    for name, fn in policies.items():
        rng = random.Random(f"degen:{name}:{seed}:{depth}")
        gens = [fn(it, rng) for it in items]
        res, _ = score_items(gens, golds, mode=mode, chance=chance)
        degen[name] = res["accuracy"]
        unparse[name] = res["unparseable_rate"]

    rng = random.Random(f"oracle:{seed}:{depth}")
    perfect = [oracle_generation(it, 1.0, rng) for it in items]
    o, _ = score_items(perfect, golds, mode=mode, chance=chance)

    rng = random.Random(f"noisy:{seed}:{depth}")
    noisy = [oracle_generation(it, per_hop_noisy, rng) for it in items]
    nz, _ = score_items(noisy, golds, mode=mode, chance=chance)

    bc, bc_label = best_constant_accuracy(golds)
    expected = per_hop_noisy ** (depth + 1)
    se = math.sqrt(max(expected * (1 - expected), 1e-12) / n)

    strongest = max(degen.values()) if degen else 0.0
    strongest_name = max(degen, key=degen.get) if degen else ""
    return DepthCalibration(
        depth=depth,
        n_items=n,
        chance=chance,
        best_constant=bc,
        best_constant_label=bc_label,
        untrained_accuracy=strongest,
        untrained_unparseable=unparse.get(strongest_name, 0.0),
        oracle_accuracy=o["accuracy"],
        oracle_noisy_accuracy=nz["accuracy"],
        oracle_noisy_expected=expected,
        oracle_noisy_se=se,
        dynamic_range_pp=100.0 * (o["accuracy"] - max(bc, strongest)),
        degenerate=degen,
    )


def calibrate_endpoint(
    items_by_depth: dict[int, list],
    vocab_words: list[str],
    chance: float,
    min_range_pp: float = 10.0,
    max_untrained_over_floor_pp: float = 1.0,
    require_depth_grading: bool = True,
    per_hop_noisy: float = 0.93,
    seed: int = 0,
    degenerate_policies: dict | None = None,
) -> Verdict:
    """Score every depth and return a usable/unusable verdict with reasons.

    `min_range_pp = 10.0` follows the rule the previous programme arrived at only
    after the fact. Note its own correction: floor + 10pp is the *range*
    requirement, and a separate power requirement can bind harder -- at a ~4.65%
    floor the binding constraint worked out to >= 23.8% absolute accuracy, not
    14.7%. Set `min_range_pp` from a power calculation, not from habit.
    """
    per_depth = [
        calibrate_depth(
            items, vocab_words, chance, per_hop_noisy=per_hop_noisy, seed=seed,
            degenerate_policies=degenerate_policies,
        )
        for _, items in sorted(items_by_depth.items())
    ]
    reasons: list[str] = []

    for c in per_depth:
        if c.dynamic_range_pp < min_range_pp:
            reasons.append(
                f"depth {c.depth}: dynamic range {c.dynamic_range_pp:.1f}pp "
                f"< {min_range_pp}pp (ceiling {100*c.oracle_accuracy:.1f}%, "
                f"floor {100*max(c.best_constant, c.untrained_accuracy):.1f}%)"
            )
        over = 100.0 * (c.untrained_accuracy - c.best_constant)
        if over > max_untrained_over_floor_pp:
            worst = max(c.degenerate, key=c.degenerate.get) if c.degenerate else "?"
            reasons.append(
                f"depth {c.depth}: degenerate policy {worst!r} beats the "
                f"best-constant floor by {over:.1f}pp -- items leak their answers"
            )
        if c.untrained_unparseable > 0.5:
            reasons.append(
                f"depth {c.depth}: {100*c.untrained_unparseable:.0f}% unparseable "
                "under the production parser"
            )

    if require_depth_grading and len(per_depth) >= 2:
        # Compare against the ANALYTIC p**n curve with a sampling tolerance, not
        # against strict step-wise monotonicity of one noisy sample. At n=40 and
        # p=0.93 the per-depth standard error is ~5pp, so a raw monotonicity test
        # fails on noise roughly half the time and would reject good endpoints.
        for c in per_depth:
            tol = 3.0 * max(c.oracle_noisy_se, 0.005)
            if abs(c.oracle_noisy_accuracy - c.oracle_noisy_expected) > tol:
                reasons.append(
                    f"depth {c.depth}: noisy-oracle accuracy "
                    f"{c.oracle_noisy_accuracy:.3f} departs from the p**n "
                    f"prediction {c.oracle_noisy_expected:.3f} by more than "
                    f"3 SE ({tol:.3f}) -- scoring or item construction is off"
                )
        drop = per_depth[0].oracle_noisy_expected - per_depth[-1].oracle_noisy_expected
        if drop < 0.05:
            reasons.append(
                f"expected accuracy falls only {100*drop:.1f}pp from depth "
                f"{per_depth[0].depth} to {per_depth[-1].depth} at per-hop "
                f"{per_hop_noisy} -- too flat to resolve a depth effect"
            )

    return Verdict(usable=not reasons, reasons=reasons, per_depth=per_depth)


def required_n_for_mde(
    mde_pp: float, sd_pp: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Items needed to resolve `mde_pp` given per-item sd, two-sided.

    Provided so the MDE is *pre-registered* rather than reconstructed afterwards.
    Post-hoc power computed from an observed effect is not informative, and the
    programme's own review says so: at n=750 against the best available anchor
    (+3.1 to +3.6pp) power was 28-29%, so "a null at n=750 is uninterpretable,
    which is precisely the outcome the whole experiment is designed to produce."
    """
    z_a = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_b = 0.841621234 if abs(power - 0.80) < 1e-9 else _z(power)
    if mde_pp <= 0:
        raise ValueError("mde_pp must be > 0")
    return int(math.ceil(((z_a + z_b) * sd_pp / mde_pp) ** 2))


def _z(p: float) -> float:
    """Inverse normal CDF, Acklam's rational approximation."""
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
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
