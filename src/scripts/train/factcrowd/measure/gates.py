"""
PRD 8.6's admission gates: what an endpoint must survive before its number is a result.

This is not analysis. It is an **admission check** that runs before anything is read, and PRD 8.6 says
``grid.run()`` raises on any endpoint that has not passed all of it. The distinction matters because
every one of the four uninterpretable nulls PRD 1 lists was a *number that got reported anyway*:

- a deduction eval scoring **below its own 0.500 floor** -- G1, which asks where the score sits between
  the measured floor and the ceiling rather than between 0 and 100;
- reasoning-gym macro-averaged over 14 families with floors from 0 to 0.5 -- G1 again, for the same
  reason, applied per endpoint because a macro-average has no floor of its own;
- two-hop composition at 2.3x the product of its parts, i.e. an endpoint answering from fact access
  rather than from the composition -- G3, which removes the premise and requires the score to collapse;
- iGSM graded on one mod-23 integer with the derivation discarded -- G7, whose resolution requirement
  is the only one of these that a single-seed run cannot fake.

G4 and G6 are the two that no single cell's score can answer: they read the b=0 arm and the parameter
sweep, which is why PRD 8.4 runs the reasoning-only control **first**.

**The numbering is the PRD's, and G5 is deliberately absent from its table.** Nothing is implemented for
it and nothing is renumbered, so a gate name in a log line means the same thing as the row in PRD 8.6.
Closing the hole would cost every future reader a search for a requirement that does not exist.

**A gate whose evidence is missing fails.** Six of the seven need evidence from an arm that may not have
run yet -- a random-init checkpoint, a premise-ablated corpus, the b=0 ceiling, a parameter sweep, k
replicates, a dilution ladder. Every one of those returns ``passed=False`` with a detail naming what to
run. Defaulting to ``True`` on absent evidence would reproduce, in code, exactly the failure this whole
section exists to prevent.

Two conventions, both enforced rather than documented:

- **Accuracies are fractions, everything reported is percentage points.** A bare float is read the way
  :attr:`~factcrowd.measure.endpoints.EndpointResult.accuracy` reads, so ``0.478`` and not ``47.8``; a
  bare float outside ``[0, 1]`` raises. A unit error is not a gate verdict.
- **Nothing here recomputes a rate.** ``accuracy``, ``above_floor``, ``headroom`` and
  ``unparseable_rate`` are :class:`~factcrowd.measure.endpoints.EndpointResult`'s, so a gate and the
  score it admits cannot disagree about arithmetic.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .endpoints import EndpointResult

Score = Union[EndpointResult, float]
"""
An accuracy, either as a scored endpoint or as a bare **fraction** in ``[0, 1]``.

Gates that only need the number accept both, so a caller with an
:class:`~factcrowd.measure.endpoints.EndpointResult` never has to unwrap it and a caller replaying a
published table (PRD 8.3's 47.8/66.0 from-scratch figures, say) never has to fabricate counts.
"""

GATES: Tuple[str, ...] = ("G1", "G2", "G3", "G4", "G6", "G7", "G8")
"""
Every gate PRD 8.6 defines, in the order its table lists them. **There is no G5** -- see the module
docstring. :func:`require_all` treats a name in here with no result as a failure, so a gate that is
never run cannot be mistaken for a gate that passed.
"""

NO_EVIDENCE: str = "no evidence"
"""
Prefix on the detail of every gate that could not be evaluated, so a log or a test can tell "this
failed" from "this was never checked" without parsing prose.
"""

IN_BAND_LOW_PCT: float = 20.0
"""Lower edge of PRD 8.6 G1's band, as a percentage of the floor-to-ceiling range. **Closed**."""

IN_BAND_HIGH_PCT: float = 80.0
"""Upper edge of the same band. Closed as well -- see :func:`g1_dynamic_range`."""

MIN_DEPTH_SPREAD_PP: float = 15.0
"""PRD 8.6 G1's ">=15pp across a task-depth sweep": how far the difficulty dial must move the score."""

TARGET_EFFECT_PP: float = 2.0
"""
The effect the design is powered for, from PRD 14 and `arXiv:2505.18091`.

One constant rather than three literals, because G4's range requirement, G6's minimum parameter
response and G8's dose are all *the same 2pp* and drifting them apart would let an endpoint pass a
gate calibrated to an effect the study is not looking for.
"""

MIN_ACHIEVABLE_RANGE_PP: float = 5.0 * TARGET_EFFECT_PP
"""
G4's floor on the achievable range. The PRD names no number here; this one is stated so it can be
argued with. Below 10pp a 2pp decline is a fifth of everything the instrument can express, and PRD
8.5's pooled linear model would be fitting most of its own range.
"""

MIN_PARAM_RISE_PP: float = TARGET_EFFECT_PP
"""
G6's minimum response to parameter count. A floor, not a target: PRD 8.3's Mano anchor moves +18.2pp
across this ladder. An endpoint that moves less than the effect being hunted cannot resolve it.
"""

MIN_PREMISE_DROP_PP: float = 15.0
"""
G3's minimum collapse when the premise is removed. Set level with :data:`MIN_DEPTH_SPREAD_PP`: an
ablation that moves the score less than the difficulty dial does has not destroyed anything.
"""

FLOOR_TOLERANCE_SD: float = 3.0
"""
How far above its floor a control may land before G2 and G3 call it a signal, in binomial SDs.

Three rather than two because the floor is itself measured on a sample
(:meth:`factcrowd.corpus.tasks.ReasoningTask.degenerate_baseline` draws 20,000 items), so the
difference carries roughly twice the variance the scored-side SE alone accounts for.
"""

MIN_REPLICATES: int = 3
"""PRD 8.6 G7's ``k >= 3``. Two replicates give a sigma with one degree of freedom, which is a rumour."""

SIGMA_MAX_PP: float = 0.65
"""
G7's run-level sigma limit, in percentage points.

PRD 8.6 derives 0.63pp and the table rounds it to 0.65; the two are not the same number, which is why
:func:`g7_resolution` publishes the MDE next to sigma instead of letting the rounding disappear.
"""

UNPARSEABLE_MAX: float = 0.05
"""
G7's cap on unreadable answers, as a fraction. Structurally zero for both endpoints in this design --
see :func:`g2_label_permuted` for why that is a property of the rendering rather than of the scorer.
"""

_DILUTION_TOLERANCE_PP = 0.5
"""
How far a dilution ladder may run backwards before it is called unordered.

Not zero: these are finite samples, and two adjacent doses will cross by a fraction of a point from noise
alone. Half a point is a quarter of the 2pp target, so a ladder that inverts by more than this is not
measuring the dose response it claims to.
"""

DILUTION_DOSES_PCT: Tuple[int, ...] = (100, 95, 90, 80, 60)
"""
G8's ladder: percent of reasoning tokens retained. ``100`` is the reference arm.

Integer percent rather than a fraction because these are dictionary keys and ``0.95`` reached by two
different routes is not reliably the same double. The PRD writes them as percents too.
"""

TREND_POINTS: int = 5
"""Cells in the trend PRD 8.6 sizes its resolution requirement on: "a one-seed 5-point trend"."""

Z_ALPHA_ONE_SIDED_05: float = 1.6448536269514722
"""
The standard normal 95th percentile, for a **one-sided** alpha of 0.05.

Hard-coded because scipy is not a dependency. One-sided is what reproduces PRD 8.6's 0.63pp; pairing
the two-sided 1.96 with the same power term gives 0.56pp, and PRD 8.5 records an error of exactly that
shape ("the docs paired a *t* threshold with a *normal* statistic") as one of the two that cost this
measurement its claimed power.
"""

Z_POWER_80: float = 0.8416212335729143
"""The standard normal 80th percentile, for 80% power."""


@dataclass(frozen=True)
class GateResult:
    """
    One gate's verdict on one endpoint.

    :param gate: The gate's name as PRD 8.6 numbers it, e.g. ``"G7"``.
    :param passed: Whether the endpoint may be read on this axis.
    :param detail: Why, in a sentence a person reading a log can act on. On a failure it names the
        number, the limit, and what to do about it -- a bare "G7 failed" sends the reader back to the
        PRD, which is how a gate becomes something people route around.
    :param value: The quantity the verdict turned on, e.g. sigma for G7. ``None`` when no number was
        reached, which is the missing-evidence case.
    :param threshold: What :attr:`value` was compared against.
    :param evidence: Every other number the gate computed, as ordered ``(name, value)`` pairs. A tuple
        rather than a dict so the dataclass stays hashable -- a frozen dataclass holding a dict raises
        only later, at the first ``hash()``, which is a trap worth not setting.
    """

    gate: str
    passed: bool
    detail: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    evidence: Tuple[Tuple[str, float], ...] = ()

    def summary(self) -> Dict[str, object]:
        """
        A flat mapping for logging, shaped like :meth:`EndpointResult.summary`.

        :returns: The verdict, the deciding numbers, and every piece of evidence, prefixed so several
            gates' rows can be merged without colliding.
        """
        out: Dict[str, object] = {
            "gate": self.gate,
            "passed": self.passed,
            "detail": self.detail,
        }
        if self.value is not None:
            out["value"] = round(self.value, 6)
        if self.threshold is not None:
            out["threshold"] = round(self.threshold, 6)
        for name, number in self.evidence:
            out[f"{self.gate.lower()}_{name}"] = round(number, 6)
        return out


def minimum_detectable_effect(
    sigma_pp: float, *, n_points: int = TREND_POINTS, power_z: float = Z_POWER_80
) -> float:
    """
    The smallest end-to-end decline a one-seed trend of ``n_points`` cells could detect.

    PRD 8.6's own sizing, written out: the slope of an evenly spaced trend has
    ``SE = sigma / sqrt(Sxx)``, PRD 8.5 reads the effect as ``D = -4*beta`` across the range, so
    ``MDE = (z_alpha + z_power) * (n_points - 1) * sigma / sqrt(Sxx)``. At five points that is
    ``3.145 * sigma``, which puts the sigma needed to see 2pp at 0.636pp -- PRD 8.6's 0.63pp, and *not*
    the 0.65pp :data:`SIGMA_MAX_PP` rounds it to. Publishing the MDE is what keeps that gap visible.

    Evenly spaced is the PRD's convention, not the grid's geometry: the count axis sits at 0.3-4.8
    demanded bits per parameter, so a real fit weights its ends harder and does slightly better than
    this. Reported as the conservative reference figure, not as the analysis model.

    :param sigma_pp: Run-level standard deviation, in percentage points.
    :param n_points: Cells in the trend.
    :param power_z: Normal quantile for the desired power. Defaults to 80%.

    :returns: The detectable end-to-end decline, in percentage points.

    :raises OLMoConfigurationError: If ``n_points`` is under 3 or ``sigma_pp`` is negative.
    """
    if n_points < 3:
        raise OLMoConfigurationError(f"a trend needs at least 3 points, got {n_points}")
    if sigma_pp < 0:
        raise OLMoConfigurationError(f"sigma must be non-negative, got {sigma_pp}")
    offsets = np.arange(n_points, dtype=np.float64)
    sxx = float(((offsets - offsets.mean()) ** 2).sum())
    return (Z_ALPHA_ONE_SIDED_05 + power_z) * (n_points - 1) * sigma_pp / math.sqrt(sxx)


def g1_dynamic_range(
    result: EndpointResult, *, depth_scores: Optional[Mapping[int, Score]] = None
) -> GateResult:
    """
    G1 -- the endpoint sits inside 20-80% of its range, and a difficulty dial moves it >=15pp.

    The range is floor-to-100: the lower anchor is the endpoint's own measured floor, never an assumed
    chance rate. Against the *achievable* ceiling the same question is G4's, kept separate so that a
    b=0 arm which has not run yet fails one gate with one message instead of silently weakening this
    one.

    PRD 8.3 is this gate's worked example. Mano at L=13 scores 8.2 against a 6.80% best-constant
    policy -- 1.5% of range, so a decline has nowhere to go -- and the endpoint was retuned to L=10 at
    47.8, which is 45% of range. That retune happened *because* this gate was applied first.

    **The band is closed.** Exactly 20.0% and exactly 80.0% pass, and so does a depth spread of exactly
    15pp. The edges are round numbers chosen for legibility rather than measured cliffs, so making the
    verdict turn on the last bit of a float would be false precision -- and PRD 8.6 writes the spread
    requirement as ``>=`` anyway.

    :param result: The endpoint's score at the checkpoint being admitted.
    :param depth_scores: Accuracy at each task depth, keyed by depth -- for Mano, ``L``. At least two,
        because a spread needs two; the design supplies more.

    :returns: The verdict, with the position in range and the depth spread as evidence.

    :raises OLMoConfigurationError: If a depth score is a bare float outside ``[0, 1]``.
    """
    if result.headroom <= 0.0:
        return GateResult(
            gate="G1",
            passed=False,
            detail=(
                f"the measured floor is {100.0 * result.floor:.1f}%, so the best fact-free policy "
                f"answers every item and there is no range to sit inside. The task is broken, not the "
                f"model."
            ),
            value=result.headroom,
            threshold=0.0,
        )

    position = 100.0 * result.above_floor / result.headroom
    accuracy_pp = 100.0 * result.accuracy
    base: Tuple[Tuple[str, float], ...] = (
        ("accuracy_pp", accuracy_pp),
        ("floor_pp", 100.0 * result.floor),
        ("above_floor_pp", result.above_floor),
        ("position_pct", position),
    )

    if depth_scores is None or len(depth_scores) < 2:
        found = 0 if depth_scores is None else len(depth_scores)
        return _missing(
            "G1",
            f"the task-depth sweep has {found} depth(s) and needs at least 2. Score the endpoint at "
            f"a second task depth (Mano's dial is L) and pass the scores as depth_scores; PRD 8.6 "
            f"wants >={MIN_DEPTH_SPREAD_PP:.0f}pp across it.",
            evidence=base,
        )

    depths = sorted(depth_scores)
    by_depth = [
        _accuracy_pp(depth_scores[depth], what=f"depth_scores[{depth}]") for depth in depths
    ]
    spread = max(by_depth) - min(by_depth)
    # Signed, deepest minus shallowest, reported but never gated. A dial that moves the score the wrong
    # way is worth seeing, but PRD 8.5 deleted the monotonicity re-run rule for inflating type-I error
    # from 5.0% to 16.7%, and a gate that demands a direction is that rule wearing a different hat.
    signed = by_depth[-1] - by_depth[0]
    evidence = base + (
        ("depth_spread_pp", spread),
        ("deepest_minus_shallowest_pp", signed),
        ("n_depths", float(len(depths))),
    )

    if position < IN_BAND_LOW_PCT:
        return GateResult(
            gate="G1",
            passed=False,
            detail=(
                f"accuracy {accuracy_pp:.1f}pp is {position:.1f}% of the floor-to-100 range, under the "
                f"{IN_BAND_LOW_PCT:.0f}% band: it sits {result.above_floor:.1f}pp above its own "
                f"{100.0 * result.floor:.1f}pp floor, so a decline has nowhere to fall. Retune the "
                f"task -- PRD 8.3 moved Mano from L=13 to L=10 on exactly this reading."
            ),
            value=position,
            threshold=IN_BAND_LOW_PCT,
            evidence=evidence,
        )
    if position > IN_BAND_HIGH_PCT:
        return GateResult(
            gate="G1",
            passed=False,
            detail=(
                f"accuracy {accuracy_pp:.1f}pp is {position:.1f}% of the floor-to-100 range, over the "
                f"{IN_BAND_HIGH_PCT:.0f}% band: the endpoint is saturated and a decline would be read "
                f"as a ceiling. Make the task harder."
            ),
            value=position,
            threshold=IN_BAND_HIGH_PCT,
            evidence=evidence,
        )
    if spread < MIN_DEPTH_SPREAD_PP:
        return GateResult(
            gate="G1",
            passed=False,
            detail=(
                f"the difficulty dial moves the endpoint {spread:.1f}pp across depths {depths}, under "
                f"the {MIN_DEPTH_SPREAD_PP:.0f}pp PRD 8.6 requires. An instrument that does not "
                f"respond to task difficulty cannot be read as responding to fact load either, so a "
                f"flat result across the grid would say nothing."
            ),
            value=spread,
            threshold=MIN_DEPTH_SPREAD_PP,
            evidence=evidence,
        )
    return GateResult(
        gate="G1",
        passed=True,
        detail=(
            f"accuracy {accuracy_pp:.1f}pp sits at {position:.1f}% of the floor-to-100 range "
            f"(floor {100.0 * result.floor:.1f}pp), and the depth dial moves it {spread:.1f}pp across "
            f"depths {depths}."
        ),
        value=position,
        threshold=IN_BAND_LOW_PCT,
        evidence=evidence,
    )


def g2_label_permuted(
    random_init_result: Optional[EndpointResult] = None,
    *,
    tolerance_sd: float = FLOOR_TOLERANCE_SD,
) -> GateResult:
    """
    G2 -- a random-init model must score at the floor, not above it.

    PRD 8.6's reading: a random-init score above the floor "measures parser strictness, not the task."
    An untrained network knows nothing, so anything it earns above the best fact-free policy came from
    the instrument -- a parser crediting a near-miss, a normaliser collapsing two answers, or an answer
    reachable from the prompt by a policy the floor search never tried.

    **This gate passes trivially here, and that is a property of the design rather than luck.** Three
    decisions, all recorded before any of this ran, remove the failure it looks for:

    - the answer is a **single token at a known position**, so
      :func:`~factcrowd.measure.spans.predicted_token` is an argmax over the vocabulary. There is no
      continuation to truncate and no string to parse, so ``n_unparseable`` is structurally zero and
      strictness has no dial to be set wrong on (PRD 8.3, :mod:`factcrowd.measure.reasoning`);
    - scoring is exact tuple equality against the rendered answer, so there is no near-miss credit;
    - the floor is the best of a **constant and a copy** policy family, not the best constant alone.
      That is the decision this gate would otherwise catch late: `<compare>` originally answered with a
      name, "always name the first person" scored 50.2% against a best-constant 0.02%, and quoting the
      floor as 0% would have given a binary task half the range its admission gate assumed (PRD 8.3).

    So the gate is kept and run rather than asserted away -- a future multi-token endpoint would need
    a parser, and PRD 8.3 already asks for Brevo1, which is verifier-scored.

    :param random_init_result: The endpoint scored on an untrained checkpoint. Any permutation of the
        label-answer mapping that destroys the task works equally; a random init is the cheapest.
    :param tolerance_sd: How many binomial SDs above the floor still counts as at the floor.

    :returns: The verdict, with the excess over the floor and the tolerance as evidence.
    """
    if random_init_result is None:
        return _missing(
            "G2",
            "the endpoint has not been scored on a random-init checkpoint. Run measure.reasoning "
            "against an untrained model of the same shape; it costs one forward pass over the eval "
            "set and no training.",
        )
    allowed = _floor_tolerance_pp(random_init_result, tolerance_sd)
    evidence: Tuple[Tuple[str, float], ...] = (
        ("above_floor_pp", random_init_result.above_floor),
        ("tolerance_pp", allowed),
        ("unparseable_rate", random_init_result.unparseable_rate),
        ("n_total", float(random_init_result.n_total)),
    )
    if random_init_result.above_floor > allowed:
        return GateResult(
            gate="G2",
            passed=False,
            detail=(
                f"a random-init model scores {random_init_result.above_floor:.2f}pp above the measured "
                f"floor, past the {allowed:.2f}pp that {tolerance_sd:.0f} binomial SDs allow. An "
                f"untrained network cannot do the task, so that margin is the instrument: either the "
                f"scorer credits something short of the exact answer, or the floor search missed a "
                f"fact-free policy the model found."
            ),
            value=random_init_result.above_floor,
            threshold=allowed,
            evidence=evidence,
        )
    return GateResult(
        gate="G2",
        passed=True,
        detail=(
            f"a random-init model lands {random_init_result.above_floor:+.2f}pp from the measured "
            f"floor, inside the {allowed:.2f}pp noise band, with "
            f"{100.0 * random_init_result.unparseable_rate:.2f}% unparseable."
        ),
        value=random_init_result.above_floor,
        threshold=allowed,
        evidence=evidence,
    )


def g3_premise_ablated(
    full_result: EndpointResult,
    ablated_result: Optional[EndpointResult] = None,
    *,
    min_drop_pp: float = MIN_PREMISE_DROP_PP,
    tolerance_sd: float = FLOOR_TOLERANCE_SD,
) -> GateResult:
    """
    G3 -- deleting the premise must destroy the score.

    The hypothesis-only probe FLD's own authors warn about (PRD 8.6). Strip whatever the item claims to
    require -- Mano's operands, `<compare>`'s two named entities -- and re-score. An endpoint that
    survives that was answering from something else, which is precisely the two-hop failure PRD 1
    lists: composition at 2.3x the product of its parts, i.e. an endpoint measuring fact access under a
    composition label.

    Two conditions, because either alone can be gamed. The ablated score must sit **at its own floor**,
    and the drop must be **large**: a probe that merely dents the score has not shown the premise was
    load-bearing.

    The comparison is floor-corrected on both sides. Removing the premise can change the task's own
    degenerate baseline -- a shorter prompt has fewer spans to copy -- and differencing raw accuracies
    would then book a floor shift as a collapse.

    :param full_result: The endpoint on the intact task.
    :param ablated_result: The endpoint on the premise-ablated task.
    :param min_drop_pp: How far the floor-corrected score must fall.
    :param tolerance_sd: How many binomial SDs above its floor the ablated score may still land.

    :returns: The verdict, with the drop and the ablated residual as evidence.
    """
    if ablated_result is None:
        return _missing(
            "G3",
            "the premise-ablated probe has not been scored. Build the task with its premise removed "
            "(Mano: the operands; <compare>: the two entity names), measure that variant's own floor, "
            "and score it -- the model does not need retraining.",
        )
    allowed = _floor_tolerance_pp(ablated_result, tolerance_sd)
    drop = full_result.above_floor - ablated_result.above_floor
    evidence: Tuple[Tuple[str, float], ...] = (
        ("full_above_floor_pp", full_result.above_floor),
        ("ablated_above_floor_pp", ablated_result.above_floor),
        ("drop_pp", drop),
        ("tolerance_pp", allowed),
    )
    if ablated_result.above_floor > allowed:
        return GateResult(
            gate="G3",
            passed=False,
            detail=(
                f"with the premise removed the endpoint still scores "
                f"{ablated_result.above_floor:.2f}pp above its own floor, past the {allowed:.2f}pp "
                f"noise band. Whatever it is reading, it is not the premise -- this is the two-hop "
                f"failure of PRD 1, an endpoint answering from fact access under a reasoning label."
            ),
            value=ablated_result.above_floor,
            threshold=allowed,
            evidence=evidence,
        )
    if drop < min_drop_pp:
        return GateResult(
            gate="G3",
            passed=False,
            detail=(
                f"removing the premise costs only {drop:.1f}pp of floor-corrected accuracy, under the "
                f"{min_drop_pp:.0f}pp required. The ablated arm is at its floor, so the intact arm is "
                f"barely above one too: there is no signal here for a fact load to crowd out."
            ),
            value=drop,
            threshold=min_drop_pp,
            evidence=evidence,
        )
    return GateResult(
        gate="G3",
        passed=True,
        detail=(
            f"removing the premise costs {drop:.1f}pp and lands the endpoint "
            f"{ablated_result.above_floor:+.2f}pp from its own floor, inside the {allowed:.2f}pp noise "
            f"band."
        ),
        value=drop,
        threshold=min_drop_pp,
        evidence=evidence,
    )


def g4_headroom(
    result: EndpointResult,
    achievable_ceiling: Optional[Score] = None,
    *,
    min_range_pp: float = MIN_ACHIEVABLE_RANGE_PP,
    min_room_pct: float = IN_BAND_LOW_PCT,
) -> GateResult:
    """
    G4 -- headroom against the ceiling the b=0 arm reaches, not against a nominal 100%.

    :attr:`EndpointResult.headroom` measures floor-to-100. That is the *oracle* range, the one G1 uses,
    and it is not the range that exists. If the reasoning-only arm -- no facts to store, every
    parameter the ladder has -- tops out at 52%, then 52% is the ceiling and the honest headroom is
    52 minus the floor, not 95. This gate is what forces that number to be measured: it is the only one
    that reads the b=0 arm, and PRD 8.4 puts that arm first in M0 for exactly this reason.

    Two things are checked, and the informative one is about the **ceiling**, not the cell:

    - the achievable range, floor to b=0 ceiling, must be wide enough to hold the effect. A control arm
      that barely beats the floor means the endpoint is unlearnable at this scale on this token budget,
      and every cell under it is measuring noise;
    - the cell must not out-score the arm that carries no facts at all. Nothing else here can see that,
      and it means the two arms differ in something other than fact load.

    **No upper bound applies, deliberately.** Under PRD P3 -- the pre-registered expectation of no
    crowding -- every cell scores *at* the b=0 arm, which is 100% of the achievable range. A saturation
    check against this ceiling would refuse the predicted result for being predicted.

    :param result: The endpoint's score at the cell being admitted.
    :param achievable_ceiling: The b=0 arm's score on the same endpoint, as an
        :class:`EndpointResult` or a fraction.
    :param min_range_pp: How wide floor-to-ceiling must be.
    :param min_room_pct: How much of that range must lie below the cell's score.

    :returns: The verdict, with the achievable range and the room to fall as evidence.

    :raises OLMoConfigurationError: If the ceiling is a bare float outside ``[0, 1]``.
    """
    if achievable_ceiling is None:
        return _missing(
            "G4",
            "the b=0 arm has not been scored on this endpoint, so the achievable ceiling is unknown "
            "and headroom can only be quoted against a nominal 100%. PRD 8.4 runs that arm first, at "
            "4 widths x 3 seeds and ~0.15 h.",
        )
    ceiling_pp = _accuracy_pp(achievable_ceiling, what="achievable_ceiling")
    floor_pp = 100.0 * result.floor
    achievable_range = ceiling_pp - floor_pp
    room = result.above_floor
    evidence: Tuple[Tuple[str, float], ...] = (
        ("ceiling_pp", ceiling_pp),
        ("floor_pp", floor_pp),
        ("achievable_range_pp", achievable_range),
        ("room_to_fall_pp", room),
        ("oracle_headroom_pp", result.headroom),
    )
    if 100.0 * result.accuracy > ceiling_pp:
        return GateResult(
            gate="G4",
            passed=False,
            detail=(
                f"the cell scores {100.0 * result.accuracy:.1f}pp, above the {ceiling_pp:.1f}pp the "
                f"b=0 arm reaches. A cell carrying facts cannot beat the arm that carries none, so "
                f"the ceiling is mis-measured -- most likely the arms differ in reasoning-token "
                f"exposure, which PRD 3.4 holds constant in absolute tokens for this reason."
            ),
            value=100.0 * result.accuracy,
            threshold=ceiling_pp,
            evidence=evidence,
        )
    if achievable_range < min_range_pp:
        return GateResult(
            gate="G4",
            passed=False,
            detail=(
                f"the achievable range is {achievable_range:.1f}pp -- floor {floor_pp:.1f}pp to a b=0 "
                f"ceiling of {ceiling_pp:.1f}pp -- under the {min_range_pp:.0f}pp required. The "
                f"{TARGET_EFFECT_PP:.0f}pp effect this study is powered for would be "
                f"{100.0 * TARGET_EFFECT_PP / max(achievable_range, 1e-9):.0f}% of everything the "
                f"endpoint can express. The oracle headroom of {result.headroom:.1f}pp is not real."
            ),
            value=achievable_range,
            threshold=min_range_pp,
            evidence=evidence,
        )
    # Below the ceiling, this is implied by G1's lower edge -- room_pct >= G1's position whenever the
    # cell is under the ceiling, so it can only fire where G1 already has. Kept anyway, because G1
    # returns its missing-evidence verdict *before* it checks any band: without this line, deleting the
    # depth sweep would delete the band check from the whole gate table.
    room_pct = 100.0 * room / achievable_range
    if room_pct < min_room_pct:
        return GateResult(
            gate="G4",
            passed=False,
            detail=(
                f"the cell sits {room:.1f}pp above its floor, {room_pct:.1f}% of the "
                f"{achievable_range:.1f}pp achievable range and under the {min_room_pct:.0f}% "
                f"required. A decline would hit the floor before it became measurable, so a null here "
                f"would be a property of the range rather than of the model."
            ),
            value=room_pct,
            threshold=min_room_pct,
            evidence=evidence + (("room_pct", room_pct),),
        )
    return GateResult(
        gate="G4",
        passed=True,
        detail=(
            f"the achievable range is {achievable_range:.1f}pp (floor {floor_pp:.1f}pp to a b=0 "
            f"ceiling of {ceiling_pp:.1f}pp) and the cell sits {room:.1f}pp above its floor, "
            f"{room_pct:.1f}% of it."
        ),
        value=room_pct,
        threshold=min_room_pct,
        evidence=evidence + (("room_pct", room_pct),),
    )


def g6_capacity_responsive(
    scores_by_params: Optional[Mapping[int, Score]] = None,
    *,
    min_rise_pp: float = MIN_PARAM_RISE_PP,
) -> GateResult:
    """
    G6 -- the endpoint must move with parameter count at fixed depth.

    PRD 8.6: "an endpoint flat in parameters cannot detect a capacity effect by construction." The
    experiment's claim is that stored facts consume capacity that reasoning would otherwise use, so an
    endpoint deaf to capacity would return a null whatever the facts did -- and PRD 8.4 already had to
    withdraw the opposite belief, that reasoning is flat in width, when the Mano anchor turned out to
    move +18.2pp across this exact ladder at fixed depth 12.

    **Monotonicity is not required, on purpose.** The gate reads the rise from the smallest parameter
    count to the largest and ignores the shape between them. Allen-Zhu's own single-seed grid violates
    parameter-order monotonicity by a median of 27.1pp with 8 of 12 rows over 10pp (PRD 8.6), and PRD
    8.5 deleted this programme's monotonicity re-run rule outright: it fired on 98.3% of rows by
    chance, pushed type-I error from 5.0% to 16.7%, and shrank the variance estimate to 0.57 sigma,
    which makes the pre-registered equivalence test *falsely* declare equivalence.

    :param scores_by_params: Accuracy keyed by non-embedding parameter count, at fixed depth. At least
        two points; PRD 8.4's control arm supplies four widths.
    :param min_rise_pp: How far the score must rise across the ladder.

    :returns: The verdict, with the rise and the rate per parameter doubling as evidence.

    :raises OLMoConfigurationError: If a parameter count is not positive, or a score is a bare float
        outside ``[0, 1]``. Both are malformed input rather than missing evidence, so they raise
        instead of becoming a verdict.
    """
    if scores_by_params is None or len(scores_by_params) < 2:
        found = 0 if scores_by_params is None else len(scores_by_params)
        return _missing(
            "G6",
            f"the parameter sweep has {found} point(s) and needs at least 2. PRD 8.4's "
            f"reasoning-only control arm runs 4 widths x 3 seeds in ~0.15 h and returns exactly this, "
            f"along with the sigma G7 needs.",
        )
    sizes = sorted(scores_by_params)
    if sizes[0] <= 0:
        raise OLMoConfigurationError(
            f"scores_by_params is keyed by parameter count; got {sizes[0]}"
        )
    scores = [
        _accuracy_pp(scores_by_params[size], what=f"scores_by_params[{size}]") for size in sizes
    ]
    rise = scores[-1] - scores[0]
    doublings = math.log2(sizes[-1] / sizes[0])
    evidence: Tuple[Tuple[str, float], ...] = (
        ("smallest_params", float(sizes[0])),
        ("largest_params", float(sizes[-1])),
        ("rise_pp", rise),
        ("pp_per_doubling", rise / doublings),
        ("n_points", float(len(sizes))),
    )
    if rise < min_rise_pp:
        return GateResult(
            gate="G6",
            passed=False,
            detail=(
                f"the endpoint moves {rise:+.1f}pp from {sizes[0]:,} to {sizes[-1]:,} non-embedding "
                f"parameters, under the {min_rise_pp:.1f}pp required. It responds to capacity by less "
                f"than the effect a fact load is supposed to cost, so it cannot detect a capacity "
                f"effect by construction and a flat grid would be uninformative."
            ),
            value=rise,
            threshold=min_rise_pp,
            evidence=evidence,
        )
    return GateResult(
        gate="G6",
        passed=True,
        detail=(
            f"the endpoint rises {rise:+.1f}pp from {sizes[0]:,} to {sizes[-1]:,} non-embedding "
            f"parameters across {len(sizes)} points at fixed depth, {rise / doublings:.1f}pp per "
            f"doubling."
        ),
        value=rise,
        threshold=min_rise_pp,
        evidence=evidence,
    )


def g7_resolution(
    replicates: Optional[Sequence[Score]] = None,
    *,
    sigma_max_pp: float = SIGMA_MAX_PP,
    unparseable_max: float = UNPARSEABLE_MAX,
    min_replicates: int = MIN_REPLICATES,
) -> GateResult:
    """
    G7 -- k>=3 replicates, sigma and MDE published, sigma <= 0.65pp and unparseable <= 5%.

    The gate revision 1 did not have, and the reason PRD 8.6 is titled "resolution, not just dynamic
    range". Every other gate asks whether the endpoint *responds*; this one asks whether it can
    **resolve** the 2pp effect. PRD 8.6's arithmetic: a one-seed 5-point trend needs run-level
    sigma <= 0.63pp, and the design was 8-50x short of that while the old gate saw nothing wrong. All
    four prior nulls are consistent with an unmeasured sigma.

    Replicates are **runs**, not eval samples. PRD 8.5's central correction is that the 0.5pp figure
    the design was built on was eval sampling noise -- both published anchors lie exactly on
    21.27/sqrt(n) -- and the seed term was never added at all. A replicate here differs in
    ``init_seed`` (PRD 7.2), not in which items were scored.

    Sigma and the MDE are published whether the gate passes or fails, because the number that decides
    the seed count is useful either way (PRD 8.5 keys 1/2/3 seeds to a measured sigma).

    Both limits are **closed**: exactly 0.65pp and exactly 5% pass. PRD 8.6 writes them as ``<=``.

    :param replicates: One score per replicate. Pass :class:`EndpointResult` -- a bare fraction
        carries no unparseable count, so the second half of the gate would have no evidence.
    :param sigma_max_pp: The run-level sigma limit, in percentage points.
    :param unparseable_max: The cap on unreadable answers, as a fraction, applied to the **worst**
        replicate. Pooling would let three clean runs hide one whose scorer broke.
    :param min_replicates: PRD 8.6's ``k``.

    :returns: The verdict, always carrying sigma and the MDE where they could be computed.

    :raises OLMoConfigurationError: If a replicate is a bare float outside ``[0, 1]``.
    """
    if replicates is None or len(replicates) < min_replicates:
        found = 0 if replicates is None else len(replicates)
        return _missing(
            "G7",
            f"k={found} replicates, under the {min_replicates} PRD 8.6 requires. Re-run the cell with "
            f"a different init_seed -- replicates are runs, not eval resamples, and PRD 8.5 shows the "
            f"seed term was never in the variance budget at all.",
        )

    scores = [
        _accuracy_pp(one, what=f"replicates[{index}]") for index, one in enumerate(replicates)
    ]
    sigma = float(np.std(np.asarray(scores, dtype=np.float64), ddof=1))
    mde = minimum_detectable_effect(sigma)
    evidence: Tuple[Tuple[str, float], ...] = (
        ("k", float(len(scores))),
        ("sigma_pp", sigma),
        ("mde_pp", mde),
        ("mean_pp", float(np.mean(scores))),
    )

    scored = [one for one in replicates if isinstance(one, EndpointResult)]
    if len(scored) < len(replicates):
        return GateResult(
            gate="G7",
            passed=False,
            detail=(
                f"{NO_EVIDENCE}: sigma is {sigma:.3f}pp and the MDE {mde:.2f}pp, but "
                f"{len(replicates) - len(scored)} of {len(replicates)} replicates arrived as bare "
                f"accuracies, so the unparseable rate could not be checked. Pass EndpointResult -- "
                f"the count is already on it."
            ),
            value=sigma,
            threshold=sigma_max_pp,
            evidence=evidence,
        )

    worst = max(scored, key=lambda one: one.unparseable_rate)
    evidence = evidence + (("worst_unparseable_rate", worst.unparseable_rate),)

    if sigma > sigma_max_pp:
        return GateResult(
            gate="G7",
            passed=False,
            detail=(
                f"run-level sigma is {sigma:.3f}pp over k={len(scores)} replicates, past the "
                f"{sigma_max_pp:.2f}pp limit, so the MDE is {mde:.2f}pp against a "
                f"{TARGET_EFFECT_PP:.0f}pp target. The endpoint cannot resolve the effect: re-scope "
                f"the seed count from this sigma (PRD 8.5) before reading anything from the grid."
            ),
            value=sigma,
            threshold=sigma_max_pp,
            evidence=evidence,
        )
    if worst.unparseable_rate > unparseable_max:
        return GateResult(
            gate="G7",
            passed=False,
            detail=(
                f"the worst replicate leaves {100.0 * worst.unparseable_rate:.1f}% of items "
                f"unreadable, over the {100.0 * unparseable_max:.0f}% cap "
                f"({worst.n_unparseable:,} of {worst.n_total:,}). Both endpoints here render a "
                f"single-token answer at a known position, so a non-zero rate is a scorer fault "
                f"rather than a model one."
            ),
            value=worst.unparseable_rate,
            threshold=unparseable_max,
            evidence=evidence,
        )
    return GateResult(
        gate="G7",
        passed=True,
        detail=(
            f"sigma {sigma:.3f}pp over k={len(scores)} replicates, MDE {mde:.2f}pp against a "
            f"{TARGET_EFFECT_PP:.0f}pp target, worst unparseable rate "
            f"{100.0 * worst.unparseable_rate:.2f}%."
        ),
        value=sigma,
        threshold=sigma_max_pp,
        evidence=evidence,
    )


def g8_calibrated_positive_control(
    dilution_scores: Optional[Mapping[int, Score]] = None, *, target_pp: float = TARGET_EFFECT_PP
) -> GateResult:
    """
    G8 -- a reasoning-token dilution ladder that brackets a dose worth ~2pp.

    The gate that makes a null mean something. Train the same cell on 100/95/90/80/60% of the reasoning
    tokens and score the endpoint on each. This is a treatment known to hurt, so if the ladder cannot
    produce a 2pp decline the endpoint cannot show one, and PRD 3.3 has the sharper version of the
    worry: Physics 3.3's Result 11 says non-knowledge-dense competitors do not interfere at all, which
    would make a null predicted regardless of crowding.

    **Calibrated means bracketed, not merely detected.** The ladder must reach the target at its
    strongest dose *and* still be under it at its gentlest. A ladder whose 95% arm already costs 5pp is
    too coarse to name the dose worth 2pp, and the dose is the calibration -- it converts "reasoning
    fell 2pp" into "worth about this much reasoning exposure", which is the only reading of the effect
    size that does not require trusting the endpoint's scale.

    :param dilution_scores: Accuracy keyed by **percent of reasoning tokens retained**; all of
        :data:`DILUTION_DOSES_PCT` are required, with 100 as the reference.
    :param target_pp: The decline the ladder must bracket.

    :returns: The verdict, with the bracketing drops and the interpolated dose as evidence.

    :raises OLMoConfigurationError: If a dose's score is a bare float outside ``[0, 1]``.
    """
    if dilution_scores is None:
        return _missing(
            "G8",
            f"the dilution ladder has not been run. Train the cell at "
            f"{'/'.join(str(dose) for dose in DILUTION_DOSES_PCT)}% of its reasoning tokens and score "
            f"the endpoint on each.",
        )
    absent = [dose for dose in DILUTION_DOSES_PCT if dose not in dilution_scores]
    if absent:
        return _missing(
            "G8",
            f"the dilution ladder is missing the "
            f"{', '.join(f'{dose}%' for dose in absent)} dose(s). PRD 8.6 fixes the ladder at "
            f"{'/'.join(str(dose) for dose in DILUTION_DOSES_PCT)}%, so a partial ladder cannot be "
            f"interpolated -- run the rest.",
        )

    by_dose = {
        dose: _accuracy_pp(dilution_scores[dose], what=f"dilution_scores[{dose}]")
        for dose in DILUTION_DOSES_PCT
    }
    reference = by_dose[100]
    drops = {dose: reference - by_dose[dose] for dose in DILUTION_DOSES_PCT if dose != 100}
    # The axis runs backwards: the dose keys are tokens *retained*, so the strongest dilution is the
    # smallest key. Naming them rather than writing 60 and 95 keeps the ladder in one constant.
    strongest = min(drops)
    gentlest = max(drops)
    evidence: Tuple[Tuple[str, float], ...] = (
        ("reference_pp", reference),
        ("gentlest_dose_pct", float(gentlest)),
        ("gentlest_drop_pp", drops[gentlest]),
        ("strongest_dose_pct", float(strongest)),
        ("strongest_drop_pp", drops[strongest]),
        ("max_drop_pp", max(drops.values())),
    )

    # THE STRONGEST DOSE, NOT THE LARGEST DROP ANYWHERE. Testing `max(drops.values())` passed a ladder
    # that cost 3pp at 80% and *nothing* at 60%, and then reported "0.00pp at 95% rising to 0.00pp at
    # 60%" -- a self-contradicting sentence attached to a pass. If removing more reasoning data does not
    # cost more, the ladder is measuring noise and cannot calibrate anything.
    if drops[strongest] < target_pp:
        return GateResult(
            gate="G8",
            passed=False,
            detail=(
                f"cutting reasoning tokens to {strongest}% costs only {drops[strongest]:.2f}pp, under "
                f"the {target_pp:.0f}pp target (the largest drop anywhere on the ladder is "
                f"{max(drops.values()):.2f}pp, at {max(drops, key=lambda d: drops[d])}%). A treatment "
                f"known to hurt does not move the endpoint at its strongest, so a null on the fact axis "
                f"would say nothing about facts."
            ),
            value=drops[strongest],
            threshold=target_pp,
            evidence=evidence,
        )

    # And the response has to be ordered. A ladder that dips and recovers is noise with a trend drawn
    # through it, and interpolating a "dose worth 2pp" from it names a number that means nothing.
    ordered = [drops[dose] for dose in sorted(drops, reverse=True)]
    inversions = [
        (before, after)
        for before, after in zip(ordered, ordered[1:])
        if after < before - _DILUTION_TOLERANCE_PP
    ]
    if inversions:
        return GateResult(
            gate="G8",
            passed=False,
            detail=(
                f"the dilution response is not ordered: removing more reasoning data costs less at "
                f"{len(inversions)} step(s) on the ladder "
                f"({', '.join(f'{a:.2f}pp then {b:.2f}pp' for a, b in inversions)}). Interpolating a "
                f"dose worth {target_pp:.0f}pp from a non-monotone ladder names a number that means "
                f"nothing. Re-run in paired replicates, replacing removed reasoning tokens with matched "
                f"filler so total steps and the schedule stay fixed."
            ),
            value=float(len(inversions)),
            threshold=0.0,
            evidence=evidence,
        )
    if drops[gentlest] > target_pp:
        return GateResult(
            gate="G8",
            passed=False,
            detail=(
                f"the gentlest dose already costs {drops[gentlest]:.2f}pp at {gentlest}% of reasoning "
                f"tokens, over the {target_pp:.0f}pp target, so the ladder brackets nothing and the "
                f"dose worth {target_pp:.0f}pp cannot be named. Re-run with doses between {gentlest}% "
                f"and 100%."
            ),
            value=drops[gentlest],
            threshold=target_pp,
            evidence=evidence,
        )

    dose = _dose_at(by_dose, target_pp)
    return GateResult(
        gate="G8",
        passed=True,
        detail=(
            f"the dilution ladder brackets the target: {drops[gentlest]:.2f}pp at {gentlest}% of "
            f"reasoning tokens rising to {drops[strongest]:.2f}pp at {strongest}%, putting the dose "
            f"worth {target_pp:.0f}pp at about {dose:.1f}% of them."
        ),
        value=dose,
        threshold=target_pp,
        evidence=evidence + (("dose_pct_at_target", dose),),
    )


def run_gates(
    result: EndpointResult,
    *,
    depth_scores: Optional[Mapping[int, Score]] = None,
    random_init_result: Optional[EndpointResult] = None,
    premise_ablated_result: Optional[EndpointResult] = None,
    achievable_ceiling: Optional[Score] = None,
    scores_by_params: Optional[Mapping[int, Score]] = None,
    replicates: Optional[Sequence[Score]] = None,
    dilution_scores: Optional[Mapping[int, Score]] = None,
) -> Tuple[GateResult, ...]:
    """
    Run every gate in :data:`GATES` against one endpoint.

    Each argument is the evidence one gate needs, and every one of them defaults to ``None`` -- calling
    this with a bare result is legal and returns seven refusals, each naming an arm still to run. That
    is the intended shape of an early M0 report: a checklist of what is owed, and not one pass in it.

    :param result: The endpoint's score at the cell being admitted.
    :param depth_scores: G1's task-depth sweep, keyed by depth.
    :param random_init_result: G2's untrained checkpoint.
    :param premise_ablated_result: G3's hypothesis-only probe.
    :param achievable_ceiling: G4's b=0 arm.
    :param scores_by_params: G6's parameter sweep at fixed depth.
    :param replicates: G7's k replicate runs.
    :param dilution_scores: G8's reasoning-token dilution ladder, keyed by percent retained.

    :returns: One :class:`GateResult` per gate, in :data:`GATES` order.

    :raises OLMoConfigurationError: If the gates run here do not match :data:`GATES` exactly. A gate
        added to the table and not wired in would otherwise be a gate that silently never runs.
    """
    results = (
        g1_dynamic_range(result, depth_scores=depth_scores),
        g2_label_permuted(random_init_result),
        g3_premise_ablated(result, premise_ablated_result),
        g4_headroom(result, achievable_ceiling),
        g6_capacity_responsive(scores_by_params),
        g7_resolution(replicates),
        g8_calibrated_positive_control(dilution_scores),
    )
    if tuple(one.gate for one in results) != GATES:
        raise OLMoConfigurationError(
            f"run_gates produced {[one.gate for one in results]} but PRD 8.6 defines {list(GATES)}"
        )
    return results


def require_all(results: Iterable[GateResult], *, endpoint: Optional[str] = None) -> None:
    """
    Admit the endpoint, or raise naming **every** reason it was refused.

    PRD 8.6: ``grid.run()`` raises on any endpoint that has not passed all of the gates. Every failure
    is listed at once rather than the first, because these are fixed by scheduling arms -- a caller
    that learns about one missing arm per run learns it six times.

    A gate in :data:`GATES` with no result here is itself a failure. Passing the subset that happened
    to run is how an unadmitted endpoint becomes a published number.

    :param results: The verdicts, from :func:`run_gates` or from individual gates.
    :param endpoint: The endpoint's name, for the message.

    :raises OLMoConfigurationError: If any gate failed or was never run.
    """
    collected = list(results)
    failures = [one for one in collected if not one.passed]
    seen = {one.gate for one in collected}
    never_run = [name for name in GATES if name not in seen]

    if not failures and not never_run:
        return

    label = f"endpoint '{endpoint}'" if endpoint else "this endpoint"
    lines = [f"  {one.gate}: {one.detail}" for one in failures]
    lines += [
        f"  {name}: {NO_EVIDENCE}: the gate was never run, and an unrun gate is not a passed one."
        for name in never_run
    ]
    raise OLMoConfigurationError(
        f"{label} has not passed PRD 8.6's admission gates -- "
        f"{len(failures) + len(never_run)} of {len(GATES)} refused it, so it cannot be read as a "
        f"result:\n" + "\n".join(lines)
    )


def _missing(gate: str, what: str, evidence: Tuple[Tuple[str, float], ...] = ()) -> GateResult:
    """
    The one shape a gate takes when the evidence it needs does not exist yet.

    Always ``passed=False``, always prefixed with :data:`NO_EVIDENCE`. Centralised so the rule is one
    line of code rather than a convention six gates have to remember.

    :param gate: The gate's name.
    :param what: What is missing and what to run, as a sentence.
    :param evidence: Anything the gate did manage to compute.

    :returns: The failed verdict.
    """
    return GateResult(gate=gate, passed=False, detail=f"{NO_EVIDENCE}: {what}", evidence=evidence)


def _accuracy_pp(score: Score, *, what: str) -> float:
    """
    A :data:`Score` as percentage points.

    :param score: An :class:`EndpointResult` or a fraction in ``[0, 1]``.
    :param what: Where the value came from, for the error message.

    :returns: The accuracy, in percentage points.

    :raises OLMoConfigurationError: If a bare float is outside ``[0, 1]``. Someone typing PRD 8.3's
        ``47.8`` where a fraction belongs would otherwise put a 4,780pp score through a gate that
        compares it against 20 -- and it would pass. A unit error is not a finding, so it raises here
        rather than becoming a verdict.
    """
    if isinstance(score, EndpointResult):
        return 100.0 * score.accuracy
    value = float(score)
    if not 0.0 <= value <= 1.0:
        raise OLMoConfigurationError(
            f"{what} is {value}, which is not an accuracy. Gates take fractions in [0, 1], the same "
            f"way EndpointResult.accuracy reports -- 0.478, not 47.8."
        )
    return 100.0 * value


def _floor_tolerance_pp(result: EndpointResult, sds: float) -> float:
    """
    How far above its floor a control may land and still be called at the floor, in pp.

    :param result: The control's score, whose own ``n_total`` sets the binomial SE.
    :param sds: How many SDs to allow.

    :returns: The tolerance, in percentage points.
    """
    # A floor of exactly 0 has no binomial spread, and a zero tolerance would fail the gate on a single
    # lucky item. One item's worth of accuracy is the smallest difference the eval can express, so it is
    # the smallest defensible p to size the SE at.
    p = min(max(result.floor, 1.0 / result.n_total), 1.0)
    return sds * 100.0 * math.sqrt(p * (1.0 - p) / result.n_total)


def _dose_at(by_dose: Mapping[int, float], target_pp: float) -> float:
    """
    The dilution dose whose decline equals ``target_pp``, by linear interpolation.

    Walks :data:`DILUTION_DOSES_PCT` from the reference downward and interpolates across the first
    crossing. First rather than largest, because the ladder is five noisy runs and need not be
    monotone -- and the gentlest dose that reaches the target is the conservative reading of it.

    :param by_dose: Accuracy in pp, keyed by percent of reasoning tokens retained.
    :param target_pp: The decline to solve for.

    :returns: Percent of reasoning tokens retained, interpolated.
    """
    reference = by_dose[100]
    previous_dose, previous_drop = 100, 0.0
    for dose in DILUTION_DOSES_PCT[1:]:
        drop = reference - by_dose[dose]
        if drop >= target_pp:
            if drop == previous_drop:
                return float(dose)
            span = (target_pp - previous_drop) / (drop - previous_drop)
            return previous_dose + (dose - previous_dose) * span
        previous_dose, previous_drop = dose, drop
    return float(DILUTION_DOSES_PCT[-1])


__all__: List[str] = [
    "GATES",
    "GateResult",
    "Score",
    "g1_dynamic_range",
    "g2_label_permuted",
    "g3_premise_ablated",
    "g4_headroom",
    "g6_capacity_responsive",
    "g7_resolution",
    "g8_calibrated_positive_control",
    "minimum_detectable_effect",
    "require_all",
    "run_gates",
]


GATE_REPORT_VERSION = "factcrowd.gates.v1"
"""Bumped whenever a gate's definition changes, so an old report cannot admit a new endpoint."""


@dataclass(frozen=True)
class GateReport:
    """
    A signed record that an endpoint passed admission, and the only thing that can mark a row
    confirmatory.

    PRD 8.6 says ``grid.run()`` raises on an endpoint that has not passed every gate. Until this existed
    that was a documentation claim: :func:`run_gates` and :func:`require_all` had no production caller, so
    scoring wrote rows regardless of whether any evidence had ever been gathered.

    The report is written by whoever gathers the evidence and read by
    :mod:`factcrowd.score_run`, which marks every row non-confirmatory when there is no report for its
    endpoint or the report does not pass. Nothing can grant confirmatory status implicitly.

    :param version: :data:`GATE_REPORT_VERSION` at the time of writing.
    :param endpoint: Which endpoint was admitted.
    :param results: One verdict per gate.
    :param commit: The commit the evidence was gathered at, so a report cannot outlive its code.
    :param identity: What the evidence was gathered *on* -- see :meth:`mismatch`. Empty on a report
        written before this field existed, which is admitted but says so.
    :param note: Free text for the person who ran it.
    """

    version: str
    endpoint: str
    results: Tuple[GateResult, ...]
    commit: str = ""
    identity: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def mismatch(self, row: Mapping[str, object]) -> str:
        """
        Why this report cannot speak for a row, or ``""`` when it can.

        **The endpoint name is not enough, and this is the hole it leaves.** A report says "``mano``
        passed", and until this existed nothing checked *which* ``mano``. The calibration sweep runs
        reasoning-only in the entropy architecture at 8,000 vocabulary words and 31.43M parameters; a count
        axis treatment is 3,554 and 29.71M. Those are different softmax widths and different networks, so
        "the task is learnable" measured on one is not evidence about the other -- and a gate report from
        the first would have admitted every row of the second without complaint. PRD 16.11 recorded the
        risk as a warning to be careful; this makes it mechanical.

        Only fields the report actually carries are compared, so a row that does not state one is not
        refused for it -- the report is the thing making a claim, and it can only be held to what it says.

        :param row: A collected row, or any mapping of the identity fields.

        :returns: A description of the first disagreement, or ``""``.
        """
        for key, expected in sorted(self.identity.items()):
            if key not in row or row[key] is None:
                continue
            actual = str(row[key])
            if actual != str(expected):
                return f"{key} is {actual!r} here and {str(expected)!r} in the report"
        return ""

    @property
    def passed(self) -> bool:
        """
        Whether this report admits its endpoint: every gate in :data:`GATES`, each one passing.

        **Coverage is part of the test, and leaving it out made the check fail open.** The earlier version
        asked only that the results be non-empty and all pass, so a report carrying a single passing ``G1``
        admitted a row -- and so did one carrying every real gate plus an invented ``G99``, since nothing
        compared the set against :data:`GATES`. An admission gate that a truncated file satisfies is not a
        gate. The set must match exactly: no gate missing, no gate invented, and no gate twice.
        """
        seen = [result.gate for result in self.results]
        if sorted(seen) != sorted(GATES):
            return False
        return all(result.passed for result in self.results)

    def coverage_problem(self) -> str:
        """
        Why the result set is not exactly :data:`GATES`, or ``""`` when it is.

        Separate from :attr:`failures` because "G7 failed" and "G7 is absent" send a reader to different
        places, and a report that fails on coverage would otherwise look like a report where everything
        passed.

        :returns: A description, or the empty string.
        """
        seen = [result.gate for result in self.results]
        missing = sorted(set(GATES) - set(seen))
        unknown = sorted(set(seen) - set(GATES))
        repeated = sorted({g for g in seen if seen.count(g) > 1})
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"not a gate: {unknown}")
        if repeated:
            parts.append(f"repeated {repeated}")
        return "; ".join(parts)

    @property
    def failures(self) -> Tuple[str, ...]:
        """The gates that did not pass, for the row's reason field."""
        return tuple(result.gate for result in self.results if not result.passed)

    def as_dict(self) -> Dict[str, object]:
        """Serialise for JSON."""
        return {
            "version": self.version,
            "endpoint": self.endpoint,
            "commit": self.commit,
            "note": self.note,
            "passed": self.passed,
            "identity": dict(self.identity),
            "results": [dict(result.summary()) for result in self.results],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "GateReport":
        """
        Read a report, refusing one written against a different gate definition.

        :param raw: The parsed JSON.

        :returns: The report.

        :raises OLMoConfigurationError: If the version does not match, or a required field is missing.
        """
        version = str(raw.get("version", ""))
        if version != GATE_REPORT_VERSION:
            raise OLMoConfigurationError(
                f"gate report version {version!r} does not match {GATE_REPORT_VERSION!r}. A gate "
                f"definition has changed since this evidence was gathered, so it cannot admit anything "
                f"-- re-run the gates."
            )
        endpoint = raw.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise OLMoConfigurationError("gate report has no endpoint")
        results = []
        entries = raw.get("results") or []
        if not isinstance(entries, (list, tuple)):
            raise OLMoConfigurationError("gate report's 'results' is not a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise OLMoConfigurationError("gate report has a malformed result entry")
            results.append(
                GateResult(
                    gate=str(entry.get("gate", "?")),
                    # STRICTLY A BOOLEAN. `bool("false")` is True, so a report whose JSON spelled its
                    # verdicts as strings -- which any hand-written or cross-language producer may do --
                    # read as all-passing and admitted every row.
                    passed=_strict_bool(entry.get("passed"), entry.get("gate", "?")),
                    detail=str(entry.get("detail", "")),
                )
            )
        raw_identity = raw.get("identity") or {}
        if not isinstance(raw_identity, Mapping):
            raise OLMoConfigurationError("gate report's 'identity' is not a mapping")
        return cls(
            version=version,
            endpoint=endpoint,
            results=tuple(results),
            commit=str(raw.get("commit", "")),
            identity={str(k): str(v) for k, v in raw_identity.items()},
            note=str(raw.get("note", "")),
        )


def _strict_bool(value: object, gate: object) -> bool:
    """
    A verdict, refusing anything that is not already a boolean.

    :param value: The raw value.
    :param gate: Which gate, for the message.

    :returns: The verdict.

    :raises OLMoConfigurationError: If it is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise OLMoConfigurationError(
            f"gate {gate!r} has passed={value!r}, which is {type(value).__name__} and not a boolean. "
            f"`bool('false')` is True, so this is refused rather than coerced."
        )
    return value


def _read_text(path: str) -> str:
    """
    Read a file that may be local or in object storage.

    ``Path("s3://b/k").read_text()`` does not fail usefully -- ``Path`` collapses the double slash to
    ``s3:/b/k`` and looks for a *local relative directory called* ``s3:``. Writing took the same route and
    silently put a gate report in ephemeral container storage, so a report that a log said had been written
    did not exist anywhere afterwards.

    :param path: A local path or a URL.

    :returns: The contents.
    """
    from pathlib import Path

    from olmo_core.io import get_bytes_range, get_file_size, is_url

    if not is_url(path):
        return Path(path).read_text()
    return get_bytes_range(path, 0, get_file_size(path)).decode("utf-8")


def load_reports(path: str) -> Dict[str, GateReport]:
    """
    Read a JSON file of gate reports, keyed by endpoint.

    :param path: A JSON file holding either one report or a list of them.

    :returns: One report per endpoint.

    :raises OLMoConfigurationError: If the file is unreadable or two reports claim one endpoint.
    """
    import json

    raw = json.loads(_read_text(path))
    entries = raw if isinstance(raw, list) else [raw]
    out: Dict[str, GateReport] = {}
    for entry in entries:
        report = GateReport.from_dict(entry)
        if report.endpoint in out:
            raise OLMoConfigurationError(
                f"two gate reports claim endpoint {report.endpoint!r}; which one admits it is ambiguous"
            )
        out[report.endpoint] = report
    return out
