#!/usr/bin/env python3
"""The training-dose adjustment, pre-registered on 2026-08-10 before any treatment endpoint was read.

WHAT THE PROBLEM IS. ``SkipStepAdamW`` declines a step by multiplying the whole update by a
0/1 factor: the parameters do not move, the two moments do not move, the decoupled weight
decay does not apply, and the Adam step counter does not increment
(``src/olmo_core/optim/adamw.py:29-47`` non-foreach, ``:71-102`` foreach, both reached with
``step_increment_bugfix=True``, which is ``SkipStepAdamWConfig``'s default). The trainer's
global step, the cosine schedule and the data loader advance anyway. **A declined step
therefore consumes its tokens, moves the schedule along, and performs no optimization.**

The rule is identical on all four arms, and the pre-registration argued from that identity
that "the contrasts are unaffected" (``hyper-connections.md``, "If it does not work"). That is
the error this module exists to correct. The rule is identical; its *action rate* is not,
because ``get_step_factor`` compares each run against a rolling window of **that run's own**
previous 128 losses and gradient norms (``skip_step_optimizer.py:94-109``). The number of
declined steps is a post-randomisation variable on the causal path from arm to endpoint, and
an arm that declines more is an arm that has been trained less at the same nominal horizon. So
the amount of training becomes a function of the arm, which is a confound of the same order as
the effect being measured -- see :data:`GATE_NATS` against :data:`CRITICAL_DECLINE_GAP`.

HOW THE COUNT ACCUMULATES, WHICH IS WHY IT BOUNDS RATHER THAN CORRECTS. The step's own loss and
gradient norm go into the rolling window *before* ``step()`` reads it, and nothing takes them
out again when the step is declined (``skip_step_optimizer.py:59-76``). A declined step therefore
sits in the window judging the next 128 steps and lifts the mean and standard deviation that
declining again would have to clear: driven against the real optimizer, an isolated spike is
declined exactly once and identical spikes right behind it are accepted. Declines are
anti-clustered, so ``delta_n`` is a slow difference in rates across 6,000 steps rather than a
burst -- and, more to the point here, the count is censored by its own history. The same
instability produces a different count depending on where in the window it arrives, by an
arm-dependent amount nothing can recover. That is the third reason the count is used only to
*bound* a contrast at the top of the slope interval and never to *correct* one.

WHY THE PRIMARY ESTIMAND DOES NOT MOVE, WHICH IS THE DECISION THIS MODULE ENCODES. Three
options were weighed and are recorded in ``hyper-connections.md`` under "The dose amendment of
2026-08-10".

* **Equal applied updates rather than equal global steps** is the clean fix and it is
  unavailable. Three arms are already running to 6,000 *global* steps and cannot be changed;
  evaluations land every 500 steps, which is ten to fifty times coarser than the tens of steps
  the confound is made of; and the cosine is indexed to the global step, so two cells matched
  on applied updates sit at different points of their own schedules. It would remove one
  confound by introducing another.
* **Declined count as a covariate** conditions on a mediator. It estimates a controlled direct
  effect rather than the total effect, under an assumption nothing here can check, and if
  hyper-connections genuinely destabilise training then the adjustment subtracts part of the
  real effect. It is reported, and it is not primary.
* **Adjusted and unadjusted side by side** is what is adopted, with the rule below that makes
  the pair decide something. Two numbers printed next to each other with no rule about what to
  do when they disagree is not a pre-registration; it is a degree of freedom with a table.

**The primary estimand is unchanged: the total effect at 6,000 global steps.** Declining is
part of what an arm does, so the intention-to-treat contrast is the quantity the module's
question is about. What is added is a band that can only ever *withhold* a claim and can never
create one -- which is what makes it safe to freeze a slope this poorly determined.

THE RULE, IN ONE PARAGRAPH. Let ``delta`` be the contrast in nats, signed so that negative is
the treatment improving on the comparator, and let ``delta_n`` be the treatment's mean declined
count minus the comparator's. A declined step is lost training, so the dose contributes
``+beta * delta_n`` nats to ``delta``. When that term carries the *same* sign as the effect the
hypothesis predicts, the dose could have manufactured the result, and the claim stands only if
``abs(delta) - abs(delta_n) * beta_high >= gate``. When it carries the opposite sign the
treatment trained less and scored well anyway, the unadjusted estimate is conservative, and no
penalty is applied -- it is reported and that is all. One-sided on purpose.

WHAT THE SLOPE IS AND WHAT IS WRONG WITH IT, STATED HERE RATHER THAN DISCOVERED LATER. It comes
from the only pre-existing measurement of what declining costs: the three baseline seeds that
never spiked, run once under ``AdamW`` and once under ``SkipStepAdamW`` at the same
``init_seed`` and the same ``data_loader.seed`` (``hyper-connections.md``, the cell-by-cell
table). Three things are wrong with it and all three are recorded rather than smoothed over.

1. It is a **ratio of means through the origin**, not a regression. It attributes the whole
   ``AdamW`` to ``SkipStepAdamW`` movement to declining. The two optimizers are also a
   different kernel path -- ``torch.optim.AdamW`` against this repository's own foreach
   re-implementation -- so part of that movement is numerics. The slope is therefore an
   **over-estimate**, which is the direction a withholding rule wants.
2. It is **three points, and it does not clear zero**. The mean movement is +0.00096 nats with
   a t of 2.27 on df = 2, p = 0.15. :data:`PER_DECLINE_NATS_HIGH` is the upper end of its 95%
   interval and is 2.9 times the point estimate.
3. The **within-sample regression** of movement on declined count over those same three cells
   is *negative*, at -1.03e-04 nats per decline. Three points at 16, 18 and 20 declines carry
   almost no leverage, so that number is noise and is not used; it is recorded because a reader
   who recomputes the slope the obvious way will get it and should know it was seen.

The honest summary is that the *direction* is certain -- a declined step is strictly less
optimization -- and the *magnitude* is known to within about a factor of three. That is why the
band is quoted at the top of the interval and why the point-adjusted estimate is a secondary.

HOW TO NARROW IT, IF ANYBODY EVER WANTS TO. Two baseline cells under ``SkipStepAdamW`` with
``--skip-step-sigma-factor`` set high enough that the rule never fires would separate the
kernel path from the dose, and two more at a factor low enough to force a target decline count
would identify the slope on more than three points. Both are pre-registered as optional in
``hyper-connections.md`` and neither is funded. Until one of them runs, the constants in this
module do not move: they are frozen literals, a test re-derives them from the published cell
table, and a second test asserts the frozen value has not drifted.
"""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

#: Nats of held-out cross-entropy per bit-per-byte, at the tranche's ``--bytes-per-token 4.57``.
#: Restated here rather than imported so this module can be read and tested without pulling in
#: the W&B reader; a test asserts it equals ``noise_floor.NATS_PER_BPB``.
NATS_PER_BPB = 4.57 * math.log(2.0)

#: The three baseline seeds that never spiked, as published in ``hyper-connections.md``'s
#: cell-by-cell table before this module existed: the movement from ``AdamW`` to
#: ``SkipStepAdamW`` on the same seed, in bits-per-byte, and the declined-step count of the
#: ``SkipStepAdamW`` cell.
#:
#: SEEDS 0 AND 1 ARE DELIBERATELY ABSENT. Both spiked under ``AdamW``, so their movement is
#: -0.011 and -0.013 BPB and is the spike penalty being recovered rather than the cost of
#: declining. Including them would give a large negative slope and would be measuring the
#: intervention's benefit, not its dose.
CLEAN_CELL_MOVEMENT_BPB: Tuple[float, ...] = (0.00024, 0.00056, 0.00011)
CLEAN_CELL_DECLINES: Tuple[int, ...] = (16, 18, 20)

#: What one declined step costs the endpoint, in nats. The ratio of the mean movement above to
#: the mean declined count above. Frozen; ``test_dose_adjustment`` re-derives it.
PER_DECLINE_NATS = 5.3381e-05

#: The upper end of the 95% interval on the same quantity, t on df = 2. **This is the constant
#: the withholding rule uses**, because the question the band answers is "how large could the
#: dose effect be", and answering it at the point estimate would make the band decorative.
PER_DECLINE_NATS_HIGH = 1.5462e-04

#: The lower end of the same interval, which is negative. Recorded because quoting only the
#: upper end of an interval that crosses zero would misrepresent what is known.
PER_DECLINE_NATS_LOW = -4.7861e-05

#: The pre-registered gate, in nats: ``2 x SE`` at sigma-hat = 0.00205 and 5 v 5.
#: Used only to state :data:`CRITICAL_DECLINE_GAP` as a standing property of the design. Every
#: live check reads the gate off the contrast it is checking rather than from here.
GATE_NATS = 0.0026

#: The difference in declined-step counts at which the dose alone spans the whole gate, at
#: :data:`PER_DECLINE_NATS_HIGH`. Seventeen steps. At the point estimate it is 49, which is the
#: figure the adversarial review of 2026-08-10 quotes; the interval is why the operative number
#: is a third of it.
CRITICAL_DECLINE_GAP = GATE_NATS / PER_DECLINE_NATS_HIGH


def slope_from_clean_cells(
    movement_bpb: Sequence[float] = CLEAN_CELL_MOVEMENT_BPB,
    declines: Sequence[int] = CLEAN_CELL_DECLINES,
) -> Tuple[float, float, float]:
    """
    Re-derive the per-declined-step slope and its 95% interval from the published cells.

    The estimator is the ratio of the mean endpoint movement to the mean declined count, and
    the interval is the t interval on the mean movement carried through the same division. It
    is a ratio of means and not a regression, for the reason the module docstring gives: three
    counts spanning 16 to 20 have no leverage, and the identification comes from the pairing --
    the same seed run twice -- rather than from the spread of the counts.

    :param movement_bpb: Endpoint movement per clean cell, ``SkipStepAdamW`` minus ``AdamW``.
    :param declines: The declined-step count of the corresponding ``SkipStepAdamW`` cell.

    :returns: ``(point, low, high)`` in nats per declined step, the interval two-sided at 95%.

    :raises ValueError: If the two sequences differ in length or carry fewer than two cells,
        where there is no interval to report.
    """
    if len(movement_bpb) != len(declines):
        raise ValueError("a movement and a declined count are needed for each cell")
    n = len(movement_bpb)
    if n < 2:
        raise ValueError("a slope with no interval is the thing this module refuses to quote")

    from scipy import stats

    mean_movement = sum(movement_bpb) / n
    mean_declines = sum(declines) / n
    variance = sum((v - mean_movement) ** 2 for v in movement_bpb) / (n - 1)
    standard_error = math.sqrt(variance / n)
    half = float(stats.t.ppf(0.975, n - 1)) * standard_error

    scale = NATS_PER_BPB / mean_declines
    return (mean_movement * scale, (mean_movement - half) * scale, (mean_movement + half) * scale)


@dataclass(frozen=True)
class DoseCheck:
    """
    What the training-dose difference does to one contrast, and whether the claim survives it.
    """

    name: str
    treatment: str
    comparator: str

    declined_treatment: Tuple[int, ...]
    declined_comparator: Tuple[int, ...]

    delta_declines: float
    """Treatment mean declined count minus comparator mean. Positive means the treatment was
    trained *less*."""

    delta_nats: float
    """The contrast under test, signed so that negative is the treatment improving."""

    gate_nats: float
    """``2 x SE`` for this contrast, in nats, read off the contrast rather than assumed."""

    dose_nats: float
    """``delta_declines x PER_DECLINE_NATS``: the dose's contribution at the point estimate."""

    dose_nats_high: float
    """The same at :data:`PER_DECLINE_NATS_HIGH`, which is what the rule below uses."""

    adjusted_delta_nats: float
    """``delta_nats - dose_nats``. A pre-committed **secondary**, never the primary reading."""

    band_nats: Tuple[float, float]
    """``delta_nats`` plus and minus ``abs(dose_nats_high)``."""

    dose_favours_the_claim: bool
    """
    Whether the dose pushes the contrast in the direction the hypothesis predicts. True is the
    dangerous case: the treatment declined *fewer* steps, so it was trained more, so it would
    look better for a reason that is not the mechanism.
    """

    clears_gate_unadjusted: bool
    survives_the_dose: bool
    """
    The pre-registered verdict. Equal to ``clears_gate_unadjusted`` whenever the dose opposes
    the claim or the arms declined the same number of steps. Otherwise it additionally requires
    that the contrast still clear the gate after the largest dose the interval allows.
    """

    critical_delta_declines: float
    """``gate / PER_DECLINE_NATS_HIGH``: the decline gap at which the dose alone spans the gate
    for this contrast. Read ``delta_declines`` against it."""

    verdict: str


def dose_check(
    name: str,
    treatment: str,
    comparator: str,
    delta_nats: float,
    gate_nats: float,
    declined_treatment: Sequence[Optional[int]],
    declined_comparator: Sequence[Optional[int]],
    predicted_sign: int = -1,
    slope: float = PER_DECLINE_NATS,
    slope_high: float = PER_DECLINE_NATS_HIGH,
) -> DoseCheck:
    """
    Apply the pre-registered dose rule to one contrast.

    :param name: The hypothesis, for the report.
    :param treatment: Arm name.
    :param comparator: Arm name.
    :param delta_nats: Treatment mean minus comparator mean, in nats, on the primary endpoint.
    :param gate_nats: ``2 x SE`` for this contrast, in nats.
    :param declined_treatment: Per-seed declined counts on the treatment arm.
    :param declined_comparator: Per-seed declined counts on the comparator arm.
    :param predicted_sign: ``-1`` when the hypothesis predicts the treatment lowers the loss,
        which is every hypothesis in this module.
    :param slope: Nats per declined step, for the adjusted point estimate.
    :param slope_high: Nats per declined step at the top of the interval, for the band.

    :returns: The check.

    :raises ValueError: If either arm is missing a declined count. **Not defaulted to zero**:
        the count arrives from ``stability/steps skipped`` and an absent key is missing data,
        whereas zero is the claim that the rule never fired. Reading the first as the second
        would silently report a dose difference of zero on exactly the arm whose instrumentation
        failed, which is the arm least entitled to the benefit of the doubt.
    """
    if any(v is None for v in declined_treatment) or any(v is None for v in declined_comparator):
        raise ValueError(
            f"{name}: a declined-step count is missing, so the training dose each arm received "
            "is unknown and the contrast cannot be checked against it. An absent "
            "'stability/steps skipped' key is missing data and is not a count of zero."
        )
    if not declined_treatment or not declined_comparator:
        raise ValueError(f"{name}: both arms need at least one cell with a declined count")

    treated = tuple(int(v) for v in declined_treatment)  # type: ignore[arg-type]
    compared = tuple(int(v) for v in declined_comparator)  # type: ignore[arg-type]

    # The difference of arm means. At equal n this is also the mean of the per-seed paired
    # differences, so it is the right quantity under either primary analysis and there is no
    # second version of it to choose between after the fact.
    delta_declines = sum(treated) / len(treated) - sum(compared) / len(compared)

    dose = delta_declines * slope
    dose_high = delta_declines * slope_high

    # A declined step is lost training, so the dose raises the loss of whichever arm declined
    # more, and it can only manufacture the predicted effect when it carries that effect's own
    # sign. `predicted_sign` is -1 and the slope is positive, so this is the case where the
    # treatment declined FEWER steps than the comparator.
    favours = dose != 0.0 and (dose < 0.0) == (predicted_sign < 0)

    clears = abs(delta_nats) >= gate_nats
    survives = clears and (not favours or abs(delta_nats) - abs(dose_high) >= gate_nats)

    if not clears:
        verdict = "does not clear the gate; the dose changes nothing"
    elif not favours:
        verdict = (
            f"clears the gate, and the dose opposes the claim: the treatment declined "
            f"{delta_declines:+.1f} steps against the comparator, so it was trained less and "
            f"scored well anyway. The unadjusted estimate is conservative by up to "
            f"{abs(dose_high):.5f} nats."
        )
    elif survives:
        verdict = (
            f"clears the gate and survives the dose: the treatment declined "
            f"{delta_declines:+.1f} steps, worth at most {abs(dose_high):.5f} nats, against a "
            f"contrast of {abs(delta_nats):.5f} and a gate of {gate_nats:.5f}."
        )
    else:
        verdict = (
            f"DOSE-LIMITED, and the claim is not made. The treatment declined "
            f"{delta_declines:+.1f} steps fewer than the comparator, which is worth up to "
            f"{abs(dose_high):.5f} nats in the direction of the claim, and the contrast of "
            f"{abs(delta_nats):.5f} does not clear the gate of {gate_nats:.5f} once that is "
            "taken off. The effect is not separable from the training-dose difference at this "
            "precision."
        )

    return DoseCheck(
        name=name,
        treatment=treatment,
        comparator=comparator,
        declined_treatment=treated,
        declined_comparator=compared,
        delta_declines=delta_declines,
        delta_nats=delta_nats,
        gate_nats=gate_nats,
        dose_nats=dose,
        dose_nats_high=dose_high,
        adjusted_delta_nats=delta_nats - dose,
        band_nats=(delta_nats - abs(dose_high), delta_nats + abs(dose_high)),
        dose_favours_the_claim=favours,
        clears_gate_unadjusted=clears,
        survives_the_dose=survives,
        critical_delta_declines=gate_nats / slope_high if slope_high > 0 else float("inf"),
        verdict=verdict,
    )


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """
    Holm-Bonferroni step-down adjustment over a family of p-values.

    THE OTHER HALF OF THE SAME AMENDMENT, AND IT IS HERE BECAUSE FUNDING ARM 4 IS WHAT MADE IT
    URGENT. The pre-registration applies no multiplicity correction to the gate and says why:
    the hypotheses are fixed in advance and each is reported with its effect size, interval and
    p-value, so the table has not been selected on. That reasoning was written for a table read
    as a family of effect sizes. The live design leads with a 2-SE gate, which at df = 16 is a
    6.3% per-comparison test, over a family that went from three comparisons to five on
    2026-08-10.

    **REPORTED BESIDE THE GATE AND NEVER INSTEAD OF IT.** The decision rule is untouched; what
    changes is that the family-wise reading is printed rather than left for a reader to
    reconstruct.

    Holm rather than Bonferroni because it is uniformly more powerful and assumes no less, and
    the running maximum enforces monotonicity so two adjusted values cannot cross.

    :param p_values: Raw two-sided p-values, keyed by hypothesis.

    :returns: Adjusted p-values under the same keys, each capped at 1.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * p))
        adjusted[name] = running
    return adjusted


def render(checks: Sequence[DoseCheck]) -> str:
    """
    The dose block of the report.

    :param checks: One per analysable hypothesis, in the report's own order.

    :returns: The rendered block, empty when there is nothing to check.
    """
    if not checks:
        return ""

    lines = [
        "TRAINING DOSE (pre-registered 2026-08-10, before any treatment endpoint was visible)",
        "",
        "  a declined step performs no update, so an arm that declines more is trained less.",
        f"  slope {PER_DECLINE_NATS:.3e} nats per declined step, 95% interval "
        f"[{PER_DECLINE_NATS_LOW:.3e}, {PER_DECLINE_NATS_HIGH:.3e}], df = 2, over-estimated on"
        " purpose.",
        "",
        f"  {'':<5} {'d(declines)':>12} {'critical':>9} {'delta':>10} {'gate':>9} "
        f"{'adjusted':>10}  verdict",
    ]
    for check in checks:
        lines.append(
            f"  {check.name:<5} {check.delta_declines:>12.1f} "
            f"{check.critical_delta_declines:>9.1f} {check.delta_nats:>10.5f} "
            f"{check.gate_nats:>9.5f} {check.adjusted_delta_nats:>10.5f}  "
            f"{'survives' if check.survives_the_dose else 'DOSE-LIMITED' if check.clears_gate_unadjusted else 'below gate'}"
        )
    lines.append("")
    for check in checks:
        lines.append(f"  {check.name}: {check.verdict}")
    return "\n".join(lines)
