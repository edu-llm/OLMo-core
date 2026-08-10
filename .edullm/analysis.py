#!/usr/bin/env python3
"""Run the pre-registered analysis of the hyper-connection tranche, end to end, in one command.

WRITTEN BEFORE THE TREATMENT ARMS LAND, WHICH IS THE ONLY TIME IT IS WORTH WRITING. Every
choice below -- which endpoint, which degrees of freedom, paired or unpaired, what counts as
clearing the gate, which test H7 gets -- is free now and is a researcher degree of freedom the
moment an arm mean is visible. The tests in ``test_analysis.py`` drive the whole pipeline over
synthetic tranches with a planted effect, so the estimators are checked against a truth nobody
could have preferred, and the reading path is checked against the frozen baseline artifact,
whose numbers were published before this file existed.

A FOURTH SIBLING, AND THE DIVISION OF LABOUR IS THE SAME ONE THE OTHER THREE KEEP.
``wandb_panels.py`` asks whether a metric *key* arrived. ``stage_gate.py`` asks whether a live
submission is healthy. ``noise_floor.py`` holds the estimators and asks what one arm's numbers
are worth. This module asks the question all three exist to make answerable: **what do the four
arms say about H1, H2a, H5 and H7**. It imports its estimators from ``noise_floor`` rather than
restating them, so the c4 correction, the randomized-block df and the exact noncentral t are
the same code the pre-registration's own tables are generated from.

WHAT IS PRE-REGISTERED AND WHAT IS NOT. The contract is ``hyper-connections.md``. Everything
this module reports as *primary* is in it:

    endpoint      unweighted mean of held-out bits-per-byte over the seven sources at the
                  final step. The strata-weighted composite is frozen in
                  ``noise-floor-skip-step.json`` and is reported beside it as a secondary,
                  because it was committed and buys 1.08x out of sample.
    sigma-hat     pooled within-arm, over all four arms, df = k(n-1) = 16. Bartlett at
                  alpha = 0.05 on the four within-arm variances, with the pre-committed
                  consequence -- Welch everywhere -- if it rejects.
    primary test  the paired difference blocked on seed, error df (k-1)(n-1) = 12, if rho-hat
                  clears the break-even; the unpaired difference of arm means if it does not.
                  Whichever is not primary is reported as the secondary.
    the gate      |delta| >= 2 SE(delta), with the exact two-sided t p-value beside it and the
                  5% line stated at this design's own df rather than at the df = 6 the
                  pre-registration quoted it for.
    H7            exact two-sided permutation test over the 5 + 5 declined-step counts, and
                  the same test on the per-run largest triggering gradient norm.
    the dose      a declined step performs no update, so an arm that declines more is trained
                  less at the same nominal horizon. Every contrast carries the paired
                  difference in declined counts and a band of that difference times the top of
                  the per-decline slope's interval. The band can withhold a claim and can never
                  create one. Pre-registered 2026-08-10, before any treatment endpoint was
                  visible; the estimand is unchanged and stays the total effect at 6,000 global
                  steps. ``dose_adjustment`` carries the whole argument.

Anything this module adds is labelled ``post-hoc`` with the date it was added and is never
allowed to displace the pre-registered reading. There are four such additions and
:data:`POST_HOC` is the list of them.

NOTHING HERE FALLS BACK TO ANYTHING. ``noise_floor.py --submission X --dry-run`` once printed a
complete, internally consistent, entirely synthetic report under a submission id it had been
handed and never read, and it was acted on for twelve hours. The lesson is not "label the
synthetic output" -- it *was* labelled, on line one and on every block -- it is that a tool
which can answer without data will eventually be asked to. So:

* the measured path takes an explicit ``--arm <name>=<submission>`` per arm and reads exactly
  those cells. There is no group-wide sweep, because a group is not an experimental unit.
* a missing arm, a short arm, a cell that did not reach the horizon, two cells sharing a seed,
  a cell whose own config says it ran a different arm, or an absent ``stability/*`` family is a
  **refusal**, not a smaller analysis. ``--allow-provisional`` downgrades the ones that are
  about completeness to warnings and stamps ``PROVISIONAL`` on every artifact it writes.
* the synthetic path is a separate verb, ``--demo``, which cannot be reached from the measured
  one, writes its files under a ``synthetic-`` prefix, prefixes every line of output, and
  stamps the figures. It exists so the figures can be reviewed before the data lands.

    python .edullm/analysis.py --self-test                       # no network
    python .edullm/analysis.py --demo --out analysis/demo        # no network, synthetic
    python .edullm/analysis.py \\
        --group hyper-connections-370m \\
        --arm baseline=run_019fe40f-c71e \\
        --arm faithful=run_019fe90b-f99e \\
        --arm output-only=run_019fe9f6-d060 \\
        --arm mhc=run_019fe9f6-d4f8 \\
        --out analysis --cache /tmp/hc-analysis.json

Needs ``scipy`` and, for the figures, ``matplotlib``. Both are in the ``dev`` extra, for the
reason ``noise_floor`` gives: this runs on a laptop against W&B and never inside a container.
"""

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from noise_floor import (  # noqa: E402
    CELL_SUFFIX,
    HELD_OUT_SOURCES,
    NATS_PER_BPB,
    c4,
    is_crash_report,
    mde,
    paired_correlation,
    pooled_sigma,
    power_of,
    sources_from_config,
    steps_from_summary_and_history,
)
from stage_gate import arms_consistent_with  # noqa: E402

import dose_adjustment  # noqa: E402  isort:skip

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: The primary endpoint, per source, exactly as ``noise_floor`` reads it.
BPB_METRIC = "eval/lm/{source}/BPB"

#: Read beside it only as a consistency check. ``bytes_per_token`` is one constant across all
#: seven sources, so ``CE = BPB * NATS_PER_BPB`` should hold to floating point on every cell of
#: every arm. If it stops holding, the nats column of this report is wrong and the reader has
#: to be told rather than shown a plausible number -- see :func:`check_the_nats_conversion`.
CE_METRIC = "eval/lm/{source}/CE loss"

#: What the training curve is drawn from, and where a loss spike would be visible if the
#: optimizer ever failed to decline one.
TRAIN_LOSS_METRIC = "train/CE loss"
GRAD_NORM_METRIC = "optim/total grad norm"

#: The three ``SkipStepMonitorCallback`` keys H7 rests on. The first two are written on every
#: optimizer step of every arm, so their *absence* is missing data and not a count of zero --
#: which is the whole reason they are checked rather than defaulted.
SKIPPED_COUNT_METRIC = "stability/steps skipped"
MAX_TRIGGER_METRIC = "stability/largest grad norm at a skipped step"
SKIP_TRIGGER_METRIC = "stability/grad norm at a skipped step"

#: Two keys that say, from the run's own history, whether it had lanes.
#:
#: ``train_hyper_connections.train`` attaches the lane monitor when ``arm.hyper_connections`` is
#: not None and never otherwise, so ``wandb_panels`` requires both of these on every arm with
#: lanes and treats either of them on a ``baseline`` cell as proof the cell did not run the arm
#: it was submitted as. That makes their presence *and* their absence evidence, which is what a
#: cell whose saved config was destroyed needs somebody to accept in place of the config. They
#: are exact keys and not the module's globbed families, because this is a history scan.
LANE_WITNESS_METRICS = ("hc/min lane norm spread", "hc/composite spectral radius")

#: The four funded arms, in the pre-registration's own numbering, which is the order every
#: table and every figure in this module uses.
ARM_ORDER: Tuple[str, ...] = ("baseline", "faithful", "output-only", "mhc")

#: The step count the tranche was submitted for.
HORIZON = 6000

#: Seeds per arm.
SEEDS_PER_ARM = 5

#: The gate, in standard errors of the contrast under test. Two, as pre-registered, and *not*
#: the 5% line -- with sigma estimated, two standard errors is a 9.2% test at df = 6 and 8.3%
#: at df = 16, so the exact t p-value is reported beside it and so is the t critical value at
#: this design's own df.
GATE_SIGMAS = 2.0

#: The break-even correlation the pre-registration commits the paired-or-unpaired choice to.
#: Below it, pairing costs more in degrees of freedom than it buys in variance and the unpaired
#: contrast becomes primary. Quoted for a 3-arm design; :func:`break_even_rho` recomputes it
#: for whatever design actually ran, and the recomputed figure is reported as post-hoc.
PRE_REGISTERED_BREAK_EVEN_RHO = 0.09


@dataclass(frozen=True)
class Hypothesis:
    """One pre-registered contrast, and the direction it claims."""

    name: str
    treatment: str
    comparator: str
    claim: str

    predicted_sign: int
    """
    ``-1`` when the hypothesis predicts the treatment *lowers* held-out cross-entropy, which is
    the direction every hypothesis here points in. Carried explicitly because the endpoint is a
    loss and "better" is "smaller", and a sign error in a one-line summary is the cheapest way
    to publish the opposite of a result.
    """

    reference_effects: Tuple[Tuple[str, float], ...] = ()
    """
    Effects from the literature, in nats, that the interval is read against. Reported as
    *bounds* -- "the data are consistent with anything between X and Y" -- and never as
    equivalence claims, which this design cannot support and the pre-registration forbids.
    """


#: The literature this module exists to adjudicate, in nats of held-out cross-entropy, signed
#: so that a negative number is an improvement.
BYTEDANCE = ("ByteDance, OLMo-1B x4 over 500B tokens", -0.030)
TENCENT = ("Tencent, 1.2B dense, downstream", -0.020)
H1_CLAIM = ("H1's own pre-registered effect", -0.025)
ATTENUATED = ("the attenuation a 370M replication should expect", -0.010)


#: The hypotheses the funded four arms can address. H2b, H3, H4 and H6 are not here because
#: the tranche does not fund the arms or the scoring job they need; the pre-registration says
#: so, and a hypothesis silently absent from an analysis script is one nobody notices was
#: never answered.
HYPOTHESES: Tuple[Hypothesis, ...] = (
    Hypothesis(
        name="H1",
        treatment="faithful",
        comparator="baseline",
        claim="Arm 2 beats arm 1 by >= 0.025 nats. CONFOUNDED with the output-init rescale, "
        "which arm 4 would disentangle and which the tranche does not fund, so an effect here "
        "is attributable to hyper-connections PLUS their initialization prescription.",
        predicted_sign=-1,
        reference_effects=(BYTEDANCE, H1_CLAIM, TENCENT, ATTENUATED),
    ),
    Hypothesis(
        name="H2a",
        treatment="faithful",
        comparator="output-only",
        claim="Arm 2 > arm 3 on held-out cross-entropy. Whether the field's negative result is "
        "an artifact of a reimplementation that kept the output mixing and dropped the learned "
        "input map. NOT evidence about the published downstream result, which is H2b and is "
        "blocked.",
        predicted_sign=-1,
        reference_effects=(BYTEDANCE, TENCENT, ATTENUATED),
    ),
    Hypothesis(
        name="H5",
        treatment="mhc",
        comparator="faithful",
        claim="Arm 9 >= arm 2, with the gap larger wherever arm 2 is unstable. mHC pins the "
        "lane-mixing spectral radius at 1 by construction, so a null here is readable: the "
        "monitor says whether the constraint bound.",
        predicted_sign=-1,
        reference_effects=(BYTEDANCE, TENCENT, ATTENUATED),
    ),
)


#: Everything this module reports that the pre-registration does not, each with the date it was
#: added. Printed in full at the foot of every report, so that the list a reader has to check
#: is in the output rather than in a commit message.
POST_HOC: Tuple[Tuple[str, str], ...] = (
    (
        "2026-08-09",
        "the break-even rho recomputed for the four-arm design that actually ran. The "
        "pre-registered 0.09 was derived at three arms, where paired df is 8 against an "
        "unpaired 12; at four arms it is 12 against 16. The pre-registered constant stays the "
        "operative threshold and the recomputed one is reported beside it.",
    ),
    (
        "2026-08-09",
        "the design-level intraclass correlation from the randomized-block fit, beside the "
        "pairwise rho-hat the pre-registration names. The pre-registration says rho-hat is "
        "estimated 'from the H1 and H2a quintuples' and does not say how two numbers decide "
        "one question, so the rule is applied per contrast on that contrast's own pair, which "
        "is the most literal reading, and the ICC is reported as the design-level summary.",
    ),
    (
        "2026-08-09",
        "reference effects read as bounds. For every contrast the 95% interval is checked "
        "against ByteDance's -0.030, Tencent's -0.020, H1's own -0.025 and the -0.010 a 370M "
        "replication should expect. Stated as 'the data are consistent with anything between X "
        "and Y' and never as an equivalence claim, which this design cannot support.",
    ),
    (
        "2026-08-09",
        "H7 run on output-only and mhc against the baseline as well as on faithful. Only the "
        "faithful-against-baseline test is H7; the other two are descriptive and carry no "
        "claim, and they are here because an arm that destabilises training is worth seeing "
        "whichever arm it is.",
    ),
)


class Refusal(RuntimeError):
    """
    A condition under which this module declines to produce numbers.

    SEPARATE FROM AN ORDINARY EXCEPTION SO THAT IT PRINTS AS A VERDICT RATHER THAN A TRACEBACK.
    Every one of these is a case where a smaller or a substituted analysis would have been
    possible and would have been wrong: four arms where three landed, a horizon where a
    mid-run read is available, a stability count where an absent key would default to zero.
    The failure this module is built against is a plausible number, so the alternative to a
    refusal is never "no answer" -- it is "an answer that reads the same as a real one".
    """


# ---------------------------------------------------------------------------------------
# Reading the runs. Real data only, addressed by submission.
# ---------------------------------------------------------------------------------------


@dataclass
class ArmSeries:
    """One arm of the tranche, reduced to everything any estimator or figure needs."""

    arm: str
    submission: str
    seeds: List[int] = field(default_factory=list)
    run_ids: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)

    excluded: List[List[str]] = field(default_factory=list)
    """
    ``[run id or cell name, reason]`` for every cell of this submission that was read and then
    left out of the arm.

    THE ONE BEHAVIOUR THAT IS NOT ALLOWED IS COMPUTING WITH n - 1 AND NOT SAYING SO. A missing
    replicate narrows the interval, drops the df and moves the mean, and it does none of those
    things at random: a cell loses its evaluations by hitting a wall, dying, or having its
    summary overwritten by its own crash report, so the cells that leave are the slow ones and
    the unlucky ones. An arm mean over the survivors is biased in a direction nobody chose. So
    every departure is named with a reason, the reasons are printed at the top of the report,
    written into the JSON, drawn on every figure, and make the whole read provisional.
    """

    recovered: List[str] = field(default_factory=list)
    """
    One sentence per cell whose record had to be reconstructed, and what it was reconstructed
    from. A recovery is not free -- it is an inference -- so it is reported rather than folded
    into a number that then looks like every other number.
    """

    summary_steps: List[Optional[int]] = field(default_factory=list)
    history_steps: List[int] = field(default_factory=list)
    """
    What each included cell's summary claims and what its history holds. Kept apart because
    ``summary None, history 4910`` is a clobbered summary and ``summary None, history -1`` is a
    cell that never started, and reading the first as the second is how the furthest-progressed
    cell of a submission was reported as one that never began.
    """

    steps: List[int] = field(default_factory=list)
    """Evaluation steps every cell of this arm reached."""

    bpb: List[List[List[float]]] = field(default_factory=list)
    """``[seed][step][source]`` bits-per-byte."""

    ce: List[List[List[float]]] = field(default_factory=list)
    """The same cells in nats, read off the run rather than converted, for the cross-check."""

    sources: List[str] = field(default_factory=list)

    declined: List[Optional[int]] = field(default_factory=list)
    """Per-seed count of declined optimizer steps. ``None`` where the key never arrived."""

    largest_trigger: List[Optional[float]] = field(default_factory=list)
    """Per-seed maximum triggering gradient norm. ``None`` where the key never arrived."""

    declined_steps: List[List[int]] = field(default_factory=list)
    """Per-seed list of the steps that were declined, for the stability figure."""

    train_curve_steps: List[List[int]] = field(default_factory=list)
    train_curve_loss: List[List[float]] = field(default_factory=list)
    """Per-seed training cross-entropy, downsampled, for the loss-curve figure."""

    def endpoint_matrix(self) -> np.ndarray:
        """``(n_seeds, n_sources)`` bits-per-byte at the last shared evaluation."""
        return np.asarray([seed[-1] for seed in self.bpb], dtype=float)

    def trajectory(self) -> np.ndarray:
        """``(n_seeds, n_steps, n_sources)`` bits-per-byte."""
        return np.asarray(self.bpb, dtype=float)


def _last_finite(rows: Sequence[Mapping[str, object]], key: str) -> Optional[float]:
    value = None
    for row in rows:
        candidate = row.get(key)
        if candidate is not None and math.isfinite(float(candidate)):
            value = float(candidate)
    return value


@dataclass
class CellRead:
    """One fan-out cell as it came off the wire, before anything decides to keep it."""

    run_id: str
    index: Optional[int]
    state: str
    summary_step: Optional[int]
    history_step: int
    seed: Optional[int]
    arms_consistent: Tuple[str, ...]
    has_lane_keys: Optional[bool]
    """
    Whether this cell logged the lane monitor: True, False, or None where the question could
    not be put -- a cell with no history has no testimony to give, and reading that silence as
    "no lanes" would convict every clobbered cell of an arm with lanes of not having run it.
    """

    config_is_readable: bool
    bpb: Dict[int, List[float]] = field(default_factory=dict)
    ce: Dict[int, List[float]] = field(default_factory=dict)
    declined: Optional[int] = None
    largest_trigger: Optional[float] = None
    declined_steps: List[int] = field(default_factory=list)
    train_steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    sources: Tuple[str, ...] = HELD_OUT_SOURCES

    @property
    def summary_was_clobbered(self) -> bool:
        """A summary that lost a record the history still holds."""
        return self.summary_step is None and self.history_step >= 0


def _read_cell(run, train_curve_samples: int) -> CellRead:
    """
    Pull one run into a :class:`CellRead`, taking every number from history and none from the
    summary.

    THE SUMMARY IS NOT A SOURCE OF DATA HERE, AND THAT IS A DELIBERATE COST. Reading the
    endpoint out of ``run.summary`` is one lookup and reading it out of ``scan_history`` is a
    paged query, so the cheap version is the tempting one -- and the cheap version is wrong on
    seven runs of this module. ``train_on_corpus.leave_the_reason_in_wandb`` calls ``wandb.init``
    with ``WANDB_RUN_ID`` still set, so a crash diagnostic is written *as* the cell instead of
    beside it, and the summary is replaced with ``step: None, runtime: 0.0``. One of the seven
    had trained 3.993 hours to step 4,910, further than any other cell in its submission, and
    every summary-based tool reported it as a cell that never started.

    The per-step history survives that, so history is what everything here is built on. The
    summary is still read, for exactly one purpose: disagreeing with the history is how a
    clobbered cell is *recognised*, and a recovery nobody is told about is not much better than
    a loss nobody is told about.

    :param run: A W&B API run.
    :param train_curve_samples: How many training-loss rows to keep, for the figure.

    :returns: The cell.
    """
    summary_step, history_step = steps_from_summary_and_history(run)
    config = run.config or {}
    readable = bool(config.get("model"))
    sources = sources_from_config(config) if readable else HELD_OUT_SOURCES
    match = CELL_SUFFIX.match(run.id)

    bpb_keys = [BPB_METRIC.format(source=s) for s in sources]
    ce_keys = [CE_METRIC.format(source=s) for s in sources]
    bpb: Dict[int, List[float]] = {}
    nats: Dict[int, List[float]] = {}
    for row in run.scan_history(keys=["_step", *bpb_keys, *ce_keys]):
        step = row.get("_step")
        values = [row.get(k) for k in bpb_keys]
        losses = [row.get(k) for k in ce_keys]
        if step is None or any(v is None for v in values) or any(v is None for v in losses):
            continue
        bpb[int(step)] = [float(v) for v in values]
        nats[int(step)] = [float(v) for v in losses]

    stability = list(run.scan_history(keys=["_step", SKIPPED_COUNT_METRIC, MAX_TRIGGER_METRIC]))
    count = _last_finite(stability, SKIPPED_COUNT_METRIC)
    declined_at = [
        int(row["_step"])
        for row in run.scan_history(keys=["_step", SKIP_TRIGGER_METRIC])
        if row.get("_step") is not None and row.get(SKIP_TRIGGER_METRIC) is not None
    ]

    curve = [
        row
        for row in run.scan_history(keys=["_step", TRAIN_LOSS_METRIC], page_size=10_000)
        if row.get("_step") is not None and row.get(TRAIN_LOSS_METRIC)
    ]
    stride = max(len(curve) // max(train_curve_samples, 1), 1)
    kept = curve[::stride]

    # READ OUT OF THE HISTORY AND NOT OUT OF THE SUMMARY, for the reason everything else here
    # is. A clobbered cell's summary holds a crash diagnostic and none of its own keys, so a
    # summary-based test would answer "no lanes" for every clobbered cell of a lane arm -- and
    # that answer is used below to decide whether such a cell ran the arm it says it did. The
    # witnesses are exact keys rather than globs, and `wandb_panels` requires both on every arm
    # with lanes and forbids them on one without, which is what makes their absence evidence.
    lane_rows = list(run.scan_history(keys=["_step", *LANE_WITNESS_METRICS]))
    saw_lane = any(row.get(key) is not None for row in lane_rows for key in LANE_WITNESS_METRICS)
    return CellRead(
        run_id=run.id,
        index=int(match.group("index")) if match else None,
        state=run.state,
        summary_step=summary_step,
        history_step=history_step,
        seed=(
            int(((config.get("data_loader") or {}).get("seed")) or 0)
            if (config.get("data_loader") is not None)
            else None
        ),
        arms_consistent=arms_consistent_with(config) if readable else (),
        has_lane_keys=saw_lane if (saw_lane or history_step >= 0) else None,
        config_is_readable=readable,
        bpb=bpb,
        ce=nats,
        declined=None if count is None else int(round(count)),
        largest_trigger=_last_finite(stability, MAX_TRIGGER_METRIC),
        declined_steps=sorted(declined_at),
        train_steps=[int(r["_step"]) for r in kept],
        train_loss=[float(r[TRAIN_LOSS_METRIC]) for r in kept],
        sources=sources,
    )


def funded_arms(consistent: Sequence[str]) -> Tuple[str, ...]:
    """
    Narrow ``arms_consistent_with``'s answer to the arms this tranche actually funded.

    WITHOUT THIS EVERY ``faithful`` CELL IS REFUSED, and it would have been refused on the first
    real read. ``decay-everything`` differs from ``faithful`` in the optimizer's weight-decay
    split and in nothing a model config records, so a genuine, correctly-configured ``faithful``
    cell comes back consistent with both and an equality test against ``('faithful',)`` fails on
    all five of them. ``stage_gate`` says so in as many words and leaves the tie unbroken on
    purpose -- a classifier that guesses is worse than one that reports what it cannot tell --
    which makes narrowing the caller's job rather than its own.

    The narrowing is sound here because ``decay-everything`` was never funded: it is not in
    ``ARM_ORDER``, no submission ran it, and so a cell consistent with ``faithful`` and with it
    is a ``faithful`` cell. What survives is the check that was wanted: a config consistent with
    two *funded* arms, or with none, still stops the read.

    :param consistent: What ``arms_consistent_with`` returned.

    :returns: Those of them the tranche funded, in table order.
    """
    return tuple(name for name in consistent if name in ARM_ORDER)


def assemble_arm(
    cells: Sequence[CellRead],
    arm: str,
    submission: str,
    expected_cells: int = SEEDS_PER_ARM,
) -> ArmSeries:
    """
    Turn the cells of one submission into an arm, keeping the ones that can be kept and naming
    the ones that cannot.

    THE TWO RECOVERIES ARE INFERENCES AND BOTH ARE CHECKED BEFORE THEY ARE USED. A cell whose
    summary was overwritten by its own crash report may also have lost its saved config, which
    takes its seed and its arm with it -- and those are the two things the analysis cannot
    proceed without, since the pairing is by seed and the contrast is between arms.

    *The seed comes back from the cell index*, but only after the index-equals-seed relation
    has been confirmed on every cell of the same submission whose config survived. These are
    ``--fanout-index-parameter seed`` submissions, so the relation should hold by construction;
    confirming it against the survivors is what turns "should" into evidence, and one
    disagreement anywhere refuses the recovery rather than averaging over it.

    *The arm comes back from its siblings*, again only when every cell whose config survived
    agrees on one arm, that arm is the one named on the command line, and the clobbered cell's
    own metric keys agree about whether it had lanes. A ``baseline`` cell cannot log ``hc/*``
    and a lane arm always does, so that last check is the cell's own testimony about the half
    of the question a sibling cannot answer for it.

    Anything that survives neither is excluded by name with a reason, and the exclusion travels
    with the arm into the report, the JSON and the figures.

    :param cells: Every cell read for this submission, crash reports already removed.
    :param arm: The arm the submission was supposed to run.
    :param submission: The platform run id, for the messages.
    :param expected_cells: The fan-out size.

    :returns: The arm.

    :raises Refusal: If nothing is left, if two kept cells share a seed, or if a cell whose
        config is readable says it ran a different arm -- which is a mislabelling rather than a
        loss, and is never recoverable.
    """
    series = ArmSeries(arm=arm, submission=submission)
    readable = [c for c in cells if c.config_is_readable]
    sibling_arms = {funded_arms(c.arms_consistent) for c in readable if c.arms_consistent}
    index_is_seed = bool(readable) and all(
        c.index is not None and c.seed == c.index for c in readable
    )

    for cell in sorted(cells, key=lambda c: (c.index if c.index is not None else 1 << 30)):
        name = cell.run_id
        if cell.config_is_readable and funded_arms(cell.arms_consistent) != (arm,):
            named = ", ".join(cell.arms_consistent) or "no arm in the table"
            raise Refusal(
                f"{name} was read as arm '{arm}' but its own saved model config is consistent "
                f"with {named}. Either the cell ran an arm nobody meant, or the --arm mapping "
                "on the command line is wrong. Both are worse than no analysis, because the "
                "contrast would be between arms the report names incorrectly. This is not a "
                "recoverable loss -- it is a disagreement -- so it is not excluded and "
                "absorbed, it stops the read."
            )

        seed = cell.seed
        if seed is None:
            if not index_is_seed or cell.index is None:
                series.excluded.append(
                    [
                        name,
                        "its saved config is gone, so its seed is unknown, and the "
                        "index-equals-seed relation could not be confirmed on the cells of "
                        "this submission whose configs survived -- so there is nothing to "
                        "recover it from that is not a guess",
                    ]
                )
                continue
            seed = cell.index
            series.recovered.append(
                f"{name}: seed {seed} recovered from the cell index, after confirming "
                f"index == seed on all {len(readable)} cell(s) of {submission} whose config "
                "survived"
            )

        if not cell.config_is_readable:
            wants_lanes = arm != "baseline"
            if len(sibling_arms) != 1 or sibling_arms != {(arm,)}:
                series.excluded.append(
                    [
                        name,
                        "its saved config is gone and its siblings do not agree on one arm, so "
                        "which arm it ran cannot be established from anything but the command "
                        "line, and a label from a command line is an assertion rather than "
                        "evidence",
                    ]
                )
                continue
            if cell.has_lane_keys is not None and cell.has_lane_keys != wants_lanes:
                series.excluded.append(
                    [
                        name,
                        f"its saved config is gone and its own metric history disagrees with "
                        f"'{arm}': it "
                        + ("logs" if cell.has_lane_keys else "logs no")
                        + f" {' or '.join(LANE_WITNESS_METRICS)}, and the lane monitor is "
                        "attached to an arm with lanes and to no other, so this cell did not "
                        "run the arm the submission says it did",
                    ]
                )
                continue
            confirmed = (
                "and confirmed against this cell's own lane monitor"
                if cell.has_lane_keys is not None
                else "though this cell logged no history of its own to confirm it against, so "
                "the arm rests on its siblings alone"
            )
            series.recovered.append(
                f"{name}: arm '{arm}' recovered from its siblings, which all carry that config, "
                + confirmed
            )

        if not cell.bpb:
            reached = (
                f"reached step {cell.history_step} but logged no held-out evaluation this read "
                "can find, so it has no endpoint to compare"
                if cell.history_step >= 0
                else "logged no history at all -- still queued, or died before its first step"
            )
            series.excluded.append([name, reached])
            continue

        series.seeds.append(int(seed))
        series.run_ids.append(name)
        series.states.append(cell.state)
        series.summary_steps.append(cell.summary_step)
        series.history_steps.append(cell.history_step)
        series.declined.append(cell.declined)
        series.largest_trigger.append(cell.largest_trigger)
        series.declined_steps.append(cell.declined_steps)
        series.train_curve_steps.append(cell.train_steps)
        series.train_curve_loss.append(cell.train_loss)
        series.sources = list(cell.sources)
        if cell.summary_was_clobbered:
            series.recovered.append(
                f"{name}: its summary reads step None while its history reaches "
                f"{cell.history_step}, so the summary was overwritten -- by its own crash "
                "report, on the evidence of the seven such records in this module. Every "
                "number for this cell is taken from history and none from the summary."
            )

    for index in range(expected_cells):
        if any(c.index == index for c in cells):
            continue
        series.excluded.append(
            [
                f"{submission}-cell-{index}",
                "no run with this cell index exists in the group at all, so the fan-out is "
                "short a replicate rather than short an evaluation",
            ]
        )

    if not series.run_ids:
        raise Refusal(
            f"no cell of {submission} carries a held-out evaluation, so arm '{arm}' has no "
            "data. This is an absent arm and not an empty one; there is nothing to average and "
            "nothing to substitute for it.\n"
            + "\n".join(f"  {run}: {why}" for run, why in series.excluded)
        )
    if len(set(series.seeds)) != len(series.seeds):
        raise Refusal(
            f"arm '{arm}' ({submission}) reports seeds {series.seeds}, which are not distinct, "
            "so at least two cells of the fan-out ran the same replicate. That is the failure "
            "resolve_seed exists to refuse: identical curves, a noise floor near zero, and "
            "every contrast against it significant."
        )

    kept = {c.run_id: c for c in cells}
    shared = set(kept[series.run_ids[0]].bpb)
    for run_id in series.run_ids[1:]:
        shared &= set(kept[run_id].bpb)
    if not shared:
        raise Refusal(
            f"the cells of arm '{arm}' ({submission}) share no evaluation step, so there is no "
            "step at which they can be compared."
        )

    order = list(np.argsort(series.seeds))
    series.steps = sorted(shared)
    for attribute in (
        "seeds",
        "run_ids",
        "states",
        "summary_steps",
        "history_steps",
        "declined",
        "largest_trigger",
        "declined_steps",
        "train_curve_steps",
        "train_curve_loss",
    ):
        setattr(series, attribute, [getattr(series, attribute)[i] for i in order])
    series.bpb = [[kept[run].bpb[st] for st in series.steps] for run in series.run_ids]
    series.ce = [[kept[run].ce[st] for st in series.steps] for run in series.run_ids]
    return series


def read_arm(
    entity: str,
    project: str,
    group: str,
    arm: str,
    submission: str,
    cells: int = SEEDS_PER_ARM,
    train_curve_samples: int = 1500,
) -> ArmSeries:
    """
    Pull one arm out of W&B, addressed by the submission that ran it.

    ADDRESSED BY SUBMISSION AND CHECKED AGAINST THE CELL'S OWN CONFIG, which are two different
    guards against the same failure and both are needed. The slug holds every attempt at this
    module -- probes, rehearsals, a cancelled L40S stage, an ``AdamW`` baseline with the same
    five seeds -- so a group-wide read of "the baseline" returns eight cells carrying seeds
    ``0, 0, 1, 1, 2, 3, 3, 4``. Naming the submission picks the pre-registered comparator.
    Reading ``arms_consistent_with`` off each cell's saved model config then checks that the
    cell ran what the submission said it would, which a name cannot: it is the run's own
    testimony, and it is what catches a fan-out cell that resolved to an arm nobody meant.

    THE SUBMISSION IS MATCHED AS A PREFIX AND THE CELL IS NEVER FETCHED BY ITS FULL ID. A
    platform run id is ``run_019fe40f-c71e-7045-9b58-537b9e2f6cb4`` and everybody, including
    the pre-registration, writes ``run_019fe40f-c71e``. Building ``<what was typed>-cell-<i>``
    and asking the API for it returns nothing for the short form, which reads exactly like a
    submission that has not started -- so the query is over the group and the filter is
    ``belongs_to_submission``, which is how ``noise_floor`` and ``tranche_watch`` already do it.

    ``noise_floor._arm_of`` is deliberately not reused for the classification. It reads the
    hyper-connection block coarsely and returns ``faithful`` for the ``mhc`` arm as well,
    because ``mhc`` differs from ``faithful`` in ``doubly_stochastic`` alone -- fine for a
    module that only had to separate three arms, and wrong for this one.

    :param entity: W&B entity.
    :param project: W&B project.
    :param group: The experiment slug.
    :param arm: Which arm this submission is supposed to have run.
    :param submission: The platform run id, or a unique prefix of one.
    :param cells: The fan-out size.
    :param train_curve_samples: How many training-loss rows to keep per cell for the figure.

    :returns: The arm.

    :raises Refusal: If nothing of the submission carries an evaluation, if a cell's config
        says it ran a different arm, or if two cells share a seed.
    """
    import wandb
    from noise_floor import belongs_to_submission

    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=100)[:200])

    found: List[CellRead] = []
    for run in runs:
        if not belongs_to_submission(run.id, submission):
            continue
        _, history_step = steps_from_summary_and_history(run)
        # Before anything else, because a crash report shares the group and the display-name
        # stem of the cell it is about, carries no model config, and would therefore be read as
        # a `baseline` cell with no evaluations -- an exclusion the report would then have to
        # explain, for a run that was never a replicate.
        if is_crash_report(run.id, run.job_type, history_step):
            continue
        found.append(_read_cell(run, train_curve_samples))

    if not found:
        raise Refusal(
            f"no run in group '{group}' has an id beginning '{submission}', so arm '{arm}' has "
            "nothing to read. Check the submission id: this is what a typo looks like and it "
            "is also what a submission that has not started looks like, and the two are worth "
            "telling apart before anything is concluded from either."
        )
    return assemble_arm(found, arm, submission, cells)


def check_the_nats_conversion(arms: Sequence[ArmSeries], tolerance: float = 5e-5) -> List[str]:
    """
    Check that ``CE = BPB * NATS_PER_BPB`` still holds on every cell of every arm.

    THE NATS COLUMN OF THIS REPORT IS A MULTIPLICATION AND NOT A MEASUREMENT, and that is only
    safe while ``bytes_per_token`` is one constant across the seven sources. The
    pre-registration records that it is, and records it as a reporting defect -- per-source BPB
    is a rigid rescaling of CE and carries no cross-source information. The moment somebody
    measures the seven constants and wires them in, that stops being true, every nats figure
    here becomes wrong by a per-source factor, and nothing would say so: the report would still
    print, still be internally consistent, and still be in the units the hypotheses are stated
    in. So it is checked against the CE the run logs beside every BPB, which costs one extra
    key per scan.

    :param arms: The arms as read.
    :param tolerance: Relative tolerance on the ratio.

    :returns: One sentence per arm that disagrees, empty when the conversion holds.
    """
    complaints = []
    for arm in arms:
        bpb = np.asarray(arm.bpb, dtype=float)
        nats = np.asarray(arm.ce, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bpb > 0, nats / np.maximum(bpb, 1e-12), np.nan)
        finite = ratio[np.isfinite(ratio)]
        if not finite.size:
            continue
        worst = float(np.max(np.abs(finite / NATS_PER_BPB - 1.0)))
        if worst > tolerance:
            complaints.append(
                f"arm '{arm.arm}': logged CE / logged BPB departs from NATS_PER_BPB "
                f"({NATS_PER_BPB:.6f}) by up to {worst:.2%}, ranging "
                f"{float(finite.min()):.6f} to {float(finite.max()):.6f}. Either bytes_per_token "
                "is no longer one constant across sources or the endpoint changed, and every "
                "figure this report gives in nats is wrong by a per-source factor."
            )
    return complaints


def completeness_refusals(
    arms: Sequence[ArmSeries],
    horizon: int = HORIZON,
    expected_seeds: int = SEEDS_PER_ARM,
) -> List[str]:
    """
    Everything about coverage that makes the tranche not yet analysable.

    Separated from the read so that ``--allow-provisional`` can downgrade exactly these and
    nothing else. A seed collision or a mislabelled arm is never downgradeable; a run that has
    not finished is, because looking at a mid-run tranche on purpose and knowing it is mid-run
    is a legitimate thing to do and pretending it is the endpoint is not.

    EVERY EXCLUSION IS A REASON HERE, WHICH IS WHAT KEEPS n - 1 FROM BEING FREE. Dropping a
    replicate shortens the df, narrows the interval and moves the mean, and it does none of it
    at random -- a cell loses its evaluations by hitting a wall or dying, so the survivors are
    the fast and the lucky. Listing the shortfall alone would not do: "4 of 5 cells" reads like
    a cell that is late, and the whole difference between a late cell and a lost one is the
    sentence beside it.

    A CELL'S ``state`` IS NOT THE TEST FOR WHETHER IT FINISHED, AND USING IT WOULD THROW AWAY
    THE RECOVERY. A cell whose summary was overwritten by its own crash report reads ``crashed``
    or ``failed`` however far it trained, so a state-based check discards exactly the cells this
    reader exists to rescue. What settles it is whether the cell reached the horizon, which the
    history holds; the state is worth saying out loud beside that, and is worth refusing over
    only when the two agree that the cell fell short.

    :param arms: The arms as read.
    :param horizon: The step count the tranche was submitted for.
    :param expected_seeds: Cells per arm.

    :returns: One sentence per problem, empty when the tranche stands.
    """
    problems = []
    for arm in arms:
        for run_id, why in arm.excluded:
            problems.append(f"arm '{arm.arm}' lost a replicate -- {run_id}: {why}")
        if len(arm.seeds) < expected_seeds:
            problems.append(
                f"arm '{arm.arm}' has {len(arm.seeds)} of {expected_seeds} cells, so its mean "
                f"is over {len(arm.seeds)} seeds and the pooled df is short by "
                f"{expected_seeds - len(arm.seeds)}"
            )
        short = [
            f"{rid} ({state}, history reaches step {reached})"
            for rid, state, reached in zip(arm.run_ids, arm.states, arm.history_steps)
            if state not in ("finished", "running") and reached < horizon
        ]
        if short:
            problems.append(
                f"arm '{arm.arm}' has cells that stopped before the horizon: {', '.join(short)}"
            )
        last = arm.steps[-1] if arm.steps else -1
        if last < horizon:
            problems.append(
                f"arm '{arm.arm}' shares its last evaluation at step {last} of {horizon}, so "
                "this is the endpoint of a partial run and not the endpoint"
            )
    shared_final = {arm.steps[-1] for arm in arms if arm.steps}
    if len(shared_final) > 1:
        problems.append(
            f"the arms end at different steps {sorted(shared_final)}, so a contrast between "
            "them would compare models that have seen different numbers of tokens"
        )
    return problems


def stability_refusals(arms: Sequence[ArmSeries]) -> List[str]:
    """
    Whether H7 has the data it needs, on every cell of every arm.

    AN ABSENT KEY IS NOT A COUNT OF ZERO, and this is the one place in the module where the
    difference is invisible in the arithmetic. ``stability/steps skipped`` is written on every
    optimizer step of every arm, so a cell that does not carry it is a cell whose optimizer
    could not skip or whose monitor was not attached -- and reading that as "declined no steps"
    would make an arm look maximally stable precisely when the instrument was missing.

    :param arms: The arms as read.

    :returns: One sentence per arm with missing stability data, empty when H7 can run.
    """
    problems = []
    for arm in arms:
        missing = [rid for rid, count in zip(arm.run_ids, arm.declined) if count is None]
        if missing:
            problems.append(
                f"arm '{arm.arm}' is missing '{SKIPPED_COUNT_METRIC}' on {', '.join(missing)}. "
                "That key is written on every optimizer step of every arm, so its absence is a "
                "missing instrument and not a count of zero, and H7 cannot be run on this arm."
            )
        no_trigger = [rid for rid, value in zip(arm.run_ids, arm.largest_trigger) if value is None]
        if no_trigger:
            problems.append(
                f"arm '{arm.arm}' is missing '{MAX_TRIGGER_METRIC}' on {', '.join(no_trigger)}, "
                "so H7's secondary statistic -- the one that separates a few noisy skips from "
                "the onset of a spike -- cannot be computed."
            )
    return problems


# ---------------------------------------------------------------------------------------
# The estimators the pre-registration names and noise_floor does not already carry.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BartlettResult:
    """Bartlett's test on the within-arm variances, and what the plan pre-committed to."""

    statistic: float
    df: int
    p_value: float
    arms: Tuple[str, ...]
    sigma: Tuple[float, ...]
    rejects: bool

    @property
    def spread(self) -> float:
        """Largest within-arm sigma over smallest, which is the thing the eye sees."""
        return max(self.sigma) / min(self.sigma) if min(self.sigma) > 0 else float("inf")


def bartlett(groups: Sequence[Sequence[float]], names: Sequence[str], alpha: float = 0.05):
    """
    Bartlett's test that the arms share one variance, which is what pooling rests on.

    THE PRE-REGISTRATION RUNS THIS AND SAYS IN ADVANCE THAT A PASS PROVES NOTHING. At five
    seeds per arm the test's power against a doubling of one arm's sigma is small, so it
    catches only gross heteroscedasticity; the pre-committed consequence is one-directional --
    if it *rejects*, the pooled sigma is abandoned and every contrast is re-run with Welch
    standard errors and Welch-Satterthwaite df, at a cost in power that is reported. If it does
    not reject, nothing changes and nothing is claimed. The per-arm standard deviations are
    printed either way, because the reader should see the quantity the test is too weak to
    adjudicate.

    Levene's is not used. With this few observations per group the median-centred residuals are
    degenerate and it rejects essentially never, at any true ratio.

    :param groups: One sequence of per-seed endpoints per arm.
    :param names: Arm names, in the same order.
    :param alpha: Significance level.

    :returns: The result.

    :raises ValueError: If any group has fewer than two observations, where it has no variance.
    """
    samples = [np.asarray(list(g), dtype=float) for g in groups]
    if any(s.size < 2 for s in samples):
        raise ValueError("Bartlett needs at least two observations in every group")

    k = len(samples)
    sizes = np.asarray([s.size for s in samples], dtype=float)
    variances = np.asarray([float(s.var(ddof=1)) for s in samples], dtype=float)
    total_df = float((sizes - 1).sum())
    pooled = float(((sizes - 1) * variances).sum() / total_df)

    numerator = total_df * math.log(pooled) - float(((sizes - 1) * np.log(variances)).sum())
    correction = 1.0 + (float((1.0 / (sizes - 1)).sum()) - 1.0 / total_df) / (3.0 * (k - 1))
    statistic = numerator / correction
    p_value = float(stats.chi2.sf(statistic, k - 1))
    return BartlettResult(
        statistic=float(statistic),
        df=k - 1,
        p_value=p_value,
        arms=tuple(names),
        sigma=tuple(float(math.sqrt(v)) for v in variances),
        rejects=bool(p_value < alpha),
    )


@dataclass(frozen=True)
class BlockFit:
    """A randomized complete block fit of arm means over shared seeds."""

    arms: Tuple[str, ...]
    seeds: Tuple[int, ...]
    arm_means: Tuple[float, ...]
    seed_means: Tuple[float, ...]
    grand_mean: float

    ms_error_paired: float
    """Residual mean square after removing arm and seed, on ``(k-1)(n-1)`` df."""

    df_paired: int

    ms_error_unpaired: float
    """Within-arm mean square, on ``k(n-1)`` df. This is the pooled sigma squared."""

    df_unpaired: int

    ms_seed: float
    seed_p_value: float
    """
    F test on the block. Not a hypothesis -- it is the diagnostic that says whether the seed
    effect the pairing removes was there at all, which is the same question rho-hat answers
    from the other side.
    """

    intraclass_rho: float
    """
    ``1 - MS_error_paired / MS_error_unpaired``, the fraction of within-arm variance the seed
    block accounts for. This is the design-level analogue of the pairwise rho-hat the
    pre-registration names, it is exactly the quantity the paired standard error depends on,
    and it is reported beside the pairwise figures rather than instead of them.
    """


def block_fit(matrix: np.ndarray, arms: Sequence[str], seeds: Sequence[int]) -> BlockFit:
    """
    Fit ``value ~ arm + seed`` over the tranche, which is what the primary analysis is.

    A PAIRED ANALYSIS OF ``k`` ARMS ACROSS ``n`` SHARED SEEDS IS THIS MODEL AND NOT ``k``
    SEPARATE PAIRED T TESTS. The difference is the error term: this pools the residual across
    all four arms and carries ``(k-1)(n-1) = 12`` df, where a standalone paired t on one
    contrast estimates its own ``sigma_delta`` from five differences and carries 4. The
    pre-registration takes the pooled version, on the same homoscedasticity assumption the
    unpaired pooling rests on, and Bartlett is what interrogates it. Both are reported.

    :param matrix: ``(n_arms, n_seeds)`` of the endpoint, in one unit.
    :param arms: Row labels.
    :param seeds: Column labels, shared across arms.

    :returns: The fit.

    :raises ValueError: If the matrix is not two-dimensional with at least two of each.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"a block fit needs at least 2 arms x 2 seeds, got {values.shape}")

    k, n = values.shape
    grand = float(values.mean())
    arm_means = values.mean(axis=1)
    seed_means = values.mean(axis=0)

    ss_arm = float(n * ((arm_means - grand) ** 2).sum())
    ss_seed = float(k * ((seed_means - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    ss_within = float(((values - arm_means[:, None]) ** 2).sum())
    ss_error = ss_total - ss_arm - ss_seed

    df_paired = (k - 1) * (n - 1)
    df_unpaired = k * (n - 1)
    ms_error_paired = ss_error / df_paired
    ms_error_unpaired = ss_within / df_unpaired
    ms_seed = ss_seed / (n - 1)
    seed_p = (
        float(stats.f.sf(ms_seed / ms_error_paired, n - 1, df_paired))
        if ms_error_paired > 0
        else float("nan")
    )

    return BlockFit(
        arms=tuple(arms),
        seeds=tuple(int(s) for s in seeds),
        arm_means=tuple(float(v) for v in arm_means),
        seed_means=tuple(float(v) for v in seed_means),
        grand_mean=grand,
        ms_error_paired=ms_error_paired,
        df_paired=df_paired,
        ms_error_unpaired=ms_error_unpaired,
        df_unpaired=df_unpaired,
        ms_seed=ms_seed,
        seed_p_value=seed_p,
        intraclass_rho=float(1.0 - ms_error_paired / ms_error_unpaired)
        if ms_error_unpaired > 0
        else float("nan"),
    )


def mde_from_se(se: float, df: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    The smallest detectable effect of a contrast whose standard error has already been formed.

    WHY NOT ``noise_floor.mde``, WHICH TAKES A SIGMA AND A CORRELATION. That function rebuilds
    the standard error from ``sigma sqrt(2(1 - rho)/n)``, which is right when rho is a design
    assumption being priced -- which is what the pre-registration's tables do -- and is an
    unnecessary round trip once a real contrast has a real standard error. Going through rho
    would also make the reported MDE inherit rho-hat's small-sample bias, which is towards zero
    and would therefore report a *larger* MDE than the design has. Conservative in the safe
    direction, but a number that moves for a reason unrelated to the design is a number nobody
    can check, and the blocked mean square already carries the pairing exactly.

    THE c4 CORRECTION IS APPLIED HERE AND NOWHERE ELSE ON THIS PATH. A standard error built
    from a sample standard deviation is biased low by the same ``c4(df)`` the standard deviation
    is, every MDE is linear in it, and the pre-registration commits to pricing the design off
    the corrected figure. It is deliberately *not* applied to the t statistic or to the interval
    beside it: that machinery is built on the distribution of ``s`` and already carries the
    bias, and correcting it twice is the error this note exists to prevent somebody re-making.

    :param se: The standard error of the contrast, uncorrected.
    :param df: Error degrees of freedom, of the variance behind ``se`` and of the test.
    :param alpha: Two-sided significance level.
    :param power: Target power.

    :returns: The minimum detectable effect, in the units of ``se``.
    """
    corrected = se / c4(df)
    approximate = corrected * (stats.norm.ppf(1.0 - alpha / 2.0) + stats.norm.ppf(power))
    upper = max(approximate * 4.0, corrected)
    while power_of(upper, corrected, df, alpha) < power:
        upper *= 2.0
    return float(
        optimize.brentq(
            lambda delta: power_of(delta, corrected, df, alpha) - power,
            1e-15,
            upper,
            xtol=1e-16,
        )
    )


def break_even_rho(
    n_arms: int,
    n_seeds: int,
    sigma: float = 0.010,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    The correlation above which blocking on seed is worth the degrees of freedom it costs.

    THE PRE-REGISTRATION QUOTES 0.09 AND THAT FIGURE IS FOR A THREE-ARM DESIGN. Pairing buys a
    factor ``sqrt(1 - rho)`` on the standard error and pays ``n - 1`` degrees of freedom out of
    the error term; the trade is a wash at whatever rho makes the two minimum detectable
    effects equal, which depends on the arm count through the df and on nothing else -- the
    result is independent of sigma, because both MDEs are linear in it.

    :param n_arms: Arms sharing the variance estimate.
    :param n_seeds: Seeds per arm.
    :param sigma: Any positive value; the answer does not depend on it.
    :param alpha: Two-sided significance level.
    :param power: Target power.

    :returns: The break-even correlation, in ``[0, 1)``.
    """
    unpaired = mde(sigma, n_seeds, n_arms, 0.0, False, alpha, power)

    def gap(rho: float) -> float:
        return mde(sigma, n_seeds, n_arms, rho, True, alpha, power) - unpaired

    if gap(0.0) <= 0.0:
        return 0.0
    return float(optimize.brentq(gap, 0.0, 0.999, xtol=1e-10))


@dataclass(frozen=True)
class Contrast:
    """One difference between two arm means, with everything the gate needs to be applied."""

    name: str
    treatment: str
    comparator: str
    analysis: str
    """``paired``, ``unpaired``, ``paired-standalone`` or ``welch``."""

    primary: bool
    endpoint: str
    """``unweighted`` or ``strata-weighted``."""

    delta_bpb: float
    delta_nats: float
    se_bpb: float
    df: int
    t_statistic: float
    p_value: float
    ci_bpb: Tuple[float, float]
    ci_nats: Tuple[float, float]

    gate_bpb: float
    """``2 x SE``, the pre-registered threshold for making a claim."""

    clears_gate: bool

    five_percent_bpb: float
    """``t_{0.975,df} x SE``. Larger than the gate, and quoted because the gate is not a 5% test."""

    clears_five_percent: bool

    mde_bpb: float
    mde_nats: float
    clears_mde: bool

    predicted_sign: int
    direction_as_predicted: bool

    excluded_reference_effects: Tuple[str, ...] = ()
    """Literature effects the interval rules out, phrased as bounds and never as equivalence."""


def _contrast(
    name: str,
    treatment: str,
    comparator: str,
    analysis: str,
    delta: float,
    se: float,
    df: int,
    predicted_sign: int,
    endpoint: str,
    primary: bool,
    references: Sequence[Tuple[str, float]] = (),
) -> Contrast:
    """Assemble one contrast from a difference, its standard error and its df."""
    t_statistic = delta / se if se > 0 else float("nan")
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df)) if se > 0 else float("nan")
    critical = float(stats.t.ppf(0.975, df))
    half = critical * se
    ci = (delta - half, delta + half)
    detectable = mde_from_se(se, df)
    excluded = tuple(
        f"{label} ({effect:+.3f} nats)"
        for label, effect in references
        if not (ci[0] * NATS_PER_BPB <= effect <= ci[1] * NATS_PER_BPB)
    )
    return Contrast(
        name=name,
        treatment=treatment,
        comparator=comparator,
        analysis=analysis,
        primary=primary,
        endpoint=endpoint,
        delta_bpb=delta,
        delta_nats=delta * NATS_PER_BPB,
        se_bpb=se,
        df=df,
        t_statistic=t_statistic,
        p_value=p_value,
        ci_bpb=ci,
        ci_nats=(ci[0] * NATS_PER_BPB, ci[1] * NATS_PER_BPB),
        gate_bpb=GATE_SIGMAS * se,
        clears_gate=bool(abs(delta) >= GATE_SIGMAS * se),
        five_percent_bpb=half,
        clears_five_percent=bool(abs(delta) >= half),
        mde_bpb=detectable,
        mde_nats=detectable * NATS_PER_BPB,
        clears_mde=bool(abs(delta) >= detectable),
        predicted_sign=predicted_sign,
        direction_as_predicted=bool(np.sign(delta) == predicted_sign),
        excluded_reference_effects=excluded,
    )


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """
    Welch's unpooled difference of means, which is what Bartlett rejecting commits the plan to.

    :param a: Treatment endpoints, one per seed.
    :param b: Comparator endpoints, one per seed.

    :returns: ``(delta, standard error, Welch-Satterthwaite df)``.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    vx, vy = float(x.var(ddof=1)) / x.size, float(y.var(ddof=1)) / y.size
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (x.size - 1) + vy**2 / (y.size - 1))
    return float(x.mean() - y.mean()), se, float(df)


# ---------------------------------------------------------------------------------------
# H7: the stability outcome.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationTest:
    """An exact two-sided permutation test over two small groups."""

    statistic_name: str
    treatment: str
    comparator: str
    treatment_values: Tuple[float, ...]
    comparator_values: Tuple[float, ...]
    observed: float
    """Difference of means, treatment minus comparator."""

    p_value: float
    n_permutations: int
    smallest_attainable_p: float
    """
    ``2 / C(n_a + n_b, n_a)``. At 5 v 5 this is 0.0079, so complete separation between the arms
    is detectable at alpha = 0.05 and partial separation mostly is not. Printed with every
    result because a p of 0.09 from this test means "the arms did not separate completely" and
    not "the arms are similar".
    """

    complete_separation: bool
    """Whether every treatment value exceeds every comparator value, or the reverse."""


def exact_permutation_test(
    treatment: Sequence[float],
    comparator: Sequence[float],
    statistic_name: str,
    treatment_name: str,
    comparator_name: str,
) -> PermutationTest:
    """
    The pre-registered test for H7: exact, two-sided, over every split of the pooled runs.

    EXACT AND NOT SAMPLED, because at 5 + 5 there are 252 splits and enumerating them is
    instantaneous, so a Monte Carlo p-value would add noise to a number that has none. No
    Poisson or negative-binomial parameterisation is assumed and none is fitted: declined steps
    cluster within a run -- a spike declines a run of consecutive updates -- so 6,000 steps is
    not 6,000 independent draws and any count model would be fitting a dispersion nobody
    measured.

    WHAT IT CANNOT DO, STATED WITH THE RESULT RATHER THAN IN A FOOTNOTE. The smallest attainable
    two-sided p is ``2 / C(10, 5) = 0.0079``, reached only when the two arms separate
    completely. One value out of order takes the smallest attainable p to 0.0159, two to 0.0317,
    and by the time the groups interleave at all the test cannot reach 0.05. So a null here is
    "these five did not separate cleanly from those five", which is a weaker statement than it
    looks and is the reason H7 is secondary.

    :param treatment: Per-run statistic on the treatment arm.
    :param comparator: Per-run statistic on the comparator arm.
    :param statistic_name: What is being permuted, for the report.
    :param treatment_name: Arm name.
    :param comparator_name: Arm name.

    :returns: The test.

    :raises ValueError: If either group is empty, where there is nothing to permute.
    """
    a = [float(v) for v in treatment]
    b = [float(v) for v in comparator]
    if not a or not b:
        raise ValueError("a permutation test needs at least one observation in each group")

    pooled = a + b
    n_a = len(a)
    observed = float(np.mean(a) - np.mean(b))

    extreme = 0
    total = 0
    for chosen in itertools.combinations(range(len(pooled)), n_a):
        picked = [pooled[i] for i in chosen]
        rest = [pooled[i] for i in range(len(pooled)) if i not in set(chosen)]
        statistic = float(np.mean(picked) - np.mean(rest))
        total += 1
        # A tolerance, because two splits that are the same difference in exact arithmetic can
        # differ in the last bit after two means, and a strict `>=` would then count one of
        # them and not the other -- which shows up as a p-value that is not a multiple of
        # 1/total and is impossible for an exact test.
        if abs(statistic) >= abs(observed) - 1e-12:
            extreme += 1

    return PermutationTest(
        statistic_name=statistic_name,
        treatment=treatment_name,
        comparator=comparator_name,
        treatment_values=tuple(a),
        comparator_values=tuple(b),
        observed=observed,
        p_value=extreme / total,
        n_permutations=total,
        smallest_attainable_p=2.0 / total,
        complete_separation=bool(min(a) > max(b) or max(a) < min(b)),
    )


# ---------------------------------------------------------------------------------------
# The analysis.
# ---------------------------------------------------------------------------------------


def frozen_weights(path: str, sources: Sequence[str]) -> Optional[np.ndarray]:
    """
    The strata weights frozen before any treatment arm existed, in this report's source order.

    :param path: The frozen artifact.
    :param sources: The source order the endpoint matrices are in.

    :returns: The weight vector, or ``None`` when the artifact is absent.

    :raises Refusal: If the artifact names different sources from the runs, which would mean
        the frozen weighting was measured on a different endpoint from the one being weighted.
    """
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        artifact = json.load(handle)
    block = artifact.get("weights") or {}
    named = list(block.get("sources") or [])
    weights = list(block.get("weights") or [])
    if sorted(named) != sorted(sources):
        raise Refusal(
            f"{path} froze weights over sources {named}, and the runs evaluated on "
            f"{list(sources)}. A weight vector measured on one set of sources cannot be applied "
            "to another, and reindexing it silently is how a committed number stops meaning "
            "what it was committed as."
        )
    index = {name: i for i, name in enumerate(named)}
    return np.asarray([weights[index[s]] for s in sources], dtype=float)


def analyse(
    arms: Sequence[ArmSeries],
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    provisional: Sequence[str] = (),
    label: str = "measured",
) -> Dict[str, object]:
    """
    Everything the pre-registration asks for, over the arms as read.

    :param arms: The arms, baseline first. Every arm must share the same seeds.
    :param weights: The frozen strata weights, or None to omit the secondary endpoint.
    :param alpha: Two-sided significance level for Bartlett and for the reported 5% line.
    :param provisional: Reasons this reading is not final, stamped on the artifact.
    :param label: ``measured`` or ``synthetic``. Carried into every artifact this writes.

    :returns: A JSON-serializable dict of the whole analysis.

    :raises Refusal: If the arms do not share one set of seeds, which is what the pairing
        rests on and what a fan-out that resolved differently across stages would break.
    """
    by_name = {arm.arm: arm for arm in arms}
    present = [name for name in ARM_ORDER if name in by_name] + [
        arm.arm for arm in arms if arm.arm not in ARM_ORDER
    ]
    ordered = [by_name[name] for name in present]

    seed_sets = {tuple(arm.seeds) for arm in ordered}
    if len(seed_sets) != 1:
        raise Refusal(
            "the arms do not share one set of seeds: "
            + "; ".join(f"{arm.arm} ran {arm.seeds}" for arm in ordered)
            + ". The paired analysis pairs arm a seed k with arm b seed k, and there is no "
            "pairing to make."
        )
    seeds = list(seed_sets.pop())
    n_seeds = len(seeds)
    n_arms = len(ordered)

    endpoints = {arm.arm: arm.endpoint_matrix() for arm in ordered}
    unweighted = {name: matrix.mean(axis=1) for name, matrix in endpoints.items()}
    weighted = (
        {name: matrix @ weights for name, matrix in endpoints.items()}
        if weights is not None
        else {}
    )

    result: Dict[str, object] = {
        "label": label,
        "generated": date.today().isoformat(),
        "provisional": list(provisional),
        "arms": [
            {
                "arm": arm.arm,
                "submission": arm.submission,
                "seeds": arm.seeds,
                "run_ids": arm.run_ids,
                "states": arm.states,
                "final_step": arm.steps[-1] if arm.steps else None,
                "sources": arm.sources,
                "endpoint_bpb": [float(v) for v in unweighted[arm.arm]],
                "endpoint_nats": [float(v) * NATS_PER_BPB for v in unweighted[arm.arm]],
                "mean_bpb": float(unweighted[arm.arm].mean()),
                "mean_nats": float(unweighted[arm.arm].mean()) * NATS_PER_BPB,
                "sd_bpb": float(unweighted[arm.arm].std(ddof=1)),
                "per_source_mean_bpb": {
                    source: float(endpoints[arm.arm][:, j].mean())
                    for j, source in enumerate(arm.sources)
                },
                "per_source_sd_bpb": {
                    source: float(endpoints[arm.arm][:, j].std(ddof=1))
                    for j, source in enumerate(arm.sources)
                },
                "declined_steps": arm.declined,
                "largest_trigger": arm.largest_trigger,
                "excluded": arm.excluded,
                "recovered": arm.recovered,
                "summary_steps": arm.summary_steps,
                "history_steps": arm.history_steps,
            }
            for arm in ordered
        ],
        "excluded": [[arm.arm, run_id, why] for arm in ordered for run_id, why in arm.excluded],
        "recovered": [[arm.arm, note] for arm in ordered for note in arm.recovered],
        "post_hoc": [{"date": when, "what": what} for when, what in POST_HOC],
    }

    # (a) the pooled noise floor, over every arm, and the assumption it rests on.
    #
    # ONE ARM IS A LEGITIMATE READ AND NOT A DEGENERATE ONE. Before the treatment stages land
    # the only thing there is to read is the comparator, and reading it is how this module is
    # checked against the frozen artifact -- which was published before this file existed, so
    # reproducing it is a test of the reader against a number nobody can have tuned it to. What
    # a single arm cannot support is a contrast, a Bartlett test or a pairing, and each of those
    # is skipped by name below rather than allowed to return something.
    pooled = pooled_sigma([unweighted[name] for name in present])
    bartlett_result = (
        bartlett([unweighted[name] for name in present], present, alpha) if n_arms >= 2 else None
    )
    result["sigma"] = {
        "endpoint": "unweighted mean of held-out BPB over the seven sources",
        "sigma_bpb": pooled.sigma,
        "sigma_bpb_unbiased": pooled.sigma_unbiased,
        "sigma_nats": pooled.sigma * NATS_PER_BPB,
        "sigma_nats_unbiased": pooled.sigma_unbiased * NATS_PER_BPB,
        "df": pooled.df,
        "ci_bpb": [pooled.ci_low, pooled.ci_high],
        "ci_nats": [pooled.ci_low * NATS_PER_BPB, pooled.ci_high * NATS_PER_BPB],
        "span": pooled.span,
        "c4": c4(pooled.df),
        "per_arm_sd_bpb": {name: float(unweighted[name].std(ddof=1)) for name in present},
        # ``spread`` is a property and ``asdict`` does not carry properties, so it is added
        # here rather than left to be recomputed by whoever reads the artifact.
        "bartlett": (
            {**asdict(bartlett_result), "spread": bartlett_result.spread}
            if bartlett_result is not None
            else None
        ),
    }

    if n_arms < 2:
        result["single_arm"] = (
            f"only '{present[0]}' landed, so there is no contrast, no pairing, no Bartlett test "
            "and no H7. sigma-hat above is this arm's own, on df = "
            f"{pooled.df}, and it is the same quantity noise_floor.py freezes."
        )
        result["contrasts"] = []
        result["per_source"] = []
        result["pairing"] = None
        result["h7"] = {
            "statement": "H7 is stated against the baseline and needs a treatment arm.",
            "limit": "",
            "tests": [],
        }
        return result

    # (b) the block fit, rho-hat, and the paired-or-unpaired decision.
    matrix = np.asarray([unweighted[name] for name in present], dtype=float)
    fit = block_fit(matrix, present, seeds)
    pairwise = {}
    for hypothesis in HYPOTHESES:
        if hypothesis.treatment in by_name and hypothesis.comparator in by_name:
            estimate = paired_correlation(
                unweighted[hypothesis.treatment], unweighted[hypothesis.comparator]
            )
            pairwise[hypothesis.name] = asdict(estimate)

    recomputed = break_even_rho(n_arms, n_seeds)
    result["pairing"] = {
        "block_fit": asdict(fit),
        "rho_by_hypothesis": pairwise,
        "intraclass_rho": fit.intraclass_rho,
        "pre_registered_break_even": PRE_REGISTERED_BREAK_EVEN_RHO,
        "recomputed_break_even": recomputed,
        "recomputed_break_even_is_post_hoc": True,
    }

    # (c) the contrasts. The MDE of each is taken from that contrast's own standard error by
    # `mde_from_se`, which applies c4 there, so there is no pooled sigma to carry down here.
    contrasts: List[Dict[str, object]] = []
    dose_checks: List[dose_adjustment.DoseCheck] = []
    for hypothesis in HYPOTHESES:
        if hypothesis.treatment not in by_name or hypothesis.comparator not in by_name:
            contrasts.append(
                {
                    "name": hypothesis.name,
                    "claim": hypothesis.claim,
                    "status": "not analysable: "
                    + ", ".join(
                        arm
                        for arm in (hypothesis.treatment, hypothesis.comparator)
                        if arm not in by_name
                    )
                    + " did not land",
                }
            )
            continue

        rho = float(pairwise[hypothesis.name]["rho_pearson"])
        paired_primary = rho >= PRE_REGISTERED_BREAK_EVEN_RHO
        rows: List[Contrast] = []

        for endpoint_name, table in (
            ("unweighted", unweighted),
            *((("strata-weighted", weighted),) if weighted else ()),
        ):
            a = np.asarray(table[hypothesis.treatment], dtype=float)
            b = np.asarray(table[hypothesis.comparator], dtype=float)
            sub_fit = block_fit(
                np.asarray([table[name] for name in present], dtype=float), present, seeds
            )
            delta = float(a.mean() - b.mean())
            welch_delta, welch_se, welch_df = welch(a, b)

            for analysis, value, se, df in (
                (
                    "paired",
                    delta,
                    math.sqrt(2.0 * sub_fit.ms_error_paired / n_seeds),
                    sub_fit.df_paired,
                ),
                (
                    "unpaired",
                    delta,
                    math.sqrt(2.0 * sub_fit.ms_error_unpaired / n_seeds),
                    sub_fit.df_unpaired,
                ),
                (
                    "paired-standalone",
                    delta,
                    float((a - b).std(ddof=1)) / math.sqrt(n_seeds),
                    n_seeds - 1,
                ),
                ("welch", welch_delta, welch_se, max(int(round(welch_df)), 1)),
            ):
                rows.append(
                    _contrast(
                        name=hypothesis.name,
                        treatment=hypothesis.treatment,
                        comparator=hypothesis.comparator,
                        analysis=analysis,
                        delta=value,
                        se=se,
                        df=df,
                        predicted_sign=hypothesis.predicted_sign,
                        endpoint=endpoint_name,
                        primary=endpoint_name == "unweighted"
                        and analysis == ("paired" if paired_primary else "unpaired"),
                        references=hypothesis.reference_effects,
                    )
                )

        if bartlett_result.rejects:
            for row in rows:
                object.__setattr__(
                    row, "primary", row.analysis == "welch" and row.endpoint == "unweighted"
                )

        # The training-dose check, pre-registered 2026-08-10 and applied to the primary row.
        # It reads the declined counts this module already carries for H7, so it costs no
        # further measurement, and it can only withhold a claim -- see `dose_adjustment`.
        primary_row = next(row for row in rows if row.primary)
        try:
            check = dose_adjustment.dose_check(
                name=hypothesis.name,
                treatment=hypothesis.treatment,
                comparator=hypothesis.comparator,
                delta_nats=primary_row.delta_nats,
                gate_nats=primary_row.gate_bpb * NATS_PER_BPB,
                declined_treatment=by_name[hypothesis.treatment].declined,
                declined_comparator=by_name[hypothesis.comparator].declined,
                predicted_sign=hypothesis.predicted_sign,
            )
            dose_result: Optional[Dict[str, object]] = asdict(check)
            dose_checks.append(check)
        except ValueError as error:
            # A missing declined count is missing data. `completeness_refusals` has already
            # named it; what this must not do is report a dose difference of zero for it.
            dose_result = {"name": hypothesis.name, "unavailable": str(error)}

        contrasts.append(
            {
                "name": hypothesis.name,
                "claim": hypothesis.claim,
                "treatment": hypothesis.treatment,
                "comparator": hypothesis.comparator,
                "rho_pearson": rho,
                "paired_is_primary": paired_primary and not bartlett_result.rejects,
                "bartlett_forced_welch": bartlett_result.rejects,
                "rows": [asdict(row) for row in rows],
                "dose": dose_result,
            }
        )
    result["contrasts"] = contrasts
    result["dose"] = {
        "pre_registered": "2026-08-10, before any treatment endpoint was visible",
        "nats_per_declined_step": dose_adjustment.PER_DECLINE_NATS,
        "nats_per_declined_step_ci": [
            dose_adjustment.PER_DECLINE_NATS_LOW,
            dose_adjustment.PER_DECLINE_NATS_HIGH,
        ],
        "rendered": dose_adjustment.render(dose_checks),
    }

    # (d) per source, so an effect concentrated in one source is visible.
    per_source: List[Dict[str, object]] = []
    for hypothesis in HYPOTHESES:
        if hypothesis.treatment not in by_name or hypothesis.comparator not in by_name:
            continue
        sources = by_name[hypothesis.treatment].sources
        rows = []
        for j, source in enumerate(sources):
            a = endpoints[hypothesis.treatment][:, j]
            b = endpoints[hypothesis.comparator][:, j]
            column = np.asarray([endpoints[name][:, j] for name in present], dtype=float)
            sub_fit = block_fit(column, present, seeds)
            se = math.sqrt(2.0 * sub_fit.ms_error_paired / n_seeds)
            delta = float(a.mean() - b.mean())
            rows.append(
                {
                    "source": source,
                    "delta_bpb": delta,
                    "delta_nats": delta * NATS_PER_BPB,
                    "se_bpb": se,
                    "df": sub_fit.df_paired,
                    "p_value": float(2.0 * stats.t.sf(abs(delta / se), sub_fit.df_paired))
                    if se > 0
                    else float("nan"),
                    "clears_gate": bool(abs(delta) >= GATE_SIGMAS * se),
                }
            )
        per_source.append({"name": hypothesis.name, "rows": rows})
    result["per_source"] = per_source

    # (e) H7.
    result["h7"] = h7(ordered)
    return result


def h7(arms: Sequence[ArmSeries]) -> Dict[str, object]:
    """
    The stability outcome, on the counts and on the trigger magnitudes.

    :param arms: The arms, including the baseline, which is the comparator.

    :returns: The tests, or a refusal note when the data are not there.

    :raises Refusal: If the baseline is absent, which is the comparator for every test here.
    """
    by_name = {arm.arm: arm for arm in arms}
    if "baseline" not in by_name:
        raise Refusal("H7 is stated against the baseline and the baseline did not land.")
    base = by_name["baseline"]

    tests: List[Dict[str, object]] = []
    for arm in arms:
        if arm.arm == "baseline":
            continue
        if any(v is None for v in arm.declined + base.declined):
            tests.append(
                {
                    "arm": arm.arm,
                    "status": "refused: the declined-step count is missing on at least one "
                    "cell, and an absent key is not a count of zero",
                }
            )
            continue
        counts = exact_permutation_test(
            [float(v) for v in arm.declined if v is not None],
            [float(v) for v in base.declined if v is not None],
            "declined optimizer steps of 6,000",
            arm.arm,
            "baseline",
        )
        entry: Dict[str, object] = {
            "arm": arm.arm,
            "pre_registered": arm.arm == "faithful",
            "primary": asdict(counts),
        }
        if not any(v is None for v in arm.largest_trigger + base.largest_trigger):
            entry["secondary"] = asdict(
                exact_permutation_test(
                    [float(v) for v in arm.largest_trigger if v is not None],
                    [float(v) for v in base.largest_trigger if v is not None],
                    "largest triggering gradient norm",
                    arm.arm,
                    "baseline",
                )
            )
        tests.append(entry)

    return {
        "statement": "H7. faithful declines more updates than baseline, and at larger "
        "triggering gradient norms. Secondary, added after stage 1 rather than before it, and "
        "no primary conclusion is conditioned on it.",
        "limit": "At 5 v 5 the smallest attainable two-sided permutation p is 2/C(10,5) = "
        "0.0079, so complete separation between the arms is detectable at alpha = 0.05 and "
        "partial separation mostly is not.",
        "tests": tests,
    }


# ---------------------------------------------------------------------------------------
# Synthetic tranches. Reachable only from --demo and --self-test.
# ---------------------------------------------------------------------------------------


def synthetic_tranche(
    effects_nats: Mapping[str, float],
    sigma_bpb: float = 0.00061,
    per_source_sigma_bpb: float = 0.00068,
    rho: float = 0.5,
    n_seeds: int = SEEDS_PER_ARM,
    declined: Optional[Mapping[str, Sequence[int]]] = None,
    triggers: Optional[Mapping[str, Sequence[float]]] = None,
    rng_seed: int = 0,
    steps: Sequence[int] = tuple(range(0, HORIZON + 1, 500)),
    sources: Sequence[str] = HELD_OUT_SOURCES,
) -> List[ArmSeries]:
    """
    A four-arm tranche with a planted effect, a planted correlation and a planted noise floor.

    THIS IS THE ONLY GROUND TRUTH THIS MODULE WILL EVER HAVE. The real tranche has one and
    nobody knows it, so every estimator is exercised against a design where the answer was
    written down first: ``test_analysis.py`` plants an effect, runs the whole pipeline, and
    asserts that the recovered difference, the recovered noise floor, the recovered correlation
    and the false-positive rate under a planted null are what they were planted as.

    THE NOISE IS BUILT COMMON-MODE ACROSS SOURCES BECAUSE THE REAL THING IS, and getting this
    wrong is not cosmetic. Seven independent sources averaged together would put the endpoint's
    sigma at ``per-source sigma / sqrt(7)``, and the measured tranche is nowhere near that:
    the mean per-source sigma is 0.00068 BPB and the composite over all seven is 0.00061, where
    independence would have given 0.00026. What moves the sources together is a whole-run event,
    and a generator without that term would test the estimators against a design 2.3 times
    quieter than the one they will meet. So the per-run noise is split into a term common to all
    seven sources and a term private to each, solved so that the endpoint's standard deviation
    is exactly ``sigma_bpb`` and each source's is exactly ``per_source_sigma_bpb``.

    :param effects_nats: Arm name -> the true difference from the baseline, in nats. Negative
        is an improvement.
    :param sigma_bpb: True standard deviation of the *endpoint*, the unweighted mean over
        sources, which is the quantity the noise floor measures.
    :param per_source_sigma_bpb: True standard deviation of one source. Must be at least
        ``sigma_bpb``; equal makes the sources move in perfect lockstep.
    :param rho: True within-seed correlation across arms.
    :param n_seeds: Seeds per arm.
    :param declined: Arm name -> per-seed declined-step counts. Defaults to the comparator's own
        measured counts on every arm, which is the null H7 is tested against.
    :param triggers: Arm name -> per-seed largest triggering gradient norm.
    :param rng_seed: Reproducibility.
    :param steps: Evaluation checkpoints.
    :param sources: Held-out sources.

    :returns: One :class:`ArmSeries` per arm, marked so nothing downstream can mistake them.

    :raises ValueError: If a source is asked to be quieter than the mean of seven of it, which
        no combination of a common and a private term can produce.
    """
    n_sources = len(sources)
    if per_source_sigma_bpb < sigma_bpb - 1e-15:
        raise ValueError(
            f"a per-source sigma of {per_source_sigma_bpb} below the endpoint's {sigma_bpb} "
            "would need the sources to be negatively correlated, which is not the structure "
            "this generator is for"
        )
    private_var = max(
        (n_sources / (n_sources - 1.0)) * (per_source_sigma_bpb**2 - sigma_bpb**2), 0.0
    )
    common_var = max(sigma_bpb**2 - private_var / n_sources, 0.0)
    sigma_private, sigma_common = math.sqrt(private_var), math.sqrt(common_var)

    rng = np.random.default_rng(rng_seed)
    names = [name for name in ARM_ORDER if name == "baseline" or name in effects_nats]
    shared_common = rng.standard_normal((n_seeds, len(steps)))
    shared_private = rng.standard_normal((n_seeds, len(steps), n_sources))
    level = np.linspace(1.30, 0.68, len(steps))[None, :, None]

    arms: List[ArmSeries] = []
    for name in names:
        effect = 0.0 if name == "baseline" else effects_nats[name] / NATS_PER_BPB
        own_common = rng.standard_normal((n_seeds, len(steps)))
        own_private = rng.standard_normal((n_seeds, len(steps), n_sources))
        root, complement = math.sqrt(rho), math.sqrt(1.0 - rho)
        common = sigma_common * (root * shared_common + complement * own_common)
        private = sigma_private * (root * shared_private + complement * own_private)
        spread = np.linspace(-0.35, 0.35, n_sources)[None, None, :]
        values = level + spread + common[:, :, None] + private + effect

        counts = list((declined or {}).get(name, (19, 10, 16, 18, 20)))
        peaks = list((triggers or {}).get(name, (0.712, 0.485, 0.387, 0.465, 0.420)))
        arms.append(
            ArmSeries(
                arm=name,
                submission=f"SYNTHETIC-{name}",
                seeds=list(range(n_seeds)),
                run_ids=[f"SYNTHETIC-{name}-cell-{i}" for i in range(n_seeds)],
                states=["finished"] * n_seeds,
                # Filled rather than left empty so that the coverage checks, which walk these
                # beside `run_ids`, see a cell per cell. A `zip` over a shorter list truncates
                # in silence, and a check that inspects nothing passes.
                summary_steps=[int(steps[-1])] * n_seeds,
                history_steps=[int(steps[-1])] * n_seeds,
                steps=list(steps),
                bpb=values.tolist(),
                ce=(values * NATS_PER_BPB).tolist(),
                sources=list(sources),
                declined=counts[:n_seeds],
                largest_trigger=peaks[:n_seeds],
                declined_steps=[[] for _ in range(n_seeds)],
                train_curve_steps=[list(steps) for _ in range(n_seeds)],
                train_curve_loss=[
                    (values[i].mean(axis=1) * NATS_PER_BPB).tolist() for i in range(n_seeds)
                ],
            )
        )
    return arms


# ---------------------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------------------


def _banner(label: str, provisional: Sequence[str], where: str = "BELOW") -> List[str]:
    lines = []
    if label != "measured":
        lines += [
            "=" * 92,
            f"  {label.upper()}.  NOTHING {where} IS A MEASUREMENT OF ANYTHING.",
            "  Every number is this file's own generator read back out, with a planted effect,",
            "  a planted correlation and a planted noise floor. It exists so the report and the",
            "  figures can be reviewed before the data lands. Pass --group and --arm to read W&B.",
            "=" * 92,
            "",
        ]
    if provisional:
        lines += ["PROVISIONAL. This is not the tranche's result, because:"]
        lines += [f"  - {reason}" for reason in provisional]
        lines += [""]
    return lines


def render(result: Mapping[str, object]) -> str:
    """
    The whole analysis as text, in the order the pre-registration argues it.

    :param result: What :func:`analyse` returned.

    :returns: The report.
    """
    label = str(result.get("label", "measured"))
    mark = "" if label == "measured" else f"[{label}] "
    out: List[str] = _banner(label, list(result.get("provisional") or []))

    out += [
        f"{mark}HYPER-CONNECTIONS AT 370M -- the pre-registered analysis",
        f"{mark}generated {result.get('generated')}, endpoint: unweighted mean of held-out "
        "bits-per-byte over seven sources at the final step",
        "",
        f"{mark}ARMS",
    ]
    for arm in result["arms"]:  # type: ignore[index]
        out.append(
            f"{mark}  {arm['arm']:<12} {arm['submission']:<22} seeds {arm['seeds']}  "
            f"step {arm['final_step']}  mean {arm['mean_bpb']:.5f} BPB "
            f"({arm['mean_nats']:.4f} nats)  sd {arm['sd_bpb']:.5f}"
        )
        out.append(
            f"{mark}  {'':<12} per seed: " + ", ".join(f"{v:.5f}" for v in arm["endpoint_bpb"])
        )
    out.append("")

    # SAID HERE, HIGH UP AND UNABBREVIATED, BECAUSE A MISSING REPLICATE CHANGES EVERY NUMBER
    # UNDERNEATH IT AND LOOKS LIKE NONE OF THEM. n falls, the df falls with it, the interval
    # narrows, and the arm mean moves in a direction nobody chose -- cells lose their
    # evaluations by hitting a wall or dying, so the ones that go are the slow and the unlucky.
    excluded = list(result.get("excluded") or [])
    if excluded:
        out += [f"{mark}CELLS READ AND LEFT OUT ({len(excluded)}). Every number below is short "]
        out += [f"{mark}of these replicates, and they did not go missing at random:"]
        for arm_name, run_id, why in excluded:
            out.append(f"{mark}  [{arm_name}] {run_id}")
            out.append(f"{mark}      {why}")
        out.append("")
    recovered = list(result.get("recovered") or [])
    if recovered:
        out += [
            f"{mark}RECORDS RECONSTRUCTED ({len(recovered)}). These cells are in the analysis, "
            "and what",
            f"{mark}they contribute was inferred rather than read directly:",
        ]
        for arm_name, note in recovered:
            out.append(f"{mark}  [{arm_name}] {note}")
        out.append("")

    sigma = result["sigma"]  # type: ignore[index]
    out += [
        f"{mark}(a) THE NOISE FLOOR, POOLED ACROSS EVERY ARM",
        f"{mark}  sigma-hat {sigma['sigma_bpb']:.5f} BPB = {sigma['sigma_nats']:.5f} nats, "
        f"df = {sigma['df']}",
        f"{mark}  95% interval [{sigma['ci_nats'][0]:.5f}, {sigma['ci_nats'][1]:.5f}] nats, "
        f"a factor of {sigma['span']:.1f} end to end",
        f"{mark}  c4({sigma['df']}) = {sigma['c4']:.4f}, so the bias-corrected point estimate is "
        f"{sigma['sigma_nats_unbiased']:.5f} nats, and every MDE below is taken from that one",
        f"{mark}  per arm: "
        + ", ".join(f"{k} {v:.5f}" for k, v in sigma["per_arm_sd_bpb"].items()),
    ]
    bart = sigma["bartlett"]
    if bart is not None:
        verdict = (
            "REJECTS. The pooled sigma is abandoned and every contrast below is re-run with "
            "unpooled Welch standard errors and Welch-Satterthwaite df, which costs power."
            if bart["rejects"]
            else "does not reject, which is not evidence of equal variances -- at five seeds "
            "per arm this test has little power and the pre-registration says so in advance."
        )
        out += [
            f"{mark}  Bartlett chi2({bart['df']}) = {bart['statistic']:.3f}, p = "
            f"{bart['p_value']:.4f}: {verdict}",
            f"{mark}  largest within-arm sd over smallest: {bart['spread']:.2f}x",
        ]
    out.append("")

    if result.get("single_arm"):
        out += [f"{mark}NOTHING FURTHER IS ANALYSABLE.", f"{mark}  {result['single_arm']}", ""]
        return "\n".join(out)

    pairing = result["pairing"]  # type: ignore[index]
    fit = pairing["block_fit"]
    out += [
        f"{mark}(b) THE PAIRING, NOW THAT A SECOND ARM EXISTS",
        f"{mark}  rho-hat per contrast, from that contrast's own quintuple:",
    ]
    for name, estimate in pairing["rho_by_hypothesis"].items():
        out.append(
            f"{mark}    {name:<5} Pearson {estimate['rho_pearson']:+.3f} "
            f"[{estimate['ci_low']:+.3f}, {estimate['ci_high']:+.3f}], "
            f"variance-components {estimate['rho_variance_components']:+.3f}, "
            f"sigma_delta {estimate['sigma_delta']:.5f} BPB"
        )
    out += [
        f"{mark}  design-level intraclass rho from the block fit: "
        f"{pairing['intraclass_rho']:+.3f}  [post-hoc]",
        f"{mark}  seed block F test: MS_seed {fit['ms_seed']:.3e} against MS_error "
        f"{fit['ms_error_paired']:.3e}, p = {fit['seed_p_value']:.4f}",
        f"{mark}  error df: paired (k-1)(n-1) = {fit['df_paired']}, unpaired k(n-1) = "
        f"{fit['df_unpaired']}",
        f"{mark}  break-even rho: {pairing['pre_registered_break_even']:.2f} as pre-registered "
        f"at three arms; {pairing['recomputed_break_even']:.3f} recomputed for the four that "
        "ran  [post-hoc; the pre-registered constant is the operative one]",
        "",
    ]

    out += [
        f"{mark}(c) THE CONTRASTS",
        f"{mark}  The gate is |delta| >= 2 SE, as pre-registered. Two standard errors is not a "
        "5% test when",
        f"{mark}  sigma is estimated, so the exact two-sided t p-value and the 5% line are "
        "printed beside it.",
        "",
    ]
    for entry in result["contrasts"]:  # type: ignore[index]
        if "rows" not in entry:
            out += [f"{mark}  {entry['name']}: {entry['status']}", ""]
            continue
        out += [
            f"{mark}  {entry['name']}: {entry['treatment']} - {entry['comparator']}",
            f"{mark}    {entry['claim']}",
        ]
        for row in entry["rows"]:
            star = " <- PRIMARY" if row["primary"] else ""
            out.append(
                f"{mark}    {row['endpoint']:<16} {row['analysis']:<18} "
                f"delta {row['delta_nats']:+.5f} nats ({row['delta_bpb']:+.6f} BPB)  "
                f"SE {row['se_bpb'] * NATS_PER_BPB:.5f}  df {row['df']:>2}  "
                f"t {row['t_statistic']:+.3f}  p {row['p_value']:.4f}{star}"
            )
            out.append(
                f"{mark}    {'':<16} {'':<18} 95% CI [{row['ci_nats'][0]:+.5f}, "
                f"{row['ci_nats'][1]:+.5f}] nats"
            )
            out.append(
                f"{mark}    {'':<16} {'':<18} gate {GATE_SIGMAS:.0f} SE = "
                f"{row['gate_bpb'] * NATS_PER_BPB:.5f} nats: "
                f"{'CLEARED' if row['clears_gate'] else 'not cleared'}; "
                f"5% line {row['five_percent_bpb'] * NATS_PER_BPB:.5f}: "
                f"{'cleared' if row['clears_five_percent'] else 'not cleared'}; "
                f"MDE {row['mde_nats']:.5f}: "
                f"{'cleared' if row['clears_mde'] else 'NOT CLEARED'}"
            )
            if row["primary"]:
                out.append(
                    f"{mark}    {'':<16} {'':<18} direction "
                    + (
                        "as predicted"
                        if row["direction_as_predicted"]
                        else "OPPOSITE to the " "prediction"
                    )
                )
                if row["excluded_reference_effects"]:
                    out.append(
                        f"{mark}    {'':<16} {'':<18} the interval excludes: "
                        + "; ".join(row["excluded_reference_effects"])
                    )
                else:
                    out.append(
                        f"{mark}    {'':<16} {'':<18} the interval excludes none of the "
                        "literature effects, so the data are consistent with all of them"
                    )
        out.append("")

    out += [f"{mark}(d) PER SOURCE, because a pooled mean hides an effect in one source"]
    for entry in result["per_source"]:  # type: ignore[index]
        out.append(f"{mark}  {entry['name']}")
        for row in entry["rows"]:
            out.append(
                f"{mark}    {row['source']:<18} delta {row['delta_nats']:+.5f} nats  "
                f"SE {row['se_bpb'] * NATS_PER_BPB:.5f}  p {row['p_value']:.4f}  "
                f"{'clears the gate' if row['clears_gate'] else '-'}"
            )
    out.append("")

    stability = result["h7"]  # type: ignore[index]
    out += [
        f"{mark}(e) H7, THE STABILITY OUTCOME  [secondary, pre-registered 2026-08-08]",
        f"{mark}  {stability['statement']}",
        f"{mark}  {stability['limit']}",
    ]
    for test in stability["tests"]:
        if "status" in test:
            out.append(f"{mark}  {test['arm']}: {test['status']}")
            continue
        tag = "H7" if test["pre_registered"] else "descriptive, post-hoc"
        out.append(f"{mark}  {test['arm']} vs baseline  [{tag}]")
        for which in ("primary", "secondary"):
            if which not in test:
                continue
            item = test[which]
            out.append(
                f"{mark}    {which:<10} {item['statistic_name']}: "
                f"{list(item['treatment_values'])} against {list(item['comparator_values'])}"
            )
            out.append(
                f"{mark}    {'':<10} difference of means {item['observed']:+.3f}, exact "
                f"two-sided p = {item['p_value']:.4f} over {item['n_permutations']} splits "
                f"(smallest attainable {item['smallest_attainable_p']:.4f})"
                + ("; the arms separate completely" if item["complete_separation"] else "")
            )
    out.append("")

    dose_block = str((result.get("dose") or {}).get("rendered") or "")
    if dose_block:
        out += [f"{mark}(f) THE TRAINING DOSE  [pre-registered 2026-08-10]"]
        out += [f"{mark}{line}" for line in dose_block.splitlines()]
        out.append("")

    out += [f"{mark}EVERYTHING ABOVE THAT THE PRE-REGISTRATION DOES NOT CONTAIN"]
    for item in result["post_hoc"]:  # type: ignore[index]
        out.append(f"{mark}  [{item['date']}] {item['what']}")
    out.append("")

    if label != "measured":
        out += _banner(label, [], where="ABOVE")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# Self-test.
# ---------------------------------------------------------------------------------------


def self_test(replicates: int = 400) -> int:
    """
    Drive the whole pipeline over planted truths, and say out loud what each check found.

    THE ESTIMATORS INHERITED FROM ``noise_floor`` HAVE THEIR OWN SELF-TEST AND THIS IS NOT IT.
    What is checked here is the machinery this module adds: the block fit's error term, the
    recovery of a planted effect and a planted correlation, the exact permutation test's
    calibration and its stated floor, Bartlett's size, and -- the one that matters most -- that
    a planted null comes back as a null at about the rate it should.

    :param replicates: Synthetic tranches per Monte Carlo check.

    :returns: A process exit status. Non-zero if any check misses.
    """
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<58} {detail}")

    def close(name: str, got: float, want: float, tolerance: float) -> None:
        check(name, abs(got - want) <= tolerance, f"{got:+.5f} against {want:+.5f}")

    print("analysis self-test, synthetic tranches with a planted truth")
    print()

    print("  the block fit recovers what was planted")
    planted_sigma, planted_effect = 0.00061, -0.010
    recovered_icc: Dict[float, float] = {}
    for planted_rho in (0.0, 0.3, 0.7):
        fits, deltas, rhos, coverage = [], [], [], []
        for r in range(replicates):
            result = analyse(
                synthetic_tranche(
                    {"faithful": planted_effect, "output-only": 0.0, "mhc": planted_effect},
                    sigma_bpb=planted_sigma,
                    rho=planted_rho,
                    rng_seed=r,
                ),
                label="synthetic",
            )
            fits.append(result["sigma"]["sigma_bpb_unbiased"])  # type: ignore[index]
            row = next(
                row
                for entry in result["contrasts"]  # type: ignore[index]
                for row in entry["rows"]
                if entry["name"] == "H1" and row["analysis"] == "paired"
            )
            deltas.append(row["delta_nats"])
            rhos.append(result["pairing"]["intraclass_rho"])  # type: ignore[index]
            coverage.append(row["ci_nats"][0] <= planted_effect <= row["ci_nats"][1])
        recovered_icc[planted_rho] = float(np.mean(rhos))

        if planted_rho == 0.3:
            close(
                "pooled sigma-hat recovers the planted noise floor",
                float(np.mean(fits)),
                planted_sigma,
                4.0 * float(np.std(fits, ddof=1)) / math.sqrt(len(fits)),
            )
        close(
            f"H1 recovers the planted effect at rho = {planted_rho:.1f}",
            float(np.mean(deltas)),
            planted_effect,
            4.0 * float(np.std(deltas, ddof=1)) / math.sqrt(len(deltas)),
        )
        rate = float(np.mean(coverage))
        check(
            f"the 95% interval covers the planted effect at rho = {planted_rho:.1f}",
            abs(rate - 0.95) < 4.0 * math.sqrt(0.95 * 0.05 / replicates) + 0.01,
            f"{rate:.3f} against 0.950",
        )

    # The ICC is one mean square over another and inherits the downward bias every ratio of
    # variance estimates has at this df -- the same bias noise_floor documents for the Pearson
    # rho-hat, which comes back 0.65 against a planted 0.70 at five pairs. It is reported and
    # is not used for anything: the paired standard error comes from the blocked mean square
    # directly, and mde_from_se prices the design off that standard error rather than off rho.
    # So what is checked is that it is monotone and biased towards zero, not that it is right.
    ordered_icc = [recovered_icc[r] for r in (0.0, 0.3, 0.7)]
    check(
        "the intraclass rho is monotone in the planted correlation",
        ordered_icc == sorted(ordered_icc),
        ", ".join(f"{v:+.3f}" for v in ordered_icc),
    )
    check(
        "and biased towards zero rather than away from it, as documented",
        all(got <= planted + 1e-9 for got, planted in zip(ordered_icc[1:], (0.3, 0.7))),
        f"{ordered_icc[1]:+.3f} for 0.3, {ordered_icc[2]:+.3f} for 0.7",
    )

    print()
    print("  a planted null stays a null, and a planted effect is found")
    null_positive = 0
    effect_positive = 0
    for r in range(replicates):
        flat = analyse(
            synthetic_tranche(
                {"faithful": 0.0, "output-only": 0.0, "mhc": 0.0},
                sigma_bpb=planted_sigma,
                rho=planted_rho,
                rng_seed=10_000 + r,
            ),
            label="synthetic",
        )
        loud = analyse(
            synthetic_tranche(
                {"faithful": -0.030, "output-only": 0.0, "mhc": 0.0},
                sigma_bpb=planted_sigma,
                rho=planted_rho,
                rng_seed=20_000 + r,
            ),
            label="synthetic",
        )
        for source, counter in ((flat, "null"), (loud, "effect")):
            row = next(
                row
                for entry in source["contrasts"]  # type: ignore[index]
                for row in entry["rows"]
                if entry["name"] == "H1" and row["analysis"] == "paired"
            )
            if row["p_value"] < 0.05:
                if counter == "null":
                    null_positive += 1
                else:
                    effect_positive += 1

    rate = null_positive / replicates
    check(
        "false-positive rate of the paired t at a planted null",
        abs(rate - 0.05) < 4.0 * math.sqrt(0.05 * 0.95 / replicates) + 0.005,
        f"{rate:.3f} against 0.050",
    )
    detection = effect_positive / replicates
    check(
        "a planted -0.030 nats is detected essentially always",
        detection > 0.99,
        f"{detection:.3f} of {replicates}",
    )

    print()
    print("  the exact permutation test, and the floor it cannot go below")
    separated = exact_permutation_test([40, 41, 42, 43, 44], [10, 11, 12, 13, 14], "x", "a", "b")
    close("complete separation reaches 2/C(10,5)", separated.p_value, 2.0 / 252.0, 1e-12)
    check(
        "and is recognised as complete separation",
        separated.complete_separation,
        f"{separated.p_value:.4f}",
    )
    overlapped = exact_permutation_test([12, 14, 16, 18, 40], [10, 11, 13, 15, 17], "x", "a", "b")
    check(
        "one arm's outlier alone does not reach 0.05",
        overlapped.p_value > 0.05,
        f"p = {overlapped.p_value:.4f}",
    )
    identical = exact_permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "x", "a", "b")
    close("identical groups give p = 1", identical.p_value, 1.0, 1e-12)

    uniform = []
    rng = np.random.default_rng(11)
    for _ in range(replicates):
        pooled = rng.standard_normal(10)
        uniform.append(exact_permutation_test(pooled[:5], pooled[5:], "x", "a", "b").p_value < 0.05)
    rate = float(np.mean(uniform))
    check(
        "permutation size under exchangeability is at or below alpha",
        rate <= 0.05 + 4.0 * math.sqrt(0.05 * 0.95 / replicates),
        f"{rate:.3f} against <= 0.050",
    )

    print()
    print("  Bartlett, and the break-even the pre-registration quotes")
    rng = np.random.default_rng(5)
    rejections = sum(
        bartlett([rng.standard_normal(5) for _ in range(4)], list("abcd")).rejects
        for _ in range(replicates)
    )
    rate = rejections / replicates
    check(
        "Bartlett holds its size on four equal-variance arms",
        rate <= 0.05 + 4.0 * math.sqrt(0.05 * 0.95 / replicates),
        f"{rate:.3f} against <= 0.050",
    )
    loud = bartlett(
        [rng.standard_normal(5) * scale for scale in (1.0, 1.0, 1.0, 40.0)], list("abcd")
    )
    check("and rejects a fortyfold difference", loud.rejects, f"p = {loud.p_value:.5f}")

    close("break-even rho at 3 arms x 5 seeds", break_even_rho(3, 5), 0.09, 0.02)
    check(
        "break-even rho does not depend on sigma",
        abs(break_even_rho(4, 5, 0.001) - break_even_rho(4, 5, 0.1)) < 1e-6,
        f"{break_even_rho(4, 5):.4f} at four arms",
    )

    print()
    print("  the refusals fire")
    try:
        block_fit(np.asarray([[1.0, 1.0]]), ["baseline"], [0, 1])
        check("a one-arm block fit", False, "no refusal raised")
    except ValueError:
        check("a one-arm block fit", True, "refused, as it must")

    mismatched = synthetic_tranche({"faithful": 0.0}, rng_seed=1)
    mismatched[1].seeds = [9, 8, 7, 6, 5]
    try:
        analyse(mismatched, label="synthetic")
        check("arms that do not share seeds", False, "no refusal raised")
    except Refusal:
        check("arms that do not share seeds", True, "refused, as it must")

    missing = synthetic_tranche({"faithful": 0.0}, rng_seed=1)
    missing[1].declined = [None] * SEEDS_PER_ARM
    outcome = h7(missing)
    check(
        "an absent stability key is refused rather than read as zero",
        any("refused" in str(t.get("status", "")) for t in outcome["tests"]),
        "refused, as it must",
    )

    print()
    print("no misses." if not failures else f"{failures} check(s) missed.")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------------------


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _cache_load(path: str) -> List[ArmSeries]:
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("label") != "measured":
        raise Refusal(
            f"{path} is not a cache of measured data -- it is labelled "
            f"{payload.get('label')!r}. Delete it and re-read W&B."
        )
    return [ArmSeries(**entry) for entry in payload["arms"]]


def _cache_save(path: str, arms: Sequence[ArmSeries]) -> None:
    payload = {"label": "measured", "arms": [asdict(arm) for arm in arms]}
    _write(path, json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="The pre-registered analysis of the hyper-connection tranche.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", help="The experiment slug the runs are grouped under.")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=SUBMISSION",
        help="An arm and the platform run id that ran it. Repeatable, and required for every "
        "arm the analysis is to include. There is no group-wide sweep: the slug holds every "
        "attempt at this module, and a group is not an experimental unit.",
    )
    parser.add_argument("--cells", type=int, default=SEEDS_PER_ARM)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument(
        "--weights",
        default=os.path.join(_HERE, "noise-floor-skip-step.json"),
        help="The frozen artifact the strata weights are read from, for the secondary endpoint.",
    )
    parser.add_argument("--out", help="Directory for the report, the JSON and the figures.")
    parser.add_argument("--cache", help="Read from and write to this JSON instead of W&B.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Downgrade the completeness refusals -- a short arm, an unfinished cell, a "
        "mid-horizon endpoint -- to warnings, and stamp PROVISIONAL on every artifact. It does "
        "not downgrade a seed collision, a mislabelled arm or a missing stability family.",
    )
    parser.add_argument("--self-test", action="store_true", help="No network. Planted truths.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the whole pipeline over a SYNTHETIC tranche and write its report and figures "
        "under a 'synthetic-' prefix. No network, no measurement, and unreachable from the "
        "measured path.",
    )
    parser.add_argument(
        "--demo-effect",
        type=float,
        default=-0.010,
        help="The effect --demo plants on 'faithful', in nats.",
    )
    opts = parser.parse_args()

    if opts.self_test:
        return self_test()

    if opts.demo and (opts.group or opts.arm or opts.cache):
        parser.error(
            "--demo generates synthetic data and cannot be combined with --group, --arm or "
            "--cache. The two paths are kept apart on purpose: this module exists partly "
            "because a tool that could answer without data was asked to, and did."
        )
    if not opts.demo and not (opts.group and opts.arm) and not opts.cache:
        parser.error(
            "give --group with one or more --arm NAME=SUBMISSION to read W&B, or --cache to "
            "re-read a cache of a previous read, or --demo to exercise the pipeline on "
            "synthetic data. There is no default and there is no fallback."
        )

    try:
        if opts.demo:
            arms = synthetic_tranche(
                {"faithful": opts.demo_effect, "output-only": 0.0, "mhc": opts.demo_effect / 2},
                declined={"faithful": (44, 51, 39, 47, 62), "mhc": (18, 22, 15, 19, 21)},
                triggers={
                    "faithful": (8.4, 11.2, 6.9, 9.1, 14.0),
                    "mhc": (0.55, 0.61, 0.44, 0.52, 0.58),
                },
            )
            label, provisional = "synthetic", []
        elif opts.cache and os.path.exists(opts.cache) and not opts.group:
            arms = _cache_load(opts.cache)
            label = "measured"
            provisional = []
        else:
            mapping = []
            for item in opts.arm:
                if "=" not in item:
                    parser.error(f"--arm takes NAME=SUBMISSION, got {item!r}")
                name, submission = item.split("=", 1)
                if name not in ARM_ORDER:
                    parser.error(f"--arm {name!r} is not one of the funded arms {list(ARM_ORDER)}")
                mapping.append((name, submission))
            arms = [
                read_arm(opts.entity, opts.project, opts.group, name, submission, opts.cells)
                for name, submission in mapping
            ]
            label, provisional = "measured", []
            if opts.cache:
                _cache_save(opts.cache, arms)

        if label == "measured":
            drift = check_the_nats_conversion(arms)
            if drift:
                raise Refusal("\n".join(drift))
            blocking = stability_refusals(arms)
            if blocking:
                raise Refusal("\n".join(blocking))
            coverage = completeness_refusals(arms, opts.horizon, opts.cells)
            if coverage and not opts.allow_provisional:
                raise Refusal(
                    "\n".join(coverage)
                    + "\n\nPass --allow-provisional to analyse this anyway. It will be stamped "
                    "PROVISIONAL on every line of every artifact, because a mid-tranche "
                    "reading that is not marked as one is indistinguishable from the result."
                )
            provisional = coverage

        sources = arms[0].sources
        weights = frozen_weights(opts.weights, sources) if opts.weights else None
        result = analyse(arms, weights, provisional=provisional, label=label)
        text = render(result)
        print(text)

        if opts.out:
            prefix = "synthetic-" if label != "measured" else ""
            _write(os.path.join(opts.out, f"{prefix}analysis.txt"), text)
            _write(
                os.path.join(opts.out, f"{prefix}analysis.json"),
                json.dumps(result, indent=2, sort_keys=True, default=float),
            )
            if not opts.no_figures:
                import analysis_figures

                written = analysis_figures.draw(arms, result, opts.out)
                print()
                print(
                    f"{'[synthetic] ' if label != 'measured' else ''}wrote {len(written)} "
                    f"figure(s) to {opts.out}"
                )
    except Refusal as refusal:
        print()
        print("=" * 92)
        print("REFUSING TO PRODUCE AN ANALYSIS.")
        print(str(refusal))
        print("=" * 92)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
