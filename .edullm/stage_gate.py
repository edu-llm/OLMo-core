#!/usr/bin/env python3
"""Decide whether stage 2 of the tranche is safe to submit, from what stage 1 has logged.

A THIRD SIBLING TO ``wandb_panels.py`` AND ``noise_floor.py``, AND THE DIVISION IS BY QUESTION.
``wandb_panels --verify`` asks whether a metric *key* arrived. ``noise_floor`` asks what the
numbers in those keys are worth once a run has finished. This one asks the only question that
has to be answered while the runs are still going: **is the configuration sound, and is it
going to finish inside its bound?** Nothing here estimates a variance and nothing here draws a
chart.

The gate is health rather than completion. Waiting for five 18-hour cells to finish before
submitting the treatment arms would cost a day and buy nothing scientific: the per-source
variance weights are computed from baseline data alone, and treatment arms running concurrently
contaminate neither them nor sigma-hat. What stage 2 actually needs to know is that the
baseline's own config is not broken in a way that would make all fifteen runs worthless, and
that is knowable by roughly the first held-out evaluation.

THE FIVE CHECKS, AND WHY EACH ONE IS HERE RATHER THAN LEFT TO THE EYE.

``seeds``       Five cells get one command and the seed comes from the fan-out index, so the
                failure mode is five bit-identical replicates and a measured noise floor of
                zero -- against which every later arm is significant. It looks like a
                beautifully clean experiment and there is no downstream check that catches it.
                Read from each cell's own logged config rather than from the submission, and
                cross-checked against the loss curves actually differing.

``loss``        Finite, decreasing, no NaN and no inf. Cheap and it has to be said.

``step``        The clean median step time, which is what the runtime bound is spent against.
                See :func:`clean_step_seconds` for what "clean" has to mean and what it cost
                this branch to learn it.

``fit``         6,000 steps at that rate, plus the evaluations, the checkpoints and start-up,
                against ``--hours``. Priced through :func:`hyper_connection_arms.arm_seconds`
                so this file and the submission cannot disagree about the cost model.

``mfu``         Recomputed from the model's own ``num_flops_per_token`` and the measured step
                time rather than trusted, against a dense BF16 peak matched to the card the
                run actually reports. See :func:`mfu_percent`.

NOTHING HERE IS TOLD WHAT SHAPE IT IS LOOKING AT. The tranche has now run on two, and the
things that differ between them -- four devices against eight, 196,608 tokens on a device
against 98,304, 362.05 TF against 312, a held-out evaluation costing 104 s against 25 -- are
every input the last two checks have. So each one is read back out of the run: the device from
its W&B metadata, the tokens on it from ``TPS / BPS``, the peak the callback used from
``flopsPS / MFU``, the evaluation from its own logged duration. A constant here would have
been a factor of two in the MFU the first time the tranche moved card.

    python .edullm/stage_gate.py --self-test                    # no network
    python .edullm/stage_gate.py --run run_019fe2f4-f528 --cells 5 --bound-hours 7
    python .edullm/stage_gate.py --run run_019fe2f4-f528 --cells 5 --bound-hours 7 --watch 300
"""

import argparse
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms  # noqa: E402
from hyper_connection_arms import ARMS  # noqa: E402

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: The seven sources ``regmix-10b-v1`` declares a validation shard for, and the labels the
#: evaluator names its per-source metrics after. All seven have to arrive: the per-source
#: inverse-variance weighting stage 2 is gated on is a weight per source, and a pooled number
#: over arxiv, code, web text and Wikipedia together is exactly the statistic the
#: pre-registration says hides the effect it is meant to measure.
#:
#: Held here rather than imported from ``noise_floor``, which carries the same list, so that
#: this module can be run against a live fan-out without pulling in an estimator library --
#: and so a gate that has to work during a run does not fail to import because a sibling is
#: mid-edit. ``test_stage_gate`` asserts the two agree whenever both are present, so the
#: duplication cannot drift.
HELD_OUT_SOURCES: Tuple[str, ...] = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)

#: Dense BF16 peak per device, in FLOP/s, keyed by the substring of ``torch.cuda`` device name
#: that identifies the part. Longest-prefix ordering matters and is the same trap
#: ``SpeedMonitorCallback`` documents: "L4" is a substring of "L40S".
#:
#: TRANSCRIBED FROM THE DATASHEETS A SECOND TIME RATHER THAN IMPORTED FROM THE CALLBACK, and
#: that duplication is the whole point of the module. This file exists to check that constant,
#: and a check that reads the value it is checking is not a check. Two independent
#: transcriptions agreeing is evidence; one transcription read twice is not.
#:
#: TWO INDEPENDENT FACTORS OF TWO SIT ON EVERY ONE OF THESE ROWS AND ONLY ONE OF THEM IS
#: SPARSITY. The other is the accumulation format, and it is the one that caught this branch
#: out. On the consumer-class dies -- AD102, which is the L40S and the L40, and GA102, which is
#: the A10G -- the tensor cores run FP32 accumulation at exactly half the FP16-accumulate rate,
#: and the product datasheets quote the FP16-accumulate figure. NVIDIA's Ada whitepaper says it
#: outright in its AD102 table, 330.3 TFLOPS for FP16 with FP16 accumulate against 165.2 for
#: FP16 or BF16 with FP32 accumulate, and its L40 appendix lists BF16 as ``181 | 362`` where
#: the datasheet's headline for the same silicon is 362.05. Torch accumulates in FP32 always,
#: so 181.03 is the peak an L40S can reach for the only matmul training performs.
#:
#: The datacenter dies have no such penalty: GA100 and GH100 hit their quoted rate with FP32
#: accumulate, so the A100 and H100 rows are their datasheet figures halved for sparsity alone.
#:
#: MATCHED BY NAME, WHICH IS THE OTHER THING THIS CATCHES. The callback's A100 figure is not an
#: A100 branch -- it is the ``else`` at the bottom of the chain, so an unrecognised card is
#: silently scored against 312 TF and reports an MFU with no relationship to its hardware. A
#: name that matches nothing here is reported rather than defaulted.
DENSE_BF16_PEAK_FLOPS: Tuple[Tuple[str, float], ...] = (
    ("H100 NVL", 1671e12 / 2),
    ("H100 PCIe", 1513e12 / 2),
    ("H100", 1979e12 / 2),
    ("B200", 4.5e15 / 2),
    ("L40S", 362.05e12 / 2),
    ("L40", 362.05e12 / 2),
    ("L4", 242e12 / 2),
    ("A10G", 140e12 / 2),
    ("A100", 624e12 / 2),
)

#: Dense BF16 peak for one L40S at FP32 accumulate, which is 181.03 TF and not the 362.05 the
#: datasheet leads with. Named separately because every MFU this branch has quoted for an L40S
#: was taken against the wrong one of those two, and several tests turn on the difference.
L40S_BF16_DENSE_FLOPS: float = 362.05e12 / 2

#: Dense BF16 peak for one A100, of either memory size: 40 GB and 80 GB differ in bandwidth
#: and capacity and not in tensor-core rate.
A100_BF16_DENSE_FLOPS: float = 312e12


def peak_bf16_flops(device_name: str) -> Optional[float]:
    """
    The dense BF16 peak for a CUDA device name, or ``None`` for a part not in the table.

    :param device_name: What ``torch.cuda.get_device_name`` returned, as W&B recorded it in
        the run's metadata -- for instance ``NVIDIA A100-SXM4-40GB``.

    :returns: FLOP/s, or ``None`` when the name matches nothing, which is a thing to report
        and not a thing to substitute a default for.
    """
    for token, peak in DENSE_BF16_PEAK_FLOPS:
        if token in device_name:
            return peak
    return None


#: Rows to drop from the head of a run's throughput history. The callback already skips the
#: first optimizer step -- ``_first_step`` in ``SpeedMonitorCallback.post_step`` -- so history
#: opens at step 2, but that row is still measured against a timer started before the
#: allocator and ``torch.compile`` have settled and it reads several per cent off in both
#: directions. One row out of six thousand.
WARMUP_ROWS_DROPPED = 1

#: A step that took more than this multiple of the running median is an instrument firing
#: rather than a step, whatever the rule-based filter thought. Only ever used to *report* a
#: disagreement between the two filters -- the median itself comes from the rule -- because a
#: purely statistical filter would happily discard a genuine slowdown, which is the one thing
#: this module must not hide.
OUTLIER_RATIO = 1.5


@dataclass(frozen=True)
class CleanStepTime:
    """
    A median step time and the full accounting of what was excluded to get it.

    The row count is not a footnote. The number this replaces -- 11.69 s/step, which re-planned
    a tranche -- was a median over five rows that were exactly the lane monitor's firing steps,
    and it was quoted with a caveat that "only about five throughput points were logged", which
    is the filter describing itself rather than the run.
    """

    median: float
    """Seconds per optimizer step over the clean rows."""

    rows_used: int
    """How many rows the median is over. Quote this beside the median, always."""

    rows_total: int
    """How many rows the history held before any filtering."""

    excluded: Mapping[str, int] = field(default_factory=dict)
    """Reason -> how many rows it removed."""

    iqr: float = 0.0
    """Interquartile spread over the clean rows. A wide one means the filter missed something."""

    outliers_remaining: int = 0
    """
    Clean rows still above :data:`OUTLIER_RATIO` times the median. Should be zero; a non-zero
    count means an instrument fires on a schedule this filter does not know about.
    """

    @property
    def enough_rows(self) -> bool:
        """Whether the median is over enough steps to be worth quoting."""
        return self.rows_used >= 20


def clean_step_seconds(
    rows: Sequence[Mapping[str, object]],
    *,
    eval_interval: int,
    warmup_rows: int = WARMUP_ROWS_DROPPED,
) -> CleanStepTime:
    """
    The median time of a step on which no instrument fired.

    GETTING THIS FILTER WRONG IS WHAT PRODUCED THE NUMBER THIS TRANCHE WAS FIRST PRICED AT, and
    the two ways to get it wrong are not symmetric.

    *Selecting* the instrument's rows is the one that shipped. Filter a history to the rows
    carrying the monitor's own ``hc/*`` keys and every surviving row is a firing step, so the
    median is the cost of the instrument sampled once per firing rather than the cost of a
    step: 11.69 s/step against a true 10.32, quoted with a caveat that "only about five
    throughput points were logged", which is the filter describing itself.

    *Contaminating* with them is milder than it looks and the difference is worth being exact
    about, because overstating it would be the same sin in the other direction. A median is
    robust to a minority, and the instruments here are a small one -- at ``--eval-interval
    500`` the evaluator touches 0.2% of steps and at ``--monitor-interval 50`` the monitor
    touches 2% -- so an unfiltered *median* lands on the right answer anyway. What the
    contamination does wreck is any estimate built on a sum: wall clock over steps, or a mean.
    Those carry the evaluator's 104 seconds thirteen times and read minutes per hour high.

    So the filter earns its keep twice: it makes the median defensible rather than lucky, and
    it makes the row count it reports a real one.

    So a row is dropped when any of these is true:

    - it is one of the first ``warmup_rows`` logged, which pay for ``torch.compile``;
    - the in-loop evaluator ran on it, read from the evaluator's own logged duration;
    - its step is a multiple of ``eval_interval``, which catches the same rows when the
      duration key has not flushed yet, and step 0;
    - it is the row *after* an evaluation. Callback ordering decides whether the evaluator's
      time lands in the step it ran on or in the next one's interval, and this module does not
      need to know which -- it can afford both rows out of six thousand;
    - the lane monitor fired on it, read from the presence of an ``hc/*`` key. The baseline arm
      attaches no monitor at all, so this removes nothing there and is not dead code: the same
      function prices the treatment arms in stage 2.

    :param rows: History rows, each holding ``_step`` and ``throughput/device/BPS``. Rows
        without both are ignored rather than dropped, since they are the evaluator's own
        summary rows and were never steps.
    :param eval_interval: ``--eval-interval``, in steps.
    :param warmup_rows: How many leading rows to drop.

    :returns: The median and the accounting behind it.

    :raises ValueError: If no row survives, which means the filter is wrong rather than the run.
    """
    timed = [r for r in rows if r.get("throughput/device/BPS") and r.get("_step") is not None]
    timed = sorted(timed, key=lambda r: int(r["_step"]))  # type: ignore[arg-type]
    total = len(timed)
    if not total:
        raise ValueError("no throughput rows in this history at all")

    evaluated_steps = {
        int(r["_step"])  # type: ignore[arg-type]
        for r in timed
        if _positive(r.get("throughput/in-loop eval time (s)"))
    }

    excluded: Dict[str, int] = {
        "warm-up": 0,
        "held-out evaluation": 0,
        "the step after an evaluation": 0,
        "lane monitor": 0,
    }
    kept: List[Tuple[int, float]] = []
    for position, row in enumerate(timed):
        step = int(row["_step"])  # type: ignore[arg-type]
        seconds = 1.0 / float(row["throughput/device/BPS"])  # type: ignore[arg-type]
        if position < warmup_rows:
            excluded["warm-up"] += 1
        elif step in evaluated_steps or (eval_interval > 0 and step % eval_interval == 0):
            excluded["held-out evaluation"] += 1
        elif (step - 1) in evaluated_steps or (
            eval_interval > 0 and (step - 1) % eval_interval == 0 and step > 1
        ):
            excluded["the step after an evaluation"] += 1
        elif any(str(k).startswith("hc/") for k in row):
            excluded["lane monitor"] += 1
        else:
            kept.append((step, seconds))

    if not kept:
        raise ValueError(
            f"every one of {total} rows was filtered out, which is this filter being wrong "
            "rather than the run being unmeasurable"
        )

    seconds = sorted(s for _, s in kept)
    median = statistics.median(seconds)
    quartiles = statistics.quantiles(seconds, n=4) if len(seconds) >= 4 else [median] * 3
    return CleanStepTime(
        median=median,
        rows_used=len(kept),
        rows_total=total,
        excluded={reason: n for reason, n in excluded.items() if n},
        iqr=quartiles[2] - quartiles[0],
        outliers_remaining=sum(1 for s in seconds if s > OUTLIER_RATIO * median),
    )


def _positive(value: object) -> bool:
    """Whether a history cell holds a number above zero. W&B writes ``None`` into gaps."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def mfu_percent(
    *,
    flops_per_token: float,
    tokens_per_device_step: float,
    seconds_per_step: float,
    peak_flops: float,
) -> float:
    """
    Model FLOPs utilization, computed the way ``SpeedMonitorCallback`` computes it.

    Written out here so that the reported figure can be checked against something rather than
    read, and the denominator is where the checking is needed. THE CONSTANT HAS BEEN WRONG
    TWICE ON THIS TABLE, in opposite directions and for different reasons.

    A100: OLMo-core v2.5.0 fixed a peak that was 2x too low, which had been reporting every
    A100 MFU at twice its true value. The current figure is ``624e12 * 0.5``, and 624 is the
    starred with-sparsity number on NVIDIA's A100 datasheet against a dense 312 -- so the
    halving is right and the constant is now the dense BF16 rate.

    L40S: until this branch added a branch for it, an L40S matched none of the names and fell
    through the same ``else``, so it was scored against the A100's 312 TF instead of its own
    362.05 and read 16% high. That is the failure mode the ``else`` still has for any part not
    in the chain, which is why :func:`peak_bf16_flops` matches by name and returns ``None``
    rather than defaulting.

    :param flops_per_token: The model's own ``num_flops_per_token`` at the run's sequence
        length. Not 6N: it counts attention, and on a hyper-connection arm it counts the lanes.
    :param tokens_per_device_step: Global batch over the number of ranks. Read back out of the
        run as ``TPS / BPS`` rather than assumed, since the two shapes this tranche has run on
        put 196,608 and 98,304 tokens on a device for the same 786,432-token global batch.
    :param seconds_per_step: The clean median.
    :param peak_flops: The device's dense peak at the dtype actually in use.

    :returns: MFU as a percentage.
    """
    return 100.0 * flops_per_token * tokens_per_device_step / seconds_per_step / peak_flops


@dataclass(frozen=True)
class Projection:
    """What a cell is going to cost in wall clock, against the bound it was submitted under."""

    hours: float
    """Projected wall clock for the whole run."""

    bound_hours: float
    """``--hours`` as submitted."""

    seconds_per_step: float
    """The step time it was projected at."""

    @property
    def fits(self) -> bool:
        return self.hours <= self.bound_hours

    @property
    def spare_fraction(self) -> float:
        """Share of the bound left unspent. Negative when the projection overruns it."""
        return (self.bound_hours - self.hours) / self.bound_hours


def project(
    arm_name: str,
    *,
    seconds_per_step: float,
    steps: int = hyper_connection_arms.TRANCHE_STEPS,
    bound_hours: float,
    eval_seconds: float = hyper_connection_arms.MEASURED_EVAL_SECONDS,
) -> Projection:
    """
    Price a cell at a measured step time.

    Delegates to :func:`hyper_connection_arms.arm_seconds` rather than re-adding the constants,
    so the projection and the submission cannot disagree about how many evaluations there are.
    That function counts fourteen -- one on startup, one on finish and one every 500 steps --
    and thirteen checkpoints, and it charges the lane monitor only to an arm that has lanes,
    which the baseline does not.

    THE STEP TIME IS NOT THE ONLY THING THAT MOVES WHEN THE SHAPE DOES, which is what
    ``eval_seconds`` is here for. ``arm_seconds`` defaults every fixed cost to its L40S
    measurement on the argument that the evaluations are the same seven shards and the
    checkpoints the same model to the same bucket -- true of the *work*, not of the time it
    takes. The A100 cells run the identical evaluation in about a quarter of the L40S's 104
    seconds, and fourteen of them is a fifth of an hour either way. Pass what the run
    measured. The checkpoint figure is left alone deliberately: it is not logged as a metric,
    so there is nothing to pass, and over-charging it is the safe direction.

    :param arm_name: The arm, as a key of :data:`hyper_connection_arms.ARMS`.
    :param seconds_per_step: The measured clean median.
    :param steps: The horizon.
    :param bound_hours: ``--hours`` as submitted.
    :param eval_seconds: One held-out evaluation, as this shape measures it.
    """
    seconds = hyper_connection_arms.arm_seconds(
        ARMS[arm_name],
        steps=steps,
        seconds_per_step=seconds_per_step,
        eval_seconds=eval_seconds,
    )
    return Projection(
        hours=seconds / 3600.0, bound_hours=bound_hours, seconds_per_step=seconds_per_step
    )


@dataclass
class CellHealth:
    """One fan-out cell, read back out of what it logged."""

    cell: int
    state: str
    step: int
    summary_lost_its_step: bool = False
    """
    Whether :attr:`step` had to be recovered from the history because the summary had none.

    True on the seven cells whose summaries a crash report overwrote. Kept as a field rather
    than folded into the step, because "step 4,910, recovered" and "step 4,910" are the same
    number and not the same statement: only one of them tells a reader that this run's summary
    is not to be believed about anything else either.
    """

    init_seed: Optional[int] = None
    model_init_seed: Optional[int] = None
    data_seed: Optional[int] = None
    first_loss: Optional[float] = None
    last_loss: Optional[float] = None
    nonfinite_losses: int = 0
    z_loss_seen: bool = False
    loss_at: Mapping[int, float] = field(default_factory=dict)
    """Training cross-entropy by step, so two cells can be compared at a step they share."""

    evaluations: int = 0
    """How many held-out evaluations have landed. One is the startup pass; two means step 500."""

    arms_consistent: Tuple[str, ...] = ()
    """Arms this cell's own logged config could have come from. See :func:`arms_consistent_with`."""

    sources: Tuple[str, ...] = ()
    bpb_sources: Tuple[str, ...] = ()
    step_time: Optional[CleanStepTime] = None
    reported_mfu: Optional[float] = None
    flops_per_token: Optional[float] = None
    implied_peak_flops: Optional[float] = None

    device_name: str = ""
    """What ``torch.cuda.get_device_name`` saw, out of the run's own W&B metadata."""

    tokens_per_device_step: Optional[float] = None
    """
    Read back as ``TPS / BPS`` rather than assumed from a rank count.

    The two shapes this tranche has run on put 196,608 and 98,304 tokens on a device for the
    same 786,432-token global batch, so a hard-coded divisor is a factor of two waiting to
    happen in the MFU -- which is the one number this module is here to be trusted about.
    """

    eval_seconds: Optional[float] = None
    """The most recent held-out evaluation's own logged duration, for the projection."""

    @property
    def loss_is_healthy(self) -> bool:
        return (
            self.nonfinite_losses == 0
            and self.first_loss is not None
            and self.last_loss is not None
            and self.last_loss < self.first_loss
        )


def arms_consistent_with(config: Mapping[str, object]) -> Tuple[str, ...]:
    """
    Which arms a cell's own logged config could have come from.

    THE ARM IS IN W&B AFTER ALL, WHICH IS WORTH SAYING BECAUSE THE BRANCH CURRENTLY BELIEVES IT
    IS NOT. ``train_on_corpus`` never writes an ``arm`` field, so a watcher looking for one
    finds nothing and has to be told the arm on its command line. But ``arm.apply`` edits the
    model config before it is saved, and those edits survive: ``model.block.hyper_connections``
    is absent on ``baseline`` and present on every lane arm, and among the funded four the
    triple ``(mode, doubly_stochastic, output_init_exponent)`` separates ``faithful`` from
    ``output-only`` from ``mhc``. A label taken from a command line is an assertion about a run;
    this is the run's own testimony, and it is the one that catches a cell that resolved to an
    arm nobody meant.

    BLOCK REUSE IS READ TOO, AND IT IS WHAT KEEPS THE ANSWER FROM BEING PERMANENTLY AMBIGUOUS.
    ``arm.apply`` writes ``model.block_reuse`` for a tied arm and leaves it absent otherwise, so
    the pair ``(lanes, tied)`` separates ``baseline`` from ``tied-baseline`` and ``faithful``
    from ``tied-faithful``. Without it every baseline cell comes back as two arms and the
    check's own output trains the reader to ignore the word "ambiguous".

    What remains genuinely unreadable is ``decay-everything``, which differs from ``faithful``
    in the optimizer alone and is not a model-config difference at all. It is unfunded, so the
    four arms that run are each uniquely identified; the function reports the tie rather than
    breaking it, because a gate that guesses is worse than one that says it cannot tell.

    :param config: A run's W&B config.

    :returns: Every arm name whose configuration matches, in table order. A single-element
        result identifies the arm; more than one means the config cannot tell them apart.
    """
    model = config.get("model")
    model = model if isinstance(model, dict) else {}
    block = model.get("block")
    logged = block.get("hyper_connections") if isinstance(block, dict) else None
    tied = model.get("block_reuse") is not None

    matches: List[str] = []
    for name, arm in ARMS.items():
        if (arm.reuse_factor is not None) != tied:
            continue
        if arm.hyper_connections is None:
            if not isinstance(logged, dict):
                matches.append(name)
            continue
        if not isinstance(logged, dict):
            continue
        wanted = arm.hyper_connections.as_config_dict()
        if all(_same(logged.get(key), value) for key, value in wanted.items()):
            matches.append(name)
    return tuple(matches)


def _same(observed: object, wanted: object) -> bool:
    """Compare one config field, tolerating the int/float round trip through JSON."""
    if isinstance(observed, (int, float)) and isinstance(wanted, (int, float)):
        return (
            not isinstance(observed, bool)
            and not isinstance(wanted, bool)
            and (math.isclose(float(observed), float(wanted), rel_tol=1e-9, abs_tol=1e-12))
            or observed == wanted
        )
    return observed == wanted


def seeds_are_distinct(cells: Sequence[CellHealth]) -> Tuple[bool, str]:
    """
    Whether the fan-out really resolved a different replicate per cell.

    THE FAILURE THIS EXISTS FOR IS SILENT AND IT DESTROYS THE TRANCHE. Every cell is handed one
    command and takes its seed from the array index, so a ``resolve_seed`` that mis-fired gives
    five bit-identical runs whose measured sigma is zero -- and five loss curves lying exactly
    on top of one another look like an unusually clean experiment, not like a bug.

    Both halves are checked, because either alone can be fooled: matching seeds in the config
    would be conclusive but a *reported* seed is not the same as an *applied* one, and distinct
    loss curves alone could come from kernel non-determinism at an identical seed.

    :param cells: The cells, with their seeds and losses filled in.

    :returns: ``(distinct, one sentence saying how it was established or what is wrong)``.
    """
    seeds = [c.model_init_seed for c in cells if c.model_init_seed is not None]
    if len(seeds) < len(cells):
        return False, f"only {len(seeds)} of {len(cells)} cells reported a seed in their config"
    if len(set(seeds)) != len(seeds):
        return False, (
            f"the cells resolved seeds {sorted(seeds)}, which repeat. Every cell of a fan-out "
            "is handed the same command, so this is the collapse `resolve_seed` exists to "
            "refuse: the replicates are one replicate and the measured noise floor is zero."
        )

    # AT A STEP THEY ALL SHARE, because the cells start minutes apart and comparing each one's
    # latest loss would compare different points on the same curve -- which would look
    # reassuringly distinct even if the five runs were bit-identical.
    shared = set.intersection(*(set(c.loss_at) for c in cells)) if cells else set()
    if not shared:
        return False, f"seeds {sorted(seeds)} are distinct in config, but no step is shared yet"
    step = max(shared)
    losses = [c.loss_at[step] for c in cells]
    if len(set(losses)) != len(losses):
        return False, (
            f"seeds {sorted(seeds)} are distinct but two cells report an identical loss at "
            f"step {step}, which means the seed reached the config and not the model"
        )
    spread = max(losses) - min(losses)
    return True, (
        f"seeds {sorted(seeds)} from the cells' own configs, and {len(losses)} distinct losses "
        f"at the shared step {step} spanning {spread:.4f} nats"
    )


def _held_out_sources(keys: Iterable[str], ending: str) -> Tuple[str, ...]:
    """
    The held-out sources named by whichever set of metric keys is handed over.

    Called with the summary's keys and then, where those hold none, with the history's -- see
    :func:`_read_one`. Both spellings of the same fact, and only one of them can be destroyed
    by a run that re-initialises W&B under an id it did not own.

    :param keys: Metric key names.
    :param ending: ``/CE loss`` or ``/BPB``.

    :returns: The source names, sorted.
    """
    return tuple(
        sorted(
            str(key).split("/")[2]
            for key in keys
            if str(key).startswith("eval/lm/") and str(key).endswith(ending)
        )
    )


def read_cells(
    entity: str, project_name: str, run_id: str, cells: int, eval_interval: int
) -> List[CellHealth]:
    """
    Read every cell of a fan-out out of W&B.

    A cell's W&B run id is the platform run id with ``-cell-<index>`` appended, while every
    cell shares the *display name*. Addressed by id: keying on the name would collapse all five
    into one, which is a real defect in ``wandb_panels.observed_keys`` and the reason its
    verdict says "1 run(s)" for a five-cell fan-out.

    :param entity: W&B entity.
    :param project_name: W&B project.
    :param run_id: The full platform run id.
    :param cells: The fan-out size.
    :param eval_interval: ``--eval-interval``, for the step-time filter.

    :returns: One :class:`CellHealth` per cell that exists yet, in cell order.
    """
    import wandb

    api = wandb.Api(timeout=120)
    found: List[CellHealth] = []
    for index in range(cells):
        try:
            run = api.run(f"{entity}/{project_name}/{run_id}-cell-{index}")
        except Exception:
            continue
        found.append(_read_one(run, index, eval_interval))
    return found


def _read_one(run, index: int, eval_interval: int) -> CellHealth:
    """Turn one W&B run into a :class:`CellHealth`."""
    config = run.config
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    loader = config.get("data_loader") if isinstance(config.get("data_loader"), dict) else {}

    # NO ``keys=`` FILTER, AND THAT IS NOT LAZINESS. Asking ``scan_history`` for a named list
    # returns zero rows on a *running* run as soon as the list contains
    # "throughput/in-loop eval time (s)" -- the same call against the same key returns rows
    # once the run has finished. A filter that silently empties itself while the run is live
    # is the worst possible failure here, because this module only ever runs while the run is
    # live, and it would report a healthy fan-out as one with no losses and no z-loss. The
    # unfiltered scan is also cheap: a finished 6,000-step cell is a few thousand rows.
    rows = list(run.scan_history(page_size=2000))

    # THE SUMMARY FIRST AND THE HISTORY WHERE THE SUMMARY HAS NOTHING, WHICH IS NOT THE SAME
    # AS BEING CAREFUL. Seven cells have a summary a crash report overwrote: no ``_step``, no
    # ``eval/lm/*`` keys, and a history holding every one of them. Read from the summary alone
    # those cells report step 0 and no held-out sources, and the per-cell checks below run
    # over `evaluated`, so a cell with no sources is not failed -- it is skipped. That is a
    # gate quietly deciding on four cells while saying five. The history is not overwritable
    # and it is already in hand here, so both are recovered from it.
    summary_keys = list(run.summary.keys())
    history_keys = {key for row in rows for key in row}
    sources = _held_out_sources(summary_keys, "/CE loss") or _held_out_sources(
        history_keys, "/CE loss"
    )
    bpb = _held_out_sources(summary_keys, "/BPB") or _held_out_sources(history_keys, "/BPB")

    summary_step = run.summary.get("_step")
    history_step = max((int(r["_step"]) for r in rows if r.get("_step") is not None), default=-1)

    losses = [r["train/CE loss"] for r in rows if r.get("train/CE loss") is not None]
    finite = [x for x in losses if math.isfinite(x)]
    loss_at = {
        int(r["_step"]): float(r["train/CE loss"])
        for r in rows
        if r.get("train/CE loss") is not None
        and r.get("_step") is not None
        and math.isfinite(float(r["train/CE loss"]))
    }
    # THE STARTUP EVALUATION IS NOT A MEASUREMENT OF AN EVALUATION and is dropped from the
    # figure the projection uses: it runs before anything is warm and the A100 cells pay 52 s
    # for it against 25 s for the step-500 pass. The L40S probe saw the same shape, 124.9 s
    # against 103.6. Thirteen of the fourteen an arm runs are the steady-state kind.
    eval_times = [
        float(r["throughput/in-loop eval time (s)"])
        for r in rows
        if _positive(r.get("throughput/in-loop eval time (s)"))
    ]
    evaluations = len(eval_times)
    steady_state_evals = eval_times[1:] or eval_times
    step_time: Optional[CleanStepTime] = None
    if rows:
        try:
            step_time = clean_step_seconds(rows, eval_interval=eval_interval)
        except ValueError:
            step_time = None

    # EVERY QUANTITY THE MFU CHECK NEEDS, RECOVERED FROM THE RUN'S OWN FOUR THROUGHPUT KEYS
    # RATHER THAN FROM A CONSTANT. The callback logs flopsPS, TPS, BPS and MFU on the same row
    # for the same step, and they are algebraically over-determined: flopsPS/TPS is the model's
    # num_flops_per_token, TPS/BPS is the tokens on this device this step, and
    # flopsPS/(MFU/100) is the peak the callback divided by. So the denominator that has been
    # wrong twice on this table can be read straight out of the run and matched against a
    # datasheet, with nothing assumed about the shape.
    reported_mfu = None
    flops_per_token = None
    implied_peak = None
    tokens_per_device_step = None
    timed = [r for r in rows if r.get("throughput/device/BPS") and r.get("throughput/device/MFU")]
    if timed:
        last = timed[-1]
        reported_mfu = float(last["throughput/device/MFU"])
        tps = float(last["throughput/device/TPS"])
        flops_ps = float(last["throughput/device/flopsPS"])
        flops_per_token = flops_ps / tps
        implied_peak = flops_ps / (reported_mfu / 100.0)
        tokens_per_device_step = tps / float(last["throughput/device/BPS"])

    return CellHealth(
        cell=index,
        state=run.state,
        step=int(summary_step) if summary_step is not None else max(history_step, 0),
        summary_lost_its_step=summary_step is None and history_step >= 0,
        init_seed=config.get("init_seed"),
        model_init_seed=model.get("init_seed"),
        data_seed=loader.get("seed"),
        first_loss=finite[0] if finite else None,
        last_loss=finite[-1] if finite else None,
        nonfinite_losses=len(losses) - len(finite),
        z_loss_seen=any(_positive(r.get("train/Z loss")) for r in rows),
        loss_at=loss_at,
        evaluations=evaluations,
        arms_consistent=arms_consistent_with(config),
        sources=sources,
        bpb_sources=bpb,
        step_time=step_time,
        reported_mfu=reported_mfu,
        flops_per_token=flops_per_token,
        implied_peak_flops=implied_peak,
        device_name=str((run.metadata or {}).get("gpu") or ""),
        tokens_per_device_step=tokens_per_device_step,
        eval_seconds=statistics.median(steady_state_evals) if steady_state_evals else None,
    )


def report(
    cells: Sequence[CellHealth],
    *,
    expected_cells: int,
    arm_name: str,
    bound_hours: float,
    expected_sources: Sequence[str] = HELD_OUT_SOURCES,
    min_evaluations: int = 2,
) -> int:
    """
    Print the gate, and say go or no-go.

    :param min_evaluations: How many held-out evaluations a cell has to have completed before
        the gate will answer at all. Two, because the first one runs on startup before a single
        optimizer step and therefore says nothing about whether *training* is sound; the second
        is the step-500 pass, which is the first evidence that the evaluator survives a real
        checkpoint boundary and that the seven per-source metrics keep arriving.

    :returns: 0 when every check the gate rests on has passed, 1 when one has failed, and 2
        when the run has not gone far enough to tell -- which is not the same answer and must
        not be reported as one.
    """
    print(f"{len(cells)} of {expected_cells} cell(s) have reached W&B\n")
    if not cells:
        print("VERDICT: too early. Nothing has started.")
        return 2

    print(
        f"{'cell':>4}  {'state':<9} {'step':>6}  {'seed':>4}  {'evals':>5}  "
        f"{'loss':>17}  {'s/step':>8} {'n':>5}"
    )
    for c in cells:
        loss = (
            f"{c.first_loss:6.3f} -> {c.last_loss:6.3f}"
            if c.first_loss is not None and c.last_loss is not None
            else "-"
        )
        st = f"{c.step_time.median:8.3f} {c.step_time.rows_used:5d}" if c.step_time else " " * 14
        print(
            f"{c.cell:>4}  {c.state:<9} {c.step:>6}  {str(c.model_init_seed):>4}  "
            f"{c.evaluations:>5}  {loss:>17}  {st}"
            + ("  [step and sources from history]" if c.summary_lost_its_step else "")
        )
    print()

    overwritten = [c for c in cells if c.summary_lost_its_step]
    if overwritten:
        print(
            "cell(s) "
            + ", ".join(str(c.cell) for c in overwritten)
            + " have a W&B summary carrying no step, so everything above about them was read "
            "from their history instead. Their summaries were overwritten and are not evidence "
            "of anything, in either direction -- least of all that the cell never started.\n"
        )

    failures: List[str] = []
    holds: List[str] = []

    dead = [c for c in cells if c.state not in ("running", "finished")]
    if dead:
        failures.append(
            "cell(s) " + ", ".join(f"{c.cell} ({c.state})" for c in dead) + " are not running"
        )
    if len(cells) < expected_cells:
        holds.append(f"only {len(cells)} of {expected_cells} cells have started")
    matured = [c for c in cells if c.evaluations >= min_evaluations]
    if not matured:
        holds.append(
            f"no cell has completed {min_evaluations} held-out evaluations yet, so the "
            "step-500 pass is unproven"
        )

    distinct, why = seeds_are_distinct(list(cells))
    print(f"[seeds]        {'ok ' if distinct else 'FAIL'}  {why}")
    if not distinct:
        failures.append("the five cells are not five replicates")

    # THE OTHER HALF OF WHAT `resolve_cell` COULD GET WRONG. The seeds check catches five cells
    # that became one replicate; this catches five cells that became the wrong arm, which is
    # equally silent and which no loss curve would reveal.
    wrong = [c for c in cells if arm_name not in c.arms_consistent]
    ambiguous = [c for c in cells if len(c.arms_consistent) > 1]
    print(
        f"[arm]          {'ok ' if not wrong else 'FAIL'}  "
        f"{len(cells) - len(wrong)}/{len(cells)} cell(s) logged a config consistent with "
        f"'{arm_name}'" + (f", {len(ambiguous)} ambiguous" if ambiguous else "")
    )
    if wrong:
        failures.append(
            "cell(s) "
            + ", ".join(f"{c.cell} ({', '.join(c.arms_consistent) or 'no match'})" for c in wrong)
            + f" did not run '{arm_name}'"
        )

    unhealthy = [c for c in cells if not c.loss_is_healthy]
    nonfinite = sum(c.nonfinite_losses for c in cells)
    print(
        f"[loss]         {'ok ' if not unhealthy else 'FAIL'}  "
        f"{len(cells) - len(unhealthy)}/{len(cells)} decreasing, {nonfinite} non-finite value(s)"
    )
    if unhealthy:
        failures.append(f"{len(unhealthy)} cell(s) are not decreasing or hold a non-finite loss")

    z = [c for c in cells if c.z_loss_seen]
    print(
        f"[z-loss]       {'ok ' if len(z) == len(cells) else 'FAIL'}  logged by {len(z)}/{len(cells)}"
    )
    if len(z) != len(cells):
        failures.append("z-loss is not being written, so the configured 1e-5 is not in force")

    evaluated = [c for c in cells if c.sources]
    if not evaluated:
        print("[held-out]     ..    no cell has reached its first evaluation yet")
        holds.append("no held-out evaluation has landed")
    else:
        for c in evaluated:
            missing = sorted(set(expected_sources) - set(c.sources))
            ok = not missing and len(c.sources) == len(expected_sources)
            mark = "ok " if ok else "FAIL"
            print(
                f"[held-out]     {mark}  cell {c.cell}: {len(c.sources)} source(s) {', '.join(c.sources)}"
            )
            if missing:
                failures.append(f"cell {c.cell} is missing held-out source(s) {missing}")
            if set(c.bpb_sources) != set(c.sources):
                failures.append(f"cell {c.cell} has CE without BPB beside it")
        print(
            f"[bits-per-byte] ok  BPB beside CE on "
            f"{sum(1 for c in evaluated if set(c.bpb_sources) == set(c.sources))}"
            f"/{len(evaluated)} evaluated cell(s)"
        )

    timed = [c for c in cells if c.step_time and c.step_time.enough_rows]
    if not timed:
        print("[throughput]   ..    not enough clean rows yet to quote a median")
        holds.append("no cell has enough clean steps for a median")
        print("\nVERDICT: too early to gate. " + "; ".join(holds))
        return 2

    slowest = max(timed, key=lambda c: c.step_time.median)  # type: ignore[union-attr]
    medians = [c.step_time.median for c in timed]  # type: ignore[union-attr]
    print(
        f"[throughput]   ok   median {statistics.median(medians):.3f} s/step across "
        f"{len(timed)} cell(s), slowest {slowest.step_time.median:.3f} "  # type: ignore[union-attr]
        f"over {slowest.step_time.rows_used} clean rows"  # type: ignore[union-attr]
    )
    for c in timed:
        st = c.step_time
        assert st is not None
        detail = ", ".join(f"{n} {reason}" for reason, n in st.excluded.items()) or "nothing"
        print(
            f"                 cell {c.cell}: {st.median:.3f} s over {st.rows_used}/{st.rows_total} "
            f"rows (IQR {st.iqr:.3f}), dropped {detail}"
        )
        if st.outliers_remaining:
            print(
                f"                 cell {c.cell}: {st.outliers_remaining} clean row(s) still "
                f"over {OUTLIER_RATIO}x the median -- an instrument fires on a schedule this "
                "filter does not know about"
            )

    # POOLED ACROSS CELLS, AND ONLY FROM THE ONES PAST THEIR STARTUP PASS. A cell that has run
    # a single evaluation has measured the cold one, which costs about twice the steady-state
    # figure; taking its number would charge the whole run at the warm-up rate. Two of the five
    # cells are far enough along to have a real measurement, and it is the same evaluation over
    # the same seven shards on the same shape for all five.
    measured_evals = [
        c.eval_seconds for c in cells if c.eval_seconds is not None and c.evaluations > 1
    ]
    eval_seconds = (
        statistics.median(measured_evals)
        if measured_evals
        else hyper_connection_arms.MEASURED_EVAL_SECONDS
    )
    projection = project(
        arm_name,
        seconds_per_step=slowest.step_time.median,  # type: ignore[union-attr]
        bound_hours=bound_hours,
        eval_seconds=eval_seconds,
    )
    mark = "ok " if projection.fits else "FAIL"
    print(
        f"[fit]          {mark}  {projection.hours:.2f} h projected against a "
        f"{bound_hours:.0f} h bound, {100 * projection.spare_fraction:+.1f}% spare "
        f"(at the slowest cell, evaluations charged at a measured {eval_seconds:.0f} s)"
    )
    if not projection.fits:
        failures.append(
            f"the projection is {projection.hours:.2f} h against a {bound_hours:.0f} h bound"
        )

    for c in timed:
        if c.flops_per_token is None or c.tokens_per_device_step is None:
            continue
        datasheet = peak_bf16_flops(c.device_name)
        if datasheet is None:
            # A CELL WHOSE SUMMARY WENT ALSO LOST ITS METADATA, AND THE TWO REFUSALS READ THE
            # SAME WHILE MEANING OPPOSITE THINGS. "This card is not in the table" is a gap in
            # the table; "the metadata was overwritten" is a gap in the record, and the card
            # is whatever its siblings in the same fan-out report. Only the first is a reason
            # to distrust the run.
            why = (
                "its W&B metadata was overwritten along with its summary, so the card it ran "
                "on is not recorded here -- read it off a sibling cell of the same submission"
                if c.summary_lost_its_step
                else f"'{c.device_name}' is not a part this module has a dense BF16 figure for"
            )
            print(f"[mfu]          FAIL  cell {c.cell}: {why}, so its MFU cannot be checked")
            failures.append(
                f"cell {c.cell} ran on '{c.device_name}', which matches no entry in "
                "DENSE_BF16_PEAK_FLOPS -- and which SpeedMonitorCallback's `else` therefore "
                f"scored against the A100's {A100_BF16_DENSE_FLOPS / 1e12:.0f} TF regardless"
            )
            continue
        hand = mfu_percent(
            flops_per_token=c.flops_per_token,
            tokens_per_device_step=c.tokens_per_device_step,
            seconds_per_step=c.step_time.median,  # type: ignore[union-attr]
            peak_flops=datasheet,
        )
        peak_ok = (
            c.implied_peak_flops is not None
            and abs(c.implied_peak_flops - datasheet) / datasheet < 1e-3
        )
        print(
            f"[mfu]          {'ok ' if peak_ok else 'FAIL'}  cell {c.cell}: {hand:.2f}% by hand "
            f"at the clean median, {c.reported_mfu:.2f}% reported at the last step; "
            f"{(c.implied_peak_flops or 0) / 1e12:.2f} TF implied against a datasheet "
            f"{datasheet / 1e12:.2f} TF for {c.device_name}"
        )
        if not peak_ok:
            failures.append(
                f"cell {c.cell} is scored against {(c.implied_peak_flops or 0) / 1e12:.2f} TF, "
                f"not the {c.device_name}'s {datasheet / 1e12:.2f} TF dense BF16. THE HAND "
                f"CALCULATION WINS: the true MFU is {hand:.2f}%."
            )

    print()
    if failures:
        print("VERDICT: NO-GO for stage 2.")
        for f in failures:
            print(f"  - {f}")
        return 1
    if holds:
        print("VERDICT: nothing is wrong, but the gate is not yet answerable. " + "; ".join(holds))
        return 2
    print("VERDICT: GO for stage 2 on the evidence above.")
    return 0


def self_test() -> int:
    """
    Check the estimators against planted data, with no network and no W&B.

    The filter is the part worth testing: it is the one piece of arithmetic here that has
    already been got wrong once on this branch, in both directions.
    """
    clean, evaluated, monitored = 8.2, 112.0, 9.6
    rows: List[Dict[str, object]] = []
    for step in range(0, 60):
        seconds = clean
        row: Dict[str, object] = {"_step": step}
        if step % 25 == 0 and step:
            seconds = evaluated
            row["throughput/in-loop eval time (s)"] = 104.0
        elif step % 10 == 0 and step:
            seconds = monitored
            row["hc/block 00/lane norm spread"] = 0.04
        row["throughput/device/BPS"] = 1.0 / seconds
        rows.append(row)

    got = clean_step_seconds(rows, eval_interval=25)
    assert abs(got.median - clean) < 1e-9, f"median {got.median} is not the planted {clean}"
    assert got.outliers_remaining == 0, "an instrument row survived the filter"
    assert got.rows_used < got.rows_total, "nothing was excluded, so nothing was filtered"
    assert got.excluded.get("lane monitor"), "the monitor rows were not recognised"
    assert got.excluded.get("held-out evaluation"), "the evaluation rows were not recognised"

    # The failure that motivates the whole filter: a median over the instrument's own rows.
    monitor_only = [r for r in rows if any(str(k).startswith("hc/") for k in r)]
    naive = statistics.median([1.0 / float(r["throughput/device/BPS"]) for r in monitor_only])
    assert abs(naive - monitored) < 1e-9, "the planted trap does not reproduce"
    assert naive > got.median, "the trap should read slower than a clean step"

    # And a median over everything, which the evaluation pulls up instead.
    everything = statistics.median([1.0 / float(r["throughput/device/BPS"]) for r in rows])
    assert everything >= got.median

    hand = mfu_percent(
        flops_per_token=3_032_684_544,
        tokens_per_device_step=196_608,
        seconds_per_step=8.2188,
        peak_flops=L40S_BF16_DENSE_FLOPS,
    )
    assert 39.9 < hand < 40.2, f"L40S MFU hand calculation drifted to {hand}"

    # The L40S run as it was actually reported, against the peak the card cannot reach.
    as_reported = mfu_percent(
        flops_per_token=3_032_684_544,
        tokens_per_device_step=196_608,
        seconds_per_step=8.2188,
        peak_flops=362.05e12,
    )
    assert abs(as_reported * 2 - hand) < 1e-6, "the FP32-accumulate correction is a clean 2x"

    # The A100 cells, at half the tokens per device for the same global batch.
    a100 = mfu_percent(
        flops_per_token=3_032_684_544,
        tokens_per_device_step=98_304,
        seconds_per_step=1.69,
        peak_flops=A100_BF16_DENSE_FLOPS,
    )
    assert 56.0 < a100 < 57.5, f"A100 MFU hand calculation drifted to {a100}"

    pre_v250 = mfu_percent(
        flops_per_token=3_032_684_544,
        tokens_per_device_step=98_304,
        seconds_per_step=1.69,
        peak_flops=A100_BF16_DENSE_FLOPS / 2,
    )
    assert abs(pre_v250 - 2 * a100) < 1e-6, "the pre-v2.5.0 A100 constant inflated MFU by 2x"

    assert peak_bf16_flops("NVIDIA A100-SXM4-40GB") == A100_BF16_DENSE_FLOPS
    assert peak_bf16_flops("NVIDIA L40S") == L40S_BF16_DENSE_FLOPS
    assert peak_bf16_flops("NVIDIA GeForce RTX 4090") is None, "an unknown part must not default"

    fits = project("baseline", seconds_per_step=8.2188, bound_hours=19.0)
    assert fits.fits, "the measured step time should fit the bound"
    assert not project("baseline", seconds_per_step=12.0, bound_hours=19.0).fits

    def cell(index: int, *, seed: int, offset: float, upto: int = 100) -> CellHealth:
        curve = {s: 10.0 - s * 0.04 + offset for s in range(upto + 1)}
        return CellHealth(
            cell=index,
            state="running",
            step=upto,
            model_init_seed=seed,
            first_loss=curve[0],
            last_loss=curve[upto],
            loss_at=curve,
        )

    good = [cell(i, seed=i, offset=i * 0.01) for i in range(5)]
    assert seeds_are_distinct(good)[0]

    collapsed = [cell(i, seed=0, offset=0.0) for i in range(5)]
    assert not seeds_are_distinct(collapsed)[0], "five identical seeds must not pass"

    same_curve = [cell(i, seed=i, offset=0.0) for i in range(5)]
    assert not seeds_are_distinct(same_curve)[0], "distinct seeds on one curve must not pass"

    # THE TRAP THE SHARED-STEP COMPARISON EXISTS FOR. Five bit-identical replicates that
    # started minutes apart are at different points on one curve, so their *latest* losses all
    # differ and a naive check calls them distinct. Compared at a step they share, they do not.
    staggered = [cell(i, seed=i, offset=0.0, upto=100 - 10 * i) for i in range(5)]
    assert len({c.last_loss for c in staggered}) == 5, "the planted trap does not reproduce"
    assert not seeds_are_distinct(staggered)[0], "a staggered identical run must not pass"

    print("self-test OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--run", help="The platform run id of the fan-out.")
    parser.add_argument("--cells", type=int, default=5)
    parser.add_argument("--arm", default="baseline")
    parser.add_argument("--bound-hours", type=float, default=19.0)
    parser.add_argument(
        "--eval-interval", type=int, default=hyper_connection_arms.TRANCHE_EVAL_INTERVAL
    )
    parser.add_argument(
        "--min-evaluations",
        type=int,
        default=2,
        help="Held-out evaluations a cell must have finished before the gate will answer. "
        "The first runs on startup, so two means the step-500 pass has landed.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Re-read every SECONDS until the gate answers go or no-go rather than 'too early'.",
    )
    parser.add_argument("--self-test", action="store_true")
    opts = parser.parse_args()

    if opts.self_test:
        return self_test()
    if not opts.run:
        parser.error("--run is required unless --self-test")

    while True:
        cells = read_cells(
            opts.entity, opts.project, opts.run, opts.cells, eval_interval=opts.eval_interval
        )
        print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        status = report(
            cells,
            expected_cells=opts.cells,
            arm_name=opts.arm,
            bound_hours=opts.bound_hours,
            min_evaluations=opts.min_evaluations,
        )
        if status != 2 or not opts.watch:
            return status
        time.sleep(opts.watch)


if __name__ == "__main__":
    raise SystemExit(main())
