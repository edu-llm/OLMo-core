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
                time rather than trusted. See :func:`mfu_percent`.

    python .edullm/stage_gate.py --self-test                    # no network
    python .edullm/stage_gate.py --run run_019fe279-4ef0 --cells 5
    python .edullm/stage_gate.py --run run_019fe279-4ef0 --cells 5 --watch 300
"""

import argparse
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

#: Dense BF16 peak for one L40S, in FLOP/s, and the denominator every MFU here is taken
#: against. NVIDIA's own datasheet publishes ``362.05 | 733*`` for BFLOAT16 Tensor Core, where
#: the starred figure is with structural sparsity -- so 362.05 is already the dense number and
#: halving it, which is what the surrounding branches of ``SpeedMonitorCallback`` do to a
#: sparse-quoted spec, would be wrong here. Kept as a literal rather than imported from the
#: callback: this module exists to check that constant, and a check that reads the value it is
#: checking is not one.
L40S_BF16_DENSE_FLOPS: float = 362.05e12

#: Ranks per node on ``gpu-4xl40s``. The speed monitor logs *per device*, so a step time read
#: back out of ``throughput/device/TPS`` needs this to reach the global batch.
RANKS_PER_NODE = 4

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
    tokens_per_device_step: int,
    seconds_per_step: float,
    peak_flops: float = L40S_BF16_DENSE_FLOPS,
) -> float:
    """
    Model FLOPs utilization, computed the way ``SpeedMonitorCallback`` computes it.

    Written out here so that the reported figure can be checked against something rather than
    read. There is history: OLMo-core v2.5.0 fixed an A100 peak-FLOPs constant in that callback
    that was 2x too low and had been inflating every MFU it reported by 2x, and until this
    branch added an L40S branch to the same table an L40S fell through to that A100 default and
    was scored against 312 TF instead of 362.05.

    :param flops_per_token: The model's own ``num_flops_per_token`` at the run's sequence
        length. Not 6N: it counts attention, and on a hyper-connection arm it counts the lanes.
    :param tokens_per_device_step: Global batch over the number of ranks.
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
) -> Projection:
    """
    Price a cell at a measured step time.

    Delegates to :func:`hyper_connection_arms.arm_seconds` rather than re-adding the constants,
    so the projection and the submission cannot disagree about how many evaluations there are.
    That function counts fourteen -- one on startup, one on finish and one every 500 steps --
    and thirteen checkpoints, and it charges the lane monitor only to an arm that has lanes,
    which the baseline does not.

    :param arm_name: The arm, as a key of :data:`hyper_connection_arms.ARMS`.
    :param seconds_per_step: The measured clean median.
    :param steps: The horizon.
    :param bound_hours: ``--hours`` as submitted.
    """
    seconds = hyper_connection_arms.arm_seconds(
        ARMS[arm_name], steps=steps, seconds_per_step=seconds_per_step
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

    It does not separate every arm in the table and does not pretend to. ``decay-everything``
    differs from ``faithful`` only in the optimizer and ``tied-faithful`` only in block reuse,
    so both come back alongside it. All of those are unfunded; the four that run are unique.

    :param config: A run's W&B config.

    :returns: Every arm name whose lane configuration matches, in table order. A single-element
        result identifies the arm; more than one means the config cannot tell them apart.
    """
    block = config.get("model")
    block = (block or {}).get("block") if isinstance(block, dict) else None  # type: ignore[union-attr]
    logged = block.get("hyper_connections") if isinstance(block, dict) else None

    if not isinstance(logged, dict):
        return tuple(name for name, arm in ARMS.items() if arm.hyper_connections is None)

    matches: List[str] = []
    for name, arm in ARMS.items():
        if arm.hyper_connections is None:
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

    summary_keys = [k for k in run.summary.keys() if not str(k).startswith("_")]
    sources = tuple(
        sorted(
            k.split("/")[2]
            for k in summary_keys
            if k.startswith("eval/lm/") and k.endswith("/CE loss")
        )
    )
    bpb = tuple(
        sorted(
            k.split("/")[2] for k in summary_keys if k.startswith("eval/lm/") and k.endswith("/BPB")
        )
    )

    # NO ``keys=`` FILTER, AND THAT IS NOT LAZINESS. Asking ``scan_history`` for a named list
    # returns zero rows on a *running* run as soon as the list contains
    # "throughput/in-loop eval time (s)" -- the same call against the same key returns rows
    # once the run has finished. A filter that silently empties itself while the run is live
    # is the worst possible failure here, because this module only ever runs while the run is
    # live, and it would report a healthy fan-out as one with no losses and no z-loss. The
    # unfiltered scan is also cheap: a finished 6,000-step cell is a few thousand rows.
    rows = list(run.scan_history(page_size=2000))

    losses = [r["train/CE loss"] for r in rows if r.get("train/CE loss") is not None]
    finite = [x for x in losses if math.isfinite(x)]
    loss_at = {
        int(r["_step"]): float(r["train/CE loss"])
        for r in rows
        if r.get("train/CE loss") is not None
        and r.get("_step") is not None
        and math.isfinite(float(r["train/CE loss"]))
    }
    evaluations = sum(1 for r in rows if _positive(r.get("throughput/in-loop eval time (s)")))
    step_time: Optional[CleanStepTime] = None
    if rows:
        try:
            step_time = clean_step_seconds(rows, eval_interval=eval_interval)
        except ValueError:
            step_time = None

    reported_mfu = None
    flops_per_token = None
    implied_peak = None
    timed = [r for r in rows if r.get("throughput/device/BPS") and r.get("throughput/device/MFU")]
    if timed:
        last = timed[-1]
        reported_mfu = float(last["throughput/device/MFU"])
        tps = float(last["throughput/device/TPS"])
        flops_ps = float(last["throughput/device/flopsPS"])
        flops_per_token = flops_ps / tps
        implied_peak = flops_ps / (reported_mfu / 100.0)

    return CellHealth(
        cell=index,
        state=run.state,
        step=int(run.summary.get("_step") or 0),
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
        )
    print()

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

    projection = project(
        arm_name, seconds_per_step=slowest.step_time.median, bound_hours=bound_hours  # type: ignore[union-attr]
    )
    mark = "ok " if projection.fits else "FAIL"
    print(
        f"[fit]          {mark}  {projection.hours:.2f} h projected against a "
        f"{bound_hours:.0f} h bound, {100 * projection.spare_fraction:+.1f}% spare"
    )
    if not projection.fits:
        failures.append(
            f"the projection is {projection.hours:.2f} h against a {bound_hours:.0f} h bound"
        )

    for c in timed:
        if c.flops_per_token is None or c.reported_mfu is None:
            continue
        hand = mfu_percent(
            flops_per_token=c.flops_per_token,
            tokens_per_device_step=hyper_connection_arms.TRANCHE_TOKENS_PER_STEP // RANKS_PER_NODE,
            seconds_per_step=c.step_time.median,  # type: ignore[union-attr]
            peak_flops=L40S_BF16_DENSE_FLOPS,
        )
        peak_ok = (
            c.implied_peak_flops is not None
            and abs(c.implied_peak_flops - L40S_BF16_DENSE_FLOPS) / L40S_BF16_DENSE_FLOPS < 1e-3
        )
        print(
            f"[mfu]          {'ok ' if peak_ok else 'FAIL'}  cell {c.cell}: "
            f"{hand:.2f}% by hand at the clean median, peak "
            f"{(c.implied_peak_flops or 0) / 1e12:.2f} TF implied by the run"
        )
        if not peak_ok:
            failures.append(
                f"cell {c.cell} is scored against {(c.implied_peak_flops or 0) / 1e12:.2f} TF, "
                f"not the L40S's {L40S_BF16_DENSE_FLOPS / 1e12:.2f} TF dense BF16"
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
    assert 19.9 < hand < 20.2, f"MFU hand calculation drifted to {hand}"

    doubled = mfu_percent(
        flops_per_token=3_032_684_544,
        tokens_per_device_step=196_608,
        seconds_per_step=8.2188,
        peak_flops=312e12 * 0.5,
    )
    assert doubled > 2 * hand, "the historical A100 constant should inflate MFU, and by a lot"

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
