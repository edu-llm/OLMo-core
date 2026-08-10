#!/usr/bin/env python3
"""Recompute the 370M endpoint contrasts from per-step history, by a second route.

WHAT THIS IS FOR. ``analysis.py`` computes the headline numbers for this branch and is the
instrument of record. This is a second instrument, written without reading the first, so that
the arithmetic behind a claim that contradicts a published result has been done twice from the
raw per-cell numbers rather than once. It shares no code with ``analysis.py`` and it is not a
replacement for it: where the two disagree, the disagreement is the output.

IT READS HISTORY AND NEVER SUMMARIES. A W&B run summary is the last value the process wrote,
and a process that died writing one leaves a summary that reads as if the run never trained.
Seven of them on this branch did. :func:`read_cell` therefore scans the full history and takes
the eval rows out of it, which also means a cell that stopped early is *visible* as a cell whose
last eval is at step 4,500 rather than silently contributing its step-4,500 number to a
step-6,000 mean. :func:`endpoint_table` refuses to build a table at a step that any requested
cell does not reach, because a contrast that mixes horizons is not a contrast.

THE UNITS. The endpoint is the unweighted mean of held-out bits-per-byte over the seven
sources. Nats of cross-entropy per token are ``BPB * ln(2) * bytes_per_token`` and this branch
uses 4.57 bytes per token. That constant is not taken on trust: the trainer logs ``CE loss``
beside ``BPB`` for every source, their ratio is ``ln(2) * bytes_per_token`` exactly, and
:func:`implied_bytes_per_token` reads it back off every cell.

FOUR INTERVALS, NOT ONE, AND THEY DISAGREE. The design is four arms over five seeds, and the
seeds are matched across arms -- ``init_seed`` 12536-12540 against data-loader seeds 0-4, the
same five in every arm. That makes at least four defensible standard errors for the same
contrast, and on these data they differ by a factor of 1.35:

* ``pooled``   -- one-way pooled sigma over the complete arms, df ``N - k``. Ignores the
  matching.
* ``blocked``  -- residual of an arm-plus-seed model, df ``(k - 1)(n - 1)``. Exploits the
  matching, and assumes one common residual variance and one common cross-arm correlation.
* ``paired``   -- the t interval on that contrast's own five paired differences, df ``n - 1``.
  Assumes nothing about the other arms.
* ``welch``    -- unpaired, unequal variances. Assumes nothing about the matching.

The report prints all four, because quoting one sigma and building the interval out of a
different one is the specific error this file was written to catch, and because on the H1
contrast the matching buys nothing: the cross-arm seed correlation is 0.25 there against 0.65
and 0.75 on the other two pairs, so the single pooled residual that the blocked interval rests
on is not something these data support.

Usage::

    python .edullm/verify_endpoint.py --self-test              # no network
    python .edullm/verify_endpoint.py                          # report from the frozen file
    python .edullm/verify_endpoint.py --fetch                  # refresh it from W&B, then report

``--fetch`` is the only path that reaches the network and it only ever reads. The frozen file
is ``.edullm/endpoint-verification.json``; unlike a history cache it is committed, because it
is the evidence for the numbers in the report and it is forty kilobytes.
"""

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: Where the fetched endpoint lives once frozen.
FROZEN = os.path.join(_HERE, "endpoint-verification.json")

#: Bytes per token for this corpus, as passed to every cell as ``--bytes-per-token``.
BYTES_PER_TOKEN = 4.57

#: One bit per byte in nats per token. Checked against the logged ``CE loss / BPB`` ratio.
NATS_PER_BPB = math.log(2.0) * BYTES_PER_TOKEN

#: The seven held-out sources, in the order the endpoint averages them. Unweighted.
SOURCES = (
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
)

#: The four arms of stage 2 on 8xA100, by the submission that carries them. The ids are the
#: full W&B run ids; cells are ``<id>-cell-<index>``.
SUBMISSIONS = {
    "baseline": "run_019fe40f-c71e-7045-9b58-537b9e2f6cb4",
    "faithful": "run_019fe90b-f99e-7027-b835-5d91feda869c",
    "output-only": "run_019fe9f6-d060-7016-8bd9-e3b1b21d6638",
    "mhc": "run_019fe9f6-d4f8-7077-841f-5da0312c3603",
}

#: The pre-registered horizon. Every contrast is taken here or not at all.
FINAL_STEP = 6000

#: Steps between held-out evaluations, and the subsampling grid of the frozen document.
EVAL_INTERVAL = 500

#: Cells per arm.
N_SEEDS = 5

#: The contrasts the branch claims, as ``(treatment, control)``. H1, H2a, and the post-hoc one.
CONTRASTS = (
    ("faithful", "baseline"),
    ("faithful", "output-only"),
    ("output-only", "baseline"),
)

BPB_KEY = "eval/lm/{source}/BPB"
CE_KEY = "eval/lm/{source}/CE loss"
TRAIN_CE_KEY = "train/CE loss"
STEP_KEY = "_step"
RUNTIME_KEY = "_runtime"


@dataclass(frozen=True)
class Interval:
    """A two-sided interval on a contrast, and the assumption that produced it."""

    method: str
    point: float
    half_width: float
    sigma: float
    df: float

    @property
    def lo(self) -> float:
        return self.point - self.half_width

    @property
    def hi(self) -> float:
        return self.point + self.half_width

    def excludes(self, value: float) -> bool:
        """Whether ``value`` lies outside the interval. Used against zero and against 0.030."""
        return not (self.lo <= value <= self.hi)


def c4(df: int) -> float:
    """
    The bias factor of a sample standard deviation: ``E[s] = c4(df) * sigma``.

    It is a function of the degrees of freedom of *that* estimate and not of the seed count,
    which is the trap. A five-seed arm has df 4 and ``c4 = 0.9400``; a sigma pooled over three
    such arms has df 12 and ``c4 = 0.9794``. Dividing a df-12 pool by 0.9400 inflates it by 4%.

    :param df: Degrees of freedom of the estimate.

    :returns: The multiplicative bias, in ``(0, 1)``.
    """
    return math.sqrt(2.0 / df) * math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2))


def implied_bytes_per_token(ce_loss: float, bpb: float) -> float:
    """
    Read the bytes-per-token constant back out of a cell's own logged numbers.

    The trainer computes ``BPB = CE / (ln(2) * bytes_per_token)``, so the ratio of the two
    metrics it logs for the same source at the same step recovers the constant exactly. A cell
    that was launched with a different value shows up here rather than in a silent 1% shift.

    :param ce_loss: Cross-entropy in nats per token.
    :param bpb: Bits per byte for the same source and step.

    :returns: Bytes per token.
    """
    return ce_loss / bpb / math.log(2.0)


def unweighted_endpoint(per_source: Mapping[str, float]) -> float:
    """
    The primary endpoint: the mean of the seven per-source BPB values, unweighted.

    :param per_source: One value per name in :data:`SOURCES`.

    :returns: Bits per byte.

    :raises KeyError: If any source is missing, which is the only safe response -- an endpoint
        over six sources is a different endpoint.
    """
    return sum(per_source[s] for s in SOURCES) / len(SOURCES)


def to_nats(bpb: float) -> float:
    """Convert bits per byte to nats of cross-entropy per token at 4.57 bytes per token."""
    return bpb * NATS_PER_BPB


def pooled_sigma(groups: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """
    One-way pooled within-arm standard deviation.

    :param groups: One sequence of cell values per arm. Arms may differ in size.

    :returns: ``(sigma, df)`` with ``df = N - k``.
    """
    ss = 0.0
    df = 0
    for g in groups:
        if len(g) < 2:
            continue
        m = statistics.fmean(g)
        ss += sum((x - m) ** 2 for x in g)
        df += len(g) - 1
    return math.sqrt(ss / df), df


def blocked_residual_sigma(table: Mapping[str, Sequence[float]]) -> Tuple[float, int]:
    """
    Residual standard deviation of an arm-plus-seed additive model, on a complete table.

    This is the estimator that exploits the matched seeds, and its residual is the arm-by-seed
    interaction. It is smaller than :func:`pooled_sigma` exactly when the seeds carry a common
    main effect, and it is only honest when the cross-arm correlations are alike.

    :param table: Arm name to that arm's values, in a consistent seed order. Every arm must
        have the same number of cells.

    :returns: ``(sigma, df)`` with ``df = (k - 1)(n - 1)``.

    :raises ValueError: If the table is ragged.
    """
    arms = list(table)
    n = len(table[arms[0]])
    if any(len(table[a]) != n for a in arms):
        raise ValueError("blocked residual needs a complete arm-by-seed table")
    k = len(arms)
    grand = statistics.fmean([x for a in arms for x in table[a]])
    arm_means = {a: statistics.fmean(table[a]) for a in arms}
    seed_means = [statistics.fmean([table[a][s] for a in arms]) for s in range(n)]
    ss_total = sum((table[a][s] - grand) ** 2 for a in arms for s in range(n))
    ss_arm = n * sum((arm_means[a] - grand) ** 2 for a in arms)
    ss_seed = k * sum((m - grand) ** 2 for m in seed_means)
    df = (k - 1) * (n - 1)
    return math.sqrt((ss_total - ss_arm - ss_seed) / df), df


def seed_effect_f(table: Mapping[str, Sequence[float]]) -> Tuple[float, float]:
    """
    Test whether the seeds carry a main effect, which is the licence for blocking on them.

    :param table: As for :func:`blocked_residual_sigma`.

    :returns: ``(F, p)`` on ``(n - 1, (k - 1)(n - 1))`` degrees of freedom. A table with no
        residual at all -- two arms that move together exactly -- gives infinity and zero
        rather than a division error.
    """
    arms = list(table)
    n = len(table[arms[0]])
    k = len(arms)
    grand = statistics.fmean([x for a in arms for x in table[a]])
    seed_means = [statistics.fmean([table[a][s] for a in arms]) for s in range(n)]
    ss_seed = k * sum((m - grand) ** 2 for m in seed_means)
    sigma, df_res = blocked_residual_sigma(table)
    if sigma == 0.0:
        return (math.inf, 0.0) if ss_seed > 0 else (math.nan, math.nan)
    f = (ss_seed / (n - 1)) / (sigma**2)
    return f, float(stats.f.sf(f, n - 1, df_res))


def intervals(
    table: Mapping[str, Sequence[float]], treat: str, control: str, alpha: float = 0.05
) -> List[Interval]:
    """
    Every defensible two-sided interval on ``treat - control``, so the reader can see the spread.

    :param table: Arm name to values in a consistent seed order, complete.
    :param treat: Arm on the left of the difference.
    :param control: Arm on the right.
    :param alpha: Two-sided level.

    :returns: The pooled, blocked, paired and Welch intervals, in that order.
    """
    a, b = list(table[treat]), list(table[control])
    n = len(a)
    point = statistics.fmean(a) - statistics.fmean(b)
    diffs = [a[i] - b[i] for i in range(n)]
    out = []

    sig, df = pooled_sigma(list(table.values()))
    se = sig * math.sqrt(2.0 / n)
    out.append(Interval("pooled", point, float(stats.t.ppf(1 - alpha / 2, df)) * se, sig, df))

    sig, df = blocked_residual_sigma(table)
    se = sig * math.sqrt(2.0 / n)
    out.append(Interval("blocked", point, float(stats.t.ppf(1 - alpha / 2, df)) * se, sig, df))

    sig = statistics.stdev(diffs)
    se = sig / math.sqrt(n)
    out.append(Interval("paired", point, float(stats.t.ppf(1 - alpha / 2, n - 1)) * se, sig, n - 1))

    va, vb = statistics.variance(a) / n, statistics.variance(b) / n
    se = math.sqrt(va + vb)
    df_w = (va + vb) ** 2 / (va**2 / (n - 1) + vb**2 / (n - 1))
    out.append(
        Interval(
            "welch",
            point,
            float(stats.t.ppf(1 - alpha / 2, df_w)) * se,
            se * math.sqrt(n / 2),
            df_w,
        )
    )
    return out


def cross_arm_correlation(table: Mapping[str, Sequence[float]], one: str, two: str) -> float:
    """
    Seed-to-seed correlation between two arms, which is what pairing trades on.

    :param table: As for :func:`blocked_residual_sigma`.
    :param one: First arm.
    :param two: Second arm.

    :returns: Pearson r over the matched seeds.
    """
    return statistics.correlation(list(table[one]), list(table[two]))


def slope_per_doubling(steps: Sequence[int], values: Sequence[float]) -> float:
    """
    Improvement per doubling of step count, from a least-squares fit against ``log2(step)``.

    Signed so that a positive number means the loss is falling. Reported per cell and then
    averaged, rather than fitted to the arm mean, so that the spread across seeds is available.

    :param steps: Eval steps, all positive.
    :param values: The endpoint at those steps, in nats.

    :returns: Nats per doubling.
    """
    xs = [math.log2(s) for s in steps]
    return -float(stats.linregress(xs, list(values)).slope)


def read_cell(entity: str, project: str, run_id: str) -> Dict[str, object]:
    """
    Pull one cell's eval rows, seeds and wall clock out of W&B history.

    Nothing here touches ``run.summary``. The eval rows are the rows that carry a non-null BPB
    for the first source; a partial cell simply has fewer of them, and that is the signal the
    caller needs rather than something to paper over.

    :param entity: W&B entity.
    :param project: W&B project.
    :param run_id: Full cell id, ``<submission>-cell-<index>``.

    The whole history is scanned but only a thin slice is kept, because the frozen document is
    committed: the thirteen eval rows, the training loss and wall clock every ``EVAL_INTERVAL``
    steps, and one scalar for the training loss averaged over the last ``EVAL_INTERVAL`` steps.
    Six thousand rows per cell of per-step training loss is derived data with no second reader.

    :param entity: W&B entity.
    :param project: W&B project.
    :param run_id: Full cell id, ``<submission>-cell-<index>``.

    :returns: A dict with ``state``, ``init_seed``, ``data_loader_seed``, ``last_step``,
        ``evals`` (step to source to BPB, plus the dclm CE loss for the units check),
        ``train_ce_at`` and ``runtime_at`` (step to value, subsampled) and ``train_ce_tail``
        (the training loss averaged over the final stretch).
    """
    import wandb  # imported here so that --self-test needs no W&B install

    api = wandb.Api(timeout=120)
    run = api.run(f"{entity}/{project}/{run_id}")
    evals: Dict[str, Dict[str, float]] = {}
    train_ce_at: Dict[str, float] = {}
    runtime_at: Dict[str, float] = {}
    tail: List[float] = []
    last_step = 0
    for row in run.scan_history():
        step = row.get(STEP_KEY)
        if step is None:
            continue
        last_step = max(last_step, int(step))
        if row.get(BPB_KEY.format(source=SOURCES[0])) is not None:
            entry = {s: row[BPB_KEY.format(source=s)] for s in SOURCES}
            entry["dclm CE loss"] = row[CE_KEY.format(source="dclm")]
            evals[str(step)] = entry
        if row.get(TRAIN_CE_KEY) is not None:
            tail.append(row[TRAIN_CE_KEY])
            del tail[:-EVAL_INTERVAL]
            if int(step) % EVAL_INTERVAL == 0:
                train_ce_at[str(step)] = row[TRAIN_CE_KEY]
        if row.get(RUNTIME_KEY) is not None and int(step) % EVAL_INTERVAL == 0:
            runtime_at[str(step)] = row[RUNTIME_KEY]
    config = dict(run.config)
    return {
        "state": run.state,
        "init_seed": config.get("init_seed"),
        "data_loader_seed": (config.get("data_loader") or {}).get("seed"),
        "last_step": last_step,
        "evals": evals,
        "train_ce_at": train_ce_at,
        "runtime_at": runtime_at,
        "train_ce_tail": statistics.fmean(tail) if tail else None,
    }


def fetch(entity: str = DEFAULT_ENTITY, project: str = DEFAULT_PROJECT) -> Dict[str, object]:
    """
    Read every cell of every arm and assemble the frozen document.

    :param entity: W&B entity.
    :param project: W&B project.

    :returns: The document that :data:`FROZEN` holds.
    """
    cells: Dict[str, Dict[str, object]] = {}
    for arm, submission in SUBMISSIONS.items():
        for index in range(N_SEEDS):
            run_id = f"{submission}-cell-{index}"
            cell = read_cell(entity, project, run_id)
            cell["arm"] = arm
            cell["cell"] = index
            cells[run_id] = cell
            print(
                f"  {arm:<12} cell-{index} state={cell['state']:<9} "
                f"evals={len(cell['evals'])}",  # type: ignore[arg-type]
                file=sys.stderr,
            )
    return {
        "entity": entity,
        "project": project,
        "bytes_per_token": BYTES_PER_TOKEN,
        "final_step": FINAL_STEP,
        "eval_interval": EVAL_INTERVAL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": list(SOURCES),
        "submissions": dict(SUBMISSIONS),
        "cells": cells,
    }


def endpoint_table(
    doc: Mapping[str, object], arms: Sequence[str], step: int = FINAL_STEP
) -> Dict[str, List[float]]:
    """
    The endpoint in nats for each requested arm, in seed order, at one shared step.

    :param doc: The frozen document.
    :param arms: Arms to include.
    :param step: The step to read. The same one for every cell, or nothing is returned.

    :returns: Arm name to five values in nats, ordered by cell index.

    :raises ValueError: If any cell of any requested arm has no eval at ``step``. Falling back
        to that cell's last eval would put two horizons in one contrast.
    """
    cells = doc["cells"]
    table: Dict[str, List[float]] = {}
    for arm in arms:
        values: List[Tuple[int, float]] = []
        for run_id, cell in cells.items():  # type: ignore[union-attr]
            if cell["arm"] != arm:
                continue
            entry = cell["evals"].get(str(step))
            if entry is None:
                last = max(int(k) for k in cell["evals"])
                raise ValueError(
                    f"{run_id} has no eval at step {step}; its last is {last}. "
                    "Drop the arm or move the horizon; do not mix the two."
                )
            values.append((cell["cell"], to_nats(unweighted_endpoint(entry))))
        table[arm] = [v for _, v in sorted(values)]
    return table


def complete_arms(doc: Mapping[str, object], step: int = FINAL_STEP) -> List[str]:
    """
    The arms whose every cell reached ``step``, in the order of :data:`SUBMISSIONS`.

    :param doc: The frozen document.
    :param step: The horizon.

    :returns: Arm names.
    """
    out = []
    for arm in SUBMISSIONS:
        cells = [c for c in doc["cells"].values() if c["arm"] == arm]  # type: ignore[union-attr]
        if len(cells) == N_SEEDS and all(str(step) in c["evals"] for c in cells):
            out.append(arm)
    return out


def step_seconds(cell: Mapping[str, object], lo: int = 1000, hi: int = 5000) -> Optional[float]:
    """
    Marginal wall clock per step over a steady-state window.

    Deliberately not total runtime over total steps: that definition folds in process startup,
    the first checkpoint and every in-loop eval, and it is about 1% larger. Both are quoted in
    the report because they answer different questions.

    :param cell: One cell of the frozen document.
    :param lo: First step of the window.
    :param hi: Last step of the window.

    :returns: Seconds per step, or ``None`` if the cell never reached the window.
    """
    rt = cell["runtime_at"]  # type: ignore[index]
    inside = [s for s in sorted(int(k) for k in rt) if lo <= s <= hi]
    if len(inside) < 2:
        return None
    return (rt[str(inside[-1])] - rt[str(inside[0])]) / (inside[-1] - inside[0])


def load(path: str = FROZEN) -> Dict[str, object]:
    """
    Read the frozen document.

    :param path: Where it lives.

    :returns: The document.

    :raises FileNotFoundError: If it has not been fetched.
    """
    with open(path) as handle:
        return json.load(handle)


def _report(doc: Mapping[str, object]) -> None:
    """Print everything the verification turns on, in the order it should be read."""
    cells = doc["cells"]
    arms = complete_arms(doc)
    print(f"entity={doc['entity']} project={doc['project']} horizon={doc['final_step']}")

    print("\n-- cell inventory (from history; summaries are not consulted) --")
    for run_id, cell in sorted(cells.items(), key=lambda kv: (kv[1]["arm"], kv[1]["cell"])):  # type: ignore[union-attr]
        last = max(int(k) for k in cell["evals"])
        print(
            f"  {cell['arm']:<12} cell-{cell['cell']} state={cell['state']:<9} "
            f"init_seed={cell['init_seed']} dl_seed={cell['data_loader_seed']} "
            f"evals={len(cell['evals']):>2} last_eval={last}"
        )
    print(f"  arms complete at step {doc['final_step']}: {', '.join(arms)}")

    print("\n-- units --")
    seen = {
        round(implied_bytes_per_token(c["evals"][str(FINAL_STEP)]["dclm CE loss"], c["evals"][str(FINAL_STEP)]["dclm"]), 6)  # type: ignore[index]
        for c in cells.values()  # type: ignore[union-attr]
        if str(FINAL_STEP) in c["evals"]
    }
    print(f"  bytes per token implied by every cell's own CE/BPB: {sorted(seen)}")
    print(f"  nats per BPB = ln(2) * {BYTES_PER_TOKEN} = {NATS_PER_BPB:.12f}")

    table = endpoint_table(doc, arms)
    print("\n-- endpoint at the shared step, nats --")
    for arm in arms:
        v = table[arm]
        print(
            f"  {arm:<12} mean={statistics.fmean(v):.8f} sd={statistics.stdev(v):.8f} "
            f"cells=[{', '.join(f'{x:.5f}' for x in v)}]"
        )
    sig, df = pooled_sigma([table[a] for a in arms])
    print(f"  pooled sigma over {len(arms)} arms = {sig:.8f} at df {df}; c4({df})={c4(df):.4f}")
    bsig, bdf = blocked_residual_sigma(table)
    f, p = seed_effect_f(table)
    print(
        f"  blocked residual sigma = {bsig:.8f} at df {bdf}; seed main effect F={f:.3f} p={p:.4f}"
    )
    for i, one in enumerate(arms):
        for two in arms[i + 1 :]:
            print(
                f"  seed correlation {one} vs {two}: r={cross_arm_correlation(table, one, two):+.4f}"
            )

    print("\n-- contrasts, nats, every estimator --")
    for treat, control in CONTRASTS:
        if treat not in table or control not in table:
            continue
        print(f"  {treat} - {control}:")
        for iv in intervals(table, treat, control):
            print(
                f"    {iv.method:<8} {iv.point:+.6f} [{iv.lo:+.6f}, {iv.hi:+.6f}] "
                f"half={iv.half_width:.6f} sigma={iv.sigma:.6f} df={iv.df:.2f}"
            )

    print("\n-- trajectory: the arm mean and the gap, at every eval --")
    steps = sorted(int(k) for k in cells[f"{SUBMISSIONS['baseline']}-cell-0"]["evals"])  # type: ignore[index]
    for step in steps:
        row = {a: statistics.fmean(endpoint_table(doc, [a], step)[a]) for a in arms}
        gaps = "  ".join(
            f"{t[:4]}-{c[:4]}={row[t] - row[c]:+.5f}" for t, c in CONTRASTS if t in row and c in row
        )
        print("  step %5d  " % step + "  ".join(f"{a}={row[a]:.5f}" for a in arms) + "  | " + gaps)

    print("\n-- improvement per doubling over 3000-6000, and whether the arms are parallel --")
    window = [s for s in steps if s >= FINAL_STEP // 2]
    per_arm = {
        a: [
            slope_per_doubling(window, [endpoint_table(doc, [a], s)[a][i] for s in window])
            for i in range(N_SEEDS)
        ]
        for a in arms
    }
    for arm in arms:
        v = per_arm[arm]
        two_point = statistics.fmean(endpoint_table(doc, [arm], window[0])[arm]) - statistics.fmean(
            endpoint_table(doc, [arm], window[-1])[arm]
        )
        print(
            f"  {arm:<12} two-point={two_point:.6f}  regression={statistics.fmean(v):.6f} "
            f"sd={statistics.stdev(v):.6f}"
        )
    f, p = stats.f_oneway(*[per_arm[a] for a in arms])
    print(
        f"  one-way ANOVA over the per-cell slopes: F={f:.3f} p={p:.4f} (large p == parallel not refuted)"
    )

    print("\n-- per source, faithful minus baseline at the horizon --")
    total = statistics.fmean(table["faithful"]) - statistics.fmean(table["baseline"])
    for source in SOURCES:
        base = [to_nats(c["evals"][str(FINAL_STEP)][source]) for c in cells.values() if c["arm"] == "baseline"]  # type: ignore[union-attr,index]
        hc = [to_nats(c["evals"][str(FINAL_STEP)][source]) for c in cells.values() if c["arm"] == "faithful"]  # type: ignore[union-attr,index]
        d = statistics.fmean(hc) - statistics.fmean(base)
        print(
            f"  {source:<16} {d:+.5f} nats  ({100 * d / statistics.fmean(base):+.3f}% of level, "
            f"{100 * d / len(SOURCES) / total:.1f}% of the total effect)"
        )

    print("\n-- wall clock, and what the contrast looks like at equal wall clock --")
    rates = {}
    for arm in SUBMISSIONS:
        v = [step_seconds(c) for c in cells.values() if c["arm"] == arm]  # type: ignore[union-attr]
        good = [x for x in v if x is not None]
        rates[arm] = statistics.fmean(good)
        total = [
            c["runtime_at"][str(c["last_step"])] / c["last_step"]  # type: ignore[index]
            for c in cells.values()  # type: ignore[union-attr]
            if c["arm"] == arm and str(c["last_step"]) in c["runtime_at"]
        ]
        print(
            f"  {arm:<12} marginal={rates[arm]:.4f} s/step  "
            f"total-runtime-over-steps={statistics.fmean(total):.4f} s/step"
        )
    budget = FINAL_STEP * rates["baseline"]
    reach = budget / rates["faithful"]
    at_final = statistics.fmean(table["baseline"])
    print(
        f"  the baseline's {FINAL_STEP} steps cost {budget:.0f} s; faithful reaches step "
        f"{reach:.0f} in that time. Evals bracketing it, both flattering and not:"
    )
    for step in (
        int(reach) // EVAL_INTERVAL * EVAL_INTERVAL,
        -(-int(reach) // EVAL_INTERVAL) * EVAL_INTERVAL,
    ):
        at = statistics.fmean(endpoint_table(doc, ["faithful"], step)["faithful"])
        print(
            f"    faithful@{step} - baseline@{FINAL_STEP} = {at - at_final:+.5f} nats "
            f"(positive == the intervention is behind at equal wall clock)"
        )


def _self_test() -> int:
    """Exercise the statistics against hand-checkable inputs. No network, no W&B install."""
    assert abs(c4(4) - 0.9400) < 5e-5, c4(4)
    assert abs(c4(12) - 0.9794) < 5e-5, c4(12)
    assert abs(implied_bytes_per_token(3.3171401023864746, 1.0471819640365154) - 4.57) < 1e-9
    assert abs(to_nats(1.0) - 3.167682615159) < 1e-9
    flat = {s: 0.5 for s in SOURCES}
    assert abs(unweighted_endpoint(flat) - 0.5) < 1e-12
    sig, df = pooled_sigma([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert df == 4 and abs(sig - 1.0) < 1e-12, (sig, df)
    # A perfectly additive arm-by-seed table has no interaction, so no residual.
    additive = {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]}
    sig, df = blocked_residual_sigma(additive)
    assert df == 2 and sig < 1e-12, (sig, df)
    print("self-test ok")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. ``--fetch`` reads W&B; everything else reads the frozen file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--frozen", default=FROZEN)
    parser.add_argument("--fetch", action="store_true", help="refresh the frozen file from W&B")
    parser.add_argument("--self-test", action="store_true", help="check the statistics offline")
    opts = parser.parse_args(argv)

    if opts.self_test:
        return _self_test()
    if opts.fetch:
        doc = fetch(opts.entity, opts.project)
        with open(opts.frozen, "w") as handle:
            json.dump(doc, handle, indent=1, sort_keys=True)
        print(f"wrote {opts.frozen}", file=sys.stderr)
    else:
        doc = load(opts.frozen)
    _report(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
