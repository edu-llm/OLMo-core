#!/usr/bin/env python3
"""Replay a ``SkipStepOptimizer`` threshold over runs that have already finished.

WHAT THIS IS FOR. The amendment of 2026-08-08 turns spike skipping on for every arm of the
hyper-connection tranche, and the two constants behind it -- ``--skip-step-sigma-factor`` and
``--skip-step-rolling-interval`` -- decide whether the intervention catches the thing it was
brought in for. Picking them by eye would be picking them after seeing which cells spiked, on
the arm that is the comparator for every hypothesis in the module. So they are the library's
own defaults, and this is what checks that the defaults do the job rather than asserting it:
it takes the per-step loss and gradient norm the finished cells already logged and asks, step
by step, what the optimizer would have decided.

IT IS A REPLAY AND NOT A COUNTERFACTUAL, and the difference matters when reading the output.
The trajectory it reads is the one that spiked. Declining an update changes every step after
it, so this cannot say what the loss would have become. What it *can* say is the only thing
the choice of constant turns on -- whether the rule fires on the **first** step of an episode
rather than only once the episode is already large -- and what the same rule costs on runs
that never spiked. Both are read directly off the recorded history and neither needs a
counterfactual.

THE RULE IS NOT REIMPLEMENTED HERE. :func:`replay` drives the real
:meth:`olmo_core.optim.SkipStepOptimizer.get_step_factor` through a throwaway optimizer over a
one-element parameter, feeding it the recorded values in order. A reimplementation would be a
second copy of the rule that agrees with the first until somebody edits one of them, and the
whole point of the exercise is to be right about what the optimizer will actually do.

Usage::

    python .edullm/skip_step_calibration.py --self-test          # no network, no GPU
    python .edullm/skip_step_calibration.py --submission run_019fe2f4-f528 --seeds 0,1,2,3,4
    python .edullm/skip_step_calibration.py --submission run_019fe2f4-f528 \\
        --cache /tmp/history.json --sigma 4,5,6,8,10

``--cache`` writes the fetched history and reads it back on the next run, so a sweep over
several thresholds costs one trip to W&B. The cache is several megabytes per five cells and is
deliberately not committed: it is derived data, the run ids that produced it are in
``.edullm/noise-floor.json``, and an image built from this commit has no use for it.
"""

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: The metric names the train module records, which are the two signals the rule reads.
GRAD_NORM_KEY = "optim/total grad norm"
CE_LOSS_KEY = "train/CE loss"
STEP_KEY = "_step"


@dataclass(frozen=True)
class Skip:
    """One step the rule would have declined, and what it was that fired."""

    step: int
    loss: float
    grad_norm: float
    loss_z: float
    grad_norm_z: float

    @property
    def fired_on(self) -> str:
        """Which signal crossed. Both can, and on the observed episodes only one did."""
        parts = []
        if self.loss_z > self.grad_norm_z:
            parts.append("loss")
        if self.grad_norm_z >= self.loss_z:
            parts.append("gradient norm")
        return " and ".join(parts)


@dataclass
class Replay:
    """What one threshold would have done to one run."""

    seed: int
    steps: int
    skips: List[Skip] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.skips)

    @property
    def largest_trigger(self) -> float:
        """
        The biggest gradient norm at a declined step.

        This is the statistic that separates a run that declined a handful of unremarkable
        steps from one that declined the onset of a spike; the count alone does not.
        """
        return max((s.grad_norm for s in self.skips), default=0.0)

    def first_at_or_after(self, step: int) -> Optional[int]:
        """The first declined step at or after ``step``, or ``None`` if the rule never fired."""
        return next((s.step for s in self.skips if s.step >= step), None)


def replay(
    history: Sequence[Dict[str, float]],
    seed: int = 0,
    sigma_factor: int = 6,
    rolling_interval_length: int = 128,
) -> Replay:
    """
    Ask the real optimizer, step by step, what it would have decided.

    :param history: Rows carrying ``_step``, ``train/CE loss`` and ``optim/total grad norm``,
        in any order. Rows missing either signal are dropped, because a step the optimizer
        never saw is not a step it declined.
    :param seed: Which replicate this is, carried through to the result for reporting.
    :param sigma_factor: The threshold under test.
    :param rolling_interval_length: The window under test.

    :returns: A :class:`Replay`.
    """
    import torch

    from olmo_core.optim import SkipStepAdamW

    rows = sorted(
        (r for r in history if r.get(GRAD_NORM_KEY) is not None and r.get(CE_LOSS_KEY) is not None),
        key=lambda r: r[STEP_KEY],
    )

    # A one-element parameter on the CPU. Nothing is ever stepped: `get_step_factor` reads only
    # the two rolling buffers, and `device` needs some parameter to point at.
    optim = SkipStepAdamW(
        [torch.zeros(1)],
        rolling_interval_length=rolling_interval_length,
        sigma_factor=sigma_factor,
    )

    out = Replay(seed=seed, steps=len(rows))
    for row in rows:
        loss = float(row[CE_LOSS_KEY])
        grad_norm = float(row[GRAD_NORM_KEY])

        # The z scores are computed off the same buffers the rule reads, and *before* the new
        # values are appended, because `get_step_factor` compares the latest value against the
        # window that excludes it. They are reporting only; the decision below is the
        # optimizer's own.
        loss_z = _z(loss, optim._losses)
        grad_norm_z = _z(grad_norm, optim._grad_norms)

        optim.latest_loss = torch.tensor(loss)
        optim.latest_grad_norm = torch.tensor(grad_norm)

        if float(optim.get_step_factor()) < 0.5:
            out.skips.append(
                Skip(
                    step=int(row[STEP_KEY]),
                    loss=loss,
                    grad_norm=grad_norm,
                    loss_z=loss_z,
                    grad_norm_z=grad_norm_z,
                )
            )
    return out


def _z(value: float, window: Sequence) -> float:
    """
    How many standard deviations ``value`` sits above the window's mean.

    :returns: 0.0 when the window is too short or flat to have an opinion, which is what the
        rule itself does with it.
    """
    if len(window) < 2:
        return 0.0
    previous = [float(v) for v in window]
    spread = statistics.stdev(previous)
    if spread <= 0.0:
        return 0.0
    return (value - statistics.mean(previous)) / spread


def fetch(
    submission: str,
    seeds: Sequence[int],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Read the per-step history of one submission's cells out of W&B.

    Read-only, and it reaches W&B rather than AWS: the run histories are the only copy of the
    per-step gradient norms, and ``AGENTS.md`` forbids this repository calling AWS directly.

    :param submission: The run id prefix, e.g. ``run_019fe2f4-f528``. Cells are its
        ``-cell-<seed>`` children.
    :param seeds: Which cells to read.

    :returns: History rows keyed by seed.
    """
    import wandb

    api = wandb.Api(timeout=180)
    out: Dict[int, List[Dict[str, float]]] = {}
    for seed in seeds:
        run = _cell(api, entity, project, submission, seed)
        out[seed] = list(run.scan_history(keys=[STEP_KEY, GRAD_NORM_KEY, CE_LOSS_KEY]))
    return out


def _cell(api, entity: str, project: str, submission: str, seed: int):
    """
    One cell of a fan-out, found by prefix rather than by its full id.

    ``noise-floor.json`` records the cells under their full uuid, and the submission slug a
    person has to hand is the short prefix ``edullm status`` prints.
    """
    suffix = f"-cell-{seed}"
    for run in api.runs(f"{entity}/{project}", per_page=200):
        if run.id.startswith(submission) and run.id.endswith(suffix):
            return run
    raise LookupError(
        f"no run in {entity}/{project} whose id starts with {submission!r} and ends with "
        f"{suffix!r}. `edullm status --json` names the submissions this account can see."
    )


def render(replays: Sequence[Replay], episodes: Optional[Dict[int, Tuple[int, int]]] = None) -> str:
    """
    The calibration table, which is what goes into the pre-registration.

    :param replays: One per seed.
    :param episodes: Known instability windows, keyed by seed, as ``(first step, last step)``.
        A seed with no entry is treated as a run that never spiked.
    """
    episodes = episodes or {}
    lines = [
        f"{'seed':>4}  {'steps':>6}  {'skips':>5}  {'rate':>7}  {'largest trigger':>15}  onset",
    ]
    for r in replays:
        rate = r.count / r.steps if r.steps else 0.0
        window = episodes.get(r.seed)
        if window is None:
            onset = "no episode recorded"
        else:
            caught = r.first_at_or_after(window[0] - 5)
            onset = (
                f"episode {window[0]}-{window[1]}, first declined step {caught}"
                if caught is not None and caught <= window[1]
                else f"episode {window[0]}-{window[1]}, NOT CAUGHT"
            )
        lines.append(
            f"{r.seed:>4}  {r.steps:>6}  {r.count:>5}  {rate:>6.2%}  "
            f"{r.largest_trigger:>15.4g}  {onset}"
        )
    return "\n".join(lines)


def self_test() -> None:
    """
    Check the replay against a planted truth, with no network and no GPU.

    Two properties, and they are the two the calibration rests on. A flat series with one
    enormous excursion is declined at exactly the excursion and nowhere else; and nothing is
    declined before the window has half filled, however large the excursion, because that is
    what ``get_step_factor`` does with a short buffer.
    """
    rolling = 16
    planted = 4321

    flat = [
        {STEP_KEY: i, CE_LOSS_KEY: 2.5 + 0.001 * (i % 3), GRAD_NORM_KEY: 0.15 + 0.001 * (i % 3)}
        for i in range(200)
    ]
    flat[planted % 200][GRAD_NORM_KEY] = 30.0
    spike_at = planted % 200

    out = replay(flat, sigma_factor=6, rolling_interval_length=rolling)
    assert [s.step for s in out.skips] == [
        spike_at
    ], f"expected exactly one declined step at {spike_at}, got {[s.step for s in out.skips]}"
    assert out.largest_trigger == 30.0, out.largest_trigger
    assert "gradient norm" in out.skips[0].fired_on, out.skips[0].fired_on

    early = [
        {STEP_KEY: i, CE_LOSS_KEY: 2.5 + 0.001 * (i % 3), GRAD_NORM_KEY: 0.15 + 0.001 * (i % 3)}
        for i in range(rolling)
    ]
    early[2][GRAD_NORM_KEY] = 500.0
    assert (
        replay(early, rolling_interval_length=rolling).count == 0
    ), "nothing may be declined before the rolling window has half filled"

    # And a loss excursion with a quiet gradient is caught too, because the rule declines when
    # either signal fires rather than when both do.
    loss_only = [
        {STEP_KEY: i, CE_LOSS_KEY: 2.5 + 0.001 * (i % 3), GRAD_NORM_KEY: 0.15 + 0.001 * (i % 3)}
        for i in range(200)
    ]
    loss_only[100][CE_LOSS_KEY] = 40.0
    assert [s.step for s in replay(loss_only, rolling_interval_length=rolling).skips] == [100]

    print("self-test OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--submission", help="Run id prefix, e.g. run_019fe2f4-f528.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated cell indices.")
    parser.add_argument(
        "--sigma", default="6", help="Comma-separated --skip-step-sigma-factor values to try."
    )
    parser.add_argument("--rolling", type=int, default=128, help="Rolling window length.")
    parser.add_argument(
        "--episodes",
        default="",
        help="Known instability windows, as seed:first-last, comma separated. For stage 1: "
        "0:1376-1418,1:1726-1773.",
    )
    parser.add_argument("--cache", help="Read the history from here, or write it here.")
    parser.add_argument("--self-test", action="store_true", help="Planted truth. No network.")
    opts = parser.parse_args()

    if opts.self_test:
        self_test()
        return 0

    if not opts.submission and not opts.cache:
        parser.error("pass --submission to fetch, or --cache to read a fetch you already made")

    seeds = [int(s) for s in opts.seeds.split(",") if s.strip()]
    history: Dict[int, List[Dict[str, float]]] = {}
    if opts.cache and os.path.exists(opts.cache):
        history = {int(k): v for k, v in json.load(open(opts.cache)).items()}
    else:
        history = fetch(opts.submission, seeds, entity=opts.entity, project=opts.project)
        if opts.cache:
            with open(opts.cache, "w") as f:
                json.dump(history, f)

    episodes = {}
    for entry in filter(None, (e.strip() for e in opts.episodes.split(","))):
        seed, _, window = entry.partition(":")
        first, _, last = window.partition("-")
        episodes[int(seed)] = (int(first), int(last))

    for sigma in (int(s) for s in opts.sigma.split(",") if s.strip()):
        print(f"\n=== sigma_factor {sigma}, rolling interval {opts.rolling} ===")
        replays = [
            replay(history[s], seed=s, sigma_factor=sigma, rolling_interval_length=opts.rolling)
            for s in seeds
            if s in history
        ]
        print(render(replays, episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
