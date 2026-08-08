"""
Report per-cell progress of the tranche, which the platform's own status cannot.

``edullm status`` answers from GitHub, and GitHub stops knowing at ADMITTED. A five-cell
fan-out that is three-quarters finished and one that has not started a single cell both
read ``ADMITTED``, so a watcher built on it says nothing for the hours that matter. W&B is
where the cells actually report, so this reads there.

The distinction this exists to surface is a fan-out where some cells are running and some
are still queuing for capacity. That is invisible from the parent's status and it is the
difference between a noise floor at df = 4 and one at df = 2. It is not hypothetical: the
L40S submission placed three of five for over two hours, and the A100 one that replaced it
started its last two cells hundreds of steps behind the first.

THE ARM COMES FROM THE RUN AND NOT FROM THE COMMAND LINE, WHICH IS A REVERSAL OF WHAT THIS
FILE USED TO DO. It used to take an ``arm=run_id`` label from its caller, on the reasoning
that ``train_on_corpus`` writes the model and trainer configs but never an ``arm`` field, so
there was nothing in W&B to read. That reasoning was wrong in a way worth naming: ``arm.apply``
edits the model config *before* it is saved, and those edits survive. ``stage_gate`` recovers
the arm from them, and this defers to it.

The difference matters for exactly one failure, and it is the failure a watcher is for. A
label supplied on a command line is an assertion about what was submitted; it agrees with
itself whatever the cell did, so it cannot catch a cell that resolved to an arm nobody meant
-- which is the single silent way this tranche can be spoiled, since a treatment arm that
quietly ran the baseline produces a loss curve that looks like a loss curve. The label is kept
only for a cell too young to have logged a config, and marked when it is being used.

    python .edullm/tranche_watch.py run_019fe2f4-f528 --cells 5
    python .edullm/tranche_watch.py baseline=run_019fe2f4-f528 --cells 5 --steps 6000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stage_gate import arms_consistent_with  # noqa: E402

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "eduLLM")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "pre-training")

#: How a fan-out cell's W&B run id ends. The platform run id is the prefix and every cell of
#: one submission shares it.
CELL_SUFFIX = re.compile(r"^(?P<submission>.+)-cell-(?P<index>\d+)$")


@dataclass(frozen=True)
class CellProgress:
    """One cell of one submission, as W&B currently has it."""

    index: int
    state: str
    step: int
    seed: Optional[int]
    arms: Sequence[str]
    """Arms this cell's own config is consistent with. Empty when it has logged none yet."""

    labelled_arm: Optional[str] = None
    """What the caller said this submission was, if anything."""

    @property
    def arm(self) -> str:
        """The arm to display, and where it came from."""
        if self.arms:
            return " or ".join(self.arms)
        if self.labelled_arm:
            return f"{self.labelled_arm}?"
        return "unknown"

    @property
    def arm_is_claimed(self) -> bool:
        """Whether the arm shown is the caller's claim rather than the run's own testimony."""
        return not self.arms

    @property
    def contradicts_label(self) -> bool:
        """Whether the run's own config rules out the arm the caller named."""
        return (
            bool(self.arms) and self.labelled_arm is not None and self.labelled_arm not in self.arms
        )


def cells_of(
    run_prefix: str,
    labelled_arm: Optional[str] = None,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> List[CellProgress]:
    """
    Every cell of one submission, read out of W&B.

    MATCHED ON THE RUN ID AND NOT THE DISPLAY NAME. The cells of a fan-out share a name and
    differ only in the id's ``-cell-<index>`` suffix, so a name is not an identifier here --
    and it is not even stable, since the L40S submission's three cells were all renamed to
    ``...-died`` after they were cancelled. The id is what the platform assigned.

    :param run_prefix: The platform run id, or a unique prefix of it.
    :param labelled_arm: What the caller says this submission is, used only for a cell that
        has not logged a config yet.
    :param entity: W&B entity.
    :param project: W&B project.

    :returns: One :class:`CellProgress` per cell that has logged anything, in cell order.
    """
    import wandb

    api = wandb.Api(timeout=120)
    found: List[CellProgress] = []
    for run in api.runs(f"{entity}/{project}", per_page=200):
        match = CELL_SUFFIX.match(run.id)
        if not match or not match.group("submission").startswith(run_prefix):
            continue
        config = run.config or {}
        model = config.get("model") if isinstance(config.get("model"), dict) else {}
        step = run.summary.get("_step")
        found.append(
            CellProgress(
                index=int(match.group("index")),
                state=run.state,
                step=int(step) if step is not None else -1,
                seed=model.get("init_seed", config.get("init_seed")),
                # An empty config means the cell has started but not yet written one, so
                # `arms_consistent_with` would answer "every arm without lanes" from no
                # evidence. That is exactly the case the caller's label is kept for.
                arms=arms_consistent_with(config) if model else (),
                labelled_arm=labelled_arm,
            )
        )
    return sorted(found, key=lambda c: c.index)


def _bar(step: int, total: int, width: int = 24) -> str:
    if total <= 0 or step < 0:
        return " " * width
    filled = min(width, max(0, round(width * step / total)))
    return "#" * filled + "." * (width - filled)


def render(
    run_prefix: str,
    cells: Sequence[CellProgress],
    expected_cells: int,
    total_steps: int,
) -> str:
    """
    One line per cell, plus whatever needs saying about the ones that are missing or wrong.

    :param run_prefix: The submission, for the header.
    :param cells: What :func:`cells_of` found.
    :param expected_cells: How many cells the submission asked for.
    :param total_steps: The step count each cell is running to.

    :returns: A printable report.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%MZ")
    lines = [f"{stamp}  {run_prefix}  {len(cells)}/{expected_cells} cells reporting"]
    for cell in cells:
        pct = 100.0 * cell.step / total_steps if total_steps and cell.step >= 0 else 0.0
        label = f"cell {cell.index} seed {cell.seed} {cell.arm}"
        lines.append(
            f"   {label:38} {cell.state:9} step {cell.step:>6}/{total_steps} "
            f"{_bar(cell.step, total_steps)} {pct:5.1f}%"
        )

    contradicted = [c for c in cells if c.contradicts_label]
    if contradicted:
        lines.append(
            "   ARM MISMATCH on cell(s) "
            + ", ".join(str(c.index) for c in contradicted)
            + ": the config each one logged is not the arm this submission was called. A cell "
            "that ran the wrong arm produces a loss curve that looks like a loss curve, so "
            "nothing downstream would report this."
        )
    claimed = [c for c in cells if c.arm_is_claimed]
    if claimed:
        lines.append(
            f"   {len(claimed)} cell(s) marked '?' have logged no model config yet, so their "
            "arm is the label you supplied rather than anything the run has said."
        )
    if len(cells) < expected_cells:
        lines.append(
            f"   {expected_cells - len(cells)} cell(s) have logged nothing -- queuing for "
            "capacity, or dead. The parent's ADMITTED cannot tell these apart."
        )
    steps = [c.step for c in cells if c.step >= 0]
    if len(steps) > 1:
        lines.append(f"   spread across cells: {max(steps) - min(steps)} steps")
    return "\n".join(lines)


def report(
    run_prefix: str, expected_cells: int, total_steps: int, arm: Optional[str] = None
) -> str:
    """
    Read one submission and render it.

    :param run_prefix: The platform run id, e.g. ``run_019fe2f4-f528``.
    :param expected_cells: How many cells the submission asked for.
    :param total_steps: The step count each cell is running to.
    :param arm: What the caller says the submission is. Only used for a cell with no config.

    :returns: A printable report, or a line saying why there is none.
    """
    try:
        cells = cells_of(run_prefix, arm)
    except Exception as failure:  # noqa: BLE001 - a watcher must not die on a transient
        return f"wandb unreadable: {failure}"
    return render(run_prefix, cells, expected_cells, total_steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_prefix",
        nargs="+",
        help="platform run ids, optionally as arm=run_id. The arm is only a fallback for a "
        "cell that has not logged a config; every cell that has is read from its own.",
    )
    parser.add_argument("--cells", type=int, default=5, help="how many cells each submission has")
    parser.add_argument("--steps", type=int, default=6000, help="steps each cell runs to")
    opts = parser.parse_args(argv)
    for entry in opts.run_prefix:
        label, _, prefix = entry.rpartition("=")
        print(report(prefix, opts.cells, opts.steps, label or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
