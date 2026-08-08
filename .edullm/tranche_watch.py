"""
Report per-cell progress of the tranche, which the platform's own status cannot.

``edullm status`` answers from GitHub, and GitHub stops knowing at ADMITTED. A five-cell
fan-out that is three-quarters finished and one that has not started a single cell both
read ``ADMITTED``, so a watcher built on it says nothing for the eighteen hours that
matter. W&B is where the cells actually report, so this reads there and falls back to the
platform only for runs that have not logged anything yet.

The distinction this exists to surface is a fan-out where some cells are running and some
are still queuing for capacity. That is invisible from the parent's status and it is the
difference between a noise floor at df = 4 and one at df = 2.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from typing import Dict, List, Tuple

WANDB_PROJECT = "pre-training"


def cells_by_run(project: str = WANDB_PROJECT) -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Group every W&B run in the project by the platform run id that spawned it.

    A fan-out's cells all carry the same run id prefix and differ in a suffix, so the
    prefix is what identifies the submission and the count of entries is how many cells
    have got far enough to log anything at all.

    :param project: The W&B project to read.

    :returns: Mapping of platform run id to a list of ``(name, step, state)``.
    """
    import wandb

    api = wandb.Api()
    grouped: Dict[str, List[Tuple[str, int, str]]] = {}
    for run in api.runs(project, per_page=100):
        name = run.name or ""
        if not name.startswith("run_"):
            continue
        short = "-".join(name.split("-")[:2])
        step = run.summary.get("_step")
        # The cell's identity is its seed, not its name: every cell of a fan-out shares the
        # run id and the suffix is the same for all of them, so a name-based label would
        # print the same string five times and hide exactly what this is here to show.
        # The arm is not in the logged config -- train_on_corpus writes the model and trainer
        # configs but not which arm chose them -- so the caller supplies the label and this
        # only distinguishes the replicates within a submission.
        seed = run.config.get("init_seed", "?")
        grouped.setdefault(short, []).append(
            (f"seed {seed}", int(step) if step is not None else -1, run.state)
        )
    return grouped


def _bar(step: int, total: int, width: int = 24) -> str:
    if total <= 0 or step < 0:
        return " " * width
    filled = min(width, max(0, round(width * step / total)))
    return "#" * filled + "." * (width - filled)


def report(run_prefix: str, expected_cells: int, total_steps: int) -> str:
    """
    One line per cell, plus a summary line naming how many are missing.

    :param run_prefix: The platform run id, e.g. ``run_019fe279-4ef0``.
    :param expected_cells: How many cells the submission asked for.
    :param total_steps: The step count each cell is running to.

    :returns: A printable report.
    """
    try:
        grouped = cells_by_run()
    except Exception as failure:  # noqa: BLE001 - a watcher must not die on a transient
        return f"wandb unreadable: {failure}"

    cells = sorted(grouped.get(run_prefix, []), key=lambda c: c[0])
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%H:%MZ")
    lines = [f"{stamp}  {run_prefix}  {len(cells)}/{expected_cells} cells reporting"]
    for name, step, state in cells:
        pct = 100.0 * step / total_steps if total_steps and step >= 0 else 0.0
        lines.append(
            f"   {name:26} {state:9} step {step:>6}/{total_steps} "
            f"{_bar(step, total_steps)} {pct:5.1f}%"
        )
    if len(cells) < expected_cells:
        lines.append(
            f"   {expected_cells - len(cells)} cell(s) have logged nothing -- queuing for "
            "capacity, or dead. The parent's ADMITTED cannot tell these apart."
        )
    steps = [s for _, s, _ in cells if s >= 0]
    if len(steps) > 1:
        lines.append(f"   spread across cells: {max(steps) - min(steps)} steps")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_prefix",
        nargs="+",
        help="platform run ids, optionally as arm=run_id so the arm is named in the output",
    )
    parser.add_argument("--cells", type=int, default=5, help="how many cells each submission has")
    parser.add_argument("--steps", type=int, default=6000, help="steps each cell runs to")
    opts = parser.parse_args(argv)
    for entry in opts.run_prefix:
        label, _, prefix = entry.rpartition("=")
        line = report(prefix, opts.cells, opts.steps)
        print(f"{label + '  ' if label else ''}{line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
