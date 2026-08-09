#!/usr/bin/env python3
"""Resume-safe local prune: keep only the latest durable permanent checkpoint.

Safe to run while training continues. It never deletes:

* the checkpoint at ``last_durable_step``
* any newer ``step*`` directory that may still be mid-finalize

It only removes checkpoints with step strictly less than the durable marker.
Regenerable ``model_eval.pt`` files are discarded after a checkpoint has a
completed task-loss result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))

from production_contract import checkpoint  # noqa: E402


def _completed_task_loss_steps(progress_dir: Path) -> set[int]:
    results = progress_dir / "task_loss_results"
    if not results.is_dir():
        return set()
    completed: set[int] = set()
    for path in results.glob("step*_task_loss.json"):
        name = path.name.removesuffix("_task_loss.json")
        if name.startswith("step") and name[4:].isdigit():
            completed.add(int(name[4:]))
    return completed


def prune_run(run_root: Path) -> dict[str, object]:
    save_folder = run_root / "checkpoints"
    progress_dir = run_root / "progress"
    steps = checkpoint.list_step_checkpoint_dirs(save_folder)
    durable = checkpoint.read_last_durable_step(progress_dir)
    completed_evals = _completed_task_loss_steps(progress_dir)

    discarded: list[str] = []
    for step, path in steps:
        if step in completed_evals and checkpoint.discard_regenerable_eval_weights(path):
            discarded.append(str(path / "model_eval.pt"))

    removed: list[str] = []
    keep_step = int(durable["last_durable_step"]) if durable is not None else None
    if keep_step is not None and (save_folder / f"step{keep_step}").is_dir():
        removed = [
            str(path)
            for path in checkpoint.prune_older_permanent_checkpoints(
                save_folder, keep_step=keep_step
            )
        ]

    return {
        "run_root": str(run_root),
        "keep_step": keep_step,
        "removed": removed,
        "discarded_model_eval": discarded,
        "remaining": [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="Curriculum run roots containing checkpoints/ and progress/",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    reports = [prune_run(path) for path in args.run_roots]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
