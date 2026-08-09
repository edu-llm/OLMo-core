#!/usr/bin/env python3
"""Upload an already-materialized final checkpoint to W&B without retraining."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import wandb

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))

from production_contract import checkpoint as cp  # noqa: E402
from production_contract import task_loss  # noqa: E402


def _read_run_env(path: Path) -> dict[str, str]:
    run_env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"export (\w+)='([^']*)'", line)
        if match:
            run_env[match.group(1)] = match.group(2)
    return run_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--step", type=int, default=2384)
    args = parser.parse_args()

    run_root = args.run_root
    run_env = _read_run_env(run_root / "run.env")
    step = int(args.step)
    out_path = run_root / "progress" / "task_loss_results" / f"step{step}_task_loss.json"

    wandb.init(
        project="curriculum-moe",
        entity="eduLLM",
        id=run_env["WANDB_RUN_ID"],
        resume="must",
        name=run_env["EDULLM_RUN_ID"],
        dir="/workspace/wandb",
    )

    def already_evaluated(_checkpoint: Path, *, out_path: Path, **_kwargs: object) -> None:
        task_loss.validate_task_loss_result(out_path)

    cp.finalize_permanent_checkpoint(
        arm=args.arm,
        checkpoint_dir=run_root / "checkpoints" / f"step{step}",
        step=step,
        run_name=run_env["EDULLM_RUN_ID"],
        task_loss_dir=run_root / "progress" / "task_loss_results",
        task_loss_enabled=True,
        progress_dir=run_root / "progress",
        fingerprint_path=run_root
        / "progress"
        / "current_fingerprint"
        / "run_fingerprint.json",
        method=args.method,
        wandb_run=wandb.run,
        wandb_mode="online",
        production=True,
        upload_checkpoint=True,
        run_evaluator=already_evaluated,
    )
    task_loss.validate_task_loss_result(out_path)
    wandb.finish()
    print(f"finalized {run_root} step {step}")


if __name__ == "__main__":
    main()
