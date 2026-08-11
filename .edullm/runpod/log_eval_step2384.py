#!/usr/bin/env python3
"""Re-log step-2384 eval metrics at the run's current W&B step."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import wandb

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))

from production_contract import task_loss  # noqa: E402
from production_contract.wandb_artifacts import task_loss_metrics  # noqa: E402


def _read_run_env(path: Path) -> dict[str, str]:
    run_env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"export (\w+)='([^']*)'", line)
        if match:
            run_env[match.group(1)] = match.group(2)
    return run_env


def main() -> None:
    run_root = Path("/workspace/edullm-runs/curriculum/warmup-quadratic10-mtld")
    run_env = _read_run_env(run_root / "run.env")
    step = 2384
    out_path = run_root / "progress" / "task_loss_results" / f"step{step}_task_loss.json"
    payload = task_loss.validate_task_loss_result(out_path)

    wandb.init(
        project="curriculum-moe",
        entity="eduLLM",
        id=run_env["WANDB_RUN_ID"],
        resume="must",
        name=run_env["EDULLM_RUN_ID"],
        dir="/workspace/wandb",
    )

    metrics = task_loss_metrics(payload)
    # Let W&B pick the next valid step (2385); explicit step=2384 is rejected.
    log_step = int(getattr(wandb.run, "_step", step))
    wandb.run.log(metrics)
    log_step = int(getattr(wandb.run, "_step", log_step))
    print(
        f"logged {len(metrics)} eval metrics at wandb step {log_step} "
        f"(checkpoint step {step})"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
