#!/usr/bin/env python3
"""Re-log step-38144 eval metrics to the arm-9 hpo-moe run without checkpoint upload."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import wandb

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))

from production_contract import task_loss  # noqa: E402
from production_contract.wandb_artifacts import task_loss_metrics  # noqa: E402

WANDB_STEP_SCALE = 2
STEP = 38144


def _read_run_env(path: Path) -> dict[str, str]:
    run_env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"export (\w+)='([^']*)'", line)
        if match:
            run_env[match.group(1)] = match.group(2)
    return run_env


def main() -> None:
    run_root = Path("/workspace/edullm-runs/hpo-moe/warmup-quadratic10-mtld-256ki")
    run_env = _read_run_env(run_root / "run.env")
    out_path = run_root / "progress" / "task_loss_results" / f"step{STEP}_task_loss.json"
    payload = task_loss.validate_task_loss_result(out_path)
    metrics = task_loss_metrics(payload)

    api = wandb.Api()
    run_path = f"eduLLM/hpo-moe/{run_env['WANDB_RUN_ID']}"
    last_step = int(api.run(run_path).lastHistoryStep or 0)
    wandb_step = max(last_step + 1, STEP * WANDB_STEP_SCALE)

    wandb.init(
        project="hpo-moe",
        entity="eduLLM",
        id=run_env["WANDB_RUN_ID"],
        resume="must",
        name=run_env["EDULLM_RUN_ID"],
        dir="/workspace/wandb",
    )

    wandb.log(metrics, step=wandb_step)
    print(
        f"logged {len(metrics)} eval metrics at wandb step {wandb_step} "
        f"(checkpoint step {STEP}, macro_bpb={metrics.get('eval/macro_bpb'):.4f})"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
