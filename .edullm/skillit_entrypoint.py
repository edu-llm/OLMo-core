#!/usr/bin/env python3
"""Validate platform bindings and launch one eight-rank Skill-It arm."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from skillit_math import DATASET_ID, DATASET_VERSION, arm_by_index

GPU_RANKS = 8


class EntrypointError(RuntimeError):
    """The platform supplied values inconsistent with the selected arm."""


def platform_values(arm_index: int, environ: Mapping[str, str]) -> tuple[str, Optional[str]]:
    arm = arm_by_index(arm_index)
    actual_source = (
        environ.get("EDULLM_DATASET_ID", ""),
        environ.get("EDULLM_DATASET_VERSION", ""),
    )
    if actual_source != (DATASET_ID, DATASET_VERSION):
        raise EntrypointError(
            f"platform dataset must be {DATASET_ID}/{DATASET_VERSION}, got "
            f"{actual_source[0]}/{actual_source[1]}"
        )
    project = environ.get("EDULLM_WANDB_PROJECT", "")
    if project != arm.wandb_project:
        raise EntrypointError(
            f"arm {arm.arm_id} requires W&B project {arm.wandb_project}, got {project!r}"
        )
    run_id = environ.get("EDULLM_RUN_ID", f"skillit-{arm.arm_id}")
    return run_id, environ.get("WANDB_ENTITY")


def torchrun_command(
    arm_index: int,
    *,
    run_id: str,
    wandb_entity: Optional[str],
    resume: bool,
    state_root: Optional[str] = None,
) -> list[str]:
    arm = arm_by_index(arm_index)
    root = Path(state_root or f"/tmp/edullm-skillit/{arm.arm_id}")
    script_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={GPU_RANKS}",
        str(script_dir / "train_skillit_370m.py"),
        "--arm-index",
        str(arm.index),
        "--run-name",
        f"{run_id}-{arm.arm_id}",
        "--work-dir",
        str(root / "work"),
        "--save-folder",
        str(root / "checkpoints"),
        "--progress-dir",
        str(root / "progress"),
        "--task-loss-dir",
        str(root / "task_loss"),
        "--task-loss-evaluator",
        str(script_dir / "eval_task_loss_olmo_core.py"),
        "--wandb-mode",
        "online",
    ]
    if wandb_entity:
        command.extend(["--wandb-entity", wandb_entity])
    if resume:
        command.append("--resume")
    return command


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--arm-index", type=int, choices=(0, 1), required=True)
    output.add_argument(
        "--resume",
        action="store_true",
        help="explicitly resume this arm from EDULLM_CHECKPOINT_DIR/checkpoints",
    )
    output.add_argument("--print-command", action="store_true")
    output.add_argument(
        "--state-root",
        help="local scratch or a locally restored complete W&B run root",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        run_id, entity = platform_values(args.arm_index, os.environ)
        command = torchrun_command(
            args.arm_index,
            run_id=run_id,
            wandb_entity=entity,
            resume=args.resume,
            state_root=args.state_root,
        )
    except (EntrypointError, RuntimeError) as exc:
        print(f"[skillit] {exc}", file=sys.stderr)
        return 2
    if args.print_command:
        print(" ".join(command))
        return 0
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
