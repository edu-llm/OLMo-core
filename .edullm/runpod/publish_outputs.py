#!/usr/bin/env python3
"""Attach RunPod-only logs and the winning checkpoint to the HPO W&B run."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def final_checkpoint(controller_state: Path) -> Path:
    """Return the checkpoint selected by the final-evaluation event."""

    checkpoint_ref: str | None = None
    for raw in controller_state.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        if event.get("kind") == "final_evaluation":
            checkpoint_ref = event.get("payload", {}).get("checkpoint_ref")
    if not checkpoint_ref:
        raise RuntimeError("controller state contains no final winner checkpoint")
    checkpoint = Path(checkpoint_ref)
    if not checkpoint.is_dir():
        raise RuntimeError(f"final winner checkpoint is missing: {checkpoint}")
    return checkpoint


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "hpo-run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from olmo_core.hpo.runtime_secrets import load_wandb_api_key

    load_wandb_api_key()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required")
    import wandb

    run = wandb.init(
        project="hpo-probe",
        id=os.environ["WANDB_RUN_ID"],
        resume="must",
        name=f"{args.run_id}-{args.mode}",
        group=args.run_id,
        job_type="runpod-publish",
    )
    log_path = args.job_root / "run.log"
    if log_path.is_file():
        logs = wandb.Artifact(
            f"{safe_name(args.run_id)}-runpod-log",
            type="hpo-run-log",
        )
        logs.add_file(str(log_path))
        run.log_artifact(logs)

    if args.mode != "proxy-cohort":
        state_path = args.job_root / "controller-state.jsonl"
        checkpoint = final_checkpoint(state_path)
        metadata = {"mode": args.mode, "run_id": args.run_id}
        result_path = args.job_root / "study-result.json"
        if result_path.is_file():
            metadata["study_result"] = json.loads(result_path.read_text(encoding="utf-8"))
        winner = wandb.Artifact(
            f"{safe_name(args.run_id)}-winner-checkpoint",
            type="hpo-winner-checkpoint",
            metadata=metadata,
        )
        winner.add_dir(str(checkpoint))
        run.log_artifact(winner)
        run.summary["runpod/winner_checkpoint"] = str(checkpoint)
    wandb.finish(exit_code=0, quiet=True)


if __name__ == "__main__":
    main()
