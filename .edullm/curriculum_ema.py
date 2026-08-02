#!/usr/bin/env python3
"""Post-hoc four-checkpoint EMA for curriculum OLMo-core checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from olmo_core.distributed.checkpoint import unshard_checkpoint

from production_contract import checkpoint, task_loss, wandb_artifacts

EMA_STEPS = (2000, 2125, 2250, 2384)
EMA_ALPHA = 0.8


def ema_weights(count: int, alpha: float = EMA_ALPHA) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return [
        alpha ** (count - 1 - index) * ((1 - alpha) if index else 1.0) for index in range(count)
    ]


def _clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _blend(average: Any, newest: Any, alpha: float) -> Any:
    if torch.is_tensor(average) and torch.is_tensor(newest):
        if average.shape != newest.shape:
            raise ValueError(f"EMA tensor shape mismatch: {average.shape} != {newest.shape}")
        output = alpha * average.float() + (1 - alpha) * newest.detach().cpu().float()
        return output.to(dtype=average.dtype)
    if isinstance(average, Mapping) and isinstance(newest, Mapping):
        if set(average) != set(newest):
            raise ValueError("EMA checkpoint keys differ")
        return {key: _blend(average[key], newest[key], alpha) for key in average}
    if type(average) is not type(newest) or average != newest:
        raise ValueError("non-tensor EMA checkpoint values differ")
    return _clone(average)


def ema_merge_state_dicts(
    state_dicts: Sequence[Mapping[str, Any]], alpha: float = EMA_ALPHA
) -> dict[str, Any]:
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    average: Any = _clone(state_dicts[0])
    for newest in state_dicts[1:]:
        average = _blend(average, newest, alpha)
    return average


def validate_checkpoint_provenance(
    checkpoints_root: Path,
    *,
    arm: str,
    steps: Sequence[int] = EMA_STEPS,
) -> tuple[list[Path], dict[str, Any]]:
    directories: list[Path] = []
    common: dict[str, Any] | None = None
    for step in steps:
        directory = checkpoints_root / f"step{int(step)}"
        checkpoint.assert_checkpoint_materialized(directory)
        fingerprint = checkpoint.read_run_fingerprint(directory)
        identity = fingerprint["identity"]
        if identity.get("family") != "curriculum" or identity.get("arm") != arm:
            raise checkpoint.CheckpointContractError(
                f"{directory} is not a curriculum-{arm} checkpoint"
            )
        if common is None:
            common = fingerprint
        elif fingerprint != common:
            raise checkpoint.CheckpointContractError(
                "EMA inputs do not share one immutable parent/order/run identity"
            )
        directories.append(directory)
    assert common is not None
    if tuple(common["identity"].get("ema_steps") or ()) != tuple(int(s) for s in steps):
        raise checkpoint.CheckpointContractError("EMA steps differ from the run fingerprint")
    return directories, common


def unshard_model(checkpoint_dir: Path, destination: Path) -> dict[str, Any]:
    model_path, _ = unshard_checkpoint(
        checkpoint_dir / "model_and_optim",
        destination,
        optim=False,
        save_overwrite=False,
        quiet=True,
    )
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"unsharded model is not a state dict: {model_path}")
    if not payload or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError(f"unsharded model contains non-tensor state: {model_path}")
    return dict(payload)


def write_ema_artifact(
    output_dir: Path,
    *,
    model: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    arm: str,
    steps: Sequence[int] = EMA_STEPS,
    alpha: float = EMA_ALPHA,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "model_eval.pt"
    final_step = int(steps[-1])
    torch.save({"step": final_step, "model": dict(model)}, target)
    (output_dir / "step.txt").write_text(f"{final_step}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "steps": [int(step) for step in steps],
        "alpha": float(alpha),
        "convention": "avg <- alpha*avg + (1-alpha)*newest",
        "weights": ema_weights(len(steps), alpha),
        "source_identity_sha256": fingerprint["identity_sha256"],
    }
    (output_dir / "ema_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / checkpoint.RUN_FINGERPRINT_FILENAME).write_text(
        json.dumps(dict(fingerprint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-loss-eval-script", type=Path)
    parser.add_argument("--task-loss-nproc", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--local-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.local_smoke:
        if args.wandb_mode != "online" or not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("production EMA requires online W&B and WANDB_API_KEY")
        if args.task_loss_eval_script is None:
            raise SystemExit("production EMA requires --task-loss-eval-script")
    directories, fingerprint = validate_checkpoint_provenance(args.checkpoints_root, arm=args.arm)
    with tempfile.TemporaryDirectory(prefix="curriculum-ema-") as temporary:
        states = [
            unshard_model(directory, Path(temporary) / f"step{step}")
            for directory, step in zip(directories, EMA_STEPS)
        ]
    merged = ema_merge_state_dicts(states)
    output = args.output_dir or args.checkpoints_root / "step2384-ema"
    write_ema_artifact(
        output,
        model=merged,
        fingerprint=fingerprint,
        arm=args.arm,
    )
    eval_path: Path | None = None
    if args.task_loss_eval_script is not None:
        eval_path = output / "step2384-ema_task_loss.json"
        task_loss.trigger_task_loss_eval(
            output,
            run_name=f"{args.arm}-step2384-ema",
            out_path=eval_path,
            eval_script=args.task_loss_eval_script,
            nproc=args.task_loss_nproc,
        )
        payload = task_loss.validate_task_loss_result(eval_path)
    else:
        payload = None

    if not args.local_smoke:
        import wandb

        project = f"curriculum-{args.arm}"
        if (os.environ.get("EDULLM_WANDB_PROJECT") or project) != project:
            raise SystemExit(f"EMA W&B project must be {project}")
        run = wandb.init(
            project=project,
            name=f"{args.arm}-ema",
            mode=args.wandb_mode,
            config={"steps": list(EMA_STEPS), "alpha": EMA_ALPHA},
        )
        try:
            wandb_artifacts.wandb_log_checkpoint(
                run,
                output,
                step=EMA_STEPS[-1],
                extra_meta={"arm": args.arm, "posthoc_ema": True},
                strict=True,
                run_name=f"{args.arm}-ema",
            )
            assert payload is not None and eval_path is not None
            wandb_artifacts.wandb_log_eval(
                run, payload, step=EMA_STEPS[-1], eval_path=eval_path, strict=True
            )
            run.finish()
        except BaseException:
            run.finish(exit_code=1)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
