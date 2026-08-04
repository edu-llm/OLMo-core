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
EMA_WANDB_STEP = 2385


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
    overwrite: bool = False,
    wandb_step: int = EMA_WANDB_STEP,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=overwrite)
    target = output_dir / "model_eval.pt"
    published_step = int(wandb_step)
    torch.save({"step": published_step, "model": dict(model)}, target)
    (output_dir / "step.txt").write_text(f"{published_step}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "arm": arm,
        "steps": [int(step) for step in steps],
        "source_final_step": int(steps[-1]),
        "alpha": float(alpha),
        "convention": "avg <- alpha*avg + (1-alpha)*newest",
        "weights": ema_weights(len(steps), alpha),
        "source_identity_sha256": fingerprint["identity_sha256"],
        "wandb_step": published_step,
    }
    (output_dir / "ema_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / checkpoint.RUN_FINGERPRINT_FILENAME).write_text(
        json.dumps(dict(fingerprint), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_ema_checkpoint(
    checkpoints_root: Path,
    *,
    arm: str,
    output_dir: Path | None = None,
    steps: Sequence[int] = EMA_STEPS,
    alpha: float = EMA_ALPHA,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    directories, fingerprint = validate_checkpoint_provenance(
        checkpoints_root, arm=arm, steps=steps
    )
    with tempfile.TemporaryDirectory(prefix="curriculum-ema-") as temporary:
        states = [
            unshard_model(directory, Path(temporary) / f"step{step}")
            for directory, step in zip(directories, steps)
        ]
    merged = ema_merge_state_dicts(states, alpha=alpha)
    output = output_dir or checkpoints_root / "step2384-ema"
    write_ema_artifact(
        output,
        model=merged,
        fingerprint=fingerprint,
        arm=arm,
        steps=steps,
        alpha=alpha,
        overwrite=overwrite,
    )
    return output, fingerprint


def finalize_ema_production(
    *,
    checkpoints_root: Path,
    arm: str,
    run_name: str,
    task_loss_dir: Path,
    eval_script: Path,
    task_loss_nproc: int,
    progress_dir: Path,
    fingerprint_path: Path | None,
    wandb_run: Any,
    wandb_mode: str,
    production: bool,
    method: str | None = None,
    ema_dir: Path | None = None,
    evaluate: Any | None = None,
) -> dict[str, Any]:
    """Merge late checkpoints, eval the EMA artifact, and publish it as the final model.

    The EMA checkpoint is the sole final W&B model artifact for the training run.
    Its task-loss eval is logged on the same W&B run at step ``EMA_WANDB_STEP`` (2385).
    """
    strict_upload = wandb_artifacts.production_online(production=production, mode=wandb_mode)
    wandb_artifacts.require_wandb_for_production(wandb_run, production=production, mode=wandb_mode)
    if strict_upload and not eval_script.is_file():
        raise checkpoint.CheckpointContractError(
            "production EMA requires the synchronous 20-label evaluator"
        )

    if ema_dir is None:
        ema_dir, _fingerprint = build_ema_checkpoint(
            checkpoints_root,
            arm=arm,
            overwrite=True,
        )
    task_loss_dir = Path(task_loss_dir)
    task_loss_dir.mkdir(parents=True, exist_ok=True)
    eval_path = task_loss_dir / f"step{EMA_WANDB_STEP}_task_loss.json"
    ema_eval_copy = ema_dir / "step2384-ema_task_loss.json"
    if evaluate is not None:
        payload = evaluate(
            ema_dir,
            eval_path,
            f"{run_name}-step{EMA_WANDB_STEP}-ema",
        )
        if payload is None:
            raise checkpoint.CheckpointContractError("EMA task-loss eval returned no payload")
    else:
        task_loss.trigger_task_loss_eval(
            ema_dir,
            run_name=f"{run_name}-step{EMA_WANDB_STEP}-ema",
            out_path=eval_path,
            eval_script=eval_script,
            nproc=task_loss_nproc,
        )
        payload = task_loss.validate_task_loss_result(eval_path)
    if eval_path.is_file():
        ema_eval_copy.write_bytes(eval_path.read_bytes())

    if fingerprint_path is not None:
        checkpoint.copy_fingerprint_into_checkpoint(fingerprint_path, ema_dir)

    # Publish under the training run's checkpoint artifact name so the EMA is
    # the durable final model (not a separate *-ema artifact, not step 2384).
    artifact_ref = wandb_artifacts.checkpoint_artifact_ref(
        run_name=run_name,
        project=str(getattr(wandb_run, "project", "") or "curriculum"),
        entity=str(getattr(wandb_run, "entity", "") or "") or None,
        alias=f"step-{EMA_WANDB_STEP:07d}",
    )
    wandb_artifacts.wandb_log_checkpoint(
        wandb_run,
        ema_dir,
        step=EMA_WANDB_STEP,
        extra_meta={
            "arm": arm,
            "method": method,
            "posthoc_ema": True,
            "ema_source_steps": list(EMA_STEPS),
            "ema_alpha": EMA_ALPHA,
            "final_model": True,
        },
        strict=strict_upload,
        run_name=run_name,
    )
    wandb_artifacts.wandb_log_eval(
        wandb_run,
        payload,
        step=EMA_WANDB_STEP,
        eval_path=eval_path,
        strict=strict_upload,
    )
    wandb_artifacts.wandb_log_directory_artifact(
        wandb_run,
        task_loss_dir,
        name=f"{run_name}-task-loss",
        artifact_type="eval",
        strict=strict_upload,
    )
    checkpoint.write_last_durable_step(
        progress_dir,
        EMA_WANDB_STEP,
        checkpoint_artifact=artifact_ref if wandb_run is not None else None,
        extra={
            "run_name": run_name,
            "task_loss_complete": True,
            "task_loss_result": str(eval_path),
            "posthoc_ema": True,
            "final_model": "ema",
            "ema_source_steps": list(EMA_STEPS),
            "ema_alpha": EMA_ALPHA,
            "fingerprint_schema_version": checkpoint.FINGERPRINT_SCHEMA_VERSION,
        },
    )
    (progress_dir / "ema_integrated.done").write_text(
        json.dumps(
            {
                "wandb_step": EMA_WANDB_STEP,
                "ema_dir": str(ema_dir),
                "eval_path": str(eval_path),
                "final_model": "ema",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-loss-eval-script", type=Path)
    parser.add_argument("--task-loss-nproc", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=("online", "disabled"), default="online")
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID"))
    parser.add_argument("--run-name", default=os.environ.get("EDULLM_RUN_ID"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-step", type=int, default=EMA_WANDB_STEP)
    parser.add_argument("--local-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.local_smoke:
        if args.wandb_mode != "online" or not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("production EMA requires online W&B and WANDB_API_KEY")
        if args.task_loss_eval_script is None:
            raise SystemExit("production EMA requires --task-loss-eval-script")
    output = args.output_dir or args.checkpoints_root / "step2384-ema"
    build_ema_checkpoint(
        args.checkpoints_root,
        arm=args.arm,
        output_dir=output,
        overwrite=True,
    )
    eval_path: Path | None = None
    payload: dict[str, Any] | None = None
    if args.task_loss_eval_script is not None:
        eval_path = (
            args.checkpoints_root.parent / "progress" / "task_loss_results" / f"step{int(args.wandb_step)}_task_loss.json"
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        task_loss.trigger_task_loss_eval(
            output,
            run_name=f"{args.arm}-step{int(args.wandb_step)}-ema",
            out_path=eval_path,
            eval_script=args.task_loss_eval_script,
            nproc=args.task_loss_nproc,
        )
        payload = task_loss.validate_task_loss_result(eval_path)
        (output / "step2384-ema_task_loss.json").write_bytes(eval_path.read_bytes())

    if not args.local_smoke:
        import wandb

        project = os.environ.get("EDULLM_WANDB_PROJECT") or "curriculum"
        if project != "curriculum":
            raise SystemExit("EMA W&B project must be curriculum")
        init_kwargs: dict[str, Any] = {
            "project": project,
            "mode": args.wandb_mode,
            "config": {
                "steps": list(EMA_STEPS),
                "alpha": EMA_ALPHA,
                "posthoc_ema": True,
                "wandb_step": int(args.wandb_step),
            },
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_id:
            init_kwargs["id"] = str(args.wandb_run_id)
            init_kwargs["resume"] = "must"
            if args.run_name:
                init_kwargs["name"] = str(args.run_name)
        else:
            init_kwargs["name"] = f"{args.arm}-ema"
        run = wandb.init(**init_kwargs)
        try:
            wandb_artifacts.wandb_log_checkpoint(
                run,
                output,
                step=int(args.wandb_step),
                extra_meta={
                    "arm": args.arm,
                    "posthoc_ema": True,
                    "ema_source_steps": list(EMA_STEPS),
                    "final_model": True,
                },
                strict=True,
                run_name=args.run_name or args.arm,
            )
            assert payload is not None and eval_path is not None
            wandb_artifacts.wandb_log_eval(
                run, payload, step=int(args.wandb_step), eval_path=eval_path, strict=True
            )
            run.finish()
        except BaseException:
            run.finish(exit_code=1)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
