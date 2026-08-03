#!/usr/bin/env python3
"""Platform entrypoint for one README-faithful token-selection arm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EDULLM_DIR = Path(__file__).resolve().parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from production_contract.checkpoint import assert_resume_fingerprint  # noqa: E402
from production_contract.wandb_artifacts import restore_checkpoint_artifact  # noqa: E402
from token_selection_370m.arms import ARM_SPECS, REFHQ, get_arm  # noqa: E402
from token_selection_370m.recipe import (  # noqa: E402
    build_trainer,
    immutable_corpus_binding,
    scientific_identity,
    total_steps,
    write_identity,
)

PRODUCTION_WORLD_SIZE = 8


def _path(name: str, default: str = "") -> str | None:
    value = os.environ.get(name, default).strip()
    return value or None


def _latest_resume_checkpoint(save_folder: Path) -> tuple[Path, bool]:
    """Return the newest normal or BLADE sync-boundary checkpoint.

    The boolean indicates a sync-boundary checkpoint, whose immutable run
    fingerprint lives at the save-folder root rather than inside the checkpoint.
    """

    explicit = os.environ.get("EDULLM_RESUME_CHECKPOINT", "").strip()
    if explicit:
        path = Path(explicit)
        if not (path / "model_and_optim" / ".metadata").is_file():
            raise SystemExit(
                f"EDULLM_RESUME_CHECKPOINT is not materialized: {path}"
            )
        return path, path.parent != save_folder

    candidates: list[tuple[int, int, Path, bool]] = []
    for path in save_folder.glob("step*"):
        if (
            not path.is_dir()
            or not path.name.removeprefix("step").isdigit()
            or not (path / "model_and_optim" / ".metadata").is_file()
        ):
            continue
        candidates.append((int(path.name.removeprefix("step")), 2, path, False))

    sync_root = save_folder / "sync_checkpoints"
    for path in (sync_root.glob("step*-*") if sync_root.is_dir() else ()):
        name = path.name.removeprefix("step")
        step_text, separator, phase = name.partition("-")
        if (
            path.is_dir()
            and separator
            and step_text.isdigit()
            and phase in {"pre", "post"}
            and (path / "model_and_optim" / ".metadata").is_file()
        ):
            # A completed normal step wins a tie. Otherwise post-sync wins pre-sync.
            phase_priority = 1 if phase == "post" else 0
            candidates.append((int(step_text), phase_priority, path, True))

    if not candidates:
        raise SystemExit("--resume found no local or restored checkpoint")
    _, _, path, is_sync_boundary = max(candidates)
    return path, is_sync_boundary


def resolve_corpus(**kwargs):
    """Import the dataset reader only in the runtime that needs to stage data."""
    from train_on_corpus import resolve_corpus as resolve

    return resolve(**kwargs)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--local", action="store_true", help="Allow offline/local W&B behavior")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--save-folder", type=Path, default=None)
    result.add_argument("--work-dir", type=Path, default=Path("/tmp/edullm-token-selection"))
    result.add_argument("--progress-dir", type=Path, default=None)
    result.add_argument("--task-loss-script", type=Path, default=None)
    return result


def assert_production_runtime(expected_world_size: int = PRODUCTION_WORLD_SIZE) -> None:
    """Fail before model construction unless torchrun supplied the locked GPU topology."""
    import torch

    from olmo_core.distributed.utils import get_world_size

    world_size = get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if world_size != expected_world_size or local_world_size != expected_world_size:
        raise RuntimeError(
            "production token-selection runs require one 8-GPU torchrun node "
            f"(WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size})"
        )
    visible_devices = torch.cuda.device_count()
    if visible_devices < local_world_size:
        raise RuntimeError(
            f"torchrun started {local_world_size} local ranks but only "
            f"{visible_devices} CUDA devices are visible"
        )


def main() -> None:
    args = parser().parse_args()
    arm = get_arm(args.arm)
    save_folder = args.save_folder or Path(
        os.environ.get("EDULLM_CHECKPOINT_DIR", f"/tmp/checkpoints/{arm.run_id}")
    )
    if str(save_folder).startswith("s3://"):
        raise SystemExit(
            "token-selection outputs must use runtime scratch plus W&B, not an S3 output URI"
        )
    progress_dir = args.progress_dir or Path(
        os.environ.get("EDULLM_PROGRESS_DIR", f"/tmp/progress/{arm.run_id}")
    )
    task_loss_script = args.task_loss_script or Path(os.environ.get("TASK_LOSS_EVAL_SCRIPT", ""))
    if not args.local and not task_loss_script.is_file():
        raise SystemExit(
            "production run requires TASK_LOSS_EVAL_SCRIPT for synchronous 20-label evaluation"
        )

    version = os.environ.get("EDULLM_DATASET_VERSION", "latest")
    if not args.local and version in ("", "latest"):
        raise SystemExit("production requires a pinned EDULLM_DATASET_VERSION")
    corpus = resolve_corpus(
        dataset_id=arm.dataset_id,
        version=version,
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    max_tokens = int(arm.max_tokens if arm.max_tokens is not None else (corpus.rows or 0))
    if max_tokens <= 0:
        raise SystemExit("reference corpus manifest must declare a positive row/token count")
    reference_path = _path("EDULLM_REFERENCE_PATH")
    early_path = _path("EDULLM_EARLY_REFERENCE_PATH")
    late_path = _path("EDULLM_LATE_REFERENCE_PATH")
    passive_path = _path("EDULLM_PASSIVE_REFERENCE_PATH")
    if arm.reference_contract and not reference_path:
        raise SystemExit(f"{arm.name} requires materialized EDULLM_REFERENCE_PATH")
    if arm.early_reference_contract and not early_path:
        raise SystemExit(f"{arm.name} requires EDULLM_EARLY_REFERENCE_PATH")
    if arm.late_reference_contract and not late_path:
        raise SystemExit(f"{arm.name} requires EDULLM_LATE_REFERENCE_PATH")

    refhq_corpus = None
    if arm.requires_refhq_stream:
        refhq_version = os.environ.get("EDULLM_REFHQ_DATASET_VERSION", "latest")
        if not args.local and refhq_version in ("", "latest"):
            raise SystemExit("production BLADE requires a pinned EDULLM_REFHQ_DATASET_VERSION")
        refhq_corpus = resolve_corpus(
            dataset_id=REFHQ,
            version=refhq_version,
            tokenizer_id="tokenizer/dolma2-bpe",
        )
    identity = scientific_identity(
        arm,
        dataset_binding=immutable_corpus_binding(arm.dataset_id, corpus),
        refhq_binding=(
            immutable_corpus_binding(REFHQ, refhq_corpus) if refhq_corpus is not None else None
        ),
        max_tokens=max_tokens,
        reference_path=reference_path,
        early_reference_path=early_path,
        late_reference_path=late_path,
        passive_reference_path=passive_path,
    )
    print(
        json.dumps(
            {
                "arm": arm.name,
                "method": arm.method,
                "dataset": f"{arm.dataset_id}/{corpus.version}",
                "run_id": arm.run_id,
                "wandb_project": arm.wandb_project,
                "max_tokens": max_tokens,
                "total_steps": total_steps(max_tokens),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    save_folder.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: Path | None = None
    if args.resume:
        artifact = os.environ.get("WANDB_RESUME_ARTIFACT", "").strip()
        if artifact and not any(save_folder.glob("step*")):
            restore_checkpoint_artifact(artifact, save_folder)
        resume_checkpoint, is_sync_boundary = _latest_resume_checkpoint(save_folder)
        assert_resume_fingerprint(
            save_folder if is_sync_boundary else resume_checkpoint,
            identity,
        )
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
    elif any(save_folder.iterdir()):
        raise SystemExit(f"fresh run refuses non-empty save folder: {save_folder}")

    from olmo_core.train import prepare_training_environment, teardown_training_environment
    from olmo_core.utils import seed_all

    prepare_training_environment(seed=6198)
    try:
        if not args.local:
            assert_production_runtime()
        torch_imported = __import__("torch")
        torch_imported.set_float32_matmul_precision("high")
        seed_all(6198)
        trainer = build_trainer(
            arm,
            corpus,
            refhq_corpus=refhq_corpus,
            max_tokens=max_tokens,
            save_folder=save_folder,
            work_dir=args.work_dir / arm.name,
            progress_dir=progress_dir,
            task_loss_script=task_loss_script,
            reference_path=reference_path,
            early_reference_path=early_path,
            late_reference_path=late_path,
            passive_reference_path=passive_path,
            resume=args.resume,
            production=not args.local,
        )
        write_identity(save_folder, progress_dir, identity)
        if resume_checkpoint is not None:
            trainer.load_checkpoint(
                resume_checkpoint,
                load_trainer_state=True,
                load_optim_state=True,
            )
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
