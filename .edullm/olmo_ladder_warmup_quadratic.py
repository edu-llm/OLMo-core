#!/usr/bin/env python3
"""Train OLMo-ladder 370M / ~10B RegMix with warmup-quadratic MTLD curriculum."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from olmo_core.config import DType
from olmo_core.data import NumpyDatasetDType, TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.hpo.curriculum import (
    ARM9_PACING_ID,
    CurriculumCorpus,
    CurriculumDataLoaderConfig,
    CurriculumExperimentConfig,
    CurriculumInputIdentity,
    ParentChunkDatasetConfig,
    token_phase_boundaries,
)
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_validation import (  # noqa: E402
    EVAL_SCRIPT,
    FinalValidationConfigError,
    platform_values,
    run_training,
    validation_steps,
)
from final_validation_wandb import FinalValidationEvalCallback  # noqa: E402

ARM_NAME = "olmo-ladder-warmup-quadratic"
MODEL_NAME = "olmo2_370M"
DATASET_ID = "pretrain/regmix-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
ORDER_DATASET_ID = "curriculum/regmix-370m"
ORDER_GROUP = "mtld"
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 256 * 1_024
WORLD_SIZE = 8
RANK_MICROBATCH_TOKENS = 16 * 1_024
TARGET_TOKENS = 10_000_000_000
TOTAL_STEPS = TARGET_TOKENS // GLOBAL_BATCH_TOKENS
TRAIN_TOKENS = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
PEAK_LR = 7.78548e-4
TERMINAL_LR_RATIO = 0.1
WARMUP_FRACTION = 0.005
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
SEED = 12_536
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-ladder"
DEFAULT_WANDB_GROUP = "hpo-ladder-curriculum"
DATA_BUCKET = "edullm-data"


def _source_ids(paths: tuple[str, ...]) -> tuple[str, ...]:
    source_ids = []
    for path in paths:
        source = PurePosixPath(str(path).replace("\\", "/")).parent.name
        if source and source not in source_ids:
            source_ids.append(source)
    return tuple(source_ids)


def _load_json(s3: Any, key: str) -> dict[str, Any]:
    payload = json.loads(s3.get(DATA_BUCKET, key).decode("utf-8"))
    if not isinstance(payload, dict):
        raise FinalValidationConfigError(f"{key} is not a JSON object")
    return payload


def _group(dataset: Mapping[str, Any], name: str | None) -> Mapping[str, Any]:
    groups = dataset.get("groups") or []
    if name is None:
        if len(groups) != 1:
            raise FinalValidationConfigError(
                f"parent dataset must have one unambiguous group, found {len(groups)}"
            )
        return groups[0]
    matches = [group for group in groups if group.get("name") == name]
    if len(matches) != 1:
        raise FinalValidationConfigError(f"expected one group {name!r}, found {len(matches)}")
    return matches[0]


def _read_dtype(read: Any, *, role: str) -> NumpyDatasetDType:
    if int(getattr(read, "header_bytes", 0) or 0) != 0:
        raise FinalValidationConfigError(f"{role} shards must be headerless")
    byte_order = getattr(read, "byte_order", None)
    if byte_order not in (None, sys.byteorder):
        raise FinalValidationConfigError(
            f"{role} byte order {byte_order!r} does not match host {sys.byteorder!r}"
        )
    dtype = getattr(read, "dtype", None)
    if dtype is None:
        raise FinalValidationConfigError(f"{role} declares no fixed-width dtype")
    return NumpyDatasetDType(dtype)


def _identity(
    read: Any,
    *,
    dataset_id: str,
    version: str,
    group: Mapping[str, Any],
    profile: str,
    source_ids: tuple[str, ...] = (),
) -> CurriculumInputIdentity:
    manifest = getattr(read, "manifest_sha256", None) or group.get("manifest_sha256")
    if not isinstance(manifest, str) or len(manifest) != 64:
        raise FinalValidationConfigError(
            f"{dataset_id}/{version} group {group.get('name')!r} has no manifest hash"
        )
    return CurriculumInputIdentity(
        dataset_id=dataset_id,
        version=version,
        group=str(group["name"]),
        profile=profile,
        manifest_sha256=manifest,
        source_ids=source_ids,
    )


def resolve_curriculum() -> CurriculumCorpus:
    """Resolve sealed RegMix tokens and the MTLD order bound to that parent."""

    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    parent_doc = _load_json(s3, f"{DATASET_ID}/{DATASET_VERSION}/dataset.json")
    parent_group = _group(parent_doc, None)
    parent_read = dataset_paths(
        DATASET_ID,
        DATASET_VERSION,
        s3=s3,
        group=str(parent_group["name"]),
    )
    parent_paths = tuple(str(path) for path in parent_read.paths)
    if not parent_paths:
        raise FinalValidationConfigError("RegMix resolved no trainable paths")
    parent_identity = _identity(
        parent_read,
        dataset_id=DATASET_ID,
        version=DATASET_VERSION,
        group=parent_group,
        profile="pretrain-tokens/v1",
        source_ids=_source_ids(parent_paths),
    )

    order_version = os.environ.get("CURRICULUM_DATASET_VERSION") or resolve_latest(
        ORDER_DATASET_ID, s3=s3
    )
    if not order_version:
        raise FinalValidationConfigError(f"no published version of {ORDER_DATASET_ID}")
    order_doc = _load_json(s3, f"{ORDER_DATASET_ID}/{order_version}/dataset.json")
    order_group = _group(order_doc, ORDER_GROUP)
    if order_group.get("profile") != "token-order/v1":
        raise FinalValidationConfigError(
            f"order profile must be token-order/v1, got {order_group.get('profile')!r}"
        )
    dependencies = [
        dependency
        for dependency in order_group.get("depends_on") or []
        if dependency.get("role") == "token_pool"
    ]
    if len(dependencies) != 1:
        raise FinalValidationConfigError("order group must declare exactly one token_pool")
    expected = {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "manifest_sha256": parent_identity.manifest_sha256,
    }
    actual = {key: dependencies[0].get(key) for key in expected}
    if actual != expected:
        raise FinalValidationConfigError(
            f"order binds {actual!r}, not the staged parent {expected!r}"
        )
    order_read = dataset_paths(
        ORDER_DATASET_ID,
        order_version,
        split="train",
        s3=s3,
        group=ORDER_GROUP,
    )
    order_paths = tuple(str(path) for path in order_read.paths)
    if not order_paths:
        raise FinalValidationConfigError("MTLD input has no order partition")
    return CurriculumCorpus(
        train_paths=parent_paths,
        val_paths=tuple(str(path) for path in (getattr(parent_read, "val", None) or ())),
        order_paths=order_paths,
        dtype=_read_dtype(parent_read, role="parent"),
        order_dtype=_read_dtype(order_read, role="order"),
        parent_identity=parent_identity,
        order_identity=_identity(
            order_read,
            dataset_id=ORDER_DATASET_ID,
            version=order_version,
            group=order_group,
            profile="token-order/v1",
        ),
    )


def build_experiment_config(
    corpus: CurriculumCorpus,
    *,
    save_folder: str,
    length_steps: int = TOTAL_STEPS,
    work_dir: str = "/tmp/olmo-core/olmo-ladder-warmup-quadratic",
    environ: Mapping[str, str] = os.environ,
) -> CurriculumExperimentConfig:
    """Build the ladder-control recipe with token-progress warmup-quadratic MTLD."""

    if length_steps < VALIDATION_POINTS - 1:
        raise FinalValidationConfigError(
            f"length_steps must be at least {VALIDATION_POINTS - 1} for endpoint evaluations"
        )
    tokenizer = TokenizerConfig.dolma2()
    checkpoints = validation_steps(length_steps, points=VALIDATION_POINTS)
    train_tokens = length_steps * GLOBAL_BATCH_TOKENS
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}
    run_name = environ.get("WANDB_NAME") or environ.get("EDULLM_RUN_ID") or ARM_NAME
    project = environ.get("EDULLM_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)

    dataset = ParentChunkDatasetConfig(
        paths=list(corpus.train_paths),
        sequence_length=SEQUENCE_LENGTH,
        dtype=corpus.dtype,
    )
    loader = CurriculumDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_TOKENS,
        seed=SEED,
        target_tokens=train_tokens,
        order_paths=list(corpus.order_paths),
        order_dtype=corpus.order_dtype,
        parent_identity=corpus.parent_identity,
        order_identity=corpus.order_identity,
        tokenizer=tokenizer,
        work_dir=work_dir,
        pacing=ARM9_PACING_ID,
        difficulty_metric="mtld",
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_TOKENS,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=PEAK_LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
            fused=True,
        ),
        scheduler=CosWithWarmup(
            warmup_fraction=WARMUP_FRACTION,
            alpha_f=TERMINAL_LR_RATIO,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=MAX_GRAD_NORM,
    )
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            work_dir=work_dir,
            max_duration=Duration.steps(length_steps),
            metrics_collect_interval=5,
            cancel_check_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
                fixed_steps=checkpoints[1:],
                ephemeral_save_interval=None,
                pre_train_checkpoint=not skip_pre_train,
                save_async=True,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project=project,
                group=environ.get("WANDB_RUN_GROUP", DEFAULT_WANDB_GROUP),
                enabled=bool(project),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            FinalValidationEvalCallback(
                vector_name=ARM_NAME,
                total_steps=length_steps,
                checkpoint_steps=checkpoints,
                save_folder=save_folder,
                run_name=run_name,
                work_dir=environ.get(
                    "EDULLM_EVAL_WORK_DIR", str(Path(work_dir) / "task-loss-eval")
                ),
                eval_script=EVAL_SCRIPT,
                nproc=WORLD_SIZE,
            ),
        )
    )
    identity = {
        "parent": corpus.parent_identity.as_dict(),
        "order": corpus.order_identity.as_dict(),
        "pacing": ARM9_PACING_ID,
        "token_phase_boundaries": list(token_phase_boundaries(train_tokens)),
        "difficulty_metric": "mtld",
        "target_tokens": train_tokens,
        "sequence_length": SEQUENCE_LENGTH,
    }
    return CurriculumExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=loader,
        trainer=trainer,
        train_module=train_module,
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        init_seed=SEED,
        curriculum_identity=identity,
    )


def scientific_identity(config: CurriculumExperimentConfig) -> dict[str, Any]:
    """Return the fixed experiment identity persisted beside checkpoints and in W&B."""

    total_steps = int(config.trainer.max_duration.value)
    return {
        "schema_version": 1,
        "arm": ARM_NAME,
        "control": "warmup_quadratic_mtld",
        "curriculum_learning": True,
        "model": MODEL_NAME,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "curriculum_dataset_id": ORDER_DATASET_ID,
        "curriculum_order_group": ORDER_GROUP,
        "pacing": ARM9_PACING_ID,
        "tokenizer_id": TOKENIZER_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "world_size": WORLD_SIZE,
        "budget_tokens_requested": TARGET_TOKENS,
        "train_tokens": total_steps * GLOBAL_BATCH_TOKENS,
        "total_steps": total_steps,
        "checkpoint_steps": validation_steps(total_steps, points=VALIDATION_POINTS),
        "optimizer": {
            "name": "AdamW",
            "lr": PEAK_LR,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "embedding_weight_decay": 0.0,
        },
        "scheduler": {
            "name": "cos_with_warmup",
            "warmup_fraction": WARMUP_FRACTION,
            "terminal_lr_ratio": TERMINAL_LR_RATIO,
            "terminal_lr": PEAK_LR * TERMINAL_LR_RATIO,
        },
        "max_grad_norm": MAX_GRAD_NORM,
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "seed": SEED,
        "curriculum_identity": dict(config.curriculum_identity or {}),
    }


def torchrun_command(length_steps: int | None = None) -> list[str]:
    """Build the fixed eight-rank launch command."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--train-worker",
    ]
    if length_steps is not None:
        command.extend(["--length-steps", str(length_steps)])
    return command


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], CurriculumCorpus] = resolve_curriculum,
) -> int:
    """Launch or execute the warmup-quadratic ladder arm."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length-steps",
        type=int,
        help="smoke-only duration override; production omits this for the full 10B budget",
    )
    parser.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker:
            os.execv(sys.executable, torchrun_command(args.length_steps))
        if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
            raise FinalValidationConfigError(f"worker requires WORLD_SIZE={WORLD_SIZE}")
        os.environ["WANDB_NAME"] = f"{run_id}-{ARM_NAME}"
        config = build_experiment_config(
            resolver(),
            save_folder=checkpoint_dir,
            length_steps=args.length_steps or TOTAL_STEPS,
            environ=os.environ,
        )
        run_training(config, scientific_identity(config))
    except FinalValidationConfigError as exc:
        print(f"[olmo-ladder-warmup-quadratic] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
