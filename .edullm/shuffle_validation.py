#!/usr/bin/env python3
"""Validate the quadratic-MTLD HPO winner with shuffled (non-curriculum) data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.float8 import Float8Config
from olmo_core.hpo.curriculum import (
    PARENT_DATASET_GROUP,
    PARENT_DATASET_ID,
    PARENT_DATASET_VERSION,
    PARENT_MANIFEST_SHA256,
    CurriculumInputIdentity,
)
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig, WSD
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from curriculum_loader import CurriculumDataLoader, ParentChunkDataset
from final_validation import validation_steps
from final_validation_wandb import FinalValidationEvalCallback

GPU_RANKS = 8
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 262_144
RANK_MICROBATCH_TOKENS = 32_768
PRODUCTION_BUDGET_TOKENS = 10_000_000_000
TRAIN_STEPS = PRODUCTION_BUDGET_TOKENS // GLOBAL_BATCH_TOKENS
TRAIN_TOKENS = TRAIN_STEPS * GLOBAL_BATCH_TOKENS
DATA_SEED = 210_007
INIT_SEED = 110_007
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-validation"
VECTOR_NAME = "curriculum-quadratic-mtld-shuffle-baseline"
REFERENCE_CURRICULUM_RUN_SLOT = "quadratic-mtld-370m-mb32k-v2"
PROBE_RUN_ID = "7f74348409e054561b12348d0f5a815b"
PROBE_TRIAL_ID = "t4_0"
PROBE_VALIDATION_CE = 2.7663214206695557
HPS: dict[str, float] = {
    "lr": 0.0006550313203869464,
    "weight_decay": 0.011568982457596117,
    "beta2_gap": 0.021090019944083084,
    "eps": 1.4198780561886085e-07,
    "warmup_fraction": 0.07599507819381664,
    "decay_fraction": 0.1756549080475558,
    "terminal_lr_ratio": 0.027642727977629657,
    "global_batch_mult": 1.4515939717052118,
    "max_grad_norm": 1.2429510134280908,
}
EVAL_SCRIPT = Path(__file__).with_name("task_loss") / "eval_olmo2_370m.py"


class ShuffleValidationError(RuntimeError):
    """The validation configuration violates the fixed experiment contract."""


@dataclass(frozen=True)
class ResolvedParent:
    """Immutable parent token pool used for the shuffle baseline."""

    train_paths: tuple[str, ...]
    dtype: str
    identity: CurriculumInputIdentity


def validate_contract() -> None:
    """Require the transferred vector and effective batch to match the curriculum run."""
    if GLOBAL_BATCH_TOKENS != 256 * 1024:
        raise ShuffleValidationError("validation global batch must remain fixed at 256 Ki tokens")
    if GLOBAL_BATCH_TOKENS % (GPU_RANKS * SEQUENCE_LENGTH):
        raise ShuffleValidationError("global batch must divide into whole sequences on 8 ranks")
    if GLOBAL_BATCH_TOKENS % RANK_MICROBATCH_TOKENS:
        raise ShuffleValidationError("global batch must contain whole rank microbatches")
    if TRAIN_TOKENS > PRODUCTION_BUDGET_TOKENS or TRAIN_STEPS <= 0:
        raise ShuffleValidationError("invalid aligned 10B horizon")
    if HPS["warmup_fraction"] + HPS["decay_fraction"] > 1.0:
        raise ShuffleValidationError("warmup and decay fractions overlap")


def resolve_parent() -> ResolvedParent:
    """Resolve the sealed parent token pool without a curriculum order."""
    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    parent = dataset_paths(
        PARENT_DATASET_ID,
        PARENT_DATASET_VERSION,
        group=PARENT_DATASET_GROUP,
        s3=Boto3S3.default(),
    )
    manifest = getattr(parent, "manifest_sha256", None)
    if manifest != PARENT_MANIFEST_SHA256:
        raise ShuffleValidationError(f"parent manifest mismatch: {manifest!r}")
    train_paths = tuple(str(path) for path in parent.paths)
    if not train_paths:
        raise ShuffleValidationError("parent input has no train partition")
    source_ids: list[str] = []
    for path in train_paths:
        source = Path(path).parent.name
        if source and source not in source_ids:
            source_ids.append(source)
    identity = CurriculumInputIdentity(
        dataset_id=PARENT_DATASET_ID,
        version=PARENT_DATASET_VERSION,
        group=PARENT_DATASET_GROUP,
        profile=str(getattr(parent, "profile", "pretrain-tokens/v1")),
        manifest_sha256=PARENT_MANIFEST_SHA256,
        source_ids=tuple(source_ids),
    )
    return ResolvedParent(
        train_paths=train_paths,
        dtype=str(parent.dtype),
        identity=identity,
    )


@dataclass
class ShuffleExperimentConfig:
    """Dense OLMo2-370M shuffle baseline with the fixed curriculum HPO winner."""

    model: TransformerConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    parent: ResolvedParent
    total_steps: int
    init_seed: int
    work_dir: str

    def as_config_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.as_config_dict(),
            "trainer": self.trainer.as_config_dict(),
            "train_module": self.train_module.as_config_dict(),
            "parent": self.parent.identity.as_dict(),
            "total_steps": self.total_steps,
            "init_seed": self.init_seed,
            "pacing": "control",
            "curriculum_learning": False,
        }


def build_experiment_config(
    parent: ResolvedParent,
    *,
    save_folder: str,
    length_tokens: int | None = None,
    work_dir: str = "/tmp/olmo-core/shuffle-validation",
    environ: Mapping[str, str] = os.environ,
) -> ShuffleExperimentConfig:
    """Build dense OLMo2-370M with the fixed winner and shuffled parent chunks."""
    validate_contract()
    requested_tokens = TRAIN_TOKENS if length_tokens is None else int(length_tokens)
    if requested_tokens <= 0 or requested_tokens % GLOBAL_BATCH_TOKENS:
        raise ShuffleValidationError(
            f"length_tokens must be a positive multiple of {GLOBAL_BATCH_TOKENS}"
        )
    total_steps = requested_tokens // GLOBAL_BATCH_TOKENS
    ladder = validation_steps(total_steps, min(VALIDATION_POINTS, total_steps + 1))
    tokenizer = TokenizerConfig.dolma2()
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_TOKENS,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=HPS["lr"],
            betas=(0.9, 1.0 - HPS["beta2_gap"]),
            eps=HPS["eps"],
            weight_decay=HPS["weight_decay"],
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=WSD(
            units=SchedulerUnits.tokens,
            warmup_fraction=HPS["warmup_fraction"],
            decay_fraction=HPS["decay_fraction"],
            decay_min_lr=HPS["terminal_lr_ratio"] * HPS["lr"],
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
        max_grad_norm=HPS["max_grad_norm"],
    )
    run_name = environ.get("WANDB_NAME") or environ.get("EDULLM_RUN_ID") or VECTOR_NAME
    project = environ.get("EDULLM_WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            work_dir=work_dir,
            max_duration=Duration.steps(total_steps),
            metrics_collect_interval=5,
            cancel_check_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=None,
                fixed_steps=ladder[1:],
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
                group=environ.get(
                    "WANDB_RUN_GROUP", "hpo-validation-olmo2-370m-quadratic-mtld-shuffle"
                ),
                enabled=bool(project),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            FinalValidationEvalCallback(
                vector_name=VECTOR_NAME,
                total_steps=total_steps,
                checkpoint_steps=ladder,
                save_folder=save_folder,
                run_name=run_name,
                work_dir=environ.get(
                    "EDULLM_EVAL_WORK_DIR", str(Path(work_dir) / "task-loss-eval")
                ),
                eval_script=EVAL_SCRIPT,
                nproc=GPU_RANKS,
            ),
        )
    )
    return ShuffleExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        trainer=trainer,
        train_module=train_module,
        parent=parent,
        total_steps=total_steps,
        init_seed=INIT_SEED,
        work_dir=work_dir,
    )


def scientific_identity(config: ShuffleExperimentConfig) -> dict[str, Any]:
    """Return the immutable identity persisted with checkpoints and W&B."""
    return {
        "schema_version": 1,
        "model": "olmo2_370M",
        "dataset_id": PARENT_DATASET_ID,
        "dataset_version": PARENT_DATASET_VERSION,
        "training_mode": "shuffle_baseline",
        "curriculum_learning": False,
        "data_ordering": "deterministic_no_replacement_shuffle",
        "pacing": "control",
        "sequence_length": SEQUENCE_LENGTH,
        "data_seed": DATA_SEED,
        "init_seed": INIT_SEED,
        "budget_tokens_requested": PRODUCTION_BUDGET_TOKENS,
        "train_tokens": config.total_steps * GLOBAL_BATCH_TOKENS,
        "total_steps": config.total_steps,
        "checkpoint_steps": validation_steps(config.total_steps),
        "probe_run_id": PROBE_RUN_ID,
        "probe_trial_id": PROBE_TRIAL_ID,
        "probe_search_validation_ce": PROBE_VALIDATION_CE,
        "optimized_hps": dict(HPS),
        "hpo_global_batch_mult": HPS["global_batch_mult"],
        "validation_global_batch_override": True,
        "validation_global_batch_override_reason": "align with the other HPO validation runs",
        "effective_global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "reference_curriculum_run_slot": REFERENCE_CURRICULUM_RUN_SLOT,
        "parent": config.parent.identity.as_dict(),
    }


def run_training(config: ShuffleExperimentConfig, identity: Mapping[str, Any]) -> None:
    """Build standard OLMo-core components and resume from the configured folder."""
    prepare_training_environment(seed=config.init_seed, shared_filesystem=False)
    try:
        tokenizer = TokenizerConfig.dolma2()
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        dataset = ParentChunkDataset(
            config.parent.train_paths,
            sequence_length=SEQUENCE_LENGTH,
            dtype=config.parent.dtype,
        )
        dp_process_group = train_module.dp_process_group
        loader = CurriculumDataLoader(
            dataset,
            ranked_chunk_indices=None,
            pacing="control",
            difficulty_metric=None,
            seed=DATA_SEED,
            total_steps=config.total_steps,
            global_batch_size=GLOBAL_BATCH_TOKENS,
            work_dir=config.work_dir,
            parent_identity=config.parent.identity.as_dict(),
            order_identity=None,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
        )
        trainer = config.trainer.build(train_module, loader)
        config_saver = trainer.callbacks["config_saver"]
        assert isinstance(config_saver, ConfigSaverCallback)
        config_saver.config = {**config.as_config_dict(), "scientific_identity": dict(identity)}
        trainer.maybe_load_checkpoint()
        trainer.fit()
    finally:
        teardown_training_environment()


def platform_values(environ: Mapping[str, str]) -> tuple[str, str]:
    """Validate staged dataset and output identities."""
    actual = (
        environ.get("EDULLM_DATASET_ID", ""),
        environ.get("EDULLM_DATASET_VERSION", ""),
        environ.get("EDULLM_DATASET_TOKENIZER", ""),
    )
    expected = (PARENT_DATASET_ID, PARENT_DATASET_VERSION, "tokenizer/dolma2-bpe")
    if actual != expected:
        raise ShuffleValidationError(f"platform dataset mismatch: {actual!r}")
    checkpoint_dir = environ.get("EDULLM_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        raise ShuffleValidationError("EDULLM_CHECKPOINT_DIR is required")
    return checkpoint_dir, environ.get("EDULLM_RUN_ID", VECTOR_NAME)


def torchrun_command(length_tokens: int | None) -> list[str]:
    """Build the fixed eight-rank training command."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={GPU_RANKS}",
        str(Path(__file__).resolve()),
        "--train-worker",
    ]
    if length_tokens is not None:
        command.extend(["--length-tokens", str(length_tokens)])
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--length-tokens", type=int)
    result.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--describe", action="store_true")
    return result


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], ResolvedParent] = resolve_parent,
) -> int:
    args = parser().parse_args(argv)
    try:
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker and not args.describe:
            os.execv(sys.executable, torchrun_command(args.length_tokens))
        parent = resolver()
        config = build_experiment_config(
            parent,
            save_folder=checkpoint_dir,
            length_tokens=args.length_tokens,
            environ=os.environ,
        )
        identity = scientific_identity(config)
        if args.describe:
            print(json.dumps(identity, indent=2, sort_keys=True))
            return 0
        if int(os.environ.get("WORLD_SIZE", "0")) != GPU_RANKS:
            raise ShuffleValidationError(f"worker requires WORLD_SIZE={GPU_RANKS}")
        os.environ["WANDB_NAME"] = f"{run_id}-{VECTOR_NAME}"
        run_training(config, identity)
    except ShuffleValidationError as exc:
        print(f"[shuffle-validation] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
