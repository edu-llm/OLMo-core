#!/usr/bin/env python3
"""Validate quadratic-MTLD curriculum with the RegMix no-proxy HPO winner HPs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.hpo.curriculum import (
    ARM9_PACING_ID,
    CURRICULUM_DATASET_ID,
    CURRICULUM_DATASET_VERSION,
    CURRICULUM_ORDER_GROUP,
    PARENT_DATASET_GROUP,
    PARENT_DATASET_ID,
    PARENT_DATASET_VERSION,
    CurriculumDataLoaderConfig,
    CurriculumExperimentConfig,
    CurriculumCorpus,
    ParentChunkDatasetConfig,
    curriculum_corpus_from_reads,
    token_phase_boundaries,
)
from olmo_core.nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig, WSD
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import TransformerDataParallelConfig, TransformerTrainModuleConfig

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_validation import validation_steps
from final_validation_wandb import FinalValidationEvalCallback

GPU_RANKS = 8
SEQUENCE_LENGTH = 2_048
BASE_HPO_GLOBAL_BATCH_TOKENS = 524_288
GLOBAL_BATCH_TOKENS = 262_144
RANK_MICROBATCH_TOKENS = 32_768
PRODUCTION_BUDGET_TOKENS = 10_000_000_000
TRAIN_STEPS = PRODUCTION_BUDGET_TOKENS // GLOBAL_BATCH_TOKENS
TRAIN_TOKENS = TRAIN_STEPS * GLOBAL_BATCH_TOKENS
DATA_SEED = 210_007
INIT_SEED = 110_007
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-validation"
VECTOR_NAME = "curriculum-quadratic-mtld-no-proxy-hps"
REFERENCE_CURRICULUM_RUN = "hpo-validation-olmo2-370m-quadratic-mtld-20260810-062816"
REFERENCE_CURRICULUM_WANDB = "3576162a7edea5d8bebeafdf50053740"
# Provenance: RegMix no-proxy probe winner (t9_0), transferred onto curriculum pacing.
PROBE_RUN_ID = "904ea39d368dfe412048a6063c1600df"
PROBE_TRIAL_ID = "t9_0"
PROBE_ARM = "no-proxy"
PROBE_VALIDATION_CE = 2.786524534225464
HPS: dict[str, float] = {
    "lr": 0.0004125460019173203,
    "weight_decay": 0.01473432082609167,
    "beta2_gap": 0.0014689794923786166,
    "eps": 1.4798352708540092e-12,
    "warmup_fraction": 0.01131976840436488,
    "decay_fraction": 0.05,
    "terminal_lr_ratio": 0.021488995515927797,
    "global_batch_mult": 0.5,
    "max_grad_norm": 0.3,
}
EVAL_SCRIPT = Path(__file__).with_name("task_loss") / "eval_olmo2_370m.py"


class CurriculumValidationError(RuntimeError):
    """The validation configuration violates the fixed experiment contract."""


def validate_contract() -> None:
    """Require the fixed 256 Ki validation batch and a valid WSD schedule."""
    if GLOBAL_BATCH_TOKENS != 256 * 1024:
        raise CurriculumValidationError("validation global batch must remain fixed at 256 Ki tokens")
    if GLOBAL_BATCH_TOKENS % (GPU_RANKS * SEQUENCE_LENGTH):
        raise CurriculumValidationError("global batch must divide into whole sequences on 8 ranks")
    if GLOBAL_BATCH_TOKENS % RANK_MICROBATCH_TOKENS:
        raise CurriculumValidationError("global batch must contain whole rank microbatches")
    if TRAIN_TOKENS > PRODUCTION_BUDGET_TOKENS or TRAIN_STEPS <= 0:
        raise CurriculumValidationError("invalid aligned 10B horizon")
    if HPS["warmup_fraction"] + HPS["decay_fraction"] > 1.0:
        raise CurriculumValidationError("warmup and decay fractions overlap")


def resolve_curriculum() -> CurriculumCorpus:
    """Resolve the sealed parent token pool and immutable MTLD order."""
    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    parent = dataset_paths(
        PARENT_DATASET_ID,
        PARENT_DATASET_VERSION,
        group=PARENT_DATASET_GROUP,
        s3=s3,
    )
    order = dataset_paths(
        CURRICULUM_DATASET_ID,
        CURRICULUM_DATASET_VERSION,
        split="train",
        group=CURRICULUM_ORDER_GROUP,
        s3=s3,
    )
    return curriculum_corpus_from_reads(parent, order)


def build_experiment_config(
    corpus: CurriculumCorpus,
    *,
    save_folder: str,
    length_tokens: int | None = None,
    work_dir: str = "/tmp/olmo-core/curriculum-noproxy-validation",
    environ: Mapping[str, str] = os.environ,
) -> CurriculumExperimentConfig:
    """Build dense OLMo2-370M with no-proxy HPs and quadratic MTLD loader."""
    validate_contract()
    requested_tokens = TRAIN_TOKENS if length_tokens is None else int(length_tokens)
    if requested_tokens <= 0 or requested_tokens % GLOBAL_BATCH_TOKENS:
        raise CurriculumValidationError(
            f"length_tokens must be a positive multiple of {GLOBAL_BATCH_TOKENS}"
        )
    total_steps = requested_tokens // GLOBAL_BATCH_TOKENS
    ladder = validation_steps(total_steps, min(VALIDATION_POINTS, total_steps + 1))
    tokenizer = TokenizerConfig.dolma2()
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}

    dataset = ParentChunkDatasetConfig(
        paths=list(corpus.train_paths),
        sequence_length=SEQUENCE_LENGTH,
        dtype=corpus.dtype,
    )
    loader = CurriculumDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_TOKENS,
        seed=DATA_SEED,
        target_tokens=requested_tokens,
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
                    "WANDB_RUN_GROUP", "hpo-validation-olmo2-370m-quadratic-mtld-noproxy-hps"
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
    identity = {
        "parent": corpus.parent_identity.as_dict(),
        "order": corpus.order_identity.as_dict(),
        "pacing": ARM9_PACING_ID,
        "token_phase_boundaries": list(token_phase_boundaries(requested_tokens)),
        "difficulty_metric": "mtld",
        "target_tokens": requested_tokens,
        "sequence_length": SEQUENCE_LENGTH,
    }
    return CurriculumExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=loader,
        trainer=trainer,
        train_module=train_module,
        dataset_id=PARENT_DATASET_ID,
        dataset_version=PARENT_DATASET_VERSION,
        init_seed=INIT_SEED,
        curriculum_identity=identity,
    )


def scientific_identity(config: CurriculumExperimentConfig) -> dict[str, Any]:
    """Return the immutable identity persisted with checkpoints and W&B."""
    total_steps = int(config.trainer.max_duration.value)
    return {
        "schema_version": 1,
        "model": "olmo2_370M",
        "dataset_id": PARENT_DATASET_ID,
        "dataset_version": PARENT_DATASET_VERSION,
        "curriculum_dataset_id": CURRICULUM_DATASET_ID,
        "curriculum_dataset_version": CURRICULUM_DATASET_VERSION,
        "curriculum_order_group": CURRICULUM_ORDER_GROUP,
        "pacing": ARM9_PACING_ID,
        "curriculum_learning": True,
        "hyperparameter_source": {
            "probe_arm": PROBE_ARM,
            "probe_run_id": PROBE_RUN_ID,
            "probe_trial_id": PROBE_TRIAL_ID,
            "probe_search_validation_ce": PROBE_VALIDATION_CE,
            "note": "RegMix no-proxy winner HPs transferred onto quadratic-MTLD curriculum pacing",
        },
        "reference_curriculum_run": REFERENCE_CURRICULUM_RUN,
        "reference_curriculum_wandb": REFERENCE_CURRICULUM_WANDB,
        "sequence_length": SEQUENCE_LENGTH,
        "data_seed": DATA_SEED,
        "init_seed": INIT_SEED,
        "budget_tokens_requested": PRODUCTION_BUDGET_TOKENS,
        "train_tokens": total_steps * GLOBAL_BATCH_TOKENS,
        "total_steps": total_steps,
        "checkpoint_steps": validation_steps(total_steps),
        "optimized_hps": dict(HPS),
        "hpo_global_batch_mult": HPS["global_batch_mult"],
        "validation_global_batch_override": True,
        "validation_global_batch_override_reason": "align with the other HPO validation runs",
        "effective_global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "curriculum_identity": dict(config.curriculum_identity or {}),
    }


def run_training(config: CurriculumExperimentConfig, identity: Mapping[str, Any]) -> None:
    """Build standard OLMo-core components and resume from the configured folder."""
    prepare_training_environment(seed=config.init_seed, shared_filesystem=False)
    try:
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        dataset = config.dataset.build()
        loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
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
        raise CurriculumValidationError(f"platform dataset mismatch: {actual!r}")
    checkpoint_dir = environ.get("EDULLM_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        raise CurriculumValidationError("EDULLM_CHECKPOINT_DIR is required")
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
    resolver: Callable[[], CurriculumCorpus] = resolve_curriculum,
) -> int:
    args = parser().parse_args(argv)
    try:
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker and not args.describe:
            os.execv(sys.executable, torchrun_command(args.length_tokens))
        corpus = resolver()
        config = build_experiment_config(
            corpus,
            save_folder=checkpoint_dir,
            length_tokens=args.length_tokens,
            environ=os.environ,
        )
        identity = scientific_identity(config)
        if args.describe:
            print(json.dumps(identity, indent=2, sort_keys=True))
            return 0
        if int(os.environ.get("WORLD_SIZE", "0")) != GPU_RANKS:
            raise CurriculumValidationError(f"worker requires WORLD_SIZE={GPU_RANKS}")
        os.environ["WANDB_NAME"] = f"{run_id}-{VECTOR_NAME}"
        run_training(config, identity)
    except CurriculumValidationError as exc:
        print(f"[curriculum-noproxy-validation] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
