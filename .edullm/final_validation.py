#!/usr/bin/env python3
"""Train stock OLMo2-370M for 10B RegMix tokens with a winning probe vector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import OptimGroupOverride, SchedulerUnits, SkipStepAdamWConfig, WSD
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import CheckpointerCallback, ConfigSaverCallback, WandBCallback
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_validation_wandb import FinalValidationEvalCallback

VECTORS_PATH = Path(__file__).with_name("final-validation-vectors.json")
EVAL_SCRIPT = Path(__file__).with_name("task_loss") / "eval_olmo2_370m.py"
DATASET_ID = "pretrain/regmix-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
GPU_RANKS = 8
SEQUENCE_LENGTH = 2_048
BASE_PROBE_GLOBAL_BATCH_TOKENS = 32_768
PRODUCTION_BUDGET_TOKENS = 10_000_000_000
SEED = 12_536
VALIDATION_POINTS = 21
DEFAULT_WANDB_PROJECT = "hpo-final-validation"
OPTIMIZED_FIELDS = frozenset(
    {
        "lr",
        "weight_decay",
        "beta2_gap",
        "eps",
        "warmup_fraction",
        "decay_fraction",
        "terminal_lr_ratio",
        "global_batch_mult",
        "max_grad_norm",
    }
)


class FinalValidationConfigError(RuntimeError):
    """The final-validation configuration violates the fixed experiment contract."""


@dataclass(frozen=True)
class WinningVector:
    """One immutable probe winner and its provenance."""

    name: str
    probe_arm: str
    probe_run_id: str
    probe_trial_id: str
    hps: Mapping[str, float]

    @property
    def global_batch_tokens(self) -> int:
        return round(BASE_PROBE_GLOBAL_BATCH_TOKENS * float(self.hps["global_batch_mult"]))

    @property
    def total_steps(self) -> int:
        return PRODUCTION_BUDGET_TOKENS // self.global_batch_tokens

    @property
    def train_tokens(self) -> int:
        return self.total_steps * self.global_batch_tokens


@dataclass(frozen=True)
class ResolvedRegMix:
    """Validated train paths from the sealed RegMix release."""

    paths: tuple[str, ...]
    dtype: NumpyDatasetDType


def load_vectors(path: Path = VECTORS_PATH) -> dict[str, WinningVector]:
    """Load the preregistered probe winners."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise FinalValidationConfigError("winner-vector file must use schema_version=1")
    if payload.get("dataset") != {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "tokenizer_id": TOKENIZER_ID,
    }:
        raise FinalValidationConfigError("winner-vector file changed the fixed RegMix dataset")
    result: dict[str, WinningVector] = {}
    for item in payload.get("vectors") or ():
        hps = {str(key): float(value) for key, value in (item.get("hps") or {}).items()}
        if frozenset(hps) != OPTIMIZED_FIELDS:
            missing = sorted(OPTIMIZED_FIELDS - frozenset(hps))
            extra = sorted(frozenset(hps) - OPTIMIZED_FIELDS)
            raise FinalValidationConfigError(
                f"{item.get('name')}: optimized fields differ (missing={missing}, extra={extra})"
            )
        vector = WinningVector(
            name=str(item["name"]),
            probe_arm=str(item["probe_arm"]),
            probe_run_id=str(item["probe_run_id"]),
            probe_trial_id=str(item["probe_trial_id"]),
            hps=hps,
        )
        if vector.name in result:
            raise FinalValidationConfigError(f"duplicate winner-vector name: {vector.name}")
        validate_vector(vector)
        result[vector.name] = vector
    if not result:
        raise FinalValidationConfigError("winner-vector file contains no vectors")
    return result


def validate_vector(vector: WinningVector) -> None:
    """Validate transfer-sensitive probe hyperparameters."""

    hps = vector.hps
    batch = vector.global_batch_tokens
    if batch <= 0 or batch % (GPU_RANKS * SEQUENCE_LENGTH):
        raise FinalValidationConfigError(
            f"{vector.name}: global batch {batch} must divide into whole sequences on 8 ranks"
        )
    if not 0.0 < hps["lr"] or not 0.0 <= hps["weight_decay"]:
        raise FinalValidationConfigError(f"{vector.name}: invalid optimizer scale")
    if not 0.0 < hps["beta2_gap"] < 1.0 or not 0.0 < hps["eps"]:
        raise FinalValidationConfigError(f"{vector.name}: invalid Adam second-moment settings")
    for field in ("warmup_fraction", "decay_fraction", "terminal_lr_ratio"):
        if not 0.0 <= hps[field] <= 1.0:
            raise FinalValidationConfigError(f"{vector.name}: {field} must be in [0, 1]")
    if hps["warmup_fraction"] + hps["decay_fraction"] > 1.0:
        raise FinalValidationConfigError(f"{vector.name}: warmup and decay overlap")
    if not 0.0 < hps["max_grad_norm"]:
        raise FinalValidationConfigError(f"{vector.name}: max_grad_norm must be positive")


def with_global_batch_tokens(vector: WinningVector, global_batch_tokens: int) -> WinningVector:
    """Return a vector rebound to an explicit whole-sequence global batch."""

    global_batch_tokens = int(global_batch_tokens)
    if global_batch_tokens <= 0 or global_batch_tokens % (GPU_RANKS * SEQUENCE_LENGTH):
        raise FinalValidationConfigError(
            "global batch must be a positive multiple of one sequence on every rank "
            f"({GPU_RANKS * SEQUENCE_LENGTH} tokens)"
        )
    hps = dict(vector.hps)
    hps["global_batch_mult"] = global_batch_tokens / BASE_PROBE_GLOBAL_BATCH_TOKENS
    overridden = replace(vector, hps=hps)
    validate_vector(overridden)
    return overridden


def validation_steps(total_steps: int, points: int = VALIDATION_POINTS) -> list[int]:
    """Return an endpoint-inclusive, approximately equally spaced checkpoint ladder."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if points < 2:
        raise ValueError("points must include at least both endpoints")
    steps = {round(index * total_steps / (points - 1)) for index in range(points)}
    if len(steps) != points:
        raise ValueError("total_steps is too small for the requested number of points")
    return sorted(steps)


def resolve_regmix() -> ResolvedRegMix:
    """Resolve and validate the sealed RegMix train split."""

    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    read = dataset_paths(DATASET_ID, DATASET_VERSION, s3=Boto3S3.default())
    paths = tuple(str(path) for path in read.paths)
    if not paths:
        raise FinalValidationConfigError("RegMix resolved no trainable paths")
    if int(read.header_bytes or 0) != 0:
        raise FinalValidationConfigError("RegMix shards must be headerless")
    if read.byte_order not in (None, sys.byteorder):
        raise FinalValidationConfigError(
            f"RegMix byte order {read.byte_order!r} does not match host {sys.byteorder!r}"
        )
    if read.dtype is None:
        raise FinalValidationConfigError("RegMix declares no fixed-width dtype")
    return ResolvedRegMix(paths=paths, dtype=NumpyDatasetDType(read.dtype))


def build_experiment_config(
    vector: WinningVector,
    corpus: ResolvedRegMix,
    *,
    save_folder: str,
    length_tokens: int | None = None,
    work_dir: str = "/tmp/olmo-core/final-validation",
    environ: Mapping[str, str] = os.environ,
) -> ExperimentConfig:
    """Build stock OLMo2-370M, changing only the nine probe-optimized fields."""

    hps = vector.hps
    global_batch = vector.global_batch_tokens
    requested_tokens = vector.train_tokens if length_tokens is None else int(length_tokens)
    if requested_tokens <= 0 or requested_tokens % global_batch:
        raise FinalValidationConfigError(
            f"length_tokens must be a positive multiple of winner batch {global_batch}"
        )
    total_steps = requested_tokens // global_batch
    train_tokens = total_steps * global_batch
    rank_microbatch = global_batch // GPU_RANKS
    ladder = validation_steps(total_steps)
    tokenizer = TokenizerConfig.dolma2()
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}

    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        dtype=corpus.dtype,
        work_dir=work_dir,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=hps["lr"],
            betas=(0.9, 1.0 - hps["beta2_gap"]),
            eps=hps["eps"],
            weight_decay=hps["weight_decay"],
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=WSD(
            units=SchedulerUnits.tokens,
            warmup_fraction=hps["warmup_fraction"],
            decay_fraction=hps["decay_fraction"],
            decay_min_lr=hps["terminal_lr_ratio"] * hps["lr"],
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
        max_grad_norm=hps["max_grad_norm"],
    )
    run_name = environ.get("WANDB_NAME") or environ.get("EDULLM_RUN_ID") or vector.name
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
                group=environ.get("WANDB_RUN_GROUP", "hpo-final-validation-370m-10b"),
                enabled=bool(project),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            FinalValidationEvalCallback(
                vector_name=vector.name,
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
    config = ExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=global_batch,
            seed=SEED,
            num_workers=4,
        ),
        train_module=train_module,
        trainer=trainer,
        init_seed=SEED,
    )
    # These are asserted here because an accidental change would still yield a valid,
    # expensive training configuration.
    if config.trainer.max_duration != Duration.steps(total_steps) or train_tokens > requested_tokens:
        raise AssertionError("final-validation duration drifted")
    return config


def scientific_identity(vector: WinningVector, config: ExperimentConfig) -> dict[str, Any]:
    """Return the immutable scientific identity saved with the run."""

    total_steps = int(config.trainer.max_duration.value)
    return {
        "schema_version": 1,
        "model": "olmo2_370M",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "seed": SEED,
        "budget_tokens_requested": PRODUCTION_BUDGET_TOKENS,
        "train_tokens": total_steps * vector.global_batch_tokens,
        "total_steps": total_steps,
        "checkpoint_steps": validation_steps(total_steps),
        "probe_arm": vector.probe_arm,
        "probe_run_id": vector.probe_run_id,
        "probe_trial_id": vector.probe_trial_id,
        "optimized_hps": dict(vector.hps),
    }


def run_training(config: ExperimentConfig, identity: Mapping[str, Any]) -> None:
    """Build and run standard OLMo-core components, resuming from the save folder."""

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
    """Validate platform-controlled data and output identity."""

    dataset = environ.get("EDULLM_DATASET_ID", "")
    version = environ.get("EDULLM_DATASET_VERSION", "")
    tokenizer = environ.get("EDULLM_DATASET_TOKENIZER", "")
    if (dataset, version, tokenizer) != (DATASET_ID, DATASET_VERSION, TOKENIZER_ID):
        raise FinalValidationConfigError(
            "platform dataset must be "
            f"{DATASET_ID}/{DATASET_VERSION} with {TOKENIZER_ID}, got "
            f"{dataset}/{version} with {tokenizer}"
        )
    checkpoint_dir = environ.get("EDULLM_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        raise FinalValidationConfigError("EDULLM_CHECKPOINT_DIR is required")
    return checkpoint_dir, environ.get("EDULLM_RUN_ID", "hpo-final-validation")


def torchrun_command(
    vector_name: str,
    length_tokens: int | None,
    global_batch_tokens: int | None = None,
) -> list[str]:
    """Build the fixed eight-rank worker command."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={GPU_RANKS}",
        str(Path(__file__).resolve()),
        "--train-worker",
        "--vector",
        vector_name,
    ]
    if length_tokens is not None:
        command.extend(["--length-tokens", str(length_tokens)])
    if global_batch_tokens is not None:
        command.extend(["--global-batch-tokens", str(global_batch_tokens)])
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--vector", required=True)
    result.add_argument(
        "--length-tokens",
        type=int,
        help="smoke-only token budget; production omits this for the full 10B budget",
    )
    result.add_argument(
        "--global-batch-tokens",
        type=int,
        help="explicit whole-sequence global batch override",
    )
    result.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], ResolvedRegMix] = resolve_regmix,
) -> int:
    args = parser().parse_args(argv)
    try:
        vectors = load_vectors()
        if args.vector not in vectors:
            raise FinalValidationConfigError(
                f"unknown vector {args.vector!r}; choose one of {sorted(vectors)}"
            )
        vector = vectors[args.vector]
        if args.global_batch_tokens is not None:
            vector = with_global_batch_tokens(vector, args.global_batch_tokens)
        checkpoint_dir, run_id = platform_values(os.environ)
        if not args.train_worker:
            os.execv(
                sys.executable,
                torchrun_command(vector.name, args.length_tokens, args.global_batch_tokens),
            )
        if int(os.environ.get("WORLD_SIZE", "0")) != GPU_RANKS:
            raise FinalValidationConfigError(f"worker requires WORLD_SIZE={GPU_RANKS}")
        os.environ["WANDB_NAME"] = f"{run_id}-{vector.name}"
        corpus = resolver()
        config = build_experiment_config(
            vector,
            corpus,
            save_folder=checkpoint_dir,
            length_tokens=args.length_tokens,
            environ=os.environ,
        )
        run_training(config, scientific_identity(vector, config))
    except FinalValidationConfigError as exc:
        print(f"[final-validation] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
