#!/usr/bin/env python3
"""Train one MixLaw arm with OLMo-core's standard 370M training stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
    SourceMixtureList,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
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
from mixlaw_wandb_policy import MixLawWandBEvalCallback

RECIPE_PATH = Path(__file__).with_name("mixlaw_recipe.json")
EVAL_SCRIPT = Path(__file__).with_name("eval_task_loss_olmo_core.py")
DATASET_ID = "pretrain/olmo-127b"
DATASET_VERSION = "v1"
DATASET_LABEL = "source"
GPU_RANKS = 8
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 4_194_304
RANK_MICROBATCH_TOKENS = 32_768
PRODUCTION_BUDGET_TOKENS = 10_000_000_000
PRODUCTION_STEPS = PRODUCTION_BUDGET_TOKENS // GLOBAL_BATCH_TOKENS
SEED = 12_536
SAVE_INTERVAL = 125


class MixLawConfigError(RuntimeError):
    """The recipe or platform environment violates the fixed experiment contract."""


@dataclass(frozen=True)
class Arm:
    index: int
    mixture_id: int
    name: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class DomainSource:
    name: str
    paths: tuple[str, ...]
    available_tokens: int


def load_recipe(path: Path = RECIPE_PATH) -> tuple[tuple[str, ...], tuple[Arm, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = tuple(payload["domain_order"])
    arms = tuple(
        Arm(
            index=int(item["arm_index"]),
            mixture_id=int(item["mixture_id"]),
            name=str(item["name"]),
            weights=tuple(float(weight) for weight in item["weights"]),
        )
        for item in payload["arms"]
    )
    expected_arms = (
        (0, 0, "olmo-mix-1124"),
        (1, 1, "mix01"),
        (2, 25, "ML-pilot_caps"),
        (3, 27, "LGB-min1pct"),
    )
    if tuple((arm.index, arm.mixture_id, arm.name) for arm in arms) != expected_arms:
        raise MixLawConfigError("recipe arms differ from the approved four-arm matrix")
    if len(set(domains)) != 7 or any(len(arm.weights) != len(domains) for arm in arms):
        raise MixLawConfigError(
            "recipe must contain seven unique domains and seven weights per arm"
        )
    for arm in arms:
        if not math.isclose(sum(arm.weights), 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise MixLawConfigError(f"{arm.name} weights sum to {sum(arm.weights)!r}, not near 1")
        if any(weight <= 0.0 for weight in arm.weights):
            raise MixLawConfigError(f"{arm.name} contains a non-positive domain weight")
    source = payload["data_source"]
    if source != {
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "label_key": DATASET_LABEL,
    }:
        raise MixLawConfigError(f"unexpected data source: {source!r}")
    if int(payload["budget_tokens"]) != PRODUCTION_BUDGET_TOKENS:
        raise MixLawConfigError("recipe budget is not the fixed 10B-token production budget")
    return domains, arms


DOMAINS, ARMS = load_recipe()


def normalized_weights(arm: Arm) -> tuple[float, ...]:
    """Normalize source-recorded decimal weights for OLMo-core's unit-sum contract."""
    total = sum(arm.weights)
    return tuple(weight / total for weight in arm.weights)


def resolve_domain_sources() -> tuple[DomainSource, ...]:
    """Resolve each labeled domain directly from the validated edullm-data release."""
    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    sources: list[DomainSource] = []
    for domain in DOMAINS:
        resolved = dataset_paths(
            DATASET_ID,
            DATASET_VERSION,
            split="train",
            s3=s3,
            labels={DATASET_LABEL: domain},
        )
        if not resolved.paths:
            raise MixLawConfigError(f"{domain}: dataset_paths resolved no training shards")
        if resolved.dtype != "uint32":
            raise MixLawConfigError(f"{domain}: expected uint32, got {resolved.dtype!r}")
        if resolved.byte_order != "little":
            raise MixLawConfigError(
                f"{domain}: expected explicit little-endian data, got {resolved.byte_order!r}"
            )
        if int(resolved.header_bytes or 0) != 0:
            raise MixLawConfigError(
                f"{domain}: expected headerless shards, got {resolved.header_bytes} header bytes"
            )
        available_tokens = int(resolved.rows or 0)
        if available_tokens <= 0:
            raise MixLawConfigError(f"{domain}: manifest has no positive token count")
        sources.append(
            DomainSource(
                name=domain,
                paths=tuple(resolved.paths),
                available_tokens=available_tokens,
            )
        )
    return tuple(sources)


def repetition_bounds(
    sources: Sequence[DomainSource],
    *,
    steps: int = PRODUCTION_STEPS,
) -> dict[str, float]:
    """Return recipe-wide bounds, identical for every arm at the requested duration."""
    by_name = {source.name: source for source in sources}
    peak_weight = {
        domain: max(normalized_weights(arm)[index] for arm in ARMS)
        for index, domain in enumerate(DOMAINS)
    }
    bounds: dict[str, float] = {}
    for domain in DOMAINS:
        source = by_name[domain]
        required = steps * GLOBAL_BATCH_TOKENS * peak_weight[domain]
        bounds[domain] = max(1.0, math.ceil(required / source.available_tokens))
    return bounds


def steps_for_length(length_tokens: int | None) -> int:
    if length_tokens is None:
        return PRODUCTION_STEPS
    if length_tokens <= 0 or length_tokens % GLOBAL_BATCH_TOKENS:
        raise MixLawConfigError(
            f"--length-tokens must be a positive multiple of {GLOBAL_BATCH_TOKENS}"
        )
    return length_tokens // GLOBAL_BATCH_TOKENS


def build_experiment_config(
    arm_index: int,
    sources: Sequence[DomainSource],
    *,
    save_folder: str,
    length_tokens: int | None = None,
    work_dir: str = "/tmp/olmo-core/mixlaw",
    environ: Mapping[str, str] = os.environ,
) -> ExperimentConfig:
    """Build the standard OLMo2-370M config; only source target ratios vary by arm."""
    arm = ARMS[arm_index]
    weights = normalized_weights(arm)
    by_name = {source.name: source for source in sources}
    if tuple(by_name) != DOMAINS:
        raise MixLawConfigError(f"domain order must be {DOMAINS!r}, got {tuple(by_name)!r}")
    max_steps = steps_for_length(length_tokens)
    bounds = repetition_bounds(sources, steps=max_steps)
    train_tokens = max_steps * GLOBAL_BATCH_TOKENS
    rank_microbatch_tokens = int(
        environ.get("EDULLM_RANK_MICROBATCH_TOKENS", str(RANK_MICROBATCH_TOKENS))
    )
    if (
        rank_microbatch_tokens <= 0
        or rank_microbatch_tokens % SEQUENCE_LENGTH
        or GLOBAL_BATCH_TOKENS % (rank_microbatch_tokens * GPU_RANKS)
    ):
        raise MixLawConfigError(
            "EDULLM_RANK_MICROBATCH_TOKENS must be a positive sequence-length multiple "
            "that evenly divides the per-step global batch"
        )
    skip_pre_train = environ.get("WANDB_RESUME", "").lower() in {"must", "allow"}
    tokenizer = TokenizerConfig.dolma2()

    dataset = NumpyFSLDatasetConfig.from_src_mix(
        SourceMixtureDatasetConfig(
            source_list=SourceMixtureList(
                [
                    SourceMixtureConfig(
                        source_name=domain,
                        target_ratio=weights[index],
                        paths=list(by_name[domain].paths),
                        max_repetition_ratio=bounds[domain],
                    )
                    for index, domain in enumerate(DOMAINS)
                ]
            ),
            requested_tokens=train_tokens,
            global_batch_size=GLOBAL_BATCH_TOKENS,
            processes=16,
            seed=SEED,
            render_tables=True,
        ),
        tokenizer=tokenizer,
        sequence_length=SEQUENCE_LENGTH,
        dtype=NumpyDatasetDType.uint32,
        work_dir=work_dir,
        include_instance_metadata=False,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=CosWithWarmup(warmup=24, alpha_f=0.1),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            work_dir=work_dir,
            max_duration=Duration.steps(max_steps),
            metrics_collect_interval=5,
            cancel_check_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                fixed_steps=[max_steps],
                pre_train_checkpoint=not skip_pre_train,
                save_async=True,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=environ.get("WANDB_NAME"),
                project=environ.get("EDULLM_WANDB_PROJECT"),
                group=environ.get("WANDB_RUN_GROUP"),
                enabled=bool(environ.get("EDULLM_WANDB_PROJECT")),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            MixLawWandBEvalCallback(
                arm=environ.get("WANDB_NAME") or "mixlaw",
                total_steps=max_steps,
                save_folder=save_folder,
                run_name=(
                    environ.get("WANDB_NAME")
                    or environ.get("EDULLM_RUN_ID")
                    or "mixlaw"
                ),
                work_dir=environ.get(
                    "EDULLM_EVAL_WORK_DIR",
                    str(Path(work_dir) / "mixlaw-eval"),
                ),
                eval_script=EVAL_SCRIPT,
                interval=SAVE_INTERVAL,
                nproc=GPU_RANKS,
            ),
        )
    )
    return ExperimentConfig(
        model=TransformerConfig.olmo2_370M(vocab_size=tokenizer.padded_vocab_size()),
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=GLOBAL_BATCH_TOKENS,
            seed=SEED,
            num_workers=4,
        ),
        train_module=train_module,
        trainer=trainer,
        init_seed=SEED,
    )


def run_training(config: ExperimentConfig) -> None:
    """Build and run only OLMo-core standard components, resuming from save_folder."""
    prepare_training_environment(seed=config.init_seed, shared_filesystem=False)
    try:
        model = config.model.build(init_device="meta")
        train_module = config.train_module.build(model)
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(
            dataset, dp_process_group=train_module.dp_process_group
        )
        trainer = config.trainer.build(train_module, data_loader)
        config_saver = trainer.callbacks["config_saver"]
        assert isinstance(config_saver, ConfigSaverCallback)
        config_saver.config = config.as_config_dict()
        trainer.maybe_load_checkpoint()
        trainer.fit()
    finally:
        teardown_training_environment()


def platform_values(environ: Mapping[str, str]) -> tuple[str, str]:
    dataset = environ.get("EDULLM_DATASET_ID", "")
    version = environ.get("EDULLM_DATASET_VERSION", "")
    checkpoint_dir = environ.get("EDULLM_CHECKPOINT_DIR", "")
    project = environ.get("EDULLM_WANDB_PROJECT", "")
    if (dataset, version) != (DATASET_ID, DATASET_VERSION):
        raise MixLawConfigError(
            f"platform dataset must be {DATASET_ID}/{DATASET_VERSION}, got {dataset}/{version}"
        )
    if not checkpoint_dir:
        raise MixLawConfigError("EDULLM_CHECKPOINT_DIR is required")
    if not project:
        raise MixLawConfigError("EDULLM_WANDB_PROJECT is required")
    return checkpoint_dir, environ.get("EDULLM_RUN_ID", "mixlaw")


def torchrun_command(arm_index: int, length_tokens: int | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={GPU_RANKS}",
        str(Path(__file__).resolve()),
        "--train-worker",
        "--arm-index",
        str(arm_index),
    ]
    if length_tokens is not None:
        command.extend(["--length-tokens", str(length_tokens)])
    return command


def _positive_batch_multiple(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer token count") from exc
    if value <= 0 or value % GLOBAL_BATCH_TOKENS:
        raise argparse.ArgumentTypeError(f"must be a positive multiple of {GLOBAL_BATCH_TOKENS}")
    return value


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--arm-index", type=int, choices=range(len(ARMS)), required=True)
    out.add_argument(
        "--length-tokens",
        type=_positive_batch_multiple,
        help="benchmark-only token budget; production omits this for 2,384 steps",
    )
    out.add_argument("--train-worker", action="store_true", help=argparse.SUPPRESS)
    return out


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[], tuple[DomainSource, ...]] = resolve_domain_sources,
) -> int:
    args = parser().parse_args(argv)
    try:
        checkpoint_dir, run_id = platform_values(os.environ)
        arm = ARMS[args.arm_index]
        if not args.train_worker:
            os.execv(sys.executable, torchrun_command(args.arm_index, args.length_tokens))
        if int(os.environ.get("WORLD_SIZE", "0")) != GPU_RANKS:
            raise MixLawConfigError(f"worker requires WORLD_SIZE={GPU_RANKS}")
        os.environ["WANDB_NAME"] = f"{run_id}-{arm.name}"
        sources = resolver()
        config = build_experiment_config(
            args.arm_index,
            sources,
            save_folder=checkpoint_dir,
            length_tokens=args.length_tokens,
            environ=os.environ,
        )
        run_training(config)
    except MixLawConfigError as exc:
        print(f"[mixlaw] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
