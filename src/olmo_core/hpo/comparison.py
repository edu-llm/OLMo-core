"""Matched OLMo2-190M configuration for default-recipe versus HPO smoke runs."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from ..config import Config, DType
from ..data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from ..distributed.parallel import DataParallelType
from ..float8 import Float8Config
from ..nn.transformer import TransformerConfig, TransformerDataParallelWrappingStrategy
from ..optim import AdamWConfig, CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from ..train import Duration, TrainerConfig
from ..train.callbacks import CheckpointerCallback, LMEvaluatorCallbackConfig
from ..train.train_module import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
)

__all__ = [
    "COMPARISON_DATASET_ID",
    "COMPARISON_DATASET_REFERENCE",
    "COMPARISON_HELDOUT_LABEL",
    "COMPARISON_HELDOUT_METRIC",
    "DEFAULT_RECIPE_HPS",
    "ComparisonDataset",
    "ComparisonExperimentConfig",
    "comparison_dataset_from_read",
    "comparison_heldout_label",
    "comparison_heldout_metric",
    "build_comparison_experiment",
    "build_olmoe_hpo_experiment",
    "build_umup_hpo_experiment",
    "smoke_final_evaluator",
]

COMPARISON_DATASET_ID = "pretrain/regmix-10b"
COMPARISON_DATASET_REFERENCE = "regmix-10b-v1"


def comparison_heldout_label(dataset_id: str) -> str:
    """Return the LM-evaluator label for a sealed pretrain corpus."""

    name = dataset_id.rsplit("/", 1)[-1]
    if not name:
        raise ValueError("dataset_id must be non-empty")
    return f"{name}-val"


def comparison_heldout_metric(dataset_id: str) -> str:
    """Return the held-out CE metric key for a sealed pretrain corpus."""

    return f"eval/lm/{comparison_heldout_label(dataset_id)}/CE loss"


COMPARISON_HELDOUT_LABEL = comparison_heldout_label(COMPARISON_DATASET_ID)
COMPARISON_HELDOUT_METRIC = comparison_heldout_metric(COMPARISON_DATASET_ID)

DEFAULT_RECIPE_HPS = {
    "lr": 1e-3,
    "weight_decay": 0.1,
    "beta2_gap": 0.05,
    "eps": 1e-8,
    "warmup_fraction": 0.05,
    "decay_fraction": 0.2,
    "terminal_lr_ratio": 0.1,
    "global_batch_mult": 1.0,
    "max_grad_norm": 1.0,
}


@dataclass(frozen=True)
class ComparisonDataset:
    dataset_id: str
    version: str
    tokenizer_id: str
    train_paths: tuple[str, ...]
    val_paths: tuple[str, ...]
    dtype: NumpyDatasetDType


@dataclass
class ComparisonExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    dataset_id: str
    dataset_version: str
    init_seed: int
    umup_backend: str | None = None
    umup_parity_validated: bool = False
    umup_metadata: dict[str, Any] | None = None


def comparison_dataset_from_read(
    read: Any,
    *,
    dataset_id: str,
    version: str,
    tokenizer_id: str,
) -> ComparisonDataset:
    if tokenizer_id != "tokenizer/dolma2-bpe":
        raise ValueError("the comparison currently supports only tokenizer/dolma2-bpe")
    train_paths = tuple(read.paths)
    val_paths = tuple(read.val or ())
    if not train_paths:
        raise ValueError("comparison dataset has no trainable split")
    if not val_paths:
        raise ValueError("comparison dataset has no held-out split")
    if set(train_paths) & set(val_paths):
        raise ValueError("train and held-out paths overlap")
    if int(read.header_bytes) != 0:
        raise ValueError("comparison dataset has a nonzero header")
    byte_order = read.byte_order
    if byte_order is not None and byte_order != sys.byteorder:
        raise ValueError(
            f"comparison dataset byte order {byte_order!r} does not match host {sys.byteorder!r}"
        )
    if read.dtype is None:
        raise ValueError("comparison dataset declares no fixed-width dtype")
    return ComparisonDataset(
        dataset_id=dataset_id,
        version=version,
        tokenizer_id=tokenizer_id,
        train_paths=train_paths,
        val_paths=val_paths,
        dtype=NumpyDatasetDType(read.dtype),
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"the eduLLM platform did not set {name}")
    return value


def build_comparison_experiment(
    *,
    sequence_length: int = 2048,
    global_batch_size: int = 32768,
    rank_microbatch_size: int = 4096,
    data_seed: int = 210007,
    init_seed: int = 110007,
    eval_steps: int = 2,
    work_dir: str = "/tmp/hpo-comparison-data",
    dataset_group: str | None = None,
    data_bucket: str | None = None,
) -> ComparisonExperimentConfig:
    """Build the matched model/data/eval contract used by both comparison arms."""
    if sequence_length <= 0 or global_batch_size <= 0:
        raise ValueError("sequence length and global batch size must be positive")
    if global_batch_size % sequence_length or rank_microbatch_size % sequence_length:
        raise ValueError("global and rank microbatch sizes must contain whole sequences")
    if rank_microbatch_size <= 0 or global_batch_size % rank_microbatch_size:
        raise ValueError("global batch size must be divisible by rank microbatch size")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be positive")

    dataset_id = _required_env("EDULLM_DATASET_ID")
    version = _required_env("EDULLM_DATASET_VERSION")
    tokenizer_id = _required_env("EDULLM_DATASET_TOKENIZER")
    checkpoint_root = _required_env("EDULLM_CHECKPOINT_DIR")

    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    read_kwargs: dict[str, Any] = {"s3": Boto3S3.default()}
    if dataset_group is not None:
        read_kwargs["group"] = dataset_group
    if data_bucket is not None:
        read_kwargs["data_bucket"] = data_bucket
    read = dataset_paths(dataset_id, version, **read_kwargs)
    corpus = comparison_dataset_from_read(
        read,
        dataset_id=dataset_id,
        version=version,
        tokenizer_id=tokenizer_id,
    )
    tokenizer = TokenizerConfig.dolma2()
    model = TransformerConfig.olmo2_190M(vocab_size=tokenizer.padded_vocab_size())
    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.train_paths),
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        dtype=corpus.dtype,
        work_dir=work_dir,
    )
    eval_dataset = NumpyPaddedFSLDatasetConfig(
        paths=list(corpus.val_paths),
        metadata=[{"label": comparison_heldout_label(dataset_id)} for _ in corpus.val_paths],
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        dtype=corpus.dtype,
        work_dir=work_dir,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=data_seed,
        num_workers=4,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=AdamWConfig(
            lr=DEFAULT_RECIPE_HPS["lr"],
            weight_decay=DEFAULT_RECIPE_HPS["weight_decay"],
            betas=(0.9, 1.0 - DEFAULT_RECIPE_HPS["beta2_gap"]),
            eps=DEFAULT_RECIPE_HPS["eps"],
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts={"weight_decay": 0.0},
                )
            ],
        ),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=DEFAULT_RECIPE_HPS["max_grad_norm"],
        scheduler=CosWithWarmup(warmup=1),
    )
    trainer = (
        TrainerConfig(
            save_folder=checkpoint_root,
            save_overwrite=False,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.tokens(global_batch_size),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1,
                ephemeral_save_interval=None,
                max_checkpoints=None,
                save_async=False,
            ),
        )
        .with_callback(
            "search_validation",
            LMEvaluatorCallbackConfig(
                eval_dataset=eval_dataset,
                eval_interval=None,
                eval_on_finish=True,
                eval_duration=Duration.steps(eval_steps),
                log_interval=1,
                deterministic=True,
            ),
        )
    )
    return ComparisonExperimentConfig(
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        trainer=trainer,
        train_module=train_module,
        dataset_id=dataset_id,
        dataset_version=version,
        init_seed=init_seed,
    )


def build_olmoe_hpo_experiment(
    *,
    sequence_length: int = 2048,
    global_batch_size: int = 262_144,
    rank_microbatch_size: int = 8_192,
    data_seed: int = 210007,
    init_seed: int = 110007,
    eval_steps: int = 2,
    work_dir: str = "/tmp/hpo-comparison-data",
    dataset_group: str | None = None,
    data_bucket: str | None = None,
) -> ComparisonExperimentConfig:
    """Build the stock OLMoE-1B-7B HPO experiment with its fixed eight-rank batch contract."""

    fixed_batch_contract = {
        "sequence_length": 2_048,
        "global_batch_size": 262_144,
        "rank_microbatch_size": 8_192,
    }
    requested_batch_contract = {
        "sequence_length": sequence_length,
        "global_batch_size": global_batch_size,
        "rank_microbatch_size": rank_microbatch_size,
    }
    if requested_batch_contract != fixed_batch_contract:
        raise ValueError(
            "OLMoE HPO uses the fixed batch contract "
            f"{fixed_batch_contract}, got {requested_batch_contract}"
        )

    config = build_comparison_experiment(
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        rank_microbatch_size=rank_microbatch_size,
        data_seed=data_seed,
        init_seed=init_seed,
        eval_steps=eval_steps,
        work_dir=work_dir,
        dataset_group=dataset_group,
        data_bucket=data_bucket,
    )
    config.model = TransformerConfig.olmoe_1B_7B(vocab_size=config.model.vocab_size)
    config.train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"],
                    opts={"weight_decay": 0.0},
                )
            ],
        ),
        scheduler=CosWithWarmup(warmup=24, alpha_f=0.1),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ep_config=TransformerExpertParallelConfig(degree=-1),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )
    return config


def build_umup_hpo_experiment(**kwargs: Any) -> ComparisonExperimentConfig:
    """Build the same-depth u-μP proxy requested by Arms 1 and 2."""

    from .umup import (
        UMUP_BACKEND,
        UMuPAdamWConfig,
        build_same_depth_umup_proxy,
        require_official_umup_forward,
        validate_umup_parity,
    )

    require_official_umup_forward()
    config = build_comparison_experiment(**kwargs)
    proxy, metadata = build_same_depth_umup_proxy(config.model.vocab_size)
    validate_umup_parity(proxy, metadata)
    config.model = proxy
    base_optim = config.train_module.optim
    if not isinstance(base_optim, AdamWConfig):
        raise TypeError("the shared u-μP proxy requires AdamW")
    config.train_module.optim = UMuPAdamWConfig(
        group_overrides=base_optim.group_overrides,
        compile=base_optim.compile,
        lr=base_optim.lr,
        betas=base_optim.betas,
        eps=base_optim.eps,
        weight_decay=base_optim.weight_decay,
        foreach=base_optim.foreach,
        fused=base_optim.fused,
    )
    config.umup_backend = UMUP_BACKEND
    config.umup_parity_validated = True
    config.umup_metadata = metadata.as_dict()
    return config


def smoke_final_evaluator(**kwargs: Any) -> dict[str, Any]:
    """Record the search metric for a functional smoke; this is not an untouched eval."""
    result = {
        "kind": "functional-smoke-search-validation",
        **kwargs,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result
