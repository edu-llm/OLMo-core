"""Shared, pipeline-safe configuration and runtime helpers for the 400M MoE arms.

Constructing a config is deliberately local-only. The placeholder data path is metadata for
config inspection and must never be opened; only the explicit ``train`` command resolves the
sealed corpus and builds training objects.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.io import clear_directory, list_directory, normalize_path
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

# Sealed experiment identity. The train path requires the platform environment to repeat these
# exact values; accepting "latest" or another tokenizer would make the three arms incomparable.
SEALED_DATASET_ID = "pretrain/regmix-10b"
SEALED_DATASET_VERSION = "v1"
SEALED_DATASET_TOKENIZER = "tokenizer/dolma2-bpe"

TOKENIZER_CONFIG = TokenizerConfig.dolma2()
PADDED_VOCAB_SIZE = TOKENIZER_CONFIG.padded_vocab_size()

TARGET_TOKENS = 10_000_000_000
SEQUENCE_LENGTH = 2_048
WORLD_SIZE = 8
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH
MAX_STEPS = math.ceil(TARGET_TOKENS / GLOBAL_BATCH_SIZE)

BASE_LEARNING_RATE = 4e-4
WARMUP_STEPS = 1_000
SAVE_INTERVAL = 1_000
DATA_SEED = 34521
INIT_SEED = 12536
DATASET_WORK_DIR = "/tmp/dataset-cache"

# This path exists only so a serializable NumpyFSLDatasetConfig can be inspected locally. It is
# intentionally not created and config-only mode never calls ``dataset.build()``.
LOCAL_DATASET_PLACEHOLDER = "__CONFIG_ONLY_DO_NOT_READ__/regmix-10b-v1.u32le.bin"
LOCAL_SAVE_FOLDER = "/tmp/engram-experiment-checkpoints"

if PADDED_VOCAB_SIZE != 100_352:
    raise RuntimeError(f"dolma2 padded vocabulary changed: {PADDED_VOCAB_SIZE}")
if GLOBAL_BATCH_SIZE % (WORLD_SIZE * RANK_MICROBATCH_SIZE) != 0:
    raise RuntimeError("global token batch must divide evenly over eight rank microbatches")


class CorpusContractError(ValueError):
    """Raised when the platform corpus identity or manifest violates the sealed contract."""


@dataclass(frozen=True)
class Corpus:
    """A validated set of token shards suitable for OLMo-core's native memmap reader."""

    dataset_id: str
    version: str
    tokenizer_id: str
    paths: List[str]
    dtype: NumpyDatasetDType


@dataclass
class ExperimentConfig(Config):
    """All runtime configuration and resolved corpus identity saved with checkpoints."""

    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    dataset_id: str
    dataset_version: str
    dataset_tokenizer: str
    dataset_paths: List[str]
    init_seed: int = INIT_SEED


def corpus_from_manifest(read: Any) -> Corpus:
    """Validate the sealed manifest without inferring any binary representation fields."""

    paths = getattr(read, "paths", None)
    if not paths:
        raise CorpusContractError(
            f"{SEALED_DATASET_ID}/{SEALED_DATASET_VERSION} resolved to no trainable shards"
        )

    declared_dtype = getattr(read, "dtype", None)
    try:
        dtype = NumpyDatasetDType(declared_dtype)
    except (TypeError, ValueError):
        raise CorpusContractError(
            f"manifest dtype must be explicitly uint32, got {declared_dtype!r}"
        ) from None
    if dtype is not NumpyDatasetDType.uint32:
        raise CorpusContractError(f"manifest dtype must be uint32, got {declared_dtype!r}")

    byte_order = getattr(read, "byte_order", None)
    if byte_order != "little":
        raise CorpusContractError(
            f"manifest byte_order must be explicitly little, got {byte_order!r}"
        )
    if sys.byteorder != "little":
        raise CorpusContractError(
            "the sealed corpus is little-endian but this host's native byte order is not"
        )

    header_bytes = getattr(read, "header_bytes", None)
    if header_bytes != 0:
        raise CorpusContractError(
            f"manifest header_bytes must be explicitly zero, got {header_bytes!r}"
        )

    return Corpus(
        dataset_id=SEALED_DATASET_ID,
        version=SEALED_DATASET_VERSION,
        tokenizer_id=SEALED_DATASET_TOKENIZER,
        paths=[str(path) for path in paths],
        dtype=dtype,
    )


def _registry_reader(dataset_id: str, version: str) -> Any:
    """Read one sealed manifest through the reader's own storage adapter.

    Imports stay inside this function so importing arm modules and constructing local configs
    does not import ``edullm_data`` or initialize a storage client.
    """

    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    return dataset_paths(dataset_id, version, s3=Boto3S3.default())


def resolve_corpus_from_environment(
    *,
    environ: Optional[Mapping[str, str]] = None,
    registry_reader: Optional[Callable[[str, str], Any]] = None,
) -> Corpus:
    """Resolve the exact sealed corpus named by the platform environment."""

    env = os.environ if environ is None else environ
    expected = {
        "EDULLM_DATASET_ID": SEALED_DATASET_ID,
        "EDULLM_DATASET_VERSION": SEALED_DATASET_VERSION,
        "EDULLM_DATASET_TOKENIZER": SEALED_DATASET_TOKENIZER,
    }
    for name, expected_value in expected.items():
        actual = env.get(name, "")
        if actual != expected_value:
            raise CorpusContractError(
                f"{name} must be {expected_value!r} for this experiment, got {actual!r}"
            )

    read = (registry_reader or _registry_reader)(
        expected["EDULLM_DATASET_ID"], expected["EDULLM_DATASET_VERSION"]
    )
    return corpus_from_manifest(read)


def memory_optim_group_override(
    learning_rate: float = BASE_LEARNING_RATE,
) -> OptimGroupOverride:
    """Return the shared five-times-LR, zero-decay memory parameter group."""

    return OptimGroupOverride(
        params=["blocks.*.memory.*"],
        opts={"lr": learning_rate * 5, "weight_decay": 0.0},
    )


def build_train_module_config(
    *,
    param_dtype: DType = DType.bfloat16,
    learning_rate: float = BASE_LEARNING_RATE,
    with_memory_optimizer: bool = False,
) -> TransformerTrainModuleConfig:
    """Build the optimizer and one-node FSDP2 train-module configuration."""

    group_overrides = [OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})]
    if with_memory_optimizer:
        group_overrides.append(memory_optim_group_override(learning_rate))

    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=group_overrides,
            fused=True,
        ),
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=param_dtype,
            reduce_dtype=DType.float32,
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_ops
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )


def build_trainer_config(
    *,
    save_folder: str,
    run_name: str,
) -> TrainerConfig:
    """Build resumable checkpointing and optional platform W&B logging."""

    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            max_duration=Duration.steps(MAX_STEPS),
            metrics_collect_interval=5,
            cancel_check_interval=5,
            no_evals=True,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                max_checkpoints=None,
                save_async=True,
            ),
        )
    )

    project = os.environ.get("EDULLM_WANDB_PROJECT")
    if project:
        trainer = trainer.with_callback(
            "wandb",
            WandBCallback(
                name=os.environ.get("EDULLM_RUN_ID", run_name),
                project=project,
                enabled=True,
                cancel_check_interval=10,
            ),
        )

    return trainer.with_callback("config_saver", ConfigSaverCallback())


def build_experiment_config(
    model: TransformerConfig,
    *,
    corpus: Optional[Corpus] = None,
    save_folder: Optional[str] = None,
    run_name: str = "local",
    param_dtype: DType = DType.bfloat16,
    learning_rate: float = BASE_LEARNING_RATE,
    with_memory_optimizer: bool = False,
) -> ExperimentConfig:
    """Build a serializable config without resolving data unless a corpus is supplied."""

    if corpus is None:
        corpus = Corpus(
            dataset_id=SEALED_DATASET_ID,
            version=SEALED_DATASET_VERSION,
            tokenizer_id=SEALED_DATASET_TOKENIZER,
            paths=[LOCAL_DATASET_PLACEHOLDER],
            dtype=NumpyDatasetDType.uint32,
        )

    dataset_paths = list(corpus.paths)
    dataset = NumpyFSLDatasetConfig(
        paths=list(dataset_paths),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        dtype=NumpyDatasetDType.uint32,
        work_dir=DATASET_WORK_DIR,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=DATA_SEED,
        num_workers=4,
    )

    return ExperimentConfig(
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        trainer=build_trainer_config(
            save_folder=save_folder or LOCAL_SAVE_FOLDER,
            run_name=run_name,
        ),
        train_module=build_train_module_config(
            param_dtype=param_dtype,
            learning_rate=learning_rate,
            with_memory_optimizer=with_memory_optimizer,
        ),
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        dataset_tokenizer=corpus.tokenizer_id,
        dataset_paths=dataset_paths,
    )


def parameter_counts(model: TransformerConfig) -> tuple[int, int]:
    """Return and validate total and active parameter accounting."""

    total = model.num_params
    active = model.num_active_params
    if not isinstance(total, int) or not isinstance(active, int):
        raise TypeError("model parameter counts must be integers")
    if total <= 0 or active <= 0 or active > total:
        raise ValueError(f"invalid parameter counts: total={total}, active={active}")
    return total, active


def print_parameter_counts(config: ExperimentConfig) -> tuple[int, int]:
    """Verify and print stable total/active accounting for config-only inspection."""

    total, active = parameter_counts(config.model)
    print(
        json.dumps(
            {
                "active_parameters": active,
                "max_steps": MAX_STEPS,
                "target_tokens": TARGET_TOKENS,
                "total_parameters": total,
            },
            sort_keys=True,
        )
    )
    return total, active


STEP_DIRECTORY = re.compile(r"^step(\d+)$")


def torn_step_directories(save_folder: str) -> List[str]:
    """Find only step directories the canonical checkpoint loader refuses to load."""

    try:
        children = list(list_directory(save_folder, include_files=False))
    except FileNotFoundError:
        return []
    return sorted(
        path
        for path in children
        if STEP_DIRECTORY.match(os.path.basename(normalize_path(path))) is not None
        and not Checkpointer.dir_is_checkpoint(path)
    )


def remove_torn_checkpoints(save_folder: str) -> List[str]:
    """Clear incomplete step directories on rank zero, preserving every valid checkpoint."""

    removed: List[str] = []
    if get_rank() == 0:
        for path in torn_step_directories(save_folder):
            log.warning("clearing incomplete checkpoint directory %s before resume", path)
            clear_directory(path)
            removed.append(path)
    barrier()
    return removed


def fit_trainer(config: ExperimentConfig, trainer: Any) -> None:
    """Attach the saved config, narrowly repair retries, resume, and fit."""

    config_saver = cast(Any, trainer.callbacks["config_saver"])
    config_saver.config = config.as_config_dict()
    remove_torn_checkpoints(trainer.save_folder)
    trainer.maybe_load_checkpoint()
    trainer.fit()


def train(config: ExperimentConfig) -> None:
    """Build training objects and run the trainer; call only from explicit train dispatch."""

    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    fit_trainer(config, trainer)


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the arm CLI parser with a single mutating command: ``train``."""

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Build or explicitly train one sealed-corpus Engram experiment arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=("train",))
    parser.add_argument(
        "run_name",
        nargs="?",
        default=os.environ.get("EDULLM_RUN_ID", "local"),
    )
    parser.add_argument(
        "--param-dtype",
        choices=(DType.bfloat16.value, DType.float16.value, DType.float32.value),
        default=DType.bfloat16.value,
    )
    parser.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR"),
        help="Checkpoint prefix; defaults to EDULLM_CHECKPOINT_DIR.",
    )
    return parser


def parse_cli_args(
    argv: Optional[Sequence[str]] = None,
    *,
    prog: Optional[str] = None,
) -> tuple[argparse.Namespace, List[str]]:
    """Parse shared flags while retaining Config dot-list overrides."""

    opts, overrides = build_parser(prog).parse_known_args(argv)
    return opts, list(overrides)


def dispatch(
    model_builder: Callable[[], TransformerConfig],
    *,
    with_memory_optimizer: bool = False,
    argv: Optional[Sequence[str]] = None,
    prog: Optional[str] = None,
) -> ExperimentConfig:
    """Build locally by default, or resolve and train only for the exact ``train`` command."""

    opts, overrides = parse_cli_args(argv, prog=prog)
    corpus: Optional[Corpus] = None
    if opts.command == "train":
        corpus = resolve_corpus_from_environment()
        if not opts.save_folder:
            raise CorpusContractError("training requires --save-folder or EDULLM_CHECKPOINT_DIR")

    config = build_experiment_config(
        model_builder(),
        corpus=corpus,
        save_folder=opts.save_folder,
        run_name=opts.run_name,
        param_dtype=DType(opts.param_dtype),
        with_memory_optimizer=with_memory_optimizer,
    )
    if overrides:
        config = config.merge(overrides)
    print_parameter_counts(config)

    if opts.command == "train":
        prepare_training_environment()
        try:
            train(config)
        finally:
            teardown_training_environment()

    return config
