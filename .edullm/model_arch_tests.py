"""Build the four frozen full-architecture arms for the Mamba comparison."""

import argparse
import contextlib
import copy
import enum
import logging
import math
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cache

import rich
import torch

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
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    GatedDeltaNetConfig,
)
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.flash_pd_native import (
    NativeFlashPDMamba3SISOMixerConfig,
    NativeFlashPDMixerConfig,
    NativePDBackend,
    NativePDMode,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.mamba3 import Mamba3MixerConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.nn.xlstm import SLSTMMixerConfig, XLSTMMixerConfig
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
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

ARMS = ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd")
ARM_ORDER = ARMS
DATA_SEEDS = (210007, 220014, 230021, 240028, 250035)
INIT_SEEDS_BY_ARM = {
    "mamba-b3": {
        210007: 110007,
        220014: 120014,
        230021: 130021,
        240028: 140028,
        250035: 150035,
    },
    "xlstm": {
        210007: 113008,
        220014: 123015,
        230021: 133022,
        240028: 143029,
        250035: 153036,
    },
    "mamba3-siso-pd": {
        210007: 116009,
        220014: 126016,
        230021: 136023,
        240028: 146030,
        250035: 156037,
    },
    "native-pd": {
        210007: 119010,
        220014: 129017,
        230021: 139024,
        240028: 149031,
        250035: 159038,
    },
}

D_MODEL = 1024
VOCAB_SIZE = 100352
N_LAYERS = 16
SEQUENCE_LENGTH = 4096
BASE_FFN_WIDTH = 4608
ATTENTION_LAYERS = (3, 7, 11, 15)
RECURRENT_LAYERS = tuple(index for index in range(N_LAYERS) if index not in ATTENTION_LAYERS)
XLSTM_SLSTM_LAYERS = (6, 14)
FROZEN_STEPS = 3721
FROZEN_GLOBAL_BATCH_SIZE = 524288
FROZEN_WARMUP_STEPS = 372
FROZEN_SAVE_INTERVAL = 1861
PARAMETER_TARGET = 390_135_552
PARAMETER_TOLERANCE = 195_068
EXACT_PARAMETER_COUNTS = {
    "mamba-b3": 390_148_736,
    "xlstm": 390_143_056,
    "mamba3-siso-pd": 390_169_664,
    "native-pd": 390_142_976,
}


class Stage(enum.IntEnum):
    """Machine-readable startup failure stages."""

    ENVIRONMENT = 64
    READER = 67
    MANIFEST = 68
    TOKENIZER = 69
    CONFIG = 70
    TRAINING_ENVIRONMENT = 71
    TRAINING = 72
    PRECISION = 73


class Refusal(SystemExit):
    """A refusal with a stable process exit status."""

    def __init__(self, stage: Stage, explanation: str):
        super().__init__(explanation)
        self.stage = stage
        self.explanation = explanation


@contextlib.contextmanager
def during(stage: Stage) -> Iterator[None]:
    try:
        yield
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(stage, f"{type(exc).__name__}: {exc}") from exc


@dataclass
class Corpus:
    """A manifest-resolved, fixed-width training corpus."""

    dataset_id: str
    version: str
    paths: list[str]
    dtype: NumpyDatasetDType
    tokenizer: TokenizerConfig
    rows: int | None


@dataclass
class ExperimentConfig(Config):
    """The complete saved training configuration."""

    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    dataset_id: str
    dataset_version: str
    arm: str
    data_seed: int
    init_seed: int


TOKENIZERS = {"tokenizer/dolma2-bpe": TokenizerConfig.dolma2}


def corpus_from_manifest(read, *, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    """Validate manifest facts before passing them to a zero-offset memmap."""
    if not read.paths or read.dtype is None:
        raise Refusal(Stage.MANIFEST, "the corpus has no trainable paths or fixed-width dtype")
    if read.header_bytes:
        raise Refusal(Stage.MANIFEST, "OLMo-core cannot memmap a corpus with header bytes")
    if read.byte_order is not None and read.byte_order != sys.byteorder:
        raise Refusal(
            Stage.MANIFEST, f"corpus byte order {read.byte_order} != host {sys.byteorder}"
        )
    try:
        tokenizer = TOKENIZERS[tokenizer_id]()
    except KeyError:
        raise Refusal(Stage.TOKENIZER, f"no image config for tokenizer {tokenizer_id}") from None
    if tokenizer.padded_vocab_size() != VOCAB_SIZE:
        raise Refusal(
            Stage.TOKENIZER,
            f"manifest tokenizer pads to {tokenizer.padded_vocab_size()}, expected {VOCAB_SIZE}",
        )
    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths=list(read.paths),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=tokenizer,
        rows=read.rows,
    )


def resolve_corpus(*, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    """Resolve only through the pinned edullm-data reader contract."""
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    if version in ("", "latest"):
        version = resolve_latest(dataset_id, s3=s3)
        if version is None:
            raise Refusal(Stage.READER, f"no published version of {dataset_id}")
    read = dataset_paths(dataset_id, version, s3=s3)
    return corpus_from_manifest(
        read,
        dataset_id=dataset_id,
        version=version,
        tokenizer_id=tokenizer_id,
    )


def _layer_norm() -> LayerNormConfig:
    return LayerNormConfig(
        name=LayerNormType.rms,
        eps=1e-6,
        bias=False,
        dtype=DType.bfloat16,
    )


def _feed_forward(width: int) -> FeedForwardConfig:
    return FeedForwardConfig(hidden_size=width, bias=False, dtype=DType.bfloat16)


def _attention_mixer() -> AttentionConfig:
    norm = _layer_norm()
    return AttentionConfig(
        name=AttentionType.default,
        n_heads=16,
        n_kv_heads=8,
        head_dim=64,
        bias=False,
        rope=RoPEConfig(name=RoPEType.default, theta=500_000),
        qk_norm=norm,
        use_head_qk_norm=False,
        backend=AttentionBackendName.flash_2,
        dtype=DType.bfloat16,
    )


def _gdn_mixer() -> GatedDeltaNetConfig:
    return GatedDeltaNetConfig(
        n_heads=16,
        n_v_heads=None,
        head_dim=32,
        expand_v=2.0,
        allow_neg_eigval=True,
        conv_size=4,
        conv_bias=False,
        norm_eps=1e-6,
        dtype=DType.bfloat16,
    )


def _mlstm_mixer() -> XLSTMMixerConfig:
    return XLSTMMixerConfig(
        n_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        conv_size=4,
        chunkwise_kernel="chunkwise--triton_xl_chunk",
        chunk_size=256,
        autocast_kernel_dtype="bfloat16",
        dtype=DType.bfloat16,
    )


def _slstm_mixer() -> SLSTMMixerConfig:
    return SLSTMMixerConfig(
        n_heads=4,
        conv_size=4,
        backend="cuda_fused",
        batch_size=2,
        kernel_dtype="float32",
        fuse_input_projections=True,
        dtype=DType.bfloat16,
    )


def _treatment_mixer(arm: str, layer_index: int):
    if arm == "mamba-b3":
        return Mamba3MixerConfig(
            n_heads=16,
            head_dim=64,
            d_state=96,
            n_groups=1,
            mimo_rank=1,
            rotation_block_size=3,
            norm_eps=1e-6,
            bc_norm=True,
            bc_bias=True,
            prefer_official_kernel=True,
            rotation_scan_impl="quaternion",
            theta_max=1 / math.sqrt(SEQUENCE_LENGTH),
            fuse_input_projections=True,
            dtype=DType.bfloat16,
        )
    if arm == "xlstm":
        return _slstm_mixer() if layer_index in XLSTM_SLSTM_LAYERS else _mlstm_mixer()
    if arm == "native-pd":
        return NativeFlashPDMixerConfig(
            n_heads=16,
            d_state=64,
            dictionary_size=16,
            chunk_size=128,
            ste_temperature=1.0,
            mode=NativePDMode.GENERAL_SCATTER,
            backend=NativePDBackend.CUDA,
            conv_kernel_size=4,
            dtype=DType.bfloat16,
        )
    if arm == "mamba3-siso-pd":
        return NativeFlashPDMamba3SISOMixerConfig(
            n_heads=16,
            d_state=64,
            dictionary_size=16,
            chunk_size=64,
            dictionary_temperature=1.0,
            router_temperature=1.0,
            mode=NativePDMode.GENERAL_SCATTER,
            backend=NativePDBackend.CUDA,
            bc_norm=True,
            norm_eps=1e-6,
            output_norm=False,
            fuse_input_projections=True,
            dtype=DType.bfloat16,
        )
    raise ValueError(f"unsupported arm: {arm}")


def _block(mixer, width: int) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=mixer,
        feed_forward=_feed_forward(width),
        layer_norm=_layer_norm(),
    )


def _model_for_widths(arm: str, widths: tuple[int, ...], init_seed: int) -> TransformerConfig:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if len(widths) != len(RECURRENT_LAYERS):
        raise ValueError(
            f"expected {len(RECURRENT_LAYERS)} recurrent FFN widths, got {len(widths)}"
        )
    blocks = {"attention": _block(_attention_mixer(), BASE_FFN_WIDTH)}
    pattern: list[str] = []
    width_by_layer = dict(zip(RECURRENT_LAYERS, widths))
    for index in range(N_LAYERS):
        if index in ATTENTION_LAYERS:
            pattern.append("attention")
        else:
            name = f"recurrent-{index}"
            blocks[name] = _block(_treatment_mixer(arm, index), width_by_layer[index])
            pattern.append(name)
    return TransformerConfig(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        block=blocks,
        block_pattern=pattern,
        lm_head=LMHeadConfig(layer_norm=_layer_norm(), bias=False, dtype=DType.bfloat16),
        dtype=DType.bfloat16,
        init_seed=init_seed,
        tie_word_embeddings=True,
    )


@cache
def solve_widths(arm: str) -> tuple[int, ...]:
    """Find the nearest /32 recurrent widths while keeping attention blocks fixed."""
    best = None
    for base_width in range(32, 8193, 32):
        for elevated_count in range(len(RECURRENT_LAYERS) + 1):
            if elevated_count and base_width == 8192:
                continue
            widths = tuple(
                base_width + (32 if position < elevated_count else 0)
                for position in range(len(RECURRENT_LAYERS))
            )
            count = _model_for_widths(arm, widths, 0).num_params
            candidate = (abs(count - PARAMETER_TARGET), widths, count)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    difference, widths, count = best
    if difference > PARAMETER_TOLERANCE:
        raise ValueError(
            f"{arm} cannot reach {PARAMETER_TARGET:,} on the /32 recurrent FFN grid; "
            f"nearest is {count:,} at widths {widths}"
        )
    return widths


def build_model_config(arm: str, init_seed: int) -> TransformerConfig:
    """Build and assert the frozen geometry and exact arm parameter count."""
    config = _model_for_widths(arm, solve_widths(arm), init_seed)
    expected = EXACT_PARAMETER_COUNTS[arm]
    if config.num_params != expected:
        raise RuntimeError(f"{arm} parameter count drifted: {config.num_params:,} != {expected:,}")
    if abs(config.num_params - PARAMETER_TARGET) > PARAMETER_TOLERANCE:
        raise RuntimeError(f"{arm} is outside the frozen parameter tolerance")
    return config


def valid_init_seeds(arm: str) -> tuple[int, ...]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    return tuple(INIT_SEEDS_BY_ARM[arm][seed] for seed in DATA_SEEDS)


def build_config(opts, overrides: list[str]) -> ExperimentConfig:
    corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    model = build_model_config(opts.arm, opts.init_seed)
    dataset = NumpyFSLDatasetConfig(
        paths=corpus.paths,
        sequence_length=opts.sequence_length,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        work_dir=opts.work_dir,
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.learning_rate,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight", "*.A_log", "*.dt_bias", "*.D"],
                    opts={"weight_decay": 0.0},
                )
            ],
        ),
        accumulate_grads_without_comm=True,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType(opts.param_dtype),
            reduce_dtype=DType.float32,
            reshard_after_forward=False,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=opts.warmup_steps),
    )
    trainer = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=None,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                project=os.environ.get("EDULLM_WANDB_PROJECT"),
                cancel_check_interval=10,
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )
    config = ExperimentConfig(
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        trainer=trainer,
        train_module=train_module,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        arm=opts.arm,
        data_seed=opts.data_seed,
        init_seed=opts.init_seed,
    )
    return config.merge(overrides)


def remove_torn_checkpoints(save_folder: str) -> None:
    """Remove only incomplete step directories that the checkpointer will not load."""
    if get_rank() == 0:
        try:
            children = list(list_directory(save_folder, include_files=False))
        except FileNotFoundError:
            children = []
        for path in children:
            name = os.path.basename(normalize_path(path))
            if (
                name.startswith("step")
                and name[4:].isdigit()
                and not Checkpointer.dir_is_checkpoint(path)
            ):
                clear_directory(path)
    barrier()


def show(config: ExperimentConfig) -> None:
    shown = copy.copy(config.dataset)
    shown.paths = [f"<{len(config.dataset.paths or [])} objects>"]
    rich.print(replace(config, dataset=shown))


def validate_precision(config: ExperimentConfig) -> None:
    """Reject BF16 on pre-Ampere NVIDIA hardware before distributed startup."""
    if config.train_module.dp_config is None:
        return
    if config.train_module.dp_config.param_dtype != DType.bfloat16 or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise Refusal(Stage.PRECISION, f"CUDA capability {major}.{minor} has no BF16 hardware")


def train(config: ExperimentConfig) -> None:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    trainer.callbacks["config_saver"].config = config.as_config_dict()
    remove_torn_checkpoints(trainer.save_folder)
    trainer.maybe_load_checkpoint()
    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen model-architecture comparison cells.")
    parser.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-seed", required=True, type=int, choices=DATA_SEEDS)
    parser.add_argument("--init-seed", required=True, type=int)
    parser.add_argument("--dataset-id", default=os.environ.get("EDULLM_DATASET_ID", ""))
    parser.add_argument("--dataset-version", default=os.environ.get("EDULLM_DATASET_VERSION", ""))
    parser.add_argument(
        "--dataset-tokenizer", default=os.environ.get("EDULLM_DATASET_TOKENIZER", "")
    )
    parser.add_argument("--save-folder", default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""))
    parser.add_argument("--output-prefix", default=os.environ.get("EDULLM_OUTPUT_PREFIX", ""))
    parser.add_argument("--work-dir", default="/tmp/dataset-cache")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--save-interval", type=int, default=FROZEN_SAVE_INTERVAL)
    parser.add_argument("--warmup-steps", type=int, default=FROZEN_WARMUP_STEPS)
    parser.add_argument("--learning-rate", type=float, default=1.4e-3)
    parser.add_argument("--global-batch-size", type=int, default=FROZEN_GLOBAL_BATCH_SIZE)
    parser.add_argument("--rank-microbatch-size", type=int, default=8192)
    parser.add_argument("--param-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv=None):
    opts, overrides = build_parser().parse_known_args(argv)
    expected = INIT_SEEDS_BY_ARM[opts.arm][opts.data_seed]
    if opts.init_seed != expected:
        raise SystemExit(
            f"init seed {opts.init_seed} is invalid for {opts.arm}/{opts.data_seed}; expected {expected}"
        )
    return opts, overrides


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = parse_args()
    missing = [
        name
        for name, value in (
            ("EDULLM_DATASET_ID", opts.dataset_id),
            ("EDULLM_DATASET_VERSION", opts.dataset_version),
            ("EDULLM_DATASET_TOKENIZER", opts.dataset_tokenizer),
            ("EDULLM_CHECKPOINT_DIR", opts.save_folder),
            ("EDULLM_OUTPUT_PREFIX", opts.output_prefix),
        )
        if not value
    ]
    if missing:
        raise Refusal(Stage.ENVIRONMENT, "the platform did not set: " + ", ".join(missing))
    with during(Stage.CONFIG):
        config = build_config(opts, overrides)
    validate_precision(config)
    if opts.dry_run:
        show(config)
        return
    with during(Stage.TRAINING_ENVIRONMENT):
        prepare_training_environment()
    try:
        with during(Stage.TRAINING):
            train(config)
    finally:
        teardown_training_environment()


def cli() -> int:
    try:
        main()
    except Refusal as refusal:
        print(refusal.explanation, file=sys.stderr)
        print(f"edullm-stage: {refusal.stage.name} exit={int(refusal.stage)}", file=sys.stderr)
        return int(refusal.stage)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
