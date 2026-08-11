"""Build the eight frozen comparison arms of the mixer wave."""

import argparse
import contextlib
import copy
import enum
import logging
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
    GatedDeltaNet2Config,
    KimiDeltaAttentionConfig,
    KimiDeltaHouseholderConfig,
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

# NEW ARMS ARE APPENDED, NEVER INSERTED, AND THE ORDER OF THIS TUPLE IS THE WAVE. It is
# arm-major and the fan-out index selects a cell by position, so each prefix stays runnable:
# `--fanout-size 12` reproduces the original four-arm study exactly, `15` the five-arm one, and
# `24` the whole thing. Inserting an arm anywhere earlier would renumber every cell after it and
# silently repoint any fan-out already described in a document or a submitted run.
#
# Every arm after the first is a peer treatment, not a control: the control is `mamba-b3`. GDN2
# was previously carried as a throughput diagnostic beside the wave, which is why the smoke specs
# already name it.
#
# THE LAST THREE ARE ONE FAMILY AND THEIR CONTRASTS ARE PAIRWISE. `kda` is the shipped Kimi Delta
# Attention operator and the baseline the other two are read against; `kda-hh-r2` adds a second
# Householder factor and negative eigenvalues; `kda-gconv` replaces the three plain short
# convolutions with LIV-style gated ones. Each moves one mechanism away from `kda`, so a
# difference against it is attributable; a difference between `kda-hh-r2` and `kda-gconv` is not.
ARMS = (
    "mamba-b3",
    "xlstm",
    "mamba3-siso-pd",
    "native-pd",
    "gdn",
    "kda",
    "kda-hh-r2",
    "kda-gconv",
)
RUNNABLE_ARMS = ARMS
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
    "gdn": {
        210007: 122011,
        220014: 132018,
        230021: 142025,
        240028: 152032,
        250035: 162039,
    },
    # The three KDA rows continue the schedule the first five were issued on -- arm `k` and data
    # seed `j` take `110007 + 3001*k + 10007*j` -- so no arm above had to give up an integer and
    # no integer is used twice across the forty. The last two columns of every row stay reserved
    # and unissued, exactly as they are for the arms that came before.
    "kda": {
        210007: 125012,
        220014: 135019,
        230021: 145026,
        240028: 155033,
        250035: 165040,
    },
    "kda-hh-r2": {
        210007: 128013,
        220014: 138020,
        230021: 148027,
        240028: 158034,
        250035: 168041,
    },
    "kda-gconv": {
        210007: 131014,
        220014: 141021,
        230021: 151028,
        240028: 161035,
        250035: 171042,
    },
}

# Every parameter in every arm is built in this one dtype. FSDP2 refuses to shard a block
# whose parameters were built in more than one, and the recurrent mixers keep their state
# parameters (A_log, dt_bias, D, the B/C biases) in float32 whatever the block asks for.
# bfloat16 compute comes from the data-parallel param_dtype, not from how the model is stored.
MASTER_DTYPE = DType.float32

D_MODEL = 1024
VOCAB_SIZE = 100352
N_LAYERS = 16
SEQUENCE_LENGTH = 4096
BASE_FFN_WIDTH = 4608
ATTENTION_LAYERS = (3, 7, 11, 15)
RECURRENT_LAYERS = tuple(index for index in range(N_LAYERS) if index not in ATTENTION_LAYERS)
XLSTM_SLSTM_LAYERS = (6, 14)
FROZEN_STEPS = 1144
FROZEN_GLOBAL_BATCH_SIZE = 524288
FROZEN_WARMUP_STEPS = 114
FROZEN_SAVE_INTERVAL = 572
PARAMETER_TARGET = 390_135_552
PARAMETER_TOLERANCE = 195_068
EXACT_PARAMETER_COUNTS = {
    "mamba-b3": 390_100_352,
    "xlstm": 390_143_056,
    "mamba3-siso-pd": 390_169_664,
    "native-pd": 390_142_976,
    "gdn": 390_119_360,
    "kda": 390_119_360,
    "kda-hh-r2": 390_119_360,
    "kda-gconv": 390_094_784,
}

# Which of an arm's own parameters stay out of weight decay, BEYOND the embeddings every arm
# exempts. This is a per-arm list rather than one shared list, and both halves of that are
# load-bearing.
#
# THE MIXERS' `_no_weight_decay` TAGS DO NOTHING BY THEMSELVES. `OptimConfig.build_groups`
# reads `group_overrides` and never looks at the tag, so an arm whose recurrence marks
# `A_log`/`dt_bias`/`D` still has AdamW's 0.01 applied to them unless a pattern names them.
# That pulls |A| toward 1 and moves `dt` for eleven hours, in the one set of parameters the
# comparison is about, while every printed field still reads correctly.
#
# AND A PATTERN THAT MATCHES NOTHING IS FATAL, NOT INERT. TransformerTrainModule builds the
# optimizer with `strict=True`, and `_expand_param_globs` raises OLMoConfigurationError for a
# pattern with no match. So the one shared four-pattern list this file used to carry could not
# run: `xlstm` has no such parameter at all, and `mamba-b3` has no `D`. Each arm therefore
# names exactly what it has, and `test_every_arm_exempts_exactly_its_tagged_timescale_parameters_from_weight_decay`
# rebuilds every arm and asserts this table against the tags themselves.
#
# `gdn` USED TO BE EMPTY, AND THAT WAS AN ASYMMETRY RATHER THAN A CHOICE. It has twelve `A_log`
# and twelve `dt_bias` parameters -- and no `D` -- but `GatedDeltaNet2` in
# `olmo_core.nn.attention.recurrent` did not set `_no_weight_decay` on them where Mamba-3 and
# both PD mixers did, so the arm decayed its recurrence timescales under AdamW's 0.01 while the
# other three recurrent arms did not: an optimizer difference wearing the costume of an operator
# difference. The mixer now tags both, so the row names both. `*.D` stays off this row because
# GDN-2 has no `D` and an unmatched pattern is fatal under `strict=True`.
#
# THE POLICY IS NOW UNIFORM: every arm exempts the timescales it has, and no arm exempts a name
# it does not have. `xlstm` is the empty row because its two recurrences carry no such parameter
# at all, not because it is treated differently. All three KDA classes tag `A_log` and `dt_bias`
# and none of them has a `D`, so the three KDA rows are the same two patterns as `gdn`'s -- and
# copying Mamba-3's three-pattern row onto them would not read as a smaller mistake in a diff
# while being a fatal one at optimizer-build time.
WEIGHT_DECAY_EXEMPT_PATTERNS_BY_ARM: dict[str, tuple[str, ...]] = {
    # The faithful Mamba arm now carries a learned ``D`` skip alongside ``A_log`` and ``dt_bias``,
    # so it exempts all three timescale-class parameters -- the same row as the two PD arms. The
    # token-dependent ``a_proj`` is a plain input GEMM and is deliberately NOT exempt; the decay
    # baseline still lives in the exempt ``A_log``.
    "mamba-b3": ("*.A_log", "*.dt_bias", "*.D"),
    # xLSTM's timescale parameters are its GATE BIASES, and the empty tuple that stood here
    # decayed every one of them. mLSTM packs the input and forget gate biases into `w_if.bias`
    # and initializes them to -10 and a 3..6 ramp; sLSTM's `_bias_` spans about -7..5 under
    # `powerlaw_blockdependent`. Those set the recurrence's retention horizon exactly as
    # `A_log` and `dt_bias` do for Mamba and the delta-rule arms, so decaying them pulls the
    # gates toward zero -- an input gate at -10 exists to start closed, and 0.01 of weight
    # decay is a force pushing it open. Every other arm in this table exempts its equivalent.
    "xlstm": ("*.w_if.bias", "*._bias_"),
    "mamba3-siso-pd": ("*.A_log", "*.dt_bias", "*.D"),
    "native-pd": ("*.A_log", "*.dt_bias", "*.D"),
    "gdn": ("*.A_log", "*.dt_bias"),
    "kda": ("*.A_log", "*.dt_bias"),
    "kda-hh-r2": ("*.A_log", "*.dt_bias"),
    "kda-gconv": ("*.A_log", "*.dt_bias"),
}


def weight_decay_group_overrides(arm: str) -> list[OptimGroupOverride]:
    """
    Return the zero-weight-decay optimizer group for one arm.

    :param arm: A runnable arm name.

    :returns: One override naming the embeddings plus that arm's tagged timescale parameters.

    :raises ValueError: If the arm is not runnable.
    """
    if arm not in RUNNABLE_ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    patterns = ["embeddings.weight", *WEIGHT_DECAY_EXEMPT_PATTERNS_BY_ARM[arm]]
    return [OptimGroupOverride(params=patterns, opts={"weight_decay": 0.0})]


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
        dtype=MASTER_DTYPE,
    )


def _feed_forward(width: int) -> FeedForwardConfig:
    return FeedForwardConfig(hidden_size=width, bias=False, dtype=MASTER_DTYPE)


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
        backend=AttentionBackendName.torch,
        dtype=MASTER_DTYPE,
    )


def _gdn_mixer() -> GatedDeltaNet2Config:
    """Build the frozen measured GDN2 mixer of the ``gdn`` arm."""
    return GatedDeltaNet2Config(n_heads=16, head_dim=64, expand_v=1.0, dtype=MASTER_DTYPE)


def _kda_mixer() -> KimiDeltaAttentionConfig:
    """Build the shipped Kimi Delta Attention mixer of the ``kda`` arm.

    The baseline of the KDA family: one delta factor per token, plain SiLU short convolutions,
    non-negative eigenvalues. Every option the other two arms move is left at its default here,
    so that each of them differs from this arm in exactly one mechanism.
    """
    return KimiDeltaAttentionConfig(n_heads=16, head_dim=64, dtype=MASTER_DTYPE)


def _kda_householder_mixer() -> KimiDeltaHouseholderConfig:
    """Build the R=2, negative-eigenvalue DeltaProduct mixer of the ``kda-hh-r2`` arm.

    Two rank-1 updates per token instead of one, which widens ``w_k``, ``w_v``, ``w_b`` and the
    ``k``/``v`` convolutions by that factor and makes this the widest mixer in the wave at
    6,608,976 parameters a layer. ``allow_neg_eigval`` lets each factor reflect rather than only
    contract, which is the point of the variant and not a tuning knob.

    ``backend="triton"`` is the fused recurrent path. The ``"torch"`` reference is the only one
    that runs on CPU and is far slower; a cell that quietly selected it would be ranked on
    throughput against seven fused arms.
    """
    return KimiDeltaHouseholderConfig(
        n_heads=16,
        head_dim=64,
        num_householder=2,
        allow_neg_eigval=True,
        backend="triton",
        dtype=MASTER_DTYPE,
    )


def _kda_gated_conv_mixer() -> KimiDeltaAttentionConfig:
    """Build the LIV-style gated-convolution mixer of the ``kda-gconv`` arm.

    The same class and head geometry as :func:`_kda_mixer` with the three plain short
    convolutions replaced by gated ones, so the arm's whole per-layer difference from ``kda`` is
    the 6,144 gate parameters -- about 0.14% of the layer, which is what keeps the contrast a
    mechanism rather than a capacity difference.

    ``gate_structure="depthwise"`` leaves ``gated_conv_activation`` at ``None``, and that is not
    an activation-free convolution: the depthwise pre-gate is a SiLU with a learnable per-channel
    slope moved ahead of the convolution. See :class:`KimiDeltaAttentionConfig` for the identity.
    """
    return KimiDeltaAttentionConfig(
        n_heads=16,
        head_dim=64,
        gated_conv=True,
        gate_structure="depthwise",
        dtype=MASTER_DTYPE,
    )


def _mlstm_mixer() -> XLSTMMixerConfig:
    return XLSTMMixerConfig(
        n_heads=4,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        conv_size=4,
        chunkwise_kernel="chunkwise--triton_xl_chunk",
        # 128, NOT 256. In mlstm-kernels 2.0.4, chunk 256 uses 128-token
        # recurrent chunks but saves only every second state. At T=4096 the
        # forward allocates 17 max-state slots while the backward indexes
        # through slot 32, producing finite logits and NaN gradients on the
        # first backward pass. At 128 every indexed recurrent state is saved;
        # it was also level with 256 in the paired local mixer benchmark.
        chunk_size=128,
        autocast_kernel_dtype="bfloat16",
        dtype=MASTER_DTYPE,
    )


def _slstm_mixer() -> SLSTMMixerConfig:
    return SLSTMMixerConfig(
        n_heads=4,
        conv_size=4,
        backend="cuda_fused",
        batch_size=2,
        kernel_dtype="bfloat16",
        fuse_input_projections=True,
        dtype=MASTER_DTYPE,
    )


def _treatment_mixer(arm: str, layer_index: int):
    if arm == "gdn":
        return _gdn_mixer()
    if arm == "kda":
        return _kda_mixer()
    if arm == "kda-hh-r2":
        return _kda_householder_mixer()
    if arm == "kda-gconv":
        return _kda_gated_conv_mixer()
    if arm == "mamba-b3":
        # Faithful published Mamba-3 SISO, with the single intentional deviation that SO(2) is
        # generalized to SO(3) b=3 for NC^1 state-tracking. The fidelity audit found the prior
        # arm departed from published SISO in six ways beyond the rotation; each is restored here:
        #   - expand=2 (32 heads x 64 = 2048 inner), the published SISO mixer width;
        #   - token-dependent decay A on top of the per-head A_log baseline (dynamic_a);
        #   - head-specific B/C bias initialized to one, applied AFTER BCNorm (bc_bias_after_norm);
        #   - a learned D skip initialized to one (d_skip);
        #   - norm-before-gate output ordering (norm_before_gate);
        #   - the official tanh(angle)*pi*dt per-head rotation over half the state, the rest
        #     identity (dt_scaled_rotation + rope_fraction=0.5).
        # b=3, d_state=192, SISO rank 1, official_fast/quaternion, and the pre-norm shell (set in
        # `_block_type`) complete the arm. The faithful options are only wired for the unfused
        # layout, so `fuse_input_projections` is False; theta_max is unused because tanh*pi*dt
        # bounds the angle itself.
        return Mamba3MixerConfig(
            n_heads=32,
            head_dim=64,
            d_state=192,
            n_groups=1,
            mimo_rank=1,
            rotation_block_size=3,
            norm_eps=1e-6,
            bc_norm=True,
            bc_bias=False,
            dynamic_a=True,
            d_skip=True,
            norm_before_gate=True,
            bc_bias_after_norm=True,
            dt_scaled_rotation=True,
            rope_fraction=0.5,
            prefer_official_kernel=True,
            rotation_scan_impl="quaternion",
            # Exact fused Mamba-3 recurrence. ``simple_gla`` is an approximate algebraic fold with
            # non-zero parity tolerances; it stays an explicit benchmark backend, not the default.
            ssd_backend="official_fast",
            theta_max=None,
            fuse_input_projections=False,
            dtype=MASTER_DTYPE,
        )
    if arm == "xlstm":
        return _slstm_mixer() if layer_index in XLSTM_SLSTM_LAYERS else _mlstm_mixer()
    if arm == "native-pd":
        return NativeFlashPDMixerConfig(
            n_heads=16,
            d_state=64,
            dictionary_size=16,
            # 64, MEASURED, NOT THE 128 THIS ARM WAS FIRST WRITTEN WITH. `paper_backward` took
            # 2.98-3.00 ms a layer-step at 128 against 2.59-2.72 ms at 64, with the forwards
            # level: about 0.3 ms a layer-step across twelve layers. The chunk blocks the scan
            # and shapes no weight, so this moved no parameter -- the arm's solved widths and
            # its 390,142,976 exact count are the ones the four-arm study was described with,
            # and `test_native_pd_chunk_size_is_64_and_the_change_shaped_no_weights` asserts
            # exactly that rather than leaving it assumed.
            chunk_size=64,
            ste_temperature=1.0,
            mode=NativePDMode.GENERAL_SCATTER,
            backend=NativePDBackend.CUDA,
            conv_kernel_size=4,
            fuse_input_projections=False,
            dtype=MASTER_DTYPE,
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
            fuse_input_projections=False,
            dtype=MASTER_DTYPE,
        )
    raise ValueError(f"unsupported arm: {arm}")


# Every arm feeds its mixer the reordered (post-)norm block that the wave was frozen with,
# except the faithful Mamba arm: published Mamba is pre-norm, and the fidelity audit found that
# feeding the raw residual stream through a post-norm shell was one of this arm's deviations.
# The switch is param-neutral (reordered_norm and default carry the same two norms, only the
# forward ordering differs), so it changes the arm's behaviour without moving its parameter count.
_PRE_NORM_ARMS = frozenset({"mamba-b3"})


def _block_type(arm: str) -> TransformerBlockType:
    return (
        TransformerBlockType.default
        if arm in _PRE_NORM_ARMS
        else TransformerBlockType.reordered_norm
    )


def _block(mixer, width: int, block_type: TransformerBlockType) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        name=block_type,
        sequence_mixer=mixer,
        feed_forward=_feed_forward(width),
        layer_norm=_layer_norm(),
    )


def _model_for_widths(arm: str, widths: tuple[int, ...], init_seed: int) -> TransformerConfig:
    if arm not in RUNNABLE_ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if len(widths) != len(RECURRENT_LAYERS):
        raise ValueError(
            f"expected {len(RECURRENT_LAYERS)} recurrent FFN widths, got {len(widths)}"
        )
    block_type = _block_type(arm)
    blocks = {"attention": _block(_attention_mixer(), BASE_FFN_WIDTH, block_type)}
    pattern: list[str] = []
    width_by_layer = dict(zip(RECURRENT_LAYERS, widths))
    for index in range(N_LAYERS):
        if index in ATTENTION_LAYERS:
            pattern.append("attention")
        else:
            name = f"recurrent-{index}"
            blocks[name] = _block(_treatment_mixer(arm, index), width_by_layer[index], block_type)
            pattern.append(name)
    return TransformerConfig(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        block=blocks,
        block_pattern=pattern,
        lm_head=LMHeadConfig(layer_norm=_layer_norm(), bias=False, dtype=MASTER_DTYPE),
        dtype=MASTER_DTYPE,
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
    if arm not in RUNNABLE_ARMS:
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
            group_overrides=weight_decay_group_overrides(opts.arm),
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
    parser.add_argument("--arm", required=True, choices=RUNNABLE_ARMS)
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
    # Kept equal to `train_core6_arm.py`'s default, which is the program every spec in
    # `.edullm/` actually invokes. This one is reachable only by running this file directly,
    # and a default that disagreed would train a different recipe under the same arm names.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
