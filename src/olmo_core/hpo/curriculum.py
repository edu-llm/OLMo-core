"""Token-progress curriculum data and experiment configuration for the HPO arm."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from ..config import Config
from ..data import NumpyDatasetDType, TokenizerConfig
from ..data.collator import DataCollator
from ..data.data_loader import DataLoaderConfig, TextDataLoaderBase
from ..distributed.parallel import get_dp_process_group
from ..distributed.utils import get_fs_local_rank, get_rank, get_world_size
from .comparison import (
    ComparisonExperimentConfig,
    build_comparison_experiment,
    build_olmoe_hpo_experiment,
    comparison_heldout_label,
)

__all__ = [
    "ARM9_BOUNDARY_NUMERATORS",
    "ARM9_PACING_DENOMINATOR",
    "ARM9_PACING_ID",
    "CURRICULUM_DATASET_ID",
    "CURRICULUM_MANIFEST_SHA256",
    "CURRICULUM_ORDER_GROUP",
    "CURRICULUM_TARGET_TOKENS",
    "PARENT_DATASET_ID",
    "PARENT_MANIFEST_SHA256",
    "CurriculumCorpus",
    "CurriculumDataError",
    "CurriculumDataLoader",
    "CurriculumDataLoaderConfig",
    "CurriculumExperimentConfig",
    "CurriculumInputIdentity",
    "ParentChunkDataset",
    "ParentChunkDatasetConfig",
    "build_curriculum_hpo_experiment",
    "build_olmoe_curriculum_hpo_experiment",
    "curriculum_corpus_from_reads",
    "curriculum_pool_for_tokens",
    "token_phase_boundaries",
    "validate_complete_permutation",
]

PARENT_DATASET_ID = "pretrain/opt-with-synthetic-10b"
PARENT_DATASET_VERSION = "v1"
PARENT_DATASET_GROUP = "tokens"
PARENT_MANIFEST_SHA256 = "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797"

CURRICULUM_DATASET_ID = "curriculum/opt-with-synthetic-10b"
CURRICULUM_DATASET_VERSION = "v1"
CURRICULUM_ORDER_GROUP = "mtld"
CURRICULUM_MANIFEST_SHA256 = "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d"

CURRICULUM_TARGET_TOKENS = 503_316_480
ARM9_PACING_ID = "arm9_warmup_quadratic_n10_token_fraction_v1"
ARM9_PACING_DENOMINATOR = 2_384
ARM9_BOUNDARY_NUMERATORS = (0, 18, 54, 109, 182, 273, 382, 509, 654, 818, 1000)
STATE_SCHEMA_VERSION = 1


class CurriculumDataError(RuntimeError):
    """The curriculum input, order, pacing, or resume state violates its contract."""


@dataclass(frozen=True)
class CurriculumInputIdentity:
    """Immutable identity for one published curriculum input group."""

    dataset_id: str
    version: str
    group: str
    profile: str
    manifest_sha256: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a complete SHA-256 digest")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form of this identity."""

        payload = asdict(self)
        payload["source_ids"] = list(self.source_ids)
        return payload


@dataclass(frozen=True)
class CurriculumCorpus:
    """Resolved parent train/validation paths and its MTLD order."""

    train_paths: tuple[str, ...]
    val_paths: tuple[str, ...]
    order_paths: tuple[str, ...]
    dtype: NumpyDatasetDType
    order_dtype: NumpyDatasetDType
    parent_identity: CurriculumInputIdentity
    order_identity: CurriculumInputIdentity


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_ids(paths: Sequence[str]) -> tuple[str, ...]:
    source_ids = []
    for path in paths:
        source = PurePosixPath(str(path).replace("\\", "/")).parent.name
        if source and source not in source_ids:
            source_ids.append(source)
    return tuple(source_ids)


def _validate_read_layout(read: Any, *, role: str) -> NumpyDatasetDType:
    if int(read.header_bytes) != 0:
        raise CurriculumDataError(f"{role} input has a nonzero header")
    byte_order = getattr(read, "byte_order", None)
    if byte_order is not None and str(byte_order) != sys.byteorder:
        raise CurriculumDataError(
            f"{role} byte order {byte_order!r} does not match host {sys.byteorder!r}"
        )
    dtype = getattr(read, "dtype", None)
    if dtype is None:
        raise CurriculumDataError(f"{role} input declares no fixed-width dtype")
    try:
        return NumpyDatasetDType(dtype)
    except ValueError as exc:
        raise CurriculumDataError(f"{role} input has unsupported dtype {dtype!r}") from exc


def _require_manifest(read: Any, expected: str, *, role: str) -> None:
    actual = getattr(read, "manifest_sha256", None)
    if actual != expected:
        raise CurriculumDataError(
            f"{role} manifest {actual!r} does not match pinned manifest {expected!r}"
        )


def curriculum_corpus_from_reads(parent_read: Any, order_read: Any) -> CurriculumCorpus:
    """Validate resolved train/validation/order reads against the immutable arm inputs."""

    parent_dtype = _validate_read_layout(parent_read, role="parent")
    order_dtype = _validate_read_layout(order_read, role="order")
    _require_manifest(parent_read, PARENT_MANIFEST_SHA256, role="parent")
    _require_manifest(order_read, CURRICULUM_MANIFEST_SHA256, role="order")

    train_paths = tuple(str(path) for path in parent_read.paths)
    val_paths = tuple(str(path) for path in (parent_read.val or ()))
    order_paths = tuple(str(path) for path in order_read.paths)
    if not train_paths:
        raise CurriculumDataError("parent input has no train partition")
    if not val_paths:
        raise CurriculumDataError("parent input has no validation partition")
    if not order_paths:
        raise CurriculumDataError("MTLD input has no order partition")
    if set(train_paths) & set(val_paths):
        raise CurriculumDataError("parent train and validation partitions overlap")

    sources = _source_ids(train_paths)
    return CurriculumCorpus(
        train_paths=train_paths,
        val_paths=val_paths,
        order_paths=order_paths,
        dtype=parent_dtype,
        order_dtype=order_dtype,
        parent_identity=CurriculumInputIdentity(
            dataset_id=PARENT_DATASET_ID,
            version=PARENT_DATASET_VERSION,
            group=PARENT_DATASET_GROUP,
            profile="pretrain-tokens/v1",
            manifest_sha256=PARENT_MANIFEST_SHA256,
            source_ids=sources,
        ),
        order_identity=CurriculumInputIdentity(
            dataset_id=CURRICULUM_DATASET_ID,
            version=CURRICULUM_DATASET_VERSION,
            group=CURRICULUM_ORDER_GROUP,
            profile="token-order/v1",
            manifest_sha256=CURRICULUM_MANIFEST_SHA256,
        ),
    )


def validate_complete_permutation(order: np.ndarray, parent_size: int) -> None:
    """Require every flat parent chunk index exactly once."""

    if order.ndim != 1 or len(order) != int(parent_size):
        raise CurriculumDataError(
            f"order shape {order.shape} does not match parent size {parent_size}"
        )
    if order.dtype.kind not in "iu":
        raise CurriculumDataError(f"order dtype must be integral, got {order.dtype}")
    normalized = np.asarray(order, dtype=np.int64)
    if normalized.min(initial=0) < 0 or normalized.max(initial=-1) >= parent_size:
        raise CurriculumDataError("order contains a chunk outside the parent pool")
    if not np.array_equal(np.sort(normalized), np.arange(parent_size, dtype=np.int64)):
        raise CurriculumDataError("order is not a complete parent-pool permutation")


class ParentChunkDataset:
    """Shard-local fixed chunks using arm 9's ``(tokens - 1) // length`` coordinates."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        sequence_length: int,
        dtype: str | np.dtype[Any] | type[np.unsignedinteger],
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.sequence_length = int(sequence_length)
        self.dtype = np.dtype(dtype)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.paths:
            raise CurriculumDataError("parent pool has no shards")
        self.source_ids = _source_ids(tuple(str(path) for path in self.paths))
        self._arrays: list[np.memmap[Any, Any]] = []
        self._ends: list[int] = []
        total = 0
        for path in self.paths:
            if not path.is_file():
                raise CurriculumDataError(f"missing staged parent shard: {path}")
            array = np.memmap(path, mode="r", dtype=self.dtype)
            chunks = (len(array) - 1) // self.sequence_length
            if chunks <= 0:
                continue
            self._arrays.append(array)
            total += chunks
            self._ends.append(total)
        if total <= 0:
            raise CurriculumDataError("parent pool contains no complete next-token chunks")
        self._size = total

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, flat_index: int) -> dict[str, Any]:
        index = int(flat_index)
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(index)
        shard = bisect.bisect_right(self._ends, index)
        prior_end = self._ends[shard - 1] if shard else 0
        local_index = index - prior_end
        start = local_index * self.sequence_length
        tokens = np.asarray(
            self._arrays[shard][start : start + self.sequence_length],
            dtype=np.int64,
        )
        return {"input_ids": torch.from_numpy(tokens.copy()), "index": index}


def token_phase_boundaries(target_tokens: int) -> tuple[int, ...]:
    """Translate arm 9's step numerators into exact token-progress boundaries."""

    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    denominator = ARM9_PACING_DENOMINATOR
    return tuple(
        (numerator * target_tokens + denominator - 1) // denominator
        for numerator in ARM9_BOUNDARY_NUMERATORS
    )


def _equal_mass_buckets(size: int, buckets: int = 10) -> tuple[tuple[int, int], ...]:
    if size <= 0 or buckets <= 0:
        raise ValueError("size and buckets must be positive")
    width, remainder = divmod(size, buckets)
    result = []
    start = 0
    for bucket in range(buckets):
        end = start + width + (1 if bucket < remainder else 0)
        result.append((start, end))
        start = end
    return tuple(result)


def curriculum_pool_for_tokens(
    tokens_seen: int,
    parent_size: int,
    target_tokens: int,
) -> tuple[int, int]:
    """Return the active MTLD decile at token progress, or the full pool after warmup."""

    tokens_seen = int(tokens_seen)
    if tokens_seen < 0:
        raise ValueError("tokens_seen must be non-negative")
    boundaries = token_phase_boundaries(target_tokens)
    if tokens_seen >= boundaries[-1]:
        return 0, int(parent_size)
    bucket = bisect.bisect_right(boundaries, tokens_seen) - 1
    start, end = _equal_mass_buckets(int(parent_size))[bucket]
    if start == end:
        end = min(int(parent_size), start + 1)
    return start, end


class CurriculumDataLoader(TextDataLoaderBase):
    """Deterministic MTLD loader paced by lineage token progress."""

    def __init__(
        self,
        dataset: ParentChunkDataset,
        *,
        ranked_chunk_indices: Sequence[int] | np.ndarray,
        pacing: str,
        difficulty_metric: str,
        seed: int,
        target_tokens: int,
        global_batch_size: int,
        work_dir: str | Path,
        parent_identity: CurriculumInputIdentity,
        order_identity: CurriculumInputIdentity,
        pad_token_id: int,
        vocab_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int | None = None,
    ) -> None:
        if pacing != ARM9_PACING_ID:
            raise ValueError(f"unknown curriculum pacing {pacing!r}")
        if difficulty_metric != "mtld":
            raise ValueError("the curriculum HPO arm requires the MTLD order")
        if global_batch_size <= 0 or global_batch_size % dataset.sequence_length:
            raise CurriculumDataError("global batch must contain whole sequences")
        if target_tokens <= 0 or target_tokens % global_batch_size:
            raise CurriculumDataError("target tokens must be divisible by the lineage batch")
        if global_batch_size % dp_world_size:
            raise CurriculumDataError("global batch must be divisible by DP world size")
        rank_tokens = global_batch_size // dp_world_size
        if rank_tokens % dataset.sequence_length:
            raise CurriculumDataError("rank batch must contain whole sequences")

        super().__init__(
            collator=DataCollator(pad_token_id=pad_token_id, vocab_size=vocab_size),
            work_dir=work_dir,
            global_batch_size=int(global_batch_size),
            dp_world_size=int(dp_world_size),
            dp_rank=int(dp_rank),
            fs_local_rank=fs_local_rank,
        )
        self.dataset = dataset
        self.pacing = pacing
        self.difficulty_metric = difficulty_metric
        self.seed = int(seed)
        self.target_tokens = int(target_tokens)
        self.parent_identity = parent_identity
        self.order_identity = order_identity
        self.ranked = np.asarray(ranked_chunk_indices, dtype=np.int64)
        self._batch_size_rebase_allowed = False
        validate_complete_permutation(self.ranked, len(self.dataset))

    @property
    def sequence_length(self) -> int:
        return self.dataset.sequence_length

    @property
    def global_sequences_per_batch(self) -> int:
        return self.global_batch_size // self.sequence_length

    @property
    def rank_sequences_per_batch(self) -> int:
        return self.rank_batch_size // self.sequence_length

    @property
    def total_batches(self) -> int:
        return self.target_tokens // self.global_batch_size

    @property
    def global_train_tokens_seen(self) -> int:
        return self.tokens_processed

    @property
    def scientific_identity(self) -> dict[str, Any]:
        """Identity fields that must remain fixed throughout one HPO lineage."""

        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "parent": self.parent_identity.as_dict(),
            "order": self.order_identity.as_dict(),
            "pacing": self.pacing,
            "token_phase_boundaries": list(token_phase_boundaries(self.target_tokens)),
            "difficulty_metric": self.difficulty_metric,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "target_tokens": self.target_tokens,
            "lineage_global_batch_size": self.global_batch_size,
        }

    def state_dict(self) -> dict[str, Any]:
        identity = self.scientific_identity
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "identity": identity,
            "identity_sha256": _identity_sha256(identity),
            "batches_processed": self.batches_processed,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "epoch": self._epoch,
        }

    def allow_batch_size_rebase(self) -> None:
        """Allow one checkpoint load to rebase token progress onto this loader's batch size."""

        self._batch_size_rebase_allowed = True

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        batch_size_rebase_allowed = self._batch_size_rebase_allowed
        self._batch_size_rebase_allowed = False
        if state_dict.get("schema_version") != STATE_SCHEMA_VERSION:
            raise CurriculumDataError("unsupported curriculum loader state schema")
        identity = state_dict.get("identity")
        if not isinstance(identity, Mapping):
            raise CurriculumDataError("loader state has no scientific identity")
        if state_dict.get("identity_sha256") != _identity_sha256(identity):
            raise CurriculumDataError("loader state identity checksum is invalid")
        current_identity = self.scientific_identity
        changed = sorted(
            key
            for key in set(identity) | set(current_identity)
            if identity.get(key) != current_identity.get(key)
        )
        rebasing_batch_size = changed == ["lineage_global_batch_size"]
        if changed and not (rebasing_batch_size and batch_size_rebase_allowed):
            raise CurriculumDataError(f"refusing loader resume with changed identity: {changed}")
        batches = int(state_dict.get("batches_processed", -1))
        tokens = int(state_dict.get("global_train_tokens_seen", -1))
        saved_batch_size = int(identity.get("lineage_global_batch_size", -1))
        if batches < 0 or saved_batch_size <= 0 or tokens != batches * saved_batch_size:
            raise CurriculumDataError("loader state does not identify the next batch exactly")
        if rebasing_batch_size:
            if tokens % self.global_batch_size:
                raise CurriculumDataError(
                    "loader token progress is not aligned to the resumed lineage batch"
                )
            batches = tokens // self.global_batch_size
        self.batches_processed = batches
        self.tokens_processed = tokens
        self._epoch = int(state_dict.get("epoch") or 1)

    def reshuffle(self, epoch: Optional[int] = None, **kwargs: Any) -> None:
        del kwargs
        self._epoch = int(epoch if epoch is not None else (self._epoch or 0) + 1)
        if self._epoch <= 0:
            raise ValueError("epoch must be positive")

    def _token_seed(self, tokens_seen: int) -> int:
        pacing_hash = int(hashlib.sha256(self.pacing.encode()).hexdigest()[:8], 16)
        return (self.seed * 1_000_003 + int(tokens_seen) * 97_651 + pacing_hash) & 0x7FFFFFFF

    def global_indices_for_tokens(self, tokens_seen: int) -> np.ndarray:
        """Select one global batch deterministically at the given token progress."""

        if not 0 <= int(tokens_seen) < self.target_tokens:
            raise IndexError(tokens_seen)
        start, end = curriculum_pool_for_tokens(
            tokens_seen,
            len(self.ranked),
            self.target_tokens,
        )
        width = end - start
        take = self.global_sequences_per_batch
        rng = np.random.default_rng(self._token_seed(tokens_seen))
        positions = rng.choice(width, size=take, replace=width < take)
        return self.ranked[start + positions]

    def global_indices_for_step(self, step: int) -> np.ndarray:
        """Select one global batch using zero-based lineage batch index."""

        if not 0 <= int(step) < self.total_batches:
            raise IndexError(step)
        return self.global_indices_for_tokens(int(step) * self.global_batch_size)

    def batch_for_step(self, step: int) -> dict[str, Any]:
        global_indices = self.global_indices_for_step(step)
        start = self.dp_rank * self.rank_sequences_per_batch
        stop = start + self.rank_sequences_per_batch
        items = [self.dataset[int(index)] for index in global_indices[start:stop]]
        return self.collator(items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.batch_for_step(index)

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        for step in range(self.batches_processed, self.total_batches):
            yield self.batch_for_step(step)

    def get_mock_batch(self) -> dict[str, Any]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.dp_rank)
        return {
            "input_ids": torch.randint(
                0,
                self.collator.vocab_size or 100_352,
                (self.rank_sequences_per_batch, self.sequence_length),
                generator=generator,
            )
        }


@dataclass
class ParentChunkDatasetConfig(Config):
    """Configuration for the immutable flat parent chunk pool."""

    paths: list[str]
    sequence_length: int
    dtype: NumpyDatasetDType

    def build(self) -> ParentChunkDataset:
        """Build the parent pool from staged immutable shards."""

        return ParentChunkDataset(
            self.paths,
            sequence_length=self.sequence_length,
            dtype=self.dtype.as_np_dtype(),
        )


def _load_order(paths: Sequence[str], dtype: NumpyDatasetDType) -> np.ndarray:
    parts = [np.memmap(path, mode="r", dtype=dtype.as_np_dtype()) for path in paths]
    if not parts:
        raise CurriculumDataError("curriculum order resolved to no objects")
    return np.asarray(
        np.concatenate(parts) if len(parts) > 1 else parts[0],
        dtype=np.int64,
    )


@dataclass
class CurriculumDataLoaderConfig(DataLoaderConfig[CurriculumDataLoader]):
    """Configuration for the token-progress MTLD loader."""

    global_batch_size: int
    seed: int
    target_tokens: int
    order_paths: list[str]
    order_dtype: NumpyDatasetDType
    parent_identity: CurriculumInputIdentity
    order_identity: CurriculumInputIdentity
    tokenizer: TokenizerConfig
    work_dir: str
    pacing: str = ARM9_PACING_ID
    difficulty_metric: str = "mtld"

    def build(
        self,
        dataset: ParentChunkDataset,
        *,
        collator: Optional[DataCollator] = None,
        mesh: Optional[DeviceMesh] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ) -> CurriculumDataLoader:
        """Build the loader after distributed training topology is known."""

        del collator
        if dp_process_group is None and mesh is not None:
            dp_process_group = get_dp_process_group(mesh)
        return CurriculumDataLoader(
            dataset,
            ranked_chunk_indices=_load_order(self.order_paths, self.order_dtype),
            pacing=self.pacing,
            difficulty_metric=self.difficulty_metric,
            seed=self.seed,
            target_tokens=self.target_tokens,
            global_batch_size=self.global_batch_size,
            work_dir=self.work_dir,
            parent_identity=self.parent_identity,
            order_identity=self.order_identity,
            pad_token_id=self.tokenizer.pad_token_id,
            vocab_size=self.tokenizer.vocab_size,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
        )


@dataclass
class CurriculumExperimentConfig(ComparisonExperimentConfig):
    """The stock 190M comparison experiment with arm 9 curriculum data."""

    dataset: ParentChunkDatasetConfig
    data_loader: CurriculumDataLoaderConfig
    curriculum_identity: dict[str, Any] | None = None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"the eduLLM platform did not set {name}")
    return value


def _build_curriculum_hpo_experiment(
    base_builder: Callable[..., ComparisonExperimentConfig],
    *,
    sequence_length: int = 2048,
    global_batch_size: int = 524_288,
    rank_microbatch_size: int = 4096,
    data_seed: int = 210007,
    init_seed: int = 110007,
    eval_steps: int = 2,
    target_tokens: int = CURRICULUM_TARGET_TOKENS,
    work_dir: str = "/tmp/hpo-curriculum-data",
    data_bucket: str | None = None,
) -> CurriculumExperimentConfig:
    """Build a supplied model recipe with arm 9's fixed token-progress MTLD pacing."""

    dataset_id = _required_env("EDULLM_DATASET_ID")
    version = _required_env("EDULLM_DATASET_VERSION")
    tokenizer_id = _required_env("EDULLM_DATASET_TOKENIZER")
    if (dataset_id, version, tokenizer_id) != (
        PARENT_DATASET_ID,
        PARENT_DATASET_VERSION,
        "tokenizer/dolma2-bpe",
    ):
        raise ValueError("curriculum HPO received a dataset other than the pinned parent release")
    if target_tokens != CURRICULUM_TARGET_TOKENS:
        raise ValueError("curriculum HPO target_tokens differs from the approved token horizon")
    if global_batch_size < 256 * 1024 or global_batch_size > 1024 * 1024:
        raise ValueError("curriculum HPO global batch must be in the approved 256 Ki-1 Mi range")

    base = base_builder(
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        rank_microbatch_size=rank_microbatch_size,
        data_seed=data_seed,
        init_seed=init_seed,
        eval_steps=eval_steps,
        work_dir=work_dir,
        dataset_group=PARENT_DATASET_GROUP,
        data_bucket=data_bucket,
    )

    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    read_kwargs: dict[str, Any] = {"s3": s3}
    if data_bucket is not None:
        read_kwargs["data_bucket"] = data_bucket
    parent_read = dataset_paths(
        PARENT_DATASET_ID,
        PARENT_DATASET_VERSION,
        group=PARENT_DATASET_GROUP,
        **read_kwargs,
    )
    order_read = dataset_paths(
        CURRICULUM_DATASET_ID,
        CURRICULUM_DATASET_VERSION,
        split="train",
        group=CURRICULUM_ORDER_GROUP,
        **read_kwargs,
    )
    corpus = curriculum_corpus_from_reads(parent_read, order_read)
    tokenizer = TokenizerConfig.dolma2()
    parent_config = ParentChunkDatasetConfig(
        paths=list(corpus.train_paths),
        sequence_length=sequence_length,
        dtype=corpus.dtype,
    )
    loader_config = CurriculumDataLoaderConfig(
        global_batch_size=global_batch_size,
        seed=data_seed,
        target_tokens=target_tokens,
        order_paths=list(corpus.order_paths),
        order_dtype=corpus.order_dtype,
        parent_identity=corpus.parent_identity,
        order_identity=corpus.order_identity,
        tokenizer=tokenizer,
        work_dir=work_dir,
    )
    curriculum_identity = {
        "parent": corpus.parent_identity.as_dict(),
        "order": corpus.order_identity.as_dict(),
        "pacing": ARM9_PACING_ID,
        "token_phase_boundaries": list(token_phase_boundaries(target_tokens)),
        "difficulty_metric": "mtld",
        "heldout_label": comparison_heldout_label(PARENT_DATASET_ID),
        "target_tokens": target_tokens,
        "sequence_length": sequence_length,
    }
    return CurriculumExperimentConfig(
        model=base.model,
        dataset=parent_config,
        data_loader=loader_config,
        trainer=base.trainer,
        train_module=base.train_module,
        dataset_id=base.dataset_id,
        dataset_version=base.dataset_version,
        init_seed=base.init_seed,
        umup_backend=base.umup_backend,
        umup_parity_validated=base.umup_parity_validated,
        umup_metadata=base.umup_metadata,
        curriculum_identity=curriculum_identity,
    )


def build_curriculum_hpo_experiment(
    *,
    sequence_length: int = 2048,
    global_batch_size: int = 524_288,
    rank_microbatch_size: int = 4096,
    data_seed: int = 210007,
    init_seed: int = 110007,
    eval_steps: int = 2,
    target_tokens: int = CURRICULUM_TARGET_TOKENS,
    work_dir: str = "/tmp/hpo-curriculum-data",
    data_bucket: str | None = None,
) -> CurriculumExperimentConfig:
    """Build stock ``olmo2_190M`` with arm 9's fixed token-progress MTLD pacing."""

    return _build_curriculum_hpo_experiment(
        build_comparison_experiment,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        rank_microbatch_size=rank_microbatch_size,
        data_seed=data_seed,
        init_seed=init_seed,
        eval_steps=eval_steps,
        target_tokens=target_tokens,
        work_dir=work_dir,
        data_bucket=data_bucket,
    )


def build_olmoe_curriculum_hpo_experiment(
    *,
    sequence_length: int = 2048,
    global_batch_size: int = 262_144,
    rank_microbatch_size: int = 32_768,
    data_seed: int = 210007,
    init_seed: int = 110007,
    eval_steps: int = 2,
    target_tokens: int = CURRICULUM_TARGET_TOKENS,
    work_dir: str = "/tmp/hpo-olmoe-curriculum-data",
    data_bucket: str | None = None,
) -> CurriculumExperimentConfig:
    """Build stock OLMoE-1B-7B with arm 9's fixed token-progress MTLD pacing."""

    return _build_curriculum_hpo_experiment(
        build_olmoe_hpo_experiment,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        rank_microbatch_size=rank_microbatch_size,
        data_seed=data_seed,
        init_seed=init_seed,
        eval_steps=eval_steps,
        target_tokens=target_tokens,
        work_dir=work_dir,
        data_bucket=data_bucket,
    )
