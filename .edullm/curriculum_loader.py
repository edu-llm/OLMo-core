"""Resumable OLMo-core loader for parent-pool curriculum orders."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from olmo_core.data import TextDataLoaderBase
from olmo_core.data.collator import DataCollator

from curriculum_pacing import DIFFICULTY_METRICS, PACING_NAMES, pool_for_step

STATE_SCHEMA = 1


class CurriculumDataError(RuntimeError):
    """The parent pool, order, or restored loader state violates the contract."""


def validate_complete_permutation(order: np.ndarray, parent_size: int) -> None:
    """Require each flat parent chunk exactly once."""
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
    """Shard-local ``(tokens - 1) // sequence_length`` flat chunk coordinates."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        sequence_length: int,
        dtype: str | np.dtype[Any],
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.sequence_length = int(sequence_length)
        self.dtype = np.dtype(dtype)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.paths:
            raise CurriculumDataError("parent pool has no shards")
        self._arrays: list[np.memmap] = []
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

    def __getitem__(self, flat_index: int) -> dict[str, torch.Tensor]:
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


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CurriculumDataLoader(TextDataLoaderBase):
    """Step-conditioned loader whose state resumes at the next zero-based batch."""

    def __init__(
        self,
        dataset: ParentChunkDataset,
        *,
        ranked_chunk_indices: Sequence[int] | np.ndarray | None,
        pacing: str,
        difficulty_metric: str | None,
        seed: int,
        total_steps: int,
        global_batch_size: int,
        work_dir: str | Path,
        parent_identity: Mapping[str, Any],
        order_identity: Mapping[str, Any] | None,
        pad_token_id: int,
        vocab_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int | None = None,
    ) -> None:
        if pacing not in PACING_NAMES:
            raise ValueError(f"unknown pacing {pacing!r}")
        if pacing == "control":
            if ranked_chunk_indices is not None or difficulty_metric is not None:
                raise CurriculumDataError("control must not consume a curriculum order")
        else:
            if difficulty_metric not in DIFFICULTY_METRICS:
                raise ValueError(f"unknown difficulty metric {difficulty_metric!r}")
            if ranked_chunk_indices is None or order_identity is None:
                raise CurriculumDataError("curriculum pacing requires an order and identity")
        if global_batch_size % dataset.sequence_length:
            raise CurriculumDataError("global batch must be divisible by sequence length")
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
        self._total_steps = int(total_steps)
        self.parent_identity = dict(parent_identity)
        self.order_identity = dict(order_identity) if order_identity is not None else None
        self.ranked = (
            None
            if ranked_chunk_indices is None
            else np.asarray(ranked_chunk_indices, dtype=np.int64)
        )
        if self.ranked is not None:
            validate_complete_permutation(self.ranked, len(self.dataset))
        self._control_cycle: int | None = None
        self._control_permutation: np.ndarray | None = None

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
        return self._total_steps

    def batches_in_epoch(self, epoch: int) -> int:
        del epoch
        return self._total_steps

    @property
    def scientific_identity(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "parent": self.parent_identity,
            "order": self.order_identity,
            "pacing": self.pacing,
            "difficulty_metric": self.difficulty_metric,
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "total_steps": self.total_batches,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA,
            "identity": self.scientific_identity,
            "identity_sha256": _identity_sha256(self.scientific_identity),
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("schema_version") != STATE_SCHEMA:
            raise CurriculumDataError("unsupported curriculum loader state schema")
        identity = state_dict.get("identity")
        if not isinstance(identity, Mapping):
            raise CurriculumDataError("loader state has no scientific identity")
        if state_dict.get("identity_sha256") != _identity_sha256(identity):
            raise CurriculumDataError("loader state identity checksum is invalid")
        if dict(identity) != self.scientific_identity:
            changed = sorted(
                key
                for key in set(identity) | set(self.scientific_identity)
                if identity.get(key) != self.scientific_identity.get(key)
            )
            raise CurriculumDataError(f"refusing loader resume with changed identity: {changed}")
        batches = int(state_dict.get("batches_processed", -1))
        tokens = int(state_dict.get("tokens_processed", -1))
        if batches < 0 or tokens != batches * self.global_batch_size:
            raise CurriculumDataError("loader state does not identify the next batch exactly")
        self.batches_processed = batches
        self.tokens_processed = tokens
        self._epoch = int(state_dict.get("epoch") or 1)

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        del kwargs
        self._epoch = int(epoch if epoch is not None else (self._epoch or 0) + 1)
        if self._epoch <= 0:
            raise ValueError("epoch must be positive")

    def _step_seed(self, step: int) -> int:
        pacing_tag = sum(ord(char) for char in self.pacing) % 10_007
        return (self.seed * 1_000_003 + int(step) * 97_651 + pacing_tag) & 0x7FFFFFFF

    def _control_indices(self, step: int) -> np.ndarray:
        per_batch = self.global_sequences_per_batch
        batches_per_cycle = len(self.dataset) // per_batch
        if batches_per_cycle <= 0:
            raise CurriculumDataError("control parent is smaller than one global batch")
        cycle, local_step = divmod(int(step), batches_per_cycle)
        if self._control_cycle != cycle:
            rng = np.random.default_rng(self.seed + cycle * 1_000_003)
            self._control_permutation = rng.permutation(len(self.dataset))
            self._control_cycle = cycle
        assert self._control_permutation is not None
        start = local_step * per_batch
        return self._control_permutation[start : start + per_batch]

    def global_indices_for_step(self, step: int) -> np.ndarray:
        if not 0 <= int(step) < self.total_batches:
            raise IndexError(step)
        if self.pacing == "control":
            return self._control_indices(step)
        assert self.ranked is not None
        pool = pool_for_step(step, len(self.ranked), self.pacing)
        take = self.global_sequences_per_batch
        if pool.ordered:
            assert pool.ordered_step is not None
            start = (pool.ordered_step * take) % len(self.ranked)
            positions = np.arange(start, start + take, dtype=np.int64) % len(self.ranked)
            return self.ranked[positions]
        width = pool.end - pool.start
        rng = np.random.default_rng(self._step_seed(step))
        positions = rng.choice(width, size=take, replace=width < take)
        return self.ranked[pool.start + positions]

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
