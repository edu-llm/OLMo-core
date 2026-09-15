"""Resumable OLMo-core loader for parent-pool curriculum orders.

Rewritten from scratch for `curriculum-new` (no code copied from the edullm
repo). Adaptive-bucket and domain-mixture pacings are gone entirely -- both
depended on modules that were never version-controlled anywhere in the prior
branch, and neither is part of the frozen arm set (plan section 6b).

The sampling mechanism is the direct fix for the audit's largest finding
(1a): every pacing, including control, now draws without replacement from
its active pool, resuming a per-pool no-replacement walk across the whole
run rather than resampling with replacement on every step. `curriculum_pacing
.pool_for_step` supplies the pool identity and a `prior_draws` count in
closed form; this module turns that into actual chunk indices via a cached
per-pool permutation, keyed by (seed, pool_start, pool_end, cycle) so that
two arms sharing a seed and visiting the same decile draw identical chunks
at the same cumulative offset (audit finding 1g, common random numbers,
achieved without special-casing pacing names).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from olmo_core.data import TextDataLoaderBase
from olmo_core.data.collator import DataCollator

from curriculum_pacing import (
    CURRICULUM_STEPS,
    DIFFICULTY_METRICS,
    N_BUCKETS,
    PACING_NAMES,
    TOTAL_STEPS,
    pool_for_step,
)
from curriculum_sampling import PoolSampler

log = logging.getLogger(__name__)

STATE_SCHEMA = 3


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
        self._shard_layout: list[tuple[Path, int]] = []
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
            self._shard_layout.append((path, chunks))
        if total <= 0:
            raise CurriculumDataError("parent pool contains no complete next-token chunks")
        self._size = total

    def __len__(self) -> int:
        return self._size

    def shard_layout(self) -> tuple[tuple[Path, int], ...]:
        """``(path, chunk_count)`` for every shard actually kept, in accumulation order."""
        return tuple(self._shard_layout)

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
        allow_short_run: bool = False,
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
        # The freeze contract's point is that a production arm cannot silently
        # differ in length, so an undeclared mismatch is refused. A run that
        # declares itself short (the pre-campaign smoke, via --length-tokens)
        # is allowed: its shortened `total_steps` is recorded in the run
        # fingerprint, so it is already distinguishable from a full-length arm
        # and can never be pooled with one.
        #
        # Note what a short run does NOT do: the pacing schedules are defined
        # against TOTAL_STEPS, so stopping early TRUNCATES the curriculum
        # rather than compressing it. A 250-step run reaches decile 2 and no
        # further. That is fine for exercising the plumbing and meaningless as
        # an arm, which is exactly why it must stay unpoolable.
        if int(total_steps) != TOTAL_STEPS:
            if not allow_short_run:
                raise CurriculumDataError(
                    f"total_steps must equal curriculum_pacing.TOTAL_STEPS ({TOTAL_STEPS}), "
                    f"got {total_steps}"
                )
            if not 0 < int(total_steps) < TOTAL_STEPS:
                raise CurriculumDataError(
                    f"a short run must satisfy 0 < total_steps < {TOTAL_STEPS}, "
                    f"got {total_steps}"
                )
            log.warning(
                "SHORT RUN: %d of %d steps. The curriculum is truncated, not "
                "compressed -- this run reaches decile %d and is not poolable "
                "with a full-length arm.",
                int(total_steps),
                TOTAL_STEPS,
                min(N_BUCKETS, 1 + (int(total_steps) * N_BUCKETS) // CURRICULUM_STEPS),
            )
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
        self._sampler = PoolSampler(self.seed)

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

    def global_indices_for_step(self, step: int) -> np.ndarray:
        if not 0 <= int(step) < self.total_batches:
            raise IndexError(step)
        take = self.global_sequences_per_batch
        size = len(self.dataset) if self.pacing == "control" else len(self.ranked)  # type: ignore[arg-type]
        pool = pool_for_step(step, size, self.pacing, take)
        positions = self._sampler.draw(pool.start, pool.end, pool.prior_draws, take)
        if self.pacing == "control":
            return positions
        assert self.ranked is not None
        return self.ranked[positions]

    def batch_for_step(self, step: int) -> dict[str, Any]:
        global_indices = self.global_indices_for_step(step)
        start = self.dp_rank * self.rank_sequences_per_batch
        stop = start + self.rank_sequences_per_batch
        items = [self.dataset[int(index)] for index in global_indices[start:stop]]
        return self.collator(items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Every pacing is now a pure function of step (no adaptive,
        # loss-history-dependent pacing remains), so random access is always
        # safe -- unlike the prior implementation, which forbade it for
        # adaptive_bucket_n10.
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
