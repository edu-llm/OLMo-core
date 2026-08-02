"""Deterministic, distributed, resumable weighted domain loader for Skill-It."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from olmo_core.data import (
    DataCollator,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TextDataLoaderBase,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetBase

from skillit_math import DATASET_ID, DATASET_VERSION, DOMAINS, initial_weights

SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 4_194_304
TOTAL_STEPS = 2_384
SEED = 42


class SkillItDataError(RuntimeError):
    """The published pool or loader resume state violates the fixed contract."""


def resolve_domain_datasets(work_dir: str | Path) -> tuple[NumpyDatasetBase, ...]:
    """Resolve the seven immutable labeled views through ``edullm_data.read``."""
    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    tokenizer = TokenizerConfig.dolma2()
    s3 = Boto3S3.default()
    datasets: list[NumpyDatasetBase] = []
    for domain in DOMAINS:
        resolved = dataset_paths(
            DATASET_ID,
            DATASET_VERSION,
            split="train",
            s3=s3,
            labels={"source": domain},
        )
        if not resolved.paths:
            raise SkillItDataError(f"{domain}: published source resolved no training shards")
        if resolved.dtype != "uint32":
            raise SkillItDataError(f"{domain}: expected uint32, got {resolved.dtype!r}")
        if resolved.byte_order != "little":
            raise SkillItDataError(
                f"{domain}: expected explicit little byte order, got {resolved.byte_order!r}"
            )
        if int(resolved.header_bytes or 0) != 0:
            raise SkillItDataError(
                f"{domain}: expected headerless shards, got {resolved.header_bytes!r}"
            )
        if any(not str(path).startswith("s3://edullm-data/") for path in resolved.paths):
            raise SkillItDataError(f"{domain}: source escaped s3://edullm-data/")
        config = NumpyFSLDatasetConfig(
            paths=list(resolved.paths),
            tokenizer=tokenizer,
            sequence_length=SEQUENCE_LENGTH,
            dtype=NumpyDatasetDType.uint32,
            work_dir=str(Path(work_dir) / domain),
            include_instance_metadata=False,
        )
        dataset = config.build()
        dataset.prepare()
        if len(dataset) <= 0:
            raise SkillItDataError(f"{domain}: source contains no full sequences")
        datasets.append(dataset)
    return tuple(datasets)


class WeightedDomainDataLoader(TextDataLoaderBase):
    """Sample a domain from ``p_t``, then a random 2,048-token chunk.

    The RNG is keyed by global batch index. Every rank independently reconstructs
    the same global choices and takes its strided rank slice, so rank count does
    not change the global stream and resume needs no opaque generator state.
    """

    def __init__(
        self,
        datasets: Sequence[NumpyDatasetBase],
        *,
        work_dir: str | Path,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        global_batch_size: int = GLOBAL_BATCH_TOKENS,
        sequence_length: int = SEQUENCE_LENGTH,
        total_batches: int = TOTAL_STEPS,
        seed: int = SEED,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ) -> None:
        if len(datasets) != len(DOMAINS):
            raise SkillItDataError(f"expected {len(DOMAINS)} domain datasets")
        self.datasets = tuple(datasets)
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)
        self._total_batches = int(total_batches)
        self._weights = initial_weights()
        tokenizer = TokenizerConfig.dolma2()
        super().__init__(
            collator=DataCollator(
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            ),
            work_dir=work_dir,
            global_batch_size=int(global_batch_size),
            dp_world_size=int(dp_world_size),
            dp_rank=int(dp_rank),
            fs_local_rank=fs_local_rank,
        )
        if self.rank_batch_size % self.sequence_length:
            raise SkillItDataError("rank batch tokens must be divisible by sequence length")
        if self.global_batch_size % self.sequence_length:
            raise SkillItDataError("global batch tokens must be divisible by sequence length")
        self.set_weights(weights if weights is not None else initial_weights())

    @property
    def total_batches(self) -> int:
        return self._total_batches

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def weights_dict(self) -> dict[str, float]:
        return {domain: float(self._weights[i]) for i, domain in enumerate(DOMAINS)}

    def set_weights(self, weights: Sequence[float] | Mapping[str, float]) -> None:
        if isinstance(weights, Mapping):
            vector = np.asarray([float(weights[domain]) for domain in DOMAINS], dtype=np.float64)
        else:
            vector = np.asarray(weights, dtype=np.float64).reshape(-1)
        if vector.shape != (len(DOMAINS),):
            raise ValueError(f"weights must have shape {(len(DOMAINS),)}")
        if not np.isfinite(vector).all() or np.any(vector < 0):
            raise ValueError("weights must be finite and non-negative")
        total = float(vector.sum())
        if total <= 0:
            raise ValueError("weights sum to zero")
        self._weights = vector / total

    def _source_fingerprints(self) -> tuple[str, ...]:
        return tuple(str(dataset.fingerprint) for dataset in self.datasets)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "domain_order": list(DOMAINS),
            "source_fingerprints": list(self._source_fingerprints()),
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "total_batches": self.total_batches,
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "seed": self.seed,
            "epoch": self._epoch,
            "weights": self.weights_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected = {
            "schema": 1,
            "domain_order": list(DOMAINS),
            "source_fingerprints": list(self._source_fingerprints()),
            "sequence_length": self.sequence_length,
            "global_batch_size": self.global_batch_size,
            "total_batches": self.total_batches,
            "seed": self.seed,
        }
        changed = [field for field, value in expected.items() if state_dict.get(field) != value]
        if changed:
            raise SkillItDataError(
                f"refusing loader resume with changed immutable fields: {changed}"
            )
        self.batches_processed = int(state_dict["batches_processed"])
        self.tokens_processed = int(state_dict["tokens_processed"])
        self._epoch = state_dict.get("epoch")
        self.set_weights(state_dict["weights"])

    def reshuffle(self, epoch: Optional[int] = None, **kwargs: Any) -> None:
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch
        if int(epoch) < 1:
            raise ValueError("epoch must be at least 1")
        self._epoch = int(epoch)

    def get_mock_batch(self) -> dict[str, torch.Tensor]:
        count = self.rank_batch_size // self.sequence_length
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.dp_rank)
        return {
            "input_ids": torch.randint(
                0,
                TokenizerConfig.dolma2().vocab_size,
                (count, self.sequence_length),
                generator=generator,
            )
        }

    def _global_choices(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        sequences = self.global_batch_size // self.sequence_length
        rng = np.random.default_rng(self.seed + 1_000_003 * int(batch_index))
        domains = rng.choice(len(DOMAINS), size=sequences, p=self._weights)
        indices = np.empty(sequences, dtype=np.int64)
        for domain_index, dataset in enumerate(self.datasets):
            mask = domains == domain_index
            indices[mask] = rng.integers(0, len(dataset), size=int(mask.sum()))
        return domains, indices

    def batch_at(self, batch_index: int) -> dict[str, Any]:
        domain_ids, instance_ids = self._global_choices(batch_index)
        positions = range(self.dp_rank, len(domain_ids), self.dp_world_size)
        items = [
            self.datasets[int(domain_ids[position])][int(instance_ids[position])]
            for position in positions
        ]
        return self.collator(items)

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        while self.batches_processed < self.total_batches:
            yield self.batch_at(self.batches_processed)


__all__ = [
    "GLOBAL_BATCH_TOKENS",
    "SEED",
    "SEQUENCE_LENGTH",
    "TOTAL_STEPS",
    "SkillItDataError",
    "WeightedDomainDataLoader",
    "resolve_domain_datasets",
]
