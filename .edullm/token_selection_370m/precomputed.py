"""Contracts for exact precomputed token-selection masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

MASK_SCHEMA_VERSION = 1
MASK_ALGORITHM = "middle-ppl-token-mask-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights_to_label_mask(weights: Tensor) -> Tensor:
    """Map loss-position weights to the source-token mask consumed by ``get_labels``."""
    if weights.ndim < 1:
        raise ValueError("token weights must have at least one dimension")
    label_mask = torch.zeros_like(weights, dtype=torch.bool)
    label_mask[..., 1:] = weights[..., :-1] > 0
    return label_mask


def binding_sha256(binding: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        binding,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_mask_manifest(
    path: str | Path,
    *,
    corpus_paths: Sequence[str],
    source_ids: Sequence[str],
    source_itemsize: int,
    sequence_length: int,
    keep_fraction: float,
    reference_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MASK_SCHEMA_VERSION:
        raise RuntimeError("unsupported precomputed-mask manifest schema")
    binding = payload.get("binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("precomputed-mask manifest has no binding")
    if payload.get("binding_sha256") != binding_sha256(binding):
        raise RuntimeError("precomputed-mask binding digest mismatch")
    expected = {
        "algorithm": MASK_ALGORITHM,
        "sequence_length": sequence_length,
        "keep_fraction": keep_fraction,
        "reference_sha256": reference_sha256,
        "source_ids": list(source_ids),
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RuntimeError(
                f"precomputed-mask binding mismatch for {key}: "
                f"expected {value!r}, found {binding.get(key)!r}"
            )
    if source_itemsize <= 0:
        raise ValueError("source_itemsize must be positive")
    records = payload.get("files")
    if not isinstance(records, list) or len(records) != len(corpus_paths):
        raise RuntimeError("precomputed-mask file count does not match corpus")
    bound_files = binding.get("mask_files")
    if not isinstance(bound_files, list) or len(bound_files) != len(records):
        raise RuntimeError("precomputed-mask binding has no ordered file digests")

    mask_paths: list[str] = []
    total_instances = 0
    for index, (source_path, source_id, record, bound_file) in enumerate(
        zip(corpus_paths, source_ids, records, bound_files)
    ):
        if not isinstance(record, Mapping) or not isinstance(bound_file, Mapping):
            raise RuntimeError(f"invalid precomputed-mask record {index}")
        source = Path(source_path)
        mask = Path(str(record.get("mask_path", "")))
        source_size = source.stat().st_size
        expected_mask_size = source_size // source_itemsize
        if (
            record.get("source_id") != source_id
            or int(record.get("source_size", -1)) != source_size
            or int(record.get("mask_size", -1)) != expected_mask_size
        ):
            raise RuntimeError(f"precomputed mask {index} is not aligned with its source")
        expected_bound_file = {
            "source_id": source_id,
            "mask_size": expected_mask_size,
            "mask_sha256": record.get("mask_sha256"),
        }
        if dict(bound_file) != expected_bound_file:
            raise RuntimeError(f"precomputed mask {index} is not bound to its recorded digest")
        if not mask.is_file() or mask.stat().st_size != expected_mask_size:
            raise RuntimeError(f"precomputed mask is missing or has changed: {mask}")
        mask_paths.append(str(mask))
        total_instances += expected_mask_size // sequence_length

    expected_selected = total_instances * round((sequence_length - 1) * keep_fraction)
    if (
        binding.get("total_instances") != total_instances
        or binding.get("selected_tokens") != expected_selected
    ):
        raise RuntimeError("precomputed-mask aggregate counts do not match the corpus")

    return mask_paths, dict(binding)
