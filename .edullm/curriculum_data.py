"""Local-scratch input resolution for `curriculum-new`.

Rewritten from scratch: no S3, no boto3, no `edullm_data`, no AWS credential
of any kind (plan section 3a -- FarmShare-only, no AWS/RunPod/eduLLM-platform
coupling of any kind). Every input is read directly from job-local scratch,
staged there by the corpus rebuild (plan section 4/5) before training starts.

The corpus rebuild produces, per difficulty metric, a re-chunked parent
whose chunk index already equals difficulty rank (plan section 5), so the
"order" for a curriculum pacing is the identity permutation and does not
need to be read from a separate file at all -- `resolve_local_inputs`
returns `order=None` for every pacing including control, and
`curriculum_entrypoint` builds `np.arange(n)` directly when a curriculum
pacing needs a `ranked` array. What IS read from disk is a small JSON
manifest (written once by the rebuild) naming the parent's shard paths,
dtype, and a sha256 over its content, so a run can still pin and verify
exactly which materialized corpus it trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class PublishedInputError(RuntimeError):
    """A local input is absent, malformed, or does not match its pinned manifest."""


@dataclass(frozen=True)
class ResolvedInput:
    """A locally-staged parent corpus (one per difficulty metric).

    `manifest_sha256` is computed over the manifest JSON itself (not the
    multi-gigabyte shard contents) -- the manifest already commits to each
    shard's own size and a per-shard content hash under `shard_hashes`,
    so this is enough to detect any accidental manifest edit or corpus swap
    without hashing tens of gigabytes on every run.
    """

    metric: str
    manifest_path: str
    manifest_sha256: str
    paths: tuple[str, ...]
    numpy_dtype: str
    header_bytes: int
    chunk_count: int

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "paths": list(self.paths),
            "numpy_dtype": self.numpy_dtype,
            "header_bytes": self.header_bytes,
            "chunk_count": self.chunk_count,
        }


def resolve_local_parent(
    *,
    manifest_path: str | Path,
    metric: str,
) -> ResolvedInput:
    """Read and validate a corpus-rebuild manifest from job-local scratch.

    Expected manifest shape (written once by the corpus rebuild, plan
    section 4/5), for example:

        {
          "metric": "mtld",
          "dtype": "<u4",
          "header_bytes": 0,
          "sequence_length": 2048,
          "chunk_count": 4710400,
          "shards": [
            {"path": "mtld/train-00000.u32le.bin", "sha256": "..."},
            ...
          ]
        }

    `metric` for `control` may be any of the rebuilt metrics -- control
    trains on the identical chunk set as whichever metric it is being
    compared against in a given run, so the caller passes the metric of
    the curriculum arm under test (or any metric, if it is a standalone
    control run establishing the noise floor).
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise PublishedInputError(f"missing corpus manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("metric")) != metric:
        raise PublishedInputError(
            f"manifest at {path} is for metric {payload.get('metric')!r}, expected {metric!r}"
        )
    dtype = str(payload.get("dtype"))
    if dtype != "<u4":
        raise PublishedInputError(f"parent must be explicit little-endian uint32, got {dtype!r}")
    header_bytes = int(payload.get("header_bytes") or 0)
    if header_bytes != 0:
        raise PublishedInputError(f"parent must be headerless, got {header_bytes} bytes")
    shards = payload.get("shards") or []
    if not shards:
        raise PublishedInputError(f"manifest at {path} names no shards")
    base_dir = path.parent
    resolved_paths: list[str] = []
    for shard in shards:
        shard_path = base_dir / str(shard["path"])
        if not shard_path.is_file():
            raise PublishedInputError(f"missing staged shard: {shard_path}")
        expected_hash = shard.get("sha256")
        if expected_hash:
            actual_hash = _sha256_file(shard_path)
            if actual_hash != expected_hash:
                raise PublishedInputError(
                    f"shard {shard_path} content hash {actual_hash} != manifest {expected_hash}"
                )
        resolved_paths.append(str(shard_path))
    manifest_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ResolvedInput(
        metric=metric,
        manifest_path=str(path),
        manifest_sha256=manifest_hash,
        paths=tuple(resolved_paths),
        numpy_dtype=dtype,
        header_bytes=header_bytes,
        chunk_count=int(payload["chunk_count"]),
    )


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_order(paths: tuple[Path, ...], dtype: str) -> np.ndarray:
    """Read a persisted order file, for the rare case one exists on disk.

    Not used by the identity-permutation path above, but kept for cases
    where an order is genuinely non-trivial (e.g. a hand-built calibration
    order) and needs loading from `.u32le.bin`-style shards.
    """
    parts = [np.memmap(path, mode="r", dtype=dtype) for path in paths]
    if not parts:
        raise PublishedInputError("curriculum order resolved to no objects")
    return np.asarray(
        np.concatenate(parts) if len(parts) > 1 else parts[0],
        dtype=np.int64,
    )
