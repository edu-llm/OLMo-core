"""Resolve ``pretrain/frontload-cl-10b`` and group shard paths by source folder.

Reuses the same refusal / dtype / byte-order checks as ``.edullm/train_on_corpus.py``.
Paths look like::

    s3://edullm-data/pretrain/frontload-cl-10b/v1/tokens/fineweb-edu-main/train-00000.u32le.bin
"""

from __future__ import annotations

import enum
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig


class Stage(enum.IntEnum):
    THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT = 64
    THE_ROLE_MAY_NOT_READ_THE_CORPUS = 65
    THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS = 66
    THE_READER_FAILED_IN_SOME_OTHER_WAY = 67
    THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP = 68
    THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER = 69
    THE_CONFIG_WOULD_NOT_BUILD = 70
    THE_TRAINING_ENVIRONMENT_WOULD_NOT_START = 71
    TRAINING_ITSELF_FAILED = 72


class Refusal(SystemExit):
    def __init__(self, stage: Stage, explanation: str) -> None:
        super().__init__(explanation)
        self.stage = stage
        self.explanation = explanation


@dataclass
class Corpus:
    dataset_id: str
    version: str
    paths_by_source: Dict[str, List[str]]
    dtype: "NumpyDatasetDType"
    tokenizer: "TokenizerConfig"
    rows: Optional[int]


def source_name_from_path(path: str) -> str:
    """Extract ``fineweb-edu-main`` from ``.../tokens/fineweb-edu-main/train-00000.u32le.bin``."""
    parsed = urlparse(path)
    parts = [p for p in (parsed.path or path).split("/") if p]
    try:
        tokens_idx = parts.index("tokens")
    except ValueError as exc:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"path is not under a tokens/ group: {path}",
        ) from exc
    if tokens_idx + 1 >= len(parts):
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"path has no source folder under tokens/: {path}",
        )
    return parts[tokens_idx + 1]


def group_paths_by_source(paths: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        grouped[source_name_from_path(path)].append(path)
    return dict(grouped)


def _looks_like(exc: BaseException, *words: str) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(word in text for word in words):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def read_failure(exc: BaseException) -> Stage:
    if _looks_like(exc, "accessdenied", "403", "forbidden", "not authorized"):
        return Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS
    if _looks_like(exc, "nosuchkey", "nosuchbucket", "404", "not found", "no such"):
        return Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS
    return Stage.THE_READER_FAILED_IN_SOME_OTHER_WAY


def _build_tokenizer(tokenizer_id: str) -> "TokenizerConfig":
    from olmo_core.data import TokenizerConfig

    factories = {
        "tokenizer/dolma2-bpe": TokenizerConfig.dolma2,
    }
    try:
        return factories[tokenizer_id]()
    except KeyError as exc:
        known = ", ".join(sorted(factories)) or "none"
        raise Refusal(
            Stage.THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER,
            f"no OLMo-core config for {tokenizer_id}; known: {known}",
        ) from exc


def corpus_from_manifest(read, *, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    from olmo_core.data import NumpyDatasetDType

    if not read.paths:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} resolved to no trainable shards",
        )
    if read.dtype is None:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} declares no dtype",
        )
    if read.header_bytes:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} declares {read.header_bytes} header bytes; "
            "OLMo-core memmaps from offset zero",
        )
    if read.byte_order is not None and read.byte_order != sys.byteorder:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} is {read.byte_order}-endian; host is {sys.byteorder}",
        )

    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths_by_source=group_paths_by_source(list(read.paths)),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=_build_tokenizer(tokenizer_id),
        rows=read.rows,
    )


def resolve_corpus(*, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    if version in ("", "latest"):
        try:
            resolved = resolve_latest(dataset_id, s3=s3)
        except Refusal:
            raise
        except BaseException as exc:
            raise Refusal(read_failure(exc), f"{type(exc).__name__}: {exc}") from exc
        if resolved is None:
            raise Refusal(
                Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS,
                f"no published version of {dataset_id}",
            )
        version = resolved

    try:
        read = dataset_paths(dataset_id, version, s3=s3)
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(
            read_failure(exc),
            f"reading {dataset_id}/{version}: {type(exc).__name__}: {exc}",
        ) from exc

    return corpus_from_manifest(
        read, dataset_id=dataset_id, version=version, tokenizer_id=tokenizer_id
    )


def require_platform_env(opts) -> None:
    missing = [
        name
        for name, value in (
            ("EDULLM_DATASET_ID", opts.dataset_id),
            ("EDULLM_DATASET_VERSION", opts.dataset_version),
            ("EDULLM_DATASET_TOKENIZER", opts.dataset_tokenizer),
            ("EDULLM_CHECKPOINT_DIR", opts.save_folder),
        )
        if not value
    ]
    if missing:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "unset: "
            + ", ".join(missing)
            + ". Pick the frontload-cl corpus on the form (not dataset_release: none).",
        )


def default_run_name() -> str:
    return os.environ.get("EDULLM_RUN_ID", "local")
