"""Pack the JSONL corpus into publishable `.u32le.bin` token shards.

Both arms read the SAME shards. There are no label masks here any more: the split
arm recomputes the fact-block boundary at load time by finding the separator token
run in the stream (`train_platform.py:DerivedMaskTrainModule`). Two reasons, and the
second is the better one:

  * `weights-sidecar/v1` is not a registered profile, so a mask array has no legal
    home in a published dataset.
  * a derived mask cannot fall out of alignment with the tokens it describes. A
    shipped one can, silently, and that is the defect `mask_alignment_test.py`
    exists to catch.

Layout, which is fixed before a byte is written because `entry.path` is hashed into
`manifest_sha256` and re-pathing later means re-copying every payload byte:

    tokens/<corpus>/<split>-<NNNNN>.u32le.bin

`<corpus>` is one of metamath, prf2, enigma, mizar, thproofs, isabelle, so a trainer
can later measure one proof style alone. The extension is self-describing and the
bytes are raw little-endian uint32 from byte 0 — **never `.npy`**. OLMo-core memmaps
from byte 0 and derives the token count from the raw file size, so a real `.npy`
header corrupts both the tokens and the count, silently. uint32 is required: the
vocab is 151,936 and uint16 would wrap.

Over-length examples are DROPPED, not truncated. A truncated proof has no ending, and
training on thousands of them teaches a model to never stop. `--suggest` reports what
each sequence length would cost before you commit to one.

Packing: every document remains intact and EOS-terminated, while several documents may
share one fixed-width sequence. The train module finds each document's separator and
OLMo-core derives intra-document attention boundaries from EOS, so proofs neither attend
to nor alter the masks of their packed neighbours.

The encoder cache and each corpus's `*.done.json` marker are persistent. Re-running the
same command rehashes and skips completed groups/shards. Size is never treated as proof
of identity: a wrong-sized or wrong-digest shard, an unsealed legacy marker, or a
mismatched build identity is preserved and refused, never deleted or overwritten.

Usage:
    python src/scripts/train/p3_math_split/tokenize_corpus.py \
        --corpus-contract-root <transaction-root> \
        --tokenizer <local-approved-tokenizer> --out <fresh-output> \
        --sequence-length 16384 --pack --jobs 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

TOKENS_DTYPE = np.uint32  # vocab is 151,936, so uint16 is not an option
TOKENS_STORAGE_DTYPE = np.dtype("<u4")
TOKENS_DTYPE_NAME = "uint32"
TOKENS_BYTE_ORDER = "little"
PACKED_GROUP_SCHEMA_VERSION = "p3-packed-group-v3"
PACKED_CORPUS_SCHEMA_VERSION = "p3-packed-corpus-v3"
ENCODING_CACHE_SCHEMA_VERSION = "p3-encoding-cache-v3"
ENCODING_CACHE_PROGRESS_SCHEMA_VERSION = "p3-encoding-cache-progress-v3"
CORPUS_BINDING_SCHEMA_VERSION = "p3-tokenizer-corpus-binding-v1"
SEALED_CORPUS_MANIFEST_SCHEMA_VERSION = "p3-sealed-corpus-manifest-v1"
TOKENIZER_SEAL_SCHEMA_VERSION = "p3-tokenizer-four-part-seal-v1"
TOKENIZE_CORPUS_CODE_VERSION = "tokenize-corpus-v4"
PACKED_ALGORITHM_VERSION = "largest-fit-decreasing-v1"
UNPACKED_ALGORITHM_VERSION = "single-document-v1"
CORPUS_MANIFEST_SCHEMA_VERSION = "corpus-generation-manifest/v2"
CORPUS_CURRENT_SCHEMA_VERSION = "corpus-generation-current/v2"
CORPUS_TRANSACTION_V2_PRODUCER_SOURCE_SHA256 = (
    "a2ae99341ab9f1868ca6d407b29310e7b24a63efa8439d1e7372366add2b8006"
)
CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT = {
    "api_version": 2,
    "accounting_scheme": "physical-occurrence-routes/v2",
    "current_schema": CORPUS_CURRENT_SCHEMA_VERSION,
    "logical_root_schema": "logical-generation-root/v1",
    "manifest_keys": sorted(
        (
            "accounting",
            "api_version",
            "directories",
            "generation_id",
            "logical_root_sha256",
            "manifest_root_sha256",
            "outputs",
            "physical_generation_id_policy",
            "plan_root_sha256",
            "requested_siblings",
            "routes",
            "schema_version",
            "source_generation_id",
        )
    ),
    "manifest_schema": CORPUS_MANIFEST_SCHEMA_VERSION,
    "physical_generation_id_policy": "caller-supplied-immutable-id/v1",
    "plan_schema": "corpus-generation-plan/v2",
    "routes_schema": "physical-occurrence-routes/v2",
    "transaction_state_schema": "generation-transaction-state/v1",
    "validator_version": 1,
}
CORPUS_TRANSACTION_V2_SEMANTIC_SHA256 = hashlib.sha256(
    json.dumps(
        CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
P3_SOURCE_SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}
FIXED_QWEN_TOKENIZER_DATASET_ID = "tokenizer/qwen25-vendored"
FIXED_QWEN_TOKENIZER_VERSION = "v1"
FIXED_QWEN_TOKENIZER_SEAL = {
    "schema_version": TOKENIZER_SEAL_SCHEMA_VERSION,
    "tokenizer_artifact_id": FIXED_QWEN_TOKENIZER_DATASET_ID,
    "tokenizer_artifact_version": FIXED_QWEN_TOKENIZER_VERSION,
    "tokenizer_file_sha256": {
        "tokenizer.json": "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
        "tokenizer_config.json": (
            "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
        ),
    },
    "tokenizer_composite_sha256": (
        "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
    ),
    "tokenizers_version": "0.22.2",
    "tokenizer_eos_token_id": 151_643,
    "tokenizer_pad_token_id": 151_643,
    "separator": "\n---\nGOAL ",
    "separator_ids": [10952, 15513, 969],
}
STAGED_TOKEN_MANIFEST_FIELDS = {
    "byte_order",
    "code_version",
    "corpus_generation",
    "cross_split_binding_schema_version",
    "dropped_over_length",
    "eos_token_id",
    "groups",
    "instances",
    "manifest_sha256",
    "packed",
    "packing_config",
    "pad_token_id",
    "real_tokens",
    "resumed_groups",
    "resumed_shards",
    "schema_version",
    "separator",
    "separator_ids",
    "separator_search",
    "sequence_length",
    "source_family_inventory",
    "source_jsonl_sha256",
    "split",
    "tokenizer",
    "tokenizer_composite_sha256",
    "tokenizer_seal",
    "tokens_dtype",
    "tokens_straddling_boundary",
}
STAGED_CORPUS_GENERATION_FIELDS = {
    "current_sha256",
    "generation_id",
    "logical_root_sha256",
    "manifest_file_sha256",
    "manifest_root_sha256",
    "producer_source_sha256",
    "schema_version",
    "semantic_contract_sha256",
}
STAGED_SEALED_CORPUS_GENERATION_FIELDS = {
    "logical_root_sha256",
    "manifest_root_sha256",
    "schema_version",
    "sealed_corpus_manifest",
}
STAGED_PACKING_FIELDS = {
    "algorithm",
    "byte_order",
    "eos_token_id",
    "packed",
    "pad_token_id",
    "separator",
    "separator_ids",
    "separator_search",
    "sequence_length",
    "shard_tokens",
    "split",
    "tokens_dtype",
}
STAGED_GROUP_FIELDS = {
    "build",
    "cache_fingerprint",
    "cache_root_sha256",
    "code_version",
    "completion_sha256",
    "corpus_generation",
    "cross_split_binding_schema_version",
    "documents",
    "dropped_over_length",
    "eos_token_id",
    "fingerprint",
    "instances",
    "name",
    "padding_fraction",
    "pad_token_id",
    "real_tokens",
    "resumed_group",
    "resumed_shards",
    "schema_version",
    "separator_ids",
    "shards",
    "source_documents",
    "straddling",
}
STAGED_BUILD_FIELDS = {
    "code_version",
    "corpus_generation",
    "encoding_cache_fingerprint",
    "packing",
    "schema_version",
    "source_jsonl",
    "tokenizer",
}
STAGED_SOURCE_FIELDS = {"family", "name", "schema", "sha256"}
STAGED_SHARD_FIELDS = {
    "byte_order",
    "bytes",
    "instances",
    "path",
    "sha256",
    "tokens",
    "tokens_dtype",
}
CROSS_SPLIT_BINDING_SCHEMA_VERSION = "p3-token-cross-split-binding-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# What the corpus builder writes between the fact block and the goal.
SEPARATOR = "\n---\nGOAL "
# What the split arm actually SEARCHES for, and it is deliberately not the string
# above. BPE does not respect the boundary: the trailing space merges rightward into
# the goal's first word (` |-`, ` ![`, ` lemma`) in 98.4% of documents, and the
# leading newline merges leftward into the fact block's last characters (` )\n`,
# `"\n`) in 88.5%. Encoding the full separator gives [198, 10952, 15513, 969, 220],
# a run that survives in 777 of 258,316 documents -- 0.30%, and 0% in four of the six
# shards. The split arm would have found no boundary and supervised everything,
# silently becoming a second dense arm.
#
# The three-token core `---\nGOAL` -> [10952, 15513, 969] survives in 258,316 of
# 258,316, never appears twice, and the token immediately after it always begins at
# or past mask_end -- so supervision starts exactly at the goal.
SEPARATOR_SEARCH = "---\nGOAL"


class TokenManifestCommitUncertainError(RuntimeError):
    """A final token manifest was replaced but durability was not confirmed."""

    def __init__(
        self,
        committed_paths: Sequence[Path],
        cause: BaseException,
    ):
        self.committed_paths = tuple(committed_paths)
        self.cause = cause
        super().__init__(
            "token manifest commit state is uncertain after replacing "
            f"{[str(path) for path in self.committed_paths]}: "
            f"{type(cause).__name__}: {cause}"
        )


def encode_rows_batched(tok, rows, eos_id: int, batch_size: int = 256):
    """Encode rows through the fast tokenizer's parallel batch path.

    Calling a fast tokenizer once per document serializes the Python/Rust boundary
    and used one of twelve local CPU cores. A list input routes through
    `encode_batch`/Rayon inside the tokenizers library, while processing in bounded
    chunks keeps peak memory predictable.

    Returns uint32 arrays (each EOS-terminated) and the total count of tokens whose
    offset crosses `mask_end`.
    """
    encoded = []
    total_straddling = 0
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo : lo + batch_size]
        batch = tok(
            [row["text"] for row in chunk],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        for row, ids, offsets in zip(chunk, batch["input_ids"], batch["offset_mapping"]):
            total_straddling += sum(1 for start, end in offsets if start < row["mask_end"] < end)
            arr = np.empty(len(ids) + 1, dtype=TOKENS_DTYPE)
            arr[:-1] = ids
            arr[-1] = eos_id
            encoded.append(arr)
    return encoded, total_straddling


def pack_indices_by_length(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Pack every document exactly once in O(n log capacity), deterministically.

    This is largest-fit decreasing implemented with a Fenwick tree over integer
    token lengths. The old implementation repeatedly scanned and popped from a
    Python list, which is O(n²) and spent tens of minutes packing one corpus.
    Token lengths are bounded by `capacity` (16,384 here), so a count tree gives
    the largest remaining document that fits in O(log capacity).
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    buckets: list[list[int]] = [[] for _ in range(capacity + 1)]
    tree = [0] * (capacity + 1)

    def add(pos: int, delta: int) -> None:
        while pos <= capacity:
            tree[pos] += delta
            pos += pos & -pos

    def prefix(pos: int) -> int:
        out = 0
        while pos:
            out += tree[pos]
            pos -= pos & -pos
        return out

    def kth(rank: int) -> int:
        """Smallest length whose cumulative count reaches one-based `rank`."""
        idx = 0
        bit = 1 << (capacity.bit_length() - 1)
        while bit:
            nxt = idx + bit
            if nxt <= capacity and tree[nxt] < rank:
                idx = nxt
                rank -= tree[nxt]
            bit >>= 1
        return idx + 1

    for i, raw in enumerate(lengths):
        length = int(raw)
        if length < 1 or length > capacity:
            raise ValueError(f"document {i} has length {length}, capacity is {capacity}")
        buckets[length].append(i)
        add(length, 1)

    remaining = len(lengths)
    packed: list[list[int]] = []
    while remaining:
        free = capacity
        row = []
        while free:
            count = prefix(free)
            if not count:
                break
            length = kth(count)  # largest populated length <= free
            row.append(buckets[length].pop())
            add(length, -1)
            remaining -= 1
            free -= length
        packed.append(row)
    return packed


def write_shard_resumable(
    path: str | Path,
    documents: Sequence[np.ndarray],
    *,
    sequence_length: int,
    pad_id: int,
    write_batch: int = 128,
) -> dict:
    """Atomically write one shard, or validate and skip one already complete.

    Existing bytes are accepted only when both size and a deterministic SHA-256 of
    the pending packed rows match. The caller can move or inspect a rejected file;
    this function never deletes or overwrites partial generated data.
    """
    path = Path(path)
    if sequence_length < 1 or write_batch < 1:
        raise ValueError("sequence_length and write_batch must be positive")
    tokens = len(documents) * sequence_length
    expected = tokens * TOKENS_STORAGE_DTYPE.itemsize
    metadata = {
        "tokens": tokens,
        "bytes": expected,
        "tokens_dtype": TOKENS_DTYPE_NAME,
        "byte_order": TOKENS_BYTE_ORDER,
    }

    def buffers():
        for lo in range(0, len(documents), write_batch):
            block = documents[lo : lo + write_batch]
            buf = np.full(
                (len(block), sequence_length),
                pad_id,
                dtype=TOKENS_STORAGE_DTYPE,
            )
            for j, ids in enumerate(block):
                ids = np.asarray(ids)
                if ids.ndim != 1:
                    raise ValueError(f"document {lo + j} is not one-dimensional")
                if len(ids) > sequence_length:
                    raise ValueError(
                        f"document {lo + j} has {len(ids):,} tokens, exceeding "
                        f"sequence length {sequence_length:,}"
                    )
                buf[j, : len(ids)] = ids
            yield buf

    if path.exists():
        actual = path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"{path} exists with {actual:,} bytes, expected {expected:,}; "
                "the partial file was preserved. Move it aside or resume from a clean "
                "output prefix after inspecting it."
            )
        expected_hasher = hashlib.sha256()
        for buf in buffers():
            expected_hasher.update(memoryview(buf).cast("B"))
        expected_sha256 = expected_hasher.hexdigest()
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{path} has SHA-256 {actual_sha256}, expected {expected_sha256} "
                "for the pending packed rows; the same-size stale file was preserved. "
                "Move it aside or use a fresh output prefix."
            )
        return {
            **metadata,
            "sha256": actual_sha256,
            "resumed": True,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    hasher = hashlib.sha256()
    with open(partial, "xb") as fh:
        for buf in buffers():
            raw = memoryview(buf).cast("B")
            hasher.update(raw)
            fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    # Any error above or during replacement naturally preserves `.partial-*`.
    os.replace(partial, path)
    fsync_directory(path.parent)
    return {
        **metadata,
        "sha256": hasher.hexdigest(),
        "resumed": False,
    }


def fsync_directory(path: str | Path) -> None:
    """Make a preceding atomic rename durable in its containing directory."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(Path(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Fsync a temporary control file, atomically replace, then fsync its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    with open(partial, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, path)
    fsync_directory(path.parent)


def stage_json_for_atomic_replace(
    path: str | Path,
    payload: dict,
    *,
    final_path: str | Path | None = None,
    _fault: Callable[[str, Path, Path], None] | None = None,
) -> Path:
    """Write and fsync a fixed pending manifest without making it final."""
    path = Path(path)
    final_path = Path(final_path) if final_path is not None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"stale staged token manifest exists: {path}; preserved")
    encoded = json.dumps(payload, indent=2).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        if _fault is not None:
            _fault("stage_write_before", path, final_path)
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        if _fault is not None:
            _fault("stage_write_after", path, final_path)
        fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        fsync_directory(path.parent)
        raise
    return path


def cache_root_sha256(payload: Mapping) -> str:
    """Seal one cache marker, including every payload and chunk digest."""
    return fingerprint_dict(
        {key: value for key, value in payload.items() if key != "cache_root_sha256"}
    )


def _cache_payload(
    path: Path,
    *,
    relative_path: str,
    dtype: str,
    documents: int,
    tokens: int,
) -> dict:
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "dtype": dtype,
        "byte_order": TOKENS_BYTE_ORDER,
        "documents": {"start": 0, "end": documents},
        "tokens": {"start": 0, "end": tokens},
    }


def _cache_marker(
    *,
    status: str,
    build: dict,
    fingerprint: str,
    documents: int,
    tokens: int,
    straddling: int,
    chunks: list[dict],
    token_path: Path,
    offset_path: Path,
) -> dict:
    if status == "complete":
        schema = ENCODING_CACHE_SCHEMA_VERSION
        token_name = "tokens.u32le.bin"
        offset_name = "offsets.u64le.bin"
    elif status == "partial":
        schema = ENCODING_CACHE_PROGRESS_SCHEMA_VERSION
        token_name = "tokens.u32le.bin.partial"
        offset_name = "offsets.u64le.bin.partial"
        else:
        raise ValueError(f"unknown cache status {status!r}")
    payload = {
        "schema_version": schema,
        "code_version": TOKENIZE_CORPUS_CODE_VERSION,
        "status": status,
        "fingerprint": fingerprint,
        "build_fingerprint": fingerprint,
        "build": build,
        "source_jsonl": build["source_jsonl"],
        "documents": documents,
        "tokens": tokens,
        "straddling": straddling,
        "chunks": chunks,
        "payloads": {
            "tokens": _cache_payload(
                token_path,
                relative_path=token_name,
                dtype=TOKENS_DTYPE_NAME,
                documents=documents,
                tokens=tokens,
            ),
            "offsets": _cache_payload(
                offset_path,
                relative_path=offset_name,
                dtype="uint64",
                documents=documents,
                tokens=tokens,
            ),
        },
    }
    payload["cache_root_sha256"] = cache_root_sha256(payload)
    return payload


def _read_json_object(path: Path, context: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return value


def _require_sha256(value, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{context} must be a lowercase SHA-256")
    return value


def _cache_expected_files(status: str) -> set[str]:
    if status == "complete":
        return {"cache.json", "tokens.u32le.bin", "offsets.u64le.bin"}
    return {
        "progress.json",
        "tokens.u32le.bin.partial",
        "offsets.u64le.bin.partial",
    }


def _require_exact_cache_inventory(cache_dir: Path, status: str) -> None:
    actual = set()
    for path in cache_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{cache_dir} has an unknown cache entry {path.name!r}; preserved")
        actual.add(path.name)
    expected = _cache_expected_files(status)
    if actual != expected:
        raise RuntimeError(
            f"{cache_dir} cache inventory is not exact; "
            f"unexpected={sorted(actual - expected)}, missing={sorted(expected - actual)}; "
            "preserved"
        )


def _hash_file_range(path: Path, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise RuntimeError(f"{path} has an invalid sealed byte range")
    digest = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"{path} ended inside a sealed byte range")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_source_chunk_sequence(
    source: Path, chunks: Sequence[Mapping], documents: int
) -> None:
    chunk_index = 0
    document_index = 0
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            if document_index < documents:
                if chunk_index >= len(chunks):
                    raise RuntimeError("cache source sequence has rows outside its chunk inventory")
                digest.update(raw)
                document_index += 1
                expected_end = int(chunks[chunk_index]["documents"]["end"])
                if document_index == expected_end:
                    expected = _require_sha256(
                        chunks[chunk_index].get("source_rows_sha256"),
                        f"cache source chunk {chunk_index}",
                    )
                    if digest.hexdigest() != expected:
                        raise RuntimeError(
                            f"cache source sequence SHA-256 mismatch in chunk {chunk_index}"
                        )
                    chunk_index += 1
                    digest = hashlib.sha256()
    if document_index != documents or chunk_index != len(chunks):
        raise RuntimeError("cache source sequence does not cover exact document ranges")


def _validate_cache_marker(
    cache_dir: Path,
    marker: dict,
    *,
    status: str,
    fingerprint: str,
    build: dict,
    source: Path,
) -> tuple[Path, Path]:
    expected_schema = (
        ENCODING_CACHE_SCHEMA_VERSION
        if status == "complete"
        else ENCODING_CACHE_PROGRESS_SCHEMA_VERSION
    )
    if marker.get("schema_version") != expected_schema:
        raise RuntimeError(
            f"{cache_dir} has a legacy cache marker without payload digests; preserved"
        )
    if marker.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION:
        raise RuntimeError(f"{cache_dir} cache code version differs; preserved")
    if marker.get("status") != status:
        raise RuntimeError(f"{cache_dir} cache status differs from its marker name; preserved")
    if marker.get("fingerprint") != marker.get("build_fingerprint"):
        raise RuntimeError(f"{cache_dir} cache fingerprint fields disagree; preserved")
    if marker.get("build_fingerprint") != fingerprint:
        raise RuntimeError(
            f"{cache_dir} contains a cache for another fingerprint; preserved. "
            "Use its fingerprinted directory or choose a fresh one"
        )
    if marker.get("build") != build or fingerprint != fingerprint_dict(build):
        raise RuntimeError(f"{cache_dir} cache build fingerprint is invalid; preserved")
    if marker.get("source_jsonl") != build.get("source_jsonl"):
        raise RuntimeError(f"{cache_dir} cache source identity differs; preserved")
    source_identity = build.get("source_jsonl")
    if not isinstance(source_identity, dict):
        raise TypeError("encoding cache build lacks source JSONL identity")
    if source_identity.get("name") != source.name or _require_sha256(
        source_identity.get("sha256"), "source JSONL digest"
    ) != file_sha256(source):
        raise RuntimeError(f"{source} source JSONL SHA-256 differs from cache build")
    if marker.get("cache_root_sha256") != cache_root_sha256(marker):
        raise RuntimeError(f"{cache_dir} cache root digest is invalid; preserved")

    _require_exact_cache_inventory(cache_dir, status)
    payloads = marker.get("payloads")
    chunks = marker.get("chunks")
    if not isinstance(payloads, dict) or set(payloads) != {"tokens", "offsets"}:
        raise RuntimeError(f"{cache_dir} cache payload inventory is invalid; preserved")
    if not isinstance(chunks, list):
        raise TypeError(f"{cache_dir} cache chunk inventory is invalid; preserved")
    documents = int(marker.get("documents", -1))
    tokens_count = int(marker.get("tokens", -1))
    if documents < 0 or tokens_count < 0:
        raise RuntimeError(f"{cache_dir} cache ranges are invalid; preserved")

    expected_names = (
        ("tokens.u32le.bin", "offsets.u64le.bin")
        if status == "complete"
        else ("tokens.u32le.bin.partial", "offsets.u64le.bin.partial")
    )
    paths = []
    for name, expected_name, dtype in zip(
        ("tokens", "offsets"),
        expected_names,
        (TOKENS_DTYPE_NAME, "uint64"),
    ):
        payload = payloads[name]
        if not isinstance(payload, dict) or payload.get("path") != expected_name:
            raise RuntimeError(f"{cache_dir} cache {name} path is stale; preserved")
        path = cache_dir / expected_name
        expected_bytes = int(payload.get("bytes", -1))
        if (
            payload.get("dtype") != dtype
            or payload.get("byte_order") != TOKENS_BYTE_ORDER
            or payload.get("documents") != {"start": 0, "end": documents}
            or payload.get("tokens") != {"start": 0, "end": tokens_count}
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
        ):
            raise RuntimeError(f"{cache_dir} cache {name} metadata differs; preserved")
        expected_digest = _require_sha256(payload.get("sha256"), f"cache {name} digest")
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"{path} cache payload SHA-256 {actual_digest} differs from "
                f"{expected_digest}; preserved"
            )
        paths.append(path)
    token_path, offset_path = paths

    next_document = next_token = chunk_straddling = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or chunk.get("index") != index:
            raise RuntimeError(f"{cache_dir} cache chunk order is invalid; preserved")
        document_range = chunk.get("documents")
        token_range = chunk.get("tokens")
        if document_range != {
            "start": next_document,
            "end": int(document_range.get("end", -1)) if isinstance(document_range, dict) else -1,
        } or token_range != {
            "start": next_token,
            "end": int(token_range.get("end", -1)) if isinstance(token_range, dict) else -1,
        }:
            raise RuntimeError(f"{cache_dir} cache chunk ranges are not contiguous; preserved")
        document_end = int(document_range["end"])
        token_end = int(token_range["end"])
        if document_end <= next_document or token_end <= next_token:
            raise RuntimeError(f"{cache_dir} cache chunk ranges are empty; preserved")
        token_bytes = {"start": next_token * 4, "end": token_end * 4}
        offset_bytes = {
            "start": (next_document + 1) * 8,
            "end": (document_end + 1) * 8,
        }
        if chunk.get("token_bytes") != token_bytes or chunk.get("offset_bytes") != offset_bytes:
            raise RuntimeError(f"{cache_dir} cache chunk byte ranges are invalid; preserved")
        for path, byte_range, field in (
            (token_path, token_bytes, "tokens_sha256"),
            (offset_path, offset_bytes, "offsets_sha256"),
        ):
            expected_digest = _require_sha256(chunk.get(field), f"cache chunk {index} {field}")
            actual_digest = _hash_file_range(path, byte_range["start"], byte_range["end"])
            if actual_digest != expected_digest:
                raise RuntimeError(f"{path} cache chunk {index} SHA-256 differs; preserved")
        chunk_crossed = chunk.get("straddling")
        if (
            not isinstance(chunk_crossed, int)
            or isinstance(chunk_crossed, bool)
            or chunk_crossed < 0
        ):
            raise RuntimeError(f"{cache_dir} cache chunk straddling count is invalid")
        chunk_straddling += chunk_crossed
        next_document = document_end
        next_token = token_end
    if (next_document != documents or next_token != tokens_count) and (
        documents or tokens_count or chunks
    ):
        raise RuntimeError(f"{cache_dir} cache chunk ranges are incomplete; preserved")
    if chunk_straddling != int(marker.get("straddling", -1)):
        raise RuntimeError(f"{cache_dir} cache chunk straddling total disagrees; preserved")
    _validate_source_chunk_sequence(source, chunks, documents)

    offsets = np.memmap(offset_path, mode="r", dtype="<u8")
    if (
        len(offsets) != documents + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != tokens_count
        or np.any(offsets[1:] < offsets[:-1])
        or token_path.stat().st_size != tokens_count * TOKENS_STORAGE_DTYPE.itemsize
    ):
        raise RuntimeError(f"{cache_dir} cache offsets or token ranges disagree; preserved")
    return token_path, offset_path


def load_encoding_cache(
    cache_dir: str | Path,
    *,
    fingerprint: str,
    build: dict,
    source: str | Path,
) -> tuple[list[np.ndarray], dict] | None:
    """Rehash and return one exact completed cache, or ``None`` when absent."""
    cache_dir = Path(cache_dir)
    source = Path(source)
    marker_path = cache_dir / "cache.json"
    if not marker_path.exists():
        if (
            cache_dir.exists()
            and any(cache_dir.iterdir())
            and not (cache_dir / "progress.json").exists()
        ):
            raise RuntimeError(f"{cache_dir} contains orphan cache files; preserved")
        return None
    marker = _read_json_object(marker_path, "encoding cache completion marker")
    token_path, offset_path = _validate_cache_marker(
        cache_dir,
        marker,
        status="complete",
        fingerprint=fingerprint,
        build=build,
        source=source,
    )
    offsets = np.memmap(offset_path, mode="r", dtype="<u8")
    tokens = np.memmap(token_path, mode="r", dtype="<u4")
    documents = [tokens[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]
    return documents, marker


def _declared_group_inventory(group_dir: Path) -> set[str]:
    declared: set[str] = set()
    for done_path in sorted(group_dir.glob("*.done.json")):
        payload = _read_json_object(done_path, "packed group completion marker")
        if (
            payload.get("schema_version") != PACKED_GROUP_SCHEMA_VERSION
            or payload.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION
            or not isinstance(payload.get("build"), dict)
            or payload.get("fingerprint") != fingerprint_dict(payload["build"])
            or payload.get("completion_sha256") != group_completion_sha256(payload)
            or not isinstance(payload.get("shards"), list)
        ):
            raise RuntimeError(f"{done_path} is not a valid sealed completed group; preserved")
        declared.add(done_path.name)
        for shard in payload["shards"]:
            if not isinstance(shard, dict):
                raise TypeError(f"{done_path} has invalid shard inventory; preserved")
            relative = Path(str(shard.get("path", "")))
            if relative.parent.name != group_dir.name:
                raise RuntimeError(f"{done_path} has a stale shard path; preserved")
            declared.add(relative.name)
    return declared


def _packing_cross_split_contract(packing: Mapping) -> dict:
    contract = dict(packing)
    split = contract.pop("split", None)
    if split not in {"train", "val"}:
        raise RuntimeError(f"cross-split packing contract has invalid split {split!r}")
    return contract


def _group_cross_split_binding(payload: Mapping) -> dict:
    if payload.get("cross_split_binding_schema_version") != CROSS_SPLIT_BINDING_SCHEMA_VERSION:
        raise RuntimeError("packed group lacks the accepted cross-split binding schema")
    build = payload.get("build")
    if not isinstance(build, Mapping):
        raise TypeError("cross-split packed group has no sealed build identity")
    source = build.get("source_jsonl")
    packing = build.get("packing")
    if not isinstance(source, Mapping) or not isinstance(packing, Mapping):
        raise TypeError("cross-split packed group lacks source or packing identity")
    family = source.get("family")
    schema = source.get("schema")
    if not isinstance(family, str) or not isinstance(schema, str):
        raise TypeError("cross-split packed group lacks family/schema source identity")
    return {
        "schema_version": CROSS_SPLIT_BINDING_SCHEMA_VERSION,
        "declared_cross_split_schema": payload.get("cross_split_binding_schema_version"),
        "group_schema_version": payload.get("schema_version"),
        "code_version": payload.get("code_version"),
        "family": family,
        "source_schema": schema,
        "corpus_generation": build.get("corpus_generation"),
        "tokenizer_seal": build.get("tokenizer"),
        "packing_contract": _packing_cross_split_contract(packing),
    }


def _require_equal_cross_split_bindings(
    left: Mapping,
    right: Mapping,
    *,
    context: str,
) -> None:
    for field in (
        "declared_cross_split_schema",
        "group_schema_version",
        "code_version",
        "family",
        "source_schema",
        "corpus_generation",
        "tokenizer_seal",
        "packing_contract",
    ):
        if left.get(field) != right.get(field):
            label = field.replace("_", " ")
            raise RuntimeError(f"cross-split {label} binding mismatch for {context}")


def _require_group_done_cross_split_binding(group_dir: Path) -> None:
    done = {
        split: _read_json_object(
            group_dir / f"{split}.done.json",
            f"{split} packed group completion marker",
        )
        for split in ("train", "val")
        if (group_dir / f"{split}.done.json").exists()
    }
    if set(done) == {"train", "val"}:
        _require_equal_cross_split_bindings(
            _group_cross_split_binding(done["train"]),
            _group_cross_split_binding(done["val"]),
            context=group_dir.name,
        )


def require_exact_group_inventory(
    group_dir: str | Path,
    *,
    pending_names: Sequence[str] = (),
) -> None:
    """Reject partials, temps, orphans, unknown controls, and missing sealed files."""
    group_dir = Path(group_dir)
    if not group_dir.exists():
        return
    if not group_dir.is_dir() or group_dir.is_symlink():
        raise RuntimeError(f"{group_dir} is not a real packed-group directory")
    declared = _declared_group_inventory(group_dir)
    allowed = declared | set(pending_names)
    actual = set()
    for path in group_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"{group_dir} has unknown group inventory entry {path.name!r}; preserved"
            )
        actual.add(path.name)
    extras = sorted(actual - allowed)
    missing = sorted(declared - actual)
    if extras or missing:
        raise RuntimeError(
            f"{group_dir} group inventory is not exact; "
            f"unexpected={extras}, missing={missing}; partial/temp/orphan files were preserved"
        )
    _require_group_done_cross_split_binding(group_dir)


def load_completed_group(
    marker_path: str | Path,
    *,
    fingerprint: str,
    output_root: str | Path,
    pending_names: Sequence[str] = (),
) -> dict | None:
    """Rehash a sealed per-corpus completion marker for manifest assembly."""
    marker_path = Path(marker_path)
    if not marker_path.exists():
        return None
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    recorded_shards = payload.get("shards")
    if (
        payload.get("schema_version") != PACKED_GROUP_SCHEMA_VERSION
        or payload.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION
        or not isinstance(payload.get("build"), dict)
        or not isinstance(recorded_shards, list)
        or not recorded_shards
        or "completion_sha256" not in payload
        or any(
            not isinstance(shard, dict)
            or "sha256" not in shard
            or "tokens_dtype" not in shard
            or "byte_order" not in shard
            for shard in recorded_shards
        )
    ):
        raise RuntimeError(
            f"{marker_path} is a legacy completion marker without the v3 digest seal; "
            "rebuild into a fresh output prefix. Existing output was preserved"
        )
    if payload["completion_sha256"] != group_completion_sha256(payload):
        raise RuntimeError(
            f"{marker_path} completion seal does not match its shard manifest; "
            "existing output was preserved"
        )
    marker_fingerprint = payload.get("fingerprint")
    if marker_fingerprint != fingerprint_dict(payload["build"]):
        raise RuntimeError(
            f"{marker_path} fingerprint does not seal its recorded build identity; "
            "existing output was preserved"
        )
    if marker_fingerprint != fingerprint:
        raise RuntimeError(
            f"{marker_path} has a different fingerprint; existing output was preserved"
        )
    output_root = Path(output_root)
    try:
        group_relative = marker_path.parent.relative_to(output_root)
    except ValueError as error:
        raise RuntimeError(
            f"{marker_path} is outside output root {output_root}; existing output was preserved"
        ) from error
    split = marker_path.name.removesuffix(".done.json")
    shards = recorded_shards

    expected_names: set[str] = set()
    problems = []
    instances = 0
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict):
            problems.append(f"shard {ordinal} metadata is not an object")
            continue
        required = {
            "path",
            "instances",
            "tokens",
            "bytes",
            "sha256",
            "tokens_dtype",
            "byte_order",
        }
        missing_fields = required - shard.keys()
        if missing_fields:
            problems.append(f"shard {ordinal} lacks {sorted(missing_fields)}")
            continue
        relative = Path(str(shard["path"]))
        expected_name = f"{split}-{ordinal:05d}.u32le.bin"
        if (
            relative.is_absolute()
            or relative.parent != group_relative
            or relative.name != expected_name
            or relative.name in expected_names
        ):
            problems.append(f"shard {ordinal} has invalid path {shard['path']!r}")
            continue
        expected_names.add(relative.name)
        path = output_root / relative
        expected_tokens = int(shard["tokens"])
        expected_bytes = int(shard["bytes"])
        expected_digest = str(shard["sha256"])
        shard_instances = int(shard["instances"])
        instances += shard_instances
        if shard["tokens_dtype"] != TOKENS_DTYPE_NAME or shard["byte_order"] != TOKENS_BYTE_ORDER:
            problems.append(f"{path} has unsupported dtype/byte-order metadata")
        if expected_tokens < 0 or expected_bytes != expected_tokens * TOKENS_STORAGE_DTYPE.itemsize:
            problems.append(f"{path} token and byte counts disagree")
        if shard_instances < 0 or expected_tokens != shard_instances * int(
            payload.get("build", {}).get("packing", {}).get("sequence_length", -1)
        ):
            problems.append(f"{path} instance and token counts disagree")
        if len(expected_digest) != hashlib.sha256().digest_size * 2 or any(
            char not in "0123456789abcdef" for char in expected_digest
        ):
            problems.append(f"{path} has an invalid SHA-256 seal")
        if not path.exists():
            problems.append(f"{path} is missing")
            continue
        actual_bytes = path.stat().st_size
        actual_digest = file_sha256(path)
        if actual_bytes != expected_bytes:
            problems.append(f"{path} has {actual_bytes:,} bytes, expected {expected_bytes:,}")
        if actual_digest != expected_digest:
            problems.append(f"{path} has SHA-256 {actual_digest}, expected {expected_digest}")

    actual_names = {path.name for path in marker_path.parent.glob(f"{split}-*.u32le.bin")}
    extras = sorted(actual_names - expected_names)
    missing_names = sorted(expected_names - actual_names)
    if extras or missing_names:
        problems.append(f"unexpected shards {extras}; missing shards {missing_names}")
    if instances != int(payload.get("instances", -1)):
        problems.append(
            f"shard instances sum to {instances}, marker records {payload.get('instances')}"
        )
    if problems:
        raise RuntimeError(
            f"{marker_path} failed completed-group integrity: {'; '.join(problems)}; "
            "existing output was preserved"
        )
    require_exact_group_inventory(marker_path.parent, pending_names=pending_names)
    return payload


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


_TXN_OUTPUT_KEYS = {
    "bytes",
    "drop_types",
    "generation_id",
    "logical_sha256",
    "path",
    "role",
    "rows",
    "schema",
    "sha256",
    "sibling",
    "source_generation_id",
    "validator",
}
_TXN_ROUTES_KEYS = {
    "bytes",
    "path",
    "root_sha256",
    "rows",
    "schema_version",
    "sha256",
}
_TXN_ROUTE_KEYS = {
    "destination_path",
    "destination_row",
    "disposition",
    "drop_type",
    "occurrence_id",
    "plan_root_sha256",
    "raw_path",
    "raw_row",
    "raw_sha256",
    "sibling",
}
_TXN_DROP_KEYS = {
    "details",
    "drop_type",
    "generation_id",
    "occurrence_id",
    "plan_root_sha256",
    "raw_path",
    "raw_row",
    "raw_sha256",
    "schema_version",
    "sibling",
    "source_generation_id",
}
_TXN_STATE_KEYS = {
    "generation_id",
    "logical_root_sha256",
    "manifest_sha256",
    "schema_version",
    "state",
}
_TXN_ROLES = {"raw", "train", "eval", "heldout", "sidecar"}
_TXN_DROP_TYPE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_TXN_SCHEMA_RE = re.compile(r".+(?:/|-)v[1-9][0-9]*\Z")
_TXN_VALIDATOR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}/v[1-9][0-9]*\Z")


def _txn_canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )


def _txn_validate_regular_metadata(metadata: os.stat_result, label: str) -> os.stat_result:
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"symlink is forbidden: {label}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"special node is forbidden: {label}")
    if metadata.st_nlink != 1:
        raise RuntimeError(f"hard link is forbidden: {label}")
    return metadata


def _txn_require_regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"missing file: {label}") from error
    return _txn_validate_regular_metadata(metadata, label)


def _txn_require_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} is a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a directory")
    return metadata


def _txn_validate_generation_id(generation_id: object) -> str:
    if not isinstance(generation_id, str) or _GENERATION_ID_RE.fullmatch(generation_id) is None:
        raise RuntimeError(f"invalid corpus transaction generation ID: {generation_id!r}")
    return generation_id


def _txn_validate_schema(schema: object) -> str:
    if not isinstance(schema, str) or _TXN_SCHEMA_RE.fullmatch(schema) is None:
        raise RuntimeError(f"schema must have an explicit /vN or -vN version: {schema!r}")
    return schema


def _txn_validate_path(path: object) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise RuntimeError(f"output path must be a non-empty POSIX path: {path!r}")
    logical = PurePosixPath(path)
    if (
        logical.is_absolute()
        or str(logical) != path
        or any(part in {"", ".", ".."} for part in logical.parts)
        or path in {"MANIFEST.json", "ROUTES.jsonl", "CURRENT"}
        or logical.parts[0] in {".staging", "generations", "quarantine"}
    ):
        raise RuntimeError(f"output path is unsafe, reserved, or non-canonical: {path!r}")
    return path


def _txn_validator(descriptor: object) -> dict:
    if not isinstance(descriptor, dict):
        raise TypeError("validator descriptor must be an object")
    kind = descriptor.get("kind")
    if kind == "jsonl-object":
        expected = {
            "allow_empty",
            "kind",
            "require_generation_links",
            "required_fields",
            "schema_version",
            "validator_version",
        }
        if set(descriptor) != expected:
            raise RuntimeError("JSONL validator descriptor is not exact")
        if (
            type(descriptor["allow_empty"]) is not bool
            or type(descriptor["require_generation_links"]) is not bool
        ):
            raise RuntimeError("JSONL validator boolean fields are not booleans")
    elif kind == "json-object":
        expected = {
            "kind",
            "require_generation_links",
            "required_fields",
            "schema_version",
            "validator_version",
        }
        if set(descriptor) != expected:
            raise RuntimeError("JSON object validator descriptor is not exact")
        if type(descriptor["require_generation_links"]) is not bool:
            raise RuntimeError("JSON object validator boolean fields are not booleans")
    elif kind == "binary":
        expected = {
            "kind",
            "schema_version",
            "validator_id",
            "validator_version",
        }
        if (
            set(descriptor) != expected
            or not isinstance(descriptor["validator_id"], str)
            or _TXN_VALIDATOR_ID_RE.fullmatch(descriptor["validator_id"]) is None
        ):
            raise RuntimeError("binary validator descriptor is not exact")
    else:
        raise RuntimeError(f"unknown transaction validator kind: {kind!r}")
    if type(descriptor["validator_version"]) is not int or descriptor["validator_version"] != 1:
        raise RuntimeError("unsupported transaction validator version")
    _txn_validate_schema(descriptor["schema_version"])
    normalized = dict(descriptor)
    if kind in {"jsonl-object", "json-object"}:
        fields = descriptor["required_fields"]
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(name, str) or not name for name in fields)
            or len(fields) != len(set(fields))
        ):
            raise RuntimeError("structured validator required fields are invalid")
        normalized["required_fields"] = sorted(fields)
    return normalized


def _txn_spec_descriptor(spec: Mapping) -> dict:
    return {
        "drop_types": list(spec["drop_types"]),
        "path": spec["path"],
        "role": spec["role"],
        "schema": spec["schema"],
        "sibling": spec["sibling"],
        "validator": spec["validator"],
    }


def _txn_reconstruct_plan(manifest: Mapping) -> dict:
    generation_id = _txn_validate_generation_id(manifest["generation_id"])
    source_generation_id = manifest["source_generation_id"]
    siblings = manifest["requested_siblings"]
    outputs = manifest["outputs"]
    if (
        not isinstance(source_generation_id, str)
        or not source_generation_id.strip()
        or any(ord(char) < 32 for char in source_generation_id)
    ):
        raise RuntimeError("manifest source generation ID is invalid")
    if (
        not isinstance(siblings, list)
        or not siblings
        or len(siblings) != len(set(siblings))
        or any(
            not isinstance(sibling, str) or _GENERATION_ID_RE.fullmatch(sibling) is None
            for sibling in siblings
        )
    ):
        raise RuntimeError("manifest requested sibling inventory is invalid")
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("manifest output inventory must be a non-empty list")

    specs = []
    paths = set()
    for item in outputs:
        if not isinstance(item, dict) or set(item) != _TXN_OUTPUT_KEYS:
            raise RuntimeError("output metadata fields are not exact")
        path = _txn_validate_path(item["path"])
        if path in paths:
            raise RuntimeError(f"duplicate manifest output path: {path}")
        paths.add(path)
        role = item["role"]
        if role not in _TXN_ROLES:
            raise RuntimeError(f"invalid output role for {path}: {role!r}")
        schema = _txn_validate_schema(item["schema"])
        validator = _txn_validator(item["validator"])
        if validator["schema_version"] != schema:
            raise RuntimeError(f"output schema and validator differ for {path}")
        sibling = item["sibling"]
        if sibling is not None and (
            not isinstance(sibling, str) or _GENERATION_ID_RE.fullmatch(sibling) is None
        ):
            raise RuntimeError(f"invalid output sibling for {path}")
        drop_types = item["drop_types"]
        if (
            not isinstance(drop_types, list)
            or len(drop_types) != len(set(drop_types))
            or any(
                not isinstance(drop_type, str) or _TXN_DROP_TYPE_RE.fullmatch(drop_type) is None
                for drop_type in drop_types
            )
        ):
            raise RuntimeError(f"{path}: drop type inventory is invalid")
        drop_types = sorted(drop_types)
        if drop_types and role != "sidecar":
            raise RuntimeError(f"{path}: drop types require a sidecar role")
        if role in {"raw", "train", "eval"} and (
            validator["kind"] != "jsonl-object"
            or validator["allow_empty"]
            or not path.endswith(".jsonl")
        ):
            raise RuntimeError(f"{path}: accounted output requires non-empty JSONL")
        if role == "heldout" and (
            validator["kind"] not in {"jsonl-object", "json-object"}
            or not validator["require_generation_links"]
        ):
            raise RuntimeError(f"{path}: heldout output does not validate generation links")
        if (
            role == "sidecar"
            and validator["kind"]
            in {
                "jsonl-object",
                "json-object",
            }
            and not validator["require_generation_links"]
        ):
            raise RuntimeError(f"{path}: structured sidecar does not validate generation links")
        if drop_types and (
            validator["kind"] != "jsonl-object" or not validator["require_generation_links"]
        ):
            raise RuntimeError(f"{path}: typed drop sidecar validator is invalid")
        if item["generation_id"] != generation_id:
            raise RuntimeError(f"cross-file generation ID mismatch for {path}")
        if item["source_generation_id"] != source_generation_id:
            raise RuntimeError(f"cross-file source generation ID mismatch for {path}")
        for name in ("sha256", "logical_sha256"):
            _require_sha256(item[name], f"{path} {name}")
        for name in ("bytes", "rows"):
            value = item[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"{path}: invalid {name}")
        specs.append(
            {
                "drop_types": drop_types,
                "path": path,
                "role": role,
                "schema": schema,
                "sibling": sibling,
                "validator": validator,
            }
        )

    requested = set(siblings)
    if any(spec["sibling"] is not None and spec["sibling"] not in requested for spec in specs):
        raise RuntimeError("output inventory names an unrequested sibling")
    if not any(spec["role"] == "heldout" for spec in specs):
        raise RuntimeError("output inventory must declare a heldout file")
    for sibling in siblings:
        for role in ("raw", "train", "eval"):
            count = sum(spec["sibling"] == sibling and spec["role"] == role for spec in specs)
            if count != 1:
                raise RuntimeError(
                    f"requested sibling {sibling} must declare exactly one {role} output"
                )
        if not any(
            spec["sibling"] == sibling and spec["role"] == "sidecar" and spec["drop_types"]
            for spec in specs
        ):
            raise RuntimeError(f"requested sibling {sibling} lacks a typed drop sidecar")
    plan_body = {
        "outputs": [
            _txn_spec_descriptor(spec) for spec in sorted(specs, key=lambda item: item["path"])
        ],
        "requested_siblings": siblings,
        "schema_version": CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["plan_schema"],
        "source_generation_id": source_generation_id,
    }
    plan_root = hashlib.sha256(_txn_canonical_bytes(plan_body)).hexdigest()
    if manifest["plan_root_sha256"] != plan_root:
        raise RuntimeError("manifest plan root is invalid")
    return {
        "generation_id": generation_id,
        "source_generation_id": source_generation_id,
        "requested_siblings": siblings,
        "specs": specs,
        "plan_root_sha256": plan_root,
    }


def _txn_read_lines(path: Path) -> list[bytes]:
    _txn_require_regular(path, str(path))
    lines = []
    with path.open("rb") as handle:
        for row, line in enumerate(handle, start=1):
            if not line.endswith(b"\n"):
                raise RuntimeError(f"{path}:{row}: JSONL row is not newline terminated")
            lines.append(line)
    return lines


def _txn_read_jsonl(path: Path) -> list[dict]:
    records = []
    for row, line in enumerate(_txn_read_lines(path), start=1):
        try:
            record = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{path}:{row}: invalid JSONL") from error
        if not isinstance(record, dict):
            raise TypeError(f"{path}:{row}: JSONL row must be an object")
        records.append(record)
    return records


def _txn_validate_structured(
    record: Mapping,
    *,
    path: Path,
    row: int | None,
    spec: Mapping,
    plan: Mapping,
) -> None:
    location = f"{path}:{row}" if row is not None else str(path)
    validator = spec["validator"]
    if record.get("schema_version") != spec["schema"]:
        raise RuntimeError(f"{location}: schema version mismatch")
    missing = sorted(set(validator["required_fields"]) - set(record))
    if missing:
        raise RuntimeError(f"{location}: missing required fields {missing}")
    if validator["require_generation_links"]:
        if record.get("generation_id") != plan["generation_id"]:
            raise RuntimeError(f"{location}: stale generation link")
        if record.get("source_generation_id") != plan["source_generation_id"]:
            raise RuntimeError(f"{location}: stale source generation link")
        if record.get("plan_root_sha256") != plan["plan_root_sha256"]:
            raise RuntimeError(f"{location}: stale plan root link")


def _txn_output_identity(path: Path, spec: Mapping, plan: Mapping) -> tuple[int, str]:
    validator = spec["validator"]
    kind = validator["kind"]
    if kind == "jsonl-object":
        records = _txn_read_jsonl(path)
        if not records and not validator["allow_empty"]:
            raise RuntimeError(f"{path}: structured JSONL output is empty")
        for row, record in enumerate(records, start=1):
            _txn_validate_structured(record, path=path, row=row, spec=spec, plan=plan)
        if validator["require_generation_links"]:
            normalized = b"".join(
                _txn_canonical_bytes(
                    {
                        **record,
                        **(
                            {"generation_id": "<physical-generation-id>"}
                            if "generation_id" in record
                            else {}
                        ),
                    }
                )
                for record in records
            )
            logical_sha = hashlib.sha256(normalized).hexdigest()
        else:
            logical_sha = file_sha256(path)
        return len(records), logical_sha
    if kind == "json-object":
        record = _read_json_object(path, str(path))
        _txn_validate_structured(record, path=path, row=None, spec=spec, plan=plan)
        normalized = dict(record)
        if validator["require_generation_links"] and "generation_id" in normalized:
            normalized["generation_id"] = "<physical-generation-id>"
        logical_sha = (
            hashlib.sha256(_txn_canonical_bytes(normalized)).hexdigest()
            if validator["require_generation_links"]
            else file_sha256(path)
        )
        return 1, logical_sha
    raise RuntimeError(
        f"{path}: binary validator {validator['validator_id']!r} is not registered "
        "in the standalone token consumer"
    )


def _txn_expected_directories(files: Iterable[str]) -> set[str]:
    directories = set()
    for logical_path in files:
        parent = PurePosixPath(logical_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _txn_physical_inventory(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories = set()
    stack = [(root, PurePosixPath("."))]
    while stack:
        current_path, current_logical = stack.pop()
        for entry in os.scandir(current_path):
            logical = (
                PurePosixPath(entry.name)
                if current_logical == PurePosixPath(".")
                else current_logical / entry.name
            )
            relative = logical.as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"symlink is forbidden in inventory: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                stack.append((Path(entry.path), logical))
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise RuntimeError(f"hard link is forbidden in inventory: {relative}")
                files.add(relative)
            else:
                raise RuntimeError(f"special node is forbidden in inventory: {relative}")
    return files, directories


def _txn_validate_read_only(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"sealed generation contains symlink: {path}")
        if metadata.st_mode & 0o222:
            raise RuntimeError(f"sealed generation path is writable: {path}")


def _txn_raw_occurrences(path: Path, sibling: str, logical_path: str) -> list[dict]:
    occurrences = []
    for row, raw_bytes in enumerate(_txn_read_lines(path), start=1):
        digest = hashlib.sha256(raw_bytes).hexdigest()
        occurrence_id = f"occurrence/v1:{sibling}:{logical_path}:{row}:{digest}"
        occurrences.append(
            {
                "occurrence_id": occurrence_id,
                "sibling": sibling,
                "raw_path": logical_path,
                "raw_row": row,
                "raw_sha256": digest,
                "raw_bytes": raw_bytes,
            }
        )
    return occurrences


def _txn_validate_accounting(generation: Path, plan: Mapping) -> dict:
    specs = {spec["path"]: spec for spec in plan["specs"]}
    occurrences = {}
    occurrences_by_sibling = {}
    for sibling in plan["requested_siblings"]:
        raw_spec = next(
            spec for spec in plan["specs"] if spec["sibling"] == sibling and spec["role"] == "raw"
        )
        raw = _txn_raw_occurrences(
            generation.joinpath(*PurePosixPath(raw_spec["path"]).parts),
            sibling,
            raw_spec["path"],
        )
        occurrences_by_sibling[sibling] = {item["occurrence_id"] for item in raw}
        for occurrence in raw:
            if occurrence["occurrence_id"] in occurrences:
                raise RuntimeError(f"duplicate occurrence ID: {occurrence['occurrence_id']}")
            occurrences[occurrence["occurrence_id"]] = occurrence

    route_records = _txn_read_jsonl(generation / "ROUTES.jsonl")
    assigned = {}
    destination_seen = set()
    destination_lines = {
        spec["path"]: _txn_read_lines(generation.joinpath(*PurePosixPath(spec["path"]).parts))
        for spec in plan["specs"]
        if spec["role"] in {"train", "eval"} or (spec["role"] == "sidecar" and spec["drop_types"])
    }
    summaries = {
        sibling: {
            "drop_rows": 0,
            "drop_types": Counter(),
            "eval_rows": 0,
            "raw_rows": len(occurrences_by_sibling[sibling]),
            "train_rows": 0,
        }
        for sibling in plan["requested_siblings"]
    }
    for route_number, route in enumerate(route_records, start=1):
        if set(route) != _TXN_ROUTE_KEYS:
            raise RuntimeError(f"route {route_number}: fields are not exact")
        occurrence_id = route["occurrence_id"]
        if occurrence_id in assigned:
            raise RuntimeError(f"occurrence assigned more than once: {occurrence_id}")
        if occurrence_id not in occurrences:
            raise RuntimeError(f"route references unknown occurrence: {occurrence_id}")
        occurrence = occurrences[occurrence_id]
        if (
            route["sibling"] != occurrence["sibling"]
            or route["raw_path"] != occurrence["raw_path"]
            or route["raw_row"] != occurrence["raw_row"]
            or route["raw_sha256"] != occurrence["raw_sha256"]
        ):
            raise RuntimeError(f"route occurrence fields mismatch: {occurrence_id}")
        if route["plan_root_sha256"] != plan["plan_root_sha256"]:
            raise RuntimeError(f"route root link mismatch: {occurrence_id}")
        destination_path = route["destination_path"]
        if destination_path not in specs:
            raise RuntimeError(f"route destination is not inventoried: {destination_path}")
        destination_spec = specs[destination_path]
        if destination_spec["sibling"] != occurrence["sibling"]:
            raise RuntimeError(f"cross-sibling occurrence route: {occurrence_id}")
        disposition = route["disposition"]
        expected_role = {"train": "train", "eval": "eval", "drop": "sidecar"}.get(disposition)
        if expected_role is None or destination_spec["role"] != expected_role:
            raise RuntimeError(f"route disposition/destination mismatch: {occurrence_id}")
        destination_row = route["destination_row"]
        lines = destination_lines[destination_path]
        if (
            not isinstance(destination_row, int)
            or isinstance(destination_row, bool)
            or not 1 <= destination_row <= len(lines)
        ):
            raise RuntimeError(f"route destination row is invalid: {occurrence_id}")
        destination_key = (destination_path, destination_row)
        if destination_key in destination_seen:
            raise RuntimeError(f"destination row has multiple routes: {destination_key}")
        destination_seen.add(destination_key)
        if disposition in {"train", "eval"}:
            if lines[destination_row - 1] != occurrence["raw_bytes"]:
                raise RuntimeError(f"routed native bytes changed: {destination_key}")
            if route["drop_type"] is not None:
                raise RuntimeError("non-drop route has a drop type")
            summaries[occurrence["sibling"]][f"{disposition}_rows"] += 1
        else:
            try:
                drop = json.loads(lines[destination_row - 1])
            except (UnicodeError, json.JSONDecodeError) as error:
                raise RuntimeError("drop route is not valid JSON") from error
            if not isinstance(drop, dict) or set(drop) != _TXN_DROP_KEYS:
                raise RuntimeError("drop sidecar fields are not exact")
            if (
                drop["occurrence_id"] != occurrence_id
                or drop["sibling"] != occurrence["sibling"]
                or drop["raw_path"] != occurrence["raw_path"]
                or drop["raw_row"] != occurrence["raw_row"]
                or drop["raw_sha256"] != occurrence["raw_sha256"]
                or drop["schema_version"] != destination_spec["schema"]
                or drop["generation_id"] != plan["generation_id"]
                or drop["source_generation_id"] != plan["source_generation_id"]
                or drop["plan_root_sha256"] != plan["plan_root_sha256"]
            ):
                raise RuntimeError("drop route does not match its physical occurrence")
            drop_type = route["drop_type"]
            if drop_type != drop["drop_type"] or drop_type not in destination_spec["drop_types"]:
                raise RuntimeError("drop type is not declared or route-linked")
            summaries[occurrence["sibling"]]["drop_rows"] += 1
            summaries[occurrence["sibling"]]["drop_types"][drop_type] += 1
        assigned[occurrence_id] = route
    unassigned = sorted(set(occurrences) - set(assigned))
    if unassigned:
        raise RuntimeError(f"unassigned raw occurrences: {len(unassigned)}")
    for path, lines in destination_lines.items():
        routed_rows = sum(route["destination_path"] == path for route in route_records)
        if routed_rows != len(lines):
            raise RuntimeError(f"{path}: output rows and occurrence routes differ")
    normalized = {
        sibling: {
            **summary,
            "drop_types": dict(sorted(summary["drop_types"].items())),
        }
        for sibling, summary in summaries.items()
    }
    return {
        "scheme": CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["accounting_scheme"],
        "siblings": normalized,
    }


def _txn_logical_root(plan: Mapping, outputs: Sequence[Mapping], routes: Mapping) -> str:
    logical_outputs = [
        {
            "drop_types": item["drop_types"],
            "logical_sha256": item["logical_sha256"],
            "path": item["path"],
            "role": item["role"],
            "rows": item["rows"],
            "schema": item["schema"],
            "sibling": item["sibling"],
            "validator": item["validator"],
        }
        for item in sorted(outputs, key=lambda value: value["path"])
    ]
    body = {
        "outputs": logical_outputs,
        "plan_root_sha256": plan["plan_root_sha256"],
        "requested_siblings": plan["requested_siblings"],
        "routes_root_sha256": routes["root_sha256"],
        "schema_version": CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["logical_root_schema"],
        "source_generation_id": plan["source_generation_id"],
    }
    return hashlib.sha256(_txn_canonical_bytes(body)).hexdigest()


def _txn_validate_journal(root: Path, current: Mapping) -> str:
    transactions = root / "transactions"
    _txn_require_directory(transactions, "transactions control path")
    committed = {}
    for path in sorted(transactions.iterdir()):
        _txn_require_regular(path, f"transaction state {path.name}")
        if not path.name.endswith(".committed.json"):
            raise RuntimeError(f"uninventoried or non-durable transaction state: {path.name}")
        record = _read_json_object(path, "transaction state")
        if (
            set(record) != _TXN_STATE_KEYS
            or record.get("schema_version")
            != CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["transaction_state_schema"]
            or record.get("state") != "committed"
        ):
            raise RuntimeError(f"invalid committed transaction state: {path.name}")
        generation_id = _txn_validate_generation_id(record.get("generation_id"))
        if path.name != f"{generation_id}.committed.json":
            raise RuntimeError(f"transaction state filename mismatch: {path.name}")
        _require_sha256(record.get("manifest_sha256"), "transaction manifest digest")
        _require_sha256(record.get("logical_root_sha256"), "transaction logical root")
        if generation_id in committed:
            raise RuntimeError(f"duplicate committed transaction: {generation_id}")
        committed[generation_id] = record
    generations = root / "generations"
    _txn_require_directory(generations, "generations control path")
    actual_generations = set()
    for path in generations.iterdir():
        _txn_require_directory(path, f"committed generation {path.name}")
        _txn_validate_generation_id(path.name)
        actual_generations.add(path.name)
    if actual_generations != set(committed):
        raise RuntimeError("committed transaction and generation inventories differ")
    generation_id = current["generation_id"]
    if generation_id not in committed:
        raise RuntimeError(f"CURRENT generation transaction state is missing for {generation_id}")
    record = committed[generation_id]
    for field in ("generation_id", "manifest_sha256", "logical_root_sha256"):
        if record[field] != current[field]:
            raise RuntimeError(f"transaction state does not match CURRENT {field}")
    return "durable"


def _txn_validate_control_layout(root: Path) -> None:
    expected = {
        ".generation.lock",
        ".staging",
        "CURRENT",
        "generations",
        "quarantine",
        "transactions",
    }
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise RuntimeError(
            "corpus transaction control inventory is not exact: "
            f"unexpected={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )
    _txn_require_regular(root / ".generation.lock", "generation lock")
    staging = root / ".staging"
    _txn_require_directory(staging, "staging control path")
    if any(staging.iterdir()):
        raise RuntimeError("stale staging material is present")
    quarantine = root / "quarantine"
    _txn_require_directory(quarantine, "quarantine control path")
    for entry in quarantine.iterdir():
        _txn_require_directory(entry, f"quarantine entry {entry.name}")
        if {child.name for child in entry.iterdir()} != {"QUARANTINE.json", "payload"}:
            raise RuntimeError(f"invalid quarantine inventory: {entry.name}")
        _txn_require_regular(entry / "QUARANTINE.json", f"{entry.name} quarantine marker")
        payload = entry / "payload"
        metadata = payload.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"quarantine payload is a symlink: {entry.name}")


def _txn_resolve_current(root: Path) -> dict:
    _txn_require_directory(root, "corpus transaction root")
    _txn_validate_control_layout(root)
    current_path = root / "CURRENT"
    _txn_require_regular(current_path, "CURRENT")
    current = _read_json_object(current_path, "corpus CURRENT seal")
    if set(current) != {
        "generation_id",
        "logical_root_sha256",
        "manifest_sha256",
        "schema_version",
    }:
        raise RuntimeError("corpus CURRENT seal fields are not exact")
    if current["schema_version"] != CORPUS_CURRENT_SCHEMA_VERSION:
        raise RuntimeError("unsupported corpus CURRENT contract schema")
    generation_id = _txn_validate_generation_id(current["generation_id"])
    logical_root = _require_sha256(current["logical_root_sha256"], "corpus CURRENT logical root")
    manifest_sha = _require_sha256(current["manifest_sha256"], "corpus CURRENT manifest digest")
    commit_state = _txn_validate_journal(root, current)
    generation = root / "generations" / generation_id
    _txn_require_directory(generation, "CURRENT generation directory")
    manifest_path = generation / "MANIFEST.json"
    _txn_require_regular(manifest_path, "MANIFEST.json")
    if file_sha256(manifest_path) != manifest_sha:
        raise RuntimeError("corpus CURRENT manifest SHA-256 does not match MANIFEST.json")
    manifest = _read_json_object(manifest_path, "corpus generation manifest")
    if set(manifest) != set(CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["manifest_keys"]):
        raise RuntimeError("corpus generation manifest fields are not exact")
    if (
        manifest["schema_version"] != CORPUS_MANIFEST_SCHEMA_VERSION
        or type(manifest["api_version"]) is not int
        or manifest["api_version"] != CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["api_version"]
        or manifest["physical_generation_id_policy"]
        != CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["physical_generation_id_policy"]
        or manifest["generation_id"] != generation_id
    ):
        raise RuntimeError("unsupported or mismatched corpus generation manifest")
    body = dict(manifest)
    declared_manifest_root = _require_sha256(
        body.pop("manifest_root_sha256"), "corpus generation manifest root"
    )
    if hashlib.sha256(_txn_canonical_bytes(body)).hexdigest() != declared_manifest_root:
        raise RuntimeError("corpus generation manifest root is invalid")
    if manifest["logical_root_sha256"] != logical_root:
        raise RuntimeError("corpus CURRENT logical root does not match generation manifest")

    plan = _txn_reconstruct_plan(manifest)
    routes = manifest["routes"]
    if not isinstance(routes, dict) or set(routes) != _TXN_ROUTES_KEYS:
        raise RuntimeError("routes metadata fields are not exact")
    if (
        routes["path"] != "ROUTES.jsonl"
        or routes["schema_version"] != CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT["routes_schema"]
    ):
        raise RuntimeError("routes metadata schema/path is invalid")
    for field in ("bytes", "rows"):
        value = routes[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"routes {field} must be a non-negative integer")
    for field in ("sha256", "root_sha256"):
        _require_sha256(routes[field], f"routes {field}")

    expected_files = {item["path"] for item in manifest["outputs"]} | {
        "MANIFEST.json",
        "ROUTES.jsonl",
    }
    expected_directories = _txn_expected_directories(expected_files)
    if (
        not isinstance(manifest["directories"], list)
        or not all(isinstance(directory, str) for directory in manifest["directories"])
        or len(manifest["directories"]) != len(set(manifest["directories"]))
        or manifest["directories"] != sorted(expected_directories)
    ):
        raise RuntimeError("manifest directories do not match canonical inventory")
    actual_files, actual_directories = _txn_physical_inventory(generation)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            "physical generation inventory is not exact: "
            f"unexpected_files={sorted(actual_files - expected_files)}, "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"unexpected_directories={sorted(actual_directories - expected_directories)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}"
        )
    _txn_validate_read_only(generation)
    specs = {spec["path"]: spec for spec in plan["specs"]}
    metadata = {item["path"]: item for item in manifest["outputs"]}
    for path in sorted(metadata):
        output = generation.joinpath(*PurePosixPath(path).parts)
        item = metadata[path]
        file_metadata = _txn_require_regular(output, path)
        if file_metadata.st_size != item["bytes"]:
            raise RuntimeError(f"{path}: byte-size mismatch")
        if file_sha256(output) != item["sha256"]:
            raise RuntimeError(f"{path}: SHA-256 mismatch")
        rows, logical_sha = _txn_output_identity(output, specs[path], plan)
        if rows != item["rows"]:
            raise RuntimeError(f"{path}: row-count mismatch")
        if logical_sha != item["logical_sha256"]:
            raise RuntimeError(f"{path}: logical SHA-256 mismatch")

    routes_path = generation / "ROUTES.jsonl"
    route_metadata = _txn_require_regular(routes_path, "ROUTES.jsonl")
    if (
        route_metadata.st_size != routes["bytes"]
        or file_sha256(routes_path) != routes["sha256"]
        or routes["root_sha256"] != routes["sha256"]
        or len(_txn_read_jsonl(routes_path)) != routes["rows"]
    ):
        raise RuntimeError("routes metadata does not match physical routes")
    accounting = _txn_validate_accounting(generation, plan)
    if accounting != manifest["accounting"]:
        raise RuntimeError("manifest accounting summary is stale")
    if _txn_logical_root(plan, manifest["outputs"], routes) != logical_root:
        raise RuntimeError("logical generation root mismatch")
    return {
        "commit_state": commit_state,
        "current": current,
        "current_sha256": file_sha256(current_path),
        "generation": generation,
        "manifest": manifest,
        "manifest_file_sha256": manifest_sha,
        "manifest_root_sha256": declared_manifest_root,
        "plan": plan,
    }


def _select_p3_source_records(
    outputs: Sequence[Mapping],
    *,
    generation_id: str,
) -> dict[str, dict[str, dict]]:
    """Select the exact P3 train/eval OutputSpecs by role and sibling, never by path."""
    family_records: dict[str, dict[str, dict]] = {family: {} for family in FAMILIES}
    seen_paths: set[str] = set()
    for entry in outputs:
        path = entry.get("path")
        if not isinstance(path, str):
            raise TypeError("corpus generation output path is missing")
        if path in seen_paths:
            raise RuntimeError(f"corpus generation duplicates output path {path!r}")
        seen_paths.add(path)
        role = entry["role"]
        sibling = entry["sibling"]
        if role not in {"train", "eval"}:
            continue
        if sibling not in FAMILIES:
            raise RuntimeError(f"corpus generation has unknown {role} family {sibling!r}")
        expected_schema = P3_SOURCE_SCHEMAS[sibling]
        if entry["schema"] != expected_schema:
            raise RuntimeError(
                f"corpus generation {role} schema for {sibling} is {entry['schema']!r}, "
                f"expected {expected_schema!r}"
            )
        key = "train" if role == "train" else "val"
        if key in family_records[sibling]:
            raise RuntimeError(f"corpus generation duplicates {role} family {sibling}")
        family_records[sibling][key] = {
            "family": sibling,
            "path": f"generations/{generation_id}/{path}",
            "generation_relative_path": path,
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
            "rows": entry["rows"],
            "schema": entry["schema"],
        }
    incomplete = {
        family: sorted({"train", "val"} - set(records))
        for family, records in family_records.items()
        if set(records) != {"train", "val"}
    }
    if incomplete:
        raise RuntimeError(
            f"corpus generation does not expose one exact train/eval role for every P3 "
            f"sibling: {incomplete}"
        )
    return family_records


def load_corpus_generation_contract(root: str | Path) -> dict:
    """Resolve and independently validate the accepted atomic transaction v2 contract."""
    unresolved_root = Path(root).expanduser()
    if unresolved_root.is_symlink():
        raise RuntimeError("corpus transaction root must not be a symlink")
    try:
        root = unresolved_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "corpus-generation CURRENT contract is required; legacy directories are refused"
        ) from error
    resolved = _txn_resolve_current(root)
    manifest = resolved["manifest"]
    generation_id = resolved["current"]["generation_id"]
    if manifest["requested_siblings"] != list(FAMILIES):
        raise RuntimeError("corpus generation does not contain the exact ordered P3 family set")
    family_records = _select_p3_source_records(
        manifest["outputs"],
        generation_id=generation_id,
    )
    return {
        "schema_version": CORPUS_BINDING_SCHEMA_VERSION,
        "contract_root": str(root),
        "generation_id": generation_id,
        "logical_root_sha256": resolved["current"]["logical_root_sha256"],
        "manifest_root_sha256": resolved["manifest_root_sha256"],
        "manifest_file_sha256": resolved["manifest_file_sha256"],
        "current_sha256": resolved["current_sha256"],
        "commit_state": resolved["commit_state"],
        "semantic_contract": json.loads(json.dumps(CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT)),
        "semantic_contract_sha256": CORPUS_TRANSACTION_V2_SEMANTIC_SHA256,
        "producer_source_sha256": CORPUS_TRANSACTION_V2_PRODUCER_SOURCE_SHA256,
        "families": family_records,
    }


def require_corpus_generation_current(binding: Mapping) -> None:
    """Re-resolve CURRENT and rehash the full transaction before accepting reuse."""
    if (
        binding.get("semantic_contract") != CORPUS_TRANSACTION_V2_SEMANTIC_CONTRACT
        or binding.get("semantic_contract_sha256") != CORPUS_TRANSACTION_V2_SEMANTIC_SHA256
    ):
        raise RuntimeError("corpus transaction semantic contract changed")
    current = load_corpus_generation_contract(str(binding.get("contract_root", "")))
    identity_fields = (
        "generation_id",
        "logical_root_sha256",
        "manifest_root_sha256",
        "manifest_file_sha256",
        "current_sha256",
        "commit_state",
    )
    if any(current.get(field) != binding.get(field) for field in identity_fields):
        raise RuntimeError("corpus CURRENT changed after token build resolution")
    if current["families"] != binding.get("families"):
        raise RuntimeError("corpus source inventory or SHA-256 changed during tokenization")


def load_sealed_corpus_manifest(path: str | Path) -> dict:
    """Resolve and independently validate a direct six-family sealed corpus manifest.

    This is the production alternative to the atomic transaction contract: it binds
    the exact final train/eval JSONL bytes by SHA-256 without requiring a committed
    corpus-generation transaction. It is fail-closed -- any missing family, extra
    family, wrong row schema, malformed entry, or manifest-root mismatch aborts the
    build. It never enables the test-only unsealed seam: the fixed Qwen tokenizer
    seal is still enforced during encoding.
    """
    manifest_path = Path(path).expanduser()
    if manifest_path.is_symlink():
        raise RuntimeError("sealed corpus manifest must not be a symlink")
    data = json.loads(manifest_path.read_text())
    if data.get("schema_version") != SEALED_CORPUS_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("sealed corpus manifest schema_version is not recognized")
    families = data.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise RuntimeError(
            "sealed corpus manifest must declare exactly the ordered P3 family set"
        )
    for family in FAMILIES:
        record = families[family]
        if not isinstance(record, dict):
            raise RuntimeError(f"sealed corpus manifest family {family} is malformed")
        if record.get("schema") != P3_SOURCE_SCHEMAS[family]:
            raise RuntimeError(
                f"sealed corpus manifest family {family} declares the wrong row schema"
            )
        for role in ("train", "eval"):
            entry = record.get(role)
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not entry["path"]
                or not isinstance(entry.get("sha256"), str)
                or len(entry["sha256"]) != 64
                or not isinstance(entry.get("bytes"), int)
                or entry["bytes"] < 0
                or not isinstance(entry.get("rows"), int)
                or entry["rows"] < 0
            ):
                raise RuntimeError(
                    f"sealed corpus manifest family {family}/{role} is malformed"
                )
    body = {"schema_version": data["schema_version"], "families": families}
    manifest_root_sha256 = fingerprint_dict(body)
    if manifest_root_sha256 != data.get("manifest_root_sha256"):
        raise RuntimeError("sealed corpus manifest root SHA-256 does not seal its body")
    return {
        "schema_version": data["schema_version"],
        "manifest_root_sha256": manifest_root_sha256,
        "families": families,
    }


@contextmanager
def corpus_generation_shared_lock(root: str | Path) -> Iterator[None]:
    """Hold the producer's no-follow generation lock in shared mode."""
    root = Path(root)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or nofollow <= 0:
        raise RuntimeError("secure corpus lock requires O_NOFOLLOW")
    if type(directory) is not int or directory <= 0:
        raise RuntimeError("secure corpus lock requires O_DIRECTORY")
    flags = os.O_RDONLY | directory | nofollow
    absolute = Path(os.path.abspath(root))
    root_descriptor = os.open(absolute.anchor, flags)
    try:
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            try:
                child = os.open(part, flags, dir_fd=root_descriptor)
            except OSError as error:
                raise RuntimeError(
                    f"unsafe or symlinked corpus lock path component: {current}"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise RuntimeError(f"corpus lock path is not a real directory: {current}")
            os.close(root_descriptor)
            root_descriptor = child
        lock_descriptor = os.open(
            ".generation.lock",
            os.O_CREAT | os.O_RDWR | nofollow,
            0o600,
            dir_fd=root_descriptor,
        )
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)
    try:
        _txn_validate_regular_metadata(
            os.fstat(lock_descriptor),
            str(root / ".generation.lock"),
        )
        fcntl.flock(lock_descriptor, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _cleanup_staged_manifests(replacements: Sequence[tuple[Path, Path]]) -> None:
    synced = set()
    for pending, _ in replacements:
        try:
            metadata = pending.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"staged token manifest is a directory: {pending}")
        pending.unlink()
        synced.add(pending.parent)
    for parent in synced:
        fsync_directory(parent)


def commit_token_manifests_under_generation_lock(
    binding: Mapping,
    replacements: Sequence[tuple[Path, Path]],
    *,
    _before_lock: Callable[[], None] | None = None,
    _on_locked: Callable[[], None] | None = None,
    _on_committed: Callable[[], None] | None = None,
    _fault: Callable[[str, Path, Path], None] | None = None,
) -> None:
    """Validate CURRENT and replace token manifests while holding ``LOCK_SH``."""
    replacements = tuple((Path(pending), Path(final)) for pending, final in replacements)
    committed = []
    caught: BaseException | None = None
    original: BaseException | None = None
    try:
        if _before_lock is not None:
            _before_lock()
        with corpus_generation_shared_lock(str(binding.get("contract_root", ""))):
            if _on_locked is not None:
                _on_locked()
            require_corpus_generation_current(binding)
            for pending, final in replacements:
                _txn_require_regular(pending, f"staged token manifest {pending}")
                _txn_require_directory(final.parent, f"token manifest parent {final.parent}")
                if final.exists() or final.is_symlink():
                    _txn_require_regular(final, f"existing token manifest {final}")
            for pending, final in replacements:
                if _fault is not None:
                    _fault("replace_before", pending, final)
                os.replace(pending, final)
                committed.append(final)
                if _fault is not None:
                    _fault("replace_after", pending, final)
                    _fault("directory_fsync_before", pending, final)
                fsync_directory(final.parent)
                if _fault is not None:
                    _fault("directory_fsync_after", pending, final)
            if _on_committed is not None:
                _on_committed()
    except BaseException as error:  # noqa: BLE001 - cleanup includes cancellation
        original = error
        caught = TokenManifestCommitUncertainError(committed, error) if committed else error
    finally:
        _cleanup_staged_manifests(replacements)
    if caught is not None:
        if isinstance(caught, TokenManifestCommitUncertainError):
            raise caught from original
        raise caught


def finalize_token_manifests_under_generation_lock(
    binding: Mapping,
    specifications: Sequence[tuple[Path, Path, dict]],
    *,
    _fault: Callable[[str, Path, Path], None] | None = None,
) -> None:
    """Stage and commit every done/meta manifest with unconditional pending cleanup."""
    specifications = tuple(
        (Path(pending), Path(final), payload) for pending, final, payload in specifications
    )
    replacements = tuple((pending, final) for pending, final, _ in specifications)
    try:
        for pending, final, payload in specifications:
            stage_json_for_atomic_replace(
                pending,
                payload,
                final_path=final,
                _fault=_fault,
            )
        commit_token_manifests_under_generation_lock(
            binding,
            replacements,
            _fault=_fault,
        )
    finally:
        _cleanup_staged_manifests(replacements)


def build_encoding_cache_from_jsonl(
    tok,
    source: str | Path,
    cache_dir: str | Path,
    *,
    fingerprint: str,
    build: dict,
    eos_id: int,
    batch_size: int,
) -> tuple[list[np.ndarray], dict]:
    """Stream JSONL through batch encoding into a resumable on-disk cache."""
    cache_dir = Path(cache_dir)
    source = Path(source)
    if fingerprint != fingerprint_dict(build):
        raise RuntimeError("encoding cache fingerprint does not seal its build identity")
    if build.get("source_jsonl") != {
        "name": source.name,
        "sha256": file_sha256(source),
    }:
        raise RuntimeError("encoding cache build does not seal the exact source JSONL")
    cached = load_encoding_cache(
        cache_dir,
        fingerprint=fingerprint,
        build=build,
        source=source,
    )
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.u32le.bin"
    offsets_path = cache_dir / "offsets.u64le.bin"
    token_tmp = cache_dir / "tokens.u32le.bin.partial"
    offset_tmp = cache_dir / "offsets.u64le.bin.partial"
    progress_path = cache_dir / "progress.json"

    if progress_path.exists():
        progress = _read_json_object(progress_path, "encoding cache progress marker")
        _validate_cache_marker(
            cache_dir,
            progress,
            status="partial",
            fingerprint=fingerprint,
            build=build,
            source=source,
        )
        n_documents = int(progress["documents"])
        token_count = int(progress["tokens"])
        straddling = int(progress["straddling"])
        chunks = list(progress["chunks"])
    else:
        if any(cache_dir.iterdir()):
            raise RuntimeError(
                f"{cache_dir} contains cache bytes without a sealed progress marker; preserved"
            )
        n_documents = token_count = straddling = 0
        chunks = []
        with token_tmp.open("xb") as token_fh:
            token_fh.flush()
            os.fsync(token_fh.fileno())
        with offset_tmp.open("xb") as offset_fh:
            offset_fh.write(np.asarray([0], dtype="<u8").tobytes())
            offset_fh.flush()
            os.fsync(offset_fh.fileno())
        fsync_directory(cache_dir)
        atomic_write_json(
            progress_path,
            _cache_marker(
                status="partial",
                build=build,
                fingerprint=fingerprint,
                documents=0,
                tokens=0,
                straddling=0,
                chunks=[],
                token_path=token_tmp,
                offset_path=offset_tmp,
            ),
        )

    def commit_batch(rows, raw_rows, token_fh, offset_fh) -> None:
        nonlocal n_documents, token_count, straddling
        if not rows:
            return
        document_start = n_documents
        token_start = token_count
        encoded, crossed = encode_rows_batched(tok, rows, eos_id=eos_id, batch_size=batch_size)
        new_offsets = np.empty(len(encoded), dtype="<u8")
        token_bytes = bytearray()
        for i, ids in enumerate(encoded):
            raw = ids.astype("<u4", copy=False).tobytes()
            token_bytes.extend(raw)
            token_count += len(ids)
            new_offsets[i] = token_count
        offset_bytes = new_offsets.tobytes()
        token_fh.write(token_bytes)
        offset_fh.write(offset_bytes)
        n_documents += len(encoded)
        straddling += crossed
        token_fh.flush()
        offset_fh.flush()
        os.fsync(token_fh.fileno())
        os.fsync(offset_fh.fileno())
        chunks.append(
            {
                "index": len(chunks),
                "documents": {"start": document_start, "end": n_documents},
                "tokens": {"start": token_start, "end": token_count},
                "token_bytes": {"start": token_start * 4, "end": token_count * 4},
                "offset_bytes": {
                    "start": (document_start + 1) * 8,
                    "end": (n_documents + 1) * 8,
                },
                "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
                "offsets_sha256": hashlib.sha256(offset_bytes).hexdigest(),
                "source_rows_sha256": hashlib.sha256(b"".join(raw_rows)).hexdigest(),
                "straddling": int(crossed),
            }
        )
        atomic_write_json(
            progress_path,
            _cache_marker(
                status="partial",
                build=build,
                fingerprint=fingerprint,
                documents=n_documents,
                tokens=token_count,
                straddling=straddling,
                chunks=chunks,
                token_path=token_tmp,
                offset_path=offset_tmp,
            ),
        )

    with (
        source.open("rb") as source_fh,
        open(token_tmp, "ab") as token_fh,
        open(offset_tmp, "ab") as offset_fh,
    ):
        rows = []
        raw_rows = []
        source_index = 0
        for raw_line in source_fh:
            if not raw_line.strip():
                continue
            if source_index < n_documents:
                source_index += 1
                continue
            try:
                rows.append(json.loads(raw_line))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"{source}:{source_index + 1}: invalid source JSONL") from error
            raw_rows.append(raw_line)
            source_index += 1
            if len(rows) >= batch_size:
                commit_batch(rows, raw_rows, token_fh, offset_fh)
                rows.clear()
                raw_rows.clear()
        commit_batch(rows, raw_rows, token_fh, offset_fh)

    os.replace(token_tmp, tokens_path)
    os.replace(offset_tmp, offsets_path)
    fsync_directory(cache_dir)
    complete = _cache_marker(
        status="complete",
        build=build,
        fingerprint=fingerprint,
        documents=n_documents,
        tokens=token_count,
        straddling=straddling,
        chunks=chunks,
        token_path=tokens_path,
        offset_path=offsets_path,
    )
    atomic_write_json(
        cache_dir / "cache.json",
        complete,
    )
    progress_path.unlink()
    fsync_directory(cache_dir)
    result = load_encoding_cache(
        cache_dir,
        fingerprint=fingerprint,
        build=build,
        source=source,
    )
    assert result is not None
    return result


def count_run(ids: np.ndarray, run: Sequence[int]) -> int:
    if len(ids) < len(run):
        return 0
    windows = np.lib.stride_tricks.sliding_window_view(ids, len(run))
    return int(np.all(windows == np.asarray(run, dtype=ids.dtype), axis=1).sum())


def tokenizer_digest(tok) -> str:
    return hashlib.sha256(tok.backend_tokenizer.to_str().encode("utf-8")).hexdigest()


def fingerprint_dict(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def group_completion_sha256(payload: dict) -> str:
    """Seal stable completed-group metadata, including every shard digest."""
    sealed = {
        key: value
        for key, value in payload.items()
        if key not in {"completion_sha256", "resumed_group", "resumed_shards"}
    }
    return fingerprint_dict(sealed)


def tokenizer_composite_seal(
    tok,
    tokenizer_id: str | Path,
    *,
    separator_ids: Sequence[int],
    test_only: bool = False,
) -> dict:
    """Return the canonical four-part local tokenizer seal."""
    import tokenizers
    from provenance import tokenizer_behavior_sha256

    root = Path(tokenizer_id).expanduser()
    if (not root.is_dir() or root.is_symlink()) and not test_only:
        raise RuntimeError(
            "production tokenization requires a real local tokenizer directory; "
            "a mutable Hugging Face identifier is forbidden"
        )
    required = ("tokenizer.json", "tokenizer_config.json")
    files_sha256 = {}
    if root.is_dir() and not root.is_symlink():
        for filename in required:
            path = root / filename
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise RuntimeError(f"required sealed tokenizer file is missing: {path}")
            files_sha256[filename] = file_sha256(path)
    else:
        files_sha256 = {
            "tokenizer.json": tokenizer_digest(tok),
            "tokenizer_config.json": fingerprint_dict(
                {
                    "eos_token_id": tok.eos_token_id,
                    "pad_token_id": tok.pad_token_id,
                }
            ),
        }
    implementation_version = str(tokenizers.__version__)
    if implementation_version != "0.22.2":
        raise RuntimeError(
            "tokenizers implementation version mismatch: "
            f"expected 0.22.2, got {implementation_version}"
        )

    behavior = tokenizer_behavior_sha256(tok.backend_tokenizer)
    eos_id = int(tok.eos_token_id)
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else eos_id
    provenance = {
        "schema_version": TOKENIZER_SEAL_SCHEMA_VERSION,
        "tokenizer_file_sha256": files_sha256,
        "tokenizer_composite_sha256": behavior,
        "tokenizers_version": implementation_version,
        "tokenizer_eos_token_id": eos_id,
        "tokenizer_pad_token_id": pad_id,
        "separator": SEPARATOR,
        "separator_ids": [int(token_id) for token_id in separator_ids],
    }
    if test_only:
        provenance["test_only"] = True
        return provenance

    from provenance import seal_tokenizer_files

    sealed = seal_tokenizer_files(root)
    if (
        sealed.file_sha256 != files_sha256
        or sealed.composite_sha256 != behavior
        or sealed.tokenizers_version != implementation_version
        or sealed.eos_token_id != eos_id
        or sealed.pad_token_id != pad_id
    ):
        raise RuntimeError(
            "AutoTokenizer runtime does not match the approved four-part tokenizer seal"
        )
    production_seal = {
        **provenance,
        "tokenizer_artifact_id": sealed.artifact_id,
        "tokenizer_artifact_version": sealed.artifact_version,
    }
    if production_seal != FIXED_QWEN_TOKENIZER_SEAL:
        raise RuntimeError("production tokenizer does not match the canonical fixed Qwen seal")
    return dict(FIXED_QWEN_TOKENIZER_SEAL)


def process_corpus(task: dict) -> dict:
    """Encode, cache, pack, and shard one corpus. Safe to run in another process."""
        from transformers import AutoTokenizer

    source = Path(task["source"])
    name = str(task.get("name", source.stem))
    output_root = Path(task["output_root"])
    split = task["split"]
    sequence_length = int(task["sequence_length"])
    tokenizer_id = task["tokenizer"]
    test_only = bool(task.get("test_only", False))
    source_mode = task.get("source_mode")
    if source_mode is None:
        source_mode = "test-only" if test_only else "transaction"
    source_mode = str(source_mode)
    corpus_generation = task.get("corpus_generation")
    if not isinstance(corpus_generation, dict):
        raise TypeError("an immutable corpus-generation binding is required")

    def require_current_source() -> None:
        if source_mode == "test-only":
            return
        if source_mode == "transaction":
            require_corpus_generation_current(task["corpus_contract"])
        source_record = task.get("source_record")
        if (
            not isinstance(source_record, dict)
            or source_record.get("sha256") != file_sha256(source)
            or source_record.get("bytes") != source.stat().st_size
            or source_record.get("family") != name
            or not isinstance(source_record.get("schema"), str)
        ):
            raise RuntimeError("source JSONL drifted from the sealed corpus inputs")

    require_current_source()
    tok = AutoTokenizer.from_pretrained(
        tokenizer_id,
        **({} if test_only else {"local_files_only": True}),
    )
    if not tok.is_fast:
        raise RuntimeError("need a fast tokenizer for parallel batches and offsets")
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"{tokenizer_id} has no eos token")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    sep_ids = tok(SEPARATOR_SEARCH, add_special_tokens=False)["input_ids"]
    if not sep_ids:
        raise RuntimeError(f"{SEPARATOR_SEARCH!r} tokenizes to nothing")

    source_record = task.get("source_record")
    source_content_identity = {
        "name": source.name,
        "sha256": file_sha256(source),
    }
    source_identity = {
        "family": name,
        **source_content_identity,
        "schema": (
            source_record.get("schema")
            if isinstance(source_record, Mapping)
            else "test-only-unsealed"
        ),
    }
    tokenizer_seal = tokenizer_composite_seal(
        tok,
        tokenizer_id,
        separator_ids=sep_ids,
        test_only=test_only,
    )
    cache_identity = {
        "schema_version": ENCODING_CACHE_SCHEMA_VERSION,
        "code_version": TOKENIZE_CORPUS_CODE_VERSION,
        "source_jsonl": source_content_identity,
        "tokenizer": tokenizer_seal,
        "eos_token_id": int(eos_id),
    }
    cache_fingerprint = fingerprint_dict(cache_identity)
    packing_identity = {
        "algorithm": (PACKED_ALGORITHM_VERSION if task["pack"] else UNPACKED_ALGORITHM_VERSION),
        "split": split,
        "packed": bool(task["pack"]),
        "sequence_length": sequence_length,
        "shard_tokens": int(task["shard_tokens"]),
        "tokens_dtype": TOKENS_DTYPE_NAME,
        "byte_order": TOKENS_BYTE_ORDER,
        "eos_token_id": int(eos_id),
        "pad_token_id": int(pad_id),
        "separator": SEPARATOR,
        "separator_search": SEPARATOR_SEARCH,
        "separator_ids": list(sep_ids),
    }
    build_identity = {
        "schema_version": PACKED_GROUP_SCHEMA_VERSION,
        "code_version": TOKENIZE_CORPUS_CODE_VERSION,
        "source_jsonl": source_identity,
        "tokenizer": tokenizer_seal,
        "encoding_cache_fingerprint": cache_fingerprint,
        "packing": packing_identity,
        "corpus_generation": corpus_generation,
    }
    group_fingerprint = fingerprint_dict(build_identity)
    group_dir = output_root / "tokens" / name
    done_path = group_dir / f"{split}.done.json"
    done = load_completed_group(done_path, fingerprint=group_fingerprint, output_root=output_root)
    if done is not None:
        require_current_source()
        done["resumed_group"] = True
        print(f"  {name:<12}complete group validated and resumed", flush=True)
        return done

    cache_dir = Path(task["cache_root"]) / split / name / cache_fingerprint[:20]
    documents, cache_stats = build_encoding_cache_from_jsonl(
        tok,
        source,
        cache_dir,
        fingerprint=cache_fingerprint,
        build=cache_identity,
        eos_id=int(eos_id),
        batch_size=int(task["batch_size"]),
    )
    lengths = [len(ids) for ids in documents]

    # Full-corpus check, not a sample. This run determines whether the experiment
    # has two arms or two copies of dense training.
    missing = repeated = 0
    for ids in documents:
        hits = count_run(ids, sep_ids)
        missing += hits == 0
        repeated += hits > 1
    if missing or repeated:
        raise RuntimeError(
            f"{name}: separator {sep_ids} missing in {missing:,} documents and "
            f"repeated in {repeated:,}; cache and output were preserved"
        )

    kept_indices = [i for i, length in enumerate(lengths) if length <= sequence_length]
    dropped = len(lengths) - len(kept_indices)
    kept_lengths = [lengths[i] for i in kept_indices]
    if not kept_indices:
        raise RuntimeError(f"{name}: every document exceeds {sequence_length:,} tokens")

    if task["suggest"]:
        require_current_source()
        return {
            "name": name,
            "lengths": lengths,
            "separator_ids": list(sep_ids),
            "eos_token_id": int(eos_id),
            "pad_token_id": int(pad_id),
        }

    if task["pack"]:
        local_rows = pack_indices_by_length(kept_lengths, sequence_length)
        packed_rows = [[kept_indices[i] for i in row] for row in local_rows]
    else:
        packed_rows = [[i] for i in kept_indices]

    per_shard = max(int(task["shard_tokens"]) // sequence_length, 1)
    expected_names = {
        f"{split}-{ordinal:05d}.u32le.bin"
        for ordinal in range((len(packed_rows) + per_shard - 1) // per_shard)
    }
    require_exact_group_inventory(group_dir, pending_names=expected_names)

    shards = []
    resumed_shards = 0
    for ordinal, lo in enumerate(range(0, len(packed_rows), per_shard)):
        rows = packed_rows[lo : lo + per_shard]
        instances = [np.concatenate([documents[i] for i in row]) for row in rows]
        path = group_dir / f"{split}-{ordinal:05d}.u32le.bin"
        result = write_shard_resumable(
            path,
            instances,
            sequence_length=sequence_length,
            pad_id=int(pad_id),
        )
        resumed_shards += int(result["resumed"])
        shards.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "instances": len(instances),
                "tokens": result["tokens"],
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "tokens_dtype": result["tokens_dtype"],
                "byte_order": result["byte_order"],
            }
        )
        del instances

    real_tokens = sum(kept_lengths)
    payload = {
        "schema_version": PACKED_GROUP_SCHEMA_VERSION,
        "code_version": TOKENIZE_CORPUS_CODE_VERSION,
        "cross_split_binding_schema_version": CROSS_SPLIT_BINDING_SCHEMA_VERSION,
        "fingerprint": group_fingerprint,
        "build": build_identity,
        "cache_fingerprint": cache_fingerprint,
        "cache_root_sha256": cache_stats["cache_root_sha256"],
        "corpus_generation": corpus_generation,
        "name": name,
        "documents": len(kept_indices),
        "source_documents": len(lengths),
        "instances": len(packed_rows),
        "real_tokens": real_tokens,
        "padding_fraction": round(1 - real_tokens / max(len(packed_rows) * sequence_length, 1), 6),
        "dropped_over_length": dropped,
        "straddling": int(cache_stats["straddling"]),
        "separator_ids": list(sep_ids),
        "eos_token_id": int(eos_id),
        "pad_token_id": int(pad_id),
        "shards": shards,
        "resumed_shards": resumed_shards,
        "resumed_group": False,
    }
    payload["completion_sha256"] = group_completion_sha256(payload)
    require_exact_group_inventory(group_dir, pending_names=expected_names)
    if task.get("defer_done_commit", False):
        require_current_source()
    else:
        atomic_write_json(done_path, payload)
        require_exact_group_inventory(group_dir)
        require_current_source()
    print(
        f"  {name:<12}{len(kept_indices):>8,} docs -> "
        f"{len(packed_rows):>7,} instances, {payload['padding_fraction']:.1%} "
        f"padding, {dropped:,} dropped, {resumed_shards} shards resumed",
        flush=True,
    )
    return payload


def require_exact_token_output_inventory(
    output_root: str | Path,
    *,
    expected_groups: Sequence[str],
    cache_root: str | Path,
) -> None:
    """Validate the exact token-output control tree without deleting anything."""
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    allowed_root = {"tokens", "train_meta.json", "val_meta.json"}
    if cache_root == output_root / ".token-cache":
        allowed_root.add(".token-cache")
    actual_root = {path.name for path in output_root.iterdir()}
    extras = sorted(actual_root - allowed_root)
    if extras:
        raise RuntimeError(f"{output_root} has unknown token-output inventory {extras}; preserved")
    tokens_root = output_root / "tokens"
    if not tokens_root.exists():
        return
    if not tokens_root.is_dir() or tokens_root.is_symlink():
        raise RuntimeError(f"{tokens_root} is not a real token group directory")
    expected = set(expected_groups)
    for path in tokens_root.iterdir():
        if path.name not in expected:
            raise RuntimeError(f"{tokens_root} has unknown token group {path.name!r}; preserved")
        require_exact_group_inventory(path)


def _token_manifest_cross_split_binding(manifest: Mapping) -> dict:
    if manifest.get("cross_split_binding_schema_version") != CROSS_SPLIT_BINDING_SCHEMA_VERSION:
        raise RuntimeError("token manifest lacks the accepted cross-split binding schema")
    packing = manifest.get("packing_config")
    source_inventory = manifest.get("source_family_inventory")
    groups = manifest.get("groups")
    if not isinstance(packing, Mapping):
        raise TypeError("cross-split token manifest lacks packing configuration")
    if not isinstance(source_inventory, Mapping) or not isinstance(groups, Mapping):
        raise TypeError("cross-split token manifest lacks exact source family inventory")
    if set(source_inventory) != set(groups):
        raise RuntimeError("cross-split source family inventory does not match token groups")
    normalized_sources = {}
    for name, record in source_inventory.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, Mapping)
            or record.get("family") != name
            or record.get("schema") != P3_SOURCE_SCHEMAS.get(name, record.get("schema"))
        ):
            raise RuntimeError(f"cross-split source family inventory is invalid for {name!r}")
        normalized_sources[name] = {
            "family": record["family"],
            "schema": record["schema"],
        }
    return {
        "schema_version": CROSS_SPLIT_BINDING_SCHEMA_VERSION,
        "declared_cross_split_schema": manifest.get("cross_split_binding_schema_version"),
        "corpus_schema_version": manifest.get("schema_version"),
        "code_version": manifest.get("code_version"),
        "corpus_generation": manifest.get("corpus_generation"),
        "tokenizer_seal": manifest.get("tokenizer_seal"),
        "packing_contract": _packing_cross_split_contract(packing),
        "source_family_inventory": normalized_sources,
    }


def _require_group_matches_token_manifest(
    group_binding: Mapping,
    manifest_binding: Mapping,
    *,
    name: str,
) -> None:
    expected = {
        "corpus_generation": manifest_binding["corpus_generation"],
        "tokenizer_seal": manifest_binding["tokenizer_seal"],
        "packing_contract": manifest_binding["packing_contract"],
    }
    for field, value in expected.items():
        if group_binding.get(field) != value:
            raise RuntimeError(
                f"cross-split group/top-level {field.replace('_', ' ')} mismatch for {name}"
            )
    source = manifest_binding["source_family_inventory"].get(name)
    if source != {
        "family": group_binding["family"],
        "schema": group_binding["source_schema"],
    }:
        raise RuntimeError(f"cross-split source family binding mismatch for {name}")


def require_cross_split_finalization(
    output_root: str | Path,
    *,
    split: str,
    groups: Sequence[Mapping],
    manifest: Mapping,
) -> None:
    """Reject a train/val provenance splice before replacing final control files."""
    if split not in {"train", "val"}:
        raise RuntimeError(f"invalid token split {split!r}")
    output_root = Path(output_root)
    candidate_groups = {str(group.get("name")): group for group in groups}
    if len(candidate_groups) != len(groups) or set(candidate_groups) != set(
        manifest.get("groups", {})
    ):
        raise RuntimeError("cross-split candidate group inventory is not exact")
    candidate_manifest_binding = _token_manifest_cross_split_binding(manifest)
    for name, group in candidate_groups.items():
        group_binding = _group_cross_split_binding(group)
        _require_group_matches_token_manifest(
            group_binding,
            candidate_manifest_binding,
            name=name,
        )

    other_split = "val" if split == "train" else "train"
    other_meta_path = output_root / f"{other_split}_meta.json"
    if not other_meta_path.exists():
        return
    other_manifest = _read_json_object(other_meta_path, f"{other_split} token manifest")
    unsigned = dict(other_manifest)
    declared_root = unsigned.pop("manifest_sha256", None)
    if declared_root != fingerprint_dict(unsigned):
        raise RuntimeError(f"cross-split {other_split} token manifest seal is invalid")
    other_binding = _token_manifest_cross_split_binding(other_manifest)
    for field in (
        "declared_cross_split_schema",
        "corpus_schema_version",
        "code_version",
        "corpus_generation",
        "tokenizer_seal",
        "packing_contract",
        "source_family_inventory",
    ):
        if candidate_manifest_binding.get(field) != other_binding.get(field):
            label = field.replace("_", " ")
            raise RuntimeError(f"cross-split {label} binding mismatch")

    other_groups = other_manifest["groups"]
    if set(other_groups) != set(candidate_groups):
        raise RuntimeError("cross-split token group inventory mismatch")
    for name, candidate in candidate_groups.items():
        group_dir = output_root / "tokens" / name
        candidate_names = tuple(
            Path(str(shard.get("path", ""))).name
            for shard in candidate.get("shards", ())
            if isinstance(shard, Mapping)
        )
        require_exact_group_inventory(
            group_dir,
            pending_names=candidate_names,
        )
        other_done_path = group_dir / f"{other_split}.done.json"
        other_group = other_groups[name]
        if not isinstance(other_group, Mapping):
            raise TypeError(f"cross-split {other_split} group metadata is invalid for {name}")
        loaded_other = load_completed_group(
            other_done_path,
            fingerprint=str(other_group.get("fingerprint", "")),
            output_root=output_root,
            pending_names=candidate_names,
        )
        if loaded_other is None:
            raise RuntimeError(f"cross-split {other_split} completed group is missing for {name}")
        loaded_other_binding = _group_cross_split_binding(loaded_other)
        _require_group_matches_token_manifest(
            loaded_other_binding,
            other_binding,
            name=name,
        )
        _require_equal_cross_split_bindings(
            _group_cross_split_binding(candidate),
            loaded_other_binding,
            context=name,
        )


def _load_staged_token_manifest(path: str | Path, *, split: str) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"required {split} token manifest is missing or unsafe: {path}")
    manifest = _read_json_object(path, f"{split} token manifest")
    unsigned = dict(manifest)
    declared_sha256 = unsigned.pop("manifest_sha256", None)
    if declared_sha256 != fingerprint_dict(unsigned):
        raise RuntimeError(f"{split} token manifest SHA-256 seal is invalid")
    if set(manifest) != STAGED_TOKEN_MANIFEST_FIELDS:
        raise RuntimeError(
            f"{split} token manifest fields are not exact: "
            f"missing={sorted(STAGED_TOKEN_MANIFEST_FIELDS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - STAGED_TOKEN_MANIFEST_FIELDS)}"
        )
    if (
        manifest.get("schema_version") != PACKED_CORPUS_SCHEMA_VERSION
        or manifest.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION
        or manifest.get("cross_split_binding_schema_version") != CROSS_SPLIT_BINDING_SCHEMA_VERSION
        or manifest.get("split") != split
    ):
        raise RuntimeError(f"{split} token manifest schema/build binding is invalid")
    return manifest


def _require_fixed_staged_manifest_contract(manifest: Mapping, *, split: str) -> None:
    if manifest.get("tokenizer_seal") != FIXED_QWEN_TOKENIZER_SEAL:
        raise RuntimeError(f"{split} manifest does not use the fixed Qwen tokenizer seal")
    expected_top_level = {
        "sequence_length": 16_384,
        "tokens_dtype": TOKENS_DTYPE_NAME,
        "byte_order": TOKENS_BYTE_ORDER,
        "eos_token_id": 151_643,
        "pad_token_id": 151_643,
        "separator": SEPARATOR,
        "separator_search": SEPARATOR_SEARCH,
        "separator_ids": [10952, 15513, 969],
        "packed": True,
    }
    for field, expected in expected_top_level.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"{split} token manifest {field} is {manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    if (
        manifest.get("tokenizer_composite_sha256")
        != FIXED_QWEN_TOKENIZER_SEAL["tokenizer_composite_sha256"]
    ):
        raise RuntimeError(f"{split} token manifest tokenizer composite drift")

    generation = manifest.get("corpus_generation")
    if not isinstance(generation, Mapping):
        raise TypeError(f"{split} token manifest lacks corpus generation")
    if generation.get("schema_version") == SEALED_CORPUS_MANIFEST_SCHEMA_VERSION:
        # Direct sealed-manifest binding: no atomic transaction, so the identity is the
        # sealed corpus manifest root rather than a generation_id/CURRENT chain.
        if (
            set(generation) != STAGED_SEALED_CORPUS_GENERATION_FIELDS
            or generation.get("sealed_corpus_manifest") is not True
        ):
            raise RuntimeError(
                f"{split} token manifest sealed corpus generation identity is invalid"
            )
        for field in ("logical_root_sha256", "manifest_root_sha256"):
            _require_sha256(generation.get(field), f"{split} corpus generation {field}")
        if generation["logical_root_sha256"] != generation["manifest_root_sha256"]:
            raise RuntimeError(f"{split} sealed corpus generation roots disagree")
    else:
        if (
            set(generation) != STAGED_CORPUS_GENERATION_FIELDS
            or generation.get("schema_version") != CORPUS_BINDING_SCHEMA_VERSION
            or not isinstance(generation.get("generation_id"), str)
            or not generation["generation_id"]
        ):
            raise RuntimeError(f"{split} token manifest corpus generation identity is invalid")
        for field in (
            "logical_root_sha256",
            "manifest_root_sha256",
            "manifest_file_sha256",
            "current_sha256",
            "semantic_contract_sha256",
            "producer_source_sha256",
        ):
            _require_sha256(generation.get(field), f"{split} corpus generation {field}")

    source_inventory = manifest.get("source_family_inventory")
    source_hashes = manifest.get("source_jsonl_sha256")
    groups = manifest.get("groups")
    if (
        not isinstance(source_inventory, Mapping)
        or not isinstance(source_hashes, Mapping)
        or not isinstance(groups, Mapping)
        or set(source_inventory) != set(FAMILIES)
        or set(source_hashes) != set(FAMILIES)
        or set(groups) != set(FAMILIES)
    ):
        raise RuntimeError(f"{split} token manifest must contain the exact six-family inventory")
    for family in FAMILIES:
        if source_inventory[family] != {
            "family": family,
            "schema": P3_SOURCE_SCHEMAS[family],
        }:
            raise RuntimeError(f"{split} source family/schema binding drift for {family}")
        _require_sha256(source_hashes[family], f"{split} source JSONL {family}")

    packing = manifest.get("packing_config")
    if not isinstance(packing, Mapping):
        raise TypeError(f"{split} token manifest lacks packing configuration")
    if set(packing) != STAGED_PACKING_FIELDS:
        raise RuntimeError(f"{split} packing/build fields are not exact")
    expected_packing = {
        "algorithm": PACKED_ALGORITHM_VERSION,
        "split": split,
        "packed": True,
        "sequence_length": 16_384,
        "tokens_dtype": TOKENS_DTYPE_NAME,
        "byte_order": TOKENS_BYTE_ORDER,
        "eos_token_id": 151_643,
        "pad_token_id": 151_643,
        "separator": SEPARATOR,
        "separator_search": SEPARATOR_SEARCH,
        "separator_ids": [10952, 15513, 969],
    }
    for field, expected in expected_packing.items():
        if packing.get(field) != expected:
            raise RuntimeError(f"{split} packing/build binding drift for {field}")
    shard_tokens = packing.get("shard_tokens")
    if isinstance(shard_tokens, bool) or not isinstance(shard_tokens, int) or shard_tokens < 1:
        raise RuntimeError(f"{split} packing shard-token target is invalid")


def validate_staged_token_payload(
    staged_root: str | Path,
    *,
    train_manifest_path: str | Path,
    val_manifest_path: str | Path,
) -> dict:
    """
    Validate an explicitly staged, data-only P3 payload before publication.

    The stage must contain only the twelve-or-more packed shard files beneath the
    exact six family directories. The train/val control manifests stay outside the
    publish source, but seal every staged path, byte count, token count, and digest.
    """
    unresolved_root = Path(staged_root).expanduser()
    if not unresolved_root.is_dir() or unresolved_root.is_symlink():
        raise RuntimeError(f"staged token payload is missing or unsafe: {unresolved_root}")
    staged_root = unresolved_root.resolve()
    manifests = {
        split: _load_staged_token_manifest(path, split=split)
        for split, path in (
            ("train", train_manifest_path),
            ("val", val_manifest_path),
        )
    }
    for split, manifest in manifests.items():
        _require_fixed_staged_manifest_contract(manifest, split=split)

    train_binding = _token_manifest_cross_split_binding(manifests["train"])
    val_binding = _token_manifest_cross_split_binding(manifests["val"])
    for field in (
        "declared_cross_split_schema",
        "corpus_schema_version",
        "code_version",
        "corpus_generation",
        "tokenizer_seal",
        "packing_contract",
        "source_family_inventory",
    ):
        if train_binding[field] != val_binding[field]:
            raise RuntimeError(f"staged train/val {field.replace('_', ' ')} mismatch")

    if {path.name for path in staged_root.iterdir()} != {"tokens"}:
        raise RuntimeError("staged publish source must contain only the tokens payload group")
    tokens_root = staged_root / "tokens"
    if (
        not tokens_root.is_dir()
        or tokens_root.is_symlink()
        or {path.name for path in tokens_root.iterdir()} != set(FAMILIES)
    ):
        raise RuntimeError("staged tokens group must contain the exact six family labels")

    expected_paths = set()
    seen_sha256 = set()
    entries = []
    partitions = {
        "train": {"files": 0, "tokens": 0},
        "val": {"files": 0, "tokens": 0},
    }
    for split, manifest in manifests.items():
        groups = manifest["groups"]
        if manifest.get("instances") != sum(
            group.get("instances", -1) for group in groups.values()
        ):
            raise RuntimeError(f"{split} manifest instance arithmetic is invalid")
        if manifest.get("real_tokens") != sum(
            group.get("real_tokens", -1) for group in groups.values()
        ):
            raise RuntimeError(f"{split} manifest real-token arithmetic is invalid")
        if manifest.get("dropped_over_length") != sum(
            group.get("dropped_over_length", -1) for group in groups.values()
        ):
            raise RuntimeError(f"{split} manifest drop arithmetic is invalid")

        for family in FAMILIES:
            group = groups[family]
            if not isinstance(group, Mapping):
                raise TypeError(f"{split}/{family} packed group is invalid")
            if (
                set(group) != STAGED_GROUP_FIELDS
                or group.get("schema_version") != PACKED_GROUP_SCHEMA_VERSION
                or group.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION
                or group.get("cross_split_binding_schema_version")
                != CROSS_SPLIT_BINDING_SCHEMA_VERSION
                or group.get("name") != family
                or group.get("corpus_generation") != manifest["corpus_generation"]
            ):
                raise RuntimeError(f"{split}/{family} packed group binding is invalid")
            if group.get("fingerprint") != fingerprint_dict(group.get("build")):
                raise RuntimeError(f"{split}/{family} packed group fingerprint is invalid")
            if group.get("completion_sha256") != group_completion_sha256(group):
                raise RuntimeError(f"{split}/{family} packed group completion seal is invalid")
            build = group.get("build")
            if (
                not isinstance(build, Mapping)
                or set(build) != STAGED_BUILD_FIELDS
                or build.get("schema_version") != PACKED_GROUP_SCHEMA_VERSION
                or build.get("code_version") != TOKENIZE_CORPUS_CODE_VERSION
                or build.get("tokenizer") != FIXED_QWEN_TOKENIZER_SEAL
                or build.get("packing") != manifest["packing_config"]
                or build.get("corpus_generation") != manifest["corpus_generation"]
            ):
                raise RuntimeError(f"{split}/{family} packing/build binding is invalid")
            source = build.get("source_jsonl")
            if (
                not isinstance(source, Mapping)
                or set(source) != STAGED_SOURCE_FIELDS
                or source.get("family") != family
                or source.get("schema") != P3_SOURCE_SCHEMAS[family]
                or source.get("sha256") != manifest["source_jsonl_sha256"][family]
            ):
                raise RuntimeError(f"{split}/{family} source family/schema/hash binding is invalid")
            _require_group_matches_token_manifest(
                _group_cross_split_binding(group),
                _token_manifest_cross_split_binding(manifest),
                name=family,
            )

            shards = group.get("shards")
            if not isinstance(shards, list) or not shards:
                raise RuntimeError(f"{split}/{family} must contain one or more token shards")
            ordinals = []
            group_instances = 0
            for shard in shards:
                if not isinstance(shard, Mapping):
                    raise TypeError(f"{split}/{family} shard entry is invalid")
                if set(shard) != STAGED_SHARD_FIELDS:
                    raise RuntimeError(f"{split}/{family} shard fields are not exact")
                relative = shard.get("path")
                match = (
                    re.fullmatch(
                        rf"tokens/{re.escape(family)}/{split}-(\d{{5}})\.u32le\.bin",
                        relative,
                    )
                    if isinstance(relative, str)
                    else None
                )
                if match is None:
                    raise RuntimeError(f"{split}/{family} shard path is invalid: {relative!r}")
                ordinal = int(match.group(1))
                ordinals.append(ordinal)
                if relative in expected_paths:
                    raise RuntimeError(f"staged shard path is duplicated: {relative}")
                expected_paths.add(relative)
                path = staged_root / relative
                instances = shard.get("instances")
                tokens = shard.get("tokens")
                byte_count = shard.get("bytes")
                if (
                    isinstance(instances, bool)
                    or not isinstance(instances, int)
                    or instances < 1
                    or isinstance(tokens, bool)
                    or not isinstance(tokens, int)
                    or tokens != instances * 16_384
                    or isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count != tokens * TOKENS_STORAGE_DTYPE.itemsize
                    or shard.get("tokens_dtype") != TOKENS_DTYPE_NAME
                    or shard.get("byte_order") != TOKENS_BYTE_ORDER
                    or not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size != byte_count
                ):
                    raise RuntimeError(f"{relative}: staged shard arithmetic/format is invalid")
                digest = _require_sha256(shard.get("sha256"), f"{relative} SHA-256")
                if file_sha256(path) != digest:
                    raise RuntimeError(f"{relative}: staged shard SHA-256 mismatch")
                if digest in seen_sha256:
                    raise RuntimeError(f"{relative}: staged shard SHA-256 is duplicated")
                seen_sha256.add(digest)
                group_instances += instances
                partitions[split]["files"] += 1
                partitions[split]["tokens"] += tokens
                entries.append(
                    {
                        "path": relative,
                        "family": family,
                        "split": split,
                        "instances": instances,
                        "tokens": tokens,
                        "bytes": byte_count,
                        "sha256": digest,
                        "tokens_dtype": TOKENS_DTYPE_NAME,
                        "byte_order": TOKENS_BYTE_ORDER,
                    }
                )
            if ordinals != list(range(len(shards))):
                raise RuntimeError(f"{split}/{family} shard ordinals are not contiguous")
            if group.get("instances") != group_instances:
                raise RuntimeError(f"{split}/{family} shard instance arithmetic is invalid")
            capacity = group_instances * 16_384
            real_tokens = group.get("real_tokens")
            if (
                isinstance(real_tokens, bool)
                or not isinstance(real_tokens, int)
                or real_tokens < 1
                or real_tokens > capacity
            ):
                raise RuntimeError(f"{split}/{family} real-token arithmetic is invalid")

    actual_paths = set()
    for path in staged_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"staged token payload contains a forbidden symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(staged_root).as_posix())
    if actual_paths != expected_paths:
        raise RuntimeError(
            "staged token payload inventory is not exact: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    if any(partitions[split]["files"] < len(FAMILIES) for split in ("train", "val")):
        raise RuntimeError("staged payload must contain train and val partitions for all families")
    return {
        "families": list(FAMILIES),
        "tokenizer": {
            "dataset_id": FIXED_QWEN_TOKENIZER_DATASET_ID,
            "version": FIXED_QWEN_TOKENIZER_VERSION,
        },
        "entries": sorted(entries, key=lambda entry: entry["path"]),
        "partitions": partitions,
    }


def _profile_gate_require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"staged pretrain profile gate: {message}")


def _file_backed_fake_s3(fake_s3_type):
    """
    Return a network-free FakeS3 that range-reads real shards from local files.

    The package FakeS3 intentionally stores bytes in memory. A real P3 stage is
    multi-gigabyte, so retaining local file references preserves FakeS3 semantics
    without loading the complete payload into RAM. Publisher hashes still stream
    every file, and validator profile reads still use the package's seeded ranges.
    """

    class FileBackedFakeS3(fake_s3_type):
        def __init__(self):
            super().__init__()
            self._local_files: dict[tuple[str, str], Path] = {}
            self.range_reads: list[dict] = []

        def put(self, bucket, key, body, *, content_type=None):
            self._local_files.pop((bucket, key), None)
            super().put(bucket, key, body, content_type=content_type)

        def put_file(self, bucket, key, local_path):
            path = Path(local_path).resolve(strict=True)
            if not path.is_file():
                raise RuntimeError(f"FakeS3 source is not a regular file: {path}")
            self._store.pop((bucket, key), None)
            self._local_files[(bucket, key)] = path

        def get(self, bucket, key):
            path = self._local_files.get((bucket, key))
            if path is not None:
                if path.name.endswith(".u32le.bin"):
                    raise RuntimeError(
                        f"whole-object FakeS3 read is forbidden for token shard {key}"
                    )
                return path.read_bytes()
            return super().get(bucket, key)

        def get_range(self, bucket, key, start, length):
            path = self._local_files.get((bucket, key))
            if path is None:
                body = super().get_range(bucket, key, start, length)
            elif length <= 0:
                body = b""
                    else:
                with path.open("rb") as handle:
                    handle.seek(start)
                    body = handle.read(length)
            self.range_reads.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "start": start,
                    "requested_bytes": length,
                    "returned_bytes": len(body),
                }
            )
            return body

        def head(self, bucket, key):
            path = self._local_files.get((bucket, key))
            if path is None:
                return super().head(bucket, key)
            return {
                "size": path.stat().st_size,
                "crc64nvme": None,
                "etag": None,
                "content_type": None,
            }

        def list(self, bucket, prefix):
            listed = {item["key"]: item for item in super().list(bucket, prefix)}
            for (item_bucket, key), path in self._local_files.items():
                if item_bucket == bucket and key.startswith(prefix):
                    listed[key] = {"key": key, "size": path.stat().st_size}
            return list(listed.values())

        def hash_object(self, bucket, key):
            path = self._local_files.get((bucket, key))
            if path is None:
                return super().hash_object(bucket, key)
            return file_sha256(path), path.stat().st_size

        def copy(self, src_bucket, src_key, dst_bucket, dst_key):
            path = self._local_files.get((src_bucket, src_key))
            if path is None:
                self._local_files.pop((dst_bucket, dst_key), None)
                super().copy(src_bucket, src_key, dst_bucket, dst_key)
        else:
                self._store.pop((dst_bucket, dst_key), None)
                self._local_files[(dst_bucket, dst_key)] = path

        def delete(self, bucket, key):
            self._local_files.pop((bucket, key), None)
            super().delete(bucket, key)

    return FileBackedFakeS3()


def _profile_gate_policy(spec: Mapping) -> tuple[str, Path, dict]:
    label = spec.get("label")
    _profile_gate_require(isinstance(label, str) and label, "validator policy lacks a label")
    families_dir = Path(str(spec.get("families_dir", ""))).resolve(strict=True)
    policy_path = families_dir / "pretrain.json"
    _profile_gate_require(policy_path.is_file(), f"{label} lacks pretrain.json")
    expected_sha256 = spec.get("pretrain_policy_sha256")
    _profile_gate_require(
        isinstance(expected_sha256, str) and file_sha256(policy_path) == expected_sha256,
        f"{label} pretrain policy SHA-256 mismatch",
    )
    policy = _read_json_object(policy_path, f"{label} pretrain policy")
    floor = policy.get("defaults", {}).get("decode_smoke_test", {}).get("distinct_ids_min")
    _profile_gate_require(
        floor == spec.get("distinct_ids_min"),
        f"{label} distinct-ID floor is {floor!r}, expected {spec.get('distinct_ids_min')!r}",
    )
    identity = {key: value for key, value in spec.items() if key != "families_dir"}
    return label, families_dir, identity


def _profile_gate_violation_index(validation: Mapping) -> dict[tuple[str, str | None], dict]:
    return {
        (str(item.get("code")), item.get("path")): {
            "code": str(item.get("code")),
            "path": item.get("path"),
        }
        for item in validation.get("violations", ())
        if isinstance(item, Mapping)
    }


def run_staged_pretrain_profile_gate(
    staged_root: str | Path,
    *,
    train_manifest_path: str | Path,
    val_manifest_path: str | Path,
    tokenizer_source: str | Path,
    scratch_root: str | Path,
    edullm_modules,
    validator_policies: Sequence[Mapping],
    payload_kind: str,
) -> dict:
    """
    Run pinned ``publish()`` and ``validate_dataset()`` against staged P3 bytes.

    This is an offline pre-publication gate. It publishes the fixed local Qwen
    tokenizer and the actual shard-only stage into a file-backed package FakeS3,
    then runs every registered ``pretrain-tokens/v1`` check under the image-pinned
    and deployed family policies. It never contacts S3 or promotes a real dataset.

    ``payload_kind="synthetic"`` exists only for hostile regression fixtures and
    is always reported as non-authorizing. A real report also remains subject to
    explicit human review before any landing-bucket upload.
    """
    _profile_gate_require(
        payload_kind in {"real", "synthetic"},
        f"payload_kind must be 'real' or 'synthetic', got {payload_kind!r}",
    )
    staged_root = Path(staged_root).expanduser()
    train_manifest_path = Path(train_manifest_path).expanduser()
    val_manifest_path = Path(val_manifest_path).expanduser()
    expected = validate_staged_token_payload(
        staged_root,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
    )
    staged_root = staged_root.resolve(strict=True)
    train_manifest_path = train_manifest_path.resolve(strict=True)
    val_manifest_path = val_manifest_path.resolve(strict=True)
    expected_entries = {entry["path"]: entry for entry in expected["entries"]}

    policies = [_profile_gate_policy(spec) for spec in validator_policies]
    _profile_gate_require(
        [label for label, _, _ in policies] == ["image-pinned-local-256", "deployed-policy-128"],
        "validator policies must be image-pinned-local-256 then deployed-policy-128",
    )

    tokenizer_source = Path(tokenizer_source).expanduser().resolve(strict=True)
    _profile_gate_require(
        tokenizer_source.is_dir() and not tokenizer_source.is_symlink(),
        f"fixed tokenizer source is missing or unsafe: {tokenizer_source}",
    )
    scratch_root = Path(scratch_root).expanduser()
    _profile_gate_require(
        not scratch_root.exists() and not scratch_root.is_symlink(),
        f"scratch root must not already exist: {scratch_root}",
    )
    tokenizer_group = scratch_root / "tokenizer-payload" / "files"
    tokenizer_group.mkdir(parents=True)
    tokenizer_hashes = {}
    for filename, expected_sha256 in FIXED_QWEN_TOKENIZER_SEAL["tokenizer_file_sha256"].items():
        source_path = tokenizer_source / filename
        _profile_gate_require(
            source_path.is_file() and not source_path.is_symlink(),
            f"fixed tokenizer file is missing or unsafe: {source_path}",
        )
        payload = source_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        _profile_gate_require(
            digest == expected_sha256,
            f"fixed tokenizer file hash drift for {filename}",
        )
        (tokenizer_group / filename).write_bytes(payload)
        tokenizer_hashes[filename] = digest

    registry = edullm_modules.registry
    profile = registry.get_profile("pretrain-tokens/v1")
    profile_checks = [check.__name__ for check in profile.CHECKS]
    _profile_gate_require(
        profile_checks
        == [
            "check_entries_declare_token_counts",
            "check_decode_smoke",
            "check_first_bytes_not_npy",
            "check_seq_len_alignment",
        ],
        f"pinned pretrain profile checks drifted: {profile_checks}",
    )
    derived_tokenizer = registry.get_profile("tokenizer/v1").derive_vocab(
        (tokenizer_group / "tokenizer.json").read_bytes()
    )
    _profile_gate_require(
        derived_tokenizer == {"vocab_size": 151_665, "eos_token_id": 151_643},
        f"fixed tokenizer derived facts drifted: {derived_tokenizer}",
    )

    s3 = _file_backed_fake_s3(edullm_modules.s3.FakeS3)
    fixed_env = {
        "EDULLM_CODE_SHA256": "a" * 64,
        "EDULLM_PACKAGES_LOCK_SHA256": "b" * 64,
    }
    previous_families_dir = edullm_modules.validate.FAMILIES_DIR
    try:
        edullm_modules.validate.FAMILIES_DIR = policies[0][1]
        tokenizer_plan = edullm_modules.publish.publish(
            tokenizer_group.parent,
            dataset_id=FIXED_QWEN_TOKENIZER_DATASET_ID,
            purpose="Fixed published Qwen2.5 tokenizer used to encode the P3 pretraining corpus",
            profile="tokenizer/v1",
            s3=s3,
            created_at="2026-08-03T23:00:00Z",
            env=fixed_env,
        )
        tokenizer_prefix = f"{tokenizer_plan.dataset_id}/{tokenizer_plan.version}"
        tokenizer_validation = edullm_modules.validate.validate_dataset(
            "edullm-landing",
            tokenizer_prefix,
            s3,
            data_bucket="edullm-data",
        )
        _profile_gate_require(
            tokenizer_validation.ok,
            f"fixed tokenizer FakeS3 validation failed: {tokenizer_validation.report()}",
        )
        edullm_modules.validate.promote(
            tokenizer_validation,
            s3,
            data_bucket="edullm-data",
            landing_bucket="edullm-landing",
        )
        tokenizer_dataset = json.loads(s3.get("edullm-data", f"{tokenizer_prefix}/dataset.json"))
        tokenizer_manifest_sha256 = tokenizer_dataset["groups"][0]["manifest_sha256"]

        plan = edullm_modules.publish.publish(
            staged_root,
            dataset_id="pretrain/formal-proof-premises-500m",
            purpose=(
                "Packed formal-proof premise tokens for OLMo pretraining to compare dense "
                "and memory-split learning"
            ),
            profile="pretrain-tokens/v1",
            tokenizer=f"{FIXED_QWEN_TOKENIZER_DATASET_ID}/{FIXED_QWEN_TOKENIZER_VERSION}",
            group_meta={"tokens": {"seq_len": 16_384}},
            s3=s3,
            created_at="2026-08-03T23:00:00Z",
            env=fixed_env,
        )
        prefix = f"{plan.dataset_id}/{plan.version}"
        dataset_bytes = s3.get("edullm-landing", f"{prefix}/dataset.json")
        manifest_bytes = s3.get("edullm-landing", f"{prefix}/tokens/manifest.json")
        dataset = json.loads(dataset_bytes)
        manifest = json.loads(manifest_bytes)

        _profile_gate_require(len(dataset.get("groups", ())) == 1, "expected one tokens group")
        group = dataset["groups"][0]
        _profile_gate_require(
            group.get("name") == "tokens"
            and group.get("profile") == "pretrain-tokens/v1"
            and group.get("seq_len") == 16_384,
            "published tokens group metadata drifted",
        )
        _profile_gate_require(
            group.get("depends_on")
            == [
                {
                    "role": "tokenizer",
                    "dataset_id": FIXED_QWEN_TOKENIZER_DATASET_ID,
                    "version": FIXED_QWEN_TOKENIZER_VERSION,
                    "manifest_sha256": tokenizer_manifest_sha256,
                }
            ],
            "fixed tokenizer must be the sole dataset dependency",
        )
        _profile_gate_require(
            "evaluator" not in json.dumps(dataset).lower(),
            "published dataset contains a forbidden evaluator reference",
        )
        _profile_gate_require(
            {partition["name"]: partition["rows"] for partition in group.get("partitions", ())}
            == {split: record["tokens"] for split, record in expected["partitions"].items()},
            "published train/val partition counts disagree with staged manifests",
        )
        _profile_gate_require(
            group.get("manifest_sha256") == edullm_modules.manifest.manifest_sha256(manifest),
            "published token manifest hash is invalid",
        )

        published_entries = {entry["path"]: entry for entry in manifest.get("entries", ())}
        _profile_gate_require(
            set(published_entries) == set(expected_entries),
            "published token manifest inventory differs from the staged manifests",
        )
        expected_format = {
            "container": "raw",
            "dtype": TOKENS_DTYPE_NAME,
            "byte_order": TOKENS_BYTE_ORDER,
            "header_bytes": 0,
            "codec": "none",
        }
        for path, entry in published_entries.items():
            staged_entry = expected_entries[path]
            _profile_gate_require(
                entry.get("labels") == {"source": staged_entry["family"]},
                f"{path}: published family label drifted",
            )
            _profile_gate_require(
                entry.get("split") == staged_entry["split"],
                f"{path}: published split drifted",
            )
            _profile_gate_require(
                entry.get("count") == {"unit": "tokens", "value": staged_entry["tokens"]},
                f"{path}: published token count drifted",
            )
            _profile_gate_require(
                entry.get("format") == expected_format,
                f"{path}: published uint32 little-endian format drifted",
            )
            _profile_gate_require(
                entry.get("bytes") == staged_entry["bytes"]
                and entry.get("sha256") == staged_entry["sha256"],
                f"{path}: published byte count or SHA-256 drifted",
            )

        policy_reports = {}
        for label, families_dir, identity in policies:
            edullm_modules.validate.FAMILIES_DIR = families_dir
            first_range_read = len(s3.range_reads)
            validation = edullm_modules.validate.validate_dataset(
                "edullm-landing",
                prefix,
                s3,
                data_bucket="edullm-data",
            )
            validation_report = validation.report()
            sampled_paths = Counter()
            sampled_bytes = 0
            for event in s3.range_reads[first_range_read:]:
                key = event["key"]
                marker = prefix + "/"
                relative = key[len(marker) :] if key.startswith(marker) else ""
                if relative in expected_entries:
                    sampled_paths[relative] += 1
                    sampled_bytes += event["returned_bytes"]
            _profile_gate_require(
                set(sampled_paths) == set(expected_entries),
                f"{label} did not sample every staged shard",
            )
            _profile_gate_require(
                all(count >= 5 for count in sampled_paths.values()),
                f"{label} did not execute all seeded byte-range profile checks",
            )
            policy_reports[label] = {
                **identity,
                "status": "PASS" if validation.ok else "REPORT",
                "validation": validation_report,
                "sampled_paths": dict(sorted(sampled_paths.items())),
                "sampled_range_reads": sum(sampled_paths.values()),
                "sampled_bytes": sampled_bytes,
            }
    finally:
        edullm_modules.validate.FAMILIES_DIR = previous_families_dir

    local_index = _profile_gate_violation_index(
        policy_reports["image-pinned-local-256"]["validation"]
    )
    deployed_index = _profile_gate_violation_index(
        policy_reports["deployed-policy-128"]["validation"]
    )
    only_local = [
        local_index[key]
        for key in sorted(
            local_index.keys() - deployed_index.keys(),
            key=lambda item: (item[0], item[1] or ""),
        )
    ]
    only_deployed = [
        deployed_index[key]
        for key in sorted(
            deployed_index.keys() - local_index.keys(),
            key=lambda item: (item[0], item[1] or ""),
        )
    ]
    outcomes_match = (
        policy_reports["image-pinned-local-256"]["validation"]["ok"]
        == policy_reports["deployed-policy-128"]["validation"]["ok"]
    )
    status = (
        "PASS"
        if all(policy["status"] == "PASS" for policy in policy_reports.values())
        and outcomes_match
        and not only_local
        and not only_deployed
        else "REPORT"
    )
    control_manifests = {}
    for split, path in (
        ("train", train_manifest_path),
        ("val", val_manifest_path),
    ):
        payload = _read_json_object(path, f"{split} token manifest")
        control_manifests[split] = {
            "file_sha256": file_sha256(path),
            "declared_manifest_sha256": payload["manifest_sha256"],
        }
    return {
        "status": status,
        "payload_kind": payload_kind,
        "authorizes_publication": False,
        "network_free": True,
        "publication_performed": False,
        "profile": "pretrain-tokens/v1",
        "profile_checks": profile_checks,
        "inputs": {
            "staged_root": str(staged_root),
            "train_manifest": str(train_manifest_path),
            "val_manifest": str(val_manifest_path),
            "tokenizer_source": str(tokenizer_source),
        },
        "edullm_data": {
            "version": edullm_modules.version,
            "source": str(edullm_modules.source),
            "available_profiles": registry.available(),
        },
        "families": expected["families"],
        "entries": expected["entries"],
        "partitions": expected["partitions"],
        "tokenizer": {
            **expected["tokenizer"],
            "file_sha256": tokenizer_hashes,
            "manifest_sha256": tokenizer_manifest_sha256,
            "derived": derived_tokenizer,
        },
        "hashes": {
            "control_manifests": control_manifests,
            "fake_published_dataset_json_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "fake_published_token_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "fake_published_token_manifest_sha256": group["manifest_sha256"],
        },
        "fake_published_inventory": dataset["inventory"],
        "fake_published_token_manifest": {
            "objects": manifest["objects"],
            "bytes": manifest["bytes"],
        },
        "policies": policy_reports,
        "policy_delta": {
            "outcomes_match": outcomes_match,
            "only_image-pinned-local-256": only_local,
            "only_deployed-policy-128": only_deployed,
        },
        "manual_review": (
            "Required before any S3 upload. PASS means both offline policies accepted the "
            "same actual bytes; REPORT or SKIP is never authorization."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source_group = ap.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--corpus-contract-root",
        help="immutable corpus transaction root containing CURRENT and generations/",
    )
    source_group.add_argument(
        "--sealed-corpus-manifest",
        help="production direct seam: a p3-sealed-corpus-manifest-v1 JSON that binds the "
        "exact six-family train/eval JSONLs by SHA-256 without a committed transaction",
    )
    source_group.add_argument(
        "--test-only-corpus-dir",
        help="TEST ONLY: legacy directory seam; requires --test-only-allow-unsealed-inputs",
    )
    ap.add_argument(
        "--test-only-allow-unsealed-inputs",
        action="store_true",
        help="TEST ONLY: permit fixture tokenizer/corpus inputs; never use for production",
    )
    ap.add_argument("--out", default="artifacts/public")
    ap.add_argument(
        "--split",
        default="train",
        choices=("train", "val"),
        help="select exact TRAIN or EVAL role records by sibling from the corpus contract",
    )
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="local approved tokenizer directory; mutable HF identifiers are forbidden",
    )
    ap.add_argument("--sequence-length", type=int, default=16384)
    ap.add_argument(
        "--shard-tokens",
        type=int,
        default=250_000_000,
        help="tokens per output shard; 250M x 4B = 1 GB",
    )
    ap.add_argument(
        "--pack",
        action="store_true",
        help="pack several EOS-terminated proofs per sequence",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="corpora processed concurrently; 2 fits the local 8 GB RAM budget",
    )
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=0,
        help="Rayon tokenizer threads per job; 0 divides available CPUs across jobs",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="documents per fast-tokenizer batch",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="persistent encoded-document cache; defaults to <out>/.token-cache",
    )
    ap.add_argument(
        "--only-corpus",
        action="append",
        default=[],
        help="process only named corpus (repeatable); normal resume should omit this",
    )
    ap.add_argument(
        "--suggest",
        action="store_true",
        help="build/reuse caches, report length distribution, write no token shards",
    )
    args = ap.parse_args()

    source_records: dict[str, dict] = {}
    if args.corpus_contract_root is not None:
        source_mode = "transaction"
        test_only = False
        corpus_contract = load_corpus_generation_contract(args.corpus_contract_root)
        require_corpus_generation_current(corpus_contract)
        for family, records in corpus_contract["families"].items():
            record = records[args.split]
            source_records[family] = {
                **record,
                "source": str(Path(corpus_contract["contract_root"]) / record["path"]),
            }
        corpus_generation = {
            key: corpus_contract[key]
            for key in (
                "schema_version",
                "generation_id",
                "logical_root_sha256",
                "manifest_root_sha256",
                "manifest_file_sha256",
                "current_sha256",
                "semantic_contract_sha256",
                "producer_source_sha256",
            )
        }
    elif args.sealed_corpus_manifest is not None:
        source_mode = "sealed"
        test_only = False
        sealed = load_sealed_corpus_manifest(args.sealed_corpus_manifest)
        role = "train" if args.split == "train" else "eval"
        for family in FAMILIES:
            entry = sealed["families"][family][role]
            source_path = Path(entry["path"]).expanduser()
            if source_path.is_symlink():
                sys.exit(f"sealed source must not be a symlink: {source_path}")
            if not source_path.is_file():
                sys.exit(f"sealed source JSONL is missing: {source_path}")
            if (
                file_sha256(source_path) != entry["sha256"]
                or source_path.stat().st_size != entry["bytes"]
            ):
                sys.exit(f"sealed source drifted from its seal: {family}/{role}")
            source_records[family] = {
                "source": str(source_path),
                "path": str(source_path),
                "sha256": entry["sha256"],
                "bytes": entry["bytes"],
                "family": family,
                "schema": sealed["families"][family]["schema"],
            }
        corpus_contract = {
            "schema_version": sealed["schema_version"],
            "sealed_corpus_manifest": True,
        }
        corpus_generation = {
            "schema_version": sealed["schema_version"],
            "logical_root_sha256": sealed["manifest_root_sha256"],
            "manifest_root_sha256": sealed["manifest_root_sha256"],
            "sealed_corpus_manifest": True,
        }
    else:
        source_mode = "test-only"
        test_only = bool(args.test_only_allow_unsealed_inputs)
        if not test_only:
            sys.exit(
                "--test-only-corpus-dir requires --test-only-allow-unsealed-inputs; "
                "production legacy directory discovery is forbidden"
            )
        legacy_root = Path(args.test_only_corpus_dir)
        sub = "shards" if args.split == "train" else "eval"
        for source in sorted((legacy_root / sub).glob("*.jsonl")):
            source_records[source.stem] = {
                "source": str(source),
                "path": str(source),
                "sha256": file_sha256(source),
                "bytes": source.stat().st_size,
            }
        corpus_contract = {
            "schema_version": CORPUS_BINDING_SCHEMA_VERSION,
            "test_only": True,
        }
        legacy_root_sha256 = fingerprint_dict(
            {
                name: {
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for name, record in source_records.items()
            }
        )
        corpus_generation = {
            "schema_version": CORPUS_BINDING_SCHEMA_VERSION,
            "test_only": True,
            "logical_root_sha256": legacy_root_sha256,
            "legacy_root_sha256": legacy_root_sha256,
        }
    all_source_names = set(source_records)
    if args.only_corpus:
        selected = set(args.only_corpus)
        missing = selected - set(source_records)
        if missing:
            sys.exit(f"unknown --only-corpus values: {sorted(missing)}")
        source_records = {
            name: record for name, record in source_records.items() if name in selected
        }
    if not source_records:
        sys.exit("the selected corpus contract exposes no source JSONL records")

    if args.sequence_length < 4:
        sys.exit("--sequence-length must leave room for the separator and a target")
    if args.jobs < 1 or args.batch_size < 1:
        sys.exit("--jobs and --batch-size must be positive")
    threads = args.threads_per_job or max(1, (os.cpu_count() or 1) // args.jobs)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(threads)

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_root / ".token-cache"
    common = {
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "split": args.split,
        "tokenizer": args.tokenizer,
        "sequence_length": args.sequence_length,
        "shard_tokens": args.shard_tokens,
        "pack": args.pack,
        "batch_size": args.batch_size,
        "suggest": args.suggest,
        "test_only": test_only,
        "source_mode": source_mode,
        "defer_done_commit": source_mode == "transaction",
        "corpus_contract": corpus_contract,
        "corpus_generation": corpus_generation,
    }
    tasks = [
        {
            **common,
            "name": name,
            "source": record["source"],
            "source_record": record,
        }
        for name, record in sorted(source_records.items())
    ]
    print(
        f"  {len(tasks)} corpora, {args.jobs} parallel jobs, "
        f"{threads} tokenizer threads/job, cache {cache_root}",
        flush=True,
    )

    results = []
    if args.jobs == 1:
        results = [process_corpus(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_corpus, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["name"])
    if source_mode == "transaction":
        require_corpus_generation_current(corpus_contract)

    if args.suggest:
        lengths = np.asarray(
            [length for result in results for length in result["lengths"]], dtype=np.int64
        )
        print(
            f"\n  {len(lengths):,} examples, median {int(np.median(lengths)):,}, "
            f"max {int(lengths.max()):,} tokens"
        )
        print(f"\n  {'seq_len':>9}{'kept':>10}{'%':>8}{'unpacked waste':>17}")
        for candidate in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
            fits = int((lengths <= candidate).sum())
            waste = 1 - lengths[lengths <= candidate].sum() / (fits * candidate) if fits else 1.0
            print(
                f"  {candidate:>9,}{fits:>10,}{100 * fits / len(lengths):>7.1f}%" f"{waste:>16.1%}"
            )
        return

    if args.only_corpus and {result["name"] for result in results} != all_source_names:
        print("  subset complete; run again without --only-corpus to finalize manifest")
        return

    if source_mode != "transaction":
        require_exact_token_output_inventory(
            output_root,
            expected_groups=all_source_names,
            cache_root=cache_root,
        )
    else:
        for result in results:
            require_exact_group_inventory(
                output_root / "tokens" / result["name"],
                pending_names=tuple(Path(shard["path"]).name for shard in result["shards"]),
            )
    first = results[0]
    first_build = first["build"]
    manifest = {
        "schema_version": PACKED_CORPUS_SCHEMA_VERSION,
        "code_version": TOKENIZE_CORPUS_CODE_VERSION,
        "cross_split_binding_schema_version": CROSS_SPLIT_BINDING_SCHEMA_VERSION,
        "sequence_length": args.sequence_length,
        "tokenizer": args.tokenizer,
        "tokenizer_seal": first_build["tokenizer"],
        "tokenizer_composite_sha256": first_build["tokenizer"]["tokenizer_composite_sha256"],
        "corpus_generation": corpus_generation,
        "source_jsonl_sha256": {
            result["name"]: result["build"]["source_jsonl"]["sha256"] for result in results
        },
        "source_family_inventory": {
            result["name"]: {
                "family": result["build"]["source_jsonl"]["family"],
                "schema": result["build"]["source_jsonl"]["schema"],
            }
            for result in results
        },
        "packing_config": first_build["packing"],
        "tokens_dtype": TOKENS_DTYPE_NAME,
        "byte_order": TOKENS_BYTE_ORDER,
        "eos_token_id": first["eos_token_id"],
        "pad_token_id": first["pad_token_id"],
        "separator": SEPARATOR,
        "separator_search": SEPARATOR_SEARCH,
        "separator_ids": first["separator_ids"],
        "split": args.split,
        "packed": bool(args.pack),
        "groups": {result["name"]: result for result in results},
    }
    for result in results[1:]:
        for field in ("eos_token_id", "pad_token_id", "separator_ids"):
            if result[field] != manifest[field]:
                raise RuntimeError(f"corpora disagree on tokenizer field {field}")
        if result["build"]["tokenizer"] != first_build["tokenizer"]:
            raise RuntimeError("corpora disagree on tokenizer composite seal")
        if result["build"]["packing"] != first_build["packing"]:
            raise RuntimeError("corpora disagree on packing configuration")
        if result["build"]["corpus_generation"] != corpus_generation:
            raise RuntimeError("corpora disagree on corpus generation binding")
    manifest["instances"] = sum(result["instances"] for result in results)
    manifest["real_tokens"] = sum(result["real_tokens"] for result in results)
    manifest["dropped_over_length"] = sum(result["dropped_over_length"] for result in results)
    manifest["tokens_straddling_boundary"] = sum(result["straddling"] for result in results)
    manifest["resumed_groups"] = sum(bool(result["resumed_group"]) for result in results)
    manifest["resumed_shards"] = sum(result["resumed_shards"] for result in results)
    manifest["manifest_sha256"] = fingerprint_dict(manifest)
    require_cross_split_finalization(
        output_root,
        split=args.split,
        groups=results,
        manifest=manifest,
    )
    meta_path = output_root / f"{args.split}_meta.json"
    if source_mode != "transaction":
        atomic_write_json(meta_path, manifest)
    else:
        specifications = []
        for result in results:
            if result["resumed_group"]:
                continue
            done_path = output_root / "tokens" / result["name"] / f"{args.split}.done.json"
            specifications.append(
                (
                    done_path.with_name(f".{done_path.name}.pending"),
                    done_path,
                    result,
                )
            )
        specifications.append(
            (
                meta_path.with_name(f".{meta_path.name}.pending"),
                meta_path,
                manifest,
            )
        )
        finalize_token_manifests_under_generation_lock(
            corpus_contract,
            specifications,
        )
    require_exact_token_output_inventory(
        output_root,
        expected_groups=all_source_names,
        cache_root=cache_root,
    )

    total = manifest["instances"] * args.sequence_length
    padding = 1 - manifest["real_tokens"] / max(total, 1)
    print(
        f"\n  {manifest['instances']:,} instances x {args.sequence_length:,} = "
        f"{total / 1e6:,.1f}M compute tokens, "
        f"{manifest['real_tokens'] / 1e6:,.1f}M real ({padding:.2%} padding)"
    )
    print(
        f"  dropped {manifest['dropped_over_length']:,} over-length documents; "
        f"resumed {manifest['resumed_groups']} groups and "
        f"{manifest['resumed_shards']} individual shards"
    )
    print(f"  wrote {output_root / f'{args.split}_meta.json'}", flush=True)


if __name__ == "__main__":
    main()
