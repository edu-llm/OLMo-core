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
same command validates and skips completed groups/shards. A wrong-sized existing shard
or mismatched fingerprint is preserved and refused, never deleted or overwritten.

Usage:
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus <dir> --suggest
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus <dir> \
        --out artifacts/public --sequence-length 16384 \
        --tokenizer ../memorysplit-requery-exact/tokenizers/qwen25-vendored \
        --pack --jobs 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

TOKENS_DTYPE = np.uint32  # vocab is 151,936, so uint16 is not an option
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
            total_straddling += sum(
                1 for start, end in offsets if start < row["mask_end"] < end
            )
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

    A wrong-sized existing file is preserved and refused. The caller can move or
    inspect it; this function never deletes or overwrites partial generated data.
    """
    path = Path(path)
    expected = len(documents) * sequence_length * np.dtype("<u4").itemsize
    if path.exists():
        actual = path.stat().st_size
        if actual == expected:
            return {"bytes": expected, "resumed": True}
        raise RuntimeError(
            f"{path} exists with {actual:,} bytes, expected {expected:,}; "
            "the partial file was preserved. Move it aside or resume from a clean "
            "output prefix after inspecting it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    try:
        with open(partial, "wb") as fh:
            for lo in range(0, len(documents), write_batch):
                block = documents[lo : lo + write_batch]
                buf = np.full(
                    (len(block), sequence_length), pad_id, dtype=np.dtype("<u4")
                )
                for j, ids in enumerate(block):
                    buf[j, : len(ids)] = ids
                buf.tofile(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, path)
    except BaseException:
        # Keep the `.partial-*` file for diagnosis/resume planning.
        raise
    return {"bytes": expected, "resumed": False}


def atomic_write_json(path: str | Path, payload: dict) -> None:
    """Write a completion/control file last, then publish it with one rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(partial, path)


def save_encoding_cache(
    cache_dir: str | Path,
    documents: Sequence[np.ndarray],
    *,
    fingerprint: str,
    straddling: int,
) -> None:
    """Persist EOS-terminated document IDs once so pass two and resumes are cheap."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.u32le.bin"
    offsets_path = cache_dir / "offsets.u64le.bin"
    marker = cache_dir / "cache.json"

    # Never overwrite a cache marker. A caller with a different fingerprint gets
    # `None` from load_encoding_cache and should choose a new fingerprinted directory.
    if marker.exists():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("fingerprint") == fingerprint:
            return
        raise RuntimeError(
            f"{cache_dir} contains a cache for another fingerprint; preserved. "
            "Choose a new cache directory rather than deleting it."
        )

    token_tmp = tokens_path.with_name(f"{tokens_path.name}.partial-{os.getpid()}")
    offset_tmp = offsets_path.with_name(f"{offsets_path.name}.partial-{os.getpid()}")
    offsets = np.empty(len(documents) + 1, dtype="<u8")
    offsets[0] = 0
    with open(token_tmp, "wb") as token_fh:
        cursor = 0
        for i, ids in enumerate(documents, 1):
            np.asarray(ids, dtype="<u4").tofile(token_fh)
            cursor += len(ids)
            offsets[i] = cursor
        token_fh.flush()
        os.fsync(token_fh.fileno())
    offsets.tofile(offset_tmp)
    os.replace(token_tmp, tokens_path)
    os.replace(offset_tmp, offsets_path)
    atomic_write_json(
        marker,
        {
            "fingerprint": fingerprint,
            "documents": len(documents),
            "tokens": int(offsets[-1]),
            "straddling": int(straddling),
        },
    )


def load_encoding_cache(
    cache_dir: str | Path, *, fingerprint: str
) -> tuple[list[np.ndarray], dict] | None:
    """Return memory-mapped document views, or None for an absent/stale cache."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / "cache.json"
    if not marker.exists():
        return None
    stats = json.loads(marker.read_text(encoding="utf-8"))
    if stats.get("fingerprint") != fingerprint:
        return None
    tokens_path = cache_dir / "tokens.u32le.bin"
    offsets_path = cache_dir / "offsets.u64le.bin"
    if not tokens_path.exists() or not offsets_path.exists():
        raise RuntimeError(f"{cache_dir} has a completion marker but missing cache bytes; preserved")
    offsets = np.memmap(offsets_path, mode="r", dtype="<u8")
    tokens = np.memmap(tokens_path, mode="r", dtype="<u4")
    if len(offsets) != int(stats["documents"]) + 1 or int(offsets[-1]) != len(tokens):
        raise RuntimeError(f"{cache_dir} cache sizes disagree with its marker; preserved")
    documents = [tokens[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(offsets) - 1)]
    return documents, stats


def load_completed_group(
    marker_path: str | Path, *, fingerprint: str, output_root: str | Path
) -> dict | None:
    """Validate a per-corpus completion marker and return it for manifest assembly."""
    marker_path = Path(marker_path)
    if not marker_path.exists():
        return None
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{marker_path} has a different fingerprint; existing output was preserved"
        )
    output_root = Path(output_root)
    for shard in payload.get("shards", []):
        path = output_root / shard["path"]
        actual = path.stat().st_size if path.exists() else -1
        if actual != int(shard["bytes"]):
            raise RuntimeError(
                f"{path} has {actual:,} bytes, expected {int(shard['bytes']):,}; "
                "existing output was preserved"
            )
    return payload


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_encoding_cache_from_jsonl(
    tok,
    source: str | Path,
    cache_dir: str | Path,
    *,
    fingerprint: str,
    eos_id: int,
    batch_size: int,
) -> tuple[list[np.ndarray], dict]:
    """Stream JSONL through batch encoding into a resumable on-disk cache."""
    cache_dir = Path(cache_dir)
    cached = load_encoding_cache(cache_dir, fingerprint=fingerprint)
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.u32le.bin"
    offsets_path = cache_dir / "offsets.u64le.bin"
    token_tmp = cache_dir / "tokens.u32le.bin.partial"
    offset_tmp = cache_dir / "offsets.u64le.bin.partial"
    progress_path = cache_dir / "progress.json"

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"{cache_dir} contains partial bytes for another fingerprint; preserved"
            )
        n_documents = int(progress["documents"])
        token_count = int(progress["tokens"])
        straddling = int(progress["straddling"])
        expected_token_bytes = token_count * np.dtype("<u4").itemsize
        expected_offset_bytes = (n_documents + 1) * np.dtype("<u8").itemsize
        if (
            not token_tmp.exists()
            or not offset_tmp.exists()
            or token_tmp.stat().st_size != expected_token_bytes
            or offset_tmp.stat().st_size != expected_offset_bytes
        ):
            raise RuntimeError(
                f"{cache_dir} partial cache disagrees with progress.json; preserved"
            )
    else:
        if token_tmp.exists() or offset_tmp.exists():
            raise RuntimeError(
                f"{cache_dir} contains partial bytes without progress.json; preserved"
            )
        n_documents = token_count = straddling = 0
        token_tmp.touch()
        np.asarray([0], dtype="<u8").tofile(offset_tmp)
        atomic_write_json(
            progress_path,
            {
                "fingerprint": fingerprint,
                "documents": 0,
                "tokens": 0,
                "straddling": 0,
            },
        )

    def commit_batch(rows, token_fh, offset_fh) -> None:
        nonlocal n_documents, token_count, straddling
        if not rows:
            return
        encoded, crossed = encode_rows_batched(
            tok, rows, eos_id=eos_id, batch_size=batch_size
        )
        new_offsets = np.empty(len(encoded), dtype="<u8")
        for i, ids in enumerate(encoded):
            ids.astype("<u4", copy=False).tofile(token_fh)
            token_count += len(ids)
            new_offsets[i] = token_count
        new_offsets.tofile(offset_fh)
        n_documents += len(encoded)
        straddling += crossed
        token_fh.flush()
        offset_fh.flush()
        os.fsync(token_fh.fileno())
        os.fsync(offset_fh.fileno())
        atomic_write_json(
            progress_path,
            {
                "fingerprint": fingerprint,
                "documents": n_documents,
                "tokens": token_count,
                "straddling": straddling,
            },
        )

    with (
        open(source, encoding="utf-8") as source_fh,
        open(token_tmp, "ab") as token_fh,
        open(offset_tmp, "ab") as offset_fh,
    ):
        rows = []
        source_index = 0
        for line in source_fh:
            if not line.strip():
                continue
            if source_index < n_documents:
                source_index += 1
                continue
            rows.append(json.loads(line))
            source_index += 1
            if len(rows) >= batch_size:
                commit_batch(rows, token_fh, offset_fh)
                rows.clear()
        commit_batch(rows, token_fh, offset_fh)

    os.replace(token_tmp, tokens_path)
    os.replace(offset_tmp, offsets_path)
    atomic_write_json(
        cache_dir / "cache.json",
        {
            "fingerprint": fingerprint,
            "documents": n_documents,
            "tokens": token_count,
            "straddling": straddling,
        },
    )
    result = load_encoding_cache(cache_dir, fingerprint=fingerprint)
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


def process_corpus(task: dict) -> dict:
    """Encode, cache, pack, and shard one corpus. Safe to run in another process."""
    from transformers import AutoTokenizer

    source = Path(task["source"])
    name = source.stem
    output_root = Path(task["output_root"])
    split = task["split"]
    sequence_length = int(task["sequence_length"])
    tokenizer_id = task["tokenizer"]

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if not tok.is_fast:
        raise RuntimeError("need a fast tokenizer for parallel batches and offsets")
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"{tokenizer_id} has no eos token")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    sep_ids = tok(SEPARATOR_SEARCH, add_special_tokens=False)["input_ids"]
    if not sep_ids:
        raise RuntimeError(f"{SEPARATOR_SEARCH!r} tokenizes to nothing")

    cache_fingerprint = fingerprint_dict(
        {
            "version": 1,
            "source_sha256": file_sha256(source),
            "tokenizer_sha256": tokenizer_digest(tok),
            "eos_token_id": int(eos_id),
        }
    )
    group_fingerprint = fingerprint_dict(
        {
            "version": 2,  # bump whenever packing or shard semantics change
            "cache": cache_fingerprint,
            "sequence_length": sequence_length,
            "packed": bool(task["pack"]),
            "shard_tokens": int(task["shard_tokens"]),
            "pad_token_id": int(pad_id),
            "split": split,
        }
    )
    group_dir = output_root / "tokens" / name
    done_path = group_dir / f"{split}.done.json"
    done = load_completed_group(
        done_path, fingerprint=group_fingerprint, output_root=output_root
    )
    if done is not None:
        done["resumed_group"] = True
        print(f"  {name:<12}complete group validated and resumed", flush=True)
        return done

    cache_dir = (
        Path(task["cache_root"]) / split / name / cache_fingerprint[:20]
    )
    documents, cache_stats = build_encoding_cache_from_jsonl(
        tok,
        source,
        cache_dir,
        fingerprint=cache_fingerprint,
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
    existing_names = {
        path.name for path in group_dir.glob(f"{split}-*.u32le.bin")
    }
    extras = existing_names - expected_names
    if extras:
        raise RuntimeError(
            f"{group_dir} has unexpected shards {sorted(extras)}; preserved. "
            "Move them aside before resuming this configuration."
        )

    shards = []
    resumed_shards = 0
    for ordinal, lo in enumerate(range(0, len(packed_rows), per_shard)):
        rows = packed_rows[lo : lo + per_shard]
        instances = [
            np.concatenate([documents[i] for i in row]) for row in rows
        ]
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
                "tokens": len(instances) * sequence_length,
                "bytes": result["bytes"],
            }
        )
        del instances

    real_tokens = sum(kept_lengths)
    payload = {
        "fingerprint": group_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "name": name,
        "documents": len(kept_indices),
        "source_documents": len(lengths),
        "instances": len(packed_rows),
        "real_tokens": real_tokens,
        "padding_fraction": round(
            1 - real_tokens / max(len(packed_rows) * sequence_length, 1), 6
        ),
        "dropped_over_length": dropped,
        "straddling": int(cache_stats["straddling"]),
        "separator_ids": list(sep_ids),
        "eos_token_id": int(eos_id),
        "pad_token_id": int(pad_id),
        "shards": shards,
        "resumed_shards": resumed_shards,
        "resumed_group": False,
    }
    atomic_write_json(done_path, payload)
    print(
        f"  {name:<12}{len(kept_indices):>8,} docs -> "
        f"{len(packed_rows):>7,} instances, {payload['padding_fraction']:.1%} "
        f"padding, {dropped:,} dropped, {resumed_shards} shards resumed",
        flush=True,
    )
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus", default="corpus", help="dir holding shards/ and eval/ of *.jsonl"
    )
    ap.add_argument("--out", default="artifacts/public")
    ap.add_argument(
        "--split",
        default="train",
        choices=("train", "val"),
        help="train reads corpus/shards/, val reads corpus/eval/",
    )
    ap.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-0.5B",
        help="publish this same tokenizer as tokenizer/qwen25-vendored",
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

    sub = "shards" if args.split == "train" else "eval"
    srcs = sorted(glob.glob(os.path.join(args.corpus, sub, "*.jsonl")))
    if args.only_corpus:
        selected = set(args.only_corpus)
        srcs = [src for src in srcs if Path(src).stem in selected]
        missing = selected - {Path(src).stem for src in srcs}
        if missing:
            sys.exit(f"unknown --only-corpus values: {sorted(missing)}")
    if not srcs:
        sys.exit(f"no *.jsonl under {os.path.join(args.corpus, sub)}")

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
    }
    tasks = [{**common, "source": src} for src in srcs]
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
            waste = (
                1 - lengths[lengths <= candidate].sum() / (fits * candidate)
                if fits
                else 1.0
            )
            print(
                f"  {candidate:>9,}{fits:>10,}{100 * fits / len(lengths):>7.1f}%"
                f"{waste:>16.1%}"
            )
        return

    if args.only_corpus and len(results) != len(
        glob.glob(os.path.join(args.corpus, sub, "*.jsonl"))
    ):
        print("  subset complete; run again without --only-corpus to finalize manifest")
        return

    first = results[0]
    manifest = {
        "sequence_length": args.sequence_length,
        "tokenizer": args.tokenizer,
        "tokens_dtype": "uint32",
        "byte_order": "little",
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
    manifest["instances"] = sum(result["instances"] for result in results)
    manifest["real_tokens"] = sum(result["real_tokens"] for result in results)
    manifest["dropped_over_length"] = sum(
        result["dropped_over_length"] for result in results
    )
    manifest["tokens_straddling_boundary"] = sum(
        result["straddling"] for result in results
    )
    manifest["resumed_groups"] = sum(bool(result["resumed_group"]) for result in results)
    manifest["resumed_shards"] = sum(result["resumed_shards"] for result in results)
    atomic_write_json(output_root / f"{args.split}_meta.json", manifest)

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
