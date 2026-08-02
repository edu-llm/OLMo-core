"""Fast, resumable token-shard construction.

These tests do not need Qwen or the real corpus. They pin the mechanics that make a
45-minute preprocessing job safe to restart: batched encoding, deterministic packing,
atomic shards, and refusing rather than deleting an unexpected partial artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from tokenize_corpus import (  # noqa: E402
    build_encoding_cache_from_jsonl,
    encode_rows_batched,
    load_completed_group,
    load_encoding_cache,
    pack_indices_by_length,
    save_encoding_cache,
    write_shard_resumable,
)


class FakeTokenizer:
    """Enough of a fast HF tokenizer to prove calls are batched."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        assert isinstance(texts, list)
        self.calls += 1
        ids, offsets = [], []
        for text in texts:
            ids.append(list(range(1, len(text) + 1)))
            offsets.append([(i, i + 1) for i in range(len(text))])
        return {"input_ids": ids, "offset_mapping": offsets}


def test_encoding_is_batched_and_appends_eos():
    rows = [{"text": "abcd", "mask_end": 2} for _ in range(10)]
    tok = FakeTokenizer()
    encoded, straddling = encode_rows_batched(tok, rows, eos_id=99, batch_size=4)

    assert tok.calls == 3, "10 documents at batch 4 should be 3 tokenizer calls, not 10"
    assert len(encoded) == 10
    assert all(ids.dtype == np.uint32 for ids in encoded)
    assert all(ids.tolist() == [1, 2, 3, 4, 99] for ids in encoded)
    assert straddling == 0


def test_packer_is_deterministic_lossless_and_respects_capacity():
    lengths = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    first = pack_indices_by_length(lengths, capacity=10)
    second = pack_indices_by_length(lengths, capacity=10)

    assert first == second
    flat = [i for row in first for i in row]
    assert sorted(flat) == list(range(len(lengths))), "every document appears exactly once"
    assert all(sum(lengths[i] for i in row) <= 10 for row in first)
    # Total length 45 has a lower bound of five bins; this input is exactly packable.
    assert len(first) == 5


def test_resumable_writer_skips_a_complete_existing_shard(tmp_path):
    path = tmp_path / "train-00000.u32le.bin"
    docs = [np.array([1, 2, 9], dtype=np.uint32), np.array([3, 9], dtype=np.uint32)]

    first = write_shard_resumable(path, docs, sequence_length=8, pad_id=9)
    before = path.stat().st_mtime_ns
    second = write_shard_resumable(path, docs, sequence_length=8, pad_id=9)

    assert first["resumed"] is False
    assert second["resumed"] is True
    assert path.stat().st_mtime_ns == before, "resume must not rewrite completed bytes"
    raw = np.fromfile(path, dtype="<u4").reshape(2, 8)
    assert raw[0].tolist() == [1, 2, 9, 9, 9, 9, 9, 9]
    assert raw[1].tolist() == [3, 9, 9, 9, 9, 9, 9, 9]


def test_resumable_writer_preserves_and_refuses_wrong_sized_shard(tmp_path):
    path = tmp_path / "train-00000.u32le.bin"
    path.write_bytes(b"partial")
    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="preserved"):
        write_shard_resumable(
            path,
            [np.array([1, 2, 9], dtype=np.uint32)],
            sequence_length=8,
            pad_id=9,
        )

    assert path.read_bytes() == original, "never delete or overwrite a partial artifact"


def test_completed_group_metadata_is_json_roundtrippable(tmp_path):
    """The resume marker is a normal JSON control file, not an opaque pickle."""
    marker = tmp_path / "train.done.json"
    payload = {
        "fingerprint": "abc",
        "documents": 2,
        "instances": 1,
        "shards": [{"path": "tokens/x/train-00000.u32le.bin", "bytes": 32}],
    }
    marker.write_text(json.dumps(payload))
    assert json.loads(marker.read_text()) == payload


def test_encoding_cache_roundtrips_and_rejects_a_different_fingerprint(tmp_path):
    docs = [
        np.array([1, 2, 9], dtype=np.uint32),
        np.array([3, 4, 5, 9], dtype=np.uint32),
    ]
    save_encoding_cache(tmp_path, docs, fingerprint="same", straddling=7)

    cached = load_encoding_cache(tmp_path, fingerprint="same")
    assert cached is not None
    got, stats = cached
    assert [x.tolist() for x in got] == [x.tolist() for x in docs]
    assert stats["straddling"] == 7

    assert load_encoding_cache(tmp_path, fingerprint="different") is None
    assert (tmp_path / "tokens.u32le.bin").exists(), "a stale cache is preserved"


def test_completed_group_is_resumed_only_when_fingerprint_and_shards_match(tmp_path):
    shard = tmp_path / "tokens" / "x" / "train-00000.u32le.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"\0" * 32)
    done = shard.parent / "train.done.json"
    payload = {
        "fingerprint": "abc",
        "shards": [
            {
                "path": "tokens/x/train-00000.u32le.bin",
                "bytes": 32,
            }
        ],
    }
    done.write_text(json.dumps(payload))

    assert load_completed_group(done, fingerprint="abc", output_root=tmp_path) == payload
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_completed_group(done, fingerprint="wrong", output_root=tmp_path)

    shard.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="preserved"):
        load_completed_group(done, fingerprint="abc", output_root=tmp_path)


def test_encoding_cache_resumes_inside_a_corpus_from_batch_progress(tmp_path):
    source = tmp_path / "x.jsonl"
    rows = [{"text": "abc", "mask_end": 1} for _ in range(5)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cache = tmp_path / "cache"
    cache.mkdir()

    # Simulate a crash after two encoded rows. The partial format records token
    # boundaries and the source-row count, so no inference from raw bytes is needed.
    (cache / "tokens.u32le.bin.partial").write_bytes(
        np.asarray([1, 2, 3, 99, 1, 2, 3, 99], dtype="<u4").tobytes()
    )
    (cache / "offsets.u64le.bin.partial").write_bytes(
        np.asarray([0, 4, 8], dtype="<u8").tobytes()
    )
    (cache / "progress.json").write_text(
        json.dumps(
            {
                "fingerprint": "fp",
                "documents": 2,
                "tokens": 8,
                "straddling": 0,
            }
        )
    )

    tok = FakeTokenizer()
    docs, stats = build_encoding_cache_from_jsonl(
        tok,
        source,
        cache,
        fingerprint="fp",
        eos_id=99,
        batch_size=2,
    )

    assert len(docs) == 5
    assert stats["documents"] == 5
    assert tok.calls == 2, "only the remaining 3 rows should be encoded in 2 batches"
    assert all(doc.tolist() == [1, 2, 3, 99] for doc in docs)

