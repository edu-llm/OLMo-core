"""Fast, resumable token-shard construction.

These tests do not need Qwen or the real corpus. They pin the mechanics that make a
45-minute preprocessing job safe to restart: batched encoding, deterministic packing,
atomic shards, and refusing rather than deleting an unexpected partial artifact.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

import tokenize_corpus as tokenization  # noqa: E402
from provenance import tokenizer_behavior_sha256  # noqa: E402
from tokenize_corpus import (  # noqa: E402
    atomic_write_json,
    build_encoding_cache_from_jsonl,
    encode_rows_batched,
    file_sha256,
    fingerprint_dict,
    load_completed_group,
    load_encoding_cache,
    main,
    pack_indices_by_length,
    process_corpus,
    tokenizer_composite_seal,
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


class PackingTokenizer:
    """Small deterministic tokenizer that exercises the complete packing path."""

    is_fast = True
    eos_token_id = 250
    pad_token_id = 250
    all_special_tokens = ("<eos>",)
    all_special_ids = (250,)

    class Backend:
        @staticmethod
        def to_str() -> str:
            return '{"fixture":"character-tokenizer-v1"}'

        @staticmethod
        def get_vocab_size(*, with_added_tokens) -> int:
            assert with_added_tokens is True
            return 251

        @staticmethod
        def token_to_id(token: str) -> int | None:
            return 250 if token == "<|endoftext|>" else None

        @staticmethod
        def encode(text: str, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(
                ids=PackingTokenizer.encode_text(text),
                tokens=list(text),
                offsets=[(i, i + 1) for i in range(len(text))],
            )

        @staticmethod
        def decode(ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return "".join(chr(token - 1) for token in ids if token != 250)

    backend_tokenizer = Backend()

    @staticmethod
    def encode_text(text: str) -> list[int]:
        return [ord(char) + 1 for char in text]

    def __len__(self) -> int:
        return 251

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        if isinstance(texts, str):
            return {"input_ids": self.encode_text(texts)}
        ids = [self.encode_text(text) for text in texts]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [[(i, i + 1) for i in range(len(text))] for text in texts]
        return result


def _expected_packed_bytes(
    documents: list[np.ndarray], *, sequence_length: int, pad_id: int
) -> bytes:
    packed = np.full((len(documents), sequence_length), pad_id, dtype="<u4")
    for row, document in zip(packed, documents):
        row[: len(document)] = document
    return packed.tobytes()


def _install_packing_tokenizer(monkeypatch, tokenizer: PackingTokenizer) -> None:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(identifier, **_kwargs):
            assert identifier == "fixture-tokenizer"
            return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))


def _group_completion_sha256(payload: dict) -> str:
    sealed = {
        key: value
        for key, value in payload.items()
        if key not in {"completion_sha256", "resumed_group", "resumed_shards"}
    }
    return fingerprint_dict(sealed)


def _completed_group_fixture(tmp_path: Path) -> tuple[Path, Path, str, dict]:
    shard = tmp_path / "tokens" / "x" / "train-00000.u32le.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(np.arange(8, dtype="<u4").tobytes())
    build = {
        "schema_version": "p3-packed-group-v3",
        "code_version": "tokenize-corpus-v4",
        "fixture": "completed-group",
        "packing": {"sequence_length": 8},
    }
    fingerprint = fingerprint_dict(build)
    payload = {
        "schema_version": "p3-packed-group-v3",
        "code_version": "tokenize-corpus-v4",
        "fingerprint": fingerprint,
        "build": build,
        "documents": 1,
        "instances": 1,
        "shards": [
            {
                "path": "tokens/x/train-00000.u32le.bin",
                "instances": 1,
                "tokens": 8,
                "bytes": 32,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "tokens_dtype": "uint32",
                "byte_order": "little",
            }
        ],
    }
    payload["completion_sha256"] = _group_completion_sha256(payload)
    done = shard.parent / "train.done.json"
    atomic_write_json(done, payload)
    return shard, done, fingerprint, payload


def _test_corpus_binding() -> dict:
    return {
        "test_only": True,
        "corpus_generation": {
            "schema_version": "p3-tokenizer-corpus-binding-v1",
            "generation_id": "test-generation",
            "logical_root_sha256": "4" * 64,
            "manifest_root_sha256": "5" * 64,
            "manifest_file_sha256": "6" * 64,
            "current_sha256": "7" * 64,
        },
    }


FIXED_QWEN_TOKENIZER_SEAL = tokenization.FIXED_QWEN_TOKENIZER_SEAL
PINNED_EDULLM_DATA_COMMIT = "38bf831a6c3f445e394784018441fd59288b876c"
PINNED_PRETRAIN_POLICY_SHA256 = "2d507a1b8b9a5ce6c361b3e2731c12678cb9f3fc3e24c87aa6dc4b75100f0fd5"
DEPLOYED_EDULLM_DATA_COMMIT = "e0984c88b7c5d3d927bda227af4f47e2014dd257"
DEPLOYED_PRETRAIN_POLICY_SHA256 = "4128a90ba8ed8bb167180a2a19a4cbfc4788d5f14413dbff5e184745253bfbf3"


def _staged_payload_fixture(
    tmp_path: Path,
    *,
    families: tuple[str, ...] = tokenization.FAMILIES,
    splits: tuple[str, ...] = ("train", "val"),
) -> tuple[Path, Path]:
    staged = tmp_path / "staged"
    controls = tmp_path / "controls"
    controls.mkdir()
    generation = {
        "schema_version": "p3-tokenizer-corpus-binding-v1",
        "generation_id": "test-generation",
        "logical_root_sha256": "1" * 64,
        "manifest_root_sha256": "2" * 64,
        "manifest_file_sha256": "3" * 64,
        "current_sha256": "4" * 64,
        "semantic_contract_sha256": "5" * 64,
        "producer_source_sha256": "6" * 64,
    }
    for split_index, split in enumerate(splits):
        packing = {
            "algorithm": "largest-fit-decreasing-v1",
            "split": split,
            "packed": True,
            "sequence_length": 16_384,
            "shard_tokens": 16_384,
            "tokens_dtype": "uint32",
            "byte_order": "little",
            "eos_token_id": 151_643,
            "pad_token_id": 151_643,
            "separator": "\n---\nGOAL ",
            "separator_search": "---\nGOAL",
            "separator_ids": [10952, 15513, 969],
        }
        groups = {}
        source_hashes = {}
        source_inventory = {}
        for family_index, family in enumerate(families):
            family_dir = staged / "tokens" / family
            family_dir.mkdir(parents=True, exist_ok=True)
            shard_path = family_dir / f"{split}-00000.u32le.bin"
            values = np.arange(16_384, dtype="<u4") % 1_024 + 1 + family_index * 2_048 + split_index
            values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
            shard_path.write_bytes(values.tobytes())
            relative = shard_path.relative_to(staged).as_posix()
            source_sha256 = hashlib.sha256(f"{family}:{split}".encode()).hexdigest()
            source_hashes[family] = source_sha256
            source_inventory[family] = {
                "family": family,
                "schema": tokenization.P3_SOURCE_SCHEMAS[family],
            }
            build = {
                "schema_version": "p3-packed-group-v3",
                "code_version": "tokenize-corpus-v4",
                "source_jsonl": {
                    "family": family,
                    "name": f"{family}.jsonl",
                    "schema": tokenization.P3_SOURCE_SCHEMAS[family],
                    "sha256": source_sha256,
                },
                "tokenizer": FIXED_QWEN_TOKENIZER_SEAL,
                "encoding_cache_fingerprint": "7" * 64,
                "packing": packing,
                "corpus_generation": generation,
            }
            group = {
                "schema_version": "p3-packed-group-v3",
                "code_version": "tokenize-corpus-v4",
                "cross_split_binding_schema_version": "p3-token-cross-split-binding-v1",
                "fingerprint": fingerprint_dict(build),
                "build": build,
                "cache_fingerprint": "7" * 64,
                "cache_root_sha256": "8" * 64,
                "corpus_generation": generation,
                "name": family,
                "documents": 1,
                "source_documents": 1,
                "instances": 1,
                "real_tokens": 16_384,
                "padding_fraction": 0.0,
                "dropped_over_length": 0,
                "straddling": 0,
                "separator_ids": [10952, 15513, 969],
                "eos_token_id": 151_643,
                "pad_token_id": 151_643,
                "shards": [
                    {
                        "path": relative,
                        "instances": 1,
                        "tokens": 16_384,
                        "bytes": shard_path.stat().st_size,
                        "sha256": file_sha256(shard_path),
                        "tokens_dtype": "uint32",
                        "byte_order": "little",
                    }
                ],
                "resumed_shards": 0,
                "resumed_group": False,
            }
            group["completion_sha256"] = _group_completion_sha256(group)
            groups[family] = group
        manifest = {
            "schema_version": "p3-packed-corpus-v3",
            "code_version": "tokenize-corpus-v4",
            "cross_split_binding_schema_version": "p3-token-cross-split-binding-v1",
            "sequence_length": 16_384,
            "tokenizer": "/fixed/qwen25-vendored",
            "tokenizer_seal": FIXED_QWEN_TOKENIZER_SEAL,
            "tokenizer_composite_sha256": FIXED_QWEN_TOKENIZER_SEAL["tokenizer_composite_sha256"],
            "corpus_generation": generation,
            "source_jsonl_sha256": source_hashes,
            "source_family_inventory": source_inventory,
            "packing_config": packing,
            "tokens_dtype": "uint32",
            "byte_order": "little",
            "eos_token_id": 151_643,
            "pad_token_id": 151_643,
            "separator": "\n---\nGOAL ",
            "separator_search": "---\nGOAL",
            "separator_ids": [10952, 15513, 969],
            "split": split,
            "packed": True,
            "groups": groups,
            "instances": len(groups),
            "real_tokens": len(groups) * 16_384,
            "dropped_over_length": 0,
            "tokens_straddling_boundary": 0,
            "resumed_groups": 0,
            "resumed_shards": 0,
        }
        manifest["manifest_sha256"] = fingerprint_dict(manifest)
        atomic_write_json(controls / f"{split}_meta.json", manifest)
    return staged, controls


def _cache_build_identity(source: Path, *, eos_id: int = 99) -> dict:
    return {
        "schema_version": "p3-encoding-cache-v3",
        "code_version": "tokenize-corpus-v4",
        "source_jsonl": {
            "name": source.name,
            "sha256": file_sha256(source),
        },
        "tokenizer": {
            "tokenizer_file_sha256": {
                "tokenizer.json": "1" * 64,
                "tokenizer_config.json": "2" * 64,
            },
            "tokenizer_composite_sha256": "3" * 64,
            "tokenizers_version": "0.22.2",
            "tokenizer_eos_token_id": eos_id,
            "tokenizer_pad_token_id": eos_id,
            "separator": "\n---\nGOAL ",
            "separator_ids": [10, 11, 12],
        },
        "eos_token_id": eos_id,
    }


def _cache_source(tmp_path: Path, *, rows: int = 5) -> Path:
    source = tmp_path / "x.jsonl"
    source.write_text(
        "".join(
            json.dumps({"id": f"row-{i}", "text": "abc", "mask_end": 1}) + "\n" for i in range(rows)
        )
    )
    return source


def test_encoding_is_batched_and_appends_eos():
    rows = [{"text": "abcd", "mask_end": 2} for _ in range(10)]
    tok = FakeTokenizer()
    encoded, straddling = encode_rows_batched(tok, rows, eos_id=99, batch_size=4)

    assert tok.calls == 3, "10 documents at batch 4 should be 3 tokenizer calls, not 10"
    assert len(encoded) == 10
    assert all(ids.dtype == np.uint32 for ids in encoded)
    assert all(ids.tolist() == [1, 2, 3, 4, 99] for ids in encoded)
    assert straddling == 0


def test_tokenizer_composite_uses_training_behavior_seal_and_local_file_hashes(
    tmp_path,
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"fixture": 1}')
    config = tokenizer_dir / "tokenizer_config.json"
    config.write_text('{"eos_token": "<eos>"}')
    tokenizer = PackingTokenizer()

    first = tokenizer_composite_seal(
        tokenizer,
        tokenizer_dir,
        separator_ids=PackingTokenizer.encode_text("---\nGOAL"),
        test_only=True,
    )
    assert first["tokenizer_composite_sha256"] == tokenizer_behavior_sha256(
        tokenizer.backend_tokenizer
    )
    assert first["tokenizer_file_sha256"] == {
        "tokenizer.json": file_sha256(tokenizer_dir / "tokenizer.json"),
        "tokenizer_config.json": file_sha256(config),
    }
    assert first["tokenizers_version"] == "0.22.2"
    assert first["tokenizer_eos_token_id"] == 250
    assert first["tokenizer_pad_token_id"] == 250
    assert first["separator"] == "\n---\nGOAL "
    assert first["separator_ids"] == PackingTokenizer.encode_text("---\nGOAL")

    config.write_text('{"eos_token": "<changed>"}')
    second = tokenizer_composite_seal(
        tokenizer,
        tokenizer_dir,
        separator_ids=PackingTokenizer.encode_text("---\nGOAL"),
        test_only=True,
    )
    assert second["tokenizer_composite_sha256"] == first["tokenizer_composite_sha256"]
    assert fingerprint_dict(second) != fingerprint_dict(first)


def test_production_tokenizer_seal_refuses_mutable_hf_identifier():
    with pytest.raises(RuntimeError, match="local|tokenizer.json|mutable"):
        tokenizer_composite_seal(
            PackingTokenizer(),
            "Qwen/Qwen2.5-0.5B",
            separator_ids=PackingTokenizer.encode_text("---\nGOAL"),
        )


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
    expected = _expected_packed_bytes(docs, sequence_length=8, pad_id=9)

    first = write_shard_resumable(path, docs, sequence_length=8, pad_id=9)
    before = path.stat().st_mtime_ns
    second = write_shard_resumable(path, docs, sequence_length=8, pad_id=9)

    assert first["resumed"] is False
    assert second["resumed"] is True
    assert path.stat().st_mtime_ns == before, "resume must not rewrite completed bytes"
    for result in (first, second):
        assert result["sha256"] == hashlib.sha256(expected).hexdigest()
        assert result["tokens"] == 16
        assert result["bytes"] == len(expected)
        assert result["tokens_dtype"] == "uint32"
        assert result["byte_order"] == "little"
    raw = np.fromfile(path, dtype="<u4").reshape(2, 8)
    assert raw[0].tolist() == [1, 2, 9, 9, 9, 9, 9, 9]
    assert raw[1].tolist() == [3, 9, 9, 9, 9, 9, 9, 9]


def test_resumable_writer_refuses_equal_size_wrong_bytes_without_overwriting(tmp_path):
    path = tmp_path / "train-00000.u32le.bin"
    docs = [np.array([1, 2, 9], dtype=np.uint32), np.array([3, 9], dtype=np.uint32)]
    expected = _expected_packed_bytes(docs, sequence_length=8, pad_id=9)
    stale = bytes(byte ^ 0xFF for byte in expected)
    assert len(stale) == len(expected)
    path.write_bytes(stale)

    with pytest.raises(RuntimeError, match="SHA-256"):
        write_shard_resumable(path, docs, sequence_length=8, pad_id=9)

    assert path.read_bytes() == stale, "same-size stale bytes must be preserved, never relabeled"


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


def test_interrupted_shard_temp_never_counts_as_complete(tmp_path, monkeypatch):
    path = tmp_path / "train-00000.u32le.bin"
    documents = [np.array([1, 2, 9], dtype=np.uint32)]
    original_replace = tokenization.os.replace

    def interrupt_replace(source, destination):
        assert Path(destination) == path
        raise OSError("simulated crash before atomic replace")

    monkeypatch.setattr(tokenization.os, "replace", interrupt_replace)
    with pytest.raises(OSError, match="simulated crash"):
        write_shard_resumable(path, documents, sequence_length=8, pad_id=9)

    partials = list(tmp_path.glob(f"{path.name}.partial-*"))
    assert len(partials) == 1
    assert not path.exists()

    monkeypatch.setattr(tokenization.os, "replace", original_replace)
    result = write_shard_resumable(path, documents, sequence_length=8, pad_id=9)
    assert result["resumed"] is False
    assert path.exists()
    assert partials[0].exists(), "diagnostic temp bytes are preserved but never counted"


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


def test_completed_encoding_cache_seals_payloads_ranges_source_and_build(tmp_path):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    cache = tmp_path / "cache"

    docs, stats = build_encoding_cache_from_jsonl(
        FakeTokenizer(),
        source,
        cache,
        fingerprint=fingerprint,
        build=build,
        eos_id=99,
        batch_size=2,
    )

    assert len(docs) == 5
    assert stats["schema_version"] == "p3-encoding-cache-v3"
    assert stats["status"] == "complete"
    assert stats["build"] == build
    assert stats["build_fingerprint"] == fingerprint
    assert stats["source_jsonl"] == build["source_jsonl"]
    assert [chunk["documents"] for chunk in stats["chunks"]] == [
        {"start": 0, "end": 2},
        {"start": 2, "end": 4},
        {"start": 4, "end": 5},
    ]
    assert [chunk["tokens"] for chunk in stats["chunks"]] == [
        {"start": 0, "end": 8},
        {"start": 8, "end": 16},
        {"start": 16, "end": 20},
    ]
    for name, dtype, byte_order in (
        ("tokens", "uint32", "little"),
        ("offsets", "uint64", "little"),
    ):
        payload = stats["payloads"][name]
        path = cache / payload["path"]
        assert payload["sha256"] == file_sha256(path)
        assert payload["bytes"] == path.stat().st_size
        assert payload["dtype"] == dtype
        assert payload["byte_order"] == byte_order
    assert stats["cache_root_sha256"] == tokenization.cache_root_sha256(stats)


@pytest.mark.parametrize("payload_name", ("tokens.u32le.bin", "offsets.u64le.bin"))
def test_completed_encoding_cache_rehashes_same_size_payload_mutation(tmp_path, payload_name):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    cache = tmp_path / "cache"
    build_encoding_cache_from_jsonl(
        FakeTokenizer(),
        source,
        cache,
        fingerprint=fingerprint,
        build=build,
        eos_id=99,
        batch_size=2,
    )
    payload = cache / payload_name
    original = bytearray(payload.read_bytes())
    original[len(original) // 2] ^= 0x80
    payload.write_bytes(original)

    with pytest.raises(RuntimeError, match="SHA-256|digest"):
        load_encoding_cache(
            cache,
            fingerprint=fingerprint,
            build=build,
            source=source,
        )


def test_partial_encoding_cache_rehashes_every_committed_chunk_before_resume(tmp_path, monkeypatch):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    cache = tmp_path / "cache"
    original_encode = tokenization.encode_rows_batched
    calls = 0

    def interrupt_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated encoder crash")
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(tokenization, "encode_rows_batched", interrupt_second_batch)
    with pytest.raises(OSError, match="simulated encoder crash"):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint,
            build=build,
            eos_id=99,
            batch_size=2,
        )

    progress = json.loads((cache / "progress.json").read_text())
    assert progress["schema_version"] == "p3-encoding-cache-progress-v3"
    assert progress["build"] == build
    assert progress["build_fingerprint"] == fingerprint
    assert progress["chunks"][0]["documents"] == {"start": 0, "end": 2}
    partial = cache / "tokens.u32le.bin.partial"
    stale = bytearray(partial.read_bytes())
    stale[len(stale) // 2] ^= 0x40
    partial.write_bytes(stale)

    monkeypatch.setattr(tokenization, "encode_rows_batched", original_encode)
    with pytest.raises(RuntimeError, match="chunk|SHA-256|digest"):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint,
            build=build,
            eos_id=99,
            batch_size=2,
        )


def test_partial_encoding_cache_binds_exact_source_sequence(tmp_path, monkeypatch):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    cache = tmp_path / "cache"
    original_encode = tokenization.encode_rows_batched
    calls = 0

    def interrupt_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated encoder crash")
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(tokenization, "encode_rows_batched", interrupt_second_batch)
    with pytest.raises(OSError):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint,
            build=build,
            eos_id=99,
            batch_size=2,
        )
    lines = source.read_text().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    source.write_text("".join(lines))

    monkeypatch.setattr(tokenization, "encode_rows_batched", original_encode)
    with pytest.raises(RuntimeError, match="source|sequence|SHA-256"):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint,
            build=build,
            eos_id=99,
            batch_size=2,
        )


def test_cache_fingerprint_mismatch_never_relabels_existing_payloads(tmp_path):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    cache = tmp_path / "cache"
    build_encoding_cache_from_jsonl(
        FakeTokenizer(),
        source,
        cache,
        fingerprint=fingerprint,
        build=build,
        eos_id=99,
        batch_size=2,
    )
    original = {path.name: path.read_bytes() for path in cache.iterdir() if path.is_file()}
    other_build = {**build, "eos_token_id": 100}

    with pytest.raises(RuntimeError, match="fingerprint|preserved"):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint_dict(other_build),
            build=other_build,
            eos_id=100,
            batch_size=2,
        )

    assert {path.name: path.read_bytes() for path in cache.iterdir() if path.is_file()} == original


def test_encoding_cache_two_clean_builds_are_byte_identical(tmp_path):
    source = _cache_source(tmp_path)
    build = _cache_build_identity(source)
    fingerprint = fingerprint_dict(build)
    trees = []
    for name in ("a", "b"):
        cache = tmp_path / name
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint,
            build=build,
            eos_id=99,
            batch_size=2,
        )
        trees.append(
            {
                path.relative_to(cache).as_posix(): path.read_bytes()
                for path in sorted(cache.rglob("*"))
                if path.is_file()
            }
        )
    assert trees[0] == trees[1]


def test_completed_group_is_resumed_only_when_fingerprint_and_shards_match(tmp_path):
    shard, done, fingerprint, payload = _completed_group_fixture(tmp_path)

    assert load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path) == payload
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_completed_group(done, fingerprint="wrong", output_root=tmp_path)

    shard.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="preserved"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)


def test_completed_group_rehashes_shards_and_refuses_same_size_mutation(tmp_path):
    shard, done, fingerprint, _ = _completed_group_fixture(tmp_path)
    marker_before = done.read_bytes()
    mutated = bytearray(shard.read_bytes())
    mutated[len(mutated) // 2] ^= 0x80
    shard.write_bytes(mutated)
    assert shard.stat().st_size == 32

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)

    assert done.read_bytes() == marker_before, "a corrupt group must never be relabeled current"


def test_completed_group_seal_prevents_relabeling_mutated_bytes_in_marker(tmp_path):
    shard, done, fingerprint, _ = _completed_group_fixture(tmp_path)
    mutated = bytearray(shard.read_bytes())
    mutated[0] ^= 0x01
    shard.write_bytes(mutated)
    payload = json.loads(done.read_text())
    payload["shards"][0]["sha256"] = file_sha256(shard)
    atomic_write_json(done, payload)

    with pytest.raises(RuntimeError, match="completion seal"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)


def test_completed_group_refuses_extra_final_shards(tmp_path):
    shard, done, fingerprint, _ = _completed_group_fixture(tmp_path)
    extra = shard.with_name("train-00001.u32le.bin")
    extra.write_bytes(shard.read_bytes())

    with pytest.raises(RuntimeError, match="unexpected shards"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)


def test_completed_group_refuses_missing_shard_even_when_temp_bytes_remain(tmp_path):
    shard, done, fingerprint, _ = _completed_group_fixture(tmp_path)
    partial = shard.with_name(f"{shard.name}.partial-crash")
    shard.replace(partial)
    assert partial.stat().st_size == 32

    with pytest.raises(RuntimeError, match="missing shards"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)


def test_completed_group_refuses_partial_temp_and_unknown_files(tmp_path):
    shard, done, fingerprint, _ = _completed_group_fixture(tmp_path)
    (shard.parent / f"{shard.name}.partial-crash").write_bytes(b"diagnostic")

    with pytest.raises(RuntimeError, match="inventory|partial|unexpected"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)

    (shard.parent / f"{shard.name}.partial-crash").unlink()
    (shard.parent / "unknown.control").write_text("stale")
    with pytest.raises(RuntimeError, match="inventory|unknown|unexpected"):
        load_completed_group(done, fingerprint=fingerprint, output_root=tmp_path)


def test_legacy_size_only_done_marker_forces_rebuild(tmp_path):
    shard = tmp_path / "tokens" / "x" / "train-00000.u32le.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"\0" * 32)
    done = shard.parent / "train.done.json"
    done.write_text(
        json.dumps(
            {
                "fingerprint": "legacy",
                "shards": [
                    {
                        "path": "tokens/x/train-00000.u32le.bin",
                        "tokens": 8,
                        "bytes": 32,
                    }
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="legacy.*rebuild"):
        load_completed_group(done, fingerprint="legacy", output_root=tmp_path)


def test_partial_done_marker_never_counts_as_completion(tmp_path):
    done = tmp_path / "tokens" / "x" / "train.done.json"
    done.parent.mkdir(parents=True)
    (done.parent / "train.done.json.partial-crash").write_text(
        json.dumps({"fingerprint": "not-complete"})
    )

    assert load_completed_group(done, fingerprint="not-complete", output_root=tmp_path) is None


def test_legacy_size_only_batch_progress_is_refused_and_preserved(tmp_path):
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
    (cache / "offsets.u64le.bin.partial").write_bytes(np.asarray([0, 4, 8], dtype="<u8").tobytes())
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

    before = {path.name: path.read_bytes() for path in cache.iterdir()}
    build = _cache_build_identity(source)
    with pytest.raises(RuntimeError, match="legacy|payload digests"):
        build_encoding_cache_from_jsonl(
            FakeTokenizer(),
            source,
            cache,
            fingerprint=fingerprint_dict(build),
            build=build,
            eos_id=99,
            batch_size=2,
        )
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == before


def test_small_packed_group_reconstructs_exact_documents_and_records_v3_seal(tmp_path, monkeypatch):
    rows = [
        {"text": "A---\nGOALx", "mask_end": 9},
        {"text": "BB---\nGOALyy", "mask_end": 10},
        {"text": "CCC---\nGOALz", "mask_end": 11},
        {"text": "DDDD---\nGOALqq", "mask_end": 12},
    ]
    source = tmp_path / "source" / "tiny.jsonl"
    source.parent.mkdir()
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "out"
    tokenizer = PackingTokenizer()

    _install_packing_tokenizer(monkeypatch, tokenizer)
    task = {
        "source": str(source),
        "output_root": str(output),
        "cache_root": str(tmp_path / "cache"),
        "split": "train",
        "tokenizer": "fixture-tokenizer",
        "sequence_length": 32,
        "shard_tokens": 64,
        "pack": True,
        "batch_size": 2,
        "suggest": False,
        **_test_corpus_binding(),
    }

    result = process_corpus(task)
    done_path = output / "tokens" / "tiny" / "train.done.json"
    done = json.loads(done_path.read_text())
    assert done == result
    assert done["schema_version"] == "p3-packed-group-v3"
    assert done["code_version"] == "tokenize-corpus-v4"
    assert done["fingerprint"] == fingerprint_dict(done["build"])
    assert done["build"]["source_jsonl"] == {
        "family": "tiny",
        "name": "tiny.jsonl",
        "schema": "test-only-unsealed",
        "sha256": file_sha256(source),
    }
    assert len(done["build"]["tokenizer"]["tokenizer_composite_sha256"]) == 64
    assert done["build"]["corpus_generation"]["generation_id"] == "test-generation"
    assert "evaluator_dependency" not in done
    assert "evaluator_dependency" not in done["build"]
    assert done["build"]["packing"] == {
        "algorithm": "largest-fit-decreasing-v1",
        "split": "train",
        "packed": True,
        "sequence_length": 32,
        "shard_tokens": 64,
        "tokens_dtype": "uint32",
        "byte_order": "little",
        "eos_token_id": 250,
        "pad_token_id": 250,
        "separator": "\n---\nGOAL ",
        "separator_search": "---\nGOAL",
        "separator_ids": PackingTokenizer.encode_text("---\nGOAL"),
    }

    recovered = []
    for shard_record in done["shards"]:
        assert shard_record["tokens_dtype"] == "uint32"
        assert shard_record["byte_order"] == "little"
        shard_path = output / shard_record["path"]
        assert shard_record["sha256"] == file_sha256(shard_path)
        assert shard_record["bytes"] == shard_path.stat().st_size
        assert shard_record["tokens"] * 4 == shard_record["bytes"]
        packed = np.fromfile(shard_path, dtype="<u4").reshape(-1, 32)
        for packed_row in packed:
            document = []
            for token in packed_row:
                if int(token) == tokenizer.eos_token_id:
                    if document:
                        recovered.append((*document, tokenizer.eos_token_id))
                        document.clear()
                else:
                    document.append(int(token))
            assert not document

    expected = Counter(
        (*tokenizer.encode_text(row["text"]), tokenizer.eos_token_id) for row in rows
    )
    assert Counter(recovered) == expected

    resumed = process_corpus(task)
    assert resumed["resumed_group"] is True
    assert json.loads(done_path.read_text())["resumed_group"] is False


def _cross_split_manifest(result: dict, *, split: str) -> dict:
    manifest = {
        "schema_version": "p3-packed-corpus-v3",
        "code_version": "tokenize-corpus-v4",
        "cross_split_binding_schema_version": "p3-token-cross-split-binding-v1",
        "corpus_generation": result["corpus_generation"],
        "tokenizer_seal": result["build"]["tokenizer"],
        "packing_config": result["build"]["packing"],
        "source_family_inventory": {
            result["name"]: {
                "family": result["build"]["source_jsonl"]["family"],
                "schema": result["build"]["source_jsonl"]["schema"],
            }
        },
        "split": split,
        "groups": {result["name"]: result},
    }
    manifest["manifest_sha256"] = fingerprint_dict(manifest)
    return manifest


@pytest.mark.parametrize(
    "drift",
    (
        "corpus_generation",
        "tokenizer_seal",
        "packing_contract",
        "source_family_inventory",
    ),
)
def test_cross_split_binding_rejects_one_field_drift_before_commit(tmp_path, monkeypatch, drift):
    output = tmp_path / "out"
    _install_packing_tokenizer(monkeypatch, PackingTokenizer())
    binding = _test_corpus_binding()
    sources = {}
    for split in ("train", "val"):
        source = tmp_path / split / "tiny.jsonl"
        source.parent.mkdir()
        source.write_text(json.dumps({"text": "A---\nGOALx", "mask_end": 9}) + "\n")
        sources[split] = source

    def task(split: str, *, corpus_generation: dict) -> dict:
        source = sources[split]
        return {
            "name": "tiny",
            "source": str(source),
            "source_record": {
                "sha256": file_sha256(source),
                "bytes": source.stat().st_size,
                "schema": "atp-v2",
            },
            "output_root": str(output),
            "cache_root": str(tmp_path / "cache"),
            "split": split,
            "tokenizer": "fixture-tokenizer",
            "sequence_length": 32,
            "shard_tokens": 64,
            "pack": True,
            "batch_size": 2,
            "suggest": False,
            "test_only": True,
            "corpus_generation": corpus_generation,
        }

    train = process_corpus(
        task(
            "train",
            corpus_generation=binding["corpus_generation"],
        )
    )
    train_meta = _cross_split_manifest(train, split="train")
    atomic_write_json(output / "train_meta.json", train_meta)

    generation_b = {
        **binding["corpus_generation"],
        "generation_id": "generation-B",
        "logical_root_sha256": "b" * 64,
    }
    val_generation = generation_b if drift == "corpus_generation" else binding["corpus_generation"]
    val = process_corpus(
        {
            **task(
                "val",
                corpus_generation=val_generation,
            ),
            "defer_done_commit": True,
        }
    )
    if drift == "tokenizer_seal":
        val["build"]["tokenizer"] = {
            **val["build"]["tokenizer"],
            "tokenizer_composite_sha256": "c" * 64,
        }
    elif drift == "packing_contract":
        val["build"]["packing"] = {
            **val["build"]["packing"],
            "shard_tokens": 65,
        }
    elif drift == "source_family_inventory":
        val["build"]["source_jsonl"] = {
            **val["build"]["source_jsonl"],
            "schema": "atp-v3",
        }
    val["fingerprint"] = fingerprint_dict(val["build"])
    val["completion_sha256"] = _group_completion_sha256(val)
    val_meta = _cross_split_manifest(val, split="val")
    label = drift.replace("_", " ")

    with pytest.raises(RuntimeError, match=rf"cross-split.*{label}"):
        tokenization.require_cross_split_finalization(
            output,
            split="val",
            groups=[val],
            manifest=val_meta,
        )

    assert not (output / "val_meta.json").exists()
    assert not (output / "tokens" / "tiny" / "val.done.json").exists()

    atomic_write_json(output / "tokens" / "tiny" / "val.done.json", val)
    with pytest.raises(RuntimeError, match="cross-split"):
        tokenization.require_exact_group_inventory(output / "tokens" / "tiny")


def test_staged_publication_gate_accepts_exact_six_family_fixture(tmp_path):
    staged, controls = _staged_payload_fixture(tmp_path)

    result = tokenization.validate_staged_token_payload(
        staged,
        train_manifest_path=controls / "train_meta.json",
        val_manifest_path=controls / "val_meta.json",
    )

    assert result["families"] == list(tokenization.FAMILIES)
    assert result["tokenizer"] == {
        "dataset_id": "tokenizer/qwen25-vendored",
        "version": "v1",
    }
    assert {entry["path"] for entry in result["entries"]} == {
        f"tokens/{family}/{split}-00000.u32le.bin"
        for family in tokenization.FAMILIES
        for split in ("train", "val")
    }
    assert result["partitions"] == {
        "train": {"files": 6, "tokens": 6 * 16_384},
        "val": {"files": 6, "tokens": 6 * 16_384},
    }


def test_documented_staging_copies_only_manifest_shards_from_internal_output(tmp_path):
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    output, controls = _staged_payload_fixture(fixture_root)
    for split in ("train", "val"):
        (output / f"{split}_meta.json").write_bytes((controls / f"{split}_meta.json").read_bytes())
    for family in tokenization.FAMILIES:
        for split in ("train", "val"):
            (output / "tokens" / family / f"{split}.done.json").write_text("{}\n")
    assert len(list(output.rglob("*.done.json"))) == 12

    readme = (SCRIPTS / "README.md").read_text()
    marker = "export PUBLISH_ROOT=/absolute/path/to/new/p3-pretrain-publish\n"
    assert marker in readme
    exact_real_gate = (
        'P3_EDULLM_DATA_SOURCE="$P3_EDULLM_DATA_SOURCE" '
        'P3_REAL_STAGED_PAYLOAD="$PUBLISH_ROOT" '
        'P3_REAL_TOKENIZED_DIR="$OUT" '
        'P3_FIXED_TOKENIZER_DIR="$TOKENIZER" '
        ".venv/bin/python -m pytest -q -s "
        "src/test/scripts/p3_math_split/tokenize_corpus_test.py::"
        "test_real_staged_payload_gate_uses_explicit_paths_and_reports_policy_delta"
    )
    assert exact_real_gate in readme
    assert "`PASS`:" in readme
    assert "`REPORT`:" in readme
    assert "`SKIP`:" in readme
    staging_script = readme.split(marker, 1)[1].split("\nP3_EDULLM_DATA_SOURCE=", 1)[0]
    publish_root = tmp_path / "publish"
    environment = {
        **os.environ,
        "OUT": str(output),
        "PUBLISH_ROOT": str(publish_root),
    }
    subprocess.run(
        ["bash", "-eu", "-c", staging_script],
        cwd=SCRIPTS.parents[3],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    result = tokenization.validate_staged_token_payload(
        publish_root,
        train_manifest_path=output / "train_meta.json",
        val_manifest_path=output / "val_meta.json",
    )
    expected_paths = {
        shard["path"]
        for split in ("train", "val")
        for group in json.loads((output / f"{split}_meta.json").read_text())["groups"].values()
        for shard in group["shards"]
    }
    assert {entry["path"] for entry in result["entries"]} == expected_paths
    assert {
        path.relative_to(publish_root).as_posix()
        for path in publish_root.rglob("*")
        if path.is_file()
    } == expected_paths
    assert not list(publish_root.rglob("*.done.json"))

    retry = subprocess.run(
        ["bash", "-eu", "-c", staging_script],
        cwd=SCRIPTS.parents[3],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert retry.returncode != 0, "documented staging must refuse an existing destination"


def test_staged_publication_gate_rejects_internal_done_controls(tmp_path):
    staged, controls = _staged_payload_fixture(tmp_path)
    for family in tokenization.FAMILIES:
        for split in ("train", "val"):
            (staged / "tokens" / family / f"{split}.done.json").write_text("{}\n")

    with pytest.raises(RuntimeError, match="extra|inventory|unexpected"):
        tokenization.run_staged_pretrain_profile_gate(
            staged,
            train_manifest_path=controls / "train_meta.json",
            val_manifest_path=controls / "val_meta.json",
            tokenizer_source=staged,
            scratch_root=tmp_path / "unused-profile-gate",
            edullm_modules=None,
            validator_policies=(),
            payload_kind="synthetic",
        )


@pytest.mark.parametrize(
    ("families", "splits", "message"),
    (
        (tokenization.FAMILIES[:-1], ("train", "val"), "six|family"),
        (tokenization.FAMILIES, ("train",), "train.*val|partition|manifest"),
    ),
)
def test_staged_publication_gate_rejects_omitted_family_or_partition(
    tmp_path,
    families,
    splits,
    message,
):
    staged, controls = _staged_payload_fixture(
        tmp_path,
        families=families,
        splits=splits,
    )

    with pytest.raises(RuntimeError, match=message):
        tokenization.run_staged_pretrain_profile_gate(
            staged,
            train_manifest_path=controls / "train_meta.json",
            val_manifest_path=controls / "val_meta.json",
            tokenizer_source=staged,
            scratch_root=tmp_path / "unused-profile-gate",
            edullm_modules=None,
            validator_policies=(),
            payload_kind="synthetic",
        )


def test_staged_publication_gate_rejects_fixed_tokenizer_drift(tmp_path):
    staged, controls = _staged_payload_fixture(tmp_path)
    manifest_path = controls / "train_meta.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tokenizer_seal"]["tokenizer_eos_token_id"] = 0
    manifest["manifest_sha256"] = fingerprint_dict(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="tokenizer"):
        tokenization.validate_staged_token_payload(
            staged,
            train_manifest_path=manifest_path,
            val_manifest_path=controls / "val_meta.json",
        )


def test_staged_publication_gate_rejects_legacy_evaluator_state(tmp_path):
    staged, controls = _staged_payload_fixture(tmp_path)
    manifest_path = controls / "train_meta.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluator_dependency"] = {"role": "evaluator"}
    manifest["manifest_sha256"] = fingerprint_dict(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="fields|exact|unexpected"):
        tokenization.validate_staged_token_payload(
            staged,
            train_manifest_path=manifest_path,
            val_manifest_path=controls / "val_meta.json",
        )


def _load_exact_edullm_package(monkeypatch):
    source_value = os.environ.get("P3_EDULLM_DATA_SOURCE")
    if source_value is None:
        pytest.skip(
            "exact profile gate requires P3_EDULLM_DATA_SOURCE; "
            "a skipped structural test cannot authorize publication"
        )
    source = Path(source_value).expanduser().resolve(strict=True)
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == PINNED_EDULLM_DATA_COMMIT

    for name in tuple(sys.modules):
        if name == "edullm_data" or name.startswith("edullm_data."):
            del sys.modules[name]
    monkeypatch.syspath_prepend(str(source / "src"))
    importlib.invalidate_caches()

    edullm_data = importlib.import_module("edullm_data")
    publish = importlib.import_module("edullm_data.publish")
    validate = importlib.import_module("edullm_data.validate")
    manifest_contract = importlib.import_module("edullm_data.manifest")
    registry = importlib.import_module("edullm_data.profiles.registry")
    s3_contract = importlib.import_module("edullm_data.s3")

    assert edullm_data.__version__ == "0.5.0"
    assert Path(edullm_data.__file__).resolve().is_relative_to(source / "src")
    assert registry.available() == [
        "eval-results/v1",
        "pretrain-tokens/v1",
        "sft-conversations/v1",
        "token-order/v1",
        "tokenizer/v1",
    ]
    pretrain_policy = source / "families" / "pretrain.json"
    assert file_sha256(pretrain_policy) == PINNED_PRETRAIN_POLICY_SHA256
    family = json.loads(pretrain_policy.read_text())
    assert family["defaults"]["decode_smoke_test"]["distinct_ids_min"] == 256
    return SimpleNamespace(
        source=source,
        version=edullm_data.__version__,
        publish=publish,
        validate=validate,
        manifest=manifest_contract,
        registry=registry,
        s3=s3_contract,
    )


def _fixed_qwen_tokenizer_source() -> Path:
    return (
        Path(__file__).resolve().parents[6]
        / "memorysplit-requery-exact"
        / "tokenizers"
        / "qwen25-vendored"
    )


def _validator_policy_fixtures(modules, tmp_path: Path) -> tuple[dict, dict]:
    deployed_commit = subprocess.run(
        [
            "git",
            "-C",
            str(modules.source),
            "rev-parse",
            f"{DEPLOYED_EDULLM_DATA_COMMIT}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert deployed_commit == DEPLOYED_EDULLM_DATA_COMMIT
    deployed_policy_bytes = subprocess.run(
        [
            "git",
            "-C",
            str(modules.source),
            "show",
            f"{DEPLOYED_EDULLM_DATA_COMMIT}:families/pretrain.json",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(deployed_policy_bytes).hexdigest() == DEPLOYED_PRETRAIN_POLICY_SHA256
    deployed_families = tmp_path / "deployed-policy-128"
    deployed_families.mkdir()
    (deployed_families / "pretrain.json").write_bytes(deployed_policy_bytes)

    return (
        {
            "label": "image-pinned-local-256",
            "families_dir": modules.source / "families",
            "source_commit": PINNED_EDULLM_DATA_COMMIT,
            "pretrain_policy_sha256": PINNED_PRETRAIN_POLICY_SHA256,
            "distinct_ids_min": 256,
        },
        {
            "label": "deployed-policy-128",
            "families_dir": deployed_families,
            "source_commit": DEPLOYED_EDULLM_DATA_COMMIT,
            "pretrain_policy_sha256": DEPLOYED_PRETRAIN_POLICY_SHA256,
            "job_definition_revision": 12,
            "distinct_ids_min": 128,
        },
    )


def _run_synthetic_profile_gate(tmp_path: Path, monkeypatch) -> tuple[dict, Path, Path]:
    modules = _load_exact_edullm_package(monkeypatch)
    staged, controls = _staged_payload_fixture(tmp_path)
    report = tokenization.run_staged_pretrain_profile_gate(
        staged,
        train_manifest_path=controls / "train_meta.json",
        val_manifest_path=controls / "val_meta.json",
        tokenizer_source=_fixed_qwen_tokenizer_source(),
        scratch_root=tmp_path / "profile-gate",
        edullm_modules=modules,
        validator_policies=_validator_policy_fixtures(modules, tmp_path),
        payload_kind="synthetic",
    )
    return report, staged, controls


def _replace_staged_shard_and_reseal(
    staged: Path,
    controls: Path,
    payload: bytes,
    *,
    family: str = "metamath",
    split: str = "train",
) -> None:
    shard_path = staged / "tokens" / family / f"{split}-00000.u32le.bin"
    assert len(payload) == shard_path.stat().st_size
    shard_path.write_bytes(payload)
    manifest_path = controls / f"{split}_meta.json"
    manifest = json.loads(manifest_path.read_text())
    group = manifest["groups"][family]
    shard = group["shards"][0]
    shard["sha256"] = file_sha256(shard_path)
    group["completion_sha256"] = _group_completion_sha256(group)
    manifest["manifest_sha256"] = fingerprint_dict(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    atomic_write_json(manifest_path, manifest)


def _publish_fixed_qwen_to_fake_s3(modules, tmp_path: Path):
    tokenizer_source = _fixed_qwen_tokenizer_source()
    tokenizer_payload = tmp_path / "tokenizer-payload" / "files"
    tokenizer_payload.mkdir(parents=True)
    for filename, expected_sha256 in FIXED_QWEN_TOKENIZER_SEAL["tokenizer_file_sha256"].items():
        tokenizer_bytes = (tokenizer_source / filename).read_bytes()
        assert hashlib.sha256(tokenizer_bytes).hexdigest() == expected_sha256
        (tokenizer_payload / filename).write_bytes(tokenizer_bytes)

    s3 = modules.s3.FakeS3()
    tokenizer_plan = modules.publish.publish(
        tokenizer_payload.parent,
        dataset_id="tokenizer/qwen25-vendored",
        purpose="Fixed published Qwen2.5 tokenizer used to encode the P3 pretraining corpus",
        profile="tokenizer/v1",
        s3=s3,
        created_at="2026-08-03T23:00:00Z",
        env={
            "EDULLM_CODE_SHA256": "a" * 64,
            "EDULLM_PACKAGES_LOCK_SHA256": "b" * 64,
        },
    )
    tokenizer_prefix = f"{tokenizer_plan.dataset_id}/{tokenizer_plan.version}"
    tokenizer_result = modules.validate.validate_dataset(
        "edullm-landing",
        tokenizer_prefix,
        s3,
        data_bucket="edullm-data",
    )
    assert tokenizer_result.ok, [str(violation) for violation in tokenizer_result.violations]
    modules.validate.promote(
        tokenizer_result,
        s3,
        data_bucket="edullm-data",
        landing_bucket="edullm-landing",
    )
    tokenizer_dataset = json.loads(s3.get("edullm-data", f"{tokenizer_prefix}/dataset.json"))
    tokenizer_manifest_sha256 = tokenizer_dataset["groups"][0]["manifest_sha256"]
    return s3, tokenizer_manifest_sha256


def test_synthetic_six_family_fake_s3_structure_matches_pinned_profile(
    tmp_path,
    monkeypatch,
):
    """Structural package test only; synthetic diversity cannot authorize real publication."""
    modules = _load_exact_edullm_package(monkeypatch)
    staged, controls = _staged_payload_fixture(tmp_path)
    expected = tokenization.validate_staged_token_payload(
        staged,
        train_manifest_path=controls / "train_meta.json",
        val_manifest_path=controls / "val_meta.json",
    )
    expected_entries = {entry["path"]: entry for entry in expected["entries"]}
    s3, tokenizer_manifest_sha256 = _publish_fixed_qwen_to_fake_s3(modules, tmp_path)

    plan = modules.publish.publish(
        staged,
        dataset_id="pretrain/formal-proof-premises-500m",
        purpose=(
            "Packed formal-proof premise tokens for OLMo pretraining to compare dense "
            "and memory-split learning"
        ),
        profile="pretrain-tokens/v1",
        s3=s3,
        created_at="2026-08-03T23:00:00Z",
        tokenizer="tokenizer/qwen25-vendored/v1",
        group_meta={"tokens": {"seq_len": 16_384}},
        env={
            "EDULLM_CODE_SHA256": "a" * 64,
            "EDULLM_PACKAGES_LOCK_SHA256": "b" * 64,
        },
    )
    prefix = f"{plan.dataset_id}/{plan.version}"
    result = modules.validate.validate_dataset(
        "edullm-landing",
        prefix,
        s3,
        data_bucket="edullm-data",
    )
    assert result.ok, [str(violation) for violation in result.violations]

    dataset = json.loads(s3.get("edullm-landing", f"{prefix}/dataset.json"))
    group = dataset["groups"][0]
    assert group["name"] == "tokens"
    assert group["seq_len"] == 16_384
    assert group["depends_on"] == [
        {
            "role": "tokenizer",
            "dataset_id": "tokenizer/qwen25-vendored",
            "version": "v1",
            "manifest_sha256": tokenizer_manifest_sha256,
        }
    ]
    assert "evaluator" not in json.dumps(dataset).lower()
    assert {partition["name"]: partition["rows"] for partition in group["partitions"]} == {
        split: record["tokens"] for split, record in expected["partitions"].items()
    }

    manifest = json.loads(s3.get("edullm-landing", f"{prefix}/tokens/manifest.json"))
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    assert set(entries) == set(expected_entries)
    assert {entry["labels"]["source"] for entry in entries.values()} == set(tokenization.FAMILIES)
    for path, entry_payload in entries.items():
        expected_entry = expected_entries[path]
        assert entry_payload["split"] == expected_entry["split"]
        assert entry_payload["count"] == {
            "unit": "tokens",
            "value": expected_entry["tokens"],
        }
        assert entry_payload["format"] == {
            "container": "raw",
            "dtype": "uint32",
            "byte_order": "little",
            "header_bytes": 0,
            "codec": "none",
        }
        assert entry_payload["bytes"] == expected_entry["bytes"]
        assert entry_payload["sha256"] == expected_entry["sha256"]
        entry = modules.manifest.ManifestEntry.from_dict(entry_payload)
        assert modules.manifest.parse_shard_name(path) == (expected_entry["split"], 0)
        assert modules.manifest.check_shard_naming(path) == []
        assert modules.manifest.verify_arithmetic(entry) == []


def test_synthetic_profile_gate_runs_pinned_publish_and_sampled_byte_validation(
    tmp_path,
    monkeypatch,
):
    """Synthetic coverage proves mechanics only; it never authorizes real publication."""
    report, _, _ = _run_synthetic_profile_gate(tmp_path, monkeypatch)

    assert report["status"] == "PASS"
    assert report["payload_kind"] == "synthetic"
    assert report["authorizes_publication"] is False
    assert report["profile"] == "pretrain-tokens/v1"
    assert report["edullm_data"]["version"] == "0.5.0"
    assert report["profile_checks"] == [
        "check_entries_declare_token_counts",
        "check_decode_smoke",
        "check_first_bytes_not_npy",
        "check_seq_len_alignment",
    ]
    assert report["families"] == list(tokenization.FAMILIES)
    assert report["partitions"] == {
        "train": {"files": 6, "tokens": 6 * 16_384},
        "val": {"files": 6, "tokens": 6 * 16_384},
    }
    assert report["tokenizer"]["file_sha256"] == FIXED_QWEN_TOKENIZER_SEAL["tokenizer_file_sha256"]
    assert report["tokenizer"]["derived"] == {
        "vocab_size": 151_665,
        "eos_token_id": 151_643,
    }
    assert report["fake_published_inventory"] == {
        "objects": 12,
        "bytes": 12 * 16_384 * 4,
    }
    assert report["fake_published_token_manifest"] == {
        "objects": 12,
        "bytes": 12 * 16_384 * 4,
    }
    assert set(report["policies"]) == {
        "image-pinned-local-256",
        "deployed-policy-128",
    }
    assert report["policies"]["image-pinned-local-256"]["source_commit"] == (
        PINNED_EDULLM_DATA_COMMIT
    )
    assert report["policies"]["image-pinned-local-256"]["pretrain_policy_sha256"] == (
        PINNED_PRETRAIN_POLICY_SHA256
    )
    assert report["policies"]["deployed-policy-128"]["source_commit"] == (
        DEPLOYED_EDULLM_DATA_COMMIT
    )
    assert report["policies"]["deployed-policy-128"]["pretrain_policy_sha256"] == (
        DEPLOYED_PRETRAIN_POLICY_SHA256
    )
    for policy in report["policies"].values():
        assert policy["status"] == "PASS"
        assert policy["validation"]["ok"] is True
        assert policy["validation"]["violations"] == []
        assert set(policy["sampled_paths"]) == {
            f"tokens/{family}/{split}-00000.u32le.bin"
            for family in tokenization.FAMILIES
            for split in ("train", "val")
        }
        assert all(count >= 5 for count in policy["sampled_paths"].values())
    assert report["policy_delta"] == {
        "outcomes_match": True,
        "only_image-pinned-local-256": [],
        "only_deployed-policy-128": [],
    }


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    (
        ("bad-endian", "vocab-out-of-range"),
        ("out-of-range", "vocab-out-of-range"),
        ("zero-run", "zero-run-in-shard"),
        ("eos", "eos-fraction-out-of-bounds"),
        ("diversity", "distinct-too-few"),
    ),
)
def test_synthetic_profile_gate_rejects_sampled_byte_faults(
    tmp_path,
    monkeypatch,
    fault,
    expected_code,
):
    modules = _load_exact_edullm_package(monkeypatch)
    staged, controls = _staged_payload_fixture(tmp_path)
    values = np.arange(16_384, dtype="<u4") % 1_024 + 1
    values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
    if fault == "bad-endian":
        payload = values.astype(">u4").tobytes()
    elif fault == "out-of-range":
        values = np.arange(16_384, dtype="<u4") % 1_024 + 151_665
        values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
        payload = values.tobytes()
    elif fault == "zero-run":
        for start in range(0, len(values), 512):
            values[start : start + 300] = 0
        values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
        payload = values.tobytes()
    elif fault == "eos":
        values[::10] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
        payload = values.tobytes()
    else:
        values = np.arange(16_384, dtype="<u4") % 8 + 1
        values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
        payload = values.tobytes()
    _replace_staged_shard_and_reseal(staged, controls, payload)

    report = tokenization.run_staged_pretrain_profile_gate(
        staged,
        train_manifest_path=controls / "train_meta.json",
        val_manifest_path=controls / "val_meta.json",
        tokenizer_source=_fixed_qwen_tokenizer_source(),
        scratch_root=tmp_path / "profile-gate",
        edullm_modules=modules,
        validator_policies=_validator_policy_fixtures(modules, tmp_path),
        payload_kind="synthetic",
    )

    assert report["status"] == "REPORT"
    for policy in report["policies"].values():
        assert policy["status"] == "REPORT"
        assert expected_code in {
            violation["code"] for violation in policy["validation"]["violations"]
        }


def test_synthetic_profile_gate_reports_local_256_deployed_128_diversity_delta(
    tmp_path,
    monkeypatch,
):
    modules = _load_exact_edullm_package(monkeypatch)
    staged, controls = _staged_payload_fixture(tmp_path)
    values = np.arange(16_384, dtype="<u4") % 200 + 1
    values[-1] = FIXED_QWEN_TOKENIZER_SEAL["tokenizer_eos_token_id"]
    _replace_staged_shard_and_reseal(staged, controls, values.tobytes())

    report = tokenization.run_staged_pretrain_profile_gate(
        staged,
        train_manifest_path=controls / "train_meta.json",
        val_manifest_path=controls / "val_meta.json",
        tokenizer_source=_fixed_qwen_tokenizer_source(),
        scratch_root=tmp_path / "profile-gate",
        edullm_modules=modules,
        validator_policies=_validator_policy_fixtures(modules, tmp_path),
        payload_kind="synthetic",
    )

    assert report["status"] == "REPORT"
    assert report["policies"]["image-pinned-local-256"]["status"] == "REPORT"
    assert report["policies"]["deployed-policy-128"]["status"] == "PASS"
    assert report["policy_delta"]["outcomes_match"] is False
    assert {item["code"] for item in report["policy_delta"]["only_image-pinned-local-256"]} == {
        "distinct-too-few"
    }
    assert report["policy_delta"]["only_deployed-policy-128"] == []


def test_real_staged_payload_gate_uses_explicit_paths_and_reports_policy_delta(
    tmp_path,
    monkeypatch,
):
    staged_value = os.environ.get("P3_REAL_STAGED_PAYLOAD")
    tokenized_value = os.environ.get("P3_REAL_TOKENIZED_DIR")
    tokenizer_value = os.environ.get("P3_FIXED_TOKENIZER_DIR")
    supplied = (staged_value, tokenized_value, tokenizer_value)
    if not any(supplied):
        pytest.skip(
            "real publication gate requires P3_REAL_STAGED_PAYLOAD, "
            "P3_REAL_TOKENIZED_DIR, and P3_FIXED_TOKENIZER_DIR; "
            "a skip cannot authorize publication"
        )
    assert all(supplied), (
        "real publication gate requires all of P3_REAL_STAGED_PAYLOAD, "
        "P3_REAL_TOKENIZED_DIR, and P3_FIXED_TOKENIZER_DIR"
    )
    modules = _load_exact_edullm_package(monkeypatch)
    staged = Path(staged_value).expanduser().resolve(strict=True)
    tokenized = Path(tokenized_value).expanduser().resolve(strict=True)
    tokenizer_source = Path(tokenizer_value).expanduser().resolve(strict=True)

    report = tokenization.run_staged_pretrain_profile_gate(
        staged,
        train_manifest_path=tokenized / "train_meta.json",
        val_manifest_path=tokenized / "val_meta.json",
        tokenizer_source=tokenizer_source,
        scratch_root=tmp_path / "real-profile-gate",
        edullm_modules=modules,
        validator_policies=_validator_policy_fixtures(modules, tmp_path),
        payload_kind="real",
    )
    print("P3_REAL_PRETRAIN_PROFILE_GATE_REPORT=" + json.dumps(report, sort_keys=True))

    assert report["payload_kind"] == "real"
    assert report["families"] == list(tokenization.FAMILIES)
    assert len(report["entries"]) >= 12
    assert all(report["partitions"][split]["files"] >= 6 for split in ("train", "val"))
    assert (
        report["status"] == "PASS"
    ), "REPORT requires manual review and does not authorize S3 upload: " + json.dumps(
        report["policy_delta"], sort_keys=True
    )


def test_process_revalidates_current_and_source_before_and_after_tokenization(
    tmp_path, monkeypatch
):
    source = tmp_path / "source" / "tiny.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps({"text": "A---\nGOALx", "mask_end": 9}) + "\n")
    _install_packing_tokenizer(monkeypatch, PackingTokenizer())
    checks = []
    binding = {
        "generation_id": "test-generation",
        "logical_root_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        tokenization,
        "require_corpus_generation_current",
        lambda current: checks.append(current),
    )
    monkeypatch.setattr(
        tokenization,
        "tokenizer_composite_seal",
        lambda *_args, **_kwargs: {
            "schema_version": "fixture-tokenizer-seal-v1",
            "tokenizer_composite_sha256": "b" * 64,
        },
    )
    corpus_binding = _test_corpus_binding()
    task = {
        "source": str(source),
        "source_record": {
            "family": "tiny",
            "sha256": file_sha256(source),
            "bytes": source.stat().st_size,
            "schema": "atp-v2",
        },
        "output_root": str(tmp_path / "out"),
        "cache_root": str(tmp_path / "cache"),
        "split": "train",
        "tokenizer": "fixture-tokenizer",
        "sequence_length": 32,
        "shard_tokens": 64,
        "pack": True,
        "batch_size": 2,
        "suggest": False,
        "test_only": False,
        "corpus_contract": binding,
        "corpus_generation": corpus_binding["corpus_generation"],
    }

    process_corpus(task)

    assert checks == [binding, binding]


def test_deferred_process_preserves_shards_without_publishing_done_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source" / "tiny.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps({"text": "A---\nGOALx", "mask_end": 9}) + "\n")
    _install_packing_tokenizer(monkeypatch, PackingTokenizer())
    task = {
        "source": str(source),
        "output_root": str(tmp_path / "out"),
        "cache_root": str(tmp_path / "cache"),
        "split": "train",
        "tokenizer": "fixture-tokenizer",
        "sequence_length": 32,
        "shard_tokens": 64,
        "pack": True,
        "batch_size": 2,
        "suggest": False,
        "defer_done_commit": True,
        **_test_corpus_binding(),
    }

    result = process_corpus(task)

    group = tmp_path / "out" / "tokens" / "tiny"
    assert result["name"] == "tiny"
    assert list(group.glob("*.u32le.bin"))
    assert not (group / "train.done.json").exists()
    assert not list(group.glob("*.pending"))


def test_completed_output_resume_refuses_orphan_file(tmp_path, monkeypatch):
    source = tmp_path / "x.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"row-{i}",
                    "text": f"{i}---\nGOALtarget",
                    "mask_end": 8,
                }
            )
            + "\n"
            for i in range(2)
        )
    )
    output = tmp_path / "out"
    _install_packing_tokenizer(monkeypatch, PackingTokenizer())
    task = {
        "source": str(source),
        "output_root": str(output),
        "cache_root": str(tmp_path / "cache"),
        "split": "train",
        "tokenizer": "fixture-tokenizer",
        "sequence_length": 32,
        "shard_tokens": 64,
        "pack": True,
        "batch_size": 2,
        "suggest": False,
        **_test_corpus_binding(),
    }
    process_corpus(task)
    orphan = output / "tokens" / "x" / "orphan.bin"
    orphan.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="inventory|unknown|unexpected"):
        process_corpus(task)


def test_cli_writes_v3_corpus_manifest_with_nested_shard_seals(tmp_path, monkeypatch):
    contract_root = tmp_path / "corpus-contract"
    source_dir = contract_root / "generations" / "test-generation" / "train"
    source_dir.mkdir(parents=True)
    (contract_root / ".generation.lock").write_bytes(b"")
    source = source_dir / "tiny.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"text": "A---\nGOALx", "mask_end": 9},
                {"text": "BB---\nGOALyy", "mask_end": 10},
            ]
        )
    )
    output = tmp_path / "v3"
    _install_packing_tokenizer(monkeypatch, PackingTokenizer())
    monkeypatch.setattr(
        tokenization,
        "tokenizer_composite_seal",
        lambda *_args, **_kwargs: {
            "schema_version": "p3-tokenizer-four-part-seal-v1",
            "tokenizer_composite_sha256": "b" * 64,
            "tokenizer_file_sha256": {},
            "tokenizers_version": "0.22.2",
            "tokenizer_eos_token_id": 250,
            "tokenizer_pad_token_id": 250,
            "separator": "\n---\nGOAL ",
            "separator_ids": PackingTokenizer.encode_text("---\nGOAL"),
        },
    )
    binding = {
        **_test_corpus_binding()["corpus_generation"],
        "contract_root": str(contract_root),
        "semantic_contract_sha256": "c" * 64,
        "producer_source_sha256": "d" * 64,
        "families": {
            "tiny": {
                "train": {
                    "family": "tiny",
                    "path": "generations/test-generation/train/tiny.jsonl",
                    "sha256": file_sha256(source),
                    "bytes": source.stat().st_size,
                    "schema": "atp-v2",
                }
            }
        },
    }
    monkeypatch.setattr(
        tokenization,
        "load_corpus_generation_contract",
        lambda _root: binding,
    )
    monkeypatch.setattr(
        tokenization,
        "require_corpus_generation_current",
        lambda _binding: None,
    )
    locked_commits = []
    real_commit = tokenization.commit_token_manifests_under_generation_lock

    def record_locked_commit(current_binding, replacements, **kwargs):
        locked_commits.append(tuple(replacements))
        return real_commit(current_binding, replacements, **kwargs)

    monkeypatch.setattr(
        tokenization,
        "commit_token_manifests_under_generation_lock",
        record_locked_commit,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize_corpus.py",
            "--corpus-contract-root",
            str(contract_root),
            "--out",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--tokenizer",
            "fixture-tokenizer",
            "--sequence-length",
            "32",
            "--shard-tokens",
            "64",
            "--batch-size",
            "2",
            "--jobs",
            "1",
            "--pack",
        ],
    )

    main()

    meta = json.loads((output / "train_meta.json").read_text())
    assert len(locked_commits) == 1
    assert {final.name for _, final in locked_commits[0]} == {
        "train.done.json",
        "train_meta.json",
    }
    assert meta["schema_version"] == "p3-packed-corpus-v3"
    assert meta["code_version"] == "tokenize-corpus-v4"
    assert meta["source_jsonl_sha256"] == {"tiny": file_sha256(source)}
    assert (
        meta["tokenizer_composite_sha256"] == meta["tokenizer_seal"]["tokenizer_composite_sha256"]
    )
    assert meta["corpus_generation"]["generation_id"] == "test-generation"
    assert meta["corpus_generation"]["semantic_contract_sha256"] == "c" * 64
    assert meta["corpus_generation"]["producer_source_sha256"] == "d" * 64
    assert "evaluator_dependency" not in meta
    assert "evaluator_dependency" not in meta["groups"]["tiny"]["build"]
    assert meta["packing_config"]["sequence_length"] == 32
    assert meta["packing_config"]["eos_token_id"] == 250
    assert meta["packing_config"]["pad_token_id"] == 250
    assert meta["packing_config"]["separator_ids"] == PackingTokenizer.encode_text("---\nGOAL")
    unsigned = dict(meta)
    manifest_sha256 = unsigned.pop("manifest_sha256")
    assert manifest_sha256 == fingerprint_dict(unsigned)
    for shard in meta["groups"]["tiny"]["shards"]:
        path = output / shard["path"]
        assert shard["sha256"] == file_sha256(path)
        assert shard["tokens"] * 4 == shard["bytes"] == path.stat().st_size
        assert shard["tokens_dtype"] == "uint32"
        assert shard["byte_order"] == "little"
