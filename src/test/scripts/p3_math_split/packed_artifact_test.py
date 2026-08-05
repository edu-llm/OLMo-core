"""Integration checks for the published raw-shard format.

Set ``TOKENIZED_DIR`` to the resumable working payload directory containing
``train_meta.json`` / ``val_meta.json``. The tests skip in the container source
tree, where those local artifacts intentionally do not exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from tokenize_corpus import (
    CROSS_SPLIT_BINDING_SCHEMA_VERSION,
    FAMILIES,
    FIXED_QWEN_TOKENIZER_SEAL,
    P3_SOURCE_SCHEMAS,
    PACKED_ALGORITHM_VERSION,
    PACKED_CORPUS_SCHEMA_VERSION,
    PACKED_GROUP_SCHEMA_VERSION,
    TOKENIZE_CORPUS_CODE_VERSION,
    _token_manifest_cross_split_binding,
    cache_root_sha256,
    file_sha256,
    fingerprint_dict,
    group_completion_sha256,
    require_exact_group_inventory,
)
from train_module import DerivedMaskTrainModule

TOKENIZED = Path(os.environ.get("TOKENIZED_DIR", "artifacts/public"))
CACHE_ROOT = os.environ.get("TOKEN_CACHE_DIR")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _assert_no_evaluator_fields(value, *, context: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert (
                "evaluator" not in str(key).lower()
            ), f"{context} contains forbidden evaluator field {key!r}"
            _assert_no_evaluator_fields(nested, context=context)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_evaluator_fields(nested, context=context)


@pytest.fixture(scope="module", params=("train", "val"))
def artifact(request):
    meta_path = TOKENIZED / f"{request.param}_meta.json"
    if not meta_path.exists():
        pytest.skip(f"tokenized artifacts unavailable: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if (
        meta.get("schema_version") != "p3-packed-corpus-v3"
        or meta.get("code_version") != "tokenize-corpus-v4"
    ):
        pytest.skip(
            f"legacy token artifact is intentionally unauditable as v3/v4: "
            f"{meta.get('schema_version')}/{meta.get('code_version')}; rebuild required"
        )
    return TOKENIZED, meta, request.param


def test_manifest_arithmetic_and_files(artifact):
    root, meta, split = artifact
    unsigned = dict(meta)
    declared_root = unsigned.pop("manifest_sha256")
    assert declared_root == fingerprint_dict(unsigned)
    assert meta["schema_version"] == PACKED_CORPUS_SCHEMA_VERSION
    assert meta["code_version"] == TOKENIZE_CORPUS_CODE_VERSION
    assert meta["cross_split_binding_schema_version"] == CROSS_SPLIT_BINDING_SCHEMA_VERSION
    _assert_no_evaluator_fields(meta, context=f"{split} final manifest")

    generation = meta["corpus_generation"]
    assert generation["schema_version"] == "p3-tokenizer-corpus-binding-v1"
    for field in (
        "logical_root_sha256",
        "manifest_root_sha256",
        "manifest_file_sha256",
        "current_sha256",
        "semantic_contract_sha256",
        "producer_source_sha256",
    ):
        assert SHA256_RE.fullmatch(generation[field])
    assert isinstance(generation["generation_id"], str) and generation["generation_id"]
    assert meta["tokenizer_seal"] == FIXED_QWEN_TOKENIZER_SEAL
    assert (
        meta["tokenizer_composite_sha256"]
        == FIXED_QWEN_TOKENIZER_SEAL["tokenizer_composite_sha256"]
    )
    assert meta["tokens_dtype"] == "uint32"
    assert meta["byte_order"] == "little"
    assert meta["sequence_length"] == 16_384
    assert meta["packed"] is True
    assert meta["eos_token_id"] == meta["pad_token_id"] == 151_643
    assert meta["separator"] == "\n---\nGOAL "
    assert meta["separator_search"] == "---\nGOAL"
    assert meta["separator_ids"] == [10952, 15513, 969]
    assert set(meta["source_jsonl_sha256"]) == set(FAMILIES)
    assert set(meta["source_family_inventory"]) == set(FAMILIES)
    assert set(meta["groups"]) == set(FAMILIES)
    assert all(meta["groups"][family]["shards"] for family in FAMILIES)
    assert meta["source_family_inventory"] == {
        family: {"family": family, "schema": P3_SOURCE_SCHEMAS[family]} for family in FAMILIES
    }
    assert all(SHA256_RE.fullmatch(meta["source_jsonl_sha256"][family]) for family in FAMILIES)

    packing = meta["packing_config"]
    assert packing == {
        "algorithm": PACKED_ALGORITHM_VERSION,
        "split": split,
        "packed": True,
        "sequence_length": 16_384,
        "shard_tokens": packing["shard_tokens"],
        "tokens_dtype": "uint32",
        "byte_order": "little",
        "eos_token_id": 151_643,
        "pad_token_id": 151_643,
        "separator": "\n---\nGOAL ",
        "separator_search": "---\nGOAL",
        "separator_ids": [10952, 15513, 969],
    }
    assert isinstance(packing["shard_tokens"], int) and packing["shard_tokens"] > 0

    assert sum(g["instances"] for g in meta["groups"].values()) == meta["instances"]
    assert sum(g["real_tokens"] for g in meta["groups"].values()) == meta["real_tokens"]
    assert (
        sum(g["dropped_over_length"] for g in meta["groups"].values())
        == meta["dropped_over_length"]
    )
    assert (
        sum(g["straddling"] for g in meta["groups"].values()) == meta["tokens_straddling_boundary"]
    )
    seen_paths = set()
    seen_sha256 = set()
    for name, group in meta["groups"].items():
        require_exact_group_inventory(root / "tokens" / name)
        _assert_no_evaluator_fields(group, context=f"{split}/{name} done manifest")
        assert group["shards"], f"{split}/{name} must contain one or more shards"
        assert group["schema_version"] == PACKED_GROUP_SCHEMA_VERSION
        assert group["code_version"] == TOKENIZE_CORPUS_CODE_VERSION
        assert group["cross_split_binding_schema_version"] == CROSS_SPLIT_BINDING_SCHEMA_VERSION
        assert group["name"] == name
        assert group["fingerprint"] == fingerprint_dict(group["build"])
        assert group["completion_sha256"] == group_completion_sha256(group)
        assert group["corpus_generation"] == generation
        assert group["build"]["schema_version"] == PACKED_GROUP_SCHEMA_VERSION
        assert group["build"]["code_version"] == TOKENIZE_CORPUS_CODE_VERSION
        assert group["build"]["corpus_generation"] == generation
        assert group["build"]["tokenizer"] == FIXED_QWEN_TOKENIZER_SEAL
        assert group["build"]["packing"] == packing
        assert group["build"]["source_jsonl"]["family"] == name
        assert group["build"]["source_jsonl"]["schema"] == P3_SOURCE_SCHEMAS[name]
        assert group["build"]["source_jsonl"]["sha256"] == meta["source_jsonl_sha256"][name]
        assert group["documents"] + group["dropped_over_length"] == group["source_documents"]
        assert sum(shard["instances"] for shard in group["shards"]) == group["instances"]
        assert sum(shard["tokens"] for shard in group["shards"]) == (
            group["instances"] * meta["sequence_length"]
        )
        assert sum(shard["bytes"] for shard in group["shards"]) == (
            group["instances"] * meta["sequence_length"] * 4
        )
        assert group["real_tokens"] <= group["instances"] * meta["sequence_length"]
        assert group["real_tokens"] > 0
        ordinals = []
        for shard in group["shards"]:
            path = root / shard["path"]
            match = re.fullmatch(
                rf"tokens/{re.escape(name)}/{split}-(\d{{5}})\.u32le\.bin",
                shard["path"],
            )
            assert match is not None
            ordinals.append(int(match.group(1)))
            assert shard["path"] not in seen_paths
            seen_paths.add(shard["path"])
            assert path.stat().st_size == shard["bytes"] == shard["tokens"] * 4
            assert file_sha256(path) == shard["sha256"]
            assert SHA256_RE.fullmatch(shard["sha256"])
            assert shard["sha256"] not in seen_sha256
            seen_sha256.add(shard["sha256"])
            assert shard["tokens"] == shard["instances"] * meta["sequence_length"]
            assert shard["tokens_dtype"] == "uint32"
            assert shard["byte_order"] == "little"
            assert shard["tokens"] % meta["sequence_length"] == 0
        assert ordinals == list(range(len(group["shards"])))

        done_path = root / "tokens" / name / f"{split}.done.json"
        done = json.loads(done_path.read_text())
        _assert_no_evaluator_fields(done, context=f"{split}/{name} persisted done manifest")
        assert done["completion_sha256"] == group["completion_sha256"]
        assert done["build"] == group["build"]

    assert {path.name for path in (root / "tokens").iterdir()} == set(FAMILIES)
    assert seen_paths == {
        shard["path"] for group in meta["groups"].values() for shard in group["shards"]
    }
    assert seen_paths == {
        path.relative_to(root).as_posix()
        for path in (root / "tokens").rglob(f"{split}-*.u32le.bin")
    }
    assert sum(
        shard["tokens"] for group in meta["groups"].values() for shard in group["shards"]
    ) == (meta["instances"] * meta["sequence_length"])
    assert sum(
        shard["bytes"] for group in meta["groups"].values() for shard in group["shards"]
    ) == (meta["instances"] * meta["sequence_length"] * 4)


def test_train_and_val_manifests_have_identical_cross_split_binding():
    manifests = {}
    for split in ("train", "val"):
        path = TOKENIZED / f"{split}_meta.json"
        if not path.exists():
            pytest.skip(f"both split manifests are required for cross-split audit: {path}")
        manifest = json.loads(path.read_text())
        if (
            manifest.get("schema_version") != "p3-packed-corpus-v3"
            or manifest.get("code_version") != "tokenize-corpus-v4"
            or manifest.get("cross_split_binding_schema_version")
            != "p3-token-cross-split-binding-v1"
        ):
            pytest.skip(f"{path} predates the accepted cross-split binding; fresh rebuild required")
        manifests[split] = manifest

    for split, manifest in manifests.items():
        unsigned = dict(manifest)
        assert unsigned.pop("manifest_sha256") == fingerprint_dict(unsigned)
        _assert_no_evaluator_fields(manifest, context=f"{split} final manifest")
        assert set(manifest["groups"]) == set(FAMILIES)
        assert set(manifest["source_family_inventory"]) == set(FAMILIES)
        assert manifest["tokenizer_seal"] == FIXED_QWEN_TOKENIZER_SEAL
        assert all(manifest["groups"][family]["shards"] for family in FAMILIES)
    assert _token_manifest_cross_split_binding(
        manifests["train"]
    ) == _token_manifest_cross_split_binding(manifests["val"])
    all_shards = [
        shard
        for manifest in manifests.values()
        for group in manifest["groups"].values()
        for shard in group["shards"]
    ]
    assert len({shard["path"] for shard in all_shards}) == len(all_shards)
    assert len({shard["sha256"] for shard in all_shards}) == len(all_shards)


def test_every_encoding_cache_payload_and_chunk_rehashes(artifact):
    _, meta, split = artifact
    if CACHE_ROOT is None:
        pytest.skip(
            "set TOKEN_CACHE_DIR to audit v3 working caches; caches are not published payloads"
        )
    cache_root = Path(CACHE_ROOT)
    for name, group in meta["groups"].items():
        cache = cache_root / split / name / group["cache_fingerprint"][:20]
        marker_path = cache / "cache.json"
        assert marker_path.is_file(), f"missing v3 cache marker: {marker_path}"
        marker = json.loads(marker_path.read_text())
        _assert_no_evaluator_fields(marker, context=f"{split}/{name} encoding cache")
        assert marker["schema_version"] == "p3-encoding-cache-v3"
        assert marker["code_version"] == "tokenize-corpus-v4"
        assert marker["status"] == "complete"
        assert marker["cache_root_sha256"] == cache_root_sha256(marker)
        assert marker["cache_root_sha256"] == group["cache_root_sha256"]
        assert marker["build"]["tokenizer"] == FIXED_QWEN_TOKENIZER_SEAL
        assert {path.name for path in cache.iterdir()} == {
            "cache.json",
            "tokens.u32le.bin",
            "offsets.u64le.bin",
        }
        for payload in marker["payloads"].values():
            path = cache / payload["path"]
            assert path.stat().st_size == payload["bytes"]
            assert file_sha256(path) == payload["sha256"]
        tokens = cache / marker["payloads"]["tokens"]["path"]
        offsets = cache / marker["payloads"]["offsets"]["path"]
        for chunk in marker["chunks"]:
            for path, byte_range, field in (
                (tokens, chunk["token_bytes"], "tokens_sha256"),
                (offsets, chunk["offset_bytes"], "offsets_sha256"),
            ):
                with path.open("rb") as handle:
                    handle.seek(byte_range["start"])
                    payload = handle.read(byte_range["end"] - byte_range["start"])
                assert hashlib.sha256(payload).hexdigest() == chunk[field]


def test_sampled_rows_decode_to_in_range_ids_and_have_one_or_more_documents(artifact):
    root, meta, _ = artifact
    eos = meta["eos_token_id"]
    for group in meta["groups"].values():
        for shard in group["shards"]:
            a = np.memmap(root / shard["path"], mode="r", dtype="<u4")
            rows = a.reshape(-1, meta["sequence_length"])
            for i in sorted({0, len(rows) // 2, len(rows) - 1}):
                row = rows[i]
                assert int(row.max()) < 151_936
                assert np.count_nonzero(row == eos) >= 1


def test_real_derived_mask_on_sampled_packed_rows(artifact):
    root, meta, _ = artifact
    eos = meta["eos_token_id"]
    model = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    model._sep = torch.tensor(meta["separator_ids"], dtype=torch.long)
    model.eos_token_id = eos
    model.pad_token_id = meta["pad_token_id"]
    model.arm = "split"

    masked = real = checked = expected_checked = 0
    for group in meta["groups"].values():
        for shard in group["shards"]:
            rows = np.memmap(root / shard["path"], mode="r", dtype="<u4").reshape(
                -1, meta["sequence_length"]
            )
            sample_indices = sorted({0, len(rows) // 2, len(rows) - 1})
            expected_checked += len(sample_indices)
            for i in sample_indices:
                ids = torch.from_numpy(np.asarray(rows[i], dtype=np.int64).copy()).unsqueeze(0)
                supervised = model.supervised_mask(ids)
                padding = model.padding_mask(ids)
                labels_live = model.label_supervision_mask(ids)
                assert not torch.any(supervised & padding)
                target_is_padding = torch.zeros_like(padding)
                target_is_padding[:, :-1] = padding[:, 1:]
                assert not torch.any(labels_live & target_is_padding)
                assert not labels_live[:, -1].any()
                # Every packed row has at least one fact block and one target.
                assert torch.any(supervised)
                assert torch.any(~supervised & ~padding)
                # Real EOS is supervised; only repeated-EOS tail padding is not.
                eos_positions = torch.nonzero(ids[0] == eos).flatten()
                assert torch.any(supervised[0, eos_positions] & ~padding[0, eos_positions])
                # OLMo shifts labels left. The position immediately before each first
                # goal token must therefore be live, or the first goal token is skipped.
                sep = model._sep
                starts = torch.nonzero((ids.unfold(1, len(sep), 1) == sep).all(dim=-1)[0]).flatten()
                assert len(starts) >= 1
                for start in starts:
                    assert labels_live[0, start + len(sep) - 1]
                masked += int((~labels_live & ~target_is_padding).sum())
                real += int((~target_is_padding).sum())
                checked += 1
    fraction = masked / real
    assert checked == expected_checked
    assert checked >= len(FAMILIES)
    assert 0.05 < fraction < 0.60, f"fact mask fraction {fraction:.2%} is implausible"
