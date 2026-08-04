"""Immutable model/tokenizer provenance shared by P3 training and export."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "src" / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

import provenance  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _independent_behavior_digest(tokenizer: Tokenizer) -> str:
    payload = {
        "schema": "qwen-tokenizer-behavior-v1",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "eos_token": provenance.TOKENIZER_EOS_TOKEN,
        "eos_token_id": tokenizer.token_to_id(provenance.TOKENIZER_EOS_TOKEN),
        "probes": [],
    }
    for text in provenance.TOKENIZER_BEHAVIOR_PROBES:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        payload["probes"].append(
            {
                "text": text,
                "ids": encoding.ids,
                "tokens": encoding.tokens,
                "offsets": [list(pair) for pair in encoding.offsets],
                "decoded": tokenizer.decode(encoding.ids, skip_special_tokens=False),
            }
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(encoded)


@pytest.fixture
def tokenizer_bytes(tmp_path, monkeypatch):
    vocab = {
        "[UNK]": 0,
        "<|endoftext|>": 1,
        "---": 2,
        "GOAL": 3,
        "proof": 4,
        "Unicode": 5,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps(config), encoding="utf-8")

    json_bytes = tokenizer_json.read_bytes()
    config_bytes = tokenizer_config.read_bytes()
    monkeypatch.setattr(
        provenance,
        "TOKENIZER_FILE_SHA256",
        {
            "tokenizer.json": _sha256(json_bytes),
            "tokenizer_config.json": _sha256(config_bytes),
        },
    )
    monkeypatch.setattr(provenance, "TOKENIZER_BACKEND_VOCAB_SIZE", len(vocab))
    monkeypatch.setattr(provenance, "TOKENIZER_EOS_TOKEN_ID", 1)
    monkeypatch.setattr(provenance, "TOKENIZER_PAD_TOKEN_ID", 1)
    original_qwen_config = provenance.qwen2_tokenizer_config

    def tiny_qwen_config():
        config = original_qwen_config()
        config.eos_token_id = 1
        config.pad_token_id = 1
        return config

    monkeypatch.setattr(provenance, "qwen2_tokenizer_config", tiny_qwen_config)
    monkeypatch.setattr(
        provenance,
        "TOKENIZER_COMPOSITE_SHA256",
        _independent_behavior_digest(tokenizer),
    )
    return {
        "tokenizer.json": json_bytes,
        "tokenizer_config.json": config_bytes,
    }


def test_approved_tokenizer_four_part_seal_is_pinned():
    assert provenance.TOKENIZER_ARTIFACT_ID == "tokenizer/qwen25-vendored"
    assert provenance.TOKENIZER_ARTIFACT_VERSION == "v1"
    assert provenance.TOKENIZER_FILE_SHA256 == {
        "tokenizer.json": "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
        "tokenizer_config.json": (
            "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
        ),
    }
    assert (
        provenance.TOKENIZER_COMPOSITE_SHA256
        == "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
    )
    assert provenance.TOKENIZERS_VERSION == "0.22.2"
    assert provenance.TOKENIZER_EOS_TOKEN_ID == provenance.TOKENIZER_PAD_TOKEN_ID == 151_643


def test_fetches_exact_version_and_seals_downloaded_bytes(
    tokenizer_bytes, tmp_path, monkeypatch
):
    calls = []
    paths = [
        (
            "s3://edullm-data/tokenizer/qwen25-vendored/v1/"
            f"tokenizer/{filename}"
        )
        for filename in tokenizer_bytes
    ]

    def dataset_paths(dataset_id, version, *, s3):
        calls.append((dataset_id, version, s3))
        return SimpleNamespace(paths=paths)

    class S3:
        def get(self, bucket, key):
            calls.append((bucket, key))
            return tokenizer_bytes[Path(key).name]

    sealed = provenance.fetch_tokenizer_artifact(
        provenance.TOKENIZER_ARTIFACT,
        tmp_path / "cache",
        dataset_paths_fn=dataset_paths,
        s3=S3(),
    )

    assert calls[0][:2] == ("tokenizer/qwen25-vendored", "v1")
    assert sealed.artifact_id == "tokenizer/qwen25-vendored"
    assert sealed.artifact_version == "v1"
    assert sealed.file_sha256 == provenance.TOKENIZER_FILE_SHA256
    assert sealed.composite_sha256 == provenance.TOKENIZER_COMPOSITE_SHA256
    assert sealed.tokenizers_version == provenance.TOKENIZERS_VERSION
    assert sealed.eos_token_id == sealed.pad_token_id == 1
    assert sealed.separator_ids("---\nGOAL")
    assert sealed.olmo_config().identifier == provenance.TOKENIZER_ARTIFACT


def test_same_sealed_bytes_have_path_independent_semantic_hashes(
    tokenizer_bytes, tmp_path
):
    first = tmp_path / "rank0"
    second = tmp_path / "different-host-path" / "rank7"
    for root in (first, second):
        root.mkdir(parents=True)
        for filename, payload in tokenizer_bytes.items():
            (root / filename).write_bytes(payload)

    left = provenance.seal_tokenizer_files(first)
    right = provenance.seal_tokenizer_files(second)

    assert left.file_sha256 == right.file_sha256
    assert left.composite_sha256 == right.composite_sha256
    assert left.provenance_dict() == right.provenance_dict()


def test_refuses_tokenizer_byte_drift_before_use(tokenizer_bytes, tmp_path):
    root = tmp_path / "drift"
    root.mkdir()
    for filename, payload in tokenizer_bytes.items():
        (root / filename).write_bytes(payload)
    with (root / "tokenizer_config.json").open("ab") as config:
        config.write(b"\n")

    with pytest.raises(RuntimeError, match="tokenizer_config.json SHA-256"):
        provenance.seal_tokenizer_files(root)


def test_refuses_unversioned_or_different_tokenizer_artifact(tmp_path):
    for artifact in (
        "tokenizer/qwen25-vendored",
        "tokenizer/qwen25-vendored/v2",
        "Qwen/Qwen2.5-0.5B",
    ):
        with pytest.raises(ValueError, match="pinned"):
            provenance.fetch_tokenizer_artifact(artifact, tmp_path)
