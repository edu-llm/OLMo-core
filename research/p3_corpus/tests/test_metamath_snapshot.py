"""The Metamath corpus and its deterministic evaluator must use one snapshot."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
FIXED_TOKENIZER = (
    Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
)


def test_source_manifest_accepts_only_the_pinned_files(tmp_path, monkeypatch):
    import build_metamath_shard as builder

    payloads = {db: f"{db} source".encode() for db in builder.DBS}
    monkeypatch.setattr(
        builder,
        "SOURCE_SHA256",
        {db: hashlib.sha256(payload).hexdigest() for db, payload in payloads.items()},
    )
    for db, payload in payloads.items():
        (tmp_path / f"{db}.mm").write_bytes(payload)

    manifest = builder.source_manifest(tmp_path)
    assert manifest["commit"] == builder.SOURCE_COMMIT
    assert set(manifest["files"]) == {"set.mm", "iset.mm", "nf.mm"}

    (tmp_path / "set.mm").write_bytes(b"different snapshot")
    with pytest.raises(SystemExit, match="not pinned commit"):
        builder.source_manifest(tmp_path)


def test_fixed_qwen_tokenizer_seal_is_exact_and_eos_is_counted():
    import build_metamath_shard as builder

    tokenizer, seal = builder.load_fixed_qwen_tokenizer(FIXED_TOKENIZER)

    assert seal == {
        "identity": "Qwen/Qwen2.5-0.5B",
        "tokenizer_json_sha256": (
            "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
        ),
        "tokenizer_config_sha256": (
            "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
        ),
        "behavior_digest": (
            "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
        ),
        "tokenizers_version": "0.22.2",
        "eos_token_id": 151643,
        "max_text_plus_eos_tokens": 16_384,
    }
    assert seal == builder.FIXED_QWEN_TOKENIZER_SEAL
    assert builder.count_text_plus_eos_tokens(tokenizer, "") == 1
    assert tokenizer.eos_token_id == 151643
