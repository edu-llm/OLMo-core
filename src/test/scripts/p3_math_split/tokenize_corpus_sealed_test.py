"""Fail-closed contract for the direct ``--sealed-corpus-manifest`` seam.

These pin the production alternative to the atomic transaction contract: a
sealed six-family manifest that binds exact train/eval JSONL bytes by SHA-256.
The loader must reject any missing/extra family, wrong row schema, malformed
entry, or manifest-root drift before tokenization can proceed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from tokenize_corpus import (  # noqa: E402
    FAMILIES,
    P3_SOURCE_SCHEMAS,
    SEALED_CORPUS_MANIFEST_SCHEMA_VERSION,
    fingerprint_dict,
    load_sealed_corpus_manifest,
)


def _entry(name: str) -> dict:
    return {"path": f"/tmp/{name}.jsonl", "sha256": "a" * 64, "bytes": 10, "rows": 5}


def _valid_families() -> dict:
    return {
        family: {
            "schema": P3_SOURCE_SCHEMAS[family],
            "train": _entry(f"{family}-train"),
            "eval": _entry(f"{family}-eval"),
        }
        for family in FAMILIES
    }


def _manifest(families: dict) -> dict:
    body = {"schema_version": SEALED_CORPUS_MANIFEST_SCHEMA_VERSION, "families": families}
    return {**body, "manifest_root_sha256": fingerprint_dict(body)}


def _write(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(manifest))
    return path


def test_accepts_a_wellformed_six_family_manifest(tmp_path):
    manifest = _manifest(_valid_families())
    resolved = load_sealed_corpus_manifest(_write(tmp_path, manifest))
    assert resolved["schema_version"] == SEALED_CORPUS_MANIFEST_SCHEMA_VERSION
    assert resolved["manifest_root_sha256"] == manifest["manifest_root_sha256"]
    assert set(resolved["families"]) == set(FAMILIES)


def test_rejects_unrecognized_schema_version(tmp_path):
    manifest = _manifest(_valid_families())
    manifest["schema_version"] = "p3-sealed-corpus-manifest-v0"
    with pytest.raises(RuntimeError, match="schema_version"):
        load_sealed_corpus_manifest(_write(tmp_path, manifest))


def test_rejects_a_missing_family(tmp_path):
    families = _valid_families()
    del families["isabelle"]
    with pytest.raises(RuntimeError, match="ordered P3 family set"):
        load_sealed_corpus_manifest(_write(tmp_path, _manifest(families)))


def test_rejects_an_extra_family(tmp_path):
    families = _valid_families()
    families["surprise"] = {
        "schema": "atp-v2",
        "train": _entry("surprise-train"),
        "eval": _entry("surprise-eval"),
    }
    with pytest.raises(RuntimeError, match="ordered P3 family set"):
        load_sealed_corpus_manifest(_write(tmp_path, _manifest(families)))


def test_rejects_a_wrong_row_schema(tmp_path):
    families = _valid_families()
    families["mizar"]["schema"] = "atp-v2"
    with pytest.raises(RuntimeError, match="wrong row schema"):
        load_sealed_corpus_manifest(_write(tmp_path, _manifest(families)))


def test_rejects_a_malformed_entry(tmp_path):
    families = _valid_families()
    families["prf2"]["train"]["sha256"] = "deadbeef"  # not 64 hex chars
    with pytest.raises(RuntimeError, match="malformed"):
        load_sealed_corpus_manifest(_write(tmp_path, _manifest(families)))


def test_rejects_manifest_root_tampering(tmp_path):
    manifest = _manifest(_valid_families())
    # Silently mutate a byte count without re-sealing the root.
    manifest["families"]["enigma"]["eval"]["bytes"] = 999
    with pytest.raises(RuntimeError, match="root SHA-256 does not seal"):
        load_sealed_corpus_manifest(_write(tmp_path, manifest))


def test_rejects_a_symlinked_manifest(tmp_path):
    manifest = _manifest(_valid_families())
    real = tmp_path / "real.json"
    real.write_text(json.dumps(manifest))
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        load_sealed_corpus_manifest(link)
