"""Dry-run tests for eval/formal-proof-premises-500m v1 publication route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
EDULLM_DATA_SRC = ROOT / ".p3-work" / "full13" / "edullm-data" / "src"
EDULLM_DATA_ROOT = ROOT / ".p3-work" / "full13" / "edullm-data"
if str(EDULLM_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(EDULLM_DATA_SRC))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from edullm_data import publish as P  # noqa: E402
from edullm_data import validate as V  # noqa: E402
from edullm_data.manifest import Format, ManifestEntry, build_manifest, canonical_json, manifest_sha256  # noqa: E402
from edullm_data.profiles.registry import available  # noqa: E402
from edullm_data.s3 import FakeS3  # noqa: E402
from publish_eval_v1_support import (  # noqa: E402
    compute_publisher_code_sha256,
    provenance_report,
    resolve_execute_provenance,
    run_streaming_preflight,
)

DATASET_ID = "eval/formal-proof-premises-500m"
PRETRAIN_ID = "pretrain/formal-proof-premises-500m"
PROFILE = "p3-evaluator-corpus/v1"
CREATED = "2026-08-06T18:00:00Z"
ENV = {"EDULLM_CODE_SHA256": "a" * 64, "EDULLM_PACKAGES_LOCK_SHA256": "b" * 64}
FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_row(path: Path, *, family: str, row_id: str) -> None:
    text = "facts block\n---\nGOAL\nproof"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": SCHEMAS[family],
        "id": row_id,
        "facts": {"f": "statement"},
        "goal": "goal",
        "target": "proof",
        "text": text,
        "mask_start": 0,
        "mask_end": text.index("---"),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _build_minimal_stage(root: Path) -> dict:
    evaluator = root / "evaluator"
    families = {}
    for family in FAMILIES:
        train_path = evaluator / "shards" / f"{family}.jsonl"
        eval_path = evaluator / "eval" / f"{family}.jsonl"
        _write_row(train_path, family=family, row_id=f"{family}-train")
        _write_row(eval_path, family=family, row_id=f"{family}-eval")
        families[family] = {
            "schema": SCHEMAS[family],
            "train": {
                "path": f"shards/{family}.jsonl",
                "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "bytes": train_path.stat().st_size,
                "rows": 1,
            },
            "eval": {
                "path": f"eval/{family}.jsonl",
                "sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                "bytes": eval_path.stat().st_size,
                "rows": 1,
            },
        }
    sidecars = {
        "heldout/atp.json": {"facts": ["f1"]},
        "heldout/isabelle.json": {"facts": ["f2"]},
        "heldout/metamath.json": {"facts": ["f3"]},
        "heldout/mizar.json": {"facts": ["f4"]},
        "metamath_sources.json": {"sources": ["s1"]},
    }
    sidecar_records = {}
    for relative, payload in sidecars.items():
        path = evaluator / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        sidecar_records[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    body = {
        "schema_version": "p3-evaluator-corpus-v1",
        "source_seal": {
            "schema_version": "p3-sealed-corpus-manifest-v1",
            "manifest_root_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
        },
        "families": families,
        "sidecars": sidecar_records,
        "total_train_rows": len(FAMILIES),
        "total_eval_rows": len(FAMILIES),
    }
    manifest = {**body, "evaluator_root_sha256": _fingerprint(body)}
    (evaluator / "evaluator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _seed_pretrain_parent(s3: FakeS3) -> str:
    body = b"\x00" * 16
    manifest = build_manifest(
        [
            ManifestEntry(
                path="tokens/train-00000.u32le.bin",
                sha256="c" * 64,
                bytes=len(body),
                count={"unit": "tokens", "value": 4},
                format=Format(
                    container="raw",
                    dtype="uint32",
                    byte_order="little",
                    header_bytes=0,
                    codec="none",
                ),
            )
        ],
        group_name="tokens",
    )
    prefix = f"{PRETRAIN_ID}/v3"
    s3.seed("edullm-data", f"{prefix}/tokens/manifest.json", canonical_json(manifest))
    s3.seed("edullm-data", f"{prefix}/tokens/train-00000.u32le.bin", body)
    dataset = {
        "groups": [
            {
                "name": "tokens",
                "manifest": "tokens/manifest.json",
                "manifest_sha256": manifest_sha256(manifest),
            }
        ]
    }
    s3.seed("edullm-data", f"{prefix}/dataset.json", canonical_json(dataset))
    return manifest_sha256(manifest)


def test_profile_is_registered():
    assert "p3-evaluator-corpus/v1" in available()


def test_stage_eval_v1_from_zip(tmp_path):
    stage = _load_module("stage_eval_v1", ROOT / "scripts" / "stage_eval_v1.py")
    zip_path = ROOT / "corpus-v3.zip"
    if not zip_path.is_file():
        pytest.skip("corpus-v3.zip not present")
    out = tmp_path / "stage"
    report = stage.stage_from_zip(zip_path, out, force=True)
    assert report["evaluator_root_sha256"] == stage.EXPECTED_EVALUATOR_ROOT
    assert report["total_train_rows"] == 181_652
    assert report["total_eval_rows"] == 4_191
    assert (out / "evaluator" / "evaluator_manifest.json").is_file()


def test_publish_eval_v1_roundtrip_minimal(tmp_path):
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    manifest = _build_minimal_stage(stage_root)

    s3 = FakeS3()
    parent_manifest_sha = _seed_pretrain_parent(s3)
    group_meta = {
        "evaluator": {
            "evaluator_root_sha256": manifest["evaluator_root_sha256"],
            "source_seal_root_sha256": manifest["source_seal"]["manifest_root_sha256"],
            "coverage": "incomplete",
            "depends_on": [
                {
                    "role": "pretrain",
                    "dataset_id": PRETRAIN_ID,
                    "version": "v3",
                    "manifest_sha256": parent_manifest_sha,
                }
            ],
        }
    }
    plan = P.publish(
        stage_root,
        dataset_id=DATASET_ID,
        purpose="Sealed P3 evaluator corpus for run_eval.py smoke validation",
        profile=PROFILE,
        s3=s3,
        created_at=CREATED,
        group_meta=group_meta,
        env=ENV,
    )
    assert plan.version == "v1"
    result = V.validate_dataset(
        "edullm-landing",
        f"{plan.dataset_id}/{plan.version}",
        s3,
        data_bucket="edullm-data",
    )
    assert result.ok, [str(v) for v in result.violations]


def test_preflight_does_not_whole_read_payload(tmp_path, monkeypatch):
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    manifest = _build_minimal_stage(stage_root)
    large = stage_root / "evaluator" / "shards" / "large.jsonl"
    large.write_bytes(b"x" * (2 * 1024 * 1024))

    def _forbid_whole_read(self, size=-1, /):  # noqa: ANN001
        if size in (-1, 0) or size > 1024 * 1024:
            raise AssertionError("whole-payload read_bytes is forbidden during preflight")
        return Path.read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _forbid_whole_read, raising=True)

    report = run_streaming_preflight(
        stage_root,
        {
            "evaluator_root_sha256": manifest["evaluator_root_sha256"],
            "source_seal_root_sha256": manifest["source_seal"]["manifest_root_sha256"],
        },
        family={"defaults": {}},
    )
    assert report["inventory"]["objects"] >= 18
    assert report["preflight_peak_read_window_bytes"] <= 8 * 1024 * 1024


def test_provenance_report_blocks_local_execute_without_lockfile():
    report = provenance_report(EDULLM_DATA_ROOT)
    assert report["local_execute_allowed"] is False
    assert report["packages_lock_path"] is None
    assert "local_execute_blocker" in report
    assert compute_publisher_code_sha256(EDULLM_DATA_ROOT) == report["publisher_code_sha256"]


def test_execute_provenance_rejects_placeholder_hashes():
    env = {
        "EDULLM_CODE_SHA256": "local",
        "EDULLM_PACKAGES_LOCK_SHA256": "unpinned-local",
    }
    executor, errors = resolve_execute_provenance(env, edullm_root=EDULLM_DATA_ROOT)
    assert executor is None
    assert any("EDULLM_CODE_SHA256" in err for err in errors)
    assert any("lockfile" in err.lower() or "LOCK" in err for err in errors)


def test_execute_provenance_accepts_batch_executor():
    executor, errors = resolve_execute_provenance(
        {"AWS_BATCH_JOB_ID": "job-123", "AWS_REGION": "us-east-1"},
        edullm_root=EDULLM_DATA_ROOT,
    )
    assert errors == []
    assert executor is not None
    assert executor["kind"] == "aws-batch"
