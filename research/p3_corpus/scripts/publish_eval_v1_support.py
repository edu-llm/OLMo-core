"""Bounded-memory preflight and provenance helpers for publish_eval_v1.py."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from edullm_data.publish import _CONTROL_BASENAMES, _build_executor_from_env  # noqa: E402
from edullm_data.s3 import NotFound  # noqa: E402

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_LOCK_CANDIDATES = ("uv.lock", "requirements.lock", "poetry.lock", "pdm.lock")
_CHUNK = 8 * 1024 * 1024


def is_valid_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX.match(value))


def enumerate_stage_payload(stage_root: Path) -> list[tuple[str, int]]:
    """Return ``(relative_path, size)`` using ``stat`` only — never reads payload bytes."""
    out: list[tuple[str, int]] = []
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage_root).as_posix()
        if rel.rsplit("/", 1)[-1] in _CONTROL_BASENAMES:
            continue
        out.append((rel, path.stat().st_size))
    return out


class LocalStreamingS3:
    """Disk-backed S3 stand-in for preflight: stream/hash one file at a time."""

    def __init__(self, root: Path, *, bucket: str = "local") -> None:
        self._root = root.resolve()
        self._bucket = bucket
        self.peak_cached_bytes = 0

    def _local_path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not str(path).startswith(str(self._root)):
            raise NotFound(f"s3://{self._bucket}/{key}")
        if not path.is_file():
            raise NotFound(f"s3://{self._bucket}/{key}")
        return path

    def get(self, bucket: str, key: str) -> bytes:
        path = self._local_path(key)
        size = path.stat().st_size
        self.peak_cached_bytes = max(self.peak_cached_bytes, size)
        with path.open("rb") as handle:
            return handle.read()

    def get_range(self, bucket: str, key: str, start: int, length: int) -> bytes:
        path = self._local_path(key)
        if length <= 0:
            return b""
        self.peak_cached_bytes = max(self.peak_cached_bytes, length)
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read(length)

    def head(self, bucket: str, key: str) -> dict[str, Any]:
        path = self._local_path(key)
        return {
            "size": path.stat().st_size,
            "crc64nvme": None,
            "etag": None,
            "content_type": None,
        }

    def hash_object(self, bucket: str, key: str) -> tuple[str, int]:
        path = self._local_path(key)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def list(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
        prefix = prefix.strip("/")
        base = self._root / prefix if prefix else self._root
        if not base.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self._root).as_posix()
            if rel.rsplit("/", 1)[-1] in _CONTROL_BASENAMES:
                continue
            out.append({"key": rel, "size": path.stat().st_size})
        return out

    def put(self, bucket: str, key: str, body: bytes, *, content_type: str | None = None) -> None:
        raise NotImplementedError("LocalStreamingS3 is read-only")

    def put_file(self, bucket: str, key: str, local_path: str) -> None:
        raise NotImplementedError("LocalStreamingS3 is read-only")

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
        raise NotImplementedError("LocalStreamingS3 is read-only")

    def delete(self, bucket: str, key: str) -> None:
        raise NotImplementedError("LocalStreamingS3 is read-only")


def find_packages_lock(edullm_root: Path) -> Path | None:
    for name in _LOCK_CANDIDATES:
        path = edullm_root / name
        if path.is_file():
            return path
    return None


def hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_publisher_code_sha256(edullm_root: Path) -> str:
    """Deterministic SHA-256 over ``src/edullm_data`` and ``families`` file bytes."""
    digest = hashlib.sha256()
    for relative_root in ("src/edullm_data", "families"):
        base = edullm_root / relative_root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(edullm_root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_CHUNK), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def provenance_report(edullm_root: Path) -> dict[str, Any]:
    lock_path = find_packages_lock(edullm_root)
    code_sha256 = compute_publisher_code_sha256(edullm_root)
    report: dict[str, Any] = {
        "edullm_data_root": str(edullm_root.resolve()),
        "publisher_code_sha256": code_sha256,
        "packages_lock_path": str(lock_path) if lock_path else None,
        "packages_lock_sha256": hash_file_sha256(lock_path) if lock_path else None,
        "local_execute_allowed": lock_path is not None,
        "compute_commands": {
            "publisher_code_sha256": (
                "python3 scripts/publish_eval_v1.py --print-provenance | jq -r .publisher_code_sha256"
            ),
            "packages_lock_sha256": (
                f"sha256sum {lock_path}" if lock_path else "blocked: no lockfile in pinned edullm-data"
            ),
        },
        "notes": (
            "Set EDULLM_CODE_SHA256 and EDULLM_PACKAGES_LOCK_SHA256 to the values above before "
            "--execute, or run from AWS Batch (AWS_BATCH_JOB_ID set) with the deployed validator wheel. "
            "Git commit SHAs and placeholders like 'local' are rejected."
        ),
    }
    if lock_path is None:
        report["local_execute_blocker"] = (
            "Pinned edullm-data has no uv.lock/requirements.lock/poetry.lock/pdm.lock. "
            "Prefer AWS Batch publication with the shipped wheel once p3-evaluator-corpus/v1 "
            "is deployed; do not fabricate EDULLM_PACKAGES_LOCK_SHA256."
        )
    return report


def run_streaming_preflight(
    stage_root: Path,
    pins: dict[str, Any],
    *,
    pretrain_manifest_sha256: str = "a" * 64,
    family: Mapping[str, Any] | None = None,
    purpose: str = "",
    about: str | None = None,
    notes: str | None = None,
    limitations: list[dict[str, Any]] | None = None,
    license: dict[str, Any] | None = None,
    owner: str = "edullm-data@alphaaiengineering.com",
    dataset_id: str = "eval/formal-proof-premises-500m",
    profile: str = "p3-evaluator-corpus/v1",
    group: str = "evaluator",
    expected_version: str = "v1",
) -> dict[str, Any]:
    """Hash staged payload by streaming from disk; never loads the full tree into RAM."""
    from edullm_data.publish import build_plan  # noqa: WPS433

    files = enumerate_stage_payload(stage_root)
    if not files:
        raise ValueError(f"no payload files under {stage_root}")
    s3 = LocalStreamingS3(stage_root)
    group_meta = {
        group: {
            "evaluator_root_sha256": pins["evaluator_root_sha256"],
            "source_seal_root_sha256": pins["source_seal_root_sha256"],
            "coverage": "incomplete",
            "depends_on": [
                {
                    "role": "pretrain",
                    "dataset_id": "pretrain/formal-proof-premises-500m",
                    "version": "v3",
                    "manifest_sha256": pretrain_manifest_sha256,
                }
            ],
        }
    }
    plan = build_plan(
        files,
        dataset_id=dataset_id,
        version=expected_version,
        purpose=purpose or "preflight",
        profile=profile,
        family=family or {},
        created_at="2026-08-06T18:00:00Z",
        build_executor={"kind": "external", "host_class": "local-preflight"},
        source_kind="local",
        s3=s3,
        source_bucket="local",
        source_prefix="",
        owner=owner,
        group_meta=group_meta,
        sources=[],
        about=about,
        notes=notes,
        limitations=limitations or [],
        license=license,
    )
    prefix = f"edullm-landing/{dataset_id}/{expected_version}"
    planned = sorted(
        {
            f"{prefix}/dataset.json",
            *[f"edullm-landing/{key}" for key in plan.payload_keys],
            *[
                f"{prefix}/{group['manifest']}"
                for group in plan.dataset_json.get("groups", [])
                if group.get("manifest")
            ],
        }
    )
    return {
        "planned_objects": planned,
        "inventory": plan.dataset_json["inventory"],
        "preflight_peak_read_window_bytes": s3.peak_cached_bytes,
    }


def resolve_execute_provenance(
    env: Mapping[str, str], *, edullm_root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    if env.get("AWS_BATCH_JOB_ID"):
        return _build_executor_from_env(env), []

    report = provenance_report(edullm_root)
    errors: list[str] = []
    code_env = env.get("EDULLM_CODE_SHA256")
    lock_env = env.get("EDULLM_PACKAGES_LOCK_SHA256")
    expected_code = report["publisher_code_sha256"]

    if not is_valid_sha256_hex(code_env):
        errors.append(
            "EDULLM_CODE_SHA256 must be a 64-character lowercase hex SHA-256 digest "
            "(not a git commit SHA or placeholder)."
        )
    elif code_env != expected_code:
        errors.append(
            "EDULLM_CODE_SHA256 does not match the pinned publisher source tree. "
            f"expected={expected_code} got={code_env}"
        )

    lock_path = find_packages_lock(edullm_root)
    if lock_path is None:
        errors.append(str(report["local_execute_blocker"]))
    elif not is_valid_sha256_hex(lock_env):
        errors.append(
            "EDULLM_PACKAGES_LOCK_SHA256 must be a 64-character lowercase hex SHA-256 digest."
        )
    else:
        expected_lock = hash_file_sha256(lock_path)
        if lock_env != expected_lock:
            errors.append(
                "EDULLM_PACKAGES_LOCK_SHA256 does not match the lockfile bytes. "
                f"expected={expected_lock} got={lock_env}"
            )

    if errors:
        return None, errors
    executor = _build_executor_from_env(env)
    if executor.get("kind") != "external":
        errors.append("unexpected build executor kind for external provenance")
        return None, errors
    if not executor.get("code_sha256") or not executor.get("packages_lock_sha256"):
        errors.append("build executor missing code_sha256 or packages_lock_sha256")
        return None, errors
    return executor, []
