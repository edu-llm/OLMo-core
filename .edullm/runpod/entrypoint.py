#!/usr/bin/env python3
"""Run unchanged curriculum training from a pre-staged RunPod manifest."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

import curriculum_entrypoint as curriculum  # noqa: E402
from curriculum_data import ResolvedInput  # noqa: E402


def resolved_input(identity: dict) -> ResolvedInput:
    return ResolvedInput(
        dataset_id=str(identity["dataset_id"]),
        version=str(identity["version"]),
        group=str(identity["group"]),
        profile=str(identity["profile"]),
        manifest_sha256=str(identity["manifest_sha256"]),
        paths=tuple(identity["paths"]),
        numpy_dtype=str(identity["numpy_dtype"]),
        header_bytes=int(identity["header_bytes"]),
    )


def checked_paths(records: list[dict]) -> tuple[Path, ...]:
    paths = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise curriculum.PublishedInputError(f"staged object is missing or changed: {path}")
        paths.append(path)
    return tuple(paths)


def resolve_local_inputs(*, arm, order_version, cache_dir):
    del order_version, cache_dir
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/curriculum/ready.json",
        )
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("family") != "curriculum"
        or payload.get("arm_index") != arm.index
        or payload.get("arm_name") != arm.name
    ):
        raise curriculum.PublishedInputError(
            f"RunPod manifest does not describe curriculum arm {arm.index}:{arm.name}"
        )
    parent = resolved_input(payload["parent"])
    order = resolved_input(payload["order"]) if payload.get("order") else None
    if order is None:
        raise curriculum.PublishedInputError(
            f"RunPod manifest is missing the order for kept arm {arm.name}"
        )
    return (
        parent,
        checked_paths(payload["parent_objects"]),
        order,
        checked_paths(payload.get("order_objects", [])),
    )


def main() -> int:
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ):
        if os.environ.get(key):
            raise SystemExit(f"refusing to train while {key} is present")
    curriculum.resolve_and_stage = resolve_local_inputs
    return curriculum.main()


if __name__ == "__main__":
    raise SystemExit(main())
