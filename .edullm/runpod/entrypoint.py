#!/usr/bin/env python3
"""Run the unchanged MixLaw entrypoint against a fully staged local corpus."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

import mixlaw_entrypoint as mixlaw  # noqa: E402


def requested_arm(argv: list[str]) -> int:
    try:
        return int(argv[argv.index("--arm-index") + 1])
    except (ValueError, IndexError):
        raise mixlaw.MixLawConfigError("--arm-index is required") from None


def staged_sources() -> tuple[mixlaw.DomainSource, ...]:
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/mixlaw/ready.json",
        )
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "family": "mixlaw",
        "dataset_id": mixlaw.DATASET_ID,
        "dataset_version": mixlaw.DATASET_VERSION,
    }
    changed = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if changed:
        raise mixlaw.MixLawConfigError(f"invalid RunPod input manifest fields: {changed}")
    arm_index = requested_arm(sys.argv[1:])
    if payload.get("arm_index") != arm_index:
        raise mixlaw.MixLawConfigError(
            f"staged arm is {payload.get('arm_index')!r}, requested arm is {arm_index}"
        )
    sources: list[mixlaw.DomainSource] = []
    for domain in payload.get("domains", []):
        paths: list[str] = []
        for record in domain.get("objects", []):
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != int(record["size"]):
                raise mixlaw.MixLawConfigError(f"staged object is missing or changed: {path}")
            paths.append(str(path))
        sources.append(
            mixlaw.DomainSource(
                name=str(domain["name"]),
                paths=tuple(paths),
                available_tokens=int(domain["available_tokens"]),
            )
        )
    if tuple(source.name for source in sources) != mixlaw.DOMAINS:
        raise mixlaw.MixLawConfigError("RunPod manifest domain order differs from the recipe")
    return tuple(sources)


def worker_argv(argv: list[str]) -> list[str]:
    return argv if "--train-worker" in argv else [*argv, "--train-worker"]


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
    return mixlaw.main(worker_argv(sys.argv[1:]), resolver=staged_sources)


if __name__ == "__main__":
    raise SystemExit(main())
