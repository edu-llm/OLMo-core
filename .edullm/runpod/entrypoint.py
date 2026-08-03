#!/usr/bin/env python3
"""Run unchanged Skill-It training with locally staged RunPod inputs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

import skillit_loader  # noqa: E402
import train_skillit_370m as training  # noqa: E402
from skillit_math import DATASET_ID, DATASET_VERSION, DOMAINS  # noqa: E402


def resolve_local_datasets(work_dir: str | Path):
    from olmo_core.data import NumpyDatasetDType, NumpyFSLDatasetConfig, TokenizerConfig

    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/skillit/ready.json",
        )
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "family": "skillit",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise skillit_loader.SkillItDataError("invalid Skill-It RunPod input manifest")
    domain_records = payload.get("domains", [])
    if tuple(record.get("name") for record in domain_records) != DOMAINS:
        raise skillit_loader.SkillItDataError("RunPod manifest domain order differs from recipe")
    tokenizer = TokenizerConfig.dolma2()
    datasets = []
    for domain in domain_records:
        paths = []
        for record in domain["objects"]:
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != int(record["size"]):
                raise skillit_loader.SkillItDataError(f"staged object is missing or changed: {path}")
            paths.append(str(path))
        config = NumpyFSLDatasetConfig(
            paths=paths,
            tokenizer=tokenizer,
            sequence_length=skillit_loader.SEQUENCE_LENGTH,
            dtype=NumpyDatasetDType.uint32,
            work_dir=str(Path(work_dir) / str(domain["name"])),
            include_instance_metadata=False,
        )
        dataset = config.build()
        dataset.prepare()
        if len(dataset) <= 0:
            raise skillit_loader.SkillItDataError(f"{domain['name']}: no full sequences")
        datasets.append(dataset)
    return tuple(datasets)


def main() -> None:
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
    training.resolve_domain_datasets = resolve_local_datasets
    training.main()


if __name__ == "__main__":
    main()
