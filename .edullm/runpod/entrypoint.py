#!/usr/bin/env python3
"""Run unchanged token-selection training from RunPod-local inputs."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

import token_selection_entrypoint as training  # noqa: E402
from token_selection_370m.arms import REFHQ, get_arm  # noqa: E402
from train_on_corpus import Corpus  # noqa: E402

MANIFEST_PATH = Path(
    os.environ.get(
        "EDULLM_RUNPOD_INPUT_MANIFEST",
        "/workspace/edullm-inputs/token-selection/ready.json",
    )
)


def manifest() -> dict:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("family") != "token-selection":
        raise RuntimeError("invalid token-selection RunPod input manifest")
    return payload


def corpus_record(dataset_id: str) -> dict:
    try:
        return manifest()["corpora"][dataset_id]
    except KeyError:
        raise RuntimeError(f"RunPod manifest has no staged corpus for {dataset_id}") from None


def resolve_local_corpus(*, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig

    if tokenizer_id != "tokenizer/dolma2-bpe":
        raise RuntimeError(f"unsupported RunPod tokenizer: {tokenizer_id}")
    record = corpus_record(dataset_id)
    if version not in ("", "latest", record["version"]):
        raise RuntimeError(
            f"requested {dataset_id}/{version}, staged version is {record['version']}"
        )
    paths = []
    for obj in record["objects"]:
        path = Path(obj["path"])
        if not path.is_file() or path.stat().st_size != int(obj["size"]):
            raise RuntimeError(f"staged object is missing or changed: {path}")
        paths.append(str(path))
    return Corpus(
        dataset_id=dataset_id,
        version=str(record["version"]),
        paths=paths,
        dtype=NumpyDatasetDType(record["dtype"]),
        tokenizer=TokenizerConfig.dolma2(),
        rows=int(record["rows"]) if record.get("rows") is not None else None,
    )


ORIGINAL_BINDING = training.immutable_corpus_binding


def logical_corpus_binding(dataset_id: str, corpus: Corpus) -> dict:
    record = corpus_record(dataset_id)
    logical = replace(corpus, paths=list(record["logical_paths"]))
    return ORIGINAL_BINDING(dataset_id, logical)


def configure_references(payload: dict) -> None:
    names = {
        "reference": "EDULLM_REFERENCE_PATH",
        "late": "EDULLM_LATE_REFERENCE_PATH",
    }
    for name, path_value in payload.get("references", {}).items():
        path = Path(path_value)
        if name not in names or not path.is_file():
            raise RuntimeError(f"invalid staged {name!r} reference: {path}")
        os.environ[names[name]] = str(path)


def requested_arm(argv: list[str]) -> str:
    try:
        return argv[argv.index("--arm") + 1]
    except (ValueError, IndexError):
        raise RuntimeError("--arm is required") from None


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
    payload = manifest()
    arm = requested_arm(sys.argv[1:])
    if payload.get("arm") != arm:
        raise SystemExit(f"staged arm is {payload.get('arm')!r}, requested {arm!r}")
    arm_spec = get_arm(arm)
    os.environ["EDULLM_DATASET_VERSION"] = str(
        payload["corpora"][arm_spec.dataset_id]["version"]
    )
    if arm_spec.requires_refhq_stream:
        os.environ["EDULLM_REFHQ_DATASET_VERSION"] = str(
            payload["corpora"][REFHQ]["version"]
        )
    configure_references(payload)
    training.resolve_corpus = resolve_local_corpus
    training.immutable_corpus_binding = logical_corpus_binding
    training.main()


if __name__ == "__main__":
    main()
