#!/usr/bin/env python3
"""Run HPO from staged local data and persistent RunPod storage."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

DATASET_ID = "pretrain/regmix-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
ARM_NAMES = {"full_acronym_soup", "no_centaur", "no_proxy"}
AWS_KEYS = (
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
)


def checked_paths(records: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """Validate staged object sizes before giving local paths to OLMo."""

    paths: list[str] = []
    for record in records:
        path = Path(str(record["path"]))
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise RuntimeError(f"staged object is missing or changed: {path}")
        paths.append(str(path))
    return tuple(paths)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the sealed local-data manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("family") != "hpo-probe"
        or dataset.get("dataset_id") != DATASET_ID
        or dataset.get("version") != DATASET_VERSION
        or dataset.get("tokenizer_id") != TOKENIZER_ID
        or int(dataset.get("header_bytes", -1)) != 0
    ):
        raise RuntimeError("RunPod manifest does not describe the sealed HPO corpus")
    train_paths = checked_paths(payload.get("train_objects", []))
    val_paths = checked_paths(payload.get("val_objects", []))
    if not train_paths or not val_paths:
        raise RuntimeError("RunPod manifest must contain train and held-out objects")
    if set(train_paths) & set(val_paths):
        raise RuntimeError("RunPod manifest train and held-out paths overlap")
    payload["_local_train_paths"] = train_paths
    payload["_local_val_paths"] = val_paths
    return payload


def install_local_dataset_reader(manifest: dict[str, Any]) -> None:
    """Replace registry access with a strict reader over the staged manifest."""

    import edullm_data.read as read_module
    import edullm_data.s3 as s3_module

    dataset = manifest["dataset"]
    train_paths = manifest["_local_train_paths"]
    val_paths = manifest["_local_val_paths"]

    def dataset_paths(dataset_id: str, version: str, *, s3):
        del s3
        if dataset_id != DATASET_ID or version != DATASET_VERSION:
            raise RuntimeError(
                f"RunPod staged {DATASET_ID}/{DATASET_VERSION}, not {dataset_id}/{version}"
            )
        return SimpleNamespace(
            paths=list(train_paths),
            train=list(train_paths),
            val=list(val_paths),
            dtype=dataset["dtype"],
            byte_order=dataset.get("byte_order"),
            header_bytes=int(dataset["header_bytes"]),
        )

    read_module.dataset_paths = dataset_paths
    s3_module.Boto3S3.default = classmethod(lambda cls: object())


def materialize_runtime_spec(
    source: Path,
    destination: Path,
    *,
    job_root: Path,
    shared_root: Path,
) -> Path:
    """Rewrite platform storage paths for persistent local RunPod storage."""

    payload = json.loads(source.read_text(encoding="utf-8"))
    arm = payload.get("arm")
    if arm not in ARM_NAMES:
        raise RuntimeError(f"unsupported three-arm HPO spec: {arm!r}")

    checkpoint_root = job_root / "checkpoints"
    payload["controller"]["checkpoint_root"] = str(checkpoint_root)
    payload["controller_state_path"] = str(job_root / "controller-state.jsonl")
    payload["study_result_path"] = str(job_root / "study-result.json")
    payload["segment_spec_dir"] = str(job_root / "segment-specs")
    payload.pop("controller_snapshot_root", None)
    if "proxy_evidence_path" in payload:
        payload["proxy_evidence_path"] = str(shared_root / "proxy-evidence.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def rewrite_spec_args(
    argv: Sequence[str],
    *,
    job_root: Path,
    shared_root: Path,
) -> list[str]:
    """Materialize every controller/cohort spec named on the command line."""

    result = list(argv)
    runtime_dir = job_root / "runtime-specs"
    for flag in ("--controller-spec", "--proxy-spec", "--reference-spec"):
        for index, value in enumerate(tuple(result)):
            source: Path | None = None
            if value == flag and index + 1 < len(result):
                source = Path(result[index + 1]).resolve()
                replace_index = index + 1
            elif value.startswith(flag + "="):
                source = Path(value.split("=", 1)[1]).resolve()
                replace_index = index
            else:
                continue
            destination = runtime_dir / source.name
            materialize_runtime_spec(
                source,
                destination,
                job_root=job_root,
                shared_root=shared_root,
            )
            result[replace_index] = (
                str(destination) if value == flag else f"{flag}={destination}"
            )
    return result


def _refuse_aws_credentials() -> None:
    present = [key for key in AWS_KEYS if os.environ.get(key)]
    if present:
        raise RuntimeError(f"refusing to train while AWS credentials are present: {present}")


def _patch_worker_entrypoint(hpo_module) -> None:
    original = hpo_module.build_worker_argv

    def build_worker_argv(**kwargs):
        argv = original(**kwargs)
        argv[1] = str(Path(__file__).resolve())
        return argv

    hpo_module.build_worker_argv = build_worker_argv


def _patch_probe_durability(hpo_module) -> None:
    """Treat local RunPod files as W&B-mirrorable; only URLs are externally durable."""

    original = hpo_module._probe_durable_roots

    def probe_durable_roots(spec, checkpoint_root):
        from olmo_core.io import is_url

        return tuple(
            root for root in original(spec, checkpoint_root) if is_url(root)
        )

    hpo_module._probe_durable_roots = probe_durable_roots


def main(argv: Sequence[str] | None = None) -> int:
    _refuse_aws_credentials()
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/hpo-probe/ready.json",
        )
    )
    manifest = load_manifest(manifest_path)
    install_local_dataset_reader(manifest)

    import hpo_on_corpus as hpo

    _patch_worker_entrypoint(hpo)
    _patch_probe_durability(hpo)
    args = list(sys.argv[1:] if argv is None else argv)
    if "--run-segment" not in args:
        job_root = Path(os.environ["EDULLM_RUNPOD_JOB_ROOT"]).resolve()
        shared_root = Path(os.environ["EDULLM_RUNPOD_SHARED_ROOT"]).resolve()
        args = rewrite_spec_args(args, job_root=job_root, shared_root=shared_root)
    return hpo.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
