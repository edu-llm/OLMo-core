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

LEGACY_DATASET_ID = "pretrain/regmix-10b"
PARENT_DATASET_ID = "pretrain/opt-with-synthetic-10b"
ORDER_DATASET_ID = "curriculum/opt-with-synthetic-10b"
DATASET_VERSION = "v1"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
PARENT_GROUP = "tokens"
ORDER_GROUP = "mtld"
PARENT_MANIFEST_SHA256 = "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797"
ORDER_MANIFEST_SHA256 = "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d"
NUMPY_DTYPE_CHARS = {
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
}
ARM_NAMES = {
    "full_acronym_soup",
    "no_centaur",
    "no_proxy",
    "curriculum_quadratic_mtld",
    "curriculum_quadratic_mtld_no_centaur",
}
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
    if payload.get("schema_version") == 2:
        return _load_multi_release_manifest(payload)

    dataset = payload.get("dataset", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("family") != "hpo-probe"
        or dataset.get("dataset_id") != LEGACY_DATASET_ID
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
    payload["_local_releases"] = {
        (LEGACY_DATASET_ID, DATASET_VERSION, None): {
            **dataset,
            "group": None,
            "_local_train_paths": train_paths,
            "_local_val_paths": val_paths,
        }
    }
    return payload


def _load_multi_release_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("family") != "hpo-probe" or not isinstance(payload.get("releases"), list):
        raise RuntimeError("RunPod manifest does not describe staged HPO releases")

    local_releases: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for release in payload["releases"]:
        if not isinstance(release, dict) or int(release.get("header_bytes", -1)) != 0:
            raise RuntimeError("RunPod manifest contains an invalid release")
        identity = (
            str(release.get("dataset_id")),
            str(release.get("version")),
            release.get("group"),
        )
        if identity in local_releases:
            raise RuntimeError(f"RunPod manifest repeats release {identity!r}")
        train_paths = checked_paths(release.get("train_objects", []))
        val_paths = checked_paths(release.get("val_objects", []))
        if not train_paths:
            raise RuntimeError(f"RunPod release {identity!r} has no train objects")
        if set(train_paths) & set(val_paths):
            raise RuntimeError(f"RunPod release {identity!r} overlaps train and held-out paths")
        local_releases[identity] = {
            **release,
            "_local_train_paths": train_paths,
            "_local_val_paths": val_paths,
        }

    parent_key = (PARENT_DATASET_ID, DATASET_VERSION, PARENT_GROUP)
    order_key = (ORDER_DATASET_ID, DATASET_VERSION, ORDER_GROUP)
    legacy_key = (LEGACY_DATASET_ID, DATASET_VERSION, None)
    has_legacy = legacy_key in local_releases
    has_parent = parent_key in local_releases
    has_order = order_key in local_releases
    if has_parent != has_order:
        missing = order_key if has_parent else parent_key
        raise RuntimeError(f"RunPod curriculum manifest is missing paired release {missing!r}")
    if not has_legacy and not has_parent:
        raise RuntimeError("RunPod manifest contains no complete HPO release set")

    if has_parent:
        parent = local_releases[parent_key]
        order = local_releases[order_key]
        if not parent["_local_val_paths"]:
            raise RuntimeError("RunPod curriculum parent must contain a held-out split")
        expected_parent = {
            "profile": "pretrain-tokens/v1",
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "tokenizer_id": TOKENIZER_ID,
        }
        expected_order = {
            "profile": "token-order/v1",
            "manifest_sha256": ORDER_MANIFEST_SHA256,
        }
        if any(parent.get(key) != value for key, value in expected_parent.items()):
            raise RuntimeError("RunPod manifest has the wrong immutable curriculum parent")
        if any(order.get(key) != value for key, value in expected_order.items()):
            raise RuntimeError("RunPod manifest has the wrong immutable MTLD order")
        if order["_local_val_paths"]:
            raise RuntimeError("RunPod MTLD order must not contain a held-out split")

    payload["_local_releases"] = local_releases
    return payload


def local_dataset_paths(
    manifest: dict[str, Any],
    dataset_id: str,
    version: str,
    *,
    split: str | None = None,
    group: str | None = None,
    include_held_out: bool = False,
    **_: Any,
) -> SimpleNamespace:
    """Resolve one registry request entirely from the staged multi-release manifest."""

    matching = [
        (identity, release)
        for identity, release in manifest["_local_releases"].items()
        if identity[:2] == (dataset_id, version) and (group is None or identity[2] == group)
    ]
    if len(matching) != 1:
        available = sorted(
            repr(identity)
            for identity in manifest["_local_releases"]
            if identity[:2] == (dataset_id, version)
        )
        raise RuntimeError(
            f"RunPod did not stage one unambiguous {dataset_id}/{version} group={group!r}; "
            f"available={available}"
        )
    identity, release = matching[0]
    train_paths = list(release["_local_train_paths"])
    val_paths = list(release["_local_val_paths"])
    if split == "train":
        paths = train_paths
    elif split in {"val", "validation", "heldout", "test"}:
        paths = val_paths
    elif split is None:
        paths = [*train_paths, *val_paths] if include_held_out else train_paths
    else:
        raise RuntimeError(f"RunPod local reader does not support split {split!r}")
    dtype = str(release["dtype"])
    byte_order = release.get("byte_order")
    byte_prefix = {"little": "<", "big": ">"}.get(byte_order)
    numpy_dtype = f"{byte_prefix}{NUMPY_DTYPE_CHARS.get(dtype, dtype)}" if byte_prefix else dtype
    return SimpleNamespace(
        dataset_id=dataset_id,
        version=version,
        group=identity[2],
        profile=release.get("profile"),
        manifest_sha256=release.get("manifest_sha256"),
        tokenizer_id=release.get("tokenizer_id"),
        split=split or "*",
        paths=paths,
        train=train_paths,
        val=val_paths or None,
        splits={"train": train_paths, **({"val": val_paths} if val_paths else {})},
        split_rows={},
        rows=None,
        kwargs={},
        dtype=dtype,
        numpy_dtype=numpy_dtype,
        byte_order=byte_order,
        header_bytes=int(release["header_bytes"]),
    )


def validate_manifest_for_mode(manifest: dict[str, Any], mode: str) -> None:
    """Refuse a launch before W&B initialization when its staged release is absent."""

    curriculum_modes = {
        "curriculum_quadratic_mtld",
        "curriculum_quadratic_mtld_no_centaur",
    }
    legacy_modes = {
        "proxy-cohort",
        "full_acronym_soup",
        "no_centaur",
        "no_proxy",
    }
    if mode not in curriculum_modes | legacy_modes:
        raise RuntimeError(f"unsupported RunPod HPO mode {mode!r}")

    releases = manifest.get("_local_releases")
    if releases is None:
        if mode in curriculum_modes:
            raise RuntimeError("selected curriculum mode requires the parent and MTLD releases")
        return

    if mode in curriculum_modes:
        required = {
            (PARENT_DATASET_ID, DATASET_VERSION, PARENT_GROUP),
            (ORDER_DATASET_ID, DATASET_VERSION, ORDER_GROUP),
        }
        if not required.issubset(releases):
            raise RuntimeError("selected curriculum mode requires the parent and MTLD releases")
    elif (LEGACY_DATASET_ID, DATASET_VERSION, None) not in releases:
        raise RuntimeError("selected historical HPO mode requires the RegMix release")


def install_local_dataset_reader(manifest: dict[str, Any]) -> None:
    """Replace registry access with a strict reader over the staged manifest."""

    import edullm_data.read as read_module
    import edullm_data.s3 as s3_module

    def dataset_paths(dataset_id: str, version: str, **kwargs):
        return local_dataset_paths(manifest, dataset_id, version, **kwargs)

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
        raise RuntimeError(f"unsupported HPO spec: {arm!r}")

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
            result[replace_index] = str(destination) if value == flag else f"{flag}={destination}"
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

        return tuple(root for root in original(spec, checkpoint_root) if is_url(root))

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
    mode = os.environ.get("MODE")
    if mode:
        validate_manifest_for_mode(manifest, mode)
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
