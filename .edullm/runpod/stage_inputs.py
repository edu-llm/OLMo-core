#!/usr/bin/env python3
"""Resolve and stage the sealed HPO corpus before RunPod training starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

AWS_CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
AWS_PROVIDER_KEYS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)
AWS_FILE_KEYS = set(AWS_CREDENTIAL_KEYS) | {
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_CREDENTIAL_EXPIRATION",
}
EXPORT = re.compile(r"^export ([A-Z0-9_]+)=(.*)$")

TOKENIZER_ID = "tokenizer/dolma2-bpe"
LEGACY_DATASET_ID = "pretrain/regmix-10b"
PARENT_DATASET_ID = "pretrain/opt-with-synthetic-10b"
ORDER_DATASET_ID = "curriculum/opt-with-synthetic-10b"
DATASET_VERSION = "v1"
PARENT_GROUP = "tokens"
ORDER_GROUP = "mtld"
PARENT_MANIFEST_SHA256 = "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797"
ORDER_MANIFEST_SHA256 = "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d"


def load_credentials(path: Path) -> None:
    """Load one short-lived, mode-0600 AWS session without accepting ambient credentials."""

    if any(os.environ.get(key) for key in (*AWS_CREDENTIAL_KEYS, *AWS_PROVIDER_KEYS)):
        raise RuntimeError("refusing ambient AWS credentials; use --credentials-file")
    if not path.is_file():
        raise RuntimeError(f"credential file does not exist: {path}")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError("credential file must not be readable by group or other users")

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "unset AWS_PROFILE":
            os.environ.pop("AWS_PROFILE", None)
            continue
        match = EXPORT.fullmatch(line)
        if not match or match.group(1) not in AWS_FILE_KEYS:
            raise RuntimeError(f"unsupported credential-file line: {line[:40]!r}")
        parsed = shlex.split(match.group(2), posix=True)
        if len(parsed) != 1:
            raise RuntimeError(f"invalid value for {match.group(1)}")
        values[match.group(1)] = parsed[0]
    missing = [key for key in AWS_CREDENTIAL_KEYS if not values.get(key)]
    if missing:
        raise RuntimeError(f"temporary credential file is missing {missing}")
    os.environ.update(values)


def destroy_credentials(path: Path) -> None:
    """Remove the short-lived session from the process and disk."""

    for key in AWS_FILE_KEYS | set(AWS_PROVIDER_KEYS):
        os.environ.pop(key, None)
    path.unlink(missing_ok=True)


def stage_one(client, transfer, root: Path, uri: str) -> dict[str, object]:
    """Download one immutable registry object into the local RunPod data tree."""

    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != "edullm-data":
        raise RuntimeError(f"HPO input escaped s3://edullm-data/: {uri}")
    key = parsed.path.lstrip("/")
    destination = root / "objects" / parsed.netloc / key
    metadata = client.head_object(Bucket=parsed.netloc, Key=key)
    size = int(metadata["ContentLength"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != size:
        partial = destination.with_name(destination.name + ".partial")
        partial.unlink(missing_ok=True)
        client.download_file(parsed.netloc, key, str(partial), Config=transfer)
        if partial.stat().st_size != size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"short S3 download for {uri}")
        partial.replace(destination)
    return {
        "uri": uri,
        "path": str(destination),
        "size": size,
        "etag": str(metadata.get("ETag", "")).strip('"'),
        "version_id": metadata.get("VersionId"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("/workspace/aws-session.env"),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/workspace/edullm-inputs/hpo-probe"),
    )
    parser.add_argument(
        "--release-set",
        choices=("all", "legacy", "curriculum"),
        default="all",
        help="Stage all releases, only RegMix, or only the curriculum parent/order pair.",
    )
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def credential_path_from_argv() -> Path:
    result = Path("/workspace/aws-session.env")
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value == "--credentials-file" and index + 1 < len(values):
            result = Path(values[index + 1])
        elif value.startswith("--credentials-file="):
            result = Path(value.split("=", 1)[1])
    return result


def _validate_read(
    read,
    *,
    label: str,
    require_val: bool,
) -> tuple[list[str], list[str]]:
    train_uris = [str(path) for path in read.paths]
    val_uris = [str(path) for path in (read.val or ())]
    if not train_uris:
        raise RuntimeError(f"{label} registry returned no train split")
    if require_val and not val_uris:
        raise RuntimeError(f"{label} registry returned no held-out split")
    if set(train_uris) & set(val_uris):
        raise RuntimeError(f"{label} train and held-out splits overlap")
    if int(read.header_bytes) != 0:
        raise RuntimeError(f"{label} declares a nonzero header")
    if read.byte_order is not None and str(read.byte_order) != sys.byteorder:
        raise RuntimeError(
            f"{label} byte order {read.byte_order!r} does not match host {sys.byteorder!r}"
        )
    if read.dtype is None:
        raise RuntimeError(f"{label} declares no fixed-width dtype")
    return train_uris, val_uris


def resolve_release_inputs(
    dataset_paths,
    s3,
    *,
    release_set: str = "all",
) -> list[dict[str, object]]:
    """Resolve every immutable release needed by the existing and curriculum HPO modes."""

    requests = [
        {
            "dataset_id": LEGACY_DATASET_ID,
            "release_set": "legacy",
            "version": DATASET_VERSION,
            "group": None,
            "profile": None,
            "manifest_sha256": None,
            "tokenizer_id": TOKENIZER_ID,
            "require_val": True,
        },
        {
            "dataset_id": PARENT_DATASET_ID,
            "release_set": "curriculum",
            "version": DATASET_VERSION,
            "group": PARENT_GROUP,
            "profile": "pretrain-tokens/v1",
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "tokenizer_id": TOKENIZER_ID,
            "require_val": True,
        },
        {
            "dataset_id": ORDER_DATASET_ID,
            "release_set": "curriculum",
            "version": DATASET_VERSION,
            "group": ORDER_GROUP,
            "profile": "token-order/v1",
            "manifest_sha256": ORDER_MANIFEST_SHA256,
            "tokenizer_id": None,
            "require_val": False,
        },
    ]
    if release_set not in {"all", "legacy", "curriculum"}:
        raise ValueError(f"unsupported release set {release_set!r}")
    selected = [
        request
        for request in requests
        if release_set == "all" or request["release_set"] == release_set
    ]
    releases: list[dict[str, object]] = []
    for request in selected:
        read = dataset_paths(
            str(request["dataset_id"]),
            str(request["version"]),
            s3=s3,
            group=request["group"],
        )
        expected_identity = {
            key: request[key]
            for key in ("profile", "manifest_sha256", "tokenizer_id")
            if request[key] is not None
        }
        actual_identity = {
            key: value
            for key in expected_identity
            if (value := getattr(read, key, None)) is not None
        }
        if any(actual_identity[key] != expected_identity[key] for key in actual_identity):
            raise RuntimeError(
                f"{request['dataset_id']}/{request['version']} registry returned the wrong "
                f"immutable release: expected {expected_identity}, got {actual_identity}"
            )
        train_uris, val_uris = _validate_read(
            read,
            label=f"{request['dataset_id']}/{request['version']}",
            require_val=bool(request["require_val"]),
        )
        releases.append(
            {
                "dataset_id": request["dataset_id"],
                "version": request["version"],
                "group": request["group"],
                "profile": request["profile"],
                "manifest_sha256": request["manifest_sha256"],
                "tokenizer_id": request["tokenizer_id"],
                "dtype": str(getattr(read.dtype, "value", read.dtype)),
                "byte_order": read.byte_order,
                "header_bytes": int(read.header_bytes),
                "train_uris": train_uris,
                "val_uris": val_uris,
            }
        )
    return releases


def main() -> None:
    credential_path = credential_path_from_argv()
    try:
        args = parse_args()
        credential_path = args.credentials_file
        if args.workers <= 0:
            raise SystemExit("--workers must be positive")
        load_credentials(args.credentials_file)

        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        from edullm_data.read import dataset_paths
        from edullm_data.s3 import Boto3S3

        releases = resolve_release_inputs(
            dataset_paths,
            Boto3S3.default(),
            release_set=args.release_set,
        )
        ordered_uris = list(
            dict.fromkeys(
                uri
                for release in releases
                for uri in [*release["train_uris"], *release["val_uris"]]
            )
        )
        client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max(16, args.workers),
            ),
        )
        transfer = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=1,
            use_threads=False,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(
                pool.map(
                    lambda uri: stage_one(client, transfer, args.stage_root, uri),
                    ordered_uris,
                )
            )
        by_uri = {str(record["uri"]): record for record in records}
        staged_releases = []
        for release in releases:
            train_uris = list(release.pop("train_uris"))
            val_uris = list(release.pop("val_uris"))
            staged_releases.append(
                {
                    **release,
                    "train_objects": [by_uri[uri] for uri in train_uris],
                    "val_objects": [by_uri[uri] for uri in val_uris],
                }
            )
        payload = {
            "schema_version": 2,
            "family": "hpo-probe",
            "releases": staged_releases,
            "object_list_sha256": hashlib.sha256(
                "\n".join(ordered_uris).encode("utf-8")
            ).hexdigest(),
            "total_bytes": sum(int(record["size"]) for record in records),
        }
        manifest = args.stage_root / "ready.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)
        print(
            f"staged {len(records)} objects / {payload['total_bytes']} bytes; "
            f"ready manifest: {manifest}"
        )
    finally:
        destroy_credentials(credential_path)


if __name__ == "__main__":
    main()
