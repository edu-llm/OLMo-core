#!/usr/bin/env python3
"""Stage MixLaw's immutable S3 inputs onto RunPod scratch, then destroy AWS credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

EDULLM_DIR = Path(__file__).resolve().parent.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))
from staging_plan import select_paths  # noqa: E402

AWS_CREDENTIAL_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
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


def load_credentials(path: Path) -> None:
    ambient = [key for key in (*AWS_CREDENTIAL_KEYS, *AWS_PROVIDER_KEYS) if os.environ.get(key)]
    if ambient:
        raise RuntimeError(
            "refusing ambient AWS credentials; copy a temporary session to --credentials-file"
        )
    if not path.is_file():
        raise RuntimeError(f"temporary credential file not found: {path}")
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
            raise RuntimeError(f"unsupported line in temporary credential file: {line[:40]!r}")
        parsed = shlex.split(match.group(2), posix=True)
        if len(parsed) != 1:
            raise RuntimeError(f"invalid value for {match.group(1)}")
        values[match.group(1)] = parsed[0]
    missing = [key for key in AWS_CREDENTIAL_KEYS if not values.get(key)]
    if missing:
        raise RuntimeError(f"temporary credential file is missing {missing}")
    os.environ.update(values)


def destroy_credentials(path: Path) -> None:
    for key in AWS_FILE_KEYS | set(AWS_PROVIDER_KEYS):
        os.environ.pop(key, None)
    path.unlink(missing_ok=True)


def local_path(root: Path, uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != "edullm-data":
        raise RuntimeError(f"MixLaw input escaped s3://edullm-data/: {uri}")
    return root / "objects" / parsed.netloc / parsed.path.lstrip("/")


def head(client, bucket: str, key: str) -> dict:
    try:
        return client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except client.exceptions.ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"InvalidArgument", "InvalidRequest", "NotImplemented"}:
            raise
        return client.head_object(Bucket=bucket, Key=key)


def stage_one(
    client,
    transfer_config,
    root: Path,
    uri: str,
    metadata: dict | None = None,
) -> dict[str, object]:
    parsed = urlparse(uri)
    destination = local_path(root, uri)
    metadata = metadata or head(client, parsed.netloc, parsed.path.lstrip("/"))
    expected_size = int(metadata["ContentLength"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != expected_size:
        partial = destination.with_name(destination.name + ".partial")
        partial.unlink(missing_ok=True)
        client.download_file(
            parsed.netloc,
            parsed.path.lstrip("/"),
            str(partial),
            Config=transfer_config,
        )
        if partial.stat().st_size != expected_size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"short S3 download for {uri}")
        partial.replace(destination)
    checksum = metadata.get("ChecksumSHA256")
    if checksum and expected_size <= 64 * 1024 * 1024:
        actual = base64.b64encode(hashlib.sha256(destination.read_bytes()).digest()).decode()
        if actual != checksum:
            raise RuntimeError(f"SHA-256 mismatch for {uri}")
    return {
        "uri": uri,
        "path": str(destination),
        "size": expected_size,
        "etag": str(metadata.get("ETag", "")).strip('"'),
        "version_id": metadata.get("VersionId"),
        "checksum_sha256": checksum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("/workspace/aws-session.env"),
    )
    parser.add_argument("--stage-root", type=Path, default=Path("/workspace/edullm-inputs/mixlaw"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--arm-index", type=int, choices=range(4), default=0)
    parser.add_argument("--length-tokens", type=int)
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.10,
        help="Extra unique tokens staged above the arm's estimated weighted demand.",
    )
    parser.add_argument(
        "--max-files-per-domain",
        type=int,
        default=None,
        help="Stage only the first N real shards per domain for bounded smoke tests.",
    )
    return parser.parse_args()


def credential_path_from_argv() -> Path:
    result = Path("/workspace/aws-session.env")
    for index, value in enumerate(sys.argv[1:]):
        if value == "--credentials-file" and index + 2 <= len(sys.argv[1:]):
            result = Path(sys.argv[index + 2])
        elif value.startswith("--credentials-file="):
            result = Path(value.split("=", 1)[1])
    return result


def main() -> None:
    credential_path = credential_path_from_argv()
    try:
        args = parse_args()
        credential_path = args.credentials_file
        if args.workers <= 0:
            raise SystemExit("--workers must be positive")
        if args.max_files_per_domain is not None and args.max_files_per_domain <= 0:
            raise SystemExit("--max-files-per-domain must be positive")
        if not math.isfinite(args.headroom) or args.headroom < 1.0:
            raise SystemExit("--headroom must be finite and at least 1.0")
        manifest_path = args.stage_root / "ready.json"
        load_credentials(args.credentials_file)
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        from mixlaw_entrypoint import (
            ARMS,
            DATASET_ID,
            DATASET_VERSION,
            GLOBAL_BATCH_TOKENS,
            normalized_weights,
            resolve_domain_sources,
            steps_for_length,
        )

        sources = resolve_domain_sources()
        arm = ARMS[args.arm_index]
        train_tokens = steps_for_length(args.length_tokens) * GLOBAL_BATCH_TOKENS
        weights = normalized_weights(arm)
        client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max(16, args.workers),
            ),
        )
        metadata_by_uri: dict[str, dict] = {}

        def metadata(uri: str) -> dict:
            if uri not in metadata_by_uri:
                parsed = urlparse(uri)
                metadata_by_uri[uri] = head(client, parsed.netloc, parsed.path.lstrip("/"))
            return metadata_by_uri[uri]

        selected_paths: dict[str, tuple[str, ...]] = {}
        staged_tokens: dict[str, int] = {}
        for index, source in enumerate(sources):
            selected, tokens = select_paths(
                source.paths,
                required_tokens=train_tokens * weights[index],
                headroom=args.headroom,
                size_of=lambda uri: int(metadata(uri)["ContentLength"]),
                max_files=args.max_files_per_domain,
            )
            selected_paths[source.name] = selected
            staged_tokens[source.name] = tokens
        ordered_uris = [uri for source in sources for uri in selected_paths[source.name]]
        if len(ordered_uris) != len(set(ordered_uris)):
            raise RuntimeError("published domain views unexpectedly contain duplicate objects")
        transfer = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=1,
            use_threads=False,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(
                pool.map(
                    lambda uri: stage_one(
                        client,
                        transfer,
                        args.stage_root,
                        uri,
                        metadata_by_uri[uri],
                    ),
                    ordered_uris,
                )
            )
        by_uri = {str(record["uri"]): record for record in records}
        payload = {
            "schema_version": 1,
            "family": "mixlaw",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "arm_index": arm.index,
            "arm_name": arm.name,
            "requested_tokens": train_tokens,
            "headroom": args.headroom,
            "selection_mode": (
                "file-cap" if args.max_files_per_domain is not None else "weighted-demand"
            ),
            "object_list_sha256": hashlib.sha256(
                "\n".join(ordered_uris).encode("utf-8")
            ).hexdigest(),
            "total_bytes": sum(int(record["size"]) for record in records),
            "domains": [
                {
                    "name": source.name,
                    "weight": weights[index],
                    "required_tokens": math.ceil(train_tokens * weights[index]),
                    "available_tokens": min(source.available_tokens, staged_tokens[source.name]),
                    "objects": [by_uri[uri] for uri in selected_paths[source.name]],
                }
                for index, source in enumerate(sources)
            ],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest_path)
        print(
            f"staged {len(records)} objects / {payload['total_bytes']} bytes; "
            f"ready manifest: {manifest_path}"
        )
    finally:
        destroy_credentials(credential_path)


if __name__ == "__main__":
    main()
