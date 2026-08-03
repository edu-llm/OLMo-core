#!/usr/bin/env python3
"""Stage Skill-It's immutable S3 corpus onto RunPod scratch and remove credentials."""

from __future__ import annotations

import argparse
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


def load_credentials(path: Path) -> None:
    if any(os.environ.get(key) for key in (*AWS_CREDENTIAL_KEYS, *AWS_PROVIDER_KEYS)):
        raise RuntimeError("refusing ambient AWS credentials; use --credentials-file")
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
    for key in AWS_FILE_KEYS | set(AWS_PROVIDER_KEYS):
        os.environ.pop(key, None)
    path.unlink(missing_ok=True)


def stage_one(
    client,
    transfer,
    root: Path,
    uri: str,
    metadata: dict | None = None,
) -> dict[str, object]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != "edullm-data":
        raise RuntimeError(f"Skill-It input escaped s3://edullm-data/: {uri}")
    key = parsed.path.lstrip("/")
    destination = root / "objects" / parsed.netloc / key
    metadata = metadata or client.head_object(Bucket=parsed.netloc, Key=key)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("/workspace/aws-session.env"),
    )
    parser.add_argument("--stage-root", type=Path, default=Path("/workspace/edullm-inputs/skillit"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--length-tokens", type=int)
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.25,
        help="Extra unique tokens above initial domain-weight demand for later Skill-It shifts.",
    )
    parser.add_argument("--max-files-per-domain", type=int)
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


def main() -> None:
    credential_path = credential_path_from_argv()
    try:
        args = parse_args()
        credential_path = args.credentials_file
        if args.workers <= 0:
            raise SystemExit("--workers must be positive")
        if args.length_tokens is not None and args.length_tokens <= 0:
            raise SystemExit("--length-tokens must be positive")
        if not math.isfinite(args.headroom) or args.headroom < 1.0:
            raise SystemExit("--headroom must be finite and at least 1.0")
        if args.max_files_per_domain is not None and args.max_files_per_domain <= 0:
            raise SystemExit("--max-files-per-domain must be positive")
        load_credentials(args.credentials_file)
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        from edullm_data.read import dataset_paths
        from edullm_data.s3 import Boto3S3

        from skillit_loader import GLOBAL_BATCH_TOKENS, TOTAL_STEPS
        from skillit_math import DATASET_ID, DATASET_VERSION, DOMAINS, initial_weights

        s3 = Boto3S3.default()
        resolved_domains = []
        for domain in DOMAINS:
            resolved = dataset_paths(
                DATASET_ID,
                DATASET_VERSION,
                split="train",
                s3=s3,
                labels={"source": domain},
            )
            if (
                not resolved.paths
                or resolved.dtype != "uint32"
                or resolved.byte_order != "little"
                or int(resolved.header_bytes or 0) != 0
            ):
                raise RuntimeError(f"{domain}: published corpus violates Skill-It contract")
            resolved_domains.append((domain, resolved))
        client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max(16, args.workers),
            ),
        )
        train_tokens = args.length_tokens or TOTAL_STEPS * GLOBAL_BATCH_TOKENS
        weights = initial_weights()
        metadata_by_uri: dict[str, dict] = {}

        def metadata(uri: str) -> dict:
            if uri not in metadata_by_uri:
                parsed = urlparse(uri)
                metadata_by_uri[uri] = client.head_object(
                    Bucket=parsed.netloc,
                    Key=parsed.path.lstrip("/"),
                )
            return metadata_by_uri[uri]

        selected_paths: dict[str, tuple[str, ...]] = {}
        staged_tokens: dict[str, int] = {}
        for index, (domain, resolved) in enumerate(resolved_domains):
            selected, tokens = select_paths(
                tuple(resolved.paths),
                required_tokens=train_tokens * float(weights[index]),
                headroom=args.headroom,
                size_of=lambda uri: int(metadata(uri)["ContentLength"]),
                max_files=args.max_files_per_domain,
            )
            selected_paths[domain] = selected
            staged_tokens[domain] = tokens
        ordered_uris = [
            uri for domain, _ in resolved_domains for uri in selected_paths[domain]
        ]
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
            "family": "skillit",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "requested_tokens": train_tokens,
            "headroom": args.headroom,
            "weight_basis": "initial_weights",
            "selection_mode": (
                "file-cap" if args.max_files_per_domain is not None else "weighted-demand"
            ),
            "object_list_sha256": hashlib.sha256(
                "\n".join(ordered_uris).encode("utf-8")
            ).hexdigest(),
            "total_bytes": sum(int(record["size"]) for record in records),
            "domains": [
                {
                    "name": domain,
                    "weight": float(weights[index]),
                    "required_tokens": math.ceil(train_tokens * float(weights[index])),
                    "rows": min(int(resolved.rows or 0), staged_tokens[domain]),
                    "objects": [by_uri[uri] for uri in selected_paths[domain]],
                }
                for index, (domain, resolved) in enumerate(resolved_domains)
            ],
        }
        manifest = args.stage_root / "ready.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(manifest)
        print(
            f"staged {len(records)} objects / {payload['total_bytes']} bytes; "
            f"ready manifest: {manifest}"
        )
    finally:
        destroy_credentials(credential_path)


if __name__ == "__main__":
    main()
