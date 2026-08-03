#!/usr/bin/env python3
"""Stage one token-selection arm and its frozen references onto RunPod scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
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
REFHQ_BASE = (
    "s3://edullm-checkpoints/olmo-370m/"
    "edullm-370M-refhq-5p5b/checkpoints"
)
LATE_STEPS = (1000, 1125, 1315)


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


def allowed_uri(uri: str) -> bool:
    return uri.startswith("s3://edullm-data/") or uri.startswith(f"{REFHQ_BASE}/step")


def stage_one(client, transfer, root: Path, uri: str) -> dict[str, object]:
    if not allowed_uri(uri):
        raise RuntimeError(f"input is outside the approved RunPod read prefixes: {uri}")
    parsed = urlparse(uri)
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


def prefix_uris(client, prefix_uri: str) -> list[str]:
    parsed = urlparse(prefix_uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        item["Key"]
        for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix)
        for item in page.get("Contents", [])
        if not item["Key"].endswith("/")
    ]
    if not keys:
        raise RuntimeError(f"checkpoint prefix resolved no objects: {prefix_uri}")
    return [f"s3://{parsed.netloc}/{key}" for key in sorted(keys)]


def corpus_payload(corpus, records_by_uri: dict[str, dict]) -> dict:
    return {
        "dataset_id": corpus.dataset_id,
        "version": corpus.version,
        "dtype": corpus.dtype.value,
        "rows": corpus.rows,
        "logical_paths": list(corpus.paths),
        "objects": [records_by_uri[uri] for uri in corpus.paths],
    }


def model_state(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"materialized checkpoint has no model state: {path}")
    return state


def average_references(paths: list[Path], output: Path) -> None:
    import torch

    states = [model_state(path) for path in paths]
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise RuntimeError("late RefHQ reference checkpoints have different parameter keys")
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            accumulator = first.detach().to(torch.float32).clone()
            for state in states[1:]:
                other = state[key]
                if other.shape != first.shape:
                    raise RuntimeError(f"late reference shape mismatch for {key}")
                accumulator.add_(other.detach().to(torch.float32))
            averaged[key] = (accumulator / len(states)).to(first.dtype)
        else:
            averaged[key] = first.detach().clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.partial")
    torch.save({"model": averaged, "steps": list(LATE_STEPS)}, temporary)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    from token_selection_370m.arms import ARM_SPECS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("/workspace/aws-session.env"),
    )
    parser.add_argument("--arm", choices=tuple(ARM_SPECS), required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--refhq-version")
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/workspace/edullm-inputs/token-selection"),
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

        from eval_task_loss_olmo_core import materialize_model_eval
        from token_selection_370m.arms import REFHQ, get_arm
        from train_on_corpus import resolve_corpus

        arm = get_arm(args.arm)
        corpora = {
            arm.dataset_id: resolve_corpus(
                dataset_id=arm.dataset_id,
                version=args.dataset_version,
                tokenizer_id="tokenizer/dolma2-bpe",
            )
        }
        if arm.requires_refhq_stream:
            if not args.refhq_version:
                raise RuntimeError("--refhq-version is required for the BLADE arm")
            corpora[REFHQ] = resolve_corpus(
                dataset_id=REFHQ,
                version=args.refhq_version,
                tokenizer_id="tokenizer/dolma2-bpe",
            )
        client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                retries={"max_attempts": 10, "mode": "adaptive"},
                max_pool_connections=max(16, args.workers),
            ),
        )
        reference_steps: set[int] = set()
        if arm.reference_contract:
            reference_steps.add(1315)
        if arm.late_reference_contract:
            reference_steps.update(LATE_STEPS)
        corpus_uris = [uri for corpus in corpora.values() for uri in corpus.paths]
        checkpoint_uris = {
            step: prefix_uris(client, f"{REFHQ_BASE}/step{step}/")
            for step in sorted(reference_steps)
        }
        ordered_uris = corpus_uris + [
            uri for step in sorted(checkpoint_uris) for uri in checkpoint_uris[step]
        ]
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
        references: dict[str, str] = {}
        materialized: dict[int, Path] = {}
        for step, uris in checkpoint_uris.items():
            prefix = urlparse(f"{REFHQ_BASE}/step{step}/")
            checkpoint = (
                args.stage_root
                / "objects"
                / prefix.netloc
                / prefix.path.lstrip("/").rstrip("/")
            )
            materialized[step] = materialize_model_eval(checkpoint)
        if arm.reference_contract:
            references["reference"] = str(materialized[1315])
        if arm.late_reference_contract:
            late = args.stage_root / "references" / "refhq_late_avg_1000_1125_1315.pt"
            average_references([materialized[step] for step in LATE_STEPS], late)
            references["late"] = str(late)
        payload = {
            "schema_version": 1,
            "family": "token-selection",
            "arm": arm.name,
            "object_list_sha256": hashlib.sha256(
                "\n".join(ordered_uris).encode("utf-8")
            ).hexdigest(),
            "total_bytes": sum(int(record["size"]) for record in records),
            "corpora": {
                name: corpus_payload(corpus, by_uri) for name, corpus in corpora.items()
            },
            "references": references,
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
