"""Validated eduLLM-data resolution and job-local staging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

DATA_BUCKET = "edullm-data"


class PublishedInputError(RuntimeError):
    """A published input is absent, mutable, or bound to another parent."""


@dataclass(frozen=True)
class ResolvedInput:
    dataset_id: str
    version: str
    group: str
    profile: str
    manifest_sha256: str
    paths: tuple[str, ...]
    numpy_dtype: str
    header_bytes: int

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "group": self.group,
            "profile": self.profile,
            "manifest_sha256": self.manifest_sha256,
            "paths": list(self.paths),
            "numpy_dtype": self.numpy_dtype,
            "header_bytes": self.header_bytes,
        }


def _load_json(s3: Any, key: str) -> dict[str, Any]:
    payload = json.loads(s3.get(DATA_BUCKET, key).decode("utf-8"))
    if not isinstance(payload, dict):
        raise PublishedInputError(f"{key} is not a JSON object")
    return payload


def _group(dataset: Mapping[str, Any], name: str | None) -> Mapping[str, Any]:
    groups = dataset.get("groups") or []
    if name is None:
        if len(groups) != 1:
            raise PublishedInputError(
                f"parent dataset must have one unambiguous group, found {len(groups)}"
            )
        return groups[0]
    matches = [group for group in groups if group.get("name") == name]
    if len(matches) != 1:
        raise PublishedInputError(f"expected one group {name!r}, found {len(matches)}")
    return matches[0]


def validate_parent_order_binding(
    *,
    parent_dataset: Mapping[str, Any],
    order_dataset: Mapping[str, Any],
    parent_dataset_id: str,
    parent_version: str,
    parent_manifest_sha256: str,
    order_group: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the exact parent manifest pinned by one token-order group."""
    parent = _group(parent_dataset, None)
    if parent.get("profile") != "pretrain-tokens/v1":
        raise PublishedInputError(
            f"parent profile must be pretrain-tokens/v1, got {parent.get('profile')!r}"
        )
    if parent.get("manifest_sha256") != parent_manifest_sha256:
        raise PublishedInputError("parent manifest does not match the immutable recipe pin")

    order = _group(order_dataset, order_group)
    if order.get("profile") != "token-order/v1":
        raise PublishedInputError(
            f"order profile must be token-order/v1, got {order.get('profile')!r}"
        )
    dependencies = [
        dependency
        for dependency in order.get("depends_on") or []
        if dependency.get("role") == "token_pool"
    ]
    if len(dependencies) != 1:
        raise PublishedInputError("order group must declare exactly one token_pool dependency")
    dependency = dependencies[0]
    expected = {
        "dataset_id": parent_dataset_id,
        "version": parent_version,
        "manifest_sha256": parent_manifest_sha256,
    }
    actual = {key: dependency.get(key) for key in expected}
    if actual != expected:
        raise PublishedInputError(f"order binds {actual!r}, not the staged parent {expected!r}")
    return parent, order


def _resolved_input(
    *,
    dataset_id: str,
    version: str,
    group_doc: Mapping[str, Any],
    resolved: Any,
) -> ResolvedInput:
    dtype = resolved.numpy_dtype
    if dtype != "<u4":
        raise PublishedInputError(
            f"{dataset_id}/{version} must be explicit little-endian uint32, got {dtype!r}"
        )
    if int(resolved.header_bytes or 0) != 0:
        raise PublishedInputError(
            f"{dataset_id}/{version} must be headerless, got {resolved.header_bytes} bytes"
        )
    manifest_hash = group_doc.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise PublishedInputError(f"{dataset_id}/{version} has no immutable manifest hash")
    return ResolvedInput(
        dataset_id=dataset_id,
        version=version,
        group=str(group_doc["name"]),
        profile=str(group_doc["profile"]),
        manifest_sha256=manifest_hash,
        paths=tuple(resolved.paths),
        numpy_dtype=dtype,
        header_bytes=int(resolved.header_bytes or 0),
    )


def resolve_published_inputs(
    *,
    parent_dataset_id: str,
    parent_version: str,
    parent_manifest_sha256: str,
    order_dataset_id: str | None,
    order_version: str | None,
    order_group: str | None,
) -> tuple[ResolvedInput, ResolvedInput | None]:
    """Resolve sealed manifests and validate the order's exact parent dependency."""
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    parent_doc = _load_json(s3, f"{parent_dataset_id}/{parent_version}/dataset.json")
    parent_group = _group(parent_doc, None)
    parent_read = dataset_paths(
        parent_dataset_id,
        parent_version,
        split="train",
        s3=s3,
        group=str(parent_group["name"]),
    )
    parent = _resolved_input(
        dataset_id=parent_dataset_id,
        version=parent_version,
        group_doc=parent_group,
        resolved=parent_read,
    )
    if parent.manifest_sha256 != parent_manifest_sha256:
        raise PublishedInputError(
            f"parent manifest {parent.manifest_sha256} != recipe pin {parent_manifest_sha256}"
        )
    if order_dataset_id is None:
        return parent, None
    if order_group is None:
        raise PublishedInputError("curriculum order group is required")

    resolved_order_version = order_version or resolve_latest(order_dataset_id, s3=s3)
    if not resolved_order_version:
        raise PublishedInputError(f"no published version of {order_dataset_id}")
    order_doc = _load_json(s3, f"{order_dataset_id}/{resolved_order_version}/dataset.json")
    _, order_group_doc = validate_parent_order_binding(
        parent_dataset=parent_doc,
        order_dataset=order_doc,
        parent_dataset_id=parent_dataset_id,
        parent_version=parent_version,
        parent_manifest_sha256=parent_manifest_sha256,
        order_group=order_group,
    )
    order_read = dataset_paths(
        order_dataset_id,
        resolved_order_version,
        split="train",
        s3=s3,
        group=order_group,
    )
    order = _resolved_input(
        dataset_id=order_dataset_id,
        version=resolved_order_version,
        group_doc=order_group_doc,
        resolved=order_read,
    )
    return parent, order


def _local_object_path(cache_dir: Path, uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != DATA_BUCKET:
        raise PublishedInputError(f"only s3://{DATA_BUCKET}/ inputs are accepted: {uri}")
    return cache_dir / parsed.netloc / parsed.path.lstrip("/")


def stage_input(resolved: ResolvedInput, cache_dir: str | Path) -> tuple[Path, ...]:
    """Atomically download missing immutable input objects into job scratch."""
    import boto3

    root = Path(cache_dir)
    client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    staged: list[Path] = []
    for uri in resolved.paths:
        destination = _local_object_path(root, uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = urlparse(uri).path.lstrip("/")
        expected_size = int(client.head_object(Bucket=DATA_BUCKET, Key=key)["ContentLength"])
        if not destination.is_file() or destination.stat().st_size != expected_size:
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            client.download_file(DATA_BUCKET, key, str(temporary))
            if temporary.stat().st_size != expected_size:
                temporary.unlink(missing_ok=True)
                raise PublishedInputError(f"short staged object: {uri}")
            temporary.replace(destination)
        staged.append(destination)
    return tuple(staged)


def load_order(paths: tuple[Path, ...], dtype: str) -> np.ndarray:
    parts = [np.memmap(path, mode="r", dtype=dtype) for path in paths]
    if not parts:
        raise PublishedInputError("curriculum order resolved to no objects")
    return np.asarray(
        np.concatenate(parts) if len(parts) > 1 else parts[0],
        dtype=np.int64,
    )
