#!/usr/bin/env python3
"""Prepare (dry-run) or execute create-only publish of eval/formal-proof-premises-500m v1.

Stages the sealed ``corpus-v3`` evaluator tree under group ``evaluator/`` with profile
``p3-evaluator-corpus/v1`` and an explicit ``depends_on`` pin to
``pretrain/formal-proof-premises-500m/v3``. Never touches immutable pretrain v3, never
writes to ``_staging/eval/p3-math-split-evaluator/...``, and never overwrites an existing
eval v1.

Local ``--execute`` requires honest 64-hex ``EDULLM_CODE_SHA256`` and
``EDULLM_PACKAGES_LOCK_SHA256`` env vars matching the pinned publisher tree and lockfile,
or an AWS Batch executor (``AWS_BATCH_JOB_ID``). Run ``--print-provenance`` for the exact
values/commands. Without a packages lockfile in the pinned checkout, local execute is
blocked — prefer AWS Batch once ``p3-evaluator-corpus/v1`` is deployed in edullm-data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = Path(__file__).resolve().parent
EDULLM_DATA_SRC = REPO_ROOT / ".p3-work" / "full13" / "edullm-data" / "src"
EDULLM_DATA_ROOT = REPO_ROOT / ".p3-work" / "full13" / "edullm-data"
if str(EDULLM_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(EDULLM_DATA_SRC))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import edullm_data  # noqa: E402
from edullm_data.publish import publish, _next_version  # noqa: E402
from edullm_data.s3 import Boto3S3  # noqa: E402

from publish_eval_v1_support import (  # noqa: E402
    provenance_report,
    resolve_execute_provenance,
    run_streaming_preflight,
)

DATASET_ID = "eval/formal-proof-premises-500m"
PRETRAIN_DATASET_ID = "pretrain/formal-proof-premises-500m"
PRETRAIN_VERSION = "v3"
LANDING_BUCKET = "edullm-landing"
DATA_BUCKET = "edullm-data"
PROFILE = "p3-evaluator-corpus/v1"
GROUP = "evaluator"
EXPECTED_VERSION = "v1"
EXPECTED_EVALUATOR_ROOT = "e54954386d85094fc8198c6530f094db505fc88025bfd779f95b419cc74442ba"

PURPOSE = (
    "Sealed P3 formal-proof evaluator corpus for dense/split checkpoint comparisons at "
    "step 23166, to drive run_eval.py with the same JSONL inventory as corpus-v3.zip"
)
OWNER = "edullm-data@alphaaiengineering.com"
ABOUT = (
    "Hardlink-equivalent projection of the v3 sealed formal-proof premises corpus for "
    "post-training evaluation. Each row supplies cited premises, a goal, and a proof target "
    "with mask boundaries for fact-masking arms. Train shards are included for visibility "
    "accounting and semantic-class analysis; held-out sidecars pin the MML/Mizar/ATP/Metamath/"
    "Isabelle splits used by the evaluator. Bytes match the locally sealed corpus-v3.zip "
    "artifact; nothing is re-tokenized or re-serialized at publish time."
)
NOTES = (
    "This eval release is intentionally separate from pretrain/formal-proof-premises-500m/v3 "
    "even though it derives from the same sealed source generation. Consumers must resolve "
    "depends_on[role=pretrain] and verify evaluator_root_sha256 before running evaluations. "
    "Do not read edullm-landing/_staging/eval/p3-math-split-evaluator/... from GPU jobs."
)
LICENSE = {"id": None, "basis": "unknown"}
LIMITATIONS = [
    {
        "kind": "scope",
        "note": (
            "Train JSONL shards are present for evaluator accounting; they are not a second "
            "pretraining release and must not be fed to the pretrain loader."
        ),
    },
    {
        "kind": "license",
        "note": (
            "Upstream formal-proof sources carry mixed or unstated license terms; this release "
            "does not assert redistribution rights beyond the declared unknown basis."
        ),
    },
]


def _load_json_from_s3(s3: Boto3S3, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(s3.get(bucket, key).decode("utf-8"))


def _resolve_pretrain_dependency(s3: Boto3S3, *, data_bucket: str) -> dict[str, Any]:
    prefix = f"{PRETRAIN_DATASET_ID}/{PRETRAIN_VERSION}"
    dataset = _load_json_from_s3(s3, data_bucket, f"{prefix}/dataset.json")
    tokens_group = next((g for g in dataset.get("groups", []) if g.get("name") == "tokens"), None)
    if tokens_group is None:
        sys.exit(f"ABORT: {prefix} has no tokens group")
    manifest_sha256 = tokens_group.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        sys.exit(f"ABORT: {prefix} tokens group missing manifest_sha256")
    return {
        "role": "pretrain",
        "dataset_id": PRETRAIN_DATASET_ID,
        "version": PRETRAIN_VERSION,
        "manifest_sha256": manifest_sha256,
    }


def _read_stage_pins(stage_root: Path) -> dict[str, Any]:
    manifest_path = stage_root / GROUP / "evaluator_manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"ABORT: missing staged evaluator manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluator_root = manifest.get("evaluator_root_sha256")
    source_seal = manifest.get("source_seal", {})
    seal_root = source_seal.get("manifest_root_sha256")
    if evaluator_root != EXPECTED_EVALUATOR_ROOT:
        sys.exit(
            f"ABORT: staged evaluator_root_sha256 {evaluator_root!r} != "
            f"expected {EXPECTED_EVALUATOR_ROOT!r}"
        )
    if not isinstance(seal_root, str) or len(seal_root) != 64:
        sys.exit("ABORT: staged evaluator manifest missing source_seal.manifest_root_sha256")
    return {
        "evaluator_root_sha256": evaluator_root,
        "source_seal_root_sha256": seal_root,
        "total_train_rows": manifest.get("total_train_rows"),
        "total_eval_rows": manifest.get("total_eval_rows"),
    }


def _group_meta(pins: dict[str, Any], *, pretrain_dep: dict[str, Any]) -> dict[str, Any]:
    return {
        GROUP: {
            "evaluator_root_sha256": pins["evaluator_root_sha256"],
            "source_seal_root_sha256": pins["source_seal_root_sha256"],
            "coverage": "incomplete",
            "depends_on": [pretrain_dep],
        }
    }


def _load_eval_family() -> dict[str, Any]:
    return json.loads((EDULLM_DATA_ROOT / "families" / "eval.json").read_text(encoding="utf-8"))


def _planned_objects(plan) -> list[str]:
    keys = [f"{LANDING_BUCKET}/{plan.dataset_id}/{plan.version}/dataset.json"]
    keys.extend(f"{LANDING_BUCKET}/{key}" for key in plan.payload_keys)
    for group in plan.dataset_json.get("groups", []):
        manifest = group.get("manifest")
        if manifest:
            keys.append(f"{LANDING_BUCKET}/{plan.dataset_id}/{plan.version}/{manifest}")
    return sorted(set(keys))


def run_local_streaming_preflight(stage_root: Path, pins: dict[str, Any]) -> dict[str, Any]:
    return run_streaming_preflight(
        stage_root,
        pins,
        family=_load_eval_family(),
        purpose=PURPOSE,
        about=ABOUT,
        notes=NOTES,
        limitations=LIMITATIONS,
        license=LICENSE,
        owner=OWNER,
        dataset_id=DATASET_ID,
        profile=PROFILE,
        group=GROUP,
        expected_version=EXPECTED_VERSION,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage-root",
        default=str(REPO_ROOT / ".p3-work" / "full13" / "publish-stage-eval-v1"),
    )
    ap.add_argument("--execute", action="store_true", help="publish to edullm-landing (default: dry run)")
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="stream-hash staged payload locally and print planned landing inventory (no AWS writes)",
    )
    ap.add_argument(
        "--print-provenance",
        action="store_true",
        help="print deterministic publisher/lock SHA-256 values and execute requirements",
    )
    args = ap.parse_args()

    if args.print_provenance:
        print(json.dumps(provenance_report(EDULLM_DATA_ROOT), indent=2, sort_keys=True))
        print("PROVENANCE_OK")
        return

    assert edullm_data.__version__ == "0.8.0", f"expected publisher 0.8.0, got {edullm_data.__version__}"
    stage_root = Path(args.stage_root).resolve(strict=True)
    pins = _read_stage_pins(stage_root)

    if args.preflight:
        report = run_local_streaming_preflight(stage_root, pins)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("PREFLIGHT_OK")
        return

    s3 = Boto3S3.default(region="us-east-1")
    next_version = _next_version(s3, LANDING_BUCKET, DATASET_ID)
    pretrain_dep = _resolve_pretrain_dependency(s3, data_bucket=DATA_BUCKET)
    group_meta = _group_meta(pins, pretrain_dep=pretrain_dep)

    print(f"edullm_data publisher version: {edullm_data.__version__}")
    print(f"resolved next version (read-only): {next_version}")
    if next_version != EXPECTED_VERSION:
        sys.exit(f"ABORT: next version is {next_version!r}, expected {EXPECTED_VERSION!r}")

    print(f"pretrain depends_on pin: {json.dumps(pretrain_dep)}")
    print(f"evaluator pins: {json.dumps({k: pins[k] for k in ('evaluator_root_sha256', 'source_seal_root_sha256')})}")
    print(f"group_meta[{GROUP}]: {json.dumps(group_meta[GROUP], indent=2)}")
    print("platform blocker: p3-evaluator-corpus/v1 is local-only until edullm-data wheel + validator redeploy")

    if not args.execute:
        print("DRY_RUN_OK (nothing written). Stage with scripts/stage_eval_v1.py, then re-run with --execute.")
        print("Run --print-provenance for execute env requirements; --preflight for bounded local inventory.")
        return

    executor, errors = resolve_execute_provenance(os.environ, edullm_root=EDULLM_DATA_ROOT)
    if errors:
        print("ABORT: --execute blocked by provenance policy:", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        print("Run: python3 scripts/publish_eval_v1.py --print-provenance", file=sys.stderr)
        sys.exit(1)

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan = publish(
        stage_root,
        dataset_id=DATASET_ID,
        purpose=PURPOSE,
        profile=PROFILE,
        s3=s3,
        created_at=created_at,
        owner=OWNER,
        group_meta=group_meta,
        build_executor=executor,
        env=os.environ,
        sources=[],
        about=ABOUT,
        notes=NOTES,
        limitations=LIMITATIONS,
        license=LICENSE,
    )
    if plan.version != EXPECTED_VERSION:
        sys.exit(f"PUBLISHED UNEXPECTED VERSION {plan.version!r}; inspect landing immediately")
    print(f"PUBLISHED {plan.dataset_id}/{plan.version} to {LANDING_BUCKET}")
    print(f"  payload objects: {len(plan.payload_keys)}")
    print("  planned landing keys:")
    for key in _planned_objects(plan):
        print(f"    {key}")
    print("PUBLISH_OK")


if __name__ == "__main__":
    main()
