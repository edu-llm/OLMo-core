#!/usr/bin/env python3
"""Publish Formal Proof Premises v3 to edullm-landing (edullm-data 0.8.0 publisher).

Two-step by design:

* default (dry run): resolve the next version read-only and *abort unless it is
  exactly ``v3``; print the intended dataset metadata. Nothing is written.
* ``--execute``: run ``publish()`` for real. It stages the payload-only tree to a
  landing ``_staging`` prefix, reserves ``v3`` with a create-only ``dataset.json``,
  server-side copies the shards to ``.../v3/``, and writes the tokens manifest last.
  It never deletes, overwrites, or copies from v2 or any other version.

Descriptive ``sources`` token totals are computed from the real train shard bytes
(bytes / 4), matching the v2 convention exactly, so they are verifiable and never
hand-typed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EDULLM_DATA_SRC = REPO_ROOT / ".p3-work" / "full13" / "edullm-data" / "src"
if str(EDULLM_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(EDULLM_DATA_SRC))

import edullm_data  # noqa: E402
from edullm_data.publish import publish, _next_version  # noqa: E402
from edullm_data.s3 import Boto3S3  # noqa: E402

DATASET_ID = "pretrain/formal-proof-premises-500m"
LANDING_BUCKET = "edullm-landing"
DATA_BUCKET = "edullm-data"
TOKENIZER = "tokenizer/qwen25-vendored/v1"
PROFILE = "pretrain-tokens/v1"
SEQ_LEN = 16_384

PURPOSE = (
    "Formal proofs with cited premises for Qwen2.5-0.5B dense/split fact-masking arms, "
    "to test whether mathematical reasoning is maintained without learning supplied facts"
)
OWNER = "edullm-data@alphaaiengineering.com"
ABOUT = (
    "A deterministic formal-mathematics continual-pretraining corpus. Each document "
    "supplies the named premises it uses, followed by a proof target. Dense training "
    "supervises the fact block; split training derives and masks that block at load time "
    "while retaining it as context. Sources are kept in separate object-key slices. v3 is "
    "a repaired supersession of v2 rebuilt from hash-sealed final splits: it recovers "
    "additional direct-Mizar proofs, adds materially distinct low-tier ENIGMA traces, "
    "applies a 16,384-token eligibility policy to Metamath before held-out selection, and "
    "rebuilds the pooled MML semantic hold-out."
)
NOTES = (
    "Packed at sequence length 16,384 with intra-document EOS boundaries. Fact masks are "
    "derived from the tokenized separator core [10952,15513,969]; no mask sidecar is "
    "published. min_distinct_ids is 128 because ATP/TPTP symbol windows measure far below "
    "the prose-oriented family default of 256 under Qwen's tokenizer."
)
LICENSE = {"id": None, "basis": "unknown"}

FAMILY_DISPLAY = {
    "enigma": "ENIGMA",
    "isabelle": "Isabelle/Magnushammer",
    "metamath": "Metamath",
    "mizar": "Mizar",
    "prf2": "MPTP prf2",
    "thproofs": "Mizar thproofs",
}


def _sources_from_train_bytes(tokenized_root: Path) -> list[dict]:
    manifest = json.loads((tokenized_root / "train_meta.json").read_text())
    out = []
    for family, display in FAMILY_DISPLAY.items():
        group = manifest["groups"][family]
        train_bytes = sum(s["bytes"] for s in group["shards"])
        if train_bytes % 4 != 0:
            sys.exit(f"{family} train bytes not divisible by 4")
        out.append({"name": display, "tokens": train_bytes // 4})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-root", default=str(REPO_ROOT / ".p3-work" / "full13" / "publish-stage-v3"))
    ap.add_argument("--tokenized-root", default=str(REPO_ROOT / ".p3-work" / "full13" / "tokenized-v3"))
    ap.add_argument("--execute", action="store_true", help="actually publish (default: dry run)")
    args = ap.parse_args()

    assert edullm_data.__version__ == "0.8.0", f"expected publisher 0.8.0, got {edullm_data.__version__}"
    stage_root = Path(args.stage_root).resolve(strict=True)
    tokenized_root = Path(args.tokenized_root).resolve(strict=True)

    sources = _sources_from_train_bytes(tokenized_root)
    limitations = [
        {"kind": "context_length", "max_tokens": SEQ_LEN, "dropped_train_documents": 0},
        {
            "kind": "source_mix",
            "note": "ATP/TPTP traces contribute most token mass; results should also be reported per source.",
        },
        {
            "kind": "repaired_release",
            "note": (
                "v3 supersedes v2: recovered direct-Mizar proofs, added materially distinct "
                "low-tier ENIGMA traces, applied a 16,384-token Metamath eligibility policy "
                "before held-out selection, and rebuilt the pooled MML semantic hold-out. "
                "Every family's tokenized bytes differ from v2; no shard is byte-identical."
            ),
        },
    ]
    group_meta = {
        "tokens": {
            "seq_len": SEQ_LEN,
            "coverage": "partition",
            "min_distinct_ids": 128,
            "partitions": [
                {"name": "train", "by": "path", "glob": "train-*.u32le.bin"},
                {"name": "val", "by": "path", "glob": "val-*.u32le.bin"},
            ],
        }
    }

    s3 = Boto3S3.default(region="us-east-1")
    next_version = _next_version(s3, LANDING_BUCKET, DATASET_ID)
    print(f"edullm_data publisher version: {edullm_data.__version__}")
    print(f"resolved next version (read-only): {next_version}")
    if next_version != "v3":
        sys.exit(f"ABORT: next version is {next_version!r}, expected 'v3'. Refusing to publish.")

    print("intended sources (train padded tokens = bytes/4):")
    for s in sources:
        print(f"  {s['name']:<22} {s['tokens']:>12,}")
    print(f"group_meta: {json.dumps(group_meta)}")

    if not args.execute:
        print("DRY_RUN_OK (nothing written). Re-run with --execute to publish.")
        return

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan = publish(
        stage_root,
        dataset_id=DATASET_ID,
        purpose=PURPOSE,
        profile=PROFILE,
        tokenizer=TOKENIZER,
        s3=s3,
        created_at=created_at,
        owner=OWNER,
        group_meta=group_meta,
        sources=sources,
        about=ABOUT,
        notes=NOTES,
        limitations=limitations,
        license=LICENSE,
    )
    if plan.version != "v3":
        sys.exit(f"PUBLISHED UNEXPECTED VERSION {plan.version!r} (expected v3); inspect landing immediately")
    print(f"PUBLISHED {plan.dataset_id}/{plan.version} to {LANDING_BUCKET}")
    print(f"  payload objects: {len(plan.payload_keys)}")
    print(f"  groups: {[g['name'] for g in plan.dataset_json['groups']]}")
    tokens_group = plan.dataset_json["groups"][0]
    print(f"  tokens group partitions: {json.dumps(tokens_group.get('partitions'))}")
    print(f"  min_distinct_ids: {tokens_group.get('min_distinct_ids')}")
    print(f"  depends_on: {json.dumps(tokens_group.get('depends_on'))}")
    print(f"  dataset schema_version: {plan.dataset_json.get('schema_version')}")
    print("PUBLISH_OK")


if __name__ == "__main__":
    main()
