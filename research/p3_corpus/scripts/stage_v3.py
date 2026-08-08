#!/usr/bin/env python3
"""Build the payload-only v3 staging tree and run local pre-publication gates.

Stages only the manifest-declared ``*.u32le.bin`` shards (hardlinked from the fresh
v3 token output), then runs:

* ``validate_staged_token_payload`` -- source-seal, per-shard SHA-256, exact dtype/
  byte-order/arithmetic, cross-split (train/val) binding, exact path-set equality,
  and six-family inventory, all sealed by the fixed Qwen tokenizer four-part seal.
* a raw byte scan of every staged shard -- vocab range (< 151665), non-negativity,
  little-endian uint32 seq-len alignment (len % 16384 == 0), EOS (151643) presence,
  distinct-ID floor (>= 128, matching the deployed pretrain policy), and a guard
  that the first bytes are not a NumPy ``.npy`` header.

The authoritative deployed profile checks re-run at promotion time; this is the
offline pre-flight that must pass before any bytes are uploaded to landing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OLMO_P3 = (
    REPO_ROOT.parent
    / "eduLLM"
    / "OLMo-core"
    / "src"
    / "scripts"
    / "train"
    / "p3_math_split"
)
if str(OLMO_P3) not in sys.path:
    sys.path.insert(0, str(OLMO_P3))

import tokenize_corpus as tc  # noqa: E402

VOCAB_SIZE = 151_665
EOS_ID = 151_643
SEQ_LEN = 16_384
DISTINCT_ID_FLOOR = 128
NPY_MAGIC = b"\x93NUMPY"


def _manifest_shard_paths(tokenized_root: Path) -> list[str]:
    paths: list[str] = []
    for split in ("train", "val"):
        manifest = json.loads((tokenized_root / f"{split}_meta.json").read_text())
        for group in manifest["groups"].values():
            for shard in group["shards"]:
                paths.append(shard["path"])  # e.g. tokens/enigma/train-00000.u32le.bin
    if len(set(paths)) != len(paths):
        sys.exit("duplicate shard path across manifests")
    return sorted(paths)


def _build_stage(tokenized_root: Path, stage_root: Path) -> list[str]:
    if stage_root.exists() or stage_root.is_symlink():
        sys.exit(f"stage root must be created fresh; refusing to reuse {stage_root}")
    rel_paths = _manifest_shard_paths(tokenized_root)
    for rel in rel_paths:
        if not rel.startswith("tokens/") or not rel.endswith(".u32le.bin"):
            sys.exit(f"unexpected manifest shard path: {rel}")
        src = (tokenized_root / rel).resolve(strict=True)
        dst = stage_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)  # hardlink from fresh v3 output only
    return rel_paths


def _byte_scan(stage_root: Path, rel_paths: list[str]) -> dict:
    summary = {}
    for rel in rel_paths:
        path = stage_root / rel
        with path.open("rb") as handle:
            head = handle.read(6)
        if head[:6] == NPY_MAGIC:
            sys.exit(f"{rel}: begins with a NumPy .npy header")
        size = path.stat().st_size
        if size % (SEQ_LEN * 4) != 0:
            sys.exit(f"{rel}: byte length {size} is not a whole number of {SEQ_LEN}-token rows")
        arr = np.fromfile(path, dtype="<u4")
        if arr.size == 0:
            sys.exit(f"{rel}: empty shard")
        max_id = int(arr.max())
        min_id = int(arr.min())
        if max_id >= VOCAB_SIZE:
            sys.exit(f"{rel}: token id {max_id} >= vocab size {VOCAB_SIZE}")
        if min_id < 0:
            sys.exit(f"{rel}: negative token id {min_id}")
        distinct = int(np.unique(arr).size)
        if distinct < DISTINCT_ID_FLOOR:
            sys.exit(f"{rel}: only {distinct} distinct ids (< {DISTINCT_ID_FLOOR})")
        eos_count = int((arr == EOS_ID).sum())
        if eos_count == 0:
            sys.exit(f"{rel}: no EOS ({EOS_ID}) token present")
        summary[rel] = {
            "tokens": int(arr.size),
            "max_id": max_id,
            "distinct_ids": distinct,
            "eos_count": eos_count,
        }
        del arr
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokenized-root",
        default=str(REPO_ROOT / ".p3-work" / "full13" / "tokenized-v3"),
    )
    ap.add_argument(
        "--stage-root",
        default=str(REPO_ROOT / ".p3-work" / "full13" / "publish-stage-v3"),
    )
    args = ap.parse_args()
    tokenized_root = Path(args.tokenized_root).resolve(strict=True)
    stage_root = Path(args.stage_root)

    rel_paths = _build_stage(tokenized_root, stage_root)
    report = tc.validate_staged_token_payload(
        stage_root,
        train_manifest_path=tokenized_root / "train_meta.json",
        val_manifest_path=tokenized_root / "val_meta.json",
    )
    scan = _byte_scan(stage_root, rel_paths)

    print(f"staged {len(rel_paths)} shards -> {stage_root}")
    print(f"  families: {report['families']}")
    for split in ("train", "val"):
        part = report["partitions"][split]
        print(f"  {split}: files={part['files']} padded_tokens={part['tokens']:,}")
    print("  byte scan (all shards passed vocab/EOS/alignment/distinct>=128):")
    for rel in rel_paths:
        s = scan[rel]
        print(
            f"    {rel:<42} tokens={s['tokens']:>11,} max_id={s['max_id']:>6} "
            f"distinct={s['distinct_ids']:>6} eos={s['eos_count']:>7,}"
        )
    print("LOCAL_STAGE_GATES_PASSED")


if __name__ == "__main__":
    main()
