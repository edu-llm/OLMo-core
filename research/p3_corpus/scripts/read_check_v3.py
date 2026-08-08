#!/usr/bin/env python3
"""Resolve the promoted v3 dataset with whatever edullm_data is on EDULLM_SRC.

Read-only: verifies the data-bucket _VALIDATED marker + recomputes the seal, then
resolves train/val shard URIs. Run once per reader version (image-pinned 0.5.0 and
publisher 0.8.0) to prove both can slice v3.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ["EDULLM_SRC"])

import edullm_data  # noqa: E402
from edullm_data.read import dataset_paths  # noqa: E402
from edullm_data.s3 import Boto3S3  # noqa: E402

DATASET_ID = "pretrain/formal-proof-premises-500m"
VERSION = "v3"


def _attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def main() -> None:
    s3 = Boto3S3.default()
    print(f"reader edullm_data {edullm_data.__version__}")
    for split in ("train", "val"):
        r = dataset_paths(DATASET_ID, VERSION, split=split, s3=s3)
        paths = _attr(r, "paths") or []
        rows = _attr(r, "rows", "split_rows")
        dtype = _attr(r, "dtype")
        print(f"  {split}: resolved {len(paths)} shards  rows={rows}  dtype={dtype}")
        for p in sorted(str(x) for x in paths):
            print(f"    {p}")
    print("READER_RESOLVE_OK")


if __name__ == "__main__":
    main()
