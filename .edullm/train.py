#!/usr/bin/env python3
"""Platform entrypoint. Thin adapter over scripts/train.py.

The platform scaffold auto-generates a command of the form

    bash -lc '<launcher> .edullm/train.py "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR"'

so this file exists to match that signature exactly: a positional run id and a
`--save-folder` flag. It maps them onto the real trainer and does nothing else.

`--save-folder` is an s3:// prefix. memsplit.checkpoint_io handles that scheme;
probing it with Path().exists() is the bug that made a sibling repo silently
repeat every attempt from step 0 at full cost.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--config", default=os.environ.get("MEMSPLIT_CONFIG", "configs/depth_d40m.yaml"))
    ap.add_argument("--condition", default=os.environ.get("MEMSPLIT_CONDITION", "dense"))
    ap.add_argument("--seed", default=os.environ.get("MEMSPLIT_SEED", "0"))
    ap.add_argument("--data-root", default=os.environ.get("MEMSPLIT_DATA_ROOT"))
    args, extra = ap.parse_known_args()

    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--config", args.config,
        "--condition", args.condition,
        "--seed", str(args.seed),
        "--run-id", args.run_id,
        "--checkpoint-dir", args.save_folder,
        *extra,
    ]
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
