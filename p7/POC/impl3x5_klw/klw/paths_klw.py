"""Filesystem layout.

    POC/
      impl5_ssd/
        data/distilled_pool.jsonl       <- the shared distillation pass, read-only here
        runs/D4/socrateach_sft_train.jsonl  <- THE training file every arm uses
        runs/D4/ckpt-923/               <- variant b's reference pi_SFT
      impl3x5_klw/                      <- KLW_ROOT
        data/
          signal_<variant>_<key>.npz            <- merged cache, one per variant
          shards/signal_<variant>_<key>.<k>.npz <- per-GPU shard, merged then kept
          weights_<arm>.json                    <- multiplier diagnostics for one T
        runs/<arm>/
          ckpt-<step>/                  <- 22 adapters, Impl 3's step numbers
          manifest.json
          train.log

The training file is **not** copied per arm. All four arms read
``impl5_ssd/runs/D4/socrateach_sft_train.jsonl`` directly, so it is impossible for two arms to
train on files that differ. Only ``--output_dir`` is per-arm.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._impl5 import paths5

KLW_ROOT = Path(__file__).resolve().parents[1]
POC_ROOT = KLW_ROOT.parent

# Owned by Impl 5 / Impl 4. Read-only from here.
IMPL5_RUNS = paths5.RUNS_DIR
DISTILLED_POOL = paths5.DISTILLED_POOL
PEDAGOGY_POOL = paths5.PEDAGOGY_POOL
ORCD_DATA_DIR = paths5.ORCD_DATA_DIR

DATA_DIR = KLW_ROOT / "data"
SHARD_DIR = DATA_DIR / "shards"
RUNS_DIR = KLW_ROOT / "runs"


def data_arm_dir(arm: str = "D4", runs_root: str | os.PathLike | None = None) -> Path:
    """Where the shared training file lives — an Impl 5 run dir, not one of ours."""
    return (Path(runs_root) if runs_root else IMPL5_RUNS) / arm


def train_file(arm: str = "D4", runs_root: str | os.PathLike | None = None) -> Path:
    return data_arm_dir(arm, runs_root) / "socrateach_sft_train.jsonl"


def reference_adapter(arm: str = "D4", step: int = 923,
                      runs_root: str | os.PathLike | None = None) -> Path:
    return data_arm_dir(arm, runs_root) / f"ckpt-{step}"


def signal_cache(variant: str, key: str, data_dir: str | os.PathLike | None = None) -> Path:
    return (Path(data_dir) if data_dir else DATA_DIR) / f"signal_{variant}_{key}.npz"


def signal_shard(variant: str, key: str, shard: int,
                 shard_dir: str | os.PathLike | None = None) -> Path:
    return (Path(shard_dir) if shard_dir else SHARD_DIR) / f"signal_{variant}_{key}.{shard}.npz"


def run_dir(arm: str, runs_root: str | os.PathLike | None = None) -> Path:
    return (Path(runs_root) if runs_root else RUNS_DIR) / arm


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
