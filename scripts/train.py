#!/usr/bin/env python3
"""Train one arm. Backend-agnostic: works locally, on Colab, and on the platform.

    python scripts/train.py --config configs/depth_d40m.yaml --condition split --seed 0

The only backend-specific input is `--checkpoint-dir`, which may be a local path
or an `s3://` prefix. Platform jobs pass `$EDULLM_CHECKPOINT_DIR`; Colab passes a
Drive path. Nothing else changes between backends, which is the point -- the
previous generation's Colab runs were not reproducible anywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memsplit import checkpoint_io as cio  # noqa: E402
from memsplit.masking import CONDITIONS  # noqa: E402
from memsplit.trainer import TrainConfig, Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="local dir or s3:// prefix; platform jobs pass $EDULLM_CHECKPOINT_DIR",
    )
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stage-dir", default="/tmp/memsplit-corpus",
                    help="local dir to stage an s3:// corpus into")
    ap.add_argument(
        "--no-resume-required",
        action="store_true",
        help="allow a fresh start even if a previous attempt is recorded",
    )
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"{raw.get('name', args.config.stem)}_{args.condition}_s{args.seed}"

    raw.pop("name", None)
    for key, val in (
        ("data_root", args.data_root),
        ("out_dir", args.out_dir),
        ("device", args.device),
    ):
        if val is not None:
            raw[key] = val
    raw.setdefault("out_dir", f"outputs/{run_id}")

    # A corpus given as an s3:// prefix must be staged to local disk: np.memmap
    # cannot read S3. Only this arm's sidecar is fetched, not all four.
    if cio.is_s3(raw["data_root"]):
        staged = cio.stage_files(
            raw["data_root"],
            args.stage_dir,
            ["tokens.bin", f"weights.{args.condition}.bin"],
        )
        print(f"staged corpus -> {staged}")
        raw["data_root"] = str(staged)

    cfg = TrainConfig(
        run_id=run_id,
        condition=args.condition,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume_required=not args.no_resume_required,
        **raw,
    )

    fp = cfg.total_steps and None
    flops = None
    trainer = Trainer(cfg)
    fl = trainer.model.flops_per_token(cfg.ctx)
    print(json.dumps({
        "run_id": run_id,
        "condition": args.condition,
        "seed": args.seed,
        "device": str(trainer.device),
        "total_steps": cfg.total_steps,
        "accum": cfg.accum,
        "loss_divisor": cfg.loss_divisor,
        "checkpoint_dir": trainer.ckpt_root,
        "n_params_total": fl["n_params_total"],
        "flops_per_token": fl["total"],
        "attention_share": round(fl["attention_share"], 4),
        "projected_pflops": round(fl["total"] * cfg.total_tokens / 1e15, 1),
    }, indent=2))

    trainer.train(resume="auto", max_steps=args.max_steps)
    print(f"[{run_id}] done at step {trainer.step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
