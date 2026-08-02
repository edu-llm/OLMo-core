#!/usr/bin/env python
"""Implementation 2 — SI-conditioned SFT (vanilla).

Fine-tunes the base model (default OLMo-2-0425-1B-Instruct) on pedagogy data whose
examples are prefixed with per-dialogue System Instructions, co-trained with ~25%
SI-free general data (PRD §2). This is the D-cell (SFT+SI) deployment model and the
baseline that Impls 3 & 4 must Pareto-beat.

All the real work is in ``common.sft_train.run_sft`` (shared with Impl 3/4). This
file is just the CLI. Per the checkpoint-sweep principle it keeps >= ~10 checkpoints.

Examples:
    # once you have data/ prepared (data is blank for now):
    python train_sft.py --start_from instruct --output_dir out/impl2-sft
    python train_sft.py --config config.yaml --output_dir out/impl2-sft --resume auto
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.cli import build_sft_parser, sft_config_from_args  # noqa: E402
from common.sft_train import run_sft  # noqa: E402


def main():
    parser = build_sft_parser(__doc__)
    args = parser.parse_args()
    cfg = sft_config_from_args(args, **({"output_dir": "out/impl2-sft"} if not (args.output_dir or args.config) else {}))
    run_sft(cfg, attach_weights=None)  # None => uniform loss = vanilla Impl 2


if __name__ == "__main__":
    main()
