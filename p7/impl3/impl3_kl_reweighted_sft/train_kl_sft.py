#!/usr/bin/env python
"""Impl 3 — KL-reweighted-loss SFT (PRD §3).

Same data and recipe as Impl 2, but each pedagogy loss token's cross-entropy is
scaled by a per-token multiplier that biases SFT toward the KL-minimal member of the
solution set (RL's Razor). This is a training-OBJECTIVE change only — no data
rewriting, no generation.

  weight signal (§3.2):  a = base-surprise (-log pi_0(y_t))   |  b = forward-KL(pi_0||pi_SFT)
  normalization (§3.3):  global mean-1 over pedagogy tokens, general tokens = 1
  temperature (§3.4):    T in {2, 4, 8, 16, 32}  (T -> inf recovers vanilla Impl 2; T<=1 was unstable)

Run ONE (variant, T) per invocation; sweep externally (see config.yaml / the cluster
runners). Output dir is tagged by variant+T so the sweep's checkpoints don't collide,
and each run keeps >=10 checkpoints for the KL–forgetting curve.

Example:
    python train_kl_sft.py --variant a --temperature 2 --config config.yaml
    python train_kl_sft.py --variant b --temperature 2 --sft_model_id ../impl1_2_prompting_sft/out/impl2-sft --config config.yaml
"""
import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.cli import build_sft_parser, sft_config_from_args  # noqa: E402
from common.sft_train import run_sft  # noqa: E402
from common.weighting import make_attach_weights  # noqa: E402


def main():
    parser = build_sft_parser(__doc__)
    parser.add_argument("--variant", choices=["a", "b"], required=True,
                        help="a = base-surprise; b = forward-KL vs a vanilla SFT.")
    parser.add_argument("--temperature", type=float, required=True,
                        help="Softmax temperature T. Use 'inf' via a huge value to recover Impl 2.")
    parser.add_argument("--sft_model_id", default=None, help="Vanilla Impl-2 SFT (required for variant b).")
    parser.add_argument("--weights_cache_dir", default="weights")
    args = parser.parse_args()

    T = args.temperature if args.temperature > 0 else math.inf
    tag = f"impl3-{args.variant}-T{args.temperature:g}"
    cfg = sft_config_from_args(args)
    # Tag the output dir by (variant, T) so sweep runs don't collide, unless the user
    # pinned an explicit --output_dir.
    if args.output_dir is None:
        base = cfg.output_dir if args.config else "out"
        cfg.output_dir = f"{base.rstrip('/')}/{tag}"
    if cfg.run_name is None:
        cfg.run_name = tag

    attach = make_attach_weights(
        args.variant, T, cfg.base_model,
        sft_model_id=args.sft_model_id, cache_dir=args.weights_cache_dir,
    )
    print(f"KL-reweighted SFT: variant={args.variant} T={args.temperature} -> {cfg.output_dir}")
    run_sft(cfg, attach_weights=attach)


if __name__ == "__main__":
    main()
