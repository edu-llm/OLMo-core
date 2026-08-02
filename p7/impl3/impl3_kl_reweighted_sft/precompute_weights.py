#!/usr/bin/env python
"""Impl 3 — precompute the per-token weight signal (PRD §3.2), cached for the sweep.

Computes ``s_t`` for every pedagogy loss token, once, and caches it. The temperature
sweep (``train_kl_sft.py`` over T) then reuses this cache, so the expensive forward
pass runs a single time per variant.

Variants (PRD §3.2):
  a  base-surprise : s_t = -log pi_0(y_t | context)             [frozen base only]
  b  forward-KL    : s_t = KL(pi_0(.|ctx) || pi_SFT(.|ctx))      [needs --sft_model_id]

The cache is keyed on the exact tokenized ``input_ids`` (same pipeline as training),
so alignment with the run is guaranteed. Optional — the first ``train_kl_sft.py`` run
computes and caches the signal on its own; this just lets you do it up front.

Example:
    python precompute_weights.py --variant a --config config.yaml
    python precompute_weights.py --variant b --sft_model_id ../impl1_2_prompting_sft/out/impl2-sft --config config.yaml
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.cli import build_sft_parser, sft_config_from_args  # noqa: E402
from common.modeling import load_tokenizer  # noqa: E402
from common.sft_train import tokenize_splits  # noqa: E402
from common.weighting import get_or_compute_signal  # noqa: E402


def main():
    parser = build_sft_parser(__doc__)
    parser.add_argument("--variant", choices=["a", "b"], required=True)
    parser.add_argument("--sft_model_id", default=None, help="Vanilla Impl-2 SFT (required for variant b).")
    parser.add_argument("--weights_cache_dir", default="weights")
    args = parser.parse_args()

    cfg = sft_config_from_args(args)
    tokenizer = load_tokenizer(cfg.base_model)
    train_tok, _ = tokenize_splits(cfg, tokenizer)
    print(f"tokenized train rows: {len(train_tok)}")

    get_or_compute_signal(
        train_tok, tokenizer, args.variant, cfg.base_model,
        sft_model_id=args.sft_model_id, cache_dir=args.weights_cache_dir,
    )
    print("Signal cached. Now sweep temperature with train_kl_sft.py.")


if __name__ == "__main__":
    main()
