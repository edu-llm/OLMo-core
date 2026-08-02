#!/usr/bin/env python
"""Impl 4 — SDFT training on a gold/distilled mix (PRD §4.2, distilled-fraction sweep).

Builds a training set that replaces a fraction ``f`` of the gold pedagogy targets with
their self-distilled rewrites (from ``self_distill.py``), optionally adds self-distilled
general-domain data, then runs the standard Impl-2 SFT recipe on it. Sweeping
``f ∈ [0, 1]`` (0 = vanilla Impl 2, 1 = full SDFT) traces the KL–forgetting plane.

The gold train file is matched to distilled rewrites by ``dialogue_id``. The mixed
split is written to ``<work_dir>/data/`` and trained via ``common.sft_train.run_sft``
(so it keeps ≥10 checkpoints like every other run).

Example:
    python train_sdft.py --distilled_frac 0.5 \
        --gold_dir ../data --distilled_pedagogy distilled/pedagogy_rewrite.jsonl \
        --distilled_general distilled/general_domains.jsonl --config config.yaml
"""
import argparse
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.cli import build_sft_parser, sft_config_from_args  # noqa: E402
from common.data import SPLIT_FILES, load_jsonl, write_jsonl  # noqa: E402
from common.sft_train import run_sft  # noqa: E402


def build_mixed_train(gold_dir, distilled_pedagogy, distilled_general, frac, seed, out_dir):
    """Replace a fraction of gold pedagogy examples with distilled rewrites; copy val/test."""
    gold_train = load_jsonl(os.path.join(gold_dir, SPLIT_FILES["train"]))
    rewrites = {r.get("dialogue_id"): r for r in load_jsonl(distilled_pedagogy)} if distilled_pedagogy else {}

    ped = [e for e in gold_train if e.get("kind", "pedagogy") != "general"]
    gen = [e for e in gold_train if e.get("kind") == "general"]

    rng = random.Random(seed)
    order = list(range(len(ped)))
    rng.shuffle(order)
    n_replace = int(round(frac * len(ped)))
    replace_idx = set(order[:n_replace])
    n_hit = 0
    mixed_ped = []
    for i, e in enumerate(ped):
        if i in replace_idx and e.get("dialogue_id") in rewrites:
            mixed_ped.append(rewrites[e["dialogue_id"]])
            n_hit += 1
        else:
            mixed_ped.append(e)

    extra_general = load_jsonl(distilled_general) if distilled_general else []
    mixed = mixed_ped + gen + extra_general
    rng.shuffle(mixed)

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, SPLIT_FILES["train"]), mixed)
    for split in ("val", "test"):
        src = os.path.join(gold_dir, SPLIT_FILES[split])
        if os.path.exists(src):
            write_jsonl(os.path.join(out_dir, SPLIT_FILES[split]), load_jsonl(src))
    print(f"mixed train: {len(mixed_ped)} pedagogy ({n_hit} distilled @ frac={frac}) "
          f"+ {len(gen)} gold-general + {len(extra_general)} distilled-general = {len(mixed)}")
    return out_dir


def main():
    parser = build_sft_parser(__doc__)
    parser.add_argument("--distilled_frac", type=float, required=True, help="0 = vanilla Impl 2, 1 = full SDFT.")
    parser.add_argument("--gold_dir", default="../data", help="Dir with the gold socrateach_sft_{train,val,test}.jsonl.")
    parser.add_argument("--distilled_pedagogy", default=None, help="JSONL of self-distilled pedagogy rewrites.")
    parser.add_argument("--distilled_general", default=None, help="JSONL of self-distilled general-domain data.")
    parser.add_argument("--work_dir", default=None, help="Where to write the mixed split (default out/<tag>).")
    args = parser.parse_args()

    tag = f"impl4-sdft-f{args.distilled_frac:g}"
    work_dir = args.work_dir or f"out/{tag}"
    mixed_data_dir = build_mixed_train(
        args.gold_dir, args.distilled_pedagogy, args.distilled_general,
        args.distilled_frac, args.seed if args.seed is not None else 13,
        os.path.join(work_dir, "data"),
    )

    # Force the local distilled mix: clear any hf_dataset a config might set, else
    # run_sft would pull raw gold from the Hub and ignore the self-distilled rewrites.
    cfg = sft_config_from_args(args, data_dir=mixed_data_dir, hf_dataset=None)
    if args.output_dir is None:
        cfg.output_dir = os.path.join(work_dir, "ckpt")
    if cfg.run_name is None:
        cfg.run_name = tag
    print(f"SDFT: frac={args.distilled_frac} -> {cfg.output_dir}")
    run_sft(cfg, attach_weights=None)


if __name__ == "__main__":
    main()
