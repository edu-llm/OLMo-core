#!/usr/bin/env python
"""Snapshot a published SFT dataset from the Hub to local JSONL.

Training reads the dataset straight from the Hub by default (``--hf_dataset
meric533/socrateach-sft``). Use this script only when the GPU node has no internet:
run it on the ORCD login node (which does), then point training at the folder with
``--data_dir data`` (leave ``--hf_dataset`` unset).

It writes ``data/socrateach_sft_{train,val,test}.jsonl`` — the POC filenames the local
loader recognizes — preserving each row verbatim (``messages``, ``kind``, ``problem_id``,
``dialogue_id``, ``answer``, ``source``).

    python snapshot_hf_dataset.py                         # -> data/socrateach_sft_{train,val,test}.jsonl
    python snapshot_hf_dataset.py --hf_dataset other/name --out_dir data_other
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common.data import write_jsonl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_dataset", default="meric533/socrateach-sft")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--prefix", default="socrateach_sft",
                   help="Output filename prefix: <prefix>_{train,val,test}.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    raw = load_dataset(args.hf_dataset)
    print(f"Splits in {args.hf_dataset}: {list(raw)}")
    train_key = next(k for k in ("train", "sft_train") if k in raw)
    val_key = next(k for k in ("validation", "val", "valid", "dev", "sft_val") if k in raw)
    test_key = next((k for k in ("test", "sft_test") if k in raw), None)

    os.makedirs(args.out_dir, exist_ok=True)
    splits = [("train", train_key), ("val", val_key)] + ([("test", test_key)] if test_key else [])
    for split, key in splits:
        rows = [dict(r) for r in raw[key]]
        out = os.path.join(args.out_dir, f"{args.prefix}_{split}.jsonl")
        write_jsonl(out, rows)
        print(f"wrote {len(rows)} rows ({key}) -> {out}")


if __name__ == "__main__":
    main()
