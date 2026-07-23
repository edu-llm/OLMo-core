#!/usr/bin/env python3
"""
Build a worked-examples pack from openai/gsm8k for arms 1–4.

Example:
  python build_from_gsm8k.py --out-dir ./data/worked_examples_gsm8k_v0 --max-train 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow `python path/to/build_from_gsm8k.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_render import (
    format_bare,
    format_complete,
    format_fade,
    parse_fade_levels,
    scaffold_prefix,
)
from gsm8k_parse import parse_gsm8k_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-train", type=int, default=500, help="Cap train instances for smoke")
    ap.add_argument("--max-test", type=int, default=200, help="Cap holdout instances for smoke")
    ap.add_argument("--fade-levels", type=str, default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--appearances-per-train-item", type=int, default=1,
                    help="How many times each train item appears in arms 2/3/4 roster (before fade expansion)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fade_levels = parse_fade_levels(args.fade_levels)

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    train_rows = list(ds["train"])
    test_rows = list(ds["test"])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    train_rows = train_rows[: args.max_train]
    test_rows = test_rows[: args.max_test]

    out = args.out_dir
    for p in [
        out / "bank",
        out / "meta",
        out / "arms" / "bare",
        out / "arms" / "complete",
        out / "arms" / "fade_ordered",
        out / "arms" / "fade_shuffled",
        out / "eval",
        out / "reports",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    instances = []
    # v0: each GSM8K item is its own family (fine-grained). Grouping can be added later.
    for i, row in enumerate(train_rows):
        fam = f"gsm8k_train_{i:05d}"
        inst = parse_gsm8k_row(row, family_id=fam, instance_id=f"{fam}_a", split="train")
        instances.append(inst)
    for i, row in enumerate(test_rows):
        fam = f"gsm8k_test_{i:05d}"
        inst = parse_gsm8k_row(row, family_id=fam, instance_id=f"{fam}_a", split="holdout")
        instances.append(inst)

    with (out / "bank" / "instances.jsonl").open("w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    train_inst = [x for x in instances if x["split"] == "train" and x["final_answer"] and x["steps"]]
    holdout_inst = [x for x in instances if x["split"] == "holdout" and x["final_answer"]]

    meta_fade = {
        "fade_levels": fade_levels,
        "source": "openai/gsm8k",
        "config": "main",
        "max_train": args.max_train,
        "max_test": args.max_test,
        "seed": args.seed,
        "note": "v0: one GSM8K problem = one family; fade = fraction of solution steps shown as context",
    }
    (out / "meta" / "fade_schedule.json").write_text(json.dumps(meta_fade, indent=2), encoding="utf-8")
    (out / "meta" / "splits.json").write_text(
        json.dumps(
            {
                "n_train": len(train_inst),
                "n_holdout": len(holdout_inst),
                "dropped_train_no_steps_or_answer": args.max_train - len(train_inst),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bare_f = (out / "arms" / "bare" / "docs.jsonl").open("w", encoding="utf-8")
    complete_f = (out / "arms" / "complete" / "docs.jsonl").open("w", encoding="utf-8")
    ordered_f = (out / "arms" / "fade_ordered" / "docs.jsonl").open("w", encoding="utf-8")
    shuffled_f = (out / "arms" / "fade_shuffled" / "docs.jsonl").open("w", encoding="utf-8")

    n_bare = n_complete = n_ordered = n_shuffled = 0

    for inst in train_inst:
        for _ in range(args.appearances_per_train_item):
            # Arm 1
            doc1 = {
                "arm": "bare",
                "family_id": inst["family_id"],
                "instance_id": inst["instance_id"],
                "text": format_bare(inst["question"], inst["final_answer"]),
                "final_answer": inst["final_answer"],
            }
            bare_f.write(json.dumps(doc1, ensure_ascii=False) + "\n")
            n_bare += 1

            # Arm 2
            doc2 = {
                "arm": "complete",
                "family_id": inst["family_id"],
                "instance_id": inst["instance_id"],
                "text": format_complete(inst["question"], inst["steps"], inst["final_answer"]),
                "final_answer": inst["final_answer"],
            }
            complete_f.write(json.dumps(doc2, ensure_ascii=False) + "\n")
            n_complete += 1

            # Arms 3/4: one doc per fade level
            fade_docs = []
            for frac in fade_levels:
                shown, hidden = scaffold_prefix(inst["steps"], frac)
                fad = format_fade(inst["question"], shown, hidden, inst["final_answer"])
                fade_docs.append(
                    {
                        "arm": "fade",
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "fade_frac": frac,
                        "text": fad["text"],
                        "context": fad["context"],
                        "target": fad["target"],
                        "loss_start_char": fad["loss_start_char"],
                        "n_shown": fad["n_shown"],
                        "n_hidden": fad["n_hidden"],
                        "final_answer": inst["final_answer"],
                    }
                )

            for d in fade_docs:
                d = dict(d)
                d["arm"] = "fade_ordered"
                ordered_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_ordered += 1

            shuffled = list(fade_docs)
            rng.shuffle(shuffled)
            for d in shuffled:
                d = dict(d)
                d["arm"] = "fade_shuffled"
                shuffled_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_shuffled += 1

    bare_f.close()
    complete_f.close()
    ordered_f.close()
    shuffled_f.close()

    # Holdout eval prompts (bare only)
    with (out / "eval" / "holdout_bare.jsonl").open("w", encoding="utf-8") as f:
        for inst in holdout_inst:
            f.write(
                json.dumps(
                    {
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "prompt": f"Problem: {inst['question']}\nAnswer:",
                        "final_answer": inst["final_answer"],
                        "question": inst["question"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "n_bare_docs": n_bare,
        "n_complete_docs": n_complete,
        "n_fade_ordered_docs": n_ordered,
        "n_fade_shuffled_docs": n_shuffled,
        "n_holdout_eval": len(holdout_inst),
        "fade_levels": fade_levels,
        "out_dir": str(out),
    }
    (out / "reports" / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Next: python validate_pack.py --pack-dir", out)


if __name__ == "__main__":
    main()
