#!/usr/bin/env python3
"""
Build a worked-examples pack from meta-math/MetaMathQA for arms 1–4.

Holdout is by original_question family (all augments of a held-out seed stay out of train).

Example:
  python build_from_metamath.py --out-dir ./data/worked_examples_metamath_v0 \\
      --max-train 10000 --max-holdout 1000 --types GSM
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_render import (
    format_bare,
    format_complete,
    format_fade,
    parse_fade_levels,
    scaffold_prefix,
)
from metamath_parse import parse_metamath_row


def type_ok(typ: str, filter_mode: str) -> bool:
    if filter_mode.upper() == "ALL":
        return True
    if filter_mode.upper() == "GSM":
        return typ.upper().startswith("GSM")
    if filter_mode.upper() == "MATH":
        return typ.upper().startswith("MATH")
    raise SystemExit(f"unknown --types {filter_mode}; use GSM, MATH, or ALL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--max-train",
        type=int,
        default=10000,
        help="Max train instances after filtering (0 = all)",
    )
    ap.add_argument(
        "--max-holdout",
        type=int,
        default=1000,
        help="Max holdout instances (0 = all remaining holdout families)",
    )
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.1,
        help="Fraction of families held out (by original_question)",
    )
    ap.add_argument("--types", type=str, default="GSM", help="GSM | MATH | ALL")
    ap.add_argument("--fade-levels", type=str, default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--appearances-per-train-item",
        type=int,
        default=1,
        help="Repeats of each train item in arms 2/3/4 roster (before fade expansion)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fade_levels = parse_fade_levels(args.fade_levels)

    from datasets import load_dataset

    print("Loading meta-math/MetaMathQA …")
    ds = load_dataset("meta-math/MetaMathQA", split="train")

    # Group by original_question family
    by_family: dict[str, list[dict]] = defaultdict(list)
    n_skip_type = 0
    for i, row in enumerate(ds):
        typ = (row.get("type") or "").strip()
        if not type_ok(typ, args.types):
            n_skip_type += 1
            continue
        original = (row.get("original_question") or row.get("query") or "").strip()
        if not original:
            continue
        by_family[original].append({"row": row, "global_i": i})

    families = list(by_family.keys())
    rng.shuffle(families)
    n_holdout_fam = max(1, int(round(args.holdout_frac * len(families))))
    holdout_fams = set(families[:n_holdout_fam])
    train_fams = set(families[n_holdout_fam:])

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

    train_inst: list[dict] = []
    holdout_inst: list[dict] = []
    n_parse_fail = 0

    def consume(fam_set: set[str], split: str, sink: list[dict], cap: int) -> None:
        nonlocal n_parse_fail
        fam_list = [f for f in families if f in fam_set]
        # Stable order after shuffle of families list
        for original in fam_list:
            for j, item in enumerate(by_family[original]):
                if cap and len(sink) >= cap:
                    return
                inst_id = f"{split}_{item['global_i']:06d}_{j}"
                parsed = parse_metamath_row(item["row"], instance_id=inst_id, split=split)
                if parsed is None:
                    n_parse_fail += 1
                    continue
                sink.append(parsed)

    consume(train_fams, "train", train_inst, args.max_train)
    consume(holdout_fams, "holdout", holdout_inst, args.max_holdout)

    with (out / "bank" / "instances.jsonl").open("w", encoding="utf-8") as f:
        for inst in train_inst + holdout_inst:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    meta_fade = {
        "fade_levels": fade_levels,
        "source": "meta-math/MetaMathQA",
        "types": args.types,
        "max_train": args.max_train,
        "max_holdout": args.max_holdout,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "n_families_total": len(families),
        "n_holdout_families": len(holdout_fams),
        "n_train_families": len(train_fams),
        "n_skipped_type": n_skip_type,
        "n_parse_fail": n_parse_fail,
        "note": (
            "family_id = hash(original_question); holdout is by family so "
            "all MetaMath augments of a seed stay together. "
            "fade = fraction of solution steps shown as context."
        ),
    }
    (out / "meta" / "fade_schedule.json").write_text(json.dumps(meta_fade, indent=2), encoding="utf-8")
    (out / "meta" / "splits.json").write_text(
        json.dumps(
            {
                "n_train": len(train_inst),
                "n_holdout": len(holdout_inst),
                "n_parse_fail": n_parse_fail,
                "n_skipped_type": n_skip_type,
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
            doc1 = {
                "arm": "bare",
                "family_id": inst["family_id"],
                "instance_id": inst["instance_id"],
                "type": inst.get("type"),
                "text": format_bare(inst["question"], inst["final_answer"]),
                "final_answer": inst["final_answer"],
            }
            bare_f.write(json.dumps(doc1, ensure_ascii=False) + "\n")
            n_bare += 1

            doc2 = {
                "arm": "complete",
                "family_id": inst["family_id"],
                "instance_id": inst["instance_id"],
                "type": inst.get("type"),
                "text": format_complete(inst["question"], inst["steps"], inst["final_answer"]),
                "final_answer": inst["final_answer"],
            }
            complete_f.write(json.dumps(doc2, ensure_ascii=False) + "\n")
            n_complete += 1

            fade_docs = []
            for frac in fade_levels:
                shown, hidden = scaffold_prefix(inst["steps"], frac)
                fad = format_fade(inst["question"], shown, hidden, inst["final_answer"])
                fade_docs.append(
                    {
                        "arm": "fade",
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "type": inst.get("type"),
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

    with (out / "eval" / "holdout_bare.jsonl").open("w", encoding="utf-8") as f:
        for inst in holdout_inst:
            f.write(
                json.dumps(
                    {
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "type": inst.get("type"),
                        "prompt": f"Problem: {inst['question']}\nAnswer:",
                        "final_answer": inst["final_answer"],
                        "question": inst["question"],
                        "original_question": inst.get("original_question"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "source": "meta-math/MetaMathQA",
        "types": args.types,
        "n_bare_docs": n_bare,
        "n_complete_docs": n_complete,
        "n_fade_ordered_docs": n_ordered,
        "n_fade_shuffled_docs": n_shuffled,
        "n_holdout_eval": len(holdout_inst),
        "n_parse_fail": n_parse_fail,
        "fade_levels": fade_levels,
        "out_dir": str(out),
    }
    (out / "reports" / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Next: python validate_pack.py --pack-dir", out)


if __name__ == "__main__":
    main()
