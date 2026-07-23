#!/usr/bin/env python3
"""Finish full MetaMath pack: slim bank, rebuild fade_shuffled, eval, summary."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

pack = Path(r"C:\Users\sunee\Projects\OLMo-core\data\worked_examples_metamath_v0")
rng = random.Random(0)

# 1) Slim bank (drop answer_raw) to free space
bank_path = pack / "bank" / "instances.jsonl"
tmp_bank = pack / "bank" / "instances.slim.jsonl"
n_train = 0
holdout = []
print("slimming bank…")
with bank_path.open(encoding="utf-8") as inp, tmp_bank.open("w", encoding="utf-8") as out:
    for line in inp:
        row = json.loads(line)
        row.pop("answer_raw", None)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["split"] == "holdout":
            holdout.append(row)
        else:
            n_train += 1
bank_path.unlink()
tmp_bank.rename(bank_path)
print(f"bank slimmed; train={n_train} holdout={len(holdout)}")

# 2) Rebuild fade_shuffled from fade_ordered
ordered_path = pack / "arms" / "fade_ordered" / "docs.jsonl"
out_path = pack / "arms" / "fade_shuffled" / "docs.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
by_inst: dict[str, list] = defaultdict(list)
print("loading fade_ordered…")
with ordered_path.open(encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        by_inst[d["instance_id"]].append(d)
print("instances", len(by_inst))
n_shuf = 0
with out_path.open("w", encoding="utf-8") as out:
    for docs in by_inst.values():
        docs = list(docs)
        rng.shuffle(docs)
        for d in docs:
            d = dict(d)
            d["arm"] = "fade_shuffled"
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_shuf += 1
print("wrote fade_shuffled", n_shuf)
del by_inst

# 3) Eval
(pack / "eval").mkdir(exist_ok=True)
with (pack / "eval" / "holdout_bare.jsonl").open("w", encoding="utf-8") as f:
    for inst in holdout:
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


def count(p: Path) -> int:
    return sum(1 for _ in p.open(encoding="utf-8"))


summary = {
    "source": "meta-math/MetaMathQA",
    "types": "ALL",
    "n_train_bank": n_train,
    "n_holdout_eval": len(holdout),
    "n_bare_docs": count(pack / "arms" / "bare" / "docs.jsonl"),
    "n_complete_docs": count(pack / "arms" / "complete" / "docs.jsonl"),
    "n_fade_ordered_docs": count(pack / "arms" / "fade_ordered" / "docs.jsonl"),
    "n_fade_shuffled_docs": n_shuf,
    "note": "Full MetaMathQA (~395k rows); 10% families held out by original_question",
}
(pack / "reports").mkdir(exist_ok=True)
(pack / "reports" / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
