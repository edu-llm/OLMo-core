#!/usr/bin/env python3
"""Rebuild all 4 arms from bank/instances.jsonl (full train split)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_render import format_bare, format_complete, format_fade, scaffold_prefix

pack = Path(r"C:\Users\sunee\Projects\OLMo-core\data\worked_examples_metamath_v0")
fade_levels = [1.0, 0.75, 0.5, 0.25, 0.0]
rng = random.Random(0)

# Free space: remove old arm files first
for arm in ["bare", "complete", "fade_ordered", "fade_shuffled"]:
    p = pack / "arms" / arm / "docs.jsonl"
    if p.exists():
        print("deleting", p)
        p.unlink()

bare_f = (pack / "arms" / "bare" / "docs.jsonl").open("w", encoding="utf-8")
complete_f = (pack / "arms" / "complete" / "docs.jsonl").open("w", encoding="utf-8")
ordered_f = (pack / "arms" / "fade_ordered" / "docs.jsonl").open("w", encoding="utf-8")
shuffled_f = (pack / "arms" / "fade_shuffled" / "docs.jsonl").open("w", encoding="utf-8")

n_bare = n_complete = n_ordered = n_shuffled = 0
n_train = 0

print("rebuilding from bank…")
with (pack / "bank" / "instances.jsonl").open(encoding="utf-8") as f:
    for line in f:
        inst = json.loads(line)
        if inst["split"] != "train":
            continue
        n_train += 1
        if n_train % 50000 == 0:
            print(f"  … {n_train} train instances")

        bare_f.write(
            json.dumps(
                {
                    "arm": "bare",
                    "family_id": inst["family_id"],
                    "instance_id": inst["instance_id"],
                    "type": inst.get("type"),
                    "text": format_bare(inst["question"], inst["final_answer"]),
                    "final_answer": inst["final_answer"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        n_bare += 1

        complete_f.write(
            json.dumps(
                {
                    "arm": "complete",
                    "family_id": inst["family_id"],
                    "instance_id": inst["instance_id"],
                    "type": inst.get("type"),
                    "text": format_complete(inst["question"], inst["steps"], inst["final_answer"]),
                    "final_answer": inst["final_answer"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
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

summary = {
    "source": "meta-math/MetaMathQA",
    "types": "ALL",
    "n_train_bank": n_train,
    "n_bare_docs": n_bare,
    "n_complete_docs": n_complete,
    "n_fade_ordered_docs": n_ordered,
    "n_fade_shuffled_docs": n_shuffled,
}
(pack / "reports").mkdir(exist_ok=True)
(pack / "reports" / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("DONE rebuild arms")
