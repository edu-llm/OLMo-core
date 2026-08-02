#!/usr/bin/env python
"""Build the SI-conditioned SFT splits (PRD §2.1–2.5). DATA IS BLANK for now.

This is the *recipe* wired up end-to-end; you supply the raw sources:

  --pedagogy_jsonl : multi-turn Socratic dialogues. One JSON object per line with:
        {"messages": [ {role:user, problem}, {role:assistant, tutor}, {role:user, ...}, ... ],
         "dialogue_id": "...", "problem_id": "..."(optional), "answer": ...(optional)}
     The messages must be SI-FREE and alternating, ending on an assistant turn — a
     per-dialogue System Instruction is generated and prefixed here (§2.2). Group by
     problem_id (falls back to dialogue_id) so no problem leaks across splits (§2.5).

  --general_jsonl  : the base model's own SI-free SFT/instruction mixture (replay).
        {"messages": [...]}  (any system message is stripped; §2.3). English-filtered.
     Optional; omit for a pedagogy-only train split.

Output: <out_dir>/socrateach_sft_{train,val,test}.jsonl  (train = ~75/25 mix by default).

Once the data team / you provide sources, e.g.:
    python prepare_data.py --pedagogy_jsonl raw_pedagogy.jsonl --general_jsonl raw_general.jsonl \
        --out_dir data --max_total 30000 --general_frac 0.25
"""
import argparse
import collections

from common import data as D
from common.data import (assemble_pedagogy_example, co_train_mix, is_english,
                         make_group_splits, normalize_general, write_jsonl)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pedagogy_jsonl", default=None, help="Raw SI-free multi-turn dialogues (see docstring).")
    p.add_argument("--general_jsonl", default=None, help="Raw SI-free general/replay conversations.")
    p.add_argument("--out_dir", default="data")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--max_total", type=int, default=30000, help="Cap on the TRAIN split (pedagogy + general).")
    p.add_argument("--general_frac", type=float, default=0.25, help="Fraction of TRAIN that is SI-free general.")
    return p.parse_args()


def load_pedagogy_groups(path):
    rows = D.load_jsonl(path)
    by_problem = collections.OrderedDict()
    for r in rows:
        msgs = r["messages"]
        did = r.get("dialogue_id")
        pid = r.get("problem_id") or did
        ex = assemble_pedagogy_example(msgs, did, problem_id=r.get("problem_id"),
                                       answer=r.get("answer"), source=r.get("source"))
        by_problem.setdefault(pid, []).append(ex)
    return list(by_problem.values())


def load_general(path, n):
    if not path or n <= 0:
        return []
    out = []
    for r in D.load_jsonl(path):
        msgs = normalize_general(r["messages"])
        if msgs is None:
            continue
        if not is_english(" ".join(m["content"] for m in msgs)):
            continue
        out.append({"messages": msgs, "kind": "general", "source": r.get("source"),
                    "dialogue_id": r.get("id"), "problem_id": None, "answer": None})
        if len(out) >= n:
            break
    return out


def main():
    args = parse_args()
    if not args.pedagogy_jsonl:
        raise SystemExit(
            "No pedagogy source given. Data is blank for now — supply --pedagogy_jsonl "
            "(SI-free multi-turn dialogues; see this script's docstring) and optionally "
            "--general_jsonl. The SI-generation + co-training recipe is already wired up."
        )

    groups = load_pedagogy_groups(args.pedagogy_jsonl)
    print(f"pedagogy: {sum(len(g) for g in groups)} examples across {len(groups)} problems")
    splits = make_group_splits(groups, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    n_general = int(round(args.max_total * args.general_frac))
    general = load_general(args.general_jsonl, n_general)
    print(f"general: {len(general)} SI-free English examples")
    splits["train"] = co_train_mix(splits["train"], general, max_total=args.max_total,
                                    general_frac=args.general_frac, seed=args.seed)

    for name, rows in splits.items():
        path = f"{args.out_dir}/{D.SPLIT_FILES[name]}"
        write_jsonl(path, rows)
        print(f"  wrote {len(rows):>6} -> {path}")
    print("Done.")


if __name__ == "__main__":
    main()
