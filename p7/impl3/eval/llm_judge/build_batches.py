"""Blind the generated tutor turns into judge batches.

Reads a ``test_results``-style JSONL (each row has an ``outputs`` dict mapping a setup/candidate
tag -> tutor reply) and, per row, shuffles the candidates into anonymous ``R1..Rk`` so the judge
can't tell which model produced which. Writes ``judge_batch_*.json`` (fed to the frontier judge)
plus ``judge_key.json`` (the rid -> setup unblinding map, consumed by ``aggregate.py``).

Setups are derived from each row's ``outputs`` keys, so this works for BOTH the POC's fixed 4
setups (raw/sft x SI/noSI) and the Impl-3 sweep (base + impl2 + each (variant, T) checkpoint).

Usage:
    python build_batches.py [SRC.jsonl] [OUT_DIR] [N_BATCHES]
    # defaults: SRC=../test_results_instruct.jsonl  OUT_DIR=.  N_BATCHES=4
"""
import json, os, random, sys

random.seed(7)
SRC = sys.argv[1] if len(sys.argv) > 1 else "../test_results_instruct.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
N_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 4

recs = [json.loads(l) for l in open(SRC) if l.strip()]
items, key = [], {}
for i, r in enumerate(recs):
    did = r.get("dialogue_id")
    stem = f"{did}__{i}" if did is not None else f"item{i}"   # i => unique even if dialogue_id repeats across turns
    setups = list(r["outputs"].keys())                        # blind over whatever candidates this row has
    random.shuffle(setups)                                    # randomize which candidate is which setup
    cands = []
    for j, s in enumerate(setups):
        rid = f"{stem}__R{j + 1}"
        cands.append({"rid": rid, "response": r["outputs"][s]})
        key[rid] = s
    items.append({
        "problem_id": did,
        "problem": r["problem"],
        "final_answer": r.get("answer"),   # numeric answer only (to detect reveal/correctness); NOT the gold tutor turn
        "candidates": cands,
    })

# split into N_BATCHES
batches = [[] for _ in range(N_BATCHES)]
for i, it in enumerate(items):
    batches[i % N_BATCHES].append(it)

for k, b in enumerate(batches):
    json.dump(b, open(os.path.join(OUT, f"judge_batch_{k}.json"), "w"), indent=2, ensure_ascii=False)
json.dump(key, open(os.path.join(OUT, "judge_key.json"), "w"), indent=2)
setups = sorted(set(key.values()))
print("items:", len(items), "| batches:", [len(b) for b in batches],
      "| candidates/item:", len(setups), "| setups:", setups)
