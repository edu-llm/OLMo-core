"""Check shard-name and partition-glob mechanics for the nested .jsonl layout we intend."""

import fnmatch

from edullm_data.contracts import validate_dataset_id
from edullm_data.manifest import check_shard_naming, parse_shard_name

PATHS = [
    "conversations/general/single-call/train-00000.jsonl",
    "conversations/general/single-call/heldout-00000.jsonl",
    "conversations/general/parallel-call/train-00000.jsonl",
    "conversations/general/irrelevance/train-00000.jsonl",
    "conversations/edu/gradebook/train-00000.jsonl",
    "conversations/edu/curriculum/heldout-00000.jsonl",
    "conversations/train.jsonl",  # no shard index — legal?
    "conversations/general/single-call/train-00000-of-00004.jsonl",  # forbidden -of-
]

print("=== parse_shard_name / check_shard_naming ===")
for p in PATHS:
    print(f"  {p}")
    print(f"      parse={parse_shard_name(p)}  violations={check_shard_naming(p) or 'OK'}")

print()
print("=== partition glob matching (fnmatch on basename OR full path) ===")
GLOBS = {"train": "train-*.jsonl", "heldout": "heldout-*.jsonl"}
for name, glob in GLOBS.items():
    matched = [
        p
        for p in PATHS
        if fnmatch.fnmatch(p, glob) or fnmatch.fnmatch(p.rsplit("/", 1)[-1], glob)
    ]
    print(f"  {name:8s} glob={glob!r} matches {len(matched)}:")
    for m in matched:
        print(f"      {m}")

print()
print("=== disjointness (coverage='partition' requires no overlap) ===")
t = {p for p in PATHS if fnmatch.fnmatch(p.rsplit('/', 1)[-1], GLOBS['train'])}
h = {p for p in PATHS if fnmatch.fnmatch(p.rsplit('/', 1)[-1], GLOBS['heldout'])}
print(f"  train={len(t)} heldout={len(h)} overlap={t & h or 'none'}")

print()
print("=== candidate dataset_ids ===")
for n in [
    "sft/tool-call-single-turn",
    "sft/tool-call-general-edu",
    "sft/function-call-abstention-mix",
    "eval/tool-call-bfcl-aligned",
    "eval/tool-call-heldout-apis",
]:
    try:
        print(f"  PASS  {validate_dataset_id(n)}")
    except Exception as e:
        print(f"  FAIL  {n} -> {e}")
