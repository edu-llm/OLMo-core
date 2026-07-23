#!/usr/bin/env python3
"""
Stream full MetaMath 4-arm pack to HF one arm at a time.
Docs written as .jsonl.gz to save disk; tokens as headerless uint32 .npy.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_render import format_bare, format_complete, format_fade, scaffold_prefix

pack = Path(r"C:\Users\sunee\Projects\OLMo-core\data\worked_examples_metamath_v0")
repo_id = "hiyasvyas/worked-examples-metamath-v0"
fade_levels = [1.0, 0.75, 0.5, 0.25, 0.0]
rng = random.Random(0)

from huggingface_hub import HfApi
from transformers import AutoTokenizer

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

# Ensure eval exists
eval_path = pack / "eval" / "holdout_bare.jsonl"
if not eval_path.exists():
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    n_h = 0
    with (pack / "bank" / "instances.jsonl").open(encoding="utf-8") as inp, eval_path.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            inst = json.loads(line)
            if inst["split"] != "holdout":
                continue
            out.write(
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
            n_h += 1
    print("wrote eval", n_h)

for rel in [
    "bank/instances.jsonl",
    "meta/splits.json",
    "meta/fade_schedule.json",
    "eval/holdout_bare.jsonl",
]:
    p = pack / rel
    if p.exists():
        print("upload", rel)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=rel.replace("\\", "/"),
            repo_id=repo_id,
            repo_type="dataset",
        )

tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", trust_remote_code=True)
eos_id = 100257


def iter_train():
    bad = 0
    with (pack / "bank" / "instances.jsonl").open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                inst = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                if bad <= 5:
                    print(f"  skip bad bank line {i}: {e}")
                continue
            if inst.get("split") == "train":
                yield inst
    if bad:
        print(f"  skipped {bad} bad bank lines")


def encode_text(t: str) -> list[int]:
    enc = tok.encode(t, add_special_tokens=True)
    if not enc or enc[-1] != eos_id:
        enc.append(eos_id)
    return enc


def build_arm(arm: str) -> tuple[Path, Path, dict]:
    docs_path = pack / "arms" / arm / "docs.jsonl.gz"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    if docs_path.exists():
        docs_path.unlink()
    shard = pack / "tokenized" / arm / "shard-00000.npy"
    shard.parent.mkdir(parents=True, exist_ok=True)
    if shard.exists():
        shard.unlink()

    n = 0
    n_tokens = 0
    buf: list[int] = []
    FLUSH = 2_000_000

    def flush_buf(fh):
        nonlocal buf, n_tokens
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint32)
        arr.tofile(fh)
        n_tokens += int(arr.size)
        buf = []

    with gzip.open(docs_path, "wt", encoding="utf-8") as out, shard.open("wb") as tok_fh:
        for inst in iter_train():
            if arm == "bare":
                docs = [
                    {
                        "arm": "bare",
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "type": inst.get("type"),
                        "text": format_bare(inst["question"], inst["final_answer"]),
                        "final_answer": inst["final_answer"],
                    }
                ]
            elif arm == "complete":
                docs = [
                    {
                        "arm": "complete",
                        "family_id": inst["family_id"],
                        "instance_id": inst["instance_id"],
                        "type": inst.get("type"),
                        "text": format_complete(inst["question"], inst["steps"], inst["final_answer"]),
                        "final_answer": inst["final_answer"],
                    }
                ]
            else:
                fade_docs = []
                for frac in fade_levels:
                    shown, hidden = scaffold_prefix(inst["steps"], frac)
                    fad = format_fade(inst["question"], shown, hidden, inst["final_answer"])
                    fade_docs.append(
                        {
                            "arm": arm,
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
                if arm == "fade_shuffled":
                    rng.shuffle(fade_docs)
                docs = fade_docs

            for doc in docs:
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                buf.extend(encode_text(doc["text"]))
                n += 1
                if len(buf) >= FLUSH:
                    flush_buf(tok_fh)
            if n and n % 100000 == 0:
                print(f"  {arm}: wrote {n} docs…")
        flush_buf(tok_fh)

    return docs_path, shard, {"n_docs": n, "n_tokens": n_tokens}


# Resume helper: WE_ARMS=fade_shuffled  (comma-separated) skips already-uploaded arms
arms = [a.strip() for a in os.environ.get("WE_ARMS", "bare,complete,fade_ordered,fade_shuffled").split(",") if a.strip()]

for arm in arms:
    for p in [
        pack / "arms" / arm / "docs.jsonl",
        pack / "arms" / arm / "docs.jsonl.gz",
        pack / "tokenized" / arm / "shard-00000.npy",
    ]:
        if p.exists():
            p.unlink()
            print("removed", p)

all_stats = {}
# Prefer prior summary if resuming
prev_summary = pack / "reports" / "build_summary.json"
if prev_summary.exists():
    try:
        all_stats.update(json.loads(prev_summary.read_text(encoding="utf-8")).get("arms") or {})
    except Exception:
        pass

for arm in arms:
    print("===", arm, "===")
    docs_path, shard, stats = build_arm(arm)
    all_stats[arm] = stats
    print("  stats", stats, "docs_mb", round(docs_path.stat().st_size / 1e6, 1), "tok_mb", round(shard.stat().st_size / 1e6, 1))
    api.upload_file(
        path_or_fileobj=str(docs_path),
        path_in_repo=f"arms/{arm}/docs.jsonl.gz",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(shard),
        path_in_repo=f"tokenized/{arm}/shard-00000.npy",
        repo_id=repo_id,
        repo_type="dataset",
    )
    docs_path.unlink()
    shard.unlink()
    print("  uploaded+deleted local", arm)

(pack / "reports").mkdir(exist_ok=True)
summary = {
    "source": "meta-math/MetaMathQA",
    "types": "ALL",
    "n_train": 355688,
    "n_holdout": 39155,
    "arms": all_stats,
    "note": "Full MetaMathQA; docs as jsonl.gz; 10% family holdout",
}
(pack / "reports" / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
api.upload_file(
    path_or_fileobj=str(pack / "reports" / "build_summary.json"),
    path_in_repo="reports/build_summary.json",
    repo_id=repo_id,
    repo_type="dataset",
)

readme = f"""---
license: mit
pretty_name: Worked Examples MetaMathQA FULL (4 CPT arms)
tags:
- metamath
- worked-examples
- faded-scaffolds
---

# Full MetaMathQA worked-examples pack

Source: [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) (**all 395k**, all types).
- Train: **355,688** instances (90% of families)
- Holdout: **39,155** (`eval/holdout_bare.jsonl`)
- Docs: `arms/<arm>/docs.jsonl.gz` (gunzip to use)
- Tokens: `tokenized/<arm>/shard-00000.npy` (dolma2, EOS 100257)

## Arm stats
```
{json.dumps(all_stats, indent=2)}
```
"""
readme_path = pack / "README.md"
readme_path.write_text(readme, encoding="utf-8")
api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)
print("DONE https://huggingface.co/datasets/" + repo_id)
print(json.dumps(all_stats, indent=2))
