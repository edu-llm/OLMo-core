#!/usr/bin/env python
"""Forward-KL over a set of checkpoints -> the KL axis of the KL–forgetting curve.

Implements the PRD KL convention (KL(pi_0 || pi) on held-out pedagogy inputs, both
+SI and no-SI). Point it at the checkpoints from any training run (Impl 2/3/4) and a
held-out pedagogy prompts file; it writes ``kl_by_checkpoint.json``:

    { "<label>": {"kl_new_SI": ..., "kl_ped_noSI": ...}, ... }

Pair these KL values with per-checkpoint pedagogy + math-retention numbers (from the
eval suite, which is out of scope / blank here) to draw the three RL's-Razor plots.

Pedagogy eval prompts are DATA (blank for now) — supply your own held-out file. Each
line: a JSON object with a ``context`` (chat message list ending on a student turn) or
a ``messages`` list; the system message, if any, is dropped (the SI is added per
condition).

Example:
    python run_kl_curve.py --base_model allenai/OLMo-2-0425-1B-Instruct \
        --ckpt c100=out/impl2-sft/checkpoint-100 c200=out/impl2-sft/checkpoint-200 \
        --pedagogy_file heldout_pedagogy.jsonl --out kl_by_checkpoint.json
"""
import argparse
import json

from common.kl import pedagogy_contexts, sweep_checkpoints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct", help="KL reference pi_0.")
    p.add_argument("--ckpt", nargs="+", required=True, help="label=path pairs (one per checkpoint).")
    p.add_argument("--pedagogy_file", required=True, help="Held-out pedagogy prompts JSONL.")
    p.add_argument("--n_prompts", type=int, default=64, help="How many pedagogy prompts to average KL over.")
    p.add_argument("--gen_max", type=int, default=200)
    p.add_argument("--out", default="kl_by_checkpoint.json")
    return p.parse_args()


def load_pedagogy_items(path, n):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = pedagogy_contexts(rows, n)
    print(f"{len(items)} KL contexts (each truncated before its first tutor turn)")
    return items


def main():
    args = parse_args()
    checkpoints = {}
    for pair in args.ckpt:
        if "=" not in pair:
            raise SystemExit(f"--ckpt entries must be label=path, got {pair!r}")
        label, path = pair.split("=", 1)
        checkpoints[label] = path

    items = load_pedagogy_items(args.pedagogy_file, args.n_prompts)
    print(f"KL over {len(items)} held-out pedagogy prompts, {len(checkpoints)} checkpoints")
    sweep_checkpoints(args.base_model, checkpoints, items, gen_max=args.gen_max, out_path=args.out)


if __name__ == "__main__":
    main()
