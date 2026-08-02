"""Score a trained arm on Lean-style multi-step Metamath proofs, under four conditions.

The headline number is `facts_present` on `eval_retrieval`: both arms get the same
correct theorem statements in context, and every example cites at least one fact that
was never supervised. That is the question — does the split model match or beat dense
when both can read the facts.

The other three conditions exist because a bare win is ambiguous, and each of them
rules something out:

  facts_present    the real setting. Correct facts, correct order.
  facts_absent     header, no statements. Isolates what each arm stored in weights.
                   Dense should degrade less here if it memorised; split should
                   degrade more. If neither degrades, the facts were never needed and
                   the whole comparison is measuring something else.
  facts_corrupted  names kept, statements swapped between examples. A model that reads
                   its context should collapse. A model that ignores the block and
                   recites from memory will not — which is how you tell the two apart.
  facts_shuffled   block order permuted. Should be a no-op for both arms. If it is not,
                   the model is keying on position rather than content and every other
                   number here needs re-reading.

There is also a direct memorisation probe (--probe): given a fact name alone, state the
fact. It measures the thing the training manipulation is supposed to change, without
routing through proof search.

Generation runs through HuggingFace, so the trained OLMo-core checkpoint is exported
first (`convert_hf.py --olmo-to-hf`). Greedy by default: the comparison is between two
models, and sampling noise is a confound you have to buy with more samples.

    python src/scripts/train/p3_math_split/run_eval.py --model runs/split/hf --arm split \\
        --corpus corpus --split eval_retrieval --out results/split_retrieval.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm_verify import norm, verify_proof  # noqa: E402

HDR = "I know these mathematical statements:"
SEP = "---"
CONDITIONS = ("facts_present", "facts_absent", "facts_corrupted", "facts_shuffled")


def build_prompt(row, condition: str, rng: random.Random, corrupt_pool):
    """Render the prompt for one condition. Everything up to and including `GOAL`.

    The prompt is assembled exactly the way build_corpus.py assembles `text`, so
    `facts_present` reproduces the training-time format character for character.
    """
    facts = dict(row["facts"])

    if condition == "facts_absent":
        facts = {}
    elif condition == "facts_shuffled":
        items = list(facts.items())
        rng.shuffle(items)
        facts = dict(items)
    elif condition == "facts_corrupted":
        # Keep the names, replace each statement with a different fact's statement.
        # The block stays well-formed and plausible; it is just wrong.
        facts = {name: rng.choice(corrupt_pool) for name in facts}

    block = HDR + ("\n" + "\n".join(f"{n} : {s}" for n, s in facts.items()) if facts else "")
    return f"{block}\n{SEP}\nGOAL {row['goal']}\n"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(model, tok, prompts, max_new_tokens, batch_size, do_sample, temperature, device):
    import torch

    outputs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1] :]
            outputs.append(tok.decode(new, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)
    return outputs


def run_probe(model, tok, rows, heldout, args, device):
    """Can the model state a fact given only its name?

    This is the mechanism check. The dense arm was trained to predict fact statements
    from their names; the split arm never was. If the probe does not separate the arms,
    the loss mask did not do what it was supposed to do, and any downstream proof
    difference is coming from somewhere else.

    Reported separately for held-out facts (neither arm saw them supervised — both
    should fail, and it calibrates the floor) and train facts (only dense saw them).
    """
    seen: dict = {}
    for r in rows:
        for name, stmt in r["facts"].items():
            seen.setdefault(name, stmt)

    held = set(heldout)
    names = sorted(seen)
    random.Random(args.seed).shuffle(names)
    names = names[: args.probe_n]

    prompts = [f"{HDR}\n{n} :" for n in names]
    gens = generate(
        model, tok, prompts, args.probe_max_new_tokens, args.batch_size, False, 1.0, device
    )

    out: dict = {
        "train_facts": {"n": 0, "exact": 0},
        "heldout_facts": {"n": 0, "exact": 0},
        "items": [],
    }
    for name, gen in zip(names, gens):
        gold = seen[name]
        got = norm(gen.strip().splitlines()[0]) if gen.strip() else ""
        exact = got == norm(gold)
        bucket = "heldout_facts" if name in held else "train_facts"
        out[bucket]["n"] += 1
        out[bucket]["exact"] += int(exact)
        out["items"].append(
            {"name": name, "heldout": name in held, "gold": gold, "generated": got, "exact": exact}
        )
    for b in ("train_facts", "heldout_facts"):
        n = out[b]["n"]
        out[b]["exact_rate"] = out[b]["exact"] / n if n else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF-format directory for the trained arm")
    ap.add_argument("--arm", required=True, choices=("dense", "split"))
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--split", default="eval_retrieval")
    ap.add_argument("--db", default="data/set.mm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--probe", action="store_true", help="also run the fact-recall probe")
    ap.add_argument("--probe-n", type=int, default=500)
    ap.add_argument("--probe-max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    import torch
    from mm_expand import MM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(os.path.join(args.corpus, f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    heldout = json.load(open(os.path.join(args.corpus, "heldout.json"), encoding="utf-8"))["facts"]

    print(f"parsing {args.db} for verification")
    mm = MM().parse(args.db)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    corrupt_pool = sorted({s for r in rows for s in r["facts"].values()})
    results = {
        "arm": args.arm,
        "split": args.split,
        "model": args.model,
        "greedy": not args.sample,
        "n": len(rows),
        "conditions": {},
    }

    for condition in args.conditions:
        print(f"\n[{args.arm}] {condition}: {len(rows):,} examples")
        rng = random.Random(args.seed)  # reset per condition so arms see identical perturbations
        prompts = [build_prompt(r, condition, rng, corrupt_pool) for r in rows]
        gens = generate(
            model,
            tok,
            prompts,
            args.max_new_tokens,
            args.batch_size,
            args.sample,
            args.temperature,
            device,
        )

        per_example: list = []
        agg: dict = {
            "valid": 0,
            "goal_reached": 0,
            "exact_match": 0,
            "all_grounded": 0,
            "any_unknown": 0,
            "parsed": 0,
        }
        for row, gen in zip(rows, gens):
            v = verify_proof(mm, gen, row["goal"], row["facts"], gold_target=row["target"])
            d = v.as_dict()
            per_example.append(
                {"id": row["id"], "theorem": row["theorem"], "cited": row["cited"], **d}
            )
            agg["valid"] += v.valid
            agg["goal_reached"] += v.goal_reached
            agg["exact_match"] += v.exact_match
            agg["all_grounded"] += v.all_grounded
            agg["any_unknown"] += v.any_unknown
            agg["parsed"] += v.parsed_steps > 0

        n = max(len(rows), 1)
        rates = {f"{k}_rate": val / n for k, val in agg.items()}
        results["conditions"][condition] = {"counts": agg, **rates, "per_example": per_example}
        print(
            f"  valid {rates['valid_rate']:.1%}  goal {rates['goal_reached_rate']:.1%}  "
            f"exact {rates['exact_match_rate']:.1%}  grounded {rates['all_grounded_rate']:.1%}"
            + (f"  UNKNOWN {rates['any_unknown_rate']:.1%}" if agg["any_unknown"] else "")
        )

    if args.probe:
        print(f"\n[{args.arm}] fact-recall probe")
        results["probe"] = run_probe(model, tok, rows, heldout, args, device)
        p = results["probe"]
        print(
            f"  train facts   {p['train_facts']['exact_rate']:.1%} "
            f"({p['train_facts']['n']} probed)"
        )
        print(
            f"  heldout facts {p['heldout_facts']['exact_rate']:.1%} "
            f"({p['heldout_facts']['n']} probed)"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
