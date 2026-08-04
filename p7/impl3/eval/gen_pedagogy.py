#!/usr/bin/env python
"""Generate Socratic tutor turns for the NEW-task pedagogy judge (the RL's-Razor y-axis).

This is the pedagogy analogue of ``generate_eval.py`` (which does the deterministic OLD-task
probes). Where the POC's 2x2 script compared four fixed setups (raw/sft x SI/noSI), this takes an
ARBITRARY set of candidate models, so every Impl-3 (variant, T) checkpoint can be scored against
the base and the vanilla SFT baseline on the same held-out contexts.

For each held-out dialogue we take the gold history up to a student turn (``context``) and, for
every candidate, greedily generate the next tutor turn. Output rows are
``test_results``-compatible so ``eval/llm_judge/build_batches.py`` can blind them into judge
batches:

    {"dialogue_id", "turn", "problem", "context", "gold_tutor", "answer",
     "outputs": {"<tag>": "<tutor reply>", ...}}

Candidates are given as ``tag=adapter_dir`` (a bare ``tag=`` or ``tag`` with no dir = the plain
base model, no adapter). A candidate that fails to load is skipped with a warning rather than
aborting the whole run (e.g. a missing ``checkpoint-923`` when variant b wasn't shipped).

``--si`` picks the system-instruction condition, which is the 2x2's second factor and the single
biggest driver of judged pedagogy (the POC measured base 0.38 -> 0.71 and SFT 0.52 -> 0.84 from
adding it). ``none`` is the default only for backward compatibility with the generations already
in ``eval/llm_judge/``; ``canonical`` is the condition the POC's headline numbers were measured in
and the one a deployed tutor would actually run with.

Note the held-out rows carry their own per-dialogue SI (that is how they were trained), but eval
deliberately uses ONE fixed SI for every dialogue so the SI is not a per-item confound — same
choice as the POC's cells B and D.

Contexts are deduplicated on (problem, context) and ``--n_dialogues`` counts what survives, so
``n`` is a count of distinct problems. The held-out file holds several dialogues per problem, and
those collapse to the same first-turn prompt; see ``build_contexts``.

Examples:
    # base + vanilla Impl-2 + one Impl-3 checkpoint, no system instruction (cell C)
    python eval/gen_pedagogy.py --test_file data/socrateach_sft_val.jsonl \
        --candidates base= impl2=checkpoint-923 impl3-a-T2=out/impl3-a-T2/checkpoint-923 \
        --out eval/llm_judge/test_results_instruct.jsonl

    # the same models with the canonical pedagogy SI (cell D)
    python eval/gen_pedagogy.py --test_file data/socrateach_sft_val.jsonl --si canonical \
        --candidates base= impl2=checkpoint-923 impl3-a-T2=out/impl3-a-T2/checkpoint-923 \
        --out eval/llm_judge/si/test_results.jsonl
"""
import argparse
import gc
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.modeling import load_for_inference  # noqa: E402
from common.system_instructions import CANONICAL_SI  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--test_file", required=True, help="Held-out JSONL with a 'messages' conversation per line.")
    p.add_argument("--candidates", nargs="+", required=True,
                   help="One or more 'tag=adapter_dir' (bare 'tag=' => plain base model, no adapter).")
    p.add_argument("--out", default="eval/llm_judge/test_results_instruct.jsonl")
    p.add_argument("--si", choices=("none", "canonical"), default="none",
                   help="System-instruction condition. 'none' = the 2x2's cells A/C; "
                        "'canonical' = cells B/D, the POC's headline condition.")
    p.add_argument("--n_dialogues", type=int, default=40,
                   help="Records to score. With --dedupe (the default) this is a count of "
                        "DISTINCT problems, not of rows read from the file.")
    p.add_argument("--no_dedupe", dest="dedupe", action="store_false",
                   help="Keep rows that repeat a (problem, context) already emitted. Only for "
                        "reproducing the pre-2026-08-03 generations, which inflated n this way.")
    p.add_argument("--max_turns", type=int, default=1, help="Tutor turns to generate per dialogue.")
    p.add_argument("--gen_max", type=int, default=256)
    return p.parse_args()


def parse_candidates(specs):
    """['tag=dir', 'base='] -> [('tag', 'dir' or None), ...] preserving order, deduped by tag."""
    out, seen = [], set()
    for spec in specs:
        tag, _, adapter = spec.partition("=")
        tag = tag.strip()
        adapter = adapter.strip() or None
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append((tag, adapter))
    return out


def strip_system(msgs):
    return [m for m in msgs if m["role"] != "system"]


def build_contexts(records, n_dialogues, max_turns, si="none", dedupe=True):
    """One record per (dialogue, tutor-turn): gold history ending on a student turn.

    The row's own per-dialogue SI is always stripped first, then the requested eval SI (if any)
    is prepended, so the condition is identical across dialogues and across candidates.

    ``dedupe`` drops rows whose (problem, context) is byte-identical to one already kept, and
    ``n_dialogues`` then counts KEPT records rather than rows read. The held-out file stores
    several dialogues per problem (``GSM8K_train_1126_0`` .. ``_3``), which for a first tutor
    turn collapse to the same prompt: the first 40 rows are only 12 distinct problems. Judging
    the copies does not add information — greedy decoding gives byte-identical responses, so the
    judge re-scores the same text — it just weights those problems 4-5x and reports n=40 for a
    sample of 12. Matches the dedupe rule in the team's REPLICATE.md.
    """
    prefix = [{"role": "system", "content": CANONICAL_SI}] if si == "canonical" else []
    recs, seen = [], set()
    for row in records:
        if n_dialogues and len(recs) >= n_dialogues:
            break
        conv = strip_system(row["messages"])  # user(problem), assistant, user, ...
        a_pos = [i for i, m in enumerate(conv) if m["role"] == "assistant"][:max_turns]
        for turn_idx, ai in enumerate(a_pos):
            ctx = conv[:ai]  # ends on a student turn
            key = (conv[0]["content"], json.dumps(ctx, sort_keys=True))
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)
            recs.append({
                "dialogue_id": row.get("dialogue_id"),
                "turn": turn_idx,
                "problem": conv[0]["content"],
                "si": si,
                "context": prefix + ctx,
                "gold_tutor": conv[ai]["content"],
                "answer": row.get("answer"),
                "outputs": {},
            })
    return recs


@torch.no_grad()
def generate_turn(model, tokenizer, messages, gen_max):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**enc, max_new_tokens=gen_max, do_sample=False,  # greedy = reproducible
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    with open(args.test_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    recs = build_contexts(records, args.n_dialogues, args.max_turns, args.si, args.dedupe)
    candidates = parse_candidates(args.candidates)
    print(f"{len(recs)} tutor-turn records ({'deduped' if args.dedupe else 'NOT deduped'}) "
          f"x {len(candidates)} candidates ({', '.join(t for t, _ in candidates)})  "
          f"|  SI condition: {args.si}")

    # Load ONE candidate at a time (many small models beat holding N in memory), generate its
    # column across all contexts, then free it before the next.
    done = []
    for tag, adapter in candidates:
        try:
            model, tok, _ = load_for_inference(args.base_model, adapter_dir=adapter, merge=bool(adapter))
        except Exception as e:  # noqa: BLE001 - skip a bad checkpoint, keep the rest
            print(f"[warn] skip candidate '{tag}' (load failed: {type(e).__name__}: {e})")
            continue
        for i, rec in enumerate(recs):
            rec["outputs"][tag] = generate_turn(model, tok, rec["context"], args.gen_max)
            if (i + 1) % 20 == 0:
                print(f"  {tag}: {i + 1}/{len(recs)}")
        free(model)
        done.append(tag)

    if not done:
        print("[warn] no candidates generated — nothing written.")
        return

    # Drop any record that didn't get every surviving candidate (keeps the judge batches square).
    recs = [r for r in recs if all(t in r["outputs"] for t in done)]
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records ({len(done)} candidates: {', '.join(done)}) -> {args.out}")


if __name__ == "__main__":
    main()
