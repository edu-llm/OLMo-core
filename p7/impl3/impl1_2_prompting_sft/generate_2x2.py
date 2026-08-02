#!/usr/bin/env python
"""Generate the 2x2 factorial outputs for the Impl 1 & 2 evaluation (PRD §3).

Factorial over {Raw, SFT} x {no-SI, +SI}, identical problems + greedy decoding:

    |            | no System Instruction | + canonical System Instruction |
    | Raw model  | A  (floor/control)    | B  (= Implementation 1)        |
    | SFT model  | C  (should act normal)| D  (= Implementation 2)        |

For each held-out problem we teacher-force the gold student turns and generate the
tutor reply under all four cells. The "+SI" cells use ONE fixed canonical SI
(``common/prompts/canonical_si.txt``), held constant even though training used varied
per-dialogue SIs. Writes ``test_results.jsonl`` for later blind rubric scoring.

Eval scoring itself is out of scope / blank for now — this only produces the model
outputs. Needs a held-out test file with ``messages`` (data is blank; supply yours).
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

SETUPS = [
    ("A_raw_noSI", "raw", False),
    ("B_raw_SI", "raw", True),    # = Implementation 1
    ("C_sft_noSI", "sft", False),
    ("D_sft_SI", "sft", True),    # = Implementation 2
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--adapter_dir", required=True, help="The SFT output dir (LoRA adapter) for cells C/D.")
    p.add_argument("--test_file", required=True, help="Held-out JSONL with a 'messages' conversation per line.")
    p.add_argument("--out", default="test_results.jsonl")
    p.add_argument("--n_eval_dialogues", type=int, default=50)
    p.add_argument("--max_eval_turns", type=int, default=1, help="Tutor turns to generate per dialogue.")
    p.add_argument("--gen_max_new", type=int, default=220)
    return p.parse_args()


def strip_system(msgs):
    return [m for m in msgs if m["role"] != "system"]


@torch.no_grad()
def generate_turn(model, tokenizer, messages, gen_max):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**enc, max_new_tokens=gen_max, do_sample=False,  # greedy = reproducible
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    with open(args.test_file, encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    raw_model, tokenizer, _ = load_for_inference(args.base_model)
    sft_model, _, _ = load_for_inference(args.base_model, adapter_dir=args.adapter_dir)
    models = {"raw": raw_model, "sft": sft_model}

    test_dialogues = test_records[: args.n_eval_dialogues]
    print(f"Generating for {len(test_dialogues)} dialogues x {args.max_eval_turns} turn(s) x 4 cells ...")

    results = []
    for row in test_dialogues:
        conv = strip_system(row["messages"])  # user(problem), assistant, user, ...
        a_pos = [i for i, m in enumerate(conv) if m["role"] == "assistant"][: args.max_eval_turns]
        for turn_idx, ai in enumerate(a_pos):
            context = conv[:ai]  # gold history, ends on a student turn
            rec = {
                "dialogue_id": row.get("dialogue_id"),
                "turn": turn_idx,
                "problem": conv[0]["content"],
                "context": context,
                "gold_tutor": conv[ai]["content"],
                "answer": row.get("answer"),
                "outputs": {},
            }
            for name, which, use_si in SETUPS:
                msgs = ([{"role": "system", "content": CANONICAL_SI}] if use_si else []) + context
                rec["outputs"][name] = generate_turn(models[which], tokenizer, msgs, args.gen_max_new)
            results.append(rec)
        if len(results) % 20 == 0:
            print(f"  {len(results)} turn-records ...")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} records -> {args.out}")

    if results:
        r = results[0]
        print("\n================ SAMPLE ================")
        print("PROBLEM:", r["problem"][:220])
        print("GOLD   :", r["gold_tutor"][:220])
        for name, _, _ in SETUPS:
            print(f"\n----- {name} -----\n{r['outputs'][name][:320]}")

    del raw_model, sft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
