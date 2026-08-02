#!/usr/bin/env python
"""Impl 4 — self-distillation of SFT targets (PRD §4.2 / SDFT §5.2).

Training on the base model's OWN outputs pulls the fine-tune toward the base
distribution (low KL => less forgetting). This script manufactures those targets:

  --mode rewrite  (target domain): for each pedagogy example, prompt the base model
      (pedagogy SI in context, gold tutor turn as a reference) to rewrite each tutor
      turn in its own words. Optional pedagogy quality-gate keeps the rewrite only if
      it still (a) doesn't reveal the final answer and (b) stays one step / one idea;
      otherwise it falls back to the gold turn. (PRD §4.2 marks gating OPTIONAL — test
      whether gating even helps; no gating matches the base distribution more closely.)

  --mode domains  (general): generate SI-free base outputs for a set of general-domain
      prompts (represents the base distribution across domains; PRD §4).

Input/output are JSONL. Data is blank for now: pass your own ``--in_file``. The gold
train file is untouched; ``train_sdft.py`` mixes gold vs distilled by fraction.

Example:
    python self_distill.py --mode rewrite --in_file ../data/socrateach_sft_train.jsonl \
        --out_file distilled/pedagogy_rewrite.jsonl --quality_gate
    python self_distill.py --mode domains --in_file domain_prompts.jsonl \
        --out_file distilled/general_domains.jsonl
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.modeling import load_for_inference  # noqa: E402

REWRITE_INSTRUCTION = (
    "Rewrite your previous tutoring message in your own words. Keep exactly the same "
    "pedagogical intent and the same single next step, stay to one idea / one question, "
    "and do NOT reveal the final answer. Reference (do not copy verbatim):\n\n{gold}"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["rewrite", "domains"], required=True)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--in_file", required=True, help="JSONL: rewrite -> pedagogy 'messages'; domains -> 'prompt'.")
    p.add_argument("--out_file", required=True)
    p.add_argument("--gen_max_new", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temp for self-distilled targets.")
    p.add_argument("--quality_gate", action="store_true", help="Fall back to gold if a rewrite fails the checks.")
    p.add_argument("--max_examples", type=int, default=0, help="0 = all.")
    return p.parse_args()


@torch.no_grad()
def _gen(model, tokenizer, messages, gen_max, temperature):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    kw = dict(max_new_tokens=gen_max, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _passes_gate(rewrite, gold, answer):
    """Cheap pedagogy hard-constraint gate (PRD §4.2 / §5.2). Replace with the blind
    judge when the eval suite exists."""
    if not rewrite or len(rewrite) > 3 * max(len(gold), 1):
        return False
    if answer is not None and str(answer).strip() and str(answer).strip() in rewrite:
        return False  # leaked the final answer
    if rewrite.count("?") > 2:
        return False  # more than "one question"
    return True


def rewrite_targets(model, tokenizer, rows, args):
    out = []
    n_fallback = 0
    for i, row in enumerate(rows):
        msgs = row["messages"]
        new_msgs, distilled = [], 0
        for j, m in enumerate(msgs):
            if m["role"] != "assistant":
                new_msgs.append(m)
                continue
            context = new_msgs + [{"role": "user", "content": REWRITE_INSTRUCTION.format(gold=m["content"])}]
            rw = _gen(model, tokenizer, context, args.gen_max_new, args.temperature)
            if args.quality_gate and not _passes_gate(rw, m["content"], row.get("answer")):
                rw = m["content"]
                n_fallback += 1
            else:
                distilled += 1
            new_msgs.append({"role": "assistant", "content": rw})
        out.append({**row, "messages": new_msgs, "kind": "pedagogy", "distilled": distilled})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} (fallbacks so far: {n_fallback})")
    print(f"rewrite done: {len(out)} examples, {n_fallback} turn-level fallbacks")
    return out


def domain_targets(model, tokenizer, rows, args):
    out = []
    for i, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("problem")
        reply = _gen(model, tokenizer, [{"role": "user", "content": prompt}], args.gen_max_new, args.temperature)
        out.append({
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}],
            "kind": "general", "source": "self-distill-domains",
            **{k: row[k] for k in ("id", "category") if k in row},
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}")
    return out


def main():
    args = parse_args()
    with open(args.in_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.max_examples:
        rows = rows[: args.max_examples]

    model, tokenizer, _ = load_for_inference(args.base_model)
    fn = rewrite_targets if args.mode == "rewrite" else domain_targets
    out = fn(model, tokenizer, rows, args)

    out_path = pathlib.Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} self-distilled records -> {out_path}")


if __name__ == "__main__":
    main()
