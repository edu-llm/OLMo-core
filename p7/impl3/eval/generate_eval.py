#!/usr/bin/env python
"""Generate base-vs-checkpoint outputs on an eval prompt file.

Produces the ``{"id", "prompt", ..., "outputs": {"base": ..., "sft": ...}}`` result
rows that ``math_eval/score_results.py`` consumes. Run it once per checkpoint
(``--adapter out/<run>/checkpoint-16``); each run compares the KL-reference base against
that one checkpoint (the "sft" column).

This is the one-checkpoint-at-a-time path, used for final checkpoints and for the hint A/B.
To score a whole sweep, use ``sweep_ckpt_eval.py`` instead — it caches the base model's answers
across checkpoints rather than regenerating them every time.

``--boxed_hint`` appends "put your final answer inside \\boxed{}" to each prompt. That is not a
formatting detail: it conflicts with the tutor persona's "never state the final answer
yourself", and a tutor-tuned model deflects rather than answers, so the same checkpoint can
score 21% hinted and 46% bare. Generate both conditions and read them together.

Examples:
    python generate_eval.py --prompts math_eval/math_logic_prompts.jsonl \
        --out math_eval/results_c16.jsonl --adapter ../out/impl3-a-T2/checkpoint-16
    python generate_eval.py --prompts math_eval/math_logic_prompts.jsonl --boxed_hint \
        --out math_eval/results_c16_hint.jsonl --adapter ../out/impl3-a-T2/checkpoint-16
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.modeling import load_for_inference  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct", help="KL reference pi_0.")
    p.add_argument("--adapter", default=None, help="Checkpoint (LoRA adapter dir) = the 'sft' column.")
    p.add_argument("--prompts", required=True, help="JSONL with id+prompt (extra fields passed through).")
    p.add_argument("--out", required=True, help="Where to write the results JSONL.")
    p.add_argument("--n", type=int, default=0, help="Cap #prompts (0 = all).")
    p.add_argument("--gen_max", type=int, default=512)
    p.add_argument("--system", default=None, help="Optional system message prepended to every prompt.")
    p.add_argument("--boxed_hint", action="store_true",
                   help="Append the POC boxed-answer hint (use for math_logic_prompts.jsonl).")
    return p.parse_args()


def boxed_hint(row):
    """The POC's math prompt suffix, kept byte-identical so our numbers stay comparable to theirs."""
    return ("Put ONLY the letter of the correct option inside \\boxed{ }, e.g. \\boxed{C}."
            if row.get("answer_type") == "mc" else "Put your final answer inside \\boxed{ }.")


def load_prompts(path, n):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows[:n] if n else rows


def generate_all(model, tok, device, prompts, system, gen_max, hint=False):
    import torch

    outs = []
    for r in prompts:
        user = r["prompt"] + ("\n\n" + boxed_hint(r) if hint else "")
        conv = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=gen_max, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts, args.n)
    print(f"{len(prompts)} prompts from {args.prompts}")

    base, tok, device = load_for_inference(args.base_model)
    base_out = generate_all(base, tok, device, prompts, args.system, args.gen_max, args.boxed_hint)
    del base

    if args.adapter:
        sft, tok, device = load_for_inference(args.base_model, adapter_dir=args.adapter, merge=True)
        sft_out = generate_all(sft, tok, device, prompts, args.system, args.gen_max, args.boxed_hint)
    else:
        sft_out = base_out  # no checkpoint given -> "sft" mirrors base (base-only baseline)

    with open(args.out, "w", encoding="utf-8") as f:
        for r, b, s in zip(prompts, base_out, sft_out):
            row = dict(r)
            row["outputs"] = {"base": b, "sft": s}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} rows -> {args.out}")


if __name__ == "__main__":
    main()
