#!/usr/bin/env python
"""Implementation 1 — Prompting-only Socratic tutor.

Impl 1 is entirely a system prompt: no training, no orchestration. This harness
loads a model (default OLMo-2-0425-1B-Instruct, but point ``--base_model`` at a
larger open model for the optional cell-B upper-reference), installs the verbatim
Impl-1 system prompt (``common/prompts/impl1_system_prompt.txt``), and either:

  - ``--interactive``: chat with the tutor in your terminal, or
  - ``--problems FILE.jsonl``: batch-generate the first tutor turn for each problem
    (one JSON object per line with a ``problem`` or ``prompt`` field), writing a
    JSONL of ``{problem, tutor}`` for later rubric scoring.

The tutor scaffolds ITSELF via the prompt (hint ladder, one-step-per-turn); we never
inject a worked solution. Evals are out of scope here (blank for now) — this just
produces the Impl-1 behavior/artifacts.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.modeling import load_for_inference  # noqa: E402
from common.system_instructions import IMPL1_SYSTEM_PROMPT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="allenai/OLMo-2-0425-1B-Instruct")
    p.add_argument("--adapter_dir", default=None, help="Optional LoRA adapter (e.g. to prompt the SFT model).")
    p.add_argument("--course", default="math", help="Fills the {course} slot in the system prompt.")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--problems", default=None, help="JSONL with a 'problem'/'prompt' field per line.")
    p.add_argument("--out", default="impl1_prompted_first_turns.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (reproducible).")
    return p.parse_args()


def build_system_prompt(course):
    return IMPL1_SYSTEM_PROMPT.replace("{course}", course)


@torch.no_grad()
def generate(model, tokenizer, messages, max_new_tokens, temperature, device):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    kw = dict(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id,
              pad_token_id=tokenizer.pad_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    system = build_system_prompt(args.course)
    model, tokenizer, device = load_for_inference(args.base_model, adapter_dir=args.adapter_dir)

    if args.interactive:
        print("Impl-1 prompted tutor. Type your problem; Ctrl-C to exit.\n")
        history = [{"role": "system", "content": system}]
        try:
            while True:
                user = input("you> ").strip()
                if not user:
                    continue
                history.append({"role": "user", "content": user})
                reply = generate(model, tokenizer, history, args.max_new_tokens, args.temperature, device)
                history.append({"role": "assistant", "content": reply})
                print(f"tutor> {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
        return

    if not args.problems:
        raise SystemExit("Provide --interactive or --problems FILE.jsonl (data is blank; supply your own).")

    with open(args.problems, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    results = []
    for i, r in enumerate(rows):
        problem = r.get("problem") or r.get("prompt")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": problem}]
        tutor = generate(model, tokenizer, msgs, args.max_new_tokens, args.temperature, device)
        results.append({"problem": problem, "tutor": tutor, **{k: r[k] for k in ("id", "answer") if k in r}})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} prompted first-turns -> {args.out}")


if __name__ == "__main__":
    main()
