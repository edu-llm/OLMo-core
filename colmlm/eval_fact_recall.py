"""Post-hoc parametric fact-recall eval for the base/split checkpoints (retrieval disabled).

Reproduces the Co-LMLM fact-recall suite (arXiv:2607.07707, Appendix A.6) on a Hugging Face
checkpoint: **TriviaQA, PopQA, SimpleQA, T-REx (EM), FactScore**. Convert the trained OLMo-core
checkpoint to HF first (SmolLM2 == Llama; use OLMo-core's HF conversion).

base/split have no retrieval, so this measures **parametric** factual knowledge -- the paper's
"w/o KB" setting. Expected outcome: base >> split, confirming the split model externalized facts
instead of memorizing them.

Protocol (paper App. A.6): greedy decoding throughout.
  * TriviaQA / PopQA : EM = any gold alias appears (case-insensitive) within the first 100 chars.
  * T-REx            : EM = reference appears within the first 5 generated content tokens.
  * SimpleQA         : LLM grader (needs OPENAI_API_KEY or GEMINI_API_KEY).
  * FactScore        : atomic-fact verification -- delegated to the official FActScore pipeline.

Each task can be driven from a prepared prompts JSONL (recommended for exact fidelity -- reuse
lil-lab/Co-LMLM's src/lmlm/eval/prepare_*_prompts.py output), or from a built-in dataset loader.

Usage:
    python -m colmlm.eval_fact_recall --model <hf_ckpt> --tasks triviaqa popqa trex \\
        --output results.json [--limit N]
    python -m colmlm.eval_fact_recall --model <hf_ckpt> --tasks simpleqa \\
        --simpleqa-prompts prompts.jsonl --grader openai
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

QA_CHAR_LIMIT = 100  # PopQA/TriviaQA: gold alias within the first 100 output chars
TREX_TOKEN_LIMIT = 5  # T-REx: reference within the first 5 content tokens


def qa_exact_match(output: str, aliases: Sequence[str], char_limit: int = QA_CHAR_LIMIT) -> bool:
    """True if any gold alias appears (case-insensitive) within the first ``char_limit`` chars."""
    window = output[:char_limit].lower()
    return any(a.strip().lower() in window for a in aliases if a.strip())


def trex_exact_match(output: str, answer: str, n_tokens: int = TREX_TOKEN_LIMIT) -> bool:
    """True if the reference appears within the first ``n_tokens`` content tokens of the output."""
    content = re.findall(r"\S+", output)[:n_tokens]
    return answer.strip().lower() in " ".join(content).lower()


def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_items(
    task: str,
    prompts_jsonl: Optional[str],
    limit: Optional[int],
) -> List[dict]:
    """Return a list of ``{"prompt": str, "aliases": [str, ...]}`` items for ``task``.

    Prefers a prepared prompts JSONL (fields ``prompt``/``question`` + ``aliases``/``answers``/
    ``answer``); otherwise falls back to a built-in HF-datasets loader.
    """
    if prompts_jsonl:
        items = []
        for r in _read_jsonl(prompts_jsonl):
            prompt = r.get("prompt") or r.get("question") or r.get("input")
            aliases = r.get("aliases") or r.get("answers") or ([r["answer"]] if "answer" in r else [])
            items.append({"prompt": prompt, "aliases": list(aliases)})
    else:
        items = _builtin_loader(task)
    return items[:limit] if limit else items


def _builtin_loader(task: str) -> List[dict]:
    from datasets import load_dataset  # lazy: only needed for the built-in path

    if task == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        return [
            {"prompt": f"Question: {r['question']}\nAnswer:", "aliases": r["answer"]["aliases"]}
            for r in ds
        ]
    if task == "popqa":
        ds = load_dataset("akariasai/PopQA", split="test")
        # Long-tail subset: fewer than 100 monthly Wikipedia page views.
        rows = [r for r in ds if (r.get("s_pop") or 0) < 100]
        return [
            {"prompt": f"Question: {r['question']}\nAnswer:", "aliases": json.loads(r["possible_answers"])}
            for r in rows
        ]
    raise SystemExit(
        f"No built-in loader for '{task}'. Provide --{task}-prompts <jsonl> "
        "(e.g. from lil-lab/Co-LMLM prepare_*_prompts.py)."
    )


def generate_greedy(
    model_path: str,
    prompts: List[str],
    *,
    max_new_tokens: int = 32,
    batch_size: int = 32,
    repetition_penalty: float = 1.0,
) -> List[str]:
    """Greedy HF generation, returning only the newly generated text per prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

    outputs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=repetition_penalty,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outputs


def run_em_task(
    model_path: str,
    items: List[dict],
    scorer: Callable[[str, Sequence[str]], bool],
    *,
    max_new_tokens: int = 32,
) -> Dict[str, float]:
    prompts = [it["prompt"] for it in items]
    outs = generate_greedy(model_path, prompts, max_new_tokens=max_new_tokens)
    correct = sum(scorer(o, it["aliases"]) for o, it in zip(outs, items))
    return {"n": len(items), "correct": correct, "accuracy": correct / max(len(items), 1)}


def grade_simpleqa(items: List[dict], outputs: List[str], grader: str) -> Dict[str, float]:
    """Grade SimpleQA answers with an LLM (paper uses gpt-4.1). Requires an API key."""
    from colmlm._graders import simpleqa_grade  # thin wrapper; raises if key/pkg missing

    correct = 0
    for it, out in zip(items, outputs):
        if simpleqa_grade(question=it["prompt"], gold=it["aliases"], prediction=out, backend=grader):
            correct += 1
    return {"n": len(items), "correct": correct, "accuracy": correct / max(len(items), 1)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colmlm.eval_fact_recall", description=__doc__)
    p.add_argument("--model", required=True, help="HF checkpoint dir/id (convert OLMo-core ckpt first).")
    p.add_argument(
        "--tasks", nargs="+",
        default=["triviaqa", "popqa", "simpleqa", "trex", "factscore"],
        choices=["triviaqa", "popqa", "simpleqa", "trex", "factscore"],
    )
    p.add_argument("--output", default="fact_recall_results.json")
    p.add_argument("--limit", type=int, default=None, help="Cap examples per task (for smoke runs).")
    for t in ("triviaqa", "popqa", "simpleqa", "trex", "factscore"):
        p.add_argument(f"--{t}-prompts", default=None, help=f"Prepared prompts JSONL for {t}.")
    p.add_argument("--grader", default="openai", choices=["openai", "gemini"], help="SimpleQA grader.")
    p.add_argument("--max-new-tokens", type=int, default=32)
    return p


def main() -> None:
    opts = build_parser().parse_args()
    results: Dict[str, dict] = {}

    for task in opts.tasks:
        prompts_jsonl = getattr(opts, f"{task}_prompts")
        if task in ("triviaqa", "popqa"):
            items = load_items(task, prompts_jsonl, opts.limit)
            results[task] = run_em_task(opts.model, items, qa_exact_match,
                                        max_new_tokens=opts.max_new_tokens)
        elif task == "trex":
            if not prompts_jsonl:
                raise SystemExit("T-REx needs --trex-prompts (LAMA left-to-right subset as JSONL).")
            items = load_items(task, prompts_jsonl, opts.limit)
            results[task] = run_em_task(
                opts.model, items,
                lambda o, al: any(trex_exact_match(o, a) for a in al),
                max_new_tokens=opts.max_new_tokens,
            )
        elif task == "simpleqa":
            if not prompts_jsonl:
                raise SystemExit("SimpleQA needs --simpleqa-prompts and an OPENAI/GEMINI key.")
            items = load_items(task, prompts_jsonl, opts.limit)
            outs = generate_greedy(opts.model, [it["prompt"] for it in items],
                                   max_new_tokens=opts.max_new_tokens)
            results[task] = grade_simpleqa(items, outs, opts.grader)
        elif task == "factscore":
            # FactScore is a full atomic-fact pipeline (biographies, gemini-2.5-flash verifier,
            # 183 labeled entities). Delegate to the official FActScore / lil-lab/Co-LMLM scorer.
            results[task] = {
                "status": "delegated",
                "how": "Generate 256-token biographies (greedy, repetition_penalty=1.2) with this "
                "model, then score with the FActScore pipeline (needs GEMINI/OPENAI). "
                "See lil-lab/Co-LMLM src/lmlm/eval/factscore/.",
            }

    Path(opts.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
