#!/usr/bin/env python
"""Deterministic IFEval grader — the paper's instruction-following metric, no subagents.

Consumes the ``generate_eval.py`` result rows for ``ifeval_prompts.jsonl`` (each row has
``instruction_ids`` + ``kwargs`` and ``outputs: {base, sft}``) and reports the four official
IFEval numbers for base vs sft:

    prompt_level_strict / loose   — every instruction in the prompt satisfied
    inst_level_strict   / loose   — fraction of individual instructions satisfied

Writes ``ifeval_graded_<tag>.json`` with per-prompt detail and the aggregate scores. The
prior-task "retention" number to drop into master_summary is ``sft.prompt_level_loose`` (or
``inst_level_loose`` if you prefer the finer-grained one).

    python grade_ifeval.py results_c16.jsonl
    python grade_ifeval.py results_c16.jsonl --acc_metric inst_level_loose
"""
import argparse
import json
import os

from ifeval_registry import check_loose, check_strict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="JSONL from generate_eval.py on ifeval_prompts.jsonl")
    p.add_argument("--acc_metric", default="prompt_level_loose",
                   choices=["prompt_level_strict", "prompt_level_loose",
                            "inst_level_strict", "inst_level_loose"],
                   help="Which score to echo as the headline 'acc' for master_summary.")
    return p.parse_args()


def score_model(rows, model):
    prompt_strict = prompt_loose = 0
    inst_total = inst_strict = inst_loose = 0
    detail = []
    for r in rows:
        resp = r["outputs"][model]
        ids = r["instruction_ids"]
        kwargs = r.get("kwargs") or [{}] * len(ids)
        per = []
        p_strict = p_loose = True
        for iid, kw in zip(ids, kwargs):
            s = check_strict(iid, resp, kw)
            l = check_loose(iid, resp, kw)
            per.append({"instruction_id": iid, "strict": s, "loose": l})
            inst_total += 1
            inst_strict += int(s)
            inst_loose += int(l)
            p_strict = p_strict and s
            p_loose = p_loose and l
        prompt_strict += int(p_strict)
        prompt_loose += int(p_loose)
        detail.append({"id": r["id"], "prompt_strict": p_strict, "prompt_loose": p_loose,
                       "instructions": per})
    n = len(rows)
    agg = {
        "prompt_level_strict": prompt_strict / n if n else 0.0,
        "prompt_level_loose": prompt_loose / n if n else 0.0,
        "inst_level_strict": inst_strict / inst_total if inst_total else 0.0,
        "inst_level_loose": inst_loose / inst_total if inst_total else 0.0,
    }
    return agg, detail


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    if not rows or "instruction_ids" not in rows[0]:
        raise SystemExit("results rows lack 'instruction_ids' — did you generate on ifeval_prompts.jsonl?")

    base_agg, _ = score_model(rows, "base")
    sft_agg, sft_detail = score_model(rows, "sft")

    print("=" * 62)
    print(f"IFEVAL (deterministic, {len(rows)} prompts)          base      sft")
    print("=" * 62)
    for k in ["prompt_level_strict", "prompt_level_loose", "inst_level_strict", "inst_level_loose"]:
        print(f"{k:<28}{base_agg[k]*100:>8.1f}%{sft_agg[k]*100:>8.1f}%")
    print("-" * 62)
    print(f"headline acc = sft.{args.acc_metric} = {sft_agg[args.acc_metric]:.4f}")

    tag = os.path.splitext(os.path.basename(args.results))[0]
    out = os.path.join(os.path.dirname(os.path.abspath(args.results)), f"ifeval_graded_{tag}.json")
    json.dump({"base": base_agg, "sft": sft_agg, "acc_metric": args.acc_metric,
               "acc": sft_agg[args.acc_metric], "detail": sft_detail},
              open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
