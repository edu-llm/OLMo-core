#!/usr/bin/env python
"""Assemble every eval axis for the Impl-3 sweep into one master_summary.json.

The sweep varies (variant, temperature) rather than walking one run's checkpoints, so a "point"
here is a whole run's final checkpoint, plus the ``base`` and vanilla ``impl2`` reference points.
Joins four sources:

  out/kl_by_checkpoint.json                     -> kl_new (KL(pi_0||pi) on the new task, with SI)
  eval/math_eval/results_<tag>.jsonl            -> math retention (base vs sft, exact match)
  eval/general_eval/ifeval_graded_results_<tag>.json -> IFEval retention (prompt_level_loose)
  eval/llm_judge/judge_summary.json             -> pedagogy quality (8 MRBench dims + OVERALL)

Forgetting is reported against the *base* model measured on the same prompt set, so it is
comparable across runs. Missing axes are left as null rather than faked.

    python eval/build_master_summary.py --out out/master_summary.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kl", default=os.path.join(ROOT, "out/kl_by_checkpoint.json"))
    p.add_argument("--judge", default=os.path.join(HERE, "llm_judge/judge_summary.json"))
    p.add_argument("--out", default=os.path.join(ROOT, "out/master_summary.json"))
    return p.parse_args()


def math_scores():
    """tag -> (base_acc, sft_acc) as fractions. Re-runs the grader per file and reads its detail."""
    d = os.path.join(HERE, "math_eval")
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "results_*.jsonl"))):
        tag = re.sub(r"^results_|\.jsonl$", "", os.path.basename(path))
        # The grader always writes math_logic_graded_nosi.json, so read it right after each run.
        subprocess.run([sys.executable, "grade_math_logic.py", os.path.basename(path)],
                       cwd=d, capture_output=True, text=True)
        graded = os.path.join(d, "math_logic_graded_nosi.json")
        if not os.path.exists(graded):
            continue
        detail = json.load(open(graded))
        acc = {}
        for model in ("base", "sft"):
            rows = [r for r in detail if r["model"] == model]
            acc[model] = sum(r["correct"] for r in rows) / len(rows) if rows else None
        out[tag] = (acc["base"], acc["sft"])
    return out


def ifeval_scores(metric="prompt_level_loose"):
    """tag -> (base, sft) from the per-tag graded JSON the IFEval grader already writes."""
    d = os.path.join(HERE, "general_eval")
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "ifeval_graded_results_*.json"))):
        tag = re.sub(r"^ifeval_graded_results_|\.json$", "", os.path.basename(path))
        g = json.load(open(path))
        scores = g.get("scores", g)
        try:
            out[tag] = (scores["base"][metric], scores["sft"][metric])
        except (KeyError, TypeError):
            out[tag] = (None, None)
    return out


def epoch_of(tag, runs_dir):
    """Epochs actually completed by the checkpoint that was evaluated, or None if unknown.

    The eval stage grades the highest-numbered checkpoint in a run dir, which is NOT necessarily a
    *finished* run: on 2026-07-30 two runs were killed mid-epoch and their partial checkpoints were
    scored as if final, making them look deceptively low-KL. Reading trainer_state.json turns that
    into a visible number instead of a silent one.
    """
    ckpts = glob.glob(os.path.join(runs_dir, tag, "checkpoint-*", "trainer_state.json"))
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    try:
        return json.load(open(latest)).get("epoch")
    except Exception:
        return None


def variant_and_temp(tag):
    m = re.match(r"impl3-([ab])-T([\d.]+)$", tag)
    return (m.group(1), float(m.group(2))) if m else (None, None)


def main():
    args = parse_args()
    kl = json.load(open(args.kl))
    judge = json.load(open(args.judge)) if os.path.exists(args.judge) else {}
    math_by_tag = math_scores()
    ifeval_by_tag = ifeval_scores()

    # Base is identical across runs (same frozen model), so take it from any graded file.
    base_math = next((b for b, _ in math_by_tag.values() if b is not None), None)
    base_ifeval = next((b for b, _ in ifeval_by_tag.values() if b is not None), None)

    def ped(tag):
        j = judge.get(tag)
        return {f"ped_{k}": v for k, v in j.items()} if j else {}

    points = [{
        "point": "base", "variant": None, "temperature": None,
        "kl_new": 0.0, "kl_ped_noSI": 0.0,
        "math_acc": base_math, "math_forget": 0.0 if base_math is not None else None,
        "ifeval": base_ifeval, "ifeval_forget": 0.0 if base_ifeval is not None else None,
        **ped("base"),
    }]

    runs_dir = os.path.dirname(os.path.abspath(args.kl))
    for tag in sorted(kl):
        v, t = variant_and_temp(tag)
        m_base, m_sft = math_by_tag.get(tag, (None, None))
        i_base, i_sft = ifeval_by_tag.get(tag, (None, None))
        points.append({
            "point": tag, "variant": v, "temperature": t,
            "epoch": epoch_of(tag, runs_dir),
            "kl_new": kl[tag].get("kl_new_SI"),
            "kl_ped_noSI": kl[tag].get("kl_ped_noSI"),
            "math_acc": m_sft,
            "math_forget": (m_base - m_sft) if (m_base is not None and m_sft is not None) else None,
            "ifeval": i_sft,
            "ifeval_forget": (i_base - i_sft) if (i_base is not None and i_sft is not None) else None,
            **ped(tag),
        })

    json.dump(points, open(args.out, "w"), indent=2)
    have_ped = sum(1 for p in points if "ped_OVERALL" in p)
    print(f"wrote {args.out}: {len(points)} points ({have_ped} with pedagogy scores)")
    hdr = f'{"point":<16}{"epoch":>7}{"KL":>8}{"math":>8}{"ifeval":>8}{"pedagogy":>10}'
    print(hdr); print("-" * len(hdr))
    for p in points:
        def f(x, w=8, d=3):
            return f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"
        print(f'{p["point"]:<16}{f(p.get("epoch"), 7, 2)}{f(p["kl_new"])}{f(p["math_acc"])}'
              f'{f(p["ifeval"])}{f(p.get("ped_OVERALL"), 10)}')

    partial = [p for p in points if isinstance(p.get("epoch"), (int, float)) and p["epoch"] < 0.99]
    if partial:
        print("\nWARNING: these runs were graded on a checkpoint from an UNFINISHED run — their KL "
              "is low mostly because training stopped early, not because the objective worked:")
        for p in partial:
            print(f'  {p["point"]}  epoch={p["epoch"]:.2f}')
        print("Re-run them before quoting their numbers.")


if __name__ == "__main__":
    main()
