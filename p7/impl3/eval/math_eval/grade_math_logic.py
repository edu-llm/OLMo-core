"""Grade math+logic results: verifiable final-answer accuracy — fully deterministic.

  - int (GSM8K/AIME): parse final integer, exact match.
  - mc  (BBH)       : parse choice letter (A-G), exact match.
MATH-500 (answer_type 'expr', which needed symbolic/LLM verification) has been removed from the
prompt set, so every item is exact-match and no subagent verifier is needed. The legacy 'expr'
/ needs_verify / --with-verify path below is dead code kept only as a safety net (it stays empty).

Usage:
  python grade_math_logic.py [results.jsonl]
Output/verify filenames are tagged from the results filename so multiple runs (e.g. no-SI vs
direct-SI) don't clobber each other:
  math_logic_results.jsonl            -> tag 'nosi'
  math_logic_results_directsi.jsonl   -> tag 'directsi'
"""
import json, re, sys, glob
from collections import defaultdict

# Extraction / equivalence live in math_scoring so the per-checkpoint sweep driver can reuse
# the exact same logic (importing this script would run a whole grading pass as a side effect).
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from math_scoring import check, extract, last_boxed, norm_expr, sympy_eq  # noqa: E402,F401

WITH_VERIFY = "--with-verify" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
RES_PATH = _pos[0] if _pos else "math_logic_results.jsonl"
_m = re.search(r"math_logic_results_(\w+?)(?:_v\d+.*)?\.jsonl", RES_PATH)
TAG = _m.group(1) if _m else "nosi"
GRADED_OUT = f"math_logic_graded_{TAG}.json"
NEEDS_OUT = f"needs_verify_{TAG}.json"
VERIFIER_GLOB = f"verifier_out_{TAG}_*.json"

PROMPTS = {r["id"]: r for r in (json.loads(l) for l in open("math_logic_prompts.jsonl"))}
RES = [json.loads(l) for l in open(RES_PATH)]
print(f"grading {RES_PATH}  (tag={TAG})")

verifier = {}
if WITH_VERIFY:
    for f in glob.glob(VERIFIER_GLOB):
        for r in json.load(open(f)):
            verifier[r["task_id"]] = str(r.get("verdict", "")).lower().startswith(("correct", "true", "yes"))

per_model = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model -> source -> [correct, n]
needs_verify = []
detail = []
for r in RES:
    meta = PROMPTS[r["id"]]
    gold, atype, src = meta["gold"], meta["answer_type"], meta["source"]
    for model in ("base", "sft"):
        resp = r["outputs"].get(model, "")
        cand = extract(resp, atype)
        res = check(atype, gold, cand)
        if res is None:
            tid = f"{r['id']}__{model}"
            if WITH_VERIFY and tid in verifier:
                res = verifier[tid]
            else:
                needs_verify.append({"task_id": tid, "id": r["id"], "model": model,
                                     "question": meta["prompt"], "gold": gold,
                                     "candidate": resp})
                res = False  # provisional until verified
        per_model[model][src][1] += 1
        per_model[model][src][0] += int(bool(res))
        detail.append({"id": r["id"], "model": model, "source": src, "gold": gold,
                       "extracted": cand, "correct": bool(res)})

# MATH-500 (answer_type 'expr') was dropped so every item is deterministic exact-match
# (int / mc) — no LLM verifier needed. Only report sources that actually have rows.
_all_sources = ["GSM8K", "MATH-500", "BBH-logical_deduction", "AIME-2024"]
sources = [s for s in _all_sources if per_model["base"][s][1] > 0]
def _pct(c, n):
    return f"{c}/{n} ({100*c/n:.0f}%)" if n else "0/0 (n/a)"
print("=" * 66)
print("MATH + LOGIC ACCURACY (final-answer, exact match)   base vs SFT")
print("=" * 66)
print(f"{'source':<26}{'base':>12}{'sft':>12}")
for s in sources:
    b, n = per_model["base"][s]
    sc, _ = per_model["sft"][s]
    print(f"{s:<26}{_pct(b, n):>12}{_pct(sc, n):>12}")
tb = sum(per_model["base"][s][0] for s in sources); tn = sum(per_model["base"][s][1] for s in sources)
ts = sum(per_model["sft"][s][0] for s in sources)
print("-" * 50)
print(f"{'OVERALL':<26}{_pct(tb, tn):>12}{_pct(ts, tn):>12}")

json.dump(detail, open(GRADED_OUT, "w"), indent=2, ensure_ascii=False)
if needs_verify and not WITH_VERIFY:
    json.dump(needs_verify, open(NEEDS_OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\n{len(needs_verify)} MATH-500 answers need subagent verification (symbolic equivalence).")
    print(f"-> python build_verify_batches.py {NEEDS_OUT}, spawn verifiers (verifier_out_{TAG}_*.json),")
    print(f"   then: python grade_math_logic.py {RES_PATH} --with-verify")
else:
    print(f"\ngrading complete. wrote {GRADED_OUT}")
