#!/usr/bin/env python
"""Score one or more generate_eval result files and print base-vs-sft accuracy per source.

Reads any number of result files and prints one table, so an A/B across conditions can be read
side by side. Shares the ``math_scoring`` helpers with the per-checkpoint sweep driver, so the
numbers here and in ``sweep_ckpt_eval.py`` are computed identically.

Reports three columns, because accuracy alone is misleading on a tutor-tuned model: overall
accuracy, the rate at which it commits to an answer at all, and accuracy among the answers it
did commit to. A model that refuses and a model that answers wrongly both score zero on the
first, and only the split tells them apart.

    python score_results.py results_ab_*.jsonl
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_scoring import last_boxed, score  # noqa: E402


def committed(resp):
    """Did the model actually commit to a final answer, rather than deflecting?

    A tutor-persona model fails this probe two separable ways: it declines to state an answer at
    all (the Socratic refusal, which its SI explicitly trains), or it states one and is wrong.
    Plain accuracy sums the two and can't tell "forgot how to do arithmetic" from "refuses to say".
    The boxed marker is the cleanest available proxy for committing, since the prompt asks for it.
    """
    return last_boxed(resp or "") is not None


def accuracies(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    per = defaultdict(lambda: defaultdict(list))
    commit = defaultdict(list)
    for r in rows:
        for col in ("base", "sft"):
            resp = r["outputs"][col]
            per[col][r.get("source", "?")].append(score(resp, r))
            commit[col].append((committed(resp), score(resp, r)))
    return rows, per, commit


def main():
    paths = [p for p in sys.argv[1:] if not p.startswith("-")]
    if not paths:
        raise SystemExit("usage: score_results.py results_a.jsonl [results_b.jsonl ...]")

    sources = []
    table = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[skip] missing {p}")
            continue
        rows, per, commit = accuracies(p)
        for col in ("base", "sft"):
            for s in per[col]:
                if s not in sources:
                    sources.append(s)
        for col in ("base", "sft"):
            allv = [v for s in per[col] for v in per[col][s]]
            c = commit[col]
            n_c = sum(1 for did, _ in c if did)
            table.append((os.path.basename(p), col, len(rows),
                          sum(allv) / len(allv) if allv else float("nan"),
                          {s: sum(v) / len(v) for s, v in per[col].items()},
                          n_c / len(c) if c else float("nan"),
                          (sum(ok for did, ok in c if did) / n_c) if n_c else float("nan")))

    sources.sort()
    head = (f"{'results file':<32}{'model':<6}{'n':>4}{'overall':>9}{'boxed%':>9}{'acc|boxed':>11}"
            + "".join(f"{s:>24}" for s in sources))
    print(head)
    print("-" * len(head))
    for name, col, n, overall, bysrc, cr, acc_c in table:
        row = (f"{name:<32}{col:<6}{n:>4}{overall * 100:>8.1f}%{cr * 100:>8.1f}%{acc_c * 100:>10.1f}%")
        row += "".join(f"{bysrc.get(s, float('nan')) * 100:>23.1f}%" for s in sources)
        print(row)

    print("\nboxed%    = committed to a final answer at all (low = Socratic refusal / persona leak)")
    print("acc|boxed = accuracy among those that did commit (low = genuine skill loss)")
    print("Splitting them matters: plain accuracy conflates 'won't answer' with 'answers wrong',")
    print("and the POC saw BOTH fall (boxed 45.7->20.0, acc|boxed 40.6->7.1 from base to c923).")
    print("\nPOC reference, same items, boxed hint ON, 70 items incl. MATH-500:")
    print("  overall base 20.0 -> c923 2.9 | GSM8K 60.0 -> 6.7 | BBH 20.0 -> 0.0 | AIME 6.7 -> 0.0")
    print("  their standalone adapter (= our checkpoint-923) scored 11% in the same 'nosi' arm,")
    print("  so same-recipe runs vary 4x on this probe -- compare runs, not just protocols.")


if __name__ == "__main__":
    main()
