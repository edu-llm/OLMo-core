"""P3: does depth substitute for Householder count on S5? (Theorem 1 route ii)"""

import glob
import json
import re
import statistics as st

LS = ["40", "64", "128", "256", "512"]
by_depth = {}
for f in glob.glob("results/depth*.json"):
    j = json.load(open(f))
    L = int(re.search(r"depth(\d+)", f).group(1))
    by_depth.setdefault(L, []).append(j)

print("P3 — S5 word problem, KDA, R=1, varying DEPTH")
print("Theorem 1: S5 solvable by (i) 1 layer at R>=4, or (iii) 4 layers at R=1.")
print("So depth 4+ should show a clear jump over depth 1-2 if the theory applies here.\n")
print(f'{"layers":>7} {"n":>2} ' + " ".join(f"{L:>8}" for L in LS))
for L in sorted(by_depth):
    js = by_depth[L]
    row = [f"{100*st.mean([x['accuracy_by_length'][k] for x in js]):7.2f}%" for k in LS]
    print(f"{L:>7} {len(js):>2} " + " ".join(row))

print("\nper-seed at len=40 and len=128 (spread check):")
for L in sorted(by_depth):
    v40 = [round(100 * x["accuracy_by_length"]["40"], 1) for x in by_depth[L]]
    v128 = [round(100 * x["accuracy_by_length"]["128"], 1) for x in by_depth[L]]
    print(f"  L={L}: len40 {v40}   len128 {v128}")
