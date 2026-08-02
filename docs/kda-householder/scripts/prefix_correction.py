"""Prefix-corrected accuracy: strip the in-distribution first 40 positions.

acc_reported = (40/L)*acc_prefix + ((L-40)/L)*acc_tail
Assume acc_prefix = the model's own accuracy at L=40 (measured). Solve for acc_tail.
"""

import glob
import json
import os
import re
import statistics as st

CH = {"parity": 1 / 2, "s3_words": 1 / 6, "s4_words": 1 / 24, "s5_words": 1 / 120}
D = {}
for f in glob.glob("/scratch/users/ericrcwu/kda/probes/results/all_night/*.json"):
    m = re.search(r"-L(\d+)-s\d+\.json$", os.path.basename(f))
    if not m:
        continue
    j = json.load(open(f))
    D[(j["num_householder"], j["task"], int(m.group(1)), j["seed"])] = j
P = 40
print("Tail-only accuracy (positions 41..L), 3 layers. 'x chance' = tail/chance.")
for task in ["s5_words", "parity", "s3_words", "s4_words"]:
    ch = CH[task]
    print(f"\n{task}  (chance {100*ch:.3f}%)")
    print(f"  {'R':>2} " + "".join(f"{('L'+L):>22}" for L in ["256", "512", "1024", "2048"]))
    for R in (1, 2, 4):
        seeds = sorted({s for (r, t, ly, s) in D if r == R and t == task and ly == 3})
        if not seeds:
            continue
        cells = []
        for L in ["256", "512", "1024", "2048"]:
            tails = []
            for s in seeds:
                a = D[(R, task, 3, s)]["accuracy_by_length"]
                Li = int(L)
                ap = a["40"]
                ar = a[L]
                tail = (ar * Li - ap * P) / (Li - P)
                tails.append(tail)
            m = st.mean(tails)
            cells.append(f"{100*m:8.3f}% ({m/ch:5.2f}x)")
        print(f"  {R:>2} " + "".join(f"{c:>22}" for c in cells))
