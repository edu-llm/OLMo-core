"""P1/P2 headline: does increasing Householder count R help state tracking?

P1 (s5_words, non-solvable group) is the discriminating task.
P2 (parity, solvable) is the control -- R should NOT matter there.
"""
import glob
import json
import math
import statistics as st

T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776}
LS = ["40", "64", "128", "256", "512"]

R = {}
for f in glob.glob("results/p12-*.json"):
    j = json.load(open(f))
    R[(j["num_householder"], j["task"], j["seed"])] = j

for task, label in [("s5_words", "P1 - S5 (DISCRIMINATING)"), ("parity", "P2 - parity (CONTROL)")]:
    print(f"\n=== {label} ===")
    print(f'{"R":>2} {"n":>2} ' + " ".join(f"{L:>8}" for L in LS))
    for r in (1, 2, 4):
        have = [s for s in (0, 1, 2) if (r, task, s) in R]
        if not have:
            continue
        row = [
            f"{100*st.mean([R[(r,task,s)]['accuracy_by_length'][L] for s in have]):7.2f}%"
            for L in LS
        ]
        print(f"{r:>2} {len(have):>2} " + " ".join(row))

    # paired contrasts vs R=1
    for r in (2, 4):
        paired = [s for s in (0, 1, 2) if (r, task, s) in R and (1, task, s) in R]
        if len(paired) < 2:
            continue
        print(f"  paired R={r} minus R=1 (n={len(paired)}):")
        for L in LS:
            d = [
                100 * (R[(r, task, s)]["accuracy_by_length"][L]
                       - R[(1, task, s)]["accuracy_by_length"][L])
                for s in paired
            ]
            mean, sd = st.mean(d), st.stdev(d)
            ci = T_CRIT.get(len(d) - 1, 2.0) * sd / math.sqrt(len(d))
            flag = "SIG" if abs(mean) > ci else "ns "
            print(f"    len{L:>4}: {mean:+6.2f}pp  95%CI [{mean-ci:+6.2f},{mean+ci:+6.2f}]  {flag}")
