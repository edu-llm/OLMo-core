"""P3 depth ladder: is 'S5 solvable at 4 layers' in-distribution or extrapolation?"""

import glob
import json
import statistics as st

CH = 1 / 120
print("mixer=kda (NOT kda_hh), n = seeds present, train_range [3,40]")
print(
    f"{'layers':>6} {'n':>2} "
    + "".join(f"{('L'+ln):>9}" for ln in ["40", "64", "128", "256", "512"])
    + "   h(tokens)"
)
for lay in (1, 2, 4, 6):
    fs = sorted(
        glob.glob(f"/scratch/users/ericrcwu/kda/probes/results/depth{lay}-kda-s5_words-s*.json")
    )
    if not fs:
        continue
    recs = [json.load(open(f)) for f in fs]
    cells = []
    hs = []
    for L in ["40", "64", "128", "256", "512"]:
        vs = [r["accuracy_by_length"][L] for r in recs if L in r["accuracy_by_length"]]
        m = st.mean(vs)
        cells.append(m)
        if m <= 0.999:
            hs.append(int(L) * (m - CH) / (1 - CH))
    print(
        f"{lay:>6} {len(recs):>2} "
        + "".join(f"{100*c:8.2f}%" for c in cells)
        + f"   {st.mean(hs):7.1f}"
    )
print()
print("The handoff's '1/2/4/6 layers = 39.0/72.7/97.6/99.0%' matches the L=40 COLUMN,")
print("i.e. IN-DISTRIBUTION accuracy (train_max=40). It is not a length-generalization result.")
