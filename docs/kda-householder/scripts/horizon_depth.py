"""Depth x R in effective-horizon terms (S5 only, n=5, L=1/2/4)."""

import glob
import json
import math
import os
import re
import statistics as st

CH = 1 / 120
LS = ["40", "64", "128", "256", "512", "1024", "2048"]
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
D = {}
for f in glob.glob("/scratch/users/ericrcwu/kda/probes/results/all_night/*.json"):
    m = re.search(r"-L(\d+)-s(\d+)\.json$", os.path.basename(f))
    if not m:
        continue
    j = json.load(open(f))
    if j["task"] != "s5_words":
        continue
    D[(j["num_householder"], int(m.group(1)), j["seed"])] = j


def hz(a, L):
    return L * (a - CH) / (1 - CH)


def runh(R, lay, s):
    a = D[(R, lay, s)]["accuracy_by_length"]
    hs = [hz(a[L], int(L)) for L in LS if a[L] <= 0.999]
    return st.mean(hs) if hs else float("nan")


print("S5 effective horizon h (tokens) by depth and R;  train_max=40")
print(
    f"{'layers':>6} {'n':>2} {'h(R1)':>7} {'h(R4)':>7} {'delta_h':>8} {'95% CI':>20} {'h(R4)/h(R1)':>11}"
)
rows = []
for lay in (1, 2, 3, 4):
    seeds = sorted(
        {s for (r, ly, s) in D if r == 1 and ly == lay}
        & {s for (r, ly, s) in D if r == 4 and ly == lay}
    )
    if not seeds:
        continue
    h1 = [runh(1, lay, s) for s in seeds]
    h4 = [runh(4, lay, s) for s in seeds]
    d = [b - a for a, b in zip(h1, h4)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else 0.0
    ci = T.get(n - 1, 2.0) * sd / math.sqrt(n) if n > 1 else 0.0
    print(
        f"{lay:>6} {n:>2} {st.mean(h1):7.1f} {st.mean(h4):7.1f} {m:+8.1f}  [{m-ci:+7.1f},{m+ci:+7.1f}] {st.mean(h4)/st.mean(h1):10.2f}x  {'SIG' if abs(m)>ci else 'ns'}"
    )
    rows.append((lay, n, st.mean(h1), st.mean(h4), m, ci))
print()
print("Does DEPTH substitute for R?  h(R=1) as depth grows vs h(R=4) at 1 layer:")
for lay, n, a, b, m, ci in rows:
    print(f"  L={lay}: h(R1)={a:6.1f}   h(R4)={b:6.1f}")
print()
print("If depth fully substituted, h(R1,L=4) would reach h(R4,L=1).")
