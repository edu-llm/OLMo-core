"""Paired kda-vs-gdn analysis across however many seeds have completed."""
import glob
import json
import math
import statistics as st

T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

R = {}
for f in glob.glob("results/*.json"):
    d = json.load(open(f))
    R[(d["mixer"], d["task"], d["seed"])] = d

seeds = sorted({s for (_, _, s) in R})
LS = ["40", "64", "128", "256", "512"]

print(f"seeds present: {seeds}")
print("\n=== mean accuracy ===")
print(f'{"task":>9} {"mixer":>5} {"n":>2} ' + " ".join(f"{L:>8}" for L in LS))
for t in ["parity", "s5_words"]:
    for m in ["gdn", "kda"]:
        have = [s for s in seeds if (m, t, s) in R]
        row = [
            f"{100*st.mean([R[(m,t,s)]['accuracy_by_length'][L] for s in have]):7.2f}%"
            for L in LS
        ]
        print(f'{t:>9} {m:>5} {len(have):>2} ' + " ".join(row))

print("\n=== PAIRED kda - gdn, percentage points (only seeds where BOTH arms ran) ===")
for t in ["parity", "s5_words"]:
    paired = [s for s in seeds if ("kda", t, s) in R and ("gdn", t, s) in R]
    print(f"  {t}  (n={len(paired)} paired seeds)")
    if len(paired) < 2:
        print("    too few paired seeds")
        continue
    for L in LS:
        d = [
            100 * (R[("kda", t, s)]["accuracy_by_length"][L]
                   - R[("gdn", t, s)]["accuracy_by_length"][L])
            for s in paired
        ]
        n = len(d)
        mean, sd = st.mean(d), st.stdev(d)
        se = sd / math.sqrt(n)
        tc = T_CRIT.get(n - 1, 1.96)
        ci = tc * se
        tstat = mean / se if se > 0 else float("inf")
        verdict = "SIG" if abs(mean) > ci else "ns "
        print(
            f"    len{L:>4}: {mean:+6.2f}pp  95%CI [{mean-ci:+6.2f},{mean+ci:+6.2f}]  "
            f"t={tstat:+5.2f}  {verdict}"
        )
