"""Aggregate probe results and compute paired kda-vs-gdn contrasts."""
import glob
import json
import statistics as st

R = {}
for f in glob.glob("results/*.json"):
    d = json.load(open(f))
    R[(d["mixer"], d["task"], d["seed"])] = d

LS = ["40", "64", "128", "256", "512"]
print("=== mean accuracy over 3 seeds ===")
print(f'{"task":>9} {"mixer":>5} ' + " ".join(f"{L:>8}" for L in LS))
for t in ["parity", "s5_words"]:
    for m in ["gdn", "kda"]:
        row = []
        for L in LS:
            v = [R[(m, t, s)]["accuracy_by_length"][L] for s in (0, 1, 2) if (m, t, s) in R]
            row.append(f"{100*st.mean(v):7.2f}%" if v else "     --")
        print(f'{t:>9} {m:>5} ' + " ".join(row))

print("\n=== PAIRED by seed: kda - gdn (percentage points), t_crit(df=2)=4.303 ===")
for t in ["parity", "s5_words"]:
    print(f"  {t}:")
    for L in LS:
        try:
            d = [
                100 * (R[("kda", t, s)]["accuracy_by_length"][L]
                       - R[("gdn", t, s)]["accuracy_by_length"][L])
                for s in (0, 1, 2)
            ]
        except KeyError:
            continue
        mean, sd = st.mean(d), st.stdev(d)
        ci = 4.303 * sd / len(d) ** 0.5
        sig = "SIG" if abs(mean) > ci else "ns "
        print(
            f"    len{L:>4}: {mean:+6.2f}pp  95%CI [{mean-ci:+6.2f},{mean+ci:+6.2f}]  "
            f"{sig}  seeds {[round(x, 1) for x in d]}"
        )
