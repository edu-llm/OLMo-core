"""P4 arity ladder (S3/S4/S5 at fixed depth) and P6 modular arithmetic (kda vs gdn)."""
import glob
import json
import math
import statistics as st

T_CRIT = {2: 4.303, 3: 3.182, 4: 2.776, 7: 2.365}
LS = ["40", "64", "128", "256", "512"]

R = {}
for f in glob.glob("results/p46-*.json"):
    j = json.load(open(f))
    R[(j["mixer"], j["task"], j["seed"])] = j

print("P4 — ARITY LADDER: S3/S4/S5, KDA, depth 3, R=1")
print("Theory: one layer with R Householders covers permutations of <= R+1 elements.")
print("At R=1 that is 2, so ALL of S3/S4/S5 exceed it -- difficulty should rise S3<S4<S5.\n")
print(f'{"task":>9} {"n":>2} ' + " ".join(f"{L:>8}" for L in LS))
for t in ["s3_words", "s4_words", "s5_words"]:
    have = [s for s in (0, 1, 2) if ("kda", t, s) in R]
    if not have:
        continue
    row = [
        f"{100*st.mean([R[('kda',t,s)]['accuracy_by_length'][L] for s in have]):7.2f}%"
        for L in LS
    ]
    print(f'{t:>9} {len(have):>2} ' + " ".join(row))

print("\nP6 — MODULAR ARITHMETIC (Z/5): kda vs gdn, depth 3")
print(f'{"mixer":>9} {"n":>2} ' + " ".join(f"{L:>8}" for L in LS))
for m in ["gdn", "kda"]:
    have = [s for s in (0, 1, 2) if (m, "mod_arith", s) in R]
    if not have:
        continue
    row = [
        f"{100*st.mean([R[(m,'mod_arith',s)]['accuracy_by_length'][L] for s in have]):7.2f}%"
        for L in LS
    ]
    print(f'{m:>9} {len(have):>2} ' + " ".join(row))

paired = [s for s in (0, 1, 2) if ("kda", "mod_arith", s) in R and ("gdn", "mod_arith", s) in R]
if len(paired) >= 2:
    print(f"\n  paired kda-gdn on mod_arith (n={len(paired)}):")
    for L in LS:
        d = [
            100 * (R[("kda", "mod_arith", s)]["accuracy_by_length"][L]
                   - R[("gdn", "mod_arith", s)]["accuracy_by_length"][L])
            for s in paired
        ]
        mean, sd = st.mean(d), st.stdev(d)
        ci = T_CRIT.get(len(d) - 1, 2.0) * sd / math.sqrt(len(d))
        print(f"    len{L:>4}: {mean:+6.2f}pp  95%CI [{mean-ci:+6.2f},{mean+ci:+6.2f}]  "
              f"{'SIG' if abs(mean) > ci else 'ns'}")
