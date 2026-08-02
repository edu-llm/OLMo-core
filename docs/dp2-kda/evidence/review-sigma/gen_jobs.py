OUT="/scratch/users/ericrcwu/agent-runs/review-sigma"
DP2_NONEMB=1400524
# arm_label, extra flags
ARMS=[
 ("R1",         "--arm R1"),
 ("R1-P",       f"--arm R1-P --match-non-embedding {DP2_NONEMB}"),
 ("DP2-strict", "--arm DP2-strict"),
 ("R1-refl",    "--mixer kda_hh --num-householder 1 --beta-regime reflection"),
 ("Reflection", "--arm Reflection"),
]
BUNDLES=list(range(9101,9113))  # 12 seeds
STEPS=4000
lines=[]
for b in BUNDLES:
    for lab,flags in ARMS:
        tag=f"s{STEPS}_{lab}_b{b}"
        lines.append(f"{tag}\t{flags} --task s5_words --bundle-id {b} --steps {STEPS} "
                     f"--eval-lengths 40 64 128 256 --out {OUT}/out/{tag}.json")
with open(f"{OUT}/jobs_main.tsv","w") as f:
    f.write("\n".join(lines)+"\n")
print(len(lines))
