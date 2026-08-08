OUT="/scratch/users/ericrcwu/agent-runs/review-sigma"
DP2_NONEMB=1400524
ARMS=[("R1","--arm R1"),
      ("R1-P",f"--arm R1-P --match-non-embedding {DP2_NONEMB}"),
      ("DP2-strict","--arm DP2-strict"),
      ("R1-refl","--mixer kda_hh --num-householder 1 --beta-regime reflection"),
      ("Reflection","--arm Reflection")]
lines=[]
# DIFFICULTY GRID cell 2: smaller model (d_model 128, 2 layers) -> harder, lower accuracy
for b in range(9101,9111):
    for lab,flags in ARMS:
        tag=f"hard_{lab}_b{b}"
        lines.append(f"{tag}\t{flags} --task s5_words --bundle-id {b} --steps 4000 "
                     f"--d-model 128 --n-layers 2 --eval-lengths 40 64 128 256 --out {OUT}/out/{tag}.json")
with open(f"{OUT}/jobs_hard.tsv","w") as f: f.write("\n".join(lines)+"\n")
print(len(lines))
