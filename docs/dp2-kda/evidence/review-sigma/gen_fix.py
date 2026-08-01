OUT="/scratch/users/ericrcwu/agent-runs/review-sigma"
lines=[]
for b in range(9101,9111):
    tag=f"hard_R1-Pfix_b{b}"
    lines.append(f"{tag}\t--arm R1-P --match-non-embedding 505352 --task s5_words --bundle-id {b} "
                 f"--steps 4000 --d-model 128 --n-layers 2 --eval-lengths 40 64 128 256 --out {OUT}/out/{tag}.json")
open(f"{OUT}/jobs_fix.tsv","w").write("\n".join(lines)+"\n")
print(len(lines))
