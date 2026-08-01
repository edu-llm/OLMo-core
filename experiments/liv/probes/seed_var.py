"""Measure seed-to-seed loss variance from the KDA runs, to size the LIV pilot.

The pilot's whole job is to detect a difference between arms. Whether a given token budget CAN
detect it depends on the noise floor -- and this repo already has repeated seeds of identical
configs on the same cluster, which is a far better estimate than any rule of thumb.
"""
import glob, json, statistics as st
from collections import defaultdict

runs = defaultdict(list)
for f in glob.glob("/scratch/users/ericrcwu/kda/lm/results/lm/*.json"):
    d = json.load(open(f))
    key = (d["arm"], d["n_params"], d["steps"], d["train_seq_len"])
    runs[key].append(d)

print("SEED-TO-SEED SPREAD IN FINAL LOSS (identical config, different seed)")
print("{:<12}{:>12}{:>7}{:>8}{:>10}{:>10}{:>10}".format(
    "arm", "params", "seeds", "mean", "sd", "sd/mean", "range"))
pooled = []
for (arm, p, steps, seq), ds in sorted(runs.items()):
    if len(ds) < 3:
        continue
    losses = [d["final_train_loss"] for d in ds]
    m, sd = st.mean(losses), st.stdev(losses)
    pooled.append(sd)
    print("{:<12}{:>12,}{:>7}{:>8.4f}{:>10.5f}{:>9.3f}%{:>10.4f}".format(
        arm, p, len(ds), m, sd, 100 * sd / m, max(losses) - min(losses)))

if not pooled:
    print("  (no config had >=3 seeds)")
    raise SystemExit

sd = st.median(pooled)
print()
print("Median within-config SD: {:.5f} nats".format(sd))
print()
print("MINIMUM DETECTABLE DIFFERENCE (two-sided t-test, alpha=0.05, power=0.80)")
print("Paired seeds, so this is the paired-difference case: MDD ~ 2.0 * sd_diff / sqrt(n).")
print("Conservatively sd_diff = sd*sqrt(2) if seeds were independent; paired data order")
print("makes it smaller, so treat these as an upper bound.")
print()
print("{:>8}{:>16}{:>16}".format("n seeds", "MDD (indep)", "MDD (paired)"))
for n in (2, 3, 4, 6, 8):
    mdd_i = 2.0 * sd * (2 ** 0.5) / (n ** 0.5)
    mdd_p = 2.0 * sd / (n ** 0.5)
    print("{:>8}{:>16.4f}{:>16.4f}".format(n, mdd_i, mdd_p))
print()
print("Context for what counts as a big effect:")
print("  * Mamba-2's 4-23%-attention sweep spans 0.06 nats total -- BELOW some of the MDDs above.")
print("  * DeltaProduct's parameter-matched LM contrast is 0.0053 nats -> needs ~n=43.")
print("  * This repo's KDA study: +8.92pp at n=3 collapsed to +2.01pp (ns) at n=8.")
print()
print("So: a pilot can only resolve a LARGE effect. The good news is that P1's proxy predicts")
print("exactly that -- lowrank retains 0.929 of activation-weighted energy vs grouped's 0.130,")
print("which if predictive is a gap far bigger than 0.06 nats. If the pilot sees NOTHING at")
print("n=3, that is itself the informative answer: the proxy does not transfer.")
