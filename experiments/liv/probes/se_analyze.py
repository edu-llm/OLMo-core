import json

d = json.load(open("/scratch/users/ericrcwu/liv/probes/structure_energy_results_local.json"))
rows = d["rows"]
print("layers:", [r["layer"] for r in rows], "n =", len(rows))
print("rank_sigma_x set:", set(r["rank_sigma_x"] for r in rows))
keys = ["lowrank_128", "grouped_4", "lowrank_256", "grouped_2", "random_mask_25pct"]


def vals(k):
    return [r["gates"][t][k] for r in rows for t in ("B_pregate", "C_postgate")]


for k in keys:
    v = vals(k)
    print("%-20s mean=%.4f  min=%.4f  max=%.4f  n=%d" % (k, sum(v) / len(v), min(v), max(v), len(v)))

lo = [r["gates"][t]["grouped_4_perm_spread"][0] for r in rows for t in ("B_pregate", "C_postgate")]
hi = [r["gates"][t]["grouped_4_perm_spread"][1] for r in rows for t in ("B_pregate", "C_postgate")]
print("perm spread mean lo=%.4f hi=%.4f" % (sum(lo) / len(lo), sum(hi) / len(hi)))
print("global min perm=%.4f global max perm=%.4f" % (min(lo), max(hi)))

print()
print("layer  B_r128  B_g4  B_rand  B_permlo B_permhi |  C_r128  C_g4  C_rand")
for r in rows:
    b = r["gates"]["B_pregate"]
    c = r["gates"]["C_postgate"]
    print("%5d  %.3f  %.3f  %.3f   %.3f  %.3f  |  %.3f  %.3f  %.3f" % (
        r["layer"], b["lowrank_128"], b["grouped_4"], b["random_mask_25pct"],
        b["grouped_4_perm_spread"][0], b["grouped_4_perm_spread"][1],
        c["lowrank_128"], c["grouped_4"], c["random_mask_25pct"]))

# how many (layer,gate) cells have grouped > random mask?
g4 = vals("grouped_4")
rm = vals("random_mask_25pct")
wins = sum(1 for a, b_ in zip(g4, rm) if a > b_)
print()
print("grouped_4 > random_mask in %d of %d cells" % (wins, len(g4)))
diffs = [a - b_ for a, b_ in zip(g4, rm)]
print("mean(grouped - random) = %.5f   min=%.5f max=%.5f" % (
    sum(diffs) / len(diffs), min(diffs), max(diffs)))

# param arithmetic check
d_model = 1024
for r in (128, 256, 512):
    fused = d_model * 2 * r + 2 * r * d_model
    sep = 2 * (d_model * r + r * d_model)
    print("r=%d fused=%d sep=%d equal=%s  frac_of_dense=%.4f" % (
        r, fused, sep, fused == sep, fused / (2 * d_model * d_model)))
for g in (2, 4, 8):
    grp = 2 * (d_model * d_model // g)
    print("g=%d grouped=%d frac_of_dense=%.4f" % (g, grp, grp / (2 * d_model * d_model)))
print("dense two gates =", 2 * d_model * d_model)
print("r=128 fused == g=4 grouped ?", d_model * 2 * 128 + 2 * 128 * d_model == 2 * (d_model * d_model // 4))
