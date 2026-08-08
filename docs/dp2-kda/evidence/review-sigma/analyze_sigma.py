"""Compute sigma_t, rho, sigma_d per runbook 5.8.0, plus power/decidability sizing."""
import json, glob, os, sys, math, itertools
from collections import defaultdict

OUT = "/scratch/users/ericrcwu/agent-runs/review-sigma/out"
prefix = sys.argv[1] if len(sys.argv) > 1 else "s4000"

recs = {}
bad = []
for p in sorted(glob.glob(f"{OUT}/{prefix}_*.json")):
    d = json.load(open(p))
    base = os.path.basename(p)[:-5]
    _, arm, b = base.rsplit("_", 2) if base.count("_") >= 2 else (None, None, None)
    arm = base[len(prefix)+1:].rsplit("_b", 1)[0]
    bundle = int(base.rsplit("_b", 1)[1])
    if d.get("outcome") != "completed":
        bad.append((base, d.get("outcome"))); continue
    if d.get("probe_source_revision") in (None, "unknown"):
        bad.append((base, "prov=" + str(d.get("probe_source_revision")))); continue
    recs[(arm, bundle)] = d

print(f"# loaded {len(recs)} completed records; rejected {len(bad)}: {bad[:10]}")
arms = sorted({a for a, _ in recs})
bundles = sorted({b for _, b in recs})
print(f"# arms={arms}")
print(f"# bundles={bundles}")

# ---- integrity checks -------------------------------------------------------
print("\n## INTEGRITY")
for a in arms:
    ds = [recs[(a, b)] for b in bundles if (a, b) in recs]
    if not ds: continue
    d0 = ds[0]
    seeds = {tuple(sorted(d["seeds"].items())) for d in ds}
    banks = {d["eval_bank_sha256"] for d in ds}
    print(f"{a:12s} n={len(ds):2d} nh={d0['num_householder']} regime={d0['beta_regime']} "
          f"nonemb={d0['param_ledger']['non_embedding']} ffn={d0['ffn_dim']} "
          f"rev={d0['probe_source_revision']} steps={d0['steps']} "
          f"distinct_seedsets={len(seeds)} distinct_evalbanks={len(banks)} "
          f"betamax={max(d['beta_stats']['beta_max'] for d in ds):.4f} "
          f"betamean={sum(d['beta_stats']['beta_mean'] for d in ds)/len(ds):.4f} "
          f"collisions={sum(d['eval_collisions'] or 0 for d in ds)} "
          f"medwall={sorted(d['wall_seconds'] for d in ds)[len(ds)//2]:.0f}s")
# cross-arm eval-bank identity within bundle (buys rho)
for b in bundles[:3]:
    bk = {a: recs[(a,b)]["eval_bank_sha256"][:12] for a in arms if (a,b) in recs}
    print(f"  bundle {b} eval banks: {set(bk.values())} (1 == byte-identical across arms)")

def mean(v): return sum(v)/len(v)
def sd(v):
    if len(v) < 2: return float("nan")
    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def corr(x, y):
    if len(x) < 3: return float("nan")
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((a-my)**2 for a in y))
    if sx == 0 or sy == 0: return float("nan")
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/(sx*sy)

LENGTHS = ["40","64","128","256"]
def acc(a, b, L): return 100.0*recs[(a,b)]["accuracy_by_length"][L]

# ---- per-arm mean/sigma_t ---------------------------------------------------
print("\n## PER-ARM MEAN +/- sigma_t (percentage points), s5_words")
print(f"{'arm':12s} " + "  ".join(f"{'L='+L:>18s}" for L in LENGTHS))
for a in arms:
    row = []
    for L in LENGTHS:
        v = [acc(a,b,L) for b in bundles if (a,b) in recs]
        row.append(f"{mean(v):6.2f} +/- {sd(v):5.2f}")
    print(f"{a:12s} " + "  ".join(f"{r:>18s}" for r in row))

# ---- paired differences -----------------------------------------------------
print("\n## PAIRED ARM DIFFERENCES: sigma_t, rho, sigma_d (per runbook 5.8.0)")
print("# sigma_d here = single-task paired SD = sd of per-bundle (A - B).")
print("# runbook two-task composite: sigma_d_comp = sigma_t*sqrt(2(1-rho))*sqrt((1+rho_T)/2)")
print(f"{'contrast':28s} {'L':>4s} {'n':>3s} {'mA':>7s} {'mB':>7s} {'sA':>6s} {'sB':>6s} "
      f"{'rho':>6s} {'d_bar':>7s} {'s_d':>6s} {'sd_1task':>9s} {'sd_comp':>8s} {'L95':>7s}")
def tcrit95(df):  # one-sided 95% t
    tab={1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,
         10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753,20:1.725,30:1.697}
    return tab.get(df, 1.645 + 1.0/max(df,1))
PAIRS = [(x,y) for x,y in itertools.combinations(arms,2)]
results={}
for A,B in PAIRS:
    for L in LENGTHS:
        bs=[b for b in bundles if (A,b) in recs and (B,b) in recs]
        if len(bs)<3: continue
        va=[acc(A,b,L) for b in bs]; vb=[acc(B,b,L) for b in bs]
        dv=[x-y for x,y in zip(va,vb)]
        r=corr(va,vb); s_d=sd(dv); n=len(bs)
        st=math.sqrt((sd(va)**2+sd(vb)**2)/2)
        sd_form = st*math.sqrt(2*(1-r)) if r==r else float("nan")
        sd_comp = sd_form*math.sqrt(0.5) if sd_form==sd_form else float("nan")  # rho_T=0, k=2
        L95 = mean(dv) - tcrit95(n-1)*s_d/math.sqrt(n)
        results[(A,B,L)]=dict(n=n,mA=mean(va),mB=mean(vb),sA=sd(va),sB=sd(vb),rho=r,
                              dbar=mean(dv),s_d=s_d,sd_1task=sd_form,sd_comp=sd_comp,L95=L95,dv=dv)
        print(f"{A+' - '+B:28s} {L:>4s} {n:>3d} {mean(va):7.2f} {mean(vb):7.2f} {sd(va):6.2f} "
              f"{sd(vb):6.2f} {r:6.3f} {mean(dv):7.2f} {s_d:6.2f} {sd_form:9.2f} {sd_comp:8.2f} {L95:7.2f}")

json.dump({f"{k[0]}|{k[1]}|{k[2]}":v for k,v in results.items()},
          open(f"/scratch/users/ericrcwu/agent-runs/review-sigma/{prefix}_pairs.json","w"), indent=1)

# ---- power / decidability sizing -------------------------------------------
print("\n## SIZING: n for power>=0.80 (true +5pp, one-sided 95% t) and decidability>=0.80 (90% CI hw<3pp)")
import random
def sizing(sigma_d, effect=5.0, delta=3.0, trials=4000, seed=0):
    rnd=random.Random(seed)
    for n in range(3,201):
        tc=tcrit95(n-1)
        # two-sided 90% t == one-sided 95% t
        pw=0; dec=0
        for _ in range(trials):
            xs=[rnd.gauss(effect,sigma_d) for _ in range(n)]
            m=mean(xs); s=sd(xs); se=s/math.sqrt(n)
            if m-tc*se>0 and m>=effect*0.0: pass
            if m-tc*se>0 and m>=5.0: pw+=1
            if tc*se<delta: dec+=1
        if pw/trials>=0.80 and dec/trials>=0.80:
            return n, pw/trials, dec/trials
    return None,None,None
