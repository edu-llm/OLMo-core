import math
import random
def mean(v): return sum(v)/len(v)
def sd(v):
    m=mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def tcrit95(df):
    tab={1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,
         10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753,16:1.746,17:1.740,
         18:1.734,19:1.729,20:1.725,25:1.708,30:1.697,40:1.684,50:1.676,60:1.671,
         80:1.664,100:1.660}
    if df in tab:
        return tab[df]
    ks=sorted(tab)
    for k in ks:
        if df<k:
            return tab[k]
    return 1.645
def sizing(sigma_d, effect, delta=3.0, trials=20000, seed=1):
    rnd=random.Random(seed)
    out=[]
    for n in range(3,121):
        tc=tcrit95(n-1)
        pw=0
        dec=0
        for _ in range(trials):
            xs=[rnd.gauss(effect,sigma_d) for _ in range(n)]
            m=mean(xs)
            se=sd(xs)/math.sqrt(n)
            if m-tc*se>0 and m>=5.0:
                pw+=1
            if tc*se<delta:
                dec+=1
        p,d=pw/trials,dec/trials
        out.append((n,p,d))
        if p>=0.80 and d>=0.80:
            return n,p,d
    return None,None,None
print("## Required n (runbook 5.8.0 stage 2: power>=0.80 at TRUE effect for cond.3 [L95>0 AND dbar>=5pp], and decidability>=0.80 [90%CI halfwidth<3pp])")
print(f"{'sigma_d(pp)':>11s} " + " ".join(f"{'eff='+str(e)+'pp':>12s}" for e in [5,7,10]))
for s in [0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0]:
    cells=[]
    for e in [5,7,10]:
        n,p,d=sizing(s,e)
        cells.append(f"n={n} ({p:.2f}/{d:.2f})" if n else "n>120")
    print(f"{s:11.1f} " + " ".join(f"{c:>12s}" for c in cells))
print()
print("## Runbook 5.8.0 launch schedule lookup")
sched=[(2.0,4),(2.5,5),(3.0,6),(3.5,8),(4.0,10),(5.0,12)]
def lookup(sd_):
    for th,n in sched:
        if sd_<=th:
            return f"n={n}, LAUNCHABLE"
    return "DO NOT LAUNCH (sigma_d > 5pp)"
for s in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0]:
    print(f"  sigma_d={s:4.1f}pp -> {lookup(s)}")
