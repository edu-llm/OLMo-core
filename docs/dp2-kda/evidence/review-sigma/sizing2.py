import numpy as np
from scipy import stats
def sizing(sigma_d, effect, delta=3.0, trials=40000, seed=1):
    rng=np.random.default_rng(seed)
    for n in range(3,201):
        tc=stats.t.ppf(0.95,n-1)
        xs=rng.normal(effect,sigma_d,size=(trials,n))
        m=xs.mean(1)
        s=xs.std(1,ddof=1)
        se=s/np.sqrt(n)
        pw=float(np.mean((m-tc*se>0)&(m>=5.0)))
        dec=float(np.mean(tc*se<delta))
        if pw>=0.80 and dec>=0.80:
            return n,pw,dec
    return None,None,None
print("## Required n per runbook 5.8.0 Stage 2")
print("## power = P(L95(d)>0 AND dbar>=5pp) at the stated TRUE effect; decidability = P(90%CI halfwidth<3pp)")
print(f"{'sigma_d(pp)':>11s} " + " ".join(f"{'true='+str(e)+'pp':>18s}" for e in [5,6,7,10]))
for s in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,8.0]:
    cells=[]
    for e in [5,6,7,10]:
        n,p,d=sizing(s,e)
        cells.append(f"n={n} (pw{p:.2f}/dc{d:.2f})" if n else "n>200")
    print(f"{s:11.1f} " + " ".join(f"{c:>18s}" for c in cells))
print()
print("## Runbook 5.8.0 launch schedule lookup (sigma_d -> n, or NO LAUNCH)")
sched=[(2.0,4),(2.5,5),(3.0,6),(3.5,8),(4.0,10),(5.0,12)]
for s in [0.42,0.51,0.83,0.85,1.04,1.53,1.80,2.04,2.55]:
    r=next((f"n={n} LAUNCHABLE" for th,n in sched if s<=th), "DO NOT LAUNCH")
    print(f"  sigma_d={s:5.2f}pp -> {r}")
