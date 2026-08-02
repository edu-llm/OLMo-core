import numpy as np
# one-sided 95% t critical values, df=1..199 (hardcoded table avoids scipy)
def t_ppf95(df):
    # Cornish-Fisher / Abramowitz-Stegun expansion for t quantile from normal z
    z = 1.6448536269514722
    g1 = (z**3 + z)/4.0
    g2 = (5*z**5 + 16*z**3 + 3*z)/96.0
    g3 = (3*z**7 + 19*z**5 + 17*z**3 - 15*z)/384.0
    g4 = (79*z**9 + 776*z**7 + 1482*z**5 - 1920*z**3 - 945*z)/92160.0
    return z + g1/df + g2/df**2 + g3/df**3 + g4/df**4
# verify against known values
for df,ref in [(2,2.920),(4,2.132),(9,1.833),(11,1.796),(19,1.729),(29,1.699)]:
    print(f"# t_0.95,{df} = {t_ppf95(df):.4f} (ref {ref})")
def sizing(sigma_d, effect, delta=3.0, trials=40000, seed=1):
    rng=np.random.default_rng(seed)
    for n in range(3,201):
        tc=t_ppf95(n-1)
        xs=rng.normal(effect,sigma_d,size=(trials,n))
        m=xs.mean(1)
        se=xs.std(1,ddof=1)/np.sqrt(n)
        pw=float(np.mean((m-tc*se>0)&(m>=5.0)))
        dec=float(np.mean(tc*se<delta))
        if pw>=0.80 and dec>=0.80:
            return n,pw,dec
    return None,None,None
print("\n## Required n per runbook 5.8.0 Stage 2")
print("## power = P(L95(d)>0 AND dbar>=5pp) at stated TRUE effect; decidability = P(90%CI halfwidth<3pp)")
print(f"{'sigma_d(pp)':>11s} " + " ".join(f"{'true='+str(e)+'pp':>20s}" for e in [5,6,7,10]))
for s in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,8.0]:
    cells=[]
    for e in [5,6,7,10]:
        n,p,d=sizing(s,e)
        cells.append(f"n={n} (pw{p:.2f}/dc{d:.2f})" if n else "n>200")
    print(f"{s:11.1f} " + " ".join(f"{c:>20s}" for c in cells))
print("\n## Runbook 5.8.0 launch schedule lookup")
sched=[(2.0,4),(2.5,5),(3.0,6),(3.5,8),(4.0,10),(5.0,12)]
for s,lab in [(0.42,"L256 DP2-R1P"),(0.51,"L256 DP2-R1"),(0.83,"L128 DP2-R1"),
              (0.85,"L128 DP2-R1P"),(1.53,"L64 DP2-R1P"),(1.80,"L64 DP2-R1"),
              (2.04,"L40 DP2-R1P"),(2.55,"L40 DP2-R1")]:
    r=next((f"n={n} LAUNCHABLE" for th,n in sched if s<=th), "DO NOT LAUNCH")
    print(f"  sigma_d_comp={s:5.2f}pp ({lab:14s}) -> {r}")
