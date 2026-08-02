import numpy as np
def t_ppf95(df):
    z=1.6448536269514722
    return z+(z**3+z)/4.0/df+(5*z**5+16*z**3+3*z)/96.0/df**2+(3*z**7+19*z**5+17*z**3-15*z)/384.0/df**3
print("## Decomposition of runbook cond.3 power: L95(d)>0  AND  dbar>=5pp, TRUE effect = +5.0pp")
print(f"{'sigma_d':>8s} {'n':>4s} {'P(L95>0)':>10s} {'P(dbar>=5)':>11s} {'P(both)':>9s}")
rng=np.random.default_rng(7)
for s in [1.0,2.0,3.0,5.0]:
    for n in [5,12,30,100,500]:
        tc=t_ppf95(n-1)
        xs=rng.normal(5.0,s,size=(200000,n))
        m=xs.mean(1)
        se=xs.std(1,ddof=1)/np.sqrt(n)
        a=float(np.mean(m-tc*se>0))
        b=float(np.mean(m>=5.0))
        c=float(np.mean((m-tc*se>0)&(m>=5.0)))
        print(f"{s:8.1f} {n:4d} {a:10.3f} {b:11.3f} {c:9.3f}")
print("\n# => P(dbar>=5) -> 0.5 as n grows when TRUE effect == 5.0. Condition 3's conjunction")
print("#    caps power at ~0.50 at a true effect of exactly +5pp, for ANY n and ANY sigma_d.")
print("#    The runbook's 5.8.0 table claims power 0.95-1.00 'to detect a true +5pp effect")
print("#    under condition 3'. That is unattainable; the table must have scored only L95>0.")
print("\n## n required for cond.3 power>=0.80 vs TRUE effect size (sigma_d = 2.0pp)")
for eff in [5.0,5.5,6.0,6.5,7.0,8.0]:
    got=None
    for n in range(3,201):
        tc=t_ppf95(n-1)
        xs=rng.normal(eff,2.0,size=(40000,n))
        m=xs.mean(1)
        se=xs.std(1,ddof=1)/np.sqrt(n)
        if float(np.mean((m-tc*se>0)&(m>=5.0)))>=0.80:
            got=n
            break
    print(f"  true={eff:4.1f}pp -> n={got}")
