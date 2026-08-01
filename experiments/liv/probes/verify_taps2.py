import json, struct, os
import numpy as np
np.set_printoptions(suppress=True)

def load(path):
    f=open(path,"rb"); n=struct.unpack("<Q",f.read(8))[0]
    hdr=json.loads(f.read(n)); base=8+n
    def get(kk):
        m=hdr[kk]; s,e=m["data_offsets"]; f.seek(base+s); b=f.read(e-s)
        if m["dtype"]=="BF16":
            a=np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16
            return a.view(np.float32).reshape(m["shape"]).copy()
        return np.frombuffer(b,dtype=np.float32).reshape(m["shape"]).copy()
    return hdr,get

root="/scratch/users/ericrcwu/liv/ckpt"

print("="*78); print("A. ANOMALY CHECK: 2.6B layer 0 shows argmax@t-2 = 46.5% but medE(t-2)=0.0019")
print("="*78)
hdr,get = load(f"{root}/LFM2-2.6B/model.safetensors")
for li in (0,1,3,4,6):
    w=get(f"model.layers.{li}.conv.conv.weight").reshape(-1,3)
    a=np.abs(w); mx=a.max(1)
    ties=(a==mx[:,None]).sum(1)
    dead=(mx==0)
    print(f"  2.6B L{li}: n_dead(all-zero)={dead.sum():5d}/{len(w)}  n_tied_argmax={np.mean(ties>1):.4f} "
          f" argmax@0 excl-ties={np.mean((a.argmax(1)==0)&(ties==1)):.4f} "
          f" strict |t-2|>|t-1| and >|t| = {np.mean((a[:,0]>a[:,1])&(a[:,0]>a[:,2])):.4f}")
    if dead.sum():
        alive=~dead
        e=w[alive]**2; en=e/e.sum(1,keepdims=True)
        print(f"        ALIVE-ONLY medE = {np.round(np.median(en,0),4)}  argmax@0={np.mean(np.abs(w[alive]).argmax(1)==0):.4f}")

print()
print("="*78); print("B. GATE / DOWNSTREAM-IMPORTANCE WEIGHTING (350M) -- does channel importance change it?")
print("="*78)
hdr,get=load(f"{root}/model.safetensors")
livs=[0,1,3,4,6,7,9,11,13,15]
print("  in_proj shape:", hdr["model.layers.0.conv.in_proj.weight"]["shape"],
      " out_proj:", hdr["model.layers.0.conv.out_proj.weight"]["shape"])
allw=[]; allimp=[]
print(f"  {'lyr':>3} {'unweighted medE':>34} | {'importance-weighted mean E':>28}")
for li in livs:
    w=get(f"model.layers.{li}.conv.conv.weight").reshape(-1,3); allw.append(w)
    ip=get(f"model.layers.{li}.conv.in_proj.weight")   # (3d, d): rows are [B; C; x]
    op=get(f"model.layers.{li}.conv.out_proj.weight")  # (d, d): cols index conv channels
    d=w.shape[0]
    B,C,X = ip[:d], ip[d:2*d], ip[2*d:]
    nB=(B**2).sum(1); nC=(C**2).sum(1); nX=(X**2).sum(1)
    nout=(op**2).sum(0)                                 # how much out_proj READS channel c
    # channel throughput proxy: input-gate * value * output-gate * downstream read
    imp = nB*nX*nC*nout
    imp = imp/imp.sum(); allimp.append(imp)
    e=w**2; en=e/np.clip(e.sum(1,keepdims=True),1e-30,None)
    wm=(en*imp[:,None]).sum(0)
    print(f"  {li:>3}  {str(np.round(np.median(en,0),4)):>32} | {str(np.round(wm,4)):>28}")
W=np.concatenate(allw,0); IMP=np.concatenate([i/len(livs) for i in allimp],0)
e=W**2; en=e/np.clip(e.sum(1,keepdims=True),1e-30,None)
print(f"\n  POOLED unweighted median      = {np.round(np.median(en,0),4)}")
print(f"  POOLED unweighted MEAN        = {np.round(en.mean(0),4)}")
print(f"  POOLED importance-wtd MEAN    = {np.round((en*IMP[:,None]).sum(0)/IMP.sum(),4)}")
# correlation: do high-importance channels use more span?
span=en[:,0]+en[:,1]
q=np.quantile(IMP,[0.5,0.9,0.99])
for lab,mask in [("bottom 50% importance",IMP<=q[0]),("top 10%",IMP>=q[1]),("top 1%",IMP>=q[2])]:
    print(f"    {lab:24s}: mean off-current energy = {span[mask].mean():.4f}  "
          f"medE(t-2)={np.median(en[mask,0]):.4f}  frac normE(t-2)>0.3 = {(en[mask,0]>0.3).mean():.4f}")

print()
print("="*78); print("C. TRUNCATION-SIGNATURE TEST: what profile would a model that WANTS more span show?")
print("="*78)
print("  Simulate: true optimal kernel is geometric r^lag over LAGS 0..14 (k=15).")
print("  Best k=3 approximation under a WHITE input is simply the truncation (proj onto span).")
print("  Report what the truncated k=3 read would look like, and true out-of-window mass.")
print(f"  {'r':>6} {'k3 normE [t-2,t-1,t]':>28} {'E(t-2)/E(t-1)':>14} {'argmax@t-2':>11} {'TRUE mass beyond lag2':>22}")
for r in (0.1,0.2,0.29,0.4,0.5,0.7,0.9,1.0):
    full=np.array([r**L for L in range(15)])          # lag 0..14
    ef=full**2; ef=ef/ef.sum()
    tr=np.array([r**2, r**1, r**0])[::-1][::-1]        # [t-2,t-1,t] = lags [2,1,0]
    tr=np.array([r**2,r**1,1.0]); e3=tr**2; e3=e3/e3.sum()
    print(f"  {r:6.2f} {str(np.round(e3,4)):>28} {e3[0]/e3[1]:14.4f} {'0.0' if r<1 else '~1/3':>11} "
          f" {ef[3:].sum()*100:20.2f}%")
print()
print("  MEASURED LFM2-350M: normE=[0.0143,0.1721,0.7439], ratio=0.083 -> implied r = %.3f"%np.sqrt(0.0833))
print("  MEASURED LFM2-1.2B: ratio=0.1434 -> implied r = %.3f"%np.sqrt(0.1434))
print("  MEASURED LFM2-2.6B: ratio=0.1746 -> implied r = %.3f"%np.sqrt(0.1746))
print()
print("  BOX-FILTER counterexample: model truly wants a uniform average over 15 lags.")
box3=np.ones(3)/3; e=box3**2; e=e/e.sum()
print(f"    truncated to k=3 -> normE={np.round(e,4)}, ratio=1.0, argmax@t-2 ~ 1/3 (ties)")
print("    -> this WOULD trip their decision rule. So the rule is not vacuous for smoothers.")
print()
print("  PER-CHANNEL geometric extrapolation on the real checkpoint:")
for nm,p in [("350M",f"{root}/model.safetensors"),("700M",f"{root}/LFM2-700M/model.safetensors"),
             ("1.2B",f"{root}/LFM2-1.2B/model.safetensors"),("2.6B",f"{root}/LFM2-2.6B/model.safetensors")]:
    h2,g2=load(p)
    ks=sorted([kk for kk in h2 if kk.endswith("conv.conv.weight")],key=lambda kk:int(kk.split(".")[2]))
    Wa=np.concatenate([g2(kk).reshape(-1,3) for kk in ks],0)
    ee=Wa**2; enn=ee/np.clip(ee.sum(1,keepdims=True),1e-30,None)
    rc=np.sqrt(np.clip(enn[:,0]/np.clip(enn[:,1],1e-12,None),0,None))   # per-channel implied ratio
    rc=np.clip(rc,0,0.999)
    oow=rc**6            # implied fraction of energy at lags>=3 if geometric continued
    print(f"    {nm}: implied per-channel r  p50={np.median(rc):.3f} p90={np.percentile(rc,90):.3f} "
          f"p99={np.percentile(rc,99):.3f} | implied out-of-window energy: "
          f"mean={oow.mean()*100:.2f}%  p90={np.percentile(oow,90)*100:.2f}%  "
          f"frac channels >10% OOW = {(oow>0.10).mean():.4f}")
