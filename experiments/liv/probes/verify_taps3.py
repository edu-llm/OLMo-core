import json, struct, os
import numpy as np
np.set_printoptions(suppress=True)
def load(path):
    f=open(path,"rb"); n=struct.unpack("<Q",f.read(8))[0]
    hdr=json.loads(f.read(n)); base=8+n
    def get(kk):
        m=hdr[kk]; s,e=m["data_offsets"]; f.seek(base+s); b=f.read(e-s)
        a=np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16
        return a.view(np.float32).reshape(m["shape"]).copy()
    return hdr,get
root="/scratch/users/ericrcwu/liv/ckpt"
paths={"350M":f"{root}/model.safetensors","700M":f"{root}/LFM2-700M/model.safetensors",
       "1.2B":f"{root}/LFM2-1.2B/model.safetensors","2.6B":f"{root}/LFM2-2.6B/model.safetensors"}
Ws={}
for nm,p in paths.items():
    h,g=load(p); ks=sorted([k for k in h if k.endswith("conv.conv.weight")],key=lambda k:int(k.split(".")[2]))
    Ws[nm]=np.concatenate([g(k).reshape(-1,3) for k in ks],0)

print("="*78); print("D. bf16 TIE ARTIFACT in the boundary-argmax statistic")
print("="*78)
print("   np.argmax returns the FIRST index on ties -> index 0 == OLDEST tap. bf16 has an 8-bit")
print("   mantissa, so exact |w| ties are common. This INFLATES 'boundary-argmax'.")
print(f"   {'ckpt':>5} {'naive argmax@0':>15} {'tie rate':>10} {'STRICT argmax@0':>16} {'inflation':>10}")
for nm,W in Ws.items():
    a=np.abs(W); mx=a.max(1); ties=(a==mx[:,None]).sum(1)
    naive=(a.argmax(1)==0).mean(); strict=((a[:,0]>a[:,1])&(a[:,0]>a[:,2])).mean()
    print(f"   {nm:>5} {naive:15.4f} {ties.mean()-1:10.4f} {strict:16.4f} {naive/max(strict,1e-9):9.1f}x")
print("   -> direction: ties make the boundary look MORE used than it is. Bug is CONSERVATIVE")
print("      for their conclusion (true numbers are even lower). Not a threat to the verdict.")

print()
print("="*78); print("E. EFFECTIVE CONTRIBUTION under CORRELATED inputs (the real objection to w^2)")
print("="*78)
print("   w^2 per tap is a variance decomposition ONLY if the conv input is white.")
print("   LM residual-stream activations are strongly autocorrelated across t.")
print("   Under AR(1) input with corr rho: Var(y)=sum_ij w_i w_j rho^|i-j|.")
print("   LEAVE-ONE-OUT: relative Var drop from deleting the oldest tap. THIS is what")
print("   'zeroing the t-2 tap' would actually cost (experiment C5), to 2nd-order.")
print(f"   {'ckpt':>5} " + " ".join(f"{'rho='+str(r):>12}" for r in (0.0,0.5,0.9,0.99)))
for nm,W in Ws.items():
    row=[]
    for rho in (0.0,0.5,0.9,0.99):
        S=np.array([[rho**abs(i-j) for j in range(3)] for i in range(3)])
        v_full=np.einsum('ci,ij,cj->c',W,S,W)
        W0=W.copy(); W0[:,0]=0
        v_drop=np.einsum('ci,ij,cj->c',W0,S,W0)
        rel=np.clip((v_full-v_drop)/np.clip(v_full,1e-20,None),-10,10)
        row.append(f"{np.median(rel)*100:11.2f}%")
    print(f"   {nm:>5} " + " ".join(row))
print("   (median over channels of the relative variance lost by zeroing the oldest tap)")
print()
W=Ws["350M"]
for rho in (0.0,0.9,0.99):
    S=np.array([[rho**abs(i-j) for j in range(3)] for i in range(3)])
    vf=np.einsum('ci,ij,cj->c',W,S,W); W0=W.copy(); W0[:,0]=0
    vd=np.einsum('ci,ij,cj->c',W0,S,W0)
    rel=(vf-vd)/np.clip(vf,1e-20,None)
    print(f"   350M rho={rho}: median={np.median(rel)*100:.2f}%  mean={rel.mean()*100:.2f}%  "
          f"p90={np.percentile(rel,90)*100:.2f}%  p99={np.percentile(rel,99)*100:.2f}%  "
          f"frac>10%={(rel>0.10).mean():.4f}")

print()
print("="*78); print("F. MEDIAN vs MEAN -- is the headline statistic representative?")
print("="*78)
print(f"   {'ckpt':>5} {'MEDIAN normE(t-2)':>18} {'MEAN normE(t-2)':>16} {'ratio':>7} "
      f"{'MEAN off-current':>17} {'MEDIAN off-current':>19}")
for nm,W in Ws.items():
    e=W**2; en=e/np.clip(e.sum(1,keepdims=True),1e-30,None); oc=en[:,0]+en[:,1]
    print(f"   {nm:>5} {np.median(en[:,0]):18.4f} {en[:,0].mean():16.4f} "
          f"{en[:,0].mean()/max(np.median(en[:,0]),1e-9):6.1f}x {oc.mean():17.4f} {np.median(oc):19.4f}")
print("   -> the distribution is heavily right-skewed; the MEDIAN understates the oldest tap by 4-6x.")
print("      Their headline '1.4%' is a median. The MEAN is ~6%, the pooled RAW energy is 4.26%.")

print()
print("="*78); print("G. THE SPAN-USING SUB-POPULATION -- per-layer counts, not fractions")
print("="*78)
h,g=load(paths["350M"])
livs=[0,1,3,4,6,7,9,11,13,15]
tot20=tot30=0
print(f"   {'lyr':>3} {'n ch normE(t-2)>0.20':>21} {'>0.30':>7} {'>0.50':>7} {'n ch off-current>0.5':>21}")
for li in livs:
    w=g(f"model.layers.{li}.conv.conv.weight").reshape(-1,3)
    e=w**2; en=e/e.sum(1,keepdims=True)
    a=int((en[:,0]>0.2).sum()); b=int((en[:,0]>0.3).sum()); c=int((en[:,0]>0.5).sum())
    d=int(((en[:,0]+en[:,1])>0.5).sum()); tot20+=a; tot30+=b
    print(f"   {li:>3} {a:21d} {b:7d} {c:7d} {d:21d}")
print(f"   TOTAL across 10240 channels: >20% = {tot20} ({tot20/10240*100:.2f}%), "
      f">30% = {tot30} ({tot30/10240*100:.2f}%)")
