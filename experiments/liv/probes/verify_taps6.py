import json, struct
import numpy as np
def load(path):
    f=open(path,"rb"); n=struct.unpack("<Q",f.read(8))[0]
    hdr=json.loads(f.read(n)); base=8+n
    def get(kk):
        m=hdr[kk]; s,e=m["data_offsets"]; f.seek(base+s); b=f.read(e-s)
        a=np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16
        return a.view(np.float32).reshape(m["shape"]).copy()
    return hdr,get
root="/scratch/users/ericrcwu/liv/ckpt"
print("K. DEAD-CHANNEL AUDIT (all-zero rows -> NaN in normalization; tapread2 clips to 1e-30")
print("   which silently maps them to normE=[0,0,0] and biases the MEDIAN downward)")
tot=0
for nm,p in [("350M",f"{root}/model.safetensors"),("700M",f"{root}/LFM2-700M/model.safetensors"),
             ("1.2B",f"{root}/LFM2-1.2B/model.safetensors"),("2.6B",f"{root}/LFM2-2.6B/model.safetensors")]:
    h,g=load(p)
    ks=sorted([k for k in h if k.endswith("conv.conv.weight")],key=lambda k:int(k.split(".")[2]))
    W=np.concatenate([g(k).reshape(-1,3) for k in ks],0)
    d=(np.abs(W).sum(1)==0)
    print(f"   {nm}: {int(d.sum())} / {len(W)} all-zero channels ({d.mean()*100:.3f}%)")
    if d.sum():
        A=W[~d]; e=A**2; en=e/e.sum(1,keepdims=True)
        print(f"        ALIVE-ONLY median normE = {np.round(np.median(en,0),4)} "
              f"(vs all-channel median incl. dead)")
print()
print("L. SCALING OF THE PURE-LAG-2 (boundary-saturated) SUB-POPULATION in layers 0-1")
print("   A pure lag-2 delay at k=3 is the CANONICAL truncation signature: the channel")
print("   is a delay line pinned at the maximum lag the window allows.")
print(f"   {'ckpt':>5} {'d':>5} {'L0 pure-t-2':>13} {'L1 pure-t-2':>13} {'L0+L1 count':>12}")
for nm,p in [("350M",f"{root}/model.safetensors"),("700M",f"{root}/LFM2-700M/model.safetensors"),
             ("1.2B",f"{root}/LFM2-1.2B/model.safetensors"),("2.6B",f"{root}/LFM2-2.6B/model.safetensors")]:
    h,g=load(p); r=[]
    for l in (0,1):
        w=g(f"model.layers.{l}.conv.conv.weight").reshape(-1,3)
        e=w**2; en=e/np.clip(e.sum(1,keepdims=True),1e-30,None); r.append(en[:,0]>0.8)
    print(f"   {nm:>5} {len(r[0]):>5} {r[0].mean()*100:11.1f}% {r[1].mean()*100:11.1f}% "
          f"{int(r[0].sum()+r[1].sum()):12d}")
