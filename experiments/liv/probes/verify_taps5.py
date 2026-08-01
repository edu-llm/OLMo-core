import json, struct
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
print("="*78)
print("I. THE BOUNDARY-SATURATED SUB-POPULATION -- pure lag-2 delay channels")
print("="*78)
print("   Hypothesis: layers 0-1 are delay lines. If a MINORITY of their channels are pure")
print("   lag-2 (not lag-1) delays, those channels ARE boundary-saturated -> they are exactly")
print("   the channels a k=5 model could extend to lag 3/4. Count them.")
for nm,p,livs in [("350M",f"{root}/model.safetensors",[0,1,3,4]),
                  ("700M",f"{root}/LFM2-700M/model.safetensors",[0,1,3,4]),
                  ("1.2B",f"{root}/LFM2-1.2B/model.safetensors",[0,1,3,4]),
                  ("2.6B",f"{root}/LFM2-2.6B/model.safetensors",[0,1,3,4])]:
    h,g=load(p)
    print(f"\n   --- {nm} ---")
    print(f"   {'lyr':>3} {'n_ch':>5} {'pure t-1 delay':>15} {'pure t-2 delay':>15} {'pure t passthru':>16} {'mixed':>7}")
    for l in livs:
        w=g(f"model.layers.{l}.conv.conv.weight").reshape(-1,3)
        e=w**2; en=e/e.sum(1,keepdims=True)
        d1=(en[:,1]>0.8); d2=(en[:,0]>0.8); pt=(en[:,2]>0.8)
        mx=~(d1|d2|pt)
        print(f"   {l:>3} {len(w):>5} {d1.sum():8d} ({d1.mean()*100:4.1f}%) {d2.sum():8d} ({d2.mean()*100:4.1f}%) "
              f"{pt.sum():9d} ({pt.mean()*100:4.1f}%) {mx.sum():6d}")
print()
print("="*78)
print("J. GLOBAL COUNT of boundary-SATURATED channels (normE(t-2) > 0.8) -- all layers")
print("="*78)
for nm,p in [("350M",f"{root}/model.safetensors"),("700M",f"{root}/LFM2-700M/model.safetensors"),
             ("1.2B",f"{root}/LFM2-1.2B/model.safetensors"),("2.6B",f"{root}/LFM2-2.6B/model.safetensors")]:
    h,g=load(p)
    ks=sorted([k for k in h if k.endswith("conv.conv.weight")],key=lambda k:int(k.split(".")[2]))
    W=np.concatenate([g(k).reshape(-1,3) for k in ks],0)
    e=W**2; en=e/e.sum(1,keepdims=True)
    print(f"   {nm}: n={len(W):6d}  normE(t-2)>0.8: {int((en[:,0]>0.8).sum()):5d} ({(en[:,0]>0.8).mean()*100:.2f}%)"
          f"   >0.5: {int((en[:,0]>0.5).sum()):5d} ({(en[:,0]>0.5).mean()*100:.2f}%)")
