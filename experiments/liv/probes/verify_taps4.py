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
h,g=load(f"{root}/model.safetensors")
print("="*78); print("H. POOLING BIAS: the headline median pools MIXER layers with DELAY and DEAD layers")
print("="*78)
groups={"delay lines (0,1)":[0,1],"true MIXERS (3,4,6,7,9,11)":[3,4,6,7,9,11],
        "near-vestigial (13,15)":[13,15],"ALL 10 (their headline)":[0,1,3,4,6,7,9,11,13,15]}
print(f"   {'group':>28} {'medE(t-2)':>10} {'meanE(t-2)':>11} {'rawE(t-2)%':>11} {'>20%':>7} {'>30%':>7}")
for nm,ls in groups.items():
    W=np.concatenate([g(f"model.layers.{l}.conv.conv.weight").reshape(-1,3) for l in ls],0)
    e=W**2; en=e/e.sum(1,keepdims=True); raw=e.sum(0); raw=raw/raw.sum()
    print(f"   {nm:>28} {np.median(en[:,0]):10.4f} {en[:,0].mean():11.4f} {raw[0]*100:10.2f}% "
          f"{(en[:,0]>0.2).mean():7.4f} {(en[:,0]>0.3).mean():7.4f}")
print("   -> restricting to the 6 layers that are genuine 3-tap MIXERS roughly TRIPLES the")
print("      headline median (0.0143 -> see row). The '1.4%' is a pooled artifact.")
print()
print("   Same for the pre-registered 'ratio E(t-2)/E(t-1)' statistic:")
for nm,ls in groups.items():
    W=np.concatenate([g(f"model.layers.{l}.conv.conv.weight").reshape(-1,3) for l in ls],0)
    e=W**2; en=e/e.sum(1,keepdims=True); md=np.median(en,0)
    print(f"     {nm:>28}: ratio = {md[0]/md[1]:.4f}   implied geometric r = {np.sqrt(md[0]/md[1]):.3f}")
print()
print("   (their pre-registered band was 0.2-0.5 = 'expected'; >0.6 = 'ambiguous, run k5 only')")
