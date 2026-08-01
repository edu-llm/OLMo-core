import json, struct, numpy as np
p="/scratch/users/ericrcwu/liv/ckpt/model.safetensors"
f=open(p,"rb"); n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n)); base=8+n
def get(k):
    m=hdr[k]; s,e=m["data_offsets"]; f.seek(base+s); b=f.read(e-s)
    a=np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16
    return a.view(np.float32).reshape(m["shape"])
convs=sorted([k for k in hdr if k.endswith("conv.conv.weight")],key=lambda k:int(k.split(".")[2]))
print("PER-CHANNEL-NORMALIZED energy, MEDIAN across 1024 channels (spec item 1)")
print("layer   med_E(t-2)  med_E(t-1)  med_E(t)   ||W||_F   mean|w| all taps")
allw=[]
for k in convs:
    w=get(k).reshape(-1,3); allw.append(w)
    e=w**2; e=e/np.clip(e.sum(1,keepdims=True),1e-30,None)
    md=np.median(e,0)
    print("%5d   %9.4f  %9.4f  %9.4f   %8.4f  %8.5f"%(int(k.split(".")[2]),md[0],md[1],md[2],np.linalg.norm(w),np.abs(w).mean()))
W=np.concatenate(allw,0)
e=W**2; e=e/np.clip(e.sum(1,keepdims=True),1e-30,None); md=np.median(e,0)
print()
print("POOLED per-channel-normalized MEDIAN energy [t-2,t-1,t]:",np.round(md,4))
print("  -> ratio med_E(t-2)/med_E(t-1) =",round(float(md[0]/md[1]),4))
# random-init control (spec item 5): PyTorch Conv1d default = U(-1/sqrt(fan_in),+), fan_in=k=3
rng=np.random.default_rng(0); b=1/np.sqrt(3)
R=rng.uniform(-b,b,size=(10240,3))
er=R**2; er=er/er.sum(1,keepdims=True)
print()
print("RANDOM-INIT CONTROL (U(-1/sqrt3,1/sqrt3), n=10240):")
print("  median normalized energy [t-2,t-1,t]:",np.round(np.median(er,0),4))
print("  boundary_argmax frac:",round(float((np.abs(R).argmax(1)==0).mean()),4)," (chance=0.3333)")
print("  frac |oldest|>0.9*max:",round(float((np.abs(R[:,0])>0.9*np.abs(R).max(1)).mean()),4))
print()
print("TRAINED:")
print("  boundary_argmax frac:",round(float((np.abs(W).argmax(1)==0).mean()),4))
print("  frac |oldest|>0.9*max:",round(float((np.abs(W[:,0])>0.9*np.abs(W).max(1)).mean()),4))
# vestigial check: conv deviation from pure identity (delta at lag 0 = current token)
print()
print("VESTIGIAL CHECK -- fraction of each layer energy NOT on the current tap:")
for k,w in zip(convs,allw):
    E=(w**2).sum(0); print("  layer %2d  off-current energy = %6.2f%%"%(int(k.split(".")[2]),100*(E[0]+E[1])/E.sum()))
