"""INDEPENDENT verification of the LFM2 conv-tap read.
Written from scratch by the verification agent. Does NOT import tapread*.py.
Part 1: settle tap-index orientation EMPIRICALLY with a one-hot impulse through
        the real nn.Conv1d(padding=k-1) + left-slice, AND through the decode path.
Part 2: recompute per-layer / pooled statistics from raw safetensors bytes.
Part 3: tail analysis, layer-0/1 argmax breakdown, control analysis.
"""
import json, struct, os, sys
import numpy as np

# ---------------------------------------------------------------- PART 1
print("="*78)
print("PART 1  -- TAP-INDEX ORIENTATION, settled empirically")
print("="*78)
import torch, torch.nn as nn

k = 3
conv = nn.Conv1d(4, 4, kernel_size=k, groups=4, bias=False, padding=k-1)
with torch.no_grad():
    # channel c gets a kernel that is one-hot at weight index c (c=0,1,2)
    conv.weight.zero_()
    for c in range(3):
        conv.weight[c, 0, c] = 1.0
    conv.weight[3, 0, :] = torch.tensor([1., 10., 100.])  # positional marker

T = 6
x = torch.zeros(1, 4, T)
x[:, :, :] = torch.arange(T).float()[None, None, :]   # x[t] = t, easy to read off
seqlen = T
out = conv(x)[..., :seqlen]
print("input x[t] = t, for t=0..5")
for c in range(3):
    print(f"  weight one-hot at INDEX {c}: out = {out[0,c].tolist()}")
print(f"  weight [1,10,100]         : out = {out[0,3].tolist()}")
print()
print("READ: if out[t]==t for the one-hot channel, that weight index multiplies the CURRENT token.")
for c in range(3):
    o = out[0, c]
    lag = None
    for L in range(k):
        # out[t] should equal max(t-L,0)-ish; test at t=5 which is unpadded
        if abs(o[5].item() - (5 - L)) < 1e-6:
            lag = L
    print(f"  index {c} -> lag {lag}   ({'CURRENT token t' if lag==0 else f't-{lag}'})")
print()

# decode path used by slow_forward: conv_out = sum(conv_state * weight[:,0,:], -1)
# where conv_state is the rolling cache, [..., -L_cache:] i.e. NEWEST AT THE END.
cs = torch.tensor([[[3., 4., 5.]]])       # history t-2=3, t-1=4, current=5
for c in range(3):
    w = torch.zeros(1, 3); w[0, c] = 1.0
    print(f"  DECODE path, weight one-hot idx {c}: sum(conv_state*w) = {(cs[0]*w).sum().item()}  "
          f"(3=t-2, 4=t-1, 5=current)")
print()

# ---------------------------------------------------------------- PART 2
print("="*78)
print("PART 2  -- INDEPENDENT RECOMPUTATION FROM RAW SAFETENSORS")
print("="*78)

def load(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n
    def get(kk):
        m = hdr[kk]; s, e = m["data_offsets"]
        f.seek(base + s); b = f.read(e - s)
        if m["dtype"] == "BF16":
            a = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
            return a.view(np.float32).reshape(m["shape"]).copy()
        if m["dtype"] == "F32":
            return np.frombuffer(b, dtype=np.float32).reshape(m["shape"]).copy()
        if m["dtype"] == "F16":
            return np.frombuffer(b, dtype=np.float16).astype(np.float32).reshape(m["shape"]).copy()
        raise ValueError(m["dtype"])
    return hdr, get

def report(name, path, cfgpath):
    hdr, get = load(path)
    cfg = json.load(open(cfgpath))
    convkeys = sorted([kk for kk in hdr if kk.endswith("conv.conv.weight")],
                      key=lambda kk: int(kk.split(".")[2]))
    attn = sorted(set(int(kk.split(".")[2]) for kk in hdr if ".self_attn." in kk))
    livs = [int(kk.split(".")[2]) for kk in convkeys]
    print(f"\n##### {name}")
    print(f"  hidden={cfg.get('hidden_size')}  conv_L_cache={cfg.get('conv_L_cache')} "
          f"conv_bias={cfg.get('conv_bias')} conv_use_xavier_init={cfg.get('conv_use_xavier_init')}")
    print(f"  n_layers={cfg.get('num_hidden_layers')}  ATTENTION layers={attn}")
    print(f"  LIV(conv) layers={livs}  (disjoint from attn: {set(attn).isdisjoint(livs)})")
    shp = hdr[convkeys[0]]["shape"]
    print(f"  conv weight shape={shp}  dtype={hdr[convkeys[0]]['dtype']}  "
          f"conv bias present={any('conv.conv.bias' in kk for kk in hdr)}")
    K = shp[-1]

    print(f"\n  {'lyr':>3} {'medEn[t-2]':>10} {'medEn[t-1]':>10} {'medEn[t]':>9} "
          f"| {'argmax@t-2':>10} {'argmax@t-1':>10} {'argmax@t':>8} "
          f"| {'>20%@t-2':>8} {'>30%@t-2':>8} {'>50%@t-2':>8} | {'rawE0%':>7} {'rawE1%':>7} {'rawE2%':>7}")
    allw = []
    for kk in convkeys:
        w = get(kk).reshape(-1, K)
        allw.append(w)
        e = w**2
        en = e / np.clip(e.sum(1, keepdims=True), 1e-30, None)
        md = np.median(en, 0)
        am = np.abs(w).argmax(1)
        raw = e.sum(0); raw = raw/raw.sum()
        li = int(kk.split(".")[2])
        print(f"  {li:>3} {md[0]:10.4f} {md[1]:10.4f} {md[-1]:9.4f} "
              f"| {(am==0).mean():10.4f} {(am==1).mean():10.4f} {(am==K-1).mean():8.4f} "
              f"| {(en[:,0]>0.20).mean():8.4f} {(en[:,0]>0.30).mean():8.4f} {(en[:,0]>0.50).mean():8.4f} "
              f"| {raw[0]*100:7.2f} {raw[1]*100:7.2f} {raw[-1]*100:7.2f}")

    W = np.concatenate(allw, 0)
    e = W**2; en = e/np.clip(e.sum(1,keepdims=True),1e-30,None); md = np.median(en,0)
    raw = e.sum(0); raw = raw/raw.sum()
    am = np.abs(W).argmax(1)
    print(f"\n  POOLED n_channels={W.shape[0]}")
    print(f"    raw pooled energy%% by tap [oldest..current] = {np.round(raw*100,2)}")
    print(f"    per-channel-normalized MEDIAN energy         = {np.round(md,4)}")
    print(f"    ratio medE(t-2)/medE(t-1)                    = {md[0]/md[1]:.4f}")
    print(f"    frac argmax==oldest                          = {(am==0).mean():.4f}")
    print(f"    frac |oldest|>|current|                      = {(np.abs(W[:,0])>np.abs(W[:,-1])).mean():.4f}")
    print(f"    frac |oldest|>0.9*max                        = {(np.abs(W[:,0])>0.9*np.abs(W).max(1)).mean():.4f}")
    print(f"    frac |oldest|>0.5*max                        = {(np.abs(W[:,0])>0.5*np.abs(W).max(1)).mean():.4f}")
    print(f"    TAIL: frac normE(t-2)>0.20 / >0.30 / >0.50   = "
          f"{(en[:,0]>0.20).mean():.4f} / {(en[:,0]>0.30).mean():.4f} / {(en[:,0]>0.50).mean():.4f}")
    print(f"    TAIL: frac normE(t-1)+normE(t-2)>0.50        = {((en[:,0]+en[:,1])>0.50).mean():.4f}")
    print(f"    normE(t-2) percentiles p50/p75/p90/p95/p99   = "
          f"{np.percentile(en[:,0],[50,75,90,95,99]).round(4)}")
    return W, allw, livs

root = "/scratch/users/ericrcwu/liv/ckpt"
W350, layers350, livs350 = report("LFM2-350M", f"{root}/model.safetensors", f"{root}/config.json")
for m in ("LFM2-700M","LFM2-1.2B","LFM2-2.6B"):
    p = f"{root}/{m}/model.safetensors"
    if os.path.exists(p):
        report(m, p, f"{root}/{m}/config.json")

# ---------------------------------------------------------------- PART 3
print()
print("="*78)
print("PART 3  -- CONTROL ANALYSIS + LAYER 0/1 DEEP DIVE (350M)")
print("="*78)
rng = np.random.default_rng(0)
for nm, R in [("U(+-1/sqrt3)  [their control]", rng.uniform(-1/np.sqrt(3),1/np.sqrt(3),(200000,3))),
              ("U(+-1.0)      [10x scale]",     rng.uniform(-1,1,(200000,3))),
              ("N(0,1)        [gaussian]",      rng.normal(0,1,(200000,3))),
              ("Xavier U(+-sqrt(6/(1+3)))",     rng.uniform(-np.sqrt(6/4),np.sqrt(6/4),(200000,3)))]:
    er = R**2; er = er/er.sum(1,keepdims=True)
    print(f"  {nm:32s} medE={np.round(np.median(er,0),4)}  argmax@0={(np.abs(R).argmax(1)==0).mean():.4f} "
          f" >0.9max={(np.abs(R[:,0])>0.9*np.abs(R).max(1)).mean():.4f}")
print("  NOTE: normalized-energy stats are SCALE-INVARIANT -> every iid symmetric control")
print("        gives argmax@oldest == 1/3 exactly, by exchangeability. Control carries 0 bits.")
print()
print("  LAYER 0 / LAYER 1 argmax breakdown and lag-1 concentration:")
for li, w in zip(livs350, layers350):
    if li > 4: continue
    e = w**2; en = e/e.sum(1,keepdims=True); am = np.abs(w).argmax(1)
    print(f"    layer {li}: argmax t-2={np.mean(am==0):.4f} t-1={np.mean(am==1):.4f} t={np.mean(am==2):.4f} "
          f"| medE=[{np.median(en[:,0]):.4f},{np.median(en[:,1]):.4f},{np.median(en[:,2]):.4f}] "
          f"| frac normE(t-1)>0.9 = {(en[:,1]>0.9).mean():.4f} "
          f"| frac normE(t-2)>0.2 = {(en[:,0]>0.2).mean():.4f}")
print()
print("  Sanity: are the per-layer weight NORMS comparable? (a near-dead conv layer with tiny")
print("  norm contributes little regardless of its tap profile)")
for li, w in zip(livs350, layers350):
    print(f"    layer {li:>2}  ||W||_F={np.linalg.norm(w):8.4f}  mean|w|={np.abs(w).mean():.5f}  "
          f"max|w|={np.abs(w).max():.4f}")
