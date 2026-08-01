import json, struct, numpy as np

p = "/scratch/users/ericrcwu/liv/ckpt/model.safetensors"
f = open(p, "rb")
n = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(n))
base = 8 + n


def get(k):
    m = hdr[k]
    s, e = m["data_offsets"]
    f.seek(base + s)
    b = f.read(e - s)
    a = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32).reshape(m["shape"])


convs = sorted(
    [k for k in hdr if k.endswith("conv.conv.weight")], key=lambda k: int(k.split(".")[2])
)
print("n_conv_layers", len(convs))
allw = []
print("layer  mean|t0|  mean|t1|  mean|t2|     E0%    E1%    E2%   fr(|t0|>|t2|)  boundary_argmax")
for k in convs:
    w = get(k).reshape(-1, 3)  # (1024,3); index 0 = OLDEST lag (t-2), index 2 = current token
    allw.append(w)
    m = np.abs(w).mean(0)
    E = (w ** 2).sum(0)
    E = E / E.sum()
    fr = float((np.abs(w[:, 0]) > np.abs(w[:, 2])).mean())
    bd = float((np.abs(w).argmax(1) == 0).mean())
    li = int(k.split(".")[2])
    print(
        "%5d  %8.4f  %8.4f  %8.4f  %6.2f %6.2f %6.2f  %13.3f  %15.3f"
        % (li, m[0], m[1], m[2], E[0] * 100, E[1] * 100, E[2] * 100, fr, bd)
    )

W = np.concatenate(allw, 0)
E = (W ** 2).sum(0)
E = E / E.sum()
print()
print("POOLED energy%% by tap (oldest t-2 -> current t):", np.round(E * 100, 2))
print("POOLED mean|w| by tap:", np.round(np.abs(W).mean(0), 4))
print(
    "frac channels where OLDEST tap has largest magnitude:",
    round(float((np.abs(W).argmax(1) == 0).mean()), 4),
)
print(
    "frac channels where |oldest| > |current|:",
    round(float((np.abs(W[:, 0]) > np.abs(W[:, 2])).mean()), 4),
)
print(
    "frac channels where |oldest| > 0.5*max|tap|  (boundary NOT decayed):",
    round(float((np.abs(W[:, 0]) > 0.5 * np.abs(W).max(1)).mean()), 4),
)
print(
    "frac channels where |oldest| > 0.9*max|tap|  (boundary saturated):",
    round(float((np.abs(W[:, 0]) > 0.9 * np.abs(W).max(1)).mean()), 4),
)
# Geometric decay ratio: if taps decayed geometrically we'd see |t0|/|t2| << 1
r = np.abs(W[:, 0]) / (np.abs(W[:, 2]) + 1e-12)
print("|oldest|/|current| ratio  median=%.3f  p25=%.3f  p75=%.3f" % (
    np.median(r), np.percentile(r, 25), np.percentile(r, 75)))
