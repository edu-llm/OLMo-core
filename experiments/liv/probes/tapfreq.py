"""Frequency-response + cross-scale tap characterization of released LFM2 conv kernels.

Zero GPU. Reads safetensors headers directly, decodes bf16 by bit-shift.
"""
import json
import os
import struct
import sys

import numpy as np


def load_convs(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

    def get(k):
        m = hdr[k]
        s, e = m["data_offsets"]
        f.seek(base + s)
        b = f.read(e - s)
        if m["dtype"] == "BF16":
            a = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
            return a.view(np.float32).reshape(m["shape"])
        if m["dtype"] == "F32":
            return np.frombuffer(b, dtype=np.float32).reshape(m["shape"])
        raise ValueError(m["dtype"])

    keys = sorted(
        [k for k in hdr if k.endswith("conv.conv.weight")], key=lambda k: int(k.split(".")[2])
    )
    return [(int(k.split(".")[2]), get(k).reshape(get(k).shape[0], -1)) for k in keys]


def analyze(name, path):
    layers = load_convs(path)
    if not layers:
        print(f"{name}: NO conv tensors found")
        return
    k = layers[0][1].shape[1]
    d = layers[0][1].shape[0]
    print(f"\n===== {name}   d={d}  k={k}  n_liv_layers={len(layers)} =====")
    print(" layer   E%[t-2]  E%[t-1]   E%[t]   | lowpass  highpass  passthru  delay | |H(0)|/|H(pi)|")
    allw = []
    for li, w in layers:
        allw.append(w)
        E = (w ** 2).sum(0)
        E = E / E.sum()
        # DC gain H(0)=sum(w); Nyquist gain H(pi)=sum(w*(-1)^lag) with lag measured from current
        # taps ordered [t-2, t-1, t]; lag of tap j is (k-1-j)
        lags = np.arange(k)[::-1]
        H0 = w.sum(1)
        Hpi = (w * ((-1.0) ** lags)).sum(1)
        lp = float((np.abs(H0) > 2 * np.abs(Hpi)).mean())
        hp = float((np.abs(Hpi) > 2 * np.abs(H0)).mean())
        # passthrough: current tap carries >90% of the channel's energy
        ce = w ** 2
        ce = ce / (ce.sum(1, keepdims=True) + 1e-20)
        pt = float((ce[:, -1] > 0.9).mean())
        dl = float((ce[:, :-1].sum(1) > 0.9).mean())  # history dominates -> pure delay
        ratio = float(np.median(np.abs(H0) / (np.abs(Hpi) + 1e-9)))
        e = list(np.round(E * 100, 2))
        print(
            f"{li:5d}   {e[0]:7.2f}  {e[1]:7.2f}  {e[-1]:7.2f}   | {lp:7.3f}  {hp:8.3f}  {pt:8.3f}  {dl:5.3f} | {ratio:8.3f}"
            if k == 3
            else f"{li:5d}   {str(e):>28s}  | {lp:7.3f} {hp:8.3f} {pt:8.3f} {dl:5.3f} | {ratio:8.3f}"
        )
    W = np.concatenate(allw, 0)
    E = (W ** 2).sum(0)
    E = E / E.sum()
    lags = np.arange(k)[::-1]
    H0 = W.sum(1)
    Hpi = (W * ((-1.0) ** lags)).sum(1)
    print(f"POOLED  energy%% by tap (oldest->current): {np.round(E * 100, 2)}")
    print(f"POOLED  oldest-tap energy share: {E[0] * 100:.2f}%")
    print(
        f"POOLED  frac channels oldest-tap-is-argmax: {(np.abs(W).argmax(1) == 0).mean():.4f}"
        f"   |oldest|>0.9max: {(np.abs(W[:, 0]) > 0.9 * np.abs(W).max(1)).mean():.4f}"
    )
    print(
        f"POOLED  lowpass(|H0|>2|Hpi|): {(np.abs(H0) > 2 * np.abs(Hpi)).mean():.3f}   "
        f"highpass(|Hpi|>2|H0|): {(np.abs(Hpi) > 2 * np.abs(H0)).mean():.3f}"
    )


if __name__ == "__main__":
    root = "/scratch/users/ericrcwu/liv/ckpt"
    targets = [("LFM2-350M", f"{root}/model.safetensors")]
    for m in ("LFM2-700M", "LFM2-1.2B", "LFM2-2.6B"):
        p = f"{root}/{m}/model.safetensors"
        if os.path.exists(p) and os.path.getsize(p) > 10_000_000:
            targets.append((m, p))
    for name, p in targets:
        try:
            analyze(name, p)
        except Exception as ex:  # noqa: BLE001
            print(f"{name}: FAILED {type(ex).__name__}: {ex}", file=sys.stderr)
