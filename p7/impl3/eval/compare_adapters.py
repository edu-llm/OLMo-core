#!/usr/bin/env python
"""Compare two LoRA adapters by the weight update they actually apply to the base model.

Motivation: if two SFT runs used the same data, seed and hyperparameters, they should land in
almost the same place. Comparing the raw ``lora_A`` / ``lora_B`` tensors does NOT show that,
because only their product is identified — any invertible R gives (B R)(R^-1 A) = B A, so two
equivalent adapters can have completely different A and B. The comparable quantity is the
effective update

    dW = (alpha / r) * B @ A

which is what gets added to the frozen base weight. Per module we report ||dW||_F for each
adapter and the cosine similarity between them, plus a relative size difference:

    cos ~ 1.0  and  size ratio ~ 1.0   -> the two runs learned the same thing
    cos ~ 1.0  but ratio far from 1    -> same direction, different distance travelled
    cos ~ 0                            -> unrelated solutions

    python compare_adapters.py A_dir B_dir [--per_module]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from safetensors import safe_open


def load_deltas(path):
    """{module_name: dW} for every LoRA-wrapped module in an adapter dir."""
    cfg = json.load(open(os.path.join(path, "adapter_config.json"), encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(os.path.join(path, "adapter_model.safetensors"), "np")
    a, b = {}, {}
    for k in f.keys():
        if ".lora_A" in k:
            a[k.split(".lora_A")[0]] = f.get_tensor(k)
        elif ".lora_B" in k:
            b[k.split(".lora_B")[0]] = f.get_tensor(k)
    common = sorted(set(a) & set(b))
    return {m: (b[m].astype(np.float32) @ a[m].astype(np.float32)) * scale for m in common}, cfg


def short(name):
    """base_model.model.model.layers.7.mlp.up_proj -> L07.up_proj"""
    parts = name.split(".")
    layer = next((parts[i + 1] for i, p in enumerate(parts) if p == "layers"), "?")
    return f"L{int(layer):02d}.{parts[-1]}" if layer.isdigit() else name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--per_module", action="store_true", help="Print every module, not just the summary.")
    args = p.parse_args()

    da, cfg_a = load_deltas(args.a)
    db, cfg_b = load_deltas(args.b)
    for label, cfg in ((args.a, cfg_a), (args.b, cfg_b)):
        print(f"{label}: r={cfg['r']} alpha={cfg['lora_alpha']} base={cfg.get('base_model_name_or_path')}")

    common = sorted(set(da) & set(db))
    if not common:
        raise SystemExit("no shared modules between the two adapters")
    only = (set(da) ^ set(db))
    if only:
        print(f"[warn] {len(only)} modules present in only one adapter")
    print(f"comparing {len(common)} modules\n")

    rows, by_kind = [], defaultdict(list)
    for m in common:
        x, y = da[m].ravel(), db[m].ravel()
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        cos = float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else float("nan")
        chance = 1.0 / np.sqrt(x.size)  # SD of cosine between random vectors in this many dims
        rows.append((short(m), nx, ny, ny / nx if nx > 0 else float("nan"), cos, chance))
        by_kind[m.split(".")[-1]].append((nx, ny, cos, chance))

    if args.per_module:
        print(f"{'module':<18}{'||dW|| A':>12}{'||dW|| B':>12}{'B/A':>8}{'cosine':>9}")
        print("-" * 59)
        for r in rows:
            print(f"{r[0]:<18}{r[1]:>12.4f}{r[2]:>12.4f}{r[3]:>8.2f}{r[4]:>9.3f}")
        print()

    # A raw cosine is unreadable here: dW lives in a space of millions of dimensions, where two
    # unrelated vectors score ~1/sqrt(dim), i.e. ~2e-4. Judging 0.16 as "weak" against a mental
    # baseline of 1.0 is wrong by three orders of magnitude, so report it in units of chance.
    print(f"{'module type':<14}{'n':>4}{'mean ||dW|| A':>15}{'mean ||dW|| B':>15}{'B/A':>8}"
          f"{'mean cos':>10}{'vs chance':>11}")
    print("-" * 77)
    for kind in sorted(by_kind):
        v = by_kind[kind]
        ma = float(np.mean([t[0] for t in v]))
        mb = float(np.mean([t[1] for t in v]))
        mc = float(np.mean([t[2] for t in v]))
        ch = float(np.mean([t[3] for t in v]))
        print(f"{kind:<14}{len(v):>4}{ma:>15.4f}{mb:>15.4f}{mb / ma:>8.2f}{mc:>10.3f}{mc / ch:>10.0f}x")

    alla = float(np.mean([r[1] for r in rows]))
    allb = float(np.mean([r[2] for r in rows]))
    allc = float(np.mean([r[4] for r in rows]))
    allch = float(np.mean([r[5] for r in rows]))
    ratio = allb / alla
    print("-" * 77)
    print(f"{'ALL':<14}{len(rows):>4}{alla:>15.4f}{allb:>15.4f}{ratio:>8.2f}{allc:>10.3f}"
          f"{allc / allch:>10.0f}x")

    print("\ninterpretation")
    same_distance = 0.95 < ratio < 1.05
    print(f"  distance from base: {'MATCHES' if same_distance else 'DIFFERS'} "
          f"(||dW|| ratio {ratio:.2f})")
    print(f"  direction:          {allc:.3f} cosine = {allc / allch:.0f}x chance")
    if same_distance and allc / allch > 20:
        print("\n  Same recipe. Both runs travelled the same distance from the base model and their")
        print("  updates are related far beyond chance, but they settled in different corners of an")
        print("  equivalent solution -- what you get from a different data shuffle or different")
        print("  numerics (dtype, GPU, library version) over hundreds of steps. This is what a")
        print("  reproduction looks like at LoRA scale; it is NOT evidence of a training difference.")
    elif not same_distance:
        print("\n  The runs moved different distances from base -> a real training difference")
        print("  (learning rate, steps, effective batch, or what the loss was computed over).")
    else:
        print("\n  Updates are near chance-level unrelated -> these did not train on the same task.")


if __name__ == "__main__":
    main()
