#!/usr/bin/env python3
"""
Sample Dirichlet domain-weight grids for RegMix / data-mixing-law proxy runs (smoke scale).

Writes JSON files under --out-dir; does not launch training.

Example:
  python sample_dirichlet_mixes.py --n-samples 16 --out-dir ./data/smoke_dolma_v0/manifests/mixes/regmix_grid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_DOMAINS = ["web", "code", "stem", "wiki", "social", "flan"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=16, help="Smoke default 16; RegMix used 512")
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration")
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.full(len(args.domains), args.alpha, dtype=np.float64)

    index = []
    for i in range(args.n_samples):
        w = rng.dirichlet(alpha)
        weights = {d: float(x) for d, x in zip(args.domains, w)}
        name = f"dirichlet_{i:04d}"
        path = args.out_dir / f"{name}.json"
        payload = {"name": name, "weights": weights}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index.append({"name": name, "path": str(path)})

    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {args.n_samples} mixes → {args.out_dir}")


if __name__ == "__main__":
    main()
