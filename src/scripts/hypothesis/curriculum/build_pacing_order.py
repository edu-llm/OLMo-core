#!/usr/bin/env python3
"""
Build document order files for curriculum pacing (smoke).

Pacing:
  - random
  - vanilla     : strict easy→hard by metric score ascending
  - linear      : mixture whose mean difficulty increases (binning)
  - warmup      : first half linear CL, second half random shuffle

Writes: <data>/orders/<pacing>__<metric>.jsonl  with ordered doc_ids

Example:
  python build_pacing_order.py --data-dir ./data/smoke_dolma_v0 --pacing linear --metric compression_ratio
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_scores(path: Path) -> dict[str, float]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["doc_id"]] = float(row["score"])
    return out


def linear_pace(sorted_ids: list[str], n_bins: int = 10) -> list[str]:
    """Within rising difficulty windows, shuffle so easy+hard mix but mean rises."""
    if not sorted_ids:
        return []
    bins = [[] for _ in range(n_bins)]
    for i, doc_id in enumerate(sorted_ids):
        b = min(n_bins - 1, int(i / max(len(sorted_ids), 1) * n_bins))
        bins[b].append(doc_id)
    order = []
    # grow available pool from easy→hard
    pool: list[str] = []
    for b in bins:
        pool.extend(b)
        chunk = pool.copy()
        random.shuffle(chunk)
        # take a slice proportional to bin size from current pool
        take = max(1, len(b))
        order.extend(chunk[:take])
        # remove taken from pool (approx)
        taken = set(chunk[:take])
        pool = [x for x in pool if x not in taken]
    # append leftovers
    random.shuffle(pool)
    order.extend(pool)
    # dedupe preserving order
    seen = set()
    final = []
    for x in order:
        if x not in seen:
            seen.add(x)
            final.append(x)
    for x in sorted_ids:
        if x not in seen:
            final.append(x)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--pacing", choices=["random", "vanilla", "linear", "warmup"], required=True)
    ap.add_argument("--metric", default="compression_ratio")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    score_path = args.data_dir / "scores" / f"{args.metric}.jsonl"
    if args.pacing != "random" and not score_path.exists():
        raise SystemExit(f"Missing {score_path}; run score_difficulty.py first")

    if args.pacing == "random":
        # any score file or docs manifest for id list
        docs_path = args.data_dir / "manifests" / "docs.jsonl"
        ids = []
        with docs_path.open(encoding="utf-8") as f:
            for line in f:
                ids.append(json.loads(line)["doc_id"])
        random.shuffle(ids)
        order = ids
        tag = f"random__none"
    else:
        scores = load_scores(score_path)
        sorted_ids = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1])]
        if args.pacing == "vanilla":
            order = sorted_ids
        elif args.pacing == "linear":
            order = linear_pace(sorted_ids)
        else:  # warmup
            n = len(sorted_ids)
            first = linear_pace(sorted_ids[: max(1, n // 2)])
            second = sorted_ids[n // 2 :]
            random.shuffle(second)
            order = first + second
        tag = f"{args.pacing}__{args.metric}"

    out_dir = args.data_dir / "orders"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, doc_id in enumerate(order):
            f.write(json.dumps({"position": i, "doc_id": doc_id, "pacing": args.pacing, "metric": args.metric}) + "\n")
    print(f"Wrote {out} ({len(order)} docs)")


if __name__ == "__main__":
    main()
