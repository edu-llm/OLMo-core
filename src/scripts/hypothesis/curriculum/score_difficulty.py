#!/usr/bin/env python3
"""
Score smoke docs for curriculum difficulty metrics (no model required for most).

Metrics:
  - random
  - compression_ratio  (zlib compressed / raw bytes; higher => harder/less compressible — we invert for "easy")
  - flesch             (needs textstat; higher reading ease => easier)
  - lexical_diversity  (type-token ratio; higher => harder in our sort)
  - learnability       (placeholder file; fill later from proxy early/late Δloss)

Writes: <data>/scores/<metric>.jsonl  with {doc_id, score, metric}

Example:
  python score_difficulty.py --data-dir ./data/smoke_dolma_v0 --metrics compression_ratio flesch lexical_diversity random
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path


def iter_docs(data_dir: Path):
    raw = data_dir / "raw"
    for domain_dir in sorted(raw.iterdir()):
        if not domain_dir.is_dir():
            continue
        fp = domain_dir / "docs.jsonl"
        if not fp.exists():
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def score_compression(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    # higher ratio = more compressible = "easier" for our curriculum sort (sort ascending = easy first)
    return len(zlib.compress(raw, 9)) / max(len(raw), 1)


def score_flesch(text: str) -> float:
    try:
        import textstat

        return float(textstat.flesch_reading_ease(text))
    except Exception:
        # fallback: crude inverse of avg word length
        words = text.split()
        if not words:
            return 0.0
        avg = sum(len(w) for w in words) / len(words)
        return 100.0 - avg * 10.0


def score_lexical_diversity(text: str) -> float:
    words = [w.lower() for w in text.split()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["random", "compression_ratio", "flesch", "lexical_diversity"],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random

    rng = random.Random(args.seed)
    docs = list(iter_docs(args.data_dir))
    if not docs:
        raise SystemExit(f"No docs under {args.data_dir}/raw")

    score_dir = args.data_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        rows = []
        for doc in docs:
            text = doc.get("text", "")
            if metric == "random":
                s = rng.random()
            elif metric == "compression_ratio":
                s = score_compression(text)
            elif metric == "flesch":
                # higher flesch = easier; negate so ascending sort = easy→hard if we sort by score ascending for "easy first"
                s = -score_flesch(text)
            elif metric == "lexical_diversity":
                s = score_lexical_diversity(text)
            elif metric == "learnability":
                s = 0.0  # placeholder
            else:
                raise SystemExit(f"Unknown metric: {metric}")
            rows.append({"doc_id": doc["doc_id"], "score": float(s), "metric": metric})

        out = score_dir / f"{metric}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {out} ({len(rows)} rows)")

    print("Note: for learnability, run a proxy early/late loss job and overwrite scores/learnability.jsonl")


if __name__ == "__main__":
    main()
