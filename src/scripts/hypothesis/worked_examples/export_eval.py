#!/usr/bin/env python3
"""Re-export / sanity-check holdout eval file (bare prompts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True)
    args = ap.parse_args()
    src = args.pack_dir / "eval" / "holdout_bare.jsonl"
    if not src.exists():
        raise SystemExit(f"Missing {src}; run build_from_metamath.py (or build_from_gsm8k.py) first")
    n = 0
    with src.open(encoding="utf-8") as f:
        for line in f:
            json.loads(line)
            n += 1
    print(f"OK: {n} holdout prompts at {src}")
    print("Eval protocol: generate completion after prompt; score final number vs final_answer.")
    print("Report Pass@N and PassRatio@N over N samples per item.")


if __name__ == "__main__":
    main()
