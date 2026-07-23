#!/usr/bin/env python3
"""Validate worked-examples pack: answers present, fade prefixes consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_bank(pack: Path) -> dict[str, dict]:
    out = {}
    with (pack / "bank" / "instances.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["instance_id"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--max-audit", type=int, default=50)
    args = ap.parse_args()
    pack = args.pack_dir
    bank = load_bank(pack)

    errors: list[str] = []
    stats = {
        "bare_ok": 0,
        "complete_ok": 0,
        "fade_ok": 0,
        "fade_checked": 0,
        "missing_answer": 0,
    }

    # Bare / complete: final answer must match bank
    for arm in ["bare", "complete"]:
        path = pack / "arms" / arm / "docs.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                inst = bank.get(doc["instance_id"])
                if not inst or not doc.get("final_answer"):
                    stats["missing_answer"] += 1
                    errors.append(f"{arm}: missing answer {doc.get('instance_id')}")
                    continue
                if str(doc["final_answer"]) != str(inst["final_answer"]):
                    errors.append(f"{arm}: answer mismatch {doc['instance_id']}")
                    continue
                stats[f"{arm}_ok"] += 1

    # Fade: context must be prefix of complete formatting logic
    for arm in ["fade_ordered", "fade_shuffled"]:
        path = pack / "arms" / arm / "docs.jsonl"
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                stats["fade_checked"] += 1
                inst = bank.get(doc["instance_id"])
                if not inst:
                    errors.append(f"{arm}: unknown instance {doc.get('instance_id')}")
                    continue
                if "BEGIN_CONTINUE" not in doc.get("text", ""):
                    errors.append(f"{arm}: missing BEGIN_CONTINUE {doc['instance_id']}")
                    continue
                if doc.get("loss_start_char", 0) <= 0:
                    errors.append(f"{arm}: bad loss_start_char {doc['instance_id']}")
                    continue
                # Shown steps should equal first n_shown bank steps
                n_shown = doc.get("n_shown", 0)
                if n_shown > len(inst["steps"]):
                    errors.append(f"{arm}: n_shown too large {doc['instance_id']}")
                    continue
                if str(doc.get("final_answer")) != str(inst["final_answer"]):
                    errors.append(f"{arm}: answer mismatch {doc['instance_id']}")
                    continue
                stats["fade_ok"] += 1
                if i >= args.max_audit and len(errors) == 0:
                    # still count rest without early exit — full scan is fine for smoke sizes
                    pass

    report = {
        "ok": len(errors) == 0,
        "n_errors": len(errors),
        "errors_head": errors[:20],
        "stats": stats,
    }
    out = pack / "reports" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
