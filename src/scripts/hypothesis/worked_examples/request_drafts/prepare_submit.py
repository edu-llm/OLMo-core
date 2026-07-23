#!/usr/bin/env python3
"""Validate the four arm drafts with the submit-edullm-job adapter (when allowlisted).

Exits non-zero if policy on this checkout still lacks ``worked-examples-cpt``.
Does **not** create GitHub Issues (that remains `/submit-edullm-job` + human confirm).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requester", required=True)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[5]
    adapter = repo / ".claude" / "skills" / "submit-edullm-job" / "scripts" / "validate_request.py"
    drafts = Path(__file__).resolve().parent
    failed = 0
    for path in sorted(drafts.glob("arm_*.json")):
        print(f"=== validate {path.name} ===")
        proc = subprocess.run(
            [
                sys.executable,
                str(adapter),
                "--input-json",
                str(path),
                "--requester",
                args.requester,
            ],
            cwd=str(repo),
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(repo / "src")},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failed += 1
            print(proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")
        else:
            print("OK")
    if failed:
        print(
            f"\n{failed} draft(s) failed validation. Typical cause: "
            "worked-examples-cpt not allowlisted on this checkout / main yet. "
            "See ../OPERATOR_ALLOWLIST.md"
        )
        return 2
    print("\nAll four drafts validate. Run /submit-edullm-job per arm with confirmation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
