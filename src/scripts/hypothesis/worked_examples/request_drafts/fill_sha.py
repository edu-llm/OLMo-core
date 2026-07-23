#!/usr/bin/env python3
"""Stamp Commit SHA in arm_*.json from current HEAD (must be pushed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(sha) != 40:
        raise SystemExit(f"bad sha: {sha}")
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob("arm_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["Commit SHA"] = sha
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"updated {path.name} -> {sha}")


if __name__ == "__main__":
    main()
