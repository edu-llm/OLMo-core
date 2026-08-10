#!/usr/bin/env python3
"""Resolve portable generation-input templates into absolute-path manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATES = ARCHIVE_ROOT / "templates/generation-inputs"

PLACEHOLDERS = (
    "{{PYTHON}}",
    "{{P3_CORPUS_ROOT}}",
    "{{P3_SOURCES_ROOT}}",
    "{{P3_WORK_ROOT}}",
)


def substitute(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace(key, value)
    if any(token in text for token in PLACEHOLDERS):
        unresolved = [token for token in PLACEHOLDERS if token in text]
        raise SystemExit(f"unresolved placeholders in manifest text: {unresolved}")
    return text


def walk_substitute(value, values: dict[str, str]):
    if isinstance(value, str):
        return substitute(value, values)
    if isinstance(value, list):
        return [walk_substitute(item, values) for item in value]
    if isinstance(value, dict):
        return {key: walk_substitute(item, values) for key, item in value.items()}
    return value


def materialize(
    templates_dir: Path,
    out_dir: Path,
    *,
    corpus_root: Path,
    sources_root: Path,
    work_root: Path,
    python_executable: str,
) -> None:
    values = {
        "{{PYTHON}}": python_executable,
        "{{P3_CORPUS_ROOT}}": str(corpus_root.resolve()),
        "{{P3_SOURCES_ROOT}}": str(sources_root.resolve()),
        "{{P3_WORK_ROOT}}": str(work_root.resolve()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(templates_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        resolved = walk_substitute(payload, values)
        target = out_dir / path.name
        target.write_text(
            json.dumps(resolved, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATES,
        help="Directory containing portable manifest templates",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for resolved generation-input manifests",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=ARCHIVE_ROOT,
        help="P3 corpus archive root (scripts, tokenizers, manifests)",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path(os.environ.get("P3_SOURCES_ROOT", "/tmp/p3-sources")),
        help="Verified upstream sources root from bootstrap_sources.py",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("P3_WORK_ROOT", "/tmp/p3-rebuild-work")),
        help="Persistent work root (ATP derived artifacts, generation work)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used in builder argv",
    )
    args = parser.parse_args()

    materialize(
        args.templates.resolve(),
        args.out.resolve(),
        corpus_root=args.corpus_root.resolve(),
        sources_root=args.sources_root.resolve(),
        work_root=args.work_root.resolve(),
        python_executable=args.python,
    )
    print("MATERIALIZE_GENERATION_INPUTS_OK")


if __name__ == "__main__":
    main()
