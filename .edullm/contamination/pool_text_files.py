#!/usr/bin/env python3
"""Resolve which TEXT files back a mixlaw pool, per domain.

The pool is tokenized `.npy` shards, so it cannot be scanned for n-grams
directly. Two files together identify the text behind it:

  pool/plan/selected_shards.json   domain -> [{path, tokens}]  (what was drawn)
  plan/tokenized_manifest.json     shard path -> {manifest_path, domain, tokens}

Scanning the whole olmo-mix text tree instead would be both more expensive and
less correct: `olmo-mix-upsample` contains UPSAMPLED duplicates, so any rate
computed over it double-counts whatever was duplicated. The selected-shard set
is what the arms actually draw from.

Emits a JSON file list per domain plus the token totals, so the scan can be
sharded and the denominator reported honestly.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("pool_text_files")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True, help="pool/plan/selected_shards.json")
    parser.add_argument("--manifest", required=True, help="plan/tokenized_manifest.json")
    parser.add_argument("--text-root", required=True, help="dir holding data/<domain>/train/...")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_shard = {s["path"]: s for s in manifest["shards"]}
    root = Path(args.text_root)

    out: dict[str, dict] = {}
    unresolved: list[str] = []
    missing_on_disk: list[str] = []

    for domain, shards in selected.items():
        files: dict[str, int] = {}
        tokens = 0
        for entry in shards:
            path = entry["path"]
            rec = by_shard.get(path)
            if rec is None:
                unresolved.append(path)
                continue
            if rec["domain"] != domain:
                raise RuntimeError(
                    f"{path}: manifest says domain {rec['domain']!r}, "
                    f"selected under {domain!r}"
                )
            tokens += int(entry["tokens"])
            # manifest_path is relative to the text root, e.g.
            # data/arxiv/train/arxiv-train-0007.json.gz
            text = root / rec["manifest_path"]
            files[str(text)] = files.get(str(text), 0) + int(entry["tokens"])
        present = {}
        for f, tok in files.items():
            if Path(f).is_file():
                present[f] = tok
            else:
                missing_on_disk.append(f)
        out[domain] = {
            "n_shards_selected": len(shards),
            "n_text_files": len(present),
            "selected_tokens": tokens,
            "files": sorted(present),
        }

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} selected shards are absent from the tokenized "
            f"manifest, so their text cannot be identified and the scan would "
            f"silently cover a subset (first: {unresolved[0]})"
        )

    log.info("%-18s %9s %9s %16s", "domain", "shards", "files", "selected tokens")
    total_tokens = 0
    total_files = 0
    for domain in sorted(out):
        d = out[domain]
        total_tokens += d["selected_tokens"]
        total_files += d["n_text_files"]
        log.info(
            "%-18s %9d %9d %16s",
            domain,
            d["n_shards_selected"],
            d["n_text_files"],
            f"{d['selected_tokens']:,}",
        )
    log.info("%-18s %9s %9d %16s", "TOTAL", "", total_files, f"{total_tokens:,}")
    if missing_on_disk:
        log.warning(
            "%d resolved text files are not on disk and were excluded; "
            "first: %s",
            len(missing_on_disk),
            missing_on_disk[0],
        )
        out["_missing_on_disk"] = missing_on_disk[:50]  # type: ignore[assignment]

    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
