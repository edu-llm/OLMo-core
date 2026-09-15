#!/usr/bin/env python3
"""Build the n-gram index from the dumped eval items (v4 sections 3 and 6.5).

One loose, provenance-tagged index: every indexed n-gram records which field
it came from (`stem` or `gold`), so the nested subsets S_gold / S_gold_or_stem
/ S_any are formed at analysis time from a single scan rather than by building
several indices.

Length floor, three outcomes, applied per field against that field's own text:

    >= 13 words   indexed at n = 13 (every window)
    8-12 words    indexed at n = field length (exact whole-string match)
    < 8 words     NOT ASSESSABLE -- excluded, counted, reported

Structure. A prefilter keyed on the first 8 words, mapping to the set of
full widths that begin with that prefix:

    prefix8 -> {13, 9, ...}
    exact[full_ngram_string] -> [(item_row, field_id, width), ...]

The prefilter is deliberately MULTI-valued. Eval stems are heavily
boilerplated, so distinct keys of different widths share 8-word prefixes; a
single-valued prefix map would silently and permanently drop every key but
one, and the loss would be item-specific and invisible. `n_prefix_targets`
is asserted equal to the number of distinct widths summed over prefixes.

Keys are the n-gram STRINGS, not hashes. That removes PYTHONHASHSEED from the
picture by construction and removes hash collisions as a source of false
positives, at the cost of memory -- which is reported here so the scan can be
sized against it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("build_item_index")

FULL_WIDTH = 13
MIN_WIDTH = 8
FIELDS = ("stem", "gold")
FIELD_ID = {name: i for i, name in enumerate(FIELDS)}


def ngrams(words: list[str], width: int) -> list[str]:
    if len(words) < width:
        return []
    return [" ".join(words[i : i + width]) for i in range(len(words) - width + 1)]


def keys_for_field(text: str) -> tuple[list[str], int]:
    """Return (index keys, width). Empty keys means NOT ASSESSABLE."""
    words = text.split()
    n = len(words)
    if n < MIN_WIDTH:
        return [], 0
    if n < FULL_WIDTH:
        return [" ".join(words)], n
    return ngrams(words, FULL_WIDTH), FULL_WIDTH


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="eval_items.jsonl.gz from section 4a")
    parser.add_argument("--out", required=True, help="output index pickle")
    parser.add_argument("--summary", required=True, help="output summary .json")
    args = parser.parse_args()

    exact: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    prefix: dict[str, set[int]] = defaultdict(set)
    items: list[dict] = []
    per_label: dict[str, dict] = {}

    with gzip.open(args.items, "rt", encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            rec = json.loads(line)
            label = rec["label"]
            stats = per_label.setdefault(
                label,
                {
                    "label": label,
                    "n_items": 0,
                    "stem_not_assessable": 0,
                    "gold_not_assessable": 0,
                    "stem_keys": 0,
                    "gold_keys": 0,
                    "items_with_lt3_keys": 0,
                },
            )
            stats["n_items"] += 1
            items.append(
                {
                    "row": row_index,
                    "label": label,
                    "doc_id": rec["doc_id"],
                    "stem_sha256": rec["stem_sha256"],
                    "gold_sha256": rec["gold_sha256"],
                }
            )

            total_keys = 0
            for field in FIELDS:
                field_keys, width = keys_for_field(rec[field])
                if not field_keys:
                    stats[f"{field}_not_assessable"] += 1
                    continue
                stats[f"{field}_keys"] += len(field_keys)
                total_keys += len(field_keys)
                for key in field_keys:
                    exact[key].append((row_index, FIELD_ID[field], width))
                    prefix[" ".join(key.split()[:MIN_WIDTH])].add(width)
            if total_keys < 3:
                stats["items_with_lt3_keys"] += 1

    # v4 section 6.5: the prefilter must be multi-valued or keys are lost.
    n_prefix_targets = sum(len(v) for v in prefix.values())
    n_multi = sum(1 for v in prefix.values() if len(v) > 1)

    index = {
        "exact": dict(exact),
        "prefix": {k: sorted(v) for k, v in prefix.items()},
        "items": items,
        "full_width": FULL_WIDTH,
        "min_width": MIN_WIDTH,
        "fields": FIELDS,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "n_items": len(items),
        "n_exact_keys": len(exact),
        "n_prefixes": len(prefix),
        "n_prefix_targets": n_prefix_targets,
        "n_prefixes_multivalued": n_multi,
        "index_bytes_on_disk": out.stat().st_size,
        "labels": sorted(per_label.values(), key=lambda s: s["label"]),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=1), encoding="utf-8")

    log.info("items                : %d", len(items))
    log.info("distinct exact keys  : %d", len(exact))
    log.info("distinct 8-w prefixes: %d", len(prefix))
    log.info("prefix targets       : %d  (multivalued prefixes: %d)", n_prefix_targets, n_multi)
    log.info("index on disk        : %.1f MB", out.stat().st_size / 2**20)
    log.info("")
    log.info(
        "%-46s %7s %9s %9s %9s", "label", "items", "stemNA", "goldNA", "<3keys"
    )
    for stats in sorted(per_label.values(), key=lambda s: s["label"]):
        log.info(
            "%-46s %7d %9d %9d %9d",
            stats["label"],
            stats["n_items"],
            stats["stem_not_assessable"],
            stats["gold_not_assessable"],
            stats["items_with_lt3_keys"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
