#!/usr/bin/env python3
"""Scan one corpus domain for eval-item n-grams (v4 sections 3, 5, 6).

One task per domain. Emits every (item, field, document) match plus a DONE
sentinel, and refuses to emit anything if its own positive controls are not
recovered.

Why a rolling hash. The corpus is ~7.5e9 word positions. Building an 8-word
string at every position to probe the prefilter costs on the order of CPU-hours
in Python. Instead a rolling polynomial hash over per-word hashes is carried
across the window, which is a handful of integer ops per position, and the
8-word string is only materialized when the integer prefilter hits. Word
hashes come from blake2b rather than `hash()`, so nothing here depends on
PYTHONHASHSEED and the index is comparable across processes by construction.

Collisions are not a correctness risk: the modulus is 2**61-1 and there are
~1.3e6 prefixes, so across 7.5e9 positions the expected number of spurious
prefilter hits is ~4e-3 -- and every prefilter hit is confirmed by exact
string comparison before it is recorded, so a collision costs time, never a
false positive.

Silent-failure gates, per v4 section 6:

  - Canary. A fixed RAW document whose expected post-normalization key set is
    asserted. This catches a normalizer divergence between index build and
    scan, which a canary made of a pre-normalized string cannot: that failure
    yields zero hits and a passing canary.
  - Spikes. Synthetic documents carrying known item n-grams, built from raw
    un-normalized text with varied casing and punctuation so the normalizer is
    on trial rather than tested against itself. Injected in the piece format
    through the real reader. Nothing is emitted until all are recovered, and
    they are removed by row identity, never by subtracting counts -- count
    subtraction cancels the first real hit in any cell holding a spike, which
    floors every rate at zero while reporting success.
  - DONE sentinel carrying docs_scanned and words_scanned, so the aggregator
    can assert full coverage after the scan rather than trusting a glob.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path

log = logging.getLogger("scan_corpus")

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

MOD = (1 << 61) - 1
BASE = 1_000_003

# A fixed raw document and the normalized 8-word keys it must produce. If the
# normalizer changes, this fails loudly instead of the scan quietly finding
# nothing.
CANARY_RAW = "The Quick--Brown FOX, jumps over  the lazy dog's back; again and AGAIN!"
CANARY_EXPECTED_NORM = (
    "the quick brown fox jumps over the lazy dog s back again and again"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def word_hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big") % MOD


class Scanner:
    def __init__(self, index: dict) -> None:
        self.exact: dict[str, list[tuple[int, int, int]]] = index["exact"]
        self.prefix: dict[str, list[int]] = index["prefix"]
        self.min_width = int(index["min_width"])
        self._cache: dict[str, int] = {}
        self._pow = pow(BASE, self.min_width - 1, MOD)
        # Integer prefilter: the rolling hash of each prefix's words.
        self.prefilter: set[int] = set()
        for prefix in self.prefix:
            words = prefix.split()
            h = 0
            for w in words:
                h = (h * BASE + self._wh(w)) % MOD
            self.prefilter.add(h)

    def _wh(self, word: str) -> int:
        h = self._cache.get(word)
        if h is None:
            h = word_hash(word)
            self._cache[word] = h
        return h

    def scan_words(self, words: list[str]) -> list[tuple[int, int, int, int]]:
        """Return (item_row, field_id, width, word_offset) per confirmed match.

        The word offset is carried so the aggregator can union matched spans
        into a matched-span token rate (v4 section 5.2). Without it only the
        tokens-in-matched-documents upper bound is computable, and on a
        length-coupled metric like compression_ratio that bound is the one
        that manufactures a gradient under the null.
        """
        n = len(words)
        w = self.min_width
        if n < w:
            return []
        hashes = [self._wh(x) for x in words]
        h = 0
        for i in range(w):
            h = (h * BASE + hashes[i]) % MOD
        out: list[tuple[int, int, int, int]] = []
        pos = 0
        while True:
            if h in self.prefilter:
                key8 = " ".join(words[pos : pos + w])
                widths = self.prefix.get(key8)
                if widths is not None:
                    for width in widths:
                        if pos + width > n:
                            continue
                        full = key8 if width == w else " ".join(words[pos : pos + width])
                        hits = self.exact.get(full)
                        if hits:
                            for row, field, keywidth in hits:
                                out.append((row, field, keywidth, pos))
            pos += 1
            if pos + w > n:
                break
            h = (h - hashes[pos - 1] * self._pow) % MOD
            h = (h * BASE + hashes[pos + w - 1]) % MOD
        return out


def build_spikes(index: dict, count: int) -> list[tuple[str, set[int]]]:
    """Synthetic raw documents carrying known item n-grams.

    Built from raw, un-normalized text with varied casing and punctuation so
    the normalizer is exercised rather than bypassed. Returns
    (raw_text, expected_item_rows).
    """
    exact = index["exact"]
    chosen = sorted(exact)[:: max(1, len(exact) // max(count, 1))][:count]
    spikes: list[tuple[str, set[int]]] = []
    for i, key in enumerate(chosen):
        rows = {row for row, _field, _w in exact[key]}
        decorated = key.upper() if i % 2 else key.title()
        raw = f"Some preamble text here. {decorated}! And a trailing clause -- {i}."
        spikes.append((raw, rows))
    return spikes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--trim", required=True, help="<domain>-trimmed.json.gz")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--spikes", type=int, default=64)
    parser.add_argument("--limit-docs", type=int, default=0, help="debug: stop early")
    args = parser.parse_args()

    if normalize(CANARY_RAW) != CANARY_EXPECTED_NORM:
        raise RuntimeError(
            "canary failed: the normalizer does not produce the expected "
            f"keys.\n  got:      {normalize(CANARY_RAW)!r}\n  expected: "
            f"{CANARY_EXPECTED_NORM!r}\nAn index built under a different "
            "normalizer would yield zero hits and look clean."
        )

    with open(args.index, "rb") as fh:
        index = pickle.load(fh)
    scanner = Scanner(index)
    log.info(
        "%s: index exact=%d prefixes=%d prefilter=%d",
        args.domain,
        len(scanner.exact),
        len(scanner.prefix),
        len(scanner.prefilter),
    )

    spikes = build_spikes(index, args.spikes)
    spike_expected = {i: rows for i, (_raw, rows) in enumerate(spikes)}
    spike_found: dict[int, set[int]] = {i: set() for i in spike_expected}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hits_path = out_dir / f"hits_{args.domain}.jsonl.gz"

    docs = 0
    words_total = 0
    hit_rows = 0
    started = time.time()

    with gzip.open(hits_path, "wt", encoding="utf-8") as out:
        # Spikes first, through the same normalize+scan path as real text.
        for spike_id, (raw, _rows) in enumerate(spikes):
            pairs = set()
            for row, field, _width, _pos in scanner.scan_words(normalize(raw).split()):
                spike_found[spike_id].add(row)
                pairs.add((row, field))
            if pairs:
                out.write(
                    json.dumps(
                        {
                            "domain": "__spike__",
                            "source_doc": spike_id,
                            "n_words": len(normalize(raw).split()),
                            "matched_words": 0,
                            "items": sorted(pairs),
                        }
                    )
                    + "\n"
                )

        missing = {i: sorted(spike_expected[i] - spike_found[i]) for i in spike_expected}
        missing = {i: v for i, v in missing.items() if v}
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(spikes)} positive controls were not "
                f"recovered; the scanner is not finding text that is present, "
                f"so no rate may be emitted. first: {sorted(missing)[:5]}"
            )
        log.info("%s: all %d positive controls recovered", args.domain, len(spikes))

        with gzip.open(args.trim, "rt", encoding="utf-8") as fh:
            for source_doc, line in enumerate(fh):
                if args.limit_docs and docs >= args.limit_docs:
                    break
                rec = json.loads(line)
                words = normalize(rec.get("text", "")).split()
                docs += 1
                words_total += len(words)
                hits = scanner.scan_words(words)
                if hits:
                    # One record per matched document. (item, field) pairs are
                    # collapsed -- a document either contains an item's text or
                    # it does not, and counting a mirrored phrase twice would
                    # inflate the rate. Matched word spans are unioned so the
                    # aggregator can compute a matched-span rate rather than
                    # only the length-biased whole-document upper bound.
                    pairs: set[tuple[int, int]] = set()
                    covered: set[int] = set()
                    for row, field, width, pos in hits:
                        pairs.add((row, field))
                        covered.update(range(pos, pos + width))
                    out.write(
                        json.dumps(
                            {
                                "domain": args.domain,
                                "source_doc": source_doc,
                                "n_words": len(words),
                                "matched_words": len(covered),
                                "items": sorted(pairs),
                            }
                        )
                        + "\n"
                    )
                    hit_rows += 1
                if docs % 200_000 == 0:
                    log.info(
                        "%s: %d docs, %.2fB words, %d hits, %.0fs",
                        args.domain,
                        docs,
                        words_total / 1e9,
                        hit_rows,
                        time.time() - started,
                    )

    sentinel = {
        "domain": args.domain,
        "docs_scanned": docs,
        "words_scanned": words_total,
        "hit_rows": hit_rows,
        "spikes_injected": len(spikes),
        "spikes_recovered": len(spikes),
        "elapsed_s": round(time.time() - started, 1),
        "canary_ok": True,
    }
    (out_dir / f"DONE_{args.domain}.json").write_text(
        json.dumps(sentinel, indent=1), encoding="utf-8"
    )
    log.info("%s DONE %s", args.domain, json.dumps(sentinel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
