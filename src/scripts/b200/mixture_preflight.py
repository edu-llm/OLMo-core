"""
Preflight a ``SourceMixtureList`` config against what is actually in the bucket.

``SourceMixtureDatasetConfig.build()`` counts tokens by opening every resolved path, then
raises ``Insufficient tokens for source`` if any source cannot meet its target ratio. On the
150B dolma2 mix that is a pass over ~9k objects before the failure surfaces, which is a bad
trade when the answer is already determined by object sizes.

This script computes the same feasibility answer from S3 ``ListObjectsV2`` metadata -- a
handful of calls instead of thousands of reads -- and reports the largest token budget the
config can actually serve, plus which source binds it.

Usage::

    python src/scripts/b200/mixture_preflight.py \\
        --config s3://edullm-datasets/olmo-150b-dolma2/configs/scaled-weighting-config.yaml \\
        --requested-tokens 7_430_000_000

Exits non-zero when the requested budget is not satisfiable.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob as globlib
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

from olmo_core.data import TokenizerConfig
from olmo_core.data.source_mixture import SourceMixtureConfig, SourceMixtureList

# A .npy v1.0 header is padded to a 64-byte boundary; 128 covers every shard in this mix.
NPY_HEADER_BYTES = 128


@dataclass
class SourceAvailability:
    """What a single source can actually supply, measured from object metadata."""

    name: str
    num_files: int
    num_bytes: int
    available_tokens: int
    target_ratio: float
    max_source_fraction: float
    max_repetition_ratio: float

    @property
    def usable_tokens(self) -> int:
        """Mirrors ``max_for_source`` in :meth:`SourceMixtureDatasetConfig.build`."""
        return int((self.available_tokens * self.max_source_fraction) * self.max_repetition_ratio)

    @property
    def budget_cap(self) -> float:
        """Largest overall budget this source can sustain at its target ratio."""
        if self.target_ratio <= 0:
            return float("inf")
        return self.usable_tokens / self.target_ratio

    def needed_tokens(self, requested_tokens: int) -> int:
        return int(requested_tokens * self.target_ratio)


def token_itemsize(tokenizer: TokenizerConfig) -> int:
    """Bytes per token, matching how the dataset picks its numpy dtype from vocab size."""
    return 2 if tokenizer.vocab_size < 2**16 else 4


def _iter_object_sizes(pattern: str) -> Iterator[Tuple[str, int]]:
    """Yield ``(path, size)`` for every object matching a local or ``s3://`` glob."""
    parsed = urlparse(str(pattern))

    if parsed.scheme != "s3":
        for path in globlib.glob(str(pattern)):
            yield path, os.path.getsize(path)
        return

    import boto3

    bucket = parsed.netloc
    key_glob = parsed.path.lstrip("/")
    # Everything before the first wildcard is a literal prefix, which is what makes this cheap.
    prefix = key_glob.split("*", 1)[0]

    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if fnmatch.fnmatch(obj["Key"], key_glob):
                yield f"s3://{bucket}/{obj['Key']}", obj["Size"]


def measure_source(source: SourceMixtureConfig, itemsize: int) -> SourceAvailability:
    num_files = 0
    num_bytes = 0
    tokens = 0
    for _, size in (pair for pattern in source.paths for pair in _iter_object_sizes(pattern)):
        num_files += 1
        num_bytes += size
        tokens += max(0, size - NPY_HEADER_BYTES) // itemsize

    return SourceAvailability(
        name=source.source_name,
        num_files=num_files,
        num_bytes=num_bytes,
        available_tokens=tokens,
        target_ratio=source.target_ratio,
        max_source_fraction=source.max_source_fraction,
        max_repetition_ratio=source.max_repetition_ratio,
    )


def load_raw_config(config_path: str) -> Dict:
    from cached_path import cached_path

    with cached_path(config_path).open() as f:
        return yaml.safe_load(f)


def drop_sources(raw: Dict, names: List[str]) -> Dict:
    """
    Remove sources by name and renormalize the remaining target ratios back to 1.0.

    Useful when one small source binds the whole token budget: dropping it costs only its own
    share of the mixture but can lift the achievable budget by an order of magnitude.
    """
    unknown = set(names) - {s["source_name"] for s in raw["sources"]}
    if unknown:
        raise ValueError(f"unknown source(s): {sorted(unknown)}")

    kept = [s for s in raw["sources"] if s["source_name"] not in names]
    if not kept:
        raise ValueError("cannot drop every source")

    remaining = sum(s["target_ratio"] for s in kept)
    for source in kept:
        source["target_ratio"] = source["target_ratio"] / remaining

    raw["sources"] = kept
    return raw


def rewrite_paths(raw: Dict, local_root: str) -> Dict:
    """
    Point ``s3://`` globs at a local staging root instead.

    The bucket layout is preserved beneath ``local_root`` so that a plain ``aws s3 sync`` of
    the prefix lands exactly where the rewritten config expects it.
    """
    for source in raw["sources"]:
        source["paths"] = [
            os.path.join(local_root, urlparse(str(p)).path.lstrip("/"))
            if urlparse(str(p)).scheme == "s3"
            else p
            for p in source["paths"]
        ]
    return raw


def _fmt_tokens(tokens: float) -> str:
    return f"{tokens / 1e9:.3f}B" if abs(tokens) >= 1e9 else f"{tokens / 1e6:.1f}M"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local path or s3:// URL of the YAML.")
    parser.add_argument("--requested-tokens", type=int, default=7_430_000_000)
    parser.add_argument("--global-batch-size", type=int, default=2_097_152)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument(
        "--drop-source",
        action="append",
        default=[],
        metavar="NAME",
        help="Drop a source and renormalize the remaining ratios. Repeatable.",
    )
    parser.add_argument(
        "--rewrite-root",
        default=None,
        help="Emit a copy of the config with s3:// paths rebased onto this local directory.",
    )
    parser.add_argument("--output", default=None, help="Where to write the resulting config.")
    args = parser.parse_args()

    tokenizer = TokenizerConfig.dolma2()
    itemsize = token_itemsize(tokenizer)

    raw = load_raw_config(args.config)
    if args.drop_source:
        raw = drop_sources(raw, args.drop_source)
    if args.rewrite_root:
        raw = rewrite_paths(raw, args.rewrite_root)

    source_list = SourceMixtureList.from_dict(raw)
    source_list.validate()

    if args.global_batch_size % args.sequence_length != 0:
        print(
            f"FAIL  global_batch_size ({args.global_batch_size}) is not a multiple of "
            f"sequence_length ({args.sequence_length}); build() asserts on this.",
            file=sys.stderr,
        )
        return 1

    print(f"config      {args.config}")
    print(f"tokenizer   dolma2, vocab {tokenizer.vocab_size} -> {itemsize} bytes/token")
    print(f"requested   {_fmt_tokens(args.requested_tokens)} tokens")
    print(
        f"batching    global_batch_size {args.global_batch_size} / sequence_length "
        f"{args.sequence_length} = {args.global_batch_size // args.sequence_length} seq/batch, "
        f"{-(-args.requested_tokens // args.global_batch_size)} steps"
    )
    print()

    measured: List[SourceAvailability] = [
        measure_source(source, itemsize) for source in source_list.sources
    ]

    header = (
        f"{'source':24}{'files':>7}{'GiB':>9}{'available':>11}{'ratio':>9}{'needed':>10}  verdict"
    )
    print(header)
    print("-" * len(header))

    shortfalls: List[SourceAvailability] = []
    for item in measured:
        needed = item.needed_tokens(args.requested_tokens)
        ok = item.usable_tokens >= needed
        if not ok:
            shortfalls.append(item)
        print(
            f"{item.name:24}{item.num_files:7d}{item.num_bytes / 2**30:9.1f}"
            f"{_fmt_tokens(item.available_tokens):>11}{item.target_ratio:9.4%}"
            f"{_fmt_tokens(needed):>10}  {'ok' if ok else 'SHORT'}"
        )

    binding = min(measured, key=lambda s: s.budget_cap)
    total_tokens = sum(s.available_tokens for s in measured)
    total_bytes = sum(s.num_bytes for s in measured)

    print()
    print(f"corpus      {total_tokens / 1e9:.1f}B tokens across {total_bytes / 2**30:.1f} GiB")
    print(f"max budget  {_fmt_tokens(binding.budget_cap)} tokens, bound by '{binding.name}'")
    print(
        f"staging     ~{args.requested_tokens * itemsize / 2**30:.1f} GiB for the requested budget"
    )

    if shortfalls:
        print()
        print("INFEASIBLE. This config cannot serve the requested budget.")
        for item in shortfalls:
            print(
                f"  {item.name}: has {_fmt_tokens(item.usable_tokens)} usable, "
                f"needs {_fmt_tokens(item.needed_tokens(args.requested_tokens))}"
            )
        print()
        print("Options, cheapest first:")
        print(f"  1. Lower --requested-tokens to {_fmt_tokens(binding.budget_cap)} or below.")
        print(
            f"  2. Raise max_repetition_ratio for '{binding.name}' to "
            f">= {args.requested_tokens * binding.target_ratio / max(1, binding.usable_tokens):.2f} "
            "to upsample it."
        )
        print(f"  3. Drop '{binding.name}' and renormalize the remaining target ratios.")
        return 1

    print()
    print("FEASIBLE.")

    if args.output:
        with open(args.output, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        print(f"wrote config -> {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
