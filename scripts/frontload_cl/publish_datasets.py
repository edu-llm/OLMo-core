#!/usr/bin/env python3
"""Publish frontload-cl pretrain + SFT datasets to edullm-landing.

``pretrain/frontload-cl-10b/v1`` and ``sft/frontload-cl-chat-sft/v1`` are already
validated on ``s3://edullm-data``. Re-running this script allocates the next version
(``v2``, …) — only do that for intentional rebuilds.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

ROOT = Path(__file__).resolve().parents[2]
PT_SRC = ROOT / "data" / "frontload-cl-publish" / "pretrain" / "frontload-cl-10b"
SFT_SRC = ROOT / "data" / "frontload-cl-publish" / "sft" / "frontload-cl-chat-sft"


def publish_sft() -> None:
    print(f"publishing SFT from {SFT_SRC}", flush=True)
    plan = publish(
        SFT_SRC,
        dataset_id="sft/frontload-cl-chat-sft",
        purpose=(
            "Shared 1-epoch chat+math SFT mix (UltraChat, Numina, OpenHermes, no_robots) "
            "for both frontload-cl 370M PT arms, to surface GSM8K/ARC/IFEval differences "
            "without confounding post-training"
        ),
        profile="sft-conversations/v1",
        s3=Boto3S3.default(),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        about=(
            "Conversation JSONL for the shared post-training stage of the frontload-cl "
            "early-behavior-primer experiment. Built from no_robots, UltraChat train_sft, "
            "NuminaMath-1.5 (250k), and OpenHermes-2.5 (100k) with seed 42069666."
        ),
        sources=[
            {
                "name": "HuggingFaceH4/no_robots",
                "scope": "subset",
                "uri": "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
            },
            {
                "name": "HuggingFaceH4/ultrachat_200k",
                "scope": "subset",
                "uri": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
            },
            {
                "name": "AI-MO/NuminaMath-1.5",
                "scope": "subset",
                "uri": "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5",
            },
            {
                "name": "brahmairesearch/OpenHermes-2.5-Formatted",
                "scope": "subset",
                "uri": "https://huggingface.co/datasets/brahmairesearch/OpenHermes-2.5-Formatted",
            },
        ],
        group_meta={
            "conversations": {
                "record_schema": {"messages": [{"role": "str", "content": "str"}]},
                "partitions": [
                    {"name": "train", "by": "path", "glob": "train-*.jsonl.gz"},
                    {"name": "val", "by": "path", "glob": "val-*.jsonl.gz"},
                ],
                "dedup": {"method": "sha256-content"},
                "leakage": {"reported_overlap": 0},
            }
        },
    )
    print("SFT plan:", plan, flush=True)


def _token_counts() -> dict[str, int]:
    root = PT_SRC / "tokens"
    out: dict[str, int] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n = 0
        for f in d.glob("*.u32le.bin"):
            n += f.stat().st_size // 4
        out[d.name] = n
    return out


PT_STAGING = "s3://edullm-landing/_staging/pretrain/frontload-cl-10b"


def publish_pretrain(*, from_staging: bool = False) -> None:
    counts = _token_counts()
    source: str | Path = PT_STAGING if from_staging else PT_SRC
    print("token counts:", counts, flush=True)
    print(f"publishing pretrain from {source}", flush=True)
    sources = [
        {
            "name": "HuggingFaceFW/fineweb-edu",
            "tokens": counts.get("fineweb-edu-main", 0) + counts.get("fineweb-edu-anneal", 0),
            "scope": "subset",
            "uri": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
        },
        {
            "name": "FineWiki",
            "tokens": counts.get("finewiki", 0),
            "scope": "subset",
            "uri": "https://huggingface.co/datasets/HuggingFaceFW/finewiki",
        },
        {
            "name": "HuggingFaceTB/cosmopedia",
            "tokens": counts.get("cosmopedia-v2", 0),
            "scope": "subset",
            "uri": "https://huggingface.co/datasets/HuggingFaceTB/cosmopedia",
        },
        {
            "name": "HuggingFaceTB/finemath",
            "tokens": counts.get("finemath-4plus", 0),
            "scope": "subset",
            "uri": "https://huggingface.co/datasets/HuggingFaceTB/finemath",
        },
        {
            "name": "OpenHermes-2.5 (PT plain-text remainder)",
            "tokens": counts.get("openhermes-pt", 0),
            "scope": "subset",
        },
        {
            "name": "NaturalReasoning",
            "tokens": counts.get("natural-reasoning", 0),
            "scope": "subset",
        },
    ]
    plan = publish(
        source,
        dataset_id="pretrain/frontload-cl-10b",
        purpose=(
            "10B-token HQ+SFT-like Dolma2 pretrain mix for OLMo2-370M frontload-cl arms, "
            "to decide whether early SFT-like timing beats flat mixing on GSM8K/ARC/IFEval "
            "after shared SFT"
        ),
        profile="pretrain-tokens/v1",
        tokenizer="tokenizer/dolma2-bpe",
        s3=Boto3S3.default(),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hash_workers=8,
        copy_workers=8,
        about=(
            "Packed uint32 Dolma2 tokens for the frontload-cl early-behavior-primer "
            "experiment: FineWeb-Edu main+anneal, FineWiki, Cosmopedia-v2, FineMath-4plus, "
            "OpenHermes-PT, and Natural Reasoning. Schedules (primer vs flat) live in the "
            "train script, not in this corpus."
        ),
        sources=sources,
        notes=(
            "Val shards were carved from the ends of packed train shards after tokenization "
            "(seed 42069666 for source sampling). OpenHermes-PT is disjoint from the SFT "
            "100k draw. FineWeb totals may slightly exceed nominal budgets due to "
            "per-worker packing."
        ),
    )
    print("pretrain plan:", plan, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["sft", "pretrain", "both"])
    p.add_argument(
        "--from-staging",
        action="store_true",
        help="Resume pretrain from s3://edullm-landing/_staging/... (skip local re-upload)",
    )
    args = p.parse_args()
    if args.which in ("sft", "both"):
        publish_sft()
    if args.which in ("pretrain", "both"):
        publish_pretrain(from_staging=args.from_staging)


if __name__ == "__main__":
    main()
