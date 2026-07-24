#!/usr/bin/env python3
"""Create the reproducible random-init tiny OLMo branch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, OlmoConfig, OlmoForCausalLM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--tokenizer", default="allenai/OLMo-1B-hf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("OLMo tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = OlmoConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=4096,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        tie_word_embeddings=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    model = OlmoForCausalLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "architecture": "OlmoForCausalLM",
        "description": "random-init OLMo tiny-scale schedule-test checkpoint",
        "parameter_count": parameter_count,
        "seed": args.seed,
        "tokenizer_source": args.tokenizer,
        "files": files,
    }
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
