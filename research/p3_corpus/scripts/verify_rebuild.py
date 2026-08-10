#!/usr/bin/env python3
"""Compare a rebuilt tokenized/publish tree against expected-release-v3.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = ARCHIVE_ROOT / "expected-release-v3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_u32le_tokens(path: Path) -> int:
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"{path} size {size} is not u32-aligned")
    return size // 4


def load_meta(root: Path, split: str) -> dict:
    meta_path = root / f"{split}_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing completion manifest: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def verify_tokenizer_seal(tokenizer_root: Path, expected: dict) -> list[str]:
    errors: list[str] = []
    seal = expected["tokenizer"]
    files = {
        "tokenizer.json": seal.get("tokenizer_json_sha256"),
        "tokenizer_config.json": seal.get("tokenizer_config_sha256"),
    }
    for name, digest in files.items():
        if digest is None:
            continue
        path = tokenizer_root / name
        if not path.exists():
            errors.append(f"missing tokenizer file {path}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            errors.append(f"tokenizer {name}: expected {digest}, got {actual}")
    return errors


def verify_tokenized_root(root: Path, expected: dict) -> list[str]:
    errors: list[str] = []
    for split in ("train", "val"):
        meta = load_meta(root, split)
        exp = expected["packed_tokens"][split]
        if meta.get("sequence_length") != expected["training_constants"]["sequence_length"]:
            errors.append("sequence_length mismatch in meta")
        if meta.get("separator_ids") != expected["tokenizer"]["separator_ids"]:
            errors.append("separator_ids mismatch in meta")
        if meta.get("instances") != exp["instances"]:
            errors.append(
                f"{split} instances: expected {exp['instances']}, got {meta.get('instances')}"
            )
        if meta.get("real_tokens") != exp["real_tokens"]:
            errors.append(
                f"{split} real_tokens: expected {exp['real_tokens']}, got {meta.get('real_tokens')}"
            )
        if meta.get("dropped_over_length") != exp["dropped_over_length"]:
            errors.append(f"{split} dropped_over_length mismatch")
        composite = meta.get("tokenizer_composite_sha256") or meta.get(
            "tokenizer_seal", {}
        ).get("tokenizer_composite_sha256")
        if composite != expected["tokenizer"]["composite_sha256"]:
            errors.append("tokenizer composite seal mismatch")
        total_tokens = 0
        for group in meta.get("groups", {}).values():
            for shard in group.get("shards", []):
                rel = shard["path"]
                shard_path = root / rel
                if not shard_path.exists():
                    errors.append(f"missing shard {shard_path}")
                    continue
                tokens = count_u32le_tokens(shard_path)
                total_tokens += tokens
                if tokens != shard["tokens"]:
                    errors.append(f"shard token count mismatch for {rel}")
                actual_digest = sha256_file(shard_path)
                if actual_digest != shard["sha256"]:
                    errors.append(f"shard digest mismatch for {rel}")
        reader_key = "reader_rows"
        if total_tokens != exp[reader_key]:
            errors.append(
                f"{split} packed tokens: expected {exp[reader_key]}, got {total_tokens}"
            )
    return errors


def verify_publish_root(root: Path, expected: dict) -> list[str]:
    errors: list[str] = []
    if not root.exists():
        return [f"publish root missing: {root}"]
    bins = sorted(root.rglob("*.u32le.bin"))
    if not bins:
        return [f"no .u32le.bin payloads under {root}"]
    total_bytes = sum(path.stat().st_size for path in bins)
    exp = expected["packed_tokens"]["published"]
    if len(bins) != exp["objects"]:
        errors.append(f"publish object count: expected {exp['objects']}, got {len(bins)}")
    if total_bytes != exp["bytes"]:
        errors.append(f"publish bytes: expected {exp['bytes']}, got {total_bytes}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="Canonical expected-release-v3.json",
    )
    parser.add_argument(
        "--tokenized-root",
        type=Path,
        required=True,
        help="Directory containing train_meta.json / val_meta.json and token shards",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        default=None,
        help="Optional publish-stage directory of .u32le.bin payloads only",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ARCHIVE_ROOT / "tokenizers/qwen25-vendored",
        help="Vendored tokenizer directory",
    )
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    errors: list[str] = []
    errors.extend(verify_tokenizer_seal(args.tokenizer.resolve(), expected))
    errors.extend(verify_tokenized_root(args.tokenized_root.resolve(), expected))
    if args.publish_root is not None:
        errors.extend(verify_publish_root(args.publish_root.resolve(), expected))

    if errors:
        print("VERIFY_REBUILD_FAILED")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(1)

    print("VERIFY_REBUILD_OK")


if __name__ == "__main__":
    main()
