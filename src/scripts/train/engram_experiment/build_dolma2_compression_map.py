"""Generate the sealed Dolma2 token-equivalence map used by Engram.

This is an offline maintainer tool. Training loads the committed artifact and
does not contact Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, normalizers

TOKENIZER_ID = "allenai/dolma2-tokenizer"
TOKENIZER_REVISION = "5292e5d6c0f40b67cc765fe41bec991cf4345b5c"
TOKENIZER_VOCAB_SIZE = 100_278
PADDED_VOCAB_SIZE = 100_352
NORMALIZATION_VERSION = "engram-nfkc-nfd-lower-whitespace-v1"


def build_normalizer():
    sentinel = "\uE000"
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )


def build_compression_map(tokenizer: Tokenizer) -> tuple[list[int], int]:
    if tokenizer.get_vocab_size(with_added_tokens=True) != TOKENIZER_VOCAB_SIZE:
        raise RuntimeError(
            "Dolma2 tokenizer size changed: "
            f"expected {TOKENIZER_VOCAB_SIZE}, got "
            f"{tokenizer.get_vocab_size(with_added_tokens=True)}"
        )

    normalizer = build_normalizer()
    key_to_compressed_id: dict[str, int] = {}
    compression_map: list[int] = []
    for token_id in range(TOKENIZER_VOCAB_SIZE):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        token = tokenizer.id_to_token(token_id)
        if token is None:
            raise RuntimeError(f"tokenizer has no token for id {token_id}")
        if "\ufffd" in text:
            key = token
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        compressed_id = key_to_compressed_id.setdefault(
            key,
            len(key_to_compressed_id),
        )
        compression_map.append(compressed_id)

    canonical_vocab_size = len(key_to_compressed_id)
    # Padded matrix rows are not tokenizer entries. Keep each one distinct.
    compression_map.extend(
        range(
            canonical_vocab_size,
            canonical_vocab_size + PADDED_VOCAB_SIZE - TOKENIZER_VOCAB_SIZE,
        )
    )
    return compression_map, canonical_vocab_size


def write_artifacts(output: Path) -> dict[str, object]:
    tokenizer_path = hf_hub_download(
        TOKENIZER_ID,
        "tokenizer.json",
        revision=TOKENIZER_REVISION,
        token=False,
    )
    compression_map, canonical_vocab_size = build_compression_map(
        Tokenizer.from_file(tokenizer_path)
    )
    payload = struct.pack(f"<{len(compression_map)}I", *compression_map)
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {
        "artifact": output.name,
        "sha256": digest,
        "dtype": "uint32",
        "byte_order": "little",
        "entries": len(compression_map),
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "padded_vocab_size": PADDED_VOCAB_SIZE,
        "canonical_tokenizer_vocab_size": canonical_vocab_size,
        "compressed_padded_vocab_size": max(compression_map) + 1,
        "collapsed_tokenizer_entries": TOKENIZER_VOCAB_SIZE - canonical_vocab_size,
        "normalization": NORMALIZATION_VERSION,
        "normalization_steps": [
            "NFKC",
            "NFD",
            "strip_accents",
            "lowercase",
            "collapse_whitespace",
            "preserve_single_space",
            "strip",
        ],
        "invalid_decode_fallback": "raw_token_string",
        "padded_rows": "unique",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    payload_tmp = output.with_suffix(output.suffix + ".tmp")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    payload_tmp.write_bytes(payload)
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload_tmp.replace(output)
    metadata_tmp.replace(metadata_path)
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/olmo_core/nn/memory/dolma2_compression_map.u32"),
    )
    opts = parser.parse_args(argv)
    metadata = write_artifacts(opts.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
