"""Immutable source identities shared by P3 training and checkpoint export."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlsplit

from olmo_core.data import TokenizerConfig
from olmo_core.nn.transformer.qwen import qwen2_tokenizer_config

TOKENIZER_ARTIFACT_ID = "tokenizer/qwen25-vendored"
TOKENIZER_ARTIFACT_VERSION = "v1"
TOKENIZER_ARTIFACT = f"{TOKENIZER_ARTIFACT_ID}/{TOKENIZER_ARTIFACT_VERSION}"
TOKENIZER_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")

# These are the independently approved bytes used to build the published corpus. The producer's
# four-part seal is also recorded in P3_DECISION_LEDGER.md and its tokenizer builder tests.
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json": "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
    "tokenizer_config.json": (
        "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
    ),
}
TOKENIZER_COMPOSITE_SHA256 = (
    "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
)
TOKENIZERS_VERSION = "0.22.2"
TOKENIZER_BACKEND_VOCAB_SIZE = 151_665
TOKENIZER_EOS_TOKEN = "<|endoftext|>"
TOKENIZER_EOS_TOKEN_ID = 151_643
TOKENIZER_PAD_TOKEN_ID = 151_643
TOKENIZER_BEHAVIOR_PROBES = (
    "",
    "TACTIC\nby simp\nSTATE_AFTER\nno goals",
    r"\<lbrakk>A; B\<rbrakk> \<Longrightarrow> A",
    'proof "sorry" (* oops *)',
    "Unicode: ∀x∈ℝ. x² ≥ 0",
    "a b b b",
    "---\nGOAL\nSTATE_BEFORE",
    "<|endoftext|>",
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_behavior_sha256(tokenizer: Any) -> str:
    """Hash encoding IDs, tokens, offsets, decoding, vocabulary, and EOS behavior."""
    payload = {
        "schema": "qwen-tokenizer-behavior-v1",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "eos_token": TOKENIZER_EOS_TOKEN,
        "eos_token_id": tokenizer.token_to_id(TOKENIZER_EOS_TOKEN),
        "probes": [],
    }
    for text in TOKENIZER_BEHAVIOR_PROBES:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        payload["probes"].append(
            {
                "text": text,
                "ids": encoding.ids,
                "tokens": encoding.tokens,
                "offsets": [list(pair) for pair in encoding.offsets],
                "decoded": tokenizer.decode(encoding.ids, skip_special_tokens=False),
            }
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SealedTokenizer:
    """Verified local tokenizer bytes plus path-independent provenance."""

    root: Path
    backend: Any
    artifact_id: str
    artifact_version: str
    file_sha256: Dict[str, str]
    composite_sha256: str
    tokenizers_version: str
    eos_token_id: int
    pad_token_id: int

    def separator_ids(self, text: str) -> list[int]:
        """Encode a separator with the exact backend whose bytes were sealed."""
        return list(self.backend.encode(text, add_special_tokens=False).ids)

    def olmo_config(self) -> TokenizerConfig:
        """Build the embedding-width tokenizer config without an HF identifier."""
        config = qwen2_tokenizer_config()
        if config.eos_token_id != self.eos_token_id or config.pad_token_id != self.pad_token_id:
            raise RuntimeError(
                "Qwen model tokenizer IDs disagree with the sealed published tokenizer: "
                f"model eos/pad={config.eos_token_id}/{config.pad_token_id}, "
                f"artifact eos/pad={self.eos_token_id}/{self.pad_token_id}"
            )
        config.identifier = TOKENIZER_ARTIFACT
        return config

    def provenance_dict(self) -> dict[str, Any]:
        """Return stable evaluator-facing fields; local cache paths are intentionally absent."""
        return {
            "tokenizer_artifact_id": self.artifact_id,
            "tokenizer_artifact_version": self.artifact_version,
            "tokenizer_file_sha256": dict(sorted(self.file_sha256.items())),
            "tokenizer_composite_sha256": self.composite_sha256,
            "tokenizers_version": self.tokenizers_version,
            "tokenizer_eos_token_id": self.eos_token_id,
            "tokenizer_pad_token_id": self.pad_token_id,
        }


def seal_tokenizer_files(root: str | Path) -> SealedTokenizer:
    """Verify the approved tokenizer files and their complete runtime behavior."""
    import tokenizers
    from tokenizers import Tokenizer

    root = Path(root)
    digests: Dict[str, str] = {}
    for filename in TOKENIZER_REQUIRED_FILES:
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"sealed tokenizer file is missing or empty: {path}")
        digest = _sha256_file(path)
        expected = TOKENIZER_FILE_SHA256[filename]
        if digest != expected:
            raise RuntimeError(
                f"{filename} SHA-256 mismatch: expected {expected}, got {digest}"
            )
        digests[filename] = digest

    implementation_version = str(tokenizers.__version__)
    if implementation_version != TOKENIZERS_VERSION:
        raise RuntimeError(
            "tokenizers implementation version mismatch: "
            f"expected {TOKENIZERS_VERSION}, got {implementation_version}"
        )

    try:
        config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid sealed tokenizer_config.json") from error
    if (
        config.get("tokenizer_class") != "Qwen2Tokenizer"
        or config.get("eos_token") != TOKENIZER_EOS_TOKEN
        or config.get("pad_token") != TOKENIZER_EOS_TOKEN
    ):
        raise RuntimeError("sealed tokenizer_config.json is not the approved Qwen2 family")

    try:
        backend = Tokenizer.from_file(str(root / "tokenizer.json"))
    except Exception as error:
        raise RuntimeError("failed to deserialize sealed tokenizer.json") from error
    backend.no_truncation()
    backend.no_padding()
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    if vocab_size != TOKENIZER_BACKEND_VOCAB_SIZE:
        raise RuntimeError(
            f"sealed tokenizer vocabulary mismatch: expected "
            f"{TOKENIZER_BACKEND_VOCAB_SIZE}, got {vocab_size}"
        )
    eos_token_id = backend.token_to_id(TOKENIZER_EOS_TOKEN)
    pad_token_id = backend.token_to_id(config["pad_token"])
    if eos_token_id != TOKENIZER_EOS_TOKEN_ID or pad_token_id != TOKENIZER_PAD_TOKEN_ID:
        raise RuntimeError(
            "sealed tokenizer EOS/pad mismatch: expected "
            f"{TOKENIZER_EOS_TOKEN_ID}/{TOKENIZER_PAD_TOKEN_ID}, "
            f"got {eos_token_id}/{pad_token_id}"
        )
    composite = tokenizer_behavior_sha256(backend)
    if composite != TOKENIZER_COMPOSITE_SHA256:
        raise RuntimeError(
            "tokenizer composite behavior SHA-256 mismatch: expected "
            f"{TOKENIZER_COMPOSITE_SHA256}, got {composite}"
        )

    return SealedTokenizer(
        root=root,
        backend=backend,
        artifact_id=TOKENIZER_ARTIFACT_ID,
        artifact_version=TOKENIZER_ARTIFACT_VERSION,
        file_sha256=digests,
        composite_sha256=composite,
        tokenizers_version=implementation_version,
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id),
    )


def fetch_tokenizer_artifact(
    artifact: str,
    work_dir: str | Path,
    *,
    dataset_paths_fn: Optional[Callable[..., Any]] = None,
    s3: Optional[Any] = None,
) -> SealedTokenizer:
    """Fetch and verify the exact published tokenizer dependency."""
    if artifact != TOKENIZER_ARTIFACT:
        raise ValueError(
            f"P3 tokenizer is pinned to {TOKENIZER_ARTIFACT!r}; got {artifact!r}"
        )
    if "://" in str(work_dir):
        raise ValueError(f"tokenizer work_dir must be local, got {work_dir!r}")

    if dataset_paths_fn is None:
        from edullm_data.read import dataset_paths

        dataset_paths_fn = dataset_paths
    if s3 is None:
        from edullm_data.s3 import Boto3S3

        s3 = Boto3S3.default()

    read = dataset_paths_fn(TOKENIZER_ARTIFACT_ID, TOKENIZER_ARTIFACT_VERSION, s3=s3)
    resolved: Dict[str, str] = {}
    for source in read.paths:
        filename = Path(urlsplit(str(source)).path).name
        if filename not in TOKENIZER_REQUIRED_FILES:
            continue
        if filename in resolved:
            raise RuntimeError(f"{TOKENIZER_ARTIFACT} publishes duplicate {filename} files")
        resolved[filename] = str(source)
    missing = sorted(set(TOKENIZER_REQUIRED_FILES) - set(resolved))
    if missing:
        raise RuntimeError(f"{TOKENIZER_ARTIFACT} is missing required files: {missing}")

    rank = os.environ.get("LOCAL_RANK", "0")
    cache_root = (
        Path(work_dir)
        / "p3-tokenizer-qwen25-vendored-v1"
        / f"rank{rank}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_REQUIRED_FILES:
        source = resolved[filename]
        uri = urlsplit(source)
        if uri.scheme != "s3" or not uri.netloc or not uri.path:
            raise RuntimeError(
                f"{TOKENIZER_ARTIFACT} resolved {filename} to a non-S3 path: {source!r}"
            )
        payload = s3.get(uri.netloc, uri.path.lstrip("/"))
        temporary = cache_root / f".{filename}.tmp"
        temporary.write_bytes(payload)
        temporary.replace(cache_root / filename)
    return seal_tokenizer_files(cache_root)
