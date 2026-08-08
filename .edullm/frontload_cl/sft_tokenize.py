"""Tokenize ``sft/frontload-cl-chat-sft`` conversations for OLMo-core SFT.

Published artifact is ``sft-conversations/v1`` JSONL. OLMo-core expects packed
``.npy`` token ids plus matching bool label masks (True = train on that token).

Uses the classic OLMo 2 chat template (``<|user|>`` / ``<|assistant|>``), not the
ChatML template that ships on ``allenai/dolma2-tokenizer`` by itself.
"""

from __future__ import annotations

import gzip
import itertools
import json
import logging
from array import array
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C
from .corpus import Refusal, Stage, read_failure

log = logging.getLogger(__name__)

TOKEN_IDS_GLOB = "token_ids_part_*.npy"
LABELS_MASK_GLOB = "labels_mask_part_*.npy"
# ~1 GiB of uint32 tokens per shard (matches open-instruct converter default).
TOKENS_PER_SHARD = (1 << 30) // 4


def load_hf_tokenizer(name_or_path: str = C.SFT_HF_TOKENIZER):
    """Load a HuggingFace tokenizer and install the OLMo 2 chat template."""
    from transformers import AutoTokenizer

    kwargs: Dict[str, Any] = {"trust_remote_code": False}
    if name_or_path == C.SFT_HF_TOKENIZER:
        kwargs["revision"] = C.SFT_HF_TOKENIZER_REVISION
    tok = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    tok.chat_template = C.SFT_CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenize_messages(
    tok,
    messages: Sequence[Dict[str, str]],
    *,
    max_seq_length: Optional[int] = None,
) -> Tuple[List[int], List[bool]]:
    """
    Encode one conversation with the OLMo 2 chat template.

    Label mask is True on assistant content tokens and the trailing EOS that
    closes an assistant turn; False on BOS / role headers / user / system text.

    :returns: ``(token_ids, label_mask)``; empty if nothing trainable remains.
    """
    if not messages:
        return [], []

    # Turn-by-turn so we can mark assistant spans without relying on
    # ``{% generation %}`` template tags (not present on all HF revisions).
    ids: List[int] = []
    mask: List[bool] = []

    bos = tok.encode(tok.bos_token, add_special_tokens=False)
    ids.extend(bos)
    mask.extend([False] * len(bos))

    for message_idx, msg in enumerate(messages):
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role in ("system",):
            piece = tok.encode(f"<|system|>\n{content}\n", add_special_tokens=False)
            ids.extend(piece)
            mask.extend([False] * len(piece))
        elif role in ("user", "human"):
            piece = tok.encode(f"<|user|>\n{content}\n", add_special_tokens=False)
            ids.extend(piece)
            mask.extend([False] * len(piece))
        elif role in ("assistant", "gpt"):
            header = tok.encode("<|assistant|>\n", add_special_tokens=False)
            ids.extend(header)
            mask.extend([False] * len(header))
            body = tok.encode(content, add_special_tokens=False)
            ids.extend(body)
            mask.extend([True] * len(body))
            eos = tok.encode(tok.eos_token, add_special_tokens=False)
            ids.extend(eos)
            mask.extend([True] * len(eos))
            if message_idx < len(messages) - 1:
                newline = tok.encode("\n", add_special_tokens=False)
                ids.extend(newline)
                mask.extend([False] * len(newline))
        else:
            log.warning("skipping message with unknown role %r", role)

    if max_seq_length is not None and len(ids) > max_seq_length:
        ids = ids[:max_seq_length]
        mask = mask[:max_seq_length]

    if not any(mask):
        return [], []
    return ids, mask


def iter_conversation_rows(paths: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from local or remote ``.jsonl`` / ``.jsonl.gz`` paths."""

    def _local_path(path: str) -> Path:
        p = Path(path)
        if p.exists():
            return p
        from cached_path import cached_path

        from olmo_core.io import add_cached_path_clients

        add_cached_path_clients()
        try:
            return Path(cached_path(path, quiet=True))
        except BaseException as exc:
            raise Refusal(
                read_failure(exc), f"reading conversation shard {path}: {type(exc).__name__}: {exc}"
            ) from exc

    for path in paths:
        local = _local_path(path)
        try:
            with local.open("rb") as raw:
                is_gzip = raw.read(2) == b"\x1f\x8b"
            opener = gzip.open if is_gzip else open
            with opener(local, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise Refusal(
                            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                            f"bad JSON in {path}:{line_no}: {exc}",
                        ) from exc
                    if not isinstance(row, dict) or "messages" not in row:
                        raise Refusal(
                            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                            f"{path}:{line_no} missing messages[]",
                        )
                    messages = row["messages"]
                    if not isinstance(messages, list):
                        raise Refusal(
                            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                            f"{path}:{line_no} messages is {type(messages).__name__}, not a list",
                        )
                    for message_no, message in enumerate(messages, start=1):
                        if not isinstance(message, dict):
                            raise Refusal(
                                Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                                f"{path}:{line_no} message {message_no} is "
                                f"{type(message).__name__}, not an object",
                            )
                        role = message.get("role")
                        if not isinstance(role, str) or role.lower() not in {
                            "system",
                            "user",
                            "human",
                            "assistant",
                            "gpt",
                        }:
                            raise Refusal(
                                Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                                f"{path}:{line_no} message {message_no} has unsupported role "
                                f"{role!r}",
                            )
                        content = message.get("content")
                        if not isinstance(content, str):
                            raise Refusal(
                                Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
                                f"{path}:{line_no} message {message_no} content is "
                                f"{type(content).__name__}, not a string",
                            )
                    yield row
        except Refusal:
            raise
        except (OSError, UnicodeError) as exc:
            raise Refusal(
                read_failure(exc), f"reading conversation shard {path}: {type(exc).__name__}: {exc}"
            ) from exc


def resolve_conversation_paths(
    *,
    dataset_id: str,
    version: str,
    split: str = "train",
) -> Tuple[str, List[str]]:
    """Resolve ``sft/frontload-cl-chat-sft`` conversation shard URIs."""
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    s3 = Boto3S3.default()
    if version in ("", "latest"):
        try:
            resolved = resolve_latest(dataset_id, s3=s3)
        except Refusal:
            raise
        except BaseException as exc:
            raise Refusal(read_failure(exc), f"{type(exc).__name__}: {exc}") from exc
        if resolved is None:
            raise Refusal(
                Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS,
                f"no published version of {dataset_id}",
            )
        version = resolved

    try:
        read = dataset_paths(dataset_id, version, split=split, s3=s3)
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(
            read_failure(exc),
            f"reading {dataset_id}/{version}: {type(exc).__name__}: {exc}",
        ) from exc

    if not read.paths:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} split={split} resolved to no shards",
        )
    # Keep the conversion contract stable across the tokenize-only and torchrun
    # phases even if a storage client returns equivalent objects in another order.
    return version, sorted(str(path) for path in read.paths)


def find_tokenized_shards(tokens_dir: str | Path) -> Tuple[List[str], List[str]]:
    """Return sorted ``(token_paths, mask_paths)`` under ``tokens_dir``, or empty lists."""
    root = Path(tokens_dir)
    if not root.exists():
        return [], []
    token_paths = sorted(str(p) for p in root.glob(TOKEN_IDS_GLOB))
    mask_paths = sorted(str(p) for p in root.glob(LABELS_MASK_GLOB))
    if token_paths and len(token_paths) != len(mask_paths):
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{tokens_dir}: {len(token_paths)} token shards vs {len(mask_paths)} mask shards",
        )
    return token_paths, mask_paths


def tokenize_conversations_to_dir(
    conversation_paths: Sequence[str],
    output_dir: str | Path,
    *,
    tokenizer_name: str = C.SFT_HF_TOKENIZER,
    max_seq_length: int = C.SFT_SEQ_LENGTH,
    tokens_per_shard: int = TOKENS_PER_SHARD,
    seed: int = C.DATA_SEED,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream conversations → ``token_ids_part_XXXX.npy`` + ``labels_mask_part_XXXX.npy``.

    The OLMo data loader performs the seeded epoch shuffle, so this conversion stays
    streaming instead of materializing the entire conversation release in memory.
    Documents longer than ``max_seq_length`` are truncated; rows with no trainable
    tokens are skipped. ``limit`` stops the input iterator after that many rows.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if tokens_per_shard <= 0:
        raise ValueError(f"tokens_per_shard must be positive, got {tokens_per_shard}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    stats_path = out / "tokenize_stats.json"
    expected_input = {
        "format_version": 2,
        "conversation_paths": list(conversation_paths),
        "tokenizer_name": tokenizer_name,
        "max_seq_length": max_seq_length,
        "limit": limit,
    }
    existing_tok = sorted(str(p) for p in out.glob(TOKEN_IDS_GLOB))
    existing_mask = sorted(str(p) for p in out.glob(LABELS_MASK_GLOB))
    if stats_path.is_file() and existing_tok and len(existing_tok) == len(existing_mask):
        try:
            existing_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_stats = {}
        token_item_size = existing_stats.get("token_item_size")
        lengths_match = token_item_size in (2, 4) and all(
            Path(token_path).stat().st_size % token_item_size == 0
            and Path(token_path).stat().st_size // token_item_size
            == Path(mask_path).stat().st_size
            for token_path, mask_path in zip(existing_tok, existing_mask)
        )
        if lengths_match and all(
            existing_stats.get(key) == value for key, value in expected_input.items()
        ):
            existing_stats.update(
                {
                    "output_dir": str(out),
                    "num_shards": len(existing_tok),
                    "token_paths": existing_tok,
                    "mask_paths": existing_mask,
                    "reused": True,
                }
            )
            log.info("reusing %d complete tokenized shards in %s", len(existing_tok), out)
            return existing_stats

    # A stats file is the commit marker. Shards without a matching marker are from a
    # failed or differently configured conversion and must never be treated as complete.
    stale = [*out.glob(TOKEN_IDS_GLOB), *out.glob(LABELS_MASK_GLOB)]
    if stale or stats_path.exists():
        log.warning("removing incomplete or stale tokenization output from %s", out)
        for path in stale:
            path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)

    tok = load_hf_tokenizer(tokenizer_name)
    vocab_size = getattr(tok, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tok)
        except TypeError:
            vocab_size = 100_278
    vocab_size = int(vocab_size)
    token_dtype = np.uint32 if vocab_size > np.iinfo(np.uint16).max else np.uint16

    shard_idx = 0
    token_typecode = "I" if np.dtype(token_dtype).itemsize == 4 else "H"
    buf_ids = array(token_typecode)
    buf_mask = bytearray()
    n_input_docs = 0
    n_docs = 0
    n_skipped = 0
    total_tokens = 0
    trainable_tokens = 0
    token_paths: List[str] = []
    mask_paths: List[str] = []

    def flush() -> None:
        nonlocal shard_idx, buf_ids, buf_mask
        if not buf_ids:
            return
        tok_path = out / f"token_ids_part_{shard_idx:04d}.npy"
        mask_path = out / f"labels_mask_part_{shard_idx:04d}.npy"
        if buf_ids.itemsize != np.dtype(token_dtype).itemsize:
            raise RuntimeError(
                f"array typecode {token_typecode!r} has {buf_ids.itemsize}-byte items, "
                f"expected {np.dtype(token_dtype).itemsize}"
            )
        # Despite the conventional .npy suffix, OLMo-core reads these as raw memmaps
        # from byte zero. np.save() would prepend a header and corrupt token/mask lengths.
        with tok_path.open("wb") as fh:
            buf_ids.tofile(fh)
        with mask_path.open("wb") as fh:
            fh.write(buf_mask)
        token_paths.append(str(tok_path))
        mask_paths.append(str(mask_path))
        log.info(
            "wrote %s (%d tokens) + %s",
            tok_path.name,
            len(buf_ids),
            mask_path.name,
        )
        shard_idx += 1
        buf_ids = array(token_typecode)
        buf_mask = bytearray()

    rows: Iterable[Dict[str, Any]] = iter_conversation_rows(conversation_paths)
    if limit is not None:
        rows = itertools.islice(rows, limit)
    for row in rows:
        n_input_docs += 1
        messages = row.get("messages") or []
        try:
            ids, mask = tokenize_messages(tok, messages, max_seq_length=max_seq_length)
        except Refusal:
            raise
        except BaseException as exc:
            source = row.get("source", "unknown")
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"tokenizing conversation {n_input_docs} (source={source!r}): "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if not ids:
            n_skipped += 1
            continue
        if len(buf_ids) + len(ids) > tokens_per_shard and buf_ids:
            flush()
        buf_ids.extend(ids)
        buf_mask.extend(mask)
        n_docs += 1
        total_tokens += len(ids)
        trainable_tokens += sum(1 for m in mask if m)
        if n_input_docs % 10_000 == 0:
            log.info(
                "read %d conversations; tokenized %d (%d tokens so far)",
                n_input_docs,
                n_docs,
                total_tokens,
            )

    flush()
    if not token_paths:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "tokenization produced no shards (all rows empty or untrainable)",
        )

    stats = {
        **expected_input,
        "output_dir": str(out),
        "num_shards": len(token_paths),
        "token_paths": token_paths,
        "mask_paths": mask_paths,
        "num_input_conversations": n_input_docs,
        "num_conversations": n_docs,
        "num_skipped": n_skipped,
        "total_tokens": total_tokens,
        "trainable_tokens": trainable_tokens,
        "token_dtype": str(token_dtype),
        "token_item_size": np.dtype(token_dtype).itemsize,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "reused": False,
    }
    stats_tmp = out / "tokenize_stats.json.tmp"
    stats_tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    stats_tmp.replace(stats_path)
    log.info(
        "SFT tokenize done: %d conversations, %.3fM tokens (%.3fM trainable) → %s",
        n_docs,
        total_tokens / 1e6,
        trainable_tokens / 1e6,
        out,
    )
    return stats
