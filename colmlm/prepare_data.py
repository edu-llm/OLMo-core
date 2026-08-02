"""Tokenize the Co-LMLM annotated corpus into OLMo-core FSL memmaps + a fact-span label mask.

For each input shard (``data/colmlm-annotate/worker-N/train-000NN.annotations.jsonl.zst``) this
writes two headerless little-endian arrays that OLMo-core's ``NumpyFSLDatasetConfig`` memmaps
directly (it reads from byte offset 0, so **no** ``.npy`` header):

    tokens/train-000NN.bin       uint16 token ids (SmolLM2 tokenizer), EOS appended after each doc
    masks/train-000NN.mask.bin   bool (1 byte) label mask, same length as the token file

The token stream is identical for the two experiment models; only the mask differs:

* ``base``  ignores the mask -> next-token loss over every token (memorizes facts).
* ``split`` uses the mask     -> a token is ``False`` (excluded from the loss via ``-100``) iff its
  character span overlaps a fact span, inclusive of the span's first and last token. This is the
  only difference between the two runs, so the A/B is clean and needs no vocab change.

Masking is computed exactly (not by first-char heuristic): a per-document character indicator of
"is this char inside a fact span" is prefix-summed, and a token is masked iff it covers >=1 fact
character. This correctly handles byte-level BPE tokens that fold a leading space into the token.

Usage:
    python colmlm/prepare_data.py --annotate-dir data/colmlm-annotate --out-dir data/tokenized
    python colmlm/prepare_data.py --workers 0 --limit-docs 3000 --out-dir data/tokenized-sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import zstandard
from transformers import AutoTokenizer

TOKEN_DTYPE = np.uint16  # SmolLM2 vocab 49,152 < 65,536


def read_zst_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
        buf = b""
        while chunk := reader.read(1 << 22):
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line.strip():
                    yield json.loads(line)
        if buf.strip():
            yield json.loads(buf)


def doc_label_mask(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]) -> np.ndarray:
    """Boolean keep-mask (True = keep in loss) for one document's tokens.

    A token is masked (False) iff its ``[start, end)`` character range covers at least one
    character that lies inside a fact span. Empty/special tokens (``end <= start``) are kept.
    """
    n = len(offsets)
    if n == 0:
        return np.ones(0, dtype=bool)
    ce_arr = np.fromiter((o[1] for o in offsets), dtype=np.int64, count=n)
    cs_arr = np.fromiter((o[0] for o in offsets), dtype=np.int64, count=n)
    if not spans:
        return np.ones(n, dtype=bool)

    length = int(ce_arr.max())
    is_fact = np.zeros(length, dtype=np.int64)
    for cs, ce in spans:
        cs = max(0, min(cs, length))
        ce = max(0, min(ce, length))
        if ce > cs:
            is_fact[cs:ce] = 1
    prefix = np.concatenate([[0], np.cumsum(is_fact)])  # prefix[k] = fact chars in [0, k)

    cs_c = np.clip(cs_arr, 0, length)
    ce_c = np.clip(ce_arr, 0, length)
    fact_in_token = prefix[ce_c] - prefix[cs_c]
    empty = ce_arr <= cs_arr
    keep = (fact_in_token == 0) | empty
    return keep


def process_shard(
    tokenizer,
    records: Iterator[dict],
    eos_id: int,
    batch_size: int,
    limit: int | None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    tokens_parts: List[np.ndarray] = []
    masks_parts: List[np.ndarray] = []
    n_docs = n_fact_spans = masked_tokens = 0

    def flush(texts: List[str], span_lists: List[List[Tuple[int, int]]]):
        nonlocal n_docs, masked_tokens
        enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        for ids, offs, spans in zip(enc["input_ids"], enc["offset_mapping"], span_lists):
            keep = doc_label_mask(offs, spans)
            ids_arr = np.asarray(ids, dtype=TOKEN_DTYPE)
            # Append EOS after each document (kept in the loss).
            tokens_parts.append(ids_arr)
            tokens_parts.append(np.asarray([eos_id], dtype=TOKEN_DTYPE))
            masks_parts.append(keep)
            masks_parts.append(np.ones(1, dtype=bool))
            masked_tokens += int((~keep).sum())
            n_docs += 1

    texts: List[str] = []
    span_lists: List[List[Tuple[int, int]]] = []
    for rec in records:
        if limit is not None and n_docs + len(texts) >= limit:
            break
        texts.append(rec["text"])
        spans = [(a["char_start"], a["char_end"]) for a in rec.get("annotations", [])]
        n_fact_spans += len(spans)
        span_lists.append(spans)
        if len(texts) >= batch_size:
            flush(texts, span_lists)
            texts, span_lists = [], []
    if texts:
        flush(texts, span_lists)

    tokens = (
        np.concatenate(tokens_parts) if tokens_parts else np.zeros(0, dtype=TOKEN_DTYPE)
    )
    masks = np.concatenate(masks_parts) if masks_parts else np.zeros(0, dtype=bool)
    assert tokens.shape == masks.shape, (tokens.shape, masks.shape)
    stats = {
        "docs": n_docs,
        "tokens": int(tokens.shape[0]),
        "fact_spans": n_fact_spans,
        "masked_tokens": masked_tokens,
        "masked_frac": (masked_tokens / tokens.shape[0]) if tokens.shape[0] else 0.0,
    }
    return tokens, masks, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotate-dir", default="data/colmlm-annotate")
    ap.add_argument("--out-dir", default="data/tokenized")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--workers", type=int, nargs="*", default=list(range(19)))
    ap.add_argument("--limit-docs", type=int, default=None, help="Cap docs per shard (for testing).")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    ann_dir = Path(args.annotate_dir)
    out = Path(args.out_dir)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    print(f"tokenizer={args.tokenizer} vocab={tok.vocab_size} eos_id={eos_id}")

    manifest = {
        "tokenizer": args.tokenizer,
        "dtype": "uint16",
        "byte_order": "little",
        "header_bytes": 0,
        "eos_token_id": int(eos_id),
        "sequence_packing": "documents concatenated with EOS separators",
        "shards": [],
    }
    total_tokens = total_docs = total_masked = 0
    t0 = time.time()
    for w in args.workers:
        src = ann_dir / f"worker-{w}" / f"train-{w:05d}.annotations.jsonl.zst"
        if not src.exists():
            print(f"  skip worker-{w}: {src} not found")
            continue
        tokens, masks, stats = process_shard(
            tok, read_zst_jsonl(src), eos_id, args.batch_size, args.limit_docs
        )
        tok_path = out / "tokens" / f"train-{w:05d}.bin"
        mask_path = out / "masks" / f"train-{w:05d}.mask.bin"
        tokens.astype("<u2").tofile(tok_path)
        masks.astype(np.bool_).tofile(mask_path)
        manifest["shards"].append(
            {
                "worker": w,
                "tokens_path": str(tok_path.as_posix()),
                "mask_path": str(mask_path.as_posix()),
                **stats,
            }
        )
        total_tokens += stats["tokens"]
        total_docs += stats["docs"]
        total_masked += stats["masked_tokens"]
        rate = total_tokens / max(time.time() - t0, 1e-6)
        print(
            f"  worker-{w}: {stats['docs']:,} docs -> {stats['tokens']:,} tokens "
            f"(masked {stats['masked_frac']:.2%})  [{rate/1e6:.2f}M tok/s]"
        )

    manifest["total_docs"] = total_docs
    manifest["total_tokens"] = total_tokens
    manifest["total_masked_tokens"] = total_masked
    manifest["total_masked_frac"] = (total_masked / total_tokens) if total_tokens else 0.0
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\n{total_docs:,} docs, {total_tokens:,} tokens, "
        f"{manifest['total_masked_frac']:.2%} masked -> {out}/manifest.json "
        f"in {time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
