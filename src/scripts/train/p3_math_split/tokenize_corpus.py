"""Turn the JSONL corpus into what OLMo-core reads: one token array, two label masks.

The experimental manipulation lives entirely in this file, and the file is structured
so that the thing being held constant cannot drift:

    tokens.npy              written ONCE, used by both arms
    label_mask_dense.npy    True on every real token (facts included)
    label_mask_split.npy    True only after the fact block

Both arms point at the same tokens.npy. "Identical input documents, identical order,
identical token count" is therefore a property of the filesystem, not a claim about two
pipelines that we hope agree.

Mask convention, which is easy to get backwards: OLMo-core applies
`labels.masked_fill_(~label_mask, -100)` and *then* left-shifts
(`olmo_core/data/utils.py:598-605`). So `label_mask[i] = False` removes token i from
being **predicted**. The mask is indexed by the token whose prediction is scored, not
by the token being conditioned on. src/test/scripts/p3_math_split/mask_alignment_test.py pins this.

Padding: every example is pre-padded to exactly --sequence-length in the flat array, so
instance i from NumpyFSLDataset is exactly example i. No cross-document packing. It
wastes some compute but makes example order auditable.

Usage:
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus corpus --out tokenized --suggest
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus corpus --out tokenized --sequence-length 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

TOKENS_DTYPE = np.uint32  # vocab is 151936, so uint16 is not an option


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def encode_one(tok, row, eos_id):
    """Tokenize one example and derive the split-arm supervision mask.

    Returns (ids, is_after_block, n_straddling). `is_after_block[i]` is True when
    token i lies entirely past the fact block, i.e. when the split arm should score it.

    A token that straddles the boundary is treated as part of the fact block — masked.
    That is the conservative direction: it can only ever remove supervision from a
    token that contains fact characters, never add supervision to one.
    """
    enc = tok(row["text"], add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])

    mask_end = row["mask_end"]
    after = []
    straddling = 0
    for start, end in offsets:
        if start >= mask_end:
            after.append(True)
        else:
            after.append(False)
            if end > mask_end:
                straddling += 1

    # EOS terminates the proof. It is a real prediction target in both arms — the
    # model has to learn to stop, and that is part of the proof-generation task, not
    # part of the fact block.
    ids.append(eos_id)
    after.append(True)
    return ids, after, straddling


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="tokenized")
    ap.add_argument("--split", default="train", help="train | eval_retrieval | eval_iid")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sequence-length", type=int, default=1024)
    ap.add_argument(
        "--suggest",
        action="store_true",
        help="report the token-length distribution and exit without writing",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers")

    src = os.path.join(args.corpus, f"{args.split}.jsonl")
    if not os.path.exists(src):
        sys.exit(f"{src} not found — run src/scripts/train/p3_math_split/build_corpus.py first")

    rows = load_jsonl(src)
    print(f"{len(rows):,} examples from {src}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tok.is_fast:
        sys.exit("need the fast tokenizer: offset_mapping is how the mask is placed")
    eos_id = tok.eos_token_id
    if eos_id is None:
        sys.exit("tokenizer has no eos_token_id")

    encoded, total_straddling = [], 0
    for row in rows:
        ids, after, straddling = encode_one(tok, row, eos_id)
        total_straddling += straddling
        encoded.append((row, ids, after))

    lengths = np.array([len(ids) for _, ids, _ in encoded])
    print(
        f"  token lengths: min {lengths.min()}  median {int(np.median(lengths))}  "
        f"p95 {int(np.percentile(lengths, 95))}  p99 {int(np.percentile(lengths, 99))}  "
        f"max {lengths.max()}"
    )
    if total_straddling:
        # Not an error — the mask stays conservative — but worth seeing, because a
        # large count would mean the separator is being merged into fact text.
        print(f"  {total_straddling:,} tokens straddle the block boundary (masked as facts)")

    if args.suggest:
        for candidate in (512, 640, 768, 1024, 1536, 2048):
            fits = int((lengths <= candidate).sum())
            waste = 1 - lengths[lengths <= candidate].sum() / (fits * candidate) if fits else 1.0
            print(
                f"    seq_len {candidate:>5}: keeps {fits:>7,}/{len(lengths):,} "
                f"({fits / len(lengths):>5.1%})  padding waste {waste:>5.1%}"
            )
        print("\npick one and re-run without --suggest")
        return

    S = args.sequence_length
    kept = [(r, ids, after) for r, ids, after in encoded if len(ids) <= S]
    dropped = len(encoded) - len(kept)
    if not kept:
        sys.exit(f"every example exceeds --sequence-length {S}")

    n = len(kept)
    tokens = np.full(n * S, tok.pad_token_id or eos_id, dtype=TOKENS_DTYPE)
    mask_dense = np.zeros(n * S, dtype=np.bool_)
    mask_split = np.zeros(n * S, dtype=np.bool_)

    index = []
    for i, (row, ids, after) in enumerate(kept):
        lo = i * S
        hi = lo + len(ids)
        tokens[lo:hi] = np.asarray(ids, dtype=TOKENS_DTYPE)
        mask_dense[lo:hi] = True  # every real token; padding stays False
        mask_split[lo:hi] = np.asarray(after, dtype=np.bool_)
        index.append(
            {
                "instance": i,
                "id": row["id"],
                "theorem": row["theorem"],
                "n_tokens": len(ids),
                "n_fact_tokens": int(len(ids) - sum(after)),
                "cited": row["cited"],
            }
        )

    os.makedirs(args.out, exist_ok=True)
    prefix = os.path.join(args.out, args.split)
    np.save(f"{prefix}_tokens.npy", tokens)
    np.save(f"{prefix}_label_mask_dense.npy", mask_dense)
    np.save(f"{prefix}_label_mask_split.npy", mask_split)

    meta = {
        "split": args.split,
        "tokenizer": args.tokenizer,
        "sequence_length": S,
        "n_instances": n,
        "n_dropped_too_long": dropped,
        "eos_token_id": eos_id,
        "pad_token_id": tok.pad_token_id or eos_id,
        "tokens_dtype": "uint32",
        "total_tokens_including_padding": int(tokens.size),
        "total_real_tokens": int(mask_dense.sum()),
        "supervised_tokens_dense": int(mask_dense.sum()),
        "supervised_tokens_split": int(mask_split.sum()),
        "fact_token_fraction": float(1 - mask_split.sum() / mask_dense.sum()),
        "padding_fraction": float(1 - mask_dense.sum() / tokens.size),
        "straddling_tokens": total_straddling,
    }
    with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(f"{prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    print()
    print(f"  wrote {prefix}_tokens.npy            {n:,} x {S} = {tokens.size:,} tokens")
    print(f"        {prefix}_label_mask_dense.npy  {mask_dense.sum():,} supervised")
    print(f"        {prefix}_label_mask_split.npy  {mask_split.sum():,} supervised")
    if dropped:
        print(f"  dropped {dropped:,} examples longer than {S} tokens")
    print(f"  fact-token fraction {meta['fact_token_fraction']:.1%}  (design band 17-30%)")
    print(f"  padding             {meta['padding_fraction']:.1%} of the array")
    if not 0.05 < meta["fact_token_fraction"] < 0.60:
        print(
            "  WARNING: fact fraction outside 5-60%; the two arms will barely differ "
            "(too low) or the split arm will have little to learn from (too high)"
        )


if __name__ == "__main__":
    main()
