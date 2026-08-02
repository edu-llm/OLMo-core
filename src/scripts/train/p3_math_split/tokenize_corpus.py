"""Pack the JSONL corpus into publishable `.u32le.bin` token shards.

Both arms read the SAME shards. There are no label masks here any more: the split
arm recomputes the fact-block boundary at load time by finding the separator token
run in the stream (`train_platform.py:DerivedMaskTrainModule`). Two reasons, and the
second is the better one:

  * `weights-sidecar/v1` is not a registered profile, so a mask array has no legal
    home in a published dataset.
  * a derived mask cannot fall out of alignment with the tokens it describes. A
    shipped one can, silently, and that is the defect `mask_alignment_test.py`
    exists to catch.

Layout, which is fixed before a byte is written because `entry.path` is hashed into
`manifest_sha256` and re-pathing later means re-copying every payload byte:

    tokens/<corpus>/<split>-<NNNNN>.u32le.bin

`<corpus>` is one of metamath, prf2, enigma, mizar, thproofs, isabelle, so a trainer
can later measure one proof style alone. The extension is self-describing and the
bytes are raw little-endian uint32 from byte 0 — **never `.npy`**. OLMo-core memmaps
from byte 0 and derives the token count from the raw file size, so a real `.npy`
header corrupts both the tokens and the count, silently. uint32 is required: the
vocab is 151,936 and uint16 would wrap.

Over-length examples are DROPPED, not truncated. A truncated proof has no ending, and
training on thousands of them teaches a model to never stop. `--suggest` reports what
each sequence length would cost before you commit to one.

Padding: every example is pre-padded to exactly --sequence-length, so instance i is
example i. No cross-document packing, which keeps example order auditable and keeps
the separator search unambiguous — one fact block per sequence.

Usage:
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus <dir> --suggest
    python src/scripts/train/p3_math_split/tokenize_corpus.py --corpus <dir> \
        --out artifacts/public --sequence-length 16384 \
        --tokenizer ../memorysplit-requery-exact/tokenizers/qwen25-vendored
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

TOKENS_DTYPE = np.uint32  # vocab is 151,936, so uint16 is not an option
# What the corpus builder writes between the fact block and the goal.
SEPARATOR = "\n---\nGOAL "
# What the split arm actually SEARCHES for, and it is deliberately not the string
# above. BPE does not respect the boundary: the trailing space merges rightward into
# the goal's first word (` |-`, ` ![`, ` lemma`) in 98.4% of documents, and the
# leading newline merges leftward into the fact block's last characters (` )\n`,
# `"\n`) in 88.5%. Encoding the full separator gives [198, 10952, 15513, 969, 220],
# a run that survives in 777 of 258,316 documents -- 0.30%, and 0% in four of the six
# shards. The split arm would have found no boundary and supervised everything,
# silently becoming a second dense arm.
#
# The three-token core `---\nGOAL` -> [10952, 15513, 969] survives in 258,316 of
# 258,316, never appears twice, and the token immediately after it always begins at
# or past mask_end -- so supervision starts exactly at the goal.
SEPARATOR_SEARCH = "---\nGOAL"


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
    ap.add_argument("--corpus", default="corpus",
                    help="dir holding shards/ and eval/ of *.jsonl")
    ap.add_argument("--out", default="artifacts/public")
    ap.add_argument("--split", default="train", choices=("train", "val"),
                    help="train reads corpus/shards/, val reads corpus/eval/")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                    help="publish this same tokenizer as tokenizer/qwen25-vendored")
    ap.add_argument("--sequence-length", type=int, default=16384)
    ap.add_argument("--shard-tokens", type=int, default=250_000_000,
                    help="tokens per output shard; 250M x 4B = 1 GB")
    ap.add_argument("--pack", action="store_true",
                    help="fill each sequence with consecutive documents instead of "
                         "padding one example per sequence. The corpus median is 564 "
                         "tokens against a window sized for a 117k tail, so unpacked "
                         "padding runs 62-93%% of all compute. The split arm handles "
                         "several fact blocks per sequence (train_module.py).")
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

    sub = "shards" if args.split == "train" else "eval"
    srcs = sorted(glob.glob(os.path.join(args.corpus, sub, "*.jsonl")))
    if not srcs:
        sys.exit(f"no *.jsonl under {os.path.join(args.corpus, sub)}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if not tok.is_fast:
        sys.exit("need the fast tokenizer: offset_mapping is how the boundary is found")
    eos_id = tok.eos_token_id
    if eos_id is None:
        sys.exit("tokenizer has no eos_token_id")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    # The separator the split arm will search for at load time. Verifying it here,
    # against this tokenizer, is the only place the two halves can be kept in step.
    sep_ids = tok(SEPARATOR_SEARCH, add_special_tokens=False)["input_ids"]
    if not sep_ids:
        sys.exit(f"{SEPARATOR_SEARCH!r} tokenizes to nothing under {args.tokenizer}")

    S = args.sequence_length

    # Two passes rather than one. Holding every encoding at once would be ~2.5 GB of
    # numpy (far more as Python lists) for a 627M-token corpus, so pass 1 keeps only
    # lengths and pass 2 re-encodes one corpus at a time. Tokenizing twice is cheap
    # next to running out of memory halfway through a write.
    all_lengths = []
    per_corpus_lengths = {}
    for src in srcs:
        name = os.path.basename(src)[:-6]
        lengths = [
            len(encode_one(tok, row, eos_id)[0]) for row in load_jsonl(src)
        ]
        per_corpus_lengths[name] = lengths
        all_lengths += lengths
        print(f"  {name:<12}{len(lengths):>8,} examples  median {int(np.median(lengths)):>7,}  "
              f"max {max(lengths):>8,}")

    L = np.array(all_lengths)
    print(f"\n  {len(L):,} examples, median {int(np.median(L)):,}, max {L.max():,} tokens")

    if args.suggest:
        print(f"\n  {'seq_len':>9}{'kept':>10}{'%':>8}{'padding waste':>15}")
        for c in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
            fits = int((L <= c).sum())
            waste = 1 - L[L <= c].sum() / (fits * c) if fits else 1.0
            print(f"  {c:>9,}{fits:>10,}{100 * fits / len(L):>7.1f}%{waste:>14.1%}")
        print("\npick one and re-run without --suggest")
        return

    # A separator that cannot survive the window is a silent killer: the split arm
    # would find no boundary and supervise the whole sequence, making it a second
    # dense arm. Refuse rather than discover it in the loss curves.
    if len(sep_ids) >= S:
        sys.exit(f"separator is {len(sep_ids)} tokens but sequence_length is {S}")

    # Prove the run is findable in real documents rather than assuming it. This exact
    # assumption already failed once: the full separator string encodes to a run that
    # 99.7% of documents do not contain, because BPE merges across both its edges.
    probe_bad = probe_dup = probe_n = 0
    for src in srcs:
        for row in load_jsonl(src)[:200]:
            ids = tok(row["text"], add_special_tokens=False)["input_ids"]
            hits = [i for i in range(len(ids) - len(sep_ids) + 1)
                    if ids[i : i + len(sep_ids)] == sep_ids]
            probe_n += 1
            probe_bad += not hits
            probe_dup += len(hits) > 1
    if probe_bad or probe_dup:
        sys.exit(
            f"separator run {sep_ids} is not a reliable boundary under "
            f"{args.tokenizer}: missing in {probe_bad}/{probe_n} probed documents, "
            f"repeated in {probe_dup}. The split arm would supervise fact tokens or "
            f"find the boundary in the wrong place. Fix SEPARATOR_SEARCH before writing "
            f"shards."
        )
    print(f"  separator {sep_ids} found exactly once in {probe_n}/{probe_n} probed docs")

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "sequence_length": S,
        "tokenizer": args.tokenizer,
        "tokens_dtype": "uint32",
        "byte_order": "little",
        "eos_token_id": int(eos_id),
        "pad_token_id": int(pad_id),
        "separator": SEPARATOR,
        "separator_search": SEPARATOR_SEARCH,
        "separator_ids": [int(x) for x in sep_ids],
        "split": args.split,
        "groups": {},
    }

    grand_inst = grand_dropped = total_straddling = grand_real = 0
    for src in srcs:
        name = os.path.basename(src)[:-6]
        kept = []
        for row in load_jsonl(src):
            ids, _after, straddling = encode_one(tok, row, eos_id)
            if len(ids) <= S:
                kept.append(np.asarray(ids, dtype=TOKENS_DTYPE))
                total_straddling += straddling
        dropped = len(per_corpus_lengths[name]) - len(kept)
        grand_dropped += dropped
        if not kept:
            print(f"  {name}: every example exceeds {S}, no shard written")
            continue

        if args.pack:
            # Greedy first-fit: start a new sequence when the next document would not
            # finish inside the window. Documents are never split, so every fact block
            # keeps its separator and the derived mask stays well defined.
            rows, cur = [], []
            used = 0
            for ids in kept:
                if used + len(ids) > S and cur:
                    rows.append(cur)
                    cur, used = [], 0
                cur.append(ids)
                used += len(ids)
            if cur:
                rows.append(cur)
            instances = [np.concatenate(r) for r in rows]
        else:
            instances = kept

        d = os.path.join(args.out, "tokens", name)
        os.makedirs(d, exist_ok=True)
        per_shard = max(args.shard_tokens // S, 1)          # whole instances only
        shards = []
        for ordinal, lo in enumerate(range(0, len(instances), per_shard)):
            block = instances[lo : lo + per_shard]
            buf = np.full(len(block) * S, pad_id, dtype=TOKENS_DTYPE)
            for j, ids in enumerate(block):
                buf[j * S : j * S + len(ids)] = ids
            # `.tofile` writes the raw buffer with no header, which is the whole point.
            # `np.save` would prepend a .npy header and OLMo-core would read it as data.
            path = os.path.join(d, f"{args.split}-{ordinal:05d}.u32le.bin")
            buf.astype("<u4").tofile(path)
            shards.append({
                "path": os.path.relpath(path, args.out).replace(os.sep, "/"),
                "instances": len(block),
                "tokens": int(buf.size),
                "bytes": int(buf.size) * 4,
            })
        real = int(sum(x.size for x in instances))
        manifest["groups"][name] = {
            "documents": len(kept),
            "instances": len(instances),
            "real_tokens": real,
            "padding_fraction": round(1 - real / max(len(instances) * S, 1), 4),
            "dropped_over_length": dropped,
            "shards": shards,
        }
        grand_inst += len(instances)
        grand_real += real
        print(f"  {name:<12}{len(kept):>8,} docs -> {len(instances):>7,} instances, "
              f"{100 * (1 - real / max(len(instances) * S, 1)):>4.1f}% padding"
              + (f", {dropped:,} dropped over {S:,}" if dropped else ""))

    manifest["packed"] = bool(args.pack)
    manifest["instances"] = grand_inst
    manifest["real_tokens"] = grand_real
    manifest["dropped_over_length"] = grand_dropped
    manifest["tokens_straddling_boundary"] = total_straddling
    with open(os.path.join(args.out, f"{args.split}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = grand_inst * S
    print(f"\n  {grand_inst:,} instances x {S:,} = {total / 1e6:,.0f}M tokens of compute, "
          f"{grand_real / 1e6:,.0f}M real ({100 * (1 - grand_real / max(total, 1)):.1f}% padding)")
    print(f"  dropped over length: {grand_dropped:,} "
          f"({100 * grand_dropped / max(grand_inst + grand_dropped, 1):.1f}%)")
    if total_straddling:
        print(f"  {total_straddling:,} tokens straddle the block boundary")
    print(f"  wrote {args.out}/tokens/<corpus>/{args.split}-NNNNN.u32le.bin "
          f"and {args.split}_meta.json")


if __name__ == "__main__":
    main()
