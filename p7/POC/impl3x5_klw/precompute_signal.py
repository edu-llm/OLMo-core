#!/usr/bin/env python
"""Compute Impl 3's per-token signal over Impl 5's training file. Both variants, one pass.

IMPL3_HANDOFF §4.1: "Computing *s_t* over the whole dataset is the expensive part." It is
also the only stage here with real headroom, so it is the stage that gets parallelised.

Three things make one pass enough for both variants and all four arms:

**Variant a is a by-product of variant b's forward.** ``s_a = −log π₀(y_t|ctx)`` is one
gather out of the same base distribution whose full support variant b reduces to a KL. Running
them separately would double a ~25-minute pass to buy nothing.

**π₀ and π_SFT come out of one set of weights.** The reference is a LoRA adapter over the same
base, so ``PeftModel.disable_adapter()`` gives π₀ and the plain forward gives π_SFT. That is
one model in memory instead of two, and it removes the possibility of the two forwards
disagreeing about anything but the adapter.

**Temperature is not in the cache key** (§4.1), so ``bT1``, ``bT2`` and ``bT451`` all read the
same variant-b cache and only ``build_row_multipliers`` sees T.

Sharding is round-robin over pedagogy row index, with the rows sorted by length *within* a
shard so batches pad tightly. Round-robin rather than contiguous because the training file is
in block order, so contiguous slices can carry systematically different length distributions
and one worker then finishes minutes after the others. Output is keyed by original row index,
so the merge restores file order exactly and the result does not depend on the shard count.

    # one worker per visible GPU, then merge
    python precompute_signal.py --shard 0 --n_shards 4
    python precompute_signal.py --merge --n_shards 4

    # everything on one GPU
    python precompute_signal.py

Determinism caveat, stated because it is real: the forwards run in bf16 (training's precision),
and bf16 matmul results depend slightly on how a batch is padded. Batching here is fully
deterministic — fixed token budget, length sort with row index as a stable tiebreak — so a
re-run on the same hardware reproduces the cache, but a re-run at a different ``--n_shards``
or on a different GPU model can move the last digits of a signal. It cannot move the
multipliers meaningfully: they are a rank-like transform of a globally standardised signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from klw import weighting                                            # noqa: E402
from klw._impl5 import chat5, manifest, mixing                       # noqa: E402
from klw.config_klw import (                                         # noqa: E402
    ALL_ARMS,
    BASE_MODEL,
    DATA_ARM,
    MAX_LEN,
    REFERENCE_ADAPTER_ARM,
    REFERENCE_ADAPTER_STEP,
    variants_needed,
)
from klw.paths_klw import (                                          # noqa: E402
    DATA_DIR,
    SHARD_DIR,
    ensure_dir,
    reference_adapter,
    signal_cache,
    signal_shard,
    train_file,
)

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", default=None, help=f"Defaults to Impl 5 arm {DATA_ARM}'s.")
    p.add_argument("--impl5_runs_root", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--reference", default=None,
                   help=f"Adapter dir for pi_SFT. Defaults to {REFERENCE_ADAPTER_ARM}'s "
                        f"ckpt-{REFERENCE_ADAPTER_STEP}.")
    p.add_argument("--variants", default=None,
                   help="Comma-separated subset of a,b. Defaults to what the arms need.")
    p.add_argument("--max_len", type=int, default=MAX_LEN)

    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="Merge shards; no GPU needed.")

    p.add_argument("--max_batch_tokens", type=int, default=16384,
                   help="Padded-token budget per forward. Raise on a bigger card.")
    p.add_argument("--max_batch_rows", type=int, default=32)
    p.add_argument("--kl_chunk", type=int, default=1024,
                   help="Label positions per full-vocab KL reduction.")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--shard_dir", default=None)
    p.add_argument("--limit", type=int, default=0, help="Smoke test: first N pedagogy rows.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# tokenised view of the training file
# --------------------------------------------------------------------------- #
def tokenised_rows(path: Path, tokenizer, max_len: int):
    """``[(row_index, is_pedagogy, input_ids, labels, digest)]`` in file order.

    Tokenises with ``chat5.make_tokenize_fn`` — literally the function the trainer calls — so
    the label mask the signal is computed over is the label mask that will be trained on.
    """
    fn = chat5.make_tokenize_fn(tokenizer, max_len)
    rows = manifest.read_jsonl(path)
    out = []
    for i, rec in enumerate(rows):
        enc = fn({"messages": rec["messages"]})
        ids, labels = enc["input_ids"], enc["labels"]
        out.append((i, bool(mixing.is_pedagogy(rec)), ids, labels,
                    weighting.row_digest(ids, labels)))
    return out, rows


def batches(items, max_tokens: int, max_rows: int):
    """Length-sorted, token-budgeted batches. Deterministic given ``items``.

    Sorted descending by length with the row index as tiebreak, so the first batch is the
    largest and an OOM shows up in the first seconds rather than 20 minutes in.
    """
    order = sorted(items, key=lambda r: (-len(r[2]), r[0]))
    batch: list = []
    for row in order:
        n = len(row[2])
        width = max(n, max(len(b[2]) for b in batch) if batch else 0)
        if batch and ((len(batch) + 1) * width > max_tokens or len(batch) + 1 > max_rows):
            yield batch
            batch = []
        batch.append(row)
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# the model: pi_0 and pi_SFT from one set of weights
# --------------------------------------------------------------------------- #
def load_base_and_reference(base_model: str, reference: str | None):
    """``(model, tokenizer, device, has_reference)``.

    ``model`` is a ``PeftModel`` when a reference is given, so ``disable_adapter()`` yields π₀
    and the plain forward yields π_SFT. bf16, ``eval()`` — eval mode is what switches LoRA
    dropout off, and with dropout live the "reference" would be a different function on every
    forward.
    """
    from transformers import AutoModelForCausalLM

    tokenizer = chat5.load_tokenizer(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    model.eval()

    has_ref = False
    if reference:
        from peft import PeftModel
        if not Path(reference, "adapter_config.json").exists():
            raise SystemExit(f"no adapter_config.json in {reference} — variant b needs "
                             f"pi_SFT. Fetch {REFERENCE_ADAPTER_ARM}/ckpt-"
                             f"{REFERENCE_ADAPTER_STEP} from S3 first.")
        model = PeftModel.from_pretrained(model, reference, is_trainable=False)
        model.eval()
        has_ref = True
    return model, tokenizer, device, has_ref


def _pad(batch, pad_id: int, device: str):
    width = max(len(r[2]) for r in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    for j, r in enumerate(batch):
        n = len(r[2])
        ids[j, :n] = torch.tensor(r[2], dtype=torch.long)
        lab[j, :n] = torch.tensor(r[3], dtype=torch.long)
        att[j, :n] = 1
    return ids.to(device), lab.to(device), att.to(device)


@torch.no_grad()
def signals_for_batch(model, batch, pad_id: int, device: str, want_b: bool, has_ref: bool,
                      kl_chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(surprise, kl, counts)`` over the batch's unmasked label positions, row-major.

    The causal shift is applied once, here: the distribution that predicts ``labels[j]`` lives
    at logits position ``j-1``, so everything is computed on ``logits[:, :-1]`` against
    ``labels[:, 1:]``. Both signals are read off the *same* selected positions, which is what
    guarantees variant a and variant b index the same tokens.

    Selected logits are kept in the model's own bf16 and promoted to fp32 one ``kl_chunk`` at a
    time. Holding ``log_softmax`` of both models over the full ``[n_sel, V]`` selection in fp32
    would be the peak allocation of the entire pass — about 5 GB at the default batch budget,
    for no accuracy the chunked form does not also have.
    """
    ids, lab, att = _pad(batch, pad_id, device)
    tgt = lab[:, 1:]
    mask = tgt != IGNORE
    counts = mask.sum(dim=1).to("cpu").numpy().astype(np.int64)
    flat_tgt = tgt[mask]                                   # [n_sel]
    n_sel = int(flat_tgt.numel())
    if n_sel == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), counts)

    def selected_logits(base: bool) -> torch.Tensor:
        """Selected ``[n_sel, V]`` logits, with the full ``[B, L, V]`` tensor freed.

        ``base=True`` means π₀. Without a reference adapter the model *is* π₀, so there is no
        adapter to disable and no second forward to run — which is the variant-a-only path.
        """
        if base and has_ref:
            with model.disable_adapter():
                out = model(input_ids=ids, attention_mask=att)
        else:
            out = model(input_ids=ids, attention_mask=att)
        sel = out.logits[:, :-1, :][mask]
        del out
        return sel

    base_sel = selected_logits(base=True)
    ref_sel = selected_logits(base=False) if want_b else None

    surprise = np.zeros(n_sel, dtype=np.float32)
    kl = np.zeros(n_sel, dtype=np.float32)
    for s in range(0, n_sel, kl_chunk):
        e = min(s + kl_chunk, n_sel)
        lp = torch.log_softmax(base_sel[s:e].float(), dim=-1)
        surprise[s:e] = (-lp.gather(1, flat_tgt[s:e].view(-1, 1)).squeeze(1)).to("cpu").numpy()
        if ref_sel is not None:
            lq = torch.log_softmax(ref_sel[s:e].float(), dim=-1)
            # KL(pi_0 || pi_SFT) = sum_v p0(v) (log p0(v) - log q(v))
            kl[s:e] = (lp.exp() * (lp - lq)).sum(-1).to("cpu").numpy()
            del lq
        del lp
    del base_sel, ref_sel
    return surprise, kl, counts


# --------------------------------------------------------------------------- #
# shard / merge
# --------------------------------------------------------------------------- #
def run_shard(args, tok_rows, n_rows: int, want: tuple[str, ...], keys: dict[str, str]):
    ped = [r for r in tok_rows if r[1]]
    if args.limit:
        ped = ped[:args.limit]
    mine = [r for i, r in enumerate(ped) if i % args.n_shards == args.shard] \
        if args.n_shards > 1 else ped
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} of {len(ped)} pedagogy rows "
          f"({sum(len(r[2]) for r in mine):,} tokens)", flush=True)

    want_b = "b" in want
    ref = args.reference if want_b else None
    model, tokenizer, device, has_ref = load_base_and_reference(args.base_model, ref)
    if want_b and not has_ref:
        raise SystemExit("variant b was requested but no reference adapter was loaded")
    if device == "cpu":
        print("WARNING: no GPU visible; this will be very slow", flush=True)

    per_row: dict[int, dict[str, np.ndarray]] = {}
    t0, done, n_batches = time.time(), 0, 0
    for batch in batches(mine, args.max_batch_tokens, args.max_batch_rows):
        sur, kl, counts = signals_for_batch(model, batch, tokenizer.pad_token_id, device,
                                            want_b, has_ref, args.kl_chunk)
        at = 0
        for row, c in zip(batch, counts):
            per_row[row[0]] = {"a": sur[at:at + c], "b": kl[at:at + c]}
            at += c
        done += len(batch)
        n_batches += 1
        if n_batches % 20 == 0 or done == len(mine):
            el = time.time() - t0
            print(f"  {done}/{len(mine)} rows  {el / 60:.1f} min  "
                  f"eta {(el / max(done, 1)) * (len(mine) - done) / 60:.1f} min", flush=True)

    ensure_dir(args.shard_dir)
    for variant in want:
        vals = np.concatenate([per_row[i][variant] for i in sorted(per_row)]) \
            if per_row else np.zeros(0, np.float32)
        idx = np.array(sorted(per_row), dtype=np.int64)
        lens = np.array([per_row[i][variant].size for i in sorted(per_row)], dtype=np.int64)
        path = signal_shard(variant, keys[variant], args.shard, args.shard_dir)
        np.savez(path, values=vals.astype(np.float32), row_index=idx, lengths=lens,
                 n_rows=np.array(n_rows), variant=np.array(variant))
        print(f"  -> {path}  ({vals.size:,} token signals)")
    return len(mine)


def merge_shards(args, tok_rows, n_rows: int, want: tuple[str, ...], keys: dict[str, str],
                 meta: dict):
    digests = np.array([r[4] for r in tok_rows], dtype=np.uint64)
    is_ped = np.array([r[1] for r in tok_rows], dtype=bool)
    expected = {r[0]: sum(1 for t in r[3] if t != IGNORE) for r in tok_rows if r[1]}
    if args.limit:
        keep = [r[0] for r in tok_rows if r[1]][:args.limit]
        expected = {i: expected[i] for i in keep}

    for variant in want:
        chunks: dict[int, np.ndarray] = {}
        for k in range(args.n_shards):
            path = signal_shard(variant, keys[variant], k, args.shard_dir)
            if not path.exists():
                raise SystemExit(f"missing shard {path}. Re-run --shard {k}.")
            z = np.load(path, allow_pickle=False)
            if int(z["n_rows"]) != n_rows:
                raise SystemExit(f"{path} was built for {int(z['n_rows'])} rows, not {n_rows}")
            at = 0
            for i, n in zip(z["row_index"], z["lengths"]):
                chunks[int(i)] = z["values"][at:at + int(n)]
                at += int(n)

        missing = sorted(set(expected) - set(chunks))
        if missing:
            raise SystemExit(f"variant {variant}: {len(missing)} pedagogy rows have no signal "
                             f"(first: {missing[:5]}). A shard did not finish.")
        for i, n in expected.items():
            if chunks[i].size != n:
                raise SystemExit(f"variant {variant}: row {i} has {chunks[i].size} signals "
                                 f"for {n} label tokens — tokenisation drifted")

        offsets = np.zeros(n_rows + 1, dtype=np.int64)
        pieces = []
        for i in range(n_rows):
            v = chunks.get(i, np.zeros(0, np.float32))
            pieces.append(v)
            offsets[i + 1] = offsets[i] + v.size
        cache = weighting.SignalCache(
            variant=variant,
            values=np.concatenate(pieces) if pieces else np.zeros(0, np.float32),
            offsets=offsets, row_hash=digests, is_pedagogy=is_ped,
            meta={**meta, "variant": variant, "n_signals": int(offsets[-1]),
                  "n_pedagogy_rows": len(expected)},
        )
        out = signal_cache(variant, keys[variant], args.data_dir)
        cache.save(out)
        med, scale = weighting.robust_stats(cache.values)
        print(f"variant {variant}: {offsets[-1]:,} token signals over {len(expected):,} rows "
              f"-> {out}")
        print(f"  median {med:.4f}  1.4826*MAD {scale:.4f}  "
              f"min {cache.values.min():.4f}  max {cache.values.max():.4f}")


def write_meta(args, keys: dict[str, str], meta: dict) -> None:
    """One meta file per variant, named after that variant's own key."""
    for variant, key in keys.items():
        (Path(args.data_dir) / f"signal_meta_{variant}_{key[:16]}.json").write_text(
            json.dumps({**meta, "variant": variant, "cache_key": key}, indent=2) + "\n",
            encoding="utf-8")


def main():
    args = parse_args()
    args.data_dir = args.data_dir or str(DATA_DIR)
    # Under data_dir, not the repo default: otherwise a run with an explicit --data_dir
    # scatters its shards into the source tree.
    args.shard_dir = args.shard_dir or (str(Path(args.data_dir) / "shards")
                                        if args.data_dir else str(SHARD_DIR))
    ensure_dir(args.data_dir)

    tf = Path(args.train_file) if args.train_file else train_file(DATA_ARM, args.impl5_runs_root)
    if not tf.exists():
        raise SystemExit(f"missing {tf}\nBuild it first:  python mix_arm5.py --arm {DATA_ARM}")
    ref = Path(args.reference) if args.reference else reference_adapter(
        REFERENCE_ADAPTER_ARM, REFERENCE_ADAPTER_STEP, args.impl5_runs_root)
    args.reference = str(ref)

    want = tuple(v.strip() for v in args.variants.split(",")) if args.variants \
        else variants_needed(ALL_ARMS)
    for v in want:
        if v not in weighting.VARIANTS:
            raise SystemExit(f"unknown variant {v!r}")

    # One key per variant, from the single shared definition. Variant a's does not include the
    # reference (its signal does not depend on pi_SFT); variant b's does. Assembling these
    # separately here and in the trainer is exactly the bug smoke_klw.py caught.
    data_key = weighting.file_digest(tf)
    keys = {v: weighting.signal_key(v, tf, args.base_model, ref, args.max_len) for v in want}

    print("=" * 74)
    print(f"signal precompute | variants {','.join(want)} | "
          + "  ".join(f"{v}:{k[:12]}" for v, k in keys.items()))
    print(f"  train file : {tf}")
    print(f"  base       : {args.base_model}")
    print(f"  reference  : {ref if 'b' in want else '(variant a only — not needed)'}")
    print("=" * 74, flush=True)

    tokenizer = chat5.load_tokenizer(args.base_model)
    tok_rows, _ = tokenised_rows(tf, tokenizer, args.max_len)
    n_rows = len(tok_rows)
    n_ped = sum(1 for r in tok_rows if r[1])
    n_lab = sum(1 for r in tok_rows if r[1] for t in r[3] if t != IGNORE)
    print(f"{n_rows:,} rows | {n_ped:,} pedagogy | {n_lab:,} pedagogy label tokens "
          f"(these are what get reweighted)", flush=True)

    meta = {
        "train_file": str(tf), "train_file_digest": data_key, "base_model": args.base_model,
        "reference": str(ref) if "b" in want else None,
        "cache_keys": keys, "max_len": args.max_len,
        "n_rows": n_rows, "n_pedagogy": n_ped, "n_pedagogy_label_tokens": n_lab,
        "n_shards": args.n_shards, "limit": args.limit or None,
        "batching": {"max_batch_tokens": args.max_batch_tokens,
                     "max_batch_rows": args.max_batch_rows, "kl_chunk": args.kl_chunk},
        "note": ("Signals are per pedagogy label token, in label order, over Impl 5's D4 mix. "
                 "Temperature is NOT part of the key (IMPL3_HANDOFF 4.1) so one cache serves "
                 "the whole temperature sweep."),
    }

    if args.merge:
        merge_shards(args, tok_rows, n_rows, want, keys, meta)
        write_meta(args, keys, meta)
        return

    done = [v for v in want if signal_cache(v, keys[v], args.data_dir).exists()]
    if done and not args.force and args.n_shards == 1:
        print(f"cache already present for {','.join(done)}; --force to rebuild")
        want = tuple(v for v in want if v not in done)
        if not want:
            return

    run_shard(args, tok_rows, n_rows, want, keys)
    if args.n_shards == 1:
        merge_shards(args, tok_rows, n_rows, want, keys, meta)
        write_meta(args, keys, meta)
    else:
        print(f"\nshard {args.shard} done. After all {args.n_shards} finish:\n"
              f"    python precompute_signal.py --merge --n_shards {args.n_shards}")


if __name__ == "__main__":
    main()
