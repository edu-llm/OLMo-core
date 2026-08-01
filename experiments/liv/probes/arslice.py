"""Is Zoology's AR-Hits slice tokenizer-dependent? Measure it on OUR corpus.

Takes the SAME text (decoded from val.npy GPT-2 ids), encodes it under GPT-2 (50257)
and LFM2 (64400), packs each into 4096-token windows, and computes the fraction of
tokens that are "the final token of a bigram that already appeared earlier in the
same window" -- the in-context-repeat half of Zoology's AR-Hits definition.

The training-frequency (<=1250x) half is approximated by a frequency table built from
a 40M-token slice of train.npy under each tokenizer, scaled to the corpus. Both halves
are reported so the parent can see which one moves.
"""
import json
import os
import time
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

DATA = "/scratch/users/ericrcwu/kda/lm/data"
EOS_GPT2 = 50256
WIN = 4096
OUT = "/scratch/users/ericrcwu/liv/arslice_results.json"
FREQ_TOK = int(os.environ.get("FREQ_TOK", "40000000"))  # train tokens for freq table
EVAL_TOK = int(os.environ.get("EVAL_TOK", "8000000"))   # val tokens to slice

res = {}
gpt2 = Tokenizer.from_pretrained("gpt2")
lfm2 = Tokenizer.from_pretrained("LiquidAI/LFM2-350M")
TOKS = {"gpt2": gpt2, "lfm2": lfm2}


def docs_from(path, ntok):
    a = np.asarray(np.load(path, mmap_mode="r")[:ntok], dtype=np.int64)
    eos = np.flatnonzero(a == EOS_GPT2)
    out, prev = [], 0
    for p in eos:
        if p > prev:
            out.append(a[prev:p])
        prev = p + 1
    return out


def encode_all(tk, texts):
    return [e.ids for e in tk.encode_batch_fast(texts, add_special_tokens=False)]


# ---- decode both slices once (same text feeds both tokenizers) --------------
t0 = time.perf_counter()
val_docs = docs_from(f"{DATA}/val.npy", EVAL_TOK)
tr_docs = docs_from(f"{DATA}/train.npy", FREQ_TOK)
val_txt = gpt2.decode_batch([d.tolist() for d in val_docs], skip_special_tokens=False)
tr_txt = gpt2.decode_batch([d.tolist() for d in tr_docs], skip_special_tokens=False)
res["n_val_docs"] = len(val_txt)
res["n_train_docs"] = len(tr_txt)
res["t_decode_s"] = time.perf_counter() - t0

for name, tk in TOKS.items():
    t0 = time.perf_counter()
    # --- training bigram frequency table (approximate; scaled to full corpus) ---
    tr_ids = encode_all(tk, tr_txt)
    n_tr = sum(len(x) for x in tr_ids)
    cnt = Counter()
    for ids in tr_ids:
        arr = np.asarray(ids, dtype=np.int64)
        if arr.size < 2:
            continue
        # pack bigram into one int64 key
        cnt.update((arr[:-1] * 200003 + arr[1:]).tolist())
    scale = 1.2e9 / n_tr  # extrapolate sample counts to the full 1.2B-token corpus
    thresh = 1250.0 / scale  # a bigram is "rare" if sample count < this

    # --- val: in-context bigram repeats, within 4096-token packed windows ---
    val_ids = encode_all(tk, val_txt)
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in val_ids])
    n_val = int(flat.size)
    nwin = n_val // WIN
    flat = flat[: nwin * WIN].reshape(nwin, WIN)

    hits_ctx = 0
    hits_ctx_rare = 0
    tot = 0
    for w in range(nwin):
        row = flat[w]
        keys = row[:-1] * 200003 + row[1:]  # bigram key at each position i -> token i+1
        seen = set()
        for i in range(keys.size):
            k = int(keys[i])
            tot += 1
            if k in seen:
                hits_ctx += 1
                if cnt.get(k, 0) < thresh:
                    hits_ctx_rare += 1
            else:
                seen.add(k)

    res[name] = {
        "vocab": tk.get_vocab_size(True),
        "train_sample_tokens": n_tr,
        "distinct_bigrams_in_sample": len(cnt),
        "freq_scale_to_1.2B": scale,
        "rare_threshold_in_sample_counts": thresh,
        "val_tokens_encoded": n_val,
        "windows": nwin,
        "scored_positions": tot,
        "in_context_repeat_frac": hits_ctx / tot,
        "AR_HITS_frac (repeat AND rare)": hits_ctx_rare / tot,
        "t_s": time.perf_counter() - t0,
    }
    print(name, json.dumps(res[name], indent=2), flush=True)

g, l = res["gpt2"], res["lfm2"]
res["ratio_lfm2_over_gpt2"] = {
    "in_context_repeat": l["in_context_repeat_frac"] / g["in_context_repeat_frac"],
    "AR_HITS": l["AR_HITS_frac (repeat AND rare)"] / g["AR_HITS_frac (repeat AND rare)"],
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["ratio_lfm2_over_gpt2"], indent=2))
