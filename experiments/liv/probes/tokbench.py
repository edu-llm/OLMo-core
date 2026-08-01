"""Tokenizer round-trip + throughput microbenchmark for the LIV vocab decision.

Read-only wrt the corpus. Single process, single core. Reads a small slice of
/scratch/users/ericrcwu/kda/lm/data/train.npy (GPT-2 uint16), splits on EOS 50256,
decodes N documents back to text, then times re-encoding under several tokenizers.

Does NOT retokenize the corpus. Costs the option only.
"""
import json
import os
import sys
import time

import numpy as np

DATA = "/scratch/users/ericrcwu/kda/lm/data/train.npy"
EOS = 50256
NDOCS = int(os.environ.get("NDOCS", "5000"))
OUT = os.environ.get("OUT", "/scratch/users/ericrcwu/liv/tokbench_results.json")

res = {"ndocs_target": NDOCS}

# ---------------------------------------------------------------- load slice
t0 = time.perf_counter()
arr = np.load(DATA, mmap_mode="r")
res["train_npy_tokens"] = int(arr.shape[0])
res["train_npy_dtype"] = str(arr.dtype)

# Take enough tokens to yield NDOCS docs. FineWeb-Edu median ~622 tok -> ~1000 avg.
take = min(int(arr.shape[0]), NDOCS * 2000)
chunk = np.asarray(arr[:take], dtype=np.int64)
res["slice_tokens"] = int(chunk.shape[0])
res["t_load_s"] = time.perf_counter() - t0

# ------------------------------------------------------------ split on EOS
t0 = time.perf_counter()
eos_pos = np.flatnonzero(chunk == EOS)
res["n_eos_in_slice"] = int(eos_pos.size)
docs_ids = []
prev = 0
for p in eos_pos:
    if p > prev:
        docs_ids.append(chunk[prev:p])
    prev = p + 1
    if len(docs_ids) >= NDOCS:
        break
res["n_docs"] = len(docs_ids)
res["t_split_s"] = time.perf_counter() - t0
res["doc_tok_lens"] = {
    "mean": float(np.mean([len(d) for d in docs_ids])),
    "median": float(np.median([len(d) for d in docs_ids])),
    "p90": float(np.percentile([len(d) for d in docs_ids], 90)),
    "max": int(max(len(d) for d in docs_ids)),
    "total": int(sum(len(d) for d in docs_ids)),
}

from tokenizers import Tokenizer  # noqa: E402

# ----------------------------------------------------- GPT-2 decode (option 1a)
gpt2 = Tokenizer.from_pretrained("gpt2")
res["gpt2_vocab_with_added"] = gpt2.get_vocab_size(True)

lists = [d.tolist() for d in docs_ids]
t0 = time.perf_counter()
texts = gpt2.decode_batch(lists, skip_special_tokens=False)
t_dec = time.perf_counter() - t0
res["decode"] = {
    "t_s": t_dec,
    "tok_per_s": res["doc_tok_lens"]["total"] / t_dec,
    "docs_per_s": len(lists) / t_dec,
}
total_chars = sum(len(t) for t in texts)
res["total_chars"] = total_chars
res["bytes_utf8"] = sum(len(t.encode("utf-8")) for t in texts)

# ------------------------------------------------ round-trip losslessness check
t0 = time.perf_counter()
re_ids = gpt2.encode_batch_fast(texts, add_special_tokens=False)
t_gpt2 = time.perf_counter() - t0
n_exact = sum(1 for a, b in zip(lists, re_ids) if a == b.ids)
res["roundtrip"] = {
    "n_docs": len(lists),
    "n_exact": n_exact,
    "frac_exact": n_exact / len(lists),
    "orig_tokens": res["doc_tok_lens"]["total"],
    "reenc_tokens": sum(len(b.ids) for b in re_ids),
}
res["gpt2_reencode"] = {
    "t_s": t_gpt2,
    "tok_per_s": sum(len(b.ids) for b in re_ids) / t_gpt2,
    "chars_per_s": total_chars / t_gpt2,
}

# ------------------------------------------- candidate 65k / 100k tokenizers
CANDS = {
    "LFM2-350M": "LiquidAI/LFM2-350M",
    "dolma2": "allenai/dolma2-tokenizer",
}
res["candidates"] = {}
for name, repo in CANDS.items():
    entry = {"repo": repo}
    try:
        tk = Tokenizer.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"
        res["candidates"][name] = entry
        continue
    entry["vocab_with_added"] = tk.get_vocab_size(True)
    entry["vocab_no_added"] = tk.get_vocab_size(False)
    # warm
    tk.encode_batch_fast(texts[:50], add_special_tokens=False)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        enc = tk.encode_batch_fast(texts, add_special_tokens=False)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    ntok = sum(len(e.ids) for e in enc)
    entry["tokens"] = ntok
    entry["t_s_best_of_3"] = best
    entry["tok_per_s"] = ntok / best
    entry["chars_per_s"] = total_chars / best
    entry["chars_per_token"] = total_chars / ntok
    entry["fertility_vs_gpt2"] = ntok / res["doc_tok_lens"]["total"]
    entry["max_id"] = int(max(max(e.ids) for e in enc))
    res["candidates"][name] = entry

res["nproc"] = os.cpu_count()
res["loadavg"] = os.getloadavg()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
