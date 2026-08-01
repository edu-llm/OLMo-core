"""Large-sample vocabulary utilization / dead-embedding estimate.

Streams ~200M GPT-2 tokens from train.npy in chunks, decodes to text, re-encodes with
each candidate tokenizer, and accumulates only bincounts (constant memory).

Resolution rationale: with S sample tokens and training budget T, a token needs
count >= 100*S/T in the sample to clear "100 gradient updates". S=200M, T=2B -> count>=10.
"""
import json, os, time
import numpy as np
from tokenizers import Tokenizer

TOKDIR = "/scratch/users/ericrcwu/liv/tok"
DATA = "/scratch/users/ericrcwu/kda/lm/data/train.npy"
OUT = os.path.join(TOKDIR, "vocab_util.json")
EOS = 50256
TARGET = int(os.environ.get("TARGET", "200000000"))
CHUNK = 2_000_000
NAMES = ["gpt2", "lfm2", "olmo2", "neox", "qwen25"]

toks = {n: Tokenizer.from_file(f"{TOKDIR}/{n}.tokenizer.json") for n in NAMES}
V = {n: toks[n].get_vocab_size(True) for n in NAMES}
cnt = {n: np.zeros(V[n], dtype=np.int64) for n in NAMES}
g2 = toks["gpt2"]

arr = np.load(DATA, mmap_mode="r")
N = min(TARGET, len(arr))
print("train.npy", arr.shape, "target", N, flush=True)

tot_chars = 0
tot_gpt2 = 0
done = 0
t0 = time.time()
while done < N:
    hi = min(done + CHUNK, N)
    ids = np.asarray(arr[done:hi]).astype(np.int64)
    done = hi
    # split on EOS into docs so decode never straddles a doc boundary oddly
    eos = np.flatnonzero(ids == EOS)
    bounds = [0] + (eos + 1).tolist() + [len(ids)]
    docs = [ids[a:b].tolist() for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0]
    docs = [[t for t in d if t != EOS] for d in docs]
    docs = [d for d in docs if d]
    texts = g2.decode_batch(docs, skip_special_tokens=False)
    tot_chars += sum(len(t) for t in texts)
    for n in NAMES:
        if n == "gpt2":
            flat = np.concatenate([np.asarray(d, dtype=np.int64) for d in docs])
        else:
            e = toks[n].encode_batch(texts, add_special_tokens=False)
            flat = np.concatenate([np.asarray(x.ids, dtype=np.int64) for x in e])
        cnt[n] += np.bincount(flat, minlength=V[n])
    tot_gpt2 = int(cnt["gpt2"].sum())
    if (done // CHUNK) % 10 == 0:
        el = time.time() - t0
        print(f"{done/1e6:.0f}M/{N/1e6:.0f}M  {el:.0f}s  eta {el*(N-done)/max(done,1):.0f}s", flush=True)
        # partial dump so a crash still leaves numbers
        np.savez(os.path.join(TOKDIR, "counts_partial.npz"),
                 **{n: cnt[n] for n in NAMES}, done=np.array([done]))

S = {n: int(cnt[n].sum()) for n in NAMES}
res = {"sample_gpt2_tokens": int(done), "sample_chars": int(tot_chars),
       "elapsed_s": time.time() - t0, "per_tok": {}}
for n in NAMES:
    f = cnt[n].astype(np.float64)
    p = f / f.sum()
    nz = f > 0
    H = float(-(p[nz] * np.log2(p[nz])).sum())
    cpt = tot_chars / f.sum()
    e = {"vocab_size": int(V[n]), "sample_tokens": S[n],
         "n_used": int(nz.sum()), "frac_used": float(nz.mean()),
         "n_never_seen": int((~nz).sum()),
         "entropy_bits_per_token": H, "chars_per_token": float(cpt),
         "bits_per_char": float(H / cpt),
         "n_count_lt10": int((f < 10).sum()), "n_count_lt100": int((f < 100).sum()),
         "top10_share": float(np.sort(p)[::-1][:10].sum()),
         "top1k_share": float(np.sort(p)[::-1][:1000].sum()),
         "top10k_share": float(np.sort(p)[::-1][:10000].sum())}
    for T in (2e9, 5e9):
        occ = p * T
        k = int(T / 1e9)
        e[f"dead_lt100_at_{k}B"] = int((occ < 100).sum())
        e[f"dead_lt100_frac_at_{k}B"] = float((occ < 100).mean())
        e[f"dead_lt1000_at_{k}B"] = int((occ < 1000).sum())
        e[f"dead_zero_at_{k}B"] = int((occ == 0).sum())
        e[f"resolution_min_sample_count_for_100_at_{k}B"] = float(100 * S[n] / T)
    res["per_tok"][n] = e

with open(OUT, "w") as fo:
    json.dump(res, fo, indent=1)
np.savez(os.path.join(TOKDIR, "counts_final.npz"), **{n: cnt[n] for n in NAMES})
print("DONE", json.dumps({n: res["per_tok"][n]["n_used"] for n in NAMES}), flush=True)
