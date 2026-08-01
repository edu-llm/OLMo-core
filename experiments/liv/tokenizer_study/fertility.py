"""Fertility measurement: GPT-2 vs LFM2 vs comparison tokenizers on OUR corpus.

Text is recovered by decoding /scratch/users/ericrcwu/kda/lm/data/val.npy (GPT-2 ids)
back to text -- no download. Writes results incrementally to results.json.
"""
import json, os, random, sys, time
import numpy as np
from tokenizers import Tokenizer

TOKDIR = "/scratch/users/ericrcwu/liv/tok"
DATA = "/scratch/users/ericrcwu/kda/lm/data/val.npy"
OUT = os.path.join(TOKDIR, "results.json")
EOS = 50256
MAX_DOCS = int(os.environ.get("MAX_DOCS", "6000"))
NAMES = ["gpt2", "lfm2", "olmo2", "neox", "qwen25"]

R = {"meta": {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "max_docs_req": MAX_DOCS,
              "source": DATA, "host": os.uname().nodename}}

def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1)
    print("[saved]", time.strftime("%H:%M:%S"), flush=True)

toks = {}
for n in NAMES:
    t = Tokenizer.from_file(f"{TOKDIR}/{n}.tokenizer.json")
    toks[n] = t
R["vocab"] = {n: {"with_added": toks[n].get_vocab_size(True),
                  "base": toks[n].get_vocab_size(False)} for n in NAMES}
save()

def enc(name, texts):
    """Encode list of texts, no special tokens. Returns list of id-lists."""
    return [e.ids for e in toks[name].encode_batch(texts, add_special_tokens=False)]

# ---------------- Step 1: recover text ----------------
arr = np.load(DATA, mmap_mode="r")
print("val.npy", arr.shape, arr.dtype, flush=True)
ids_all = np.asarray(arr)                      # 8M uint16 -> fine in RAM
eos_pos = np.flatnonzero(ids_all == EOS)
print("n_eos", len(eos_pos), flush=True)

# Documents are the spans BETWEEN eos markers. Drop the first (truncated head) and
# last (truncated tail) so every doc used is complete.
docs_ids = []
for a, b in zip(eos_pos[:-1], eos_pos[1:]):
    seg = ids_all[a + 1:b]
    if len(seg) >= 16:
        docs_ids.append(seg.astype(np.int64).tolist())
    if len(docs_ids) >= MAX_DOCS:
        break
print("n_docs", len(docs_ids), flush=True)

g2 = toks["gpt2"]
texts = g2.decode_batch(docs_ids, skip_special_tokens=False)

# ---------------- round-trip check ----------------
re_ids = enc("gpt2", texts)
fails = [i for i, (a, b) in enumerate(zip(re_ids, docs_ids)) if a != b]
R["roundtrip"] = {"n_docs": len(docs_ids), "n_fail": len(fails),
                  "fail_rate": len(fails) / len(docs_ids),
                  "example_fail_idx": fails[:5]}
print("roundtrip fails", len(fails), "/", len(docs_ids), flush=True)
save()

# Keep only docs that round-trip exactly -> the text is provably our corpus.
keep = [i for i in range(len(docs_ids)) if i not in set(fails)]
texts = [texts[i] for i in keep]
docs_ids = [docs_ids[i] for i in keep]

chars = np.array([len(t) for t in texts], dtype=np.float64)
nbytes = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.float64)
words = np.array([len(t.split()) for t in texts], dtype=np.float64)
R["corpus_sample"] = {"n_docs": len(texts), "total_chars": int(chars.sum()),
                      "total_bytes": int(nbytes.sum()), "total_words": int(words.sum()),
                      "mean_chars_per_doc": float(chars.mean())}
save()

# ---------------- Step 2: fertility ----------------
counts = {}
freqs = {}
for n in NAMES:
    t0 = time.time()
    if n == "gpt2":
        ids = docs_ids
    else:
        ids = enc(n, texts)
    counts[n] = np.array([len(x) for x in ids], dtype=np.float64)
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in ids])
    freqs[n] = np.bincount(flat, minlength=toks[n].get_vocab_size(True))
    print(n, "encoded", time.time() - t0, "s  tokens", int(counts[n].sum()), flush=True)

rng = np.random.default_rng(0)
NB = 5000
fert = {}
base = counts["gpt2"]
for n in NAMES:
    c = counts[n]
    ratio_agg = c.sum() / base.sum()
    # bootstrap over documents on the aggregate (ratio-of-sums) estimator
    idx = rng.integers(0, len(c), size=(NB, len(c)))
    bs = c[idx].sum(1) / base[idx].sum(1)
    per_doc = c / base
    fert[n] = {
        "total_tokens": int(c.sum()),
        "tokens_per_doc_mean": float(c.mean()),
        "tokens_per_doc_median": float(np.median(c)),
        "tokens_per_doc_p90": float(np.percentile(c, 90)),
        "tokens_per_1000_chars": float(1000 * c.sum() / chars.sum()),
        "tokens_per_word": float(c.sum() / words.sum()),
        "bytes_per_token": float(nbytes.sum() / c.sum()),
        "chars_per_token": float(chars.sum() / c.sum()),
        "ratio_vs_gpt2_agg": float(ratio_agg),
        "ratio_vs_gpt2_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "ratio_per_doc_mean": float(per_doc.mean()),
        "ratio_per_doc_median": float(np.median(per_doc)),
        "ratio_per_doc_p05_p95": [float(np.percentile(per_doc, 5)), float(np.percentile(per_doc, 95))],
        "frac_docs_more_tokens_than_gpt2": float((c > base).mean()),
        "implied_corpus_tokens_for_1.2B_gpt2": float(1.2e9 * ratio_agg),
    }
R["fertility"] = fert
save()

# ---------------- Step 3a: digits / passkeys ----------------
rnd = random.Random(1234)
dig = {}
for nd in (5, 6, 7):
    keys = ["".join(rnd.choice("0123456789") for _ in range(nd)) for _ in range(1000)]
    ctx_pre = "The pass key is "
    ctx_post = ". Remember it."
    ctxs = [ctx_pre + k + ctx_post for k in keys]
    base_ctx_len = {n: len(enc(n, [ctx_pre + ctx_post])[0]) for n in NAMES}
    entry = {}
    for n in NAMES:
        bare = np.array([len(x) for x in enc(n, keys)])
        inctx = np.array([len(x) for x in enc(n, ctxs)])
        entry[n] = {
            "bare_mean": float(bare.mean()), "bare_median": float(np.median(bare)),
            "bare_min": int(bare.min()), "bare_max": int(bare.max()),
            "bare_hist": {str(k): int(v) for k, v in zip(*np.unique(bare, return_counts=True))},
            "always_one_token_per_digit": bool((bare == nd).all()),
            "in_context_total_mean": float(inctx.mean()),
            "context_only_tokens": base_ctx_len[n],
        }
    dig[f"{nd}_digit"] = entry
R["digits"] = dig
save()

# also: digit pretokenizer rule
R["digit_rule"] = {}
for n in NAMES:
    d = json.load(open(f"{TOKDIR}/{n}.tokenizer.json"))
    pt = json.dumps(d.get("pre_tokenizer"))
    R["digit_rule"][n] = {"has_pN_1_3": "\\\\p{N}{1,3}" in pt or "p{N}{1,3}" in pt,
                          "pre_tokenizer": d.get("pre_tokenizer")}
save()

# ---------------- Step 3b: phonebook ----------------
FIRST = ["John","Maria","Alexander","Li","Priya","Bob","Christopher","Zoe","Mohammed","Ana",
         "Jae","Katarzyna","Tom","Ngozi","Sven","Isabella","Wei","Omar","Frida","Hank"]
LAST = ["Smith","Garcia","Nakamura","Okonkwo","Petrov","Lee","Vandenberg","Ali","Rossi","Chen",
        "Johnson","Kowalski","Mbeki","Andersson","Silva","Kim","Dubois","Haddad","Novak","Ito"]
fmts = ["{a}-{b}-{c}", "({a}) {b}-{c}", "+1-{a}-{b}-{c}", "{a}.{b}.{c}", "{a}{b}{c}"]
entries = []
for i in range(200):
    fn = rnd.choice(FIRST); ln = rnd.choice(LAST)
    a = rnd.randint(200, 989); b = rnd.randint(200, 989); c = rnd.randint(0, 9999)
    num = rnd.choice(fmts).format(a=a, b=b, c=f"{c:04d}")
    entries.append(f"{fn} {ln}: {num}\n")
pb = {}
for n in NAMES:
    L = np.array([len(x) for x in enc(n, entries)], dtype=np.float64)
    pb[n] = {"tokens_per_entry_mean": float(L.mean()),
             "tokens_per_entry_median": float(np.median(L)),
             "tokens_per_entry_p90": float(np.percentile(L, 90)),
             "min": int(L.min()), "max": int(L.max()),
             "entries_in_4096_ctx": float(4096 / L.mean()),
             "entries_in_2048_ctx": float(2048 / L.mean()),
             "entries_in_8192_ctx": float(8192 / L.mean())}
R["phonebook"] = {"n_entries": len(entries), "example": entries[:5], "per_tok": pb}
save()

# ---------------- Step 3c: needle haystack / effective distance ----------------
# Take contiguous 4096-GPT2-token windows of real corpus text; how much text is that,
# and how many tokens under each tokenizer?
W = 4096
stream = np.concatenate([np.asarray(d, dtype=np.int64) for d in docs_ids])
nwin = min(300, len(stream) // W)
wins = [stream[i * W:(i + 1) * W].tolist() for i in range(nwin)]
wtexts = g2.decode_batch(wins, skip_special_tokens=False)
wc = np.array([len(t) for t in wtexts], dtype=np.float64)
ww = np.array([len(t.split()) for t in wtexts], dtype=np.float64)
nd_res = {"n_windows": nwin, "gpt2_window_tokens": W,
          "chars_per_window_mean": float(wc.mean()), "chars_per_window_sd": float(wc.std()),
          "words_per_window_mean": float(ww.mean()), "words_per_window_sd": float(ww.std()),
          "per_tok": {}}
for n in NAMES:
    L = np.array([len(x) for x in enc(n, wtexts)], dtype=np.float64)
    bsi = rng.integers(0, nwin, size=(2000, nwin))
    bsr = L[bsi].mean(1) / W
    nd_res["per_tok"][n] = {
        "tokens_for_same_text_mean": float(L.mean()),
        "tokens_for_same_text_sd": float(L.std()),
        "ratio_vs_gpt2": float(L.mean() / W),
        "ratio_ci95": [float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5))],
        # inverse view: at a FIXED 4096-token budget, how much text fits?
        "chars_in_4096_tok": float(4096 * wc.mean() / L.mean()),
        "words_in_4096_tok": float(4096 * ww.mean() / L.mean()),
    }
R["needle"] = nd_res
save()

# ---------------- Step 3d: conv receptive field ----------------
conv = {}
for k in (3, 5, 9, 15):
    per_layer = k - 1
    for nl in (10, 16):
        reach = per_layer * nl
        conv[f"k{k}_L{nl}"] = {"reach_tokens": reach, "per_tok": {
            n: {"chars": reach * float(chars.sum() / counts[n].sum()),
                "words": reach * float(words.sum() / counts[n].sum())} for n in NAMES}}
R["conv_reach"] = conv
save()

# ---------------- Step 4: vocab utilization / dead embeddings ----------------
vu = {}
S = float(counts["gpt2"].sum())  # not used; per-tok below
for n in NAMES:
    f = freqs[n].astype(np.float64)
    V = len(f)
    tot = f.sum()
    p = f / tot
    nz = f > 0
    H = float(-(p[nz] * np.log2(p[nz])).sum())
    cpt = chars.sum() / tot
    ent = {
        "vocab_size": int(V),
        "n_used": int(nz.sum()),
        "frac_used": float(nz.mean()),
        "sample_tokens": int(tot),
        "entropy_bits_per_token": H,
        "bits_per_char": float(H / cpt),
        "top10_share": float(np.sort(p)[::-1][:10].sum()),
        "top1000_share": float(np.sort(p)[::-1][:1000].sum()),
        "n_seen_once": int((f == 1).sum()),
    }
    # extrapolate expected occurrences at training budget T
    for T in (2e9, 5e9):
        exp_occ = p * T
        ent[f"dead_lt100_at_{int(T/1e9)}B"] = int((exp_occ < 100).sum())
        ent[f"dead_lt100_frac_at_{int(T/1e9)}B"] = float((exp_occ < 100).mean())
        ent[f"dead_lt10_at_{int(T/1e9)}B"] = int((exp_occ < 10).sum())
        ent[f"dead_zero_at_{int(T/1e9)}B"] = int((exp_occ == 0).sum())
        # rule-of-three upper bound for unseen tokens: rate < 3/tot
        ent[f"unseen_upper_occ_at_{int(T/1e9)}B"] = float(3.0 / tot * T)
    vu[n] = ent
R["vocab_util"] = vu
save()

R["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
save()
print("DONE", flush=True)
