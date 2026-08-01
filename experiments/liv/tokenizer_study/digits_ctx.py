"""Context-sensitivity of passkey tokenization.

The passkey task needs the SAME digit string to tokenize the SAME WAY in the haystack
(where it is preceded by a space) and at the answer position. GPT-2's regex is
` ?\p{N}+`, so a leading space JOINS the number pretoken. LFM2/OLMo2 use bare
`\p{N}{1,3}`, so digits never absorb neighbours. This measures the consequence.
"""
import json, random
import numpy as np
from tokenizers import Tokenizer
TOKDIR = "/scratch/users/ericrcwu/liv/tok"
NAMES = ["gpt2", "lfm2", "olmo2", "neox", "qwen25"]
T = {n: Tokenizer.from_file(f"{TOKDIR}/{n}.tokenizer.json") for n in NAMES}
def e(n, s): return T[n].encode(s, add_special_tokens=False).ids
rnd = random.Random(7)
out = {}
for nd in (5, 6, 7):
    keys = ["".join(rnd.choice("0123456789") for _ in range(nd)) for _ in range(1000)]
    ent = {}
    for n in NAMES:
        bare = [e(n, k) for k in keys]
        sp = [e(n, " " + k) for k in keys]
        nl = [e(n, "\n" + k) for k in keys]
        q = [e(n, '"' + k) for k in keys]
        # does the digit-substring tokenization survive a leading space?
        # compare token count of key ALONE vs key-with-space minus any pure-prefix token
        same_sp = np.mean([len(a) == len(b) for a, b in zip(bare, sp)])
        # exact-suffix test: are bare ids a suffix of the spaced ids?
        suffix_sp = np.mean([b[-len(a):] == a for a, b in zip(bare, sp)])
        nsp = np.array([len(x) for x in sp]); nb = np.array([len(x) for x in bare])
        nnl = np.array([len(x) for x in nl]); nq = np.array([len(x) for x in q])
        # ambiguity: how many DISTINCT tokenization lengths does one nd produce?
        ent[n] = {
            "bare_len_mean": float(nb.mean()), "bare_len_set": sorted(set(nb.tolist())),
            "space_len_mean": float(nsp.mean()), "space_len_set": sorted(set(nsp.tolist())),
            "newline_len_mean": float(nnl.mean()), "quote_len_mean": float(nq.mean()),
            "frac_same_len_bare_vs_space": float(same_sp),
            "frac_bare_is_suffix_of_space": float(suffix_sp),
            "n_distinct_lengths": len(set(nb.tolist())),
            "deterministic_length": len(set(nb.tolist())) == 1,
        }
    out[f"{nd}_digit"] = ent

# phone-number formats: does the hyphenated form tokenize consistently?
fm = {}
for label, mk in [("dashed", lambda: f"{rnd.randint(200,989)}-{rnd.randint(200,989)}-{rnd.randint(0,9999):04d}"),
                  ("plain", lambda: f"{rnd.randint(2000000000,9899999999)}")]:
    ss = [mk() for _ in range(500)]
    fm[label] = {}
    for n in NAMES:
        L = np.array([len(e(n, s)) for s in ss])
        Ls = np.array([len(e(n, " " + s)) for s in ss])
        fm[label][n] = {"mean": float(L.mean()), "sd": float(L.std()),
                        "set": sorted(set(L.tolist()))[:10],
                        "n_distinct": len(set(L.tolist())),
                        "with_space_mean": float(Ls.mean())}
out["phone_formats"] = fm
with open(f"{TOKDIR}/digits_ctx.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
