# 20 — Fertility measurement: GPT-2 (50,257) vs LFM2 (64,400) on OUR corpus

All numbers MEASURED on FarmShare `rice-03` unless tagged otherwise. Text is our actual
tokenized corpus (`/scratch/users/ericrcwu/kda/lm/data/`), recovered by GPT-2-decoding the
stored ids — no dataset download. Scripts and raw JSON: `/scratch/users/ericrcwu/liv/tok/`.

---

## BOTTOM LINE / MEASURED NUMBERS

| quantity | GPT-2 (50,257) | **LFM2 (64,400)** | verdict |
|---|---:|---:|---|
| **Fertility ratio vs GPT-2** (tokens for identical text) | 1.000 | **0.9954** (95% CI **0.9940–0.9967**) | **0.46% FEWER tokens. Negligible.** |
| ↑ replicated on independent 200M-token train sample | 1.000 | **0.9921** | 0.79% fewer — same verdict |
| bytes / token | 4.599 | 4.620 | +0.46% |
| tokens / 1,000 chars | 218.41 | 217.41 | −0.46% |
| tokens / whitespace word | 1.3393 | 1.3331 | −0.46% |
| implied corpus size (our 1.2B GPT-2 tokens) | 1.200 B | **1.194 B** | −5.5M tokens (−0.46%) |
| **Passkey: tokens for a bare 7-digit key** | 3.084 mean, **{2,3,4}** | **3.000, always {3}** | LFM2 **deterministic**, GPT-2 **not** |
| Passkey: bare 5-digit | 2.134, {1,2,3} | 2.000, {2} | same |
| **Digit tokenization invariant to a leading space?** | **NO — 0.0%** | **YES — 100%** | **the real finding** |
| **Phonebook entries in a 4,096-tok context** | **325.2** | **314.2** | LFM2 fits **3.4% fewer** |
| phonebook tokens / entry (mean) | 12.595 | 13.035 | +3.5% |
| Text in a 4,096-token window | 18,564 chars / 3,015 words | 18,597 chars / 3,020 words | +0.18% |
| **Rows with <100 updates over a 2B-token run** (200M-token estimate) | 1,033 (**2.06%**) | **19,108 (29.7%)** | **18.5× more dead rows** |
| ↑ at 5B tokens | 636 (1.27%) | 14,498 (22.5%) | 22.8× |
| dead-embedding params @ d=1024, tied, 2B run | 1.06 M | **19.57 M** | **+18.5 M inert params (5.3% of 350M)** |
| MQAR affected by tokenizer? | — | — | **REFUTED — synthetic vocab 256** |

**One-line read: fertility is a non-issue (0.46%, and in LFM2's *favour*). Two things are real:
(1) GPT-2's digit tokenization is context-dependent and non-deterministic, which is a
*confound for the passkey endpoint itself*, not a fertility problem — and LFM2 is the
better tokenizer here; (2) LFM2 leaves ~29.7% of its embedding table under-trained
(<100 updates over a 2B-token run) on English-only FineWeb-Edu — ~19.6M inert parameters,
5.6% of a 350M ledger, vs 0.30% under GPT-2. Neither threatens the endpoints; (2) is a
parameter-accounting issue, fixed by reporting embedding and non-embedding params separately.**

---

## Methods

### Tokenizers obtained (Step 0)

Downloaded directly from HF over HTTPS **from the FarmShare login node — HF is reachable**,
no local fetch/scp needed. All are `tokenizer.json` + `tokenizer_config.json`, total 4.7 MB
on disk. Loaded with `tokenizers.Tokenizer.from_file` — no weights, no network at load.

| name | repo | vocab (incl. added) | model | pre-tokenizer digit rule |
|---|---|---:|---|---|
| `gpt2` | `openai-community/gpt2` | 50,257 | byte-level BPE, 50,000 merges | ByteLevel regex ` ?\p{N}+` → **greedy, unbounded** |
| `lfm2` | `LiquidAI/LFM2-350M` | **64,400** | BPE, 63,683 merges, 507 added | Split `…\|\p{N}{1,3}\|…` **Isolated** + ByteLevel |
| `olmo2` | `allenai/OLMo-2-1124-7B` | 100,278 | BPE, 100,000 merges | `\p{N}{1,3}`, **Removed/invert** |
| `neox` | `EleutherAI/gpt-neox-20b` | 50,277 (base 50,254) | BPE + NFC normalizer | ByteLevel `use_regex:true` → same as GPT-2 |
| `qwen25` | `Qwen/Qwen2.5-0.5B` | 151,665 | BPE + NFC | `\p{N}` — **one token per digit** |

**Could NOT obtain:** `meta-llama/Llama-3.2-1B` — HTTP **401**, gated. Skipped as instructed.

> **FINDING (MEASURED) — the design doc's "65,536" is not LFM2's vocab.**
> LFM2-350M's tokenizer has **64,400** entries (63,893 base + 507 added), not 65,536.
> Verified identical across `LFM2-350M` and `LFM2.5-1.2B-Base` (byte-identical 4,732,426-byte
> `tokenizer.json`). The model config pads the embedding matrix to 65,536 for hardware
> alignment; **1,136 rows (1.73%) are padding that can never be emitted by the tokenizer**
> and are dead by construction, before any data-dependent deadness. The parent should state
> "65,536 padded / 64,400 usable", not "vocab 65,536".

### Step 1 — text recovery and round-trip validation

`val.npy` (8,000,000 uint16 GPT-2 ids) memmapped; 7,702 EOS(50256) markers found. Documents
taken as the spans strictly *between* consecutive EOS markers, so the truncated head and tail
are excluded and **every document used is complete**. Docs shorter than 16 tokens dropped.

**Round-trip check: 7,700 / 7,700 documents satisfy `gpt2.encode(gpt2.decode(ids)).ids == ids`
exactly. Failure rate 0.000% (0/7700).** This is the expected result for byte-level BPE on its
own output, and it means the text I measure on is *provably* our corpus, bit-for-bit.
Only round-tripping docs were carried forward (all 7,700).

**Sample size:** 7,700 documents · **36,576,972 characters** · 36,739,535 UTF-8 bytes ·
5,965,203 whitespace words · 7,988,924 GPT-2 tokens. Mean 4,750 chars/doc.
This is the *entire* validation split — not a subsample — so there is no sampling error in
the corpus selection, only document-level variance, which the bootstrap captures.

Scripts: `/scratch/users/ericrcwu/liv/tok/fertility.py` (263 lines, run with `MAX_DOCS=7700`),
`/scratch/users/ericrcwu/liv/tok/digits_ctx.py`, `/scratch/users/ericrcwu/liv/tok/vocab_util.py`.
Raw output: `results.json`, `digits_ctx.json`, `vocab_util.json` (all pasted below).

---

## Step 2 — Headline fertility (MEASURED, n=7,700 docs)

| tokenizer | vocab | tok/doc mean | median | p90 | tok/1k chars | tok/word | bytes/tok | **ratio vs GPT-2** | 95% CI (bootstrap, 5,000 resamples over docs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 50,257 | 1037.5 | 595 | 1928.3 | 218.41 | 1.3393 | 4.599 | 1.0000 | — |
| **lfm2** | **64,400** | 1032.7 | 591 | 1925.0 | 217.41 | 1.3331 | 4.620 | **0.99539** | **[0.99402, 0.99672]** |
| olmo2 | 100,278 | 1013.7 | 580 | 1889.1 | 213.40 | 1.3085 | 4.707 | 0.97705 | [0.97590, 0.97824] |
| neox | 50,277 | 1035.7 | 595 | 1925.3 | 218.02 | 1.3368 | 4.607 | 0.99820 | [0.99721, 0.99920] |
| qwen25 | 151,665 | 1035.9 | 588 | 1924.1 | 218.08 | 1.3372 | 4.606 | 0.99847 | [0.99641, 1.00066] |

Per-document ratio distribution for LFM2: mean 0.9930, median 0.9931,
**p05–p95 = [0.9418, 1.0444]**. **38.5% of documents need *more* LFM2 tokens than GPT-2
tokens** — i.e. the aggregate advantage is small and the sign flips on a large minority of
documents. This is the honest picture: the two tokenizers are, on English educational web
text, **interchangeable to within half a percent**.

**Implied corpus size:** our 1,200,000,000 GPT-2 tokens = **1,194,462,583 LFM2 tokens** for the
identical text — a **5.5M-token (0.46%) reduction**. At a 2B-token budget this is a 9.2M-token
difference: **irrelevant to the token ledger**, far below run-to-run noise.

**Direction of the effect, measured not assumed:** the task brief warned a multilingual 65k
tokenizer might be *worse* on English. It is not — it is very slightly *better* (0.46% fewer
tokens), and the CI excludes 1.0, so the direction is statistically resolved even though the
magnitude is trivial. Interestingly the *largest* vocab tested (Qwen, 151,665) is **not** the
most efficient on this text (0.9985); OLMo-2's 100,278-entry Dolma2 tokenizer is
(0.9771, −2.3%), which is unsurprising since it was trained on Dolma — the same web-text
distribution family as FineWeb-Edu. **Vocabulary size past ~50k buys almost nothing on
English; training-corpus match buys ~2%.**

> **Side note relevant to the design doc's *intended* corpus.** The doc points at
> `s3://edullm-datasets/olmo-150b-dolma2/`. The Dolma2 tokenizer (100,278) is the most
> efficient of the five on this text at ratio 0.9771 [0.9759, 0.9782] — **2.3% fewer tokens
> than GPT-2, and 1.8% fewer than LFM2**. If the study ever switches to the Dolma2 corpus it
> should use the Dolma2 tokenizer, and that is a *larger* fertility change than the
> GPT-2→LFM2 one it is currently worrying about — though still under 3%.

---

## Step 3 — Endpoint-specific measurements

### (a) Digits and passkeys — the one place tokenizers genuinely differ

Distribution over **1,000 random keys per digit-length** (not one example).

**Bare key (no leading space):**

| tokenizer | 5-digit | 6-digit | 7-digit | 7-digit length set | deterministic? |
|---|---:|---:|---:|---|---|
| gpt2 | 2.143 | 2.696 | **3.090** | **{2, 3, 4}** (15 / 880 / 105) | **NO** |
| **lfm2** | 2.000 | 2.000 | **3.000** | **{3}** (1000 / 1000) | **YES** |
| olmo2 | 2.000 | 2.000 | 3.000 | {3} | YES |
| neox | 2.077 | 2.617 | 3.038 | {2, 3, 4} | NO |
| qwen25 | **5.000** | **6.000** | **7.000** | {7} | YES (1 tok/digit) |

The brief's hypothesis was "LFM2 splits digits individually → 7 tokens vs GPT-2's ~3".
**REFUTED.** LFM2 uses `\p{N}{1,3}`, i.e. **3-digit chunking**, giving `ceil(nd/3)` = 3 tokens
for a 7-digit key — *the same mean as GPT-2's 3.09*. The tokenizer that splits individually is
**Qwen** (`\p{N}`, 7 tokens). So on raw count the GPT-2↔LFM2 passkey difference is **0.09
tokens on a 7-digit key (2.9%)** — nothing.

**The real difference is variance and context-dependence, and it is larger than it looks:**

| property (7-digit, n=1000) | gpt2 | lfm2 | olmo2 | neox | qwen25 |
|---|---:|---:|---:|---:|---:|
| distinct token-counts the same key length can produce | **3** | **1** | 1 | 3 | 1 |
| tokens for `"<key>"` | 3.084 | 3.000 | 3.000 | 3.031 | 7.000 |
| tokens for `" <key>"` (leading space) | 3.243 | 4.000 | 4.000 | 3.171 | 8.000 |
| **fraction where bare ids are an exact suffix of spaced ids** | **0.000** | **1.000** | **1.000** | 0.003 | **1.000** |
| fraction where bare and spaced have the same length | 0.759 | 0.000 | 0.000 | 0.795 | 0.000 |

> **FINDING (MEASURED) — GPT-2 has zero digit-tokenization invariance; LFM2 has perfect
> invariance.** For **0 out of 1,000** random 7-digit keys does GPT-2 tokenize the digits the
> same way with and without a preceding space: its regex ` ?\p{N}+` lets the space **join and
> re-segment** the numeric run. For LFM2 (and OLMo2, Qwen) the space is `Isolated` into its own
> pre-token, so the key's own token ids are **identical in every context — 1,000/1,000**.
>
> **This matters for the passkey/needle endpoint specifically.** A passkey task writes the key
> once in the haystack (`"The pass key is 4821973."`, space-preceded) and expects it at the
> answer position (often line-initial or quote-preceded). Under GPT-2 those are **different
> token sequences for the same number**, so the model must generalize across segmentations —
> and *how much* it must generalize varies per key (2, 3, or 4 tokens). Under LFM2 the
> retrieval target is a fixed 3-token string every time.
>
> Consequence for the study: this is **not a reason to prefer GPT-2**. It is a reason to note
> that **GPT-2 makes the passkey endpoint noisier and slightly harder**, in a
> *key-dependent* way. If the study runs passkey under GPT-2, per-key difficulty is
> heterogeneous (some keys are 2 tokens, some 4) and that heterogeneity is a nuisance variance
> component that LFM2 would eliminate. Also verified: `"\n" + key` and `'"' + key` both add
> exactly 1 token for every tokenizer, so the effect is specific to the space-joining rule,
> and GPT-2's `\n`/`"` cases inherit the bare (variable) segmentation.

**Mitigation if the study stays on GPT-2 (ASSUMED, standard practice):** pad passkeys to a
fixed digit count and always emit them in an identical local context (same preceding
character) in both haystack and prompt; or use the space-preceded form uniformly. That
reduces but does not remove the per-key length variability (`{2,3,4}` persists in the
space-preceded case too: measured set `{2,3,4}` with mean 3.243).

### (b) Phonebook — the directly-quotable capacity number

200 synthetic entries, varied name length (20 first × 20 last names, incl. multi-token
non-English names) and 5 number formats (`617-555-0142`, `(617) 555-0142`, `+1-617-555-0142`,
`617.555.0142`, `6175550142`), one per line with trailing `\n`.

| tokenizer | tok/entry mean | median | p90 | min–max | **entries in 4,096 ctx** | in 2,048 | in 8,192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 12.595 | 13 | 15 | 8–18 | **325.2** | 162.6 | 650.4 |
| **lfm2** | **13.035** | 13 | 16 | 9–18 | **314.2** | 157.1 | 628.5 |
| olmo2 | 12.700 | 13 | 15 | 9–18 | 322.5 | 161.3 | 645.0 |
| neox | 12.780 | 13 | 16 | 8–18 | 320.5 | 160.3 | 641.0 |
| qwen25 | 18.700 | 19 | 21 | 15–24 | 219.0 | 109.5 | 438.1 |

**LFM2 fits 314 entries per 4K context vs GPT-2's 325 — 3.4% fewer, a difference of 11
entries.** Note the sign **flips** relative to natural text: on prose LFM2 is 0.46% *cheaper*,
but on digit-dense phonebook lines it is 3.5% *more expensive*, because its rigid `\p{N}{1,3}`
chunking cannot exploit GPT-2's opportunistic longer numeric merges. This is the single
largest GPT-2↔LFM2 gap I measured on any endpoint, and it is **3.4%** — well under the
effect sizes a recall study is powered to detect. Qwen would be a genuine problem here
(−33% capacity); LFM2 is not.

Isolated number formats (n=500 each): dashed `NNN-NNN-NNNN` → gpt2 **6.494 ± 0.595**, lengths
`{5,6,7,8}`; lfm2 **6.000 ± 0.000**, lengths `{6}`. Same story as (a): equal mean,
**zero variance for LFM2 vs 4 distinct lengths for GPT-2**.

### (c) Needle-in-a-haystack — effective distance for the same text

300 contiguous non-overlapping 4,096-GPT-2-token windows drawn from the concatenated corpus
stream. Each window's text: **18,564 ± 1,141 characters, 3,015 ± 187 words**.

| tokenizer | tokens for that same text | ratio | 95% CI (2,000 bootstraps) | **chars in a 4,096-token budget** | **words in 4,096 tokens** |
|---|---:|---:|---|---:|---:|
| gpt2 | 4,095.9 ± 0.3 | 1.0000 | — | 18,564 | 3,014.6 |
| **lfm2** | 4,088.6 ± 83.7 | **0.99820** | [0.99591, 1.00054] | **18,597** | **3,019.9** |
| olmo2 | 4,010.3 ± 76.9 | 0.97907 | [0.97701, 0.98134] | 18,961 | 3,078.9 |
| neox | 4,092.7 ± 64.1 | 0.99919 | [0.99740, 1.00091] | 18,579 | 3,017.0 |
| qwen25 | 4,109.2 ± 139.5 | 1.00323 | [0.99958, 1.00708] | 18,504 | 3,004.8 |

At a 4,096-token training length, **LFM2 holds 5 more words (0.18%) than GPT-2** —
3,020 vs 3,015. The needle's "effective retention distance in text" is **identical for
practical purposes**. Note the CI for LFM2 here spans 1.0 (n=300 windows, wider than the
7,700-doc estimate), which only reinforces that the effect is indistinguishable from zero on
long contiguous spans.

### (d) Conv receptive field in semantic terms

k-wide causal depthwise conv reaches `k−1` tokens/layer; stacked over `L` LIV layers → `(k−1)·L`.
Using measured chars/token (gpt2 4.5785, lfm2 4.5997) and words/token (0.7467, 0.7502).

| k | reach (tokens) | GPT-2 chars / words | LFM2 chars / words | LFM2 advantage |
|---:|---:|---|---|---:|
| **L = 10 LIV layers** | | | | |
| 3 | 20 | 92 ch / **14.93 w** | 92 ch / **15.00 w** | +0.46% |
| 5 | 40 | 183 ch / 29.87 w | 184 ch / 30.01 w | +0.46% |
| 9 | 80 | 366 ch / 59.74 w | 368 ch / 60.02 w | +0.46% |
| 15 | 140 | 641 ch / 104.5 w | 644 ch / 105.0 w | +0.46% |
| **L = 16 layers** | | | | |
| 3 | 32 | 147 ch / 23.89 w | 147 ch / 24.00 w | +0.46% |
| 5 | 64 | 293 ch / 47.79 w | 294 ch / 48.02 w | +0.46% |
| 9 | 128 | 586 ch / 95.58 w | 589 ch / 96.03 w | +0.46% |
| 15 | 224 | 1,026 ch / 167.3 w | 1,030 ch / 168.0 w | +0.46% |

**Can the relative ordering of k=3/5/9/15 change? NO — and this is a structural argument, not
an empirical one.** The map tokens→words is a single scalar multiply (0.7467 vs 0.7502 words
per token) applied to *every* k identically. A strictly positive scalar multiple is a
monotone increasing bijection on the reach axis, so it preserves order: if
`reach_w(k=3) < reach_w(k=5)` under GPT-2, the same holds under LFM2, for any positive
chars/token ratio whatsoever. **No fertility difference — 0.46%, 20%, or 200% — can reorder
k=3/5/9/15.**

The honest caveat, stated because it is the *only* way fertility could bite here: what
fertility *can* do is shift **where a k-sweep crosses an absolute, text-defined threshold**.
If the true phenomenon is "the conv must span a typical clause of ~15 words", then the value
of k at which reach first exceeds 15 words could in principle differ between tokenizers. At
0.46% it does not: k=3/L=10 gives 14.93 words under GPT-2 and 15.00 under LFM2 — nominally
straddling 15, but that is a coincidence of the threshold I picked, and the gap is 0.07 words.
**A fertility difference would need to be ~15–20% before it could plausibly move a k-sweep
conclusion by one rung.** Ours is 0.46%. Not a threat.

### (e) MQAR — CONFIRMED unaffected

**CONFIRMED: MQAR does not touch any natural-language tokenizer.** Evidence, read not run:

- `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/mqar/mqar_data.py`
  — the full import list is `__future__`, `dataclasses`, `typing`, `numpy`, `torch`. **No
  `tokenizers`, no `transformers`, no `tiktoken`.** Verified by grepping all imports across
  `mqar_data.py`, `mqar_model.py`, `mqar_calibrate.py`: the union is
  `{__future__, typing, torch, torch.nn, torch.nn.functional, olmo_core…ShortConv, argparse,
  json, sys, time, pathlib, dataclasses, numpy, mqar_data, mqar_model}` — zero tokenizer deps.
- `mqar_data.py:170` — `CALIBRATED_VOCAB = 256`, and all difficulty points at lines 213–216 and
  223–224 pass `vocab_size=CALIBRATED_VOCAB`.
- `mqar_data.py:100-101` — tokens are drawn as integers from a synthetic space:
  `d, n, v = cfg.num_pairs, cfg.seq_len, cfg.vocab_size` / `half = v // 2  # keys are [0, half), values are [half, v)`,
  then `keys = torch.randperm(half, …)[:d]` and `values = torch.randint(half, v - 1, …)`.
  Symbols are integer ids with no text realization.
- `mqar_model.py:97` — `self.embed = nn.Embedding(vocab_size, d_model)` with `vocab_size` passed
  straight from the MQAR config (`mqar_calibrate.py:83`), so the embedding table is 256 rows.

The operating point `N512_D64` (seq_len 512, 64 pairs, vocab 256) is entirely synthetic.
**The tokenizer decision has no bearing on the MQAR endpoint.**

---

## Step 4 — Vocabulary utilization and dead embeddings

### Small-sample (validation split: 7.99M GPT-2 tokens / 36.6M chars)

| tokenizer | vocab | rows used | **frac used** | rows never seen | seen exactly once | entropy (bits/tok) | bits/char | top-10 share | top-1k share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 50,257 | 47,758 | **95.03%** | **2,499 (4.97%)** | 1,599 | 11.012 | 2.405 | 24.34% | 63.93% |
| **lfm2** | **64,400** | 42,086 | **65.35%** | **22,314 (34.65%)** | 2,747 | 11.124 | 2.418 | 22.37% | 63.08% |
| olmo2 | 100,278 | 60,683 | 60.51% | 39,595 (39.49%) | 5,946 | 11.167 | 2.383 | 22.78% | 63.15% |
| neox | 50,277 | 44,925 | 89.35% | 5,352 (10.64%) | — | — | — | — | — |
| qwen25 | 151,665 | ~54,268* | ~35.8%* | ~97,400* | — | — | — | — | — |

\* Qwen figures from the 4M-token smoke run; the 8M-token value is in `results.json`.

> **SUPERSEDED — see the 200M-token section below.** At 8M tokens 22,314 of LFM2's 64,400 rows
> (34.65%) appear unseen vs 2,499 of 50,257 (4.97%) for GPT-2. **This overstates deadness**:
> at 200M tokens only 11.16% of LFM2's rows are truly never seen. The 8M sample cannot
> distinguish "rare" from "absent" in a Zipfian tail. Retained here only to show the
> sample-size sensitivity; **the 200M numbers are the ones to quote.**

**Extrapolation to a training run (INFERRED — SUPERSEDED by the 200M section; kept to document
why the larger sample was necessary).**
*Assumption:* the unigram token distribution of our 8M-token sample is stationary and
representative of the full 1.2B-token corpus, so a token with sample probability `p` receives
`p·T` updates over a `T`-token run. This is the standard assumption and it is **optimistic in
LFM2's favour** for the head and pessimistic in the tail (Zipfian tails are under-sampled at
8M tokens, so *some* of the 22,314 "never seen" rows would appear at 1.2B). I bound that
below.

Because every row that is unseen in the sample has estimated `p = 0` exactly, the naive
`p·T < 100` count equals the never-seen count for any `T`:

| | GPT-2 | **LFM2** | OLMo2 |
|---|---:|---:|---:|
| rows with est. <100 updates at T = 2B | 2,499 (4.97%) | **22,314 (34.65%)** | 39,595 (39.49%) |
| rows with est. <100 updates at T = 5B | 2,499 (4.97%) | **22,314 (34.65%)** | 39,595 (39.49%) |

**Bounding the extrapolation error (rule of three).** A token unseen in `S` tokens has true
rate `< 3/S` at 95% confidence. With `S ≈ 8M`, that is `< 3.75e-7`, so at `T = 2B` an unseen
row's occurrence count is **< 751 (95% upper bound)** — i.e. the sample cannot by itself rule
out that an unseen row crosses 100 updates. **This is why I ran a 25× larger sample**
(below): at `S = 200M`, the 95% bound at `T = 2B` drops to `< 30` updates, which *does*
establish deadness against a 100-update threshold.

**Entropy (8M sample).** Unigram entropy is nearly identical across tokenizers: 11.01 (gpt2)
vs 11.12 (lfm2) bits/token → **2.405 vs 2.418 bits/char**, a 0.5% difference consistent with
the fertility gap. **Neither tokenizer compresses our corpus meaningfully better in an
information-theoretic sense.** Confirmed at 200M tokens below.

### Large-sample (200M GPT-2 tokens from `train.npy`) — COMPLETED, and it revises the above

`vocab_util.py` streamed **200,000,000 GPT-2 tokens** (918,669,467 characters) from
`train.npy` in 2M chunks, decoding and re-encoding with all five tokenizers, accumulating only
bincounts. Runtime 586 s on the login node. **This is a 25× larger and fully independent
sample (train split, not val), so it supersedes the 8M-token numbers above.** At `S = 200M`, a
row needs **≥10 sample occurrences** to clear 100 updates at `T = 2B` — real resolution, no
`p = 0` degeneracy.

**Fertility replicates.** Token totals on the same 918.7M characters:

| tokenizer | tokens for 200M-GPT-2-token text | ratio vs GPT-2 | chars/token |
|---|---:|---:|---:|
| gpt2 | 199,806,313 | 1.00000 | 4.5978 |
| **lfm2** | 198,224,760 | **0.99208** | 4.6345 |
| olmo2 | 194,579,963 | 0.97384 | 4.7213 |
| neox | 199,158,876 | 0.99676 | 4.6127 |
| qwen25 | 198,710,493 | 0.99452 | 4.6232 |

LFM2 is **0.79% cheaper** than GPT-2 here vs 0.46% on the val split — same sign, same order of
magnitude, still negligible. The val-split CI [0.9940, 0.9967] does not contain 0.9921, so
there is real between-split variation of a few tenths of a percent; **the honest statement is
"LFM2 costs 0.5–0.8% fewer tokens than GPT-2 on FineWeb-Edu", not a single 4-decimal number.**
Every conclusion in Steps 2–4 is unchanged by this.

**Vocabulary utilization at 200M tokens (MEASURED — use these, not the 8M numbers):**

| tokenizer | vocab | rows used | frac used | **never seen** | <10 counts | <100 counts | entropy b/tok | bits/char |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 50,257 | 49,866 | **99.22%** | **391 (0.78%)** | 1,033 | 5,821 | 11.028 | 2.3986 |
| **lfm2** | **64,400** | 57,210 | **88.84%** | **7,190 (11.16%)** | 19,108 | 27,194 | 11.152 | 2.4064 |
| olmo2 | 100,278 | 81,483 | 81.26% | 18,795 (18.74%) | 33,338 | 51,668 | 11.199 | **2.3720** |
| neox | 50,277 | 48,711 | 96.89% | 1,566 (3.11%) | 2,992 | 9,101 | 11.051 | 2.3958 |
| qwen25 | 151,665 | 92,735 | 61.14% | 58,930 (38.86%) | 85,054 | 104,182 | 11.043 | 2.3887 |

**Dead-embedding estimates (INFERRED from the 200M sample; assumption: stationary unigram
distribution, `expected updates = p · T`).** A row is "dead" if it expects <100 updates.

| | GPT-2 50,257 | **LFM2 64,400** | OLMo2 100,278 | Qwen 151,665 |
|---|---:|---:|---:|---:|
| **dead (<100 upd) at T = 2B** | **1,033 (2.06%)** | **19,108 (29.67%)** | 33,338 (33.25%) | 85,054 (56.08%) |
| dead (<100 upd) at T = 5B | 636 (1.27%) | 14,498 (22.51%) | 27,300 (27.22%) | 76,662 (50.55%) |
| expects literally 0 updates | 391 | 7,190 | 18,795 | 58,930 |
| <1,000 updates at T = 2B | 5,821 (11.6%) | 27,194 (42.2%) | 51,471 | 104,182 |

**Dead-parameter cost at the frozen 350M design (`d_model = 1024`, tied embeddings):**

| | vocab rows | emb params | dead rows @2B | **dead params** | % of 350M |
|---|---:|---:|---:|---:|---:|
| GPT-2 50,257 | 50,257 | 51.46 M | 1,033 | **1.06 M** | 0.30% |
| **LFM2 64,400** | 64,400 | 65.95 M | 19,108 | **19.57 M** | **5.59%** |
| LFM2 padded to 65,536 | 65,536 | 67.11 M | 19,108 + 1,136 = 20,244 | **20.73 M** | **5.92%** |

> **REVISED FINDING (MEASURED at 200M tokens) — the dead-embedding gap is real but smaller
> than the 8M-token sample suggested, and it is now properly resolved.** At 8M tokens I
> measured 34.65% of LFM2's rows unseen; at 200M tokens only **11.16% are never seen**, and
> **29.67% fall below 100 expected updates over a 2B-token run**. The 8M number conflated
> "rare" with "absent" — this is exactly the Zipfian-tail under-sampling I flagged, and the
> 25× sample corrects it. **Use 29.7% / 19.6M params, not 34.6% / 22.9M.**
>
> The comparison still stands strongly: **LFM2 has 18.5× more sub-100-update rows than GPT-2
> (19,108 vs 1,033)**, and switching to LFM2's vocab adds 14.5M embedding parameters of which
> **~18.5M net are inert** (the dead-row delta exceeds the parameter delta because GPT-2's
> rows are nearly all live). At the frozen tied-350M design that is **~5.6–5.9% of the
> parameter budget doing nothing** vs 0.30% under GPT-2.
>
> **Why this matters and why it mostly doesn't.** It matters for *parameter-matched
> comparison bookkeeping*: two arms "both 350M" are not matched on trained parameters if their
> vocabs differ, and a reviewer will notice ~6% inert. It does *not* plausibly matter for the
> endpoints — dead rows for tokens that never appear cannot affect recall, length
> extrapolation, or AR-Hits perplexity on this corpus. The fix, if the study wants one, is to
> **report embedding params and non-embedding params separately** (standard practice) rather
> than to change tokenizer.

**Entropy at 200M tokens:** 11.028 (gpt2) vs 11.152 (lfm2) bits/token → **2.3986 vs 2.4064
bits/char**, a 0.33% difference. OLMo2 remains the best compressor at 2.3720 bits/char.
Head-of-distribution behaviour is nearly identical across all five (top-1k share 0.63–0.65,
top-10k share 0.884–0.894), confirming that the vocab-size differences live entirely in the
tail — which is precisely the dead-row story.

---

## RAW RESULT JSON

`/scratch/users/ericrcwu/liv/tok/results.json` (key sections; full file on FarmShare):

```json
{
 "meta": {"started": "2026-08-01T11:18:49", "finished": "2026-08-01T11:19:21",
          "max_docs_req": 7700, "source": "/scratch/users/ericrcwu/kda/lm/data/val.npy",
          "host": "rice-03"},
 "vocab": {"gpt2": {"with_added": 50257, "base": 50257},
           "lfm2": {"with_added": 64400, "base": 64400},
           "olmo2": {"with_added": 100278, "base": 100278},
           "neox": {"with_added": 50277, "base": 50254},
           "qwen25": {"with_added": 151665, "base": 151643}},
 "roundtrip": {"n_docs": 7700, "n_fail": 0, "fail_rate": 0.0, "example_fail_idx": []},
 "corpus_sample": {"n_docs": 7700, "total_chars": 36576972, "total_bytes": 36739535,
                   "total_words": 5965203, "mean_chars_per_doc": 4750.256103896104},
 "fertility": {
  "gpt2":  {"total_tokens": 7988924, "tokens_per_doc_mean": 1037.5225974025975,
            "tokens_per_doc_median": 595.0, "tokens_per_doc_p90": 1928.300000000001,
            "tokens_per_1000_chars": 218.41403383527756, "tokens_per_word": 1.3392543388716194,
            "bytes_per_token": 4.5988089259579885, "chars_per_token": 4.578460378393887,
            "ratio_vs_gpt2_agg": 1.0, "implied_corpus_tokens_for_1.2B_gpt2": 1200000000.0},
  "lfm2":  {"total_tokens": 7952059, "tokens_per_doc_mean": 1032.7349350649351,
            "tokens_per_doc_median": 591.0, "tokens_per_doc_p90": 1925.0,
            "tokens_per_1000_chars": 217.40615926326544, "tokens_per_word": 1.3330743312507554,
            "bytes_per_token": 4.620128572989713, "chars_per_token": 4.599685691466826,
            "ratio_vs_gpt2_agg": 0.9953854862056517,
            "ratio_vs_gpt2_ci95": [0.9940174513154197, 0.9967207518572225],
            "ratio_per_doc_mean": 0.9929949792805997, "ratio_per_doc_median": 0.9930675701653211,
            "ratio_per_doc_p05_p95": [0.9417945704006182, 1.0443516133573731],
            "frac_docs_more_tokens_than_gpt2": 0.3845454545454545,
            "implied_corpus_tokens_for_1.2B_gpt2": 1194462583.446782},
  "olmo2": {"total_tokens": 7805565, "tokens_per_1000_chars": 213.4010710345296,
            "tokens_per_word": 1.3085162399334942, "bytes_per_token": 4.706838646529752,
            "ratio_vs_gpt2_agg": 0.9770483484384129,
            "ratio_vs_gpt2_ci95": [0.9758966341541677, 0.9782366901381101],
            "frac_docs_more_tokens_than_gpt2": 0.15012987012987014,
            "implied_corpus_tokens_for_1.2B_gpt2": 1172458018.1260955},
  "neox":  {"total_tokens": 7974561, "tokens_per_1000_chars": 218.02135507553768,
            "bytes_per_token": 4.607091851200336, "ratio_vs_gpt2_agg": 0.9982021358570942,
            "ratio_vs_gpt2_ci95": [0.99721270278366, 0.9992021929207193],
            "implied_corpus_tokens_for_1.2B_gpt2": 1197842563.028513},
  "qwen25":{"total_tokens": 7976664, "tokens_per_1000_chars": 218.0788502667744,
            "bytes_per_token": 4.605877218847378, "ratio_vs_gpt2_agg": 0.9984653753121197,
            "ratio_vs_gpt2_ci95": [0.9964066080529551, 1.0006555823915957],
            "ratio_per_doc_p05_p95": [0.931816631526727, 1.0756651093374863],
            "implied_corpus_tokens_for_1.2B_gpt2": 1198158450.3745437}},

 "digits": {
  "5_digit": {"gpt2":   {"bare_mean": 2.143, "bare_hist": {"1":1,"2":855,"3":144}},
              "lfm2":   {"bare_mean": 2.000, "bare_hist": {"2":1000}},
              "olmo2":  {"bare_mean": 2.000, "bare_hist": {"2":1000}},
              "neox":   {"bare_mean": 2.077, "bare_hist": {"2":923,"3":77}},
              "qwen25": {"bare_mean": 5.000, "bare_hist": {"5":1000},
                         "always_one_token_per_digit": true}},
  "6_digit": {"gpt2":   {"bare_mean": 2.696, "bare_hist": {"2":306,"3":692,"4":2}},
              "lfm2":   {"bare_mean": 2.000, "bare_hist": {"2":1000}},
              "olmo2":  {"bare_mean": 2.000, "bare_hist": {"2":1000}},
              "neox":   {"bare_mean": 2.617, "bare_hist": {"2":383,"3":617}},
              "qwen25": {"bare_mean": 6.000, "bare_hist": {"6":1000}}},
  "7_digit": {"gpt2":   {"bare_mean": 3.090, "bare_hist": {"2":15,"3":880,"4":105}},
              "lfm2":   {"bare_mean": 3.000, "bare_hist": {"3":1000}},
              "olmo2":  {"bare_mean": 3.000, "bare_hist": {"3":1000}},
              "neox":   {"bare_mean": 3.038, "bare_hist": {"2":24,"3":914,"4":62}},
              "qwen25": {"bare_mean": 7.000, "bare_hist": {"7":1000}}}},

 "digit_rule_has_pN_1_3": {"gpt2": false, "lfm2": true, "olmo2": true,
                           "neox": false, "qwen25": false},

 "phonebook": {"n_entries": 200,
  "example": ["Jae Kowalski: +1-895-475-8721\n", "Tom Smith: (897) 457-9198\n",
              "Mohammed Novak: +1-216-748-5039\n", "Tom Kim: 739-468-7032\n",
              "Priya Smith: 8138672805\n"],
  "per_tok": {
   "gpt2":   {"tokens_per_entry_mean": 12.595, "median": 13.0, "p90": 15.0,
              "min": 8, "max": 18, "entries_in_4096_ctx": 325.20841603811033,
              "entries_in_2048_ctx": 162.60420801905516, "entries_in_8192_ctx": 650.4168320762207},
   "lfm2":   {"tokens_per_entry_mean": 13.035, "median": 13.0, "p90": 16.0,
              "min": 9, "max": 18, "entries_in_4096_ctx": 314.23091676256234,
              "entries_in_2048_ctx": 157.11545838128117, "entries_in_8192_ctx": 628.4618335251247},
   "olmo2":  {"tokens_per_entry_mean": 12.700, "entries_in_4096_ctx": 322.5196850393701},
   "neox":   {"tokens_per_entry_mean": 12.780, "entries_in_4096_ctx": 320.5007824726135},
   "qwen25": {"tokens_per_entry_mean": 18.700, "entries_in_4096_ctx": 219.0374331550802}}},

 "needle": {"n_windows": 300, "gpt2_window_tokens": 4096,
  "chars_per_window_mean": 18563.92, "chars_per_window_sd": 1141.0629256384887,
  "words_per_window_mean": 3014.5033333333336, "words_per_window_sd": 186.5816978936811,
  "per_tok": {
   "gpt2":   {"tokens_for_same_text_mean": 4095.9066666666668, "ratio_vs_gpt2": 0.9999772135416667,
              "chars_in_4096_tok": 18564.34301562861, "words_in_4096_tok": 3014.5720247531685},
   "lfm2":   {"tokens_for_same_text_mean": 4088.633333333333, "tokens_for_same_text_sd": 83.69352158653355,
              "ratio_vs_gpt2": 0.9982014973958333,
              "ratio_ci95": [0.9959126586914063, 1.000536376953125],
              "chars_in_4096_tok": 18597.367413724227, "words_in_4096_tok": 3019.934693744446},
   "olmo2":  {"tokens_for_same_text_mean": 4010.266666666667, "ratio_vs_gpt2": 0.9790690104166667,
              "ratio_ci95": [0.9770115356445312, 0.9813355509440105],
              "chars_in_4096_tok": 18960.788057319543, "words_in_4096_tok": 3078.948778136117},
   "neox":   {"tokens_for_same_text_mean": 4092.6633333333334, "ratio_vs_gpt2": 0.9991853841145834,
              "chars_in_4096_tok": 18579.05479317054, "words_in_4096_tok": 3016.9609976877323},
   "qwen25": {"tokens_for_same_text_mean": 4109.233333333334, "ratio_vs_gpt2": 1.0032307942708334,
              "ratio_ci95": [0.9995762532552084, 1.007079325358073],
              "chars_in_4096_tok": 18504.13694038628, "words_in_4096_tok": 3004.795457384589}}},

 "vocab_util": {
  "gpt2":  {"vocab_size": 50257, "n_used": 47758, "frac_used": 0.9502755835008059,
            "sample_tokens": 7988924, "entropy_bits_per_token": 11.011625498379717,
            "bits_per_char": 2.4050935441845125, "top10_share": 0.24342114157050435,
            "top1000_share": 0.6392835881277629, "n_seen_once": 1599,
            "dead_lt100_at_2B": 2499, "dead_lt100_frac_at_2B": 0.049724416499194145,
            "dead_lt100_at_5B": 2499, "unseen_upper_occ_at_2B": 751.039814623346},
  "lfm2":  {"vocab_size": 64400, "n_used": 42086, "frac_used": 0.6535093167701863,
            "sample_tokens": 7952059, "entropy_bits_per_token": 11.123659759189078,
            "bits_per_char": 2.4183521451966374, "top10_share": 0.22372645876998648,
            "top1000_share": 0.630757141012158, "n_seen_once": 2747,
            "dead_lt100_at_2B": 22314, "dead_lt100_frac_at_2B": 0.3464906832298137,
            "dead_lt100_at_5B": 22314, "unseen_upper_occ_at_2B": 754.5215647922129},
  "olmo2": {"vocab_size": 100278, "n_used": 60683, "frac_used": 0.6051476894234029,
            "sample_tokens": 7805565, "entropy_bits_per_token": 11.166962375177691,
            "bits_per_char": 2.3830417310652137, "n_seen_once": 5946,
            "dead_lt100_at_2B": 39595, "dead_lt100_frac_at_2B": 0.3948523105765971},
  "neox":  {"vocab_size": 50277, "n_used": 44925, "frac_used": 0.8935497344710305}}
}
```

`/scratch/users/ericrcwu/liv/tok/digits_ctx.json` (context-invariance, n=1000/length):

```json
{
 "5_digit": {
  "gpt2":   {"bare_len_mean": 2.134, "bare_len_set": [1,2,3], "space_len_mean": 2.369,
             "space_len_set": [2,3], "newline_len_mean": 3.134, "quote_len_mean": 3.134,
             "frac_same_len_bare_vs_space": 0.759, "frac_bare_is_suffix_of_space": 0.000,
             "n_distinct_lengths": 3, "deterministic_length": false},
  "lfm2":   {"bare_len_mean": 2.000, "bare_len_set": [2], "space_len_mean": 3.000,
             "space_len_set": [3], "newline_len_mean": 3.000, "quote_len_mean": 3.000,
             "frac_same_len_bare_vs_space": 0.000, "frac_bare_is_suffix_of_space": 1.000,
             "n_distinct_lengths": 1, "deterministic_length": true},
  "olmo2":  {"bare_len_mean": 2.000, "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true},
  "neox":   {"bare_len_mean": 2.082, "bare_len_set": [1,2,3], "frac_bare_is_suffix_of_space": 0.004,
             "n_distinct_lengths": 3, "deterministic_length": false},
  "qwen25": {"bare_len_mean": 5.000, "bare_len_set": [5], "space_len_mean": 6.000,
             "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true}},
 "6_digit": {
  "gpt2":   {"bare_len_mean": 2.698, "bare_len_set": [2,3,4], "space_len_mean": 2.804,
             "frac_bare_is_suffix_of_space": 0.000, "deterministic_length": false},
  "lfm2":   {"bare_len_mean": 2.000, "bare_len_set": [2], "space_len_mean": 3.000,
             "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true},
  "olmo2":  {"bare_len_mean": 2.000, "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true},
  "neox":   {"bare_len_mean": 2.637, "bare_len_set": [2,3], "frac_bare_is_suffix_of_space": 0.001,
             "deterministic_length": false},
  "qwen25": {"bare_len_mean": 6.000, "space_len_mean": 7.000, "deterministic_length": true}},
 "7_digit": {
  "gpt2":   {"bare_len_mean": 3.084, "bare_len_set": [2,3,4], "space_len_mean": 3.243,
             "frac_bare_is_suffix_of_space": 0.000, "n_distinct_lengths": 3,
             "deterministic_length": false},
  "lfm2":   {"bare_len_mean": 3.000, "bare_len_set": [3], "space_len_mean": 4.000,
             "frac_bare_is_suffix_of_space": 1.000, "n_distinct_lengths": 1,
             "deterministic_length": true},
  "olmo2":  {"bare_len_mean": 3.000, "bare_len_set": [3], "space_len_mean": 4.000,
             "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true},
  "neox":   {"bare_len_mean": 3.031, "bare_len_set": [2,3,4], "frac_bare_is_suffix_of_space": 0.003,
             "deterministic_length": false},
  "qwen25": {"bare_len_mean": 7.000, "bare_len_set": [7], "space_len_mean": 8.000,
             "frac_bare_is_suffix_of_space": 1.000, "deterministic_length": true}},
 "phone_formats": {
  "dashed": {"gpt2":   {"mean": 6.494, "sd": 0.5949487372875079, "set": [5,6,7,8],
                        "n_distinct": 4, "with_space_mean": 6.886},
             "lfm2":   {"mean": 6.000, "sd": 0.0, "set": [6], "n_distinct": 1, "with_space_mean": 7.0},
             "olmo2":  {"mean": 6.000, "sd": 0.0, "set": [6], "n_distinct": 1, "with_space_mean": 7.0},
             "neox":   {"mean": 6.306, "sd": 0.5141633981527662, "set": [5,6,7,8], "n_distinct": 4,
                        "with_space_mean": 6.592},
             "qwen25": {"mean": 12.000, "sd": 0.0, "set": [12], "n_distinct": 1, "with_space_mean": 13.0}}}
}
```

`/scratch/users/ericrcwu/liv/tok/vocab_util.json` (200M-token run — **the authoritative
vocab-utilization numbers**):

```json
{
 "sample_gpt2_tokens": 200000000, "sample_chars": 918669467, "elapsed_s": 586,
 "per_tok": {
  "gpt2": {"vocab_size": 50257, "sample_tokens": 199806313, "n_used": 49866,
           "frac_used": 0.9922199892552281, "n_never_seen": 391,
           "entropy_bits_per_token": 11.028146572535654, "chars_per_token": 4.597800005448276,
           "bits_per_char": 2.3985703074225895, "n_count_lt10": 1033, "n_count_lt100": 5821,
           "top10_share": 0.24172812797961996, "top1k_share": 0.6392821381975051,
           "top10k_share": 0.8920214547975771,
           "dead_lt100_at_2B": 1033, "dead_lt100_frac_at_2B": 0.020554350637722107,
           "dead_lt1000_at_2B": 5821, "dead_zero_at_2B": 391,
           "dead_lt100_at_5B": 636, "dead_lt100_frac_at_5B": 0.012654953538810514,
           "dead_lt1000_at_5B": 2593,
           "resolution_min_sample_count_for_100_at_2B": 9.99031565,
           "resolution_min_sample_count_for_100_at_5B": 3.99612626},
  "lfm2": {"vocab_size": 64400, "sample_tokens": 198224760, "n_used": 57210,
           "frac_used": 0.8883540372670807, "n_never_seen": 7190,
           "entropy_bits_per_token": 11.152489380744676, "chars_per_token": 4.634483941360302,
           "bits_per_char": 2.406414505229945, "n_count_lt10": 19108, "n_count_lt100": 27194,
           "top10_share": 0.22227015938876657, "top1k_share": 0.629870905128098,
           "top10k_share": 0.8941739619208015,
           "dead_lt100_at_2B": 19108, "dead_lt100_frac_at_2B": 0.2967080745341615,
           "dead_lt1000_at_2B": 27194, "dead_zero_at_2B": 7190,
           "dead_lt100_at_5B": 14498, "dead_lt100_frac_at_5B": 0.22512422360248446,
           "dead_lt1000_at_5B": 24226,
           "resolution_min_sample_count_for_100_at_2B": 9.911238},
  "olmo2": {"vocab_size": 100278, "sample_tokens": 194579963, "n_used": 81483,
            "frac_used": 0.812571052474122, "n_never_seen": 18795,
            "entropy_bits_per_token": 11.199146606759946, "chars_per_token": 4.721295311377975,
            "bits_per_char": 2.3720495898171894, "n_count_lt10": 33338, "n_count_lt100": 51668,
            "top1k_share": 0.6308901497735407, "top10k_share": 0.8840645323794207,
            "dead_lt100_at_2B": 33338, "dead_lt100_frac_at_2B": 0.33245577295119566,
            "dead_zero_at_2B": 18795, "dead_lt100_at_5B": 27300,
            "dead_lt100_frac_at_5B": 0.27224316400406867},
  "neox": {"vocab_size": 50277, "sample_tokens": 199158876, "n_used": 48711,
           "frac_used": 0.9688525568351334, "n_never_seen": 1566,
           "entropy_bits_per_token": 11.051074921207558, "chars_per_token": 4.612746795176731,
           "bits_per_char": 2.395768814529988, "n_count_lt10": 2992, "n_count_lt100": 9101,
           "dead_lt100_at_2B": 2992, "dead_lt100_frac_at_2B": 0.05951031286671838,
           "dead_zero_at_2B": 1566, "dead_lt100_at_5B": 2168,
           "dead_lt100_frac_at_5B": 0.0431211090558307},
  "qwen25": {"vocab_size": 151665, "sample_tokens": 198710493, "n_used": 92735,
             "frac_used": 0.6114462796294465, "n_never_seen": 58930,
             "entropy_bits_per_token": 11.04325361647404, "chars_per_token": 4.623155290546232,
             "bits_per_char": 2.3886832525518007, "n_count_lt10": 85054, "n_count_lt100": 104182,
             "top1k_share": 0.6488516638122377, "top10k_share": 0.8904841074497258,
             "dead_lt100_at_2B": 85054, "dead_lt100_frac_at_2B": 0.5608017670523852,
             "dead_zero_at_2B": 58930, "dead_lt100_at_5B": 76662,
             "dead_lt100_frac_at_5B": 0.5054692908713283}}
}
```

LFM2-350M `config.json`, fetched and parsed on FarmShare (evidence for the padding claim):

```json
{"vocab_size": 65536, "hidden_size": 1024, "block_dim": 1024,
 "num_hidden_layers": 16, "conv_L_cache": 3, "conv_bias": false}
```

i.e. **`config.vocab_size = 65536` but `tokenizer` has 64,400 entries → 1,136 unreachable
padding rows.** Also worth noting for the LIV design: LFM2-350M's own short conv is
`conv_L_cache = 3`, i.e. **k = 3**, with `conv_bias = false`, and it has 16 layers at
`d_model = 1024` — matching the study's L0 shape.

---

## Reproducibility

All on FarmShare `rice-03`, login-node CPU only, no GPU, no Slurm, ~5 MB disk footprint:

| path | what |
|---|---|
| `/scratch/users/ericrcwu/liv/tok/fertility.py` | Steps 1–4 small-sample. `MAX_DOCS=7700 python fertility.py` → `results.json` (~32 s) |
| `/scratch/users/ericrcwu/liv/tok/digits_ctx.py` | Digit context-invariance + phone formats → `digits_ctx.json` (~20 s) |
| `/scratch/users/ericrcwu/liv/tok/vocab_util.py` | 200M-token streaming vocab counts → `vocab_util.json`, `counts_final.npz` (~10 min) |
| `/scratch/users/ericrcwu/liv/tok/{gpt2,lfm2,olmo2,neox,qwen25}.tokenizer.json` | the five tokenizers, 4.7 MB total |
| `/scratch/users/ericrcwu/liv/tok/{full.log,vu.log}` | run logs |

Python `/scratch/users/ericrcwu/kda/venv/bin/python`, tokenizers 0.22.2, transformers 5.14.1.
No GPU used, no files under `/scratch/users/ericrcwu/agent-runs/` touched, nothing deleted.
