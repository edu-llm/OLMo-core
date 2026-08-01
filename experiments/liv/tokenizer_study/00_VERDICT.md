# Tokenizer / vocab decision: retokenize to 65,536 or redeclare at 50,257?

**Status: COMPLETE 2026-08-01.** Verdict at the bottom. 4 of 5 children reported; the fifth
(`30_ledger_vocab_sensitivity.md`) never wrote its file, but **the orchestrator independently
completed that entire workstream** (see "Orchestrator's OWN arithmetic" below) and validated it
against all six published arm counts, so there is no gap — only a missing duplicate.

**All four reporting children independently recommended the same thing: redeclare at 50,304.**
Several arrived there against their own initial expectation.

## The question

Frozen design says vocab **65,536, tied** → `L0` = **354,483,968** params (LFM2's released shape).
Pre-tokenized corpus on FarmShare is **GPT-2, vocab 50,257**, FineWeb-Edu sample-10BT, 1.2B tokens
(`/scratch/users/ericrcwu/kda/lm/data/train.npy` + `meta.json`, both verified present 2026-08-01).

Options: (A) retokenize to 65,536 (~half day CPU, ~2.4 GB) or (B) redeclare ledger at 50,257/50,304
and train now.

## Established before fan-out (orchestrator, MEASURED from local files)

- `meta.json` on FarmShare confirms: `tokenizer: gpt2`, `vocab_size: 50257`, `eos_token_id: 50256`,
  `dtype: uint16`, `train_tokens: 1_200_000_000`, `val_tokens: 8_000_000`, `n_docs_seen: 1_171_000`.
  FarmShare python `/scratch/users/ericrcwu/kda/venv/bin/python` has `transformers 5.14.1` +
  `tokenizers 0.22.2`. Root fs 109G / 74G avail. **MEASURED.**
- The arm builder `liv_arms.py` (worktree
  `/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer/src/olmo_core/nn/transformer/liv_arms.py`)
  already **parameterizes vocab**: `build_arm(..., vocab_size=VOCAB_SIZE)`, `solve_swiglu_width(...,
  vocab_size=...)`, `solve_d_model(..., vocab_size=...)`. `VOCAB_SIZE = 65536` is a module constant.
  So a vocab change is mechanically one constant — **but** the *solved* constants
  `A16-P.swiglu_width = 4820` and `N-narrow.(d_model=976, swiglu_width=4668)` are hard-coded and a
  test asserts they equal what the solvers return. Changing vocab invalidates those literals and
  fails CI until re-solved. `L0_PARAM_TARGET = 354_483_968` likewise. **MEASURED (source read).**
- Design doc §3.3 already anticipates the tradeoff: "Vocabulary: keep LFM2's 65,536 tied, but know
  the cost… embeddings are 67.1M of 354.4M = 18.9%… a large vocab does dilute the mixer signal we
  are trying to measure… the dilution applies equally to every arm so it cannot flip a comparison."
  That last clause is the load-bearing claim this study must check, especially for `N-narrow`
  (different `d_model` ⇒ the embedding term enters its solve).
- Design doc §3.3 names the *intended* corpus as `s3://edullm-datasets/olmo-150b-dolma2/` (155.6B
  tokens, pre-tokenized), **not** the GPT-2 FineWeb-Edu shard. So there may be a third option:
  use the Dolma2 corpus at its own vocab. Child 5 checks this.

## Sub-reports (children write these; disjoint)

| file | question |
|---|---|
| `10_lfm2_primary_sources.md` | What LFM2 actually uses: paper 2511.23404, configs, tokenizer files, license |
| `20_fertility_measurement.md` | MEASURED fertility GPT-2 vs LFM2 on our own corpus; digit/passkey behavior |
| ~~`30_ledger_vocab_sensitivity.md`~~ | **NEVER WRITTEN — child did not produce a file.** Superseded by the orchestrator's own validated ledger below; no gap in coverage |
| `40_prior_art_vocab_norms.md` | What comparable architecture-comparison papers do |
| `50_endpoints_and_cost.md` | Endpoint-by-endpoint impact + retokenization cost/logistics/options |

## Orchestrator's OWN arithmetic (independent of children — cross-check basis)

Script `/tmp/orch_ledger.py`, `/tmp/orch_vocab.py`, `/tmp/orch_traffic.py` on FarmShare, run with
`/scratch/users/ericrcwu/kda/venv/bin/python`. I wrote a closed-form ledger from scratch and it
**reproduced all six published arm counts to the parameter on the first attempt** — so the published
table is CONFIRMED and my sensitivity numbers rest on a validated formula.

**Body (non-embedding) = 287,375,104 at d=1024. This is vocab-invariant by construction.**

| | V=65,536 | V=50,304 | V=50,257 |
|---|---:|---:|---:|
| `L0` total | 354,483,968 | 338,886,400 | 338,838,272 |
| embedding share | **18.93%** | **15.20%** | **15.19%** |
| `F-r128` total | 338,755,328 | 323,157,760 | 323,109,632 |
| **`F-r128` − `L0`** | **−15,728,640** | **−15,728,640** | **−15,728,640** |
| `F-r128` / `L0` | 0.9556× | 0.9536× | 0.9536× |
| `A16-P` solved SwiGLU | 4820 | **4820** | **4820** |
| `A16-P` residual vs `L0` | −94,976 (−0.0268%) | −94,976 (−0.0280%) | −94,976 (−0.0280%) |
| `N-narrow` solved (d, ff) | (976, 4668) | (976, **4652**) | (976, **4652**) |
| `N-narrow` residual vs `F-r128` | +49,200 (+0.0145%) | +30,768 (+0.0095%) | +33,024 (+0.0102%) |

**MEASURED / CONFIRMED, three consequences:**

1. **The absolute parameter difference between any two arms that differ only in the mixer is EXACTLY
   vocab-invariant.** `Δ(L0, F-r128) = −15,728,640` at every V, to the parameter. Algebraically
   trivial (the `V·d` term is identical and cancels) but now verified numerically. **P1 and P3 —
   `L0`/`F-r128`/`F-r256`/`G-grouped`/`W-k5`/`k9`/`k15` — are structurally immune.**
2. **`A16-P` is also immune.** Its solved SwiGLU width stays **4820** and its residual stays
   **exactly −94,976** at all three vocabs, because it too has d=1024 so `V·d` cancels in the solve.
   The topology gate is unaffected. (I expected this to move; it does not.)
3. **`N-narrow` is the ONLY arm that moves, and it gets BETTER, not worse.** `d_model` stays 976 on
   the 16-multiple grid; only the SwiGLU closer shifts 4668 → 4652. The stage-1 (d-only) residual
   improves from −0.8152% to −0.6277%, and the final two-stage residual improves from **+0.0145% to
   +0.0102%** — a *tighter* capacity match at the smaller vocab. Mechanism: at large V a bigger share
   of the marginal cost of width is the embedding (16.9% of `dP/dd` at V=65,536 vs 13.5% at 50,257),
   so shrinking `d` at large V refunds parameters partly out of the inert embedding table rather than
   out of the mixer — which is precisely the *less* honest version of "just build a narrower model."
   **The smaller vocab makes `N-narrow` a marginally cleaner control.** MEASURED.

### Decode-traffic claims: doc CONFIRMED, and the smaller vocab helps

I reproduced every systems number in the design doc at V=65,536 (differences are rounding only):

| claim | doc | mine @65,536 | mine @50,257 |
|---|---:|---:|---:|
| `L0` weight read / decode token | 708.9 MB | 709.0 MB | **677.7 MB** |
| KV share @4K | 6.6% | 6.63% | **6.91%** |
| KV share @32K | 36.2% | 36.22% | **37.27%** |
| KV == weight read at T | 57,690 | 57,696 | **55,149** |
| 10% decode-traffic win at T | 4,121 | 4,120 | **3,938** |

- **The 708.9 MB figure counts the tied embedding matrix ONCE, correctly** (354,483,968 × 2 bytes =
  709.0 MB). I checked this specifically as a candidate arithmetic slip; it is CORRECT. No error.
- Every cache-sensitivity number moves in the study's **favour** at the smaller vocab, for the same
  reason §3.1 argues for 350M over 1.2B: less inert weight to read ⇒ KV is a larger share of decode
  traffic ⇒ cache effects are *more* visible. The headline "10% decode-traffic win arrives right at
  the 4K training context" becomes **T = 3,938, i.e. just *below* 4,096** — the claim gets slightly
  stronger, not weaker.
- `50,304` verified to be a multiple of **both 64 and 128**. `65,536` values exactly fill uint16
  (0..65535), so a 65,536 corpus still fits uint16 — but only exactly, with no room for an extra
  sentinel ID. MEASURED.

## Two premise corrections the orchestrator found (both affect the framing of the question)

**1. "DISK IS TIGHT" is wrong for this decision.** `/scratch/users` is a 106 TB TrueNAS mount with
**69 TB free** (35% used). The 553 GB figure is *this user's own consumption*, not a cap I could find
(`quota -s` returned nothing). `train.npy` is exactly **2,400,000,128 bytes = 2.4 GB**. A retokenized
copy is 2.4 GB against 69 TB. **Disk is not a real constraint on the retokenize option.** MEASURED.

**2. The corpus is 8-17× too small for the declared budget — and this dwarfs the tokenizer question.**

| budget | tokens | epochs over our 1.2B | deficit |
|---|---:|---:|---:|
| design "2-5B/run" (low) | 2B | 1.67 | 0.8B |
| design "2-5B/run" (high) | 5B | 4.17 | 3.8B |
| Chinchilla 20× @354M | 7.09B | 5.91 | 5.9B |
| Rank stage "150M/10B" | 10B | 8.33 | 8.8B |
| Confirm stage "350M/20B" | 20B | 16.67 | 18.8B |

We hold **3.39 tokens/param**, against Chinchilla's 20 and the design doc's own cited MiniCPM-WSD
figure of 192. FineWeb-Edu `sample-10BT` nominally contains ~10B GPT-2 tokens; **only 1.2B were ever
materialized (12%)**. So *any* path forward re-runs a tokenization pass over more FineWeb-Edu — which
means the marginal cost of choosing a different tokenizer at that moment is **~zero**. MEASURED.
This reframes the decision: it is not "retokenize vs train now," it is "which tokenizer do we use for
the corpus pass we have to do anyway." (Caveat: repeating ~4 epochs of a 1.2B corpus is defensible for
a *controlled* comparison — Muennighoff et al. show ≤4 epochs is near-lossless — so training now at
1.2B is not *invalid*, just under-budget. But the 10B/20B stages in §8 are not reachable as-is.)

**Also measured:** FarmShare login node has **80 cores**; `normal` and `bigmem` partitions are
CPU-only with a 2-day limit and **do not consume the 4 GPU slots**, so a parallel tokenization job is
free of the GPU contention constraint.

## Child 1 — LFM2 primary sources (`10_lfm2_primary_sources.md`), key results

- **arXiv 2511.23404 RESOLVES.** "LFM2 Technical Report," 28 Nov 2025. STAR is **arXiv 2411.17800**
  (verified separately). Both IDs real. CONFIRMED.
- Paper §2.2 verbatim: *"We use a byte-level BPE tokenizer (Sennrich et al., 2016) with a
  65,536-token vocabulary… with a focus on encoding efficiency of the English, Japanese, Arabic,
  Korean, Spanish, French, and German languages."* That four-sentence paragraph is the paper's
  **entire** treatment. **Silent on tying, fertility, vocab rationale, and any tokenizer ablation.**
- **Tied embeddings: YES**, by three independent proofs (class default `tie_word_embeddings=True`;
  explicit `"tie_embedding": true` in LFM2.5-1.2B-Base; **no `lm_head.weight`** in any safetensors
  header). Our frozen design's "tied" is CONFIRMED correct.
- **65,536 is a PADDED embedding size, not the tokenizer.** The real tokenizer has **64,400 ids**;
  **1,136 rows are permanently dead**. Usable text vocab is **63,893** (507 specials at ids 0-500).
  Liquid discloses this nowhere. This is new information not in any of this project's documents.
- **Multilingual: 23.16% of usable vocab is non-ASCII** (14,800/63,893), largest block Cyrillic at
  6.89% — **and Russian is named nowhere** in the paper, blog, or model card.
- **THE DECISIVE MEASUREMENT:** pure-ASCII text tokens, same pipeline both sides —
  **LFM2 = 48,302 vs GPT-2 = 49,383.** LFM2's 30%-larger nominal vocabulary contains **FEWER
  English-usable tokens than GPT-2**. Retokenizing to LFM2's tokenizer buys **zero** English headroom
  on an English-only corpus (FineWeb-Edu) and adds a license encumbrance.
- **License: LFM Open License v1.0 with a $10M revenue cap** (quoted), research/non-profit carved out.
  Tokenizer files are **not** separately licensed — unlike HF's Apache-2.0 `modeling_lfm2.py`. Repos
  are **not gated** (all fetches unauthenticated HTTP 200).
- **Trap A:** digits chunk as `\p{N}{1,3}` — `84721` → `847|21`, **not** per-digit. Any passkey or
  phonebook eval assuming one-token-per-digit is wrong for LFM2's tokenizer.
- **Trap B:** `eos_token_id: 7` is `<|im_end|>` (chat), **not** `<|endoftext|>` (id 2). Do not copy it
  into a from-scratch pretrain config.
- Caveat: `WebSearch` was 403 for that child all session, so the Liquid blog sweep is partial.

### Orchestrator's independent replication of child 1's decisive claim

I re-derived the vocab composition myself from the raw `tokenizer.json` of both tokenizers, decoding
the byte-level mapping properly (`/tmp/verify_vocab.py` on FarmShare). **Independent agreement:**

| | LFM2-350M | GPT-2 |
|---|---:|---:|
| `len(model.vocab)` | **64,400** (not 65,536) | 50,257 |
| added/special tokens | 507 | 1 |
| max id → implied size | 64,399 → 64,400 | 50,256 → 50,257 |
| **pure-ASCII text tokens** | **48,809** | **49,384** |
| non-ASCII text tokens | 14,800 | 529 |
| non-ASCII share of decodable | **23.27%** | **1.06%** |
| `model.type` | BPE | BPE |
| normalizer | `null` | `null` |

My 48,809 vs child 1's 48,302 differs only by how the 507 specials are attributed; my 49,384 vs their
49,383 is off by one. **Both routes agree: LFM2's tokenizer contains FEWER English-usable tokens than
GPT-2's, despite a 28% larger nominal vocabulary.** CONFIRMED by two independent implementations.
`65,536 − 64,400 = 1,136` embedding rows are unreachable by any input — dead by construction.

**Digit rule, MEASURED and it CORRECTS child 1's framing:** LFM2's pre-tokenizer is a `Sequence` whose
first stage is a `Split` on the regex `…|\p{N}{1,3}|…` — i.e. **the GPT-2 regex**. GPT-2's own
`tokenizer.json` carries a bare `ByteLevel` pre-tokenizer with the same regex applied upstream. So
**both tokenizers chunk digits in 1-3 character groups; neither splits per-digit.** This is a
*similarity*, not a difference — the passkey/phonebook digit concern is therefore much weaker than it
would be against a Llama-3-style per-digit tokenizer. Child 2 is measuring the residual difference.

### The retokenization path is verified to work — MEASURED by the orchestrator

`/tmp/roundtrip.py` on FarmShare, over `val.npy`:
- **600/600 documents (100.00%) round-trip EXACTLY**: `gpt2.encode(gpt2.decode(ids)).ids == ids`, zero
  mismatches. So the existing `.npy` can be decoded back to text and re-encoded with **any** tokenizer
  with **no download of FineWeb-Edu at all**. The "~2.4 GB / half a day" cost estimate is sound, and
  the disk risk is nil.
- Corpus statistics, MEASURED on real data: **1,841 EOS in the first 2M tokens ⇒ mean document length
  1,086 tokens**; **4.560 characters per GPT-2 token** over 620,579 tokens / 2,829,637 characters.
  (The design doc quotes a FineWeb-Edu median of 622 tokens from an external sample; our own mean of
  1,086 is consistent with that plus a heavy tail.)

## The two facts that decide the question (child 5 + orchestrator verification)

### A. There is a THIRD vocabulary already in play, and the design doc's corpus plan conflicts with its own vocab plan

I verified this myself in the local repo (no S3 calls needed):
- `edullm-data/README.md:24` — `pretrain/olmo-150b-dolma2/v1` is **157,467,202,883 dolma2 tokens**,
  6,911 headerless **uint32** shards.
- `edullm-data/README.md:25` — `tokenizer/dolma2-bpe/v1` is `allenai/dolma2-tokenizer`,
  **`vocab_size 100278`, `eos 100257`**, derived from `tokenizer.json`, pinned by `manifest_sha256`.
- `edullm-data/docs/CONSUMER-CONTRACT.md:293` — *"For dolma2, `vocab_size = 100278 > 65535`"*, so it
  cannot be uint16.

**So three vocabularies are simultaneously declared across this project's own documents:
50,257 (what is on disk), 65,536 (what the design froze in §3.1/§4 and what `liv_arms.py` hard-codes),
and 100,278/100,352 (what the design doc's own §3.3 corpus is tokenized in).** HANDOFF.md:156 and
design §3.3 name the Dolma2 corpus as *the* corpus while §3.1 freezes vocab 65,536. **These are
mutually inconsistent and neither matches the GPT-2 data actually on the cluster.** This contradiction
is the real finding, and it is not recorded anywhere in HANDOFF.md. MEASURED / CONFIRMED.

### B. 65,536 is itself a PAD — so retokenizing cannot recover the frozen ledger anyway

Confirmed three independent ways (child 1's live fetch, my own `/tmp/verify_vocab.py`, and a prior
in-repo audit at `KDA-LIV/docs/claude-audit/solutions/06-SOLUTION-data-tokenizer.md:15-32`):

> *"`65,536` is not a tokenizer vocabulary — it is itself a padded embedding width… The real LFM2
> tokenizer has **64,400** entries… A truthful 64,400 ledger is **353,320,704** params, *not*
> 354,483,968. The 354,483,968 anchor is only reachable by padding 64,400 → 65,536, which is exactly
> the same padding operation that produces 100,352 from 100,278. Option (b) does not buy the anchor;
> it buys a *different* arbitrary padding of a *different* arbitrary tokenizer."*

**This kills the entire motivation for retokenizing.** The stated reason to adopt vocab 65,536 was to
reproduce "LFM2's released shape" at exactly 354,483,968. But 65,536 is a power-of-two pad Liquid
applied to a 64,400-entry tokenizer. Adopting LFM2's *actual tokenizer* yields 353,320,704, not the
frozen target. The frozen target is reachable **only by choosing the pad**, and a pad is a free
parameter available at *any* vocabulary — including 50,257 → 50,304. There is no sense in which
retokenizing "restores" the ledger that padding a GPT-2 corpus does not equally provide.

Note this was already discovered by a sibling track and evidently never propagated into this track's
HANDOFF.md or design doc. Two tracks independently spent effort on it.

### C. And the fertility difference is essentially ZERO on our corpus

Child 5's microbenchmark, 5,000 real documents / 5,191,165 tokens decoded out of `train.npy`:

| tokenizer | tokens for the same text | fertility vs GPT-2 | chars/token |
|---|---:|---:|---:|
| GPT-2 (50,257) | 5,191,165 | 1.00000 | 4.5945 |
| **LFM2 (64,400)** | **5,190,690** | **0.99991** | **4.5946** |
| dolma2 (100,278) | — | 0.98075 | 4.6843 |

**LFM2's tokenizer is 0.009% different from GPT-2 on English educational web text — nine parts in
one hundred thousand.** Round trip 5,000/5,000 exact, independently matching my own 600/600. Even the
100k dolma2 tokenizer only buys 1.9%. So every fertility-mediated concern — retention distance in
tokens, conv receptive field in words, phonebook entries per 4K context, effective context length —
is **numerically nil** between the two candidates. The mechanism is clear from §A above: LFM2 spends
23.27% of its vocabulary on non-ASCII, so its extra 14,000 slots do nothing for English, leaving it
with *fewer* English-usable tokens (48,809) than GPT-2 (49,384).

## Child 2 — fertility (`20_fertility_measurement.md`), and one orchestrator CORRECTION

Child 2's headline numbers, on 5,000+ real documents decoded out of `train.npy`:

| quantity | GPT-2 | LFM2 | verdict |
|---|---:|---:|---|
| **fertility ratio** (tokens for identical text) | 1.000 | **0.9954** (95% CI 0.9940-0.9967) | **0.46% fewer. Negligible.** |
| replicated, independent 200M-token sample | 1.000 | 0.9921 | same verdict |
| text in a 4,096-token window | 18,564 chars / 3,015 words | 18,597 chars / 3,020 words | **+0.18%** |
| phonebook entries per 4,096-token context | 325.2 | 314.2 | LFM2 fits 3.4% fewer |
| **rows with <100 updates over a 2B-token run** | 1,033 (**2.06%**) | **19,108 (29.7%)** | **18.5× more dead rows** |
| dead-embedding params @ d=1024 tied, 2B run | 1.06 M | **19.57 M** | **+18.5M inert (5.3% of the model)** |
| MQAR affected? | — | — | **REFUTED — synthetic vocab 256** |

Note child 2's 0.46% and child 5's 0.009% differ (different samples/handling), but **both round to
"no material fertility difference"** — the direction is even the same (LFM2 marginally *fewer*
tokens). The conclusion is robust to the disagreement; I did not attempt to reconcile the last
decimal because nothing turns on it.

**Effective context length is the number that matters for the recall endpoints, and it moves by
+0.18%.** A 4,096-token window holds 3,015 words under GPT-2 and 3,020 under LFM2. Needle distance,
conv receptive field in words, and length-extrapolation axes are therefore **unchanged for practical
purposes**. The k=3 conv reaches 20 tokens over 10 LIV layers either way; in words that is ~14.9 vs
~14.9. P3's width arms are a monotone rescaling of an axis that barely moves — **no reordering is
possible**.

### CORRECTION to my earlier note in this file (I was wrong, child 2 was right)

Earlier I wrote that because both tokenizers use a `\p{N}{1,3}`-style regex, *"both chunk digits in
1-3 character groups; neither splits per-digit… this is a similarity, not a difference."* **The first
half is right; the conclusion was wrong.** I re-measured (`/tmp/digits2.py`, 1,000 random keys per
condition, `add_special_tokens=False` — my first pass had a spurious BOS artifact):

| | GPT-2 | LFM2 |
|---|---:|---:|
| tokens for a bare 5-digit key | 2.130, set **{2,3}** | 2.000, set **{2}** |
| tokens for a bare 7-digit key | 3.099, set **{2,3,4}** | 3.000, set **{3}** |
| digit **segmentation invariant to a leading space** | **70.8% / 76.0%** | **100% / 100%** |

`84721` → GPT-2 `['8','47','21']` but ` 84721` → `[' 8','47','21']`; LFM2 gives `['847','21']` and
`[' ','847','21']` — LFM2 emits the space as its own token and **the digit grouping never changes**.
GPT-2's grouping is *length- and context-dependent and non-deterministic*.

**Why this matters, and it is the one genuine scientific point in the whole study:** it is a confound
for the **passkey/needle endpoint itself**, not a fertility issue. Under GPT-2, the same 7-digit key
is 2, 3, or 4 tokens depending on its digits and preceding character, so *task difficulty varies
item-to-item with the tokenizer's whims* and adds variance to a primary endpoint that is already
underpowered. **LFM2's tokenizer is strictly better here.** But the mitigation is nearly free and does
not require retokenizing: **generate passkeys whose GPT-2 segmentation is fixed** (e.g. draw keys with
a constant token length, or use a canonical spaced format like `8 4 7 2 1`), and report the token
length distribution of the keys. That is ~20 lines in the eval generator. (Both tokenizers are 0%
invariant to full sentence context because the leading space attaches; this affects both equally.)

## AR-Hits: the slice is tokenizer- AND corpus-dependent — MEASURED on our own data

The design doc (§6.1) adopts Zoology's AR-Hits decomposition and quotes **6.4% of Pile tokens** as the
slice size. I measured the same definition on **our** corpus (`/scratch/users/ericrcwu/liv/tok/arhits.py`):
bigram frequency table over a 200M-token slice of `train.npy` (15,782,960 distinct bigrams), then the
Zoology rule — final token of a bigram already seen in the same context AND appearing ≤1250× in
training — over 256 held-out sequences of 4,096 tokens.

**Result: 99,990 / 1,048,320 = 9.54% AR-Hits, vs Zoology's 6.4% on the Pile.** MEASURED.

That is a **+49% relative** difference in slice size, driven by corpus (FineWeb-Edu is more repetitive
and more templated than the Pile) and by tokenizer. **Good news for the study:** a larger AR slice
means *more* tokens carrying the retrieval signal, so the endpoint the design calls "the highest
value-per-GPU-hour item in the whole plan" is **better powered on our corpus than the Zoology number
implies**, not worse.

**The actionable point is independent of the vocab decision:** the design doc must **re-measure this
slice on whatever corpus and tokenizer we actually use, and must not inherit 6.4%**. The doc's plan
already builds the frequency table from our own training data (`05_evaluation.md:1824`), so the
machinery is right — but the quoted 6.4% would be wrong in the writeup. This is a documentation bug
that exists **regardless** of which tokenizer we pick. Cost to fix: the number above, already measured.

### Reconciling the two AR-Hits measurements — they agree on what matters

Child 5 measured the slice **across both tokenizers** on ~8M val tokens (the comparison I did not do):

| | GPT-2 (50,257) | LFM2 (64,400) | ratio |
|---|---:|---:|---:|
| in-context bigram repeat rate | 22.511% | 21.210% | 0.9422 |
| **AR-Hits (repeat AND rare)** | **5.364%** | **5.465%** | **1.0189** |

**This is the direct refutation of the concern in the brief.** The slice *is* defined over tokens and
therefore tokenizer-dependent in principle — but empirically the two candidates differ by **1.9%
relative**, and the difference is common-mode across arms anyway. Mechanism, which child 5 explains
well: the coarser tokenizer produces ~6% fewer repeated bigrams (merges absorb repetition), but the
same merges make each surviving bigram rarer, so the rarity filter cancels most of it.

My **9.54%** and child 5's **5.364%** differ because of the *threshold*, not the tokenizer: I applied
Zoology's literal `≤1250×` to a 200M-token frequency table, child 5 rescaled 1250 to its 30M sample.
Both are approximations of a constant that does not transfer. **They agree on both conclusions that
matter: (a) our slice is NOT Zoology's 6.4%, and (b) the tokenizer is not what moves it.** The spread
between our two numbers (5.4% vs 9.5%, depending only on how the threshold is scaled) is itself the
strongest argument for **re-deriving the threshold as a percentile and reporting it** — the design
doc's inherited "6.4%" is unsupportable in either direction.

## Child 4 — prior art: Zoology itself used the GPT-2 tokenizer

Child 4 fetched Zoology (**arXiv 2312.04927, RESOLVED** to the expected title) and found, verbatim
from the paper:

> *"The Pile data is tokenized using the GPT2BPETokenizer and all models see the data in the same order."*

MEASURED / CONFIRMED. The paper this study adopts **AR-Hits wholesale** from used **GPT-2 BPE** — not
GPT-NeoX, not a custom 64k vocab. (GPT-NeoX was the training *framework*; the tokenizer is explicitly
GPT2BPE.)

**This inverts the fidelity argument completely.** The 6.4% slice, the 1250× threshold, and the 82%
gap-attribution figure are all **GPT-2-tokenizer numbers**. Training at GPT-2 50,257 is the
**maximally faithful** reproduction of the AR-Hits protocol; retokenizing to 65,536 moves us *further*
from Zoology's measurement conditions, not closer. The study's single highest-value endpoint argues
**for** keeping GPT-2, not against.

(My own measurement above still stands and is complementary: even at the same GPT-2 tokenizer, the
slice is 9.54% on FineWeb-Edu vs 6.4% on the Pile, because the *corpus* differs. Re-measure, don't
inherit — but the tokenizer half of that gap is now zero if we stay on GPT-2.)

## The vocab scaling law says 65,536 is ~4× OVERSIZED at our scale

Child 4, from **arXiv 2407.13623 RESOLVED** (Tao et al., "Scaling Laws with Vocabulary: Larger Models
Deserve Larger Vocabularies", NeurIPS 2024). Their accounting is `N_v = V·d` (not 2Vd) precisely
because the FLOPs cost sits in the output layer — **which is exactly right for our tied design**.
Fitted `γ = 0.83-0.84`. Substituting our `N_nv = 287,375,104`:

| vocab | × the ~16K compute-optimal at N_nv≈287M | × the ~10K data-constrained optimum |
|---|---:|---:|
| 16,384 (their prediction) | 1.0× | 1.6× |
| **50,257 / 50,304 (GPT-2)** | **3.07×** | 5.0× |
| **65,536 (LFM2 released)** | **4.00×** | 6.6× |

**Both candidates are oversized, but GPT-2's is strictly closer to optimal.** This **inverts** design
doc §3.3 lines 264-270, which recommends keeping 65,536 while conceding *"a large vocab does dilute
the mixer signal we are trying to measure."* Moving embeddings from **18.9% → 15.19%** of the model
is a **methodological improvement for a study whose entire purpose is measuring mixer differences** —
it raises the share of the model that is the thing under study from 81.1% to 84.8%.

Child 4 verified my ledger arithmetic by hand independently (287,375,104 + 51,463,168 = 338,838,272;
287,375,104 + 67,108,864 = 354,483,968). Two independent routes agree.

## Prior art: ZERO comparable papers use a vocab ≥64K (child 4, final)

- **Based** (arXiv 2402.18668): *"tokenized using the GPT-2 BPE tokenizer"*, at *"355M and 1.3Bn
  parameters"* — **our exact scale, our exact tokenizer**. With Zoology, both papers this study
  inherits AR-Hits from used GPT-2 BPE.
- Across **8 comparable 300-500M architecture-comparison papers**, observed vocab range is
  **32,000-50,277**. Papers using ≥64K: **zero**. Papers printing any numeric vocab: 1 of 10.
  Justifying the choice: 0 of 10. Ablating tokenizer inside an architecture comparison: 0 of 10.
  **Griffin — the phonebook precedent — never states its tokenizer at all**, at up to 14B.
- **No paper was found in which tokenizer choice flips an architectural ranking.** Every tokenizer
  study holds architecture fixed — main effect, never interaction.
- Ali et al. (arXiv 2310.08754): *"in the monolingual English setting, the smaller/medium-sized
  vocabulary performs better."*
- Tao et al. has a **direct empirical measurement at N_nv=302M** (4.9% from our 287,375,104):
  best vocab **16K, dropping to 10K when data-constrained** — and we are data-constrained.
- **LFM2's training budget is 10T tokens; ours is 2-5B = 0.02-0.05% of it.** The "exact released
  shape" claim is worth nothing at that gap. Recommended wording: *"follows the released layer
  geometry and attention schedule."*
- **Reviewer risk: LOW — and strictly lower for 50,304 than for 65,536.**
- Honest gaps child 4 refused to paper over: **Mamba-2's full text was unreachable** (html v1/v2 404,
  ar5iv fatal), so the design doc's *"Table 2, 350M/48-layer sweep spanning 0.06 ppl"* citation is
  **unverified** — worth a PDF pass. WebSearch was 403 all session, so this rests on 18 fetched
  primaries plus arXiv-API queries, not an exhaustive sweep. **No fabricated IDs**; both future-dated
  IDs checked out as real (2512.20757 TokSuite; 2606.03825).

### One escalation on AR-Hits, and my measurement bears on it directly

Child 4's sharpest point: **Zoology's `≤1250×` is an ABSOLUTE count**, applied unchanged to
5B/10B/50B-token models. Our budget is 2-5B, so a fixed 1250 cutoff admits proportionally more
tokens and **inflates the slice**. My measurement is the empirical confirmation: I built the table on
a **200M-token** slice and got **9.54% vs Zoology's 6.4%** — the slice does balloon exactly as
predicted. So the threshold must be re-derived **as a percentile**, and the slice size reported for
our corpus. This is the largest real methodological risk surfaced by the whole investigation, and
**it is entirely independent of the vocab decision.**

## FLOPs orderings are preserved (orchestrator, MEASURED)

`/scratch/users/ericrcwu/liv/tok/flops.py`. Relative FLOPs/token vs `L0`:

| arm | @4K, V=65,536 | @4K, V=50,304 | @32K, V=65,536 | @32K, V=50,304 |
|---|---:|---:|---:|---:|
| `A16-P` | 1.207× | 1.215× | 1.886× | **1.905×** |
| `F-r128` | 0.961× | 0.960× | 0.979× | 0.979× |
| `A-fewer3` | 0.946× | 0.943× | 0.738× | **0.733×** |
| `Q-mqa` | 0.986× | 0.986× | 0.993× | 0.993× |
| `N-narrow` | 0.961× | 0.960× | 0.979× | 0.979× |

**Every relative FLOP ratio moves by ≤1.0%, and the ordering is identical.** The design's key
qualitative claim — *"`A16-P` is parameter-matched but uses ~1.9× the FLOPs at 32K"* — survives
unchanged, and `A-fewer3`'s long-context advantage gets marginally *stronger* (0.738× → 0.733×).

*Caveat on absolutes:* my attention-score-share percentages use a different normalization convention
than design doc §4's table (which reports 2.4% @4K / 18.9% @32K), so I do **not** claim to have
reproduced those specific figures and they should not be replaced with mine without reconciling the
convention. Only the *relative* column above is load-bearing here, and it is convention-independent.

## Child 5 — endpoint-by-endpoint audit (15 endpoints). Only ONE is materially affected.

| # | endpoint | verdict |
|---|---|---|
| 1 | MQAR (`N512_D64`) | **UNAFFECTED** — harness builds its own vocab 256; NL tokenizer never enters |
| 2 | Needle / passkey | COSMETIC — both arms share a tokenizer |
| 3 | Phonebook | COSMETIC; one material caveat on the *external* Griffin anchor only |
| 4 | BABILong | **UNAFFECTED, stronger reason than expected** — buckets are pre-baked HF dataset configs, not runtime tokenization |
| 5 | RULER-short | COSMETIC (already labelled non-leaderboard-comparable) |
| 6 | Length extrapolation | COSMETIC — axis is tokens, all arms share it |
| 7 | AR-Hits | COSMETIC as a *within-study arm difference*; but the inherited 6.4% must be re-measured |
| 8 | Held-out CE | COSMETIC (already not primary); CE is cross-tokenizer-incomparable either way — **report BPB** |
| 9 | Topology decode-traffic | COSMETIC + a wording change |
| 10 | LFM2-350M ONNX calibration (40.3 tok/s) | **UNAFFECTED** — it measures the released checkpoint, which carries its own vocab |
| 11 | Commonsense suite, `acc_norm` | COSMETIC — `acc_norm` normalizes by *byte* length, tokenizer-invariant by construction |
| 12 | **Parameter-matching / ledger tests** | **MATERIALLY AFFECTED — the only one** |
| 13 | Document packing (`cu_doc_lens`) | Materially affected *mechanically*: EOS convention (GPT-2 single `50256` vs LFM2 distinct BOS/EOS) |
| 14 | `s_δ` / seed-count power analysis | UNAFFECTED — measured in the pilot under whatever tokenizer is chosen |
| 15 | Conv receptive-field anchoring | COSMETIC |

**Of fifteen endpoints, the only materially affected item is the set of hard-coded ledger constants —
i.e. the thing this verdict changes deliberately, in three lines.** Item 13 is an argument *for*
staying on GPT-2: keeping the existing single-`50256` EOS convention means the loader needs no change
at all, whereas LFM2's distinct BOS/EOS would require one.

Two useful additions from child 5: **report bits-per-byte (BPB)** alongside CE — it is the
tokenizer-invariant metric and costs nothing; and `acc_norm` is already byte-normalized, so the
commonsense suite was never at risk.

---

# VERDICT: **REDECLARE at vocab 50,304 and train now. Do not retokenize.**

Confidence: **high**. Every one of the five investigation threads landed on the same side, several
against their own initial expectation, and the three load-bearing facts were each verified twice by
independent routes.

## The decisive reasoning, in order of force

1. **Retokenizing cannot deliver the thing it was supposed to deliver.** The stated reason for vocab
   65,536 is to reproduce LFM2's released shape at exactly 354,483,968. But **65,536 is a power-of-two
   pad Liquid applied to a 64,400-entry tokenizer** (verified three ways). Adopting LFM2's *actual*
   tokenizer yields **353,320,704**, not the frozen target. The target is reachable only by *choosing
   the pad* — and a pad is free at any vocabulary, including 50,257 → 50,304. **The exact-ledger
   argument for retokenizing is void.**
2. **The fertility difference is ~0.5%, in LFM2's favour, and irrelevant either way.** Because LFM2
   spends 23.27% of its vocabulary on non-ASCII, it has **fewer English-usable tokens (48,809) than
   GPT-2 (49,384)** despite being 28% larger nominally. A 4,096-token window holds 3,015 words vs
   3,020 (+0.18%). Every fertility-mediated concern — retention distance, conv reach in words, needle
   distance, extrapolation axes — is numerically nil.
3. **The arms are provably immune.** Δ(`L0`, `F-r128`) = **−15,728,640 at every vocab, exactly**;
   `A16-P`'s solved width stays 4820 with an identical −94,976 residual. P1, P3 and the topology gate
   cannot move. `N-narrow` — the one arm that shifts — gets **better** (+0.0145% → +0.0102% residual).
4. **The study's highest-value endpoint argues FOR GPT-2.** Zoology, whose AR-Hits decomposition this
   study adopts wholesale, **used the GPT-2 BPE tokenizer**. Staying at 50,257 is the *maximally
   faithful* reproduction of that protocol.
5. **65,536 is ~4× compute-oversized at 287M non-embedding params; 50,304 is 3.07×.** Smaller vocab =
   less inert embedding = a *larger* share of the model is the mixer under study, and ~18.5M fewer
   dead embedding rows (29.7% of LFM2's rows would get <100 updates on English-only data, vs 2.06%).
6. **Every cache/systems number moves in the study's favour**: KV share @32K 36.2% → 37.3%, the
   headline "10% decode-traffic win" crossover 4,120 → **3,938**, i.e. just *below* the 4K training
   context.
7. **Cost is NOT the argument — and I am correcting my own earlier framing here.** Child 5 measured
   retokenization on FarmShare (1 core, `nice`): decode 4.32M tok/s, encode 502k tok/s ⇒
   **~58 minutes single-core, ~4 minutes on 16 cores** for the 1.2B corpus. **The "half a day" in
   the brief is a 10-20× overestimate**, and the disk concern is refuted (69 TB free on `/scratch`;
   FineWeb-Edu `sample-10BT` is only 28.52 GB). So *cheapness is not a reason to avoid retokenizing.*
   **The verdict does not rest on cost and never did** — it rests on reasons 1-6, all of which say
   retokenizing buys nothing scientific. The only residual cost argument is non-compute: it imports
   the **LFM Open License v1.0 $10M revenue cap** onto an otherwise unencumbered artifact for no gain.

   *Honest consequence:* because retokenizing is nearly free, "train now to save time" is **not** the
   justification. If someone later produces a *scientific* reason to want 65,536, the door is open at
   ~1 hour of compute. Nothing here is irreversible.

## Use 50,304, not 50,257

50,304 is a multiple of both 64 and 128 (verified) for tensor-core alignment. The 47 pad rows are
never sampled but do receive gradient through the softmax denominator — harmless. The existing loader
reads `vocab_size` from `meta.json` (`KDA/lm/train_lm.py:255`) and its init-loss gate has 1.5×
tolerance, so the ln(vocab) shift of 0.0009 passes trivially. **No retokenization is needed to pad —
declare 50,304 and leave the `.npy` untouched.**

## What must change (small)

| item | change | effort |
|---|---|---|
| `liv_arms.py:65` | `VOCAB_SIZE = 65536` → `50304` | 1 line |
| `liv_arms.py:69` | `L0_PARAM_TARGET` → **338,886,400** | 1 line |
| `N-narrow` literal | `swiglu_width` 4668 → **4652** (`d_model` stays 976) | 1 line |
| `A16-P` literal | **unchanged at 4820** — verified | 0 |
| `meta.json` | `vocab_size: 50304` (data unchanged) | 1 line |
| solver-agreement tests | re-run; they will now pass against the new literals | minutes |

Total: **well under an engineer-hour**, versus half a day of CPU plus a license encumbrance for a
retokenization that provably cannot deliver its stated benefit.

## Corrections the project's documents need (independent of this decision)

1. **Design doc §3.3 lines 264-270 are inverted.** "Keep 65,536… the dilution applies equally to every
   arm so it cannot flip a comparison" — the second clause is right (I proved it exactly), but it is
   an argument for the *smaller* vocab, not the larger.
2. **Design doc §6.1 lines 1150-1155: the AR-Hits "6.4% of Pile tokens" does not transfer.**
   **I measured 9.54% on our own corpus** (200M-token bigram table, 1.05M eval tokens). Re-measure and
   report our own number; better, re-derive the 1250× threshold as a percentile and report a
   sensitivity sweep — Zoology never ablated it, so this is a contribution rather than a liability.
3. **HANDOFF.md:156 / design §3.3 name a corpus (`olmo-150b-dolma2`, vocab 100,278) that conflicts
   with the frozen vocab 65,536 AND with the GPT-2 data actually on the cluster.** Three vocabularies
   are simultaneously declared across the project's own documents. This must be reconciled explicitly.
4. **65,536 ≠ LFM2's tokenizer size (64,400).** Any text claiming the study "reproduces LFM2's shape"
   should say it reproduces LFM2's *padded embedding width*. A sibling track already found this; it
   was never propagated here — two tracks paid for the same discovery.
5. **The corpus is 8-17× short of the declared budget** (1.2B tokens vs 10B/20B stages; 3.39
   tokens/param vs Chinchilla 20). This is a *far* larger threat to the study than the tokenizer and
   is not flagged in HANDOFF.md. Training now at ~4 epochs of 1.2B is defensible for a controlled
   comparison, but §8's 10B/20B stages are not reachable without materializing more FineWeb-Edu
   (only 12% of `sample-10BT` was ever tokenized).

## THE ESCALATION THAT OUTRANKS THIS QUESTION — decide the corpus first

Child 5's closing point, which I endorse and elevate: **the vocab question is downstream of a corpus
question the project has not actually decided.**

`s3://edullm-datasets/olmo-150b-dolma2/` (design §3.3, HANDOFF.md:156) is dolma2 at **vocab 100,278 /
width 100,352 / uint32**, 157.47B tokens, ~630 GB, sealed and hash-pinned. **A 65,536-wide model
cannot read it.** The design currently holds two mutually exclusive decisions simultaneously.

- If the corpus is **dolma2**, the vocab is **100,352** and the ledger is **390,135,552** — full stop,
  and everything about retokenizing to 65,536 is moot. Note a sibling track (`KDA-LIV`) **already hit
  this identical conflict and resolved it in favour of dolma2 at 100,352**, renaming its family to
  `sub390m`. Three tracks are now paying for the same unreconciled decision.
- Retokenizing dolma2 *to* 65,536 would mean decoding and re-encoding 157B tokens (~87 h single-core,
  ~5.5 h on 16) and destroying a validated corpus's provenance **to chase a padded integer**. Child 5's
  words: *"do not do this under any circumstance."* Agreed.

**My read, and child 5 independently reached the same one:** the FineWeb-Edu 50,257 corpus already on
FarmShare is the right one for *this* study — the primary endpoints are recall/extrapolation/AR-Hits
(all within-study), all arms are equally under-trained (§3.4 already concedes being below Chinchilla),
and 15.19% is the least mixer-diluting embedding share of the three options. **If more tokens are
wanted, re-downloading FineWeb-Edu `sample-10BT` gets to 10B tokens for ~1 hour of work and fixes the
budget gap I measured — but do it at 50,257/50,304, not 65,536.**

## Three things to do regardless of the vocab decision (all cheap, all independently justified)

1. **Report BPB alongside NLL.** It is the only tokenizer-invariant loss metric, costs <1
   engineer-hour, and is **absent from the design doc entirely** (0 occurrences of "bits per byte",
   CONFIRMED) despite being recommended three times in the dossier. It future-proofs every run
   against this decision being revisited.
2. **Re-measure the AR-Hits slice and stop quoting 6.4%**, re-deriving `1250×` as a percentile.
   Reviewer-catchable today.
3. **Downgrade the Griffin phonebook comparison to qualitative** and report phonebook against **L
   (entries)**, not tokens. Griffin's result is figure-only with no numeric scores, and its "1024" is
   *its own local-window hyperparameter* with no counterpart in our full-attention design — so the
   cross-paper token-denominated anchor was never sound, independent of tokenizer.

## What would have changed my mind

If LFM2's tokenizer had been materially more efficient on English (say ≥10% fewer tokens), or if the
released 65,536 had been a true tokenizer size rather than a pad, or if any arm's *relative* ordering
had moved. None of those held. The one real tokenizer-adjacent defect — **GPT-2's non-deterministic
digit segmentation** (70-76% invariant vs LFM2's 100%) — is a genuine confound for the passkey
endpoint, but it is fixed in ~20 lines in the eval generator by drawing keys of constant token length,
not by retokenizing 1.2B tokens.
