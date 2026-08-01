# 50 — Endpoint impact audit + cost/logistics/risk of the vocab decision

**Agent:** child 5 of 5 (endpoints + cost). **Started:** 2026-08-01.
**Owns only this file.** Siblings cover: LFM2 primary sources/licensing, fertility measurement,
parameter-ledger arithmetic. Not duplicated here.

Every finding tagged **MEASURED / INFERRED / ASSUMED** and **CONFIRMED / REFUTED / UNCLEAR**.

---

## BOTTOM LINE (written first, revised in place as evidence lands)

> ### The single decisive fact I found
>
> **The design doc's *intended* corpus is already tokenized at a THIRD vocabulary.**
> `s3://edullm-data/pretrain/olmo-150b-dolma2/v1` is **dolma2 BPE, true vocab 100,278,
> padded embedding width 100,352, dtype uint32**, 157.47B tokens, 6,911 shards.
> (`edullm-data/README.md:24-25`, `edullm-data/docs/CONSUMER-CONTRACT.md:293`.)
> So "retokenize to 65,536" does not reconcile with §3.3's corpus plan — it *conflicts* with it.
> There are not two options on the table, there are three vocabularies already in play:
> **50,257 (what we have on disk), 65,536 (what the design froze), 100,352 (what the corpus is).**
> **MEASURED / CONFIRMED** (from local repo files, no S3 calls).
>
> ### The second decisive fact (already established in-repo, apparently not propagated)
>
> **65,536 is not a tokenizer vocabulary — it is itself a pad.** A prior audit in this same repo
> fetched all three LFM2 `tokenizer.json` files live and found **64,400** entries, max id 64,399.
> `65,536 = 2^16` pad of 64,400 (1,136 dead rows). So **even retokenizing with LFM2's actual
> tokenizer does not produce the 354,483,968 ledger** unless you also deliberately replicate
> Liquid's padding constant. (`KDA-LIV/docs/claude-audit/solutions/06-SOLUTION-data-tokenizer.md:17-29,
> 50-75`.) **MEASURED (by the prior audit) / CONFIRMED.** A sibling agent is re-verifying; I inherit it.

> ### RECOMMENDATION, in one sentence
>
> **Redeclare at 50,304 (GPT-2 padded to a multiple of 128) and train now** — because the
> 65,536 target is a padding constant rather than a tokenizer property, so retokenizing cannot
> deliver what it was proposed to deliver, while **none of the 15 endpoints I audited changes
> its conclusion** and the design doc's own mixer-dilution argument (§3.3) points at the
> smaller vocabulary once the ledger justification is removed.

### Endpoint verdict table — one line each (detail in Part A)

| # | Endpoint | Verdict | One-line reason |
|---|---|---|---|
| 1 | MQAR `N512_D64` | **UNAFFECTED** | Synthetic vocab 256; zero tokenizer imports in the harness (CONFIRMED, file+line) |
| 2 | Needle / passkey | COSMETIC | Same haystack text; regression check only, 1 seed |
| 3 | Phonebook | COSMETIC | Real axis is **entries (L)**, not tokens. Griffin's "1024" is *its* local-window hyperparameter, figure-only |
| 4 | BABILong | **UNAFFECTED** | Buckets are **pre-baked HF dataset configs**; harness never tokenizes to a budget (CONFIRMED from source) |
| 5 | RULER-short | COSMETIC | Synthesized to a token budget with *our* tokenizer; already "not leaderboard-comparable" |
| 6 | Length extrapolation | COSMETIC | Token axis, shared by all arms, no external number quoted |
| 7 | AR-Hits sliced ppl | COSMETIC | **MEASURED 5.364% (GPT-2) vs 5.465% (LFM2)** on our corpus — 1.9% and common-mode |
| 8 | Held-out CE / ppl | COSMETIC | Already non-comparable externally for 2 other reasons; explicitly not primary |
| 9 | Decode-traffic claim (T≈4,121) | COSMETIC | **T → 3,939 at 50,257**; the win arrives *before* the 4K context, i.e. slightly stronger |
| 10 | LFM2 ONNX 40.3 tok/s | **UNAFFECTED** | Measures an unmodified released artifact; no arm is compared to it |
| 11 | Commonsense `acc_norm` | COSMETIC | Normalized by **bytes** — already tokenizer-invariant |
| 12 | Param ledger + arm-builder tests | **MATERIALLY AFFECTED** | The only real breakage: 1 constant, 2 solver re-runs, 1 test literal (~1 h) |
| 13 | Document isolation / EOS | MATERIALLY AFFECTED (mechanically) | EOS id changes ⇒ silent bad `doc_lens`. One assertion fixes it |
| 14 | `s_δ` / power analysis | UNAFFECTED | Measured in the pilot under whichever vocab is chosen |
| 15 | Conv receptive-field anchoring | COSMETIC | `ℓ·k` in tokens; text it spans shifts 0.009% |

**Endpoints that would CHANGE THEIR CONCLUSION: none.** Three effects are genuinely
non-common-mode (embedding dilution 18.93%→15.19%, `N-narrow`'s solve granularity, the
decode-traffic crossover) and **all three favour the smaller vocabulary.**

### Cost table (detail in Part B)

| Option | Compute wall-clock | Disk | Eng-hours | Delivers 354,483,968? | Reversible? |
|---|---|---|---|---|---|
| **1a** retokenize the 1.2B corpus in place | **~58 min @1 core, ~4 min @16** (MEASURED-extrapolated) | 2.4 GB / **69 TB free** | ~5.5 | Only by copying Liquid's pad | High |
| **1b** re-download FineWeb-Edu 10BT + retokenize | ~6 h @1 core, ~1 h @16, +30 min download | **28.5 GB** (MEASURED) / 69 TB free | ~6.5 | Same | High |
| **1c** retokenize dolma2 (157B) | ~87 h @1 core, ~5.5 h @16 | 630 GB staging | high | Same | Destroys a sealed corpus — **reject** |
| **2** declare 50,257 / **50,304** | **0** | 0 | **~1** | No — honestly | **Very high** |
| **3b** train our own 65,536 BPE | 2-5 h @16 cores + bigmem | — | 8-16 | **Yes, genuinely** | Low |
| **5** use dolma2 @ 100,352 | **0** | 0 local | ~4 | No | High |

Key cost corrections: the claimed "~half a day of CPU" is a **10-20× overestimate** (measured
502k tok/s/core encode, 4.3M tok/s decode); the disk concern is **REFUTED** — the 74 GB figure
is the *root* fs, while `/scratch` has **69 TB free**; and **65,536 fits uint16 exactly with
zero headroom**, so it does not overflow, but one added special token would force uint32 and
double every file.

---

<!-- APPEND-ONLY BELOW; sections written in order as evidence lands -->

## 0. MEASURED: the retokenization microbenchmark (FarmShare login node, 1 core)

Script: `/scratch/users/ericrcwu/liv/tokbench.py` (written by me; FarmShare only, nothing
executed on the Mac). Raw JSON: `/scratch/users/ericrcwu/liv/tokbench_results.json`.
Run `nice -n 15`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`
— single core, polite. Login node has **80 cores**, loadavg ~1.4-8 at the time.
5,000 documents / **5,191,165 GPT-2 tokens** / 23,848,948 chars decoded out of the head of
the real `train.npy`. **I did not retokenize the corpus.**

| Quantity | MEASURED value |
|---|---:|
| `train.npy` | 1,200,000,000 tokens, **uint16**, 2,400,000,128 B |
| docs in slice | 5,000; mean 1,038.2 tok, median **590**, p90 2,035, max 26,697 |
| **GPT-2 decode** (ids → text) | **4,318,229 tok/s** = 4,159 docs/s, 1 core |
| **GPT-2 re-encode** (text → ids) | **501,766 tok/s** = 2.31 MB-chars/s, 1 core |
| **LFM2-350M encode** (vocab **64,400**) | **502,134 tok/s** = 2.31 MB-chars/s, 1 core |
| **dolma2 encode** (vocab **100,278**) | **407,316 tok/s** = 1.91 MB-chars/s, 1 core |
| **Round-trip losslessness (GPT-2 decode→encode)** | **5,000 / 5,000 documents EXACT**, 5,191,165 → 5,191,165 tokens |
| LFM2 fertility vs GPT-2 on this corpus | **0.99991** (5,190,690 vs 5,191,165 tokens) |
| dolma2 fertility vs GPT-2 on this corpus | 0.98075 |
| LFM2 chars/token | 4.5946 · dolma2 4.6843 · GPT-2 4.5945 |
| max observed id: LFM2 / dolma2 | 64,389 / 100,255 |

**MEASURED / CONFIRMED.** Three consequences the parent should have:

1. **The GPT-2 round trip is exactly lossless on this corpus** — 100% of 5,000 documents
   re-encode to the identical id sequence. Document boundaries survive because we split on
   EOS `50256` *before* decoding, so the boundary is carried in the loop, not in the text.
   (Sibling agent is verifying this independently; my sample agrees.) **MEASURED / CONFIRMED.**
2. **LFM2's tokenizer is fertility-neutral vs GPT-2 on FineWeb-Edu** — 0.99991, i.e. a
   **0.009% difference**, statistically indistinguishable. So the "a token means a different
   amount of text" worry, which is the mechanism behind almost every endpoint concern below,
   is **empirically ~zero for the 50,257 ↔ 64,400/65,536 comparison.** It is a real 1.96%
   for dolma2. **MEASURED / CONFIRMED.** (This is the sibling's assignment; I report my
   byproduct number so the parent can cross-check. Note it is measured on a 5.2M-token head
   slice, not the full corpus.)
3. **The claimed cost of "~half a day of CPU" is a ~10-20x overestimate.**

### Extrapolated wall-clock for a full retokenization of the 1.2B-token corpus

Corpus text = 1.2e9 GPT-2 tok × 4.5945 chars/tok = **5.513e9 chars ≈ 5.51 GB UTF-8.**

| Stage | 1 core | 8 cores | 16 cores | 32 cores |
|---|---:|---:|---:|---:|
| Decode GPT-2 ids → text | 4.6 min | 35 s | 17 s | 9 s |
| Re-encode with LFM2 (64,400) | **39.8 min** | 5.0 min | 2.5 min | 75 s |
| Re-encode with dolma2 (100,278) | 49.1 min | 6.1 min | 3.1 min | 92 s |
| **Total (LFM2), incl. ~30% overhead for I/O + writing 2.4 GB** | **~58 min** | **~7.5 min** | **~4 min** | **~2.5 min** |

**INFERRED (linear extrapolation from a MEASURED single-core rate) / CONFIRMED direction.**
Caveat: extrapolation assumes perfect scaling; `tokenizers`' Rust batch encoder parallelises
near-linearly with Rayon, so 8-16x is realistic, 32x optimistic. Even at 1 core it is **under
one hour**, not half a day. For the *full* FineWeb-Edu sample-10BT (~10B tokens) multiply by
8.3: ~5.5 h single-core, ~35 min on 16 cores.

**Output size:** 1.2B tokens × 2 B = **2.400 GB** at uint16 (the claimed 2.4 GB is exact),
or 4.800 GB at uint32. Val adds 16 MB / 32 MB.

### The uint16 dtype trap — stated precisely

`uint16` spans `[0, 65535]`; a vocab of size `V` uses ids `[0, V-1]`, so the requirement is
**`V ≤ 65536`**.

| declared vocab | max id | fits uint16? | file size @1.2B tok |
|---:|---:|---|---:|
| 50,257 (GPT-2, today) | 50,256 | yes, 15,279 ids of headroom | 2.400 GB |
| 50,304 (padded GPT-2) | 50,303 | yes | 2.400 GB |
| **64,400 (LFM2 TRUE vocab)** | 64,399 | yes, 1,136 ids headroom | 2.400 GB |
| **65,536 (LFM2 padded width)** | 65,535 | **yes — exactly, ZERO headroom** | 2.400 GB |
| 65,537 (65,536 + one special token) | 65,536 | **NO → uint32** | **4.800 GB** |
| 100,278 / 100,352 (dolma2) | 100,277 | **NO → uint32** | **4.800 GB** |

**So the parent's framing is half right and the correction makes it sharper, not weaker:**
65,536 does *not* overflow uint16 — it fills it exactly. The trap is that it has **zero
headroom**, so the very next special token added (a `<|pad|>`, a chat marker, a domain
sentinel) silently doubles the corpus on disk and forces a full rewrite. That is worse than
an outright overflow because it is a cliff you cannot see until you step off it.
**MEASURED (arithmetic) / CONFIRMED**, and independently confirmed against the airlock
validator's own `_min_dtype_size_for_vocab`
(`edullm-data/src/edullm_data/validate.py:796-808`, quoted in
`KDA-LIV/docs/claude-audit/solutions/06-SOLUTION-data-tokenizer.md:262-283`).

Note also the *ids you would actually write* are ≤ 64,399 for LFM2 (max observed 64,389 over
5.2M tokens), because the 1,136 padded rows are never emitted. So uint16 is safe for the
data even at declared width 65,536 — the risk is entirely about future tokenizer edits.

### Login-node etiquette / Slurm

- `nproc` = **80** on the login node; my run used 1 core at `nice -n 15` for ~25 s of CPU.
  Do **not** run a 16-way retokenization on the login node.
- **CPU-only Slurm jobs do NOT consume the 4 occupied GPU slots.** MEASURED from `sinfo`:
  partitions are `normal*` (21 nodes: barley-01..04, oat-01..06, rye-01..02, wheat-01..08),
  `bigmem` (rye-01..02), and `gpu` (oat-01..06). A `--partition=normal` job with no
  `--gres=gpu` lands on a `normal` node and touches no GPU allocation. `oat-*` appears in
  both partitions, but the GPU is only allocated via `--gres`. **MEASURED / CONFIRMED.**
  Time limit is **2 days** on `normal`, which is 50x more than this job needs.
- Disk: root fs `109G / 74G avail`; **`/scratch/users` is a separate 106 TB NFS mount with
  69 TB free**, and `/tmp` is a local 1.1 TB disk with 1.1 TB free. `agent-runs` is 532 GB —
  untouched. **A 2.4 GB (or even 30 GB) write to `/scratch` is not a disk risk at all.**
  The "74 G avail" figure is the *root* filesystem, which is not where data lives.
  **MEASURED / CONFIRMED — this substantially de-risks the disk concern in the brief.**

---

# PART A — endpoint-by-endpoint audit

## A.0 The governing principle, stated once

**All arms share one tokenizer** (design doc §4, first line: *"All arms share tokenizer, data
snapshot, data order, token budget, precision, optimizer, context curriculum, and eval
harness"*). Therefore a vocab change is **common-mode across arms** for anything that is a
*comparison between arms*. It can only be MATERIAL where:

- (i) an **external, published** number is being compared against (Griffin's phonebook curve,
  RULER's 85.6% threshold, Zoology's 6.4%, LFM2's 40.3 tok/s), or
- (ii) the **absolute placement of a difficulty knob** relative to a fixed architectural
  constant matters (the conv receptive field `ℓ·k` is measured in *tokens*, so a token means
  a different amount of text), or
- (iii) the **embedding fraction of the model** changes enough to dilute the mixer signal
  differentially — which, at fixed geometry, it does *not* (all arms carry the same table).

The measured LFM2/GPT-2 fertility ratio of **0.99991** removes (ii) almost entirely for the
50,257 ↔ 65,536 choice: a "4,096-token context" contains the same amount of text under either
tokenizer to within 0.01%. It does **not** remove it for dolma2 (1.96%).

Embedding fraction at the frozen d=1024 geometry (**MEASURED arithmetic / CONFIRMED**):

| vocab | tied emb params | `L0` total | **emb % of model** |
|---:|---:|---:|---:|
| 50,257 (GPT-2, on disk) | 51,463,168 | 338,838,272 | **15.19%** |
| 50,304 (padded GPT-2) | 51,511,296 | 338,886,400 | 15.20% |
| 64,400 (LFM2 **true**) | 65,945,600 | 353,320,704 | 18.66% |
| **65,536 (frozen)** | 67,108,864 | **354,483,968** | **18.93%** |
| 100,352 (dolma2, intended corpus) | 102,760,448 | 390,135,552 | **26.34%** |

(non-embedding = 287,375,104, invariant.) Note §3.3 of the design doc *itself* already argues
that a large vocab dilutes the mixer signal and that a 32k vocab would cut embeddings to 10.2%
— **50,257 moves in the direction the design doc says it prefers on scientific grounds**
(18.93% → 15.19%), and 100,352 moves sharply the wrong way (26.34%).

## A.1 The table — one row per endpoint

Verdicts: **UNAFFECTED** = same conclusion, same number. **COSMETIC** = same conclusion, the
units/labels shift. **MATERIALLY AFFECTED** = a conclusion could change.

| # | Endpoint | Verdict (50,257 vs 65,536) | Mechanism | Number |
|---|---|---|---|---|
| 1 | **MQAR** (`N512_D64`) | **UNAFFECTED** | Harness builds its own vocabulary; NL tokenizer never enters. See A.2 | vocab **256**, not 50k/65k |
| 2 | **Needle / passkey** | **COSMETIC** | Haystack length is quoted in tokens; needle is a digit string. Both arms same tokenizer | 0.009% length shift |
| 3 | **Phonebook** | **COSMETIC**, with one **MATERIAL caveat on the *external* Griffin comparison** | The L-sweep (entries) is the real x-axis and is tokenizer-free. Only the *secondary* "1024-token window" cross-paper anchor is affected. See A.4 | see A.4 |
| 4 | **BABILong** | **UNAFFECTED, and for a stronger reason than expected** | Buckets are **pre-baked HF dataset configs**, not runtime tokenization. The harness never tokenizes to a budget. See A.5 | buckets fixed on disk |
| 5 | **RULER-short** | **COSMETIC** (already non-comparable) | Harness synthesizes to a token budget with *our* tokenizer, so it adapts. Already labelled "reconfigured, not leaderboard-comparable" | — |
| 6 | **Length extrapolation** (4K→8/16/32K) | **COSMETIC** | Axis is tokens; all arms share it. No external number is quoted on this axis | 0.009% |
| 7 | **AR-Hits sliced perplexity** | **COSMETIC** — but the design **inherits 6.4% from Zoology and must not.** See A.6 | Slice is *defined over tokens*; its size and composition are tokenizer-dependent. But the endpoint is a **within-study arm difference**, so common-mode | measured, see A.6 |
| 8 | **Held-out CE / perplexity** | **COSMETIC** (already not primary) | CE is per-token and not comparable across tokenizers, either way. BPB is the invariant. See A.7 | — |
| 9 | **Topology decode-traffic claim** (`L0` vs `A16-P`, 10% at T≈4,121) | **COSMETIC**, with a real wording change | T is in tokens; the *crossover token count barely moves*, but the amount of *text* it corresponds to changes, and the **weight-byte term moves with vocab**. See A.8 | T shifts ≈ **4,121 → ~4,020** |
| 10 | **LFM2-350M ONNX calibration (40.3 tok/s)** | **UNAFFECTED** | It is a measurement of the **unmodified released checkpoint**, which carries its own vocab regardless of ours. See A.9 | — |
| 11 | Commonsense suite (HellaSwag/PIQA/ARC), `acc_norm` | **COSMETIC** | `acc_norm` normalizes by **byte** length, already tokenizer-invariant by construction | — |
| 12 | **Parameter-matching / ledger tests** (`liv_arms.py`) | **MATERIALLY AFFECTED — the only one in Part A** | `L0_PARAM_TARGET = 354_483_968` and the solved widths `A16-P.swiglu_width=4820`, `N-narrow.(976, 4668)` are all computed at `VOCAB_SIZE = 65536`. See A.10 | see A.10 |
| 13 | **Document-isolation / packing** (`cu_doc_lens`) | **MATERIALLY AFFECTED (mechanically, not scientifically)** | EOS convention changes: GPT-2 uses a single `50256`; LFM2 uses distinct BOS/EOS. Loader must change. See A.11 | 1 constant + 1 test |
| 14 | **`s_δ` / seed-count power analysis** | **UNAFFECTED** | `s_δ` is measured in the pilot under whatever tokenizer is chosen; nothing inherited | — |
| 15 | **Conv receptive-field anchoring** (`ℓ·k` in tokens) | **COSMETIC** | §9 open item 6: "measure the conv receptive field… every distance sweep should be anchored to it." Receptive field is in tokens; the *text* it spans shifts by fertility | 0.009% |

## A.2 MQAR — CONFIRMED, with file+line

**The natural-language tokenizer never enters the MQAR harness.** Verified by reading every
file in `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/mqar/`:

- `mqar_data.py:63` — `vocab_size: int = 8192` is a field of `MQARConfig`, a synthetic
  parameter. `mqar_data.py:100-105`: `d, n, v = cfg.num_pairs, cfg.seq_len, cfg.vocab_size` /
  `half = v // 2  # keys are [0, half), values are [half, v)` / `filler = v - 1`. Tokens are
  **integers sampled directly**, never text.
- `mqar_data.py:213-216` — the calibrated grid uses `CALIBRATED_VOCAB`, measured at **256**.
- `mqar_model.py:97` — `self.embed = nn.Embedding(vocab_size, d_model)` with `vocab_size`
  taken from the MQAR config, not from the arm builder.
- **`mqar_model.py:7-11` says so explicitly in a docstring:** *"MQAR uses a synthetic 8k
  vocabulary… At d=1024 with a 65,536 vocabulary the embedding alone dwarfs the mixer, so a
  difficulty calibrated there would be measuring the embedding table."* — i.e. the harness
  **deliberately avoids** the production vocab.
- **The complete import set of the harness** (all four non-test modules) is:
  `argparse, json, sys, time, pathlib, dataclasses, typing, numpy, torch, torch.nn,
  torch.nn.functional`, plus `from olmo_core.nn.attention.short_conv import ShortConv`
  (`mqar_model.py:25`). **Zero** occurrences of `tokenizer`, `AutoTokenizer`,
  `from_pretrained`, `transformers`, `huggingface`, `gpt2`, `fineweb`, or any text file.

> **CONFIRMED / MEASURED. MQAR is completely UNAFFECTED by the vocabulary decision.**
> It shares only the `ShortConv` *operator* with the real arms, not the embedding table.
> A corollary the parent should note: **MQAR can be calibrated and run today, at zero risk,
> regardless of how the vocab question resolves.**

## A.3 Needle / passkey — COSMETIC

Design lists passkey as Tier-1 item 1.8, "fixed total length, random depth, ≥500 trials,
first-integer-in-100-tokens accuracy" (`05_evaluation.md:2021`). The haystack is natural text
and the needle a digit string.

- Within-study: both arms tokenize the same haystack the same way → **common-mode**.
- The one place tokenizer touches it: *"first integer in the next 100 tokens"* is a token
  window, so at 0.009% fertility difference it contains the same text. **Negligible.**
- Passkey is explicitly labelled a **regression check only, 1 seed** — it is not an endpoint
  that ranks arms, so even a real shift would not change a conclusion.
- One genuine mechanical note: digit tokenization differs between BPE vocabularies (GPT-2
  splits digits idiosyncratically; many newer tokenizers split digits individually). This
  changes the *number of decode steps* for the answer, not whether the answer is right, and
  scoring is exact-match on the decoded string. **COSMETIC / CONFIRMED.**

## A.4 Phonebook — the one place a cross-paper comparison is at stake. Address it head-on.

**The design doc makes this a *primary* endpoint** (§6.1b: *"Make this a primary endpoint — it
targets exactly the LIV-vs-attention division of labor… and there is a published curve to
compare against"*). The Griffin claim, verbatim from `05_evaluation.md:518`:

> *"Griffin solves perfectly up to its 1024-token local-attention window and extrapolates
> somewhat beyond."*

**Is a cross-paper comparison at "1024 tokens" meaningful across tokenizers? Answer: the
comparison as currently framed is ALREADY not meaningful, and this is true regardless of which
vocab we choose. INFERRED / CONFIRMED.** Four independent reasons:

1. **Griffin's 1024 is a *model hyperparameter*, not a property of the task.** It is the width
   of Griffin's local-attention window — an architectural constant of *their* model measured
   in *their* tokens (Griffin/RecurrentGemma uses the Gemma SentencePiece tokenizer, 256k
   vocab, a very different fertility from GPT-2 or LFM2). Our model has **no local attention
   window at all** — the 6 GQA layers are *full* attention. **There is no corresponding
   constant in our architecture to compare 1024 against.** The analogous constant for us is
   the **conv receptive field** `ℓ·k`, which for k=3 across 10 LIV layers is ~20-30 tokens,
   three orders of magnitude away. The two numbers are not on the same axis.
2. **Griffin's phonebook result is figure-only with no numeric scores** — `05_evaluation.md:518`
   flags this explicitly (⚠️ *"figure-only, no numeric scores"*). You cannot do a quantitative
   cross-paper comparison against a curve you can only eyeball, at any vocab.
3. **The scale gap is 20×.** Griffin is 7B trained on 300B tokens; we are 350M on 2-5B. Any
   difference would be dominated by scale, not by tokenizer.
4. **The x-axis of the phonebook task is `L`, the number of phonebook ENTRIES — not tokens.**
   From Jelassi et al. §4.3 as quoted in `05_evaluation.md:477`: *"The x-axis of Figure 1c is
   'Number of entries in phone-book.'"* Each entry is a fixed-format line
   `"John Powell: 609-323-7777"`. So **the primary difficulty knob is tokenizer-free by
   construction.** Only the derived statement "L=70 entries ≈ 1-2k tokens of context"
   (`05_evaluation.md:492`) carries a token unit, and that is a parenthetical.

> **VERDICT: COSMETIC.** The phonebook endpoint should be reported **against L (entries)**,
> which is what Jelassi et al. actually swept and what our arms differ on. The Griffin
> comparison should be **downgraded from "a published curve to compare against" to a
> *qualitative* precedent** — "a hybrid whose only long-range mechanism is *local* fails past
> its window; ours uses *full* attention, so we predict no such ceiling" — and the sentence in
> §6.1b that promises a numeric comparison should be softened. **This is a finding independent
> of the vocab decision, and it is arguably the most useful thing in Part A**: the design doc
> is currently over-promising a cross-paper comparison that its own dossier already documented
> as figure-only.
>
> If the comparison is kept in any form, the honest way to state it is in **entries**, not
> tokens: "Griffin solved up to roughly its window; we solve up to L=X entries." And if a
> token figure is quoted at all, quote it as **bytes or characters of haystack**, which is
> tokenizer-invariant.

## A.5 BABILong — UNAFFECTED, and the reason is stronger than "it uses our tokenizer"

I checked the actual harness implementation. Two layers, and they disagree in an important way:

**Layer 1 — the upstream BABILong generator (`booydar/babilong`).** Length buckets ARE defined
in tokens, measured by an **injected** tokenizer. From `babilong/babilong_utils.py`,
`SentenceSampler.get_sample(sample_size)` (fetched verbatim):
```
tokenized = self.tokenizer.encode(' ' + sent, add_special_tokens=False)
total_len += len(tokenized)
if total_len >= sample_size:
    cutoff = total_len - sample_size
    sample[-1] = sample[-1][:-cutoff]
```
and `NoiseInjectionDataset.__getitem__`: `background_text_len = sample_size - task_len`.
So the *generator* pads PG19 background text to a token budget using whatever tokenizer it is
handed. **No tokenizer is hardcoded.** The HF card for `RMT-team/babilong-1k-samples` says only
*"9 configs, corresponding to different sequence lengths in tokens: 0k, 1k, 2k, 4k, 8k, 16k,
32k, 128k"* — **it never names the tokenizer.** **UNCLEAR which tokenizer baked the buckets**
(likely GPT-4/tiktoken or Llama-2, but I could not confirm it from the card or the code).

**Layer 2 — how lm-eval-harness actually consumes it, and this is decisive.** From
`lm_eval/tasks/babilong/common_utils.py` (fetched verbatim):
```
config_name = kwargs.get("max_seq_lengths", "0k")
dataset = datasets.load_dataset("RMT-team/babilong-1k-samples", name=config_name,
                                split=qa_split)
```
`max_seq_lengths` selects an **HF dataset configuration name** — a *pre-built, already-padded*
subset downloaded from the Hub. **There is no runtime tokenization and no runtime truncation.**
A `get_tokenizer` helper exists in that file but **is never called** anywhere in it.

> **CONSEQUENCE, and it is the opposite of the brief's hypothesis: BABILong does NOT adapt to
> the model's tokenizer, and it does NOT need to.** Every model — ours at 50,257, ours at
> 65,536, Mamba-130M, GPT-2-137M — receives the **byte-identical prompt string** for a given
> bucket. The bucket label ("4k") is a fixed property of the shipped data, not of the consumer.
> So a tokenizer change on our side **changes nothing about which text lands in which bucket.**
> It only changes how many of *our* tokens that fixed text occupies — which matters solely for
> whether it fits in our context window, and at 0.009% fertility difference it does.
>
> **UNAFFECTED / CONFIRMED.** One residual note worth putting in the paper: because the buckets
> were baked with *someone else's* tokenizer, the label "4k" is approximate for every model,
> including the published RMT/Mamba-137M baselines. That is a pre-existing property of
> BABILong shared by everyone who uses it, not something our decision creates. Report the
> bucket name, and if precision is wanted, report the measured token count under our tokenizer
> alongside it (one line of code).

## A.6 AR-Hits — the design INHERITS Zoology's 6.4% and it must not. Recommendation below.

**What OUR design actually says.** Design doc §6.1, verbatim:
> *"**AR Hits** — the final token of a bigram that already appeared *in the same context* and
> appeared **≤1250×** in training data (**6.4%** of Pile tokens)"*

and `05_evaluation.md:50`:
> | **AR Hits** | Final token of a bigram that previously appeared in the same context, and
> appeared **≤ 1250×** in the *training* data | **6.4 %** |

**Two inherited constants, and they are inherited for different reasons.**

| constant | inherited from | is it transferable? |
|---|---|---|
| **6.4%** (slice size) | Zoology, on **the Pile**, with **GPT-2/NeoX tokenization** | **NO — it is a property of the corpus × tokenizer pair.** We are on FineWeb-Edu, not the Pile. |
| **1250×** (rarity threshold) | Zoology, over a **10B-token** Pile training set | **NO — it is an absolute count over a training set of a specific size.** Our train set is 1.2B tokens, **8.3× smaller**, so the same absolute threshold selects a *much larger* fraction of bigrams as "rare." |

**The 1250 threshold is the bigger problem and nobody has flagged it.** A count threshold does
not transfer across training-set sizes. At 1.2B tokens, a bigram that Zoology would have seen
1250 times we see ~150 times, so a naive `≤1250` cut on our counts marks nearly everything
rare and the slice **loses its discriminative power** — it degenerates toward "all in-context
bigram repeats," which is a much larger and less AR-specific slice.
**INFERRED / CONFIRMED (arithmetic).** The fix is to **rescale the threshold by the
training-token ratio** (1250 × 1.2/10 ≈ **150** for our corpus) or, better, to define the cut
by a **percentile of the bigram frequency distribution** so it is corpus-size invariant.

**Is the slice tokenizer-dependent? Yes — but it is common-mode.** The slice is defined over
tokens: which positions count as "final token of a repeated bigram" depends on where the BPE
merges fall. A vocab change moves the *membership* of the slice. But the endpoint is
`Δ NLL_AR-slice between two arms trained on the same tokenized corpus`, and the gap-attribution
formula `%gap = [Δlog(φ_AR)·|T_AR|] / [Δlog(φ)·|T|]` is a **ratio in which |T_AR| appears in
both the numerator and (implicitly, via the arm-difference) the comparison**. Both arms are
scored on the *same* slice. **So the ranking of arms is unaffected; only the absolute
"% of tokens" label moves.** **COSMETIC / CONFIRMED for the within-study use.**

> ### RECOMMENDATION (this is the actionable output of A.6, and it is vocab-independent)
>
> **Re-measure the slice on our own corpus and publish our own number. Do not print 6.4%.**
> The design doc currently reads as if 6.4% is a constant of nature. It is a Pile × GPT-2 ×
> 10B-tokens measurement. Concretely:
> 1. Build the bigram frequency table on **our** train split (a counting pass — I measured the
>    encode side at ~500k tok/s/core, so a full 1.2B-token counting pass is **~40 min at 1 core,
>    ~4 min at 16**; the `Counter` is the real cost, not the tokenization).
> 2. Set the rarity threshold as a **percentile**, not the literal 1250, and report which.
> 3. Report the resulting slice size for our corpus and say plainly that it differs from
>    Zoology's 6.4% because the corpus, the tokenizer, and the training-set size all differ.
> 4. `05_evaluation.md:1824` already scoped this as *"Write it (~100 LOC)… a training-corpus
>    bigram frequency table with the ≤1250× threshold. (c) is the only real work"* — the change
>    is to make the threshold a parameter rather than the literal 1250.
>
> **This costs nothing extra and it removes a claim a reviewer would catch.** It is also
> **the same amount of work at either vocabulary**, so it does not favour either option.

### A.6b MEASURED: the AR-Hits slice under both tokenizers, on OUR corpus

Rather than argue about it, I measured it. Script `/scratch/users/ericrcwu/liv/arslice.py`,
raw JSON `/scratch/users/ericrcwu/liv/arslice_results.json`. Method: decode 7,702 val docs and
27,974 train docs from the real `.npy` files, encode the **identical text** under both
tokenizers, build a bigram frequency table on ~30M train tokens (scaled ×40 to the 1.2B corpus,
so the "≤1250× in training" cut becomes ≤31.2 in the sample), pack val into 4096-token windows,
and count positions where the bigram ending there already appeared earlier in the same window
AND is rare. ~8M val tokens scored per tokenizer.

| | GPT-2 (50,257) | LFM2 (64,400) | ratio |
|---|---:|---:|---:|
| val tokens encoded (same text) | 7,992,244 | 7,955,478 | 0.9954 |
| 4096-token windows | 1,951 | 1,942 | — |
| distinct bigrams in 30M train sample | 4,743,791 | 4,688,459 | 0.988 |
| **in-context bigram repeat rate** | **22.511%** | **21.210%** | **0.9422** |
| **AR-Hits (repeat AND rare)** | **5.364%** | **5.465%** | **1.0189** |

**MEASURED / CONFIRMED. Three results:**

1. **The AR-Hits slice is essentially tokenizer-invariant between 50,257 and 64,400 on this
   corpus: 5.364% vs 5.465%, a 1.9% relative difference.** The mechanism worry in the brief
   ("this slice is DEFINED over tokens, so its size and composition are tokenizer-dependent")
   is **real in principle but empirically ~2%** for this pair. **REFUTED as a material concern.**
2. **Our measured slice is 5.4%, not Zoology's 6.4%** — a 16% relative difference, arising from
   the corpus (FineWeb-Edu vs the Pile) and the training-set size, not from the tokenizer.
   This directly supports the A.6 recommendation: **re-measure, do not inherit.**
   (My threshold rescaling — 1250 scaled to the sample — is itself an approximation; the point
   stands that the number is ours to measure, not Zoology's to lend.)
3. The **raw in-context repeat rate does move** (22.5% → 21.2%, −5.8%): a coarser tokenizer
   produces slightly fewer repeated bigrams because merges absorb repetition into single
   tokens. But the *rarity* filter cancels most of it, because the same merges also make each
   surviving bigram rarer. That cancellation is why the final slice barely moves.

Both numbers are common-mode across arms in any case. **Endpoint verdict stands: COSMETIC.**

## A.7 Held-out CE / perplexity, and the BPB recommendation

**CE is not comparable across tokenizers, and the dossier already says so.**
`05_evaluation.md:1838-1840`: *"If tokenizers differ, per-token perplexity is **not comparable**
(different event spaces — Paloma G4). Use **bits per byte**."* And `:1709`: *"if all
architectures share one tokenizer (they should — hold it fixed), **fix the vocabulary and
report plain per-token NLL**; use BPB only when comparing against external models."*

So: **any comparison of our CE to an external published perplexity is invalid under EITHER
option.** This is not a cost of choosing 50,257 — it is already true at 65,536, because
OLMo-core's defaults (reordered-norm, QK-norm, z-loss) already make us non-comparable to
published Mamba-2/GDN numbers (design §3.4, stated twice). **CE is explicitly demoted from
primary in §6.1.** **UNAFFECTED-in-practice / CONFIRMED.**

### Does the design mention BPB? Yes — in the dossier, NOT in the design doc.

- `05_evaluation.md` has BPB in **three** places: §7.5 with the verbatim formula
  (`BPB = −ℓ / (B · ln 2)`, `:1842`), Tier-1 item 1.2 ("NLL / BPB per domain", `:2010`), and
  the SNR argument (`:1866`).
- **`docs/liv-brainlift-experiment-design.md` never says "BPB", "bits-per-byte", or
  "bits per byte" — zero occurrences.** The recommendation was made in the research thread and
  **did not propagate into the design.** **CONFIRMED (grep).**

> ### RECOMMENDATION: report BPB regardless of the decision. It is nearly free.
>
> **Cost: one extra accumulator in the eval loop.** `BPB = (L_tok / L_bytes) · (NLL_per_token
> / ln 2)`. You already compute `NLL_per_token` and you already know `L_tok`; the only new
> quantity is `L_bytes`, the UTF-8 byte count of the eval text, which is **a single integer
> computed once per eval set and cached**. I measured the byte count of my 5,000-doc sample in
> the same pass as everything else: 23,932,410 bytes for 23,848,948 chars, i.e. `len(s.encode())`
> over the corpus — **sub-second for the whole 8M-token val split.** There is no extra forward
> pass, no extra GPU time, no new dependency.
>
> **Engineer-hours: <1.** **Value: it is the ONLY metric in the entire endpoint suite that
> survives a tokenizer change**, which means (a) if the vocab decision is ever revisited, the
> old runs remain comparable to the new ones, and (b) it is the only loss number that can
> legitimately appear next to an externally published one. Given that this study's whole
> problem is that its primary quality endpoint is underpowered, adding a *free* metric that is
> robust to the one decision currently in dispute is an obvious buy.
>
> Two conditions from Paloma, both cheap: evaluate **documents individually with disjoint
> chunking** (G5), and **state the context/stride policy** — `05_evaluation.md:1846-1852` warns
> that a sliding-window stride flatters the attention-heavy arm, which is exactly the `A16-P`
> vs `L0` comparison. That warning applies at any vocabulary.

## A.8 The topology / decode-traffic systems claim — COSMETIC, but the number moves

Design §3.1: *"A **10% end-to-end decode-traffic win arrives at T ≈ 4,121 tokens** — i.e.
exactly at the training context."* T is in tokens, and vocab enters through the **LM-head
weight bytes**, which are re-read every decode step (tied embeddings; the head is
`vocab × d`).

I reproduced the design's model and confirmed it reproduces **both** published anchors exactly
— T = 4,121 and the 6.6% / 36.2% KV shares at vocab 65,536 — which validates the model before
I perturb it. **MEASURED / CONFIRMED.**

| vocab | LM-head bytes (bf16) | weight read / token | **T at the 10% win** | KV share @4K | KV share @32K |
|---:|---:|---:|---:|---:|---:|
| 50,257 | 102.9 MB | 677.6 MB | **3,939** | 6.91% | 37.27% |
| 50,304 | 103.0 MB | 677.7 MB | 3,939 | 6.91% | 37.27% |
| **65,536 (design)** | 134.2 MB | **708.9 MB** ✓ | **4,121** ✓ | **6.63%** ✓ | **36.22%** ✓ |
| 100,352 | 205.5 MB | 780.2 MB | 4,535 | 6.06% | 34.04% |

**Does vocab change the crossover? Yes, by −4.4% at 50,257 and +10.0% at 100,352.**
And the direction is *favourable* for the smaller vocab: **a smaller vocab means fewer weight
bytes, so the KV term reaches 10% of the total sooner.** At 50,257 the 10% win arrives at
**T ≈ 3,939**, and KV is **6.91%** of decode traffic at 4K instead of 6.63% — i.e. the claim
gets slightly *stronger*, not weaker.

**Does the CLAIM's meaning change?** Two parts, and they separate cleanly:

- **"The win arrives at T ≈ 4,121 tokens"** — the *token* number moves to 3,939. The *text*
  it corresponds to is unchanged to within 0.009% (measured fertility). So the claim's
  physical content — "at roughly the training context, the mostly-LIV topology is 10% cheaper
  in decode traffic" — is **identical**. Only the printed integer changes.
- **"— i.e. exactly at the training context"** — this is the rhetorically load-bearing half,
  and it **survives and improves**: 3,939 is *below* 4,096, so at 50,257 the win is already
  slightly **past** 10% at the training context, whereas at 65,536 it arrives just after.
  **The 50,257 version of this sentence is marginally easier to write, not harder.**

**Verdict: COSMETIC.** One number in one table must be recomputed. **The sibling computing the
ledger should also recompute this table** — it is the only systems number in the design that
is vocab-sensitive, and it is easy to forget because it lives in §3.1, not in §6.

## A.9 The LFM2-350M ONNX/GGUF calibration datapoint (40.3 tok/s) — UNAFFECTED

Design §7.3 and HANDOFF key-decision 5: *"Take **one** calibration datapoint on the
**unmodified** LFM2-350M GGUF/ONNX (the ONNX path already runs: 40.3 tok/s)."*

**Reasoning it out:**

1. **It is a measurement of somebody else's artifact.** `onnx-community/LFM2.5-350M-ONNX` q4
   ships with its own embedded tokenizer at whatever vocab Liquid chose. Running it produces
   40.3 tok/s **whether or not our models share that vocab** — we are not feeding it our data
   or our weights.
2. **Its purpose is calibration, not comparison.** It anchors "what does a real deployed model
   of this shape actually run at," and — more importantly — it produced the **per-op decode
   profile** (`MatMulNBits` 91.2%, `Conv` 1.0%) that the design calls *"the single most useful
   artifact produced by this research"* (§7.2 item 4). That profile is a statement about
   **where time goes in an LFM2-shaped graph**, and the shape is set by the layer schedule and
   d, not by the vocab.
3. **Would a vocab change make it misleading?** Only in one respect, and it is a *quantitative
   footnote*: the `MatMulNBits` 91.2% share includes the LM head, whose size is proportional to
   vocab. At 50,257 the head is 23% smaller, so the matmul share would be a point or two lower
   and the conv share a hair higher. **This slightly weakens the "P3 attacks the 1%" argument —
   by roughly 0.1 percentage points.** Immaterial, but worth a sentence if a reviewer is
   pedantic.
4. **Crucially, our models are never latency-compared to this datapoint.** The design's §7.3
   frames it as "one real edge datapoint," and §7.1's ranking of P1/P2/P3 is derived from the
   *op shares*, not from tok/s. No arm is benchmarked against 40.3 tok/s.

> **UNAFFECTED / CONFIRMED. This datapoint does NOT require our models to share LFM2's vocab.**
> The one honest caveat to add to the paper: *"the per-op profile is measured on the released
> 65,536-vocab checkpoint; our models use vocab V, which shifts the LM-head share of
> `MatMulNBits` proportionally."* One sentence. Nothing else changes.

## A.10 The parameter ledger and the arm-builder tests — the ONE materially affected item

This is the only place in the whole audit where a vocab change actually **breaks something that
currently passes**. Located precisely, in the uncommitted worktree
`/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer`:

| file:line | constant | what it is |
|---|---|---|
| `src/olmo_core/nn/transformer/liv_arms.py:65` | `VOCAB_SIZE = 65536` | the single source of truth; every builder takes `vocab_size: int = VOCAB_SIZE` (lines 197, 268, 324, 359) |
| `.../liv_arms.py:69` | `L0_PARAM_TARGET = 354_483_968` | the frozen ledger |
| `.../liv_arms.py:126` | `A16-P.swiglu_width = 4820` | **solved** at vocab 65,536 |
| `.../liv_arms.py:155-156` | `N-narrow.d_model = 976, swiglu_width = 4668` | **two-stage solved** at vocab 65,536 |
| `.../liv_arms.py:16, 21` | docstring "Vocabulary (tied) 65,536" | documentation |
| `src/test/nn/transformer/liv_arms_test.py:36` | `d, vocab, k = D_MODEL, 65536, 3` | a hardcoded literal in a test |

Tests that would fail (from `liv_arms_test.py`):
`test_l0_hits_the_exact_frozen_parameter_target` (`:29`),
`test_l0_ledger_reconciles_component_by_component` (`:54`),
`test_solvers_reproduce_the_committed_widths` (`:158,161`),
`test_narrow_control_is_solved_against_the_arm_it_controls` (`:106`),
`test_all_attention_control_is_parameter_matched_to_l0` (`:112`),
`test_kernel_width_arms_differ_only_in_kernel_width` (`:85`, asserts exact counts).

**The good news, and it is substantial: the design was built to absorb this.**
`vocab_size` is **already a parameter everywhere**, never hardcoded inside the builders, and
the widths are produced by `solve_swiglu_width()` and `solve_d_model()` — **solvers, not
constants** — with `test_solvers_reproduce_the_committed_widths` asserting the declaration
still equals what the solver returns. HANDOFF says this explicitly:
*"Derived widths are solved, never guessed… a test asserts the committed constants still equal
what the solvers return — so a drift between declaration and derivation fails CI."*

> **So the change is: edit ONE constant (`VOCAB_SIZE`), re-run the two solvers, paste the two
> new widths and the new `L0_PARAM_TARGET`, fix one test literal, update a docstring.**
> The CI design turns what would otherwise be a silent-corruption risk into a **mechanical,
> self-verifying edit**. **MEASURED (code read) / CONFIRMED.**
>
> **Estimated engineer-time: 30-60 minutes**, dominated by running the test suite. This is the
> single most reassuring finding in Part B's cost analysis, and it is worth stating that it is
> a *direct payoff* of the arm builder's design discipline.
>
> **Scientific impact: zero.** As established in
> `KDA-LIV/.../06-SOLUTION-data-tokenizer.md:89-91` for the sibling protocol and as it applies
> identically here: every arm carries the **same** `V × d` table, so a vocab change adds the
> **same** `ΔV · d` to every arm and **cancels exactly** in every arm-to-arm contrast.
> The `A16-P` and `N-narrow` matching *targets* move, but they are re-solved to the same
> relative tolerances (0.03% and 0.0145%).

## A.11 Document separation / EOS convention — mechanically affected, scientifically not

The design treats document isolation as *"a confound with teeth — not a detail"* (§3.3) and
Phase 0 requires `cu_seqlens` threaded through convs **and** attention. The current corpus uses
GPT-2's single `50256` as both EOS and document separator (`meta.json`:
`"eos_token_id": 50256`), and `doc_lens` is derived by **scanning for that id**.

- **GPT-2**: one token, `<|endoftext|>` = 50256, serving as BOS, EOS, and separator.
- **LFM2**: distinct `<|startoftext|>` / `<|im_end|>`-style tokens. The prior audit measured
  LFM2's `eos_token_id` = **2** (`06-SOLUTION-data-tokenizer.md:281`). So the separator id, and
  possibly the *convention* (is a BOS emitted per document?), both change.
- **dolma2**: `eos_token_id` = **100257**, `pad_token_id` = 100277 (`edullm-data/README.md:25`).

**What must change:** the EOS constant in `meta.json`, the EOS constant wherever `doc_lens` is
derived, and the assertion that pins it. **What must NOT change:** the *convention* must be
identical across arms — which it trivially is, since all arms read the same corpus.

> **Risk rating: LOW but real, and it is a silent-failure class.** A wrong EOS id yields
> **silently garbage document boundaries** — the conv bleeds across documents in a way that
> looks like a slightly worse model rather than a bug, which is exactly the failure the design
> §3.3 warns about ("If document masking is on for the attention arms while the conv silently
> bleeds… every comparison is broken"). **Mitigation is one assertion**, and the prior audit
> already wrote it (`06-SOLUTION-data-tokenizer.md:290-296`, assertion 4).
>
> **This risk exists in EVERY option including "do nothing"**, because Phase 0 has to thread
> `cu_doc_lens` through attention regardless. Retokenizing adds one constant to get right.

---

## A.12 WHICH ENDPOINTS WOULD CHANGE THEIR CONCLUSION? Being ruthless about it.

The brief asks for the distinction between *changing units* and *changing conclusions*. Here it
is, stated as harshly as the evidence permits.

### Endpoints that change their CONCLUSION under a vocab change: **NONE.**

I looked for one and could not construct a case. The reason is structural, not lucky:

1. **All arms share the tokenizer** (design §4). Every endpoint that ranks arms is a
   *difference*, and the embedding table is byte-identical across arms, so the vocab
   contribution **cancels exactly**. This is arithmetic, not an approximation.
2. **The primary endpoints are recall, extrapolation, and AR-Hits** (§6.1, HANDOFF decision 2)
   — all within-study comparisons.
3. **MQAR, the sharpest endpoint, does not use the tokenizer at all** (A.2, CONFIRMED at
   file+line).
4. **BABILong receives byte-identical prompts regardless** (A.5, CONFIRMED from the harness
   source).
5. The one endpoint whose *slice definition* is genuinely token-dependent — AR-Hits — I
   **measured** at 5.364% (GPT-2) vs 5.465% (LFM2): a **1.9% relative difference** on our own
   corpus, and common-mode besides (A.6b).
6. The one metric that could be compared externally — CE — **is already non-comparable** for
   two other reasons the design already states (OLMo-core defaults; different corpus), so
   the tokenizer adds nothing to a wound that is already fatal.

### Endpoints whose UNITS or LABELS change: 4, all trivially fixable.

| endpoint | what changes | fix |
|---|---|---|
| Topology decode-traffic (§3.1) | `T ≈ 4,121` → `T ≈ 3,939`; KV@4K 6.63% → 6.91% | recompute one table |
| Parameter ledger (§3.1, `liv_arms.py`) | `354,483,968` → `338,838,272`; two solved widths | edit 1 constant, re-run 2 solvers |
| AR-Hits slice label | "6.4% of Pile tokens" → "5.4% of our tokens" | **must change anyway** — see A.6 |
| ONNX per-op profile footnote | LM-head share of `MatMulNBits` shifts ~0.1pp | one sentence |

### The three things that are NOT common-mode — named, as requested.

Being scrupulous: there **are** three effects that are not perfectly common-mode. All three are
small and **two of them favour the smaller vocab.**

1. **Embedding fraction, and therefore mixer dilution.** At 65,536 the embedding is 18.93% of
   the model; at 50,257 it is 15.19%. This is *not* an arm-to-arm difference — every arm carries
   the same table — but it **is** a change in how much of the model's capacity is *not* the
   thing we are studying. **§3.3 of the design doc argues this explicitly and in the direction
   of a smaller vocab** (*"a large vocab does dilute the mixer signal we are trying to measure…
   a 32k vocab would cut that to 10.2%"*), then keeps 65,536 anyway solely *"to preserve the
   exact released-scale ledger."* **Since the ledger is unreachable regardless (the 65,536 is
   itself a pad of 64,400), that justification has evaporated — and the design doc's own
   scientific argument now points at the smaller vocab.** This is the strongest *positive*
   argument for 50,257 in the entire analysis. **INFERRED / CONFIRMED.**
2. **`N-narrow`'s solve gets slightly easier or harder.** `N-narrow` reduces `d_model` to match
   `F-r128`'s parameter count. A larger embedding table means a larger fraction of the model
   moves with `d_model` (the table is `V × d`), so the `d_model` grid is *coarser* in relative
   terms at a large vocab. HANDOFF records that at 65,536 the 16-multiple `d_model` grid alone
   only reached **0.815%** of target and needed a second SwiGLU-width stage to close to 0.0145%.
   At 50,257 the embedding is a smaller share, so the same grid step moves fewer parameters and
   the solve is **marginally tighter**. Second-order, but it is a real non-common-mode effect
   and it favours the smaller vocab. **INFERRED / UNCLEAR magnitude** (the sibling doing the
   ledger can quantify it; I did not re-run the solver).
3. **Decode-traffic crossover shifts by −4.4%** (A.8) — and again this **favours** the smaller
   vocab, since the win arrives *before* rather than *after* the 4,096 training context.

> **Summary: I found no endpoint whose conclusion flips, and the three genuine non-common-mode
> effects all point the same way — toward the smaller vocabulary.** That is a stronger result
> than "it doesn't matter."

---

# PART B — cost, logistics, and risk

## B.0 THE FINDING THE PARENT NEEDS FIRST: the intended corpus is a THIRD vocabulary

The brief asked me to determine what tokenizer `s3://edullm-datasets/olmo-150b-dolma2/` is in.
**Settled entirely from local files. Zero AWS calls.**

**Answer: `allenai/dolma2-tokenizer`, true vocab 100,278, EOS 100,257, pad 100,277, stored as
headerless little-endian uint32.** Padded embedding width **100,352** (= 128·ceil(100278/128)).
**MEASURED / CONFIRMED**, from three independent local sources:

| source | line | statement |
|---|---|---|
| `edullm-data/README.md` | :25 | *"`tokenizer/dolma2-bpe/v1` — `allenai/dolma2-tokenizer`… **`vocab_size 100278` / `eos 100257` derived** from tokenizer.json, never typed. The corpus above pins it by `manifest_sha256`."* |
| `edullm-data/README.md` | :24 | *"`pretrain/olmo-150b-dolma2/v1` — **157,467,202,883 dolma2 tokens** across 6,911 headerless **uint32** shards"* |
| `edullm-data/docs/CONSUMER-CONTRACT.md` | :293 | *"For dolma2, `vocab_size = 100278 > 65535`, so it lands on `uint32`"* |
| `KDA-LIV/docs/claude-audit/solutions/06-SOLUTION-data-tokenizer.md` | :185-190 | pins `manifest_sha256 b37b8954…a772267`, `TokenizerConfig.dolma2().padded_vocab_size()` = **100,352** |

There **is** a manifest: the published dataset carries `dataset.json` with a
`manifest_sha256` and 6,911 CRC64NVME refs, and the tokenizer is published as its own dataset
(`tokenizer/dolma2-bpe/v1`) that the corpus pins by hash. Provenance is intact and auditable.

> ### THE CONFLICT, STATED PLAINLY
>
> **The design doc simultaneously freezes `vocab 65536 tied` (§3.1, §3.3) and names
> `s3://edullm-datasets/olmo-150b-dolma2/` as the corpus (§3.3, HANDOFF decision 7).
> These two decisions are mutually exclusive.** dolma2 emits ids up to 100,277, which would
> index out of bounds on a 65,536-row embedding table. As the prior audit put it
> (`06-SOLUTION-data-tokenizer.md:213-216`):
> *"set `vocab_size=65536` in `TransformerConfig` while tokenizing with dolma2 — this is
> **illegal** and must be refused… **There is no way to have the number and the corpus.**"*
>
> **So "retokenize to 65,536" does not reconcile the design with its own corpus plan — it
> makes the conflict worse**, because it would mean building a *third* corpus while a 157B-token
> validated one already sits in S3.
>
> **The parent must resolve which corpus this study actually runs on before the vocab question
> is even well-posed.** There are three live vocabularies:
>
> | | vocab | corpus | status |
> |---|---:|---|---|
> | **on disk today** | 50,257 | FineWeb-Edu 10BT, 1.2B tok, `/scratch/.../kda/lm/data` | **runnable now**, uint16, 2.4 GB |
> | **frozen in the design** | 65,536 | *does not exist* | must be built |
> | **named as the corpus** | 100,352 | `olmo-150b-dolma2/v1`, 157.5B tok, sealed + validated | **runnable now**, uint32, 630 GB in S3 |
>
> Note the two runnable ones bracket the frozen one, and **neither is 65,536.**

**A sibling protocol in this same repo already faced this exact decision and resolved it.**
`KDA-LIV/docs/claude-audit/solutions/06-SOLUTION-data-tokenizer.md` (dated 2026-07-30) is a
full decision document for the *sub-500m KDA* track, reaching: **"(a) — accept dolma2, rename
the family, recompute the ledger"**, at embedding width 100,352, family renamed
`liv_kda_gqa_sub390m_v1`. **That track has already abandoned the 65,536 anchor.** Whatever this
study decides, it should decide *knowingly* against a sibling that went the other way — or
align with it. **CONFIRMED / this is a coordination finding, not a technical one.**

## B.1 Option 1 — retokenize to 65,536. Real cost, three sub-cases.

First, a correction that reframes the whole option: **there is no tokenizer with vocabulary
65,536.** LFM2's real tokenizer has **64,400** entries (verified live by the prior audit across
all three LFM2 repos; independently confirmed by my own benchmark run, which loaded
`LiquidAI/LFM2-350M` and reported `get_vocab_size(True) = 64400`, max emitted id 64,389 over
5.2M tokens — **MEASURED / CONFIRMED**). So "retokenize to 65,536" actually means "tokenize
with LFM2's 64,400-entry tokenizer and then declare a 65,536-wide embedding with 1,136 dead
rows." That is a legitimate thing to do — it is what Liquid did — but it should be named
accurately, because it is the *padding*, not the tokenizer, that produces the 354,483,968 anchor.

### (a) Decode the existing GPT-2 corpus and re-encode — **the cheapest by an order of magnitude**

**Round trip verified lossless: 5,000/5,000 documents exact, 5,191,165 → 5,191,165 tokens**
(§0, MEASURED). Document boundaries are preserved because we split on EOS *before* decoding.

| Item | Cost |
|---|---|
| Download | **none** — data is already on FarmShare `/scratch` |
| Disk needed | **2.4 GB** output (uint16) + ~5.5 GB transient text if materialized; stream it and it is ~0 |
| Wall-clock, 1 core | **~58 min** (measured 4.3M tok/s decode, 502k tok/s encode) |
| Wall-clock, 16-core Slurm CPU job on `normal` | **~4 min** |
| GPU slots consumed | **zero** (CPU-only partition, verified from `sinfo`) |
| Corpus obtained | 1.2B tokens (the same text we have) |
| Engineer-hours | ~2-3 (script + validate + new `meta.json`) |

**Caveats.** (i) It reproduces the *existing* 1.2B-token corpus only — it cannot grow the
corpus, and 1.2B is **below** the design's own 2-5B/run budget and far below the 20 tok/param
Chinchilla floor of 7B at 350M. (ii) It inherits FineWeb-Edu's single-source narrowness.
(iii) Any document that GPT-2 tokenized lossily would be silently altered — my 5,000-doc sample
shows **zero** such documents, but that is 0.4% of the corpus, so the honest statement is
"lossless on a 5,000-document sample; assert per-document round-trip during the real pass, it
costs nothing."

### (b) Re-download FineWeb-Edu sample-10BT from HF — **the disk risk is real but on the wrong filesystem**

**MEASURED from the HF tree API** (`/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT?recursive=1`):
> **14 parquet files, 28,518,193,415 bytes = 28.52 GB.**
Row count 9.67M documents; license **ODC-By**; the card does not name the tokenizer for its
`token_count` column. **MEASURED / CONFIRMED** — this matches the brief's "~27-28 GB" estimate.

**But the disk framing in the brief is misdirected.** The 74 GB figure is the **root filesystem**
(`/dev/sda2`, 109 G, 74 G avail). Data does not live there. **MEASURED:**
- `/scratch/users` — TrueNAS NFS, **106 T total, 69 T available** (35% used)
- `/tmp` — local disk, **1.1 T total, 1.1 T available** (2% used)
- `agent-runs` is **532 GB** on `/scratch`, which has 69 TB free.

> **So a 28.5 GB download is ~0.04% of available `/scratch`. It is NOT a disk risk.**
> **REFUTED as a concern.** What it *is*: a ~30-60 min download, 28.5 GB of parquet to keep or
> delete, and a full 10B-token tokenization pass (**~5.5 h at 1 core, ~35 min at 16 cores**,
> extrapolated from my measured 502k tok/s). Output at 10B tokens = **20 GB uint16**.
>
> The one genuine reason to prefer (b) over (a): **it gives 10B tokens instead of 1.2B**, which
> is 8.3× more and finally clears the design's 2-5B/run budget. If a retokenization happens at
> all, **(b) is the better buy** — it costs ~30 min more download and ~30 min more CPU, and it
> fixes a token-budget problem the design already has.

### (c) The intended corpus, `olmo-150b-dolma2` — **incompatible with 65,536, see B.0**

Already tokenized at **100,278 / padded 100,352, uint32**. 157.47B tokens, 6,911 shards,
~630 GB in S3, sealed and validator-clean. **Retokenizing it to 65,536 would require
downloading and decoding 630 GB of uint32 shards, decoding them with the dolma2 tokenizer, and
re-encoding with LFM2's** — a ~157B-token pass, i.e. **~87 hours single-core, ~5.5 h on 16
cores**, plus 630 GB of transfer and staging.

> **Recommendation: do not do this under any circumstance.** It destroys a sealed, validated,
> hash-pinned corpus's provenance to chase a padded integer. If dolma2 is the corpus, the
> vocabulary is **100,352** and the ledger is **390,135,552**, full stop.

### Downstream work if Option 1 proceeds (either sub-case)

| Item | Where | Hours |
|---|---|---|
| `VOCAB_SIZE` constant | `liv_arms.py:65` | 0.05 |
| `L0_PARAM_TARGET` | `liv_arms.py:69` | 0.05 |
| Re-solve `A16-P.swiglu_width`, `N-narrow.(d_model, swiglu_width)` | run the two solvers | 0.25 |
| Test literal `65536` | `liv_arms_test.py:36` | 0.05 |
| Re-run 55-test suite (23 arm + 32 mixer) | | 0.5 |
| Docstrings + design-doc §3.1/§3.3 edits | | 0.5 |
| New `meta.json`: tokenizer, vocab, **eos_token_id**, dtype, counts | data dir | 0.25 |
| Loader EOS constant + assertion | wherever `doc_lens` is derived | 0.5 |
| Retokenization script + per-doc round-trip assertion | new | 2.0 |
| Slurm CPU job + validation pass | | 1.0 |
| Eval-harness configs (none hardcode vocab; RULER/BABILong take the model's tokenizer) | | 0.25 |
| **Total** | | **~5.5 engineer-hours**, ~1 hour wall-clock compute |

### Risks of Option 1

| Risk | Severity | Note |
|---|---|---|
| **License** | LOW | FineWeb-Edu is **ODC-By** (MEASURED from the HF card). LFM2 *code* is Apache-2.0; the **tokenizer artifact** is under the LFM Open License, which a sibling agent is auditing — **defer to them**; I flag only that it is not automatically Apache-2.0 like `modeling_lfm2.py`. **UNCLEAR — sibling's call.** |
| **Disk** | **NEGLIGIBLE** | 2.4-20 GB against 69 TB free on `/scratch`. **REFUTED as a concern.** |
| **Subtly different tokenizer** | LOW | The `tokenizers` library loads `tokenizer.json` verbatim; no normalizer ambiguity. But **do assert `get_vocab_size(True) == 64400` and `max(id) < 64400` after the pass.** |
| **Round-trip losing document boundaries** | **NEGLIGIBLE** | Boundaries carried in the loop, not the text. 5,000/5,000 exact. **MEASURED.** |
| **EOS/BOS convention change** | **MEDIUM — the real one** | GPT-2's single `50256` → LFM2's distinct BOS/EOS (eos id **2**). Silent-failure class: wrong id ⇒ garbage `doc_lens` ⇒ conv bleeds across documents ⇒ looks like a worse model, not a bug. **Mitigation: one assertion.** See A.11. |
| **The 65,536 pad has ZERO uint16 headroom** | MEDIUM | One future special token ⇒ uint32 ⇒ 2× file ⇒ full rewrite. **Mitigation: declare 64,400 (1,136 ids of headroom) or write uint32 from the start.** |
| **Opportunity cost / schedule** | **HIGH — the real cost** | Not the CPU hour. It is the decision latency, the re-verification of a 55-test suite that currently passes, and the risk of introducing a silent data bug into a study whose Phase 0 is *finished and green*. |
| **It still does not deliver the anchor** | **CERTAIN** | 64,400 → 353,320,704 honest, or 354,483,968 only by copying Liquid's pad. **The stated purpose of the option is unreachable by the option.** |

## B.2 Option 2 — redeclare at 50,257 / 50,304

### 50,304 vs 50,257: verify the alignment claim

**MEASURED arithmetic / CONFIRMED:**
```
50,257 mod 64 = 17    mod 128 = 81     -> NOT aligned
50,304 mod 64 =  0    mod 128 =  0     -> aligned to 64 AND 128 (50304 = 64*786 = 128*393)
                      mod 256 = 128    -> NOT a multiple of 256
50,304 - 50,257 = 47 extra rows
```
So **50,304 is a multiple of 64 and 128 but not of 256.** (Contrast 65,536 = 2^16, aligned to
everything; and 100,352 = 128·784, aligned to 128 but not 256 either.)

### The speedup: what is actually documented

The canonical reference is nanoGPT. Its `train.py` prints, verbatim (**MEASURED — fetched**):
> `"defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"`

Karpathy's widely-cited report of this change was a **~25% step-time improvement** on his A100
setup — a striking number, but note it was measured in a specific 124M-parameter, high-batch,
`torch.compile`-era configuration where the vocab matmul is a large share of work.
**ASSUMED / UNCLEAR at our scale.** I did not reproduce it and cannot on a login node.
The honest statement for our setting: at d=1024 with 350M params and 16 layers, the LM head is
15.2% of parameters, so a tensor-core-alignment gain on the head bounds out well below 25% —
plausibly **1-5% step time**. It is a free-and-positive effect of unknown size. **Take it; do
not claim a number for it.**

### Are the 47 extra rows free, correctness-wise? YES — with the mechanism stated correctly

**The brief's framing is right and worth stating precisely.** The 47 padded rows correspond to
token ids 50,257-50,303, which **the tokenizer never emits**. Therefore:

- They **never appear as a target**, so they never receive gradient through the numerator of
  the cross-entropy.
- **They DO receive gradient** — through the **softmax denominator**. Every step, `logsumexp`
  over all 50,304 logits includes them, so `∂L/∂z_j = p_j > 0` for each dead row `j`: a
  **positive** gradient on their logits, which after the minus sign in the update pushes them
  **down**. Training drives them toward −∞ logit, i.e. probability zero. This is exactly the
  behaviour you want and it is self-correcting.
- **Correctness impact: none.** The model can only ever be *wrong* by assigning mass to a dead
  row, which costs it loss, so the optimizer removes it. It cannot corrupt anything because no
  dead id is ever a label and no dead id is ever an input.
- **Cost: 47 × 1,024 = 48,128 parameters** (tied, so counted once) = **0.014% of the model.**
  Utterly negligible.
- One real caveat: **at generation time, sample from the first 50,257 logits or mask the rest.**
  Early in training a dead row can win an argmax. Standard practice; one line. Note the design's
  generation path is only used for `generate_until` evals (BABILong, RULER, phonebook), so this
  is a genuine correctness item, not theoretical. **INFERRED / CONFIRMED.**

> **Verdict: padding to 50,304 is free correctness-wise, costs 0.014% of parameters, and buys
> an unquantified but positive alignment gain. Do it.** The only discipline required is masking
> dead logits at decode.

### What breaks, and what it costs

Exactly what A.10 enumerated: one constant, two solver re-runs, one test literal, one docstring,
one design-doc paragraph. **~1 engineer-hour including the test suite.** No data work at all —
the corpus already exists, already validated, already loaded by a script that has run.

New ledger (sibling computes authoritatively; my arithmetic for cross-check):
`L0` = 287,375,104 + 50,304 × 1,024 = **338,886,400** at 50,304
(**338,838,272** at 50,257). Embedding share **15.20%**.

### What the paper would have to say instead — one sentence

> *"We train an LFM2.5-350M-shaped research model (16 layers, d=1024, 10 gated short-conv + 6
> GQA at [2,5,8,10,12,14], SwiGLU 4608) at a 50,304-token embedding width — GPT-2's vocabulary
> padded to a multiple of 128 — giving 338,886,400 parameters; geometry fidelity to the release
> is established by a bit-for-bit tensor-sum audit of the released checkpoint rather than by
> matching its padded vocabulary width."*

That sentence is **more** defensible than the current one, because the current one implicitly
claims a tokenizer relationship that does not exist (65,536 is not LFM2's vocabulary).

## B.3 Option 3 — keep the 65,536 width but use a DIFFERENT tokenizer

Two sub-variants. Both are worse than they look.

### (3a) Some other off-the-shelf 65,536-vocab tokenizer

There are a few (e.g. some Qwen/Yi-lineage and multilingual BPEs land near 64k). But note the
framing question the parent posed: **does "some 65k tokenizer" retain any of the value of
"LFM2's actual tokenizer"?**

**No. And the reason is that "LFM2's actual tokenizer" itself retains almost none of the value
it is imagined to have.** The chain of value was supposed to be:
`LFM2 tokenizer → 65,536 vocab → 354,483,968 params → "exact released-scale ledger"`.
The chain is broken at the first link (LFM2's tokenizer is 64,400) and at the second
(65,536 is a pad, not a vocab). So substituting a different 65k tokenizer loses nothing that
was actually there — but it also **gains** nothing, while adding an unfamiliar tokenizer with
unmeasured fertility, unknown digit handling, unknown EOS convention, and possibly a
non-permissive license. **Strictly dominated. Reject.**

### (3b) Train a fresh 65,536-vocab BPE on FineWeb-Edu ourselves

**Cost estimate.** `tokenizers`' `BpeTrainer` is the standard tool. Training a 65k BPE needs a
representative text sample; 2-5 GB is the usual guidance and our corpus decodes to **5.51 GB**
of UTF-8 for the full 1.2B tokens (MEASURED), so the entire corpus is already in the right size
band. `BpeTrainer` is multi-threaded and memory-hungry:

- **Wall-clock: 1-4 hours on 16 cores** for a 65k merge table over 2-5 GB.
  **ASSUMED / UNCLEAR** — I did not run it (it would be a multi-hour multi-core job, outside
  the "tiny microbenchmark" scope I was given, and it would be impolite on the login node).
- **Memory: 30-80 GB peak** is typical for `BpeTrainer` at this vocab and corpus size — this
  is a genuine reason to use `bigmem` (rye-01/02) rather than a normal node.
- Then the retokenization pass on top: another ~40 min at 1 core / ~4 min at 16.
- Plus: pre-tokenizer/normalizer design decisions, special-token design, a fertility
  measurement, a sanity audit of the merge table, and a publish through the airlock.
- **Engineer-hours: 8-16**, dominated by the design decisions and validation, not the CPU.

**What it buys: a vocabulary of exactly 65,536 with zero dead rows and full uint16 utilization**
— i.e. the *only* way to get a genuine 65,536-entry vocabulary. **What it costs scientifically:
it makes the study MORE bespoke, not less.** Any reviewer asking "why 65,536?" now gets "because
we trained one to be" rather than "because that is what the released model uses" — which is a
*weaker* provenance story than either 50,257 (a standard, universally understood vocabulary) or
100,352 (a published, hash-pinned, validated corpus).

> **Verdict: technically the only way to actually hit 65,536, and precisely for that reason the
> option that most exposes the arbitrariness of the target.** Reject. The parent's framing is
> exactly right: these are *different and unequal claims*, and this variant makes the claim
> weakest while costing the most.

## B.4 Option 4 — a hybrid (train P1/P3 now at 50,257, retokenize later if needed)

**Assess honestly, in two parts, because they have opposite answers.**

### Mixing vocabularies ACROSS ARMS: fatal. Not a judgement call.

Every arm-to-arm comparison — `F-r128` vs `L0`, `L0` vs `A16-P`, `C-near` vs `C-far` — is a
difference of losses or accuracies computed on **the same tokenized data**. Change the tokenizer
on one arm and the two arms are no longer solving the same prediction problem: per-token CE is
in different units, the AR-Hits slice has different membership, and the parameter counts differ
by `ΔV · d` on only one side, so the *entire* matching apparatus (`A16-P` at 0.03%, `N-narrow`
at 0.0145%) becomes meaningless. **Design §4's first sentence forbids it explicitly.** No.

### Mixing vocabularies ACROSS PHASES: coherent only if no cross-phase inference is drawn — and
the design's own statistical structure means inference IS drawn across phases. So: no.

This is the interesting half, and the answer is a firm no for a reason specific to *this* design.
The design's ladder (§8) is: Phase 2 pilot measures `s_δ` → Phase 3 **screens** at 5 paired seeds
→ Phase 4 **confirms** on survivors with ≥8 fresh seeds. HANDOFF §5: *"≥8 fresh paired seeds
never used in selection."*

**That is a selection-then-confirmation design, and its validity depends on Phase 4 being a
replication of the same experiment.** If Phase 3 screens at 50,257 and Phase 4 confirms at
65,536, then:
- Phase 4 is a *different experiment that reuses the arm names*. The selection made in Phase 3
  does not license the confirmation in Phase 4 — there is an unquantified selection-transfer gap.
- `s_δ` measured in Phase 2 at one vocab does not determine the required `n` at the other.
- The CE margin (+0.010 nats) is in different units on the two sides.

The prior audit reached the identical conclusion for the sibling protocol and put it more
sharply (`06-SOLUTION-data-tokenizer.md:154-172`): *"(c) is the worst option and should be
rejected outright… It destroys the ladder's statistical meaning… A screen that selects on
dolma2 CE and confirms on LFM2 CE has an unquantified selection-transfer gap."*
**CONFIRMED — the same argument transfers exactly.**

### The one defensible hybrid: temporal, not scientific.

Run the **whole** study at one vocabulary. If someone later wants an LFM2-tokenizer replication,
run it as a **separately named family with its own manifest**, comparing nothing across the two.
That is not a hybrid; it is Option 2 plus optional future work. **And note it is cheap to keep
open**: retokenizing costs ~1 hour of compute and ~5 engineer-hours (B.1a), so **the decision is
highly reversible** — which is the single strongest argument for not agonizing over it now.

**A genuine partial-hybrid that IS coherent, and worth naming:** MQAR (A.2) uses no natural-
language tokenizer at all. So **MQAR calibration on the real `L0` — which HANDOFF flags as
still-required ("Re-run the sweep on real `L0` before using these settings in the study") — can
proceed TODAY at zero risk, whatever the vocab decision.** That is real parallelism, not a
compromise on comparability. Same for the P1 thin-matmul and P3 conv microbenchmarks, which are
already done and are vocab-free anyway.

---

## B.5 DECISION TABLE

| | **Opt 1a** retok 1.2B→64,400/65,536 | **Opt 1b** re-download 10BT→65,536 | **Opt 2** declare 50,304 | **Opt 3b** train own 65,536 BPE | **Opt 5** use dolma2 @100,352 |
|---|---|---|---|---|---|
| **Scientific validity** | Fine. All arms shared | Fine, and 8.3× more tokens | **Fine — and mixer dilution is LOWEST (15.2% emb)** | Fine but most bespoke | Fine; **emb 26.3%** = worst dilution |
| **Delivers the 354,483,968 anchor?** | Only by copying Liquid's pad | Only by copying the pad | **No — and says so honestly** | Yes, genuinely | No |
| **Wall-clock (compute)** | ~1 h @1 core, ~4 min @16 | ~6 h @1 core, ~1 h @16 (+30 min download) | **0** | 2-5 h @16 + bigmem | **0** |
| **Disk risk** | **None** (2.4 GB / 69 TB free) | **None** (28.5 GB / 69 TB free) | **None** | None | None locally; 630 GB staging |
| **Engineer-hours** | ~5.5 | ~6.5 | **~1** | 8-16 | ~4 (loader + dtype + ledger) |
| **Reviewer optics** | Neutral-to-bad: "you padded someone's pad" | Same | **Good, if stated as in B.2.** Slight "why GPT-2 in 2026?" | Bad: "why 65,536? because we made it so" | **Best on corpus provenance** (sealed, hash-pinned, 157B tok) |
| **Reversibility** | High | High | **Very high** (~5.5 h to change later) | Low (a bespoke artifact to maintain) | High |
| **Token budget** | 1.2B — **below the design's own 2-5B** | 10B ✓ | 1.2B — same problem | 1.2B or 10B | **157B, 31× headroom** ✓ |
| **Consistent with the sibling KDA-LIV track?** | No | No | No | No | **Yes** — that track chose dolma2/100,352 |
| **Blocks anything today?** | Yes, ~1 day of latency | Yes, ~1-2 days | **No** | Yes, ~3-5 days | Partly (needs uint32 loader work) |

---

## B.6 MY RECOMMENDATION

> ### Redeclare at **50,304** and train now — Option 2.
>
> **The single decisive reason: the 65,536 target is a padding constant, not a tokenizer
> property, so retokenizing cannot deliver the thing it was proposed to deliver — while the
> design doc's own scientific argument (§3.3: "a large vocab does dilute the mixer signal we
> are trying to measure") points at the smaller vocabulary the moment the ledger justification
> is removed.**
>
> Retokenizing buys a *different arbitrary padding of a different arbitrary tokenizer*, costs
> ~5.5 engineer-hours and a day of decision latency, and improves **zero** endpoints — I audited
> all 15 and **none** changes its conclusion (A.12).

**Supporting reasons, in descending force:**

1. **No endpoint's conclusion changes** (A.12). MQAR never sees the tokenizer (CONFIRMED at
   file+line). BABILong receives byte-identical prompts (CONFIRMED from harness source).
   The AR-Hits slice moves 1.9% and is common-mode (MEASURED: 5.364% vs 5.465%).
   Phonebook's real axis is entries, not tokens.
2. **Fertility is 0.99991** (MEASURED). A "4,096-token context" is the same amount of text
   either way. Every token-denominated claim in the design keeps its physical meaning.
3. **The mixer-dilution argument now favours 50,257** (15.19% embedding vs 18.93%), and the
   design doc makes that argument itself.
4. **The decode-traffic claim gets slightly stronger** (T: 4,121 → 3,939, i.e. the 10% win
   arrives *before* the training context rather than after).
5. **Phase 0 is finished and green** — 55 tests, exact ledger, float64 parity. Changing the
   vocab means re-verifying it; changing the *corpus* means new failure surface. The value of
   a green Phase 0 in a study that has died mid-run before is high.
6. **It is highly reversible.** ~1 hour of compute, ~5.5 engineer-hours. Nothing is foreclosed.
7. **Pad to 50,304, not 50,257** — free, aligned to 64 and 128, costs 48,128 params (0.014%),
   and buys an unquantified positive matmul gain. Mask dead logits at generation.

**Three things I recommend regardless of the decision** (all cheap, all independently justified):

- **Report BPB alongside NLL** (A.7). One integer per eval set, <1 engineer-hour. It is the
  only tokenizer-invariant loss metric, and it is currently **absent from the design doc
  entirely** (0 occurrences — CONFIRMED) despite being recommended three times in the dossier.
  It future-proofs every run against exactly this decision being revisited.
- **Re-measure the AR-Hits slice on our corpus and stop quoting 6.4%** (A.6). Our measured
  value is **5.4%**, and the **1250× rarity threshold does not transfer to a 1.2B-token
  training set** — rescale it or make it a percentile. This is a reviewer-catchable error today.
- **Downgrade the Griffin phonebook comparison from quantitative to qualitative** (A.4), and
  report phonebook against **L (entries)**. The dossier already records that Griffin's result
  is figure-only with no numeric scores, and Griffin's "1024" is *its own* local-window
  hyperparameter with no counterpart in our full-attention design.

**And one escalation the parent must handle, which outranks the vocab question:**

> **§3.3 names `s3://edullm-datasets/olmo-150b-dolma2/` as the corpus. That corpus is dolma2 at
> vocab 100,278 / width 100,352 / uint32. It CANNOT be read by a 65,536-wide model.** The design
> currently holds two mutually exclusive decisions. A sibling track in this repo
> (`KDA-LIV`) already resolved the identical conflict in favour of dolma2 at 100,352, renaming
> its family to `sub390m`. **Decide corpus first, then vocabulary follows from it** — because if
> the answer is "dolma2," the vocab question is already settled at 100,352 and everything above
> about retokenizing to 65,536 is moot.
>
> My read: **the FineWeb-Edu 50,257 corpus on FarmShare is the right one for this study anyway**
> — 1.2B tokens is thin, but the study's primary endpoints are recall/extrapolation/AR-Hits, the
> arms are all under-trained equally (design §3.4 already concedes being below Chinchilla), and
> the 15.19% embedding share is the least mixer-diluting of the three. If more tokens are wanted,
> Option 1b's re-download gets to 10B for ~1 hour of work — **but do it at 50,257, not 65,536.**
