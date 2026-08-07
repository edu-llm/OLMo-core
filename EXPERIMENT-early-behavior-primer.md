# Early behavior primer (370M)

Implementation: `.edullm/frontload_cl/` (design note: `.edullm/frontload_cl/DESIGN.md`).  
Dataset layout / publish: `DATASET-DESIGN-frontload-cl.md`.

**Status (2026-08-07):** train scripts and curricula match this doc (ladder 370M, primer vs control ordering, shared pools). `pretrain/frontload-cl-10b/v1` and `sft/frontload-cl-chat-sft/v1` are on `s3://edullm-data`; the pretrain release is registered as `frontload-cl-10b-v1`. Platform submit path is `.edullm/run.yaml` + `edullm check`/`submit` (see DESIGN). Held-out SFT-like NLL logging during PT is still deferred.

## Claim

A short dose of SFT-like data early in pretraining — after LR warmup, while most of the corpus is still ahead — helps a 370M model get more out of an otherwise normal high-quality (HQ) pretrain mix. The thing being tested is **timing**, not whether SFT-like data is useful at all. Both runs see the same SFT-like token budget; one front-loads part of it, the other spreads it evenly.

## Setup

Train an OLMo2-arch **370M** from scratch for **10B** tokens, then run the **same real SFT** on both checkpoints. Two pretrain runs only. Match tokenizer, sequence length, optimizer, HQ mix, anneal, and final SFT. The only intentional difference is how the SFT-like tokens are ordered.

Use the **AI2 model-ladder 370M hyperparameters** (Bhagia et al. / OLMo-ladder Table 1), not OLMo2 1B/7B production settings:


|                 | Ladder 370M                                                        |
| --------------- | ------------------------------------------------------------------ |
| Architecture    | d_model 1024, 16 layers, 16 heads (`TransformerConfig.olmo2_370M`) |
| Sequence length | 4096                                                               |
| Global batch    | **192 × 4096 = 786,432** tokens                                    |
| Peak LR         | **7.8×10⁻⁴**                                                       |
| Warmup          | **472 steps** (~371M tokens at that batch)                         |
| Schedule        | linear warmup → cosine decay over the 10B run                      |
| Grad clip       | 1.0                                                                |
| Optimizer       | AdamW / SkipStepAdamW, weight decay 0.1, betas (0.9, 0.95)         |


At 10B tokens that is about **12,715 steps**. 10B is a bit above ladder 1×C (~7.4B) for this size.

## Pretraining data (shared pools; arms differ only in schedule)

Tokenizer: **OLMo 2 / Dolma2 BPE**. All token targets are after that tokenizer. Subsamples use seed **42069666**. Build **one** copy of each pool; both arms read the same shards.

### HQ — 9.8B tokens

| Dataset | Split / take | Tokens |
|---|---|---|
| `HuggingFaceFW/fineweb-edu` | **main pool**: random subsample of the released corpus (already `int_score >= 3`), seed **42069666**; exclude anneal-pool docs | **8.36B** |
| `HuggingFaceFW/fineweb-edu` | **anneal pool**: `int_score >= 4` (still within FineWeb-Edu); disjoint from main; seed **42069666** | **950M** |
| `HuggingFaceFW/finewiki` | random subsample, seed **42069666** | **490M** |
| **HQ total** | | **9.80B** |

FineWiki is mixed at **5% of HQ tokens** in every HQ-bearing phase (main and anneal).

**FineWeb-Edu anneal** means: for the last **1B** tokens of pretrain, stop the normal main mix and train only on a higher-quality slice of the *same* FineWeb-Edu family — here the **950M** `int_score >= 4` pool plus **50M** FineWiki — while LR is low on the cosine tail. It is still general web/edu text, not instruct. The point is a short end-of-training upweight of stronger educational documents (common “cooldown” practice), not a new data domain.

Remaining FineWiki (**440M**) is mixed into the pre-anneal HQ stream with the **8.36B** main FineWeb-Edu.

Held out (not in train): **20M** FineWeb-Edu + **5M** FineWiki (seed **42069666**, disjoint) for HQ NLL.


### SFT-like — 200M tokens (PT objective only)

Plain packed documents; **no** chat template; loss on all tokens. OpenHermes here is **disjoint** from the SFT OpenHermes 100k: draw the SFT 100k first (seed **42069666**), then sample this PT slice from the remainder until the token budget is hit.

| Dataset | Split / take | Tokens |
|---|---|---|
| `HuggingFaceTB/smollm-corpus` / `cosmopedia-v2` | random subsample of `text`, seed **42069666** | **80M** |
| `HuggingFaceTB/finemath` / `finemath-4plus` | random subsample of `text`, seed **42069666** | **60M** |
| `brahmairesearch/OpenHermes-2.5-Formatted` | plain-text reformat; remainder after SFT draw; seed **42069666** | **30M** |
| `facebook/natural_reasoning` | plain-text reformat; seed **42069666** | **30M** |
| **SFT-like total** | | **200M** |

Held out (not in train): **5M** tokens total from the same four sources in the same 40/30/15/15 proportions (seed **42069666**, disjoint) for SFT-like domain NLL.

**PT grand total:** 9.8B HQ + 0.2B SFT-like = **10.0B** train tokens.

## Warmup

**472 steps / ~371M tokens**, per the ladder 370M recipe. Same on both arms. Linear 0 → peak, then cosine over the rest of the 10B.

Warmup is only an LR ramp. Data during warmup is **normal pretraining data** for that arm — the start of the usual mix — not a special HQ-only stage. On the primer arm, the 100M SFT-like *block* is scheduled after warmup, so those first tokens are ordinary HQ main-mix until the block starts. On the control arm, the dispersed SFT-like rate is part of normal data from step 0, including during warmup. Match total SFT-like tokens across arms over the full 10B (200M each).

## Schedule

Durations below are in **tokens**. Shared skeleton:

1. **Warmup (~371M)** — normal pretrain mix for that arm; LR ramp only.
2. **SFT-like timing** — arms differ (below).
3. **Main** — FineWeb-Edu + FineWiki as above, plus SFT-like per arm.
4. **Anneal (last 1B, 10%)** — FineWeb-Edu anneal pool (`int_score >= 4`) + FineWiki at 5%. General corpus only. No SFT-like. Same on both arms.
5. **Real SFT** — shared post-training.



### Primer arm

During warmup: normal HQ main mix (no concentrated block yet).  
After warmup: **100M** SFT-like as one contiguous block.  
Then: HQ main, with the **other 100M** SFT-like mixed uniformly through the post-primer, pre-anneal window. Anneal has no SFT-like (best-HQ upweight only, FineWiki still at 5%).

### Control arm

No early block. Spread all **200M** SFT-like uniformly through the pre-anneal run, **including warmup** (normal mix = HQ + that thin SFT-like rate from step 0). Anneal matches the primer arm.

## Decision rule

After shared SFT, compare:

1. GSM8K
2. ARC-Easy + ARC-Challenge
3. IFEval

Primer wins if it beats control clearly on at least two of those three. HellaSwag is a sanity check that general ability did not collapse.

Also log held-out SFT-like domain NLL through pretrain. On the primer arm it should drop during the early block, then rise toward HQ NLL during main. If it never rises, do not claim washout.

## Final SFT (shared)

Tokenizer: **same as OLMo 2** (Dolma2 BPE). **One epoch only** — no repeats. Same mix for both PT arms.


| Dataset                                    | Split / take                        | Examples    |
| ------------------------------------------ | ----------------------------------- | ----------- |
| `HuggingFaceH4/no_robots`                  | **all** `train`                     | **9,500**   |
| `HuggingFaceH4/ultrachat_200k`             | **all** `train_sft`                 | **207,865** |
| `AI-MO/NuminaMath-1.5`                     | random subsample, seed **42069666** | **250,000** |
| `brahmairesearch/OpenHermes-2.5-Formatted` | random subsample, seed **42069666** | **100,000** |
| **Total**                                  |                                     | **567,365** |


Why this shape: full UltraChat + no_robots for IFEval-style following; 250k Numina for a readable GSM8K signal; a thin OpenHermes slice for instruct diversity without a second huge chat pile that could wash out PT differences. Held out: no_robots `test` (500), UltraChat `test_sft`, and a fixed 5k Numina slice (seed **42069666**, disjoint from the 250k train draw).

After tokenization with the OLMo 2 tokenizer, expect on the order of **~400–700M** packed SFT tokens for one epoch (measure and record; do not grow the mix to chase a token target). No preference/DPO data.

## Out of scope

Late massed instruct/CoT. Real SFT mid-pretrain. A pure-HQ arm with zero SFT-like tokens. Gated Nemotron/NVIDIA pretrain sets. Alternate web backbones.