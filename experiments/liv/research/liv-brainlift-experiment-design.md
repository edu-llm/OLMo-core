# Designing the LIV Brainlift Experiment on the Liquid AI Architecture

**Status:** design proposal, research-backed; not yet reviewed, not yet runnable.
**Date:** 2026-07-30.
**Source brainlift:** [`Brainlifts/Eric_LIV_Brainlift-1.pdf`](../Brainlifts/Eric_LIV_Brainlift-1.pdf)
**Research dossier:** [`Brainlifts/liv_experiment_research/`](../Brainlifts/liv_experiment_research/)
(9 files, ~17,000 lines, every claim URL-cited and evidence-labelled)

## Material Passport

- **Origin:** deep research sweep (8 parallel agents) + independent arithmetic verification
- **Origin date:** 2026-07-30
- **Verification status:** all brainlift parameter arithmetic **verified exact** (reproduced twice,
  independently); LIV operator **verified against released Apache-2.0 code** (instantiated and diffed
  to 0.0); crossover arithmetic independently re-derived and confirmed to ~2%; three claims
  **corrected** (§1.3) and **one motivating premise falsified** by measuring the released LFM2-350M
  singular-value spectra (§5.1). No language-model or systems result has been produced.
- **Version label:** `liv_brainlift_v1`
- **Relationship to prior work:** this **re-sequences** the existing
  [`liv-kda-gqa-sub500m-experiment.md`](liv-kda-gqa-sub500m-experiment.md), which defers all three
  brainlift proposals. See §2.

---

## 0. Executive summary

The brainlift proposes three changes to an LFM2-style mostly-LIV hybrid: (P1) low-rank gates,
(P2) cross-layer KV sharing, (P3) routed multiscale convolution. **Every number in it checks out
exactly.** But the research says the three are not equally promising, and — more importantly — they
are not equally *measurable*. The single most consequential finding:

> **KV cache is only 6.6% of decode traffic at 4K context and 36.2% at 32K (350M geometry).**
> The "mostly-LIV saves memory and latency" thesis is a long-context thesis. At the contexts an
> academic study can afford to train, most of it is invisible.

This reshapes the design. The recommendation is **not** to run the three proposals as a bundle of
efficiency claims. It is:

| | Proposal | Verdict | Reframe as |
|---|---|---|---|
| **P1** | Low-rank gates | **Run first, as a QUALITY + parameter-efficiency claim.** ~~Decode-latency story~~ — **the microbenchmark gate has been run and the latency claim FAILED** (L40S, CUDA-graphed, 3 trials, ≤0.3% spread: best case `lowrank_fused r=128` is **8.2% slower** than stock LIV while reading 4× fewer bytes; skinny GEMVs achieve only 161 GB/s vs dense's 695). Provably outside Liquid's own search space (novel). Motivating premise ("gates are low-rank") is **falsified** by the released weights → "gates tolerate low rank". Surviving quantitative hook: rank-128 retains **92.6%** of activation-weighted energy at **0.25×** the parameters. | Quality-vs-parameters. **No latency claim.** `grouped` is the systems competitor: +15.3% faster but retains only **0.130** energy vs low-rank's 0.929 at identical cost |
| **P2** | Cross-layer KV sharing | **Run second, with a much narrower claim.** Saves capacity, **not** bandwidth — latency ≈ 0 by construction. **Anticipated three times** (Hymba, Character.AI, Gemma 3n). Its only defensible niche: the sole method that halves resident KV *without changing any layer's receptive field*. | Retrieval-safety study — **no** cross-layer-sharing paper reports needle/passkey/MQAR, and the two available data points both show recall falling |
| **P3** | Routed multiscale conv | **Reframe, don't drop.** Best published width sweep is *flat past k=3*, and the conv is only **1% of decode** — so it cannot produce a speed win. But dilation is measurably ~free, so "multi-scale span at negligible latency cost" is a **true and defensible** claim. | A **quality** claim with a measured latency-neutrality result (§5.3, §7.1) |

Three decisions the design must make up front, because they change everything downstream:
**scale** (350M, not 1.2B — §3.1), **primary endpoint** (recall/extrapolation/AR-Hits, not held-out
CE — §6.1), and **scope** (GPU-primary for methodological reasons, plus one calibration datapoint on
the ONNX path that already runs — §7).

**The one table to internalize before anything else** — measured per-op decode profile of the real
LFM2.5-350M ONNX build: `MatMulNBits` **91.2%**, `Conv` (the LIV depthwise) **1.0%**. Independently
corroborated by the parameter analysis (the depthwise kernel is 0.07% of the mixer). This ranks the
proposals for you: P1 attacks the 91%, P2 attacks bytes, **P3 attacks the 1% and makes it 5.6× worse.**

**The strongest version of this project is not "three efficiency wins."** It is:

> A controlled study of what the LFM2 family's own published record leaves open. Liquid published no
> conv:attention ratio ablation, no kernel-width ablation, and no recall benchmark; CLA and Hymba
> established cross-layer sharing but measured it only against perplexity and a 2-task recall average
> — where Hymba's own numbers show recall *falling* while aggregate metrics rise. The contribution is
> rigorous, seed-replicated, retrieval-focused measurement of a widely-deployed architecture family
> that has never received it.

That framing survives every possible outcome, including all three proposals coming out flat — which
is the single most important property of a capstone experiment design.

---

## 1. What the research established

### 1.1 The LIV operator, pinned to released code

Verified directly against `transformers` v5.0.0rc1 `modeling_lfm2.py` (**Apache-2.0**, so usable
without touching the weights license):

```python
self.in_proj  = nn.Linear(d, 3*d, bias=False)          # conv_bias: false
self.out_proj = nn.Linear(d, d,   bias=False)
self.conv = nn.Conv1d(d, d, kernel_size=3, groups=d, bias=False, padding=2)

BCx     = self.in_proj(x).transpose(-1, -2)
B, C, x = BCx.chunk(3, dim=-2)          # order is (B, C, x) — NOT (B, x, C)
Bx      = B * x
z       = self.conv(Bx)[..., :seqlen]
y       = self.out_proj((C * z).transpose(-1, -2).contiguous())
```

Three details that a reimplementation gets wrong by default, all confirmed by an agent that
instantiated the real module and diffed against a reimplementation:

1. **Chunk order is `(B, C, x)`.** Wrong order → 0.083 max diff (silently trains, worse model).
2. **No activation anywhere in the conv path.** The fused path passes `activation=None` explicitly.
   Adding Mamba-style SiLU → 0.041 diff. **This is a live trap in our repo** — OLMo-core's
   `CausalConv1d` defaults to `activation="silu"` *inside* the fused kernel
   (`OLMo-core/src/olmo_core/nn/convolution.py:37`), so reusing it unchanged implements a different
   operator.
3. **No normalization inside the block.** `operator_norm` (RMSNorm) is owned by the decoder layer.

Correct spec reproduces released output to **0.0**. Decode state is `(batch, d, k)` holding the
**gated pre-conv stream** `Bx`, updated by `roll(shifts=-1)`.

**The `block_ff_dim` trap, now solved.** Config says `intermediate_size: 12288` but
`block_auto_adjust_ff_dim: true`, `block_multiple_of: 256`. The transform is
`ff = 256 · ceil(int(2/3 · block_ff_dim)/256)`, which reproduces both released configs exactly:
1.2B → 8192, 350M → 4608. **Miss this and every parameter count is ~50% wrong on the MLP, which is
69% of the model.**

### 1.2 Brainlift arithmetic: all verified exact

| Claim | Verified |
|---|---|
| stock LIV = 4d² + kd = **16.783M** (d=2048, k=3) | 16,783,360 ✓ |
| GQA mixer = 2.5d² = **10.486M** | 10,485,760 ✓ |
| factorized (r=128) = 2d² + 4dr + kd = **9.443M** | 9,443,328 ✓ |
| "7.340M fewer than stock LIV" / "1.043M fewer than GQA" | ✓ / ✓ |
| **LIV mixer is *larger* than GQA** | ✓ 1.60× |
| 12 KiB/token KV, ≈384 MiB at 32K | ✓ |

Formulas were independently validated against **six released checkpoints** to the exact parameter.
The 2.5d² GQA coefficient holds *because* hkv=8 at d=2048; the general form is `2 + 2·(hkv·hd/d)`,
which becomes **3.0d² at d=1024**. Restate for whichever scale is chosen.

### 1.3 Three claims that needed correcting

- **P2 does not reduce bandwidth.** The CLA paper states it directly: *"CLA has no direct effect on
  the memory bandwidth consumed by the attention mechanism in each decoding step"* — consumers
  re-read the shared bank. Only *writes* halve. So P2's latency effect is ≈0 at every context
  length, not the 1-7% a naive resident-KV calculation suggests.
- **A low-rank init error is monotone, not sign-flipping** (§5.1). An earlier draft in the dossier
  claimed the error flips sign with `d·r`; re-derivation shows it is always an undershoot, shrinking
  by exactly the factor r. Corrected in `02_lowrank_gates.md`.
- **`max_position_embeddings: 128000` is not a validated context.** Liquid's own documentation gives
  **32,768**. Do not cite 128K.

### 1.4 Novelty: the good news

- **Zamba is not prior art for P2.** It shares attention *weights* (one block applied 13 times,
  input `LN([x_l, x_0])`) and explicitly recomputes K/V with *"independent activations and KV-cache
  entries at each invocation."* Orthogonal mechanism. (Note Zamba's §II claims weight sharing cuts
  KV cache — it does not, and its own §III says so. Worth a sentence in related work.)
- **P2 is anticipated THREE times, not once — handle this honestly.** See §5.2. The bald claim "apply
  CLA to a hybrid" is *not* novel:
  1. **Hymba** (arXiv 2411.13676, Nov 2024) — cites CLA directly, ships config + working code.
  2. **Character.AI** — deployed in production at >20k QPS, and *does the non-adjacent global-layer
     variant we propose*, reporting no quality regression (no ablation published).
  3. **Gemma 3n** — shipped cross-layer sharing *upstream* into HF transformers and vLLM
     (`num_kv_shared_layers=15`).

  What survives: sharing across an intervening **sequence-mixing block** in a *sequentially
  interleaved* hybrid (Hymba is parallel hybrid-head — its "adjacent" layers have no conv between
  them), sharing between **full-attention** rather than local layers, LFM2's specific GQA-8 ratio, and
  above all **the retrieval evaluation nobody has run.**
- **No public ratio ablation, kernel-size ablation, or recall benchmark exists for LFM2.** STAR
  (the architecture-search paper) never mentions LFM2 and reports no conv:attention ratios; the
  LFM2 report explicitly disowns STAR's proxies as not transferring. `LiquidRULER` is empty
  plumbing. **The 10-conv/6-attn ratio has no published quantitative justification.** This is a
  genuine open contribution, not a replication.
- **P1 is provably outside Liquid's own search space — an auditable novelty claim.** STAR's LaTeX
  source shows its searchable channel-mixing options are exactly `{Diagonal, Dense, Grouped}`, and the
  featurizer genome has **no rank field at all**. So Liquid's architecture search *could not have
  expressed* low-rank gates even in principle. Combined with zero hits across systematic arXiv
  full-text searches, this is an auditable claim rather than "I couldn't find it" — the strongest
  novelty result in the whole dossier.

  The flip side: **`Grouped` (block-diagonal) *is* in that search space and was not selected** for the
  gated-conv featurizer, and gate *sharing* is STAR's own evolutionarily-selected incumbent. So a
  reviewer can fairly say we are proposing an untested structure while a searched one was available.
  That makes `G-grouped` and `S-shared` mandatory controls, not optional — and grouped is also the
  strongest *systems* competitor, since grouped matmuls are better supported than skinny ones and it
  will likely win on latency.

- **A token-dependent router over multiscale conv branches is absent from the entire
  multiscale-conv literature** (MixConv/Res2Net/Inception are all static). Novel — and
  correspondingly unsupported by any prior numbers.
- **Character.AI's production model does P2's exact non-adjacent variant** (§5.2) with ~1-in-6
  global attention, reporting no needle regression — but publishes no controlled ablation. Gap to fill.

---

## 2. Relationship to the existing protocol — this is a re-sequencing

`docs/liv-kda-gqa-sub500m-experiment.md` (dated the same day, "DESIGN REVIEWED") already freezes an
LFM2.5-350M-shaped backbone and **explicitly defers all three brainlift proposals**: factorized
gates at `:206`, multiscale routing at `:207`, cross-layer KV sharing at `:214`, each gated behind
"L0 survives vs A16-P."

That protocol's primary contrast is **K2 vs L0-P** — a *KDA insertion* question. **The brainlift is
not about KDA at all.** These are two different experiments sharing a backbone.

**Proposed resolution.** Treat this as the **LIV-internal** study and state the divergence openly:

- **Inherit** the frozen geometry, layer schedule, parameter ledger, Phase-0 correctness gates,
  statistical discipline, and artifact contract. Do not re-litigate them.
- **Diverge** on sequencing. The existing protocol gates everything behind `L0 vs A16-P` topology
  survival. That gate is expensive (it needs a parameter-matched all-GQA control trained to
  convergence) and it answers a question Liquid has already answered for its own target. P1 is
  *independent* of it: "do factorized gates match full gates" is a within-LIV comparison whose
  control is stock-LIV, not all-GQA. **Recommend running P1 in parallel with the topology gate, not
  after it.**
- **Drop KDA from this study entirely.** It is a separate mechanism with its own completed
  Householder result. Mixing them destroys attribution.

If the topology gate fails, P1/P2/P3 results remain meaningful as *LIV-internal* findings — they
just stop being claims about a competitive architecture. Say so in advance.

---

## 3. Frozen design decisions

### 3.1 Scale: use 350M (d=1024), and this is a *scientific* choice

KV bytes/token depend only on `hkv`, `hd`, and attention-layer count — **none of which change with
d**. So KV/token is 12 KiB at *both* 350M and 1.2B, while weight bytes fall 3.3×:

| | 350M (d=1024) | 1.2B (d=2048) |
|---|---:|---:|
| weight read / decode token | 708.9 MB | 2.341 GB |
| KV read == weight read at | **T = 57,690** | T = 190,474 |
| KV share of decode traffic @ 4K | **6.6%** | 2.1% |
| KV share @ 32K | **36.2%** | 14.7% |

**The smaller model is a ~2.5× more sensitive testbed for every cache claim.** This inverts the
usual "scale up to show a systems win" instinct and is worth stating explicitly in the paper. Adopt
the existing protocol's frozen 350M geometry (16 layers, d=1024, 16/8 heads, hd=64, SwiGLU 4608,
vocab 65536 tied, GQA at [2,5,8,10,12,14], final layer LIV).

**The one efficiency claim that IS testable at affordable contexts belongs to the topology, not to
any of the three proposals.** The mostly-LIV topology cuts KV/token from 32 KiB (all-16-GQA) to
12 KiB — a 20 KiB/token saving against a parameter-matched all-GQA control:

| context T | all-GQA decode traffic | saving | % |
|---:|---:|---:|---:|
| 2,048 | 776.0 MB | 41.9 MB | 5.4% |
| **4,096** | 843.1 MB | 83.9 MB | **9.9%** |
| 8,192 | 977.3 MB | 167.8 MB | 17.2% |
| 16,384 | 1,245.8 MB | 335.5 MB | 26.9% |
| 32,768 | 1,782.6 MB | 671.1 MB | 37.6% |

A **10% end-to-end decode-traffic win arrives at T ≈ 4,121 tokens** — i.e. exactly at the training
context. So `L0 vs A16-P` is a real, measurable systems comparison at 4K and a strong one at 16-32K.
**Lead the systems story with this**, and treat P1/P2/P3 as quality-and-parameters questions layered
on top. This also means the existing protocol's topology gate is not merely a hurdle — it is where
the publishable efficiency result actually lives.

### 3.2 Context: train at 4K, headline at 32K

The FLOP crossover is ~1.4K tokens — at T=1024 the conv mixer costs *more* FLOPs than GQA. **Train
at ≥4K or the architecture's benefit is invisible.** Report by length bin (train length, 2×, 4×, 8×)
and label anything beyond trained length as extrapolation, per the existing protocol.

### 3.3 Corpus, tokenizer, infra: reuse what exists

- **Corpus:** `s3://edullm-datasets/olmo-150b-dolma2/` — 155.6B tokens, already tokenized. Far
  exceeds the 2-5B/run budget. **Open action:** verify its document-length distribution supports
  16K/32K sequences before promising long-context results.

  **The long-document concern is now substantially de-risked.** A direct measurement of FineWeb-Edu
  (1,800 docs, its own `token_count` column) gives median **622** tokens but a heavy tail:
  **30.7% of tokens sit in docs >4K, 8.4% >16K, 3.1% >32K** — sufficient for an extension stage. And
  ABF found *"even with most of the long texts removed, the model can still obtain most of the
  performance gain"*, so this is not a blocker even if Dolma2's tail is thinner. Still worth measuring
  ours rather than assuming.

- **Vocabulary: keep LFM2's 65,536 tied, but know the cost.** At the frozen d=1024 geometry,
  embeddings are **67.1M of 354.4M = 18.9%** of the model; a 32k vocab would cut that to 10.2%. So a
  large vocab does dilute the mixer signal we are trying to measure. I recommend keeping 65,536 anyway
  — it preserves the exact released-scale ledger the existing protocol froze, and the dilution applies
  equally to every arm so it cannot flip a comparison. But if the study moves to a smaller width
  (d=768), revisit: the same vocab would then be ~34% of the model, and the existing protocol already
  warns that a different tokenizer creates a separately-named family.
- **Codebase:** OLMo-core, and the fit is better than expected. Verified present in the local copy:
  - `SequenceMixerConfig.register()` registry — adding a mixer is a one-line decorator
    (`nn/attention/base.py:67`). `GatedDeltaNet` composes `CausalConv1d` three times and implements
    all five abstract methods plus FLOP/param accounting — a line-by-line template.
  - **`block_pattern: List[str]` + `block_overrides: Dict[int, TransformerBlockConfig]`**
    (`nn/transformer/config.py:330-331`) — irregular schedules like LFM2's `[2,5,8,10,12,14]` are
    expressible natively, generated from Python, no YAML duplication across ~20 arms.
  - **`num_params(d_model)` and `num_flops_per_token(seq_len)` are part of the mixer API.** Since
    this entire experiment is defined by parameter/token/compute matching, having the framework
    compute both per-mixer-per-layer is worth more than any other feature — a matching bug silently
    invalidates every result.
  - `WSD` / `WSDS` schedulers (`optim/scheduler.py:160,964`) + `model_ladder` — DataDecide-style
    many-variant ablation methodology out of the box. WSD matters specifically because you can
    branch many arms from one stable-phase checkpoint.
  - `SkipStepAdamW` (`optim/adamw.py:105`) + async checkpointing — relevant given this machine has
    died mid-run before.
  - `src/scripts/train/OLMo_hybrid/` — five scripts for a released, parameter-matched 3:1 hybrid, so
    the transformer/hybrid baseline is *published and credible* rather than self-tuned. That is
    exactly the criticism reviewers level at architecture papers.

  Gaps to accept: **μP is not a first-class coordinate-checked feature** (use `fan_in` init + the
  ladder's empirical LR formula, or port μP); and OLMo-core's defaults (reordered-norm, QK-norm,
  z-loss) differ from the Mamba/`fla`-lineage defaults, so **numbers will not be directly comparable
  to published Mamba-2/GDN perplexities** — this is an internally-consistent study only. Say so.

  **Note the `CausalConv1d` caveat from §1.1:** it is ~90% of the LIV conv (fused triton/cuda
  backend, `cu_seqlens` for document boundaries, `apply_cp` for Ulysses CP) but its `activation`
  defaults to `"silu"` and it has no dilation. Pass `activation=None` explicitly and handle the
  dropped final state.

- **Pre-screen before spending GPU-hours:** run MAD-lab / zoology synthetic tasks (in-context recall,
  noisy recall, selective copying) across candidate configurations first. Given that recall gaps in
  the hybrid literature run to tens of points while perplexity differences run to ~0.06, the
  synthetic tasks are where this experiment's signal actually lives. Caveat: mad-lab is unmaintained
  (last commit 2024-12-17) — budget time for dependency repair.
- **Packing: document-isolated, and this is a confound with teeth — not a detail.** A k=3 causal conv
  that bleeds across a document boundary is a *different operator* than one that respects it, and the
  bleed rate scales with documents-per-sequence. At FineWeb-Edu's median ~622 tokens, a 4K sequence
  holds ~6 documents and a 32K sequence ~50. **If document masking is on for the attention arms while
  the conv silently bleeds (or vice versa), every comparison is broken.** Thread `cu_seqlens` through
  convs *and* attention for every arm, and assert it in a unit test — OLMo-core's `CausalConv1d`
  already accepts `cu_seqlens`, and `GatedDeltaNet` threads `cu_doc_lens`, so the plumbing exists.
  The existing `KDA/lm/prepare_data.py` does EOS-concatenation only and is not sufficient. Consider
  best-fit packing (arXiv 2404.10830), which reports +16.8% context-following and up to 58.3% less
  closed-domain hallucination versus concatenate-then-split.

- **Long-context protocol:** pretrain at 4K for the full budget, then a short extension stage to
  16K→32K at **~2-5% of total tokens** with linear warmup at ~1/10 pretrain LR. **Run positional
  encoding both ways** — NoPE/DroPE (OLMo-core's shipped recipe) and RoPE θ=1e6 (LFM2's choice) — it
  is one extra short run and is the mechanism most likely to drive the long-context result. Note the
  literature's apparent disagreement about full vs sliding-window attention in hybrids (Samba's full
  attention blew up beyond training length, 13.66 ppl @16K vs 9.57 for SWA; Waleffe's extended fine to
  128K) probably reduces to *extrapolation protocol* — NoPE + continued pretraining vs RoPE zero-shot —
  which is exactly why this arm matters.

### 3.4 Training recipe (freeze once, apply to every arm)

Distilled from the 2024-2026 small-model consensus (Samba, Gated DeltaNet, Mamba-2 scaling, OLMo 2,
MiniCPM), all at comparable scale:

| Knob | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), grad clip 1.0, BF16, no dropout |
| Weight decay | 0.1, **not on embeddings** (OLMo 2's choice) |
| Peak LR | ~4e-4 at 350M (GDN and Samba both use 4e-4 at 400M) |
| Schedule | **WSD**, not cosine — see below |
| Warmup | ~1-2% of tokens |
| Batch | ~0.5M tokens |
| Sequence length | 4096 |
| Stability | z-loss 1e-5, QK-norm |
| Init | `fan_in` (`std = 1/√d_in`) — scales better across widths than fixed 0.02, and matters here because arms differ in width |

**Use WSD (warmup-stable-decay), not cosine.** This is the single most important recipe choice for a
many-arm study: with cosine, every arm needs its own full run; with WSD you branch all arms from one
stable-phase checkpoint and pay only the decay. OLMo-core ships `WSD`/`WSDS`. MiniCPM
(arXiv 2404.06395) also re-measured the compute-optimal ratio under WSD and found ~**192 tokens per
parameter**, far above Chinchilla's 20 — so **treat 20 tok/param as a floor, not a target.** At 350M,
20× is 7B tokens; the protocol's 2-5B/run budget is therefore *below* Chinchilla-optimal, which is
acceptable for a controlled comparison but must be stated (all arms equally under-trained).

**Comparability caveat to state in the paper:** OLMo-core's defaults (reordered-norm, QK-norm, z-loss)
differ from the Mamba/`fla`-lineage defaults, so absolute numbers will not line up with published
Mamba-2/GDN perplexities. This is an internally-consistent study.

### 3.5 Two blockers must be cleared first (Phase 0)

1. **No LIV mixer exists.** `grep -riE "\bLIV\b|lfm2|conv_L_cache"` over `OLMo-core/src/` returns
   zero matches. Must be written, with a **copied-weight parity test against HF `Lfm2ShortConv`**
   (the three traps in §1.1 make this non-optional).
2. **Hybrid cached generation is asserted impossible.**
   `generate/.../generation_module.py:108` does `assert isinstance(block.attention, Attention)`.
   Every latency and cache measurement is blocked until this becomes a typed per-layer state API.

Note the existing `CausalConv1d` cannot serve P3 regardless: it has **no dilation parameter**,
hardcodes `padding = kernel_size - 1`, and `return output[0]` drops the final state.

---

## 4. Experimental arms

All arms share tokenizer, data snapshot, data order, token budget, precision, optimizer, context
curriculum, and eval harness. Parameter-match via FFN width; assert counts on a meta device before
every launch.

**Parameter-matched is NOT compute-matched at long context — match on `num_flops_per_token` too.**
Verified at the frozen 350M geometry (N=354.4M, so 6ND = 2.127 GFLOP/token), attention-score FLOPs as
a share of 6ND:

| context T | `L0` (6 attn) | `A16-P` (16 attn) | **difference** |
|---:|---:|---:|---:|
| 4,096 | 2.4% | 6.3% | **3.9%** |
| 16,384 | 9.5% | — | **15.8%** |
| 32,768 | 18.9% | 50.5% | **31.6%** |

So at 32K a parameter-matched `L0` vs `A16-P` comparison is **~32% apart in actual compute** — the
naive `6ND` estimate misses it entirely because it ignores the quadratic term. Two consequences:
report `num_flops_per_token` per arm (OLMo-core computes it), and state which matching convention
each table uses. A "same parameters, better loss" claim at 32K is partly a "more compute" claim
unless you say otherwise.

**Watch the LIV-vs-GQA layer cost when swapping layers.** A LIV mixer costs **1.33× a GQA mixer** at
d=1024 (4.20M vs 3.15M) and 1.60× at d=2048 — both are ~4d² and ~2.5-3d², because GQA shrinks K/V but
LIV is full-width in three streams. **Naive layer-swapping silently inflates the LIV-heavy arms.**
This is the same trap the brainlift already identifies; it just needs enforcing in the arm builder.

### Tier A — foundation (must exist before any proposal is tested)

| Arm | Definition | Role |
|---|---|---|
| `L0` | Stock 10-LIV / 6-GQA, full gates, k=3 | The null hypothesis |
| `A16-P` | 16 GQA, FFN solved to match `L0` | Topology control (inherited) |

### Tier B — P1, low-rank gates (run first)

| Arm | Definition | Role |
|---|---|---|
| `F-r64/128/256/512` | Gates `d→r→d`, value+out full width | The rank sweep |
| `L0` | full-rank gates | Primary control |
| **`N-narrow`** | stock LIV, `d` reduced to match `F-r128` params | **Mandatory.** "Just build a narrower model" is the obvious competing way to spend the parameters |
| **`S-shared`** | one gate projection shared by both gates | Cheap competing reduction; also the structure Liquid's own search space contained |
| `G-grouped` | block-diagonal/grouped gate projections | Structured competitor already inside the searched space |
| `1G` | single gate instead of two | Tests whether two gates are needed at all |

Rank sweep must hold **gate output variance constant per arm** (§5.1) — otherwise it measures scale.

### Tier C — P2, cross-layer KV sharing (run second)

Attention sits at [2,5,8,10,12,14] — **no two attention layers are adjacent**, so CLA's
"pair consecutive layers" recommendation is unreachable. Pair the *closest* layers and make
inter-pair distance the primary axis:

| Arm | Pairing | Gap |
|---|---|---|
| `C-near` | (8→10), (12→14) | 1 conv block between |
| `C-far` | (2→5), (8→10), (12→14) | mixed 2 and 1 |
| `C-all3` | (2→5), (8→10), (12→14) → 3 resident banks | the brainlift's target |
| **`Q-mqa`** | reduce `hkv` 8→4→2 instead | Competing control: "just use fewer KV heads." **But do NOT go to hkv=1 as the primary arm** — see the from-scratch caveat below. Keep GQA-8 as primary, MQA as a *monitored secondary* |
| **`A-fewer3`** | **3 attention layers instead of 6** | **Mandatory and it is the strongest competitor.** It matches CLA2's resident capacity **and halves read bandwidth too**, which CLA cannot do. If this wins, the honest answer is "use fewer attention layers" |
| **`A-fewer`** | 5 or 4 attention layers instead of 6 | **Mandatory.** The cheapest way to cut KV banks |
| **`SWA`** | make some attention layers sliding-window instead of sharing | **Mandatory.** Hymba shows SWA and CLA are largely *substitutes* on capacity (its cache fell 3.8× from SWA, then only 4.4% more from sharing). If SWA gets the same win more simply, that is the answer |
| `MLA` | low-rank joint KV compression | Reviewer will ask — Kimi-Linear (2025, 27 layers, `kv_lora_rank=512`) chose MLA for its full-attention layers and did *no* cross-layer sharing. Include if budget allows |

**A note on the `SWA` control — the redundancy intuition is wrong, and the evidence is clear.** I had
reasoned that a short conv and a sliding window are partly redundant. **No source supports that, and
three contradict it:** Samba shows short conv is *additive on top of* SWA (11.12 → 10.83 ppl); DeltaNet
shows SWA recovers only ~19% of the conv→global recall gap (49.5 → 53.3 against 71.0 for global); and
**LFM2's own STAR search had SWA available and chose full attention.** So conv ≠ sliding window, and
swapping one for the other is not neutral. Keep full attention in the shared pair — now for an
evidence-backed reason rather than an intuition — and keep `SWA` as a capacity competitor only. Relatedly, Granite-4.0-h uses
`position_embedding_type="nope"` (no positional encoding on attention at all, position carried by the
Mamba layers), which is an existence proof that in a mostly-linear hybrid the RoPE question can
partly dissolve.

**Always pair forward (producer = lower layer index).** Two independent reasons: (a) chunked prefill
requires the producer to run before its consumer within a chunk, which is automatic only for forward
pairing — backward sharing would need two passes; (b) CLA's `DenseBack` result independently shows
backward-ish concentration is the worst configuration. Also keep pairs *close*: a "layer 2 produces for
layer 14" pairing is maximally hostile to pipeline parallelism (not an issue at 350M, where we won't
use PP, but worth one sentence for anyone scaling it later).

**Parameter delta must be compensated.** Dropping a consumer's K/V projections saves
`2 × d × (hkv·hd)` params — at d=1024, KV width 512, that is 1.05M per consumer, so ~3.1M for three
consumers (~0.9% of 354M). Small, but non-zero: **this is a cache optimization, not a parameter
optimization**, so the fair control must add those parameters back (via FFN width) or the comparison
silently conflates "sharing helped" with "fewer parameters hurt."

**Implementation sketch** (both reference implementations agree): add
`kv_reuse_group: list[list[int]]` to config (Hymba's shape, e.g. `[[8,10],[12,14]]`); validate every
index is an attention layer, producer < consumer, no layer appears twice; guard `k_proj`/`v_proj`
construction behind `if not reuse_kv` while keeping `q_proj`/`out_proj` unconditional; thread a
`shared_kv_states` mapping through the decoder loop (use `UserDict`, not `dict` — plain dict breaks
FSDP2, a bug HF hit); consumers skip both projection *and* cache update. Follow CLA in giving the
producer's KV path its own norm parameters, since CLA is the result being reproduced.

**One real hazard if layer offloading is ever used:** a scheme that evicts layer *p*'s bank after
layer *p* executes is simply **wrong** under sharing, because the consumer still needs it. Eviction
must trigger on the *last consumer* in the group. No paper or issue discusses this.

**RoPE decision — settled.** The CLA paper does not address pre- vs post-rotary sharing at all (I
checked: absent from the mechanism section, no ablation). But **Hymba's released implementation
settles it**: the producer applies `apply_rotary_pos_emb(None, key_states, cos, sin)` before handing
the tensor on, and the consumer applies RoPE only to its own Q. So **cache and share post-rotary K.**
This is correct because rotation is position-dependent, not layer-dependent, and producer and consumer
see the same `position_ids`; pre-rotary sharing would force each consumer to re-rotate for zero
benefit. Cite Hymba's code for the precedent rather than presenting it as our invention.

### Tier D — P3, multiscale (redesign; see §5.3)

| Arm | Definition | Role |
|---|---|---|
| `k5` / `k9` / `k15` | single dense kernel, **fused via FLA Triton (no width limit)** | Locates where the span curve saturates. **Run these before anything routed.** Not tooling-handicapped, so it is a fair baseline |
| **`L1b-schedule`** | **per-layer span schedule** (e.g. narrow early, wider late) at fixed total params | **Strongly recommended.** This is the axis *all* the learned-receptive-field evidence actually supports, and it is cheaper than any router |
| **`M2-fixed`** | **2 branches**, input-independent | **Add this rung.** Every branch-count sweep in the literature saturates at two (SKNet M=2→3 buys 0.03%); half the cost of the 4-branch arm, and where a real effect would first appear |
| `M-fixed` | 4 branches, input-**independent** weights | Provably ≡ one sparse 15-tap kernel, but covering only **7 of 15 lags** (§5.3) |
| `M-router` | 4 branches, token-dependent softmax router. **Requires τ annealing 30→1, fp32 logits, z-loss** | The proposal. Without the recipe guards a negative result is uninterpretable (§5.3) |
| **`D-dyn`** | unconstrained dynamic short conv (Sieberling low-rank generator, k=3) | **Mandatory.** A general input-dependent filter; expected to beat the constrained router. Note their *rank* sweep gains 0.25 ppl while *span* gains 0.00 — conditioning capacity is the real axis |
| `RepVGG-style` | train 4-branch fixed, **fuse to one 15-tap kernel** for inference | Free at inference; the one place branch structure could add value (§5.3) |

---

## 5. Per-proposal analysis

### 5.1 P1 — low-rank gates: real, bounded, and easy to fake

**The ceiling is ~6%, and it is a mixer-level claim.** The mixer is only 14% of the model:

```
embeddings 11.5% | 10 LIV mixers 14.3% | 6 GQA mixers 5.4% | 16 MLPs 68.8%
```

At r=128 the LIV mixer falls 44%, which is a **6.27% cut to the whole model at d=2048**. Diminishing
fast: r=128→64 buys only 0.45pp more, because the full-width value and output projections (2d²) are the
floor. Report this as a mixer-level result or explicitly justify why the MLP is untouched.

**Mind the scale, because the two figures get confused easily.** The gate reduction is `d/(2r)` — 8× at
d=2048 but only **4× at d=1024**, and we chose the 350M/d=1024 geometry. So:

| geometry | gate reduction | whole-model weight cut = **decode ceiling** |
|---|---:|---:|
| d=2048 (1.2B) | 8× | 6.27% |
| **d=1024 (350M) — ours** | **4×** | **4.44%** |

**Why P1 is nonetheless the best of the three:** decode is weight-bandwidth-bound, so a weight-byte
reduction converts directly into decode latency. **4.44% is the ceiling we can actually claim** — the
only positive latency story any of the three proposals offers at trainable contexts, and small enough
that the methodology below decides whether it is visible at all.

#### ⚠️ CUDA graphs decide the SIGN of this result, not just its size

This is the most important methodological finding for P1. Factorizing turns 1 GEMV into 2 per gate =
**20 extra kernel launches per decode token** (2 gates × 10 LIV layers). Against the time saved by
reading fewer weight bytes:

> #### ⚠️ CORRECTION (2026-07-31): the table below is d=2048. The frozen geometry is d=1024.
>
> The `4.72 µs` figure quoted throughout this document and in `HANDOFF.md` was computed at
> **d=2048** (the 1.2B geometry). The scale decision froze **d=1024**, where the byte saving is
> **4× smaller** (gate reduction is `d/(2r)`), so breakeven falls proportionally. Recomputed by
> [`probes/l40s_breakeven.py`](../Brainlifts/liv_experiment_research/probes/l40s_breakeven.py) at
> r=128, 0.75× achievable bandwidth:
>
> | card | saved/token | breakeven/launch | at 5 µs | at 10 µs |
> |---|---:|---:|---|---|
> | A100-40GB | 27.0 µs | **1.35 µs** | LOSS −73 µs | LOSS −173 µs |
> | H100-80GB | 12.5 µs | **0.63 µs** | LOSS −87 µs | LOSS −187 µs |
> | **L40S-46GB (FarmShare)** | 48.5 µs | **2.43 µs** | LOSS −51 µs | LOSS −151 µs |
>
> Also note **r=512 saves nothing at d=1024** — `2dr ≥ d²` once `r ≥ d/2`, so that rung of the
> planned sweep is a pure loss and exists only as a quality datapoint.
>
> **Consequence: P1's latency claim at d=1024 survives only with BOTH mitigations** — fused
> `d→2r` gates (halves launches to 10, doubling breakeven) **and** CUDA graphs (launch → ~0.5-1.5 µs).
> Fused+graphed on L40S: breakeven **4.85 µs**, i.e. a win at 1.5 µs (+34 µs) and roughly break-even
> at 5 µs. Separate+un-graphed loses everywhere. **This is now the pass/fail line for arm 3a.**
>
> Silver lining: **breakeven scales as 1/bandwidth, so the slower L40S is the most forgiving card
> available** — FarmShare is a *better* venue for this benchmark than an H100 would be.

> #### 🔴 MEASURED 2026-07-31 on L40S: P1's latency claim is DEAD. Keep P1 as a quality claim.
>
> The gate has been run — twice (jobs 1670883, 1670884 on `oat-04`), the second on an idle GPU with
> 3 trials (**spread ≤0.3%**) and **profiler-measured** kernel counts confirming 20/30/40 exactly as
> predicted. Scripts: `probes/p1_launch_bench.py`, `probes/p1_verify.py`. All numbers CUDA-graphed:
>
> | arm | kernels | MiB/tok | graphed | vs dense |
> |---|---:|---:|---:|---|
> | `dense` (stock LIV) | 20 | 40.0 | 56.2 µs | — |
> | `lowrank_fused` r=128 | 30 | 10.0 | 60.8 µs | **−8.2% SLOWER** |
> | `lowrank_sep` r=128 | 40 | 10.0 | 76.5 µs | −36.0% SLOWER |
> | `lowrank_fused` r=512 | 30 | 40.0 | 90.0 µs | −60.1% SLOWER |
> | **`grouped` g=4** | 20 | 10.0 | **47.6 µs** | **+15.3% FASTER** |
> | `grouped` g=2 | 20 | 20.0 | 47.7 µs | +15.2% FASTER |
>
> **Even the best case — fused gates *and* CUDA graphs, the configuration §5.1 predicted would win —
> is 8.2% slower than stock LIV while reading 4× fewer bytes.** So: **drop the decode-latency claim
> from P1 entirely.** It is not recoverable by tuning; see the mechanism below.
>
> **The iso-byte control identifies the cause.** Hold bytes constant and vary only structure:
>
> | at equal bytes | | | penalty |
> |---|---|---|---|
> | 10 MiB | `grouped g=4` 47.6 µs | `lowrank_fused r=128` 60.8 µs | grouped **+21.8%** faster |
> | 40 MiB | `dense` 56.2 µs | `lowrank_fused r=512` 90.0 µs | dense **+37.5%** faster |
>
> The second row is decisive: **identical bytes, +10 kernels, +33.8 µs ⇒ ~3.4 µs per extra kernel
> even under CUDA graphs.** Dense hits 695 GB/s (80% of L40S peak) — genuinely bandwidth-bound — while
> `lowrank_fused r=128` achieves only 161 GB/s. The thin `d→r` GEMV cannot saturate the memory system,
> so the bytes it saves buy far less time than the roofline model predicted. **The analytic model in
> the block above assumed launch overhead was the only penalty; the real penalty is that skinny GEMVs
> are bandwidth-inefficient.** This is the same effect FLAR-SVD measured on Snapdragon (slower than
> baseline despite 2× fewer params) and it flagged r=128 as below the inflection point.
>
> #### But `grouped` does NOT displace low-rank — it fails on quality by a wide margin
>
> `grouped g=4` and `lowrank r=128` are **exactly equal in cost** (0.25× params, 10 MiB/token), so the
> choice between them is pure quality. Measured on the released checkpoint
> (`probes/structure_energy.py`, activation-weighted retained energy, 32,768 tokens):
>
> | structure | params | retained energy |
> |---|---:|---:|
> | `lowrank r=128` | 0.25× | **0.929** |
> | `grouped g=4` | 0.25× | **0.130** |
> | *random mask, 25% density* — null | 0.25× | *0.130* |
> | `lowrank r=256` | 0.50× | 0.965 |
> | `grouped g=2` | 0.50× | 0.336 |
>
> **Block-diagonal structure buys nothing over random sparsity at the same density** (0.130 vs 0.130),
> and channel ordering does not rescue it — random permutations give `[0.125, 0.133]` against 0.130 for
> the identity, so the deficit is structural, not an artifact of how LFM2 happens to index channels.
>
> ⚠️ **Caveat that bounds this conclusion:** the metric measures how well each structure
> *approximates trained dense weights*, which structurally favors low-rank — Eckart-Young gives
> rank-r truncation the *optimal* rank-r approximation, while the block mask is not optimized at all.
> A from-scratch trained grouped layer is not the block-diagonal part of a trained dense layer.
> (Compare GaLore: plain `W=BA` **collapses** from scratch, 142.53 vs 15.56 ppl at 1B, at *more*
> generous rank fractions than these.) So this is a prior, not a verdict — but an 80-point gap is
> large enough that `grouped` should not be promoted to headline arm on its latency win alone.
>
> **Net effect on the design:** P1 becomes a **parameter-efficiency and quality** claim —
> *"rank-128 gates cost 0.25× the parameters and discard 7% of the energy reaching the output; here is
> what that costs in quality"* — with `grouped` retained as the **systems competitor that wins latency
> but loses representational fidelity**. That tension is a more interesting result than either
> proposal winning outright, and it is measured rather than asserted.

| achievable BW | saved/token | breakeven per-launch cost |
|---|---:|---:|
| 1,300 GB/s (A100 realistic) | 112.9 µs | **5.65 µs** |
| 1,555 GB/s (A100 peak) | 94.4 µs | **4.72 µs** |
| 3,350 GB/s (H100 SXM peak) | 43.8 µs | **2.19 µs** |

**Typical kernel-launch overhead is 5-10 µs — above every one of those breakevens.** At 10 µs/launch
on an A100 the overhead is 200 µs against a 94 µs saving: **P1 makes decode ~106 µs/token *slower*, a
negative result of about the same magnitude as the hoped-for win.** And it gets *worse* on faster
hardware, because the byte saving shrinks with bandwidth while launch cost is fixed.

**So: a batch=1 decode benchmark of P1 without CUDA graphs is not a measurement of the architecture —
it measures dispatch overhead, and it will report the wrong sign.** CUDA graphs (or
`torch.compile(mode="reduce-overhead")`) are **mandatory**, not an optimization.

Two more consequences:
- **Concatenate the gate pairs** into one `d → 2r` and one `2r → d` matmul. This halves the extra
  launches from 20 to 10 and doubles the breakeven to ~9.4 µs. Pure win; do it regardless.
- **Do not attempt the four-stage fusion** (`d→r`, `r→d`, multiply, conv). The intermediate is only
  r=128 elements, so the HBM round-trip you would remove is negligible while the weight-matrix
  traffic — the actual cost — is unavoidable either way. `torch.compile` sends both matmuls to cuBLAS
  extern kernels and cannot fuse into them; fusing the two GEMMs needs a grid-wide barrier (the
  `r→d` stage needs the complete r-vector), which Triton lacks cleanly. CUTLASS B2B GEMM is the right
  tool in principle but buys almost nothing at M=1. Weeks of work to remove a near-zero cost.

**Report the roofline ceiling (4.44% at 350M) alongside the measurement**, so a smaller measured
number is interpretable rather than ambiguous.

**Independent confirmation that the latency case may come out negative.** FLAR-SVD (CVPR 2025 Mobile
AI Workshop) measured FW-SVD/ASVD on **Snapdragon 8 Gen 2 INT8**: params and FLOPs cut 2×, and the
model got **slower than baseline** (8.8 / 9.0 vs 8.0 ms). Their stated finding — that a 128-wide
projection "is not even achieving this inflection point at 10%" — targets **exactly the proposed
r=128**. A CPU microbenchmark in the dossier independently found r ≥ d/2 losing (0.42-0.67×) despite
equal FLOPs.

→ **Make a thin-matmul microbenchmark a pre-training gate.** Before spending any training compute,
measure whether `d→r→d` at the chosen rank is actually faster than `d→d` on the target hardware, with
CUDA graphs on. If it is not, P1 remains a worthwhile *parameter/quality* study but the latency claim
should be dropped from the headline rather than defended.

**Also note the Amdahl ceiling from the other direction:** the gates are only ~8.1% of FLOPs (SwiGLU
MLPs are 68.8% of parameters), which bounds any prefill-side speedup at ~8.8% regardless of how well
the factorization works.

**Rank sweep guidance revised:** {256, 512} are the systems-viable ranks; treat **128 as an aggressive
probe** rather than the default; never go below 64. Parameter savings saturate almost immediately
(r=128 saves 6.27%, r=32 saves 6.94% — 0.67pp for a 4× rank cut), while channel coupling grows as
`sqrt(2/(πr))`. So low ranks buy essentially nothing and cost coupling.

**One more scope argument you must make explicitly.** GaLore reports that a plain `W=BA` from-scratch
baseline **collapses** — 78.18 vs 34.06 ppl at 60M, and **142.53 vs 15.56 at 1B** — at *more generous*
rank fractions than proposed here. P1 escapes this only because it factorizes ~7.2% of parameters
while leaving the value and output paths dense. **State that scope limitation up front**, or a reviewer
who knows GaLore will reject the design on sight.

**The silent-failure mode.** With `W = BA`, `Var(y) = d·r·σ_A²σ_B²`. The stock block uses **Xavier**
(`conv_use_xavier_init: true`), giving a gate output variance of 0.5, so matching requires
`σ_A σ_B = 1/sqrt(2dr)`. Initializing both factors at the usual `0.02` instead gives:

| | r=64 | r=128 | r=256 | r=512 |
|---|---|---|---|---|
| d=1024 | 47.7× too small | 23.8× | 11.9× | 6.0× |
| d=2048 | 23.8× too small | 11.9× | 6.0× | 3.0× |

**Monotone in r.** So a fixed-std sweep produces a smooth, plausible curve that says "higher rank is
better" — the expected answer, so nobody checks. It is an init-scale curve, not a rank curve. This
bites harder than in LoRA because the low-rank output feeds a **multiplicative** path.

*Mitigations (all cheap, all mandatory):* per-arm init calibration; **log gate output variance and
activation RMS at step 0 and assert parity with `L0`**; use **spectral init (Khodak) *plus* Frobenius
decay** — their ablation shows each *alone* is worse than plain low-rank (92.52 / 92.92 vs 93.59) and
only the combination recovers baseline (94.34); set **LR ratio η_B/η_A = d/r** and re-tune base LR per
rank (published corrections disagree even in *direction* — 1.5-2×, 0.05-0.1×, and 0.5× all appear).

**Add a full-width gate bias initialized to 1.0 — highest-value single implementation detail.** It is
the from-scratch analogue of LoRA's `B=0` trick (unavailable here), supplies the per-channel offsets a
rank-r factor cannot express, and matches universal prior practice (Mamba puts its `dt` bias at full
width after the bottleneck for exactly this reason). Measured effect: block-output kurtosis
**27.8 → 4.5**. **The bias is itself a confound — the dense control must get it too**, or you measure
the bias rather than the rank.

#### ⚠️ The motivating premise is falsified — reframe the claim

Singular-value spectra were measured on the **actual released `LiquidAI/LFM2-350M`** weights. Because
`in_proj` is one `[3072,1024]` tensor whose row blocks are B, C, and value, this is a perfectly
controlled within-matrix comparison:

| tensor | effective rank (of 1024) | energy in top 128 | r for 90% energy |
|---|---:|---:|---:|
| conv B_gate | **790.1** | 0.458 | 482 |
| conv C_gate | **770.9** | 0.499 | 459 |
| conv value | 790.5 | 0.456 | 482 |
| *(true rank-128 product, reference)* | *123.8* | *1.000* | *99* |

**Trained gates are ~77% of full rank, and are statistically indistinguishable from the value stream
(790.1 vs 790.5 — a 0.4 difference out of 1024).** A rank-128 truncation discards over half the
spectral energy. Activation-weighting (embedding-covariance proxy) only moves this to ~748.

So the hypothesis "gates are intrinsically lower-rank than the value path, therefore factorize them
preferentially" is **not supported by the trained weights.** Reframe from *"gates are low-rank"* to
**"gates tolerate being low-rank"** — a weaker claim, but the one the literature supports, and one the
experiment can actually test. Note this does not refute the experiment: it measures `W` in the plain
Frobenius metric, not `W·Σ_x^{1/2}`, and the entire activation-aware-SVD literature exists because
those differ enormously (plain SVD at 20% removal destroys LLaMA-7B; activation-aware costs ~2 ppl).
But it does kill the easy version of the motivation, and a reviewer will check this.

##### ✅ RESOLVED (2026-07-31): activation-aware spectra measured. Premise stays falsified.

The gap flagged in §9a is now closed. `Σ_x` was accumulated over **32,768 real tokens** (32× the
hidden size; `rank(Σ_x) = 1024`, confirmed full) through forward hooks on `in_proj`, on the released
checkpoint. Script: [`probes/spectra_v2.py`](../Brainlifts/liv_experiment_research/probes/spectra_v2.py),
raw output `probes/spectra_v2_results.json`. Mean over all 10 LIV layers:

| tensor | plain eff. rank | **activation-aware** | ratio | aware E@128 |
|---|---:|---:|---:|---:|
| B (pre-gate) | 790.1 | **505.9** | 0.640 | 0.926 |
| C (post-gate) | 770.9 | **480.7** | 0.624 | 0.931 |
| x (value stream) — *control* | 790.5 | **507.8** | 0.642 | 0.925 |
| `out_proj` — *different input, control* | 778.5 | 609.2 | 0.783 | 0.787 |
| random Gaussian — *null* | 824.2 | 646.3 | 0.784 | 0.760 |

Two findings, pulling in opposite directions:

1. **The premise is still falsified, now more decisively.** Activation-aware rank does drop ~36%,
   but it drops *identically* for the value stream (gates 493.3 vs value 507.8 — a 2.9% difference).
   All three read the same `x`, hence the same `Σ_x^{1/2}`, so the collapse is a property of the
   **input distribution**, not of gates. The `out_proj` and random-Gaussian controls confirm this:
   they see different inputs / have no structure and collapse *less* (0.784). **Gates are not
   preferentially low-rank under either metric.** Keep the "gates *tolerate* low rank" framing —
   permanently. Do not re-litigate this.
2. **But P1's *feasibility* case is materially stronger.** Rank-128 retains **92.6% of
   activation-weighted energy**, against only 45.8% of plain Frobenius energy. So the truncation P1
   proposes is far less destructive *in the metric that governs output error* than the plain spectrum
   suggested. The honest claim becomes: *"rank-128 gates discard 7% of the energy that actually
   reaches the output, and we test whether that costs quality."* That is a testable, defensible
   framing — and it is **not** "gates are low-rank."

Methodological note worth keeping: a first pass used **568 calibration tokens for a 1024×1024
covariance**, making `Σ_x` rank-deficient by construction (rank ≤ 568) and reporting a spurious
**3.0× collapse to eff. rank 267** — which would have looked like strong support for P1. The
convergence check (8,192 → 16,384 → 32,768 tokens gives 573.5 → 600.2 → 608.0 for L0) shows the
estimate is still rising at 32k, so treat these as a mild **under**estimate. **Any activation-aware
spectral claim needs tokens ≫ d and a reported `rank(Σ_x)`, or it is an artifact.**

Reporting note: **use effective rank and energy-at-r, not stable rank.** Stable rank reads 26-48 for
these matrices because σ₁ dominates it — and a *random* Gaussian baseline scores 258, i.e. *higher*
than every trained matrix. Report `srank` only with the random baseline alongside.

**The LIV gates are LINEAR — and this makes the risk worse, not better.** There is no sigmoid,
SiLU, or softplus on `B` or `C`; they are raw projections multiplied elementwise. Every gate in the
SSM/RNN literature passes through a bounded or positive nonlinearity. So the usual *gate saturation*
failure mode does not apply here, but an **unbounded variance-amplification** mode does — a
mis-scaled linear gate multiplies straight into the conv input with nothing to clamp it. This is
precisely why the init calibration above is load-bearing rather than cosmetic.

**And there is no bias to absorb a scale error.** `conv_bias: false`, so `in_proj`/`out_proj` have no
bias at all. Contrast Mamba, which puts its `dt` bias at **full output width after the bottleneck** —
deliberately, because a rank-r factor with r ≪ d *cannot* produce d independent channel offsets, so
all per-channel gate diversity at init comes from that bias. A factorized LIV gate has neither the
bias nor the rank to provide channel diversity.
→ **Design decision to make explicitly:** add a full-width bias after the bottleneck (Mamba's
pattern, citable) or accept reduced channel diversity. Recommend adding it, and adding it to the
full-rank control too so the arms stay comparable.

**Prior expectation: mixed, and weaker than I first assumed.** Mamba's `dt` projection and GLA's
forget gate are both low-rank in production, and Zoology attributes gated-conv recall limits to
filter *input-dependence*, which factorization does not touch — so MQAR should be safe. But the one
real rank sweep of a gate projection in this literature (Mamba Table 9, ~350M, Chinchilla-optimal)
is **monotone improving all the way to the top of its range**: rank 1 → 64 buys 0.26 ppl, with no
plateau. That is a caution: gate rank was not a free parameter there. Note it is not a clean analogue
(Mamba's Δ gate has a full-width bias and a nonlinearity; ranks are far smaller relative to width),
but it argues for sweeping rank up to **r=512** rather than stopping at 128, and for treating a flat
curve as a finding to verify rather than assume.

Zoology also shows gated convs are *already* the width-starved component, so a recall probe is
mandatory, not optional: perplexity hid 82% of the recall gap in their setting.

### 5.2 P2 — KV sharing: a capacity result, and predict the latency null

Honest accounting:

| quantity | effect |
|---|---|
| resident KV bytes | **−50%** (real, clean, measurable) |
| KV write traffic | −50% (writes ≪ reads in decode) |
| KV read traffic | **0%** |
| end-to-end decode latency | **≈0%**, any context |

**Predict the null in advance.** Otherwise a correctly-predicted null reads as a failed experiment.

**Evidence is genuinely encouraging on quality.** CLA2 at 1B/30B and 3B/100B: 2× cache reduction
for **0.04-0.05 ppl**, sometimes an improvement; at equal cache budget it beats halving head dim by
0.31-0.35 ppl. And Character.AI ran the non-adjacent global-layer variant in production
(~1-in-6 global attention, >20× total KV reduction, no needle regression).

**Two constraints from CLA's own ablations.** (a) Non-uniform pairing lost; their `DenseBack`
variant was **+0.43 ppl worse**. Reconciliation with Character.AI: `DenseBack` forced *many*
consumers onto *one* early producer, whereas pairing two nearby global layers is much milder. Still
— make inter-pair distance the primary axis and expect degradation to grow with it. (b) CLA pairs
best with **MQA, not GQA**; GQA+CLA2 mostly *lost* to equal-footprint baselines. We start from
CLA's weaker partner.

**But we cannot simply follow CLA's MQA recipe, because we train from scratch.** GQA's own Appendix A
(arXiv 2305.13245) reports that **MQA trained from scratch had "frequent loss spikes" and diverged on
long inputs** — CLA's MQA results are *uptrained* from an existing checkpoint, not from scratch. We are
in exactly the regime GQA warns about. **So keep GQA-8 as the primary configuration and treat MQA as a
monitored secondary arm**, with loss-spike detection on. This is a real divergence from CLA's
recommended recipe and should be stated as such rather than glossed.

**The pairing question is a live disagreement in the literature — exploit that.** Three sources
contradict each other: CLA's `DenseBack` says non-adjacent loses (+0.43 ppl); Character.AI and Gemma 3n
both deploy non-adjacent sharing and report it fine, with **no published ablation either way**. Our
`C-near` vs `C-far` arms are a controlled test of that disagreement, which is a cleaner framing than
"we hope non-adjacent works."

**And the competition is stronger than the proposal.** Stated plainly, because a reviewer will: MLA
reaches ~**3,456 B/token** vs CLA2's 6,144 (and DeepSeek-V2 claims MLA beats MHA outright), Gemma-3
style 5:1 SWA cuts more for "minimal" perplexity cost, and **simply using 3 attention layers instead of
6 matches CLA2's resident capacity while *also* halving read bandwidth** — which CLA structurally
cannot do. **CLA2's defensible niche is narrow and should be named exactly: it is the only method that
halves resident KV without changing any layer's receptive field.** That is the whole pitch.

Note also there are exactly **15** possible pairings of `[2,5,8,10,12,14]`, and
`(2,5)(8,10)(12,14)` is the unique minimum-total-distance one — so the arm choice is principled rather
than arbitrary.

**Hymba is prior art — narrow the claim.** `nvidia/Hymba-1.5B-Base`'s config contains an explicit
`kv_reuse_group = [[1,2],[3,4],...,[16,17,18],...]` with `kv_weight_reuse: false`, and its
`modeling_hymba.py` is a working CLA-in-a-hybrid reference: consumers construct **no** `k_proj`/
`v_proj` at all, project only Q, skip all cache writes, and **consume post-rotary K** (which
independently confirms the RoPE decision above — Hymba applies RoPE to K in the producer and only to
Q in the consumer). Two features of its design cut *against* the proposal:

- **Pairing is strictly adjacent**, matching CLA's recommendation — which LFM2's schedule cannot do.
- **Its 3 global-attention layers `[0,15,31]` are excluded from every sharing group.** Hymba shares
  only between *sliding-window* layers. **Our proposal would share between full-attention layers —
  the opposite of what Hymba validated.** The config proves the choice; the paper gives no ablation
  isolating "share the global layers too," so the reason is unstated.

**And Hymba's own ablation points the wrong way on recall.** Row C → D of its roadmap table
(300M/100B tokens), adding KV sharing: commonsense **+0.60** (44.56→45.16), throughput **+14.9%** —
but **recall −0.75** (48.79→48.04). Weak evidence (2-task average, small scale) and the paper frames
it as "maintaining comparable recall," but the split is *exactly* the failure mode the brainlift
worries about: aggregate metric up, retrieval down, unfollowed-up.

Also sobering: Hymba's cache only fell 41.2→39.4 MB (**4.4%**) from sharing, because sliding-window
attention had *already* cut it 3.8×. **SWA and CLA are substantially substitutes on the capacity
axis, not complements.** In our design the attention layers are full attention, so there is more
cache to remove — which is the one structural respect in which our setting is *more* favorable.

**So the honest framing is:**

> Hymba established that cross-layer KV sharing works in a *parallel hybrid-head* architecture
> between *adjacent sliding-window* layers. We ask the untested questions: does it survive
> (a) sharing across an intervening sequence-mixing block in a *sequentially interleaved* hybrid,
> (b) between *full-attention* rather than local layers, and (c) under *exact-retrieval* evaluation
> rather than aggregate perplexity — where Hymba's own ablation shows a recall regression it did not
> investigate.

Each sub-question is genuinely open: no paper studies cross-layer sharing with an intervening
conv/SSM block in a sequentially interleaved stack, and none searches or learns *which* attention
layers to pair. Do **not** claim "first to combine CLA with a hybrid" — a reviewer who knows Hymba
would reject that outright.

**A second, cheap contribution: measure the capacity-vs-bandwidth split that CLA only asserted.**
The CLA authors state analytically that shared banks "must be separately re-read from main memory in
each attention layer" but never measured it. With Nsight Compute you can show directly that sharing
**halves `dram__bytes_write.sum` while leaving `dram__bytes_read.sum` flat**. Verified recipe:

```bash
ncu --nvtx --nvtx-include "kv_read/" \
    --replay-mode application --cache-control none --clock-control none \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_hit_rate.pct,gpu__time_duration.sum \
    --csv --print-summary per-nvtx python decode_bench.py
```

Four traps that would invalidate this measurement:
- **`--cache-control` defaults to `all`**, which flushes all GPU caches before every replay iteration
  — giving *cold-cache* traffic that **overstates DRAM reads** and could fake or erase an L2-reuse
  effect. Use `--replay-mode application --cache-control none` for traffic; lock clocks externally.
- **Collect hit rates and byte counts in separate runs.** With `--cache-control none` and multi-pass
  collection, ratio metrics can be wrong when numerator and denominator land in different passes.
- **Prefer `dram__bytes_read.sum + dram__bytes_write.sum` over `dram__bytes.sum`** — NVIDIA's own
  shipped section file does not gate the latter for all recent compute capabilities. And do not
  hard-code a 32 B sector size for HBM; if you use sector counts, divide bytes by sectors empirically.
- **Never read latency from an ncu run** (it serializes, cross-process, via per-device lock files),
  and note the Memory Chart shows instructions/requests, **not bytes** — bytes live only in the tables
  (`--metrics group:memory__dram_table`). Don't screenshot the chart and call it traffic.

Label the result **capacity + write traffic**, never "KV cache traffic."

**Report five memory buckets separately — a single "peak memory" number is not a result.** The three
proposals affect different buckets, so conflating them makes the memory claim uninterpretable:

| bucket | scaling | which proposal touches it |
|---|---|---|
| **Weight bytes** | fixed (`params × bytes/param`) | **P1 reduces** |
| **KV cache bytes** | `batch × context` | **P2 reduces — and only this** |
| **Conv state bytes** | `batch`, not context | **P3 increases** (7×) |
| Activation peak | `batch × context`, transient | none directly |
| Allocator overhead | not a model property | none |

Use `max_memory_allocated()` (live-tensor bytes) as the reported figure and `max_memory_reserved()`
(caching-allocator holdings) only as an operational footnote — the `reserved − allocated` gap is
fragmentation, not architecture. Cross-check every bucket against its analytic value; a mismatch means
the measurement is wrong, not that the model is surprising.

**The strongest remaining contribution is the evaluation gap, and it is now confirmed exhaustively.**
**Not one cross-layer-sharing paper reports needle, passkey, or MQAR** — checked across CLA, the NAACL
2025 systematic study, LCKV, and SwiftKV. CLA says so outright: *"we leave end-to-end inference
efficiency evaluations of large, long-context models employing CLA as an interesting problem for future
work."*

Two data points suggest something real is hiding there, and **both point the wrong way**:
- Hymba's row C→D: commonsense **+0.60**, throughput **+14.9%**, **recall −0.75** — the only metric that
  fell, measured in the configuration *least* likely to show damage (sharing among SWA layers, global
  layers protected).
- MLKV shows the same dissociation at 1 KV head: LAMBADA collapses **33.63 → 8.56** while PIQA barely
  moves.

And Character.AI's needle claim, read carefully, attaches to their **sliding window**, not their
sharing — so it is not the counter-evidence it appears to be.

This is cheap for us, directly targets the brainlift's staleness worry, and is the one question where
the existing literature is uniformly silent while the two available numbers both hint at a real cost.
**Make it the headline of the P2 arm.**

*Feasibility note:* serving support now exists (vLLM `kv_sharing_target_layer_name`; llama.cpp has a
real `il → cache index` indirection map whose `reuse` callback would accept a CLA pattern as a
one-line lambda). HF transformers supports it only for Gemma via a side-channel, not the generic
`Cache` API.

### 5.3 P3 — multiscale: the published evidence is close to fatal, but there is a better experiment

**The decisive table.** Sieberling et al. 2026 (*Dynamic Short Convolutions Improve Transformers*,
arXiv 2606.03825) — the paper the brainlift cites — ran the width sweep on exactly this primitive,
300M/15B tokens, with an *input-dependent* filter (the most expressive case):

| W | 1 | 2 | **3** | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| PPL | 18.42 | 18.17 | **18.08** | 18.10 | 18.09 | 18.10 |

W=2→3 gains 0.09; **W=3→6 is −0.02, i.e. slightly worse.** Their conclusion: "3 or 4 is generally
the sweet spot." The proposal's spans of 9 and 15 sit far right of a curve already flat at 3. Their
generator also uses **no softmax** over taps — free real-valued taps — unlike the proposed router.

**Three structural problems:**

0a. **⚠️ IMPORTANT CAVEAT on the adverse evidence — it was collected in a DIFFERENT structural slot.**
   Neither adverse LM paper used Liquid's double gate. **Tian et al.** add a *residual* depthwise conv
   post-QKV (`X ← X + Conv(X)`), **ungated**. **Sieberling et al.** apply theirs *residually* to Q/K/V
   before RoPE, also with **no multiplicative gate** (their filters are affine, no softmax). SKNet and
   Chen et al. are 2-D vision. **In LFM2 the conv sits *inside* two input-dependent gates**
   (`C ⊙ conv(B ⊙ x)` → `out_proj`), so branch weights and gate values interact **multiplicatively** —
   a different optimization landscape than any of them searched. **The negative results may not
   transfer, and this is a legitimate reason the experiment is not redundant.** State it this way in
   the write-up rather than overclaiming the adverse evidence.

   Three counter-considerations, so it is weighed fairly: (i) the *span* finding is about where
   information lives in language, which a gate does not change — a gate cannot create signal at lag 12;
   (ii) Sieberling's filters were **already input-dependent**, i.e. the most favorable case, and extra
   span still bought nothing; (iii) their rank-vs-span split is the tell — generator rank 16→128 bought
   0.25 ppl *and was still climbing*, while span 3→6 bought **0.00**, implying gains come from
   **conditioning capacity, not receptive field**. The router adds a little of the former and a lot of
   the latter. Net: nudge the prior on the full router from ~15% to ~20-25%, no higher.

   **This makes the cheap arms more important, not less.** `k5/k9/k15` **inside the real gated LIV
   block** is now doing real scientific work — testing whether the published negative result survives
   the gate. If a plain wider kernel does not help there, the gate is not rescuing span and the router
   almost certainly will not either. If it does help, that disconfirms Sieberling and the router earns
   its build.

0b. **The exact mechanism was tested in an LM and it LOST** — subject to 0a. *"Convolution for Large Language Models"*
   (arXiv 2607.18413, PKU/Huawei/Tsinghua) added a residual depthwise Conv1D to Qwen3-1.7B and swept
   width: no-conv 13.42 → k=2 12.99 → **k=3 12.79** → k=4 13.13. They then tried **multi-branch mixed
   kernel sizes: 12.79 → 13.28**, i.e. *worse than a single k=3.* **That is this proposal's core
   mechanism, in a language model, and it regressed.** Combined with Sieberling's independent flat
   curve, two labs on different base models both peak at **k=3**. This is the single most damaging
   finding for P3 and it must be cited up front.

1. **The fixed-weight version is vacuous — and the dilation pattern is worse than I stated.** A mixture
   of four dilated 3-tap causal kernels with input-independent weights **is exactly a single sparse
   15-tap causal kernel** (verified numerically to ~1e-15). But dilations {1,2,4,7} reach only **7 of
   the 15 lags** — 8 lags are *structurally unlearnable* and lag 0 is covered 4× redundantly. So 12
   parameters buy **7 degrees of freedom**, versus 15 DOF for 15 parameters in a dense kernel. The
   proposal is a **rank-4, simplex-constrained special case of dynamic short convolution.** `k15` is
   mandatory or the claim is empty.

2. ~~No fast kernel exists.~~ **CORRECTION: a fused wide kernel IS buildable today.** `causal-conv1d`
   hard-aborts above width 4, but **FLA's Triton conv backend has no width limit.** So a fused dense
   k=15 is available, and the mandatory wide-kernel baseline is **neither tooling-handicapped nor
   unfair** — which removes P3's best excuse for losing to it. (Dilation is still unsupported
   everywhere, so the dilated variant remains the disadvantaged one.)
3. **State grows 5× and dilation buys nothing there.** State is set by *max lag*, not tap count:
   dilations {1,2,4,7} reach lag 14, so the ring buffer must hold 15 vectors against the stock 3.
   At d=1024 that is **60 KiB → 300 KiB** across the 10 LIV layers (bf16) — a **5×** increase, and the
   "cheap" 12-tap dilated version has **exactly the same state** as the dense 15-tap version. Dilation
   saves parameters and FLOPs, never state. *(Corrected: an earlier draft said 7×, using the wrong
   cache convention — LFM2's conv cache is `[B, d, L_cache]` with `L_cache = k`, so the ratio is
   15/3 = 5.)*

   The state is still negligible against the KV cache (0.02% → 0.08% of 384 MiB at 32K), so "bounded
   tiny state" survives — **but note it cuts against LFM2's embedded-CPU design target**, where the
   whole point is a small fixed working set. The real cost remains bandwidth: 4 branches multiply
   traffic ~4× on a memory-bound op.

Also: a softmax router **saves no compute** unless the implementation genuinely skips branches, and
hard top-k routing over four tiny depthwise convs will lose to dense evaluation on GPU. RepVGG
measured a **41% throughput loss** from multi-branch fragmentation in an analogous setting.

**The better experiment (RepVGG-style reparameterization).** Because the branches are *linear*, a
fixed-weight 4-branch block **fuses losslessly into one 15-tap depthwise kernel at inference**.
RepVGG's finding is that the training-time topology can matter even when the inference-time function
is identical. That yields a real, cheap, novel question the proposal does not currently contain:

> Does the 4-branch *training-time* parameterization of a 15-tap kernel optimize better than a
> directly-trained 15-tap kernel, given they are the same function class?

Free at inference, no kernel work, and it is an optimization question rather than a capacity claim.
**A token-dependent router destroys fusibility and locks in the fragmentation penalty permanently**
— which is the strongest argument for dropping the router specifically.

#### ⚠️ A recipe hazard that can manufacture a false negative

**A plain 4-way softmax router over kernels scored *below* static mixing** — 64.8% vs 65.4% (Chen et
al., K=4). It was rescued only by **temperature annealing τ=30→1**, because a near-one-hot router
*"only allows a small subset of kernels to be optimized."* If we run the naive router and it loses, we
will not know whether the *architecture* failed or the *recipe* did.

**Therefore, mandatory for any router arm:** temperature annealing schedule (declared in advance),
**fp32 router logits**, and router z-loss. Without these the negative result is uninterpretable.

**Two more calibrations worth knowing:**
- **Every branch-count sweep in the literature saturates at two.** SKNet (the closest image analogue)
  goes M=2 → M=3 for **+0.03%** at +1.8M params and concludes *"M = 2 is preferred"*; MixConv's knee is
  g=2; Res2Net's is s=2. Three methodologies, one answer. **Add a 2-branch rung at half the cost** —
  if 4 branches are going to lose, 2 is where any real effect would show first.
- **The likely failure mode has direct precedent.** SE blocks saturate to where *"an SE block reduces
  to the identity operator"* and can be removed for <0.1% top-5; Branchformer's learned input-dependent
  branch weights had **std < 0.01** and **lost to static concatenation**. Expect the router to collapse
  toward uniform — and note that a collapsed router is a *finding*, provided the annealing recipe above
  rules out the optimization explanation.

**Stated fairly, the evidence *for* input-dependence:** in the kernel-mixture setting it *is*
load-bearing — replacing Chen et al.'s router with plain kernel averaging collapses top-1
**69.4% → 36.0%**, SKNet's router beat a uniform sum by **+0.97**, and single large kernels are much
worse than k=3 alone (K5 25.14, K7 25.51 vs K3 22.23) while the *mixture* beats both. So "wide kernels
only help inside a mixture" has real support. Caveat: those are inference-time ablations in 2-D vision.

**Prior probability that the full router beats a parameter-matched dense k=15 across 3 seeds: ~15%.**
The cost structure is inverted — the op is memory-bound (1.5-7.5 FLOP/byte against an H100 ridge of
~296), so widening *one* fused kernel is nearly free while 4 branches multiply traffic ~4×.

**Recommendation, revised — reframe rather than drop.** Two days of profiling changed the verdict here.
The conv is **1% of decode** and dilation is measurably **~free** (§7.1), so P3 cannot be a speed claim —
but that same fact makes *"multi-scale receptive field at negligible latency cost"* a **true, measurable,
defensible** result. That is a better paper sentence than a speed claim, because it survives review.

Sequence:
1. **Week-1 de-risk (~2 days, do this before anything else):** export one 4-branch dilated depthwise
   conv and profile it in isolation against the 3-tap baseline on the target. Isolated numbers are
   already known (155.1 vs 27.5 µs, 5.6× on the conv → 0.3-1.6% total decode regression), so this is
   confirmation, not discovery — and it tells you the architectural claim's cost up front.
2. **Run `k5/k9/k15`** — cheap, and the published width curve says it likely settles the question.
3. Only if a wider span actually gains, proceed to the **reparameterization arm** (train 4-branch, fuse
   to one 15-tap kernel at inference — free, and the only place the branch structure could add value).
4. Treat the router as the last resort, and require branch-knockout evidence (§6.3) before crediting it.
   Note the router is also what destroys fusibility, so it carries the fragmentation cost permanently.

**Frame the deliverable as: does a wider/multi-scale span buy quality, given we can now show it costs
essentially nothing?** That question is answerable, honest, and interesting regardless of sign.

---

## 6. Endpoints, statistics, and what will actually detect an effect

### 6.0 The central methodological risk: the ratio basin is flat, and perplexity cannot see the gap

This is the most important single fact for planning, and it independently confirms §6.1's power
concern from a different direction.

**The published ratio basin is flat to within noise.** Mamba-2's Table 2 sweep (350M, 48 layers, 7B
tokens) spans 2-11 attention blocks (4-23% attention) and every point lies within **0.06 ppl** —
against Samba's reported run-to-run noise of **±0.3%**. So the perplexity differences the experiment
would be chasing across ratio/gate/pairing arms are *at or below the noise floor.*

**Meanwhile the same architectures differ enormously on capability metrics.** Hymba measured a
**20.75-point recall gap** and Jamba a **35.3-point format-following gap** at near-identical
perplexity.

**Conclusion: rank arms on recall, not perplexity, with multiple seeds — and pre-register that.** Every
sharper endpoint in §6.1-6.4 exists because of this. It also means a flat perplexity result across
arms is the *expected* outcome and must not be reported as "no difference."

Useful context for positioning: stock LFM2 is a **37.5%-attention** outlier (6 of 16 layers,
non-uniformly spaced with gaps 3,3,3,2,2,2 — *late-heavy*), whereas every SSM-hybrid that published a
ratio ablation landed at **7-25%**: Mamba-2's optimum was 6/48 (ppl 8.26, vs 8.60 at zero attention
and 8.68 for Transformer++), Waleffe et al. found ~8% independently at both 130M and 840M, Falcon-H1's
21-configuration channel sweep landed on 1/8, and Jamba found 1:3 and 1:7 indistinguishable. **Nobody
has published a ratio ablation on a *short-conv* mixer** — which is the gap, and it is a more specific
and defensible claim than "mostly-LIV works."

Also worth stealing from Waleffe: **30-50% MLP is free, and 50% trains ~20% faster.** A cheap
throughput lever that does not touch the scientific question.

**One pre-registered risk to state up front:** Samba reports that adding short convolution to *both*
mixer types "produces negative results." A mostly-LIV stack is short-conv-everywhere by construction,
so this is a known adverse finding pointing at our architecture. Name it in advance rather than
discovering it.

### 6.1 Held-out CE is almost certainly underpowered — do not make it the primary endpoint

From this repo's own completed KDA study (`KDA/lm/train_lm.py`): the DeltaProduct paper's one
strictly parameter-matched LM comparison is **+0.0053 nats, needing ~n=43 seeds**. Measured locally,
seed-to-seed variance at ~1M params **swamped a 4× change in task difficulty** (one-way ANOVA
η²=5.9%, F(3,16)=0.337, ns).

The existing protocol's gate is CE non-inferiority at **+0.010 nats**. With paired seeds and
`n = ceil(((1.645+0.842)·s_δ/m)²)`, that margin is reachable only if `s_δ ≲ 0.011` at n≥8.
**Action: measure `s_δ` in the pilot and publish the required n before committing to the gate.** If
it is out of reach, say "inconclusive" rather than "non-inferior" — the protocol already mandates this.

**Primary endpoints should be large-effect:**
1. **Recall composite** (MQAR hard, overwrite/conflict binding, long-gap retrieval, local delayed
   composition), scored as success rate over seeds (§6.2).
2. **Length extrapolation** (train 4K, eval 8K/16K/32K) — the endpoint the prior study adopted
   precisely because CE lacked power, and where effects were tens of points rather than 0.005 nats.
3. **AR-Hits sliced perplexity** — see below. This is the highest value-per-GPU-hour item in the
   whole plan.
4. **Component-level state accounting** (weights / KV / conv state / allocator, separately).

**Adopt Zoology's AR-Hits decomposition as a Tier-1 diagnostic.** Rather than reporting one
aggregate CE, split held-out tokens into:

- **AR Hits** — the final token of a bigram that already appeared *in the same context* and appeared
  **≤1250×** in training data (**6.4%** of Pile tokens)
- **Other** — everything else (93.6%)

Zoology's result is the entire justification for this design: gated convolutions trail attention by
up to 2.1 ppl, **82% of that gap comes from the 6.4% AR slice, and on "other" tokens there is
essentially no gap at all.** Average perplexity hides the retrieval deficit almost perfectly. A 70M
attention model beat a 1.4B Hyena on AR-slice perplexity.

Why this matters here specifically: it converts the underpowered endpoint into a powerful one *at
zero extra training cost*. It is a re-weighted pass over held-out text with a bigram-repeat mask plus
a training-frequency table — no new runs. If P1's low-rank gates or P2's shared K/V damage retrieval,
this is where it shows up first, and it will show up as a large effect on a small token slice rather
than a 0.005-nat shift on everything.

**Report the gap attribution explicitly:**
`% of gap due to AR = [Δlog(φ_AR)·|T_AR|] / [Δlog(φ)·|T|]`

### 6.1a Use BABILong, not RULER, as the long-context benchmark

**RULER is the wrong primary metric at this scale.** Mamba-2.8B and RWKV-v5 already fall below its
4K floor, and the smallest leaderboard entry is EXAONE-4.0-1.2B (87.0 @4K against an 85.6 threshold).
A 350M model will floor out, and a floored metric cannot rank arms.

**BABILong measured the discriminability question directly: it separates models from 2K tokens,
where RULER needs ≥128K.** It also has published results at **130M-137M** (Mamba-130M, RMT/ARMT on
GPT-2-137M) — i.e. at our scale, which RULER does not. Use BABILong as the primary long-context
benchmark and RULER-short only as a secondary sanity check.

Tooling is better than expected: **RULER (13 tasks), BABILong (20 tasks), Paloma, IFEval, LongBench,
and `squad_completion` are all native lm-eval-harness tasks (MIT).** Harness RULER defaults to 4096
with lengths settable via `--metadata='{"max_seq_lengths":[...]}'`, so RULER-short is one command on
plain HF models — no vLLM needed. This removes most of the "must be built" eval burden from the audit.

### 6.1b Phonebook is the direct test — and Griffin is the precedent to beat

A full-text check across the hybrid literature found **phonebook is used only by Griffin** (citing
Jelassi et al.); Jamba uses NIAH, Zamba2 uses passkey, Zamba uses neither, Liquid's own blog uses
RULER. So **Griffin/RecurrentGemma, not Jamba/Hymba/Zamba, is the direct precedent.**

The decisive fact: **Griffin solves phonebook up to its 1024-token local-attention window and no
further.** That makes a *phonebook length sweep past the conv receptive field* the single cleanest
test of whether full GQA layers remove the ceiling that a purely local mechanism imposes. Make this a
primary endpoint — it targets exactly the LIV-vs-attention division of labor the brainlift argues for,
and there is a published curve to compare against.

### 6.1c A caution on framing: "a few attention layers fix retrieval" is already a theorem

Three independent results establish it: Wen et al. Thm 5.6 (*"adding one Transformer layer… is
sufficient to close the representation gap"*), Zoology's finding that <10% of layers being attention
closes >80% of the MQAR gap, and Based recovering 90.8% with a 64-128 token window.

**So "mostly-LIV plus a few attention layers works" is not a publishable claim — it is known.** The
contribution must be a *quantitative* placement/budget claim (how many, where, at what cost), and the
evaluation must be sharp enough to resolve the **residual** gap after attention is added. This is a
direct argument for the sharper endpoints above, and against reporting aggregate perplexity.

### 6.2 MQAR: use success rate, not mean accuracy

`KDA/run_mqar_var.sbatch` documents that MQAR trainability at ~1M params is **bimodal** — a run
either finds the recall algorithm or stays at chance. An n=1 ladder was non-monotone in load
(64.5% → 0.88% → 97.8% → 40.3%). **So the endpoint is success rate over seeds**, and difficulty must
be calibrated on pilot data *before* looking at the treatment arms — easy MQAR at ceiling has no
evidentiary value.

Reusable: `probes/mqar_patch.py` already decouples the two difficulty axes (pairs D = capacity,
capped via `MQAR_MAX_PAIRS`; filler = distance).

**The exact generator, from `zoology/data/multiquery_ar.py`:** disjoint key/value vocab halves, D
pairs in the first 2D positions, power-law gaps (`power_a=0.01`; 1.0 = uniform), labels `-100` except
at query positions. Reproduction grid: vocab 8192, (N,D) ∈ {(64,4),(128,8),(256,16),(512,64)}, d_model
{64,128,256,512}, 2 layers, 1 head, 100k train / 3k test, 64 epochs, LR `logspace(-4,-2,4)`.

**Two reproduction gotchas** that will silently change results: the paper's configs set
`random_non_queries=False` while the *class default is `True`*, and they use `state_mixer=Identity`,
which contradicts the paper's own Appendix E.2. Pin both explicitly.

**State the theory precisely — it is easy to misquote.** Gated *convolutions* need **d ≳ N** (sequence
length, i.e. number of distinct interaction distances); it is **D (number of KV pairs)** that drives
the requirement for gated *recurrences*. Zoology's Thm 4.4 is an *upper* bound (Õ(N log c) params),
not a width lower bound; the clean lower bounds are Based Thm 3.1 (any causal recurrent model needs
**Ω(N) bits** of state) and depth Ω(max(log log c, log log N)).

Note our d=1024 against a 4K training context sits on the wrong side of the `d ≳ N` line — worth
measuring directly, since it predicts where the LIV layers should fail without attention's help.

### 6.3 Attribution: router probabilities are not evidence

The brainlift is right that router weights don't explain behavior (cf. "attention is not
explanation", Jain & Wallace 1902.10186). Required protocol for any P3 claim:
**branch knockout at inference**, **knockout with retraining** (distinguishes "used" from
"necessary"), and comparison against equal-weight mixing, static-learned-weight mixing, and
parameter-matched single-span. A branch with small router weight can still dominate if its output
magnitude is large.

**Use resample-or-mean ablation, not zeroing.** Zeroing a branch pushes activations off-distribution,
so the measured Δ conflates "this branch mattered" with "the model has never seen this input
distribution." Replace the branch output with its dataset mean (or a resampled value from another
token) instead. This is standard in the causal-mediation literature and it is the difference between
an attribution and an artifact.

### 6.4 Statistical discipline (inherit from the existing protocol)

Seed is the unit of inference; pair arms by seed; report every run including failures. From the
prior study's hard-won lessons: **n=3 is not enough for small effects** (a "+8.92pp" result
collapsed to "+2.01pp ns" at n=8; sign-consistency across 3 seeds is p=1/8, not evidence), and
**never compare single runs across backends** (individual runs differed ~4pp due to basin
selection). Screening at 5 paired seeds selects one configuration; confirmation uses ≥8 *fresh*
paired seeds never used in selection.

**Per-benchmark seed counts (Madaan et al., measured 7B seed σ), for δ=2pt detection:**

| Benchmark | seed σ | seeds for δ=2pt | verdict |
|---|---:|---:|---|
| HellaSwag | 0.21 | **1** | keep |
| ARC-Challenge | 0.80 | **3** | keep |
| COPA | 2.15 | **19** | **drop — unaffordable** |
| WinoGrande | — | — | **drop below 1B** (51.3 at 135M = chance) |
| MMLU (MC form) | — | — | **drop** (at chance, monotonicity 0.09) |

Two cheap wins from that literature:
- **Use continuous metrics (per-token likelihood) rather than accuracy** wherever possible: a
  **2-18× SNR gain**, which is far cheaper than buying the equivalent in extra seeds.
- **MMLU works as cloze but not multiple-choice** at this scale (37.47 accuracy / 0.95 monotonicity
  as cloze vs at-chance / 0.09 as MC). If MMLU is wanted, use the cloze formulation.

This turns "how many seeds?" from a guess into a per-endpoint calculation, and it prunes three
benchmarks that would otherwise burn budget producing noise.

**Instruction persistence: no standard eval exists.** Section 5 of the eval dossier confirms this
after a full search. If the brainlift's "instruction persistence" endpoint is kept, it needs a
purpose-built synthetic: place an instruction at distance *d* from the query, measure a normalized
persistence half-life `d₅₀`, with **mandatory `d=0` and no-instruction controls** (without them a
"persistence" result is indistinguishable from a model that never followed the instruction at all).

---

## 7. Scope: measure on GPU, drop edge and energy

The brainlift's §5 question 3 asks for CPU/edge latency and energy.

> **⚠️ CORRECTION.** An earlier version of this section justified GPU-only by claiming edge runtimes
> cannot run the variants. **That is false and checkable.** Someone exported the 4-branch dilated conv
> end-to-end and **all 4 convs delegated to XNNPACK with zero CPU fallback**; ORT CPU and Core ML EP
> both accept it; MLX supports `groups` + `dilation` natively; and dilation is measurably **~free**
> (prefill 332.1 / 333.6 / 375.4 / 303.3 µs for dilations 1/2/4/7 — noise). "We couldn't run it on
> edge" would not survive review. **Do not use that argument.**

### 7.1 The real blocker is Amdahl's law — and it is the most valuable artifact in the dossier

Per-op profile of the real `onnx-community/LFM2.5-350M-ONNX` q4, decode, ORT CPU, 4 threads, 128-token
past:

| op | µs/step | share |
|---|---:|---:|
| `MatMulNBits` | 42,389.3 | **91.2%** |
| `SimplifiedLayerNormalization` | 704.2 | 1.5% |
| `GroupQueryAttention` | 682.4 | 1.5% |
| **`Conv`** (the LIV depthwise) | **452.6** | **1.0%** |
| total | 46,461.0 | |

**The short conv is 1% of decode. Matmul is 91%.** This independently corroborates the parameter
analysis: the depthwise kernel is only `k·d` = 3,072 weights, **0.07% of the LIV mixer**, while the
projections are 99.9%. Two different methods, same conclusion.

**This settles the ranking of the three proposals on latency, decisively:**

| | attacks | measured effect |
|---|---|---|
| **P1** low-rank gates | the **91%** (`d→d` becomes `d→r→d` in `in_proj`, measured 257.7 µs in-context) | the only positive latency story |
| **P2** KV sharing | bytes — which is what a memory-bound decode responds to | capacity; ~0 latency |
| **P3** multiscale conv | the **1%**, and makes it **5.6× worse** (155.1 vs 27.5 µs isolated) | **−0.3 to −1.6% total decode regression** |

### 7.2 What to claim — and one genuinely better framing for P3

1. **P1 is the latency story.** It touches the 91%.
2. **P2 is the memory story**, and in a memory-bound decode the most likely honest speedup. Cheapest to
   implement — cross-layer KV donation already exists in **two** runtimes (ExecuTorch's
   `ForwardOptions.shared_kv` / `_prepare_qkv_shared`, and llama.cpp's `layer_reuse_cb`).
3. **P3 is a quality story, not a speed story — and this is an upgrade, not a concession.**
   *"Multi-scale receptive field at negligible latency cost (measured +0.3-1.6% decode)"* is a
   **strong, defensible, true** result. It is better than a speed claim because it survives scrutiny.
4. **Always report the per-op breakdown**, not just end-to-end tok/s. The 1%-vs-91% table is the single
   most useful artifact produced by this research.
5. **Claim no NPU or ANE wins.** iOS ANE fails outright for LFM2 today (`ANECCompile FAILED`,
   executorch#19635, maintainer-confirmed CoreML bug), and on Snapdragon, NPU decode is *slower* than
   CPU (Granite-4.0 hybrid: 11.3 vs 17.8 tok/s).
6. **Never call MLX numbers "edge"** — that is Apple unified-memory GPU.

### 7.3 Revised scope recommendation

**Still GPU-primary, but justify it by *methodology*, not by false impossibility.** GPU is where locked
clocks, CUDA graphs, and ncu counters make a ~4% effect attributable at kernel level; on CPU/mobile you
cannot lock clocks and run-to-run variance routinely exceeds 10%, which is larger than the effect.

Two cheap additions that are now clearly worth it:

- **Week-1 de-risk (~2 days):** export a single 4-branch dilated depthwise conv and profile it in
  isolation against the 3-tap baseline. You get the ratio immediately and learn whether P3's
  architectural claim survives *before* committing weeks. Do this first.
- **One real edge datapoint:** the ONNX path already runs — measured **24.83 ms/token = 40.3 tok/s**
  decode, prefill 32 tokens in 480.1 ms. `onnxruntime-genai` has first-class LFM2 support
  (`builders/lfm2.py`, `src/models/lfm2.{h,cpp}`, and `struct LFM2Cache : KeyValueCache` combining KV
  cache with fixed-size conv state). Notably **`LFM2Cache` reads state shape from the graph**, so
  raising `conv_L_cache` to 15 for P3 needs *no C++ change*.

Energy stays cut: nvml/RAPL sampling caveats for no added scientific content at this stage.

### 7.4 One correctness warning that affects Phase 0

**Validate the LIV mixer against HF *prefill*, not HF decode.** The transformers LFM2 decode path
reportedly does `conv_state.roll(-1)` in a way that drops one tap of history per step. This was *not*
independently confirmed on current `main`, so treat it as a flag rather than a fact — but since our
Phase 0 parity test is the foundation of everything, test against prefill (where the full causal conv is
unambiguous) and treat any decode mismatch as suspect on *their* side until proven otherwise.

Also: the conv-state `copy_` must sit under `torch.no_grad()`, or PyTorch raises *"a leaf Variable that
requires grad is being used in an in-place operation."*

---

## 8. Phased plan

| Phase | Work | Gate to pass |
|---|---|---|
| **0** | LIV mixer + HF weight-parity test; typed per-layer state API; declarative arm builder + meta-device count assertion; document-isolated packing; corpus doc-length audit | Full forward ≡ chunked prefill ≡ one-token decode; parity vs `Lfm2ShortConv`; state constant in context length; counts match |
| **1** | Synthetic calibration: MQAR difficulty tuned to non-saturation on `L0` only | Endpoint discriminates controls; bimodality characterized |
| **2** | Equal-budget pilot, ~0.25B tok/arm. **Measure `s_δ` for CE and recall** | Publish required n per endpoint; freeze one recipe per arm |
| **0b** | **Thin-matmul microbenchmark**, before any training compute: is `d→r→d` at r∈{128,256,512} actually faster than `d→d` on target hardware, **with CUDA graphs on**? Also benchmark `G-grouped`. | If no rank wins, drop the latency claim from the headline now rather than defending it later. Cheap, and it decides how P1 is framed |
| **3a** | **P1 rank sweep** {128, 256, 512} + `N-narrow`, `S-shared`, `G-grouped`, `1G`. 5 paired seeds, ~2B tok. Gate bias in **both** factorized and dense arms | Init-scale parity asserted at step 0; rank curve survives controls |
| **3b** | **P2** pairing study (`C-near` vs `C-far`, testing the literature's own disagreement) + `A-fewer3` (strongest competitor), `Q-mqa` secondary w/ loss-spike monitoring, `SWA`. **Retrieval endpoints primary** | Resident KV halved **and** retrieval preserved; latency null reported as predicted; must beat or tie `A-fewer3` to be worth anything |
| **3c** | **P3** `k5/k9/k15` cheap span sweep | Only proceed to routing if a wider span actually gains |
| **4** | Fresh confirmation on survivors, ≥8 new paired seeds | Pre-registered margins met, or report inconclusive |
| **5** | Long-context + systems on survivors only | No 32K quality claim without matched 32K training |

### Compute budget

Estimated as `6ND / (n_gpu × peak × MFU)` at 40% MFU on A100, 35% on H100. **These are the least
certain numbers here — measure your own MFU before committing.** Expect hybrid arms at ~0.8-0.95× the
pure-GQA arm's MFU, since the depthwise conv is memory-bound and its kernel is small.

| Stage | Work | 8×A100 |
|---|---|---:|
| Rank | 12 arms × 2 seeds × 150M/10B (~2.5 h each) | ~2.5 days |
| Confirm | 5 arms × 350M/20B (~12 h each) | ~2.5 days |
| Headline | 3 arms × 750M/50B (~2.6 d each) | ~8 days |
| Long-context (~3% tokens) + placement/PE sub-studies | | ~2 days |
| **Total** | | **~15-16 days (~3,000 A100-hours)**, or ~5-6 days on 8×H100 |

**This is feasible on one 8-GPU node in ~2-3 weeks.** On a single A100 it is only feasible at 150M (a
350M/50B run is ~10 days on one GPU) — which is exactly where **WSD forking pays for itself**, since
all arms branch from one stable-phase checkpoint. Given FarmShare allocates 1 GPU per job with a 6-hour
limit (per `KDA/run_mqar_*.sbatch`), the array-worker + idempotent-skip pattern already in the repo is
the right execution shape there; SB-AWS is the better fit for the 8-GPU stages.

**Halving option:** make **350M the headline** and drop the 750M stage. That removes ~8 of the ~16 days
for a study whose cache claims are *more* sensitive at 350M anyway (§3.1). Recommended unless a
scaling trend across three sizes is specifically wanted.

Kill rules, stated in advance: if P1's rank curve is flat *and* `N-narrow` matches it, the honest
conclusion is "spend the parameters wherever you like" — publish that. If P2 preserves retrieval,
report a capacity win and an explicit latency null. If `k15` doesn't beat `k3`, **drop P3** rather
than escalating to the router.

---

## 9. Open questions for the human

1. **Sequencing.** Accept running P1 in parallel with the `L0 vs A16-P` topology gate (§2), or keep
   the existing strict gating?
2. **KDA in or out?** Recommend out. Confirm.
3. **Compute target and headline scale.** The program is **~15-16 days on one 8×A100 node**, or
   **~8 days if 350M is the headline** and the 750M stage is dropped (recommended — cache claims are
   more sensitive at 350M anyway). FarmShare gives 1 GPU / 6 h per job, which suits the synthetic and
   rank stages via the array-worker pattern already in the repo; SB-AWS suits the 8-GPU stages and is
   where the corpora live. Confirm: single-node 8×GPU on SB-AWS, and 350M or 750M headline?
4. **Scope of P3.** Drop the router and run the reparameterization question instead, or keep the
   router as specified?
5. **Edge/energy.** Confirm cutting to GPU-only.

---

## 9a. Known gaps in the research behind this design

Stated so nobody mistakes coverage for completeness. Two research agents on the latency/kernels topic
died to API stalls; `07_latency_kernels.md` contains only the independently-verified subset.

| Gap | Why it matters | Blocking? |
|---|---|---|
| ~~Whether low-rank gates can be fused~~ | **RESOLVED.** Deep fusion is not worth it (128-element intermediate; the unavoidable cost is weight-matrix traffic). The real issue is **20 extra launches/token**; at the frozen **d=1024** breakeven is **1.35 µs (A100) / 2.43 µs (L40S)** vs a 5-10 µs launch cost → **CUDA graphs AND fused `d→2r` gates are both mandatory or the result flips sign.** (The oft-quoted 4.72 µs was d=2048.) See §5.1 | Resolved |
| ~~`fla` causal_conv1d signature / dilation~~ | **RESOLVED.** No `dilation` parameter; the Triton tap loop uses stride-1 offsets only (`o_x = o_t + i_w`). `fla` gives nothing for P3 | Resolved |
| ~~Corpus long-document distribution~~ | **De-risked.** FineWeb-Edu: 8.4% of tokens in >16K docs; ABF shows most gain survives removing long texts. Still measure Dolma2's | Not blocking |
| ~~Memory-accounting methodology~~ | **RESOLVED.** Five buckets reported separately, `allocated` as the figure and `reserved` as a footnote, each cross-checked analytically. See §5.2 | Resolved |
| ~~Edge runtime support / energy protocol / edge roofline~~ | **RESOLVED.** llama.cpp has first-class `LLM_ARCH_LFM2` but its conv is `GGML_OP_SSM_CONV` with **no dilation**, so variants need a new op per backend. GPU-only confirmed; take one calibration datapoint on the *unmodified* GGUF. See §7 | Resolved |
| ~~Whether activation-aware spectra (`W·Σ_x^{1/2}`) are more favorable~~ | **RESOLVED 2026-07-31** (`probes/spectra_v2.py`, 32,768 tokens, `rank(Σ_x)=1024`). Aware rank drops ~36% but **identically for the value-stream control** (gates 493.3 vs value 507.8) → premise stays falsified, permanently. However **rank-128 retains 92.6% of activation-weighted energy** vs 45.8% plain, so P1's *feasibility* case is much stronger. See §5.1 | Resolved |
| Corpus document-length distribution in `olmo-150b-dolma2` | Determines whether 16K/32K training is possible | No longer blocking — FineWeb-Edu measures 8.4% of tokens in >16K docs, and ABF shows most of the gain survives removing long texts. Still measure ours |
| Whether OLMo-core's μP is coordinate-checked | Affects how LR transfers across the width-varying arms (`N-narrow`) | No — use `fan_in` init + the ladder's empirical LR formula |

## 10. Source record

Research dossier (all in [`Brainlifts/liv_experiment_research/`](../Brainlifts/liv_experiment_research/)):
`00_my_arithmetic_check.md` (independent verification + crossover math, with `crossover.py`,
`proposals.py`), `01_lfm2_architecture.md`, `02_lowrank_gates.md`, `03_kv_sharing.md`,
`04_multiscale_routing.md`, `05_evaluation.md`, `06_baselines_infra.md`,
`08_local_infra_audit.md`.

Primary sources most load-bearing here:
- [LFM2 technical report](https://arxiv.org/abs/2511.23404); [LFM2.5-1.2B-Base config](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base/raw/main/config.json)
- [`modeling_lfm2.py`, transformers v5.0.0rc1](https://raw.githubusercontent.com/huggingface/transformers/v5.0.0rc1/src/transformers/models/lfm2/modeling_lfm2.py) (Apache-2.0)
- [Cross-Layer Attention](https://arxiv.org/abs/2405.12981); [Character.AI inference post (Wayback)](http://web.archive.org/web/20240624161133/https://research.character.ai/optimizing-inference/)
- [Zamba](https://arxiv.org/html/2405.16712v1); [Zoology](https://arxiv.org/abs/2312.04927); [Just Read Twice](https://arxiv.org/abs/2407.05483)
- [Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825); [RepVGG](https://arxiv.org/abs/2101.03697)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (kernel size 2/3/4, no dilation)
- Local: [`docs/liv-kda-gqa-sub500m-experiment.md`](liv-kda-gqa-sub500m-experiment.md),
  [`docs/kda-liv-architecture-redesign.md`](kda-liv-architecture-redesign.md), [`KDA/HANDOFF.md`](../KDA/HANDOFF.md)
