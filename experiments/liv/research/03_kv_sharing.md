# KV-Cache Sharing and Reduction for a Hybrid Conv+Attention LM

Research dossier supporting the proposed **Cross-Layer Attention (CLA) on the attention layers
of an LFM2-style hybrid** experiment.

**Date:** 2026-07-30
**Scope:** the full space of KV-cache reduction mechanisms (cross-layer sharing, head sharing,
low-rank latent compression, local/global interleaving, post-hoc eviction/quantization);
whether any published hybrid does cross-layer KV sharing; implementation reality in serving
stacks; the retrieval-evaluation gap; and what must be measured.

**Object under study.** 16 layers, 10 gated-short-conv ("LIV") blocks + 6 GQA attention layers
at indices `[2, 5, 8, 10, 12, 14]`, `d = 2048` (1.2B) or `d = 1024` (350M), 32 query heads /
8 KV heads (350M: 16/8), `head_dim = 64` → KV width 512 per attention layer.
**Proposal:** pair nearby attention layers into producer/consumer pairs via CLA so that only
**3 KV banks** are resident instead of 6. The consumer keeps its own `q_proj` and `o_proj` and
drops `k_proj`/`v_proj`, reading the producer's K/V.

**Conventions.**
- Claims sourced to a paper/blog are marked with **title + arXiv id or URL**.
- My own arithmetic or reasoning is labelled **[DERIVATION]** or **[REASONING]**.
- Places where the literature is silent are labelled **[GAP]**. Several of these are
  load-bearing for the novelty claim.
- **[UNVERIFIED]** marks a number I could not confirm against a primary source.

---

## 0. What is already established (carried in, not re-derived here)

These were settled by earlier passes on this dossier and are stated here only so the rest of
the document can build on them. Sources: *Reducing Transformer Key-Value Cache Size with
Cross-Layer Attention* (Brandon, Mishra, Nrusimha, Panda, Kelly; arXiv:2405.12981) and
*Zamba: A Compact 7B SSM Hybrid Model* (arXiv:2405.16712).

**CLA (arXiv:2405.12981) — the base mechanism being proposed.**
- Validated at two scales trained from scratch on SlimPajama: **1B params / 30B tokens** (the
  design-space sweep) and **3B params / 100B tokens** (the confirmation run).
- **CLA2** (sharing factor 2, i.e. one producer feeds one consumer) gives **2× KV cache
  reduction for 0.04–0.05 ppl degradation**, and in some configurations *improves* perplexity
  relative to an equal-footprint baseline.
- The consumer drops **only** the K and V projections. It keeps its own **Q** and **output**
  projections. This is what makes CLA cheap in parameters but not free in expressivity.
- **Uniform adjacent pairing won.** Non-uniform pairings lost — `DenseBack` was **+0.43 ppl
  worse**. The paper explicitly recommends pairing **consecutive** layers.
- CLA pairs best with **MQA**, not GQA. **GQA + CLA2 mostly lost** to equal-KV-footprint
  baselines. This is the single most important caveat for the proposal, because LFM2 is GQA-8.
- Uses **separately learnable affine LayerNorm parameters** for the KV-producing block vs the
  Q block.
- **CLA saves cache CAPACITY, not BANDWIDTH.** The paper is explicit: it has *"no direct effect
  on the memory bandwidth consumed by the attention mechanism in each decoding step"* —
  consumers re-read the shared bank, so KV **read** bytes are unchanged; only KV **write** bytes
  and **resident** bytes fall.
- Sharing factor > 2 is worse on the Pareto frontier: **CLA3 = 3584 B/tok at ppl 13.77**,
  **CLA4 = 2560 B/tok at ppl 13.95**. Both Pareto-dominate MQA but lose to *CLA2 + larger head
  dim*.
- **RoPE pre- vs post-rotary sharing is NOT discussed anywhere in the CLA paper** — confirmed
  absent. This is a genuine [GAP] the experiment must resolve on its own (see §10.3).

**Zamba (arXiv:2405.16712) is NOT prior art for this proposal.** Zamba shares attention
*weights*: one attention block is applied 13 times, with input `LN([x_l, x_0])` (the current
hidden state concatenated with the original embedding). The paper explicitly states the shared
block has *"independent activations and KV-cache entries at each invocation"*. Weight sharing
and KV sharing are **orthogonal** mechanisms — Zamba reduces *parameters*, CLA reduces *cache*.
A model can do both.

**The scheduling obstacle.** In the LFM2 layer schedule, **no two attention layers are
adjacent**:

| pair | layers | intervening blocks |
|---|---|---|
| (2, 5) | 2 → 5 | 2 conv blocks (3, 4) |
| (5, 8) | 5 → 8 | 2 conv blocks (6, 7) |
| (8, 10) | 8 → 10 | 1 conv block (9) |
| (10, 12) | 10 → 12 | 1 conv block (11) |
| (12, 14) | 12 → 14 | 1 conv block (13) |

So CLA's own strongest recommendation — *pair consecutive layers* — **cannot be satisfied
exactly**. The closest available pairings are separated by one conv block: (8,10), (10,12),
(12,14). The two low-index pairs (2,5) and (5,8) are separated by two conv blocks. Whether
"consecutive in the attention subsequence" is a good enough substitute for "consecutive in the
layer stack" is the central empirical unknown, and §7 shows the literature does not answer it.

---

## 1. The other cache-reduction families

All numbers in this section were read from arXiv full text (HTML or PDF). Figure-only values are
flagged, because several headline numbers in this literature exist only inside plots.

### 1.1 The unifying framework — read this before the individual papers

The single most useful paper for positioning our work is
**"A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference" (arXiv:2410.14442,
Wu, Wu & Tu — NAACL 2025 main)**. It defines `kv(i)` = the layer whose KV pairs with layer *i*'s
queries; layers with `kv(i) = i` are **"KV layers"** (count `l`), and non-KV layers **drop their
`W_K`, `W_V`**. Two orthogonal axes give 9 configurations:
- **Partitioning:** **pizza** (first `l−1` layers are KV layers, the rest share one target) /
  **sandwich** (first `⌈(l−1)/2⌉` + last `⌊(l−1)/2⌋` are KV layers, middle block shares) /
  **lasagna** (L layers split into `l` consecutive groups, each with its own target).
- **Target position:** **bottom** / **top** / **middle** (+ middle-1/4, middle-3/4 in App. E).

**Crucially, this framework shows the three main methods are the same mechanism with different
pairing choices:**

| Existing method | = configuration |
|---|---|
| **CLA** (arXiv:2405.12981) | **lasagna-bottom** |
| **YOCO** (arXiv:2405.05254) | **pizza-bottom** (differing only in using efficient attention in the bottom half) |
| **LCKV** (arXiv:2405.10637) | **sandwich-top** |

**[REASONING] This is the frame our paper should adopt**, because it makes the contribution
precise: we are not inventing a mechanism, we are testing *lasagna-bottom* (= CLA) in a hybrid
where the group boundaries are forced by a conv/attention schedule, under an evaluation
(retrieval) that this whole literature omits.

Its headline conclusions, at 110M and 1.1B trained **from scratch** (MiniPile 1.7B tokens;
100B-token SlimPajama subset; batch 2M tokens, LR 4e-4→4e-5, A800×128):
- **At 2× reduction, the cheap "bottom" configs suffice** — *"most configurations can achieve
  higher throughput than standard transformers while maintaining competitive performance"*, with
  **no extra training or prefill cost**.
- **Beyond 2×, upper-layer KV wins** — *"pairing queries of all layers with KVs of upper layers
  performs better"*, but *"at the expense of additional training cost and prefilling latency"*;
  `sandwich-middle` is best at high compression.
- **Layer-role asymmetry:** they hard-code the first layer as a KV layer because *"there is a
  significant drop in performance if the first layer is not a KV layer."*
- Throughput caveat: at long prompts (512+1024) top/middle configs *"degrade dramatically,
  falling below the baseline in some cases"* due to iterative prompt encoding; bottom configs
  stay above baseline.
- Table 4 (1.1B, zero-shot, 22 KV layers = baseline, 11 = the 2× point). Baseline avg row:
  HellaSwag 44.58 / OBQA 30.2 / WinoGrande 50.99 / ARC-c 25.00 / ARC-e 46.38 / BoolQ 60.46 /
  PIQA 68.93 / SciQ 74.8. At 11 KV layers: **pizza-bottom** 44.20/29.4/51.93/25.00/46.55/59.51/
  68.28/72.1; **lasagna-bottom (CLA)** 43.43/30.8/50.51/24.49/44.61/59.24/69.21/71.5;
  **sandwich-top (LCKV)** 44.74/31.0/51.70/24.83/46.38/61.38/67.90/72.5.
- ⚠️ Perplexity and throughput values are **figure-only**; only Table 4 is numeric.
- **No retrieval/needle/passkey evaluation** — see §7.
- Code: `github.com/whyNLP/LCKV`.

**[REASONING] Direct implication for our design.** We target exactly 2× (6 banks → 3). This paper
says that at 2× the pairing choice barely matters for *aggregate* quality, and that bottom-target
(= CLA-style, producer below consumer) is the option with **no prefill penalty**. Both facts
support our proposal's shape, and both also imply aggregate metrics will not discriminate our
pairing variants — reinforcing §7's argument that retrieval must be the discriminating metric.

### 1.2 YOCO — decoder-decoder, one global KV cache

**"You Only Cache Once: Decoder-Decoder Architectures for Language Models" (arXiv:2405.05254,
Sun et al., Microsoft Research + Tsinghua).**

- **Mechanism.** L blocks split in half: the **first L/2 layers are the self-decoder**, the
  **last L/2 are the cross-decoder**. The self-decoder's output `M = X^(L/2)` is projected
  **once** through learned `W_K`, `W_V` into global caches `K̂`, `V̂`; **all L/2 cross-decoder
  layers reuse those same `K̂`, `V̂`**, each keeping its own `W_Q`. Causal masking throughout.
- **Self-decoder efficient attention (ESA)**, two variants: **gated retention** (the default in
  all main experiments; head-wise decay `γ = sigmoid(XW_γ)^(1/τ)` so it uses tensor cores;
  parallel/chunkwise for training, **recurrent at inference for constant memory**, chunk 256) and
  **sliding-window attention** (window **1024**). Scaling curves: **gated retention beat both a
  Transformer and YOCO_SWA**.
- **How many caches are resident.** One *global* KV cache, with an honest footnote: *"The word
  'once' refers to global KV cache. Strictly, self-decoder also needs to store a certain number
  of caches. As the self-decoder utilizes an efficient attention module, the cache size is
  bounded to a constant."* Complexity **O(N + CL) ≈ O(N)** vs Transformer **O(LND)**.
- **Reported reductions.** **1M context, 3B model: 12.4 GB total inference memory; a Transformer
  needs 9.4× more.** At **32K, ~2× less**. Per-token KV **~L× fewer**. At **65B scale, KV cache
  reduced ~80×** — YOCO serves **128K tokens in 1 GB** where a GQA Transformer manages only
  **1.6K tokens** in the same 1 GB. ⚠️ The per-length GB table (64K/128K/256K/512K) is
  **figure-only**.
- **Prefill implications — the strongest part of the paper.** Prefill can **early-exit before the
  cross-decoder**, so only half the layers run: *"at least 2× speedup even for short inputs."*
  Prefill complexity **O(LN²D) → O(LND)**.
  **512K context: ~180 s → under 6 s.** **1M context: ~300 s → 71.8× overall speedup.**
  **32K: 2.87×.** Throughput at 512K: **4.5 tok/s → 43.1 tok/s = 9.6×**.
- **Evals.** 1M needle-in-a-haystack: *"near perfect accuracy"* (⚠️ figure-only, no scalar).
  Multi-needle at 128K, YOCO-3B-1M: **N=1 0.98, N=2 0.98, N=4 0.84, N=8 0.56** (vs LWM-1M-text
  7B: 1.00/0.90/0.76/0.62). LM-Eval zero-shot avg: YOCO-3B@1T **0.634**, @1.6T **0.636**,
  YOCO-3B-1M **0.645**.
- **Scale.** YOCO-3B: hidden 3072, **26 layers**, 24 query / 8 KV heads, head dim 128, 2.8B
  non-embedding params, **1.6T tokens** (400k steps × 4M batch). Scaling curves at 160M–13B,
  10B tokens each. Length extension 64K → 256K → 1M.

**[REASONING] Why YOCO is not our design, and why it is the right thing to cite for prefill.**
YOCO's sharing factor is L/2 with a *single* producer — extreme, and it fundamentally restructures
the model (half of it stops being a normal decoder). It is `pizza-bottom` in §1.1's taxonomy. Our
proposal is far more conservative. But YOCO is the citation that establishes the one benefit CLA
*cannot* claim: **prefill compute reduction**. CLA consumers still run their full attention over
the shared bank, so CLA saves nothing at prefill. If a reviewer asks "why not get prefill savings
too", the answer is that doing so requires YOCO's structural commitment (and, per §1.1,
top/middle targets cost prefill latency in the general case).

### 1.3 Layer-Condensed KV Cache (LCKV) — and the training-stability correction

**"Layer-Condensed KV Cache for Efficient Inference of Large Language Models"
(arXiv:2405.10637, Haoyi Wu & Kewei Tu, ShanghaiTech — ACL 2024 main).**

- **Mechanism.** **Queries of all layers pair with keys and values of the top layer only** — one
  layer's KV is cached. Since other layers' KV are never needed, they **discard `W_K`, `W_V` for
  all other layers**. Motivated by reading layer stacking as iterative refinement (the top layer
  is most informative), and as analogous to encoder-decoder cross-attention.
- **The cyclic dependency and its fix.** A token needs its *own* top-layer KV for its lower-layer
  attention, but the top layer cannot be computed until lower layers finish. Fix: **mask the
  attention diagonal** (drop each token's self-attention); the first token attends to
  **zero-vector dummy KVs**. Residual connections still carry the token's own information;
  empirically the diagonal mask does not hurt.
- **Warmup layers `w`.** `w` layers keep standard attention, in a **"sandwich"**: top `w/2` +
  bottom `w/2`. Placement ablation (w=2, dev ppl) is decisive:
  **50M: all-bottom 14.556 / all-top 221.850 / sandwich 14.069**;
  **1.1B: sandwich 7.381 vs all-bottom 7.668 vs all-top 9.098.**
  Note the all-top catastrophe at 50M (221.85) — layer role matters enormously.
- **Parallel-training trick.** Replace *n* sequential bottom-up passes with ***n* parallel
  iterations over all tokens**, each pairing all-layer queries with the **previous iteration's**
  top-layer KV; loss only after the last iteration; iteration 1 uses dummy zero KVs.
  **Theorem 1** proves equivalence (token *i* correct from iteration *i* on).
- **TRAINING STABILITY — important correction to the brief.** ⚠️ **LCKV reports no divergence
  and no loss spikes.** Grepping v1 and v2 for instab/diverg/spike/NaN turns up only:
  1. On backprop depth `b`: *"larger b leads to more unstable training, thus the model is harder
     to converge"* — hence **`b = 2` by default**. 1.1B numbers: **b=2 → 10.390 ppl / 8h;
     b=3 → 10.476 / 10h; b=4 → 10.885 / 13h.** And **`b = 1` is broken by construction** — the
     KVs used in the last iteration come from the second-to-last, so **the KV-producing
     parameters receive no gradient at all.** `b ≥ 2` is comparable to a standard transformer.
  2. Appendix A: parallel training is *"theoretically guaranteed to converge to the same solution
     as the sequential training. Once it is converged, it will not diverge."*
  3. A **KV Loss** auxiliary term (MSE between KVs before/after the last iteration) was tried and
     **abandoned** — it helped only at 50M with w=0 (15.965 → 15.610) and **hurt otherwise**
     (50M w=2: 15.004 → 15.065; 1.1B w=2: 9.746 → 10.073).
  4. Forward iterations **`m = 7`** (a randomly initialized model needs ~15–20 to converge;
     performance converges at m ≥ 6); prompt encoding at inference uses **m + b = 9** iterations.
  5. A `bfloat16` numerical quirk in the repo (Llama MLP down-projection differing by token
     count) — a precision artifact, not a training instability.
  **The loss-spike/divergence result the brief was thinking of is in the GQA paper** — see §1.6.
- **Throughput: the 26× figure is a CPU-offload case.** 30B with CPU offload on an RTX 3090,
  512+1024: **Llama 0.23 tok/s (batch 4) → w=2 5.99 tok/s (batch 83) = 26.0×**; w=10 gives 1.63
  tok/s (7.1×). More representative: 7B on 3090, 5+8187 → **32.02 → 151.91 tok/s (4.7×)**;
  A100 30B 2048+2048 → **14.10 → 108.29 (7.7×)**. Memory is reported as **batch headroom, up to
  32× larger batches**; ⚠️ **no GB figures**.
- **Quality.** SlimPajama dev ppl: **TinyLlama 9.219 / w=2 9.746 / w=10 9.265**. Zero-shot avg
  (7 tasks): **TinyLlama 46.65 / w=2 45.45 (−1.20) / w=10 46.84 (+0.19)**. So `w=10` slightly
  *beats* baseline; `w=2` costs ~1.2 points.
- **Setup.** Architectures not checkpoints (TinyLlama config at 1.1B); main 1.1B models trained
  **from scratch** on a **100B-token SlimPajama subset**, 128× A800. **Training cost ~2.7–2.8×
  TinyLlama's** (14:43 h → 1d 16:44) — a stated limitation. 7B/30B are throughput benchmarks
  only, not trained. Continued-pretraining variant from a TinyLlama 2.5T checkpoint reaches ppl
  8.514 / avg 49.55. Trained at context 2048; integrates with StreamingLLM stably to **4M tokens**.
  Stated limitation: throughput degrades when prompts far exceed generation length.

**[REASONING] Relevance to us: LCKV is the cautionary tale about aggressive top-layer sharing.**
Its `all-top` ablation blowing up to 221.85 ppl at 50M, and its need for iterative training with
a proven-equivalence theorem, are the cost of choosing a *cyclic* pairing. Our proposal is
strictly feed-forward (producer below consumer), so **none of LCKV's machinery is needed** — no
iterations, no diagonal masking, no `m`/`b` hyperparameters, no ~2.8× training cost. That is a
concrete advantage to state explicitly, and LCKV is the citation that quantifies what we avoid.

### 1.4 MLKV — sharing KV heads across layers, taken to the extreme

**"MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding"
(arXiv:2406.09297, Zuhri, Adilazuarda, Purwarianti, Aji).**

- **Mechanism.** Share KV heads **not only within a layer (GQA/MQA) but also between layers**, so
  the total KV head count can fall **below the layer count**. With `l` layers, `h` query heads,
  `m` = layers owning KV heads, `g` = KV groups per layer: **total KV heads = m·g**, cache size
  **2·b·s·m·g·d_k**. ⚠️ Table 1 gives formulas only, no byte figures.
- **Configurations and results** (all `l=12`, `h=12`; uptrained from `pythia-160m-deduped`):

  | Model | m | g | KV heads | ARC-e | LAMBADA | PIQA | SciQ | **Avg** |
  |---|---|---|---|---|---|---|---|---|
  | Pythia-160M (base) | 12 | 12 | 144 | 43.94 | 33.63 | 61.37 | 72.2 | **52.79** |
  | GQA-48 | 12 | 4 | 48 | 41.92 | 29.38 | 60.77 | 68.6 | 50.17 |
  | MLKV-48 | 4 | 12 | 48 | 42.13 | 26.18 | 59.96 | 68.9 | 49.29 |
  | MQA-12 | 12 | 1 | 12 | 40.19 | 26.74 | 61.10 | 69.7 | 49.43 |
  | MLKV-12 | 4 | 3 | 12 | 41.08 | 23.44 | 60.28 | 70.3 | 48.78 |
  | MLKV-6 | 6 | 1 | 6 | 41.41 | 24.35 | 60.55 | 69.9 | 49.06 |
  | MLKV-2 | 2 | 1 | 2 | 40.91 | 22.03 | 59.47 | 64.7 | 46.78 |
  | **MLKV-1** | 1 | 1 | 1 | 38.26 | **8.56** | 59.25 | 58.4 | **41.12** |

- **The 6× claim is relative to MQA** (MQA-12 → MLKV-2), and costs **52.79 → 46.78 avg
  (−6.01 points)**. **MLKV-1 (one KV head for all 12 layers) collapses** — LAMBADA **33.63 →
  8.56**; the authors call it *"basically unusable."* Their actual recommendation is the mild
  setting (**share every second layer, 2× beyond MQA**); Pareto-optimal points are **MQA-12 and
  MLKV-6**.
- **Uptrained, not from scratch:** GQA's recipe on **5% of the deduplicated Pile** (6M of 134M
  docs, 2.46M packed rows at seq 2048), KV head weights **averaged** across merged layers,
  parameters equalized via MLP width (~162.32M each), ~22 h on 2× A100. ⚠️ Token/step count not
  reported.
- **Memory vs speed — a result that matters for §9.** Max batch before OOM: baseline 48 →
  MLKV-2 **940** → MLKV-1 **1100**. But: *"we do not see any significant speed-up through
  MLKV"*, attributed to per-layer cache-fetch overhead being unchanged.

**[REASONING] MLKV is the cleanest independent confirmation of CLA's capacity-not-bandwidth
point**, from a different mechanism: huge capacity/batch gains, **no throughput gain**. Cite it
alongside CLA in §9 to justify why our measurement table must separate resident bytes from read
bytes. It is also a warning about the *shape* of the degradation curve: MLKV's collapse is
concentrated in **LAMBADA**, the most retrieval/completion-like task in its suite, while PIQA
barely moves (61.37 → 59.25). **That is the same dissociation §7 predicts.**

### 1.5 DeepSeek MLA — the "why not MLA" answer

**DeepSeek-V2 (arXiv:2405.04434)** and **DeepSeek-V3 (arXiv:2412.19437)**.

- **Low-rank joint KV compression** (V2 Eqs. 9–11): `c_t^KV = W^DKV h_t` with `c_t^KV ∈ ℝ^{d_c}`
  (down-projection), then per-head up-projections `k_t^C = W^UK c_t^KV`, `v_t^C = W^UV c_t^KV`.
  **Only `c_t^KV` is cached, not K and V.** Query compression (`c_t^Q`, Eqs. 12–13) exists too,
  but only to cut *training activation* memory.
- **Decoupled RoPE key** (Eqs. 14–19): per-head decoupled queries `q_{t,i}^R ∈ ℝ^{d_h^R}` and
  **one shared decoupled key `k_t^R ∈ ℝ^{d_h^R}`** across all heads;
  `q_t^R = RoPE(W^QR c_t^Q)`, `k_t^R = RoPE(W^KR h_t)`; concatenated `q_{t,i} = [q^C; q^R]`,
  `k_{t,i} = [k^C_{t,i}; k_t^R]`; softmax scaled by `√(d_h + d_h^R)`.
  **Both `c_t^KV` and `k_t^R` are cached** → `(d_c + d_h^R)·l` elements.
- **Cache per token (V2 Table 1):**

  | Mechanism | KV cache per token (# elements) | Capability |
  |---|---|---|
  | MHA | **2 n_h d_h l** | Strong |
  | GQA | 2 n_g d_h l | Moderate |
  | MQA | 2 d_h l | Weak |
  | **MLA** | **(d_c + d_h^R) l ≈ (9/2) d_h l** | **Stronger** |

  With `d_c = 4d_h` and `d_h^R = d_h/2`: `4.5 d_h l`, which the paper notes is *"equal to GQA with
  only 2.25 groups, but its performance is stronger than MHA."*
- **Hyperparameters.** V2: 60 layers, hidden 5120, `n_h=128`, `d_h=128`, **`d_c=512` (=4·d_h)**,
  `d_c′=1536`, **`d_h^R=64` (=d_h/2)**; 236B total / 21B active; 8.1T tokens. Extra RMSNorm after
  compressed latents + scaling factors at width bottlenecks **for stability**. V3: 61 layers,
  hidden 7168, **identical MLA dims**.
- **93.3% reduction** is vs the dense predecessor DeepSeek 67B: *"saves 42.5% of training costs,
  reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times."*
- **The quality claim, exactly (V2 Appendix D.2)** — MLA vs MHA at two MoE scales:

  | Benchmark | Small MoE MHA | Small MoE MLA | Large MoE MHA | Large MoE MLA |
  |---|---|---|---|---|
  | **KV cache/token (# elem)** | **110.6K** | **15.6K** | **860.2K** | **34.6K** |
  | BBH (EM) 3-shot | 37.9 | **39.0** | 46.6 | **50.7** |
  | MMLU 5-shot | 48.7 | **50.0** | 57.5 | **59.0** |
  | C-Eval 5-shot | **51.6** | 50.9 | 57.9 | **59.2** |
  | CMMLU 5-shot | 52.3 | **53.4** | 60.7 | **62.5** |

  **MLA beats MHA on 7 of 8 cells while using 14% (small) / 4% (large) of MHA's cache.**
  Appendix D.1 separately shows MHA ≫ GQA-8 ≫ MQA at 7B dense on hard benchmarks (MMLU
  **45.2 / 41.2 / 37.9**; C-Eval **42.9 / 37.7 / 30.0**), which is DeepSeek's stated reason for
  not settling for GQA/MQA. ⚠️ **V3 contains no MLA ablation** — it defers to V2.
- **Why MLA is awkward with standard kernels.** MLA's efficiency depends on an **absorption
  trick**: fold `W^UK` into `W^Q` and `W^UV` into `W^O` at inference so K/V are **never
  materialized**. RoPE breaks this — a position-dependent rotation sits between `W^UK` and `W^Q`,
  and since matmul is not commutative, `W^UK` *"cannot be absorbed into W^Q any more during
  inference"*. Hence the decoupled RoPE dims. The resulting path is non-standard; DeepSeek states
  MLA *"is also optimized based on an improved version of FlashAttention-2"* rather than using it
  unmodified.

**[REASONING] The honest answer to "why not MLA?", which the paper must contain.** MLA is
**strictly stronger than CLA on the capacity/quality frontier** — it is the only method in this
survey that reports *beating* full MHA while cutting cache >7×, and Kimi Linear (§4.1) chose it
for a 2025 hybrid. So we cannot claim CLA dominates MLA. The defensible position has four parts:
(a) **they are orthogonal** — MLA compresses *within* a layer, CLA shares *across* layers, and
nothing prevents composing them (an MLA+CLA hybrid is itself unexplored, **[GAP]**);
(b) **MLA is a much larger architectural change** — it replaces the attention block, needs the
absorption trick, decoupled RoPE, extra RMSNorms "for stability", and a modified FlashAttention;
CLA deletes two `nn.Linear`s and is supported by vLLM and HF today (§5);
(c) **MLA's evidence is at MoE scale on 8.1T tokens**, not at 350M–1.2B on an academic budget,
and its quality-beats-MHA claim has not been reproduced in the small-dense regime;
(d) **for our specific model, MLA's win is smaller than advertised** — MLA is compared against
*MHA*, but LFM2 is already GQA-4:1 with KV width `d/4`, so much of MLA's headline reduction is
already banked. **[DERIVATION]** LFM2 at `n_h=32, n_g=8, d_h=64`: GQA cache = `2·8·64 = 1024`
elements/layer/token; MLA at `d_c=4d_h=256`, `d_h^R=32` would be `288` elements/layer/token —
still **3.6× better than our GQA**, and better than CLA2-on-GQA's `512`. **So MLA genuinely wins
on capacity and we must say so, and run it as a control (§10.2).**

### 1.6 GQA / MQA — the baseline axis, and the real training-stability finding

**MQA: "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv:1911.02150, Shazeer).**
K and V shared across **all** heads (one write-head), shrinking the tensors reloaded per decode
step. WMT14 EN-DE (h=8, d_k=d_v=128; `d_ff` enlarged 4096→5440 for MQA to match params):
**multi-head ln(PPL) 1.424, BLEU dev 26.7, test beam-1/4 27.7/28.4** vs
**multi-query ln(PPL) 1.439, BLEU dev 26.5, test 27.5/28.5** — i.e. **−0.2 BLEU dev, and MQA
actually had the higher beam-4 test BLEU (28.5 vs 28.4)**. Every alternative that shrank `h` or
`d_k` was clearly worse (h=1: ln(PPL) 1.518, BLEU 25.8). Cost per output token (TPUv2-µs, len
128): **multi-head 1.7 enc + 46 dec → multi-query 1.5 + 3.8**, i.e. **~12.1× faster decoding**;
training essentially unchanged (13.2 → 13.0). Billion-Word LM dev ppl: **29.9 → 30.2**, again far
better than shrinking h/d_k (31.2/31.1/31.0/30.9).

**GQA: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
(arXiv:2305.13245, Ainslie et al., EMNLP 2023).**
- **Conversion:** key/value projections of all heads in a group are **mean-pooled** into one head
  per group. Mean pooling **> selecting a single head > random init**.
- **Uptraining:** additional pretraining for proportion **α of original steps**, same recipe/data
  (T5/C4). **α = 0.05 (5%)** in the main experiments — **~600 TPUv3 chip-days** — with
  **diminishing returns past 10%**. Applied to decoder self- and cross-attention, **not** encoder
  self-attention. **GQA is usable immediately after conversion; MQA needs uptraining to be viable
  at all.**
- **T5-XXL, seconds/sample/TPUv4 chip:** **MHA-XXL 1.51 s / avg 47.2**; **MQA-XXL 0.24 s / 46.6**;
  **GQA-8-XXL 0.28 s / 47.1**; MHA-Large 0.37 s / 46.0. So **GQA-8 recovers essentially all MHA
  quality (−0.1) at ~5.4× the speed**, only 17% slower than MQA, and beats MHA-XXL on PubMed
  (47.7 vs 47.5) and MultiNews (47.2 vs 46.9).
- **Group-count sweep:** 1→8 groups adds only modest cost, growing **non-linearly** toward full
  head count; **8 groups chosen as "a favorable middle ground."** GQA also **avoids MQA's waste
  under sharding**, where the lone KV head must be replicated across partitions.
- **THE TRAINING-STABILITY FINDING (Appendix A) — this is what the brief was after.**
  **MQA models trained from scratch showed "frequent loss spikes" and diverged on long-input
  fine-tuning.** Uptrained MQA remains **high-variance** (unstable-task results averaged over
  three runs), while **uptrained GQA models "appear to be stable."**

**[REASONING] This correction materially changes a design decision.** CLA's headline result is
**CLA2 + MQA**, and CLA reports that **GQA+CLA2 mostly lost**. The naive reading is "switch LFM2
to MQA and apply CLA2." But GQA's Appendix A says **MQA trained from scratch has frequent loss
spikes and diverges on long inputs** — and we are training from scratch at 350M–1.2B, exactly the
regime where that was observed, with no uptraining checkpoint to fall back on. So the
CLA-recommended recipe carries a **stability risk that CLA itself never had to face** (CLA's runs
were from scratch too, so they either got lucky or did not report spikes — **[GAP]**, CLA reports
no stability analysis). **Recommendation: do not switch to MQA as the primary configuration.**
Keep GQA-8 (LFM2's native setting), accept that this is the harder setting for CLA, and treat
"CLA2 + MQA" as a clearly-labelled secondary arm with loss-spike monitoring. This makes the GQA
result in §0 a *feature* of the experiment — we are testing CLA in the setting where it is known
to be weakest, which is honest and is where the open question actually is.

### 1.7 2025–2026 cross-layer follow-ups

Verified titles/IDs. These matter because two of them directly touch our RoPE decision and our
"which layers to pair" gap.

- **KVSharer — "KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing"
  (arXiv:2410.18517).** Training-free, post-hoc. Searches a sharing strategy by averaging each
  layer's KV, computing pairwise Euclidean distance, and **sorting DESCENDING — most dissimilar
  first** — greedily accepting pairs while final-layer hidden-state cosine similarity to the
  original stays above **𝒯 = 0.5**. Direction rule: **replace the layer closer to the output with
  the one closer to the input** (input-side layers are more sensitive). Calibration: **30 random
  Wikipedia sentences of 64 tokens**; search **~1 minute**. **The counterintuitive finding:**
  *"sharing dissimilar KV caches better preserves the model performance"* — reversing to
  similarity-ascending order gives perplexity *"often nearly twice as high or more."*
  Headline: **30% KV computation reduction, >95% performance at 70% memory, ≥1.3× (up to 1.88×
  with PyramidInfer)**. Llama2-13B-Chat (512+2048): **51,639 MB / 18.2 tok/s → 37,049 MB (72%) /
  30.0 tok/s (1.65×)**. **Prefill time essentially unchanged (~0.087–0.089 s)**.
  **[REASONING] This directly contradicts Hymba's and MiniCache's stated rationale** ("adjacent
  layers have similar KV, therefore share them"). Both cannot be right in general. For us this is
  a live warning: **the intuition that similar-KV layers are the safe ones to pair is contested**,
  so our pairing choice should be *measured*, not assumed — which is exactly what §10.1 proposes.
- **SwiftKV — "SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model
  Transformation" (arXiv:2410.03960, Snowflake AI Research).** Two parts: **SingleInputKV** —
  compute KV for **all layers deeper than `l` directly from layer `l`'s output**
  (`KV_j = W_KV^j · x_{l+1}` for `j > l`), so prefill tokens skip the Q/O projections, attention
  and MLP of every layer `> l` and prefill compute approaches `l/L` (decode still traverses all
  layers); and **AcrossKV** — cross-layer KV sharing among the skipped layers, merging **more
  than two** layers. **Knowledge-preserving distillation** trains only `W_QKV` from layer `l+1`
  on: **<10% of weights, <1B tokens** (≈680M Llama-3.1 tokens), **8B in 3 h on 8×H100**.
  Numbers: **prefill compute −25–50%** (70B: 302 → 228 GFlops/token at 25%, → 154 at 50%);
  **KV memory −25% (2-way) / −37.5% (4-way) / −43.75% (8-way) / −46.875% (16-way)**; **up to 2×
  throughput, TTFT −50%, TPOT −60%.** Quality (7-task avg): Llama-3.1-8B-Instruct **73.71 → 73.59
  (25%) → 72.70 (50%)**; +4-way AcrossKV 71.49. 70B: **84.31 → 82.98 (50%)**, +4-way 82.96.
  **Distillation ablation: no distillation 70.06 vs 72.70 with (+2.64); full-model tuning is
  WORSE than partial (68.23 vs 72.70).**
  **[REASONING] SwiftKV is the strongest argument that our mechanism is retrofittable** — it
  converts an existing checkpoint with <1B tokens of distillation. Worth noting as future work:
  if from-scratch CLA on LFM2 works, a *conversion* recipe from stock LFM2 would be cheap.
- **FusedKV — "Reconstructing KV Caches with Cross-layer Fusion For Enhanced Transformers"
  (arXiv:2512.03870, Lin et al. — ACCEPTED AT ICLR 2026).** **[VERIFIED]** This is the most
  important recent paper for us, for two reasons.
  1. It states plainly that cross-layer sharing *"typically underperforms within-layer methods
     like GQA"* — an independent, 2026, peer-reviewed echo of CLA's own GQA caveat. **A reviewer
     may well cite this at us.**
  2. It diagnoses *why*, by studying information flow into top-layer KV: **values are
     predominantly derived from the bottom layer, while keys draw from both bottom and middle
     layers.** So it makes top-layer KV a **learnable fusion** of the most informative
     bottom/middle caches. Result: **332M–4B models, 50% cache reduction with LOWER validation
     perplexity than a standard Transformer.**
  3. **It operates on POST-RoPE keys**, explicitly *"to preserve relative positional information
     while avoiding the cost of re-applying rotary embeddings."* **[VERIFIED from abstract]**
  **[REASONING] Three implications.** (a) The K/V asymmetry finding suggests **K and V may want
  different pairings** — a variant nobody has tried in the CLA setting and a cheap ablation for
  us (share V only, share K only, share both). (b) Its post-RoPE choice is a **third independent
  vote** for post-rotary sharing (with Hymba and Gemma 3n), and the first with a stated *reason*.
  (c) Its "lower perplexity than standard Transformer at 50% cache" result raises the bar for
  what counts as a good outcome.
- **Stochastic KV Routing — "Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing"
  (arXiv:2604.22782, Filippova, Grangier, Cuturi, Monteiro; Apr 2026).** **[VERIFIED]** Argues the
  **depth** dimension is *"an orthogonal and robust avenue"* vs the temporal axis, and notes
  existing cross-layer approaches *"typically suffer from reduced throughput or increased
  time-to-first-token."* Mechanism: **"random cross-layer attention"** — during training,
  *"layers randomly choose to attend either to their own KV states or those of a preceding
  layer"*, making the model *"robust to various depth-wise cache sharing strategies, ensuring
  flexibility for unknown hardware constraints at deployment time."* Reports a
  **regularization-like effect** for bigger models under limited data, often maintaining or
  improving quality while cutting cache.
  **[REASONING] This partially closes our "which layers to pair" gap and must be cited.** It is
  the answer-by-refusal: don't pick a pairing, train for all of them. It also gives independent
  support for the "sharing can act as a regularizer" reading of CLA's occasional *improvements*.
  It does **not** cover hybrids or retrieval, so our gap survives — but the framing "nobody
  studies which layers to pair" is no longer strictly true and should be softened.
- **YOCO++ (arXiv:2604.13556)** — adds a **weighted residual connection between each bottom-half
  layer's KVs and the bottom layer's KVs**. Claims **SOTA among cross-layer KV compression at 50%
  compression, outperforming the standard Transformer.** ⚠️ **[UNVERIFIED]** — I did not fetch
  this one directly; treat the numbers as second-hand.
- **xKV (arXiv:2503.18893)** — post-training, no pretraining. Uses **Centered Kernel Alignment**
  to show the **dominant singular vectors** of KV caches align across layers (an explicit rebuttal
  of per-token cross-layer cosine similarity), then jointly factorizes grouped-layer KV into a
  shared low-rank subspace. **Up to 8× compression, up to 4.23× end-to-end speedup.**
  **[REASONING] The CKA-vs-cosine point is a useful sharpening of the KVSharer/MiniCache dispute:
  layers may be similar in *subspace* while dissimilar *per-token*.**
- **CLLA (arXiv:2410.15252)** — Cross-Layer Latent Attention; unifies head/dim reduction + layer
  sharing + quantization; **KV cache to <2% of original**, near-lossless on most tasks.
- **CommonKV (arXiv:2508.16134)** — training-free cross-layer compression via SVD weight sharing
  across adjacent parameters + adaptive budget by cosine similarity; **98% compression** combined
  with quantization and eviction.
- **LISA (arXiv:2408.01890)** — shares attention **weights** (not KV) across layers; relevant for
  its two obstacles: sharing without **rearranging attention heads** is ineffective, and
  **shallow layers are vulnerable to small deviations**. **6× compression of Q and K; redundant
  attention in 53–84% of layers; throughput +19.5% (LLaMA3-8B) to +40.1% (LLaMA2-13B).**
  **[REASONING] The "shallow layers are vulnerable" finding independently corroborates LCKV's
  "first layer must be a KV layer" and argues against making layer 2 a consumer in our schedule.**
- **LightTransfer (arXiv:2410.13846)** — identifies **"lazy layers"** attending only to
  recent/initial tokens and replaces their full attention with streaming attention; **half the
  layers lazy → up to 2.17× throughput, <1.5% loss on LongBench.**
- **Survey: "A Survey on Large Language Model Acceleration based on KV Cache Management"
  (arXiv:2412.19442).** Useful because its taxonomy separates **§4.3.2 "Cross-layer Merging"**
  (token-level, post-hoc: MiniCache arXiv:2405.14366, KVSharer) from **§5.1.2 "Cross-layer
  Sharing"** (model-level, architectural: CLA, LCKV, Shared Attention 2407.12866, MLKV, LISA,
  the systematic study 2410.14442, CLLA, DHA 2406.06567, ResFormer/SVFormer 2410.17897), with
  **YOCO filed separately under §5.2 architecture alteration** as "single global KV cache."
  **Adopt this taxonomy** — it is the cleanest way to show a reviewer we know where our method
  sits (§5.1.2, model-level, architectural).

---

## 2. Sliding-window / local+global interleaving — the competing way to cut cache

This is the **most serious competitor** to CLA in our setting, for a blunt reason: it is what
Gemma 3, Hymba, Samba, and Character.AI all actually chose, and it cuts cache far more
aggressively than 2×. It must be a control (§10.2), and the conv-redundancy question must be
answered head-on.

### 2.1 Gemma 2 → Gemma 3: the ratio went from 1:1 to 5:1

**Gemma 2 (arXiv:2408.00118) — verified.** **[VERIFIED]** Alternates local sliding-window and
global attention **1:1** — *"alternate between a local sliding window attention … and global
attention … in every other layer."* **Sliding window = 4096** (*"The sliding window size of local
attention layers is set to 4096 tokens"*; Table 1 lists `Sliding window: 4096` for 2B/9B/27B) and
**global attention span = 8192** (*"the span of the global attention layers is set to 8192
tokens"*), which equals the context length. GQA with `num_groups = 2`.

⚠️ **Important correction: Gemma 2 contains NO local/global interleaving quality ablation.**
Its §5 ablations cover distillation, GQA-vs-MHA (MHA 50.3 vs GQA 50.8 avg), wide-vs-deep, and
sliding-window *size*, but **never** compares interleaved against global-only. **Do not claim
Gemma 2 measured the interleaving's quality cost.** The closest result is Table 10, varying the
window at *inference* time on 9B: validation ppl **1.63 @ sw=4096, 1.63 @ sw=2048, 1.64 @
sw=1024** — *"moderate impact on perplexity."* Note also that Gemma 2's config has **no
`layer_types` field**; the 1:1 alternation is implicit in the modeling code.

**Gemma 3 (arXiv:2503.19786) — verified in detail.** **[VERIFIED]**
- **Ratio: 5:1 confirmed.** The architecture section is literally headed *"5:1 interleaving of
  local/global layers"*, described as *"a pattern of 5 local layers for every global layer,
  starting with a local layer as the first layer."* The intro phrases it as *"we have 1 global for
  every 5 local layers."*
- **Sliding window: 1024** — local layers get *"a smaller span of only 1024 tokens"*, down from
  Gemma 2's 4096. So Gemma 3 both **increased the local:global ratio 5×** and **shrank the window
  4×** simultaneously.
- **Context 128K.** RoPE base raised **from 10k to 1M for global layers**, while **local layers
  stay at 10k**; positional-interpolation rescaling with *"a scaling factor of 8."*
- **KV cache memory (§5.2, Fig. 5): global-only = 60% KV-cache memory overhead at 32K context;
  1:3 with sw=1024 = "less than 15%."** ⚠️ Note precisely: the prose gives numeric overheads only
  for **global-only** and **1:3/sw=1024** — **not** for 5:1 at 32K. Figure 6 plots the
  `(L:G=5:1, sw=1024)` 2B architecture against global-only over context length, but the curve
  values are image-only. **Do not cite a 5:1 percentage; cite 60% → <15% for 1:3.**
- **Perplexity impact is minimal, and this is the load-bearing finding.** On the ratio (Fig. 3):
  *"We observe minimal impact on perplexity when changing this ratio"*, and the caption goes
  further — *"The impact is minimal, even with 7-to-1 local to global."* On the window (Fig. 4):
  *"The sliding window can be reduced significantly without impacting perplexity."*
  ⚠️ No perplexity *values* are printed; they are in the figure images.

**[REASONING] Why this is the hardest number in the dossier for our proposal.** Gemma 3 reports
that going from 1 global layer in 2 to **1 global layer in 6** — a **3× cut in global-KV-bearing
layers, on top of a 4× window reduction** — costs *"minimal"* perplexity, and even 7:1 is fine.
Our CLA2 proposal delivers **2×** on the same axis (resident KV capacity). So a reviewer can
reasonably say: *sliding-window interleaving achieves a larger cache reduction, needs no
producer/consumer plumbing, is supported everywhere, and Google reports it as nearly free.* We
must have an answer. The honest answers are: (a) they are **composable, not exclusive** — Hymba
does both, and Hymba's ablation (row C → D) shows sharing still adds ~4.4% cache and +14.9%
throughput *after* SWA; (b) SWA **changes the model's function** (it removes long-range paths from
those layers, which is why Hymba lost >20% recall when it went all-SWA), whereas **CLA preserves
every layer's full global receptive field** — every attention layer still sees the whole context;
(c) sliding window's perplexity-neutrality was measured on perplexity, and §7 argues perplexity
is the wrong metric for the capability at issue. **(b) is the strongest and should be the paper's
framing: CLA is the cache reduction that does not shrink anyone's receptive field.**

### 2.2 Character.AI — production evidence

**"Optimizing AI Inference at Character.AI"** (Jun 20, 2024). ⚠️ **Citation hygiene: the original
`research.character.ai/optimizing-inference/` URL now 301-redirects to `blog.character.ai` and the
technical post is GONE from the live site. Cite the Wayback snapshot:**
`https://web.archive.org/web/20240726230425/https://research.character.ai/optimizing-inference/`

All six claims below are **[VERIFIED verbatim from the archived snapshot]**. This is the single
best *production* datapoint in the dossier, because it is the only source that stacks cross-layer
KV sharing, sliding-window interleaving, and int8 KV at scale (>20,000 queries/second).

Framing quote: *"The key bottleneck of LLM inference throughput is the size of the cache of
attention keys and values (KV). It not only determines the maximum batch size that can fit on a
GPU, but also dominates the I/O cost on attention layers."*

1. **Total reduction:** *"We use the following techniques to **reduce KV cache size by more than
   20X without regressing quality**. With these techniques, GPU memory is no longer a bottleneck
   for serving large batch sizes."*
2. **MQA:** *"We adopt Multi-Query Attention (Shazeer, 2019) in all attention layers. This
   **reduces KV cache size by 8X compared to the Grouped-Query Attention adopted in most open
   source models.**"* ⚠️ Note precisely: the 8× is **vs GQA, not vs MHA.**
3. **Hybrid attention horizons:** *"We interleave local attention (Beltagy et al., 2020) with
   global attention layers. … We found that **reducing attention horizon to 1024 on most attention
   layers does not have a significant impact on evaluation metrics, including the long context
   needle-in-haystack benchmark. In our production model, only 1 out of every 6 layers uses global
   attention.**"* **[REASONING] This is important twice over: it is the same 5:1 local:global with
   window 1024 that Gemma 3 arrived at ~9 months LATER and independently — and, uniquely in this
   literature, they explicitly state they checked **needle-in-a-haystack** and it held.**
4. **Cross-layer KV sharing:** *"**We tie the KV cache across neighboring attention layers, which
   further reduces KV cache size by a factor of 2-3x.** For global attention layers, we tie the KV
   cache of multiple global layers across blocks, since the global attention layers dominate the
   KV cache size under long context use cases. Similar to a recent publication (Brandon et al.,
   2024), **we find that sharing KV across layers does not regress quality.**"* Figure 1 caption:
   *"For global attention layers, we share KV across multiple non-adjacent layers."*
   **[REASONING] Three things here matter enormously for our proposal.** (i) They cite **Brandon et
   al. (CLA)** — so this is a production deployment of our exact mechanism. (ii) They **do** share
   between **global** attention layers, which is precisely what Hymba declined to do and what we
   propose — a direct precedent in our favour. (iii) They share across **non-adjacent** layers and
   report no quality regression, which is **contrary to CLA's own DenseBack finding** (+0.43 ppl).
   That contradiction is unresolved in the literature and is worth flagging in the paper: a
   production system reports non-adjacent global-layer sharing works, while CLA's ablation says
   non-uniform pairing loses. This is direct support for testing arm **A2** (§10.1).
5. **int8:** *"We use **int8 quantization on model weights, activations, and attention KV cache.**
   … Different from commonly adopted 'post-training quantization' techniques, **we natively train
   our models in int8 precision, eliminating the risk of training/serving mismatch**."*
6. **Economics:** *"We have **reduced serving costs by a factor of 33** compared to when we began
   in late 2022."* … *"it would cost at least **13.5X more**"* via commercial APIs. Stateful
   inter-turn caching (LRU tree keyed by rolling hash of prefix tokens, RadixAttention-like,
   sticky sessions) achieves *"a **95% cache rate**"*; *"Since our KV cache size is small, each
   server can cache thousands of dialogues concurrently."* Average message has 180 messages of
   dialogue history.

**Follow-up: "Optimizing AI Inference at Character.AI (Part Deux)" (Nov 21, 2024)**
(`https://web.archive.org/web/20241126230517/https://research.character.ai/optimizing-ai-inference-at-character-ai-part-deux/`).
About **speed, not cache size**: a custom int8 FlashAttention-3 fork with a "half int8" design
(first matmul on int8 tensor cores, second on bf16, chosen *"to avoid potential regression in model
quality"*) plus query-head packing for MQA decoding. **Up to 10% prefill and 30% decoding
speedup** over a Triton baseline; query-head packing alone cuts decode kernel time *"up to 9.3x."*
Adds no new KV-sharing numbers.

**[REASONING] Bottom line: item 4 upgrades Character.AI from "supporting evidence" to a second
piece of prior art.** Together with Hymba, cross-layer KV sharing in a local/global-interleaved
model is **deployed in production at 20k QPS with a CLA citation**. Our novelty claim must be
scoped accordingly (§4.3) — but note that Character.AI is a *transformer*, not a conv hybrid, and
publishes no ablation isolating the sharing, so the hybrid and retrieval questions remain open.

### 2.3 StreamingLLM — attention sinks

**"Efficient Streaming Language Models with Attention Sinks" (arXiv:2309.17453, ICLR 2024).**
**[VERIFIED from HTML]**
- **Mechanism:** keep the KV of a few **initial "sink" tokens** plus a rolling local window.
  The insight is that large attention mass accumulates on the first tokens *"even if they are not
  semantically important"*, and *"keeping the KV of initial tokens will largely recover the
  performance of window attention."*
- **4 sink tokens**, confirmed by ablation (Table 2). Llama-2-7B, PPL by cache config:
  **0+4096 → 3359.95**; **1+4095 → 11.88**; **2+4094 → 10.51**; **4+4092 → 9.59**;
  **8+4088 → 9.54.** Conclusion: *"a threshold of four initial tokens appears enough, with
  subsequent additions contributing marginal effects."* (MPT-7B is even more dramatic:
  0+2048 → 460.29 vs 1+2047 → 14.99.)
- **Stable LM to 4M+ tokens**; **up to 22.2× per-token speedup** vs the sliding-window-**with
  recomputation** baseline (the only baseline with acceptable quality), Llama-2-7B/13B on one
  A6000 — because StreamingLLM's per-token latency grows linearly in cache size while
  recomputation grows quadratically.
- **Dedicated learnable sink token, pretrained (Table 3, 160M):**
  Vanilla **27.87 / 18.49 / 18.05 / 18.05** at 0+1024 / 1+1023 / 2+1022 / 4+1020;
  Zero Sink **29214 / 19.90 / 18.27 / 18.01**; **Learnable Sink 1235 / 18.01 / 18.01 / 18.02.**
  So one learnable sink suffices where vanilla needs four. Convergence and zero-shot accuracy are
  essentially unchanged (ARC-e 45.2 → 45.6). A second sink token gave no further gain.
- **The distinction they insist on, which matters for us:** *"StreamingLLM efficiently generates
  coherent text from tokens within the KV cache without extending the LLMs' context length"* and
  *"it does not extend the models' context window or enhance their long-term memory
  capabilities"* — the model *"is limited to operating within the confines of its current cache."*
  Appendix C makes it concrete: with 4+2044, accuracy **75.30 at 80 lines (1840 tokens) → 0.00 at
  100 lines (2300 tokens)** once the answer falls outside the cache.

**[REASONING] Two consequences.** (1) StreamingLLM is a **cache-size** method, not a
context-length method, and its own retrieval-style appendix shows the catastrophic cliff when the
needle leaves the window — a clean illustration of why **retrieval evaluation is the metric that
distinguishes cache methods** (§7), since its perplexity looks fine throughout. (2) The sink
phenomenon is a warning about CLA specifically: if the first tokens carry disproportionate
attention mass, then **which layer produces the K/V for those positions is not a neutral choice**.
A consumer borrowing a producer's sink keys inherits the producer's sink structure. Nobody has
looked at this. **[GAP]** Cheap diagnostic: compare attention-mass-on-first-4-tokens between
producer and consumer layers in a trained CLA model.

### 2.4 Is sliding-window attention redundant with a conv layer in a mostly-LIV model?

This is the reviewer question the brief flags, and it deserves a direct answer. **The answer is
no — they are not redundant — and there is good evidence for that, chiefly from Samba.**

**Samba (arXiv:2406.07522) is the right experiment to cite.** **[VERIFIED from HTML]**
- **Pattern:** Mamba+MLP alternating with SWA+MLP — *"SWA at odd indices"* of 12 blocks; at 1.7B
  that is **48 layers, 24 attention + 24 Mamba**, i.e. **1:1**. **Window w = 2048**, RoPE applied
  inside the window; *"the optimal ratio of sequence length/window size observed is 2"* (train at
  4096).
- **The division of labour is stated explicitly**, and it is exactly the framing our paper needs:
  *"Mamba as the capture of recurrent sequence structures, SWA as the precise retrieval of memory,
  and MLP as the recall of factual knowledge."* And: since *"Mamba can already capture low-rank
  information in the sequences through recurrent compression"*, attention *"will only need to
  focus on information retrieval where a small number of attention heads should suffice"* —
  corroborated by Samba needing a **2× smaller optimal number of query heads** than an SWA-only
  model. An entropy analysis adds that given attention's recall, *"the Mamba layers can focus more
  on modeling the recurrent structure rather than performing retrieval."*
- **Retrieval is where the difference shows.** Table 2 (1.7B class, 230B tokens), **SQuAD**:
  pure Mamba 1.8B **67.66**, Mamba-MLP 1.9B **63.86**, **Samba 1.7B 77.64**, Llama-3 1.6B 74.88.
  The paper notes pure Mamba *"fall[s] short on retrieval intensive tasks such as SQuAD."*
  Average over 15 benchmarks: Samba **54.33**, Mamba-SWA-MLP 53.77, Mamba 52.31,
  Mamba-MLP 51.38, Llama-3 51.17.
- **The direct redundancy test exists — Samba's Table 8, "+SC" (short convolution added).**
  SlimPajama ppl at 4K/8K/16K:
  Llama-2-SWA 438M **11.12/10.66/10.57 → +SC 10.83/10.39/10.31**;
  Sliding GLA **10.43/10.00/9.92 → +SC 10.39/9.96/9.87**;
  Sliding RetNet **10.38/9.96/9.87 → +SC 10.25/9.82/9.74**.
  **Adding a short conv to a sliding-window-attention model IMPROVES it** — so the two operators
  are **not** substitutes; the conv adds something SWA lacks. And the reverse holds too: none of
  these +SC variants matches Samba (421M: 10.06/9.65/9.57), so SWA-plus-conv ≠ Mamba-plus-SWA.
  ⚠️ One caveat the paper flags honestly: *"adding SC to both the SWA and the linear attention
  layers in hybrid models produces negative results"* — left as future work. So the composition is
  beneficial but not unconditionally.

**[REASONING] The answer to give the reviewer, in four steps.**

1. **A short conv and a sliding-window attention layer are both local, but they are different
   operators.** LFM2's LIV block is a **depthwise, k=3, input-independent** convolution (a fixed
   learned kernel per channel, modulated by two linear gates). SWA is **content-based,
   input-dependent** matching over a window of 1024–4096. The conv's receptive field is **3
   tokens**; SWA's is 300–1300× larger. Calling them redundant conflates "local" with "the same".
2. **The empirical evidence says they compose, not substitute.** Samba's Table 8 shows adding a
   short conv to an SWA model *improves* perplexity at every length tested, and Samba's own design
   deliberately keeps **both** Mamba's internal `k=4` short conv **and** SWA.
3. **But the substitution question is genuinely different in LFM2 than in Samba**, and this is
   where honesty is required. **[REASONING]** Samba's local operator is a **Mamba** layer — a
   selective SSM with a *global*, input-dependent recurrent state, not merely a short conv. LFM2's
   LIV block is *only* a short conv with a **3-token** receptive field and **no recurrent state
   spanning the sequence**. So in LFM2 the mid-range band — beyond 3 tokens, below full context —
   is covered **only** by the 6 full-attention layers. Replacing any of them with SWA would leave
   that band covered by SWA instead; replacing them with *conv* would leave a genuine gap. This
   makes LFM2 **more** dependent on its attention layers than Samba is, and therefore **more
   vulnerable to anything that degrades them — including CLA.** That is an argument for the
   experiment's importance, and simultaneously a warning about its likely outcome.
4. **The strongest supporting fact is from LFM2 itself.** Per this project's own architecture
   dossier (`01_lfm2_architecture.md`), LFM2's STAR architecture search space **included
   sliding-window attention as a candidate operator**, and the released models use
   `full_attention` **exclusively** — no released LFM2 config contains a sliding window.
   **[REASONING] So the designers of this exact architecture had SWA available, searched over it,
   and chose full attention with a short-conv companion instead.** That is the cleanest possible
   rebuttal of "just use sliding window": in this architecture family, it was tried and not
   selected. (Caveat: STAR optimized proxy objectives — quality and cache size — on specific
   target hardware, so this is evidence about their objective, not a universal claim.)

**Three further sources, all pointing the same way.** **[VERIFIED]**

- **Based (arXiv:2402.18668, "Simple linear attention language models balance the recall-throughput
  tradeoff")** is the most explicit statement that conv and local attention are complementary
  *local* mixers. It uses surprisingly tiny windows — *"softmax attention in surprisingly small
  sliding windows (e.g., 64–128 tokens) that recover **90.8% of full softmax attention's recall
  accuracy at 1e-5× its latency**"* — and then says directly: *"including gated convolution layers
  with short convolutions (e.g., **filter size 3**) gives additional benefit over only using
  tcWindow layers"*, because *"Short convolutions can help perform local, precise shifts for token
  comparisons since they operate over the full sequence, while tcWindow does not… **These local
  mixers can complement one-another.**"* Its final mixture is ≈20% linear attention / 20% sliding
  window / **60% gated convolution**. **[REASONING] Note the filter size is 3 — exactly LFM2's k.
  This is the closest thing in the literature to a direct statement that a k=3 conv and a local
  attention window do different jobs.**
- **Griffin / Hawk (arXiv:2402.19427)** uses local attention with window **1024** in a **2:1
  recurrent:local-attention** pattern, framing it as *"local attention accurately models the recent
  past, while the recurrent layers can transmit information across long sequences."* Its Conv1D
  (temporal filter dim 4) lives *inside* the recurrent block and is never compared against local
  attention. RecurrentGemma (arXiv:2404.07839) uses window **2048** with no ablations.
- **DeltaNet (arXiv:2406.06484)** gives the nearest quantitative near-neutrality result. At
  340M/15B tokens (Wikitext ppl / LAMBADA ppl / commonsense avg): DeltaNet **w/o conv 29.08 /
  50.87 / 41.3**; **w/ conv 28.24 / 37.37 / 42.1**; **+Sliding Attn 27.06 / 38.17 / 42.1**;
  **+Global Attn (2 layers) 27.51 / 35.04 / 42.1.** At 1.3B/100B the two hybrids are within noise.
  But on **recall (SWDE, 1.3B): conv-only 49.5, +Sliding Attn 53.3, +Global Attn 71.0.**
  **[REASONING] This is the single most useful number in this subsection for our purposes: sliding
  attention recovers only ~19% of the conv→global recall gap (49.5 → 53.3 of a possible 49.5 →
  71.0). Local attention is NOT a substitute for global attention on retrieval — and retrieval is
  exactly our metric of concern (§7). It also shows conv is not free: removing it costs 0.84
  Wikitext ppl and 13.5 LAMBADA ppl even with attention present.**

**A caution — Hymba's redundancy claim runs the OTHER way.** **[VERIFIED]** Hymba argues its SSM
heads are redundant with *global* attention, which licenses *more local* attention:
*"with the presence of SSM heads in our hybrid-head module, which already summarize global context,
we can more aggressively replace global full attention with local attention."* Its numbers
(General / Recall / tok/s / cache MB): all-SWA **44.42 / 29.78 / 4485.09 / 5.51**; all-SWA + 3
full-attn **44.56 / 48.79 / 2399.7 / 41.19**; all-global + GQA **45.19 / 49.90 / 876.7 / 148.24**.
**[REASONING] Note the recall column: 29.78 for all-SWA vs 48.79 with just 3 global layers — a
19-point recall collapse for a 7.5× cache saving. This is the strongest single argument against
answering our cache problem with sliding windows, and it comes from the SWA advocates themselves.**
The asymmetry with LFM2 is that Hymba's "already summarize global context" premise rests on
**SSM** heads with a global recurrent state — which LFM2's k=3 conv does **not** have (§2.4 step 3).

**Conclusion for the paper.** State plainly: *a k=3 depthwise conv is a local operator but not a
substitute for a 1024-token content-based window. No primary source claims a conv ↔ SWA swap is
quality-neutral; the only direct evidence (Samba Table 8) shows short conv is **additive on top of
SWA** (11.12 → 10.83 at 4K) while being nearly redundant with GLA's channel-level decay, and Based
composes both deliberately. LFM2's own architecture search selected full attention over the
sliding-window option it had available. Sliding-window interleaving nonetheless remains the
strongest competing cache reduction and is a mandatory control (§10.2), because it cuts more cache
than CLA2 does — but DeltaNet's SWDE numbers (53.3 vs 71.0) and Hymba's all-SWA recall (29.78 vs
48.79) show it pays for that in retrieval, which is precisely the axis CLA leaves untouched.*
**That contrast — SWA buys cache by shrinking receptive fields, CLA buys cache without touching
them — is the paper's cleanest positioning statement.**

---

## 3. Post-hoc eviction and quantization — positioning only

These are **orthogonal** to CLA and are included only so the paper can say why they are not
competitors. Deliberately brief.

| Method | Mechanism (one line) | Reported compression |
|---|---|---|
| **H2O** (arXiv:2306.14048, *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"*) | Keep a small set of "heavy hitter" tokens (high accumulated attention score) plus recent tokens; greedily evict the rest at decode time. | Commonly cited at **~20% of full KV** with comparable quality; reports large throughput gains over then-current systems. **[UNVERIFIED numbers — not re-checked this pass]** |
| **SnapKV** (arXiv:2404.14469, *"SnapKV: LLM Knows What You are Looking for Before Generation"*) | At the **end of prefill**, use the observation window of the last prompt tokens to vote for important prefix positions per head, then keep only those — a one-shot prompt-time selection, not per-step eviction. | ~**adaptive per-head selection**, large prompt-cache reduction with near-lossless long-context accuracy. **[UNVERIFIED]** |
| **PyramidKV** (arXiv:2406.02069, *"PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"*) | Allocate a **non-uniform, layer-wise budget** — more cache to lower layers, less to upper layers, since attention concentrates as depth increases. | Retains performance with roughly **12%** of the KV cache; strong at very small budgets. **[UNVERIFIED]** |
| **KIVI** (arXiv:2402.02750, *"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"*) | **Asymmetric** quantization — keys **per-channel**, values **per-token** — because key outliers are channel-structured; tuning-free, so no retraining. | **2-bit** KV (int2), reported **~2.6× peak memory reduction and ~2.35–3.47× throughput**; int4 variants are easier still. **[UNVERIFIED]** |

⚠️ I did not re-verify this table's numbers against primary sources this pass — the brief asked for
one line each for positioning, and re-verifying four papers was a poor use of the budget. **Every
number in this table should be checked before it appears in a paper.** Mechanisms are stated
confidently; compression figures are flagged.

### 3.1 Why they compose with CLA rather than compete

**[REASONING]** The composition argument has three independent legs, and it is worth making
explicitly because it converts a potential "why not just use SnapKV" objection into a strength:

1. **They act on different axes of the same tensor.** A KV cache is indexed by
   **(layer, token, head, channel, precision)**. CLA reduces the **layer** extent. H2O/SnapKV/
   PyramidKV reduce the **token** extent. GQA/MQA/MLKV reduce the **head** extent. MLA reduces the
   **channel** extent. KIVI reduces the **precision** extent. Reductions on different axes
   **multiply**. Concretely: CLA2 (2×) × KIVI-int4 from fp16 (4×) = 8× resident reduction, and
   neither mechanism knows the other exists.
2. **They act at different times in the model lifecycle.** CLA is a **pretraining-time
   architectural** choice (§5.1.2 of the arXiv:2412.19442 taxonomy) that changes the parameter
   count and must be trained in. H2O/SnapKV/PyramidKV/KIVI are **inference-time, training-free**
   post-hoc methods applied to any checkpoint. You cannot substitute one for the other: a
   post-hoc method cannot recover the parameters and prefill compute that CLA never spends, and
   CLA cannot adapt per-prompt the way SnapKV does.
3. **There is direct empirical evidence of composition, not just an argument.** KVSharer
   (arXiv:2410.18517) — itself a cross-layer method — reports stacking with token-level methods:
   Llama2-13B-Chat goes **51,639 MB / 18.2 tok/s** → KVSharer-25% **37,049 MB / 30.0 tok/s
   (1.65×)** → **+H2O** 30,891 MB / 1.55× → **+PyramidInfer** 30,141 MB (58%) / **1.87×**. Its
   authors state the method is orthogonal to intra-layer compression. Similarly **SwiftKV**
   composes AcrossKV with **FP8 KV quantization** (2-way + FP8 → 62.5% reduction; 4-way + FP8 →
   68.75%), and **CLLA** (arXiv:2410.15252) and **CommonKV** (arXiv:2508.16134) both *define*
   themselves as unions of layer sharing + head/dim reduction + quantization, reaching **<2%** and
   **98% compression** respectively.

**One caveat to state honestly.** **[REASONING]** Composition is not perfectly free in one case:
token-eviction methods make per-layer decisions (PyramidKV explicitly allocates *different budgets
per layer*). Under CLA, a producer's bank is read by two layers with **different** query
distributions, so a per-layer eviction policy no longer has a well-defined single owner — you must
either evict on the union of both layers' importance scores (less compression) or on the producer's
alone (risking eviction of tokens the consumer needed). **PyramidKV's layer-wise budget allocation
is therefore in genuine tension with layer sharing**, since the pyramid assumes each layer has its
own cache to size. I found no paper addressing this interaction. **[GAP]** — worth one sentence in
the paper's limitations, and it is a legitimate future-work item.

---

## 4. Hybrid-specific interaction — the novelty question

This is the section that decides whether the experiment has a novelty claim. I checked every
major conv/SSM/linear + attention hybrid for **cross-layer KV sharing** (reusing another
layer's K/V tensors), which is distinct from **weight sharing** (reusing another layer's
parameters) and from **intra-layer head sharing** (GQA/MQA).

### 4.1 Per-model findings

All layer counts and ratios below were read directly from the released `config.json` on the
HuggingFace Hub (fetched 2026-07-30) or from the paper, and are therefore **[VERIFIED]** unless
marked otherwise.

| Model | Total layers | Attention layers | attn : linear ratio | Cross-layer KV sharing? |
|---|---|---|---|---|
| **Hymba-1.5B** (arXiv:2411.13676) | 32 (all "hybrid-head") | 32 attn heads-in-parallel; **3 global** at `[0,15,31]`, rest SWA-1024 | parallel, not interleaved | **YES — explicit, cites CLA** |
| **Jamba** (arXiv:2403.19887) | 32 (4 blocks × l=8) | 4 (`a:m = 1:7`) | 1 : 7 | **No** |
| **Samba** (arXiv:2406.07522) | Mamba + SWA + MLP interleave | SWA layers | 1 : 1 (Mamba : SWA) | **No** |
| **Zamba2-2.7B** (arXiv:2411.15242) | 54 | 9 "hybrid" at `[6,12,18,24,30,36,42,47,51]`, 45 mamba | 1 : 5 | **No** — shares *weights* (`num_mem_blocks=2`), not KV |
| **Nemotron-H-8B** (arXiv:2504.03624) | 52 | **4** (`*` at pattern idx `[7,18,29,40]`); 24 Mamba2 + 24 MLP | 4 : 24 Mamba (1 : 6) | **No** |
| **Falcon-H1-1.5B** | 24 | parallel Mamba2 ∥ attention in every block (`attn_layer_indices=null`) | parallel | **No** |
| **Granite-4.0-h-small** | 40 | **4** attention at `[5,15,25,35]`, 36 mamba | 1 : 9 | **No**; notable: `position_embedding_type="nope"` |
| **MiniMax-01** (arXiv:2501.08313) | 80 | 1 softmax per 8 blocks | 1 : 7 lightning | **No** |
| **Qwen3-Next-80B-A3B** | 48 | `full_attention_interval=4` → 12 full-attn, 36 gated DeltaNet | 1 : 3 | **No** |
| **Kimi-Linear-48B-A3B** | 27 | `full_attn_layers=[4,8,12,16,20,24,27]` (7, MLA `kv_lora_rank=512`), 20 KDA | ~1 : 3 | **No** — uses **MLA** instead |

Supporting detail on the two that matter most:

**Hymba — the one genuine precedent. [VERIFIED against code + paper + config]**
- Paper: *Hymba: A Hybrid-head Architecture for Small Language Models* (arXiv:2411.13676, NVIDIA).
- The abstract itself says the model is *"further optimized by incorporating cross-layer
  key-value (KV) sharing and partial sliding window attention, resulting in a compact cache
  size."*
- It **cites Brandon et al. (CLA, arXiv:2405.12981) directly** for the technique, and cites
  MiniCache for the motivating observation that *"KV cache shares a high similarity between
  adjacent layers"*, plus CLA for *"consecutive layers have a high correlation in the KV cache."*
- Mechanism, from `config.json` of `nvidia/Hymba-1.5B-Base`:
  ```
  num_hidden_layers = 32,  hidden_size = 1600
  num_attention_heads = 25,  num_key_value_heads = 5,  sliding_window = 1024
  global_attn_idx  = [0, 15, 31]
  kv_reuse_group   = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],
                      [16,17,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30]]
  kv_weight_reuse  = false
  ```
  Note carefully: the pairing is **strictly adjacent** (`[1,2]`, `[3,4]`, …), exactly CLA's
  recommendation, and the **3 global-attention layers `[0,15,31]` are excluded from every
  sharing group** — the global layers each keep their own bank. One group, `[16,17,18]`, is a
  **3-way** share (sharing factor 3) used to absorb the odd layer count around the middle
  global layer.
- The released `modeling_hymba.py` is a **working reference implementation** of CLA-in-a-hybrid.
  The construction loop is:
  ```python
  self.kv_reuse_group = [{'producer': group[0], 'consumer': group[1:]} for group in self.kv_reuse_group]
  ...
  for i in range(config.num_hidden_layers):
      reuse_kv = any(i in item['consumer'] for item in self.kv_reuse_group)
      decoder_layer = HymbaDecoderLayer(config, num_experts=1, layer_idx=i, reuse_kv=reuse_kv)
  ```
  and inside the attention module the consumer literally skips its own K/V:
  ```python
  if self.reuse_kv:
      assert kv_last_layer is not None
      key_states, value_states = kv_last_layer   # (batch, num_heads, slen, head_dim)
  else:
      key_states  = self.k_proj(hidden_states)
      value_states = self.v_proj(hidden_states)
      ...
      _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)
  ```
  Two implementation facts fall straight out of this and answer open design questions for us:
  1. **The consumer allocates no K/V projections at all** — `if not self.attn_only_wo_proj and
     not self.reuse_kv:` guards the `k_proj`/`v_proj` `nn.Linear` construction, so the weights
     genuinely do not exist. Q is still projected per-layer (`query_states = self.q_proj(...)`,
     unguarded by `reuse_kv`).
  2. **Hymba shares POST-rotary keys.** The producer applies
     `apply_rotary_pos_emb(None, key_states, cos, sin)` *before* the tensor is handed on as
     `key_states_no_repeat`, and the consumer consumes that already-rotated tensor. The consumer
     applies RoPE only to its **own Q** (`query_states, _ = apply_rotary_pos_emb(query_states,
     None, cos, sin)`). Since producer and consumer see the same `position_ids`, this is
     consistent — see §10.3 for why this matters and when it would not be.
  3. **The consumer also does not write to the cache**: every cache-update path is guarded
     `if past_key_value is not None and use_cache and not self.reuse_kv`. So KV write bytes and
     resident bytes both fall; read bytes do not. This is exactly CLA's capacity-not-bandwidth
     property, now confirmed in a real hybrid implementation.
- **Hymba's reported numbers** (Table 1 "roadmap" ablation, 300M params / 100B tokens; columns
  are Commonsense % | Recall % | Throughput tok/s | Cache MB):

  | Row | Commonsense | Recall | tok/s | Cache MB |
  |---|---|---|---|---|
  | Transformer (Llama) | 44.08 | 39.98 | 721.1 | 414.7 |
  | SSM (Mamba) | 42.98 | 19.23 | 4720.8 | 1.9 |
  | A. + Attention heads (sequential) | 44.07 | 45.16 | 776.3 | 156.3 |
  | B. + Multi-head structure (parallel) | 45.19 | 49.90 | 876.7 | 148.2 |
  | C. + Local / global attention | 44.56 | 48.79 | 2399.7 | 41.2 |
  | **D. + KV cache sharing** | **45.16** | **48.04** | **2756.5** | **39.4** |
  | E. + Meta tokens | 45.59 | 51.79 | 2695.8 | 40.0 |

  **Read row C → row D very carefully, because it is the single most relevant data point in the
  entire literature for this proposal:** adding cross-layer KV sharing moved commonsense
  **+0.60 (44.56 → 45.16)** and throughput **+14.9% (2399.7 → 2756.5)** — but moved
  **recall −0.75 (48.79 → 48.04)**. The paper's own summary of the row is that it *"improves
  throughput by 1.15× while maintaining comparable recall accuracy and boosting commonsense
  accuracy by +0.60%."* **[REASONING]** The direction of that split is precisely the failure
  mode this brainlift worries about: sharing helped the aggregate/commonsense metric and *hurt*
  the recall metric. It is a 2-task recall average at 300M/100B tokens, so it is weak evidence,
  but it is evidence, and it points the wrong way for retrieval. See §7.
- Also note the cache MB delta is small (41.2 → 39.4, only **4.4%**) because rows C already cut
  the cache 3.8× with SWA — **once you have gone local/global, cross-layer sharing has little
  cache left to remove.** That is a direct argument that SWA and CLA are substantially
  *substitutes* on the capacity axis, not complements. **[REASONING]**
- Headline model comparisons: Hymba-1.5B has **79 MB cache / 664 tok/s** (8K seq, batch 128,
  A100, FP16 cache), vs Llama-3.2-3B at 918 MB / 191 tok/s → *"1.32% higher average accuracy,
  an 11.67× cache size reduction, and 3.49× throughput"*; vs SmolLM2-1.7B at 1573 MB / 238 tok/s
  → *"1.02% average accuracy improvement, a 19.91× cache size reduction, and 2.79× throughput."*
  Its local/global + cross-layer sharing combination together *"improves throughput by 3× and
  reduces cache by almost 4×."* Note the 11.67× is against a *different, larger* model, so it
  bundles the SSM-heads, SWA, and KV-sharing effects — it is **not** a KV-sharing ablation.
- Global-attention finding worth carrying: making everything SWA dropped *"recall accuracy by
  over 20% on recall-intensive tasks"*, and restoring global attention *"in just three layers
  (i.e., the first, middle, and last layers) is sufficient to recover recall-intensive
  accuracy."*

**Kimi Linear — the "why not MLA" precedent. [VERIFIED from config]**
`moonshotai/Kimi-Linear-48B-A3B-Instruct`: 27 layers, `linear_attn_config.full_attn_layers =
[4,8,12,16,20,24,27]` (7 full-attention layers) with 20 KDA layers, i.e. a **3:1 KDA:full**
ratio; `kv_lora_rank = 512`, `q_lora_rank = null`, `mla_use_nope` present in the config. So a
2025 frontier hybrid chose **MLA for its full-attention layers and did no cross-layer sharing
at all**. Any reviewer asking "why not MLA" is pointing at this model. See §1.4.

**Granite 4.0** deserves one extra note: `position_embedding_type = "nope"` — no positional
encoding on the attention layers at all, with the Mamba2 layers carrying position information.
**[REASONING]** That is an existence proof that in a mostly-linear hybrid the attention layers'
positional encoding can be dispensed with, which is relevant to §10.3: if RoPE can be removed
entirely, the pre/post-rotary sharing question partly dissolves.

### 4.2 Explicit prior-art search on the exact combination

I searched for the specific combination (cross-layer KV sharing × conv/SSM hybrid). Findings:

- **Hymba is the only hit**, and it is a real one. It is cross-layer KV sharing inside a
  hybrid, adjacent-paired, CLA-cited, with a released implementation.
- No paper found that studies cross-layer KV sharing where the *intervening* layer is a
  conv/SSM block in a **sequentially interleaved** stack. **[GAP]**
- No paper found that searches or learns *which* attention layers to pair in a hybrid. **[GAP]**

### 4.3 NOVELTY VERDICT

**Verdict: PARTIALLY ANTICIPATED — by Hymba, and (see §2.2) by Character.AI in production.** The
bald claim "apply CLA to the attention layers of a hybrid" is **not novel**; Hymba
(arXiv:2411.13676) did it in Nov 2024, cited CLA for it, shipped weights and code, and reported an
ablation row. Independently, **Character.AI deployed cross-layer KV sharing in a local/global
interleaved production model at >20k QPS, explicitly citing Brandon et al., and — unlike Hymba —
they DID share between global attention layers** (*"we tie the KV cache of multiple global layers
across blocks"*), reporting *"does not regress quality."* And **Gemma 3n** shipped a
one-producer-many-consumers variant into HF transformers and vLLM (§5.1). Writing the proposal as
if CLA-in-a-hybrid, or CLA-between-global-layers, were unexplored would be a straightforward
novelty-overclaim, and a reviewer who knows any of these three would reject it.

**A useful unresolved contradiction to exploit rather than hide.** **[REASONING]** CLA's own
ablation says non-uniform/non-adjacent pairing loses (`DenseBack` +0.43 ppl); Character.AI reports
sharing across **non-adjacent** global layers with no quality regression; Gemma 3n ships 13-way and
4-way non-adjacent sharing. These cannot all be straightforwardly true. Nobody has reconciled them,
and none of the three production sources publishes a controlled ablation. Our arm **A1 vs A2**
(§10.1) is a direct, controlled test of exactly this disagreement — which is a stronger framing
than "we apply CLA to a hybrid."

What **is** still open, and where a defensible contribution lives:

1. **Architecture shape differs in a way that changes the mechanism.** Hymba is a **parallel
   hybrid-head** model: attention heads and SSM heads sit *inside the same layer*, processing
   the same input in parallel. So Hymba's "adjacent layers" `[1,2]` are adjacent *hybrid* layers
   — the producer's K/V is consumed by the very next layer, with only that layer's own norm and
   residual in between. LFM2 is a **sequentially interleaved** hybrid: between attention layer 8
   and attention layer 10 sits a *complete gated-short-conv block* that rewrites the residual
   stream. Sharing K/V across an intervening sequence-mixing block is structurally a different
   ask, and **nobody has measured it**. **[GAP]**
2. **Hymba shares between SWA layers; LFM2's attention layers are full attention.** Hymba
   explicitly excludes its 3 global layers from sharing. Our proposal would share between
   **full-attention** layers — the opposite of what Hymba validated. Hymba's config proves the
   choice but not the reason; it gives no ablation isolating "share the global layers too."
   **[GAP]** ⚠️ **Partial counter-evidence: Character.AI does exactly this** (*"we tie the KV cache
   of multiple global layers across blocks, since the global attention layers dominate the KV cache
   size under long context use cases"*) and reports no regression — but publishes **no controlled
   ablation and no numbers**, so it is an existence claim, not a measurement. The controlled
   comparison "share global vs share local" remains unpublished.
3. **GQA-8 vs Hymba's GQA-5 and CLA's MQA preference.** CLA found GQA+CLA2 mostly *lost*.
   Hymba runs `num_key_value_heads=5` with `num_attention_heads=25` (5:1). LFM2 is 8 KV heads at
   4:1. Whether CLA2 pays off at LFM2's specific GQA ratio is unmeasured. **[GAP]**
4. **The retrieval question is unanswered by everyone, including Hymba** (its recall went *down*
   0.75 on a 2-task average and it did not investigate). See §7.

**Recommended framing.** Do **not** claim "first to combine CLA with a hybrid." Claim instead:
*"Cross-layer KV sharing is established practice: Hymba applies it between adjacent sliding-window
layers of a parallel hybrid-head model, Character.AI deploys it across non-adjacent global layers in
production, and Gemma 3n ships it upstream in vLLM and HF. Yet these three sources disagree about
pairing (adjacent vs non-adjacent), none shares across an intervening sequence-mixing block in a
sequentially interleaved hybrid, and **not one of them — nor any paper in the cross-layer sharing
literature — reports an exact-retrieval evaluation.** We supply the controlled ablation and the
missing evaluation, in the architecture where the mechanism is most suspect."* That framing is
honest, cites all the precedent, and turns the crowded prior art into the motivation.

---

## 5. Implementation reality — what exists today, and what breaks

**Bottom line up front: cross-layer KV sharing is already supported first-class in both HF
transformers and vLLM**, because Google shipped Gemma 3n with it in mid-2025. This is very good
news for the experiment: it means the proposal does not need novel serving infrastructure, and
there are two independent reference implementations to copy conventions from.

### 5.1 HuggingFace transformers — YES, native, merged

Gemma 3n is a merged upstream implementation of cross-layer KV sharing.
Source: `src/transformers/models/gemma3n/configuration_gemma3n.py` and `modeling_gemma3n.py`
(read at `main`, 2026-07-30).

**Config surface.** `Gemma3nTextConfig` has:
- **`num_kv_shared_layers`, default `15`.** Docstring: *"The number of layers that share KV
  cache values."* and *"During the forward pass, the last `num_kv_shared_layers` layers in the
  model 'share' the KV values"* — each local and global layer in the trailing range reuses the
  KV computed by the **last local or global layer respectively** that preceded the range. The
  docstring notes the value *"should be a multiple of the attention pattern size."*
- **`layer_types`**, `list[str] | None`, resolved in `__post_init__` by the rule
  `(i + 1) % 5 == 0 → "full_attention"`, else `"sliding_attention"`. At the default
  `num_hidden_layers = 35` this gives `["sliding_attention"]*4 + ["full_attention"]` repeated 7×,
  so full attention at indices `[4, 9, 14, 19, 24, 29, 34]`. **[VERIFIED]**
- `sliding_window` default `512`; separate RoPE bases per layer type
  (`{"global": 1_000_000.0, "local": 10_000.0}`).

**[DERIVATION] Gemma 3n's resulting sharing topology**, computed from those rules:
`first_kv_shared_layer_idx = 35 − 15 = 20`. Producers are layer **18** (last sliding layer
before 20) and layer **19** (last full-attention layer before 20). Consumers: the 12 sliding
layers `[20,21,22,23,25,26,27,28,30,31,32,33]` all read layer 18's bank; the 3 full-attention
layers `[24,29,34]` all read layer 19's bank. So **20 of 35 layers hold a bank → 1.75×
reduction**, and the sharing factors are extreme and *non-uniform*: **13-way** on the local
producer and **4-way** on the global producer. Note this is a "one producer, many consumers,
all downstream" topology — much closer to **Layer-Condensed KV Cache** than to CLA's uniform
adjacent pairing, and it flatly contradicts CLA's finding that uniform adjacent pairing is best.
**[REASONING]** Either CLA's pairing conclusion does not transfer to this regime, or Gemma 3n is
paying a quality cost Google accepted for on-device memory; the Gemma 3n report does not, to my
knowledge, ablate this. **[GAP]**

**The three code facts worth copying.**
1. **The consumer allocates no K/V weights at all** — and neither their norms:
   ```python
   # Layers sharing kv states don't need any weight matrices
   if not self.is_kv_shared_layer:
       self.k_proj = nn.Linear(...); self.v_proj = nn.Linear(...)
       self.k_norm = Gemma3nRMSNorm(...); self.v_norm = Gemma3nRMSNorm(..., with_scale=False)
   ```
   `q_proj` and `q_norm` are constructed unconditionally. This matches CLA's "drop only K and V".
   They also register the absent weights so checkpoints load cleanly:
   ```python
   self._keys_to_ignore_on_load_unexpected.extend(
       [f"layers.{i}.self_attn.{name}" for name in ("k_proj","v_proj","k_norm","v_norm")])
   ```
2. **Sharing is POST-rotary.** The producer does
   `key_states = apply_rotary_pos_emb(key_states, cos, sin, ...)` *before* publishing, and the
   consumer takes the rotated tensor and applies RoPE only to its own Q. Same as Hymba (§4.1).
   **Two independent production implementations both share post-rotary K.** This is the single
   most useful empirical input to our RoPE decision (§10.3), and it is a convention, not a paper
   result — CLA itself is silent.
3. **A side channel, not the Cache object, carries the shared tensors.** The mechanism is a
   plain dict threaded through the layer loop:
   ```python
   shared_kv_states = UserDict()   # UserDict, not dict, for FSDP2 correctness
   for i, decoder_layer in enumerate(...):
       hidden_states = decoder_layer(..., shared_kv_states=shared_kv_states, past_key_values=...)
   ```
   and in the attention forward:
   ```python
   if self.is_kv_shared_layer:
       key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
       key_states = key_states.to(query_states.device)   # PP/device-split guard
       value_states = value_states.to(query_states.device)
   else:
       ... project, norm, rope ...
   if past_key_values is not None and not self.is_kv_shared_layer:
       key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
   if self.store_full_length_kv:
       shared_kv_states[self.layer_idx] = key_states, value_states
   ```
   **The `Cache` object itself has no concept of layer aliasing** — consumers simply never call
   `.update()`. **[REASONING]** So in HF, "the cache keyed by layer group" is not a Cache-level
   feature; it is the model's responsibility. Our implementation should do the same thing rather
   than trying to subclass `DynamicCache`.
   Note also the comment explaining *why* a side channel is needed rather than reading the
   consumer's own cache entry: *"We cannot simply reuse the cached state if we have a Cache, as
   sliding layers will not remember the full states in their Cache once we are past the sliding
   window."* That is a sliding-window-specific problem; **for full-attention producers, reading
   the producer's cache entry directly would be equivalent.** **[REASONING]**
4. Note the explicit `.to(query_states.device)` on the borrowed tensors — that is the
   device-placement hazard of §5.4 handled defensively, plus
   `_skip_keys_device_placement = ["past_key_values", "shared_kv_states"]`.

**Hymba** is not native — it is `trust_remote_code`, with `kv_reuse_group` handled inside its own
`modeling_hymba.py` (see §4.1 for the code). So transformers has **two** patterns available:
Gemma 3n's `num_kv_shared_layers` (trailing-block, one-producer-many-consumers) and Hymba's
`kv_reuse_group` (explicit list of `{producer, consumers}` groups). **For our proposal, Hymba's
`kv_reuse_group` list-of-groups is the right config shape** — it expresses arbitrary pairings
including non-adjacent ones, which is exactly what LFM2's `[2,5,8,10,12,14]` schedule needs.

### 5.2 vLLM — YES, native, threaded through every backend

The flag is **`kv_sharing_target_layer_name: str | None`** on `Attention`. It is not a
special-case hack; it is plumbed through the whole V1 stack. Confirmed present in (paths at
`main`, 2026-07-30):
- `vllm/model_executor/layers/attention/attention.py` (the `Attention` ctor)
- `vllm/v1/attention/backend.py` (base backend signature)
- every backend: `cpu_attn.py`, `rocm_attn.py`, `hpc_attn.py`, `mla/triton_mla.py`,
  `mla/cutlass_mla.py`, `mla/flashmla.py`, `mla/flashattn_mla_sparse.py`,
  `mla/aiter_triton_mla.py`
- `vllm/model_executor/layers/attention/chunked_local_attention.py` (so it composes with
  chunked local attention)
- `vllm/model_executor/models/gemma3n.py` — the consumer
- `vllm/v1/spec_decode/gemma4.py` and `vllm/v1/spec_decode/step3p5.py` — **speculative decoding
  paths explicitly manipulate it**: `attn.kv_sharing_target_layer_name = target_layer_name`
- `tests/v1/core/test_kv_sharing.py` — there is a dedicated test file

**How the cache is keyed by layer group — the actual mechanism.** Three steps:

1. **Consumers are excluded from spec generation, so no memory is budgeted for them**
   (`vllm/v1/worker/gpu/attn_utils.py`):
   ```python
   for layer_name, attn_module in attn_layers.items():
       if getattr(attn_module, "kv_sharing_target_layer_name", None):
           # This layer will use KV cache of the sharing target layer.
           continue
   ```
2. **Consumers are then appended to their producer's KV cache group** so attention metadata and
   block tables are assigned (`vllm/v1/worker/utils.py::add_kv_sharing_layers_to_kv_cache_groups`):
   ```python
   for layer_name, target_layer_name in shared_kv_cache_layers.items():
       tgt_kv_cache_group = layer_to_kv_cache_group[target_layer_name]
       tgt_kv_cache_group.layer_names.append(layer_name)
   ```
   Its docstring: *"Layer pairings for cross-layer KV sharing. If an Attention layer
   `layer_name` is in the keys of this dict, it means this layer will perform attention using
   the keys and values from the KV cache of `shared_kv_cache_layers[layer_name]`."*
3. **The tensor is literally aliased** — paged attention needs no changes at all, because the
   consumer receives the *same* tensor object and therefore the *same* block table:
   ```python
   for layer_name, target_layer_name in shared_kv_cache_layers.items():
       kv_caches[layer_name] = kv_caches[target_layer_name]
   ```
   with allocation skipped earlier via `if layer_name in shared_kv_cache_layers: continue`
   (comment: *"Shared layer — tensor will be aliased to its target later."*).

**[REASONING] This answers the "how does paged attention handle a shared bank" question
cleanly: it doesn't have to.** Because producer and consumer are in the same `KVCacheGroupSpec`,
they share one block table and one page mapping; the consumer's attention op reads the identical
paged tensor. There is no new allocator concept, no reference counting, no new page state. That
is a strong argument that the serving story for this proposal is low-risk.

vLLM's Gemma 3n wiring also shows the naming convention — the target is identified by **module
path string**, and only the consumer declares the relationship
(`# Only the greater layer is required to specify sharing.`):
```python
offset = 2 if self.sliding_window is not None else 1
kv_shared_layer_index = first_kv_shared_layer_idx - offset
kv_sharing_target_layer_name = f"{param_name_before_layers}.layers.{kv_shared_layer_index}.self_attn.attn"
```

### 5.3 SGLang — YES, and it is going further than vLLM

**[VERIFIED]** SGLang supports KV-shared models and has an **in-flight YOCO-style fast-prefill
optimization** built on top of that support. The clearest artifact is
**PR/issue sgl-project/sglang#27183, "perf(gemma4): YOCO fast-prefill for Gemma4 E2B/E4B"
(open as of this pass)**, which is worth quoting because it documents the mechanism precisely:

> *"Gemma4 E2B (35 layers / 20 KV-shared) and E4B (42 / 18) set `num_kv_shared_layers > 0`: the
> last K of N decoder layers reuse KV state from earlier layers (see
> `Gemma4Attention.is_kv_shared_layer` / `kv_shared_layer_index`) and do **not** write KV during
> prefill."*

The optimization's rationale is the important part for us:

> *"The baseline forward still runs the full Q-side compute — RMSNorm, Q-proj, RoPE, attention,
> MLP, residuals — for every prefill token in those shared-KV layers. But the only Q-side outputs
> that ever feed the LM head are the last-token-per-request rows … So all the per-token work in the
> cross-decoder layers except those rows is wasted. This is the 'You Only Cache Once' fast-prefill
> split (arXiv:2405.05254)."*

It adds a server flag **`--kv-sharing-fast-prefill`** (default **off**), a
`kv_sharing_fast_prefill: bool = False` field in `python/sglang/srt/server_args.py`, and
`_yoco_eligibility` / `_yoco_truncate_to_last_tokens` helpers in
`python/sglang/srt/models/gemma4_causal.py`.

**[REASONING] This is a materially useful finding for the proposal, beyond portability.** It shows
that **once a trailing block of layers is KV-shared, you can additionally skip their Q-side prefill
compute** — recovering YOCO's prefill win on a CLA-style architecture, at serving time, behind a
flag. That is an argument for choosing a *trailing-block* sharing topology (Gemma-3n-like) over
scattered pairs if prefill latency matters. **It does NOT apply to our A1 pairing**, because our
consumers (5, 10, 14) are interleaved with producers and later layers still need their outputs —
the trick only works when *all* layers after the sharing point are consumers. Worth one sentence
in the paper's discussion as a reason someone might prefer a different topology than ours.

SGLang also has related work in flight — e.g. **#32285 "[AMD] WIP-MiniMax-M3: support cross-layer
sparse index sharing"** and **#30923 "[DSA] Compact indexer K cache: drop slots for skip_topk
(shared) layers (+15.8% KV capacity on GLM-5.2)"** (closed) — indicating cross-layer sharing is an
active, maintained concern rather than a one-off.

### 5.3b llama.cpp — NOT CONFIRMED

**[UNVERIFIED / GAP]** Code and issue searches for cross-layer KV sharing symbols in
`ggml-org/llama.cpp` returned nothing this pass. Treat as **unknown**, not as absent — the search
was shallow (symbol-name based) and a differently-named implementation could exist. **[REASONING]**
The relevant question is whether llama.cpp's unified `llama_kv_cache` can alias per-layer views;
nothing blocks it in principle. This does not block the research, since training and evaluation run
in HF/PyTorch — but do not claim llama.cpp portability in the paper without checking.

### 5.4 What actually breaks

**Pipeline parallelism — confirmed by the CLA paper itself.** Quoting arXiv:2405.12981:
*"In the presence of pipeline parallelism (Huang et al., 2019), either different layers which
share a KV cache must be kept [in the same pipeline stage]"* — the alternative being to
transmit KV activations across pipeline stages. The same passage confirms CLA *"is fully
compatible with standard tensor parallelism techniques."*
**[REASONING] For our 16-layer model this is a non-issue in practice** — at 350M–1.2B we will
not use PP. But it constrains the pairing choice if anyone later scales it: a pairing that
straddles a natural PP boundary forces either an activation transfer or an unbalanced split.
Pairing (12,14) and (10,12) keeps producers/consumers close, which is PP-friendly; a "layer 2
produces for layer 14" pairing would be maximally PP-hostile. Worth one sentence in the paper.

**Device placement / layer-wise offloading.** Gemma 3n's code carries a live workaround:
`key_states.to(query_states.device)` on every borrowed tensor, plus
`_skip_keys_device_placement = ["past_key_values", "shared_kv_states"]`. **[REASONING] This is
direct evidence that naive `device_map="auto"` sharding across devices does interact with
cross-layer sharing** — the borrowed K/V may live on a different device than the consumer, and
you eat a cross-device copy per consumer per step. For per-layer CPU **offloading** specifically
the hazard is worse in kind: a scheme that evicts layer *p*'s bank after layer *p* executes is
simply **wrong** for CLA, because layer *c* still needs it. Offloading must become
group-aware — evict on the *last consumer* in the group, not on the producer. I found no paper
or issue discussing this. **[GAP]**

**Speculative decoding — it does interact, and vLLM has already had to handle it.** The
presence of `kv_sharing_target_layer_name` manipulation inside `vllm/v1/spec_decode/gemma4.py`
and `vllm/v1/spec_decode/step3p5.py` (`attn.kv_sharing_target_layer_name = target_layer_name`)
is concrete evidence that spec-decode paths must be made explicitly aware of the sharing map.
**[REASONING]** The reason is structural: on a rejected draft you must roll the cache back, and
with aliasing there is one physical bank behind several logical layers, so a naive per-layer
rollback would either double-rollback or miss. Also, a draft model whose layer count differs
cannot inherit the target's sharing map. I did not find a paper documenting this. **[GAP]**

**Prefix caching / radix caching — [REASONING], no citation found.** My analysis: prefix caching
should be *unaffected in its logic and improved in its economics*. The cache key is the token
prefix, not the layer; since producer and consumer share one block table, a reused prefix's
blocks are reused by both simultaneously. And because there are fewer resident banks, a fixed
HBM budget holds **more** cached prefixes — so prefix-cache hit rate should *rise*. I found no
source stating this; flagging as inference. **[GAP]**

**Chunked prefill — composes.** `chunked_local_attention.py` accepts
`kv_sharing_target_layer_name`, which is evidence the two features were designed to coexist.
**[REASONING]** The ordering invariant to respect is that within a chunk the producer must run
before its consumer — which is automatic for a forward pass where producer index < consumer
index. A pairing with producer *after* consumer (backward sharing) would break this and require
two passes; **this is an independent reason to only ever pair forward** (producer = lower layer
index), on top of CLA's `DenseBack` result.

**Tensor parallelism — fine, per CLA's own statement above.** K/V heads shard the same way for
producer and consumer since there is only one set of them.

### 5.5 The actual code change for our model — concrete

**[REASONING]** Synthesizing the two reference implementations, the change to `Lfm2Attention` /
`Lfm2Model` is small and well-precedented:

1. **Config:** add `kv_reuse_group: list[list[int]] | None` (Hymba's shape), e.g.
   `[[8,10],[12,14]]` or `[[2,5],[8,10],[12,14]]`. Validate that every index is in
   `full_attn_idxs`, that producer < consumer, and that no layer appears twice.
2. **`Lfm2Attention.__init__`:** take `reuse_kv: bool`; guard construction of `k_proj` and
   `v_proj` behind `if not reuse_kv`. **Keep `q_proj` and `out_proj` unconditional.** Add the
   `_keys_to_ignore_on_load_unexpected` entries.
3. **`Lfm2Attention.forward`:** accept `shared_kv_states: dict[int, tuple[Tensor, Tensor]]`.
   If `reuse_kv`, read `key_states, value_states = shared_kv_states[self.kv_producer_idx]` and
   skip both projection and cache update; else project, RoPE, `cache.update(...)`, and if this
   layer is a producer, publish into `shared_kv_states`.
4. **`Lfm2Model.forward`:** create `shared_kv_states = UserDict()` (not `dict` — FSDP2) and
   thread it through the decoder-layer loop, exactly as Gemma 3n does.
5. **Norms:** CLA uses *separately learnable affine LayerNorm params* for the KV block vs the Q
   block. LFM2 uses RMSNorm; Gemma 3n gives consumers no `k_norm`/`v_norm` at all. **Decision:
   follow CLA** — give the producer's KV path its own norm parameters — since CLA is the result
   we are reproducing.
6. Parameter savings per dropped pair, at d=2048, KV width 512: `2 × (2048 × 512) = 2.097M`
   params per consumer. **[DERIVATION]** For 3 consumers that is **6.29M params**, ~0.5% of a
   1.2B model — i.e. this is a *cache* optimization, not a parameter optimization, and the
   parameter delta must be compensated in any fair control (see §10.2).

---

## 7. Failure modes on RETRIEVAL specifically — the contribution gap

This is the brainlift's real worry: a consumer layer attending over K/V that were computed from a
*different* layer's residual stream may be reading **stale borrowed keys** — keys that encode
"what layer 8 thought token *j* was", queried by "what layer 14 wants to find". Aggregate
perplexity is dominated by high-frequency local prediction and can hide a specific loss of
exact-match retrieval capability.

### 7.1 The literature does not measure this. Plainly.

I checked every cross-layer sharing paper I could reach for a needle/passkey/MQAR-style
exact-retrieval evaluation:

| Paper | Retrieval / needle / passkey eval? |
|---|---|
| **CLA** (arXiv:2405.12981) | **NO.** Evals are Wikitext ppl + Hellaswag, PIQA, WinoGrande, SciQ, OpenBookQA, BoolQ, ARC-E via LM Eval Harness. |
| **A Systematic Study of Cross-Layer KV Sharing** (arXiv:2410.14442, NAACL 2025) | **NO.** Perplexity + the same 8 zero-shot commonsense benchmarks. Nothing else. |
| **Layer-Condensed KV Cache** (arXiv:2405.10637) | **NO** retrieval probe found. |
| **Hymba** (arXiv:2411.13676) | **PARTIAL** — has a 2-task "Recall" column, and it *regressed*. |
| **MLKV** (arXiv:2406.09297) | **PARTIAL** — LAMBADA (completion-style) collapsed 33.63 → 8.56 at 1 KV head. |
| **Character.AI** (blog, archived) | **PARTIAL/CONFOUNDED** — needle-in-a-haystack was checked, but for the **sliding-window** change, not the KV sharing. See below. |
| **Gemma 3n** (`num_kv_shared_layers=15`) | **NO** public ablation of the sharing at all. |
| **SwiftKV** (arXiv:2410.03960) | **NO** — 7-task aggregate only (ARC-C, WinoGrande, HellaSwag, TruthfulQA, MMLU, MMLU-CoT, GSM8K). |

**The CLA paper's own words settle its side of it.** Its downstream results were a wash, and it
says so explicitly: at 1B scale *"we found that none of our three models model consistently wins
or loses across different benchmarks,"* with all models *"within 1–5 percentage points of each
other"*; at 3B, *"we do not find that any model consistently wins or loses in these downstream
evaluations"* and *"all models perform similarly."* And on long context it defers entirely:
*"We leave end-to-end inference efficiency evaluations of large, long-context models employing
CLA as an interesting problem for future work."* Retrieval-adjacent methods (Landmark Attention,
Memorizing Transformers) appear only in related work. **No needle-in-a-haystack or passkey test
is mentioned anywhere in the paper.** **[VERIFIED]**

**[REASONING] Why "downstream benchmarks were a wash" is not reassurance.** The seven benchmarks
CLA used are commonsense/science multiple-choice. None of them requires copying a specific token
from a specific earlier position. A model can lose most of its induction-head-style exact-copy
capability and still score identically on Hellaswag and PIQA. So CLA's "wash" result is
*consistent with* a retrieval regression, and provides no evidence against one. This is exactly
the kind of gap a targeted probe closes.

### 7.2 The one data point that exists points the wrong way

Hymba's roadmap ablation (§4.1) is, as far as I can find, **the only published measurement of
recall-type accuracy with and without cross-layer KV sharing in any model**:

| Row | Commonsense | **Recall** | tok/s | Cache MB |
|---|---|---|---|---|
| C. + Local / global attention | 44.56 | **48.79** | 2399.7 | 41.2 |
| D. + **KV cache sharing** | 45.16 | **48.04** | 2756.5 | 39.4 |

**Commonsense +0.60, recall −0.75.** The paper characterizes this as *"maintaining comparable
recall accuracy"*, which is a fair reading of a 0.75-point move on a 2-task average at 300M
params / 100B tokens — but it is **the wrong sign**, and it is the *only* metric of the four that
moved down. Note also the internal tension: this is the paper that motivated its own sharing
design by citing *"KV cache shares a high similarity between adjacent layers"* — high similarity
predicts a small loss, and a small loss is what appeared, concentrated in recall.

**[REASONING] Caveats that keep this from being conclusive**, and which are themselves the
argument for running the experiment: (a) 2 tasks only, no error bars; (b) 300M/100B tokens;
(c) Hymba's sharing is between **sliding-window** layers, whose recall role is limited by
construction, while its 3 **global** layers — the ones that actually do long-range retrieval —
are *excluded* from sharing. So Hymba's number is a *lower bound on the damage*: it measures the
recall cost of sharing among layers that were not doing the retrieval. Our proposal would share
between full-attention layers, i.e. exactly the layers Hymba protected. **The expected effect is
therefore larger than −0.75, not smaller.** This is the strongest single argument in the dossier
for making retrieval the headline evaluation.

### 7.2b The Character.AI needle claim — read the scope carefully

**[VERIFIED]** Character.AI's post is the only source in this dossier that mentions
needle-in-a-haystack in connection with a cache reduction. But the attribution matters:
*"We found that **reducing attention horizon to 1024 on most attention layers** does not have a
significant impact on evaluation metrics, **including the long context needle-in-haystack
benchmark**."* That sentence is in the **Hybrid Attention Horizons** section — it is about the
**sliding window**. The **Cross Layer KV-sharing** section makes only the unquantified claim
*"we find that sharing KV across layers does not regress quality"*, with **no benchmark named and
no numbers**.

**[REASONING] So the strongest-sounding retrieval evidence for cross-layer sharing is actually
evidence for sliding windows, and the sharing claim is an unquantified assertion from a blog post
with no ablation.** That is worth stating precisely in the paper: it neither confirms nor refutes
our worry, and its existence should not be mistaken for the missing measurement. If anything it
sharpens the gap — a production team that clearly *had* a needle benchmark on hand chose to report
it for the window change and not for the sharing change.

### 7.3 The related evidence that *is* available, and why it is suggestive

Two indirect results from arXiv:2410.14442 bear on retrieval even though they never test it:

1. **"There is a significant drop in performance if the first layer is not a KV layer."** They
   hard-code the first layer as a KV layer in every `lasagna` configuration because of this.
   **[REASONING]** This is a layer-role asymmetry result: not all layers are equally safe to make
   consumers. It supports the hypothesis that *which* attention layers you pair matters for
   reasons beyond adjacency, and by extension that a pairing choice could preserve or destroy
   retrieval depending on which layers host the retrieval heads.
2. **Beyond 2× compression, "pairing queries of all layers with KVs of upper layers performs
   better"** than bottom-sourced KV — with `sandwich-middle` best at high compression. At exactly
   2× (our regime), *"most configurations can achieve higher throughput than standard
   transformers while maintaining competitive performance"* and the bottom configurations
   (= CLA-like) are fine. **[REASONING]** So at our target ratio the literature says the
   *aggregate* choice barely matters — which means aggregate metrics cannot discriminate between
   pairings, and a retrieval probe is the only thing that could. That is a methodological
   argument that our contribution is well-targeted.

### 7.4 Does sharing across an intervening conv/SSM block differ from across an attention block?

**Nobody has measured this. [GAP] — and this is the cleanest novel question in the proposal.**

What is established: CLA found adjacent pairing best and non-uniform pairing worse
(`DenseBack` +0.43 ppl), and Hymba paired strictly adjacent hybrid layers. Both are consistent
with "distance in the layer stack degrades sharing quality", but neither isolates *what* the
intervening computation is. In a standard transformer the only thing you can put between two
attention layers is an MLP (and in CLA's non-adjacent variants, another attention layer). In
LFM2, the intervening block is a **gated short convolution**.

**[REASONING] Two competing predictions, and this is worth stating in the paper because it is a
real theoretical fork:**

- **Optimistic:** a k=3 depthwise short conv is a *local, low-capacity* operator. It mixes each
  channel over a 3-token window with a diagonal kernel and two linear gates. Compared to an MLP
  (which is `d → 8d/3 → d` with a nonlinearity, i.e. a large pointwise transform), a short conv
  arguably *perturbs the residual stream less* in the directions that keys are read from. If so,
  K/V borrowed across one conv block could be *fresher* than K/V borrowed across one MLP block —
  and CLA-in-LFM2 could work *better* than CLA-in-a-transformer at the same layer distance.
- **Pessimistic:** the conv block is a **sequence-mixing** operator, unlike an MLP. It moves
  information *between positions*. A key vector `k_j` produced at layer 8 describes the content
  at position *j* as of layer 8; after a conv block, position *j*'s residual now contains a
  blend of positions *j−2, j−1, j*. The consumer's query at layer 10 is formed from that blended
  state and is being matched against *unblended* keys. That is a genuine
  **representation-alignment mismatch that has no analogue in the transformer case**, and it
  should hurt exact retrieval specifically (position-precise matching) far more than perplexity.

I can find no experiment distinguishing these. **[GAP]** The pessimistic story predicts the
damage grows with the number of intervening conv blocks, which yields a directly testable
prediction: **pairs separated by one conv block ((8,10), (10,12), (12,14)) should outperform
pairs separated by two ((2,5), (5,8))** on retrieval, with little difference in perplexity.
That is a clean, cheap, falsifiable experiment and it is the core scientific content available
here.

### 7.5 What to run (feeding into §10)

**[REASONING]** Given that no cross-layer sharing paper reports exact retrieval, the honest
framing of the contribution is: *"we supply the retrieval evaluation the cross-layer KV sharing
literature omits, in the architecture where the mechanism is most suspect."* Minimum probe set:

- **MQAR** (multi-query associative recall) — synthetic, cheap, directly measures the
  induction/associative-recall capability most at risk, and is the standard diagnostic in the
  linear-attention/hybrid literature. Sweep number of key-value pairs and sequence length.
- **Passkey / needle-in-a-haystack** at the model's trained context and at its stated limit.
  Report per-depth, not just aggregate — a shared-KV regression could appear only at particular
  needle depths.
- **Copying / verbatim n-gram retrieval** from earlier in the context.
- Report these **alongside** perplexity so the paper can show the dissociation (or its absence)
  explicitly. If perplexity moves 0.04 and MQAR moves 15 points, that is the paper.
- **Also report the null cleanly.** If retrieval is unaffected, that is a genuinely useful
  negative result which retroactively validates CLA's and Hymba's aggregate-only reporting, and
  it should be written up as such rather than buried.

---

## 9. Measurement — the quantities that must be reported separately

**[REASONING] The framing.** The single most common error in this literature is collapsing
distinct quantities into "memory saved". CLA saves **capacity but not bandwidth** (its own words);
MLKV delivered a **23× batch-size increase with no throughput gain**; Hymba's cache fell only
4.4% from KV sharing because SWA had already taken the bulk. A paper that reports one number
cannot distinguish these situations. The table below is the minimum honest set.

### 9.1 The eight distinct quantities

| # | Quantity | Definition | What it is sensitive to | Tooling |
|---|---|---|---|---|
| 1 | **Weight bytes** | Parameter tensors only | CLA removes `k_proj`+`v_proj` per consumer | `sum(p.numel()*p.element_size() for p in model.parameters())`; verify against a config-driven analytic formula |
| 2 | **Resident KV bytes** | KV cache actually held at a given `(batch, T)` | **This is what CLA reduces — 2× at CLA2** | Analytic formula, cross-checked against `torch.cuda.max_memory_allocated()` delta with/without cache |
| 3 | **KV write bytes / token** | Bytes written into the cache per decode step | **Falls with CLA** — consumers never write | Analytic; verify by counting `cache.update()` calls or a `nsys`/`ncu` `dram__bytes_write` delta |
| 4 | **KV read bytes / token** | Bytes read from the cache per decode step | **Does NOT fall with CLA** — consumers re-read the shared bank | Analytic; `ncu` `dram__bytes_read.sum` |
| 5 | **Conv state bytes** | The `d × (k−1)` (or `d × k`) rolling conv state per LIV layer | Fixed in `T`; 10 layers × d | Analytic; must be reported to make the hybrid comparison fair |
| 6 | **Peak allocator bytes** | End-to-end high-water mark | Fragmentation, activations, autograd | `torch.cuda.max_memory_allocated()` **and** `max_memory_reserved()` — report both |
| 7 | **HBM traffic** | Actual DRAM bytes moved | The thing latency tracks | `ncu --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum`; `nsys` for timeline attribution |
| 8 | **Attention FLOPs** | `2·B·H·T²·d_h`-style count for prefill; `2·B·H·T·d_h` per decode step | Unchanged by CLA at fixed head count | Analytic formula; `torch.utils.flop_counter.FlopCounterMode` |

**[DERIVATION] For our model, quantities 2–4 diverge in a way that must be shown explicitly.**
At `d=2048`, 6 attention layers, `n_g=8`, `d_h=64`, fp16 (2 B):
- Per attention layer per token: `2 (K,V) × 8 × 64 × 2 B = 2048 B`.
- **Baseline (6 banks):** resident `= 6 × 2048 = 12,288 B/token` (12.0 KiB/token).
- **CLA2 (3 banks):** resident `= 3 × 2048 = 6144 B/token`. **Exactly 2×.**
- **KV writes/token:** baseline `12,288 B`; CLA2 **`6144 B` — also 2×.**
- **KV reads/token at decode, over the whole context:** all **6** attention layers still read a
  full bank of length `T`. Baseline `= 6 × 2048 × T`. **CLA2 `= 6 × 2048 × T` — IDENTICAL.**
  The three consumers read their producer's bank; nothing is saved.
- **Conv state (fixed, all 10 LIV layers, `k=3`):** `10 × 2048 × 2 × 2 B = 81,920 B = 80 KiB`
  total (using `d·(k−1)` per layer), i.e. **negligible against KV beyond ~7 tokens of context.**
  Report it anyway, because a reviewer will ask.

**This trio — resident 2×, writes 2×, reads 1× — is the honest summary of what CLA2 buys, and it
should appear as three separate rows, not one "2× memory saving" claim.**

### 9.2 Pitfalls, each of which has burned someone

1. **Allocator caching hides real usage.** `torch.cuda.max_memory_allocated()` reports bytes the
   allocator handed to tensors; `max_memory_reserved()` reports what CUDA actually holds. The gap
   is fragmentation and can be large. **Report both.** Also call
   `torch.cuda.reset_peak_memory_stats()` before each measured region, or you will report a peak
   from a previous phase (typically training warmup) and conclude nothing.
2. **A saving that shows up only as batch-size headroom is still real — but it is a different
   claim.** MLKV's honest result was "max batch 48 → 940, and no speedup." If our CLA2 model shows
   the same shape, we must say **"2× resident KV → N× larger batch at fixed HBM, with unchanged
   per-token decode latency"**, not "2× faster."
3. **FLOPs are not latency, and CLA changes neither.** CLA removes only the K/V *projection* FLOPs
   (small); the attention FLOPs are untouched because the consumer still attends over the full
   bank. Do not present a FLOP reduction as a speed result. Conversely a *measured* throughput
   gain (Hymba saw +14.9%) comes from cache-pressure effects — larger batches, better occupancy —
   not from less arithmetic, and should be attributed as such.
4. **Capacity vs bandwidth confusion is the field's signature error.** Quote CLA's own sentence in
   the paper (*"no direct effect on the memory bandwidth consumed by the attention mechanism in
   each decoding step"*) so no reader mistakes row 2 for row 4.
5. **Peak memory is dominated by activations during training, not by KV.** A KV-cache result must
   be measured in an **inference/decode** harness with `torch.no_grad()`, at a stated
   `(batch, prompt_len, gen_len)`. State them; Hymba's 79 MB is at "8K seq, batch 128, A100, FP16
   cache" and is meaningless without that.
6. **Parameter deltas confound quality comparisons.** CLA2 on our model removes **6.29M params**
   (§5.5). MLKV had to enlarge MLP widths to equalize params before comparing; the systematic
   study notes KV-layer count *sets model size*. **Any quality claim needs a param-matched
   control** (§10.2).
7. **`memory._record_memory_history` is the tool for "where did the peak come from".** Use
   `torch.cuda.memory._record_memory_history(max_entries=...)` then
   `torch.cuda.memory._dump_snapshot("snap.pickle")` and view at `pytorch.org/memory_viz`. Use the
   **torch profiler memory timeline** (`profile(profile_memory=True, record_shapes=True)` →
   `export_memory_timeline`) to attribute bytes to categories over time. These answer *why* peak
   is what it is; the counters in §9.1 answer *what* it is.
8. **`ncu` serializes kernels and inflates wall time.** Use `ncu` for byte counters on isolated
   kernels, `nsys` for end-to-end timelines, and a plain wall-clock harness for the headline
   latency number. Never quote latency measured under `ncu`.
9. **fp16 vs bf16 vs fp8 KV changes every byte number.** State the cache dtype in every table.
   DeepSeek quotes cache in **elements** partly to sidestep this — consider doing the same and
   giving bytes separately.

### 9.3 The table the paper must report

**[REASONING]** One table, rows = configurations, columns = the quantities. Every cell analytic
where possible, with a measured cross-check column. At fixed `(batch=1, T=4096, fp16)`:

| Config | Params (M) | Resident KV B/tok | KV write B/tok | KV read B/tok | Conv state (KiB) | Peak alloc (MiB) | Peak reserved (MiB) | val ppl | MQAR | Passkey | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline LFM2 (6 attn, GQA-8) | — | 12,288 | 12,288 | 12,288·T | 80 | | | | | | |
| **CLA2, pairs (8,10),(12,14)** | −4.19 | 8,192 | 8,192 | 12,288·T | 80 | | | | | | |
| **CLA2, 3 pairs (2,5),(8,10),(12,14)** | −6.29 | **6,144** | **6,144** | 12,288·T | 80 | | | | | | |
| **CLA2, 3 pairs (8,10),(10,12),(12,14)-style** | −6.29 | 6,144 | 6,144 | 12,288·T | 80 | | | | | | |
| Control: 3 attention layers | −large | 6,144 | 6,144 | **6,144·T** | 80 | | | | | | |
| Control: GQA-4 (halve `n_g`) | −small | 6,144 | 6,144 | **6,144·T** | 80 | | | | | | |
| Control: MQA (`n_g`=1) | — | 1,536 | 1,536 | 1,536·T | 80 | | | | | | |
| Control: MLA (`d_c`=256, `d_h^R`=32) | — | ~3,456 | ~3,456 | ~3,456·T | 80 | | | | | | |
| Control: SWA on 3 of 6 attn layers | — | varies w/ window | | | 80 | | | | | | |

**The column that makes the paper honest is "KV read B/tok".** It is the only column where CLA2
does *not* match its equal-capacity controls — and it is precisely why "fewer attention layers"
and "lower `n_g`" are not strawmen. If CLA2 does not beat them on quality, it has no case, because
they beat it on bandwidth.

---

## 10. Experiment design implications

### 10.1 The pairing configurations to test

**[DERIVATION] The design space is small and fully enumerable, which is a gift — say so in the
paper.** With attention layers at `[2, 5, 8, 10, 12, 14]`, there are exactly **15 ways** to
partition them into 3 forward producer→consumer pairs. Ranked by total intervening blocks
(the quantity §7.4 predicts matters):

| Pairing | gaps (intervening blocks per pair) | total |
|---|---|---|
| **`(2,5) (8,10) (12,14)`** | 2, 1, 1 | **4 — unique minimum** |
| `(2,5) (8,12) (10,14)` | 2, 3, 3 | 8 |
| `(2,5) (8,14) (10,12)` | 2, 5, 1 | 8 |
| `(2,8) (5,10) (12,14)` | 5, 4, 1 | 10 |
| `(2,10) (5,8) (12,14)` | 7, 2, 1 | 10 |
| … 10 more … | | 14–18 |
| `(2,10) (5,12) (8,14)` / `(2,12) (5,10) (8,14)` etc. | | **18 — maximum** |

**Recommended arms (5 training runs for the sharing sweep):**

- **A1 — `CLA2-adjacent`: `(2,5) (8,10) (12,14)`.** The **primary arm**. It is the unique
  minimum-distance perfect matching, it is "adjacent in the attention subsequence", and it is the
  closest realizable analogue of CLA's `lasagna-bottom` recommendation and of Hymba's strictly
  adjacent `kv_reuse_group`. 6 banks → 3.
- **A2 — `CLA2-far`: `(2,12) (5,10) (8,14)`** (total 18, a maximum-distance matching).
  **This is the scientific control that answers §7.4.** Same parameter count, same resident bytes,
  same everything — only the distance differs. If A1 ≫ A2 on retrieval but A1 ≈ A2 on perplexity,
  that is the paper's headline result and it is a finding nobody has published.
- **A3 — `CLA2-tail-only`: `(8,10) (12,14)`, layers 2 and 5 keep their own banks.**
  6 banks → 4 (1.5× reduction). Motivated by **three independent findings that shallow layers are
  special**: LCKV's *"significant drop in performance if the first layer is not a KV layer"*,
  LISA's *"shallow layers are vulnerable to small deviations"*, and Hymba's exclusion of layer 0
  from every sharing group. A3 is the "safe" configuration and may well be the best
  quality-per-byte point.
- **A4 — `CLA2-deep-pairs`: `(8,10) (10,12) (12,14)` is NOT a valid matching** (layer 10 and 12
  would be both producer and consumer). Replace with **`CLA3` on the deep block:
  `(8,10,12)` sharing one bank + `(2,5)` + `14` alone** → 6 banks → 3, but with one 3-way group.
  Tests sharing factor 3 locally, where CLA reported CLA3 = 13.77 vs CLA2's better frontier, and
  where Hymba used a 3-way group `[16,17,18]`.
- **A5 — `CLA2-adjacent, share V only` (or K only).** Motivated directly by **FusedKV
  (arXiv:2512.03870, ICLR 2026)**, which found **values derive predominantly from the bottom layer
  while keys draw from bottom and middle layers** — implying K and V have different cross-layer
  transferability. Sharing only V gives 1.33× (6 → 4.5 bank-equivalents) but may be nearly free;
  sharing only K tests the opposite. **No CLA-family paper has run this ablation.** Cheap, and a
  genuine novelty.

**[REASONING] If the budget allows only two runs: A1 and A2.** That pair alone answers the one
question the literature does not, and it is a controlled comparison in the strict sense — identical
parameter count, identical byte counts, one variable.

### 10.2 Mandatory controls

The controls are what make this a paper rather than an anecdote. All must be **byte-matched at
the resident-KV level** (6144 B/token at d=2048) so the comparison is at equal capacity, and the
parameter delta must be reported and where possible compensated.

| Control | Configuration | Why it is mandatory |
|---|---|---|
| **C0 — stock LFM2** | 6 attention layers, GQA-8, no sharing | The reference point. 12,288 B/tok. |
| **C1 — fewer attention layers** | **3** attention layers (e.g. `[5,10,14]`), 13 conv | **The most dangerous control.** Same 6144 B/tok resident, and it **also halves KV read bytes** where CLA does not (§9.3). If CLA2 does not beat this on quality, CLA2 has no case. CLA itself did not run this control. |
| **C2 — lower `n_kv` / MQA** | GQA-4 (`n_g=4` → 6144 B/tok, exactly byte-matched) and MQA (`n_g=1` → 1536 B/tok) | CLA's own recommendation is CLA2+MQA, and CLA reported **GQA+CLA2 mostly lost** to equal-footprint baselines. GQA-4 is the precise byte-matched competitor. ⚠️ **Monitor MQA for loss spikes** — GQA arXiv:2305.13245 App. A reports MQA-from-scratch had *"frequent loss spikes"* and diverged on long inputs, and we train from scratch (§1.6). |
| **C3 — MLA** | `d_c = 256 (=4d_h)`, `d_h^R = 32 (=d_h/2)` → **[DERIVATION]** `(256+32)·2 B·6 = 3456 B/tok` | A reviewer *will* ask. MLA reaches **3456 B/tok vs CLA2's 6144** — it wins on capacity, and DeepSeek-V2 App. D.2 claims MLA beats MHA outright. We must show the number and argue orthogonality + implementation cost (§1.5), not pretend MLA is worse. |
| **C4 — sliding-window** | SWA (window 1024) on 3 of the 6 attention layers, keeping 3 global | Gemma 3 gets 60% → <15% overhead at 32K for *"minimal"* perplexity cost (§2.1). This is the strongest competitor. Also enables the **composition arm**: SWA **+** CLA2, which is what Hymba actually did. |
| **C5 — param-matched C0** | Stock LFM2 with 6.29M params **removed** elsewhere (e.g. slightly narrower MLP) | CLA2 removes 6.29M params (§5.5). MLKV had to equalize params via MLP width before comparing; without this, any quality delta is confounded. |
| **C6 — "borrowed-KV is stale" probe** | A1 at init/early training vs late training, measuring producer-vs-consumer key cosine similarity | Not a training run — a diagnostic. Tests the mechanism's assumption directly, and connects to the KVSharer-vs-MiniCache dispute about whether similar or dissimilar layers should be paired (§1.7). |

**Note on C1 and C4 being *stronger* on some axes than our method.** **[REASONING]** This should
be stated up front rather than buried. C1 wins on KV read bytes; C3 and C4 win on resident bytes.
CLA2's distinguishing property is narrower and should be claimed narrowly: **it is the only option
that halves resident KV capacity while leaving every layer's attention pattern and global
receptive field completely unchanged.** C1 removes attention layers; C2 removes head diversity;
C3 changes the attention block; C4 truncates receptive fields. That is a real and defensible
niche, and it is testable: it predicts CLA2 should degrade *retrieval* least among the byte-matched
options, even if it is not best on perplexity.

### 10.3 The RoPE decision, with justification

**Decision: share POST-rotary keys. Do not re-apply RoPE in the consumer. Report this explicitly
as a design decision, because CLA does not discuss it.**

**Justification, in descending order of strength:**

1. **Three independent implementations all chose post-rotary.** **[VERIFIED]**
   - **Gemma 3n** (`modeling_gemma3n.py`, merged upstream HF): the producer applies
     `apply_rotary_pos_emb(key_states, cos, sin)` *before* publishing to `shared_kv_states`; the
     consumer applies RoPE to its **own Q only**.
   - **Hymba** (`modeling_hymba.py`, NVIDIA): identical structure — producer rotates K, publishes
     `key_states_no_repeat`; consumer takes the rotated tensor and rotates only its Q.
   - **FusedKV** (arXiv:2512.03870, ICLR 2026) fuses **post-RoPE keys** explicitly, and is the only
     source that states a *reason*: to *"preserve relative positional information while avoiding
     the cost of re-applying rotary embeddings."*
2. **It is correct, not merely conventional, when producer and consumer see the same positions.**
   **[DERIVATION]** RoPE's dot product depends only on the *relative* offset:
   `⟨R_m q, R_n k⟩ = f(q, k, m−n)`. The consumer's query at position `m` is rotated by `R_m`; the
   borrowed key at position `n` was rotated by `R_n` **by the producer using the same
   `position_ids`**. So the relative geometry `m − n` is exactly right. There is no positional
   error introduced by borrowing post-rotary keys. Sharing *pre*-rotary keys and re-rotating in the
   consumer would give the **identical** result mathematically, at strictly higher cost — one extra
   rotation per consumer per token — with the *added* burden of caching un-rotated keys (which
   breaks the "one physical bank" property that makes vLLM's aliasing free, §5.2).
3. **It preserves the capacity saving.** Caching pre-rotary K to allow per-consumer re-rotation
   would require either storing pre-rotary K (and rotating on every read — extra FLOPs and no
   capacity gain) or storing **both** forms (destroying the entire point).

**The one case where this reasoning fails, and it must be checked.** **[REASONING]** If producer
and consumer ever use **different** RoPE parameters — different `rope_theta`, different scaling, or
one being a sliding-window layer and the other global — then the borrowed post-rotary keys are
rotated with the *wrong* base and the relative-offset argument collapses. This is exactly why
**Gemma 3n pairs sliding-with-sliding and global-with-global** (`kv_shared_layer_index` is computed
per `layer_types` entry) and why **vLLM uses `offset = 2 if self.sliding_window is not None else 1`**.
**For our model this is safe** — all 6 LFM2 attention layers are `full_attention` with one shared
`rope_theta` (no released LFM2 config has a sliding window), so all pairings are type-homogeneous.
**But if control C4 (SWA on some layers) is combined with CLA2, pairing must respect layer type.**
Encode that as a config validation assert.

**Secondary norm decision.** CLA uses *separately learnable affine LayerNorm parameters* for the
KV block vs the Q block; Gemma 3n gives consumers **no** `k_norm`/`v_norm` at all. **Follow CLA**
(give the producer's KV path its own norm params) since CLA is the result being reproduced, and
note the divergence from Gemma 3n as an untested alternative.

### 10.4 The measurement table the paper must report

Per §9.3, with these non-negotiables:
- **Three separate KV rows: resident B/tok, write B/tok, read B/tok.** CLA2 gives **2× / 2× / 1×**.
  Collapsing these into one "2× memory saving" is the field's signature error and a reviewer will
  catch it.
- **Both `max_memory_allocated()` and `max_memory_reserved()`**, with
  `reset_peak_memory_stats()` before each measured region.
- **Conv state bytes** (80 KiB total at d=2048, k=3, fp16, using `d·(k−1)`), so the hybrid
  accounting is complete.
- **Params, with the −6.29M CLA2 delta shown**, and C5 as the param-matched control.
- **Quality columns: val ppl AND MQAR AND passkey/needle (per-depth) AND the standard 7–8 zero-shot
  benchmarks.** The whole point of §7 is that the last two columns can move independently of the
  first, so all must be present.
- **tok/s and max batch at fixed HBM.** Following MLKV's honest precedent: if the gain is
  batch-headroom rather than latency, report it as batch headroom.
- **Cache dtype stated in every table caption**, plus `(batch, prompt_len, gen_len, device)`.

### 10.5 Honest summary of the position

**[REASONING]** After this survey, the proposal's standing is:
- **The mechanism is not novel** — Hymba did CLA-in-a-hybrid in Nov 2024, cited CLA, shipped code.
  Gemma 3n shipped cross-layer sharing to production. vLLM and HF support it natively.
- **The specific questions are novel and unanswered**: sharing across an intervening *sequence-
  mixing* block in a sequentially interleaved hybrid; sharing between *full-attention* rather than
  local layers; the effect on *exact retrieval*; and whether pairing *distance* matters more for
  retrieval than for perplexity.
- **The evaluation gap is real and is the strongest card**: not one cross-layer sharing paper
  reports needle/passkey/MQAR. The single relevant number that exists (Hymba row C→D) shows recall
  going **down** while everything else went up, in the configuration *least* likely to show damage.
- **The competition is strong**: MLA reaches 3456 B/tok vs CLA2's 6144; Gemma-3-style 5:1 SWA cuts
  more for *"minimal"* perplexity cost; and simply using 3 attention layers instead of 6 matches
  CLA2's capacity while also halving read bandwidth. **CLA2's defensible niche is narrow and should
  be claimed narrowly: the only method that halves resident KV without changing any layer's
  receptive field or attention block.**

---

### 10.6 Open items a follow-up pass should close

Flagged honestly rather than papered over:

1. **llama.cpp cross-layer KV sharing support — UNKNOWN** (§5.3b). Symbol-name search only.
2. **§3's compression numbers for H2O / SnapKV / PyramidKV / KIVI were not re-verified** against
   primary sources. Mechanisms are right; the ratios must be checked before publication.
3. **Gemma 3's per-configuration KV-memory percentages are figure-only.** Only *global-only = 60%*
   and *1:3 + sw=1024 = <15%* at 32K appear in prose. **There is no published 5:1 percentage** —
   do not invent one. Likewise Gemma 3's perplexity ablation values (Figs. 3, 4) are image-only.
4. **YOCO's per-length memory table, LCKV's memory in GB, MLKV's absolute MB/throughput, and the
   systematic study's perplexity/throughput values are all figure-only** in their respective
   papers. Any of these quoted as text must come from reading the plots.
5. **YOCO++ (arXiv:2604.13556) numbers are second-hand** — not fetched directly.
6. **Gemma 2's 1:1 + sw=4096 is verified, but Gemma 2 published no interleaving-quality ablation.**
   Do not cite it as evidence that interleaving is free.
7. **Zamba2's arXiv id (2411.15242) was not directly verified** this pass; its no-KV-sharing status
   is established from its `config.json` (`num_mem_blocks=2`, weight sharing) and Zamba1's paper.

---

*End of dossier. Sections follow the task brief: 0 = carried-in context, 1 = other cache-reduction
families, 2 = sliding-window/local-global, 3 = post-hoc methods, 4 = hybrid novelty verdict,
5 = implementation reality, 7 = retrieval failure modes, 9 = measurement, 10 = experiment design.
Brief items 6 and 8 are folded into §5.4/§7.4 (what breaks; conv-vs-attention intervening block)
and §10 (design implications) respectively.*

