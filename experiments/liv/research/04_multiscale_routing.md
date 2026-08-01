# Multiscale / Dilated Causal Convolution and Token-Dependent Routing

Research dossier supporting the proposed experiment: replace the LFM2-style gated short causal
convolution (k=3, depthwise, causal) with **four parallel branches** at effective spans 3/5/9/15,
mixed by a **token-dependent softmax router** (4 nonnegative weights summing to 1).

Two parameterizations are on the table:

| Variant | Branch kernels | Taps per channel | Lag sets covered |
|---|---|---|---|
| **Dense multi-width** | widths 3, 5, 9, 15 | 3+5+9+15 = **32** | contiguous {0..2}, {0..4}, {0..8}, {0..14} |
| **Dilated 3-tap** | k=3 at dilations 1, 2, 4, 7 | 4x3 = **12** | {0,1,2}, {0,2,4}, {0,4,8}, {0,7,14} |

Conventions used throughout: *lag* = number of positions back from the current token (lag 0 = self).
A causal kernel of width W spans lags {0..W-1}. A causal 3-tap kernel at dilation d spans lags
{0, d, 2d}, so its **max lag is 2d** and its **effective span is 2d+1**.

**Legend for evidence grading:**
- **[FACT]** = published, cited, with numbers I retrieved.
- **[REASONING]** = my derivation or inference, clearly not from a paper.
- **[DIGITIZED]** = read off a rendered figure because the paper published no table; approximate.
- **[UNKNOWN]** = I could not verify; flagged explicitly.

**Contents** (sections are numbered by research task, not file order):
- [0. Executive summary](#0-executive-summary-read-this-first)
- [1. Sieberling et al. 2026, dynamic short convolutions](#1-the-key-paper-sieberling-et-al-2026-dynamic-short-convolutions-improve-transformers) — task 2's key paper, **retrieved**
- [2. Dilated / multiscale prior art](#2-dilated--multiscale-causal-convolution-prior-art) — task 1
- [3. Adaptive receptive field & input-conditioned kernels](#3-learnedadaptive-receptive-field-and-input-conditioned-kernel-mixtures) — task 3
- [4. Routing mechanics and pitfalls](#4-routing-mechanics-and-its-pitfalls) — task 4
- [5. **Equivalence and identifiability**](#5-equivalence-and-identifiability-the-most-important-section) — task 5, **the core analytical result**
- [6. Interpretation caveats and knockout protocol](#6-interpretation-caveats-and-a-rigorous-knockout-protocol) — task 6
- [7. Throughput reality](#7-throughput-reality) — task 7
- [9. Short-conv width consensus in modern LMs](#9-short-convolutions-in-modern-lm-architectures--the-width-consensus) — task 2, **contains a second direct negative result**
- [8. **Experiment design implications**](#8-experiment-design-implications) — **the deliverable (last section)**

---

## 0. Executive summary (read this first)

1. **The single most decisive piece of evidence exists and it is bad for this proposal.** The paper
   the brainlift cites — Sieberling et al., *Dynamic Short Convolutions Improve Transformers*
   (arXiv 2606.03825) — **was retrieved successfully** and contains a direct kernel-width ablation
   at 300M params / 15B tokens. Perplexity: W=1 → 18.42, W=2 → 18.17, **W=3 → 18.08**,
   W=4 → 18.10, W=5 → 18.09, W=6 → 18.10. The curve is **flat from W=3 onward**. Their verbatim
   conclusion: "3 or 4 is generally the sweet spot," and wider kernels add parameters without gains.
   The proposal's whole premise is that spans out to 15 are useful. The closest and most
   authoritative published measurement says spans past 3–4 are worth **~0.00–0.02 ppl**, i.e. noise.
2. **★ Someone has now tried the dense-multi-width idea in an LM short-conv slot, and it LOST.**
   "Convolution for Large Language Models" (arXiv 2607.18413) added a residual depthwise Conv1D to
   Qwen3-1.7B and swept kernel size: no-conv 13.42 → k=2 12.99 → **k=3 12.79** → k=4 13.13 PPL. Then
   they tried **multi-branch reparameterization over MIXED kernel sizes and perplexity got WORSE:
   12.79 → 13.28.** That is this proposal's core mechanism (parallel branches at several widths),
   tested in an LM, losing to a single k=3 kernel by 0.49 PPL. Caveats: single runs, grid only {2,3,4},
   recent preprint — suggestive, not decisive, but direct and concordant with everything else.
   **Two independent labs on different base models both peak at k=3.**

3. **The input-independent variant of the dilated proposal is mathematically vacuous, and worse than
   expected.** A fixed-weight mixture of the four dilated 3-tap branches is *exactly* a single sparse
   15-tap causal kernel (proved and numerically verified to 1e-15 in Section 5). It is not a new
   architecture. Moreover I computed the lag coverage: dilations {1,2,4,7} reach only lags
   **{0,1,2,4,7,8,14} = 7 of 15**, leaving **8 lags ({3,5,6,9,10,11,12,13}) structurally unlearnable**,
   while lag 0 is covered 4x redundantly. So its 12 parameters/channel buy only **7 effective DOF**,
   versus **15 DOF for 15 parameters** in a plain dense k=15 kernel. A dense 15-tap kernel is
   *cheaper in parameters than the dense-branch variant (15 vs 32)* and strictly more expressive
   than both.
4. **The dense softmax router saves zero compute** — all four branches are always evaluated. The
   proposal is strictly *more* expensive than the baseline, so it must win on quality alone.
5. **Hard/top-k routing over 4 tiny depthwise branches cannot pay off on a GPU.** Depthwise conv is
   memory-bandwidth-bound at single-digit % of peak FLOPs; gather/scatter and kernel-launch overhead
   dominate at this granularity. Evidence in Section 4/7.
6. **Prior art did this, and every branch-count sweep saturates at TWO.** MixConv (arXiv 1907.09595)
   is mixed depthwise kernel sizes in one layer but by **channel partitioning, not routing**, and its
   dilated variant *regressed to baseline*. **SKNet (arXiv 1903.06586) is the exact image analogue** — a
   softmax router over different kernel sizes — and its branch-count ablation is decisive:
   **M=2 → 20.79% err, M=3 → 20.76%, i.e. +0.03% for +1.8M params and +0.23 GFLOPs**; the paper
   concludes "M = 2 is preferred." Res2Net saturates at s=2→4 (0.15 pts for +12% latency) and MixConv at
   g=2. **Three independent sweeps, all with the knee at 2 branches. The proposal uses four.**
7. **★ A plain 4-way softmax router over kernels performed WORSE THAN STATIC in the one paper that
   measured it.** Chen et al. Dynamic Convolution (arXiv 1912.03458), K=4 — the same arity — scored
   **64.8% vs a 65.4% static baseline** with plain softmax (τ=1). The entire +4.5-point gain came from
   **temperature annealing (τ=30 → 1 over 10 epochs)**, because a near-one-hot early router "only allows
   a small subset of kernels across layers to be optimized." **Implication: a temperature schedule is
   mandatory, not optional. Without it, a null result is uninterpretable** — you cannot distinguish
   architectural failure from three starved branches.
8. **★ The cleanest precedent for the most likely failure mode: gates degenerate into constants.**
   Squeeze-and-Excitation (arXiv 1709.01507) found late-stage blocks saturate to "activations close to
   one," at which point "an SE block reduces to the identity operator" — and **removing them cost
   <0.1% top-5 while halving the parameter overhead.** Branchformer (arXiv 2207.02971) is even more
   direct: its learned input-dependent branch weights had **standard deviation < 0.01**, and **static
   concatenation BEAT the input-dependent merge (4.19/4.43 vs 4.23/4.61 CER).** Also: SKNet's own
   input-sensitivity vanishes by layer SK_5_3.
9. **State-size accounting is materially worse than advertised, and this is a hard cost.** Any span-15
   variant — including the "cheap" 12-tap dilated one — needs **15** cache slots instead of 3, a
   **5x** increase (LFM2's real config has `conv_L_cache: 3`, `conv_dim: 2048`, 10 conv layers). That
   is **120 KB → 600 KB** of persistent decode state per sequence. Sparsity buys **zero** state
   savings, since max lag sets the buffer depth (Section 5.4). This matters more than it looks:
   LFM2's stated design driver is **embedded SoC CPU** deployment, where peak memory is the objective
   STAR optimized against.

10. **When a model is allowed to choose its own receptive field, it chooses SHORT — and making that
   choice token-dependent bought nothing.** Adaptive Attention Span (arXiv 1905.07799): with the span
   limit at **8192**, the learned average span was **314** (12-layer) and **245** (24-layer) — 3–4% of
   what was permitted — and **the lowest 5 of 12 layers pinned at the minimum (R=32)**. Their
   *dynamic, input-dependent* span variant tied the static one exactly (**1.08 vs 1.08 bpc**). This is
   the closest published test of the proposal's core premise and it is negative on both halves: spans
   collapse short, and token-dependence of the span adds nothing.
11. **The best argument FOR the proposal, stated fairly.** LightConv/DynamicConv (arXiv 1901.10430)
   found that an input-dependent depthwise filter **diverged without normalization**, and softmax over
   taps worked best (26.9 BLEU vs divergence). So normalized input-dependent filters have real
   precedent. **But note the mechanism differs:** LightConv softmaxes over *kernel taps* (bounding the
   filter); this proposal softmaxes over *branches* (bounding the mixture) while leaving branch taps
   free — so it does **not** inherit LightConv's stability guarantee (Section 3.1). And LightConv's
   published multiscale design was a **per-layer increasing schedule (3,7,15,31x4)**, not parallel
   branches — a cheaper alternative the experiment must test as a control.
12. **Effect sizes in this whole literature are small.** LightConv's increasing-kernel schedule bought
   **+0.3 BLEU**; input-dependence bought another **+0.3**; MixConv bought ~+0.25–0.5 top-1;
   SGConv beat Transformer-XL by **0.05 PPL**. Nothing here suggests a large win is available, so the
   experiment must be powered (≥3 seeds) to resolve differences of ~0.05 ppl — otherwise it cannot
   distinguish success from noise.

**Honest prior: this proposal is unlikely to pay off as stated — I estimate 10–20% that the full router
beats a parameter-matched dense k=15 by a margin surviving 3 seeds (reasoning in Section 8.5).** The
strongest defensible version of it is not "multiscale" but either (a) the **RepVGG reparameterization
question** — do 4 branches *train* better than the identical fused 15-tap kernel, which is free at
inference — or (b) the router as a *cheap input-dependent gate*, which is a strictly weaker form of what
arXiv 2606.03825 already showed works better (input-dependent *taps* via a low-rank generator, not
input-dependent *branch weights*). Full ladder and salvage paths in Sections 8.1 and 8.6.

---

## 1. THE key paper: Sieberling et al. 2026, *Dynamic Short Convolutions Improve Transformers*

**Retrieval status: SUCCESS.** arXiv:2606.03825, <https://arxiv.org/abs/2606.03825>,
HTML: <https://arxiv.org/html/2606.03825v1>. Submitted 2 June 2026, CC BY 4.0.
Authors: Oliver Sieberling, Bharat Runwal, Rameswar Panda, Yoon Kim.
(Note: this is a 2026 paper, past my training cutoff — everything below is from the retrieved HTML,
not memory. The HTML extraction truncated at Appendix B, so Tables 5–6 per-task/RULER breakdowns and
Appendices C–D are **[UNKNOWN]** to me.)

### 1.1 What they actually did

**[FACT]** Static baseline formulation, as written in the paper:

$$y_t := \sum_{k=0}^{W-1} w_k \odot x_{t-k}, \qquad x_t \in \mathbb{R}^D,\; w \in \mathbb{R}^{W \times D}$$

with $\odot$ elementwise. Only taps $k \ge 0$ → **causal**; per-channel weights → **depthwise**.
This is the LFM2-style short conv.

**[FACT]** The **dynamic** version regenerates the filter at every position:

$$y_t := \sum_{k=0}^{W-1} w_k^{(t)} \odot x_{t-k}, \qquad w^{(t)} \in \mathbb{R}^{W \times D}$$

where $w^{(t)}$ comes from a "weight generator (e.g., a linear projection)" applied to the input.

**Critical detail for our purposes — [FACT]:** the generated filters are **affine transformations of
the input**, with a bias term. **There is no softmax normalization of the generated filter anywhere
in the paper.** This is a direct contrast with Wu et al. 2019's LightConv, which *did* softmax over
kernel taps (Section 3.1). Sieberling et al. simply let the taps be free real numbers.

**[FACT]** Two parameterizations, because a naive $D \to W\!\cdot\!D$ projection would "roughly double
the parameter count":
- **Low-rank**: factorize the generating projection through rank $R$. Generally the better performer.
- **Head-wise**: project $D \to W\!\cdot\!(D/H)$ and broadcast each generated filter across a head of
  $H$ channels ("head-wise tying"). Simplifies the GPU kernel.

**[FACT]** Placement mechanics: the generator consumes the **post-attention-norm activations** (not
Q/K/V themselves), which "allows the projection to be fused with the qkv_projection." Applied
**residually**: $X \leftarrow X + \mathrm{dynamicShortConv}(X)$ for $X \in \{Q, K, V\}$, and placed
**before RoPE**.

**[FACT]** Their stated inductive-bias claim: unlike attention's query-key similarity, dynamic convs
"generate them directly from the querying position," giving a bias "toward retrieving by relative
position within the filter window."

### 1.2 Kernel width ablation — THE decisive table

**[FACT]** 300M model, 15B tokens, low-rank $R=16$, applied to Q+K+V, Nemotron-CC perplexity:

| Width W | Params | PPL |
|---|---|---|
| 1 | 306.8M | 18.42 |
| 2 | 307.6M | 18.17 |
| **3** | 308.5M | **18.08** |
| 4 | 309.3M | 18.10 |
| 5 | 310.1M | 18.09 |
| 6 | 311.0M | 18.10 |

**[FACT]** Their conclusion, quoted: "3 or 4 is generally the sweet spot"; wider kernels add
parameters without gains. Main experiments use **W=4**.

**[REASONING] Why this is close to fatal for the multiscale proposal.** The measured gain from
W=2→3 is 0.09 ppl. From W=3→6 it is **-0.02 ppl (i.e. slightly worse)**. This is the
best-controlled published width sweep on exactly the primitive in question, in an LM, at a
respectable scale, with an *input-dependent* filter (i.e. the most expressive version — if extra
span helped anyone it should help the dynamic variant most). The proposal's spans of 9 and 15 sit
far to the right of a curve that is already flat at 3. Note also that the ablation is *dense
contiguous* widths — the proposal's dilated branches cover *fewer* lags within the same span, so
they can only be worse than the dense-W curve at equal max lag, not better (Section 5.2).

**Caveat that could rescue the proposal — [REASONING]:** this ablation is a *single* width per layer,
not a *mixture*. It is logically possible that a mixture of {3,5,9,15} beats every single width
because different tokens want different spans, even though no single wider width beats 3. That is
precisely the hypothesis the experiment should test — but note the prior is now unfavorable, because
the flat curve at W≥3 says the *marginal information* in lags 3–14 is near zero on average. For a
mixture to win, that information must be near-zero on average but high on a token subset **and**
the router must find it. Section 5.3 pins down what baselines are needed to demonstrate that.

### 1.3 Throughput cost they report

**[FACT]** End-to-end, single H100 80GB HBM3, seq len 4096, bf16, `torch.compile`, measured at
**300M and 2B** scales:
- **QKV dynamic conv** (both head-wise and low-rank): **within 8% overhead** vs Transformer baseline.
- **Static conv**: roughly **6% slowdown**.
- **All-linear placement** (conv after every linear layer): **22–25% reduction in end-to-end
  training throughput** (abstract cites 22% at 2B).

**[FACT]** Kernel-level microbenchmarks (B=4, T=4096, D=2048, W=4, BF16), fwd+bwd ms:

| Variant | Custom Triton | Best `torch.compile` |
|---|---|---|
| head-wise H=1 | 0.382 | 0.697 |
| head-wise H=4 | 0.184 | 0.484 |
| head-wise H=16 | 0.143 | 0.421 |
| low-rank R=16 | 0.242 total (0.171 bwd) | 0.946 |
| static CUDA `causal_conv1d` | 0.161 | — |

**[FACT]** Triton kernels are "**1.8–3.9x faster than the best torch.compile baseline**"; head-wise
kernels sustain **2.6–3.0 TB/s** against a **3.35 TB/s** peak.

**[REASONING] Two things to extract from this table.** (a) The op is **flatly memory-bandwidth-bound**
— they report achieved TB/s against peak TB/s, not TFLOP/s against peak TFLOP/s, and they hit
78–90% of peak *bandwidth*. That is the signature of a bandwidth-bound kernel, and it means **the
cost of this op scales with the number of times you touch the activation tensor, not with tap count.**
Four branches implemented as four separate ops read the input four times → expect ~4x the op's cost
unless fused. (b) The gap between hand-written Triton and `torch.compile` is 1.8–3.9x, so a naive
PyTorch implementation of a 4-branch router will be dominated by framework overhead and will *not*
be a fair measurement of the idea. Any throughput claim from a naive implementation is meaningless.

### 1.4 Other results (context)

**[FACT]** Scales trained: dense **150M, 300M, 600M, 1B, 2B** at ~50 tokens/param ("2.5x the
compute-optimal recipe" of Hoffmann et al.); MoE at **7B total / 1B active on 100B tokens**.
Setup: lm-engine, Nemotron-CC, Granite-4 BPE (100,352 vocab), seq len 4096, RMSNorm, SwiGLU, RoPE,
Llama-style pre-norm, AdamW, LR 3e-4, WD 0.1, 10% warmup + cosine to zero.

**[FACT]** Placement ablation (Table 3b; 300M/15B, low-rank R=16, W=4):

| Placement | PPL |
|---|---|
| baseline, no conv | 19.12 (305.2M) |
| Q only | 18.69 |
| K only | 18.83 |
| V only | 18.56 |
| Q+K | 18.44 |
| Q+V | 18.36 |
| K+V | 18.35 |
| **Q+K+V** | **18.10** (309.3M) |

Value projection gives the largest single-projection gain.

**[FACT]** Scaling-law compute advantage over compute-matched Transformers: **1.33x** for the QKV
placement, **1.60x** for the after-every-linear-layer placement.

**[FACT]** Rank ablation (low-rank, W=4): R=4 → 18.26, R=8 → 18.19, R=16 → 18.10, R=32 → 18.04,
R=64 → 17.87, R=128 → 17.85 (336.8M). They pick R=16 as the best performance/parameter trade-off.
Head-size ablation (head-wise, W=4): H=8 → 18.03 (330.5M), H=16 → 18.08, H=32 → 18.21,
H=64 → 18.25, H=128 → 18.40 (306.9M).

**[REASONING] This is the most important secondary finding in the paper.** Compare the two axes:
- Increasing **span** (W: 3→6): **0.00 ppl**, essentially free-but-useless.
- Increasing **router/generator capacity** (R: 16→128): **0.25 ppl**, monotone, still improving at
  R=128.

The gains in this family of methods come from *how expressively the filter is conditioned on the
input*, **not from how far back the filter reaches**. The proposed experiment allocates its entire
innovation budget to the axis that the literature says is flat (span) and only a token amount to the
axis that is steep (conditioning capacity — a 4-way softmax is an extremely low-capacity conditioning
mechanism, about the weakest possible: 4 numbers per token, simplex-constrained).

**[FACT]** Cross-architecture: gains extend to Gated DeltaNet (18.93 → 17.95 low-rank) and
Mamba-2 (20.26 → 18.72), replacing their built-in static convs. And with QK-norm (Table 3c):
QK-norm baseline 18.69, +static 18.56, +dynamic head-wise 18.30, +dynamic low-rank 17.95 — with the
note that static convs "provide little benefit when combined with QK-norm."

**[FACT]** Table 1 headline rows (Nemotron ppl ↓ / LAMBADA ppl ↓ / Wikitext ppl ↓ / task avg ↑):

| Model | Nemotron | LAMBADA | Wikitext | Task avg |
|---|---|---|---|---|
| 305M Transformer | 19.12 | 76.62 | 30.50 | 47.26 |
| 305M + dynamic low-rank | 18.01 | 56.66 | 27.98 | 48.90 |
| 1.82B Transformer | 11.71 | 17.28 | 15.86 | 58.35 |
| 1.88B + dynamic low-rank (QKV) | 11.24 | 15.43 | 14.98 | 59.70 |
| 1.88B + dynamic (all-linear) | 10.95 | 12.51 | 14.43 | 60.70 |
| MoE Transformer | 9.86 | 11.55 | 13.27 | 62.46 |
| MoE + dynamic low-rank | 9.58 | 10.92 | 12.77 | 63.42 |

---

## 5. EQUIVALENCE AND IDENTIFIABILITY (the most important section)

This section is **[REASONING]** plus **numerical verification I ran** (script logic below; all
residuals at machine precision, ~1e-15).

### 5.1 The fixed-weight collapse — verified

**Claim.** A mixture of four causal 3-tap depthwise kernels at dilations {1,2,4,7} with
**input-independent** mixing weights $\alpha \in \Delta^3$ is *exactly* a single causal 15-tap
depthwise kernel with a fixed sparsity pattern.

**Proof.** Branch $b$ with dilation $d_b$ and taps $w^{(b)}_k$, $k \in \{0,1,2\}$, computes
$y^{(b)}_t = \sum_{k=0}^{2} w^{(b)}_k \, x_{t - k d_b}$. The mixture is

$$y_t = \sum_{b=0}^{3} \alpha_b \sum_{k=0}^{2} w^{(b)}_k\, x_{t-k d_b}
      = \sum_{\ell=0}^{14} \Big( \underbrace{\sum_{b,k \,:\, k d_b = \ell} \alpha_b w^{(b)}_k}_{=:\ \kappa_\ell} \Big)\, x_{t-\ell}
      = \sum_{\ell=0}^{14} \kappa_\ell\, x_{t-\ell}.$$

Convolution is linear in its kernel, so a fixed convex combination of convolutions is the convolution
by the combined kernel. **VERIFIED NUMERICALLY: max abs error 1.3e-15** between the 4-branch mixture
and a single 15-tap dense causal conv with $\kappa$ constructed as above.

**Same holds for the dense-width variant** ({3,5,9,15}): $\kappa_\ell = \sum_{b: W_b > \ell} \alpha_b w^{(b)}_\ell$,
also exactly a single 15-tap kernel. So **both** input-independent variants collapse to
"one 15-tap causal depthwise kernel."

### 5.2 The collapse is worse than "equivalent" — it is a *degenerate, over-parameterized* 15-tap kernel

I computed the exact lag coverage. **This is the sharpest finding in this dossier.**

**Dilated 3-tap, dilations {1,2,4,7}** — lags hit and their multiplicity:

| Lag | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| # branches contributing | **4** | 1 | 2 | **0** | 2 | **0** | **0** | 1 | 1 | **0** | **0** | **0** | **0** | **0** | 1 |

- **12 parameters per channel, but only 7 distinct lags reached: {0, 1, 2, 4, 7, 8, 14}.**
- **8 of 15 lags are STRUCTURALLY ZERO and unlearnable: {3, 5, 6, 9, 10, 11, 12, 13}.** Coverage is
  **7/15 = 47%**.
- Lag 0 is covered **4x redundantly** (every branch has a $k=0$ tap at lag 0). Lags 2 and 4 are
  covered 2x. So of 12 parameters, **5 are pure redundancy** (3 duplicate lag-0, 1 duplicate lag-2,
  1 duplicate lag-4), leaving **7 effective degrees of freedom**.

**[REASONING] Consequences.**
1. The dilated variant is **not** a 15-span kernel. It is a **7-tap kernel scattered over a 15-wide
   window**, spending 12 parameters to buy 7 DOF. A dense 15-tap kernel has **15 DOF for 15
   parameters** — strictly more expressive, strictly better parameter efficiency, and *fewer taps
   than the 32-tap dense-branch variant*.
2. The gaps are not benign. **The model cannot see lags 3, 5, 6, or 9–13 at all** in this layer. For
   a language model, lag 3–6 is exactly the range where local n-gram/morphology structure lives —
   and per Sieberling et al.'s width sweep, lags 0–2 carry nearly all the signal, so the taps that
   *are* present at 7, 8, 14 are the least valuable ones while the moderately valuable 3–6 are
   deleted. This is the **gridding artifact** pathology in 1-D.
3. **MixConv (arXiv 1907.09595) tested exactly this substitution and rejected it — [FACT].** They
   emulated a KxK kernel with 3x3 at dilation (K-1)/2 and report accuracy "drops quickly for large
   kernels," attributed explicitly to **skipping local information at high dilation rates**. They
   **excluded dilated convs from their NAS search space** as a result. This is direct published
   evidence against the dilated parameterization for mixed-kernel designs.

### 5.3 What token-dependent routing actually adds — and the exact baselines it forces

**[REASONING]** With **token-dependent** weights $\alpha^{(t)} \in \Delta^3$, the same algebra gives

$$y_t = \sum_{\ell=0}^{14} \kappa^{(t)}_\ell\, x_{t-\ell}, \qquad \kappa^{(t)}_\ell = \sum_{b,k\,:\,k d_b = \ell} \alpha^{(t)}_b w^{(b)}_k .$$

**VERIFIED NUMERICALLY: max abs error 8.9e-16** against a per-token 15-tap kernel construction.

So the router does **not** create a new operator class. It produces a **dynamic (input-dependent)
15-tap causal depthwise kernel** — i.e. exactly the object arXiv 2606.03825 already studies — but
with the tap vector **constrained to a 4-dimensional convex cone** spanned by the four fixed branch
filters $\{w^{(b)}\}$ embedded at their dilated lags, rather than free in $\mathbb{R}^{15}$.

**This reframing is decisive and should be stated plainly in the writeup:**

> The proposal is a **rank-4, simplex-constrained special case of dynamic short convolution.**
> Sieberling et al. (arXiv 2606.03825) implement the **unconstrained** version (any $w^{(t)} \in
> \mathbb{R}^{W \times D}$, affine in the input, no simplex constraint), show it beats static convs,
> and separately show that **the payoff comes from generator capacity (rank 16→128 buys 0.25 ppl) and
> not from span (W 3→6 buys 0.00 ppl)**. The proposal restricts the generator to the lowest possible
> capacity (4 simplex weights) and spends its budget on the axis measured to be flat.

**[REASONING] Therefore the following baselines are MANDATORY. Without them, any "multiscale helps"
claim is vacuous, because each baseline is a cheaper explanation of any observed gain:**

| # | Baseline | What it rules out | Params/channel |
|---|---|---|---|
| **B0** | LFM2 baseline, k=3 | the null | 3 |
| **B1** | **Single dense k=15 causal depthwise conv** | **"the gain is just more span"** — MANDATORY. This is a *parameter-cheaper* (15 vs 32) and strictly more expressive superset of the fixed-mix dense variant. | 15 |
| **B2** | Single dense k=5 and k=9 | locates where the span curve actually saturates; replicates Sieberling's flat curve in your setup | 5, 9 |
| **B3** | **4 branches, FIXED (learned, input-independent) mixing weights** | **"the gain is input-dependence"** — MANDATORY. By 5.1 this is provably equal to B1-with-a-sparsity-mask, so if it matches the router, the router adds nothing. | 12 or 32 (→7 or 15 DOF) |
| **B4** | 4 branches, **uniform frozen** weights (1/4 each) | "the gain is from the learned mix at all" | same |
| **B5** | **Parameter-matched dense conv**: single dense kernel with the *same total tap count* (k=32 for the dense-branch variant, k=12 for the dilated) | **"the gain is just more parameters"** — MANDATORY. Note this is a *longer* span, so it also tests whether the branch *structure* matters vs raw capacity. | 32 / 12 |
| **B6** | **Unconstrained dynamic short conv at k=3 and k=15** (Sieberling low-rank, R∈{16,32}) | **"a general input-dependent filter beats your constrained one"** — MANDATORY if you want to claim the router design is good. Expect this to win. | 3/15 + generator |
| **B7** | Router present but **input replaced by a constant** (ablate the router's input, keep its params) | that the router is reading the token at all vs acting as a learned constant | — |
| **B8** | **FLOP/wall-clock-matched B0**: baseline k=3 with the saved compute spent on width/depth/more tokens | **"the gain is just more compute"** — MANDATORY. The router costs ~4x the conv op; if you can buy more with those FLOPs elsewhere, the idea loses even if it beats B0. | — |

**[REASONING] The critical logical point:** B1 (dense k=15) has **15 params/channel** and **15 DOF**.
The dilated 4-branch variant has **12 params/channel** and only **7 DOF**. The dense 4-branch variant
has **32 params/channel** and still only **15 DOF**. So:

- The dense-branch variant uses **2.1x the parameters of B1 to achieve exactly B1's expressiveness**
  (before routing). It is strictly dominated in the fixed-weight case.
- The dilated variant is **cheaper than B1 but strictly less expressive** (7 DOF vs 15, with 8 lags
  structurally zeroed).
- **Only input-dependence can rescue either.** And input-dependence is better obtained by the
  unconstrained method (B6) that the cited paper already validated.

### 5.4 State-size accounting — the "bounded tiny state" claim degrades 7x

**[REASONING], arithmetic verified.** For autoregressive decode, a causal depthwise conv must cache
the previous $\max\text{lag}$ activations per channel (a ring buffer). Max lag = span - 1.

**Grounding in the real LFM2 config [FACT]** — from
<https://huggingface.co/LiquidAI/LFM2-1.2B/raw/main/config.json>: `conv_L_cache: 3`,
`conv_bias: false`, `conv_dim: 2048`, `hidden_size: 2048`, `num_hidden_layers: 16`,
`full_attn_idxs: [2,5,8,10,12,14]` — so **10 conv layers and 6 GQA layers**, and the conv cache length
is **3**. In the HF implementation the kernel size *is* `conv_L_cache`, i.e. the buffer holds the last
$k$ positions (current included) rather than the $k-1$ strictly-past ones.

So the growth ratio is **15/3 = 5x** under LFM2's own convention, or **14/2 = 7x** counting
strictly-past lags. Both are correct under their respective conventions; I report the LFM2 convention
as primary since it matches the shipped code.

| Variant | Max lag | Cache slots/channel (LFM2 conv.) | Taps/channel | Cache @ d=2048, bf16 |
|---|---|---|---|---|
| **LFM2 baseline k=3** | **2** | **3** | 3 | **12.0 KB/layer** |
| dense k=15 (B1) | 14 | 15 | 15 | 60.0 KB/layer |
| dense branches {3,5,9,15} | 14 | 15 | 32 | 60.0 KB/layer |
| **dilated 3-tap d={1,2,4,7}** | **14** | **15** | 12 | **60.0 KB/layer** |

Across the whole model (10 conv layers): **120 KB → 600 KB**, i.e. **+480 KB of persistent decode
state**, independent of batch (per sequence).

**Key points:**
1. **The cache length is set by MAX LAG, not by tap count.** All span-15 variants — including the
   "cheap" 12-tap dilated one — need **15** cache slots. The dilated variant gets **no state savings**
   from its sparsity: $x_{t-14}$ must be retained, and retaining it means holding 14 steps of history
   in the buffer regardless of which lags you read. (You could store only the 7 needed lags in 7
   separate shift registers with different strides — but $x_{t-14}$ still has to survive 14 decode
   steps, so the aggregate history depth is unchanged. **No saving, more bookkeeping.**)
2. **Conv state grows 5x (3 → 15 slots, LFM2 convention).** At d_model=2048, bf16:
   **12 KB → 60 KB per conv layer**; **120 KB → 600 KB** across the 10 conv layers.
3. **[REASONING] How much this matters depends on the deployment target, and here the answer is
   uncomfortable.** Sanity check against attention: one GQA layer's KV cache at
   `num_key_value_heads=8`, head_dim=64 (so 512 KV channels), 4096 context, bf16 is
   $512 \times 4096 \times 2 \times 2 = 8$ MB — so 600 KB of conv state across all 10 conv layers is
   still small next to 6 attention layers' ~48 MB. **The absolute claim "tiny state" survives.**
   **But two things do not.** (a) The specific claim that the conv contributes a *k=3-sized* bounded
   state is simply false at 5x. (b) **[FACT]** LFM2's blog states the reliance on short convolutions
   "originates from the target device class, the **embedded SoC CPU**," and that STAR's efficiency
   objective was "peak memory usage and prefill+decode speed on Qualcomm Snapdragon embedded SoC CPUs."
   **[REASONING] On an embedded CPU, a 5x conv-state increase plus a 4-branch fragmented op is exactly
   the wrong direction** — and note ShuffleNetV2's fragmentation penalty was *milder* on ARM than GPU
   (Section 7.3), so the branch cost may be tolerable on CPU while the *memory* cost is not. If the
   motivation for this architecture family is on-device inference, the state growth is a first-order
   regression, not a footnote.
4. **[REASONING] The honest framing:** if you want span 15, you pay 15 slots regardless of
   parameterization. There is **no version of this proposal that keeps the k=3 state footprint.**
   The state cost is a property of the *span*, and it is identical for B1 (dense k=15), which is
   more expressive and cheaper in parameters. **The dilated variant's parameter savings buy nothing
   in state and cost expressiveness** — it is the worst point on this tradeoff.


---

## 6. Interpretation caveats and a rigorous knockout protocol

The brainlift is correct that router weights are not explanations. The literature backs this strongly,
and the direct analogue is the attention-interpretability debate.

### 6.1 "Attention is not explanation" — the direct methodological analogue

**[FACT]** **Jain & Wallace, "Attention is not Explanation"** — arXiv:1902.10186,
<https://arxiv.org/abs/1902.10186> (NAACL 2019). Code:
<https://github.com/successar/AttentionExplanation>. From the abstract: attention weights are
"frequently uncorrelated with gradient-based measures of feature importance," and one can
"identify very different attention distributions that nonetheless yield equivalent predictions."
Conclusion: standard attention modules "do not provide meaningful explanations and should not be
treated as though they do."
**[UNKNOWN]** I could not retrieve the full text (ar5iv conversion is broken for this ID); the exact
median TVD / Kendall-tau numbers are not verified here. The two headline mechanisms — (a) correlation
with gradient importance, (b) existence of alternative attention distributions preserving predictions
— are confirmed from the abstract.

**[FACT]** **Wiegreffe & Pinter, "Attention is not not Explanation"** — arXiv:1908.04626,
<https://arxiv.org/abs/1908.04626>, EMNLP-IJCNLP 2019 pp. 11–20, <https://aclanthology.org/D19-1002/>.
They argue the claim "depends on one's definition of explanation" and that a test "needs to take into
account all elements of the model" rather than the attention layer in isolation. They propose
**four tests**, quoted from the abstract:
1. "a simple uniform-weights baseline"
2. "a variance calibration based on multiple random seed runs"
3. "a diagnostic framework using frozen weights from pretrained models"
4. "an end-to-end adversarial attention training protocol"

Their conclusion: where dependable adversarial attention distributions exist, "they don't perform well
on the simple diagnostic," so "prior work does not disprove the usefulness of attention mechanisms for
explainability."
**[UNKNOWN]** Full-text numbers (per-dataset F1 deltas for the uniform baseline, the seed-variance
JS-divergence figures) not retrieved — the PDF is a scanned/compressed stream I could not extract.

**[REASONING] Direct transfer to this experiment — this is the single most actionable methodological
import.** Wiegreffe & Pinter's tests 1 and 2 map onto the router almost exactly, and they are cheap:
- **Uniform-weights baseline** → retrain with router weights **frozen at 1/4**. If performance matches
  the learned router, the router's weights carried no usable information, exactly as uniform attention
  matching learned attention would indict attention. This is baseline **B4** in Section 5.3.
- **Seed-variance calibration** → **train ≥3 seeds** and report the spread. Any router-weight pattern
  (e.g. "layer 7 prefers the span-15 branch") that is not stable across seeds is a seed artifact, not
  a finding. This is the most commonly skipped and most important control.

### 6.2 Head/branch knockout: ablation-at-inference vs ablation-with-retraining

**[FACT]** **Michel, Levy & Neubig, "Are Sixteen Heads Really Better than One?"** — arXiv:1905.10650,
<https://arxiv.org/abs/1905.10650> (NeurIPS 2019). Exact numbers:
- WMT encoder self-attention: **only 8 of 96 heads** cause a statistically significant change when
  removed individually — i.e. **88/96 ≈ 91.7% show no significant change**. Base BLEU 36.05. **Half
  of those 8 significant cases actually *raised* BLEU.**
- Their conclusion: "at test time, most heads are redundant given the rest of the model."
- Ablating all-but-one head in a layer: WMT's last encoder-decoder layer lost **−13.56 BLEU**; but
  **no BERT layer's single-head reduction was significant at p<0.01** (deltas −0.96% to +0.10%).
- Greedy iterative pruning: **20% of heads** prunable on WMT and **~40% on BERT/MNLI** with no
  noticeable impact (appendix: ~60% SST-2, ~50% CoLA/MRPC). Enc-Dec attention was the fragile one:
  beyond 60% pruned → "catastrophic performance degradation."
- **Importance proxy score (exact):** $I_h = \mathbb{E}_{x\sim X}\left|\partial \mathcal{L}(x)/\partial \xi_h\right|
  = \mathbb{E}_{x\sim X}\left|\mathrm{Att}_h(x)^\top \partial\mathcal{L}(x)/\partial \mathrm{Att}_h(x)\right|$
  where $\xi_h$ is a mask variable on head $h$. Absolute value is taken "to avoid datapoints with
  highly negative or positive contributions from nullifying each other." Normalized per layer by
  $\ell_2$ norm. Cost: one forward + one backward pass.
- **Ablation-at-inference vs retraining — [FACT]:** the paper does **NOT** run retraining after
  ablation. All results are test-time ablation on trained models. They explicitly note neither model
  "can be reduced to a purely single-head attention model without retraining or incurring substantial
  losses to performance." So retraining is flagged as the untested alternative.
- **Speedup from pruning 50% of BERT heads** (MNLI-matched, examples/sec, GTX 1080Ti): batch 1
  17.0→17.3 (**+1.9%**), batch 4 67.3→69.1 (**+2.7%**), batch 16 114.0→134.0 (**+17.5%**),
  batch 64 124.7→146.6 (**+17.5%**). **Gains only appear at larger batch sizes.**

**[REASONING] Two lessons that bear directly on the branch-knockout plan.**
1. **Test-time knockout systematically *understates* importance** because the rest of the network has
   redundant pathways — 91.7% of heads looked useless at inference. Symmetrically, test-time knockout
   can *overstate* importance when the model is off-distribution. Therefore: **branch knockout at
   inference is necessary but not sufficient. You must also retrain without the branch.** The
   retrain-from-scratch comparison is the only one that answers "does this branch add capability."
2. **The +1.9% at batch 1 vs +17.5% at batch 64 result is a warning about your own throughput
   measurement**: removing components gives near-zero wall-clock benefit at small batch because the
   op is latency/overhead-bound, not FLOP-bound. Same physics will govern your 4-branch conv.

### 6.3 Activation patching best practices — how to do the causal test correctly

**[FACT]** **Zhang & Nanda, "Towards Best Practices of Activation Patching in Language Models:
Metrics and Methods"** — arXiv:2309.16042, <https://arxiv.org/abs/2309.16042> (ICLR 2024).
They "systematically examine the impact of methodological details in activation patching, including
evaluation metrics and corruption methods," and find "varying these hyperparameters could lead to
disparate interpretability results." Scope: decoder-only models up to 6B params.

Concrete recommendations **[FACT]**:
- **Corruption method: prefer symmetric token replacement (STR) over Gaussian noise (GN).** Quote:
  "we recommend STR whenever possible. GN may be considered as an alternative when token alignment or
  lack of analogous tokens makes STR unsuitable."
- **The off-distribution argument (this is the key citation for "don't zero-ablate")** — quote:
  "GN corruption puts the model off distribution by introducing noise never seen during training,"
  which "may induce unreliable or illusory results." STR "provides counterfactual prompts... that are
  in-distribution and thus induces in-distribution activations." Crucially they state the concern
  **generalizes "to any intervention techniques that introduce OOD inputs to the model or its
  internal layers, including ablations."**
- **Evidence for it:** on IOI, Name Mover heads put 0.58 attention on the indirect object on clean
  prompts; under GN this splits (0.26 / 0.21). Restoring S-Inhibition head values recovers logit
  difference **1.04 under STR but only 0.49 under GN** — a ~2x distortion purely from the corruption
  choice.
- **Metric: prefer logit difference; avoid raw probability.** "we generally recommend avoiding using
  probability as the metric, given that it may fail to detect negative model components." Probability
  missed negative name mover head 11.10 under STR. KL divergence "can be a reasonable metric for
  circuit discovery as well."
- **Sliding windows:** window results "are to be interpreted as the joint effects of the full window,
  rather than of a single layer"; "we recommend experimenting with single-layer patching first and
  only consider sliding window patching when individual layers seem to induce small effects."
  Windows gave **1.40x–1.75x higher peaks** than summed single-layer effects.
- **Vary which token you corrupt:** "we recommend trying out different tokens to corrupt when the
  problem setting offers such flexibility." Corrupting S1 and IO surfaced all three Name Mover heads;
  corrupting S2 largely missed them.
- **[FACT]** Noising-vs-denoising: they study **only** corrupt→restore-clean, and flag the reverse as
  future work: "The other direction—patching corrupted to clean—has also been used for circuit
  discovery, and it is interesting to compare these two." So **[UNKNOWN]/open**: no published
  asymmetry analysis in this paper.

**[REASONING] Transfer:** the "zero-ablation is off-distribution" warning applies directly. If you
knock out branch 4 by setting its output to **zero**, you have pushed the block's activations off the
distribution the downstream layers were trained on, and the resulting damage conflates "this branch
was doing useful work" with "you broke the activation statistics." **Prefer mean-ablation (replace the
branch output with its dataset mean) or resample-ablation (substitute the branch output from a
different, random token/sequence position).** Resample-ablation is the in-distribution analogue of STR.

### 6.4 RECOMMENDED KNOCKOUT PROTOCOL (concrete, ordered, cheap-first)

**[REASONING]** — my synthesis of 6.1–6.3, designed so each step can kill the hypothesis early.

**Tier 0 — necessary conditions on the router itself (free, no retraining).**
1. **Router entropy AND per-layer weight variance over training.** Log mean per-token entropy of the
   4-way softmax per layer per step, plus **the standard deviation of each branch weight across tokens**.
   If entropy converges to ~$\ln 4 = 1.386$ (uniform), the router is not routing → fixed mixture → jump
   to L3/L4. If it collapses to ~0 with the *same* branch everywhere, it is a constant → also a fixed
   mixture. **Benchmark against Branchformer: its learned input-dependent branch weights had std < 0.01
   and lost to a static merge. If your per-token std is that small, you have reproduced that result** —
   and per SE's late-stage saturation, the honest move is to delete the router in those layers and
   confirm the loss is unchanged (SE: <0.1% top-5 for removing its degenerate gates).
2. **Input-dependence test (the crucial one).** Compute the variance of $\alpha^{(t)}$ decomposed
   into between-layer, between-position, and between-token-identity components. **Report the fraction
   of router-weight variance explained by token identity alone** (i.e. fit $\alpha$ from the token ID
   with a lookup table). If a static per-token-ID table reproduces the router, the router is a learned
   embedding, not a context-sensitive mechanism. If a per-*position* table reproduces it, it is a
   positional schedule and should be replaced by a fixed per-layer span (much cheaper).
3. **Seed stability (Wiegreffe & Pinter test 2).** ≥3 seeds. Report whether the per-layer branch
   preference pattern replicates. **Do not report any router-weight narrative that is not stable
   across seeds.**

**Tier 1 — inference-time causal knockout (cheap, but understates; per Michel et al.).**
4. For each branch $b$: **resample-ablate** it (substitute its output from another random position in
   the batch) and measure Δ loss. Also **mean-ablate**. **Report both, and report zero-ablation only
   as a labeled-biased third number** with the Zhang & Nanda off-distribution caveat.
5. **Renormalize vs not**: after killing branch $b$, either (a) renormalize the remaining softmax over
   3 branches, or (b) leave the mixture summing to <1. These give different answers; report (a) as
   primary since it preserves output scale.
6. **Metric**: use **Δ log-likelihood / logit difference on targeted probes**, not just aggregate ppl.
   Aggregate ppl will move ~0 for a single branch in a redundant network (Michel's 91.7% lesson).
7. **Targeted probe suites** where long-range local structure should matter: bracket/quote matching at
   controlled distances, induction/associative recall at controlled lag, and — most direct —
   **construct a synthetic task whose answer depends on a token exactly $L$ positions back, sweep
   $L \in \{1..14\}$, and check that knocking out the span-15 branch selectively damages large $L$.**
   This is the *only* test that would show branch specialization is real and span-shaped. If the damage
   curve is flat in $L$, the branches are not doing what the multiscale story claims.

**Tier 2 — retraining knockout (expensive, but the only sufficient test).**
8. **Retrain from scratch without branch $b$** (each of the 4, or at least the widest). Compare to the
   full model. **The gap here — not the inference-time gap — is the branch's true contribution.**
9. **Retrain with the router frozen to the *learned average* weights.** If this matches the live
   router, input-dependence is worthless (this is B3, and by Section 5.1 it collapses to one 15-tap
   kernel).
10. **Freeze the trained router and transplant it** into a fresh model trained from scratch
    (Wiegreffe & Pinter test 3, "frozen weights from pretrained models"). If the frozen router
    performs as well as a learned one, the routing *function* is trivial/transferable.

**Tier 3 — adversarial router (Wiegreffe & Pinter test 4).**
11. Train an **adversarial router** end-to-end: maximize divergence from the original router's weights
    subject to matching the original model's predictions within $\epsilon$. If such a router exists,
    the specific weight pattern you observed is **not** the unique explanation and no interpretive
    claim about it survives.

**Reporting rule [REASONING]:** state up front that router weights are a *description of the
computation performed*, not evidence about *what information is used*. The only claims licensed are
those surviving Tier 1 + Tier 2.


---

## 7. Throughput reality

### 7.1 Depthwise causal conv is hard memory-bound — the roofline

**[REASONING], arithmetic I computed.** For a depthwise causal conv with width $W$ on
$B{\times}T{\times}D$ = 4x4096x2048 in bf16, FLOPs $= BTDW\cdot 2$ but HBM traffic is
$\approx BTD \cdot 2\,\text{bytes} \cdot 2$ (read $x$ once, write $y$ once; the weight tensor is
$D{\times}W$, negligible):

| W | FLOPs | HBM bytes | Arithmetic intensity |
|---|---|---|---|
| 3 | 2.01e8 | 1.34e8 | **1.5 FLOP/byte** |
| 15 | 1.01e9 | 1.34e8 | **7.5 FLOP/byte** |

An H100 SXM has ~990 TFLOP/s bf16 dense and ~3.35 TB/s HBM → **ridge point ≈ 296 FLOP/byte**.
At 1.5–7.5 FLOP/byte the op sits **~39–200x below the ridge point**: it is *hard* memory-bandwidth-bound,
and its cost is essentially "how many times you stream the activation tensor."

**[FACT] This is confirmed empirically by arXiv 2606.03825 itself**: their Triton short-conv kernels
sustain **2.6–3.0 TB/s against a 3.35 TB/s peak** (78–90% of peak *bandwidth*), and they report
achieved TB/s rather than TFLOP/s — the signature of a bandwidth-bound kernel.

**[FACT] ShuffleNet V2 (arXiv 1807.11164, <https://arxiv.org/abs/1807.11164>) guideline G4**
explicitly classifies **depthwise convolution** as belonging with element-wise ops "given its high
MAC-to-FLOPs ratio." Stripping ReLU + shortcut from a bottleneck gave **~20% speedup** on both GPU and
ARM, showing pure-memory-traffic ops are a first-order cost.

### 7.2 THE most important consequence: a fused wide kernel is nearly free; four branches are not

**[REASONING] — this reframes the entire cost/benefit of the proposal.**

Because traffic is $O(BTD)$ and **independent of $W$**:
- A **single fused 15-tap** depthwise causal conv streams $x$ once and writes $y$ once — **the same
  traffic as the k=3 baseline**. In a memory-bound regime its wall-clock cost is **nearly identical to
  k=3** (only the tiny weight read and the in-register FMA count grow, and FMAs are 39x under the
  ridge). **Widening the kernel from 3 to 15 is essentially free on a GPU.**
- **Four separate branch convs** each re-read $x$ (67 MB at this shape) and each write their own output,
  then a 4-way weighted sum adds several more elementwise passes. Traffic goes up **~4x or more**.

**So the proposal has the cost structure exactly backwards.** The expensive thing is *branch
multiplicity* (memory traffic), and the free thing is *span* (FLOPs). If you want span 15, the
cheapest possible implementation is **one fused 15-tap kernel** — which by Section 5.1 is *exactly*
what the input-independent 4-branch version computes anyway, and by Section 5.2 is *more expressive*.

**[REASONING] Fusion escape hatch.** You can avoid the 4x traffic even with the router: materialize
the per-token 15-tap kernel $\kappa^{(t)}_\ell = \sum_{b,k: kd_b=\ell}\alpha^{(t)}_b w^{(b)}_k$ and run
**one** dynamic 15-tap conv. This is the CondConv "combine kernels, then convolve once" trick, and it
makes the router's cost ~the router MLP plus one wide dynamic conv. **But note what this reveals:**
once fused, you are literally running Sieberling et al.'s dynamic short conv at W=15 with a rank-4
simplex-constrained generator. There is no separate "multiscale" object left.

### 7.3 Network fragmentation: the hard measured penalty for 4 branches

**[FACT] ShuffleNet V2 (arXiv 1807.11164) guideline G3, "Network fragmentation reduces degree of
parallelism."** Blocks of 1–4 1x1 convs in series or parallel, stacked 10x, 56x56 input, **channel
counts adjusted for FLOPs parity**. Throughput (GPU = batches/sec, CPU = images/sec):

| Structure | GPU c=128 | c=256 | c=512 | CPU c=64 | c=128 | c=256 |
|---|---|---|---|---|---|---|
| **1-fragment** | **2446** | **1274** | **434** | 40.2 | 10.1 | 2.3 |
| 2-frag-series | 1790 | 909 | 336 | 38.6 | 10.1 | 2.2 |
| 4-frag-series | 752 | 745 | 349 | 38.4 | 10.1 | 2.3 |
| 2-frag-parallel | 1537 | 803 | 320 | 33.4 | 9.1 | 2.2 |
| **4-frag-parallel** | **691** | **572** | **292** | 35.0 | 8.4 | 2.1 |

**At matched FLOPs, 4-way parallel fragmentation is 2446/691 = 3.54x slower on GPU at c=128**
(2.23x at c=256, 1.49x at c=512). The paper attributes this to "extra overheads such as kernel
launching and synchronization," and notes NASNET-A uses 13 fragmented operators per block vs 2–3 in
ResNet. **[REASONING] This is the single most directly transferable number in the dossier**: a
4-branch parallel decomposition of a small op, at matched FLOPs, cost ~2–3.5x on GPU. The penalty
shrinks as channels grow (more work per launch), so at LM scale (D=2048, T=4096) expect the milder
end — but the direction and mechanism are established.

**[FACT] G2, excessive group convolution increases MAC** (10 stacked pointwise group convs, 56x56):
g=1 → 2451, g=2 → 1725, g=4 → 1026, g=8 → 634 batches/sec on GPU at 1x. **g=8 is >2x slower than g=1**
at matched FLOPs. Depthwise conv is the extreme $g = C$ case.

### 7.4 Available fast implementations — and a hard blocker for the dilated variant

**[FACT] `causal-conv1d` (Mamba ecosystem), <https://github.com/Dao-AILab/causal-conv1d>.** From the
README, the fast CUDA kernel's stated Features are exactly two bullets: "Support fp32, fp16, bf16" and
**"Kernel size 2, 3, 4."** The signature is
`causal_conv1d_fn(x, weight, bias=None, activation=None)` with `x: (batch, dim, seqlen)`,
`weight: (dim, width)` — **there is no `dilation` argument and dilation is never mentioned anywhere on
the page.** Reference equivalent given in the README:
`F.conv1d(x, weight.unsqueeze(1), bias, padding=width-1, groups=dim)[..., :seqlen]`.

**[REASONING] This is a hard, concrete blocker and should drive the implementation plan:**
1. **Width 15 is NOT supported** by the fast kernel. Any dense 15-tap branch falls back to
   `F.conv1d` with `groups=dim`, which is the slow path.
2. **Dilation is NOT supported at all.** So the "cheap" dilated variant — the one whose selling point
   is fewer taps — **cannot use the fast kernel for 3 of its 4 branches** (only dilation=1 works).
   It would run on `F.conv1d(..., dilation=d, groups=dim)`, PyTorch's grouped-conv path, which is
   well known to be poorly optimized for depthwise 1-D.
3. **Corollary:** the dilated variant, marketed as the cheap option (12 taps vs 32), is likely the
   **slowest to actually run**, while the dense k=15 single kernel (15 taps) is the one you can most
   plausibly get a fast fused kernel for. **The parameter count is not the cost model here.**

**★ IMPORTANT UPDATE — the width cap is NOT fundamental (verified in source).** The
`TORCH_CHECK(width >= 2 && width <= 4, ...)` in `causal-conv1d`'s `csrc/causal_conv1d.cpp` (three sites:
fwd 182, bwd 289, update 496) is a **hard abort, not a slow fallback** — width 5+ raises immediately.
**But `flash-linear-attention`'s Triton conv backend has no width limit:**
`fla/modules/conv/triton/ops.py` uses `BW = triton.next_power_of_2(W)`, i.e. arbitrary width.
**[REASONING] Two consequences that matter for the experiment design:**
- **A dense k=15 causal depthwise conv IS implementable with a fast fused kernel today** via FLA's
  Triton path. So the mandatory wide-kernel baseline (B1/L1) is **not** handicapped by tooling, and the
  "one fused wide kernel is nearly free" argument of Section 7.2 is practically realizable. Use that
  path for L1 rather than `F.conv1d`, or the baseline will be unfairly slow.
- The field's width-4 consensus **outlives the constraint that helped create it** (every FLA model still
  defaults to `conv_size=4` despite the backend supporting more). That is a fair argument *for*
  re-examining width — but note the two labs that did re-examine it found flat (W 3→6) or worse (k=4,
  and mixed-width branches at 13.28 vs 12.79).
**[UNKNOWN]** Whether FLA's Triton conv supports **dilation** is still unverified — assume not.
4. **[FACT]** arXiv 2606.03825 measured static CUDA `causal_conv1d` at **0.161 ms** fwd+bwd
   (B=4,T=4096,D=2048,W=4,bf16) and their best `torch.compile` baseline for the dynamic variants at
   **0.42–0.95 ms** vs **0.14–0.38 ms** hand-written Triton — a **1.8–3.9x** gap. **[REASONING] Any
   4-branch router implemented in plain PyTorch will be measuring framework overhead, not the idea.**
   **[UNKNOWN]** I did not verify flash-linear-attention's short-conv width/dilation support; check
   <https://github.com/fla-org/flash-linear-attention> before assuming a fast path exists.

### 7.5 FFT vs direct convolution — not relevant at this scale

**[FACT]** The long-conv literature uses FFT because kernels are as long as the sequence:
Hyena (arXiv 2302.10866, <https://arxiv.org/abs/2302.10866>), H3's FlashConv (arXiv 2212.14052,
<https://arxiv.org/abs/2212.14052>), and FlashButterfly in "Simple Hardware-Efficient Long Convolutions
for Sequence Modeling" (arXiv 2302.06646, <https://arxiv.org/abs/2302.06646>, Fu et al.). The latter
reports **2.2x convolution speedup**, Path256 (length 64K) SOTA by **+29.1 points** and **7.2x** faster
training, and WikiText103 **0.2 PPL** better than a Transformer with **30% fewer** parameters, and
finds "a key requirement to achieving high performance is keeping the convolution kernels smooth" —
achieved by "squashing the kernel weights."
**[UNKNOWN]** None of the pages I retrieved states an explicit numeric FFT-vs-direct crossover in
(seq len, kernel len).

**[REASONING] Verdict for this experiment: FFT is irrelevant.** Direct conv costs $O(TW)$ per channel;
FFT costs $O(T\log T)$ regardless of $W$. At $T \in [1\text{k}, 4\text{k}]$, $\log_2 T \approx 10\text{–}12$,
so FFT only wins when $W \gtrsim \log_2 T \approx 10\text{–}12$ *in FLOP count alone* — and that ignores
FFT's much worse constants (complex arithmetic, 3 passes over a padded 2T buffer, poor tensor-core
utilization). At $W=15$ in a memory-bound regime where direct conv is already ~free (7.2), **FFT
would be strictly worse.** Use direct convolution. The one caveat is that the smoothness/squashing
finding above suggests wide learned kernels may need regularization to train well — relevant if you
adopt the dense k=15 baseline.

### 7.6 The sparsity crossover: why hard routing cannot pay off here

**[FACT] Gale, Zaharia, Young & Elsen, "Sparse GPU Kernels for Deep Learning"** — arXiv:2006.10901,
<https://arxiv.org/abs/2006.10901> (SC20). The crossover number, quoted: **"Using our approach, sparse
computation exceeds the performance of dense at as low as 71% sparsity"** (weight-sparse LSTM problem,
input 8192, hidden 2048, batch 128, FP32, V100). Vendor libraries needed **14x** fewer non-zeros to
reach dense-equivalent performance. Their kernels hit **27.3% of single-precision peak** (4.29 TFLOP/s
SpMM FP32); geometric-mean speedups over cuSPARSE of **3.58x** (SpMM FP32), **2.19x** (SDDMM),
**5.97x** (mixed-precision SpMM). End-to-end: sparse Transformer **2.09x** speedup with **12.8x**
memory savings at matched accuracy (3.77 vs 3.76 bits/dim); sparse MobileNetV1 at 90% sparse 1x1 convs
gave **21–24%** speedups.

**[REASONING] Apply this to top-1 routing over 4 branches.** Top-1 of 4 is **75% sparsity** — barely
above the 71% crossover, and that 71% figure was achieved with *hand-optimized SC20-grade kernels on a
large, compute-bound GEMM* (8192x2048, an op with high arithmetic intensity where there is real FLOP
work to skip). Our op has arithmetic intensity **1.5–7.5 FLOP/byte**, i.e. **there are almost no FLOPs
to save** — the cost is streaming $x$, and a top-1 router **still has to read every token's $x$**.
Worse, hard routing *adds* a gather/scatter (or a mask-and-still-compute), extra kernel launches, and
a synchronization. **You would pay new memory traffic to avoid FLOPs that cost nothing.**

**[REASONING] Concrete verdict: hard/top-k routing over 4 tiny depthwise causal conv branches cannot
beat dense evaluation of all 4 on a GPU.** Reasons, in order of decisiveness:
1. The branches are memory-bound, not FLOP-bound — sparsity saves the wrong resource.
2. Top-1 of 4 = 75% sparsity, at the very edge of the 71% crossover measured for a *compute-bound* op
   with bespoke kernels.
3. Gather/scatter and fragmentation overheads (ShuffleNetV2 G3: up to 3.5x for 4-way parallel at
   matched FLOPs) exceed any plausible saving.
4. The correct optimization is the opposite direction: **fuse all four branches into one 15-tap
   kernel** (Section 7.2), which makes them free — and simultaneously proves the branches were never
   the point.

### 7.7 Expected slowdown ranges and how to measure

**[REASONING] Expected wall-clock, relative to the k=3 baseline conv op** (the op only; end-to-end will
be diluted by attention/MLP, roughly 5–10x, per 2606.03825's ~8% end-to-end for a ~1 op change):

| Implementation | Expected op-level slowdown | Confidence |
|---|---|---|
| Single fused dense k=15 via **FLA Triton** conv (supports arbitrary width) | **1.0–1.3x** — traffic is O(BTD), independent of W | medium-high |
| Single fused dense k=15 via `F.conv1d(groups=D)` (no fast kernel) | **2–4x** (PyTorch slow grouped path) | medium |
| 4 separate `F.conv1d` branches + weighted sum (naive) | **4–8x** (4x traffic + 4 launches + sum passes) | high |
| 4 branches, dilated, `F.conv1d(dilation=d, groups=D)` | **5–12x** (no fast kernel exists; dilated depthwise is the worst-supported path) | medium |
| 4 branches under `torch.compile` (fused epilogue) | **2–4x** | low-medium |
| Kernel-fused: build per-token 15-tap $\kappa^{(t)}$, one dynamic conv (custom Triton) | **1.2–2x** | low (needs the kernel written) |
| Router MLP itself (D→4) | negligible (<1% — it is a rank-4 projection) | high |

**End-to-end training throughput [REASONING]:** expect **5–25% slowdown** for the naive 4-branch
version, benchmarked against arXiv 2606.03825's measured **~8%** for QKV dynamic conv and **22–25%**
for conv-after-every-linear. If your naive implementation shows >30%, the implementation is the
bottleneck, not the architecture.

**Measurement protocol [REASONING]:**
1. **Op-level microbenchmark first**, isolated: fixed shapes B=4, T=4096, D=2048, bf16. **≥25 warmup
   iters** (torch.compile needs recompilation to settle), then ≥100 timed iters. **`torch.cuda.synchronize()`
   around the timed region only** — or better, use CUDA events. Report **median and p10/p90**, not mean.
2. **Time forward AND backward separately** (`loss.backward()` on a scalar reduction). The backward of
   a 4-branch mixture is where the traffic really multiplies — 2606.03825 reports fwd+bwd for this
   reason.
3. **Always report the `torch.compile` and eager numbers side by side.** Given the 1.8–3.9x
   compile-vs-Triton gap in 2606.03825, an eager-only number is not evidence about the architecture.
4. **Report achieved HBM bandwidth (TB/s) against peak**, not TFLOP/s. This is the correct efficiency
   metric for a memory-bound op and is how 2606.03825 reports it (2.6–3.0 of 3.35 TB/s).
5. **End-to-end tokens/sec** on the real training config, ≥200 steps after warmup, same global batch
   and seq len, same seeds. Report as % of baseline.
6. **Decode latency separately**: batch 1 and batch 64. Per Michel et al.'s +1.9% (batch 1) vs +17.5%
   (batch 64), small-batch decode is overhead-bound and will show a different (usually worse) picture.
   Also report the **conv state memory** (Section 5.4) at your target context.
7. **Count kernel launches** (`torch.profiler` or Nsight). If the 4-branch version launches 5+ kernels
   where the baseline launches 1, you have found your slowdown and should fuse before drawing any
   architectural conclusion.


---

## 2. Dilated / multiscale causal convolution prior art

Evidence grades as before. **[DIGITIZED]** marks values read off rendered figures because the paper
published them *only* as plots — treat as approximate (±0.03% on accuracy axes).

### 2.1 WaveNet — arXiv 1609.03499, <https://arxiv.org/abs/1609.03499>

**[FACT]** Dilation doubles per layer then resets; schedule printed verbatim as
`1, 2, 4, ..., 512, 1, 2, 4, ..., 512, 1, 2, 4, ..., 512`. Each block has receptive field **1024**
samples; 3 blocks = 30 dilated layers. Gated activation
$z = \tanh(W_{f,k} * x) \odot \sigma(W_{g,k} * x)$, which "worked significantly better than the
rectified linear activation function." Residual + parameterised skip connections.

**[FACT] Why dilation rather than a wide kernel** (the canonical justification, quoted):
- "One of the problems of causal convolutions is that they require many layers, or large filters to
  increase the receptive field."
- dilation increases the receptive field "by orders of magnitude, without greatly increasing
  computational cost."
- It "is equivalent to a convolution with a larger filter derived from the original filter by dilating
  it with zeros, **but is significantly more efficient**."

**[FACT] What it bought:** MOS naturalness 4.21±0.081 (EN) / 4.08±0.085 (ZH) vs LSTM-RNN parametric
3.67/3.79 and HMM concatenative 3.86/3.47 (natural speech 4.55/4.21). Gap to natural narrowed "from
0.69 to 0.34 (51%)" EN and "0.42 to 0.13 (69%)" ZH. TIMIT **18.8 PER**.

**[FACT] Cost: not reported.** No layer/channel counts, no params, no FLOPs, no latency anywhere.
**[FACT] Admitted limitation** — the 240 ms RF "was not long enough" for F0 prosody (needed an external
F0 model), and for music "even with a receptive field of several seconds, the models did not" achieve
long-range consistency. **[REASONING]** That admission is the earliest documented instance of the core
dilation tradeoff: RF grows exponentially but *coverage density* does not, so the model becomes
long-sighted and locally sparse — exactly the failure mode Section 5.2 quantifies for this proposal.

### 2.2 TCN — Bai, Kolter & Koltun, arXiv 1803.01271, <https://arxiv.org/abs/1803.01271>

**[FACT]** "TCN = 1D FCN + causal convolutions." Dilated conv
$F(s) = \sum_{i=0}^{k-1} f(i)\, x_{s - d\cdot i}$ with $d = O(2^i)$ at level $i$. Residual block = two
dilated causal conv layers + ReLU + weight norm + spatial dropout, plus a 1x1 conv when widths differ.
Explicitly *omits* WaveNet's "skip connections across layers, conditioning, context stacking, or gated
activations."

**[FACT] Correction worth noting:** the paper gives **only the per-layer effective history $(k-1)d$**.
It never states a closed-form total-RF formula. The widely cited $1 + 2(k-1)(2^n - 1)$ is
community-derived, **not in this paper**.

**[FACT] Results** (TCN vs LSTM/GRU/RNN at matched size), selected rows:

| Task | LSTM | GRU | RNN | TCN |
|---|---|---|---|---|
| Seq. MNIST (acc↑) | 87.2 | 96.2 | 21.5 | **99.0** |
| P-MNIST (acc↑) | 85.7 | 87.3 | 25.3 | **97.2** |
| Copy memory T=1000 (loss↓) | 0.0204 | 0.0197 | 0.0202 | **3.5e-5** |
| **Word PTB (ppl↓)** | **78.93** | 92.48 | 114.50 | 88.68 |
| Wiki-103 (ppl↓) | 48.4 | – | – | **45.19** |
| Char PTB (bpc↓) | 1.36 | 1.37 | 1.48 | **1.31** |
| text8 (bpc↓) | 1.50 | 1.53 | 1.69 | **1.45** |

**[FACT] The ablation finding that matters most here — kernel size is task-dependent, and *language*
wants SMALL.** Larger $k$ helped copy-memory and P-MNIST (at $k \le 3$ copy-memory "only converges to
the same level as random guessing"), **but on word-level PTB $k=3$ was best**, and word PTB is the one
task where a tuned LSTM beat TCN (78.93 vs 88.68). The paper's stated reason: "because a smaller kernel
… tends to focus more on the local context."

**[FACT] Dilation was NOT ablated.** Verbatim: "we believe dilation is required for modeling long-term
dependencies, and so we mainly focus on two other factors here." **[REASONING] Flag this:** the two
foundational dilated-causal-conv papers (WaveNet, TCN) and ByteNet all *assert* dilation's value rather
than measuring it against a dense-kernel control. There is no clean published head-to-head of
"dilated sparse taps vs dense taps at equal max lag" in sequence modeling — which is precisely the
comparison this experiment needs, and part of why baseline B1 is mandatory.

**[FACT] Gating ablation (Appendix D) is a genuine mixed result:** Word PTB 88.68→87.94 and Char PTB
1.31→1.306 improved, but Copy memory 3.5e-5→0.00508 (**~145x worse**), P-MNIST 97.2→96.9, JSB
8.10→8.13, Nottingham 3.07→3.12, text8 1.45→1.485. Gating "roughly doubles the conv layers."

**[FACT] Cost:** at eval a TCN "must retain raw input up to the effective history length, thus possibly
requiring more memory during evaluation," vs an RNN keeping only $h_t$. **[REASONING] This is exactly
the Section 5.4 state-growth point, stated by the TCN authors themselves.** No wall-clock anywhere.

### 2.3 ByteNet — arXiv 1610.10099, <https://arxiv.org/abs/1610.10099>

**[FACT]** Decoder stacked on encoder preserving temporal resolution. Dilation "doubled every layer up
to a maximum rate r (for our experiments r=16)", "repeated multiple times … always starting from a
dilation rate of 1" ⇒ 1,2,4,8,16 | 1,2,4,8,16 | … Decoder convs masked; encoder not. LayerNorm not
BatchNorm (BN "would compute statistics over future tokens"). LM config: 30 residual blocks, masked
kernel 3, d=512 ⇒ **receptive field 315 characters**.
**[FACT] Results:** enwik8 **1.31 bits/byte** (beating Recurrent Highway Networks 1.32, HM-LSTM 1.40,
Stacked LSTM 1.67); WMT En→De **23.75 BLEU (Test'14)**, **26.26 (Test'15)**.
**[FACT] Internal discrepancy to flag:** the v2 abstract claims "22.85 … and 25.53" while Tables 2/4
report 23.75 and 26.26; never reconciled. Cite the tables.
**[FACT] No dilation ablation, no params/FLOPs/latency.**

### 2.4 MixConv / MixNet — arXiv 1907.09595, <https://arxiv.org/abs/1907.09595> — MOST RELEVANT

Tan & Le, BMVC 2019. This is the closest published analogue to the dense-multi-width variant.

**[FACT] Mechanism — CONFIRMED: there is NO router.** Static channel partitioning + concat, fixed at
design/search time. The paper's own TF demo is literally `tf.split` → per-group `depthwise_conv2d` →
`tf.concat(y, axis=-1)`. No gate, no softmax over branches, no learned branch weighting anywhere.
Despite the name "Mix," it is architecturally unrelated to mixture-of-experts. **[REASONING]** The only
place cross-branch information actually combines is the *subsequent 1x1 pointwise conv* in the
MobileNet block.

**[FACT] Kernel sizes are deterministic, not searched:** "we restrict kernel size always starts from
3x3, and monotonically increases by 2 per group … **group $i$ always has kernel size $2i+1$**." So g=4
→ {3,5,7,9}. Recommendation "**g=4 is generally a safe choice for MobileNets**" — stated with **no
supporting number**. Partitioning: equal (8,8,8,8) or exponential (16,8,4,4) for 4 groups/32 filters;
exponential "only performs slightly better … on MobileNetV1, but **there is no clear winner**."

**[FACT/DIGITIZED] Group-count sweep exists ONLY as Figure 4 — no table in the paper.** Digitized:

| g | kernels | V1 Top-1 | Δ | V2 Top-1 | Δ |
|---|---|---|---|---|---|
| 1 | {3} | 71.10 | — | 72.90 | — |
| 2 | {3,5} | 72.02 | **+0.92** | 73.49 | **+0.59** |
| 3 | {3,5,7} | 72.15 | +0.13 | 73.86 | +0.37 |
| 4 | {3,5,7,9} | 72.40 | +0.25 | 74.06 | +0.20 |
| 5 | {3..11} | 72.45 | +0.05 | 74.29 | +0.23 |
| 6 | {3..13} | 72.56 | +0.11 | 74.31 | **+0.02** |

**[REASONING] The knee is at g=2, not g=4.** ~55–65% of the entire multiscale gain arrives from adding
*one* extra kernel size. By g=6 the marginal return is +0.11 / +0.02 — zero. Mixed kernels **saturate
but do not degrade**.

**[FACT] What DOES degrade — two findings, both directly adverse to this proposal:**

**(a) Single large kernels degrade.** [DIGITIZED] MobileNetV1 top-1: 3x3 71.10 → 5x5 71.87 →
**7x7 72.03 (peak)** → 9x9 71.93 → 11x11 71.53 → 13x13 **70.83** (a 1.20-point collapse from peak).
Verbatim: accuracy "first goes up from 3x3 to 7x7, but then **drops down quickly** when the kernel size
is larger than 9x9."

**(b) DILATED SUBSTITUTES DEGRADE INSIDE THE MIX — the single most on-point published result.** They
replaced each KxK by a **3x3 kernel at dilation (K-1)/2** — structurally the same substitution this
proposal makes. [DIGITIZED] MobileNetV2 `MixConv+dilated`: 72.90 → **73.12 (peak at g=2)** → 73.05 →
72.95 → **72.90**, i.e. **it peaks at two branches then falls all the way back to the g=1 baseline**,
while plain MixConv reaches 74.31 — **a ~1.4-point gap.** Authors' verbatim conclusion and mechanism:

> "dilated convolution has reasonable performance for small kernels, but the **accuracy drops quickly
> for large kernels**. Our hypothesis is that when dilation rate is big for large kernels, a dilated
> convolution will **skip a lot of local information**, which would hurt the accuracy."

And in design choices: "**dilated convolutions usually have inferior accuracy than large kernel
sizes**." They **excluded dilated convs from the NAS search space** as a result.
**[FACT] Caveat:** "Tensorflow dilated convolution is not compatible with stride 2, we only use dilated
convolutions for a layer if its stride is 1" — handicapping the dilated arm at exactly the stride-2
layers where large kernels help most. **[REASONING]** This weakens but does not overturn the result:
the V2 dilated curve turns *downward*, which a partially-un-dilated subset cannot explain.

**[FACT] COCO detection (Table 1) — the only numeric table for MixConv:**

| Network | V1 params/FLOPS/mAP | V2 params/FLOPS/mAP |
|---|---|---|
| baseline3x3 | 5.12M / 1.31B / 21.7 | 4.35M / 0.79B / 21.5 |
| depthwise5x5 | 5.20M / 1.38B / 22.3 | 4.47M / 0.87B / 22.1 |
| mixconv35 | 5.16M / 1.35B / 22.2 | 4.41M / 0.83B / 22.1 |
| depthwise7x7 | 5.32M / 1.47B / 21.8 | 4.64M / 0.98B / **21.2** |
| mixconv357 | 5.22M / 1.39B / **22.4** | 4.49M / 0.88B / **22.3** |

MixConv357 beats depthwise7x7 by **+0.6 (V1)** and **+1.1 mAP (V2)** with fewer params/FLOPs.
**[REASONING] But note `mixconv35` ≈ `depthwise5x5` (22.2 vs 22.3 V1; tie on V2)** — the mixed-kernel
win only materializes once the single-kernel baseline has been pushed *past its own peak*. Against a
*well-chosen* single kernel size, multiscale bought ~nothing. That is a direct warning: if you compare
your 4-branch block only to k=3 and not to the best single width (B1/B2), you will manufacture a win.

**[FACT] Per-layer ablation (Fig. 5):** replacing 1 of 15 MobileNetV2 layers — "for most of layers, the
accuracy doesn't change much, but **for certain layers with stride 2**, a larger kernel can
significantly improve the accuracy." **[REASONING]** Multiscale value is *concentrated at specific
layers*, not uniform — argues for a per-layer span schedule (cheap) over a per-token router (expensive).

**[FACT] MixNet ImageNet:** MixNet-S 4.1M/256M/**75.8**, MixNet-M 5.0M/360M/**77.0**, MixNet-L
7.3M/565M/**78.9** top-1. **[FACT] Attribution caveat:** the NAS space also contained **swish, SE, and
grouped 1x1 convs**, and MixNet-L is MixNet-M scaled 1.3x — so MixNet headline numbers are **not** a
clean measurement of MixConv. The clean measurements are Fig. 4 (+0.25 to +0.5 top-1 at matched FLOPs)
and Table 1 (+0.6/+1.1 mAP).
**[FACT] No latency/wall-clock measurement anywhere in the paper** — efficiency is params and FLOPs
only. Yet Fig. 4's caption asserts MixConv is "smaller, **faster**, and achieves higher accuracy."
**That speed claim is unsupported by any timing data in the paper** — do not cite MixConv for speed.

### 2.5 Inception — arXiv 1409.4842 / 1512.00567

**[FACT]** GoogLeNet (<https://arxiv.org/abs/1409.4842>) combines parallel 1x1/3x3/5x5 branches by
**concatenation**: "their output filter banks concatenated into a single output vector." Rationale:
"visual information should be processed at various scales and then aggregated so that the next stage
can abstract features from different scales simultaneously." Two candid admissions worth quoting:
filter sizes were restricted to 1x1/3x3/5x5 "In order to avoid patch-alignment issues," and this
"decision was based more on **convenience rather than necessity**"; and "the ratio of 3x3 and 5x5
convolutions **should increase as we move to higher layers**." Cost: naively "even a modest number of
5x5 convolutions can be prohibitively expensive," causing "a **computational blow up within a few
stages**"; fixed with 1x1 reductions. Results: 6.67% top-5 (ILSVRC 2014 winner), ~1.5B multiply-adds,
12x fewer params than AlexNet.

**[FACT] How multi-kernel branches were superseded — by the same authors.** Inception-v2/v3
(<https://arxiv.org/abs/1512.00567>): 5x5 is "**2.78 times more computationally expensive**" than 3x3;
replacing it with two stacked 3x3 gives "**a relative gain of 28%**." Asymmetric $n\times n \to
1\times n + n\times 1$ is "**33% cheaper**" at equal RF but "**does not work well on early layers**."
Ablation: **factorized 7x7 was v3's largest single win (−1.2 top-1)**, vs RMSProp −0.3, label smoothing
−0.3, BN-auxiliary −0.4; final v3 21.2% top-1 / 5.6% top-5.
**[REASONING]** The trajectory is instructive: Inception's parallel multi-kernel branch was
progressively dismantled by its own authors in favor of factorized stacks of small kernels, and later
ResNeXt/RepVGG showed uniform 3x3 stacks match or beat it per unit wall-clock. Multi-branch
multi-kernel is a design pattern the vision field *tried and largely abandoned*, on wall-clock grounds.

### 2.6 Res2Net — arXiv 1904.01169, <https://arxiv.org/abs/1904.01169>

**[FACT]** Splits features into $s$ subsets; each gets its own 3x3, **chained hierarchically**:
$y_i = K_i(x_i + y_{i-1})$ for $2 < i \le s$, with $y_1 = x_1$ (first split's conv omitted for
parameter reuse), then concat + 1x1 fuse. Because branches are *chained not independent*, each branch's
effective RF grows as it crosses more filters, yielding "a different number and different combination
of receptive field sizes/scales."

**[FACT] Res2Net is the ONLY paper in this set that reports wall-clock latency for its multi-scale
block** (Table III, matched 4.2G FLOPs):

| Setting | FLOPs | **Runtime** | top-1 err |
|---|---|---|---|
| ResNet-50, 64w | 4.2G | **149ms** | 23.85 |
| 48w x 2s | 4.2G | **148ms** | 22.68 |
| 26w x 4s | 4.2G | **153ms** | 22.01 |
| 14w x 8s | 4.2G | **172ms** | 21.86 |

**[REASONING] Read the saturation:** s=1→2 buys **1.17** points; 2→4 buys 0.67; **4→8 buys only 0.15
while runtime goes 153→172 ms (+12%)**. Same knee shape as MixConv. Four branches is already at/past
the knee. **[FACT]** The paper concedes splits "must be computed in sequence due to the hierarchical
links," arguing the overhead "can often be ignored" because a GPU "generally has spare parallel
capacity within a clock period" at s=4.

### 2.7 The gridding pathology — dilation's known failure mode

**[FACT] Dilated Residual Networks** (Yu, Koltun & Funkhouser), arXiv 1705.09914,
<https://arxiv.org/abs/1705.09914>. Section 4 opens: "**The use of dilated convolutions can cause
gridding artifacts.**" Cause, verbatim: "**Gridding artifacts occur when a feature map has
higher-frequency content than the sampling rate of the dilated convolution.**" Severity: DRN-A-50's
outputs "are marred by gridding artifacts **even though the model was trained with dense pixel-level
supervision**" — i.e. **supervision does not train the artifact away.** Degridding (progressively
*decreasing* dilation, remove residual connections in those layers) bought, at identical 21.1M params:
DRN-B-26 **25.19 → DRN-C-26 24.86 top-1** (−0.33) and 7.91→7.55 top-5; Cityscapes mIoU
DRN-A-50 67.3 → DRN-C-26 **68.0** (the *smaller* degridded model beats the larger gridded one).
Per-class gains concentrate on large/thin structures: Train 36.2→54.7, Bus 59.5→74.3, Fence 42.8→52.6.

**[FACT] Hybrid Dilated Convolution / "Understanding Convolution for Semantic Segmentation"**,
arXiv 1702.08502, <https://arxiv.org/abs/1702.08502>. Quantified: a dilated kernel's "receptive field
**only covers an area with checkerboard patterns**"; worked example — with $k=3, r=2$, "**only 9 out of
25 pixels in the region are used for the computation**," losing "**at least 75%**" of information.
Also a consistency failure: neighboring pixels within an $r \times r$ region draw from **disjoint
grids**, which "may impair the consistency of local information." Design criterion for *stacked* layers:
$M_i = \max[M_{i+1} - 2r_i,\; M_{i+1} - 2(M_{i+1}-r_i),\; r_i]$ with $M_n = r_n$, require $M_2 \le K$.
For $K=3$: $r=[1,2,5]$ **valid**; $r=[1,2,9]$ **fails**. Heuristic: sawtooth (use 1,2,3 not 2,2,2), and
rates within a group **must not share a common factor** — "avoid 2,4,8." They name this as the explicit
difference from **ASPP** and Yu & Koltun's context module, "both of which use common-factor rates."
Results: HDC degridding bought **+1.4 mIoU** over plain dilation (76.4 vs 75.0 Cityscapes val);
final ResNet-DUC-HDC **80.1% single model, no CRF**.

**[REASONING] Careful about transferring HDC's criterion.** HDC's $M_2 \le K$ test governs
*sequentially stacked* dilated layers, where gaps compound multiplicatively. This proposal uses
**parallel branches summed in one layer**, where the correct analysis is simply the **union of lags**
— which I computed directly in Section 5.2: 7 of 15 lags reached, 8 structurally zero. So do not cite
HDC's "avoid 2,4,8" rule as if it directly applies; cite instead (a) its *coverage arithmetic* ("9 of
25 pixels," ">=75% lost"), which is the same phenomenon, and (b) MixConv's direct empirical result
(2.4b), which *is* the parallel-mixed-kernel case. **Three independent confirmations of the same
mechanism** (DRN 2017, HDC 2017, MixConv 2019), from three different methodologies.

### 2.8 Fragmentation: multi-branch is slower than a fused kernel at equal FLOPs

ShuffleNetV2's G3 numbers are in Section 7.3 (they are throughput evidence). Two additional items:

**[FACT] RepVGG** — "RepVGG: Making VGG-style ConvNets Great Again," arXiv 2101.03697,
<https://arxiv.org/abs/2101.03697> (CVPR 2021). Measures the training-time branch cost and then fuses
it away. Table 6 (RepVGG-B0, 120 epochs):

| Identity branch | 1x1 branch | Top-1 | Speed **before** re-param |
|---|---|---|---|
| — | — | 72.39 | **1810** |
| yes | — | 74.79 | 1569 |
| — | yes | 73.15 | 1230 |
| yes | yes | **75.14** | **1061** |

**Two branches cost 41% of throughput** (1810 → 1061 ex/s) for +2.75% top-1. After fusion all variants
run at **1817 ex/s** — accuracy kept, branch cost zero. Their stated arguments: fragmentation "is
unfriendly to devices with strong parallel computing powers like GPU and introduces extra overheads
such as kernel launching and synchronization"; "**the multi-branch topology is memory-inefficient
because the results of every branch need to be kept until the addition or concatenation**"; and
3x3 has ~4x the computational density of other kernel sizes (**38.10 vs ~9–10 TFLOPS**), "suggesting
the total theoretical FLOPs is **not a comparable proxy for the actual speed**." Key caveat **[FACT]**:
"the inference-time equivalence **does not imply the training-time equivalence**."
**[FACT]** Their non-fusible row: adding a ReLU inside a branch gives 75.69 but forfeits fusion.

**[REASONING] RepVGG is the most constructive result in this dossier and suggests the best version of
this experiment.** Because dilated/multi-width depthwise branches are **linear**, a fixed-weight
4-branch block **fuses losslessly into one depthwise kernel of width $2\max(d)+1 = 15$** — which is
exactly the Section 5.1 equivalence, independently confirmed as an engineering technique. So:
*train* multi-branch (if it helps optimization — RepVGG's whole point is that the training-time
topology matters even when the inference-time function is identical) and *ship* one fused 15-tap kernel.
**But this only works if the mixing weights are input-independent.** A token-dependent router destroys
fusibility across tokens and locks in the fragmentation penalty permanently. **[REASONING] This gives a
genuinely interesting and cheap experiment the proposal does not currently contain: does the 4-branch
*training-time* parameterization of a 15-tap kernel optimize better than a directly-trained 15-tap
kernel, even though they are the same function class?** That is a reparameterization/optimization
question, is free at inference, and is the one place the branch structure could add real value.

**[FACT] Confirmed absent from this literature:** no paper in this set uses a **learned router or gate
over multi-scale branches** — MixConv is static `tf.split`/`tf.concat`, Res2Net is a fixed hierarchical
chain, Inception is fixed concat. **[REASONING] So a token-dependent router over multi-scale branches
is genuinely novel relative to the multiscale-conv literature — and correspondingly unsupported by any
prior numbers.** The nearest thing is SKNet (Section 3), which does exactly this in vision.
Also absent: any wall-clock measurement for multi-branch **causal/1-D** blocks (all fragmentation
timing is 2-D vision), and any dilation ablation in WaveNet, ByteNet, or TCN.


---

## 3. Learned/adaptive receptive field and input-conditioned kernel mixtures

### 3.1 LightConv / DynamicConv — Wu et al., arXiv 1901.10430

<https://arxiv.org/abs/1901.10430> (ICLR 2019 oral). Wu, Fan, Baevski, Dauphin, Auli.
**The most relevant single paper for the router design**, and it contains three findings that directly
constrain this proposal.

**[FACT] LightConv** = depthwise conv with **softmax-normalized weights over the kernel taps**, plus
weight sharing across channel groups:

$$\mathrm{LightConv}(X, W_{\lceil cH/d\rceil,:}, i, c) = \mathrm{DepthwiseConv}(X, \mathrm{softmax}(W_{\lceil cH/d\rceil,:}), i, c)$$

with $\mathrm{softmax}(W)_{h,j} = \exp W_{h,j} / \sum_{j'=1}^{k}\exp W_{h,j'}$ — **normalization runs
across the $k$ taps (the temporal dimension)**. Weight sharing ties parameters across each group of
$d/H$ channels: for $d{=}1024, k{=}7$, a regular conv needs 7,340,032 weights, depthwise 7,168, and with
$H{=}16$ only **112**. DropConnect is applied to the normalized weights.

**[FACT] DynamicConv** = same shape, kernel predicted per timestep by a **linear projection of the
current token only**: $f(X_i) = \sum_{c} W^Q_{h,j,c} X_{i,c}$, with
$\mathrm{DynamicConv}(X,i,c) = \mathrm{LightConv}(X, f(X_i)_{h,:}, i, c)$. Because the kernel depends
only on the current position, cost "scales linearly in the sequence length." The parameter reduction
from weight sharing is what makes it feasible — "crucial to make dynamic convolutions possible on
current hardware."

**[FACT] CRITICAL FINDING 1 — softmax over taps was REQUIRED for convergence.** For DynamicConv,
"**training diverged in our experiments when removing it**." Appendix A sweep on newstest2013:

| Normalization of the predicted kernel | BLEU |
|---|---|
| none | **diverges** |
| **softmax** | **26.9 ±0.2** |
| $W/(\|W\|_2+\epsilon)$ | 26.8 ±0.2 |
| $\mathrm{abs}(W)/(\|W\|_2+\epsilon)$ | 26.7 ±0.2 |
| sigmoid $\sigma(W)$ | 26.6 ±0.3 |
| tanh | 25.6 ±0.2 |
| $\ell_1$-norm / squaring / plain abs | **diverge** |

Their conclusion: "having all non-negative weights is **not** critical" — softmax simply works best.
**[REASONING] This is a genuine argument in the proposal's favour, and it is the best one available:**
an input-dependent filter needs *some* normalization or it diverges, and a softmax-normalized convex
mixture is a principled way to get one. Note, though, that this normalizes **the filter's taps**, and
Sieberling et al. (arXiv 2606.03825) later trained unnormalized **affine** dynamic filters
successfully at 150M–2B scale — so the divergence problem appears solvable by other means (their
low-rank/head-wise parameterization plus modern normalization), and normalization is no longer a
forced choice.

**[REASONING] The distinction the proposal must not blur — softmax over TAPS vs softmax over BRANCHES.**
These are different mechanisms with different effects:
- **Softmax over taps** (LightConv): constrains one filter to be a *convex combination of its input
  positions* — i.e. the conv output is a weighted **average** of the window. This is a smoothing /
  scale-control constraint. It guarantees bounded output magnitude and non-negative taps, and it is
  what prevented divergence.
- **Softmax over branches** (this proposal): constrains the *mixture of four already-computed
  convolutions* to be convex. The individual branch filters remain unconstrained real-valued, so
  **the effective 15-tap kernel is NOT convex-normalized and can have arbitrary sign and magnitude**
  (see the Section 5.3 derivation: $\kappa^{(t)}_\ell = \sum_b \alpha^{(t)}_b w^{(b)}_k$ with $w$ free).
  **So the proposal does NOT inherit LightConv's stability property.** It buys a bounded *mixing*
  simplex, not a bounded *filter*. This is worth stating explicitly, because "we use a softmax so it's
  like LightConv" would be a false inference.

**[FACT] CRITICAL FINDING 2 — the kernel-width schedule, and a correction.** Translation models use
kernel sizes "**3, 7, 15, 31x4** for each block respectively" across 7 blocks (decoder: "only three top
layers with kernel size 31"). LM on Billion Word: N=17 blocks with **15x2, 31x4, 63x11**. The
often-quoted `3,7,15,31,31,31` applies specifically to the **6-layer reduced decoder** in the Appendix B
speed study, not the main model.
**[REASONING] This is important prior art the proposal should cite:** the established way to get
multiple scales is a **per-layer increasing schedule** (cheap, no router, no branch multiplicity, fully
fusable), *not* parallel branches within a layer. It also converges with MixConv's per-layer finding
(value concentrated at particular layers) and with adaptive-span results below. **A per-layer span
schedule is a strictly cheaper alternative hypothesis that the experiment should include as a rung**
(see 8.6).

**[FACT] CRITICAL FINDING 3 — how much the increasing schedule actually bought.** Table 3 (WMT En-De
newstest2013, params, BLEU, sent/sec on P100):

| Configuration | Param | BLEU | Sent/sec |
|---|---|---|---|
| Self-attention baseline ($k=\infty$, H=16) | 210M | 26.9 ±0.1 | 52.1 |
| **Self-attention restricted to k=3,7,15,31x3** | 210M | **26.9 ±0.3** | 54.9 |
| CNN (k=3) | 208M | 25.9 ±0.2 | 68.1 |
| CNN Depthwise (k=3, H=1024) | 195M | 26.1 ±0.2 | 67.1 |
| **+ Increasing kernel (k=3,7,15,31x4)** | 195M | **26.4 ±0.2** | 63.3 |
| + DropConnect | 195M | 26.5 ±0.2 | 63.3 |
| + Weight sharing (H=16) | 195M | 26.5 ±0.1 | 63.7 |
| + Softmax weights [**LightConv**] | 195M | 26.6 ±0.2 | 63.6 |
| + Dynamic weights [**DynamicConv**] | 200M | **26.9 ±0.2** | 62.6 |
| DynamicConv w/o softmax | 200M | **diverges** | — |

**[REASONING] Read the deltas carefully — they are small and they cost throughput.** The increasing
kernel schedule bought **+0.3 BLEU** (26.1 → 26.4) and cost ~6% throughput (67.1 → 63.3 sent/sec).
Input-dependence (DynamicConv over LightConv) bought **+0.3 BLEU** (26.6 → 26.9) for +5M params and
another ~2% throughput. And critically: **restricting self-attention to the same bounded windows left
BLEU unchanged at 26.9** — "restricting context has no impact on validation accuracy," which is the
paper's motivation for bounded windows in the first place.

**[FACT] Headline results (newstest2014):** LightConv 202M → 28.9 (En-De) / 43.1 (En-Fr); DynamicConv
213M → **29.7** / 43.2 (a new SOTA on En-De by 0.4 BLEU over Ott et al.'s 29.3). IWSLT De-En:
self-attn 34.4, LightConv 34.8, DynamicConv 35.2. Speed claim ~20% faster than unrestricted
self-attention (62.6 vs 52.1 sent/sec) — **but only ~14% faster than the context-restricted baseline at
54.9.**
**[FACT] Implementation caveat, highly relevant to Section 7:** "existing convolution primitives
underperformed, so they expand weights into a band matrix of size $BH \times n \times n$ and use
batched matmul," expecting "a dedicated CUDA kernel to be much more efficient." **[REASONING] So even
this paper's speed numbers came from a workaround, reinforcing that dynamic/multi-branch convs are
kernel-limited, not FLOP-limited.**

### 3.2 CondConv — sigmoid router, and why its cheap trick does NOT transfer

**[FACT] Yang, Bender, Le & Ngiam, "CondConv: Conditionally Parameterized Convolutions for Efficient
Inference"** — arXiv 1904.04971, <https://arxiv.org/abs/1904.04971> (NeurIPS 2019).

**Router is SIGMOID, not softmax** (verified): `r(x) = Sigmoid(GlobalAveragePool(x) · R)`. Softmax was
ablated and **lost: 60.5% vs 62.0%** top-1 (CC-MobileNetV1 0.25x). Their reading: sigmoid's advantage
"suggests that multiple experts are often useful for a single example."

**[FACT] The efficiency identity:** $\sigma((\alpha_1 W_1 + \ldots + \alpha_n W_n) * x) =
\sigma(\alpha_1(W_1 * x) + \ldots + \alpha_n(W_n * x))$ — same capacity as a linear MoE but "requires
computing only one expensive convolution"; "each additional parameter requires only 1 additional
multiply-add."

**★ [FACT] CONFIRMED — the trick holds ONLY because $\alpha$ is per-SAMPLE.** The paper defines
$\alpha_i$ as "an example-dependent scalar weight," and global average pooling collapses all spatial
structure to one weight vector per example. **[REASONING] A per-token router breaks the factorization
outright:** the kernel can no longer be formed once and reused across positions, so you pay $n$
convolutions (or materialize a per-position kernel). **This is exactly why CondConv's cheapness does not
transfer to this proposal**, and why DynamicConv had to obtain cheapness by a totally different route
(depthwise + H=16 weight sharing, shrinking the kernel to 112 numbers). It is also why Sieberling et al.
needed custom Triton kernels rather than a library call.

**[FACT] ImageNet, 8 experts/layer:** MobileNetV1 71.9 → **73.7**; MobileNetV2 71.6 → **74.6**;
MnasNet-A1 74.9 → **76.2**; ResNet-50 77.7 → **78.6**; EfficientNet-B0 77.2 → **78.3**. Overhead <10%
MADDs. **[FACT] Parameter cost is severe and unreported in the paper** — cross-referenced from DY-CNN's
table, CondConv-MobileNetV2 x1.0 is **27.5M params vs 3.5M static, a 7.9x blowup.** No latency numbers
anywhere; cost is reported exclusively in multiply-adds.
**[FACT] Depth of routing:** applying CondConv from layer 1 → 62.5%, layer 5 → 62.0%, layer 7 → 62.0%,
layer 13 → 59.5%, FC-only → 54.2%. Routing-weight distributions are near-identical across classes in
early layers and become class-specific with depth. **Compute verdict: ADDED only, never saved.**
**[UNKNOWN]** Per-$n$ expert-count accuracies exist only in Figure 2, not extractable.

### 3.3 ★ Dynamic Convolution (Chen et al.) — a plain 4-way softmax router LOSES TO STATIC

**[FACT] Chen, Dai, Liu, Chen, Yuan & Liu, "Dynamic Convolution: Attention over Convolution Kernels"** —
arXiv 1912.03458, <https://arxiv.org/abs/1912.03458> (CVPR 2020). Router: GAP → FC (÷4) → ReLU → FC →
**softmax with temperature**, **K=4 kernels** — the same arity as this proposal.

**★★ THE MOST DIRECTLY TRANSFERABLE ROUTING-INSTABILITY RESULT IN THE DOSSIER ★★**
$\pi_k = \exp(z_k/\tau)/\sum_j \exp(z_j/\tau)$. Schedule: **train at $\tau=30$, anneal linearly 30 → 1
over the first 10 epochs.** MobileNetV2 x0.5, static baseline **65.4** top-1:

| $\tau$ | Top-1 | vs static |
|---|---|---|
| **1 (plain softmax)** | **64.8** | **−0.6 (WORSE THAN STATIC)** |
| 5 | 65.7 | +0.3 |
| 10 | 67.5 | +2.1 |
| 20 | 69.4 | +4.0 |
| 30 | 69.4 | +4.0 |
| 40 | 69.2 | +3.8 |
| **annealed 30→1** | **69.9** | **+4.5** |

**[FACT] Why (verbatim):** plain softmax "does NOT work well on this due to its near one-hot output,"
which "only allows a small subset of kernels across layers to be optimized." Conversely "near-uniform
attention can facilitate the learning of all kernels" because it "enables more convolution kernels to be
optimized simultaneously." Corroborating diagnostic: reducing the *number* of dynamic layers **sped up
convergence** — the pathology compounds with the depth of dynamic layers.

**[REASONING] This is a direct, quantified warning for the proposed router.** A naive 4-way softmax over
branches scored **below the static baseline**, and the *entire* +4.5 gain came from flattening the router
early. **If the experiment implements a plain softmax router with no temperature schedule and it
underperforms the k=3 baseline, that result will be uninterpretable** — you will not know whether the
architecture failed or the router simply starved three branches. **A temperature schedule (or equivalent
entropy floor) is therefore mandatory, not optional**, and belongs in the training recipe from day one
alongside the fp32 router logits and z-loss from Section 4.3.

**[FACT] Their sum-to-one / kernel-space-compression argument, and the critique of CondConv:**
$\sum\pi_k = 1$ keeps the aggregated kernel "within the convex hull" of the K kernels; with 3 kernels,
$0 \le \pi_k \le 1$ alone confines the result to two pyramids, and sum-to-one "further compresses the
kernel space to a triangle." They argue CondConv's sigmoid "retains the much larger two-pyramid space,
making attention learning harder." Head-to-head: CondConv x1.0 (8 kernels) 27.5M / 329M / 74.6 vs
**DY-CNN x1.0 (4 kernels) 11.1M / 312.9M / 75.2**.

**[FACT] ImageNet gains and costs:** MobileNetV2 x1.0 3.5M/300M/72.0 → **11.1M/312.9M/75.2 (+3.2)**;
x0.5 2.0M/97M/65.4 → **4.0M/101.4M/69.9 (+4.5)**; ResNet-18 11.1M/1.81G/70.4 → **42.7M/1.85G/72.7
(+2.3)**. **Params inflate 3–4x for ~4% extra MAdds; CPU latency overhead ~10%, higher than the MAdds
increase**, attributed to unoptimized pooling and small inner products.
**[FACT] K ablation:** dynamic beats static at every width even at K=2, but "**the accuracy stops
increasing once K is larger than 4**" — harder joint optimization and overfitting. **[UNKNOWN]**
accuracy-vs-K values are Figure 6 only.

**★ [FACT] Here input-dependence IS load-bearing — the important counterweight.** Replacing learned
attention with **kernel averaging collapses top-1 from 69.4% → 36.0%**; argmax kernel → 0.1%; shuffling
attention per image → 14.8%; across images → 27.3%. **[REASONING] So unlike LightConv (where dynamic
bought only +0.3 BLEU) and Adaptive Span (where dynamic tied static), the kernel-mixture setting is one
where input-dependence genuinely matters.** This is the strongest single piece of evidence *for* the
proposal's mechanism, and it should be cited as such. Two caveats before leaning on it: (a) these are
inference-time ablations of a model *trained* with a router, which per Section 6.2 conflates "this
component is load-bearing" with "you moved the model off-distribution" — the fair test is retraining
(baseline L4); (b) it is vision at K=4 over *full* kernels, not causal 1-D over spans.
**[FACT] Attention is flat at low levels, sparse at high levels:** enabling attention only at
14²/7² resolutions yields 67.0% (near the full 69.4%), while only at 112²/56²/28² yields just **42.5%**.

**[FACT] DCD, "Revisiting Dynamic Convolution via Matrix Decomposition"** — arXiv 2103.08756,
<https://arxiv.org/abs/2103.08756>. Diagnoses CondConv/DY-CNN as a *parameterization* problem: the
residual assembles a rank-≤C matrix from KC unshared rank-1 terms → "model redundancy," and "a small
attention score $\pi$ may suppress the learning of corresponding columns $u_i, v_i$" — **the same
starvation pathology DY-CNN's $\tau=30$ patches.** Beats both at far lower cost (MobileNetV2 x0.5
65.4 → **70.2 at 3.1M** vs DY-Conv 4.0M/69.9 and CondConv 15.5M/68.4). Also: "**dynamic fusion is more
effective across channels than across kernel elements**" (73.1 vs 71.3, ResNet-18 3x3). Inference cost
**+12–14% wall clock**.

### 3.4 ★ SKNet — the exact image analogue: 2 branches is the sweet spot, 3 buys 0.03%

**[FACT] Li, Wang, Hu & Yang, "Selective Kernel Networks"** — arXiv 1903.06586,
<https://arxiv.org/abs/1903.06586> (CVPR 2019). **Split–Fuse–Select:** two branches (kernel 3 and
kernel 5) → elementwise **sum** → GAP → one FC with reduction ($d = \max(C/r, L)$, L=32) → **softmax
across branches, independently per channel**: $a_c = e^{A_c z}/(e^{A_c z} + e^{B_c z})$, $b_c = 1 - a_c$,
$V_c = a_c \tilde U_c + b_c \hat U_c$. Note the efficiency choice: "the conventional convolution with a
5x5 kernel is **replaced with the dilated convolution with a 3x3 kernel and dilation size 2**."

**[FACT] ImageNet top-1 error (lower better):** ResNeXt-50 22.23 → SENet-50 21.12 → **SKNet-50 20.79**
(27.5M, 4.47 GFLOPs); ResNeXt-101 21.11 → SENet-101 20.58 → **SKNet-101 20.19**. Cost: "**10% increase
in the number of parameters and 5% increase in computational cost**."
**[FACT] The capacity control (Table 3) — the gain is not just capacity:** ResNeXt-50 22.23 →
SKNet-50 20.79 = **+1.44**, versus spending the same budget on wider (+0.10), deeper (+0.19), or higher
cardinality (+0.23). **~6x what naive capacity buys.**

**★★ [FACT] BRANCH-COUNT ABLATION — the answer to "do >2 branches help?" is NO ★★** (Table 7, 224²):

| Config | Top-1 err | Params | GFLOPs |
|---|---|---|---|
| K3 only | 22.23 | 25.0M | 4.24 |
| K5 only | **25.14** | 25.0M | 4.24 |
| K7 only | **25.51** | 25.0M | 4.24 |
| K3+K5, plain sum (M=2) | 21.76 | 26.5M | 4.46 |
| **K3+K5 + SK router (M=2)** | **20.79** | 27.5M | 4.47 |
| K3+K5+K7, plain sum (M=3) | 21.47 | 28.0M | 4.69 |
| **K3+K5+K7 + SK router (M=3)** | **20.76** | 29.3M | 4.70 |

**M=2 → M=3 buys 0.03% for +1.8M params and +0.23 GFLOPs**; the paper concludes "**M = 2 is
preferred**." **[REASONING] Three readings, all adverse to a 4-branch design.** (a) The knee is at
**two** branches — the third is free of value, and the proposal uses **four**. This is now the *third*
independent multiscale-branch sweep saturating at 2 (MixConv g=2, Res2Net s=2, SKNet M=2). (b) A single
*large* kernel is far worse than 3x3 alone (K5 25.14, K7 25.51 vs K3 22.23) — **big kernels only help
inside a mixture**, which is a point in the proposal's favour and should be cited. (c) The router *does*
beat plain summation: 21.76 → 20.79 at M=2 (**+0.97**) and 21.47 → 20.76 at M=3 (+0.71) — i.e. a learned
input-dependent mix beat a fixed uniform sum, which is exactly hypothesis L5-vs-L3.

**★ [FACT] Do selection weights actually vary with input? YES — verified with a clean protocol.** They
enlarged the central object **1.0x → 2.0x** and measured mean(5x5 weight) − mean(3x3 weight) per SK unit:
"when the target object enlarges, the attention weight for the large kernel (5x5) increases"; "**The
larger the target object is, the more attention will be assigned to larger kernels**," verified
"consistently and simultaneously" across all 1,000 classes (units SK_2_3, SK_3_4).
**★ [FACT] But the effect DISAPPEARS in late layers:** "at much higher layers (e.g., SK_5_3), all scale
information is getting lost and such a pattern disappears." **[REASONING] This is the vision analogue of
the "router becomes a constant" failure mode**, and it tells you *where* to expect a live router and
where to expect a dead one — a per-layer prediction the knockout protocol (Section 6.4, Tier 0) can test
directly. Note the three papers disagree on *which* depth is the useful zone (SKNet early/middle;
CondConv late; DY-CNN high-resolution-only), which is unresolved.
**[FACT] Dilation vs plain kernel:** second branch as 3x3/D2 → 20.79 (RF 5x5) ties plain 5x5/G64 → 20.80
while being cheaper; but 3x3/D3 → 20.97 (RF 7x7) is worse. **[REASONING] Consistent with MixConv:
mild dilation is fine, aggressive dilation degrades.** The proposal's dilation 7 is aggressive.

### 3.5 Mixture-of-Depths — the one routing method that genuinely saved wall clock

**[FACT] Raposo, Ritter, Richards, Lillicrap, Humphreys & Santoro** — arXiv 2404.02258,
<https://arxiv.org/abs/2404.02258>. Router is a **raw linear scalar** $r_i^l = w_\theta^\top x_i^l$ —
*not* a softmax over paths. Two paths: full block or residual. Best config **12.5% capacity, routing
every other block** (at seq len 2048: top-k = 256 processed, **1792 (87.5%) bypassed**). Because k is
fixed a priori, **tensor shapes are static** — this is what makes it hardware-friendly. Uses
expert-choice, which "obviates the need for an auxiliary balancing loss."

**★ [FACT] The non-causality problem, exactly as suspected:** whether a token is in the top-k depends on
router weights of **later** tokens, unavailable at autoregressive sampling — "the top-k operation is
non-causal." Two fixes: (1) an **auxiliary BCE loss** on router outputs vs top-k membership, which
"centers the sigmoid of the router's outputs around 0.5" but **perturbs the LM objective by 0.2–0.3%**;
(2) an **auxiliary MLP predictor** with a stop-gradient, which "does not affect the language modeling
objective." **Their autoregressive evaluations used the predictor.** Switching from training-time top-k
to the causal predictor caused "minimal performance degradation." **[FACT] Note an internal
contradiction in the paper:** the text claims the predictor "quickly achieves 99%" while Figure 6's
caption says "upwards of 97% accurate" — flagging as a discrepancy.

**[FACT] Compute — genuinely saved:** "upwards of 60% faster to step" in training at 220M; a 6e18-FLOP
variant matches the isoFLOP-optimal baseline while stepping **66% faster**; "upwards of 50% faster to
step during post-training sampling"; baseline loss parity at "upwards of 50%" fewer FLOPs per forward.
**Adding depth beat adding width.** Quality gain is modest: up to **1.5%** on final log-prob at equal
FLOPs *and* equal wall clock. **[FACT] No downstream evals at all.**

**[REASONING] Why MoD works and this proposal cannot copy it — the granularity lesson, made concrete.**
MoD skips an **entire transformer block** (attention + MLP: millions of FLOPs, compute-bound) for 87.5%
of tokens, with **static tensor shapes**. That is the regime where conditional compute pays. The proposal
would skip a **3-tap depthwise conv** (1.5–7.5 FLOP/byte, memory-bound) for at most 75% of tokens. Same
mechanism, ~5 orders of magnitude difference in the size of the skipped unit. **MoD is the strongest
available evidence that conditional compute needs coarse granularity to pay off** — it does not support
routing at the granularity this proposal proposes. And note the causality tax: even MoD needed an extra
predictor network to make top-k usable at decode.

### 3.6 Squeeze-and-Excitation — and the cleanest evidence that a gate can degenerate

**[FACT] Hu, Shen & Sun** — arXiv 1709.01507, <https://arxiv.org/abs/1709.01507> (CVPR 2018 / TPAMI).
Squeeze $z_c = \frac{1}{HW}\sum\sum u_c(i,j)$; excite $s = \sigma(W_2 \delta(W_1 z))$, default **r=16**;
rescale. **Sigmoid, chosen explicitly** for a "non-mutually-exclusive relationship" allowing several
channels to be emphasized "rather than enforcing a one-hot activation."
**[FACT] Cost:** ResNet-50 3.86 → 3.87 GFLOPs ("a 0.26% relative increase"), params +~2.5M ("~10%"),
**wall clock 190ms → 209ms (+10%)**. Gains: ResNet-50 24.80 → **23.29** top-1 err; VGG-16 27.02 →
**25.22**; SE-ResNet-50's 6.62% top-5 nearly matches ResNet-101 (6.52%) at half the FLOPs.

**★ [FACT] Gating capacity barely matters — the reduction-ratio sweep is nearly flat:** r=2 22.29
(45.7M), r=4 22.25 (35.7M), r=8 22.26 (30.7M), **r=16 22.28 (28.1M)**, r=32 22.72 (26.9M).
"Increased complexity does not improve performance monotonically" — **flat accuracy across a 19M
parameter range.** **[FACT] Excitation nonlinearity matters much more than capacity:** ReLU **23.47
(worse than the 23.30 baseline!)**, Tanh 23.00, **Sigmoid 22.28** — "careful construction of the
excitation operator is important."

**★★ [FACT] THE DEGENERATION RESULT — the single best precedent for "the router becomes a constant."**
Late-stage SE blocks saturate: SE_5_2 shows "an interesting tendency towards a saturated state in which
most of the activations are close to one," and the paper notes that when all activations equal one, "**an
SE block reduces to the identity operator**." SE_5_3 differs across classes only by "a modest change in
scale." Consequently, **removing the last-stage SE blocks costs only "<0.1% top-5 error" while cutting
the parameter increase from ~10% to ~4%.**
**[REASONING] This is the outcome I consider most likely for the proposed router (Section 4.6), now with
a published precedent and a measurement protocol attached.** An input-dependent gate, trained
end-to-end, drifted to approximately constant in the layers where the information it gated was no longer
scale-relevant — and was removable for free. **Directly actionable: log per-layer router-weight variance,
and for any layer where it approaches zero, test removing the router entirely (collapsing to a fixed
mix, i.e. baseline L4, which by Section 5.1 collapses further to a single 15-tap kernel).**

### 3.7 Does a learned STATIC mix match an input-dependent router? Mostly yes — with two exceptions

**[REASONING] This subsection is the most important addition to the baseline design, because it says the
learned-constant-mix arm (L4) is not a formality — it is a live competitor with substantial published
support.** Evidence, strongest first.

**[FACT] Branchformer — the smoking gun** (arXiv 2207.02971, <https://arxiv.org/abs/2207.02971>). Two
merge methods for a 2-branch (attention / cgMLP) block. Aishell CER dev/test: **concatenation+linear
(static) 4.19/4.43 @ 45.43M** vs **weighted average (input-dependent) 4.23/4.61 @ 43.88M** — **the static
merge WINS by 0.18 CER.** And critically: the learned input-dependent branch weights turn out to be
nearly constant — "**The standard deviation is usually smaller than 0.01.**" **[REASONING] This is the
closest published analogue to the proposed router (softmax-ish mix over 2 parallel sequence-mixing
branches in a real LM-adjacent model), and both its headline comparison and its weight statistics are
negative.** Also: early layers alternate branches, mid layers are attention-dominated, final layers stack
the local branch — again a *depth* pattern, not a token pattern.

**[FACT] Synthesizer** (arXiv 2005.00743, ICML 2021). **Random Synthesizer** is a learned but
*input-independent* attention matrix $Y = \mathrm{softmax}(R)G(X)$, R shared across all samples. EnDe /
EnFr BLEU: Transformer 27.67/41.57; **Syn(Random) learned-static 27.27/41.12**; Syn(Dense)
input-dependent 27.43/41.39; **Syn(Fixed Random) frozen 23.89/38.31**. Learned static is within
**0.40 BLEU** of full self-attention and within **0.16** of the input-dependent version — but *freezing*
costs **3.4 BLEU**. Abstract: "learning attention weights from token-token (query-key) interactions is
useful but not that important after all." **Where it fails:** GLUE/SuperGLUE 75.1/61.1 vs T5-Base
83.5/70.3, and summarization — i.e. wherever cross-sequence alignment is needed.

**[FACT] Hard-Coded Gaussian Attention** (arXiv 2005.00742, ACL 2020). Fully hand-set, nothing learned,
Q/K projections deleted. `hc-sa` vs baseline: IWSLT16 En-De **30.3 vs 30.0**, De-En **34.8 vs 34.4**;
WMT14 En-De 26.3 vs 26.8. **But `hc-all` (cross-attention also hard-coded) collapses to 21.1/25.7.**
Also relevant: truncating their Gaussian to a **3-tap kernel [0.242, 0.399, 0.242]** cost only 0.2 BLEU.
**[REASONING] Note that this hand-set 3-tap kernel is a decaying, normalized, convex short filter —
precisely the SGConv decay prior and the LightConv tap-softmax, arrived at independently.**

**[FACT] Demystifying Local Vision Transformer** (arXiv 2106.04263): static depthwise conv **DWNet-T
24M/3.8G/81.3 exactly ties Swin-T 28M/4.5G/81.3**; at base scale their own *dynamic* variant gives
**83.2 vs static 83.2 — literally zero gain** while tripling params 74M → 162M.

**[FACT] MoE routing:** Hash Layers (arXiv 2106.04426, NeurIPS 2021) — a **fixed hash of the token id**,
no trained router, beats Switch: 751M pushshift Reddit valid/test **Hash 23.16/23.23 vs Switch
23.65/23.73** (dense 24.90/24.96); balanced hash ≈ **fixed random hash** (23.22/23.27). Verbatim: "our
results perhaps suggest that none of the current approaches are routing particularly well." THOR
(arXiv 2110.04260, ICLR 2022): "the commonly-used routing methods based on gating mechanisms do not work
better than randomly routing inputs to experts"; **removing the trained router at inference drops BLEU
only 20.6 → 20.4.** **[REASONING] Caveat I want to flag rather than bury: THOR's own 2-BLEU win comes
from its consistency regularizer, not from random routing.**

**★ [FACT] The counter-evidence, and it is specific.** (a) **Dikkala et al.** (EMNLP 2023,
<https://aclanthology.org/2023.emnlp-main.583/>): T5-Base on mC4, **frozen vs trainable router 70.27% vs
70.34%** next-token accuracy — no effect; they had to shrink $d_{model}$ 768 → 32 to manufacture a gap.
**[REASONING] Read plainly, their normal-width result supports the static-mix thesis; learned routing
mattered only in capacity-starved models.** (b) **In the conv-kernel setting specifically,
input-dependence IS load-bearing** — DY-CNN's kernel-averaging collapse (69.4 → 36.0) and DCD's +3–5
points. **[REASONING] So the honest synthesis is: for mixing *parallel branches* (Branchformer, SKNet
M=3, DWNet) a learned constant is usually competitive; for mixing *whole convolution kernels* in vision
(DY-CNN, DCD) input-dependence genuinely helps. This proposal is structurally the former (mix of branch
outputs) while rhetorically claiming the latter.** By Section 5.1 it is mathematically the latter *only*
because the mixture is algebraically foldable into a per-token kernel — which is precisely the reframing
that makes the unconstrained generator (L6) the better implementation.
**Consistent finding across all of the above:** *frozen-at-init* works; *resampled-per-step* does not.
The claim is "a fixed mapping the model can co-adapt to suffices," not "routing is irrelevant."

**[UNKNOWN]** No paper was found replacing SE-style sigmoid gating with a **learned per-channel
constant**; multiple searches returned nothing. Closest artifact is LayerScale (arXiv 2103.17239), which
has no input-dependent baseline. **[REASONING] That specific ablation appears genuinely unclaimed — and
it is close to this experiment's L4-vs-L5 comparison, which is a small point in favour of running it.**

### 3.8 Adaptive Attention Span — VERIFIED, and it is strong convergent evidence

**[FACT] Sukhbaatar, Grave, Bojanowski & Joulin, "Adaptive Attention Span in Transformers"** —
arXiv 1905.07799, <https://arxiv.org/abs/1905.07799> (ACL 2019, <https://aclanthology.org/P19-1032/>).
Each head learns its own span via a soft mask parameterized by a learnable $z \in [0,S]$:

$$m_z(x) = \min\!\big[\max[\tfrac{1}{R}(R + z - x),\, 0],\, 1\big]$$

($R$ = ramp softness, $R=32$ used), multiplying the exponentiated similarities inside the softmax, plus
an $\ell_1$ penalty pushing spans down:
$L = -\log P(w_1..w_T) + \tfrac{\lambda}{M}\sum_i z_i$ with $\lambda = 2\times10^{-6}$.
A **dynamic** (input-dependent) variant sets $z_t = S\sigma(v^\top x_t + b)$.

**[FACT] THE KEY RESULT — learned spans collapse to very short, and this is now verified:**
- **With the limit set to $S=8192$, the 12-layer model's average learned span is just 314; the 24-layer
  model's is 245** — roughly **4% and 3% of the permitted maximum**. The authors flag it explicitly:
  "even with a limit on span sets to 8192, the average span is only 314." (At $S=1024$: average 123.)
- **"the lowest 5 layers have the smallest possible attention span, which is R=32"** — i.e. 5 of 12
  layers (40 of 96 heads) pin at the *floor*. Only "few attention heads in the higher layers have very
  long spans, exceeding several thousand." Figure caption: "**Few attention heads require long
  attention spans.**" The trend is not monotonic in layer height.
- **[FACT]** They had to *lower* $\lambda$ to $0.5\times10^{-6}$ at $S=8192$ "because z was not growing
  longer than 4000" — the model resisted long spans even when permitted.
- **[FACT] Dynamic vs static spans:** on text8 both reached **1.08 dev bpc**, with average spans 123
  (adaptive/static) vs 149 (dynamic) at $S=1024$. **Input-dependence bought nothing here.**
- **[FACT] Results:** text8 24-layer adaptive 1.01 dev / **1.07 test** at 209M params / 179M FLOPS vs
  T-XL 277M / 438M / 1.08; enwik8 24-layer adaptive **0.98 test** at 209M/181M vs T-XL 277M/438M/0.99.
  FLOPS cut "up to 70%" at inference.

**[REASONING] Why this matters a great deal for the proposal.** This is the cleanest published
experiment on the question "if you let a model choose its own receptive field, what does it choose?"
The answer, in a *character-level* LM (the setting most favorable to long local context), is
**overwhelmingly short — 3–4% of the available span, with the lowest 5 layers pinned at the minimum.**
Three further points:
1. It converges with everything else in this dossier: Sieberling's flat W≥3 curve, TCN's k=3-optimal on
   word PTB, LightConv's small +0.3 BLEU from its span schedule, and MixConv's per-layer concentration.
   **Four independent lines of evidence say most layers want short context.**
2. **The *dynamic* (input-dependent) span variant tied the static one** (1.08 vs 1.08 bpc). That is a
   direct precedent for the proposal's central mechanism failing to add value: making the receptive
   field token-dependent, rather than merely learned-per-head, bought zero.
3. **[FACT] Their memory caveat transfers exactly to Section 5.4:** because heads in a layer share state
   vectors, it is the **maximum** span in a layer, not the average, that governs memory — "so a single
   long-span head limits savings for its layer." **Identical to the max-lag argument for the conv
   cache: one branch reaching lag 14 forces a 15-slot buffer regardless of what the others do.**

---

## 4. Routing mechanics and its pitfalls

### 4.1 Why the dense softmax router saves NO compute

**[REASONING], and it is not subtle.** A softmax router producing $\alpha^{(t)} \in \Delta^3$ with
$y_t = \sum_b \alpha^{(t)}_b y^{(b)}_t$ requires **every** $y^{(b)}_t$ to exist. All four branches are
evaluated for every token. The proposal therefore costs:

$$\text{cost} = \underbrace{4 \times \text{conv}}_{\text{branches}} + \underbrace{D\!\to\!4 \text{ projection}}_{\text{router, negligible}} + \underbrace{\text{softmax} + \text{4-way weighted sum}}_{\text{elementwise passes}}$$

versus $1 \times \text{conv}$ for the baseline. **It is strictly more expensive — roughly 4x on the conv
op, which is the memory-traffic argument of Section 7.2.** There is no compute-saving story available
for the dense variant; it must win purely on quality per unit compute, and it must beat the
**FLOP-matched** baseline B8 (spend the same extra compute on width/depth/tokens instead).

### 4.2 Top-k / hard routing: gradient mechanics

**[FACT] Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
Layer"** — arXiv 1701.06538, <https://arxiv.org/abs/1701.06538>. Noisy top-k gating. They
"conjectured that routing to k>1 experts was necessary in order to have non-trivial gradients to the
routing functions," reasoning a router cannot learn without comparing at least two experts.

**[FACT] Switch Transformer contradicted this** — arXiv 2101.03961, <https://arxiv.org/abs/2101.03961>.
Routing to a single expert "preserves model quality, reduces routing computation and performs better."
The mechanism: the gate value $p_i(x)$ **multiplies the expert output**, so the router stays
differentiable at k=1 — no cross-expert comparison is needed for gradient flow.

**[REASONING] Consequence for this proposal:** the **dense softmax over 4 branches has no gradient
pathology at all** — every branch receives gradient every step. That is a genuine advantage over MoE
and means load-balancing losses and z-losses are *not needed* for the dense variant. The pathologies
below only bite if you go to hard/top-1 routing. So: **the dense router is the safe design; it just
doesn't save anything.**

### 4.3 Load balancing and the z-loss (needed only if you go sparse)

**[FACT] Switch Transformer load-balancing auxiliary loss:**
$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$, where $f_i$ is the fraction of tokens
dispatched to expert $i$ and $P_i$ the fraction of router probability mass to expert $i$. Both ideally
$1/N$; the $N$ factor makes it scale-invariant in expert count. **Only $P$ is differentiable; $f$ is
not.** Coefficient **$\alpha = 10^{-2}$**, swept from $10^{-1}$ to $10^{-5}$, chosen as "sufficiently
large to ensure load balancing while small enough to not to overwhelm the primary cross-entropy
objective."

**[FACT] Capacity factor:** $\text{expert capacity} = (\text{tokens per batch}/\text{num experts})
\times \text{capacity factor}$. Overflow tokens are **dropped** — computation skipped, representation
passed forward via the residual. Tested 2.0 / 1.25 / 1.0 (128 experts, 32 TPUv3 cores). Token drop
rates typically **under 1%**.

**[FACT] Selective precision — the router must be float32.** Table 2 (32 experts):

| Model | Neg. log perp ↑ | Examples/sec ↑ |
|---|---|---|
| Switch-Base (float32) | −1.718 | 1160 |
| Switch-Base (bfloat16) | **−3.780 [diverged]** | 1390 |
| Switch-Base (selective precision) | −1.716 | 1390 |

Pure bfloat16 **diverges**; casting only the router body to float32 recovers float32 quality at full
bfloat16 speed. **[REASONING] Directly actionable: compute your 4-way router logits and softmax in
fp32 even under bf16 training.** It is nearly free (a rank-4 projection) and this table shows the
failure mode is divergence, not mild degradation.

**[FACT] Speedups:** "up to 7x increases in pre-training speed" — Switch-Base/64-experts reaches
T5-Base's step-60k quality at step 450k (~7.5x fewer steps), and on wall-clock hits comparable
perplexity in "one-seventh the time" at equal FLOPs/example.

**[FACT] ST-MoE router z-loss** — arXiv 2202.08906, <https://arxiv.org/abs/2202.08906> (Zoph, Bello,
Kumar, Du, Huang, Dean, Shazeer, Fedus). Exact formula:

$$L_z(x) = \frac{1}{B}\sum_{i=1}^{B}\left(\log \sum_{j=1}^{N} e^{x_j^{(i)}}\right)^2$$

with $x$ the router logits, added as $L_{tot} = L_{CE} + c_B L_B + c_z L_Z$ and coefficient
**$c_z = 0.001$**, chosen by sweep.

**[FACT] Why large logits destabilize:** roundoff amplified through exponentials. bfloat16 has 7
mantissa bits vs float32's 23, so ~"65,536x" worse roundoff. Their worked example: ten logits at 128 and
one at 128.5 — a 0.5 bfloat16 roundoff shifts the softmax output by **36%**, collapsing 0.142 → 0.091.
Clipping is argued inferior because it acts *after* rounding, creating larger discontinuities; z-loss
instead pushes logits toward small, precisely-representable magnitudes.

**[FACT] The stability-vs-quality table — this is the key result.** Baseline stable in **4/6** seeds,
quality **−1.755 ±0.02**. All variants 3 seeds, stable 3/3:

| Intervention | Fraction stable | Quality ↑ | Effect |
|---|---|---|---|
| Baseline | 4/6 | −1.755 ±0.02 | — |
| Remove GEGLU | 3/3 | −1.849 ±0.02 | stabilized, hurt quality |
| Remove RMSNorm scale param | 3/3 | −2.020 ±0.06 | stabilized, hurt badly |
| Input jitter (1e-2) | 3/3 | −1.777 ±0.03 | stabilized, mild loss |
| Dropout (0.1) | 3/3 | −1.822 ±0.11 | stabilized, hurt |
| Update clipping (0.1) | 3/3 | **−4.206 ±0.17** | "catastrophic loss of quality" |
| **Router z-loss** | 3/3 | **−1.741 ±0.02** | **stabilized AND slightly improved** |

**[REASONING] Two lessons.** (a) Router z-loss is the only free lunch — if you use any softmax router,
add it at $c_z = 10^{-3}$. (b) **The baseline was unstable in 2 of 6 seeds.** Routing-based
architectures have a real seed-variance problem, which is independent corroboration of the
multi-seed requirement in the knockout protocol (Section 6.4, Tier 0 step 3). **A 1-seed comparison of
a routed architecture against a non-routed baseline is not evidence.**

**[FACT] Expert Choice routing** — arXiv 2202.09368, <https://arxiv.org/abs/2202.09368> — inverts
routing so experts pick tokens, guaranteeing balance without an auxiliary loss.
**[UNKNOWN]** I did not verify its exact speedup numbers.
**[FACT] GShard** — arXiv 2006.16668, <https://arxiv.org/abs/2006.16668> — top-2 routing + aux loss;
trained MoE entirely in float32 (contrast with Switch's selective precision).

### 4.4 Hard routing: Gumbel / straight-through

**[FACT]** Gumbel-Softmax — arXiv 1611.01144, <https://arxiv.org/abs/1611.01144>; Concrete
distribution — arXiv 1611.00712, <https://arxiv.org/abs/1611.00712>. Both provide a continuous
relaxation of categorical sampling with a temperature that anneals toward hard one-hot.
**[REASONING]** For 4 branches this is technically easy but adds: a temperature schedule (another
hyperparameter that interacts with LR), gradient bias from the straight-through estimator, and
train/inference mismatch (soft at train, hard at inference). Given Section 4.5's verdict that hard
routing cannot save wall clock here, **there is no reason to pay these costs.**

### 4.5 Granularity: when does conditional compute actually pay off?

**[FACT] Sparsity crossover** — Gale et al., arXiv 2006.10901 (detail in Section 7.6): "sparse
computation exceeds the performance of dense at as low as **71% sparsity**" — with hand-optimized
SC20-grade kernels on a **compute-bound** GEMM (8192x2048), reaching 27.3% of fp32 peak.

**[FACT] Fragmentation cost** — ShuffleNetV2 G3 (Section 7.3): 4-way parallel branches at **matched
FLOPs** run **3.54x slower** on GPU (2446 → 691 batches/s at c=128).

**[FACT] MegaBlocks** — arXiv 2211.15841, <https://arxiv.org/abs/2211.15841> — block-sparse MoE
addressing token dropping/padding. **[UNKNOWN]** exact speedup figures not verified by me.

**[FACT] Structured vs unstructured sparsity:** NVIDIA's 2:4 structured sparsity is the only sparsity
pattern with hardware support on Ampere+ tensor cores, with a **~2x ceiling**; unstructured sparsity has
no such path. **[UNKNOWN]** I did not retrieve a primary NVIDIA citation for this in-session — treat the
2x figure as widely-reported-but-unverified-here.

**[REASONING] The granularity principle these results share:** conditional compute pays off when the
skipped unit is (a) **large** (a whole FFN of millions of FLOPs, so the gather/scatter amortizes), and
(b) **compute-bound** (there are real FLOPs to skip). MoE works because an expert FFN is both. Our
branches are the opposite on both counts: each is a **depthwise 3-tap conv at 1.5–7.5 FLOP/byte**,
i.e. tiny and hard memory-bound. Top-1 of 4 is only **75% sparsity** — barely past the 71% crossover
that required bespoke kernels on a *compute-bound* op.

**CONCRETE VERDICT [REASONING]: hard/top-k routing over 4 tiny depthwise causal conv branches cannot
beat dense evaluation of all 4 on a GPU. Do not build it.** Reasons, ordered:
1. The branches are memory-bound — sparsity saves FLOPs, which are not the bottleneck. A top-1 router
   still reads every token's activations.
2. 75% sparsity sits at the very edge of the measured 71% crossover, which was established for a
   compute-bound GEMM with hand-tuned kernels.
3. Gather/scatter, extra launches, and synchronization dominate: ShuffleNetV2 measured up to **3.5x**
   penalty for 4-way parallelism at matched FLOPs.
4. The correct optimization runs the opposite way — **fuse the branches into one 15-tap kernel**
   (Section 7.2), which makes them nearly free and simultaneously reveals (Section 5.1) that the
   branch decomposition was never load-bearing.

### 4.6 Routing collapse with few experts

**[FACT]** ST-MoE's own baseline was unstable in **2 of 6 seeds**, and every stability intervention
except z-loss cost quality (table above).
**[UNKNOWN]** I did **not** find a paper specifically studying router degeneracy at very small expert
counts (2–8) with entropy measurements. This appears to be a genuine gap in the literature — MoE work
concentrates on 8–2048 experts where balance is the concern, not on 4 branches where the concern would
be that the router simply learns a constant. **Flag this as unknown rather than assuming either way,
and measure it directly** (Section 6.4, Tier 0 steps 1–2). **[REASONING]** My prior: with only 4
branches and a dense softmax, the most likely outcome is not instability but *blandness* — the router
converges to a near-constant per-layer mixture, making the whole thing equivalent to baseline B3 and
hence to a single 15-tap kernel. This is the outcome the experiment most needs to be able to detect,
which is why the router-entropy and input-dependence diagnostics are Tier 0.


---

## 9. Short convolutions in modern LM architectures — the width consensus

### 9.1 The consensus, and TWO independent width sweeps that both peak at 3

**[FACT] Every modern LM short conv I could verify uses width in {2, 3, 4}, and none uses dilation:**

| Architecture | Width | Depthwise | Causal | Position | Width ablated? |
|---|---|---|---|---|---|
| **LFM2** | **3** (`conv_L_cache`) | yes | yes | inside double-gated conv block | NAS-searched, **curve unpublished** |
| Hyena | 3 | yes | yes | after in_proj, before long conv | no |
| M2 / Monarch Mixer | 3 (hardcoded) | yes | yes | after X1/X2/V projections | no |
| Based | 3 | yes | yes | gated conv layers | on/off only |
| Mamba / Mamba-2 | **4** (`d_conv=4`) | yes | yes | pre-SSM (Mamba-2: on `xBC`) | **NO — value appears only in code** |
| Samba | 4 | yes | yes | in Mamba layer, pre-SSM | on/off only (Table 10) |
| Zamba / Zamba2 | 4 | yes | yes | Mamba block | no |
| DeltaNet / Gated DeltaNet | 4 | yes | yes | after Q/K/V | on/off only |
| Hawk / Griffin | 4 | yes | yes | recurrent branch, pre-RG-LRU | no |
| Canon layers | 4 | yes | yes | 4 insertion points | **no — named as future work** |
| RWKV-4/5/7 | **2** (token shift) | yes | yes | on R,K,V | no |
| flash-linear-attention | **`conv_size=4`** in 15/15 models | yes | yes | on q/k/v | no |
| H3 shift-SSM (the ancestor) | **m=64 state** | yes | yes | on K, pre-diag-SSM | no |
| StripedHyena-2 Hyena-SE | **7** (range 4–7) | yes, grouped | yes | *inner sequence mixer* | layouts yes, width no |

**[FACT] SWEEP 1 — arXiv 2606.03825** (Table 3a; 300M, 15B tokens, Nemotron-CC ppl, Q+K+V, R=16):
W=1 18.42, W=2 18.17, **W=3 18.08**, W=4 18.10, W=5 18.09, W=6 18.10; no-conv 19.12. Verbatim:
"3 or 4 is generally the sweet spot… **Widths beyond this sweet spot do not provide additional gains
even though they add parameters.** This has generally been found to be the case for static convolutions
as well."

**[FACT] SWEEP 2 — "Convolution for Large Language Models," arXiv 2607.18413**,
<https://arxiv.org/abs/2607.18413> (Tian et al., PKU / Huawei / Tsinghua). Residual depthwise Conv1D on
post-QKV in **Qwen3-1.7B**:

| Kernel | Mean loss | PPL | Params (M) |
|---|---|---|---|
| no Conv1D | 2.5144 | 13.42 | 1720.57 |
| k=2 | 2.4894 | 12.99 | 1720.92 |
| **k=3** | **2.4795** | **12.79** | 1721.03 |
| k=4 | 2.4881 | 13.13 | 1721.15 |

Verbatim: "k = 3 gives the lowest reported perplexity, 12.79. **Increasing the kernel size to k = 4 does
not yield a further improvement**… although the ablation does not identify why k = 3 performs best."
Downstream: +1.99 to +3.76 average points on 7 benchmarks at **<0.01% added params**.

**★★ [FACT] THE MOST DIRECTLY ADVERSE RESULT IN THE ENTIRE DOSSIER ★★** — the same paper tried
**multi-branch reparameterization over MIXED kernel sizes, and it HURT: 12.79 → 13.28.**
**[REASONING] This is the proposal's core idea (parallel branches at several widths, in an LM short-conv
slot), tested, and it lost to a single k=3 kernel by 0.49 PPL.** It is the closest published test of the
dense-multi-width variant that exists, and the result is negative. Caveats to state honestly: single
runs, grid only {2,3,4}, and a recent (July 2026) tech report — so it is suggestive rather than
decisive, but it is *direct* and it points the same way as everything else.

**ANSWER: no one has found a gain from short-conv width > 4 in an LM.** W=5 and W=6 were explicitly
tested and gave nothing (2606.03825); k=4 was already worse than k=3 (2607.18413); mixing widths hurt;
FlexConv's fixed k=33 was catastrophic (78.0 vs 89.5 at k=3). The apparent exception, StripedHyena-2's
length-7 Hyena-SE (arXiv 2503.01868, <https://arxiv.org/abs/2503.01868>, "range of 4 to 7"), is an
**inner sequence mixer replacing a long conv**, not a featurizer short conv, and it is byte-level DNA.

### 9.2 Why 3–4 — including a tooling cause that is *causal*, not incidental

**[FACT] (a) The job is local composition.** Based: "local, precise shifts for token comparisons."
arXiv 2607.18413: "a compact kernel being sufficient for short-range composition."

**[FACT] (b) Inherited lineage, with a silently dropped width.** Griffin/Hawk (arXiv 2402.19427):
"we apply a small separable Conv1D layer, inspired by the **Shift-SSM in H3**," temporal filter
dimension 4, "just 4·D_RNN parameters." But **H3's own shift SSM used state size 64** — an effective
width-64 short conv (H3 explicitly notes "B can also be fixed to e₁, in which case the output is a 1D
conv. with kernel size m"). **[REASONING] So the field narrowed 64 → 4 without ever publishing the
ablation that justified it.** That is a genuine gap — and it is the one place where "nobody tested wider"
is literally true rather than "tested and flat."

**[FACT] (c) The `causal_conv1d` width cap is a *causal* factor, verified in source.** In
`csrc/causal_conv1d.cpp` at **three** sites (fwd 182, bwd 289, update 496):
`TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");`
**This is a hard abort, not a slow fallback.** And two papers say outright that this determined their
choice: Samba (arXiv 2406.07522) sets k=4 "**for hardware-aware efficiency**"; **Canon layers**
(arXiv 2512.17351, <https://arxiv.org/abs/2512.17351>, Allen-Zhu, NeurIPS 2025) chose kernel size 4
"for simplicity and efficiency, **available through efficient CUDA kernels implemented by the
open-source H3 library (pip package causal_conv1d)**," and explicitly flag the gap: "We focused on
simple linear convolutional (kernel size 4) Canon layers… **Future work should explore dynamic,
adaptive convolutions.**"

**★ [FACT] IMPORTANT CORRECTION to Section 7.4 — the cap is NOT fundamental.**
`flash-linear-attention`'s **Triton** conv backend has **no width limit**:
`fla/modules/conv/triton/ops.py` uses `BW = triton.next_power_of_2(W)`, supporting arbitrary width.
**[REASONING] Two consequences.** (1) A **dense k=15 causal depthwise conv IS implementable with a fast
fused kernel today** via FLA's Triton path — so baseline B1/L1 is not handicapped, and the "widening one
fused kernel is nearly free" argument of Section 7.2 is *practically* realizable, not just theoretical.
This makes the mandatory wide-kernel baseline both cheap and fair. (2) The consensus at width 4
**outlives the constraint that helped create it** — every FLA model still defaults to `conv_size=4`
despite the backend supporting more. So part of "everyone uses 4" is path dependence, which is a fair
point *for* re-examining width. But note the two direct sweeps that did re-examine it found flat/worse.
**[UNKNOWN]** Whether FLA's Triton conv supports **dilation** was still not established — assume not
until checked.

### 9.3 Presence matters far more than width; and input-dependence has been reverted before

**[FACT]** Conv-vs-no-conv is worth ~1.0 PPL (19.12 → 18.10) and 0.63 PPL (13.42 → 12.79). Width 3-vs-6
is worth **0.02 PPL**. Placement also dominates width (V-only 18.56 vs Q+K+V 18.10).
**[FACT] Samba's Table 10** (SlimPajama val ppl @4096/8192/16384), short conv off → on:
Llama-2-SWA 438M 11.12→**10.83** / 10.66→**10.39** / 10.57→**10.31**; Sliding GLA 10.43→10.39 (small);
Sliding RetNet 10.38→**10.25** / 9.96→**9.82**. Their comment: adding SC "**significantly improve[s] the
SWA's performance**, while the effect on GLA is less prominent… we leave it as future work to understand
the **surprising effectiveness of SC** in language modeling."

**★ [FACT] RWKV tried input-dependent shift and REVERTED it.** RWKV token shift is a 2-tap causal
depthwise conv with learned per-channel interpolation $\mu$. RWKV-6 made it **data-dependent**; RWKV-7
(arXiv 2503.14456, <https://arxiv.org/abs/2503.14456>) **removed that**, stating the data-dependent
version "was beneficial in terms of loss decrease per step, [but] **the improvement in training and
inference efficiency was not worthwhile**." RWKV-7 also confirms the framing: "Token shift is a variety
of 1D short convolution… Many other modern models (e.g. Mamba) have begun including short convolutions
before attention replacements."
**[REASONING] This is a second precedent — alongside Adaptive Attention Span's dynamic-span tie
(1.08 vs 1.08 bpc) — for input-dependence in a short-conv/shift primitive being measurable but not
worth its cost.** Exactly the outcome the proposal risks.

**[FACT] LFM2 Technical Report** — arXiv 2511.23404, <https://arxiv.org/abs/2511.23404>. Table 1 column
"Conv k" = **3 for every model size**. Exact block:
$(B, C, \tilde h) = \mathrm{Linear}(h);\; y = B \odot \tilde h;\; z = \mathrm{Conv}_k(y);\;
o = \mathrm{Linear_{out}}(C \odot z)$. Layer mix confirmed: LFM2-1.2B = 16 layers, attention at
[2,5,8,10,12,14] → 10 conv + 6 GQA; LFM2-2.6B = 22 conv + 8 attn. **[FACT] STAR's search space
explicitly included "gated short convolution blocks with varying kernel sizes"** — but the selected
per-layer widths and any width-vs-quality curve are **not published**. Their conclusion: once a few GQA
blocks exist, "the inexpensive gated short convolution alone is sufficient to reach the best
quality–latency–memory trade-off… **without additional linear attention/SSM/long convolution
branches**."
**[REASONING] Read that last clause carefully — it is LFM2's own finding that adding extra
sequence-mixing branches to the short conv was unnecessary.** The proposal adds exactly such branches.

### 9.4 Learned receptive fields: small early, growing with depth, far below any global bound

**[FACT] FlexConv** — arXiv 2110.08059, <https://arxiv.org/abs/2110.08059> (Romero, Bruintjes et al.,
ICLR 2022). Learns kernel size via a Gaussian mask multiplying a continuous kernel,
$\psi(x,y) = w_{\text{gauss}}(x,y;\theta_{\text{mask}})\cdot \mathrm{MLP}_\psi(x,y)$, learning
$(\mu_X,\sigma^2_X,\mu_Y,\sigma^2_Y)$ per axis per layer, initialized **small** ($\sigma^2 = 0.125$),
mask LR 0.1x base. **Fig. 6 caption, verbatim: "FlexNets learn very small kernels at shallow layers,
which become larger as a function of depth."** And §5: "FlexNet **could** learn a different prior…
e.g., large kernels first, and small kernels next. **However, FlexNets learn to increase kernel sizes
progressively (Fig. 6)**." Since init is uniformly small, the depth profile is genuinely learned.
**[FACT] The middle beats both extremes (CIFAR-10):** fixed k=3 → 89.5±0.3 (0.17M);
**fixed k=33 → 78.0±0.3 (20.0M)**; **learned FlexNet-16 → 92.2±0.1 (0.67M)**. "The best solution is
somewhere in the middle." Cropping where mask > 0.1 gives **11.8x** (SpeechCommands) and **5.5x**
(CIFAR-10) per-epoch speedups.
**[UNKNOWN]** No per-layer *numeric* learned sizes are published (Fig. 6 is qualitative heatmaps).

**[FACT] CKConv** — arXiv 2102.02611, <https://arxiv.org/abs/2102.02611>. Kernel = 3-layer SIREN MLP,
32 hidden units, full-sequence kernel, FFT-evaluated. **Extremely sensitive to the $\omega_0$ frequency
prior: "performance on pMNIST may vary from 98.54 to 65.22 for values of $\omega_0$ in [1,100]."**
Depth becomes unnecessary once kernels are global: 2/4/8/16 blocks → 99.21/99.26/99.29/99.19.

**[REASONING] Convergence across five independent methodologies.** Learned-span attention (avg 314 of
8192 permitted; lowest 5 of 12 layers at the floor), learned-size convolution (small early, growing with
depth, k=33 catastrophic), two LM width sweeps (flat past 3), TCN (k=3 best on word PTB), and LFM2's own
NAS (k=3 everywhere, no extra branches needed) **all say the same thing: the useful receptive field of a
short conv in a language model is small, and it varies by DEPTH rather than by TOKEN.** The proposal
varies it by token, within a single layer, which is the axis none of this evidence supports — and
L1b (per-layer schedule) is the cheap control that tests the axis the evidence *does* support.

### 9.5 Long-conv context: the decay prior is the real lesson

**[FACT] SGConv** — arXiv 2210.09298 (details and construction in the earlier long-conv discussion).
Its **decay ablation is the load-bearing one** (Fig. 3, IMDB, d=8, sweeping decay speed $t$):
t=0 (no decay) ≈**67.7%** → t=0.5 ≈83.3 → t=0.75 ≈86.9 → t=1.0 ≈**88.9** → t=1.5 ≈88.8 → t=2.0 ≈88.9.
**~21 accuracy points from adding decay alone.** Base sub-kernel width $d$ (t=1): d=1 88.7, **d=2 89.1**,
d=4 88.8, d=8 88.9, d=16 88.8, d=32 88.3, d=64 88.5 — nearly flat, mild decline past 8 (overfitting).
**[DIGITIZED]** These are read off Figure 3; no numeric table exists; ±0.2 in the flat region, ±0.5 at
t=0. **[UNKNOWN]** No ablation on the *number of scales* N — N is derived from L and d, not free.

**[FACT] "Simple Hardware-Efficient Long Convolutions"** — arXiv 2302.06646. Exact regularizers:
**Squash** $\bar K = \mathrm{sign}(K)\cdot\max(|K|-\lambda, 0)$ (one L1 prox step), $\lambda \in
[0.001, 0.005]$; **Smooth** $\bar K_k = (2p+1)^{-1}\sum_j K_{k+j-p}$. LRA avg: no-reg 69.5 →
Rand+Squash 86.1 → **Exp+Squash 86.6**. Striking: **only Squash-without-Smooth clears Path-X at all**
(96.9 / 96.0, beating S4-LegS's 96.4; all other variants fail to beat chance). FlashButterfly: **7.0x**
over HF Transformer at 4K LRA; **2.2x** over cuFFT long conv at 128K. Kernel fusion works "only for
convolutions short enough to fit into SRAM (**length 8K or shorter on A100**)."

**★ [FACT] CONFIRMS my earlier [UNKNOWN]: there is NO FFT-vs-direct crossover discussion in that
paper.** A grep of both arXiv v1 and the ICML camera-ready for "direct convolution"/"crossover" returned
zero hits; everything is FFT-based, and kernels are always full sequence length (no kernel-length
ablation). The only threshold stated is the SRAM one above. **[REASONING] So Section 7.5's verdict
stands on first-principles grounds, not on a citation: at $W=15$ and $T \le 4$k, direct convolution
wins, and FFT is irrelevant here.** **[UNKNOWN]** Smooth's window $p$ is never given a numeric value.

**[REASONING] The transferable lesson for baseline L1:** the single most valuable inductive bias in the
long-conv literature is **magnitude decay with lag** (~21 points in SGConv; L1-shrinkage essential in
2302.06646). A dense k=15 kernel can learn decay freely and should be *initialized* with decaying
magnitude to make it a strong baseline. The **dilated variant actively violates this prior** — it
assigns full-magnitude taps at lags 7, 8, 14 while structurally zeroing lags 3, 5, 6 and 9–13
(Section 5.2). That is the opposite of a decay prior, and it is a further mechanistic reason to expect
it to underperform.

---

## 8. EXPERIMENT DESIGN IMPLICATIONS

### 8.1 The ladder of variants, cheapest control first

**[REASONING]** Ordered so that each rung can kill the hypothesis before you spend the next rung's
compute. **Stop early if a rung fails.** All rungs: same tokens, same data order, **≥3 seeds**, report
mean ± spread.

**Three non-negotiable recipe requirements for any router rung**, each from a specific published failure:
1. **Router logits and softmax in fp32** even under bf16 — pure bf16 *diverged* in Switch Transformer
   (−3.780 vs −1.718); selective precision recovers full quality at full speed.
2. **Softmax temperature schedule, τ≈30 annealed to 1 over the first ~10% of training** — a plain
   τ=1 softmax over K=4 kernels scored *below the static baseline* in Chen et al. (64.8 vs 65.4), and
   the whole +4.5 gain came from annealing. **Without this, a null result is uninterpretable.**
3. **Router z-loss at c_z = 1e-3** — the only intervention in ST-MoE's stability table that improved
   quality rather than trading it away (−1.741 vs −1.755 baseline, 3/3 vs 4/6 seeds stable).

| Rung | Variant | Params/ch | Cache | Cost | What it decides |
|---|---|---|---|---|---|
| **L0** | LFM2 baseline, k=3 | 3 | 3 | 1.0x | reference |
| **L1** | Single dense k=5, k=9, k=15 (3 runs) | 5/9/15 | 5/9/15 | ~1.0x | **Does span help at all?** Replicates the Sieberling + Tian curves in your setup. **Implement via FLA's Triton conv (arbitrary width) so the baseline is not tooling-handicapped**, and initialize with decaying tap magnitude (SGConv's decay prior was worth ~21 pts). **If flat, STOP — the premise is false.** |
| **L1b** | **Per-layer increasing span schedule** (e.g. 3,3,5,5,9,9,15,... across the 10 conv layers), one dense kernel per layer | 3–15 | 3–15 | ~1.0x | **Multiscale ACROSS layers instead of within one.** This is LightConv's actual published design (k=3,7,15,31x4) and bought +0.3 BLEU there. No router, no branches, fully fusable, ~free. **If this captures the gain, the whole router is unnecessary.** |
| **L2** | Single dense k=32 (tap-matched to the branch variant) | 32 | 32 | ~1.0x | Is any gain just parameters? |
| **L3** | 4 branches, **frozen uniform** weights (1/4) | 12 or 32 | 15 | ~4x op | Does branch *structure* help without any learned mix? (= a fixed sparse 15-tap kernel) |
| **L4** | 4 branches, **learned input-independent** weights | 12 or 32 | 15 | ~4x op | **Provably equal to L1/L2's function class (Sec 5.1).** Any gain here over L1/L2 is a *reparameterization/optimization* effect, not expressiveness — and is the RepVGG question, which is genuinely interesting and free at inference (fuse it). |
| **L4b** | **TWO branches only** (k=3 + k=9, or dilations 1+4) with the router | 6–14 | 9 | ~2x op | **Every published branch-count sweep saturates at 2** (SKNet M=2→3 buys 0.03%; MixConv g=2; Res2Net s=2). If 2 branches capture the gain, 4 is pure waste — and this rung is half the cost. **Run this BEFORE L5.** |
| **L5** | 4 branches + **token-dependent softmax router** (the proposal) | 12/32 + D·4 | 15 | ~4x op + router | **Does input-dependence of the tap pattern help, above everything else?** |
| **L6** | **Unconstrained dynamic short conv** (Sieberling low-rank, R=16/32) at k=3 and k=15 | 3/15 + gen | 3/15 | ~1.1x | **The real competitor.** Published 1.33x compute advantage. **Expect this to beat L5.** |
| **L7** | Hard/top-1 router (Gumbel or ST) | — | 15 | ≥4x op | **DO NOT BUILD** (Section 4.5 verdict). Include only if you want a negative throughput result. |

**Priority if compute is tight [REASONING]: run L0, L1, L1b, L4, L6 and nothing else** (add L4b before
L5 if you do proceed to a router). That set answers
the entire question: L1 tells you whether span matters at all, **L1b tells you whether multiscale is
better obtained across layers than within one (the published approach, and nearly free)**, L4 tells you
whether the branch structure matters absent input-dependence, L6 tells you whether the *right* form of
input-dependence beats your constrained one. L5 (the actual proposal) is only worth running if L1 shows
a real span gain **and** neither L1b nor L4 captures it.

### 8.2 Mandatory parameter-matched baselines (non-negotiable)

**[REASONING]** Without all four of these, a "multiscale helps" claim is vacuous — each is a cheaper
explanation of any observed gain:

1. **Single dense k=15** (L1). *Cheaper in parameters than the dense-branch variant: 15 vs 32.*
   Rules out "the gain is span." **This is the single most important baseline and it is currently
   missing from the proposal.**
2. **Tap-count-matched single dense kernel** (L2: k=32 for the dense-branch variant, k=12 for the
   dilated). Rules out "the gain is parameters."
3. **Fixed-weight 4-branch** (L4). Rules out "the gain is input-dependence." Provably the same function
   class as (1)/(2) per Section 5.1.
4. **FLOP- and wall-clock-matched L0**: baseline k=3 with the ~4x conv compute spent instead on more
   width, more depth, or more tokens. Rules out "the gain is just more compute." Report both
   *iso-parameter* and *iso-wall-clock* comparisons — they will disagree, and the honest headline is
   the iso-wall-clock one.

**Plus, for interpreting the router:** router-input-ablated control (feed the router a constant), and
router-frozen-at-learned-mean control.

### 8.3 Knockout protocol

Full version in Section 6.4. The irreducible core:
- **Tier 0 (free):** router entropy per layer over training (is it uniform → fixed mixture, or constant
  → single branch?); variance decomposition of $\alpha^{(t)}$ into position vs token-identity vs context
  (**if a per-position or per-token-ID lookup table reproduces the router, replace it with that and
  save the compute**); seed stability of any claimed pattern.
- **Tier 1 (cheap, understates):** **resample-ablate** and **mean-ablate** each branch — *not* zero-ablate
  (Zhang & Nanda: OOD interventions give "unreliable or illusory results"; report zero-ablation only
  as a labeled-biased third number). Renormalize the remaining softmax. Metric = **logit difference /
  Δ log-likelihood on targeted probes**, not aggregate ppl (Michel et al.: 91.7% of heads look useless
  under aggregate metrics).
- **Tier 1's decisive test:** **synthetic lag-$L$ dependency task, sweep $L \in \{1..14\}$**, and check
  that knocking out the wide branch selectively damages large $L$. **If the damage curve is flat in $L$,
  the branches are not span-specialized and the multiscale story is dead** regardless of any perplexity
  win.
- **Tier 2 (expensive, sufficient):** **retrain from scratch without the branch.** This gap — not the
  inference-time gap — is the branch's true contribution.

### 8.4 State-size accounting (summary)

**[FACT]** LFM2-1.2B real config: `conv_L_cache: 3`, `conv_dim: 2048`, `num_hidden_layers: 16`,
`full_attn_idxs: [2,5,8,10,12,14]` → 10 conv + 6 GQA layers, `num_key_value_heads: 8`, head_dim 64.

| | Baseline k=3 | Any span-15 variant | Ratio |
|---|---|---|---|
| Cache slots/channel | 3 | 15 | **5x** |
| Per conv layer (d=2048, bf16) | 12.0 KB | 60.0 KB | 5x |
| All 10 conv layers | **120 KB** | **600 KB** | 5x |
| For scale: 6 GQA layers @ 4k ctx | 50.3 MB | 50.3 MB | — |
| For scale: 6 GQA layers @ 32k ctx | 393.2 MB | 393.2 MB | — |

**[REASONING]** The conv state remains small *relative to attention*, so "bounded tiny state" survives
in absolute terms. Three caveats that must be stated: (a) the **5x** growth is real and unavoidable for
any span-15 design; (b) **sparsity buys zero state savings** — max lag sets buffer depth, so the
12-tap dilated variant pays the same 600 KB as a dense 15-tap kernel while being less expressive;
(c) LFM2's stated design target is **embedded SoC CPU with peak memory as the search objective**, so a
5x increase in the one state component the architecture was chosen to minimize is a regression against
the architecture's own design criterion, not a rounding error.

### 8.5 Honest prior on whether this will pay off

**[REASONING]** My assessment, with the reasoning exposed so it can be disagreed with:

**Probability the full proposal (L5) beats a parameter-matched dense k=15 (L1) by a margin that
survives 3 seeds: I estimate 15%** — revised *down* from 20% by arXiv 2607.18413's direct negative
result on mixed-width branches in an LM, then partly back *up* by the verified kernel-mixture evidence in
point 7 below (Chen et al.'s 69.4→36.0 averaging collapse and SKNet's router-beats-sum at M=2), which is
the strongest pro-proposal evidence in the dossier. Grounds:

1. **The span premise is contradicted by TWO independent LM width sweeps.** arXiv 2606.03825 is flat
   from W=3 to W=6 (18.08 → 18.10) at 300M/15B *with input-dependent filters*; arXiv 2607.18413 on
   Qwen3-1.7B peaks at k=3 (12.79) with k=4 already worse (13.13). The proposal's spans of 9 and 15 sit
   far right of a curve that saturated at 3. TCN independently found $k=3$ optimal on word-level PTB
   because "a smaller kernel … tends to focus more on the local context."
1b. **★ The mixed-width branch idea itself was tested in an LM and LOST.** arXiv 2607.18413's
   multi-branch reparameterization over mixed kernel sizes moved perplexity **12.79 → 13.28**. This is
   the single most on-point negative result available, and it is on the dense-multi-width variant
   specifically. (Single runs, small grid — suggestive not decisive, but direct.)
2. **The dilated parameterization was tried in the closest analogous setting and lost.** MixConv's
   dilated variant peaked at 2 branches and then **fell back to the no-mixing baseline** (~1.4 points
   below plain MixConv), and the authors excluded dilation from their search space, citing exactly the
   mechanism I quantified in Section 5.2 — skipping local information. My lag-coverage computation is
   independent confirmation: **7 of 15 lags reached, 8 structurally unlearnable, lag 0 covered 4x.**
3. **★ Every published branch-count sweep saturates at TWO branches, and the proposal uses four.**
   SKNet M=2 → M=3 buys **0.03%** for +1.8M params ("M = 2 is preferred"); MixConv's knee is g=2
   (55–65% of the gain); Res2Net s=4 → s=8 buys 0.15 pts for +12% latency. Three methodologies, one
   answer. **A 2-branch variant (L4b) at half the cost should be tested before a 4-branch one.**
3b. **★ A plain 4-way softmax over kernels lost to static in the only paper that measured it** —
   64.8% vs 65.4% (Chen et al., K=4), rescued only by τ=30→1 annealing. This is a recipe hazard that
   can manufacture a false negative, and it must be controlled for.
4. **The innovation budget is on the flat axis.** In arXiv 2606.03825, span (W 3→6) bought **0.00 ppl**
   while generator capacity (R 16→128) bought **0.25 ppl**, monotonically, still improving at R=128.
   A 4-way simplex softmax is about the **lowest-capacity** conditioning mechanism available. The
   proposal spends everything on span and almost nothing on conditioning capacity.
5. **The cost structure is inverted.** Because the op is memory-bound (1.5–7.5 FLOP/byte vs an H100
   ridge point of ~296), **widening one fused kernel to 15 taps is nearly free**, while **splitting
   into 4 branches multiplies memory traffic ~4x** and incurs fragmentation penalties (ShuffleNetV2:
   up to 3.54x at matched FLOPs for 4-way parallel). The proposal pays for the expensive thing to
   obtain the free thing.
6. **★ Input-dependence in this exact primitive has been tried and REVERTED twice.** RWKV-6 made token
   shift (a 2-tap causal depthwise conv) data-dependent; **RWKV-7 removed it** because "the improvement
   in training and inference efficiency was not worthwhile" (arXiv 2503.14456). And Adaptive Attention
   Span's dynamic token-conditional span **tied** the static one exactly (1.08 vs 1.08 bpc). Two
   independent reversions of "make the local receptive field token-dependent."
7. **★ LFM2's own authors concluded extra branches were unnecessary.** The LFM2 technical report
   (arXiv 2511.23404) states that once a few GQA blocks exist, "the inexpensive gated short convolution
   alone is sufficient to reach the best quality–latency–memory trade-off… **without additional linear
   attention/SSM/long convolution branches**." STAR's search space *did* include "gated short convolution
   blocks with varying kernel sizes," and it selected **k=3 for every model size** (Table 1).
8. **Implementation reality bites the "cheap" variant hardest.** `causal-conv1d` supports **widths
   2/3/4 only and has no dilation argument**; FLA likewise defaults to `conv_size: 4` with no dilation
   found. So the 12-tap dilated variant — marketed as cheap — has **no fast kernel for 3 of its 4
   branches** and will likely be the slowest thing you can build here.
9. **Where I could be wrong (the honest counter-case) — and it is stronger than I first assessed.**
   Three genuine points for the proposal, all now verified:
   (a) **In the kernel-mixture setting specifically, input-dependence IS load-bearing.** Chen et al.:
   replacing the learned router with kernel *averaging* collapses top-1 **69.4% → 36.0%**; shuffling the
   attention gives 14.8%/27.3%; DCD confirms with +3–5 points over static. That is the opposite of
   Branchformer/DWNet/Synthesizer, and it is the closest domain match for "mix several conv kernels."
   (b) **SKNet's router beat a plain uniform sum by +0.97 (21.76 → 20.79 err) at M=2** — exactly the
   L5-vs-L3 comparison, and it favours the router.
   (c) **A single large kernel is far worse than k=3 alone (SKNet: K5 25.14, K7 25.51 vs K3 22.23), yet
   the k3+k5 mixture beats both.** So "wide kernels only help inside a mixture" has direct support — a
   real argument that L1 (dense k=15) could lose to a mixture even though it is a superset.
   Counter-caveats to (a): those are *inference-time* ablations of a router-trained model, which
   Section 6.2 shows conflates importance with off-distribution damage — the fair test is retraining
   (L4); and it is 2-D vision at K=4 over full kernels, not causal 1-D over spans.
10. **Where I could also be wrong (structural).** The width sweep is a *single* width per layer,
   not a mixture. It is logically consistent for no single width >3 to help while a *token-conditional*
   mixture still helps, if the marginal information at lags 3–14 is near-zero *on average* but
   concentrated on a token subset (code, structured text, long identifiers) that a router can find.
   Nothing I retrieved tests that directly — **no paper in the multiscale-conv literature uses a
   learned router over multi-scale branches** (MixConv is `tf.split`/`tf.concat`, Res2Net is a fixed
   chain, Inception is fixed concat). So the specific combination is genuinely novel and genuinely
   untested. SKNet is the nearest vision analogue. **Novelty is real; the prior on it working is
   nonetheless poor**, because the mechanism it would exploit (information at lags 3–14) is exactly
   what measurement 1 says is nearly absent.

**Most likely empirical outcome [REASONING]:** L1 (dense k=15) roughly ties L0 (k=3), consistent with
the published flat curve. L4 and L5 also tie, with the router converging to a near-constant per-layer
mixture — i.e. **the null result is the modal outcome, and it will look like "everything is within
noise."** This is precisely why L1 and the ≥3-seed requirement matter: without them a 0.02-ppl seed
fluctuation on a single run will be over-read as a multiscale win.

### 8.6 The reframe I would actually recommend

**[REASONING]** Two salvage paths, both cheaper and more defensible than the proposal as written:

**(A) The RepVGG reparameterization question (cheap, novel, free at inference).** Since fixed-weight
branches fuse *exactly* into one 15-tap kernel (Section 5.1, verified to 1e-15), ask: **does training
the 4-branch parameterization reach a better optimum than directly training a 15-tap kernel, even
though the function classes are identical?** RepVGG's whole thesis is that training-time topology
matters when inference-time function does not (their Table 6: branches bought +2.75% top-1 and were
then fused away for free). This is a well-posed, inexpensive question, it needs no router, it costs
nothing at deployment, and a positive result would be genuinely publishable. **This is the strongest
version of the "multiscale" idea available.**

**(B) If you want input-dependence, do it the way that is measured to work.** Replace the 4-way
simplex router with a **low-rank unconstrained filter generator** (Sieberling low-rank, R=16–32) at
**W=3 or 4** — the configuration with a published **1.33x compute advantage** and **~8% end-to-end
overhead**. Then, if you still want to test span, sweep W within that framework. This subsumes the
proposal: your router is a rank-4 simplex-constrained special case of it (Section 5.3), so the
unconstrained version is a strict generalization that the literature says is better.

**What I would drop entirely:** the dilated parameterization (contradicted by MixConv, structurally
crippled at 7/15 lag coverage, and unsupported by every fast kernel in the ecosystem), and hard/top-k
routing (Section 4.5 verdict — it cannot pay off at this granularity).


---
