# Baseline Architectures & Training Infrastructure for the "Mostly-LIV" Hybrid Experiment

**Research date:** 2026-07-30
**Scope:** baselines, training codebases, small-scale pretraining recipes, corpora/tokenizers,
long-context protocol, and compute budgeting for a parameter-/token-/compute-matched comparison
of a modified LFM2-style mostly-LIV hybrid (gated short causal conv + few GQA attention layers)
against an all-GQA transformer, stock LFM2, and recurrent hybrids, at 100M-1B params.

**Epistemic convention used throughout:**

- **[FACT]** — read directly out of the cited paper / repo / model card / config file. Numbers as-published.
- **[INFER]** — my inference or arithmetic from published facts. Marked wherever used.
- **[UNKNOWN]** — could not verify from a primary source; flagged rather than guessed.

---

## 0. The reference architecture: what stock LFM2 actually is

This matters first because the whole experiment is defined relative to it, and because the HF
`config.json` files give exact ground truth (better than the blog post).

**Source of truth:** `https://huggingface.co/LiquidAI/LFM2-350M/raw/main/config.json` (and 700M /
1.2B / 2.6B equivalents), plus the reference implementation in HF transformers at
`src/transformers/models/lfm2/modeling_lfm2.py`.

### 0.1 LFM2 layer composition [FACT — from config.json + model card]

| | LFM2-350M | LFM2-700M | LFM2-1.2B | LFM2-2.6B |
|---|---|---|---|---|
| Params (exact) | 354,483,968 | 742,489,344 | 1,170,340,608 | 2,569,272,320 |
| `num_hidden_layers` | 16 | 16 | 16 | 30 |
| Layer split (model card) | 10 conv + 6 attn | 10 conv + 6 attn | 10 conv + 6 attn | 22 conv + 8 attn |
| `hidden_size` / `block_dim` | 1024 | 1536 | 2048 | 2048 |
| `block_ff_dim` (SwiGLU) | 6656 | 10240 | 12288 | 10752 |
| `num_attention_heads` | 16 | 24 | 32 | 32 |
| `num_key_value_heads` | 8 | 8 | 8 | 8 |
| `conv_L_cache` (kernel) | 3 | 3 | 3 | 3 |
| `conv_bias` | false | false | false | false |
| `vocab_size` | 65,536 | 65,536 | 65,536 | 65,536 |
| `rope_theta` | 1,000,000 | 1,000,000 | 1,000,000 | 1,000,000 |
| `max_position_embeddings` | 128,000 | 128,000 | 128,000 | 128,000 |
| Context length (card) | 32,768 | 32,768 | 32,768 | 32,768 |
| Training budget (card) | 10T tokens | 10T | 10T | 10T |
| License | LFM Open License v1.0 | same | same | same |

**Attention placement — exact, this is the key fact for the experiment** [FACT]:

For 350M/700M/1.2B, `full_attn_idxs = [2, 5, 8, 10, 12, 14]` over 16 layers (0-indexed).
So the stack is:

```
idx:  0    1    2     3    4    5     6    7    8     9    10    11   12    13   14    15
      conv conv ATTN  conv conv ATTN  conv conv ATTN  conv ATTN  conv ATTN  conv ATTN  conv
```

That is **6/16 = 37.5% attention**, and note the spacing is *non-uniform*: gaps of 3,3,3,2,2,2 —
attention gets **denser toward the top of the stack**, and the first two and the last layer are conv.

For LFM2-2.6B the newer `layer_types` list is given explicitly (30 layers), yielding attention at
indices 2, 5, 9, 13, 17, 21, 24, 27 → **8/30 = 26.7% attention** [FACT], again starting with 2 conv
layers and ending with 2 conv layers.

> **[INFER]** LFM2 is far *more* attention-heavy than the SSM-hybrid literature consensus (~7-10%,
> see §2). The premise of a "mostly-LIV" variant — pushing 37.5% down toward ~10-15% — is therefore
> a well-motivated, genuinely open question rather than a re-run of published work. Nobody has
> published a ratio ablation on the *short-conv* (LIV) mixer specifically; all published ratio
> ablations use SSM/linear-attention mixers, which have larger recurrent state than a K=3 conv.
> This is the clearest research gap I found and is the strongest justification for the experiment.

### 0.2 The LFM2 "LIV" / short-conv block, exactly [FACT — from modeling_lfm2.py]

`Lfm2ShortConv` is a **double-gated depthwise causal convolution**:

```python
self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=conv_L_cache,           # = 3
                      groups=hidden_size,                  # depthwise
                      bias=conv_bias, padding=conv_kernel_size - 1)
self.in_proj  = nn.Linear(hidden_size, 3 * hidden_size, bias=conv_bias)
self.out_proj = nn.Linear(hidden_size, hidden_size, bias=conv_bias)
```

forward:

```python
BCx = self.in_proj(hidden_states).transpose(-1, -2)
B, C, x = BCx.chunk(3, dim=-2)
hidden_states = B * x                       # multiplicative gate BEFORE the conv
hidden_states = causal_conv1d_fn(hidden_states, self.conv.weight.squeeze(1), self.conv.bias, seq_idx=seq_idx)
y = C * hidden_states                       # multiplicative gate AFTER the conv
y = self.out_proj(y.transpose(-1, -2).contiguous())
```

Facts worth noting for implementation:
- **Kernel size is 3.** Very short. Cheaper than Mamba/Mamba-2's K=4 conv and with no SSM scan at all.
- **No activation function** inside the conv path (unlike Mamba's `SiLU(conv(x))`); the nonlinearity
  is purely multiplicative gating. Contrast with Samba's short-conv which is `SiLU(depthwise conv K=4)`.
- **Three projections up (`3*d`), one down (`d`)** → parameter cost per LIV layer is `4*d^2`
  (no bias), identical to an MHA layer's `4*d^2` QKVO. [INFER] This is why LFM2 can swap conv↔attention
  layer-for-layer at near-constant parameter count — extremely convenient for a param-matched study.
  With GQA at `num_key_value_heads=8 < num_attention_heads`, an attention layer is actually *cheaper*
  than a LIV layer, so exact param matching needs a small width or FFN adjustment.
- **State at inference is `(d, K-1) = (d, 2)` per layer** — a 2-token cache. Tiny compared to
  Mamba-2's `d_state * d_head` per head, and tiny compared to a KV cache.
- The fused kernels used are `causal_conv1d_fn` / `causal_conv1d_update` from the
  `causal-conv1d` package (Tri Dao) — i.e. **the fused depthwise causal conv kernel this experiment
  needs already exists and is pip-installable**, and `seq_idx` is threaded through for
  document-boundary-aware convolution.
- `use_pos_enc: true`, `rope_theta = 1e6` — RoPE is applied **only in the attention layers**;
  conv layers carry position implicitly.
- LFM2-2.6B has `tie_embedding: true`. [UNKNOWN] whether the smaller three tie embeddings — the
  older configs do not carry the key. With vocab 65,536 and d=1024, an untied pair of embedding
  matrices is 2 × 67.1M = 134M params, i.e. **38% of LFM2-350M**. This is a first-order concern for
  the experiment (see §6.2).

### 0.3 What is *not* published about LFM2 [UNKNOWN — flag clearly]

- No LFM2 paper exists (only a blog post: `https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models`). There is **no published ratio ablation** for the conv:attention split, no
  published optimizer/LR/schedule, no published data mixture beyond "10T tokens", and no published
  intermediate checkpoints.
- The `full_attn_idxs = [2,5,8,10,12,14]` pattern is stated in config but **not justified anywhere**.
- License is **LFM Open License v1.0** ("license_name: lfm1.0", `license: other`) — *not* Apache/MIT.
  [INFER] This is fine for a research comparison (you are re-implementing an architecture, not
  redistributing weights), but do not ship their weights or tokenizer files without reading the license.

---

## 1. Recurrent / hybrid baseline architectures

### 1.1 Master comparison table

Attention fraction is "layers that are softmax/quadratic attention ÷ total sequence-mixing layers"
unless noted. "Ratio ablation?" means the paper published a sweep over the attention:linear ratio.

| Model | Year | Layer composition | Attn fraction | Attn type | Placement | Ratio ablation? | License |
|---|---|---|---|---|---|---|---|
| **Mamba** (2312.00752) | 2023 | pure selective-SSM, no attn, no MLP | **0%** | — | — | No | Apache-2.0 |
| **Mamba-2** (2405.21060) | 2024 | pure SSD; hybrid variants ablated | 0% pure; **~10% best** | full | spaced, not first/last | **YES — best in field** | Apache-2.0 |
| **Mamba-2-Hybrid** (2406.07887) | 2024 | 24 Mamba-2 + 4 attn + 28 MLP = 56 | **4/56 = 7.1%** | full GQA | evenly dispersed, Mamba first | **YES (130M + 840M)** | Apache-2.0 (Megatron-LM) |
| **Hawk** (2402.19427) | 2024 | pure RG-LRU recurrent + MLP | **0%** | — | — | No | [UNKNOWN] |
| **Griffin** (2402.19427) | 2024 | repeating (recurrent, recurrent, local-attn) | **~33%** | **local SWA, w=1024**, MQA | strict 1-in-3 interleave | No (window-size appendix only) | [UNKNOWN] |
| **Jamba** (2403.19887) | 2024 | blocks of l=8, a:m = 1:7, MoE every 2 | **1/8 = 12.5%** | full | 1 per 8-layer block (layers 4,12,20) | **YES (1:3 vs 1:7)** | Apache-2.0 |
| **Samba** (2406.07522) | 2024 | (Mamba, MLP, SWA, MLP) repeating | **~25%** of mixers | **SWA only, w=2048** | strict alternation | **YES (hybridization strategies)** | MIT (repo) |
| **Hymba** (2411.13676) | 2024 | attn + SSM heads **in parallel** in every layer | every layer has attn heads; **3 layers global** | mostly SWA + 3 global | global at **first, middle, last** | **YES (roadmap + local/global)** | NVIDIA OSS (see §1.7) |
| **Zamba / Zamba2** (2405.16712 / 2411.15242) | 2024 | Mamba backbone + **one shared** global attn block reused every ~6 layers | ~1 unique attn block | full, shared weights | periodic | Partial | Apache-2.0 |
| **RetNet** (2307.08621) | 2023 | pure retention + FFN | **0%** | — | — | No | MIT |
| **GLA** (2312.06635) | 2023 | pure gated linear attention + FFN | **0%** | — | — | No | MIT (fla) |
| **DeltaNet** (2406.06484) | 2024 | pure delta-rule linear attn | 0%; hybrids ablated | SWA in hybrids | interleaved | Partial | MIT (fla) |
| **Gated DeltaNet** (2412.06464) | 2024 | **H1** = GDN+SWA; **H2** = Mamba2+GDN+SWA | H1 ~50%, H2 ~33% of mixers | **SWA, w=2048** | strict repeating triple | **YES (order ablation, 500M/15B)** | MIT (fla) |
| **RWKV-6 / RWKV-7** | 2024/25 | pure RWKV time-mix + channel-mix | **0%** | — | — | No | Apache-2.0 |
| **Nemotron-H** (2504.03624) | 2025 | Mamba-2 + attn + FFN | **4/52 = ~7.7% (8B)** | full | evenly dispersed; Mamba first, FFN last | Follows 2406.07887 | NVIDIA Open Model License |
| **MiniMax-01** (2501.08313) | 2025 | lightning attn ×7 then 1 softmax | **1/8 = 12.5%** | full softmax | every 8th layer | **YES (scaling-law study)** | MiniMax model license |
| **Qwen3-Next** | 2025 | Gated DeltaNet ×3 : Gated Attention ×1 | **25%** | full gated attn | strict 3:1 | Not published | Apache-2.0 |
| **Kimi Linear / KDA** (2510.26692) | 2025 | KDA ×3 : full MLA ×1 | **25%** | full MLA, **NoPE** | strict 3:1 | **YES** | [see §1.9] |
| **Falcon-H1** (2507.22448) | 2025 | attention **‖ parallel** Mamba-2 in same block | parallel, not ratio | full | all layers (parallel) | **YES (channel-ratio)** | Falcon LLM license |
| **IBM Granite 4.0** | 2025 | Mamba-2 : attention ≈ 9:1 | **~10%** | full | periodic | Not published | Apache-2.0 |
| **LFM2** (no paper) | 2025 | 10 gated-short-conv + 6 GQA (16L) | **6/16 = 37.5%** | full GQA | idx 2,5,8,10,12,14 | **No** | LFM Open License v1.0 |

**[INFER] The single most striking row-to-row comparison:** every SSM/linear-attention hybrid that
published a ratio ablation converged on **7-25% attention**, while LFM2 — the only *short-conv*
hybrid — sits at **37.5%**. Two competing explanations, and distinguishing them is exactly what this
experiment can do:
1. A K=3 depthwise conv has a far smaller effective state than an SSM, so it needs *more* attention
   to compensate → 37.5% is genuinely necessary.
2. LFM2's ratio was chosen for CPU/NPU edge-inference engineering reasons (short conv is extremely
   cheap on CPU; they optimize for decode latency and cache size at 32K), not for loss-optimality
   → a mostly-LIV variant at 10-15% could match or beat it at fixed params/tokens.

### 1.2 Mamba (2312.00752) — the pure-SSM floor

[FACT] Architecture: selective SSM (S6), input-dependent Δ/B/C, hardware-aware scan.
Explicitly **"without attention or even MLP blocks"** — 0% attention.
Includes a **depthwise causal conv of kernel size 4** inside the block (relevant: this is the closest
published relative of the LIV mixer).

[FACT] Scaling-law model sizes (Table 12), taken from GPT-3 specs, trained on **The Pile** with the
**GPT-2 tokenizer**:

| Params | n_layers | d_model | n_heads/d_head | Training steps | LR | Batch size | Tokens |
|---|---|---|---|---|---|---|---|
| 125M | 12 | 768 | 12/64 | 4800 | 6e-4 | 0.5M tok | 2.5B |
| 350M | 24 | 1024 | 16/64 | 13500 | 3e-4 | 0.5M tok | 7B |
| 760M | 24 | 1536 | 16/96 | 29000 | 2.5e-4 | 0.5M tok | 15B |
| 1.3B | 24 | 2048 | 32/64 | 50000 | 2e-4 | 0.5M tok | 26B |

(This table is *identical* in Mamba-1 Table 12 and Mamba-2 Table 9 — the same scaling-law harness.)

[FACT] Training recipe, "improved recipe" (GPT-3 + PaLM/LLaMA modernizations):
- AdamW, β = (0.9, 0.95), **no dropout**
- **gradient clip 1.0**, **weight decay 0.1**
- linear LR warmup → **cosine decay to 1e-5**, with **peak LR = 5× the GPT-3 value**
- no linear bias terms; **RMSNorm** instead of LayerNorm
- Downstream models: 300B tokens on the Pile with the **GPT-NeoX tokenizer**; batch 1M tokens at 1.3B/2.7B.

> **[INFER] This is the single best small-scale recipe to copy for the experiment.** It is the
> recipe used by Mamba, Mamba-2, GLA, Gated DeltaNet, and (with minor edits) most of the linear-attention
> literature, so results become directly comparable to a large body of published numbers. Note the
> "5× GPT-3 LR" rule gives ~3e-3 at 125M — aggressive; see §5.

### 1.3 Mamba-2 (2405.21060) — **the cleanest published ratio ablation in the field**

[FACT] **Table 2 — "Combining SSD and Attention Blocks."** A **350M model with 48 layers**, trained to
**7B tokens on the Pile with the GPT-2 tokenizer**, *"same number of parameters, same hyperparameters,
same training and validation set"*:

| Num. attn blocks | 0 (Mamba-2) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 | 15 | 24 | Transformer++ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Perplexity ↓** | 8.60 | 8.38 | 8.32 | 8.29 | 8.29 | 8.28 | **8.26** | 8.27 | 8.28 | 8.30 | 8.34 | 8.50 | 8.68 |

Paper's conclusion, verbatim-ish: *"Having around a 10% ratio of attention layers performs best."*

**[INFER] Read the shape of this curve carefully — it is the most decision-relevant curve for this
experiment.** The optimum (6/48 = 12.5%) is worth **0.34 ppl vs. pure Mamba-2** and **0.42 ppl vs.
Transformer++**, but the basin is extremely flat: **anything from 2 to 11 attention blocks (4%-23%)
lands within 0.06 ppl of the optimum**, which is within typical seed noise. The strong, robust claims
are (a) *some* attention beats none (0.22 ppl from the very first attention layer), and (b) *too much*
attention is clearly bad (24/48 = 8.50, worse than 6/48 by 0.24). Any experiment claiming to resolve
the optimum *within* 4-23% needs multiple seeds and a stated noise floor, or it will be measuring nothing.

[FACT] Placement, from footnote 6: *"as long as the attention layers are spaced out, not at the very
beginning or at the very end, the model quality does not depend very much on the exact location of the
attention layers."* — small-scale experiments.

[FACT] The 2.7B / 64-layer / 300B-token comparison (same params, same hyperparameters, same
validation set, **same data order**) used these five configurations:
1. Transformer++: 32 attn + 32 gated MLP, interleaved
2. Mamba-2: 64 SSD layers
3. Mamba-2-MLP: 32 SSD + 32 gated MLP, interleaved
4. **Mamba-2-Attention: 58 SSD + 6 attn at indices 9, 18, 27, 36, 45, 56**
5. Mamba-2-MLP-Attention: 28 SSD + 4 attn interleaved with 32 gated MLP

[FACT] Findings: Transformer++ ≈ Mamba-2; adding 6 attention layers noticeably improves over both;
adding MLP layers *reduces* quality but speeds up training/inference and eases MoE up-cycling.

> **[INFER] Configuration list (1)-(5) above is a near-perfect template for this experiment's
> baseline set** — it is already parameter-, token-, hyperparameter-, and data-order-matched, and
> published. Swapping "SSD" for "LIV short-conv" reproduces the study in the conv regime.

### 1.4 Mamba-2-Hybrid / Waleffe et al. (2406.07887) — **the ratio finding the task asked about**

**The headline recipe [FACT]: 43% Mamba-2, 7% attention, 50% MLP.**

[FACT] Exact 8B config: **56 layers**, 8.66B params, d_model 4096, 32 attn heads, 8 GQA groups,
state dim 128, **no position embeddings**, seq len 4096. Breakdown: **4 (7.1%) self-attention, 24
(42.9%) Mamba-2, 28 (50%) MLP**. Published pattern (M=Mamba-2, `*`=attention, `+`=MLP):

```
M+M+M++M+M*+M+M+M+M++M*+M+M+M+M+M*++M+M+M+M+M*+M++M+M+M+
```

**Why this ratio [FACT]:**
- **Number of attention layers:** Figure 4 sweeps attention percentage in **130M-param, 24-layer**
  hybrids; validation loss is *"minimized when roughly 8% of the layers are self-attention layers"*,
  confirmed again at **840M** scale, and *"consistent with Dao et al. 2024"* (i.e. the Mamba-2 Table 2
  ~10% result — **two independent labs converge**).
- **MLP fraction:** holding attention at 8% and sweeping MLP 5%→50%: *"30%-50% of the layers can be
  MLPs without increasing model loss"*, and 50% MLP **trains 20% faster** than 5% MLP. So 50% MLP is
  a free speedup. **[INFER] This is the most actionable and least-known finding of the paper** — it
  says the MLP fraction is nearly free to choose, so choose the fast one.
- **Placement:** Algorithm 1 (Appendix A) spaces attention so intervening Mamba runs are near-equal,
  begins and ends with Mamba runs, then distributes MLPs biased away from the start so the stack ends
  with an MLP. They *"found no significantly better configuration than to evenly distribute
  self-attention and MLP layers"*, and a repeated block pattern was unnecessary.
- **Mamba first** so no position embeddings are needed (the SSM encodes position).
- **GQA:** ablated — GQA instead of MHA costs only **~0.04% validation perplexity**, so GQA adopted.
- **RoPE: skip it.** Table 5 (840M, 8% attention, 1.1T tokens): at 4K, no-RoPE avg **52.19** vs
  RoPE(θ=10K) 52.39; at 16K, no-RoPE **53.43** vs RoPE-10K 52.61 vs RoPE-500K 51.52 → no RoPE wins at
  16K. **[INFER] Notable and counterintuitive:** in a hybrid with only 7% attention, the recurrent
  layers supply position well enough that RoPE actively *hurts* long-context. Directly relevant to §7.
- Sliding-window attention was **not** ablated; all 4 attention layers are **full global**, including
  in the 128K extension.

[FACT] Hyperparameters (identical across all architectures compared):
- 1.1T-token runs: **batch 256**, peak LR **1e-4**, min LR 1e-5
- 3.5T-token runs: **batch 1024**, peak LR **3e-4**, min LR 3e-5
- Both: LR warmup over 122K samples, **cosine decay**, **weight decay 0.1**, Adam β=(0.9, 0.95), BF16
- Seq len 4096; long-context extension adds **50B tokens** with max LR 3e-5 → min 3e-6
- Tokenizer: **SentencePiece, 256K vocab**; data 70% English / 15% non-English / 15% code

[FACT] Results: 8B hybrid beats the 8B Transformer on **all 12** standard tasks (**+2.65 avg**); pure
SSMs lag on copying / in-context learning (5-shot MMLU, Phonebook); hybrid matches or exceeds the
Transformer across 23 long-context tasks; up to 8× faster generation.

### 1.5 Griffin / Hawk (2402.19427)

[FACT] **Hawk** = pure recurrent: every block is (RG-LRU recurrent block + MLP), 0% attention.
**Griffin** = *"alternating two residual blocks with a recurrent block followed by one residual block
which uses the local (MQA) attention block"* → **repeating 2 recurrent : 1 local attention (~33% attention)**.

[FACT] Recurrent block internals — **structurally very close to the LIV block**: input dim D → two
parallel linear layers of width D_RNN; branch one gets a **separable Conv1D of temporal filter
dimension 4** (only `4·D_RNN` params) then the RG-LRU; branch two gets GeLU; **branches multiplied
elementwise**; final linear back to D. D_RNN ≈ 4D/3 to match MHA param count.

[FACT] RG-LRU: `a = σ(Λ)`, `a_t = a^(c·r_t)` with **c = 8**, `h_t = a_t⊙h_{t−1} + sqrt(1−a_t²)⊙(i_t⊙x_t)`.
Gates depend only on `x_t`, **not** on `h_{t−1}` (so it is parallelizable). Real-valued — complex
recurrences *"were not beneficial for language modelling in practice"*. Λ init so `a^c` is uniform in
**[0.9, 0.999]**.

[FACT] Attention: **local sliding-window MQA, window fixed at 1024 tokens**, RoPE inside the window,
**no global attention layers at all**.

[FACT] Model sizes (Appendix C, Table 2) — sub-1B rows are directly usable:

| Size | D | D_RNN | Depth | MLP exp. | Heads | Chinchilla-optimal tokens |
|---|---|---|---|---|---|---|
| **100M** | 768 | 1024 | 12 | 3 | 6 | **1.9B** |
| **200M** | 1024 | 1536 | 12 | 3 | 8 | **3.9B** |
| **400M** | 1536 | 2048 | 12 | 3 | 12 | **7.8B** |
| 1.3B | 2048 | 2560 | 24 | 3 | 16 | 25B |
| 3B | 3072 | 4096 | 24 | 3 | 24 | 60B |
| 7B | 4096 | 5632 | 32 | 3 | 32 | 132.5B |
| 14B | 5120 | 8192 | 40 | 3 | 40 | 300B |

Attention head dim fixed at 128. Downstream-eval models all **overtrained to 300B tokens**.
Seq len **2048** (a 1B long-context run used 8192 with batch cut 4× to hold tokens fixed).
Data: **MassiveText** (not public). AdamW; LR/WD/β₂ tuned small then extrapolated by fitted scaling rules.

[UNKNOWN] Exact LR values, schedule shape, batch sizes, tokenizer, vocab size, repo, license — none
stated. Griffin/Hawk are **not reproducible from the paper**, and there is no official code release.
> **[INFER] Recommendation: do not use Griffin as a primary baseline.** Use it as a *design* citation
> for the 2:1 local-attention pattern, and if you want a Griffin-like arm, implement RG-LRU from `fla`
> (which has an HGRN/RG-LRU-family implementation) and label it "Griffin-style", not "Griffin".

[FACT] No recurrent:attention ratio ablation. Appendix E is titled "The Local Attention Window Size of
Griffin". For synthetic tasks they used a **single local-attention layer in the middle (third) block of
a 5-block model** — an implicit placement choice.

### 1.6 Jamba (2403.19887) — ratio ablation at 1.3B and 7B

[FACT] Released Jamba: 4 blocks, each **l = 8 layers, a:m = 1:7** (1 attention per 7 Mamba),
**e = 2** (MoE replaces MLP every other layer), **n = 16 experts, top-K = 2**. 12B active / 52B total,
256K context, fits one 80GB GPU in int8. **Apache-2.0** (`huggingface.co/ai21labs/Jamba-v0.1`).
1:7 chosen as *"the most compute-efficient variant amongst the best performing variants in terms of quality."*

[FACT] **Table 4 — 1.3B params, 250B tokens** (the sub-1B-adjacent scale most relevant here):

| Model | HellaSwag | WinoGrande | OLLM | NQ | log-prob C4 | Books | Code |
|---|---|---|---|---|---|---|---|
| Attention | 36.4 | 62.4 | 59.6 | 14.5 | −0.543 | −0.659 | −0.331 |
| Mamba | 36.1 | 62.6 | 59.4 | 14.5 | −0.543 | −0.661 | −0.334 |
| **Jamba 1:3** (no MoE) | 37.2 | 65.1 | 61.7 | **16.5** | **−0.533** | −0.649 | −0.321 |
| **Jamba 1:7** (no MoE) | 37.2 | 65.1 | 61.7 | 16.0 | **−0.533** | −0.650 | −0.321 |

**[FACT] 1:3 (25%) and 1:7 (12.5%) are essentially indistinguishable**; both beat pure Attention and
pure Mamba. Training-loss curves show hybrids below both pure models throughout, with **no visible gap
between the two ratios**. → 1:7 chosen purely because it is cheaper.

[FACT] Table 5 — 7B params, 50B tokens: Attention 36.1/60.4/59.7/13.7; Mamba 35.3/60.2/55.8/14.0;
Jamba 1:7 no-MoE 36.6/62.5/58.8/15.4. Pure Mamba *"quite competitive, but lags slightly behind pure Attention."*
Adding MoE at 7B/50B lifts Jamba to 38.1/66.0/61.2/18.9.

[FACT] **Positional encoding not needed:** at 1.3B/250B, no-PE vs +RoPE gave near-identical results
(log-probs identical at −0.516/−0.623/−0.299). Authors attribute this to Mamba layers preceding
attention supplying position implicitly. **[INFER] Third independent confirmation of the NoPE finding
(with Waleffe and Kimi Linear) — this is now a well-supported hybrid design rule.**

[FACT] **Stability:** training was smooth to 1.3B, but the 12B-active/52B model hit large loss spikes
traced to oversized activations inside Mamba layers; **RMSNorm on internal Mamba activations** fixed it.
**[INFER] At ≤1B this is unlikely to bite, so it should not drive design here — but if a run diverges,
this is the first thing to check.**

[FACT] **Why attention is needed (the ICL/induction-head argument):** at 1.3B/250B, pure Mamba
collapsed on format-following — IMDB 48.8 vs attention 84.1, QuAC 20.2 vs 27.9, NarrativeQA 27.7 vs 45.8
— emitting "Very Good"/"Funny"/"3/10" instead of Positive/Negative. The hybrid tracked attention
(90.9/26.6/43.7) and *"does ICL successfully even when only 1 out of 8 layers is an Attention one."*
They located **12 induction-like heads spread across all three attention layers (layers 4, 12, 20)**.

> **[INFER] This is the strongest evidence in the literature for *why* the ratio can be so low:
> induction-head circuits are cheap in layer count but apparently cannot be built out of recurrent
> layers alone. It also suggests the right *metric* for this experiment is not perplexity alone —
> perplexity differences across 4-25% attention are within noise (§1.3), while format-following /
> recall / induction metrics show 30+ point gaps. Design the eval suite around recall, not ppl.**

### 1.7 Hymba (2411.13676) — parallel heads, and the "first/middle/last" placement result

[FACT] Hymba fuses attention heads and SSM (Mamba) heads **in parallel inside the same layer**, rather
than interleaving layer-by-layer. Plus **128 learnable meta tokens** prepended to prompts, **cross-layer
KV sharing** (every two layers share a KV cache), and **partial sliding-window attention**.

[FACT] **Global attention in just three layers — the first, middle, and last** — *"is sufficient to
recover recall-intensive accuracy while maintaining comparable commonsense reasoning accuracy"*,
giving **2.7× throughput and 3.8× cache reduction**. Replacing global attention in *all* layers with
SWA first cost **>20% accuracy on recall-intensive tasks**, which is what motivated reinstating a few.

[FACT] **Design roadmap (Table 1) — ablations at 300M params / 100B tokens.** Columns: commonsense
accuracy (avg 8 tasks) / recall accuracy (avg 2 tasks) / throughput tok/s / cache MB:

| Config | Commonsense | Recall | Throughput | Cache (MB) |
|---|---|---|---|---|
| Transformer (Llama) | 44.08 | **39.98** | 721.1 | 414.7 |
| State Space Models (Mamba) | 42.98 | 19.23 | 4720.8 | **1.9** |
| A. + Attention heads (sequential) | 44.07 | 45.16 | 776.3 | 156.3 |
| B. + Multi-head structure (**parallel**) | **45.19** | 49.90 | 876.7 | 148.2 |
| C. + Local/global attention | 44.56 | 48.79 | 2399.7 | 41.2 |
| D. + KV cache sharing | 45.16 | 48.04 | 2756.5 | 39.4 |
| E. + Meta tokens | **45.59** | **51.79** | 2695.8 | 40.0 |
| F. + Size/data (1.5B, 1.5T tok) | 60.56 | 64.15 | 664.1 | 78.6 |
| G. + Extended context 2K→8K | 60.64 | 68.79 | 664.1 | 78.6 |

**[FACT] Two results here are directly load-bearing for this experiment:**
1. **Pure Mamba's recall is catastrophic: 19.23 vs Transformer 39.98** — a 20.75-point gap, whereas
   commonsense differs by only 1.10 points. **Recall is where architecture shows up; perplexity and
   commonsense hide it.**
2. **Parallel fusion (B) beats sequential interleaving (A)** on both metrics (+1.12 commonsense,
   +4.74 recall) at 300M/100B. **[INFER] This is a genuine threat to the layer-interleaved framing of
   the whole experiment** — it suggests the *within-layer* parallel topology may dominate the
   *between-layer* ratio question. Worth one arm of the sweep, and worth citing as a limitation.

[FACT] Hymba-1.5B-Base: 1.5T tokens, surpasses all sub-2B public models; vs Llama-3.2-3B it is
**+1.32% avg accuracy, 11.67× smaller cache, 3.49× throughput**. Models at
`huggingface.co/nvidia/Hymba-1.5B-Base`. Paper is CC BY 4.0; ablation models 300M/100B and 350M also mentioned.
[UNKNOWN] Exact optimizer/LR/schedule/batch for Hymba pretraining; the paper's roadmap table does not
give them. [UNKNOWN] Precise per-layer head counts and SWA window size were not recoverable from the
sections I extracted. Weight license is NVIDIA's own open-model license, **not** Apache — check before reuse.

### 1.8 Samba (2406.07522) — the best small-scale, short-window, MIT-licensed reference

[FACT] Layer pattern: **Mamba → MLP → SWA → MLP**, repeating. Attention is **sliding-window only
(w = 2048), never full**, with RoPE applied inside the window, FlashAttention-2. Separate MLPs for the
Mamba and SWA outputs. So **~25% of layers are (windowed) attention** counting MLPs, or ~50% of
sequence-mixing layers.

[FACT] **Table 3 — SlimPajama validation perplexity, trained at 4K context, window 2048.**
This is the most directly comparable small-scale controlled comparison in the literature:

*438M-class models, 20B tokens, 8×A100:*

| Arch | Size | Layers | Speed (×10⁵ tok/s) | ppl@4096 | ppl@8192 | ppl@16384 |
|---|---|---|---|---|---|---|
| Llama-2 (full attn) | 438M | 24 | 4.85 | 11.14 | 47.23 | 249.03 |
| Llama-2-SWA | 438M | 24 | 4.96 | 11.12 | 10.66 | 10.57 |
| Mamba | 432M | 60 | 2.46 | 10.70 | 10.30 | 10.24 |
| Sliding GLA | 438M | 24 | 4.94 | 10.43 | 10.00 | 9.92 |
| Sliding RetNet | 438M | 24 | 4.32 | 10.38 | 9.96 | 9.87 |
| **Samba** | 421M | 24 | 4.46 | **10.06** | **9.65** | **9.57** |

*1.3B-class, 100B tokens, 64×H100:*

| Arch | Size | Layers | ppl@4096 | ppl@8192 | ppl@16384 |
|---|---|---|---|---|---|
| Llama-2 | 1.3B | 40 | 7.60 | 44.32 | 249.64 |
| Llama-2-SWA | 1.3B | 40 | 7.60 | 7.37 | 7.21 |
| Mamba | 1.3B | 48 | 7.47 | 7.26 | 7.15 |
| Sliding GLA | 1.2B | 36 | 7.58 | 7.35 | 7.19 |
| Sliding RetNet | 1.4B | 36 | 7.56 | 7.35 | 7.56 |
| **Samba** | 1.3B | 36 | **7.32** | **7.11** | **6.96** |

Reported run-to-run fluctuation: **±0.3%**. **[INFER] At ppl ≈ 10, ±0.3% ≈ ±0.03 ppl. Compare to the
Mamba-2 Table 2 basin width of 0.06 ppl across 4-23% attention — i.e. the ratio-optimum question sits
only ~2× above this noise floor. Multiple seeds are mandatory.**

[FACT] **Table 6 — full attention in a Mamba-MLP hybrid (12 blocks, 20B tokens SlimPajama), ppl at
4K/8K/16K:** attention at block 11 → 10.29/10.53/13.66; block 5 → 10.10/10.05/12.83; block 0 →
10.89/10.55/**10.63**; blocks 1&5 (443M) → 10.06/10.34/13.57; Samba (SWA) → 10.06/9.65/**9.57**.
Conclusion: **even one full-attention layer causes exploding perplexity beyond the training length**
— *unless* it is at block 0, which extrapolates but is worse in-distribution.

> **[INFER] This is the most important placement result in the literature and it directly contradicts
> the "placement doesn't matter" reading of Mamba-2/Waleffe.** Reconciliation: placement doesn't matter
> much *in-distribution* (Mamba-2 footnote 6, Waleffe Alg. 1), but it matters enormously *for length
> extrapolation* (Samba Table 6). If this experiment makes long-context claims at 16K/32K while
> training shorter, placement must be an explicit variable, not a fixed detail. LFM2 places its first
> attention at index 2 (not 0), and uses RoPE θ=1e6 with 32K training — consistent with "train long
> natively rather than rely on extrapolation."

[FACT] Model configs (Table 10):

| | 421M | 1.3B | 1.7B | 3.8B |
|---|---|---|---|---|
| Data | SlimPajama | SlimPajama | Phi-2 | Phi-3 |
| Tokens | 20B | 100B | 230B | 3.2T |
| Layers | 24 | 36 | 48 | 64 |
| d_model | 1536 | 2304 | 2048 | 2816 |
| MLP dim | 4096 | 6144 | 8196 | 9984 |
| Query heads | 12 | 18 | 32 | 11 |
| KV heads | 12 | 18 | 4 | 1 |
| Vocab | 32000 | 32000 | 50304 | 32064 |

[FACT] Hyperparameters: **batch 512** (421M/1.3B) or 2048 (1.7B/3.8B); **LR 4e-4** (small) / 6e-4 (large);
**weight decay 0.1**; **gradient clipping 1.0**; **seq len 4096**; window 2048.
[UNKNOWN] LR decay schedule shape not specified in the extracted text.

[FACT] **Short Convolution transfer ablation (Table 8, 438M, SlimPajama)** — highly relevant, this is
the LIV mixer being bolted onto other architectures. SC = depthwise conv **kernel size 4 + SiLU**:

| Arch | baseline (4K/8K/16K) | + Short Conv |
|---|---|---|
| Llama-2-SWA | 11.12/10.66/10.57 | **10.83/10.39/10.31** |
| Sliding GLA | 10.43/10.00/9.92 | 10.39/9.96/9.87 |
| Sliding RetNet | 10.38/9.96/9.87 | **10.25/9.82/9.74** |

**[FACT] Critical caveat, verbatim intent:** adding SC to **both** the SWA and the linear-attention
layers in hybrids *"produces negative results."*
> **[INFER] Direct warning for this experiment:** a short conv gives a real, free gain (0.26-0.29 ppl
> on SWA) when applied to *one* mixer type, but stacking short convs everywhere hurts. A "mostly-LIV"
> stack is by construction short-conv-everywhere. This is published evidence that the mostly-LIV
> direction could underperform, and it should be pre-registered as the main risk / most likely
> negative result.

[FACT] **Sequence-length vs window ablation (Table 5, Llama-2-SWA 438M, window fixed 2048, ~2M tok/step):**
full attention @2048 → 11.59 then blows up (38.12/156.18/357.32). With SWA: train@4096 →
11.87/11.16/10.69/10.61; @8192 → 11.98/11.26/10.79/10.69; @16384 → 12.37/11.63/11.12/11.02;
@32768 → 12.94/12.46/11.96/11.86. **Longer training sequences (at fixed tokens/step, hence smaller
batch) hurt at every eval length; best sequence-length:window ratio is 2, i.e. train at 4096.**
> **[INFER] Strong, quantified argument for short-then-extend over native-long training at small
> scale** (§7): training natively at 32K cost **+1.07 ppl at 4K eval and +1.25 at 32K eval** vs
> training at 4K, at equal tokens. Part of this is the batch-size confound, but the direction is clear.

[FACT] Head-count ablation (Table 7, ~430M): **one KV head is best** for both Llama-2-SWA (12Q/1KV →
10.89/10.44/10.35) and Samba (6Q/1KV → 9.99/9.59/9.51); **Samba's optimal query-head count is half
that of the pure-SWA model.** **[INFER] So aggressive GQA/MQA is not just an inference optimization —
at 430M it is *quality-optimal*, and hybrids want fewer query heads than pure-attention models. Worth
holding head config fixed across arms, or you will confound ratio with head count.**

[FACT] Repo `https://github.com/microsoft/Samba`. [FACT] Table 2 downstream (1.6-1.9B, 230B Phi-2
tokens, 15 benchmarks) averages: Llama-3 51.17, Mistral 51.12, Mamba 52.31, Mamba-SWA-MLP 53.77,
Mamba-MLP 51.38, **Samba 54.33**.

### 1.9 Gated DeltaNet (2412.06464) — hybrid *ordering* ablation, and a clean 400M/1.3B protocol

[FACT] Base architecture: Llama macro-architecture (token mixer + SwiGLU MLP), self-attention replaced
by the gated delta rule. Block design: q/k/v paths = linear proj → **short conv** → SiLU, with **L2 norm
on q,k**; α/β from linear proj only; output → norm → gate → out-proj. Head dim **128** found optimal.

[FACT] Hybrids: **GatedDeltaNet-H1 = GDN + SWA**; **GatedDeltaNet-H2 = Mamba2 + GDN + SWA** (repeating).

[FACT] **Protocol — copy this one:** all models trained under identical conditions, **1.3B params on
100B tokens sampled from FineWeb-Edu**; AdamW, **peak LR 4e-4**, **weight decay 0.1**, **grad clip 1.0**,
**cosine annealing with 1B-token warmup**, **batch 0.5M tokens**, **Llama-2 tokenizer, vocab 32,000**,
**training length 4K**, **SWA window 2K**. Also a 400M scale. Baselines: RetNet, HGRN2, Mamba, Mamba2,
Samba, DeltaNet, Transformer++.

[FACT] Table 3 (1.3B) selected — wiki ppl / LMB ppl / avg acc:
Transformer++ 18.53/18.32/52.25; Samba 16.13/13.29/54.00; **GDN-H1 16.07/12.12/56.40**;
**GDN-H2 15.91/12.55/56.18**. Both hybrids beat Transformer++ by ~4 points average accuracy.

[FACT] **Table S.2 — hybrid ordering ablation, 500M params / 15B tokens, FineWeb-Edu, Llama tokenizer.**
Four orderings of the same three mixers:

| Ordering | (col A) | (col B) | (col C) | (col D) | (col E) | (col F) |
|---|---|---|---|---|---|---|
| GDN + SWA + Mamba2 | 34.77 | 67.08 | 40.84 | 50.74 | 38.94 | 61.49 |
| GDN + Mamba2 + SWA | 36.17 | 67.51 | 41.51 | 51.85 | 38.58 | 53.73 |
| Mamba2 + SWA + GDN | 36.79 | 64.96 | 41.18 | 52.01 | 38.07 | 59.44 |
| **Mamba2 + GDN + SWA** | **36.92** | 66.48 | **41.70** | **52.72** | **39.91** | 60.51 |

Paper's conclusion: *"the combination of Mamba2, Gated DeltaNet, and SWA in this specific order
produces superior results."* Also: short conv and output gate are **crucial**; output norm marginal;
L2 norm essential; SiLU best activation.

> **[INFER] Note how small these ordering differences are (36.92 vs 36.17 vs 36.79 on col A) relative
> to a 15B-token budget. "Superior" here is ~0.1-0.8 points, single-seed. Treat published ordering
> preferences as weak priors, not settled facts. But note the *winning* order again ends with SWA and
> starts with the biggest-state mixer — consistent with "big-state mixer first, attention later."**
> **[INFER] The 500M/15B ablation scale is an excellent template for this experiment's own sweep
> scale** — big enough to rank architectures, small enough to run many variants.

### 1.10 RWKV-6/7, RetNet, GLA — the pure-linear floor arms

- **RWKV-6 "Eagle & Finch"** (arXiv **2404.05892**) and **RWKV-7 "Goose"** (arXiv **2503.14456**):
  pure RWKV time-mix + channel-mix, **0% softmax attention**. Apache-2.0. RWKV-7 is in `fla` (`rwkv7.py`).
  **[FACT] RWKV-6** introduced the **RWKV World Tokenizer** (greedy-matching, covering
  underrepresented languages) and the **RWKV World v2 corpus, 1.12T tokens**.
  **[FACT] RWKV-7 released seven Apache-2.0 models in two families — and the Pile family is *ideal*
  as a sanity reference here:** *"Trained on Pile: RWKV7-Pile of sizes 0.1B, 0.4B, and 1.4B"* using the
  **GPT-NeoX-20B tokenizer**, *"trained from scratch on the Pile dataset, which has 332 billion
  tokens"*; plus *"RWKV7-World-3 of sizes 0.1B, 0.4B, 1.5B, and 2.9B"* on the **RWKV World v3 corpus
  (3.119T tokens)**. Important caveat the paper states plainly: *"Due to compute budget constraints, the
  Goose World 3 0.1B and 0.4B models were trained from pre-existing RWKV-5 World v1 and v2
  checkpoints"* and the 1.5B/2.9B from RWKV-6 checkpoints, converted to RWKV-7 format — *"some
  documents were seen two or even three times."*
  **[INFER] So the RWKV7-**Pile** models (0.1B/0.4B/1.4B, from scratch, GPT-NeoX tokenizer, 332B
  tokens) are clean comparables, whereas the World-3 models are *not* trained from scratch and should
  not be used as controlled baselines.**
  [UNKNOWN] I did not extract the per-size layer/d_model tables (Appendix E) or benchmark numbers.
- **RetNet** (arXiv **2307.08621**): pure retention + FFN, **0% attention**; multi-scale exponential
  decay; parallel / recurrent / **chunkwise-recurrent** forms. MIT via `torchscale` and `fla`.
  [FACT, from Samba Table 3] "Sliding RetNet" at 438M/20B gets ppl 10.38 → **RetNet-family mixers are
  perfectly serviceable at this scale**, and Samba's numbers give a ready-made comparison point.
  [INFER] Widely reported as harder to reproduce than GLA; GLA/Gated-DeltaNet superseded it in the
  `fla` line of work.
- **GLA** (arXiv **2312.06635**, "Gated Linear Attention Transformers with Hardware-Efficient Training"):
  pure gated linear attention + FFN (*"multi-head GLA layers with feed-forward networks (FFN)"*),
  **0% attention**, data-dependent gating, chunkwise parallel form. Origin of the
  **`flash-linear-attention` (fla)** library. [FACT, from Samba Table 3] "Sliding GLA" 438M/20B →
  10.43 ppl.
  **[FACT] GLA's own protocol — another clean, copyable small-scale recipe:** two scales, **340M and
  1.3B**, trained on **15B and 100B tokens** respectively, on *"the same subset of the SlimPajama
  dataset"* (627B original, 100B subset used) tokenized with the **Mistral tokenizer**; peak
  **LR 3e-4**, **cosine schedule** with warmup of **0.5B / 1B tokens**, *"initial and final learning
  rates are 3e-5"*; **batch 0.5M tokens (340M)** and **2M tokens (1.3B)**; all baselines *"trained for
  the exact same number of tokens on the same [data]"*. Transformer++ at 340M/15B gives ppl **28.39**.
  They also evaluate **MQAR-style recall** (*"a model has to recall the token following a query token
  multiple times"*, following Arora et al.) and **length extrapolation on SlimPajama and PG19**.
  **[INFER] Note the 340M/15B point (44 tok/param, batch 0.5M, LR 3e-4, cosine) is nearly identical to
  what §9 recommends — convergent evidence that this recipe family is the right one at this scale.**

> **[INFER] Practical consequence:** for this experiment, RWKV-7 / GLA / Gated-DeltaNet arms are
> essentially *free* to add if the codebase is `fla`-based, because they are already implemented and
> kernel-optimized there. That is a major argument for the codebase choice in §3.

### 1.11 2025-2026 production hybrids

**Nemotron-H (arXiv 2504.03624)** [FACT]: *"we set the number of attention layers to be roughly 8% of
the total number of layers and evenly disperse them throughout the model. This amounts to 4
self-attention layers (out of 52 layers)"* for Nemotron-H-8B, with the remainder an **even split
between FFN and Mamba-2 layers**. Explicit constraints: **(a) first layer is Mamba-2, (b) last layer
is FFN, (c) self-attention layers always** [followed by FFN — per §2.1 of that paper]. **No position
embeddings.** Nemotron-H-56B was the first Nemotron fully pretrained with an **FP8 recipe** (E4M3 for
weights/activations, E5M2 for gradients, per-tensor dynamic scaling, **first 4 and last 4 layers kept
in BF16**), reaching *"equal or better downstream task accuracy compared to BF16"*. Nemotron-H-47B is
compressed from 56B *"using only 63 billion training tokens"*. License: NVIDIA Open Model License.

> **[INFER] Nemotron-H is essentially Waleffe et al.'s recipe productionized at 8B-56B: same ~8%
> attention, same even dispersion, same Mamba-first/FFN-last, same NoPE. That is a strong vote of
> confidence in the 8% number from a lab that could afford to test alternatives.**

**MiniMax-01 (arXiv 2501.08313)** [FACT]: hybrid of **lightning attention** (I/O-aware linear
attention) with **softmax attention layers substituted at intervals of every eight layers → 1/8 =
12.5%**. 456B total / 45.9B active MoE, 1M-token context.

**[FACT] Its scaling-law study is unusually well-matched to this experiment's scale:** softmax
(FlashAttention-2), pure lightning attention, and hybrid-lightning were each trained *"across various
scales: 70 million, 160 million, 410 million, 1 billion, 3 billion, and 7 billion parameters. Each
model was trained on a dataset consisting of up to 300 billion tokens, with a context length of 8192."*
They also benchmarked **three hybrid-linear variants at 1B parameters** (Table 3) — hybrid-cosformer2,
hybrid-hgrn2, and hybrid-lightning, all using the same *"every eight layers"* substitution — and
*"the hybrid-lightning model achieves the best performance"*, evaluated on CSR average, NIAH weighted
accuracy, and SCROLLS. Table 4 separately compares hybrid-lightning to **hybrid-window** (SWA)
variants, finding *"larger window sizes lead to slower training speeds"* and that hybrid-lightning
*"outperforms all other models"*. They note pure linear attention *"falls short in retrieval tasks,
rendering it unsuitable for LLMs"*, while *"lightning attention possesses a larger capacity than
softmax attention"*.

> **[INFER] The 70M-7B × 300B-token, three-architecture design is the closest published analogue to
> what this experiment proposes, just with lightning attention instead of LIV conv — worth citing as
> methodological precedent for the scale ladder. And their retrieval finding is yet another instance of
> the §2.2 pattern: the pure-linear model's failure shows up in *retrieval*, not perplexity.**

**Qwen3-Next** [FACT, from model card/blog]: **Gated DeltaNet : Gated Attention = 3:1 → 25% attention**.
Qwen3-Next-80B-A3B, Apache-2.0. **No public ratio ablation.**

**Kimi Linear / KDA** (arXiv **2510.26692**) [FACT]: **Kimi Delta Attention : full MLA = 3:1 → 25%
attention**, with the full-attention layers using **NoPE** (no positional encoding) — position handled
by the KDA layers. 48B total / ~3B active. This repo's own `KDA/` directory concerns this line of work.
[UNKNOWN] Exact small-scale ablation token budgets not verified in this pass.

**Falcon-H1 (arXiv 2507.22448)** — **a genuine third ratio ablation, in the channel domain.**
[FACT] A **parallel** hybrid: attention and Mamba-2 run side-by-side within each block and their
outputs are **concatenated** before the block output projection (Figure 1), *"Attention and SSM run in
parallel within each block... The number of SSM/Attention heads can be flexibly tuned."* Uses **μP with
tunable multipliers** (§3.2.3) and context/mixer parallelism. Sizes and configs (Table 1; **embedding
and projection layers are untied for all models**):

| Model | Params (B) | Layers | Vocab | d_model | Heads (Q/KV, SSM) | d_head (attn/SSM) | d_state | Context | Tokens |
|---|---|---|---|---|---|---|---|---|---|
| Falcon-H1-0.5B | 0.52 | 36 | **32,778** | 1024 | 8/2, 24 | 64/64 | 128 | 16K | 2.5T |
| Falcon-H1-1.5B | 1.55 | 24 | 65,536 | 2048 | 8/2, 48 | 128/64 | 256 | 128K | 3T |
| Falcon-H1-1.5B-deep | 1.55 | **66** | 65,536 | 1280 | 6/2, 24 | 128/64 | 256 | 128K | 3T |
| Falcon-H1-3B | 3.15 | 32 | 65,536 | 2560 | 10/2, 32 | 128/128 | 256 | 128K | 2.5T |
| Falcon-H1-7B | 7.59 | 44 | 130,048 | 3072 | 12/2, 24 | 128/128 | 256 | 256K | ~12T |
| Falcon-H1-34B | 33.6 | 72 | 261,120 | 5120 | 20/4, 32 | 128/128 | 256 | 256K | ~18T |

**[FACT] The channel-allocation ablation — ablation scale is directly relevant here:** run on
*"a relatively deep model with L = 60 layers, hidden dimension d = 1280, resulting in approximately
1.2B parameters. All other training and architecture hyperparameters were identical, and we measured
the loss after 70GT of training."* Channels are split into 8 chunks across SSM/attention/MLP as
`d_ssm = α_S×4096`, `d_attn = α_A×6144`, `d_MLP = α_M×4864` with `α ∈ {1/8..6/8}`, `α_S+α_A+α_M = 1`
— base ratio `4096:6144:4864 = 2:3:2.375` chosen so parameter count is (near-)constant across
allocations. All **21 admissible partitions** were run.

**Result, verbatim:** *"We see a clear separation of magnitude between the impact of the number of
attention channels and SSM ↔ MLP channel switching. **Having more attention channels significantly
degrades the performance**, while SSM ↔ MLP channel switching has a noticeable but much weaker
effect."* The optimum fixes attention at the **minimum tested, α_A = 1/8**, and the final adopted
allocation is **(α_S, α_A, α_M) = (2/8, 1/8, 5/8)**, i.e. a **2:1:5** SSM:attention:MLP channel ratio,
*"with slight deviation for different model sizes, which is possible thanks to flat dependence on
SSM/MLP allocations near optimum."* Loss values on the plotted axes span ~2.56-2.65 (left panel) and
~2.560-2.580 (right panel).

They also compared three block arrangements — fully parallel **SAM**, semi-parallel **SA_M**, fully
sequential **S_A_M** — finding *"semi-parallel SA_M configuration provides the best results"*, and
noting *"as block configuration becomes more sequential SAM → SA_M → S_A_M, the optimal SSM fraction
reduces as 3/8 → 2/8 → 1/8. At the moment, we don't have an explanation of this behavior."*
License: Falcon LLM license.

> **[INFER] Two things make this the most useful 2025 data point for this experiment.** First, it is an
> **independent, 21-configuration, parameter-controlled confirmation that *less attention is better*** —
> arriving at 1/8 = 12.5% of *channels*, strikingly close to the 8-12.5% *layer*-fraction consensus, via
> a completely different mechanism. That two orthogonal parameterizations land on the same number is the
> strongest evidence in this document that ~10% is a real optimum rather than an artifact.
> Second, note the interaction they could not explain: **the optimal recurrent-mixer fraction depends on
> the parallel-vs-sequential topology.** That is a warning that "optimal ratio" is not topology-invariant,
> so a mostly-LIV *sequential* stack should not assume LFM2's or anyone else's ratio transfers.
> Also worth copying: their **1.2B / 70GT / identical-hyperparameters / near-constant-params** ablation
> design is a well-executed template, and Falcon-H1-0.5B (32,778 vocab, d=1024, 36 layers) is the closest
> published production model to this experiment's target size.

**IBM Granite 4.0** (Oct 2025) [FACT, from model cards/blog]: hybrid Mamba-2 + attention at
approximately **9:1 → ~10% attention**, Apache-2.0, with H-Micro / H-Tiny / H-Small variants (some MoE).
**No public ratio ablation.** **[INFER] Another independent production landing at ~10%.**

**Zamba (2405.16712) / Zamba2 (2411.15242)** [FACT]: a Mamba backbone plus **a single global attention
block whose weights are *shared* and re-applied every ~6 layers**. Zamba2-1.2B and 2.7B are in range.
Apache-2.0. **[INFER] Weight-sharing is a clever way to get attention's benefit at ~1 block's parameter
cost; it is an orthogonal trick to the ratio question and probably out of scope here, but worth one
sentence in the paper as related work.**

---

## 2. Empirical consensus on the attention:linear-layer ratio

### 2.1 Every published ratio ablation, in one table

| Source | Scale of ablation | Attention fractions tested | Best | Verdict |
|---|---|---|---|---|
| **Mamba-2** (2405.21060) Tbl 2 | 350M, 48L, **7B tokens**, Pile, GPT-2 tok | 0,1,2,3,4,5,6,7,9,11,15,24 of 48 (0-50%) | **6/48 = 12.5%** (ppl 8.26) | *"around a 10% ratio of attention layers performs best"* |
| **Waleffe et al.** (2406.07887) Fig 4 | **130M, 24L**, confirmed at **840M** | swept attention % | **~8%** | *"minimized when roughly 8% of the layers are self-attention"* |
| **Waleffe et al.** MLP sweep | 8% attn fixed, MLP 5%→50% | — | **50% MLP** | 30-50% MLP free; 50% is **20% faster** |
| **Jamba** (2403.19887) Tbl 4 | **1.3B, 250B tokens** | a:m = 1:3 (25%) vs 1:7 (12.5%) | **tie** | both beat pure; 1:7 chosen for cost |
| **Jamba** Tbl 5 | 7B, 50B tokens | 1:7 vs pure | 1:7 | hybrid > both pure |
| **MAD** (2403.17844) Fig 4.2 | IsoFLOP, 70M-7B, Pile | 0, 8.3, 25, 50, 100% attention stripes | **25%** | *"compute-optimal hybridization ratio for striped models is 25% across all IsoFLOP groups"* |
| **Hymba** (2411.13676) Tbl 1/10 | 300M, 100B tokens | all-SWA → +global in some layers | **3 global layers (first/middle/last)** | all-SWA cost **>20% recall** |
| **Samba** (2406.07522) Tbl 6 | 438M, 20B tokens | 0, 1, 2 full-attn layers in Mamba-MLP | **0 full (use SWA)** | 1 full-attn layer → **explodes at 16K** |
| **Gated DeltaNet** (2412.06464) Tbl S.2 | **500M, 15B tokens** | 4 orderings of Mamba2/GDN/SWA | Mamba2+GDN+SWA | ordering worth ~0.1-0.8 pts |
| **Falcon-H1** (2507.22448) §2.1 | **1.2B, 60L, 70GT**, near-constant params | **21 partitions** of channels across SSM/attn/MLP, α_A from 1/8 to 6/8 | **α_A = 1/8 (12.5% of channels)**; adopted 2:1:5 | *"Having more attention channels significantly degrades the performance"*; optimal SSM fraction shifts 3/8→2/8→1/8 as topology goes parallel→sequential |
| **Nemotron-H** (2504.03624) | 8B production | adopts ~8% | **4/52 = 7.7%** | follows Waleffe |
| **MiniMax-01** (2501.08313) | production + scaling law | 1 softmax per 8 | **12.5%** | — |
| **Qwen3-Next**, **Kimi Linear** | production | 3:1 adopted | **25%** | no ablation published |
| **IBM Granite 4.0** | production | 9:1 adopted | **~10%** | no ablation published |
| **LFM2** | production | 6/16 adopted | **37.5%** | no ablation published |

### 2.2 The convergent answer

**[FACT-grounded synthesis] There are two stable attractors, not one:**

- **~7-12.5% attention** when the linear mixer is a **large-state SSM** (Mamba-2, with `d_state`=128
  per head) *and* MLP layers are present: Mamba-2 (10%), Waleffe (8%), Nemotron-H (7.7%),
  Granite 4.0 (~10%), Jamba (12.5%), MiniMax-01 (12.5%), **Falcon-H1 (12.5% of channels, arrived at
  independently via a 21-point channel sweep)**.
- **~25%** when counting **sequence-mixing layers only** or when the linear mixer has smaller state:
  MAD (25% compute-optimal), Qwen3-Next (25%), Kimi Linear (25%), Jamba 1:3 (25%, tied with 1:7),
  Griffin (33%), Samba (~50% of mixers, but windowed).

**[INFER] These two attractors are substantially the same finding under different denominators.**
A "1 attention : 7 Mamba" Jamba block with MoE/MLP every other layer is 12.5% of *all* layers but
~12.5% of mixers; whereas Mamba-2-Hybrid's 4/56 = 7.1% of all layers is 4/28 = **14.3% of
sequence-mixing layers** (since 28 of 56 are MLP). **Normalizing every entry to "fraction of
sequence-mixing layers that are attention" collapses the range to roughly 12-25%.** This
normalization is the single most useful analytical move for the paper, and I have not seen it done in
any of these papers — it would be a genuine contribution of the write-up.

**The robust claims** (multiple independent labs, multiple scales):
1. **Some attention is necessary.** Zero-attention models fail at recall/copying/ICL — Hymba 300M:
   recall 19.23 vs 39.98 (−20.75); Jamba 1.3B: IMDB 48.8 vs 84.1 (−35.3). This is the strongest and
   most reproducible finding in the entire literature.
2. **Very little attention suffices.** 1-in-8 to 1-in-14 layers recovers most of it. Jamba: ICL works
   *"even when only 1 out of 8 layers is an Attention one"*; induction heads found in all 3 attention layers.
3. **Too much attention is worse than the optimum**, even at fixed params — Mamba-2: 24/48 → 8.50 vs
   6/48 → 8.26. Hybrids beat *both* pure endpoints.
4. **The basin is flat.** Mamba-2's 2-11 attention blocks (4-23%) all within 0.06 ppl; Jamba's 1:3 and
   1:7 indistinguishable. **So "what is the optimum" is a genuinely hard measurement, and any claim to
   resolve it needs seeds + a reported noise floor.**
5. **MLP fraction is nearly free (30-50%), and 50% is fastest.**

**Genuine disagreements:**

- **Full vs windowed attention.** Waleffe/Nemotron-H use **full global** attention in their few
  attention layers and never ablate SWA. Samba/Griffin/Gated-DeltaNet use **only SWA** and never use
  full. Samba Table 6 shows a *full*-attention layer in a Mamba hybrid **explodes beyond training
  length** (13.66 ppl at 16K vs 9.57 for SWA), while Waleffe's full-attention hybrid extends fine to
  128K. **[INFER] Probable reconciliation: Waleffe used NoPE and continued pretraining at the longer
  length; Samba used RoPE and evaluated zero-shot extrapolation. So the disagreement is really about
  extrapolation protocol, not about attention type.** This is a real confound to control in §7.
- **Whether the optimum is ~10% or ~25%.** Partly the denominator issue above; partly that MAD's 25%
  is *compute*-optimal at IsoFLOP (attention is cheaper per param than Hyena's long convs in their
  setup) rather than *loss*-optimal at fixed params.
- **Interleaved vs parallel.** Hymba Table 1 A→B shows **parallel beats sequential** (+4.74 recall at
  300M/100B); Falcon-H1 independently chose parallel. Everyone else interleaves. **Unresolved.**

### 2.3 Placement — where the attention layers go

**[FACT] Convergent rules across papers:**

| Rule | Sources |
|---|---|
| **Do NOT put attention in the first layer** | Mamba-2 fn.6 (*"not at the very beginning"*); Waleffe (Mamba first, so NoPE works); Nemotron-H (first layer is Mamba-2); LFM2 (first attn at idx 2) |
| **Do NOT put attention in the last layer** | Mamba-2 fn.6; Nemotron-H (last layer is FFN); Gated DeltaNet (winning order ends SWA→...); LFM2-2.6B (last 2 are conv) |
| **Space them evenly / disperse** | Mamba-2 (indices 9,18,27,36,45,56 of 64); Waleffe Alg.1 (near-equal Mamba runs); Nemotron-H (*"evenly disperse"*); Jamba (layers 4,12,20) |
| **In-distribution, exact position matters little** | Mamba-2 fn.6: *"as long as the attention layers are spaced out... the model quality does not depend very much on the exact location"* |
| **For extrapolation, position matters a lot** | Samba Tbl 6: full attn at block 11 → 13.66@16K; block 5 → 12.83; block 0 → 10.63 |
| **A few global layers at first/middle/last recovers recall** | Hymba (3 global layers suffice; >20% recall recovered) |

**[INFER] Synthesis for this experiment:** hold placement fixed by the "evenly dispersed, never first,
never last" rule for the main ratio sweep (this is the published consensus and avoids confounding),
then run a small dedicated placement sub-study (early-heavy vs uniform vs late-heavy) at one ratio,
evaluated **at and beyond training length**, because that is where the literature actually disagrees.
Note LFM2's own pattern is *late-heavy* (gaps 3,3,3,2,2,2) — testing uniform-vs-LFM2's-late-heavy at
matched ratio is a cheap, novel, publishable comparison.

---

## 3. Training codebases that can actually express this architecture

### 3.1 Headline finding

**[FACT] OLMo-core already contains everything this experiment needs, including a production-grade
3:1 hybrid that AI2 trained and released, plus a fused depthwise causal conv layer.** I did not expect
this and it changes the recommendation decisively. Specific evidence, all from the upstream repo:

1. **`src/olmo_core/nn/convolution.py` defines `CausalConv1d`** — *"CausalConv1d (aka short
   convolution) layer... implements a depthwise separable 1D convolution with causal padding"*,
   built as `nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size,
   groups=hidden_size, bias=bias, padding=kernel_size-1)`, dispatching to a **fused kernel** via
   `dispatch_causal_conv1d(..., backend="triton"|"cuda")` imported from
   `olmo_core.nn.attention.flash_linear_attn_api`. It accepts **`cu_seqlens`** for
   variable-length/document-boundary-aware convolution, has an `activation` argument
   (`"silu"|"swish"|None`), and implements `apply_cp()` for Ulysses-style **channel-parallel context
   parallelism**. **This is ~90% of the LIV mixer already written, kernel-optimized, and CP-aware.**
2. **`src/olmo_core/nn/attention/base.py` defines a `SequenceMixer` / `SequenceMixerConfig`
   abstraction with a `Registrable` registry.** The `SequenceMixer` ABC has exactly **five** abstract
   methods: `apply_tp`, `apply_cp`, `num_flops_per_token(seq_len)`, `init_weights`, plus `forward`.
   `SequenceMixerConfig` adds `num_params(d_model)` and `build(...)`. Registration is a one-line
   decorator: `@SequenceMixerConfig.register("gated_delta_net")`, `@SequenceMixerConfig.register("attention")`.
3. **`TransformerConfig` supports `block: dict[str, TransformerBlockConfig]` plus
   `block_pattern: list[str]`, and `block_overrides: dict[int, TransformerBlockConfig]`** — i.e. both
   *repeating patterns* and *per-layer-index overrides*, resolved by `resolve_block_configs(n_layers=,
   block=, block_pattern=, block_overrides=)`. This is exactly the config surface needed to express
   LFM2's irregular `full_attn_idxs=[2,5,8,10,12,14]` **and** clean repeating N:1 patterns **and**
   arbitrary placement studies, without touching model code.
4. **`src/olmo_core/nn/attention/recurrent.py` implements `GatedDeltaNet`** (adapted from `fla`, with
   the source commit cited in the docstring), using three `CausalConv1d` instances for q/k/v and
   dispatching to `dispatch_chunk_gated_delta_rule(...)` with `use_qk_l2norm_in_kernel=True`.
   `GatedDeltaNetConfig` exposes `conv_size: int = 4`, `conv_bias: bool = False`, `allow_neg_eigval`.
   It also implements a **correct analytic `num_flops_per_token`** including conv FLOPs and a
   **correct `num_params`** including conv params — meaning **compute- and param-matching are
   supported by first-class API, not by hand arithmetic.** This is unusually important for this
   experiment and is the single feature I would most struggle to replicate elsewhere.
5. **`src/scripts/official/OLMo-hybrid/` — AI2's released hybrid.** From the README, verbatim facts:
   *"OLMo-Hybrid-7B is a hybrid architecture combining Gated Delta Net (GDN) recurrent layers with
   standard attention layers in a 3:1 ratio (3 GDN layers followed by 1 attention layer, repeating).
   The model is based on OLMo3 7B but with reduced attention heads to match params/TPS for fair
   comparison with the pure transformer variant."* Config: d_model 3840, n_layers 32, 30 attention
   heads (reduced from 32), `block_pattern = ["gdn", "gdn", "gdn", "attn"]`, context 65,536.

   In `OLMo-hybrid-7B-pretrain.py` the param-matching is done explicitly and is worth quoting as a
   template:

   ```python
   REMOVE_HEADS = 2   # "Remove heads to match params/TPS of OLMo3 7B transformer.
                      #  This is to enable a fair comparison"
   model_config.d_model -= REMOVE_HEADS * 128
   num_heads = model_config.block.sequence_mixer.n_heads - REMOVE_HEADS
   ...
   gdn_block = attn_block.replace(sequence_mixer=GatedDeltaNetConfig(
       n_heads=num_heads, head_dim=int(0.75 * model_config.d_model / num_heads),
       allow_neg_eigval=True))
   model_config.block = {"gdn": gdn_block, "attn": attn_block}
   model_config.block_pattern = ["gdn", "gdn", "gdn", "attn"]
   ```

   **[INFER] This is the exact experimental pattern this project needs — swap one mixer, shrink width
   to restore parameter parity, keep everything else byte-identical — already implemented and
   validated at 7B by the lab that wrote the framework.** Adapting it from GDN to LIV short-conv is a
   small delta, not a project.
6. **Built-in `WSD` and `WSDS` schedulers** (`olmo_core/optim/scheduler.py` defines `WSD`,
   `ConstantWithWarmup`, `CosWithWarmup`, `LinearWithWarmup`, `PowerLR`, `HalfCosWithWarmup`,
   `CosWithWarmupAndLinearDecay`, `ComposableScheduler`, `SequentialScheduler`).
7. **A `model_ladder` package** (`base.py`, `transformer_model_configurator.py`,
   `wsds_chinchilla_run_configurator.py`, `utils.py`) purpose-built for **scaling-ladder ablation
   sweeps** — see §5.5, it hard-codes tested batch-size and LR scaling formulas.
8. **Callbacks include `wandb.py`** (plus `comet.py`), `checkpointer.py` (with
   `save_async=True`, `ephemeral_save_interval`), `speed_monitor.py`, `stability_monitor.py`,
   `gpu_memory_monitor.py`, `profiler.py`, `evaluator_callback.py`, `sequence_length_scheduler.py`,
   `batch_size_scheduler.py`, `model_merger.py`, `hf_converter.py`.
9. **Long-context stage is already scripted** (`OLMo-hybrid-7B-long-context.py`): extends to
   `DEFAULT_SEQUENCE_LENGTH = 65536`, `MAX_TOKENS = 100_000_000_000` (100B), `LR ≈ 2.07e-4`,
   **drops RoPE entirely ("DroPE")** via `attn_block.sequence_mixer.replace(rope=None)`, uses
   `NumpyPackedFSLDatasetConfig` and context parallelism (Ulysses degree 2), `LinearWithWarmup`.

### 3.2 Codebase evaluation matrix

Legend: ✅ first-class · 🟡 possible with work · ❌ absent/hostile. Stars/license/last-push from the
GitHub API on 2026-07-30 [FACT].

| Codebase | Custom mixer / hybrid stack | Fused depthwise causal conv | GQA + FlashAttn | μP | Ckpt/resume | W&B | Multi-node | License | Stars | Last push |
|---|---|---|---|---|---|---|---|---|---|---|
| **allenai/OLMo-core** | ✅ `SequenceMixerConfig` registry + `block_pattern` + `block_overrides` | ✅ **`CausalConv1d`** w/ triton+cuda backends, `cu_seqlens`, CP | ✅ (+ ring attn, TE backend) | 🟡 see §3.3 | ✅ async, data-order-preserving | ✅ `WandBCallback` | ✅ HSDP/FSDP2/TP/CP/Float8 | Apache-2.0 | 1,442 | 2026-07-30 |
| **fla-org/flash-linear-attention** | ✅ 40+ mixer layers + `hybrid.py` spec | ✅ `ShortConvolution` used throughout | ✅ `attn.py` w/ FA2, `window_size` | ❌ | n/a (library) | n/a | n/a | MIT | 5,487 | 2026-07-30 |
| **fla-org/flame** (trainer for fla) | ✅ via fla configs | ✅ via fla | ✅ | ❌ | ✅ (torchtitan DCP) | ✅ | ✅ (torchtitan FSDP/TP) | MIT | 408 | **2026-04-22** |
| **pytorch/torchtitan** | 🟡 model-per-folder, no mixer registry | ❌ | ✅ | ❌ | ✅ DCP | ✅ | ✅ 4D parallel | BSD-3 | 5,575 | 2026-07-30 |
| **NVIDIA/Megatron-LM** | ✅ `hybrid_override_pattern` string | ✅ `causal-conv1d` via Mamba stack | ✅ | 🟡 | ✅ | 🟡 (TB-first) | ✅ best-in-class | NOASSERTION (custom) | 17,265 | 2026-07-30 |
| **huggingface/nanotron** | 🟡 hand-edit model file | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ 3D | Apache-2.0 | 2,769 | 2026-05-26 |
| **Lightning-AI/litgpt** | 🟡 single `model.py`, one block type | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ FSDP | Apache-2.0 | 13,596 | 2026-07-20 |
| **EleutherAI/gpt-neox** | 🟡 Megatron-DeepSpeed fork | 🟡 (has Mamba/RWKV) | ✅ | ❌ | ✅ | ✅ | ✅ | Apache-2.0 | 7,448 | 2026-06-11 |
| **marin-community/levanter** (was stanford-crfm) | 🟡 JAX/Haliax named tensors | ❌ (no CUDA conv kernel) | ✅ (splash/flash) | ✅ **best μP story** | ✅ **bitwise-deterministic resume** | ✅ | ✅ TPU-first | Apache-2.0 | 708 | **2026-01-26** |
| **HazyResearch/zoology** | ✅ (synthetics only) | 🟡 | ✅ | ❌ | ❌ | ✅ | ❌ | Apache-2.0 | 280 | 2026-03-22 |
| **athms/mad-lab** | ✅ (synthetics only) | 🟡 | ✅ | ❌ | ❌ | ✅ | ❌ | MIT | 147 | **2024-12-17** |
| **HF transformers + Trainer** | 🟡 `Lfm2ForCausalLM` **exists** | ✅ via `causal-conv1d` | ✅ | ❌ | ✅ | ✅ | 🟡 | Apache-2.0 | — | active |

### 3.3 Per-codebase notes

**OLMo-core** — see §3.1. Additional facts: config system is **Python dataclasses**
(`TransformerConfig`, `TransformerBlockConfig`, `SequenceMixerConfig`, `TrainerConfig`,
`TransformerTrainModuleConfig`), driven by `python script.py --dry-run` and CLI overrides of the form
`--train_module.optim.lr=1e-5`. **[INFER] Python-dataclass configs (not YAML) are a real advantage
for a sweep**: the ratio/placement patterns are computed programmatically (`block_pattern` lists,
`block_overrides` dicts) rather than hand-written per variant.
`TransformerBlockType` includes `moe_hybrid` / `moe_hybrid_reordered_norm` (beta).
`InitMethod` enum offers `normal`, `normalized`, `llama`, `llama_depth`, **`fan_in`** (`std = 1/√d_in`,
embeddings `std = 1.0`) — **[INFER] `fan_in` is μP-flavored (μP's core width rule is fan-in-scaled init
+ 1/fan-in output scaling), and `embed_scale` + `OptimGroupOverride` provide the hooks for per-group LR
scaling.** [UNKNOWN] I found `mup` mentioned only in CHANGELOG/AGENTS.md and various train scripts,
not as a dedicated coordinate-check-verified `mup` module; **treat "full μP with coordinate checks" as
NOT verified in OLMo-core.** Pre-built configs `olmo2_1M / 14M / 30M / 60M / 100M / 190M / 370M /
600M / 760M / 1B / 1B_v2` [FACT] cover this experiment's whole range out of the box.
Also present: `z_loss_multiplier=1e-5` in the train module, `SkipStepAdamWConfig` (skips
non-finite steps — useful for unattended sweeps), `Float8Config`, `InstanceFilterConfig`
(repetition filtering), `DataMix` registry, `olmo_core/eval`, `olmo_core/kernels`, `olmo_core/launch`.

**fla + flame** — `fla/layers/` contains (exact listing, [FACT]): `abc, attn, based, bitattn, comba,
delta_net, deltaformer, forgetting_attn, gated_deltanet, gated_deltaproduct, gdn2, gla, gsa, hgrn,
hgrn2, kda, lightnet, linear_attn, log_linear_mamba2, mamba, mamba2, mamba3, mesa_net, mla, moba, mom,
multiscale_retention, nsa, parallax, path_attn, raven, rebased, rodimus, rwkv6, rwkv7, simple_gla,
wall_attn, yoco`. `fla/models/` additionally has a **`samba`** model and a **`hybrid.py`**.
`hybrid.py` defines a JSON-serializable `HybridAttentionSpec` TypedDict:

```python
class HybridAttentionSpec(_RequiredHybridAttentionSpec, total=False):
    """JSON-serializable settings for standard attention at selected model layers."""
    layers: list[int]        # required
    num_heads: int           # required
    num_kv_heads: int        # GQA
    qkv_bias: bool
    window_size: int | None  # SWA
    rope_theta: float
```
with validation that layer indices are in range, unique, and non-conflicting.
**[INFER] This is the single best "arbitrary attention placement in a linear-attention backbone" API in
existence — `layers: [2,5,8,10,12,14]` reproduces LFM2's pattern literally, and `num_kv_heads` +
`window_size` + `rope_theta` per-spec means full/SWA/NoPE arms cost a config edit each.** Note `kda.py`
is present, i.e. Kimi Delta Attention is already implemented — relevant to this repo's `KDA/` directory.
**Caveat [FACT]: `flame` was last pushed 2026-04-22 (3+ months stale) and pins a specific torchtitan
commit (`git+https://github.com/pytorch/torchtitan.git@0b44d4c`).** Its documented recipe trains a
**340M model on FineWeb-Edu `sample-100BT`** with `--optimizer.lr 1e-3`, `--lr_scheduler.warmup_steps
1024`, `--lr_scheduler.decay_type cosine`, `--training.seq_len 65536 --training.context_len 4096
--training.varlen`, `--training.steps 20480`, `--training.max_norm 1.0`, `--training.skip_nan_inf`,
`--training.compile`, `--checkpoint.interval 2048`. **[INFER] That recipe is a ready-made,
directly-citable small-scale baseline and its `varlen` + `context_len` split is exactly the
document-packing control §7 needs.** The README also explicitly warns: *"Do not use streaming mode if
you are concerned about resuming training."*

**Megatron-LM** — [FACT] contains the Mamba-2-Hybrid implementation from 2406.07887 with a
**`hybrid_override_pattern`** string, referenced in `megatron/core/models/hybrid/hybrid_model.py`,
`megatron/training/models/hybrid.py`, `megatron/core/transformer/transformer_config.py`,
`megatron/training/arguments.py`, `megatron/training/checkpointing.py`. The pattern string uses the
`M`/`*`/`+` alphabet from the paper. **[INFER] So Megatron is the *only* way to reproduce Waleffe et al.
bit-for-bit, and if the goal were to extend that specific paper it would be the pick. But it is heavy,
its license is non-standard (`NOASSERTION` — a custom NVIDIA license, not Apache), its config surface is
hundreds of CLI flags, and adding a genuinely new mixer means engaging with `TransformerConfig` +
`TransformerLayerSpec` + TE plumbing. Poor fit for a fast many-variant academic sweep at ≤1B.**

**levanter** — [FACT] the repo has **moved to `marin-community/levanter`**; last push 2026-01-26 (the
stalest of the serious trainers). Strengths are real and unique: JAX + Haliax **named tensors**,
**bitwise-deterministic and fully reproducible data-order-preserving resume**, and the best μP support
of any candidate. **[INFER] Weaknesses are disqualifying here: no fused CUDA depthwise causal conv (you
would write a Pallas kernel), the entire linear-attention/SSM kernel ecosystem is PyTorch/Triton, and
it is TPU-first. Choosing levanter means giving up `fla`, `causal-conv1d`, Mamba-2 SSD, and Gated
DeltaNet baselines.** Its determinism story is nevertheless the gold standard the experiment should try
to match in PyTorch.

**zoology** — [FACT] *"Understand and test language model architectures on synthetic tasks."*
Apache-2.0, 280 stars. **What it does:** multi-query associative recall (MQAR) and related synthetic
recall tasks, with a sweep harness over architectures/dimensions, and is the empirical backbone of the
Based/"just read twice" line of work. **What it does NOT do:** real-corpus pretraining, tokenized data
pipelines, multi-node, checkpoint/resume at scale, GQA-transformer baselines at 100M-1B. **[INFER] Use
it (or MAD) as a *pre-screen*, never as the main training codebase.**

**mad-lab** — see §3.4. **[FACT] Last push 2024-12-17 — effectively unmaintained.**

**safari / savanna / flash-fft-conv** — [INFER, low confidence] `safari` is the H3/Hyena research
codebase (long convolutions via FFT), `savanna` is the Evo-2/StripedHyena-2 training stack (very
heavyweight, genomics-oriented), `flash-fft-conv` provides fused FFT convolution kernels for *long*
convolutions. **All three target long/global convolutions, whereas LIV is a K=3 short depthwise conv
— FFT machinery is irrelevant and slower for K=3.** Not recommended. [UNKNOWN] I did not verify their
current maintenance status in this pass.

**HF transformers + Trainer** — [FACT] `Lfm2ForCausalLM` and `Lfm2ShortConv` ship in transformers
today, calling `causal_conv1d_fn`/`causal_conv1d_update`, with `layer_types` in config. **[INFER] So
transformers is by far the *fastest path to a correct stock-LFM2 reference implementation* — worth
using as a numerical cross-check oracle for your own LIV mixer (assert outputs match to 1e-5 on random
input). But `Trainer` is a poor pretraining harness: no first-class token-budget/data-order control, no
mixer registry, weak multi-node ergonomics, and its config surface invites silent inconsistency across
arms.** Use as an oracle and for eval/export, not for the sweep.

**litgpt / nanotron / torchtitan / gpt-neox** — all competent trainers, none has a **per-layer mixer
registry** or a fused short-conv. In each, "add a new mixer + interleave it" means editing the single
model file and inventing your own layer-pattern config. **[INFER] That is perhaps 200-400 LOC of
model+config work each, plus you still must write the param/FLOP accounting for matching by hand — the
exact thing OLMo-core gives you for free. torchtitan is the best of these (and is what `flame` builds
on), so it is the natural base if you ever outgrow `flame` but want to stay in that ecosystem.**

### 3.4 MAD (Mechanistic Architecture Design) — exactly what it offers

**Paper:** "Mechanistic Design and Scaling of Hybrid Architectures", arXiv **2403.17844** (v2, 19 Aug
2024). **Repo:** `https://github.com/athms/mad-lab` (MIT, 147 stars, last push **2024-12-17**).
The paper states: *"An implementation of the MAD tasks are available at https://github.com/athms/mad-lab."*

**[FACT] The complete synthetic task suite (6 tasks):**

| Task | What it probes |
|---|---|
| **In-context recall** | Given a sequence of key-value pairs, recall the value for a key — tests direct lookup with no external knowledge, i.e. the associative-recall/induction capability |
| **Fuzzy in-context recall** | Variable-length keys and values, so the model must semantically group adjacent tokens ("blue sky" vs "gray sky") before recalling — harder variant |
| **Noisy in-context recall** | Irrelevant noise tokens from a special vocab subset are interleaved between key-value pairs; model must ignore a *fixed* noise dictionary |
| **Selective copying** | Copy tokens from one position to another in order, ignoring inserted noise tokens — tests position-aware, order-preserving copying |
| **Compression** | Compress a random input sequence into a **single aggregation token** such that an MLP can reconstruct it — tests "token concatenation" / state capacity |
| **Memorization** | Learn a *fixed* key-value mapping from training data (facts), constant across samples — needs no in-context computation |

**[FACT] The MAD protocol (4 principles):**
1. Each MAD score **averages over a range of task difficulty levels**, varying input sequence length,
   vocabulary size, and training-set size independently (plus task-specific vars like noise ratio).
2. **Fixed-state architectures are normalized to iso-state and iso-parameter**, all normalized to a
   **common total state dimension of 4096**, including MoE models.
3. Each architecture is **swept over a grid of learning rate and weight decay**, and **only the best
   runs** enter the analysis.
4. Evaluation always on an independent per-setting eval set.

**[FACT] Model scale:** *"small two-blocks architectures"* — 2-block models built from attention,
SwiGLU, and variants of efficient implicit recurrent/convolutional layers. **21 distinct architectures**
evaluated in total, including sequential, striped, and sparse-parallel (MoE) topologies, plus a novel
**Hyena experts** layer (top-K routing over smaller Hyena mixers).
[UNKNOWN] The paper text I extracted does not state wall-clock per sweep beyond *"requiring only
minutes of training time"* per task — I could not verify total sweep hours. Flagging.

**[FACT] Correlation with scale:** *"MAD accuracy is rank-correlated with compute-optimal perplexity at
scale"*, with *"particularly strong correlation for models in the [hybrid/striped class]"*.
**[UNKNOWN] No numeric correlation coefficient (Spearman ρ / Kendall τ) appears in the text I
extracted** — the claim is made via Figure 1.1 panel [D]. **This is an important caveat: "MAD predicts
scaling" is asserted qualitatively, so treat MAD as a screening heuristic, not a validated predictor.**

**[FACT] Scaling-law setup:** *"training over 500 language models between 70 million and 7 billion
parameters"*, all on **The Pile**, using an **IsoFLOP** approach (FLOP budgets shown include 4e18,
8e18, 2e19, 4e19, 8e19). Baselines: **Transformer++, Hyena, Mamba**, plus striped variants
(StripedHyena, StripedMamba, Striped Hyena + SwiGLU / + MoE, Striped Hyena Experts + MoE,
Striped MH Hyena + MoE/SwiGLU).

**[FACT] The findings most relevant to this experiment:**
- **Finding 1:** *"Striped architectures outperform all non-striped architectures on composite
  metrics."*
- **Finding 6:** *"The off compute-optimal perplexity gap is proportional to the hybridization
  ratio"*, for all IsoFLOP groups — and *"the suboptimality gap in hybrids is smaller than
  Transformers, meaning they are better suited to training outside the optimal frontier."*
  **[INFER] Directly relevant: this experiment will almost certainly *overtrain* small models
  (tokens/param ≫ 20), and MAD says hybrids are advantaged exactly there. Good news for the
  hypothesis, and a reason to report an overtrained point, not just Chinchilla-optimal.**
- **Finding 7:** *"The compute-optimal hybridization ratio for striped models is 25% across all
  IsoFLOP groups"* (Fig 4.2 / Table D.1). Fig 4.2 sweeps **100%, 50%, 25%, 8%, 0% attention** stripes
  and finds StripedHyena beats both pure Hyena (0%) and Transformer++ (100%) at every FLOP budget,
  with **25% optimal**.
- **Finding 8 (state-optimal scaling):** *"There exists a relation of the type P* ∝ M^c between
  compute-optimal perplexity P* and total state size M, with c ≈ −0.28"*, consistent across model
  classes. **[INFER] This is the most under-cited result in the hybrid literature and it is *directly*
  about this experiment's core tension: a K=3 LIV conv has state `d×2` per layer, versus Mamba-2's
  `d_state×d_head` per head. If perplexity scales as (total state)^−0.28, a mostly-LIV model is
  state-starved and the theory predicts it needs *more* attention (which has unbounded state) to
  compensate — a quantitative, falsifiable prediction that this experiment can test. I would make
  "total state size" an explicit reported axis for every arm.**
- **Batch size:** *"scaling the batch size with FLOP budgets, thus keeping it fixed within each
  IsoFLOP group, to be a simple and robust approach."*

### 3.5 Recommendation

**PRIMARY: `allenai/OLMo-core`** (Apache-2.0, `https://github.com/allenai/OLMo-core`, docs at
`https://olmo-core.readthedocs.io/`).

Justification, tied to the axes:
1. **It is the only candidate with a fused depthwise causal conv (`CausalConv1d`) *and* a per-layer
   mixer registry *and* a released, param-matched 3:1 hybrid training script.** Every other candidate
   is missing at least one of the three.
2. **`num_params()` and `num_flops_per_token()` are part of the `SequenceMixerConfig` API.** Given
   that the entire experiment is defined by parameter-/token-/compute-matching, having the framework
   compute both per mixer, per layer, is worth more than any other single feature. Elsewhere this is
   hand-rolled and error-prone — and a matching bug silently invalidates every result.
3. **Adding the LIV mixer is genuinely small.** Concretely: create
   `src/olmo_core/nn/attention/liv.py` with `class ShortConvMixer(SequenceMixer)` implementing
   `forward`, `apply_tp`, `apply_cp`, `num_flops_per_token`, `init_weights`, plus
   `@SequenceMixerConfig.register("liv_short_conv")` on a `ShortConvMixerConfig` dataclass with
   `num_params()`. The conv itself is `CausalConv1d(hidden_size=d, kernel_size=3, activation=None)`;
   the gating is `in_proj: d→3d`, chunk into (B, C, x), `B*x` → conv → `C*` → `out_proj: d→d`.
   **[INFER] Estimate ~150-250 LOC including the config class and a numerical-equivalence test
   against HF `Lfm2ShortConv`.** `recurrent.py`'s `GatedDeltaNet` (which already composes
   `CausalConv1d` three times and implements all five methods plus FLOP/param accounting) is a
   line-by-line template.
4. **Sweep ergonomics:** `block_pattern` for regular N:1 ratios, `block_overrides` for irregular
   placements (LFM2's `[2,5,8,10,12,14]`, early/middle/late studies) — programmatically generated from
   Python, no YAML duplication across ~20 variants.
5. **The ablation protocol is pre-built:** `WSD`/`WSDS` schedulers + the `model_ladder` package +
   `wsds_chinchilla_run_configurator.py` (tested LR/batch formulas, §5.5) = DataDecide-style
   methodology out of the box.
6. **Ops:** `WandBCallback`, async checkpointing with ephemeral saves, `SkipStepAdamWConfig`
   (skip non-finite steps), `stability_monitor`, `speed_monitor`, HSDP/FSDP2/TP/CP, Float8.
   **[INFER] `save_async` + `SkipStepAdamW` matter a lot for the recorded constraint that this machine
   has died mid-run — the framework is built for unattended, resumable, many-run operation.**
7. **Long context is solved and scripted:** the DroPE 65K stage, `NumpyPackedFSLDatasetConfig`,
   `sequence_length_scheduler` callback, and Ulysses CP (which `CausalConv1d.apply_cp` supports).
8. **Provenance:** it is the framework behind OLMo 2 (arXiv 2501.00656) and OLMo 3, so the
   transformer baseline is a *credible, published* baseline rather than one you tuned yourself —
   which is exactly the criticism a reviewer levels at architecture papers.
9. **Local availability:** the repo already sits at `/Users/ericwu/Developer/Capstone_LLM/OLMo-core/`.
   (Per instructions I did not audit that copy; another agent is doing so. **Do verify its checked-out
   commit actually contains `nn/convolution.py`, `nn/attention/base.py`, `nn/attention/recurrent.py`,
   and `src/scripts/official/OLMo-hybrid/` — all of my findings are against upstream `main` as of
   2026-07-30.**)

**Known gaps to accept:** (a) μP is **not** verified as a first-class coordinate-checked feature — plan
to use the `fan_in` init method plus the ladder's empirical LR formula instead of true μP, or port μP
yourself; (b) OLMo-core's own defaults (reordered-norm, QK-norm, z-loss) differ from the
Mamba/`fla`-lineage defaults, so **numbers will not be directly comparable to published Mamba-2/GDN
perplexities** — an internal-consistency-only study; (c) fewer exotic mixers than `fla` (no RWKV-7,
RetNet, GLA out of the box).

**FALLBACK: `fla-org/flash-linear-attention` + `fla-org/flame`** (both MIT).

Justification: **the widest mixer library in existence (40+ layers incl. `mamba2`, `gated_deltanet`,
`kda`, `rwkv7`, `gla`, `retnet`, plus a `samba` model), `ShortConvolution` already used throughout, and
`hybrid.py`'s `HybridAttentionSpec` giving literal `layers: [...]` placement with per-spec
`num_kv_heads` / `window_size` / `rope_theta`.** If the experiment's emphasis shifts from "LIV vs GQA"
toward "LIV vs every recurrent mixer", `fla` wins outright because those baselines are free and
kernel-optimized, and the published GDN protocol (§1.9) is native to it.
**Risks to accept:** `flame` last pushed 2026-04-22 and pins an old torchtitan commit; no μP; no
built-in param/FLOP matching API; and OLMo-core's own GDN is *adapted from fla* anyway, so you can
often cite/borrow `fla` while training in OLMo-core.

**Pre-screen tool (not a trainer): `athms/mad-lab` or `HazyResearch/zoology`.** [INFER] Run the 6 MAD
tasks (especially in-context recall, noisy recall, selective copying, compression) across candidate
LIV:GQA ratios *before* spending GPU-hours. Given Hymba's 20.75-point recall gap and Jamba's 35-point
format-following gap versus ~0.06 ppl differences, **the synthetic recall tasks are where this
experiment's signal actually lives.** Caveat the MAD-predicts-scaling claim as unquantified, and note
mad-lab is unmaintained (2024-12-17) so budget time for dependency repair.

---

## 4. OLMo-core suitability (research-only; local audit delegated elsewhere)

Answered in depth in §3.1 and §3.3. Summary of the three questions asked:

**Q: Does OLMo-core support custom attention/mixer replacement?** **[FACT] Yes, first-class.**
`olmo_core/nn/attention/base.py` defines `SequenceMixer` (ABC, 5 abstract methods: `apply_tp`,
`apply_cp`, `num_flops_per_token`, `init_weights`, + `forward`) and
`SequenceMixerConfig(ModuleConfig, Registrable, Generic[SeqMixer])` with `num_params(d_model)` and
`build(...)`. Registration is `@SequenceMixerConfig.register("<name>")`. Two mixers ship today:
`"attention"` (`AttentionConfig`, in `nn/attention/__init__.py`) and `"gated_delta_net"`
(`GatedDeltaNetConfig`, in `nn/attention/recurrent.py`).

**Q: Does it support hybrid layer stacks?** **[FACT] Yes, two mechanisms, both in
`nn/transformer/config.py` + `model.py`:**
- `block: dict[str, TransformerBlockConfig]` + `block_pattern: list[str]` — a named-block dict plus a
  repeating pattern, e.g. `block_pattern = ["gdn","gdn","gdn","attn"]`. `TransformerConfig` warns if
  `n_layers % len(block_pattern) != 0` (pattern is cycled and truncated).
- `block_overrides: Optional[Dict[int, TransformerBlockConfig]]` — per-layer-index override (note:
  *"Not supported if `block` is a dict of named blocks"*), so use one mechanism or the other.
Both are resolved by `resolve_block_configs(n_layers=, block=, block_pattern=, block_overrides=)`,
then `model.blocks` is an `nn.ModuleDict` keyed by string layer index.
**[INFER] `block_pattern` covers regular N:1 ratio arms; `block_overrides` covers LFM2's irregular
`[2,5,8,10,12,14]` and any placement study. Together they cover every arm this experiment needs.**

**Q: What is the config system like?** **[FACT] Python dataclasses, not YAML.** Layers:
`TransformerConfig(ModelConfig)` → `d_model`, `n_layers`, `vocab_size`, `block`, `block_pattern`,
`block_overrides`, `lm_head`, `init_method`, `init_std`, `embedding_init_std`, `embed_scale`,
`tie_word_embeddings`, `dtype`, `init_seed`; `TransformerBlockConfig` → `sequence_mixer`
(*"e.g. attention, recurrent, convolution, etc."*), `feed_forward`, norms, and a residual-scaling
factor; `TransformerTrainModuleConfig` → `optim`, `scheduler`, `compile_model`, `dp_config`,
`float8_config`, `z_loss_multiplier`, `max_grad_norm`, `rank_microbatch_size`, `max_sequence_length`;
`TrainerConfig` → `save_folder`, `max_duration`, `.with_callback(name, cb)`.
Entry points are plain Python scripts run under `torchrun`, with `--dry-run` to print the resolved
config and dotted CLI overrides (`--train_module.optim.lr=1e-5`, `--trainer.max_duration.value=3`).
Pre-built ladders `TransformerConfig.olmo2_{1M,14M,30M,60M,100M,190M,370M,600M,760M,1B,1B_v2}` and
`olmo3_7B` exist [FACT]. `TransformerBlockType` includes `reordered_norm` (the OLMo 2 default),
`moe_hybrid`, `moe_hybrid_reordered_norm`.
**[INFER] Dataclass configs are a significant practical win for a ~20-variant sweep: patterns are
*computed* (a loop that emits `block_pattern` lists for each ratio), not copy-pasted, which is how you
avoid the classic "arm 7 accidentally had a different LR" failure.**

**OLMo 2 architectural choices worth inheriting** (arXiv **2501.00656**, *"2 OLMo 2 Furious"*) [FACT,
from §2.1 and Table 1 of the paper]:
- **RMSNorm** without bias (replacing nonparametric LayerNorm)
- **Reordered norm** — normalize *outputs* of attention and MLP, not inputs:
  `h := x + RMSNorm(Attention(x))`, `h_out := h + RMSNorm(MLP(h))` — *"to stabilize training"*
- **QK-norm** — RMSNorm on key and query projections before attention, *"avoids attention logits being
  too large, which can lead to training loss divergence"*
- **Z-loss** regularization, *"empirically shown to improve run stability"*; **Table 1 gives the weight
  as 1e-5** (and OLMo-core's train module default is `z_loss_multiplier=1e-5`)
- **RoPE θ = 500,000** (up from 10,000), *"matching Grattafiori et al. (2024)"* (Llama 3)
- **No weight decay on embeddings** in OLMo 2 (Table 1 changes "Weight Decay on Embeddings" to No);
  implemented as `OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))`
- Sequence length **4096**; gradient clipping **1.0**; batch size 1024 (7B) / 2048 (13B, 32B)
- 7B is MHA 32/32; 13B MHA 40/40; **32B is GQA 40/8**

**Tokenizer, and a caution that matters for §6** [FACT]: OLMo 2 switched to a **cl100k-based
tokenizer** (pre-tokenizer + vocabulary borrowed from cl100k, Apache-2.0 licensed) plus PII masking
tokens, *"As suggested by Tao et al. (2024)"* (= Scaling Laws with Vocabulary, arXiv 2407.13623).
**Table 2 compares the two tokenizers on a 1B model pretrained for 100B tokens from DCLM-baseline:**

| Tokenizer | OLMES (CF) | OLMES Gen | MMLU (CF) |
|---|---|---|---|
| OLMo 1 tokenizer (GPT-NeoX-20B based) | 59.8 | 42.4 | 34.8 |
| OLMo 2 tokenizer (cl100k based) | 60.6 | 42.7 | 35.2 |

and the paper explicitly states: *"Per Tao et al. (2024), at this model size and compute budget, the
larger OLMo 2 tokenizer is at a slight disadvantage; we expect improvement coming from larger
vocabulary to be more decisive at larger scales and for models trained on more tokens."*
**[INFER] Read carefully: at exactly this experiment's scale (1B, 100B tokens), AI2 says a large vocab
is theoretically disadvantaged, yet measured it slightly *better* (+0.8 OLMES). The effect is small
either way — so vocab choice should be driven by the embedding-parameter accounting in §6.2, not by
hoped-for quality gains.**

**OLMo-core also ships the full training-stage machinery** [FACT]: `DataMix.OLMo_mix_0925` (dolma3),
midtraining mixes, `OLMo-longmino-mix-0925` for long context, `InstanceFilterConfig` (repetition
filter: `repetition_max_period=13, repetition_min_period=1, repetition_max_count=32`),
`NumpyFSLDatasetConfig` / `NumpyPaddedFSLDatasetConfig` / `NumpyPackedFSLDatasetConfig`,
`TokenizerConfig.dolma2()` with `padded_vocab_size()`.

**Caveats / unknowns** [UNKNOWN]: (a) full μP with coordinate checks is **not** verified present —
`mup` appears in CHANGELOG/AGENTS.md and train scripts but I found no dedicated verified module;
(b) docs at `https://olmo-core.readthedocs.io/` exist but I evaluated the repo source primarily;
(c) OLMo-core's defaults (reordered norm, QK-norm, z-loss, cl100k vocab) differ from the
Mamba/`fla`-lineage defaults, so **absolute perplexities will not be comparable to published
Mamba-2/Samba/GDN numbers** — this must be an internally-controlled study.

---

## 5. Small-scale pretraining recipes (100M-1B)

### 5.1 Reference recipes, side by side

All [FACT] from the cited papers.

| Source | Size(s) | Tokens | Tok/param | Batch | Peak LR | Schedule | WD | Seq len | Clip | Vocab/tokenizer |
|---|---|---|---|---|---|---|---|---|---|---|
| **Mamba / Mamba-2 scaling** (2312.00752 Tbl 12 / 2405.21060 Tbl 9) | 125M / 350M / 760M / 1.3B | 2.5B / 7B / 15B / 26B | ~20 | 0.5M tok | 6e-4 / 3e-4 / 2.5e-4 / 2e-4 (**"5× GPT-3"** in improved recipe) | linear warmup + **cosine → 1e-5** | 0.1 | 2048 | 1.0 | GPT-2 50257 (scaling) / GPT-NeoX (300B runs) |
| **Mamba downstream** | ≤2.7B | 300B | ~110 | 1M tok @1.3B+ | 5× GPT-3 | cosine | 0.1 | 2048 | 1.0 | GPT-NeoX |
| **Samba** (2406.07522 Tbl 10) | 421M / 1.3B | 20B / 100B | 47 / 77 | **512** | **4e-4** | [UNKNOWN shape] | 0.1 | **4096** | 1.0 | 32000 |
| **Gated DeltaNet** (2412.06464) | 400M / 1.3B | 100B | 77 @1.3B | **0.5M tok** | **4e-4** | **cosine, 1B-token warmup** | 0.1 | **4096** | 1.0 | Llama-2 **32000** |
| **flame** recipe (fla-org/flame README) | 340M | ~10B ("10B" in dump path) | ~30 | seq 65536, ctx 4096, varlen | **1e-3** | cosine, **1024 warmup steps**, `lr_min 0.1` | — | 4096 ctx | 1.0 | fla-hub 1.3B tok. |
| **Waleffe** (2406.07887) | 130M / 840M / 8B | 1.1T or 3.5T | — | 256 (1.1T) / 1024 (3.5T) | 1e-4 / 3e-4 → min 1e-5 / 3e-5 | warmup 122K samples + **cosine** | 0.1 | 4096 | — | SentencePiece **256K** |
| **Griffin/Hawk** (2402.19427 Tbl 2) | 100M / 200M / 400M | 1.9B / 3.9B / 7.8B (Chinchilla) | ~20 | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] | [UNKNOWN] | 2048 | — | [UNKNOWN] |
| **TinyLlama** (2401.02385) | 1.1B | 3T (~3 epochs of 950B) | **~2700** | **2M tok** | **4e-4 → min 4e-5** | cosine, **2000 warmup steps** | 0.1 | 2048 | — | Llama 32000 |
| **MobileLLM** (2402.14905) | 125M / 350M | **1T** (480k iters); ablations 0.25T (120k iters) | ~8000 / ~2900 | 32/GPU × 32 GPUs | [UNKNOWN] | [UNKNOWN] | — | — | — | **32k**, embeddings **shared** |
| **Pythia** (2304.01373) | 70M-12B | 300B (Pile) / 207B (dedup) | — | **1024 samples** (= 2.1M tok) | per Tbl (size-dependent) | cosine | — | 2048 | — | Pile-trained BPE 50k |
| **OLMo 2** (2501.00656) | 7B/13B/32B | up to 6T | — | 1024 / 2048 | — | two-stage + anneal | — | **4096** | **1.0** | cl100k-based, z-loss 1e-5 |
| **DCLM ladder** (2406.11794) | 400M / 1B / 3B / 7B | **20 × params × multiplier** | **20 × mult** | — | — | — | — | — | — | — |

**[INFER] The modern consensus small-model recipe, distilled:**
- **AdamW, β = (0.9, 0.95)**, ε 1e-8 (fla uses 1e-15), **weight decay 0.1**, **no weight decay on
  embeddings** (OLMo 2), **gradient clip 1.0**, no dropout, BF16
- **RMSNorm**, no linear biases, SwiGLU, RoPE, **QK-norm** and **z-loss 1e-5** for stability
- **Init std 0.02** (OLMo-core default `init_std=0.02`; LFM2 `initializer_range: 0.02`) or fan-in
  (`std = 1/√d_in`) — the field is split; **fan-in/μP-flavored scales better across widths**
- **Seq len 4096** is now standard at this scale (2048 was the 2023 norm; Samba, GDN, Waleffe, OLMo 2
  all use 4096)
- **Peak LR ~4e-4 to 1e-3 at 300-500M**, decaying with size roughly as N^(−1/3) (see §5.5)
- **Batch 0.5M tokens** at 400M-1.3B (Samba 512×4096 = 2.1M is on the high side; GDN's 0.5M is the
  common choice)
- **Warmup ~1-2% of tokens** (GDN: 1B of 100B; Mamba: linear warmup; OLMo-hybrid: 2000 steps)
- **Tokens/param: 20 (Chinchilla) is the *floor*, not the target** — see §5.3

### 5.2 Vocabulary and z-loss specifics

- **z-loss**: OLMo 2 [FACT] adopts z-loss *"as it has been empirically shown to improve run
  stability"*, weight **1e-5** (Table 1; OLMo-core default `z_loss_multiplier=1e-5`), citing PaLM,
  Chameleon, and Wortsman et al.
- **init std**: OLMo-core `InitMethod` = `normal` (0.02) / `normalized` / `llama` / `llama_depth` /
  `fan_in`. For `fan_in` [FACT]: *"`std = 1/√d_in` where `d_in` is the fan-in... Embeddings use
  `std = 1.0`"*, and FFN `w3` gets `std / (2*num_blocks)**0.5` (depth-scaled residual init).
- **Weight decay on embeddings**: OLMo 1/0424 = Yes, **OLMo 2 = No** [FACT, Table 1].

### 5.3 Tokens-per-parameter: Chinchilla vs over-training

**[FACT] MiniCPM (arXiv 2404.06395)** re-measured the compute-optimal ratio with a WSD schedule and
found: *"the data size should be 192 times larger than the model size on average, as opposed to 20
times in Hoffmann et al. (2022)."* Their explanation is that Chinchilla's estimate came from a cosine
schedule, whose mid-training loss is *not* on the optimal envelope: *"Since they use Cosine LRS, the
loss is not optimal in the middle of the training, depicted by the concave curve"* — they refill the
concave part with a straight line to estimate what a WSD envelope would have given.

**[FACT] MAD Finding 6:** *"The off compute-optimal perplexity gap is proportional to the
hybridization ratio"*, and *"the suboptimality gap in hybrids is smaller than Transformers, meaning
they are better suited to training outside the optimal frontier."*

**[FACT] Practice at this scale is heavily over-trained:** TinyLlama 1.1B → 3T tokens (~2700
tok/param); MobileLLM 125M/350M → 1T (~8000 / ~2900); Griffin downstream models overtrained to 300B;
LFM2 → 10T at 350M-2.6B (~28,000 tok/param).

> **[INFER] Recommendation: target 40-100 tokens/param, i.e. 2-5× Chinchilla, and report at least two
> budgets.** Rationale: (a) 20× is the *floor* and MiniCPM says it is simply wrong for WSD;
> (b) 2700-28,000× is unaffordable here; (c) the deployment regime this architecture family targets
> (edge inference) is over-trained, so an under-trained comparison would be off-distribution for the
> claim; (d) MAD Finding 6 says hybrids *gain* relative advantage when over-trained, so a
> Chinchilla-only result would understate the hypothesis; (e) published comparables cluster here
> (GDN 1.3B/100B = 77×, Samba 421M/20B = 47×, Samba 1.3B/100B = 77×) so results stay interpretable.

### 5.4 WSD / warmup-stable-decay — and why it is the right protocol here

**[FACT, MiniCPM arXiv 2404.06395]** WSD has three phases: linear **warmup**, a long **stable** phase
at constant peak LR, then a short **decay** phase. Key measured findings:
- **The decay phase produces a sudden, large loss drop:** *"in the decay stage, as the learning rate
  begins to decrease, the loss experiences a significant rapid decline and quickly decreases to be
  equal to or lower than the Cosine LRS at step T = S."*
- **You can branch from the stable checkpoint:** *"we can reuse the model before decay and continue
  training with the previous high learning rate. After more steps of training S′, we can also perform
  annealing to achieve the same loss as the Cosine LRS at Cosine(S′)."*
- **"10% Steps are Enough":** across stable checkpoints at 40N, 60N, and 80N tokens, *"having a decay
  of 10% of the total tokens is sufficient to achieve the best results, while a decay of 2.5% of total
  tokens falls short. Therefore, in the subsequent training experiments, we use a decay of about 10% to
  ensure full convergence."*
- Notation used in their figures: `WSD(D, 0.1D)` — total tokens D with decay 0.1D. Sweeps shown
  include WSD(20N,2N), WSD(40N,2N), WSD(40N,4N), WSD(60N,2N), WSD(60N,6N), WSD(80N,2N), WSD(80N,8N),
  WSD(160N,16N), WSD(320N,32N), compared against Cosine(80N).
- **Optimal batch size** fitted on 0.009B/0.03B/0.17B models against C4 loss: `bs = 1.21 × 10^…/L^…`
  (a power law in loss; the paper reports the fit with LR 0.01 and cosine schedule per size).
- Their scaling-hyperparameter transfer uses **μP width scaling + depth scaling** (Yang et al. 2022,
  2023), *"We do not apply the attention softmax scaling techniques"* [FACT].

> **[INFER] Why WSD is the right schedule for *this* experiment specifically:** with cosine, every
> token budget is a separate run and every architecture variant must be re-run end-to-end for each
> budget. With WSD you train the stable phase once per architecture, then **fork cheap 10% decays at
> multiple token budgets** — giving 2-3 budget points per arm for ~1.2× the cost of one, and making the
> "does the optimum shift with token budget?" question (which §1.3's flat basin makes urgent)
> affordable. It also means a machine death mid-stable-phase costs you nothing: the stable checkpoint
> is the reusable asset. **Caveat:** cross-arm comparisons must always be decayed-to-decayed; comparing
> a stable-phase loss to a decayed loss is meaningless because of the size of that end-of-decay drop.

### 5.5 DataDecide / DCLM-style ablation methodology

**[FACT, DCLM arXiv 2406.11794]** DCLM defines **competition scales** `400M-1x, 1B-1x, 3B-1x, 7B-1x,
7B-2x`, where each names a parameter count and a **Chinchilla multiplier**, and *"The number of
training tokens for each scale is 20 × number of parameters × Chinchilla multiplier so that a
multiplier of 1x corresponds to a compute allocation that Hoffmann et al. found near-optimal."*

**The load-bearing methodological result [FACT]:** ranking of 10 curation methods at small scale
transfers to 7B-1x with **Pearson r = 0.838 (400M-1x), r = 0.956 (1B-1x), r = 0.982 (3B-1x)**, and
*"dataset improvements are largely orthogonal to training hyperparameters"* (Appendix H).
Also [FACT]: DCLM-POOL is **240T tokens** from Common Crawl (200B documents, 370TB); **DCLM-BASELINE
is 3.8T tokens**; they *"use a Bloom filter for DCLM-BASELINE and MinHash for other experiments"*,
having found both *"provide comparable downstream performance: within 0.2 CORE percentage points at
the 7B-2x scale."*

**[FACT] Pythia (arXiv 2304.01373)** is the canonical controlled-suite design: 8 sizes 70M-12B, trained
twice (Pile and deduplicated Pile — 300B vs ~207B tokens), **154 checkpoints per model**, *"identical
architectures"*, **consistent data ordering across all models**, and a public script to reproduce the
exact data order. On batch size [FACT]: *"we find no convergence issues with using batch sizes 4× to 8×
what is considered [standard]... Consequently, we use a batch size of 1024 samples"* (= ~2.1M tokens at
seq 2048).

**[FACT] OLMo-core operationalizes this** in `model_ladder/wsds_chinchilla_run_configurator.py`, whose
`WSDSChinchillaRunConfigurator` exposes `chinchilla_multiple` (must be ≥0.5 and a power of 2),
`decay_fraction: float = 0.1` (*"The duration of each decay as a fraction of the period. Must be at
least 10%"* — matching MiniCPM's finding exactly), `tokens_per_param: int = 20`, `lr_multiplier`, and
`stepped_schedule`. Its **tested formulas** are worth copying verbatim:

```python
# batch size (assumes sequence length 2048)
target_batch_size = round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))
# learning rate, optimal for 1xC, then halved
lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
lr /= 2.0          # "empirically seems to be near optimal, at least for Olmo models"
beta2 = 0.95 if batch_size >= 524_288 else 0.99   # larger beta2 for small batches
warmup = num_params          # "Warm up 1 token per parameter"
weight_decay = 0.1
# no weight decay on embeddings
```

**[INFER] Evaluating those formulas** (arithmetic is mine):

| Params | Batch (tokens) | LR (after ÷2) | β₂ | Warmup tokens | 20× tokens |
|---|---|---|---|---|---|
| 150M | ~406K | ~2.1e-3 | 0.99 | 150M | 3.0B |
| 350M | ~715K | ~1.6e-3 | 0.95 | 350M | 7.0B |
| 750M | ~1.19M | ~1.2e-3 | 0.95 | 750M | 15B |

**[INFER] Note these LRs (1.2e-3 - 2.1e-3) are 3-5× higher than Samba/GDN's 4e-4 at similar sizes, and
close to `flame`'s 1e-3 at 340M.** The discrepancy is real and matters: it reflects OLMo's WSD + higher
β₂ + `SkipStepAdamW` setup versus the cosine recipes. **Do not mix and match — pick one recipe family
and hold it fixed across all arms.** Also: if you deviate from seq len 2048, the batch formula's
2048 factor needs re-deriving (it targets a token count, so at 4096 you halve the *sample* count).

**[INFER] Recommended sweep protocol, assembled from the above:**
1. **Screen** candidate ratios/placements on MAD synthetics (§3.4) — minutes each, no GPUs at risk.
2. **Rank** at a single small scale (~150M, ~1x-2x Chinchilla ≈ 3-6B tokens), **≥2 seeds**, WSD with
   10% decay. Report a seed-noise band; Samba's own ±0.3% and Mamba-2's 0.06-ppl basin mean anything
   inside ~±0.05 ppl is a tie.
3. **Confirm** the top 3-4 arms at ~350M and ~750M, 40-100 tok/param, forking multiple decays from each
   stable phase.
4. **Verify rank transfer** across scales the way DCLM did (report the rank correlation across your own
   scales) — this is what makes small-scale architecture conclusions defensible.
5. Fix tokenizer, corpus, **data order (same seed)**, optimizer, schedule, seq len, and param count
   across every arm; vary only the mixer/ratio/placement.

---

## 6. Corpora and tokenizers

### 6.1 Corpus comparison

| Corpus | Size | Dedup | License | Long docs? | Notes |
|---|---|---|---|---|---|
| **FineWeb-Edu** (2406.17557, `HuggingFaceFW/fineweb-edu`) | **1.3T tokens**; HF reports **3,496,736,741 rows / ~10.36 TB** original files [FACT, datasets-server] | MinHash per-dump (inherits FineWeb) | **ODC-By 1.0** | Web-short; **measured below** | Edu-classifier filtered (score ≥3). `sample-10BT`/`100BT`/`350BT` subsets. **Used by Gated DeltaNet and flame** |
| **FineWeb** (2406.17557) | 15T tokens, 44TB | MinHash per-dump | ODC-By 1.0 | Web-short | The unfiltered parent |
| **DCLM-baseline** (2406.11794) | **3.8T tokens** [FACT]; pool 240T / 200B docs / 370TB | **Bloom filter** for baseline, MinHash elsewhere (within 0.2 CORE pts) [FACT] | CC-BY-4.0 (code MIT) | Web-short | fastText model-based filtering; strongest CORE scores of the web corpora |
| **C4** (`allenai/c4`) | ~156B tokens EN; HF reports 114,005,516 rows [FACT] | exact-dup 3-sentence spans | ODC-By | **Shortest** — badly truncated by design | Only for legacy comparability |
| **SlimPajama** (`cerebras/SlimPajama-627B`) | **627B tokens** (50% of RedPajama-1.2T after filtering) [FACT via TinyLlama: *"SlimPajama retains only 50% of the original tokens from RedPajama"*] | MinHashLSH, aggressive | mixed per-source (**incl. Books3 → contested**) | **YES — Books/arXiv/StackExchange** | **Used by Samba (both scales)** → best comparability for long-context ppl |
| **The Pile** (2101.00027) | 300B tokens (207B deduped) [FACT via Pythia] | mostly not deduped | mixed (**Books3 → removed from some mirrors**) | **YES — Books3, PG-19, arXiv, FreeLaw** | The Mamba/Mamba-2/MAD reference corpus. Legally awkward now |
| **Dolma** (2402.00159) | v1 3T; dolma3 mixes in OLMo-core | MinHash + exact | **ODC-By**, AI2 ImpACT | **YES** — books, papers, code, CC | OLMo's corpus; `DataMix.OLMo_mix_0925` and `OLMo-longmino-mix-0925` are wired into OLMo-core |
| **Nemotron-CC** (2412.02595) | 6.3T (4.4T real + 1.9B synthetic) | global fuzzy | permissive | Web-short | Higher quality-per-token than DCLM at long budgets |
| **Common Pile / Comma v0.1** (2506.05209) | ~8TB / HF reports 2,547,527 rows for the training set [FACT] | per-source | **openly licensed ONLY** — the point of the dataset | mixed (see caveat) | Best answer if strict license cleanliness is required |
| **PG-19** (`deepmind/pg19`) | ~11B tokens, ~28k books | n/a | **public domain** | **YES — entire books, ≫32K tokens** | The cleanest long-document source available |

### 6.2 Document-length distribution — measured, not guessed

The task flagged this as CRITICAL and published statistics are thin, so **I measured it directly.**
Method [FACT]: 1,800 documents sampled from `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, via the
HF datasets-server `/rows` endpoint at 18 random offsets in [0, 9M), reading the dataset's own
**`token_count`** column (GPT-2 tokenizer, provided by the dataset authors — so these are real token
counts, not estimates).

**FineWeb-Edu document length (n = 1,800):**

| percentile | tokens |
|---|---|
| p1 | 106 |
| p10 | 211 |
| p25 | 344 |
| **p50 (median)** | **622** |
| p75 | 1,047 |
| p90 | 1,827 |
| p95 | 3,072 |
| p99 | 9,238 |
| max observed | 60,202 |
| mean | 1,070 |

| threshold | frac of **documents** longer | frac of all **tokens** living in those docs |
|---|---|---|
| > 1,024 | 26.1% | **65.9%** |
| > 2,048 | 8.6% | **43.1%** |
| > 4,096 | 3.6% | **30.7%** |
| > 8,192 | 1.3% | **18.3%** |
| > 16,384 | 0.33% | **8.4%** |
| > 32,768 | 0.06% | **3.1%** |

**[INFER] This resolves the concern, and the two columns tell opposite-sounding but compatible
stories.** By *document* count FineWeb-Edu is overwhelmingly short (median 622 tokens; only 1 in 300
documents reaches 16K). But because length is heavy-tailed, **8.4% of all tokens sit in documents
longer than 16K and 3.1% in documents longer than 32K.** On the `sample-100BT` subset that is
~8.4B tokens in ≥16K documents and ~3.1B in ≥32K documents — **which is more than enough for a
long-context extension stage** (Waleffe used 50B tokens for 16K/32K/128K extension at 8B; ABF used
400B at 7B-70B; both far larger models). It is *not* enough to train natively at 32K without either
massive repetition or packing unrelated documents together.

**Caveat on my Comma measurement [FACT + flag]:** sampling 900 docs from
`common-pile/comma_v0.1_training_dataset` gave a median of only 136 words and a **max of 333 words**,
with 0% above 4K tokens. That distribution is implausibly narrow for an 8TB multi-source corpus, and
inspecting a row showed arXiv-abstract-like text — **the datasets-server offsets I hit almost certainly
landed inside one homogeneous shard (abstracts), so this is NOT a valid estimate of Comma's overall
length distribution.** Flagging as **[UNKNOWN]** rather than reporting it as a corpus property.

**[UNKNOWN] I did not obtain measured length distributions for DCLM-baseline, SlimPajama, Dolma, or
The Pile** (datasets-server returned no size/rows for several, and SlimPajama requires auth). Based on
composition rather than measurement: **[INFER]** DCLM-baseline and C4 will look like FineWeb-Edu or
shorter (all CommonCrawl-derived; C4 shorter still due to its aggressive line filtering);
SlimPajama/Pile/Dolma will have materially fatter tails because Books3/PG-19/arXiv/FreeLaw contribute
whole books and papers.

### 6.3 Tokenizer choice

| Tokenizer | Vocab | Notes |
|---|---|---|
| **LFM2** (`LiquidAI/LFM2-*`) | **65,536** [FACT, config.json] | Power-of-two; 8 languages; ships under the LFM Open License — check terms before redistributing |
| Llama 3 | 128,256 | Very large; huge embedding cost at ≤1B |
| GPT-NeoX-20B | 50,432 (50,254 used) | The Pile-native; Mamba's 300B-token runs use it |
| GPT-2 | 50,257 | Mamba/Mamba-2 **scaling-law** runs use it; maximal comparability with that literature |
| Mistral / Llama 2 | 32,000 | **Samba and Gated DeltaNet both use 32K** |
| OLMo 2 | cl100k-based (~100,278) | Apache-2.0 vocabulary; OLMo-core `TokenizerConfig.dolma2()` |
| SmolLM2 | 49,152 | Purpose-built for small models |

**[FACT] Scaling Laws with Vocabulary (Tao et al., arXiv 2407.13623):**
- Trained models **33M to 3B** params (non-vocabulary params 33M-1.13B) on **up to 500B characters**,
  with vocab sizes swept over `4096, 6144, 8192, 10240, 16384, 24576, 32768, 48128, 64512, 96256`.
- Three prediction methods (IsoFLOPs, derivative, parametric loss fit) converge; optimal vocabulary
  **parameters** scale as `Nv_opt ∝ Nnv^γ` with **γ ≈ 0.83 < 1** — i.e. **vocabulary should grow
  *slower* than the rest of the model.**
- Predicted optima (their table, at compute-optimal token counts): **0.1B non-vocab params → V_opt
  ≈ 37-43K**; 0.2B → ~60-67K; 0.4B → ~81-91K; 0.9B → ~142-154K; 1.8B → ~212-231K; 3.0B → ~237-258K;
  6.3B → ~356-389K. Headline example: *"we predict that the optimal vocabulary size of Llama2-70B
  should have been at least 216K, 7 times larger than its vocabulary of 32K"*, and increasing 32K→43K
  improved ARC-Challenge from **29.1 to 32.0**.
- **The crucial caveat for a data-limited study [FACT]:** with `Nnv = 302M` fixed, *"when available
  data is the bottleneck, the optimal vocabulary size decreases empirically, i.e. 16K → 10K. This is a
  mechanism to prevent over-fitting. Conversely, when training on excessive amounts of data... the
  optimal vocabulary size increases, i.e. 16K → 24K."*

**[INFER] Embedding-parameter arithmetic (mine), the decisive practical consideration:**

Embedding matrix = `V × d`. Untied input+output = `2 × V × d`.

| V | d=768 | d=1024 | d=1536 | d=2048 |
|---|---|---|---|---|
| 32,000 | 24.6M | 32.8M | 49.2M | 65.5M |
| 49,152 | 37.7M | 50.3M | 75.5M | 100.7M |
| 50,257 | 38.6M | 51.5M | 77.2M | 102.9M |
| **65,536 (LFM2)** | **50.3M** | **67.1M** | **100.7M** | **134.2M** |
| 100,278 (OLMo 2) | 77.0M | 102.7M | 154.0M | 205.4M |
| 128,256 (Llama 3) | 98.5M | 131.3M | 197.0M | 262.7M |

**As a fraction of total params — this is the trap:**
- A **150M** model at d=768: V=32K tied = 16% of params; V=65,536 tied = **34%**; V=65,536 **untied =
  67%** — the "architecture" would be a minority of the model, and any mixer-ratio effect gets diluted
  into noise.
- A **350M** model at d=1024: V=32K tied = 9%; V=65,536 tied = **19%**; untied = **38%**.
- A **750M** model at d=1536: V=32K tied = 6.6%; V=65,536 tied = **13%**; untied = **27%**.

**[INFER] Conclusions:**
1. **Tie embeddings.** LFM2-2.6B itself sets `tie_embedding: true` [FACT]. Untied doubles the cost for
   no benefit at this scale and wrecks the param budget.
2. **Use a 32K vocabulary, not LFM2's 65,536.** Tao et al. predict ~37-43K optimal for 0.1B *non-vocab*
   params at compute-optimal data — but this experiment is deliberately data-limited relative to
   compute-optimal-for-vocab, and their own 302M experiment says the optimum *falls to 10-16K* when data
   is the bottleneck. 32K keeps the "architecture fraction" high (where the signal is), matches **Samba
   and Gated DeltaNet exactly** (so their published numbers are comparable), and is what Llama 2 /
   Mistral use.
3. **Deviating from LFM2's 65,536 is a documented deviation, not an error** — say so explicitly in the
   write-up, and note that it makes *your* LFM2 arm not parameter-identical to shipped LFM2 (that is
   fine and desirable: all arms share your vocab).
4. **[INFER] If maximum comparability to the linear-attention literature is the priority, use the
   Llama-2 32K tokenizer** (Samba, GDN). If OLMo-core convenience is the priority,
   `TokenizerConfig.dolma2()` works but costs ~100K vocab → use tied embeddings and accept the larger
   embedding share, or train a 32K tokenizer on your corpus.

### 6.4 Corpus recommendation

**Primary: FineWeb-Edu (`sample-100BT`), ODC-By 1.0.** Reasons: (a) **ODC-By is genuinely permissive**,
unlike Pile/SlimPajama's Books3 exposure; (b) **Gated DeltaNet's 100B-token protocol and `flame`'s
340M recipe both use exactly this subset** [FACT], so the closest published comparables share the
corpus; (c) 3.6% of docs / **30.7% of tokens exceed 4K** and **8.4% of tokens exceed 16K** (measured
§6.2) — sufficient for a 4K base plus a long-context stage; (d) it is a single-source corpus, so no
mixture-weight confound across arms; (e) trivially available (`sample-10BT` for pilots, `sample-100BT`
for the real runs).

**Long-context supplement (needed): filter FineWeb-Edu by `token_count` ≥ 8192 and/or add PG-19.**
[INFER] FineWeb-Edu ships `token_count` per row, so building a long-document subset is a cheap filter,
not a new download. PG-19 (public domain, whole books) is the cleanest way to get genuinely ≥32K
documents. **[FACT] But note the ABF finding in §7 that long *data* matters much less than expected.**

**If strict open licensing is a hard requirement:** Common Pile / Comma v0.1 (arXiv 2506.05209) — but
**[UNKNOWN]** verify its true length distribution yourself; my sample was not valid (§6.2).

**Do not use The Pile or SlimPajama as the primary** despite their excellent comparability to
Mamba/Samba, because of Books3 licensing. **[INFER] If you want Samba's ppl numbers to be directly
comparable, run one *additional* SlimPajama arm rather than making it the base corpus.**

---

## 7. Long-context training protocol (16K / 32K)

### 7.1 Native-long vs short-then-extend — the evidence

**[FACT, Samba Table 5]** Llama-2-SWA 438M, window fixed 2048, ~2M tokens/step (so longer sequences
mean proportionally smaller batch), SlimPajama ppl at eval lengths 2K/4K/8K/16K:

| Train seq len | ppl@2048 | ppl@4096 | ppl@8192 | ppl@16384 |
|---|---|---|---|---|
| **4096** | **11.87** | **11.16** | **10.69** | **10.61** |
| 8192 | 11.98 | 11.26 | 10.79 | 10.69 |
| 16384 | 12.37 | 11.63 | 11.12 | 11.02 |
| 32768 | 12.94 | 12.46 | 11.96 | 11.86 |

*"Longer training sequences hurt at every eval length; the best sequence-length-to-window ratio
observed is 2, i.e. 4096 training length."*
**[INFER] Training natively at 32K cost +1.07 ppl @4K and +1.25 ppl @16K versus training at 4K, at
equal total tokens. Some of that is the batch-size confound (32K sequences at fixed tokens/step means
1/8 the sample count), but the direction is unambiguous: at ~430M scale, native-long training is a
substantial quality regression.** → **Short-then-extend, decisively.**

**[FACT, "Effective Long-Context Scaling" / ABF, arXiv 2309.16039]** Llama-2 continual pretraining to
32,768 tokens: 7B/13B trained with 32,768-token sequences, *"for a total of 400B tokens over 100,000
steps"*, keeping *"the original LLAMA 2 architecture nearly intact"*. The critical mechanism:
increasing the RoPE **base frequency b from 10,000 to 500,000**, because the baseline
*"was unable to effectively attend beyond 4,000 - 6,000 tokens even after extensive long-context
continual pretraining"* due to RoPE's decay on distant tokens. Table 5 (all samples 32,768 tokens):
**RoPE ABF 6.323 / XPOS ABF 6.331** on their two validation sets — ABF variants win, and they conclude
*"long context continual pretraining is more efficient and similarly effective"* compared to training
long from scratch.

**[FACT, the most surprising ABF result — and it directly de-risks §6's corpus concern]** Two data
ablations: (1) remove long text from the mix and continue-pretrain on mostly short documents;
(2) up-weight long text. Result: *"even with most of the long texts removed, the model can still obtain
most of the performance gain over LLAMA 2"*, and *"there is no clear and consistent advantage as we
greatly increase the long data ratio"*. Conclusion: *"adjusting the length distribution of the pretrain
data does not provide major benefits... long-context LLMs can be effectively trained even with very
limited long data and the improvements... mostly come from the quality of the data itself, instead of
the length distribution difference."* They note the *"ablation experiments suggest that having abundant
long [data]"* matters less than *"the quality"*.

> **[INFER] Taken together with §6.2's measurement, the long-context data worry largely dissolves:
> FineWeb-Edu's 8.4%-of-tokens-above-16K is comfortably sufficient, and ABF's ablation says even far
> less would do. Spend the effort on the positional-encoding choice, not on hunting long corpora.**

**[FACT, YaRN arXiv 2309.00071]** Extends context *"requiring 10x less tokens and 2.5x less training
steps than previous methods"*, reaching SOTA *"after fine-tuning on less than ~0.1% of the original
pre-training data"*; defines scale factor `s = L'/L`; notes prior PI-style fine-tunes *"achieve a
scaling factor of roughly s = 8 before the LLM's outputs start to degrade, even after fine-tuning"*;
supports dynamic scaling `s = max(1, l'/L)` per forward pass. PI (arXiv **2306.15595**) is the linear
position-interpolation baseline; LongRoPE extends further via non-uniform search.

**[FACT] What the hybrids actually do — and they disagree with the RoPE-scaling literature:**
- **Waleffe et al. (2406.07887):** **no position embeddings at all.** Table 5 at 840M/1.1T: at 16K
  eval, **no-RoPE 53.43 > RoPE-10K 52.61 > RoPE-500K 51.52** — RoPE actively hurts, and *more* ABF-style
  θ hurts more. Long-context extension = **+50B tokens**, LR 3e-5 → 3e-6.
- **Jamba:** no-PE ≈ RoPE at 1.3B/250B (log-probs identical at −0.516/−0.623/−0.299).
- **Kimi Linear:** full-MLA layers use **NoPE**.
- **OLMo-core's own hybrid:** `OLMo-hybrid-7B-long-context.py` **drops RoPE ("DroPE")** at the start of
  long-context training — `attn_block.sequence_mixer.replace(rope=None)` — extending to
  `sequence_length = 65536` for **100B tokens** at LR ≈ 2.07e-4 with `LinearWithWarmup`, Ulysses CP
  degree 2, and `NumpyPackedFSLDatasetConfig`.
- **LFM2 (contrast):** keeps RoPE with **θ = 1e6** and `max_position_embeddings = 128000`, trained to
  32,768 context [FACT].

> **[INFER] This is the sharpest actionable divergence in the whole document. In hybrids where
> recurrent/conv layers already encode position, four independent groups (NVIDIA, AI21, Moonshot, AI2)
> found positional encoding in the attention layers to be unnecessary or harmful, and AI2's production
> recipe *removes* RoPE specifically to extend context. LFM2 does the opposite. Since a mostly-LIV
> stack has *more* position-encoding conv layers per attention layer than LFM2 does, the NoPE/DroPE
> route is well-motivated — and "RoPE θ=1e6 vs NoPE/DroPE at matched ratio" is a cheap, high-value arm
> that also happens to be the mechanism most likely to explain any long-context result.**

### 7.2 Data packing and document masking

**[FACT, "Fewer Truncations Improve Language Modeling", arXiv 2404.10830]** The standard
concatenate-then-split approach *"compromises data integrity—it inevitably breaks many documents into
incomplete pieces, leading to excessive truncations"*, and *"truncation reduces [context], thus making
models more prone to hallucination"*. **Best-fit Packing** *"packs documents into training sequences
[and] eliminates unnecessary truncations while retaining the same training efficiency as
concatenation"* — *"only documents beyond model's context length need to be segmented"*. Measured
gains: **+4.7% reading comprehension, +16.8% context following, +9.2% program synthesis, and closed-domain
hallucination reduced by up to 58.3%** (relative).

**[FACT] Intra-document attention masking** is supported natively in the recommended stack: OLMo-core's
`CausalConv1d.forward(x, cu_seqlens=...)` takes cumulative sequence lengths, `GatedDeltaNet` threads
`cu_doc_lens` into both the convs and the chunked kernel, `NumpyPackedFSLDatasetConfig` exists, and
HF's `Lfm2ShortConv` passes `seq_idx` to `causal_conv1d_fn`. `flame` exposes
`--training.varlen` with separate `--training.seq_len` and `--training.context_len`.

> **[INFER] Packing/masking is not a detail here — it is a confound with teeth.** A K=3 causal conv
> that bleeds across a document boundary is a *different operator* than one that respects it, and the
> bleed rate scales with how many documents are packed per sequence. With FineWeb-Edu's median 622
> tokens, a 4K sequence contains ~6 documents and a 32K sequence ~50, so boundary effects are frequent.
> If document masking is on for the attention arms but the conv silently bleeds (or vice versa),
> the comparison is broken. **Fix `cu_seqlens`/document masking ON for every arm and every mixer, state
> it, and assert it in a unit test.**

### 7.3 Recommended long-context protocol

**[INFER], synthesizing §7.1-7.2:**
1. **Pretrain at 4,096** for the full token budget. Samba Table 5 shows this dominates native-long at
   this scale, and every published small-model recipe (Samba, GDN, Waleffe, OLMo 2) uses 4K.
2. **Extend to 16K, then 32K** in a short continued-pretraining stage. Budget **~2-5% of total tokens**
   — [INFER] scaled down from Waleffe's 50B-on-3.5T (~1.4%) and ABF's 400B (larger models); at a 15B
   base budget that is ~0.5-1B tokens, cheap. Use a **linear warmup + low LR** (ABF/Waleffe both drop
   to ~1/10 of pretrain LR; OLMo-core's long-context script uses `LinearWithWarmup` at ~2e-4).
3. **Positional encoding: run both.** Default arm = **NoPE/DroPE** (drop RoPE at the start of the
   extension stage, following OLMo-core's shipped recipe and Waleffe's Table 5). Control arm =
   **RoPE θ=1e6** (LFM2's choice) or ABF θ=500K. This is one extra short run and it is the mechanism
   most likely to drive the long-context result.
4. **Document masking / best-fit packing ON everywhere**, `cu_seqlens` threaded through convs and
   attention, verified by test.
5. **Long data:** filter FineWeb-Edu by `token_count ≥ 8192` for the extension stage; optionally mix in
   PG-19. Per ABF, do **not** over-invest here.
6. **Evaluate at 4K / 8K / 16K / 32K** (Samba's protocol) **plus a recall probe** — needle/passkey or
   MAD-style in-context recall. Perplexity alone will hide the effect (§2.2); Samba Table 6's blowup and
   Hymba's 20.75-point recall gap are both invisible in short-context ppl.

---

## 8. Compute budgeting

### 8.1 Assumptions (state these in the paper)

- **FLOP model: `C ≈ 6ND`** (Kaplan et al. 2020; also used by OLMo 2 Figure 1, which writes
  *"pretraining FLOPs (≈ 6 × training tokens × model size)"* [FACT]). N = non-embedding params is the
  stricter convention; I use total params below, which is slightly conservative.
- **6ND omits the attention quadratic term** `~12 · n_attn_layers · d · L_ctx` per token. §8.4 shows
  this is *not* negligible here.
- **Peak BF16 dense throughput** (no sparsity) [FACT, vendor specs]: **A100 SXM = 312 TFLOP/s**,
  **H100 SXM = 989 TFLOP/s**.
- **MFU assumptions [INFER]: 40% on A100, 35% on H100.** Rationale: 35-45% is typical for
  well-tuned small-model training; H100 MFU is habitually *lower* than A100 in percentage terms because
  its peak is much higher relative to memory bandwidth, and small models at ≤1B are more
  bandwidth/launch-bound. **These are the least certain numbers in this section — measure your own MFU
  in the first hour and rescale everything.**
- **[INFER] Hybrid arms will have somewhat lower MFU than the pure-GQA arm** because the depthwise conv
  is memory-bound and the fused conv kernel is small; expect maybe 0.8-0.95× the transformer's MFU.
  This *helps* wall-clock (fewer FLOPs) but *hurts* utilization — so report tokens/sec alongside FLOPs.
- Ignores dataloading stalls, checkpoint writes, evaluation, and restarts. **[INFER] Add 15-25%
  overhead for a realistic plan.**

### 8.2 Total training FLOPs (6ND)

| Model | 10B tokens | 20B tokens | 50B tokens |
|---|---|---|---|
| 150M | 9.00e18 | 1.80e19 | 4.50e19 |
| 350M | 2.10e19 | 4.20e19 | 1.05e20 |
| 750M | 4.50e19 | 9.00e19 | 2.25e20 |

For scale reference [FACT]: MAD's IsoFLOP groups span 4e18-8e19, so **a 150M/10B run (9e18) sits
squarely inside MAD's studied range**, and a 750M/50B run (2.25e20) exceeds their largest group.

### 8.3 Wall-clock estimates

**Hours** = `6ND / (n_gpu × peak × MFU)`, at MFU 40% (A100) / 35% (H100):

| Model | Tokens | A100×1 | A100×4 | A100×8 | H100×1 | H100×4 | H100×8 |
|---|---|---|---|---|---|---|---|
| **150M** | 10B | 20.0 h | 5.0 h | 2.5 h | 7.2 h | 1.8 h | **0.9 h** |
| 150M | 20B | 40.1 h | 10.0 h | 5.0 h | 14.4 h | 3.6 h | 1.8 h |
| 150M | 50B | 100.2 h | 25.0 h | 12.5 h | 36.1 h | 9.0 h | 4.5 h |
| **350M** | 10B | 46.7 h | 11.7 h | 5.8 h | 16.9 h | 4.2 h | **2.1 h** |
| 350M | 20B | 93.5 h | 23.4 h | 11.7 h | 33.7 h | 8.4 h | 4.2 h |
| 350M | 50B | 233.7 h | 58.4 h | 29.2 h | 84.3 h | 21.1 h | 10.5 h |
| **750M** | 10B | 100.2 h | 25.0 h | 12.5 h | 36.1 h | 9.0 h | 4.5 h |
| 750M | 20B | 200.3 h | 50.1 h | 25.0 h | 72.2 h | 18.1 h | 9.0 h |
| 750M | 50B | 500.8 h | 125.2 h | 62.6 h | 180.6 h | 45.1 h | 22.6 h |

Same numbers in **days**, for the two most likely configurations:

| Model | Tokens | A100×1 | A100×8 | H100×8 |
|---|---|---|---|---|
| 150M | 10B | 0.83 d | 0.10 d | 0.04 d |
| 150M | 20B | 1.67 d | 0.21 d | 0.08 d |
| 350M | 20B | 3.90 d | 0.49 d | 0.18 d |
| 350M | 50B | 9.74 d | 1.22 d | 0.44 d |
| 750M | 20B | 8.35 d | 1.04 d | 0.38 d |
| 750M | 50B | 20.87 d | 2.61 d | 0.94 d |

**[INFER] Budget reading:** a **150M / 10B** screening run is **~2.5 h on 8×A100 or ~1 h on 8×H100** —
so a **12-arm sweep at that scale is ~30 A100-hours × 8 GPUs ≈ 1.25 days of an 8-GPU node**, which is
the right unit of work for the ranking stage. A **350M / 20B** confirmation is ~12 h on 8×A100, so 4
arms ≈ 2 days. A **750M / 50B** headline run is ~2.6 days on 8×A100. **The whole program is feasible on
a single 8-GPU node in ~2-3 weeks**, or on 1×A100 only if restricted to 150M (a 350M/50B run would be
~10 days on one A100, which is where WSD forking pays for itself).

### 8.4 The attention term that 6ND hides — and why it constrains the design

**[INFER, my arithmetic]** Attention FLOPs per token per attention layer ≈ `12 · d · L_ctx`
(fwd+bwd across the two `QK^T` and `AV` matmuls). As a fraction of 6ND:

| Config | ctx 4,096 | ctx 16,384 | ctx 32,768 |
|---|---|---|---|
| 150M, d=768, 12L, **2 attn (17%)** | 8.4% | 33.6% | 67.1% |
| 150M, d=768, 12L, **4 attn (33%)** | 16.8% | 67.1% | **134.2%** |
| 350M, d=1024, 16L, **3 attn (19%)** | 7.2% | 28.8% | 57.5% |
| 350M, d=1024, 16L, **6 attn (37.5%, LFM2-like)** | 14.4% | 57.5% | **115.0%** |
| 750M, d=1536, 16L, **3 attn (19%)** | 5.0% | 20.1% | 40.3% |

**[INFER] Three consequences that should shape the experiment:**
1. **At 4K, 6ND is a decent approximation** (5-17% understatement) and compute-matching across arms via
   parameter-matching is roughly sound.
2. **At 32K, the attention term exceeds the entire 6ND estimate** for LFM2-like ratios at 150-350M.
   **This means "compute-matched" is meaningless at long context unless you count it** — a
   37.5%-attention arm and a 12.5%-attention arm at identical parameter counts differ by >2× in true
   FLOPs at 32K. **Use OLMo-core's `num_flops_per_token(seq_len)` (which is seq-len aware and
   implemented per mixer) as the matching currency, not parameter count.** This is the strongest
   practical argument for the codebase choice.
3. **The efficiency story lives at long context, not at 4K.** At 4K, dropping from 37.5% to 12.5%
   attention saves ~7% of FLOPs — unexciting. At 32K it saves ~58% of a 115% overhead, i.e. the model
   gets roughly **2× cheaper**. So the mostly-LIV thesis should be *framed and evaluated at 16K/32K*,
   where it is a large effect, not at 4K where it is noise.

### 8.5 Parameter matching: LIV costs more than GQA

**[INFER, my arithmetic, no biases]** Per sequence-mixing layer:
- **LIV** = `in_proj(d→3d) + out_proj(d→d) + depthwise conv(3d)` = `4d² + 3d`
- **GQA** = `Q(d²) + K(d·n_kv·h_d) + V(d·n_kv·h_d) + O(d²)`
- **MHA** = `4d²`

| d | heads | kv heads | head_dim | LIV | GQA | MHA | **LIV/GQA** |
|---|---|---|---|---|---|---|---|
| 768 | 12 | 4 | 64 | 2,361,600 | 1,572,864 | 2,359,296 | **1.50×** |
| 1024 | 16 | 8 | 64 | 4,197,376 | 3,145,728 | 4,194,304 | **1.33×** |
| 1536 | 12 | 4 | 128 | 9,441,792 | 6,291,456 | 9,437,184 | **1.50×** |
| 2048 | 32 | 8 | 64 | 16,783,360 | 10,485,760 | 16,777,216 | **1.60×** |

**[INFER] Important and easy to get wrong:** a LIV layer ≈ an **MHA** layer in parameters (both ~4d²),
but **1.33-1.60× a GQA layer**, because GQA shrinks K and V. So **swapping GQA→LIV layer-for-layer
*increases* parameters** — the more LIV-heavy the stack, the bigger the model. Any naive "mostly-LIV vs
all-GQA" comparison at equal layer count silently gives the LIV model 10-25% more parameters and would
be a fatal confound.

**Fixes, in order of preference [INFER]:**
1. **Shrink `d_model` (and thus heads) in the LIV-heavy arms until total params match** — exactly what
   AI2 did for OLMo-Hybrid-7B (`REMOVE_HEADS = 2`; `d_model -= REMOVE_HEADS * 128`, keeping
   `d_model/n_heads == 128`). Cleanest, and precedented.
2. **Adjust the SwiGLU FFN width** (LFM2's `block_auto_adjust_ff_dim: true` and `block_multiple_of: 256`
   suggest Liquid does exactly this [FACT: those keys exist; INFER: that is their purpose]).
3. **Adjust layer count** — worst option, since depth is itself a strong confound (MobileLLM's whole
   finding is that depth matters a lot at ≤350M).

**[INFER] Also report, per arm, as first-class metrics:** total params, non-embedding params,
`num_flops_per_token` at each eval length, **total recurrent/conv state size** (MAD Finding 8's
`P* ∝ M^−0.28`), KV-cache bytes at 32K, and measured tokens/sec. Matching on one axis while silently
varying another is the single most common way this class of experiment fails.

---

## 9. Recommended experimental setup

Everything below is **[INFER]** — my synthesis — but each choice cites the [FACT] that motivates it.

### 9.1 Codebase

**Primary: `allenai/OLMo-core`** (Apache-2.0). It uniquely combines a fused `CausalConv1d` short-conv,
a `SequenceMixerConfig` registry with per-mixer `num_params`/`num_flops_per_token`, `block_pattern` +
`block_overrides` per-layer stacking, built-in `WSD`/`WSDS` + a Chinchilla model-ladder configurator,
`WandBCallback`, async resumable checkpointing, and **a released, parameter-matched 3:1 GDN:attention
hybrid with a DroPE long-context stage** (`src/scripts/official/OLMo-hybrid/`) to use as a template.
Add one file, `nn/attention/liv.py` (~150-250 LOC incl. config + equivalence test vs HF `Lfm2ShortConv`).

**Fallback: `fla-org/flash-linear-attention` + `flame`** (MIT) — 40+ mixers (`mamba2`, `gated_deltanet`,
`kda`, `rwkv7`, `gla`, `retnet`, `samba`) and `hybrid.py`'s `HybridAttentionSpec` with literal
`layers: [...]` placement plus per-spec `num_kv_heads`/`window_size`/`rope_theta`. Accept: `flame` last
pushed 2026-04-22, pinned torchtitan commit, no μP, no built-in matching API.

**Pre-screen: `athms/mad-lab`** (6 synthetic tasks, minutes each) — but note it is unmaintained
(2024-12-17) and its "predicts scaling" claim is qualitative, with no published correlation coefficient.

### 9.2 Corpus and tokenizer

- **Corpus: FineWeb-Edu `sample-100BT`** (ODC-By 1.0). Matches Gated DeltaNet's and `flame`'s published
  protocols; single-source (no mixture confound); **measured 30.7% of tokens in docs >4K and 8.4% in
  docs >16K** (§6.2) — enough for the long-context stage.
- **Long-context stage data:** FineWeb-Edu filtered on its own `token_count ≥ 8192` column, optionally
  + PG-19. Per ABF's ablation, do not over-invest — *"long-context LLMs can be effectively trained even
  with very limited long data."*
- **Tokenizer: a 32,000-vocab BPE (Llama-2/Mistral), tied embeddings.** Not LFM2's 65,536: at d=768 a
  65,536 tied embedding is **34% of a 150M model** vs 16% for 32K (§6.3), which dilutes exactly the
  architectural signal being measured; and Tao et al.'s own data-limited result at Nnv=302M moves the
  optimum *down* to 10-16K. 32K also matches Samba and Gated DeltaNet for comparability.
  **Document this as a deliberate deviation from stock LFM2.**
- **Fix and record:** tokenizer, corpus shard list, **data order seed**, packing/masking mode. Pythia's
  design (identical data order across all models, public reproduction script) is the standard to hit.

### 9.3 Model sizes and token budgets

| Stage | Size | d_model | Layers | Tokens | Tok/param | Purpose |
|---|---|---|---|---|---|---|
| Screen | MAD 2-block | — | 2 | minutes | — | Rank ratios/placements on 6 synthetics |
| **Rank** | **150M** | 768 | 12 | **10B** | ~67 | All arms, **≥2 seeds** |
| **Confirm** | **350M** | 1024 | 16 | **20B** | ~57 | Top 4-5 arms |
| **Headline** | **750M** | 1536 | 16 | **50B** (fork decays at 20B/35B/50B) | ~67 | Top 2-3 arms + rank-transfer check |

Tokens/param ≈ 55-70 (≈3× Chinchilla) — justified by MiniCPM's 192× compute-optimal finding, MAD
Finding 6 (hybrids gain when over-trained), and comparability with GDN (77×) and Samba (47-77×).
Cost (§8.3, 8×A100 @40% MFU): 150M/10B ≈ 2.5 h/arm; 350M/20B ≈ 12 h/arm; 750M/50B ≈ 2.6 d/arm.

### 9.4 Schedule and optimizer (identical for every arm)

- **AdamW**, β = (0.9, 0.95) at batch ≥0.5M tokens (β₂ = 0.99 if smaller, per OLMo-core's ladder),
  **weight decay 0.1**, **no weight decay on embeddings**, **grad clip 1.0**, BF16, no dropout.
- **WSD**, decay = **10% of tokens** (MiniCPM: *"having a decay of 10% of the total tokens is
  sufficient... while a decay of 2.5% falls short"*), warmup ≈ **1 token per parameter**.
  **Fork multiple decays from one stable phase** to get several token budgets per arm cheaply — the
  main reason to prefer WSD over cosine for a many-variant sweep.
- **Peak LR:** pick *one* family and hold it. Either OLMo-core's ladder formula
  `lr = 0.0047·(N/108M)^(−1/3) / 2` (≈2.1e-3 @150M, 1.6e-3 @350M, 1.2e-3 @750M) **or** the
  Samba/GDN convention (4e-4). Do not blend; note the 3-5× discrepancy in the writeup.
- **Batch:** `round(2048 · 160 · (N/108M)^(2/3))` tokens ≈ 406K @150M, 715K @350M, 1.19M @750M
  (re-derive the sample count for seq len 4096).
- **Stability:** RMSNorm, QK-norm, **z-loss 1e-5**, reordered norm, init std 0.02 or `fan_in`;
  `SkipStepAdamW` to survive non-finite steps unattended.
- **Seq len 4096** for pretraining; **document masking / best-fit packing ON for every arm**, with
  `cu_seqlens` threaded through both convs and attention (assert in a test — a bleeding K=3 conv is a
  different operator, and FineWeb-Edu's 622-token median means ~6 docs per 4K sequence).
- **Long-context stage:** extend 4K → 16K → 32K using ~2-5% of total tokens at ~1/10 LR with linear
  warmup. Run **two positional variants: NoPE/DroPE (default, per Waleffe Tbl 5, Jamba, Kimi Linear,
  and OLMo-core's shipped DroPE script) and RoPE θ=1e6 (LFM2's choice)**.

### 9.5 The baseline list

Every arm: identical tokenizer, corpus, data order, optimizer, schedule, seq len, packing/masking, and
**matched total params (± <1%, via `d_model`/heads or FFN width) and reported
`num_flops_per_token`.** Placement default = evenly dispersed, **never layer 0, never the last layer**
(the one placement rule all papers agree on).

| # | Arm | Composition | Why it must be there |
|---|---|---|---|
| **1** | **All-GQA transformer** | 100% GQA + SwiGLU | The essential control. Mamba-2 Tbl 2's Transformer++ (8.68) and Samba Tbl 3's Llama-2 (11.14) show pure attention *loses* to hybrids at this scale — if your transformer doesn't behave, the harness is wrong |
| **2** | **Stock LFM2** | 16L, conv at all but `[2,5,8,10,12,14]` → **37.5% GQA**, K=3 double-gated conv | The published reference point; the thing being modified. Reproduce its exact irregular, late-heavy pattern via `block_overrides` |
| **3** | **Mostly-LIV @ ~12.5%** | 2/16 attn, evenly dispersed | **The hypothesis.** Targets the Mamba-2 (10%) / Waleffe (8%) / Granite (10%) consensus optimum |
| **4** | **Mostly-LIV @ ~25%** | 4/16 attn | MAD Finding 7's compute-optimal 25%; Qwen3-Next / Kimi Linear / Jamba-1:3 ratio |
| **5** | **Mostly-LIV @ ~6%** | 1/16 attn | Tests the low edge; Jamba shows ICL survives at 1-in-8, so 1-in-16 is the real frontier |
| **6** | **Pure LIV, 0% attention** | all conv | The floor. Hymba's Mamba row (recall 19.23 vs 39.98) predicts recall collapse — this arm is what makes the "attention is necessary" claim quantitative in the *conv* regime |
| **7** | **Mamba-2 hybrid @ ~8%** | Mamba-2 mixers, Waleffe ratio | The strongest published recurrent hybrid; isolates **mixer type** at matched ratio. Tests MAD Finding 8: does LIV's tiny `d×2` state lose to Mamba-2's large state? |
| **8** | **Gated DeltaNet hybrid 3:1** | GDN + attn, `block_pattern=["gdn","gdn","gdn","attn"]` | **Free** — already in OLMo-core, and AI2's own param-matched 7B recipe. 2025 SOTA linear mixer |
| **9** | **Samba-style: LIV + SWA** | conv + sliding-window (w=2048) attention | Samba Tbl 3 (421M: 10.06 vs Llama-2 11.14) is the best small-scale hybrid number published, and Tbl 6 says SWA — not full attention — is what extrapolates |
| **10** | **LIV + full-attention placement study** | ratio fixed at #3's, placement ∈ {early-heavy, uniform, late-heavy (LFM2's), first/middle/last (Hymba)} | Resolves the one genuine literature disagreement: Mamba-2 fn.6 says placement barely matters in-distribution, Samba Tbl 6 says it decides length extrapolation |
| **11** | **NoPE/DroPE vs RoPE θ=1e6** | #3 with both | Four labs found PE unnecessary/harmful in hybrids; LFM2 disagrees. Cheap, and likely explains any long-context result |
| *12 (optional)* | **Parallel-fusion arm** | attn + conv heads in the *same* layer (Hymba/Falcon-H1 style) | Hymba Tbl 1 A→B: parallel beat sequential by +4.74 recall at 300M/100B. This is the main threat to the layer-interleaved framing — include it or name it as a limitation |
| *13 (optional)* | **Short-conv-everywhere control** | Samba's SC transferred to both mixer types | Samba [FACT]: adding short conv to *both* mixers *"produces negative results."* A mostly-LIV stack is short-conv-everywhere by construction — **this is the pre-registered main risk** |

Arms 1-6 are the core (all conv-based, one variable: ratio). 7-9 are mixer-type controls, mostly free in
OLMo-core. 10-11 are cheap, high-value sub-studies. 12-13 are the honest threats.

### 9.6 Evaluation — the most important design choice

**Do not rank on perplexity alone.** Mamba-2 Table 2's entire 4%→23% attention range spans **0.06 ppl**,
and Samba reports **±0.3%** run-to-run noise — so the ratio question is only ~2× above the noise floor
in ppl. Meanwhile **Hymba measured a 20.75-point recall gap** (19.23 vs 39.98) and **Jamba a 35.3-point
format-following gap** (IMDB 48.8 vs 84.1) between pure-recurrent and hybrid models whose perplexities
were nearly identical.

So: (a) **validation ppl at 4K/8K/16K/32K** (Samba's protocol, catches extrapolation blowups);
(b) **synthetic recall** — MAD's in-context / fuzzy / noisy recall + selective copying, and
needle/passkey at 32K; (c) **format-following / ICL** few-shot tasks in Jamba's spirit;
(d) **≥2 seeds on every arm with an explicitly reported noise band**; (e) **efficiency**: measured
tokens/sec, `num_flops_per_token` at each length, KV+conv state bytes at 32K; (f) **total state size**
per arm, to test MAD's `P* ∝ M^−0.28`.

### 9.7 Principal risks

1. **The flat basin.** Published data says 4-23% attention is within noise in ppl. **Mitigation:**
   power the study on recall metrics, not ppl; multiple seeds; report the noise band first.
2. **Short-conv-everywhere may genuinely lose.** Samba [FACT] found SC on both mixers *"produces
   negative results"*, and MAD Finding 8 (`P* ∝ M^−0.28`) predicts a K=3 conv's tiny state is a
   liability. **A negative result is a real, publishable finding — pre-register it.**
3. **Parameter-matching confound.** LIV costs **1.33-1.60× a GQA layer** (§8.5); naive layer-swapping
   inflates the LIV arms. **Mitigation:** OLMo-core's `num_params`/`num_flops_per_token` + the
   `REMOVE_HEADS` pattern; report both param and FLOP accounting per arm.
4. **Compute-matching breaks at long context.** At 32K the attention term is 57-134% of 6ND (§8.4), so
   arms that are param-matched are *not* compute-matched. **Mitigation:** match on
   `num_flops_per_token(seq_len)`, and frame the efficiency claim at 16K/32K where it is large.
5. **Parallel fusion may dominate the ratio question** (Hymba +4.74 recall). **Mitigation:** arm 12, or
   an explicit limitation.
6. **Absolute numbers won't match published work** (OLMo-core's reordered-norm/QK-norm/z-loss/vocab
   differ from the Mamba/`fla` lineage). **Mitigation:** state that it is an internally-controlled
   study; optionally add one SlimPajama arm for a bridge to Samba's table.
7. **No LFM2 paper exists** — no published ratio ablation, recipe, or data mix. **Everything about
   LFM2 in this document comes from `config.json`, the HF model card, and the transformers
   implementation.** That is also the opportunity: the ablation Liquid never published is the
   contribution.

### 9.8 Total compute ask

At 8×A100 (40% MFU assumed) and 15-25% overhead:
- Screening (MAD): negligible GPU.
- Rank stage: 12 arms × 2 seeds × 150M/10B × 2.5 h ≈ **60 h ≈ 2.5 days**.
- Confirm stage: 5 arms × 350M/20B × 12 h ≈ **60 h ≈ 2.5 days**.
- Headline: 3 arms × 750M/50B × 2.6 d ≈ **8 days**.
- Long-context stages (~3% tokens) + placement/PE sub-studies: **~2 days**.
- **Total ≈ 15-16 days on one 8×A100 node (≈3,000 A100-hours), or ~5-6 days on 8×H100.**
Halve it by dropping the 750M tier and making 350M the headline — the published comparables
(Samba 421M, Mamba-2 350M, GDN 400M/500M) are all at that scale anyway.

