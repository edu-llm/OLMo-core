# Low-Rank Factorization of Gates in Sequence-Mixing Layers

Research dossier supporting the proposed LFM2 "LIV" gate-factorization experiment.

**Date:** 2026-07-30
**Scope:** prior art on low-rank gating; from-scratch low-rank pretraining evidence; init/optimization
of factorized layers; effective-rank diagnostics; parameter-count vs latency; competing controls.

**Convention used throughout:** claims sourced to a paper are marked with a title + URL.
Statements that are my own derivation or inference are explicitly labelled
**[DERIVATION]** or **[REASONING]**. Places where the literature is silent are labelled
**[GAP]** — several of these are load-bearing for the novelty claim.
**[MEAS]** marks original measurements I ran on released LFM2 checkpoints and on hardware; these are
reproducible and the scripts are short enough to check into the experiment repo.

125 unique sources cited.

---

## Executive summary — the eight findings that should change the design

1. **The motivating premise is falsified as stated.** [MEAS §4.4-4.5] Trained LFM2-350M gate matrices
   have effective rank **771-790 of 1024** (~77% of full rank) and are **statistically
   indistinguishable from the value stream** (790.1 vs 790.5). Activation weighting only moves it to
   ~748. Reframe from "gates are low-rank" to "gates *tolerate* being low-rank" (§7.1).
2. **But the narrow claim is well-supported by production systems.** GLA ships rank 16, Mamba
   r = d_inner/32, RWKV-6 rank 64, RWKV-7 ~1.8·sqrt(d) — all from-scratch pretrained, none reporting a
   quality cost (§1). Tolerance and intrinsic low-rankness are different claims; only the former has
   evidence.
3. **Naive from-scratch low-rank fails catastrophically — but only when applied globally.** [§2.4]
   GaLore's plain `W=BA` baseline: 78.18 vs 34.06 ppl at 60M, **142.53 vs 15.56 at 1B**. The proposal
   escapes this because it touches **7.17% of parameters** and leaves the value/output paths dense
   (§2.8).
4. **The novelty claim is strong and auditable.** [§1.8, §8] Liquid's own STAR search space
   *cannot express* a factorized gate — its channel-mixing options are exactly
   `{Diagonal, Dense, Grouped}`, and the featurizer genome has **no rank field** (verified from LaTeX
   source). No published low-rank gate exists in any gated short-conv block.
5. **Init is not optional and one specific recipe dominates.** [MEAS §5.8] The naive "both factors at
   0.02" init makes the block output **12x too small**; scale errors **square** through the double
   gate. **Adding a full-width gate bias initialized to 1.0 is the single highest-value
   recommendation**: it is the from-scratch analogue of LoRA's unavailable `B=0` trick, and it cuts
   block-output kurtosis from **27.8 to 4.5** (§7.3).
6. **The bias is a CONFOUND.** [MEAS §5.9] It changes conditioning independently of rank, so the
   **dense control must also get it**, or the experiment measures the bias rather than the rank.
7. **The latency case is weak and may be negative.** [§5.3, §5B] The Amdahl ceiling is **8.8%**
   (the SwiGLU MLPs are 68.8% of parameters), and published measurements on **Snapdragon INT8** show
   2x parameter/FLOP cuts producing **0-12% slowdowns**. A ~5-8% predicted gain sits *inside* the
   measured slowdown band. Make the microbenchmark a **pre-training gate** and drop wall-clock as a
   headline claim.
8. **Four controls are mandatory, and all are simpler than the proposal.** [§6.8] Narrower model at
   matched params; gate/featurizer sharing (Liquid's own incumbent method, evolutionarily selected for
   gated convs); diagonal gate (2048x cheaper); and dense+bias as the confound control. Grouped /
   block-diagonal is the strongest systems competitor and is **likely to beat low-rank on latency**.

**Reading order if short on time:** §7 (design implications) → §5B (the latency correction) →
§4.4-4.5 (original measurements) → §1.8 (novelty) → §6.8 (control ranking).

---

## 0. The object under study (verified against released code)

The stock LFM2 short-conv block, verified against
`transformers/models/lfm2/modeling_lfm2.py` (class `Lfm2ShortConv`) and the
`LiquidAI/LFM2-1.2B` `config.json`:

```python
self.conv     = nn.Conv1d(d, d, kernel_size=conv_L_cache, groups=d, bias=conv_bias)  # depthwise
self.in_proj  = nn.Linear(d, 3 * d, bias=conv_bias)
self.out_proj = nn.Linear(d, d,     bias=conv_bias)

BCx     = self.in_proj(hidden_states).transpose(-1, -2)
B, C, x = BCx.chunk(3, dim=-2)
h = B * x                  # pre-conv gate
h = causal_conv1d_fn(h, self.conv.weight.squeeze(1), self.conv.bias)
h = C * h                  # post-conv gate
h = self.out_proj(h)
```

Verified config facts for LFM2-1.2B: `hidden_size = 2048`, `conv_L_cache = 3` (k=3),
`conv_bias = false`, `num_hidden_layers = 16`, `full_attn_idxs = [2,5,8,10,12,14]`
(so 6 GQA + 10 conv blocks), `conv_use_xavier_init = true`, `block_use_xavier_init = true`,
`initializer_range = 0.02`.

Three facts here matter a great deal for the experiment and are easy to miss:

1. **The gates are LINEAR — there is no sigmoid, no SiLU, no softplus on B or C.**
   They are raw projections multiplied elementwise. This is materially different from
   every gate in the SSM/RNN literature surveyed in §1 (all of which pass the gate through a
   bounded or positive nonlinearity). Consequences are developed in §3.5 — in short, the
   usual "gate saturation" failure mode does not apply, but an *unbounded variance
   amplification* failure mode does, and it is worse.
2. **`conv_bias = false`, so `in_proj` and `out_proj` have no bias.** A factorized gate
   therefore has no bias to absorb a scale/offset error, unless one is added (which is a
   design decision to make explicitly — see §3.6).
3. **The stock init is Xavier**, not the `initializer_range=0.02` normal used elsewhere
   (`conv_use_xavier_init: true`). Xavier/Glorot for a `d -> 3d` layer gives
   std = sqrt(2/(d + 3d)) = sqrt(1/(2d)). Any factorized replacement must be scale-matched
   against *that*, not against 0.02. See §3.1.

Proposed modification: keep `x̃` (value) and `out_proj` full width; factorize only the two gate
maps as d -> r -> d.

Parameter accounting (agrees with the project's design doc):

```
stock LIV        : 3d^2 (in_proj) + d^2 (out_proj) + kd  = 4d^2 + kd
factorized gates : 1d^2 (value)  + d^2 (out_proj) + 2*(2dr) + kd = 2d^2 + 4dr + kd
```

At d=2048, k=3, r=128: 16.783M -> 9.443M params (-43.7%).


---

## 1. Prior art on low-rank / factorized gating specifically

This is the strongest part of the literature: **low-rank data-dependent gates are not novel
in the abstract — they are the dominant design in modern linear-attention/SSM blocks.**
What *is* novel is the specific placement (see §1.8).

### 1.1 Mamba — the Δ (dt) projection is literally a d -> r -> d low-rank gate

**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, Gu & Dao.
https://arxiv.org/abs/2312.00752

The authors *themselves* name it a low-rank projection. §3.6, "Parameterization of Δ":

> We defined the selective adjustment to Δ as s_Δ(x) = Broadcast_D(Linear_1(x)) [...] We observe
> that it can be generalized from dimension 1 to a larger dimension R. We set this to be a small
> fraction of D, which uses a negligible number of parameters compared to the main Linear
> projections in the block. We additionally note that the broadcasting operation can instead be
> viewed as another Linear projection [...] this leads to the alternative
> s_Δ(x) = Linear_D(Linear_R(x)), **which can be viewed as a low-rank projection.**

Verified code shapes (`mamba_ssm/modules/mamba_simple.py`):

```python
self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)
```

Note the down-projection is **fused into the same `x_proj` matrix that also emits B and C** —
there is no standalone down-proj module. This is a d_inner -> r -> d_inner factorization of an
input-dependent gate, with the bottleneck factor shared with the B/C projection.

**Rank heuristic: `math.ceil(d_model / 16)`.** Confirmed. Note it is d_model/16, *not* d_inner/16,
so relative to the d_inner-wide gate it produces, the rank ratio is **1/32**.

| model | d_model | d_inner | dt_rank | r/d_inner |
|---|---|---|---|---|
| mamba-130m | 768 | 1536 | 48 | 1/32 |
| mamba-370m | 1024 | 2048 | 64 | 1/32 |
| mamba-790m | 1536 | 3072 | 96 | 1/32 |
| mamba-1.4b | 2048 | 4096 | 128 | 1/32 |
| mamba-2.8b | 2560 | 5120 | 160 | 1/32 |

**Initialization — this is the single most transferable engineering detail in this dossier.**

```python
dt_init_std = self.dt_rank**-0.5 * dt_scale     # comment: "to preserve variance at initialization"
nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)   # dt_init="random" (default)
# dt_init="constant" instead sets all entries to dt_init_std

dt = torch.exp(torch.rand(self.d_inner) * (log(dt_max) - log(dt_min)) + log(dt_min)
      ).clamp(min=dt_init_floor)                # dt_min=1e-3, dt_max=1e-1, floor=1e-4
inv_dt = dt + torch.log(-torch.expm1(-dt))      # exact inverse of softplus
self.dt_proj.bias.copy_(inv_dt)
self.dt_proj.bias._no_reinit = True             # exempt from the repo's global zero-bias init
```

Three points that bear directly on the proposed experiment:

- The **up-projection** factor gets a fan-in-scaled `r^{-1/2}` uniform init. The **down-projection**
  factor (the dt_rank slice of `x_proj`) gets **no special treatment** — it inherits the model-wide
  default. So Mamba does *not* use a symmetric/balanced init; it scale-corrects only the second factor.
- The bias lives in the **full output dimension d_inner**, not in the bottleneck. Therefore
  **all per-channel gate diversity at initialization comes from the bias, not from the low-rank
  factor.** A rank-r factor with r << d cannot produce d independent channel offsets; Mamba solves
  this by putting a full-width bias after the bottleneck. This is a concrete, citable design
  pattern the LIV experiment should copy (§3.6).
- `_no_reinit` exists because a global init sweep would otherwise clobber the bias. Any
  implementation must guard against the same bug.

**They DID ablate the rank — Table 9, "Ablations: Expressivity of Δ"** (~350M params,
Chinchilla-optimal tokens, state size N=16 fixed):

| Δ proj. size | Params (M) | Perplexity |
|---|---|---|
| – (non-selective) | 358.9 | 9.12 |
| 1 | 359.1 | 8.97 |
| 2 | 359.3 | 8.97 |
| 4 | 359.7 | 8.91 |
| 8 | 360.5 | 8.83 |
| 16 | 362.1 | 8.84 |
| 32 | 365.2 | 8.80 |
| 64 | 371.5 | **8.71** |

This is **the only real rank sweep of a gate projection in this entire literature.** Read it
carefully, because it cuts both ways:

- Rank 1 -> 64 buys 0.26 ppl for +12.4M params. Non-selective -> rank 1 buys 0.15 ppl for +0.2M.
  So *most of the value is in being input-dependent at all, not in the rank.*
- The curve is monotone-ish but **noisy** (rank 8 = 8.83 beats rank 16 = 8.84), which sets a
  floor on the noise a rank sweep must resolve — see the pre-registered criterion in §7.
- The curve is **not saturated at 64**, and 64 is exactly what `dt_rank="auto"` gives at
  d_model=1024. The shipped heuristic sits at the *top* of the tested sweep. **Full-rank Δ was
  never run.** [GAP]

Also relevant, **Table 7 "Ablations: Selective parameters"**: selective Δ alone 9.81, selective B
alone 10.15, selective C alone 9.98, none 10.93, all three 8.71. "Δ is the most important
parameter due to its connection to RNN gating."

And **Table 10 "Ablations: SSM state dimension"** (Δ proj fixed at 64) is arguably the most
important number here for capacity allocation:

| N | Params (M) | ppl (const B,C) | ppl (selective B,C) |
|---|---|---|---|
| 1 | 367.1 | 9.88 | 9.73 |
| 8 | 369.1 | 9.82 | 8.84 |
| 16 | 371.5 | 9.81 | **8.71** |

Widening the *state* gives >1.0 ppl for ~1% params; widening Δ's *rank* 1->64 gives only 0.26.
**Capacity in the state is worth roughly 4x more per parameter than capacity in the gate rank.**
[REASONING] Read as a prior for the LIV experiment, this favours the hypothesis: gate projections
are the *right* place to spend fewer parameters, because gates appear to be the least
rank-hungry component measured.

**On Mamba's B and C projections** — worth stating precisely to avoid a common conflation.
`s_B(x) = Linear_N(x)` with N = d_state = 16 is a d_inner -> 16 map, an extreme bottleneck by
ratio. But it is **not a factorization**: 16 is the genuine terminal dimensionality of the SSM
state axis, not a compression of something that must come back out at width d. Δ's `dt_rank` is a
true rank choice on a gate; B/C's width is a state-size choice. Do not cite B/C as evidence about
low-rank gates.

### 1.2 Mamba-2 — the follow-up REMOVED the low-rank bottleneck (but not for quality reasons)

**Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured
State Space Duality**, Dao & Gu. https://arxiv.org/abs/2405.21060

Confirmed: **neither `dt_rank` nor `dt_proj` appears anywhere in `mamba_ssm/modules/mamba2.py`.**
Δ is now one `nheads`-wide slice of a single fused input projection:

```python
d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads   # [z, x, B, C, dt]
self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)
```

The inverse-softplus init survives, but resized from d_inner to nheads and moved to a bare
`nn.Parameter` (`self.dt_bias`, with `_no_weight_decay = True`). A becomes scalar-per-head
(`A_log`, uniform in [1,16], also `_no_weight_decay`).

**The stated rationale is tensor parallelism, NOT gate quality.** This distinction is essential to
report honestly, because it is the obvious counter-citation a reviewer will reach for. §8.1
"Tensor Parallel" sets up Mamba-1 as "Δ, B, C = low-rank projection(x_c)" and then:

> However, we see that since Δ, B, C are functions [of] x_c, so we would need an extra all-reduce
> between the GPUs to get the whole of x_c before computing Δ, B, C. [...] Compared to
> Transformers, we would incur two all-reduces instead of one, doubling the time spent in
> communication.

> With Mamba-2, our goal is to have only one all-reduce per block [...] As a result, we have the
> projection to get Δ, B, C **directly from u instead of from x_c**, allowing us to split these
> projection matrices.

The causal chain is: the killer is the **sequential dependency** (Δ read the *post-conv* tensor
x_c, which is sharded), not the rank. Once the projection is moved to read the block input u, it
is naturally absorbed into the one big fused `in_proj`, at which point a separate rank-r module
has no reason to exist. §7.1 adds the secondary motivation:

> Note that adopting parallel projections for the A, B, C, X inputs to the SSM **slightly reduces
> parameters** and more importantly is more amenable to tensor parallelism for larger models.

And the A -> scalar simplification is separately motivated by tensor-core matmul efficiency:
"these changes can be viewed as **slightly decreasing the expressive power in return for
significant training efficiency improvements**."

**Measured cost (Table 4, §9.4.1, 126.5M params):** parallel + extra norm = 11.49 ppl, parallel
without extra norm = 11.66. Text: "parallel projections to create (A,B,C,X) **saves parameters and
performs slightly better** than Mamba's sequential projections."

[GAP] **They never ran an isolated "remove only the dt bottleneck" ablation.** Table 4 bundles the
sequential->parallel move (which also relocates B/C and changes which tensor is read) with other
block changes; the A-scalar and head-structure changes are ablated separately (Table 5). There is
no clean attribution of any quality delta to the rank change alone.

**Bottom line for the writeup:** Mamba-2 is *not* evidence that low-rank gates hurt quality. It is
evidence that a low-rank gate reading a *post-mixing* tensor is hostile to tensor parallelism.
Note that this critique **does not apply to the LIV proposal**, where the gates read the block
input h (pre-conv), so a factorized gate remains TP-friendly. [REASONING]

### 1.3 Gated Linear Attention (GLA) — explicit low-rank forget gate at rank 16

**Gated Linear Attention Transformers with Hardware-Efficient Training**, Yang, Wang, Shen,
Panda, Kim. https://arxiv.org/abs/2312.06635

§4.4, "Parameter allocation" (an **unnumbered** display equation — cite by section, not number):

```
α_t = σ(x_t W_α1 W_α2 + b_α)^(1/τ) ∈ R^(1 × d_k),
      W_α1 ∈ R^(d × 16),  W_α2 ∈ R^(16 × d_k),  τ = 16
```

- Rank: **16, a fixed constant, not a function of d** (contrast Mamba's ceil(d/16)).
- Temperature **τ = 16**, "to encourage model to have a slower forgetting rate."
- Bias `b_α` **is present, on the up-projection only.**
- d_k = d/2, d_v = d.

**Stated reason — pure parameter budget, quoted:**

> **Parameter allocation.** As presented, our GLA layer employs two additional matrices for
> predicting α_t, r_t (i.e., W_α, W_r) compared to a regular softmax attention layer. **For
> parameter-efficiency, we use a low-rank parameterization** [...] We further set d_k = d/2 and
> d_v = d and use full-rank parameterizations for (W_Q, W_K, W_V, W_O, W_r). Ultimately, one GLA
> layer collectively needs (roughly) 4d² parameters, as in regular softmax attention.

Note the **deliberate asymmetry: only the gate is low-rank; Q/K/V/O and the *output gate* r_t are
all full-rank.** A full d x d_k gate would cost d²/2; rank 16 costs ~24d. This is a strong signal
that the authors regarded the gate specifically as the affordable thing to compress — and it is
the closest published precedent for the LIV proposal's *shape* (factorize gates, keep value and
output full width).

**Verified defaults in `fla/layers/gla.py`:** `gate_low_rank_dim = 16`,
`gate_logit_normalizer = 16`, `num_heads = 4`, `expand_k = 0.5`, `expand_v = 1.0`.

```python
self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                             nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
...
gk = F.logsigmoid(gk) / self.gate_logit_normalizer     # == log(σ(x)^(1/τ)); the stable form
```

Two implementation details worth flagging:

1. The code works in **log space** (`logsigmoid(x)/16`) where the paper writes `σ(x)^(1/16)`.
   Mathematically identical; `gate_logit_normalizer` *is* τ.
2. **`gk_proj` reads `hidden_states`, not the short-conv output** — i.e. the gate is a function of
   the block input, TP-friendly by construction, exactly the property Mamba-2 had to retrofit.
   The LIV proposal has this property too.

**Gate init: no special-casing at all.** `fla/models/gla/modeling_gla.py::_init_weights` applies
the generic `normal_(std=initializer_range=0.02)` + `zeros_(bias)` to every `nn.Linear`; neither
`gk_proj` layer is exempted (contrast Mamba's `_no_reinit`). Consequence at init:
`gk ≈ logsigmoid(0)/16 = -0.0433`, a per-step decay of **≈0.9576, uniform across all channels.**

[REASONING] This is a meaningful contrast worth putting in the design: **Mamba engineers
per-channel diversity into a full-width bias; GLA instead engineers a global temperature and
starts every channel identical**, letting training differentiate them. Same goal (start near
"remember"), opposite mechanism. Both are mechanisms for keeping a low-rank gate in a safe region
at init, and the LIV block — whose gates are *linear and unbounded* — has neither by default.

**Gate ablations — Table 4** (340M, 7B tokens, training ppl):

| Variant | Train ppl |
|---|---|
| GLA Transformer (4 heads) | **14.77** |
| No gate (= Linear Attention) | 23.21 |
| Data-independent scalar decay (= RetNet) | 16.55 |
| Data-dependent scalar gate | 15.56 |
| Small head dim (8 heads) | 15.29 |
| Large head dim (1 head) | 14.61 |

Gating is worth 8.4 ppl; data-dependence over fixed decay 1.79 ppl; vector-valued over scalar
0.79 ppl. Footnote 5 also reports a negative result on richer gate *structure*: the
outer-product `G_t = α_t^T β_t` parameterization "resulted in only marginal improvements."

[GAP] **GLA never ablated the rank, never ablated τ, and never compared low-rank vs full-rank α.**
I searched the full text and appendices. The rank-16 and τ=16 constants are **asserted, not
tuned.** So GLA reports no quality cost for its low-rank gate because it never measured one.

### 1.3b Summary table — the three best-documented low-rank gates

| | Mamba-1 Δ | Mamba-2 Δ | GLA α |
|---|---|---|---|
| form | d_inner -> r -> d_inner factorized | d_model -> nheads direct | d -> 16 -> d_k factorized |
| rank | ceil(d_model/16) = 48..160 | nheads (no bottleneck) | 16, constant in d |
| reads | post-conv x_c (TP-hostile) | block input u | block input (TP-friendly) |
| bias | d_inner-wide, inv-softplus log-U[1e-3,1e-1] | nheads-wide, same | up-proj only, **zero-init** |
| special init on factor | `r^{-1/2}` uniform on up-proj; none on down-proj | n/a | none (generic 0.02) |
| rank ablated? | **YES, Table 9** | no | no |

### 1.4 RWKV-6 / RWKV-7 — LoRA-style low-rank data-dependent gates, and the ONLY published rank-scaling rule

**RWKV-4: Reinventing RNNs for the Transformer Era**. https://arxiv.org/abs/2305.13048 —
decay is a **data-independent learned per-channel vector** `w ∈ (R≥0)^d`, applied as
`w_{t,i} = -(t-i)w`. Init `-5 + 8·(i/(d-1))^{0.7 + 1.3l/(L-1)}`. No projection, no data dependence,
no low rank. Baseline for "the cheapest per-channel gate."

**Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence** (RWKV-5/6).
https://arxiv.org/abs/2404.05892 — **the closest published prior art in spirit to the LIV
proposal.** RWKV-5 (Eagle) keeps decay data-independent but per-head. RWKV-6 (Finch) makes it
data-dependent via explicitly LoRA-shaped low-rank maps:

```
lora_□(x)     = λ_□ + tanh(x A_□) B_□
d_t           = lora_d(ddlerp_d(x_t, x_{t-1}))
w_t           = exp(-exp(d_t))
```

Verbatim ranks from the paper: *"each A_□ ∈ R^{D×32}, B_□ ∈ R^{32×D}"* for the five token-shift
mixers (r,k,v,g,w); *"For the special case of LoRA_ω ... we introduce double-sized trainable weight
matrices A_ω ∈ R^{D×64}, B_ω ∈ R^{64×D}."* So **rank 32 for mixers, rank 64 for decay**, confirmed
in `BlinkDL/RWKV-LM` (`RWKV-v5/src/model.py`, `RWKV_Tmix_x060`): `D_MIX_LORA = 32`,
`D_DECAY_LORA = 64`. **These are constants, not functions of D** — the paper says larger models are
"expected to further increase the size of these weight matrices by double or more," i.e. no rule.

Three implementation details that are directly transferable design choices:

- **There is a nonlinearity inside the bottleneck**: `tanh(x A) B`, not a bare product. This is a
  meaningful departure from a pure linear factorization, and it is *not* rank-limited in the same
  way (tanh breaks the rank-r bound on the map, though the *pre-activation* is still rank r).
- **There is an additive base λ_□**, i.e. the low-rank part is a *perturbation of a learned
  full-width vector*, not the whole gate. Same role as Mamba's full-width `dt_bias`.
- **The five mixer LoRAs share one fused down-projection**: `time_maa_w1` is `(n_embd, 32*5)`, then
  five separate up-projections via `bmm`. Not five independent adapters.
- **Finch's output gate `g` is FULL RANK d→d** (`self.gate = nn.Linear(n_embd, dim_att)`).
  Low rank is used for token-shift and decay only. A rank-64 gate LoRA exists only in the
  experimental `RWKV_Tmix_x060a` variant, not the released model. [REASONING] Note the pattern
  across GLA, Finch, and Mamba: **the *decay/forget* gate gets factorized; the *output* gate stays
  dense.** The LIV proposal factorizes both B (pre-conv, decay-like) and C (post-conv,
  output-gate-like). The literature's revealed preference suggests **C is the riskier of the two**,
  and the design should be able to factorize them independently (§7).

**RWKV-7 "Goose" with Expressive Dynamic State Evolution**. https://arxiv.org/abs/2503.14456 —
four low-rank MLPs `loramlp_□(f,x,bias) = f(x A_□) B_□ + λ_□` for decay `w`, in-context learning
rate `a`, value-residual mix `ν`, and gate `g`; `r_t, k_t, v_t` stay full rank. RWKV-7 *removed*
RWKV-6's data-dependent token shift "to improve training speed" — another instance of a low-rank
component deleted for throughput, not quality.

**Table 16, "Suggested Intermediate Dimensions"** — the only published rank-vs-width table in this
literature:

| D | d_w (decay) | d_a (ICLR) | d_v (val-resid) | d_g (gate) |
|---|---|---|---|---|
| 768 | 64 | 64 | 32 | 128 |
| 1024 | 64 | 64 | 32 | 128 |
| 2048 | 96 | 96 | 64 | 256 |
| 2560 | 96 | 96 | 64 | 320 |
| 4096 | 128 | 128 | 96 | 480 |
| 6144 | 128 | 128 | 96 | 640 |

The paper is refreshingly candid that these "are based on our **mere speculation** of how much
information can be passed through" — **not tuned.** The code gives the closed forms
(`RWKV_Tmix_x070`), which were verified numerically to reproduce Table 16:

```python
D_DECAY_LORA = max(32, int(round((1.8*(C**0.5))/32)*32))   # decay      ~1.8*sqrt(d)
D_AAA_LORA   = max(32, int(round((1.8*(C**0.5))/32)*32))   # ICL rate   ~1.8*sqrt(d)
D_MV_LORA    = max(32, int(round((1.3*(C**0.5))/32)*32))   # val resid  ~1.3*sqrt(d)
D_GATE_LORA  = max(32, int(round((0.6*(C**0.8))/32)*32))   # gate       ~0.6*d^0.8
```

A later `RWKV-v7/train_temp` variant uses larger coefficients (`2.5*sqrt(C)` for decay/ICLR,
`1.7*sqrt(C)` for value-residual, `5*sqrt(C)` for gate); both are labelled `# suggestion` in code.
**Highly relevant scaling observations:**

- **Ranks are rounded to multiples of 32** in every case. This is presumably for kernel efficiency
  and is a free design constraint to adopt (§5).
- The **decay rank scales as ~sqrt(d)**, i.e. *sublinearly* — so r/d SHRINKS as models grow.
  At d=2048 the decay rank is 96 (r/d = 1/21).
- The **gate rank scales as ~d^0.8**, i.e. much faster than sqrt(d), and is 2-4x larger than the
  decay rank at every width (256 vs 96 at d=2048). [REASONING] This is the strongest published
  signal that **gates need more rank than decays**, and again suggests treating the LIV block's two
  gates asymmetrically rather than giving both the same r.
- In-code comment: `# Note: for some data, you can reduce D_GATE_LORA or even remove this gate`.

**Ablations — the honest answer: neither RWKV paper ablates gate RANK.** What exists:
- Finch Appendix K (Table 19), 6-layer d=768, 1.6B MiniPile tokens, ctx 512, final val loss:
  Finch **2.910**; DDLerp on decay only **2.923**; no DDLerp **2.926**. The whole data-dependent
  token-shift LoRA machinery is worth ~0.016 loss at that scale.
- Goose Appendix (Table 19), same setup, train/val: Goose **2.834/2.541**; scalar decay
  2.873/**2.609**; scalar in-context learning rate 2.843/2.591; no bonus 2.841/2.588.
  This ablates **vector vs scalar** (vector wins ~0.068 val loss on decay) — never rank r vs r'.

[GAP] **No RWKV paper compares a full-rank d→d data-dependent decay against the rank-32/64/1.8√d
version.** The low-rank form is an unquestioned parameter-efficiency device throughout.

### 1.5 RetNet — the cheapest possible gate (data-independent, zero parameters)

**Retentive Network: A Successor to Transformer for Large Language Models**.
https://arxiv.org/abs/2307.08621

Eq. 8, exact: **`γ = 1 - 2^(-5-arange(0,h)) ∈ R^h`**, *"identical among different layers and keep
them fixed."* Fully data-independent, one scalar per head, **zero parameters.** Derivation:
`A = Λ(γe^{iθ})Λ^{-1}` diagonalized to `γ, θ ∈ R^d`, then *"we further simplify γ as a scalar."*
§3.1 notes an experimental override held constant across sizes:
`γ = 1 - exp(linspace(log 1/32, log 1/512, h))`. Decay enters as `D_{nm} = γ^{n-m}` for n ≥ m.

Note the same asymmetry as elsewhere: RetNet pairs a **free decay** with an **expensive dense
gate** — `MSR(X) = (swish(X W_G) ⊙ Y) W_O` with `W_G, W_V ∈ R^{d×2d}`, `W_O ∈ R^{2d×d}`.

### 1.6 HGRN / HGRN2 — FULL-RANK gate + a lower-bound trick (an anti-saturation control)

**Hierarchically Gated Recurrent Neural Network for Sequence Modeling**.
https://arxiv.org/abs/2311.04823

Forget gate is a **full dense d→d projection — no low rank anywhere**:
`μ_t = Sigmoid(x_t W_μ + b_μ)`, `W_μ ∈ R^{d×d}`. Output gate `g_t = Sigmoid(W_g x_t + b_g) ∈ R^{2d}`.
Gates depend only on `x_t`, not `h_{t-1}`, which is what permits the parallel scan.

**The lower-bound trick, precisely** — a learnable `Γ ∈ R^{H×d}` (H = layers) holding an
independent bound per unit per layer:

```
P    = Softmax(Γ, dim=0) ∈ R^{H×d}         # softmax over the LAYER axis
γ^k  = [Cumsum(P, dim=0)]_k                 # with [Cumsum(x)]_k = (Σ_{i≤k} x_i) - x_1
λ_t  = γ^k + (1 - γ^k) ⊙ μ_t
```

The shifted cumsum keeps the top layer able to forget; softmax positivity + sum-to-1 across layers
makes `γ^k` monotonically increasing in depth and bounded below 1. **This is an anti-saturation
mechanism:** to reach a target decay γ̄≈1 the sigmoid only needs `μ_t = (γ̄-γ^k)/(1-γ^k) < γ̄`,
keeping pre-activations off the saturated tail.

Ablations (WikiText-103 ppl, Table 10): HGRN 24.14; no lower bound 24.71; random per-layer 24.60;
decreasing 24.63; lower bound only / data-independent decay 27.70. On LRA the bound is decisive:
average 86.91 with, 51.53 without (Path-X fails to converge). Pile 1B ppl: HGRN 4.14; **removing
the forget gate entirely collapses to 57.42.**

**HGRN2: Gated Linear RNNs with State Expansion**. https://arxiv.org/abs/2404.07904 — same
full dense d→d gate; input gate tied to `(1-f_t)`, which expands state "without introducing any
additional parameters" (this is control (e), tied gates, working in practice).

**Highly relevant precedent:** HGRN2 §3.1 *did* evaluate a **low-rank (LR)** parameterization among
its "Parameter Efficient State Expansion" candidates — 4.76 ppl at n=4 — and **rejected it for
HARDWARE reasons, not quality**: elementwise-recurrence variants *"cannot leverage tensor cores."*
This is a second independent instance (with Mamba-2 and HGRN2 both) of low-rank being abandoned on
throughput grounds while being quality-competitive. It is exactly the brainlift's worry, twice
confirmed in the literature.

### 1.7 Gated DeltaNet and Griffin/Hawk

**Gated Delta Networks: Improving Mamba2 with Delta Rule**. https://arxiv.org/abs/2412.06464 —
both gates are **scalar per head**, neither low-rank nor d→d.
`S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T`. Code (`fla/layers/gated_deltanet.py`):

```python
self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
self.A_log  = nn.Parameter(torch.log(A))    # per-head
self.dt_bias = nn.Parameter(inv_dt)          # per-head
```

Gate cost is `2·d·H` (~65K at d=2048, H=16) vs ~4d² for the layer — negligible. The paper notes
gating *"only performs elementwise multiplication ... without affecting matrix multiply
structures"* and *"maintains the same speed as DeltaNet."* Related work explicitly lists adopting
"GLA-like diagonal gating" as **future work** — i.e. this line went *cheaper* than low rank, not
richer. **DeltaNet** (https://arxiv.org/abs/2406.06484): `β_t = σ(W_β x_t)` scalar per head,
*"the additional parameters ... are negligible"*; no ablation of β_t's parameterization.

**Griffin: Mixing Gated Linear Recurrences with Local Attention**.
https://arxiv.org/abs/2402.19427 — **the "diagonal" claim needs care; two different things are
called diagonal.**

- The **recurrent weight `a` IS diagonal**: *"The recurrent weight a in Equation (4) is diagonal.
  Hence all operations are element-wise."* `a = σ(Λ)`, Λ a learnable vector.
- The **gate weight matrices are NOT diagonal — they are BLOCK-DIAGONAL dense blocks**, and for a
  communication reason: §4.1, *"To avoid additional cross-device communication, we use
  block-diagonal weights for the gates in the RG-LRU ... instead of dense matrices."*

```
r_t = σ(W_a x_t + b_a)                   # recurrence gate
i_t = σ(W_x x_t + b_x)                   # input gate
a_t = a^{c·r_t},  a = σ(Λ),  c = 8       # "c is a scalar-valued constant set to 8"
h_t = a_t ⊙ h_{t-1} + sqrt(1 - a_t²) ⊙ (i_t ⊙ x_t)
log a_t = -c · softplus(Λ) ⊙ r_t         # Appendix A Eq. 6, the stable log-space form
```

Init: Λ set so `a^c` is uniform in **[0.9, 0.999]**; `W_a`, `W_x` use LeCun init.
**Parameter cost:** paper states 16 blocks so `W_x`, `W_a` each have `D_RNN²/16`, i.e. both gates
together `D_RNN²/8` with `D_RNN ≈ 4D/3` — roughly 15% of an MHA block's 4D². Not cheap.

[FLAG — discrepancy] The released `google-deepmind/recurrentgemma`
(`recurrentgemma/torch/layers.py`) builds both gates as
`BlockDiagonalLinear(width=self.width, num_blocks=self.num_heads)`, i.e. **`D²/num_heads`, not
`D²/16`**: 2B is `num_heads=10, lru_width=2560` (block_width 256, 655,360 params per gate); 9B is
`num_heads=32, lru_width=5632` (block_width 176, 991,232 per gate). `c = 8` is the hardcoded
literal `-8.0`. Also note the `SqrtBoundDerivative` gradient-clipped sqrt guarding bf16 NaNs — a
concrete instance of a *multiplicative* path needing a numerical guard.

[GAP] **No quantitative ablation of Griffin's gate parameterization.** The comparison to
GRU/Mamba/LRU is purely analytical (Appendix A, Fig. 7). **Low rank is never discussed as a gate
option**, and no low-rank option exists anywhere in the RecurrentGemma code. Griffin's design bias
is stated as: the gate *"is biased towards retaining information, and does not allow to fully
discard the contribution of h_{t-1}"* — the opposite bias from Mamba.

**Griffin is the single most useful data point for control (d):** a production model from a major
lab chose **block-diagonal gates over dense**, for hardware reasons, at ~1/10 to 1/16 the dense
parameter cost. That is control (d) already validated at scale by someone else — and it is *not*
low rank.

### 1.7b Cross-architecture gate summary

| Architecture | Gate form | Rank / cost | Rank ablated? |
|---|---|---|---|
| RetNet decay | data-indep. scalar/head, fixed formula | **0 params** | n/a |
| RWKV-4 decay | data-indep. per-channel vector | d | n/a |
| RWKV-5 decay | data-indep. per-head vector | d/h | n/a |
| Mamba-1 Δ | **low-rank d→r→d** + full-width bias | r = ceil(d_model/16) | **YES (Table 9)** |
| Mamba-2 Δ | direct slice of fused in_proj | nheads | no |
| GLA α | **low-rank d→16→d_k** + bias, τ=16 | r = 16 const | no |
| RWKV-6 decay | **low-rank d→64→d**, tanh inside, + base λ | r = 64 const | no |
| RWKV-7 decay | **low-rank**, ~1.8·sqrt(d), mult. of 32 | 96 at d=2048 | no |
| RWKV-7 gate | **low-rank**, ~0.6·d^0.8, mult. of 32 | 256 at d=2048 | no |
| HGRN/HGRN2 forget | **full dense d→d** + layer-wise lower bound | d² | no (LR rejected on HW) |
| Gated DeltaNet α,β | scalar per head | 2dH | no |
| Griffin RG-LRU | **block-diagonal** (grouped) | D²/num_heads | no |
| **LFM2 LIV B, C** | **full dense, NO nonlinearity, no bias** | 2d² | **no — this experiment** |

[REASONING] Two structural patterns worth stating explicitly, because they shape the design:

1. **Every low-rank gate in the literature has a full-width additive component** — Mamba's
   d_inner-wide `dt_bias`, GLA's `b_α`, RWKV's `λ_□`. None is a bare `B·A·x`. The reason is
   capacity for *per-channel offsets*, which a rank-r map cannot supply. The LIV block currently has
   **no bias at all** (`conv_bias=false`). Adding a full-width bias to a factorized gate costs 2d
   parameters and is, by the revealed preference of every prior system, close to mandatory.
2. **Every low-rank gate in the literature is BOUNDED** — sigmoid, `exp(-exp(·))`, softplus,
   `σ(·)^{1/τ}`. **The LIV gates are raw linear.** This means the LIV block is the *first* place
   where a low-rank gate would feed an unbounded multiplicative path. That is simultaneously the
   novelty and the principal risk (§3.5).

### 1.8 [GAP + IMPORTANT] Liquid's own architecture search does NOT contain low-rank gates —
### but it DOES contain two of the proposed control conditions

I downloaded and read the LaTeX source of Liquid's own NAS paper:

**STAR: Synthesis of Tailored Architectures**, Thomas, Parnichkun, Amini, Massaroli, Poli.
https://arxiv.org/abs/2411.17800 (source: `arxiv.org/e-print/2411.17800`)

This is the paper that *produced* the LIV operator taxonomy the LFM2 block comes from, so it is
the single most important source for the novelty question. Verified from
`sections/appendix/2_details.tex` (Appendix "Linear Input-Varying Systems and Featurizers: Option
Pools"):

**The searchable CHANNEL-MIXING structures are exactly three (operator genome, 5th integer):**

```
1. Diagonal
2. Dense
3. Grouped (block-structured)
```

**"Low rank" appears in the genome ONLY as a TOKEN-mixing structure** (operator genome, 2nd
integer: `1. Diagonal (GMemless) / 2. Low rank (SA) / 3. Scaled Toeplitz (GConv) /
4. Sequentially semi-separable (Rec)`) — i.e. "low rank" there denotes the rank structure of the
**sequence-mixing matrix T_ij** in linear attention, *not* a factorized channel projection. The
featurizer genome's per-feature-group fields are: token-mixing structure, sparsity, nonlinearity,
channel-mixing structure, **expansion factor**, and **repeat factor** — there is
**no rank field.** The gated-conv classes are:

```
GConv-1  73111   short convolution, kernel length 3   <-- the LFM2 LIV block
GConv-2  83111   implicitly-parameterized long conv (Hyena-family)
```

**Verdict on novelty:** Liquid's own search space, by construction, **cannot express a low-rank
factorized gate projection.** The proposed modification is *outside* the space STAR searched. This
is a genuine, auditable, citable novelty claim, and it is stronger than a mere "I could not find
it" — the option pool is enumerated in the appendix and rank is absent.

**But the same appendix delivers two hard blows to the experiment's framing, and both must be
absorbed into the design:**

1. **STAR's actual mechanism for cutting parameters is FEATURIZER WEIGHT SHARING and FEATURE GROUP
   SHARING — which is precisely control (b) in §6.** From
   `sections/appendix/3_analysis.tex`: *"a key mechanism for STAR to reduce parameter counts is to
   identify which LIVs can be connected through featurizer or feature group sharing without
   degrading performance."* The backbone genome dedicates **four of its five integers** (2-5) to
   sharing structure: featurizer sharing group, featurizer sharing strategy (`1. No weights are
   shared / 2. All weights are shared`), feature-group sharing group, and feature-group sharing
   strategy. And crucially, §"Recurring Motifs" reports that sharing between gated convolutions is
   an *evolved winner*: *"Backbones optimized for quality often include two differential variants
   of short gated convolutions connected through featurizer sharing"* and *"A recurring motif from
   the evolutionary process involves LIVs with a block-Toeplitz token-mixing structure (e.g.,
   convolutions, gated convolutions). In these cases, earlier LIVs in the model are connected
   through feature group sharing to later LIVs."*

   [REASONING] Implication: **weight sharing is not a strawman control — it is the incumbent
   method, discovered by evolutionary search from the same lab, on the same operator class.**
   Any claim that "low-rank gates are a good parameter trade" that does not beat sharing is
   not a publishable claim. This promotes control (b) from optional to **mandatory**. Note also
   that the sharing STAR found is *cross-depth* (between different LIVs), whereas control (b) as
   framed in §6 is *within-block* (tie B's projection to C's). Both should be run; cross-depth
   sharing is the one with Liquid's endorsement.

2. **"Grouped (block-structured)" channel mixing IS already in the search space; low-rank is not.**
   [REASONING] A reviewer can therefore say: "Liquid searched over grouped/block-diagonal channel
   mixing and did not select it for the gated-conv featurizer; you are proposing an *untested*
   structure while an *already-searched* structure was available." This makes control (d)
   (grouped/block-diagonal gates) the natural structured competitor, and it is cheap to add
   because grouped matmuls are better supported than skinny ones (§5).

[GAP] STAR reports no per-featurizer rank ablation, no gate-parameterization ablation, and
optimizes only (quality, params, cache) — **not latency**. Note the contrast with LFM2 itself,
where the blog states they replaced the cache-size proxy with *direct* measurement of "peak memory
plus prefill+decode speed on Qualcomm Snapdragon embedded SoC CPUs". So Liquid's shipped
methodology treats **measured on-device latency, not parameter count, as the efficiency
objective** — which is exactly the brainlift's worry, endorsed by the vendor.

### 1.9 The capacity constraint that should worry you most: Zoology / MQAR

**Zoology: Measuring and Improving Recall in Efficient Language Models**, Arora, Eyuboglu,
Timalsina, Johnson, Poli, Zou, Rudra, Ré. https://arxiv.org/abs/2312.04927

This is the sharpest published statement about **capacity limits of gated convolutions
specifically**, and it is about *width*, which is what a low-rank gate reduces.

- Empirical Claim 1: gated convolutions *"require model dimension to scale at least linearly in
  sequence length"*, whereas attention achieves *"near-constant dimensionality."* Concretely,
  across a sweep of model dimension and sequence length **both from 64 to 512** (2-layer models,
  vocab 8192, 4 LRs each, max test accuracy reported), attention *"solves MQAR perfectly at all
  sequence lengths using a constant model dimension of 64"*, while BaseConv/H3/Hyena/RWKV do not
  exceed 0.9 accuracy **unless d ≥ N**.
- Theorem 4.4 (data-independent filters) gives a positive result at
  Õ(N log c) parameters and Õ(1) layers — note **parameter count grows with sequence length N**.
  Proposition 4.3 for attention needs O(c²) parameters, independent of N.
- Theorem 4.5 (input-dependent filters) shows input-dependent kernels reduce this to O(t·Nc)
  parameters with t distinct interaction distances.
- Headline scale gap: *"a 70M parameter attention model outperforms a 1.4 billion parameter
  gated-convolution model on associative recall"*; gated convs trail attention by up to 2.1 ppl on
  the Pile, with 82% of the gap attributable to in-context recall.

**Important nuance the paper is explicit about, and which cuts in the experiment's favour:** the
recall bottleneck is attributed to input-dependent **sequence mixing** (the filters), *not* to the
gating multiplication or the channel projections. Gating alone is input-dependent yet insufficient.

[REASONING] Two consequences for the design:
- The *mechanism* Zoology identifies as capacity-critical (filter input-dependence) is untouched by
  gate factorization, so the first-order prediction is that gate rank should not hurt MQAR.
- But because gated convs are *already* the width-starved component of the architecture, any
  eval suite must include an **explicit recall/MQAR-style probe**, not just perplexity. Perplexity
  is precisely the metric Zoology showed hides an 82%-of-the-gap recall deficit. A rank sweep
  measured only in ppl could report "non-inferior" while silently degrading recall. **This makes a
  recall probe mandatory, not optional** (§7).



---

## 2. Low-rank structure in projections generally — and the from-scratch question

### 2.1 LoRA is NOT evidence for this experiment. Two independent reasons.

**LoRA: Low-Rank Adaptation of Large Language Models**, Hu et al. https://arxiv.org/abs/2106.09685

Intrinsic-dimension lineage: **Measuring the Intrinsic Dimension of Objective Landscapes**, Li et
al. https://arxiv.org/abs/1804.08838 (MNIST FC d_int90 ≈ 750 of D=199,210; across 20 FC nets the
native parameter count varied 24.1x while d_int90 varied only 1.33x) and **Intrinsic Dimensionality
Explains the Effectiveness of Language Model Fine-Tuning**, Aghajanyan et al.
https://arxiv.org/abs/2012.13255 (RoBERTa-Large MRPC d_int90 = 207).

**Reason 1 — the claim is about ΔW, not W.** LoRA's own §4.1 states dense layer weights
*"typically have full-rank."* Its hypothesis is that *"the change in weights during model adaptation
also has a low 'intrinsic rank'."* The conclusion relegates the question at issue here to future
work: the rank-deficiency of ΔW *"suggests that W could be rank-deficient as well, which can also be
a source of inspiration for future works."*

Worse for the transfer argument, **LoRA §7.3 is active evidence that ΔW is not aligned with W's
dominant structure.** Projecting W onto ΔW's top-r subspace (GPT-3 layer 48, ‖W_q‖_F = 61.95):

| r | ‖U^T W_q V^T‖_F using ΔW_q's directions | using W_q's own top directions | random |
|---|---|---|---|
| 4 | 0.32 | 21.67 | 0.02 |
| 64 | 1.90 | 37.71 | 0.33 |

Their conclusion: ΔW *"only amplifies directions that are not emphasized in W"*, with an
amplification factor of ~21.5. [REASONING] So LoRA's success is *compatible with, indeed evidence
for*, a picture where the useful adaptation signal lives in a few directions a **full-rank,
richly-trained W has already learned but underweights.** A from-scratch rank-r layer has no W to
underweight anything; it must synthesize the whole function inside an r-dimensional column space.
Different mathematical objects; the result does not transfer.

Note too that LoRA's rank ablation (Table 6, GPT-3 175B, WikiSQL, {W_q,W_v}) shows r=1 nearly
saturates: 73.4 / 73.3 / 73.7 / 73.8 / 73.5 for r = 1/2/4/8/64. **No from-scratch result looks
remotely like this**, which is itself a signal the regimes are not comparable.

**Reason 2 — LoRA's init trick is structurally unavailable here.** Confirmed in `peft`
(`src/peft/tuners/lora/layer.py::reset_lora_parameters`): the default path is
`kaiming_uniform_(lora_A.weight, a=sqrt(5))` (PyTorch's `nn.Linear` default, *not* Gaussian — the
paper's Gaussian is the opt-in `"gaussian"` mode with `std = 1/r`), and `zeros_(lora_B.weight)`.
Either way `BA = 0` at init, scaled by `α/r`.

This works because LoRA computes `W_0·x + BA·x`: zeroing BA makes the layer *exactly* the pretrained
layer at step 0, signal still flows through `W_0`, and A receives gradient because B's gradient path
runs through the live `W_0` branch. **In the LIV proposal the gate IS the product** — there is no
additive dense term. `B = 0` makes the gate identically zero, so `y = B ⊙ x̃ = 0`, the block output
is zero, `∂L/∂A = 0`, and the conv receives no gradient. **The single most famous low-rank init
trick is inapplicable.** (Same objection kills **PiSSA**, https://arxiv.org/abs/2404.02948, which
initializes from the principal components of an existing pretrained W.)

### 2.2 ALBERT — genuine from-scratch evidence, but scoped to embeddings

**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**.
https://arxiv.org/abs/1909.11942 — factorized embedding V→E→H, reducing O(V×H) to O(V×E + E×H),
V = 30,000. Their reasoning is the load-bearing part: WordPiece embeddings *"are meant to learn
context-independent representations, whereas hidden-layer embeddings are meant to learn
context-dependent representations."*

Table 3 ablation, ALBERT-base, **not-shared** (the clean low-rank-only comparison), avg score:

| E | 64 | 128 | 256 | 768 |
|---|---|---|---|---|
| Params | 87M | 89M | 93M | 108M |
| Avg | 81.3 | 81.7 | 81.8 | **82.3** |

*"larger embedding sizes give better performance, but not by much"* — 81.3 → 82.3 over a 12x
embedding-parameter range. In the all-shared setting the curve is non-monotone and peaks at E=128
(80.1), and they adopt E=128 for all larger models (H=2048, H=4096).

[REASONING] This is real from-scratch low-rank success, but its scope is exactly what their argument
predicts: a **context-independent** map where V ≫ H makes the dense form pathologically heavy.
**ALBERT never factorizes attention or FFN weights**, and its justification does not extend there.
Note also that even here more rank is monotonically better — just cheaply forgone.

### 2.3 Low-rank bottleneck in attention — evidence AGAINST rank starvation in mixing paths

**Low-Rank Bottleneck in Multi-head Attention Models**, Bhojanapalli, Yun, Rawat, Reddi, Kumar.
https://arxiv.org/abs/2002.07028

Theorem 1: if `d_q = d_k = d ≥ n` (n = sequence length), any positive column-stochastic n×n `P` is
realizable as `Softmax[(W_k X)^T (W_q X)/sqrt(d_k)]`; **if `d < n` there exist X and P that no
W_q, W_k can realize.** Since the standard heuristic sets head size = d/h, once `h > d/n` each head
projects below n and *"loses its ability to represent arbitrary context vectors."*

Empirically, BERT_LARGE (d=1024, 336M params throughout) **degrades as heads increase** under the
standard heuristic — SQuAD F1 90.89 (8 heads) → 90.61 (16) → 90.45 (32); and with head size fixed
at 128 and embedding 512, performance rises monotonically to 90.95 F1 at 32 heads / 319M params,
matching the embedding-1024 baseline. The cleanest "rank hurts" curve is head-size sweep at 8 heads,
embedding 512: F1 88.53 / 89.51 / 89.60 / 90.33 for head size 32 / 64 / 128 / 256.

**Do not confuse this with Linformer.** **Linformer: Self-Attention with Linear Complexity**,
https://arxiv.org/abs/2006.04768, Theorem 1 shows the **softmax attention matrix P** (n×n,
data-dependent, recomputed per input) is approximately rank `Θ(log n)` by Johnson–Lindenstrauss.
The theorem quantifies over *all* `W^Q, W^K, W^V` — the learned weights are explicitly **not** the
low-rank object. Their own phrasing: *"the stochastic matrix formed by self-attention mechanism is
low-rank."* **Linformer is about activations; Bhojanapalli is about weights. They are not in
conflict, and conflating them is a common error.**

[REASONING] Relevance to LIV: the LIV gate is *not* a softmax logit matrix, so Bhojanapalli's
theorem does not apply directly. But it establishes the general principle that **rank starvation in
a sequence-mixing path has a hard expressivity cost**, and it pairs with Zoology (§1.9) — which
found gated convs specifically need `d ≥ N` for recall — to define the risk.

### 2.4 [THE CENTRAL RESULT] Naive from-scratch low-rank fails, and fails WORSE with scale

Two independent groups, same conclusion, both with no incentive to make low-rank weights look bad.

**GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection**, Zhao et al.
https://arxiv.org/abs/2403.03507 — **Table 2, C4 validation perplexity.** The "Low-Rank" row is a
learnable `W = BA` trained from scratch, i.e. exactly the object in question:

| Method | 60M | 130M | 350M | 1B |
|---|---|---|---|---|
| Full-Rank | 34.06 | 25.08 | 18.80 | 15.56 |
| GaLore | 34.88 | 25.36 | 18.95 | 15.64 |
| **Low-Rank (plain W = BA)** | **78.18** | **45.51** | **37.41** | **142.53** |
| LoRA | 34.99 | 33.92 | 25.58 | 19.21 |
| ReLoRA | 37.04 | 29.37 | 29.08 | 18.33 |

(r/d = 128/256, 256/768, 256/1024, 512/2048 — note these are *generous* rank fractions, 1/2 to 1/4.)
At 60M it is **more than 2x** the full-rank perplexity. At 1B it **collapses to 142.53** — worse
than the 60M full-rank model. **This is a catastrophic, scale-worsening failure at rank fractions
far more generous than r=128 at d=2048 (= 1/16).**

**Crucially, GaLore is explicit that it keeps WEIGHTS full-rank and projects only GRADIENTS.**
Abstract: GaLore *"allows full-parameter learning but is more memory-efficient."* §3.3: *"GaLore
explicitly utilizes the low-rank updates instead of introducing additional low-rank adaptors and
hence does not alter the training dynamics."* Their critique of adapter methods: they *"limit the
parameter search to a low-rank subspace and alter the training dynamics"* and *"may require
full-rank warm start."*

**Stack More Layers Differently: High-Rank Training Through Low-Rank Updates** (ReLoRA), Lialin et
al. https://arxiv.org/abs/2307.05695 — Table 2 pretraining perplexity:

| Method | 60M | 130M | 250M | 350M |
|---|---|---|---|---|
| Full training | 33.81 | 23.65 | 22.39 | 18.66 |
| **LoRA (plain, from scratch)** | **47.44** | **34.17** | **36.60** | **57.11** |
| LoRA + Warm Start | 34.73 | 25.46 | 22.86 | 19.73 |
| ReLoRA | 34.46 | 25.04 | 22.48 | 19.32 |

Gaps of +13.6 / +10.5 / +14.2 / **+38.5** ppl, and **non-monotone in scale** (34.17 → 36.60 → 57.11)
— it fails to scale at all. Their conclusion: *"This approach yields remarkably high perplexity,
indicating that a simple matrix decomposition has significantly different training dynamics from
full-rank training."* On spectra: *"most of the singular values for LoRA are zero."*

**ReLoRA's own ablation (Table 6, 130M) is the most informative single table:**

| Config | ppl |
|---|---|
| LoRA baseline | 34.17 |
| + restarts | 34.25 |
| + restarts + optimizer reset | **diverged** |
| + restarts + reset + jagged schedule | 29.77 |
| **warm start only** | **25.46** |
| all four (= ReLoRA) | 25.04 |
| regular training | 23.65 |

[REASONING] **Restarts alone buy nothing; optimizer reset alone diverges; the overwhelming majority
of the gain comes from the FULL-RANK WARM START (34.17 → 25.46), with ReLoRA's actual machinery
contributing the final 0.42.** ReLoRA is closer to "full-rank pretraining, then low-rank
continuation" than to "low-rank pretraining" — and it still trails by 1.39 ppl. ReLoRA also reports
needing **1.5-2x larger LR** than full-rank training.

**Training Neural Networks from Scratch with Parallel Low-Rank Adapters** (LTE), Huh et al.
https://arxiv.org/abs/2402.16828 — a single r=64 LoRA on ViT-S/ImageNet100 is *"inferior ... to
models trained using standard optimization"*; baseline is recovered only at **full rank**. They
state the reason plainly: LoRA *"is fundamentally incapable of recovering weights that exceed the
rank r < min(m,n)"*, and empirically *"the rank of the gradient tends to increase throughout
training, hinting at the necessity for high-rank updates."* Their fix reconstitutes a high-rank
weight from many low-rank pieces (32 heads × r=64, merge every T=10 steps, heuristic: heads × rank
> largest layer dim). Rank sweep at 32 heads: 44.0 / 60.4 / 71.2 / **73.7** / 73.5% for
r = 8/16/32/64/128. Their LR is **0.05-0.1x standard**, with α=4096, and semi-orthogonal init scaled
by `sqrt(d_out/d_in)` beat Kaiming/Xavier.

[REASONING] **The common thread across ReLoRA, LTE, GaLore, Pufferfish, Cuttlefish: every method
that "works" reconstitutes a full-rank weight** — by merging, by rotating the gradient subspace, or
by warm-starting full-rank then factorizing. **None trains a rank-constrained weight for the whole
run and matches dense.**

### 2.5 The successes, and what they required

**Initialization and Regularization of Factorized Neural Layers**, Khodak et al.
https://arxiv.org/abs/2105.01029 (recipes in §3.2). **Table 1, ResNet32^x2, CIFAR-10 at 10% params
(baseline 94.49):**

| Method | acc |
|---|---|
| Low-Rank (plain) | 93.59 |
| Low-Rank (Spectral Init alone) | 92.52 |
| Low-Rank (Frobenius Decay alone) | 92.92 |
| **Low-Rank (SI & FD together)** | **94.34** |

**This is a trap worth internalizing: used SEPARATELY, SI and FD each make things WORSE than plain
low-rank; used TOGETHER they recover the dense baseline.** Testing one without the other would lead
you to reject both. Their BERT result (Table 4) is more sobering: BERT-base at 0.5 MHA compression
with FLAMBé(SI&FD) gets 87.69 F1 vs 88.55 dense; **BERT-large at 0.5 compression loses 0.86 F1.**
Their own practitioner note: *"for Transformers, SI is the critical ingredient when compressing"*,
and under compression *"SI is in fact necessary for FD to work at all."*

**Exploring Low Rank Training of Deep Neural Networks**, Kamalakara et al.
https://arxiv.org/abs/2209.13569 — the source of GaLore's Low-Rank baseline, and it partially
falsifies Khodak's *explanations* while agreeing on the recipes. They overwrite all singular values
with **ones** ("spectral ones") and match or beat plain SI, concluding *"it is the direction of the
singular vectors that matters."* Their language result is the key data point:

**GPT-2 / LM1B, rank fraction 0.62 throughout:** full-rank baseline **37.67**; low-rank He init
39.6; low-rank spectral 38.78; low-rank spectral-ones 38.47. **No from-scratch config reaches
baseline**, at a rank fraction of 0.62. Vision is the opposite story: ResNet-50/ImageNet rank 0.5 +
spectral = 76.13 vs 76.39 baseline; WRN-28/CIFAR-100 at rank 0.2-0.3 + spectral *exceeds* baseline.
**The domain split is stark: vision largely closes the gap at rank ≥ 0.2-0.5; language does not.**
For language they endorse warm-start (40K unfactorized steps, then 200K factorized with **halved
LR**). Caveat: no seeds or variances reported.

**Pufferfish**, https://arxiv.org/abs/2103.03936 — naive full factorization at rank ratio 0.25 costs
~0.4% on VGG-11/CIFAR-10 but **~3%** top-1 on ResNet-50/ImageNet: **the loss grows with scale.**
Two fixes, both structural: (i) **hybrid architecture** — keep the first K-1 layers dense, because
*"the approximation error in the early layers can be accumulated and propagated to the later
layers"*; (ii) full-rank warm-up for ~10% of epochs, then per-layer truncated SVD with
`U = ŨΣ^{1/2}, V^T = Σ^{1/2}Ṽ^T`. Ablation (ResNet-18/CIFAR-10, vanilla 95.09): plain low-rank
93.75; hybrid no warm-up 93.92; **hybrid + warm-up 94.87**. Their one big positive:
**Transformer/WMT16 De→En, 48.98M → 26.70M params, val ppl 11.88 → 7.34 and BLEU 19.05 → 26.87** —
the factorized model *beat* dense, attributed to *"some implicit regularization."* But their
LSTM/WikiText-2 result shows warm-up is worth 3.3 ppl (88.72 with, 92.04 without).

**Cuttlefish: Low-Rank Model Training without All the Tuning**,
https://arxiv.org/abs/2305.02538 — same shape: full-rank first, then switch, *"once the stable ranks
of all layers have converged."* Relies on the finding that after a few epochs the **stable rank of
each layer stabilizes at a constant value** — which is a direct endorsement of the §4.1 diagnostic
as a *rank-selection* method.

### 2.6 Trained transformer weights are empirically NOT close to low-rank

**ASVD**, https://arxiv.org/abs/2312.05821 — LLaMA-7B WikiText-2 ppl (dense 5.68): plain SVD at
**95% params retained** gives **2800.94**; on LLaMA-2-7B and -13B plain SVD at 0.95 produces
**NaN**. ASVD (activation-aware) holds 5.78 at 0.95 and 6.09 at 0.9.

**SVD-LLM**, https://arxiv.org/abs/2403.07378 — Table 1, LLaMA-7B WikiText-2 (dense 5.68):

| Ratio removed | SVD | FWSVD | ASVD | SVD-LLM |
|---|---|---|---|---|
| 20% | 20061 | 1727 | 11.14 | **7.73** |
| 40% | 52489 | 18156 | 1407 | **9.27** |
| 60% | 105474 | 32194 | 57057 | **15.00** |

**Removing 20% of parameters by plain SVD of the weights takes LLaMA-7B from 5.68 to 20,061.**
This is the strongest available evidence that pretrained transformer weights are not close to
low-rank in the plain Frobenius metric — and note that the methods which *do* work succeed by
weighting the decomposition with **activation statistics** (§4.2).

**LoRA vs Full Fine-tuning: An Illusion of Equivalence**, https://arxiv.org/abs/2410.21228 — two
findings directly relevant. (i) **Effective rank:** LoRA's update effective rank is *"less than half
that of full fine-tuning and a quarter of the adapter rank"*; at r=768 on RoBERTa it averages ≈300.
**A rank-r factorization does not even use its r degrees of freedom.** (ii) **Rank stabilization is
a prerequisite:** with fixed α=8 across ranks, at r=768 the effective rank *"is never above 100"*,
versus above 768 with α=2r. This makes rsLoRA-style `α ∝ sqrt(r)` / `α = 2r` scaling load-bearing,
not cosmetic. They also introduce **intruder dimensions** — high-singular-value directions
approximately orthogonal to `W_0`'s singular vectors, present in LoRA but *"almost never"* in full
FT — and trace the cause to the **BA product** (freezing a unit-singular-value A and training only
B sharply reduces them).

### 2.7 What breaks — pathologies, with the honest pro

**The PRO — implicit acceleration.** **On the Optimization of Deep Networks: Implicit Acceleration
by Overparameterization**, Arora, Cohen, Hazan. https://arxiv.org/abs/1802.06509 — factorizing a
layer induces a PSD preconditioner that acts as *"an adaptive learning rate"* plus *"a certain type
of momentum that favors movement along the azimuth taken so far"*; Theorem 2 shows for N>2 the
induced field is **not the gradient of any function**, so the effect cannot be replicated by any
regularizer. Also **Algorithmic Regularization in Learning Deep Homogeneous Models**, Du, Hu, Lee,
https://arxiv.org/abs/1806.00900 — gradient flow *"effectively enforces the differences between
squared norms across different layers to remain invariant"*, so at small init the factors
self-balance.

[REASONING — important caveat] **This is about DEPTH at preserved expressiveness, not about a rank
cap.** Arora et al.'s main task is scalar regression, so `W_e` is a 1×d row vector — rank 1 already,
making a width-1 bottleneck literally free. The tell is their MNIST ConvNet experiment, where they
explicitly chose hidden widths as *"the minimal values that do not deteriorate expressiveness."*
Their "overparameterization" means parameter redundancy from factorizing, **not** rank reduction.
And their own **Fig. 5-left** documents the opposite pathology: at depths 4 and 8 with near-zero
init, progress *stalls* — depth-4 moves only after ~65K iterations, depth-8 stuck past 100K — fixed
by near-identity init satisfying balancedness.

**The CONS, each with a citation:**

1. **Hard rank cap**, compounded by **GD's implicit bias toward even lower rank**, which
   *strengthens* with depth — **Implicit Regularization in Deep Matrix Factorization**, Arora et al.
   https://arxiv.org/abs/1905.13655 (predecessor: Gunasekar et al. https://arxiv.org/abs/1705.09280).
2. **Naive per-factor L2 silently penalizes the NUCLEAR norm.** Khodak et al.: by Srebro &
   Shraibman, `(λ/2)(‖U‖²_F + ‖V‖²_F) ≥ λ‖UV^T‖_*`, and their Fig. 1 shows the bound is tight
   throughout ResNet20 training. So per-factor weight decay adds a **third** rank penalty on top of
   the cap you already imposed.
3. **Realized effective rank is under half the nominal rank** (§2.6). Four independent
   rank-reducing pressures stack.
4. **Per-factor LR mismatch is provable.** LoRA+ (https://arxiv.org/abs/2402.12354) Proposition 1:
   with a single LR it is **impossible** for both factors' feature updates to be Θ(1); Theorem 1
   gives efficiency at `η_A = Θ(n^{-1})`, `η_B = Θ(1)`, i.e. **ratio Θ(n)**. They name the shape
   explicitly: LoRA layers are *"2-layers linear networks with a 'bottleneck' in the middle"* which
   *"might induce some numerical challenges in training stability and efficiency."*
5. **GL(r) gauge degeneracy.** **Riemannian Preconditioned LoRA**, Zhang & Pilanci,
   https://arxiv.org/abs/2402.02347: *"each (A,B) pair is equivalent to (AO, BO^{-1}) for any
   O ∈ GL(r)"*. And the update is *"approximately constrained to the column space of A_t and the row
   space of B_t"* — `X_{t+1} ≈ X_t - ηA_tA_t^T∇L - η(∇L)B_tB_t^T`. Their fix right-multiplies each
   gradient by the inverse Gram of the *other* factor (with damping δ=1e-6). Gains are large where
   conditioning is bad: Mistral-7B GLUE avg SGD 71.76 → 80.49 scaled.
6. **Reduced effective step size through the bottleneck.** Khodak Claim 3.1: with normalization,
   effective step ≈ `η/‖W‖²_F`, and the direction updates by a projection of
   `∇̂ = ∇_Ŵ VV^T + UU^T∇_Ŵ` — the `VV^T`/`UU^T` factors are the bottleneck acting on the gradient.
7. **Empirical LR corrections DISAGREE IN DIRECTION across papers** — ReLoRA needs 1.5-2x *larger*;
   LTE needs 0.05-0.1x *smaller*; Kamalakara *halves* it; LoRA+ needs a Θ(n) *ratio*.
   [REASONING] The only safe reading is that **a factorized layer does not inherit a tuned dense LR,
   and the correct value depends on r.** Budget a real LR sweep per rank.
8. **Small-init saddle proximity.** **Saddle-to-Saddle Dynamics in Deep Linear Networks**,
   https://arxiv.org/abs/2106.15933 (for small init variance, init is *"close to a saddle point and
   far from any global minimum"*; the multi-saddle trajectory is explicitly **conjectural**);
   **Gradient Descent Can Take Exponential Time to Escape Saddle Points**,
   https://arxiv.org/abs/1705.10412; **Exact solutions to the nonlinear dynamics of learning in deep
   linear networks**, Saxe et al. https://arxiv.org/abs/1312.6120 (unsupervised pretraining finds the
   fast-learning region *"while scaled random Gaussian initializations cannot"*; recommends
   orthogonal init).

[GAP — the sharpest one in this dossier] **No paper cleanly proves that the origin of a
rank-constrained BA layer is a degenerate saddle causing stalling.** The composite argument is mine:
(i) `B=0` ⟹ output ≡ 0 and `∂L/∂A = 0`, so the origin is stationary; (ii) small init sits near
saddles; (iii) GD can need exponential time to escape; (iv) Arora Fig. 5-left directly observes
factorized stalling at near-zero init. Direct empirical evidence exists but a single citable theorem
does not.

### 2.8 [SYNTHESIS] Why §2.4 is NOT fatal to this experiment — the scope argument

This is the most important reasoning step in the dossier, and it should be stated explicitly in the
experiment write-up, because a reviewer who knows GaLore Table 2 will otherwise reject the proposal
on sight.

[REASONING] **Every from-scratch failure in §2.4 factorizes ALL (or nearly all) weight matrices.**
GaLore's Low-Rank row, ReLoRA's plain LoRA, LTE's single adapter, Kamalakara's GPT-2 — all replace
the model's *entire* linear-layer budget with rank-r products, at rank fractions of 1/4 to 0.62.
The LIV proposal is categorically narrower. Measured against the released LFM2-1.2B checkpoint
(exact parameter counts verified by parsing the safetensors header, total 1170.341M — my computed
figure matches to the digit):

| component | params | share of model |
|---|---|---|
| embeddings (tied) | 134.218M | 11.5% |
| SwiGLU MLPs (16 × 3·d·8192) | 805.306M | 68.8% |
| GQA blocks (6 × 10.486M) | 62.915M | 5.4% |
| conv blocks (10 × 16.783M) | 167.772M | 14.3% |
| **— of which the two gate slices** | **83.886M** | **7.17%** |

So the proposal touches **7.17% of the model's parameters**, leaving 92.8% — including every MLP,
every attention projection, every value stream and output projection — fully dense. Contrast with
the failures, which touched ~all of it. Moreover:

- **The value path stays full rank.** In the LIV block the *information* flows through `x̃` and
  `out_proj`, both dense. The factorized tensors only *modulate* it. This is structurally analogous
  to Pufferfish's "hybrid" fix (keep the layers whose approximation error propagates dense) and to
  Compute Better Spent keeping the input layer dense.
- **The precedent in §1 is exactly this shape and it works at scale.** GLA ships rank 16 out of
  d/2; Mamba ships r = d_inner/32; RWKV-6 ships rank 64; RWKV-7 ships ~1.8·sqrt(d). All are
  from-scratch pretrained production models. **These are existence proofs that a low-rank gate,
  specifically, survives from-scratch pretraining** — which is precisely the narrow claim the
  experiment needs, and precisely what §2.4's global-factorization failures do not contradict.

**Savings by rank** (10 conv blocks, d=2048), which is what the Amdahl analysis in §5 must work with:

| r | r/d | gate params | saved | % of model |
|---|---|---|---|---|
| 32 | 1/64 | 2.62M | 81.26M | 6.94% |
| 64 | 1/32 | 5.24M | 78.64M | 6.72% |
| 128 | 1/16 | 10.49M | 73.40M | 6.27% |
| 256 | 1/8 | 20.97M | 62.92M | 5.38% |
| 512 | 1/4 | 41.94M | 41.94M | 3.58% |
| 1024 | 1/2 | 83.89M | 0 | 0% |

[REASONING] **Note the flatness: r=32 and r=256 differ by only 1.6 percentage points of total model
size.** The parameter saving is nearly saturated by r=128 and is capped at ~7% no matter how
aggressive the rank. This has a sharp design consequence: **there is little reason to sweep very
small r.** The marginal parameter gain from r=128 → r=32 is 0.67% of the model, which cannot
plausibly justify any quality risk. **Sweep upward from 128, not downward.**

## 3. Initialization and optimization of factorized layers trained from scratch

This section is the highest-risk part of the experiment, and it is where the literature is
thinnest for the *specific* case at hand. I separate (a) settled arithmetic, (b) published recipes,
(c) my own derivations for the multiplicative case.

### 3.1 [DERIVATION] Init scale arithmetic for a product of two matrices

Let `A ∈ R^{r×d}` with i.i.d. zero-mean entries of variance σ_A², `B ∈ R^{d×r}` likewise with σ_B²,
and `x ∈ R^d` with i.i.d. unit-variance entries. Write `W = BA`. Then for entry (i,j):

```
W_ij = Σ_{k=1..r} B_ik A_kj
Var(W_ij) = r · σ_A² σ_B²                        (independent, zero-mean terms)
```

and for the output `y = BAx`:

```
Var(y_i) = d · Var(W_ij) = d · r · σ_A² σ_B²
```

A dense `d→d` layer at `std = d^{-1/2}` gives `Var(y_i) = d · (1/d) = 1`. **So the gain-matching
condition for the factorized layer is:**

```
    d · r · σ_A² σ_B² = 1        i.e.        σ_A σ_B = 1/sqrt(d·r)
```

This is **one equation in two unknowns** — the product is pinned, the split is free. Three natural
choices, all verified numerically (d=1024, r=128, measured output variance vs target 1.0):

| Scheme | σ_A | σ_B | measured out var |
|---|---|---|---|
| dense reference (`std = d^{-1/2}`) | — | — | 0.9985 |
| **balanced / geometric** `σ_A = σ_B = (dr)^{-1/4}` | 0.05256 | 0.05256 | 0.9964 |
| **fan-in per factor** `σ_A = d^{-1/2}`, `σ_B = r^{-1/2}` | 0.03125 | 0.08839 | 1.0009 |
| **naive: both = 0.02 (`initializer_range`)** | 0.02 | 0.02 | **0.0209** |

**The last row is the trap, and it is the single most likely way this experiment fails silently.**
Reusing the codebase's `initializer_range = 0.02` for both factors makes the gate's output variance
far too small.

> **[CORRECTED 2026-07-30 — arithmetic re-derived independently.]** The error magnitude stated in an
> earlier draft of this section was wrong, though the conclusion is unchanged and if anything
> cleaner. With both factors at std 0.02, `Var(y) = d·r·σ_A²σ_B² = d·r·(0.02)⁴ = d·r·1.6e-7`:
>
> | | r=64 | r=128 | r=256 | r=512 |
> |---|---|---|---|---|
> | **d=1024** | 0.0105 (47.7× too small) | 0.0210 (23.8×) | 0.0419 (11.9×) | 0.0839 (6.0×) |
> | **d=2048** | 0.0210 (23.8× too small) | 0.0419 (11.9×) | 0.0839 (6.0×) | 0.1678 (3.0×) |
>
> (ratios against the Xavier gate target of 0.5, derived below.) The earlier claim that the error is
> `0.0004·d·r` and therefore "~100× too *large*" at d=2048/r=128, with a **sign that flips** with
> `d·r`, is incorrect — it dropped two factors of 0.02. Overshoot would require `d·r > 6.25e6`
> (i.e. r > 3052 at d=2048), which no realistic sweep reaches. **The error is always an
> undershoot, and it is monotone: it shrinks as r grows, by exactly the factor r.**
>
> This makes the trap *more* insidious, not less. Because the miscalibration is monotone in r, a
> naive fixed-std sweep produces a smooth, plausible-looking rank curve that is in fact a **gate
> initialization-scale curve**. It will not look like a bug; it will look like a finding — most
> likely "higher rank is better", which is the expected result and so will not trigger suspicion.

**A rank sweep with fixed init std does not hold gate scale constant across arms, so it does not
measure rank — it measures scale.** This alone invalidates a naive sweep, and the required fix is
one line: set `σ_A σ_B = 1/sqrt(d·r)` (or match the Xavier target below) per arm.

Note the **"fan-in per factor" scheme is what Mamba does** (`dt_init_std = dt_rank**-0.5` on the
up-projection, default init on the down-projection), and its code comment — *"to preserve variance
at initialization"* — is exactly this calculation.

Also relevant: the stock LIV block uses **Xavier**, not `0.02`. Glorot/Xavier for `d→3d` gives
`std = sqrt(2/(d+3d)) = 1/sqrt(2d)`, so the gate slices' effective std is `1/sqrt(2d)`, and the
target becomes `σ_A σ_B = 1/sqrt(2dr)`. Match against *that*.

Standard init references: **Understanding the difficulty of training deep feedforward neural
networks** (Glorot & Bengio, AISTATS 2010),
https://proceedings.mlr.press/v9/glorot10a.html ; **Delving Deep into Rectifiers** (He et al.),
https://arxiv.org/abs/1502.01852.

### 3.2 The one paper that directly addresses from-scratch factorized layers

**Initialization and Regularization of Factorized Neural Layers**, Khodak, Tenenholtz, Mackey,
Fusi (ICLR 2021). https://arxiv.org/abs/2105.01029

This is the most actionable citation in the dossier for §3, and its two recommendations should be
treated as defaults:

1. **Spectral initialization** — initialize `A, B` from the top-r SVD of a *freshly initialized
   dense* matrix `W_0`, splitting the singular values as `sqrt(Σ_r)` into each factor:
   `W_0 = UΣV^T`, then `B ← U_r Σ_r^{1/2}`, `A ← Σ_r^{1/2} V_r^T`. This makes `BA` the optimal
   rank-r approximation of a correctly-scaled dense init, so the product inherits the dense init's
   spectrum (truncated) rather than a random-product spectrum.
2. **Frobenius decay** — apply weight decay to the **product**, `||BA||_F²`, not to the factors
   separately. L2 on the factors is the wrong regularizer because the factorization is
   scale-invariant (`A → cA, B → B/c` leaves `BA` fixed but changes `||A||² + ||B||²`), so per-factor
   L2 penalizes an arbitrary gauge choice and drives the factors toward a *balanced* gauge rather
   than shrinking the map.

[REASONING] The gauge-invariance point generalizes to a warning about **weight decay + AdamW**: the
scale-invariance means the factorized layer has a flat direction that decoupled weight decay will
push on arbitrarily. Practical consequence: either use Frobenius decay, or **exclude the gate
factors from weight decay** (note the precedent — Mamba marks `dt_bias._no_weight_decay = True`,
Mamba-2 marks both `dt_bias` and `A_log`).

### 3.3 muP and the spectral condition applied to low-rank factors

**Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer**, Yang,
Hu, Babuschkin et al. https://arxiv.org/abs/2203.03466 — the standard three weight classes
(input-like: fan_in fixed; hidden: both scale; output-like: fan_out fixed) with Adam LR scaling
`Θ(1/fan_in)` for hidden weights.

**A Spectral Condition for Feature Learning**, Yang, Bernstein et al.
https://arxiv.org/abs/2310.17813 — the condition is on spectral norms:
`||W||_* ≍ sqrt(fan_out/fan_in)` and `||ΔW||_* ≍ sqrt(fan_out/fan_in)`.
See also **Modular Duality in Deep Learning** https://arxiv.org/abs/2410.21265 and
**Old Optimizer, New Norm** https://arxiv.org/abs/2409.20325.

[DERIVATION — flagged as mine, though §3.4 shows it agrees with a published table] Apply the
spectral condition to the composition. If `r` is held FIXED while `d` scales:

- `A: d → r` has fan_in = d, fan_out = r. Target `||A||_* ≍ sqrt(r/d)`.
- `B: r → d` has fan_in = r, fan_out = d. Target `||B||_* ≍ sqrt(d/r)`.
- Composite: `||BA||_* ≤ ||B||_* ||A||_* ≍ sqrt(d/r)·sqrt(r/d) = 1 ≍ sqrt(d/d)` ✓ — consistent with
  a `d→d` dense map.

**The muP class of each factor changes when r is fixed.** With r constant and d growing, `A` has a
*fixed fan_out* (output-like) and `B` has a *fixed fan_in* (input-like). Neither is a standard
"hidden" weight, so **neither gets the plain `1/d` Adam LR rule.** Concretely, `B`'s LR should not
scale with d at all (fan_in = r is constant), while `A`'s should scale as `1/d`. **If instead you
scale r proportionally to d (r = d/16, Mamba-style), both factors are ordinary hidden weights and
the standard muP rules apply unchanged.**

[REASONING] This yields a clean, decision-relevant recommendation: **a fixed-r design (GLA-style
r=16, or "r=128 regardless of d") breaks standard muP LR transfer, whereas an r ∝ d design
(Mamba-style) preserves it.** If the experiment intends to sweep width later or transfer HPs from a
proxy model, prefer parameterizing rank as a *ratio* (r = d/16, d/8, d/4) rather than an absolute
constant. RWKV-7's `~sqrt(d)` rule sits in between and would also break plain muP transfer.

### 3.4 Different learning rates for the two factors — published and convergent

Two independent lines arrive at the same conclusion: **the down-projection and up-projection need
different learning rates, with the up-projection (B) getting the larger one.**

**LoRA+: Efficient Low Rank Adaptation of Large Models**, Hayou, Ghosh, Yu.
https://arxiv.org/abs/2402.12354 — argues from a feature-learning/scaling analysis that setting
`η_A = η_B` is *provably* suboptimal (the two factors have different sensitivities in r), and
recommends `η_B / η_A = λ` with **λ ≈ 2^4 = 16** empirically, reporting 1-2% accuracy gains and up
to ~2x speedup.

**Compute Better Spent: Replacing Dense Layers with Structured Matrices**, Qiu, Potapczynski,
Finzi, Goldblum, Wilson (ICML 2024). https://arxiv.org/abs/2406.06248 — derives per-structure muP
multipliers from the spectral condition. **Table 2, Low-Rank UV: `κ_U = d/2r`, `κ_V = 1/2`.**
The paper notes the asymmetry *"matches the concurrent LoRA+ finding that U should have a higher
learning rate compared to V."*

[REASONING] Note this is a much *larger* asymmetry than LoRA+'s 16: `κ_U/κ_V = d/r`, which at
d=2048, r=128 is **16** — coincidentally matching LoRA+ exactly at that ratio, but which would be
**64** at r=32 and **4** at r=512. **So the LR ratio should be set to d/r, not to a constant 16.**
This is a concrete, cheap, high-value design decision: it makes the LR treatment correct across the
whole rank sweep instead of only at one rank.

Their general recipe, per dense core of a structured layer:
`init std = Θ(sqrt(min(d_in^i, d_out^i)/(d_in^i)²))`, `Adam LR = Θ(1/d_in^i)`, transfer
`η_i* = κ_i · η*` with `κ_i = (d_in/d_in^i)·δ_i` and heuristic `δ_i = 1/k` for k learnable cores.
They also report that BTT transformers **required weight normalization**
(`M̃ = γ_M · min(1, σ_M/RMS(M)) · M`), without which activations *"grow without bound"* and
*"lead to NaN"* — a direct precedent for needing a norm guard on structured layers.

**rsLoRA: A Rank-Stabilized Scaling Factor for Low-Rank Adaptation**, Kalajdzievski.
https://arxiv.org/abs/2312.03732 — shows LoRA's `α/r` scaling causes gradient collapse at higher
ranks and that the rank-stable choice is **`α/sqrt(r)`**. Directly relevant: it is the same
"keep the product's scale invariant as r changes" requirement as §3.1, expressed as a scaling factor.

**The Impact of Initialization on LoRA Finetuning Dynamics**, Hayou et al.
https://arxiv.org/abs/2406.08447 — init-A-random/B-zero vs A-zero/B-random are *not* equivalent;
one permits larger stable LRs. Confirms the two factors are not interchangeable.

**DoRA: Weight-Decomposed Low-Rank Adaptation**, https://arxiv.org/abs/2402.09353 — decouples
magnitude from direction. [REASONING] The magnitude/direction split is a plausible cheap guard for
a gate: learn a per-channel magnitude vector (d params) times a normalized low-rank direction. This
is close to what Mamba/RWKV achieve with a full-width bias/base, and worth listing as a variant.

### 3.5 [KEY RISK — mostly my own derivation; literature is silent] Low-rank output into a MULTIPLICATIVE path

The literature does **not** answer this question directly. [GAP] I found no paper studying
instability of a low-rank factor feeding a multiplicative/gating path as opposed to an additive
residual path. What follows is derivation plus numerical verification, and it should be treated as
the experiment's primary hypothesis about *how* it might fail.

**Fact 1 — the LIV gate is unbounded, unlike every gate in §1.** All prior low-rank gates pass
through sigmoid / `exp(-exp(·))` / softplus / `σ(·)^{1/τ}`. LFM2's `B * x` and `C * z` are raw
products of linear projections. So the usual failure mode (*gate saturation*, which all of
HGRN's lower bound, Griffin's `c=8` log-space form, GLA's τ=16 and Mamba's inverse-softplus
`dt_bias` exist to prevent) **does not apply here.** The opposite mode does: unbounded variance and
heavy tails.

**Fact 2 [DERIVATION + numerically verified] — a multiplicative path amplifies KURTOSIS, and the
double gate compounds it.** For independent zero-mean unit-variance `u, v`: `Var(uv) = 1` (variance
is preserved), but `E[(uv)^4] = E[u^4]E[v^4] = 9` for Gaussians, so **kurtosis goes 3 → 9.**
Measured on the actual LIV computation graph (d=512, gates and value all projections of the same
`h`, so correlated — the realistic case):

| stage | variance | kurtosis | max abs |
|---|---|---|---|
| `x̃` (linear projection) | 0.994 | 3.02 | 4.73 |
| `y = B ⊙ x̃` (one gate) | 0.992 | **9.05** | 10.54 |
| `o = C ⊙ z` (double gate) | 1.002 | **26.78** | 19.97 |

Variance is preserved at every stage, so **a variance-based init check will pass while the tails
blow up.** Kurtosis compounds roughly 3 → 9 → 27 through the two gates. In bf16 with a 4x wider
dynamic range demand, this is the mechanism by which a "correctly initialized" block still produces
outliers. Note that Griffin's released code carries a bespoke gradient-clipped sqrt
(`SqrtBoundDerivative`) specifically to guard bf16 NaNs in its gate path — independent evidence
that multiplicative paths need numerical guards.

**Fact 3 [DERIVATION + verified] — scale errors SQUARE through a double gate.** If both gate
projections are mis-scaled by a factor `s`, the block output scales by `s²`:

| gate std error `s` | output std multiplier |
|---|---|
| 0.25 | 0.06x |
| 0.5 | 0.25x |
| 2 | 4.0x |
| 4 | **16.0x** |

Combined with §3.1: the naive "both factors at 0.02" error is a factor of ~sqrt(0.0004·d·r) per
gate, hence **that squared** at the block output. At d=2048, r=128 that is ~105x per gate and
~10^4x at the block output. This is not a subtle degradation; it is a diverged run. **This is the
concrete reason the init treatment is not optional.**

**Fact 4 [REASONING] — the low-rank gate is, if anything, biased toward being too SMALL, which
hurts differently.** A correctly gain-matched rank-r gate reproduces the *variance* of a dense gate
but its output lies in an r-dimensional subspace of R^d. Per-channel gate values are therefore
strongly correlated across channels: with r=128 and d=2048, the gate cannot express 2048
independent per-channel multipliers, only 128 degrees of freedom broadcast through B. **The
functional loss is per-channel gate selectivity, not gate magnitude.** This is precisely the
capacity that Mamba's full-width `dt_bias` and RWKV's `λ_□` base restore at O(d) cost, and is the
strongest argument for adding a full-width bias to the factorized gate.

**Fact 5 — normalization inside the bottleneck.** [GAP] I found no established practice of putting
LayerNorm/RMSNorm on a rank-r LoRA-style intermediate. The closest published precedents are
(i) RWKV's `tanh` inside the bottleneck (a bounded nonlinearity, which caps the intermediate),
(ii) Compute Better Spent's weight normalization on structured cores, and (iii) Mamba-2 / Mamba's
`extra norm` before the out-projection (Mamba-2 Table 4: 11.49 with extra norm vs 11.66 without —
so an extra norm in a gated block was worth 0.17 ppl). [REASONING] An RMSNorm on the rank-r
intermediate is cheap (r params), makes the layer scale-invariant in `A` (which incidentally fixes
the gauge/weight-decay problem in §3.2), and is worth including as a variant arm — but it is
*non-standard*, so it should be a labelled arm, not folded into the main comparison.

### 3.6 Practical init recipe [synthesis — labelled as such]

Ranked by confidence, most-recommended first:

1. **Spectral init** (Khodak et al. 2105.01029): SVD a Xavier-initialized `d×d` dense matrix,
   take top-r, split `sqrt(Σ_r)`. Highest fidelity to the dense control; makes the r→d limit exact.
2. **Fan-in-per-factor** (`σ_A = (2d)^{-1/2}` to match the block's Xavier, `σ_B = r^{-1/2}`) —
   what Mamba does; simplest, and gain-correct by §3.1.
3. **Add a full-width bias** to each factorized gate (2d params total). Universal in prior art
   (Mamba `dt_bias`, GLA `b_α`, RWKV `λ_□`); restores per-channel offsets a rank-r map cannot
   express; exempt it from weight decay.
4. **LR ratio `η_B/η_A = d/r`** (Compute Better Spent Table 2; agrees with LoRA+ at r=d/16).
5. **Exclude gate factors from weight decay**, or use Frobenius decay on `||BA||_F`.
6. Do **not** use LoRA's `B = 0` init. It is available only because LoRA adds to a frozen dense `W`;
   here the product *is* the gate, and `B = 0` would zero the gate, killing the block's output
   entirely at step 0 (and `y = B ⊙ x̃ = 0` gives no gradient to the conv). **Flag this explicitly:
   the single most famous low-rank init trick is inapplicable to this setting.**


---

## 3B. Additions to §3 (init/optimization) from deeper search

### 3B.1 [THE CENTRAL TENSION — DERIVATION] You cannot match both RMS and spectral norm

This is the sharpest technical result in the dossier and it should drive the design.

Using the random-matrix fact that a p×q Gaussian with entry std `s` has `‖·‖_2 ≈ s(sqrt(p)+sqrt(q))`
(stated in the spectral-condition paper, arXiv 2310.17813):

**Option 1 — match the dense RMS/gain** (σ_A² = 1/d, σ_B² = 1/r; §3.1's fan-in-per-factor, which is
what Mamba does). Then `‖BA‖_F = sqrt(d)`, identical to dense. But `BA` has rank r, so
`‖BA‖_2 ≈ ‖BA‖_F/sqrt(r) = sqrt(d/r)` — the **spectral norm is inflated by sqrt(d/r)**, a factor of
**4x at d=2048, r=128.**

**Option 2 — match the spectral condition** (σ_A = sqrt(r)/d, σ_B = 1/sqrt(r), which follows from
applying `‖W‖_2 ≍ sqrt(fan_out/fan_in)` to each factor: `‖A‖_2 ≍ sqrt(r/d)`, `‖B‖_2 ≍ sqrt(d/r)`, product
≍ 1 ✓). Then the composite gain is `d·r·σ_A²σ_B² = r/d`, so the gate pre-activation std is
**sqrt(r/d) = 1/4 of dense.**

**A rank-r product cannot simultaneously match a dense layer's entry/RMS scale and its spectral
norm.** This is not an artifact of a bad init choice — it is a structural consequence of rank-r
structure, since `‖·‖_F/‖·‖_2 ≤ sqrt(rank)`.

Why each horn hurts, specifically for a *gate*:
- Option 1's 4x spectral inflation lands exactly in the regime Muon's paper
  (**Muon is Scalable for LLM Training**, https://arxiv.org/abs/2502.16982) associates with harm:
  they note small max(fan_in,fan_out) means *"updates too large, causing training instabilities"*, and
  their Lemma 1 gives update RMS `sqrt(1/max(A,B))` for full-rank and (App. A)
  **`sqrt(r/mn)` for a rank-r matrix.** It is also the regime that
  **Stabilizing Transformer Training by Preventing Attention Entropy Collapse**
  (https://arxiv.org/abs/2303.06296) links to instability: their Theorem 3.1 shows the minimum
  reachable attention entropy falls **exponentially in the spectral norm**, `Ω(T σ e^{-σ})`, and their
  Prop 3.2 shows spectral norms grow like `sqrt(width)` under naive parameterization with Adam.
- Option 2's 4x deflation pushes the gate toward its **linear regime** (see §3B.2).

[REASONING] **The literature's consistent resolution is to refuse the dilemma: initialize the
low-rank path SMALL (spectral-consistent) and put the operating point in a BIAS.** That is exactly
what Mamba (inverse-softplus of log-uniform [1e-3, 1e-1]), Griffin (Λ so `a^c ~ U[0.9, 0.999]`,
c=8), HGRN (learned per-layer lower bound), GLA (τ=16), and Gated DeltaNet all do. **Decouple the
operating point from the projection scale.** For the LIV block, whose gates are linear and unbiased,
this means: add a full-width bias initialized to **1.0** (identity gate), and let the low-rank path
learn a small multiplicative deviation around it.

[REASONING] That bias-at-1.0 choice deserves emphasis because it is the LIV-specific analogue of
LoRA's `B=0` trick, and it *restores* the property §2.1 said was lost: with `gate = 1 + BAx`, the
block at init computes `y = x̃` (an identity gate), the conv and out_proj receive full signal and
gradient, and the factorized path starts as a small perturbation. **This recovers a benign
initialization without needing a dense W to add to** — it is the single most valuable design
recommendation in this dossier, and it is cheap (2d params).

### 3B.2 [REASONING] A low-rank gate is likely UNDER-driven, not over-driven

Under Option 2 the gate pre-activation std is `sqrt(r/d)`, 1/4 of dense at r=128, d=2048.
Consequences, all pointing the same way:

- For a *bounded* gate (sigmoid etc.) this sits in the near-linear region, so the gate degenerates
  toward a learned **constant** — losing exactly the input-dependent selectivity that motivates a
  gate. Note this predicts a specific, testable failure signature: a factorized-gate model whose
  gate output *variance across tokens* is much lower than the dense control's.
- μA's result (below) is the analogous statement: under LoRA's standard init, `δ¹ = Θ(r^{-1/2})`, so
  **A behaves nearly like a frozen random projection** and learning flows mainly through B.
- For the LIV block's *unbounded linear* gate there is no saturation, so the failure mode is instead
  a systematically **weaker gate** — the block drifts toward `y ≈ const · x̃`, i.e. toward an
  ungated depthwise conv. **Diagnostic: log the ratio std(B)/mean|B| per layer against the dense
  control.** If the factorized arm's gates are less input-dependent, that is the mechanism, and it is
  fixable with a temperature/scale rather than more rank.

### 3B.3 μP for low-rank factors — resolved by a 2026 paper, and it confirms the derivation

**Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning**, Chen, Villar, Hayou.
https://arxiv.org/abs/2602.06204 — introduces **Maximal-Update Adaptation (μA)**, a μP analogue
taking the **joint limit n, r → ∞** rather than width-only. This is the first work that pins the
**rank** exponent as well as the width exponent, and it directly answers the question §3.3 posed.

With α = r^{-γ} the LoRA scaling factor, under Init[A] (A random, B=0), their Corollary 4.4:

```
η = Θ( n^{-1/2} · r^{-(1-γ)/2} )
```

| γ | α | η | Z_A |
|---|---|---|---|
| 0 | 1 | n^{-1/2} r^{-1/2} | n^{1/2} r^{-1/2} |
| 1/2 | r^{-1/2} (rsLoRA) | n^{-1/2} r^{-1/4} | n^{1/2} r^{-1/4} |
| 1 | r^{-1} (standard LoRA) | n^{-1/2} | n^{1/2} |

Key findings: the rank-dependent regime scales as **η ∝ r^{-1/2}, not 1/r** — so a 4x rank increase
calls for **halving** the LR. Term scalings under Init[A]: `δ¹ = Θ(r^{-1/2})`, `δ² = Θ(1)`,
`δ³ = Θ(r^{-1/2})`. Empirically the optimal LR shifts monotonically left with rank, ~one log₂ unit
per 4x rank. **Directly actionable: an LR tuned at r=128 is wrong at r=512 by ~2x, so a rank sweep
must re-tune LR per rank or explicitly apply the r^{-1/2} correction.**

**And the μP taxonomy derivation checks out.** [DERIVATION] With r fixed while d scales:
`A: d→r` has fan_in=d (grows), fan_out=r (fixed) ⇒ **output-like** ⇒ Adam LR ∝ 1/d.
`B: r→d` has fan_in=r (fixed), fan_out=d (grows) ⇒ **input-like** ⇒ Adam LR ∝ Θ(1).
Predicted ratio `η_B/η_A = Θ(d)` — which is **exactly LoRA+'s Theorem 1** (`η_A = Θ(n^{-1})`,
`η_B = Θ(1)`), derived there by a completely different route. Strong mutual corroboration:
**LoRA+'s asymmetric LR is what μP's input/output taxonomy predicts once you notice A is
output-like and B is input-like.**

[REASONING — a caveat that matters] μP's *output-weight* init variance is `1/fan_in² = 1/d²`, which
is designed so a final readout starts negligible. For a bottleneck feeding a **gate** you want Θ(1)
pre-activations, and σ_A = 1/d gives `Var(u) = d/d² = 1/d → 0`. **So take the output-like LR scaling
but NOT the output-like init**; use σ_A = d^{-1/2} (LeCun) or the spectral value sqrt(r)/d. The init
and LR class assignments come apart here because A is a hidden-ish bottleneck wearing output-like
shape. Note also `sqrt(r)/d` vs `1/d` differ only by `sqrt(r)`, a d-independent constant — both are
correct in *width* scaling, and **the sqrt(r) is precisely the rank dependence μP leaves unpinned
and μA resolves.**

**Also relevant: LoRA+'s recommended λ, recovered from the PDF (§5.3, Fig. 7), not any abstract:**
λ = η_B/η_A ≈ **2⁴ = 16** with Init[2]; **2²-2³** with Init[1]; **2¹-2²** for Llama. They stress it
*"is model and task sensitive and shows significant variance."* Gains: ~1-2%, up to 2x speedup;
Llama-7b flan-v2→MMLU ~1.3% over the best equal-LR setting.

**The Impact of Initialization on LoRA Finetuning Dynamics**, https://arxiv.org/abs/2406.08447 —
Init[A] (A random, B=0) tolerates a **maximal stable LR of Θ(n^{-1/2})** but with bottleneck
features growing as Θ(n^{1/2}) (they call this **"internal instability"**, an explicit
feature-learning/stability tradeoff); Init[B] caps at Θ(n^{-1}) with no internal instability but B
effectively undertrained (`B_t → B_0` in the limit). **Verdict: Init[A] wins.** RoBERTa-Large MNLI
r=8: **90.69 (Init[A], η*=8e-5) vs 89.47 (Init[B], η*=1e-5)** — an 8x LR gap, as predicted.

### 3B.4 [DERIVATION] The second-order term is NOT negligible

`Δ(BA) = ΔB·A + B·ΔA + ΔB·ΔA`. Applying the spectral condition targets
(`‖ΔB‖_2 ≍ sqrt(d/r)`, `‖ΔA‖_2 ≍ sqrt(r/d)`):

```
‖ΔB·A‖_2  ≲ sqrt(d/r)·sqrt(r/d) = 1
‖B·ΔA‖_2  ≲ sqrt(d/r)·sqrt(r/d) = 1
‖ΔB·ΔA‖_2 ≲ sqrt(d/r)·sqrt(r/d) = 1     <-- SAME ORDER
```

**Unlike a dense layer, where ΔW is a first-order perturbation, the factorized composite carries a
Θ(1) quadratic term at every step.** This is not an artifact: it is LoRA+'s δ³ term and μA's
`δ³ = Θ(r^{-1/2})`. Practical implication: **you cannot tune A and B independently as if they were
separate layers** — the interaction contributes at leading order. This is the real reason the LR
ratio and the scale-split matter.

### 3B.5 The Adam scale-split pathology — a reason to prefer spectral (balanced) init

**LoRA-RITE: Transformation-Invariant Low-Rank Adaptation**, https://arxiv.org/abs/2410.20625
(ICLR 2025 oral). Since `(A,B)` and `(AR, BR^{-T})` give the *same product* for any `R ∈ GL(r)`, an
optimizer ought to be invariant to the split. **It is not.** Under Adam without momentum, for
`R = sI`, gradient scaling cancels in the normalization so `δA₂ = δA₁`, `δB₂ = δB₁`, and expanding:

```
A₁B₁ᵀ + (1/s)·δA₁B₁ᵀ + s·A₁δB₁ᵀ + δA₁δB₁ᵀ
```

**The effective update to the product depends on the arbitrary scale split s.** For plain GD it is
worse (`1/s²`, `s²`). They further prove **diagonal preconditioning cannot fix this** — matrix
preconditioning is necessary — and observe empirically that one factor *"often dominates the
optimization process, receiving substantial updates while the other remains nearly fixed."* Their
Theorem 1: any transformation-invariant optimizer with a shared update rule attains efficient
feature learning, i.e. **invariance is an alternative to per-factor LRs.** Gemma-2B: +4.6% on
Super-Natural Instructions vs Adam.

[REASONING] Four ways to handle this, in order of practicality here: (a) **fix the split by
construction via spectral init, which is balanced by design** — cheapest and recommended;
(b) per-factor LRs (LoRA+); (c) an invariant optimizer (LoRA-RITE); (d) Frobenius decay, which is
itself split-invariant. Note (a) and (d) are the two things Khodak et al. already recommend
together, which is a satisfying convergence. Note also that `‖A‖_F² - ‖B‖_F²` is conserved under
gradient *flow* (**Algorithmic Regularization in Learning Deep Homogeneous Models**,
https://arxiv.org/abs/1806.00900), so the init split persists indefinitely under GD — but **Adam
breaks that conservation law**, which is precisely why the split becomes a live pathology.

### 3B.6 rsLoRA's from-scratch translation

**rsLoRA**, https://arxiv.org/abs/2312.03732 — Theorem 3.2: adapters are rank-stabilized **iff the
scaling `γ_r ∈ Θ(1/sqrt(r))`**, because the key step `E[A₀ᵀA₀] = rσ_A I` injects a factor of r, making
both passes scale as `Θ(γ_r² r)`. Conventional `α/r` makes this `∝ 1/r → 0` — *"overly aggressive and
causes gradient collapse as the rank increases."* Evidence: LoRA *"converge[s] to a similar loss
irrespective of the rank"* while **rsLoRA improves monotonically with rank**; best LoRA r=4 got
1.863 ppl at a 10x-larger LR vs **1.836 for rsLoRA r=2048 at default LR** (so the gain is not merely
an effective-LR increase).

[REASONING] **From-scratch translation:** there is no explicit α to set, so rank-stabilization must
be absorbed into σ_B. Per §3.1, `σ_B² = 1/r` achieves exactly this — the composite gain
`d·r·(1/d)(1/r) = 1` is **r-independent** — whereas `σ_B² = 1/d` gives `r/d` and drifts with rank.
**`σ_B² = 1/r` is the from-scratch equivalent of rsLoRA, and it is what Mamba's
`dt_init_std = dt_rank^{-0.5}` implements.** Their §B.3 is a warning worth heeding: scaling A's init
by `1/sqrt(r)` *without* reparameterizing was unstable at r=2048.

### 3B.7 [KEY RISK, now with a citation] The GLU/SwiGLU alignment pathology

§3.5 flagged the multiplicative-path risk as unsupported by literature. **There is in fact a
well-documented case, and it is closely analogous to the LIV block.**

**Scaling FP8 training to trillion-token LLMs**, https://arxiv.org/abs/2409.12517

`SwiGLU(x) = (xᵀw₁) · Swish(xᵀw₂)`. Because **two linear projections of the same input are
multiplied**, the output grows **quadratically** in input scale (with `w₁ = w₂` and `w₁ᵀx = 1`,
`SwiGLU(cx)/c² → 1`), whereas ReLU/GELU/Swish are at most linear.

**Their Theorem 1 is the alarming part:** under ℓ₂ regularization, at a stationary point where
`σ'(xᵀw₂) → 0`, **`w₁ → w₂` or `w₁ → -w₂`.** Proof: zeroing gradients yields a symmetric
`A = Σ λ_n x_n x_nᵀ` with `w₁ = Aw₂, w₂ = Aw₁`, so both are eigenvectors of `A²` with eigenvalue 1.
Crucially, *"weak regularization will lead to larger weight norms, strengthening this effect"*, and
the result **generalizes to other GLU variants** since no Swish-specific property is used.

**Timeline: the elementwise correlation between `w₁` and `w₂` in the outlier channel
*"increases drastically between 125B and 210B tokens"*, with FP8 divergence at ~200-220B.** Prior FP8
work topped out at 100B tokens, so this was **invisible below 100B.** Llama2-7B, 2T tokens:
FP8+Smooth-SwiGLU reached +33.5% throughput at parity (HellaSwag 68.3 vs 68.37); **plain FP8 was
faster but diverged.** A GELU control at 125M showed no instability.

[REASONING] **Direct relevance:** the LIV block computes `B ⊙ x̃` where **B and x̃ are two linear
projections of the same hidden state h** — structurally identical to SwiGLU's setup, and *worse* in
one respect (no Swish to bound either factor) and better in another (no saturation to trigger the
`σ' → 0` condition in Theorem 1, so the specific alignment proof does not directly apply). The
transferable lesson is the **diagnostic**: log the elementwise correlation between the gate slice and
the value slice, per layer, and treat rising correlation as an early-warning signal. It moved late
(125B-210B tokens), so a short run will not surface it — state this as a *scope limitation* of any
sub-100B-token experiment rather than pretending the risk is excluded.

Corroborating: **PowLU**, https://arxiv.org/abs/2605.25704, attributes SwiGLU instability to the same
"approximate quadratic amplification." **Dissecting Outlier Dynamics in LLM NVFP4 Pretraining**,
https://arxiv.org/abs/2602.02047, names SwiGLU **and gating in linear attention** as outlier sources,
with outliers shifting from transient spikes to persistent hot channels.

Note also **GLU Variants Improve Transformer** (https://arxiv.org/abs/2002.05202) offers **no scale
analysis whatsoever** — Shazeer attributes the gains to *"divine benevolence."* The instability was
found 4.5 years later at 200B+ tokens. [GAP] So the standard multiplicative FFN has no published
init theory, and the bf16 blast radius of the alignment mechanism is **unquantified** (the theorem is
precision-independent, but only FP8 divergence is demonstrated).

### 3B.8 Instability lessons from Wortsman et al.

**Small-scale proxies for large-scale Transformer training instabilities**,
https://arxiv.org/abs/2309.14322 — several findings bear directly on this design:

- **Attention logits are dangerous because they are the one feature whose magnitude depends
  QUADRATICALLY on parameter RMS** (entries are `⟨XW₁, XW₂⟩`). **Every run with max logit above 1e4
  diverged.** [REASONING] A multiplicative gate has exactly the same quadratic-in-RMS structure, so
  the same reasoning applies — and it is the reason §3.5's "scale errors square" result matters.
- **qk-layernorm** let them train 1.2B at LR 0.3. The analogue here is a norm on the gate path.
- **μP stabilized the optimal LR but "does not improve loss or reduce LR sensitivity"** and did not
  remove the need for qk-layernorm. **Important: μP is not a substitute for architectural gate
  control.** Do not assume correct μP scaling makes a gate safe.
- **"Scaling depth increases LR sensitivity at a faster rate than scaling width."**
- **AdamW's default ε=1e-8 is too large at scale**: at 4.8B/LR 0.3, ε=1e-15 improved loss and
  mitigated a collapse in gradient RMS, while ε=1e-6 diverged. Gradient RMS falls with both
  parameter count and LR, and at the largest scale tested lands **around the default ε**, after
  which *"parameters will not receive learning signals as intended"* and *"a layer may collapse."*
  [REASONING] **B's small fan_in makes it the prime collapse candidate — monitor per-factor gradient
  RMS against ε, and consider ε=1e-15.**
- **Independent (decoupled) weight decay reduces LR sensitivity** vs the PyTorch/Optax default form.
- **z-loss with coefficient 1e-4** fixes output-logit divergence, which occurs *"towards the end of
  training"* in models with no weight decay, **regardless of scale.**
- Their extrapolation methodology is directly reusable: fit a *model characteristic* (not loss)
  quadratically across scales per LR to predict instability. They predicted a 4.8B model would be
  unstable at LR 1e-2 without qk-layernorm and confirmed it.

### 3B.9 DoRA — the cleanest mechanism for gate scale control

**DoRA: Weight-Decomposed Low-Rank Adaptation**, https://arxiv.org/abs/2402.09353 —
`W' = m · (W₀ + BA)/‖W₀+BA‖_c` with `‖·‖_c` a **column-wise** norm and `m` a learnable 1×k magnitude
vector. The gradient (their Eq. 6) is rescaled by `m/‖V'‖_c` **and projected away from the current
weight**, pushing the gradient covariance toward identity — *"advantageous for optimization"* and
*"enhancing the learning stability of LoRA."*

**The rank-robustness result is the headline for this experiment:** at r=8 LoRA collapses to 40.7
while **DoRA holds 77.9**; at r=4, 39.5 vs 61.9. Decomposition correlations (ΔD vs ΔM): full FT
−0.62, DoRA −0.31, **LoRA +0.83** — LoRA is *forced* to move magnitude and direction together.

[REASONING] **This is the most directly transferable idea for gate scale control**: let a cheap,
well-conditioned per-channel vector `m` (d params) set the gate's output scale while the low-rank
product handles direction only. It is structurally the same move as Mamba/Griffin putting the
operating point in a bias, and it composes with the bias-at-1.0 recommendation in §3B.1.

### 3B.10 [GAP — a possible novel contribution] Normalization inside the bottleneck

A systematic search found **no paper that places LayerNorm or RMSNorm on the rank-r intermediate of a
low-rank factorization.** The nearby work all normalizes *elsewhere*:

- **Parameter-Efficient Transfer Learning for NLP** (Houlsby et al.,
  https://arxiv.org/abs/1902.00751) puts LayerNorm **after** the adapter, not inside the bottleneck.
  Worth noting their **near-identity init**: zero-mean Gaussian, **std 1e-2, truncated at 2σ**, so the
  adapter ≈ identity at start and *"the original network is unaffected when training starts."* Their
  robustness sweep over std ∈ [1e-7, 1] is stable at or below **1e-2** and degrades above it, with
  models sometimes **failing to train** when init strays far from identity. This is independent
  support for the "start the factorized path small, near identity" recommendation in §3B.1.
- **GLA** normalizes after the layer output and on the output-gate path — outside the bottleneck.
- **DoRA** normalizes **columns of the composite W** — the closest published thing.
- **σReparam** normalizes the **spectral norm of each weight matrix** — applicable per factor.
- **PiSSA** (https://arxiv.org/abs/2404.02948) uses the **same sqrt(S)-split-between-factors as
  Khodak's spectral init**, but on a *pretrained* W. Its ablation (**principal > medium > minor**
  singular directions) independently confirms the spectral-init intuition that the top subspace is
  what matters.

[REASONING] An RMSNorm on the rank-r intermediate would (a) make the layer **invariant to the A/B
scale split**, killing §3B.5's Adam pathology by construction, and (b) pin the bottleneck scale
independently of d and r, addressing §3B.1's tension head-on. Costs: it breaks the exact linearity of
`BA` (the layer can no longer be folded into a dense matrix, which matters for the deployment story)
and makes `‖BA‖_F` no longer the right object for Frobenius decay. **Nobody appears to have tried
it; it is a defensible novel contribution rather than a re-implementation — but it should be a
clearly labelled extra arm, not folded into the main comparison.**


---

## 4. Post-hoc compression as a diagnostic: measuring whether a trained gate is effectively low-rank

### 4.1 Effective-rank metrics — formulas and canonical citations

Let `W` have singular values `σ_1 ≥ σ_2 ≥ ... ≥ σ_n > 0`.

**(1) Stable rank / numerical rank.**
```
    srank(W) = ||W||_F² / ||W||_2² = (Σ_i σ_i²) / σ_1²
```
Range `[1, rank(W)]`. Canonical: Rudolf & Vershynin, **Sampling from large matrices: an approach
through geometric functional analysis**, J. ACM 54(4), 2007, https://doi.org/10.1145/1255443.1255449
(arXiv https://arxiv.org/abs/math/0503442); see also Tropp, **User-friendly tail bounds for sums of
random matrices**, https://arxiv.org/abs/1004.4389.
[REASONING] Caution: stable rank is dominated by `σ_1` and is therefore a *pessimistic*, very
low-valued statistic for heavy-tailed spectra — a matrix with one dominant direction has srank ≈ 1
even if 500 directions carry real signal. Report it, but do not use it as the primary metric.

**(2) Effective rank (spectral entropy).** With `p_i = σ_i / Σ_j σ_j`:
```
    erank(W) = exp( H(p) ),      H(p) = -Σ_i p_i log p_i
```
Canonical: Roy & Vetterli, **The effective rank: a measure of effective dimensionality**,
EUSIPCO 2007, https://ieeexplore.ieee.org/document/7098875 (also
https://infoscience.epfl.ch/record/110188). This is the best-behaved single number: it equals `k`
exactly for a flat rank-k spectrum and degrades gracefully.

**(3) Participation ratio.**
```
    PR(W) = (Σ_i σ_i²)² / Σ_i σ_i⁴          [singular-value form]
    PR    = (Σ_i λ_i)²  / Σ_i λ_i²          [eigenvalue form, λ_i = σ_i²]
```
Used for neural dimensionality in Gao, Trautmann, Yu, Santhanam, Ryu, Shenoy, Ganguli,
**A theory of multineuronal dimensionality, dynamics and measurement**,
https://doi.org/10.1101/214262 . Note the eigenvalue form of PR *is* the stable rank of the
Gram matrix; the two families are related.

**(4) Energy captured at rank r, and optimality of truncation.**
```
    E(r) = Σ_{i≤r} σ_i² / Σ_i σ_i²
```
Eckart & Young, **The approximation of one matrix by another of lower rank**, Psychometrika 1(3),
1936, https://doi.org/10.1007/BF02288367 (with Mirsky's extension to all unitarily invariant norms):
truncated SVD is the optimal rank-r approximation in both Frobenius and spectral norm. So `E(r)` is
a *tight* statement about the best achievable rank-r reconstruction of `W` itself.

**(5) Spectral decay exponent / heavy-tailed self-regularization.** Martin & Mahoney,
**Implicit Self-Regularization in Deep Neural Networks**, https://arxiv.org/abs/1810.01075, and
**Traditional and Heavy-Tailed Self Regularization in Neural Network Models**,
https://arxiv.org/abs/1901.08276 — fit a power law `ρ(λ) ~ λ^{-α}` to the ESD; well-trained layers
are heavy-tailed with α typically in [2,4]. **Implication for this experiment:** a heavy-tailed
spectrum means there is *no clean gap* to truncate at, so small singular values are not
"noise" — they carry correlated signal. This is a caution against reading a slowly-decaying gate
spectrum as "needs full rank" *or* as "safe to truncate"; the ESD shape alone underdetermines it.

### 4.2 [CRITICAL NUANCE] The spectrum of W alone is the WRONG diagnostic

This point should be stated prominently in the experiment write-up, because it is the difference
between a diagnostic that answers the question and one that does not.

What matters is not the rank of `W` but the rank of the map **restricted to the activation
distribution actually seen.** If activations `x` have covariance `Σ_x`, the relevant object is
`W Σ_x^{1/2}` (equivalently, the SVD weighted by input second moments). A `W` with a flat spectrum
can still act as an effectively low-rank map if `Σ_x` is concentrated; conversely a low-rank-looking
`W` can be critical if it happens to align with the dominant activation directions.

The entire activation-aware compression literature exists because of this, and each paper is a
citation for the point:
- **Language model compression with weighted low-rank factorization** (FWSVD), Hsu et al., ICLR
  2022, https://arxiv.org/abs/2207.00112 — Fisher-information-weighted SVD; weights rows by
  parameter importance to the task loss rather than treating all directions equally.
- **ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models**,
  https://arxiv.org/abs/2312.05821 — scales columns by activation magnitude (handles outlier
  channels) before SVD; training-free.
- **SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression**,
  https://arxiv.org/abs/2403.07378 — whitening transform derived so that truncating a singular
  value has a *provable* direct relation to the output reconstruction loss. This is the cleanest
  formalization of "weight the SVD by the data."
- **DRONE: Data-aware Low-rank Compression for Large NLP Models**, NeurIPS 2021,
  https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html

[REASONING] Concrete recommendation: the diagnostic to log is `E(r)` and `erank` of the
**activation-whitened** gate matrix `W Σ_x^{1/2}` (estimate `Σ_x` from a calibration batch of the
block's input `h`), reported alongside the plain-`W` version. Reporting only plain `W` invites the
criticism that the diagnostic does not bear on the question. This is cheap: one calibration pass,
one Cholesky/eigendecomposition per layer.

### 4.3 Post-hoc SVD compression of trained LLM weights — recoverable ratios

Established consensus, with the caveat that these numbers are for *general* weight matrices in
*decoder LLMs*, not gates:

- **Plain (unweighted) SVD is poor.** ASVD and SVD-LLM both report that vanilla truncated SVD
  degrades catastrophically at modest compression; ASVD's premise is that naive SVD ignores
  activation outliers. SVD-LLM reports vanilla SVD becoming unusable around 20% parameter
  reduction, motivating the whitening.
- **Activation/Fisher-aware SVD reaches roughly 20-30% parameter reduction** with modest
  degradation and no retraining, and degrades sharply beyond ~40-50%. ASVD reports ~10-20%
  compression essentially free and up to ~30% tolerable; SVD-LLM extends the usable range and is
  reported to dominate ASVD, especially at higher ratios and under low-resource conditions.
- **LoRD: Low-Rank Decomposition of Monolingual Code LLMs for One-Shot Compression**,
  https://arxiv.org/abs/2309.14021 — ~39% parameter reduction on code LLMs with small quality loss
  and, notably, *reported latency improvement* because the factors are dense GEMMs.
- **LASER — The Truth is in There: Improving Reasoning in Language Models with Layer-Selective
  Rank Reduction**, Sharma, Ash, Misra, https://arxiv.org/abs/2312.13558 — the striking result that
  **aggressively rank-reducing SPECIFIC matrices (predominantly later-layer MLP weights) can
  IMPROVE task performance**, sometimes dramatically, with no retraining. Their interpretation is
  that higher-order components encode noise/rare-fact interference. This is the strongest published
  evidence that some trained matrices are "effectively low-rank plus noise."

[REASONING] LASER is a double-edged citation and should be presented carefully. It supports "some
matrices are effectively low-rank," but it is a *post-hoc, layer-selective* result on *MLP* weights
and offers no license to make a matrix low-rank *from scratch*, nor any claim about gates.

**Layer-type ordering (consensus across the above):** embeddings and MLP matrices tolerate rank
reduction best; attention out-projection is intermediate; Q/K/V and the first/last layers tolerate
it worst. The recurring, and directly relevant, methodological practice is that
**Compute Better Spent keeps the input layer dense "for low rank to avoid an information bottleneck
at the first layer"** — i.e. structured-layer papers routinely *exempt* specific layers.

[GAP — a positive finding for novelty] **No paper in this literature reports rank sensitivity for
GATE matrices specifically**, and no per-matrix-type breakdown covering the SwiGLU gate vs up vs
down projection at the granularity needed. Likewise I found **no published singular-value spectra
for Mamba/SSM/gated-conv weight matrices** (`dt_proj`, `in_proj`, conv weights). The proposed
post-hoc spectrum analysis of trained LFM2 gate slices would therefore be, as far as this search
can establish, **novel measurement** — and it is cheap, since LFM2 checkpoints are public. Note the
`in_proj` weight is a single `[6144, 2048]` tensor whose first two 2048-row blocks are the B and C
gates and whose third is the value stream, so all three can be compared *within the same matrix*,
controlling for layer and training history. That is an unusually clean natural experiment and I
recommend it as a cheap precursor to any training run (§7).


---

### 4.4 [ORIGINAL MEASUREMENT — the most decision-relevant result in this dossier]

Because §4.3 established that nobody has published rank statistics for gate matrices or for
gated-conv/SSM weights, I ran the measurement directly on the released **`LiquidAI/LFM2-350M`**
checkpoint (d=1024, 16 layers, 10 conv blocks, 6 GQA). The `in_proj` weight is a single
`[3072, 1024]` tensor whose row blocks are `[0:1024] = B gate`, `[1024:2048] = C gate`,
`[2048:3072] = value` — so **B, C, and the value stream can be compared within the same matrix**,
controlling perfectly for layer, initialization, and training history. This is an unusually clean
natural experiment.

**Calibration first** (this is what makes the numbers interpretable, and is the step most such
analyses omit):

| reference (1024×1024) | stable rank | erank | E@128 | r for 90% energy |
|---|---|---|---|---|
| i.i.d. Gaussian (full rank, no structure) | 257.8 | 824.3 | 0.376 | 523 |
| Xavier-scaled Gaussian | 258.0 | 823.9 | 0.376 | 522 |
| **true rank-128 product `B A`** | 52.8 | **123.8** | **1.000** | 99 |

**Measured on trained LFM2-350M** (means over layers):

| tensor | stable rank | erank | E@128 | r for 90% energy | n |
|---|---|---|---|---|---|
| **conv B_gate** | 29.6 | **790.1** | 0.458 | 482 | 10 |
| **conv C_gate** | 26.4 | **770.9** | 0.499 | 459 | 10 |
| conv value | 30.2 | 790.5 | 0.456 | 482 | 10 |
| conv out_proj | 41.1 | 778.5 | 0.466 | 473 | 10 |
| mlp w1 (SwiGLU gate) | 47.9 | 960.0 | 0.328 | 729 | 16 |
| mlp w3 (up) | 178.2 | 975.9 | 0.274 | 753 | 16 |
| mlp w2 (down) | 134.5 | 964.5 | 0.283 | 730 | 16 |
| attn q_proj | 35.7 | 629.7 | 0.673 | 306 | 6 |
| attn out_proj | 111.6 | 748.5 | 0.449 | 444 | 6 |

**Interpretation — and it is a warning, not an endorsement:**

1. **The trained gate matrices are NOT low-rank in any useful sense.** `erank ≈ 771-790` out of 1024,
   i.e. **~77% of full rank**, versus 124 for a genuinely rank-128 matrix. Only 46-50% of spectral
   energy sits in the top 128 directions, and **459-482 directions are needed for 90% of the
   energy.** A rank-128 truncation of a trained gate discards over half its energy.
2. **The gates are barely distinguishable from the value stream.** B_gate erank 790.1 vs value
   790.5 — a difference of 0.4 out of 1024. The hypothesis "gates are intrinsically lower-rank than
   the value path, so factorize them preferentially" is **not supported by the trained weights.**
   If anything C_gate is marginally lower (770.9) — worth noting as the *more* compressible of the
   two, consistent with §1's observation that output-style gates differ from decay gates, but the
   margin is ~2.5%.
3. **The gates are LESS full-rank than the MLP matrices** (erank 771-790 vs 960-976). So *relative*
   to the rest of the model the conv block's projections are the more compressible ones — but the
   absolute level is still ~77% of full rank.
4. **The one genuinely low-rank-ish matrix is `attn q_proj`** (erank 630, E@128 = 0.673, r90 = 306),
   which is expected: it is a GQA query projection whose effective rank is limited by head structure
   and is the subject of Bhojanapalli et al. (§2.3).
5. **Stable rank is misleading and should not be the headline metric.** It reads 26-48 for matrices
   whose erank is 771-976, because it is dominated by σ₁ (§4.1). Note the *random* Gaussian baseline
   has stable rank 258 — *higher* than every trained matrix — so a low stable rank here reflects
   trained spectral concentration in a few top directions, not low rank. **Report erank and E(r);
   report srank only with the random baseline alongside.**

[REASONING] **What this does and does not imply.** It does *not* refute the experiment, for the
reason developed in §4.2: this is the spectrum of `W` in the plain Frobenius metric, not of
`W Σ_x^{1/2}`, and the entire ASVD/SVD-LLM literature exists because those differ enormously (plain
SVD at 20% removal destroys LLaMA-7B while activation-aware SVD at the same ratio costs ~2 ppl).
It is entirely possible that the gates act as effectively low-rank maps on the actual activation
distribution. **But it does refute the easy version of the motivation.** Specifically:

- The claim "trained LIV gates are effectively low-rank, so we can parameterize them that way from
  the start" is **falsified as stated** for the plain weight spectrum. Do not put that claim in the
  writeup without the activation-weighted measurement.
- **The activation-whitened measurement is therefore not optional — it is the load-bearing
  diagnostic**, and it must be run *before* committing compute to training runs. If
  `erank(W Σ_x^{1/2})` is also ~770, the experiment's premise has no support from the trained model
  and the honest framing shifts from "gates are low-rank" to "gates *tolerate* being low-rank," which
  is a *different and weaker* claim requiring the training run to establish.
- A useful cheap precursor: **post-hoc truncate the trained LFM2 gates to rank r and measure the
  perplexity hit** (with and without activation-aware weighting). That costs a few GPU-hours, uses
  a public checkpoint, and bounds the from-scratch experiment's plausibility. If rank-128 post-hoc
  truncation of only the gates is catastrophic, from-scratch may still work (different objects — the
  §2.1 argument cuts both ways) but the motivation must be re-grounded.

**Reproduction note:** the numbers above come from `torch.linalg.svdvals` on fp32-upcast slices of
`model.safetensors` from `LiquidAI/LFM2-350M`; the script is a few lines and should be checked into
the experiment repo so the diagnostic can be re-run on the trained checkpoints.

### 4.5 [ORIGINAL MEASUREMENT, part 2] The activation-weighted spectrum does not rescue the premise

Following §4.2's own methodological demand, I repeated the measurement in an
activation-weighted metric. **Caveat: this uses a PROXY for `Σ_x`** — the second-moment matrix of the
token-embedding table, `Σ_x = E^T E / |V|` — rather than real per-layer activation statistics
harvested from a forward pass. That proxy is legitimate only for the first block; deeper layers see
residual-stream statistics that drift from the embedding distribution. **Treat these as indicative,
and re-run with real activations before relying on them.** (The proxy itself is far from isotropic:
its own erank is 484.9 of 1024, so it does carry real anisotropy.)

Means over the 10 conv layers of LFM2-350M:

| tensor | erank(W) | E@128 | **erank(W Σ_x^{1/2})** | **E@128** |
|---|---|---|---|---|
| B_gate | 790.1 | 0.458 | **748.0** | 0.597 |
| C_gate | 770.9 | 0.499 | **732.8** | 0.617 |
| value | 790.5 | 0.456 | **748.2** | 0.597 |

**The activation weighting moves the numbers in the expected direction but nowhere near far
enough.** Effective rank drops only ~5% (790 → 748), and energy captured at rank 128 rises from
0.46 to 0.60 — still leaving **40% of the energy outside a rank-128 subspace.** And the gates remain
statistically indistinguishable from the value stream (748.0 vs 748.2).

[REASONING] **Conclusion for the experiment's framing, stated bluntly:** the "trained LIV gates are
effectively low-rank" motivation is **not supported by the released checkpoint**, in either the plain
or the (proxy) activation-weighted metric, and the gates are not measurably more compressible than
the value stream they modulate. **The experiment should therefore be motivated and pre-registered as
a *tolerance* question, not a *discovery* question:** "does a gated short-conv block still train to
parity when its gate maps are rank-constrained from scratch, and does that buy real on-device
latency?" — not "gates are secretly low-rank, so exploit it."

That reframing is not a retreat, and it is well-supported: §1's production systems (GLA rank 16,
Mamba r=d/32, RWKV-6 rank 64) demonstrate that gates *tolerate* severe rank constraints from scratch
**even though nothing suggests their dense counterparts would have been low-rank.** Tolerance and
intrinsic low-rankness are different claims, and only the former has evidence. Note this also
neatly explains why nobody found a quality cost: a from-scratch low-rank gate is not approximating
a dense gate — it is learning a different, adequate function.


## 5. Parameter count vs LATENCY — the brainlift's worry is correct, and the Amdahl ceiling is low

### 5.1 [DERIVATION] Roofline arithmetic

For a linear layer over M tokens, `d → d`, with `b` bytes per element:

```
dense:      FLOPs = 2 M d²            weight bytes = d² b        act bytes = 2 M d b
factorized: FLOPs = 2 M d r (x2) = 4 M d r    weight bytes = 2 d r b
            plus an intermediate M r write AND read = 2 M r b
```

**FLOP ratio = 4Mdr / 2Md² = 2r/d.** Break-even at **r = d/2**; FLOPs halve only at **r = d/4.**
**Weight-byte ratio = 2dr/d² = 2r/d** — identical. So both the compute-bound and the
weight-bandwidth-bound analyses give the same threshold, and the intuition "r must be well below
d/2 to win anything" is correct. At d=2048, r=128: ratio 0.125, an 8x reduction in both FLOPs and
weight bytes for the gate projections.

Arithmetic intensity: the dense layer's is `2Md²/(d²b + 2Mdb)`, which for large M approaches `2M/(2b)`
— i.e. **intensity grows with M**, the standard result. In decode (M=1) the layer is entirely
weight-bandwidth-bound and the achievable speedup tracks the **weight-byte ratio**, which is where
low rank genuinely helps. Reference for the hardware constants and the tile/wave-quantization
effects: NVIDIA's **Matrix Multiplication Background User's Guide**,
https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
(A100 bf16 ≈ 312 TFLOPS / 1555 GB/s ≈ 200 ops:byte; H100 SXM ≈ 990 TFLOPS / 3350 GB/s ≈ 295).

**The small-K problem.** The second GEMM `(M×r) @ (r×2d)` has **K = r**. Small K means poor
tensor-core reuse and often needs split-K to fill the machine; and both GEMMs want dimensions that
are multiples of 8/16/64/128 to avoid tile quantization. [REASONING] Practical rule: **keep r a
multiple of 64 (ideally 128).** Note RWKV-7 independently rounds all its ranks to **multiples of
32** (§1.4) — presumably for the same reason. This is a free constraint: adopt it.

### 5.2 [MEASURED — my own benchmark] Factorization does win on the matmul in isolation

Measured on this machine (Apple Silicon, 6 CPU threads, fp32, d=2048, both gates as one `d → 2d`
matmul; min of 7 repeats of 200 iterations). Speedup of `d→r→2d` vs dense `d→2d`:

| M (tokens) | dense (µs) | r=64 | r=128 | r=256 | r=512 | r=1024 |
|---|---|---|---|---|---|---|
| 1 | 561 | **19.6x** | **13.6x** | 4.8x | 1.19x | 0.58x |
| 4 | 2072 | 7.6x | 3.2x | 1.25x | 0.99x | 0.67x |
| 16 | 772 | 3.2x | 2.0x | 0.90x | 0.61x | 0.42x |
| 128 | 1470 | 3.7x | 2.5x | 1.62x | 0.75x | 0.45x |
| 1024 | 8972 | 4.6x | 3.0x | 1.86x | 1.22x | 0.65x |
| FLOP ratio (2r/d) | — | 0.062 | 0.125 | 0.250 | 0.500 | 1.000 |

**Findings, and they are informative in both directions:**

1. **At r ≤ 128 the factorization wins comfortably at every batch size**, and at M=1 (decode) it wins
   *more* than the FLOP ratio predicts (19.6x measured vs 16x FLOP-implied at r=64) — consistent
   with decode being weight-bandwidth-bound, where the win tracks bytes and also improves cache
   residency.
2. **At r = d/2 = 1024 the factorization LOSES at every batch size (0.42-0.67x)** despite having
   *equal* FLOPs. This is direct measurement of exactly the effect the brainlift worried about: two
   kernel invocations plus an intermediate write/read cost real time. **The crossover on this machine
   is around r = d/4 = 512**, where results straddle 1.0x.
3. **The realized speedup consistently UNDERPERFORMS the FLOP ratio at moderate r** (r=256: 1.6-1.9x
   measured vs 4x FLOP-implied). So "FLOPs are not latency" is confirmed quantitatively.

[REASONING] These are fp32 PyTorch CPU numbers, *not* the deployment target (which is
ExecuTorch 8da4w / llama.cpp Q4_0 on Snapdragon). They establish that the matmul-level win is real
at r ≤ 128 and that the loss at r ≥ d/2 is real, but **they do not establish an end-to-end win** —
see §5.3, which is the binding constraint. Treat this table as a sanity check, and re-measure on the
actual backend.

### 5.3 [DERIVATION — THE BINDING CONSTRAINT] The Amdahl ceiling is ~8.8%

This is the single most important number in the latency analysis, and it should be stated up front in
any writeup because it bounds the entire value proposition.

For LFM2-1.2B geometry (d=2048, 16 layers, 10 conv + 6 GQA, ff=8192, V=65536), forward matmul
FLOPs per token total **2072 MFLOP**, of which the two gate projections across all 10 conv blocks are
**167.8 MFLOP = 8.10%.** Decode weight bytes give essentially the same fraction (8.10% excluding
embeddings, 7.17% including them).

```
Amdahl:  speedup(r) = 1 / (1 - p + p·(2r/d)),   p = 0.081
```

| r | gate cost multiplier | end-to-end speedup | % faster |
|---|---|---|---|
| — (free, r→0) | 0 | 1.088x | **8.8% — the absolute ceiling** |
| 64 | 0.062 | 1.082x | 8.2% |
| 128 | 0.125 | 1.076x | **7.6%** |
| 256 | 0.250 | 1.065x | 6.5% |
| 512 | 0.500 | 1.042x | 4.2% |

**Even if the gate projections became entirely free, the model gets at most 8.8% faster.** The reason
is structural and worth stating plainly: **the SwiGLU MLPs are 68.8% of parameters** (805M of 1170M),
so the conv block's gates are a small slice of the model. At r=128 the realistic ceiling is **7.6%**,
and that assumes the factorized matmuls hit their full FLOP-ratio speedup — which §5.2 shows they do
not at moderate r.

[REASONING] **Design consequences, and they are sharp:**
- **A latency claim must be measured end-to-end, not per-layer.** A per-layer microbenchmark showing
  13x will be reported as a 13x win by accident; the honest number is <8%.
- **The experiment cannot be justified primarily on latency.** An 8% ceiling is within the range that
  kernel tuning, quantization choice, or threading achieves without any architecture change — and
  Liquid's own materials claim "2x faster decode and prefill than Qwen3 on CPU" from architecture +
  kernels, which dwarfs 8%.
- **The stronger justification is the parameter/memory budget** (7.17% of model params, §2.8) **spent
  elsewhere.** The credible framing is: "factorized gates free ~73M params at d=2048; reinvested in
  depth/width/MLP they buy X quality" — which makes the *narrower-model control* (§6a) the primary
  comparison, not a latency benchmark.
- **If latency is nonetheless the goal, the MLPs are the target, not the gates.** That is a defensible
  finding to report even though it is negative for the original framing.

### 5.4 Measured evidence from the literature on FLOPs-vs-latency

- **Monarch: Expressive Structured Matrices for Efficient and Accurate Training**,
  https://arxiv.org/abs/2204.00595 — the best case for structured layers actually delivering wall-clock
  gains (they report real end-to-end speedups, from scratch), and the reason Monarch is the strongest
  structured competitor in §6g.
- **The Efficiency Misnomer**, https://arxiv.org/abs/2110.12894 — the methodological citation for why
  reporting parameter count alone is misleading: params, FLOPs, and latency are non-interchangeable
  proxies and conclusions flip depending on which is reported. **This paper is the reviewer's weapon
  and should be cited pre-emptively**, with all three numbers reported.
- **The Hardware Lottery**, https://arxiv.org/abs/2009.06489 — the framing for why structurally
  cheaper is not practically faster.
- **Efficiently Scaling Transformer Inference**, https://arxiv.org/abs/2211.05102 — the standard
  reference for the memory-bound decode regime and operational-intensity reasoning.
- LoRA-serving systems quantify the cost of extra small matmuls: **Punica: Multi-Tenant LoRA
  Serving**, https://arxiv.org/abs/2310.18547, and **S-LoRA: Serving Thousands of Concurrent LoRA
  Adapters**, https://arxiv.org/abs/2311.03285 — both had to write **custom kernels (SGMV/BGMV)**
  because naively batched small-r matmuls were too slow. [REASONING] That is itself the lesson: the
  ecosystem's experience is that rank-r paths need bespoke kernels to be fast, so an off-the-shelf
  `d→r→d` in a deployment runtime should not be assumed efficient.
- LLM SVD-compression papers reporting latency: **ASVD** https://arxiv.org/abs/2312.05821,
  **SVD-LLM** https://arxiv.org/abs/2403.07378, **Palu: KV-Cache Compression with Low-Rank
  Projection** https://arxiv.org/abs/2407.21118. **LoRD**, https://arxiv.org/abs/2309.14021, reports
  ~39% parameter reduction *with* latency improvement — notable because the factors remain dense
  GEMMs.
- Classic vision results documenting the FLOP-vs-wall-clock gap: **Speeding up Convolutional Neural
  Networks with Low Rank Expansions**, https://arxiv.org/abs/1405.3866; **Compression of Deep
  Convolutional Neural Networks for Fast and Low Power Mobile Applications** (Tucker),
  https://arxiv.org/abs/1511.06530; **Accelerating Very Deep Convolutional Networks for
  Classification and Detection**, https://arxiv.org/abs/1505.06798.

### 5.5 CPU / edge specifics — the actual LFM2 target

Liquid's published LFM2 materials (https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
establish the deployment context, and two details matter:

- **Their STAR search for LFM2 replaced the cache-size proxy with direct measurement of "peak memory
  plus prefill+decode speed on Qualcomm Snapdragon embedded SoC CPUs."** So the vendor's own
  methodology treats **measured on-device latency, not parameter count, as the efficiency
  objective** — precisely the brainlift's position, and a strong citation for it.
- Benchmarks are **ExecuTorch with 8da4w quantization** and **llama.cpp with Q4_0**, on a Samsung
  Galaxy S24 Ultra (Snapdragon) and AMD Ryzen HX370. Claim: "2x faster decode and prefill than Qwen3
  on CPU"; LFM2-700M beats Qwen-0.6B on both despite being 16% larger.
- Short convolutions were chosen *because* of the target hardware class and because "kernel libraries
  [are] already optimized for those operations."

[REASONING] Three CPU/edge-specific considerations, flagged as reasoning:
1. **CPU decode is weight-bandwidth- and cache-bound**, so halving weight bytes tends to translate
   more directly to latency than on GPU — this is the regime where low rank helps most, and my M=1
   measurements (19.6x at r=64) are consistent with it. Kernel-launch overhead is a GPU concern, not
   a CPU one.
2. **But quantized kernels may lack a fast path for skinny matrices.** A `2048×128` weight in Q4_0
   (block size 32) is fine, but the `M×r` intermediate must round-trip through a different layout,
   and 4-bit kernels are typically tuned for K being a large multiple of the block size. **This must
   be measured on the actual backend, not assumed.**
3. **[GAP] Quantization × low-rank interaction is a real and under-documented risk.** If a `d→r→d`
   gate runs in 4-bit on device, the rank-r bottleneck concentrates all the layer's information into
   r channels, so per-channel quantization error is no longer averaged over d dimensions. I found no
   paper quantifying quantization sensitivity of from-scratch low-rank factors specifically (the
   QLoRA line quantizes the *frozen base* and keeps adapters in higher precision — the opposite
   configuration). **This should be an explicit, pre-registered measurement, not an afterthought**,
   because it could invert the entire deployment case: a factorization that saves 7% of weights but
   requires the gate factors in fp16 rather than int4 may be net negative.

### 5.6 [SYNTHESIS] Rule of thumb

Sourced where noted; the thresholds are my synthesis of §5.1-5.3 plus the measurements.

| Target | Where `d→r→d` wins | Where it loses |
|---|---|---|
| **Big GPU, training** (large M, compute-bound) | `r ≲ d/8`, and only if r is a multiple of 64-128 | `r ≳ d/4`: two kernels + intermediate + small-K inefficiency erase the FLOP saving |
| **GPU decode** (M=1, memory-bound) | `r ≲ d/4` — win tracks weight bytes, most favourable regime | `r ≳ d/2` (no byte saving at all) |
| **CPU/edge decode** (bandwidth + cache bound) | `r ≲ d/4`; measured 13-20x on the matmul at r ≤ 128 | `r ≥ d/2` measured **0.4-0.7x** (slower despite equal FLOPs) |

**Amdahl bound (mandatory to report):** `speedup = 1/(1 - p + p·(2r/d))` with **p = 0.081** for
LFM2-1.2B geometry. **Ceiling 8.8%; realistic 7.6% at r=128.** Any measured end-to-end number above
that is a measurement error.

### 5.7 [ORIGINAL MEASUREMENT] The trained gate scale — a concrete init target

Trained weight standard deviations in LFM2-350M (d=1024). Xavier for the `d→3d` in_proj gives
`sqrt(2/(d+3d)) = 1/sqrt(2d) = 0.02210`:

| layer | B_gate | C_gate | value | out_proj |
|---|---|---|---|---|
| 0 | 0.02678 | 0.02596 | 0.02677 | 0.02536 |
| 4 | 0.02605 | 0.02545 | 0.02601 | 0.02420 |
| 9 | 0.02572 | 0.02572 | 0.02576 | 0.02380 |
| 15 | 0.02797 | 0.02728 | 0.02817 | 0.02553 |

[REASONING] Three useful facts: (i) trained gate scales stay within ~15-25% of the Xavier init value,
so **the init scale is a good proxy for the trained scale** — a factorized replacement matched to
Xavier at init will be in the right regime throughout; (ii) **B_gate and value track each other to 3
decimal places** at every depth, while **C_gate is consistently ~2-3% smaller** — mild independent
support for treating C as the more compressible gate; (iii) the scales are remarkably uniform across
depth, so a single global init rule is adequate (no per-layer schedule needed).

### 5.8 [ORIGINAL MEASUREMENT — validates the recommended recipe] Init simulated through the block

I simulated each candidate init through the actual LIV computation (d=2048, r=128, gates and value
all projections of the same `h`, Xavier value stream, conv approximated as identity):

| init scheme | gate std | y std | o std | **kurtosis(o)** |
|---|---|---|---|---|
| **DENSE control (Xavier)** | 0.707 | 0.500 | 0.354 | **27.8** |
| naive: both factors 0.02, no bias | 0.204 | 0.144 | **0.030** | 28.7 |
| fan-in per factor (Mamba-style), no bias | 0.704 | 0.497 | 0.352 | 27.8 |
| spectral-condition scale, no bias | 0.251 | 0.178 | 0.044 | 27.8 |
| **spectral scale + BIAS = 1.0 (recommended)** | 0.250 | 0.729 | 0.750 | **4.5** |
| fan-in + BIAS = 1.0 | 0.708 | 0.866 | 1.060 | 13.4 |

Four conclusions, and the last is the important one:

1. **The naive "both factors at 0.02" init produces a block output 12x too small** (0.030 vs the dense
   control's 0.354) — confirming §3.1's warning with the double-gate squaring of §3.5 visible in the
   numbers. This is the failure mode to guard against.
2. **Fan-in-per-factor reproduces the dense control almost exactly** (0.352 vs 0.354, gate std 0.704 vs
   0.707). If the goal is "match the dense block at init," this is the recipe — and it is what Mamba
   uses.
3. **Spectral-condition scaling is 8x too small at the block output** (0.044) — the §3B.1 tension made
   concrete. Do not use it *without* a bias.
4. **The bias-at-1.0 variant is the only scheme that cuts KURTOSIS** — 27.8 → **4.5**, a ~6x reduction
   in tail heaviness, while keeping the output scale healthy (0.750). [REASONING] The mechanism: with
   `gate = 1 + BAx` the gate is dominated by a deterministic constant, so the product `gate ⊙ x̃` is
   close to a *linear* function of `x̃` rather than a product of two random variables — which is
   exactly what removes the fourth-moment amplification derived in §3.5. **This makes the
   bias-at-1.0 recommendation doubly motivated: it fixes the init-scale problem AND it is the only
   intervention found that reduces the heavy tails intrinsic to a multiplicative path.** Note it makes
   the factorized block *better conditioned than the dense control* on this metric (4.5 vs 27.8),
   which is a testable and somewhat surprising prediction: the factorized arm may be *more* stable
   than stock LIV, not less.

### 5.9 [CONFOUND WARNING — original analysis] The gate bias must be added to the DENSE control too

| variant | o std | kurtosis(o) |
|---|---|---|
| DENSE, no bias (**stock LFM2**) | 0.354 | **27.8** |
| DENSE + bias 1.0 (**the fair control**) | 1.061 | **13.5** |
| LOW-RANK spectral + bias 1.0 | 0.751 | **4.5** |

**The bias changes the block's conditioning independently of rank.** If the factorized arm gets a
bias-at-1.0 and the dense control does not, then a measured difference could be entirely attributable
to the bias — the experiment would be measuring the bias, not the rank.

[REASONING] **Mandatory design consequence: run `dense + gate bias` as an explicit arm.** Three arms
minimum on this axis: (i) stock LIV (no bias, dense) — reproduces the published architecture;
(ii) dense + bias — isolates the bias; (iii) low-rank + bias — the proposal. Only (iii) vs (ii)
isolates the effect of rank. This is a cheap addition (2d params, one arm) that converts a
confounded comparison into a clean one, and a reviewer will catch it if you do not.

[REASONING] It also raises a genuinely interesting possibility worth pre-registering as a secondary
hypothesis: **`dense + gate bias` may itself beat stock LIV**, since it reduces output kurtosis 2x at
a cost of 2d parameters. If that is true it is a publishable finding independent of the low-rank
question — and it would mean the stock LFM2 block has a cheap, unexploited improvement.

### 5.10 [DERIVATION + MEASUREMENT] What rank actually costs: per-channel gate diversity

§3.5 Fact 4 argued the real functional loss from a low-rank gate is **per-channel selectivity**, not
magnitude. Here is that made quantitative — and it turns out to obey a clean closed form.

Mean absolute correlation between distinct gate channels (d=2048, over 4000 random inputs; a dense
gate's channels are near-independent, a rank-r gate's cannot be):

| gate | mean |corr| between channels | `sqrt(2/(πr))` prediction |
|---|---|---|
| dense | 0.0215 | — |
| r = 16 | 0.2015 | 0.1995 |
| r = 64 | 0.1025 | 0.0997 |
| **r = 128** | **0.0734** | 0.0705 |
| r = 256 | 0.0547 | 0.0499 |
| r = 512 | 0.0413 | 0.0353 |
| r = 1024 | 0.0330 | 0.0250 |

**The measured values track `sqrt(2/(πr))` closely** — the expected absolute correlation between two
random unit vectors in an r-dimensional space. [DERIVATION] So the interpretable cost of rank r is:

```
    channel-coupling ≈ sqrt(2/(π r))        (independent of d)
```

[REASONING] This is a useful design instrument for three reasons:

1. **It is d-independent**, so it gives a rank rule that transfers across model widths — unlike an
   `r/d` ratio. If a target coupling level is what matters, **r should be held constant as d grows,
   not scaled with d.** That is an argument *for* GLA/RWKV-6's fixed-rank convention and *against*
   Mamba's `r ∝ d` — and it is in tension with the μP argument in §3.3 (which prefers `r ∝ d` for
   clean LR transfer). **Flag this tension explicitly: the two considerations pull opposite ways**,
   and the resolution depends on whether you care more about HP transfer (use `r ∝ d`) or about
   holding the functional cost fixed (use constant r).
2. **It shows r=128 is a sensible operating point**: coupling 0.073, i.e. ~3.4x the dense baseline's
   0.0215 but still small in absolute terms. Below r=64 the coupling exceeds 0.10 and rises steeply
   (the `1/sqrt(r)` tail), which is a principled reason to **not sweep below r=64** — reinforcing the
   parameter-saving flatness argument in §2.8 from an entirely independent direction.
3. **It is a cheap diagnostic to log during training** (§7): if the factorized arm's *trained* channel
   coupling greatly exceeds `sqrt(2/(πr))`, the factors have collapsed to an even lower effective rank
   — exactly the pathology §2.6 documented (realized effective rank under half of nominal).


---

## 5B. CORRECTION to §5: on the actual target hardware, low-rank factorization has MEASURED SLOWDOWNS

**This section supersedes the optimistic reading of §5.2.** My CPU benchmark in §5.2 was fp32 PyTorch
on Apple Silicon; the deployment target is INT8/INT4 on Snapdragon via ExecuTorch/llama.cpp, and
published measurements on **that exact target class** are negative. This is the most important
systems finding in the dossier and it should be read before any latency claim is made.

### 5B.1 [LIT — decisive] FLAR-SVD: 2x parameter and FLOP cuts produced SLOWDOWNS on Snapdragon

**FLAR-SVD: Fast and Latency-Aware Singular Value Decomposition for Model Compression**, Thoma et al.,
CVPR 2025 Mobile AI Workshop.
https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Thoma_FLAR-SVD_Fast_and_Latency-Aware_Singular_Value_Decomposition_for_Model_Compression_CVPRW_2025_paper.pdf

DeiT-Base, 224x224, batch 1, 200 inferences averaged, **Snapdragon 8 Gen 2 INT8** (Qualcomm AI Direct
SDK) and Jetson Orin FP16 — i.e. the same silicon class and quantization regime as LFM2's deployment:

| Method | Params (M) | GFLOPs | Top-1 | **Snapdragon INT8** | **Jetson FP16** |
|---|---|---|---|---|---|
| Base (uncompressed) | 86.6 | 33.7 | 81.8 | **8.0 ms** | **14.6 ms** |
| PELA | 44.1 | 17.0 | 61.1 | 6.0 ms | 22.7 ms |
| FW-SVD | 43.9 | 16.9 | 67.8 | **8.8 ms** | **24.4 ms** |
| ASVD | 43.7 | 16.9 | 67.2 | **9.0 ms** | **24.5 ms** |
| FLAR-SVD (latency-aware) | 49.2 | 19.0 | 78.9 | — | 11.4 ms |

**FW-SVD and ASVD cut parameters and FLOPs ~2x and became SLOWER than the uncompressed baseline on
Snapdragon** (8.8 and 9.0 vs 8.0 ms), and **67% slower on Jetson** (24.4/24.5 vs 14.6 ms). Only
explicitly latency-aware rank search beat the baseline. The authors concede *"the majority of
approaches struggle to replicate the latency improvements seen on the V100"* and that *"optimizing for
Qualcomm is harder, resulting in rank search falling back to basic uniform rank choices."*

**The stated mechanism is exactly the concern:** *"having a disproportion between input and weights
sizes can introduce overheads in inference even at low rank ratios"*, and most sharply — *"the
projection matrix (sized 128 x 128) is not even achieving this inflection point at 10%, a very low
rank."*

### 5B.2 [LIT] Corroborating evidence: skinny GEMMs lose vectorization entirely

- **llama.cpp issue #956** (https://github.com/ggml-org/llama.cpp/issues/956): a LoRA-shaped
  `F32 mul_mat([16 x 5120], [16 x 5120])` took **120 ms vs ~5 ms expected — a 24x deviation.**
  Diagnosis: *"the vectorization of BLAS and GGML happens along the row dimension. **Tall and skinny
  matrices essentially get 0 vectorization.**"* This is ggml, the exact runtime in llama.cpp.
- **ARM KleidiAI** i8mm micro-kernel source (`kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_...`):
  `kai_kr = 16`, `kai_sr = 2`, and `kai_k_roundedup()` rounds K up to `kr*sr = 32`. **K is padded to a
  multiple of 32 on ARM's INT4/INT8 fast path** — an independent 32-multiple constraint alongside
  ggml's `QK_K = 256`.
- **ARM Compute Library** hard-dispatches `args._Msize == 1` to separate GEMV kernels
  (`gemv_batched.hpp`), confirming skinny/decode shapes leave the GEMM path entirely.
- **BLIS** (IPDPS 2014, https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf) gives the mechanism:
  the micro-kernel holds an `m_r × n_r` block of C in registers and accumulates over `k_c` rank-1
  updates, so **register load/store amortizes over K — small K means poor amortization.**

### 5B.3 Why my §5.2 measurement was misleading (and the correction)

[REASONING] My §5.2 benchmark measured **fp32 dense BLAS on Apple Silicon**, where the factorized
form's reduced byte traffic converts fairly directly to time. Two things break that on the real
target: (i) **quantized kernels have fixed block/tile structure in K** (ggml `QK_K=256`, KleidiAI
`kr*sr=32`), so a rank-r reduction dimension is padded and the nominal saving is not realized;
(ii) **skinny matrices leave the vectorized GEMM path**. Quantization shrinks bytes *without changing
kernel shape or op count*; low-rank shrinks bytes *by splitting one GEMM into two skinny ones*. **These
are different mechanisms and the first does not license optimism about the second.**

**Revised rule of thumb for CPU/edge decode** (replacing the §5.6 row):

> **Do not assume a win.** Published measurements on this exact target class show 2x parameter/FLOP
> cuts producing **0-12% slowdowns**. Decode being bandwidth-bound is *necessary but not sufficient*.
> The byte-ratio model of §5.1 is an **upper bound that measurements do not reach.**

**This strengthens the case for r = 256 rather than 128 or below**, since 256 simultaneously satisfies
ggml's `QK_K=256`, ARM's 32-multiple padding, and standard cuBLAS tiling — and FLAR-SVD's finding that
a 128×128 projection *"is not even achieving this inflection point at 10%"* is a direct warning against
small factors. Note this **conflicts with the §7.2 recommendation derived from parameter savings and
channel coupling**, which favoured 128. **Resolution: sweep {256, 512} as the systems-viable range and
treat 128 as the aggressive-quality probe**, reporting latency only for ranks that clear the
microbenchmark gate below.

### 5B.4 [REVISED, and this is now a hard gate] Consequences for the experiment

1. **The pre-training microbenchmark is a GATE, not a sanity check.** Measure `d→r→d` vs `d→d` at
   r ∈ {512, 256, 128, 64} on the **real Snapdragon + ExecuTorch/llama.cpp stack in the real
   quantization format, before training anything.** If the factorized layer is not faster in
   isolation, no accuracy result can rescue a wall-clock claim. This costs hours, not GPU-weeks, and
   it can kill or redirect the project early — which is exactly what a good gate does.
2. **Drop wall-clock as a headline claim.** With a **~8% Amdahl ceiling** (§5.3) and published evidence
   of **0-12% slowdowns** on this silicon class, the predicted gain sits *inside* the measured
   slowdown band. The defensible framing is **parameter/memory reduction plus a gating-science
   result**, with latency reported as measured, whatever the sign.
3. **This substantially raises the relative value of the grouped/block-diagonal control** (§6.4). A
   block-diagonal gate is **one GEMM with a large K** and no skinny intermediate — it avoids every
   mechanism identified above while achieving the same parameter reduction. Given this evidence,
   **grouped is now the favourite on the systems axis**, and the experiment should be prepared for the
   outcome "low-rank matches on quality but grouped wins on latency."

[GAP] Explicitly unmeasured: no documented XNNPACK minimum-K; no oneDNN/MKL GFLOPS-vs-small-K curve;
no ExecuTorch issue quantifying skinny-matmul cost; **no low-rank implementation in llama.cpp at all**
(a clean negative — this is unmeasured territory); and no literature on quantization error amplifying
through a narrow bottleneck.

---

## 6. Competing parameter reductions a reviewer will demand as controls

Detailed evidence per alternative, then a ranking. The ranking is the deliverable; read §6.8 first if
short on time.

### 6.1 (a) Just make the model narrower — THE mandatory control

**Scaling Laws for Neural Language Models**, Kaplan et al. https://arxiv.org/abs/2001.08361 — the
finding that at fixed non-embedding parameter count N, loss depends only **weakly on shape**
(aspect ratio `d_model/n_layer`, `d_ff/d_model`, `d_attn`). This is the reviewer's core objection in
one citation: **if performance is nearly shape-invariant at fixed N, then any parameter-saving trick
must be compared against simply spending those parameters on a smaller-but-standard model.**

**Scaling Laws vs Model Architectures**, Tay et al. https://arxiv.org/abs/2207.10551 — architectural
changes that look good at small scale frequently **do not hold up** at larger scale. Directly
relevant, because a sub-1B proxy experiment is exactly the regime this paper warns about.

**The Efficiency Misnomer**, https://arxiv.org/abs/2110.12894 — params, FLOPs, and latency are
non-interchangeable proxies and conclusions **flip** depending on which is reported. Combined with the
§5.3 Amdahl ceiling, this is the paper that forces reporting all three.

See also **Chinchilla**, https://arxiv.org/abs/2203.15556, and **The Depth-to-Width Interplay in
Self-Attention**, https://arxiv.org/abs/2006.12467.

[REASONING] **Why this is the strongest control and must be run:** the factorization frees ~73M params
(6.3% of the model at r=128). The narrower-model control asks whether those 73M are better spent as
width. Concretely: the parameter-matched comparison to a factorized-gate model at d=2048 is a
**dense-gate model at d ≈ 1980** (since total params scale ~d²), or equivalently the factorized model
should be *grown* to consume the freed budget. **If a factorized-gate model at d=2048 does not beat a
dense-gate model at the same total parameter count, the proposal has no claim.** Note the corollary:
the honest experiment is not "does low-rank hurt?" but "is low-rank the best use of these
parameters?" — a harder and more interesting question.

### 6.2 (b) Share one gate projection between B and C — Liquid's own incumbent method

Weight-sharing citations: **ALBERT** cross-layer sharing, https://arxiv.org/abs/1909.11942;
**Lessons on Parameter Sharing across Layers in Transformers**, https://arxiv.org/abs/2104.06022;
**Universal Transformers**, https://arxiv.org/abs/1807.03819; **Subformer**,
https://arxiv.org/abs/2101.00234; **Sharing Attention Weights for Fast Transformer**,
https://arxiv.org/abs/1906.01787.

**But the decisive citation is STAR itself (§1.8).** Liquid's search space devotes four of five
backbone-genome integers to sharing structure, its stated primary parameter-reduction mechanism is
*"identify which LIVs can be connected through featurizer or feature group sharing without degrading
performance"*, and evolutionary search **selected featurizer sharing between gated convolutions** as a
recurring quality-optimal motif. **This is the incumbent, vendor-validated method for exactly this
parameter budget on exactly this operator.** Promoted to mandatory.

Two distinct variants to run: **within-block** tying (B's projection = C's projection, saving d²
per block — the same saving as r = d/4) and **cross-depth** sharing (share featurizers between conv
blocks, which is what STAR actually found). ALBERT's all-shared ablation (§2.2) is the cautionary
data point that sharing has a real cost (81.7 → 80.1 avg at E=128).

### 6.3 (c) Use one gate instead of two

**GLU Variants Improve Transformer**, https://arxiv.org/abs/2002.05202 — establishes that gating
helps, with **no** scale analysis (§3B.7). **GLA's Table 4** (§1.3) is the most quantitative gate
ablation available: removing gating entirely costs **8.4 ppl**; data-independent decay costs 1.79;
scalar rather than vector gate costs 0.79. **Mamba Table 7** (§1.1): selective Δ alone 9.81 vs all
three 8.71 vs none 10.93.

Also: **H3**, https://arxiv.org/abs/2212.14052; **Hyena**, https://arxiv.org/abs/2302.10866;
**Gated State Spaces (GSS)**, https://arxiv.org/abs/2206.13947; **Mega**,
https://arxiv.org/abs/2209.10655; **Zoology**, https://arxiv.org/abs/2312.04927 (§1.9);
**Based / Simple linear attention balances the recall-throughput tradeoff**,
https://arxiv.org/abs/2402.18668.

[GAP] **No published ablation removes one of the two gates in a double-gated short-conv block
specifically.** LFM2's blog describes the double gate without ablating it. So "is the second gate
worth its d²?" is itself an open question — and a *cheaper* intervention than factorizing (saves d²
per block vs 2d² − 4dr). [REASONING] This makes (c) a strong control: **if deleting the C gate
entirely is non-inferior, the whole factorization exercise is moot for that gate**, and the finding
would be more surprising and more citable than a rank sweep.

### 6.4 (d) Grouped / block-diagonal gate projections — already validated at scale by Griffin

**Griffin** (§1.7) is the key citation: a production model from a major lab uses **block-diagonal
gates specifically to avoid a hardware cost**, at `D²/num_heads` parameters (~1/10 to 1/16 of dense).
**And "Grouped (block-structured)" is already in STAR's searchable channel-mixing options** while
low-rank is not (§1.8).

Supporting: **Deep Expander Networks**, https://arxiv.org/abs/1711.08757; **ShuffleNet**,
https://arxiv.org/abs/1707.01083; **Interleaved Group Convolutions**,
https://arxiv.org/abs/1707.02725; **DeFINE**, https://arxiv.org/abs/1906.06826; **Pay Less Attention
with Lightweight and Dynamic Convolutions**, https://arxiv.org/abs/1901.10430.

[REASONING] **Grouped is the most dangerous competitor to low-rank on the systems axis**, and this
should be stated plainly rather than discovered by a reviewer. A block-diagonal matmul is **one
kernel with good arithmetic intensity and no small-K dimension** — it avoids every hardware problem
§5 identified for `d→r→d` (two launches, intermediate write, K=r inefficiency), while achieving the
same parameter reduction. It is also trivially quantization-friendly. Given §5.3's ~8% Amdahl ceiling,
**a grouped gate plausibly captures the same parameter saving with a better latency profile**, which
would make it the correct engineering answer even if low-rank matched it on quality. Run it.

### 6.5 (e) Coupled / tied gates — the classic RNN literature already answered this

This is the most under-appreciated body of evidence, and it is directly on point: **the RNN community
ran exhaustive gate ablations 10 years ago.**

- **LSTM: A Search Space Odyssey**, Greff et al. https://arxiv.org/abs/1503.04069 — the canonical
  study. Headline findings: the **forget gate is essential** (removing it is among the most damaging
  changes), the **output activation function is essential**, and **coupling the input and forget gates
  (CIFG) is essentially free** — it reduces parameters with no significant loss. Removing the
  **output gate** is among the least damaging ablations.
- **An Empirical Exploration of Recurrent Network Architectures**, Jozefowicz et al., ICML 2015,
  https://proceedings.mlr.press/v37/jozefowicz15.html — architecture search over gate structures;
  agrees the forget gate is critical (and that initializing its bias to 1 is important).
- **Highway Networks**, https://arxiv.org/abs/1505.00387 — coupled gates `t` and `1-t`.
- **Simple Recurrent Units (SRU)**, https://arxiv.org/abs/1709.02755.
- **HGRN2** (§1.6) ties the input gate to `(1 - f_t)`, expanding state *"without introducing any
  additional parameters"* — coupling working in a modern architecture.

[REASONING] **The transferable prior is sharp and it argues for asymmetry**: across LSTM ablations the
*forget/input* gate (the LIV block's pre-conv `B`) is the load-bearing one, and the *output* gate (the
LIV block's post-conv `C`) is the cheapest to cheapen or remove. Combined with the independent signals
that RWKV-7 gives its gate a *larger* rank than its decay (§1.4), that every prior system keeps the
*output* gate dense while factorizing the *decay* gate (§1.7b), and my measurement that C_gate has
slightly lower erank and slightly smaller trained scale than B_gate (§4.4, §5.7) — **the literature
and the measurements disagree about which gate to cheapen, which is itself a reason to treat B and C
as separate axes rather than applying one rank to both.**

### 6.6 (f) Diagonal / elementwise / data-independent gates — the cheapest baseline

**RWKV-4** per-channel data-independent decay (§1.4); **RetNet's** zero-parameter fixed decay (§1.5);
**Griffin's** diagonal recurrent weight `a = σ(Λ)` (§1.7). Also **Diagonal State Spaces**,
https://arxiv.org/abs/2203.14343; **S4D**, https://arxiv.org/abs/2206.11893; **S5**,
https://arxiv.org/abs/2208.04933.

**The most relevant ablations are already in hand:** GLA Table 4 — data-independent scalar decay
(RetNet-style) costs **1.79 ppl** vs the low-rank data-dependent gate, and a data-dependent *scalar*
gate costs **0.79 ppl** vs vector. Mamba Table 9 — non-selective Δ costs 0.41 ppl vs rank-64.
RWKV-7 Table 19 — scalar decay costs ~0.068 val loss vs vector.

[REASONING] **This is the informative lower bound on the whole exercise.** A diagonal gate costs `d`
parameters instead of `d²` — a 2048x reduction, versus low-rank's 8x at r=128. If the quality gap
between a diagonal gate and a dense gate is small in the LIV block, then **the interesting design
point is far cheaper than r=128 and the low-rank framing is the wrong abstraction.** Conversely if
diagonal is clearly worse, that establishes gate expressivity *does* matter and makes the low-rank
result meaningful. **Either outcome is informative, which makes this the highest-information-per-FLOP
control.** It is also the cheapest arm to build.

### 6.7 (g) Structured-matrix substitutes — and the crucial finding about low-rank's ranking

**Monarch: Expressive Structured Matrices for Efficient and Accurate Training**,
https://arxiv.org/abs/2204.00595 — two block-diagonal factors with permutations; notable as a
structured-layer success **from scratch** with **real measured wall-clock speedups**. See also
**Monarch Mixer (M2)**, https://arxiv.org/abs/2310.12109; **butterfly factorizations**,
https://arxiv.org/abs/1903.05895; **Kaleidoscope**, https://arxiv.org/abs/2012.14966;
**Pixelated Butterfly**, https://arxiv.org/abs/2112.00029.

**The two most important papers here systematically compare low-rank against alternatives at matched
compute, from scratch:**

**Compute Better Spent: Replacing Dense Layers with Structured Matrices**, Qiu, Potapczynski, Finzi,
Goldblum, Wilson (ICML 2024). https://arxiv.org/abs/2406.06248 — Table 1 compares Dense, **Low-Rank**,
Convolution, Kronecker, Monarch, TT, BTT. Low-rank is characterized as `2rd` FLOPs and `2rd` params,
modelling assumption *"Compression"*, applications *"Bottleneck layers, Linear attention."*

Three findings that matter:
- **"BTT has the largest scaling exponent and consistently outperforms all other structures"** — so
  low-rank is beaten by BTT and Monarch.
- Their criticism is aimed at **parameter-sharing** structures, not low-rank:
  *"commonly used structures such as the Kronecker product and Tensor-Train decomposition violate
  this principle and underperform dense matrices in our experiments"*; *"Structures that do not share
  parameters are more flexible per unit of compute, and consistently achieve better scaling laws."*
  **Low-rank satisfies their principle** (params = FLOPs = 2rd), and is grouped *with* dense and BTT
  on the favourable side: *"β=1 for dense, low-rank, and BTT, but β=3/2 for Kronecker and TT."*
- **The telling methodological caveat: they "keep the input layer dense for low rank to avoid an
  information bottleneck at the first layer."** [REASONING] Independent confirmation of the §2.8
  scope argument — low-rank works when applied *selectively*, sparing the layers whose error
  propagates. That is precisely the LIV proposal's shape.
- Their **muP multipliers for Low-Rank UV: `κ_U = d/2r`, `κ_V = 1/2`** (§3.4), noting the asymmetry
  *"matches the concurrent LoRA+ finding."*
- BTT transformers **required weight normalization**, without which activations *"grow without
  bound"* and *"lead to NaN"* — precedent for structured layers needing a norm guard.

**Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices**, Potapczynski,
Qiu, Finzi et al. (NeurIPS 2024). https://arxiv.org/abs/2410.02117 — an Einsum-based taxonomy
covering *"low-rank, Kronecker, Tensor-Train, Block Tensor-Train (BTT), and Monarch."* Their headline:
scaling-law differences are governed by two quantities — **ω (parameter sharing, want small) and
ψ (rank, want large)** — and *"full-rank structures that maximize parameters per unit of compute
perform the best."*

[REASONING — the sharpest counter-argument to the whole proposal] **This is the paper a hostile
reviewer will cite.** Its conclusion is that **high ψ (full-rankness) predicts good scaling**, and
low-rank is by construction the *minimum-ψ* structure. Taken at face value it says: at matched
compute, prefer Monarch or BTT over low-rank. The honest responses, which should be in the writeup:
(i) both papers study replacing **general dense layers carrying the main signal path**, not gates
modulating a preserved full-rank path; (ii) their own practice of sparing the input layer concedes
that selectivity matters; (iii) §1's production systems demonstrate low-rank *gates* work at scale
even if low-rank *general layers* scale worse. **But this also means Monarch/BTT gates are a
legitimate and well-motivated competing arm** — and the structured-matrix literature would predict
they *beat* low-rank gates at matched parameters. If the budget allows one structured competitor,
Monarch is the one to run (it has the best latency story of the three).

### 6.8 [DELIVERABLE] Ranking of controls by strength

Ranked by how much each threatens the claim **"low-rank gates specifically are a good trade."**

| # | Control | Strength | Verdict |
|---|---|---|---|
| **1** | **(a) Narrower model at matched params** | **Decisive.** If low-rank loses here it has no claim at all — this is the null hypothesis the whole proposal must beat. Backed by Kaplan's shape-invariance and The Efficiency Misnomer. | **MANDATORY** |
| **2** | **(b) Featurizer / gate sharing** | **Near-decisive, and specific to this vendor.** STAR's own stated primary parameter-reduction mechanism, evolutionarily selected for gated convs. Not beating it means not beating the incumbent. | **MANDATORY** |
| **3** | **(f) Diagonal / data-independent gate** | **Highest information per FLOP.** 2048x cheaper than dense. If near-free, the low-rank framing is the wrong abstraction; if costly, it validates that gate rank matters. Cheapest arm to build. | **MANDATORY** |
| **4** | **Dense + gate bias** (from §5.9) | **Mandatory as a CONFOUND control**, not a competitor. Without it, a bias effect is misattributed to rank. | **MANDATORY** |
| **5** | **(d) Grouped / block-diagonal gate** | **Strongest systems competitor.** Same parameter saving, one kernel, no small-K penalty, quantization-friendly; already in STAR's option pool and shipped in Griffin. Likely to win on latency. | **STRONGLY RECOMMENDED** |
| **6** | **(c) Single gate (delete C)** | **Cheaper intervention than factorizing, never ablated in this block type.** If non-inferior, moots the C-gate factorization entirely. | **STRONGLY RECOMMENDED** |
| **7** | **(g) Monarch gate** | The structured-matrix literature predicts it **beats** low-rank at matched compute (high ψ). Best-in-class latency evidence. Expensive to implement. | Recommended if budget allows |
| **8** | **(e) Tied gate / coupled gates** | Well-supported by classic RNN work (CIFG "essentially free") and cheap, but a weaker *competitor* than (b)-(d) since it addresses a different axis. Most useful as motivation for treating B and C asymmetrically. | Nice-to-have |

[REASONING] **The four mandatory controls plus the proposal is a 5-6 arm experiment** — modest, and
each arm is cheap relative to the main run. Note that controls 1-4 are all *simpler* than the
proposal, which is the right property for a control set: **each one is a chance for the experiment to
fail cheaply and informatively.**


---

## 7. Experiment design implications

Concrete recommendations. Items marked **[LIT]** follow published practice; **[MEAS]** follow my own
measurements in §4.4-4.5, §5.2, §5.7-5.10; **[REASONING]** are inferences.

### 7.1 Reframe the claim before designing anything

**[MEAS]** The motivation "trained LIV gates are effectively low-rank" is **falsified** on the released
LFM2-350M checkpoint: gate erank is 771-790 of 1024 (~77% of full rank), only 46-50% of energy sits in
the top 128 directions, and the gates are **indistinguishable from the value stream** (790.1 vs 790.5).
Activation weighting (proxy) moves it only to ~748. Do not build the writeup on that claim.

**[REASONING]** Reframe as a **tolerance + budget-reallocation** question:

> Does a gated short-conv block still train to parity when its gate maps are rank-constrained from
> scratch — and is the freed parameter budget better spent on rank, or on width?

This is defensible, is what §1's production systems actually demonstrate (GLA r=16, Mamba r=d/32,
RWKV-6 r=64 all trained from scratch with no reported quality cost), and survives the §2.4
from-scratch failures because those factorized *everything* while this touches **7.17% of parameters**
with the value and output paths left dense.

### 7.2 Which ranks to sweep

**[MEAS + REASONING] Sweep r ∈ {128, 256, 512}, plus dense (r = ∞). Do NOT sweep below 64.**
**Revised emphasis after §5B: treat {256, 512} as the systems-viable range and 128 as the
aggressive-quality probe.** FLAR-SVD found a 128-wide projection *"not even achieving [the latency]
inflection point at 10%"*, and r=256 uniquely satisfies ggml's `QK_K=256`, ARM KleidiAI's 32-multiple
K padding, and standard cuBLAS tiling at once. Reasons for the range, each independent:

- **Parameter saving saturates** (§2.8): r=128 saves 6.27% of the model, r=32 saves 6.94%. The extra
  0.67% cannot justify any quality risk.
- **Channel-coupling cost rises as `sqrt(2/(πr))`** (§5.10), which is steep below r=64: coupling 0.073
  at r=128 but 0.102 at r=64 and 0.20 at r=16.
- **The Amdahl ceiling is 8.8%** (§5.3), so r=128 (7.6%) already captures 86% of the maximum possible
  latency benefit. Lower r buys almost nothing.
- **r ≥ d/2 measurably LOSES wall-clock** (§5.2: 0.42-0.67x at r=1024), so r=512 is the useful upper
  probe and r=1024 is a known-negative not worth the compute.
- **[LIT]** Keep every r a **multiple of 64** for tensor-core/tile efficiency; RWKV-7 independently
  rounds all ranks to multiples of 32.

**[REASONING] Treat B and C as separate axes, at least once.** The evidence conflicts about which gate
to cheapen: the RNN literature says the output gate is cheapest (§6.5) and my measurements agree
weakly (C has lower erank and smaller trained scale, §4.4/§5.7), but every prior system keeps the
*output* gate dense while factorizing the *decay* gate (§1.7b) and RWKV-7 gives its gate **2-4x more
rank** than its decay (§1.4). Run at least `(B dense, C rank-128)` and `(B rank-128, C dense)` at one
rank. This is cheap and resolves a genuine open disagreement.

**[REASONING] Parameterize rank as a ratio (r = d/16) if you intend to scale width later**, because
fixed r breaks standard μP LR transfer (§3.3) — but note this conflicts with §5.10, where fixed r holds
the functional cost constant. Pick deliberately and say which.

### 7.3 Init and LR treatment — the non-negotiable part

**[MEAS]** Getting this wrong produces a **12x-too-small block output** (naive 0.02 both factors) or a
~10⁴x error at d=2048 — not a subtle degradation, a diverged run. Recipe, in priority order:

1. **[LIT]** **Spectral init** (Khodak et al. https://arxiv.org/abs/2105.01029): SVD the *Xavier*
   `d×d` init the block would otherwise use, take top-r, split `sqrt(Σ_r)` into each factor. It is
   scale-balanced by construction, which immunizes against the Adam split pathology (§3B.5), and makes
   the `r → d` limit exactly recover the dense control.
2. **[LIT]** **Pair it with Frobenius decay** — weight decay on `‖BA‖_F`, never on the factors
   separately (per-factor L2 secretly penalizes the *nuclear* norm, adding a third rank penalty).
   **Critical: Khodak's ablation shows SI alone (92.52) and FD alone (92.92) are each WORSE than plain
   low-rank (93.59); only together (94.34) do they recover the dense baseline.** Do not test one
   without the other, and do not conclude from a single-component test.
3. **[MEAS] Add a full-width gate bias initialized to 1.0** (2d params). This is the highest-value
   single recommendation in the dossier: it makes the block compute `y = x̃` at init (identity gate,
   full signal and gradient to conv and out_proj), it is the from-scratch analogue of LoRA's
   unavailable `B=0` trick (§2.1), it supplies the per-channel offsets a rank-r map cannot express
   (§3.5 Fact 4), it matches universal prior practice (Mamba `dt_bias`, GLA `b_α`, RWKV `λ_□`), and
   **it cuts block-output kurtosis from 27.8 to 4.5** (§5.8) — the only intervention found that reduces
   the heavy tails intrinsic to a multiplicative path.
4. **[LIT]** **LR ratio `η_B/η_A = d/r`** (Compute Better Spent Table 2: `κ_U = d/2r`, `κ_V = 1/2`);
   at d=2048, r=128 this is 16, coinciding with LoRA+'s empirical `λ=2⁴`. Using `d/r` rather than a
   constant 16 keeps the treatment correct across the whole sweep.
5. **[LIT]** **Re-tune the base LR per rank, or apply the `r^{-1/2}` correction** (μA,
   https://arxiv.org/abs/2602.06204: optimal LR shifts ~one log₂ unit per 4x rank). Published LR
   corrections for factorized layers **disagree in direction** (ReLoRA 1.5-2x larger, LTE 0.05-0.1x,
   Kamalakara 0.5x), so a factorized layer does not inherit a tuned dense LR. **A rank sweep at fixed
   LR measures LR-mismatch, not rank.**
6. **[LIT]** **Exempt the gate bias from weight decay and from any global re-init sweep** (Mamba's
   `_no_reinit`, Gated DeltaNet's `_no_weight_decay`). This is a real, documented implementation bug
   class.
7. **[LIT]** Use **decoupled** weight decay and consider **AdamW ε = 1e-15** (Wortsman et al.
   https://arxiv.org/abs/2309.14322: default 1e-8 is too large at scale; gradient RMS collapsing to ε
   makes a layer stop learning). **B's small fan_in makes it the prime collapse candidate.**
8. **[REASONING]** Do **not** use LoRA's `B=0`: with no additive dense path the gate would be
   identically zero, the block output zero, and `∂L/∂A = 0`.

### 7.4 Mandatory controls (from §6.8)

Five arms minimum, four of which are *simpler* than the proposal:

1. **Stock LIV** (dense gates, no bias) — reproduces the published architecture.
2. **Dense + gate bias** — **the confound control.** Without it, the bias effect (§5.9: kurtosis
   27.8 → 13.5) is misattributed to rank. May itself beat stock LIV, which would be a publishable
   finding on its own.
3. **Narrower dense model at matched total params** — the null hypothesis the proposal must beat.
4. **Gate/featurizer sharing** — the vendor's own incumbent method, evolutionarily selected by STAR
   for gated convolutions.
5. **Diagonal (elementwise) gate** — 2048x cheaper; the highest-information-per-FLOP arm.

Strongly recommended if budget allows: **grouped/block-diagonal gate** (the strongest systems
competitor — same saving, one kernel, no small-K penalty, quantization-friendly, already in STAR's
option pool and shipped in Griffin) and **delete the C gate** (cheaper than factorizing, never
ablated in this block type).

### 7.5 Diagnostics to log

Cheap, and each targets a specific documented failure mode:

| Diagnostic | Why | Threshold / reference |
|---|---|---|
| **erank and E(r) of each gate map** (`BA` product, and `W Σ_x^{1/2}` with real activations) | Detects effective-rank collapse below nominal r | LoRA realizes **< half** its nominal rank (§2.6) |
| **Channel coupling of the gate** | Detects collapse to lower effective rank | Should track `sqrt(2/(πr))` ≈ 0.073 at r=128 (§5.10) |
| **Gate output std ACROSS TOKENS, per layer** | Detects the gate degenerating to a learned constant — the predicted low-rank failure mode | Compare to dense control (§3B.2) |
| **Block output kurtosis** | Multiplicative paths amplify 4th moments; variance checks pass while tails blow up | Dense ≈ 27.8; bias variant ≈ 4.5 (§5.8) |
| **Elementwise correlation between gate slice and value slice, per layer** | The SwiGLU alignment pathology; rising correlation is the early warning | Moved **between 125B and 210B tokens** in the FP8 study (§3B.7) |
| **‖BA‖_F and ‖BA‖_2 separately** | Ratio is the stable rank; also tracks the §3B.1 spectral/RMS tension | Spectral norm inflation `sqrt(d/r)` = 4x at r=128 |
| **Per-factor gradient RMS vs Adam ε** | B's small fan_in is the collapse candidate | ε = 1e-15 recommended (§3B.8) |
| **‖A‖_F² − ‖B‖_F² drift** | Conserved under gradient flow, **not** under Adam — drift signals the split pathology | §3B.5 |
| **End-to-end latency on the real backend** (ExecuTorch 8da4w, llama.cpp Q4_0), not per-layer | Per-layer microbenchmarks overstate by ~2 orders of magnitude | Amdahl ceiling **8.8%**; 7.6% at r=128 (§5.3) |
| **Quantized (int4) quality delta, factorized vs dense** | Undocumented risk: the bottleneck concentrates information into r channels | [GAP] no literature (§5.5) |

**[LIT] A recall/MQAR-style probe is mandatory, not optional.** Zoology showed gated convolutions
need `d ≥ N` for multi-query associative recall and that **82% of the gated-conv vs attention gap is
in-context recall** which perplexity hides (§1.9). A rank sweep measured only in ppl could report
"non-inferior" while degrading recall.

### 7.6 Pre-registered failure criterion

**[REASONING]** Set these before running. The noise floor matters: **Mamba's own rank sweep is
non-monotone** (rank 8 = 8.83 beats rank 16 = 8.84, §1.1), so any criterion must exceed
seed-to-seed variance — run **≥2 seeds per arm** and state the observed spread.

**Primary (quality).** Declare the factorization a **failure** if, at matched token budget and with
per-rank-tuned LR, the r=128 arm is worse than **BOTH**:
- the **stock-LIV control** by more than **1.5x the seed-to-seed std** on held-out loss, **AND**
- the **narrower-dense matched-parameter control** by any margin.

The second clause is the one that matters: **losing to the narrower model is disqualifying regardless
of how close to stock LIV the arm gets**, because it means the parameters were better spent as width.

**Secondary (recall).** Declare failure if the MQAR/recall probe degrades by more than the
seed spread, *even if perplexity is non-inferior*. Perplexity parity with recall loss is the specific
outcome Zoology warns about.

**Systems — now a PRE-TRAINING GATE, per §5B.4.** Before any training compute is spent, microbenchmark
`d→r→d` vs `d→d` at r ∈ {512, 256, 128, 64} on the **real Snapdragon + ExecuTorch/llama.cpp stack in
the real quantization format.** If the factorized layer is not faster *in isolation* at the candidate
rank, **do not make a wall-clock claim at all** — proceed only as a parameter-efficiency /
gating-science experiment. Post-training, declare the *deployment* case unproven if measured end-to-end
decode latency improves by **less than 4%** (about half the 8.8% ceiling), or if int4 quantization costs
more quality in the factorized arm than the dense arm.

**[REASONING]** This criterion can fail while the quality criterion passes, and given that a ~5-8%
predicted gain sits *inside* the 0-12% slowdown band FW-SVD/ASVD actually measured on this silicon
(§5B.1), **"quality-neutral, latency-negative" is arguably the single most likely outcome.** Plan the
writeup so that result is still publishable: it would be a clean, well-measured negative on a
plausible idea, plus the gating-science and diagnostic contributions.

**Diagnostic-based early abort.** Abort a run if gate output std across tokens falls below ~25% of the
dense control's (gate has degenerated to a constant), or if trained channel coupling exceeds
**2x** `sqrt(2/(πr))` (factors collapsed below nominal rank). Both are detectable within a small
fraction of a full run.

**Scope limitation to state honestly.** The SwiGLU alignment pathology surfaced only **between 125B
and 210B tokens** and was invisible below 100B (§3B.7). Any sub-100B-token experiment **cannot exclude
it.** Say so rather than implying the multiplicative-path risk has been ruled out.

### 7.7 Cheapest high-value precursor before spending training compute

**[REASONING]** Two experiments costing a few GPU-hours on public checkpoints, either of which could
redirect the project:

1. **Post-hoc truncate the trained LFM2 gate slices to rank r and measure perplexity** — plain SVD and
   activation-aware (SVD-LLM-style whitening). This bounds the plausibility of the whole idea and, per
   §4.3, plain SVD at even 5% removal can be catastrophic while activation-aware at 20% costs ~2 ppl.
   The gap between those two numbers *is* the diagnostic.
2. **Re-run §4.4/§4.5 with real per-layer activation statistics** rather than the embedding proxy. This
   is the load-bearing measurement for the motivation and it is currently unresolved.

**[REASONING] And a strategic note worth stating plainly:** §5.3 shows the SwiGLU MLPs are **68.8% of
parameters** while the conv gates are 7.17%. If the actual goal is on-device latency or parameter
efficiency, **the MLPs are where the budget is**, and the same low-rank/structured-matrix machinery
developed here applies to them with ~9x the leverage. That is a legitimate finding to report from this
research even though it is negative for the original framing — and it is arguably the most actionable
conclusion in this document.


---

## 8. Summary of literature gaps (positive findings for novelty)

Auditable negatives, each searched deliberately:

1. **[GAP] No published low-rank gate in a gated short/depthwise causal convolution block.** All
   low-rank gates in the literature live in linear-attention/SSM *recurrences* (GLA, Mamba-1, RWKV-6/7).
   Every short-conv implementation checked (LFM2 `Lfm2ShortConv`, Hyena, H3, fla `ShortConvolution`)
   uses **full-rank dense** projections. Systematic arXiv full-text searches returned 0 relevant hits
   for `"low-rank gate" AND "language model"`, `"short convolution" AND "low-rank"`,
   `"factorized gate"`, and `"low-rank" AND "forget gate" AND "sequence model"` (validated against a
   positive control).
2. **[GAP] Liquid's own STAR search space cannot express a factorized gate.** Its channel-mixing
   options are exactly `{Diagonal, Dense, Grouped}`; "low rank" appears only as a *token*-mixing
   structure. The featurizer genome has expansion and repeat factors but **no rank field**. Verified
   from the LaTeX source. The proposal is outside the space Liquid searched.
3. **[GAP] Nobody has ablated gate RANK against quality except Mamba (Table 9).** GLA asserts r=16 and
   τ=16 without tuning; RWKV's ranks are *"mere speculation"* by the authors' own words; Mamba-2
   removed the bottleneck without isolating it. Full-rank Δ was never run by anyone.
4. **[GAP] No rank-sensitivity or effective-rank study broken down by matrix type for gates**, and no
   published singular-value spectra for Mamba/SSM/gated-conv weights. §4.4 appears to be the first
   such measurement for this architecture.
5. **[GAP] No study of a low-rank factor feeding a MULTIPLICATIVE path** as opposed to an additive
   residual. §3.5/§3B.1's derivations and §5.8-5.10's measurements are original.
6. **[GAP] No normalization inside a low-rank bottleneck.** Houlsby's LayerNorm is *after* the
   adapter; DoRA normalizes composite columns; σReparam normalizes whole matrices. A defensible novel
   contribution, with a clear rationale (scale-split invariance) and a clear cost (loses foldability).
7. **[GAP] μP class assignment for low-rank factors is stated nowhere**; μA (2026) resolves the rank
   exponent but uses a single shared LR and does not use μP's input/output taxonomy. The §3.3
   derivation is mine, corroborated by independently reproducing LoRA+'s Θ(d) ratio.
8. **[GAP] Quantization × from-scratch low-rank interaction is undocumented.** The QLoRA line
   quantizes the frozen base and keeps adapters in *higher* precision — the opposite configuration
   from an on-device factorized gate.
9. **[GAP] The bf16 blast radius of the GLU alignment pathology is unquantified.** The FP8 paper's
   Theorem 1 is precision-independent, but only FP8 divergence is demonstrated.

