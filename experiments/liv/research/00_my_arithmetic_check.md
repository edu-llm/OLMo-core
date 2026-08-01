# Independent arithmetic check of the brainlift's claims

Date: 2026-07-30. Computed directly (`crossover.py`, `proposals.py` in this directory).
Geometry: LFM2-1.2B-like — d=2048, 16 layers (10 LIV + 6 GQA), 32 q heads, 8 kv heads,
head_dim 64, SwiGLU ff=8192, vocab 65,536 tied, bf16. This is the geometry the brainlift's
own numbers imply.

## Every stated number is correct

| Brainlift claim | Verified |
|---|---|
| stock LIV mixer ≈ 4d² + kd = **16.783M** at d=2048, k=3 | **16,783,360** ✓ exact |
| comparable GQA mixer ≈ 2.5d² = **10.486M** | **10,485,760** ✓ exact (2.50 d²) |
| factorized gates ≈ 2d² + 4dr + kd = **9.443M** at r=128 | **9,443,328** ✓ exact |
| "7.340M fewer than stock LIV" | 16.783 − 9.443 = **7.340M** ✓ |
| "1.043M fewer than the comparable GQA mixer" | 10.486 − 9.443 = **1.043M** ✓ |
| "replacing attention with the released LIV mixer *increases* mixer params" | ✓ 16.78M > 10.49M |
| 6 GQA layers × 8 kv heads × 64 × 2(K,V) × 2B = **12 KiB/token** | ✓ 12,288 B |
| ≈ **384 MiB** raw K/V at 32K | ✓ 402.7 MB = 384 MiB |
| 2.5 d² for GQA depends on `num_key_value_heads` | ✓ — with hq=32, hkv=8, hd=64: q+o = 2d², k+v = 0.5d² |

The GQA figure is only 2.5d² *because* hkv=8 while hq=32. The brainlift's caveat about this
being config-dependent is correct and worth keeping.

## Verified against released code and config (fetched directly, 2026-07-30)

**`Lfm2ShortConv` forward, from `transformers` v5.0.0rc1 `modeling_lfm2.py`** (Apache-2.0):

```python
self.in_proj  = nn.Linear(hidden_size, 3 * hidden_size, bias=conv_bias)
self.out_proj = nn.Linear(hidden_size, hidden_size, bias=conv_bias)
self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=L_cache,
                      groups=hidden_size, bias=conv_bias, padding=L_cache - 1)
...
BCx = self.in_proj(x).transpose(-1, -2)
B, C, x = BCx.chunk(3, dim=-2)     # chunk order is B (pre-gate), C (post-gate), x (value)
Bx = B * x
conv_out = self.conv(Bx)[..., :seqlen]
y = C * conv_out
y = self.out_proj(y.transpose(-1, -2).contiguous())
```

This confirms the brainlift's operator description exactly, and adds three facts the brainlift
does not state:

1. **There is NO activation inside the block.** The fused path passes `activation=None`
   explicitly to `causal_conv1d_fn`. Nonlinearity lives only in the MLP (`F.silu` in `Lfm2MLP`).
   → This is decisive for the local code audit: OLMo-core's `CausalConv1d` defaults to
   `activation="silu"` *inside* the fused kernel, so **reusing it unchanged would implement a
   different operator.** The existing protocol's warning is now verified against both sides.
2. **No normalization inside the block.** `operator_norm` (RMSNorm) is owned by
   `Lfm2DecoderLayer` and applied before the call — matching the brainlift's `u = RMSNorm(h)`.
3. **Decode cache is `[batch, hidden_size, L_cache]`**, updated by `roll(shifts=-1, dims=-1)` then
   writing at the clamped position; the cached value is `Bx` (the *gated pre-conv* stream), which
   matches the brainlift's "previous k−1 vectors of y" and the protocol's "LIV history of the
   gated pre-convolution stream".

**`LFM2.5-1.2B-Base/config.json`**: `hidden_size` 2048, `num_hidden_layers` 16,
`num_attention_heads` 32, `num_key_value_heads` 8, `conv_L_cache` 3, `conv_bias` false,
`vocab_size` 65536, `tie_embedding: true` (note: *not* `tie_word_embeddings`), `norm_eps` 1e-5,
`rope_theta` 1e6, `max_position_embeddings` **128000**, `block_use_swiglu` true.
`layer_types` gives attention at **[2, 5, 8, 10, 12, 14]** — confirming 10 conv / 6 attention and
matching the local protocol's frozen schedule exactly.

**The `block_ff_dim` trap, now derived.** Config says `intermediate_size: 12288`, but
`block_auto_adjust_ff_dim: true` with `block_multiple_of: 256`, `block_ffn_dim_multiplier: 1.0`.
The transformation is the Llama-style `ff = multiple_of * ceil(int(2/3 * block_ff_dim) / multiple_of)`:

- 1.2B: `int(2/3 × 12288)` = 8192 → already a multiple of 256 → **effective 8192** ✓ (the value I
  used above, so the 1.170B total stands)
- 350M: `int(2/3 × 6656)` = 4437 → round up to 4608 → **effective 4608** ✓ reproduces the local
  protocol's stated 4,608

So the protocol's warning at `liv-kda-gqa-sub500m-experiment.md:97` is correct and the formula is
now pinned down. **Any reimplementation must apply this transform or every parameter count will be
wrong by ~50% on the MLP** — which is 69% of the model.

**`causal-conv1d` (Dao-AILab) hard constraints** — this materially damages proposal 3:

- README feature list, quoted: `"Kernel size 2, 3, 4."` — an enumerated set, not a range.
- `causal_conv1d_fn(x, weight, bias=None, activation=None)` — **there is no `dilation` argument
  anywhere**, and the documented equivalence is `F.conv1d(..., padding=width-1, groups=dim)` with
  dilation left at its default of 1.

→ **The multiscale proposal has no fast kernel on either variant.** The dense widths 5, 9, 15 are
outside the supported set {2,3,4}, and dilation is unsupported entirely. Both the 32-tap dense and
the 12-tap dilated version would fall back to `F.conv1d` with `groups=d` (memory-bound, low
occupancy) or need a custom Triton kernel. Combined with the fact that P3 has no state or
parameter benefit (below), **P3 is the weakest of the three proposals on every efficiency axis and
must be justified on quality alone.**

## P2 novelty: Hymba IS prior art (found later — supersedes the Zamba-only reading below)

The Zamba analysis below is correct but incomplete. The real precedent is **Hymba**
(arXiv 2411.13676, NVIDIA, Nov 2024), whose `config.json` contains
`kv_reuse_group = [[1,2],[3,4],...,[16,17,18],...]` with `kv_weight_reuse: false`, and whose released
`modeling_hymba.py` implements CLA-in-a-hybrid: consumers build no `k_proj`/`v_proj`, project only Q,
skip cache writes, and consume **post-rotary** K. It cites CLA explicitly.

So "apply CLA to a hybrid" is **not novel**. What remains open: Hymba is a *parallel hybrid-head*
model (attention and SSM heads inside the same layer) sharing between *adjacent sliding-window*
layers, and it **excludes its 3 global-attention layers from every sharing group**. Our proposal
shares between *full-attention* layers separated by a complete conv block — structurally different,
and the opposite of what Hymba validated.

And its ablation (row C→D, 300M/100B) is a warning: commonsense **+0.60**, throughput **+14.9%**,
**recall −0.75**. Aggregate up, retrieval down, not investigated — exactly the brainlift's worry.

Also: Hymba's cache fell only 41.2→39.4 MB (4.4%) from sharing because SWA had already cut it 3.8×.
**SWA and CLA are largely substitutes on the capacity axis.** Our full-attention layers leave more to
remove, which is the one way our setting is more favorable.

## P2 novelty check: Zamba is NOT prior art (verified)

Zamba (arXiv 2405.16712) is the nearest-looking prior work and needed ruling out. From the paper's
own HTML:

- 80 layers, Mamba backbone, one "global shared attention" (GSA) block **applied 13 times**, every
  6 Mamba blocks, with **tied weights**.
- The GSA input is `LN([x_l, x_0])` — the layer-`l` residual **concatenated with the post-embedding
  activations**. So the input differs at every call.
- Therefore, in the paper's own words, the block is applied 13 times "**with independent
  activations and KV-cache entries at each invocation**."

**So Zamba shares parameters and recomputes K/V. Cross-layer attention (CLA) shares the K/V
tensors themselves and does not recompute. These are orthogonal mechanisms**, and the brainlift's
proposal is the CLA kind. Zamba does not anticipate it.

Worth noting for the write-up: Zamba's §II claims weight sharing reduces "the KV cache size during
generation," which **does not follow** from its own §III admission of independent per-invocation KV
entries. Parameter tying gives zero KV-cache reduction under recomputation. Zamba's real cache
saving comes from attention sparsity (13 attention calls in 80 layers) — i.e. from the same
mechanism the mostly-LIV topology uses, not from sharing. This is a useful contrast to draw
explicitly, and a caution: it is easy to conflate parameter sharing with cache sharing, and a
published 7B paper did exactly that.

### Serving-stack support now exists (changed since 2024)

Relevant to whether P2's memory claim can be *measured* rather than just computed: cross-layer KV
sharing is now first-class in vLLM (`kv_sharing_target_layer_name`, PR #18212, merged 2025-06-03),
in llama.cpp (a genuine `il -> cache index` indirection map, `map_layer_ids`, plus a `reuse`
callback that would accept a CLA pattern as a one-line lambda), and in HF transformers for the
Gemma family only (via a `shared_kv_states` side-channel, *not* the generic `Cache` API — a generic
mechanism is proposed in open PR #47290 but explicitly notes the current design "would not allow
for complex KV cache sharing patterns like CLA"). Two cautions:

- **vLLM's validator only permits sharing with an *earlier layer of the same type*** — workable for
  our case, since the producer is earlier — but its Hunyuan CLA support needed a *completely
  separate* mechanism (`HunYuanCrossAttention`, K/V threaded through the call stack), which is
  evidence that the general primitive does not cleanly express mid-model CLA.
- **SGLang appears to allocate the full layer count anyway** (`num_attention_layers =
  num_hidden_layers`, no subtraction for shared layers), so it may be functionally correct while
  saving no memory. Flagged as inference from the allocation chain, not empirically confirmed.
- **A sliding-window producer truncates its cache while a consumer needs full-length states** — HF
  keeps an un-truncated copy alive for the whole forward pass specifically for this reason, which
  *claws back part of the memory saving*. Not an issue for us (our attention layers are full
  attention, not sliding), but it is a trap if a sliding-window control arm is added.

### But CLA's own ablation predicts trouble for the brainlift's pairing

CLA (arXiv 2405.12981) validated at 1B (30B tok) and 3B (100B tok) on SlimPajama and reports CLA2
at 2× cache reduction for **0.04-0.05 ppl** degradation — sometimes an improvement. Strong support
for the proposal's premise. Three findings that constrain the design, though:

1. **Uniform ADJACENT pairing won; non-uniform variants all lost.** Their `DenseBack` variant —
   which forces a long run of layers onto one early layer's activations — was **+0.43 ppl worse**
   than uniform. Their recommendation, quoted: "CLA should be used between pairs of consecutive
   layers." **In the LFM2 schedule, attention sits at [2, 5, 8, 10, 12, 14] — no two attention
   layers are adjacent.** The closest pairs are (8,10), (10,12), (12,14), separated by one LIV
   block; (2,5) and (5,8) are separated by two. So the brainlift's plan necessarily violates CLA's
   own best-practice, and the staleness worry it raises is precisely what CLA measured as harmful.
   → Pair the *closest* attention layers — (8,10), (12,14) — not arbitrary ones, and treat
   inter-pair distance as the primary axis of the ablation.
2. **CLA pairs best with MQA, not GQA.** Their GQA+CLA2 results were mostly *worse* than
   equal-footprint baselines; only GQA2+CLA2 broke even. LFM2 uses GQA with hkv=8. So the proposal
   is starting from CLA's weaker partner, and "reduce hkv instead" (i.e. move toward MQA) is a
   mandatory competing control — it is CLA's own recommended alternative.
3. **RoPE pre/post-rotary is NOT addressed in the CLA paper at all.** I checked specifically: no
   statement, no ablation, no mention in the mechanism section. So this is a genuine open
   implementation question the experiment must decide and document, not look up. (Caching
   post-rotary is the natural single-copy choice since consumers at different depths share one
   tensor and the *position* is the same — but the rotation is position-dependent, not
   layer-dependent, so post-rotary sharing is sound. Pre-rotary would require each consumer to
   re-rotate, costing compute for no benefit.)

### Production precedent that RESCUES the non-adjacent pairing concern

Character.AI's inference post (delisted; recovered via Wayback,
`web.archive.org/web/20240624161133/https://research.character.ai/optimizing-inference/`,
Jun 2024) describes a production model that does exactly what the brainlift proposes, verbatim:

> **3. Cross Layer KV-sharing.** We tie the KV cache across neighboring attention layers... **For
> global attention layers, we tie the KV cache of multiple global layers across blocks**, since the
> global attention layers dominate the KV cache size under long context use cases. Similar to a
> recent publication (Brandon et al., 2024), **we find that sharing KV across layers does not
> regress quality.**

And the figure caption: *"For global attention layers, we share KV across multiple **non-adjacent**
layers."* Their production model also uses **"only 1 out of every 6 layers uses global attention"**
— i.e. essentially the same attention density as LFM2's 6-of-16 — with local attention at window
1024 elsewhere, and reports >20× total KV reduction with no needle-in-haystack regression.

**This substantially de-risks P2.** CLA's own paper found non-uniform pairing harmful, but
Character.AI reports non-adjacent *global-layer* sharing working in production at scale, in an
architecture with the same sparse-global structure as LFM2. The reconciliation: CLA's harmful
`DenseBack` variant forced *many* layers onto *one* early producer; sharing between two nearby
global layers is a different and much milder configuration.

So the honest framing is: non-adjacent sharing between global layers has production precedent but
**no public controlled ablation** — Character.AI is a blog post with no numbers on the sharing
component in isolation, and CLA never tested a sparse-global topology. That is exactly the gap this
experiment can fill.

Zamba's sharing ablations are qualitative only ("early ablation studies on small models", no
tables, no seeds, no isolation of shared-vs-independent weights). So there is also **room for a
genuine contribution in ablating sharing schemes rigorously at small scale** — which is what this
experiment is positioned to do.

## The framing problem: the mixer is 14% of the model

This is the finding the brainlift does not state, and it changes how the results must be reported.

```
embeddings (tied)      134.2M    11.5%
10 LIV mixers          167.8M    14.3%
 6 GQA mixers           62.9M     5.4%
16 MLPs                805.3M    68.8%   <-- dominates
TOTAL                1,170.3M
```

So the low-rank gate proposal, which cuts 7.34M per LIV layer:

| rank r | LIV mixer | whole model | model Δ |
|---:|---:|---:|---:|
| stock | 16.783M | 1.170B | — |
| 512 | 12.589M | 1.128B | −3.58% |
| 256 | 10.492M | 1.107B | −5.38% |
| **128** | **9.443M** | **1.097B** | **−6.27%** |
| 64 | 8.919M | 1.092B | −6.72% |

**A 44% cut to the LIV mixer is a 6.3% cut to the model.** Note also the sharp diminishing
return: going r=128 → r=64 buys only 0.45 more percentage points, because at that point the
full-width value and output projections (2d² = 8.39M) are the floor. The brainlift's choice to
keep those full width is what caps the achievable saving at ~7%.

Implication: "fewer parameters" must be reported as a *mixer-level* claim, or the experiment must
also shrink the MLP — but shrinking the MLP is the obvious competing control (just narrow the
model), which is exactly the baseline a reviewer will demand.

## Crossover numbers — the decisive result

Decode reads all weights every token (2.341 GB in bf16) plus the KV cache for the context.

```
KV read == weight read  at  T = 190,474 tokens   (6-GQA hybrid)
KV read == weight read  at  T =  71,428 tokens   (all-16-GQA control)
Attention-score FLOPs == all dense FLOPs at T = 84,314 tokens  (prefill)
```

### Correction: P2 saves capacity, NOT bandwidth — so its latency saving is ~zero, not 1-7%

My table below computes what halving the *resident* KV would save if it halved *read traffic*. It
does not. From the CLA paper directly (arXiv 2405.12981): **"CLA has no direct effect on the memory
bandwidth consumed by the attention mechanism in each decoding step"** — each consumer layer must
still re-read the shared bank from main memory. Reads are unchanged; only *writes* halve (which the
brainlift states correctly), and writes are the small half of KV traffic during decode.

So the honest accounting for P2 is:
- resident KV bytes: **−50%** (real, clean, measurable)
- KV write traffic: **−50%** (real, but writes ≪ reads in decode)
- KV read traffic: **0%**
- end-to-end decode latency: **≈0%** at any context length

The rows below should therefore be read as an **upper bound that P2 cannot actually reach**. Its
true latency effect is ~0. This strengthens the conclusion: P2 must be framed as a capacity result
(bigger batch, longer context per GB), never as a speed result.

KV as a share of decode traffic, and the upper bound on what halving resident KV could save:

| context T | KV MB | % of decode traffic | P2 saves | % of total |
|---:|---:|---:|---:|---:|
| 4,096 | 50.3 | 2.1% | 25.2 MB | **1.05%** |
| 8,192 | 100.7 | 4.1% | 50.3 MB | 2.06% |
| 16,384 | 201.3 | 7.9% | 100.7 MB | 3.96% |
| 32,768 | 402.7 | 14.7% | 201.3 MB | **7.34%** |
| 131,072 | 1,610.6 | 40.8% | 805.3 MB | 20.38% |
| 262,144 | 3,221.2 | 57.9% | 1,610.6 MB | 28.96% |

Prefill attention share: 4.6% at 4K, 8.9% at 8K, 16.3% at 16K, 28.0% at 32K, 43.7% at 64K.

**This is the single most important number in the whole design.** At the 4K-32K contexts a
sub-500M academic study can afford to train, KV cache is 2-15% of decode traffic. The
"mostly-LIV saves memory and latency" thesis is a *long-context* thesis: it only becomes a
first-order effect past ~100K tokens. Consequences:

1. P2 (cross-layer KV sharing) cannot show *any* end-to-end decode win, at any context length,
   because it does not reduce read bandwidth (see correction above). It should be reported as a
   **resident-memory / cache-capacity** result — 2× fewer banks is a real, clean, honest claim —
   and the paper should predict the latency null in advance rather than discover it.
2. Any latency claim at ≤32K must come from the **weight-read** path, which is where P1
   (low-rank gates) lives — it cuts 6.27% of weight bytes, and decode is weight-bandwidth-bound.
   So **P1 is the only one of the three proposals with a plausible decode-latency story at
   trainable context lengths**, and its ceiling is ~6%.
3. The honest headline for the mostly-LIV topology is not speed at 32K; it is that state grows
   6/16 as fast, so the *slope* is better. Report slope, not a single-point win.

## P3: parameters are trivial, but state grows 5×

**[CORRECTED 2026-07-30.]** An earlier version of this table said 7×, using the wrong cache
convention. LFM2's conv cache is `[B, d, L_cache]` with `L_cache = k`, so a 3-tap kernel stores 3
vectors and a max-lag-14 kernel stores 15 → the ratio is **15/3 = 5×**, not 7×. Table below at d=2048
(bf16); the d=1024 figures are exactly half.

| variant | taps/channel | max lag | conv params | state/layer | 10 LIV layers |
|---|---:|---:|---:|---:|---:|
| stock k=3 | 3 | 2 | 6,144 | 12.0 KiB | 120 KiB |
| dilated 1,2,4,7 | 12 | 14 | 24,576 | 60.0 KiB | 600 KiB |
| dense 3,5,9,15 | 32 | 14 | 65,536 | 60.0 KiB | 600 KiB |

Three points, the third added later:

- **State is set by max lag, not tap count.** Dilations {1,2,4,7} reach lag 14, so the ring buffer
  holds 15 vectors, not 3. The dilated "cheap" version (12 taps) has *exactly the same state* as the
  dense 15-tap version — dilation saves parameters and FLOPs but buys **nothing** on state. If state
  is the selling point, dilation is the wrong lever.
- **The state is negligible against KV anyway.** 120 KiB → 600 KiB is 0.03% → 0.15% of the 384 MiB KV
  cache at 32K, so "bounded tiny state" survives on a GPU. **But it cuts against LFM2's embedded-CPU
  design target**, where a small fixed working set is the entire point. The real cost of P3 is
  bandwidth from evaluating 4 branches, not state.
- **The dilation pattern is worse than "sparse".** {1,2,4,7} covers only **7 of the 15 lags** — 8 lags
  are structurally unlearnable and lag 0 is covered 4× redundantly. So 12 parameters buy **7 degrees
  of freedom** against 15 DOF for 15 parameters in a dense kernel. The dilated variant is strictly
  dominated on expressiveness-per-state.

Also worth flagging: a mixture of 4 dilated 3-tap causal kernels with **input-independent** weights
is *exactly* a single sparse 15-tap causal kernel. So the fixed-weight multiscale variant collapses
to "one 15-tap kernel with a sparsity pattern", and a dense 15-tap kernel (65,536 params, still
trivial) is a strict superset of it at 2.7× the params. The *only* thing token-dependent routing
adds is input-dependence of the tap pattern. That makes a dense-15-tap control mandatory, or the
multiscale claim is vacuous.

## Scale reversal: the 350M geometry is a BETTER testbed than 1.2B

The local protocol freezes a d=1024 / 350M geometry (16 layers, 16 q heads, 8 kv heads, hd=64,
SwiGLU 4,608, vocab 65,536 tied). Recomputing there:

```
LIV mixer   4,197,376   <-- matches the protocol ledger exactly
GQA mixer   3,145,728   (3.00 d^2, not 2.50 — because hq=16 not 32)
TOTAL     354,449,408   (protocol says 354,483,968; delta -34,560 = RMSNorm weights I omitted)
KV/token       12 KiB   <-- IDENTICAL to the 1.2B model (hkv=8, hd=64, 6 layers all unchanged)
weight read  708.9 MB/token
```

KV cache per token is **the same 12 KiB at both scales** (it depends only on hkv, hd, and the
number of attention layers — none of which change), but weight bytes fall 3.3×. So:

| | 350M (d=1024) | 1.2B (d=2048) |
|---|---:|---:|
| KV read == weight read at | **T = 57,690** | T = 190,474 |
| KV share of decode traffic @ 4K | **6.6%** | 2.1% |
| KV share of decode traffic @ 32K | **36.2%** | 14.7% |

**The smaller model makes the KV-sharing claim ~2.5× more visible.** At 350M and 32K context, KV
is 36% of decode traffic and halving the banks removes ~18% of it — that is a measurable effect,
unlike the 7% at 1.2B. This inverts the usual assumption that you need to scale up to show a
systems win. It is a strong argument for keeping the study at 350M rather than pushing to 1B, and
for making 32K the headline context.

Note also GQA is **3.00 d²** at this geometry, not 2.50 d² — because hq=16 gives q+o = 2d² but
k+v = 0.5d² over a smaller d². So the LIV-vs-GQA mixer comparison is 4.20M vs 3.15M here (LIV is
1.33× larger) rather than 16.78M vs 10.49M (1.60× larger). The brainlift's "LIV costs more than
GQA" point holds at both scales but is weaker at 350M — restate it for whichever scale is chosen.

## The silent-failure trap in P1 (derived and verified)

Independent of the literature, here is the arithmetic that most threatens the low-rank gate study.

For `W = BA` with `A ∈ R^{r×d}`, `B ∈ R^{d×r}`, i.i.d. entries of std σ_A, σ_B, and unit-variance
input:

```
Var(y_i) = d · r · σ_A² · σ_B²
```

The stock LIV block initializes with **Xavier** (`conv_use_xavier_init: true` in the released
config), so for the `d→3d` in_proj the gate slices have `std = sqrt(2/(d+3d)) = 1/sqrt(2d)`, giving
a gate output variance of `d · 1/(2d) = 0.5`. To hold gate scale fixed across a rank sweep you
therefore need `σ_A σ_B = 1/sqrt(2dr)`.

If instead both factors are initialized at the common `initializer_range = 0.02`:

| | r=64 | r=128 | r=256 | r=512 |
|---|---|---|---|---|
| **d=1024** | 47.7× too small | 23.8× | 11.9× | 6.0× |
| **d=2048** | 23.8× too small | 11.9× | 6.0× | 3.0× |

**The miscalibration is monotone in r — it shrinks by exactly the factor r.** So a fixed-std rank
sweep yields a smooth curve that reads as "higher rank is better." That is also the *expected*
result, so it will not look like a bug. It is a gate-initialization-scale curve wearing a rank
curve's clothing.

Two consequences for the protocol:
1. Per-arm init calibration (`σ_A σ_B = 1/sqrt(2dr)`) is **mandatory**, not a refinement.
2. Add a **cheap falsification check**: log gate output variance (and activation RMS) at step 0 for
   every arm and assert they match the full-rank control to within a tolerance. A one-line
   assertion kills the entire failure mode.

This matters more here than in typical LoRA settings because the low-rank output feeds a
**multiplicative** path (the gates), not an additive residual — so a scale error does not merely
shift a residual contribution, it rescales the whole convolution input.

## What this implies for the experiment

- The three proposals are **not** on equal footing. P1 has a real (if bounded, ~6%) decode-latency
  story; P2 is a clean memory-capacity story with no latency story at trainable contexts; P3 has
  no efficiency story at all and must be justified purely on quality.
- Report P2 as "resident KV banks halved" and be explicit that end-to-end decode is unchanged at
  ≤32K. Attempting a latency claim there will produce a null and look like a failed experiment
  when it is actually a correctly-predicted null.
- Any "our model is faster" claim needs a stated context length and should show the *slope* of
  state growth versus the all-GQA control, not a single-point comparison.
