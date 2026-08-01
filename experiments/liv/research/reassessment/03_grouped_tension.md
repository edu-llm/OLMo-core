# 03 — The grouped-vs-lowrank tension: is it the result, and is it publishable?

**Author:** reassessment team member 3. **Started:** 2026-08-01. **Status:** IN PROGRESS (written
incrementally; sections appended in order).

**Execution constraint honored:** no code was run on the local Mac. All arithmetic below is done by
hand and shown; anything requiring compute is specified as a FarmShare job, not executed.

Labels used throughout: **MEASURED** (a number that exists in a results file on disk, or a primary
source's reported number), **INFERRED** (my derivation from measured numbers), **ASSUMED** (a modeling
choice I am making that could be wrong).

---

## 1. Verification of the measurement

### 1.1 The latency numbers — MEASURED, and they replicate

Source of truth: `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/p1_verify_results.json`
(device string `"NVIDIA L40S"`, 3 trials each, CUDA-graphed).

| arm | trial medians (µs) | median | spread | kernels | MiB/tok |
|---|---|---:|---:|---:|---:|
| `dense` | 56.288 / 56.224 / 56.192 | **56.224** | 0.171% | 20 | 40.0 |
| `lowrank_fused r=128` | 60.800 / 60.864 / 60.832 | **60.832** | 0.105% | 30 | 10.0 |
| `lowrank_fused r=512` | 90.016 / 90.016 / 90.048 | **90.016** | 0.036% | 30 | 40.0 |
| `lowrank_sep r=128` | 76.320 / 76.480 / 76.576 | **76.480** | 0.335% | 40 | 10.0 |
| `grouped g=2` | 47.680 / 47.712 / 47.648 | **47.680** | 0.134% | 20 | 20.0 |
| `grouped g=4` | 47.616 / 47.584 / 47.600 | **47.600** | 0.067% | 20 | 10.0 |

**Deltas re-derived by hand (INFERRED from the above):**
- `lowrank_fused r=128` vs dense: (60.832 − 56.224)/56.224 = 4.608/56.224 = **+8.196% slower**. The
  docs say 8.2%. ✅ correct.
- `grouped g=4` vs dense: (56.224 − 47.600)/56.224 = 8.624/56.224 = **15.339% faster**. Docs say
  15.3%. ✅ correct.
- `lowrank_sep r=128` vs dense: (56.224 − 76.480)/56.224 = **−36.03%**. Docs say −36.0%. ✅ correct.

**Spread is genuinely tiny** (≤0.335%, all six arms) and the effect sizes (8.2%, 15.3%) are 25–200×
the spread. The latency ordering `grouped < dense < lowrank_fused < lowrank_sep` is **not in doubt**
on this hardware, at this shape, in this harness.

**One thing that is NOT established and is worth flagging:** `grouped g=2` (20 MiB) and `grouped g=4`
(10 MiB) are within 0.17% of each other — 47.680 vs 47.600 µs — despite g=2 reading **twice** the
bytes. That is the sharpest evidence in the whole dataset that this regime is **not bandwidth-bound at
all** for the structured arms; it is bound by something else (kernel count × per-kernel fixed cost, and
GEMV inefficiency). Note the implication the existing docs do not draw: if halving bytes from 20→10
MiB buys 0.17%, then `grouped`'s 15.3% win over dense is **not** a bandwidth win either. Dense reads
40 MiB in 56.2 µs; grouped reads 10 MiB in 47.6 µs. If bytes drove the time, grouped should be ~4×
faster, not 1.18×. So the correct mechanistic statement is: **all four arms are dominated by a
fixed per-kernel cost and by GEMM/GEMV shape efficiency, and bytes are close to irrelevant in this
batch-1 decode regime.** The README's framing ("skinny GEMVs cannot saturate the memory system") is
right in direction but understates it — the g=2/g=4 pair shows byte traffic buying essentially *zero*
time even for a shape that *is* efficient.

### 1.2 The iso-cost claim — MEASURED/verified, no arithmetic error

Checked by hand and re-checked numerically on the FarmShare login node
(`/scratch/users/ericrcwu/liv/probes/se_analyze.py`).

Per **one** gate, at d=1024:

| structure | formula | value | ÷ dense |
|---|---|---:|---:|
| dense | `d²` | 1,048,576 | 1.000 |
| lowrank r=128, separate | `d·r + r·d = 2dr` | 262,144 | 0.250 |
| lowrank r=128, **fused** (`d→2r` then two `r→d`) | `(d·2r + 2·r·d)/2 = 2dr` | 262,144 | 0.250 |
| grouped g=4 | `d²/g` | 262,144 | 0.250 |

For **both** gates: dense 2,097,152; lowrank r=128 fused = `d·2r + 2rd` = 262,144 + 262,144 =
**524,288**; lowrank r=128 separate = `2·(dr + rd)` = **524,288**; grouped g=4 = `2·d²/4` =
**524,288**. All three agree exactly. ✅

**The prompt's arithmetic is correct.** The general iso-cost relation is `2dr = d²/g` ⟺ **`r = d/(2g)`**.
At d=1024: (r=128 ↔ g=4), (r=256 ↔ g=2), (r=64 ↔ g=8). The probe's matched pairs (128,4) and (256,2)
are correct. Verified numerically: `r=128 fused == g=4 grouped → True`.

**Does the "fused d→2r shared down-projection" change the count?** No — and, importantly, it does not
change the *function class* either. `gate_down: d→2r` followed by `up_pre` reading `h[..., :r]` and
`up_post` reading `h[..., r:]` is *exactly* two independent `d→r→d` chains: `pre` depends on nothing
`post` uses. The name "fused"/"shared" is misleading — **nothing is shared**; it is a concatenation done
for kernel-count reasons only. Confirmed in
`/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer/src/olmo_core/nn/attention/short_conv.py:96-100,135-138`
and in `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/p1_launch_bench.py:88-103`.
So fused vs separate is a pure systems choice with **zero** modelling consequence — the design doc says
this and it is right.

⚠️ **Naming hazard, worth fixing in the docs:** a *genuinely* shared bottleneck (one `d→r`, two `r→d`)
would cost `3dr` = 393,216 for both gates (0.1875× dense), a *different* and cheaper arm. The design's
`S-shared` arm is presumably that. Do not let "fused" and "shared" be used interchangeably.

Also verified: `r=512` at d=1024 gives `4dr` = 2,097,152 = exactly `2d²` — **zero byte saving**, matching
the docs' `2dr ≥ d²` once `r ≥ d/2`. ✅

### 1.3 The energy numbers — MEASURED, reproduce exactly, but the metric is broken for masks

Recomputed from `structure_energy_results.json` (10 LIV layers × 2 gates = 20 cells, all with
`rank(Σ_x) = 1024`):

| structure | cost | mean retained | min | max |
|---|---:|---:|---:|---:|
| `lowrank r=128` | 0.25× | **0.9285** | 0.8215 | 0.9806 |
| `grouped g=4` | 0.25× | **0.1296** | 0.1034 | 0.2203 |
| *random mask 25%* — null | 0.25× | *0.1296* | 0.1003 | 0.2260 |
| `lowrank r=256` | 0.50× | 0.9650 | 0.9114 | 0.9914 |
| `grouped g=2` | 0.50× | 0.3356 | 0.2899 | 0.4621 |

The docs' 0.929 / 0.130 are correct. ✅ `grouped` beats the random mask in **11 of 20** cells (a coin
flip); mean difference **−0.00006**. The "identical to random" claim is MEASURED and exact.

#### FINDING 1 (methodological, moderate): "identical to random" is a near-tautology, not a discovery

The docs treat `grouped ≈ random mask` as evidence that "block structure buys nothing." It is weaker
than that: **it is what the metric must produce absent block-alignment.** For row `i` the metric is a
quadratic form `Σ_{j,k} m_ij m_ik w_ij w_ik Σ_jk`. Under a *random* mask of density `p`, the diagonal
`j=k` terms survive with probability `p` and the off-diagonal `j≠k` terms with probability `p²`. Under a
*block* mask with `g` blocks, row `i` keeps the `j=k` terms in its own block (fraction `1/g = p`) and
the `j≠k` terms only when both `j,k` land in that block (fraction `1/g² = p²`). **The two masks have
identical expected retention, term for term.** The measured coincidence therefore confirms only that the
trained `W` and `Σ_x` are not block-aligned in the identity channel order — which is exactly what the
permutation sweep separately shows. It is not independent evidence.

The same decomposition lets me back out something the docs never state. Writing `f` = fraction of
activation-weighted energy carried by *cross-channel* (`j≠k`) terms: retained = `p(1−f) + p²f`
= `0.25 − 0.1875f`. Setting that equal to the measured 0.1296 gives **f ≈ 0.642** — INFERRED:
**~64% of the energy that reaches the LIV gate output lives in cross-channel correlation, not in
per-channel magnitude.** That is the real reason any 25%-density mask scores 0.13 rather than 0.25, and
it is a genuinely interesting number for the paper.

#### FINDING 2 (methodological, SERIOUS): retained energy is not an error metric for masks

For the **low-rank** arm the probe truncates the SVD of `A = W·Σ^{1/2}`
(`structure_energy.py:42-46`). That is an **orthogonal projection**, so
`retained = 1 − ‖A − A_r‖²/‖A‖²` holds *exactly*: retention and approximation error are two names for
one quantity.

For the **mask** arms this identity fails. With `A = A_kept + A_dropped`,
`‖A‖² = ‖A_kept‖² + ‖A_dropped‖² + 2⟨A_kept, A_dropped⟩`, and the cross term
`tr((M⊙W)Σ((1−M)⊙W)ᵀ)` is **nonzero** whenever `Σ` is not diagonal — which, per Finding 1, is 64% of the
mass here. So `1 − retained ≠ relative error` for the grouped and random arms. **The headline table
compares an error metric against a non-error metric and reports the difference as "80 points."**

Sizing it (INFERRED, same decomposition): dropped energy fraction
= `(1−p)(1−f) + (1−p)²f` = `0.75·0.358 + 0.5625·0.642` = **0.630**. The cross term therefore carries
`1 − 0.1296 − 0.630 = 0.240` of the mass. The honest, apples-to-apples statement is:

> At iso-cost 0.25×, activation-aware **low-rank r=128 leaves 7.2% relative squared error;
> block-diagonal g=4 leaves ~63%.** An 8.8× gap in error — not a "7× gap in retention," and not
> "80 points."

Directionally the docs' conclusion survives. The magnitude and the phrasing do not.

#### FINDING 3 (methodological, SERIOUS — this one could flip a decision)

**The two arms were not given equal effort.** `retained_lowrank` computes the **optimal**
activation-weighted rank-r approximation (Eckart–Young after whitening). `retained_grouped` computes a
**naive mask of the trained weights** (`structure_energy.py:69-70`: `kept = ((W*mask) @ S_half)`). The
docstring concedes "the block mask is not optimized at all" but treats that as unavoidable. It is not.

The optimal block-diagonal approximation in the *same* `Σ`-weighted norm has a closed form. Minimising
`(w_i − m_i) Σ (w_i − m_i)ᵀ` over `m_i` supported on block `S = block(i)` gives the normal equations
`Σ[S,S] · m_Sᵀ = (Σ · w_iᵀ)[S]`, i.e.

```
m_S = ( Σ[S,S]⁻¹ · (Σ wᵀ)[S] )ᵀ        # NOT w[S]
```

This is exactly the **SparseGPT / OBS reconstruction step**: after fixing a support, re-solve the
surviving weights to absorb the pruned columns' contribution through `Σ`'s correlations. SparseGPT's
headline result is that this step is the difference between collapse and near-lossless at 50% sparsity
on OPT-175B, where naive magnitude masking fails. With 64% of the energy in cross terms here, the
recovery available to a block-diagonal fit is potentially very large.

**Cost to fix: negligible.** Rows within a block share the same `Σ[S,S]`, so it is **`g` Cholesky
factorisations of size (d/g)×(d/g) per gate per layer** — 4 × 256×256 factorisations plus 1024
triangular solves, per gate, per layer. Seconds of compute once `Σ` is in hand. The expensive part is
re-collecting `Σ` (32,768 tokens through LFM2-350M), which the existing probe already does in minutes.

**Until this is run, "the deficit is structural" is not supported.** What is supported is "the deficit is
structural *for the naive mask*." Those are different claims, and the second is the one SparseGPT tells
us to distrust. **This is my recommended immediate action — it is hours of work and it either kills or
substantially strengthens the headline.**

### 1.4 The ShuffleNet / channel-shuffle question — the prompt is RIGHT that there is an error, but for a different reason than stated

This is the item I was asked to investigate carefully. My verdict has three parts and they do not all
point the same way.

**(a) The literal ShuffleNet analogy does not apply, because the two gate projections are PARALLEL, not
serial.** MEASURED from the code — `short_conv.py:269-271`:

```python
pre_gate, post_gate, value = self.in_proj(x)       # three parallel maps of the SAME x
z = self._conv(pre_gate * value, cu_doc_lens)      # depthwise conv, per-channel over time
return self.out_proj(post_gate * z)
```

`B` (pre) and `C` (post) are both `d→d` maps applied to the *same* input `x` and multiplied into
*different* points of the path. There is no "between the two gate projections" — nothing flows from `B`
into `C`. So a channel shuffle inserted "between them" is not even well-defined in ShuffleNet's sense.

**(b) More importantly, the pathology ShuffleNet exists to fix is ABSENT here.** ShuffleNet's problem is
that stacking grouped convolutions with the same partition produces *permanently disconnected* channel
groups: information in group 1 can never reach group 2, no matter how deep. In the LIV block,
`value_proj` (`short_conv.py:89`) and `out_proj` (`short_conv.py:190`) are **dense in every variant** —
the code comment at line 62-63 says so explicitly and it is correct. So even with both gates
block-diagonal, output channel `i` still reads all 1024 input channels through `out_proj` and
`value_proj`. **The channel graph is never disconnected.** The grouped structure restricts only *which
channels may modulate a given lane's gate*, not which may reach the output. The existing analysis never
says this, and it materially weakens the "grouped is structurally crippled" story — the naive intuition
imported from grouped CNNs does not transfer to this block.

**(c) But the prompt's underlying charge lands, in a stronger form: the probe tested a SINGLE
block-diagonal factor, and the Monarch/ShuffleNet remedy is a TWO-factor product with a permutation
between — which is iso-cost at g=8 and was never tested.**

A Monarch matrix is `M = P₂ · BlockDiag_b · P₁ · BlockDiag_b`. Parameter count: `2 · b · (d/b)² = 2d²/b`.
Iso-cost with the r=128 / g=4 pair requires `2d²/b = d²/4` ⟹ **b = 8**. At d=1024 that is
`2 · 8 · 128² = 262,144` per gate — **exactly** the 262,144 of `lowrank r=128` and `grouped g=4`
(verified above). So there is a clean, perfectly iso-cost **third structure** the probe omitted:

| structure | per-gate params | mixes all channels? | rank |
|---|---:|---|---|
| `lowrank r=128` | 262,144 | yes | ≤128 (rank-deficient by construction) |
| `grouped g=4` | 262,144 | no (within-block only) | full (1024), but block-supported |
| **`monarch b=8`** | **262,144** | **yes** (via the inter-factor permutation) | **full** |

Monarch is the *literal* union of the two things being contrasted — it is block-diagonal (so it is
dense-tile, tensor-core friendly, the thing the L40S likes) *and* it is full-rank with global channel
mixing (the thing low-rank buys). **This is the arm that turns a two-way tension into a three-way
result with a resolution.** Its omission is the single biggest gap in the current arm set.

**(d) Separately, the permutation sweep's conclusion is over-claimed, and the reason is not subtle.**
The probe drew **3 uniformly random permutations** per cell (`structure_energy.py:138-139`) and found
[0.125, 0.133] against 0.130 for identity. That establishes: *the identity channel order is a typical
order*. It does **not** establish *no order helps*. Choosing a partition of 1024 channels into 4 groups
of 256 to maximise captured energy is a graph-partitioning / spectral-co-clustering problem over
1024!/(256!⁴·4!) ≈ 10⁶¹⁰ partitions; three uniform draws from that space are, by concentration, all
equivalent to each other and carry **zero** information about the optimum. The docs' inference "channel
ordering doesn't rescue it → the deficit is structural" is a **non-sequitur**.

**The cheap decisive test the probe should have run instead of the permutation sweep:** the **oracle
unstructured 25% mask** — for each row `i`, greedily keep the 256 columns with the largest marginal
contribution to `w_i Σ w_iᵀ` (or just solve the OBS-style support selection). This is an **upper bound on
any 25%-density mask whatsoever**, including the best possible block-diagonal one under the best
possible permutation. It costs the same as the existing probe. Two clean outcomes:
- oracle ≲ 0.4 → permutation search is provably futile, "structural deficit" is *proved*, and the
  paper's claim gets much stronger than it currently is;
- oracle ≳ 0.8 → the 0.130 number is an artifact of a bad support, the "structural" claim is dead, and
  the whole grouped-vs-lowrank framing needs rebuilding.

Either way the headline improves. **I rate this the second-highest-value cheap action after Finding 3.**

**(e) One genuinely novel structure the topology suggests, which no prior work I can find has named:**
give `B` and `C` **different** block partitions (e.g. `C`'s partition is `B`'s partition shuffled). Then
for lane `j`, `pre[j]` reads block `S_B(j)` and `post[j]` reads block `S_C(j)`, and the product
`post[j] · conv(pre[j]·value[j])` is bilinear in two *different* channel subsets — so the set of input
channels that can jointly modulate a lane grows from 256 to up to 512, at **zero** extra parameters.
This is the correct translation of "channel shuffle" into a parallel-gate topology. Cost: one line of
indexing. Worth including as a free ablation if the grouped arm is trained at all.

---

## 2. The scientific question, named — and the prior-art verdict

### 2.1 What the tension actually is

Stated precisely: **at identical parameter count and identical weight-byte traffic, the structure with
the best hardware realization on an L40S at batch-1 decode is the one with the worst activation-weighted
approximation of the trained dense operator.** 15.3% faster vs 8.2% slower; 63% relative error vs 7.2%.

The general principle underneath — and it is a real one, not an artifact — is a **conflict between two
different notions of "cheap":**

- **Approximation theory optimizes for rank.** Eckart–Young says the best `k`-parameter approximation of
  a matrix in *any* unitarily invariant norm, when the parameter budget is spent on a rank constraint, is
  the truncated SVD. Low-rank is *provably* the best structure at approximating an existing dense matrix.
  There is no competing structure; this is a theorem.
- **Hardware optimizes for arithmetic intensity and tile shape.** A `1024×1024` GEMV at batch 1 has
  arithmetic intensity ~1 FLOP/byte; so does a `1024×128`. But the `1024×128` cannot fill an L40S SM's
  tile, cannot use tensor cores effectively, and pays a fixed per-kernel cost that is now a large
  fraction of its total. A block-diagonal map is `g` independent `256×256` GEMMs issued as one `bmm` —
  **one** kernel, dense tiles, full occupancy per tile.
- **The two objectives are close to orthogonal, and here they are anti-correlated.** Low-rank is the
  unique minimizer of one and (with 2 kernels and a skinny inner dimension) close to the *maximizer* of
  the other cost.

The sharpest way to phrase it for a paper: **rank is what approximation theory pays for; tile shape is
what silicon pays for; the two currencies do not convert.** MEASURED exchange rate at this geometry:
buying back the 8.8× approximation-error advantage of low-rank costs **23.2%** of decode time
(47.6 → 60.8 µs) — i.e. the two structures sit on opposite ends of a quality/latency frontier that dense
does not lie on at all (dense is dominated on latency by grouped and dominated on approximation by
low-rank at equal cost... no: dense is *better* on approximation, at 4× the cost).

### 2.2 Is it real or an artifact? — mostly real, but two components are artifacts

| component | verdict | evidence |
|---|---|---|
| grouped is faster than lowrank at iso-cost | **REAL, and general** | MEASURED, ≤0.34% spread; mechanism (kernel count + tile shape) is hardware-generic, not L40S-specific |
| grouped is faster than *dense* | **REAL but narrow** | only at batch 1 with 10 layers of a 1024-dim gate on this card; at batch ≥ 32 the dense GEMM becomes compute-bound and grouped's advantage shrinks. NOT MEASURED — should be. |
| lowrank approximates trained weights far better | **REAL but partly tautological** | Eckart–Young makes it optimal *by construction*, and the probe gave grouped no reconstruction step (Finding 3) |
| the 7× / "80 point" gap size | **ARTIFACT of the metric** | correct number is 8.8× in error terms (Finding 2) |
| "block structure buys nothing over random" | **ARTIFACT** — it is what the metric must produce (Finding 1) | mean(grouped − random) = −0.00006, exactly as the `p` vs `p²` decomposition predicts |
| "channel ordering doesn't rescue it" | **NOT ESTABLISHED** | 3 uniform draws from a 10⁶¹⁰-element space (Finding, §1.4d) |

### 2.3 Prior art — and this is where the novelty claim gets hard

I searched the structured-matrix literature directly. The relevant thread is not the LLM-compression
literature the design doc has been reading (GaLore, FLAR-SVD, SparseGPT); it is the
**structured-matrices-for-efficient-training** line from the Ré and Wilson groups. It has already run
this comparison, and it has an answer.

**[1] Dao et al., "Monarch: Expressive Structured Matrices for Efficient and Accurate Training,"
ICML 2022 (arXiv:2204.00595).** MEASURED (from the paper): Monarch = product of **two block-diagonal
matrices with a permutation between them** — literally the ShuffleNet fix, formalized. Explicitly
motivated as "parameterized as products of two block-diagonal matrices **for better hardware
utilization**." Reports GPT-2 on OpenWebText and BERT pretraining at ~2× speedup at no quality drop; 23%
faster than the MLPerf 1.1 record BERT. Also proves the dense→Monarch projection has an **analytical
optimal solution** (which is exactly what my Finding 3 asks for, in the Monarch case).

**[2] Qiu, Potapczynski, Finzi, Goldblum, Wilson, "Compute Better Spent: Replacing Dense Layers with
Structured Matrices," ICML 2024 (arXiv:2406.06248).** Introduces Block Tensor-Train (BTT), a
superset of Monarch. Finds BTT beats dense at matched compute; ImageNet ViT-S/32 at **3.8× less
compute**. **Crucially for us: the paper's central methodological claim is that different structures
need different init scales and learning rates, derived via µP, and that this matters *more* as models
grow.** This is a direct hit on the design doc's Phase 3a note about init-scale confounds — the doc
already worries about this for the rank sweep, and this paper says it is the dominant confound for
*structure* comparisons too.

**[3] Potapczynski, Qiu, Finzi, Ferri, Chen, Goldblum, Bruss, De Sa, Wilson, "Searching for Efficient
Linear Layers over a Continuous Space of Structured Matrices," NeurIPS 2024 (arXiv:2410.02117).**
**This is the paper that most directly pre-empts the proposed experiment.** It searches all Einsum-
expressible linear operators, including low-rank, Kronecker, TT, BTT, and Monarch, and reduces the
scaling-law differences to two exponents:
- **ω** = parameter sharing (`N/F = Θ(d^ω)`); dense has ω=1, Monarch/BTT have ω=0.
- **ψ** = rank exponent (`rank(W) = Θ(d^ψ)`); full rank is ψ=1, **low-rank has ψ<1 (specifically ψ=ν)**.

Their finding, quoted: *"among structures without parameter sharing (ω=0), full-rank structures (ψ=1)
scale better than low-rank structures (ψ<1)"* and *"we show that small values of ψ indeed lead to worse
scaling laws."* Their mechanism, quoted: low-ψ structures *"introduce information bottlenecks in the
model by preventing the linear layers from accessing information from all the feature dimensions."*
And the summary judgment: *"full-rank structures that maximize parameters per unit of compute perform
the best"*; dense's strength comes *"not... from being dense, but rather from not sharing parameters and
being full-rank."*

Their scale (MEASURED): GPT-2 on OpenWebText, **120k to 76M parameters**, d ∈ [256, 4096], L ∈ {3,6},
100k steps × 65,536 tokens/batch, seq len 128 → **~6.5B tokens**, vocab reduced to **96** symbols so the
LM head doesn't swamp small-scale trends. µP-scaled LR, base 0.003.

⚠️ **They also record the exact caveat that keeps this project alive**, quoted: low-rank *"can beat dense
when controlling for memory, model size, or inference compute rather than training compute."*

**[4] Dao et al., "Pixelated Butterfly," ICLR 2022 spotlight (arXiv:2112.00029).** States flatly that
*"butterfly matrices are not hardware efficient,"* and introduces block/flat butterfly variants for
modern hardware — a **3× speedup over plain butterfly**. Its shipped sparsity pattern is
**flat block butterfly + low-rank**, i.e. the field's existing answer to the exact tension in this
document is *"use both."*

**[5] Fu, Arora, Grogan, Johnson, Eyuboglu, Thomas, Spector, Poli, Rudra, Ré, "Monarch Mixer,"
NeurIPS 2023 Oral (arXiv:2310.12109).** Monarch along both axes; at **360M params** matches GPT-quality
pretraining perplexity on The PILE; up to 9.1× higher throughput at 4K. Establishes that
block-diagonal-with-permutation trains from scratch fine at **exactly our target scale**.

### 2.4 Novelty verdict — ADVERSARIAL

**The general phenomenon is known and named, and someone has already run the from-scratch comparison.**
Specifically:

- "Block-diagonal-with-permutation is hardware-friendly and low-rank is not" — **[1], [4], known since
  2022.** Not novel.
- "Full-rank structured beats low-rank at matched compute, from scratch, on GPT-2 LM" — **[3], NeurIPS
  2024, with a scaling law and a mechanism.** The proposed from-scratch experiment as currently scoped
  (**lowrank r=128 vs grouped g=4 vs dense, iso-param, GPT-style LM**) is a **re-run of [3]'s Figure 4
  middle panel** at a different scale, with one fewer structure. **As stated, it is not publishable as a
  novel finding.**
- **⚠️ The single most damaging item:** [3] predicts the from-scratch outcome in advance and predicts
  it **against** the probe's prior. Plain block-diagonal (ω=0, ψ=1, full-rank) is in the *good* region;
  low-rank (ψ<1) is in the *bad* region. **The published scaling law says `grouped` should train BETTER
  than `lowrank` from scratch, while the energy probe says it should be 8.8× worse.** These are not
  reconcilable by hand-waving. That collision — a published scaling-law prediction versus a measured
  post-hoc approximation prior, pointing in **opposite directions** at iso-cost — is a far more
  interesting object than either the latency finding or the energy finding alone.

**What survives as genuinely novel:**

1. **The regime.** [3] compares at **training-compute-optimal** allocation. Everything in this project is
   **inference-decode at batch 1**, where [3] explicitly declines to claim ("low-rank can beat dense when
   controlling for memory, model size, or inference compute rather than training compute"). Nobody in
   [1]–[5] reports **CUDA-graphed batch-1 decode microbenchmarks with measured kernel counts and an
   iso-byte control**. The measurement in `p1_verify_results.json` is, as far as I can find, **not in the
   literature**.
2. **The negative systems result.** *"Factorization saves 4× the bytes and is 8.2% slower; the roofline
   model gets the sign wrong; you need an iso-byte control"* — MEASURED, generalizable, and I find no
   paper stating it this cleanly for decode-time LLM factorization.
3. **The operator.** All of [1]–[5] structure **dense linear layers in MLPs/attention projections**.
   Nobody has structured **the gates of a gated short convolution** — an operator that is multiplicative,
   not additive, so the usual "information bottleneck" argument for why low-ψ hurts may not transfer:
   a bottlenecked *gate* still multiplies a full-rank value stream, and `value_proj`/`out_proj` stay
   dense (§1.4b). **This is the only place a from-scratch result could genuinely surprise.**
4. **Monarch as the resolution.** [3] says the winning region is ω=0, ψ=1. Monarch b=8 is exactly that,
   and is **exactly iso-cost** with both existing arms at d=1024 (§1.4c). Adding it converts a re-run
   into a test with a predicted, falsifiable, and *hardware-favorable* answer.

---

## 3. The decisive experiment

### 3.1 What the from-scratch prior actually says — and it is contested

Two bodies of evidence point in **opposite** directions. This is the crux, so I am laying both out with
numbers.

**Evidence that low-rank collapses from scratch (favors `grouped`):**

GaLore Table 2, LLaMA pre-training on C4 (MEASURED, quoted from arXiv:2403.03507v2). Validation
perplexity, "Low-Rank" = plain `W = BA` reparameterization from scratch:

| size | tokens | rank r / d | Full-Rank | **Low-Rank** | GaLore | LoRA |
|---|---:|---|---:|---:|---:|---:|
| 60M | 1.1B | 128 / 256 | 34.06 | **78.18** | 34.88 | 34.99 |
| 130M | 2.2B | 256 / 768 | 25.08 | **45.51** | 25.36 | 33.92 |
| 350M | 6.4B | 256 / 1024 | 18.80 | **37.41** | 18.95 | 25.58 |
| 1B | 13.1B | 512 / 2048 | 15.56 | **142.53** | 15.64 | 19.21 |

Note the **350M / d=1024 row is literally our geometry**: r=256 (r/d = 0.25) gives **37.41 vs 18.80** —
a **1.99×** perplexity blowup. And our r=128 at d=1024 is **twice as aggressive** as that. This is a
much stronger adverse datapoint for low-rank than the docs currently make of it (the docs cite only the
1B row).

⚠️ **But GaLore's "Low-Rank" baseline factorizes EVERY linear layer including the MLP.** P1 factorizes
**only the two gates of 10 LIV layers**, leaving `value_proj`, `out_proj`, all attention, and the whole
MLP (69% of the model) dense. The relevant fraction is 524,288 × 10 = 5.24M of 354.5M params = **1.5% of
the model.** GaLore's numbers are therefore an upper bound on the damage by a very wide margin — I would
not expect anything like a 2× perplexity effect. INFERRED.

**Evidence that block-diagonal trains fine from scratch and low-rank is the loser (favors `grouped`
too, from the other direction):**

- **[3] (NeurIPS 2024)**: within ω=0, *"full-rank structures (ψ=1) scale better than low-rank structures
  (ψ<1)"*; block-diagonal is ψ=1, low-rank is ψ<1. Structures in the ω=0/ψ=1 subspace have *"nearly
  indistinguishable scaling laws compared to dense matrices."*
- **[5] Monarch Mixer, 360M on The PILE**: matches GPT-quality perplexity with block-diagonal products.
- **[1] Monarch, GPT-2/BERT**: ~2× speedup at no quality drop.

**Both literatures agree the from-scratch ordering should be the OPPOSITE of what the energy probe
predicts.** The probe says `grouped` is 8.8× worse in approximation error; the scaling-law literature
says full-rank-block-diagonal ≈ dense while low-rank is the one that degrades. This is not a small
discrepancy — it is a sign flip.

**Resolution (my INFERRED reading, and the paper's real thesis):** *approximation quality of a trained
dense operator is a bad predictor of from-scratch trainability, and the two disagree in a
systematic, explainable direction.* Post-hoc approximation rewards concentrating energy (Eckart–Young);
from-scratch training rewards **not bottlenecking gradient flow** and **full-rank reachability**.
Low-rank wins the first and loses the second. **That is a publishable claim, and it is the thing this
project is uniquely positioned to demonstrate, because it holds both metrics on the same operator at
exact iso-cost.**

### 3.2 Expected effect size — SMALL, and this is the design's biggest risk

Three independent reasons the from-scratch gap will be far smaller than 0.929-vs-0.130 suggests:

1. **Only 1.5% of parameters are touched** (5.24M of 354.5M, computed above). The gates are 2 of 4 `d²`
   blocks in a mixer that is itself ~31% of the model, in 10 of 16 layers.
2. **The gates are multiplicative, not additive.** A rank-128 gate still multiplies a full-rank value
   stream; a block-diagonal gate still feeds `out_proj`, which is dense. Neither creates the
   "information bottleneck" [3] blames for low-ψ failure (§1.4b). ASSUMED, and it is the assumption most
   worth stating in a paper.
3. **Sieberling et al.'s span sweep gained 0.00 ppl** and published LFM2-family ratio sweeps span
   **0.06 ppl** — this repo's own §2 notes those are below noise.

**My prediction (ASSUMED, stated in advance so it is falsifiable):** at 350M / 7B tokens the
`dense` vs `lowrank r=128` vs `grouped g=4` CE spread will be **< 0.02 nats**, i.e. **below the
detection threshold at any seed count this project can afford.** The design doc's own power analysis
says a +0.010-nat margin needs `s_δ ≲ 0.011` at n≥8, and the KDA study measured a +0.0053-nat effect
needing **n≈43 seeds**.

⚠️ **This is the strongest argument against running the experiment as scoped.** A three-arm iso-param
CE comparison at n=3-5 seeds is **designed to return a null**, and a null here is uninformative because
it is exactly what the power analysis predicts regardless of the truth.

**The fix — do not measure CE.** Use endpoints where the arms *must* differ if the mechanism is real:
- **MQAR at the `N512_D64` operating point** (already calibrated, §3 of HANDOFF): recall is where the
  published ratio sweeps show 20+ point spreads against 0.06 ppl. If a bottlenecked gate hurts, it hurts
  *binding*, not average likelihood.
- **Gate rank / participation ratio at convergence** — for each arm, measure the achieved effective rank
  and the achieved cross-channel energy fraction `f` (§1.3 Finding 1) of the *trained* gate. This
  directly tests whether `grouped` recovers globality it structurally lacks and whether `lowrank`
  actually uses its 128 directions. Cheap, and it is the mechanistic figure of the paper.
- **AR-Hits sliced perplexity**, per the existing protocol.
- **The 2×2 the paper needs:** post-hoc approximation error (already have it) × from-scratch loss (the
  new run), on the *same* operator at exact iso-cost. That scatter plot, if it is off-diagonal, IS the
  paper.

### 3.3 GPU-hours for the full-scale version — the prompt's 6ND figure, checked

6ND at N = 354,483,968 and D = 7.09B tokens (20×N, Chinchilla):

`C = 6 · 3.545e8 · 7.09e9 = 1.508e19` FLOPs.

At 40% MFU on 8×A100-40GB (312 TFLOP/s bf16 dense each ⇒ 2.496e15 aggregate; ×0.40 = 9.98e14):
`1.508e19 / 9.98e14 = 15,110 s = 4.20 wall-clock hours = 33.6 A100-hours` **per arm**.

- 3 arms × 3 seeds = 9 runs → **302 A100-hours ≈ 38 wall-clock hours** on one 8×A100 node.
- 4 arms (adding Monarch) × 5 seeds = 20 runs → **672 A100-hours ≈ 84 hours**.
- 4 arms × 8 seeds (the count the power analysis actually demands) = 32 runs → **1,075 A100-hours ≈
  134 hours ≈ 5.6 days** on one 8×A100 node.

The prompt's framing of "8×GPU-days" is right: **the properly-powered version is ~5-6 node-days.**
INFERRED; the 40% MFU assumption is generous for a hybrid conv/attention model in OLMo-core with an
unfused custom mixer — at 30% MFU multiply by 1.33.

**Adverse note:** at 33.6 A100-hours per run this is *cheap enough to just do*, which makes the
"can we do it cheaper" question less important than the **"is the endpoint powered"** question. Spending
1,075 A100-hours to measure a quantity your own power analysis says needs n=43 is the actual risk here,
not the GPU bill.

---

## 4. The cheap version — yes, and it fits FarmShare comfortably

### 4.0 ⚠️ Correction to a load-bearing premise

I was told FarmShare gives "1 GPU per job with a 6-hour limit." **That is wrong, and I checked it
live.** MEASURED via `scontrol` / `sacctmgr` on 2026-08-01:

```
PartitionName=gpu   Nodes=oat-[01-06]   DefaultTime=02:00:00   MaxTime=2-00:00:00
                    TotalNodes=6  gres/gpu=24  (4x L40S 46GB per node)
QOS gpu:  MaxWall=(none)  MaxTRESPU=gres/gpu=4  MaxJobsPU=4  MaxSubmitPU=32  DenyOnLimit
```

So the real envelope is: **up to 4 concurrent GPUs per user, 32 queued jobs, and up to 48 hours of
walltime per job** (default 2 h if you do not ask; you must pass `-t`). Other users are currently
running 2-day GPU jobs, so it is enforced as stated. `--exclusive` **never schedules** (KDA/HANDOFF.md
notes this, verified) — every node has other tenants, so keep the memory footprint small instead.

**This changes item 4's answer substantially: the cheap version is not merely possible, it is not even
tight.** Four concurrent single-GPU jobs × 48 h = **192 L40S-hours per wave**, and 32 can be queued.

### 4.1 Throughput calibration — MEASURED, from this repo's own completed run

From `/Users/ericwu/Developer/Capstone_LLM/KDA/HANDOFF.md:541-596`, a real LM pretraining run on
FarmShare L40S:

- 52.1M non-embed (77.8M total), 12 layers, d=512, T=2048, micro-batch 4 × accum 4
- 31,800 steps → **1.04B tokens**
- measured **~4.9 h/run** for the cheapest arm (`hh1`), 11.9 h for the most expensive (KDA R=4, a much
  heavier mixer than a short conv)

⇒ **INFERRED calibration: ~1.04B tokens in ~4.9 h on one L40S at ~78M total params ≈ 59k tok/s.**
A gated short conv is *cheaper* per token than KDA-Householder, so this is conservative for us.
Sanity check against the roofline: 6ND at N=52.1M, D=1.04B = 3.25e17 FLOPs / 17,640 s = 18.4 TFLOP/s
= **12.5% MFU** of the L40S's ~147 TFLOP/s bf16 dense. Low, but that is a real measured number on a
custom-mixer model with a micro-batch of 4 — and it is the right number to plan against, not a
theoretical MFU.

### 4.2 The cheap design — precise

**Scale: d=768, 16 layers, LFM2 topology (10 ShortConv + 6 GQA at `[2,5,8,10,12,14]`).**

Why d=768 and not smaller: the iso-cost relation `r = d/(2g)` needs `d` divisible by `2g` for all arms,
and Monarch b=8 needs `d/b` integer. 768 = 2⁸·3 works for g∈{2,4}, r∈{192,96}, b∈{8,12}. It also keeps
`r/d = 0.125` **identical to the headline 350M configuration** (128/1024), so the compression ratio —
the thing being studied — transfers exactly. A d=512 model would force r=64, and at r=64 you are
measuring a different point on the curve.

Estimated params (INFERRED, using the arm-builder formulas): mixer per LIV layer `4d² + kd` = 2.36M;
MLP with LFM2's `ff = 256·ceil(⌊2/3·b⌋/256)` transform; GPT-2 vocab 50,257 untied. **≈ 90M non-embed,
≈167M total with untied GPT-2 embeddings.** Use a **32k tokenizer or tied embeddings** to cut the
embedding tax — at d=768 untied GPT-2 embeddings are 77M, i.e. **46% of the model**, which would swamp
a 1.5%-of-params intervention. ⚠️ This is a real design hazard the small scale introduces and the 350M
scale does not.

**Token budget: 1.8B (20× the 90M non-embed).** Chinchilla-optimal, no repeats.

**Wall-clock per run (INFERRED from §4.1):** 90M/52M × 1.8B/1.04B × 4.9 h = **14.7 h.** Under the 48 h
limit with ~3× headroom. At micro-batch 8 (a short conv is far lighter than KDA, ~1.75 GiB/layer was
the KDA figure at mb=4) expect materially better; budget **12-15 h and request `-t 24:00:00`**.

**Arms (4) — and Monarch is not optional, it is the point:**

| arm | gate structure | per-gate params @ d=768 | iso-cost? |
|---|---|---:|---|
| `D` | dense | 589,824 (`d²`) | 1.00× (control) |
| `F` | lowrank r=96 (fused `d→2r`) | 147,456 (`2dr`) | **0.25×** |
| `G` | grouped g=4 | 147,456 (`d²/4`) | **0.25×** |
| `M` | **monarch b=8** | 147,456 (`2d²/b`) | **0.25×** |

Verified by hand: `2·768·96 = 147,456`; `768²/4 = 589,824/4 = 147,456`; `2·768²/8 = 1,179,648/8 =
147,456`. **All three cheap arms are exactly equal**, and exactly 0.25× dense. ✅

**Seeds: 5 per arm** (not 3). 4 arms × 5 seeds = **20 runs.**

**Wall-clock: 20 runs × ~15 h / 4 concurrent GPUs = 5 waves × 15 h = ~75 h ≈ 3.2 days**, unattended,
as a `--array=0-19%4` job. **Total ~300 L40S-hours.** That is the honest cost — "hours on 1 GPU" as the
prompt hoped is **not** achievable for a properly-seeded LM comparison; three days of a 4-GPU
allocation is.

**A genuinely hours-scale option that IS available and should run first:** the **MQAR arm comparison**
at the already-calibrated `N512_D64` operating point. 4 layers, d=128, 8000 steps × batch 64. The
existing calibration ran 45 configs in single FarmShare jobs. 4 arms × 10 seeds = 40 runs, and at
~5 min/run that is **~3.5 GPU-hours total, one afternoon.** ⚠️ HANDOFF.md:410-414 warns the operating
point does not transfer to real `L0` — true for the *numbers*, but for a **relative comparison of four
gate structures inside the same 4-layer model**, the calibration is exactly what it is for.
**This is the highest information-per-GPU-hour experiment available and it should be run before
anything else.**

### 4.3 Would a small-scale result be convincing? — Partly. Be honest about which half.

**Convincing at 90M:**
- **A positive result.** If `G` or `M` beats `F` on MQAR or CE at 90M, that is a real dissociation
  between approximation error and trainability, and small scale does not undermine it — the effect
  exists. [3] ran its entire scaling law at **120k-76M params**, *below* this proposal, and published
  it at NeurIPS. Small scale is not disqualifying in this literature; it is the norm.
- **Rank/energy diagnostics of the trained gates.** Scale-robust mechanism evidence.

**NOT convincing at 90M:**
- **A null.** Three arms within noise at 90M tells you nothing about 350M, and the power analysis says
  a null is the expected outcome (§3.2).
- **Any latency claim.** The 15.3% number is a d=1024 batch-1 result; at d=768 the tile geometry
  changes. Re-benchmark, do not extrapolate.
- **Any argument about compute-optimal scaling.** [3] owns that and did it properly.

**Scale-sensitivity caveat that cuts the other way:** [2] (ICML 2024) finds structure-specific init and
LR matter *more* as models grow. So a small-scale tie could become a large-scale gap, or vice versa. A
single scale cannot distinguish. **If any headline scaling claim is wanted, two scales are mandatory**
(e.g. d=768 and d=1024, ~2.2× apart) — that doubles the bill to ~700 L40S-hours, still inside FarmShare.

### 4.4 ⚠️ The confound that will invalidate the cheap run if ignored

[2]'s central methodological finding is that **structured layers need structure-specific init scale and
LR**, and the design doc already flags the analogous trap for the rank sweep ("error is monotone in r:
24-48× too small at default init"). The three cheap arms have **very different fan-in**:

| arm | effective fan-in of the gate map |
|---|---|
| `F` lowrank | `d`=768 into the down-proj, `r`=96 into the up-proj |
| `G` grouped | `d/g`=192 |
| `M` monarch | `d/b`=96 per factor, twice |

At a fixed `std`, these produce **wildly different step-0 gate output variance**. The design doc already
requires "step-0 gate output variance parity with `L0`" for the rank sweep and there is a test for it
(HANDOFF.md:264-269). **Extend that assertion to `G` and `M`, and additionally sweep LR per arm** (at
minimum 3 points, µP-style), or the result measures init scale rather than structure. Without this the
run is worthless — this is the single most likely way the cheap experiment produces a confident wrong
answer.

---

## 5. 🔴 THE DECISIVE PRIOR ART — found late, and it changes the verdict

**[6] Wei, Moalla, Pascanu, Gulcehre, "Building on Efficient Foundations: Effectively Training LLMs
with Structured Feedforward Layers," NeurIPS 2024 (arXiv:2406.16450).** Code:
`github.com/CLAIRE-Labo/StructuredFFN`.

**This paper runs almost exactly the experiment being proposed, from scratch, on LMs, at matched
parameters, at 110M → 1.3B.** Its three arms are:

| their name | definition | our arm |
|---|---|---|
| **LowRank** | `U^r(V^r x)`, cost `(M+N)R` | ≡ our `F` (`lowrank r`) |
| **BlockShuffle** | `f⁻¹(U^b f(V^b x))` — two block-diagonal factors **with a shuffle between them**, explicitly motivated by **ShuffleNet and Monarch** | ≡ our proposed `M` (`monarch`) |
| **BlockDense** | `U^r(V^b x)` — block-diagonal then dense | (no analogue; `B=1` recovers LowRank) |

So **the "channel shuffle between two block-diagonal factors" idea in my §1.4c/e — and in the
assignment prompt — is already published, tested at LM scale, and named BlockShuffle.**

### 5.1 Their result (MEASURED, quoted from the paper, RefinedWeb, ~20 tok/param Chinchilla)

| scale | params | tokens | Dense | LowRank | BlockDense | **BlockShuffle** |
|---|---:|---:|---:|---:|---:|---:|
| -s | 90.17M (from 110M) | 2.2B | 25.97 | **27.16** | 27.20 | **27.63** |
| -s | 73.95M (32%) | 2.2B | 25.97 | **29.22** | 29.17 | **29.95** |
| -m | ~262M (from 335M) | 6.7B | 18.29 | **19.12** | 19.26 | **19.34** |
| -m | ~202M (32%) | 6.7B | 18.29 | **20.60** | 20.85 | **21.12** |
| -l | 566M (from 729M) | 14.6B | 14.29 | **14.82** | 14.94 | 14.91 |
| -xl | ~985M (from 1274M) | 25.5B | 12.46 | **12.86** | 12.97 | 12.98 |

Their stated conclusion: **LowRank wins, BlockDense a hair behind, BlockShuffle consistently last** —
*"showing a 0.8 lower perplexity on Transformer-s and a 0.4 lower perplexity on Transformer-m"* — and
they speculate that for LM FFNs *"BlockShuffle may not be the optimal choice."* They also report the
ranking **flips on CIFAR-10** where locality helps (BlockShuffle 67.08% vs LowRank 64.04%).

### 5.2 What this does to the project — three consequences, in order of severity

**(1) The from-scratch prior now favors LOW-RANK, not grouped, and it agrees with the energy probe.**
This is the opposite of what I concluded in §3.1 from [3]. Both literatures are real; they disagree:

| source | prediction for from-scratch iso-param LM | basis |
|---|---|---|
| [3] Potapczynski et al., NeurIPS 2024 | **block-diagonal/Monarch (ψ=1) > low-rank (ψ<1)** | Einsum scaling laws, GPT-2, 120k–76M, compute-optimal allocation |
| [6] Wei et al., NeurIPS 2024 | **low-rank > BlockShuffle by 0.4–0.8 PPL** | FFN replacement, 110M–1.3B, param-matched, ~20 tok/param |
| this repo's energy probe | **low-rank ≫ grouped (8.8× error)** | post-hoc activation-weighted approximation |
| this repo's L40S bench | **grouped 24% faster than low-rank at iso-cost** | batch-1 decode, CUDA-graphed |

**Two NeurIPS 2024 papers from different groups predict opposite orderings for structured-vs-low-rank
at matched parameters in language models.** [3] compares at matched *training compute*; [6] at matched
*parameters*. That is very likely the reconciliation — the [3] caveat quoted in §2.3 says exactly this
("low-rank can beat dense when controlling for memory, model size, or inference compute rather than
training compute"). But nobody has stated the reconciliation, and nobody has tested it on a gated
operator. ⚠️ **This unresolved disagreement is more interesting than the original tension and it is
citable to two published papers.**

**(2) The novelty verdict tightens to NEGATIVE for the experiment as scoped.** With [6] in hand:
- `lowrank vs grouped-with-shuffle at iso-param from scratch on an LM, 110M–1.3B` — **done, NeurIPS
  2024.** Our proposed 90M cheap version is *below* their smallest scale.
- Plain `grouped` (no shuffle) is not in [6] — but [6]'s BlockShuffle *strictly dominates* plain grouped
  in expressivity, and it already loses. Testing a weaker structure that the stronger version already
  lost with is not a contribution.
- **What is still untouched: the operator.** [6] structures FFNs; [1][2][3][5] structure dense linear
  layers and attention projections. **Nobody has structured the gates of a gated short convolution.**
  That is the only surviving axis.

**(3) [6] independently confirms my §4.4 confound warning, and escalates it.** Their "self-guided
training" section documents exactly the failure mode: factorization `W=UV` introduces the symmetry
`UV = (UC)(C⁻¹V)`, multiplying saddle points; the effective update is
`Δ(UV) = −lr·(UUᵀgxᵀ + gxᵀVᵀV) + O(lr²)`, so the *measured* spectral norms of `UUᵀ`/`VᵀV` swing far
above and below 1 depending on rank/block count. Consequence they report: *"at large learning rates it
shows instability and loss spikes, at small ones it converges much slower than dense."*

**Their fix and its size:** blend a decaying dense residual, `o = α·Wx + (1−α)·U(Vx)`, `α` cosine 1→0,
with `W₀ = U₀V₀`. Effect: **PPL improves 1.2 (-s) and 0.8 (-m)**, and at -xl the dense gaps shrink from
1.0/1.2/1.3 → **0.4/0.5/0.6** for LowRank/BlockDense/BlockShuffle.

⚠️ **This is decisive for our experiment design and the current plan does not account for it.** The
self-guided-training correction (**up to 1.2 PPL**) is **larger than the entire between-structure gap
being measured (0.4–0.8 PPL)**. A naive from-scratch run therefore measures *optimization pathology*,
not *structure*, and the pathology is **structure-dependent** (different `B`/`R` give different
projector norms). **Any from-scratch arm comparison that does not either (a) apply self-guided training
to all arms, or (b) tune LR and init per arm and report the sweep, is measuring the wrong thing.** This
is now the #1 validity threat to the whole from-scratch plan, ahead of statistical power.

### 5.3 Revised novelty verdict

**NOT NOVEL** as scoped (lowrank vs grouped vs dense, iso-param, LM, from scratch). Covered by [6] with
more arms, more scales, more tokens, and a diagnosis of the confound.

**NOVEL, if narrowed to what nobody has done:**
1. **Structuring the gates of a gated convolution** rather than an FFN or an attention projection. The
   multiplicative topology (§1.4b: `value_proj`/`out_proj` stay dense, so no channel is disconnected;
   the gate multiplies a full-rank stream) is a mechanistically different setting from every paper
   above, and [6]'s own CIFAR-10 flip shows the ordering is **operator-dependent**, not universal.
2. **Batch-1 CUDA-graphed decode latency with measured kernel counts and an iso-byte control.** No
   paper in [1]–[6] reports this. [6] reports decode gains only via a "pre-merge technique."
3. **The two-metric dissociation figure**: post-hoc activation-weighted approximation error vs
   from-scratch loss, on the same operator, at exact iso-cost, with latency as a third axis.

---

## 6. Publishability

### 6.1 Is the tension the most interesting available result? — NO, but it contains the most interesting one

**Adversarial answer to the assignment's headline question.** The tension as currently stated — *"the
fastest structure approximates trained weights worst at iso-cost"* — is **not** the most interesting
available result, for three reasons established above:

1. Half of it is a theorem, not a finding. Low-rank is *provably* the best iso-cost approximator
   (Eckart–Young). "The non-optimal approximator approximates worse" is not news.
2. The magnitude is a metric artifact (Finding 2: 8.8× in error, not 7× in retention / "80 points"),
   and the grouped arm was handicapped by omitting the reconstruction step (Finding 3).
3. The hardware half is known since 2022 ([1] Monarch, [4] Pixelated Butterfly both state block-diagonal
   is hardware-friendly and butterfly/skinny is not).

**What IS the most interesting available result, ranked:**

**#1 — The measured decode negative result, and the methodological lesson.** *"A 4× reduction in weight
bytes made decode 8.2% slower; the roofline model predicted the wrong sign; the same parameter budget
spent as block-diagonal was 15.3% faster."* MEASURED, ≤0.34% spread, kernel counts profiled, with an
iso-byte control that identifies the mechanism. **I could not find this in the literature.** It is
directly actionable for anyone shipping a factorized decode path, and it is *already done* — zero
additional GPU-hours.

**#2 — The prediction collision.** Two NeurIPS 2024 papers ([3] and [6]) give **opposite** orderings for
low-rank vs block-structured at iso-cost; a post-hoc approximation metric agrees with one and a
scaling-law taxonomy with the other; and the fastest structure is the one they disagree about. Nobody
has stated this, and it is checkable.

**#3 — The three-way iso-cost coincidence.** At d=1024, `lowrank r=128`, `grouped g=4`, and
`monarch b=8` are **exactly** 262,144 params per gate (verified §1.4c). Three structures, one budget,
spanning (low-rank, full-rank-local, full-rank-global). That is a clean experimental design and it is
currently missing its third arm.

The grouped-vs-lowrank tension is the *frame* that makes #1–#3 legible. It is not itself the result.

### 6.2 Venue class

**Realistic ceiling: a strong workshop paper, or a short/findings-track conference paper.** Concretely:

| venue class | fit | why |
|---|---|---|
| **NeurIPS/ICML/ICLR main track** | ❌ | [6] covers the from-scratch iso-param comparison at 110M–1.3B with more arms and a mechanism; [3] covers the scaling law. A 90M-or-350M three-arm re-run does not clear the bar. |
| **ICML/NeurIPS ES-FoMo / ENLSP / Efficient Natural Language and Speech Processing workshop** | ✅ **best fit** | Exactly their remit: measured efficiency negative results, kernel-level analysis, small-scale controlled ablations. The decode negative result alone is a good ES-FoMo paper. |
| **MLSys** (short/poster) | ✅ if systems-forward | The iso-byte control, kernel counts, and "roofline gets the sign wrong" story is a systems contribution. Needs a second GPU (A100 or H100) to show generality — currently L40S-only, which is a real weakness. |
| **ACL/EMNLP Findings** | 🟡 | Possible if the gated-conv-specific quality result is real and the LFM2 framing (a shipped edge architecture with no published ablations) is foregrounded. |
| **Capstone deliverable** | ✅ | Comfortably sufficient as-is. |

⚠️ **The single biggest publishability liability is L40S-only.** Every latency claim rests on one card.
[4] and [1] both frame hardware efficiency as architecture-dependent. Getting the same three arms on an
A100 or H100 (SB-AWS has these per HANDOFF §0) is **cheap — under an hour** — and converts "on an L40S"
into "across two/three architectures." **Do this before writing anything.** It is the highest
value-per-GPU-minute action in this entire document.

### 6.3 Single-sentence headlines, both outcomes

**Framing that survives either way (the key design property):**

> *At exactly equal parameter count and equal weight-byte traffic, the choice among low-rank,
> block-diagonal, and Monarch gate factorizations in a gated short convolution moves decode latency by
> 24% and post-hoc approximation error by 8.8×, in opposite directions — we measure which one the
> from-scratch training actually prefers, and find that [X].*

**Outcome A — grouped/Monarch matches or beats low-rank in quality:**
> *"The structure that best approximates a trained gate is the worst choice for training one: at
> identical cost, block-diagonal gates match low-rank quality from scratch while decoding 24% faster,
> so activation-weighted approximation error — the standard proxy for choosing a compression structure
> — mispredicts the from-scratch ordering by 8.8×."*
This is the strong outcome. It is a **falsification of a widely-used selection proxy** (activation-aware
SVD / SparseGPT-style Hessian criteria are used exactly this way in practice), plus a free speedup. It
also lands on the side of [3] against [6], which makes the collision in §5.2 the paper's spine.

**Outcome B — low-rank wins quality, as the energy probe and [6] both predict:**
> *"Post-hoc activation-weighted energy correctly ranks gate factorizations for from-scratch training,
> but the ranking is the reverse of the hardware ranking: buying low-rank's 8.8× approximation
> advantage costs 24% of decode latency, so at fixed parameters there is no free structure — the
> quality-optimal and latency-optimal factorizations are distinct and the gap is quantified."*
Weaker (it confirms [6]) but still a real, reportable frontier, and the decode measurement carries it.

⚠️ **Outcome C, the likely one, and it needs a pre-registered plan:** all three arms tie within noise
(§3.2 predicts CE spread < 0.02 nats against a detection threshold needing n≈43 seeds). The honest
headline is then:
> *"At 1.5% of parameters, gate factorization structure does not measurably affect language-modeling
> quality at n=5 seeds — so choose it on latency, where the difference is 24% and unambiguous."*
That is a **perfectly good workshop result** and arguably the most useful one for practitioners. But it
must be **pre-registered as an acceptable outcome with a stated equivalence margin**, or it reads as a
failed experiment. **State the margin before running.** Recommend: declare equivalence if the 95% CI on
each pairwise CE difference falls within ±0.02 nats.

### 6.4 A third structure — yes, and specifically these

**Mandatory: `monarch b=8`.** Exactly iso-cost (262,144 at d=1024, verified). It is the union of the two
things being contrasted — block-diagonal tiles (hardware) and full-rank global mixing (expressivity) —
and [3]'s taxonomy predicts it should win (ω=0, ψ=1) while [6]'s measurement says it should lose
(BlockShuffle last). **A structure that two published papers disagree about, at exact iso-cost with both
existing arms, is the best possible third arm.** Without it the study is two points and no resolution.

**Recommended: `grouped g=4` with different partitions for the pre- and post-gate** (§1.4e). Free,
one line, and it is the correct translation of "channel shuffle" into a *parallel* gate topology — which
is genuinely not what [6]'s BlockShuffle does (theirs is serial). Small but possibly novel.

**Recommended control: `N-narrow`.** Already in the arm builder. Without it, any quality difference is
confounded with "the model is just smaller."

**Do NOT add: BlockDense.** [6] measured it and it tracks LowRank within 0.1–0.25 PPL. No information.

### 6.5 Recommended action ordering (cost-ranked, highest information per GPU-hour first)

| # | action | cost | what it buys |
|---|---|---|---|
| 1 | **Re-run `structure_energy.py` with the OBS/SparseGPT reconstruction step** (§1.3 Finding 3) and an **oracle unstructured 25% mask** (§1.4d) | CPU/1 GPU, ~1 h | Either kills or proves "the deficit is structural." Currently unsupported. |
| 2 | **Re-run `p1_verify.py` on an A100/H100** via SB-AWS | <1 h | Removes the single-card liability; converts the headline into a cross-architecture claim |
| 3 | **Add `monarch b=8` to `ShortConv` + the arm builder**, with an iso-cost test | 0 GPU (code) | The missing third arm; a test asserting 262,144 == 262,144 == 262,144 |
| 4 | **MQAR arm comparison, 4 arms × 10 seeds** at the calibrated `N512_D64` point | ~3.5 L40S-h | First real quality signal, one afternoon |
| 5 | **Batch sweep of the decode bench (B=1,4,16,64)** | ~0.5 h | Establishes where grouped's 15.3% win survives — currently a batch-1-only claim |
| 6 | Cheap from-scratch: 4 arms × 5 seeds, d=768, 1.8B tokens, **with per-arm LR sweep or self-guided training** (§4.4, §5.2) | ~300 L40S-h / ~3 days at 4 concurrent | The actual test — but only worth running after 1–5 |
| 7 | Full 350M × 8 seeds on 8×A100 | ~1,075 A100-h / ~5.6 days | Only if 6 shows a signal, or if a specific scale claim is needed |

**Items 1–5 total under 6 GPU-hours and would materially change what the paper says.** Item 6 should not
be launched before item 1 (which may invalidate its premise) and item 3 (without which it is a two-arm
re-run of published work).

---

## 7. Summary of errors and unsupported claims found in the existing analysis

| # | claim in the docs | verdict | where |
|---|---|---|---|
| 1 | latency numbers, 8.2% / 15.3% / −36.0% | ✅ **correct**, replicated, tiny spread | §1.1 |
| 2 | `lowrank r=128` and `grouped g=4` are exactly iso-cost | ✅ **correct** — both 262,144/gate; fused changes nothing | §1.2 |
| 3 | energy 0.929 vs 0.130; grouped ≡ random mask | ✅ **numerically correct** (mean diff −0.00006) | §1.3 |
| 4 | "the gap is 80 points" / 7× | ⚠️ **metric artifact** — correct comparison is 7.2% vs ~63% relative error, an 8.8× gap | Finding 2 |
| 5 | "block structure buys nothing over random sparsity" | ⚠️ **near-tautological** — the `p`/`p²` decomposition forces equality absent block-alignment | Finding 1 |
| 6 | "channel ordering doesn't rescue it → the deficit is structural" | ❌ **NOT SUPPORTED** — 3 uniform draws from ~10⁶¹⁰ partitions | §1.4d |
| 7 | grouped's score is a fair measurement of block-diagonal | ❌ **NO** — low-rank got the Eckart–Young optimum, grouped got a naive mask with no OBS reconstruction | Finding 3 |
| 8 | grouped's 15.3% win is a bandwidth win | ❌ **NO** — g=2 (20 MiB) and g=4 (10 MiB) differ by 0.17%; bytes are near-irrelevant at batch 1 | §1.1 |
| 9 | "fused"/"shared" down-projection | ⚠️ **misnomer** — nothing is shared; a true shared bottleneck costs `3dr`, a different arm | §1.2 |
| 10 | the low-rank-vs-grouped tension is a better result than either winning | 🟡 **half right** — it is the right frame, but the decode negative result is the actual contribution, and a third arm is needed for a resolution | §6.1 |
| 11 | (implicit) this comparison is unexplored | ❌ **NO** — [6] Wei et al., NeurIPS 2024, ran LowRank vs BlockShuffle vs BlockDense from scratch at 110M–1.3B, iso-param. LowRank won. | §5 |
| 12 | ShuffleNet-style channel shuffle is an untested idea here | ⚠️ **published as BlockShuffle** [6], and it *lost*; also the LIV gates are **parallel**, so the serial-shuffle analogy does not apply | §1.4a-c, §5.1 |

**Two things the docs get exactly right and should keep:** the Eckart–Young caveat (it is the correct
caveat, and it is stronger than written), and the decision to treat the energy result as "a strong
prior, not a verdict."

*End of document.*
