# 06 — P2 / P3 reassessment (adversarial)

**Author:** reassessment teammate #6. **Started:** 2026-08-01.
**Constraint honored:** no code executed on the local Mac. Everything below is reading, grepping,
web verification, and arithmetic done by hand.

**Label key:** MEASURED (someone ran it and I can point at the artifact) / INFERRED (derived by
arithmetic or logic from a MEASURED or published fact) / ASSUMED (neither — a belief).

**Status: COMPLETE.**

## TL;DR

**Cut both P2 and P3.** ≈650-800 A100-hours saved (~25% of the program).

- **P2 is cut on novelty, not just economics.** arXiv **2606.06467** (MSRA, 4 Jun 2026) already ran
  RULER × 12 subtasks × {16K, 32K} on from-scratch 4B KV-sharing models vs a matched non-sharing
  control. The docs' headline ("no cross-layer-sharing paper reports retrieval") is **now false**,
  and the result went the *opposite* way from the capstone's motivating worry (**+6.1 RULER avg at
  32K**). Separately, `A-fewer3` dominates CLA2 on every quantitative axis (verified from geometry;
  my independent param count reproduces the arm builder's 357,638,528 exactly).
- **P3 is cut on evidence I generated.** A **2-second, 0-GPU-hour** read of the released LFM2-350M
  conv weights settles the width question: the boundary tap holds **1.4%** of median per-channel
  energy (random-init control 30.0%), only **2.08%** of channels have the oldest tap largest
  (control 34.2%), decay ratio **0.083** — 4× steeper than I pre-registered. **k=3 is nowhere near
  binding, measured inside Liquid's double gate**, which was P3's last defense. Bonus: the conv
  **de-activates monotonically with depth** (98.2% off-current energy at layer 0 → **2.4%** at
  layer 15) and layers 0-1 are **pure lag-1 delay lines**. Both findings are new and publishable.
- **P3's steelman is also dead:** the RepVGG-style reparameterization question the docs propose as
  the salvage has **already been run** — Tian et al. Table 7 is exactly that experiment and it lost
  (12.79 → 13.28).
- **Citation audit: no hallucinated arXiv IDs.** Both suspicious 2026 IDs are real and accurate.
  But the docs' P2 literature sweep is ~8 months stale and misses 7 relevant papers.

---

Everything below is the working; predictions in §4 were written to disk **before** the measurement
in §5 was run.

---

## 0. Citation audit (running) — do the cited arXiv IDs exist?

| ID | claimed in docs to be | resolves? | content matches claim? |
|---|---|---|---|
| **2607.18413** | Tian et al., *Convolution for Large Language Models*, PKU/Huawei/Tsinghua; Qwen3-1.7B; residual depthwise conv; k=3 best; **multi-branch mixed widths 12.79 → 13.28 (worse)** | **YES — real.** Submitted 20 Jul 2026, cs.CL, 12 pp, 5 figs. Authors: Yuchuan Tian, Yingte Shu, Wei He, Shuo Zhang, Tianchen Zhao, Chao Xu, Xinghao Chen, Yunhe Wang, Hanting Chen, Yu Wang. | Abstract confirms the *shape* of the claim: 17-position macro ablation, best = conv on projected Q/K/V pre-attention; micro-analysis prefers **residual depthwise conv, k=3, no norm, no activation**; Qwen3, <0.01% params. **Numeric table (12.79 → 13.28) checked separately below.** |
| **2606.03825** | Sieberling et al., *Dynamic Short Convolutions Improve Transformers*; 300M/15B; width flat past k=3 (18.42/18.17/**18.08**/18.10/18.09/18.10) | **YES — real.** Submitted 2 Jun 2026, cs.LG/cs.CL. Authors: Oliver Sieberling, Bharat Runwal, Rameswar Panda, Yoon Kim. | **Table 3a reproduces the docs' numbers EXACTLY**: W=1 18.42 / W=2 18.17 / W=3 **18.08** / W=4 18.10 / W=5 18.09 / W=6 18.10, at 300M / 15B tokens / Nemotron-CC, dynamic conv on Q+K+V, R=16. Stated conclusion verbatim: "3 or 4 is generally the sweet spot." **Rank sweep also confirmed**: R=4 18.26 / R=8 18.19 / R=16 18.10 / R=32 18.04 / R=64 17.87 / R=128 17.85 → R=16→128 buys **0.25 ppl**, matching the docs' claim to 2 dp. No-conv baseline 19.12. |
| **2405.12981** | Brandon et al., CLA | **YES — real.** *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention*, Brandon, Mishra, Nrusimha, Panda, Ragan-Kelley. | Abstract mentions **no** named benchmark at all, and **no** needle/passkey/RULER/MQAR. Consistent with the docs' "reports no retrieval" claim (full-text check below). |
| 2411.13676 (Hymba) | prior art for KV sharing | *pending* | *pending* |
| 2305.13245 (GQA) | Appendix A: MQA-from-scratch loss spikes | *pending* | *pending* |

(Table filled in as each fetch lands. Any ID that does not resolve is called out in bold.)

### 0.1 ⚠️ THE BIGGEST SINGLE FINDING OF THIS REVIEW — the docs mis-describe Tian's Table 7, and the true description KILLS P3's steelman

The design doc (§5.3 item 0b, line ~991) says:

> "They then tried **multi-branch mixed kernel sizes: 12.79 → 13.28**, i.e. *worse than a single k=3.*"

That is understated in the one way that matters. **Tian et al.'s multi-branch experiment is
Table 7, and its title/purpose is REPARAMETERIZATION** — branches with different kernel sizes are
trained separately and then **merged into one equivalent convolution at inference** (MEASURED, from
the paper's own HTML):

| Table 7 setting | mean loss | PPL | params (M) |
|---|---:|---:|---:|
| No reparameterization (single k=3) | 2.4795 | **12.79** | 1721.03 |
| + kernel-1 branch | 2.5029 | **13.28** | 1721.26 |
| + kernel-1 and kernel-2 branches | 2.5048 | **13.28** | 1721.61 |

Read what that is. A k=1 branch plus a k=3 branch, summed and fused, **is a k=3 kernel** — identical
function class, different training-time parameterization, fused away at inference. That is *precisely*
the RepVGG-style structural-reparameterization question the design doc proposes at line 1031 as
"a real, cheap, novel question the proposal does not currently contain":

> "Does the 4-branch *training-time* parameterization of a 15-tap kernel optimize better than a
> directly-trained 15-tap kernel, given they are the same function class?"

**It has been run, in a language model, on the same primitive (depthwise conv in the QKV path), and
it lost by 0.49 ppl (12.79 → 13.28) — and adding a second branch did not recover it (13.28).**
The docs' framing of Table 7 as merely "mixed widths" hides the fact that it is the fusion/
reparameterization experiment specifically. The steelman is not novel and the one existing
datapoint is negative.

Caveats kept honest: (i) Tian's conv is **ungated residual** `Y = X + Conv(X)` — Table 2 shows
conv+shortcut 12.79 beats plain conv 13.05, and Table 6 shows *every* activation they tried (SiLU
13.06, LeakyReLU 12.96, sigmoid 12.94) made it worse. So the "LFM2's double gate changes the
landscape" caveat still technically stands. (ii) They report one seed's worth of numbers with no
variance, and the authors themselves say the ablation "doesn't pin down the cause." (iii) They
merged *only* to reach the same k=3 function class; nobody has tested branch-reparameterization of
a genuinely *wider* (k=15) kernel. But the direction of the only evidence is negative, and Tian's
Table 6 result that adding *any* nonlinearity to the conv path hurts is mild evidence against the
"gates change everything" escape hatch too.


---

## 1. The `A-fewer3` arithmetic — verified independently from geometry

**Geometry, MEASURED** from `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/01_lfm2_architecture.md`
(lines 1216-1218, config table across 8 released checkpoints; corroborated at lines 280-282):

- LFM2-350M: `d = 1024`, `num_attention_heads = 16`, **`num_key_value_heads = 8`**, **`head_dim = 64`**
  (derived, = 64 for *every* released LFM2 dense and MoE model).
- Attention layers at `[2, 5, 8, 10, 12, 14]` → **6** of 16.
- `head_dim` is fixed at 64 and `num_key_value_heads` is fixed at 8 **at every scale**, which is why
  KV bytes/token is scale-invariant.

**Per-layer KV, INFERRED (pure arithmetic):**

```
KV width per layer  = 2 (K and V) × hkv × head_dim = 2 × 8 × 64 = 1024 values
KV bytes per layer  = 1024 × 2 B (bf16)            = 2,048 B = 2 KiB / token / layer
L0 resident KV      = 6 layers × 2 KiB             = 12 KiB / token      ✓ matches the doc's 12 KiB
at 32K context      = 12 KiB × 32,768              = 384 MiB             ✓ matches
```

**The three-way comparison, INFERRED:**

| | resident KV B/token | KV **read** traffic per decode step (per token of ctx) | attn-score FLOPs | params |
|---|---:|---:|---:|---:|
| `L0` (stock, 6 attn) | 12 KiB | 6 × 2 KiB = **12 KiB** | 1.000× | 354,483,968 |
| **P2 = CLA2** (3 banks ⇒ 6 layers) | **6 KiB** (−50%) | 6 layers still each read a 2 KiB bank = **12 KiB** (−0%) | **1.000×** (unchanged) | ≈ L0 minus 3×(k_proj+v_proj) |
| **`A-fewer3`** (3 attn layers) | **6 KiB** (−50%) | 3 × 2 KiB = **6 KiB** (**−50%**) | **0.5×** | 357,638,528 |

*(§2 sits below §1.5, which is the P2 novelty audit.)*

**Verdict on the arithmetic: the docs are right, and I reproduce their arm-builder number to the
byte.** Independent check of `A-fewer3`'s parameter count from first principles:

```
ShortConv mixer   = 4d² + kd = 4(1024²) + 3(1024) = 4,194,304 + 3,072 = 4,197,376
GQA attn layer    = q(d×1024) + k(d×512) + v(d×512) + o(1024×d) + qk_norm(2×64)
                  = 1,048,576 + 524,288 + 524,288 + 1,048,576 + 128 = 3,145,856
Δ per swapped layer = 4,197,376 − 3,145,856 = 1,051,520
A-fewer3 = 354,483,968 + 3 × 1,051,520 = 354,483,968 + 3,154,560 = 357,638,528   ✓ EXACT
```

That is an exact match to `liv_arms.py`'s reported 357,638,528 — including the 384-byte per-head
QK-norm term the HANDOFF says was the last gap to close. So the arm ledger is trustworthy here.

**`A-fewer3` dominates CLA2 on every quantitative axis** (INFERRED, but the inference is a one-liner):
same resident capacity, **half** the read bandwidth, **half** the attention-score FLOPs, and
0.717× FLOPs/token at 32K vs CLA2's ~1.000×. CLA2 wins on exactly one non-quantitative axis: it
preserves 6 layers' worth of *attention receptive field* — every one of the 16 blocks still sits at
the same distance from a global-mixing layer. `A-fewer3` opens a 5-block gap (e.g. layers 3-7 with
attention only at 2 and 8 becomes worse). That is the whole of P2's remaining case, and §2 below
asks whether it is worth 2+ GPU-days.

---

## 1.5 P2's novelty gap — "no cross-layer-sharing paper reports needle/passkey/MQAR" — IS NOW FALSE

The docs (design doc line 923-927) call this "confirmed exhaustively," checked "across CLA, the NAACL
2025 systematic study, LCKV, and SwiftKV." That check was done against the pre-mid-2025 literature.
**It has been overtaken.** I swept arXiv (`export.arxiv.org` API, `all:"cross-layer" AND all:"KV
cache"`, 33 hits, newest first) and the sharing line now runs:

`CLA (2405.12981)` → `systematic study (2410.14442, NAACL 2025)` → `CLLA (2410.15252)` →
`FusedKV (2512.03870)` → `YOCO++ (2604.13556)` → `Stochastic KV Routing (2604.22782)` →
`CLSA / You Only Index Once (2606.06467)`, plus a post-training branch
(`xKV 2503.18893`, `CommonKV 2508.16134`, `Krul 2507.08045`, `DepthWeave-KV 2607.06523`,
`XQuant 2510.11236`).

**The killer: arXiv 2606.06467 (Sun, Zhang, Dong, Wang, Wei — MSRA — 4 Jun 2026), *You Only Index
Once: Cross-Layer Sparse Attention with Shared Routing*, evaluates RULER at 16K and 32K, all 12
subtasks, from-scratch 4B models, against a non-sharing Transformer of matched geometry.** MEASURED,
from the paper's own HTML:

| ctx | model | S1 | S2 | S3 | MK1 | MK2 | MK3 | MQ | MV | QH | QS | CWE | FWE | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| 16K | Transformer (no sharing) | 100.0 | 99.8 | 98.4 | 88.2 | 71.4 | 14.4 | 85.7 | 85.6 | 28.8 | 33.2 | 15.6 | 52.1 | **64.4** |
| 16K | YOCO (Dense, KV-sharing) | 100.0 | 99.8 | 96.4 | 69.4 | 91.6 | 61.2 | 45.8 | 49.3 | 30.8 | 31.4 | 9.4 | 67.0 | 62.7 |
| 16K | YOCO (CLSA) | 100.0 | 100.0 | 98.4 | 70.4 | 92.4 | 58.4 | 53.0 | 47.2 | 31.2 | 32.7 | 9.8 | 61.6 | 62.9 |
| 32K | Transformer (no sharing) | 100.0 | 98.8 | 83.4 | 57.0 | 38.8 | 0.8 | 45.6 | 42.6 | 21.2 | 20.2 | 1.8 | 43.8 | 46.2 |
| 32K | YOCO (Dense, KV-sharing) | 100.0 | 90.2 | 74.8 | 53.2 | 84.0 | 43.6 | 27.0 | 29.0 | 30.6 | 30.6 | 4.6 | 60.3 | 52.3 |
| 32K | YOCO (CLSA) | 100.0 | 93.6 | 83.2 | 58.4 | 88.8 | 38.0 | 31.6 | 29.8 | 29.2 | 29.2 | 5.1 | 50.2 | **53.1** |

Setup: 4B, hidden 2560, FFN 7680, 32 layers, 20 heads, **4 KV heads, head_dim 128**, no weight tying,
**pretrained from scratch** (8K × 125k steps @ 3e-4, then 32K × 10k steps @ 3e-5, 8M tok/batch).

**This is exactly the experiment the capstone proposes to be first at, and it is more thorough than
the capstone's plan** — 12 RULER subtasks × 2 context lengths × from-scratch training × a matched
non-sharing control, at 4B rather than 350M. It even answers the question in the interesting
direction: KV sharing is **−1.7 avg at 16K but +6.1 avg at 32K** (52.3 vs 46.2), and the paper says
the gains come "mainly from the harder multi-needle settings" (MK2 38.8→84.0, MK3 0.8→43.6). So the
prior implied by the docs — that retrieval is where sharing quietly breaks — is *contradicted at
scale* by the first paper that actually measured it.

Two other 2026 entries also close the gap, in the post-training branch:
- **`DepthWeave-KV` (2607.06523, 7 Jul 2026)** — "Token-Adaptive Cross-Layer Residual Factorization,"
  shares low-rank bases across neighboring layers, evaluated on **LongBench, Needle-in-a-Haystack,
  and L-Eval**.
- **`CommonKV` (2508.16134, 22 Aug 2025)** — cross-layer parameter sharing, evaluated on **LongBench
  and RULER**.

**Caveats that keep some of the gap alive (be precise about what survives):**
1. **MQAR specifically is still unreported.** Nothing in the 33-paper sweep names MQAR or passkey.
   RULER's needle subtasks are the closest analogue, and CLSA covers them. So "no one reports
   *MQAR*" is technically still true — but it is a much weaker claim than "no one reports retrieval,"
   and RULER is the benchmark a reviewer would ask for anyway.
2. **CLSA is YOCO-style (global self-decoder → cross-decoder), not CLA-style pairwise banks.** YOCO
   shares one KV bank across *all* upper layers; CLA2 pairs adjacent layers. Different sharing
   topology, and the docs are right that no one has done CLA-style pairing *across an intervening
   conv block in a sequentially interleaved hybrid*.
3. **Nobody has done it at 350M** or in an LFM2-shaped hybrid.
4. `Stochastic KV Routing` (2604.22782) is architectural and trains the sharing in, but its abstract
   names **no benchmark at all** — so it does not close the gap.

**Net effect on P2's pitch.** The headline the design doc chose — "Make [the evaluation gap] the
headline of the P2 arm" (line 941) — is no longer available as stated. What is left is a scoped
version: *CLA-style pairwise sharing between full-attention layers separated by conv blocks, at
350M, under RULER/MQAR.* That is a real gap, but it is a gap of *configuration*, not of *question*,
and the question has now been answered by a stronger paper in the direction opposite to the
capstone's motivating worry. A reviewer who knows 2606.06467 will ask why 350M/CLA-pairwise is
expected to behave differently from 4B/YOCO, and the honest answer is "we don't have a mechanism."

---

## 3. P3 arithmetic — the two corrections, both verified by hand

### 3.1 Dilation coverage: {1,2,4,7} reaches 7 of 15 lags — **CONFIRMED**

A causal 3-tap kernel at dilation `d` touches lags `{0, d, 2d}`. So:

| branch | dilation | lags touched | max lag |
|---|---:|---|---:|
| b1 | 1 | {0, 1, 2} | 2 |
| b2 | 2 | {0, 2, 4} | 4 |
| b3 | 4 | {0, 4, 8} | 8 |
| b4 | 7 | {0, 7, 14} | **14** |

Union = **{0, 1, 2, 4, 7, 8, 14} = 7 distinct lags.**
Missing = **{3, 5, 6, 9, 10, 11, 12, 13} = 8 lags, structurally unreachable.**
Multiplicities: lag 0 ×4, lag 2 ×2, lag 4 ×2, lags 1/7/8/14 ×1 each. Sum = 4+2+2+1+1+1+1 = **12 taps** ✓.
So **12 parameters/channel buy 7 degrees of freedom**; 5 parameters are pure redundancy.
A dense k=15 kernel gets 15 DOF for 15 parameters. **The docs' claim is exactly right** (and the
`04_multiscale_routing.md` §5.2 table at line 331 states it correctly). INFERRED, trivially checkable.

**The consequence the docs state but do not emphasize enough:** the proposed dilated variant is
**strictly dominated** by a dense k=15 kernel — same state, more DOF, 3 more parameters, fusible,
and (per §5.3 item 2) buildable today with FLA's Triton conv backend which has no width limit.
There is no configuration in which the dilated branch set is the right design choice on the merits.
Its only advantage is FLOPs, on an op that is **1.0% of decode time** (MEASURED, ONNX per-op profile
in `HANDOFF.md:88-96`), so the advantage rounds to zero.

### 3.2 "5× state growth, not 7×" — **CONFIRMED, and the byte figures check out**

`conv_L_cache` is the cache depth AND the kernel size — MEASURED from the released config
(`/scratch/users/ericrcwu/liv/ckpt/config.json`: `"conv_L_cache": 3`, `"conv_bias": false`,
`"conv_dim": 1024`), and from `01_lfm2_architecture.md:202` — conv state shape is
**`(batch, d, k)`**, statically allocated. Cache depth is set by **max lag + 1**, not tap count.

```
k=3  : 1024 ch × 3  slots × 2 B (bf16) = 6,144 B/layer × 10 LIV layers =  61,440 B =  60 KiB
k=15 : 1024 ch × 15 slots × 2 B (bf16) = 30,720 B/layer × 10 LIV layers = 307,200 B = 300 KiB
ratio = 15/3 = 5.0×                                                       ✓ 5×, not 7×
```

**Confirmed: 60 KiB → 300 KiB, exactly 5×.** The retracted "7×" came from a `k−1` vs `k` convention
error. And the docs are right that the *dilated 12-tap* variant has **identical** state to the dense
15-tap variant (both max-lag 14 → both need 15 slots), so dilation saves parameters and FLOPs,
**never state**.

Against the KV cache: at 32K, KV = 384 MiB and conv state = 300 KiB = **0.076%**. So the conv state
is genuinely negligible on GPU. The docs' point that it cuts against LFM2's embedded-CPU design
target is fair but soft — 300 KiB still fits in most L2s.

---

## 4. PRE-REGISTERED PREDICTION for the tap-energy measurement (written BEFORE seeing any number)

This is item 4 of my assignment: the single cheapest thing that would settle P3's width question.
I am writing the prediction first so the result cannot be read post-hoc.

### 4.1 What the measurement is

The entire kernel-width question is a **30,720-scalar read** off a checkpoint that is already on
FarmShare (`/scratch/users/ericrcwu/liv/ckpt/model.safetensors`, 709 MB, MEASURED). Per LIV layer:
`model.layers.{0,1,3,4,6,7,9,11,13,15}.conv.conv.weight`, shape `(1024, 1, 3)`. Ten layers × 1024
channels × 3 taps. **Zero GPU-hours, zero training, seconds of CPU on a login node.**

Tap indexing (INFERRED from `nn.Conv1d` semantics + LFM2's `padding=k-1` then left-slice, confirmed
by `01_lfm2_architecture.md:1174`): index **0 = oldest lag (t−2)**, index 1 = t−1, index **2 =
current token (t)**.

### 4.2 The theory, and why the profile is diagnostic

If a k=3 kernel is **binding** — the model wants more span and the architecture is refusing — the
learned taps cannot decay to the boundary, because the optimizer would be pushing mass *out* of the
window and the boundary tap is the last place it can put it. The signature is a **flat or
boundary-heavy** profile: `|w_{t-2}| ≳ |w_{t-1}|`, and a large fraction of channels whose largest
tap is the oldest one.

If k=3 is **not binding** — 3 taps is already more than the signal needs — the profile decays
monotonically toward the boundary and the oldest tap is nearly dead. The optimizer has spare capacity
it declined to use, which is a *direct* observation that lag 3+ carries nothing worth having.

### 4.3 A quantitative bridge from Sieberling's width sweep (this is the part that makes it a real prediction)

Sieberling's marginal ppl gain per added lag (MEASURED, Table 3a, verified above):

| added lag | ppl gain |
|---|---:|
| lag 1 (W=1→2) | **0.25** |
| lag 2 (W=2→3) | **0.09** |
| lag 3 (W=3→4) | **−0.02** |
| lag 4 (W=4→5) | +0.01 |
| lag 5 (W=5→6) | −0.01 |

Ratio lag2/lag1 = 0.09/0.25 = **0.36**. Under the assumption that a lag's marginal ppl gain is
monotone in the variance that lag's tap explains, the *energy* profile should fall at a comparable
geometric rate. **So I predict, before looking:**

| quantity | prediction | reasoning |
|---|---|---|
| Energy ordering | **E(t) > E(t−1) > E(t−2)**, strictly monotone | decay hypothesis |
| E(t−2)/E(t−1) | **0.2 – 0.5**, centred ≈ 0.36 | Sieberling ratio |
| Fraction of channels whose **argmax tap is the oldest** | **< 25%** | if k=3 were binding this would approach or exceed 1/3 (chance) and likely exceed it |
| Fraction with \|w_{t−2}\| > \|w_t\| | **< 35%** | same |
| Fraction with \|w_{t−2}\| > 0.9·max\|tap\| ("boundary saturated") | **< 15%** | a saturated boundary is the specific signature of a binding constraint |

### 4.4 The decision rule, pre-committed

- **k=3 NOT binding** (predicted): monotone decay, boundary-argmax fraction < 25%, boundary-saturated
  fraction < 15%. ⇒ **The width arms `k5/k9/k15` will find nothing**, and P3's width branch should be
  cut. Sieberling's flat curve is confirmed *inside Liquid's gated block by the trained weights
  themselves* — which is a publishable one-paragraph result at zero cost.
- **k=3 BINDING** (would surprise me): flat or increasing energy toward the boundary, or
  boundary-argmax fraction ≳ 1/3. ⇒ the gate really does change the landscape, Sieberling does not
  transfer, and `k5/k9/k15` is worth the ~2 GPU-days. **This is the only evidence that would revive P3.**
- **Ambiguous** (monotone but shallow, ratio > 0.6): run **`k5` only** as a single cheap rung, not the
  full `k5/k9/k15` ladder.

### 4.5 Measurement-spec refinements that matter (methodology, not decoration)

1. **Normalize per channel before pooling.** Raw pooled `Σ w²` per tap is dominated by high-magnitude
   channels, which are the ones with large `out_proj` down-weighting. Report the per-channel
   normalized profile `w_c² / Σ_j w_{c,j}²` and take the **median across channels**, plus the
   fraction-based statistics (argmax, ratio) which are scale-free by construction.
2. **Report per-layer, not just pooled.** LFM2's LIV layers sit at depths
   {0,1,3,4,6,7,9,11,13,15}; the two early layers (0,1) plausibly do genuine local n-gram work while
   deep layers may be near-vestigial. A pooled average could hide a real signal in layers 0-1.
   **If early layers are boundary-saturated and deep ones are not, the correct arm is "widen the
   first two LIV layers only" — which nobody has proposed and which is much cheaper than a global
   width change.**
3. **Guard against a dead-conv reading.** If the profile is ≈[0, 0, 1] the conv is *vestigial* — the
   block has degenerated into a pure gated pointwise op. That is a different (and more interesting)
   finding than "k=3 is enough," and it argues for a `k=1` **ablation** arm rather than a `k=15` arm.
4. **This is correlational, not causal.** A trained-weight profile tells you what the optimizer did
   under k=3, not what it would do under k=15. A model can decay to the boundary and still benefit
   from more span if the wider window changes the optimization path. So the measurement can
   **strongly de-prioritize** the width arms; it cannot logically prove them worthless. State it that
   way. (Counterweight: it agrees with two independent published sweeps, so the combined evidence is
   much stronger than the probe alone.)
5. **Cross-check the null against a control.** Compute the same statistic on a **randomly initialized**
   k=3 conv (uniform init → flat expected energy, boundary-argmax ≈ 1/3). The gap between the trained
   profile and 1/3 is the effect size. Without that control, "22% boundary-argmax" is not obviously
   low.

**Cost of the whole thing: one login-node CPU invocation, ~2 seconds, 0 GPU-hours.** It is the
highest information-per-dollar item anywhere in the P3 plan by several orders of magnitude, and it
should run before a single GPU is allocated to `k5/k9/k15`.

---

## 5. THE MEASUREMENT — run, and it settles P3's width question decisively

**MEASURED, 2026-08-01, FarmShare login node (CPU only, ~2 s, 0 GPU-hours).** Read directly out of
the safetensors header bytes of the released `LiquidAI/LFM2-350M` checkpoint at
`/scratch/users/ericrcwu/liv/ckpt/model.safetensors`. Scripts:
`/scratch/users/ericrcwu/liv/tapread.py` (raw profile) and
`/scratch/users/ericrcwu/liv/tapread2.py` (per-channel-normalized + random-init control + vestigial
check — written by me to the spec in §4.5). Ten LIV layers × 1024 channels × 3 taps = 30,720 scalars.
Tap index 0 = oldest (t−2), 2 = current token (t).

### 5.1 Raw per-layer profile

| layer | mean\|w(t−2)\| | mean\|w(t−1)\| | mean\|w(t)\| | E(t−2)% | E(t−1)% | E(t)% | frac \|t−2\|>\|t\| | boundary-argmax |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0218 | **0.1502** | 0.0109 | 5.25 | **92.98** | 1.77 | 0.730 | 0.059 |
| 1 | 0.0318 | **0.1321** | 0.0228 | 6.49 | **88.74** | 4.77 | 0.675 | 0.042 |
| 3 | 0.0412 | 0.0809 | 0.0900 | 11.37 | 35.46 | 53.16 | 0.304 | 0.061 |
| 4 | 0.0349 | 0.0777 | 0.1044 | 7.17 | 30.51 | 62.32 | 0.222 | 0.017 |
| 6 | 0.0232 | 0.0474 | 0.1316 | 3.23 | 12.41 | 84.36 | 0.097 | 0.003 |
| 7 | 0.0260 | 0.0554 | 0.1292 | 4.13 | 15.79 | 80.07 | 0.095 | 0.005 |
| 9 | 0.0310 | 0.0610 | 0.1258 | 5.51 | 17.87 | 76.62 | 0.123 | 0.014 |
| 11 | 0.0157 | 0.0449 | 0.1360 | 1.82 | 11.70 | 86.48 | 0.055 | 0.009 |
| 13 | 0.0081 | 0.0286 | 0.1500 | 0.62 | 6.35 | 93.03 | 0.052 | 0.000 |
| 15 | 0.0038 | 0.0133 | **0.1797** | **0.11** | 2.27 | **97.62** | 0.004 | 0.000 |

Pooled energy by tap (oldest → current): **[4.26%, 29.62%, 66.12%]**.

### 5.2 Per-channel-normalized median (the scale-free version, spec item 1) + the random control

| | med E(t−2) | med E(t−1) | med E(t) | boundary-argmax | frac \|t−2\|>0.9·max |
|---|---:|---:|---:|---:|---:|
| **Trained LFM2-350M** | **0.0143** | 0.1721 | 0.7439 | **0.0208** | **0.0246** |
| Random-init control (U(±1/√3), n=10240) | 0.3001 | 0.3002 | 0.2968 | 0.3421 | 0.4076 |

### 5.3 Scored against the predictions I pre-registered in §4.3 — **5 for 5, all by large margins**

| pre-registered quantity | predicted | **MEASURED** | verdict |
|---|---|---:|---|
| Energy ordering | E(t) > E(t−1) > E(t−2) monotone | 0.744 > 0.172 > 0.014 | ✅ |
| E(t−2)/E(t−1) | 0.2 – 0.5, centred 0.36 | **0.083** | ✅ **decays far faster than predicted** |
| boundary-argmax frac | < 25% | **2.08%** (control 34.2%) | ✅ **16× below chance** |
| frac \|t−2\| > \|t\| | < 35% | 23.6% | ✅ |
| boundary-saturated frac | < 15% | **2.46%** (control 40.8%) | ✅ **17× below control** |

**Effect size against the control is enormous.** A randomly initialized k=3 conv has 34.2% of
channels with the oldest tap largest; the trained model has **2.08%**. The optimizer had three taps
available and drove the boundary tap to near-zero in 98% of channels. **k=3 is not binding — it is
not even close to binding.** The model does not want more span; it is actively discarding the span it
already has.

Note the measured decay (ratio 0.083) is **4× steeper than the Sieberling-derived prediction (0.36)**.
That is the strongest possible version of the result: even the flat published width curve *understates*
how uninterested this architecture is in longer lags.

### 5.4 TWO GENUINELY NEW FINDINGS the docs do not contain, both from the same 2-second read

**(a) A monotone depth gradient — the conv de-activates with depth.** Off-current energy (spec item 3):

| layer | 0 | 1 | 3 | 4 | 6 | 7 | 9 | 11 | 13 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **% energy NOT on the current token** | **98.2** | **95.2** | 46.8 | 37.7 | 15.6 | 19.9 | 23.4 | 13.5 | 7.0 | **2.4** |

Near-perfectly monotone from 98.2% to 2.4%. **By layer 15 the depthwise conv is 97.6% a scalar
per-channel gain on the current token — it is vestigial.** The LIV block there has degenerated into
a gated pointwise op. This is a first-ever published observation about a shipped LFM2 checkpoint,
it costs nothing, and it directly suggests the arm nobody proposed: **`k=1` on the deep LIV layers**
(strictly cheaper than stock, predicted free) rather than `k=15` anywhere.

**(b) Layers 0 and 1 are pure lag-1 shift operators, not local averagers.** Their median normalized
energy on tap t−1 is **0.9905 and 0.9259** — the current token holds 0.17% and 1.10%. The first two
LIV layers have learned to be an almost pure **one-token delay line**, feeding the previous token's
representation into the residual stream. That is the "previous-token head" function, implemented as
a conv. It is the *only* place in the model where span matters at all — and even there, the span it
uses is **exactly 1**, not 2. Layer 0's oldest tap (t−2) holds 0.57% of median energy.

This kills the last escape hatch. My §4.5 item 2 said: *"if early layers are boundary-saturated and
deep ones are not, the correct arm is widen the first two LIV layers only."* Measured: the early
layers are the ones **most** concentrated (on lag 1), and their t−2 tap is the deadest in the model
relative to their peak. **There is no sub-population anywhere in the checkpoint that wants a wider
kernel.**

### 5.5 The honest limitation, stated plainly

This is **correlational** (spec item 4). It shows what the optimizer did *under* k=3 at 10T tokens;
it does not logically prove a k=15 model trained from scratch would not find a different solution.
But: (i) the margins are 16-17× against control, not marginal; (ii) it agrees with **two independent
published sweeps** on different base models (Sieberling 300M/15B: W=3→6 is −0.02; Tian Qwen3-1.7B:
k=4 is worse than k=3); (iii) it is measured **inside Liquid's actual double gate**, which is the one
thing the docs said the published sweeps could not speak to. That was P3's entire remaining defense
(design doc §5.3 item 0a), and **this measurement removes it.** The gate does not rescue span.
Combined, the evidence is about as strong as pre-training evidence gets without pre-training.

---

## 6. Full citation audit — final

Everything I could resolve. **No hallucinated arXiv IDs found.** The two "suspicious 2026 IDs"
(2607.18413, 2606.03825) are both real, both say what the docs claim, and in one case the docs
*understate* the finding (§0.1).

| ID | title | resolves | docs' claim accurate? |
|---|---|---|---|
| 2607.18413 | Tian et al., *Convolution for Large Language Models* | ✅ 20 Jul 2026 | ✅ table verified; **but docs mis-frame Table 7 — see §0.1** |
| 2606.03825 | Sieberling, Runwal, Panda, Kim, *Dynamic Short Convolutions Improve Transformers* | ✅ 2 Jun 2026 | ✅ width AND rank sweeps reproduce exactly |
| 2405.12981 | Brandon et al., CLA | ✅ 21 May 2024 | ✅ no retrieval benchmark named |
| 2411.13676 | Dong et al., *Hymba* | ✅ 20 Nov 2024 | ✅ roadmap row C→D verified from paper HTML: commonsense 44.56→**45.16**, recall 48.79→**48.04**, throughput 2399.7→**2756.5** tok/s, cache 41.2→**39.4 MB**. Docs' numbers exact. **New detail:** the "recall" column is a **2-task average the paper never names in the table caption** (SWDE + SQuAD-C elsewhere) — so it is even weaker evidence than the docs concede. Hymba **does** run needle-in-a-haystack (Fig. 10, vs Mamba2 and Llama3, to 16K) but **not as a sharing ablation** — so it does not close the gap either. |
| 2410.14442 | Wu, Wu, Tu, *A Systematic Study of Cross-Layer KV Sharing* (NAACL 2025) | ✅ 18 Oct 2024 | ✅ abstract names no benchmark; "language modeling and downstream tasks" only |
| 2606.06467 | Sun, Zhang, Dong, Wang, Wei (MSRA), *You Only Index Once* | ✅ 4 Jun 2026 | ❌ **NOT IN THE DOCS AT ALL — and it is the paper that breaks P2's headline.** See §1.5 |
| 2607.06523 | *DepthWeave-KV* | ✅ 7 Jul 2026 | ❌ not in docs; cross-layer sharing + NIAH + LongBench + L-Eval |
| 2508.16134 | *CommonKV* | ✅ 22 Aug 2025 | ❌ not in docs; cross-layer sharing + RULER + LongBench |
| 2604.22782 | *Stochastic KV Routing* | ✅ 3 Apr 2026 | ❌ not in docs; architectural sharing, but names no benchmark |
| 2604.13556 | *YOCO++* | ✅ 15 Apr 2026 | ❌ not in docs |
| 2512.03870 | *FusedKV* | ✅ 3 Dec 2025 | ❌ not in docs |
| 2503.18893 | *xKV* | ✅ 24 Mar 2025 | ❌ not in docs; post-training, long-context tasks |
| 2305.13245 | GQA | not re-fetched (widely known; Appendix A MQA-from-scratch instability claim not independently re-verified this pass) | ⚠️ **UNVERIFIED this pass** |

**One weakness in the docs' method, worth naming:** the P2 literature sweep was thorough *as of its
date* and is now ~8 months stale on a fast-moving topic. Seven relevant cross-layer-sharing papers
published since are absent, and one of them (2606.06467) directly performs the experiment the
capstone proposes as its contribution. **Any "nobody has done X" claim in this project needs a
re-check dated within weeks of submission, not months.**

---

## 7. VERDICTS

### 7.1 P2 (cross-layer KV sharing) — **CUT**

Not "narrow." **Cut.**

The case is a chain, and every link has now broken:

1. **Latency: dead by construction.** The docs concede this (line 801: "end-to-end decode latency
   ≈0%, any context"). A capstone arm whose headline efficiency metric is *predicted to be zero*
   needs its quality/novelty story to carry the entire weight.
2. **Capacity: strictly dominated by `A-fewer3`.** Verified in §1 from geometry, and my independent
   param count reproduces the arm builder exactly (357,638,528). `A-fewer3` gets the same −50%
   resident KV, **plus** −50% read bandwidth, **plus** 0.5× attention-score FLOPs, **plus** 0.717×
   total FLOPs/token at 32K. CLA cannot touch any of those three. On the efficiency axis P2 is not
   competitive with a change that is *simpler to implement than P2 itself* (delete three layers).
3. **Novelty: broken by arXiv 2606.06467 (§1.5).** The docs' headline — "not one cross-layer-sharing
   paper reports needle/passkey/MQAR, make it the headline of the P2 arm" — was true when written
   and is now false. MSRA ran RULER at 16K and 32K, 12 subtasks, from-scratch 4B, against a matched
   non-sharing control. **And it came out the opposite way from the capstone's motivating worry:**
   sharing is −1.7 avg at 16K but **+6.1 avg at 32K** (52.3 vs 46.2), with the gains concentrated
   in the hard multi-needle subtasks (MK2 38.8→84.0, MK3 0.8→43.6).
4. **The remaining question is not interesting enough to price.** After the above, P2 reduces to:
   *"does CLA-pairwise lose less quality than deleting three attention layers, at 350M?"* The
   literature already answers the direction — Hymba's own all-SWA row shows recall collapsing
   **48.79 → 29.78** when global attention is removed (`03_kv_sharing.md:781-784`), i.e. removing
   attention layers is exactly the thing that destroys retrieval. So the expected result is "yes,
   CLA loses less than `A-fewer3` on recall, and `A-fewer3` wins on everything else." That is a
   predictable trade-off curve, not a finding.

**Is it worth 2+ GPU-days?** No. Stage 3b in the phased plan (design doc line 1380) is P2's
pairing study plus `A-fewer3`, `Q-mqa`, and `SWA` — 4-5 arms. Against the ~2.5-day rank stage
and ~2.5-day confirm stage at 8×A100, P2's share is **≈ 2.5-3 days on 8×A100 ≈ 500-600 A100-hours**,
plus the ncu measurement work in §5.2 of the design doc (lines 879-905) and its four documented
traps. That is 15-20% of the entire program's budget for a predictable answer to a question a
better paper already answered at 4B.

**What replaces it.** Keep exactly one thing from P2 and throw the rest away: **`A-fewer3` as a
first-class arm in the topology study**, not as P2's competitor. It is the cheapest real efficiency
result in the whole project (0.717× FLOPs at 32K, −50% resident KV, −50% read bandwidth, +0.9%
params) and it belongs to the **topology** claim the docs already identify as the one testable
efficiency claim (HANDOFF line 483-486). Reframe: *"how few attention layers does LFM2 actually
need, and what does retrieval cost per layer removed?"* — a ratio ablation Liquid never published,
which is clause 1 of the project's own framing sentence. Same GPU-hours, no dependence on any
literature gap that can close under you.

**If the human insists on keeping something CLA-shaped**, the only version I would defend is a
**zero-training** one: apply CLA-style pairing post-hoc to the released LFM2-350M weights (copy
producer K/V, drop consumer k_proj/v_proj) and measure the recall cliff on the passkey/BABILong
harness teammate #4 is already building. Costs ~0.5 L40S-hours on top of work already planned,
and answers "how much retrieval does forced sharing destroy in a *shipped* hybrid" without a single
pretraining run. Negative-result-safe either way.

### 7.2 P3 (multi-span dilated conv + router) — **CUT**, and the cut is now evidence-backed

Also not "narrow." **Cut — all three rungs: the router, the dilated branches, AND the `k5/k9/k15`
width sweep.**

The design doc's own kill rule (line 1411) says: *"If `k15` doesn't beat `k3`, drop P3."* **§5
answers that question without training anything**, at 16-17× margins against a random-init control:

- Boundary tap holds **1.4%** of median per-channel energy (control: 30.0%).
- **2.08%** of channels have the oldest tap largest (control: 34.2%, chance 1/3).
- **2.46%** are boundary-saturated (control: 40.8%).
- Decay ratio E(t−2)/E(t−1) = **0.083**, four times steeper than the 0.36 I predicted from
  Sieberling's published marginal gains.
- The conv **de-activates monotonically with depth**: 98.2% off-current energy at layer 0 → **2.4%**
  at layer 15. The deep LIV layers' convs are vestigial.
- Layers 0-1 are **pure lag-1 delay lines** (median 0.99 / 0.93 of energy on tap t−1) — the only
  span-using layers in the model use span exactly **1**.

The gate was P3's last defense (design doc §5.3 item 0a: the adverse evidence "was collected in a
DIFFERENT structural slot… the negative results may not transfer"). **This measurement is taken
inside Liquid's actual double gate, on weights trained for 10T tokens, and it agrees with the
ungated literature.** The gate does not rescue span. That defense is now closed empirically rather
than argued about.

And the steelman is gone too: §0.1 shows the RepVGG-style reparameterization question the docs
propose as P3's salvage has **already been run in an LLM on this exact primitive and lost**
(Tian Table 7: 12.79 → 13.28, and a second branch did not recover it).

Meanwhile the cost side never justified it: the conv is **1.0% of measured decode time**, the
dilated variant is **strictly dominated** by a dense k=15 (7 DOF vs 15, identical state — §3.1/3.2),
a router **destroys fusibility** permanently, and RepVGG measured a **41% throughput loss** from
multi-branch fragmentation.

**Is the reparameterization question interesting enough for a capstone?** It *is* a real question
with real prior art (RepVGG, DiracNet, ExpandNets, over-parameterization-helps-optimization). But
for this capstone: no. It requires the width question to have come out the other way — you cannot
ask "does the 4-branch parameterization of a 15-tap kernel train better" when the measurement says
the 15-tap function class contains nothing the 3-tap one lacks. You would be comparing two
parameterizations of a capacity the model demonstrably does not want.

**GPU-hours saved.** Stage 3c (`k5/k9/k15`, design doc line 1381) is 3 arms in the rank stage's
12-arm × 2-seed pool → **≈ 0.6 days on 8×A100 ≈ 120 A100-hours**, plus the "week-1 de-risk"
2-day conv profiling job (line 1075-1078) which is now pointless, plus any router escalation
(which the phased plan gates behind a width win that will not come) — realistically **≈ 150-200
A100-hours plus ~2-3 engineer-days** avoided.

**What replaces it.** Three things, all cheap, all from the same 30 KB read:

1. **Publish the tap-energy analysis as a result.** "We opened Liquid's shipped 350M checkpoint and
   measured where the depthwise conv actually looks. Answer: 66% of its energy is on the current
   token, 2% of channels use the boundary tap, and the conv de-activates monotonically with depth
   until it is a scalar gain by layer 15." **Nobody has published this**, it is a direct empirical
   answer to the kernel-width clause of the project's own framing sentence, and it cost 2 seconds
   of CPU. It is a strictly better use of the P3 slot than a training run.
2. **Flip the arm from `k15` to `k1`.** The measurement predicts a **narrower** kernel is free on
   the deep layers. `k1` on layers {6,7,9,11,13,15} is *cheaper* than stock (no conv state at all on
   those layers, −6 KiB, and one fewer op), and predicted quality-neutral. A win there is a real
   efficiency result on a shipped architecture, in the direction opposite to the brainlift's
   proposal — which is a *better* story, not a worse one, because it is surprising and it is the
   model's own weights telling you.
3. **One `k5` rung as insurance** if the human wants any width evidence from training at all —
   ~40 A100-hours, one rung, not a ladder. I would skip it.

### 7.3 Both should be cut — and that is fine, say so plainly

Yes: **cut both.** Combined savings ≈ **650-800 A100-hours (~25% of the ~3,000-hour program)** and
several engineer-days. This does not leave the capstone empty — it leaves it with a *better* project,
because everything that replaces P2 and P3 is negative-result-safe and needs no from-scratch training:

| was | becomes | cost |
|---|---|---:|
| P2 pairing study + ncu capacity/bandwidth split | **`A-fewer3` inside a topology/ratio ablation** ("how few attention layers does LFM2 need") | same slot, better claim |
| P2 retrieval-gap headline (now closed by 2606.06467) | **Retrieval sweep on the *released* LFM2 checkpoints** — Liquid publishes zero long-context numbers for a model it markets for RAG (teammate #4, §1.4) | ≈ 3.4 L40S-h |
| P3 `k5/k9/k15` + router + conv profiling | **Tap-energy analysis (DONE, §5) + a `k1` narrowing arm** | 0 GPU-h + ~40 A100-h |

The project's own framing sentence (HANDOFF line 158-161) has three clauses — no ratio ablation,
no kernel-width ablation, no recall benchmark. After this reassessment: **the kernel-width clause is
answered (§5, free), the recall clause is answered by inference on released weights (teammate #4,
~3 L40S-h), and only the ratio clause needs GPUs.** Cutting P2 and P3 is what makes that focus
possible.

---

## 8. Loose ends I did not close

- **GQA arXiv 2305.13245 Appendix A** ("MQA from scratch had frequent loss spikes and diverged") —
  not re-verified this pass. It only matters if P2 survives; it does not, so I deprioritized it.
- **Gemma 3n `num_kv_shared_layers=15`** — HF config fetch returned **HTTP 401** (gated repo), so
  I could not verify the value first-hand. The claim is plausible and widely repeated, but treat it
  as **UNVERIFIED** in the write-up rather than citing a number I could not read.
- **Tian Table 7 seed count / variance** — the paper reports single numbers with no error bars. My
  §0.1 conclusion rests on a 0.49 ppl gap that is large relative to typical LM noise but is n=1.
- **The tap measurement is correlational** (§5.5). I have stated the limitation rather than papered
  over it, but the case against P3 rests on it converging with two published sweeps, not on it alone.
- **`A-fewer3` layer placement** — I verified the *arithmetic*, not which 3 of the 6 attention layers
  should be kept. Hymba's "first, middle, last" finding (recall recovers with 3 global layers)
  suggests `[2, 8, 14]`; the arm builder's choice was not checked against that.

**Status: COMPLETE.**
