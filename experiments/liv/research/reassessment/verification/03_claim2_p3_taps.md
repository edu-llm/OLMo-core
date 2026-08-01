# Verification 03 — Claim 2: "conv-tap reads show k=3 nowhere near binding, measured inside the gate"

**Status:** COMPLETE (2026-08-01)
**Verifier:** independent verification agent (no subagents used)
**Target claims:** `06_p2_p3_verdict.md` §4-§5, `04_cheap_experiments.md` (tap replication)

---

## 0. Bottom line

**UNCLEAR — split verdict.** Full detail in §9.

- **Tap orientation: CONFIRMED.** Index 0 = oldest (t−2), index 2 = current (t). Settled by impulse
  test on both the prefill and decode paths, plus source reading. The conclusion does not invert.
  This was the biggest risk and it is retired.
- **Arithmetic: CONFIRMED.** Every number reproduces to the digit across all four checkpoints, via
  my own independent code path.
- **"Measured inside the gate": REFUTED.** It is a weights-only statistic. `in_proj`/`out_proj` are
  never touched, no token is ever run. This is the same error class as the P1 `spectra_v2` episode,
  which the prior team explicitly flagged as owed (C3) in `04` and then implicitly claimed as
  discharged in `06` §5.5. Those two positions are inconsistent.
- **"Nowhere near binding": OVERSTATED but directionally right.** Headline 1.4% is a median; the
  mean is 6.1%, pooled raw is 4.26%, and under plausible AR(1) input correlation the leave-one-out
  contribution of the oldest tap is **14.4%**.
- **"No sub-population wants a wider kernel": REFUTED.** 792/10240 channels above 20% boundary
  energy; 102 fully boundary-saturated; fraction grows with scale.
- **"Independent replication": REFUTED as independence.** Same script lineage, shared bf16 decoder
  and key selector, written 2m40s apart. Numbers real; independence not.
- **Net:** do not reverse the deprioritization of P3, but reverse the *reasoning*. It rests on the
  two published width sweeps, not on this tap read. **C3 (~0.2 GPU-h) is the unrun item that would
  settle the gate question.**

---

## 1. Tap-index orientation (the load-bearing assumption) — **CONFIRMED, prior team is CORRECT**

My own script: `/scratch/users/ericrcwu/liv/verify_taps.py` (written from scratch; does not import
or call `tapread.py` / `tapread2.py` / `tapfreq.py`).

### 1.1 Code reading (`transformers` v4.x `Lfm2ShortConv`)

`/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/.venv-spectra/lib/python3.11/site-packages/transformers/models/lfm2/modeling_lfm2.py:313-320, 402, 378`

```python
self.conv = nn.Conv1d(..., kernel_size=self.L_cache, groups=hidden, bias=False, padding=self.L_cache - 1)
...
conv_out = self.conv(Bx)[..., :seqlen]                                        # prefill, L402
conv_out = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)          # decode,  L378
```

`padding=k-1` on both sides + left-slice `[..., :seqlen]` = causal. `conv_state` is
`[..., -L_cache:]`, i.e. **newest frame last**.

### 1.2 Empirical impulse test (decisive; run on FarmShare, CPU, torch 2.11)

One-hot kernels through the real `nn.Conv1d(padding=k-1)` + left-slice, input `x[t] = t`:

| weight index one-hot | output | implied lag |
|---|---|---|
| 0 | `[0,0,0,1,2,3]` | **t−2 (OLDEST)** |
| 1 | `[0,0,1,2,3,4]` | t−1 |
| 2 | `[0,1,2,3,4,5]` | **t (CURRENT)** |

Decode path, `conv_state = [3(t−2), 4(t−1), 5(current)]`: one-hot idx 0 → 3.0, idx 1 → 4.0,
idx 2 → 5.0. **Prefill and decode agree.**

**VERDICT: index 0 = t−2 = oldest, index 2 = current token. The prior team's orientation is
correct. The conclusion does NOT invert.** This was the biggest single risk and it is retired.

---

## 2. Independent recomputation of the statistics — **numbers REPRODUCE exactly**

Recomputed from raw safetensors bytes (bf16 bit-shift decode) with my own code path.

### 2.1 Structural facts (all confirmed)

- `conv.conv.weight` shape `[1024, 1, 3]`, BF16, **no conv bias** (`conv_bias: false`) — the 3 taps
  per channel are the entire conv parameterization. Confirmed.
- 350M/700M/1.2B: attention layers `{2,5,8,10,12,14}`, LIV layers `{0,1,3,4,6,7,9,11,13,15}`,
  disjoint. Confirmed. 2.6B: 30 layers, attn `{2,5,9,13,17,21,24,27}`, 22 LIV layers. Confirmed.
- All four checkpoints are physically present on FarmShare (709 MB / 1.48 GB / 2.34 GB / 5.14 GB).
  Their cross-scale numbers are **not** unverifiable — I read all four myself.

### 2.2 Headline statistics: mine vs theirs (LFM2-350M)

| statistic | their claim | **my recomputation** | match |
|---|---:|---:|---|
| pooled raw energy [t−2, t−1, t] | 4.26 / 29.62 / 66.12 % | **4.26 / 29.62 / 66.12 %** | ✅ |
| per-channel-normalized MEDIAN energy | 0.0143 / 0.1721 / 0.7439 | **0.0143 / 0.1721 / 0.7439** | ✅ |
| ratio medE(t−2)/medE(t−1) | 0.083 | **0.0833** | ✅ |
| frac argmax == oldest | 0.0208 | **0.0208** | ✅ |
| frac \|oldest\| > \|current\| | 0.2355 | **0.2355** | ✅ |
| frac \|oldest\| > 0.9·max | 0.0246 | **0.0246** | ✅ |
| frac \|oldest\| > 0.5·max | 0.1033 | **0.1033** | ✅ |
| layer 0 med normE(t−1) | 0.9905 | **0.9905** | ✅ |
| layer 1 med normE(t−1) | 0.9259 | **0.9259** | ✅ |
| layer 15 raw E(t) | 97.62 % | **97.62 %** | ✅ |
| layer 0 med normE(t−2) | 0.0057 ("0.57 %") | **0.0057** | ✅ |
| cross-scale oldest-tap E% (350/700/1.2/2.6) | 4.26 / 5.24 / 5.34 / 4.78 | **4.26 / 5.24 / 5.34 / 4.78** | ✅ |

**Arithmetic is clean. Every reported number reproduces to the digit.** No transcription errors,
no wrong tensors, no dtype bug. The dispute, if any, is entirely about *interpretation and method*.

## 3. Method soundness: THE GATE OBJECTION — **the claim "measured inside the gate" is FALSE**

This is the crux, and the prior team gets it wrong.

### 3.1 What they claimed

`06_p2_p3_verdict.md` §5.5:
> "(iii) it is measured **inside Liquid's actual double gate**, which is the one thing the docs said
> the published sweeps could not speak to. That was P3's entire remaining defense (design doc §5.3
> item 0a), and **this measurement removes it.** The gate does not rescue span."

`04_cheap_experiments.md` §4.2 repeats it:
> "The counter-argument in `HANDOFF.md:455-462` ('their negative results may not transfer because
> LFM2's conv sits inside two gates') is now weakened by evidence from inside the gates themselves."

### 3.2 Why it is false

The measurement is `w_{c,j}²` read off `conv.conv.weight`. **It never touches `in_proj` or `out_proj`,
never runs a token, and has no access to any activation.** The gates are `B` and `C`, produced by
`in_proj`:

```python
BCx = self.in_proj(x).transpose(-1,-2); B, C, x = BCx.chunk(3, dim=-2)
Bx  = B * x                      # INPUT gate — multiplies the conv's input
conv_out = self.conv(Bx)[..., :seqlen]
y = C * conv_out                 # OUTPUT gate — multiplies the conv's output
y = self.out_proj(y.transpose(-1,-2))
```

"Inside the gate" in a *topological* sense (the conv weight sits between two gates in the graph) is
being equivocated with "inside the gate" in the *evidential* sense the objection required (the
measured quantity reflects gate behaviour). Only the first is true. The tap read is a **pure
weight-space statistic** — exactly the class of evidence the project already learned is misleading.

### 3.3 This is the P1 `spectra_v2` failure mode, repeated

The project's own record (`01_p1_verdict.md` §2.1, `04_cheap_experiments.md:367-373`) says plain
weight spectra of the LFM2 gates gave a **misleading** answer and only the **activation-weighted**
version was correct: rank-128 retains **45.8%** of plain Frobenius energy but **92.6%** of
activation-weighted energy — a 2× swing that inverted the conclusion.

The prior team **knew this**. `04_cheap_experiments.md` twice flags the owed follow-up:

- line 391: *"C3 — activation-weighted version … **Required before publishing C1**"*
- line 459: *"⚠️ Same caveat … **weight-space, not activation-weighted.** C3 remains owed."*
- line 864: *"**Required before publishing rank 0.** The `spectra_v2` lesson says weight-space alone
  is not enough"*

But `06_p2_p3_verdict.md` §5.5 then **promotes the very same un-activation-weighted read into the
refutation of the gate objection**, and `00_SYNTHESIS.md` carries it forward. **The caveat that C3
is owed and the claim that the gate objection is answered cannot both be true.** The gate objection
is precisely the objection that C3 was going to address, and C3 has not been run.

### 3.4 How much does it actually matter? I ran a partial version.

I cannot run activations (no GPU allocation, and none needed for the verdict), but I did two
weight-space proxies that bound the effect.

**(a) Channel-importance weighting** (script `verify_taps.py` / `verify_taps2.py` part B). Weight each
channel's normalized tap profile by a throughput proxy
`imp_c = ||B_c||²·||x_c||²·||C_c||²·||out_proj[:,c]||²` — i.e. how much the input gate, value path,
output gate, and downstream read all care about channel `c`:

| | normE [t−2, t−1, t] |
|---|---|
| unweighted MEDIAN (their headline) | [0.0143, 0.1721, 0.7439] |
| unweighted MEAN | [0.0613, 0.3580, 0.5807] |
| **importance-weighted MEAN** | [0.0533, 0.3398, 0.6070] |

Importance weighting **moves the answer in their favour**: the channels the gates care about most
use *less* span, not more.

| importance stratum | mean off-current energy | med normE(t−2) | frac normE(t−2)>0.3 |
|---|---:|---:|---:|
| bottom 50% | 0.4616 | 0.0242 | 5.61% |
| top 10% | 0.2350 | 0.0037 | 0.88% |
| top 1% | 0.0661 | 0.0013 | 1.94% |

**This is a real result and it is the strongest single piece of evidence the prior team should have
had and didn't.** It does not close the gate objection (a static norm proxy is not an activation
measurement — the gates are *input-dependent*, so `C` could be large exactly on the tokens where the
history tap matters, which no static norm can see), but it substantially de-risks it.

**(b) Correlated-input correction — this one cuts the OTHER way and is quantitatively large.**
`w²` is a per-tap variance decomposition **only if the conv input is white**. LM residual-stream
activations are strongly autocorrelated across `t`. Under an AR(1) input with autocorrelation ρ, the
output variance is `Σ_ij w_i w_j ρ^|i−j|`, and the honest "what does the oldest tap contribute"
statistic is the **leave-one-out variance drop**, not `w_0²/Σw²`. Median over channels:

| checkpoint | ρ=0 (their implicit assumption) | ρ=0.5 | ρ=0.9 | ρ=0.99 |
|---|---:|---:|---:|---:|
| 350M | **1.43%** | 8.91% | **14.38%** | 15.75% |
| 700M | 1.92% | 9.91% | 16.21% | 17.63% |
| 1.2B | 2.15% | 10.30% | 16.94% | 18.58% |
| 2.6B | 1.09% | 6.60% | 11.09% | 12.22% |

At ρ=0.9 the headline "1.4%" becomes **14.4%**, a **10× change**, and the fraction of channels where
the oldest tap carries >10% of the variance goes from 17% to **60%**. ρ for LM hidden states between
adjacent positions is routinely 0.7-0.95. **The headline number is not robust to the one distributional
assumption it silently makes**, and that assumption is exactly what an activation-weighted measurement
(C3) would have pinned down. Note this does not resurrect P3 by itself — leave-one-out on the *oldest
of three* taps says nothing directly about a hypothetical fourth — but it destroys the "1.4% therefore
obviously nothing" rhetoric.

### 3.5 Verdict on the gate objection

**The tap read does NOT answer the gate objection. It is a weights-only analysis and the project's own
`spectra_v2` precedent says weights-only analyses of this architecture have already misled once.**
The sentence "measured inside Liquid's actual double gate … this measurement removes it" should be
**retracted** and replaced with "measured on the conv weights, with the activation-weighted version
(C3) still owed." My importance-weighting proxy suggests C3 will probably land in their favour, but
"probably" is not what §5.5 asserts.

---

## 4. Correlational vs causal, and whether the decision rule was falsifiable

### 4.1 Was the pre-registered rule near-unfalsifiable?

**Partly, but less than I expected — I tested it.** I simulated what the k=3 tap read *would* look
like for a model whose true optimal kernel is geometric `r^lag` over 15 lags, truncated to k=3
(script `verify_taps2.py` part C):

| true r | k=3 normE [t−2,t−1,t] | ratio E(t−2)/E(t−1) | true mass beyond lag 2 |
|---:|---|---:|---:|
| 0.29 | [0.0065, 0.0771, 0.9164] | 0.084 | **0.06%** |
| 0.5 | [0.0476, 0.1905, 0.7619] | 0.250 | 1.56% |
| 0.7 | [0.1388, 0.2832, 0.5780] | 0.490 | **11.8%** |
| 0.9 | [0.2660, 0.3285, 0.4055] | 0.810 | **51.1%** |
| 1.0 (box filter) | [0.333, 0.333, 0.333] | 1.000 | **80%** |

**The rule is NOT vacuous.** A model that genuinely wanted a wide box/smoothing filter would show
ratio ≈ 1.0 and boundary-argmax ≈ 1/3 and **would have tripped their "k=3 BINDING" branch**. So
"what realistic checkpoint would have failed it?" has an answer: any checkpoint whose convs are
broad smoothers. That is a real and non-trivial pass. **Credit where due — the pre-registration was
methodologically sound and the falsifier existed.**

### 4.2 But the inference from ratio to out-of-window mass is much weaker than presented

The measured pooled ratio is 0.083 → implied `r ≈ 0.289` → implied out-of-window energy ≈ 0.06%. That
looks devastating. **It is an extrapolation from two points to eleven**, and it is extremely sensitive:

| checkpoint | pooled ratio | implied r | note |
|---|---:|---:|---|
| 350M | 0.0833 | 0.289 | |
| 700M | 0.1201 | 0.347 | |
| 1.2B | 0.1434 | 0.379 | **r grows monotonically with scale** |
| 2.6B | 0.1746 | 0.418 | |

`r` is **increasing with model size** across all four released checkpoints (0.289 → 0.418). Extrapolating
that trend is not warranted from n=4, but it is the *opposite* of the direction the "nowhere near
binding" story wants, and the prior team did not report it. And per-channel rather than pooled:

| ckpt | implied per-channel r: p50 / p90 / p99 | implied out-of-window energy: mean / p90 | frac channels >10% OOW |
|---|---|---|---:|
| 350M | 0.375 / 0.785 / 0.999 | 9.5% / 23.4% | **14.1%** |
| 700M | 0.424 / 0.840 / 0.999 | 11.5% / 35.2% | **18.2%** |
| 1.2B | 0.444 / 0.866 / 0.999 | 12.3% / 42.3% | **20.2%** |
| 2.6B | 0.426 / 0.843 / 0.999 | 11.2% / 36.0% | **19.3%** |

**14-20% of channels, at every scale, have a decay profile consistent with >10% of their energy lying
outside the k=3 window.** That is a very different sentence from "nowhere near binding."

### 4.3 The core logical point stands, and they conceded it

Their §5.5 admits correlationality. The strongest version of the counterargument: **a k=3 model's tap
profile shows only how it allocated the span it had.** Conditional on k=3, monotone decay toward the
boundary is *generic* — it is what almost any optimizer produces under a truncated basis with an
autocorrelated input, including one that would gladly use lag 5 if offered. The table in §4.1 makes
this concrete: at true r=0.7 the model leaves **11.8%** of its desired filter mass outside the window
and still shows monotone decay with only 13.9% on the boundary tap. Their §4.2 framing — *"if k=3 were
binding the optimizer would push mass onto the boundary tap"* — is **only correct for a filter whose
ideal shape is flat**. For a decaying ideal filter, truncation produces exactly the profile they
observed and called dispositive. **The prediction "boundary tap large" is the signature of a *flat*
truncated filter, not of a *truncated* filter in general.** This is the central logical gap.

---

## 5. Depth heterogeneity — cuts BOTH ways, and their pooling hides it

### 5.1 Layer 0/1 numbers reproduce

| layer | med normE(t−2) | med normE(t−1) | med normE(t) | argmax@t−2 | argmax@t−1 | argmax@t |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0057 | **0.9905** | 0.0017 | 0.0586 | **0.9277** | 0.0137 |
| 1 | 0.0346 | **0.9259** | 0.0110 | 0.0420 | **0.9346** | 0.0234 |

Confirmed. Layers 0-1 are dominantly lag-1 delay lines: 92.8% / 93.5% of channels have argmax at t−1.

### 5.2 The argument in both directions

**Their reading (supports "not binding"):** the two layers that *do* use span use exactly lag 1, not
lag 2, so even where span matters the model wants 1 lag. Fair as far as it goes.

**The counter-reading:** a lag-1 delay line is a *token-shift* primitive (RWKV `token_shift`, H3
shift-SSM), and it is a **structurally different function** from a smoother. It is not obviously a
function that wants more span — an argument in their favour. **But** it also means layers 0-1 are
**not evidence about width at all**, in either direction, and pooling them into a headline distorts it.

### 5.3 The pooling bias (a real methodological defect)

The headline "median 1.4%" pools three functionally distinct populations:

| group | med normE(t−2) | mean normE(t−2) | raw E(t−2)% | ratio E(t−2)/E(t−1) | implied r |
|---|---:|---:|---:|---:|---:|
| delay lines (L0,1) | 0.0171 | 0.0801 | 5.80% | 0.0176 | 0.133 |
| **true MIXERS (L3,4,6,7,9,11)** | **0.0284** | 0.0735 | 5.44% | **0.1839** | **0.429** |
| near-vestigial (L13,15) | 0.0003 | 0.0059 | 0.33% | 0.1438 | 0.379 |
| **ALL 10 (their headline)** | **0.0143** | 0.0613 | 4.26% | **0.0833** | 0.289 |

**Restricting to the six layers that are genuine 3-tap mixers doubles the median (0.0143 → 0.0284) and
more than doubles the pre-registered ratio statistic (0.083 → 0.184).** The headline ratio of 0.083 is
dragged down by the delay lines (0.018) and the dead layers. The "4× steeper than the Sieberling-derived
prediction (0.36)" boast in §5.3 largely evaporates: for the actual mixer layers the ratio is 0.184,
about half the Sieberling-implied 0.36 rather than a quarter — still on the right side, but not the
"strongest possible version of the result."

### 5.4 The near-vestigial layers should be excluded, not averaged in

Layers 13 and 15 have `||W||_F` of 5.27 and 6.06, comparable to every other layer (5.0-5.3) — they are
not dead in norm, they have simply converged to a scalar gain. Including them in a "does the model want
more span" statistic is like including a `k=1` layer: it can only pull the average toward "no span
needed" and carries no information about the question. That is a **pooling choice that biases toward
their conclusion.**

---

## 6. Tail analysis — there IS a sub-population, and it is small but real

The median hides it, as suspected.

### 6.1 Median vs mean: the distribution is heavily right-skewed

| checkpoint | MEDIAN normE(t−2) | MEAN normE(t−2) | ratio |
|---|---:|---:|---:|
| 350M | 0.0143 | 0.0613 | **4.3×** |
| 700M | 0.0192 | 0.0784 | 4.1× |
| 1.2B | 0.0215 | 0.0855 | 4.0× |
| 2.6B | 0.0109 | 0.0646 | **5.9×** |

**The "1.4%" headline is the most favourable of the three available summary statistics** (median 1.4%
< pooled raw 4.26% < mean 6.13%). Reporting the median without the mean, given a 4× skew, understates.

### 6.2 The tail, per layer (350M, counts out of 1024 channels/layer)

| layer | normE(t−2)>0.20 | >0.30 | >0.50 | off-current energy >0.50 |
|---:|---:|---:|---:|---:|
| 0 | 79 | 68 | 58 | **1010** |
| 1 | 100 | 70 | 39 | **1002** |
| 3 | **189** | 99 | 58 | 520 |
| 4 | 141 | 58 | 13 | 409 |
| 6 | 73 | 32 | 3 | 167 |
| 7 | 94 | 27 | 3 | 252 |
| 9 | 91 | 38 | 12 | 233 |
| 11 | 19 | 14 | 7 | 202 |
| 13 | 6 | 2 | 0 | 109 |
| 15 | 0 | 0 | 0 | 58 |
| **total** | **792 (7.73%)** | **408 (3.98%)** | 193 (1.88%) | 3962 (38.7%) |

Cross-scale, `frac normE(t−2) > 0.20`: **7.7% / 10.3% / 12.0% / 9.5%** (350M/700M/1.2B/2.6B) —
increasing with scale up to 1.2B.

### 6.3 The canonical truncation signature: pure lag-2 delay channels

The clean boundary-saturation signature is a channel that is a **pure delay pinned at the maximum lag
the window allows** (normE(t−2) > 0.8). Those exist, concentrated exactly where the mixing happens:

| ckpt | L0 pure-t−2 | L1 pure-t−2 | L0 pure-t−1 (for contrast) | whole-model normE(t−2)>0.8 |
|---|---:|---:|---:|---:|
| 350M | 50 (4.9%) | 14 (1.4%) | 912 (89.1%) | 102 (1.00%) |
| 700M | 127 (8.3%) | 43 (2.8%) | 1291 (84.0%) | 279 (1.82%) |
| 1.2B | 172 (8.4%) | 146 (7.1%) | 1697 (82.9%) | 382 (1.87%) |
| 2.6B | 146 (7.1%) | 30 (1.5%) | 922 (45.0%) | 366 (0.81%) |

**5-8% of layer-0 channels are pure lag-2 delay lines — pinned against the boundary — and the fraction
grows with model size (4.9% → 8.4%).** These are precisely the channels for which "would you like a
lag-3 or lag-4 delay?" is a live question, and they are the *only* clean truncation signature in the
checkpoint. 1-2% of all channels model-wide.

**Honest weighing:** 1-2% model-wide is small, and the prior team's directional conclusion survives it.
But their §5.4 sentence — *"**There is no sub-population anywhere in the checkpoint that wants a wider
kernel.**"* — is **factually wrong as stated**. There are 102 channels at 350M and 382 at 1.2B with
>80% of their energy pinned on the boundary tap, and 792/10240 with >20%. That is a small
sub-population, not no sub-population, and the "no" was reached by looking only at medians.

---

## 7. The random-init control is a strawman (their §4.5 item 5 / §5.2)

The control is `U(±1/√3)`, chosen to match PyTorch's `nn.Conv1d` default `U(±1/√fan_in)` with fan_in=3.
Two problems:

1. **The statistics are scale-invariant, so the control's *scale* is irrelevant.** I ran four controls
   (n=200k each):

   | control | median normE | argmax@oldest | frac >0.9·max |
   |---|---|---:|---:|
   | U(±1/√3) [theirs] | [0.2995, 0.2992, 0.2973] | 0.3350 | 0.4016 |
   | U(±1.0) [10× scale] | [0.2981, 0.3028, 0.2972] | 0.3330 | 0.3998 |
   | N(0,1) | [0.2490, 0.2502, 0.2498] | 0.3323 | 0.3718 |
   | Xavier U(±√(6/4)) | [0.2981, 0.2991, 0.2997] | 0.3327 | 0.3994 |

   **Every iid symmetric control gives argmax@oldest = 1/3 exactly, by exchangeability.** The control
   is analytically determined and carries zero bits of empirical information. "34.2% vs 2.08%, a 16×
   effect size" is rhetorically stated as if the control were an experiment; it is `1/3`.

2. **It is not even the right init.** The config says `conv_use_xavier_init: true`, so LFM2 does not
   use the PyTorch default; and per `04_cheap_experiments.md:451-452` the reference `short_conv.py`
   `init_weights` sets `weight[:, :, -1] = 1.0` — **current-token identity**, i.e. normE = [0,0,1],
   boundary-argmax = **0%**, not 33%. Against the *actual* initialization the trained model has moved
   *away* from pure passthrough toward using history — the exact opposite framing. The prior team
   noted this init fact in `04` but did not connect it to their own control in `06`.

**The right null is not "random," it is "what does a model with a genuinely wider useful span look like
after truncation to k=3" — which is my §4.1 table, and which produces a far less dramatic contrast.**

---

## 8. The "independent replication" claim — **NOT independent**

`04_cheap_experiments.md` §4.6 presents the four-checkpoint numbers as a replication that "turns §4.2
from an observation into a finding." Checking the actual artifacts:

| | |
|---|---|
| `tapread.py` | 2026-08-01 08:49:59, 2203 B |
| `tapfreq.py` | 2026-08-01 08:52:05, 3760 B |
| `tapread2.py` | 2026-08-01 08:52:39, 2244 B |

All three written within **2m 40s of each other**, on the same host, by the same session. And
`tapfreq.py:25` is a **character-for-character copy** of `tapread.py:15`:

```python
a = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
```

as is the key selector (`tapread.py:20` == `tapfreq.py:32`):

```python
[k for k in hdr if k.endswith("conv.conv.weight")], key=lambda k: int(k.split(".")[2])
```

**Findings:**
- The 350M value 4.26% is identical across the two documents because **it is the same computation
  reported twice**, not two agents agreeing. `tapfreq.py` is `tapread.py` with a frequency-response
  block bolted on and a file loop around it.
- **Every shared assumption is shared code**: same bf16 decoder, same key selector, same
  `reshape(-1,3)`, same tap-orientation assumption. Had the orientation been wrong, all four
  "replications" would have been wrong identically. The independence claim provides **zero**
  protection against the single risk it is invoked to cover.
- **What IS genuine:** the four checkpoints are real, present on FarmShare (709 MB / 1.48 GB / 2.34 GB
  / 5.14 GB), and **I read all four myself with my own code and got their numbers.** So the cross-scale
  *numbers* are verified. What is not verified is that they constitute independent *evidence* — they are
  four datapoints from one architecture family trained by one lab with one recipe, computed by one
  script. Call it a cross-scale consistency check, not a replication.

### 8.1 A real bug I found (direction: conservative)

`np.argmax` returns the **first** maximal index, and index 0 is the oldest tap. With bf16's 8-bit
mantissa, exact `|w|` ties are common — and every tie is silently credited to the boundary:

| ckpt | naive argmax@oldest | tie rate | **strict** argmax@oldest | inflation |
|---|---:|---:|---:|---:|
| 350M | 0.0208 | 0.0007 | 0.0207 | 1.0× |
| 700M | 0.0316 | 0.0010 | 0.0315 | 1.0× |
| 1.2B | 0.0353 | 0.0010 | 0.0351 | 1.0× |
| **2.6B** | **0.0975** | **0.1554** | **0.0203** | **4.8×** |

At 2.6B, **15.5% of channels have tied maxima** and the reported "9.75% oldest-is-argmax" is
**4.8× inflated**; the true value is 2.03%. Their `04` §4.6 table reports 9.75%/10.30% for 2.6B and
treats the jump as meaningful ("2.6B proves it is depth-relative"). It is a tie artifact. The bug
happens to be **conservative for their conclusion** (it makes the boundary look more used than it is),
so it does not threaten the verdict — but it means the 2.6B row of their headline table is wrong, and
the "layer 0 argmax@t−2 = 46.5%" I computed at 2.6B is likewise a tie artifact (strict value: 9.2%).

Also checked and clean: **zero all-zero channels** in any checkpoint, so the `1e-30` clip in
`tapread2.py` never fires and does not bias the median. Good.

---

## 9. VERDICT

### Bottom line on "conv-tap reads show k=3 nowhere near binding, **measured inside the gate**"

# UNCLEAR — split verdict

The claim is a conjunction, and the two halves have different answers:

| component | verdict |
|---|---|
| Tap-index orientation (index 0 = oldest, 2 = current) | **CONFIRMED** — verified from `modeling_lfm2.py` and by direct impulse test on both prefill and decode paths |
| Every reported number reproduces | **CONFIRMED** — exact, to the digit, all four checkpoints, independent code |
| "the trained k=3 conv puts most weight on the current token and decays toward the boundary" | **CONFIRMED** — robust, not fragile, replicates across scale |
| "k=3 is nowhere near binding" | **UNCLEAR / OVERSTATED** — directionally supported, but the margin is far smaller than presented (§4.2, §5.3, §6) |
| **"measured inside the gate"** | **REFUTED** — it is a weights-only measurement; the gates are never evaluated. The claim that this "removes P3's remaining defense" should be retracted. |
| "no sub-population anywhere wants a wider kernel" | **REFUTED** — 792/10240 channels at >20% boundary energy, 102 at >80%, growing with scale |
| "independent replication across four checkpoints" | **REFUTED as independence** — same script lineage, shared decoder and selector; the numbers are real, the independence is not |

### Is P3 dead?

**The direction of the evidence is right and I would not reverse the deprioritization.** But the case
as written is oversold in five specific ways, each of which I verified:

1. The gate objection is **not** answered (§3) — the project's own `spectra_v2` lesson applies, C3 is
   still owed, and `06` §5.5 asserts the opposite.
2. The headline "1.4%" is the most favourable of three summary statistics (mean is 6.1%, pooled raw is
   4.26%) and collapses to **14.4%** under a plausible AR(1) input correlation (§3.4b).
3. Pooling delay lines and vestigial layers with true mixers **halves** the key ratio statistic; the
   mixer-only ratio is 0.184, not 0.083 (§5.3).
4. The random-init control is analytically 1/3 and carries no information; the *actual* LFM2 init is
   current-token identity, against which the trained model moved **toward** history (§7).
5. "No sub-population" is false; there is a small (1-2% saturated, 8-12% substantial) and
   **scale-growing** one (§6.3).

Offsetting, in their favour and *not* in their write-up: my channel-importance weighting shows the
gates' high-throughput channels use **less** span than average (top 1%: 6.6% off-current vs 46% for the
bottom half). That is the best argument in the file for their conclusion and they did not make it.

### What would settle it, and the cost

**C3 — activation-weighted tap energy (the one they said was required and did not run).** Run 32k
calibration tokens through LFM2-350M, capture `Bx` at each LIV layer, and compute the true per-tap
variance decomposition `Var(y) = Σ_ij w_i w_j Cov(Bx_{t−i}, Bx_{t−j})` and the leave-one-out drop from
zeroing the oldest tap. This directly measures the empirical ρ that my §3.4b table parameterizes, and
it is the only thing that converts "measured on the weights" into "measured inside the gate."
**Cost: ~0.2 GPU-hours** (their own estimate in `04:391`, and it is right — one forward pass over 32k
tokens on a 350M model). This is the single highest-value unrun item.

**C5 — causal tap zeroing.** Zero the t−2 tap at inference and measure Δppl / Δrecall. Converts the
correlational claim into a causal one for the taps that exist. **Cost: ~0.3 GPU-hours.**

**What neither settles:** whether a from-scratch k=15 model would find a different solution. Only
training answers that, and at ~2 GPU-days for the `k5/k9/k15` ladder it remains correctly
deprioritized — **but on the strength of the two published width sweeps, not on the strength of this
tap read**, which cannot bear the weight §5.5 places on it.

### Recommended edits to the prior documents

- `06_p2_p3_verdict.md` §5.5: **strike** "it is measured inside Liquid's actual double gate … this
  measurement removes it. The gate does not rescue span." Replace with the weights-only caveat.
- `06_p2_p3_verdict.md` §5.4: **strike** "There is no sub-population anywhere in the checkpoint that
  wants a wider kernel." Replace with the §6.3 counts.
- `04_cheap_experiments.md` §4.6: relabel "independent replication" → "cross-scale consistency check
  (same script)". Fix the 2.6B argmax row (9.75% → 2.03% strict).
- Anywhere the "16-17× vs control" effect size appears: note the control is analytically 1/3.
- Report the mixer-only ratio (0.184) alongside the pooled one (0.083).

### Artifacts

- My scripts (FarmShare, all CPU, login node, ~0 GPU-hours):
  `/scratch/users/ericrcwu/liv/verify_taps.py` (orientation + full recomputation + tail),
  `verify_taps2.py` (tie/anomaly, importance weighting, truncation simulation),
  `verify_taps3.py` (bf16 ties, AR(1) leave-one-out, median-vs-mean, sub-population counts),
  `verify_taps4.py` (pooling bias by layer group), `verify_taps5.py` / `verify_taps6.py`
  (pure-delay sub-population, dead-channel audit).
- Prior team's scripts: `/scratch/users/ericrcwu/liv/tapread.py`, `tapread2.py`, `tapfreq.py`.
- Checkpoints: `/scratch/users/ericrcwu/liv/ckpt/{model.safetensors, LFM2-700M/, LFM2-1.2B/, LFM2-2.6B/}`.
- LFM2 source read:
  `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/.venv-spectra/lib/python3.11/site-packages/transformers/models/lfm2/modeling_lfm2.py:301-419`.
