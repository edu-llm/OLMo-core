# 04 — Claim 2 (P3 steelman) verification: Tian et al. arXiv 2607.18413 Table 7

**Verifier:** verification agent #4. **Date:** 2026-08-01. **Status: COMPLETE.**
**Constraint honored:** no code executed on the local Mac. All fetching/parsing done via
`curl` + `python3` on the FarmShare login node (CPU, no GPU jobs queued). Local work was
reading and grepping only.

**Legend:** **MEASURED** = read verbatim in the primary source. **INFERRED** = arithmetic or
logic on top of a MEASURED fact. **ASSUMED** = belief, not verifiable from primary source.

---

## TL;DR

| Sub-claim | Verdict |
|---|---|
| arXiv 2607.18413 exists, is Tian et al. *Convolution for Large Language Models* | **CONFIRMED** |
| Table 7 exists, caption is about **reparameterization**, branches merged at inference | **CONFIRMED** |
| The three number triples (2.4795/12.79/1721.03, 2.5029/13.28/1721.26, 2.5048/13.28/1721.61) | **CONFIRMED verbatim, exact** |
| Baseline row is genuinely "single k=3, no reparameterization" | **CONFIRMED** |
| The k=1+k=3 fusion preserves the function class | **CONFIRMED** (no norm, no activation on the branches) |
| Tian's conv is an **ungated residual** `Y = X + Conv(X)` | **CONFIRMED** |
| arXiv 2606.03825 (Sieberling et al.) exists; width/rank numbers as claimed | **CONFIRMED, exact to 2 dp** |
| **Headline: "P3's steelman was already run in Table 7"** | **REFUTED as stated** — see §3 |
| Table 6 is "mild evidence against the gates escape hatch" | **REFUTED** — it is weak evidence *for* it (§4) |

**Bottom line: REFUTED.** Table 7 is real, its numbers are exactly as quoted, and its direction
is negative — but it is **not** the capstone's steelman. It tests over-parameterization at
**fixed span** (3 lags), with **no per-branch normalization** (so it is not RepVGG's method), at
n=1 with no error bars. The capstone's steelman couples over-parameterization with a **5× span
increase to lag 14** via dilations. Table 7 is *adjacent negative prior*, not a closed question.

---

## 1. Existence check: arXiv 2607.18413 — **REAL**

**MEASURED.** Three independent resolutions, all consistent:

- `https://arxiv.org/abs/2607.18413` → resolves, not an error page.
- `http://export.arxiv.org/api/query?id_list=2607.18413` → 1 entry, no error entry.
  Entry id `http://arxiv.org/abs/2607.18413v1`.
- `https://arxiv.org/html/2607.18413v1` → 169,334 bytes of LaTeXML HTML, generated
  "Mon Jul 20 18:01:14 2026". (`v2` → HTTP 404; v1 is the only version.)

**Verbatim from the HTML masthead (MEASURED):**

```
arXiv:2607.18413v1 [cs.CL] 20 Jul 2026
Convolution for Large Language Models
Technical Report
Yuchuan Tian(1), Yingte Shu(1), Wei He(2), Shuo Zhang(2), Tianchen Zhao(3),
Chao Xu(1), Xinghao Chen(2), Yunhe Wang(2), Hanting Chen(2,†), Yu Wang(3)
(1) Peking University   (2) Huawei Technologies   (3) Tsinghua University
(July 2026)
```

Published/updated 2026-07-20T18:02:25Z. Primary category cs.CL. Comment "12 pages, 5 figures".
DOI `10.48550/arXiv.2607.18413`.

**Every descriptive detail the prior team asserted checks out**: title, the PKU/Huawei/Tsinghua
affiliation set, 20 Jul 2026 submission, Qwen3 backbone, residual depthwise conv, k=3 best,
<0.01% params. Also MEASURED from §5.1: backbone **Qwen3-1.7B** (and 4B for the main results),
trained **from scratch on FineWeb-100B**, metric = mean training loss over the **final 10,000
iterations** plus **WikiText-103** perplexity; context length 4096, global batch 256 sequences
(1,048,576 tok/step); peak LR 1.0e-3 (1.7B). Causality: "we use a causal depthwise convolution.
For kernel size k, left padding by k−1 positions…" **No hallucination anywhere in the prior
team's description of this paper.**

Note on the future-dated ID: 2607 = July 2026, chronologically consistent with today
(2026-08-01) and with the LaTeXML generation timestamp embedded in the HTML. One WebFetch
summarizer flagged the ID as implausible; that flag is a stale-knowledge artifact of the
summarizer model, contradicted by the export-API Atom feed and the raw HTML I downloaded.

---

## 2. Table 7 — verbatim from the primary source

**MEASURED**, transcribed from the raw HTML (parsed on FarmShare, not from a summarizer):

> **Table 7: Effect of convolutional reparameterization in Qwen3-1.7B.**
>
> | Configuration | Mean loss | Perplexity | Params (M) |
> |---|---|---|---|
> | No reparameterization | 2.4795 | 12.79 | 1721.03 |
> | Kernel 1 branch | 2.5029 | 13.28 | 1721.26 |
> | Kernel 1 and 2 branches | 2.5048 | 13.28 | 1721.61 |

**The entire body paragraph, verbatim (MEASURED):**

> **Reparameterization.**
> We also train multi-branch Conv1D modules with different kernel sizes and merge the branches
> into an equivalent convolution for inference. Both reparameterized variants perform worse than
> the single-branch configuration in Table 7. One possible explanation is that the branches mix
> local patterns over different spans, although this ablation does not isolate the cause. We
> therefore use a single compact Conv1D module.

And the §3.5 lead-in (MEASURED):

> After selecting the location, module design, kernel size, and initialization, we test three
> additional choices: a second convolution, nonlinear activation, and multi-branch
> reparameterization. None improves both loss and perplexity over the selected configuration.

**Answers to the three sub-questions in the brief:**

**(a) Is Table 7 about reparameterization, and are branches merged at inference?** **YES,
MEASURED.** The caption says "convolutional reparameterization"; the body says "merge the
branches into an equivalent convolution for inference". The word *equivalent* is the authors'
own, and it asserts exact fusibility.

**(b) Do the exact numbers appear?** **YES, MEASURED — all nine cells match the prior team's
table digit-for-digit.** No transcription error.

**(c) Is the baseline row genuinely "single k=3, no reparameterization"?** **YES, INFERRED with
high confidence.** 12.79 / 2.4795 / 1721.03 is the *same* row that appears as the selected
configuration in Table 2 ("Conv + Shortcut"), Table 3 (k=3), Table 4 (random weight + random
bias), and Table 6 ("No activation"). By §3.5 it is the running selected config: **residual
depthwise causal Conv1D, k=3, at P5 (post-QKV projection, pre-attention), random init, no extra
norm, no activation.**

### 2.1 Supporting tables, verbatim (MEASURED)

> **Table 2: Effect of the Conv1D module design on Qwen3-1.7B.**
>
> | Configuration | Mean loss | Perplexity | Params (M) |
> |---|---|---|---|
> | No Conv1D | 2.5144 | 13.42 | 1720.57 |
> | Convolution | 2.4931 | 13.05 | 1721.15 |
> | Conv + Shortcut | 2.4795 | 12.79 | 1721.03 |
> | Pre-Norm | 2.4876 | 12.94 | 1721.03 |
> | Sandwich Norm | 2.5110 | 13.33 | 1721.26 |

> **Table 3: Effect of kernel size on Qwen3-1.7B.**
>
> | Kernel size | Mean loss | Perplexity | Params (M) |
> |---|---|---|---|
> | No Conv1D | 2.5144 | 13.42 | 1720.57 |
> | k=2 | 2.4894 | 12.99 | 1720.92 |
> | k=3 | 2.4795 | 12.79 | 1721.03 |
> | k=4 | 2.4881 | 13.13 | 1721.15 |

> **Table 4: Effect of Conv1D initialization on Qwen3-1.7B.** "Random" denotes sampling the
> weight or bias from N(0, 0.02).
>
> | Weight | Bias | Mean loss | Perplexity | Params (M) |
> |---|---|---|---|---|
> | zero | N/A | 3.0065 | 61.52 | 1720.92 |
> | zero | zero | 2.5056 | **13.27** | 1721.03 |
> | zero | random | 2.4908 | 12.85 | 1721.03 |
> | random | random | 2.4795 | 12.79 | 1721.03 |

> **Table 6: Effect of activation functions in the Qwen3-1.7B Conv1D module.**
>
> | Configuration | Mean loss | Perplexity | Params (M) |
> |---|---|---|---|
> | No activation | 2.4795 | 12.79 | 1721.03 |
> | SiLU | 2.4962 | 13.06 | 1721.03 |
> | LeakyReLU | 2.4892 | 12.96 | 1721.03 |
> | sigmoid | 2.4889 | 12.94 | 1721.03 |

The four module designs, verbatim (MEASURED, §3.3):

> (1) plain convolution, Y = Conv(X);
> (2) convolution with a shortcut, Y = X + Conv(X);
> (3) convolution with pre-normalization, Y = X + Conv(Norm(X)); and
> (4) convolution with sandwich normalization, Y = X + Norm(Conv(Norm(X))).

> The shortcut preserves the projected QKV features and lets the convolution learn a residual
> local correction. … We therefore retain the simpler Conv + Shortcut design for the remaining
> experiments.

**No gating variant is tested anywhere in the paper.** (MEASURED by exhaustive grep: the only
occurrences of "gate/gating" in the full text are describing Qwen3's own SwiGLU FFN and the
related-work mention of ConViT's "gated positional bias".)

### 2.2 The params column is the TRAINING-TIME count — proved by arithmetic

**INFERRED (exact).** Qwen3-1.7B: d_model = 2048, L = 28 (both MEASURED from §3.1). Concatenated
QKV width at P5 = 2048 (Q: 16×128) + 1024 (K: 8×128) + 1024 (V: 8×128) = **4096**. A depthwise
causal conv with k taps **plus a bias** costs (k+1) × 4096 × 28 parameters:

| | predicted Δ params | reported |
|---|---|---|
| k=3 over no-conv | (3+1)·4096·28 = 458,752 = 0.4588M → 1720.57 + 0.459 = **1721.029** | **1721.03** ✓ |
| k=2 over no-conv | (2+1)·4096·28 = 344,064 = 0.3441M → **1720.914** | **1720.92** ✓ |
| k=4 over no-conv | (4+1)·4096·28 = 573,440 = 0.5734M → **1721.143** | **1721.15** ✓ |
| + k=1 branch | +(1+1)·4096·28 = 0.2294M → 1721.029 + 0.229 = **1721.259** | **1721.26** ✓ |
| + k=1 and k=2 branches | +0.2294 + 0.3441 → **1721.602** | **1721.61** ✓ |

Every cell reproduces. Two consequences:

1. **The params column is the un-fused, training-time count.** The prior team's hunch was right.
   After fusion, all three Table 7 rows would report 1721.03. This is cosmetic for the argument
   but confirms nothing is being hidden.
2. **Each branch carries its own bias term** (the +2/(k+1) pattern only closes with the bias).
   This matters — see §3.4.

---

## 3. Is Table 7 the capstone's steelman? — **NO**

### 3.1 The function-class claim IS mathematically correct (prior team right on this)

**CONFIRMED.** The selected module is `Y = X + Conv(X)` with **no norm and no activation** on the
conv path (MEASURED, §3.3/§3.4/Table 6). Both branches are therefore linear depthwise causal
convs, and

```
Conv_{k=3}(X) + Conv_{k=1}(X)
  = [w3_0, w3_1, w3_2] * X + [w1_0] * X   (aligned at lag 0)
  = [w3_0 + w1_0, w3_1, w3_2] * X          — still a 3-tap depthwise causal kernel
```

with biases summing likewise. So yes: **k=1 ⊕ k=3, summed and fused, is exactly a k=3 kernel.**
Same function class, different training-time parameterization, zero inference cost. The paper's
own word "equivalent" independently asserts this. The prior team's core mathematical reading is
sound.

### 3.2 …but it is a *different question* from the capstone's steelman

The capstone's steelman, verbatim from
`/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/04_multiscale_routing.md`
line 1161 (MEASURED, local file):

> **does the 4-branch *training-time* parameterization of a 15-tap kernel optimize better than a
> directly-trained 15-tap kernel, even though they are the same function class?**

P3's branches, MEASURED from `07_latency_kernels.md` lines 451-458: **four 3-tap branches at
dilations 1 / 2 / 4 / 7**, reaching lags

```
dilation 1: {0, 1,  2}
dilation 2: {0, 2,  4}
dilation 4: {0, 4,  8}
dilation 7: {0, 7, 14}
```

Now put the two side by side (INFERRED):

| | Tian Table 7 | Capstone P3 steelman |
|---|---|---|
| base branch | k=3, lags {0,1,2} | 4 branches, dilated |
| added branches | k=1 (lag {0}); then k=2 (lags {0,1}) | — |
| **union of lags** | **{0,1,2} — unchanged** | **{0,1,2,4,7,8,14}** |
| **max lag / span** | **2 → 2 (no change)** | **2 → 14 (7×)** |
| free taps → distinct lags | 4→3, then 6→3 | 12→7 |
| per-branch normalization | **none** | (unspecified; RepVGG uses BN) |
| fused kernel | dense 3-tap | **sparse 15-tap: 7 of 15 lags nonzero** |

Two distinct mechanisms are in play in any multi-branch conv: **(a) over-parameterization** (more
training-time params than the fused kernel has degrees of freedom) and **(b) span expansion**
(the union of branch supports is wider than the base branch). **Tian's Table 7 varies (a) with
(b) held completely fixed. The capstone's P3 varies (a) and (b) together.** They are not the
same experiment. Table 7 is the correct *control* for the capstone's steelman, not the
experiment itself.

### 3.3 A correction that cuts against the capstone too

The capstone's own framing — "the same function class as a directly-trained 15-tap kernel" — is
**imprecise**. The 4-branch dilated block fuses into a 15-tap *container* but only **7 of 15 lags
are free**; lags 3, 5, 6, 9, 10, 11, 12, 13 are structurally zero. So the fused function class is
a strict *subset* of the dense 15-tap class. A clean reparameterization experiment would have to
compare against a directly-trained kernel with the **same sparse support**, not a dense k=15.
Compared against a dense k=15, the experiment is confounded (it also tests sparsity). The prior
team did not catch this; neither did the design doc. This does not rescue Table 7 as the answer —
it means the steelman needs re-specification before it is even well-posed.

### 3.4 Table 7 is not RepVGG's method — the load-bearing structural difference

RepVGG's block is `BN(3×3) + BN(1×1) + BN(identity)`, summed. **The per-branch BatchNorm is
essential** to the training-time effect; without it, the three branches collapse into a nearly
trivial reparameterization at init and the claimed optimization benefit largely evaporates. The
capstone dossier already records RepVGG's own caveat verbatim (`04_multiscale_routing.md` ~line
1147): *"the inference-time equivalence does not imply the training-time equivalence."*

**Tian's branches have no normalization at all** (MEASURED — his selected module has no norm, and
Table 2 shows he *tested and rejected* norm on the conv path). So Table 7 evaluates a **norm-free
sum of linear branches**, which for the k=1 case amounts to reparameterizing the centre tap as
`w3_0 + w1_0` — one redundant scalar per channel per layer. That is the most degenerate possible
instance of "structural reparameterization." Calling it "the RepVGG-style structural
reparameterization question" **overstates what was run.** It answers "does adding one redundant
scalar per channel help?" It does not answer "does a normalized multi-branch training topology
optimize better?"

### 3.5 Effect size and reliability — the negative result is fragile

**MEASURED:** the paper **never mentions seeds, repeated runs, variance, standard deviations, or
error bars** for any ablation. Exhaustive grep for `seed|variance|error bar|std|standard
deviation|repeated run` over the full text returns exactly one hit, in §5 and about the *main
results*, not the ablations:

> …additional training budgets and **repeated runs would be needed** to establish a general
> scaling trend.

So **every Table 7 delta is n=1.** Four reasons to distrust the −0.49 ppl (INFERRED):

1. **It is implausibly large.** The *entire* effect of adding the convolution at all is
   13.42 → 12.79 = **0.63 ppl** (Table 2). Table 7 claims that adding **one redundant scalar per
   channel** — a change that is a no-op at inference — destroys **78% of the whole intervention**.
2. **It is worse than removing the residual shortcut entirely.** Plain `Y = Conv(X)` scores 13.05
   (Table 2). The reparameterized variant scores 13.28 — i.e. adding a fusible k=1 branch to the
   best design is reported as *more* damaging than deleting the residual connection. That is a
   red flag for a confound, not a mechanism.
3. **13.28 ≈ 13.27 = the degraded-init row.** Table 4's "zero weight, zero bias" init lands at
   **13.27**. Both Table 7 reparam rows land at **13.28**. Table 4 also shows init is enormously
   load-bearing here (61.52 ppl for one bad setting). Since every added branch carries **its own
   independently random-initialized bias** (proven by the param arithmetic in §2.2), the summed
   module has ~√2 and ~√3 the init bias-scale of the baseline. **A plain init-scale confound is a
   live alternative explanation** for the whole Table 7 result, and the authors did not control
   for it.
4. **The authors themselves decline to claim a mechanism**: *"One possible explanation is that the
   branches mix local patterns over different spans, although this ablation does not isolate the
   cause."* Note the irony — their own guess ("different spans") does not even apply to their
   experiment, since k=1 and k=2 branches add **no new spans** over k=3. That sentence is a
   non-sequitur and further suggests the ablation was not thought through as a reparameterization
   test.

### 3.6 Does the prior team's own caveat (iii) undercut their headline? — **YES**

Their caveat (iii), `06_p2_p3_verdict.md` line 90: *"They merged only to reach the same k=3
function class; nobody has tested branch-reparameterization of a genuinely wider (k=15) kernel."*

This is correct and it is **not a footnote — it is the whole distinction.** Their headline is
"The steelman is not novel and the one existing datapoint is negative" (line 82-83) and, in the
TL;DR, "P3's steelman is also dead… already been run — Tian et al. Table 7 is exactly that
experiment." Caveat (iii) says, in their own words, that it is *not* that experiment. The
headline and the caveat are in direct contradiction, and the caveat is the one that is right.

---

## 4. The gate nuance — Tian's Table 6 does **not** support the prior team's inference

**Fact base, all MEASURED:**
- Tian's conv is `Y = X + Conv(X)`: **ungated residual**, no norm, no activation, at P5. The
  prior team's characterization is **correct**.
- LFM2's conv sits **inside two multiplicative gates** (project-established; the LIV block).
- Tian's Table 6 tests three **pointwise activations** on the conv output: SiLU 13.06,
  LeakyReLU 12.96, sigmoid 12.94, vs 12.79 for none.

**Prior team's inference** (line 92-93): Table 6's "adding *any* nonlinearity to the conv path
hurts" is "mild evidence against the 'gates change everything' escape hatch."

**Verdict: REFUTED. The inference is invalid, and the table arguably points the other way.**

1. **Different operator.** An activation is a *fixed, parameter-free, elementwise* map applied to
   the conv output: `Y = X + σ(Conv(X))`. A gate is a *learned, input-dependent, per-channel
   multiplicative* modulation computed from a separate projection: `Y = g(X) ⊙ Conv(X)`. They
   share only the label "nonlinearity." A sigmoid squashes; a gate routes and rescales with its
   own parameters and its own gradient path. Tian never trains a gated variant — there is no row
   in the paper that bears on gating.
2. **Different location and different claim.** The escape hatch is not "a nonlinearity on the conv
   path helps." It is "the *surrounding* multiplicative gates change the optimization landscape
   and the energy allocation across taps, so conclusions drawn about an ungated conv may not
   transfer." Table 6 does not vary the surround at all — it varies what sits on the conv output
   inside an otherwise-fixed ungated residual.
3. **Bolted-on vs. native.** Tian's activations are *added to* an architecture that was designed
   and tuned without them (his whole method section is "minimal disruption to the Transformer
   block"). LFM2's gates are **native** — the conv was co-trained with them from scratch. "Adding
   X to a tuned design hurts" says nothing about "a design built around X behaves differently."
4. **The direction of the evidence is backwards.** What Table 6 actually shows is that a
   *parameter-free pointwise map* on the conv output moves perplexity by **0.15-0.27 ppl** —
   comparable to the entire k=2-vs-k=3 gap (0.20) and to a third of the whole conv effect.
   Combined with Table 2 (pre-norm costs 0.15, sandwich norm costs 0.54), the honest summary is:
   **what surrounds this convolution matters a great deal.** That is, if anything, *mild support*
   for the escape hatch, not evidence against it. The prior team read a sensitivity result as a
   robustness result.

The residual honest point in their favour: Tian's paper collectively establishes that in *his*
setting the linear, minimal, ungated form optimizes best, and every embellishment tried lost. That
is a prior over embellishments in general, weakly transferable. But it is a prior, not evidence,
and it should not be labelled "evidence against the gates escape hatch."

---

## 5. Sibling citation: arXiv 2606.03825 (Sieberling et al.) — **REAL, numbers exact**

**MEASURED.** Export API + `https://arxiv.org/html/2606.03825v1` (701 KB; v2 → 404).

- **Title:** *Dynamic Short Convolutions Improve Transformers*
- **Authors:** Oliver Sieberling, Bharat Runwal, Rameswar Panda, Yoon Kim
- **Submitted:** 2026-06-02T16:07:55Z. Categories cs.LG (primary), cs.CL.
- Triton kernels released: `https://github.com/OliverSieberling/dynamic-conv1d`

**Table 3(a) verbatim (MEASURED)** — caption: *"Ablations on the 300M models trained on 15B
tokens, reporting Nemotron-CC perplexity. (a) Sweep over kernel width W, head size H, and rank R
for dynamic convolutions on Q+K+V."*

| Width W (low-rank, R=16) | Params | PPL | | Rank R (low-rank, W=4) | Params | PPL |
|---|---|---|---|---|---|---|
| W=1 | 306.8M | 18.42 | | R=4 | 306.3M | 18.26 |
| W=2 | 307.6M | 18.17 | | R=8 | 307.3M | 18.19 |
| W=3 | 308.5M | **18.08** | | R=16 | 309.3M | 18.10 |
| W=4 | 309.3M | 18.10 | | R=32 | 313.2M | 18.04 |
| W=5 | 310.1M | 18.09 | | R=64 | 321.1M | 17.87 |
| W=6 | 311.0M | 18.10 | | R=128 | 336.8M | **17.85** |

Head-size sweep (head-wise, W=4): H=8 18.03 / H=16 18.08 / H=32 18.21 / H=64 18.25 / H=128 18.40.
Table 3(b) no-conv Transformer baseline: **19.12**.

**Every claim in the docs checks out (INFERRED, arithmetic on MEASURED cells):**

| Claim | Check |
|---|---|
| width sequence 18.42/18.17/18.08/18.10/18.09/18.10 | **exact** |
| marginal gain per added lag: +0.25, +0.09, −0.02, +0.01, −0.01 | 18.42−18.17=**0.25**; 18.17−18.08=**0.09**; 18.08−18.10=**−0.02**; 18.10−18.09=**+0.01**; 18.09−18.10=**−0.01** — **exact** |
| rank sweep buys 0.25 ppl | 18.10 (R=16) − 17.85 (R=128) = **0.25** — **exact** |
| span past k=3 buys 0.00 | 18.08 (W=3) → 18.10 (W=6) = **−0.02**, i.e. ≤0 — **confirmed** |
| authors' own conclusion | verbatim: *"For width, we find that 3 or 4 is generally the sweet spot… Widths beyond this sweet spot do not provide additional gains even though they add parameters."* — **confirmed** |

**Gated or ungated? — UNGATED. MEASURED, verbatim §3:**

> We apply each with a residual, i.e., **X = X + dynamicShortConv(X)** for X ∈ {Q, K, V}.

So **both** load-bearing citations for "width is flat past k=3" measured it in an **ungated
residual** conv on the QKV path. Neither measured it inside a multiplicative gate. Sieberling
*does* touch gated architectures (Mamba-2, Gated DeltaNet in Table 1), but only to swap static
convs for dynamic ones — **no width sweep is run in a gated mixer.** The gated-context evidence
in this whole program remains the capstone's own LFM2 weight measurement (boundary tap 1.4%
energy), which is observational, not interventional. Worth stating plainly rather than letting
the two ungated citations look like independent gated confirmation.

**Seeds:** the only seeded result is the synthetic MQAR figure — *"Left: Performance (median over
5 seeds)… The error bars depict the minimum and maximum values."* **The LM ablations in Table 3
carry no seed or variance information.** Same n=1 caveat as Tian, on a 0.02-0.09 ppl scale where
it matters even more.

**Adjacent finding worth flagging for P3's redesign (INFERRED):** Sieberling's rank sweep buys
0.25 ppl (R=16→128) while span buys 0.00. The payoff axis in that paper is the **expressivity of
the input-dependent filter generator**, not the receptive field. That is simultaneously bad news
for P3's dilation/span premise and mildly good news for P3's *router* premise — but Sieberling's
dynamic conv generates filter weights per token directly, which is a strictly more general (and
already-published, already-kernelized) version of a softmax router over 4 fixed branches. If P3
survives in any form, this is the thing it would have to beat, and it has released Triton kernels.

---

## 6. BOTTOM LINE

**Proposition under test:** *"P3's steelman (RepVGG-style structural reparameterization) was
already run and found negative in Tian et al. Table 7."*

### **REFUTED.**

Decomposed:

- **"Tian et al. arXiv 2607.18413 Table 7 exists and reports 12.79 → 13.28 → 13.28 for
  reparameterization with inference-time branch merging"** — **CONFIRMED.** Verbatim, exact, every
  cell. The prior team transcribed the paper faithfully and hallucinated nothing. Their upgrade of
  the design doc's sloppy "mixed widths" framing to "this is the *reparameterization* table" is a
  genuine and correct catch.
- **"…and it IS the capstone's steelman"** — **REFUTED.** Table 7 adds *narrower* branches (k=1,
  k=2) to a k=3 kernel; the union of lags is unchanged at {0,1,2}. The capstone's steelman adds
  *dilated* branches that extend the reachable lag set from {0,1,2} to {0,1,2,4,7,8,14}. Table 7
  isolates over-parameterization; P3 confounds over-parameterization with a 7× span increase.
  Table 7 is the right **control** for the steelman, not the steelman.
- **"…RepVGG-style"** — **REFUTED.** Tian's branches carry **no per-branch normalization**, which
  is the component RepVGG's training-time benefit actually rests on. Table 7 tests a norm-free
  linear branch sum — for the k=1 case, one redundant scalar per channel. That is the weakest
  instance of the idea, not a faithful port of RepVGG's method.
- **"…found negative"** — **CONFIRMED in sign, UNRELIABLE in magnitude.** n=1, no error bars
  anywhere in the paper. The −0.49 ppl is 78% of the paper's entire conv effect, is worse than
  deleting the residual shortcut, and lands within 0.01 of the paper's own degraded-init row
  (13.27) while each added branch demonstrably introduces another independently random-initialized
  bias (proved by the param arithmetic in §2.2). Init-scale confound is a live alternative. The
  authors themselves say the ablation "does not isolate the cause," and their offered explanation
  ("branches mix local patterns over different spans") does not even apply, since their branches
  add no new spans.

**Net effect on the P3 decision — the verdict changes, the recommendation mostly does not.**
The steelman is **weakened but alive**: nobody has run branch-reparameterization of a genuinely
wider kernel, and nobody has run it with per-branch normalization or inside a gated block. But
the case for *cutting P3* never rested on Table 7 — it rests on the capstone team's own LFM2
weight measurement (boundary tap = 1.4% of median per-channel energy, measured **inside** Liquid's
double gate) plus Sieberling's flat width sweep. Those stand. What must change is the **framing**:
Table 7 should be cited as *"adjacent negative prior, n=1, fixed-span, norm-free"* — **not** as
*"the experiment has been run."* Claiming the latter in a proposal or defense is a
mischaracterization a reviewer with the paper open would catch in one minute.

**Secondary correction that should be propagated:** `06_p2_p3_verdict.md` line 92-93's reading of
Table 6 as "mild evidence against the gates-change-everything escape hatch" is invalid (§4). A
pointwise activation on the conv output is not a multiplicative gate around it, and the magnitude
of Table 6's swings is better read as evidence that the conv's surround matters — i.e. weakly
*supporting* the escape hatch.

### Exact URLs used

- `https://arxiv.org/abs/2607.18413`
- `http://export.arxiv.org/api/query?id_list=2607.18413`
- `https://arxiv.org/html/2607.18413v1` (v2 → HTTP 404)
- `http://export.arxiv.org/api/query?id_list=2606.03825`
- `https://arxiv.org/html/2606.03825v1` (v2 → HTTP 404)
- `https://arxiv.org/abs/2101.03697` (RepVGG, cited from the local dossier, not re-fetched)

### What I could not verify

- **PDF cross-check.** I worked from the LaTeXML HTML only. HTML is auto-generated from the same
  source, so table cells are reliable; figures (Fig. 3's module diagrams) were not inspected.
  Nothing in my analysis depends on a figure.
- **Whether Tian's Table 7 branches were initialized identically to the baseline** — the paper
  does not say. §3.5's init-confound hypothesis is therefore INFERRED from the param arithmetic
  (each branch has its own bias) plus Table 4 (init is load-bearing), not stated by the authors.
- **Whether Tian's branch sum included per-branch scaling** — not stated. The exact fusibility
  asserted by "equivalent convolution" rules out norms and nonlinearities but is silent on fixed
  scalars, which would not change the function class anyway.
