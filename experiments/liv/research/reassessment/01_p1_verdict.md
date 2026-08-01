# P1 reassessment — adversarial adjudication

**Author:** reassessment team member (P1 track). **Date:** 2026-08-01.
**Sources re-derived from raw JSON, not from the README.**
Every claim tagged **[MEASURED]** (a number I read out of a results JSON or recomputed from one),
**[INFERRED]** (a derivation from measured numbers), or **[ASSUMED]** (I could not check it here).

**Constraint honored:** no code executed on the Mac after this section was started; arithmetic
below is either hand-derived or run as a CPU check on the FarmShare login node.

---

## 0. Bottom line up front

| question | verdict |
|---|---|
| Is the latency claim dead? | **Yes — but the evidence is weaker than the README says.** The *sign* replicates across two jobs; the *magnitude does not* (−1.8% in job 1670883 vs −8.2% in job 1670884, a 4.5× discrepancy the README does not mention). The defensible statement is "**no speedup, somewhere between 2% and 8% slower**", not "8.2% slower". |
| Is the spectra result a falsification? | **Of the stated premise, yes. But the more interesting reading is being thrown away** — see §2. The honest finding is *"the activation-weighted input subspace of the LIV block is low-rank; every d→d matrix reading it inherits that"*, which is a **broader** claim than P1 and points at a **bigger** experiment. |
| Is P1 worth GPU-days? | **No, not as currently scoped.** ~12 arms × 2 seeds is ~130–260 GPU-hours (§3) to measure a 4.4% parameter saving against a control (`N-narrow`) that is expected to match it. Cut to **2 arms** or fold P1 into a different question. |
| Param-efficiency claim | **Both HANDOFF numbers are defensible; they measure different things and the docs conflate them** — see §4. The model-level cut is **4.41%** (params) and the LIV-gate-bytes cut is 4× — neither is 6.27% at the frozen geometry, and I could not reproduce 6.27% from the L0 ledger (§4). |

---

## 1. Verification of the latency measurements

### 1.1 What the microbenchmark actually measures — and it is NOT the mixer

**[MEASURED, from `probes/p1_launch_bench.py:64-121, 194`]** The benchmark instantiates *only the two
gate projections*, stacked 10 deep, and times `x` of shape `(1, 1024)` bf16:

- `Dense` = `nn.Linear(1024,1024)` **× 2** (`b`, `c`) → 2 kernels/layer, 2d² = 2,097,152 params/layer.
- `LowRankFused` = `Linear(1024, 2r)` + 2 × `Linear(r, 1024)` → 3 kernels/layer.
- `Grouped` = 2 × `torch.bmm` over `(g, bs, bs)` → 2 kernels/layer.

**Three confounds this creates, in order of severity:**

1. **The `dense` control is not stock LIV.** **[MEASURED, from the parity work recorded in
   `HANDOFF.md:250-256` and `probes/spectra_v2.py:128` `W.chunk(3, dim=0)`]** the released
   `Lfm2ShortConv` has **one fused `in_proj` of shape `(3d, d)`** producing `(B, C, x)` in a *single*
   GEMM. The benchmark's `dense` splits that into **two** separate `d→d` GEMMs and **omits the `x`
   projection entirely**. So the control reads 2d² in 2 kernels where the real thing reads 3d² in
   **1** kernel.
   **[INFERRED]** Direction of the bias: this *pessimizes* the control. A single wide `1024→3072`
   GEMV will achieve at least the bandwidth of a `1024→1024` one, so real stock LIV is *more*
   efficient than the benchmarked `dense`. **The real regression from P1 is therefore ≥ the measured
   one.** The confound does not threaten the conclusion; it strengthens it. **Worth stating in the
   writeup, because a reviewer will ask.**
2. **The −8.2% is a percentage of gates-only time, not of anything a user experiences.**
   **[MEASURED]** 4.61 µs absolute (60.832 − 56.224) added per decoded token across 10 LIV layers.
   **[INFERRED]** Against the measured real-model decode profile in `HANDOFF.md:88-94` — where the
   LIV `Conv` is 1.0% and `MatMulNBits` is 91.2% of a 42.4 ms/step ONNX q4 decode — a 4.6 µs
   regression is *unmeasurable at the model level on that runtime*. The correct framing is
   **"P1 does not produce the speedup it promised"**, not "P1 makes the model 8% slower."
3. **batch = 1, seq = 1, exactly one point.** **[MEASURED, `p1_launch_bench.py:194`]**
   `x = torch.randn(1, D, ...)`. **[INFERRED]** Every arm is a pure GEMV. At batch ≥ 32 these become
   GEMMs, the kernels stop being latency-bound, and P1's 4× FLOP reduction on the gates could flip
   the sign. **The benchmark does not test this and the docs do not say so.** It is *defensible* to
   ignore — LFM2 is an edge model and batch=1 is the deployment regime — but the claim must be
   scoped as **"dead at batch-1 edge decode"**, not "dead."

**[MEASURED]** Apples-to-apples on bytes: `bytes_per_token()` (`p1_launch_bench.py:123-137`) is
correct. `lowrank_fused` = `d·2r + 2·r·d = 4dr`; `lowrank_sep` = `2·(2dr) = 4dr`. Identical, as the
docstring claims. At r=512, `4dr = 4·1024·512 = 2,097,152 = 2d²` — **exactly equal to dense**, so the
40 MiB iso-byte control is genuinely iso-byte. ✅ No error here.

### 1.2 Arithmetic check: −8.2% ✅, but the replication ❌

**[MEASURED]** From `p1_verify_results.json`: dense median **56.223999708890915 µs**, `lowrank_fused
r=128` median **60.83200126886368 µs**.
`(56.224 − 60.832) / 56.224 = −0.081960…` → **−8.196%, rounds to −8.2%. The arithmetic is correct.**

Full recomputation of the verify job (all medians, all deltas), independently derived:

| arm | median µs | kernels | MiB | vs dense | µs/kernel |
|---|---:|---:|---:|---:|---:|
| `dense` | 56.224 | 20 | 40.0 | — | 2.811 |
| `lowrank_fused r=128` | 60.832 | 30 | 10.0 | **−8.20%** | 2.028 |
| `lowrank_fused r=512` | 90.016 | 30 | 40.0 | −60.10% | 3.001 |
| `lowrank_sep r=128` | 76.480 | 40 | 10.0 | −36.03% | 1.912 |
| `grouped g=2` | 47.680 | 20 | 20.0 | +15.20% | 2.384 |
| `grouped g=4` | 47.600 | 20 | 10.0 | +15.34% | 2.380 |

All README/HANDOFF table entries reproduce. ✅

> ### 🔴 DISCREPANCY 1 — the "replication" did not replicate the effect size, and nobody said so
>
> **[MEASURED]** The two jobs disagree by 4.5× on the headline number:
>
> | arm | job 1670883 (`p1_bench_results.json`) | job 1670884 (`p1_verify_results.json`) | between-job drift |
> |---|---:|---:|---:|
> | `dense` | 56.320 µs | 56.224 µs | **−0.17%** |
> | `lowrank_fused r=128` | 57.344 µs | 60.832 µs | **+6.08%** |
> | `lowrank_fused r=512` | 86.384 µs | 90.016 µs | +4.20% |
> | `lowrank_sep r=128` | 72.608 µs | 76.480 µs | +5.33% |
> | `grouped g=2` | 44.864 µs | 47.680 µs | +6.28% |
> | `grouped g=4` | 44.928 µs | 47.600 µs | +5.95% |
>
> **The headline "−8.2% slower" is −1.82% in the first job** (and that value is *in
> `p1_bench_results.json` as `speedup_graphed_pct: -1.8181824195006246`* — the field is right there
> in the file the README cites). Likewise `grouped` is **+20.3%** in job 1 and **+15.3%** in job 2.
>
> **The "≤0.3% spread" the README leans on is a within-job, back-to-back-trial spread. It bounds
> nothing about between-job variance, which is measured at 4–6% — comparable to the headline effect
> itself.** Note also that `dense` is the *only* arm that agrees between jobs, and it is the *first
> arm run* in both scripts — consistent with a clock/thermal or first-touch systematic that the
> protocol does not control (no clock locking anywhere in either script).
>
> **CORRECTION to adopt:** report *"`lowrank_fused r=128` is **2–8% slower** than dense across two
> independent jobs; no configuration is faster despite reading 4× fewer bytes."* The **sign is
> robust** (2/2 jobs, all ranks, both fused and separate) and that is what kills the claim. The
> **point estimate is not**, and quoting 8.2% to one decimal from n=1 job is over-precision that a
> reviewer can dismantle in one question.

### 1.3 The bandwidth numbers are wrong by a unit error

> ### 🔴 DISCREPANCY 2 — GiB/s reported as GB/s, then compared against a GB/s peak
>
> **[MEASURED]** `p1_verify.py:87` computes `mb/median*1e6/1024`, i.e. **MiB → GiB/s**, and the
> README labels the column "achieved BW … **GB/s**".
>
> Re-derived, both conventions, against L40S peak **864 GB/s**:
>
> | arm | GiB/s (what was printed) | **GB/s (correct)** | % of 864 GB/s peak |
> |---|---:|---:|---:|
> | `dense` | 694.8 | **746.0** | **86.3%** |
> | `lowrank_fused r=128` | 160.5 | **172.4** | **20.0%** |
> | `lowrank_sep r=128` | 127.7 | **137.1** | 15.9% |
> | `grouped g=4` | 205.2 | **220.3** | 25.5% |
> | `lowrank_fused r=512` | 434.0 | **466.0** | 53.9% |
> | `grouped g=2` | 409.6 | **439.8** | 50.9% |
>
> So the arithmetic *from bytes and time* is internally consistent — 40 MiB in 56.224 µs **is**
> 694.8 GiB/s — but the **unit label is wrong** and the derived claim **"695 GB/s = 80% of L40S
> peak"** mixes a GiB/s numerator with a GB/s denominator (694.8/864 = 80.4%, which is where the
> "80%" came from).
>
> **CORRECTION:** dense achieves **746 GB/s = 86.3% of peak**; `lowrank_fused r=128` achieves
> **172 GB/s = 20.0% of peak**. **This makes the argument *stronger*, not weaker** — dense is even
> more thoroughly bandwidth-saturated than claimed, so there was even less headroom for a
> byte-reduction to buy time. Fix the numbers; keep the conclusion.

### 1.4 The "3.4 µs per extra kernel" derivation is arithmetically right and conceptually wrong

**[MEASURED]** (90.016 − 56.224)/10 = **3.3792 µs**. The subtraction is correct and the two arms are
genuinely iso-byte (both 40.0 MiB, verified by the assertion at `p1_verify.py:95`). ✅

> ### 🔴 DISCREPANCY 3 — 3.4 µs/kernel cannot be a marginal cost; it over-predicts *every* arm
>
> **[INFERRED]** If 3.379 µs were the per-kernel cost, the *floor* for each arm is `n_kernels ×
> 3.379`:
>
> | arm | kernels | floor implied by 3.379 µs/kernel | **measured** | verdict |
> |---|---:|---:|---:|---|
> | `dense` | 20 | 67.58 µs | **56.22 µs** | floor exceeds measured — **impossible** |
> | `lowrank_fused r=128` | 30 | 101.38 µs | **60.83 µs** | floor exceeds measured by 1.67× |
> | `lowrank_sep r=128` | 40 | 135.17 µs | **76.48 µs** | floor exceeds measured by 1.77× |
> | `grouped g=4` | 20 | 67.58 µs | **47.60 µs** | floor exceeds measured |
>
> **A constant that over-predicts every observation is not a marginal cost.** What the 40 MiB
> control actually varies is *not only kernel count*: `dense` is 2 × `[1024×1024]` GEMV per layer,
> while `lowrank_fused r=512` is 1 × `[1024→1024]` + 2 × `[512→1024]`. The **shape** changed, not
> just the count. And crucially the two extra kernels are the *narrow-contraction* up-projections.
>
> **The measurement that actually supports the conclusion is the µs/kernel column in §1.2:** every
> arm sits in **1.91–3.00 µs per kernel** while the bytes each kernel reads vary **8×** (0.25 MiB
> for an r=128 up-projection vs 2.0 MiB for a dense square GEMV). *That* is the finding: **at
> batch-1 on L40S, a GEMV kernel's cost is nearly independent of its size in this range, so time
> tracks kernel count and reducing bytes buys almost nothing.**
>
> **CORRECTION:** strike "~3.4 µs per extra kernel." Replace with: *"iso-byte at 40 MiB, splitting
> one square GEMV into a down- plus two narrow up-projections costs **+33.8 µs across 10 layers
> (+60%)**. Per-kernel time is 1.9–3.0 µs across all arms while per-kernel bytes vary 8×, so cost is
> set by kernel count and shape, not by bytes read."* Same conclusion, defensible derivation. The
> README's own next sentence ("the real cause is GEMV inefficiency") already contradicts the
> "per extra kernel" framing — the two sentences cannot both be right.

### 1.5 What survives §1

**[MEASURED, robust]**
- No factorized arm is faster than `dense` in either job, at any rank, fused or separate. **2/2 jobs,
  6/6 factorized configurations.** This is the load-bearing result and it holds.
- `r=512` saves literally zero bytes at d=1024 (`4dr = 2d²`) and is **60% slower** — a clean
  demonstration that the penalty is structural, not bandwidth-driven.
- `grouped` is faster than `dense` in both jobs (+20.3%, +15.3%) at identical cost to `lowrank r=128`.

**[MEASURED, NOT robust]** the specific figures −8.2%, +15.3%, 695 GB/s, 3.4 µs/kernel. All four need
the corrections above before they go in a writeup.

**Verdict on §1: P1's decode-latency claim is correctly dead. The evidence is over-stated in the
docs in four separate places, none of which reverses the conclusion, all of which are exploitable by
a reviewer.**

---

## 2. Is the spectra result a falsification? — Yes, and the docs are burying the better finding

### 2.1 The numbers, re-derived from `spectra_v2_results.json`

**[MEASURED]** 10 LIV layers, **32,768 tokens each**, `rank(Σ_x) = 1024` for **all 10** layers
(confirmed full — the v1 artifact is genuinely fixed). Means over layers, recomputed by me:

| tensor | plain eff.rank | aware eff.rank | ratio | E@64 aware | **E@128 aware** | E@256 aware | E@512 aware |
|---|---:|---:|---:|---:|---:|---:|---:|
| B (pre-gate) | 790.1 | 505.9 | 0.640 | 0.889 | **0.926** | 0.963 | 0.992 |
| C (post-gate) | 770.9 | 480.7 | 0.624 | 0.894 | **0.931** | 0.967 | 0.993 |
| **x (value stream)** — control | 790.5 | 507.8 | 0.642 | 0.887 | **0.925** | 0.963 | 0.992 |
| `out_proj` — control | 778.5 | 609.2 | 0.783 | 0.677 | 0.787 | 0.897 | 0.979 |
| random Gaussian — null | 824.2 | 646.3 | 0.784 | 0.649 | 0.760 | 0.879 | 0.972 |

Gates mean aware eff.rank = **493.30**; value = **507.81**. All README/design-doc figures reproduce
exactly. ✅ **No arithmetic error in the spectra work.** Convergence check also reproduces
(L0: 573.5 → 600.2 → 608.0 at 8k/16k/32k, still rising — numbers are a mild underestimate, as stated).

### 2.2 Yes, it falsifies the stated premise — and the parent's reframing is the right one

**[MEASURED]** B 0.926, C 0.931, **x 0.925**. The value stream is within **0.1–0.6 percentage points**
of the gates at every rank. Per-layer the tracking is even tighter than the means suggest — L0
B=0.822 / x=0.823, L9 B=0.971 / x=0.969, L15 B=0.953 / x=0.956. **These are not two distributions
that happen to have similar means; they are the same number layer by layer.** So:

> **"Gates are preferentially low-rank" is falsified. Correct. Do not re-litigate.**

But the parent's proposed reading is **the correct one, and it is materially more interesting than
P1**:

> **[INFERRED — and I endorse it]** The right conclusion is not "gates tolerate low rank." It is:
> **the activation-weighted input subspace of the LIV block is itself low-rank, and *every* d→d
> matrix reading that input inherits it.** The 0.926 / 0.931 / 0.925 triple is *the same measurement
> made three times* — all three tensors are row-blocks of one `in_proj` and are multiplied by the
> *identical* `Σ_x^{1/2}`. The controls prove the point rather than muddying it: `out_proj` reads a
> **different** input (post-conv, post-gate) and scores **0.787**; a random Gaussian under the same
> `Σ_x` scores **0.760**. So the ~14pp lift of {B, C, x} over the null is **a property of the block's
> input distribution**, and it is shared, not gate-specific.

**Three consequences the current design doc does not draw:**

1. **P1's own scope is the arbitrary part.** If B, C, and x are equally compressible, then
   factorizing *only* B and C is not motivated by anything in the data. The measurement says
   "factorize the whole `in_proj`." **[INFERRED]** At d=1024 factorizing the full `[3072, 1024]`
   `in_proj` as `d→r→3d` at r=128 saves **26,214,400 params = 7.40% of the model** vs P1's
   4.44% — **1.67× the saving, with a per-tensor fidelity that the probe already measured as
   equal.** And it *reduces* kernel count relative to P1: one `d→r` plus one `r→3d` is **2 kernels
   per layer vs the dense `in_proj`'s 1**, whereas P1-fused is 1 dense `in_proj` + 3 gate kernels if
   you keep the value stream dense. **This variant was never considered and it dominates P1 on both
   axes the project cares about.**
2. **The probe's own control structure argues against the design's control structure.** The design
   doc's controls for P1 (`N-narrow`, `S-shared`, `G-grouped`, `1G`) all ask *"is there a cheaper way
   to spend gate parameters?"* The spectra result asks a different and better question: *"is the
   gate the right place to spend the compression at all?"* **[INFERRED]** The natural control is
   therefore **`V-lowrank` (factorize the value stream at the same rank)** — which the plan does not
   contain. If `V-lowrank` matches `F-r128` in quality, the finding is *"in a LIV block, the
   compressible thing is the input, not the gate"* — a clean, quotable, falsifiable claim. If it
   loses, P1 has an actual gate-specific result for the first time.
3. **`out_proj` at 0.787 is a real signal that is being discarded.** It is 14pp *below* the
   in_proj tensors and only 2.7pp above a random matrix. **[INFERRED]** Its input is the
   post-conv-post-gate stream, whose covariance is far closer to isotropic. This says **the
   information bottleneck in the LIV block is on the way out, not on the way in** — the exact
   opposite of where P1 puts its compression. That is a publishable one-line observation from data
   already on disk.

### 2.3 Does the E@128 = 0.926 number actually support P1's feasibility?

**Partially, and less than the docs claim. [INFERRED]** Two caveats the current write-up omits:

- **0.926 is the Eckart-Young *optimum* for a trained matrix, not an achievable from-scratch
  result.** It answers "how well can rank-128 approximate this trained W," not "will a rank-128
  factor trained from scratch reach the same loss." The `structure_energy.py` docstring
  (lines 20-23) states this limitation correctly for `grouped`, but the design doc then quotes 0.926
  for low-rank without the same hedge. **Apply the hedge symmetrically.**
- **The 7.4% discarded energy has no calibration to a loss.** "Discards 7% of the energy that
  reaches the output" is not a quality claim until someone maps energy-loss → CE. **[MEASURED]**
  the same table says a **random Gaussian retains 0.760** — i.e. an *unstructured* matrix keeps 76%
  of the activation-weighted energy at r=128. So the trained gates' 0.926 is only **+16.6pp over an
  arbitrary matrix**, and 92.6% "sounds like" 92.6% good while its actual information content is
  "somewhat more compressible than a random matrix under the same input covariance." **Always quote
  0.926 against the 0.760 null, never against 1.0.**

**Verdict on §2: the falsification is sound and final. The honest reading is broader than the docs
allow and it points at a better experiment (`V-lowrank` / full-`in_proj` factorization) that the
plan does not contain.**

---

## 3. Is P1 worth GPU-days? — No, not at 12 arms

### 3.1 The strongest surviving P1 claim, stated precisely

Everything that is left, after §1 and §2, is exactly this:

> **[the surviving claim]** *At the LFM2-350M geometry, replacing each LIV gate projection `d→d`
> with `d→r→d` at r=128 removes **15,728,640 parameters (4.44% of the model)** and, on the released
> checkpoint, discards **7.4% of the activation-weighted energy** reaching the gate output (vs 24.0%
> for an unstructured matrix at the same rank). **Trained from scratch, does this cost quality — and
> does it cost more or less than simply making the model 4.7% narrower?***

Note what is *not* in it: no speed, no "gates are special," no novelty (Mamba, GLA, and every LoRA
derivative already factorize gates in production). **It is a controlled measurement of a known
technique in a specific architecture that nobody has ablated.** That is legitimate — it is the
project's own §8 framing decision — but it is a *small* claim, and it must compete for GPU-hours
against P2 and P3 on that basis.

### 3.2 Cost, computed at 40% MFU on A100 (`6ND / (n_gpu · peak · MFU)`)

**⚠️ [MEASURED] The plan is internally inconsistent about what stage 3a even is.** Three different
scopes appear in the same document:

| where | scope | runs | **A100-hours** | days on 8×A100 | L40S-hours |
|---|---|---:|---:|---:|---:|
| §8 budget table (line 1393) | 12 arms × 2 seeds, **150M / 10B** | 24 | **481** | 2.50 | 947 |
| §8 phase text (line 1379) | ~12 arms × **5** seeds, **350M / ~2B** | 60 | **568** | 2.96 | 1,119 |
| parent's brief | 12 arms × 2 seeds, 350M / 2B | 24 | **227** | 1.18 | 448 |

**These differ by 2.5× and nobody has reconciled them.** Note also the 150M/10B row in the budget
table contradicts the frozen "350M is the headline scale" decision — the rank sweep is scoped at a
*different model size* than everything it feeds. **Flag this as a plan defect independent of the P1
verdict.**

Minimal alternatives, same MFU assumption:

| option | runs | **A100-hours** | on 8×A100 | L40S-hours |
|---|---:|---:|---:|---:|
| **(b) 2 arms (`L0`, `F-r128`) × 8 seeds, 350M/2B** | 16 | **151** | 18.9 h | 298 |
| (b2) 3 arms (`L0`, `F-r128`, `N-narrow`) × 8 seeds, 350M/2B | 24 | **227** | 1.18 d | 448 |
| (b3) 2 arms × 8 seeds at 350M/**5B** | 16 | **379** | 1.97 d | 746 |

**[INFERRED] The right trade is (b2), not (a).** Same GPU-hours as the parent's reading of the full
sweep (227 h), but spent on **8 seeds of the three arms that matter** instead of 2 seeds of twelve.
Which brings us to why 2 seeds is the actual fatal flaw.

### 3.3 The KDA precedent makes the 2-seed sweep statistically void

**[MEASURED, `KDA/HANDOFF.md:167`]** "+8.92pp KDA>GDN at n=3 collapsed to **+2.01pp ns** at n=8."
**[MEASURED, `KDA/HANDOFF.md:158`]** the one parameter-matched LM pair is **+0.0053 nats** and
"needs n≈43 seeds."

**[INFERRED] What that implies for P1, quantitatively.** Back out the implied paired SD: for
+2.01pp to be non-significant at n=8 (t_crit 2.365) requires `s_δ ≳ 2.40pp`; for +8.92pp to be
significant at n=3 (t_crit 4.303) requires `s_δ ≲ 3.59pp`. So **`s_δ` for a seed-paired
architectural contrast in this repo's own measurements is ~2.4–3.6pp.** Take 2.85pp. Minimum
detectable effect at 80% power, α=0.05, paired:

| n seeds | MDE on a pp-scale endpoint (s_δ=2.85pp) | MDE on CE (s_δ=0.011 nat, the protocol's own figure) |
|---:|---:|---:|
| **2** | **5.65 pp** | **0.0218 nats** |
| 3 | 4.61 pp | 0.0178 nats |
| 5 | 3.57 pp | 0.0138 nats |
| **8** | **2.82 pp** | **0.0109 nats** |
| 16 | 2.00 pp | 0.0077 nats |

**[INFERRED] Now the expected effect size.** Scaling-law estimate of what a 4.44% parameter cut
*should* cost, using Hoffmann's `A/N^α` with A=406.4, α=0.34: **ΔCE ≈ +0.0078 nats**. The
`N-narrow` control differs from `F-r128` by **0.0145% of params → ΔCE ≈ +0.00002 nats**, i.e.
**structurally zero**.

Put those together and the conclusion is not close:

- **The `F-r128` vs `L0` contrast has an expected size of ~0.008 nats. n=2 detects 0.022 nats.
  The planned experiment is underpowered by ~2.8× on its own primary contrast.**
- **The `F-r128` vs `N-narrow` contrast — the one that decides whether P1 has any headroom at all —
  has an expected pure-capacity difference of 0.00002 nats.** Any observed difference is *entirely*
  an allocation effect, which is a real thing to measure, but the measurement needs **n≈34 seeds at
  s_δ=0.011 to resolve 0.0053 nats**, per this repo's own KDA arithmetic. **At n=2 it is not an
  experiment, it is a coin flip that will be reported as a result.**
- The plan's §8 kill rule already anticipates the answer: *"if P1's rank curve is flat and
  `N-narrow` matches it, the honest conclusion is 'spend the parameters wherever you like'."*
  **[INFERRED] That is the modal outcome, and it is predictable from the arithmetic above without
  running anything.** Both arms are 338.8M params, both are within 0.015% of each other, and the
  spectra probe already says the compressed subspace is not gate-specific.

### 3.4 The smallest experiment that tests the surviving claim

**Two arms, eight seeds, one rank. `L0` vs `F-r128`, at 350M / 2B tokens, WSD-forked.**
**151 A100-hours / ~19 hours on 8×A100 / ~298 L40S-hours.**

Rationale, point by point:
- **Drop the rank sweep entirely.** **[MEASURED]** r=512 saves **exactly 0 parameters** at d=1024
  (`4dr = 2d²` at r=d/2) — it is not a rung, it is `L0` with two extra kernels, and the design doc
  already admits this. r=256 saves 2.96%; r=128 saves 4.44%; r=64 saves 5.18%; r=32 saves 5.55%.
  **The whole sweep spans 4.44%→5.55% between r=128 and r=32 — 1.1pp of model params for a 4×
  rank change.** The parameter axis is saturated. A three-point sweep at 2 seeds costs 3× the
  compute to trace a curve whose y-axis barely moves, and §5.1 line 661 already reached this
  conclusion ("parameter savings saturate almost immediately") without acting on it.
- **Keep `N-narrow` only if you can afford n=8 on it too** (that is option b2, 227 A100-h). At
  n=2 it is worse than not running it: a null result at n=2 will be read as "N-narrow matches"
  when the experiment could not have detected a difference either way.
- **Drop `S-shared`, `1G`, `G-grouped` from the *training* budget.** **[MEASURED]** `G-grouped`
  already has a strong prior from `structure_energy.py` (0.130 retained vs 0.929, and *identical to
  a random mask of the same density* — 0.130 vs 0.130, with permutation spread [0.125, 0.133]).
  Spending 2 training runs to confirm a 7× energy deficit is a poor use of GPU-hours; if you want
  the grouped datapoint, take it at **one seed as a sanity check**, not as an arm.
- **Endpoints: MQAR + AR-Hits sliced perplexity, not held-out CE.** Already the project's decision
  (`HANDOFF.md:144-146`) and the power table above is the reason.

---

## 4. The parameter-efficiency claim — RESOLVED, both numbers are right, in different geometries

### 4.1 The ledger, reproduced exactly

**[MEASURED]** I rebuilt the LFM2 parameter ledger from the released module shapes
(`01_lfm2_architecture.md:147-155, 242-245, 341-343`) and it reproduces **both** published totals to
the parameter:

```
LIV mixer  = 3d² (in_proj) + d² (out_proj) + k·d (depthwise)   = 4d² + kd
GQA mixer  = 2·H·h·d + 2·G·h·d   (+ 2·h for per-head QK-norm)
MLP        = 3·d·F
per layer  += 2d (operator_norm + ffn_norm);  + d final norm;  embeddings V·d, TIED
```

| | **350M, d=1024, F=4608, H=16, G=8** | **1.2B, d=2048, F=8192, H=32, G=8** |
|---|---:|---:|
| computed total | **354,483,968** | **1,170,340,608** |
| published (HF) | 354,483,968 ✅ | 1,170,340,608 ✅ |
| embeddings | 67,108,864 — **18.93%** | 134,217,728 — 11.47% |
| 10 LIV mixers | 41,973,760 — **11.84%** | 167,833,600 — **14.34%** |
| 6 GQA mixers | 18,874,368 — 5.32% | 62,914,560 — 5.38% |
| 16 MLPs | 226,492,416 — **63.89%** | 805,306,368 — **68.81%** |
| one LIV mixer | 4,197,376 | 16,783,360 |
| gates `2d²` per mixer | 2,097,152 = **50.0%** of the mixer | 8,388,608 = 50.0% of the mixer |

> ### 🟡 DISCREPANCY 4 (minor but everywhere) — the "14% mixer / 69% MLP" split is the d=2048 model
>
> **[MEASURED]** `docs/liv-brainlift-experiment-design.md:498` prints
> `embeddings 11.5% | 10 LIV mixers 14.3% | 6 GQA 5.4% | 16 MLPs 68.8%` and `HANDOFF.md:354` says the
> MLP is **"69% of the model."** Both are the **1.2B / d=2048** figures. At the **frozen 350M
> geometry** the split is **embeddings 18.9% | LIV 11.8% | GQA 5.3% | MLP 63.9%**.
> Embeddings nearly *double* in share (11.5% → 18.9%) because `V·d` is linear in d while everything
> else is quadratic. The memory note `liv-experiment-key-numbers.md` ("mixer is 14% of the model")
> carries the same d=2048 figure. **Not load-bearing for any decision, but it is quoted as if it
> were the frozen geometry in at least three places.**

### 4.2 The 6.27% vs 4.44% conflict: **both correct, different d. HANDOFF is right; the memory note is stale.**

**[MEASURED]** Saving from factorizing both gates at rank r, across 10 LIV layers, is
`10 · (2d² − 4dr)`:

| r | **d=1024 (frozen, 350M)** | d=2048 (1.2B) |
|---:|---:|---:|
| 32 | 19,660,800 = **5.55%** | 81,264,640 = 6.94% |
| 64 | 18,350,080 = **5.18%** | 78,643,200 = 6.72% |
| **128** | **15,728,640 = 4.44%** | 73,400,320 = **6.27%** |
| 256 | 10,485,760 = **2.96%** | 62,914,560 = 5.38% |
| 512 | **0 = 0.00%** | 41,943,040 = 3.58% |

**Resolution:**
- **4.44% is the correct figure for this project.** It is `10·(2·1024² − 4·1024·128)/354,483,968`.
  `HANDOFF.md:511` and design-doc §5.1's correction table are **right**.
- **6.27% is the d=2048 value** and is *also* arithmetically correct — for the 1.2B model that was
  descoped. It survives in `docs/…:501` ("a 6.27% cut to the whole model at d=2048" — correctly
  labelled), in `docs/…:661` ("r=128 saves 6.27%, r=32 saves 6.94%" — **not** labelled, and this
  line is inside the rank-sweep guidance where it will mislead), and in the memory note.
- **There is no third number and no error.** The two docs are not in conflict; they are in different
  geometries, and one call site forgot to say so.
- **[MEASURED] Sanity check on the two derivations:** gate reduction is `2d²/(4dr) = d/(2r)` =
  **4× at d=1024**, 8× at d=2048 — matching HANDOFF exactly. And the *mixer*-level cut at r=128 is
  **37.5% at d=1024** (4,197,376 → 2,624,512), not the 43.7% quoted at d=2048. **A third
  d=2048-only figure ("the LIV mixer falls 44%", `docs/…:501`) also needs the geometry label.**

**ACTION: change `docs/liv-brainlift-experiment-design.md:661` to read "r=128 saves 4.44%, r=32
saves 5.55% — 1.1pp for a 4× rank cut" and update `liv-experiment-key-numbers.md`.** The
*argument* at that line (savings saturate immediately) is if anything **stronger** at d=1024.

### 4.3 Does anyone care about a 4.4% parameter saving? — **No, not on its own.**

**[INFERRED], three independent reasons:**

1. **`N-narrow` gets the same saving with zero mechanism.** **[MEASURED]** `F-r128` = 338,755,328
   params; `N-narrow` (d=976, F=4668) = 338,804,528 — a **0.0145% difference**. The narrow model is
   **4.69% narrower** (1024→976) and is a one-line config change with no init calibration, no
   spectral init, no LR-ratio tuning, no gate-bias confound, no extra kernels, and **no latency
   regression** (it is strictly *faster* — fewer bytes AND fewer kernels, unlike P1 which is fewer
   bytes and MORE kernels). **[INFERRED] Scaling-law prediction for the F-r128 vs N-narrow
   contrast: 0.00002 nats.** There is no capacity headroom for P1 to win on; it can only win on
   *allocation* — the claim that the same parameter budget is better spent as full-width-but-
   low-rank-gated than as uniformly-narrower. That is a genuine question, but it is a **second-order
   allocation question worth roughly 0.008 nats at most**, needing n≥8 to see.
2. **The saving does not convert to anything.** Not latency (§1, measured: it is a *regression*).
   Not memory in any regime that matters — 15.7 MB of bf16 weights is 4.4% of a model that already
   fits on a phone. Not FLOPs at any meaningful level: the design doc's own line 655 caps
   prefill-side speedup at **~8.8%** and the gates are only **8.1% of FLOPs**.
3. **It is not novel.** Mamba's `dt_proj`, GLA's forget gate, and the entire LoRA/SVD-compression
   literature already do this. The contribution can only ever be *"we ablated it in LFM2, which
   Liquid did not"* — which is the project's chosen framing and is fine, but it is worth **one arm
   and a table row**, not 12 arms and a rank sweep.

> **[INFERRED] Direct answer to the parent's question: if `N-narrow` matches, P1 has no headroom, and
> the arithmetic above says it is expected to match to within 0.00002 nats of pure capacity effect.
> The only surviving P1 result is an allocation result, and the study should be sized and framed as
> one — or cut.**

---

## 5. What would make P1 interesting again?

Ranked by claim strength per GPU-hour. All are **[INFERRED]** proposals built on the **[MEASURED]**
probes above; none requires new measurement to justify proposing.

### 5.1 ★ Best: factorize the **whole `in_proj`**, not just the gates — "compress the input, not the gate"

**The spectra probe's own control structure is the argument.** B, C, and x are three row-blocks of
one `[3072, 1024]` matrix, all multiplied by the same `Σ_x^{1/2}`, and all three retain
**0.925–0.931** at r=128. There is no measured reason to compress two of the three.

**[MEASURED] Arithmetic at the frozen geometry:** replacing `in_proj: d→3d` with `d→r→3d` at r=128
saves `10·(3d² − (d·r + r·3d)) = 10·(3,145,728 − 524,288) = 26,214,400` = **7.40% of the model** —
**1.67× P1's saving.** And on kernels it is **strictly better than P1**: the released block already
does the three streams in **one** GEMM, so the factorized version is **2 kernels/layer** where
P1-fused (dense value + factorized gates) needs **at least 3**, and the measured penalty in §1 is
per-kernel. **[INFERRED] This variant might actually be latency-neutral or positive**, which P1
provably is not — and testing that is a **one-hour L40S microbenchmark**, reusing `p1_launch_bench.py`
with one new `nn.Module`. That is the single highest-value next measurement in this whole track.

**Why it is a better *claim*:** "In a gated short-conv block, the compressible object is the block's
input subspace, not the gate. We show a single rank-128 bottleneck can feed all three streams, at
7.4% parameter reduction, and measure what it costs." That is architecture-level, mechanism-first,
and directly contradicts the brainlift's own framing — which makes it a *finding* rather than a
confirmation.

### 5.2 ★ Cheapest real upgrade: add **`V-lowrank`** as the control instead of `N-narrow`-only

One arm. Factorize the **value stream only**, r=128, gates left dense — the exact mirror of P1.
**[MEASURED]** the probe says it should perform identically (0.925 vs 0.926 retained energy).
**[INFERRED]** Whichever way it lands is a result:
- **Ties `F-r128`** → the compression is input-driven, gates are not special, and §5.1's whole-`in_proj`
  variant is licensed. *This is the predicted outcome and it is the interesting one.*
- **Loses to `F-r128`** → gates genuinely do tolerate rank-loss better than the value path *in
  training*, even though they do not in the trained-weight approximation metric. That is a
  **dissociation between the offline spectral proxy and the from-scratch result** — which is exactly
  the kind of methodological finding this project's KDA sibling produced and got value from.

Cost: **+8 runs = +76 A100-hours** on top of option (b). Total for `L0` / `F-r128` / `V-lowrank` /
`N-narrow` × 8 seeds at 350M/2B = **~303 A100-hours (~1.6 days on 8×A100)** — *less than the
originally budgeted 3a* and with 4× the statistical power.

### 5.3 The MLP is 63.9% of the model and nobody is touching it

**[MEASURED]** at the frozen geometry the 16 MLPs are **226,492,416 params = 63.89%**. A rank-128
factorization of all three SwiGLU matrices would save **196,083,712 = 55.3% of the model** — an order
of magnitude more than P1.
**[INFERRED] I do not recommend this as a P1 rescue**, for a reason the design doc already has and
which is dispositive: GaLore measured plain `W=BA` from-scratch **collapsing** at 1B (142.53 vs 15.56
ppl) at *more generous* rank fractions. The reason P1 escapes that verdict is precisely that it
factorizes ~7% of the parameters. **Scaling it to 55% walks straight into the known failure.** Worth
**one sentence in the writeup** as the scope boundary, not an arm. (It does, however, mean any
"parameter efficiency" framing that ignores the MLP is answering a question nobody asked — say so
explicitly, as `docs/…:503` already recommends.)

### 5.4 Joint activation-aware factorization (initialize from `Σ_x`, don't train from scratch)

**[INFERRED]** Every from-scratch low-rank result in the literature that works uses a
data-dependent init. The project already has `Σ_x^{1/2}` for all 10 layers computed and on disk
(implicit in `spectra_v2_results.json` / recomputable by `structure_energy.py` in minutes on CPU).
An arm that **initializes the r=128 factors from the activation-aware SVD of a short warmup run's
weights** and then trains, vs one initialized from spectral-init-plus-Frobenius-decay (Khodak, which
§5.1 line 685 already specifies), is a **1-arm, 8-seed** test of a mechanism nobody in this
literature has ablated in a gated conv. It is a better use of a marginal arm than a third rank rung.

### 5.5 What I would NOT do

- **Do not chase the latency claim on other hardware.** The mechanism in §1.4 (per-kernel cost
  1.9–3.0 µs while per-kernel bytes vary 8×) gets **worse** on faster cards — bandwidth rises,
  fixed per-kernel cost does not. H100 would be a stronger negative, not a rescue.
- **Do not promote `grouped`.** Correct call already in HANDOFF. **[MEASURED]** 0.130 retained,
  identical to a random mask at the same density.
- **Do not run r=512.** **[MEASURED]** zero parameter saving, +60% latency. It is not a data point,
  it is a strictly-dominated configuration.

---

## 6. Recommendation

**CUT the P1 rank sweep. Do not cut P1.**

Replace stage 3a with:

| arm | seeds | role |
|---|---:|---|
| `L0` | 8 | control |
| `F-r128` | 8 | P1, one rank |
| **`V-lowrank` r=128** | 8 | **NEW** — tests the actual spectra finding |
| `N-narrow` | 8 | "just build it narrower" |

**~303 A100-hours (~1.6 days on 8×A100)** — cheaper than every version of 3a currently written down,
with **4× the seeds** and an arm that can produce a *finding* rather than a confirmation.

**Plus one microbenchmark before any of it** (~1 GPU-hour on L40S): does the fused whole-`in_proj`
`d→r→3d` variant (§5.1) beat the released single dense `in_proj` at batch-1? It is 2 kernels vs 1
instead of P1's 3+ vs 1, and it is the only remaining configuration with a live latency story.
**Also add a *correct* `dense` control to that benchmark** — a single `Linear(1024, 3072)` — since
§1.1 establishes the current `dense` arm is not the released operator.

**Corrections to land in the docs regardless of what is run:**

| # | where | fix |
|---|---|---|
| 1 | `HANDOFF.md:201`, `docs/…:558`, `probes/README.md:24` | −8.2% → **"2–8% slower, sign robust across 2 jobs"**; state the 4–6% between-job drift (the −1.82% is in `p1_bench_results.json` already) |
| 2 | `probes/README.md:23-26`, `HANDOFF.md:209`, `docs/…:576` | GiB/s mislabelled GB/s. Dense = **746 GB/s = 86.3%** of peak; r=128 = **172 GB/s = 20.0%** |
| 3 | `HANDOFF.md:208`, `docs/…:575`, `probes/README.md:32` | strike **"~3.4 µs per extra kernel"** — it over-predicts every arm including dense. Replace with the per-kernel-time-vs-per-kernel-bytes framing (§1.4) |
| 4 | `docs/…:498, 501, 661`, `liv-experiment-key-numbers.md` | 14%/69%/6.27%/44% are **d=2048**. At the frozen d=1024: **11.8% / 63.9% / 4.44% / 37.5%** |
| 5 | `docs/…:1379` vs `docs/…:1393` | stage 3a is specified at two different scales and seed counts (150M/10B ×2 vs 350M/2B ×5). Reconcile |
| 6 | `probes/README.md:91`, `docs/…:745` | always quote E@128 = 0.926 **against the 0.760 random null**, not against 1.0 |

