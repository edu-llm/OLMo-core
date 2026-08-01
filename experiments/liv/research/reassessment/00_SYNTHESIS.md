# LIV track — first-principles reassessment (SYNTHESIS)

**Date:** 2026-08-01. **Author:** orchestrator of the reassessment team.
**Status:** IN PROGRESS — written incrementally. Sections marked ⏳ await a child agent.

Children and their files (disjoint ownership):

| file | topic |
|---|---|
| `01_p1_verdict.md` | Is P1 still worth GPU-days? |
| `02_topology_claim.md` | Is the topology result the real contribution? |
| `03_grouped_tension.md` | grouped-wins-latency / loses-energy — publishable? |
| `04_cheap_experiments.md` | What is answerable without the 8-day run |
| `05_deliverable_and_framing.md` | Deadline, competing tracks, strongest framing |
| `06_p2_p3_verdict.md` | P2 and P3 keep/narrow/cut |
| `07_risk_audit.md` | Plan-level contradictions, power, fail-open gates |

---

## PART A — Orchestrator's own verification (done before any child reported)

I re-derived every load-bearing number myself rather than trusting HANDOFF.md. Results below.
**MEASURED** = a number produced by a run whose artifact I can point at. **DERIVED** = I recomputed
it here from first principles. **CLAIMED** = asserted in a doc, not yet independently checked.

### A.1 The 350M parameter ledger reproduces EXACTLY — and pins the geometry

HANDOFF asserts `L0 = 354,483,968` but never states the geometry that produces it. I solved for it:

| component | formula | count |
|---|---|---:|
| Embedding (**TIED** — lm_head shares it) | 65,536 × 1,024 | 67,108,864 |
| MLP, 16 layers | 16 × 3 × 1,024 × 4,608 | 226,492,416 |
| LIV mixer, 10 layers | 10 × (4d² + kd) = 10 × 4,197,376 | 41,973,760 |
| GQA mixer, 6 layers | 6 × **3d²** | 18,874,368 |
| Norms | 16×2×1,024 + 1,024 + 6×2×64 | 34,560 |
| **Total** | | **354,483,968** ✅ |

**DERIVED, and this is new information not written down anywhere in the repo:**
- Embeddings are **tied**. (HANDOFF records that `llama_like`'s untied default caused a
  +67,108,864 overshoot; the fix is tying, but the doc never says so.)
- GQA is **3d² per layer, not the 2.5d² quoted in HANDOFF's "Verified sound" section.**
  2.5d² implies kv-width d/4 = 256 (4 kv heads). 3d² implies kv-width **d/2 = 512 → hkv=8, hd=64**,
  which is what "GQA-8 primary" says elsewhere. **At 2.5d² the ledger misses by exactly 3,145,728.**
  → **ERROR FOUND: the "GQA 10,485,760 (2.5d²)" line in HANDOFF is a d=2048 (1.2B) number reused in a
  d=1024 context.** It is right for the brainlift's original 1.2B arithmetic, wrong for the frozen arm.

### A.2 P1's parameter saving is 4.44%, and 6.27% appears in three places where it is wrong

P1 factorizes the two gate projections (B pre-gate, C post-gate), which are 2d² of the LIV mixer's 4d².

At d=1024, r=128: `2·(2dr) = 4dr = 524,288` replaces `2d² = 2,097,152` → saves 1,572,864/layer
× 10 layers = **15,728,640 params**.

Cross-check against the committed arm builder: `L0 − F-r128 = 354,483,968 − 338,755,328 =
15,728,640`. ✅ **Exact match — the arm builder and my derivation agree independently.**

**15,728,640 / 354,483,968 = 4.437%.**

**ERROR FOUND (systematic).** The memory note `liv-experiment-key-numbers.md` contains the correction
("4.44%, not 6.27%") in one bullet while three *other* bullets still quote the superseded d=2048
figures as if they applied to the frozen geometry:

| claim as written | true at frozen d=1024 | where the wrong number came from |
|---|---:|---|
| "a 44% mixer cut … is a **6.27%** model cut" | **4.44%** | d=2048 / 1.2B |
| "the LIV mixer is only **14.3%** of the model" | **11.8%** | d=2048 / 1.2B |
| "MLPs are **68.8%**" | **63.9%** | d=2048 / 1.2B |
| "savings saturate (r=128 → **6.27%**, r=32 → **6.94%**)" | 4.44% / 4.91% | d=2048 / 1.2B |

This is the *same error class* the team already caught once (the "4.72 µs break-even computed at
d=2048"). It has not been swept for. **Every d=2048-era number in the dossier should be re-audited.**

Consequence: P1's headline saving is **4.44% of parameters**, i.e. the entire proposal is worth less
than the difference between two rounding choices on the SwiGLU width. This materially weakens the
"parameter efficiency" survival story that the plan retreated to after the latency claim died.

### A.3 The topology crossover T≈4,121 is CORRECT — but only under one specific definition

KV bytes/token (bf16, 2 B/elem):
- `L0`: 6 attn layers × 2 (K,V) × 512 × 2 B = **12,288 B/token = 12 KiB** ✅ (matches claim)
- `A16-P`: 16 × 2 × 512 × 2 = **32,768 B = 32 KiB** → saving **20 KiB/token** ✅ (matches claim)
- Scale-invariance ✅ **DERIVED**: depends only on (n_attn_layers, hkv, hd), not d. True.

Decode traffic at context T (weights in bf16, all weights read every step, lm_head tied so read once):
- L0: 708,967,936 + 12,288·T
- A16-P: 708,777,984 + 32,768·T

Solving for a "10% decode-traffic win":

| definition | equation | T |
|---|---|---:|
| saving as fraction of **A16-P** traffic | 20,480T / (708,777,984 + 32,768T) = 0.10 | **T = 4,120** ✅ |
| A16-P is 1.10× L0's traffic | (708,777,984+32,768T) = 1.10(708,967,936+12,288T) | T = 3,693 |

**The claimed 4,121 reproduces to 1 part in 4,000 under the first definition.** ✅ The doc never states
which definition it used; it should. Both are in the same ballpark, so the claim survives either way.

Corroborating checks, all ✅:
- KV = 6.63% of decode traffic at T=4,096 (claimed 6.6%)
- KV = 36.2% at T=32,768 (claimed 36.2%)
- KV read = weight read at T = 708,967,936/12,288 = **57,696** (claimed 57,690 — rounding)
- 12,288 × 32,768 = 402,653,184 B = **384 MiB** at 32K (claimed 384 MiB)

### A.4 ⚠️ A consequence nobody in the repo has drawn: quantization moves the crossover 3.3× closer

Every crossover above assumes **bf16 weights**. But the only real decode profile the project has
(`MatMulNBits` 91.2% / `Conv` 1.0%) was measured on a **q4** ONNX build — and q4 is the *deployment*
regime for LFM2, which is an on-device family. Weight traffic drops ~4× while KV traffic does not.

**DERIVED** at ~0.6 B/param effective (int4 + scales): weight bytes ≈ 212.7 MB, so
20,480T / (212,690,381 + 32,768T) = 0.10 → **T ≈ 1,236.**

| regime | KV = 10% of A16-P traffic | KV = weight traffic |
|---|---:|---:|
| bf16 weights | T ≈ 4,120 | T ≈ 57,700 |
| **q4 weights (deployment)** | **T ≈ 1,236** | **T ≈ 17,300** |

**This is the single strongest un-exploited fact in the dossier.** The project's own framing —
"mostly-LIV saves memory/latency is a *long-context* thesis, largely invisible at trainable contexts"
— is an artifact of assuming bf16. At the quantization LFM2 actually ships in, the topology win is
already 10% at **T ≈ 1,236**, well inside the 4K training context. It reframes the topology arm from
"a claim we cannot demonstrate at affordable context" to "a claim that is *most* visible exactly where
the model is deployed." Requires no new training to state; requires a decode-path implementation to
*measure*.

### A.5 The P1 latency benchmark — I reverse-engineered its scope, and it changes the reading

Nothing in HANDOFF, the probe README, or the design doc states **what subgraph** the 56.2 µs covers.
I recovered it from the reported bytes and kernel counts, and every arm checks out:

| arm | reported | derived from geometry | ✓ |
|---|---|---|---|
| dense | 20 kernels, 40.0 MiB | 10 layers × 2 gates = **20**; 10 × 2d² × 2 B = 41,943,040 B = **40.0 MiB** | ✅ |
| lowrank_fused r=128 | 30 kernels, 10.0 MiB | 10 × (1 down + 2 up) = **30**; 10 × 4dr × 2 B = **10.0 MiB** | ✅ |
| lowrank_sep r=128 | 40 kernels, 10.0 MiB | 10 × (2 down + 2 up) = **40**; same bytes | ✅ |
| grouped g=4 | 20 kernels, 10.0 MiB | 10 × 2 = **20**; 10 × 2 × d²/4 × 2 B = **10.0 MiB** | ✅ |
| lowrank_fused r=512 (iso-byte) | 30 kernels, 40.0 MiB | down d→1024 = d², ups 2×(512×1024) = d² ⇒ 2d² = dense | ✅ |

**So the benchmark measures the ten LIV layers' GATE PROJECTIONS ONLY, batch 1, bf16.** It is not a
model, not a mixer, not even a block. That is a perfectly good microbenchmark — but:

> **⚠️ FRAMING ERROR, and it is in the headline of three documents.** "P1 is 8.2% slower than stock
> LIV" reads as a model-level number. The gates are `10 × 2d² = 20,971,520` params = **5.92% of the
> 350M model**, and (cross-checked against the project's own ONNX profile: MatMulNBits 91.2%) about
> **5.4% of decode time**. So end-to-end: **P1 r=128 costs ≈ 0.4–0.5%** and **`grouped` gains
> ≈ 0.8–0.9%**. Both are inside end-to-end run noise. *Neither the death of P1's latency claim nor
> grouped's "win" is a systems result anyone would act on at the model level.* The qualitative
> conclusion (factorizing does not buy decode time) is sound and important; the 8.2%/15.3% magnitudes
> must always be labelled "gate subgraph."

**ERROR FOUND — the bandwidth figures are GiB/s mislabelled as GB/s, and the "80% of peak" is a
unit mismatch.** I checked all four:

| arm | bytes/time | GB/s (10⁹) | GiB/s (2³⁰) | doc says |
|---|---|---:|---:|---:|
| dense | 41,943,040 B / 56.2 µs | **746.3** | **695.1** | "695 GB/s" |
| lowrank_fused r=128 | 10,485,760 / 60.8 µs | 172.5 | **160.6** | "161 GB/s" |
| lowrank_sep | 10,485,760 / 76.5 µs | 137.1 | **127.7** | "128 GB/s" |
| grouped | 10,485,760 / 47.6 µs | 220.3 | **205.2** | "205 GB/s" |

All four are GiB/s. The doc then divides the GiB/s numerator by L40S's 864 **GB/s** spec sheet to get
"80% of peak." Correct is **746/864 = 86.4%**. This is a *self-harming* error — it makes the dense
baseline look less saturated than it is, and the true 86% figure **strengthens** the paper's argument
that dense is genuinely bandwidth-bound. Fix the units and raise the claim to 86%.

**The stated mechanism is backwards in magnitude — and fixing it produces a better result.**
The docs say: *"The analytic model blamed launch overhead; the real cause is GEMV inefficiency."*
I decomposed the r=128 fused arm against dense using only measured quantities:

| term | µs | how obtained |
|---|---:|---|
| dense baseline | 56.2 | MEASURED |
| − bytes saved (30 MiB at dense's achieved 746 GB/s) | −42.2 | DERIVED from measured |
| + 10 extra kernels × 3.38 µs (from the **iso-byte** control) | +33.8 | MEASURED (90.0−56.2)/10 |
| + residual = skinny-GEMV inefficiency | **+13.0** | INFERRED (balance) |
| **= lowrank_fused r=128** | **60.8** | MEASURED ✅ |

**Kernel count is the dominant term (33.8 µs), GEMV shape is the minority term (13.0 µs) — 2.6× the
other way round from what the docs assert.** And the attribution is clean, because the iso-byte
control runs at r=512 where *no* matrix is skinny (min dimension 512), so its 3.38 µs/kernel isolates
dispatch cost from shape.

Two consequences the project has not drawn:
1. **The analytic launch model was not wrong — it was right about the mechanism and wrong about the
   constant.** It predicted break-even at 2.43 µs/launch un-fused, 4.85 µs fused; the measured cost is
   3.38 µs. That correctly predicts the un-fused arm loses. It predicts the *fused* arm should win,
   and the 13 µs GEMV residual is what flips it. So the honest lesson is **"kernel count dominates,
   and shape penalty is what tips a marginal case"** — not "launch overhead was a red herring."
2. **This directly explains why `grouped` wins, and it is the interesting part.** Grouped keeps
   dense's 20 kernels while cutting bytes 4×, so it pays *zero* dispatch penalty. It still misses
   roofline badly (205 GiB/s vs dense's 695), i.e. it eats ~33.6 µs of shape penalty of its own — it
   wins purely by not adding kernels. **Generalized lesson, better than the one currently written
   down:** *at decode, the cheapest structure is the one that preserves the kernel graph, not the one
   that removes the most bytes.* That is a crisp, portable, already-measured claim.

Percentage checks, all ✅: 8.19% / 36.1% / 15.30% / 3.38 µs.

### A.5b 🔴🔴 THE HEADLINE RESULT IS CONFOUNDED — the benchmark's working set fits in L2 cache

This is the most important finding in this reassessment. **MEASURED on the actual hardware** (FarmShare
job 1671407, `torch.cuda.get_device_properties`):

```
name NVIDIA L40S    L2_cache_size_bytes 100663296  = 96.0 MiB
```

**The L40S has 96 MiB of L2 cache. The benchmark's largest arm has a 40 MiB working set.**

Reading `p1_launch_bench.py::time_stack`: it builds 10 small modules (40 MiB of weights total for
dense), warms up 50 iterations, captures **one** CUDA graph, then replays it 300 times, timing each
replay. Nothing between replays evicts anything. **After the first replay every weight is L2-resident
and stays resident for all 300 timed iterations.** Every arm (10, 20, and 40 MiB) is comfortably
inside 96 MiB. **The benchmark never touches HBM after warm-up.**

Therefore **every bandwidth statement in the project's headline is invalid**:

| claim, appears in 3 documents | status |
|---|---|
| "dense achieves 695 GB/s" | measuring **L2**, not HBM |
| "80% of L40S peak — genuinely bandwidth-bound" | ❌ compares an L2-resident rate to the **HBM** spec sheet |
| "skinny GEMVs cannot saturate the memory system" | ❌ nothing in this benchmark reaches the memory system |
| "for any decode-time factorization, run an iso-byte control before believing a roofline estimate" | the advice is good; **this** roofline reading is itself the error it warns about |

**The smoking gun was already in the raw JSON and nobody analyzed it.** From
`p1_verify_results.json`, two arms with **identical kernel counts (20) and a 2× byte difference**:

| arm | MiB/tok | median µs |
|---|---:|---:|
| `grouped g=2` | **20.0** | 47.680 |
| `grouped g=4` | **10.0** | 47.600 |

**Halving the bytes changed the time by 0.17%.** A bandwidth-bound benchmark *cannot* produce that.
This single pair falsifies the roofline interpretation of the whole probe, and it is sitting
unremarked in the committed results file. (Consistent with the L2 rate being ~4 TB/s: at 20 kernels /
56.2 µs, dense spends **2.81 µs per kernel** while its 2 MiB/kernel would take ~0.5 µs at L2 speed —
so the benchmark is dominated by **per-kernel fixed cost**, not by data movement, exactly as the
grouped pair shows.)

**Direction of the bias is unambiguous, and it runs AGAINST P1.** Real batch-1 decode reads ~709 MB
of weights per token — 7.4× the L2 — so it is genuinely HBM-bound. Bytes cost ~4.6× more time there
(HBM ~0.86 TB/s peak vs L2 ~4 TB/s) while extra-kernel cost is roughly regime-independent. **A test
run entirely in cache is the one regime in which byte savings buy the least. P1 was killed by a
benchmark constructed so that its central benefit could not appear.**

**Corrected end-to-end estimate (INFERRED — a model, not a measurement):**

| term | value |
|---|---:|
| decode token, 709 MB at ~700 GB/s achieved | ~1,013 µs |
| gate bytes today (41.9 MB) | ~60 µs |
| gate bytes at r=128 (10.5 MB) | ~15 µs |
| **saved** | **~45 µs** |
| cost of 10 extra graphed launches (1–2 µs true dispatch … 3.38 µs worst case) | −10 to −34 µs |
| **net** | **+11 to +35 µs = +1.1% to +3.5% FASTER** |

**So the sign plausibly flips.** And note that even taking the flawed benchmark entirely at face
value, its 4.6 µs gate-subgraph penalty is **0.45%** of a real token, not 8.2%. **Both the sign and
the magnitude of the project's most-repeated result are wrong.**

**This is cheap to settle and it is the single highest-value experiment available.** Add an L2 flush
between graph replays (write a >96 MiB scratch buffer, the standard technique) or size the stack past
96 MiB, and re-run. ~15 lines changed, one ~10-minute L40S job, and it decides whether P1 lives.

### A.5c 🔴🔴🔴 CONFIRMED BY MEASUREMENT — P1'S LATENCY CLAIM IS NOT DEAD. THE SIGN FLIPS.

I ran the test. **FarmShare job 1671420, L40S, CUDA-graphed, 3 trials × 60 timed replays per rung.**
Script: `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/reassessment/p1_scaled.py`
(also at `/scratch/users/ericrcwu/liv/p1_scaled.py`, results
`/scratch/users/ericrcwu/liv/p1_scaled_results.json`).

**Method.** Identical arms and identical kernel-count ratios (20/30/40) to the original probe. The
*only* variable is the working-set size: stack more layers so the weights genuinely exceed the 96 MiB
L2, which is what the real model does. (A first attempt used an explicit L2 flush; that was wrong —
zeroing a buffer leaves dirty lines whose writeback bleeds into the timed region and added a uniform
+26 µs to every arm. Scaling the working set is the clean control. Both attempts are recorded.)

| working set | regime | dense | `lowrank_fused r=128` | `lowrank_sep r=128` | `grouped g=4` |
|---|---|---:|---:|---:|---:|
| **40 MiB** (the original probe) | **L2-resident** | 56.32 µs | 57.34 µs → **−1.82%** | 72.64 → −28.98% | 44.93 → **+20.23%** |
| **320 MiB** | **exceeds L2** | 728.06 µs | 433.95 µs → **+40.40%** | 553.98 → **+23.91%** | 331.78 → **+54.43%** |
| **960 MiB** (≈ real 709 MB decode) | **exceeds L2** | 2180.06 µs | 1497.09 µs → **+31.33%** | 1904.54 → **+12.64%** | 1142.78 → **+47.58%** |

**Every factorized arm goes from LOSING to WINNING the moment the working set leaves cache.**
`lowrank_fused` swings from −1.82% to **+31…40% faster**. Even `lowrank_sep` — the naive 40-kernel
version the project called "−36%, not recoverable by tuning" — is **+12.6% faster** in the real
regime.

**My 40 MiB rung reproduces −1.82%, matching job 1670883's stored `speedup_graphed_pct: -1.818` to
three decimals.** So the reproducible L2-resident value is −1.8%, and **the −8.2% figure promoted to
the headline of three documents is the outlier of the two jobs**, not the replication it is described
as. (Child `01_p1_verdict.md` found the same thing independently from the JSON.)

**The mechanism, corrected.** The "skinny GEMVs underperform" observation was *real* — at 960 MiB
`lowrank_fused` still achieves only 168 GB/s vs dense's 462 GB/s. That part of the analysis was right.
What was wrong is the conclusion drawn from it: **a 4× byte reduction beats a 2.7× bandwidth-efficiency
penalty once there are actually HBM bytes to save.** In cache there are none, so only the penalty
showed up. Note dense achieves 745 GB/s in-cache but 462 GB/s out-of-cache — the "80% of peak,
genuinely bandwidth-bound" claim described the *cache*, and dense is in fact only ~53% of HBM peak.

**End-to-end (INFERRED from the above):** gates are 41.9 MB of the ~709 MB read per token = 5.9%, so a
31–40% gate-subgraph win is **≈ +1.8% to +2.4% on decode**. Small — but *positive*, where the record
says −8.2%. **The sign, the mechanism, and the recommended action were all wrong.**

**Caveats, stated honestly.** (1) This is still the gate subgraph, not a model; the gold-standard test
is a full-model decode A/B, which needs the incremental decode path that does not yet exist. (2) My
rungs stream *only* gate weights; in the real model gates are interleaved with 667 MB of other
traffic, so gates arrive cold (captured here) but dispatch is amortized differently (not captured).
(3) Nodes were shared — though the 1671420 rungs are self-consistent and the 40 MiB rung reproduces a
prior job exactly. (4) `grouped` still wins on latency in *both* regimes, so the grouped-vs-lowrank
quality question (child `03_grouped_tension.md`) is unaffected and remains live.

**What this changes:** "strike the decode-latency claim from P1" — the standing instruction in
HANDOFF, the design doc, both memory notes and the probe README — is **based on a measurement artifact
and must be reversed.** P1's latency claim is *back on the table, with a positive but modest measured
effect.*

### A.6 What this all implies before any child has reported

Three of the project's four load-bearing quantitative headlines are **subgraph-scoped or
regime-scoped** in ways the documents do not disclose:
- "P1 8.2% slower / grouped 15.3% faster" → **gate subgraph only; ≈0.5%/0.9% end-to-end.**
- "P1 saves 6.27% of params" → **4.44%** at the frozen geometry (6.27% is a d=2048 number).
- "the topology win needs T≈4,121" → **true at bf16; T≈1,240 at the q4 the model actually ships in.**

None of these are fatal, but all three change what is worth doing next.

### A.6 Compute-budget arithmetic reproduces — but the machine to run it on does not exist

Using 6ND at 8×A100, 40% MFU (= 9.984e14 FLOP/s aggregate):

| stage | FLOPs | wall-clock | doc claim |
|---|---:|---:|---|
| Rank: 24 runs × 150M × 10B | 2.16e20 | 60.1 h = **2.5 d** | 2.5 d ✅ |
| Confirm: 5 × 350M × 20B | 2.10e20 | 58.4 h = **2.4 d** | 2.5 d ✅ |
| Headline: 3 × 750M × 50B | 6.75e20 | 187.8 h = **7.8 d** | 8 d ✅ |

So the "~8 days if 350M is the headline" figure is internally sound. **But:**

**⚠️ THE PROGRAM HAS NO HOST.** 8 days on 8×A100 = **1,536 A100-hours**. The two compute sources in
the plan are (a) FarmShare, which gives **1 L40S, 6-hour wall-clock, per job**, and (b) "SB-AWS",
which appears in the plan as an assumption, not as a provisioned resource. On FarmShare, 1,536
A100-hours ≈ 1,300–1,900 L40S-hours (L40S has more dense bf16 TFLOPs than A100 but **half the HBM
bandwidth**, 864 vs 1,555 GB/s, so MFU on a memory-bound depthwise-conv hybrid will be *worse*, not
better) — which is **220–320 sequential 6-hour jobs on a single GPU.** At even 4 concurrent jobs that
is 2–3 weeks of pure queue time with zero failures, on a shared cluster, with no checkpoint-resume
story written. **INFERRED, high confidence: the 8-day plan is not executable on the compute the human
demonstrably has.** This should have been the plan's binding constraint from the start and it is not
listed as a risk anywhere.

Every recommendation below is filtered through this: **prefer work that fits in 6-hour single-L40S
jobs, or needs no training at all.**

---

## PART B — Ranked recommendations

⏳ Draft below; will be revised as children report.

*(placeholder — see PART C once children land)*

---

## PART C — Child findings

### C.1 P1 (`01_p1_verdict.md`) — CUT the rank sweep; one arm survives; a better variant exists

- **The replication did NOT replicate the effect size.** Job 1670883 gives **−1.82%** for the same arm
  that job 1670884 gives −8.2%. The field `speedup_graphed_pct: -1.818` is in the committed
  `p1_bench_results.json` that the README cites. Between-job drift 4–6% on every arm but `dense`. The
  advertised "≤0.3% spread" is *within-job back-to-back replays* and bounds nothing.
  → Honest claim: **2–8% slower on the gate subgraph, sign robust 2/2 jobs.** Combined with my L2
  finding (A.5b), even the sign is now in question.
- **695 GB/s is GiB/s** (`p1_verify.py:87` divides by 1024, README labels GB/s, then compares to a
  GB/s peak). Independently reproduces my A.5 finding. Correct: dense **746 GB/s = 86.3% of peak**.
- **"3.4 µs per extra kernel" is not a marginal cost**: 20 × 3.379 = 67.6 µs floor for dense, which
  measured 56.2. It over-predicts every arm. Confirms my A.5 objection.
- **The `dense` control is not stock LIV.** Released `Lfm2ShortConv` does ONE fused `(3d,d)` GEMM; the
  benchmark uses two separate `d→d` and omits the value stream. This *pessimizes* the control.
- **6.27% vs 4.44% is NOT an error — it is a mislabel.** 6.27% is correct at d=2048, 4.44% at the
  frozen d=1024. Both HANDOFF and I are right; `docs/…:661` and the memory note quote the d=2048
  figure without saying so. (Refines my A.2 — downgrade "error" to "unlabelled scope".)
- **The best idea in the reassessment so far:** the spectra show B/C/x collapse *identically* because
  they read the same `x`. So gate-only scope is arbitrary — factorize the **whole `in_proj` as
  `d→r→3d`**: saves **26,214,400 = 7.40% of the model (1.67× P1)** at **2 kernels/layer vs P1's 3**.
  It is the one variant with a live latency story and it was never considered.
- **Power:** `F-r128` vs `N-narrow` differ by 0.0145% params ⇒ ΔCE ≈ 0.00002 nats. Whole 4.44% cut
  ⇒ ~0.008 nats expected. With s_δ ≈ 2.4–3.6pp backed out of KDA, **n=2 detects 0.022 nats —
  underpowered ~2.8× on P1's own primary contrast.**
- Stage 3a is specified at **two different scales in the same document** (150M/10B×2 = 481 A100-h vs
  350M/2B×5 = 568 A100-h), and the 150M version contradicts the frozen headline scale.
- Recommended replacement: `L0` / `F-r128` / `V-lowrank` / `N-narrow`, 8 seeds, 350M/2B ≈ **303
  A100-h (~1.6 d on 8×A100)** — cheaper than every written 3a with 4× the power.

### C.2 P2 + P3 (`06_p2_p3_verdict.md`) — CUT BOTH. ~650–800 A100-hours saved (~25% of the program)

- **Citation audit: no hallucinations.** 2607.18413 and 2606.03825 both real and accurately
  described. Hymba row C→D verified. Only `num_kv_shared_layers=15` for Gemma 3n is UNVERIFIED
  (config gated, HTTP 401) and should be marked so.
- **P2 dies on novelty, not economics.** The claim "no cross-layer-sharing paper reports
  needle/passkey/MQAR" — the designated *headline* — is **now false**. arXiv 2606.06467 (MSRA,
  2026-06-04) ran RULER × 12 subtasks × {16K, 32K} on from-scratch 4B KV-sharing models against a
  matched control: **more thorough than the capstone's plan**, and it lands **opposite** to the
  motivating worry (**+6.1 RULER avg at 32K**, gains concentrated in hard multi-needle subtasks). The
  P2 literature sweep is ~8 months stale and misses 7 relevant papers.
- **P3 dies on evidence the child generated** — a 2-second, zero-GPU-hour read of the released
  LFM2-350M conv weights on the login node, with **five pre-registered predictions written to disk
  first**. All five confirmed by 16–17× margins:
  - boundary tap holds **1.4%** of median per-channel energy (random-init control 30.0%)
  - only **2.08%** of channels have the oldest tap largest (control 34.2%)
  - decay ratio **0.083**, 4× steeper than predicted from Sieberling
  → **k=3 is nowhere near binding, measured *inside* Liquid's double gate** — which was P3's last
  defense (design doc §5.3 item 0a). This is exactly the cheap-decisive-experiment pattern that
  killed P1's latency claim, and it cost nothing.
  - Two bonus findings: the conv **de-activates monotonically with depth** (98.2% off-current energy
    at layer 0 → **2.4%** at layer 15 — vestigial), and **layers 0–1 are pure lag-1 delay lines**
    (0.99/0.93 median energy on tap t−1).
- **P3's steelman is also dead.** The RepVGG-style reparameterization question the docs propose as the
  salvage **has already been run**: Tian et al.'s **Table 7 is literally that experiment** (branches
  trained, fused at inference), 12.79 → 13.28. The docs describe it only as "mixed widths," concealing
  that it is the fusion experiment.
- Verified: dilations {1,2,4,7} reach exactly 7 of 15 lags; the 5× state correction is right.
- Replacement: promote `A-fewer3` into a topology/ratio ablation; publish the tap-energy analysis
  as-is; run a **`k1` narrowing** arm instead of `k15` (a narrower kernel is predicted free on deep
  layers — more surprising than the original proposal).

### C.3 Deliverable + framing (`05_deliverable_and_framing.md`) — there is no deadline, and the human has a finished paper he isn't writing

- **No deadline exists anywhere in the repo.** Exhaustive search: all 5 HANDOFFs, `docs/`,
  `handoffs/`, `.claude/`, all 16 memory files, every venue name, date patterns through Dec 2026.
  **Zero commitments. No venue or audience is named either** — so the HANDOFF has been making
  "what is publishable" decisions against an undefined target. That is itself a finding.
- **`KDA/HANDOFF.md` says "COMPLETE… The science is done. Only committing and write-up remain."**
  Novel Triton kernel in no library, 6-level verification chain with an 18/18 mutation-testing
  negative control, **four result families at n=5–8 with CIs**, a solvability control that kills the
  difficulty confound, and it already self-retracted an n=3 result that collapsed at n=8.
  **Distance to submittable: days, $0.** LIV: zero trained models, ~3,000 A100-hours to go.
- Child's recommendation: **the axe should fall on KDA-LIV/CORE-6, not LIV** ($965, one gate already
  failed by 19×, governing factor `A` unmeasured, and it asks about a mechanism whose own track is
  COMPLETE).
- **The design doc misread the source PDF in five ways.** Most important: **P1 was a parameter-count
  claim, not a latency claim** — the single latency sentence is Eric's own *conditional* ("*before*
  the parameter reduction becomes a latency reduction"). The benchmark **discharged his condition and
  answered it**; that is executing the proposal, not refuting it. Also: Eric pre-wrote ~70–80% of the
  "adverse findings" himself; hardware priority was **inverted** (his p.6 says edge/CPU primary with
  GPU as a *control*; the design cut edge entirely); and **§2 "Learning Science as an Inspiration" is
  22% of his document and has no counterpart in 1,330 lines of design or 14,600 lines of research.**
  His actual thesis is *"a mostly-LIV hybrid separates two jobs"* — a **division-of-labour
  hypothesis**, not an efficiency paper.
- **Correction to my own brief:** "bytes ≠ time" is **not novel as a phenomenon** — the repo's own
  `02_lowrank_gates.md` §5B already documents FLAR-SVD measuring 2× param cuts running *slower*.
  What is new is **the control** (iso-byte, achieved-bandwidth attribution, graphs on). Frame as
  "the missing control," never "a refutation."
- **The floor:** a 6–10 page methods-and-measurement report, 2–5 days of writing, $0, 11 measured
  results, zero trained models. Honest grade: solid B+/weak-A capstone, plausible workshop short paper
  *if* the iso-byte control is the headline. **This floor is below what could already be written from
  `KDA/` for the same zero dollars.**

### C.4 Topology (`02_topology_claim.md`) — arithmetic CONFIRMED, novelty verdict is harsh

- **Every headline number re-derived from the live HF config and confirmed**: KV 12/32/20 KiB;
  6.63% @4K; 36.22% @32K; KV==weight at T=57,696; **T = 4,121.14 to 4 sig figs**; `L0` =
  354,483,968 matching HF safetensors; `A16-P` SwiGLU width 4,820 re-solved by brute force. My own
  A.3 derivation agrees independently.
- 🔴 **BUG IN COMMITTED CODE — I verified this myself.** `short_conv.py:373` uses
  `linear_flops = 2 * params` (forward-only) where the library's own `Attention` uses
  `param_flops = 6 * ...` with the comment *"6 FLOPs per parameter (2 ops * 3 for forward+backward)"*.
  ShortConv is **3× under-counted**. Effect: **the reported compute gap 1.297×/1.959× should be
  1.207×/1.886×.** The committed test cannot catch it because it only compares arms with equal
  ShortConv counts — and `A16-P` has zero. **Must be fixed before solving any compute-matched arm.**
  (One correction to the child: it said "every sibling uses 6×"; `recurrent.py:298` also uses `2 *`,
  so this is a *pre-existing library inconsistency* affecting two mixers, not a one-line slip. That
  makes it more important, not less — `num_flops_per_token` is the quantity the plan says to match on.)
- **Factual error:** `06_baselines_infra.md:118` says "No LFM2 paper exists (only a blog post)." It
  exists — **arXiv:2511.23404** — and `01_lfm2_architecture.md` cites it correctly. The repo
  contradicts itself. Reading the paper *upgrades* the gap claim from inferred to verified: no ratio
  ablation, no kernel-width ablation, no recall benchmark, and the "matches attention-heavier
  baselines" claim is prose with no supporting table.
- `crossover.py` is hard-coded to **d=2048** and does not compute the numbers HANDOFF cites
  (prints 2.1%/14.7%/190,474). Confirms the d=2048 contamination pattern from A.2.
- "~2.5× more visible" → **3.30×**. "Scale-invariant" → "independent of `d_model`" (breaks at 2.6B).
- **Novelty verdict, harsh and correct:** "mostly-conv hybrid ≈ all-attention at equal params with
  less KV" is **decisively established** (Mamba-2, Waleffe, Jamba, MAD, Samba, Falcon-H1, MiniMax,
  Hymba's 11.67× cache reduction). **The two-arm `L0` vs `A16-P` comparison is a re-measurement and is
  not worth GPU-hours.** What survives is narrow but real: no published ratio ablation uses a *k=3
  gated short-conv* mixer (all use large-state SSMs, ~100× more state), only Hymba reports recall, and
  LFM2 ships 37.5% attention where six labs converged on 7–25%.
- Minimum credible version is therefore not 2 arms but a **6-point ratio sweep** (`A0`/`A2`/`A3`/`L0`/
  `A16-P` + iso-compute `A16-C`): ~970 GPU-h ideal, **~1,550 GPU-h at a realistic 25% MFU**.
- Traffic is a **ceiling, not a prediction**, and the 9.93% is a near-cancellation of two larger
  effects — exactly the regime where 10% measures as 0%.

### C.5 Cheap experiments (`04_cheap_experiments.md`) — three plan constraints are FALSE

- 🔴 **The 6-hour / 1-GPU FarmShare constraint — which shaped the entire plan and my own A.6 —
  is wrong.** Measured: partition `MaxTime` is **48 h**, QOS `gpu` has **no `MaxWall`** and allows
  **4 concurrent GPU jobs**. The sibling KDA track already ran 20-hour jobs there. (Also `--mem=48G
  -c 8` is *rejected*: `MaxMemPerCPU=4000`. The recipe printed in HANDOFF and in the probe docstrings
  cannot run as written.) **This materially relaxes A.6 — FarmShare is a far more capable venue than
  the plan assumes, though still nowhere near 1,536 A100-hours.**
- Independently reproduced child C.2's conv-tap result across **all four** released checkpoints
  (350M/700M/1.2B/2.6B): pooled oldest-tap energy **4.26/5.24/5.34/4.78%**; only 2–10% of channels
  peak on the boundary tap. **k=3 is not boundary-saturated at any scale** — two independent agents,
  different code paths, same conclusion. Strong.
- **LIV layers are strongly heterogeneous by depth**: layer 0 puts >90% of energy on history in 95%
  of channels (a learned token-shift, not a convolution); layer 15 is 97.6% current-token
  passthrough. Replicates across all four including 2.6B's different 22-LIV topology.
  → Two consequences: the right width arm is **depth-varying** (not in `liv_arms.py`), and
  `short_conv.py:347-348` **initializes the conv to current-token identity — the opposite of what
  layers 0–1 converge to.**
- **The best zero-training result available:** LFM2-350M's model card declares a 32,768-token
  context, markets the model for "RAG and data extraction," was trained on 10T tokens, and publishes
  **zero** retrieval numbers — while `config.json` says `max_position_embeddings: 128000`. A passkey
  length×depth sweep on 3 checkpoints costs **~3.4 L40S-h / ~1.5 h wall-clock** and **every possible
  outcome is a first-ever published table.**
- MQAR recalibration on real `L0` costs **~153 L40S-h and produces no finding**; a d=256 proxy
  captures the same confounds at **~15 L40S-h**.
- Full recommended cheap program: **~23 L40S-h + ~40 engineering hours ≈ 1.5% of the 8-day program.**

### C.6 Risk audit (`07_risk_audit.md`) — the planned screen is statistically incapable

- 🔴 **The 12-arm × 2-seed screen (Phase 3a) cannot distinguish anything.** At n=2 a paired t-test
  has **1 degree of freedom** and rejects at α=0.05 **iff the two per-seed differences agree to
  within a 1.171 : 1 ratio**. Against the plan's **own measured** MQAR seed spread at its **own
  chosen operating point** (`N512_D64`: 0.05/0.09/0.20/0.56/0.98, σ ≈ 39.3 pp), a true effect of one
  full seed-SD gets **~9% power** — barely above the 5% false-positive rate. **The plan's main screen
  is a coin flip that costs ~500 A100-hours.**
- 🔴 **The CE gate is already known to be unreachable.** The plan says "measure `s_δ` in the pilot."
  The sibling KDA track **already measured it**: MDE ≈ 0.014 nats at n=5 paired ⇒ `s_δ ≈
  0.0113–0.0126`. The gate needs `s_δ ≲ 0.0114`. The measured value **straddles or exceeds** the
  requirement — the pilot would spend GPU-days confirming what is on disk.
- **Corrects me and the brief on embeddings:** the frozen ledger is **TIED**, and untied is
  *arithmetically impossible* (it forces a non-integer `ff` = 3242.67). My A.1 said tied and
  reconstructed the same ledger; the brief's "untied" was wrong. Confirmed by two independent routes.
- Confirms the 6.27%-is-a-d=2048-number finding (A.2, C.1) — three agents, three methods, same result.
- 🔴 **The CE gate is vacuous by MARGIN, independent of seed count** — a distinct and worse problem
  than the power one. The full Mamba-2 ratio basin (4–23% attention) spans 0.06 ppl = **0.0030–0.0072
  nats**, and the plan's non-inferiority margin is **+0.010 nats**. **The acceptance region strictly
  contains the entire phenomenon** — it would declare an all-conv model non-inferior to a transformer.
  So yes: **every perplexity gate in the plan is useless.** Constructive fix: AR-Hits gives ~3.2× SNR,
  dividing required n by ~10.
- 🔴 **Five of the plan's gates are FAIL-OPEN, and the defect is structural:** six criteria are
  non-inferiority statements confirmed by *failing to reject*. At n=5, **1 in 4 catastrophic 60-pp
  regressions scores as "retrieval preserved"**; at n=2, **91% do.** Free fix: CI-upper-bound vs an
  explicit Δ. Also §6.2's mandated success-rate endpoint at 2-vs-2 gives Fisher **p=1/6 — provably
  unable to reject at any outcome.** Family-wise error over 11 comparisons is **43%**.
- **The FLOP table is internally inconsistent** — no single linear model yields both 1.297×@4K and
  1.959×@32K. Independent route to the same bug I confirmed in C.4; §4's percentage table uses
  `2·T·d` where fwd+bwd needs `12·T·d`. **A committed test locks in the wrong number.**
- **`F-r512` saves exactly zero params at d=1024** — one of three rank rungs is a **null arm by
  construction**. §5.1's "systems-viable ranks {256,512}" is stale d=2048 advice.
- **32K: compute is NOT the blocker (0.76 h/arm; the budget's "2 days" is 50× generous) — DATA is.**
  And an interaction nobody drew: **document-isolated packing and long-context training are mutually
  destructive** — a 32K sequence packed from 622-token docs has no dependency past ~622 tokens.
  Ceiling on useful long-range signal: **0.09% of the budget.** But the topology *systems* claim
  doesn't need 32K training — decode bytes don't care that quality is bad. **Decouple them.**
- **Unlisted dependency:** per-rank **LR retuning** is in no phase and no budget. The plan caught the
  *init*-scale version of this trap and missed the LR version.
- **P2 has a full competitor set and no treatment:** 0 of 3 sharing arms built (24–40 h unbudgeted);
  11 of 27 declared arms exist; `F-r512`, `S-shared`, `1G` in Phase 3a do not.
- **One free clarification worth more than any experiment:** the plan never says whether MQAR is
  trained-from-scratch or evaluated on pretrained arms. If it is an *eval*, the 39.3-pp variance may
  not apply and the power picture improves dramatically.
- **Independently reached my A.5b/A.5c conclusion by pure reasoning:** *"the topology claim is P1's
  latency claim one day before it died — same argument form (bytes⇒time), exact byte arithmetic, zero
  latency measurement."* Two agents, one by measurement and one by analogy, converged on the same
  structural flaw. **This is the strongest signal in the reassessment.**

### C.7 Grouped tension (`03_grouped_tension.md`) — my Rank 4 was wrong; this is prior art

- **Verification holds:** iso-cost confirmed (both **exactly 262,144 params/gate**); 0.929/0.130
  reproduces. But **"80 points"/7× is a metric artifact** — retained-energy is an error metric only
  for the orthogonal projection (low-rank); for masks the cross-term is nonzero. Correct comparison:
  **7.2% vs ~63% relative error, an 8.8× gap.**
- **"Grouped ≡ random mask" is near-tautological** (a p/p² decomposition forces it absent
  block-alignment) — but backing it out yields a number the docs never state: **~64% of gate-output
  energy is cross-channel.**
- 🔴 **The two arms were not given equal effort.** Low-rank got the Eckart–Young *optimum*; grouped
  got a **naive mask with no OBS/SparseGPT reconstruction step**. That step has a closed form and
  costs 4 Cholesky factorizations of 256×256. **So "the deficit is structural" is NOT supported.**
- **My ShuffleNet hypothesis was wrong, and the child found the better version.** The LIV gates are
  **parallel, not serial** (`short_conv.py:269-271`), so "shuffle between them" is ill-defined; and
  `value_proj`/`out_proj` stay dense so the channel graph is never disconnected — the pathology
  ShuffleNet fixes is absent. The real missing arm is **Monarch b=8, which is exactly iso-cost
  (262,144) with both existing arms at d=1024** and was never tested.
- 🔴 **NOVELTY: NEGATIVE as scoped. Wei et al., NeurIPS 2024 (arXiv:2406.16450) already ran this** —
  LowRank vs **BlockShuffle** vs BlockDense, from scratch, iso-param, 110M–1.3B, Chinchilla.
  **LowRank won; BlockShuffle last by 0.4–0.8 PPL.** → **This retires my Rank 4 as a headline.**
- **Two validity threats that outrank cost:** (1) Wei et al.'s "self-guided training" correction is
  worth **up to 1.2 PPL — larger than the 0.4–0.8 PPL between-structure gap**, so a naive from-scratch
  run measures optimization pathology, not structure, *and the pathology is structure-dependent*.
  (2) Only **1.5% of params** are touched (5.24M of 354.5M) ⇒ predicted CE spread **< 0.02 nats**,
  below detection at any affordable seed count.
- **What genuinely survives:** the *operator* (nobody has structured **gated-conv gates** —
  multiplicative, unlike FFNs), and the **batch-1 CUDA-graphed decode measurement with kernel counts
  and an iso-byte control**, which is not in the literature. Also a real **prediction collision**:
  Potapczynski et al. (NeurIPS 2024) says full-rank block structures scale *better* than low-rank;
  Wei et al. measures the opposite at matched params — two NeurIPS 2024 papers, unreconciled.
- Independently confirms C.5: **FarmShare MaxTime is 2 days, 4 concurrent GPUs** — not 6h/1GPU.
- Cheapest real option: **MQAR at `N512_D64`, 4 arms × 10 seeds ≈ 3.5 GPU-hours.**
- **Biggest liability: every latency claim rests on one card.** Re-running on an A100/H100 costs
  under an hour — highest value per GPU-minute available. (This now compounds with my A.5c finding:
  cache size differs across cards, so the L2 artifact and the card-generality question are the *same*
  experiment.)

---

## PART B (revised) — RANKED RECOMMENDATIONS

**Framing decision first.** There is **no deadline and no named venue** (C.3). So the binding
constraint is not time — it is that ~1,536 A100-hours of training compute **do not exist**, while
FarmShare offers 4 concurrent L40S at up to 48 h (C.5). Every recommendation respects that.

### Rank 1 — Reverse the P1 latency verdict and lock down the cache result. ~2 L40S-h, ~1 day.
**Already 80% done (A.5c).** The project's most-repeated finding is a measurement artifact: the probe
fit inside the L40S's 96 MiB L2, and at a realistic working set every factorized arm *wins*
(`lowrank_fused` −1.8% → **+31…40%**). Remaining work: (a) re-run with `dense` as a correct single
fused `(3d,d)` GEMM (C.1's confound), (b) add an ncu HBM-counter confirmation, (c) propagate the
correction into HANDOFF, both memory notes, the probe README and design doc §5.1.
**Why first:** it is nearly free, it reverses a decision already made, and — combined with FLAR-SVD
and the iso-byte control — it converts the project's biggest negative into its **best positive
methodological result**: *cache residency silently inverts decode microbenchmarks; always report the
working set against L2.* That is a genuinely reusable finding and it is defensible today.

### Rank 2 — Publish the released-checkpoint measurements. ~5 L40S-h, ~3–4 days.
Passkey/recall length×depth sweep on 3–4 checkpoints (C.5) + the conv-tap analysis, now replicated by
two independent agents across four checkpoints (C.2, C.5). LFM2 has **no published recall benchmark
and no width ablation**, verified against the actual paper (C.4). Every outcome is a first-ever table.
Add the activation-weighted follow-up the `spectra_v2` lesson demands before publishing tap energies.

### Rank 3 — Fix the two code defects. ~3 engineering hours.
`short_conv.py:373` FLOPs 3× undercount (I verified it against the library's own convention) — it
corrupts `num_flops_per_token`, the exact quantity the plan says to match arms on. And the conv
identity-init points the wrong way versus what layers 0–1 learn (C.5). Cheap, and both silently
invalidate results rather than failing loudly.

### Rank 4 — ~4 GPU-hours of quality signal, NOT a from-scratch pretraining study.
**I had ranked a from-scratch grouped-vs-lowrank run here; C.7 retired it.** Wei et al. (NeurIPS 2024)
already ran LowRank vs BlockShuffle vs BlockDense from scratch, iso-param, 110M–1.3B — LowRank won.
Worse, their "self-guided training" correction (up to 1.2 PPL) is *larger than the between-structure
gap*, so a naive replication measures optimization pathology, and only 1.5% of params are touched
(predicted CE spread <0.02 nats — undetectable at any affordable n).
**Do instead, in this order:** (a) the free OBS/SparseGPT reconstruction re-probe — grouped was scored
with a naive mask against low-rank's Eckart–Young optimum, so *"the deficit is structural" is
currently unsupported*; (b) add the **Monarch b=8** arm, exactly iso-cost at 262,144 and never tested;
(c) **MQAR at `N512_D64`, 4 arms × 10 seeds ≈ 3.5 GPU-hours** for a quality signal that is actually
powered. What survives as novel is the *operator* (structured **gated-conv gates** are untouched in
the literature) plus the decode measurement — not the structure comparison itself.

### Rank 4b — Re-run the corrected latency benchmark on a second card. <1 GPU-hour.
Every latency claim in the project rests on one L40S (C.7). This now compounds with A.5c: **L2 size
differs across cards (L40S 96 MiB, A100 40 MiB, H100 50 MiB), so the cache artifact and the
card-generality question are the same experiment.** An A100 with 40 MiB L2 would have shown the
*original* probe as partially cache-resident at 40 MiB — the artifact is card-dependent, which is
itself the publishable point. Highest value per GPU-minute available.

### Rank 4c — Three free document fixes, ~30 engineering hours, zero GPU.
(a) Rewrite every gate as **CI-upper-bound vs an explicit Δ** and delete the CE gate and the 2-seed
row — 5 of 9 gates currently pass a catastrophic regression (C.6). (b) Thread document isolation
through **attention**, which is currently biased *in favour of the control*. (c) State whether MQAR is
trained-from-scratch or evaluated on pretrained arms — if it is an eval, the power picture improves
dramatically and it costs one sentence to find out.

### Rank 5 — Everything else: ABANDON.
- **P3** — cut. Killed by two independent measurements on released weights (C.2, C.5); its steelman
  is Tian et al.'s Table 7, already run and negative.
- **P2** — cut. Its designated headline ("nobody reports retrieval for KV sharing") is **false** as of
  arXiv 2606.06467 (C.2), which ran a better version and got the opposite sign.
- **The 12-arm × 2-seed rank sweep** — cut, on statistical grounds alone. r=512 saves exactly zero
  params at d=1024; the whole span is 4.44%→5.55%; and at n=2 the paired test has **1 df and ~9%
  power** against the project's own measured seed spread (C.6). It is ~500 A100-hours for a coin flip.
- **The Phase-2 `s_δ` pilot** — cut. KDA already measured it; the answer is on disk and says the CE
  gate is unreachable (C.6).
- **`L0` vs `A16-P` as a two-arm topology headline** — cut as a *novelty* claim (C.4). Keep the
  arithmetic as context.
- **MQAR recalibration on real `L0`** — defer (~153 L40S-h, no finding).
- **The `ShortConv` decode path** — defer. It is a prerequisite for *training*, not for any Rank 1–3
  item (C.5), though it is what would eventually turn the topology traffic ceiling into a measurement.

### Errors found, consolidated (for the correction pass)

| # | error | correction | where |
|---|---|---|---|
| 1 | **P1's latency claim "is dead"** | **MEASURED FALSE.** Probe fit in the L40S's 96 MiB L2. At a realistic working set `lowrank_fused` is **+31…40% faster**, not −8.2%. Job 1671420 | HANDOFF, design §5.1, both memory notes, probes/README |
| 2 | "−8.2%, replicated, ≤0.3% spread" | The two jobs give **−8.2% and −1.82%**; ≤0.3% is *within-job* replay noise. The reproducible value is −1.8% | same |
| 3 | "dense achieves 695 GB/s = 80% of peak, genuinely bandwidth-bound" | Units are **GiB/s**; correct is 746 GB/s. And it describes **L2**, not HBM — out of cache dense gets 462 GB/s ≈ 53% of peak | probes/README, HANDOFF |
| 4 | "3.4 µs per extra kernel" | Not a marginal cost: 20 × 3.379 = 67.6 µs > dense's measured 56.2 µs | HANDOFF, probes/README |
| 5 | `short_conv.py:373` `linear_flops = 2 * params` | **3× undercount** vs the library's own `6 *` convention. Compute gap 1.297×/1.959× → **1.207×/1.886×** | committed code |
| 6 | "6.27% model cut / mixer is 14.3% / MLP 68.8%" | d=2048 numbers in a d=1024 design → **4.44% / 11.8% / 63.9%** | memory note, design §5.1 |
| 7 | "No LFM2 paper exists (only a blog post)" | **arXiv:2511.23404 exists**; the repo contradicts itself | `06_baselines_infra.md:118` |
| 8 | "No cross-layer-sharing paper reports needle/passkey/MQAR" — P2's designated headline | **False.** arXiv 2606.06467 ran RULER × 12 subtasks × {16K,32K} and got the *opposite* sign (+6.1) | design §5.2, memory note |
| 9 | "FarmShare = 1 GPU, 6-hour limit" | **MaxTime 2 days, 4 concurrent GPUs.** Also `--mem=48G -c 8` is rejected (`MaxMemPerCPU=4000`) — the printed recipe cannot run | HANDOFF, probe docstrings |
| 10 | "grouped loses by 80 points; the deficit is structural" | Metric artifact (energy is an error metric only for projections) → **7.2% vs 63% rel. error**. And grouped got a **naive mask** vs low-rank's Eckart–Young optimum, so "structural" is **unsupported** | probes/README, HANDOFF |
| 11 | 12-arm × 2-seed screen | **1 df, ~9% power** against the project's own measured seed spread | design §8 |
| 12 | "brainlift is 2.5× more sensitive at 350M" / "KV is scale-invariant" | **3.30×**; "independent of `d_model`", breaks at 2.6B | HANDOFF, memory note |
| 13 | P3's steelman (RepVGG reparameterization) is an open question | **Already run** — Tian et al. **Table 7 is that experiment**, 12.79→13.28 | design §5.3 |
| 14 | Grouped-vs-lowrank is an unplanned novel tension | **Wei et al. NeurIPS 2024 (2406.16450)** ran it from scratch, iso-param, 110M–1.3B | this reassessment |
| 15 | CE non-inferiority margin +0.010 nats | **Vacuous** — the entire Mamba-2 ratio basin is 0.0030–0.0072 nats, so the acceptance region contains the whole phenomenon | design §6.1 |
| 16 | Retrieval/quality gates | **5 of 9 FAIL-OPEN.** At n=5, 1 in 4 catastrophic 60-pp regressions passes as "preserved" | design §6, §8 |
| 17 | `F-r512` is a rank rung | **Null arm** — saves exactly zero params at d=1024 | design §8, liv_arms |
| 18 | "32K stage is compute-limited (~2 days)" | Compute is 0.76 h/arm (50× generous); **data** is the blocker, and doc-isolated packing caps useful long-range signal at **0.09%** of the budget | design §8 |
| 19 | Long-context quality and the topology systems claim are coupled | **Decouple** — decode bytes don't care that quality is bad | design §3.2, §5 |

Non-errors worth recording: the **T≈4,121** crossover, the **354,483,968** ledger, the **12/32/20 KiB**
KV figures, the tied-embedding geometry, and the arXiv IDs 2607.18413 / 2606.03825 all **verified
correct** by two or more independent routes.

### The honest bottom line
The floor (C.3) is a 6–10 page measurement report writable today from 11 existing results. Ranks 1–4b
cost **~12 L40S-hours and under a week** and lift that floor materially — Rank 1 in particular
converts the headline from "our proposal failed" to "the standard way of measuring this is wrong, and
here is the control that shows it." **That is a better capstone than the original three-proposal
efficiency story would have been even if it had succeeded.** Note also (C.3) that `KDA/` is already
complete and unwritten — if the goal is a finished artifact rather than more measurement, that is the
cheaper path, and the two are not in conflict since Ranks 1–4b need almost no compute.

**The single most important sentence in this document:** the project spent its credibility on
"P1's latency claim is dead," propagated it into four documents and a frozen design decision, and
**it was an artifact of a benchmark whose working set fit in cache.** The team's own stated lesson —
*"for any decode-time factorization, run an iso-byte control before believing a roofline estimate"* —
was the right instinct applied one level too shallow: the iso-**byte** control was run, but the
iso-**residency** control was not. Fixing that is cheap, it is already 80% done, and it is the result
worth writing up.

### Meta-note on this reassessment's own reliability

Four findings were reached independently by two or more agents using different methods (the d=2048
contamination, the conv taps not being boundary-saturated, the GiB/s mislabel, the FarmShare limits
being wrong) — treat those as solid. Three findings rest on a **single** source and should be
re-checked before they are acted on: the Wei et al. prior art (C.7, retires a whole direction), the
arXiv 2606.06467 result that kills P2's headline (C.2), and my own A.5c cache result — which,
although it is a direct measurement I ran and which reproduces a stored prior value exactly, is still
a *gate-subgraph* proxy and not a full-model decode A/B. **Do not let A.5c become the next
insufficiently-scoped headline.** State it as: *"at a realistic working set the sign reverses on the
gate subgraph; the model-level effect is estimated at +1.8–2.4% and has not been measured."*
