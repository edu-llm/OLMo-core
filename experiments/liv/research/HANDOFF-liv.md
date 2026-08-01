# HANDOFF — LIV brainlift experiment design

**Scope:** this file covers **only** the LIV/LFM2 architecture experiment design track.
Sibling handoffs are independent and still current: [`KDA/HANDOFF.md`](KDA/HANDOFF.md) (KDA-Householder,
COMPLETE), [`KDA-LIV/HANDOFF.md`](KDA-LIV/HANDOFF.md) (sub-500M KDA-insertion protocol — audited and
redesigned 2026-07-31; this track has **KDA out**, that one is *about* KDA insertion),
[`quant_research/HANDOFF.md`](quant_research/HANDOFF.md),
[`edullm-data/HANDOFF.md`](edullm-data/HANDOFF.md).

**Last updated:** 2026-08-01. **Status: infrastructure COMPLETE and committed to a worktree branch.
No model trained yet. Ready to launch the first training pilot.**

Frozen: **350M d=1024**, **vocab 50,304**, corpus = the 1.2B-token GPT-2 FineWeb-Edu set already on
FarmShare, P1 parallel with the topology gate, P3 widths-first, KDA out, energy cut.
**All compute on FarmShare** (48 h × up to 4 GPUs — not the 6 h × 1 the plan assumed). SB-AWS unused.

Four things changed since the last update, two of them reversals of my own recorded conclusions:

1. **P1's decode-latency claim is ALIVE.** The "8.2% slower" was an **L2-cache artifact** — the
   benchmark's 40 MiB working set fit inside the L40S's 96 MiB L2. Past L2 it is **+29.9%** on the
   gate subgraph, ≈**+1.8% end-to-end**. Verified twice. See the box in §1.
2. **`ShortConv.num_flops_per_token` had a 3× undercount** (`2*params` where `Attention` uses
   `6*params`). Fixed in `f9abaa1`; `A16-P` ratios are 1.207×/1.886×, not 1.297×/1.959×.
3. **The 65,536 "exact released shape" target does not exist** and has been abandoned. LFM2's
   tokenizer has 64,400 entries against a declared 65,536 — the released number is itself a pad.
   Vocab is now 50,304 (`bdc5051`).
4. **Compute is 8× larger than assumed**, which makes the real 350M study feasible on FarmShare.

**The mixer is at exact float64 parity with released LFM2, and both original Phase 0 blockers are
cleared.**

---

## Goal

Turn [`Brainlifts/Eric_LIV_Brainlift-1.pdf`](Brainlifts/Eric_LIV_Brainlift-1.pdf) into a runnable,
falsifiable experiment on Liquid AI's LFM2 architecture (LFM2 = 16 layers, 10 gated short-conv "LIV"
+ 6 GQA attention).

The brainlift proposes three changes to the stock LIV block:
- **P1** — factorize the two gate projections low-rank (`d→r→d` instead of `d→d`)
- **P2** — cross-layer KV sharing: pair attention layers so 3 KV banks serve 6
- **P3** — replace the single k=3 conv with 4 multi-span branches + a token-dependent softmax router

## Deliverables produced

| Artifact | Lines | What it is |
|---|---:|---|
| [`docs/liv-brainlift-experiment-design.md`](docs/liv-brainlift-experiment-design.md) | ~1,330 | **The deliverable.** Full design: arms, endpoints, phases, budget, open questions |
| `Brainlifts/liv_experiment_research/00_my_arithmetic_check.md` | 422 | My own verification + crossover math. Has runnable `crossover.py`, `proposals.py` |
| `…/01_lfm2_architecture.md` | 1,391 | LFM2 spec, configs, param counts vs 6 checkpoints, license |
| `…/02_lowrank_gates.md` | 2,798 | P1: prior art, init/optimization, measured LFM2-350M spectra |
| `…/03_kv_sharing.md` | 1,797 | P2: CLA/YOCO/MLA/Hymba, novelty verdict, RoPE decision |
| `…/04_multiscale_routing.md` | 2,181 | P3: width sweeps, router hazards, equivalence proof |
| `…/05_evaluation.md` | 2,077 | MQAR, BABILong, phonebook, seed counts, eval noise |
| `…/06_baselines_infra.md` | 1,962 | Hybrid baselines, ratio consensus, codebase pick, recipes |
| `…/07_latency_kernels.md` | 1,641 | Kernels, roofline, ncu recipe, edge reality |
| `…/08_local_infra_audit.md` | 347 | What exists in this repo vs must be built |

Memory notes written: `liv-experiment-key-numbers.md`, `liv-experiment-prior-art-verdicts.md`.

---

## Current progress

Research **complete** (8 threads). Design doc **written**. §9 decisions **made** (2026-07-31, table in
Next Steps §0). **Phase 0 mixer implemented and committed to a worktree branch** (§2).

All five probe scripts in `Brainlifts/liv_experiment_research/probes/` have been **run**, with results
JSON alongside and a README summarizing them:

| script | result |
|---|---|
| `l40s_breakeven.py` | Corrected the break-even for d=1024 + L40S (superseded by measurement) |
| `spectra_v2.py` | Closed the activation-aware gap — P1's premise stays falsified |
| `p1_launch_bench.py` | **P1's latency claim is dead** (FarmShare 1670883) |
| `p1_verify.py` | Replicated it, ≤0.3% spread, kernel counts measured (1670884) |
| `structure_energy.py` | `grouped` wins latency but loses quality by 80 points |

**Environments** (three, and they are not interchangeable):
- `Brainlifts/liv_experiment_research/.venv-spectra` — torch 2.13.0 + transformers 5.14.1, for probes.
- `OLMo-core/.venv` — now also has `transformers==5.14.1` for the parity test; `torchvision` removed.
  **Run with `PYTHONNOUSERSITE=1`**: a user-site `torchvision` in `~/.local/lib/python3.13` is built
  against a different torch and breaks `transformers` imports with `torchvision::nms does not exist`.
- FarmShare `/scratch/users/ericrcwu/kda/venv` — torch 2.11+cu128, GPU work. Workdir for this track is
  `/scratch/users/ericrcwu/liv/probes`. GPU is **L40S sm_89 46 GB** (not A100/H100).

This Mac has **no CUDA** (`mps` only), so every latency measurement is a FarmShare job by necessity.

### Verified sound

- **Every number in the brainlift is exact**, reproduced twice independently: LIV mixer 16,783,360 at
  d=2048/k=3; GQA 10,485,760 (2.5d²); factorized r=128 = 9,443,328; 12 KiB/token KV; 384 MiB at 32K.
  Formulas validated against **six released checkpoints** to the parameter.
- **LIV operator pinned to released Apache-2.0 code** (`modeling_lfm2.py`, transformers v5.0.0rc1),
  instantiated and diffed to **0.0**.

### The finding that reorganized the design

Measured per-op decode profile of real LFM2.5-350M ONNX q4 (ORT CPU, 4 threads, 128-token past):

| op | µs/step | share |
|---|---:|---:|
| `MatMulNBits` | 42,389.3 | **91.2%** |
| `Conv` (the LIV depthwise) | 452.6 | **1.0%** |

Corroborated independently by params: the depthwise kernel is `k·d` = 3,072 weights = **0.07%** of the
mixer. **P1 attacks the 91%; P2 attacks bytes; P3 attacks the 1% and makes it 5.6× worse.**

### Verdicts

| | Verdict | Key numbers |
|---|---|---|
| **P1** | **Run first — as a QUALITY claim. Latency claim killed by measurement (2026-07-31).** | Best case (fused + CUDA graphs) is **8.2% SLOWER** than stock LIV on L40S. Surviving hook: rank-128 keeps **92.6%** of activation-weighted energy at **0.25×** params |
| **P2** | Run second, **narrow claim** | Saves capacity **not bandwidth** → latency ≈ 0 by construction. Anticipated 3× |
| **P3** | **Reframe** as a quality claim, don't sell as speed | Published width sweeps flat past k=3; prior on router ~20-25% |

---

## What worked

- **Fan-out of 8 parallel research agents, one topic each**, each writing incrementally to its own file.
  ~14,600 lines of cited research. Disjoint file ownership meant zero write conflicts.
- **Verifying agent claims myself with arithmetic before acting on them.** Caught two agent errors
  (a low-rank init magnitude that was off by 0.02⁴, and a "sign flips with d·r" claim that was false).
- **Fetching primary sources directly** (`modeling_lfm2.py`, `config.json`, `causal-conv1d` README,
  Zamba/CLA HTML) rather than trusting summaries. Every load-bearing fact came from a primary source.
- **Writing findings to disk continuously.** Two agents died mid-run to API stalls; because siblings had
  already flushed to disk, nothing was lost.
- **Adversarial second-pass agents, checking the first pass's claims.** This is what caught the L2-cache
  artifact that had wrongly killed P1, and the 3× FLOPs undercount I had shipped. Then a *third* pass
  checking the second found four of its claims overstated. **Each layer of checking paid for itself.**
- **Re-testing a reversal myself before acting on it.** When an agent claimed my frozen result was an
  artifact, I wrote an independent probe (`probes/p1_cache_check.py`) that varied *only* residency with
  kernel count held fixed. It reproduced the flip, so the reversal was real — but the same discipline
  also let me reject three of that team's other claims.
- **Checking a number against something it should have matched.** Both of my own bugs were caught this
  way: achieved bandwidth against the card's HBM spec (759 > 864 peak ⇒ must be cache), and my FLOPs
  convention against `Attention`'s. Neither crashed; neither failed a test. **Ask "what would this
  number look like if I were wrong?" — if you can't answer, you can't distinguish a result from a
  broken setup.**

## What didn't work

- **Letting the agent tree go two levels deep.** Research agents spawned their own children. When a
  parent died to an API stall, its children kept running unsupervised with nobody to report to — this
  is what the user saw as "the agents stalled." Had to hunt and `TaskStop` three orphans
  (`CUDA Best Practices timing section`, `GPU latency measurement methodology`,
  `energy measurement and bandwidth`), all re-treading ground their parent had already written.
  **Next time: cap subagents at one level and poll them directly.**
- **Presenting findings as settled before all threads reported.** The substance shifted twice under the
  user: the P2 novelty verdict (Zamba-only → Hymba + Character.AI + Gemma 3n) and the P3 gate caveat.
  **Next time: hold the summary until all threads land, or label confidence explicitly.**
- **Benchmarking an isolated subgraph small enough to fit in cache.** The single most costly mistake in
  the project: it produced a confident, replicated, *wrong-signed* result that killed P1 for a day and
  got written into three documents. The subgraph was 40 MiB against a 96 MiB L2. **Always report the
  working set and the achieved bandwidth next to any latency number** — 759 GB/s against an 864 GB/s
  peak was the visible tell, and I looked straight past it.
- **Trusting a convention instead of reading the neighbouring implementation.** `ShortConv` used
  `2*params` for FLOPs where `Attention` uses `6*params`. Nothing failed; the arm-matching quantity was
  just silently a third of its true value.
- **Three of my own claims were wrong and had to be retracted in-document:**
  - "Edge runtimes can't run the variants" — **false and checkable.** Someone exported the 4-branch
    dilated conv and all 4 convs delegated to XNNPACK with **zero** CPU fallback.
  - "Conv state grows 7×" — it's **5×** (LFM2 cache is `[B,d,L_cache]` with `L_cache=k`, so 15/3).
  - "MQAR is bimodal, so success-rate-over-seeds is the endpoint" — true at low capacity load, **false
    at high load.** `N512_D64` spread continuously (0.05/0.09/0.20/0.56/0.98); with 64 pairs a model
    binds *some* pairs. Collapsing that rung to a success rate scores an arm at 0.55 the same as 0.05.
- **Letting a subagent's confident phrasing stand in for its method.** A reassessment claimed a result
  was "measured inside the gate" and "independently replicated"; verification found it was weights-only
  (never ran a token) and the same script 2m40s apart. **Ask what was actually executed, not what the
  summary calls it.**
- **Over-claiming adverse evidence for P3.** I said "two labs tested your mechanism and it lost." The
  user correctly pushed back: **neither used Liquid's double gate** (both bolt an *ungated* residual
  conv onto a transformer). Corrected in §5.3 item 0a.

## Key decisions

1. **Scale: 350M (d=1024), not 1.2B.** KV bytes/token is **scale-invariant** (12 KiB — depends only on
   hkv, hd, attention-layer count) while weight bytes scale with d². So the smaller model makes every
   cache effect **~2.5× more visible**. Inverts the usual "scale up" instinct. Also halves the budget.
2. **Primary endpoints: recall + length extrapolation + AR-Hits sliced perplexity.** NOT held-out CE.
   Published ratio sweeps span **0.06 ppl** (below noise) while the same models differ **20+ points on
   recall**. This repo's own KDA study measured that a +0.0053-nat effect needs **~n=43 seeds**.
3. **Drop KDA from this study.** It's a separate mechanism with its own completed result
   (`KDA/HANDOFF.md`). Mixing destroys attribution.
4. **RoPE: share post-rotary K.** Settled by Hymba's released implementation (producer rotates K,
   consumer rotates only its own Q). CLA's paper never addresses it.
5. **Scope: GPU-primary, justified by *methodology*** (can't lock clocks on mobile; variance >10%
   exceeds a 4% effect) — **not** by false impossibility. Energy cut. Take **one** calibration datapoint
   on the *unmodified* LFM2-350M GGUF/ONNX (the ONNX path already runs: 40.3 tok/s).
6. **Codebase: OLMo-core.** `SequenceMixerConfig.register()` is a clean extension point;
   `block_pattern`/`block_overrides` express LFM2's irregular `[2,5,8,10,12,14]` natively;
   `num_params()`/`num_flops_per_token()` are part of the mixer API; WSD scheduler for arm forking.
7. ~~**Corpus:** `s3://edullm-datasets/olmo-150b-dolma2/` (155.6B tokens, pre-tokenized).~~
   **SUPERSEDED 2026-08-01: the corpus is the 1.2B-token GPT-2 FineWeb-Edu set already on FarmShare**
   at `/scratch/users/ericrcwu/kda/lm/data/` (`sample-10BT`, gpt2 tokenizer, uint16, EOS 50256).
   Consequences: **vocab is 50,304** (see §3.5 and `bdc5051`), all compute is FarmShare, and SB-AWS is
   unused. **This corpus is 8-17× short of the declared token budget** — ample for the pilot and a
   scaled study, not for the full 2B-token × 8-seed plan. If the headline study needs to pool with the
   sibling KDA-LIV track, that track resolved to **dolma2 at vocab 100,352**; reconcile *before*
   committing GPU-weeks, because three tracks have now each paid for this one unmade decision.
8. **Framing — the most important decision.** Do **not** sell this as "three efficiency wins" (that
   story dies if results come out flat). Sell it as: *Liquid ships this architecture to real devices and
   published no ratio ablation, no kernel-width ablation, and no recall benchmark; CLA/Hymba measured
   averages and skipped retrieval entirely. We measured it properly, with seeds.* **True regardless of
   which way any result lands.**

---

## Next steps (priority order)

### 0. ✅ RESOLVED 2026-07-31 — the five §9 decisions are made

| Question | Decision |
|---|---|
| Sequencing | **Parallel** — P1 runs alongside the `L0 vs A16-P` topology gate |
| Headline scale | **350M, d=1024** |
| P3 scope | **Widths (`k5/k9/k15`) inside the real gated block first.** Router only if a width beats k=3 |
| Compute | ~~Both~~ → **FarmShare only** (2026-08-01). Verified live: `MaxTime=2-00:00:00` (48 h) and the `gpu` QOS allows `gres/gpu=4`, **not** the 6 h × 1 GPU the plan assumed. SB-AWS is unnecessary. Caveats: multi-GPU needs an explicit `--qos=gpu` (default `normal` caps at 1); only **4 concurrent jobs**; `--qos=long` is 7 days but **CPU-only** — its TRES line has no GPU entry |
| KDA | **Out** (separate mechanism, already has its own completed result) |
| Edge/energy | **Cut.** GPU-only, justified by methodology |

Rationale for the two load-bearing ones: P1's control is *stock LIV*, not all-GQA, so it is internally
valid regardless of how the topology gate lands — and LFM2 ships to real devices, so improving its
gates stays publishable even if our `L0` loses. 350M because KV bytes/token is scale-invariant, making
cache effects ~2.5× more visible, and because the freed budget buys **seeds** — this repo's own KDA
study saw +8.92pp at n=3 collapse to +2.01pp (ns) at n=8.

**⚠️ Consequence of choosing d=1024: P1's break-even numbers changed. See below.**

### 1. Two microbenchmarks — days, not weeks. Do these BEFORE any training code.

Each can kill a proposal cheaply. **FarmShare's GPU is L40S (sm_89), not A100/H100** — see
`KDA/HANDOFF.md:320-340` for the access recipe (`ssh -M -S /tmp/farmshare-ericrcwu.sock ...`, socket
must be opened by a human for Duo).

> ## 🔴 SECTION 1 IS SUPERSEDED — READ THIS BOX FIRST (2026-08-01)
>
> **P1's latency claim is ALIVE. The "8.2% slower" result below is an L2-cache artifact and its
> sign is not trustworthy.** Everything in §1 up to the correction block should be read as a record
> of a mistake, not a finding.
>
> The L40S's L2 is **96.0 MiB** (measured). The benchmark's working set was **40 MiB** — fully
> cache-resident, with a CUDA graph replaying over the same weights and nothing to evict them. The
> tell was in the output all along: `dense` reported **759 GB/s**, *above* the card's 864 GB/s HBM
> peak, which is only possible from cache. Cache residency is precisely the regime where reading
> fewer bytes buys least, so the benchmark was built such that P1's only benefit could not appear.
>
> Re-tested with the working set scaled past L2 (`probes/p1_cache_check.py`, FarmShare **1671421**),
> kernel count per timed step held identical so only residency changes:
>
> | dense working set | `lowrank_fused r=128` | `grouped g=4` |
> |---|---:|---:|
> | 40 MiB (original, in L2) | −3.7% | +20.4% |
> | 160 MiB (past L2) | **+39.1%** | +52.5% |
> | 960 MiB (past L2) | **+29.9%** | +46.1% |
>
> **The past-L2 rung is the representative one:** the real 350M model reads **709 MB/token = 7.0× L2**,
> so decode is genuinely HBM-bound. Gates alone (40 MiB) fit in cache, but inside a full model the
> other 94% of weights evicts them.
>
> **Honest magnitude:** gates are 5.9% of weights, so +29.9% on the gate subgraph share-weights to
> **≈ +1.8% end-to-end**. Real, worth having, *not* a 30% headline. This is a gate-subgraph proxy,
> not a full-model A/B — do not let it become the next over-scoped claim.
>
> Two things below are still correct and worth keeping: skinny GEMVs really do achieve far less
> bandwidth than dense (161 vs 759 GB/s), and `grouped` really does win on latency at every working
> set. What was wrong was concluding that low-rank therefore *loses*.
>
> Also note "695 GB/s = 80% of L40S peak" below is wrong twice: it is GiB/s, and it describes L2
> throughput, not HBM. Ignore that parenthetical.

- **P1 thin-matmul bench — ⚠️ SUPERSEDED (see box above). Recorded as-run for the audit trail.**
  FarmShare jobs **1670883** and **1670884** on `oat-04` (L40S). 1670884 replicated 1670883 with 3
  trials (**spread ≤0.3%**) and profiler-**measured** kernel counts (20/30/40, exactly as predicted)
  — so it is a sound replication *of an unrepresentative configuration*, which is a different and
  more interesting failure than noise. Scripts `probes/p1_launch_bench.py` + `probes/p1_verify.py`;
  raw JSON on FarmShare at `/scratch/users/ericrcwu/liv/probes/`. All CUDA-graphed:

  | arm | kernels | MiB/tok | graphed | vs dense |
  |---|---:|---:|---:|---|
  | `dense` (stock LIV) | 20 | 40.0 | 56.2 µs | — |
  | `lowrank_fused` r=128 | 30 | 10.0 | 60.8 µs | **−8.2% SLOWER** |
  | `lowrank_sep` r=128 | 40 | 10.0 | 76.5 µs | −36.0% |
  | **`grouped` g=4** | 20 | 10.0 | **47.6 µs** | **+15.3% FASTER** |

  ⚠️ *Every row above is cache-resident. The 40 MiB working set fits in the L40S's 96 MiB L2, so the
  `vs dense` column measures the wrong regime — see the correction box. Past L2 the same arms give
  −3.7% → +39.1% → +29.9% as the working set grows.*

  ~~**Even fused + graphed — the configuration predicted to win — is 8.2% slower while reading 4× fewer
  bytes.** Not recoverable by tuning.~~ **Mechanism, from the iso-byte control:** at 40 MiB held
  constant, `dense` 56.2 µs vs `lowrank_fused r=512` 90.0 µs ⇒ **~3.4 µs per extra kernel even under
  graphs**. Dense achieves 695 GB/s (80% of L40S peak, genuinely bandwidth-bound); `lowrank_fused
  r=128` achieves only **161 GB/s** — skinny GEMVs cannot saturate the memory system, so saved bytes
  buy far less time than the roofline model predicts. The analytic model blamed launch overhead; the
  real cause is GEMV inefficiency. (Matches FLAR-SVD on Snapdragon: slower despite 2× fewer params,
  with r=128 flagged below the inflection point.)

  **→ ~~Action: strike the decode-latency claim from P1.~~ REVERSED — see the box above. Keep the
  claim, sized honestly at ≈+1.8% end-to-end, and always report the working set alongside it.**
  Superseded-but-recorded: the analytic break-even (`probes/l40s_breakeven.py`) said A100 1.35 µs /
  L40S 2.43 µs at d=1024, correcting the widely-quoted 4.72 µs which was computed at d=2048. Also
  **r=512 saves zero bytes at d=1024** (`2dr ≥ d²` once `r ≥ d/2`).

- **`grouped` wins latency but loses quality — do NOT promote it.** `grouped g=4` and `lowrank r=128`
  cost *exactly* the same (0.25× params, 10 MiB/tok), so the choice is pure quality. Measured
  activation-weighted retained energy (`probes/structure_energy.py`, released checkpoint, 32,768
  tokens): **`lowrank r=128` 0.929 vs `grouped g=4` 0.130** — and 0.130 is *identical to a random mask
  of the same density*, so block structure buys nothing. Channel ordering doesn't rescue it (random
  permutations `[0.125, 0.133]` vs 0.130 identity) → the deficit is structural.
  **Caveat bounding this:** the metric measures approximation of *trained dense weights*, which favors
  low-rank by construction (Eckart-Young makes rank-r truncation optimal; the block mask is not
  optimized at all). A from-scratch grouped layer ≠ the block-diagonal part of a trained dense layer —
  cf. GaLore's `W=BA` collapsing from scratch (142.53 vs 15.56 ppl at 1B) at more generous rank
  fractions. Treat as a strong prior, not a verdict. Keep `grouped` as the mandatory systems
  competitor; the low-rank-vs-grouped tension is a better result than either winning outright.
- **P3 conv bench.** Profile one 4-branch dilated conv vs the 3-tap baseline. Isolated numbers already
  known (155.1 vs 27.5 µs → 0.3-1.6% total decode regression); this confirms on target hardware.
  Now **lower priority** than it was: since P3 is scoped to widths-first, the deciding question is
  quality (does k9/k15 beat k3 *inside the gate*), not latency.

### 2. Phase 0 — ✅ BOTH HARD BLOCKERS CLEARED + ARM BUILDER DONE (2026-07-31)

Two commits on branch `agent/claude-01/liv-short-conv-mixer`, worktree
`/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer`
(per OLMo-core's CLAUDE.md, which requires agent work in a worktree; branched from `f17824e`):
**`83e4dce`** the mixer, **`c2aac8e`** the arm builder. **Neither merged.** isort + black + ruff +
mypy all clean on both.

**Everything is new modules.** Across both commits, **1,470 of 1,472 changed lines are new files**;
the only edit to existing OLMo-core code is a 2-line assertion relaxation in `generation_module.py`
(detailed below). The mixer plugs in through the library's own extension points —
`SequenceMixerConfig.register()` and `block_overrides` — with no changes to `TransformerConfig`,
`TransformerBlock`, or the model builder, and `CausalConv1d` left untouched for its existing caller.

- **✅ LIV mixer written.** `src/olmo_core/nn/attention/short_conv.py` — `ShortConv(SequenceMixer)` +
  `ShortConvConfig`, registered as `"short_conv"`. Three gate structures: `dense` (as released),
  `lowrank` (shared `d→2r` down-projection — the fused variant the benchmark showed is the only
  viable one), `grouped` (block-diagonal).
  **Parity: exactly 0.0 float64 difference vs the released `Lfm2ShortConv`** at k=3/5/9, asserted in
  the test suite (not just a one-off script). Mixer params reproduce `4d² + kd` exactly — **16,783,360
  at d=2048, k=3**, matching the brainlift to the parameter.
- **✅ Generation unblocked.** The two `assert isinstance(block.attention, Attention)` calls now skip
  non-attention mixers instead of crashing. The adjacent comment already said *"not all models use
  key-value caches… Mamba requires cache state but doesn't use a kv-cache"* — the assertion
  contradicted it. **153 passed / 0 failed** across `src/test/nn/attention/` + `src/test/generate/`.
- **✅ End-to-end topology verified.** A full 16-layer hybrid builds with **10 ShortConv + 6 Attention
  at exactly `[2,5,8,10,12,14]`**, forward and backward clean, every parameter receiving gradient.

**32 new tests.** Beyond parity they pin: causality; receptive field is *exactly* k−1 (probed against a
nonzero background — with a zero background the multiplicative gates make every lag read as "no
reach", a false pass); `num_params` agreeing with built modules; the matched-cost lowrank/grouped pair
being genuinely equal; step-0 gate-variance parity; document isolation matching independent
per-document forwards; identity-init equivalence across k=3/5/9/15; grouped having no cross-block
mixing; meta-device construction; `r ≥ d/2` saving nothing.

**⚠️ FOURTH TRAP, found the hard way — the per-layer override field is `block.sequence_mixer`, NOT
`block.attention`.** Setting `.attention` on a block config silently creates a new attribute, the
override is ignored, and **every layer stays attention** — I hit this and the model built, ran, and
backpropagated cleanly with *zero* ShortConv layers. It only surfaced because the check asserted layer
*types* rather than that the forward pass worked. A test now guards it. Note `block.attention` *is*
the right accessor on a built **module**; the mismatch is config vs module.

#### ✅ Declarative arm builder — `c2aac8e`

`src/olmo_core/nn/transformer/liv_arms.py`. Every arm is one `LivArm` entry built by one function, so
two arms that differ in one field differ in exactly one visible line. **11 arms declared:** `L0`,
`A16-P`, `F-r128`, `F-r256`, `G-grouped`, `N-narrow`, `W-k5/k9/k15`, `A-fewer3`, `Q-mqa`.
Run `python -m olmo_core.nn.transformer.liv_arms` for the cost table.

**`L0` hits its ledger exactly: 338,886,400** at vocab 50,304 (was 354,483,968 at 65,536 — see the
corpus note in Key decisions #7 for why that target was abandoned). Getting the *original* number to
land caught **two geometry omissions**, each of which would have silently invalidated every
arm-matching decision — and both fixes still apply at the new vocabulary:

| omission | cost | how found |
|---|---:|---|
| `llama_like` defaults to **untied** embeddings | **+67,108,864** (~19% of the model) | ledger overshoot |
| **Per-head QK-norm** missing — LFM2's `Lfm2Attention` builds `q_layernorm`/`k_layernorm` of size `head_dim`; both `qk_norm` flags default `False` | **+768** = 6 layers × 2 × 64 | residual −768 gap |

The 768 gap is the reason the component-by-component reconciliation test exists: an exact *total* can
hide two offsetting errors, so the test checks embeddings, mixers, MLP, and norms independently.

**Derived widths are solved, never guessed.** `solve_swiglu_width` and `solve_d_model` produce
`A16-P`'s width and `N-narrow`'s `(d_model, width)` pair, and a test asserts the committed constants
still equal what the solvers return — so a drift between declaration and derivation fails CI.
`N-narrow` needed a **two-stage** solve: `d_model` alone on the 16-multiple grid (the head count is the
binding constraint) only reaches **0.815%** of `F-r128`, far too coarse for a capacity control, so the
SwiGLU width closes the remainder to **0.0145%**.

**Final cost table** — regenerate any time with
`python -m olmo_core.nn.transformer.liv_arms`. Current, at **vocab 50,304** and the corrected 6×
FLOPs convention (params, then FLOPs/token — the two diverge, which is the point):

| arm | params | vs L0 | flops@4K | flops@32K |
|---|---:|---:|---:|---:|
| `L0` | 338,886,400 | 1.000× | 1.000× | 1.000× |
| `A16-P` | 338,791,424 | **1.000×** | **1.215×** | **1.905×** |
| `F-r128` / `G-grouped` | 323,157,760 | 0.954× | 0.960× | 0.979× |
| `N-narrow` | 323,188,528 | 0.954× | 0.960× | 0.979× |
| `A-fewer3` | 342,040,960 | 1.009× | 0.943× | **0.733×** |
| `Q-mqa` | 333,381,376 | 0.984× | 0.986× | 0.993× |

**`A16-P` is parameter-matched to L0 within 0.03% but uses 1.91× the FLOPs at 32K** — attention's score
term grows with context, a convolution's does not. A test asserts this gap so nobody quietly
param-matches a compute-controlled comparison. `A-fewer3` is the mirror image and the reason it is P2's
strongest competitor: it cuts long-context compute to 0.733× *and* halves read bandwidth, which
cross-layer KV sharing structurally cannot do.

**`F-r128`, `G-grouped` and `N-narrow` are all 323M and within 0.01% of each other** — that three-way
match is what makes the pilot in §3.5 a clean quality comparison. `L0 − F-r128` = **15,728,640 at every
vocab size, bit-identical**, so none of this moves if the vocabulary is revisited.

**23 arm tests** (55 total with the mixer suite): exact ledger, component reconciliation, mixer
placement per arm, width arms differing *only* in width, the matched-cost pair being exactly equal,
solver/declaration agreement, and forward+backward for every arm.

**Environment notes:** a user-site `torchvision` in `~/.local/lib/python3.13` is built against a
different torch and raises `operator torchvision::nms does not exist` when `transformers` imports it —
run with **`PYTHONNOUSERSITE=1`**. `transformers==5.14.1` was installed into `OLMo-core/.venv` for the
parity test (the test `importorskip`s it, so the suite still runs without it).

**Regression status, and one pre-existing failure — do not re-diagnose it.**

| suite | result |
|---|---|
| `nn/attention` + `generate` | 153 passed, 0 failed |
| `nn/transformer` + `generate` | 96 passed, 0 failed |
| full `nn` + `generate` | 427 passed, **1 failed** |

The single failure is `nn/hf/convert_test.py::test_logprobs_match_after_roundtrip[gemma3]`:
**HTTP 403 on the gated repo `google/gemma-3-270m`**. Verified to fail *identically on the untouched
canonical checkout*, so it is an HF access/credentials issue, unrelated to this work. Fix by
authenticating to that gated repo, or skip it.

**Traps that silently produce a different model — status:**
1. ✅ **Handled + tested.** Chunk order is **`(B, C, x)`** — not `(B, x, C)`. Wrong order trains fine
   and is merely worse, so only a value comparison catches it.
2. ✅ **Handled.** **No activation anywhere in the conv path** (LFM2 passes `activation=None`).
   `OLMo-core/src/olmo_core/nn/convolution.py` `CausalConv1d` **defaults to `activation="silu"` inside
   the fused kernel** (line **27**) → a different operator. `ShortConv` therefore does **not** reuse
   `CausalConv1d`: it holds a plain `nn.Conv1d` (which is what HF does, and works on CPU) and passes
   `activation=None` explicitly on the fused `fla` path. `CausalConv1d` also has **no dilation**
   parameter and its `return output[0]` (line **83**) **drops the final conv state** — both still true
   and both reasons to avoid it here.
3. ⏳ **Still open — applies to the MLP, not the mixer.** **`block_ff_dim` is transformed:**
   `ff = 256·ceil(int(2/3·block_ff_dim)/256)`. Reproduces both released configs (1.2B→8192,
   350M→4608). Miss it and the MLP — **69% of the model** — is ~50% off.
4. ✅ **NEW, found the hard way + tested.** Per-layer overrides go through **`block.sequence_mixer`**,
   not `block.attention`. See §2 above — this one produced a fully-working model with zero LIV layers.

**Parity test caveat:** validate against HF **prefill**, not HF decode. The transformers LFM2 decode
path reportedly drops one tap of history per step via `conv_state.roll(-1)` (flagged, unconfirmed on
`main`). The committed parity test uses prefill for this reason. Also: conv-state `copy_` must be under
`torch.no_grad()`.

Remaining Phase 0 items:
- **Decode/conv state for `ShortConv`.** The mixer is training-complete but has **no incremental decode
  path** — no `conv_state` cache. Needed before any latency or cache measurement. Note the state is
  `[B, d, L_cache]` with `L_cache = k`, so k=15 grows it **5×** over k=3 (not 7×).
- Declarative 16-slot arm builder (meta-device construction already tested).
- Document-isolated packing: `ShortConv` **already honors `cu_doc_lens`** and is tested against
  independent per-document forwards; still needs threading through **attention** in the same batch.
- Audit Dolma2 document-length distribution.

### 3. Phase 1-2 — ✅ MQAR CALIBRATED (2026-07-31), power still open

Harness in `Brainlifts/liv_experiment_research/probes/mqar/` (standalone modules, **not** an
in-place patch of `probes/mqar_patch.py`; the model imports the **same `ShortConv`** the real arms
use). Results JSON committed alongside; full write-up in that directory's `README.md`.
FarmShare jobs **1670928** (positive control) and **1670987** (calibration).

**Calibrated settings:** vocab **256** — *not* Zoology's 8192 — lr 3e-3, attention at (1,3),
**8000 steps × batch 64 = 512k examples**. At 8192 the best of 6 configs reached 0.214 and four sat
at loss 8.32 = `ln(4096)`; at 256 two reached 0.995/1.000. An 8192-way softmax over 4 answers spends
capacity on the output distribution rather than the binding, at this budget.

| capacity grid | floor | success | | distance sweep (D=8) | success | median |
|---|---:|---:|---|---|---:|---:|
| `N64_D4` | 0.250 | 80% | | 64 / 128 / 256 | 100% | 1.00 |
| `N128_D8` | 0.125 | 100% | | **512** | **40%** | 0.16 |
| `N256_D16` | 0.062 | 100% | | 1024 | 0% | 0.15 |
| **`N512_D64`** | 0.016 | **20%** | | | | |

**Use `N512_D64` as the operating point** — off-ceiling on *both* axes, graded scores carry more
information per seed, and the 0.016 floor leaves the most headroom. (The script's own recommendation
of `N512_D8` picks by proximity to 50%; keep that as the secondary, since same-length/8×-less-load
separates capacity from distance.) The distance sweep holds D fixed, so its 100% → 40% → 0% cliff is
**pure retention distance**.

**⚠️ THE 1/D FLOOR — read every MQAR number against it.** A model that learns "the answer is one of
the D values present" without binding anything scores **exactly 1/D**. The loss plateaus form a
legible ladder, all three observed to 2 decimals: `ln(vocab/2)` = "it's a value token" (8.32),
`ln(D)` = "it's one of these D" (1.39), 0 = bound. **The chance baseline is 1/D, not 1/vocab, and it
moves with the config** — 0.10 at D=64 is real work, 0.10 at D=4 is *below* the degenerate strategy.
`degenerate_floor()` and an `above_floor` field are in the harness.

**⚠️ Bimodality holds at low load and BREAKS at high load — this corrects the earlier blanket claim.**
41/45 runs sat at an extreme, but all four exceptions were `N512_D64` (0.05/0.09/0.20/0.56/0.98):
with 64 pairs a model binds *some* pairs, so accuracy is graded. **At high-load rungs report success
rate AND median accuracy vs floor** — a binary threshold scores an arm at 0.55 the same as one at
0.05. The script now reports bimodality per-config instead of pooled.

**⚠️ Do NOT transfer the operating point to real `L0`.** Calibration used 4 layers / attention at
(1,3) / d=128; real `L0` is 16 layers / 6 attention / d=1024. The cliff is **not** a receptive-field
limit — the attention layers are global, so reach is not binding; what degrades is the difficulty of
*finding* the recall circuit as distractors multiply. **The method and the 1/D floor transfer; the
numbers do not.** Re-run the sweep on real `L0` before using these settings in the study.

**Two process failures, both now guarded.** (1) Job 1670922 swept difficulty *before* a positive
control and returned 0.000 everywhere — uninterpretable, since a sweep whose easiest rung scores zero
cannot separate "hard task" from "broken setup". (2) Job 1670963 ran at 96k examples because I fixed
the script but resubmitted a stale sbatch; the config the control solved at 1.000 then scored
0.24/0.25/0.25/0.26/0.93 on the floor. **Under-training is indistinguishable from a too-hard task in
the output.** `mqar_calibrate.py` now owns the budget constants and **refuses to run below them**
(verified). 43 generator tests pin correctness.
- **Measure `s_δ` in the pilot and publish the required n per endpoint before committing to any gate.**
  The existing protocol's CE margin (+0.010 nats) is only reachable if `s_δ ≲ 0.011` at n≥8.
- Drop COPA (needs 19 seeds), WinoGrande and MC-form MMLU (at chance below 1B). Use continuous metrics
  for a 2-18× SNR gain.

### 3.5 ⬅️ **START HERE (2026-08-01): the P1 quality pilot. ~2 GPU-hours. Not yet run.**

Everything needed to launch exists. This is the next action, and it is deliberately *not* the 12-day
study.

**Arms:** `L0` · `F-r128` · `G-grouped` · `N-narrow`, 4 paired seeds each.
**Data:** `/scratch/users/ericrcwu/kda/lm/data/train.npy` — 1.2B GPT-2 FineWeb-Edu tokens, already
tokenized (`meta.json` alongside). The plan had assumed building a corpus from S3; it was already done.
**Compute:** FarmShare `sbatch -p gpu --gres=gpu:1`. Multi-GPU needs an explicit **`--qos=gpu`** —
the default `normal` QOS silently caps at `gres/gpu=1`.

**Why this before the 12-day run.** P1's quality question is *completely open*, and the big study
assumes an answer to it. We know `lowrank r=128` retains **0.929** of activation-weighted energy and
`G-grouped` retains **0.130** — but that was measured on *Liquid's already-trained weights*, and
`F-r128`/`G-grouped` are **exactly parameter-matched** (bit-identical difference from `L0`:
15,728,640). Whether that proxy predicts *from-scratch* training quality is unknown, and GaLore is a
documented case of exactly this kind of proxy failing badly (plain `W=BA` collapses 142.53 vs 15.56
ppl at 1B). **If low-rank and grouped train indistinguishably, the energy metric is not predictive and
the 12-day design needs rethinking.** Two GPU-hours to test the premise under twelve days of compute.

**Blocked on nothing.** Note the `gpu` QOS allows only **4 concurrent jobs** and all 4 slots were
occupied by the sibling KDA-Householder array (`1671411_0..3`, 63 results on disk — **do not cancel
them**). Queue behind them.

**Traps to respect when writing the runner:** `--array` elements each count against
`MaxSubmitPU=32`, so `--array=0-47%4` is rejected outright. And `-c 8` gets silently bumped to 14
CPUs by Slurm — harmless, but don't read it as a config error.

### 4. Phase 3 — the arms

- **3a (P1):** rank sweep {128, 256, 512} + mandatory controls `N-narrow` (just build a narrower
  model), `S-shared`, `G-grouped`, `1G`. **Assert step-0 gate output variance parity with `L0`** —
  otherwise a fixed-std sweep produces a smooth "higher rank is better" curve that is really an
  init-scale curve (error is **monotone in r**: 24-48× too small at default init). Add a full-width
  gate bias init to 1.0 — **and give it to the dense control too**, or you measure the bias not the rank.
- **3b (P2):** `C-near` vs `C-far` (a controlled test of a real literature disagreement — CLA says
  non-adjacent loses, Character.AI and Gemma 3n deploy it fine, nobody ablated it). Mandatory
  competitor **`A-fewer3`**: 3 attention layers instead of 6 matches CLA2's capacity **and halves read
  bandwidth**, which CLA structurally cannot. Keep **GQA-8 primary, MQA secondary with loss-spike
  monitoring** — GQA's Appendix A reports MQA-from-scratch had "frequent loss spikes" and diverged, and
  CLA's MQA results were *uptrained*, not from scratch. **Retrieval endpoints primary** — this is the
  contribution: no cross-layer-sharing paper reports needle/passkey/MQAR.
- **3c (P3):** `k5/k9/k15` **inside the real gated block first.** This arm now does real science: it
  tests whether the published negative result survives the gate (see caveat below). FLA's Triton conv
  backend has **no width limit**, so a fused dense k=15 baseline is fair and available.

### 5. Phase 4-5 — confirmation, then systems

≥8 **fresh** paired seeds never used in selection. Systems only on survivors. No 32K quality claim
without a matched 32K training stage.

---

## Open items / live caveats

- **§5.3 item 0a is the newest change.** The adverse P3 evidence used **ungated** conv slots — Tian et
  al. (arXiv 2607.18413, Qwen3-1.7B, multi-branch mixed widths **12.79 → 13.28**, worse) and Sieberling
  et al. (arXiv 2606.03825, width flat past k=3) both bolt a *residual* conv onto a transformer with
  no multiplicative gate. In LFM2 the conv sits **inside** two gates, so branch weights and gates
  interact multiplicatively — a different landscape. **Their negative results may not transfer.**
  Counterweight: Sieberling's *rank* sweep gained 0.25 ppl (still climbing) while *span* gained
  **0.00** → gains come from conditioning capacity, not receptive field. Prior on the full router:
  **~20-25%**, up from 15%.
- **P1's motivating premise is falsified — CONFIRMED under both metrics, do not re-litigate.**
  Plain spectra of released `LiquidAI/LFM2-350M`: gates at effective rank **771-790 of 1024**,
  indistinguishable from the value stream (790.1 vs 790.5). The activation-aware follow-up
  (`probes/spectra_v2.py`, **32,768 tokens**, `rank(Σ_x)=1024` confirmed full) settles it: aware rank
  drops ~36% (gates 493.3) but **the value-stream control drops identically (507.8)**, because all
  three tensors read the same `x`. The `out_proj` and random-Gaussian controls collapse *less* (0.784).
  So the collapse is a property of the input distribution, not of gates. Keep *"gates **tolerate** low
  rank"* permanently.
  **But P1's feasibility case got materially stronger:** rank-128 retains **92.6% of
  activation-weighted energy** vs only **45.8%** of plain Frobenius energy. Claim this instead:
  *"rank-128 discards 7% of the energy that actually reaches the output — we test what that costs."*
  **Methodology trap, recorded because it nearly produced a false positive:** a first pass used 568
  calibration tokens for a 1024×1024 covariance, making `Σ_x` rank-deficient by construction and
  reporting a spurious **3.0× collapse to 267** that looked like strong support for P1. Any
  activation-aware spectral claim needs tokens ≫ d and a reported `rank(Σ_x)`. Convergence is still
  rising at 32k (573.5 → 600.2 → 608.0 for L0), so current numbers are a mild underestimate.
- **Report effective rank and energy-at-r, not stable rank** (srank reads 26-48 here while a random
  Gaussian scores 258 — it's dominated by σ₁).
- **Param-matched ≠ compute-matched.** At 350M, attention-score FLOPs are 2.4% of 6ND at 4K but
  **18.9% at 32K**; the `L0` vs `A16-P` difference is **31.6% at 32K**. Match on `num_flops_per_token`.
- **The one efficiency claim that IS testable at trainable contexts belongs to the TOPOLOGY, not the
  three proposals:** mostly-LIV vs param-matched all-GQA saves 20 KiB/token, hitting a **10%
  decode-traffic win at T ≈ 4,121** — right at the training context. Do not let this be attributed to
  P1/P2/P3.
- Unresolved: μP is not coordinate-checked in OLMo-core (use `fan_in` init + the ladder's empirical LR
  formula); Dolma2 doc-length distribution unmeasured.
