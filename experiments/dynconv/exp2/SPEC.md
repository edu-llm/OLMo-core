# EXP-2 FROZEN SPEC — read this before writing a line of code

**Owner:** team lead. Sub-agents A and B must NOT edit this file. If you believe it is wrong,
report to the lead; do not silently diverge.

Date frozen: 2026-08-05.

## 0. What Exp-2 is

The synthetic mechanism study. Does making the LFM2 short-conv filter input-dependent buy
context-dependent local composition, **beyond** what LFM2's existing B/C gates already do and
beyond "one more multiplicative degree of freedom"?

Small models: `d_model=128`, 6 layers, MQAR from scratch. ~1.8M params, ~minutes/run on CPU,
seconds on a GPU.

## 1. Arms — 4, each in TWO topologies

| Arm | Name | Mechanism |
|---|---|---|
| **S1** | `static` | static-LIV baseline. `ShortConv` as released. |
| **S2** | `permuted` | **permuted-conditioning control.** Identical params, FLOPs and kernel to S4, but the conditioning stream `z` is SHUFFLED along the sequence axis so it carries zero positional content. |
| **S3** | `dynqkv` | static LIV everywhere + dynamic conv on Q,K,V inside the GQA blocks. (Only defined in the hybrid topology; see §1.2.) |
| **S4** | `dynamic` | Dynamic-LIV in every LIV block. |

### 1.1 Why S2 and not a param-matched MLP-widened arm

The council killed the param-matched arm. Do not revert it. Reasons, in order:

1. The added params are **0.185% of N**, worth ~**0.0003 nats** by Hoffmann scaling — **95x below**
   the effect being chased and ~1/34 of one seed-SD. So `C4 − C2` is statistically identical to
   `C4 − C1`: two "independent" primaries collapse to one.
2. It is literally unbuildable: params per +1 MLP unit = `3·d·L`, giving a target of **13.33 units
   — not an integer**, and OLMo-core rounds ff to a multiple of 256, overshooting ~**19x**.
3. **Capacity was never the live confound.** The live confounds are (a) LFM2's existing B/C gates
   and (b) "one more multiplicative degree of freedom."

**S2 is the only arm that can distinguish "input-dependent local composition" from "one more
degree."** It is the scientific core of Exp-2.

> **The decision rule that matters: if S4 beats S1 but does NOT beat S2, the hypothesis is
> unsupported.** Pre-register this.

### 1.2 Topologies — every arm gets both

| Topology | Layers | Why |
|---|---|---|
| `hybrid` | 6 layers, 4 LIV + 2 GQA | LFM2-shaped; the proposal's configuration |
| `allliv` | 6 layers, 0 attention | **Ceiling guard.** With 2 GQA layers present, attention can solve MQAR by itself and mask any conv-mechanism difference — a null for the wrong reason. R5 F5(i) measured the in-tree probe at **100% success, all seeds 1.00** on `N128_D8`/`N256_D16` with attention at 2 of 4 layers, and states the cliff is *not* a receptive-field limit because "the attention layers are global." |

S3 is undefined in `allliv` (no GQA blocks to put a dynamic conv in). Report it as N/A, do not
silently substitute S1.

## 2. W sweep — and W=2 is a NEGATIVE CONTROL, not just a data point

Sweep `W ∈ {2, 3, 4, 8}` if budget allows; `{2, 3, 4}` is the floor. `W=3` must always be present
(LFM2 fidelity anchor).

**It is a verified theorem** (`docs/dynconv-review/orch_verify_W_minus_2.py`, reproduced
independently by R5 analytically) that:

- static tap family `κ[t,k] = C_t·a_k·B_{t−k}` has Jacobian rank **exactly `2T+W−3`**;
- genuinely new dynamic DOF = **`W − 2` per position per channel**;
- **at W=2 the dynamic block is an EXACT reparameterization of the static block** — max
  log-residual `8.3e-16`, a constructive realization, not a fit.

Therefore:

> **W=2 FALSIFICATION CONTROL.** At W=2 the dynamic arm has **zero** new degrees of freedom.
> A W=2 dynamic-vs-static difference that exceeds seed noise **is a bug, not a result.** This is
> the cheapest falsification test in the whole program.

Also required: a **static-W-matched control at every W**. Without it the width sweep is confounded
and cannot separate "span helps" from "dynamism helps."

## 3. Init spec — NON-NEGOTIABLE

```
U = 0          (zeros)
V = random     (kaiming/normal, fan-in over the TRUE contraction — see §5.3)
alpha = 1      LEARNABLE
```

**Zeroing both `U` and `α` makes `∂L/∂U = ∂L/∂V = ∂L/∂α ≡ 0` FOREVER.** The run trains stably,
every arm ties, and it reads as a clean replicable negative — the most expensive possible failure
because it looks like science. NVIDIA shipped this exact bug (`dynamic_conv.py:80-84`).

R5's measured gradient table:

| variant | ‖∂L/∂U‖ | ‖∂L/∂V‖ | ‖∂L/∂α‖ | verdict |
|---|---|---|---|---|
| `U=0, V rand, α=1` | 3.08e-02 | 0.00e+00 | — | **CORRECT** |
| `U=0 AND α=0 learnable` | 0.00e+00 | 0.00e+00 | 0.00e+00 | **DEAD START** |
| `α=0 fixed` | 0.00e+00 | 0.00e+00 | — | **DEAD** |

`‖∂L/∂V‖ = 0` at step 0 is **correct and expected** (LoRA). What must be true is `‖V.grad‖ > 0`
**after one optimizer step.**

## 4. Endpoints and seeds — the proposal's design is a coin flip; do not reproduce it

### 4.1 Do NOT gate on accuracy

Measured MQAR σ in this repo is **42–48.4 pp** (`KDA/HANDOFF.md:420-457`, n=20 one-way ANOVA,
η²=5.9% load vs **94.1% seed**). Per-seed accuracy on ONE identical config:
`50.00 / 90.82 / 99.80 / 98.44 / 2.73%`.

Required n for 80% power (R3 F8, exact noncentral-t, paired):

| per-arm σ | ρ | s_δ | n for 5 pp | n for 10 pp | n for 15 pp |
|---:|---:|---:|---:|---:|---:|
| 42.0 pp | 0.0 | 59.4 pp | **1,110** | 279 | 126 |
| 42.0 pp | 0.5 | 42.0 pp | **556** | 141 | 64 |
| 48.4 pp | 0.0 | 68.5 pp | **1,473** | 370 | 166 |
| 48.4 pp | 0.5 | 48.4 pp | **738** | 186 | 84 |

The proposal's n=5 has **24–38% power** at the hypothesized 8–15 pp while firing **18.75%** of the
time under the pure null (`P(≥4 of 5) = 6/32`). A likelihood ratio of ~1.5–2:1. That is a coin flip.

> **PRIMARY JOB OF EXP-2: MEASURE σ and report the required n**, so later experiments can be
> powered honestly. Run **~10 seeds, paired across arms**. Report σ per (arm, topology, W, config)
> plus the pooled within-cell σ, and invert it to a required-n table.

### 4.2 Success rate over seeds, not mean accuracy

`docs/liv-brainlift-experiment-design.md:1207` — the distribution is **bimodal**: a run either
finds the recall algorithm or sits at chance. A bimodal metric's mean is not a location parameter.

**But** the in-tree README refines this: bimodality holds for 41/45 runs and **BREAKS at
`N512_D64`**, whose seeds spread continuously (0.05, 0.09, 0.20, 0.56, 0.98). So:

> Report **success rate AND median accuracy vs the `1/D` floor**, per cell. Never a bare mean.

### 4.3 The floor is `1/D`, not `1/vocab`

`degenerate_floor(cfg) = 1.0 / num_pairs`. A model that learns "the answer is one of the D values
in this sequence" without binding anything scores exactly `1/D`. Six of twelve control trials sat
at 0.208–0.274 with losses of 1.40–1.76 against `ln(4) = 1.386` — a **fully-learned wrong
algorithm**, not partial recall. Always report against the floor.

### 4.4 Continuous scoring where possible

Per-token likelihood rather than accuracy is worth a **2–18x SNR gain** — far cheaper than buying
the equivalent in extra seeds. Log both. This is the single biggest free lever available.

### 4.5 Calibrate off ceiling AND off floor first

The in-tree harness has calibration recorded (`mqar_calibration.json`, FarmShare 1670987;
`mqar_positive_control.json`, 1670928). Use it.

Measured, 4-layer d=128, attention at (1,3), vocab 256, lr 3e-3, 8000x64:

| config | 1/D floor | success | per-seed |
|---|---:|---:|---|
| `N64_D4` | 0.250 | 80% | 0.27 0.99 1.00 1.00 1.00 |
| `N128_D8` | 0.125 | **100%** | all 1.00 — **CEILING, unusable** |
| `N256_D16` | 0.062 | **100%** | all 1.00 — **CEILING, unusable** |
| `N512_D64` | 0.016 | 20% | 0.05 0.09 0.20 0.56 0.98 |

**Drop the ceiling-saturated configs.** A task at 1.000 cannot discriminate arms — this is exactly
the mistake the cited paper makes (In-Context Recall and Noisy Recall both saturate at 1.000 for a
static baseline).

**`N512_D64` is primary** (off-ceiling on both axes, graded, floor 0.016). `N512_D8` secondary
(same length, 8x less capacity load — the pair separates capacity from distance).

**The calibration constants do not transfer unchanged.** They are for a 4-layer d=128 model with
attention at (1,3). Exp-2's `allliv` topology has NO attention, so its cliff will move a lot.
Re-calibrate the `allliv` baseline before trusting any operating point. Run the positive control
first: a sweep whose easiest rung scores zero cannot separate "hard task" from "broken setup."

### 4.6 PRE-REGISTER that Memorize REGRESSES

The cited paper's own numbers: **0.856 static → 0.795 dynamic = 6.1 points down.** That would fail
the proposal's own "control tasks must not drop >2 points" criterion. Expect it; do not be
surprised by it; do not treat it as a bug.

### 4.7 Budget constants are part of the calibration

`CALIBRATED_STEPS=8000`, `CALIBRATED_BATCH_SIZE=64` = **512,000 examples**. Job 1670963 failed
exactly here: a stale sbatch carried 96k examples (5.3x short) and `N64_D4` — which the control
solved at 1.000 — scored 0.24/0.25/0.25/0.26/0.93 with four runs parked on the `1/D` floor.
**Under-training is indistinguishable from a too-hard task in the output.** The harness must
refuse to run under-budget outside an explicit smoke-test flag.

## 5. Traps — assert against every one

1. **Zero-init both factors = dead branch.** §3. `U=0`, `V` random, `α=1` learnable.
2. **`block.sequence_mixer`, NOT `block.attention`** — the wrong attribute **silently no-ops** and
   trains happily. Assert the exact integer count of dynamic modules per arm.
3. **`CausalConv1d` defaults to `activation="silu"`** (`olmo_core/nn/convolution.py:37`); real LFM2
   has **no activation** in the conv path. Chunk order is **(B, C, x)** — pre-gate, post-gate,
   value. Permuting still trains, just worse: a silent failure.
4. **DTensor-unsafe init:** `w[...] = x` in `init_weights` lowers to `aten.fill_.Tensor`, which has
   no sharding strategy, and dies under FSDP. CPU tests **structurally cannot catch it.** Route
   every zero-init through `_apply_init`. This already killed `run_019fbf9f`.
5. **bf16 dead zone:** `1.0 + ε` rounds away below `2^-8 = 3.90625e-3`. Derive engagement floors
   from this, not from taste.
6. **Assert magnitudes, not existence.** `loss ≈ ln(vocab)` at init, not `grad is not None`.
7. **Fan-in must be corrected on ALL branches or none.** A one-sided correction previously biased a
   contrast toward the hypothesis — see §5.3.

### 5.3 The fan-in trap, specifically

Repo memory `fan-in-correct-one-branch-only`: `mqar_model.py` **constructs `ShortConv` directly and
never calls `init_weights`**, so the grouped arm ran at ~1/128 of dense activation scale *on the
probe used to justify that arm*. And `kaiming_uniform_` on a **3-D** parameter derives
`fan_in = size(1) * receptive_field`, not the true contraction — off by `√W` for a `(R, d, W)`
tensor.

Requirements:
- Call `init_weights` (or an explicit equivalent) on **every** arm, including S1.
- The dynamic generator's `V` must be initialized against the **true contraction** (`d`), spelled
  out rather than delegated to a helper that can be misled by the 3-D shape.
- **Assert step-0 activation-scale parity across ALL arms**, not just the new one.

## 6. Mandatory pre-flight (adopted from `R7-redteam.md` §3)

Adapted to `d_model=128`, `R` per §7, and `W` swept. Every check asserts a **magnitude predictable
from theory**.

| # | check | expected | catches |
|---|---|---|---|
| 1 | `numel(V)`, `numel(U)` exact integers | `d·R` / `R·W·d` | wrong rank/reshape (a common error forgets the `k` factor and is off by `W`, and still trains) |
| 2 | `α` learnable, nonzero, **in the optimizer**, NOT in the weight-decay group | — | param created after optimizer ⇒ never updates while every other check passes |
| 3 | `α=0` ≡ static path | rel err < 1e-6 fp32 | logic bug. **Necessary, NOT sufficient — passes trivially for a mechanism wired to nothing.** Pair with check 7. |
| 4 | bf16 tap dead zone characterised | `w[-1]==1.0` below 3.90625e-3 | documents, does not fail |
| 5 | grad magnitude ratio vs `out_proj` in the SAME block | ∈ [1e-4, 1e2] | a grad 12 orders below its neighbours is functionally zero and `is not None` passes |
| **5b** | **`‖V.grad‖ > 0` AFTER one optimizer step** | > 0 | **the dead-branch bug.** At step 0, `‖V.grad‖ == 0` is CORRECT. |
| 6 | shared params bit-identical across arms at same seed — `torch.equal`, not `allclose` | exact | pairing is false ⇒ the power analysis is void |
| 7 | param counts per arm (exact integers) **and** module counts **and** layer indices | pre-declared | the silent-no-op trap. An exact total can hide two offsetting errors. |
| 8 | init loss ∈ `[ln V, ln V + 0.25]`, **every arm** | vocab 256 ⇒ `[5.5452, 5.7952]` | uninitialized weights, broken loss. Has caught uninit weights ~4x in this repo. |
| 9 | **engagement floor** `E_l = ‖α·Δw‖_F / ‖a‖_F` per layer, logged from step 0 | ≥1e-2 rising | **ABORT below 1e-3.** Physical floor: at 1e-3 the deviation is below bf16's rounding threshold on the current-token tap, so the mechanism provably cannot affect the dominant tap. |
| 9b | ablate-at-eval `Δloss = loss(α=0) − loss(α̂)` | >0.01 if load-bearing | separates "bug" from "redundant" from "harmful" |
| 10 | `‖U‖` trend not monotone-decreasing after first 5% | — | weight decay winning |
| 11 | grad-norm parity across arms | within 2x | global clipping biting one arm differently |
| 12 | `activation is None` in the conv path | `None` | the silu default |
| 13 | **W=2 exact-reparameterization check** | dynamic ≡ representable by static | the theorem; see §2 |

Report `E_l` **per layer, never averaged** — depth-scaled `out_proj` init means late layers start
smaller, so a mean over layers can sit above the floor while most layers are dead.

Where a check has a **negative control** (a source mutation that must make it fail), state it and
run it. Per `test-must-call-not-recompute`: a test that re-derives the code's own formula passes
when the code changes. A guard that has never failed is not known to work.

## 6.5 TWO ADDITIONAL MANDATORY GATES (added 2026-08-05, after the venue change)

### 6.5a Assert the ABSOLUTE loss lands in band BEFORE reading any delta

The Exp-0 team measured that a **missing BOS token puts LFM2-350M 2.4–3.8 nats off** — about **100x
the effect being chased** — and it **fails silently**: the run trains, the curve looks plausible, and
every between-arm delta is computed on top of a broken absolute number.

At this scale the equivalent discipline is:

- **init loss ∈ `[ln 256, ln 256 + 0.25]` = `[5.5452, 5.7952]`, asserted on EVERY arm** — this is
  check 8, and it is now a **hard gate**, not a warning.
- If the generator uses a BOS/sentinel at position 0, **assert it is present at position 0 of every
  batch**.
- **Refuse to report a between-arm difference if the absolute number is out of band.** A delta
  computed on top of a broken absolute is not a small error, it is a different experiment.

This is the `green-that-means-nothing` scar restated: assert magnitudes, not existence.

### 6.5b Pin `use_fla` IDENTICALLY on every arm, and LOG the realised kernel path

`short_conv.py:185` defaults `use_fla=True`, while `fla` is **absent** in many environments — so
`has_fla()` returns False and the forward silently runs plain `nn.Conv1d`.

The failure mode: if one arm ever resolves to a fused kernel and another does not, that is a **fused
treatment against an unfused baseline**, which biases the contrast **toward the hypothesis**.

Required:
- `use_fla` pinned to the **same value on every arm**, set explicitly, never left to the default.
- **Log which backend each arm's conv resolved to, per arm, at step 0.**
- **Assert baseline and treatment resolved to the same backend family.**

## 7. Geometry

- `d_model = 128`, `n_layers = 6`
- `hybrid`: attention at 2 of 6 layers. `allliv`: no attention.
- vocab 256 (calibrated — NOT Zoology's 8192; see §4.5 rationale in `mqar_data.py:163-170`)
- `R` (generator bottleneck): scale from the 350M spec. At d=1024, R=16 ⇒ `R/d = 1/64`. At d=128
  that is R=2, which is degenerately small. **Use R=16 at d=128** and report `R/d = 1/8`
  explicitly as a deviation, with the reason: the paper's own rank curve is still descending at
  R=128, so R is the steep axis and starving it at d=128 would test the wrong thing. Pre-register
  this choice; do not tune it.
- Generator: `Δw = α · U(V h)`, conditioned on the **normalized block input `h`**, not the gated
  stream. Reasons: matches the cited paper (allows fusion with the input projection); `h` is
  normalized whereas `B_t ⊙ x_t` is a product of two unnormalized projections and LightConv's
  headline failure was that unnormalized dynamic filters **diverged**; and conditioning on the
  gated stream would re-entangle the two mechanisms the experiment exists to separate.
- Optional cheap ablation if budget allows: `Δw = α·U·σ(V h)` (SiLU) vs linear. JetBlock's shipped
  generator is a nonlinear bottleneck MLP with SiLU, not a bare linear map.

## 8. Venue

**AWS platform**, via the `edu-llm/platform` GitHub Actions form, using the `edullm-platform-runs`
skill. Routine single card is `gpu-1xl40s` at $1.86/hr. **There is NO H100.**

> **NOBODY SUBMITS.** Prepare the submission, validate offline, then STOP and report the exact
> command with evidence offline validation passes. The user gives the go-ahead.

Local CPU correctness checks and smoke tests are **encouraged** — they are not AWS submissions.

## 9. Deliverables

- Code: `docs/dynconv-review/build/exp2/`
- Design doc: `docs/dynconv-review/build/EXP2-DESIGN.md` — arms, seeds, the σ measurement plan,
  calibration evidence, pre-flight results, the prepared submission. Written **incrementally**;
  this machine has died mid-run before.

## 10. File ownership — do not cross

| Owner | Files |
|---|---|
| **Lead** | `SPEC.md`, `w2_falsification.py`, `EXP2-DESIGN.md`, submission config |
| **Sub-agent A** | `arms.py`, `dynamic_conv.py`, `preflight.py`, `test_arms.py`, `test_preflight.py` |
| **Sub-agent B** | `mqar_harness.py`, `calibration.py`, `sigma.py`, `test_harness.py`, `test_sigma.py` |

Shared read-only inputs: everything under
`/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer/`
and `/Users/ericwu/Developer/Capstone_LLM/Brainlifts/liv_experiment_research/probes/mqar/`.
