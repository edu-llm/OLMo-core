# Audit scratch — independently verified numbers (2026-07-31)

Working notes backing the audit of `phase-0-1-runbook.md`. Every number here was recomputed
locally, not taken from a subagent report. Scripts were throwaway (`/tmp`); the results are below.

## Mathematics — Phase 0 is SOUND

The §4.5 rank-two algebraic oracle is **correct as written**. Verified numerically over 200 random
trials (float64, K=6, V=5):

| Variant | Max error vs sequential recurrence |
|---|---:|
| Runbook formula, `U=[u1-u2ρ, u2]`, `V=[Dk1, Dk2]` | **4.0e-15** (exact) |
| Drop the cross-term ρ | 2.2e+01 |
| Move ρ onto the u2 column | 4.3e+01 |
| Use plain `k` instead of `Dk` in V | 3.2e+01 |

Every term is load-bearing; the oracle is genuinely discriminating. `V = Dk` is right because D is
diagonal (hence symmetric), so `V^T S_{t-1} = k^T (D S_{t-1})`.

§4.6 identity controls also verified:

- `β2=0` vs R1: max diff **exactly 0.0** — a true identity.
- `v2=0, β2≠0` vs R1: max diff **2.99** — the negative control is a real trap, correctly specified.
- tied-K state update rank = **1**; independent-K rank = **2**. The tied-K arm is a genuine rank
  control, so the §5.8 "DP2 ties tied-K" interpretation is mechanistically well founded.

## Corrections and additions to the math verdict

The §4.5 oracle is correct (above), but five further math findings survive verification. Two of them
**correct** my earlier "the math is sound, problems lie elsewhere" framing.

**The 1e-11 bar can never touch the production kernel.** `kda_householder.py:737-739` rejects float32
on the Triton backend and the fwd kernel accumulates in `tl.float32` unconditionally. So P0.3's
float64 oracle can only validate `backend="torch"` — the very reference the kernel is supposed to be
checked against. The kernel is comparable only at bf16, where the repo's own constant is
`ATOL = RTOL = 2e-2` (`kda_householder_test.py:30-31`). A seeded cross-term bug at K=V=64 produces
median relative error 3.5e-3, so **~90% of seeds pass the 2e-2 check.** This is the principal
false-PASS pathway and the runbook presents 1e-11 as *the* Phase-0 bar without noting it applies to
a backend nobody trains on.

**FP32 at atol 1e-11 with rtol=0 is unsatisfiable.** `float32 eps = 1.19e-7`; attainable absolute
accuracy at these output scales is ~1e-8, four orders above the demand. §4.3's clause makes any FP32
semantic test fail unconditionally. The 1e-11 bar is float64-only.

**Final-state comparison is blind to the readout position.** After the doubled sequence, `S` is
*identical* whether you read after factor 1 or factor 2, while the outputs differ ~44%. P0.1 item 3
and the P0.2 case list emphasize final state; a state-only test would pass an implementation that
reads out at the wrong microstep — exactly the bug §4.7's virtual-token test claims to exclude.

**tied-K is EXACTLY rank 1 — this escalates a §5.8 decision.** Verified: 2nd singular value of the
tied-K update = **2.26e-15** (machine zero). Algebraically it collapses to one R1 step with
`β_eff = β1+β2−β1β2‖k‖²` and `v_eff` a convex combination of `v1,v2`. So tied-K spans the *same
function class* as R1. Therefore "DP2 ties tied-K → may not claim rank-two key geometry" is too
weak: it means DP2 ties an R1-equivalent model, which is nearly the "R1-P matches DP2 → **stop the
program**" row. Merge the rows or escalate tied-K to a stop condition.
Bonus: under reflection `β1=β2=2, ‖k‖=1` gives `β_eff = 0` exactly — **but the update does NOT
vanish.** (Correction to an earlier version of this file, caught in review.) Measured: `max|S₂−A| =
5.75`, a rank-1 write equal to `k(2v₂−2v₁)ᵀ` to 8.9e-16, 2nd singular value 1.1e-15. What `β_eff = 0`
means is that the `β_eff/v_eff` reduction **degenerates**: `v_eff = (c₁v₁+c₂v₂)/β_eff` is a 0/0 form,
undefined rather than zero. The composed erase multiplier `(1−β₁‖k‖²)(1−β₂‖k‖²) = (−1)(−1) = +1`
restores the `S_{t−1}` read-back exactly while a nonzero value write survives. So reflection tied-K is
not an R1-with-`β_eff` model at all — which is the actual (and stronger) mechanism for why the two
beta regimes must never be pooled.

**"No negative-eigenvalue multiplier" is a claim about `β‖k‖²`, not `β`.** The erase `I − βkkᵀ` has
eigenvalue `1 − β‖k‖²`; a **strict** β=0.9 with ‖k‖=1.5 gives **−1.025**. The module normalizes k
(`recurrent.py:1300`) but P0.3/P0.4 are *operator*-level and `kda_householder.py` never normalizes
(grep: zero hits). Restate as `β‖k‖² < 1`, or make key normalization an explicit §3.1 precondition.

**A zero query NaNs the oracle.** `functional/__init__.py:16-18` `l2_normalize` is a bare `x/‖x‖`
with no epsilon → NaN on zeros (verified; `F.normalize` returns zeros). `recurrent.py:1299` applies
it after the conv, i.e. after §4.4's injection point. §4.4's "zero query" on virtual position 1 would
NaN the entire doubled sequence. Inject zeros *after* `l2_normalize`, or use a unit dummy query and
discard its output.

**Both negative controls can be made degenerate.** Factor-order-swap separation scales with
`|k1·k2| = O(K^{-1/2})`: at K=256 the 1st percentile is 2.1e-6 and exactly-orthogonal keys give
**0.0** — the control cannot fail. The `v2=0, β2≠0` control gives **0.0** when `S_prev=0` and
`β1v1=0`. Both need an asserted minimum-separation floor, not just "must fail."

**No external R=2 reference — Phase 0 is self-referential.** P0.2 and P0.3 are independent
*implementations* of §3.1, not independent *specifications*. If §3.1 is wrong, everything passes.
The repo documents an external anchor (`kda_householder.py:689-693`: `fla.ops.gated_delta_product.naive`
to float64 ulp when `g` is constant along K) that the runbook uses for neither.

**Triton drops the final-state gradient path.** `mark_non_differentiable(final_state)`
(`kda_householder.py:639-640`) and the backward deletes `dht` (`:654`) — no cotangent for the carried
state, while the torch backend's *is* differentiable. P0.2's "prefix/suffix recurrence handoff" is
exactly where you want those gradients. Also `l2_normalize`-free operator vs module differ in the
differentiable set; P0.4 never says which level it gates.

**`0 < β < 1` fires spuriously in bf16.** `σ(ℓ) == 1.0` exactly for ℓ ≥ 6.235 in bf16. One
unconstrained `w_b` logit crossing ~6.2 trips the strict-arm assertion on a correct model. Assert
`0 ≤ β ≤ 1` on the realized tensor, or evaluate in fp32.

**"No superlinear error growth" is not operationalized.** Measured (bf16-in/fp32-state vs fp64,
T=32→1024): 6.3e-3 → 8.6e-3, doubling ratios 1.02/1.25/0.94/1.09/1.05 (√T would be 1.41, linear 2.0).
Specify: fit `log e = a + p·log T` over a stated ladder, require `p ≤ 1 + margin`.

**`D_t` is never defined in §3.1**, and §4.5's `[D_t k]` is correct only because `diag(exp(g))` is
symmetric. With a general `D`, the literal form errs 2.31 while `[Dᵀk]` gives 3.8e-16. Define
`D_t = diag(exp(g_t))` or write the erase as `Kᵀ(D_t S_{t-1})`.

**Prefix/suffix handoff isn't testable at module level.** `CausalConv1d.forward` (`convolution.py:53`)
takes and returns no conv state, so a module-level split zero-pads the suffix's first 3 positions and
fails for reasons unrelated to the recurrence. Operator-level only.

## §3.2 DP2-budgeted — vacuous assertion + unmatched write mass

`β1+β2 = b·π + b·(1-π) = b` identically (verified to 1.1e-16). Since `b = σ(ℓ_b) ∈ (0,1)`, the
required assertion `β1+β2 ≤ 1` **can never fire**. It is decoration, not a check.

Worse, the arm is not write-mass matched:

| Arm | Mean total β per token | Max |
|---|---:|---:|
| DP2-strict (`2×σ`) | 1.000 | 2.0 |
| DP2-budgeted as written (`b=σ`) | 0.500 | 1.0 |
| DP2-budgeted with `b=2σ(ℓ_b)` | 1.004 | 2.0 |

As written, budgeted has **half** strict's write mass, so "strict beats budgeted" is confounded
with strict simply writing twice as hard — the exact confound P1.4 condition 6 exists to exclude.
Fix: `b = 2σ(ℓ_b)`, which matches the strict distribution while still sharing one budget.

## Resolution — the two-stage sizing rule (§5.8.0), adopted 2026-07-31

The gate was rebuilt. Conditions 3 and 4 are now confidence bounds (`L₉₅(d) > 0` with a separate
+5pp practical floor; guardrails fail only when `U₉₅(d_g) < −2pp`), and the sign clause is gone —
it has a floor p of 1/8 at n=3 and gets *harder* as n grows, penalizing the fix.

**The seed count is measured, not assumed.** P1.1 already runs 5 bundles × 4 tasks × 3 settings
(60 R1 jobs) plus 20 R1-P confirmations, so σ_t falls out at **zero marginal cost** — the earlier
framing of this as a separate "~$30 probe" was wrong.

Sizing requires **both** power ≥0.80 at +5pp *and* decidability ≥0.80 (P(CI half-width < δ=3pp)).
Decidability is the binding constraint — sizing on power alone leaves the equivalence rows
undecidable ~50% of the time:

| measured σ_d | n | triage jobs | total waves | ~cost @10min/wave | power | decidability |
|---:|---:|---:|---:|---:|---:|---:|
| ≤2.0pp | 4 | 160 | 32 | $350 | 1.00 | 0.82 |
| ≤2.5pp | 5 | 200 | 37 | $395 | 0.99 | 0.82 |
| ≤3.0pp | 6 | 240 | 42 | $440 | 0.98 | 0.81 |
| ≤3.5pp | 8 | 320 | 52 | $530 | 0.98 | 0.88 |
| ≤4.0pp | 10 | 400 | 62 | $625 | 0.98 | 0.91 |
| ≤5.0pp | 12 | 480 | 72 | $715 | 0.95 | 0.80 |
| **>5.0pp** | — | — | — | — | — | **do not launch** |

`n=3` appears nowhere: beyond 21% power, its equivalence half-width needs `s_d < 1.78pp`, so the tie
rows return "underpowered" 78–99% of the time. Three bundles cannot carry the gate in either
direction.

σ_t → σ_d conversion (3-task composite, ρ_T=0): `σ_d = σ_t·√(2(1−ρ))·√(1/3)`.

| σ_t | ρ=0.5 | ρ=0.7 | ρ=0.85 |
|---:|---:|---:|---:|
| 6pp | 3.46 | 2.68 | 1.90 |
| 8pp | 4.62 | 3.58 | 2.53 |
| 12pp | 6.93 | 5.37 | 3.79 |
| 20pp | 11.55 | 8.94 | 6.32 |

ρ is a design variable bought free by byte-identical data/task/bank streams across arms — 0.5→0.85 is
worth ~1.8× in σ_d and ~4× in n. Buy it before buying seeds.

**Full-plan cost:** Phase 0 $45–$134 + Phase 1 $395–$530 at the likely sizings → **~$440–$665**;
worst plausible ~$870. Wave overhead is a fixed 12 (smoke 1 + calibration 8 + confirmation 3) on top
of `5n` triage waves.

## Corrections to THIS FILE's own earlier claims (caught by the review teams)

Three things I asserted here were wrong. Recorded rather than quietly deleted.

1. **"The reflection update vanishes"** — false. See the tied-K section above; the update is 5.75, a
   rank-1 write of `k(2v₂−2v₁)ᵀ`. `β_eff = 0` makes the reduction *degenerate* (`v_eff` is 0/0), not
   the update zero.
2. **"Changing `GatedDeltaNet`'s default silently changes 10 production 7B runs"** — false. All 10
   scripts pass `allow_neg_eigval=True` **explicitly** at the call site (verified: 10/10, e.g.
   `OLMo-hybrid-7B-pretrain.py:78`). Flipping the default would change nothing there. The real
   default-reliant consumers are the tests in `src/test/nn/` and
   `src/scripts/train/ladder/gemma_like_ladder.py`. The conclusion (don't touch `recurrent.py`)
   survives; the stated reason did not.
3. **"26 waves total"** — wrong by one. Calibration and R1-P confirmation **cannot be pooled**,
   because confirmation runs at the *selected* setting and selection needs calibration results:
   `⌈60/8⌉ + ⌈20/8⌉ = 8 + 3 = 11`, so the total is **27** (29 with MQAR), and the overhead over the
   triage row is ~80%, not 73%. `⌈80/8⌉ = 10` is the pooled figure and assumes an ordering the
   design forbids.

Also corrected: the §4.5 corruption magnitudes (22/43/32) are **scale-dependent**, not reproducible
constants — they track `‖S₀‖‖v‖` and land near ~1 at other scales. The durable claim is the ~15
orders of magnitude separation from the correct residual. And "final state identical across readout
position, outputs differ 44%" was two errors: `S₁ ≠ S₂` (the invariant is the *returned* final state,
always post-factor-2), and 44% was a maximum presented as typical — the median is 11%.

## Two always-fires defects my own D5 fix introduced

**The half-width conjunct was vacuous.** I wrote "CI within [−δ,+δ] **and** half-width below δ."
Containment implies the width: from `m−h ≥ −δ` and `m+h ≤ δ`, subtract to get `2h ≤ 2δ`. Zero
violations in 2M random trials. Removed; containment now carries it explicitly.

**And δ=3pp at n=3 made "inconclusive" the near-certain verdict.** Half-width at n=3 is
`t₀.₉₅,₂·s_d/√3 = 1.686·s_d`, so containment needs `s_d < 1.78pp`:

| σ_d | P(any decision reachable) | P(inconclusive) |
|---:|---:|---:|
| 3.58pp (best case) | 0.219 | **0.781** |
| 10.95pp | 0.026 | 0.974 |
| 20.0pp (baseline §5.5 permits) | 0.008 | 0.992 |

So I replaced an always-reject gate with an always-inconclusive one. The runbook now carries the
sizing warning: size from the measured σ_t; at σ_d=3.58pp, `h<3pp` needs n≈5–6.

**The 3–5pp dead zone.** With δ=3 and the +5pp gate held, a clean `d={3.5,4.0,4.5}` (mean +4.0,
90% CI [+3.16,+4.84]) fails condition 3, is not contained in ±3, spans neither 0 nor +5, and is not
underpowered — no row matched it, so it defaulted to **stop**: a genuine well-measured positive
killing the program. ~8% of triples at a true +4pp effect. Closed with a sixth inconclusive bullet.

**The 20pp headroom clause was vacuous too** — the 75% window ceiling already guarantees ≥25pp, so it
could never reject anything. Deleted from §5.5 and from §6.1's frozen-threshold list.

## Regression introduced by the fix pass (caught in review)

The F18/F19 edit replaced the vacuous `β₁+β₂ ≤ 1` with **another vacuous assertion**. Under the new
`b = 2σ(ℓ_b)`, measured over 200k draws: `b ∈ (2.1e-12, 1.9999999999990652)`, and
`β₁+β₂ = b` identically (max deviation 2.2e-16). Since `2σ(ℓ) < 2` strictly, **`β₁+β₂ ≤ 2` can never
fire either** — same disease, new number.

Only the **identity** check is non-vacuous: assert `β₁+β₂ == b` to dtype tolerance. That one catches a
real implementation bug (a factor computing its own budget, or π applied to the wrong factor). The
`≤ 2` bound should be dropped or explicitly labelled a documentation-only invariant.

Verified sigmoid saturation points (binary search, PyTorch): bf16 **6.234**, fp16 **8.316**,
fp32 **16.636** — the doc's 6.235 / 8.32 / 16.64 are correct to rounding.

## Statistics — the P1.4 gate is effectively always-reject

Job arithmetic is correct: 8×5×3=120 → 15 waves; 9×5×3=135 → 17 waves.

Condition 3 ("mean ≥ +5pp AND all three paired differences nonnegative"), power of the 3/3
requirement for a **true** +5pp effect:

| Between-seed SD | P(all 3 nonneg) |
|---:|---:|
| 5pp | 0.596 |
| 10pp | 0.331 |
| 15pp | 0.251 |
| 20pp (the max §5.5 tolerates) | **0.215** |

Under the null, 3/3 nonnegative occurs with p = 1/8 = **0.125**. So at the tolerated noise level the
gate fires on a real effect 21% of the time and on nothing 12.5% of the time.

Condition 4 (no guardrail loses >2pp in a 3-seed mean, across five tasks) is an unadjusted
conjunction. With DP2 **exactly neutral** on every guardrail:

| SD | P(one trips) | P(≥1 of 5 trips) |
|---:|---:|---:|
| 10pp | 0.365 | **0.896** |
| 20pp | 0.431 | **0.940** |

Joint probability that a genuinely good DP2 (+5pp real, neutral guardrails) clears both:

| SD | P(cond3) | P(cond4) | Joint |
|---:|---:|---:|---:|
| 10pp | 0.331 | 0.104 | **3.4%** |
| 20pp | 0.215 | 0.060 | **1.3%** |

Measurement precision is *not* the binding constraint: binomial SE at 1,000 scored targets (p=0.5)
is 1.58pp, so the 5pp gate is 3.2 measurement-SEs wide. The problem is between-seed variance with
n=3, not bank size.

## Cost envelope (never stated in any doc)

p5.48xlarge at $55.04/hr, 15 waves (120 jobs):

| Per-wave duration | 15 waves | 17 waves |
|---|---:|---:|
| 20 min | $272 | $309 |
| 1 hr | $826 | $936 |
| 2 hr | $1,651 | $1,871 |
| 3 hr | $2,477 | $2,807 |

Phase 0 g6e.xlarge at $1.861/hr: $45 (24h) to $134 (72h).

## Code state — spot checks

- `probes/` is under **no version control** (`git rev-parse` fails; only `.git` dirs in the
  workspace are `OLMo-core/.git` and `edullm-data/.git`). There is no parent repo.
- OLMo-core HEAD `f17824e2a9ae325e1eda1430273d98f55a1c9bee`; `Householder` appears **0** times in
  the HEAD version of `recurrent.py` and 21 times in the working tree.
- DP2 spans **7** dirty files, not the 5 that P0.0 lists. The two omissions carry the wiring:
  `attention/__init__.py` (+13, exports `KimiDeltaHouseholder`/`Config`) and
  `flash_linear_attn_api.py` (+85). Restoring only P0.0's list yields a tree where the class is
  not exported.
- All **9** required test IDs from §4.7 are absent (0 hits), against 61 existing test functions in
  the two gate files.
- The ignore-index fix §5.2 demands is **already applied** (`train_probe.py:70-74`).
- `mqar_d16` is byte-identical to `mqar_p16` (`tasks.py:325-329` vs `:340-344`).
- Beta is computed once for all R factors (`recurrent.py:1257-1259`), so the single
  `allow_neg_eigval` flag cannot express per-factor regimes; budgeted/tied-K need a new
  parameterization, not a flag flip.
- `GatedDeltaNet` defaults `allow_neg_eigval=True` (`recurrent.py:76,447`) and is consumed by 10
  production 7B scripts; the KDA classes default `False`. "Fix the hard-coded True" in
  `recurrent.py` would silently change those runs. The real fix is 3 lines in `train_probe.py`.

## Further statistics, independently re-verified

**Degenerate calibrations pass P1.1.** Mean 15–85% + sample SD ≤ 20pp admits triples where two of
three seeds never learned the task: `{0,10,35}` (mean 15.0, SD 18.0), `{4,4,37}` (mean 15.0,
SD 19.1), `{95,95,65}` (mean 85.0, SD 17.3). All three pass and may enter the primary composite.

**The SD screen is nearly inert at n=3.** P(observed s ≤ 20pp) = 0.982 / 0.632 / **0.359** for true
σ = 10 / 20 / 30pp. A task with true 30pp seed noise passes the screen a third of the time.

**Infrastructure alone breaks condition 2.** P(≥1 failure in 120 jobs) = 0.113 / 0.452 / **0.701** /
0.911 at per-job failure rates of 0.1% / 0.5% / 1% / 2%. At a modest 1%, the "all jobs valid"
condition fails 70% of the time, and no sanctioned retry path exists.

**Noise reduction is worth more than seeds.** `σ_d(comp) = σ_t·√(2(1−ρ))·√(1/3)`:
σ_t=20, ρ=0.5 → **11.55pp**; σ_t=8, ρ=0.7 → **3.58pp**. Enforcing identical data order and task
instances across arms (buying ρ) is free and worth ~5x in required n.

**Caveat on "just add seeds" — mine, not the subagent's.** The "all n differences nonnegative"
clause gets *harder* with more seeds: at σ_d=3.58 and a true +5pp effect, P(all nonneg) = 0.776
(n=3) → 0.655 (n=5) → 0.508 (n=8). Buying seeds while keeping condition 3 as written partly
self-defeats. The sign clause must be dropped or replaced, not supplemented.

**The clean replacement.** One-sided lower CI bound `L_95 > 0`, no sign clause, at σ_d=3.58:

| n | required mean | power at true +5pp |
|---:|---:|---:|
| 3 | 6.04pp | 0.308 |
| 5 | 3.41pp | **0.839** |
| 8 | 2.40pp | 0.980 |

n=5 is 200 jobs / 25 waves — roughly $500 more than the current plan for a gate that actually works.

**D1 sorts before T1.** Under §5.6's "lexical by task ID," `sorted([T1..T8, D1])[0] == 'D1'`. If MQAR
qualifies it runs first — ~2 waves of p5 time on the one task that feeds no decision rule. T1, the
primary discriminator, fully resolves by wave 2 of 15, with no stopping or blinding rule and
`aws-operations.md:84` mandating 30-minute progress reports.

## Dangling references (specifications that do not exist)

`the declared difficulty grid` (§5.5), `the documented minimum` filler span (§5.3), `the frozen
tolerance` for R1-P mismatch (§5.4), and the image base/Dockerfile/registry — each is referenced as
if defined, and none appears anywhere in the doc set. `p0-decision.md` (required by §4.8) has no
home in the §3.3 artifact tree.

The `#mandatory-pre-launch-checklist` anchor **does** resolve correctly.

## The silent-skip trap (highest-consequence operational defect)

`kda_householder_test.py:34-58` loads its correctness oracle from **outside** the repo
(`probes/naive_kda_householder.py`) and calls `pytest.skip(...)` when it cannot be found. A skipped
test suite exits 0. So on a node where the oracle is missing, the Phase-0 semantic gate **reports
green having verified nothing.**

The trap is that it is invisible where you would test for it. The fallback candidate is
`Path(__file__).parents[5]/"probes"`, which on this machine resolves to
`/Users/ericwu/Developer/Capstone_LLM/probes` and **does** exist — so it passes locally. It fires
only where `OLMo-core` is checked out without its sibling `probes/` directory, i.e. exactly the
Phase-0 GPU node.

Compounding it: §4.7 says `KDA_PROBES_DIR` "must come from the P0.0 environment manifest," but §4.2
(P0.0) never records env vars, and §3.3's `environment` field does not list them. P0.5 reads a field
P0.0 never writes.

Minimum fix: P0.5 must assert **zero skips** in the two gate files (e.g. `-p no:randomly -ra` plus a
check that the collected/passed counts match), and P0.0 step 4 must record `KDA_PROBES_DIR` and the
Python executable.

## Arm-name drift

§3.3 requires the manifest `arm` field be "Exact arm ID from Section 3.2." The canonical ID
`R1-2step-tiedK` appears **once** in the runbook; the informal `tied-K` appears **five** times (and
throughout `phase-2-deferred.md`). A validator implementing §3.3 literally would reject the name
used everywhere else. Similarly `Reflection` (runbook) vs `Reflection / EDA2` (README), where EDA2
is never defined.

## Wave-count undercount

§5.6's "15 waves" counts the P1.2 triage matrix only. Also on the same node: P1.0 smoke (1 wave) and
P1.1 calibration (3 bundles × new tasks × grid settings — uncomputable, since the grid is undefined;
at 4 tasks × 3 settings it is 36 jobs = 5 waves). Honest minimum ≈ **21 waves, not 15** (~40% more).
The cost equation in `aws-operations.md` needs a measured per-job `t`, but §5.4's smoke exit criteria
are all correctness — **timing is never recorded**, so the equation's only input is never produced.
