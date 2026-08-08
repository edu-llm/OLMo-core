# Pre-registration — mixer bake-off, RUN 2

**Written 2026-08-08, before run 2 dispatches.** [`seeds-run2.json`](seeds-run2.json) is the
machine-readable authority for every integer; this file carries the reasoning and the decision
rules. Both are frozen at submission.

**Run 1's [`PREREGISTRATION.md`](PREREGISTRATION.md) and [`seeds.json`](seeds.json) are the frozen
record of a completed experiment and are not edited by run 2.** This document carries forward what
worked, and fixes three statistical defects **by name** in §6.

> **What run 1 delivered.** 18/18 cells, no failures, ~$790. Clean separation on throughput and
> memory. **CE unresolved:** pooled σ̂ = 0.020415 at df 12 → MDE **0.0636 nats** against a
> 0.010–0.030 literature target. A 13-agent audit confirmed every core statistic in the *generated*
> analysis and found six wrong claims, **all in hand-written prose**. That asymmetry is itself a
> finding, and §10 acts on it.

---

## 1. What run 2 is for

Choose a linear-attention mixer for a forthcoming **large production training run**. Three things
must come out of it: a **CE** reading good enough to rule arms out, a **throughput** ranking, and a
**memory** ranking. Run 1 delivered the last two. Run 2 exists to make CE informative and to add
the two things run 1 measured nothing about: **long-range recall** and **decode**.

## 2. Arms — five, control first

| arm | role | mixer config | `ARM_L0_DELTA` |
|---|---|---|---|
| **`KDA_BASE`** | **shared control** (Dunnett reference) | `KimiDeltaAttentionConfig(allow_neg_eigval=False, conv_activation="silu", gated_conv=False)` | `K2_L0_DELTA` = −10,080 |
| `KDA_NOACT` | isolates the gate; the correct comparator for `KDA_GCONV` | `KimiDeltaAttentionConfig(gated_conv=False, conv_activation=None)` | −10,080 |
| `KDA_NEGEIG` | **new** — `allow_neg_eigval=True` on the **fast chunked** kernel | `KimiDeltaAttentionConfig(allow_neg_eigval=True)` | −10,080 |
| `KDA_GCONV` | gated short convolutions (LFM2/LIV-style) | `KimiDeltaAttentionConfig(gated_conv=True, gated_conv_activation=None, gate_structure="depthwise")` | **+2,208** |
| `GDN2` | Gated DeltaNet-2 competitor | `GatedDeltaNet2Config(expand_v=1.0, allow_neg_eigval=False)` | **+22,688** |

**Dropped:** `KDA_R1`, `KDA_R2` — the hand-written Householder kernel has no fast chunked path at
R > 1 and measured 5–21× slower locally. `KDA_R1` alone carried **81.6%** of run 1's pooled sum of
squares. `KDA_K3` (sketched in run 1's `seeds.json`) was never implemented and is not in run 2.

**`KDA_NEGEIG` is the scientific replacement for the Householder arms.** They were testing whether
β can exceed 1 so `(I − βkkᵀ)` becomes a true reflection rather than a contraction; `allow_neg_eigval`
buys that mechanism on the shipped kernel, at the shipped kernel's speed and with **zero** extra
parameters.

Two traps carried forward, both verified in-tree:

- With `gate_structure="depthwise"`, `gated_conv_activation=None` **does not** mean activation-free
  — the depthwise pre-gate is algebraically a SiLU with a learnable per-channel slope
  (`2·sigmoid(a·u)·u == (2/a)·silu(a·u)`, the `2/a` absorbed into the conv taps). So the contrast
  that isolates the **gate** is **`KDA_GCONV` − `KDA_NOACT`**, never `− KDA_BASE`.
- `allow_neg_eigval` defaults to **`False`**. On `KDA_NEGEIG` it is the entire mechanism, so if it
  fails to reach the kernel the arm trains stably at the same cost as the control while measuring
  nothing. P1's `kda_negeig_test.py` gates exactly this.

## 3. Design, and where the precision comes from

**25 cells = 5 arms × 5 replicates.** 3,721 steps × 524,288 tokens = **1,950,875,648 tokens/cell**,
TPP **5.0006** at 390,125,472 parameters. `gpu-8xa100`, **`attempts=1` pinned**, declared 2.2 h/cell
→ worst-case ceiling **$1,191.67**.

MDE falls from run 1's **0.0636** to **0.0229**, a 2.8× improvement, and it is worth being precise
about where it comes from, because only one of the three sources is a free lunch:

| source | effect on MDE |
|---|---|
| **Dropping `KDA_R1`/`KDA_R2`** — σ̂ 0.020415 (df 12) → **0.01024761** (df 8) | ÷1.99 |
| **n = 3 → 5** — SE scales √(2/n) | ÷1.29 |
| **k = 5 → 4** — Dunnett critical value 2.8905 → 2.6510 at fixed df | small |
| *(cost)* **df 12 → 8** on σ̂ itself | widens the bracket — see §8 |

**The σ̂ improvement is a re-estimate, not a variance reduction.** Nothing was done to the
remaining arms to make them quieter; two noisy arms left the pool. If the true σ of the surviving
arms is what run 1's 12 cells say, run 2 resolves 0.023 nats. **§8 states the bracket on that bet.**

### 3.1 A correction to the design brief's token count

The brief states 1,952,153,600 tokens/cell. **That is not reachable at this batch size:**
1,952,153,600 / 524,288 = 3,723.4375, not an integer, and it exceeds 3,721 steps by 1,277,952
tokens. **`steps` = 3,721 is the frozen quantity** — it is what the flag takes and what the pairing
requires — and tokens are derived from it: **1,950,875,648**. TPP is 5.0006 either way.

### 3.2 TPP 5 is better than run 1's, and still below the literature

Run 1 realised **TPP 1.54** (its plan said 2.6; the budget was cut). Run 2 sits at **5.0006**,
which is *inside* the bottom edge of the academic cluster (3–20) rather than below it, and still far
from this project's 1B flagship at TPP 27–44.

**Architecture effects measured at low token budgets systematically overstate.** Measured in-tree,
GDN's edge over baseline shrank **0.0103 nats @1B → 0.0059 @15B** — roughly halved for 15× the
budget. So run 2's CE magnitudes remain **upper bounds** on the production effect, and 3.25× run 1's
budget makes them *less* inflated, not uninflated. Throughput and memory are budget-independent and
fully valid. What a 1.95B-token run structurally **cannot** see — effects that emerge late in
training — is a **limitation, not a null**.

## 4. Endpoints

Exact keys in the rank-0 JSON that `summarise()` prints — the only channel the platform reads
results back through.

| endpoint | key | **can it move the decision?** |
|---|---|---|
| **Primary: held-out CE** | `val_ce` | **Yes** — rules arms out, or returns a bound |
| **Co-primary: throughput** | `throughput_tok_s_steady`, `..._per_device` | **Yes** |
| **Co-primary: memory** | `peak_memory_gib`, `peak_memory_reserved_gib` | **Yes** |
| Secondary: recall by gap band | `sliced_eval` | **No** — hypothesis-generating only |
| Secondary: decode | decode-probe fields | **No** by itself — see below |

**Which secondaries can move a decision, stated now.** The two co-primaries and CE decide the arm.
The **recall slices cannot**: they arrive on one budget, at n=5, with no pre-declared per-band
effect size and a multiplicity across bands that this design does not correct — so a band-level
difference is **hypothesis-generating**, full stop. **Decode is the one asymmetric case:** its
latency and recurrent-state bytes cannot *select* an arm, but a **disqualifying** decode cost — an
arm whose recurrent state does not fit the production serving budget — is a veto, because the
deliverable is a production mixer. A veto needs a threshold, and **no serving budget is declared
here**, so run 2 can only *report* decode. **Any decode-based exclusion is a post-hoc decision and
must be labelled one.**

**Denominators before differences.** `val_tokens`, `val_tokens_present`, `val_tokens_declared` and
`val_shards` are compared across cells before any CE is differenced. All 18 of run 1's cells agreed
exactly (974,917,632 / 975,077,376 / 975,077,376 / 39). A cell that differs has not evaluated the
same held-out set.

**Memory readings are gated on `peak_memory_source`.** Only `per_step_running_max` is a real peak.
`final_step_only` is **a lower bound wearing the name of a peak** — the GPU monitor resets peak
stats every step — and `unavailable` is a null. Either disqualifies that cell's memory reading and
the count is reported. `throughput_tok_s_whole_run` and `tps_naive_wall_clock` are **not**
endpoints: they charge fixed costs against the hardware and penalise bigger shapes hardest.

**`last_loss` is NOT an endpoint.** A decay-to-zero LR schedule ends at a mechanically lower train
loss at equal held-out quality and can **invert** the comparison. Run 2 changes the step count,
which changes the schedule's length, which changes `last_loss` for reasons unrelated to the mixer.

**The `first_loss` receipt is calibrated to 11.7124, not to ln(vocab) — and here is why.** Run 1
reported `first_loss` ≈ **11.7124** on every cell, against ln(100,352) = **11.5164**: a **+0.196**
nat offset. **That offset is expected, not a fault.** Random initial logits have nonzero variance,
so by Jensen the step-0 CE sits *above* uniform by ≈ s²/2 where s is the logit standard deviation;
+0.196 implies s ≈ 0.626, and d_model = 1024 with embedding σ ≈ 0.02 predicts s ≈ 0.64. The
agreement is close enough that the offset is fully accounted for. **This is recorded because
`exp(11.7124)` = 122,076 looks like a 122,880-vocabulary run, and reading it that way would be a
false alarm.** The in-tree band `STEP0_LOSS_BAND = (11.016, 12.016)` admits both and its own test
already states that the band does **not** catch a wrong vocabulary — the embedding row count does.
**Run 2's receipt: `first_loss` within the band, AND within ±0.02 of run 1's 11.7124 for the nine
cells whose init seed is unchanged** (that is a sharper check than the band, and it is only
available because of the seed reuse).

## 5. Pairing — what is new, and what it does and does not buy

**Data seeds are shared by all five arms.** Data order is a function of `(data_seed + epoch,
len(dataset))` **only** and does not depend on the model. This is unconditionally valid and run 1
already had it.

**Init seeds are now paired across three arms — the first time this study can do that.** Run 1's
`seeds.json` refused to pair init seeds and was **right**: arms with different tensor inventories
diverge at the first differing tensor, so a shared integer buys nothing. Dropping the Householder
arms changes that for three arms. `KDA_BASE`, `KDA_NOACT` and `KDA_NEGEIG` are the **same config
class** differing only in constructor arguments, all three sit at `ARM_L0_DELTA == K2_L0_DELTA` so
the width solver gives them the same shape, **removing an activation removes no tensor**, and
`allow_neg_eigval` is a plain bool that allocates nothing. One init seed gives all three
**literally identical starting weights**: same weights, same data, same order, only the operator
differs.

**`KDA_GCONV` and `GDN2` are declared exceptions.** `KDA_GCONV` adds gate tensors (+2,208); `GDN2`
has `w_w` that KDA has not (+22,688) and a larger mixer that `solve_widths` respends into different
FFN widths. They share the **data** seeds and have their **own** init seeds.

> **The consequence, stated rather than buried.** `KDA_GCONV` − `KDA_BASE` and `GDN2` − `KDA_BASE`
> **carry an init-variance component that the three paired contrasts do not.** At equal n they are
> genuinely noisier, even though the pooled procedure assigns every contrast the same SE. **Do not
> read a larger `KDA_GCONV`/`GDN2` p-value as weaker mechanism evidence at face value.** The
> gate-isolating contrast `KDA_GCONV` − `KDA_NOACT` is also not init-paired.

**What the pairing buys is interpretational, and the primary analysis does not claim variance
reduction from it.** Measured on run 1's four surviving arms, the replicate/data-seed **block effect
is F = 1.13 on df (2,6)** — not significant — and blocking moves σ̂ only 0.010248 → 0.010082 (a
**1.6%** reduction) while costing 2 df. **So the primary analysis is an unblocked one-way ANOVA.**
Blocking on replicate is a **pre-declared secondary sensitivity** (§7), not the headline.

What pairing *does* buy: a `KDA_NEGEIG` − `KDA_BASE` difference within a replicate is attributable
to `allow_neg_eigval` **alone**, with no init component in it at all.

**Sharing init makes the trio's cells positively correlated within a replicate, so the unpaired
Dunnett SE overstates the variance of a trio contrast.** Those contrasts are therefore
**conservative** (true type I error ≤ α); it is exact for the two exception arms. Stated, not
exploited.

### 5.1 The step-count trap, and why cross-run CE is void

Run 1's frozen plan said **1,907** steps; the cells ran **1,144**. The cut was legitimate and
documented — and it is precisely the failure this precondition names, landing on a real experiment.

**Run 2's 3,721 must be identical across all 25 cells or the pairing is void.** A cell on a shorter
budget consumes a **prefix** of the stream, not the same stream. 3,721 is a frozen quantity, not a
cost dial: if the budget must move it moves for all 25 cells and `seeds-run2.json` is amended before
submission — **never trimmed per cell to fit an approval band.**

**Cross-run comparisons to run 1 are only valid where the step count matches, and it does not.**
Run 2 trains 3.25× longer. Said plainly:

- **VALID:** `first_loss` (a step-0 quantity, independent of the budget) for the **9** cells whose
  init seed is also unchanged; throughput and peak memory (budget-independent) for the **12** cells
  with a run-1 counterpart arm; structural fields everywhere.
- **INVALID: `val_ce`, for any cell, for any purpose.** Every run-2 CE is lower for a reason that
  has nothing to do with the mixer. **No run1-vs-run2 CE difference may be reported as an arm
  effect, as drift, or as a replication.** The drift check is real but narrower than run 1's
  `seeds.json` anticipated: it covers the image, kernels, platform speed and initialisation — not CE.

Every contrast that moves a decision is therefore **internal to run 2**, against run 2's own
`KDA_BASE`.

## 6. Analysis plan, pre-committed — including run 1's three defects fixed by name

0. **Admissibility is step 0 and runs BEFORE pooling.** A cell that diverged (non-finite loss),
   OOM'd, failed to converge, or **ran a different step count** is excluded **before any variance is
   pooled**, and the exclusion is declared **with a count**. Never exclude-then-cite. Gates are
   absolute-magnitude, not existence checks: `val_ce` in a plausible band, `first_loss` per §4,
   `steps == 3721`, denominator tuple identical, `peak_memory_source == per_step_running_max`.
1. **Never pool σ over cells where the endpoint cannot move.** A saturated cell contributes ≈0
   variance and would deflate σ̂. **Fail open** — if it is unclear whether a cell can move, keep it
   in — and report the drop count (expected: 0; CE has no ceiling in this design).
2. **Pooled-variance one-way ANOVA across the five arms**, `df = 5 × (5 − 1) = 20`. Not independent
   pairwise t-tests, which throw away most of the df.
3. **Dunnett against `KDA_BASE`, k = 4, two-sided α = 0.05, df = 20 → critical value 2.6510.**
   Computed by Gauss–Legendre quadrature over the ρ = ½ equicorrelated max-|t| integral. ρ = ½ is
   *exact* here (balanced, shared control). The implementation is validated three ways: it reduces
   to Student's t at k = 1 (matching textbook values to 5 d.p. at df 2–20), it reproduces published
   Dunnett tables (k=4/df=20 → 2.651 vs 2.65; k=3/df=12 → 2.683 vs 2.68; k=2/df=10 → 2.568 vs
   2.57), and it reproduces run 1's own 2.901255 at k=5/df=12. Arm-vs-arm comparisons **not**
   against the control — including the gate-isolating `KDA_GCONV` − `KDA_NOACT` — are
   **exploratory** and declared as such.

### Defect (a) — the MDE must NOT be used as an inference threshold

Run 1's code required **`CI excludes zero` AND `|estimate| > MDE`**. An MDE is a **design**
quantity, not a decision threshold, and the conjunction is not conservative bookkeeping — it
**silently changes the test's size**.

Because the MDE is `ncp₈₀ × SE` computed with the *same* pooled σ̂ that is in the SE, `|estimate| >
MDE` is exactly `|t| > ncp₈₀`. The conjunction is therefore `|t| > max(crit, ncp₈₀) = ncp₈₀`:

| geometry | Dunnett crit | ncp₈₀ | effective crit | declared α | **real familywise size** |
|---|---:|---:|---:|---:|---:|
| Run 1 (k=5, df=12) | 2.9013 | 3.8154 | **3.8154** | 0.05 | **0.0099** (5.0× conservative) |
| Run 2 (k=4, df=20) | 2.6510 | 3.5293 | **3.5293** | 0.05 | **0.0074** (6.8× conservative) |

**Run 2 commits to the Dunnett CI alone.** An arm is resolved iff its Dunnett-adjusted CI excludes
zero. The MDE appears **only** in §8, as a design quantity describing what the run was powered to
see. It is not applied to any estimate.

### Defect (b) — Levene is THE decision test, and the fallback fires ONLY if Levene rejects

Run 1's generated analysis got this right (`decision_test: "Levene (median-centred)"`,
`fallback_engaged: false`) — and the **hand-written write-up invoked the fallback anyway** and
reported a `GDN2` significance that **no procedure reproduces**. The arithmetic:

- Levene (median-centred), the pre-registered decision test: **p = 0.484** → does **not** reject →
  **the fallback must not fire.**
- Bartlett: p = 0.0059 → rejects. Bartlett assumes normality and is notoriously sensitive to
  departures from it at n = 3. **It is reported, and it does not decide.**
- On the correct pooled path, `GDN2` − `KDA_BASE` = +0.037055, SE = 0.016669, **t = 2.223 vs crit
  2.9013 → NOT significant.**
- The Welch path that was never triggered gives t = 5.588 on df 2.56, which *looks* significant
  against an **unadjusted** critical value. That is the number that reached the prose.

**The rule, unambiguously:** compute **Levene (median-centred)** and **Bartlett**; report both;
**Levene alone decides**. The fallback (Dunnett's T3 — the studentized-maximum-modulus,
unequal-variance analogue for k-vs-one-control; *not* Games-Howell, which is the all-pairs
procedure) is engaged **if and only if Levene rejects at α = 0.05**, and the results JSON records
`fallback_engaged` with the Levene p-value beside it. **At n = 5 both tests still have low power, so
failing to reject is weak evidence of homogeneity, not a demonstration of it** — which is an
argument for reporting §7's per-arm σ̂, not for switching procedures after the fact.

### Defect (c) — σ̂ is reported pooled AND leave-one-arm-out, pre-declared

Run 1 had **one arm inflate σ̂ by 2.17×** (dropping `KDA_R1` moved pooled σ̂ 0.020415 → 0.009605).
That was discovered *after* the fact. Run 2 pre-declares both, so nobody chooses after seeing the
answer:

- **Headline:** pooled σ̂ over all five arms, df 20, with its χ² interval.
- **Alongside, always:** the five leave-one-arm-out σ̂ values (df 16 each; the corresponding
  Dunnett critical value at k=4, df=16 is **2.7079**) **and the ratio of each to the pooled value.**
- **Interpretation rule, fixed now:** the leave-one-out table is a **sensitivity display, not a
  selection menu.** The headline inference uses the pooled σ̂ regardless. If any single arm moves
  pooled σ̂ by **more than 1.5×**, that is **reported as a finding about that arm** — and every
  contrast is additionally reported under the leave-that-arm-out σ̂ so the reader can see whether
  any conclusion depends on it. **No conclusion is restated as the headline on the basis of a
  leave-one-out σ̂.**

4. **Reporting:** effect, Dunnett-adjusted CI and n for every contrast. Never a bare p-value, and
   never "n.s." as though it meant "no effect".

## 7. Pre-declared sensitivities (all computed, none selected after the fact)

1. **σ̂ pooled and leave-one-arm-out**, per defect (c).
2. **Blocked (two-way) analysis** treating replicate as a block, `df = 16`, reported beside the
   unblocked headline. Pre-declared **because** run 1's data say the block effect is small
   (F = 1.13); if run 2's block effect is materially larger, both numbers are already on the page.
3. **Trio-only re-analysis** — `KDA_BASE`/`KDA_NOACT`/`KDA_NEGEIG` at k = 2, the three arms with a
   clean init-paired contrast, where a paired-differences test is legitimate.
4. **CE against the two co-primaries** — the recommendation must state throughput and memory ratios
   beside any CE claim, because an arm that ties on CE and costs 2× the step time is not a tie.

## 8. Power — exact non-central t, and the bet is stated as a bet

**Estimator: exact non-central t, dominant-tail survival only** (`nct.sf(crit, df, ncp)`). The
naive two-tail form suffers catastrophic cancellation. **The normal approximation is not used
anywhere** — it was **2.2× too optimistic on required-n at n = 3** in a prior project incident
(n = 3 → 5 buys 0.0190, not 0.0044). Implementation validated against
`moe/audit/findings/power.md`, reproducing its 0.03917 (n=3) and 0.02018 (n=5) at σ = 0.0120 to
5 s.f., and against run 1's own generated analysis (σ̂ 0.0204151, crit 2.901255, MDE 0.0636).

**σ̂ = 0.01024761**, the leave-out-`KDA_R1`/`KDA_R2` pooled value from run 1's four surviving arms
(SS = 8.40109e-4, **df = 8**). 80% power, two-sided α = 0.05, Dunnett k = 4, unpaired difference of
two arm means, SE = σ√(2/n):

| n | df | SE(diff) | Dunnett crit | **MDE (nats)** | CI half-width |
|---:|---:|---:|---:|---:|---:|
| 3 | 10 | 0.008367 | 2.8905 | **0.0319** | 0.0242 |
| 4 | 15 | 0.007246 | 2.7273 | **0.0262** | 0.0198 |
| **5** | **20** | **0.006481** | **2.6510** | **0.0229** | **0.0172** |
| 6 | 25 | 0.005916 | 2.6069 | **0.0206** | 0.0154 |

*(The brief's ≈0.0236 corresponds to σ ≈ 0.01057; the measured 0.01024761 gives 0.0229. Both round
to "about 0.023".)*

### 8.1 df = 8 makes that σ̂ fragile — the MDE is a bet, not a promise

**σ̂ rests on 8 degrees of freedom.** Its χ² interval at df 8 is `σ̂ × [0.6755, 1.9158]`:

| | value |
|---|---|
| σ̂ | **0.01024761** (df 8) |
| χ²(0.025, 8), χ²(0.975, 8) | 2.1797, 17.5345 |
| **95% interval on σ** | **[0.006922, 0.019632]** — a factor-**2.84** bracket |
| **⇒ MDE at n = 5** | **[0.0155, 0.0438]** |

**So the honest statement is: run 2 is powered to detect ~0.023 nats if σ̂ is right, and anywhere
from 0.016 to 0.044 across the interval that 8 df cannot rule out.** The upper end is worse than
the 0.010–0.030 literature target. This is a **bet on σ̂**, and the bet is placed knowingly because
the alternative — sizing off run 1's all-six-arm σ̂ = 0.0204 — would require n ≈ 24 (120 cells) and
is not fundable. Note also that run 1's own σ̂ came from **TPP 1.54**; run 2 runs at **TPP 5.0**, and
σ at a different budget is not guaranteed to be the same σ.

**Run 2's own pooled σ̂ at df 20** (χ² bracket `× [0.7651, 1.4441]`, a factor-1.89 spread) is a
**pre-registered deliverable regardless of what the CE contrasts do**, and it is what any run 3 must
be sized from.

### 8.2 CE may again not resolve — pre-committed, with the fallback named

**The literature CE gap between these mixer families is ≈0.010–0.030 nats. The bottom half of that
window sits below run 2's MDE of 0.0229, and the whole window sits below the upper end of §8.1's
bracket. CE may again fail to resolve. That is pre-committed as an expected outcome, not something
to be discovered during analysis.** At this σ̂, MDE ≤ 0.010 needs **n ≈ 24** (120 cells); ≤ 0.020
needs n = 7; ≤ 0.030 needs n = 4.

**The fallback, named in advance** — identical in structure to run 1's, which is what made run 1 a
planned outcome rather than a scramble:

1. If **no** arm's Dunnett CI excludes zero, **CE is declared unresolved at n = 5** and the arms are
   **not ranked on CE**. The recommendation falls to **throughput and memory**, and *that the
   fallback was taken is stated in the recommendation itself*, not buried.
2. **The choice is then: the fastest/leanest arm not shown to be worse on CE.** Run 1 chose
   `KDA_BASE` on exactly this rule.
3. **A null is a BOUND, not equivalence.** A non-significant contrast licenses exactly one claim:
   *the difference is smaller than the Dunnett-adjusted CI half-width* (0.0172 nats at n=5, σ̂ as
   measured). Equivalence needs a pre-declared margin and a TOST. **No margin is declared here, so
   no equivalence claim may be made from this design.** A flat CE table must not be read as "no
   difference".
4. `KDA_NEGEIG` is the one arm where a null is *interesting on its own terms*: it is
   parameter-free, init-paired to the control, and costs nothing if it does not help — so "no
   detectable CE change, no throughput cost" is a usable production finding, reported as the bound
   it is.

## 9. Preconditions — what run 2 depends on, and what changes if each fails

| precondition | owner | if it fails |
|---|---|---|
| **Tensor-inventory test**: `KDA_BASE`/`KDA_NOACT`/`KDA_NEGEIG` have identical parameter **names and shapes** | **P1** | **The paired trio collapses.** Fall back to run 1's rule — every arm its own init seeds — and **delete every pairing claim.** The 25-cell structure, step count, data seeds and MDE are **unaffected**: the primary analysis is unblocked one-way and the MDE gain comes from dropping the Householder arms and k 5→4, not from pairing. |
| `KDA_NEGEIG` kernel correct at β > 1, and the flag reaches the kernel | **P1** | Arm comes out; run 2 becomes 4 arms / 20 cells at **k = 3, df = 16 → crit 2.5923, MDE 0.0225**. Re-derive before submitting. |
| Slice masks (`slice_manifest.json` + `*.mask.u8`) | **P2**, **P3** | `sliced_eval` stays null; the recall endpoint is absent. It is secondary and moves no gate — **run 2 still submits.** |
| Decode probe | **P3** | On by default; `--no-decode-probe` is deliberately not passed. |

> **A COUNT-ONLY CHECK IS NOT ENOUGH, and the counterexample is already in the tree.** `KDA_R1`
> sits at `ARM_L0_DELTA == K2_L0_DELTA` — the *same declared delta as `KDA_BASE`* — while being a
> different config class with `w_b` of a different width. Equal parameter **count** therefore
> demonstrably does **not** imply equal **inventory**, and a count-only test would pass an arm whose
> RNG stream diverges, producing a pairing claim that is false. **The test P1 owns must compare
> parameter names and shapes.** At freeze, `core6_arms.py:155` declares the count and
> `test_every_arm_lands_on_its_declared_delta` covers it; the name-and-shape assertion was not yet
> present.

## 10. Analysis will be generated, not hand-written

Run 1's audit found **six wrong claims, all in hand-written prose**, while every statistic in the
generated JSON held up. The response is structural: **every number in run 2's report is emitted by
the analysis script from the cell JSONs**, and the prose cites those fields rather than restating
them. The `GDN2` incident in defect (b) is the archetype — a hand-written sentence asserted a
significance the machine-readable output correctly denied.

## 11. Deviations from this pre-registration

*(To be completed at analysis time. Run 1's honesty here was one of its strengths — this section
exists to be filled in, not to stay empty.)*

- Nothing yet; run 2 has not dispatched.

**Recorded now, as deviations of run 2 from run 1's plan:**

1. **Run 2 is not the study run 1's `seeds.json` sketched.** That file described run 2 as a 2-arm
   `KDA_BASE`-vs-`KDA_K3` study at n=3. The lead re-scoped it to 5 arms × 5 seeds and `KDA_K3` was
   never implemented. Run 1's file is left unedited as the frozen record it is.
2. **`KDA_NOACT`'s init seeds change between runs** (113008/123015/133022 → the trio's
   110007/120014/130021). Pairing *within* run 2 is worth more than reproducing run 1's isolated
   arm. Consequence: run 2's `KDA_NOACT` is data-paired to run 1 but **not** init-paired, so it is
   one of the three counterpart arms **excluded** from the 9-cell `first_loss` drift receipt.
3. **The unequal-variance fallback is Dunnett's T3, not Games-Howell** — carried forward from run
   1's own recorded deviation, which was correct: T3 is the k-vs-one-control analogue; Games-Howell
   is all-pairs.
4. **The MDE-as-threshold conjunction is removed** (defect (a)). This is a deliberate, declared
   departure from run 1's implemented procedure, and it makes run 2's test *less* conservative than
   run 1's actual behaviour while matching what run 1 *declared*.
