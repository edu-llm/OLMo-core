# Hyper-connections at 370M

Branch `edullm/hyper-connections-370m`. Pre-registration, arm table, and the numbers that have
to be measured before anything expensive is submitted.

Written before the first arm so that the hypotheses are timestamped by the commit that
contains them.

## The question

Hyper-connections have been measured twice at essentially the same parameter scale with
opposite signs.

- **ByteDance** (Hyper-Connections, [arXiv 2409.19606](https://arxiv.org/abs/2409.19606), ICLR
  2025) ran their ablation on **OLMo-1B, on Dolma** — the same architecture family and data
  family as this lab. At an expansion rate of 4 over 500B tokens they report −0.030 V2 eval
  loss and +1.3 points of downstream average, and no loss spikes in any hyper-connection run.
- **Tencent** (Most Transformer Modifications Still Do Not Transfer at 1–3B,
  [arXiv 2605.20798](https://arxiv.org/abs/2605.20798)) measure −0.020 downstream at **1.2B
  dense**, z = −9.79, which survives Bonferroni as a significant *degradation*, and diverges at
  3B across all three seeds.

The module is not "reproduce hyper-connections" — three public reproductions already exist.
It is: **explain the inversion**. There are five documented differences between the two setups,
and each one is an arm.

1. **Output-side-only versus full input-side mixing.** Tencent state that their
   reimplementation omitted the input-side lane mixing because their shared residual interface
   does not expose the sublayer input. OLMo-core's `ResidualStream.forward(residual, x)` has the
   identical limitation, so anyone implementing this by swapping that module alone reproduces
   their crippled variant exactly. → arm 3.

   **What the arm removes is the *learned* input map and nothing else.** The read stays on
   eq. 14's staggered one-hot. This is the difference between an arm that is crippled and an
   arm that is degenerate, and it is not a matter of taste: `B` is all-ones, `A_r` is the
   identity and every lane starts as the same copy, so `A_m`'s stagger is the only object in
   the construction that is not symmetric under permuting the lane index. An earlier version of
   this branch read the lanes with a uniform mean instead, which makes every remaining object
   permutation-equivariant, puts the model in the permutation-symmetric subspace and keeps the
   gradients in that subspace's tangent space — where an elementwise optimizer holds them
   forever. Measured over 60 AdamW steps: lane dispersion 8.2e-05 against `full`'s 2.0e-01,
   with `A_r`'s diagonal spread and `B`'s spread both exactly zero, and a 1e-6 asymmetry seeded
   by hand still at 1.3e-04 after 200 steps. That arm would have run to completion and reported
   a clean null meaning nothing. With the one-hot read it lands at 1.5e-01 to 3.6e-01 over
   three seeds, against the faithful arm's 1.7e-01 to 3.2e-01 under the same treatment — the
   same order, which is what makes arm 3 differ from arm 2 in the one respect it claims to.
2. **The √n output-initialization scaling.** ByteDance scale the second linear of the
   feed-forward network and the attention output projector so the pre-unembedding standard
   deviation is unchanged. Tencent's paper does not mention doing it. → arm 4.
3. **The weight-decay split.** ByteDance exclude the static component from weight decay and
   keep it on the dynamic one. Also unmentioned in the replication. → arm 5.
4. **Token budget.** 500B against 23B. This branch is at 10B, so if the effect needs a long
   horizon it should fail here, and that is itself the answer.
5. **Parameter reuse.** Hyper-connections raise the recurrence-equivalence exponent from 0.46
   to 0.65 in a *looped* model at small scale
   ([arXiv 2604.21106](https://arxiv.org/abs/2604.21106)) and help at MoE scale, while hurting
   in a plain dense stack. Lane value may track how much the same parameters get reused rather
   than how big the model is. → arms 10 and 11.

## Configuration

370M OLMo-3: `d_model` 1024, 16 layers, 16 heads, head_dim 64, SwiGLU, RMSNorm, reordered norm,
QK-norm, RoPE, z-loss, dolma2 at vocab 100,278 (padded to 100,352), untied embeddings, at
sequence length 4096 over a 786,432-token batch.

**The first tranche runs 6,000 steps and 4.72B tokens, not the 10B and 12,715 steps the rest of
this section is written against.** The full horizon is 37.7 hours at the measured step time and
the workload's per-attempt ceiling is 24, and the second attempt that would cover the
difference is not one the platform's retry rules reliably grant — the whole argument is in
[What a full arm actually costs](#what-a-full-arm-actually-costs-and-why-it-cannot-be-submitted-as-one-run).
All nine runs share the shorter horizon, so no contrast inside the tranche is confounded by it.
The figures below are for the full horizon and are what a second tranche needs.

10B dolma2 tokens is 12,715 steps of that batch. **3.0e19 FLOPs per arm**, from the run's own
accounting rather than from a rule of thumb: the
370M probe billed 238.74 petaflops over 78,643,200 tokens, which is 3.036e9 FLOPs per token,
and 12,715 steps of 786,432 tokens comes to 3.04e19. The 2.2e19 that stood here was 6ND read
off the name plate, and it is low on both factors — the model is 474,220,352 parameters once
the two embedding tables are counted, not 370M, and `num_flops_per_token` counts attention and
the lane mixing besides.

## Arms

Generated from `hyper_connection_arms.ARMS`; the tests assert these properties rather than
trusting the table.

> **Correction of 2026-08-10.** The seeds column said **3, 3, 3, 0** and every other arm 0, and
> had said so since the design moved to five seeds. The sentence above was false of that column
> in particular: the parameter and FLOP columns are parsed and asserted, and the seeds column —
> the number this document's own history records as having been wrong twice before, as "fifteen"
> and then "seventeen" — was not read by any test, so 186 tests passed with it wrong. It is
> corrected here to the live values and
> `test_the_arm_table_in_the_pre_registration_states_the_seeds_that_are_funded` now parses it.

| # | arm | seeds | params | vs baseline | FLOPs/token vs baseline |
| --- | --- | --- | --- | --- | --- |
| 1 | `baseline` | **5** | 474,022,912 | +0.0000% | +0.0000% |
| 2 | `faithful` | **5** | 474,220,352 | +0.0417% | +0.0994% |
| 3 | `output-only` | **5** | 474,187,456 | +0.0347% | +0.0908% |
| 4 | `no-output-init` | **5** | 474,220,352 | +0.0417% | +0.0994% |
| 5 | `decay-everything` | 0 | 474,220,352 | +0.0417% | +0.0994% |
| 6 | `n1` | 0 | 474,121,376 | +0.0208% | +0.0119% |
| 7 | `n2` | 0 | 474,154,304 | +0.0277% | +0.0324% |
| 8 | `n8` | 0 | 474,353,216 | +0.0697% | +0.3371% |
| 9 | `mhc` | **5** | 474,220,352 | +0.0417% | +0.0994% |
| 10 | `tied-faithful` | 0 | 339,871,136 | −28.3007% | +0.0994% |
| 11 | `tied-baseline` | 0 | 339,772,416 | −28.3215% | +0.0000% |

Every untied arm is iso-parameter to within 0.07% and iso-FLOP to within 0.34%. The tied arms
are deliberately not iso-parameter — that is what they test — but they are iso-FLOP with their
own control, because they are matched on *effective* depth: 16 layers running 8 distinct
blocks twice on a cycle.

**Nine runs in the first tranche**, and the number is `hyper_connection_arms.total_runs()` with
a test on it rather than a sentence, because the "fifteen" and then the "seventeen" that stood
here were each wrong for as long as they were written and nothing in the repository could say
so. Three arms at three seeds and nothing at one or two: see
[Where the seeds went](#where-the-seeds-went).

**Arm 3 is only worth running because of `b7983ea9`.** Until that commit `output-only` dropped
the paper's fixed staggered one-hot read along with the learned input map, which left every
lane reading the same vector — the baseline with dead parameters, not a crippled
hyper-connection. Three seeds of that would have measured nothing and given no sign that it had
measured nothing. The fix restores the staggered read so that the learned input map is the only
difference from arm 2, which is exactly the difference H2a is about. The arm's `summary` field
carries the commit hash so this cannot be forgotten again.

**Cut order** is now all eight unfunded arms, and read backwards it is the order a second
tranche restores them in: `n8`, `n2`, `tied-faithful`, `tied-baseline`, `decay-everything`,
`n1`, `no-output-init`, **`mhc` last**. A test asserts the unfunded set is exactly the head of
that list, so the budget cannot be balanced by quietly cutting something the plan never
nominated.

A zero in the seeds column is not the same as an absent row. Every one of those eight still
builds, still stays iso-parameter and iso-FLOP, and still passes every property test the funded
three pass; funding one later costs a number in one file and no design work at all.

## Pre-registered hypotheses

Every hypothesis is a directional claim about held-out cross-entropy in nats on the seven
validation sources, reported per source and as their mean, with bits-per-byte beside it as the
same quantity divided by a constant. "Beats" means the paired difference defined in
[The analysis plan](#the-analysis-plan) clears the gate stated there.

- **H1 (replication).** Arm 2 beats arm 1 by ≥0.025 nats. **Confounded with an initialization
  change, and it has to be written up that way.** See below.
- **H2a (the artifact, in-loop).** Arm 2 > arm 3 on held-out cross-entropy.
- **H2b (the artifact, downstream).** Arm 2 > arm 3 on the downstream average, and arm 3 but
  not arm 2 reproduces the published degradation. **Blocked**, and see below: this is the one
  that carries the headline, and nothing in this plan produces the number it needs.
- **H3 (initialization).** Arm 2 > arm 4; the output-init scaling is load-bearing rather than
  cosmetic.
- **H4 (the seesaw), restated as a superiority test.** Arm 4 > arm 6, **not arm 2 > arm 6**.
  ByteDance found n=1 does not help; if one lane buys as much as four, their mechanism story is
  incomplete at this scale. The difference between arm 6 and arm 1 is reported with its
  interval as a *bound*, with no equivalence claim attached to it — see below for why the
  design cannot support one.

  Read against arm 4 because arm 6 differs from arm 2 in **two** ways, not one.
  `output_init_scale` returns 1.0 whenever `n_lanes == 1`, so arm 6 silently loses the
  output-init rescale as well as three lanes. That is principled — with one lane there is no
  sum to compensate, and any other answer would be arbitrary — but it means an arm 2 versus
  arm 6 gap mixes the expansion rate with the initialization prescription, which is the
  contrast H3 already owns. Arm 4 is `faithful` with `output_init_exponent=0.0`, so arm 4
  versus arm 6 moves the lane count alone. The cost is that the contrast is 2 seeds against 1
  rather than 3 against 1, which widens its standard error; the alternative is a clean-looking
  number that answers a different question.
- **H5 (constraint).** Arm 9 ≥ arm 2, with the gap larger wherever arm 2 is unstable.
- **H6 (reuse).** Arm 10 − arm 11 > arm 2 − arm 1. Lane value tracks parameter reuse, so the
  effect is larger when the same parameters run twice. Reported as an estimate with an
  interval and no gate applied — at one seed on each tied arm the contrast cannot resolve
  anything the literature predicts, and pretending otherwise is worse than saying so.
- **H7 (stability), secondary, and added after stage 1 rather than before it.** Arm 2 declines
  more optimizer updates than arm 1, and at larger triggering gradient norms. It exists because
  every arm now trains under `SkipStepAdamW`, which moves an instability out of the loss and
  would otherwise take it out of the measurement entirely. It is listed here so that the
  hypothesis list is complete, and the fact that it is post-hoc is part of the entry: the
  statistic, the test and what it cannot resolve are all fixed in [The amendment of
  2026-08-08](#the-amendment-of-2026-08-08-spike-skipping-on-every-arm). No primary conclusion
  is conditioned on it.

**H2b, and why it is marked blocked rather than quietly dropped.** The result H2 sets out to
explain is Tencent's −0.020 on a *downstream average* at 1.2B, and arm 3 produces in-loop
bits-per-byte. Those are not the same measurement and this document already says why they
cannot stand in for one another: loss and downstream decouple by 6 to 16 points for changes in
this class, which is the stated reason downstream is reported at all. So an arm 2 versus arm 3
gap in held-out cross-entropy is evidence that the two implementations differ — H2a, which is
worth having and is testable here — but it is not evidence about the published negative
result, and writing it up as though it were would be exactly the like-for-like failure this
module exists to expose in somebody else's work.

**H1 is confounded, and arm 4 is what disentangles it.** The faithful arm does not start where
the baseline starts. Eq. 14 makes hyper-connections compute exactly what the residual stack
computes at initialization, and the test suite asserts it — but only with the output-init
rescale off. Turn the rescale on at the paper's `output_init_exponent=0.5` and a same-seed
baseline and a same-seed arm 2 differ in their logits by 1.0e+00 in the max, a relative 8.4e-01,
against 7.2e-07 at exponent 0.0. So arm 2 is two changes away from arm 1, not one: the mechanism
and a smaller initialization for every attention output projector and second feed-forward
linear. An H1 gap is therefore attributable to **hyper-connections plus their initialization
prescription**, and the write-up says exactly that unless arm 4 comes back flat against arm 1 —
in which case the rescale is doing nothing on its own and H1 can be read as the mechanism.
This is not a reason to change arm 2, which is the published method and has to stay it. It is a
reason arm 4 is load-bearing for H1 as well as for H3.

Downstream is deliberately not produced in-loop, for the reasons under
[Where the runs land](#where-the-runs-land-and-what-they-log), so H2b needs the separate
checkpoint-scoring job that checkpoint-as-input is for. That job is not in this plan and not
in this budget. **The pre-registered consequence is that H2b is unclaimable until it exists**,
and that the arms save the checkpoints it would read — which they already do. Scheduling it is
the single cheapest thing that would raise what this module can conclude.

## The analysis plan

Fixed here, before any treatment arm runs, because every choice below is free now and is a
degree of freedom afterwards. All the thresholds are stated at a planning σ of 0.010 nats,
which is the middle of the 0.008–0.012 literature estimate; every one of them scales linearly
with σ̂ and gets recomputed from the measurement the moment the noise floor table below has
numbers in it. **No treatment arm is submitted until it does.**

### The gate: two standard errors of the contrast under test

"Nothing under 2σ gets claimed" is three different rules, and at σ = 0.010 they are three
different numbers.

| reading | threshold | what it is |
| --- | --- | --- |
| twice the per-seed σ | 0.020 | the spread of single runs, applied to a difference of means |
| twice the SE of the contrast | 0.016 for 3 vs 3 | the spread of the thing actually being tested |
| twice the SE of the widest contrast | 0.033 for H6 | one threshold, set by the worst case |

**The rule is the second: a claim requires |Δ̂| ≥ 2 × SÊ(Δ̂), where SÊ is built from the
pooled σ̂ and the seed counts of the two arms in that contrast.** So the number is 0.016
for a three-versus-three difference, 0.018 for two versus three, 0.023 for one versus three
and 0.033 for the H6 interaction, and it is not one number at all.

The justification is that the other two readings hold the threshold fixed while the standard
error moves under it, so their false-positive rate is different for every hypothesis. A flat
0.020 is a 1.4% test of a three-versus-three difference, an 8.3% test of one versus three and
a 22% test of H6 — the same words, applied to the six hypotheses, meaning six different things
and admitting the weakest evidence exactly where the design is weakest. Scaling the threshold
with the contrast's own SE holds the rate constant across all six, which is the only property
that makes a single sentence in a pre-registration mean one thing.

Two honest qualifications on the constant. Two standard errors is a 4.6% two-sided test only
when σ is known; with σ̂ estimated it is a t statistic, and the realised per-comparison
false-positive rate is **9.2% at the pooled df = 6 and 18.4% at the df = 2 the baseline alone
would give**. The exact two-sided t p-value is therefore reported beside every contrast, and
so is the 5% line, which sits at 2.45 SE rather than 2.00. And no multiplicity correction is
applied to the gate: the six hypotheses are fixed in advance and each is reported with its
effect size, its interval and its p-value, so a reader who wants Holm across the family can
apply it to a table that has not been selected on.

### σ is pooled across all nine three-seed runs

Arms 1, 2 and 9 each carry three seeds. σ̂ is the pooled within-arm standard deviation over all
three, with **df = 6**, not the baseline's own with df = 2.

This is not a refinement. A variance estimate on df = 2 has a 95% confidence interval running
from 0.52 to 6.28 times the point estimate — **a factor of 12.1 end to end** — so a gate built
on it is a gate whose height nobody knows. Pooling takes the interval to 0.64–2.20, a factor
of 3.4, and takes the gate's false-positive rate from 18.4% to 9.2%. The power gain is real
too and it is large where the design is thinnest: for a one-versus-three contrast against a
0.030-nat effect, a proper 5% test goes from **31.6% power at df = 2 to 58.5% at df = 6**.
Three runs' worth of information that has already been paid for, retrieved by writing one
sentence down before the fact instead of after it.

It rests on **homoscedasticity**: that arms 1, 2 and 9 have the same run-to-run variance. That
is an assumption and not a fact, and it is not obviously safe — arm 2 is the arm this document
predicts may be unstable, and H5's whole content is that arm 9 is the stable one, so the two
arms being pooled are the two the plan expects to differ in exactly this quantity.

It is tested from the same nine runs, by **Bartlett's test on the three within-arm variances at
α = 0.05**, reported whatever it says. Levene's test is not used: with three observations per
group the median-centred residuals are degenerate and it rejects essentially never, at any true
ratio. Bartlett does hold its size — 4.5% under the null in simulation — but its power at this
size is 11% against a doubling of one arm's σ, 23% against a tripling and 38% against a
quadrupling, so **a pass is not evidence of equal variances and will not be written up as
one**. The eyeball check is no better: under true homoscedasticity the largest of three
n = 3 standard deviations exceeds the smallest by a median factor of 2.5 and by more than 9.4
in five cases out of a hundred, so a ratio of three or four means nothing at all.

The pre-committed consequence: if Bartlett rejects, the pooled σ̂ is abandoned and every
contrast is re-run with unpooled Welch standard errors and Welch–Satterthwaite degrees of
freedom, which costs power and is reported as costing it. The three per-arm standard
deviations are printed in the noise-floor table either way, so a reader can see the thing the
test is too weak to adjudicate.

### Paired by seed is the primary analysis, and it is free

`build_config` in `.edullm/train_hyper_connections.py` sets `opts.data_seed = opts.data_seed +
opts.seed` before it builds the data loader, and it does so identically on every arm. Arm *a*
seed *k* and arm *b* seed *k* therefore stream the corpus in the same order, from the same
shuffle, and see the same documents in the same batches. `--seed` defaults to 0, so the
single-seed arms all run seed 0 and pair with seed 0 of the three-seed arms.

**The primary analysis is therefore the paired difference**: form Δ_k = (arm *a*, seed *k*) −
(arm *b*, seed *k*) for each shared seed, and test the mean of the Δ_k. The unpaired
difference of arm means is reported as a secondary, so that the two can be compared and the
size of the shuffle component is visible rather than assumed.

Be precise about what the pairing removes. It removes the data-order component and nothing
else. The same line also adds `opts.seed` to `config.init_seed` and to `config.model.init_seed`,
but two arms with different parameter shapes draw from that generator in different orders, so
the initial weights do not match across arms even when the seed does; what is left in the
paired difference is initialization plus kernel non-determinism. This is why the variance
reduction has to be measured rather than claimed.

At a within-pair correlation ρ the paired difference has standard deviation σ√(2(1−ρ)), so the
mean of n paired differences has SE = σ√(2(1−ρ)/n). The minimum detectable effect at 80% power,
two-sided α = 0.05, σ = 0.010 nats, by exact noncentral t, for the **three arms and five seeds
that are actually running**:

| ρ | 3 pairs (df 4) | 4 pairs (df 6) | 5 pairs (df 8) |
| --- | --- | --- | --- |
| 0.0 | 0.031 | 0.024 | 0.020 |
| 0.3 | 0.026 | 0.020 | 0.017 |
| 0.5 | 0.022 | 0.017 | 0.014 |
| 0.7 | 0.017 | 0.013 | 0.011 |

The operative column is the last one. The unpaired five-versus-five comparison is **0.019**.
The table is generated by `noise_floor.render_mde_table`, and `test_noise_floor.py` parses it
back out of this file and asserts it cell by cell, so the document and the estimator cannot
drift apart the way the two versions below did.

ρ̂ is estimated from the H1 and H2a quintuples, which are now the only two contrasts there are,
and is reported before either of them is interpreted.

#### The df convention changed on 2026-08-08, and every number in that table went up

**The table this replaces used pooled df = 6 for a paired analysis of 3 arms across 3 shared
seeds, and that is wrong.** A paired analysis of k arms across n shared seeds is a randomized
complete block design. The total df is kn − 1; the arm effect takes k − 1, **the seed effect
takes n − 1**, and error gets what is left, which is (k − 1)(n − 1). At three arms and three
seeds that is **4, not 6**. The 6 is k(n − 1), which is the *unpaired* count — correct for a
design with no block term, and wrong the moment a seed effect is removed. Removing the seed
effect is not free, and treating it as free was the error.

What it cost, at the 3 × 3 design the old table was written for:

| ρ | as printed, df = 6 | correct, df = 4 | optimistic by |
| --- | --- | --- | --- |
| 0.0 | 0.028 | 0.031 | 11.7% |
| 0.3 | 0.023 | 0.026 | 11.7% |
| 0.5 | 0.019 | 0.022 | 11.7% |
| 0.7 | 0.015 | 0.017 | 11.7% |

Uniformly 11.7%, because the df enters only through two t quantiles and the effect is otherwise
linear in σ. The old numbers are left standing in that column rather than deleted: the point is
not that the table now holds different numbers but that it held wrong ones for as long as it
stood, and a reader who saw the earlier version should be able to tell which one they read. At
the 3 × 5 design the same mistake would be worth 4.9% — the penalty shrinks as the block eats a
smaller share of a larger df — so moving to five seeds would have hidden most of it without
fixing it.

The old table's 1-pair and 2-pair columns are gone rather than corrected. At n = 1 the block
design has (k − 1)(n − 1) = 0 error df and there is no test at all, which the old table
concealed by borrowing the pooled df; at n = 2 there are two. Both columns existed to price
single-seed arms, and the tranche no longer funds any.

**Two consequences that are not just a number moving.**

*Pairing is not free, and the sentence that said it was has gone.* The old text read "pairing
at any ρ above about 0.3 buys more than a second seed does, for nothing." The block costs
n − 1 degrees of freedom, and at ρ = 0 that is a pure loss: paired at five seeds is **0.020**
against unpaired's **0.019**. Break-even is **ρ ≈ 0.09**, above which pairing wins and below
which it does not. That is a low bar and the pairing will very likely clear it — but it is a
bar, it is measured rather than assumed, and the pre-registered rule is that the paired
analysis stays primary only if ρ̂ clears it. If ρ̂ lands under 0.09 the unpaired contrast is
reported as primary and the pairing as secondary, which is the reverse of the order stated
above, and it is written here so that the order is not chosen after ρ̂ is seen.

*The df = 8 is only available because σ is pooled across the three arms.* A standalone paired
t test on one contrast estimates σ_Δ from its own five differences and carries df = 4, which at
ρ = 0.5 is an MDE of **0.017** against the block design's **0.014**. So the residual pooling is
doing real work, and it rests on the same homoscedasticity assumption the pooled σ̂ does — which
Bartlett's test is too weak to adjudicate, and which is reported either way.

**A third correction of the same shape, found while recomputing this table.** σ̂ is a sample
standard deviation, and a sample standard deviation is unbiased for the *variance* and not for
itself: E[s] = c₄(df)·σ, and c₄(4) = 0.9400. So five seeds understate σ by 6% on average, every
MDE is linear in σ, and an MDE quoted from the raw s is 6% smaller than the design really has.
`noise_floor.mde_from` divides by c₄ before it prices anything, and `SigmaEstimate` carries the
raw value beside `sigma_unbiased` so both stay visible. The correction belongs to the point
estimate only: the t machinery is already built on the distribution of s, and applying c₄ to a
t test as well would count it twice.

Neither error was caught by this plan's own reasoning, and both were caught by writing the
estimator down as code with a test against a planted truth. That is the argument for having
written it before the treatment arms rather than after them.

### What each hypothesis can actually detect

At σ = 0.010, pooled df = 6, two-sided α = 0.05, 80% power, unpaired. Literature effects for
comparison are ByteDance's −0.030 and Tencent's −0.020.

| | contrast | seeds | SE | MDE | reads |
| --- | --- | --- | --- | --- | --- |
| H1 | 2 − 1 | 3 v 3 | 0.0082 | 0.028 | marginal against −0.030 |
| H2a | 2 − 3 | 3 v 2 | 0.0091 | 0.031 | marginal |
| H3 | 2 − 4 | 3 v 2 | 0.0091 | 0.031 | marginal |
| H4 | 4 − 6 | 2 v 1 | 0.0122 | 0.041 | under-powered |
| H5 | 9 − 2 | 3 v 3 | 0.0082 | 0.028 | marginal against −0.030 |
| H6 | (10 − 11) − (2 − 1) | 1,1 v 3,3 | 0.0163 | 0.055 | cannot resolve anything predicted |

Pairing moves every row of that table left by the factor in the previous section, which is the
difference between "marginal" and "adequate" for H1, H2a, H3 and H5 if ρ̂ comes in at 0.5. It
does not save H6, and nothing in any affordable budget does: three seeds on all four tied arms
would still leave the interaction at an MDE of 0.039 against a predicted effect of 0.030.
**H6 is pre-registered as descriptive.** It is reported as a point estimate with an interval,
labelled as under-powered, and no claim of any sign is attached to it.

### Why H4 is not an equivalence test

H4 as originally written — "arm 6 ≈ arm 1 or worse" — asks for a null, and failing to detect a
difference is not evidence that there is none. The honest instrument for a null is TOST against
a stated margin, and the arithmetic says this design cannot supply one worth stating.

With arm 6 at one seed and arm 1 at three, the smallest equivalence margin TOST could reject at
with 80% power is **0.039 nats**. That is larger than the 0.025 H1 claims for the full method,
so the test would certify arm 6 as "equivalent to the baseline" in the very case where one lane
had bought the entire hyper-connection effect. A margin of 0.025 would need the standard error
down to about 0.0074, which takes **three seeds on arm 6 and a measured ρ of at least 0.3** —
two more runs than the budget has, contingent on a quantity that has not been measured.

So H4 is restated above as the superiority test the design can run, arm 4 versus arm 6, which
is the seesaw claim in the form that has content: the question is whether the fourth lane is
doing anything the first one is not. The arm 6 minus arm 1 difference is still reported, with
its 95% interval, explicitly as a bound — "the data are consistent with anything between X and
Y" — and no sentence anywhere in the write-up will say arm 6 is equivalent to the baseline.

### Where the seeds went

Eight of the eleven arms began with one seed each, and H2, H3, H4 and H6 rested on them. At
one seed those four hypotheses have minimum detectable effects of 0.039 to 0.055 nats against
literature effects of 0.020 to 0.030 — which is to say the design was written so that its four
mechanism hypotheses could not detect the mechanism. Worse, on the baseline-only σ̂ with df = 2
those same MDEs are 0.065 to 0.092, because the threshold inherits the uncertainty in σ̂.

Three things could be done about it and only the third is dishonest.

**Reallocate.** `n2` and `n8` are the first two entries in the cut order, and the
expansion-rate curve is the only claim in the plan that nothing else in the plan depends on:
H1, H2, H3, H4 and H5 all read arm 2 at n = 4, and none of them cares what n = 2 does. Their
two runs move to second seeds on arms 3 and 4, which carry H2a and H3, taking both from one
versus three to two versus three: MDE 0.039 → 0.031 unpaired, and 0.034 → 0.024 paired at
ρ = 0.5. **Seventeen runs before and seventeen after**, so it needs no new approval and costs
no compute.

**Keep everything and claim nothing from it.** Report arms 3 through 8, 10 and 11 as point
estimates with intervals and no hypothesis attached, and accept that the module answers H1 and
H5 only. Cheap, defensible, and it gives up the thing the module was built to do.

**Keep everything and claim it anyway**, which is what the plan said before this revision, and
is a design that spends 3.0e19 FLOPs per arm to produce numbers whose gate it cannot clear.

**Taken: reallocate.** The arm table above carries it. Arm 6 stays at one seed deliberately,
because H4 is now the arm 4 versus arm 6 superiority test and the bound on arm 6 versus arm 1 —
neither of which a second seed rescues, since the equivalence claim it would have to support
needs three seeds and a measured ρ. The review this revision came from suggested second seeds
on arms 3, 4 *and* 6, which is eighteen runs rather than seventeen; the third one is the one
that buys the least, so it is the one left out.

### The first tranche is nine runs, and the reallocation above is superseded

Everything above this heading was written against seventeen runs. Seventeen is no longer the
budget: at $4,000 and the measured cost of an arm, nine is. That is not a proportional trim of
the seventeen — a design that answered six hypotheses at two seeds each now answers two at
three, which is a different experiment and is written down as one.

**What nine buys.**

| | contrast | seeds | what it settles |
| --- | --- | --- | --- |
| **H1** | arm 2 vs arm 1 | 3 v 3 | replication. Does DHC ×4 beat the baseline at 370M at all. |
| **H2a** | arm 3 vs arm 2 | 3 v 3 | the implementation-artifact question. Whether the field's negative result is an artifact of a reimplementation that kept the output mixing and dropped the input map. |

Three against three is the smallest design in which σ is estimated from the data instead of
assumed, and it is what [the gate](#the-gate-two-standard-errors-of-the-contrast-under-test)
was written against. Both contrasts share the same three baseline runs, so σ̂ is pooled across
all nine and neither hypothesis is paying for its own noise floor. Every arm runs the same
horizon, so neither contrast has a horizon confound.

**What nine gives up, and why in this order.** Nothing at one seed and nothing at two. An arm
at one seed cannot separate its effect from the seed it drew; an arm at two estimates σ from a
single difference. Six partial answers at 0.031–0.055 nats MDE against literature effects of
0.020–0.030 is the same failure the section above diagnosed, arrived at by spending the money
more thinly rather than by spending it wrong. So H3, H4 and Cause 5 leave together.

**`mhc` and H5 are deferred to a second tranche and are not abandoned.** This is the one cut
worth arguing with. H5 is the best-designed hypothesis in the module: mHC's claim is
mechanistic rather than empirical — the lane-mixing matrix normalized towards the Birkhoff
polytope has a spectral radius the monitor already measures, and the composite condition number
across depth is instrumented too, so a null there is a *readable* null rather than a shrug. It
ships in DeepSeek V4, so both directions are publishable. It goes because it is the only
three-seed arm whose question does not presuppose H1, and because H2a is worth more when H1 is
undecided: if the method does nothing at 370M, knowing whether a constraint would have rescued
it is a smaller question than knowing whether the field measured the method at all. It is last
in `CUT_ORDER`, which makes it the first three runs a second tranche buys, and a test asserts
that position so the deferral cannot silently become a deletion.

**The horizon moved too, and that is a separate cut with a separate reason.** Each of the nine
runs 6,000 steps and 4.72B tokens rather than 12,715 and 10B. It is not a budget decision —
see [What a full arm actually costs](#what-a-full-arm-actually-costs-and-why-it-cannot-be-submitted-as-one-run),
where the constraint is the platform's per-attempt runtime ceiling and the retry rules behind
it. Inside the tranche nothing is confounded, because all nine share it; what is deferred is
the comparison to ByteDance's 500B-token and Tencent's 1.2B-scale results, which now needs the
second tranche as well.

**What is still reported alongside.** Downstream, wherever it exists, because loss and
downstream decouple by 6 to 16 points for changes in this class and a loss-only readout can
miss a catastrophe. Per-source cross-entropy, never only the pooled mean. And the lane
telemetry, because an arm whose lanes never differentiated is not evidence about
hyper-connections in either direction.

**The risk, stated up front.** *Review Residuals*
([arXiv 2606.31859](https://arxiv.org/abs/2606.31859)) found residual-topology changes invisible
below roughly 500M, with every difference inside noise through 320M. Against that, MHAR
([arXiv 2607.27230](https://arxiv.org/abs/2607.27230)) measures −0.149 nats at 350M for a
depth-routing change. So the effect is intervention-dependent, and if arms 2 through 9 are all
flat this is a clean scale-boundary result rather than a failure — but that is said here, in
advance, rather than afterwards.

## Track B: MHAR, scoped but not yet wired into a model

`MHARRoutingSite` and `MHARConfig` are built and tested in
`src/olmo_core/nn/residual_stream.py`. What is *not* built is the model-level threading, and
that is deliberate: it changes the block contract from `Tensor -> Tensor` to
`List[Tensor] -> List[Tensor]`, which is shared code that Track A's runs pass through. It waits
until the baseline seeds are launched.

Four things the scoping turned up that change how this has to be run.

**"Zero added parameters" is only true against single-head AttnRes.** The `H` queries are a
reshape of the same `d_model` numbers, so `H=8` is iso-parameter, iso-FLOP *and* iso-wall-clock
with `H=1` — an unusually clean control, and the paper's real contribution. Against a plain
transformer it adds `(2L+1) × 2 × d_model`, which is 67,584 parameters here, 0.014%. The test
suite reproduces the paper's own +100K at 350M and +187K at 1B from that formula, which is what
confirms the site count is right. Write it up as parameter-matched, not zero-parameter.

**MHAR needs its own baseline.** It is specified and validated under pre-norm: the routed
mixture goes through the sublayer's pre-norm before the sublayer. OLMo-2 and OLMo-3 use
reordered norm, where the sublayer receives its input unnormalized — and the routed mixture is
a convex combination of *raw sublayer outputs*, whose scale is nothing like a residual sum's.
So Track B cannot share Track A's baseline, and that is one more three-seed run in the budget.

The sentence that stood here pointed at `HyperConnectionStream` reading the lanes with a mean
as the same trap. It no longer reads them with a mean, and the reasoning was wrong where it
stood: the scale concern was real but a uniform mean was a permutation-symmetric answer to it,
which cost arm 3 the only asymmetry it had. With a one-hot read there is no sum to rescale.

**There is no free kernel.** `flash-linear-attention` does ship AttnRes at
`fla.ops.attnres.fused_attnres`, but it is strictly single-head — the online softmax state is
scalar and there is no head axis anywhere in the signature. It also first appears in fla 0.5.1
and this repo pins 0.4.1.

**Never stack the sources.** At `d_model` 1024, 16 layers, sequence 4096, batch 8, the 33
sources are 2.1 GiB held once each and 35 GiB if each site materializes its own stacked copy
for autograd. That quadratic is what makes the authors' own reference implementation run out of
memory above 1B, and `torch.compile` does not remove it. `MHARRoutingSite.forward` takes a list
and scores it in two passes for exactly this reason.

The arms, once it is wired: pre-norm baseline, single-head AttnRes, MHAR `H=8`, then `H=16` and
a random-init-query control conditional on `H=8` clearing the noise floor. Not `H=4` versus
`H=8` — the paper's own 1B numbers put those 0.003 nats apart, which this scale cannot resolve.
The interesting arm is single-head: the paper's thesis is not "MHAR is better" but "single-head
AttnRes is not robust and the free reshape fixes it", and that claim is corpus-dependent.
Dolma2 is a third corpus, neither of theirs.

## Preflight, and why it exists

```bash
AWS_PROFILE=sbsandbox python .edullm/train_hyper_connections.py pf --preflight --arm faithful \
  --dataset-id pretrain/regmix-10b --dataset-version v1 \
  --dataset-tokenizer tokenizer/dolma2-bpe --save-folder /tmp/x --work-dir /tmp/cache
```

Builds the config *and* the held-out dataset, prints what they came out as, exits without
training. Needs corpus credentials, no GPU, runs in about four seconds.

Three submissions died on things this catches: an attention backend the image does not carry,
a dataset class the evaluator refuses, and a missing metadata label. Each cost a queue wait and
a container to find, and each was a config error visible before a single token moved. The
evaluator in particular validates by *building* its dataset and then checking the result, so
its refusals can only ever arrive inside a running container — unless something builds it
first, which is what this does. Run it before every submission.

## The rehearsal passed, on the fourth attempt

`run_019fdfe9-e6c0`, 200 steps, `faithful` at the rehearsal size on `gpu-4xl40s`. Every metric
family the decision rule rests on is present, and the guard cleared.

**The lanes differentiate, and they start identical.** The relative spread across lanes reads
0.00064 at the first logged step, which is step 10, and the median over the eight blocks and
twenty logged steps is 0.0456 — **9.1 times the 5e-3 floor now in force**, and 45.6 times the
1e-3 floor that was in force when the run was submitted. At the last step the median is 0.0292
and the quietest block reads 0.0227, still 5.8 and 4.5 times the current floor. Starting near
zero is the initialization equivalence of eq. 14 confirmed in a real run rather than in a unit
test: at step zero every lane holds the same vector, exactly as the ordinary residual stack
would, and they separate once the mixing matrix moves. The mechanism is live, not inert.

**The spectral radius is already above 1.** ρ(A_r) on block 0's attention stream reads 1.001 at
the first logged step — the identity, as initialized — and climbs to 1.196 by step 200. Parcae's
signature for a diverging run is ρ ≥ 1, and Tencent's 3B divergence had a multi-lane drift.
Two hundred steps of a 96M model predicts nothing about a 370M run, and this is exactly the
quantity the instrumentation exists to watch. It is also the sharp prediction for arm 9: mHC
pins ρ at exactly 1 by construction, so if unconstrained HC drifts and mHC does not, H5 has a
mechanism rather than a correlation.

**Bits-per-byte arrives per source**, seven of them, at step 200: arxiv 1.513, open-web-math
1.601, algebraic-stack 1.615, starcoder 1.825, dclm 1.927, pes2o 1.928, wiki 2.059. Early and
therefore high, but already spread by a wide enough margin that a pooled average over them
would be the wrong statistic.

The row that stood here before was labelled step 200 and was the step-100 evaluation — arxiv
1.66, algebraic-stack 1.75, open-web-math 1.74, starcoder 1.98, dclm 2.03, pes2o 2.06, wiki
2.17, which is exactly what W&B holds at step 100. The correction is not only a shift in level.
At step 100 dclm and pes2o sit 0.027 apart and by step 200 they are 0.002 apart and have
swapped, so the cross-source ordering that looked settled is not, and open-web-math has moved
ahead of algebraic-stack. Half the visible spread between the middle sources at step 100 was
the model still moving, which is the argument for reading these per source at the end of a real
arm and not off a rehearsal at all.

### What analysing it changed

Three findings from the run's own telemetry, two of them bugs in this branch.

**The monitor was reading the evaluator's forward pass.** The hook fires on every forward, and
the held-out evaluation runs in `post_step` — so on eval steps the lane norms came from padded
held-out sequences rather than the training batch, reading 11% to 50% low. Worst at step 200,
which is the value that lands in the run summary: block 02's spread reads 0.0237 there against
a true 0.0470 ten steps earlier. Gating on `module.training` would not have worked, because the
evaluator's own `self.trainer.model.eval()` line is commented out. It now gates on being
between `pre_step` and `post_train_batch` instead.

**The fail-closed floor was unreachable, and the guard was loosened afterwards — recorded as
an instrument change, with its date.** The pre-registration was written on 2026-08-07 at 21:17.
The floor moved from 1e-3 to 5e-3 on 2026-08-08 at 01:31, and on the same day at 09:59 the rule
changed from a minimum across blocks to a majority of blocks and the quantity it reads changed
from lane-norm spread to lane dispersion. **Both changes were calibrated on data from the runs
the guard polices**, and both came after the hypotheses were timestamped. That makes them
instrument fixes rather than original calibration, and they are labelled as such here so that
nobody later reads the current numbers as having been set in advance.

They are defensible on the merits. Exactly one reading in the whole rehearsal ever fell under
1e-3 — 6.4e-4 at step 10, while the lanes were still separating — and from step 20 on every
block sat between 2e-2 and 7e-2, so the old floor could only have caught total failure. And the
minimum-across-blocks rule let the single least informative block speak for the whole model.
But the honest test of an instrument fix is whether it changes
a verdict, and this one does: **at 370M the minimum lane spread sits below the 5e-3 floor at
four of the five measured steps** — 0.0041 at step 20, 0.0023 at 60, 0.0018 at 80 and 0.0032 at
100, clearing it only at step 40 with 0.0069. A minimum rule at the current floor would abort
at four steps out of five and pass at the fifth. **The 370M probe passes only because of the
majority rule.**

What makes that defensible rather than convenient is which blocks they are and how many. It is
blocks 01 and 02 and never more than two of the sixteen at any step, while the median block sits
3.3 to 7.0 times over the floor; the readings at those two blocks are inside their own
step-to-step noise and do not keep a stable ordering between measurements. A shallow dead zone
at 370M is a finding about depth, and the guard exists to catch an inert mechanism across the
model, not to adjudicate two blocks against a number the instrument cannot resolve. It is still
a threshold set after seeing the data it is applied to, and the arms are the first runs it will
police from the start.

One caveat on the figures just quoted. The guard now reads lane *dispersion*, and the two
runs logged only lane *spread*, which is never larger. So 0.0018 to 0.0069 is a lower bound on
what the current guard would have seen, and the first arm is the first run that will report the
quantity the rule is actually written against.

**z-loss was off, though the configuration calls for it.** `train_on_corpus` never sets
`z_loss_multiplier`, so `train/Z loss` was never written. That matters more here than usual:
RMSNorm readouts are scale-invariant, so cross-entropy cannot see hidden-state scale at all,
and the rehearsal's hidden norms rose 50% and then gave back a third with nothing in the loss
curve reflecting either move. Now on at 1e-5.

Three more worth recording without a code change.

**Lane differentiation peaks mid-run and then retreats** in the rehearsal: the mean spread over
the eight blocks rises to 0.0596 at step 80 of 200 and falls back to 0.0279 by the end. So a
short run's endpoint understates it, and thresholds should not be ported from this rehearsal to
a differently-shaped schedule. The r = 0.72 against the learning rate that stood here is not in
either run. **In the rehearsal the correlation is +0.27** over twenty logged steps, whose 95%
interval is −0.19 to +0.64 and contains zero; the probe reaches +0.80, but over five points,
whose interval runs from −0.27 to +0.99 and contains almost everything. A figure near 0.72 is
only reachable in the probe, so if it came from anywhere it came from the run it was not
attributed to — and neither run establishes it. It is recorded as an observation with the
interval attached and is not used for anything. The probe also does not show the mid-run peak
at all: its lane spread is highest at the first logged step and declines monotonically, which
is one more reason not to port a shape from 96M to 370M.

**Block 0's attention spectral radius is the only one of eight in the rehearsal that never
turns over**, not one of sixteen — the rehearsal is an eight-block model, and the sixteen was
the 370M layer count written into a sentence about the 96M run. Every other block peaks between
steps 90 and 160 and decays; block 0 climbs monotonically to 1.196.

**At 370M the picture inverts, and that is the finding.** Thirteen of the sixteen blocks are
still climbing monotonically at step 100, and only blocks 01, 02 and 15 have turned over. Block
0 is still the steepest, at 1.140 against a median block's 1.03, and it still sits at the input
end where its amplification compounds through everything above it — so it remains the one to
watch. But "one block drifting while the rest settle" is a rehearsal phenomenon and the 370M
run is not doing it. A hundred steps is far too early to say whether the other twelve turn over
later, which is precisely why this is written down before the arms rather than recalled after
them.

**The composite spectral radius is moving fast and nothing gates on it.** `hc/composite
spectral radius` reads 1.299 at step 20 of the probe and 5.019 at step 100, having gained a
factor of 2.7 in the first twenty steps after that and still rising at the end; the composite
condition number tracks it almost exactly, at 5.060. The rehearsal's ran to 2.23 over twice as
many steps and had already turned over by step 130, so this is not the same trajectory at twice
the depth — it is a different one.

**No threshold is set on it, and that is deliberate.** A gate needs a number that separates a
healthy run from a sick one, and there are two of those from two model sizes, both healthy, one
still rising when its run ended. Parcae's ρ ≥ 1 signature is about the per-layer mixing matrix,
which is instrumented and which reads 1.01 to 1.14 here, not about the product across sixteen
of them — a composite of 5 is what sixteen radii averaging 1.05 multiply out to, and a stack of
identity matrices would give exactly 1. So the honest position is that this quantity has no
calibration, that setting one from the two runs that exist would repeat the mistake the section
above documents, and that **the first three baseline seeds are what turns it into a measurement
with a spread**. Until then it is logged, plotted and read by eye, and a run whose composite
leaves the range those seeds establish is a reason to look rather than a reason to stop.

Still outstanding: `bytes_per_token` is a single constant across all seven sources, so the
per-source BPB is a rigid rescaling of CE and carries no cross-source information. Arm-to-arm
comparison *within* a source is unaffected, which is what the decision rule actually rests on,
so this is a reporting defect rather than a scientific one — but the cross-source ordering
above should not be read as bits-per-byte until the per-source constants are measured.

## What the first three rehearsals found, for about $4 each

They died, which is what they were for.

**`flash_2` does not exist on this platform.** `RuntimeError: 'FlashAttention2Backend' is
missing the flash-attn package or is not supported on this platform.` The training image
installs `torch==2.9.0` and never installs flash-attn, so this is a property of the image and
not of the card — L40S is Ada and flash-2 supports it. `TransformerConfig.olmo3_370M` asks for
flash-2 by default, so **every arm would have died at startup**, 21 hours and $220 at a time.
Both factories now pin the torch backend, and `hc_370M` exists so that the flash-2 default
cannot come back the next time somebody copies a command.

The sliding window goes with it. OLMo-3's pattern is `[4096, 4096, 4096, -1]` and these runs
are at sequence length 4096, so a window of 4096 covers every position's entire history and
the windowed layers are exactly full causal attention — provably the same model, not an
approximation. Keeping it would not change a logit, but it would make the torch backend build
an explicit mask, and SDPA with an explicit mask gives up the fused causal kernel it would
otherwise use. Free to drop, not free to keep.

**The LM evaluator refuses a plain FSL dataset**, and it refuses it after building rather than
at configuration time. The held-out set has to be a `NumpyPaddedFSLDatasetConfig`, which is
also the right shape: one padded instance per document scores each document whole, where the
training dataset's contiguous blocks would cut documents across instance boundaries and score
the fragments.

**Every held-out shard needs a `label` in its metadata.** The evaluator names each metric after
it and raises on a shard without one. The label is the shard's source directory, which is the
same value the manifest carries as `labels.source`.

## Where the runs land, and what they log

| | |
| --- | --- |
| Entity | `eduLLM` |
| Project | `pre-training` |
| Group | the `--experiment` slug, e.g. `hyper-connections-370m` |
| Run name | the platform run id |

Per-topic project rather than per-team, which is the convention across the lab's other work
and puts these arms next to the other 370M runs.

`python .edullm/wandb_panels.py --verify --group <slug>` checks a group's runs against the
metric families each clause of the decision rule rests on and names what is missing; it exits
non-zero when a required family is absent, so it can gate a submission rather than only
inform one. `--report` builds the panels over the families that exist.

**The corpus does declare a validation split**, contrary to the comment in `train_on_corpus.py`
that says it does not. `regmix-10b-v1` publishes seven `val-00000` shards — one each for
algebraic-stack, arxiv, dclm, open-web-math, pes2o, starcoder and wiki — totalling 15,007,207
tokens. The reader resolves every declared split whatever you ask it for, so they come back on
`.val` even though `resolve_corpus` asks only for trainable shards.

Using them beats carving on three counts. No training tokens are lost, so the budget stays at
the full 9,989,799,834. The split is the publisher's rather than an arbitrary slice of ours.
And it arrives stratified by source, so the run reports bits-per-byte per source rather than
one pooled number — an average over arxiv, code, web text and Wikipedia together is exactly
the kind that hides the effect it is meant to measure. Carving survives only as a fallback for
a corpus that declares nothing, and it is worse in a specific way: shard paths sort by source,
so taking the last two would draw the whole evaluation set from one source category.

Without any held-out set the only loss in the run is training loss, and since `--seed` moves
the shuffle, its variance across seeds is partly a different sample of the corpus rather than
the run-to-run noise σ is supposed to measure.

Bits-per-byte is reported beside every cross-entropy metric, as CE in nats over
`bytes_per_token × ln 2`. That constant sets the absolute level only — it is identical across
arms, so a BPB difference between two arms is a CE difference times a fixed factor whatever
the constant turns out to be.

Downstream is **not** produced in-loop, and that is deliberate: the downstream evaluator
fetches from the public internet, which does not belong inside a run whose claim is that it
read a sealed corpus, and whose failure would look like a training failure. It comes from a
separate job over saved checkpoints, which is what checkpoint-as-input is for.

## Measurements still to be taken

Nothing below has been measured. These are the gates, not predictions.

### Noise floor

| quantity | stage that fills it | value |
| --- | --- | --- |
| σ̂ on held-out BPB at the final step, df = 4 | 1 | **0.00627 BPB = 0.0199 nats; 0.0211 c₄-corrected** |
| σ̂ at each of the twelve intermediate checkpoints | 1 | **measured; still falling at step 6,000** |
| per-source σ̂ over the seven held-out sources, df = 4 each | 1 | **0.0047–0.0091 BPB, a 1.93× spread** |
| the per-source inverse-variance weights, and what they buy | 1 | **0.1890 ×4 and 0.0813 ×3; 1.18× of variance** |
| pooled σ̂ across all fifteen runs, df = 12 | 2 | **not measured** |
| Bartlett p over the three within-arm variances | 2 | **not measured** |
| ρ̂, the within-seed correlation the pairing exploits | 2 | **not measured** |
| σ̂_Δ, the paired difference, from the H1 and H2a quintuples | 2 | **not measured** |
| per-seed σ, downstream average | neither | **not measured** |
| minimum detectable effect, per contrast, 80% power | 1 for σ̂, 2 for ρ̂ | **0.040 nats unpaired 5 v 5; the paired column still needs ρ̂** |

Every row is computed by `.edullm/noise_floor.py`, which is committed with a test against a
planted truth for each estimator. `--dry-run` reads whatever has landed and labels the reading
provisional; `--freeze` writes the numbers to JSON and **refuses anything provisional**, which
is what makes "frozen before stage 2" a property of the tool rather than a promise.

```bash
python .edullm/noise_floor.py --self-test                       # no network
python .edullm/noise_floor.py --dry-run --group hyper-connections-370m
python .edullm/noise_floor.py --group hyper-connections-370m --freeze .edullm/noise-floor.json
```

The estimate this plan was written against is σ ≈ 0.008–0.012 nats, and every threshold in
[The analysis plan](#the-analysis-plan) is quoted at the 0.010 midpoint. That is an estimate
from the literature, not a measurement of this configuration, and all of it scales linearly
with σ̂. **No treatment arm is submitted until the stage-1 rows have numbers in them.**

**The first two rows are what stage 2 is gated on, and the second one is a degree of freedom
if it is left until later.** Per-source seed σ spans an order of magnitude on DataDecide, with
code-type sources several times noisier than web text, so the unweighted mean over the seven
sources that this document names as the endpoint is inefficient by construction. Weights
estimated from the baseline alone and committed before a treatment arm exists are a
measurement; the same weights estimated afterwards are a choice with a preferred answer
available. That is the whole reason stage 1 went out on its own.

**Not GLS, and the reason is arithmetic rather than taste.** The efficient weighting over seven
correlated sources is Σ⁻¹1 normalized, and a 7 × 7 covariance estimated from five seeds has
rank 4. It is singular and has no inverse, and every regularization that would give it one is a
knob set after the fact. So the weights come from the diagonal only — either plain 1/s², or two
variance strata split where the within-group scatter of log s is smallest — and the covariance
is used once, to *report* what the resulting fixed vector achieves rather than to choose it.
Strata is the default: at df = 4 a per-source 1/s² is itself so noisy that the estimator stops
behaving like a fixed-weight average, and the tool prints a leave-one-seed-out reduction beside
the in-sample one precisely so that an over-fitted weight vector is visible as the gap between
them.

**One number in `run.baseline-stage.yaml` is wrong and it is worth knowing which.** That file
says the 95% interval on a variance estimate at df = 2 "spans a factor of about 3.4 end to
end." 3.4 is the df = 6 figure, quoted correctly elsewhere in this document. At df = 2 the span
is **12.1**; at the df = 4 five seeds actually buy, it is **4.8**. The argument that file makes
— that five seeds are worth it — is strengthened rather than weakened by the correction, since
the interval it is escaping is three and a half times wider than it claimed.

That file now carries the correction itself, as a labelled note beside the claim rather than as
an edit to it. The 3.4 is left standing where it was written, for the reason the rest of that
file gives about its own paragraphs: the five admitted cells launched from it, and a reader who
saw the number needs to be able to tell that they did.

### Stage 1 landed, and σ̂ is twice the value the design was priced against

All five cells of `run_019fe2f4-f528` reached step 6,000 and finished, on `gpu-8xa100`, seeds
0 through 4, thirteen evaluations each over seven sources. `.edullm/noise-floor.json` is the
frozen artifact and names the five run ids behind every number in it. Nothing below was
computed before all five were terminal, and no treatment cell existed when it was written.

| | value |
| --- | --- |
| σ̂, held-out BPB, unweighted mean of seven sources, step 6,000 | **0.00627 BPB** |
| the same in nats of held-out cross-entropy | **0.0199 nats** |
| c₄-corrected point estimate, which every MDE below is taken from | **0.0211 nats** |
| 95% chi-square interval, df = 4 | **0.0119 – 0.0571 nats**, a factor of 4.8 |
| planning value the analysis plan is quoted at | 0.010 nats |

**The point estimate is 2.1× the planning value and above the top of both stated ranges** —
0.008–0.012 from the literature, 0.007–0.013 from DataDecide around a central 0.009. The band
is not *excluded*: the low end of the 95% interval is 0.0119 nats, which still touches 0.013.
But at df = 4 that interval excludes almost nothing, and a design has to be priced off the
point estimate. **Every threshold in [The analysis plan](#the-analysis-plan) doubles**, because
every one of them is linear in σ.

#### Two of the five runs took a loss spike, and that is 99% of the variance

σ̂ is not seed jitter. The five endpoints fall into two groups that do not overlap:

| seed | held-out BPB at step 6,000 | post-warmup instability |
| --- | --- | --- |
| 0 | 0.68664 | steps 1376–1418, peak grad norm **9.30**, train CE to 6.46 |
| 1 | 0.68789 | steps 1726–1773, peak grad norm **20.45**, train CE to 10.26 |
| 2 | 0.67541 | none |
| 3 | 0.67628 | none |
| 4 | 0.67589 | none |

Typical grad norm is 0.11–0.17 and `max_grad_norm` is 1, so clipping bounded each episode
without preventing it; the optimizer is plain `AdamW` and no step was skipped. Every step
counter is monotonic, so neither episode is a resume or an infrastructure artifact. Both fall
in the same phase of training, near peak LR, roughly a quarter of the way through the cosine.

The consequences are the whole of what stage 1 found.

- **The spike costs 0.0114 BPB — 0.0361 nats — and it is permanent.** The gap between the two
  groups was 0.0141 BPB at step 2,500 and 0.0114 at step 6,000. It is closing at a rate that
  does not close it.
- **It is larger than the effect the module is hunting.** ByteDance's whole ablation is 0.030
  nats. One spike costs more than that.
- **Within a group, σ is 0.00197 nats**, pooled on df = 3. The all-five σ̂ is **10.1×** that,
  and **99.0% of the endpoint variance is the spike-or-not split** rather than run-to-run
  scatter. This configuration is not noisy. It is bimodal.
- **σ̂ is still falling at the horizon**: σ(6,000)/σ(3,000) = 0.88, bootstrap interval
  [0.50, 0.94], which excludes 1. That verdict is real but it is not the endpoint settling —
  it is the two spiked runs slowly failing to catch up. Read as a noise floor the number is
  still moving; read as a spike gap it has essentially stopped.

#### The code-versus-web ordering does not reproduce; it inverts

DataDecide reports code-type sources several times noisier than web text, this corpus is
code-heavy, and the weighting was justified on that basis. Measured here, at df = 4 each:

| source | σ̂ (BPB) | stratum | weight | df behind the weight |
| --- | --- | --- | --- | --- |
| starcoder | 0.00472 | 0 | 0.1890 | 16 |
| algebraic-stack | 0.00497 | 0 | 0.1890 | 16 |
| arxiv | 0.00533 | 0 | 0.1890 | 16 |
| open-web-math | 0.00553 | 0 | 0.1890 | 16 |
| dclm | 0.00699 | 1 | 0.0813 | 12 |
| pes2o | 0.00728 | 1 | 0.0813 | 12 |
| wiki | 0.00911 | 1 | 0.0813 | 12 |

**The two code sources are the two quietest and `wiki` is the noisiest**, and the whole spread
is 1.93× rather than the 4–7× the literature claims. So the premise the weighting was argued
from is not true of this configuration, and the weights are worth correspondingly little: a
variance reduction of **1.18× in sample and 1.18× leave-one-seed-out**, which is 8% off a
standard error. They are at least not over-fitted — the two figures agreeing is what that
looks like.

The reason the ceiling is so low is visible in one comparison. Mean per-source σ̂ is 0.00628
and the composite of all seven is 0.00627, where seven independent sources would have given
0.00237. **The sources move together almost exactly**, because what moves them is a whole-run
event, and no diagonal weighting reaches a common-mode term. The weights are frozen and will be
applied to stage 2 as a constant, which is what was committed; they are simply not the
instrument the plan hoped for.

#### What the tranche can now detect

At the c₄-corrected σ̂, exact noncentral t, two-sided α = 0.05, 80% power. The funded design is
four arms — `baseline`, `faithful`, `output-only`, `mhc` — so k = 4, and the error df is
k(n−1) = 16 unpaired and (k−1)(n−1) = 12 paired. H1 is `faithful` − `baseline`, H2a is
`faithful` − `output-only`, H5 is `mhc` − `faithful`; all three are 5 v 5 and all three share
the pooled σ̂, so this one table is all of them.

| analysis | df | MDE, unweighted | MDE, strata-weighted |
| --- | --- | --- | --- |
| unpaired | 16 | **0.0399** | 0.0368 |
| paired, ρ = 0.0 | 12 | 0.0408 | 0.0376 |
| paired, ρ = 0.3 | 12 | 0.0341 | 0.0315 |
| paired, ρ = 0.5 | 12 | 0.0288 | 0.0266 |
| paired, ρ = 0.7 | 12 | 0.0223 | 0.0206 |
| paired, ρ = 0.9 | 12 | 0.0129 | 0.0119 |

**Unpaired, the design detects 0.040 nats. ByteDance's effect is 0.030 and Tencent's is
0.020.** The tranche as priced cannot resolve the literature effect at full strength, let
alone the attenuation to ~0.010 that a 370M replication should expect. Pairing only reaches
0.030 at ρ ≥ 0.5 and only reaches 0.020 at ρ ≥ 0.75, and ρ̂ cannot be measured from one arm —
which is the pre-registered position and is not revised here.

**The counterfactual is the reason this is worth acting on rather than absorbing.** At the
within-group σ of 0.00197 nats the same 5 v 5 design detects **0.0040 nats unpaired**. That is
a factor of ten, it is bought by removing an instability rather than by buying replicates, and
`SkipStepAdamW` is in this library and is not enabled on any arm.

**What is not concluded here.** Whether the spikes are driven by data order or by numerics is
unknown and decides everything: `build_config` gives arm *a* seed *k* and arm *b* seed *k* the
same data order, so a data-driven spike recurs at the same step in both arms, ρ goes near 1 and
the paired analysis is excellent — and a numerics-driven one is independent across arms, ρ goes
near 0, and pairing costs df for nothing. Both episodes here landed in the same phase of
training on different data, which is suggestive of neither. It is cheaply testable and it has
not been tested.

**A treatment arm that changes spike propensity breaks the contrast rather than answering it.**
This document already predicts `faithful` may be unstable, and the spectral radius climbing to
1.196 by step 200 on the 96M probe is the same worry. An arm where four of five runs spike
reads 0.029 nats worse than the baseline on the endpoint, which clears no gate in the right
direction and would be written up as a decisive negative H1. It would be a finding about
training stability wearing the clothes of a finding about loss. Five Bernoulli draws cannot
separate them: the two spikes in five runs measured here put spike propensity at 0.4 with an
exact 95% interval of **0.053 to 0.853**, so the tranche cannot tell an arm that spikes half as
often from one that spikes twice as often, in either direction.

### The amendment of 2026-08-08: spike skipping on every arm

**This is a change to the pre-registration made after seeing data, and it is recorded as one.**
Everything above this line was written before any treatment cell existed; this section was not.
It was written after `run_019fe2f4-f528` finished, because of what that submission measured,
and no reader should be asked to work that out for themselves. What follows says what was seen,
what changed, what the change costs, and what is now claimed that was not claimed before.

`.edullm/noise-floor.json` is **not** revised, regenerated or deleted. It is the frozen record
of what the design's own comparator did under plain `AdamW`, it is what forced this amendment,
and an amendment that quietly replaced the evidence for itself would be worth nothing. Every
number in the two sections above still stands as measured.

#### What changed

One setting, and one bound.

| | before | after |
| --- | --- | --- |
| optimizer, all four arms | `AdamW` | **`SkipStepAdamW`**, σ-factor 6, rolling window 128 |
| `--hours`, all four stages | 7 | **4** |

The optimizer is `--optimizer skip_step_adamw` in all four A100 specs and is the parser's
default, so an arm cannot lose it by omission. `SkipStepOptimizer.get_step_factor` declines an
update whose batch loss **or** pre-clipping gradient norm is more than six standard deviations
above its own rolling mean of the previous 128 steps. Nothing else about any arm moves. The
`--hours` change is unrelated bookkeeping and is [argued in the spec
header](run.baseline-a100.yaml): the five measured cells ran 2.92 to 3.00 hours against a
7-hour bound, and `--hours` is the hours factor of the approved ceiling, multiplied by attempts
and by every cell.

#### Why, and why the number is not ours

The case is entirely in [Two of the five runs took a loss
spike](#two-of-the-five-runs-took-a-loss-spike-and-that-is-99-of-the-variance) and is not
restated. In one line: 99.0% of the endpoint variance was the spike-or-not split, one episode
costs 0.036 nats permanently, that is more than ByteDance's entire effect, and the optimizer
that would have declined those updates ships in this library and was not enabled.

**The two constants are `SkipStepAdamWConfig`'s own defaults, taken unchanged, and that is the
argument for them.** Every official OLMo-2 and OLMo-3 pre-training script in
`src/scripts/official/` builds `SkipStepAdamWConfig` and not one overrides either. A threshold
chosen here instead would be a threshold chosen after seeing which cells spiked, on the arm
that is the comparator for every hypothesis in the module, and there would be no way to
distinguish it from one tuned until the comparator looked good.

The defaults were nevertheless *checked* against the episodes rather than assumed to cover
them. `.edullm/skip_step_calibration.py` drives the real `get_step_factor` over the five
recorded histories, step by step:

| seed | steps declined of 6,000 | largest triggering grad norm | episode onset |
| --- | --- | --- | --- |
| 0 | 26 | **9.30** | 1376–1418, first declined step **1,374** |
| 1 | 31 | **20.45** | 1726–1773, first declined step **1,726** |
| 2 | 7 | 0.35 | none |
| 3 | 13 | 0.24 | none |
| 4 | 14 | 0.33 | none |

Both episodes are caught **at or before their first step**, on the gradient-norm channel, at
z = +10.7 and z = +26.0 against a rolling norm of 0.153 ± 0.017. The loss channel is blind at
both onsets (z = +0.4 and z = −1.4), which is why the rule has to fire when *either* signal
does and why a loss-only guard would have changed nothing. Onset detection is insensitive to
the constant — 4, 5, 6, 8 and 10 all catch both — so the choice trades only the false-positive
rate on a healthy run, which at 6 is 0.12% to 0.23% of steps.

**This is a replay and not a counterfactual, and the distinction bounds what it licenses.** The
trajectory replayed is the one that spiked; declining an update changes every step after it, so
none of this says what the loss would have become. It says the rule fires at the onset rather
than after the damage, and what it costs a clean run. Both are read off the recorded history.
Neither is a prediction that the spikes are gone.

#### Why uniformly, which is the part that is not optional

Turning skipping on for the arms suspected of instability and not for the comparator would be
the worst available choice. This document [already
predicts](#what-the-tranche-can-now-detect) that `faithful` may spike more than `baseline`
does, so the optimizer would be correlated with the treatment, and every H1 and H2a number
would mix the mechanism with the intervention meant to clean up after it. That is a larger
confound than the noise it removes. So the setting is identical on all twenty cells, pinned in
`A100_STAGE_PINNED`, exempt from nothing in `STAGE_CONTRAST_EXEMPT`, and
`test_the_a100_specs_differ_in_the_arm_and_in_nothing_else` fails on a laptop if one arm
disagrees.

**The comparator has to be run again, which is the real price of this amendment.** An arm that
can decline updates is not comparable to one that cannot, so `run_019fe2f4-f528` cannot serve
as stage 1 of the amended tranche however good its data is. All four `A100_STAGE_SPECS` carry
`run_id=None` and the baseline goes first, exactly as the original ordering requires. σ̂ is
re-measured from the new baseline; the frozen artifact is not carried forward as if the change
had not happened.

#### The confound this creates, and the outcome that replaces it

Suppressing an instability does not make it not have happened; it moves it out of the loss and
into nothing at all, unless something records it. An arm that would have spiked in four runs of
five now reads a couple of hundredths of a nat worse and gets written up as a **result about
quality**, when it is a result about **stability**. Those are different claims and the tranche
would silently return the wrong one.

So skipping is logged as a first-class per-arm outcome, on every arm including the comparator,
by `SkipStepMonitorCallback`:

- `stability/steps skipped` — cumulative, so the last value is the run's total.
- `stability/grad norm at a skipped step` — recorded **only** on a declined step, so the steps
  at which the key exists *are* the list of declined steps and no separate index is needed.
- `stability/largest grad norm at a skipped step` — running maximum, because the count alone
  does not separate a run that declined a dozen unremarkable steps from one that declined the
  onset of a spike. Magnitude does: measured above, the largest trigger on a clean run was 0.35
  against 9.30 and 20.45 on the two that spiked.

The callback reads values the train module has already recorded and reduced, so it adds no
collective and no host-device sync, and it cannot disagree with the optimizer because it
reports the decision taken rather than re-deriving one. It refuses to attach to a run whose
optimizer cannot skip, so an arm that lost the setting fails at start-up instead of reporting a
clean zero for six thousand steps.

#### H7 (stability), pre-registered here rather than after the fact

**H7. `faithful` declines more updates than `baseline`, and at larger triggering gradient
norms.** Directional, against the same comparator as H1, on the twenty cells of the amended
tranche.

- **Primary statistic:** per-run count of declined steps over the 6,000-step horizon.
- **Test:** exact two-sided permutation test on the 5 + 5 per-run counts, α = 0.05. Counts
  cluster within a run and are not 6,000 independent draws, so no Poisson or negative-binomial
  parameterisation is assumed and none is fitted.
- **Secondary statistic:** the per-run maximum triggering gradient norm, same test. This is the
  one that separates "declined a few noisy steps" from "declined the onset of a spike", and a
  count that moves without it is not evidence of instability.
- **Reported regardless of outcome**, with the per-arm counts in full, because a null here is
  informative and is one of the things stage 1 could not produce at all.

**What H7 can and cannot resolve, stated now.** With 5 v 5 the smallest attainable two-sided
permutation p is 2/C(10,5) = **0.0079**, so complete separation between the arms *is*
detectable at α = 0.05 and partial separation mostly is not. This is nonetheless a strict
improvement on what the tranche could say before: a spiked-or-not indicator is one bit per run
and put spike propensity at 0.4 with an exact interval of [0.053, 0.853], which resolves
nothing, whereas a count over 6,000 steps carries considerably more. H7 is a **secondary**
hypothesis and no primary conclusion is conditioned on it.

#### What the fix buys, if it works

At the within-group σ of the three clean cells — 0.00197 nats, 0.00214 after the df = 3 c₄
correction — the same 5 v 5 design detects **0.0040 nats unpaired**, a factor of ten.
**That number is the ceiling and should not be the expectation.** It is estimated on three runs
and is four to five times smaller than the lowest σ anyone has published for this class of
model; believing it means believing this configuration is quieter than anything DataDecide
measured. The honest range:

| residual σ, if the spikes are gone | unpaired MDE | paired, ρ = 0.5 |
| --- | --- | --- |
| 0.0021, the clean-cell floor | 0.0040 | 0.0027 |
| 0.0063, mid | 0.0119 | 0.0086 |
| 0.0105, half the measured σ̂ | 0.0198 | 0.0143 |
| 0.0211, unchanged | 0.0398 | 0.0288 |

**The pre-registered expectation is the literature band, σ ≈ 0.008–0.012, so an unpaired MDE
near 0.015–0.020 nats.** That resolves ByteDance's 0.030 comfortably, sits right at Tencent's
0.020, and still does **not** resolve the ~0.010 a 370M replication should expect from
attenuation. The amendment makes the tranche able to answer H1 at full strength. It does not
make it able to answer H1 attenuated, and nothing in this section should be read as claiming
otherwise.

#### If it does not work

Three outcomes, and the response to each is fixed now so that none of them becomes a second
post-hoc adjustment.

- **Episodes still occur at the same rate.** The replay says the rule fires at the onset of
  both known episodes, so this would mean new episodes with a signature neither channel sees.
  **Do not re-tune the threshold.** A second threshold chosen after a second look at the
  comparator is fitting, not fixing. Publish σ̂ as measured, and treat the instability as the
  finding — it is a more interesting one than the effect the tranche went looking for.
- **Episodes are suppressed but σ̂ stays high.** Then the variance was never only spikes, and
  the answer is replicates rather than another intervention. At σ = 0.0105 an unpaired 0.010
  needs 18 seeds per arm; at the measured 0.0211 it needs 71. Both are re-scoping decisions
  with a price, and the right response is to state the price rather than to keep adjusting the
  design until the number looks affordable.
- **Skipping suppresses the episodes but changes the endpoint.** Watch for it: the three clean
  cells declined 7, 13 and 14 steps apiece, so `baseline` under this rule is not quite the
  `baseline` that was measured. ~~It is the same comparison for all four arms, so the contrasts
  are unaffected~~, but any comparison drawn to `run_019fe2f4-f528`'s absolute numbers is not
  like-for-like and is not to be made.

  **SUPERSEDED BY [The dose amendment of 2026-08-10](#the-dose-amendment-of-2026-08-10-a-declined-step-is-a-dose-and-the-dose-is-endogenous).**
  The struck clause is the error. An identical *rule* is not an identical *treatment* when the
  rule's action rate is endogenous to the treatment: `get_step_factor` compares each run against
  a rolling window of that run's own previous 128 steps, so the number of declined steps is a
  post-randomisation variable on the causal path from arm to endpoint, and an arm that declines
  more has been trained less at the same nominal horizon. The rest of the bullet stands.

### The amended baseline landed, and the first outcome of the three above is the one that happened

`run_019fe40f-c71e`, five cells, seeds 0 through 4, `gpu-8xa100`, 6,000 steps, thirteen
evaluations each over seven sources, all five terminal. It differs from `run_019fe2f4-f528` in
the optimizer and in nothing else, which is checked rather than asserted: both submissions
carry the same `init_seed` 12536–12540 and the same `data_loader.seed` 0–4 cell for cell, no
cell of either carries a `hyper_connections` block, and the configs read
`AdamWConfig` against `SkipStepAdamWConfig` with `sigma_factor` 6 and
`rolling_interval_length` 128.

**`.edullm/noise-floor-skip-step.json` is the frozen floor for the amended tranche.**
`.edullm/noise-floor.json` is not touched, for the reason the amendment already gives about
itself: it is the record of what forced the change and it is what the change is read against.
The two artifacts sit side by side and name their own submissions.

| | `run_019fe2f4-f528`, `AdamW` | `run_019fe40f-c71e`, `SkipStepAdamW` |
| --- | --- | --- |
| σ̂, held-out BPB, unweighted mean of seven sources, step 6,000 | 0.00627 | **0.00061** |
| the same in nats of held-out cross-entropy | 0.0199 | **0.00193** |
| c₄-corrected point estimate, which every MDE below is taken from | 0.0211 | **0.00205** |
| 95% chi-square interval, df = 4 | 0.0119 – 0.0571 nats | **0.00116 – 0.00555 nats** |
| σ(6,000)/σ(3,000) | 0.88 [0.50, 0.94], still moving | **0.89 [0.28, 1.54], settled** |

**σ̂ fell by 10.3×**, and the second of the three outcomes above — episodes suppressed but σ̂
still high — did not happen. Neither did the third. What happened is that the variance really
was the spikes, in the proportion the earlier section measured it at, and removing them leaves
the floor the three clean cells had already implied.

##### Correction of 2026-08-10: what the agreement establishes, and what it does not

The paragraph that stood here claimed the clean-cell floor "has now been measured a second time
… and the two estimates agree. That is not proof that the configuration is quiet; it is **two
independent samples** saying so." That was wrong twice and the corrections are written before
any treatment endpoint is visible.

**The two samples are not independent and are not disjoint.** Both submissions run seeds 0
through 4, at the same `init_seed` 12536–12540 and the same `data_loader.seed` cell for cell —
this document says so itself two paragraphs above, and both frozen artifacts list
`"seeds": [0,1,2,3,4]`. The clean-cell estimate is built on seeds 2, 3 and 4 plus the
within-pair spread of 0 and 1; the amended estimate is built on seeds 0 through 4. **The second
sample is a superset of the first, not a second draw.** σ is by definition the spread across
seeds, so two estimates over the same seed draw share the entire quantity being estimated: if
those five seeds happen to be a tight draw, both estimates are low together and their agreement
says nothing about the population. **The degrees of freedom are 4, and stay 4.** The
`test_noise_floor` assertion that the two artifacts share no run id is a provenance check and
was read here as though it were an independence check; it is not one and cannot be.

**What the agreement does establish, which is worth having and is a smaller thing.** It is a
**reproducibility check on the whole pipeline**. On the three shared clean seeds the two
submissions correlate at **r = 0.96** and reproduce each other to within +0.00011 to +0.00056
BPB, under a change of optimizer, on different hosts, through a separate read of W&B. That says
the endpoint is a stable function of the seed and that nothing in the reading path is adding
noise of the size being measured. It does not say the five seeds are a representative draw, and
no claim resting on that reading may be made.

**And the mechanism the low σ̂ was attributed to never fired.** The largest triggering gradient
norm across all five amended cells is 0.712, no cell put a gradient norm above 1.0 after warmup,
and no episode began. The rule declined 10 to 20 ordinary steps per run and did not once decline
a spike onset. So the 10.3× fall in σ̂ **cannot be attributed to spike suppression**: whatever
prevented the episodes, it was not the pre-registered mechanism doing the thing the replay
predicted. Three explanations fit the data equally well and this document commits to none of
them — the handful of ordinary declined steps before step ~1,374 averted the episode; the
changed kernel path did (`AdamWConfig` builds `torch.optim.AdamW` and `SkipStepAdamWConfig`
builds this repository's own foreach re-implementation, so "differs in the optimizer and in
nothing else" understates it to a numerics perturbation); or nothing averted it and 0 spikes in
5 was a draw at p = 0.078 against the pre-amendment propensity of 0.4. **The sentence "the
variance really was the spikes" above is retained as what was believed at the freeze and is not
supported by the amended run.**

#### The endpoint, cell by cell, and what the skipping cost a clean run

| seed | `AdamW` BPB | `SkipStepAdamW` BPB | Δ | declined steps of 6,000 | largest triggering grad norm |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.68664 | 0.67565 | −0.01098 | 19 | 0.712 |
| 1 | 0.68789 | 0.67521 | −0.01268 | 10 | 0.485 |
| 2 | 0.67541 | 0.67566 | +0.00024 | 16 | 0.387 |
| 3 | 0.67628 | 0.67684 | +0.00056 | 18 | 0.465 |
| 4 | 0.67589 | 0.67600 | +0.00011 | 20 | 0.420 |
| σ̂ over the five | **0.00627** | **0.00061** | | | |
| range, best to worst | 0.01248 | 0.00163 | | | |

**The bimodality is gone and the two cells that produced it are now the two best.** Seeds 0 and
1 are the pair that spiked; under the amended optimizer they finish at 0.67565 and 0.67521,
which are the lowest two of the five, and their movement is −0.011 and −0.013 BPB, or −0.035
and −0.040 nats. That is the whole of the spike penalty this document priced at 0.0114 BPB,
recovered.

**The largest triggering gradient norm across all five cells is 0.712, against 9.30 and
20.45.** No cell of the amended run ever put a gradient norm above 1.0 after warmup: the
per-cell maxima of 3.33 to 3.75 all occur at steps 1 to 3, before the rolling window has half
filled, and the identical maxima appear at the identical steps in the `AdamW` run, so they are
the initialization transient and not an episode. Declining continues at a low rate to the end
of every cell — last declined steps 5663 to 5897 — and every trigger after step 3,000 is at or
below 0.291. There is no late episode in any cell, and the check is at step 6,000 rather than
mid-flight.

> **Provenance, 2026-08-10.** This sentence used to end "rather than at the ~2,700 the mid-run
> read covered". That figure had no recorded invocation behind it. The only mid-flight read of
> this submission on record is the synthetic one described under
> [The reading this decision was nearly made on was synthetic](#the-reading-this-decision-was-nearly-made-on-was-synthetic-and-the-tool-now-refuses-it),
> which read no data at all — so the clause was citing a step count from a report that had none,
> in the same section that exists because a read of that submission had no provenance. The claim
> it supported does not need it: "there is no late episode in any cell" is established directly
> above, at step 6,000, from the frozen artifact. The clause is deleted rather than sourced.

**Skipping did not move the endpoint, which is the third outcome the section above said to
watch for.** The five amended cells average 0.67587 BPB; the three clean `AdamW` cells average
0.67586. The difference is +0.00001 BPB, +0.00003 nats. Per seed the three clean cells moved by
+0.00024, +0.00056 and +0.00011 BPB — all one sign, all inside the floor, and consistent with
declining 16 to 20 updates costing a hair of progress rather than nothing. This is the
comparison the section above forbids drawing conclusions from, and none is drawn from it: it is
reported because that section asked for it to be watched, and what it establishes is the
absence of an effect rather than the presence of one.

#### Per source, and what the frozen weights are now worth

| source | σ̂ (BPB) | stratum | weight | df behind the weight | σ̂ under `AdamW` |
| --- | --- | --- | --- | --- | --- |
| starcoder | 0.00102 | 1 | 0.0892 | 16 | 0.00472 |
| dclm | 0.00075 | 1 | 0.0892 | 16 | 0.00699 |
| arxiv | 0.00073 | 1 | 0.0892 | 16 | 0.00533 |
| algebraic-stack | 0.00068 | 1 | 0.0892 | 16 | 0.00497 |
| open-web-math | 0.00057 | 0 | 0.2144 | 12 | 0.00553 |
| pes2o | 0.00057 | 0 | 0.2144 | 12 | 0.00728 |
| wiki | 0.00040 | 0 | 0.2144 | 12 | 0.00911 |

Every source fell. The spread is **2.55×** end to end against the `AdamW` run's 1.93×, the
weights run from 0.0892 to 0.2144, and **no source is down-weighted to anything near zero**.
`starcoder` is now nominally the noisiest rather than the quietest, which is a reordering
inside a factor of two and a half at df = 4 — under true equality the largest of seven such
estimates exceeds the smallest by about this much routinely, and nothing should be read into
it in either direction.

The weighting buys **1.33× of variance in sample and 1.08× leave-one-seed-out**, which is 3.6%
off a standard error. Plain inverse-variance would give 1.70× in sample and 1.15× out. Both
gaps between the in-sample and out-of-sample figures are the over-fitting the tool prints them
to expose, and both out-of-sample figures are small enough that the choice does not matter.

**The primary endpoint stays the unweighted mean of the seven sources, which is what
[Pre-registered hypotheses](#pre-registered-hypotheses) named before any of this.** The
weighted composite is frozen in the artifact and reported beside every contrast as a
secondary, because that is what was committed and a committed number is reported whatever it
turns out to be worth. Making it primary would be the deviation, it would be a deviation
adopted after seeing that it flatters the design, and it would buy 3.6%.

Three post-hoc diagnostics, run because a per-source reordering is the kind of thing that
turns out to be one bad cell, and recorded here as post-hoc because they were chosen after
seeing the data and no hypothesis rests on them.

- **No cell is an outlier on any source.** The largest studentized residual over all
  forty-nine cell-by-source values is 1.67, at seed 3 on `open-web-math`; the Grubbs 5%
  critical value at n = 5 is 1.72. Seed 3 is the highest of the five on six of the seven
  sources, which is what the maximum of five draws looks like and not what one anomalous run
  looks like.
- **No single deletion carries the floor.** Dropping each cell in turn gives σ̂ of 0.00218,
  0.00177, 0.00218, 0.00104 and 0.00221 nats. The worst of the five still prices the unpaired
  design at 0.0044 nats.
- **The code sources are not where the residual variance lives.** Over the five non-code
  sources alone σ̂ is 0.00052 BPB, 0.00165 nats; over the two code sources alone it is 0.00085
  BPB. Under `AdamW` those two figures were 0.00684 and 0.00484 — the non-code sources were the
  noisier group then, and they are the quieter group now, and both orderings are inside the
  scatter that df = 4 produces.

The sources still move together, and less than they did: the mean pairwise correlation across
the seven fell from +0.997 to +0.685. That is the common-mode term the earlier section
identified as the ceiling on any diagonal weighting, and it is what still holds the composite
at 2.3× the value seven independent sources would give.

#### What the tranche can now detect

At the c₄-corrected σ̂ of 0.00205 nats, exact noncentral t, two-sided α = 0.05, 80% power, four
arms and five seeds, so error df is k(n−1) = 16 unpaired and (k−1)(n−1) = 12 paired.

| analysis | error df | MDE, unweighted | MDE, strata-weighted |
| --- | --- | --- | --- |
| unpaired | 16 | **0.0039** | 0.0034 |
| paired, ρ = 0.0 | 12 | 0.0040 | 0.0034 |
| paired, ρ = 0.3 | 12 | 0.0033 | 0.0029 |
| paired, ρ = 0.5 | 12 | 0.0028 | 0.0024 |
| paired, ρ = 0.7 | 12 | 0.0022 | 0.0019 |
| paired, ρ = 0.9 | 12 | 0.0013 | 0.0011 |

**Unpaired, the design detects 0.0039 nats.** ByteDance's effect is 0.030, Tencent's is 0.020,
and the attenuation to roughly 0.010 that a 370M replication should expect — the one this
document has said twice that no affordable allocation reaches — is resolved with a factor of
2.6 to spare. The pairing is no longer load-bearing for anything, which retires the risk that
ρ̂ lands under the 0.09 break-even and the primary analysis has to be swapped.

**The design survives the whole 95% interval on σ̂, which is the reading that should carry the
decision rather than the point estimate.** At df = 4 the interval spans a factor of 4.8, and at
its pessimistic end σ is 0.00555 nats and the unpaired MDE is **0.0105 nats** — still inside
Tencent's 0.020 and still at the attenuated 0.010. At its optimistic end it is 0.0022. There is
no point in that interval at which the tranche cannot answer H1 and H2a against the published
effects, and that is what changed: the earlier reading could not answer them anywhere in its
interval.

Two things this does not buy. It does not buy H2b, which needs downstream scoring that is not
in this plan or this budget. And it does not make H7 resolvable beyond what
[H7 (stability)](#h7-stability-pre-registered-here-rather-than-after-the-fact) already
states — with no cell of the comparator spiking, the per-run counts are 10 to 20 and the
largest triggers 0.387 to 0.712, so a treatment arm that spikes will separate on the secondary
statistic and one that merely declines a few more steps will not.

#### The reading this decision was nearly made on was synthetic, and the tool now refuses it

**`python .edullm/noise_floor.py --submission run_019fe40f-c71e --dry-run` reads nothing.** The
W&B query sits behind `--group`; `--submission` narrows that query and does nothing on its own.
With no `--group` the run fell through to the synthetic generator and printed a complete,
internally consistent report — per-source σ, strata, weights, in-sample and leave-one-seed-out
variance reductions, a settling verdict, and both MDE columns — of a fiction, under the
submission id it had been handed on the command line.

It said so. `falling back to SYNTHETIC data with a known truth. Nothing below is measured.` is
the first line, and every block below it is prefixed `[synthetic]`. It was read as a
measurement anyway, and the numbers in it are worth writing down beside the real ones because
of how nearly they were acted on:

| | synthetic, read as measured | measured |
| --- | --- | --- |
| unweighted composite σ | 0.00836 BPB | **0.00061 BPB** |
| strata-weighted composite σ | 0.00242 BPB | **0.00053 BPB** |
| variance reduction | 11.95× in sample, 9.18× out | **1.33× in sample, 1.08× out** |
| per-source spread | 36× | **2.55×** |
| σ̂ on `starcoder` | 0.06073 | **0.00102** |
| unpaired MDE, unweighted | 0.0544 nats | **0.0039 nats** |

The generator plants a DataDecide-shaped spread with the code sources at the noisy end —
`starcoder` at 0.0300 and `algebraic-stack` at 0.0240 against `dclm` at 0.0035 — so the
"finding" that the two code sources had become thirteen times noisier than everything else was
the generator's own parameters read back out, and the 12× variance reduction was the weighting
correctly deleting sources whose noise was never in this corpus. Acting on it would have made
the weighted composite the primary endpoint of a module whose two hypotheses are about a
residual-topology change, on a weighting that suppressed the two sources such a change would
most plausibly show up on, to buy a factor that does not exist. It cost about twelve hours and
it stopped one commit short of the freeze.

Two changes, both in `noise_floor.py`, both tested:

- **`--submission` without `--group` is now an argparse error.** It named a submission and read
  none of it, and there is no reading under which that combination means anything.
- **The synthetic banner is repeated at the foot of the report.** The report is sixty lines and
  the MDE table is the last thing in it, so a warning on line one is a warning below the fold
  by the time anybody is looking at the number they came for.

This is the same failure as
[`wandb_panels.py --verify` passing for the wrong reason](#wandb_panelspy---verify-passed-for-the-wrong-reason-and-now-scopes-to-a-submission),
and it is the second time a tool in this directory has answered a question about one submission
using data that was not that submission's. The pattern is worth naming: every one of these
tools takes a submission id, and the ones that got it wrong were the ones where the id was
accepted and then not used.

#### Go, on the unweighted endpoint

The three treatment arms are cleared to submit. `edullm check --json` on
`.edullm/run.{faithful,output-only,mhc}-a100.yaml` at `--compute gpu-8xa100 --hours 4
--attempts 2 --fanout-size 5 --fanout-index-parameter seed` returns `refused: false` with an
empty `refusals` list and exit 0 for all three, `approval_class: routine`,
`approving_environment: run-approval-lead`. Read `cost` out of that output rather than out of
this paragraph.

The pre-registered gate is unchanged and is now read at σ̂ = 0.00205 nats: a claim needs
|Δ̂| ≥ 2 × SÊ(Δ̂), which at 5 v 5 on the pooled σ̂ is 0.0026 nats, with the exact two-sided t
p-value reported beside it and the 5% line at 2.45 SE. The primary analysis is the paired
difference if ρ̂ clears 0.09 and the unpaired difference if it does not, exactly as
[Paired by seed](#paired-by-seed-is-the-primary-analysis-and-it-is-free) commits; the design
now clears every literature effect under both, so that choice no longer decides anything.

**σ̂ is re-pooled across all twenty cells once the treatment arms land, and Bartlett is run on
the four within-arm variances**, as the plan commits. The frozen floor prices the design; it is
not the σ the contrasts are tested against. A treatment arm that is noisier than the comparator
raises the gate, which is the correct behaviour and is why the pooling was pre-registered.

### The dose amendment of 2026-08-10: a declined step is a dose, and the dose is endogenous

**Written while `run_019fe90b-f99e` had about three hours left to run and the other two
treatment stages were queuing, so that no treatment arm had an endpoint to be tempted by.
Committed later the same day, by which point `faithful` and `output-only` had joined `baseline`
at 5/5 and `mhc` stood at two cells done and three running.** The gap between writing and
committing is a machine timeout and nothing else, and it is recorded here rather than quietly
closed, because the honest claim is not "no endpoint existed" — by the time of the commit three
arms of endpoints did exist — but this:

**No treatment arm's endpoint, per-source table, declined-step count or largest trigger has been
read by the author of this amendment, at any point, and none was read before the commit that
carries it.** The three completed arms were reported as *counts of finished cells* by the
researcher; a cell count is not an endpoint. Every constant in the rule below is derived from
the `baseline`-versus-`baseline` pair in [the cell-by-cell
table](#the-endpoint-cell-by-cell-and-what-the-skipping-cost-a-clean-run), which is a
control-arm reproducibility comparison that predates all of this and contains no treatment
number.

That distinction is the whole value of the section. Every choice below is free while it is
unread and is a researcher degree of freedom the moment an arm mean is in front of you. The
estimand does not move, and the rule added can only ever withhold a claim — so even a reader who
disbelieves the paragraph above can check that the machinery has no channel through which a
result could be manufactured, only one through which it can be refused.

#### The finding, verified against the code rather than taken on report

`SkipStepAdamW` declines a step by multiplying the entire update by a 0/1 factor. Verified
directly, and asserted by
[`test_dose_adjustment.py`](test_dose_adjustment.py) against the real optimizer on both kernel
paths rather than restated from a reading:

- the parameters do not move — `update.mul_(step_factor)` then `p.add_(update)`
  (`src/olmo_core/optim/adamw.py:44-45`, and `:99-100` in the foreach kernel);
- neither moment moves — the lerp weight is `step_factor * (1 - beta1)` and the second moment's
  is `step_factor * (1 - beta2)` (`:32-34`, `:78-86`);
- the decoupled weight decay does not apply — `p.mul_(1 - step_factor * (lr * weight_decay))`
  (`:29`, `:71`);
- **and the Adam step counter does not increment** — `step.add_(step_factor)` (`:47`, `:101-102`),
  reached whenever `step_increment_bugfix` is true, which is `SkipStepAdamWConfig`'s default and
  therefore this tranche's setting.

The trainer's global step, the cosine schedule and the data loader advance regardless. **So a
declined step consumes its tokens, moves the schedule along, and performs no optimization.** An
arm that declines more is an arm that has been trained less at the same nominal horizon.

The rule is identical on all four arms and that is what the earlier section argued from. It is
the wrong thing to argue from. `get_step_factor` compares each step against the mean and
standard deviation of **that run's own** previous 128 losses and gradient norms
(`skip_step_optimizer.py:94-109`), so the *action rate* is endogenous to the treatment even
though the *rule* is fixed. Declined count is a post-randomisation variable on the causal path
from arm to endpoint. Holding the rule fixed converts a loss confound into a training-duration
confound, and nothing in the design bounds it.

Two consequences worth stating separately, because they cut in opposite directions.

- **H7 is not rescued by this and is further damaged by it.** The threshold being run-relative
  means an arm whose gradient norms are uniformly elevated but smooth raises its own bar and may
  decline *fewer* steps than the baseline. The count is not monotone in instability, which
  [`test_dose_adjustment.py`](test_dose_adjustment.py) demonstrates by rescaling a synthetic
  run's gradient norms tenfold and getting an identical count. H7 stays secondary and stays
  reported; nothing primary was ever conditioned on it.
- **The dose is nonetheless the right variable for the contrasts**, precisely because it is a
  count of updates not taken. Whatever it says about stability, it says something exact about how
  much training happened.

**A third mechanism fact, which bounds how large `Δn` can plausibly get.** The step's own loss
and gradient norm are appended to the rolling window *before* `step()` consults it — the setters
append unconditionally (`skip_step_optimizer.py:59-76`) and nothing removes the value when the
step is declined. So a declined step stays in the window that judges the next 128 steps, and
having declined once it raises the very mean and standard deviation that would be needed to
decline again. Driven against the real optimizer, a spike is declined **once** and the two
identical spikes that follow it are both accepted; the mechanism poisons its own detector.
[`test_dose_adjustment.py`](test_dose_adjustment.py) asserts exactly that.

Two things follow, and the second is the one that matters here.

1. Declines are **anti-clustered**, not clustered. A single instability episode costs at most
   one update per window rather than a run of them, so `Δn` accumulates as a slow difference in
   rates over 6,000 steps rather than as a burst. The 49-step and 17-step gaps in the table below
   remain reachable that way, so this tightens nothing in the rule — but it does mean a gap of
   that size implies a *sustained* difference in behaviour between arms and not one bad minute.
2. It is a **third independent reason not to use the count as a covariate**. A regression of
   endpoint on declined count assumes the count measures the missing dose. Here the count is
   censored by its own history — the same underlying instability yields a different count
   depending on when in the window it arrived — so the count understates the dose by an
   arm-dependent and unknowable amount. The rule below therefore uses the count only to bound a
   contrast, never to correct one, and the bound is taken at the top of the slope interval.

#### The calibration, and the interval nobody had put on it

The only pre-existing measurement of what declining costs is the three baseline seeds that never
spiked, run once under `AdamW` and once under `SkipStepAdamW` at the same `init_seed` and the
same `data_loader.seed`. From this document's own
[cell-by-cell table](#the-endpoint-cell-by-cell-and-what-the-skipping-cost-a-clean-run):
+0.00024, +0.00056 and +0.00011 BPB at 16, 18 and 20 declines.

| | nats per declined step | the decline gap that spans the 0.0026 gate |
| --- | --- | --- |
| point estimate, ratio of means | 5.34e-05 | 49 steps |
| **95% upper, t on df = 2** | **1.55e-04** | **17 steps** |
| 95% lower | −4.79e-05 | — |

**The point estimate is right and it is not the operative number.** The mean movement is
+0.00096 nats and it does not clear zero: t = 2.27 on df = 2, p = 0.15. The upper end of its
interval is 2.9 times the point, and at that slope the entire pre-registered gate is spanned by
a **seventeen-step** difference in declined counts — against an amended baseline whose own five
cells declined between 10 and 20. A spread that size is inside what one arm has already
produced.

Three things are wrong with the estimator and all three are recorded rather than smoothed over.

1. It is a **ratio of means through the origin**, not a regression: it attributes the whole
   `AdamW`→`SkipStepAdamW` movement to declining. The two optimizers are also a different kernel
   path, so part of that movement is numerics. The slope is therefore an **over-estimate**, which
   is the direction a withholding rule wants and the reason the band is quoted at the top of the
   interval rather than at the point.
2. It rests on **three cells**, and the identification comes from the pairing — the same seed run
   twice under a near-identity perturbation — rather than from the spread of the counts.
3. The **within-sample regression** of movement on declined count over those same three cells is
   **negative**, at −1.03e-04 nats per decline. Sixteen, eighteen and twenty declines carry
   almost no leverage, so that number is noise; it is recorded because anybody who recomputes the
   slope the obvious way will find it and should know it was seen and rejected on those grounds.

The direction is certain — a declined step is strictly less optimization — and the magnitude is
known to within about a factor of three.

#### What was chosen, and what was rejected

**The primary estimand does not move. It stays the total effect at 6,000 global steps.**
Declining is part of what an arm does, so the intention-to-treat contrast is the quantity the
module's question is about, and swapping it for a mediator-adjusted quantity after a review, on a
slope this imprecise, would be the larger error. What is added is a band beside every contrast.

- **Equal applied updates rather than equal global steps** is the clean fix and it is
  unavailable. Three arms are already running to 6,000 *global* steps. Evaluations land every 500
  steps, which is ten to fifty times coarser than the tens of steps the confound is made of. And
  the cosine is indexed to the global step, so two cells matched on applied updates sit at
  different points of their own schedules — it would remove one confound by introducing another.
  It is the right definition for the *next* tranche and it is written down here for that reason.
- **Declined count as a covariate in the contrast** conditions on a mediator. It estimates a
  controlled direct effect rather than a total effect, under an unverifiable assumption, and if
  hyper-connections genuinely destabilise training then the adjustment subtracts part of the real
  effect. It is reported as a secondary and it is not primary.
- **Adjusted and unadjusted side by side** is what is adopted — with the rule below, because two
  numbers printed next to each other and no rule about what to do when they disagree is not a
  pre-registration, it is a degree of freedom with a table.

**The rule.** Let Δ̂ be the contrast in nats, signed so that negative is the treatment improving,
and Δn the treatment's mean declined count minus the comparator's. The dose contributes +β·Δn
nats to Δ̂.

- When **β·Δn carries the same sign as the hypothesis's predicted effect**, the dose could have
  manufactured the result. The claim stands only if |Δ̂| − |Δn|·β_high ≥ gate, with
  β_high = 1.55e-04. Otherwise the contrast is reported as **dose-limited** and **the claim is
  not made**.
- When it carries the opposite sign, the treatment was trained less and scored well anyway. The
  unadjusted estimate is conservative, the amount is reported, and **no penalty is applied**.
- Δn and the critical gap `gate / β_high` are printed beside every contrast whatever the verdict,
  so a reader can see how close the design came without reading this section.
- The point-adjusted estimate Δ̂ − β̂·Δn is reported as a pre-committed secondary.

**The assumptions, stated so they can be attacked.** That the per-declined-step cost is
approximately linear in the count over the range that arises; that it does not depend strongly on
*where* in the schedule the declines fall, which is false in detail — a declined step early is
worth more than one at the horizon — and is why β is taken at the top of its interval rather than
at the point; that β estimated on the comparator transfers to the treatment arms, which is the
assumption the two optional calibration cells below would test; and that the endpoint is
monotone in applied updates, which is what makes the rule one-sided.

**Why it is safe to freeze a slope this poorly determined.** Because the rule is one-sided and
can only subtract. Nothing it does can turn a contrast that failed the gate into one that passed
— a property asserted over a grid of contrasts and dose gaps in
[`test_dose_adjustment.py`](test_dose_adjustment.py). A constant that can only ever cost the
experiment a claim is not a knob worth tuning.

**Not funded, and available if anybody wants the slope narrowed.** Two baseline cells under
`SkipStepAdamW` with `--skip-step-sigma-factor` set high enough that the rule never fires would
separate the kernel path from the dose; two more at a factor low enough to force a target decline
count would identify the slope on more than three points. Until one of them runs, the constants
do not move: they are frozen literals in
[`dose_adjustment.py`](dose_adjustment.py), a test re-derives them from the table above, and a
second test asserts they have not drifted.

#### The thirteen checkpoints are not horizon leverage, and the claim is dropped

The adversarial review proposes pre-registering the H1 and H2a contrasts as trajectories over the
thirteen evaluation checkpoints on the grounds that "a contrast growing at the horizon is
evidence for cause 4" — the token-budget explanation, ByteDance's 500B against Tencent's 23B —
and that it "is the only horizon leverage this tranche has". **That claim is dropped and the
reason is the review's own.**

The same review objects, correctly, that DataDecide's exponent is well supported as a within-run
checkpoint slope and much less supported as a between-horizon substitution, because its
intermediate checkpoints are "checkpoints of runs scheduled to a much longer horizon at a much
higher LR" and "not the same object as the endpoint of a run whose cosine completes at 4.72B".
Every one of these thirteen checkpoints is exactly that object: a point on a cosine scheduled to
complete at step 6,000, at an LR the corresponding short run would never have had. The objection
does not stop applying because it is this tranche's data. Beyond that, the trajectory spans 0 to
4.72B and cause 4 is a question about 23B against 500B, so even a clean trend would be a two-order
extrapolation from a single completing schedule; and thirteen points per cell are heavily
autocorrelated and add no degrees of freedom to five seeds.

**So: no horizon claim, no extrapolation, and nothing about cause 4 from this tranche.** An H2a
gap will not support "the field's negative result is an implementation artifact" and this is the
second place the document says so.

What *is* pre-registered, now, is the weaker and genuinely free version. Both are secondary,
descriptive, and carry no claim about horizon.

- **The contrast trajectory as a specification check.** The H1, H2a and H5 contrasts are reported
  at all thirteen checkpoints. A contrast that is present throughout and a contrast that appears
  only at step 6,000 are different objects, and the second is the one that wants explaining
  before it is believed. This is a plot and a table, not a test, and no p-value is computed from
  the trajectory.
- **The endpoint progress rate as a cross-check on β.** The local slope of held-out loss against
  step over the last checkpoint interval bounds what one lost update can cost, from above, since
  optimization partially recovers from a skipped step. If the frozen β_high exceeds that bound the
  band is too wide and the report says so. This costs nothing and it is the only independent
  handle on the dose constant the tranche has.

### Arm 4 is funded, because H1 cannot be decomposed without it

**Staged 2026-08-10, in the same window and under the same condition: no treatment endpoint was
visible.**

`faithful` differs from `baseline` in **two** things and this document has said so from the
start — the mechanism, and a smaller initialization for every attention output projector and
second feed-forward linear, at the paper's `output_init_exponent=0.5`. Same-seed logits differ by
a relative 8.4e-01 against 7.2e-07 at 0.0, so it is not a rounding difference. The remedy was
named in the same breath — "unless arm 4 comes back flat against arm 1" — and arm 4 then carried
`seeds=0`, at the end of `CUT_ORDER`, on the reasoning that a question about whether a scaling is
load-bearing "only means something once H1 says the method does anything at all".

**That reasoning has the dependency backwards, and the amended σ̂ is what makes it matter.** H1 is
a joint test of the mechanism and its initialization prescription whatever it returns. At the
pre-amendment MDE the point was academic, because the design could not resolve the effect at all.
At an unpaired MDE of 0.0039 nats it can now return a **significant** H1 attributable to either
cause, with nothing in the tranche able to say which — and the scaling was put behind a flag in
the original build order precisely so that arm 4 could turn it off. Arm 4 is not a follow-up to
H1; it is a precondition for reading it.

`no-output-init` therefore goes to five seeds, leaves `CUT_ORDER`, and is staged as
[`run.no-output-init-a100.yaml`](run.no-output-init-a100.yaml) at `--compute gpu-8xa100
--attempts 2 --fanout-size 5 --fanout-index-parameter seed`. The tranche is twenty-five cells.
Read `cost` and `approval_class` out of `edullm check --json` rather than out of this paragraph.

Three things about the spec that are not the other three stages'.

- **`--hours 7`, and it is the only header that departs from the shared bound.** The four hours
  the others quote is `A100_MEASURED_CELL_HOURS`, which is the *baseline's* cell at 1.700 s/step;
  the lane arms run at 2.87 to 3.15 and every cell of all three treatment stages died at that
  wall. Arm 4 is `faithful` with two module families initialized differently and nothing a kernel
  sees, so its step time is `faithful`'s: slowest 3.074 s/step, projecting 5.17 hours. Seven
  leaves 26%; six would leave 14%.
- **The equal-bound rule is departed from deliberately and the departure is bounded.**
  `STAGE_HOURS` records why a bound should be the same across stages — an arm under a looser bound
  survives drift another dies of, so an arm missing its slowest cell is not missing a random one.
  If a live arm loses a cell to its own bound and arm 4 does not, arm 4 carries a slow cell its
  comparator does not, and the bias runs towards arm 4 looking *worse*, which reads as the
  initialization being load-bearing — a false positive on the exact question the arm was funded to
  answer. It needs a 16% rate excursion against a measured host-to-host spread of 1.8%. And
  `A100_LANE_ARM_SURVIVORSHIP_HOURS` pre-registers, now, that **any arm-4 cell whose runtime
  exceeds 6.0 hours is reported and the contrast recomputed without it**.
- **`--fail-closed-by-step 400` is present, and that is measured rather than assumed.** It is on
  `faithful` and `output-only`, off on `baseline` where the monitor never attaches, and off on
  `mhc` where Sinkhorn compresses the statistic the guard reads. Turning the output scaling off
  makes the module writes *larger*, so lane dispersion should read higher than `faithful` rather
  than lower. Measured at the rehearsal size, arm 4 reads **0.0107 to 0.0243** against
  `faithful`'s 0.0116 to 0.0188 and a floor of 5e-03 — four blocks of four over it on both. This
  arm is the opposite of the `mhc` case and the flag belongs on it.

**What arm 4 buys.** H1 stays as pre-registered and stays a joint test; it is not redefined after
the fact. Two contrasts are added and both are pre-registered here rather than derived later:
`no-output-init` against `baseline` isolates the mechanism without the initialization
prescription, and `no-output-init` against `faithful` isolates the prescription itself. Both
carry the same gate and the same dose band as H1, H2a and H5.

**And both enlarge the multiplicity the design already declines to correct for.** The
no-correction paragraph under
[The gate](#the-gate-two-standard-errors-of-the-contrast-under-test) was written for a
table of six hypotheses reported as a family of effect sizes; the live design leads with a
2-SE gate, which at df = 16 is a 6.3% per-comparison test, and this takes the family from three
comparisons to five. Nothing here changes the pre-registered rule — the gate stays uncorrected
and every contrast is reported with its effect size, interval and exact p-value, as committed.
What is added is that **the Holm-adjusted p-values are printed beside the raw ones**, so a reader
who wants the family-wise reading has it without recomputing anything, and the growth of the
family from three to five is visible rather than implicit.

### Throughput

Measured, not planned. Read from run history at the steady state rather than from the run
summary — the last logged value is taken during the end-of-run evaluation, where the model is
not training and throughput reads near zero.

| run | shape | config | µbatch | steps | device TPS | s/step |
| --- | --- | --- | --- | --- | --- | --- |
| `run_019fdfe9-e6c0` | `gpu-4xl40s` | `hc_rehearsal`, `faithful` | 8,192 | 200 | 57,354 | 1.14 |
| `run_019fe008-5877` | `gpu-4xl40s` | `hc_370M`, `faithful` | 8,192 | 100 | 12,645 | 15.54 |
| `run_019fe1f6-8692` | `gpu-4xl40s` | `hc_370M`, `faithful` | **16,384** | 100 | **19,051** | **10.32** |
| `run_019fe262-778d` | `gpu-8xa100` | `hc_370M`, `faithful` | 12,288 | 74 of 100 | **33,803** | **2.91** |
| `run_019fe279-4ef0` | `gpu-4xl40s` | `hc_370M`, **`baseline`** | 16,384 | 370 of 6,000 | 23,977 | **8.20** |

TPS is per device, so the shape's total is the device count times it, and seconds per step is
the batch over that total: 786,432 / (4 × 19,051) = 10.32 on the L40S, and
786,432 / (8 × 33,803) = 2.91 on the A100.

**The last row is the only `baseline` measurement there is, it is provisional, and the gap
between it and the row above `run_019fe262-778d` is worth more than either.** It is stage 1
running: three of five cells live, clean medians of 8.201, 8.071 and 8.256 s/step over 372, 372
and 362 rows with interquartile spreads of 0.041 to 0.056, one warm-up step dropped, and no lane
monitor attached at all because `train` only attaches it to an arm that has lanes. Every other
row in this table is `faithful`. So on one shape, at one microbatch, at one world size, the
**hyper-connection arms cost about 26% of wall clock against the baseline — 10.32 s against
8.20 — for the +0.0994% of FLOPs the arm table reports.** Iso-FLOP is not iso-time here and was
never going to be: `n_lanes` 4 makes the residual stream four times wider, and at `d_model` 1024
this model is bound by what it moves rather than by what it multiplies, so a change that is a
rounding error in FLOPs is a quarter of the step in bytes.

Two consequences and one thing not to conclude. The tranche's own pricing is unaffected, because
`arm_seconds` prices the treatment arms at 10.32 and that is the arm the bound has to hold: the
faithful cells are 17.85 hours against a 19-hour bound and the baseline cells come in at 14.36,
which is spare the plan did not budget and does not need. And `MONITOR_SECONDS_PER_FIRING` at
1.37 s is the cost of a firing and is *not* the cost of the instrument, since the forward hook
is registered on every block for the whole run — it returns early off its firing steps, but
`@torch._dynamo.disable()` means it is a graph break at every block boundary whether it measures
or not, and the blocks are compiled. What must **not** be concluded is which of those two the
26% is. The lanes and the always-registered hook are confounded in every run this branch has,
and separating them is one 100-step `faithful` probe with the monitor off, for about $4.

**Every median here is over *clean* steps, and getting that filter wrong is what produced the
number this tranche was first priced at.** A clean step is one on which neither the held-out
evaluator nor the lane monitor ran. In `run_019fe1f6` that is 93 of the 100 steps and the
median is 10.32 s, with an interquartile spread under 0.05 s and a wall-clock cross-check of
10.40 s from the run's flush timestamps. The four steps the monitor fired on cost 11.65–11.80 s
and the two the evaluator ran on cost 115 s and 121 s.

**Where 11.69 s/step came from.** Reading the same run's history filtered to rows carrying
`hc/*` keys returns exactly five rows — steps 20, 40, 60, 80 and 100, which are the monitor's
firing steps at `--monitor-interval 20` — whose median is 11.6956. That is the cost of the
*instrument*, sampled five times, not the cost of a step, and the "only about five throughput
points were logged" caveat that came with it is the filter describing itself. The device TPS of
16,822 and MFU of 14.11% quoted with it are step 80 exactly.

The correction matters twice over. It is 11.7% off the runtime, which is four hours on an
eighteen-hour run. And it means the two 370M probes were never compared the same way: the
superseded probe's headline 15.55 is *its* clean median, so doubling the rank microbatch bought
15.54 → 10.32, a 33.6% reduction, not 15.55 → 11.69. That is the number the wire arithmetic in
`run.yaml` predicts — halving the model's crossings of a PCIe interconnect — and it is
reassuring that it lands there.

**The two MFU columns.** Both runs fell through `SpeedMonitorCallback`'s device table to its
A100 default and were scored against a 312 TF peak — that is exactly the ratio of the logged
`flopsPS` to the logged MFU in every row of both histories. The card is an L40S, whose dense
BF16 peak is 362.05 TF, so the honest figure is the second column, 13.8% lower. The callback
gained an L40S branch on 2026-08-08 at 10:14, four hours after both of these runs, so anything
submitted from here lands in the corrected column directly and arm 1 seed 0 will be the first.

> **Superseded, and by a factor of two rather than by 13.8%.** 362.05 is the L40S's BF16 rate
> *with FP16 accumulation*, and torch accumulates in FP32, which on AD102 runs at half rate. The
> figure that belongs in an MFU denominator for this card is **181.03 TF**, so every L40S MFU on
> this page — both columns of this table included — is understated by two. See "The A100 baseline,
> and the second factor of two in the MFU". The TPS figures below are unaffected: tokens per
> second are counted and have no peak in them, which is the same distinction the next paragraph
> makes.

**The rehearsal TPS was recorded as 49,620 and the measured median is 57,354.** The recorded
figure is the measured one divided by 1.156, which is the peak-FLOPs correction from the
paragraph above — a correction that belongs to MFU, a ratio against a hardware constant, and
not to tokens per second, which is counted directly and has no peak in it. The same divisor
also produced the 7.89% MFU that stood here, which is within rounding of the 7.86% the correct
factor gives, so the MFU column was right by accident and the TPS column was wrong by 13.5%.
It is the TPS number that every runtime and cost projection is built on.

#### The A100 probe, read with the same filter, and what it says

`run_019fe262-778d` on `gpu-8xa100`, rank microbatch 12,288, 8 processes, `faithful`. **It
stopped at step 74 of 100.** The paragraph that stood here said it crashed, that reserved memory
at 58% of the card ruled out an OOM, and that the cause was inside AWS behind a workflow
dispatch "which nothing here needs: 74 steps is more than the measurement takes." The dispatch
was worth spending, because it did not crash.

`edullm status run_019fe262-778d` returns `FAILED`, **exit code 137**, and `Why: Cancelled by
philote-dev`, carrying the reason recorded on the cancellation:

> Not decision-relevant. A power analysis on Ai2 DataDecide (1,050 models, 3 seeds) measures
> seed sigma scaling as D^-0.17, not D^-0.5, so at fixed budget standard error scales as
> D^+0.33 and seeds dominate horizon. A100 and L40S also price within 1.5% per token ($47.78/B
> vs $47.08/B), so clearing the probe would not have made tokens cheaper, only a longer run
> legal. The tranche goes to L40S at more seeds regardless of what this measures. Cancelling
> returns $131.75 to a $4,000 ceiling.

**The instrument was switched off by the decision it was measuring for, on the number it was
measuring.** The probe was queued at 17:20:29, started at 17:23:39 and stopped at 17:30:34 —
six minutes fifty-five seconds, attempt 1 of 2 — and by then it had already produced the median
this section is written around. Neither figure in that reason is a measurement: 1.5% is the gap
at the 5.00 s/step `run.yaml` nominates as a budget threshold, and $47.78/B and $47.08/B are
about a quarter above the measured per-token prices in the table below even for the L40S, which
is a second sign they were derived from a bound rather than read off a run.

**Nothing about the shape failed, and nothing here would recur on a long run.** 137 is
128 + SIGKILL, which is what Batch terminating a job looks like from inside the container; it
is not the program's own code and it is not a stage of the platform. There was no OOM, no host
loss and no timeout. `RETRY_ONLY_WHAT_A_RETRY_FIXES` never came into it either — a cancellation
does not reach the retry rules at all, which is why the second attempt the submission had paid
a ceiling for was never taken. Of the three failure modes worth fearing on a new shape, an OOM
is the one this run positively rules out at 58% of the card, a lost host is the one rule one
does retry, and a timeout is the one that only retries by recording no exit code — and none of
them is what happened.

**What the 74 steps do not establish, stated because it is the residual risk on this shape.**
`--save-interval 100` was never reached, so this branch has never written a distributed
checkpoint from eight ranks and has never resumed one, and stage 4 reads a `step6000` checkpoint
back on a single L4. The evaluator, which is the other thing that had to work at a new world
size, did run: it fired at step 50 and the whole step took 39.63 s against the L40S's 104 s for
an evaluation alone.

**2.91 s/step, over 72 rows.** Steps 2–49 and 51–74, which is every step except step 0, step 1
and step 50. Step 1 pays for `torch.compile`; step 50 is where `--monitor-interval 50` and
`--eval-interval 50` fire together and it reads **39.63 s**, 13.6 times a clean step. The
interquartile spread over the 72 is **0.0072 s**, and the flush-group wall clock agrees at
14.55 s per five steps, which is 2.910. Median device TPS is 33,803, so the shape's total is
270,426 tokens per second against the L40S's 76,204.

**Which statistic you take matters more than which rows you drop, and that is worth being
precise about.** With step 50 left in, the *median* over all 73 rows is 2.9082 — it moves by a
ten-thousandth, because a median does not care how large one outlier is. The *mean* over the
same 73 is **3.4164**, which is 17% high and would have priced a full-horizon arm four hours
long. So the clean filter is not what protects a median here; what protects it is that only one
step in this run is instrumented. The failure that cost this project a tranche was worse than
either: filtering *to* the rows carrying `hc/*` keys keeps the dirty steps and nothing else, and
on this run that is step 50 by itself, reporting **39.63 s/step**.

**The A100 is 3.55× the throughput at 2.09× the price, so it is 41% cheaper per token.** That
is not what this branch expected. The probe was justified on a compute ratio of 1.72× — 2,496
TF against 1,448 — and it was pre-committed to a threshold of 5.76 s/step, which is where the
full 12,715-step horizon fits a 24-hour attempt with 10% spare. The measurement beats that
threshold by a factor of two. At `cost.hourly_rate_usd` of 10.4926 for `gpu-4xl40s` and 21.9576
for `gpu-8xa100`, read from `edullm check --json` on 2026-08-08 and worth re-reading rather
than quoting from here:

| shape | s/step | $/hour | $ per 1M tokens |
| --- | --- | --- | --- |
| `gpu-4xl40s` | 10.32 | 10.4926 | 0.0382 |
| `gpu-8xa100` | 2.91 | 21.9576 | **0.0226** |

Per-token parity would be at 4.93 s/step; the measurement is 2.91. **So the two shapes do not
price within a rounding error of each other and any claim that they do should be read against
this table.** They would have, at the 5.00 s/step figure `run.yaml` nominated as its budget
threshold — 1.4% apart — and that is the likeliest origin of a parity claim, but it was a
threshold and not a measurement.

**The tranche did not move to A100 anyway, and the reason is not price.** Three things, in the
order they bind.

1. **Horizon is the wrong thing to buy, whatever it costs.** DataDecide puts seed σ at
   D^−0.172, so at a fixed budget the standard error of an arm mean goes as D^+0.328 and every
   dollar moved from replicates into tokens makes the experiment less sensitive. A cheaper
   shape does not change that exponent. It means the *savings* should buy seeds, which is what
   the five-seed stages already do — it does not mean the 10B horizon is now worth buying.
2. **The measurement arrived after stage 1 was admitted.** The baseline fan-out went out at
   17:42 UTC and this probe's last row is from before that, but its median was not read until
   afterwards. Moving stage 2 to a different shape from stage 1 would confound the arm contrast
   with the machine, which is the one thing no budget saving is worth.
3. **`gpu-8xa100` is $21.96 an hour, which is over the $20 admin threshold**, so it needs an
   admin at any duration against a lead for `gpu-4xl40s`, and its median queue is 89 minutes
   against 19.

**Reason 3 is stale and reason 2 is no longer the whole of it.** `edullm check --json` on
2026-08-08 returns `approval_class: routine` and `approving_environment: run-approval-lead` for
a five-cell fan-out on `gpu-8xa100`: policy v5 removed the $20 rate ceiling deliberately, on the
argument that a rate cannot rank two requests by what they commit and a worst-case total can, so
both shapes go to a lead. The same output puts the A100 queue at a **61-minute** median against
the L40S's 19, with a worst observed 12.6 hours against 6.4 — and, which matters more than
either median, fourteen A100 nodes arrived over three days with nothing ever cancelled for want
of capacity, against six L40S nodes of which two of ten queued runs were cancelled by hand while
stuck in `RUNNABLE`. Reason 2 stands but has to be argued rather than asserted, and it is, in
[The shape for stages 2, 3 and 4](#the-shape-for-stages-2-3-and-4).

Recorded because it is worth money to the next module rather than to this one. A 370M-class run
on this corpus is 41% cheaper per token and 3.55× faster per step on the NVLink shape, and the
wire hypothesis in `run.yaml` — that the L40S step is bound by PCIe and not by the card — is
confirmed well past what its own arithmetic predicted: 1.72× of compute bought 3.55× of step
time, and the excess is the interconnect. Track B, and any second tranche that wants the 10B
horizon, should start from this row rather than re-derive it.

The rehearsal figure was never a prediction for 370M: it is a 96M model of which 77M is
embedding and unembedding, so it spends an unusually large share of its time in two matmuls
that do not grow with depth. The probe row is the one to plan against, and it is 4.5 times
slower per token.

### The shape for stages 2, 3 and 4

**Stages 2 and 3 stay on `gpu-4xl40s`. Stage 4 is on `gpu-1xl4` and this does not touch it.**
The A100 is 41% cheaper per token and that is a real number, correctly measured, and it is not
enough. What follows is the arithmetic rather than the assertion, because the decision is close
and the case against it is not the one the record has been making.

**What the saving is worth, on the axis that actually gates a submission.** Two numbers price a
tranche and they behave differently. The *bill* is wall clock — `run_costs._attempt_seconds`
times the rate — and there the A100 keeps its whole advantage. The *ceiling* is what admission
prices and what an approver reads, `rate × nodes × hours × attempts × cells`, and there it does
not, because the bound has to hold the run with margin whatever the run costs, and a 5.50-hour
cell under a 7-hour bound wastes 21% of its ceiling where a 17.85-hour cell under a 19-hour
bound wastes 6%.

| | `gpu-4xl40s` | `gpu-8xa100` | A100 |
| --- | --- | --- | --- |
| s/step, `faithful`, measured | 10.32 | 2.91 | 3.55× faster |
| $ per 1M tokens, steps only | 0.0382 | 0.0226 | **41% cheaper** |
| hours per 6,000-step cell, `arm_seconds` | 17.85 | 5.50 | 3.25× faster |
| bill per cell | $187.28 | $120.74 | 36% cheaper |
| ceiling per cell, 2 attempts | $398.72 (19 h) | $307.41 (7 h) | **23% cheaper** |

So **41% per token is 23% per approval**, and 23% is what the budget sees. In seeds, for the
four funded arms at 4.72B tokens each:

| budget, as ceiling | `gpu-4xl40s` | `gpu-8xa100` |
| --- | --- | --- |
| $4,000 | 10 cells — **2 seeds** an arm | 13 cells — **3 seeds** an arm |
| $8,000 | 20 cells — **5 seeds** an arm | 26 cells — **6 seeds** an arm |

Two things fall out of that table and the first one is not about shapes at all. **Four arms at
five seeds has not fitted $4,000 since `mhc` was funded**: it is $7,974 of L40S ceiling, of
which stage 1 has already committed $1,993.59, and the grant that bought the fourth arm is what
the design is actually running on. At that grant the A100 buys **one more seed per arm**, which
by `noise_floor.mde` at four arms takes the unpaired MDE from **0.0189 nats to 0.0170** — a 10%
improvement, real, and the right kind of thing to buy under D^(+0.328). At $4,000 it buys
0.0376 against 0.0261, which is the difference between a design the analysis plan calls
inadequate and a design the analysis plan calls inadequate: both are above the 0.020 literature
effect this module exists to detect.

**Now the confound, which is the crux, and which is not a numerics problem.** Running stages 2
and 3 on eight ranks at microbatch 12,288 while stage 1 ran on four at 16,384 changes four
things, and three of them are rounding.

- *The global batch is identical.* `NumpyFSLDataLoader` shuffles `np.arange` with
  `seed + epoch` and nothing else, truncates to a multiple of the global batch, reshapes, and
  only then shards with `indices[:, dp_rank :: dp_world_size]` — so both world sizes see the
  same documents in the same global batch at every step, partitioned differently.
- *Gradient accumulation is split-invariant in exact arithmetic.* Microbatches run with
  `loss_reduction="sum"` against one shared `loss_div_factor`, the rank's own non-ignore token
  count, so 8 microbatches and 12 microbatches give the same gradient up to summation order. The
  FSDP reduce is fp32 — `reduce_dtype=DType.float32`, set explicitly in `train_on_corpus.py` and
  also the library default — so the cross-rank average is not a bf16 accumulation over 8 terms
  against 4.
- *The initial draw does not move.* `_apply_init` materializes the full tensor, initializes it,
  and copies out the local shard, so the weights a seed draws are world-size independent and
  only their layout changes.
- *bf16 is bf16.* A100 is sm_80 and L40S is sm_89; both accumulate bf16 matmuls in fp32 in the
  tensor cores. Different kernels get selected and different split-K reductions get taken, which
  moves results at the 1e-3 relative level per op, in no particular direction.

All of that is rounding, it is mean-zero, and — this is the part worth saying out loud — **it is
already inside the quantity the design measures.** The pre-registration states that what survives
the seed pairing is "initialization plus kernel non-determinism", so σ̂ is an estimate of a spread
that already contains kernel-level irreproducibility. A shape change adds more of the same kind
of thing to a term the noise floor is measuring anyway.

**One difference is systematic, and it is in the instrument rather than in the model.**
`LMEvaluator` runs on a `NumpyFSLDataLoader`, which drops the tail so the instance count divides
the batch, and the evaluator's global batch is `rank_microbatch_size × dp_world_size`. That is
**16 padded documents per eval batch on four L40S ranks and 24 on eight A100 ranks.** Unless a
held-out source holds a multiple of 48 documents, the two shapes **score a different set of
documents**, deterministically, for the whole run. Within a shape it cancels exactly and no
contrast can see it; across shapes it is a fixed offset. The size is probably small — dropping
k tail documents out of N shifts a token-weighted mean by about σ_doc·√k⁄N, which at a few
thousand documents a source and a few tenths of a nat of per-document spread is order 0.001
nats, well under a tenth of the 0.016 gate — and `--preflight` builds the held-out set and
prints it, so the document counts are checkable on a laptop for free. But it is systematic, it
lands wholly inside **H1**, which is the module's replication claim, and no cell in the design
can bound it, because arm and machine would be perfectly aliased.

**Which leaves three options, and the honest arithmetic kills the cheap one and prices the clean
one at par.**

| | ceiling | bill | wall clock | H1 |
| --- | --- | --- | --- | --- |
| (a) 15 remaining cells on L40S | $5,981 | $2,809 | ~55 h over a 6-node pool | clean |
| (b) 15 remaining cells on A100 | $4,611 | $1,811 | ~13 h over a 14-node pool | **aliased with the machine** |
| (c) those 15 plus a 5-cell A100 baseline | $6,148 | $2,415 | ~13 h | clean |

Option (c) is the one to take seriously, and it is the one that shows what the saving is really
worth: making the A100 move *scientifically clean* costs a baseline re-run, and the re-run eats
the saving. Against (a) it is **$167 more ceiling and $394 less bill** — par, inside the error on
any of these figures — in exchange for about forty hours of wall clock and a deeper capacity
pool. Its five A100 baseline cells are priced here at the `faithful` step time, which is
conservative by roughly the 26% the row above measures. And it is not obviously wrong: forty
hours is forty hours, and 14 nodes that never starved is a better place to put 15 cells than
6 nodes that have already stranded two runs in `RUNNABLE`.

**What decides it against (c) is the noise floor, and it is a pre-registration argument rather
than a cost one.** Stage 2 is gated on σ̂ and the per-source inverse-variance weights being
frozen from the baseline *before any treatment arm exists*, and that gate is the entire reason
stage 1 went out on its own. Under (c) there would be two baselines: the L40S one, which lands
in about twelve hours, and an A100 one, which lands later. Whichever gets frozen is then a choice
made with both visible — and pre-committing now to freeze the A100 one is a commitment to
discard a measurement already in hand in favour of one that does not exist yet, justified by the
shape being right, which is the thing under decision. That is circular, and it re-opens by hand
the one degree of freedom this design spent a whole stage and $1,993.59 of ceiling closing.
Option (b) does not have that problem and has the worse one. Option (a) has neither.

So: **10% off the MDE, or a clean freeze and a clean H1.** The freeze wins, and it is not close
once the saving is priced at 23% rather than 41%.

**The running L40S baseline is untouched.** It stays the H1 and H2a comparator, `noise_floor.py
--freeze` reads it as planned, and stages 2 and 3 go out on `gpu-4xl40s` at
`--rank-microbatch-size 16384`, `--nproc-per-node=4` and `--hours 19` — identical to stage 1 in
everything but `--arm`, which is what
`test_the_three_stage_specs_differ_in_the_arm_and_in_nothing_else` exists to hold. Nothing is
re-run, nothing is discarded, and nothing is kept as a cross-shape check, because under this
recommendation there is no second shape to check against.

**Two conditions flip it, and both are worth watching for rather than arguing about.**

1. **Stage 1 loses a cell.** If a baseline cell is cancelled in `RUNNABLE`, lost to the 19-hour
   bound, or has to be re-run for any reason, the noise floor has to be re-measured anyway,
   there is no incumbent estimate to discard, and the circularity above evaporates. At that
   point take option (c) whole: all 20 cells on `gpu-8xa100` at 12,288 and eight ranks, freeze
   from the A100 baseline, and keep whatever L40S cells completed as a cross-shape measurement
   of exactly the offset this section could only bound by argument. That is the better
   experiment and the only thing standing in front of it is that stage 1 is currently fine.
2. **The budget is re-based on the bill, or the runtime bound can be lowered per submission.**
   The 41%-to-23% dilution is entirely an artifact of the ceiling pricing a bound that a fast
   run does not use. Price on wall clock and the A100 saving is $1,331 on a four-arm five-seed
   tranche, which is about two more seeds an arm rather than one — 0.0189 nats to 0.0156, a 17%
   improvement — and two seeds is worth re-opening the freeze for in a way that one is not.

A third thing would not flip it but would change what (c) costs: **nothing in this branch has
written a distributed checkpoint from eight ranks**, because the probe was cancelled 26 steps
before `--save-interval 100`. Any move to A100 should find that out in the first ten minutes of
the first cell rather than at step 6,000 of fifteen of them.

### The capacity stall of 2026-08-08, and what it does to the section above

**Condition 1 of the two that flip the section above has not been met, and something the
section did not weigh has happened instead: the L40S pool cannot deliver this tranche on any
schedule the module can use.** The recommendation here is option (c) whole — all 20 cells on
`gpu-8xa100`, freeze from the A100 baseline — and the argument is capacity rather than price.
Read at 19:50 UTC.

**What AWS says, rather than what the parent's state implies.** `edullm status` was spent on
all four run ids. Every one of them reports the same thing:

| submission | arm | queued | Batch status | attempts | cells logging in W&B |
| --- | --- | --- | --- | --- | --- |
| `run_019fe279-4ef0` | `baseline` | 17:45:45Z | `PENDING` | 0 of 2 | 3 of 5, step ~840 |
| `run_019fe2c2-afa8` | `faithful` | 19:07:02Z | `PENDING` | 0 of 2 | 0 of 5 |
| `run_019fe2c2-bc91` | `output-only` | 19:07:36Z | `PENDING` | 0 of 2 | 0 of 5 |
| `run_019fe2c2-f498` | `mhc` | 19:07:50Z | `PENDING` | 0 of 2 | 0 of 5 |

**Two things in that table are instrument rather than fact, and both are worth writing down
before anybody reads a decision off it.**

- **`PENDING` is the array parent and not a cell.** The baseline parent reads `PENDING` with
  `0 of 2` attempts and a log stream "not yet assigned" while three of its children have been
  training for two hours. `cancel-run.yml` reports `describe_jobs` on the parent id and never
  reaches `arrayProperties.statusSummary`, so **the authorised read-only surface cannot
  distinguish a `RUNNABLE` child from a `PENDING` one**, and no amount of spending on it will.
  W&B is the only per-cell instrument this project has, which is what `tranche_watch.py` was
  built for and is the whole of the evidence in the last column.
- **`edullm status run_019fe2c2-bc91` prints `CANCELLED` and no compute was cancelled.** The
  submission workflow's final job — "Index the run id against the workflow run that minted it"
  — was cancelled, which makes GitHub call the whole run `cancelled`, which is what the CLI
  reads. Every job before it succeeded, including "Start the admission execution", "Wait for
  the admission decision" and "Say where this run went", and AWS holds a Batch job for it
  queued at 19:07:36Z. So the `output-only` stage is admitted, is in the queue, and reads as
  cancelled. Anything that keys off that word — a watcher, a resubmission, an audit of what
  the tranche spent — is wrong about this stage in whichever direction it guesses.

**What is not an instrument artifact: 3 of 20 cells are running, and 17 have never started.**
Baseline seeds 2 and 4 have waited **2h04m** against a 19-minute median and a 6.4-hour worst
observed. The fifteen treatment cells have waited **43 minutes**. Nothing has started since
17:47.

**The projection, and the pool is the ceiling rather than the queue.** Remaining work is
3 × 12.2 hours on the running cells, 2 × 14.36 on the stranded baseline pair, and 15 × 17.85
on the treatments: **333 node-hours**. The shape has **six nodes** and other teams are on them
(`olmoe-specdec-baseline-l40s`, several `p3-evals-*`).

| concurrency | wall clock to 20 cells |
| --- | --- |
| 3, the observed steady state | **111 hours — 4.6 days** |
| 4 | 83 hours — 3.5 days |
| 6, the entire pool with no other team on it | 55 hours — 2.3 days |

**2.3 days is the floor and it assumes something that has never happened.** The honest range
is **3.5 to 7 days**, centred near 4.5, and the upper end is not invented: the platform's own
capacity line for this shape says two of ten queued runs were cancelled by hand while stuck in
`RUNNABLE`, one at a hundred minutes, so a fifth of what queues here has historically never
run at all.

**The same 20 cells on `gpu-8xa100` are 110 node-hours.** At the probe's clean 2.91 s/step a
lane cell is 5.50 hours and a baseline cell 5.45, and 2.91 is the `faithful` figure so the
baseline five are priced 21% conservative. Fourteen nodes arrived over three days, thirteen
runs have queued on them, **none was ever cancelled for want of capacity**, and the probe got
a machine four minutes after admission. At six concurrent that is 18 hours and at ten it is
11. **Call it 8 to 20 hours against 3.5 to 7 days.**

#### `frontload-cl` is direct evidence, and it is stronger than the probe

Another team ran this shape yesterday at what is materially this model. `run_019fddc9-3aaa`
(`control`) and `run_019fddc9-44a6` (`primer`), W&B group `frontload-cl`, both **finished
12,715 steps** on **8 × A100-SXM4-40GB**:

| | `frontload-cl` control | this module |
| --- | --- | --- |
| `d_model` / `n_layers` / `n_heads` | 1024 / 16 / 16 | 1024 / 16 / 16 |
| feed-forward hidden, sequence length | 4096, 4096 | 4096, 4096 |
| global batch | 786,432 | 786,432 |
| wall clock | **23,109 s for 12,715 steps — 1.82 s/step** | 2.91 s/step measured |

**Do not plan against 1.82.** They run `attn_backend: flash_2`, full activation
checkpointing, HSDP and a rank microbatch of 98,304 — one microbatch a rank. Every one of
those is a numerics or a memory change this tranche has ruled out mid-experiment, and
`PLATFORM_ATTN_BACKEND` pins every arm to torch SDPA. What the row is evidence of is the two
things that are not numerics: **the shape places, twice in one evening for six and a half
hours each**, and 10B tokens of this model fits it comfortably.

**It also retires most of the checkpoint risk.** Their `checkpointer` ran at
`save_interval: 1000` with `save_async: true` and `load_strategy: if_available`, so
**distributed checkpoints from eight ranks were written on this shape, at this model size, to
this bucket**, twenty-six times without incident. What that does not cover is stage 4, and see
below.

#### Pre-committing the freeze now is not the circular move the section above refused

The objection was precise and it is worth keeping precise: under (c) there are two baselines,
"whichever gets frozen is then a choice made with both visible", and pre-committing to the
A100 one is "a commitment to discard a measurement already in hand in favour of one that does
not exist yet". Three things have changed and each weakens a different clause.

1. **Neither measurement is in hand and neither is visible.** σ̂ and the per-source
   inverse-variance weights are read off finished 6,000-step cells. The L40S baseline is at
   step 840 of 6,000 on three of five seeds. There is nothing to discard yet, and a rule
   written now is written before either number exists — which is what a pre-registration is.
2. **The L40S baseline cannot become the thing the freeze needs.** It has three live cells and
   two that have never started, so at best it is **df = 2**, whose 95% interval on a variance
   estimate spans a factor of **12.1** — the "rumour of a noise floor" this stage exists to
   escape. The whole reason stage 1 went out alone was to buy df = 4. **The pool has already
   taken that away**, and the candidate the circularity worried about is not a candidate.
3. **The confound the section priced is gone rather than moved.** Under (c) whole, all four
   arms and the baseline are on one shape at one world size, so the 16-against-24 padded
   documents per eval batch cancels exactly, as it does within any shape. It is only the split
   (option (b)) that aliases arm with machine, and that option is not being taken.

So the rule, and it is committed here **before any A100 cell has been submitted**: **the A100
baseline is the primary comparator for H1, H2a and H5 and the source of the frozen σ̂ and the
frozen per-source weights. Whatever the three running L40S baseline cells reach before they
are cancelled is a secondary cross-shape check and enters no gate.** That inverts nothing
later; it is the last sentence written before the submissions go out.

#### What makes this hard to reverse, and the ten-dollar thing that should precede it

- **Stage 4 reads a step-6000 checkpoint back on one L4, and nothing has done that from an
  eight-rank write on this branch.** The library says it can:
  `load_model_and_optim_state` is documented "agnostic to the distributed topology in that it
  can load checkpoints saved with a different distributed topology", it goes through
  `dist_cp.state_dict_loader.load`, and `frontload-cl` proves the write half. The read half at
  world size 1 is still unexercised **here**, and finding out at step 6,000 of twenty cells is
  the expensive way. **Run one baseline cell to step 100 with `--save-interval 100`, then point
  `score_checkpoints.py` at what it wrote from `gpu-1xl4`.** That is about $10 of ceiling and
  half an hour, and it is the only pre-flight this move needs.
- **Cancelling the three running baseline cells discards about 6 node-hours, which is $63.**
  Their step-500 checkpoints stay in S3 under the old run id and are not resumable into a new
  submission, so this is a real discard and a small one.
- **`--rank-microbatch-size` stays at 12,288 and is not a knob.** It is the probe's own value,
  ~28.9 GiB of a 40 GB card; 16,384 is ~35.7 GiB, and an OOM is not retried.
- **The pre-commitment above is the one-way door.** It is worth more than the compute it
  governs and it is worth nothing at all if it is revisited after a number appears.

#### The four A100 specs

`run.baseline-a100.yaml`, `run.faithful-a100.yaml`, `run.output-only-a100.yaml` and
`run.mhc-a100.yaml`. The four L40S stage specs are **not edited**, because submissions were
admitted from them. Each A100 spec differs from its L40S counterpart in `--nproc-per-node`
(4 → 8), `--rank-microbatch-size` (16,384 → 12,288) and `suggested_compute`, and in nothing
else: parsing all eight commands through `train_hyper_connections.build_parser` returns
identical option sets whose only differing value is `rank_microbatch_size`. The baseline spec
writes out the five options its L40S predecessor left to defaults — the literals
`STAGE_PINNED` already holds — so the four stages cannot drift apart across the commits
between them.

`edullm check` on each returns `approval_class: routine`, `run-approval-lead`, and
**$1,537.03** of ceiling at `--hours 7 --attempts 2 --fanout-size 5`: **$6,148.13** for the
tranche, against $7,974 of L40S ceiling for the same twenty cells, and an expected bill near
**$2,415** against $3,563.

### What a full arm actually costs, and why it cannot be submitted as one run

`hyper_connection_arms.arm_seconds` builds a run out of the measured constants — 10.32 s a
step, 104 s an evaluation, 46 s a checkpoint, 118 s of start-up and shutdown, and 1.37 s on
each step the lane monitor fires on — and a test checks that rebuilding the probe's own shape
out of them lands on the probe's own wall clock.

| horizon | tokens | hours per run |
| --- | --- | --- |
| 12,715 steps (10B, what the experiment wants) | 10.0B | **37.7** |
| 6,000 steps (what the tranche runs) | 4.72B | **17.9** |

**37.7 hours does not fit and cannot be made to.** `olmo-core-train` declares
`maximum_runtime_hours: 24`, and `--hours` only lowers it: an override above the workload bound
is refused with `runtime_above_the_workload_bound`, whose detail says raising it for everybody
is a pull request against the platform's `config/workload-catalog.yaml`. The 54.9 hours and
$580 that stood here were the same fact through the wrong step time; the conclusion survives
the correction, which is the only reason that paragraph is not simply deleted.

**The two-attempt plan does not survive reading the retry rules, and this is the finding that
set the step count.** The resume is sound — see
[Resuming across attempts](#resuming-across-attempts-what-holds-and-what-does-not) — but the
attempt has to be granted, and `RETRY_ONLY_WHAT_A_RETRY_FIXES` in the platform's
`execution.py` is, in the order Batch reads them:

```
OnStatusReason "Host EC2*"     RETRY
OnReason       "OutOfMemoryError*"  EXIT
OnExitCode     "*"                  EXIT
```

A timed-out attempt carries `Job attempt duration exceeded timeout`, which is not `Host EC2*`,
so it reaches a second attempt only by recording no container exit code at all — nothing
matches, and Batch's documented fall-through retries. The platform's own catalog says exactly
that, citing an observed timeout whose `container_exit_code` was null. **But an attempt that
records any exit code falls to rule three and is not retried**, and torchrun is a program that
exits non-zero on SIGTERM: its elastic agent raises `SignalException`, forwards the signal to
the four ranks, waits 30 s, kills them and re-raises, and the launcher lets it out. That is a
dead heat with the container stop timeout. Losing it costs the arm at hour 21, at step ~7,000,
having paid for the whole ceiling. Nine cells makes it nine coin flips.

**So the tranche fits one attempt.** 6,000 steps is 17.9 hours against a 21-hour bound, which
is 15% of margin for step-time drift over eighteen hours. The second attempt is still asked
for and still worth its ceiling: it covers the one thing rule one *does* retry, a host going
away, and the resume then does exactly what it is supposed to.

**The cost, both numbers, because they are different and both bind.** Expected spend is 160.6
node-hours for the nine runs, which `estimated_cost_usd` will multiply by whatever rate
`edullm check --json` reports. What gets *approved* is a ceiling: attempts × the runtime bound
× the rate × the cell count, reported as `cost.maximum_compute_cost_usd`. At 24 hours that
ceiling is $4,532.79 for nine cells and over budget; at `--hours 21` it is $3,966.21 and under.
Read both out of `check --json` rather than out of this paragraph.

The `gpu-8xa100` figures that once stood here — about 6 hours and $135 — were removed for having
no measurement under them, and the honest next step was called as a 100-step probe there for a
few dollars rather than a $4,000 tranche. **That probe ran, and it says the reach constraint is
an L40S constraint and not a physical one.** At the measured 2.91 s/step the full 12,715-step
arm is **11.5 hours and about $253 a cell** on `gpu-8xa100`, against 37.7 hours and $395 on
`gpu-4xl40s` — comfortably inside the 24-hour workload bound, with no second attempt needed and
none of the `RETRY_ONLY_WHAT_A_RETRY_FIXES` reasoning above coming into play at all. The removed
figures were low by about half and were right about the conclusion.

This changes nothing for this tranche and everything for the sentence that says the 10B horizon
"needs a second tranche as well". The horizon is still the wrong purchase — D^(+0.328) is not a
statement about what a machine costs — but it is no longer *unreachable*, and the deferral
should be read as a choice rather than as the platform's ceiling. See
[The shape for stages 2, 3 and 4](#the-shape-for-stages-2-3-and-4) for why the money it would
cost still buys replicates instead.

### Resuming across attempts: what holds, and what does not

Read out of the code rather than assumed, because the whole two-attempt argument rested on it.

**The resume itself holds.** `Trainer.fit` calls
`maybe_load_checkpoint(self.save_folder, load_trainer_state=True, load_optim_state=True)`
before anything else and only then falls back to `load_path`
(`src/olmo_core/train/trainer.py:694-713`). `load_state_dict` restores the data loader's
position, `global_step`, `global_train_tokens_seen`, `global_train_petaflops`, the epoch, every
callback's state and the per-rank RNG (`trainer.py:836-881`), so step, token count, data order,
optimizer state and shuffle all come back. `.edullm/train_on_corpus.py:819-824` clears torn
step directories and calls `maybe_load_checkpoint()` before `fit()` for the reason
`remove_torn_checkpoints` gives. Under a fan-out each cell has its own checkpoint prefix —
the platform's `FANOUT_PROLOGUE` re-exports `EDULLM_CHECKPOINT_DIR` as
`…/cell-$AWS_BATCH_JOB_ARRAY_INDEX/checkpoints/` — and the array index is stable across
attempts of the same child, so a retry resumes its own seed and not a sibling's.

**The learning-rate trap is real and is not handled for you.** `Trainer.state_dict` writes
`max_steps` into the checkpoint (`trainer.py:828`) and `load_state_dict` never reads it back —
the key is simply absent from `trainer.py:836-881`. `max_steps` is a property recomputed from
`max_duration` (`trainer.py:476-481`), which `.edullm/train_on_corpus.py:653` sets from
`--steps` at construction, and `Scheduler.set_lr` reads `trainer.max_steps` live on every step
(`src/olmo_core/optim/scheduler.py:68-77`). **So every attempt must be launched with the same
`--steps`.** A second attempt given a smaller number does not resume a cosine, it starts a new
one from wherever the first left off and collapses the remaining decay into that segment — and
because the loss curve continues smoothly it looks like a run that finished. The command lives
in `.edullm/run.yaml` and a Batch retry re-runs the identical container, so this is safe as
long as nobody edits the file between attempts, which is worth knowing before somebody does.

**The clean-exit race is lost in OLMo-core and won by torchrun, which is not the same as being
safe.** `Trainer._handle_os_signal` turns SIGTERM into `cancel_run` (`trainer.py:1267-1283`),
which sets a cancelling rank but no `_error`; `_check_if_canceled` broadcasts
`(reason, self._error is not None)` so `_canceled_by_error` stays false
(`trainer.py:1435-1444`); and `fit` raises only when that flag is set (`trainer.py:770-776`),
otherwise running `post_train`, `_shutdown` and returning normally. `train_on_corpus.cli` then
returns 0. **A trainer that catches Batch's SIGTERM exits successfully.** What stops that
reaching Batch is the launcher: the elastic agent re-raises `SignalException` after shutting
the workers down (`torch/distributed/elastic/agent/server/api.py:738-742`), `launch_agent`
re-raises it again, and torchrun exits non-zero. The trainer would not have finished in time
anyway — `cancel_check_interval` is 5, so it needs up to five steps (52 s) just to notice, and
then a synchronous 46-second checkpoint, against a 30-second grace period.

Both halves of that are worth stating because they point opposite ways. Under `torchrun` the
container exits non-zero, so nothing is ever mistaken for success — but that same non-zero exit
is what makes rule three fire and the timeout *not* retry. And a future entry point that ran
`train_hyper_connections.py` directly on one device, without the launcher, would exit 0 on a
cancellation and report a half-finished run as a completed one.

### Noise floor

Still not measured. See above — it needs arm 1's three seeds.

Three notes on reading throughput numbers here. OLMo-core v2.5.0 fixed an A100 peak-FLOPs
constant in `SpeedMonitorCallback` that was 2× too low and had been inflating reported MFU by
2×, so any figure from before that is wrong by a factor of two; this branch is on v2.5.0. The
L40S constant this branch itself added had a second factor of two in it, the other way — 362.05
is the FP16-accumulate rate and torch accumulates in FP32 — so every L40S MFU written before
2026-08-08 evening is understated by two and the corrected peak is 181.03 TF. And
`num_flops_per_token` here counts the hyper-connection cost, so MFU across arms is comparable.

**`PLATFORM_ATTN_BACKEND` is still `"torch"`.** The image now installs flash-attention 2 from a
prebuilt wheel, but `hyper_connection_arms` pins every arm to torch SDPA, so the wheel is inert
and the 15.55 s/step above is an SDPA measurement. Flipping the constant is the most likely way
to get under the 24-hour ceiling and it is also a change to the numerics of every arm, so it
gets a probe of its own at 370M — a throughput reading and a loss curve against the SDPA probe
— before it is flipped, and it is flipped before arm 1 rather than between arms.

### Compute reality

`gpu-4xl40s` is lead-approved at this duration and cost, with a 19-minute median queue.
`gpu-8xa100` is cheaper per run but its hourly rate is above the $20 admin threshold, so it
needs an admin at any duration and the median wait is 89 minutes. A fan-out never self-releases
regardless of cost, so submitting the seeds or the treatment arms as an array will hit the gate.
`gpu-8xb200` is priced but capacity-block-backed with nothing purchased: a submission is
admitted and then dies at Batch on a queue that does not exist. Not an option.

Two caveats on that paragraph. "Lead-approved at this duration and cost" was written against
the 21 hours that turn out to be 55, and 55 is over the platform's runtime ceiling, so the
approval class an arm falls into has not been established at the real duration and will not be
until a submission clears `check` — read `approval_class` out of `check --json` rather than out
of this document. The 19-minute median queue is the one figure here that `check --json` still
confirms, alongside a worst observed wait of 6.4 hours. Nothing on `gpu-8xa100` has been
re-derived, because nothing in this branch has run there.

**A third caveat, which retires most of the second.** Something has now run there —
`run_019fe262-778d`, 74 steps — and `check --json` has been read against that shape. **The $20
admin threshold no longer exists.** Policy v5 removed it deliberately, on the argument that an
hourly rate cannot rank two requests by what they commit and a worst-case total can, and
`classify_request` cannot return `EXCEPTION` at all under v5 or v6; `run-approval-admin` is
reserved for capacity blocks, which nothing has designed. A five-cell fan-out on `gpu-8xa100`
comes back `approval_class: routine`, `approving_environment: run-approval-lead` — the same gate
as the L40S, which is also what released the probe. **The 89-minute queue figure is stale too**:
`check` now reports a 61-minute median over thirteen runs on that shape, worst observed 12.6
hours, fourteen nodes arrived over three days and none of the thirteen was ever cancelled for
want of capacity. Against the L40S's own line — six nodes, ten queued runs, eight started at a
19-minute median, and two cancelled by hand while stuck in `RUNNABLE`, one at a hundred minutes
— the A100 is the slower queue and the deeper pool, and only the first of those two was ever
written down here. Read both out of `check --json` rather than out of this paragraph.

**What `check --json` reported on 2026-08-08** for a three-cell fan-out of this repository on
`gpu-4xl40s`: `approval_class: routine`, `approving_environment: run-approval-lead`,
`maximum_attempts: 2`, `maximum_runtime_hours: 24`. A fan-out never self-releases whatever it
totals, which the platform's own catalog states, so all three submissions go to a lead
regardless. Re-read it rather than quoting this.

## Stage 1's health gate, and the three things checking it turned up

`.edullm/stage_gate.py` is the gate. It is a third sibling to `wandb_panels.py`, which asks
whether a metric *key* arrived, and `noise_floor.py`, which asks what the numbers in those keys
are worth once a run has finished. This one asks the only question that has to be answered while
the cells are still going — whether the configuration is sound and whether it fits its bound —
and it answers `go`, `no-go`, or `too early`, which is a third answer and not a soft `no-go`.

```bash
python .edullm/stage_gate.py --self-test                                   # no network
python .edullm/stage_gate.py --run <full run id> --cells 5 --watch 420
```

Read at step ~750 of 6,000, `run_019fe279-4ef0` on `gpu-4xl40s` passed every check the gate
makes: three live cells at seeds 0, 1 and 3 whose losses differ at every shared step, clean
medians of 8.211, 8.076 and 8.259 s/step over 750, 760 and 740 rows, 14.37 projected hours
against the 19-hour bound, z-loss written on all three, and seven per-source metrics with
bits-per-byte beside every cross-entropy at both the startup evaluation and the one at step 500.
It did **not** report `go`, because two of the five cells had logged nothing.

**It never did report `go`, and the reason is not any of the things it was watching for.** The
SIGTERM all three live cells took at 19:58Z was a deliberate `edullm cancel` landing on the
submission, and the array parent reads `FAILED` because a cancelled array child is how Batch
reports one. Nothing failed and nothing was starved. The reason for the cancellation was the
capacity finding recorded above — six L40S nodes sustaining three of twenty submitted cells for
over two hours, projecting the tranche at 3.5 to 7 days — and the whole tranche moved to
`gpu-8xa100`.

So **stage 1 on the L40S measured no noise floor.** σ̂ at the final step is what stage 2 is gated
on and the run stopped at 15% of the horizon, which is why its numbers are all instrument
readings — step time, MFU, metric coverage — and none of them is a variance. What it does
establish is that the configuration is sound, and none of that had to be re-established on the
new shape *as configuration*. Every *measurement* did, which is the next section.

## The A100 baseline, and the second factor of two in the MFU

`run_019fe2f4-f528` on `gpu-8xa100` is the baseline resubmitted: five of five cells, same commit
lineage and same pinned flags, differing from the L40S command only in `--nproc-per-node` (4 → 8)
and `--rank-microbatch-size` (16,384 → 12,288). Read at steps 659 to 1,179 of 6,000, it passes
every check the gate makes and reports **`go`**:

| check | reading |
| --- | --- |
| seeds | 0–4 from the cells' own configs, five distinct losses at the shared step 659 spanning 0.1893 nats |
| arm | 5/5 consistent with `baseline`, unambiguously |
| loss | 11.7 → 2.74–2.92, decreasing on all five, no non-finite value |
| z-loss | written by 5/5, so the configured 1e-5 is in force |
| held-out | seven sources on all five cells, BPB beside CE on all seven |
| throughput | median 1.700 s/step, slowest cell 1.728 over 865 clean rows |
| fit | 3.18 h projected against the 7-hour bound, 55% spare |
| MFU | 55.28–56.85% by hand, 55.35–56.87% reported |

The clean medians are 1.681, 1.700, 1.692, 1.728 and 1.712 s over 1,173, 1,163, 700, 865 and 655
rows, at an interquartile spread of 0.003–0.005 s. What is excluded from each is the first logged
row, the rows the held-out evaluator ran on, and the row after each of those; the lane-monitor
exclusion removes nothing here because the baseline attaches no monitor. The projection charges
evaluations at the 26 s this shape measures rather than the L40S's 104 s, which is why
`arm_seconds` grew keyword arguments for its fixed costs.

The two cells the L40S pool never placed are here and are 500 steps behind the first two, which
costs nothing: the bound is per cell and per attempt, and every cell projects at 3.2 hours.

### The A100 constant is right. The L40S constant was wrong, by exactly two

The A100 half of this is clean, and it is worth saying precisely because of the history. OLMo-core
v2.5.0 fixed an A100 peak that was 2× too low and had been inflating reported MFU by 2×. That fix
is present and correct on this branch:

- **The peak divides out of the run's own logs at exactly 312.00 TF.** `flopsPS` over `MFU/100` is
  3.12e14 in every row.
- **312 is the dense BF16 figure for an A100**, half of the starred 624 on NVIDIA's datasheet, and
  GA100 reaches it with FP32 accumulation.
- **The hand calculation agrees.** `hc_370M` builds `num_flops_per_token` = 3,032,684,544 at
  sequence length 4096 and the padded vocabulary of 100,352 — the padding matters, and is 0.015%
  — against 3,032,684,620 divided out of the run. At 98,304 tokens a device-step and a clean
  median of 1.681 s that is **56.85%**, against 56.87% reported. Agreement to two decimal places.

**The L40S constant is the one that was wrong, and this branch is what put it there.** There are
two independent factors of two on these datasheet figures and only one of them is sparsity. The
other is the accumulation format, and the L40S branch — added by this branch, with a comment
reasoning carefully about sparsity and not at all about accumulation — took the wrong one:

- NVIDIA's Ada whitepaper gives the AD102 rates directly: **330.3 TFLOPS for FP16 with FP16
  accumulate against 165.2 for FP16 or BF16 with FP32 accumulate.** Exactly half.
- Its L40 appendix lists BF16 as **`181 | 362`** for silicon whose product datasheet headline is
  `362.05 | 733*`. The datasheet quotes the FP16-accumulate rate.
- Torch accumulates in FP32 unconditionally — cuBLAS is called with `CUBLAS_COMPUTE_32F` for BF16
  inputs and there is no BF16-accumulate path to opt into. So 362.05 is a rate no training kernel
  on that card can reach.

The dense BF16 peak of an L40S, for the only kind of matmul training performs, is **181.03 TF**.
`SpeedMonitorCallback` and `stage_gate` are both corrected. The consequence for the record is that
**every L40S MFU this branch has quoted is understated by two**: the stage-1 baseline was at
**40.08%**, not the 20.04% reported above and in the section before it. The two agreed with each
other because they were the same arithmetic over the same wrong denominator, which is the failure
mode a hand calculation only catches if it sources its own constant. The A100 and H100 branches
are unaffected — the datacenter dies have no accumulation penalty — and the L4 and A10G branches
are suspected wrong the same way and left alone with a comment, since nothing here has run on
either and reasoning by analogy is how the L40S got its first wrong value.

### Where the 4.8× actually comes from

8.219 s/step to 1.700 is **4.84×** on nominally 1.72× the compute, and the excess is mostly not
efficiency — it is the denominator above.

| | L40S ×4 | A100 ×8 | ratio |
| --- | --- | --- | --- |
| wall clock per step | 8.219 s | 1.700 s | 4.84× |
| node peak, datasheet | 1,448 TF | 2,496 TF | 1.72× |
| node peak, FP32 accumulate | 724 TF | 2,496 TF | **3.45×** |
| MFU | 40.08% | 56.21% | 1.40× |

So 3.45× of the 4.84× is hardware the L40S never had, and only **1.40×** is the machine being used
better. That residual is two things, neither of which is the tensor cores:

- **Memory bandwidth.** 1,555 GB/s of HBM2e per A100 against 864 GB/s of GDDR6 per L40S, and eight
  devices against four: 12.4 TB/s against 3.5 TB/s per node. A 370M model spends a large share of
  its step in norms, RoPE, SwiGLU elementwise and the optimizer, none of which touch a tensor core.
- **The interconnect, and FSDP's exposure to it.** The train module runs `fsdp` with
  `wrapping_strategy: full` and `prefetch_factor: 0`, so every parameter all-gather is fully
  exposed rather than overlapped with compute. At 474M parameters in bf16 that is 12 microbatches
  × 2 all-gathers × 711 MB ≈ 17 GB per rank per step on the L40S, over PCIe Gen4 — the L40S
  datasheet says `NVIDIA NVLink Support: No`, so there is no peer path at all — against 8 × 2 ×
  830 MB ≈ 13 GB over NVSwitch on the A100. Order 1–2 s of the L40S step against order 0.05 s of
  the A100's.

The two are not separable without a profile and the arithmetic above does not pretend to have
separated them. What it does establish is that the puzzle was mostly in the denominator, and that
the earlier note calling the L40S "bound by its wire rather than its tensor cores" was right about
the direction while being wrong about the size by a factor of two.

### The arm *is* in the logged config, and `tranche_watch` now reads it

`train_on_corpus` writes no `arm` field, which is why the watcher used to take the arm from its
command line. But `arm.apply` edits the model config before `ConfigSaverCallback` saves it, and
those edits survive into W&B: `model.block.hyper_connections` is absent on `baseline` and present
on every lane arm, among the funded arms the triple `(mode, doubly_stochastic,
output_init_exponent)` separates `faithful` from `output-only` from `mhc`, and `model.block_reuse`
separates the tied arms from the untied ones. `stage_gate.arms_consistent_with` reads all of it
and `tranche_watch` now defers to it, keeping the command-line label only as a fallback for a cell
that has not yet written a config, and marking it with a `?` when it is being used.

This is worth more than a label. A watcher told its arm on a command line will print whatever it
was told, so the one failure it could catch — a cell that resolved to an arm nobody meant, which
`resolve_cell` is shaped around and which no loss curve would reveal — is exactly the failure it
cannot. The config is the run's own testimony, and when the two disagree the watcher now says
`ARM MISMATCH` and names the cells.

It does not separate every arm and does not claim to. `decay-everything` differs from `faithful`
in the optimizer alone, which is not a model-config difference at all, so the two come back
together; it is unfunded, and each of the four arms that run is uniquely identified. Reading
`block_reuse` is what removed the other ambiguity: without it every baseline cell came back as
`baseline` and `tied-baseline` together, and a check whose own output says "ambiguous" five times
out of five teaches the reader to skip the word.

### `wandb_panels.py --verify` passed for the wrong reason, and now scopes to a submission

Run against the group while stage 1 was live it printed `VERDICT: everything the pre-registration
rests on is present` and exited 0, for two independent and equally disqualifying reasons.

**It unioned keys across every run in the group.** The group is the whole module — every probe,
every rehearsal, every arm, months of them — so the `hc/*` families it reported `ok` were carried
entirely by old `faithful` probes, while the runs being gated were five `baseline` cells that
cannot log a lane metric at all, since `train` attaches the monitor only to an arm with lanes.

**It keyed `observed_keys` by `run.name`, and every cell of a fan-out shares a name.** A five-cell
submission reported as `1 run(s)`, and a cell missing a family was hidden behind its siblings. The
cell index is in `run.id`, as `<run id>-cell-<index>`, and the name is not even stable — the three
cancelled L40S cells were all renamed to `…-died`.

A verifier that passes for the wrong reason is worse than none, because it is the thing standing
between the tranche and an analysis over a missing metric family. `--verify` now takes `--run`
rather than `--group` and refuses the combination outright, addresses cells by id, and carries a
scope per family: `hc/*` is required on an arm with lanes, **not expected** on one without, and
*forbidden* there — a lane metric on a baseline cell means that cell did not run the arm it was
submitted as, which is a failure the old "is anything missing" framing scored as a pass with
extras. The weight-decay split stays advisory on every arm, because the live baseline cells log
`optim/LR (group 1)` too: OLMo-core already splits decay off the norms and biases, so that family
confirms groups exist and not that the lane split is why.

```bash
python .edullm/wandb_panels.py --verify --run <full run id> --cells 5 --arm baseline
```

## `--hours 4` was the baseline's number carried onto arms that have lanes

`run_019fe7bc-49d0`, the `faithful` stage, died complete: five of five cells, at steps 4,640 to
4,699 of 6,000, each at a runtime of 3.99 hours. Nothing was wrong with any of them.

The bound came from `A100_MEASURED_CELL_HOURS = 3.00`, and that is the *baseline's* measured cell.
The baseline has no hyper-connection lanes to compute and runs at 1.700 s/step; measured over their
own histories the three lane arms run at 2.87 to 3.15. Four hours buys a lane arm about 4,950
steps and the tranche asks for 6,000. The 33% margin the four hours was chosen for was 33% over an
arm that was not being submitted.

### What a wall looks like, so that the next one is recognised in a minute

A fault takes one cell, at one step, at one moment. A bound takes every cell at the same *runtime*,
at whatever step each of them had reached:

| cell | started | last step | died | runtime |
| --- | --- | --- | --- | --- |
| 0 | 18:27:42Z | 4,694 | 22:27:42Z | 4.00 h |
| 1 | 19:35:41Z | 4,640 | 23:34:57Z | 3.99 h |
| 2 | 18:34:40Z | 4,699 | 22:34:40Z | 4.00 h |
| 3 | 18:28:00Z | 4,660 | 22:27:31Z | 3.99 h |
| 4 | 19:40:46Z | 4,694 | 23:40:47Z | 4.00 h |

The five deaths are spread over 73 minutes of wall clock and **none of that spread is information**
— it is the 73 minutes the starts were spread over, because `gpu-8xa100` places after a wait and
`check` reports a median of 61 minutes for it. What is information is the second column against
the fifth.

The 59 steps of spread in the third column is the confirming detail rather than slack, and it is
worth the arithmetic because it is what distinguishes a bound from a coincidence. The five cells
ran at 3.044, 3.074, 3.043, 3.065 and 3.043 s/step — a 1.0% spread over five different hosts — and
1.0% of 4,700 steps is 47. Same time, slightly different rates, therefore slightly different steps.
A fault that caught five cells within 59 steps of one another would have had to be triggered *by*
the step count, and then the runtimes are what would have disagreed.

## A timeout gets no second attempt, and the run that proves the instrument got one

`--attempts 2` does not cover a run that ran out of time. This was measured on seven cells and it
is the opposite of what the platform's own text says, so it is worth setting out in full.

**Fifteen cells have hit the four-hour wall** — every cell of all three treatment stages,
`run_019fe7bc-49d0`, `run_019fe7bc-53f3` and `run_019fe7bc-73a6`, between 22:27Z and 03:41Z.
Fourteen of the fifteen still had an unused second attempt. **Not one of them started a second
process.** Every one reads as a single continuous run of system metrics ending at its wall, with a
largest internal gap of 0.2 minutes, which is the sampling interval.

The last of them was watched through its wall deliberately, because it was the cheapest available
test. Cell 3 of `run_019fe7bc-53f3` was projected from its own fitted rate to stop at step 4,995,
and it stopped at **step 4,995**, at a runtime of 3.993 hours, and did not come back.

**The instrument is not merely silent, because one retry did fire and is plainly visible.** Cell 1
of the `faithful` stage lost its first attempt about a minute in: its system-metrics stream runs
18:38:15Z to 18:39:00Z and then stops for **56.9 minutes**, resuming at 19:35:57Z, and the
`wandb-metadata.json` the second process left behind reports `startedAt` 19:35:41Z against a run
created at 18:38:00Z. A retry reuses the cell's W&B run id, so it appears as a second process under
the first one's name, and a 57-minute wait for it is what the capacity note predicts. That cell is
the positive control: retries happen, they are placed within about an hour, and they show up here.
Fourteen timeouts with an attempt in hand produced nothing that looks like it, over five hours.

**When that retry ran it started from step 0, correctly**, because its first attempt died before
writing a checkpoint. It then ran a clean 3.96 hours and hit the same wall as its siblings. So the
one retry this tranche has been granted was also the one that had nothing to resume from, and the
resume path is still unexercised on the platform.

### The mechanism, which is an exit code the platform was not expected to see

`policy.yaml` reasons that a timeout records no container exit code, so `OnExitCode "*"` cannot
match it and Batch's fall-through grants the attempt. `TRANCHE_STEPS` already doubted this on the
grounds that torchrun re-raises `SignalException` on SIGTERM and exits non-zero, calling it a dead
heat with the container stop timeout. The heat is not dead, and cell 0 of `run_019fe7bc-53f3` says
so in its own summary:

```
edullm_stage       TRAINING_ITSELF_FAILED
edullm_exit_code   72
edullm_explanation RuntimeError: DataLoader worker (pid 9068) is killed by signal: Terminated.
```

The SIGTERM at the wall reaches the dataloader workers first, surfaces inside the training loop as
an ordinary `RuntimeError`, and `cli` turns it into a `Refusal` and returns its stage as the
process's exit status. **The container exits 72.** Rule three matches an exit code of any value, so
the retry is refused — not because the platform failed to notice the timeout, but because this
program is well enough behaved to report one. The better the error handling, the more certainly the
attempt is forfeited.

What `edullm check --json` says under `retries.said` is that "the attempt a retry is actually spent
on is the one that ran out of time, and it gets the same bound again". On this workload that is not
what happens, fifteen times out of fifteen.

**Keep `--attempts 2` anyway.** It buys exactly what `A100_STAGE_ATTEMPTS` always said it bought —
a lost host — and cell 1 above is a cell that would otherwise have been missing from the arm mean
for a failure sixty seconds long. It costs nothing in expected spend, because billing is wall clock.

### A cell that reported its own death overwrote the evidence of its life

Cell 0 of `run_019fe7bc-53f3` was read as "seed 0 failed at step None" and treated as a cell that
never trained. It trained for 3.993 hours and reached **step 4,910**, further than any other cell
in the tranche, and then died at the wall like the rest.

`leave_the_reason_in_wandb` is why the summary said otherwise. It calls `wandb.init` when
`wandb.run` is None, which it is by the time the trainer's callback has gone, and the platform sets
`WANDB_RUN_ID` in the environment — so the diagnostic run is not created *beside* the cell, it is
created *as* the cell. It renames the run to `…-died` and replaces the summary, taking `_step` with
it. The history survives and the step count is recoverable from it, but nothing that reads
`summary["_step"]` — `tranche_watch` included — can see it.

It took four more cells the same way at their own walls — cell 4 of `run_019fe7bc-53f3` and cells 1
and 3 of `run_019fe7bc-73a6`, whose `startedAt` all post-date their own last system metric — and
three cells of the stage-1 baseline `run_019fe279-4ef0` carry the same `-died` name from before
anybody noticed. Seven clobbered summaries and every one of them read as a cell that never trained.

**The cost is not the lost diagnostic, it is the missing replicate.** The analysis reads endpoints
out of W&B, a clobbered summary looks like a crash, and a replicate excluded for looking like a
crash is excluded *non-randomly* — it happens to cells that hit a wall, never to cells drawn at
random. It changes n, it changes the df, and nothing downstream says so.

#### The fix: a report can create a run and can never attach to one

The report now gets a run of its own, at the cell's id with `-died` appended, and two things make
that a guarantee rather than an intention. The id is passed to `wandb.init` explicitly, and
`WANDB_RUN_ID`, `WANDB_RESUME` and `WANDB_NAME` are out of the environment while it runs, so no
precedence rule inside a client that gets rewritten can put the cell's id back. And `resume="never"`
is W&B's own instruction to *fail* rather than attach if a run with that id already exists — so a
future edit that got the id wrong would be refused instead of quietly overwriting a training
record. The cost is one case: a second attempt that also dies finds its own report already there
and is refused, leaving the first attempt's, which is on stderr either way.

Appending to the cell's id rather than replacing it is what keeps `…-cell-3-died` readable as cell
3, so a report is still placed in cell position by anything that addresses cells by index. Where
the training run is still open in the process the reason goes into it instead: a write through a
handle already held adds keys and creates nothing, which is the one way of touching the training
run that cannot reset it. The tag `died-before-training` is gone — it was the sentence that made a
four-hour cell read as one that never started — and `edullm-crash-report` says what the run is
while the stage tag beside it says when.

The two runs this was always for are unharmed and are the control: `run_019fdf85-b356` and
`run_019fdfcf-0822` carry a `-died` name, no config and no history, because they really did die
before the trainer reached W&B. There was nothing there to overwrite.

#### What is recoverable, and it is everything that matters

`lastHistoryStep` is the recovery and it is a field rather than a scan — W&B keeps it beside the
history, and on an intact run it equals the summary's `_step`, which is what makes it safe to
prefer. All seven cells keep their full per-step history, their saved config and their held-out
evaluations:

| submission | arm | cell | seed | step, from history | runtime | held-out evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| `run_019fe279-4ef0` | `baseline` (stage 1) | 0 | 0 | 910 | 2.13 h | 2 |
| `run_019fe279-4ef0` | `baseline` (stage 1) | 1 | 1 | 915 | 2.12 h | 2 |
| `run_019fe279-4ef0` | `baseline` (stage 1) | 3 | 3 | 900 | 2.13 h | 2 |
| `run_019fe7bc-53f3` | `output-only` | 0 | 0 | **4,910** | 3.99 h | 10 |
| `run_019fe7bc-53f3` | `output-only` | 4 | 4 | 4,835 | 3.99 h | 10 |
| `run_019fe7bc-73a6` | `mhc` | 1 | 1 | 4,565 | 3.99 h | 10 |
| `run_019fe7bc-73a6` | `mhc` | 3 | 3 | 4,580 | 3.99 h | 10 |

**All seven cells are usable data and none of them was ever missing from a noise floor**, because
`read_seed_series` takes its per-source endpoints from `scan_history` and not from the summary. Each
of the four treatment cells carries the same ten evaluations to step 4,500 that its intact siblings
do, so both arms read at five seeds and df 4, not three and df 2. What was wrong was every number
*about* those runs: `step None`, `runtime 0.0`, and a name saying they died before training.

Genuinely lost, and not worth chasing: the summary's own copy of the final metric values, which the
history holds anyway; the display name; and the run's W&B metadata, so `gpu` reads empty and
`startedAt` is the crash process's clock. The card is the one thing a reader might want from that,
and every cell of a submission runs on the same shape, so a sibling names it.

#### What the analysis does now when it meets one

Nothing rewrites W&B. The recovery is in the readers:

- `noise_floor` records `summary_step` and `history_step` separately, prefers the history where the
  summary has none, and prints which cells it had to recover and from what. `excluded()` is the
  complement of `contributing()`: every run that is dropped is now named with the reason it was
  dropped, and any exclusion is a `provisional` reason, which is what `--freeze` refuses on. A
  reading short a replicate can no longer be written into the frozen artifact.
- `stage_gate` recovers both the step and the held-out source list from history. This one was a
  silent skip rather than a wrong number: the per-cell held-out check runs over the cells that have
  sources, so a cell whose summary lost its `eval/lm/*` keys was not failed, it was passed over.
- `tranche_watch` recovers the step from `lastHistoryStep`, which arrives with the run object and so
  costs a poller nothing, and reads `…-cell-N-died` as a report against cell N rather than as a cell.

**A report test that reads only the metadata deletes the runs it is meant to rescue**, and this was
caught by running it: the seven clobbered cells carry `job_type: crash` *themselves*, because the
report was written onto them. Filtering reports out of an arm on `job_type` alone dropped cells 0
and 4 of `run_019fe7bc-53f3` and turned a five-cell submission into a three-cell one — the original
failure, one layer up and with the fix's name on it. A run that logged a step trained, whatever its
metadata was overwritten to say, so `is_crash_report` will not call anything with history a report.

## What the lane arms cost, measured on the arms themselves

Every cell's history gives a least-squares fit of runtime against step. The slope is the marginal
cost of a step with the twelve evaluations and thirteen checkpoints amortised into it, and the
intercept is start-up. Fitted from step 200 onward, over 4,900 steps where they exist:

| arm | marginal s/step, per cell | slowest | 6,000 steps | at `--hours 6` |
| --- | --- | --- | --- | --- |
| `output-only` | 2.914, 2.929, 2.942, 2.867, 2.961 | 2.961 | **4.98 h** | 1.02 h spare, 17% |
| `faithful` | 3.044, 3.074, 3.043, 3.065, 3.043 | 3.074 | **5.17 h** | 0.83 h spare, 14% |
| `mhc` | 3.149, 3.128, 3.129, 3.126, 3.110 | 3.149 | **5.30 h** | 0.70 h spare, 12% |

The projection is `intercept + slope × 6000` plus one more evaluation-and-checkpoint pair for step
6,000 itself, which the histories measure at 57.5 to 60.6 seconds and which is therefore 0.017 h
and not worth arguing about. Start-up is 79 to 97 seconds on every cell.

**Six hours covers all three arms and is the right bound.** `mhc` has the thinnest margin at 12%
and it is still four times the shortfall that killed the `faithful` stage.

**The `faithful` resubmission is already confirming it.** `run_019fe90b-f99e` went out at
`--hours 6` and its five cells fit at 3.050, 3.043, 3.083, 3.046 and 3.027 s/step — the same arm on
new hosts, projecting 5.07 to 5.16 hours against a six-hour bound. The host-to-host spread across
the two submissions of this arm is 1.8%, which is the number to carry into any later bound rather
than the 1.0% a single submission suggested.

**The `mhc` estimate is trustworthy despite being taken early.** It is the one arm with no
completed cell, and Sinkhorn-Knopp was expected to make it the slowest, which it is — but only by
2.3% over `faithful`, not by the margin the kernel-launch count suggested. Truncating the finished
`faithful` cells to the same prefix says how much an early read misleads: fitted on their first
1,000 steps they give 3.033, 3.069, 3.033, 3.056 and 3.035 against full-history values of 3.044,
3.074, 3.043, 3.065 and 3.043. **A 1,000-step prefix understates the settled rate by 0.36%**, and
by 2,000 steps it is within 0.1%. `mhc` is read at 900 to 2,200 steps, so 3.145 is low by about a
hundredth of a second and 5.29 hours is low by about a minute. Its cells then ran to their walls and
settled at 3.110 to 3.149, so the prefix was low by 0.13% and the projection by 24 seconds.

Seven hours is available at a ceiling of $1,537.03 against six hours' $1,317.46, and both price as
`routine`. It buys nothing the measurement asks for. It is also not free of a cost that is not
money: `STAGE_HOURS` records why a bound has to be the same across stages, which is that an arm
under a looser bound survives drift the others die of, and a treatment arm missing its slowest cell
is not missing it at random.

### A resubmission cannot inherit a dead cell's checkpoints

`EDULLM_CHECKPOINT_DIR` is `…/runs/<run id>/cell-<index>/checkpoints/` and a resubmission is a new
run id, so it is a new and empty prefix. Nothing the three stages wrote is reachable from the runs
that replace them. That is the whole reason the retry question was worth an hour of anybody's time:
a granted retry resumes inside the run and costs one save interval, and a resubmission starts from
zero and costs the arm.

**The four-hour bound cost 60.9 node-hours across fifteen cells, and none of it survives.** Every
cell reached between 75% and 83% of its steps, which is the expensive place to stop: far enough in
to have paid for nearly all of a cell and not far enough to leave a checkpoint anybody will load.
At the rate `check` reports it is about $1,340, and the arithmetic that would have avoided it —
6,000 steps times a step time measured on an arm that has lanes — is one line.

It follows that **a cell that cannot reach 6,000 within its bound should be cancelled as soon as
that is known, not left to reach its wall.** It is not accruing anything that survives it. Ten of
the fifteen were still running when the bound error was understood and every one of them was
allowed to reach its wall regardless, which is about $385 of the total.

## Order of operations

1. **Rehearse.** Done: `run_019fdfe9-e6c0`, `faithful` at the rehearsal size, 200 steps,
   `gpu-4xl40s`. It fails closed if the lanes have not differentiated by step 150.
2. **Probe throughput** at 370M. Done twice: `run_019fe008-5877` at rank microbatch 8,192 and
   `run_019fe1f6-8692` at 16,384, and the table above.
3. **Get an arm under the 24-hour ceiling.** Settled, by shortening the horizon to 6,000 steps
   rather than by making the step faster — see
   [What a full arm actually costs](#what-a-full-arm-actually-costs-and-why-it-cannot-be-submitted-as-one-run).
   Flash-attention and a bfloat16 gradient reduction are both still on the table and neither
   closes a gap this size on its own; an NVLink shape would, and is unmeasured. **It is measured
   now** — `run_019fe262-778d` puts the full horizon at 11.5 hours on `gpu-8xa100` — and the
   horizon still does not come back, for the reason in
   [The shape for stages 2, 3 and 4](#the-shape-for-stages-2-3-and-4), which is that a fixed
   budget spends it better on replicates.
4. **Arm 1, three seeds, as one three-cell fan-out.** Fills the noise floor table above,
   including ρ̂ and Bartlett. It is also the first submission, so it is the one that finds out
   whether the fan-out resolves three different seeds — check that before submitting the
   other two, by reading the three cells' `init_seed` out of their W&B configs.
5. **Arm 2, three seeds**, which answers H1 against those three.
6. **Arm 3, three seeds**, which answers H2a against arm 2.
7. **Second tranche**, in `CUT_ORDER` read backwards: `mhc` first, then `no-output-init`. The
   10B horizon comes back here too, and on a shape that can hold it.

## One ambiguity in the source, recorded because it is an arm

ByteDance write that they "scale the std of the weights of the output module at all layers ...
by a factor of √n". The stated purpose is to keep the standard deviation of the pre-unembedding
hidden state consistent with the original — but hyper-connections make that quantity *larger*,
because the lanes are summed, so the factor has to be a divisor. It is implemented as an
exponent rather than hard-coded:

- `output_init_exponent=0.5` — the paper as written, read as a divisor. The default, and arm 2.
- `output_init_exponent=1.0` — what exactly cancels the sum at initialization, where the lanes
  are still identical copies of each other.
- `output_init_exponent=0.0` — off. Arm 4.

At initialization the lanes are identical, so the sum is exactly n times the baseline hidden
state and the correct canceling exponent is 1.0; the paper's 0.5 is what you get from assuming
the lanes are independent, which is a statement about the trained model rather than the
initialized one. The rehearsal logs the pre-final-norm hidden-state norm, so this is settled by
measurement rather than by reading. Whatever it says, arm 4 is unaffected: it turns the
correction off either way.

## What is instrumented, and why a null here is still readable

`HyperConnectionMonitorCallback` logs, per layer:

- **Per-lane norm, the spread across lanes, and the dispersion of the lanes about their mean.**
  The primary guard. Lanes that all carry the same vector make the model the baseline with extra
  parameters, and no downstream number is interpretable in either direction. Dispersion is what
  the guard reads, because equal lane norms are also what a rotation produces and the spread
  cannot tell the two apart. The rehearsal turns this into an error once a majority of blocks
  fail it; a minority is logged as a warning, since the 370M probe found blocks 01 and 02 flat
  while the other fourteen separated.
- **Spectral radius of the lane-mixing matrix.** Parcae
  ([arXiv 2604.12946](https://arxiv.org/abs/2604.12946)) found diverging runs learn a radius at
  or above 1. Tencent's 3B divergence had a multi-lane drift signature. Under `mhc` this is
  pinned at exactly 1 by construction, which is the claim being tested.
- **Condition number of the composite mapping across depth.** mHC's argument for the constraint
  is that the constrained matrices are closed under multiplication, so the composite stays well
  conditioned. This measures it instead of citing it. The product follows
  `Transformer.block_execution_order`, so the tied arms compose the 16 matrices the model
  applies rather than the 8 distinct ones it holds.

  On the constraint itself: at the shipped eight Sinkhorn sweeps `A_r` comes back **column**-
  stochastic and only approximately row-stochastic — row residuals reach 4.6e-01 at drifted
  logit scales. The property mHC's argument needs survives that intact, because a nonnegative
  column-stochastic matrix already has spectral radius exactly 1 and is closed under
  multiplication, and the measured radius sits within 1.3e-06 of 1 in every case. The Birkhoff
  polytope is not where this lands, and the docstrings that said it was have been corrected.
- **Hidden-state norm per layer.** RMSNorm readouts are scale-invariant, so cross-entropy
  cannot see hidden-state scale at all, and pre-norm stacks have been measured driving norms
  into the 10^3–10^4 range invisibly.

## Running it

```bash
# The arm table.
python .edullm/train_hyper_connections.py --list-arms

# Tests, none of which need a GPU.
pytest -v .edullm/test_hyper_connection_arms.py .edullm/test_train_hyper_connections.py \
          .edullm/test_skip_step_calibration.py
pytest -v src/test/nn/transformer/hyper_connection_test.py \
          src/test/nn/transformer/block_reuse_test.py \
          src/test/train/callbacks/hyper_connection_monitor_test.py \
          src/test/train/callbacks/skip_step_monitor_test.py

# What the amendment's skip threshold would have done to the runs that are already finished.
# --self-test needs no network; the replay needs W&B and reads the histories of one submission.
python .edullm/skip_step_calibration.py --self-test
python .edullm/skip_step_calibration.py --submission run_019fe2f4-f528 \
  --cache /tmp/hc-history.json --sigma 4,5,6,8,10 --episodes 0:1376-1418,1:1726-1773

# Preflight, on a laptop, before anything is submitted. Needs corpus credentials, no GPU.
for arm in baseline faithful output-only; do
  AWS_PROFILE=sbsandbox PYTHONPATH=src python .edullm/train_hyper_connections.py pf \
    --preflight --arm "$arm" \
    --dataset-id pretrain/regmix-10b --dataset-version v1 \
    --dataset-tokenizer tokenizer/dolma2-bpe \
    --save-folder /tmp/x --work-dir /tmp/hc-cache
done

# And that a fan-out cell resolves the seed it should. Three different init seeds or stop.
for i in 0 1 2; do
  AWS_BATCH_JOB_ARRAY_INDEX=$i EDULLM_FANOUT_INDEX_PARAMETER=seed \
  AWS_PROFILE=sbsandbox PYTHONPATH=src python .edullm/train_hyper_connections.py pf \
    --preflight --arm baseline \
    --dataset-id pretrain/regmix-10b --dataset-version v1 \
    --dataset-tokenizer tokenizer/dolma2-bpe \
    --save-folder /tmp/x --work-dir /tmp/hc-cache | grep -E '^seed'
done

# On the platform: edit --arm in .edullm/run.yaml, commit, push to edullm/**, then
edullm check --json --experiment hyper-connections-370m --dataset regmix-10b-v1 \
  --team <team> --hours 21 --fanout-size 3 --fanout-index-parameter seed
```

The command lives in `.edullm/run.yaml` rather than in a flag, so the commit and the arm cannot
disagree about what was run. The **seed** is not in it, for the opposite reason: every cell of
a fan-out is handed one command, so a seed written down there would run one replicate three
times. `resolve_seed` refuses that combination rather than honouring it, and a test asserts
that `run.yaml` carries no `--seed`.

`--hours 21` rather than the 24 the profile allows, because `check` prices a ceiling and nine
cells at 24 hours is $4,532.79 against a $4,000 budget. Twenty-one is $3,966.21 and still
leaves 42 hours of allowance for a 17.9-hour run that may lose a host.
