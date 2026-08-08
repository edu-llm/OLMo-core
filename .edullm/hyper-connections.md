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

| # | arm | seeds | params | vs baseline | FLOPs/token vs baseline |
| --- | --- | --- | --- | --- | --- |
| 1 | `baseline` | **3** | 474,022,912 | +0.0000% | +0.0000% |
| 2 | `faithful` | **3** | 474,220,352 | +0.0417% | +0.0994% |
| 3 | `output-only` | **3** | 474,187,456 | +0.0347% | +0.0908% |
| 4 | `no-output-init` | 0 | 474,220,352 | +0.0417% | +0.0994% |
| 5 | `decay-everything` | 0 | 474,220,352 | +0.0417% | +0.0994% |
| 6 | `n1` | 0 | 474,121,376 | +0.0208% | +0.0119% |
| 7 | `n2` | 0 | 474,154,304 | +0.0277% | +0.0324% |
| 8 | `n8` | 0 | 474,353,216 | +0.0697% | +0.3371% |
| 9 | `mhc` | 0 | 474,220,352 | +0.0417% | +0.0994% |
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
| σ̂ on held-out BPB at the final step, df = 4 | 1 | **not measured** |
| σ̂ at each of the twelve intermediate checkpoints | 1 | **not measured** |
| per-source σ̂ over the seven held-out sources, df = 4 each | 1 | **not measured** |
| the per-source inverse-variance weights, and what they buy | 1 | **not measured** |
| pooled σ̂ across all fifteen runs, df = 12 | 2 | **not measured** |
| Bartlett p over the three within-arm variances | 2 | **not measured** |
| ρ̂, the within-seed correlation the pairing exploits | 2 | **not measured** |
| σ̂_Δ, the paired difference, from the H1 and H2a quintuples | 2 | **not measured** |
| per-seed σ, downstream average | neither | **not measured** |
| minimum detectable effect, per contrast, 80% power | 1 for σ̂, 2 for ρ̂ | **not measured** |

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

Two notes on reading throughput numbers here. OLMo-core v2.5.0 fixed an A100 peak-FLOPs
constant in `SpeedMonitorCallback` that was 2× too low and had been inflating reported MFU by
2×, so any figure from before that is wrong by a factor of two; this branch is on v2.5.0. And
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
pytest -v .edullm/test_hyper_connection_arms.py .edullm/test_train_hyper_connections.py
pytest -v src/test/nn/transformer/hyper_connection_test.py \
          src/test/nn/transformer/block_reuse_test.py \
          src/test/train/callbacks/hyper_connection_monitor_test.py

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
