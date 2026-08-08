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
QK-norm, RoPE, z-loss, dolma2 at vocab 100,278 (padded to 100,352), untied embeddings. 10B
dolma2 tokens at sequence length 4096, which is 12,715 steps of a 786,432-token batch.
**3.0e19 FLOPs per arm**, from the run's own accounting rather than from a rule of thumb: the
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
| 1 | `baseline` | 3 | 474,022,912 | +0.0000% | +0.0000% |
| 2 | `faithful` | 3 | 474,220,352 | +0.0417% | +0.0994% |
| 3 | `output-only` | 2 | 474,187,456 | +0.0347% | +0.0908% |
| 4 | `no-output-init` | 2 | 474,220,352 | +0.0417% | +0.0994% |
| 5 | `decay-everything` | 1 | 474,220,352 | +0.0417% | +0.0994% |
| 6 | `n1` | 1 | 474,121,376 | +0.0208% | +0.0119% |
| 7 | `n2` | 0 | 474,154,304 | +0.0277% | +0.0324% |
| 8 | `n8` | 0 | 474,353,216 | +0.0697% | +0.3371% |
| 9 | `mhc` | 3 | 474,220,352 | +0.0417% | +0.0994% |
| 10 | `tied-faithful` | 1 | 339,871,136 | −28.3007% | +0.0994% |
| 11 | `tied-baseline` | 1 | 339,772,416 | −28.3215% | +0.0000% |

Every untied arm is iso-parameter to within 0.07% and iso-FLOP to within 0.34%. The tied arms
are deliberately not iso-parameter — that is what they test — but they are iso-FLOP with their
own control, because they are matched on *effective* depth: 16 layers running 8 distinct
blocks twice on a cycle.

Seventeen runs in total once seeds are counted. The number is now
`hyper_connection_arms.total_runs()` with a test on it, because the "fifteen" that stood here
was wrong from the day it was written and nothing in the repository could say so. Cut order if
the budget does not stretch: `n8`, then `n2`, then `tied-faithful` and `tied-baseline` as a
pair. The first two are already cut, and [Where the seeds went](#where-the-seeds-went) says
what bought what.

A zero in the seeds column is not the same as an absent row. Arms 7 and 8 are still specified,
still build, and still have to pass every property the other arms pass; funding them later
costs a number in one file and no design work at all.

## Pre-registered hypotheses

Every hypothesis is a directional claim about held-out cross-entropy in nats on the seven
validation sources, reported per source and as their mean, with bits-per-byte beside it as the
same quantity divided by a constant. "Beats" means the paired difference defined in
[The analysis plan](#the-analysis-plan) clears the gate stated there.

- **H1 (replication).** Arm 2 beats arm 1 by ≥0.025 nats.
- **H2a (the artifact, in-loop).** Arm 2 > arm 3 on held-out cross-entropy.
- **H2b (the artifact, downstream).** Arm 2 > arm 3 on the downstream average, and arm 3 but
  not arm 2 reproduces the published degradation. **Blocked**, and see below: this is the one
  that carries the headline, and nothing in this plan produces the number it needs.
- **H3 (initialization).** Arm 2 > arm 4; the output-init scaling is load-bearing rather than
  cosmetic.
- **H4 (the seesaw), restated as a superiority test.** Arm 2 > arm 6. ByteDance found n=1 does
  not help; if one lane buys as much as four, their mechanism story is incomplete at this
  scale. The difference between arm 6 and arm 1 is reported with its interval as a *bound*,
  with no equivalence claim attached to it — see below for why the design cannot support one.
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

At a within-pair correlation ρ the paired difference has standard deviation σ√(2(1−ρ)), and
the minimum detectable effect at 80% power, α = 0.05, pooled df = 6, σ = 0.010 goes:

| ρ | 1 pair | 2 pairs | 3 pairs |
| --- | --- | --- | --- |
| 0.0 | 0.048 | 0.034 | 0.028 |
| 0.3 | 0.040 | 0.028 | 0.023 |
| 0.5 | 0.034 | 0.024 | 0.019 |
| 0.7 | 0.026 | 0.018 | 0.015 |

The unpaired three-versus-three comparison is 0.028 and the unpaired one-versus-three is 0.039,
so **pairing at any ρ above about 0.3 buys more than a second seed does, for nothing**. ρ̂ is
estimated from the H1 and H5 triples — the only two contrasts with three pairs — and is
reported before any single-seed arm is interpreted, because those arms borrow σ̂_Δ from it.

### What each hypothesis can actually detect

At σ = 0.010, pooled df = 6, two-sided α = 0.05, 80% power, unpaired. Literature effects for
comparison are ByteDance's −0.030 and Tencent's −0.020.

| | contrast | seeds | SE | MDE | reads |
| --- | --- | --- | --- | --- | --- |
| H1 | 2 − 1 | 3 v 3 | 0.0082 | 0.028 | marginal against −0.030 |
| H2a | 2 − 3 | 3 v 2 | 0.0091 | 0.031 | marginal |
| H3 | 2 − 4 | 3 v 2 | 0.0091 | 0.031 | marginal |
| H4 | 2 − 6 | 3 v 1 | 0.0115 | 0.039 | under-powered |
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

So H4 is restated above as the superiority test the design can run, arm 2 versus arm 6, which
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
because H4 is now the arm 2 versus arm 6 superiority test and the bound on arm 6 versus arm 1 —
neither of which a second seed rescues, since the equivalence claim it would have to support
needs three seeds and a measured ρ. The review this revision came from suggested second seeds
on arms 3, 4 *and* 6, which is eighteen runs rather than seventeen; the third one is the one
that buys the least, so it is the one left out.

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
This is the same trap `HyperConnectionStream` documents when it reads the lanes with a mean
rather than a sum. So Track B cannot share Track A's baseline, and that is one more three-seed
run in the budget.

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

| quantity | value |
| --- | --- |
| per-arm σ̂, held-out CE, arms 1 / 2 / 9 | **not measured** |
| pooled σ̂ across the nine runs, df = 6 | **not measured** |
| Bartlett p over the three within-arm variances | **not measured** |
| ρ̂, the within-seed correlation the pairing exploits | **not measured** |
| σ̂_Δ, the paired difference, from the H1 and H5 triples | **not measured** |
| per-seed σ, downstream average | **not measured** |
| minimum detectable effect, per contrast, 80% power | **not measured** |

The estimate this plan was written against is σ ≈ 0.008–0.012 nats, and every threshold in
[The analysis plan](#the-analysis-plan) is quoted at the 0.010 midpoint. That is an estimate
from the literature, not a measurement of this configuration, and all of it scales linearly
with σ̂. The three baseline seeds and the two other triples exist to replace it. **No treatment
arm is submitted until this table has numbers in it**, and no single-seed arm is interpreted
until ρ̂ does, because those arms borrow σ̂_Δ from the triples.

### Throughput

Measured, not planned. Read from run history at the steady state rather than from the run
summary — the last logged value is taken during the end-of-run evaluation, where the model is
not training and throughput reads near zero.

| run | config | steps | MFU logged | MFU L40S | device TPS | s/step |
| --- | --- | --- | --- | --- | --- | --- |
| `run_019fdfe9-e6c0` | `hc_rehearsal`, `faithful` | 200 | 9.12% | 7.86% | 57,354 | 1.14 |
| `run_019fe008-5877` | `hc_370M`, `faithful` | 100 | 12.30% | 10.60% | 12,645 | 15.55 |

Both on `gpu-4xl40s`, and all four medians are over the logged steps of each run. TPS is per
device, so the shape's total is four times it, and seconds per step is the batch over that
total: 786,432 / (4 × 12,645) = 15.55 at 370M, and 262,144 / (4 × 57,354) = 1.14 at the
rehearsal size.

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

The rehearsal figure was never a prediction for 370M: it is a 96M model of which 77M is
embedding and unembedding, so it spends an unusually large share of its time in two matmuls
that do not grow with depth. The probe row is the one to plan against, and it is 4.5 times
slower per token.

### What a full arm actually costs, and why it cannot be submitted as one run

At 15.55 seconds a step, 12,715 steps is **197,700 seconds, or 54.9 hours** of training steps
alone, before in-loop evaluation and checkpointing. `edullm check --json` on 2026-08-08 priced
`gpu-4xl40s` at $10.4926 an hour for one node, which puts an arm near **$580**. Read those two
figures out of `check --json` again before anybody acts on them; the rate is reviewed
configuration and changes without notice, and the hours are the only half of this that belongs
to the branch.

**The 21 hours and $220 that stood here is wrong by a factor of 2.6.** It implies 5.95 seconds
a step, which is not a number this configuration has ever produced on this card. And the same
`check --json` reports a maximum runtime of 24 hours, so at the measured step time **a full arm
cannot be admitted at all** — it is not a matter of cost. Something has to change before arm 1
seed 0 is submitted, and the options are a faster attention backend, fewer tokens, a bigger
shape, or splitting each arm across resumed segments under the ceiling. That is a decision for
the throughput work and not for this document, but it is the binding constraint on the whole
module and it only became visible once the throughput number was right.

The `gpu-8xa100` figures that stood beside them — about 6 hours and $135 — are removed rather
than corrected. Nothing in this branch has run on an A100, so there is nothing to correct them
against, and a second unmeasured number beside a corrected one is worse than no number.

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

## Order of operations

1. **Rehearse.** Done: `run_019fdfe9-e6c0`, `faithful` at the rehearsal size, 200 steps,
   `gpu-4xl40s`. It fails closed if the lanes have not differentiated by step 150.
2. **Probe throughput** at 370M. Done: `run_019fe008-5877`, 100 steps, and the table above.
3. **Get an arm under the 24-hour ceiling**, which the probe says it is not. Nothing below can
   be submitted until this is settled, and the flash-attention backend is the first thing to
   try because the wheel is already in the image.
4. **Three seeds of arm 1.** Fill the noise floor table above, including ρ̂ and Bartlett.
5. **Three seeds of arm 2 and of arm 9**, which complete the pooled σ̂ and answer H1 and H5.
6. **Only then** the arms that borrow σ̂_Δ from those triples: 3 and 4 at two seeds each, then
   5 and 6, then the tied pair 10 and 11. Arms 7 and 8 carry no seeds and are not submitted.

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
- **Condition number of the composite mapping across depth.** mHC's argument for the Birkhoff
  constraint is that doubly stochastic matrices are closed under multiplication, so the
  composite stays well conditioned. This measures it instead of citing it.
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

# On the platform: edit --arm in .edullm/run.yaml, commit, push, then
edullm check --json && edullm submit
```

The command lives in `.edullm/run.yaml` rather than in a flag, so the commit and the arm cannot
disagree about what was run.
