# Stream load balancing on hyper-connected MoE: the design, and what it can answer

**Status:** pre-registration. Written before any arm has run, which is the only time the
choices below are free rather than degrees of freedom.
**Companion files:** `.edullm/run.hc-baseline.yaml` (stage 1),
`.edullm/run.hc-treatment.yaml` (stage 2), `src/scripts/ablations/hc_power.py` (every number
in section 6).

**The one-line answer to "can this experiment work":** not on validation loss, and the
arithmetic in section 6 says so plainly. It can work on a mechanism endpoint whose effect
size is measured in orders of magnitude rather than in the third decimal place of a nat, and
the design below is built around that endpoint with loss reported as an under-powered
secondary. A version of this document that put loss first would be proposing a run that
cannot answer its own question.

---

## 1. The question, and the one change

Hyper-connections replace a sub-layer's single residual with `n` streams, read one vector in,
and write the output back while an `n x n` matrix `H_res` mixes the streams. Three independent
lines of work now say the mixing matrix does nothing: fixing `H_res = I` beat learned mixing in
11 of 16 cells in Alimaskina et al. (arXiv:2606.03483) and at 1B in Oldenburg et al.
(arXiv:2607.18130), and a public mHC reproduction measured the `H_res` gradient norm at about
`1e-9` against `1.84` on the branch weights — the matrix never left its initialisation.

The likely mechanism is **stream collapse**: one stream comes to dominate, read and write
concentrate on it, and cross-stream mixing has nothing left to mix. Alimaskina et al. confirm
collapse with four independent probes.

Stream collapse is isomorphic to **expert collapse**, which MoE already solves with a
load-balancing auxiliary loss. In roughly eight papers surveyed, nobody has applied load
balancing to residual-*stream* usage. OLMo-core already carries the MoE machinery to mirror.

**The falsifiable prediction.** If collapse is why mixing looks useless, then balancing stream
usage should make learned mixing start to matter.

**The one isolated change** is a single boolean: `stream_balance_loss_weight > 0` on the
hyper-connection config, which is off by default and leaves the untreated path bit-identical.
Nothing else moves between the treatment arm and its reference.

**What that flag turns on is a package, and the estimand is named as one.** Enabling it chooses
a utilisation statistic (`dispersion`, the share of residual energy each stream carries that no
other stream carries) *and* a penalty form (`entropy`), and both default away from the literal
mirror of MoE's loss for reasons measured in `src/test/nn/stream_balance_test.py`: the naive
statistic is uniform at full collapse, and the squared-share form stops responding as the
collapse deepens. So the contrast under test is **"the dispersion-entropy stream-balancing
package, on or off"**, not "stream balancing in the abstract". The two components are exposed as
config values so a later tranche can separate them; this one does not, and does not claim to.

Two things the treatment cannot do, stated here rather than discovered later. Its gradient is
proportional to the deviation between streams, which is quadratic in the utilisation statistic,
so **at exact collapse the treatment's own gradient is zero too** — it amplifies an existing
asymmetry rather than creating one, which is what `init_noise_std` is for and why it must stay
nonzero in every arm. And the weight is applied **per wrapped sub-layer**: 12 blocks x 2
hyper-connections at 0.01 each, so the model-level total is up to 0.24 and is not comparable to
`MoEConfig.lb_loss_weight`'s 0.01, which is one router's.

## 2. What the machine actually is, and why it is not the p5 node

The brief for this work said "one p5 node (8xH100 80GB) for 12 hours", and observed correctly
that a 190M-370M model fits one H100, so eight GPUs should be eight concurrent runs rather than
one eight-way job. The second half of that is exactly right and the design exploits it. The
first half is not available, and it is worth writing down why rather than discovering it at
submit time.

| finding | evidence |
| --- | --- |
| `gpu-8xh100` is **not provisioned** | `config/workload-catalog.yaml`: `provisioned: false`. `resolve_compute_profile_for_execution` refuses it before pricing. |
| `gpu-8xh100` **does not place** | `config/capacity.yaml`: `places: unreliably`, measured by queue on 2026-08-05. `gpu-1xh100` is the same. |
| the whole P pool was dry at the instant it was probed | `config/capacity.yaml`'s header: "The P pool is dry, on-demand and on spot, in all fourteen type-and-zone combinations". `gpu-8xa100` is the exception and carries `after_a_wait` with a 61-minute median. |

Read those out of the reviewed configuration rather than out of this document, which goes
stale: `edullm check --json` prints the refusal and the placement warning for whatever shape a
submission names.

**A fan-out cell is a whole node, not a GPU.** `execution.batch_submit_request` sets
`ArrayProperties.Size` and each child is a separate Batch job with the full container
requirement, so `--fanout-size 8` on an eight-GPU shape is eight eight-GPU nodes and eight
times the price. Eight concurrent independent runs is therefore bought as **eight one-GPU
cells**, and that is both cheaper and better-placed than one eight-GPU node:

| way to buy 8 concurrent 12-hour runs | priced worst case | places |
| --- | --- | --- |
| 8 cells of `gpu-1xa10g` at $1.006/h for 12h | $193.15 | reliably |
| 8 cells of `gpu-1xl4` at $0.8048/h | $154.52 | unreliably |
| 1 cell of `gpu-8xa100` at $21.958/h (one 8-way job, not 8 runs) | $526.98 | after a wait, 61-minute median |

Priced cost is `rate x nodes x hours x attempts x cells` with `attempts = 2`. **The second
attempt is narrower than it looks and the design depends on knowing that.**
`RETRY_ONLY_WHAT_A_RETRY_FIXES` in the platform's execution module retries `Host EC2*`, exits on
`OutOfMemoryError*`, and exits on every other exit code — so a cell that runs out of wall clock
is **not** retried. The second attempt is for a host that went away. That is why `--hours` is 20
against an estimated 12: the bound is the approval ceiling and billing is by actual runtime, so
buying margin there is free, and buying it by cutting `--steps` is not.

**So: `gpu-1xa10g`, one card per cell, one cell per seed-and-arm.** It is the only 24 GB-class
shape in the catalog that both has bfloat16 in hardware and places reliably. `gpu-1xl4` is 20%
cheaper and `places: unreliably`; `gpu-1xt4` places reliably and is Turing, which has no
bfloat16 at all and is refused by the precision guard the moment the command names the dtype —
which is one of the reasons the command names it.

## 3. The shape and the horizon

`TransformerConfig.smallmoe`, unmodified, at the dolma2 padded vocabulary of 100,352:

| | |
| --- | --- |
| total parameters | 565,036,800 |
| non-embedding | 487,966,464 |
| **active per token** | **267,765,504** as `TransformerConfig.num_active_params` counts it, of which **190,695,168** are in matmuls — the difference is the embedding table, which is a lookup |
| FLOPs per token | **1.3706 GFLOP** at sequence length 2048, from the model's own `num_flops_per_token` rather than from `6N` |
| shape | `d_model` 768, 12 layers, 12 heads, block `moe_reordered_norm` |
| experts | 32, top-4, expert hidden 384, shared MLP hidden 1536, `lb_loss_weight` 0.01, `z_loss_weight` 0.001 |
| sequence length | 2,048 |
| global batch | 262,144 tokens (128 sequences) — the same matched budget `hc_ablation.py` already defines |
| horizon | 3,000 steps = **786M tokens** per cell |

Two things about this shape are chosen and one is inherited.

**Chosen: MoE rather than dense.** Hyper-connections are cheapest exactly here. HC Table 9
measures activation-memory overhead at `n = 4` as +28.28% on dense OLMo-7B and only +9.7% on
OLMoE-1B-7B, because the expert activations dominate and the residual stream is a small share
of the total. It is also the regime the idea is *about*: the treatment borrows MoE's own
solution to MoE's own collapse problem.

**Chosen: `smallmoe` rather than something tuned.** `--model-factory` in
`.edullm/train_on_corpus.py` resolves a bare `getattr(TransformerConfig, name)(vocab_size=...)`,
so the MoE factories reachable without new code are exactly `smallmoe`, `small_hybrid_moe` and
`olmoe_1B_7B`. The third does not fit the budget. Using an untouched factory is what lets stage
1 run before any of this branch's code exists.

**Inherited and uncomfortable: 786M tokens is short.** One step is 262,144 x 1.3706 GFLOP =
359.3 TFLOP; an A10G is 125 TFLOP/s bf16, so at an assumed 20% MFU that is 14.4 s/step and 3,000
steps is 12.0 hours. The smoke replaces the assumption with a measurement. Against Chinchilla it
is about a twentieth of what this active size would want. Section 7 says what that costs.

**And the card is 22.35 GiB, not 24.** `gpu-1xa10g` carries 22,888 MiB. The term that decides
whether `--rank-microbatch-size 8192` fits is not the 9.47 GiB of parameter, gradient and Adam
state but the **fp32 logits**: 8,192 tokens x 100,352 vocab is 3.06 GiB per copy, and at the LM
head's backward two or three copies can be live. The estimate lands between 18 and 21.5 GiB. It
probably fits, it has no margin, and doubling the microbatch doubles that term specifically — so
a comfortable `peak_memory_gib` at 8192 does not license 16384.

## 4. The arms

Four arms, a 2x2 of {learned mixer, pinned `H_res = I`} x {balancing off, balancing on}. Every
arm is `smallmoe` with both sub-layers of every block hyper-connected at `n = 4`, Sinkhorn
mixer, `init_noise_std = 1e-2`, `residual_dropout_p = 0.1`, mean stream collapse. Every arm
shares the shape, the optimizer, the schedule, the data, the horizon and the seed set.

| arm | mixer | stream balancing |
| --- | --- | --- |
| `mhc_moe` | Sinkhorn, learned | off — **the reference** |
| `mhc_moe_balanced` | Sinkhorn, learned | on — **the treatment** |
| `mhc_moe_identity` | `H_res = I`, no parameters | off |
| `mhc_moe_identity_balanced` | `H_res = I`, no parameters | on |

### Recommendation on the identity control: include it. It roughly doubles the runs and it is what makes a positive result mean the thing the hypothesis says.

The brief asked for this to be justified against the budget rather than asserted, so:

**Against the budget it is nearly free.** Two arms at five seeds is 10 cells and $402.40
priced at 20 hours; four arms at five seeds is 20 cells and $804.80. The comparable tranche this team
funded — nine cells at 19 hours on `gpu-4xl40s`, commit `38b66591` — was priced against a
$4,000 ceiling. `edullm check` classifies all three of 10, 20 and 32 cells as `routine`,
released by a team lead, with no denied-outright condition. **The budget is not the binding
constraint on this experiment; throughput per card is.** That single fact is what decides the
recommendation, and it is worth re-deriving rather than trusting: run
`edullm check --json --spec .edullm/run.hc-treatment.yaml ... --fanout-size 20` and read `cost`
and `approval_class` out of the output.

(Those figures are the approval ceiling at `--hours 20`; the expected spend is a little over
half, because the horizon is sized for 12.)

**Against the science it is not optional.** Without the identity arms, the result is
`mhc_moe_balanced` > `mhc_moe`, and the cheapest explanation for that is not the hypothesis: an
auxiliary loss on a set of gates is a regulariser, and regularisers help under-trained models.
The hypothesis is specifically that balancing makes *mixing* start to matter, which is a claim
about an interaction — balancing should buy more where there is a learned matrix to rescue than
where `H_res` is pinned to the identity and there is nothing to rescue. That is H4 in section 6,
and it does not exist in a two-arm design.

**And there is now a second reason, which was not available when the 2x2 was chosen.** The
mechanism turns out to be *caused by the constraint map*, not by the streams: with identical
streams a constrained mixer's gradient with respect to its logits is exactly zero, because every
constraint map's Jacobian annihilates the only direction identical streams produce. The
`identity` arms have no constraint map and no logits at all, so they are the arm in which the
treatment cannot possibly act through the mechanism the hypothesis names. That makes H5 a sharp
control rather than a courtesy: a gain there is balancing doing something else.

**What it costs is power.** A 2x2 interaction has `sum(w^2) = 4` against a simple difference's
2, so its standard error is `sqrt(2)` larger and its minimum detectable effect is 41% worse.
Section 6 carries both. This is the honest trade and the reason the recommendation is not
free: the design buys the *right* question at a 41% worse resolution on it.

### What is deliberately not an arm

- **A no-hyper-connection MoE control.** Stage 1 is exactly that, at the same shape, corpus,
  horizon and seeds, and it runs first for the reason the pre-registration gives: an arm that
  supplies its own noise floor supplies it circularly.
- **`n` other than 4.** `n = 4` is convention rather than a measured optimum — the only real
  sweep, HC Table 1 on OLMo-1B over 500B tokens, has `n = 8` beating `n = 4` on three of four
  LM metrics, and `n = 4`'s single win is a downstream average whose entire margin is COPA, 100
  examples against a roughly 4-point standard error. That is a good experiment and it is a
  different one. Moving `n` here would confound the treatment with the stream count.
- **The other mixers.** `birkhoff` and `kronecker` are in the harness and stay out of this
  tranche for the same reason: one change.

## 5. The primary metric, and why it is not validation loss

**Primary endpoint: does the residual mixer move off its initialisation?**

    D = mean over hyper-connected sub-layers of  ||H_res - H_res(0)||_F / ||H_res(0)||_F

reported at the end of training, with the `H_res` gradient norm beside it as a per-step trace,
normalised by the gradient norm of a reference parameter in the same block (the attention
output projection). Both are logged by the diagnostics in
`src/olmo_core/train/callbacks/hyper_connection_monitor.py`.

Three reasons this is the primary rather than a diagnostic.

**It is the quantity the hypothesis is about.** "Learned mixing does nothing" is, mechanically,
"`H_res` never leaves initialisation and its gradient is nine orders of magnitude below its
neighbours". The prediction under test is that balancing changes that. A loss difference is
downstream of it and confounded with everything else in the model.

**Its effect size is enormous where a loss effect is tiny.** The reference measurement is
`1e-9` against `1.84`. An intervention that moves a gradient norm by three orders of magnitude
does not need a tight noise floor to be visible; a loss intervention worth 0.01 nats does.

**It is falsifiable in the direction that kills the idea.** If `D` moves and the loss does not,
the mechanism hypothesis is refuted rather than merely unsupported: the streams were
un-collapsed, the mixer did start to move, and it still bought nothing. Section 7's decision
rule is written on that.

**Secondary endpoints, reported and not claimed:** final training cross-entropy in nats (see the
limitation in section 8), the stream-collapse metrics — per-stream L2 norm, read-gate and
write-gate concentration, normalised stream-usage entropy — and throughput, peak memory and
step time. The stream-usage entropy is a **manipulation check and not a result**: the treatment
optimises it directly, so a rise in it says the loss is wired up, and nothing more.

## 6. Power, and the plain statement the brief asked for

Every number below is `python src/scripts/ablations/hc_power.py --sweep`, at the planning sigma.
Re-run it with stage 1's measured sigma before submitting stage 2; the thresholds scale
linearly and none of them should be read out of this file after that.

**The planning sigma is 0.0184 nats and it is not a measurement.** It is
`0.010 x (786M / 4.72B)^-0.172 x 1.35`: the middle of the 0.008-0.012 literature range at
this team's earlier 370M/4.72B tranche, moved to this horizon along DataDecide's measured
`sigma ~ D^-0.172`, times a factor of 1.35 for a smaller and sparser model. The 1.35 is a
guess and is written as its own factor so it can be disagreed with separately.

At 5 seeds per arm, 4 arms, pooled df = 16, two-sided alpha 0.05, 80% power:

| hypothesis | contrast | SE | 2·SE gate | MDE |
| --- | --- | --- | --- | --- |
| H1 balancing at learned mixing | `mhc_moe_balanced − mhc_moe` | 0.0116 | 0.0232 | **0.0347** |
| H2 learned mixing, unbalanced | `mhc_moe − mhc_moe_identity` | 0.0116 | 0.0232 | 0.0347 |
| H3 learned mixing, balanced | `mhc_moe_balanced − mhc_moe_identity_balanced` | 0.0116 | 0.0232 | 0.0347 |
| H4 **the interaction** | `(H3) − (H2)` | 0.0164 | 0.0329 | **0.0490** |
| H5 balancing at pinned mixing | `mhc_moe_identity_balanced − mhc_moe_identity` | 0.0116 | 0.0232 | 0.0347 |

### Does the expected effect exceed the MDE? No. Not on loss, at any seed count this budget can buy.

The largest effect the literature attributes to hyper-connections *in total* is ByteDance's
−0.030 nats, at 500B tokens on OLMo-1B. The treatment here is second-order on top of that: it
does not add hyper-connections, it tries to make the mixing matrix inside them useful, and the
same literature says that matrix is currently worth zero or slightly negative. So the honest
prior on H1 is **well under 0.030 nats**, against an MDE of 0.0347 at five seeds and 0.0267 at
ten. Buying seeds does not close it: the MDE falls as `1/sqrt(seeds)` and is 0.0267 at eight seeds
and 0.0237 at ten, so reaching 0.014 needs about 30 seeds an arm — 120 cells — which is still
only a guess at the effect.

**Two caveats on that table, both of which make it an upper bound rather than the number.** The
standard errors are the independent-arm ones and the primary analysis is the *paired*
difference, which can only be smaller; the within-pair correlation that would say how much
smaller has not been measured, and the table is quoted unpaired because that is the honest
direction to be wrong in. And the `2·SE` gate is not the same test as the 5% MDE: at df = 16 a
two-sided `2·SE` threshold is alpha 0.063, not 0.05. Both numbers are reported per contrast so
a reader can apply whichever they want; the gate is `2·SE` and the exact two-sided t p-value is
printed beside every contrast.

Two further reasons to expect less, not more:

- **Scale.** *Review Residuals* (arXiv:2606.31859) found residual-topology changes invisible
  below roughly 500M, with every difference inside noise through 320M. This model has about
  190M active parameters. Against that, MHAR (arXiv:2607.27230) measures −0.149 nats at 350M
  for a depth-routing change, so the effect is intervention-dependent rather than uniformly
  absent — but the prior is not favourable.
- **Horizon.** 786M tokens is a twentieth of Chinchilla for this active size. Architectural
  differences that compound over depth and time have less time to compound.

**So the pre-registered position is that the loss endpoint cannot answer the question, and it
is not the primary endpoint.** It is reported with its interval and its p-value, labelled
under-powered, and no claim of any sign is attached to it. Writing it up as a result would be
exactly the failure this team's own `38b66591` calls out — "a 0.001 single-run loss gap cannot
support a superiority claim" — arrived at with more seeds and the same lack of resolution.

### What the primary endpoint's power is, and the one honest gap

`D`, the mixer's displacement from initialisation, has no published seed variance and stage 1
cannot supply one, because stage 1 has no hyper-connection in it. **That is a real gap and this
document does not paper over it.**

What has changed since this section was first written is that the mechanism is no longer a
hypothesis about a quantity nobody has measured. Two things are now measured, on a CPU, in
seconds, and both are asserted as tests:

- **Why the gradient is ~1e-9.** With the `n` streams carrying the same vector, every
  constrained mixer's gradient with respect to its logits is exactly zero — the constraint map's
  Jacobian annihilates the only direction identical streams can produce. At the uniform doubly
  stochastic initialisation, which averages the streams and so destroys their dispersion, the
  Sinkhorn, Birkhoff and Kronecker mixers get gradient norms of 2e-8, 0.0 and 4e-8 against the
  unconstrained mixer's 2.3 on the same block, the same inputs and the same loss.
  (`hc_moe_block_test.py::test_constrained_mixer_gradient_is_orders_below_the_unconstrained_one`)
- **That the treatment moves it.** Over 200 AdamW steps on a four-block model, against an
  otherwise identical untreated model: stream dispersion 5.9e-08 to 3.0e-03, `H_res` gradient
  norm 1.6e-09 to 2.5e-04, and the ratio against a reference parameter 2.4e-08 to 8.5e-05. The
  degenerate `energy` statistic — the literal mirror of MoE's loss — moves neither, which is the
  negative control. (`stream_balance_test.py::test_balancing_revives_the_mixer_gradient`)

That does not make the endpoint powered on a GPU at this shape, and it is not a claim that the
model is better for it. What it does is turn the manipulation check below from a hope into a
prediction with a measured effect size behind it:

> If, at step 500, the median `H_res` gradient norm in the `mhc_moe_balanced` cells has not
> risen by at least three orders of magnitude over the `mhc_moe` cells at the same step, the
> treatment is not doing what it was built to do. Stop the tranche and fix the implementation.
> Nothing about the idea has been tested at that point and nothing will be claimed about it.

Three orders of magnitude is chosen against the `1e-9` versus `1.84` reference, which is nine.
A statistic whose control value sits nine orders of magnitude below its neighbours does not
need a noise floor estimated to two significant figures to be told apart from one that does
not; what it needs is for the ratio to be reported per seed, which the monitor does.

**Sigma for `D` is estimated from the tranche itself and reported before any H4 claim.** Five
seeds on each of the four arms gives df = 16 for it exactly as for the loss, and the H4
interaction on `D` uses the same 2·SE gate. Until those numbers exist, this document does not
quote an MDE for the primary endpoint, because it would be an invention.

## 7. Pre-registration: the decision rule, written before any run

**Analysis.** All contrasts are differences of arm means over five seeds, tested against
`2 x SE(contrast)` built from the pooled within-arm sigma at df = 16, with the exact two-sided
t p-value reported beside every one. No multiplicity correction is applied to the gate; the five
hypotheses are fixed here in advance and each is reported with its effect size, its interval and
its p-value, so a reader who wants Holm across the family can apply it to a table that has not
been selected on. Cell *k* of every arm shares its three seeds, so the **paired** difference is
the primary form and the unpaired difference of means is reported beside it; the pairing removes
the data-order component and nothing else, so the variance reduction is measured rather than
claimed.

**Homoscedasticity** is assumed by the pooling and is not safe — the treatment is expected to
change how the run behaves, which is the same quantity. Bartlett's test on the four within-arm
variances at alpha = 0.05 is reported whatever it says, with the pre-committed consequence that
a rejection abandons the pooled sigma for unpooled Welch standard errors and Welch-Satterthwaite
degrees of freedom, and that this costs power and is reported as costing it. Bartlett's power at
n = 5 is low; **a pass is not evidence of equal variances and will not be written up as one.**

**The four outcomes, and what each one means.**

| primary `D` and gradient ratio | loss (H1/H4) | reading | what we do |
| --- | --- | --- | --- |
| does not move | anything | the treatment does not do its own job | fix the implementation; claim nothing |
| moves | inside the interval, upper bound below 0.03 | the mechanism happened and bought nothing **detectable at this power** | see the sentence below the table: this is not on its own a refutation |
| moves | H1 clears the gate, H4 does not | balancing helps and the interaction is unresolved | do not claim the mechanism. H5 says whether it is a generic regulariser |
| moves | H1 and H4 both clear the gate | the prediction holds | fund a replication at 1B and a longer horizon before claiming anything |

**What would make us abandon the idea.** An earlier version of this document said: a treatment
that demonstrably un-collapses the streams, demonstrably moves `H_res`, and whose loss interval
contains zero with an upper bound below 0.03 nats, kills the hypothesis. **That rule is not
sound and is withdrawn.** The MDE on that contrast is 0.035, so a real effect of 0.01 nats — a
third of what the whole of hyper-connections is worth in the literature, and a perfectly
interesting result — satisfies it. It is a rule that abandons the idea for being under-powered
against it, which is the failure this document spends section 6 diagnosing in somebody else's
design.

What replaces it is narrower and is what this budget can actually support:

- **The idea is abandoned if the mechanism does not happen.** If the balanced arms do not
  un-collapse the streams and do not move `H_res` off its initialisation, then the claim that
  collapse is what freezes the mixer has been tested by the one instrument that can test it here
  and has failed. That is a mechanism claim answered on a mechanism endpoint, and it is
  conclusive at this budget.
- **The idea is not abandoned on a loss null.** If the mechanism happens and the loss interval
  contains zero, the honest conclusion is that the effect on loss is smaller than 0.035 nats at
  this scale and this horizon, which is reported as a bound and as an argument for a replication
  at 1B, not as a refutation. Turning that into a real equivalence claim needs a stated smallest
  relevant effect and a TOST powered against it, and this design cannot supply one: the smallest
  margin it could reject is larger than the effect the literature attributes to the entire
  method.

**What is fixed here and may not move afterwards:** the arms and their definitions, the primary
and secondary endpoints, the gate, the seed count, the horizon, the corpus, and the four rows
of the table above. What may still move: `--steps`, once the smoke measures step time, applied
identically to every arm; and the planning sigma, replaced by stage 1's measurement.

## 8. The limitation that matters most, named rather than buried

**There is no held-out metric.** `.edullm/train_on_corpus.py` deliberately wires no evaluator —
its header says so and says why — and `regmix-10b-v1` declares no validation split, so there is
no `.val` for an evaluator to read and no per-source breakdown to inverse-variance weight. Every
loss number in this design is therefore **final training cross-entropy**, which is a fair
comparison across arms that share a data order but is not the held-out quantity the gate was
originally written against, and is not a quantity anybody should publish.

The pre-registration this design follows wanted two things out of stage 1: sigma-hat at df = 4,
and per-source inverse-variance weights worth another 1.2-2.9x on sigma at zero compute. **Stage
1 as specified delivers the first and not the second.**

Two routes to closing it, both costed and neither taken here:

1. **Score the checkpoints afterwards.** Every cell saves a checkpoint every 500 steps. A
   separate evaluation submission over those checkpoints produces held-out and per-source
   numbers after the fact, on a cheap shape, with no retraining. This is the cheapest thing that
   would raise what the tranche can conclude and it is what `38b66591` calls the
   checkpoint-scoring job.
2. **Carve a held-out set out of the corpus.** `regmix-10b-v1` publishes 41 shards over seven
   source categories; holding out a fixed subset by name is what `--held-out-shards` does on
   `edullm/hyper-connections-370m`'s `train_hyper_connections.py`. Porting that one file is the
   smallest change that makes stage 1 produce the metric the gate is written against, and it
   would have to be done **before** stage 1 runs, not after.

Both are in `docs/hc-ablation/AGENT-STATUS.md` under DECISIONS NEEDED FROM HUMAN.

## 9. The order to spend the first twelve hours in

0. **The two GPU tests nothing has run**, before any of this:
   `pytest -v src/test/nn/hc_moe_block_test.py -k "real_moe or router_auxiliary"`. They cost
   minutes on any card. The second asks whether the MoE router's auxiliary loss survives the
   hyper-connection's write-out gate; if it does not, every arm below trains with an unbalanced
   router and looks healthy.
1. **The smoke, `.edullm/run.hc-smoke.yaml`, $2.01, about 20 minutes.** Nothing else starts
   until it has printed a summary JSON. It settles step time, peak memory, and whether the
   corpus opens at all. Read `peak_memory_gib` against 22.35 and not 24.
2. **Set `--steps` and `--save-interval` from it** in both the baseline and the treatment spec,
   identically, commit, push. The horizon has to be the same in every cell of both stages.
3. **A four-cell mechanism pilot, about $50, one hour per cell.** One seed of each of the four
   arms at `--steps 500` and `--seeds-per-arm 1`, for the manipulation check in section 6 and
   nothing else. If the `H_res` gradient ratio has not moved by three orders of magnitude, the
   twenty-cell tranche would have measured nothing and this is where that is found out. This is
   the single highest value hour in the plan and it is not in the original brief. Run it with
   `--monitor-interval 10`, because the monitor callback that carries the primary endpoint has
   never run inside a trainer.
4. **Stage 1, `.edullm/run.hc-baseline.yaml`, $201.20 priced, five cells.** The noise floor, and
   the MoE-without-hyper-connections reference. Recompute the power table from its sigma:
   `python src/scripts/ablations/hc_power.py --sigma <measured>`.
5. **Stage 2, `.edullm/run.hc-treatment.yaml`, $804.80 priced, twenty cells.** Only after 3 and
   4 have both reported.

Total priced worst case for the whole plan is about $1,060 at `--hours 20`, of which a little
over half is expected to be spent, against a tranche ceiling this team has previously funded at
$4,000. If more money appears, **buy seeds and not horizon** — `SE ~ D^+0.328` — and buy them on
the arms in H4, which is the contrast that carries the mechanism claim and is the widest of the
five.
