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
dolma2 tokens at sequence length 4096, which is 12,715 steps of a 786,432-token batch. About
2.2e19 FLOPs per arm.

## Arms

Generated from `hyper_connection_arms.ARMS`; the tests assert these properties rather than
trusting the table.

| # | arm | seeds | params | vs baseline | FLOPs/token vs baseline |
| --- | --- | --- | --- | --- | --- |
| 1 | `baseline` | 3 | 474,022,912 | +0.0000% | +0.0000% |
| 2 | `faithful` | 3 | 474,220,352 | +0.0417% | +0.0994% |
| 3 | `output-only` | 1 | 474,187,456 | +0.0347% | +0.0908% |
| 4 | `no-output-init` | 1 | 474,220,352 | +0.0417% | +0.0994% |
| 5 | `decay-everything` | 1 | 474,220,352 | +0.0417% | +0.0994% |
| 6 | `n1` | 1 | 474,121,376 | +0.0208% | +0.0119% |
| 7 | `n2` | 1 | 474,154,304 | +0.0277% | +0.0324% |
| 8 | `n8` | 1 | 474,353,216 | +0.0697% | +0.3371% |
| 9 | `mhc` | 3 | 474,220,352 | +0.0417% | +0.0994% |
| 10 | `tied-faithful` | 1 | 339,871,136 | −28.3007% | +0.0994% |
| 11 | `tied-baseline` | 1 | 339,772,416 | −28.3215% | +0.0000% |

Every untied arm is iso-parameter to within 0.07% and iso-FLOP to within 0.34%. The tied arms
are deliberately not iso-parameter — that is what they test — but they are iso-FLOP with their
own control, because they are matched on *effective* depth: 16 layers running 8 distinct
blocks twice on a cycle.

Fifteen runs in total once seeds are counted. Cut order if the budget does not stretch: `n8`,
then `n2`, then `tied-faithful` and `tied-baseline` as a pair.

## Pre-registered hypotheses

- **H1 (replication).** Arm 2 beats arm 1 by ≥0.025 nats.
- **H2 (the artifact).** Arm 2 > arm 3. If arm 3 reproduces the published degradation and arm 2
  does not, the field's negative result is an implementation artifact, and that is the headline.
- **H3 (initialization).** Arm 2 > arm 4; the output-init scaling is load-bearing rather than
  cosmetic.
- **H4 (the seesaw).** Arm 6 ≈ arm 1 or worse. ByteDance found n=1 does not help; if it helps
  here, their mechanism story is incomplete at this scale.
- **H5 (constraint).** Arm 9 ≥ arm 2, with the gap larger wherever arm 2 is unstable.
- **H6 (reuse).** Arm 10 − arm 11 > arm 2 − arm 1. Lane value tracks parameter reuse, so the
  effect is larger when the same parameters run twice.

**Decision rule, fixed in advance.** σ is measured from arm 1's three seeds. Nothing under 2σ
gets claimed. Downstream is reported alongside bits-per-byte, because loss and downstream
decouple by 6 to 16 points for changes in this class and a loss-only readout can miss a
catastrophe.

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
0.00064 at the first logged step and rises to a median of 0.0298 — twenty-five times the 1e-3
floor the run would have failed closed at. Starting near zero is the initialization equivalence
of eq. 14 confirmed in a real run rather than in a unit test: at step zero every lane holds the
same vector, exactly as the ordinary residual stack would, and they separate once the mixing
matrix moves. The mechanism is live, not inert.

**The spectral radius is already above 1.** ρ(A_r) on block 0's attention stream reads 1.001 at
the first step — the identity, as initialized — and climbs to 1.196 by step 200. Parcae's
signature for a diverging run is ρ ≥ 1, and Tencent's 3B divergence had a multi-lane drift.
Two hundred steps of a 96M model predicts nothing about a 370M run, and this is exactly the
quantity the instrumentation exists to watch. It is also the sharp prediction for arm 9: mHC
pins ρ at exactly 1 by construction, so if unconstrained HC drifts and mHC does not, H5 has a
mechanism rather than a correlation.

**Bits-per-byte arrives per source**, seven of them: arxiv 1.66, algebraic-stack 1.75,
open-web-math 1.74, starcoder 1.98, dclm 2.03, pes2o 2.06, wiki 2.17 at step 200. Early and
therefore high, but already spread by a wide enough margin that a pooled average over them
would be the wrong statistic.

### What analysing it changed

Three findings from the run's own telemetry, two of them bugs in this branch.

**The monitor was reading the evaluator's forward pass.** The hook fires on every forward, and
the held-out evaluation runs in `post_step` — so on eval steps the lane norms came from padded
held-out sequences rather than the training batch, reading 11% to 50% low. Worst at step 200,
which is the value that lands in the run summary: block 02's spread reads 0.0237 there against
a true 0.0470 ten steps earlier. Gating on `module.training` would not have worked, because the
evaluator's own `self.trainer.model.eval()` line is commented out. It now gates on being
between `pre_step` and `post_train_batch` instead.

**The fail-closed floor was unreachable.** 1e-3 sat below anything the run ever produced —
6.4e-4 while the lanes were still separating, 2e-2 to 4e-2 once they had. It could only have
caught total failure. Now 5e-3: an order of magnitude above the inert reading, four times below
the working one.

**z-loss was off, though the configuration calls for it.** `train_on_corpus` never sets
`z_loss_multiplier`, so `train/Z loss` was never written. That matters more here than usual:
RMSNorm readouts are scale-invariant, so cross-entropy cannot see hidden-state scale at all,
and the rehearsal's hidden norms rose 50% and then gave back a third with nothing in the loss
curve reflecting either move. Now on at 1e-5.

Two more worth recording without a code change. Lane differentiation **peaks mid-run and then
retreats**, correlating with the learning rate at r = 0.72 — so a short run's endpoint
understates it, and thresholds should not be ported from this rehearsal to a differently-shaped
schedule. And **block 0's attention spectral radius is the only one of sixteen that never turns
over**: every other block peaks by step 160 and decays, while block 0 climbs monotonically to
1.196. It sits at the input end, where its amplification compounds through everything above it.
That is the one to watch at 370M.

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
| per-seed σ, validation BPB | **not measured** |
| per-seed σ, downstream average | **not measured** |
| minimum detectable effect at 3 seeds, 80% power | **not measured** |

The estimate this plan was written against is σ ≈ 0.008–0.012 nats, giving an MDE near 0.025
nats. That is an estimate from the literature, not a measurement of this configuration, and the
three baseline seeds exist to replace it. **No treatment arm is submitted until this table has
numbers in it.**

### Throughput

Measured, not planned. Read from run history at the steady state rather than from the run
summary — the last logged value is taken during the end-of-run evaluation, where the model is
not training and throughput reads near zero.

| shape | config | steps | MFU (median) | TPS (median) | source |
| --- | --- | --- | --- | --- | --- |
| gpu-4xl40s | `hc_rehearsal`, `faithful` | 200 | 7.89% | 49,620 | `run_019fdfe9-e6c0` |
| gpu-4xl40s | `hc_370M` | — | **not measured** | | |

The rehearsal figure is not a prediction for 370M: it is a 96M model of which 77M is embedding
and unembedding, so it spends an unusually large share of its time in two matmuls that do not
grow with depth. The 370M probe is what fills the row below it.

The planning figures for a full arm remain roughly 21 hours and $220 on `gpu-4xl40s`, and about
6 hours and $135 on `gpu-8xa100`. Both are the plan's, not this branch's.

### Noise floor

Still not measured. See above — it needs arm 1's three seeds.

Two notes on reading throughput numbers here. OLMo-core v2.5.0 fixed an A100 peak-FLOPs
constant in `SpeedMonitorCallback` that was 2× too low and had been inflating reported MFU by
2×, so any figure from before that is wrong by a factor of two; this branch is on v2.5.0. And
`num_flops_per_token` here counts the hyper-connection cost, so MFU across arms is comparable.

### Compute reality

`gpu-4xl40s` is lead-approved at this duration and cost, with a 19-minute median queue.
`gpu-8xa100` is cheaper per run but its hourly rate is above the $20 admin threshold, so it
needs an admin at any duration and the median wait is 89 minutes. A fan-out never self-releases
regardless of cost, so submitting the seeds or the treatment arms as an array will hit the gate.
`gpu-8xb200` is priced but capacity-block-backed with nothing purchased: a submission is
admitted and then dies at Batch on a queue that does not exist. Not an option.

## Order of operations

1. **Rehearse.** `.edullm/run.yaml` as committed: `faithful` at the rehearsal size, 200 steps,
   `gpu-4xl40s`, roughly $4. It fails closed if the lanes have not differentiated by step 150.
2. **Probe throughput** at 370M for 200 steps per distinct shape, and fill the table above.
3. **Three seeds of arm 1.** Fill the noise floor table above.
4. **Only then** the treatment arms, core first: 2, 6, 9, then the forensic arms 3, 4, 5, then
   7, 8, 10, 11.

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
