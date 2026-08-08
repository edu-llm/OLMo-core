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

## What the first rehearsal found, in eighteen minutes for about $4

It died, which is what it was for.

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

**The corpus declares no validation split**, so the arms carve one: `--held-out-shards 2`
reserves two of `regmix-10b-v1`'s 41 shards, sorted and taken from the end so every arm and
every seed evaluates on exactly the same data. Without it the only loss in the run is training
loss, and since `--seed` moves the shuffle, its variance across seeds is partly a different
sample of the corpus rather than the run-to-run noise σ is supposed to measure.

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

### Throughput and cost

| shape | arm | measured s/step | measured MFU | hours for 12,715 steps | cost |
| --- | --- | --- | --- | --- | --- |
| gpu-4xl40s | baseline | **not measured** | | | |
| gpu-4xl40s | faithful | **not measured** | | | |

The planning figures are roughly 21 hours and $220 per arm on `gpu-4xl40s` at $10.493/hour, and
about 6 hours and $135 on `gpu-8xa100`. Those are the plan's numbers, not this branch's. Fill
this table from a 200-step probe at 370M — about 2% of an arm, so a few dollars each — before
proposing any full submission.

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

- **Per-lane norm and the spread across lanes.** The primary guard. Identical lane norms mean
  every lane carries the same vector, the model is the baseline with extra parameters, and no
  downstream number is interpretable in either direction. The rehearsal turns this into an
  error.
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
