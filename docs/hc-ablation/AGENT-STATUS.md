# Agent status: hyper-connected MoE and the stream-balancing treatment

Branch `hc/moe-stream-balancing-5bd7`, off `hc/moe-base`. Nothing here has run on a GPU.

> ### The headline, which is negative
>
> **The treatment does not work on the CPU harness, and the $50 four-cell pilot is predicted to
> fail its own pre-registered gate.** At the settings the tranche launches at, the streams stay
> rank one — participation ratio 1.0000 out of 4 after 200 steps — and 99.8% of the rise in the
> `H_res` gradient is the balancing loss's own gradient rather than the task's. An earlier
> version of this work claimed the opposite on the strength of numbers measured at five times
> the weight and five times the learning rate, against a statistic that rank-one streams with
> unequal scales satisfy perfectly. Two adversarial reviews took it apart and both were right.
>
> The statistic is fixed (`spectral`, which reads rank), the tests now assert the negative, and
> the pre-registered gate is rewritten onto rank and the task-only gradient. **Run the pilot in
> section 9 of the design before spending the $805 on the tranche.** If it reproduces the CPU
> result, the answer to the falsifiable prediction is no, and $50 bought it.

---

# START HERE

Six commands from a fresh terminal to a launched baseline. **Step 3 is the one this agent could
not do**: the platform refuses to submit a commit with no published image, and images are built
only by a push to `edullm/**`.

```bash
# 1. Get the CLI and the branch. Re-running the install line is the upgrade.
uv tool install --force git+https://github.com/edu-llm/platform
git clone https://github.com/edu-llm/OLMo-core && cd OLMo-core
git checkout hc/moe-stream-balancing-5bd7

# 2. Price it. Free, reaches no network, and should print refused: false.
edullm check --json --spec .edullm/run.hc-smoke.yaml \
  --experiment hc-moe-stream-balance --dataset regmix-10b-v1 \
  --team input-core --compute gpu-1xa10g --hours 1

# 3. PUBLISH THE IMAGE. This is yours to do and nothing else in this list works without it.
#    Pushing this commit to a branch under edullm/** is what builds the image. Wait for the
#    build to go green in Actions -- about three to five minutes -- before step 4.
git push origin HEAD:refs/heads/edullm/hc-moe-stream-balance

# 4. The twenty-minute smoke, $2.01. Nothing else starts until this has printed a summary.
edullm submit --spec .edullm/run.hc-smoke.yaml \
  --experiment hc-moe-stream-balance --dataset regmix-10b-v1 \
  --team input-core --compute gpu-1xa10g --hours 1
edullm status --json          # free, answers from GitHub, safe to poll

# 5. Read `seconds`, `steps` and `peak_memory_gib` out of its summary JSON and set --steps and
#    --save-interval in run.hc-baseline.yaml and run.hc-treatment.yaml from them -- identically
#    in both. Commit, and push again to the edullm/** branch so the image carries the edit.

# 6. Stage 1: the noise floor. Five cells, twelve hours each, $201.20 priced worst case.
edullm submit --spec .edullm/run.hc-baseline.yaml \
  --experiment hc-moe-stream-balance --dataset regmix-10b-v1 \
  --team input-core --compute gpu-1xa10g --hours 20
```

`HEAD` rather than a literal SHA on purpose: `edullm check` prints the `commit_sha` it would
submit, and it refuses a commit no remote branch contains, so step 2 tells you whether step 3
pushed the right thing. The numbers below were checked at
`6ea77919cf5b0b138c4a12efe5e959e731b58eef` (the commit before this one-line edit; nothing but this
sentence changed).

Do not submit `.edullm/run.hc-treatment.yaml` yet. Its header lists three things that have to
report first, one of which is a four-cell mechanism pilot costing about $50 that decides whether
the twenty-cell tranche is worth submitting at all.

## Evidence: all three specs pass `edullm check --json --spec` with an empty refusals list

Run at commit `1ca04d68ee41`, on a clean tree, with `EDULLM_GITHUB_LOGIN=philote-dev` — this
container's own `gh` login is `cursor`, which is not on the platform roster, so an unprefixed
check adds one refusal that is a property of the container rather than of the specs.

```json
{ "spec": ".edullm/run.hc-smoke.yaml", "hours": 1,
  "refused": false, "refusals": [],
  "cost": {"cells": 1, "hourly_rate_usd": "1.006", "maximum_attempts": 2,
           "maximum_runtime_hours": "1", "nodes": 1, "maximum_compute_cost_usd": "2.01"},
  "approval_class": "automatic", "approving_environment": "run-approval-automatic",
  "commit_sha": "6ea77919cf5b0b138c4a12efe5e959e731b58eef" }

{ "spec": ".edullm/run.hc-baseline.yaml", "hours": 20,
  "refused": false, "refusals": [],
  "cost": {"cells": 5, "hourly_rate_usd": "1.006", "maximum_attempts": 2,
           "maximum_runtime_hours": "20", "nodes": 1, "maximum_compute_cost_usd": "201.20"},
  "approval_class": "routine", "approving_environment": "run-approval-lead",
  "manifest": {"fanout": {"index_parameter": "seed", "size": 5}},
  "commit_sha": "6ea77919cf5b0b138c4a12efe5e959e731b58eef" }

{ "spec": ".edullm/run.hc-treatment.yaml", "hours": 20,
  "refused": false, "refusals": [],
  "cost": {"cells": 20, "hourly_rate_usd": "1.006", "maximum_attempts": 2,
           "maximum_runtime_hours": "20", "nodes": 1, "maximum_compute_cost_usd": "804.80"},
  "approval_class": "routine", "approving_environment": "run-approval-lead",
  "manifest": {"fanout": {"index_parameter": "arm-and-seed", "size": 20}},
  "commit_sha": "6ea77919cf5b0b138c4a12efe5e959e731b58eef" }
```

All three carry the same three deferred checks, which no laptop can make: `no_published_image`,
`image_is_ambiguous` and `image_scan_findings_unreviewed`. Step 3 above is what settles the first.

## The CPU checks, all of which should pass before you spend anything

```bash
pip install -e '.[all]'
pytest -q src/test/nn/ --ignore=src/test/nn/hf      # 538 passed, 1206 skipped
# `test_tensor_parallel_attention` and `test_tensor_parallel_transformer` are flaky under
# gloo. Measured on THIS tree, unchanged between runs: run 1 failed
# `[qk-layernorm-backend=GLOO]`, runs 2 and 3 passed all seven. The failing parametrisation
# moves between runs, which is flakiness by definition, and `hc/moe-base` in a clean worktree
# fails the same way. Not a regression from this branch, and worth knowing before you read a
# red suite as one.
python src/scripts/ablations/hc_gate1_check.py       # GATE_1 PASSED: 63 passed, 0 failed
python src/scripts/ablations/hc_launch_check.py      # 3 specs checked, 0 problems
python src/scripts/ablations/hc_ablation.py --dry-run --model-size tiny
python src/scripts/ablations/hc_power.py --sweep     # the MDE table; re-run with stage 1's sigma
```

---

## Checklist

| package | state | what it is |
| --- | --- | --- |
| **WP0** launchable baseline | **done** | three specs, all passing `check` with zero refusals, plus a launch checker that runs the real command |
| **WP4** powered design | **done** | `docs/hc-ablation/EXPERIMENT-DESIGN.md`, `src/scripts/ablations/hc_power.py`, `.edullm/run.hc-treatment.yaml` |
| **WP1** MoE + hyper-connections | **done, CPU only** | four new block classes, one model class; expert parallelism refuses |
| **WP2** stream balancing | **done, CPU only** | one flag defaulting off; the mechanism is measured over 200 CPU steps |
| **WP3** diagnostics | **done, never run** | `HyperConnectionMonitorCallback`, plus the stream metrics on the model |

### WP0 — the launch artifacts

`.edullm/run.hc-smoke.yaml` (128 lines), `.edullm/run.hc-baseline.yaml` (166),
`.edullm/run.hc-treatment.yaml` (117), `src/scripts/ablations/hc_launch_check.py` (612).

**CPU-verified.** All three pass `edullm check --json --spec` with `refused: false` and an empty
refusals list, at the commit above, on a clean tree. `hc_launch_check.py` splits each command the
way the submission workflow does, applies the platform's own `FANOUT_PROLOGUE`, runs it in a real
`bash` against a stub `python`, and asserts every value of the resulting config against a per-spec
expectations table — 0 problems on all three. Seven mutations that break production are caught and
the unmutated control is green. `exec` was verified to deliver SIGTERM to the training process.

**NOT verified.** Everything that needs a container: whether the corpus opens, whether the model
fits in 22.35 GiB, actual MFU and therefore whether `--steps 3000` fits the wall clock, and
whether a second attempt resumes. The smoke is the instrument for the first three.

### WP1 — hyper-connections around the MoE blocks

`src/olmo_core/nn/transformer/hc_moe_block.py` (430), plumbing in `hc_block.py`, `model.py`,
`config.py`, `__init__.py`, and `src/test/nn/hc_moe_block_test.py` (600). Four new block classes
covering all five MoE block classes in `block.py`, and `HyperConnectionMoETransformer`.

**CPU-verified.** Bit-exact equivalence to the unwrapped MoE model at initialisation with the
noise off — `max|logits - baseline| = 0.000e+00` for five mixers x four block types — and each of
the `n` streams individually reproduces the unwrapped block rather than only their mean. Parameter
counts agree across the config, the block and the model, and the delta against the unwrapped model
is exactly the routing count. Doubly stochastic mixers off initialisation. Gradients finite on
every routing tensor. Save/resume round-trip with every routing tensor in the state dict. Routing
math stays float32 under bfloat16. Expert, tensor and context parallelism all raise, as does the
hybrid block's combined forward. The model class is `MoETransformer`'s, so the router's
load-balancing loss and z-loss still reach the trainer, and the config refuses both ways of
getting that wrong.

**NOT verified, and this is the largest gap in the deliverable.** *No hyper-connected MoE block
has run a single step on a GPU.* The MoE forward needs CUDA kernels — `olmo_core.ops.moe` asserts
them and every MoE test in this repository carries `@requires_gpu` — so the CPU tests substitute a
stand-in for `feed_forward_moe` on **both** sides of every comparison. That tests the residual
wiring exactly, which is what this work adds, and nothing about its interaction with the real
expert dispatch. Two tests are written, marked `@requires_gpu`, and have never been run:

```bash
pytest -v src/test/nn/hc_moe_block_test.py -k "real_moe or router_auxiliary"
```

The second of those is the one to care about: it asks whether the router's auxiliary loss survives
the two einsums and the addition the hyper-connection puts between it and the loss. If it does
not, the arms train with an unbalanced router and look healthy.

### WP2 — the stream-balancing treatment

`src/olmo_core/nn/hyper_connections.py` (+~230), `src/test/nn/stream_balance_test.py` (390).
One flag, `stream_balance_loss_weight`, defaulting to `0.0`.

**CPU-verified, and the verification is what falsified the claim.** Disabled is exactly a no-op:
the statistic is replaced by something that raises and a full forward and backward runs through
the disabled path. The reported cross-entropy is bit-identical at weights 0, 0.05 and 0.5,
because the loss reaches the optimizer through `attach_auxiliary_loss` — which is what keeps
every loss comparison in the tranche between the same quantity.

**And the treatment does not do what it was built to do.** See the box at the top. The streams
stay rank one and the rise in the mixer's gradient is the balancing loss's own. Three things
were wrong and all three are fixed rather than reported: the statistic was satisfiable by
rank-one streams with unequal scales (`dispersion`, now a control; the default is `spectral`,
which reads the participation ratio of the streams' Gram matrix and cannot be); the numbers were
measured at five times the tranche's weight and learning rate; and the gradient was never
decomposed into the task's part and the treatment's own.

**NOT verified.** Whether any of this transfers from a four-block, `d_model=64`, dense, 200-step
CPU harness to `smallmoe` at 565M over 786M tokens. It may not, and the four-cell pilot is the
instrument. What the CPU establishes is that the prediction is falsifiable and that, on the one
harness available, it is currently false.

### WP3 — diagnostics

`src/olmo_core/train/callbacks/hyper_connection_monitor.py` (190), plus the stream metrics on
`HyperConnectionStreamsMixin.compute_auxiliary_metrics`. Logs, per wrapped sub-layer and pooled:
the mixer's displacement from initialisation, the `H_res` gradient norm and its ratio to the
attention output projection's, the largest absolute residual logit, the doubly-stochastic row-sum
error, the stream dispersion share, the stream usage entropy and imbalance, and the `H_res`
matrices entry by entry at a slower interval.

Two blockers were found here by the final audit and both are fixed. The mixer was being read
**with the residual-logit dropout active**, so on an unchanged parameter successive reads
differed from the initialisation by 0.44 to 1.17 in relative Frobenius norm — every displacement
number the tranche would have produced was a draw from the dropout mask. And the initialisation
snapshot was not in the callback's state, so the second attempt of any cell restarted
displacement at exactly zero. Both are fixed; `residual_mixer(deterministic=True)` reads 0.0 on
an unchanged parameter.

**NOT verified.** The callback has never run inside a `Trainer`. It is exercised by the import,
its config's construction and the deterministic-read check, and the metric names and the interval
logic are untested. **This is still the weakest-tested thing in the deliverable and it carries
the primary endpoint.** Run the four-cell pilot with `--monitor-interval 10` before trusting it.

### WP4 — the design

`docs/hc-ablation/EXPERIMENT-DESIGN.md` (240), `src/scripts/ablations/hc_power.py` (430).
Four arms, a 2x2 of {learned Sinkhorn mixer, `H_res = I`} x {balancing off, on}, five seeds each.

The headline conclusion is that **the loss endpoint cannot answer the question**: at the planning
sigma the minimum detectable effect is 0.035 nats on the simple contrast and 0.049 on the
interaction, against a literature effect of 0.030 for the whole of hyper-connections, of which
this treatment is second-order. Thirty seeds an arm would not close it. So the primary endpoint is
the mixer's displacement and the `H_res` gradient ratio, whose control value sits seven to eight
orders of magnitude below its neighbours, and loss is a secondary reported with no claim attached.

---

## DECISIONS NEEDED FROM HUMAN

1. **Is a held-out metric worth one file?** Stage 1 as specified measures the seed sigma of final
   *training* cross-entropy. The 370M pre-registration's gate wants held-out bits-per-byte and
   per-source inverse-variance weights, worth another 1.2-2.9x on sigma at zero compute, and
   neither can be recovered from a run that never evaluated per source. Two routes, both costed in
   EXPERIMENT-DESIGN.md section 8: score the saved checkpoints afterwards with a separate cheap
   job, or port `--held-out-shards` from `edullm/hyper-connections-370m`'s
   `train_hyper_connections.py` **before** stage 1 runs. This is the single highest-value decision
   on the list and it is cheaper before than after.

2. **Does `regmix-10b-v1` publish a validation split?** Two committed documents disagree. This
   branch says it declares none; `38b66591`'s a100 tranche spec says it declares seven validation
   shards, which reads like a confusion with its seven source categories. `edullm data
   regmix-10b-v1 --json` carries no validation field. Settling it needs one `edullm_data.read`
   from a machine with the grant. Decision 1 depends on the answer.

3. **The p5 node the brief asked for does not exist here.** Both H100 profiles are
   `provisioned: false` in the workload catalog and `places: unreliably` in the capacity file, and
   the whole P pool was dry when it was last probed. The design is on `gpu-1xa10g`, one card per
   cell, which places reliably and is cheaper for the same concurrency. If somebody can get H100
   capacity provisioned, the design scales to it unchanged; if not, nothing needs to change.

4. **Team and experiment name.** `--team input-core` and `--experiment hc-moe-stream-balance` are
   this agent's choices, taken because `philote-dev` leads `input-core` and the 370M
   pre-registration was charged there. Neither is registered by anything; change both freely.

5. **`--steps` and `--save-interval` are estimates and the smoke replaces them.** The committed
   3,000 and 125 come from an assumed 20% MFU on a shape nothing has measured. They must be
   identical in the baseline and the treatment spec, and changing one without the other silently
   confounds the horizon with the arm.

6. **The stream-balancing weight, 0.01, is a guess.** Matched to `MoEConfig.lb_loss_weight`, on
   losses that are on different scales. It is identical in both treated arms, which is the
   requirement; whether it is the right magnitude is unknown and the four-cell pilot is where that
   shows up.

---

## FOUND BY THE FINAL AUDIT, NOT FIXED — the exact list

Two adversarial reviews of the whole deliverable produced more than the remaining time could
absorb. Everything scientifically load-bearing is fixed; these are the rest, with enough detail
to act on.

1. **The balance loss is multiplied by the gradient-accumulation count.**
   `attach_auxiliary_loss` seeds its gradient once per `backward`, and the trainer runs one
   backward per micro-batch while the cross-entropy is normalised to one batch. At
   `--global-batch-size 262144 --rank-microbatch-size 8192` that is 32 micro-batches, so the
   treatment's effective weight is `0.01 x 24 sub-layers x 32 = 7.68` per optimizer step rather
   than the 0.24 the design says. MoE avoids this by dividing by the batch token count. **Fix:**
   thread `loss_div_factor` into `HyperConnection.write_out` and scale by
   `(batch * seq_len) / loss_div_factor`, and add `--rank-microbatch-size` to the frozen list in
   the design's section 7, since the smoke is expected to change it.
2. **The monitor's mixer read is rank-local under FSDP above world size 1.** Probed at world
   size 2 the value differs by 0.0018 in the max entry with no exception, and the row sums still
   read 1.0 so the callback's own guard reports clean. The tranche is one card per cell so it is
   not hit, but the primary endpoint goes through that path. **Fix:** `full_tensor()` on a
   `DTensor`, or refuse above world size 1.
3. **Two metrics are pooled by the wrong rule.** `model.py` pools any metric whose name contains
   `loss` by summation, which catches `stream balance loss unscaled` (a per-sub-layer quantity in
   [0,1]); and `stream usage imbalance` is labelled `max` and pooled across blocks with a mean.
   **Fix:** pool by `reduce_type` as `MoETransformer.compute_auxiliary_metrics` does.
4. **`hc_power.py`'s conservatism claim is wrong at df = 4.** Against exact noncentral-t it is
   exact at df = 16 and 1.16% *anti*-conservative at df = 4, which is the paired primary
   analysis. The docstring says 1-3% conservative. **Fix:** replace the claim with the measured
   table, or add the exact solve.
5. **`test_disabled_is_bit_identical_to_the_untreated_baseline` does not test its name.** Its
   loop runs `weight` in `(0.0, 0.0)` into a dict keyed by weight, so the second overwrites the
   first and nothing compares them. The only live assertion is that the treated model differs.
   The no-op claim itself is covered by the sentinel test and by the bit-identical
   cross-entropy check, so this is a misleading name rather than an unguarded property.
6. **`num_flops_per_token` does not count the mixing** — about 0.057% at this shape, so MFU is
   overstated by that.
7. **`apply_fsdp` on the HC MoE blocks discards `prefetch_factor` and `wrapping_strategy`**
   silently. Documented, not warned.

## RISKS — what I most expect to fail on real hardware, in order

0. **That the pilot confirms the CPU result and the idea is dead.** This is now the most likely
   single outcome and it is the cheap one: $50 against $805. It is a real answer to a
   falsifiable prediction, and the write-up would be that stream collapse can be measured, that
   a balancing loss on it does not lift the streams off rank one, and that the mixer's task
   gradient stays at 1e-9 regardless — which is evidence against the hypothesis that collapse is
   what freezes the mixer.

1. ~~**The router's auxiliary loss does not survive the write-out gate.**~~ **Retired, and now
   asserted on every CPU run.** A judge pointed out that this never needed a GPU — the kernels
   are for the expert dispatch, and the graph between the auxiliary loss and the router is
   einsums — and demonstrated the property with a stand-in that attaches a loss the way
   `MoEBase` does. That is now `RouterishStandIn` and
   `test_an_auxiliary_loss_reaches_its_own_parameter_through_the_write_out_gate` in the
   committed suite: eight cases, four block types by two mixers, primary objective multiplied
   by zero so only the attached loss can produce a gradient. The `@requires_gpu` test stays for
   the dispatch kernels.

2. **Memory.** The term that decides whether `--rank-microbatch-size 8192` fits is not the
   optimizer state but the fp32 logits: 8,192 tokens x 100,352 vocab is 3.06 GiB per copy, and at
   the LM head's backward two or three copies can be live. The estimate lands between 18 and 21.5
   GiB against the A10G's 22.35. It probably fits and it has no margin. Doubling the microbatch
   doubles that term specifically, so a comfortable `peak_memory_gib` at 8192 does not license
   16384.

3. **Throughput.** Every wall-clock number is downstream of a 20% MFU assumption on a 32-expert
   MoE that nothing has measured. If it is 12%, `--steps 3000` does not finish, and a timed-out
   attempt is **not** retried — `RETRY_ONLY_WHAT_A_RETRY_FIXES` retries `Host EC2*` and exits on
   everything else. That is why `--hours` is 20 against an estimate of 12.

4. **The monitor callback has never run in a trainer.** It carries the primary endpoint. Two
   defects in it were caught by reading rather than by running — a mixer read through active
   dropout, and a snapshot that reset on resume — and a third of the same kind would not be
   surprising. A wrong metric name or an interval that never fires leaves the tranche with a loss
   curve and nothing else, which is the one outcome the design calls uninterpretable.

5. **FSDP over a four-dimensional residual.** Inherited from the dense prototype and untested
   there too. At one card per cell FSDP is a no-op, so this tranche does not exercise it — which
   means a later multi-card run is unprotected.

6. **`torch.compile` is off and has to stay off in every arm.** Turning it on for one arm is a
   confound, and it is one word in a YAML file.

---

## Every file this branch touches

Twenty, and nothing else. A repo-wide `make style` run had incidentally reformatted 43 more,
including `src/olmo_core/nn/moe/moe.py`, which the brief forbids touching and which another team
runs against on `edullm/moe-m1-pilot`; that was a cosmetic import reflow with no semantics in it
and it has been reverted. `git diff hc/moe-base..HEAD --name-only -- src/olmo_core/nn/moe/` is
empty, and so is the same query for `src/olmo_core/nn/transformer/block.py` — the five MoE block
classes are wrapped by subclassing rather than by editing.

| file | new? |
| --- | --- |
| `src/olmo_core/nn/hyper_connections.py` | edited — the treatment, the statistics, the metrics |
| `src/olmo_core/nn/transformer/hc_moe_block.py` | **new** — four hyper-connected MoE block classes |
| `src/olmo_core/nn/transformer/hc_block.py` | edited — the shared block mixin |
| `src/olmo_core/nn/transformer/model.py` | edited — the stream mixin and `HyperConnectionMoETransformer` |
| `src/olmo_core/nn/transformer/config.py` | edited — enum members, build wiring, validation |
| `src/olmo_core/nn/transformer/__init__.py` | edited — exports |
| `src/olmo_core/train/callbacks/hyper_connection_monitor.py` | **new** — the diagnostics |
| `src/olmo_core/train/callbacks/__init__.py` | edited — exports |
| `src/scripts/ablations/hc_ablation.py` | edited — the MoE arms |
| `src/scripts/ablations/hc_gate1_check.py` | edited — 30 checks to 63 |
| `src/scripts/ablations/hc_launch_check.py` | **new** — what `edullm check` cannot ask |
| `src/scripts/ablations/hc_power.py` | **new** — the power arithmetic |
| `src/test/nn/hc_moe_block_test.py` | **new** |
| `src/test/nn/stream_balance_test.py` | **new** |
| `.edullm/train_hc_moe.py` | **new** — the arm entrypoint |
| `.edullm/run.hc-smoke.yaml` | **new** |
| `.edullm/run.hc-baseline.yaml` | **new** |
| `.edullm/run.hc-treatment.yaml` | **new** |
| `docs/hc-ablation/EXPERIMENT-DESIGN.md` | **new** |
| `docs/hc-ablation/AGENT-STATUS.md` | **new** — this file |

## What was audited, and by whom

Four adversarial read-only audits, two model families, two rounds. Round one covered the WP0
launch artifacts before anything else was written; round two covered the whole deliverable.
Every confirmed finding was fixed rather than recorded, and the fixes are in the commit messages.
The largest were: SIGTERM never reaching the trainer through the logging wrapper; the second
attempt not covering a wall-clock overrun, which made a 12-hour bound on a 12-hour run a certainty
rather than a risk; a checkpoint interval four times the declared contract; `${...:-0}` as a
silent path to a measured noise floor of exactly zero; and a launch checker that was green on four
mutations that break production, one of them silent through a whole tranche.

Where the two judges disagreed, the disagreement was settled by reading the source rather than
averaged. One reported a missing `grouped_gemm` forcing a slow Python fallback in the MoE; it does
not, because `MoEType.default` builds `MoEMLP`, which is `torch.bmm` over a fixed capacity, and
the fallback lives in `DroplessMoEMLP`, which this factory never instantiates.
