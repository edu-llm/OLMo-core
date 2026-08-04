# The three submissions, in order

Run them in this order. Each one's output is an input to the next, which is why they are not
one submission with three phases.

| # | what | cells | expected | ceiling | wall |
|---|---|---|---|---|---|
| 1 | throughput probe | 1 | ~$5 | $10.98 | ~15 min |
| 2 | LR probe | 3 | ~$105 | $165 | ~4.8 h |
| 3 | main grid | 18 | ~$632 | $988 | ~29 h |

**Total: ~$742 expected. The largest ceiling an approver sees is $988.**

All three are `gpu-8xa100` (p4d.24xlarge, $21.9576/hr). Every one is EXCEPTION class and needs
`run-approval-admin` — not because of the cost but because the rate exceeds
`EXCEPTION_RATE_CEILING_USD_PER_HOUR = 20` (`contracts/policy.py`), which no A100 submission can
ever satisfy. A cheaper A100 run takes the identical approval path, so there is nothing to buy
by shrinking one.

## Why this order, and not in parallel

**1 → 2:** the throughput probe is the only thing that has ever executed 8-rank FSDP, the
8-rank checkpoint write, or this model on an A100 at all. Two of this project's three
run-killing bugs lived in exactly those paths (a DTensor init crash, and a 48-minute checkpoint
hang). It also replaces a ±30% extrapolation: the last time throughput was extrapolated here it
was **2.7x optimistic**, and every wall-clock and cost number below rests on the A100 figure.

**2 → 3:** the LR probe's result *changes the main grid's settings*. All arms currently train at
one LR, and `experiments/liv/research/02_lowrank_gates.md` establishes that a factorized layer
does not inherit a tuned dense LR. If `F-r128` wants a different value, running the grid first
means running it at the wrong one and paying twice.

It is also the only thing that makes a null attributable. A flat result has four candidate
causes: the energy proxy does not transfer, the budget is too short, one arm sits off its LR, or
4% of parameters cannot move CE. The held-out ladder resolves the second. Nothing else in the
design resolves the third, and the repo's own research notes rank it the **#1 validity threat,
above statistical power**.

## What the probe cannot tell you

3 arms x 1 seed at a second LR detects an arm that is *badly* off. It does not
locate each arm's optimum -- that is a real sweep, 3 arms x 3+ LRs, and it costs triple. This
is the cheap version on purpose: it catches the failure that would invalidate the grid without
pretending to tune.

## Numbers that are shared across all three

- **Global batch 524,288 tokens** (128 x 4096). Restored from the 131,072 an earlier draft used.
  `3e-4` and the 0.0105-nat noise floor were both calibrated at 524,288, and
  `src/test/edullm_train_liv_arm_test.py` asserts this value -- the earlier draft disagreed with
  the file's own test.
- **`--save-interval` never divides `--steps`.** When it divides, the terminal checkpoint is
  claimed by the interval save and routed through the async path, which stages the whole state
  dict to host RAM twice. That hung `run_019fbfbe` for 48 minutes with no traceback. Every
  command below is checked: 762 % 200 = 162, 60 % 50 = 10.
- **`maximum_attempts=1`.** A retry doubles the declared ceiling to pay for a resume that only
  fires on a lost machine, and there is a known unguarded defect on restarting a *finished* run.
- **`--no-compile-model`.** Removes a variable while the A100 path is unproven. Costs throughput,
  which is one reason the probe's measurement matters before the grid inherits it.

## 1. Throughput probe

Answers: real A100 tok/s, whether 8-rank FSDP initialises, whether the 8-rank checkpoint writes,
how large that checkpoint is, and container-start overhead. Uses `olmo-core-train` rather than
`olmo-core-check` so the checkpoint contract is actually exercised -- `olmo-core-check` declares
`checkpoint: null`, which would skip the one path that has never run.

```bash
gh workflow run submit-run.yml --repo edu-llm/platform --ref main \
  -f repository=OLMo-core \
  -f commit_sha=<HEAD of edullm/liv-p1-gate-structure> \
  -f workload_profile=olmo-core-train \
  -f compute_profile=gpu-8xa100 \
  -f team=scratch -f experiment=liv-p1-throughput-probe \
  -f dataset_release=olmo-150b-dolma2-v1 -f wandb_project=eduLLM \
  -f maximum_runtime_hours=0.5 -f maximum_attempts=1 \
  -f command='bash -lc '"'"'python .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --prepare-heldout-only && python -m torch.distributed.run --nproc-per-node=8 --standalone .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --arm L0 --arm-seed 0 --data-seed 0 --steps 60 --save-interval 50 --warmup-steps 10 --sequence-length 4096 --global-batch-size 524288 --rank-microbatch-size 8192 --no-compile-model --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

31.5M tokens. The 0.5h bound caps the loss at $10.98 in exactly the failure mode that has
already happened once.

**Read out of it:** `throughput/device/TPS` in W&B, the `peak_memory_gib` and `wall_seconds` in
the stdout JSON summary, and `checkpoint/save_duration_s`. Then recompute every row below.

## 2. LR probe

Three arms, one seed, at a second learning rate.

**`--steps 762`, the same as the main grid, and NOT 381.** An earlier draft used 381 to halve the
cost and it was triple-confounded: `CosWithWarmup`'s period is `max_duration`, so a 381-step run
is not "the same run at a lower LR" -- it is a different cosine. At step 381 a 381-step schedule
has already decayed to its `alpha_f` floor while the 762-step grid is still mid-cosine, an ~11x
LR difference at the identical step, on top of half the token budget. Nothing about arm-vs-LR
could be read out of that.

Same steps, same schedule, one variable. Read the comparison at the ladder rungs against the
grid's `3e-4` cells.

```bash
gh workflow run submit-run.yml --repo edu-llm/platform --ref main \
  -f repository=OLMo-core \
  -f commit_sha=<same commit> \
  -f workload_profile=olmo-core-train \
  -f compute_profile=gpu-8xa100 \
  -f team=scratch -f experiment=liv-p1-lr-probe \
  -f dataset_release=olmo-150b-dolma2-v1 -f wandb_project=eduLLM \
  -f maximum_runtime_hours=2.5 -f maximum_attempts=1 \
  -f fanout_size=3 -f fanout_index_parameter=arm \
  -f command='bash -lc '"'"'python .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --prepare-heldout-only && python -m torch.distributed.run --nproc-per-node=8 --standalone .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --fanout-grid L0:0,F-r128:0,G-grouped:0 --steps 762 --save-interval 200 --warmup-steps 15 --sequence-length 4096 --global-batch-size 524288 --rank-microbatch-size 8192 --learning-rate 1.5e-4 --no-compile-model --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

`1.5e-4` is the sqrt-scaling neighbour of `3e-4`, i.e. a deliberate factor-of-2 step rather than
a guess. **What you are looking for is not "which LR is better" but whether the ARMS DISAGREE
about it.** If all three move the same direction by a similar amount, one LR is fine and the
grid is safe. If `F-r128` prefers `1.5e-4` while `L0` prefers `3e-4`, a single-LR grid would
have measured optimization, not gate structure, and the grid needs per-arm LRs.

## 3. Main grid

3 arms x 6 seeds = 18 cells, 400M tokens each, 7.2B total.

```bash
gh workflow run submit-run.yml --repo edu-llm/platform --ref main \
  -f repository=OLMo-core \
  -f commit_sha=<same commit> \
  -f workload_profile=olmo-core-train \
  -f compute_profile=gpu-8xa100 \
  -f team=scratch -f experiment=liv-p1-gate-structure \
  -f dataset_release=olmo-150b-dolma2-v1 -f wandb_project=eduLLM \
  -f maximum_runtime_hours=2.5 -f maximum_attempts=1 \
  -f fanout_size=18 -f fanout_index_parameter=arm-and-seed \
  -f command='bash -lc '"'"'python .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --prepare-heldout-only && python -m torch.distributed.run --nproc-per-node=8 --standalone .edullm/train_liv_arm.py "$EDULLM_RUN_ID" --fanout-grid L0:0,L0:1,L0:2,L0:3,L0:4,L0:5,F-r128:0,F-r128:1,F-r128:2,F-r128:3,F-r128:4,F-r128:5,G-grouped:0,G-grouped:1,G-grouped:2,G-grouped:3,G-grouped:4,G-grouped:5 --steps 762 --save-interval 200 --warmup-steps 15 --sequence-length 4096 --global-batch-size 524288 --rank-microbatch-size 8192 --no-compile-model --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

**Set `maximum_runtime_hours` from the probe, not from this file.** 2.5 is a placeholder carrying
~55% headroom over the extrapolated 1.6 h/cell. With a measured number, use
`ceil(measured * 1.3 * 2) / 2`.

**Seeds are paired:** `--fanout-grid` sets `--arm-seed`, and the entry point sets
`data_seed = arm_seed`, so seed *s* fixes model init and batch order identically across all
three arms. That is what makes the paired comparison valid and the resolution arithmetic honest.

**Held-out ladder** fires at steps 38/76/152/266/381/571 (20M-300M tokens) plus the final step,
against 4 of the corpus's 60 val shards. The endpoint is the *trajectory* of the between-arm
gap, not the final value. Read it against `optim/LR (group 0)`: cosine decays to `alpha_f = 0.1`
of peak (3e-5, not zero), so the tail is damped and a curve flattening there is weak evidence of
convergence at best -- the interior rungs, where LR is still near peak, are the
ones that can distinguish a real null from undertraining.

## Dropped from the earlier 4-arm draft, and why

- **`N-narrow`** answers "why not just build a smaller dense model?" That only becomes a live
  question if there *is* an effect. Deferring it funds 3 more seeds, which improves resolution
  2.1x -- the df=2 t-penalty is severe, so seeds buy more than the old `2*SD/sqrt(n)` formula
  implied.
- **`A16-P`** (all-attention) is the closest thing to "beats a plain transformer", and it is not
  worth buying: it gets 1.27x `L0`'s FLOPs at 4K, so an `A16-P` win cannot be attributed to
  topology rather than compute. Hybrid-vs-all-attention at equal params is also already
  established (Mamba-2, Jamba, Falcon-H1).

**No arm here is an OLMo baseline.** Nothing in this grid answers "beats base OLMo" literally.
The claim it supports is about gate structure inside the LIV block.
