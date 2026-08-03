# Impl 3×5 — what this build is

`README.md` says what the experiment asks. This file records what was built, what was verified
before spending GPU time, and every place the build knowingly departs from what it inherits.
Read it before quoting a number from `runs/`.

## Status: implemented and locally validated. NOT YET RUN.

No training has happened. Everything below about the objective is **verified**; everything about
results is **absent**. There are no numbers in this directory yet.

What is verified, locally, on CPU, against the platform's exact pins (transformers 5.14.1,
peft 0.20.0, datasets 5.0.1, torch 2.13.0):

| check | result |
|---|---|
| **W1** `m ≡ 1` reproduces the stock loss | **0.000e+00** — bitwise, with `--loss_denom global` |
| W1 discriminates | `microbatch` differs by 6.94e+01, so the check is not vacuous |
| **W3** multiplier↔label alignment through padding + causal shift | **0.00e+00** over every label position; weights of 1e6 on padding contribute exactly 0 |
| **W2** mean-1 over pedagogy tokens | worst \|mean−1\| = 2.2e-16 across 4 signal shapes × 5 temperatures |
| **W6** T→∞ limit | ESS(451) = 0.99997, mean\|m−1\| = 0.0031, deviation O(1/T) confirmed |
| unit tests | 29 passed |
| end-to-end smoke (precompute → cache → attach → weighted train, both variants) | passed |
| `ruff check --no-cache` (pinned 0.15.22) | clean |

## The objective

IMPL3_HANDOFF §4.1, unchanged:

```
m_t = N_ped · softmax_ped(−z(s_t)/T)      for pedagogy tokens
m_t = 1                                    for general (replay) tokens

variant a:  s_t = −log π₀(y_t | ctx)
variant b:  s_t = KL(π₀(·|ctx_t) ‖ π_SFT(·|ctx_t))
```

z is a robust z-score (median / MAD × 1.4826) computed **once, globally**, over every pedagogy
label token in the corpus.

### This is a reimplementation, not a port

**James's `common/weighting.py` and `common/sft_train.py` are not in the handoff bundle.** It
ships `common/kl.py`, `common/system_instructions.py`, `common/prompts/`, the eval tree and the
results file — but not the weighting or the trainer. §4.1 specifies the objective completely
enough to rebuild, so `klw/weighting.py` is written against the spec with the constants quoted
from it.

The consequence to state plainly: **this is spec-equivalent, not code-identical.** The mitigation
is arm `bT451` — if the reimplementation matches his, T→∞ lands on D4, exactly as his `b-T451`
landed on his SFT baseline to within 0.002. That is the check that prices the reimplementation,
and it is why the control is not optional.

## Deviations from what this inherits, and why

| # | Inherited behaviour | This build | Why |
|---|---|---|---|
| 1 | §4.1: variant b's reference is `checkpoint-923`, "the POC's Impl-2 adapter" — "keep this fixed" | **D4's `ckpt-923`** | §1 defines the reference as "a vanilla SFT run on identical data". On Impl 5's mix that *is* D4 — vanilla SFT, no reweighting, on exactly this training file. The gold Impl-2 adapter would measure how far gold-SFT moved on contexts it never saw. Fixed across every b arm, as §4.1 requires. |
| 2 | Impl 3: `RandomSampler`, log-spaced 12-point grid, `per_device_batch 32 × accum 1` | **Impl 5's**: `SequentialSampler`, 24/8 blocks, 22-point grid, `8 × 4` | The baseline is D4, not James's SFT run. Every one of these held at D4's is what makes the multiplier the single moving part. |
| 3 | Impl 3: `gradient_checkpointing` **off** ("H200 has memory to spare; ~30% faster") | **on** | D4 ran with it on. Activation recompute is a numerical no-op only if it reproduces the dropout masks exactly, and LoRA dropout is 0.05. Not worth a second variable; the wall clock is recovered by arm-level parallelism instead. |
| 4 | HF default `remove_unused_columns=True` | **False** | Required, not chosen. True strips the `weights` column *before* the collator and the arm trains unweighted at full strength with no error and a plausible loss curve. `assert_weighting_ran()` fails the run if no batch ever carried weights. |
| 5 | Impl 5 deviation 4: replay slot byte-identical to A1's, ped:gen token ratio +8.9% | **inherited unchanged** | Every arm reads D4's file, so the drift is common-mode across the whole contrast rather than a difference within it. Still has to be reported. |
| 6 | Impl 3's `math` grid: 12 log-spaced steps | **the same 11-step subset impl4 used** | D4 is the baseline and a step D4 never measured has no baseline. |

## GPU utilisation

The instruction was to use all available GPU capacity. D4 ran on **one** L40S of a `gpu-4xl40s`
and left three idle, so there was real headroom. Where it was taken from, and where it was
deliberately not:

**Not taken from bigger micro-batches.** `per_device_batch 8 × grad_accum 4` is A1's and D4's.
`impl4_ssd/probe_loss_norm.py` exists because `transformers>=4.48` normalises the loss by
`num_items_in_batch` across the whole accumulation group, and that makes stream weight
token-proportional. Regrouping the micro-batches changes what each example contributes to a step
whenever a group's token counts are uneven — and pedagogy rows (~190 label tokens) and Tülu rows
(~80) are very uneven. Impl 5 *inherited* its loss-normalisation acceptance check from A1 on the
grounds of "same recipe, same pins, same PEFT wrapping"; changing the batch shape voids that
inheritance and puts a second variable into a one-variable contrast. `train_sft_klw.py` refuses
a different batch shape unless `--allow_batch_change` is passed.

**Taken from concurrency instead**, which changes no arithmetic anywhere:

| stage | serial | here | speedup on 4 GPUs |
|---|---|---|---|
| precompute | one pass over 22,152 rows | 4 round-robin row shards + a merge | ~4× |
| train | 4 arms back to back | 4 arms, one per GPU | ~4× |
| ped_nll / math | 4 arms back to back | 4 arms, one per GPU | ~4× |

Each job gets `CUDA_VISIBLE_DEVICES` set to one device and sees it as `cuda:0`. Nothing is
distributed; it is four independent single-GPU jobs, which is why the numbers are unaffected.

**The arm count was chosen to match the GPU count.** Three conditions plus the `bT451` control
fills a `gpu-4xl40s` exactly, which is how the control became free rather than a fourth training
run nobody would pay for.

Two further savings in the precompute, both structural:

- **Both variants come out of one pass.** Variant a's `−log π₀(y_t)` is a gather from the same
  base distribution variant b reduces to a KL, so `aT8` costs no extra forward.
- **π₀ and π_SFT come from one set of weights** via `PeftModel.disable_adapter()`. One model in
  memory, and the two forwards cannot disagree about anything except the adapter.

The forward budget is sized from GPU memory (`precompute_budget`), with selected logits kept in
bf16 and promoted to fp32 one `kl_chunk` at a time — holding `log_softmax` of both models over
the full `[n_sel, V]` selection would be the peak allocation of the whole pass, ~5 GB, for no
accuracy the chunked form does not also have.

**One redundancy left in on purpose:** `math_only.py` regenerates the base model's ~500 math
completions and its KL continuations once per shard, because it hoists them out of its checkpoint
loop but keeps them in memory. Four concurrent shards therefore do ~4× the GPU-seconds for 1× the
wall clock, on GPUs that would otherwise be idle. The alternative is editing their eval code,
which is the one change that would make these numbers incomparable to Impl 3's and Impl 4's.

## Two bugs the local gates caught

Both would have produced a completed run with wrong or missing numbers, and neither would have
raised at the point of the mistake. Recorded because the gates are the reason they cost minutes
instead of GPU-hours.

**1. Split cache key (caught by `smoke_klw.py`).** The precompute folded the reference adapter's
digest into a single key shared by both variants; the trainer left it out for variant a, because
variant a has no reference. Same data, two keys. The symptom was `missing signal cache` for a
cache written thirty seconds earlier. Fixed by making `weighting.signal_key()` the single
definition both call — variant a's key excludes the reference (its signal genuinely does not
depend on π_SFT), variant b's includes it (§4.1: changing it "changes both the signal and the
precompute cache key"). Regression test:
`test_signal_key_ignores_the_reference_for_variant_a`.

**2. Collator assumed weights (caught by `smoke_klw.py`).** The Trainer uses one collator for
both dataloaders, and the eval dataset deliberately has no `weights` column — held-out loss stays
unweighted so it stays comparable to D4's. The collator indexed the column unconditionally, so
training worked and then died at the first `eval_steps` boundary. On the real run that is ~200
steps in, roughly 10 GPU-minutes per arm. The collator now decides per batch from the rows, and
raises if a batch mixes weighted and unweighted rows.

## What the checks are for

Every failure mode guarded here produces **a completed run with a plausible loss curve and wrong
numbers**. None of them raises on its own, which is why they are checks rather than assertions in
the training path.

- **W1** — the weighted and unweighted paths must normalise identically, or `bT451` cannot
  reproduce D4 and nothing is interpretable. Also *decides* `--loss_denom`, which the driver then
  passes to every arm explicitly rather than letting `auto` infer it. Measured on the same PEFT
  wrapping training uses, because `probe_loss_norm.py` found the answer depends on it. Answer on
  these pins: **`global`**, and `auto` agrees.
- **W2** — mean-1 over pedagogy tokens. Without it a temperature sweep is also a learning-rate
  sweep and the pedagogy:general loss ratio moves with T.
- **W3** — alignment through the collator's padding and the causal shift. An off-by-one here is
  invisible in a loss curve.
- **W4** — per-row digests: the cache describes the tokens training will actually see.
- **W5** — pedagogy rows are reweighted, general rows carry no signal at all.
- **W6** — the T→∞ limit behaves, so the control is valid; and ESS makes the low-T collapse
  visible before training. James's `a-T0.5`/`a-T1` finished at NLL 2.743/2.138, *above* base's
  1.416, because nearly all the gradient landed on nearly no tokens.
- **W7** — the chosen arm's multiplier distribution on the *real* corpus, including ESS.

## Submitting this on the platform

Two gates beyond the four `impl5_ssd/BUILD.md` documents, both discovered while writing the
submission and both specific to a multi-GPU shape:

**1. A multi-GPU profile refuses a one-process command.**
`edullm_platform.launchers.require_a_process_for_every_device` reads the command the way a shell
would and treats "no launcher" as **one process**; on a profile with 4 GPUs that is a refusal,
not a warning, and the message is about idling three cards at the whole machine's price. It is
right to refuse — but the fix it suggests, `torch.distributed.run --nproc-per-node=4`, is the one
thing this run must not do: HF `Trainer` detects `WORLD_SIZE`/`RANK` and switches to DDP, which
multiplies the effective batch by the world size and destroys the very batch shape deviation 3
exists to preserve.

So the submission carries the platform's own waiver, `EDULLM_LAUNCH_CHECK=waived`, which exists
for a command whose process count is deliberate. It is honest here: `run_klw.py` genuinely starts
one process per device, pinned with `CUDA_VISIBLE_DEVICES`, just not through `torchrun`. The
waiver is recorded on the run and `waived_launch_check_note` puts a line on the approver's page,
which is the right outcome — a lead should see that this run opted out of the launcher check and
why.

Written as an environment assignment because `carries_the_token` matches exact words and recurses
into `bash -lc` strings, so it must appear as its own word and it is inert to the program.

**2. `gpu-1xl40s` *is* on the submission form.** An earlier note in this project claimed it was
not. It is, along with `gpu-4xl40s` and 14 others, and the form also has `fanout_size` /
`fanout_index_parameter` for array jobs.

That makes a second, waiver-free shape available: **4 fan-out cells on `gpu-1xl40s`**, one arm per
cell, which is what this workload actually is. It needs the precompute to run as its own job first
and the signal caches to round-trip through S3 per cell, which the driver does not do yet. The
single `gpu-4xl40s` job below works with the code as written; the fan-out is the more
platform-idiomatic shape and the one to build if a lead would rather not approve a waiver.

### Job 1 — data, precompute, all four arms, ped_nll

`repository: OLMo-core` · `workload_profile: olmo-core-train` · `compute_profile: gpu-4xl40s` ·
`dataset_release: none` · `team: post-training` · `experiment: impl3x5-klw` ·
`wandb_project: edullm-p7` · `commit_sha:` this commit

```
bash -lc 'EDULLM_LAUNCH_CHECK=waived python p7/POC/impl3x5_klw/run_klw.py --stages bundle,fetch,pool,slot,mix,checks_fast,precompute,checks_full,train,bridge,eval --bundle_tar p7/POC/impl3_handoff.tar.gz --bundle /tmp/impl3_handoff --pool_from s3://sbsandbox-intern-edullm-outputs/teams/post-training/runs/run_019fc3ec-96a2-70fe-8153-21545ef0e908 --reference_from s3://sbsandbox-intern-edullm-outputs/teams/post-training/runs/run_019fc3ec-96a2-70fe-8153-21545ef0e908/checkpoints --checkpoint_dir "$EDULLM_CHECKPOINT_DIR" --output_prefix "$EDULLM_OUTPUT_PREFIX"'
```

`$EDULLM_CHECKPOINT_DIR` appears expanded under `bash -lc`, which is gate 4 of
`impl5_ssd/BUILD.md` — the container execs with no shell, so without `bash -lc` the variable
arrives as literal text.

`--pool_from` and `--reference_from` are D4's run (`run_019fc3ec-…`), **not this job's own
prefixes**, which are derived per run id and empty by construction.

### Job 2 — the forgetting axis, after job 1 finishes

Same fields, `--adapters_from` pointing at **job 1's** checkpoint prefix. Do not also request
`bridge`: `bridge.py` skips checkpoint dirs that already exist, so an unfiltered bridge first
would expose all 22 steps and the step filter would silently do nothing.

```
bash -lc 'EDULLM_LAUNCH_CHECK=waived python p7/POC/impl3x5_klw/run_klw.py --stages bundle,math --bundle_tar p7/POC/impl3_handoff.tar.gz --bundle /tmp/impl3_handoff --adapters_from s3://sbsandbox-intern-edullm-outputs/teams/post-training/runs/<JOB-1-RUN-ID>/checkpoints --checkpoint_dir "$EDULLM_CHECKPOINT_DIR" --output_prefix "$EDULLM_OUTPUT_PREFIX"'
```

One thing left unverified because it cannot be checked from outside a container: whether the
workload execs from the repository root, which is what makes `p7/POC/impl3_handoff.tar.gz`
resolve. If it does not, pass an absolute path. Everything in `run_klw.py` resolves relative to
its own file, so nothing else depends on the cwd.

## Cost estimate (not yet incurred)

On a `gpu-4xl40s`, all four GPUs busy:

| stage | wall clock | note |
|---|---|---|
| fetch + mix | ~10 min | CPU; mix asserts the replay slot reproduces A1 exactly |
| precompute | ~10–15 min | 2 forwards over 22,152 rows, 4-way sharded |
| train (4 arms) | ~45–60 min | concurrent; ~45 min is one arm's serial time |
| ped_nll (4 arms) | ~10 min | 22 checkpoints each, concurrent |
| math + KL (4 arms) | ~50 min | 11 checkpoints each at ~4 GPU-min, concurrent |

Roughly **2–2.5 h wall clock**. Against D4's ~3.2 h / $33 for one arm on one GPU, this is four
arms and three eval axes for about one and a half times that.

## What this run will and will not be able to say

**Can say:** whether James's reweighting changes forgetting, KL, or pedagogy NLL relative to D4
on identical data — a clean one-variable contrast, four arms on one plane with Impl 3's, Impl 4's
and Impl 5's existing points.

**Cannot say** anything about *judged* pedagogy quality. That needs `llm_judge/generate_arms.py`
+ `judge_pedagogy.py`, which is a separate CPU/API job, and it is the axis that decides whether a
forgetting win is real or just "changed less". Impl 5's own Definition of Done is reduced
forgetting **at matched pedagogy quality**, and the same standard applies here. A `bT1` that
forgets less than D4 and judges worse is not a win.

**Also cannot separate** "helped because reweighting composes with distillation" from "helped
because reweighting would have helped on gold too". That needs James's three configs re-run on
*gold* through this same pipeline — three more training runs, which would complete the 2×2. Not
run; the primary question (does reweighting add anything on top of Impl 5's targets) is answered
by these four arms against D4 without it.
