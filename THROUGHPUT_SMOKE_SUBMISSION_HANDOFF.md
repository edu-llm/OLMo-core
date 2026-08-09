# Throughput smoke submission handoff

Snapshot: 2026-08-09

This is the operational handoff for the five-cell Mamba comparison throughput
smoke. It records the submission path, the measurement contract, and the
current blocker. It is not an authority for price, runtime limits, approval, or
capacity; obtain those from a fresh `edullm check --json`.

## Current state

- Repository: `edu-llm/OLMo-core`
- Local checkout: `/home/vs/AlphaAI/eduLLM/OLMo-core-flash-pd`
- Branch: `edullm/mamba-comparison`
- Current pushed commit: `dac343fe232d6d7ee57f17cba5ccfd2235bbd8f9`
- CLI observed while writing this handoff: `edullm 4.5.0`
- Smoke spec: `.edullm/run-throughput-smoke.yaml`
- Entrypoint: `.edullm/train_core6_arm.py`
- Dataset release: `reservoir-dolma2-v1`
- Compute profile: `gpu-8xa100`
- Team and W&B project: `memory-split`
- Recent `edullm status --json` contains no Mamba comparison throughput run.

The pushed commit has a successful image publication:

- [green image workflow for `dac343f`](https://github.com/edu-llm/OLMo-core/actions/runs/31322456424)
- all three checks, including `Build and publish image`, completed successfully.

The local working tree now contains a later, uncommitted comparison-budget and
review update. Those bytes are **not** in the `dac343f` image:

- the image imports Torch before `_flash_pd_native_cuda`, loading `libc10.so`;
- bare/default submissions now run a bounded ten-step functional smoke through
  the same preflighted runner rather than a full 1,144-step cell;
- every arm stores uniform FP32 master parameters while FSDP supplies BF16
  compute parameters, eliminating mixed-original-dtype FSDP failures;
- all accelerated arms require rank-local A100 sm80, BF16 hardware support,
  and their exact kernel/package contract before training;
- the `gdn` diagnostic key now realizes the frozen mixer-bakeoff GDN2 control
  with FLA/fla-core 0.5.1;
- xLSTM prewarms after rank-local device selection and converts vanilla sLSTM
  parameters into the exact FlashRNN layout on every forward, with coherent
  BF16 kernel roles and no stale FSDP cache;
- native PD saves forward-contiguous operands and performs no hot-path host
  readbacks;
- Mamba3-SISO-PD keeps complex diagonals in FP32, uses a chunk-parallel
  backward, and performs no temperature-buffer host readbacks;
- `.dockerignore` excludes local native artifacts, and source builds use a
  fully pinned, non-isolated build-tool closure;
- the full comparison now matches mixer-bakeoff Run 1's measured per-cell
  budget: 1,144 steps, 599,785,472 tokens, TPP 1.53724–1.53735;
- `MIXER_OPTIMIZATION_REVIEW.md` records the independently checked architecture
  and remaining optimization gates.

The latest non-dispatching check was run while only two red-first test files had
been edited. It correctly refused `uncommitted_changes` and
`commit_not_pushed`. The branch already contains `dac343f` on GitHub, but this
checkout's narrow remote-tracking ref is stale. Commit the complete intended
working tree, push the resulting new SHA, fetch the exact ref, and require a
green image for that new SHA before submitting.

### Current Mamba-b3 contract

`mamba-b3` selects the strict `simple_gla` scan with `rotation_block_size=3`,
fused projections, and the quaternion/Rodrigues rotation path. `official_fast`
remains available by name but is no longer the comparison-arm default. The
switch is an A100 occupancy hypothesis: the latest source-hashed RTX 5050 layer
benchmark put `official_fast` 0.6% ahead, so the 8xA100 whole-model smoke must
validate the choice before a full comparison wave.

## What the smoke measures

The fan-out has five cells, selected by `AWS_BATCH_JOB_ARRAY_INDEX`:

- index 0: `mamba-b3`, init seed `110007`
- index 1: `xlstm`, init seed `113008`
- index 2: `mamba3-siso-pd`, init seed `116009`
- index 3: `native-pd`, init seed `119010`
- index 4: `gdn` (frozen GDN2), init seed `122011`

All cells use data seed `210007`, sequence length 4096, 100 optimizer steps,
a 524,288-token global batch, an 8,192-token rank microbatch, bfloat16, and
eight A100 processes. Held-out evaluation and the decode probe are disabled.
The checkpoint interval is 101, outside the timed run.

`gpu-8xa100` means one eight-A100 node **per fan-out cell**. A five-cell smoke
can therefore occupy up to 5 nodes / 40 A100s if every cell is admitted
concurrently. The twenty-cell full wave can occupy up to 20 nodes / 160 A100s.
Actual concurrency is controlled by the platform and capacity queue; cells may
start in staggered groups. The `nodes: 1` field in a check is per cell, not for
the entire fan-out.

`WARMUP_STEPS_EXCLUDED=50` in the runner excludes compilation, allocator
growth, and the first 50 steps from the throughput endpoint. The remaining 50
steps define:

- `throughput_tok_s_steady`: total tokens/second across all eight GPUs
- `throughput_tok_s_steady_per_device`: tokens/second per A100
- `step_time_s_p50` and `step_time_s_p90`
- `steady_window_seconds`
- `flops_per_token`, `mfu_pct`, and `mfu_basis`
- peak allocated and reserved GPU memory

Rank arms only by the two steady-throughput fields. Do not rank them by the
whole-run rate, the final step, platform history, or startup-inclusive timing.
GDN2 is the contemporaneous frozen speed control, not a fifth science arm. Its
implementation and FLA 0.5.1 dependency are copied from mixer-bakeoff commit
`092f2c2bd582c4daa9b3bbfae0effce76b0f833a` and must not be optimized further.
One seed and 50 measured steps detect gross speed problems; they do not replace
the five-seed comparison.

### Local CUDA sanity result

On 2026-08-08, the frozen single GDN2 mixer completed a local BF16
forward+backward benchmark on an RTX 5050 Laptop GPU at
`B=2,T=4096,D=1024,H=16,K=V=64`:

- backend: `fla.ops.gdn2.chunk_gdn2`, FLA/fla-core 0.5.1;
- 20 warmups and 50 CUDA-event measurements;
- median 36.553 ms, p90 37.853 ms;
- 224,115 tokens/s and 9.273 achieved TFLOP/s under the mixer's declared
  forward+backward FLOP convention;
- peak allocated 908,104,704 bytes and reserved 922,746,880 bytes.

MFU is intentionally absent because no documented dense BF16 peak was found
for this laptop GPU's power configuration. This is a mixer-only local sanity
result, not a five-arm model comparison and not comparable to the 8xA100
throughput smoke.

The CLI argument `--warmup-steps 10` belongs to the training schedule. It does
not replace the runner's fixed 50-step throughput exclusion.

## Submission procedure

Run every command from the comparison checkout.

There are three sequential gates:

1. `.edullm/run-smoke.yaml`: 10-step functional smoke, five cells.
2. `.edullm/run-throughput-smoke.yaml`: 100-step throughput smoke, five cells.
3. `.edullm/run-comparison.yaml`: full 1,144-step comparison, twenty cells.

Use the same commit, image, dataset release, compute profile, and
`--attempts 1` contract for all three. Replace the `--spec`, experiment name,
and fan-out size only as shown below.

### 1. Commit and push the exact source

The platform builds a commit, not a working tree.

```bash
cd /home/vs/AlphaAI/eduLLM/OLMo-core-flash-pd
git status --short --branch
git switch edullm/mamba-comparison
git push -u origin HEAD
```

Do not continue with uncommitted source, an unpushed commit, or a branch outside
`edullm/**`.

### 2. Refresh the ref used by the offline check

This checkout was originally configured with a narrow fetch refspec, so a
plain `git fetch` has previously left the comparison ref stale. Refresh the
exact ref:

```bash
git fetch origin \
  "refs/heads/edullm/mamba-comparison:refs/remotes/origin/edullm/mamba-comparison"
git branch -r --contains HEAD
```

The second command must print `origin/edullm/mamba-comparison`. GitHub having
the commit is insufficient: `edullm check` deliberately reads local
remote-tracking refs without asking GitHub.

### 3. Wait for the image, and require green

Every push to `edullm/**` starts the research-image workflow:

```bash
gh run list \
  --repo edu-llm/OLMo-core \
  --branch edullm/mamba-comparison \
  --workflow "Build eduLLM research image" \
  --limit 3
```

Open or watch the run for the exact `git rev-parse HEAD` SHA. Require the
`Build and publish image` job to succeed. A re-run does not repair a
deterministic Docker failure; fix the failure, commit, and push a new SHA.

### 4. Run the non-dispatching check

```bash
edullm check --json \
  --experiment mamba-comparison-throughput-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-throughput-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 5 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Read stdout separately from stderr. Interpret the exit status first:

- 0: the local checks stand
- 1: refused on the merits
- 2: command or installation error
- 3: transient platform-check failure; retry is appropriate

Match `refusals[].code`, not the wording in `detail`. Review the live `cost`,
`approval_class`, runtime bound, placement, and every deferred check before
submitting. The current command does not override `--hours`; the workload
default therefore applies. If an explicit bound is chosen, pass the same
`--hours N` to both `check` and `submit`.

A capacity paragraph is informational unless a corresponding entry appears in
`refusals`. Conversely, a green local check is not proof that an image exists:
image publication, image uniqueness, and scan review are resolved later.

### 5. Submit only after all gates above

Use the same arguments, changing only the verb:

```bash
edullm submit \
  --experiment mamba-comparison-throughput-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-throughput-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 5 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Do not use `--force`. Keep the run ID printed by `submit`.

### 6. Observe and retrieve results

Free status polling:

```bash
edullm status --json <run-id>
```

Do not poll plain `edullm status` or `edullm logs`; each dispatches a workflow.
Use logs once when diagnosis or the final stdout record is needed:

```bash
edullm logs <run-id>
```

The run also reports to W&B project `memory-split`. Confirm that all five array
cells completed and that each rank-0 JSON record names the expected arm,
data/init seeds, world size, steady-state step count, throughput, memory,
FLOPs/token, and MFU basis.

## Functional smoke and full comparison commands

The ten-step functional smoke uses:

```bash
edullm check --json \
  --experiment mamba-comparison-smoke \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-smoke.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 5 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

After a clean check, use the identical arguments with `edullm submit`. Require
all five cells to complete before the throughput smoke.

After both five-cell smokes pass, check the twenty-cell science wave:

```bash
edullm check --json \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 12 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Then submit with the same arguments, changing only `check --json` to `submit`.
The full wave is arm-major: three Mamba-b3 cells, three xLSTM cells, three
Mamba3-SISO-PD cells, then three native-PD cells.

### Running only Mamba-b3 and xLSTM

Because the wave is arm-major, the first six cells of the same spec are exactly
Mamba-b3 and xLSTM, three seeds each. No separate spec is needed; lower the
fan-out and leave everything else identical:

```bash
edullm check --json \
  --experiment mamba-comparison \
  --dataset reservoir-dolma2-v1 \
  --team memory-split \
  --spec .edullm/run-comparison.yaml \
  --compute gpu-8xa100 \
  --attempts 1 \
  --fanout-size 6 \
  --fanout-index-parameter AWS_BATCH_JOB_ARRAY_INDEX
```

Indices 0-2 are Mamba-b3 and 3-5 are xLSTM, on the same corpus, budget, seeds,
and geometry as the four-arm wave, so the two arms stay comparable to each other
and to the remaining arms if those are run later.

For one arm only, `--fanout-size 3` gives Mamba-b3. xLSTM has no contiguous
prefix of its own, so run it inside the six-cell subset above.

### Local verification standing behind this state

On a local RTX 5050 (sm120, not the sm80 submission target), the xLSTM,
Mamba-3, platform-entrypoint, runner, and static contract suites report 608
passed and 2 failed. Both failures are `fp8_test.py` raising
`ModuleNotFoundError: torchao`, an optional package absent from that laptop
environment; no arm enables FP8 and the runner never imports Float8.

FlashRNN's sLSTM BF16 kernel compiles and backpropagates locally only when the
cuBLAS, cuSPARSE, and cuSOLVER include directories are on `CPATH`. Without them
its just-in-time build fails in a way that reads like a kernel defect and is
not one.

## Dataset and image contract

The platform supplies the published release `reservoir-dolma2-v1`; do not
replace it with a private S3 path or hand-written manifest. The runner resolves
paths, dtype, tokenizer, and held-out metadata through `edullm-data`.

The target image pins `edullm-data` commit
`38bf831a6c3f445e394784018441fd59288b876c`. Its registry source was inspected
at that exact commit and ships:

- `eval-results/v1`
- `pretrain-tokens/v1`
- `sft-conversations/v1`
- `token-order/v1`
- `tokenizer/v1`

The Dockerfile repeats that list as a build-time assertion. The `dac343f`
image workflow reached and passed the final build checks. Any new commit still
requires its own green publication.

The image is intended to contain Torch 2.10 built for CUDA 12.8 and sm80,
Mamba-3 at revision `e9594ce1c732d97440f0332fdc43170a2294dbfa`,
`flash-linear-attention==0.5.1`, `fla-core==0.5.1`, `xlstm==2.0.5`,
`mlstm-kernels==2.0.4`, `flashrnn==1.0.6`, both native PD entrypoints, and
PyTorch fused SDPA. The image workflow, not this statement, must prove that
contract for the submitted SHA.

## Failure routing

`commit_not_pushed`

: Push the exact SHA to `edullm/**`, then fetch the exact remote-tracking ref
  in this same clone and verify `git branch -r --contains HEAD`.

`no_published_image` or a failed image workflow

: Do not submit. Read the first failed image-build step, fix it on a new
  commit, push, and wait for green.

`image_is_ambiguous` or scan-related deferred checks

: Do not guess an image digest. The submission/admission workflow resolves
  these against the registry; a failed gate is a stop condition.

`THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION` (exit 75)

: The rank-local device cannot execute the explicit BF16 contract. This is
  distinct from held-out corpus failures and is not retryable on the same
  hardware.

Training cell failure

: Identify the array index and arm first. Preserve the common geometry when
  comparing throughput. Do not replace a failed accelerated backend with a
  CPU or eager fallback and report it as a comparable result.

## Safety invariants

- Use `edullm`; never call AWS directly with `aws`, `boto3`, or endpoint curls.
- Never use Beaker or the unreachable `ai2/*` clusters.
- Never use `--force`.
- Keep `bfloat16` in the command text so the platform precision guard can see
  it.
- Keep check and submit arguments identical.
- Keep the five arms, seeds, fan-out size, sequence length, batches, step
  count, and compute profile matched.
- `edullm run` and `edullm shell` are exploratory and do not create a citable
  comparison run.

## Immediate next action

Review and commit the consolidated working tree, including this handoff, then
push the final SHA and wait for its green image workflow. Fetch the exact
remote-tracking ref, repeat the non-dispatching check, and submit only after
`refusals` is empty. Local sm120 tests cover FlashRNN BF16
forward/backward, native-PD parity and zero-sync dispatch, Mamba-PD mixed/tail
parity, and the scatter deadlock fix; the five-cell sm80 functional smoke is
still the end-to-end runtime gate.
