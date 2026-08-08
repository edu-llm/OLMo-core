# Throughput smoke submission handoff

Snapshot: 2026-08-08

This is the operational handoff for the five-cell Mamba comparison throughput
smoke. It records the submission path, the measurement contract, and the
current blocker. It is not an authority for price, runtime limits, approval, or
capacity; obtain those from a fresh `edullm check --json`.

## Current state

- Repository: `edu-llm/OLMo-core`
- Local checkout: `/home/vs/AlphaAI/eduLLM/OLMo-core-flash-pd`
- Branch: `edullm/mamba-comparison`
- Current pushed commit: `0c6e0f54ca4c49d7f38467723ad9337b4b0a894f`
- CLI observed while writing this handoff: `edullm 4.5.0`
- Smoke spec: `.edullm/run-throughput-smoke.yaml`
- Entrypoint: `.edullm/train_core6_arm.py`
- Dataset release: `reservoir-dolma2-v1`
- Compute profile: `gpu-8xa100`
- Team and W&B project: `memory-split`
- Recent `edullm status --json` contains no Mamba comparison throughput run.

The commit is pushed, and the local remote-tracking ref now contains it. The
latest image build did **not** publish an image:

- [failed image workflow for `0c6e0f5`](https://github.com/edu-llm/OLMo-core/actions/runs/31282749986)
- First build error: `ATen/cuda/CUDAContextLight.h` includes `cusparse.h`, but
  the CUDA compiler image does not currently contain that header.

Therefore the throughput smoke is **not submit-ready**. A local check can have
no refusals while image questions remain deferred; do not submit until a newer
commit's image workflow is green and has published exactly one image.

The local working tree now contains an uncommitted fix in
`flash_pd_native_setup.py`: the native extension discovers the cuBLAS,
cuSPARSE, and cuSOLVER header directories supplied by the pinned Torch CUDA
wheels and passes them to `CUDAExtension`. Its focused contract test passes,
but this working-tree fix has not built or published an image.

## What the smoke measures

The fan-out has five cells, selected by `AWS_BATCH_JOB_ARRAY_INDEX`:

- index 0: `mamba-b3`, init seed `110007`
- index 1: `xlstm`, init seed `113008`
- index 2: `mamba3-siso-pd`, init seed `116009`
- index 3: `native-pd`, init seed `119010`
- index 4: `gdn`, init seed `122011`

All cells use data seed `210007`, sequence length 4096, 100 optimizer steps,
a 524,288-token global batch, an 8,192-token rank microbatch, bfloat16, and
eight A100 processes. Held-out evaluation and the decode probe are disabled.
The checkpoint interval is 101, outside the timed run.

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
GDN is the contemporaneous speed control, not a fifth science arm. One seed and
50 measured steps detect gross speed problems; they do not replace the
five-seed comparison.

The CLI argument `--warmup-steps 10` belongs to the training schedule. It does
not replace the runner's fixed 50-step throughput exclusion.

## Submission procedure

Run every command from the comparison checkout.

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

The Dockerfile repeats that list as a build-time assertion. No image has yet
reached that final assertion for the current commit because the native CUDA
extension fails earlier.

The image is intended to contain Torch 2.10 built for CUDA 12.8 and sm80,
Mamba-3 at revision `e9594ce1c732d97440f0332fdc43170a2294dbfa`,
`xlstm==2.0.5`, `mlstm-kernels==2.0.4`, `flashrnn==1.0.6`, both native PD
entrypoints, and PyTorch fused SDPA. The image workflow, not this statement,
must prove that contract for the submitted SHA.

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

Commit the native-extension header fix, its contract test, and this handoff;
push the new SHA to `edullm/mamba-comparison`; then wait for a green image
workflow. Only after the image publishes should the exact-ref fetch, `check`,
and `submit` sequence above be repeated.
