# Skill-It OLMo2-370M

This branch migrates only the two full 370M arms. The completed DataDecide-60M
pilots are final scientific inputs and are not runnable here. Their measured
offline matrix, the derivative fit, source commit, and source-file SHA-256
values are embedded in `skillit_recipe.json`; `skillit_math.py` also verifies
the complete recipe file checksum before any arm is constructed.

## Fixed methodology

- source: labeled views of `s3://edullm-data/pretrain/olmo-127b/v1`, resolved
  through `edullm_data.read.dataset_paths`
- domains: dclm, arxiv, starcoder, pes2o, open-web-math,
  algebraic-stack, wiki
- model: `TransformerConfig.olmo2_370M`
- sequence/global/rank-microbatch tokens: 2,048 / 4,194,304 / 65,536
- optimizer: SkipStepAdamW, LR `4e-4`, betas `(0.9, 0.95)`, weight decay
  `0.1` except embeddings at `0`
- schedule: 24 warmup steps, `alpha_f=1.0`
- HSDP bf16 parameters/fp32 reductions, z-loss `1e-5`, max grad norm `1`,
  compiled model
- seed/steps: 42 / 2,384
- permanent checkpoints: step 0, every 125 except 2,375, and final step 2,384
- synchronous fail-closed 20-label RC 5-shot BPB evaluation at every permanent
  checkpoint

`WeightedDomainDataLoader` chooses a domain from the current weights and then a
random 2,048-token chunk. Its RNG is keyed by the global batch index. Every
rank reconstructs one global batch and consumes a strided partition, making
sampling deterministic across ranks and exact after checkpoint restore. Loader
state includes the seven source fingerprints, batch/token position, seed,
domain order, and current weights; changed immutable fields reject resume.

At steps 500, 875, 1,250, 1,625, and 2,000, the shared task-loss callback
finishes the checkpoint, strict suite, W&B evaluation uploads, and durable marker before
the lower-priority `SkillItController` runs. The controller reads the six curve
families, computes the README equation with eta `0.2` and w `1`, broadcasts
`p_after`, and applies it to the next batch.

The controller appends the full state to
`progress/skillit_updates.jsonl`, writes `stepN_A.json` and
`stepN_weights.json`, and restores the latest recorded `p_after` on resume.
The derivative arm logs the evaluation point `r == p_before`. At baseline and
every update, all 42 matrix cells plus every before/after domain weight are
logged as W&B metrics, and the complete matrix/weight JSON history is uploaded
as the versioned `skillit-<arm>-state` artifact.

| arm index | arm | A | W&B project |
|---:|---|---|---|
| 0 | probe | completed-pilot offline A | `skillit-probe` |
| 1 | deriv | mixing-law derivative at current weights | `skillit-deriv` |

The index is the production CLI value for `--arm-index`; the arm ID is exactly
`probe` or `deriv` (not the older `skillit-probe`/`skillit-deriv` run labels).
Everything except how `A` is obtained is identical. `probe` uses the embedded,
fixed 7×6 completed-pilot matrix. `deriv` evaluates
`max(0, -t_ij * (L_j(r) - c_j))` at the current `r == p_before`.

## Immutable inputs and provenance

`skillit_recipe.json` is the complete scientific lock:

- source README commit:
  `b435cbe9c352399fc4ab54b310f36d28f6c9746f`
- completed offline-A JSON SHA-256:
  `e542e3e66f70c752110b51f60d1ee84f5f7860931dce5684e7a621f35dd74a21`
- mixing-law derivative fit SHA-256:
  `acb4754b46cd6a588dffce7e7ad0d9bd70b0188db010669a7cfccf8622da2bcc`
- training dataset: `pretrain/olmo-127b`, exactly version `v1`, platform release
  `olmo-127b-v1`

`skillit_math.load_recipe()` verifies the checked-in recipe checksum before it
constructs an arm. Dataset resolution must return that exact ID/version, and
the loader fingerprint binds the ordered paths, domain order, dtype, seed,
batch geometry, and each of the seven source fingerprints. Resume fails closed
on any difference. The completed 60M pilots and their fits are immutable
inputs; this branch neither downloads a mutable replacement nor reruns them.

## Image and launch

`.edullm/Dockerfile` extends the platform-pinned base. In addition to the
OLMo-core branch, it pins `edullm-data` and the OLMES-compatible `ai2-olmo`
evaluation dependency by commit. The complete 20-label evaluator and its
Dolma2/OLMo2-370M evaluation config live in
`.edullm/eval_task_loss_olmo_core.py`; the image compiles that checked-in file
and does not fetch code from the separate eduLLM checkout. No pilot code or
output is regenerated.

Before `Trainer.fit()`, `train_skillit_370m.py` verifies the concrete trainer
is attached to `WeightedDomainDataLoader`, the resume-aware task-loss callback,
and the lower-priority Skill-It controller. Production also requires exactly
eight data-parallel ranks and the branch-local evaluator path. The explicit
fingerprint/resume handoff is then executed immediately before the single
`Trainer.fit()` call.

The two credential-free platform fixtures are:

- `.edullm/fixtures/skillit-probe-submission.json`
- `.edullm/fixtures/skillit-deriv-submission.json`

Each selects the platform's provisioned `gpu-8xa100` (8 × A100) compute
profile, launches eight ranks directly, keeps checkpoint/progress/eval state
on runtime scratch, and selects the arm's required W&B project. The explicit
`EDULLM_CHECKPOINT_CHECK=waived` records why the platform's S3 checkpoint path
is intentionally unused. Every evaluation is uploaded, but only the true final
checkpoint is a W&B model artifact; intermediate checkpoints remain on runtime
scratch. They use one attempt so production never silently infers resume. To
resume a locally restored completed W&B run explicitly, append
`--resume` and point all three state directories at that restored run root. A
resume is rejected unless its saved scientific fingerprint, durable marker,
task-loss output, and current arm all agree.

For a local non-production smoke only, pass `--allow-local-only
--wandb-mode=disabled`. Production refuses non-online W&B. The task-loss
evaluator is still mandatory because checkpoint-gated updates cannot use a
partial or missing suite.

## Exact commands

From this branch on a CUDA host, the credential-free full representative
benchmark is the probe arm. It keeps the scientific data path, all 2,384
steps, checkpoint ladder, and 20-label evaluator, but disables W&B:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  .edullm/train_skillit_370m.py \
  --arm-index 0 --run-name skillit-probe-benchmark \
  --work-dir /tmp/edullm-skillit-benchmark/probe/work \
  --save-folder /tmp/edullm-skillit-benchmark/probe/checkpoints \
  --progress-dir /tmp/edullm-skillit-benchmark/probe/progress \
  --task-loss-dir /tmp/edullm-skillit-benchmark/probe/task_loss \
  --task-loss-evaluator .edullm/eval_task_loss_olmo_core.py \
  --wandb-mode disabled --allow-local-only
```

The exact platform production and benchmark forms are checked in as:

- `.edullm/fixtures/skillit-probe-submission.json`
- `.edullm/fixtures/skillit-deriv-submission.json`
- `.edullm/fixtures/skillit-probe-benchmark-submission.json`

Replace each fixture's all-zero `commit_sha` with the immutable, published
branch commit before submission. Do not alter the command or W&B project while
doing so. Compile all three locally against the unchanged platform checkout:

```powershell
$platform = "C:\alpha_ai\platform"
$repo = "$PWD\.edullm\fixtures"
$env:PYTHONPATH = "$platform\src"
foreach ($name in @("skillit-probe-submission", "skillit-deriv-submission",
                     "skillit-probe-benchmark-submission")) {
  py -3 "$platform\tools\compile_submission.py" `
    --inputs "$repo\$name.json" --config-dir "$platform\config" `
    --published-images "$repo\skillit-published-image.json" `
    --submitter caiiris --repository-url https://github.com/edu-llm/OLMo-core `
    --output "$env:TEMP\$name-compiled.json"
}
```

The fixture `skillit-published-image.json` is compiler-only evidence and is
never a claim that the placeholder commit was actually published. For a real
run, use **Submit a run** in `edu-llm/platform`, copy the selected fixture
fields exactly, and leave `image_digest` blank so the platform resolves the
image built from the real commit.

All fixtures deliberately override the workload shape to the provisioned
`gpu-8xa100` profile: one `p4d.24xlarge`, 8 × A100, with eight torchrun ranks.
Its $21.9576/hour catalog rate exceeds the platform's $20/hour routine-rate
ceiling, so successful compilation returns approval class `exception` and
environment `run-approval-admin`. A team lead cannot release it; a platform
admin must approve. `EDULLM_CHECKPOINT_CHECK=waived` is intentional because
durability is W&B-only, is recorded in the manifest, and is shown to that
admin. Compilation performs no submission and no live call.

Run the credential-free benchmark first. Let `T` be its measured end-to-end
runtime in hours, including startup, data staging, every checkpoint, and every
task-loss evaluation. Set each production form's
`maximum_runtime_hours` to `1.25 * T` (round upward only to the precision the
form accepts). The benchmark fixture's 12-hour value is only its safety bound;
it is not a production-runtime estimate. If the benchmark reaches that bound,
investigate rather than extrapolating a truncated runtime.

The A100 pool may have capacity for only one such job. Submit the benchmark,
wait for it to finish, then submit `probe`, wait for completion, and finally
submit `deriv`. Do not use fan-out or overlap the two production arms unless
the platform owner confirms spare 8×A100 capacity.

## Focused validation

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\.edullm"
py -3 -m pytest .edullm\tests\test_skillit_methodology.py `
  .edullm\tests\test_production_contract.py -q
py -3 -m ruff check .edullm
```

Production readiness is therefore two-stage: code, documentation, and compiler
fixtures are ready now; production dispatch remains blocked until a real image
commit exists, the full benchmark completes, its measured runtime is padded by
25%, and an admin approves each sequential 8×A100 submission.
