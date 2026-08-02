# Token-selection 370M branch

This branch contains the complete RegMix token-selection family outside
`src/olmo_core`. Every arm uses the same `TransformerConfig.olmo2_370M` recipe:
`d_model=1024`, 16 layers/heads, reordered norm, gated-SiLU 4096 MLP, full
attention, QK-RMSNorm, RoPE theta 500,000, Dolma2 vocabulary 100,352, sequence
2048, global batch 4,194,304 tokens, and 65,536 rank microbatch tokens.
Optimization is SkipStepAdamW at peak LR `4e-4`, betas `(0.9, 0.95)`, weight
decay `0.1` except embeddings at `0`, 24-step cosine warmup, `alpha_f=0.1`,
Z-loss `1e-5`, grad norm 1.0, HSDP bf16 parameters/fp32 reductions, init seed
6198, data seed 42, and compilation enabled.

Production launches use `.edullm/platform/entrypoint.sh`, which runs the selected
arm under one-node `torch.distributed.run --nproc_per_node=8`. The Python
entrypoint independently refuses a production topology other than eight local
CUDA ranks before constructing the model. The accepted arms are `control`,
`rho-1`, `rel-ema-exp`, `rel-ema-refhq`, `middle-ppl-token`, `middle-ppl-doc`,
`learnability-token`, `learnability-doc`, `attention`, `reference`, and `blade`.

The immutable handoff/submission order is:

| index | exact arm ID | method and experimental delta | train input |
|---:|---|---|---|
| 0 | `control` | deterministic random 60% token keep | `pretrain/regmix-10b` |
| 1 | `rho-1` | top 60% `L_curr-L_ref` | `pretrain/regmix-10b` |
| 2 | `rel-ema-exp` | top 60% `L_hist-L_curr`; zero-seeded bias-corrected EMA, alpha `1-exp(-t/300)` | `pretrain/regmix-10b` |
| 3 | `rel-ema-refhq` | same REL score; RefHQ-step1315 seed, alpha `0.9985` | `pretrain/regmix-10b` |
| 4 | `middle-ppl-token` | middle 60% tokens by late RefHQ loss | `pretrain/regmix-10b` |
| 5 | `middle-ppl-doc` | full CE over the immutable middle-60% document export | `pretrain/middle-ppl-doc-mid60` |
| 6 | `learnability-token` | top 60% `L_early-L_late` | `pretrain/regmix-10b` |
| 7 | `learnability-doc` | full CE over the immutable top-60% document export | `pretrain/learnability-doc-top60` |
| 8 | `attention` | top 60% causal attention-received tokens | `pretrain/regmix-10b` |
| 9 | `reference` | unchanged full RefHQ CE | `pretrain/refhq-regmix-5p5b` |
| 10 | `blade` | top 60% dynamic excess after proxy warmup | `pretrain/regmix-10b` plus RefHQ stream |

These indices are the matrix and sequential-submission mapping; the production
CLI takes the exact string with `--arm`, not a numeric index. Every arm routes
to W&B project `token-selection-<arm ID>`.

Inputs are resolved by pinned dataset version through the seal-verifying
`edullm_data.read` path. Checkpoint identity binds the resolved version, dtype,
row count, and SHA-256 of the ordered object list; BLADE binds its RefHQ stream
independently. Outputs stay on runtime scratch and synchronously upload to W&B.
The image packages the exact 20-label task-loss evaluator and production runs
evaluate it on eight ranks in project `token-selection-<arm>`.

Bootstrap files are local, immutable exports:

- `EDULLM_REFERENCE_PATH`: RefHQ step1315 for RHO-1 and seeded REL.
- `EDULLM_LATE_REFERENCE_PATH`: average of RefHQ steps 1000/1125/1315 for
  middle-PPL token and learnability token.
- `EDULLM_EARLY_REFERENCE_PATH`: RefHQ step250 for learnability token.
- `EDULLM_PASSIVE_REFERENCE_PATH`: optional passive excess-loss metric only; it
  never changes token weights.

The methodology source is
`edu-llm/edullm@b435cbe9c352399fc4ab54b310f36d28f6c9746f`,
`experiments/token-selection/README.md`. Main RegMix production uses platform
release `regmix-10b-v1` and requires the resolved `EDULLM_DATASET_VERSION`, not
`latest`. RefHQ contracts are immutable step URIs: step250 for early,
step1315 for RHO-1/seeded REL, and the materialized average of
steps1000/1125/1315 for late. Each materialized reference file is SHA-256
hashed into the run identity. The two document exports and BLADE RefHQ corpus
must likewise resolve a sealed, pinned version. Resume rejects a changed
dataset object order, dtype, row count, reference hash, BLADE secondary stream,
arm, or hyperparameter. There are no pilot outputs in this family.

BLADE uses RegMix for the proxy and penalty stream plus a separately resolved
RefHQ stream. It locks warmup 500, syncs 500/875/1250/1625/2000, `tau=375`,
`K=75`, gamma 0.6, and lambda 1.0. OLMo checkpoints carry proxy/optimizer state;
the callback checkpoint state carries the post-K dynamic reference and optimizer,
both secondary stream cursors, last sync, and completed step. Resume therefore
never inserts an unscheduled reference sync.

`.edullm/platform/token-selection-arms.json` records the full arm matrix.
`.edullm/fixtures/token-selection-control-submission.json` is a
credential-free compiler fixture that selects the provisioned `gpu-8xa100`
(8 × A100 on `p4d.24xlarge`) compute override and an eight-rank command. The
profile is admin-gated by rate, as expected. This branch does not submit jobs
or publish data/models.

## Checkpoints, evaluator, resume, and artifacts

The permanent ladder is step 0, every 125 steps, and true final, omitting the
last grid point when it is less than 125 steps before final. Main RegMix arms
use 9.9B tokens / 2,360 steps; `reference` uses the sealed RefHQ row count.
Every save is permanent. The branch-local evaluator is self-contained:
`.edullm/eval_task_loss_olmo_core.py` carries the exact 20
`*_rc_5shot_bpb` labels and the image pins its OLMES-compatible dependencies.
Training pauses until the complete suite and awaited W&B uploads succeed.
Partial evals and upload failures do not advance `last_durable_step.json`.

Checkpoints, progress, metrics, selection state, BLADE state, and task-loss
JSON stay on runtime scratch. Every evaluation and run-state artifact uploads
to W&B, but only the true final checkpoint uploads as a model artifact; S3 is
input-only. A fresh run refuses a non-empty checkpoint directory. `--resume`
restores a local checkpoint, or a completed run's final checkpoint through
`WANDB_RESUME_ARTIFACT`, and requires its schema-v2 fingerprint.
Token-selection EMA/history state is callback-checkpointed. BLADE additionally
restores proxy optimizer/model state, post-K dynamic reference and optimizer,
both secondary stream cursors, completed step, and last sync, so resume cannot
insert an unscheduled sync.

## Exact commands and fixtures

Focused local validation:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\.edullm"
py -3 -m pytest .edullm\tests\test_token_selection_370m.py `
  .edullm\tests\test_production_contract.py -q
py -3 -m ruff check .edullm
```

The exact credential-free, full-length representative benchmark uses control.
It keeps eight ranks, 2,360 steps, the checkpoint ladder, and full task loss;
`WANDB_MODE=disabled` disables W&B durability, while `--local` disables the
production credential/topology gate only:

```bash
WANDB_MODE=disabled \
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  .edullm/token_selection_entrypoint.py \
  --arm control --local \
  --save-folder /tmp/token-selection-control-benchmark/checkpoints \
  --work-dir /tmp/token-selection-control-benchmark/work \
  --progress-dir /tmp/token-selection-control-benchmark/progress \
  --task-loss-script .edullm/eval_task_loss_olmo_core.py
```

The exact platform forms are:

- production control:
  `.edullm/fixtures/token-selection-control-submission.json`
- credential-free benchmark:
  `.edullm/fixtures/token-selection-control-benchmark-submission.json`

Both select `gpu-8xa100` (one `p4d.24xlarge`, 8 × A100) through
`olmo-core-train-4gpu`, launch eight ranks, use `regmix-10b-v1`, and contain no
credential. Replace the all-zero `commit_sha` with the immutable published
branch commit before submission. In **Submit a run**, leave `image_digest`
blank so the platform resolves that commit's image. For another arm, copy the
production form and change `--arm`, `experiment`, `dataset_release`, and
`wandb_project` together according to the table and
`.edullm/platform/token-selection-arms.json`; add the required materialized
reference environment only through the approved runtime staging mechanism.

Compile both checked-in forms against the unchanged local platform:

```powershell
$platform = "C:\alpha_ai\platform"
$fixtures = "$PWD\.edullm\fixtures"
$env:PYTHONPATH = "$platform\src"
foreach ($name in @("token-selection-control-submission",
                     "token-selection-control-benchmark-submission")) {
  py -3 "$platform\tools\compile_submission.py" `
    --inputs "$fixtures\$name.json" --config-dir "$platform\config" `
    --published-images "$fixtures\token-selection-published-image.json" `
    --submitter caiiris --repository-url https://github.com/edu-llm/OLMo-core `
    --output "$env:TEMP\$name-compiled.json"
}
```

`token-selection-published-image.json` is synthetic compiler evidence only.
Compilation is local, makes no live call, and submits nothing.

The profile's $21.9576/hour rate exceeds the platform's $20/hour routine-rate
ceiling. Successful compilation therefore returns approval class `exception`
and environment `run-approval-admin`; a team lead cannot release it.
`EDULLM_CHECKPOINT_CHECK=waived` records that the family intentionally uses
W&B rather than the platform checkpoint prefix and is displayed to the
approving platform admin.

Run the credential-free control benchmark first. Let `T` be its measured
end-to-end runtime in hours, including staging, every checkpoint, and every
task-loss evaluation. Set each production form's `maximum_runtime_hours` to
`1.25 * T`, rounding upward only to form precision. The benchmark fixture's
12-hour value is a safety bound, not an estimate; a timed-out benchmark cannot
be used for this calculation.

The 8×A100 environment may admit only one job. When capacity is constrained,
submit in table-index order and wait for each run to finish before submitting
the next; do not fan out the family. The already-fixed RefHQ/reference exports
are inputs, not permission to retrain or overwrite them.

Branch code and fixtures are handoff-ready; production dispatch remains gated
on a published commit/image, the benchmark-derived 25%-padded runtime, sealed
versions for arm-specific inputs, and per-run admin approval.
