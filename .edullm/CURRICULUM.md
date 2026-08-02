# Curriculum 370M migration

This branch implements the five approved arms in `curriculum_recipe.json` without edits
under `src/olmo_core`. The methodology source is
`edu-llm/edullm/experiments/curriculum`: current README and pacing tests take
precedence over older control scripts.

Fixed contract:

- OLMo2-370M (`d_model=1024`, 16 layers, 16 heads, reordered norm,
  gated-SiLU 4096 MLP, full attention, QK-RMSNorm, RoPE theta 500,000),
  Dolma2 tokenizer/vocabulary 100,352, sequence length 2,048, global batch
  4,194,304 tokens, rank microbatch 65,536 tokens
- SkipStepAdamW at `4e-4`, betas `(0.9, 0.95)`, weight decay `0.1`
  except embeddings at `0`, 24-step warmup, `alpha_f=1.0`
- HSDP bf16 parameters/fp32 reductions, z-loss `1e-5`, max grad norm `1`,
  compiled model, random initialization
- seed 42 and 2,384 production steps
- step 0, 125-grid (omitting 2,375), and true-final permanent checkpoints
- synchronous exact 20-label task loss at every permanent checkpoint
- awaited, fail-closed W&B artifacts in `curriculum-<arm>`
- post-hoc EMA over 2,000 / 2,125 / 2,250 / 2,384 with alpha 0.8

Only the loader's pacing/ordering policy differs between arms:

| index | arm ID | pacing | metric / order group |
|---:|---|---|---|
| 0 | `linear10-flesch` | `linear_n10` | `flesch` / `flesch` |
| 1 | `linear10-mtld` | `linear_n10` | `mtld` / `mtld` |
| 2 | `linear10-learn` | `linear_n10` | `learnability` / `learnability` |
| 3 | `warmup-mtld` | `warmup_1000` | `mtld` / `mtld` |
| 4 | `interleave-mtld` | `interleave_i10_linear` | `mtld` / `mtld` |

The table index is exactly the `--arm-index` CLI value. Each arm routes to
W&B project `curriculum-<arm ID>`; no arm shares the old `curriculum` project.

## Immutable inputs and provenance

The methodology was reconciled against
`edu-llm/edullm@b435cbe9c352399fc4ab54b310f36d28f6c9746f`,
`experiments/curriculum/README.md`, and its pacing implementation/tests.
`curriculum_recipe.json` pins the parent to
`pretrain/regmix-10b/v1` with manifest SHA-256
`a24992f53dc4a900bacf8fa571d77e343fd28ffa9054c14b93d54204b0a38cb4`.
The platform release is `regmix-10b-v1`.

Curriculum arms additionally resolve a sealed version of
`curriculum/regmix-370m`; the selected version, group, profile, manifest, and
its `depends_on` parent ID/version/manifest are recorded in the run
fingerprint. Resolution checks the catalog seal. An order must be a complete
permutation of the exact parent's shard-local 2,048-token chunks, and
same-length orders for another parent are rejected. Resume and EMA require one
identical fingerprint. There are no pilot checkpoints or frozen model
references in this family.

The loader uses the next zero-based batch index as its pacing step. Its
checkpoint state binds the parent dataset version, parent manifest, selected
order version/group/manifest, pacing, metric, seed, and batch geometry. A
same-length order for another parent is rejected before any model is built.
Control uses a deterministic no-replacement permutation of the flat parent
chunks. Curriculum modes preserve the current source pacing tests exactly.

Production inputs are resolved from sealed `edullm-data` releases and staged
to job-local scratch. Checkpoints, task-loss files, and progress stay on local
scratch and are uploaded to W&B; they are never written to S3. A launch must
choose `--fresh` or `--load-path`. W&B artifact resume accepts
`wandb-artifact://entity/project/name:version`.

Print the matrix without submitting anything:

```bash
bash .edullm/launch_curriculum_matrix.sh --print-only
```

Local smoke explicitly disables durability and task loss:

```bash
ARM_INDEX=0 NPROC=1 FRESH=1 LOCAL_SMOKE=1 WANDB_MODE=disabled \
  bash .edullm/launch_curriculum_arm.sh \
  --length-tokens 4194304
```

Production additionally requires `WANDB_API_KEY`. The exact 20-label OLMES BPB
evaluator and its OLMo2-370M base config are packaged under
`.edullm/task_loss/`; the Dockerfile pins the compatible `ai2-olmo` commit and
evaluation dependencies and exports both paths. Curriculum arms may pin
`CURRICULUM_DATASET_VERSION`; if omitted, the sealed latest version is resolved
once and recorded in the run fingerprint, and resume requires that exact
identity.

The concrete eight-GPU launcher is `.edullm/platform/entrypoint.sh`. It starts
eight training ranks, checks that the initialized process group and visible
CUDA topology both contain eight local ranks, and runs task loss on eight
released devices. The Python entrypoint remains topology-parameterized so the
README's 1–N GPU contract is preserved for explicit `--nproc` launches.

The existing-platform compilation fixture uses the catalog's provisioned
`gpu-8xa100` compute profile (8 × A100 on `p4d.24xlarge`) through the
submission form's supported `compute_profile` override. The workload profile remains
`olmo-core-train-4gpu` because it supplies the OLMo training bounds and
checkpoint contract; the compiler validates the resolved eight-GPU shape
against the submitted eight-rank torchrun command. No platform configuration
change is required. Because the A100 profile's hourly rate exceeds the routine
ceiling, compilation correctly routes it to the admin exception gate.

## Checkpoint, evaluator, resume, and artifacts

The evaluator is self-contained under `.edullm/task_loss/`: the checked-in
script, OLMo2-370M ladder config, fixed 20 `*_rc_5shot_bpb` labels, and pinned
evaluation dependencies are copied into the image. At every permanent
checkpoint all ranks pause, release the train module, run the complete suite,
reload the checkpoint, and only then continue. Partial suites fail closed.

Production writes checkpoints, progress, metrics, and task-loss JSON only to
job-local scratch, then awaits every W&B eval/run-state upload before advancing
the durable marker. Only the true final checkpoint is uploaded as a W&B model
artifact; intermediate checkpoints stay on scratch. S3 is input-only. Fresh
and resume are explicit: use `--fresh`, or `--load-path` with a local step
directory. A W&B artifact can restore only a completed run's final checkpoint;
never infer resume from scratch.
The loader restores its exact zero-based batch position and bound order.
Post-hoc EMA accepts only steps 2000/2125/2250/2384 from one fingerprint and
uploads the merged checkpoint and its full task-loss result.

## Exact commands and fixtures

Local validation:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\.edullm"
py -3 -m pytest .edullm\tests\test_curriculum_methodology.py `
  .edullm\tests\test_production_contract.py -q
py -3 -m ruff check .edullm
```

The exact credential-free, full-length representative benchmark keeps eight
ranks, all 2,384 steps, the checkpoint ladder, and full task loss while
disabling only W&B durability:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  .edullm/curriculum_entrypoint.py --train-worker \
  --arm-index 0 --nproc 8 --fresh \
  --run-dir /tmp/curriculum-linear10-flesch-benchmark \
  --wandb-mode disabled --local-smoke \
  --task-loss-eval-script .edullm/task_loss/eval_task_loss_olmo_core.py \
  --ladder-base-config .edullm/task_loss/ladder_base_config.yaml \
  --task-loss-nproc 8
```

Do not use `launch_curriculum_arm.sh` for this benchmark: its `LOCAL_SMOKE=1`
shortcut also adds `--no-task-loss`, which would under-measure production.

The exact platform forms are:

- production `linear10-flesch`:
  `.edullm/fixtures/curriculum-linear10-flesch-submission.json`
- credential-free benchmark:
  `.edullm/fixtures/curriculum-linear10-flesch-benchmark-submission.json`

Both select the provisioned `gpu-8xa100` profile (one `p4d.24xlarge`, 8 ×
A100) through the `olmo-core-train-4gpu` workload and launch eight ranks. The
benchmark contains no W&B/API credential and does not upload artifacts. Replace
the all-zero `commit_sha` with the immutable published branch commit before a
real submission; leave `image_digest` blank in **Submit a run** so the platform
resolves that commit's image.

Compile both forms locally against the unchanged platform checkout:

```powershell
$platform = "C:\alpha_ai\platform"
$fixtures = "$PWD\.edullm\fixtures"
$env:PYTHONPATH = "$platform\src"
foreach ($name in @("curriculum-linear10-flesch-submission",
                     "curriculum-linear10-flesch-benchmark-submission")) {
  py -3 "$platform\tools\compile_submission.py" `
    --inputs "$fixtures\$name.json" --config-dir "$platform\config" `
    --published-images "$fixtures\curriculum-published-image.json" `
    --submitter caiiris --repository-url https://github.com/edu-llm/OLMo-core `
    --output "$env:TEMP\$name-compiled.json"
}
```

`curriculum-published-image.json` is synthetic compiler evidence only. Local
compilation makes no live call and submits nothing.

At the catalog rate of $21.9576/hour, `gpu-8xa100` exceeds the platform's
$20/hour routine ceiling. A successful compile therefore returns approval
class `exception` and environment `run-approval-admin`; a team lead cannot
release it. `EDULLM_CHECKPOINT_CHECK=waived` is intentional because outputs
are W&B-only, and the waiver is recorded and displayed to the approving
platform admin.

Run the credential-free `linear10-flesch` benchmark first. If its complete end-to-end
runtime is `T` hours, including staging, checkpoints, and all evals, set every
production form's `maximum_runtime_hours` to `1.25 * T`, rounding upward only
to form precision. The benchmark fixture's 12-hour value is a safety bound,
not an estimate; a timeout is not a valid measurement.

The 8×A100 environment may admit only one job. Submit production arms in index
order, waiting for each to finish before submitting the next whenever capacity
is constrained. Do not submit the five-arm matrix as fan-out on that profile.
After all arms finish, perform each arm's EMA workflow sequentially if it also
needs the same worker capacity.

Branch code and fixtures are handoff-ready; production dispatch remains gated
on a published commit/image, the completed benchmark and 25%-padded runtime,
and per-run admin approval.
