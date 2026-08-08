# Running the P3 math-split evaluations on eduLLM

This is the platform-specific execution runbook for the final dense-versus-split
evaluation at checkpoint step `23166`. It is deliberately separate from
`p3-math-split-evals.md`, which describes the evaluator itself and its local
execution flow.

No platform run was submitted while writing this document.

## Sources and audit scope

This runbook was checked against:

- every Markdown document in `edu-llm/platform` at commit
  [`60ccf0fdfe573386b5ee688faa618de90dcdf0ef`](https://github.com/edu-llm/platform/tree/60ccf0fdfe573386b5ee688faa618de90dcdf0ef);
- the complete
  [Submit a run form](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml),
  its submission schema, current workload/compute/dataset catalogs, IAM
  templates, and admission checks;
- the last committed OLMo-core baseline on branch `edullm/p3-math-split` at
  [`fd8ef89d4129f559030fa1bd86c5a3227dcd58c2`](https://github.com/edu-llm/OLMo-core/tree/fd8ef89d4129f559030fa1bd86c5a3227dcd58c2);
- `the-platform.md`, `olmo-core.md`, `p3-math-split-evals.md`, and the assembled
  `corpus-v3/` in this checkout.

The root-level local copies of `the-platform.md` and `olmo-core.md` are stale on
important operational details. In particular, their workload names, approval
rules, runtime ceilings, dataset list, and image-scan guidance do not all match
current platform `main`. For submission decisions, use the current upstream
[platform guide](https://github.com/edu-llm/platform/blob/main/guides/the-platform.md),
[OLMo-core guide](https://github.com/edu-llm/platform/blob/main/guides/olmo-core.md),
and the output of `edullm check --json`.

## Current readiness: staged, but not yet GPU-readable

The evaluator code and both step-`23166` checkpoints are reachable from an
OLMo-core GPU job. The raw evaluator archive has now been uploaded to a
hash-addressed key in the landing bucket:

```text
s3://edullm-landing/_staging/eval/p3-math-split-evaluator/corpus-v3/79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e/corpus-v3.zip

size:          300229499 bytes
sha256:        79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e
S3 checksum:   eeebL5vRL71CWSb7N2q4brtN7PbVriUn4HlMT5XSiy4=
S3 version id: fQdQL.EEA.ibwTNddSpYH8jD17BetIRP
```

The upload used `If-None-Match: *` against a key confirmed absent immediately
beforeward. It created that one object, could not overwrite an existing object,
and did not alter any published v1/v2/v3 object.

The packed training release selected by
`formal-proof-premises-500m-v3` contains token shards, not the raw evaluator
JSONLs, held-out manifests, or train shards that `run_eval.py` fingerprints.
Therefore:

- do not select `formal-proof-premises-500m-v3` for these evaluations;
- do not point the evaluator at the legacy `corpus/` tree;
- do not point a GPU job directly at the landing URI, because the GPU workload
  role cannot read `edullm-landing`.

The CPU workload role can read the landing bucket and write its own recorded run
output. Form 0 below performs that one-object bridge. The later GPU jobs read
the resulting output object and verify the SHA-256 again. This keeps the
evaluation itself on the recorded platform path without modifying the packed
dataset release.

Longer term, the dataset owner may publish the archive as a sealed evaluator
artifact under `edullm-data`, but that is not required for the immediate run
once Form 0 succeeds.

## Why this uses OLMo-core rather than olmo-eval-full

These are custom P3 evaluator scripts in OLMo-core:

```text
src/scripts/train/p3_math_split/evals/export_checkpoint.py
src/scripts/train/p3_math_split/evals/run_eval.py
src/scripts/train/p3_math_split/evals/compare_arms.py
```

They also require sibling `mm_verify.py` and `provenance.py`, the OLMo-core
package, PyTorch, Transformers, the vendored Qwen tokenizer, and the Qwen export
implementation. All are in the OLMo-core research image built from the branch.

The registered `olmo-eval-full` image is not an alternative. Its current
published image supports only its mock provider and does not carry PyTorch or
the P3 evaluator.

## Platform access that this flow relies on

Use a GPU compute profile for every step that reads checkpoints or earlier eval
results. The deployed GPU workload role can read:

- both checkpoint roots below, including across team prefixes;
- the vendored tokenizer under `s3://edullm-data/`;
- a corpus artifact under a platform-approved GPU-readable prefix.

The CPU workload role cannot read the output-bucket checkpoint/result objects.
This is why even the CPU-only comparison step below uses the cheapest
provisioned GPU profile.

The final checkpoint roots are:

```text
dense:
s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints

split:
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints
```

Both `run_eval.py` and `compare_arms.py` reject reportable results from any step
other than `23166`.

The OLMo-core image includes `boto3`, `cached-path`, and `olmo_core.io`; it does
not explicitly install the AWS CLI. The commands below therefore use
`olmo_core.io.copy_file` and `copy_dir`, not `aws s3 cp`.

Metamath validity additionally needs these source databases from
[`metamath/set.mm`](https://github.com/metamath/set.mm) commit
`82830c78861b96e906d9868c30c35dbd98be5db5`:

```text
set.mm   7695d59e1c5c9182231e002425c82c86569bc044f30770bb32c276f7bafbf644
iset.mm  2851ed617e011b08b4d61c8312f34183aaf4da6b06b19512dac1e397ce709e4f
nf.mm    727a3707545e13ec53f03502eb07dc4635a8c176f275d4014a17fbd823e66083
```

The smoke and full commands download those exact commit-pinned files over
HTTPS. `run_eval.py --mm-dir` independently checks all three hashes against
`corpus-v3/metamath_sources.json` before reporting validity. A failed download
or hash mismatch stops the run rather than silently disabling a requested
validity metric.

## Scientific contract

For each arm, run all six families:

```text
enigma isabelle metamath mizar prf2 thproofs
```

The six source eval shards contain 4,191 rows in total:

```text
enigma 263 + isabelle 590 + metamath 494 + mizar 2485 + prf2 313 + thproofs 46
```

IDs are paired within family, not across one global ID namespace. After
`run_eval.py` applies its `text + EOS <= 16,384` runtime gate, all three
conditions must use the same complete context-eligible cohort for that family:

```text
facts_present facts_absent facts_corrupted
```

Use:

```text
checkpoint step:         23166
context length:          16384
maximum generated tokens: 8192
NLL chunk size:          256
evaluator seed:          20260801
decoding:                greedy
```

The evaluator scores target tokens only for both arms; premise-block tokens are
not included in evaluation loss.

Metamath additionally reports versioned tri-state generated-proof validity:

- verifier schema: `p3-metamath-tristate-v1`;
- `facts_present`: check against the correct facts visible in context;
- `facts_absent`: check against the canonical row facts withheld from context,
  so a recalled valid proof can pass;
- `facts_corrupted`: explicitly excluded from validity because checking against
  hidden canonical statements would not validate what the model saw;
- `unknown` remains distinct from `invalid`;
- `valid_rate_decided = valid / (valid + invalid)`, with unknowns reported
  separately.

Isabelle and ATP theorem validity are not reconstructible from the current
targets. Mizar validity is omitted because the native toolchain is not simple
or sufficiently cleared for this run.

## Before opening the submission form

1. Finish, commit, and push all evaluator changes to
   `edullm/p3-math-split`. The current Metamath integration is uncommitted, so
   the older `fd8ef89...` image is not sufficient.
2. Confirm the working tree is clean:

   ```bash
   git status --short
   ```

3. Record the full commit:

   ```bash
   git rev-parse HEAD
   ```

4. Confirm **Build eduLLM research image** succeeded for that exact commit.
5. Wait for the image security scan after the build. A green build alone is not
   sufficient.
6. Run Form 0 below and retain its run id as `CORPUS_STAGE_RUN_ID`.
7. Choose an existing W&B project. The form requires `wandb_project` even though
   this evaluator does not create a W&B run itself.
8. Install or refresh the terminal client if you will use it to inspect runs:

   ```bash
   uv tool install --force git+https://github.com/edu-llm/platform
   edullm --version
   gh auth login
   ```

   `edullm --version` must be at least `3.4.8`.

Do not use a branch name in place of the final SHA on reportable runs. Every
form below must use the same new 40-character commit whose image build and
security scan completed.

## Running through GitHub Actions

All five submissions use the platform's
[Submit a run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml)
workflow:

1. Open the workflow link and select **Run workflow**.
2. In **Use workflow from**, select platform branch `main`. This selects the
   platform workflow, not the OLMo-core code.
3. Enter the OLMo-core 40-character SHA in `commit_sha`.
4. Fill every field exactly as specified by the relevant form below.
5. Paste the command as one line beginning with `bash -lc`. Do not wrap the
   entire value in another pair of quotes.
6. Select the green **Run workflow** button.
7. Open the workflow run it creates. The compile stage resolves the image,
   validates the command, prices the request, and mints the `run_...` id.
8. Record both the workflow URL and run id. If the run waits at an approval
   gate, a team lead must release it; cancelling the workflow is not a release
   or a Batch cancellation.
9. `ADMITTED` means the job reached AWS. The Submit workflow does not continue
   tracking the Batch job after admission.

Do not submit Forms 2 and 3 until Form 1 has completed successfully. Forms 2
and 3 may then be submitted in parallel. Submit Form 4 only after both result
objects exist. Before full runs, inspect the smoke metadata for
`excluded_over_context_examples`. Any nonzero value is a stop for manual review;
the expected final corpus has zero exclusions, but the runtime output—not this
document—is the proof.

## Recommended job layout

Use five recorded submissions:

1. one CPU bridge from the landing object to a run output;
2. one two-arm, one-row smoke job;
3. one full dense job;
4. one full split job;
5. one paired comparison job after both arm jobs finish.

This isolates expensive failures, allows the two full arm jobs to run in
parallel, and avoids retaining both Hugging Face exports on one small root
volume. Do not start the full jobs until the smoke job has completed and
uploaded both smoke JSON files.

The full wall time has not been measured on platform hardware. A full arm is
4,191 rows across three conditions, with both teacher-forced scoring and greedy
generation. If the smoke-derived projection exceeds the 24-hour workload
ceiling, stop and request a dedicated null-checkpoint evaluation profile or a
reviewed per-family runner. Do not assume the current monolithic result survives
a timeout: `run_eval.py` writes its result JSON at the end.

## Form 0: make the staged archive GPU-readable

Open
[Submit a run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml)
and fill every field as follows:

| Form field | Value |
| --- | --- |
| `repository` | `OLMo-core` |
| `commit_sha` | Full 40-character SHA from `git rev-parse HEAD` |
| `image_digest` | Leave blank |
| `workload_profile` | `olmo-core-check` |
| `compute_profile` | `cpu-32vcpu` |
| `dataset_release` | `none` |
| `team` | `memory-split` |
| `experiment` | `p3-math-split-eval-corpus-stage` |
| `wandb_project` | Your existing W&B project |
| `command` | Paste the command below as one line |
| `maximum_runtime_hours` | Leave blank |
| `maximum_attempts` | Leave blank |
| `fanout_size` | Leave blank |
| `fanout_index_parameter` | Leave blank |
| `edullm_version` | Leave blank |

Paste this command unchanged:

```bash
bash -lc 'set -euo pipefail; SOURCE="s3://edullm-landing/_staging/eval/p3-math-split-evaluator/corpus-v3/79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e/corpus-v3.zip"; SHA="79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e"; LOCAL=/tmp/corpus-v3.zip; TARGET="${EDULLM_OUTPUT_PREFIX}corpus-v3.zip"; python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2])" "$SOURCE" "$LOCAL"; printf "%s  %s\n" "$SHA" "$LOCAL" | sha256sum -c -; python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2], save_overwrite=True)" "$LOCAL" "$TARGET"; printf "staged evaluator archive at %s\n" "$TARGET"'
```

After it succeeds, record its run id as `CORPUS_STAGE_RUN_ID`. The exact input
for every GPU job below is:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/<CORPUS_STAGE_RUN_ID>/corpus-v3.zip
```

The smoke job downloads that object and checks the SHA-256 before extracting it.
Do not continue if the checksum fails.

## Form 1: two-arm smoke

Open
[Submit a run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml)
and fill every field as follows.

| Form field | Value |
| --- | --- |
| `repository` | `OLMo-core` |
| `commit_sha` | Full 40-character SHA from `git rev-parse HEAD` |
| `image_digest` | Leave blank |
| `workload_profile` | `olmo-core-check` |
| `compute_profile` | `gpu-1xa10g` |
| `dataset_release` | `none` |
| `team` | `memory-split` |
| `experiment` | `p3-math-split-eval-smoke` |
| `wandb_project` | Your existing W&B project |
| `command` | Paste the command below as one line |
| `maximum_runtime_hours` | Leave blank |
| `maximum_attempts` | Leave blank |
| `fanout_size` | Leave blank |
| `fanout_index_parameter` | Leave blank |
| `edullm_version` | Leave blank |

Replace only `REPLACE_CORPUS_STAGE_RUN_ID` before pasting:

```bash
bash -lc 'set -euo pipefail; copy_file() { python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2])" "$1" "$2"; }; copy_dir() { python -c "import sys; from olmo_core.io import copy_dir; copy_dir(sys.argv[1], sys.argv[2])" "$1" "$2"; }; CORPUS_V3_URI="s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/REPLACE_CORPUS_STAGE_RUN_ID/corpus-v3.zip"; CORPUS_SHA="79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e"; MM_BASE="https://raw.githubusercontent.com/metamath/set.mm/82830c78861b96e906d9868c30c35dbd98be5db5"; WORK=/tmp/p3-eval-smoke; RESULTS="$WORK/results"; MM_DIR="$WORK/mm"; mkdir -p "$WORK"/hf "$WORK"/staging "$RESULTS" "$MM_DIR"; copy_file "$CORPUS_V3_URI" "$WORK/corpus-v3.zip"; printf "%s  %s\n" "$CORPUS_SHA" "$WORK/corpus-v3.zip" | sha256sum -c -; python -m zipfile -e "$WORK/corpus-v3.zip" "$WORK"; copy_file "$MM_BASE/set.mm" "$MM_DIR/set.mm"; copy_file "$MM_BASE/iset.mm" "$MM_DIR/iset.mm"; copy_file "$MM_BASE/nf.mm" "$MM_DIR/nf.mm"; python src/scripts/train/p3_math_split/evals/export_checkpoint.py --run s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints --step 23166 --out "$WORK/hf/dense" --work-dir "$WORK/staging/dense"; python src/scripts/train/p3_math_split/evals/run_eval.py --model "$WORK/hf/dense" --arm dense --corpus "$WORK/corpus-v3" --mm-dir "$MM_DIR" --families enigma isabelle metamath mizar prf2 thproofs --conditions facts_present --limit 1 --batch-size 1 --context-length 16384 --max-new-tokens 32 --nll-chunk-size 256 --seed 20260801 --out "$RESULTS/dense-smoke.json"; rm -rf "$WORK/hf/dense" "$WORK/staging/dense"; python src/scripts/train/p3_math_split/evals/export_checkpoint.py --run s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints --step 23166 --out "$WORK/hf/split" --work-dir "$WORK/staging/split"; python src/scripts/train/p3_math_split/evals/run_eval.py --model "$WORK/hf/split" --arm split --corpus "$WORK/corpus-v3" --mm-dir "$MM_DIR" --families enigma isabelle metamath mizar prf2 thproofs --conditions facts_present --limit 1 --batch-size 1 --context-length 16384 --max-new-tokens 32 --nll-chunk-size 256 --seed 20260801 --out "$RESULTS/split-smoke.json"; copy_dir "$RESULTS" "${EDULLM_OUTPUT_PREFIX}results"'
```

The expected output prefix is:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/<smoke-run-id>/results/
```

It must contain both `dense-smoke.json` and `split-smoke.json`.
Each smoke result must contain all six families, one evaluated example per
family, and the pre-limit source/context-eligible/excluded counts for every
family.

## Form 2: full dense evaluation

Use these form values:

| Form field | Value |
| --- | --- |
| `repository` | `OLMo-core` |
| `commit_sha` | The exact same full SHA used for the smoke |
| `image_digest` | Leave blank |
| `workload_profile` | `olmo-core-train` |
| `compute_profile` | `gpu-1xl40s` |
| `dataset_release` | `none` |
| `team` | `memory-split` |
| `experiment` | `p3-math-split-eval-step23166` |
| `wandb_project` | The same W&B project used for the smoke |
| `command` | Paste the dense command below as one line |
| `maximum_runtime_hours` | Leave blank; use the profile ceiling |
| `maximum_attempts` | `1` |
| `fanout_size` | Leave blank |
| `fanout_index_parameter` | Leave blank |
| `edullm_version` | Leave blank |

`olmo-core-train` carries a checkpoint contract, but evaluation creates results
rather than resumable training checkpoints. The command therefore contains the
recorded `EDULLM_CHECKPOINT_CHECK=waived` token. With one attempt, a lost machine
does not silently restart the whole arm.

Replace only `REPLACE_CORPUS_STAGE_RUN_ID`:

```bash
bash -lc 'set -euo pipefail; copy_file() { python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2])" "$1" "$2"; }; copy_dir() { python -c "import sys; from olmo_core.io import copy_dir; copy_dir(sys.argv[1], sys.argv[2])" "$1" "$2"; }; CORPUS_V3_URI="s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/REPLACE_CORPUS_STAGE_RUN_ID/corpus-v3.zip"; CORPUS_SHA="79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e"; MM_BASE="https://raw.githubusercontent.com/metamath/set.mm/82830c78861b96e906d9868c30c35dbd98be5db5"; CHECKPOINT_ROOT="s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints"; WORK=/tmp/p3-eval-dense; RESULTS="$WORK/results"; MM_DIR="$WORK/mm"; mkdir -p "$WORK"/hf "$WORK"/staging "$RESULTS" "$MM_DIR"; copy_file "$CORPUS_V3_URI" "$WORK/corpus-v3.zip"; printf "%s  %s\n" "$CORPUS_SHA" "$WORK/corpus-v3.zip" | sha256sum -c -; python -m zipfile -e "$WORK/corpus-v3.zip" "$WORK"; copy_file "$MM_BASE/set.mm" "$MM_DIR/set.mm"; copy_file "$MM_BASE/iset.mm" "$MM_DIR/iset.mm"; copy_file "$MM_BASE/nf.mm" "$MM_DIR/nf.mm"; copy_file "$CHECKPOINT_ROOT/step23166/config.json" "$RESULTS/dense-training-config.json"; python src/scripts/train/p3_math_split/evals/export_checkpoint.py --run "$CHECKPOINT_ROOT" --step 23166 --out "$WORK/hf/dense" --work-dir "$WORK/staging/dense"; cp "$WORK/hf/dense/model_provenance.json" "$RESULTS/dense-model-provenance.json"; EDULLM_CHECKPOINT_CHECK=waived python src/scripts/train/p3_math_split/evals/run_eval.py --model "$WORK/hf/dense" --arm dense --corpus "$WORK/corpus-v3" --mm-dir "$MM_DIR" --families enigma isabelle metamath mizar prf2 thproofs --conditions facts_present facts_absent facts_corrupted --batch-size 8 --context-length 16384 --max-new-tokens 8192 --nll-chunk-size 256 --seed 20260801 --out "$RESULTS/dense.json"; copy_dir "$RESULTS" "${EDULLM_OUTPUT_PREFIX}results"'
```

Record the run id as `DENSE_EVAL_RUN_ID`. Its result URI will be:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/<DENSE_EVAL_RUN_ID>/results/dense.json
```

## Form 3: full split evaluation

Use the same values as Form 2 except for the command. The experiment remains
`p3-math-split-eval-step23166`, so the cost view groups the two arms together.

Replace only `REPLACE_CORPUS_STAGE_RUN_ID`:

```bash
bash -lc 'set -euo pipefail; copy_file() { python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2])" "$1" "$2"; }; copy_dir() { python -c "import sys; from olmo_core.io import copy_dir; copy_dir(sys.argv[1], sys.argv[2])" "$1" "$2"; }; CORPUS_V3_URI="s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/REPLACE_CORPUS_STAGE_RUN_ID/corpus-v3.zip"; CORPUS_SHA="79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e"; MM_BASE="https://raw.githubusercontent.com/metamath/set.mm/82830c78861b96e906d9868c30c35dbd98be5db5"; CHECKPOINT_ROOT="s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints"; WORK=/tmp/p3-eval-split; RESULTS="$WORK/results"; MM_DIR="$WORK/mm"; mkdir -p "$WORK"/hf "$WORK"/staging "$RESULTS" "$MM_DIR"; copy_file "$CORPUS_V3_URI" "$WORK/corpus-v3.zip"; printf "%s  %s\n" "$CORPUS_SHA" "$WORK/corpus-v3.zip" | sha256sum -c -; python -m zipfile -e "$WORK/corpus-v3.zip" "$WORK"; copy_file "$MM_BASE/set.mm" "$MM_DIR/set.mm"; copy_file "$MM_BASE/iset.mm" "$MM_DIR/iset.mm"; copy_file "$MM_BASE/nf.mm" "$MM_DIR/nf.mm"; copy_file "$CHECKPOINT_ROOT/step23166/config.json" "$RESULTS/split-training-config.json"; python src/scripts/train/p3_math_split/evals/export_checkpoint.py --run "$CHECKPOINT_ROOT" --step 23166 --out "$WORK/hf/split" --work-dir "$WORK/staging/split"; cp "$WORK/hf/split/model_provenance.json" "$RESULTS/split-model-provenance.json"; EDULLM_CHECKPOINT_CHECK=waived python src/scripts/train/p3_math_split/evals/run_eval.py --model "$WORK/hf/split" --arm split --corpus "$WORK/corpus-v3" --mm-dir "$MM_DIR" --families enigma isabelle metamath mizar prf2 thproofs --conditions facts_present facts_absent facts_corrupted --batch-size 8 --context-length 16384 --max-new-tokens 8192 --nll-chunk-size 256 --seed 20260801 --out "$RESULTS/split.json"; copy_dir "$RESULTS" "${EDULLM_OUTPUT_PREFIX}results"'
```

Record the run id as `SPLIT_EVAL_RUN_ID`. Its result URI will be:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/<SPLIT_EVAL_RUN_ID>/results/split.json
```

## Form 4: paired comparison

Wait until both full arm jobs have succeeded and their result objects exist.
Then submit:

| Form field | Value |
| --- | --- |
| `repository` | `OLMo-core` |
| `commit_sha` | The exact same full SHA used for both arm evaluations |
| `image_digest` | Leave blank |
| `workload_profile` | `olmo-core-check` |
| `compute_profile` | `gpu-1xt4` |
| `dataset_release` | `none` |
| `team` | `memory-split` |
| `experiment` | `p3-math-split-eval-step23166` |
| `wandb_project` | The same W&B project |
| `command` | Paste the comparison command below as one line |
| `maximum_runtime_hours` | Leave blank |
| `maximum_attempts` | Leave blank |
| `fanout_size` | Leave blank |
| `fanout_index_parameter` | Leave blank |
| `edullm_version` | Leave blank |

The comparison itself does not need a GPU, but a GPU compute profile is required
because the CPU workload role cannot read the two earlier result prefixes.

Replace `REPLACE_DENSE_EVAL_RUN_ID` and `REPLACE_SPLIT_EVAL_RUN_ID`:

```bash
bash -lc 'set -euo pipefail; copy_file() { python -c "import sys; from olmo_core.io import copy_file; copy_file(sys.argv[1], sys.argv[2])" "$1" "$2"; }; BASE="s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs"; DENSE_RUN_ID="REPLACE_DENSE_EVAL_RUN_ID"; SPLIT_RUN_ID="REPLACE_SPLIT_EVAL_RUN_ID"; WORK=/tmp/p3-eval-compare; mkdir -p "$WORK"; copy_file "$BASE/$DENSE_RUN_ID/results/dense.json" "$WORK/dense.json"; copy_file "$BASE/$SPLIT_RUN_ID/results/split.json" "$WORK/split.json"; copy_file "$BASE/$DENSE_RUN_ID/results/dense-training-config.json" "$WORK/dense-training-config.json"; copy_file "$BASE/$SPLIT_RUN_ID/results/split-training-config.json" "$WORK/split-training-config.json"; python src/scripts/train/p3_math_split/evals/compare_arms.py --dense "$WORK/dense.json" --split "$WORK/split.json" --dense-config "$WORK/dense-training-config.json" --split-config "$WORK/split-training-config.json" --n-boot 10000 --seed 20260801 --out "$WORK/comparison.json"; copy_file "$WORK/comparison.json" "${EDULLM_OUTPUT_PREFIX}results/comparison.json"'
```

The final comparison will be:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/<comparison-run-id>/results/comparison.json
```

## Submission checks

Before each expensive submission, use the current tool and read its JSON rather
than relying on prices or approval thresholds copied into a document:

```bash
edullm check --json \
  --team memory-split \
  --experiment p3-math-split-eval-step23166 \
  --dataset none
```

The OLMo-core branch currently has no `.edullm/run.yaml`, and `edullm` takes the
command from that committed file rather than from a command-line flag. Therefore:

- the exact GitHub form above is immediately usable once the corpus URI and image
  prerequisites are satisfied;
- to use `edullm check` against the exact P3 command, first add a reviewed
  `.edullm/run.yaml` containing that command, commit it, push it, and wait for the
  resulting image;
- do not let an automatically generated placeholder command replace the commands
  in this runbook;
- never use `--force` to bypass a refusal.

For every `check --json`, read and report:

- `refusals` by stable `code`;
- `deferred` image checks;
- `cost.maximum_compute_cost_usd`;
- `approval_class`;
- the exact resolved commit, workload, compute profile, team, and dataset.

Do not quote an approval threshold, price, or runtime bound from this document.
Those are reviewed configuration and can change.

## Monitoring and stopping

Keep every run id printed by the submission workflow.

### GitHub Actions UI

Use
[Look at a run, or stop it](https://github.com/edu-llm/platform/actions/workflows/cancel-run.yml):

1. Open the workflow and select **Run workflow**.
2. Select platform branch `main`.
3. Enter the exact `run_...` id. Leaving it blank reports your own most recent
   Batch-visible run, but an explicit id is safer for this five-run sequence.
4. Leave **stop** unticked and `reason` blank.
5. Select **Run workflow**, open the resulting workflow run, then read its job
   summary and the **Say what the run is doing, and stop it if asked** logs.

The report searches every configured queue and shows the Batch state, queue,
status reason, attempts, exit information, CloudWatch stream, and the last 50
lines of container output in the `### The last lines this run printed` summary
section. Use that tail to confirm:

- Form 0 printed the final GPU-readable corpus URI;
- Form 1 wrote both smoke JSON files;
- Forms 2 and 3 wrote their full result JSONs;
- Form 4 reached the final comparisons and wrote `comparison.json`.

`RUNNABLE` means the job is waiting for a machine and is not billing. The reason
in the report distinguishes an ordinary wait from an unavailable shape.

The Submit workflow stops at `ADMITTED`; refreshes there will never show a
completed Batch job. Dispatch the look-up workflow above instead.

GitHub Actions is a status and log-tail viewer, not an S3 file browser. Fifty
lines cannot show the complete six-family by three-condition comparison.
The exact result object URIs are listed in each form section. Treat the complete
machine-readable `comparison.json` at its recorded S3 URI as the acceptance
artifact; the Form 4 log tail is only a completion and spot-check surface.

### Optional terminal equivalent

```bash
edullm status --json <run-id>
```

This can be polled while the submission is still on GitHub. Once
`needs_a_dispatch` is true, use the slower AWS-backed forms only when needed:

```bash
edullm status <run-id>
edullm logs <run-id>
```

Do not put those two commands in a polling loop. A job in `RUNNABLE` is waiting
for a machine and bills nothing.

### Stopping through GitHub Actions

Open the same **Look at a run, or stop it** workflow:

1. enter the exact run id;
2. tick **stop**;
3. provide a non-empty reason;
4. dispatch the workflow and read its summary.

The terminal equivalent is:

```bash
edullm cancel <run-id> --reason "why this evaluation is being stopped"
```

Do not cancel the **Submit a run** workflow itself; that can leave the Batch job
running.

## Required outputs and acceptance checks

Corpus bridge run:

```text
corpus-v3.zip
```

Smoke run:

```text
results/dense-smoke.json
results/split-smoke.json
```

Dense run:

```text
results/dense.json
results/dense-training-config.json
results/dense-model-provenance.json
```

Split run:

```text
results/split.json
results/split-training-config.json
results/split-model-provenance.json
```

Comparison run:

```text
results/comparison.json
```

Accept the evaluation only if:

1. both arm results declare schema `p3-eval-v8`;
2. both declare checkpoint step `23166`;
3. all six families are present;
4. within every family, all three conditions use the full
   `context_eligible_examples` cohort and identical row IDs;
5. dense and split input/corpus/tokenizer provenance matches;
6. both Metamath results declare verifier schema
   `p3-metamath-tristate-v1` and verified source hashes;
7. Metamath `facts_present` and `facts_absent` report separate
   valid/invalid/unknown counts, while `facts_corrupted` reports no validity
   rate;
8. `compare_arms.py` exits successfully without a compatibility waiver and
   writes comparison schema `p3-comparison-v4`;
9. the comparator excludes unknown Metamath pairs from the decided denominator
   rather than counting them as invalid;
10. the sum of `context_eligible_examples` across families is 4,191 and every
    family reports zero `excluded_over_context_examples`;
11. every result object is present under its recorded run prefix.

## Known limitations

- The archive is staged in `edullm-landing`, but the recorded CPU bridge job
  must copy it into a GPU-readable run output before evaluation.
- Full-arm runtime on A10G/L40S has not been measured.
- `run_eval.py` does not checkpoint partial evaluation progress.
- The current platform has no dedicated long, null-checkpoint P3 evaluation
  workload; full runs reuse `olmo-core-train` with a recorded checkpoint waiver.
- The evaluator writes JSON locally and needs the explicit S3 copy at the end.
- Metamath source databases are fetched at runtime from the commit-pinned
  `raw.githubusercontent.com` URLs. Outbound HTTPS or GitHub availability can
  therefore fail an otherwise healthy run; source hashes prevent substitution
  but do not remove this availability dependency.
- H100 profiles are not provisioned. Do not put `gpu-1xh100` or `gpu-8xh100` in
  a plan.
- Generated-proof validity is reportable only for Metamath
  `facts_present`/`facts_absent`; the other families retain NLL, accuracy,
  exact-match, and generation-budget metrics.
