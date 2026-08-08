# P3 math split: pre-final-run checklist

Current parent-owned progress, subagent, bug, and acceptance status:
`P3_WORK_STATUS.md`. This checklist contains historical completed boxes and
numbers that are being reconciled; where they disagree, do not execute them.
`P3_DECISION_LEDGER.md` remains authoritative for approved scientific choices.

This is the ordered path from the current local state to the final dense/split jobs.
Do not delete partial generated artifacts. If tokenization is interrupted, preserve
the tree, inventory completed shard ordinals, and decide whether to resume or write
to a new directory.

## Current v3 platform launch rule (authoritative)

The promoted corpus is `pretrain/formal-proof-premises-500m/v3`. Until the
platform submission registry exposes a named v3 release, select **Data: None**
(`dataset_release=none`) in the form and bind the immutable corpus explicitly in
every P3 command:

```text
--dataset-id pretrain/formal-proof-premises-500m
--dataset-version v3
--dataset-tokenizer tokenizer/qwen25-vendored/v1
```

Do not set `EDULLM_DATASET_*` variables yourself and do not use a raw S3 path.
The command-line values are validated by `train_platform.py`, then resolved
through `edullm_data.read.dataset_paths(..., split="train")`. With Data: None,
the platform run manifest and saved `dataset_release` field say `none`; the saved
config must therefore be checked for the authoritative
`dataset_id=pretrain/formal-proof-premises-500m` and `dataset_version=v3`.

Verified locally with the image-pinned `edullm-data` 0.5.0 reader:

- six v3 train shards, dtype `uint32`, little-endian, tokenizer
  `tokenizer/qwen25-vendored/v1`;
- 467,206,144 packed train tokens;
- global batch 262,144 tokens, 1,782 complete batches/epoch;
- 13 loader epochs = **23,166 steps**.

## Current verified state

- [x] Qwen2.5-0.5B tokenizer vendored at `tokenizers/qwen25-vendored/`.
- [x] Corpus deep scan and train/eval leakage checks pass.
- [x] Dense and split configs differ only in `arm`.
- [x] Packing-aware fact mask has regression tests.
- [x] Separator search uses `---\nGOAL` (`[10952, 15513, 969]`), measured once
      in every one of 258,316 source documents.
- [x] Intra-document attention is enabled and FlashAttention2 is required explicitly.
- [x] Released 494M Qwen checkpoint maps strictly into the OLMo-core port.
- [x] Local ruff and P3/Qwen suites pass.
- [x] Current effective checkpoint interval is 2,000 steps; pruning is disabled.
- [x] Tokenizer supports batched multi-core encoding, persistent per-corpus caches,
      batch-level resume, atomic shards, and completed-group resume.
- [x] Final local check after causal-label review: ruff clean; 89 tests pass,
      19 hardware-only tests skip.
- [x] Fixed and regression-tested Qwen's shared EOS/pad id: genuine proof-ending
      EOS remains supervised, while only the repeated-EOS padding tail is ignored.
- [x] Fixed and regression-tested OLMo's left-shifted label semantics:
      `labels[i]` predicts `input_ids[i+1]`, so fact/padding masks are shifted to
      target positions. The prior image skipped every proof's first goal token and
      scored the first padding target. Exact final fractions are now dense 99.915%
      and split 83.803%.
- [x] Fake-published-corpus integration test builds the complete platform config
      and pins train-only paths, dtype, sequence/doc masking, Qwen/FlashAttention,
      all optimizer controls, fixed divisor, steps, checkpoint retention, and W&B.
- [x] Strict real-model smoke: all 494,032,768 Qwen parameters load exactly;
      dense/split losses are finite and distinct; gradients flow; both tiny arm
      runs decrease loss; fixed-divisor substitution is active.
- [x] Platform tokenizer resolver downloaded validated
      `tokenizer/qwen25-vendored/v1` from S3 and derived `[10952, 15513, 969]`.
- [x] Read-only Gate A with the proposed family floor of 128 passes corpus v2
      with zero violations.

## Decision gate before publishing

- [x] Final v3 release uses the 16,384-token build:
  - 181,652 sealed train rows and 4,191 sealed eval rows;
  - 28,516 packed train sequences / 467,206,144 train tokens;
  - 614 packed validation sequences / 10,059,776 validation tokens;
  - 477,265,920 total stored tokens;
  - zero tokenizer-time overlength drops (the Metamath 16k eligibility policy
    was applied before final held-out selection).
- [x] Keep 13 epochs. The v3 train shards give 6,073,679,872 packed token
      exposures and exactly 23,166 complete loader steps. Do not reuse v2's
      fact-saturation counts; recompute them from v3 if they are reported.
- [x] Final dataset id: `pretrain/formal-proof-premises-500m`
      (`validate_dataset_id`: PASS).

## Finish and verify tokenization

- [x] Fresh v3 tokenization finished under the persistent root
      `.p3-work/full13/tokenized-v3`; no v1/v2 payload was read or copied.
- [x] Both completion manifests exist:
  - `.p3-work/full13/tokenized-v3/train_meta.json`
  - `.p3-work/full13/tokenized-v3/val_meta.json`
- [x] Verified each manifest:
  - tokenizer composite SHA-256 is the fixed Qwen seal `aa90434a…`;
  - dtype `uint32`, byte order `little`, header bytes zero;
  - `sequence_length == 16384`, `packed == true`;
  - separator ids are `[10952, 15513, 969]`;
  - all listed files exist;
  - each file size equals `tokens × 4`;
  - total instances/tokens equal the per-corpus sums;
  - train: 28,516 instances / 466,733,493 real tokens / 0.10% padding /
    zero over-length documents dropped;
  - val: 614 instances / 9,966,430 real tokens / 0.93% padding /
    zero over-length documents dropped.
- [x] Read all 12 staged shards as `<u4`; byte sizes equal `tokens × 4`,
      maximum token id is 151,643 (<151,665 backend vocabulary), EOS is present,
      every shard has at least 128 distinct IDs, and every file is exactly
      sequence-length aligned.
- [x] The encoder appended EOS to every source document and performed a full-corpus
      exactly-one separator check before writing each completed group.
- [x] Created `.p3-work/full13/publish-stage-v3` using hard links from only the
      fresh v3 token output. It contains exactly the 12 manifest-declared
      `.u32le.bin` payloads—no caches, done markers, JSON controls, or symlinks.
- [x] The aggregate v3 shard-digest root differs from v2 and no individual v3
      shard digest equals its v2 counterpart.

## Publish data (tokenizer first)

- [x] Broker login active and `sbsandbox` shell profile installed.
- [x] Published `tokenizer/qwen25-vendored/v1` to `edullm-landing` with profile
      `tokenizer/v1` (2 objects, 11,422,589 bytes).
- [x] Tokenizer promoted and cataloged as `tokenizer/qwen25-vendored/v1`.
      Validation seal pins manifest SHA-256
      `52e4a776b0256bc6daac992380d9dacdecb0823db40dfc9123d56e0c3f9c74c9`.
- [x] Published, deployed-validated, promoted, and cataloged
      `pretrain/formal-proof-premises-500m/v3`:
  - objects: 12;
  - bytes: 1,909,063,680;
  - dataset SHA-256:
    `7360db01af5cfc3ec7cea02ffaafc5c6b1b0b4536796c44e6b8c9b3da9862a69`;
  - token manifest SHA-256:
    `ef320b4345493de7fa38bf197e05756315ace192566eef76c56f88b292d42a58`;
  - tokenizer dependency manifest:
    `52e4a776b0256bc6daac992380d9dacdecb0823db40dfc9123d56e0c3f9c74c9`.
- [x] Both image-pinned `edullm-data` 0.5.0 and publisher 0.8.0 readers
      recompute the validation seal and resolve six train paths / 467,206,144
      rows and six val paths / 10,059,776 rows.
- [x] Until a named v3 registry release exists, select Data: None and use the
      three explicit immutable dataset flags shown at the top of this file.
- [x] A real S3-backed `train_platform.py --dry-run` with Data: None and those
      flags resolves v3 and prints Qwen vocab 151,936, FlashAttention2, sequence
      length 16,384, global batch 262,144, rank microbatch 16,384, LR 2e-5,
      warmup 2,400, **23,166 steps**, save interval 2,000, and the fixed divisor
      262,144.

## Commit and build the image

- [ ] Put the P3 corpus builders, tests, source manifests, and decision ledger
      under a dedicated Git branch/commit before publishing the repaired corpus.
      They are currently untracked in `memorysplit-requery-exact`; local bytes
      without a Git identity are not a reproducible release recipe.

- [x] In OLMo-core, reviewed and staged only six intended correctness/test files.
      Do not `git add -A`:
      `graphify-out/`, `.agents/`, and `next-steps.md` are currently untracked.
- [x] Committed the post-`4f2e05f` local correctness fixes:
  - `train_module.py` (real EOS vs padding);
  - updated derived/packed-mask tests;
  - new packed-artifact and platform-config integration tests.
- [x] Run the pinned build check:

  ```bash
  uv run --with ruff==0.15.22 ruff check --no-cache .
  ```

- [x] Run:

  ```bash
  .venv/bin/python -m pytest \
    src/test/scripts/p3_math_split/ \
    src/test/nn/transformer/qwen_test.py -q
  ```

- [x] Committed intended files on `edullm/p3-math-split`:
      `e81fafa fixed fact packing`.
- [x] Pushed branch; local/remote divergence is `0 0`.
      Full SHA: `e81fafab0c587f2d15ca9d1a10af707629692f0d`.
      Only excluded untracked paths remain: `.agents/`, `graphify-out/`,
      and `next-steps.md`.
- [x] **Build eduLLM research image** passed for
      `e81fafab0c587f2d15ca9d1a10af707629692f0d`.
      ECR image digest:
      `sha256:96cbc14fbdb101972e62e964fcfb8c9f4713350dabacee93601ed5be754e43e1`.
- [x] Registry security scan completed successfully. Findings visible for
      platform review: 4 critical, 8 high, 3 medium.
- [x] OLMo fixes committed and pushed on `edullm/p3-math-split`:
      `87df895fe5136d9d0af1010c638e4ccfa8b8f2ba`.
      Image `e81fafa…` predates:
  - shifted-label and EOS/padding corrections;
  - the real `DerivedMaskTrainModule` constructor fix;
  - post-FSDP loading of pretrained Qwen weights;
  - pinned Transformers 5.14.1, Tokenizers 0.22.2, and the official
    FlashAttention 2.8.3 Torch-2.9/CUDA-12 binary wheel.
  P3 verification was 104 passed / 19 hardware-only skipped; repository ruff
  and diff checks passed.
- [ ] Build and security-scan the research image for `87df895…`; do not use the
      old `e81fafa…` image for another GPU check.

## Platform checks

The submission form supplies `WANDB_PROJECT` and `WANDB_ENTITY`; no local W&B
configuration is required. Enter the project in the form.

### 1. Generic CPU path check (can be done before our dataset exists)

- [x] Generic CPU path check completed:
  - repository: `OLMo-core`
  - commit: `main`
  - workload profile: `olmo-core-check`
  - compute profile: `cpu-32vcpu`
  - team: `scratch`
  - dataset release: `none`
  - leave the default command
  - W&B project: your project

This tests platform submission → machine → result, not our code.

### 2. Our image + published-data CPU dry run

**IAM blocker:** the CPU workload role
`sbsandbox-intern-edullm-batch-workload` currently has no `s3:GetObject` or
`s3:ListBucket` permission on `edullm-data`; it can only write team outputs.
This form will fail while resolving the corpus unless the role is updated.
The GPU workload role does have bucket-wide read/list access, so the same dry
run can be performed on `gpu-1xa10g` if needed.

- [x] Skipped by decision. Use the combined A10G GPU visibility + published-data
      dry run below instead.

<details>
<summary>Retained CPU form for reference</summary>

- Fill the **Submit a run** form:

  | field | value |
  |---|---|
  | `repository` | `OLMo-core` |
  | `commit_sha` | full SHA from `git rev-parse HEAD` |
  | `workload_profile` | `olmo-core-check` |
  | `compute_profile` | `cpu-32vcpu` |
  | `team` | `memory-split` |
  | `experiment` | `p3-math-split-preflight` |
  | Data / `dataset_release` | `None` / `none` |
  | `wandb_project` | your W&B project |
  | advanced fields | leave blank/default |
  | `command` | paste the command below |

- Command field:

  ```bash
  bash -lc 'python src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR" --dry-run'
  ```

- Confirm the printed config names the right dataset version, tokenizer,
      uint32/little-endian shards, sequence length, steps, warmup, and FlashAttention2.

</details>

### 3. Combined GPU visibility + published-data dry run

Use a full built SHA from `edullm/p3-math-split`. Do **not** use `main` or
`edullm/qwen25-tokenizer`: the latter enables Qwen only in the stock dense-only
`.edullm/train_on_corpus.py`, while this experiment also needs the custom split
module, packed-document attention, P0 fixes, and image dependencies.

- [x] Submitted the form as run
      `run_019fd243-ebcf-7088-a135-aab0b080618d`:

  | field | value |
  |---|---|
  | `repository` | `OLMo-core` |
  | `commit_sha` | same full built SHA |
  | `workload_profile` | `olmo-core-check` |
  | `compute_profile` | `gpu-1xa10g` |
  | `team` | `memory-split` |
  | `experiment` | `p3-math-split-preflight` |
  | Data / `dataset_release` | `None` / `none` |
  | `wandb_project` | your W&B project |
  | advanced fields | leave blank/default |
  | `command` | paste the command below |

- [x] Command field matched:

  ```bash
  bash -lc 'python -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1).cuda(); print(torch.cuda.get_device_name(),x)" && python src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR" --dry-run'
  ```

This proves GPU visibility and published corpus/tokenizer/config resolution under
the GPU workload role. Because Data is None, the platform record says
`dataset_release=none`; the printed and saved config must name the exact v3
dataset ID/version/tokenizer supplied on the command line.

- [x] First A10G attempt proved CUDA visibility and resolved all six uint32
      corpus shards, then correctly failed with exit 70 because image `e81fafa…`
      did not install the `tokenizers` Python package. The replacement image
      pins and imports both Tokenizers and Transformers at build time.
- [x] Replacement run used commit
      `da70caf81eec7786f08cf7b4e6d5dd3284eaa475` / image
      `sha256:a171ade18c4909560b91f1ed58755972cf87de2d7a8a2b5440870da6339b8583`,
      reached Batch `SUCCEEDED`, and exited 0.
- [x] CloudWatch prints `NVIDIA A10G tensor([1.], device='cuda:0')`.
- [x] Corpus resolution says:
  `pretrain/formal-proof-premises-500m/v3: 6 shards, dtype uint32`,
  tokenizer `tokenizer/qwen25-vendored/v1`.
- [x] The printed config includes:
  - Qwen vocab 151,936, 24 layers, tied embeddings, `backend='flash_2'`;
  - six train paths only; no validation paths;
  - sequence length 16,384 and `generate_doc_lengths=True`;
  - global batch 262,144, rank microbatch 16,384, seed 42;
  - arm `dense`, separator `[10952,15513,969]`, fixed divisor 262,144;
  - LR 2e-5, betas `(0.9,0.95)`, warmup 2,400, 23,166 steps;
  - checkpointer interval 2,000, ephemeral interval `None`,
    `max_checkpoints=None`.
- [x] No model training or checkpoint occurred: the lineage checkpoint survey
      reports zero objects / zero bytes and the run output prefix is empty. A warning about
      optional torchao/CUDA extensions is acceptable; an `edullm-stage:` refusal,
      missing dataset/tokenizer, wrong dtype, or nonzero exit is not.
- [ ] After committing the source-identity lint fixes, repeat this dry run on the
      new final built SHA. The run above proves the functional path, but section 4
      must use the same final image identity as the preflight.

Where to inspect this check without opening the AWS Batch job:

- GitHub Actions → **Look at a run, or stop it** gives status, status reason,
  exit code, and the CloudWatch stream name. Leave `run_id` blank for the latest
  run or paste the id returned by submission.
- Because the command is `CUDA assertion && published-data dry-run`, Batch
  `SUCCEEDED` with exit code 0 proves both halves passed. Either failure makes
  the whole command nonzero.
- The exact config values were already verified by both local S3 dry runs and
  integration tests, so opening CloudWatch is optional.
- If detailed stdout is wanted, use the reported CloudWatch stream (console or
  `aws logs get-log-events`). **Look at a run** does not display log contents.
- Do not expect W&B from `--dry-run`; it exits before trainer/callback startup.
  W&B becomes the primary surface for the 100-step A100/H100 smoke.

### 4. Real 8-rank A100/H100 end-to-end smoke

- [x] Pre-fused-loss A100 smoke
      `run_019fd251-f5f5-70df-8532-9bcf0af24521` reached the distributed
      forward dry-run and then OOMed on all eight 40-GiB A100s while allocating
      the 9.27-GiB fp32 logits tensor. The later `gather_object` exception was
      async step-0 checkpoint teardown noise, not the root cause.
- [ ] Rebuild the image with the P3-only Liger fused-linear loss and repeat this
      smoke on `gpu-8xa100`. If the final jobs use H100, repeat on
      `gpu-8xh100` to measure its actual throughput.

- [ ] Fill the form:

  | field | value |
  |---|---|
  | `repository` | `OLMo-core` |
  | `commit_sha` | same full built SHA |
  | `workload_profile` | `olmo-core-train` (checkpoint/retry contract; machine size still comes from `compute_profile`) |
  | `compute_profile` | `gpu-8xa100` to prove 40-GiB support; `gpu-8xh100` for the recommended faster final shape |
  | `team` | `memory-split` |
  | `experiment` | `p3-math-split-smoke` |
  | Data / `dataset_release` | `None` / `none` |
  | `wandb_project` | your W&B project |
  | advanced fields | leave blank/default |
  | `command` | paste the command below |

- [ ] Command field (dense arm, 100 steps):

  ```bash
  bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR" --runtime-smoke'
  ```

- [ ] Confirm:
  - all eight ranks start;
  - pretrained Qwen weights load strictly;
  - FlashAttention2 accepts `cu_doc_lens`;
  - `lm_head.loss_implementation == 'fused_linear'`;
  - FSDP preserves tied embeddings;
  - the log says global divisor 262,144 / 8 DP ranks = 32,768 rank-local tokens;
  - A100 peak allocated memory stays below 40 GiB with useful headroom;
  - dense supervised-token fraction is near 1 minus padding;
  - split supervised-token fraction is lower and stable;
  - a checkpoint writes to `$EDULLM_CHECKPOINT_DIR`;
  - W&B receives throughput, MFU, loss, and GPU-memory metrics.
- [ ] Extrapolate full runtime from measured tokens/sec. Do not rely on FLOP estimates
      after this point. SpeedMonitor is added automatically and skips the first
      unusually slow step; use the stabilized latter part of the 100-step run,
      not startup/model-download time.

This is the remaining **runtime** correctness surface that cannot be exercised
locally: 8-rank FSDP with tied embeddings, post-init full-state loading,
FlashAttention2 varlen plus Liger fused-linear kernels on A100/H100, async S3
checkpoint write/resume, and W&B under the workload role.

The quoted `gpu-4xa10g` / `gpu-4xl40s` advice applies to the stock dense-only
script at its 2,048-token default. It is not this run: our arm configs pin
16,384, the A10G is dry-run only, and the supported smoke/final shapes are
8xA100 and 8xH100.

### 5. Remaining scientific correctness before final jobs

- [x] Corrected fixed-loss normalization for data parallelism. The saved config
      keeps the arm-shared 262,144-token global control; the train module now
      divides by the actual DP world size before forward, giving 32,768 nominal
      tokens/rank at 8-way DP before FSDP averages gradients. Single-rank and
      8-rank divisor regressions pass.
- [x] Implemented the Metamath renderer/verifier repair with TDD:
  - only decoded/used theorem-local `$e` assumptions enter a `Local assumptions:`
    category before the unchanged `---` separator;
  - assumptions remain visible to both arms, split-masked and dense-supervised;
  - local `$e` pushes and `(reuse)` stack bookkeeping are omitted from targets;
  - source expansion still replays the original reuse operations and final stack;
  - semantic verification seeds local assumptions and reuses matching earlier
    derived expressions.
  This affects 43,376 train / 898 eval examples and removes 294,796 reuse lines.
- [x] Closed the deep-verifier blind spot for the repaired schema:
      Metamath rows must declare `local_assumptions`; mask reconstruction includes
      them; target labels must be supplied global facts; local assumptions and
      `(reuse)` are forbidden in target text.
- [x] Rebuilt, audited, retokenized, and published the repaired Metamath split
      as part of v3. The 16k eligibility policy is applied before held-out
      selection, so the final tokenization drops zero train/eval rows.
- [x] Recovered and pinned the exact Metamath source snapshot used by v2:
      `metamath/set.mm@82830c78861b96e906d9868c30c35dbd98be5db5`.
      Its three file hashes reproduce 1,151/1,151 eval targets byte-for-byte;
      builders now refuse drift and deterministic evaluation checks the manifest.
- [x] Made evaluation family-aware and context-safe. It discovers all six raw
      eval shards and their four shared family manifests, reports chunked
      target-token NLL plus teacher-forced next-token match for every family,
      retains normalized exact whole-output match as a diagnostic, and adds
      versioned tri-state Metamath generated-proof validity under
      `p3-metamath-tristate-v1` when all three source databases pass the pinned
      manifest hashes. It applies the training-identical
      `text + EOS <= 16,384`
      gate against the final v3-eligible cohort; the published v3 val
      tokenization drops zero rows.
      NLL materializes logits for 256 targets at a time; every retained row keeps
      its complete fact block, goal, and preceding gold proof in context.
      Whole-proof generation is explicitly secondary and defaults to an
      8,192-token ceiling, with per-family budget coverage reported. All three
      conditions (`facts_present`, `facts_absent`, `facts_corrupted`) use each
      family's complete context-eligible cohort; the six source cohorts total
      4,191 rows and use identical IDs within family across conditions.
      Metamath validity covers present/absent with unknown separate from invalid;
      corrupted is explicitly unsupported. Evaluate only final step 23166;
      do not run intermediate checkpoint evaluations. Facts-shuffled is not
      evaluated.
- [x] Updated `compare_arms.py` for family-keyed paired IDs, per-family NLL,
      paired per-example token match, generic exact-match outcomes,
      tri-state Metamath-valid outcomes, decided-pair denominators, and paired
      bootstrap intervals. Schema `p3-comparison-v4` validates per-example
      status counts, source availability/provenance, evaluator controls/cohorts,
      and rejects old boolean validity. It
      requires the two checkpoints' saved `config.json` files to match outside
      the arm/run-output identity; it no longer expects obsolete local sidecars.

### 6. Historical v2 audit findings (superseded by v3)

The findings below describe the scientifically stale v2 source paths. They are
retained as history, not as v3 launch blockers. The repaired pooled-MML,
direct-Mizar, ENIGMA, and Isabelle decisions and gates are tracked in
`P3_WORK_STATUS.md` and `P3_DECISION_LEDGER.md`.

- [ ] Repair the shared ATP held-out split. Alternate ENIGMA theorem IDs retain
      `#2`, so 19 training rows remain proofs of held facts; aliases expose exact
      held statements too. In total, 26/500 ATP held facts have exact training
      exposure, affecting 42 ENIGMA and 51 prf2 eval rows.
- [ ] Rebuild ATP traces without lossy parent extraction. Current targets cap
      parent lists at six and omit external/input parents: 286/465 ENIGMA and
      431/632 prf2 eval traces have confirmed absent references. They are token
      continuations, not verifiable proof objects.
- [ ] Fix Mizar theorem/proof block parsing before scientific use. The block
      regex can cross later theorem headers before finding `proof`, attaching the
      wrong proof body. The broad audit finds 1,955/46,616 raw and 31/990 eval;
      an independent literal-header scan confirms at least 1,938 raw and 29 eval,
      including the inspected `AFF_1:50` and canceled `JORDAN2C:71` cases.
- [ ] Rebuild thproofs against a matching MML snapshot with complete reference
      resolution. At least 55/76 eval rows are defective; 53 cite absent local,
      numbered, scheme, or definition premises and four include `canceled;`
      statements as facts.
- [ ] Rebuild Isabelle from the intended Magnushammer **raw human trajectory**
      source. Chat history confirms it replaced IsarStep only under a next-state
      content target; tactic-only prediction was an unapproved implementation
      substitution. Pin HF dataset revision
      `f947ccc827ccd236464e19cd4cc23dfda7fc5575` and raw-file commit
      `7d87bf8c7495327a25fde5b380415549938fc104`. The user selected
      `facts + state_before → tactic + state_after`; an independent read-only
      verification of trajectory alignment is required before implementation.
      Current defects remain: all 623 targets are one command; 73 theorem prefixes
      are clipped at 400 characters; 182 rows show evident omitted local
      context/definitions. The broad normalized audit finds ten held facts/12 eval
      rows with statement aliases; an exact-string scan confirms at least nine/11.
- [ ] Correct evaluator/report wording and denominators when these families are
      repaired: own-theorem-only rows touch but do not cite a held fact; exact-all
      includes generation-budget-ineligible proofs; non-Metamath exact output is
      reproduction rather than proof validity; NLL/token match omit EOS/stopping.

## Final two jobs (one seed)

**DATA READY, BUT RUN THE FUSED-LOSS 8-RANK SMOKE FIRST.** v3 is independently
validated, promoted, and reader-resolvable. The remaining gate before the full
jobs is the exact built image passing section 4's 100-step A100 smoke (and an
H100 throughput smoke if H100 is the final shape).

**Chosen plan:** two separate `gpu-8xh100` nodes, one full eight-rank job per
arm, submitted concurrently. This requires no launcher waiver or custom 4+4
orchestration; each arm gets its own run id, W&B run, checkpoint prefix, retry,
and failure status. Estimated wall time is 4–6 hours; total allocation cost is
approximately $440–$660.

- [ ] Submit the dense form:

  | field | value |
  |---|---|
  | `repository` | `OLMo-core` |
  | `commit_sha` | full smoke-tested SHA |
  | `workload_profile` | `olmo-core-train` |
  | `compute_profile` | `gpu-8xh100` |
  | `team` | `memory-split` |
  | `experiment` | `p3-math-split-seed42` |
  | Data / `dataset_release` | `None` / `none` |
  | `wandb_project` | your W&B project |
  | advanced fields | leave blank/default |
  | `command` | dense command below |

- [ ] Dense command field:

  ```bash
  bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm dense --config src/scripts/train/p3_math_split/configs/dense.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR"'
  ```

- [ ] Submit the split form with every field identical except the command:

  ```bash
  bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm split --config src/scripts/train/p3_math_split/configs/split.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR"'
  ```
- [ ] For the requested split normalization ablation, replace the split command
      above with this one (the final flag is the only launch difference):

  ```bash
  bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm split --config src/scripts/train/p3_math_split/configs/split.yaml --dataset-id pretrain/formal-proof-premises-500m --dataset-version v3 --dataset-tokenizer tokenizer/qwen25-vendored/v1 --save-folder "$EDULLM_CHECKPOINT_DIR" --split-supervised-token-divisor'
  ```
- [ ] Recompute and then check the first 100 steps before trusting the run:
  - finite losses;
  - `train/supervised token fraction` matches exact values recomputed from the
    new token bytes (do not copy v2's 99.915% / 83.803% values);
  - ignore OLMo-core's generic `train/masked labels (%)` for this comparison:
    it is computed before the runtime-derived mask. The experiment-specific
    supervised-token metric is recorded after masking and is the valid diagnostic;
  - expected MFU/tokens per second;
  - no separator-missing metrics;
  - checkpoints appearing at the expected interval.
- [ ] To stop a run, use **Look at a run, or stop it** with the run id. Do not
      cancel the submission workflow; that leaves the Batch job running.

