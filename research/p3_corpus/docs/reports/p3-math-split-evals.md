# P3 math split evaluation handoff

This document is the execution handoff for evaluating the dense and split
Qwen2.5-0.5B checkpoints. It contains only the files, evaluation design,
commands, and outputs needed by the evaluation team.

For the current sharded AWS/vLLM run, the authoritative operational commands
are in `olmo-eval-full/Vishnu-Evals/HANDOFF_README.md`. The single-process
commands below remain the HF fallback/local path. In particular, do not use
`$OLMO_ROOT/.venv` for vLLM on the L4 DLAMI; run
`bootstrap_vllm_env.sh` and use `/mnt/work/p3-vllm-venv/bin/python`.

## Evaluation design

Evaluate the same checkpoint step for both arms on all six formal-proof
families:

- ENIGMA: 263 source eval rows
- Isabelle: 590
- Metamath: 494
- Mizar: 2,485
- prf2: 313
- thproofs: 46

The six family shards contain 4,191 source eval rows in total. `facts_present`
uses each family's complete post-context-gate cohort. `facts_absent` and
`facts_corrupted` each use a separate deterministic, ceiling-rounded 10%
sample per family: 422 rows per diagnostic condition at seed `20260801`.
Dense and split must use identical row IDs within each family and condition;
the two diagnostic conditions need not use the same IDs.

- `facts_present`: correct global premise block
- `facts_absent`: no global premise block
- `facts_corrupted`: premise names retained but statements replaced

Local assumptions and local ATP inputs remain visible in every condition. There is
no `facts_shuffled` condition.

For every family and condition, report:

- teacher-forced target-token negative log-likelihood;
- teacher-forced next-token accuracy;
- example-macro and token-micro versions of both metrics;
- greedy whole-proof normalized exact match as a secondary diagnostic;
- whole-proof generation-budget coverage.

Use:

- context length: 16,384;
- maximum generated tokens: 8,192;
- NLL chunk size: 256;
- greedy decoding;
- evaluator seed: `20260801`;
- Metamath tri-state generated-proof validity for `facts_present` and
  `facts_absent` under `p3-metamath-tristate-v1`;
- no Metamath validity rate for `facts_corrupted`;
- no cross-family aggregate.

The dense-versus-split comparison is paired by row ID within each
family/condition and uses 10,000 paired bootstrap samples with seed
`20260801`.

## Required local files

Use these two repository checkouts:

```text
/home/vs/AlphaAI/memorysplit-requery-exact
/home/vs/AlphaAI/eduLLM/OLMo-core
```

The evaluator corpus must be the root-level `corpus-v3/` directory. It must
contain exactly:

```text
corpus-v3/
├── shards/
│   ├── enigma.jsonl
│   ├── isabelle.jsonl
│   ├── metamath.jsonl
│   ├── mizar.jsonl
│   ├── prf2.jsonl
│   └── thproofs.jsonl
├── eval/
│   ├── enigma.jsonl
│   ├── isabelle.jsonl
│   ├── metamath.jsonl
│   ├── mizar.jsonl
│   ├── prf2.jsonl
│   └── thproofs.jsonl
├── heldout/
│   ├── atp.json
│   ├── isabelle.json
│   ├── metamath.json
│   └── mizar.json
├── metamath_sources.json
├── evaluator_manifest.json
└── README.md
```

Do not use the legacy `corpus/` directory.

The raw evaluator JSONLs are not published under the S3 packed-token v3
dataset. If evaluation runs on another machine, copy the complete local
`corpus-v3/` directory to that machine. The hardlinks do not need to remain
hardlinks after transfer.

## Required S3 files

### Dense checkpoint

Checkpoint root:

```text
s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints
```

For the selected `STEP`, the exporter needs:

```text
step${STEP}/.metadata.json
step${STEP}/config.json
step${STEP}/model_and_optim/.metadata
step${STEP}/model_and_optim/*.distcp
step${STEP}/train/rank0.pt
```

Every `*.distcp` object under `model_and_optim/` is required. Do not select or
copy only one rank's objects.

### Split checkpoint

Checkpoint root:

```text
s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints
```

The required relative files are identical to the dense list above.

Evaluate only the completed final checkpoint `step23166` under both roots. Do
not run intermediate checkpoint evaluations.

### Tokenizer

Checkpoint export resolves the tokenizer from:

```text
s3://edullm-data/tokenizer/qwen25-vendored/v1/
```

The required objects are:

```text
_VALIDATED.json
dataset.json
tokenizer/manifest.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

The exporter downloads these automatically. Do not substitute a Hugging Face
tokenizer.

Checkpoint export also needs network access, or an existing local cache, for
only this Hugging Face file:

```text
Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987/config.json
```

It does not download Qwen model weights from Hugging Face.

### S3 files that are not needed

Do not download
`s3://edullm-data/pretrain/formal-proof-premises-500m/v3/tokens/*.u32le.bin`.
Those are packed training/validation token shards and cannot replace the raw
JSONL evaluator corpus.

## Exact commands

The commands below evaluate only the completed final checkpoint `step23166`
shared by both arms. Do not substitute intermediate checkpoints.

### 1. Set paths

```bash
set -euo pipefail

export P3_ROOT=/home/vs/AlphaAI/memorysplit-requery-exact
export OLMO_ROOT=/home/vs/AlphaAI/eduLLM/OLMo-core
export PYTHON="$OLMO_ROOT/.venv/bin/python"
export PYTHONPATH="$OLMO_ROOT/src"

export STEP=23166
export DENSE_RUN=s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints
export SPLIT_RUN=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints
export EVAL_WORK="$P3_ROOT/eval-work/step${STEP}"
export MM_DIR="$P3_ROOT/.p3-work/sources/p3-audit-mm"

mkdir -p "$EVAL_WORK"/{configs,hf,results,staging}
```

Use a new `EVAL_WORK` directory for every checkpoint step. Do not point any
output command into either S3 checkpoint root.

`MM_DIR` must contain `set.mm`, `iset.mm`, and `nf.mm` from
`metamath/set.mm@82830c78861b96e906d9868c30c35dbd98be5db5`.
`run_eval.py` verifies all three hashes against
`corpus-v3/metamath_sources.json` before reporting validity.

### 2. Check the evaluator corpus

```bash
cd "$P3_ROOT"
"$PYTHON" scripts/assemble_v3_evaluator_root.py \
  --out "$P3_ROOT/corpus-v3" \
  --check-only
```

The expected totals are 181,652 train rows and 4,191 source eval rows.

### 3. Export both checkpoints to Hugging Face format

```bash
cd "$OLMO_ROOT"

"$PYTHON" src/scripts/train/p3_math_split/evals/export_checkpoint.py \
  --run "$DENSE_RUN" \
  --step "$STEP" \
  --out "$EVAL_WORK/hf/dense" \
  --work-dir "$EVAL_WORK/staging/dense"

"$PYTHON" src/scripts/train/p3_math_split/evals/export_checkpoint.py \
  --run "$SPLIT_RUN" \
  --step "$STEP" \
  --out "$EVAL_WORK/hf/split" \
  --work-dir "$EVAL_WORK/staging/split"
```

The exporter reads the distributed checkpoint directly from S3. Do not run
`aws s3 cp --recursive` on the checkpoint roots first.

Copy the two saved training configs for the paired comparison:

```bash
aws s3 cp \
  "$DENSE_RUN/step${STEP}/config.json" \
  "$EVAL_WORK/configs/dense.json"

aws s3 cp \
  "$SPLIT_RUN/step${STEP}/config.json" \
  "$EVAL_WORK/configs/split.json"
```

### 4. Run one-row smoke evaluations

```bash
cd "$OLMO_ROOT"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/dense" \
  --arm dense \
  --corpus "$P3_ROOT/corpus-v3" \
  --mm-dir "$MM_DIR" \
  --families enigma isabelle metamath mizar prf2 thproofs \
  --conditions facts_present \
  --limit 1 \
  --batch-size 1 \
  --context-length 16384 \
  --max-new-tokens 32 \
  --nll-chunk-size 256 \
  --seed 20260801 \
  --out "$EVAL_WORK/results/dense-smoke.json"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/split" \
  --arm split \
  --corpus "$P3_ROOT/corpus-v3" \
  --mm-dir "$MM_DIR" \
  --families enigma isabelle metamath mizar prf2 thproofs \
  --conditions facts_present \
  --limit 1 \
  --batch-size 1 \
  --context-length 16384 \
  --max-new-tokens 32 \
  --nll-chunk-size 256 \
  --seed 20260801 \
  --out "$EVAL_WORK/results/split-smoke.json"
```

Both commands must finish and produce JSON before starting the full runs.

### 5. Run the full dense evaluation

```bash
cd "$OLMO_ROOT"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/dense" \
  --arm dense \
  --corpus "$P3_ROOT/corpus-v3" \
  --mm-dir "$MM_DIR" \
  --families enigma isabelle metamath mizar prf2 thproofs \
  --conditions facts_present facts_absent facts_corrupted \
  --batch-size 8 \
  --context-length 16384 \
  --max-new-tokens 8192 \
  --nll-chunk-size 256 \
  --seed 20260801 \
  --out "$EVAL_WORK/results/dense.json"
```

### 6. Run the full split evaluation

Run this after the dense command finishes if both use the same GPU:

```bash
cd "$OLMO_ROOT"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/split" \
  --arm split \
  --corpus "$P3_ROOT/corpus-v3" \
  --mm-dir "$MM_DIR" \
  --families enigma isabelle metamath mizar prf2 thproofs \
  --conditions facts_present facts_absent facts_corrupted \
  --batch-size 8 \
  --context-length 16384 \
  --max-new-tokens 8192 \
  --nll-chunk-size 256 \
  --seed 20260801 \
  --out "$EVAL_WORK/results/split.json"
```

### 7. Compare dense and split

```bash
cd "$OLMO_ROOT"

"$PYTHON" src/scripts/train/p3_math_split/evals/compare_arms.py \
  --dense "$EVAL_WORK/results/dense.json" \
  --split "$EVAL_WORK/results/split.json" \
  --dense-config "$EVAL_WORK/configs/dense.json" \
  --split-config "$EVAL_WORK/configs/split.json" \
  --n-boot 10000 \
  --seed 20260801 \
  --out "$EVAL_WORK/results/comparison.json"
```

## Required outputs

Preserve these files together:

```text
eval-work/step${STEP}/
├── configs/
│   ├── dense.json
│   └── split.json
├── hf/
│   ├── dense/
│   └── split/
└── results/
    ├── dense-smoke.json
    ├── split-smoke.json
    ├── dense.json
    ├── split.json
    └── comparison.json
```

The headline result is the per-family paired dense-versus-split difference
under `facts_present`. `facts_absent` and `facts_corrupted` are mechanism
diagnostics and should not replace the headline result.
