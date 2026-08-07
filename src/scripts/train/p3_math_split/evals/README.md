# P3 math split evaluation

Compare dense and split Qwen2.5-0.5B checkpoints on the same formal-proof
evaluator rows. Dense supervises supplied premise tokens; split can attend to
those tokens but masks their loss. The evaluation asks whether this training
difference changes teacher-forced likelihood, next-token accuracy, and greedy
whole-proof exact match on held-out proofs under controlled premise conditions.

## Entrypoints

All evaluator commands live in this directory:

- `export_checkpoint.py` — download a distributed checkpoint from S3 and write
  Hugging Face weights for inference.
- `run_eval.py` — score one arm on the local `corpus-v3/` evaluator root.
- `compare_arms.py` — paired dense-versus-split comparison from two result JSON
  files.
- `bootstrap_vllm_env.sh` — create the isolated Python 3.12/CUDA 12 environment
  required by the vLLM wheel on the L4 DLAMI.
- `preflight_vllm.py` — import the native vLLM extension and reject incompatible
  Python, package, CUDA, or GPU versions before a fleet launch.
- `prepare_base_model.sh` — materialize the untrained `base` control model dir
  (pinned `Qwen/Qwen2.5-0.5B` weights + the vendored qwen2.5 tokenizer).

## Base control arm

The `base` arm scores the untrained `Qwen/Qwen2.5-0.5B` checkpoint the two arms
were initialized from, to test whether the base model's own pretrained knowledge
already explains a trained arm's `facts_present` behavior. It reuses the same
corpus, deterministic cohorts, conditions, and seed, so its rows line up with
dense/split; only the served weights differ. It carries no training provenance
and is compared standalone (never through `compare_arms.py`, which is dense vs
split only).

```bash
# after bootstrap_vllm_env.sh; $PYTHON is /mnt/work/venv/bin/python
P3_PYTHON="$PYTHON" bash src/scripts/train/p3_math_split/evals/prepare_base_model.sh \
  "$EVAL_WORK/hf/base"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/base" --arm base \
  --base-model-id Qwen/Qwen2.5-0.5B \
  --base-model-revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --corpus "$P3_ROOT/corpus-v3" --mm-dir "$MM_DIR" \
  --conditions facts_present facts_absent facts_corrupted \
  --generation-backend vllm --vllm-gpu-memory-utilization 0.55 \
  --vllm-max-model-len 16384 \
  --context-length 16384 --max-new-tokens 8192 --nll-chunk-size 256 \
  --seed 20260801 --out "$EVAL_WORK/results/base.json"
```

The corpus assembler is **not** here. It is local-only in the
`memorysplit-requery-exact` checkout because it reads absolute `.p3-work` sealed
paths, not S3. The packed v3 S3 release cannot reconstruct the raw evaluator
JSONLs; copy the complete portable `corpus-v3/` directory to any remote machine.

## Required corpus

Point every `run_eval.py` invocation at the root-level `corpus-v3/` directory
(with `shards/`, `eval/`, `heldout/`, `evaluator_manifest.json`, and
`README.md`). Do not use the legacy `corpus/` tree or packed token shards from
S3.

## Conditions and checkpoint

Use only the completed final checkpoint step **23166** for both arms. Do not
run intermediate checkpoint evaluations.

The six family shards contain 4,191 source eval rows in total.
`facts_present` uses every post-context-gate row. `facts_absent` and
`facts_corrupted` each use a separate deterministic, ceiling-rounded 10%
sample per family (422 rows each at seed `20260801`). Dense and split use
identical IDs within each condition; diagnostic conditions need not share IDs.
There is no `facts_shuffled`.

Result schema is `p3-eval-v9`; paired comparison schema is
`p3-comparison-v5`.

Metamath reports versioned tri-state generated-proof validity for
`facts_present` and `facts_absent` when `--mm-dir` supplies hash-verified
`set.mm`, `iset.mm`, and `nf.mm`. `facts_corrupted` has no validity rate.

- `facts_present` — correct global premise block
- `facts_absent` — no global premise block
- `facts_corrupted` — premise names retained but statements replaced

## Execution order

Run these steps in order from the handoff document
`memorysplit-requery-exact/p3-math-split-evals.md`:

1. **Validate corpus** — `assemble_v3_evaluator_root.py --check-only` on
   `corpus-v3/`.
2. **Prepare and preflight vLLM** — run `bootstrap_vllm_env.sh` on one L4. Do
   not use the DLAMI's Python 3.13/PyTorch cu130 environment.
3. **Export both checkpoints** — dense and split to Hugging Face format with
   `export_checkpoint.py`.
4. **Smoke evals** — one-row vLLM `run_eval.py` runs for dense and split on the
   same L4 and with the same 16,384-token engine settings as production.
5. **Full evals** — all six families and three conditions with `run_eval.py`.
6. **Compare** — `compare_arms.py` on the dense and split result JSON files.

Example invocations (after setting paths as in the handoff):

```bash
cd "$OLMO_ROOT"

P3_WORK_ROOT=/mnt/work \
  bash src/scripts/train/p3_math_split/evals/bootstrap_vllm_env.sh
PYTHON=/mnt/work/p3-vllm-venv/bin/python

"$PYTHON" src/scripts/train/p3_math_split/evals/export_checkpoint.py \
  --run "$DENSE_RUN" --step 23166 --out "$EVAL_WORK/hf/dense" \
  --work-dir "$EVAL_WORK/staging/dense"

"$PYTHON" src/scripts/train/p3_math_split/evals/run_eval.py \
  --model "$EVAL_WORK/hf/dense" --arm dense --corpus "$P3_ROOT/corpus-v3" \
  --mm-dir "$MM_DIR" \
  --families enigma isabelle metamath mizar prf2 thproofs \
  --conditions facts_present facts_absent facts_corrupted \
  --generation-backend vllm --vllm-gpu-memory-utilization 0.55 \
  --vllm-max-model-len 16384 \
  --batch-size 8 --context-length 16384 --max-new-tokens 8192 \
  --nll-chunk-size 256 --seed 20260801 --out "$EVAL_WORK/results/dense.json"

"$PYTHON" src/scripts/train/p3_math_split/evals/compare_arms.py \
  --dense "$EVAL_WORK/results/dense.json" \
  --split "$EVAL_WORK/results/split.json" \
  --dense-config "$EVAL_WORK/configs/dense.json" \
  --split-config "$EVAL_WORK/configs/split.json" \
  --n-boot 10000 --seed 20260801 --out "$EVAL_WORK/results/comparison.json"
```

See `p3-math-split-evals.md` for full path setup, S3 roots, smoke commands,
and required output layout.
