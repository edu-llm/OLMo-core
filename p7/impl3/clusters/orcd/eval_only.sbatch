#!/bin/bash
# ============================================================================
# EVAL ONLY — regenerate eval outputs from checkpoints that are ALREADY trained.
#
# Stage 3 of impl3_h200.sbatch, standalone: no precompute, no training. Use it whenever the
# training checkpoints are fine but the eval stage failed or needs redoing (e.g. the 2026-07-30
# tokenizer bug, or new prompt sets), so you never re-pay for the sweep.
#
#     sbatch -p mit_preemptable clusters/orcd/eval_only.sbatch
#
# Only re-do some runs (shell glob over out/, default all):
#     PATTERN='impl3-b-*'      sbatch --export=ALL,PATTERN='impl3-b-*' clusters/orcd/eval_only.sbatch
#     PATTERN='impl3-a-T0.5'   ... etc
#
# Grading/plotting stay off-GPU afterwards: grade_math_logic.py, grade_ifeval.py,
# plot_kl_forgetting.py, wandb_eval.py, and the frontier judge for pedagogy.
# ============================================================================
#SBATCH -p mit_preemptable       # Standing choice: preemptable schedules far sooner.
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1   # generation only; one H200 is plenty and schedules fastest
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J p7-eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"   # vanilla Impl-2 baseline point
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
PATTERN="${PATTERN:-impl3-*}"                  # which out/ run dirs to evaluate
cd "$PROJECT"

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT | PATTERN=$PATTERN"

# ---- preflight (same contract as the training job: fail loudly, never fake a "Completed") --
die() { echo "FATAL: $*" >&2; echo "Fix: run 'bash clusters/orcd/setup_orcd_env.sh' on a login node once, then resubmit." >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH (Miniforge not installed)."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
python -c "import torch, transformers, peft" 2>/dev/null || die "env '$ENV_NAME' missing torch/transformers/peft."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not visible to torch."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "preflight OK: $(python -c 'import torch;print("torch",torch.__version__)') | HF_HOME=$HF_HOME"

# The eval tokenizer must resolve even though LoRA checkpoint dirs ship no tokenizer files.
python -c "from common.modeling import load_tokenizer_for" 2>/dev/null \
    || die "common/modeling.py has no load_tokenizer_for -> stale code on ORCD. rsync the project up and resubmit."

# ---- held-out pedagogy split (KL axis + pedagogy prompts) ---------------------------------
if [ ! -f data/socrateach_sft_val.jsonl ]; then
  python snapshot_hf_dataset.py || die "no data/socrateach_sft_val.jsonl and could not fetch it (prestage on the login node: python snapshot_hf_dataset.py)."
fi

# highest-numbered checkpoint in a run dir
final_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

CKPTS=""
for d in out/$PATTERN/; do
  [ -d "$d" ] || continue
  step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  CKPTS="$CKPTS $(basename "$d")=$d/checkpoint-$step"
done
echo "final checkpoints:$CKPTS"
[ -z "$CKPTS" ] && die "no checkpoints matched out/$PATTERN/ — nothing to evaluate."

# ---- (a) KL axis vs base, including the vanilla Impl-2 point ------------------------------
python run_kl_curve.py --base_model "$BASE_MODEL" \
    --ckpt $CKPTS impl2="$VANILLA_SFT" \
    --pedagogy_file data/socrateach_sft_val.jsonl --n_prompts 64 \
    --out out/kl_by_checkpoint.json || echo "[warn] KL curve failed"

# ---- (b) math retention + (c) IFEval — deterministic, no judge ---------------------------
FAILED=""
for d in out/$PATTERN/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d"); step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl \
      --adapter "$d/checkpoint-$step" --out eval/math_eval/results_$tag.jsonl \
      || { echo "[warn] math gen $tag failed"; FAILED="$FAILED math:$tag"; }
  python eval/generate_eval.py --prompts eval/general_eval/ifeval_prompts.jsonl \
      --adapter "$d/checkpoint-$step" --out eval/general_eval/results_$tag.jsonl \
      || { echo "[warn] ifeval gen $tag failed"; FAILED="$FAILED ifeval:$tag"; }
done

# Vanilla Impl-2 on the same probes. Its adapter lives outside out/, so the loop above misses it —
# without this the baseline has a KL and a pedagogy score but no retention numbers, and "Impl 3
# forgets less than vanilla SFT" can't be checked on the deterministic axes at all.
if [ -e "$VANILLA_SFT/adapter_config.json" ] || [ -e "$VANILLA_SFT/adapter_model.safetensors" ]; then
  python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl \
      --adapter "$VANILLA_SFT" --out eval/math_eval/results_impl2.jsonl \
      || { echo "[warn] math gen impl2 failed"; FAILED="$FAILED math:impl2"; }
  python eval/generate_eval.py --prompts eval/general_eval/ifeval_prompts.jsonl \
      --adapter "$VANILLA_SFT" --out eval/general_eval/results_impl2.jsonl \
      || { echo "[warn] ifeval gen impl2 failed"; FAILED="$FAILED ifeval:impl2"; }
else
  echo "[warn] vanilla adapter '$VANILLA_SFT' not found -> impl2 retention baseline skipped."
fi

# ---- (d) pedagogy generation (NEW-task quality; judged off-GPU afterwards) ----------------
PED_CANDS="base= impl2=$VANILLA_SFT"
for d in out/$PATTERN/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d"); step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  PED_CANDS="$PED_CANDS $tag=$d/checkpoint-$step"
done
echo "pedagogy candidates:$PED_CANDS"
python eval/gen_pedagogy.py --base_model "$BASE_MODEL" \
    --test_file data/socrateach_sft_val.jsonl --candidates $PED_CANDS \
    --n_dialogues 40 --out eval/llm_judge/test_results_instruct.jsonl \
    && ( cd eval/llm_judge && python build_batches.py test_results_instruct.jsonl ) \
    || { echo "[warn] pedagogy generation / judge-batch build failed"; FAILED="$FAILED pedagogy"; }

echo "== EVAL DONE =="
if [ -n "$FAILED" ]; then
  echo "FAILED points:$FAILED  (re-run just these with PATTERN=<run dir>)"
else
  echo "all eval points generated cleanly."
fi
echo "Next (off-GPU): grade_math_logic.py / grade_ifeval.py -> plot_kl_forgetting.py -> wandb_eval.py"
echo "                pedagogy: judge eval/llm_judge/judge_batch_*.json -> judge_out_*.json -> aggregate.py"
