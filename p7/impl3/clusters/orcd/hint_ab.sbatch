#!/bin/bash
# ============================================================================
# A/B: does the POC's boxed-answer hint explain the math-forgetting discrepancy?
#
# The POC measured vanilla SFT collapsing on math (base 20.0% -> c923 2.9%, GSM8K 60 -> 6.7).
# Our sweep measures the same items as dead flat (24.4% -> 24.3%). Everything comparable between
# the two setups is byte-identical -- data, LoRA config, optimizer, schedule, loss masking,
# collate, prompt items, generation length, and the answer extractor -- with exactly one
# exception: the POC appended a boxed-answer hint to every math prompt and we send the bare
# question. This job measures that one variable.
#
# Two adapters x two prompt formats, base regenerated in each pass as its own control:
#   impl2       = checkpoint-923, the POC-lineage vanilla SFT
#   impl2-rerun = the vanilla SFT we trained this week (tests whether the MODELS differ instead)
#
# Reading the result:
#   bare ~22% and hinted ~3%   -> the hint is the whole discrepancy; our probe simply got easier.
#   both ~3%                   -> the models differ, not the probe.
#   both ~22%                  -> the POC's 2.9% came from something else in that pipeline.
#
#     sbatch clusters/orcd/hint_ab.sbatch
# ============================================================================
#SBATCH -p mit_preemptable       # Standing choice: preemptable schedules far sooner.
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1   # 45 prompts x 4 passes; one H200 is plenty
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J p7-hint-ab
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"
RERUN="${RERUN:-out/impl2-rerun}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
cd "$PROJECT"

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT"

die() { echo "FATAL: $*" >&2; echo "Fix: run 'bash clusters/orcd/setup_orcd_env.sh' on a login node once, then resubmit." >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH (Miniforge not installed)."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
python -c "import torch, transformers, peft" 2>/dev/null || die "env '$ENV_NAME' missing torch/transformers/peft."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not visible to torch."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "preflight OK: $(python -c 'import torch;print("torch",torch.__version__)')"

# The --boxed_hint flag is the entire point of this job; an old copy on ORCD would silently run
# the bare condition twice and produce a null result that looks like a real answer.
grep -q "boxed_hint" eval/generate_eval.py \
    || die "eval/generate_eval.py has no --boxed_hint -> stale code on ORCD. rsync the project up and resubmit."
[ -f eval/math_eval/score_results.py ] || die "eval/math_eval/score_results.py missing -> rsync the project up."

final_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

[ -e "$VANILLA_SFT/adapter_config.json" ] || die "vanilla adapter '$VANILLA_SFT' not found."
RERUN_STEP=$(final_ckpt "$RERUN")
if [ -n "$RERUN_STEP" ]; then
  RERUN_CKPT="$RERUN/checkpoint-$RERUN_STEP"
  echo "adapters: impl2=$VANILLA_SFT | impl2-rerun=$RERUN_CKPT"
else
  RERUN_CKPT=""
  echo "[warn] $RERUN has no checkpoints -> only the POC-lineage adapter will be A/B'd"
fi

PROMPTS=eval/math_eval/math_logic_prompts.jsonl
run_pair() {  # <tag> <adapter> <extra flags...>
  local tag="$1" adapter="$2"; shift 2
  echo "--- generating: $tag ---"
  python eval/generate_eval.py --prompts "$PROMPTS" --adapter "$adapter" \
      --out "eval/math_eval/results_ab_${tag}.jsonl" "$@" \
      || echo "[warn] generation failed for $tag"
}

run_pair impl2_bare       "$VANILLA_SFT"
run_pair impl2_hint       "$VANILLA_SFT" --boxed_hint
if [ -n "$RERUN_CKPT" ]; then
  run_pair rerun_bare     "$RERUN_CKPT"
  run_pair rerun_hint     "$RERUN_CKPT" --boxed_hint
fi

echo
echo "================ A/B RESULT ================"
( cd eval/math_eval && python score_results.py results_ab_impl2_bare.jsonl results_ab_impl2_hint.jsonl \
                                               results_ab_rerun_bare.jsonl results_ab_rerun_hint.jsonl )

echo
echo "--- sample SFT generations under each format (is the tutor deflecting?) ---"
python - <<'PY'
import json, os
for tag in ("impl2_bare", "impl2_hint", "rerun_bare", "rerun_hint"):
    p = f"eval/math_eval/results_ab_{tag}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    g = [r for r in rows if r.get("source") == "GSM8K"]
    if not g:
        continue
    print("=" * 90)
    print(f"### {tag}")
    print("PROMPT:", g[0]["prompt"][:150].replace("\n", " "))
    print("SFT   :", (g[0]["outputs"]["sft"] or "")[:500].replace("\n", " "))
PY

echo "== HINT A/B DONE =="
