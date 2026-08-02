#!/bin/bash
# ============================================================================
# Fold every LoRA checkpoint into standalone HF weights for the eval team's tutor_cat pipeline.
#
# CPU-ONLY BY DESIGN. Merging is a bf16 matmul per adapted module (dW = (alpha/r) B@A) plus a
# save; no forward pass runs, so a GPU would sit idle behind the disk write. This also lets the
# merge proceed while the tutorcat env is still installing on the login node.
#
# Runs in p7post, NOT tutorcat: peft is what does the merge and it lives in the training env.
#
#     sbatch clusters/orcd/merge_tutorcat.sbatch      # from post-training/
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J p7-merge
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
OUT_DIR="${OUT_DIR:-$HOME/tutorcat_models}"
MANIFEST="${MANIFEST:-$HOME/tutorcat_models.yaml}"
POC_CKPT="${POC_CKPT:-checkpoint-923}"
cd "$PROJECT"

die() { echo "FATAL: $*" >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' missing."
python -c "import peft, transformers, torch" 2>/dev/null || die "$ENV_NAME lacks peft/transformers/torch."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ~2.4 GB per merged 1B model at bf16; 18 models is ~43 GB. Fail early rather than half way
# through, which would leave a truncated safetensors that vLLM reports as a corrupt checkpoint.
AVAIL_GB=$(( $(quota -s 2>/dev/null | awk '/home/{gsub(/[^0-9]/,"",$3); print $3; exit}') - $(quota -s 2>/dev/null | awk '/home/{gsub(/[^0-9]/,"",$2); print $2; exit}') ))
echo "quota headroom (GB, approximate): ${AVAIL_GB:-unknown}"

EXTRA=()
[ -d "$POC_CKPT" ] && EXTRA=(--extra "poc-c923=$POC_CKPT") \
    || echo "[warn] $POC_CKPT not found — POC lineage will be absent from the manifest."

echo "== merging =="
python eval/merge_for_tutorcat.py \
    --runs "${RUNS:-out/*}" \
    --out_dir "$OUT_DIR" \
    --manifest "$MANIFEST" \
    --require_epoch 0.99 \
    ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?

echo
echo "== result =="
du -sh "$OUT_DIR" 2>/dev/null
ls "$OUT_DIR" 2>/dev/null | wc -l | xargs echo "merged model dirs:"
echo "manifest: $MANIFEST"
exit "$STATUS"
