#!/bin/bash
# ============================================================================
# SMOKE TEST — tiny 1-GPU / ~20-min end-to-end check before the real sweep.
#
# Exercises the whole Impl-3 path on a handful of examples: preflight -> model load ->
# tokenize -> per-token weighted-loss train loop -> checkpoint save. Its job is to catch
# dependency/API/code breakage in minutes instead of after a long queue + full sweep.
#
#     sbatch --partition mit_preemptable clusters/orcd/smoke.sbatch
#     tail -f logs/p7-smoke_<id>.out     # want: "SMOKE OK" at the end
#
# A short 1-GPU job backfills fast, so it usually starts almost immediately.
# ============================================================================
#SBATCH -p mit_preemptable       # Standing choice: preemptable schedules far sooner.
#SBATCH -G h200:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -J p7-smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs
PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
CFG="impl3_kl_reweighted_sft/config.yaml"
cd "$PROJECT"

die() { echo "SMOKE FAILED: $*" >&2; exit 1; }
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH (run setup_orcd_env.sh)."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "env '$ENV_NAME' missing."
python -c "import torch,transformers,peft; assert torch.cuda.is_available()" || die "env/CUDA not ready."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "smoke on $(hostname): $(python -c 'import torch,transformers;print("torch",torch.__version__,"tf",transformers.__version__)')"

OUT=out/smoke
rm -rf "$OUT"

# tiny variant-a run: 32 examples, no grad-ckpt, no W&B, few checkpoints, log-spaced.
CUDA_VISIBLE_DEVICES=0 python impl3_kl_reweighted_sft/train_kl_sft.py \
    --variant a --temperature 2 --config "$CFG" \
    --train_total 32 --num_epochs 1 --per_device_batch 8 --grad_accum 1 \
    --min_checkpoints 2 --no_grad_checkpointing --no_wandb --output_dir "$OUT" \
    || die "training path errored — see traceback above (likely a dependency/API mismatch)."

# assert at least one checkpoint with a LoRA adapter was actually written.
if ls "$OUT"/checkpoint-*/adapter_model.safetensors >/dev/null 2>&1 || [ -e "$OUT/adapter_model.safetensors" ]; then
    echo "SMOKE OK — full path works (model load, tokenize, weighted loss, checkpoint save)."
else
    die "no checkpoint/adapter written under $OUT — checkpointing path is broken."
fi
