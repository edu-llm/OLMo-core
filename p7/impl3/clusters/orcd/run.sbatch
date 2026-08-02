#!/bin/bash
#SBATCH -p mit_preemptable          # Standing choice: preemptable schedules far sooner.
#SBATCH -G l40s:1                   # 1 L40S (44GB) fits a 1B LoRA easily; free tier allows up to 2
#SBATCH -c 16                       # <=16 CPUs per L40S GPU on mit_normal_gpu
#SBATCH --mem=64G
#SBATCH -t 04:00:00                 # under the 6h partition limit; each entrypoint --resume auto's
#SBATCH -J p7post
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Generic launcher for any P7 post-training entrypoint. Pass the command via $CMD and
# (optionally) the project root via $PROJECT. Examples:
#
#   # Impl 2 vanilla SFT:
#   sbatch --export=ALL,CMD="python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml --output_dir out/impl2-sft --resume auto" run.sbatch
#
#   # Impl 3 one (variant,T):
#   sbatch --export=ALL,CMD="python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature 2 --config impl3_kl_reweighted_sft/config.yaml --resume auto" run.sbatch
#
#   # Impl 4 one distilled fraction:
#   sbatch --export=ALL,CMD="python impl4_sdft/train_sdft.py --distilled_frac 0.5 --gold_dir data --config impl4_sdft/config.yaml --resume auto" run.sbatch
#
# submit_sweep.sh wraps the Impl-3 / Impl-4 sweeps in one call.
set -euo pipefail
mkdir -p logs

: "${CMD:?Set CMD to the python command to run, e.g. --export=ALL,CMD='python impl1_2_prompting_sft/train_sft.py ...'}"
PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "PROJECT=$PROJECT"
echo "CMD=$CMD"
nvidia-smi || true

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Cache HF models/datasets on big shared storage, not your small home quota.
# Set HF_HOME before submitting, e.g. export HF_HOME=/orcd/pool/<yourpath>/hf_cache
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF_HOME=$HF_HOME"

# W&B logging is ON by default (report_to: wandb). Export WANDB_API_KEY in the shell you
# `sbatch` from (it propagates via --export=ALL). Optionally set WANDB_PROJECT/WANDB_ENTITY.
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — W&B will run in offline/disabled mode. "
    echo "         export WANDB_API_KEY=... before sbatch, or pass --no_wandb in CMD."
fi

cd "$PROJECT"
eval "$CMD"
echo "Done: $CMD"
