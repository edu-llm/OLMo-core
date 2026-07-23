#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G l40s:1
#SBATCH -t 00:05:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -J edullm-gpu-wandb-smoke
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
umask 077

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
: "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"
WANDB_ENV="${WANDB_ENV:-$HOME/.config/edullm/wandb.env}"

mkdir -p "$EDULLM_SCRATCH/logs" "$EDULLM_SCRATCH/probes" "$EDULLM_SCRATCH/wandb"
exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1

test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
EDULLM_BRANCH="$(git -C "$EDULLM_REPO_ROOT" branch --show-current)"
test -n "$EDULLM_BRANCH"
test "$EDULLM_BRANCH" != main

source "$EDULLM_VENV/bin/activate"
source "$WANDB_ENV"
: "${WANDB_API_KEY:?W&B key is not configured}"
: "${WANDB_ENTITY:?W&B entity is not configured}"
: "${WANDB_PROJECT:?W&B project is not configured}"

RUN_NAME="${RUN_NAME:-gpu-wandb-access-${SLURM_JOB_ID}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"
export WANDB_DIR="$EDULLM_SCRATCH/wandb"
export PYTHONPATH="$EDULLM_REPO_ROOT/src"

nvidia-smi
python "$EDULLM_REPO_ROOT/src/scripts/orcd/gpu_wandb_smoke.py" \
  --run-name "$RUN_NAME" \
  --output "$EDULLM_SCRATCH/probes/${SLURM_JOB_ID}-gpu-wandb.json"
