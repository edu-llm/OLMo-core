#!/bin/bash
#
# Submit an OLMo-core training run to the shared AWS SLURM scheduler.
#
# Usage:
#   ./submit.sh <branch> <train_script> <run_name> <gpus> [-- <config overrides...>]
#
# Example:
#   ./submit.sh my-attn-experiment OLMo2/OLMo2-190M.py attn-exp-lr1e3 1 \
#       -- --train_module.optim.lr=1e-3 --dataset.mix=<your-mix>
#
# Requires (set in your shell or env.example.sh):
#   OLMO_CHECKPOINT_S3   e.g. s3://<project>-checkpoints
# Optional:
#   OLMO_REPO_URL, OLMO_CLUSTER_LABEL, OLMO_VENV, OLMO_ECR_IMAGE, OLMO_WORKROOT
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <branch> <train_script> <run_name> <gpus> [-- <overrides...>]" >&2
    exit 1
fi

BRANCH="$1"; TRAIN_SCRIPT="$2"; RUN_NAME="$3"; GPUS="$4"; shift 4
[ "${1:-}" = "--" ] && shift

: "${OLMO_CHECKPOINT_S3:?set OLMO_CHECKPOINT_S3 (e.g. s3://<project>-checkpoints)}"

HERE="$(cd "$(dirname "$0")" && pwd)"

sbatch \
    --job-name="$RUN_NAME" \
    --gpus-per-node="$GPUS" \
    --export="ALL,OLMO_BRANCH=${BRANCH},OLMO_TRAIN_SCRIPT=${TRAIN_SCRIPT},OLMO_RUN_NAME=${RUN_NAME},OLMO_CHECKPOINT_S3=${OLMO_CHECKPOINT_S3}" \
    "$HERE/train-job.sbatch" "$@"
