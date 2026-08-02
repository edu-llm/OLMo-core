#!/bin/bash
# Run a P7 post-training entrypoint on an AWS GPU box, with S3 checkpoint sync so runs
# survive a box teardown and resume elsewhere. Run inside tmux for long jobs.
#
# Env vars:
#   CMD        (required) the python command, e.g.
#              CMD="python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml --output_dir out/impl2-sft --resume auto"
#   OUT_DIR    local output dir the CMD writes to (default: out)
#   S3_DEST    s3 prefix to sync to/from (default: s3://alphaai-edullm-checkpoints/p7/<user>)
#   GPUS       CUDA_VISIBLE_DEVICES to use (default: 0). Multi-GPU: use torchrun/accelerate (see AWS_RUN.md).
#
# Example:
#   export CMD="python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature 2 --config impl3_kl_reweighted_sft/config.yaml --resume auto"
#   export OUT_DIR=out/impl3-a-T2
#   bash clusters/aws/run_aws.sh
set -euo pipefail

: "${CMD:?Set CMD to the python command to run.}"
VENV="${VENV:-$HOME/p7post-venv}"
OUT_DIR="${OUT_DIR:-out}"
S3_DEST="${S3_DEST:-s3://alphaai-edullm-checkpoints/p7/${USER:-run}}"
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"

# Box uses its EC2 instance role for S3 (region us-east-2). A laptop profile => ProfileNotFound.
unset AWS_PROFILE || true
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}"

source "$VENV/bin/activate"
cd "$(dirname "$0")/../.."

# W&B logging is ON by default (report_to: wandb). Export WANDB_API_KEY before running,
# or pass --no_wandb in CMD. Optionally set WANDB_PROJECT / WANDB_ENTITY.
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — W&B disabled/offline. export it or add --no_wandb."
fi

echo "CMD=$CMD"
echo "OUT_DIR=$OUT_DIR | S3_DEST=$S3_DEST/$(basename "$OUT_DIR") | GPUS=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Pull any existing checkpoints for this run so --resume auto can continue.
aws s3 sync "$S3_DEST/$(basename "$OUT_DIR")" "$OUT_DIR" || echo "(no prior checkpoints in S3)"

# Train. On SIGINT the HF Trainer saves before exiting.
eval "$CMD"

# Push results back to S3 (durable across box teardown).
aws s3 sync "$OUT_DIR" "$S3_DEST/$(basename "$OUT_DIR")"
echo "Synced $OUT_DIR -> $S3_DEST/$(basename "$OUT_DIR")"
