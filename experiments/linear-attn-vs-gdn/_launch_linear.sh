#!/bin/bash
# Launch the full 10B LINEAR-attention run on GPU 5.
# W&B key is sourced from the live `equal` baseline process env (PID 422430) and
# is NEVER echoed. Output -> /mnt/nvme/olmo/logs/linear-10b.log
set -uo pipefail
cd /mnt/nvme/olmo

DPFX="s3://edullm-datasets/olmo-150b-dolma2/data/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer"
EVAL="${DPFX}/arxiv/part-00-00000.npy,${DPFX}/finemath-3plus/part-000-00000.npy,${DPFX}/wikipedia/part-00-00000.npy"

# Source the W&B key without printing it.
export WANDB_API_KEY="$(tr '\0' '\n' < /proc/422430/environ | grep '^WANDB_API_KEY=' | cut -d= -f2-)"
if [ -z "${WANDB_API_KEY}" ]; then echo "FATAL: could not source WANDB_API_KEY"; exit 3; fi
export CUDA_VISIBLE_DEVICES=5

mkdir -p /mnt/nvme/olmo/logs
LOG=/mnt/nvme/olmo/logs/linear-10b.log

setsid nohup /mnt/nvme/olmo/venv/bin/torchrun --standalone --nproc-per-node=1 \
  experiments/linear-attn-vs-gdn/train_mixer.py linear-attn-370m-10b --mixer linear \
  --dp-type fsdp \
  --save-folder s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear \
  --work-dir /mnt/nvme/olmo-work-linear \
  --eval-data "${EVAL}" \
  --eval-interval 1000 \
  > "${LOG}" 2>&1 &

echo "LINEAR launched: pid=$! gpu=5 log=${LOG}"
