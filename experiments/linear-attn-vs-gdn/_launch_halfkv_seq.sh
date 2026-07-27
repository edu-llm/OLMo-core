#!/bin/bash
# Half-KV ablation on GPU 4, run SEQUENTIALLY: linear-halfkv first, then gdn-halfkv.
# "Half KV" = head_dim 64 -> 32, which halves both the K and V projection output
# dims (key_dim, value_dim) AND the recurrent KV-state dimensions. Everything else
# (n_heads=16, n_v_heads=16, expand_v=1.0, conv_size=4, data mix, seq-len, schedule)
# is identical to the full runs. LR auto-rescales off the smaller non-embedding count.
#
# W&B key is sourced from the live process env (PID 422430) and is NEVER echoed.
# One shared warm work-dir: the two runs use the SAME data mix, so the second reuses
# the first's dataset index (dataset index is independent of the model geometry).
set -uo pipefail
cd /mnt/nvme/olmo

DPFX="s3://edullm-datasets/olmo-150b-dolma2/data/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer"
EVAL="${DPFX}/arxiv/part-00-00000.npy,${DPFX}/finemath-3plus/part-000-00000.npy,${DPFX}/wikipedia/part-00-00000.npy"

export WANDB_API_KEY="$(tr '\0' '\n' < /proc/422430/environ | grep '^WANDB_API_KEY=' | cut -d= -f2-)"
if [ -z "${WANDB_API_KEY}" ]; then echo "FATAL: could not source WANDB_API_KEY"; exit 3; fi
export CUDA_VISIBLE_DEVICES=4

mkdir -p /mnt/nvme/olmo/logs /mnt/nvme/olmo-work-halfkv

run_one () {
  local NAME="$1" MIXER="$2" SAVE="$3" LOG="$4"
  echo "[$(date -u +%FT%TZ)] START ${NAME} (mixer=${MIXER})" | tee -a "${LOG}"
  /mnt/nvme/olmo/venv/bin/torchrun --standalone --nproc-per-node=1 \
    experiments/linear-attn-vs-gdn/train_mixer.py "${NAME}" --mixer "${MIXER}" \
    --head-dim 32 \
    --dp-type fsdp \
    --save-folder "${SAVE}" \
    --work-dir /mnt/nvme/olmo-work-halfkv \
    --eval-data "${EVAL}" \
    --eval-interval 1000 \
    >> "${LOG}" 2>&1
  local RC=$?
  echo "[$(date -u +%FT%TZ)] END ${NAME} rc=${RC}" | tee -a "${LOG}"
  return ${RC}
}

# Sequential: linear-halfkv, then gdn-halfkv. gdn runs even if linear exits nonzero
# (so a late failure on run 1 does not block run 2); each has its own log.
run_one linear-halfkv-hd32-10b linear \
  s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear-halfkv \
  /mnt/nvme/olmo/logs/linear-halfkv-10b.log

run_one gdn-halfkv-hd32-10b gdn \
  s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/gdn-halfkv \
  /mnt/nvme/olmo/logs/gdn-halfkv-10b.log

echo "[$(date -u +%FT%TZ)] half-KV sequence complete"
