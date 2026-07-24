#!/bin/bash
#SBATCH --job-name=p4-simple
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=p4-simple-%j.out

set -euo pipefail

: "${MODEL:?Set MODEL to the local OLMo-170M checkpoint or converted HF directory}"

ROOT="${SLURM_SUBMIT_DIR}"
LOCAL_ROOT="/tmp/${USER}-p4-${SLURM_JOB_ID}"
mkdir -p "${LOCAL_ROOT}"

# PyTorch imports were extremely slow over FarmShare's shared scratch
# filesystem. Copy one compressed environment archive to node-local storage.
cp "${ROOT}/environment.tar.zst" "${LOCAL_ROOT}/environment.tar.zst"
tar --zstd -xf "${LOCAL_ROOT}/environment.tar.zst" -C "${LOCAL_ROOT}"
cp "${ROOT}/generate_data.py" "${ROOT}/train_compare.py" "${LOCAL_ROOT}/"
cp -r "${ROOT}/data" "${LOCAL_ROOT}/data"

mkdir -p "${LOCAL_ROOT}/huggingface"
if [ -d "/scratch/users/${USER}/.cache/huggingface" ]; then
  cp -r "/scratch/users/${USER}/.cache/huggingface/." "${LOCAL_ROOT}/huggingface/"
fi
export HF_HOME="${LOCAL_ROOT}/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -e "${MODEL}" ]; then
  cp -r "${MODEL}" "${LOCAL_ROOT}/model"
  MODEL_ARG="${LOCAL_ROOT}/model"
else
  MODEL_ARG="${MODEL}"
fi

EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [ -n "${ORDER_INDICES:-}" ]; then
  EXTRA_ARGS+=(--order-indices "${ORDER_INDICES}")
fi

nvidia-smi
"${LOCAL_ROOT}/.venv/bin/python" "${LOCAL_ROOT}/generate_data.py" \
  --output-dir "${LOCAL_ROOT}/data"
"${LOCAL_ROOT}/.venv/bin/python" "${LOCAL_ROOT}/train_compare.py" \
  --model "${MODEL_ARG}" \
  --data-dir "${LOCAL_ROOT}/data" \
  --output-dir "${ROOT}/outputs/${SLURM_JOB_ID}" \
  --deterministic \
  "${EXTRA_ARGS[@]}"
