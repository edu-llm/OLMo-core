#!/usr/bin/env bash
# Shared FarmShare runtime for OLMo-core 370M experiments.
set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"

SCRIPTS_DIR="${SCRIPTS_DIR:-${RUN_DIR}/scripts}"
if [[ -f "${SCRIPTS_DIR}/config.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPTS_DIR}/config.env"
fi

source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.3 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || module load cuda/12.9.0 2>/dev/null || true

export OLMO_FLASH_ATTENTION=0
export OLMO_ATTN_BACKEND=torch
export OLMO_FUSED_LOSS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# recipe.py's RANK_MICROBATCH_TOKENS=32768 (16 sequences/microbatch) was
# sized for flash-attention on A100/H100. It OOMs an L40S (44GB) running the
# `torch` SDPA backend forced above (materializes full attention score
# matrices), independent of how many GPUs/ranks are used -- each rank still
# handles a full 32768-token microbatch on its own. 16384 (8 sequences) is
# verified to fit with headroom to spare; override if a run needs otherwise.
export EDULLM_RANK_MICROBATCH_TOKENS="${EDULLM_RANK_MICROBATCH_TOKENS:-16384}"

# $HOME is quota-limited and can already be at/over quota independent of
# anything this run does -- torch.compile's Triton/Inductor caches, W&B's
# artifact staging (~/.local/share/wandb), and HuggingFace's dataset cache
# (~/.cache/huggingface) have each separately hit this mid-run. Scratch has
# effectively no quota, so redirect every XDG-respecting tool's base dirs
# there wholesale rather than chasing each tool's own env var one at a time.
mkdir -p "${RUN_DIR}/.cache/triton" "${RUN_DIR}/.cache/inductor" \
  "${RUN_DIR}/.xdg/cache" "${RUN_DIR}/.xdg/data" "${RUN_DIR}/.xdg/config"
export TRITON_CACHE_DIR="${RUN_DIR}/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUN_DIR}/.cache/inductor"
export XDG_CACHE_HOME="${RUN_DIR}/.xdg/cache"
export XDG_DATA_HOME="${RUN_DIR}/.xdg/data"
export XDG_CONFIG_HOME="${RUN_DIR}/.xdg/config"
# HuggingFace's XDG_CACHE_HOME support has been inconsistent across versions;
# set HF_HOME explicitly too rather than relying on it alone.
export HF_HOME="${RUN_DIR}/.xdg/cache/huggingface"

REPO_DIR="${REPO_DIR:-${RUN_DIR}/OLMo-core}"
VENV="${VENV:-${RUN_DIR}/venv}"
RUN_ROOT="${RUN_ROOT:-${RUN_DIR}/runs}"
STAGE_ROOT="${STAGE_ROOT:-${RUN_DIR}/${STAGE_ROOT_REL:-inputs}}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-${STAGE_ROOT}/ready.json}"
AWS_ENV_FILE="${AWS_ENV_FILE:-${RUN_DIR}/aws-session.env}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${RUN_DIR}/wandb-session.env}"
if [[ -x "${VENV}/bin/python3" ]]; then
  PYTHON="${VENV}/bin/python3"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_ROOT}" "${STAGE_ROOT}"
