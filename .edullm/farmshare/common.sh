#!/usr/bin/env bash
# Shared FarmShare runtime for OLMo-core 370M curriculum experiments.
#
# Rewritten for `curriculum-new`: no AWS, no S3 staging manifest -- the
# corpus rebuild (plan section 4/5) materializes each metric's parent
# directly on FarmShare scratch, once, ahead of any training job. A run
# just points --parent-manifest at that already-materialized location.
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

# Nothing may land in $HOME. FarmShare home carries a 48 GB quota, and the
# tools below all default to ~/.cache or ~ without saying so: a tokenizer
# download, W&B's artifact staging copy, and Triton/Inductor kernel caches
# will fill it mid-run and fail the job for a reason that looks like
# anything but a quota. Every one of them is redirected onto scratch here.
# (Model artifacts are additionally never uploaded at all -- see
# ALLOW_MODEL_ARTIFACT_UPLOAD in .edullm/production_contract/wandb_artifacts.py.)
CACHE_ROOT="${CACHE_ROOT:-${RUN_DIR}/cache}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${CACHE_ROOT}/xdg-config"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
export CUDA_CACHE_PATH="${CACHE_ROOT}/nv"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
export WANDB_ARTIFACT_DIR="${CACHE_ROOT}/wandb-artifacts"
export WANDB_DATA_DIR="${CACHE_ROOT}/wandb-data"
export WANDB_CONFIG_DIR="${CACHE_ROOT}/wandb-config"

mkdir -p \
  "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${PIP_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
  "${WANDB_ARTIFACT_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CONFIG_DIR}"

# Fail loudly rather than silently filling the quota: assert that none of
# the redirects (or the run root itself) actually resolve under $HOME. A
# typo in RUN_DIR would otherwise be discovered hours into a run.
if [[ -n "${HOME:-}" ]]; then
  _home_real="$(readlink -f "${HOME}")"
  for _var in RUN_DIR CACHE_ROOT XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME \
              HF_HUB_CACHE TORCH_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR \
              CUDA_CACHE_PATH PIP_CACHE_DIR WANDB_DIR WANDB_CACHE_DIR \
              WANDB_ARTIFACT_DIR WANDB_DATA_DIR WANDB_CONFIG_DIR; do
    _val="$(readlink -f "${!_var}")"
    if [[ "${_val}" == "${_home_real}" || "${_val}" == "${_home_real}/"* ]]; then
      echo "refusing to run: ${_var}=${_val} is under \$HOME (48 GB quota)." >&2
      echo "point RUN_DIR at /scratch/users/${USER}/... instead" >&2
      exit 1
    fi
  done
  unset _home_real _var _val
fi

REPO_DIR="${REPO_DIR:-${RUN_DIR}/OLMo-core}"
VENV="${VENV:-${RUN_DIR}/venv}"
RUN_ROOT="${RUN_ROOT:-${RUN_DIR}/runs}"
# Root under which the corpus rebuild materialized per-metric parents; each
# metric's manifest is expected at "${CORPUS_ROOT}/<metric>/manifest.json".
CORPUS_ROOT="${CORPUS_ROOT:-/scratch/users/${USER}/curriculum-new-corpus}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${RUN_DIR}/wandb-session.env}"
if [[ -x "${VENV}/bin/python3" ]]; then
  PYTHON="${VENV}/bin/python3"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_ROOT}"
