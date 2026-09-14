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
