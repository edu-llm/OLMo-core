#!/usr/bin/env bash
# Laptop-side submit: sync code, push sessions, stage inputs, launch training.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

EDULLM_ROOT="${EDULLM_ROOT:-/mnt/c/alpha_ai/edullm}"
SUNET="${FARMSHARE_SUNET:-nzhao2}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu}"
LOCAL_REPO="${LOCAL_REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

ARM_INDEX="${ARM_INDEX:-0}"
ARM="${ARM:-}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
SKIP_STAGE="${SKIP_STAGE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${EXPERIMENT_SLUG}-${TS}}"

export RUN_DIR LOCAL_REPO SOCK HOST
export ARM_INDEX ARM RECOVERY_MODE
export STAGE_CPUS STAGE_MEM STAGE_TIME TRAIN_GPUS TRAIN_CPUS TRAIN_MEM TRAIN_TIME
export CURRICULUM_VERSION REFHQ_VERSION DATASET_VERSION

bash "${SCRIPT_DIR}/sync_repo.sh"

bash "${EDULLM_ROOT}/scripts/farmshare/push_aws_session_to_farmshare.sh" "${RUN_DIR}"
bash "${EDULLM_ROOT}/scripts/farmshare/push_wandb_session_to_farmshare.sh" "${RUN_DIR}"

STAGE_EXPORT="RUN_DIR='${RUN_DIR}',SCRIPTS_DIR='${RUN_DIR}/scripts',ARM_INDEX='${ARM_INDEX}'"
STAGE_EXPORT+=",STAGE_HEADROOM='${STAGE_HEADROOM:-1.10}'"
if [[ -n "${ARM}" ]]; then
  STAGE_EXPORT+=",ARM='${ARM}'"
fi
if [[ -n "${LENGTH_TOKENS:-}" ]]; then
  STAGE_EXPORT+=",LENGTH_TOKENS='${LENGTH_TOKENS}'"
fi
if [[ -n "${CURRICULUM_VERSION:-}" ]]; then
  STAGE_EXPORT+=",CURRICULUM_VERSION='${CURRICULUM_VERSION}'"
fi

STAGE_JOB=""
if [[ "${SKIP_STAGE}" != "1" ]]; then
  STAGE_JOB="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
JOB=\$(sbatch --parsable --exclude=wheat-01 \
  --partition=gpu \
  --qos=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=${STAGE_GPUS:-1} \
  --cpus-per-task=${STAGE_CPUS} \
  --mem=${STAGE_MEM} \
  --time=${STAGE_TIME} \
  --job-name=${EXPERIMENT_SLUG}-stage \
  --chdir='${RUN_DIR}' \
  --output='${RUN_DIR}/logs/stage-%j.out' \
  --error='${RUN_DIR}/logs/stage-%j.err' \
  --export=ALL,${STAGE_EXPORT} \
  '${RUN_DIR}/scripts/stage_job.sbatch')
echo "\${JOB}"
EOF
)"
  echo "stage_job=${STAGE_JOB}"
fi

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "RUN_DIR=${RUN_DIR}"
  exit 0
fi

DEP_FLAG=""
if [[ -n "${STAGE_JOB}" ]]; then
  DEP_FLAG="--dependency=afterok:${STAGE_JOB}"
fi

TRAIN_EXPORT="RUN_DIR='${RUN_DIR}',SCRIPTS_DIR='${RUN_DIR}/scripts',ARM_INDEX='${ARM_INDEX}',RECOVERY_MODE='${RECOVERY_MODE}'"
if [[ -n "${ARM}" ]]; then
  TRAIN_EXPORT+=",ARM='${ARM}'"
fi
if [[ -n "${LENGTH_TOKENS:-}" ]]; then
  TRAIN_EXPORT+=",LENGTH_TOKENS='${LENGTH_TOKENS}'"
fi

TRAIN_JOB="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
JOB=\$(sbatch --parsable --exclude=wheat-01 ${DEP_FLAG} \
  --partition=gpu \
  --qos=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=${TRAIN_GPUS} \
  --cpus-per-task=${TRAIN_CPUS} \
  --mem=${TRAIN_MEM} \
  --time=${TRAIN_TIME} \
  --job-name=${EXPERIMENT_SLUG}-train \
  --chdir='${RUN_DIR}' \
  --output='${RUN_DIR}/logs/train-%j.out' \
  --error='${RUN_DIR}/logs/train-%j.err' \
  --export=ALL,${TRAIN_EXPORT} \
  '${RUN_DIR}/scripts/train_job.sbatch')
echo "\${JOB}"
EOF
)"

echo "RUN_DIR=${RUN_DIR}"
echo "train_job=${TRAIN_JOB}"
