#!/usr/bin/env bash
# Laptop-side submit: sync code, push the W&B session, launch training.
#
# Rewritten for `curriculum-new`: no staging job, no AWS session push (plan
# section 3a). The corpus rebuild materializes every metric's parent on
# scratch once, separately from any individual training submission -- there
# is nothing left to stage per run.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

SUNET="${FARMSHARE_SUNET:-nzhao2}"
SOCK="${FARMSHARE_SOCK:-/tmp/farmshare-${SUNET}.sock}"
HOST="${SUNET}@login.farmshare.stanford.edu"
LOCAL_REPO="${LOCAL_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

PACING="${PACING:?PACING is required}"
METRIC="${METRIC:-}"
CONTROL_METRIC="${CONTROL_METRIC:-}"
LR_SCHEDULE="${LR_SCHEDULE:?LR_SCHEDULE is required}"
SEED="${SEED:?SEED is required}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${RUN_DIR:-/scratch/users/${SUNET}/agent-runs/${EXPERIMENT_SLUG}-${TS}}"

export RUN_DIR LOCAL_REPO SOCK HOST
export PACING METRIC CONTROL_METRIC LR_SCHEDULE SEED RECOVERY_MODE
export TRAIN_GPUS TRAIN_CPUS TRAIN_MEM TRAIN_TIME

bash "${SCRIPT_DIR}/sync_repo.sh"
# Was a call into a separate repository, which left this branch unable to
# submit from a clean checkout. Now self-contained (no edullm repo).
bash "${SCRIPT_DIR}/push_wandb_key.sh"

TRAIN_EXPORT="RUN_DIR='${RUN_DIR}',SCRIPTS_DIR='${RUN_DIR}/scripts',PACING='${PACING}',LR_SCHEDULE='${LR_SCHEDULE}',SEED='${SEED}',RECOVERY_MODE='${RECOVERY_MODE}'"
if [[ -n "${METRIC}" ]]; then
  TRAIN_EXPORT+=",METRIC='${METRIC}'"
fi
if [[ -n "${CONTROL_METRIC}" ]]; then
  TRAIN_EXPORT+=",CONTROL_METRIC='${CONTROL_METRIC}'"
fi
if [[ -n "${LENGTH_TOKENS:-}" ]]; then
  TRAIN_EXPORT+=",LENGTH_TOKENS='${LENGTH_TOKENS}'"
fi
if [[ -n "${PARENT_MANIFEST:-}" ]]; then
  TRAIN_EXPORT+=",PARENT_MANIFEST='${PARENT_MANIFEST}'"
fi
# launch.sh reads KEEP_CHECKPOINTS but nothing was threading it through, so
# `KEEP_CHECKPOINTS=all` on the laptop was silently ignored and the run
# pruned its ladder anyway.
if [[ -n "${KEEP_CHECKPOINTS:-}" ]]; then
  TRAIN_EXPORT+=",KEEP_CHECKPOINTS='${KEEP_CHECKPOINTS}'"
fi
if [[ -n "${CORPUS_ROOT:-}" ]]; then
  TRAIN_EXPORT+=",CORPUS_ROOT='${CORPUS_ROOT}'"
fi
# --export=ALL on the remote sbatch call only inherits the SSH session's own
# environment, not this laptop shell's -- a laptop-side EDULLM_WANDB_PROJECT
# override must be explicitly threaded through the same way.
if [[ -n "${EDULLM_WANDB_PROJECT:-}" ]]; then
  TRAIN_EXPORT+=",EDULLM_WANDB_PROJECT='${EDULLM_WANDB_PROJECT}'"
fi

DEP_FLAG=""
if [[ -n "${DEPENDS_ON:-}" ]]; then
  DEP_FLAG="--dependency=afterany:${DEPENDS_ON}"
fi

TRAIN_JOB="$(ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" bash -s <<EOF
set -Eeuo pipefail
export PATH=/usr/bin:/bin:/usr/local/bin:\${PATH:-}
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
