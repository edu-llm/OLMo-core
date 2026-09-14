#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --exclude=wheat-01
#SBATCH --job-name=curriculum-new-train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -Eeuo pipefail
: "${RUN_DIR:?}"
cd "${RUN_DIR}"
SCRIPTS_DIR="${SCRIPTS_DIR:-${RUN_DIR}/scripts}"
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/common.sh"
# No staging step (plan section 3a): the corpus rebuild materializes every
# metric's parent on scratch once, ahead of any training submission.
bash "${SCRIPTS_DIR}/setup_venv.sh"
# Re-source now that setup_venv.sh has created the venv, so $PYTHON resolves
# to it (needed below to import curriculum_entrypoint, which pulls in
# torch/olmo_core).
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/common.sh"

# Every non-final permanent checkpoint (step 0 excluded) makes the training
# worker hard-stop cleanly after its synchronous task-loss eval, to clear
# compiled CUDA state (see curriculum_entrypoint.py's
# CurriculumCheckpointCallback._finalize). FarmShare has no external
# supervisor, so loop here: relaunch with RECOVERY_MODE=resume whenever a
# restart request is left behind, until training actually finishes or
# MAX_RESTARTS is hit.
PACING="${PACING:?}"
LR_SCHEDULE="${LR_SCHEDULE:?}"
SEED="${SEED:?}"
METRIC="${METRIC:-}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

arm_name="${PACING}"
if [[ "${PACING}" != "control" ]]; then
  arm_name="${arm_name}-${METRIC}"
fi
arm_name="${arm_name}-${LR_SCHEDULE}-seed${SEED}"

arm_root="${RUN_ROOT}/${arm_name}"
restart_request="${arm_root}/progress/restart_after_checkpoint.json"

mode="${RECOVERY_MODE:-fresh}"
max_restarts="${MAX_RESTARTS:-40}"
attempt=0
while true; do
  attempt=$((attempt + 1))
  if [[ ${attempt} -gt ${max_restarts} ]]; then
    echo "exceeded MAX_RESTARTS=${max_restarts} without training finishing" >&2
    exit 3
  fi
  rm -f "${restart_request}"
  echo "=== launch attempt ${attempt}, RECOVERY_MODE=${mode} ==="
  RECOVERY_MODE="${mode}" bash "${SCRIPTS_DIR}/launch.sh"
  if [[ -f "${restart_request}" ]]; then
    echo "=== checkpoint boundary reached; resuming as attempt $((attempt + 1)) ==="
    mode="resume"
    continue
  fi
  echo "=== launch.sh exited with no pending restart request; training run finished ==="
  break
done
