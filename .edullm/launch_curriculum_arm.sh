#!/usr/bin/env bash
# Launch one arm locally; this script never submits compute.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
ARM_INDEX="${ARM_INDEX:?ARM_INDEX (0..4) is required}"
NPROC="${NPROC:-1}"
RUN_DIR="${RUN_DIR:-${TMPDIR:-/tmp}/curriculum-${ARM_INDEX}}"

if [[ "${FRESH:-0}" == "1" && -n "${LOAD_PATH:-}" ]]; then
  echo "FRESH=1 and LOAD_PATH are mutually exclusive" >&2
  exit 2
fi
if [[ "${FRESH:-0}" != "1" && -z "${LOAD_PATH:-}" ]]; then
  echo "choose recovery explicitly with FRESH=1 or LOAD_PATH=..." >&2
  exit 2
fi

ARM_NAME="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["arms"][int(sys.argv[2])]["name"])' "${ROOT}/curriculum_recipe.json" "${ARM_INDEX}")"
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"

ARGS=(
  --arm-index "${ARM_INDEX}"
  --nproc "${NPROC}"
  --run-dir "${RUN_DIR}"
  --wandb-mode "${WANDB_MODE:-online}"
)
if [[ "${FRESH:-0}" == "1" ]]; then
  ARGS+=(--fresh)
else
  ARGS+=(--load-path "${LOAD_PATH}")
fi
if [[ -n "${CURRICULUM_DATASET_VERSION:-}" ]]; then
  ARGS+=(--curriculum-version "${CURRICULUM_DATASET_VERSION}")
fi
if [[ -n "${TASK_LOSS_EVAL_SCRIPT:-}" ]]; then
  ARGS+=(--task-loss-eval-script "${TASK_LOSS_EVAL_SCRIPT}")
fi
if [[ -n "${TASK_LOSS_NPROC:-}" ]]; then
  ARGS+=(--task-loss-nproc "${TASK_LOSS_NPROC}")
fi
if [[ "${LOCAL_SMOKE:-0}" == "1" ]]; then
  ARGS+=(--local-smoke --no-task-loss)
fi

exec "${PYTHON}" "${ROOT}/curriculum_entrypoint.py" "${ARGS[@]}" "$@"
