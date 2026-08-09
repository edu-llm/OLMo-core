#!/usr/bin/env bash
# Launch one curriculum arm after staging completes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

ARM_INDEX="${ARM_INDEX:-0}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
LENGTH_TOKENS="${LENGTH_TOKENS:-}"

[[ -f "${INPUT_MANIFEST}" ]] || {
  echo "stage inputs first: ${INPUT_MANIFEST}" >&2
  exit 2
}
if [[ -e "${AWS_ENV_FILE}" ]]; then
  echo "temporary AWS credential file still exists; refusing training" >&2
  exit 2
fi
for name in \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
  AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE \
  AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
  AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI; do
  [[ -z "${!name:-}" ]] || {
    echo "${name} is present; refusing training" >&2
    exit 2
  }
done
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
[[ -n "${WANDB_API_KEY:-}" ]] || {
  echo "WANDB_API_KEY is required (${WANDB_ENV_FILE})" >&2
  exit 2
}

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"
case "${ARM_INDEX}" in
  0|1|2|3|4|5|6|7|8) ;;
  *) echo "ARM_INDEX must be 0..8" >&2; exit 2 ;;
esac
arm_name="$(
  ARM_INDEX="${ARM_INDEX}" "${PYTHON}" -c \
    'import os; from curriculum_entrypoint import ARMS; print(ARMS[int(os.environ["ARM_INDEX"])].name)'
)"

arm_root="${RUN_ROOT}/${arm_name}"
mkdir -p "${arm_root}"/{checkpoints,progress,cache}
identity_file="${arm_root}/run.env"
shopt -s nullglob dotglob
checkpoint_entries=("${arm_root}/checkpoints"/*)
shopt -u nullglob dotglob
case "${RECOVERY_MODE}" in
  fresh)
    if [[ -e "${identity_file}" || ${#checkpoint_entries[@]} -ne 0 ]]; then
      echo "fresh run refuses existing state under ${arm_root}" >&2
      exit 2
    fi
    umask 077
    run_name="curriculum-${arm_name}-farmshare-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$("${PYTHON}" -c 'import secrets; print(secrets.token_hex(16))')"
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    recovery=(--fresh)
    ;;
  resume)
    [[ -f "${identity_file}" ]] || {
      echo "resume requires ${identity_file}" >&2
      exit 2
    }
    export WANDB_RESUME=must
    recovery=(--load-path "${LOAD_PATH:-${arm_root}/checkpoints}")
    ;;
  retry-start)
    [[ -f "${identity_file}" ]] || {
      echo "retry-start requires ${identity_file}" >&2
      exit 2
    }
    shopt -s nullglob
    step_entries=("${arm_root}/checkpoints"/step*)
    shopt -u nullglob
    [[ ${#step_entries[@]} -eq 0 ]] || {
      echo "retry-start is only valid before the first checkpoint; use resume" >&2
      exit 2
    }
    export WANDB_RESUME=allow
    recovery=(--fresh)
    ;;
  *)
    echo "RECOVERY_MODE must be fresh, retry-start, or resume" >&2
    exit 2
  ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_DATASET_ID="${DATASET_ID}"
export EDULLM_DATASET_VERSION="${DATASET_VERSION}"
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"

args=(
  --train-worker
  --arm-index "${ARM_INDEX}"
  --nproc "${TRAIN_GPUS}"
  --run-dir "${arm_root}"
  --save-folder "${arm_root}/checkpoints"
  --progress-dir "${arm_root}/progress"
  --cache-dir "${arm_root}/cache"
  --wandb-mode online
  --task-loss-eval-script "${REPO_DIR}/.edullm/task_loss/eval_task_loss_olmo_core.py"
  --ladder-base-config "${REPO_DIR}/.edullm/task_loss/ladder_base_config.yaml"
  --task-loss-nproc "${TRAIN_GPUS}"
  "${recovery[@]}"
)
if [[ -n "${LENGTH_TOKENS}" ]]; then
  args+=(--length-tokens "${LENGTH_TOKENS}")
fi

exec "${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node="${TRAIN_GPUS}" -- \
  "${REPO_DIR}/.edullm/runpod/entrypoint.py" "${args[@]}"
