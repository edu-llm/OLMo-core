#!/usr/bin/env bash
# Launch a curriculum arm after stage_inputs.py has removed the temporary AWS session.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/curriculum}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/curriculum/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
ARM_INDEX="${ARM_INDEX:-0}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"

[[ -f "${INPUT_MANIFEST}" ]] || { echo "stage inputs first: ${INPUT_MANIFEST}" >&2; exit 2; }
if [[ -e "${AWS_ENV_FILE:-/workspace/aws-session.env}" ]]; then
  echo "temporary AWS credential file still exists; refusing training" >&2
  exit 2
fi
for name in \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
  AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE \
  AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
  AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI; do
  [[ -z "${!name:-}" ]] || { echo "${name} is present; refusing training" >&2; exit 2; }
done
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required" >&2; exit 2; }

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"
case "${ARM_INDEX}" in
  0|1|2|3|4|5|6) ;;
  *) echo "ARM_INDEX must be 0..6" >&2; exit 2 ;;
esac
arm_name="$(
  ARM_INDEX="${ARM_INDEX}" python3 -c \
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
    run_name="curriculum-${arm_name}-runpod-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    recovery=(--fresh)
    ;;
  resume)
    [[ -f "${identity_file}" ]] || { echo "resume requires ${identity_file}" >&2; exit 2; }
    export WANDB_RESUME=must
    recovery=(--load-path "${LOAD_PATH:-${arm_root}/checkpoints}")
    ;;
  retry-start)
    [[ -f "${identity_file}" ]] || { echo "retry-start requires ${identity_file}" >&2; exit 2; }
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
  *) echo "RECOVERY_MODE must be fresh, retry-start, or resume" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_DATASET_ID="pretrain/regmix-10b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"

args=(
  --train-worker
  --arm-index "${ARM_INDEX}"
  --nproc 8
  --run-dir "${arm_root}"
  --save-folder "${arm_root}/checkpoints"
  --progress-dir "${arm_root}/progress"
  --cache-dir "${arm_root}/cache"
  --wandb-mode online
  --task-loss-eval-script "${REPO_DIR}/.edullm/task_loss/eval_task_loss_olmo_core.py"
  --ladder-base-config "${REPO_DIR}/.edullm/task_loss/ladder_base_config.yaml"
  --task-loss-nproc 8
  "${recovery[@]}"
)
if [[ -n "${LENGTH_TOKENS:-}" ]]; then
  args+=(--length-tokens "${LENGTH_TOKENS}")
fi

exec python3 -m torch.distributed.run --standalone --nproc-per-node=8 -- \
  "${REPO_DIR}/.edullm/runpod/entrypoint.py" "${args[@]}"
