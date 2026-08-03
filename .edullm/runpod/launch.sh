#!/usr/bin/env bash
# Launch one token-selection arm after all S3 inputs have been staged locally.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/token-selection}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/token-selection/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
ARM="${ARM:-attention}"
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
ARM="${ARM}" python3 -c \
  'import os; from token_selection_370m.arms import get_arm; get_arm(os.environ["ARM"])' \
  >/dev/null
arm_root="${RUN_ROOT}/${ARM}"
mkdir -p "${arm_root}"/{checkpoints,work,progress}
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
    run_name="token-selection-${ARM}-runpod-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    recovery=()
    ;;
  resume)
    [[ -f "${identity_file}" ]] || { echo "resume requires ${identity_file}" >&2; exit 2; }
    export WANDB_RESUME=must
    recovery=(--resume)
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
    rm -f \
      "${arm_root}/checkpoints/run_fingerprint.json" \
      "${arm_root}/progress/run_identity.json"
    export WANDB_RESUME=allow
    recovery=()
    ;;
  *) echo "RECOVERY_MODE must be fresh, retry-start, or resume" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_WANDB_PROJECT="token-selection-${ARM}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"
export WANDB_MODE=online

exec python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${REPO_DIR}/.edullm/runpod/entrypoint.py" \
  --arm "${ARM}" \
  --save-folder "${arm_root}/checkpoints" \
  --work-dir "${arm_root}/work" \
  --progress-dir "${arm_root}/progress" \
  --task-loss-script "${REPO_DIR}/.edullm/eval_task_loss_olmo_core.py" \
  "${recovery[@]}"
