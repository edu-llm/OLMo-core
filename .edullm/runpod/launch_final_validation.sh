#!/usr/bin/env bash
# Launch one stock OLMo2-370M / RegMix-10B validation on 8 GPUs.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/hpo-final-validation}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/hpo-probe/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
AWS_ENV_FILE="${AWS_ENV_FILE:-/workspace/aws-session.env}"
VECTOR="${VECTOR:?set VECTOR=no-proxy-winner or VECTOR=no-centaur-winner}"
RUN_SLOT="${RUN_SLOT:-${VECTOR}}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
HARD_TIME_LIMIT="${HARD_TIME_LIMIT:-48h}"

case "${VECTOR}" in
  no-proxy-winner|no-centaur-winner) ;;
  *)
    echo "VECTOR must be no-proxy-winner or no-centaur-winner" >&2
    exit 2
    ;;
esac
[[ "${RUN_SLOT}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "RUN_SLOT may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}
[[ -d "${REPO_DIR}/.git" ]] || { echo "missing checkout: ${REPO_DIR}" >&2; exit 2; }
[[ -f "${INPUT_MANIFEST}" ]] || { echo "stage RegMix first: ${INPUT_MANIFEST}" >&2; exit 2; }
[[ ! -e "${AWS_ENV_FILE}" ]] || {
  echo "temporary AWS credential file still exists; refusing training" >&2
  exit 2
}
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

job_root="${RUN_ROOT}/${RUN_SLOT}"
checkpoint_root="${job_root}/checkpoints"
identity_file="${job_root}/run.env"
mkdir -p "${job_root}"
case "${RECOVERY_MODE}" in
  fresh)
    shopt -s nullglob dotglob
    existing=("${job_root}"/*)
    shopt -u nullglob dotglob
    if [[ ${#existing[@]} -ne 0 ]]; then
      echo "fresh run refuses existing state under ${job_root}" >&2
      exit 2
    fi
    run_name="hpo-final-370m-10b-${VECTOR}-runpod-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    umask 077
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    ;;
  resume)
    [[ -f "${identity_file}" ]] || { echo "resume requires ${identity_file}" >&2; exit 2; }
    export WANDB_RESUME=must
    ;;
  retry-startup)
    [[ -f "${identity_file}" ]] || {
      echo "retry-startup requires ${identity_file}" >&2
      exit 2
    }
    export WANDB_RESUME=allow
    ;;
  *)
    echo "RECOVERY_MODE must be fresh, retry-startup, or resume" >&2
    exit 2
    ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm:${REPO_DIR}/.edullm/runpod"
export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_DATASET_ID="pretrain/regmix-10b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_DATASET_TOKENIZER="tokenizer/dolma2-bpe"
export EDULLM_CHECKPOINT_DIR="${checkpoint_root}"
export EDULLM_WANDB_PROJECT="hpo-final-validation"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"
export WANDB_MODE=online
export WANDB_RUN_GROUP="hpo-final-validation-370m-10b"
export EDULLM_EVAL_WORK_DIR="${job_root}/eval-work"
export PYTHONDONTWRITEBYTECODE=1

echo "Launching ${VECTOR}/${RUN_SLOT}; hard wall-time limit=${HARD_TIME_LIMIT}"
set +e
timeout --signal=TERM --kill-after=120s "${HARD_TIME_LIMIT}" \
  python3 "${REPO_DIR}/.edullm/runpod/final_validation_entrypoint.py" \
    --vector "${VECTOR}" 2>&1 | tee -a "${job_root}/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${job_root}/last-exit-code"
if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
  echo "Run reached ${HARD_TIME_LIMIT}; resume with RECOVERY_MODE=resume" >&2
fi
exit "${status}"
