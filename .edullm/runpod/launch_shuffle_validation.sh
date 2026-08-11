#!/usr/bin/env bash
# Launch dense OLMo2-370M shuffle baseline with the curriculum HPO winner on 8 A100s.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/hpo-validation}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/hpo-probe/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
AWS_ENV_FILE="${AWS_ENV_FILE:-/workspace/aws-session.env}"
RUN_SLOT="${RUN_SLOT:-shuffle-mtld-370m-mb32k-v1}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
HARD_TIME_LIMIT="${HARD_TIME_LIMIT:-72h}"

[[ "${RUN_SLOT}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_SLOT" >&2; exit 2; }
[[ -d "${REPO_DIR}/.git" ]] || { echo "missing checkout: ${REPO_DIR}" >&2; exit 2; }
[[ -f "${INPUT_MANIFEST}" ]] || { echo "missing staged dataset manifest" >&2; exit 2; }
[[ ! -e "${AWS_ENV_FILE}" ]] || { echo "AWS credential file is present; refusing" >&2; exit 2; }
for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI; do
  [[ -z "${!name:-}" ]] || { echo "${name} is present; refusing" >&2; exit 2; }
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
    [[ ${#existing[@]} -eq 0 ]] || { echo "fresh run refuses existing state" >&2; exit 2; }
    run_name="hpo-validation-olmo2-370m-shuffle-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    umask 077
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" "${run_name}" "${wandb_id}" >"${identity_file}"
    export WANDB_RESUME=never
    ;;
  resume)
    [[ -f "${identity_file}" ]] || { echo "resume requires ${identity_file}" >&2; exit 2; }
    export WANDB_RESUME=must
    ;;
  *) echo "RECOVERY_MODE must be fresh or resume" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

available_kib="$(df --output=avail /workspace | awk 'NR==2 {print $1}')"
required_kib="$((300 * 1024 * 1024))"
[[ "${available_kib}" -ge "${required_kib}" ]] || {
  echo "insufficient free workspace: $((available_kib / 1024 / 1024)) GiB available, 300 GiB required" >&2
  exit 2
}

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm:${REPO_DIR}/.edullm/runpod"
export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_DATASET_ID="pretrain/opt-with-synthetic-10b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_DATASET_TOKENIZER="tokenizer/dolma2-bpe"
export EDULLM_CHECKPOINT_DIR="${checkpoint_root}"
export EDULLM_WANDB_PROJECT="hpo-validation"
export WANDB_PROJECT="hpo-validation"
export WANDB_MODE=online
export WANDB_RUN_GROUP="hpo-validation-olmo2-370m-quadratic-mtld-shuffle"
export EDULLM_EVAL_WORK_DIR="${job_root}/eval-work"
export PYTHONDONTWRITEBYTECODE=1

echo "Launching ${RUN_SLOT}; project=hpo-validation; hard limit=${HARD_TIME_LIMIT}"
set +e
timeout --signal=TERM --kill-after=120s "${HARD_TIME_LIMIT}" \
  python3 "${REPO_DIR}/.edullm/runpod/shuffle_validation_entrypoint.py" \
  2>&1 | tee -a "${job_root}/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" >"${job_root}/last-exit-code"
exit "${status}"
