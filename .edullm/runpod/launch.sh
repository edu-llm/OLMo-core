#!/usr/bin/env bash
# Launch one three-arm HPO controller or the paired proxy cohort on 8 GPUs.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/hpo-probe}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/hpo-probe/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
AWS_ENV_FILE="${AWS_ENV_FILE:-/workspace/aws-session.env}"
MODE="${MODE:-no_proxy}"
RUN_SLOT="${RUN_SLOT:-default}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
CONTROLLER_SPEC="${CONTROLLER_SPEC:-}"
readonly HARD_TIME_LIMIT="4h"
readonly HARD_LIMIT_SECONDS=14400
launch_started_at="$(date +%s)"

case "${MODE}" in
  proxy-cohort|full_acronym_soup|no_centaur|no_proxy) ;;
  *)
    echo "MODE must be proxy-cohort, full_acronym_soup, no_centaur, or no_proxy" >&2
    exit 2
    ;;
esac
[[ "${RUN_SLOT}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "RUN_SLOT may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}
[[ -d "${REPO_DIR}/.git" ]] || { echo "missing checkout: ${REPO_DIR}" >&2; exit 2; }
[[ -f "${INPUT_MANIFEST}" ]] || { echo "stage inputs first: ${INPUT_MANIFEST}" >&2; exit 2; }
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
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is required (${WANDB_ENV_FILE})" >&2; exit 2; }
case "${MODE}" in
  full_acronym_soup|no_proxy)
    [[ -n "${OPENAI_API_KEY:-}" ]] || {
      echo "OPENAI_API_KEY is required for the Centaur arms" >&2
      exit 2
    }
    ;;
esac

job_root="${RUN_ROOT}/${MODE}/${RUN_SLOT}"
shared_root="${RUN_ROOT}/shared"
identity_file="${job_root}/run.env"
mkdir -p "${job_root}" "${shared_root}"
case "${RECOVERY_MODE}" in
  fresh)
    shopt -s nullglob dotglob
    existing=("${job_root}"/*)
    shopt -u nullglob dotglob
    if [[ ${#existing[@]} -ne 0 ]]; then
      echo "fresh run refuses existing state under ${job_root}" >&2
      exit 2
    fi
    if [[ "${MODE}" == "proxy-cohort" && -e "${shared_root}/proxy-evidence.json" ]]; then
      echo "fresh proxy cohort refuses existing shared proxy evidence" >&2
      exit 2
    fi
    run_name="hpo-${MODE}-${RUN_SLOT}-runpod-$(date -u +%Y%m%d-%H%M%S)"
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

if [[ "${MODE}" == "full_acronym_soup" || "${MODE}" == "no_centaur" ]]; then
  [[ -f "${shared_root}/proxy-evidence.json" ]] || {
    echo "run MODE=proxy-cohort successfully before launching ${MODE}" >&2
    exit 2
  }
fi

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"
export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_RUNPOD_JOB_ROOT="${job_root}"
export EDULLM_RUNPOD_SHARED_ROOT="${shared_root}"
export EDULLM_DATASET_ID="pretrain/regmix-10b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_DATASET_TOKENIZER="tokenizer/dolma2-bpe"
export EDULLM_CHECKPOINT_DIR="${job_root}"
export WANDB_PROJECT="hpo-probe"
export WANDB_MODE=online
export PYTHONDONTWRITEBYTECODE=1

entrypoint="${REPO_DIR}/.edullm/runpod/entrypoint.py"
case "${MODE}" in
  proxy-cohort)
    args=(
      "${EDULLM_RUN_ID}"
      --run-proxy-cohort
      --proxy-spec "${REPO_DIR}/.edullm/hpo-full-acronym-soup.json"
      --reference-spec "${REPO_DIR}/.edullm/hpo-no-proxy.json"
      --checkpoint-root "${job_root}"
      --param-dtype bfloat16
    )
    ;;
  *)
    controller_spec="${CONTROLLER_SPEC:-${REPO_DIR}/.edullm/hpo-${MODE//_/-}.json}"
    [[ -f "${controller_spec}" ]] || {
      echo "missing controller spec: ${controller_spec}" >&2
      exit 2
    }
    args=(
      "${EDULLM_RUN_ID}"
      --controller-spec "${controller_spec}"
      --checkpoint-root "${job_root}"
      --param-dtype bfloat16
    )
    ;;
esac

echo "Launching ${MODE}/${RUN_SLOT}; hard wall-time limit=${HARD_TIME_LIMIT}"
elapsed="$(($(date +%s) - launch_started_at))"
remaining="$((HARD_LIMIT_SECONDS - elapsed))"
[[ ${remaining} -gt 0 ]] || { echo "no time remains under the hard limit" >&2; exit 124; }
set +e
timeout --signal=TERM --kill-after=60s "${remaining}s" \
  python3 "${entrypoint}" "${args[@]}" 2>&1 | tee -a "${job_root}/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${status}" > "${job_root}/last-exit-code"
if [[ ${status} -eq 0 ]]; then
  elapsed="$(($(date +%s) - launch_started_at))"
  remaining="$((HARD_LIMIT_SECONDS - elapsed))"
  if [[ ${remaining} -le 0 ]]; then
    status=124
  else
    export WANDB_RESUME=must
    set +e
    timeout --signal=TERM --kill-after=60s "${remaining}s" \
      python3 "${REPO_DIR}/.edullm/runpod/publish_outputs.py" \
        --job-root "${job_root}" \
        --mode "${MODE}" \
        --run-id "${EDULLM_RUN_ID}" 2>&1 | tee -a "${job_root}/run.log"
    status=${PIPESTATUS[0]}
    set -e
  fi
  printf '%s\n' "${status}" > "${job_root}/last-exit-code"
fi
if [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
  echo "Run reached the hard ${HARD_TIME_LIMIT} wall-time limit; use RECOVERY_MODE=resume" >&2
fi
exit "${status}"
