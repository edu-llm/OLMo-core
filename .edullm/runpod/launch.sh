#!/usr/bin/env bash
# Launch one MixLaw arm after stage_inputs.py has removed the temporary AWS session.
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="${RUN_ROOT:-/workspace/edullm-runs/mixlaw}"
INPUT_MANIFEST="${EDULLM_RUNPOD_INPUT_MANIFEST:-/workspace/edullm-inputs/mixlaw/ready.json}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/workspace/wandb-session.env}"
ARM_INDEX="${ARM_INDEX:-0}"
LENGTH_TOKENS="${LENGTH_TOKENS:-}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"

if [[ ! -f "${INPUT_MANIFEST}" ]]; then
  echo "stage inputs first; missing ${INPUT_MANIFEST}" >&2
  exit 2
fi
if [[ -e "${AWS_ENV_FILE:-/workspace/aws-session.env}" ]]; then
  echo "temporary AWS credential file still exists; refusing training" >&2
  exit 2
fi
for name in \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
  AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE \
  AWS_WEB_IDENTITY_TOKEN_FILE AWS_ROLE_ARN \
  AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI; do
  if [[ -n "${!name:-}" ]]; then
    echo "${name} is present; refusing training after staging" >&2
    exit 2
  fi
done
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required (copy ${WANDB_ENV_FILE} over SSH)" >&2
  exit 2
fi

case "${ARM_INDEX}" in
  0) arm_name="olmo-mix-1124" ;;
  1) arm_name="mix01" ;;
  2) arm_name="ML-pilot_caps" ;;
  3) arm_name="LGB-min1pct" ;;
  *) echo "ARM_INDEX must be 0..3" >&2; exit 2 ;;
esac

arm_root="${RUN_ROOT}/${arm_name}"
checkpoint_dir="${arm_root}/checkpoints"
eval_work_dir="${arm_root}/eval-work"
identity_file="${arm_root}/run.env"
mkdir -p "${checkpoint_dir}" "${eval_work_dir}"
shopt -s nullglob dotglob
checkpoint_entries=("${checkpoint_dir}"/*)
shopt -u nullglob dotglob
case "${RECOVERY_MODE}" in
  fresh)
    if [[ -e "${identity_file}" || ${#checkpoint_entries[@]} -ne 0 ]]; then
      echo "fresh run refuses existing state under ${arm_root}" >&2
      exit 2
    fi
    umask 077
    run_name="mixlaw-${arm_name}-runpod-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    ;;
  resume)
    [[ -f "${identity_file}" ]] || { echo "resume requires ${identity_file}" >&2; exit 2; }
    export WANDB_RESUME=must
    ;;
  retry-start)
    [[ -f "${identity_file}" ]] || { echo "retry-start requires ${identity_file}" >&2; exit 2; }
    shopt -s nullglob
    step_entries=("${checkpoint_dir}"/step*)
    shopt -u nullglob
    [[ ${#step_entries[@]} -eq 0 ]] || {
      echo "retry-start is only valid before the first checkpoint; use resume" >&2
      exit 2
    }
    export WANDB_RESUME=allow
    ;;
  *) echo "RECOVERY_MODE must be fresh, retry-start, or resume" >&2; exit 2 ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"
export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_DATASET_ID="pretrain/olmo-127b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_CHECKPOINT_DIR="${checkpoint_dir}"
export EDULLM_EVAL_WORK_DIR="${eval_work_dir}"
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-mixlaw}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mixlaw-370m-runpod}"

args=(--arm-index "${ARM_INDEX}")
if [[ -n "${LENGTH_TOKENS}" ]]; then
  args+=(--length-tokens "${LENGTH_TOKENS}")
fi

exec python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${REPO_DIR}/.edullm/runpod/entrypoint.py" "${args[@]}"
