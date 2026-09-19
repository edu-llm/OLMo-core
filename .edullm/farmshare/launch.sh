#!/usr/bin/env bash
# Launch one token-selection arm after staging completes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

ARM="${ARM:-attention}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"

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
ARM="${ARM}" "${PYTHON}" -c \
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
    run_name="token-selection-${ARM}-farmshare-$(date -u +%Y%m%d-%H%M%S)"
    wandb_id="$("${PYTHON}" -c 'import secrets; print(secrets.token_hex(16))')"
    printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
      "${run_name}" "${wandb_id}" > "${identity_file}"
    export WANDB_RESUME=never
    recovery=()
    ;;
  resume)
    [[ -f "${identity_file}" ]] || {
      echo "resume requires ${identity_file}" >&2
      exit 2
    }
    export WANDB_RESUME=must
    recovery=(--resume)
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
    rm -f \
      "${arm_root}/checkpoints/run_fingerprint.json" \
      "${arm_root}/progress/run_identity.json"
    export WANDB_RESUME=allow
    recovery=()
    ;;
  *)
    echo "RECOVERY_MODE must be fresh, retry-start, or resume" >&2
    exit 2
  ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export EDULLM_RUNPOD_INPUT_MANIFEST="${INPUT_MANIFEST}"
export EDULLM_WANDB_PROJECT="token-selection-${ARM}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"
export WANDB_MODE=online

# The platform entrypoint asserts an 8-GPU torchrun topology unless --local
# is set. FarmShare's default 8x L40S node satisfies that; smaller
# allocations (e.g. a 4-GPU control-arm run) need the assertion bypassed.
#
# This is set via EDULLM_LOCAL=1, not a literal "--local" token on the
# torchrun command line: torchrun's own argparse (parse_args, not
# parse_known_args) scans the entire argv for abbreviation matches against
# its own flags regardless of position, and "--local" ambiguously matches
# its own --local-addr / --local-ranks-filter, so torchrun itself refuses to
# start if it's passed there. token_selection_entrypoint.main() reads
# EDULLM_LOCAL as an equivalent to --local.
if [[ "${TRAIN_GPUS}" != "8" ]]; then
  export EDULLM_LOCAL=1
  # production=False (via EDULLM_LOCAL) makes recipe.py leave task_loss_nproc
  # unset, which defaults the eval subprocess to a bare single process. That
  # deadlocks against this still-live multi-GPU trainer's own process group
  # on the same GPUs. Route eval through the same multi-process
  # torch.distributed.run wrapper production always uses instead.
  export TASK_LOSS_NPROC="${TRAIN_GPUS}"
fi

exec "${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node="${TRAIN_GPUS}" \
  "${REPO_DIR}/.edullm/runpod/entrypoint.py" \
  --arm "${ARM}" \
  --save-folder "${arm_root}/checkpoints" \
  --work-dir "${arm_root}/work" \
  --progress-dir "${arm_root}/progress" \
  --task-loss-script "${REPO_DIR}/.edullm/eval_task_loss_olmo_core.py" \
  "${recovery[@]}"
