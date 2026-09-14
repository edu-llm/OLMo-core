#!/usr/bin/env bash
# Launch one curriculum run directly -- no staging step (plan section 3a:
# the corpus rebuild materializes every metric's parent on scratch once,
# ahead of any training job; there is nothing left to stage per-run).
#
# Rewritten for `curriculum-new`: no AWS, no --arm-index/ARMS lookup (the
# prior frozen 16-arm matrix is gone -- see curriculum_entrypoint.py's
# module docstring). A run is fully specified by PACING, LR_SCHEDULE, SEED,
# and (for non-control pacings) METRIC.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

PACING="${PACING:?PACING is required (control, linear_n10, anti_linear_n10, quadratic_n10, warmup_ramp_1000, warmup_quadratic_n10_1000, interleave_i10_linear)}"
LR_SCHEDULE="${LR_SCHEDULE:?LR_SCHEDULE is required (constant or cosine)}"
SEED="${SEED:?SEED is required}"
RECOVERY_MODE="${RECOVERY_MODE:-fresh}"
LENGTH_TOKENS="${LENGTH_TOKENS:-}"

if [[ "${PACING}" == "control" ]]; then
  METRIC_FOR_PARENT="${CONTROL_METRIC:?CONTROL_METRIC is required for --pacing control}"
  metric_args=(--control-metric "${METRIC_FOR_PARENT}")
else
  METRIC="${METRIC:?METRIC is required for a non-control pacing}"
  METRIC_FOR_PARENT="${METRIC}"
  metric_args=(--metric "${METRIC}")
fi

PARENT_MANIFEST="${PARENT_MANIFEST:-${CORPUS_ROOT}/${METRIC_FOR_PARENT}/manifest.json}"
[[ -f "${PARENT_MANIFEST}" ]] || {
  echo "corpus not rebuilt for metric ${METRIC_FOR_PARENT}: ${PARENT_MANIFEST}" >&2
  exit 2
}

if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
[[ -n "${WANDB_API_KEY:-}" ]] || {
  echo "WANDB_API_KEY is required (${WANDB_ENV_FILE})" >&2
  exit 2
}

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

arm_name="${PACING}"
if [[ "${PACING}" != "control" ]]; then
  arm_name="${arm_name}-${METRIC}"
fi
arm_name="${arm_name}-${LR_SCHEDULE}-seed${SEED}"

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
    run_name="curriculum-new-${arm_name}-farmshare-$(date -u +%Y%m%d-%H%M%S)"
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
  *)
    echo "RECOVERY_MODE must be fresh or resume" >&2
    exit 2
    ;;
esac
# shellcheck disable=SC1090
source "${identity_file}"

export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum-new}"
export WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"

args=(
  --train-worker
  --pacing "${PACING}"
  --lr-schedule "${LR_SCHEDULE}"
  --seed "${SEED}"
  --parent-manifest "${PARENT_MANIFEST}"
  "${metric_args[@]}"
  --nproc "${TRAIN_GPUS}"
  --run-dir "${arm_root}"
  --save-folder "${arm_root}/checkpoints"
  --progress-dir "${arm_root}/progress"
  --cache-dir "${arm_root}/cache"
  --wandb-mode online
  --device-batch-size "${DEVICE_BATCH_SIZE:-8}"
  --task-loss-eval-script "${REPO_DIR}/.edullm/task_loss/eval_task_loss_olmo_core.py"
  --ladder-base-config "${REPO_DIR}/.edullm/task_loss/ladder_base_config.yaml"
  --task-loss-nproc "${TRAIN_GPUS}"
  "${recovery[@]}"
)
if [[ -n "${LENGTH_TOKENS}" ]]; then
  args+=(--length-tokens "${LENGTH_TOKENS}")
fi

exec "${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node="${TRAIN_GPUS}" -- \
  "${REPO_DIR}/.edullm/curriculum_entrypoint.py" "${args[@]}"
