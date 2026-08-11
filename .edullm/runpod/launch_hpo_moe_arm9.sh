#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
RUN_ROOT="/workspace/edullm-runs/hpo-moe"
arm_name="warmup-quadratic10-mtld-256ki"
arm_root="${RUN_ROOT}/${arm_name}"
input_manifest="/workspace/edullm-inputs/curriculum/ready.json"
entrypoint="/workspace/hpo_moe_arm9_entrypoint.py"

source /workspace/wandb-session.env
[[ -s "${input_manifest}" ]] || { echo "missing arm-9 input manifest" >&2; exit 2; }
[[ -x "${entrypoint}" ]] || { echo "missing executable entrypoint: ${entrypoint}" >&2; exit 2; }
[[ ! -e "${arm_root}/run.env" ]] || {
  echo "fresh run already exists: ${arm_root}" >&2
  exit 2
}
for name in \
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
  AWS_PROFILE AWS_DEFAULT_PROFILE AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE; do
  [[ -z "${!name:-}" ]] || { echo "${name} is present; refusing training" >&2; exit 2; }
done
[[ ! -e /workspace/aws-session.env ]] || {
  echo "temporary AWS credential file still exists; refusing training" >&2
  exit 2
}

mkdir -p "${arm_root}"/{checkpoints,progress,cache}
run_name="hpo-moe-warmup-quadratic10-mtld-256ki-$(date -u +%Y%m%d-%H%M%S)"
wandb_id="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
printf "export EDULLM_RUN_ID='%s'\nexport WANDB_RUN_ID='%s'\n" \
  "${run_name}" "${wandb_id}" > "${arm_root}/run.env"
source "${arm_root}/run.env"

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm:${REPO_DIR}/.edullm/runpod"
export EDULLM_RUNPOD_INPUT_MANIFEST="${input_manifest}"
export EDULLM_DATASET_ID="pretrain/opt-with-synthetic-10b"
export EDULLM_DATASET_VERSION="v1"
export EDULLM_WANDB_PROJECT="hpo-moe"
export WANDB_PROJECT="hpo-moe"
export WANDB_RUN_GROUP="curriculum-quadratic-mtld-optimized-256ki"
export WANDB_NAME="${run_name}"
export WANDB_RESUME="never"
export EDULLM_BENCH_REDUCE_BF16=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

args=(
  --train-worker
  --arm-index 9
  --nproc 8
  --device-batch-size 16
  --curriculum-version v1
  --run-dir "${arm_root}"
  --save-folder "${arm_root}/checkpoints"
  --progress-dir "${arm_root}/progress"
  --cache-dir "${arm_root}/cache"
  --wandb-mode online
  --task-loss-eval-script "${REPO_DIR}/.edullm/task_loss/eval_task_loss_olmo_core.py"
  --ladder-base-config "${REPO_DIR}/.edullm/task_loss/ladder_base_config.yaml"
  --task-loss-nproc 8
)
recovery=(--fresh)
restart_request="${arm_root}/progress/restart_after_checkpoint.json"

while true; do
  set +e
  python3 -m torch.distributed.run --standalone --nproc-per-node=8 -- \
    "${entrypoint}" "${args[@]}" "${recovery[@]}"
  status=$?
  set -e
  if [[ ${status} -ne 0 ]]; then
    echo "${status}" > "${arm_root}/last-exit-code"
    exit "${status}"
  fi
  if [[ ! -f "${restart_request}" ]]; then
    echo 0 > "${arm_root}/last-exit-code"
    exit 0
  fi
  durable_step="$(
    python3 -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["durable_step"]))' \
      "${restart_request}"
  )"
  marker_step="$(
    python3 -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["last_durable_step"]))' \
      "${arm_root}/progress/last_durable_step.json"
  )"
  [[ "${durable_step}" == "${marker_step}" ]] || {
    echo "durable marker mismatch" >&2
    exit 2
  }
  python3 "${REPO_DIR}/.edullm/runpod/prune_old_checkpoints.py" "${arm_root}"
  rm -f "${restart_request}"
  export WANDB_RESUME=must
  recovery=(--load-path "${arm_root}/checkpoints")
  echo "Resuming arm 9 from durable step ${durable_step}"
done
