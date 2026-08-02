#!/usr/bin/env bash
set -euo pipefail

: "${EDULLM_ARM:?EDULLM_ARM must name a token-selection arm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TASK_LOSS_EVAL_SCRIPT="${TASK_LOSS_EVAL_SCRIPT:-${ROOT}/eval_task_loss_olmo_core.py}"
: "${TASK_LOSS_EVAL_SCRIPT:?TASK_LOSS_EVAL_SCRIPT is required}"

exec python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  .edullm/token_selection_entrypoint.py \
  --arm "${EDULLM_ARM}" \
  "$@"
