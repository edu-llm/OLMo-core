#!/usr/bin/env bash
# Concrete single-node eight-GPU production launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NPROC=8
export TASK_LOSS_NPROC=8
export TASK_LOSS_EVAL_SCRIPT="${ROOT}/task_loss/eval_task_loss_olmo_core.py"
export LADDER_BASE_CONFIG="${ROOT}/task_loss/ladder_base_config.yaml"

exec "${ROOT}/launch_curriculum_arm.sh" "$@"
