#!/usr/bin/env bash
# Pre-campaign smoke: a SHORT run of the real production path on 4xL40S.
#
# This deliberately does NOT use --local-smoke. That flag disables exactly
# the machinery most worth testing -- it turns off checkpointing, the
# 20-label task-loss evaluation, the W&B callback, the $HOME path guard and
# the distributed-runtime assertion -- so a run under it proves only that
# the model builds and a step executes. The smoke instead runs in full
# production mode and shortens the run with --length-tokens.
#
# 250 steps is the minimum that exercises the whole contract: the permanent
# ladder becomes {0, 125, 250}, which is a pre-train checkpoint, one
# INTERMEDIATE checkpoint (so the hard-stop/resume cycle in train_job.sbatch
# actually runs), and a true final one. At 125 steps the ladder would be
# {0, 125} with no intermediate, and the restart path would go untested.
#
# It submits the real train_job.sbatch rather than a copy, so the launcher
# under test is the launcher the campaign uses.
#
# Verify the result with assert_smoke.py afterwards; a green Slurm state is
# not evidence that the contract held.
#
# Env (all overridable):
#   SMOKE_STEPS        default 250
#   PACING/METRIC      default linear_n10 / mtld -- a curriculum arm, since
#                      the pool machinery is the novel part; control alone
#                      would not exercise it
#   LR_SCHEDULE        default constant (condition A, the primary)
#   SEED               default 0, reserved for smokes so a smoke can never
#                      be mistaken for a campaign replicate
#   TRAIN_TIME         default 04:00:00
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE_STEPS="${SMOKE_STEPS:-250}"
GLOBAL_BATCH_TOKENS=4194304
if (( SMOKE_STEPS % 125 )); then
  echo "submit_smoke: SMOKE_STEPS must be a multiple of 125 so the permanent" >&2
  echo "  ladder lands on it exactly (got ${SMOKE_STEPS})" >&2
  exit 2
fi
if (( SMOKE_STEPS < 250 )); then
  echo "submit_smoke: SMOKE_STEPS < 250 leaves no intermediate checkpoint, so" >&2
  echo "  the hard-stop/resume cycle would go untested" >&2
  exit 2
fi

export LENGTH_TOKENS="$(( SMOKE_STEPS * GLOBAL_BATCH_TOKENS ))"
# The point of the smoke is partly to prove retention works, and the ladder
# is what assert_smoke.py inspects.
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-all}"
export PACING="${PACING:-linear_n10}"
export METRIC="${METRIC:-mtld}"
export LR_SCHEDULE="${LR_SCHEDULE:-constant}"
export SEED="${SEED:-0}"
export TRAIN_TIME="${TRAIN_TIME:-04:00:00}"
# Keep a shakedown out of the campaign's project while still exercising the
# real W&B path (see WANDB_SMOKE_PROJECT_NAME in curriculum_entrypoint.py).
export EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-curriculum-new-smoke}"

echo "smoke: ${SMOKE_STEPS} steps (${LENGTH_TOKENS} tokens), ladder {0, 125, ..., ${SMOKE_STEPS}}"
echo "smoke: arm=${PACING}/${METRIC}/${LR_SCHEDULE} seed=${SEED} keep=${KEEP_CHECKPOINTS}"
echo "smoke: wandb project=${EDULLM_WANDB_PROJECT}"

bash "${SCRIPT_DIR}/submit_from_laptop.sh"

cat <<'NEXT'

Submitted. When it finishes, do NOT read the Slurm state as a pass -- run:

  python3 <RUN_DIR>/OLMo-core/.edullm/farmshare/assert_smoke.py <RUN_DIR>

NEXT
