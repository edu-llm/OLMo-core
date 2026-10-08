#!/bin/bash
#SBATCH --job-name=p4-spacing-calibration
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=22:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --exclude=wheat-01
#SBATCH --output=slurm-%x-%j.log

set -euo pipefail
: "${SPACING_ROOT:?Path to the pushed repository snapshot}"
: "${SPACING_VENV:?Path to the experiment virtual environment}"
: "${SPACING_CODE_COMMIT:?Full commit pushed before this submission}"
export PYTHONPATH="$SPACING_ROOT/experiments/spacing_rerun${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled
cd "$SPACING_ROOT"

# The batch shell receives the Slurm warning; forward it so Python checkpoints.
child_pid=""
forward_signal() { if [[ -n "$child_pid" ]]; then kill -USR1 "$child_pid" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$SPACING_VENV/bin/python" -u -m spacing_rerun "$@" &
child_pid=$!
set +e
wait "$child_pid"
result=$?
# A trapped signal can interrupt wait while Python finishes its checkpoint.
while kill -0 "$child_pid" 2>/dev/null; do
  wait "$child_pid"
  result=$?
done
exit "$result"
