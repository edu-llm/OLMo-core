#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --exclude=wheat-01
#SBATCH --output=/scratch/users/ericrcwu/agent-runs/review-sigma/logs/%x-%A_%a.log
set -u
export ROOT=/scratch/users/ericrcwu/agent-runs/dp2-kda-p0
export KDA_PROBES_DIR=$ROOT/probes
export PYTHONPATH=$ROOT/OLMo-core/src
BASE=/scratch/users/ericrcwu/agent-runs/review-sigma
JOBS=${JOBS_FILE:-$BASE/jobs_main.tsv}
NSHARD=${NSHARD:-6}
IDX=${SLURM_ARRAY_TASK_ID}
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name --format=csv,noheader) SHARD $IDX/$NSHARD JOBS=$JOBS"
n=0
while IFS=$'\t' read -r tag flags; do
  [ -z "$tag" ] && continue
  if [ $(( n % NSHARD )) -eq $IDX ]; then
    echo "=== BEGIN $tag ==="
    $ROOT/venv/bin/python $ROOT/probes/train_probe.py $flags 2>&1 | grep -E "^RESULT|^EVAL_BANK|^FFN_SOLVE|Error|Traceback|error"
    echo "=== END $tag rc=$? ==="
  fi
  n=$((n+1))
done < "$JOBS"
echo "SHARD $IDX DONE"
