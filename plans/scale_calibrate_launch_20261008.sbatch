#!/bin/bash
#SBATCH --job-name=p4-scale-acquisition-grid
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=1-3%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --exclude=wheat-01
set -euo pipefail
export OMP_NUM_THREADS=4
"$SPACING_VENV/bin/python" -u "$SPACING_OPERATOR" --config "$SPACING_OPERATOR_CONFIG" calibrate --index "$SLURM_ARRAY_TASK_ID"
