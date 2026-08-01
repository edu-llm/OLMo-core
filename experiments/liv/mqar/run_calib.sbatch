#!/bin/bash
#SBATCH -J mqar_calib
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /scratch/users/ericrcwu/liv/mqar/calib_%j.out
#SBATCH -e /scratch/users/ericrcwu/liv/mqar/calib_%j.err
set -euo pipefail
cd /scratch/users/ericrcwu/liv/mqar
nvidia-smi --query-gpu=name --format=csv,noheader
# Budget MUST match the positive control (8000 x 64 = 512k examples), which is the only
# configuration measured to solve MQAR here. Job 1670963 failed because this file still had
# 3000 x 32 = 96k, i.e. 5.3x less, while the script had the corrected vocab/lr/attention.
/scratch/users/ericrcwu/kda/venv/bin/python mqar_calibrate.py \
  --seeds 5 --steps 8000 --batch-size 64 --grid both \
  --out /scratch/users/ericrcwu/liv/mqar/mqar_calibration.json
