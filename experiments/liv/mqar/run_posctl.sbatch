#!/bin/bash
#SBATCH -J mqar_posctl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/users/ericrcwu/liv/mqar/posctl_%j.out
#SBATCH -e /scratch/users/ericrcwu/liv/mqar/posctl_%j.err
set -euo pipefail
cd /scratch/users/ericrcwu/liv/mqar
nvidia-smi --query-gpu=name --format=csv,noheader
/scratch/users/ericrcwu/kda/venv/bin/python mqar_positive_control.py \
  --steps 8000 --batch-size 64 \
  --out /scratch/users/ericrcwu/liv/mqar/mqar_positive_control.json
