#!/bin/bash
#SBATCH -J p1_verify
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o /scratch/users/ericrcwu/liv/probes/p1_verify_%j.out
#SBATCH -e /scratch/users/ericrcwu/liv/probes/p1_verify_%j.err
set -euo pipefail
cd /scratch/users/ericrcwu/liv/probes
nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader
/scratch/users/ericrcwu/kda/venv/bin/python p1_verify.py
