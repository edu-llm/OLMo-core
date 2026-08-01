#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -J liv-throughput-measure
#SBATCH -o /scratch/users/ericrcwu/liv/throughput2_%j.log
set -euo pipefail
export PYTHONNOUSERSITE=1
export TRITON_F32_DEFAULT=ieee
cd /scratch/users/ericrcwu/kda/olmo
exec /scratch/users/ericrcwu/kda/venv/bin/python /scratch/users/ericrcwu/liv/throughput.py
