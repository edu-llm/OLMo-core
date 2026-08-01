#!/bin/bash
#SBATCH -J p1_cachechk
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/users/ericrcwu/liv/probes/cachechk_%j.out
#SBATCH -e /scratch/users/ericrcwu/liv/probes/cachechk_%j.err
set -euo pipefail
cd /scratch/users/ericrcwu/liv/probes
/scratch/users/ericrcwu/kda/venv/bin/python p1_cache_check.py --out cachechk_results.json
