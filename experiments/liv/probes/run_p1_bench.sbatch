#!/bin/bash
#SBATCH -J p1_bench
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o /scratch/users/ericrcwu/liv/probes/p1_bench_%j.out
#SBATCH -e /scratch/users/ericrcwu/liv/probes/p1_bench_%j.err
set -euo pipefail
cd /scratch/users/ericrcwu/liv/probes
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
echo "=== analytic breakeven (reference) ==="
/scratch/users/ericrcwu/kda/venv/bin/python l40s_breakeven.py
echo
echo "=== measured benchmark ==="
/scratch/users/ericrcwu/kda/venv/bin/python p1_launch_bench.py --out p1_bench_results.json
