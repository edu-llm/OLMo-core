#!/bin/bash
# FarmShare GPU job for the multiply-free ternary MoE kernel. UNRUN as of authoring.
#
# L40S, sm_89 -- NOT A100 sm_80, NOT sm_90. Correctness is architecture-independent; nothing
# timed here is an A100 number.
#
# Submit from the FarmShare login node (rice-01 has NO GPU, so this must go through Slurm):
#   sbatch /scratch/users/ericrcwu/maple-verify/T-MULFREE/src/src/scripts/maple/mulfree.sbatch
#
#SBATCH --job-name=mulfree
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/users/ericrcwu/maple-verify/T-MULFREE/mulfree_%j.out

set -uo pipefail

V=/scratch/users/ericrcwu/venvs/olmo-gdn
MINE=/scratch/users/ericrcwu/maple-verify/T-MULFREE/src

# MANDATORY. torch's TF32 flag does NOT control Triton; this has been worth ~166x in fp32
# accuracy in this project. Exported HERE, inside the payload, and each script re-asserts it from
# os.environ -- an export that does not reach the payload looks identical to success in the log.
export TRITON_F32_DEFAULT=ieee
export PYTHONPATH="$MINE/src"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "=== TRITON_F32_DEFAULT=$TRITON_F32_DEFAULT"

# Correctness FIRST and gate on it. A speed number on an incorrect kernel is worse than none.
# Capture python's status directly -- `echo "EXIT=$?"` masks it and produced a false
# "COMPLETED 0:0" in this project once already (D-104/D-105).
for RUNG in M7B M20; do
  echo "############ CORRECTNESS $RUNG"
  "$V/bin/python" "$MINE/src/scripts/maple/mulfree_correctness.py" --rung "$RUNG"
  rc=$?
  echo "CORRECTNESS_${RUNG}_RC=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: correctness failed for $RUNG (rc=$rc). NOT running the benchmark."
    exit "$rc"
  fi
done

for RUNG in M7B M20; do
  echo "############ BENCH $RUNG"
  "$V/bin/python" "$MINE/src/scripts/maple/mulfree_bench.py" --rung "$RUNG" --l2-multiple 4.0
  echo "BENCH_${RUNG}_RC=$?"
done
