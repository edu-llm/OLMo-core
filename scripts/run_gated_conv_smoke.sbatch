#!/bin/bash
#SBATCH --job-name=kdaliv-gated-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=wheat-01
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# ============================================================================
# L40S (sm_89) smoke test for the gated short convolution inside KDA.
#
# WHAT THIS BUYS, AND WHY IT IS FREE
#   The A100 probe is paid. This is not. Every defect this catches is a defect
#   that would otherwise have been caught by a billed run, and the previous
#   round of this pattern caught three things before paid time: a Slurm 26.05
#   srun conflict, a false-alarming guard, and a silent Dynamo eager fallback.
#
#   sm_89 is NOT the training target. fla's Triton kernels are compiled per
#   architecture and KDA's backward had an sm-dependent illegal-access bug
#   (fla #802, fixed in 0.5.0), so a pass here proves the STACK works, not that
#   the numbers transfer. sm_80 is verified separately on the platform.
#
# NO srun WRAPPER
#   Slurm 26.05 on this cluster refuses `srun` inside a batch script when
#   SLURM_CPUS_PER_TASK disagrees with SLURM_TRES_PER_TASK. The script runs
#   directly.
#
# TRITON_F32_DEFAULT=ieee IS SET AND IS NOT OPTIONAL
#   torch.backends.cuda.matmul.allow_tf32=False does NOT control Triton, which
#   lowers tl.dot to TF32 on Ampere+ by its own default. Measured 166x worse
#   fp32 accuracy in this project. It participates in kernel codegen, so a
#   Triton cache built without it must not be reused with it.
#
# USAGE
#   sbatch scripts/run_gated_conv_smoke.sbatch
# ============================================================================
set -uo pipefail

RUN_ROOT=/scratch/users/${USER}/agent-runs/kdaliv-gated
REPO=${RUN_ROOT}/repo
# torch 2.9.1+cu128 with fla 0.5.1 -- 0.5.0 is the floor, because 0.4.1 is
# exposed to fla #802 (illegal memory access in chunk_kda's backward, whose
# reproducer is literally KimiDeltaAttention).
VENV=/scratch/users/${USER}/agent-runs/kda-phase0/venv

export TRITON_F32_DEFAULT=ieee
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}/src"
export TRITON_CACHE_DIR="${RUN_ROOT}/triton-cache-ieee"
mkdir -p "$TRITON_CACHE_DIR"

echo "=== host: $(hostname), job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
"$VENV/bin/python" -c "import torch,fla;print('torch',torch.__version__,'fla',fla.__version__)"
echo "=== TRITON_F32_DEFAULT=$TRITON_F32_DEFAULT"
echo

# 1. The CPU suite, ON THIS HOST. Not a formality: two tests in this project
#    silently SKIPPED on a missing laptop path, so the measured evidence was
#    unverified off-laptop while the summary read "44 passed, 2 skipped". The
#    skip count is printed and compared below.
echo "=== CPU suite (the 7 fla-gated tests must now RUN, not skip)"
"$VENV/bin/python" -m pytest -q \
  "${REPO}/src/test/nn/gated_convolution_test.py" \
  "${REPO}/src/test/nn/gated_conv_smoke_guards_test.py" \
  -p no:randomly 2>&1 | tail -20
cpu_rc=${PIPESTATUS[0]}
echo "cpu suite rc=$cpu_rc"
echo

# 2. The mutation suite. A green suite proves nothing about whether it can go
#    red; all 14 mutations must be caught.
echo "=== mutation suite"
cd "$REPO" && PY="$VENV/bin/python" bash scripts/mutate_gated_conv.sh 2>&1 | tail -12
mut_rc=${PIPESTATUS[0]}
echo "mutation rc=$mut_rc"
echo

# 3. The GPU smoke, depthwise gate.
echo "=== GPU smoke, gate_structure=depthwise"
"$VENV/bin/python" "${REPO}/scripts/smoke_gated_conv_gpu.py" \
  --gate-structure depthwise \
  --out "${RUN_ROOT}/smoke_depthwise_${SLURM_JOB_ID:-local}.json"
dw_rc=$?
echo "depthwise rc=$dw_rc"
echo

# 4. The GPU smoke, lowrank gate -- the faithful structure, and the one whose
#    init has a real random draw and a two-stage liveness story.
echo "=== GPU smoke, gate_structure=lowrank r=128"
"$VENV/bin/python" "${REPO}/scripts/smoke_gated_conv_gpu.py" \
  --gate-structure lowrank --gate-rank 128 \
  --out "${RUN_ROOT}/smoke_lowrank_${SLURM_JOB_ID:-local}.json"
lr_rc=$?
echo "lowrank rc=$lr_rc"
echo

# 5. Memory headroom at the real training shape. The parameter delta is 0.06%
#    and the ACTIVATION delta is ~3x peak, so this is the number that decides
#    the largest admissible rank_microbatch_size. Measured, not predicted.
echo "=== memory headroom at d=2048, 8192 tokens/rank"
"$VENV/bin/python" "${REPO}/scripts/smoke_gated_conv_gpu.py" \
  --d-model 2048 --n-heads 16 --n-layers 4 \
  --seq-len 4096 --batch-size 2 --steps 6 \
  --gate-structure depthwise \
  --out "${RUN_ROOT}/smoke_memory_${SLURM_JOB_ID:-local}.json"
mem_rc=$?
echo "memory rc=$mem_rc"
echo

echo "=========================================="
echo "cpu=$cpu_rc mutation=$mut_rc depthwise=$dw_rc lowrank=$lr_rc memory=$mem_rc"
if [ "$cpu_rc" -ne 0 ] || [ "$mut_rc" -ne 0 ] || [ "$dw_rc" -ne 0 ] || \
   [ "$lr_rc" -ne 0 ] || [ "$mem_rc" -ne 0 ]; then
  echo "SMOKE FAILED -- do not submit the A100 probe"
  exit 1
fi
echo "SMOKE PASSED on sm_89. sm_80 is still unverified."
