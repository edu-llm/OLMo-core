#!/bin/bash
# L40S smoke test for the A100 gate-latency harness.
#
# WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
#   It is NOT the measurement. FarmShare has L40S (sm_89) and no A100, so it cannot answer the
#   question the study asks -- the number has to come from the card the training runs used.
#
#   It is a free rehearsal of everything that only exercises on a GPU: CUDA-graph capture, the
#   NVML clock receipt, the soak, the interleaved rounds, graph replay under `logits_to_keep=1`,
#   and above all whether the A/A control comes back bracketing zero. Every one of those is
#   unreachable from a CPU test, and every one of them can fail in a way that wastes the paid
#   A100 slot. Two of this project's runs have died on paths nothing had ever executed.
#
#   A100 is $10.98 worst case with a 61-minute median queue. This is $0 and a few minutes.
#
# ONE THING THAT DIFFERS HERE AND MATTERS
#   `fla` IS installed in this venv (0.4.1) and is ABSENT from the AWS research image. That is
#   precisely the uncontrolled variable `--use-fla false` exists to pin: with the module default
#   of True, an arm's convolution would fuse here and not there, and the contrast would compare
#   kernels while reading as a verdict on gate structure. So this run pins it false and the
#   harness asserts the realised path -- which makes this host a genuine test of the pin, better
#   than the AWS image where `fla` is missing and the pin is untested because it cannot matter.
#
# Reduced rounds because this is a rehearsal, not the measurement. The A/A width scales with
# rounds, so expect a wider interval here than the A100 run will produce.
#
#SBATCH --job-name=liv-gate-latency-l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=wheat-01
#SBATCH --output=/scratch/users/ericrcwu/lat-harness/l40s_%j.out
#SBATCH --error=/scratch/users/ericrcwu/lat-harness/l40s_%j.err

set -euo pipefail

D=/scratch/users/ericrcwu/lat-harness
cd "$D"

# PYTHONNOUSERSITE guards against the torchvision clash in ~/.local.
export PYTHONNOUSERSITE=1
# PYTHONPATH is load-bearing: without it this venv resolves `olmo_core` to a DIFFERENT
# worktree (agent-runs/olmo-gdn-halflife...), so the run would measure code from another
# experiment and report it as this one's result.
export PYTHONPATH="$D/src"
# Triton ignores torch's tf32 flag; ieee keeps fp32 reductions honest.
export TRITON_F32_DEFAULT=ieee

V=/scratch/users/ericrcwu/venvs/olmo-gdn/bin/python

echo "=== host and card ==="
hostname
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
scontrol show node "$(hostname -s)" | grep -o 'AvailableFeatures=[^ ]*' || true

echo
echo "=== provenance: which olmo_core is this? ==="
$V -c "import olmo_core; print(olmo_core.__file__)"
$V -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
$V -c "import fla; print('fla present:', fla.__version__)" 2>&1 | tail -1

echo
echo "=== the harness ==="
# --device-peak-gbs/-tflops: the L40S is in the harness's tables, so these are redundant and
# passed anyway to prove the override path works before the A100 run needs it.
#
# --rounds 8: a rehearsal. The A/A interval narrows with rounds, so 8 will look worse than the
# A100 run's 20 and that is expected, not a failure of the design.
#
# --allow-unresolvable: set DELIBERATELY, because at 8 rounds the A/A interval will very likely
# exceed the 1.8% target and a non-zero exit would mask any genuine defect further down. The
# verdict is still PRINTED, and reading it is the point of this run.
#
# NO `srun`, DELIBERATELY. Slurm 26.05 aborts when --cpus-per-task and the partition's per-GPU
# CPU default disagree: "cpus-per-task set by two different environment variables
# SLURM_CPUS_PER_TASK=17 != SLURM_TRES_PER_TASK=cpu=8", which killed job 1676750 six seconds in.
# A single-process benchmark gains nothing from a step launcher, so the fix is to drop it rather
# than to negotiate two Slurm variables into agreement. Removing --cpus-per-task too, so the
# partition's own default applies and there is nothing left to conflict.
# `set -e` is relaxed around this one command so the exit code can be REPORTED rather than
# merely inherited. Under `set -e` a non-zero exit would abort the script before the echo, and
# "$?" after an aborted command reads the wrong value -- so the code that says whether a guard
# fired would be the one thing the log did not contain.
set +e
$V .edullm/bench_gate_latency.py "l40s-smoke-${SLURM_JOB_ID}" \
  --rounds 8 \
  --iters-per-round 3 \
  --warmup 5 \
  --soak-seconds 45 \
  --use-fla false \
  --device-peak-gbs 864 \
  --device-peak-tflops 362 \
  --allow-unresolvable \
  --out "$D/gate_latency_l40s_${SLURM_JOB_ID}.json"
RC=$?
set -e

echo
echo "=== harness exit code: $RC ==="
# Non-zero here is INFORMATION, not necessarily failure: it means a guard fired, and which one
# is in the stderr above. Exit 0 regardless so Slurm's state does not overwrite that distinction.
exit 0
