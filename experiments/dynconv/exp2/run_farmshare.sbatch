#!/bin/bash
#SBATCH --job-name=exp2-mqar
#SBATCH --partition=normal
#SBATCH --exclude=wheat-01,oat-06
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/%u/exp2/exp2-%j.out
#SBATCH --error=/scratch/users/%u/exp2/exp2-%j.err

# Exp-2 MQAR harness on FarmShare. Sub-agent B's entry point.
#
# WHY THIS FILE EXISTS: the user's laptop is failing and NOTHING may run locally, so every number
# -- including the seconds-per-cell figure needed to price the AWS submission -- has to come from
# here.
#
# ============================================================================================
# CPU PARTITION, NOT GPU, AND THE EXCLUSIONS ARE LOAD-BEARING. Changed by the Exp-2 lead.
# ============================================================================================
# This file originally requested `--partition=gpu --gres=gpu:1`. That is REFUSED for Exp-2:
#
#   1. `oat-06` HOLDS EXP-1'S JOB 1676346, a TIMING benchmark whose validity depends on an
#      uncontended device and stable clocks.
#
#      THE RISK IS CONTENTION, NOT EVICTION -- and the distinction was checked, not assumed.
#      Slurm here reports `PreemptType=(null)`, `PreemptMode=OFF` cluster-wide and
#      `OverSubscribe=NO` on both `normal` and `gpu`, so a job of ours could never have evicted
#      1676346; it would simply have waited for free resources. What it COULD have done is land
#      on the same shared node and compete for memory bandwidth and cache while a benchmark was
#      measuring. For a timing experiment that is arguably worse than eviction, because it
#      corrupts the numbers SILENTLY instead of failing loudly.
#
#      Note `normal` contains Nodes=barley-[01-04],oat-[01-06],rye-[01-02],wheat-[01-08] --
#      oat-06 IS in this partition, so an unconstrained `normal` submission can land there.
#      The exclusion is therefore load-bearing even though we are not asking for a GPU.
#   2. THE WORK DOES NOT NEED A GPU. These models are ~1.1M params at d_model=128. One cell
#      cannot come close to saturating an L40S, so a GPU would sit mostly idle while blocking a
#      scarce resource another experiment is actively using.
#   3. THE SWEEP IS EMBARRASSINGLY PARALLEL -- 240 independent (arm, topology, W, seed) cells
#      with no communication. `barley-04` has 128 idle cores and 384 GB; that suits this shape
#      far better than one card.
#
# `wheat-01` is excluded by STANDING POLICY, permanently, regardless of what `sinfo` reports.
# Preserve BOTH names if you ever add another exclusion.
#
# Verified read-only 2026-08-05: barley-04 (128 CPU / 384 GB) and wheat-[06,08] (80 CPU / 191 GB)
# idle in `normal`; MaxTime 2-00:00:00; /scratch/users has 80 TB free.
#
# THE FAILURE THIS FILE IS WRITTEN AGAINST: job 1670963 carried a STALE sbatch with
# `--steps 3000 --batch-size 32` = 96,000 examples against the calibrated 512,000, a 5.3x
# shortfall, and produced a table that read as "the task is too hard" rather than "we
# under-trained". The budget below is therefore NOT parameterised -- it is written out explicitly,
# and the harness itself refuses anything smaller outside --smoke.
#
#   sbatch run_farmshare.sbatch timing     # step-1: measure seconds-per-cell (SMOKE, not a result)
#   sbatch run_farmshare.sbatch control    # step-2: positive control, easiest rung
#   sbatch run_farmshare.sbatch calibrate  # step-3: difficulty recalibration, BASELINE ONLY
#   sbatch run_farmshare.sbatch sweep      # step-4: the real paired-seed sweep

set -euo pipefail

MODE="${1:-timing}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="/scratch/users/${USER}/exp2"
mkdir -p "$OUTDIR"

# PYTHONNOUSERSITE=1: repo memory `olmo-core-toolchain-gotchas` -- the user-site torchvision
# clashes with the venv torch and produces an import-time failure that looks like a code bug.
export PYTHONNOUSERSITE=1
# Repo memory `triton-f32-default-mandatory`: the torch tf32 flag does NOT control Triton, and
# fp32 accuracy differs by 166x without this. It must reach the TRAINER's env, not just a shell.
export TRITON_F32_DEFAULT=ieee
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  mode ${MODE} ==="
# Refuse to run on the node Exp-1 is benchmarking on, even if Slurm somehow placed us there.
# Belt and braces against the --exclude above: contention on a shared node corrupts a timing
# measurement silently, so the guard that costs nothing is worth keeping.
if [[ "$(hostname -s)" == "oat-06" || "$(hostname -s)" == "wheat-01" ]]; then
  echo "REFUSING: landed on $(hostname -s), which is excluded (oat-06 = Exp-1 job 1676346)." >&2
  exit 3
fi
# DEVICE AND DTYPE ON EVERY NUMBER -- the label is printed here so every artifact carries it.
python3 -c "import torch; print(f'torch {torch.__version__}  device=cpu  dtype=fp32  threads={torch.get_num_threads()}')"

cd "$HERE"

case "$MODE" in
  timing)
    # STEP 1 -- the OUTSTANDING measurement, on CPU. 200 steps is a SMOKE budget: its accuracy
    # numbers are NOT results, only its timing is. Read the per-step rate and multiply by 8,000;
    # per repo memory `measure-throughput-post-startup`, do NOT read whole-run wall clock -- it ran
    # 3.1x low on a 40-step probe and penalises bigger shapes hardest.
    #
    # NOTE: this uses --stub (no dynamic mechanism). arms.py needs `olmo_core`, which was NOT
    # importable where this was authored, so the stub timing is a LOWER BOUND for S2/S4 -- they add
    # a generator the stub does not have. Re-time with the real arms once olmo_core resolves.
    for CFG in "512 64" "512 8"; do
      set -- $CFG
      for TOP in allliv hybrid; do
        echo "--- timing N$1_D$2 / $TOP ---"
        python3 mqar_harness.py --stub --device cpu --smoke \
          --steps 200 --batch-size 64 --seq-len "$1" --num-pairs "$2" \
          --seeds 1 --arms static --topologies "$TOP" \
          --out "$OUTDIR/timing_N$1_D$2_$TOP.jsonl"
      done
    done
    ;;

  control)
    # STEP 2 -- POSITIVE CONTROL FIRST. A sweep whose easiest rung scores zero cannot separate
    # "hard task" from "broken setup" (job 1670922 returned 0.000 everywhere at vocab 8192).
    # calibration.py aborts the sweep with exit code 3 if this fails.
    python3 calibration.py --device cpu --topologies allliv hybrid \
      --seeds 5 --out "$OUTDIR/control.jsonl" --evidence-only
    echo "NOTE: --evidence-only prints the recorded evidence without compute; drop it to run."
    ;;

  calibrate)
    # STEP 3 -- difficulty recalibration, BASELINE (S1) ONLY, at the full calibrated budget.
    # 8000 x 64 = 512,000 examples. Do NOT add --smoke here.
    python3 calibration.py --device cpu --topologies allliv hybrid \
      --steps 8000 --batch-size 64 --seeds 5 \
      --out "$OUTDIR/calibration.jsonl"
    ;;

  sweep)
    # STEP 4 -- the real sweep. Paired seeds; data order shared across arms within a pair.
    # W=2 is the FALSIFICATION control: a W=2 dynamic-vs-static difference beyond seed noise is a
    # bug, not a result.
    python3 mqar_harness.py --device cpu \
      --steps 8000 --batch-size 64 --seeds 10 \
      --seq-len 512 --num-pairs 64 \
      --arms static permuted dynamic --topologies allliv \
      --kernel-sizes 2 3 4 8 \
      --out "$OUTDIR/sweep_allliv.jsonl"
    ;;

  *)
    echo "unknown mode '$MODE' (want: timing | control | calibrate | sweep)" >&2
    exit 2
    ;;
esac

echo "=== done: $OUTDIR ==="
