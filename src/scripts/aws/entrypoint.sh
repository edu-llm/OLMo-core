#!/bin/bash
#
# torchrun entrypoint for AWS SLURM training jobs.
#
# Invoked by `srun` from train-job.sbatch. Resolves the rendezvous address/port from the
# SLURM allocation and execs the given training command under torchrun. Everything after the
# entrypoint (the training script + its CLI args) is passed through verbatim.
set -euo pipefail

# Rendezvous coordinates for (multi-node) torchrun.
MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)"
export MASTER_ADDR
export MASTER_PORT=$((60000 + ${SLURM_JOB_ID:-0} % 5000))

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OLMO_RICH_LOGGING="${OLMO_RICH_LOGGING:-1}"
ulimit -n 65536 || true

exec torchrun \
    --nnodes="${SLURM_NNODES:-1}" \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-1}" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="${SLURM_NODEID:-0}" \
    "$@"
