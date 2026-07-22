#!/bin/bash
#SBATCH --job-name=olmo-train
#SBATCH --output=logs/%x-%j.log
#SBATCH --nodes=1
#
# Generic OLMo-core training job for an AWS SLURM cluster.
#
# This is a TEMPLATE. It is intentionally cluster-agnostic; finalize the marked sections once
# the real compute channel is known (self-managed SLURM box, AWS ParallelCluster, or an
# enroot/pyxis container setup). Submit it via ./submit.sh, which exports the OLMO_* env vars.
#
# Overrides for the training script are passed through as positional args ("$@").
set -euo pipefail

# ---- Inputs (exported by submit.sh) -------------------------------------------------------
: "${OLMO_BRANCH:?set OLMO_BRANCH (git branch/commit to train)}"
: "${OLMO_TRAIN_SCRIPT:?set OLMO_TRAIN_SCRIPT (e.g. OLMo2/OLMo2-190M.py)}"
: "${OLMO_RUN_NAME:?set OLMO_RUN_NAME (unique run name)}"
: "${OLMO_CHECKPOINT_S3:?set OLMO_CHECKPOINT_S3 (e.g. s3://<project>-checkpoints)}"
REPO_URL="${OLMO_REPO_URL:-https://github.com/edu-llm/OLMo-core.git}"
CLUSTER_LABEL="${OLMO_CLUSTER_LABEL:-aws}"

# ---- Per-job checkout: isolates each hypothesis group's branch ----------------------------
WORKDIR="${OLMO_WORKROOT:-$HOME/olmo-runs}/${OLMO_RUN_NAME}-${SLURM_JOB_ID:-manual}"
mkdir -p "$WORKDIR" logs
git clone --depth 1 --branch "$OLMO_BRANCH" "$REPO_URL" "$WORKDIR/OLMo-core"
cd "$WORKDIR/OLMo-core"

# ---- Environment ---------------------------------------------------------------------------
# FINALIZE ME: choose ONE of the following once the cluster type is confirmed.
#
#   Option A (recommended) — prebuilt OLMo-core container from ECR (deps baked in):
#     module load ...            # or rely on enroot/pyxis
#     srun --container-image="$OLMO_ECR_IMAGE" --container-mounts="$PWD:/app/olmo-core" ...
#
#   Option B — shared virtualenv that already has the heavy deps installed:
source "${OLMO_VENV:-$HOME/olmo-venv}/bin/activate"
pip install -e . --no-deps   # install just this branch's olmo_core, deps already present

# ---- Launch --------------------------------------------------------------------------------
srun --kill-on-bad-exit=1 \
    "$PWD/src/scripts/aws/entrypoint.sh" \
    "src/scripts/train/${OLMO_TRAIN_SCRIPT}" train "$OLMO_RUN_NAME" "$CLUSTER_LABEL" \
    --trainer.save_folder="${OLMO_CHECKPOINT_S3%/}/${OLMO_RUN_NAME}" \
    "$@"
