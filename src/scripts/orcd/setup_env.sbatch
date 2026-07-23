#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -J edullm-env
#SBATCH -o %x-%j.log
set -euo pipefail

: "${EDULLM_REPO_ROOT:?export EDULLM_REPO_ROOT before sbatch}"
: "${EDULLM_COMMIT_SHA:?export EDULLM_COMMIT_SHA before sbatch}"
: "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"
EDULLM_VENV="${EDULLM_VENV:-$HOME/venvs/edullm}"

module load miniforge/24.3.0-0
test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"
test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"
mkdir -p "$EDULLM_SCRATCH"
python -m venv "$EDULLM_VENV"
source "$EDULLM_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools uv
uv pip install torch --torch-backend=cu128
uv pip install "${EDULLM_REPO_ROOT}[wandb]"
python -c "import torch, wandb, olmo_core; print(torch.__version__)"
