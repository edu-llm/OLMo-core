#!/usr/bin/env bash
# One-time ORCD setup for the eval team's tutor_cat pipeline (pedagogy CAT).
#
# Separate conda env from p7post ON PURPOSE: tutor_cat's `gen` extra pulls vLLM, which pins its
# own torch build. Installing that on top of the training env would resolve p7post's pinned
# torch/transformers out from under the training and eval code we already validated.
#
# Run ON THE LOGIN NODE (needs network; compute nodes have none):
#     bash clusters/orcd/setup_tutorcat.sh 2>&1 | tee ~/tutorcat_setup.log
set -uo pipefail

ENV_NAME="${ENV_NAME:-tutorcat}"
REPO="${REPO:-$HOME/olmo-eval-full}"
PKG="$REPO/eduLLM-Evals"
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"

die() { echo "FATAL: $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH"
[ -d "$PKG" ] || die "$PKG missing. Clone first: git clone --depth 1 --branch AdaptiveEvals https://github.com/edu-llm/olmo-eval-full.git ~/olmo-eval-full"
# shellcheck disable=SC1090
source "$CONDA_SH"

step "conda env $ENV_NAME (python 3.11; tutor_cat requires >=3.10)"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "exists, reusing"
else
    conda create -y -n "$ENV_NAME" python=3.11 || die "conda create failed"
fi
conda activate "$ENV_NAME" || die "cannot activate $ENV_NAME"
python --version

step "tutor_cat core + dev + irt (CPU)"
cd "$PKG" || die "cd $PKG"
pip install -q --upgrade pip
pip install -e ".[dev,irt]" || die "core install failed"

step "response-generation extra (vLLM + torch — this is the slow one)"
# Not fatal: the CAT engine, judge client and MIRT math are all CPU-only. If vLLM fails to
# resolve we can still grade cached responses; only local generation would be blocked.
pip install -e ".[gen]" || echo "[warn] gen extra failed — generation on this env will not work"

step "verify"
python -c "import tutor_cat, numpy, yaml; print('tutor_cat import OK')" || die "tutor_cat not importable"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "[warn] vllm not importable"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || echo "[warn] torch not importable"
command -v tutor-cat >/dev/null && echo "tutor-cat CLI on PATH" || echo "[warn] tutor-cat CLI missing"

step "dataset self-check"
tutor-cat validate || echo "[warn] 'tutor-cat validate' non-zero — inspect before running"

step "offline tests"
python -m pytest tests -q 2>&1 | tail -15 || echo "[warn] tests non-zero"

echo
echo "=== DONE: conda activate $ENV_NAME ==="
