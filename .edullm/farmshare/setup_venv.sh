#!/usr/bin/env bash
# Create or refresh the per-run venv on FarmShare scratch.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

VENV="${VENV:-${RUN_DIR}/venv}"
PYTHON="${PYTHON:-python3}"
EDULLM_DATA_REF="https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"

if [[ -x "${VENV}/bin/python" ]]; then
  if "${VENV}/bin/python" -c "import torch, olmo_core; assert torch.__version__.startswith('2.9')" 2>/dev/null; then
    echo "venv ready: ${VENV}"
    exit 0
  fi
fi

"${PYTHON}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -q -U pip wheel
pip uninstall -q -y torch torchvision torchaudio 2>/dev/null || true
pip install -q --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0"
pip install -q --no-cache-dir -e "${REPO_DIR}[wandb]" boto3
pip install -q --no-cache-dir "edullm-data @ ${EDULLM_DATA_REF}"
if [[ -n "${EVAL_REQUIREMENTS:-}" && -f "${REPO_DIR}/.edullm/${EVAL_REQUIREMENTS}" ]]; then
  pip install -q --no-cache-dir -r "${REPO_DIR}/.edullm/${EVAL_REQUIREMENTS}"
fi
if [[ "${CURRICULUM_EVAL:-0}" == "1" ]]; then
  pip install -q --no-cache-dir \
    "ai2-olmo @ https://github.com/allenai/OLMo/archive/090253dac6688f2532509daa7aa2eb5fae50e956.tar.gz" \
    "boto3==1.43.60" \
    "datasets==5.0.0" \
    "numpy==1.26.4" \
    "omegaconf==2.3.0" \
    "PyYAML==6.0.3" \
    "safetensors==0.8.0" \
    "scikit-learn==1.9.0" \
    "tokenizers==0.22.2" \
    "torchmetrics==1.9.0" \
    "transformers==4.57.6" \
    "wandb==0.28.1"
fi

PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm" "${VENV}/bin/python" - <<'PY'
import olmo_core
import torch

print("farmshare venv ok", torch.__version__, olmo_core.__file__)
PY
