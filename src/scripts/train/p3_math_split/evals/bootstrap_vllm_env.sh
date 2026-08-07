#!/usr/bin/env bash
set -euo pipefail

# Keep the CUDA-12 vLLM wheel isolated from the DLAMI's CUDA-13 Python stack.
PYTHON_VERSION="3.12.13"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${P3_OLMO_CORE_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
WORK_ROOT="${P3_WORK_ROOT:-/mnt/work}"
VENV="${P3_VENV:-${WORK_ROOT}/p3-vllm-venv}"

export HOME="${HOME:-/root}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK_ROOT}/uv-cache}"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"
mkdir -p "${WORK_ROOT}" "${UV_CACHE_DIR}" "${HF_HOME}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install "${PYTHON_VERSION}"
uv venv --clear --python "${PYTHON_VERSION}" "${VENV}"

# The checkpoint exporter resolves the vendored Qwen2.5 tokenizer through the
# eduLLM dataset reader, so the evaluator process needs this package too. Pin the
# image reader commit (package 0.5.0) that verified the v3 corpus.
EDULLM_DATA_PIN="git+https://github.com/edu-llm/edullm-data.git@38bf831a6c3f445e394784018441fd59288b876c"

PYTHON="${VENV}/bin/python"
uv pip install --python "${PYTHON}" \
  --editable "${REPO_ROOT}[transformers]" \
  "torch==2.10.0" \
  "transformers==5.7.0" \
  "vllm==0.19.1" \
  "opencv-python-headless==4.13.0.92" \
  "torch-c-dlpack-ext==0.1.5" \
  "boto3==1.42.95" \
  "edullm-data @ ${EDULLM_DATA_PIN}"

uv pip check --python "${PYTHON}"
"${PYTHON}" "${SCRIPT_DIR}/preflight_vllm.py"

printf '\nP3 vLLM environment is ready.\nUse this interpreter for export and eval:\n  %s\n' "${PYTHON}"
