#!/usr/bin/env bash
# Prepare a credential-free RunPod image for the curriculum OLMo-core branch.
set -Eeuo pipefail

readonly BRANCH="edullm/curriculum-370m"
readonly REPOSITORY_URL="https://github.com/edu-llm/OLMo-core.git"
REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
PYTHON="${PYTHON:-python3}"

if [[ -e "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then
  echo "REPO_DIR exists but is not a git checkout: ${REPO_DIR}" >&2
  exit 2
fi
had_repo=0
if [[ -d "${REPO_DIR}/.git" ]]; then
  had_repo=1
fi
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${REPOSITORY_URL}" "${REPO_DIR}"
fi
if [[ ${had_repo} -eq 1 && -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
  echo "refusing to replace a dirty RunPod checkout: ${REPO_DIR}" >&2
  exit 2
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
resolved="$(git -C "${REPO_DIR}" rev-parse FETCH_HEAD)"
if [[ -n "${OLMO_CORE_COMMIT_SHA:-}" && "${resolved}" != "${OLMO_CORE_COMMIT_SHA}" ]]; then
  echo "branch resolved to ${resolved}, not OLMO_CORE_COMMIT_SHA=${OLMO_CORE_COMMIT_SHA}" >&2
  exit 2
fi
git -C "${REPO_DIR}" checkout --detach "${resolved}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc g++ git ca-certificates
rm -rf /var/lib/apt/lists/*
export PIP_BREAK_SYSTEM_PACKAGES=1
"${PYTHON}" -m pip install --quiet --upgrade pip wheel ninja
"${PYTHON}" -m pip uninstall --quiet --yes torch torchvision torchaudio
"${PYTHON}" -m pip install --quiet --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.9.0" "torchvision==0.24.0" "torchaudio==2.9.0"
TORCH_CUDA_ARCH_LIST="9.0" GROUPED_GEMM_CUTLASS="1" MAX_JOBS="${MAX_JOBS:-32}" \
  "${PYTHON}" -m pip install --quiet --no-build-isolation --no-cache-dir \
  "grouped_gemm @ git+https://github.com/tgale96/grouped_gemm.git@f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS="${MAX_JOBS:-32}" \
  "${PYTHON}" -m pip install --quiet --no-build-isolation --no-cache-dir "flash-attn==2.8.2"
"${PYTHON}" -m pip install --quiet --no-cache-dir -e "${REPO_DIR}[wandb]" boto3
"${PYTHON}" -m pip install --quiet --no-cache-dir \
  "edullm-data @ https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"
"${PYTHON}" -m pip install --quiet --no-cache-dir \
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

PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm" "${PYTHON}" - <<'PY'
import torch
import torchaudio
import torchvision
import grouped_gemm
from olmo.eval.downstream import label_to_task_map
from edullm_data.read import dataset_paths
from curriculum_model import MODEL_IDENTITY, build_model_config
from olmo_core.nn.attention import flash_attn_api

assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torchvision.__version__.startswith("0.24.0"), torchvision.__version__
assert torchaudio.__version__.startswith("2.9.0"), torchaudio.__version__
assert torch.version.cuda, "CPU-only torch wheel installed"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert grouped_gemm.ops.gmm is not None
assert flash_attn_api.has_flash_attn_2()
assert "arc_easy_val_rc_5shot_bpb" in label_to_task_map
cfg = build_model_config()
assert cfg.block.feed_forward_moe is not None
assert cfg.block.feed_forward_moe.num_experts == 64
assert cfg.block.feed_forward_moe.router.top_k == 8
assert cfg.d_model == 2048
print("RunPod bootstrap ready:", torch.__version__, torch.version.cuda, dataset_paths.__module__)
print(MODEL_IDENTITY, cfg.__class__.__name__)
PY

mkdir -p /workspace/edullm-bootstrap
printf '%s\n' "${resolved}" > /workspace/edullm-bootstrap/curriculum.commit
echo "BOOTSTRAP_DONE repo=${REPO_DIR} commit=${resolved}"
