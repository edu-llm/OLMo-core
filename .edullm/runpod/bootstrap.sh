#!/usr/bin/env bash
# Prepare an 8-GPU RunPod for the local three-arm HPO adapter.
set -Eeuo pipefail

readonly REPOSITORY_URL="https://github.com/edu-llm/OLMo-core.git"
readonly DEFAULT_COMMIT="4f385fe54918b96756042a89d504ac19b928e1b4"
COMMIT_SHA="${OLMO_CORE_COMMIT_SHA:-${DEFAULT_COMMIT}}"
REPO_DIR="${REPO_DIR:-/workspace/OLMo-core}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -e "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then
  echo "REPO_DIR exists but is not a git checkout: ${REPO_DIR}" >&2
  exit 2
fi
new_clone=0
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${REPOSITORY_URL}" "${REPO_DIR}"
  new_clone=1
fi
if [[ "${new_clone}" -eq 0 ]]; then
  dirty="$(
    git -C "${REPO_DIR}" status --porcelain --untracked-files=all |
      grep -vE '^\?\? \.edullm/runpod/' || true
  )"
  if [[ -n "${dirty}" ]]; then
    echo "refusing to replace a dirty RunPod checkout outside .edullm/runpod" >&2
    printf '%s\n' "${dirty}" >&2
    exit 2
  fi
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT_SHA}"
resolved="$(git -C "${REPO_DIR}" rev-parse FETCH_HEAD)"
if [[ "${resolved}" != "${COMMIT_SHA}" ]]; then
  echo "fetched ${resolved}, not requested OLMO_CORE_COMMIT_SHA=${COMMIT_SHA}" >&2
  exit 2
fi
git -C "${REPO_DIR}" checkout --detach "${resolved}"

# These files intentionally remain local and uncommitted. Copy the adapter that was transferred
# to the pod over the clean pinned checkout.
target_adapter="${REPO_DIR}/.edullm/runpod"
mkdir -p "${target_adapter}"
if [[ "$(readlink -f "${SCRIPT_DIR}")" != "$(readlink -f "${target_adapter}")" ]]; then
  cp -a "${SCRIPT_DIR}/." "${target_adapter}/"
fi
chmod 0755 "${target_adapter}/bootstrap.sh" "${target_adapter}/launch.sh"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc g++ git ca-certificates coreutils
rm -rf /var/lib/apt/lists/*

export PIP_BREAK_SYSTEM_PACKAGES=1
"${PYTHON}" -m pip install --quiet --upgrade pip wheel ninja packaging
"${PYTHON}" -m pip uninstall --quiet --yes torch torchvision torchaudio || true
"${PYTHON}" -m pip install --quiet --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.9.0"
TORCH_CUDA_ARCH_LIST="8.0" FLASH_ATTN_CUDA_ARCHS="80" MAX_JOBS="${MAX_JOBS:-32}" \
  "${PYTHON}" -m pip install --quiet --no-build-isolation --no-cache-dir "flash-attn==2.8.2"
"${PYTHON}" -m pip install --quiet --no-cache-dir \
  -e "${REPO_DIR}[wandb,hpo]" \
  boto3 "botocore[crt]"
"${PYTHON}" -m pip install --quiet --no-cache-dir \
  "edullm-data @ https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"

export OLMO_CORE_HPO_ARTIFACT_CACHE="${OLMO_CORE_HPO_ARTIFACT_CACHE:-/workspace/olmo-artifacts}"
PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm" "${PYTHON}" - <<'PY'
import torch
import ifbo
import openai
import unit_scaling
from edullm_data.read import dataset_paths
from olmo_core.hpo.artifacts import ensure_ftpfn_artifact
from olmo_core.nn.attention import flash_attn_api

assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torch.version.cuda, "CPU-only torch wheel installed"
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), index
assert flash_attn_api.has_flash_attn_2()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(8)])
print("FT-PFN:", ensure_ftpfn_artifact())
print("RunPod HPO bootstrap ready:", torch.__version__, torch.version.cuda)
print(ifbo.__name__, openai.__name__, unit_scaling.__name__, dataset_paths.__module__)
PY

mkdir -p /workspace/edullm-bootstrap
printf '%s\n' "${resolved}" > /workspace/edullm-bootstrap/hpo-probe.commit
echo "BOOTSTRAP_DONE repo=${REPO_DIR} commit=${resolved}"
