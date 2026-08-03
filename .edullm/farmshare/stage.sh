#!/usr/bin/env bash
# Stage the weighted shard subset required by one MixLaw arm.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config.env"

ARM_INDEX="${ARM_INDEX:-0}"

[[ -f "${AWS_ENV_FILE}" ]] || {
  echo "missing ${AWS_ENV_FILE}; push aws-session.env from the laptop first" >&2
  exit 2
}

bash "${SCRIPT_DIR}/setup_venv.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"

stage_args=(
  "${REPO_DIR}/.edullm/runpod/stage_inputs.py"
  --credentials-file "${AWS_ENV_FILE}"
  --stage-root "${STAGE_ROOT}"
  --workers "${STAGE_WORKERS:-12}"
  --arm-index "${ARM_INDEX}"
  --headroom "${STAGE_HEADROOM:-1.10}"
)
if [[ -n "${LENGTH_TOKENS:-}" ]]; then
  stage_args+=(--length-tokens "${LENGTH_TOKENS}")
fi

"${PYTHON}" "${stage_args[@]}"

if [[ -e "${AWS_ENV_FILE}" ]]; then
  echo "staging finished but ${AWS_ENV_FILE} still exists" >&2
  exit 2
fi
[[ -f "${INPUT_MANIFEST}" ]] || {
  echo "staging did not publish ${INPUT_MANIFEST}" >&2
  exit 2
}
echo "stage_ok manifest=${INPUT_MANIFEST}"
