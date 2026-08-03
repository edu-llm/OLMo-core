#!/usr/bin/env bash
# Stage curriculum parent + order files for one arm.
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
  --arm-index "${ARM_INDEX}"
  --workers "${STAGE_WORKERS:-12}"
)
if [[ -n "${CURRICULUM_VERSION:-}" ]]; then
  stage_args+=(--curriculum-version "${CURRICULUM_VERSION}")
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
