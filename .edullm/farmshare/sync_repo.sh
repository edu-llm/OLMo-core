#!/usr/bin/env bash
# Sync a local OLMo-core worktree to FarmShare scratch (no secrets).
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${LOCAL_REPO:?}"
: "${SOCK:?}"
: "${HOST:?}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${RUN_DIR}/OLMo-core' '${RUN_DIR}/scripts' '${RUN_DIR}/logs' && chmod 700 '${RUN_DIR}'"

tar -C "${LOCAL_REPO}" -czf - pyproject.toml src .edullm | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/OLMo-core'"

tar -C "${LOCAL_REPO}/.edullm/farmshare" -czf - . | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/scripts'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "find '${RUN_DIR}/scripts' -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r$//' {} + && \
   chmod +x '${RUN_DIR}/scripts'/*.sh 2>/dev/null || true"

echo "sync_ok run_dir=${RUN_DIR}"
