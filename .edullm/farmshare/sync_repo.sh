#!/usr/bin/env bash
# Sync a local OLMo-core worktree to FarmShare scratch (no secrets).
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${LOCAL_REPO:?}"
: "${SOCK:?}"
: "${HOST:?}"

python3 - "${LOCAL_REPO}/.edullm/farmshare" <<'PY' || true
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in root.rglob("*"):
    if path.is_file():
        data = path.read_bytes()
        if b"\r" in data:
            path.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${RUN_DIR}/OLMo-core' '${RUN_DIR}/scripts' '${RUN_DIR}/logs' && chmod 700 '${RUN_DIR}'"

tar -C "${LOCAL_REPO}" -czf - pyproject.toml src .edullm | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/OLMo-core'"

tar -C "${LOCAL_REPO}/.edullm/farmshare" -czf - . | \
  ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
    "tar -xzf - -C '${RUN_DIR}/scripts'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "find '${RUN_DIR}/scripts' -type f \( -name '*.sh' -o -name '*.sbatch' -o -name 'config.env' \) -exec sed -i 's/\r$//' {} + && \
   chmod +x '${RUN_DIR}/scripts'/*.sh 2>/dev/null || true"

echo "sync_ok run_dir=${RUN_DIR}"
