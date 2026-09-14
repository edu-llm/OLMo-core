#!/usr/bin/env bash
# Push the W&B API key into a FarmShare scratch run directory.
#
# Writes "${RUN_DIR}/wandb-session.env", which common.sh points
# WANDB_ENV_FILE at and launch.sh sources; launch.sh hard-fails when
# WANDB_API_KEY is unset, so no run starts without this.
#
# This replaces a call into a separate repository, which made the submit
# path unable to run from a clean checkout of this one. Behaviour is
# deliberately the same; only the machine-specific key-file guesses are
# gone.
#
# Handling rules, all load-bearing:
#   * the key is never echoed, logged, or passed as a command argument, so
#     it cannot land in the process table or a shell history
#   * it is never written inside the repository -- only to scratch, and via
#     a mode-600 temp file that is removed on exit
#   * only a byte count is reported
#   * it is never committed: the image is built from the commit
#
# Needs ssh and the control socket. Needs no git, so on a machine where git
# and the socket live in different environments this runs alongside
# sync_repo.sh's ship phase.
#
# Env:
#   RUN_DIR          scratch run directory to write into (required)
#   SOCK             ssh control socket (required)
#   HOST             user@host (required)
#   WANDB_API_KEY    the key; preferred, keeps it out of the filesystem
#   WANDB_KEY_FILE   file holding the key on one line, used when the
#                    variable is unset. Defaults to ~/.wandb_api_key.
set -Eeuo pipefail
umask 077

: "${RUN_DIR:?RUN_DIR is required}"
: "${SOCK:?SOCK is required}"
: "${HOST:?HOST is required}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  key_file="${WANDB_KEY_FILE:-${HOME}/.wandb_api_key}"
  if [[ ! -f "${key_file}" ]]; then
    echo "push_wandb_key: no key available." >&2
    echo "  Either export WANDB_API_KEY, or put the key on one line in" >&2
    echo "  ${key_file} (or point WANDB_KEY_FILE at it)." >&2
    exit 2
  fi
  WANDB_API_KEY="$(tr -d ' \t\r\n' < "${key_file}")"
fi

# Refuse anything that would not survive single-quoting into the env file,
# rather than writing a file that sources incorrectly at 3am. W&B keys are
# hex; this is deliberately narrow.
if [[ ! "${WANDB_API_KEY}" =~ ^[A-Za-z0-9_-]{16,128}$ ]]; then
  echo "push_wandb_key: the key does not look like a W&B API key" >&2
  echo "  (expected 16-128 chars of [A-Za-z0-9_-]; value not shown)" >&2
  exit 2
fi

staged="$(mktemp)"
trap 'rm -f "${staged}"' EXIT
chmod 600 "${staged}"
cat > "${staged}" <<ENVFILE
# Generated for FarmShare Slurm jobs by .edullm/farmshare/push_wandb_key.sh.
# Contains a credential. Do not commit, and do not copy off scratch.
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_START_METHOD=thread
ENVFILE

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${RUN_DIR}' && chmod 700 '${RUN_DIR}'"

# Piped through ssh rather than scp so the only requirement is the control
# socket, and so the destination mode is set before any bytes arrive.
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "umask 077 && cat > '${RUN_DIR}/wandb-session.env' && chmod 600 '${RUN_DIR}/wandb-session.env'" \
  < "${staged}"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "wc -c < '${RUN_DIR}/wandb-session.env' | awk '{print \"wandb_session_bytes\", \$1}'"

echo "push_wandb_key_ok -> ${RUN_DIR}/wandb-session.env"
