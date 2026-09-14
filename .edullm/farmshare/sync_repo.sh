#!/usr/bin/env bash
# Sync a committed snapshot of this repo to FarmShare scratch (no secrets).
#
# A run is attributable only if the tree it came from is a git object rather
# than a working copy. The previous version of this script tarred the live
# working tree, so a run could not be tied to a commit and two runs with the
# same recorded identity could differ. This version archives HEAD, refuses a
# dirty tree by default, and writes the commit and archive digests next to
# the code so run_fingerprint.json can assert against them.
#
# Env:
#   RUN_DIR      scratch run directory on the remote host (required)
#   LOCAL_REPO   path to the local clone (required)
#   SOCK         ssh control socket (required)
#   HOST         user@host (required)
#   ALLOW_DIRTY  set to 1 to archive a dirty tree anyway. The provenance
#                records dirty=true, so such a run cannot later be mistaken
#                for a clean one -- but it is not poolable and the freeze
#                contract rejects it.
set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${LOCAL_REPO:?}"
: "${SOCK:?}"
: "${HOST:?}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

git -C "${LOCAL_REPO}" rev-parse --is-inside-work-tree >/dev/null

dirty_files="$(git -C "${LOCAL_REPO}" status --porcelain)"
if [ -n "${dirty_files}" ]; then
  if [ "${ALLOW_DIRTY}" != "1" ]; then
    echo "sync_repo: refusing to sync a dirty tree (commit, or set ALLOW_DIRTY=1):" >&2
    printf '%s\n' "${dirty_files}" >&2
    exit 1
  fi
  echo "sync_repo: WARNING archiving a dirty tree; this run is not poolable" >&2
fi

commit="$(git -C "${LOCAL_REPO}" rev-parse HEAD)"
describe="$(git -C "${LOCAL_REPO}" describe --always --dirty --tags 2>/dev/null \
  || git -C "${LOCAL_REPO}" rev-parse --short HEAD)"
branch="$(git -C "${LOCAL_REPO}" rev-parse --abbrev-ref HEAD)"
if [ -n "${dirty_files}" ]; then is_dirty=true; else is_dirty=false; fi

staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT

# gzip -n omits the timestamp so the digest depends only on content, which is
# what makes it comparable across syncs of the same commit.
git -C "${LOCAL_REPO}" archive --format=tar HEAD -- pyproject.toml src .edullm \
  | gzip -n > "${staging}/code.tar.gz"
git -C "${LOCAL_REPO}" archive --format=tar HEAD -- .edullm/farmshare \
  | gzip -n > "${staging}/scripts.tar.gz"

code_sha="$(sha256sum "${staging}/code.tar.gz" | cut -d' ' -f1)"
scripts_sha="$(sha256sum "${staging}/scripts.tar.gz" | cut -d' ' -f1)"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "mkdir -p '${RUN_DIR}/OLMo-core' '${RUN_DIR}/scripts' '${RUN_DIR}/logs' && chmod 700 '${RUN_DIR}'"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "tar -xzf - -C '${RUN_DIR}/OLMo-core'" < "${staging}/code.tar.gz"

# git archive keeps the full path, so strip .edullm/farmshare/ to land the
# launcher scripts flat in scripts/ the way the sbatch files expect.
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "tar -xzf - -C '${RUN_DIR}/scripts' --strip-components=2" < "${staging}/scripts.tar.gz"

ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "find '${RUN_DIR}/scripts' -type f \( -name '*.sh' -o -name '*.sbatch' -o -name '*.env' \) -exec sed -i 's/\r$//' {} + && \
   chmod +x '${RUN_DIR}/scripts'/*.sh 2>/dev/null || true"

# Provenance is written from the values computed above rather than re-derived
# on the remote, which has no git clone to ask.
ssh -S "${SOCK}" -o BatchMode=yes "${HOST}" \
  "cat > '${RUN_DIR}/code_provenance.json'" <<PROVENANCE
{
  "commit": "${commit}",
  "describe": "${describe}",
  "branch": "${branch}",
  "dirty": ${is_dirty},
  "code_tar_sha256": "${code_sha}",
  "scripts_tar_sha256": "${scripts_sha}"
}
PROVENANCE

echo "sync_ok run_dir=${RUN_DIR} commit=${commit} dirty=${is_dirty} code_sha256=${code_sha}"
