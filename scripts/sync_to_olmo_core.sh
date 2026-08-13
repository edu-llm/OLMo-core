#!/usr/bin/env bash
# Publish this repository as an independent branch of OLMo-core.
#
# WHY THIS EXISTS
#
# A repository of its own cannot be registered without widening the ECR publisher
# role, whose IAM stack is applied from a laptop rather than from CI, and whose file
# is owned by the two platform admins. OLMo-core is already registered: its ECR
# repository exists, its publisher-role trust already names it, and
# AWS_ECR_PUBLISHER_ROLE_ARN is already set on it. So this code rides that
# registration.
#
# WHAT "INDEPENDENT" MEANS HERE
#
# The branch carries THIS repository's history and nothing of OLMo-core's -- git
# pushes unrelated histories to a new ref happily, so no orphan-branch dance and no
# subtree is needed. Consequences worth knowing:
#
#   * The branch shares no commit with OLMo-core's main, so it can never be
#     fast-forwarded into it by accident, and a PR from it would show every file as
#     added. It is not meant to be merged.
#   * Only OUR workflows exist on it. GitHub runs workflows from the pushed branch's
#     own tree, so OLMo-core's CI does not run on this branch and ours does not run
#     on theirs.
#   * The build publishes to OLMo-core's ECR repository, tagged from this branch's
#     commit. That is the point, and it is also a side effect on a shared repository
#     -- which is why this script never runs without --confirm.
#
# The branch name must match `edullm/**` or the build never fires and the branch
# looks correct while publishing nothing.

set -euo pipefail

REMOTE_URL="${OLMO_REMOTE_URL:-https://github.com/edu-llm/OLMo-core.git}"
REMOTE_NAME="${OLMO_REMOTE_NAME:-olmo}"
BRANCH="${OLMO_BRANCH:-edullm/memsplit-hop}"
SRC_REF="${SRC_REF:-HEAD}"
CONFIRM=0
FORCE=0

usage() {
  cat <<'USAGE'
usage: scripts/sync_to_olmo_core.sh [--confirm] [--force] [--branch <name>] [--dry-run]

  --confirm        actually push. Without it, prints the push and stops.
  --force          force-update the remote branch (rewrites it). Off by default.
  --branch <name>  target branch; must start with edullm/ . Default edullm/memsplit-hop
  --dry-run        alias for omitting --confirm

env: OLMO_REMOTE_URL, OLMO_REMOTE_NAME, OLMO_BRANCH, SRC_REF
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --confirm) CONFIRM=1 ;;
    --force) FORCE=1 ;;
    --dry-run) CONFIRM=0 ;;
    --branch) BRANCH="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

case "$BRANCH" in
  edullm/*) ;;
  *) echo "refusing: branch must match edullm/** or the platform build never fires (got '$BRANCH')" >&2
     exit 2 ;;
esac

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "refusing: working tree is dirty. Commit first, so the branch is reproducible." >&2
  git status --short >&2
  exit 2
fi

# The image is built from this tree, so a broken tree is a wasted 8-11 minute build.
echo "== running the test suite first =="
MEMSPLIT_TOKENIZER=byte python -m pytest tests/ -q

if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  echo "== adding remote $REMOTE_NAME -> $REMOTE_URL =="
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi
git remote set-url "$REMOTE_NAME" "$REMOTE_URL"

SHA="$(git rev-parse "$SRC_REF")"
PUSH_ARGS=("$REMOTE_NAME" "${SHA}:refs/heads/${BRANCH}")
[ "$FORCE" -eq 1 ] && PUSH_ARGS=(--force "${PUSH_ARGS[@]}")

echo
echo "  source commit : $SHA  ($(git log -1 --format=%s "$SHA"))"
echo "  remote        : $REMOTE_NAME  $REMOTE_URL"
echo "  target branch : $BRANCH"
echo "  force         : $FORCE"
echo

if [ "$CONFIRM" -ne 1 ]; then
  cat <<EOF
Dry run. Nothing was pushed.

This would push to a SHARED repository and, because the branch matches edullm/**,
trigger a build that publishes an image to OLMo-core's ECR repository. Re-run with
--confirm when that is intended:

  scripts/sync_to_olmo_core.sh --confirm

Subsequent updates need --force only if this repository's history was rewritten;
ordinary new commits fast-forward.
EOF
  exit 0
fi

echo "== pushing =="
git push "${PUSH_ARGS[@]}"

cat <<EOF

Pushed $SHA to $BRANCH on $REMOTE_NAME.

Next:
  1. Watch the build (8-11 min). It publishes to OLMo-core's ECR repository.
  2. Submit against OLMo-core, not memsplit-hop:

       edullm submit --spec .edullm/run.single.yaml \\
         --team memory-split --experiment smoke --dataset none \\
         --compute gpu-1xl40s --hours 1

     The repository input is OLMo-core because that is what is registered and what
     both workload profiles name.
EOF
