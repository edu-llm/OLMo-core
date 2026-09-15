#!/usr/bin/env bash
# Scan a mixlaw pool's text, one array task per selected text shard.
#
# Sharded by FILE rather than by domain because the pool's domains are wildly
# uneven: dclm is 68 files while algebraic-stack is 1, and a per-domain array
# would leave one task doing 6B tokens while others idle. Sizing from the
# measured 576k words/sec/core: the largest single file is ~3B tokens (~2.2B
# words), so about an hour, and --time is set from that with margin.
#
# Everything in this pipeline is stdlib-only, so plain python3 works and no
# venv is needed.
#
# Requires: CN_RUN_DIR, CN_TASKLIST, CN_INDEX, CN_OUT
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=mx-scan
#SBATCH --output=logs/poolscan-%A_%a.out
#SBATCH --error=logs/poolscan-%A_%a.err
set -Eeuo pipefail

: "${CN_RUN_DIR:?}"
: "${CN_TASKLIST:?}"
: "${CN_INDEX:?}"
: "${CN_OUT:?}"

# Task list lines are: <domain><TAB><shard-id><TAB><absolute text path>
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CN_TASKLIST}")
[ -n "${LINE}" ] || { echo "no task line for id ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
DOMAIN=$(printf '%s' "${LINE}" | cut -f1)
SHARD=$(printf '%s' "${LINE}" | cut -f2)
TEXT=$(printf '%s' "${LINE}" | cut -f3)

echo "task ${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} shard=${SHARD} on $(hostname)"
echo "text=${TEXT}"

exec python3 "${CN_RUN_DIR}/OLMo-core/.edullm/contamination/scan_corpus.py" \
  --index "${CN_INDEX}" \
  --trim "${TEXT}" \
  --domain "${DOMAIN}" \
  --shard "${SHARD}" \
  --out-dir "${CN_OUT}"
