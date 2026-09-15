#!/usr/bin/env bash
# Scan the corpus for eval-item n-grams, one array task per domain.
#
# Sizing is from a measured pilot: 557k words/sec/core on wheat-08, so dclm
# (~2.7e9 words) is the long pole at roughly 80 minutes and everything else
# finishes sooner. --time is set from that with margin rather than copied.
#
# Requires: CN_RUN_DIR, CN_CORPUS, CN_VENV
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:30:00
#SBATCH --job-name=cn-scan
#SBATCH --array=0-7
#SBATCH --output=logs/scan-%A_%a.out
#SBATCH --error=logs/scan-%A_%a.err
set -Eeuo pipefail

: "${CN_RUN_DIR:?}"
: "${CN_CORPUS:?}"
: "${CN_VENV:?}"

DOMAINS=(dclm arxiv nemotron-hqdqa starcoder pes2o open-web-math algebraic-stack wiki)
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID}]}"

echo "task ${SLURM_ARRAY_TASK_ID} -> ${DOMAIN} on $(hostname)"

exec "${CN_VENV}/bin/python" \
  "${CN_RUN_DIR}/OLMo-core/.edullm/contamination/scan_corpus.py" \
  --index "${CN_RUN_DIR}/index/item_index.pkl" \
  --trim "${CN_CORPUS}/trim/${DOMAIN}/${DOMAIN}-trimmed.json.gz" \
  --domain "${DOMAIN}" \
  --out-dir "${CN_RUN_DIR}/hits"
