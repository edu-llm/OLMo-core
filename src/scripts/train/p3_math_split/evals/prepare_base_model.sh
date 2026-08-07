#!/usr/bin/env bash
set -euo pipefail

# Materialize the untrained control ("base") model directory for --arm base:
# the exact Qwen2.5-0.5B snapshot the trained arms initialized from, with the
# VENDORED qwen2.5 tokenizer overlaid so the base arm's tokenizer_sha256 matches
# dense/split. The stock Hub tokenizer is deliberately not used, and no export
# model_provenance.json is written (the control has no training identity).
#
#   P3_PYTHON=/mnt/work/venv/bin/python \
#     bash prepare_base_model.sh /mnt/work/hf/base
#
# Then evaluate it (standalone, same corpus/conditions/seed as the arms):
#   "$P3_PYTHON" run_eval.py --model /mnt/work/hf/base --arm base \
#     --base-model-id Qwen/Qwen2.5-0.5B \
#     --base-model-revision 060db6499f32faf8b98477b0a26969ef7d8b9987 ...

BASE_MODEL_ID="${P3_BASE_MODEL_ID:-Qwen/Qwen2.5-0.5B}"
BASE_MODEL_REVISION="${P3_BASE_MODEL_REVISION:-060db6499f32faf8b98477b0a26969ef7d8b9987}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${P3_PYTHON:?set P3_PYTHON to the venv interpreter from bootstrap_vllm_env.sh}"
OUT_DIR="${1:?usage: prepare_base_model.sh <out-dir>}"
STAGING_DIR="${P3_BASE_STAGING_DIR:-${OUT_DIR}.tokenizer-staging}"
mkdir -p "${OUT_DIR}" "${STAGING_DIR}"

echo "Downloading pinned base weights ${BASE_MODEL_ID}@${BASE_MODEL_REVISION}"
"${PYTHON}" - "${BASE_MODEL_ID}" "${BASE_MODEL_REVISION}" "${OUT_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, revision, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
# Weights + config only. The stock tokenizer is intentionally excluded; the
# vendored tokenizer is overlaid in the next step for byte-identical scoring.
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=out_dir,
    allow_patterns=["config.json", "model.safetensors"],
)
print(f"downloaded config.json + model.safetensors for {model_id}@{revision}")
PY

echo "Overlaying the vendored qwen2.5 tokenizer (same artifact the exporter uses)"
PYTHONPATH="${SCRIPT_DIR}/.." "${PYTHON}" - "${OUT_DIR}" "${STAGING_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path

import provenance

out_dir, staging_dir = Path(sys.argv[1]), Path(sys.argv[2])
# Fetches and seals the pinned tokenizer/qwen25-vendored/v1 files from S3 exactly
# as export_checkpoint.py does, so the base arm shares the arms' tokenizer bytes.
provenance.fetch_tokenizer_artifact(provenance.TOKENIZER_ARTIFACT, staging_dir)
for name in provenance.TOKENIZER_REQUIRED_FILES:
    matches = sorted(staging_dir.rglob(name))
    if not matches:
        raise SystemExit(f"vendored tokenizer file was not fetched: {name}")
    shutil.copyfile(matches[0], out_dir / name)
print("overlaid vendored tokenizer:", ", ".join(provenance.TOKENIZER_REQUIRED_FILES))
PY

# The control arm must never carry export provenance, and must have weights,
# config, and the vendored tokenizer in place.
if [[ -f "${OUT_DIR}/model_provenance.json" ]]; then
  echo "unexpected model_provenance.json in a base control directory" >&2
  exit 3
fi
for required in config.json model.safetensors tokenizer.json tokenizer_config.json; do
  test -f "${OUT_DIR}/${required}" || { echo "missing ${required} in ${OUT_DIR}" >&2; exit 4; }
done

echo "base control model prepared at ${OUT_DIR}"
