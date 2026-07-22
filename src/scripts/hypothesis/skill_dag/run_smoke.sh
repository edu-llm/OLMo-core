#!/usr/bin/env bash
# Skill-DAG smoke: short train via existing OLMo2-190M.py (unchanged).
# Usage: bash run_smoke.sh <smoke_data_dir> [natural|fixed_uniform|skillit_init]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_DIR="${1:?data dir}"
MIX_NAME="${2:-natural}"
RUN_NAME="skilldag-smoke-${MIX_NAME}"
STEPS="${SMOKE_STEPS:-50}"
SCRIPT="${TRAIN_SCRIPT:-OLMo2/OLMo2-190M.py}"

cd "$ROOT"
MIX_FILE="$ROOT/src/scripts/hypothesis/skill_dag/configs/${MIX_NAME}.json"
if [[ ! -f "$MIX_FILE" ]]; then
  # fall back to pack-generated natural mix
  MIX_FILE="$DATA_DIR/manifests/mixes/natural.json"
fi

echo "Skill-DAG smoke"
echo "  data: $DATA_DIR"
echo "  mix:  $MIX_FILE"
echo "  script: src/scripts/train/$SCRIPT"
echo "  hard_stop: $STEPS steps"

# Prefer dry_run if the script supports it; then a tiny train_single/train.
# Paths/overrides may need team finalize once dataset.mix wiring for custom shards lands.
python "src/scripts/train/$SCRIPT" dry_run "$RUN_NAME" local \
  --save-folder="./runs/$RUN_NAME" \
  --trainer.hard_stop="{value: $STEPS, unit: steps}" \
  || python "src/scripts/train/$SCRIPT" train_single "$RUN_NAME" \
  --save-folder="./runs/$RUN_NAME" \
  --trainer.hard_stop="{value: $STEPS, unit: steps}"

echo "Smoke invoke finished. If dry_run only printed config, next: wire --dataset paths to $DATA_DIR/tokenized"
echo "Mix file for this condition: $MIX_FILE"
