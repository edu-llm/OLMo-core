#!/usr/bin/env bash
# Curriculum smoke: score → order → short train via existing OLMo2-190M.py
# Usage: bash run_smoke.sh <smoke_data_dir> <pacing> [metric]
#   pacing: random | vanilla | linear | warmup
#   metric: compression_ratio | flesch | lexical_diversity | learnability

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
DATA_DIR="${1:?data dir}"
PACING="${2:-random}"
METRIC="${3:-compression_ratio}"
RUN_NAME="cl-smoke-${PACING}-${METRIC}"
STEPS="${SMOKE_STEPS:-50}"
SCRIPT="${TRAIN_SCRIPT:-OLMo2/OLMo2-190M.py}"

cd "$ROOT"
HYP="$ROOT/src/scripts/hypothesis"

python "$HYP/curriculum/score_difficulty.py" --data-dir "$DATA_DIR"
python "$HYP/curriculum/build_pacing_order.py" --data-dir "$DATA_DIR" --pacing "$PACING" --metric "$METRIC"

ORDER_FILE="$DATA_DIR/orders/${PACING}__${METRIC}.jsonl"
if [[ "$PACING" == "random" ]]; then
  ORDER_FILE="$DATA_DIR/orders/random__none.jsonl"
fi

echo "CL smoke"
echo "  order: $ORDER_FILE"
echo "  script: src/scripts/train/$SCRIPT"

python "src/scripts/train/$SCRIPT" dry_run "$RUN_NAME" local \
  --save-folder="./runs/$RUN_NAME" \
  --trainer.hard_stop="{value: $STEPS, unit: steps}" \
  || python "src/scripts/train/$SCRIPT" train_single "$RUN_NAME" \
  --save-folder="./runs/$RUN_NAME" \
  --trainer.hard_stop="{value: $STEPS, unit: steps}"

echo "Order file for this condition: $ORDER_FILE"
echo "LR arms (cosine vs constant+EMA) are trainer overrides — finalize with training lead for smoke."
