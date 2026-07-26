#!/usr/bin/env bash
# Sequential 4-arm CPT on one B200 GPU.
# REFUSES to train unless ALLOW_TRAIN=1 is set (Hiya explicit go).
set -euo pipefail

if [[ "${ALLOW_TRAIN:-0}" != "1" ]]; then
  echo "Refusing to train. Set ALLOW_TRAIN=1 only when Hiya says go."
  echo "For staging use: prepare_b200.sh"
  echo "For config check: train_cpt_arm.py … --dry-run"
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to your reserved GPU index}"
WE_ROOT="${WE_ROOT:-/mnt/nvme/we}"
CODE_DIR="${CODE_DIR:-$WE_ROOT/code/OLMo-core}"
PACK_DIR="${PACK_DIR:-$WE_ROOT/pack}"
CKPT_DIR="${CKPT_DIR:-$WE_ROOT/ckpt/370m}"
S3_PREFIX="${S3_PREFIX:-s3://memorysplit-stephen-056956104102-us-east-1/runs/worked-examples}"
TOKEN_BUDGET="${TOKEN_BUDGET:-50000000}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY="${WANDB_ENTITY:-eduLLM}"
export WANDB_PROJECT="${WANDB_PROJECT:-pretraining}"

for ARM in bare complete fade_ordered fade_shuffled; do
  echo "======== TRAIN $ARM ========"
  torchrun --standalone --nproc-per-node=1 \
    src/scripts/hypothesis/worked_examples/train_cpt_arm.py "we-cpt-${ARM}-b200" \
    --arm "$ARM" \
    --pack-dir "$PACK_DIR" \
    --load-path "$CKPT_DIR" \
    --token-budget "$TOKEN_BUDGET" \
    --model-factory olmo3_370M \
    --run-tag b200 \
    --global-batch-size "${GLOBAL_BATCH_SIZE:-65536}" \
    --rank-microbatch-size "${RANK_MICROBATCH_SIZE:-8192}" \
    --save-folder "$WE_ROOT/runs/$ARM" \
    --wandb-group worked-examples-faded-scaffolds-b200

  aws s3 sync "$WE_ROOT/runs/$ARM/" "$S3_PREFIX/$ARM/"
done

echo "All arms finished + synced. Run holdout_passn when ready."
