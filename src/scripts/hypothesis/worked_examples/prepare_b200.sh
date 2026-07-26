#!/usr/bin/env bash
# Stage scaffolding pack + dirs on the MemorySplit B200 NVMe.
# Does NOT train. Does NOT buy capacity. Does NOT stop the instance.
#
# Usage (on the box as ubuntu):
#   export CKPT_URI='s3://…/nathan-370m-chinchilla/'   # required
#   export HF_PACK_ID='hiyasvyas/worked-examples-metamath-v0'  # default
#   bash src/scripts/hypothesis/worked_examples/prepare_b200.sh
set -euo pipefail

WE_ROOT="${WE_ROOT:-/mnt/nvme/we}"
PACK_DIR="${PACK_DIR:-$WE_ROOT/pack}"
CKPT_DIR="${CKPT_DIR:-$WE_ROOT/ckpt/370m}"
RUNS_DIR="${RUNS_DIR:-$WE_ROOT/runs}"
CODE_DIR="${CODE_DIR:-$WE_ROOT/code/OLMo-core}"
HF_PACK_ID="${HF_PACK_ID:-hiyasvyas/worked-examples-metamath-v0}"
CKPT_URI="${CKPT_URI:-}"

echo "== B200 prepare (no train) =="
echo "WE_ROOT=$WE_ROOT"

mkdir -p "$PACK_DIR" "$CKPT_DIR" "$RUNS_DIR"/{bare,complete,fade_ordered,fade_shuffled}

if [[ ! -d "$CODE_DIR/.git" ]]; then
  echo "Clone code first, e.g.:"
  echo "  git clone https://github.com/edu-llm/OLMo-core.git $CODE_DIR"
  echo "  cd $CODE_DIR && git fetch origin && git checkout hypothesis/we-metamath-wandb-smoke"
  exit 2
fi

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --- pack from Hugging Face ---
if [[ ! -f "$PACK_DIR/eval/holdout_bare.jsonl" ]]; then
  echo "Downloading HF pack $HF_PACK_ID -> $PACK_DIR"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_PACK_ID}", repo_type="dataset", local_dir="${PACK_DIR}")
print("HF pack ready")
PY
else
  echo "Pack eval present; skipping HF download"
fi

need_tokenize=0
for arm in bare complete fade_ordered fade_shuffled; do
  shard="$PACK_DIR/tokenized/$arm/shard-00000.npy"
  mask="$PACK_DIR/tokenized/$arm/label_mask-00000.npy"
  if [[ ! -f "$shard" || ! -f "$mask" ]]; then
    need_tokenize=1
  fi
done

if [[ "$need_tokenize" -eq 1 ]]; then
  echo "Tokenizing + writing label masks (CPU)…"
  python3 src/scripts/hypothesis/worked_examples/tokenize_arms.py \
    --pack-dir "$PACK_DIR" \
    --tokenizer dolma2
else
  echo "All shards + label_mask files present"
fi

# --- checkpoint ---
if [[ -z "$CKPT_URI" ]]; then
  echo "WARNING: CKPT_URI not set. Set it to Nathan's 370M OLMo-core checkpoint S3 URI, then re-run:"
  echo "  export CKPT_URI=s3://…/…"
  echo "  aws s3 sync \"\$CKPT_URI\" \"$CKPT_DIR/\""
else
  echo "Syncing ckpt $CKPT_URI -> $CKPT_DIR"
  aws s3 sync "$CKPT_URI" "$CKPT_DIR/"
fi

echo
echo "Sanity:"
ls -lh "$PACK_DIR/tokenized"/*/shard-00000.npy "$PACK_DIR/tokenized"/*/label_mask-00000.npy || true
ls -lh "$PACK_DIR/eval/holdout_bare.jsonl" || true
ls -lh "$CKPT_DIR" | head || true
echo
echo "Prepare done. Next: dry-run from B200_RUNBOOK.md (still no full train until go)."
