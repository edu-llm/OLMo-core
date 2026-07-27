#!/bin/bash
# Autonomous long-context (increasing seq-len) breaking-point eval, one model per GPU,
# each launched on its OWN GPU as soon as that model finishes TRAINING.
#
#   GPU5 -> linear (full)        s3://.../linear
#   GPU6 -> gdn    (full)        s3://.../gdn
#   GPU4 -> linear-halfkv, then gdn-halfkv (both trained sequentially on GPU4;
#           GPU4 is busy training until BOTH finish, so both half-KV evals run on
#           GPU4 at the end, in order).
#
# "Finished training" is detected by SUSTAINED GPU idle (5 consecutive 60s checks;
# the brief inter-run gap on GPU4 is far shorter, so it is not mistaken for done).
# The final checkpoint is the highest stepNNNNN dir present under the save folder.
#
# Results: printed table + JSON (uploaded next to the checkpoints as longctx_eval.json).
# W&B is intentionally NOT used here (its key lives only in the training procs' env,
# which have exited by eval time); the JSON + table are the durable record.
set -uo pipefail
cd /mnt/nvme/olmo

DPFX="s3://edullm-datasets/olmo-150b-dolma2/data/preprocessed/dolma2-0625/v0.1-150b/allenai/dolma2-tokenizer"
EVAL="${DPFX}/arxiv/part-00-00000.npy,${DPFX}/finemath-3plus/part-000-00000.npy,${DPFX}/wikipedia/part-00-00000.npy"
SWEEP="4096,8192,16384,32768,65536,131072,262144"
PY=/mnt/nvme/olmo/venv/bin/torchrun
EVALPY=experiments/linear-attn-vs-gdn/eval_long_context.py
mkdir -p /mnt/nvme/olmo/logs

# highest stepNNNNN under an s3 save folder
highest_step () {
  aws s3 ls "$1/" 2>/dev/null | grep -oE 'step[0-9]+' | sort -t p -k2 -n | tail -1
}

wait_gpu_idle () {  # $1=gpu index ; returns when idle for 5 consecutive checks
  local gpu="$1" n=0
  while [ "$n" -lt 5 ]; do
    local m
    m=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$m" ] && [ "$m" -lt 5000 ]; then n=$((n+1)); else n=0; fi
    sleep 60
  done
}

run_eval () {  # $1=gpu $2=save_folder $3=run_name
  local gpu="$1" save="$2" run="$3"
  local step ckpt log
  step=$(highest_step "$save")
  if [ -z "$step" ]; then echo "[$(date -u +%FT%TZ)] $run: no checkpoint under $save; skip" ; return 1; fi
  ckpt="$save/$step"
  log=/mnt/nvme/olmo/logs/longctx-${run}.log
  echo "[$(date -u +%FT%TZ)] $run: eval on GPU${gpu} ckpt=${ckpt}" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" --standalone --nproc-per-node=1 "$EVALPY" "$run" \
    --checkpoint "$ckpt" --eval-data "$EVAL" \
    --seq-lens "$SWEEP" --tokens-per-len 2000000 --max-windows 256 \
    --work-dir "/mnt/nvme/olmo-longctx-${run}" \
    --output "/mnt/nvme/olmo/logs/longctx-${run}.json" \
    --upload-to "${save}/longctx_eval.json" \
    >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] $run: eval done rc=$? -> ${save}/longctx_eval.json" | tee -a "$log"
}

# --- GPU5: linear (full) ---
( wait_gpu_idle 5
  run_eval 5 s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear linear-attn-370m-10b
) > /mnt/nvme/olmo/logs/longctx-orch-gpu5.log 2>&1 &
echo "orchestrator GPU5 (linear) pid=$!"

# --- GPU6: gdn (full) ---
( wait_gpu_idle 6
  run_eval 6 s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/gdn gdn-370m-10b
) > /mnt/nvme/olmo/logs/longctx-orch-gpu6.log 2>&1 &
echo "orchestrator GPU6 (gdn) pid=$!"

# --- GPU4: linear-halfkv then gdn-halfkv (both after GPU4 training fully done) ---
( wait_gpu_idle 4
  run_eval 4 s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear-halfkv linear-halfkv-hd32-10b
  run_eval 4 s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/gdn-halfkv     gdn-halfkv-hd32-10b
) > /mnt/nvme/olmo/logs/longctx-orch-gpu4.log 2>&1 &
echo "orchestrator GPU4 (halfkv x2) pid=$!"

echo "[$(date -u +%FT%TZ)] all long-context orchestrators deployed"
