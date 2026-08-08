#!/bin/bash
# GPU-pinned serial worker for the B200 box.
#
# One worker per GPU. Each claims jobs from a shared queue file using an atomic
# mkdir lock, so N workers on N GPUs never take the same job. A worker WAITS for
# its GPU to be idle before starting, so it can be launched on a busy GPU and
# will simply begin when the current tenant finishes.
#
# usage: nohup ./b200_worker.sh <GPU_INDEX> > worker<GPU>.log 2>&1 &
set -uo pipefail
GPU="${1:?need gpu index}"
ROOT=/mnt/nvme/kda-probes
QUEUE=$ROOT/queue.txt
CLAIMS=$ROOT/claims
RESULTS=$ROOT/results
FREE_MIB="${FREE_MIB:-2000}"     # consider GPU idle below this used-memory
mkdir -p "$CLAIMS" "$RESULTS"

used() { nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits; }

# Wait (indefinitely) for this GPU to be free, so we can queue behind someone else.
while :; do
  u=$(used)
  [ "${u:-999999}" -lt "$FREE_MIB" ] && break
  echo "[gpu$GPU] busy (${u} MiB used); waiting for the current run to finish..."
  sleep 120
done
echo "[gpu$GPU] GPU is free (${u} MiB); starting work"

n=0
while :; do
  claimed=""
  # Atomic claim: mkdir succeeds for exactly one worker per job id.
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    id=$(echo "$line" | tr ',/ ' '___')
    if mkdir "$CLAIMS/$id" 2>/dev/null; then claimed="$line"; break; fi
  done < "$QUEUE"
  [ -z "$claimed" ] && { echo "[gpu$GPU] queue exhausted after $n jobs"; break; }

  MIXER=$(echo "$claimed" | cut -d, -f1)
  TASK=$(echo  "$claimed" | cut -d, -f2)
  SEED=$(echo  "$claimed" | cut -d, -f3)
  LAYERS=$(echo "$claimed" | cut -d, -f4)
  RH=$(echo    "$claimed" | cut -d, -f5)
  OUT="$RESULTS/${MIXER}-${TASK}-L${LAYERS}-R${RH}-s${SEED}.json"

  if [ -s "$OUT" ]; then echo "[gpu$GPU] skip (exists) $claimed"; continue; fi
  echo "[gpu$GPU] RUN $claimed -> $(basename "$OUT")"
  CUDA_VISIBLE_DEVICES="$GPU" "$ROOT/venv/bin/python" "$ROOT/train_probe.py" \
      --mixer "$MIXER" --task "$TASK" --seed "$SEED" \
      --n-layers "$LAYERS" --num-householder "$RH" \
      --steps 2000 --eval-lengths 40 64 128 256 512 \
      --out "$OUT" 2>&1 | tail -4
  rc=$?
  [ $rc -ne 0 ] && echo "[gpu$GPU] FAILED rc=$rc for $claimed"
  n=$((n+1))
done
echo "[gpu$GPU] worker done"
