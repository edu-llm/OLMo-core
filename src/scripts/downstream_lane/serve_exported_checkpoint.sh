#!/usr/bin/env bash
#
# Serve a HuggingFace directory this repository exported, over vLLM's OpenAI-compatible
# API, and do not return until it answers.
#
# THE SERVER RATHER THAN `python -m olmo_core.generate.chat`, AND THE DIFFERENCE IS WHO
# IS DRIVING. That module is a single-process rich terminal over an OLMo-core checkpoint:
# one person, one keyboard, weights reloaded per invocation, and marked beta. This holds
# the weights across requests, speaks the protocol every chat front end already
# implements, and can be pointed at from a browser on somebody else's laptop. For a
# demonstration where the person typing is not the person who built it, that is the whole
# argument.
#
# --max-model-len IS PASSED AND IS NOT OPTIONAL. `get_hf_config` builds the config with
# `max_position_embeddings=-1` and `convert_checkpoint_to_hf` only overwrites it when it
# was given `--max-sequence-length` or could read a `model_max_length` off the tokenizer.
# An export that got neither carries -1 into config.json, and vLLM reads that as the
# context window before it reads anything else. Naming the length here means a bad export
# fails on its weights rather than on its metadata.
#
# --gpu-memory-utilization defaults low enough to leave room beside the server, because
# the lane node runs an export or an evaluation next to it more often than not. Raise it
# for a machine doing nothing else.
set -euo pipefail

MODEL=${1:?usage: serve_exported_checkpoint.sh MODEL_DIR [PORT] [SERVED_NAME]}
PORT=${2:-8000}
SERVED_NAME=${3:-edullm}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_FRACTION=${GPU_FRACTION:-0.60}
LOG=${SERVE_LOG:-/tmp/vllm-${PORT}.log}

vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_FRACTION}" \
  --dtype bfloat16 \
  ${EXTRA_SERVE_ARGS:-} \
  > "${LOG}" 2>&1 &
SERVER=$!

# Ten minutes. A cold start on a 7B MoE is dominated by reading the weights off disk, and
# a poll that gives up sooner reports a failure that has not happened. The process check
# is what stops the loop early when the server died rather than waiting the full budget
# for a port nothing will ever open.
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER}" 2>/dev/null; then
    echo "vllm serve exited before it opened ${PORT}; last 40 lines of ${LOG}:" >&2
    tail -40 "${LOG}" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then
    echo "serving ${SERVED_NAME} from ${MODEL} on ${PORT}, pid ${SERVER}, log ${LOG}"
    exit 0
  fi
  sleep 5
done

echo "vllm serve did not answer on ${PORT} within ten minutes; last 40 lines of ${LOG}:" >&2
tail -40 "${LOG}" >&2
kill "${SERVER}" 2>/dev/null || true
exit 1
