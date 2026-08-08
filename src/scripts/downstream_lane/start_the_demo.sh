#!/usr/bin/env bash
#
# Leave an endpoint and a chat page running on this machine, and come back.
#
#   edullm run --project <project> --compute gpu-1xl40s --hours 4 \
#     -- bash src/scripts/downstream_lane/start_the_demo.sh <checkpoint-uri>
#   edullm shell --project <project> --notebook     # then http://localhost:8890/
#
# `setsid` IS THE WHOLE POINT OF THIS FILE AND EVERYTHING ELSE IN IT IS SCAFFOLDING. The
# lane runs a command through a Systems Manager session and tears the process group down
# when that session ends, so a server started with a plain `&` dies the moment the verb
# that started it returns -- leaving a machine that is running, billing, and serving
# nothing. A new session detaches the servers from the one that started them, and then the
# person who started them can close the laptop.
#
# It waits for the page to answer before returning, so that the verb's exit status means
# "there is something to connect to" rather than "the command was accepted".
set -euo pipefail

CHECKPOINT=${1:?usage: start_the_demo.sh CHECKPOINT_URI [SERVED_NAME]}
SERVED=${2:-edullm}
PORT=${PORT:-8000}
PAGE_PORT=${PAGE_PORT:-8888}
LOG=${LOG:-/tmp/edullm-demo.log}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PATH="${HOME}/.local/bin:${PATH}"

# Anything still holding the ports is a previous attempt, and two vLLMs on one card is an
# out-of-memory failure that reads as a broken checkpoint.
pkill -f 'vllm serve' 2>/dev/null || true
pkill -f chat_page.py 2>/dev/null || true
sleep 5

setsid nohup python3 "${HERE}/serve_a_checkpoint.py" "$CHECKPOINT" \
  --port "$PORT" --served-name "$SERVED" --chat-page-port "$PAGE_PORT" \
  > "$LOG" 2>&1 < /dev/null &

echo "starting; following ${LOG}"
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PAGE_PORT}/healthz" > /dev/null; then
    tail -20 "$LOG"
    echo
    echo "The endpoint is on ${PORT} and the page is on ${PAGE_PORT}, both detached."
    echo "From a laptop:  edullm shell --project <project> --notebook"
    echo "then open       http://localhost:8890/"
    exit 0
  fi
  sleep 5
done

echo "nothing answered on ${PAGE_PORT} within twenty minutes. The log:" >&2
tail -60 "$LOG" >&2
exit 1
