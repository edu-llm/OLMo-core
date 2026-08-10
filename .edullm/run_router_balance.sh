#!/usr/bin/env bash
# Run the router-balance arms back to back in one container, gating on the arm table first.
#
#     bash .edullm/run_router_balance.sh                 # every arm, in priority order
#     bash .edullm/run_router_balance.sh rb-anchor rb-g1e4
#
# ONE CONTAINER FOR EVERY ARM, WHICH IS THE ONLY WAY THIS IS AFFORDABLE. A claim, a clone and a
# torch import cost about four minutes and buy nothing; paying that once instead of six times is
# most of an arm. It also removes the largest confound in the earlier probe -- the imbalanced arm
# is straggler-bound, so it amplifies whatever is slowest on the box, and a ratio taken across two
# machines measures the machines. Every ratio this produces is between arms that ran on one node
# minutes apart.
#
# THE GATE IS NOT OPTIONAL AND RUNS BEFORE ANY GPU WORK. `run_name` is a positional with
# `nargs="?"`, so a setting appended as a bare word renames the run instead of applying, and the
# arm trains for forty minutes as a duplicate of the control. `--verify` builds every arm through
# the real `build_config` and asserts the value on the built module. If it refuses, nothing runs.
#
# ARMS ARE INDEPENDENT. One that dies -- of memory, most likely, since imbalance sizes the
# dropless buffers -- does not stop the rest, and its partial log is still summarised.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

RB_DIR="${RB_DIR:-/tmp/router-balance}"
NPROC="${NPROC:-8}"
mkdir -p "${RB_DIR}"

say() { printf '\n===== %s :: %s =====\n' "$(date -u +%H:%M:%SZ)" "$*"; }

say "router-balance sweep starting"
python - <<'PY'
import os, subprocess, sys
print("nvidia-smi:")
sys.stdout.flush()
subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], check=False)
for key in ("EDULLM_RUN_ID", "EDULLM_OUTPUT_PREFIX", "EDULLM_DATA_BUCKET", "EDULLM_CHECKPOINT_DIR"):
    print(f"{key}={os.environ.get(key, '<unset>')}")
PY

say "GATE 1/2: the three overrides the earlier report recommends still reach their modules"
if ! python .edullm/verify_router_overrides.py; then
  say "GATE 1 REFUSED -- nothing dispatched"
  exit 1
fi

say "GATE 2/2: every arm in the table reaches the module it names"
if ! python .edullm/router_balance_arms.py --verify; then
  say "GATE 2 REFUSED -- nothing dispatched"
  exit 1
fi

say "the commands that will run"
python .edullm/router_balance_arms.py --commands --nproc "${NPROC}"

if [ "$#" -gt 0 ]; then
  ARMS=("$@")
else
  mapfile -t ARMS < <(python .edullm/router_balance_arms.py --names)
fi
say "arms: ${ARMS[*]}"

DONE=()
for arm in "${ARMS[@]}"; do
  LOG="${RB_DIR}/${arm}.log"
  say "ARM ${arm} starting -> ${LOG}"
  started=$(date +%s)

  # The full output goes to the file; only progress every fiftieth step, and anything that looks
  # like a failure, reaches stdout. Stdout is what the node copies to S3 once a minute, so the
  # compact form is the one that survives the machine and the long one is for the parser beside
  # it. `--commands` and this line take the same argv from the same table.
  cmd=$(python .edullm/router_balance_arms.py --commands --arm "${arm}" --nproc "${NPROC}")
  if [ -z "${cmd}" ]; then
    say "ARM ${arm} is not in the table -- skipped"
    continue
  fi
  eval "${cmd}" > >(tee "${LOG}" | grep --line-buffered -E \
      '\[step=[0-9]*[05]0/|Error|error:|Traceback|out of memory|OutOfMemory|refus|Refusal|OVERRIDES|"run_id"' \
      || true) 2>&1
  status=$?
  wait
  elapsed=$(( $(date +%s) - started ))

  if [ "${status}" -ne 0 ]; then
    say "ARM ${arm} EXITED ${status} after ${elapsed}s -- last 40 lines follow"
    tail -n 40 "${LOG}"
  else
    say "ARM ${arm} finished in ${elapsed}s"
  fi
  DONE+=("${LOG}")

  # Printed after every arm rather than once at the end, so that a container taken back by the
  # fleet still leaves a readable result for everything that had finished.
  say "ARM ${arm} curve"
  python .edullm/router_balance_report.py "${LOG}" --every 50 --window 100 || true

  say "everything so far"
  python .edullm/router_balance_report.py "${DONE[@]}" --every 250 --window 100 --json || true
done

say "router-balance sweep complete: ${#DONE[@]} arm(s)"
python .edullm/router_balance_report.py "${DONE[@]}" --every 100 --window 100 --json || true
say "ROUTER_BALANCE_SWEEP_DONE"
