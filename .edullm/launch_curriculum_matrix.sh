#!/usr/bin/env bash
# Print all recipe arms, or pass each launch to an explicitly supplied local wrapper.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

mapfile -t ARMS < <("${PYTHON}" -c 'import json,sys; [print("{}|{}|{}|{}".format(a["index"], a["name"], a["pacing"], a["metric"] or "-")) for a in json.load(open(sys.argv[1]))["arms"]]' "${ROOT}/curriculum_recipe.json")

printf '# index arm pacing metric\n'
printf '%s\n' "${ARMS[@]}"
if [[ "${1:-}" == "--print-only" || -z "${SUBMIT_CMD:-}" ]]; then
  exit 0
fi

for entry in "${ARMS[@]}"; do
  IFS='|' read -r index _name _pacing _metric <<<"${entry}"
  export ARM_INDEX="${index}"
  # SUBMIT_CMD is intentionally caller-owned; this repository does not choose a service.
  # shellcheck disable=SC2086
  ${SUBMIT_CMD} "${ROOT}/launch_curriculum_arm.sh"
done
