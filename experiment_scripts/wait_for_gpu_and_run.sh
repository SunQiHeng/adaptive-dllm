#!/usr/bin/env bash
set -euo pipefail

# Wait until a physical GPU is sustainably idle, then run the supplied command.
#
# Required environment:
#   GPU_ID=<physical GPU index>
#   RUN_LOG=<waiter and command log>
#
# Optional:
#   MAX_USED_MIB=10000
#   MAX_UTIL_PERCENT=10
#   REQUIRED_IDLE_CHECKS=3
#   CHECK_INTERVAL_SECONDS=60

GPU_ID=${GPU_ID:?Set GPU_ID to the physical GPU index to monitor.}
RUN_LOG=${RUN_LOG:?Set RUN_LOG to the waiter and command log path.}
MAX_USED_MIB=${MAX_USED_MIB:-10000}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-10}
REQUIRED_IDLE_CHECKS=${REQUIRED_IDLE_CHECKS:-3}
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-60}

if [ "$#" -eq 0 ]; then
  echo "ERROR: provide the command to run after the GPU becomes idle." >&2
  exit 2
fi

mkdir -p "$(dirname "${RUN_LOG}")"
touch "${RUN_LOG}"

log() {
  printf "[%s] %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$*" | tee -a "${RUN_LOG}"
}

idle_checks=0
log "Waiting for GPU ${GPU_ID}: memory.used<=${MAX_USED_MIB} MiB and utilization<=${MAX_UTIL_PERCENT}% for ${REQUIRED_IDLE_CHECKS} consecutive checks."

while true; do
  snapshot="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  used_mib="$(echo "${snapshot}" | cut -d',' -f1 | xargs)"
  util_percent="$(echo "${snapshot}" | cut -d',' -f2 | xargs)"

  if [ "${used_mib}" -le "${MAX_USED_MIB}" ] && [ "${util_percent}" -le "${MAX_UTIL_PERCENT}" ]; then
    idle_checks=$((idle_checks + 1))
    log "GPU ${GPU_ID} idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}: used=${used_mib} MiB util=${util_percent}%."
  else
    if [ "${idle_checks}" -gt 0 ]; then
      log "GPU ${GPU_ID} became busy again; resetting idle counter. used=${used_mib} MiB util=${util_percent}%."
    else
      log "GPU ${GPU_ID} busy: used=${used_mib} MiB util=${util_percent}%."
    fi
    idle_checks=0
  fi

  if [ "${idle_checks}" -ge "${REQUIRED_IDLE_CHECKS}" ]; then
    break
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

log "GPU ${GPU_ID} is sustainably idle. Starting command: $*"
set +e
"$@" >> "${RUN_LOG}" 2>&1
rc=$?
set -e
log "Command finished with rc=${rc}."
exit "${rc}"
