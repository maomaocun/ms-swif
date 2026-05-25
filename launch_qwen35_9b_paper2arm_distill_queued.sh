#!/usr/bin/env bash
set -uo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LAUNCH_TS="${LAUNCH_TS:-$(date +%Y%m%d-%H%M%S)}"
LAUNCH_LOG="${LAUNCH_LOG:-${SCRIPT_DIR}/logs/qwen35-9b-paper2arm-distill-launcher/queued-${LAUNCH_TS}.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-1024}"

mkdir -p "$(dirname "${LAUNCH_LOG}")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

qwen36_running() {
  pgrep -f 'train_qwen36_27b_paper2arm_distill|qwen36-27b-paper2arm-distill' >/dev/null 2>&1
}

gpus_idle() {
  local max_used
  max_used="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
      awk 'BEGIN { max = 0 } { if ($1 > max) max = $1 } END { print max }'
  )"
  [[ "${max_used}" -lt "${GPU_IDLE_MEMORY_MIB}" ]]
}

while qwen36_running || ! gpus_idle; do
  log "Waiting for qwen36/fallback to finish and GPUs to become idle."
  sleep "${POLL_SECONDS}"
done

log "Starting Qwen3.5-9B Megatron full SFT."
bash "${SCRIPT_DIR}/train_qwen35_9b_paper2arm_distill_megatron.sh"
status=$?
log "Qwen3.5-9B Megatron full SFT exited with status ${status}."
exit "${status}"
