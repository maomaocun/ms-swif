#!/usr/bin/env bash
set -uo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LAUNCH_TS="${LAUNCH_TS:-$(date +%Y%m%d-%H%M%S)}"
LAUNCH_LOG="${LAUNCH_LOG:-${SCRIPT_DIR}/logs/qwen36-27b-paper2arm-distill-launcher/full-with-fallback-${LAUNCH_TS}.log}"
MEGATRON_LOG_ROOT="${MEGATRON_LOG_ROOT:-${SCRIPT_DIR}/logs/qwen36-27b-paper2arm-distill-megatron}"

mkdir -p "$(dirname "${LAUNCH_LOG}")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}"
export LINEAR_CE_CHUNK_SIZE

log "Starting Megatron full SFT: TP=8 CP=1 max_length=262144 optimizer_cpu_offload=true linear_ce_chunk=${LINEAR_CE_CHUNK_SIZE}"
bash "${SCRIPT_DIR}/train_qwen36_27b_paper2arm_distill_megatron.sh"
status=$?

if [[ "${status}" -eq 0 ]]; then
  log "Megatron full SFT completed successfully."
  exit 0
fi

log "Megatron full SFT exited with status ${status}. Checking for OOM signature."
latest_dir="$(ls -td "${MEGATRON_LOG_ROOT}"/qwen36-27b-paper2arm-distill-megatron-tp8-* 2>/dev/null | head -n1 || true)"
latest_log="${latest_dir}/train.log"

if [[ -f "${latest_log}" ]] && grep -Eiq 'out of memory|CUDA.*OOM|CUDA error: out of memory|CUDACachingAllocator|alloc.*CUDA' "${latest_log}"; then
  log "OOM detected in ${latest_log}; starting HF/DeepSpeed zero3_offload fallback."
  DEEPSPEED=zero3_offload bash "${SCRIPT_DIR}/train_qwen36_27b_paper2arm_distill_full_sp4.sh"
  exit $?
fi

log "No OOM signature found; leaving failure for inspection. Latest log: ${latest_log}"
exit "${status}"
