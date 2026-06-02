#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE="${SMOKE:-0}"
export DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/data/processed/paper2arm_dsv4pro/paper2arm_dsv4pro_sft_reward_ge_0.6.jsonl}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/qwen36-27b-paper2arm-dsv4pro-distill-megatron}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/qwen36-27b-paper2arm-dsv4pro-distill-megatron}"
export CACHE_ROOT="${CACHE_ROOT:-${SCRIPT_DIR}/cache/megatron-qwen36-27b-paper2arm-dsv4pro-distill}"
export LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${SCRIPT_DIR}/local_cache/megatron-qwen36-27b-paper2arm-dsv4pro-distill-${USER:-root}}"

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    export RUN_NAME="qwen36-27b-paper2arm-dsv4pro-distill-megatron-tp8-smoke-$(date +%Y%m%d-%H%M%S)"
  else
    export RUN_NAME="qwen36-27b-paper2arm-dsv4pro-distill-megatron-tp8-$(date +%Y%m%d-%H%M%S)"
  fi
fi

exec bash "${SCRIPT_DIR}/train_qwen36_27b_paper2arm_distill_megatron.sh"
