#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen36-27b-paper2arm-distill-megatron-tp8-smoke-bf16-$(date +%Y%m%d-%H%M%S)}"
export FP8_FORMAT=""
export FP8_RECIPE="${FP8_RECIPE:-delayed}"

exec "${SCRIPT_DIR}/run_qwen36_27b_smoke.sh"
