#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen36-27b-paper2arm-distill-megatron-tp8-smoke-fp8-$(date +%Y%m%d-%H%M%S)}"
export FP8_FORMAT="${FP8_FORMAT:-hybrid}"
export FP8_RECIPE="${FP8_RECIPE:-delayed}"

# Qwen3.6 GDN in_proj has a TP-local output row count of 2060, which is not a
# valid Transformer Engine FP8 matrix dimension. Pad the outer module weight to
# 2064 and slice the virtual channels away after projection.
export MCORE_GDN_DISABLE_FP8_PROJ="${MCORE_GDN_DISABLE_FP8_PROJ:-false}"
export MCORE_GDN_PAD_TO_FP8_MULTIPLE="${MCORE_GDN_PAD_TO_FP8_MULTIPLE:-true}"

exec "${SCRIPT_DIR}/run_qwen36_27b_smoke.sh"
