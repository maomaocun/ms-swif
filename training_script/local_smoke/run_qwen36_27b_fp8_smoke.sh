#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-qwen36-27b-paper2arm-distill-megatron-tp8-smoke-fp8-$(date +%Y%m%d-%H%M%S)}"
export FP8_FORMAT="${FP8_FORMAT:-hybrid}"
export FP8_RECIPE="${FP8_RECIPE:-delayed}"

# Qwen3.6 GDN can be unpadded inside Transformer Engine to a token count that is
# not divisible by 8. Keep the rest of the model in FP8 and run GDN projections
# in bf16 unless explicitly testing a fully aligned pure-FP8 GDN case.
export MCORE_GDN_DISABLE_FP8_PROJ="${MCORE_GDN_DISABLE_FP8_PROJ:-true}"

exec "${SCRIPT_DIR}/run_qwen36_27b_smoke.sh"
