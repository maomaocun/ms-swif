#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/model_cache/hf_home}"

MODEL_PATH="${MODEL_PATH:-/model_cache/qwen36_27b}"
DATASET_PATH="${DATASET_PATH:-/model_cache/coding_trajectory/processed/smoke_trajectory_one.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/model_cache/smoke_outputs}"
LOG_ROOT="${LOG_ROOT:-/model_cache/smoke_logs}"
CACHE_ROOT="${CACHE_ROOT:-/model_cache/ms_swift_cache}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/model_cache/ms_swift_local_cache}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
  echo "Download Qwen/Qwen3.6-27B to /model_cache/qwen36_27b first." >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Missing DATASET_PATH: ${DATASET_PATH}" >&2
  echo "Prepare one Swift-format jsonl sample first." >&2
  exit 1
fi

SMOKE=1 \
REPORT_TO="${REPORT_TO:-tensorboard}" \
MEGATRON_PYTHON="${MEGATRON_PYTHON:-/usr/local/bin/python}" \
MODEL_PATH="${MODEL_PATH}" \
DATASET_PATH="${DATASET_PATH}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
LOG_ROOT="${LOG_ROOT}" \
CACHE_ROOT="${CACHE_ROOT}" \
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT}" \
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}" \
VIT_ATTN_IMPL="${VIT_ATTN_IMPL:-sdpa}" \
PADDING_FREE="${PADDING_FREE:-false}" \
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}" \
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}" \
LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}" \
FP8_FORMAT="${FP8_FORMAT-}" \
FP8_RECIPE="${FP8_RECIPE:-delayed}" \
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-true}" \
SAVE_STEPS="${SAVE_STEPS:-1000000}" \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
