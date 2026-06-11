#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"

MAX_LENGTH="${MAX_LENGTH:-32768}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-4}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
REPORT_TO="${REPORT_TO:-tensorboard}"
STAGE_MEGATRON_TO_LOCAL="${STAGE_MEGATRON_TO_LOCAL:-true}"

DATASET_PATH="${DATASET_PATH:-${MS_SWIFT_DIR}/data/processed/paper2arm_dsv4pro/paper2arm_dsv4pro_sft_reward_ge_0.6.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/yangyicun/data/agent_checkpoints/qwen36-27b-linear-ce-canary-32g}"
LOG_ROOT="${LOG_ROOT:-${MS_SWIFT_DIR}/logs/qwen36-27b-linear-ce-canary-32g}"
CACHE_ROOT="${CACHE_ROOT:-${MS_SWIFT_DIR}/cache/megatron-qwen36-27b-linear-ce-canary-32g}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-linear-ce-canary-32g}"

COMMON_ENVS=(
  "MAX_LENGTH=${MAX_LENGTH}"
  "TRAIN_ITERS=${TRAIN_ITERS}"
  "NUM_TRAIN_EPOCHS="
  "TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
  "PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE}"
  "CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE}"
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
  "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  "SAVE_STRATEGY=steps"
  "SAVE_STEPS=1000000"
  "SKIP_FINAL_SAVE=true"
  "REPORT_TO=${REPORT_TO}"
  "WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "LOG_ROOT=${LOG_ROOT}"
  "CACHE_ROOT=${CACHE_ROOT}"
  "STAGE_MEGATRON_TO_LOCAL=${STAGE_MEGATRON_TO_LOCAL}"
  "SKIP_REASONING_DUP_CHECK=1"
  "LINEAR_CE_DEBUG=1"
)

if [[ -n "${CACHED_DATASET:-}" ]]; then
  COMMON_ENVS+=("CACHED_DATASET=${CACHED_DATASET}")
elif [[ -f "${DATASET_PATH}" ]]; then
  COMMON_ENVS+=("DATASET_PATH=${DATASET_PATH}")
else
  echo "[ERROR] DATASET_PATH missing and CACHED_DATASET unset: ${DATASET_PATH}" >&2
  exit 1
fi

join_by_comma() {
  local IFS=,
  echo "$*"
}

submit_case() {
  local key="$1"
  local linear_ce_impl="$2"
  local linear_ce_chunk_size="$3"
  local run_name="qwen36-27b-linear-ce-${key}-32g-${STAMP}"
  local job_name="sft_qwen36_27b_linear_ce_${key}_32g_${STAMP}"
  local log_dir="${SCRIPT_DIR}/logs/linear_ce_${key}_32g_${STAMP}"
  local envs

  envs="$(join_by_comma \
    "${COMMON_ENVS[@]}" \
    "RUN_NAME=${run_name}" \
    "WANDB_NAME=${run_name}" \
    "LINEAR_CE_IMPL=${linear_ce_impl}" \
    "LINEAR_CE_CHUNK_SIZE=${linear_ce_chunk_size}")"

  echo "[INFO] ${key}: RUN_NAME=${run_name}"
  echo "[INFO] ${key}: LOG_DIR=${log_dir}"
  echo "[INFO] ${key}: LINEAR_CE_IMPL=${linear_ce_impl} LINEAR_CE_CHUNK_SIZE=${linear_ce_chunk_size}"

  DRY_RUN="${DRY_RUN}" \
  JOB_NAME="${job_name}" \
  LOG_DIR="${log_dir}" \
  DLC_ENVS="${envs}" \
    bash "${SCRIPT_DIR}/submit_qwen36_27b_dsv4pro_distill.sh"
}

echo "[INFO] Submitting 32G linear CE canary pair. DRY_RUN=${DRY_RUN}"
echo "[INFO] Compare logs after completion with:"
echo "  python ${MS_SWIFT_DIR}/scripts/compare_linear_ce_canary.py \\"
echo "    ${SCRIPT_DIR}/logs/linear_ce_native_32g_${STAMP} \\"
echo "    ${SCRIPT_DIR}/logs/linear_ce_streaming_32g_${STAMP}"

submit_case "native" "torch" "0"
submit_case "streaming" "streaming" "2048"
