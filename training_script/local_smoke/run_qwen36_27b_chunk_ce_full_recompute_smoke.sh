#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_NAME="${RUN_NAME:-qwen36-27b-local-smoke-chunk-ce-full-recompute-${STAMP}}"

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B}"
export CACHED_DATASET="${CACHED_DATASET:-/mnt/cpfs/yangyicun/data/tokenized_datasets/reward>0/xqjin_terminal_termigen_20260608/qwen36_27b_sft_reward_gt0_max262k/train}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/local_smoke_debug}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/dlc/local_smoke_debug}"
export LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_node0.log}"
export CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/cache/local_smoke_debug}"
export LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/msw-local-smoke-${USER:-root}}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CACHED_DATASET}" ]]; then
  echo "Missing CACHED_DATASET: ${CACHED_DATASET}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${LOCAL_CACHE_ROOT}"

export REPORT_TO="${REPORT_TO:-tensorboard}"
export SWIFT_DISABLE_LOGGING_JSONL="${SWIFT_DISABLE_LOGGING_JSONL:-1}"
export SMOKE=1
export DIRECT_TORCHRUN="${DIRECT_TORCHRUN:-true}"
export MEGATRON_PYTHON="${MEGATRON_PYTHON:-/usr/local/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}"
export TRAIN_ITERS="${TRAIN_ITERS:-1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-1000000}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-true}"

export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-}"

export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
export LINEAR_CE_IMPL="${LINEAR_CE_IMPL:-torch}"
export LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}"
export LINEAR_CE_DEBUG="${LINEAR_CE_DEBUG:-1}"

export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
export USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD="${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD:-false}"
export PIN_CPU_GRADS="${PIN_CPU_GRADS:-true}"
export PIN_CPU_PARAMS="${PIN_CPU_PARAMS:-true}"
export VIT_ATTN_IMPL="${VIT_ATTN_IMPL:-flash_attention_3}"

echo "[INFO] local smoke log dir: ${LOG_DIR}"
echo "[INFO] local smoke log file: ${LOG_FILE}"
exec bash "${REPO_ROOT}/train_qwen36_27b_paper2arm_distill_megatron.sh"
