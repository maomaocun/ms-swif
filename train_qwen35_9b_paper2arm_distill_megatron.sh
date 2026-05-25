#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/megatron_env.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/qwen35-9b-paper2arm-distill-megatron}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/qwen35-9b-paper2arm-distill-megatron}"
SMOKE="${SMOKE:-0}"
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    RUN_NAME="qwen35-9b-paper2arm-distill-megatron-tp8-smoke-$(date +%Y%m%d-%H%M%S)"
  else
    RUN_NAME="qwen35-9b-paper2arm-distill-megatron-tp8-$(date +%Y%m%d-%H%M%S)"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

CACHE_ROOT="${CACHE_ROOT:-${SCRIPT_DIR}/cache/megatron-qwen35-9b-paper2arm-distill}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${SCRIPT_DIR}/local_cache/megatron-qwen35-9b-paper2arm-distill-${USER:-root}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

TUNER_TYPE="${TUNER_TYPE:-full}"
LOSS_SCALE="${LOSS_SCALE:-default}"
# Qwen3.5/3.6 uses Megatron Core gated_delta_net; Megatron Core 0.17 hard-asserts
# context_parallel_size == 1 for this attention variant. Use TP+sequence parallel instead.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"

if [[ "${SMOKE}" == "1" ]]; then
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}"
  TRAIN_ITERS="${TRAIN_ITERS:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
  SAVE_STEPS="${SAVE_STEPS:-1000000}"
else
  MAX_LENGTH="${MAX_LENGTH:-262144}"
  TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-delete}"
  TRAIN_ITERS="${TRAIN_ITERS:-}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
  SAVE_STEPS="${SAVE_STEPS:-50}"
fi

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
PACKING="${PACKING:-false}"
PADDING_FREE="${PADDING_FREE:-true}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-false}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DDP_TIMEOUT="${DDP_TIMEOUT:-3600000}"
REPORT_TO="${REPORT_TO:-tensorboard}"

TOTAL_MODEL_PARALLEL=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( CONTEXT_PARALLEL_SIZE != 1 )); then
  echo "ERROR: Qwen3.5/3.6 Megatron gated_delta_net does not support context_parallel_size > 1." >&2
  echo "       Current supported fallback is CP=1 with tensor/sequence parallelism; use the HF zero3_offload script if this OOMs." >&2
  exit 1
fi
if (( TOTAL_MODEL_PARALLEL > NPROC_PER_NODE || NPROC_PER_NODE % TOTAL_MODEL_PARALLEL != 0 )); then
  echo "ERROR: TP*PP*CP must divide NPROC_PER_NODE." >&2
  echo "       TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} NPROC=${NPROC_PER_NODE}" >&2
  exit 1
fi
DATA_PARALLEL_SIZE=$((NPROC_PER_NODE / TOTAL_MODEL_PARALLEL))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DATA_PARALLEL_SIZE." >&2
  echo "       GLOBAL=${GLOBAL_BATCH_SIZE} MICRO=${MICRO_BATCH_SIZE} DP=${DATA_PARALLEL_SIZE}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${CACHE_ROOT}" "${LOCAL_CACHE_ROOT}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

timestamp_output() {
  awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
}

cmd=(
  megatron sft
  --model "${MODEL_PATH}"
  --model_type qwen3_5
  --template qwen3_5
  --tuner_type "${TUNER_TYPE}"
  --save_safetensors true
  --dataset "${DATASET_PATH}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --loss_scale "${LOSS_SCALE}"
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --packing "${PACKING}"
  --padding_free "${PADDING_FREE}"
  --lazy_tokenize "${LAZY_TOKENIZE}"
  --recompute_granularity "${RECOMPUTE_GRANULARITY}"
  --recompute_method "${RECOMPUTE_METHOD}"
  --recompute_num_layers "${RECOMPUTE_NUM_LAYERS}"
  --finetune true
  --cross_entropy_loss_fusion "${CROSS_ENTROPY_LOSS_FUSION}"
  --lr "${LR}"
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --add_version false
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --logging_steps "${LOGGING_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --no_save_optim true
  --no_save_rng true
  --sequence_parallel "${SEQUENCE_PARALLEL}"
  --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}"
  --use_precision_aware_optimizer "${USE_PRECISION_AWARE_OPTIMIZER}"
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --attention_backend "${ATTENTION_BACKEND}"
  --report_to "${REPORT_TO}"
  --ddp_timeout "${DDP_TIMEOUT}"
)

if [[ -n "${TRAIN_ITERS}" ]]; then
  cmd+=(--train_iters "${TRAIN_ITERS}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  cmd+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

{
  echo "Run name: ${RUN_NAME}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "Model: ${MODEL_PATH}"
  echo "Dataset: ${DATASET_PATH}"
  echo "Loss scale: ${LOSS_SCALE}"
  echo "Max length: ${MAX_LENGTH}"
  echo "Truncation strategy: ${TRUNCATION_STRATEGY}"
  echo "TP/PP/CP/DP: ${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}/${DATA_PARALLEL_SIZE}"
  echo "Optimizer CPU offload: ${OPTIMIZER_CPU_OFFLOAD} fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
  echo "Cross entropy loss fusion: ${CROSS_ENTROPY_LOSS_FUSION}"
  echo "LR/min_lr/warmup: ${LR}/${MIN_LR}/${LR_WARMUP_FRACTION}"
  echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Cache root: ${CACHE_ROOT}"
  echo "Local cache root: ${LOCAL_CACHE_ROOT}"
  echo
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
} | timestamp_output | tee -a "${LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | timestamp_output | tee -a "${LOG_FILE}"
