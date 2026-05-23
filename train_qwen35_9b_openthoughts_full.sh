#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B}"
DATASET_PATH="${DATASET_PATH:-/mnt/cpfs/yangyicun/data/datasets/openthoughts_prepared/openthoughts_sft_400k_filtered.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft}"
SMOKE="${SMOKE:-0}"
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    RUN_NAME="qwen35-9b-openthoughts-smoke-$(date +%Y%m%d-%H%M%S)"
  else
    RUN_NAME="qwen35-9b-openthoughts-$(date +%Y%m%d-%H%M%S)"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

CACHE_ROOT="${CACHE_ROOT:-/mnt/cpfs/yangyicun/cache/ms-swift-openthoughts}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/ms-swift-openthoughts-${USER:-root}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

TUNER_TYPE="${TUNER_TYPE:-full}"
DEEPSPEED="${DEEPSPEED:-zero2}"
if [[ "${SMOKE}" == "1" ]]; then
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"
  PADDING_FREE="${PADDING_FREE:-false}"
  USE_LOGITS_TO_KEEP="${USE_LOGITS_TO_KEEP:-true}"
else
  # The OpenThoughts filtered data is mostly ~16k-18k tokens; 4k truncates almost all samples.
  # Benchmarks on 8 GPUs showed SP=1 + logits_to_keep is faster than SP=2/4 here.
  MAX_LENGTH="${MAX_LENGTH:-18432}"
  SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"
  PADDING_FREE="${PADDING_FREE:-true}"
  USE_LOGITS_TO_KEEP="${USE_LOGITS_TO_KEEP:-true}"
fi
if [[ "${SMOKE}" == "1" ]]; then
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
else
  PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
fi
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

# Fast startup path: avoid full upfront tokenization and length sorting unless explicitly enabled.
LAZY_TOKENIZE="${LAZY_TOKENIZE:-true}"
if [[ "${SMOKE}" == "1" ]]; then
  STREAMING="${STREAMING:-true}"
else
  STREAMING="${STREAMING:-false}"
fi
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
PACKING="${PACKING:-false}"
PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
if [[ "${STREAMING}" == "true" && "${LAZY_TOKENIZE}" == "true" ]]; then
  LAZY_TOKENIZE=false
fi

SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
if [[ "${SMOKE}" == "1" ]]; then
  SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
else
  SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
fi
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
REPORT_TO="${REPORT_TO:-none}"
DDP_TIMEOUT="${DDP_TIMEOUT:-3600000}"
DISABLE_TQDM="${DISABLE_TQDM:-true}"
CELOSS_PARALLEL_SIZE="${CELOSS_PARALLEL_SIZE:-2048}"

MAX_STEPS="${MAX_STEPS:-}"
if [[ "${SMOKE}" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-1}"
  NUM_TRAIN_EPOCHS=1
  GRADIENT_ACCUMULATION_STEPS="${SMOKE_GRADIENT_ACCUMULATION_STEPS:-1}"
  EVAL_STRATEGY=no
  SPLIT_DATASET_RATIO=0
  LOGGING_STEPS=1
fi

if [[ ! -x "${SCRIPT_DIR}/.venv/bin/swift" ]]; then
  echo "ERROR: ${SCRIPT_DIR}/.venv/bin/swift is not executable" >&2
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

export PATH="${SCRIPT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CELOSS_PARALLEL_SIZE

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

cmd=(
  swift sft
  --model "${MODEL_PATH}"
  --model_type qwen3_5
  --template qwen3_5
  --tuner_type "${TUNER_TYPE}"
  --dataset "${DATASET_PATH}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --eval_strategy "${EVAL_STRATEGY}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_only_model "${SAVE_ONLY_MODEL}"
  --logging_steps "${LOGGING_STEPS}"
  --logging_first_step true
  --output_dir "${OUTPUT_DIR}"
  --deepspeed "${DEEPSPEED}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --lazy_tokenize "${LAZY_TOKENIZE}"
  --streaming "${STREAMING}"
  --group_by_length "${GROUP_BY_LENGTH}"
  --packing "${PACKING}"
  --packing_num_proc "${PACKING_NUM_PROC}"
  --padding_free "${PADDING_FREE}"
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}"
  --use_logits_to_keep "${USE_LOGITS_TO_KEEP}"
  --attn_impl "${ATTN_IMPL}"
  --use_liger_kernel "${USE_LIGER_KERNEL}"
  --report_to "${REPORT_TO}"
  --ddp_timeout "${DDP_TIMEOUT}"
  --disable_tqdm "${DISABLE_TQDM}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  cmd+=(--max_steps "${MAX_STEPS}")
fi

{
  echo "Run name: ${RUN_NAME}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "Model: ${MODEL_PATH}"
  echo "Dataset: ${DATASET_PATH}"
  echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Cache root: ${CACHE_ROOT}"
  echo "Local cache root: ${LOCAL_CACHE_ROOT}"
  echo
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
} | tee -a "${LOG_FILE}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
