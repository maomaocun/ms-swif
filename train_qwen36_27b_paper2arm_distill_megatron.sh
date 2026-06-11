#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/megatron_env.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
TEMPLATE="${TEMPLATE:-qwen3_5}"
MCORE_MODEL_PATH="${MCORE_MODEL_PATH:-}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/data/processed/paper2arm_qwen37_max/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl}"
CACHED_DATASET="${CACHED_DATASET:-}"
CACHED_VAL_DATASET="${CACHED_VAL_DATASET:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/yangyicun/data/agent_checkpoints/qwen36-27b-paper2arm-distill-megatron}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/qwen36-27b-paper2arm-distill-megatron}"
SMOKE="${SMOKE:-0}"
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    RUN_NAME="qwen36-27b-paper2arm-distill-megatron-tp8-smoke-$(date +%Y%m%d-%H%M%S)"
  else
    RUN_NAME="qwen36-27b-paper2arm-distill-megatron-tp8-$(date +%Y%m%d-%H%M%S)"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

CACHE_ROOT="${CACHE_ROOT:-${SCRIPT_DIR}/cache/megatron-qwen36-27b-paper2arm-distill}"
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${SCRIPT_DIR}/local_cache/megatron-qwen36-27b-paper2arm-distill-${USER:-root}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TUNER_TYPE="${TUNER_TYPE:-full}"
LOSS_SCALE="${LOSS_SCALE:-default}"
# Qwen3.5/3.6 uses gated_delta_net. Keep CP disabled by default until the
# mcore_bridge CP path is smoke-tested for this exact model/loss setup.
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
ALLOW_EXPERIMENTAL_CP="${ALLOW_EXPERIMENTAL_CP:-false}"
CP_GDN_HEAD_CHECK="${CP_GDN_HEAD_CHECK:-true}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE="${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE:-}"
MICROBATCH_GROUP_SIZE_PER_VP_STAGE="${MICROBATCH_GROUP_SIZE_PER_VP_STAGE:-}"
PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT:-}"
DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}"
DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-}"
ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT="${ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT:-false}"
ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT="${ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT:-false}"
OVERLAP_P2P_COMM="${OVERLAP_P2P_COMM:-true}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD="${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD:-false}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-false}"
PIN_CPU_GRADS="${PIN_CPU_GRADS:-true}"
PIN_CPU_PARAMS="${PIN_CPU_PARAMS:-true}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"

TRAIN_ITERS_WAS_SET=0
NUM_TRAIN_EPOCHS_WAS_SET=0
[[ -v TRAIN_ITERS ]] && TRAIN_ITERS_WAS_SET=1
[[ -v NUM_TRAIN_EPOCHS ]] && NUM_TRAIN_EPOCHS_WAS_SET=1

if [[ "${SMOKE}" == "1" ]]; then
  MAX_LENGTH="${MAX_LENGTH:-4096}"
  TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}"
  TRAIN_ITERS="${TRAIN_ITERS:-1}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
  SAVE_STEPS="${SAVE_STEPS:-1000000}"
else
  MAX_LENGTH="${MAX_LENGTH:-262144}"
  TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-delete}"
  TRAIN_ITERS="${TRAIN_ITERS:-}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
  SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
  SAVE_STEPS="${SAVE_STEPS:-50}"
fi
if [[ "${TRAIN_ITERS_WAS_SET}" == "1" && -n "${TRAIN_ITERS}" && "${NUM_TRAIN_EPOCHS_WAS_SET}" == "0" ]]; then
  NUM_TRAIN_EPOCHS=""
fi

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
PACKING="${PACKING:-false}"
PACKING_LENGTH="${PACKING_LENGTH:-}"
PADDING_FREE="${PADDING_FREE:-true}"
LAZY_TOKENIZE="${LAZY_TOKENIZE:-false}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
if [[ -z "${RECOMPUTE_METHOD+x}" ]]; then
  if [[ "${RECOMPUTE_GRANULARITY}" == "full" ]]; then
    RECOMPUTE_METHOD="uniform"
  else
    RECOMPUTE_METHOD=""
  fi
fi
if [[ -z "${RECOMPUTE_NUM_LAYERS+x}" ]]; then
  if [[ "${RECOMPUTE_GRANULARITY}" == "full" ]]; then
    RECOMPUTE_NUM_LAYERS="1"
  else
    RECOMPUTE_NUM_LAYERS=""
  fi
fi
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-}"
CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-false}"
OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP:-false}"
GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-false}"
DATA_SHARDING="${DATA_SHARDING:-false}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
FP8_FORMAT="${FP8_FORMAT:-}"
FP8_RECIPE="${FP8_RECIPE:-delayed}"
FP8_PARAM_GATHER="${FP8_PARAM_GATHER:-false}"
ASYNC_SAVE="${ASYNC_SAVE:-false}"
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-true}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DDP_TIMEOUT="${DDP_TIMEOUT:-3600000}"
REPORT_TO="${REPORT_TO:-tensorboard}"
export WANDB_PROJECT="${WANDB_PROJECT:-agent-distillation-training}"
export WANDB_ENTITY="${WANDB_ENTITY:-infinite-frontier}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT}}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-${WANDB_NAME:-${RUN_NAME}}}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/mnt/cpfs/yangyicun/.secrets/wandb_api_key}"
DIRECT_TORCHRUN="${DIRECT_TORCHRUN:-true}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-${BENCHMARK_SKIP_FINAL_SAVE:-false}}"
MEGATRON_PYTHON="${MEGATRON_PYTHON:-${SCRIPT_DIR}/.venv-megatron/bin/python}"
MEGATRON_SFT_ENTRY="${MEGATRON_SFT_ENTRY:-${SCRIPT_DIR}/swift/cli/_megatron/sft.py}"
USE_MCORE_GDN="${USE_MCORE_GDN:-true}"

restore_xtrace=0
if [[ $- == *x* ]]; then
  set +x
  restore_xtrace=1
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
  export WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
fi
if [[ "${restore_xtrace}" == "1" ]]; then
  set -x
fi
unset restore_xtrace
if [[ "${REPORT_TO}" == *wandb* && -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: REPORT_TO includes wandb, but WANDB_API_KEY is unset and ${WANDB_API_KEY_FILE} is missing." >&2
  exit 1
fi

TOTAL_MODEL_PARALLEL=$((TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if (( CONTEXT_PARALLEL_SIZE != 1 )); then
  case "${ALLOW_EXPERIMENTAL_CP,,}" in
    1|true|yes|y|on) ;;
    *)
      echo "ERROR: CP>1 is still experimental for Qwen3.5/3.6 gated_delta_net in this stack." >&2
      echo "       Set ALLOW_EXPERIMENTAL_CP=true to run a smoke test." >&2
      exit 1
      ;;
  esac
  case "${USE_MCORE_GDN,,}" in
    1|true|yes|y|on) ;;
    *)
      echo "ERROR: CP>1 requires USE_MCORE_GDN=true; the non-MCore GDN path asserts context_parallel_size == 1." >&2
      exit 1
      ;;
  esac
  case "${LINEAR_CE_CHUNK_SIZE,,}" in
    ""|0|false|none|off) ;;
    *)
      echo "ERROR: LINEAR_CE_CHUNK_SIZE=${LINEAR_CE_CHUNK_SIZE} is not CP-safe yet." >&2
      echo "       Use LINEAR_CE_CHUNK_SIZE=0 while validating CP; then patch chunked CE separately." >&2
      exit 1
      ;;
  esac
  ALLOW_MCORE_GDN_CP="${ALLOW_MCORE_GDN_CP:-true}"
  if [[ "${CP_GDN_HEAD_CHECK,,}" == "true" && -f "${MODEL_PATH}/config.json" ]]; then
    mapfile -t qwen36_head_info < <(python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
text = cfg.get("text_config") or cfg
for key in ("linear_num_key_heads", "linear_num_value_heads", "num_attention_heads", "num_key_value_heads"):
    value = text.get(key)
    print("" if value is None else value)
PY
)
    LINEAR_NUM_KEY_HEADS="${qwen36_head_info[0]:-}"
    if [[ "${LINEAR_NUM_KEY_HEADS}" =~ ^[0-9]+$ ]]; then
      GDN_HEAD_DIVISOR=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
      if (( LINEAR_NUM_KEY_HEADS < GDN_HEAD_DIVISOR || LINEAR_NUM_KEY_HEADS % GDN_HEAD_DIVISOR != 0 )); then
        echo "ERROR: GDN CP head split is invalid: linear_num_key_heads=${LINEAR_NUM_KEY_HEADS}, TP=${TENSOR_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}." >&2
        echo "       Need linear_num_key_heads % (TP*CP) == 0. For this 27B config, try TP=8 CP=2 or TP=4 CP=4." >&2
        exit 1
      fi
    fi
  fi
fi
WORLD_SIZE_TOTAL=$((NPROC_PER_NODE * NNODES))
if (( TOTAL_MODEL_PARALLEL > WORLD_SIZE_TOTAL || WORLD_SIZE_TOTAL % TOTAL_MODEL_PARALLEL != 0 )); then
  echo "ERROR: TP*PP*CP must divide total world size." >&2
  echo "       TP=${TENSOR_MODEL_PARALLEL_SIZE} PP=${PIPELINE_MODEL_PARALLEL_SIZE} CP=${CONTEXT_PARALLEL_SIZE} NNODES=${NNODES} NPROC=${NPROC_PER_NODE} WORLD=${WORLD_SIZE_TOTAL}" >&2
  exit 1
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE_TOTAL / TOTAL_MODEL_PARALLEL))
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DATA_PARALLEL_SIZE." >&2
  echo "       GLOBAL=${GLOBAL_BATCH_SIZE} MICRO=${MICRO_BATCH_SIZE} DP=${DATA_PARALLEL_SIZE}" >&2
  exit 1
fi
if [[ "${GROUP_BY_LENGTH}" == "true" && "${PADDING_FREE}" == "true" ]]; then
  echo "ERROR: Megatron group_by_length is not compatible with padding_free=true." >&2
  echo "       Set PADDING_FREE=false if you explicitly want to test GROUP_BY_LENGTH=true." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -n "${MCORE_MODEL_PATH}" && ! -d "${MCORE_MODEL_PATH}" ]]; then
  echo "ERROR: MCORE_MODEL_PATH does not exist: ${MCORE_MODEL_PATH}" >&2
  exit 1
fi
DATASET_FILE_PATH="${DATASET_PATH%%#*}"
if [[ -z "${DATASET_PATH}" && -z "${CACHED_DATASET}" ]]; then
  echo "ERROR: DATASET_PATH or CACHED_DATASET must be set." >&2
  exit 1
fi
if [[ -z "${CACHED_DATASET}" && -n "${DATASET_PATH}" && ! -f "${DATASET_FILE_PATH}" ]]; then
  echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ -n "${CACHED_DATASET}" ]]; then
  # shellcheck disable=SC2206
  cached_dataset_array=(${CACHED_DATASET})
  for cached_dataset_dir in "${cached_dataset_array[@]}"; do
    if [[ ! -d "${cached_dataset_dir}" ]]; then
      echo "ERROR: CACHED_DATASET path does not exist: ${cached_dataset_dir}" >&2
      exit 1
    fi
  done
fi
if [[ -n "${CACHED_VAL_DATASET}" ]]; then
  # shellcheck disable=SC2206
  cached_val_dataset_array=(${CACHED_VAL_DATASET})
  for cached_val_dataset_dir in "${cached_val_dataset_array[@]}"; do
    if [[ ! -d "${cached_val_dataset_dir}" ]]; then
      echo "ERROR: CACHED_VAL_DATASET path does not exist: ${cached_val_dataset_dir}" >&2
      exit 1
    fi
  done
fi
if [[ "${DIRECT_TORCHRUN}" == "true" ]]; then
  if [[ ! -x "${MEGATRON_PYTHON}" ]]; then
    echo "ERROR: MEGATRON_PYTHON is not executable: ${MEGATRON_PYTHON}" >&2
    exit 1
  fi
  if [[ ! -f "${MEGATRON_SFT_ENTRY}" ]]; then
    echo "ERROR: MEGATRON_SFT_ENTRY does not exist: ${MEGATRON_SFT_ENTRY}" >&2
    exit 1
  fi
fi
if [[ -z "${CACHED_DATASET}" && -n "${DATASET_PATH}" && "${SKIP_REASONING_DUP_CHECK:-0}" != "1" ]]; then
  python3 "${SCRIPT_DIR}/scripts/check_reasoning_content_duplicates.py" "${DATASET_FILE_PATH}" --max-report 20
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${CACHE_ROOT}" "${LOCAL_CACHE_ROOT}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export NNODES
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
export LINEAR_CE_CHUNK_SIZE
export USE_MCORE_GDN
export ALLOW_MCORE_GDN_CP="${ALLOW_MCORE_GDN_CP:-false}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

timestamp_output() {
  awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
}

training_args=(
  --model "${MODEL_PATH}"
  --model_type "${MODEL_TYPE}"
  --template "${TEMPLATE}"
  --tuner_type "${TUNER_TYPE}"
  --save_safetensors "${SAVE_SAFETENSORS}"
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
  --torch_dtype bfloat16
  --max_length "${MAX_LENGTH}"
  --truncation_strategy "${TRUNCATION_STRATEGY}"
  --loss_scale "${LOSS_SCALE}"
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context_parallel_size "${CONTEXT_PARALLEL_SIZE}"
  --account_for_embedding_in_pipeline_split "${ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT}"
  --account_for_loss_in_pipeline_split "${ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT}"
  --overlap_p2p_comm "${OVERLAP_P2P_COMM}"
  --micro_batch_size "${MICRO_BATCH_SIZE}"
  --global_batch_size "${GLOBAL_BATCH_SIZE}"
  --packing "${PACKING}"
  --padding_free "${PADDING_FREE}"
  --lazy_tokenize "${LAZY_TOKENIZE}"
  --recompute_granularity "${RECOMPUTE_GRANULARITY}"
  --finetune true
  --cross_entropy_loss_fusion "${CROSS_ENTROPY_LOSS_FUSION}"
  --gradient_accumulation_fusion "${GRADIENT_ACCUMULATION_FUSION}"
  --lr "${LR}"
  --lr_warmup_fraction "${LR_WARMUP_FRACTION}"
  --min_lr "${MIN_LR}"
  --output_dir "${OUTPUT_DIR}"
  --add_version false
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --skip_final_save "${SKIP_FINAL_SAVE}"
  --logging_steps "${LOGGING_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_pin_memory "${DATALOADER_PIN_MEMORY}"
  --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --no_save_optim true
  --no_save_rng true
  --async_save "${ASYNC_SAVE}"
  --sequence_parallel "${SEQUENCE_PARALLEL}"
  --optimizer_cpu_offload "${OPTIMIZER_CPU_OFFLOAD}"
  --use_precision_aware_optimizer "${USE_PRECISION_AWARE_OPTIMIZER}"
  --optimizer_offload_fraction "${OPTIMIZER_OFFLOAD_FRACTION}"
  --use_torch_optimizer_for_cpu_offload "${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD}"
  --overlap_cpu_optimizer_d2h_h2d "${OVERLAP_CPU_OPTIMIZER_D2H_H2D}"
  --pin_cpu_grads "${PIN_CPU_GRADS}"
  --pin_cpu_params "${PIN_CPU_PARAMS}"
  --main_grads_dtype "${MAIN_GRADS_DTYPE}"
  --main_params_dtype "${MAIN_PARAMS_DTYPE}"
  --exp_avg_dtype "${EXP_AVG_DTYPE}"
  --exp_avg_sq_dtype "${EXP_AVG_SQ_DTYPE}"
  --tp_comm_overlap "${TP_COMM_OVERLAP}"
  --overlap_grad_reduce "${OVERLAP_GRAD_REDUCE}"
  --overlap_param_gather "${OVERLAP_PARAM_GATHER}"
  --overlap_param_gather_with_optimizer_step "${OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP}"
  --data_sharding "${DATA_SHARDING}"
  --group_by_length "${GROUP_BY_LENGTH}"
  --attention_backend "${ATTENTION_BACKEND}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT_NAME}"
  --wandb_exp_name "${WANDB_EXP_NAME}"
  --ddp_timeout "${DDP_TIMEOUT}"
)

if [[ -n "${MCORE_MODEL_PATH}" ]]; then
  training_args+=(--mcore_model "${MCORE_MODEL_PATH}")
fi
if [[ -n "${CACHED_DATASET}" ]]; then
  training_args+=(--cached_dataset "${cached_dataset_array[@]}")
else
  training_args+=(--dataset "${DATASET_PATH}")
fi
if [[ -n "${CACHED_VAL_DATASET}" ]]; then
  training_args+=(--cached_val_dataset "${cached_val_dataset_array[@]}")
fi
if [[ -n "${PACKING_LENGTH}" ]]; then
  training_args+=(--packing_length "${PACKING_LENGTH}")
fi

if [[ -n "${RECOMPUTE_METHOD}" ]]; then
  training_args+=(--recompute_method "${RECOMPUTE_METHOD}")
fi
if [[ -n "${RECOMPUTE_NUM_LAYERS}" ]]; then
  training_args+=(--recompute_num_layers "${RECOMPUTE_NUM_LAYERS}")
fi
if [[ -n "${RECOMPUTE_MODULES}" ]]; then
  # shellcheck disable=SC2206
  recompute_modules_array=(${RECOMPUTE_MODULES})
  training_args+=(--recompute_modules "${recompute_modules_array[@]}")
fi
if [[ -n "${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE}" ]]; then
  training_args+=(--virtual_pipeline_model_parallel_size "${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE}")
fi
if [[ -n "${MICROBATCH_GROUP_SIZE_PER_VP_STAGE}" ]]; then
  training_args+=(--microbatch_group_size_per_vp_stage "${MICROBATCH_GROUP_SIZE_PER_VP_STAGE}")
fi
if [[ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT}" ]]; then
  training_args+=(--pipeline_model_parallel_layout "${PIPELINE_MODEL_PARALLEL_LAYOUT}")
fi
if [[ -n "${DECODER_FIRST_PIPELINE_NUM_LAYERS}" ]]; then
  training_args+=(--decoder_first_pipeline_num_layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}")
fi
if [[ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS}" ]]; then
  training_args+=(--decoder_last_pipeline_num_layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi
if [[ -n "${FP8_FORMAT}" ]]; then
  training_args+=(--fp8_format "${FP8_FORMAT}" --fp8_recipe "${FP8_RECIPE}" --fp8_param_gather "${FP8_PARAM_GATHER}")
fi
if [[ -n "${TRAIN_ITERS}" ]]; then
  training_args+=(--train_iters "${TRAIN_ITERS}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  training_args+=(--num_train_epochs "${NUM_TRAIN_EPOCHS}")
fi

if [[ "${DIRECT_TORCHRUN}" == "true" ]]; then
  cmd=(
    "${MEGATRON_PYTHON}" -u -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    "${MEGATRON_SFT_ENTRY}"
    "${training_args[@]}"
  )
else
  cmd=(megatron sft "${training_args[@]}")
fi

export SWIFT_RUN_NAME="${RUN_NAME}"
export SWIFT_OUTPUT_DIR="${OUTPUT_DIR}"
export SWIFT_LOG_DIR="${LOG_DIR}"
export SWIFT_LOG_FILE="${LOG_FILE}"
printf -v SWIFT_LAUNCH_COMMAND ' %q' "${cmd[@]}"
export SWIFT_LAUNCH_COMMAND="${SWIFT_LAUNCH_COMMAND# }"

{
  echo "Run name: ${RUN_NAME}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log dir: ${LOG_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "Step metrics: ${LOG_DIR}/logging.jsonl"
  echo "Run metadata: ${LOG_DIR}/run_metadata.json"
  echo "TensorBoard dir: ${LOG_DIR}/runs"
  echo "W&B local dir: ${LOG_DIR}/wandb"
  echo "SwanLab local dir: ${LOG_DIR}/swanlab"
  echo "Model: ${MODEL_PATH}"
  echo "Model type/template: ${MODEL_TYPE}/${TEMPLATE}"
  echo "MCore model: ${MCORE_MODEL_PATH:-<off>}"
  if [[ -n "${CACHED_DATASET}" ]]; then
    echo "Dataset: <skipped because CACHED_DATASET is set>"
  else
    echo "Dataset: ${DATASET_PATH}"
  fi
  echo "Cached dataset: ${CACHED_DATASET:-<off>}"
  echo "Cached val dataset: ${CACHED_VAL_DATASET:-<off>}"
  echo "Loss scale: ${LOSS_SCALE}"
  echo "Max length: ${MAX_LENGTH}"
  echo "Truncation strategy: ${TRUNCATION_STRATEGY}"
  echo "Train iters / epochs: ${TRAIN_ITERS:-<auto>} / ${NUM_TRAIN_EPOCHS:-<auto>}"
  echo "TP/PP/CP/DP: ${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}/${DATA_PARALLEL_SIZE}"
  echo "Pipeline extras: vpp=${VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE:-<off>} microbatch_group=${MICROBATCH_GROUP_SIZE_PER_VP_STAGE:-<off>} layout=${PIPELINE_MODEL_PARALLEL_LAYOUT:-<off>} first_layers=${DECODER_FIRST_PIPELINE_NUM_LAYERS:-<auto>} last_layers=${DECODER_LAST_PIPELINE_NUM_LAYERS:-<auto>} account_embedding=${ACCOUNT_FOR_EMBEDDING_IN_PIPELINE_SPLIT} account_loss=${ACCOUNT_FOR_LOSS_IN_PIPELINE_SPLIT} overlap_p2p=${OVERLAP_P2P_COMM}"
  echo "Optimizer CPU offload: ${OPTIMIZER_CPU_OFFLOAD} fraction=${OPTIMIZER_OFFLOAD_FRACTION} torch_optimizer=${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD} overlap_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D} pin_grads=${PIN_CPU_GRADS} pin_params=${PIN_CPU_PARAMS}"
  echo "Precision-aware optimizer: ${USE_PRECISION_AWARE_OPTIMIZER} main_grads=${MAIN_GRADS_DTYPE} main_params=${MAIN_PARAMS_DTYPE} exp_avg=${EXP_AVG_DTYPE} exp_avg_sq=${EXP_AVG_SQ_DTYPE}"
  echo "Cross entropy loss fusion: ${CROSS_ENTROPY_LOSS_FUSION}"
  echo "Chunked linear CE chunk size: ${LINEAR_CE_CHUNK_SIZE}"
  echo "Recompute: granularity=${RECOMPUTE_GRANULARITY} method=${RECOMPUTE_METHOD} num_layers=${RECOMPUTE_NUM_LAYERS} modules=${RECOMPUTE_MODULES:-<default>}"
  echo "Overlap: tp_comm=${TP_COMM_OVERLAP} grad_reduce=${OVERLAP_GRAD_REDUCE} param_gather=${OVERLAP_PARAM_GATHER} param_gather_with_step=${OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP}"
  echo "Data: data_sharding=${DATA_SHARDING} group_by_length=${GROUP_BY_LENGTH} packing=${PACKING} packing_length=${PACKING_LENGTH:-<auto>} padding_free=${PADDING_FREE} dataloader_pin_memory=${DATALOADER_PIN_MEMORY} persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
  echo "FP8: format=${FP8_FORMAT:-<off>} recipe=${FP8_RECIPE} param_gather=${FP8_PARAM_GATHER}"
  echo "Gradient accumulation fusion: ${GRADIENT_ACCUMULATION_FUSION}"
  echo "Async save: ${ASYNC_SAVE}"
  echo "Save safetensors: ${SAVE_SAFETENSORS}"
  echo "Save strategy/steps/limit: ${SAVE_STRATEGY}/${SAVE_STEPS}/${SAVE_TOTAL_LIMIT}"
  echo "Skip final save: ${SKIP_FINAL_SAVE}"
  echo "Report to: ${REPORT_TO}"
  echo "W&B entity/project/exp: ${WANDB_ENTITY}/${WANDB_PROJECT_NAME}/${WANDB_EXP_NAME}"
  echo "Launch mode: $([[ "${DIRECT_TORCHRUN}" == "true" ]] && echo direct_torchrun || echo megatron_cli)"
  echo "LR/min_lr/warmup: ${LR}/${MIN_LR}/${LR_WARMUP_FRACTION}"
  echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
  echo "NNODES/NODE_RANK: ${NNODES}/${NODE_RANK}"
  echo "MASTER_ADDR/MASTER_PORT: ${MASTER_ADDR}/${MASTER_PORT}"
  echo "Total world size: ${WORLD_SIZE_TOTAL}"
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
