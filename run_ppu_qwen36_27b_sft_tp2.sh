#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /usr/local/PPU_SDK/envsetup.sh
  set -u
fi

export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B}"
export MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
export TEMPLATE="${TEMPLATE:-qwen3_5}"
export DATASET_PATH="${DATASET_PATH:-/mnt/cpfs/yangyicun/data/datasets/ziyue_swe-smith-2k/ziyue_swe_smith_success_messages.jsonl}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"

export MAX_LENGTH="${MAX_LENGTH:-8192}"
export TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-100}"
export TRAIN_ITERS="${TRAIN_ITERS:-}"

export RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-}"

export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
export PIN_CPU_GRADS="${PIN_CPU_GRADS:-false}"
export PIN_CPU_PARAMS="${PIN_CPU_PARAMS:-false}"
export MAIN_GRADS_DTYPE="${MAIN_GRADS_DTYPE:-fp32}"
export MAIN_PARAMS_DTYPE="${MAIN_PARAMS_DTYPE:-fp32}"
export EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-fp32}"
export EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-fp32}"

export PACKING="${PACKING:-false}"
export PADDING_FREE="${PADDING_FREE:-true}"
export LAZY_TOKENIZE="${LAZY_TOKENIZE:-false}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
export DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"

export ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
export GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-false}"
export APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-false}"
export TP_COMM_OVERLAP="${TP_COMM_OVERLAP:-false}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"
export OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP="${OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP:-false}"
export OVERLAP_P2P_COMM="${OVERLAP_P2P_COMM:-true}"

export LINEAR_CE_IMPL="${LINEAR_CE_IMPL:-torch}"
export LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-0}"
export LINEAR_CE_DEBUG="${LINEAR_CE_DEBUG:-1}"
export LINEAR_DECOUPLED_IN_PROJ="${LINEAR_DECOUPLED_IN_PROJ:-false}"
export USE_MCORE_GDN="${USE_MCORE_GDN:-true}"

export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-1000000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-true}"
export ASYNC_SAVE="${ASYNC_SAVE:-false}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-agent-distillation-training}"
export WANDB_ENTITY="${WANDB_ENTITY:-infinite-frontier}"

export REALTIME_LOG_MODE="${REALTIME_LOG_MODE:-direct}"
export DIRECT_TORCHRUN="${DIRECT_TORCHRUN:-true}"
export MEGATRON_PYTHON="${MEGATRON_PYTHON:-/usr/local/bin/python3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/yangyicun/data/agent_checkpoints/qwen36-27b-paper2arm-distill-megatron}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/qwen36-27b-paper2arm-distill-megatron}"
export RUN_NAME="${RUN_NAME:-qwen36-27b-paper2arm-distill-megatron-ppu-tp2-pp1-8k-gbs128-ep100-$(date +%Y%m%d-%H%M%S)}"

exec bash "${SCRIPT_DIR}/train_qwen36_27b_paper2arm_distill_megatron.sh"
