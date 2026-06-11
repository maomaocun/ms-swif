#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-pair}"
STAMP="$(date +%Y%m%d-%H%M%S)"

MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/processed/paper2arm_qwen37_max/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl}"
BASE_MASTER_PORT="${MASTER_PORT:-29690}"

run_one() {
  local cp_size="$1"
  local tp_size="$2"
  local nproc="$3"
  local devices="$4"
  local port="$5"
  local run_name="${RUN_NAME_PREFIX:-qwen36-27b-sft-legacy-loss-smoke}-${STAMP}-cp${cp_size}-tp${tp_size}"
  local recompute_granularity="${RECOMPUTE_GRANULARITY:-full}"
  local recompute_method="${RECOMPUTE_METHOD:-}"
  local recompute_num_layers="${RECOMPUTE_NUM_LAYERS:-}"
  local recompute_modules="${RECOMPUTE_MODULES:-}"

  if [[ -z "${recompute_method}" && "${recompute_granularity}" == "full" ]]; then
    recompute_method="uniform"
  fi
  if [[ -z "${recompute_num_layers}" && "${recompute_granularity}" == "full" ]]; then
    recompute_num_layers="1"
  fi
  if [[ -z "${recompute_modules}" && "${recompute_granularity}" == "selective" ]]; then
    recompute_modules="core_attn"
  fi

  echo "[debug_sft_legacy_loss_smoke] run=${run_name} cp=${cp_size} tp=${tp_size} nproc=${nproc} devices=${devices}"
  SMOKE=1 \
  RUN_NAME="${run_name}" \
  MODEL_PATH="${MODEL_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  MAX_LENGTH="${MAX_LENGTH:-4096}" \
  TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-right}" \
  TENSOR_MODEL_PARALLEL_SIZE="${tp_size}" \
  CONTEXT_PARALLEL_SIZE="${cp_size}" \
  NPROC_PER_NODE="${nproc}" \
  CUDA_VISIBLE_DEVICES="${devices}" \
  MASTER_PORT="${port}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}" \
  TRAIN_ITERS="${TRAIN_ITERS:-1}" \
  RECOMPUTE_GRANULARITY="${recompute_granularity}" \
  RECOMPUTE_METHOD="${recompute_method}" \
  RECOMPUTE_NUM_LAYERS="${recompute_num_layers}" \
  RECOMPUTE_MODULES="${recompute_modules}" \
  LINEAR_CE_CHUNK_SIZE=0 \
  CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}" \
  GRADIENT_ACCUMULATION_FUSION="${GRADIENT_ACCUMULATION_FUSION:-false}" \
  APPLY_ROPE_FUSION="${APPLY_ROPE_FUSION:-false}" \
  PADDING_FREE="${PADDING_FREE:-true}" \
  LAZY_TOKENIZE="${LAZY_TOKENIZE:-false}" \
  DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}" \
  SAVE_STRATEGY=steps \
  SAVE_STEPS=1000000 \
  SKIP_FINAL_SAVE=true \
  REPORT_TO="${REPORT_TO:-tensorboard}" \
  ALLOW_EXPERIMENTAL_CP=true \
  USE_MCORE_GDN=true \
  SFT_CP_LOSS_DEBUG="${SFT_CP_LOSS_DEBUG:-1}" \
  GDN_CP_DEBUG="${GDN_CP_DEBUG:-0}" \
  SKIP_REASONING_DUP_CHECK="${SKIP_REASONING_DUP_CHECK:-1}" \
  bash "${REPO_ROOT}/train_qwen36_27b_paper2arm_distill_megatron.sh"
}

case "${MODE}" in
  cp1)
    run_one 1 "${TP_SIZE:-4}" "${NPROC_PER_NODE_CP1:-4}" "${CUDA_VISIBLE_DEVICES_CP1:-0,1,2,3}" "${BASE_MASTER_PORT}"
    ;;
  cp2)
    run_one 2 "${TP_SIZE:-4}" "${NPROC_PER_NODE_CP2:-8}" "${CUDA_VISIBLE_DEVICES_CP2:-0,1,2,3,4,5,6,7}" "${BASE_MASTER_PORT}"
    ;;
  pair)
    run_one 1 "${TP_SIZE:-4}" "${NPROC_PER_NODE_CP1:-4}" "${CUDA_VISIBLE_DEVICES_CP1:-0,1,2,3}" "${BASE_MASTER_PORT}"
    run_one 2 "${TP_SIZE:-4}" "${NPROC_PER_NODE_CP2:-8}" "${CUDA_VISIBLE_DEVICES_CP2:-0,1,2,3,4,5,6,7}" "$((BASE_MASTER_PORT + 1))"
    ;;
  *)
    echo "Usage: $0 [cp1|cp2|pair]" >&2
    exit 2
    ;;
esac
