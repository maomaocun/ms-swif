#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

DLC_LOG_DIR="${1:-/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/dlc/logs/manual}"
mkdir -p "${DLC_LOG_DIR}"

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WRAPPER_LOG="${DLC_LOG_DIR}/worker_rank${RANK}_${TIMESTAMP}.log"

exec > >(tee -a "${WRAPPER_LOG}") 2>&1

MS_SWIFT_DIR="/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift"
TRAIN_SCRIPT="${MS_SWIFT_DIR}/train_qwen36_27b_paper2arm_dsv4pro_distill_megatron.sh"
DATASET_PATH="${DATASET_PATH:-${MS_SWIFT_DIR}/data/processed/paper2arm_dsv4pro/paper2arm_dsv4pro_sft_reward_ge_0.6.jsonl}"
RUN_STAMP="$(printf '%s' "$(basename "${DLC_LOG_DIR}")" | tr -c '[:alnum:]_-' '-')"
RUN_NAME="${RUN_NAME:-qwen36-27b-paper2arm-dsv4pro-distill-megatron-dlc4-${RUN_STAMP}}"
CPFS_PYTHON="${CPFS_PYTHON:-${MS_SWIFT_DIR}/.uv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}"
MEGATRON_VENV="${MS_SWIFT_DIR}/.venv-megatron"

echo "[INFO] DLC worker starting on $(hostname)"
echo "[INFO] RANK=${RANK} WORLD_SIZE=${WORLD_SIZE} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[INFO] DLC_LOG_DIR=${DLC_LOG_DIR}"

if [[ ! -x "${CPFS_PYTHON}" ]]; then
  echo "[ERROR] CPFS Python runtime not found: ${CPFS_PYTHON}" >&2
  exit 1
fi
ln -sfn "${CPFS_PYTHON}" "${MEGATRON_VENV}/bin/python"
ln -sfn python "${MEGATRON_VENV}/bin/python3"
ln -sfn python "${MEGATRON_VENV}/bin/python3.12"
echo "[INFO] Megatron venv python: $("${MEGATRON_VENV}/bin/python" --version 2>&1)"

if [[ ! -x "${MEGATRON_VENV}/bin/megatron" ]]; then
  echo "[ERROR] Megatron CLI not found: ${MEGATRON_VENV}/bin/megatron" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[ERROR] DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] TRAIN_SCRIPT does not exist: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

export NNODES="${WORLD_SIZE}"
export NODE_RANK="${RANK}"
export NPROC_PER_NODE
export MASTER_ADDR
export MASTER_PORT
export RUN_NAME
export DATASET_PATH
export MODEL_PATH="${MODEL_PATH:-/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${MS_SWIFT_DIR}/outputs/qwen36-27b-paper2arm-dsv4pro-distill-megatron-dlc}"
export LOG_ROOT="${LOG_ROOT:-${MS_SWIFT_DIR}/logs/qwen36-27b-paper2arm-dsv4pro-distill-megatron-dlc}"
export LOG_DIR="${LOG_DIR:-${DLC_LOG_DIR}}"
export LOG_FILE="${LOG_FILE:-${DLC_LOG_DIR}/train_node${RANK}.log}"
export CACHE_ROOT="${CACHE_ROOT:-${MS_SWIFT_DIR}/cache/megatron-qwen36-27b-paper2arm-dsv4pro-distill}"
export LOCAL_CACHE_ROOT="${DLC_LOCAL_CACHE_ROOT:-/tmp/msw-r${RANK}}"
export HF_HOME="${DLC_HF_HOME:-${LOCAL_CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${DLC_HF_DATASETS_CACHE:-${LOCAL_CACHE_ROOT}/ds}"
export MODELSCOPE_CACHE="${DLC_MODELSCOPE_CACHE:-${LOCAL_CACHE_ROOT}/ms}"
export TRITON_CACHE_DIR="${DLC_TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${DLC_TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_ext}"
export TMPDIR="${DLC_TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${TMPDIR}"

export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-train_distillation}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT}}"
export WANDB_ENTITY="${WANDB_ENTITY:-yfgao-sjtu}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-distillation}"
export WANDB_TAGS="${WANDB_TAGS:-innovator,distillation,dlc4,qwen36-27b,dsv4pro}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export WANDB_EXP_NAME="${WANDB_EXP_NAME:-${WANDB_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/wandb}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/mnt/cpfs/yangyicun/.secrets/wandb_api_key}"

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

if [[ "${REPORT_TO}" == *wandb* ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[ERROR] REPORT_TO includes wandb, but WANDB_API_KEY is unset and ${WANDB_API_KEY_FILE} is missing." >&2
    exit 1
  fi
  mkdir -p "${WANDB_DIR}"
  if ! "${MEGATRON_VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util

if importlib.util.find_spec("wandb") is None:
    raise SystemExit("wandb package is not importable")
print("[INFO] wandb package version:", metadata.version("wandb"))
PY
  then
    echo "[ERROR] wandb is not importable from ${MEGATRON_VENV}/bin/python" >&2
    exit 1
  fi
fi

# Qwen3.6 gated_delta_net currently blocks CP>1 in Megatron Core. On 4 DLC
# workers, TP8 + PP4 consumes all 32 GPUs and gives DP=1.
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-4}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-262144}"
export TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-delete}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
export LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}"
export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"

# Keep optimizer states offloaded by default. The no-offload DLC run OOMed at
# the first optimizer step on A100 80G.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1}"
export USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD="${USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD:-false}"
export OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-false}"
export PIN_CPU_GRADS="${PIN_CPU_GRADS:-true}"
export PIN_CPU_PARAMS="${PIN_CPU_PARAMS:-true}"

export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${no_proxy:-}"
export NO_PROXY="${no_proxy}"

echo "[INFO] Effective training env:"
echo "  NNODES=${NNODES}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  TP/PP/CP=${TENSOR_MODEL_PARALLEL_SIZE}/${PIPELINE_MODEL_PARALLEL_SIZE}/${CONTEXT_PARALLEL_SIZE}"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "  DATASET_NUM_PROC=${DATASET_NUM_PROC}"
echo "  OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD}"
echo "  OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION}"
echo "  REPORT_TO=${REPORT_TO}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"
echo "  WANDB_NAME=${WANDB_NAME}"
echo "  WANDB_DIR=${WANDB_DIR}"
echo "  LOCAL_CACHE_ROOT=${LOCAL_CACHE_ROOT}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "  TMPDIR=${TMPDIR}"
echo "  LOG_FILE=${LOG_FILE}"

cd "${MS_SWIFT_DIR}"
exec bash "${TRAIN_SCRIPT}"
