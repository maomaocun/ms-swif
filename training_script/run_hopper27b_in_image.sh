#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-verify}"
REPO_DIR="${REPO_DIR:-/mnt/workspace/ms-swif}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train_qwen36_27b_paper2arm_distill_megatron.sh}"
VERIFY_SCRIPT="${VERIFY_SCRIPT:-${REPO_DIR}/scripts/verify_hopper_27b_ops.py}"

usage() {
  cat <<'EOF'
Usage:
  bash training_script/run_hopper27b_in_image.sh verify
  bash training_script/run_hopper27b_in_image.sh dry-run
  bash training_script/run_hopper27b_in_image.sh smoke
  bash training_script/run_hopper27b_in_image.sh train

Required for smoke/train unless the repo defaults exist:
  MODEL_PATH=/path/to/Qwen3.6-27B
  DATASET_PATH=/path/to/train.jsonl

Useful overrides:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  NPROC_PER_NODE=8
  FP8_FORMAT=hybrid
  FP8_RECIPE=delayed
  REPORT_TO=none|tensorboard|wandb
  OUTPUT_ROOT=/mnt/cpfs/.../checkpoints
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR does not exist: ${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

export SYSTEM_PYTHON="${SYSTEM_PYTHON:-${PYTHON_BIN}}"
export MEGATRON_PYTHON="${MEGATRON_PYTHON:-${PYTHON_BIN}}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FP8_FORMAT="${FP8_FORMAT:-hybrid}"
export FP8_RECIPE="${FP8_RECIPE:-delayed}"
export FP8_PARAM_GATHER="${FP8_PARAM_GATHER:-false}"
export CACHE_ROOT="${CACHE_ROOT:-${REPO_DIR}/cache/image-run-qwen36-27b}"
export LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-${REPO_DIR}/local_cache/image-run-qwen36-27b-${USER:-root}}"
export TMPDIR="${TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_CACHE_ROOT}/torch_extensions}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

echo "[INFO] repo=${REPO_DIR}"
echo "[INFO] python=$("${PYTHON_BIN}" --version 2>&1)"
echo "[INFO] mode=${MODE}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] fp8=${FP8_FORMAT}/${FP8_RECIPE} param_gather=${FP8_PARAM_GATHER}"

case "${MODE}" in
  verify)
    exec "${PYTHON_BIN}" "${VERIFY_SCRIPT}" --recommendations
    ;;
  dry-run)
    export DRY_RUN=1
    export REPORT_TO="${REPORT_TO:-none}"
    exec bash "${TRAIN_SCRIPT}"
    ;;
  smoke)
    export SMOKE=1
    export REPORT_TO="${REPORT_TO:-none}"
    exec bash "${TRAIN_SCRIPT}"
    ;;
  train)
    exec bash "${TRAIN_SCRIPT}"
    ;;
  *)
    echo "[ERROR] unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
