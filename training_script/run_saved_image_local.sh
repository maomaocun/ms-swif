#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-verify}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  IMAGE_URI=registry/namespace/image:tag bash training_script/run_saved_image_local.sh verify
  IMAGE_URI=registry/namespace/image:tag bash training_script/run_saved_image_local.sh dry-run
  IMAGE_URI=registry/namespace/image:tag bash training_script/run_saved_image_local.sh smoke
  IMAGE_URI=registry/namespace/image:tag bash training_script/run_saved_image_local.sh train

Optional SFT_* variables:
  SFT_MODEL_PATH
  SFT_DATASET_PATH
  SFT_OUTPUT_ROOT
  SFT_LOG_ROOT
  SFT_CACHE_ROOT
  SFT_REPORT_TO
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "${IMAGE_URI:-}" ]]; then
  echo "[ERROR] set IMAGE_URI before running." >&2
  usage >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is required for local image runs." >&2
  exit 1
fi

docker_args=(
  run --rm -it
  --gpus all
  --ipc=host
  --shm-size="${DOCKER_SHM_SIZE:-256g}"
  -v /mnt:/mnt
  -w "${REPO_DIR}"
)

add_env() {
  local container_name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    docker_args+=(-e "${container_name}=${value}")
  fi
}

add_env "MODEL_PATH" "${SFT_MODEL_PATH:-${MODEL_PATH:-}}"
add_env "DATASET_PATH" "${SFT_DATASET_PATH:-${DATASET_PATH:-}}"
add_env "OUTPUT_ROOT" "${SFT_OUTPUT_ROOT:-${OUTPUT_ROOT:-}}"
add_env "LOG_ROOT" "${SFT_LOG_ROOT:-${LOG_ROOT:-}}"
add_env "CACHE_ROOT" "${SFT_CACHE_ROOT:-${CACHE_ROOT:-}}"
add_env "REPORT_TO" "${SFT_REPORT_TO:-${REPORT_TO:-none}}"
add_env "WANDB_PROJECT" "${SFT_WANDB_PROJECT:-${WANDB_PROJECT:-}}"
add_env "WANDB_ENTITY" "${SFT_WANDB_ENTITY:-${WANDB_ENTITY:-}}"
add_env "WANDB_API_KEY_FILE" "${SFT_WANDB_API_KEY_FILE:-${WANDB_API_KEY_FILE:-}}"
add_env "NPROC_PER_NODE" "${SFT_NPROC_PER_NODE:-${NPROC_PER_NODE:-}}"
add_env "TENSOR_MODEL_PARALLEL_SIZE" "${SFT_TENSOR_MODEL_PARALLEL_SIZE:-${TENSOR_MODEL_PARALLEL_SIZE:-}}"
add_env "PIPELINE_MODEL_PARALLEL_SIZE" "${SFT_PIPELINE_MODEL_PARALLEL_SIZE:-${PIPELINE_MODEL_PARALLEL_SIZE:-}}"
add_env "CONTEXT_PARALLEL_SIZE" "${SFT_CONTEXT_PARALLEL_SIZE:-${CONTEXT_PARALLEL_SIZE:-}}"
add_env "GLOBAL_BATCH_SIZE" "${SFT_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-}}"
add_env "MICRO_BATCH_SIZE" "${SFT_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE:-}}"
add_env "MAX_LENGTH" "${SFT_MAX_LENGTH:-${MAX_LENGTH:-}}"
add_env "LINEAR_CE_CHUNK_SIZE" "${SFT_LINEAR_CE_CHUNK_SIZE:-${LINEAR_CE_CHUNK_SIZE:-}}"

echo "[INFO] image=${IMAGE_URI}"
echo "[INFO] mode=${MODE}"
echo "[INFO] repo=${REPO_DIR}"

exec docker "${docker_args[@]}" "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh "${MODE}"
