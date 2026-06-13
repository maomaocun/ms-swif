#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DLC_CONFIG="${BASE_DLC_CONFIG:-${REPO_DIR}/dlc/config_qwen36_27b_dsv4pro_distill.json}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-${REPO_DIR}/dlc/submit_qwen36_27b_dsv4pro_distill.sh}"

join_by_comma() {
  local IFS=","
  echo "$*"
}

append_env_if_set() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    GENERATED_DLC_ENVS+=("${name}=${value}")
  fi
}

if [[ -z "${IMAGE_URI:-}" ]]; then
  echo "[ERROR] set IMAGE_URI to the saved image URI before submitting." >&2
  echo "Example: IMAGE_URI=registry/namespace/image:tag bash training_script/submit_dlc_with_saved_image.sh" >&2
  exit 1
fi
if [[ ! -f "${BASE_DLC_CONFIG}" ]]; then
  echo "[ERROR] BASE_DLC_CONFIG does not exist: ${BASE_DLC_CONFIG}" >&2
  exit 1
fi
if [[ ! -x "${SUBMIT_SCRIPT}" && ! -f "${SUBMIT_SCRIPT}" ]]; then
  echo "[ERROR] SUBMIT_SCRIPT does not exist: ${SUBMIT_SCRIPT}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is required." >&2
  exit 1
fi

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/dlc-image-config.XXXXXX.json")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

jq --arg image "${IMAGE_URI}" '.dlc.worker_image = $image' "${BASE_DLC_CONFIG}" > "${TMP_CONFIG}"

GENERATED_DLC_ENVS=()
append_env_if_set "MODEL_PATH" "${SFT_MODEL_PATH:-}"
append_env_if_set "DATASET_PATH" "${SFT_DATASET_PATH:-}"
append_env_if_set "OUTPUT_ROOT" "${SFT_OUTPUT_ROOT:-}"
append_env_if_set "LOG_ROOT" "${SFT_LOG_ROOT:-}"
append_env_if_set "CACHE_ROOT" "${SFT_CACHE_ROOT:-}"
append_env_if_set "REPORT_TO" "${SFT_REPORT_TO:-}"
append_env_if_set "WANDB_PROJECT" "${SFT_WANDB_PROJECT:-}"
append_env_if_set "WANDB_ENTITY" "${SFT_WANDB_ENTITY:-}"
append_env_if_set "WANDB_API_KEY_FILE" "${SFT_WANDB_API_KEY_FILE:-}"
append_env_if_set "NPROC_PER_NODE" "${SFT_NPROC_PER_NODE:-}"
append_env_if_set "TENSOR_MODEL_PARALLEL_SIZE" "${SFT_TENSOR_MODEL_PARALLEL_SIZE:-}"
append_env_if_set "PIPELINE_MODEL_PARALLEL_SIZE" "${SFT_PIPELINE_MODEL_PARALLEL_SIZE:-}"
append_env_if_set "CONTEXT_PARALLEL_SIZE" "${SFT_CONTEXT_PARALLEL_SIZE:-}"
append_env_if_set "GLOBAL_BATCH_SIZE" "${SFT_GLOBAL_BATCH_SIZE:-}"
append_env_if_set "MICRO_BATCH_SIZE" "${SFT_MICRO_BATCH_SIZE:-}"
append_env_if_set "MAX_LENGTH" "${SFT_MAX_LENGTH:-}"
append_env_if_set "LINEAR_CE_CHUNK_SIZE" "${SFT_LINEAR_CE_CHUNK_SIZE:-}"

if [[ -z "${DLC_ENVS:-}" && "${#GENERATED_DLC_ENVS[@]}" -gt 0 ]]; then
  export DLC_ENVS="$(join_by_comma "${GENERATED_DLC_ENVS[@]}")"
fi

echo "[INFO] using image: ${IMAGE_URI}"
echo "[INFO] temporary DLC config: ${TMP_CONFIG}"
if [[ -n "${DLC_ENVS:-}" ]]; then
  echo "[INFO] DLC envs: ${DLC_ENVS}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  jq '.dlc.worker_image' "${TMP_CONFIG}"
fi

DLC_CONFIG="${TMP_CONFIG}" exec bash "${SUBMIT_SCRIPT}"
