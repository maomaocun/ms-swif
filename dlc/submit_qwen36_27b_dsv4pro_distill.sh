#!/usr/bin/env bash
set -euo pipefail

[[ "${DEBUG_SHELL_TRACE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLC_CONFIG="${DLC_CONFIG:-${SCRIPT_DIR}/config_qwen36_27b_dsv4pro_distill.json}"

if [[ ! -f "${DLC_CONFIG}" ]]; then
  echo "[ERROR] DLC config not found: ${DLC_CONFIG}" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is required to read ${DLC_CONFIG}" >&2
  exit 1
fi

dlc_cfg() { jq -r "$1" "${DLC_CONFIG}"; }
dlc_cfg_int() { jq -r "$1 // 0" "${DLC_CONFIG}"; }

DLC_BINARY="$(dlc_cfg '.dlc.binary // "/usr/local/bin/dlc"')"
JOB_NAME="${JOB_NAME:-$(dlc_cfg '.dlc.job_name // empty')}"
if [[ -z "${JOB_NAME}" || "${JOB_NAME}" == "null" ]]; then
  JOB_NAME="sft_qwen36_27b_dsv4pro_distill_$(date +%m%d_%H%M%S)"
fi

WORKER_SCRIPT="$(dlc_cfg '.dlc.run_script')"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/submit.log") 2>&1

WORKERS="$(dlc_cfg_int '.dlc.workers')"
WORKER_GPU="$(dlc_cfg_int '.dlc.worker_gpu')"
WORKER_CPU="$(dlc_cfg_int '.dlc.worker_cpu')"
WORKER_MEMORY="$(dlc_cfg '.dlc.worker_memory')"
WORKER_SHARED_MEMORY="$(dlc_cfg '.dlc.worker_shared_memory')"
PRIORITY="$(dlc_cfg_int '.dlc.priority')"
WORKER_IMAGE="$(dlc_cfg '.dlc.worker_image')"
DATA_SOURCE_URIS="$(dlc_cfg '.dlc.data_source_uris')"
RESOURCE_ID="$(dlc_cfg '.dlc.resource_id')"
WORKSPACE_ID="$(dlc_cfg '.dlc.workspace_id')"
VPC_ID="$(dlc_cfg '.dlc.vpc_id')"
SWITCH_ID="$(dlc_cfg '.dlc.switch_id')"
SECURITY_GROUP_ID="$(dlc_cfg '.dlc.security_group_id')"
EXTENDED_CIDRS="$(dlc_cfg '.dlc.extended_cidrs')"

COMMAND="bash ${WORKER_SCRIPT} ${LOG_DIR}"
DLC_ENVS="${DLC_ENVS:-}"

echo "[INFO] Submitting DLC job: ${JOB_NAME}"
echo "[INFO] Workers: ${WORKERS}, GPUs per worker: ${WORKER_GPU}"
echo "[INFO] Worker script: ${WORKER_SCRIPT}"
echo "[INFO] Log directory: ${LOG_DIR}"
echo "[INFO] Command: ${COMMAND}"
if [[ -n "${DLC_ENVS}" ]]; then
  echo "[INFO] Env overrides: ${DLC_ENVS}"
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1, skip DLC submission."
  exit 0
fi

submit_args=(
  --name="${JOB_NAME}" \
  --priority="${PRIORITY}" \
  --workers="${WORKERS}" \
  --worker_cpu="${WORKER_CPU}" \
  --worker_gpu="${WORKER_GPU}" \
  --worker_memory="${WORKER_MEMORY}" \
  --worker_shared_memory="${WORKER_SHARED_MEMORY}" \
  --worker_image="${WORKER_IMAGE}" \
  --data_source_uris="${DATA_SOURCE_URIS}" \
  --resource_id="${RESOURCE_ID}" \
  --workspace_id="${WORKSPACE_ID}" \
  --vpc_id="${VPC_ID}" \
  --switch_id="${SWITCH_ID}" \
  --security_group_id="${SECURITY_GROUP_ID}" \
  --extended_cidrs="${EXTENDED_CIDRS}" \
  --command="${COMMAND}"
)

if [[ -n "${DLC_ENVS}" ]]; then
  submit_args+=(--envs="${DLC_ENVS}")
fi

"${DLC_BINARY}" submit pytorchjob "${submit_args[@]}"

echo "[INFO] Job submitted successfully."
