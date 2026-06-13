#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DLC_CONFIG="${BASE_DLC_CONFIG:-${REPO_DIR}/dlc/config_qwen36_27b_dsv4pro_distill.json}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-${REPO_DIR}/dlc/submit_qwen36_27b_dsv4pro_distill.sh}"

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

echo "[INFO] using image: ${IMAGE_URI}"
echo "[INFO] temporary DLC config: ${TMP_CONFIG}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  jq '.dlc.worker_image' "${TMP_CONFIG}"
fi

DLC_CONFIG="${TMP_CONFIG}" exec bash "${SUBMIT_SCRIPT}"
