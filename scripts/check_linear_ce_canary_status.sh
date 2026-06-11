#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-20260611_2148}"
NATIVE_JOB_ID="${NATIVE_JOB_ID:-dlc1b1rle24ctrkw}"
STREAMING_JOB_ID="${STREAMING_JOB_ID:-dlc1bbr6ruu5jlhk}"
NATIVE_LOG_DIR="${NATIVE_LOG_DIR:-${REPO_ROOT}/dlc/logs/linear_ce_native_32g_${STAMP}}"
STREAMING_LOG_DIR="${STREAMING_LOG_DIR:-${REPO_ROOT}/dlc/logs/linear_ce_streaming_32g_${STAMP}}"

print_job() {
  local label="$1"
  local job_id="$2"
  echo "== ${label}: ${job_id} =="
  if [[ -z "${job_id}" ]]; then
    echo "job_id=<unset>"
    return
  fi
  dlc get job "${job_id}" 2>/dev/null | python -c '
import json
import sys

d = json.load(sys.stdin)
print("name=", d.get("DisplayName"))
print("status=", d.get("Status"))
print("reason=", d.get("ReasonCode"), d.get("ReasonMessage"))
print("duration=", d.get("Duration"))
'
}

find_logging_jsonl() {
  local dir="$1"
  if [[ -f "${dir}/logging.jsonl" ]]; then
    echo "${dir}/logging.jsonl"
    return
  fi
  find "${dir}" -maxdepth 2 -type f -name logging.jsonl 2>/dev/null | sort | head -n 1
}

print_job native "${NATIVE_JOB_ID}"
print_job streaming "${STREAMING_JOB_ID}"

echo "== local logs =="
find "${NATIVE_LOG_DIR}" "${STREAMING_LOG_DIR}" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true

native_logging="$(find_logging_jsonl "${NATIVE_LOG_DIR}")"
streaming_logging="$(find_logging_jsonl "${STREAMING_LOG_DIR}")"

if [[ -n "${native_logging}" && -n "${streaming_logging}" ]]; then
  echo "== compare =="
  python "${REPO_ROOT}/scripts/compare_linear_ce_canary.py" "${native_logging}" "${streaming_logging}"
else
  echo "== compare =="
  echo "logging.jsonl not ready yet"
  echo "native_logging=${native_logging:-<missing>}"
  echo "streaming_logging=${streaming_logging:-<missing>}"
fi
