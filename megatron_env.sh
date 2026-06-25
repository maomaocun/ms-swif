#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file from a Megatron launch script." >&2
  exit 1
fi

MEGATRON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_VENV_SITE="${MEGATRON_ENV_DIR}/.venv-megatron/lib/python3.12/site-packages"
HF_VENV_SITE="${MEGATRON_ENV_DIR}/.venv/lib/python3.12/site-packages"
GLOBAL_VENV_SITE="/mnt/cpfs/yangyicun/.venv/lib/python3.12/site-packages"
PPU_SDK_ENV="${PPU_SDK_ENV:-/usr/local/PPU_SDK/envsetup.sh}"

if [[ -f "${PPU_SDK_ENV}" ]]; then
  # PPU PyTorch/Megatron uses the SDK CUDA-compatibility environment.
  # shellcheck source=/dev/null
  ppu_restore_nounset=0
  if [[ $- == *u* ]]; then
    set +u
    ppu_restore_nounset=1
  fi
  source "${PPU_SDK_ENV}" >/dev/null
  if [[ "${ppu_restore_nounset}" == "1" ]]; then
    set -u
  fi
  unset ppu_restore_nounset
fi

SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/local/bin/python}"
SYSTEM_SITE="$("${SYSTEM_PYTHON}" - <<'PY'
import site

paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"

if [[ -x "${MEGATRON_ENV_DIR}/.venv-megatron/bin/megatron" ]]; then
  export PATH="${MEGATRON_ENV_DIR}/.venv-megatron/bin:${PATH}"
elif ! command -v megatron >/dev/null 2>&1; then
  echo "ERROR: neither ${MEGATRON_ENV_DIR}/.venv-megatron/bin/megatron nor a global megatron command is executable" >&2
  return 1
fi

[[ -d "${MEGATRON_ENV_DIR}/.venv/bin" ]] && export PATH="${MEGATRON_ENV_DIR}/.venv/bin:${PATH}"

pythonpath_entries=("${MEGATRON_ENV_DIR}")
for site_dir in "${MEGATRON_VENV_SITE}" "${HF_VENV_SITE}" "${GLOBAL_VENV_SITE}"; do
  [[ -d "${site_dir}" ]] && pythonpath_entries+=("${site_dir}")
done
PYTHONPATH_JOINED="$(IFS=:; echo "${pythonpath_entries[*]}")"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH_JOINED}:${PYTHONPATH}"
else
  export PYTHONPATH="${PYTHONPATH_JOINED}"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
