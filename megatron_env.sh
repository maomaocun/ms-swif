#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file from a Megatron launch script." >&2
  exit 1
fi

MEGATRON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_VENV_SITE="${MEGATRON_ENV_DIR}/.venv-megatron/lib/python3.12/site-packages"
HF_VENV_SITE="${MEGATRON_ENV_DIR}/.venv/lib/python3.12/site-packages"
GLOBAL_VENV_SITE="/mnt/cpfs/yangyicun/.venv/lib/python3.12/site-packages"

if [[ ! -x "${MEGATRON_ENV_DIR}/.venv-megatron/bin/megatron" ]]; then
  echo "ERROR: ${MEGATRON_ENV_DIR}/.venv-megatron/bin/megatron is not executable" >&2
  return 1
fi
if [[ ! -d "${HF_VENV_SITE}" ]]; then
  echo "ERROR: HF/DeepSpeed dependency site-packages not found: ${HF_VENV_SITE}" >&2
  return 1
fi
NVIDIA_LIBS="$(
  find "${HF_VENV_SITE}/nvidia" "${GLOBAL_VENV_SITE}/nvidia" \
    -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd: -
)"

export PATH="${MEGATRON_ENV_DIR}/.venv-megatron/bin:${MEGATRON_ENV_DIR}/.venv/bin:${PATH}"
if [[ -d "${GLOBAL_VENV_SITE}" ]]; then
  export PYTHONPATH="${MEGATRON_ENV_DIR}:${MEGATRON_VENV_SITE}:${HF_VENV_SITE}:${GLOBAL_VENV_SITE}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${MEGATRON_ENV_DIR}:${MEGATRON_VENV_SITE}:${HF_VENV_SITE}:${PYTHONPATH:-}"
fi
if [[ -n "${NVIDIA_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# Transformer Engine loaded from the global environment uses these to find CUDA
# wheel libraries when the active interpreter is .venv-megatron.
export CUDNN_HOME="${CUDNN_HOME:-${HF_VENV_SITE}/nvidia/cudnn}"
export NVRTC_HOME="${NVRTC_HOME:-${HF_VENV_SITE}/nvidia/cuda_nvrtc}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
