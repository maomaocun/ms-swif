#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file from a Megatron launch script." >&2
  exit 1
fi

MEGATRON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_VENV_SITE="${MEGATRON_ENV_DIR}/.venv-megatron/lib/python3.12/site-packages"
HF_VENV_SITE="${MEGATRON_ENV_DIR}/.venv/lib/python3.12/site-packages"
GLOBAL_VENV_SITE="/mnt/cpfs/yangyicun/.venv/lib/python3.12/site-packages"
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

nvidia_search_roots=()
for site_dir in "${HF_VENV_SITE}" "${GLOBAL_VENV_SITE}" "${SYSTEM_SITE}"; do
  [[ -d "${site_dir}/nvidia" ]] && nvidia_search_roots+=("${site_dir}/nvidia")
done
NVIDIA_LIBS=""
if ((${#nvidia_search_roots[@]} > 0)); then
  NVIDIA_LIBS="$(find "${nvidia_search_roots[@]}" -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd: -)"
fi

[[ -d "${MEGATRON_ENV_DIR}/.venv/bin" ]] && export PATH="${MEGATRON_ENV_DIR}/.venv/bin:${PATH}"

pythonpath_entries=("${MEGATRON_ENV_DIR}")
for site_dir in "${MEGATRON_VENV_SITE}" "${HF_VENV_SITE}" "${GLOBAL_VENV_SITE}" "${SYSTEM_SITE}"; do
  [[ -d "${site_dir}" ]] && pythonpath_entries+=("${site_dir}")
done
PYTHONPATH_JOINED="$(IFS=:; echo "${pythonpath_entries[*]}")"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH_JOINED}:${PYTHONPATH}"
else
  export PYTHONPATH="${PYTHONPATH_JOINED}"
fi
if [[ -n "${NVIDIA_LIBS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# Transformer Engine loaded from the global environment uses these to find CUDA
# wheel libraries when the active interpreter is .venv-megatron.
for site_dir in "${HF_VENV_SITE}" "${SYSTEM_SITE}" "${GLOBAL_VENV_SITE}"; do
  if [[ -z "${CUDNN_HOME:-}" && -d "${site_dir}/nvidia/cudnn" ]]; then
    export CUDNN_HOME="${site_dir}/nvidia/cudnn"
  fi
  if [[ -z "${NVRTC_HOME:-}" && -d "${site_dir}/nvidia/cuda_nvrtc" ]]; then
    export NVRTC_HOME="${site_dir}/nvidia/cuda_nvrtc"
  fi
done

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
