#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PPU_SDK_ENV="${PPU_SDK_ENV:-/usr/local/PPU_SDK/envsetup.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "${PPU_SDK_ENV}" ]]; then
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
else
  echo "ERROR: PPU SDK envsetup not found: ${PPU_SDK_ENV}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -U \
  "megatron-core==0.17.0" \
  "mcore-bridge==1.4.0" \
  "peft==0.19.1" \
  "modelscope>=1.23" \
  wandb

"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m pip install --no-deps -e .

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import torch

for dist in ["torch", "megatron-core", "mcore-bridge", "modelscope", "wandb", "ms-swift"]:
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist}: missing")

print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count():
    print("device_0:", torch.cuda.get_device_name(0))
PY
