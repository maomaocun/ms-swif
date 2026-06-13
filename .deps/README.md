# External Dependency Checkouts

This directory is intentionally outside the normal `ms-swift` source tree.
Only this README should be tracked by git. The dependency repositories under
`.deps/` are local source checkouts used to build/install the environment and
should be recreated from the commands below.

Current target environment:

- Python: `/usr/local/bin/python`
- CUDA/PyTorch: Torch `2.9.0+cu130`
- GPU architecture: Hopper/SM90, compile FA3 with `TORCH_CUDA_ARCH_LIST=9.0a`
- Installed packages of interest:
  - `flash_attn_3==3.0.0`
  - `flash-linear-attention==0.5.1`
  - `mcore-bridge==1.4.0`
  - `megatron-core==0.17.1`
  - `transformer-engine==2.9.0`
  - `triton==3.5.0`

## Recreate `.deps`

Run from the repository root:

```bash
mkdir -p .deps

git clone https://github.com/Dao-AILab/flash-attention.git .deps/flash-attention
git -C .deps/flash-attention checkout fc8cbad6b6b90220cf6ef8121c29e299a3ba7d9a

git clone https://github.com/fla-org/flash-linear-attention .deps/flash-linear-attention
git -C .deps/flash-linear-attention checkout c525f4957f11a6f197b52c0c222377446c3eab56

git clone https://github.com/modelscope/mcore-bridge.git .deps/mcore-bridge
git -C .deps/mcore-bridge checkout df06ab3b12ad1f84f18798026269ca83b12f8621

git clone --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git .deps/Megatron-LM
git -C .deps/Megatron-LM checkout aa1057124ef53e99cbfc799c916ad948824bbff6
```

If direct GitHub access is unavailable on the target machine, clone these
repositories on a machine that can access GitHub and copy the resulting
directories into `.deps/`.

## Install FlashAttention 3 for Hopper

Use the `hopper/` package from Dao-AILab FlashAttention. This environment was
built with the full FA3 Hopper build, not a reduced build.

```bash
export PYTHON_BIN=/usr/local/bin/python
export MAX_JOBS=64
export NVCC_THREADS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST='9.0a'
export TMPDIR=/usr/local/src/tmp
mkdir -p "${TMPDIR}"

cd .deps/flash-attention/hopper
"${PYTHON_BIN}" -m pip install -v --no-build-isolation .
```

Expected import check:

```bash
/usr/local/bin/python - <<'PY'
import flash_attn_interface
import flash_attn_config

print("flash_attn_interface:", flash_attn_interface.__file__)
flash_attn_config.show()
PY
```

All `DISABLE_*` flags should be `False` for the full build.

## Install the other local dependencies

These commands assume the main Python environment already has CUDA/PyTorch and
the normal project requirements installed.

```bash
export PYTHON_BIN=/usr/local/bin/python

"${PYTHON_BIN}" -m pip install -v --no-build-isolation .deps/flash-linear-attention
"${PYTHON_BIN}" -m pip install -v .deps/mcore-bridge
"${PYTHON_BIN}" -m pip install -v .deps/Megatron-LM
```

If a package's upstream install requirements change, prefer matching the pinned
commits above before changing the main training scripts.

## Transformer Engine FA3 compatibility shim

Transformer Engine `2.9.0` imports `flash_attn_3.flash_attn_interface`, while
the Dao-AILab FA3 Hopper package exposes the implementation as the top-level
`flash_attn_interface` module. If TE import fails with a missing
`flash_attn_3.flash_attn_interface`, create this shim in the active
site-packages:

```bash
/usr/local/bin/python - <<'PY'
import site
from pathlib import Path

site_dir = Path(site.getsitepackages()[0])
pkg = site_dir / "flash_attn_3"
pkg.mkdir(exist_ok=True)
(pkg / "__init__.py").write_text(
    "from .flash_attn_interface import *\n",
    encoding="utf-8",
)
(pkg / "flash_attn_interface.py").write_text(
    "from flash_attn_interface import *\n"
    "from flash_attn_interface import _flash_attn_forward, _flash_attn_backward\n",
    encoding="utf-8",
)
print(pkg)
PY
```

## Verify

From the repository root:

```bash
/usr/local/bin/python scripts/verify_hopper_27b_ops.py --recommendations
/usr/local/bin/python -m pip check
```

The verification script checks CUDA SM90, FA3 Hopper attention, FLA gated delta
rule, Transformer Engine FP8 support, TE FP8 Linear, and TE FP8 SwiGLU MLP.
