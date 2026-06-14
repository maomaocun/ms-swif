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

git clone https://github.com/maomaocun/mcore-bridge .deps/mcore-bridge
git -C .deps/mcore-bridge checkout 9b6dd2e9de3b015721a73897ecd31f02f10213c2

git clone --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git .deps/Megatron-LM
git -C .deps/Megatron-LM checkout aa1057124ef53e99cbfc799c916ad948824bbff6
```

Current local dependency sync status:

- `flash-attention`: clean, `origin/main` at
  `fc8cbad6b6b90220cf6ef8121c29e299a3ba7d9a`, no local-only commits.
- `flash-linear-attention`: clean, `origin/main` at
  `c525f4957f11a6f197b52c0c222377446c3eab56`, no local-only commits.
- `Megatron-LM`: clean, `origin/main` at
  `aa1057124ef53e99cbfc799c916ad948824bbff6`, no local-only commits.
- `mcore-bridge`: clean local branch `gdn-fp8-weightpad` at
  `9b6dd2e9de3b015721a73897ecd31f02f10213c2`. Relative to
  `https://github.com/maomaocun/mcore-bridge` `origin/main`
  (`cebb791f0e5e04ce162f5eed2d29e526b98ee9c5`), this branch has 16
  local-only commits and is missing 33 commits from `origin/main`.
  Push `gdn-fp8-weightpad` before relying on the checkout command above on a
  fresh machine.

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

## Qwen3.6 GDN FP8 smoke note

The current local image uses the `mcore-bridge` branch `gdn-fp8-weightpad`.
For Qwen3.6 27B with TP=8, GDN `in_proj.weight` has a local output row count of
`2060`, which is not divisible by 8 and fails Transformer Engine FP8 execution.
This is a TP-local weight shape issue, not a dataset token-padding issue.

The local patch pads that outer module weight from `2060` to `2064`, runs the
projection in FP8, then slices the virtual channels away before GDN split/reshape
logic sees them. HF-to-MCore loading pads the corresponding TP chunks, and
MCore-to-HF export strips the virtual padded rows.

If `.deps/mcore-bridge` is recreated or reinstalled, re-apply that behavior and
install it into `/usr/local/bin/python` before running the FP8 local smoke:

```bash
training_script/local_smoke/run_qwen36_27b_fp8_smoke.sh
```

ModelScope's documented Qwen3.5/Qwen3-Next FP8 route uses
`--linear_decoupled_in_proj true`, `--fp8_recipe blockwise`,
`--fp8_format e4m3`, and `--fp8_param_gather true`. This keeps `in_proj_ba` in
the original precision and mainly places FP8 around TE linear projection paths;
it does not make the gated-delta-rule recurrent kernel itself FP8.

The verified local Qwen3.6 27B smoke used `FP8_FORMAT=hybrid`,
`FP8_RECIPE=delayed`, `FP8_PARAM_GATHER=false`, and the local GDN weight-padding
patch, so it is conceptually aligned with FP8-on-TE-projections but not
identical to the ModelScope example settings. To test the documented route with
the local training script, set:

```bash
LINEAR_DECOUPLED_IN_PROJ=true \
FP8_FORMAT=e4m3 \
FP8_RECIPE=blockwise \
FP8_PARAM_GATHER=true \
training_script/local_smoke/run_qwen36_27b_smoke.sh
```

## Apex note

Apex is not installed in this image. Transformer Engine `2.9.0` is installed,
but Megatron Core `0.17.1` tries to import
`transformer_engine.pytorch.optimizers.multi_tensor_scale_tensor`, which this TE
build does not export. Megatron then tries Apex (`amp_C` and
`apex.multi_tensor_apply`), does not find it, and falls back to local
multi-tensor helpers.

Megatron-SWIFT can run without Apex when `--gradient_accumulation_fusion false`
is used, which is the current local script default. Installing Apex is possible
but not required for the validated smoke path; it requires building
`NVIDIA/apex` against the exact active PyTorch/CUDA toolchain with `--cpp_ext`
and `--cuda_ext`, and failures are common when the compiler, CUDA toolkit, or
PyTorch ABI are mismatched.
