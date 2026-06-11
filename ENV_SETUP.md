# ms-swift SFT 环境交接说明

日期：2026-05-25

本文档说明 `/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift` 下 SFT 训练环境的结构、初始化方式、验证方式和常用训练入口。

## 1. 一键配置

进入目录：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
```

复用并校验当前环境：

```bash
bash setup_uv_env.sh
```

从头重建环境：

```bash
RECREATE=1 bash setup_uv_env.sh
```

只校验现有环境、不安装依赖：

```bash
SKIP_INSTALL=1 bash setup_uv_env.sh
```

不跑训练 dry-run，只做 Python/package/patch 校验：

```bash
RUN_DRY_RUN=0 bash setup_uv_env.sh
```

脚本会做这些事：

- 创建或复用 `.venv`
- 创建或复用 `.venv-megatron`
- 安装 HF/DeepSpeed 路径依赖
- 安装 Megatron/mcore_bridge 路径依赖
- 安装当前 ms-swift 源码为 editable package
- clone/复用 `.deps/mcore-bridge`，给源码打 `LINEAR_CE_CHUNK_SIZE` patch，并以 editable 方式安装
- 校验 `torch`、`transformers`、`deepspeed`、`swift`、`mcore_bridge` 是否可导入
- 校验 `megatron` / `swift` 命令是否可用
- 默认执行一次 27B Megatron smoke dry-run，不真正启动训练

setup 里的 dry-run 输出会写到 `/tmp/ms-swift-setup-*`，不会污染当前目录下的正式 `logs/` 和 `outputs/`。

## 2. 环境结构

当前目录维护两个 Python 环境：

```text
.venv
.venv-megatron
```

`.venv` 用于 HF/Transformers + DeepSpeed 路径，主要承载：

- torch / CUDA Python wheels
- transformers
- deepspeed
- datasets
- peft / trl / accelerate
- flash-attn
- tensorboard
- ms-swift editable package

`.venv-megatron` 用于 Megatron 路径，主要承载：

- megatron-core
- mcore-bridge editable install（源码在 `.deps/mcore-bridge`）
- Megatron 入口命令 `megatron`
- ms-swift editable package

Megatron 训练脚本不会直接 `source .venv-megatron/bin/activate`，而是 source：

```bash
source ./megatron_env.sh
```

`megatron_env.sh` 会把以下路径拼起来：

```text
当前 ms-swift 源码目录
.venv-megatron/site-packages
.venv/site-packages
可选的 /mnt/cpfs/yangyicun/.venv/site-packages
```

这样做的原因是：Megatron 侧需要自己的 `mcore_bridge` / `megatron-core`，但也要复用 HF 环境里的 torch、CUDA wheel、Transformer Engine、flash-attn 等依赖。

## 3. 关键脚本

一键配置：

```text
setup_uv_env.sh
```

Megatron 环境注入：

```text
megatron_env.sh
```

27B Megatron 长上下文 SFT：

```text
train_qwen36_27b_paper2arm_distill_megatron.sh
```

27B Megatron fallback launcher：

```text
launch_qwen36_27b_paper2arm_distill_with_fallback.sh
```

27B HF/DeepSpeed 兜底：

```text
train_qwen36_27b_paper2arm_distill_full_sp4.sh
```

9B Megatron 长上下文 SFT：

```text
train_qwen35_9b_paper2arm_distill_megatron.sh
```

## 4. 推荐训练入口

27B Megatron 正式训练：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

27B Megatron smoke：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

27B Megatron dry-run：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
DRY_RUN=1 SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

27B HF/DeepSpeed 兜底：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
DEEPSPEED=zero3_offload bash train_qwen36_27b_paper2arm_distill_full_sp4.sh
```

## 5. 当前 27B Megatron 默认配置

`train_qwen36_27b_paper2arm_distill_megatron.sh` 的关键默认值：

```text
model=/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B
dataset=data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl
max_length=262144
truncation_strategy=delete
loss_scale=default
tensor_model_parallel_size=8
pipeline_model_parallel_size=1
context_parallel_size=1
sequence_parallel=true
optimizer_cpu_offload=true
optimizer_offload_fraction=1
lr=1e-5
min_lr=1e-6
lr_warmup_fraction=0.05
cross_entropy_loss_fusion=true
LINEAR_CE_CHUNK_SIZE=2048
```

注意：

- Qwen3.5/Qwen3.6 当前 Megatron `gated_delta_net` 路径不支持 `context_parallel_size > 1`。
- 因此 CP 不能开 4 或 8，当前可用方案是 `TP=8, PP=1, CP=1`。
- `sequence_parallel=true` 保持开启。
- optimizer 使用 CPU offload，但参数本身不做 ZeRO3 参数 offload。

## 6. Chunked Linear CE Patch

当前环境使用源码 patch，而不是直接修改 `.venv-megatron/site-packages`：

```text
.deps/mcore-bridge/src/mcore_bridge/model/gpt_model.py
```

`.venv-megatron` 中的 `mcore_bridge` 是 editable install：

```text
.venv-megatron/lib/python3.12/site-packages/__editable__.mcore_bridge-1.4.0.pth
```

该 patch 增加：

```text
LINEAR_CE_CHUNK_SIZE
```

默认：

```text
LINEAR_CE_CHUNK_SIZE=2048
```

作用：在 Megatron SFT 中绕过完整 `[tokens, vocab/TP]` logits 常驻，只对 `labels != -100` 的 supervised token 按扁平 token chunk 计算 linear CE，直接返回 per-token loss。

开启：

```bash
LINEAR_CE_CHUNK_SIZE=2048 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

关闭：

```bash
LINEAR_CE_CHUNK_SIZE=0 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

默认生产训练不会打印 CE 分支命中日志。需要排查时显式开启：

```bash
LINEAR_CE_DEBUG=1 LINEAR_CE_CHUNK_SIZE=2048 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

此时只会打印一次简短日志：

```text
[INFO:mcore_bridge] Using supervised-token chunked linear CE loss; chunk_size=2048.
```

如果重建 `.venv-megatron`，`setup_uv_env.sh` 会从 `.deps/mcore-bridge` 安装 patched 源码。若 `.deps/mcore-bridge` 不存在，脚本会先 clone 上游 `modelscope/mcore-bridge` 的 `v1.4.0`，再给源码打 patch。

单算子显存验证报告见：

```text
fuseCEloss.md
```

## 7. 数据与模型路径

默认数据：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl
```

默认 27B 模型：

```text
/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B
```

默认 9B 模型：

```text
/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B
```

当前数据统计：

```text
size=48
mean=70948.666667
std=33792.611034
min=17723
max=169996
```

所以 `max_length=262144` 可以覆盖当前所有轨迹，避免截断。

## 8. 输出与日志

训练脚本将 checkpoint/model 产物和日志产物分开保存：

```text
OUTPUT_DIR = ${OUTPUT_ROOT}/${RUN_NAME}
LOG_DIR    = ${LOG_ROOT}/${RUN_NAME}
```

27B Megatron 默认 checkpoint/model 输出：

```text
/mnt/cpfs/yangyicun/data/agent_checkpoints/qwen36-27b-paper2arm-distill-megatron/<run-name>
```

27B Megatron 默认日志：

```text
logs/qwen36-27b-paper2arm-distill-megatron/<run-name>/train.log
```

日志目录内的稳定文件：

```text
train.log          # stdout/stderr，脚本会加时间戳
logging.jsonl      # step metrics + final train_msg
run_metadata.json  # 启动参数、环境、包版本、git 信息、关键路径
run_summary.json   # 结束状态、best/last checkpoint
runs/              # TensorBoard event files
wandb/             # W&B 本地目录
swanlab/           # SwanLab 本地目录
images/            # TensorBoard 可视化图片
```

`OUTPUT_DIR` 只承载 checkpoint/model/args 等训练产物，避免日志和模型保存目录混在一起。

## 9. 手动验证

基础环境验证：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
bash setup_uv_env.sh
```

Megatron 命令验证：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
source ./megatron_env.sh
which megatron
python - <<'PY'
import torch
import transformers
import deepspeed
import swift
from mcore_bridge.model import gpt_model
print(torch.__version__)
print(hasattr(gpt_model, "_ChunkedLinearCrossEntropy"))
PY
```

训练参数 dry-run：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
DRY_RUN=1 SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

单算子 CE 显存 benchmark：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
source ./megatron_env.sh
CUDA_VISIBLE_DEVICES=0 python scripts/bench_chunked_linear_ce_memory.py \
  --mode chunked \
  --tokens 262144 \
  --hidden 5120 \
  --vocab-shard 31040 \
  --dtype bf16 \
  --chunk-size 2048 \
  --backward
```

## 10. 常用环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `RECREATE` | 重建 `.venv` 和 `.venv-megatron` | `0` |
| `SKIP_INSTALL` | 只校验，不安装依赖 | `0` |
| `RUN_DRY_RUN` | setup 末尾跑训练 dry-run | `1` |
| `APPLY_CHUNKED_CE_PATCH` | setup 时自动应用/校验 `.deps/mcore-bridge` 源码 patch | `1` |
| `MCORE_BRIDGE_SOURCE_DIR` | mcore-bridge 源码目录 | `.deps/mcore-bridge` |
| `MCORE_BRIDGE_REF` | mcore-bridge checkout ref | `v1.4.0` |
| `PYTHON_BIN` | 创建 venv 使用的 Python | `python3.12` |
| `UV_BIN` | uv 路径 | 自动检测 |
| `LINEAR_CE_CHUNK_SIZE` | Megatron chunked linear CE token chunk | `2048` |
| `LINEAR_CE_DEBUG` | 打印一次 chunked linear CE debug 日志 | unset |
| `MODEL_PATH` | 覆盖模型路径 | 脚本内默认 |
| `DATASET_PATH` | 覆盖数据路径 | 脚本内默认 |
| `OUTPUT_ROOT` | checkpoint/model 输出根目录 | `/mnt/cpfs/yangyicun/data/agent_checkpoints/...` |
| `LOG_ROOT` | 日志根目录 | `logs/...` |
| `SMOKE` | 小长度 smoke 训练 | `0` |
| `DRY_RUN` | 只打印训练命令，不启动训练 | `0` |

## 11. 常见问题

### 1. `megatron_env.sh` 报找不到 `.venv-megatron/bin/megatron`

先跑：

```bash
bash setup_uv_env.sh
```

如果仍失败，强制重建：

```bash
RECREATE=1 bash setup_uv_env.sh
```

### 2. 训练日志没有 `Using supervised-token chunked linear CE`

这是默认行为。生产训练默认不打印 CE 分支命中日志。

确认环境变量：

```bash
LINEAR_CE_DEBUG=1 LINEAR_CE_CHUNK_SIZE=2048 DRY_RUN=1 SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

确认 patch：

```bash
source ./megatron_env.sh
python - <<'PY'
from mcore_bridge.model import gpt_model
print(gpt_model.__file__)
print(hasattr(gpt_model, "_ChunkedLinearCrossEntropy"))
PY
```

### 3. CP=4 或 CP=8 失败

这是当前模型路径限制。Qwen3.5/Qwen3.6 的 Megatron `gated_delta_net` 目前不支持 `context_parallel_size > 1`。不要在 Megatron 脚本里开 CP。

### 4. 27B HF/DeepSpeed OOM

HF/DeepSpeed 路径推荐直接使用：

```bash
DEEPSPEED=zero3_offload bash train_qwen36_27b_paper2arm_distill_full_sp4.sh
```

不要再优先尝试 `zero2_offload` 或 `SEQUENCE_PARALLEL_SIZE=8`。

## 12. 交接建议

交给别人时，建议让对方按顺序执行：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
bash setup_uv_env.sh
DRY_RUN=1 SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
SMOKE=1 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

如果 smoke 正常，再启动正式 27B 长上下文训练：

```bash
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```
