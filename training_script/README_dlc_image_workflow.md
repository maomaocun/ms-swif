# DLC 镜像训练工作流

本文档说明如何用“保存好的训练镜像 + CPFS 持久存储 + DLC 作业”管理
Qwen3.6 27B SFT 训练。当前目录的脚本目标是让镜像地址晚一点再填也没问题：
镜像保存并推送后，只需要设置 `IMAGE_URI`，然后按 verify、dry-run、smoke、
train 的顺序推进。

不要把 Aliyun 密码、AccessKey、W&B key 写进仓库脚本。优先使用控制台、
RAM role、`aliyun configure`、DLC 环境变量或 CPFS 上的私有 secret 文件。

## 1. 分层原则

推荐边界：

- 镜像：CUDA、Python、PyTorch、FlashAttention 3、FLA、Transformer Engine、
  Megatron、ms-swift 运行依赖，以及可选的默认代码快照。
- CPFS：模型权重、训练数据、输出 checkpoint、日志、缓存、私有 secret。
- DLC：拉取指定镜像，挂载 CPFS，注入环境变量，启动训练脚本。

不要把大模型权重、训练数据、checkpoint 打进镜像。它们变化频率高、体积大，
放进 CPFS 更容易复用、回滚和审计。

建议 CPFS 目录结构：

```text
/mnt/cpfs/YOUR_TEAM/sft/
  repos/
    ms-swift/
  datasets/
    paper2arm/
      paper2arm_dsv4pro_sft_reward_ge_0.6.jsonl
  models/
    Qwen3.6-27B/
  outputs/
    qwen36-27b/
  logs/
    qwen36-27b/
  cache/
    qwen36-27b/
  secrets/
    wandb_api_key
  images/
    20260613-hopper/
      manifest.txt
```

## 2. 镜像命名

镜像 tag 建议不可变，不要长期用 `latest`：

```text
ms-swift-sft:20260613-hopper-cu129-fa3
ms-swift-sft:20260613-git-<shortsha>
ms-swift-sft:prod-qwen36-27b-20260613
```

每个镜像最好保存一份 manifest，例如：

```text
image_uri=...
git_commit=...
python_version=...
torch_version=...
cuda_version=...
flash_attn_version=...
transformer_engine_version=...
megatron_core_version=...
verified_at=...
verify_command=bash training_script/run_hopper27b_in_image.sh verify
```

## 3. 环境变量模板

提交 DLC 前需要本机或提交容器里有 `jq`，因为底层
`dlc/submit_qwen36_27b_dsv4pro_distill.sh` 用它读取 JSON 配置。

先复制模板到 CPFS 私有位置，不要直接在仓库里填私有路径和 secret：

```bash
cp training_script/env.example /mnt/cpfs/YOUR_TEAM/sft/env.qwen36.sh
vim /mnt/cpfs/YOUR_TEAM/sft/env.qwen36.sh
source /mnt/cpfs/YOUR_TEAM/sft/env.qwen36.sh
```

镜像保存并推送后，至少需要填：

```bash
export IMAGE_URI='REGISTRY/namespace/image:tag'
export SFT_MODEL_PATH='/mnt/cpfs/.../models/Qwen3.6-27B'
export SFT_DATASET_PATH='/mnt/cpfs/.../datasets/paper2arm/train.jsonl'
export SFT_OUTPUT_ROOT='/mnt/cpfs/.../outputs/qwen36-27b'
export SFT_LOG_ROOT='/mnt/cpfs/.../logs/qwen36-27b'
export SFT_CACHE_ROOT='/mnt/cpfs/.../cache/qwen36-27b'
```

## 4. 本地验证镜像

镜像保存并且 registry 可拉取后，先在本机或 DSW 环境验证：

```bash
docker login REGISTRY
docker pull "${IMAGE_URI}"

source /mnt/cpfs/YOUR_TEAM/sft/env.qwen36.sh
bash training_script/run_saved_image_local.sh verify
```

上面的封装脚本等价于下面的显式 Docker 命令：

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=256g \
  -v /mnt:/mnt \
  -w /mnt/workspace/ms-swif \
  "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh verify
```

`verify` 会检查：

- 当前 GPU 是否是 Hopper/SM90 类设备。
- FlashAttention 3 Hopper kernel 是否可运行。
- FLA `chunk_gated_delta_rule` 是否可运行。
- Transformer Engine 是否报告 FP8 支持。
- TE FP8 Linear 和 SwiGLU MLP forward/backward 是否可运行。

## 5. 本地 dry-run / smoke

dry-run 不启动完整训练，适合验证脚本参数、路径和环境：

```bash
source /mnt/cpfs/YOUR_TEAM/sft/env.qwen36.sh
bash training_script/run_saved_image_local.sh dry-run
```

如果需要看完整 Docker 参数，可以直接执行：

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=256g \
  -v /mnt:/mnt \
  -w /mnt/workspace/ms-swif \
  -e MODEL_PATH="${SFT_MODEL_PATH}" \
  -e DATASET_PATH="${SFT_DATASET_PATH}" \
  -e OUTPUT_ROOT="${SFT_OUTPUT_ROOT}" \
  -e LOG_ROOT="${SFT_LOG_ROOT}" \
  -e CACHE_ROOT="${SFT_CACHE_ROOT}" \
  -e REPORT_TO="${SFT_REPORT_TO:-none}" \
  "${IMAGE_URI}" \
  bash training_script/run_hopper27b_in_image.sh dry-run
```

smoke 会启动小规模训练检查链路：

```bash
bash training_script/run_saved_image_local.sh smoke
```

完整本地训练把最后的 `smoke` 改成 `train`。

## 6. DLC 提交

当前 DLC 基础配置在：

```text
dlc/config_qwen36_27b_dsv4pro_distill.json
```

提交脚本会临时生成 JSON，只替换 `.dlc.worker_image`，不会修改原配置：

```bash
export IMAGE_URI='REGISTRY/namespace/image:tag'

DRY_RUN=1 bash training_script/submit_dlc_with_saved_image.sh
bash training_script/submit_dlc_with_saved_image.sh
```

如果要覆盖训练路径和训练参数，使用 `DLC_ENVS`：

```bash
export DLC_ENVS="REPORT_TO=none,MODEL_PATH=${SFT_MODEL_PATH},DATASET_PATH=${SFT_DATASET_PATH},OUTPUT_ROOT=${SFT_OUTPUT_ROOT},LOG_ROOT=${SFT_LOG_ROOT},CACHE_ROOT=${SFT_CACHE_ROOT}"

DRY_RUN=1 bash training_script/submit_dlc_with_saved_image.sh
bash training_script/submit_dlc_with_saved_image.sh
```

如果已经 `source training_script/env.example` 的私有副本，且没有手动设置
`DLC_ENVS`，`submit_dlc_with_saved_image.sh` 会自动从常用 `SFT_*` 变量生成：

```text
MODEL_PATH
DATASET_PATH
OUTPUT_ROOT
LOG_ROOT
CACHE_ROOT
REPORT_TO
WANDB_PROJECT
WANDB_ENTITY
WANDB_API_KEY_FILE
NPROC_PER_NODE
TENSOR_MODEL_PARALLEL_SIZE
PIPELINE_MODEL_PARALLEL_SIZE
CONTEXT_PARALLEL_SIZE
GLOBAL_BATCH_SIZE
MICRO_BATCH_SIZE
MAX_LENGTH
LINEAR_CE_CHUNK_SIZE
```

如果使用 W&B，确保容器内能拿到 key：

```bash
export DLC_ENVS="REPORT_TO=wandb,WANDB_API_KEY_FILE=/mnt/cpfs/YOUR_TEAM/sft/secrets/wandb_api_key,..."
```

## 7. 当前 27B Megatron 默认建议

当前 Qwen3.6 27B Megatron SFT 建议从下面配置开始：

```text
NPROC_PER_NODE=8
TENSOR_MODEL_PARALLEL_SIZE=8
PIPELINE_MODEL_PARALLEL_SIZE=4
CONTEXT_PARALLEL_SIZE=1
SEQUENCE_PARALLEL=true
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
MAX_LENGTH=262144
LINEAR_CE_CHUNK_SIZE=2048
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1
```

注意：当前 Qwen3.6 `gated_delta_net` 路径在 Megatron Core 中不建议开
`CONTEXT_PARALLEL_SIZE > 1`。在没有重新验证前，DLC 多机优先使用
`TP8 + PP4 + CP1`。

## 8. 发布和回滚

推荐发布顺序：

1. 保存或构建镜像。
2. 推送到 DLC 同地域 registry。
3. 写入 CPFS manifest。
4. 本地 `verify`。
5. 本地 `dry-run`。
6. DLC `DRY_RUN=1`。
7. DLC smoke。
8. DLC full train。

回滚规则：

- 环境或依赖问题：换回上一版 `IMAGE_URI`。
- 代码问题：切换 CPFS 上的 repo commit，或者换回内置旧代码的镜像。
- 数据问题：切换 `DATASET_PATH`。
- 训练产物问题：查看 `OUTPUT_ROOT`、`LOG_ROOT`、W&B run name。

## 9. 相关脚本

- `training_script/README_dlc_image_workflow.md`：本文档。
- `training_script/env.example`：CPFS 私有环境变量模板。
- `training_script/run_saved_image_local.sh`：本地 Docker verify/dry-run/smoke/train 封装。
- `training_script/run_hopper27b_in_image.sh`：容器内 verify/dry-run/smoke/train 入口。
- `training_script/submit_dlc_with_saved_image.sh`：用指定 `IMAGE_URI` 提交 DLC。
- `dlc/submit_qwen36_27b_dsv4pro_distill.sh`：底层 DLC submit 封装。
- `dlc/worker_wrapper_qwen36_27b_dsv4pro_distill.sh`：DLC worker 训练入口。
