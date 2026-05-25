# Megatron 27B SFT 蒸馏训练加速与省显存调研

调研日期：2026-05-25

## 结论摘要

当前 27B SFT 路径已经处在一个可继续优化的状态：`TP=8/PP=1/CP=1`、`sequence_parallel=true`、`optimizer_cpu_offload=true`、`optimizer_offload_fraction=1`、`padding_free=true`、`max_length=262144`、`LINEAR_CE_CHUNK_SIZE=2048`。

短期优先级最高的不是 FP8，也不是 CP=8。原因很明确：

- 本机是 8 张 A100-SXM4-80GB；Transformer Engine 2.15.0 返回 FP8 不可用，原因是 FP8 execution 需要 compute capability 8.9+，A100 不满足。
- Qwen3.5/Qwen3.6 的 Megatron `gated_delta_net` 路径当前不支持 `context_parallel_size > 1`，脚本已保守限制为 CP=1。
- 当前训练日志记录的是 `torch.cuda.max_memory_reserved()`，不能完全反映 chunked CE 的 `allocated` 峰值收益；但单算子测试已证明 CE 本身收益很大。

下一步建议按这个顺序做 A/B：

1. `LINEAR_CE_CHUNK_SIZE=4096/8192`：当前 2048 更省显存，但循环次数多；256K 下还有显存余量，先用更大 chunk 换速度。
2. `RECOMPUTE_GRANULARITY=selective`：当前 full recompute 很省显存但慢；现有 256K reserved memory 约 53 GiB，A100 80G 上有空间测试 selective。
3. `GRADIENT_ACCUMULATION_FUSION=true`：当前 global batch 8、micro batch 1，存在 8 个 microbatch，梯度累加融合有机会直接减少开销。
4. CPU optimizer offload overlap：Megatron Core 已支持 `overlap_cpu_optimizer_d2h_h2d`，本轮已在 ms-swift 参数层和 27B 脚本里暴露。
5. optimizer state 精度和 offload fraction：尝试 `EXP_AVG_DTYPE=bf16 EXP_AVG_SQ_DTYPE=bf16`，然后把 `OPTIMIZER_OFFLOAD_FRACTION` 从 1 降到 0.7/0.5，用 GPU 显存换 CPU offload 带宽和速度。
6. `TP_COMM_OVERLAP=true` 单独测试；`overlap_grad_reduce/overlap_param_gather` 在当前 DP=1 下收益有限，未来多机 DP>1 再优先打开。

## 当前状态证据

### 硬件与软件

本机查询结果：

| 项目 | 当前值 |
| --- | --- |
| GPU | 8 x NVIDIA A100-SXM4-80GB |
| Driver | 580.105.08 |
| PyTorch | 2.6.0+cu126 |
| CUDA runtime | 12.6 |
| Transformer Engine | 2.15.0 |
| Megatron Core | 0.17.0 |
| mcore_bridge | 1.4.0 |
| BF16 | 可用 |
| FP8/MXFP8/NVFP4 | 不可用 |

Transformer Engine 的实际探测输出显示：

- `is_fp8_available`: False，原因是 `Device compute capability 8.9 or higher required for FP8 execution.`
- `is_mxfp8_available`: False，原因是 compute capability 10.0+。
- `is_fp8_block_scaling_available`: False，原因是 compute capability 9.0+ 且 CUDA >= 12.9。
- `is_bf16_available`: True。

因此当前 A100 机器上不建议继续花时间试 Megatron FP8 训练；FP8 只能作为 H100/H800/B200 迁移项。

### 模型结构

`/mnt/cpfs/public_data/public_model/Qwen3.6/Qwen3.6-27B/config.json` 的 `text_config`：

| 项目 | 值 |
| --- | --- |
| model_type | `qwen3_5_text` |
| hidden_size | 5120 |
| num_hidden_layers | 64 |
| vocab_size | 248320 |
| max_position_embeddings | 262144 |
| layer_types | 每 4 层 1 层 full attention，其余为 linear attention |

在 TP=8 下，output layer 的 vocab shard 是 `248320 / 8 = 31040`。完整 logits 的理论临时形状是 `[tokens, vocab/TP]`，256K token 下是 `[262144, 31040]`，bf16 也约 15.2 GiB/rank，若 CE 内部转 fp32 会翻倍。这就是 chunked linear CE 必须保留的原因。

### 现有训练指标

已有 256K 默认 run：

| run | chunk CE 日志 | step | loss | grad_norm | memory(GiB) | train_speed(s/it) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen36-27b-paper2arm-distill-megatron-tp8-20260524-173420` | 否 | 1/18 | 0.20536338 | 0.69596642 | 53.84 | 509.881 |
| 同上 | 否 | 4/18 | 0.20705265 | 0.47357491 | 53.84 | 338.054 |
| `qwen36-27b-paper2arm-distill-megatron-tp8-256k-tokenchunkce-20260524-234709` | train.log 未搜到 | 1/18 | 0.20536332 | 0.69443589 | 53.24 | 318.391 |

注意：

- 256K chunk run 的 loss/grad 与 baseline 对齐，说明 loss 数值正常。
- 256K run 的 `train.log` 没搜到 chunked CE 命中日志，所以端到端显存下降不能全部归因于 chunk CE。
- 单算子 benchmark 已严格证明 CE 段收益：262144 tokens 下 full bf16 CE 增量 peak allocated 约 60.63 GiB，full+fp32 CE OOM，chunked CE 约 4.17 GiB。详见 `fuseCEloss.md`。

## 已补充的代码开关

本轮额外暴露了 Megatron Core 已有但 ms-swift 参数层未暴露的 CPU offload overlap 相关开关。

改动文件：

- `swift/megatron/arguments/megatron_args.py`
- `train_qwen36_27b_paper2arm_distill_megatron.sh`
- `docs/source/Megatron-SWIFT/Command-line-parameters.md`
- `docs/source_en/Megatron-SWIFT/Command-line-parameters.md`

新增脚本环境变量：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `USE_TORCH_OPTIMIZER_FOR_CPU_OFFLOAD` | `false` | CPU offload 时使用 torch optimizer |
| `OVERLAP_CPU_OPTIMIZER_D2H_H2D` | `false` | CPU optimizer 更新与 D2H/H2D 传输重叠 |
| `PIN_CPU_GRADS` | `true` | CPU 侧 grads 使用 pinned memory |
| `PIN_CPU_PARAMS` | `true` | CPU 侧 params 使用 pinned memory |

验证结果：

- `bash -n train_qwen36_27b_paper2arm_distill_megatron.sh` 通过。
- `DRY_RUN=1 SMOKE=1 OVERLAP_CPU_OPTIMIZER_D2H_H2D=true ...` 通过。
- `megatron sft --help` 已能看到 `--overlap_cpu_optimizer_d2h_h2d`、`--pin_cpu_grads`、`--pin_cpu_params`、`--use_torch_optimizer_for_cpu_offload`。

另外修正了 A/B 实验的一个脚本问题：用户显式设置 `TRAIN_ITERS` 时，不再默认追加 `--num_train_epochs 3`。现在 `TRAIN_ITERS=2` 的 dry-run 命令只带 `--train_iters 2`，不会再出现本想跑 1 step 却显示 `1/18` 的情况。

## 优先级方案

### P0：保留并调大 chunked linear CE

现状：`LINEAR_CE_CHUNK_SIZE=2048`。

建议：

- 256K 当前 reserved memory 约 53 GiB，还有空间。
- 先试 `4096`，再试 `8192`。
- 如果未来上下文超过 256K 后 OOM，再退回 `2048` 或 `1024`。

命令：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
LINEAR_CE_CHUNK_SIZE=4096 \
RUN_NAME=ab-ce-chunk4096 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

判断标准：

- loss 与 2048 对齐。
- `memory(GiB)` 不超过 80G。
- `train_speed(s/it)` 下降。

### P1：从 full recompute 改测 selective recompute

现状：

```bash
RECOMPUTE_GRANULARITY=full
RECOMPUTE_METHOD=uniform
RECOMPUTE_NUM_LAYERS=1
```

问题：full recompute 省显存最强，但会重算整层，速度代价大。ms-swift Megatron 文档也写到 `selective` 通常是推荐设置。

建议分两档：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
RECOMPUTE_GRANULARITY=selective \
RECOMPUTE_MODULES=core_attn \
RUN_NAME=ab-recompute-selective-coreattn \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

如果显存接近上限，再试：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
RECOMPUTE_GRANULARITY=selective \
RECOMPUTE_MODULES="core_attn mlp" \
RUN_NAME=ab-recompute-selective-coreattn-mlp \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

如果 selective OOM，则回到 full，并只在更长上下文时调大 `RECOMPUTE_NUM_LAYERS=2/4`。

### P1：打开梯度累加融合

当前 `GLOBAL_BATCH_SIZE=8`、`MICRO_BATCH_SIZE=1`、`DP=1`，等价于每 step 8 个 microbatch。`gradient_accumulation_fusion` 对这种场景有机会减少累加开销。

命令：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
GRADIENT_ACCUMULATION_FUSION=true \
RUN_NAME=ab-grad-acc-fusion \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

### P1：CPU optimizer offload overlap

当前为了省显存，optimizer 状态 100% offload 到 CPU：

```bash
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1
```

这能省显存，但会引入 CPU optimizer 和 H2D/D2H 传输等待。Megatron Core 0.17 已有 `overlap_cpu_optimizer_d2h_h2d`，本轮已暴露到 ms-swift CLI。

命令：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
OVERLAP_CPU_OPTIMIZER_D2H_H2D=true \
RUN_NAME=ab-cpuopt-overlap \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

如果有效，再叠加 optimizer state bf16：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
OVERLAP_CPU_OPTIMIZER_D2H_H2D=true \
EXP_AVG_DTYPE=bf16 \
EXP_AVG_SQ_DTYPE=bf16 \
RUN_NAME=ab-cpuopt-overlap-state-bf16 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

### P1：降低 optimizer offload fraction

当前 256K run 的 `memory(GiB)` 约 53 GiB，A100 80G 上有约 25 GiB 余量。可以把一部分 optimizer state 留在 GPU，换速度。

建议顺序：

1. `EXP_AVG_DTYPE=bf16 EXP_AVG_SQ_DTYPE=bf16 OPTIMIZER_OFFLOAD_FRACTION=0.7`
2. 如果显存仍稳，再试 `OPTIMIZER_OFFLOAD_FRACTION=0.5`
3. 不建议一开始改 `MAIN_PARAMS_DTYPE=fp16`，这是主参数精度，风险比 Adam 状态 bf16 更高。

命令：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
EXP_AVG_DTYPE=bf16 \
EXP_AVG_SQ_DTYPE=bf16 \
OPTIMIZER_OFFLOAD_FRACTION=0.7 \
RUN_NAME=ab-optstate-bf16-offload07 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

### P2：TP communication overlap

当前 TP=8，理论上 TP 通信可能较重。可以单独测试：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
TP_COMM_OVERLAP=true \
RUN_NAME=ab-tp-comm-overlap \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

`OVERLAP_GRAD_REDUCE=true` 和 `OVERLAP_PARAM_GATHER=true` 当前优先级较低，因为当前 `TP=8/PP=1/CP=1/NPROC=8` 导致 `DP=1`，DP 通信本身几乎没有收益空间。未来多机或降低 TP 产生 DP>1 时再打开。

### P2：TP4 + PP2 拓扑试验

当前 TP8/PP1 的优点是切得细，单卡参数和 vocab shard 小；缺点是 TP 通信重。可以测试 TP4/PP2：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
PIPELINE_MODEL_PARALLEL_SIZE=2 \
RUN_NAME=ab-tp4-pp2 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

预期：

- 每层 TP shard 变大，但每个 PP rank 只放一部分层。
- TP 通信可能下降，pipeline bubble 会增加。
- 当前 global batch 8 有 8 个 microbatch，PP2 bubble 尚可接受。
- 这不是确定优化项，需要看实测速度和显存分布。

### P2：packing 只作为吞吐优化，不作为默认

当前数据集 48 条，长度统计：

```text
mean=70948.67, min=17723, max=169996, size=48
```

`padding_free=true` 已经避免了普通 padding 浪费。`packing=true` 可以把多条短轨迹拼到一个长序列里，减少 step 数并提升 tokens/step，但会改变“一个样本一条轨迹”的训练组织方式，需要确认是否接受。

如果只看吞吐，可以另开实验：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
PACKING=true \
RUN_NAME=ab-packing \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

### P3：环境级优化

当前环境有两个 warning：

- Apex 未安装，Megatron fallback 到 Torch Norm。
- Triton 3.2.0 低于推荐 3.3.0。

这两个不一定是当前主瓶颈，但可以作为环境稳定后的小优化：

1. 升级 Triton 到 3.3.x 后做 4K smoke 与 256K 2-step 对比。
2. 若环境允许，安装 Apex 后观察 Norm kernel 和启动 warning 是否改善。
3. 不建议在这台 A100 上优先折腾 FlashAttention-3/FP8；FA3 主要面向 Hopper/H800/H100，FP8 当前硬件不可用。

## 不建议现在投入的方向

### FP8

不建议当前 A100 机器上继续试。实际 TE 探测已经给出不可用原因：FP8 execution 需要 compute capability 8.9+；blockwise FP8 还要求 compute capability 9.0+ 且 CUDA >= 12.9。当前是 A100 + CUDA 12.6。

迁移到 H100/H800/B200 后再测：

```bash
FP8_FORMAT=e4m3 \
FP8_RECIPE=delayed \
RUN_NAME=ab-fp8-delayed \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

blockwise FP8 应只在 CUDA 12.9+ 且硬件满足时测试。

### CP=4/8

从长上下文训练角度看，CP=4/8 本来是最想要的方向，因为它直接切 sequence 维度。但当前 Qwen3.5/Qwen3.6 Megatron `gated_delta_net` 路径不支持 `context_parallel_size > 1`，脚本已经明确 guard。

下一步如果要真正解决更长上下文，应该投入到：

- 升级 mcore_bridge/Megatron 到支持 Qwen3.5 GatedDeltaNet CP 的版本；
- 或者在 GatedDeltaNet/linear attention 路径补齐 CP split/gather 逻辑；
- 然后再开 `CONTEXT_PARALLEL_SIZE=4/8`。

### HF/Transformers 的 `sequence_parallel_size`

ms-swift HF 路径有 `sequence_parallel_size`、`CELOSS_PARALLEL_SIZE`、`use_logits_to_keep` 等长上下文技巧，但 Megatron-SWIFT 的参数体系不同。当前 Megatron 参数里 `sequence_parallel_size` 被映射为 `context_parallel_size`，而 CP 又被 Qwen3.5/3.6 GatedDeltaNet 卡住，所以不能把 HF 路径经验直接搬过来。

## 推荐实验矩阵

每次只改一个主要变量，统一：

```bash
TRAIN_ITERS=2
SAVE_STEPS=1000000
LOGGING_STEPS=1
MAX_LENGTH=262144
```

记录：

- loss
- grad_norm
- memory(GiB)
- train_speed(s/it)
- 是否 OOM
- `logs/.../train.log` 是否出现 chunked CE 命中日志
- `outputs/.../logging.jsonl`

| 序号 | 目的 | 变量 | 预期 |
| --- | --- | --- | --- |
| A0 | 基线 | 默认脚本 | 对齐当前 256K 指标 |
| A1 | CE chunk 提速 | `LINEAR_CE_CHUNK_SIZE=4096` | 显存略升，速度提升 |
| A2 | CE chunk 提速上限 | `LINEAR_CE_CHUNK_SIZE=8192` | 若不 OOM，速度可能更好 |
| B1 | 降低 recompute 开销 | `RECOMPUTE_GRANULARITY=selective RECOMPUTE_MODULES=core_attn` | 显存升，速度升 |
| B2 | selective 稳妥版 | `RECOMPUTE_MODULES="core_attn mlp"` | 介于 full 与 core_attn 之间 |
| C1 | 梯度累加融合 | `GRADIENT_ACCUMULATION_FUSION=true` | 速度提升，显存变化小 |
| D1 | CPU offload overlap | `OVERLAP_CPU_OPTIMIZER_D2H_H2D=true` | offload 等待下降 |
| D2 | optimizer 状态降精度 | `EXP_AVG_DTYPE=bf16 EXP_AVG_SQ_DTYPE=bf16` | CPU/GPU optimizer state 内存下降 |
| D3 | 用显存换速度 | `OPTIMIZER_OFFLOAD_FRACTION=0.7` + D2 | 显存升，速度可能升 |
| E1 | TP 通信 overlap | `TP_COMM_OVERLAP=true` | TP 通信瓶颈时有效 |
| F1 | 拓扑对比 | `TP=4 PP=2` | 需要实测，可能降 TP 通信 |

## 推荐的下一条实际命令

我建议第一条不要跑过多变量，先测 chunk size 4096：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
LINEAR_CE_CHUNK_SIZE=4096 \
RUN_NAME=ab-ce-chunk4096 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

如果这条显存稳定且速度提升，再跑：

```bash
TRAIN_ITERS=2 \
SAVE_STEPS=1000000 \
LINEAR_CE_CHUNK_SIZE=4096 \
RECOMPUTE_GRANULARITY=selective \
RECOMPUTE_MODULES=core_attn \
RUN_NAME=ab-ce4096-recompute-selective-coreattn \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

## 参考来源

- 本仓库：`docs/source/Megatron-SWIFT/Command-line-parameters.md`
- 本仓库：`examples/train/sequence_parallel/sequence_parallel_512k.sh`
- 本仓库：`examples/train/flash_attention_3/mcore.sh`
- 本仓库：`fuseCEloss.md`
- Megatron Core 文档：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html
- Transformer Engine 项目：https://github.com/NVIDIA/TransformerEngine
- FlashAttention 项目：https://github.com/Dao-AILab/flash-attention
