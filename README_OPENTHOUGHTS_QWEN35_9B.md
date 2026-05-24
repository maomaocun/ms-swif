# Qwen3.5-9B OpenThoughts SFT

本目录提供一套基于本地 ms-swift 的 Qwen3.5-9B OpenThoughts 全参数 SFT 启动与验证记录。

- 模型：`/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B`
- 数据集：`/mnt/cpfs/yangyicun/data/datasets/openthoughts_prepared/openthoughts_sft_400k_filtered.jsonl`
- 启动脚本：`train_qwen35_9b_openthoughts_full.sh`

## 开发摘要

本次变更记录了这台机器上完成的 Qwen3.5-9B OpenThoughts SFT 准备、验证和参数选择工作：

- 修复了本地 ms-swift 虚拟环境；原来的 `.venv` 指向了已经缺失的 uv 管理 Python 解释器。
- 核验了 Qwen3.5 attention 相关依赖和算子：flash-linear-attention/FLA、causal-conv1d、flash-attn 2、Liger kernel、Triton、DeepSpeed、PyTorch、Transformers。
- 确认 Qwen3.5-9B 配置包含 32 个 text layer，其中 24 个是 linear-attention layer，8 个是 full-attention layer。
- 新增 `train_qwen35_9b_openthoughts_full.sh`，作为可复现的 SFT 启动脚本，包含路径检查、dry-run、smoke 模式、日志、缓存隔离和环境变量覆盖能力。
- 基于实测结果选择生产默认参数，而不是沿用短上下文假设：`MAX_LENGTH=18432`、`SEQUENCE_PARALLEL_SIZE=1`、`PADDING_FREE=true`、`USE_LOGITS_TO_KEEP=true`、`PER_DEVICE_TRAIN_BATCH_SIZE=2`、`GRADIENT_ACCUMULATION_STEPS=4`、`DEEPSPEED=zero2`。
- 统计了数据长度分布，确认 4k/8k 上下文会截断绝大多数样本。
- 跑过快速 smoke、长上下文 smoke 和多组接近生产配置的 benchmark，用于选择当前默认参数。
- 启动过一次 full-run verification，确认训练到 step 10；之后根据操作者要求停止，正式完整训练不再继续。
- 停止时尚未到 `save_steps=500`，因此不应期待该 verification run 产出最终 checkpoint。

## 使用命令

快速启动 smoke 测试：

```bash
SMOKE=1 ./train_qwen35_9b_openthoughts_full.sh
```

匹配生产上下文设置的长上下文 smoke 测试：

```bash
SMOKE=1 \
MAX_LENGTH=18432 \
SEQUENCE_PARALLEL_SIZE=4 \
PADDING_FREE=true \
USE_LOGITS_TO_KEEP=false \
./train_qwen35_9b_openthoughts_full.sh
```

生产配置启动命令：

```bash
./train_qwen35_9b_openthoughts_full.sh
```

常用覆盖参数：

```bash
# 覆盖更多上下文，但显存和稳定性风险更高。
MAX_LENGTH=32768 SEQUENCE_PARALLEL_SIZE=4 USE_LOGITS_TO_KEEP=false ./train_qwen35_9b_openthoughts_full.sh

# 低显存回退配置。
MAX_LENGTH=8192 SEQUENCE_PARALLEL_SIZE=1 PADDING_FREE=false USE_LOGITS_TO_KEEP=true ./train_qwen35_9b_openthoughts_full.sh
```

## 当前默认参数

生产默认值是针对这份 OpenThoughts 长样本数据调过的，不是通用短上下文 SFT 默认值：

- `MAX_LENGTH=18432`
- `SEQUENCE_PARALLEL_SIZE=1`
- `PADDING_FREE=true`
- `USE_LOGITS_TO_KEEP=true`
- `PER_DEVICE_TRAIN_BATCH_SIZE=2`
- `GRADIENT_ACCUMULATION_STEPS=4`
- `CELOSS_PARALLEL_SIZE=2048`
- `DEEPSPEED=zero2`
- `TUNER_TYPE=full`
- `LAZY_TOKENIZE=true`
- `STREAMING=false`
- `ATTN_IMPL=flash_attention_2`
- `USE_LIGER_KERNEL=true`

Smoke 模式保持快速启动，默认使用 `MAX_LENGTH=4096`、`STREAMING=true`、`GRADIENT_ACCUMULATION_STEPS=1`。

## 验证证据

环境修复与依赖检查：

- `.venv` 原本指向缺失的 uv-managed Python 3.12 解释器。
- 已通过 `uv python install 3.12.13` 恢复。
- 已验证可导入：`torch 2.6.0+cu126`、`transformers 5.8.1`、`deepspeed 0.19.0`、`flash_attn 2.8.3`、`fla 0.5.0`、`causal_conv1d 1.6.2.post1`、`liger_kernel 0.8.0`。

数据集规模与长度：

- 行数：`387522`
- 文件大小：约 `19G`
- 字符长度：p50 `51911`，p75 `59523`，p90 `67003`，p95 `70682`，p99 `75379`，max `246613`
- 前 1000 行 token 抽样：mean `15538.7`，p50 `17255`，p95 `17708`，max `18327`
- 前 1000 行中，`MAX_LENGTH=4096` 会截断 `97.3%`，`MAX_LENGTH=8192` 会截断 `93.9%`，`MAX_LENGTH=16384` 会截断 `70.2%`。

Smoke 结果：

- 快速 smoke，`MAX_LENGTH=4096`，脚本调优前 GA=4：
  - 日志：`/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-smoke-20260523-185431/v0-20260523-185543/logging.jsonl`
  - 结果：`global_step=1`，`loss=1.1104`，峰值显存 `45.09GiB`
- 长上下文 smoke，`MAX_LENGTH=18432`，`SEQUENCE_PARALLEL_SIZE=4`，`PADDING_FREE=true`，`USE_LOGITS_TO_KEEP=false`：
  - 日志：`/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-smoke-20260523-190932/v0-20260523-191042/logging.jsonl`
  - 结果：`global_step=1`，`loss=0.8809`，`token_acc=0.7499`，峰值显存 `46.5GiB`

8 卡、生产接近配置、`MAX_LENGTH=18432` benchmark 结果：

- `SP=4`，`bs=1`，`GA=4`：第一个生产 step 可跑通，但完整运行太慢，约 `97s/it`，每 step 只有 16 samples。
- `SP=2`，`bs=1`，`GA=4`：`0.28 samples/s`，峰值显存 `49.77GiB`。
- `SP=2`，`bs=2`，`GA=2`：`0.43 samples/s`，峰值显存 `49.92GiB`。
- `SP=1`，`bs=1`，`GA=4`，`USE_LOGITS_TO_KEEP=true`：`0.424 samples/s`，峰值显存 `49.31GiB`。
- `SP=1`，`bs=2`，`GA=2`，`USE_LOGITS_TO_KEEP=true`：`0.43 samples/s`，峰值显存 `52.34GiB`。
- `SP=1`，`bs=2`，`GA=4`，`USE_LOGITS_TO_KEEP=true`：`0.602 samples/s`，峰值显存 `51.13GiB`。这是当前默认配置。
- `SP=1`，`bs=4`，`GA=2`，`USE_LOGITS_TO_KEEP=true`：`0.584 samples/s`，峰值显存 `60.72GiB`；相比 `bs=2,GA=4` 更慢且显存余量更小。

已停止的 verification run：

- Screen：`qwen35_sft_200919`
- Run name：`qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919`
- 输出目录：`/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919`
- 日志：`/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/logs/train.log`
- 指标：`/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/v0-20260523-201036/logging.jsonl`
- 首个已验证 step：`global_step/max_steps=1/6055`，`loss=1.01296473`，`memory(GiB)=45.6`。
- 按操作者要求停止前的最新已验证 step：`global_step/max_steps=10/6055`，`loss=1.01102744`，`memory(GiB)=51.77`，`train_speed(s/it)=80.667375`，预计剩余时间 `5d 15h 27m 14s`。
- 状态：验证后已主动停止；预期不再存在该 run 的 `screen` session 或 GPU 进程。

监控命令：

```bash
screen -ls
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 200 /mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/logs/train.log
tail -n 20 /mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/v0-20260523-201036/logging.jsonl
```

## 算子与实现说明

- Qwen3.5 同时使用 linear attention 和 full attention，配置里 `full_attention_interval=4`，因此 FLA/causal-conv 路径和 flash attention 路径都需要确认。
- 9B 配置包含 32 个 text layer：24 个 `linear_attention` layer，8 个 `full_attention` layer。
- `flash-linear-attention`、`causal_conv1d`、`flash_attn`、`liger_kernel` 均已安装并可成功导入。
- `transformers` 检测结果为：`is_flash_linear_attention_available=True`、`is_causal_conv1d_available=True`、`is_flash_attn_2_available=True`、`is_fast_path_available=True`。
- Qwen3.5 GatedDeltaNet fast path 已命中：`causal_conv1d_fn`、`causal_conv1d_update`、`chunk_gated_delta_rule`、`fused_recurrent_gated_delta_rule`、`FusedRMSNormGated` 都解析到了已安装实现。
- 长上下文配合 sequence parallel 时，ms-swift 的长上下文模式要求 `USE_LOGITS_TO_KEEP=false`；此时 ms-swift 会提示 Liger cross-entropy 不会生效。
- 对当前 18k-token 配置，`SEQUENCE_PARALLEL_SIZE=1` 可以放下，并且保留 `USE_LOGITS_TO_KEEP=true`，因此 Liger cross-entropy 可以生效。
- Triton 当前版本是 `3.2.0`；部分库建议 `>=3.3.0`，但 PyTorch 2.6 固定依赖 Triton 3.2.0，且已验证的训练能够跑通，因此没有中途升级 Triton。

## 后续可选迭代

- 如果需要覆盖更多上下文，可尝试 `MAX_LENGTH=32768` 搭配 `SEQUENCE_PARALLEL_SIZE=4` 或 `8`。
- 如果全参数训练吞吐太慢，可用相同长上下文和 SP 设置测试 LoRA。
- 如果启动时间不再是主要问题、吞吐更重要，可测试 `LAZY_TOKENIZE=false GROUP_BY_LENGTH=true`，构建可复用的 tokenized/length cache。
