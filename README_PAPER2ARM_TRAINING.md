# Paper2ARM 轨迹蒸馏数据 → ms-swift 训练指南

## Qwen3.5-9B 轨迹蒸馏

参考 OpenThoughts 的 9B 全参脚本，新增了面向 paper2arm Qwen3.7-max teacher 轨迹的启动脚本：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift

# 快速 smoke，只验证启动链路。
SMOKE=1 ./train_qwen35_9b_paper2arm_distill_full.sh

# 生产默认：reward>=0.6 数据，262k 上下文，SP=4，全参 SFT。
./train_qwen35_9b_paper2arm_distill_full.sh
```

默认数据：

```text
data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl
```

关键默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `MODEL_PATH` | `/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B` | student |
| `DATASET_PATH` | `data/paper2arm_qwen37_max_sft_reward_ge_0.6.jsonl` | teacher 轨迹，reward>=0.6 |
| `OUTPUT_ROOT` | `outputs/qwen35-9b-paper2arm-distill-sft` | ms-swift 训练产物默认写到当前目录下 |
| `LOG_ROOT` | `logs/qwen35-9b-paper2arm-distill-sft` | 外层 `train.log` 默认写到 ms-swift 目录下 |
| `CACHE_ROOT` | `cache/ms-swift-paper2arm-distill` | HF/datasets/modelscope cache 默认写到 ms-swift 目录下 |
| `LOSS_SCALE` | `default` | ms-swift 默认 SFT mask |
| `MAX_LENGTH` | `262144` | Qwen3.5-9B tokenizer 上限；当前 48 条 reward>=0.6 轨迹全部不截断 |
| `TRUNCATION_STRATEGY` | `delete` | 不做 silent truncation；若未来样本超过上限则过滤而不是截断 |
| `SEQUENCE_PARALLEL_SIZE` | `4` | ms-swift HF SFT 的长上下文并行参数；Qwen3.5 linear attention 不支持 SP=8 派生的 ring attention |
| `USE_LOGITS_TO_KEEP` | `false` | Qwen3.5 forward 支持按 label 选择性计算 logits，但 ms-swift 当前 `prepare_logits_to_keep()` 不支持 `sequence_parallel_size>1` |
| `CELOSS_PARALLEL_SIZE` | `2048` | SP 路径下启用 `ChunkedCrossEntropyLoss`，分块计算 CE，避免 CE 本身再额外放大显存 |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | `1` | 降低长上下文显存压力 |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 小数据集保持有效 batch |
| `LEARNING_RATE` | `5e-6` | 全参蒸馏保守学习率 |

### 长上下文 logits 显存路径

Qwen3.5 的 HF forward 有 `logits_to_keep` 参数，会先按 mask/slice 选择 hidden states，再执行 `lm_head`，因此能避免完整 `[batch, seq, vocab]` logits 实例化。ms-swift 的 SFT trainer 也有 `--use_logits_to_keep`，但本地源码里 `prepare_logits_to_keep()` 在 `sequence_parallel_size > 1` 时直接 `NotImplementedError`，所以不能和当前 SP=4 同时打开。

当前 262k 默认选择是：
- `SEQUENCE_PARALLEL_SIZE=4`：把序列维度切到 4 卡上，Qwen3.5 linear attention 也已走 ms-swift 的 SP patch。
- `USE_LOGITS_TO_KEEP=false`：避免触发 ms-swift 的 SP 不支持路径。
- `CELOSS_PARALLEL_SIZE=2048`：SP loss 路径进入 `per_token_loss_func_sp()`，并用 `ChunkedCrossEntropyLoss` 分块算 CE。

如果要强制使用 `logits_to_keep`，需要改成 `SEQUENCE_PARALLEL_SIZE=1 USE_LOGITS_TO_KEEP=true`，但这会失去长上下文 sequence parallel，262k 全参训练更容易被 attention/activation 显存卡住。

当前用 Qwen3.5-9B + `qwen3_5` 模板实测长度：

| 阈值 | 完整覆盖样本数 |
|---:|---:|
| 32,768 | 5/48 |
| 65,536 | 25/48 |
| 98,304 | 35/48 |
| 131,072 | 46/48 |
| 196,608 | 48/48 |
| 262,144 | 48/48 |

长度分位：p50 `61,438`，p75 `99,527`，p95 `126,176`，max `169,996` tokens。默认开到 `262144` 是为了贴住模型上限并保证不截断。

## Loss Mask 结论

`--loss_scale default` 的源码逻辑在 `swift/loss_scale/base.py`：只对 `ContextType.RESPONSE/SUFFIX` 计算 loss，其他 context mask 为 `-100`。

对 paper2arm agent 数据要注意一个 ms-swift 细节：

| 数据 role | ms-swift swift backend 行为 | 默认是否算 loss |
|---|---|---|
| `system` | system prompt/context | 否 |
| `user` | user query/context | 否 |
| `tool_response` / `tool` | 转成下一轮 tool query，即环境反馈 | 否 |
| `assistant` | assistant response | 是 |
| `tool_call` | 先渲染成 assistant 的 `<tool_call>...bash command...</tool_call>` 文本 | 是 |

所以，当前默认训练会学习 teacher 的思考、bash tool call 和最终回复；不会学习 bash 命令输出结果。原始 `tool_call` JSON 只是数据中间表示，训练时会被 qwen3_5 agent template 渲染成模型需要输出的工具调用文本。

如果确实要让 `tool_call` 也完全 `labels=-100`，转换数据时加 `--mask-tool-calls`：

```bash
python3 scripts/convert_paper2arm_qwen37_to_swift.py \
  --input-dir data/paper2arm_Qwen3.7-max \
  --output-file data/paper2arm_qwen37_max_sft_reward_ge_0.6_mask_tool_calls.jsonl \
  --min-reward 0.6 \
  --mask-tool-calls

python3 scripts/verify_swift_format.py \
  --input-file data/paper2arm_qwen37_max_sft_reward_ge_0.6_mask_tool_calls.jsonl \
  --expect-mask-tool-calls
```

但这会同时 mask 掉模型实际要输出的 bash command/tool-call 文本，不适合作为默认的 agent 轨迹蒸馏目标。

## 数据格式转换

### 源数据
- 路径：`/mnt/cpfs/guixiyan/innovator-agent/trials/runs/paper2arm-full-ack-c90-qwen3-6-max-preview-20260517153117`
- 75 个 trial 目录，每个包含 `agent/trajectory.json`
- trajectory 结构：system → user → agent(thinking+action) → observation(tool result) → ...

### 转换脚本
```bash
python3 scripts/convert_paper2arm_to_swift.py \
  --input-dir /mnt/cpfs/guixiyan/innovator-agent/trials/runs/paper2arm-full-ack-c90-qwen3-6-max-preview-20260517153117 \
  --output-file data/paper2arm_train.jsonl
```

### 转换逻辑
| 源数据 | ms-swift message role | 是否计算 loss |
|--------|----------------------|--------------|
| `source=system` | `system` | ❌ (被 default 策略 mask) |
| `source=user` | `user` | ❌ (被 default 策略 mask) |
| `source=agent` (THOUGHT + action) | `assistant` | ✅ (计算 loss) |
| `observation` (命令执行结果) | `tool` | ❌ (被 default 策略 mask) |

### Thinking 格式适配
Qwen3.5/3.6 使用 `<think>\n...\n</think>\n\n` 包裹 reasoning：
- 有 `THOUGHT:` 的 step：提取 thinking 放入 `<think>` 标签
- 无 `THOUGHT:` 的 step：使用空 think `<think>\n\n</think>\n\n{action}`

## 验证

```bash
python3 scripts/verify_swift_format.py --input-file data/paper2arm_train.jsonl
```

验证结果（75 个样本）：
- ✅ 全部通过格式校验
- 平均每个样本 ~68 条消息，~35K tokens
- assistant 内容占比 ~70%（这部分计算 loss）

## 训练命令

### LoRA 微调（推荐，显存友好）
```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --template qwen3_5 \
  --dataset data/paper2arm_train.jsonl \
  --output_dir output/paper2arm-qwen36-27b-lora \
  --max_length 32768 \
  --loss_scale default \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.05 \
  --tuner_backend peft \
  --target_modules all-linear \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --save_steps 50 \
  --logging_steps 5
```

### 全参数微调（需要更多显存）
```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --template qwen3_5 \
  --dataset data/paper2arm_train.jsonl \
  --output_dir output/paper2arm-qwen36-27b-full \
  --max_length 32768 \
  --loss_scale default \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --warmup_ratio 0.05 \
  --tuner_backend swift \
  --save_steps 50 \
  --logging_steps 5
```

### 多卡训练（DeepSpeed ZeRO-2）
添加 `--deepspeed default-zero2` 即可自动启用多卡。

## Loss Mask 说明

使用 `--loss_scale default`（默认值，可省略）：

```
- system tokens  → labels = -100  (mask)
- user tokens    → labels = -100  (mask)
- tool tokens    → labels = -100  (mask)  ← observation 不算 loss
- assistant tokens → labels = input_ids  ← thinking + action + response 算 loss
```

这与你的需求完全一致：
- ✅ 搜索/执行结果（observation）加入上下文但不计算 loss
- ✅ 只有 assistant 的 thinking、调用工具、回答计算 loss

## 关键超参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `max_length` | 32768 | paper2arm trajectory 很长，单条约 35K tokens |
| `learning_rate` | 1e-5 (LoRA) / 5e-6 (全参数) | 蒸馏数据用较小 LR |
| `num_train_epochs` | 3 | 75 条数据，epoch 不宜过多防止过拟合 |
| `warmup_ratio` | 0.05 | 标准 warmup |
| `lora_rank` | 64 | 适中 rank，平衡效果和显存 |
| `lora_alpha` | 128 | alpha = 2 * rank |

## 注意事项

1. **transformers 版本**：Qwen3.6 需要 `transformers>=5.0.0.dev`，请确保环境满足。
2. **上下文长度**：若显存不足，可将 `max_length` 降至 16384 或 8192，但会截断长 trajectory。
3. **数据量**：仅 75 条样本，建议配合适当的数据增强或与其他 SFT 数据混合训练。
