# Chunked Linear CE Loss 探索报告

日期：2026-05-25

## 1. 结论摘要

本轮探索的目标是降低 27B 长上下文 SFT 时最后一层 `hidden -> vocab logits -> CE loss` 的显存占用，尤其是 256K context 下完整 logits 张量可能非常大。

当前结论如下：

- Megatron / ms-swift 现有 `cross_entropy_loss_fusion=true` 仍然需要先实例化 output layer 产生的 vocab-parallel logits；它融合的是 CE/softmax 计算路径，不是完全跳过 logits 的 linear CE。
- Megatron Core 当前没有直接可用的“完全不实例化 logits”的原生训练算子。
- 已在本机 `.venv-megatron` 运行环境里实现了一个 chunked linear CE 路径，用 `LINEAR_CE_CHUNK_SIZE=2048` 控制。
- 该实现按扁平 token 维度切块，临时 logits 形状为 `[<=2048, vocab/TP]`，不是 `[batch, seq, vocab/TP]` 全量 logits。
- 4K smoke run 已确认命中 chunked linear CE 分支，loss 正常。
- 2026-05-25 重新跑了当前脚本的真实 Megatron SFT smoke：`verify-sft-chunkce-20260525-202158`。这次日志同时确认了 SFT mask 与 chunk CE 命中：4096 tokens 中 `ignore(-100)=3551`、`supervised=545`、`first_supervised_index=3366`，并打印 `[INFO:mcore_bridge] Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE=2048.`；训练 step 正常完成，loss 为 `0.29145688`。
- 256K run 已使用相同环境变量配置，第一步 loss/grad 与原 baseline 基本一致，显存记录从 `53.84 GiB` 到 `53.24 GiB`，但 `train.log` 没搜到 chunked CE 分支命中日志，因此这次端到端训练的显存改善不能完全归因于 chunked CE。
- 后续补充的单算子测试已经严格证明：只看 `hidden @ lm_head.T -> CE` 这段，chunked linear CE 有明确显存收益。262144 tokens 下 full logits + fp32 CE 路径在 A100 80G 上 OOM；即使不显式转 fp32、直接用 bf16 logits 做 CE，full 路径增量 peak allocated 也有 `60.63 GiB`，而 chunked 路径约 `4.17 GiB`。

当前最稳妥的训练方案仍然是：

```text
Megatron TP=8, PP=1, CP=1
sequence_parallel=true
optimizer_cpu_offload=true
optimizer_offload_fraction=1
LINEAR_CE_CHUNK_SIZE=2048
max_length=262144
loss_scale=default
```

其中 CP 不能开到 4 或 8，因为 Qwen3.5/Qwen3.6 的 Megatron `gated_delta_net` 路径当前要求 `context_parallel_size == 1`。

## 2. 背景与问题

27B SFT 的长上下文配置是 `max_length=262144`。当前数据集统计为：

```text
size=48
mean=70948.666667
std=33792.611034
min=17723
max=169996
```

所以 `262144` 覆盖了当前样本长度，可以避免截断轨迹。

在 TP=8 下，Qwen3.6 27B 的 padded vocab size 为：

```text
padded_vocab_size=248320
vocab_per_tp_rank=248320 / 8 = 31040
```

如果 256K context 直接实例化 vocab-parallel logits，则单 rank logits 大小近似为：

```text
[seq * batch, vocab/TP] = [262144, 31040]
```

显存量级约为：

```text
bf16 logits: 262144 * 31040 * 2 bytes ~= 15.2 GiB / rank
fp32 logits: 262144 * 31040 * 4 bytes ~= 30.3 GiB / rank
```

这就是为什么长上下文下只优化 attention/activation 还不够，最后的 logits + loss 路径也需要处理。

## 3. Loss 语义确认

SFT 训练继续使用：

```text
--loss_scale default
```

当前 loss 语义是只监督 assistant 输出，其余上下文全部 mask：

| 角色 | labels | 是否计算 loss |
| --- | --- | --- |
| system | `-100` | 否 |
| user | `-100` | 否 |
| tool / observation | `-100` | 否 |
| tool_call / tool_response | `-100` | 否 |
| assistant | token id | 是 |

也就是说，模型学习的是“给定系统、用户、环境反馈等上下文后，assistant 应该输出什么”。环境返回的 bash 输出、tool response 不作为模型要生成的内容参与 loss。

### 3.1 当前 SFT 路径复核

这次复核目标是确认 chunked CE 不是只在单算子测试里成立，而是在 `megatron sft` 的真实 SFT 训练路径里保持 SFT loss 语义。

运行命令：

```bash
RUN_NAME=verify-sft-chunkce-20260525-202158 \
SMOKE=1 \
LINEAR_CE_CHUNK_SIZE=2048 \
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

日志路径：

```text
logs/qwen36-27b-paper2arm-distill-megatron/verify-sft-chunkce-20260525-202158/train.log
outputs/qwen36-27b-paper2arm-distill-megatron/verify-sft-chunkce-20260525-202158/logging.jsonl
```

关键证据：

```text
[INFO:swift] [LABELS_IDS] {'summary': 'len=4096, omitted=3968, ignore(-100)=3551, supervised=545, first_supervised_index=3366', ...}
[INFO:mcore_bridge] Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE=2048.
{'loss': 0.29145688, 'grad_norm': 9.8802309, 'learning_rate': 1e-06, 'iteration': '1/1', 'memory(GiB)': 19.67, 'train_speed(s/it)': 99.010097}
```

代码路径确认：

1. `swift/loss_scale/base.py` 中 `default` 策略只对 `ContextType.RESPONSE` / assistant suffix 给非零 loss scale，其余上下文为 0。
2. 模板/tokenize 后的 labels 已体现 SFT mask：非监督 token 为 `-100`。
3. `swift/megatron/trainers/utils.py` 在 causal LM 下对 labels 做 `torch.roll(..., -1)`，保持 next-token 预测对齐。
4. `mcore_bridge.model.gpt_model._postprocess` 在 `labels is not None`、`task_type == causal_lm`、`LINEAR_CE_CHUNK_SIZE > 0`、非 inference 时直接返回 chunked linear CE 的 per-token loss，不再走完整 `output_layer -> logits -> compute_language_model_loss` 路径。
5. `_ChunkedLinearCrossEntropy.forward` 内对 `target == -100` 的 token 输出 0 loss。
6. `swift/megatron/trainers/trainer.py` 再用 `loss_mask = labels != -100` 聚合 loss，只除以 supervised token 数量。

因此，这条路径确实是 SFT loss：上下文/user/tool/observation 不参与 loss，assistant 输出 token 参与 loss；chunked CE 只替换了 per-token CE 的计算方式，不改变 SFT mask 语义。

### 3.2 单算子数值等价复核

为了排除“训练日志命中了 chunk CE，但 chunk CE 本身改变了 SFT loss 语义”的风险，又用训练脚本同一套环境执行了一个小规模 autograd 等价测试。

测试方式：

- `source ./megatron_env.sh` 后导入当前运行环境中的 `_ChunkedLinearCrossEntropy`。
- 构造 `[seq, batch, hidden]` hidden states、output weight 和 `[batch, seq]` labels。
- labels 中手动设置大段 `-100`，模拟 SFT 中 system/user/tool/observation 被 mask、assistant suffix 被监督的情况。
- 对比 chunk CE 与标准 `torch.nn.functional.cross_entropy(ignore_index=-100, reduction='none')`。
- 聚合方式与 trainer 一致：`(losses * (labels != -100)).sum() / (labels != -100).sum()`。

实测结果：

```text
python= /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/.venv-megatron/bin/python
torch= 2.6.0+cu126
labels_shape= (3, 17)
supervised_tokens= 20
ignored_tokens= 31
chunk_loss= 9.586441040039062
standard_loss= 9.586441040039062
max_per_token_loss_diff= 1.9073486328125e-06
max_hidden_grad_diff= 2.9802322387695312e-08
max_weight_grad_diff= 2.9802322387695312e-08
ignored_chunk_loss_abs_sum= 0.0
```

这个测试说明：

1. 在 `ignore_index=-100` 语义下，chunk CE 的 per-token loss 与标准 CE 数值一致，差异只在 float 误差级别。
2. 被 SFT mask 掉的 token 在 chunk CE 中 loss 为 0。
3. 对 hidden states 和 output weight 的反向梯度也与标准 CE 一致。

结合 3.1 的真实 `megatron sft` smoke，可以确认当前 chunk CE 在 SFT 中没有把 user/tool/observation token 纳入 loss，也没有改变 assistant-only supervision 的语义。

## 4. Megatron 原生 fused CE 调查结论

当前脚本里已有：

```text
--cross_entropy_loss_fusion true
```

日志中 Megatron 参数也显示：

```text
cross_entropy_loss_fusion=True
cross_entropy_fusion_impl='te'
calculate_per_token_loss=True
```

但这条路径的结构仍然是：

```text
hidden_states
  -> output_layer linear
  -> vocab-parallel logits
  -> fused cross entropy
  -> per-token loss
```

所以 fused CE 能减少 softmax/CE 内部中间张量，但不能消除 output layer 后完整 logits 的实例化。对于超长上下文，这个 logits 张量本身就是显存压力来源。

本轮没有在当前 Megatron Core / mcore_bridge 路径里找到可直接打开的原生 linear CE 或 chunked CE 开关。NeMo-RL 的 chunked linear CE 思路更接近我们需要的目标：按 token chunk 计算 `hidden @ lm_head.T + CE`，直接返回 per-token loss，避免整段上下文的全量 logits 常驻。

## 5. 当前实现

实现位置：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/.venv-megatron/lib/python3.12/site-packages/mcore_bridge/model/gpt_model.py
```

新增的主要逻辑：

- `_parse_linear_ce_chunk_size()`：读取环境变量 `LINEAR_CE_CHUNK_SIZE`，支持 `2048`、`2k`、`0/off/false`。
- `_ChunkedLinearCrossEntropy`：自定义 autograd function，forward/backward 都按 token chunk 重新计算局部 logits。
- `_chunked_linear_cross_entropy_loss()`：处理 sequence parallel gather、TP vocab partition index、梯度 reduce。
- `_postprocess` hook：满足条件时绕过 `self.output_layer(...)->logits` 的默认路径，直接返回 per-token CE loss。

触发条件：

```text
LINEAR_CE_CHUNK_SIZE > 0
labels is not None
task_type == causal_lm
not in_inference_mode
runtime_gather_output == false
use_mup == false
context_parallel_size == 1
```

命中后日志应出现：

```text
[INFO:mcore_bridge] Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE=2048.
```

## 6. Tensor shape 说明

这里的 chunk 是“扁平 token chunk”，不是 batch chunk，也不是 sequence chunk。

输入 hidden states：

```text
hidden_states: [seq, batch, hidden]
labels:        [batch, seq]
```

实现中先转成：

```text
hidden_flat: [seq * batch, hidden]
target_flat: [seq * batch]
```

然后按 `LINEAR_CE_CHUNK_SIZE=2048` 切：

```text
hidden_chunk: [<=2048, hidden]
logits_chunk: [<=2048, vocab/TP]
```

以当前 27B TP=8 为例：

```text
hidden_chunk: [<=2048, 5120]
output_weight_per_rank: [31040, 5120]
logits_chunk: [<=2048, 31040]
```

所以不是：

```text
[2048, batch, vocab/TP]
```

也不是：

```text
[batch/2048, seq, vocab/TP]
```

而是：

```text
[token_chunk, vocab/TP] = [<=2048, 31040]
```

这个临时 logits 如果按 fp32 计算，大约：

```text
2048 * 31040 * 4 bytes ~= 0.24 GiB / rank
```

相比完整 256K logits 的 15-30 GiB/rank 量级，理论峰值应显著下降。但实际训练显存还受 activation、CUDA allocator reserved memory、重计算、TE kernel workspace 等影响，所以日志里的 `memory(GiB)` 不一定能直接体现这一部分的全部收益。

## 7. 脚本变更

已更新脚本：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/train_qwen36_27b_paper2arm_distill_megatron.sh
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/launch_qwen36_27b_paper2arm_distill_with_fallback.sh
```

默认配置：

```bash
LINEAR_CE_CHUNK_SIZE="${LINEAR_CE_CHUNK_SIZE:-2048}"
export LINEAR_CE_CHUNK_SIZE
```

训练日志开头会打印：

```text
Chunked linear CE chunk size: 2048
```

关闭方式：

```bash
LINEAR_CE_CHUNK_SIZE=0 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

保留原生 fused CE：

```text
--cross_entropy_loss_fusion true
```

但在 chunked linear CE 命中时，代码会在 `_postprocess` 里直接返回 per-token loss，不再走默认 logits + fused CE 路径。

## 8. 验证结果

### 8.1 静态检查

已完成：

```text
python -m py_compile gpt_model.py
bash -n train_qwen36_27b_paper2arm_distill_megatron.sh
bash -n launch_qwen36_27b_paper2arm_distill_with_fallback.sh
```

结果均通过。

### 8.2 数值与梯度一致性

做过 standalone 对比测试，将 chunked linear CE 与标准 logits + CE 对齐。

结果：

```text
forward max diff     = 9.5367431640625e-07
hidden grad max diff = 2.384185791015625e-07
weight grad max diff = 2.384185791015625e-07
```

这个误差在 fp32 对齐测试下是正常浮点误差量级，说明当前公式和 backward 梯度基本正确。

### 8.3 4K smoke run

运行：

```text
SMOKE=1 RUN_NAME=qwen36-27b-paper2arm-distill-megatron-tp8-tokenchunkce-smoke-20260524-233155 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

日志：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-paper2arm-distill-megatron-tp8-tokenchunkce-smoke-20260524-233155/train.log
```

确认命中分支：

```text
[INFO:mcore_bridge] Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE=2048.
```

训练结果：

| 项 | 值 |
| --- | --- |
| max_length | `4096` |
| loss | `0.29104421` |
| grad_norm | `10.08913898` |
| memory(GiB) | `19.67` |
| train_speed(s/it) | `82.244807` |

结论：4K 下 chunked CE 分支可正常 forward/backward，loss 正常，没有 NaN/Inf。

### 8.4 256K 长上下文 run

运行：

```text
MAX_LENGTH=262144 TRAIN_ITERS=1 DATASET_NUM_PROC=1 DATALOADER_NUM_WORKERS=1 SAVE_STEPS=1000000 LINEAR_CE_CHUNK_SIZE=2048 RUN_NAME=qwen36-27b-paper2arm-distill-megatron-tp8-256k-tokenchunkce-20260524-234709 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

日志：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/logs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-paper2arm-distill-megatron-tp8-256k-tokenchunkce-20260524-234709/train.log
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/outputs/qwen36-27b-paper2arm-distill-megatron/qwen36-27b-paper2arm-distill-megatron-tp8-256k-tokenchunkce-20260524-234709/logging.jsonl
```

第一步结果：

| 项 | 值 |
| --- | --- |
| max_length | `262144` |
| loss | `0.20536332` |
| grad_norm | `0.69443589` |
| learning_rate | `1e-05` |
| iteration | `1/18` |
| memory(GiB) | `53.24` |
| train_speed(s/it) | `318.390549` |

对比之前未加 chunked linear CE 的 256K baseline：

```text
run: qwen36-27b-paper2arm-distill-megatron-tp8-20260524-173420
```

| 项 | baseline | chunk 配置 run |
| --- | ---: | ---: |
| loss step 1 | `0.20536338` | `0.20536332` |
| grad_norm step 1 | `0.69596642` | `0.69443589` |
| memory(GiB) | `53.84` | `53.24` |
| train_speed(s/it) | `509.881` | `318.390549` |

结论：

- loss 基本完全一致，说明训练语义没有明显偏移。
- grad_norm 很接近，说明反向路径至少没有明显异常。
- 记录显存降低约 `0.60 GiB`，不是理论 logits 显存量级的巨大下降。
- 速度记录变快，但这类单 step 速度受 cache、数据处理、warmup、日志等因素影响，不能作为严格性能结论。

重要 caveat：

这次 256K run 的脚本开头确实打印了：

```text
Chunked linear CE chunk size: 2048
```

但在 `train.log` 中没有搜到：

```text
Using chunked linear CE loss with LINEAR_CE_CHUNK_SIZE=2048.
```

因此这次 256K run 只能说明“配置已打开，loss/grad 正常，显存记录略低”，还不能完全证明 256K step 一定命中了 chunked CE 分支。需要下一轮加更强日志或计数器确认。

## 9. 显存结论

ms-swift Megatron 打印的 `memory(GiB)` 来自：

```text
torch.cuda.max_memory_reserved()
```

并在 model parallel group 上取最大值。对应代码位置：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/swift/megatron/callbacks/print.py
```

这意味着它统计的是 CUDA allocator reserved memory，不是瞬时 allocated memory，也不是单独 logits 张量的峰值。因此：

- 4K 下 logits 体积本来不大，`memory(GiB)=19.67` 没明显变化是正常的。
- 256K 下理论 logits 显存很大，但当前记录只下降 `0.60 GiB`，不能直接否定 chunked CE 的价值，也不能直接证明收益足够。
- 如果 full logits 曾经被 allocator reserve 后缓存，`max_memory_reserved()` 可能不会下降到理论值。
- 当前 256K 数据实际最长样本是 `169996` tokens，不是满 262144 tokens。
- 256K run 的 chunked CE 命中日志缺失，显存归因仍需补充验证。

所以当前对端到端训练显存收益的判断是：

```text
4K: 无明显变化，符合预期。
256K: 日志显存略降，但证据不足，不能声称已经显著节省。
理论上: 如果确实绕过完整 logits，CE 部分峰值应从 15-30 GiB/rank 量级降到约 0.24 GiB/rank 量级的 chunk 临时 logits。
```

补充说明：上面这段说的是“训练整链路日志”的归因。后续单算子测试已经单独证明 CE 算子部分确实节省显存，详见后面的“单算子显存验证”。

## 10. 当前限制与风险

1. 这是运行环境补丁，不是源码仓库补丁

当前修改在：

```text
.venv-megatron/lib/python3.12/site-packages/mcore_bridge/model/gpt_model.py
```

如果重装 `.venv-megatron`、升级 ms-swift 或重新安装 mcore_bridge，这个补丁可能被覆盖。

2. CP>1 不支持

代码里显式限制：

```text
LINEAR_CE_CHUNK_SIZE does not support context_parallel_size > 1.
```

脚本也会阻止 `CONTEXT_PARALLEL_SIZE != 1`。当前 Qwen3.5/Qwen3.6 Megatron `gated_delta_net` 自身也不支持 CP>1。

3. MuP 不支持

如果 `use_mup=true`，当前 chunked CE 会直接报错。

4. 256K 分支命中证据不足

4K smoke 已确认命中。256K run 没搜到命中日志，需要下一轮加更明确的 forward counter 或 rank0 强制日志。

5. 当前 backward 会重算 chunk logits

这种实现节省显存，但 backward 需要按 chunk 重新计算 logits，因此会增加计算量。`LINEAR_CE_CHUNK_SIZE` 越小，显存越低但循环次数越多；越大则速度更好但峰值更高。

6. `TRAIN_ITERS=1` 和 `NUM_TRAIN_EPOCHS=3` 同时出现

256K 测试里虽然传了 `TRAIN_ITERS=1`，但脚本非 smoke 分支仍默认 `NUM_TRAIN_EPOCHS=3`，日志显示 `iteration=1/18`。这说明参数层面不是一个干净的“一步退出”配置。后续要做严格对比，应修改脚本让 `TRAIN_ITERS` 与 `NUM_TRAIN_EPOCHS` 互斥。

## 11. 后续建议

1. 给 chunked CE 增加强制可观测性

建议在 `_ChunkedLinearCrossEntropy.forward` 内增加 rank0 只打印一次的日志，或者写入一个明确 counter，例如：

```text
linear_ce_forward_calls
linear_ce_total_tokens
linear_ce_chunk_size
```

这样能在 256K run 中明确证明分支是否命中。

2. 做严格 A/B 显存对比

建议用同一个样本顺序、同一个 fresh process，分别跑：

```bash
LINEAR_CE_CHUNK_SIZE=0
LINEAR_CE_CHUNK_SIZE=2048
```

同时记录：

```text
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

只看 reserved memory 不够，需要 allocated memory 辅助判断。

3. 测试不同 chunk size

建议尝试：

```text
512
1024
2048
4096
```

预期：

- chunk 越小，logits 峰值越低，但 matmul/通信循环更多。
- chunk 越大，速度更好，但显存更高。
- 当前默认 `2048` 是保守折中。

4. 把补丁沉淀为可复现补丁

建议不要长期只改 venv。可以增加：

```text
patches/mcore_bridge_chunked_linear_ce.patch
scripts/apply_chunked_linear_ce_patch.sh
```

或者直接维护 fork/source patch，保证环境重建后不会丢。

5. 若后续数据真到 256K+ 满长度

当前数据最长约 170K。后续如果数据普遍接近 256K 或更长，chunked CE 的必要性会更高，也更容易在 allocated memory 上看到收益。

## 12. 当前可执行命令

正式 27B 长上下文 SFT：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

显式使用 chunked CE：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
LINEAR_CE_CHUNK_SIZE=2048 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

关闭 chunked CE 做 baseline：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
LINEAR_CE_CHUNK_SIZE=0 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

4K smoke：

```bash
cd /mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift
SMOKE=1 LINEAR_CE_CHUNK_SIZE=2048 bash train_qwen36_27b_paper2arm_distill_megatron.sh
```

## 13. 单算子显存验证

为了把 transformer、dataloader、activation、CUDA allocator 缓存等因素剥离掉，后续增加了单算子 benchmark，只测这一段：

```text
hidden [tokens, hidden]
  @ lm_head.T [hidden, vocab/TP]
  -> logits [tokens, vocab/TP]
  -> CE
  -> backward
```

测试脚本：

```text
/mnt/cpfs/yangyicun/innovator-agent/training/sft/ms-swift/scripts/bench_chunked_linear_ce_memory.py
```

测试环境：

```text
GPU: NVIDIA A100-SXM4-80GB
hidden=5120
vocab_shard=31040
dtype=bf16
chunk_size=2048
backward=true
chunked path: forward/backward 都按 2048 token chunk 计算
```

full path 做了两个版本：

```text
full + fp32 CE: logits.float() 后做 F.cross_entropy，用来模拟更保守的 fp32 CE 内部计算压力。
full bf16 CE: 不显式 logits.float()，直接把 bf16 logits 交给 F.cross_entropy，用来隔离完整 logits 本身的压力。
```

测试指标：

```text
base_alloc_gib: 输入 hidden/weight/target 分配后的显存
peak_alloc_gib: 算子 forward+backward 过程中的 max_memory_allocated
incremental_peak_alloc_gib: peak_alloc_gib - base_alloc_gib
peak_reserved_gib: max_memory_reserved
```

结果一：full + fp32 CE 对比 chunked。

| tokens | full incremental peak allocated | chunked incremental peak allocated | full reserved peak | chunked reserved peak | loss 是否一致 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | `4.2707 GiB` | `1.7518 GiB` | `4.6504 GiB` | `2.2871 GiB` | 是 |
| 32768 | `17.0588 GiB` | `1.9862 GiB` | `17.6777 GiB` | `2.9707 GiB` | 是 |
| 65536 | `34.1097 GiB` | `2.2989 GiB` | `35.0410 GiB` | `3.5957 GiB` | 是 |
| 131072 | `68.2115 GiB` | `2.9241 GiB` | `69.7695 GiB` | `4.8477 GiB` | 是 |
| 262144 | OOM | `4.1746 GiB` | OOM | `7.3457 GiB` | full OOM，chunked 可完成 |

262144 tokens 的 full + fp32 CE 路径 OOM 信息：

```text
CUDA out of memory. Tried to allocate 30.31 GiB.
```

结果二：full bf16 CE 对比 chunked。

| tokens | full bf16 incremental peak allocated | chunked incremental peak allocated | full bf16 reserved peak | chunked reserved peak |
| ---: | ---: | ---: | ---: | ---: |
| 131072 | `30.3207 GiB` | `2.9241 GiB` | `31.8789 GiB` | `4.8477 GiB` |
| 262144 | `60.6334 GiB` | `4.1746 GiB` | `63.4395 GiB` | `7.3457 GiB` |

262144 tokens 的 chunked 路径结果：

```text
base_alloc_gib = 2.797974
peak_alloc_gib = 6.972569
incremental_peak_alloc_gib = 4.174594
peak_reserved_gib = 7.345703
loss = 10.343268
```

这个结果说明：

- 单算子层面，chunked linear CE 的显存收益已经被严格证明。
- full logits 的显存随 tokens 近似线性增长。如果 full CE 显式 fp32 计算，131072 tokens 已经需要约 `68.21 GiB` 的额外 allocated peak，262144 tokens 在 80G 卡上 OOM。
- 即使 full CE 不显式转 fp32，262144 tokens 仍需要约 `60.63 GiB` 的额外 allocated peak；chunked 只有约 `4.17 GiB`。
- chunked linear CE 的峰值主要受 `chunk_size`、`vocab/TP`、hidden/weight grad 影响，随 tokens 增长慢很多。
- loss 在 full + fp32 CE 可运行的点上完全一致，说明单算子语义对齐。full bf16 CE 的 loss 精度更低，不作为语义对齐依据，只作为保守显存对照。

需要注意：这个结论证明的是 CE 算子本身；端到端训练里还要受到 transformer activation、重计算、TE workspace、optimizer/offload、CUDA reserved memory 等影响，所以训练日志里的 `memory(GiB)` 不会等比例下降。

## 14. 总体判断

本轮已经完成了从“确认 fused CE 仍实例化 logits”到“实现一个可跑的 chunked linear CE 原型”的探索闭环。

当前最关键的事实是：

- 4K smoke 明确命中 chunked CE，loss 正常。
- standalone 数值/梯度对齐通过。
- 单算子显存 benchmark 已证明 chunked linear CE 明确节省显存：262144 tokens 下 full + fp32 CE OOM；full bf16 CE 也需要约 `60.63 GiB` 增量 peak allocated，而 chunked 约 `4.17 GiB`。
- 256K loss/grad 与 baseline 对齐，说明训练结果没有明显异常。
- 端到端训练显存收益的归因还没有被严格证明，原因是 256K 命中日志缺失以及现有训练指标是 reserved memory。

因此，`LINEAR_CE_CHUNK_SIZE=2048` 可以作为后续 27B 长上下文 SFT 的默认探索配置继续使用。现在可以明确说“CE 算子本身显著省显存”；如果要证明“完整训练 step 显著省显存”，还应补一次带强 instrumentation 的 256K A/B 测试。
