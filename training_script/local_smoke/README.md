# Local Qwen3.6 27B Smoke

These scripts run a one-step Megatron SFT smoke on the local H100/L20Z image.

Defaults:

- Python: `/usr/local/bin/python`
- Model: `/model_cache/qwen36_27b`
- Dataset: `/model_cache/coding_trajectory/processed/smoke_trajectory_one.jsonl`
- TP: `8`
- SP: `true`
- CP/PP: `1`
- Optimizer CPU offload: `true`
- Chunked linear CE: `LINEAR_CE_CHUNK_SIZE=2048`
- Attention backend: `flash`
- Padding-free: `false`

Run BF16:

```bash
training_script/local_smoke/run_qwen36_27b_bf16_smoke.sh
```

Run FP8 hybrid:

```bash
training_script/local_smoke/run_qwen36_27b_fp8_smoke.sh
```

Compare losses:

```bash
training_script/local_smoke/compare_smoke_losses.py \
  --bf16-log /model_cache/smoke_logs/qwen36-27b-paper2arm-distill-megatron-tp8-smoke-20260613-162411/train.log \
  --fp8-log /model_cache/smoke_logs/qwen36-27b-paper2arm-distill-megatron-tp8-smoke-fp8-20260613-170045/train.log
```

Current verified local runs:

- BF16: loss `0.40912953`, speed `111.570275 s/it`
- FP8 hybrid: loss `0.39192179`, speed `66.329293 s/it`

The FP8 path sets `MCORE_GDN_DISABLE_FP8_PROJ=true` by default. Transformer
Engine can internally unpad Qwen3.6 GDN projection inputs to a valid-token count
such as `2060`, which is not divisible by 8 and fails FP8 execution. This keeps
the rest of the Transformer Engine model in FP8 while running GDN projections in
bf16. To test pure GDN FP8 on an aligned batch, set:

```bash
MCORE_GDN_DISABLE_FP8_PROJ=false training_script/local_smoke/run_qwen36_27b_fp8_smoke.sh
```

Padding and loss masking:

- Swift's Megatron collator pads `input_ids`, `attention_mask`, and `labels` to
  the required multiple from `get_padding_to(args)`.
- Label padding uses `-100`, so padded positions are ignored by CE loss.
- The GDN FP8 issue is downstream of the collator: Megatron/TE can compress to
  valid tokens before the GDN projection, so dataset-level padding alone does not
  guarantee FP8's token-multiple requirement inside that projection.
