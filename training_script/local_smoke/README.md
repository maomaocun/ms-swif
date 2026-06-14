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
- FP8 hybrid with GDN weight padding: loss `0.38547987`, speed `63.980134 s/it`

The FP8 path sets `MCORE_GDN_DISABLE_FP8_PROJ=false` and
`MCORE_GDN_PAD_TO_FP8_MULTIPLE=true` by default. Qwen3.6 GDN `in_proj` has a
TP-local output row count of `2060`, which is not a valid Transformer Engine FP8
matrix dimension. The local mcore-bridge patch pads that outer module weight to
`2064`, runs the projection in FP8, then slices the virtual channels away before
the GDN split/reshape logic sees them. To compare against the conservative
workaround that keeps only GDN projections in bf16, set:

```bash
MCORE_GDN_DISABLE_FP8_PROJ=true training_script/local_smoke/run_qwen36_27b_fp8_smoke.sh
```

Padding and loss masking:

- Swift's Megatron collator pads `input_ids`, `attention_mask`, and `labels` to
  the required multiple from `get_padding_to(args)`.
- Label padding uses `-100`, so padded positions are ignored by CE loss.
- The GDN FP8 issue is not fixed by dataset padding because the failing `2060`
  dimension is the TP-local `in_proj.weight` output row count, not the number of
  input tokens in the cached sample.
