# Qwen3.5-9B OpenThoughts SFT

This directory contains the local ms-swift launch script for full-parameter SFT on:

- Model: `/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-9B`
- Dataset: `/mnt/cpfs/yangyicun/data/datasets/openthoughts_prepared/openthoughts_sft_400k_filtered.jsonl`
- Runner: `train_qwen35_9b_openthoughts_full.sh`

## Development Summary

This change records the Qwen3.5-9B OpenThoughts SFT setup work done on this host:

- Restored the local ms-swift virtual environment after `.venv` pointed at a missing uv-managed Python interpreter.
- Verified the Qwen3.5 attention stack and required kernels: flash-linear-attention/FLA, causal-conv1d, flash-attn 2, Liger kernel, Triton, DeepSpeed, PyTorch, and Transformers.
- Confirmed the Qwen3.5-9B config uses 32 text layers: 24 linear-attention layers and 8 full-attention layers.
- Added `train_qwen35_9b_openthoughts_full.sh` as a reproducible SFT launcher with path checks, dry-run support, smoke mode, logging, cache isolation, and environment-variable overrides.
- Tuned production defaults from measured runs instead of short-context assumptions: `MAX_LENGTH=18432`, `SEQUENCE_PARALLEL_SIZE=1`, `PADDING_FREE=true`, `USE_LOGITS_TO_KEEP=true`, `PER_DEVICE_TRAIN_BATCH_SIZE=2`, `GRADIENT_ACCUMULATION_STEPS=4`, and `DEEPSPEED=zero2`.
- Measured dataset length distribution and confirmed 4k/8k context would truncate most examples.
- Ran fast smoke, long-context smoke, and several production-like benchmark variants to choose the current default.
- Started one full-run verification, confirmed it reached step 10, then stopped it intentionally when the operator decided a formal full run was no longer needed.
- No final checkpoint is expected from the stopped verification run because it stopped before `save_steps=500`.

## Commands

Fast startup smoke test:

```bash
SMOKE=1 ./train_qwen35_9b_openthoughts_full.sh
```

Long-context smoke test matching the production context settings:

```bash
SMOKE=1 \
MAX_LENGTH=18432 \
SEQUENCE_PARALLEL_SIZE=4 \
PADDING_FREE=true \
USE_LOGITS_TO_KEEP=false \
./train_qwen35_9b_openthoughts_full.sh
```

Production run:

```bash
./train_qwen35_9b_openthoughts_full.sh
```

Useful overrides:

```bash
# More complete but riskier context coverage.
MAX_LENGTH=32768 SEQUENCE_PARALLEL_SIZE=4 USE_LOGITS_TO_KEEP=false ./train_qwen35_9b_openthoughts_full.sh

# Lower memory fallback.
MAX_LENGTH=8192 SEQUENCE_PARALLEL_SIZE=1 PADDING_FREE=false USE_LOGITS_TO_KEEP=true ./train_qwen35_9b_openthoughts_full.sh
```

## Current Defaults

Production defaults are tuned for this dataset rather than generic short SFT:

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

Smoke defaults stay faster and intentionally use `MAX_LENGTH=4096`, `STREAMING=true`, and `GRADIENT_ACCUMULATION_STEPS=1`.

## Evidence

Environment repaired:

- `.venv` originally pointed to a missing uv-managed Python 3.12 interpreter.
- Restored with `uv python install 3.12.13`.
- Verified imports: `torch 2.6.0+cu126`, `transformers 5.8.1`, `deepspeed 0.19.0`, `flash_attn 2.8.3`, `fla 0.5.0`, `causal_conv1d 1.6.2.post1`, `liger_kernel 0.8.0`.

Dataset sizing:

- Rows: `387522`
- File size: about `19G`
- Character lengths: p50 `51911`, p75 `59523`, p90 `67003`, p95 `70682`, p99 `75379`, max `246613`
- Token sample on first 1000 rows: mean `15538.7`, p50 `17255`, p95 `17708`, max `18327`
- In the first 1000 rows, `MAX_LENGTH=4096` truncates `97.3%`; `MAX_LENGTH=8192` truncates `93.9%`; `MAX_LENGTH=16384` truncates `70.2%`.

Smoke results:

- Fast smoke, `MAX_LENGTH=4096`, GA=4 before script tuning:
  - Log: `/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-smoke-20260523-185431/v0-20260523-185543/logging.jsonl`
  - Result: `global_step=1`, `loss=1.1104`, peak memory `45.09GiB`
- Long-context smoke, `MAX_LENGTH=18432`, `SEQUENCE_PARALLEL_SIZE=4`, `PADDING_FREE=true`, `USE_LOGITS_TO_KEEP=false`:
  - Log: `/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-smoke-20260523-190932/v0-20260523-191042/logging.jsonl`
  - Result: `global_step=1`, `loss=0.8809`, `token_acc=0.7499`, peak memory `46.5GiB`

Benchmark results for production-like `MAX_LENGTH=18432` on 8 GPUs:

- `SP=4`, `bs=1`, `GA=4`: first production step worked but was too slow for a full run (`97s/it`, only 16 samples/step).
- `SP=2`, `bs=1`, `GA=4`: `0.28 samples/s`, peak `49.77GiB`.
- `SP=2`, `bs=2`, `GA=2`: `0.43 samples/s`, peak `49.92GiB`.
- `SP=1`, `bs=1`, `GA=4`, `USE_LOGITS_TO_KEEP=true`: `0.424 samples/s`, peak `49.31GiB`.
- `SP=1`, `bs=2`, `GA=2`, `USE_LOGITS_TO_KEEP=true`: `0.43 samples/s`, peak `52.34GiB`.
- `SP=1`, `bs=2`, `GA=4`, `USE_LOGITS_TO_KEEP=true`: `0.602 samples/s`, peak `51.13GiB`. Current default.
- `SP=1`, `bs=4`, `GA=2`, `USE_LOGITS_TO_KEEP=true`: `0.584 samples/s`, peak `60.72GiB`; slower and less memory headroom than `bs=2,GA=4`.

Stopped verification run:

- Screen: `qwen35_sft_200919`
- Run name: `qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919`
- Output: `/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919`
- Log: `/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/logs/train.log`
- Metrics: `/mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/v0-20260523-201036/logging.jsonl`
- First verified step: `global_step/max_steps=1/6055`, `loss=1.01296473`, `memory(GiB)=45.6`.
- Latest verified step before stopping per operator request: `global_step/max_steps=10/6055`, `loss=1.01102744`, `memory(GiB)=51.77`, `train_speed(s/it)=80.667375`, estimated remaining time `5d 15h 27m 14s`.
- Status: stopped intentionally after verification; no `screen` session or GPU process is expected to remain.

Monitor commands:

```bash
screen -ls
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
tail -n 200 /mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/logs/train.log
tail -n 20 /mnt/cpfs/yangyicun/output/qwen35-9b-openthoughts-sft/qwen35-9b-openthoughts-full-18432-sp1-bs2ga4-20260523-200919/v0-20260523-201036/logging.jsonl
```

## Operator Notes

- Qwen3.5 alternates linear attention and full attention (`full_attention_interval=4`), so both FLA/causal-conv and flash attention paths matter.
- The 9B config has 32 text layers: 24 `linear_attention` layers and 8 `full_attention` layers.
- `flash-linear-attention`, `causal_conv1d`, `flash_attn`, and `liger_kernel` are installed and imported successfully.
- `transformers` reports `is_flash_linear_attention_available=True`, `is_causal_conv1d_available=True`, `is_flash_attn_2_available=True`, and `is_fast_path_available=True`.
- The Qwen3.5 GatedDeltaNet fast path is active: `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, and `FusedRMSNormGated` all resolve to installed implementations.
- For long context with sequence parallel, `USE_LOGITS_TO_KEEP=false` is required by the ms-swift long-context pattern; ms-swift warns that Liger cross-entropy will not take effect in this mode.
- For this 18k-token run, `SEQUENCE_PARALLEL_SIZE=1` fits and keeps `USE_LOGITS_TO_KEEP=true`, so Liger cross-entropy can take effect.
- Triton is `3.2.0`; some libraries recommend `>=3.3.0`, but PyTorch 2.6 pins Triton 3.2.0 and the validated runs succeeded, so Triton was not upgraded.

## Next Iterations

- Try `MAX_LENGTH=32768` with `SEQUENCE_PARALLEL_SIZE=4` or `8` if more context coverage is required.
- If full-parameter throughput is too slow, test LoRA with the same long-context/SP settings.
- If startup time becomes less important than throughput, test `LAZY_TOKENIZE=false GROUP_BY_LENGTH=true` to build a reusable tokenized/length cache.
