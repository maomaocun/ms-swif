#!/bin/bash
# Training script for paper2arm data on Qwen3.6-27B using ms-swift
#
# Loss mask strategy:
#   --loss_scale default
#   This automatically masks system/user/tool tokens (-100) and only computes
#   loss on assistant tokens (thinking + action + response).

set -e

# ============================
# Configuration
# ============================
MODEL="Qwen/Qwen3.6-27B"
TEMPLATE="qwen3_5"
DATASET="data/processed/paper2arm_legacy/paper2arm_train.jsonl"
OUTPUT_DIR="output/paper2arm-qwen36-27b"

# Training hyperparameters
MAX_LENGTH=32768        # Max sequence length; paper2arm trajectories are long
BATCH_SIZE=1            # Per-device batch size (adjust based on GPU memory)
GRADIENT_ACCUMULATION=8 # Effective batch size = 1 * 8 * num_gpus
LEARNING_RATE=1e-5
NUM_EPOCHS=3
WARMUP_RATIO=0.05
SAVE_STEPS=50
LOGGING_STEPS=5

# LoRA config (set USE_LORA=true for LoRA, false for full-parameter)
USE_LORA=true
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

# ============================
# Build command
# ============================
CMD="swift sft \
  --model ${MODEL} \
  --template ${TEMPLATE} \
  --dataset ${DATASET} \
  --output_dir ${OUTPUT_DIR} \
  --max_length ${MAX_LENGTH} \
  --loss_scale default \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${NUM_EPOCHS} \
  --warmup_ratio ${WARMUP_RATIO} \
  --save_steps ${SAVE_STEPS} \
  --logging_steps ${LOGGING_STEPS} \
  --save_total_limit 3 \
  --logging_first_step true \
  --dataloader_num_workers 4"

if [ "$USE_LORA" = true ]; then
  CMD="${CMD} \
    --tuner_backend peft \
    --target_modules all-linear \
    --init_lora_weights true \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT}"
else
  CMD="${CMD} \
    --tuner_backend swift"
fi

# ============================
# Optional: deepspeed / multi-GPU
# ============================
# For multi-GPU training, uncomment:
# CMD="${CMD} --deepspeed default-zero2"

# ============================
# Run
# ============================
echo "Running command:"
echo "${CMD}"
echo ""
mkdir -p "${OUTPUT_DIR}"
eval "${CMD}"
