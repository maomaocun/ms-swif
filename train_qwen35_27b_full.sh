#!/bin/bash
set -e

# 模型和数据路径
MODEL_PATH="/mnt/cpfs/public_data/public_model/Qwen3.5/Qwen3.5-27B"
DATA_PATH="/mnt/cpfs/yangyicun/data/innovator_agent/sft_jsonl/innovator_agent_sft_h0.jsonl"
OUTPUT_DIR="/mnt/cpfs/yangyicun/output/qwen35-27b-full-sft-h0"

# 多卡配置
NPROC_PER_NODE=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$NPROC_PER_NODE \
swift sft \
    --model "$MODEL_PATH" \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --tuner_type full \
    --dataset "$DATA_PATH" \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed zero3_offload \
    --gradient_checkpointing true \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --split_dataset_ratio 0.05 \
    --attn_impl sdpa \
    --use_liger_kernel true
