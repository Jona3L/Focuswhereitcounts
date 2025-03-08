#!/bin/bash
# run_train_MLP.sh
# This script launches the LLAVA MLP fine‑tuning using DeepSpeed

echo "Hello, this is a test message. We are working with MLP llava finetune-v1"

#!/bin/bash
deepspeed /scratch/jl9356/salience_llava/LLaVA/llava/train/train_mem_MLP.py \
    --deepspeed /scratch/jl9356/salience_llava/LLaVA/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --data_path /scratch/jl9356/salience_llava/dataset/split/train_annotation_hpc.json \
    --image_folder /scratch/jl9356/salience_llava/dataset/split/train \
    --version v1 \
    --freeze_backbone True \
    --mm_projector_lr 2e-5 \
    --learning_rate 1e-5 \
    --num_train_epochs 7 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    # --model_max_length 3072 \
    --save_steps 200 \
    --save_total_limit 1 \
    --output_dir ./checkpoints/llava-v1.5-7b-task-MLP_only