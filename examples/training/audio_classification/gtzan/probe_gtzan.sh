#!/bin/bash

DEVICE="1"

# CHECKPOINT_DIR="facebook/wav2vec2-base"
# OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan/finetune"

# CHECKPOINT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan/confit/iter10-len10-bs32-lr5e-5/final/0_Transformer"
# OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan/confit/iter10-len10-bs32-lr5e-5/stage2"

CHECKPOINT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan/confit/iter10-len10-bs32-lr5e-5/stage2/len10-epoch10-bs16-lr5e-5"
OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/gtzan/confit/iter10-len10-bs32-lr5e-5/stage3"

FREEZE_BASE_MODEL="False"
USE_WEIGHTED_LAYER_SUM="False"

MAX_LENGTH="10"
BATCH_SIZE="16"
LEARNING_RATE="2e-4"
NUM_EPOCHS="10"

EXP_NAME="len$MAX_LENGTH-epoch$NUM_EPOCHS-bs$BATCH_SIZE-lr$LEARNING_RATE"

CUDA_VISIBLE_DEVICES=$DEVICE /home/yang/miniconda3/envs/confit/bin/python finetune.py \
    --model_name_or_path $CHECKPOINT_DIR \
    --dataset_name "confit/gtzan" \
    --audio_column_name "audio" \
    --label_column_name "label" \
    --output_dir $OUTPUT_DIR/$EXP_NAME \
    --overwrite_output_dir \
    --trust_remote_code "True" \
    --remove_unused_columns "False" \
    --freeze_feature_encoder "True" \
    --freeze_base_model $FREEZE_BASE_MODEL \
    --use_weighted_layer_sum $USE_WEIGHTED_LAYER_SUM \
    --eval_split_name "validation" \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 \
    --learning_rate $LEARNING_RATE \
    --max_length_seconds $MAX_LENGTH \
    --return_attention_mask "False" \
    --warmup_ratio "0.1" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps "1" \
    --per_device_eval_batch_size "1" \
    --dataloader_num_workers "8" \
    --logging_strategy "steps" \
    --logging_steps "1000" \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end "True" \
    --metric_for_best_model "accuracy" \
    --save_total_limit "1" \
    --report_to "none" \
    --seed "914"