#!/bin/bash

DEVICE="1"

MODEL_NAME="facebook/wav2vec2-base"
OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/timit/finetune"

FREEZE_BASE_MODEL="False"
USE_WEIGHTED_LAYER_SUM="False"

MAX_LENGTH="3"
BATCH_SIZE="32"
LEARNING_RATE="1e-4"
NUM_EPOCHS="20"

CUDA_VISIBLE_DEVICES=$DEVICE /home/yang/miniconda3/envs/confit/bin/python finetune.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_name "confit/timit" \
    --dataset_config_name "si" \
    --audio_column_name "audio" \
    --label_column_name "label" \
    --output_dir $OUTPUT_DIR \
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
    --logging_steps "100" \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end "True" \
    --metric_for_best_model "accuracy" \
    --save_total_limit "1" \
    --report_to "none" \
    --seed "914"