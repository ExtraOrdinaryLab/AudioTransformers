#!/bin/bash

DEVICE="0"

MODEL_NAME="facebook/wav2vec2-base"
OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/timit/probe"

FREEZE_BASE_MODEL="True"
USE_WEIGHTED_LAYER_SUM="False"

MAX_LENGTH="3"
BATCH_SIZES=("16" "32")
LEARNING_RATES=("1e-4" "2e-4" "5e-4" "1e-3" "2e-3" "5e-3" "1e-2" "2e-2" "5e-2")
NUM_EPOCHS="20"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
        echo "Running training with batch size: $BATCH_SIZE, learning rate: $LEARNING_RATE"
        EXP_NAME="len$MAX_LENGTH-epoch$NUM_EPOCHS-bs$BATCH_SIZE-lr$LEARNING_RATE"

        CUDA_VISIBLE_DEVICES=$DEVICE /home/yang/miniconda3/envs/confit/bin/python finetune.py \
            --model_name_or_path $MODEL_NAME \
            --dataset_name "confit/timit" \
            --dataset_config_name "si" \
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
    done
done