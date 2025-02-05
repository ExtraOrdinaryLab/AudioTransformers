#!/bin/bash

DEVICE="0"

MODEL_NAME="facebook/wav2vec2-base"
OUTPUT_DIR="/mnt/data4_HDD_14TB/yang/confit-checkpoints/timit/confit"

NUM_ITERATIONS=("10" "20")
MAX_LENGTH="3"
BATCH_SIZES=("256" "512" "1024")
LEARNING_RATES=("2e-5" "5e-5" "1e-4" "2e-4")

for ITERATIONS in "${NUM_ITERATIONS[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LEARNING_RATE in "${LEARNING_RATES[@]}"; do

            echo "Running training with batch size: $BATCH_SIZE, learning rate: $LEARNING_RATE, iterations: $ITERATIONS"
            EXP_NAME="iter$ITERATIONS-len$MAX_LENGTH-bs$BATCH_SIZE-lr$LEARNING_RATE"

            # Train the model
            CUDA_VISIBLE_DEVICES=$DEVICE /home/yang/miniconda3/envs/confit/bin/python mnr.py \
                --model_name_or_path $MODEL_NAME \
                --dataset_name "confit/timit" \
                --dataset_config_name "si" \
                --trust_remote_code \
                --output_dir "$OUTPUT_DIR/$EXP_NAME" \
                --num_iterations $ITERATIONS \
                --max_pairs "-1" \
                --eval_num_iterations "1" \
                --max_length_seconds $MAX_LENGTH \
                --mini_batch_size "8" \
                --per_device_train_batch_size $BATCH_SIZE \
                --per_device_eval_batch_size "32" \
                --dataloader_num_workers "4" \
                --gradient_checkpointing \
                --fp16 \
                --num_train_epochs "1" \
                --learning_rate $LEARNING_RATE \
                --report_to "none"

        done
    done
done