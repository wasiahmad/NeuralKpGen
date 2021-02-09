#!/usr/bin/env bash

if [[ $# != 3 ]]; then
    echo "Must provide three arguments";
    echo "bash train.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

GPU=${1:-0};
DATASET_NAME=${2:-"KP20k"};
KEYWORD_TYPE=${3:-"present"};

DATA_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/data";
if [[ $DATASET_NAME != "KP20k" ]] && [[ $DATASET_NAME != "KPTimes" ]]; then
    echo "Dataset name must be either KP20k or KPTimes.";
    echo "bash script_train.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

train_file="${DATA_DIR}/${DATASET_NAME}.train.jsonl";
dev_file="${DATA_DIR}/${DATASET_NAME}.valid.jsonl";

MODEL_BASE_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models";
CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/${DATASET_NAME}_${KEYWORD_TYPE}";
mkdir -p $CHECKPOINT_DIR_PATH;

LOG_FILE="${CHECKPOINT_DIR_PATH}/training.log";
CKPT_FILENAME="dpr_biencoder";
pretrained_model="allenai/scibert_scivocab_uncased";

CODE_BASE_DIR=`realpath ..`;
script="${CODE_BASE_DIR}/retrieval/source/train.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

IFS=',' read -a GPU_IDS <<< "$1";
NUM_GPU=${#GPU_IDS[@]}

BATCH_SIZE=32;
PER_GPU_TRAIN_BATCH_SIZE=16;
if [[ "$(($PER_GPU_TRAIN_BATCH_SIZE * $NUM_GPU))" -gt "$BATCH_SIZE" ]]; then
    UPDATE_FREQ=1
else
    UPDATE_FREQ=$(($BATCH_SIZE / $(($PER_GPU_TRAIN_BATCH_SIZE * $NUM_GPU))));
fi

python ${script} \
    --dataset $DATASET_NAME \
    --keyword $KEYWORD_TYPE \
    --output_dir ${CHECKPOINT_DIR_PATH} \
    --checkpoint_file_name ${CKPT_FILENAME} \
    --fp16 \
    --batch_size $BATCH_SIZE \
    --dev_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $UPDATE_FREQ \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --sequence_length 256 \
    --num_train_epochs 5 \
    --eval_per_epoch 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg ${pretrained_model} \
    --val_av_rank_start_epoch 1 \
    --warmup_steps 1237 \
    --seed 1234 2>&1 | tee $LOG_FILE;
