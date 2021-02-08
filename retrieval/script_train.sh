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
    echo "bash train.sh <dataset> <gpuids>";
    exit;
fi

train_file="${DATA_DIR}/${DATASET_NAME}.train.jsonl";
dev_file="${DATA_DIR}/${DATASET_NAME}.valid.jsonl";

MODEL_BASE_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models";
CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/${DATASET_NAME}_${KEYWORD_TYPE}";
mkdir -p $CHECKPOINT_DIR_PATH;

LOG_FILE="${CHECKPOINT_DIR_PATH}/training.log";
CKPT_FILENAME="dpr_biencoder";
# pick model from https://huggingface.co/models?search=google/bert_uncase
pretrained_model="google/bert_uncased_L-6_H-512_A-8";

CODE_BASE_DIR=`realpath ../`;
script="train_encoder.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

BATCH_SIZE=64;
VALID_BATCH_SIZE=64;

python ${script} \
      --dataset $DATASET_NAME \
      --keyword $KEYWORD_TYPE \
      --output_dir ${CHECKPOINT_DIR_PATH} \
      --checkpoint_file_name ${CKPT_FILENAME} \
      --fp16 \
      --batch_size $BATCH_SIZE \
      --dev_batch_size $VALID_BATCH_SIZE \
      --train_file ${train_file} \
      --dev_file ${dev_file} \
      --sequence_length 256 \
      --num_train_epochs 20 \
      --eval_per_epoch 1 \
      --learning_rate 2e-5 \
      --max_grad_norm 2.0 \
      --encoder_model_type hf_bert \
      --pretrained_model_cfg ${pretrained_model} \
      --val_av_rank_start_epoch 1 \
      --warmup_steps 1237 \
      --seed 1234 2>&1 | tee $LOG_FILE;
