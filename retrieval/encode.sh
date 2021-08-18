#!/usr/bin/env bash

if [[ $# -lt 1 ]]; then
    echo "Must provide one argument";
    echo "bash encode.sh <gpuids> [<dataset>] [<keyword-type>] [<context-type>]";
    exit;
fi

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`

GPU=${1:-0};
DATASET_NAME=${2:-"kp20k"};
KEYWORD_TYPE=${3:-"present"};
CONTEXT_TYPE=${4:-"abstract"};

DATA_DIR=$CURRENT_DIR/data;

if [[ $DATASET_NAME != "kp20k" ]] && [[ $DATASET_NAME != "kptimes" ]]; then
    echo "Dataset name must be either kp20k or kptimes.";
    echo "bash encode.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

FILES=()
if [[ $DATASET_NAME == "kp20k" ]]; then
    FILES+=(${DATA_DIR}/kp20k.train.jsonl)
    FILES+=(${DATA_DIR}/kp20k.valid.jsonl)
    FILES+=(${DATA_DIR}/kp20k.test.jsonl)
    FILES+=(${DATA_DIR}/inspec.test.jsonl)
    FILES+=(${DATA_DIR}/krapivin.test.jsonl)
    FILES+=(${DATA_DIR}/nus.test.jsonl)
    FILES+=(${DATA_DIR}/semeval.test.jsonl)
    encoder_model_type=hf_bert
    pretrained_model="allenai/scibert_scivocab_uncased";
    OUTPUT_FILE="${CURRENT_DIR}/outputs/scikp_${KEYWORD_TYPE}";
elif [[ $DATASET_NAME == "kptimes" ]]; then
    FILES+=(${DATA_DIR}/kptimes.train.jsonl)
    FILES+=(${DATA_DIR}/kptimes.valid.jsonl)
    FILES+=(${DATA_DIR}/kptimes.test.jsonl)
    encoder_model_type=hf_roberta
    pretrained_model="roberta-base";
    OUTPUT_FILE="${CURRENT_DIR}/outputs/web_${KEYWORD_TYPE}";
fi

context_length=512;
shard_size=100000;
if [[ $CONTEXT_TYPE == 'keywords' ]]; then
    OUTPUT_FILE=${OUTPUT_FILE}_${CONTEXT_TYPE};
    context_length=32;
fi

MODEL_BASE_DIR="${CURRENT_DIR}/models";
CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/${DATASET_NAME}_${KEYWORD_TYPE}";
CKPT_FILENAME="checkpoint_best.pt";
LOG_FILE="${CHECKPOINT_DIR_PATH}/encoding.log";

CODE_BASE_DIR=`realpath ..`;
script="${CODE_BASE_DIR}/retrieval/source/encode.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

BATCH_SIZE=512;

python $script \
    --fp16 \
    --dataset $DATASET_NAME \
    --encoder_model_type $encoder_model_type \
    --pretrained_model_cfg $pretrained_model \
    --model_file ${CHECKPOINT_DIR_PATH}/${CKPT_FILENAME} \
    --batch_size $BATCH_SIZE \
    --ctx_file "${FILES[@]}" \
    --shard_size $shard_size \
    --context_length $context_length \
    --context_type $CONTEXT_TYPE \
    --out_file $OUTPUT_FILE 2>&1 | tee $LOG_FILE;
