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
    echo "bash encode.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

FILES=()
if [[ $DATASET_NAME == "KP20k" ]]; then
    FILES+=(${DATA_DIR}/KP20k.train.jsonl)
    FILES+=(${DATA_DIR}/KP20k.valid.jsonl)
    FILES+=(${DATA_DIR}/KP20k.test.jsonl)
    FILES+=(${DATA_DIR}/inspec.test.jsonl)
    FILES+=(${DATA_DIR}/krapivin.test.jsonl)
    FILES+=(${DATA_DIR}/nus.test.jsonl)
    FILES+=(${DATA_DIR}/semeval.test.jsonl)
    encoder_model_type=hf_bert
    pretrained_model="allenai/scibert_scivocab_uncased";
    OUTPUT_FILE="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs/scikp";
elif [[ $DATASET_NAME == "KPTimes" ]]; then
    FILES+=(${DATA_DIR}/KPTimes.train.jsonl)
    FILES+=(${DATA_DIR}/KPTimes.valid.jsonl)
    FILES+=(${DATA_DIR}/KPTimes.test.jsonl)
    encoder_model_type=hf_bert
    pretrained_model="bert-base-uncased";
    OUTPUT_FILE="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs/web";
fi

MODEL_BASE_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models";
CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/${DATASET_NAME}_${KEYWORD_TYPE}";
CKPT_FILENAME="checkpoint_best.pt";
LOG_FILE="${CHECKPOINT_DIR_PATH}/encoding.log";

CODE_BASE_DIR=`realpath ..`;
script="${CODE_BASE_DIR}/retrieval/source/encode.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

BATCH_SIZE=128;

python ${script} \
    --dataset $DATASET_NAME \
    --encoder_model_type $encoder_model_type \
    --pretrained_model_cfg $pretrained_model \
    --model_file ${CHECKPOINT_DIR_PATH}/${CKPT_FILENAME} \
    --batch_size $BATCH_SIZE \
    --ctx_file "${FILES[@]}" \
    --shard_size 100000 \
    --out_file $OUTPUT_FILE 2>&1 | tee $LOG_FILE;
