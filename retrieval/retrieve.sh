#!/usr/bin/env bash

DATASETS=(
    KP20k
    inspec
    nus
    semeval
    krapivin
    KPTimes
)

if [[ $# != 3 ]]; then
    echo "Must provide three arguments";
    echo "bash train.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

GPU=${1:-0};
DATASET_NAME=${2:-"KP20k"};
KEYWORD_TYPE=${3:-"present"};

if [[ ! " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash retrieve.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

DATA_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/data";
MODEL_BASE_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models";
OUT_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs;

if [[ $DATASET_NAME == "KPTimes" ]]; then
    OUTPUT_ENCODED_FILE=${OUT_DIR}/web_${KEYWORD_TYPE}_*.pkl
    CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/KPTimes_${KEYWORD_TYPE}";
else
    OUTPUT_ENCODED_FILE=${OUT_DIR}/scikp_${KEYWORD_TYPE}_*.pkl
    CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/KP20k_${KEYWORD_TYPE}";
fi

SPLIT=test
TOP_K=100

INPUT_FILE="${DATA_DIR}/${DATASET_NAME}.${SPLIT}.jsonl";
CKPT_FILENAME="checkpoint_best.pt";
OUT_FILE="${OUT_DIR}/${DATASET_NAME}_${SPLIT}_${KEYWORD_TYPE}_${TOP_K}.json"
LOG_FILE="${CHECKPOINT_DIR_PATH}/retrieval.log";

CODE_BASE_DIR=`realpath ..`;
script="${CODE_BASE_DIR}/retrieval/source/retrieve.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;

BATCH_SIZE=512;

if [[ ! -f $OUT_FILE ]]; then
    python ${script} \
        --fp16 \
        --keyword $KEYWORD_TYPE \
        --model_file ${CHECKPOINT_DIR_PATH}/${CKPT_FILENAME} \
        --qa_file $INPUT_FILE \
        --encoded_ctx_file $OUTPUT_ENCODED_FILE \
        --out_file $OUT_FILE \
        --n-docs $TOP_K \
        --batch_size $BATCH_SIZE \
        --match exact \
        --sequence_length 256 \
        --save_or_load_index 2>&1 | tee $LOG_FILE;
fi

script="${CODE_BASE_DIR}/retrieval/source/evaluate.py";
python ${script} --input_file $OUT_FILE;
