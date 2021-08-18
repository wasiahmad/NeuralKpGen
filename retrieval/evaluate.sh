#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`

DATASETS=(
    kp20k
    inspec
    nus
    semeval
    krapivin
    kptimes
)

if [[ $# -lt 1 ]]; then
    echo "Must provide at least one argument";
    echo "bash kpeval.sh <gpuids> [<dataset> <keyword-type> <model>]";
    exit;
fi

GPU=${1:-0};
DATASET_NAME=${2:-"kp20k"};
KEYWORD_TYPE=${3:-"present"};
MODEL=${4:-"mass"};
CONTEXT_TYPE=${5:-"abstract"};

if [[ ! " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash retrieve.sh <gpuids> <dataset> <keyword-type>";
    exit;
fi

DATA_DIR=${CURRENT_DIR}/data;
MODEL_BASE_DIR=${CURRENT_DIR}/models;
OUT_DIR=${CURRENT_DIR}/outputs;

if [[ $DATASET_NAME == "kptimes" ]]; then
    OUTPUT_ENCODED_FILE=${OUT_DIR}/web_${KEYWORD_TYPE};
    CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/kptimes_${KEYWORD_TYPE}";
else
    OUTPUT_ENCODED_FILE=${OUT_DIR}/scikp_${KEYWORD_TYPE};
    CHECKPOINT_DIR_PATH="${MODEL_BASE_DIR}/kp20k_${KEYWORD_TYPE}";
fi

if [[ $CONTEXT_TYPE == 'keywords' ]]; then
    OUTPUT_ENCODED_FILE=${OUTPUT_ENCODED_FILE}_${CONTEXT_TYPE}*.pkl
else
    OUTPUT_ENCODED_FILE=${OUTPUT_ENCODED_FILE}_*.pkl
fi

SPLIT=test; TOP_K=100;
INPUT_FILE="${DATA_DIR}/${DATASET_NAME}.${SPLIT}.jsonl";
CKPT_FILENAME="checkpoint_best.pt";
OUT_FILE="${OUT_DIR}/${MODEL}_${DATASET_NAME}_${SPLIT}_${CONTEXT_TYPE}_${KEYWORD_TYPE}_${TOP_K}.json"
LOG_FILE="${CHECKPOINT_DIR_PATH}/retrieval.log";

CODE_BASE_DIR=`realpath ..`;
script="${CODE_BASE_DIR}/retrieval/source/retrieve.py";

export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;
export CUDA_VISIBLE_DEVICES=$GPU;
BATCH_SIZE=512;

PRED_DIR=/home/wasiahmad/workspace/projects/NeuralKpGen
if [[ $MODEL == catSeq* ]]; then
    PREFIX=/home/wasiahmad/workspace/git_repos/keyphrase-generation-rl
    PRED_FILE="${PREFIX}/pred/${DATASET_NAME}_${MODEL}/predictions_filtered.txt";
elif [[ $MODEL = 'bart' ]] || [[ $MODEL = 'mass' ]]; then
    PRED_FILE="${PRED_DIR}/${MODEL}/logs/${DATASET_NAME}_predictions.txt";
else
    PRED_FILE="${PRED_DIR}/unilm/${MODEL}_${DATASET_NAME}/${DATASET_NAME}_predictions.txt";
fi

if [[ ! -f $OUT_FILE ]]; then
    python ${script} \
        --fp16 \
        --keyword $KEYWORD_TYPE \
        --model_file ${CHECKPOINT_DIR_PATH}/${CKPT_FILENAME} \
        --qa_file $INPUT_FILE \
        --pred_file $PRED_FILE \
        --encoded_ctx_file $OUTPUT_ENCODED_FILE \
        --out_file $OUT_FILE \
        --n-docs $TOP_K \
        --batch_size $BATCH_SIZE \
        --match exact \
        --question_length 32 \
        --save_or_load_index 2>&1 | tee $LOG_FILE;
fi

script="${CODE_BASE_DIR}/retrieval/source/evaluate.py";
python ${script} --input_file $OUT_FILE;
