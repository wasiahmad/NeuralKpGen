#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

# for tiny, mini, small, medium sized BERT - check
# https://huggingface.co/models?search=google%2Fbert_uncased_L-
AVAILABLE_MODEL_CHOICES=(
    unilm
    minilm
    bert-tiny
    bert-mini
    bert-small
    bert-medium
    bert-base
    scibert
    roberta
    bart
)

GPU=${1:-0}
MODEL_NAME=${2:-scibert}
USE_CRF=${3:-false}

if [[ $USE_CRF == true ]]; then
    USE_CRF="--use_crf"; SUFFIX="_crf";
else
    USE_CRF=""; SUFFIX="";
fi

export CUDA_VISIBLE_DEVICES=$GPU
export MODEL_TYPE=bert


function train () {

python $CURRENT_DIR/source/run_tag.py \
    --fp16 $USE_CRF \
    --data_dir $DATA_DIR_PREFIX/kp20k/bioformat \
    --dataset_name kp20k \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 384 \
    --max_steps 20000 \
    --warmup_steps 1000 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --seed 1234 \
    --learning_rate 1e-4 \
    --do_train \
    --log_file $OUTPUT_DIR/train.log \
    --eval_patience 5 \
    --overwrite_output_dir \
    --save_only_best_checkpoint \
    --workers 60;

}


function evaluate () {

EVAL_DATASET=$1

python $CURRENT_DIR/source/run_tag.py \
    --fp16 $USE_CRF \
    --data_dir $DATA_DIR_PREFIX/$EVAL_DATASET/bioformat \
    --dataset_name $EVAL_DATASET \
    --model_type $MODEL_TYPE \
    --model_name_or_path $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size 32 \
    --seed 1234 \
    --do_predict \
    --log_file $OUTPUT_DIR/$1-eval.log \
    --workers 60;

}


function compute_score () {

EVAL_DATASET=$1

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_dir ${DATA_DIR_PREFIX}/$EVAL_DATASET/bioformat \
    --file_prefix $OUTPUT_DIR/$EVAL_DATASET \
    --tgt_dir $OUTPUT_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 10 M;

}


while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID MODEL_NAME USE_CRF"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "MODEL_NAME     Model name; choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
        echo "USE_CRF        Boolean [true | false]"
        echo
        exit;;
   esac
done


if [[ $MODEL_NAME == 'unilm' ]]; then
    export BERT_MODEL=unilm-base-cased
elif  [[ $MODEL_NAME == 'minilm' ]]; then
    export BERT_MODEL=microsoft/MiniLM-L12-H384-uncased
elif  [[ $MODEL_NAME == 'bert-tiny' ]]; then
    export BERT_MODEL=google/bert_uncased_L-2_H-128_A-2
elif  [[ $MODEL_NAME == 'bert-mini' ]]; then
    export BERT_MODEL=google/bert_uncased_L-4_H-256_A-4
elif  [[ $MODEL_NAME == 'bert-small' ]]; then
    export BERT_MODEL=google/bert_uncased_L-4_H-512_A-8
elif  [[ $MODEL_NAME == 'bert-medium' ]]; then
    export BERT_MODEL=google/bert_uncased_L-8_H-512_A-8
elif  [[ $MODEL_NAME == 'bert-base' ]]; then
    export BERT_MODEL=bert-base-uncased
elif  [[ $MODEL_NAME == 'scibert' ]]; then
    export BERT_MODEL=allenai/scibert_scivocab_uncased
elif  [[ $MODEL_NAME == 'roberta' ]]; then
    export MODEL_TYPE=roberta
    export BERT_MODEL=roberta-base
elif  [[ $MODEL_NAME == 'bart' ]]; then
    export MODEL_TYPE=bart
    export BERT_MODEL=facebook/bart-base
else
    echo -n "... Wrong model choice!! available choices: \
                [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]" ;
    exit 1
fi

export OUTPUT_DIR=${CURRENT_DIR}/kp20k-${BERT_MODEL//\//_}${SUFFIX}
export MODEL_NAME_OR_PATH=$BERT_MODEL
mkdir -p $OUTPUT_DIR

if [[ $BERT_MODEL == 'unilm-base-cased' ]]; then
    MODEL_NAME_OR_PATH=${CURRENT_DIR}/models/${BERT_MODEL}
fi


DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
train
for dataset in kp20k krapivin nus inspec semeval; do
    evaluate $dataset
    compute_score $dataset
done
