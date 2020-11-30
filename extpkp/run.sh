#!/usr/bin/env bash

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

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID MODEL_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "MODEL_NAME     Model name; choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
        echo
        exit;;
   esac
done


BASE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/extpkp
mkdir -p $BASE_DIR

export MAX_LENGTH=464
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=16
export GRADIENT_ACCUM_STEPS=1
export MAX_STEPS=20000
export WARMPUP_STEPS=1000
export SAVE_STEPS=2000
export LR=1e-4
export SEED=1


function train () {

mkdir -p $OUTPUT_DIR

python run_tag.py \
--fp16 \
--data_dir ${BASE_DIR}/data/processed/kp20k \
--dataset_name kp20k \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_LENGTH \
--max_steps $MAX_STEPS \
--warmup_steps $WARMPUP_STEPS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
--save_steps $SAVE_STEPS \
--seed $SEED \
--learning_rate $LR \
--do_train \
--log_file $OUTPUT_DIR/train.log \
--eval_patience 5 \
--overwrite_output_dir \
--save_only_best_checkpoint \
--workers 60;

}


function evaluate () {

python run_tag.py \
--fp16 \
--data_dir ${BASE_DIR}/data/processed/$1 \
--dataset_name $1 \
--model_type $MODEL_TYPE \
--model_name_or_path $OUTPUT_DIR \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_LENGTH \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--seed $SEED \
--do_predict \
--log_file $OUTPUT_DIR/$2-${BERT_MODEL//\//_}-eval.log \
--workers 60;

}


function compute_score () {

python -W ignore evaluate.py \
--src_dir ${BASE_DIR}/data/processed/$1 \
--pred_file ${OUTPUT_DIR}/$1_predictions.txt \
--tgt_dir ${OUTPUT_DIR} \
--log_file $1 \
--k_list 5 10 M;

}


export CUDA_VISIBLE_DEVICES=$1
export MODEL_NAME=$2
export MODEL_TYPE=bert


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
    export BERT_MODEL=roberta-base
elif  [[ $MODEL_NAME == 'bart' ]]; then
    export MODEL_TYPE=bart
    export BERT_MODEL=facebook/bart-base
else
    echo -n "... Wrong model choice!! available choices: \
                [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]" ;
    exit 1
fi


export OUTPUT_DIR=${BASE_DIR}/kp20k-${BERT_MODEL//\//_}
export MODEL_NAME_OR_PATH=$BERT_MODEL

if [[ $BERT_MODEL == 'unilm-base-cased' ]]; then
    MODEL_NAME_OR_PATH=${BASE_DIR}/models/${BERT_MODEL}
fi


train
for dataset in kp20k krapivin nus inspec semeval; do
    evaluate $dataset
    compute_score $dataset
done
