#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/extpkp
mkdir -p $BASE_DIR

export MODEL_TYPE=bert
export BERT_MODEL=bert-base-uncased
#export BERT_MODEL=roberta-base
#export BERT_MODEL=microsoft/MiniLM-L12-H384-uncased
#export BERT_MODEL=allenai/scibert_scivocab_uncased
#export BERT_MODEL=unilm-base-cased

#export MODEL_TYPE=bart
#export BERT_MODEL=bart-base

export MAX_LENGTH=464
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=16
export GRADIENT_ACCUM_STEPS=1
export MAX_STEPS=20000
export WARMPUP_STEPS=1000
export SAVE_STEPS=2000
export LR=1e-4
export SEED=1

export OUTPUT_DIR=${BASE_DIR}/kp20k-${BERT_MODEL//\//_}
export MODEL_NAME_OR_PATH=$BERT_MODEL

if [[ $BERT_MODEL == 'unilm-base-cased' ]]; then
    MODEL_NAME_OR_PATH=${BASE_DIR}/models/${BERT_MODEL}
fi


function train () {

export CUDA_VISIBLE_DEVICES=$1
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

export CUDA_VISIBLE_DEVICES=$1

python run_tag.py \
--fp16 \
--data_dir ${BASE_DIR}/data/processed/$2 \
--dataset_name $2 \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
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

train $1
for dataset in kp20k krapivin nus inspec semeval; do
    evaluate $1 $dataset
    compute_score $dataset
done
