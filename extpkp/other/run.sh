#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/extpkp
mkdir -p $BASE_DIR

export BERT_MODEL=bert-base-uncased
#export BERT_MODEL=roberta-base
#export BERT_MODEL=microsoft/unilm-base-cased
#export BERT_MODEL=microsoft/MiniLM-L12-H384-uncased
#export BERT_MODEL=allenai/scibert_scivocab_uncased

export MAX_LENGTH=464
export BATCH_SIZE=8
export GRADIENT_ACCUM_STEPS=1
export MAX_STEPS=20000
export SAVE_STEPS=2000
export SEED=1


function train () {

export CUDA_VISIBLE_DEVICES=$1
export OUTPUT_DIR=${BASE_DIR}/kp20k-${BERT_MODEL}

python run_ner.py \
--fp16 \
--data_dir ${BASE_DIR}/data/processed/kp20k \
--dataset_name kp20k \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--max_steps $MAX_STEPS \
--per_device_train_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUM_STEPS \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--workers 60 2>&1 | tee ${BASE_DIR}/kp20k-${BERT_MODEL}-train.log

}


function evaluate () {

export CUDA_VISIBLE_DEVICES=$1
export OUTPUT_DIR=${BASE_DIR}/kp20k-${BERT_MODEL}

python run_ner.py \
--fp16 \
--data_dir ${BASE_DIR}/data/processed/$2 \
--dataset_name $2 \
--model_name_or_path ${OUTPUT_DIR}/checkpoint-${MAX_STEPS} \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_predict \
--workers 60 2>&1 | tee ${BASE_DIR}/$2-${BERT_MODEL}-eval.log

}


function compute_score () {

export OUTPUT_DIR=${BASE_DIR}/kp20k-${BERT_MODEL}

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
