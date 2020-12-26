#!/usr/bin/env bash

COMMON_PATH=wasiahmad/workspace/projects/NeuralKpGen
SAVE_DIR=/local/${COMMON_PATH}/kgbart/eval
DATA_DIR=/home/${COMMON_PATH}/bart/data
MODEL_DIR=/local/${COMMON_PATH}/kgbart/models
mkdir -p $SAVE_DIR

export CUDA_VISIBLE_DEVICES=$1


function decode () {

DATASET=$1
python decode.py \
--data_name_or_path $DATA_DIR/${DATASET}-bin \
--data_dir $DATA_DIR/${DATASET} \
--checkpoint_dir $MODEL_DIR \
--checkpoint_file checkpoint_best.pt \
--output_file $SAVE_DIR/${DATASET}_test.hypo \
--batch_size 64 \
--beam 1 \
--min_len 1 \
--lenpen 1.0 \
--no_repeat_ngram_size 3 \
--max_len_b 60;

}

function evaluate () {

DATASET=$1
python -W ignore kp_eval.py \
--src_dir $DATA_DIR/${DATASET} \
--pred_file $SAVE_DIR/${DATASET}_test.hypo \
--tgt_dir $SAVE_DIR \
--log_file ${DATASET}_test \
--k_list 5 M;

}


for dataset in kp20k inspec krapivin nus semeval; do
    decode $dataset && evaluate $dataset
done
