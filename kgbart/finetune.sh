#!/usr/bin/env bash

COMMON_PATH=wasiahmad/workspace/projects/NeuralKpGen
DATA_DIR=/home/${COMMON_PATH}/bart/data
PRETRAINED=/local/${COMMON_PATH}/kgbart/models
SAVE_DIR=/local/${COMMON_PATH}/kgbart/finetuning
CKPT_DIR=${SAVE_DIR}/checkpoints
mkdir -p $CKPT_DIR

export CUDA_VISIBLE_DEVICES=$1


function train () {

DATASET=kp20k

BATCH_SIZE=8
TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=1000
UPDATE_FREQ=1
LR=3e-05

fairseq-train ${DATA_DIR}/${DATASET}-bin \
--fp16 --arch bart_base \
--restore-file ${PRETRAINED}/checkpoint_best.pt \
--reset-optimizer --reset-dataloader --reset-meters \
--batch-size $BATCH_SIZE \
--max-update $TOTAL_NUM_UPDATES \
--warmup-updates $WARMUP_UPDATES \
--update-freq $UPDATE_FREQ \
--task translation \
--truncate-source \
--max-source-positions 1024 --max-target-positions 1024 \
--source-lang source --target-lang target \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--skip-invalid-size-inputs-valid-test \
--log-format json --log-interval 100 \
--num-workers 4 --seed 1234 \
--find-unused-parameters --ddp-backend=no_c10d \
--save-dir $CKPT_DIR 2>&1 | tee $CKPT_DIR/output.log;

}

function decode () {

DATASET=$1
python decode.py \
--data_name_or_path $DATA_DIR/${DATASET}-bin \
--data_dir $DATA_DIR/${DATASET} \
--checkpoint_dir $CKPT_DIR \
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

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo
        exit;;
   esac
done


train
for dataset in kp20k inspec krapivin nus semeval; do
    decode $dataset && evaluate $dataset
done
