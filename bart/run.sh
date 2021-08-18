#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR=${CURRENT_DIR}/${DATASET}
mkdir -p $SAVE_DIR


function train () {

TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=1000
LR=3e-05
MAX_TOKENS=4096
UPDATE_FREQ=4
ARCH=bart_base # bart_large
BART_PATH=${HOME_DIR}/models/bart.base/model.pt # bart.large/model.pt
DATA_DIR=${DATA_DIR_PREFIX}/${DATASET}/fairseq/gpt2_bpe/binary

fairseq-train $DATA_DIR \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --truncate-source \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --source-lang source \
    --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --arch $ARCH \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --max-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --save-dir $SAVE_DIR \
    --log-format json \
    2>&1 | tee $SAVE_DIR/finetune.log;

}


function decode () {

EVAL_DATASET=$1
DATA_DIR=${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq/gpt2_bpe/binary
OUT_FILE=$SAVE_DIR/${EVAL_DATASET}_out.txt
HYP_FILE=$SAVE_DIR/${EVAL_DATASET}_hypotheses.txt

fairseq-generate $DATA_DIR \
    --path $SAVE_DIR/checkpoint_best.pt \
    --task translation \
    --batch-size 64 \
    --beam 1 \
    --no-repeat-ngram-size 0 \
    --max-len-b 60 \
    2>&1 | tee $OUT_FILE;

grep ^H $OUT_FILE | sort -V | cut -f3- > $HYP_FILE;

}


function evaluate () {

EVAL_DATASET=$1

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_dir ${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq \
    --file_prefix $SAVE_DIR/${EVAL_DATASET} \
    --tgt_dir $SAVE_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 M;

}


while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID DATASET_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
        echo
        exit;;
   esac
done


if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    train
    for dataset in kp20k inspec krapivin nus semeval; do
        decode $dataset
        evaluate $dataset
    done
elif [[ $DATASET == 'kptimes' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data
    train
    decode $DATASET
    evaluate $DATASET
fi
