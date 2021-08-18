#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR=${CURRENT_DIR}/${DATASET}
mkdir -p $SAVE_DIR

#FAIRSEQ_PATH=$(python -c "import fairseq; print(fairseq.__path__[0])")
#FAIRSEQ_PATH="$(dirname "$FAIRSEQ_PATH")" # parent directory
#echo "Fairseq path: $FAIRSEQ_PATH"
#export PYTHONPATH=${FAIRSEQ_PATH}:"${PYTHONPATH}"


function train () {

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=1000
UPDATE_FREQ=4
MAX_TOKENS=4096
PRETRAINED_MODEL=${HOME_DIR}/models/mass-base-uncased/mass-base-uncased.pt
DATA_DIR=${DATA_DIR_PREFIX}/${DATASET}/fairseq/bert_bpe/binary

fairseq-train $DATA_DIR \
    --user-dir source \
    --task translation_mass \
    --arch transformer_mass_base \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --max-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --update-freq $UPDATE_FREQ \
    --max-tokens $MAX_TOKENS \
    --fp16 \
    --ddp-backend=no_c10d \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model $PRETRAINED_MODEL \
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
    --user-dir source \
    --task translation_mass \
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
