#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs

function rnn_train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR=rnn_${DATASET}_checkpoints
LOG_FILE=logs/rnn_${DATASET}.log
TOTAL_NUM_UPDATES=50000

fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--fp16 --num-workers 4 --save-dir $SAVE_DIR \
--skip-invalid-size-inputs-valid-test \
--arch lstm --task translation \
--max-tokens 16384 --truncate-source \
--max-source-positions 512 --max-target-positions 512 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--source-lang source --target-lang target \
--encoder-layers 3 --decoder-layers 3 --encoder-bidirectional \
--encoder-hidden-size 512 --decoder-hidden-size 512 \
--decoder-attention 1 --dropout 0.2 \
--share-all-embeddings --share-decoder-input-output-embed \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 1.0 --lr 1e-03 \
--max-update $TOTAL_NUM_UPDATES --max-epoch 25 --update-freq 1 \
--validate-interval 1 --patience 5 --no-epoch-checkpoints \
--find-unused-parameters --ddp-backend=no_c10d \
--log-format=json 2>&1 | tee $LOG_FILE

}

function transformer_train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR=transformer_${DATASET}_checkpoints
LOG_FILE=logs/transformer_${DATASET}.log
TOTAL_NUM_UPDATES=50000
WARMUP_UPDATES=1000

if [[ $DATASET == 'kp20k' ]]; then
    MAX_TOKENS=6144
elif [[ $DATASET == 'kptimes' ]]; then
    MAX_TOKENS=8192
fi

fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--fp16 --num-workers 4 --save-dir $SAVE_DIR \
--skip-invalid-size-inputs-valid-test \
--arch transformer --task translation \
--max-tokens $MAX_TOKENS --truncate-source \
--max-source-positions 512 --max-target-positions 512 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--source-lang source --target-lang target \
--encoder-layers 6 --decoder-layers 6 \
--encoder-attention-heads 8 --decoder-attention-heads 8 \
--attention-dropout 0.2 --activation-dropout 0.2 --dropout 0.2 \
--share-all-embeddings --share-decoder-input-output-embed \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 1.0 --lr-scheduler polynomial_decay --lr 1e-04 \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--max-epoch 25 --update-freq 2 \
--validate-interval 1 --patience 5 --no-epoch-checkpoints \
--find-unused-parameters --ddp-backend=no_c10d \
--log-format=json 2>&1 | tee $LOG_FILE

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
MODEL_DIR_PREFIX=$3
LOG_FILE=$4
MODEL=${MODEL_DIR_PREFIX}/checkpoint_best.pt

fairseq-generate $SRCDIR/${DATASET}-bin/ \
--path $MODEL \
--task translation \
--log-format simple \
--batch-size 64 \
--beam 1 \
--min-len 1 \
--no-repeat-ngram-size 3 \
--max-len-b 60 \
|& tee $LOG_FILE

}

function evaluate () {

python -W ignore kp_eval.py \
--src_dir $1 \
--file_prefix $2 \
--tgt_dir . \
--log_file $3 \
--k_list 5 M

}

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID DATASET_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. Choices: kp20k, kptimes."
        echo "MODEL_TYPE     Model type. Choices: rnn, transformer."
        echo
        exit;;
   esac
done

if [[ $2 == 'kp20k' ]]; then
    $3_train "$1" $2
    for dataset in kp20k inspec krapivin nus semeval; do
        decode "$1" $dataset ${3}_${2}_checkpoints logs/${3}_${dataset}_test.txt
        grep ^S logs/${3}_${2}_test.txt | cut -f1 > "logs/${3}_${2}_source.txt"
        grep ^T logs/${3}_${2}_test.txt | cut -f2- > "logs/${3}_${2}_target.txt"
        grep ^H logs/${3}_${2}_test.txt | cut -f3- > "logs/${3}_${2}_hypotheses.txt"
        evaluate ${SRCDIR}/${dataset} logs/${3}_${2} ${3}_${dataset}
    done
elif [[ $2 == 'kptimes' ]]; then
    $3_train "$1" $2
    decode "$1" $2 ${3}_${2}_checkpoints logs/${3}_${2}_test.txt
    grep ^S logs/${3}_${2}_test.txt | cut -f1 > "logs/${3}_${2}_source.txt"
    grep ^T logs/${3}_${2}_test.txt | cut -f2- > "logs/${3}_${2}_target.txt"
    grep ^H logs/${3}_${2}_test.txt | cut -f3- > "logs/${3}_${2}_hypotheses.txt"
    evaluate ${SRCDIR}/${2} logs/${3}_${2} ${3}_${2}
fi
