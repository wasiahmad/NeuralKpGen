#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs

function rnn_train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
TOTAL_NUM_UPDATES=50000
ARCH=lstm

# https://github.com/pytorch/fairseq/blob/ffecb4e3496379edf5ecae1483df5b7e0886c264/fairseq/models/transformer.py#L902
fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--fp16 --num-workers 4 --save-dir ${DATASET}_checkpoints \
--skip-invalid-size-inputs-valid-test \
--arch $ARCH --task translation \
--max-tokens 8192 --truncate-source \
--max-source-positions 512 --max-target-positions 512 \
--encoder-embed-dim 512 --decoder-embed-dim 512 \
--source-lang source --target-lang target \
--encoder-layers 2 --decoder-layers 2 --encoder-bidirectional \
--encoder-hidden-size 512 --decoder-hidden-size 512 \
--decoder-attention 1 --dropout 0.2 \
--share-all-embeddings --share-decoder-input-output-embed \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 --lr-scheduler polynomial_decay --lr 1e-04 \
--max-update $TOTAL_NUM_UPDATES --max-epoch 25 --update-freq 1 \
--validate-interval 1 --patience 3 --no-epoch-checkpoints \
--find-unused-parameters --ddp-backend=no_c10d \
--log-format=json 2>&1 | tee logs/rnn.log

}

function transformer_train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
TOTAL_NUM_UPDATES=50000
WARMUP_UPDATES=1000
ARCH=transformer

# https://github.com/pytorch/fairseq/blob/ffecb4e3496379edf5ecae1483df5b7e0886c264/fairseq/models/transformer.py#L902
fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--fp16 --num-workers 4 --save-dir ${DATASET}_checkpoints \
--skip-invalid-size-inputs-valid-test \
--arch $ARCH --task translation \
--max-tokens 8192 --truncate-source \
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
--clip-norm 0.1 --lr-scheduler polynomial_decay --lr 1e-04 \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--max-epoch 25 --update-freq 2 \
--validate-interval 1 --patience 3 --no-epoch-checkpoints \
--find-unused-parameters --ddp-backend=no_c10d \
--log-format=json 2>&1 | tee logs/transformer.log

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR_PREFIX=$3
OUTPUT_FILE=$4

python decode.py \
--data_name_or_path "$SRCDIR/${DATASET}-bin/" \
--data_dir "$SRCDIR/${DATASET}/" \
--checkpoint_dir ${SAVE_DIR_PREFIX}_checkpoints \
--checkpoint_file checkpoint_best.pt \
--output_file $OUTPUT_FILE \
--batch_size 64 \
--beam 1 \
--min_len 1 \
--lenpen 1.0 \
--no_repeat_ngram_size 3 \
--max_len_b 60;

}

function evaluate () {

python -W ignore kp_eval.py \
--src_dir $1 \
--pred_file $2 \
--tgt_dir . \
--log_file $3 \
--k_list 5 M;

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
        decode "$1" $dataset $2 logs/${3}_${dataset}_test.hypo
        evaluate ${SRCDIR}/${dataset} logs/${3}_${dataset}_test.hypo ${3}_${dataset}
    done
elif [[ $2 == 'kptimes' ]]; then
    $3_train "$1" $2
    decode "$1" $2 $2 logs/${3}_${2}_test.hypo
    evaluate ${SRCDIR}/${2} logs/${3}_${2}_test.hypo ${3}_${2}
fi
