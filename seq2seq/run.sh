#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

export CUDA_VISIBLE_DEVICES=$1
export DATASET=$2
export MODEL_TYPE=$3

SAVE_DIR=${CURRENT_DIR}/${MODEL_TYPE}_${DATASET}
mkdir -p $SAVE_DIR

NUM_GPUS=`echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c`;
NUM_GPUS=`expr ${NUM_GPUS} + 1`;


function rnn_train () {

declare -A BATCH_SIZE
BATCH_SIZE['kp20k']=16
BATCH_SIZE['kptimes']=32
declare -A NUM_UPDATES
NUM_UPDATES['kp20k']=50000
NUM_UPDATES['kptimes']=50000

DATA_DIR=${DATA_DIR_PREFIX}/${DATASET}/fairseq/bert_bpe/binary

fairseq-train $DATA_DIR \
    --fp16 \
    --num-workers 4 \
    --save-dir $SAVE_DIR \
    --skip-invalid-size-inputs-valid-test \
    --arch lstm \
    --task translation \
    --batch-size ${BATCH_SIZE[${DATASET}]} \
    --truncate-source \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --encoder-embed-dim 512 \
    --decoder-embed-dim 512 \
    --source-lang source \
    --target-lang target \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --encoder-bidirectional \
    --encoder-hidden-size 512 \
    --decoder-hidden-size 512 \
    --decoder-attention 1 \
    --dropout 0.2 \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 1.0 \
    --lr 1e-03 \
    --max-epoch 25 \
    --max-update ${NUM_UPDATES[${DATASET}]} \
    --update-freq 1 \
    --validate-interval 1 \
    --patience 5 \
    --no-epoch-checkpoints \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --log-format json \
    2>&1 | tee $SAVE_DIR/train.log;

}


function transformer_train () {

if [[ $NUM_GPUS < 4 ]]; then
    echo "Warning: Use of fours GPUs is recommended"
fi

NUM_LAYERS=6
ATTN_HEADS=12
MODEL_SIZE=768

# EFFECTIVE_BATCH_SIZE = batch_size * update_freq * num_gpus
# kp20k: 510k, kptimes: 260k
# We train models for ~25 epochs

if [[ $DATASET == 'kp20k' ]]; then
    BATCH_SIZE=16
    EFFECTIVE_BATCH_SIZE=64
    TOTAL_NUM_UPDATES=100000
elif [[ $DATASET == 'kptimes' ]]; then
    BATCH_SIZE=16
    EFFECTIVE_BATCH_SIZE=64
    TOTAL_NUM_UPDATES=100000
fi

BSZ_PER_GPU=$((EFFECTIVE_BATCH_SIZE/NUM_GPUS))
UPDATE_FERQ=$((BSZ_PER_GPU/BATCH_SIZE))
WARMUP_UPDATES=$((TOTAL_NUM_UPDATES/20)) # (1/20) * of total_num_updates
DATA_DIR=${DATA_DIR_PREFIX}/${DATASET}/fairseq/bert_bpe/binary

fairseq-train $DATA_DIR \
    --fp16 \
    --num-workers 4 \
    --save-dir $SAVE_DIR \
    --skip-invalid-size-inputs-valid-test \
    --arch transformer \
    --task translation \
    --batch-size $BATCH_SIZE \
    --truncate-source \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --encoder-embed-dim $MODEL_SIZE \
    --decoder-embed-dim $MODEL_SIZE \
    --encoder-learned-pos \
    --decoder-learned-pos \
    --layernorm-embedding \
    --no-scale-embedding \
    --source-lang source \
    --target-lang target \
    --encoder-layers $NUM_LAYERS \
    --decoder-layers $NUM_LAYERS \
    --encoder-attention-heads $ATTN_HEADS \
    --decoder-attention-heads $ATTN_HEADS \
    --attention-dropout 0.2 \
    --activation-dropout 0.2 --dropout 0.2 \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 1.0 \
    --lr-scheduler polynomial_decay \
    --lr 1e-04 \
    --max-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --max-epoch 25 \
    --update-freq $UPDATE_FERQ \
    --validate-interval 1 \
    --patience 5 \
    --no-epoch-checkpoints \
    --find-unused-parameters \
    --ddp-backend=no_c10d \
    --reset-dataloader \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-optimizer \
    --log-format json \
    2>&1 | tee $SAVE_DIR/train.log;

}


function decode () {

EVAL_DATASET=$1
DATA_DIR=${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq/gpt2_bpe/binary
OUT_FILE=$SAVE_DIR/${EVAL_DATASET}_out.txt
HYP_FILE=$SAVE_DIR/${EVAL_DATASET}_hypotheses.txt

fairseq-generate $DATA_DIR \
    --path ${SAVE_DIR}/checkpoint_best.pt \
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
        echo "Syntax: run.sh GPU_ID DATASET_NAME MODEL_TYPE"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. Choices: kp20k, kptimes."
        echo "MODEL_TYPE     Model type. Choices: rnn, transformer."
        echo
        exit;;
   esac
done

if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    ${MODEL_TYPE}_train
    for dataset in kp20k inspec krapivin nus semeval; do
        decode $dataset
        evaluate $dataset
    done
elif [[ $DATASET == 'kptimes' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data
    ${MODEL_TYPE}_train
    decode $DATASET
    evaluate $DATASET
fi
