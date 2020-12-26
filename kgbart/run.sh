#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/oagk
BART_PATH=/local/wasiahmad/workspace/projects/NeuralKpGen/kgbart/bart.base/model.pt
SAVE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/kgbart/models
mkdir -p $SAVE_DIR

DATA_DIR=""
NUM_SPLIT=11
for (( idx=0; idx<=10; idx++ )); do
    DATA_DIR+="$SRCDIR/shards/shard${idx}"
    if [[ $idx -lt 10 ]]; then
        DATA_DIR+=":"
    fi
done

export CUDA_VISIBLE_DEVICES=$1

TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=10000
LR=3e-05
BATCH_SIZE=8
UPDATE_FREQ=4
ARCH=bart_base


if [[ -f $SAVE_DIR/checkpoint_last.pt ]]; then
    RESTORE=''
else
    RESTORE='--restore-file ${BART_PATH} --reset-optimizer --reset-dataloader --reset-meters'
fi


fairseq-train $DATA_DIR \
--fp16 $RESTORE \
--truncate-source \
--source-lang source \
--target-lang target \
--arch $ARCH \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--task translation \
--max-source-positions 1024 \
--max-target-positions 1024 \
--batch-size $BATCH_SIZE \
--max-update $TOTAL_NUM_UPDATES \
--warmup-updates $WARMUP_UPDATES \
--update-freq $UPDATE_FREQ \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam \
--adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr $LR --lr-scheduler polynomial_decay \
--skip-invalid-size-inputs-valid-test \
--no-epoch-checkpoints \
--save-interval-updates 5000 \
--keep-interval-updates 5 \
--log-format json --log-interval 100 \
--num-workers 4 --seed 1234 \
--find-unused-parameters --ddp-backend=no_c10d \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/output.log;

