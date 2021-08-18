#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

DATA_DIR=${HOME_DIR}/data
MODEL_DIR=${HOME_DIR}/models
mkdir -p $MODEL_DIR

DL_URL_PREFIX=https://msraprophetnet.blob.core.windows.net/prophetnet

if [[ ! -f $MODEL_DIR/prophetnet_en.pt ]]; then
    wget -N ${DL_URL_PREFIX}/release_checkpoints/prophetnet_en.pt -P $MODEL_DIR
fi

DICT_FILE=${CURRENT_DIR}/vocab.txt


function preprocess () {

if [[ "$TASK" =~ ^(kp20k|nus|inspec|krapivin|semeval)$ ]]; then
    IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/bert_bpe
else
    IN_DIR=$DATA_DIR/kptimes/fairseq
    OUT_DIR=$DATA_DIR/kptimes/fairseq/bert_bpe
fi

if [[ -d $OUT_DIR ]]; then return; fi
mkdir -p $OUT_DIR

for SPLIT in train valid test; do
    for LANG in source target; do
        python ${HOME_DIR}/utils/encode.py \
            --model prophetnet \
            --inputs $IN_DIR/$SPLIT.$LANG \
            --outputs $OUT_DIR/$SPLIT.bpe.$LANG \
            --max_len 510 \
            --workers 60; \
    done
done

}


function process () {

if [[ "$TASK" =~ ^(kp20k|nus|inspec|krapivin|semeval)$ ]]; then
    OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/bert_bpe
else
    OUT_DIR=$DATA_DIR/kptimes/fairseq/bert_bpe
fi

if [[ -d $OUT_DIR/binary ]]; then return; fi

fairseq-preprocess \
    --user-dir source \
    --task translation_prophetnet \
    --source-lang source \
    --target-lang target \
    --trainpref $OUT_DIR/train.bpe \
    --validpref $OUT_DIR/valid.bpe \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE;

}


function preprocess_test_only () {

IN_DIR=$DATA_DIR/scikp/$TASK/fairseq
OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/bert_bpe
if [[ -d $OUT_DIR ]]; then return; fi

for SPLIT in test; do
    for LANG in source target; do
        python ${HOME_DIR}/utils/encode.py \
            --model prophetnet \
            --inputs $IN_DIR/$SPLIT.$LANG \
            --outputs $OUT_DIR/$SPLIT.bpe.$LANG \
            --max_len 510 \
            --workers 60;
    done
done

}


function process_test_only () {

OUT_DIR=$DATA_DIR/scikp/$TASK/fairseq/bert_bpe
if [[ -d $OUT_DIR/binary ]]; then return; fi

fairseq-preprocess \
    --user-dir source \
    --task translation_prophetnet \
    --source-lang source \
    --target-lang target \
    --testpref $OUT_DIR/test.bpe \
    --destdir $OUT_DIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE;

}

for task in kp20k kptimes; do
    TASK=$task
    preprocess
    process
done
for task in nus inspec krapivin semeval; do
    TASK=$task
    preprocess_test_only
    process_test_only
done
