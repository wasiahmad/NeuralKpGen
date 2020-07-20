#!/usr/bin/env bash

SRCDIR=data

if [[ ! -d mass-base-uncased ]]; then
    wget -c https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz
    mkdir -p mass-base-uncased
    tar -xvf mass-base-uncased.tar.gz -C mass-base-uncased
    rm mass-base-uncased.tar.gz
fi
if [[ ! -d mass-middle-uncased ]]; then
    wget -c https://modelrelease.blob.core.windows.net/mass/mass-middle-uncased.tar.gz
    mkdir -p mass-middle-uncased
    tar -xvf mass-middle-uncased.tar.gz -C mass-middle-uncased
    rm mass-middle-uncased.tar.gz
fi

if [[ ! -d $SRCDIR ]]; then
    python agg_data.py
fi

DICT_FILE=mass-base-uncased/dict.txt

function preprocess () {

mkdir -p "$SRCDIR/tokenized/$TASK"
for SPLIT in train valid test; do
    for LANG in source target; do
        python encode.py \
            --inputs "$SRCDIR/raw/$TASK/$SPLIT.$LANG" \
            --outputs "$SRCDIR/tokenized/$TASK/$SPLIT.$LANG" \
            --max_len 510 \
            --workers 60; \
    done
done

}

function process () {

fairseq-preprocess \
--user-dir mass --task masked_s2s \
--source-lang "source" \
--target-lang "target" \
--trainpref "$SRCDIR/tokenized/$TASK/train" \
--validpref "$SRCDIR/tokenized/$TASK/valid" \
--testpref "$SRCDIR/tokenized/$TASK/test" \
--destdir "$SRCDIR/${TASK}-bin/" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

function preprocess_test_only () {

mkdir -p "$SRCDIR/tokenized/$TASK"
for SPLIT in test; do
    for LANG in source target
    do
        python encode.py \
            --inputs "$SRCDIR/raw/$TASK/$SPLIT.$LANG" \
            --outputs "$SRCDIR/tokenized/$TASK/$SPLIT.$LANG" \
            --max_len 510 \
            --workers 60; \
    done
done

}

function process_test_only () {

fairseq-preprocess \
--user-dir mass --task masked_s2s \
--source-lang "source" \
--target-lang "target" \
--testpref "$SRCDIR/tokenized/$TASK/test" \
--destdir "$SRCDIR/${TASK}-bin/" \
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


