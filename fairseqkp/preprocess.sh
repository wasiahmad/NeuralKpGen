#!/usr/bin/env bash

SRCDIR=data

if [[ ! -d $SRCDIR ]]; then
    python agg_data.py
fi

function process () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "$SRCDIR/${TASK}/train" \
--validpref "$SRCDIR/${TASK}/valid" \
--testpref "$SRCDIR/${TASK}/test" \
--destdir "$SRCDIR/${TASK}-bin/" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

function process_test_only () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--testpref "$SRCDIR/${TASK}/test" \
--destdir "$SRCDIR/${TASK}-bin/" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

for task in kp20k kptimes; do
    TASK=$task
    DICT_FILE=data/${task}/dict.txt
    process
done
for task in nus inspec krapivin semeval; do
    TASK=$task
    DICT_FILE=data/kp20k/dict.txt
    process_test_only
done

