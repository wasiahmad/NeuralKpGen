#!/usr/bin/env bash

SRCDIR=data

if [[ ! -d $SRCDIR ]]; then
    python agg_data.py
fi

function preprocess () {

for SPLIT in train valid test; do
  for LANG in source target; do
    python encode.py \
        --inputs "$SRCDIR/$TASK/$SPLIT.$LANG" \
        --outputs "$SRCDIR/$TASK/$SPLIT.enc.$LANG" \
        --max_len 510 \
        --workers 60;
  done
done

}

function process () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "$SRCDIR/${TASK}/train.enc" \
--validpref "$SRCDIR/${TASK}/valid.enc" \
--testpref "$SRCDIR/${TASK}/test.enc" \
--destdir "$SRCDIR/${TASK}-bin/" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

function preprocess_test_only () {

for SPLIT in test; do
  for LANG in source target; do
    python encode.py \
        --inputs "$SRCDIR/$TASK/$SPLIT.$LANG" \
        --outputs "$SRCDIR/$TASK/$SPLIT.enc.$LANG" \
        --max_len 510 \
        --workers 60;
  done
done

}

function process_test_only () {

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--testpref "$SRCDIR/${TASK}/test.enc" \
--destdir "$SRCDIR/${TASK}-bin/" \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

# aggregate data first
python agg_data.py

for task in kp20k oagk kptimes; do
    TASK=$task
    DICT_FILE=data/${task}/dict.txt
    preprocess
    process
done
for task in nus inspec krapivin semeval; do
    TASK=$task
    DICT_FILE=data/kp20k/dict.txt
    preprocess_test_only
    process_test_only
done
