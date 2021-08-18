#!/usr/bin/env bash

function download_data () {

URL_PREFIX=https://msmarco.blob.core.windows.net/openkp
downloaded=false

for split in Train Dev EvalPublic
do
    FILE=OpenKP${split}.jsonl
    if [[ ! -f "$FILE" ]]; then
        downloaded=true
        wget ${URL_PREFIX}/${FILE}
    fi
done

if [[ $downloaded == true ]]; then
    echo "Aggregating statistics of the raw openkp dataset"
    python ../data_stat.py \
        -choice 'raw' \
        -train_file 'OpenKPTrain.jsonl' \
        -valid_file 'OpenKPDev.jsonl' \
        -test_file 'OpenKPEvalPublic.jsonl';
fi

}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing OpenKP dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset OpenKP \
        -data_dir . \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60;
    echo "Aggregating statistics of the processed KPTimes dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json \
        -test_file $OUTDIR/test.json;
fi

}

download_data
prepare WhiteSpace processed
#prepare BertTokenizer processed
