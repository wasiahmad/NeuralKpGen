#!/usr/bin/env bash


function download_data () {

declare -A FILEIDS
FILEIDS["train"]=1LGR62JPHL2-zesX5lT53KqfzHB78g0d_
FILEIDS["valid"]=1Fq3oZR99OTYKKe88-5ocwbdMT_CMdcEb
FILEIDS["test"]=1F-HDwjI23f6nvtFiea-CGIO_IaKi_2fK

URL_PREFIX=https://drive.google.com/uc?export=download
downloaded=false

for split in train valid test
do
    FILE=KPTimes.${split}.jsonl
    if [[ ! -f "$FILE" ]]; then
        downloaded=true
        FILEID=${FILEIDS[${split}]}
        curl -c ./cookie -s -L "${URL_PREFIX}&id=${FILEID}" > /dev/null
        curl -Lb ./cookie "${URL_PREFIX}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILE}.gz
        gunzip ${FILE}.gz
        rm ./cookie
    fi
done

if [[ $downloaded == true ]]; then
    echo "Aggregating statistics of the raw KPTimes dataset"
    python ../data_stat.py \
        -choice 'raw' \
        -train_file 'KPTimes.train.jsonl' \
        -valid_file 'KPTimes.valid.jsonl' \
        -test_file 'KPTimes.test.jsonl'
fi

}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing KPTimes dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset KPTimes \
        -data_dir . \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60
    echo "Aggregating statistics of the processed KPTimes dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json \
        -test_file $OUTDIR/test.json
fi

}

download_data
prepare BertTokenizer processed
prepare SpacyTokenizer spacy_processed


