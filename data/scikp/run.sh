#!/usr/bin/env bash


function download_data () {

FILE=kp_datasets.zip
if [[ ! -f $FILE ]]; then
    # https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp
    fileid="1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp"
    baseurl="https://drive.google.com/uc?export=download"
    curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
    curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    unzip ${FILE} && rm ./cookie
    rm -rf kp20k_sorted && rm -rf cross_domain_sorted
fi

}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing KP20k dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset KP20k \
        -data_dir kp20k_separated \
        -out_dir ${OUTDIR}/kp20k \
        -tokenizer $1 \
        -workers 60
    echo "Aggregating statistics of the processed KP20k dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file ${OUTDIR}/kp20k/train.json \
        -valid_file ${OUTDIR}/kp20k/valid.json \
        -test_file ${OUTDIR}/kp20k/test.json
    echo "============Processing Cross-Domain datasets============"
    for dataset in inspec nus krapivin semeval; do
        PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
            -dataset $dataset \
            -data_dir cross_domain_separated \
            -out_dir ${OUTDIR}/${dataset} \
            -tokenizer $1 \
            -workers 60
    done
fi

}

download_data
prepare BertTokenizer processed
prepare WhiteSpace basic_processed

