#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/extpkp/data
mkdir -p $BASE_DIR
REPO=$PWD

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

OUTDIR=$1/processed

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing KP20k dataset============"
    python -W ignore bioConverter.py \
        -dataset KP20k \
        -data_dir $1/kp20k_separated \
        -out_dir ${OUTDIR}/kp20k \
        -tokenizer WhiteSpace \
        -max_src_len 510 \
        -workers 60;
    echo "============Processing Cross-Domain datasets============"
    for dataset in inspec nus krapivin semeval; do
        python -W ignore bioConverter.py \
            -dataset $dataset \
            -data_dir $1/cross_domain_separated \
            -out_dir ${OUTDIR}/${dataset} \
            -tokenizer WhiteSpace \
            -max_src_len 510 \
            -workers 60;
    done
fi

}

cd $BASE_DIR
download_data
cd $REPO
prepare $BASE_DIR
