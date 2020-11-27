#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/extpkp/data
mkdir -p $BASE_DIR
REPO=$PWD

function download_data () {

# https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp
fileid="1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp"
baseurl="https://drive.google.com/uc?export=download"
curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o $1
rm ./cookie

}

function prepare () {

FILE=kp_datasets.zip
if [[ ! -f ${BASE_DIR}/${FILE} ]]; then
    cd $BASE_DIR
    download_data $FILE
    cd $REPO
fi
unzip ${BASE_DIR}/${FILE} -d ${BASE_DIR}

OUTDIR=${BASE_DIR}/processed
if [[ ! -d $OUTDIR ]]; then
    echo "============Processing KP20k dataset============"
    python -W ignore bioConverter.py \
        -dataset KP20k \
        -data_dir ${BASE_DIR}/kp20k_separated \
        -out_dir ${OUTDIR}/kp20k \
        -tokenizer WhiteSpace \
        -max_src_len 510 \
        -workers 60;
    echo "============Processing Cross-Domain datasets============"
    for dataset in inspec nus krapivin semeval; do
        python -W ignore bioConverter.py \
            -dataset $dataset \
            -data_dir ${BASE_DIR}/cross_domain_separated \
            -out_dir ${OUTDIR}/${dataset} \
            -tokenizer WhiteSpace \
            -max_src_len 510 \
            -workers 60;
    done
fi

rm -rf ${BASE_DIR}/kp20k_sorted && rm -rf ${BASE_DIR}/kp20k_separated
rm -rf ${BASE_DIR}/cross_domain_sorted && rm -rf ${BASE_DIR}/cross_domain_separated

}

prepare
