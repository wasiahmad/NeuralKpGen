#!/usr/bin/env bash

DATA_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/cord19
DATE=2020-12-12
T_DIR=$DATA_DIR/${DATE}


function download () {

BASE_URL=https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases
if [[ ! -d $T_DIR ]]; then
    FILE=cord-19_${DATE}.tar.gz
    curl -o $T_DIR/${FILE} ${BASE_URL}/${FILE}
    tar -xvzf $T_DIR/${FILE} -C $T_DIR
    rm $T_DIR/${FILE}
fi

}

python extract.py \
--csv_file $T_DIR/metadata.csv \
--out_dir $T_DIR;
