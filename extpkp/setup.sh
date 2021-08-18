#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;
DATA_DIR=${HOME_DIR}/data/scikp

if [[ ! -d $DATA_DIR/kp20k/bioformat ]]; then
    python -W ignore $HOME_DIR/data/bioConverter.py \
        -dataset KP20k \
        -data_dir $DATA_DIR/kp20k/processed \
        -out_dir $DATA_DIR/kp20k/bioformat \
        -max_src_len 510 \
        -workers 60;
fi
for dataset in inspec nus krapivin semeval; do
    if [[ ! -d $DATA_DIR/$dataset/bioformat ]]; then
        python -W ignore $HOME_DIR/data/bioConverter.py \
            -dataset $dataset \
            -data_dir $DATA_DIR/$dataset/processed \
            -out_dir $DATA_DIR/$dataset/bioformat \
            -max_src_len 510 \
            -workers 60;
    fi
done
