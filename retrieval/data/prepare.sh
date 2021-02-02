#!/usr/bin/env bash

#DATA_DIR=../../data/scikp/kp20k_separated
#
#for split in train valid test; do
#    python convert.py \
#        -src_file $DATA_DIR/${split}_src.txt \
#        -tgt_file $DATA_DIR/${split}_trg.txt \
#        -out_file $DATA_DIR/KP20k.${split}.jsonl;
#done

DATA_DIR=../../data/scikp/cross_domain_separated

for dataset in inspec nus krapivin semeval; do
    python convert.py \
        -src_file $DATA_DIR/word_${dataset}_testing_context.txt \
        -tgt_file $DATA_DIR/word_${dataset}_testing_allkeywords.txt \
        -out_file $DATA_DIR/${dataset}.test.jsonl;
done
