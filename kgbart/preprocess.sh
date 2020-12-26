#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/oagk
SHARD_DIR=${SRCDIR}/shards

if [[ ! -d bart.base ]]; then
    wget -N https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
    tar -xvzf bart.base.tar.gz
    rm bart.base.tar.gz
fi
for filename in "encoder.json" "vocab.bpe"; do
    if [[ ! -f $filename ]]; then
        wget -N https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/${filename}
    fi
done

DICT_FILE=bart.base/dict.txt # dict.txt


function bpe_preprocess () {

for LANG in source target; do
    for i in $(seq 0 11); do
        python encode.py \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$SRCDIR/oagkx_processed/split.$i.$LANG" \
        --outputs "$SRCDIR/oagkx_processed/split.$i.bpe.$LANG" \
        --max_len 510 \
        --workers 60;
    done
done

}


function binarize () {

mkdir -p $SHARD_DIR/shard0
fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref $SRCDIR/oagkx_processed/split.0.bpe \
    --validpref $SRCDIR/oagkx_processed/split.11.bpe \
    --destdir $SHARD_DIR/shard0 \
    --srcdict $DICT_FILE \
    --tgtdict $DICT_FILE \
    --workers 60;

for i in $(seq 1 10); do
    mkdir -p $SHARD_DIR/shard${i}
    fairseq-preprocess \
        --source-lang "source" \
        --target-lang "target" \
        --trainpref $SRCDIR/oagkx_processed/split.$i.bpe \
        --destdir $SHARD_DIR/shard${i} \
        --srcdict $DICT_FILE \
        --tgtdict $DICT_FILE \
        --workers 60;
done

}


bpe_preprocess
binarize
