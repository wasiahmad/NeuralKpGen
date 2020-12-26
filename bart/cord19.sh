#!/usr/bin/env bash

DATA_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/cord19
DATE=2020-12-12
T_DIR=$DATA_DIR/${DATE}

SRCDIR=data/cord19
mkdir -p $SRCDIR

DICT_FILE=bart.base/dict.txt # dict.txt


function bpe_preprocess () {

if [[ ! -f $SRCDIR/test.bpe.source ]]; then
python encode.py \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs $T_DIR/test.source \
    --outputs $SRCDIR/test.bpe.source \
    --max_len 510 \
    --workers 60;
fi

}


function process () {

if [[ ! -d $SRCDIR/binary ]]; then
fairseq-preprocess \
    --only-source \
    --source-lang source \
    --target-lang target \
    --testpref $SRCDIR/test.bpe \
    --destdir $SRCDIR/binary \
    --workers 60 \
    --srcdict $DICT_FILE;
fi

cp $DICT_FILE $SRCDIR/binary/dict.target.txt

}


function decode () {

export CUDA_VISIBLE_DEVICES=$1
MODEL_DIR=$2

python decode.py \
--data_name_or_path $SRCDIR/binary \
--data_dir $T_DIR \
--checkpoint_dir $MODEL_DIR \
--checkpoint_file checkpoint_best.pt \
--output_file $SRCDIR/test.hypo \
--batch_size 64 \
--beam 1 \
--min_len 1 \
--lenpen 1.0 \
--no_repeat_ngram_size 3 \
--max_len_b 60;

}


function generate () {

export CUDA_VISIBLE_DEVICES=$1
MODEL=$2/checkpoint_best.pt

fairseq-generate $SRCDIR/binary \
--path $MODEL \
--task translation \
--log-format simple \
--batch-size 256 \
--bpe 'gpt2' \
--beam 1 \
--min-len 1 \
--no-repeat-ngram-size 3 \
--max-len-b 60 2>&1 | tee $SRCDIR/output.txt;

cat $SRCDIR/output.txt | grep -P "^D" |sort -V |cut -f 3- | cut -d' ' -f 2- > $SRCDIR/test.hypo;

}


bpe_preprocess
process
#decode $1 kp20k_checkpoints
generate $1 kp20k_checkpoints
