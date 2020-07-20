#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

function Help () {
   # Display Help
   echo
   echo "Syntax: train.sh GPU_ID DATASET_NAME MODEL_NAME"
   echo
   echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
   echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
   echo "MODEL_NAME     A string to name the model and log files."
   echo
}

SRC_DIR=..

function train () {

RGPU=$1
DATASET=$2
MODEL_NAME=$3

array=("kp20k nus semeval krapivin inspec")
if [[ " ${array[@]} " =~ " ${DATASET} " ]]; then
    DATA_DIR=${SRC_DIR}/data/scikp
    TRAIN_FILE=${DATASET}/train.json
    VALID_FILE=${DATASET}/valid.json
    VOCAB_FILE=${DATASET}/vocab.txt
else
    DATA_DIR=${SRC_DIR}/data/${DATASET}
    TRAIN_FILE=processed/train.json
    VALID_FILE=processed/valid.json
    VOCAB_FILE=processed/vocab.txt
fi

MODEL_DIR=${SRC_DIR}/tmp
make_dir $MODEL_DIR
MODEL_DIR=${SRC_DIR}/tmp/${DATASET}
make_dir $MODEL_DIR


echo "============TRAINING============"

export PYTHONPATH=$SRC_DIR
export CUDA_VISIBLE_DEVICES=$RGPU

python -W ignore ${SRC_DIR}/seq2seq/train.py \
--config config.yaml \
--data_workers 5 \
--fp16 True \
--dataset_name $DATASET \
--data_dir $DATA_DIR \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--train_file $TRAIN_FILE \
--dev_file $VALID_FILE \
--vocab_file $VOCAB_FILE \
--max_examples -1 \
--batch_size 80 \
--test_batch_size 80 \
--num_epochs 50 \
--early_stop 3 \
--valid_metric f1_at_m_all \
--checkpoint True \
--disable_extra_one_word_filter True \
--no_source_output True \
--k_list 5 M

}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

# RUN THE SCRIPT as follows
# bash train.sh GPU_ID DATASET_NAME MODEL_NAME
# bash train.sh 1 kp20k kp20k_rnn
train $1 $2 $3
