#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

function Help () {
   # Display Help
   echo
   echo "Syntax: test.sh GPU_ID MODEL_DIR MODEL_NAME DATASET_NAME"
   echo
   echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
   echo "MODEL_DIR      Directory name which should be at ../tmp/"
   echo "MODEL_NAME     A string to name the model and log files."
   echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
   echo
}

SRC_DIR=..

function test () {

echo "============TESTING============"

array=("kp20k nus semeval krapivin inspec")
if [[ " ${array[@]} " =~ " ${4} " ]]; then
    data_dir=${SRC_DIR}/data/scikp
    test_file=processed/${4}/test.json
else
    data_dir=${SRC_DIR}/data/${4}
    test_file=processed/test.json
fi

export PYTHONPATH=$SRC_DIR
export CUDA_VISIBLE_DEVICES=$1

python -W ignore ${SRC_DIR}/seq2seq/train.py \
--only_test True \
--data_workers 0 \
--dataset_name $4 \
--data_dir $data_dir \
--model_dir ${SRC_DIR}/tmp/${2} \
--model_name $3 \
--dev_file $test_file \
--max_examples -1 \
--test_batch_size 256 \
--disable_extra_one_word_filter True \
--no_source_output True \
--k_list 5 M \
--log_file $4_$3 \
--pred_file $4_$3 \

}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

# RUN THE SCRIPT as follows
# bash test.sh GPU_ID MODEL_DIR MODEL_NAME DATASET_NAME
# bash test.sh 1 kp20k kp20k_rnn nus
test $1 $2 $3 $4
