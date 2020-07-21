#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

AVAILABLE_MODEL_CHOICES=(
    unilm1
    unilm2
    minilm
    bert-tiny
    bert-mini
    bert-small
    bert-medium
    bert-base
    bert-large
    scibert
    roberta
)

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID MODEL_NAME DATASET_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "MODEL_NAME     Model name; choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
        echo "DATASET_NAME   Name of the training dataset. choices: [kp20k|kptimes]"
        echo
        exit;;
   esac
done

PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
NUM_WARM_STEPS=1000
NUM_TRAIN_STEPS=20000


# TRAIN COMMAND
# train GPU_ID DATASET_NAME OUTPUT_DIR MODEL_TYPE MODEL_NAME_OR_PATH
function train () {

# path of training data
TRAIN_FILE=data/${2}_train.json
#TRAIN_FILE=data/sample.json # for debugging

# folder used to save fine-tuned checkpoints
OUTPUT_DIR=$3
make_dir $OUTPUT_DIR

# folder used to cache package dependencies
CACHE_DIR=~/.cache/torch/transformers
LOG_FILENAME=${OUTPUT_DIR}/train_log.txt

export CUDA_VISIBLE_DEVICES=$1
#export CUDA_LAUNCH_BLOCKING="1"

IFS=','
# read the split words into an array based on comma delimiter
read -a GPU_IDS <<< "$1"
PROC_PER_NODE=${#GPU_IDS[@]}

# - m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE

python s2s-ft/run_seq2seq.py \
--train_file ${TRAIN_FILE} \
--output_dir ${OUTPUT_DIR} \
--model_type $4 \
--model_name_or_path $5 \
--save_steps 2000 \
--do_lower_case --fp16 --fp16_opt_level O1 \
--max_source_seq_length 464 --max_target_seq_length 48 \
--per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--learning_rate $LR \
--num_warmup_steps $NUM_WARM_STEPS \
--num_training_steps $NUM_TRAIN_STEPS \
--cache_dir ${CACHE_DIR} \
--workers 60 \
|& tee $LOG_FILENAME

}


# DECODE COMMAND
# decode GPU_ID DATASET_NAME MODEL_TYPE TOKENIZER_NAME MODEL_DIR CHECKPOINT_NAME
function decode () {

export CUDA_VISIBLE_DEVICES=$1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Decoding with setting = $1 $2 $3 $4 $5 $6"

DATASET_NAME=$2
SPLIT=test
MODEL_TYPE=$3
TOKENIZER_NAME=$4
MODEL_DIR=$5
CHECKPOINT=$6
# path of the fine-tuned checkpoint
MODEL_PATH=${MODEL_DIR}/${CHECKPOINT}
# input file that you would like to decode
INPUT_JSON=data/${DATASET_NAME}_${SPLIT}.json

LOG_FILENAME=${MODEL_DIR}/test_log.txt

python s2s-ft/decode_seq2seq.py \
--fp16 --model_type $MODEL_TYPE --tokenizer_name $TOKENIZER_NAME \
--input_file ${INPUT_JSON} --split $SPLIT --do_lower_case \
--model_path ${MODEL_PATH} \
--max_seq_length 512 --max_tgt_length 48 --batch_size 256 \
--beam_size 1 --length_penalty 0 --forbid_duplicate_ngrams --mode s2s \
--forbid_ignore_word ";" \
--output_file ${MODEL_DIR}/${CHECKPOINT}.${DATASET_NAME}.${SPLIT} \
--workers 60 \
|& tee $LOG_FILENAME

}


# EVALUATE COMMAND
# evaluate DATASET_NAME MODEL_DIR CHECKPOINT_NAME
function evaluate () {

DATASET_NAME=$1
MODEL_DIR=$2
CHECKPOINT=$3

python -W ignore kp_eval.py \
--src_file data/${DATASET_NAME}_test.json \
--pred_file ${MODEL_DIR}/${CHECKPOINT}.${DATASET_NAME}.test \
--tgt_dir . \
--log_file $4 \
--k_list 5 M

}

model_choice=$2
dataset_choice=$3

if [[ $model_choice == 'unilm1' ]]; then
    model_type=unilm
    model_name_or_path=unilm1-base-cased
    checkpoint_name=ckpt-20000
elif [[ $model_choice == 'unilm2' ]]; then
    model_type=unilm
    model_name_or_path=unilm1.2-base-uncased
    checkpoint_name=ckpt-20000
elif  [[ $model_choice == 'minilm' ]]; then
    model_type=minilm
    model_name_or_path=minilm-l12-h384-uncased
    checkpoint_name=ckpt-20000
elif  [[ $model_choice == 'bert-tiny' ]]; then
    model_type=xbert
    model_name_or_path=bert-tiny-uncased
    checkpoint_name=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $model_choice == 'bert-mini' ]]; then
    model_type=xbert
    model_name_or_path=bert-mini-uncased
    checkpoint_name=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $model_choice == 'bert-small' ]]; then
    model_type=xbert
    model_name_or_path=bert-small-uncased
    checkpoint_name=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $model_choice == 'bert-medium' ]]; then
    model_type=xbert
    model_name_or_path=bert-medium-uncased
    checkpoint_name=ckpt-20000
elif  [[ $model_choice == 'bert-base' ]]; then
    model_type=bert
    model_name_or_path=bert-base-uncased
    checkpoint_name=ckpt-20000
elif  [[ $model_choice == 'bert-large' ]]; then
    model_type=bert
    model_name_or_path=bert-large-uncased
    checkpoint_name=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=4
    LR=3e-5
    NUM_WARM_STEPS=1000
    NUM_TRAIN_STEPS=20000
elif  [[ $model_choice == 'scibert' ]]; then
    model_type=xbert
    model_name_or_path=scibert_scivocab_uncased
    checkpoint_name=ckpt-20000
elif  [[ $model_choice == 'roberta' ]]; then
    model_type=roberta
    model_name_or_path=roberta-base
    checkpoint_name=ckpt-150000
else
    echo -n "... Wrong model choice!! available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]" ;
    exit 1
fi

output_dir=${model_choice}_${dataset_choice}
if [[ $dataset_choice == 'kp20k' ]]; then
    train "$1" $dataset_choice $output_dir $model_type $model_name_or_path
    for dataset in kp20k inspec krapivin nus semeval; do
        decode "$1" $dataset $model_type $model_name_or_path $output_dir $checkpoint_name
        evaluate $dataset $output_dir $checkpoint_name ${model_choice}_${dataset}
    done
elif [[ $dataset_choice == 'kptimes' ]]; then
    train "$1" $dataset_choice $output_dir $model_type $model_name_or_path
    dataset=kptimes
    decode "$1" $dataset $model_type $model_name_or_path $output_dir $checkpoint_name
    evaluate $dataset $output_dir $checkpoint_name ${model_choice}_${dataset}
else
    echo -n "... Wrong dataset choice!! available choices are: " ;
    echo "kp20k, kptimes" ;
    exit 1
fi
