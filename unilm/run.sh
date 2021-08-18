#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;
# folder used to cache package dependencies
CACHE_DIR=~/.cache/torch/transformers
MODEL_DIR=${HOME_DIR}/models

export PYTHONPATH=$HOME_DIR
export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
DATASET=$3


function train () {

PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
NUM_WARM_STEPS=1000
NUM_TRAIN_STEPS=20000
TRAIN_FILE=${DATA_DIR_PREFIX}/${DATASET}/json/train.json

python train.py \
    --train_file ${TRAIN_FILE} \
    --output_dir $SAVE_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --save_steps 2000 \
    --do_lower_case \
    --fp16 --fp16_opt_level O1 \
    --max_source_seq_length 464 \
    --max_target_seq_length 48 \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LR \
    --num_warmup_steps $NUM_WARM_STEPS \
    --num_training_steps $NUM_TRAIN_STEPS \
    --cache_dir $CACHE_DIR \
    --workers 60 \
    2>&1 | tee ${SAVE_DIR}/finetune.log;

}


function decode () {

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

EVAL_DATASET=$1
SPLIT=test
INPUT_FILE=${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/${SPLIT}.json

python decode.py \
    --fp16 \
    --model_type $MODEL_TYPE \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --input_file $INPUT_FILE \
    --split $SPLIT \
    --do_lower_case \
    --model_path $SAVE_DIR/$CKPT_NAME \
    --max_seq_length 512 \
    --max_tgt_length 48 \
    --batch_size 256 \
    --beam_size 1 \
    --mode s2s \
    --output_file $SAVE_DIR/$CKPT_NAME.$EVAL_DATASET.$SPLIT \
    --workers 60 \
    2>&1 | tee $SAVE_DIR/decoding.log;

}


function evaluate () {

EVAL_DATASET=$1
SPLIT=test
INPUT_FILE=${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/${SPLIT}.json

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_file $INPUT_FILE \
    --pred_file $SAVE_DIR/$CKPT_NAME.$EVAL_DATASET.$SPLIT \
    --file_prefix $SAVE_DIR/${EVAL_DATASET} \
    --tgt_dir $SAVE_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 M;

}

AVAILABLE_MODEL_CHOICES=(
    unilm1
    unilm2
    minilm
    minilm2-bert-base
    minilm2-bert-large
    minilm2-roberta
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

if [[ $MODEL == 'unilm1' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1-base-cased
    CKPT_NAME=ckpt-20000
elif [[ $MODEL == 'unilm2' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1.2-base-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm' ]]; then
    MODEL_TYPE=minilm
    MODEL_NAME_OR_PATH=minilm-l12-h384-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'bert-tiny' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=bert-tiny-uncased
    CKPT_NAME=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $MODEL == 'bert-mini' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=bert-mini-uncased
    CKPT_NAME=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $MODEL == 'bert-small' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=bert-small-uncased
    CKPT_NAME=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $MODEL == 'bert-medium' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=bert-medium-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'bert-base' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-base-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'bert-large' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-large-uncased
    CKPT_NAME=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=4
    LR=3e-5
    NUM_WARM_STEPS=1000
    NUM_TRAIN_STEPS=20000
elif  [[ $MODEL == 'scibert' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=scibert_scivocab_uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'roberta' ]]; then
    MODEL_TYPE=roberta
    MODEL_NAME_OR_PATH=roberta-base
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-bert-base' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Base
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-bert-large' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Large
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-roberta' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-RoBERTa-Large
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == bert_uncased_L-*_H-*_A-* ]]; then
    L=$(echo $MODEL | cut -f2 -d- | cut -f1 -d_)
    H=$(echo $MODEL | cut -f3 -d- | cut -f1 -d_)
    A=$(echo $MODEL | cut -f4 -d-)
    [[ "$L" =~ ^(2|4|6|8|10|12) ]] || { echo "Unknown BERT model configuration."; exit 1; }
    [[ "$H" =~ ^(128|256|512|768) ]] || { echo "Unknown BERT model configuration."; exit 1; }
    [[ "$A" == "$((H / 64))" ]] || { echo "Unknown BERT model configuration."; exit 1; }
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=$MODEL
    CKPT_NAME=ckpt-20000
else
    echo -n "... Wrong model choice!! available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]";
    exit 1
fi

SAVE_DIR=${CURRENT_DIR}/${MODEL}_${DATASET}
mkdir -p $SAVE_DIR

if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    train
    for dataset in kp20k inspec krapivin nus semeval; do
        decode $dataset
        evaluate $dataset
    done
elif [[ $DATASET == 'kptimes' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data
    train
    decode $DATASET
    evaluate $DATASET
else
    echo -n "... Wrong dataset choice!! available choices are: " ;
    echo "kp20k, kptimes" ;
    exit 1
fi
