#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs

FAIRSEQ_PATH=$(python -c "import fairseq; print(fairseq.__path__[0])")
FAIRSEQ_PATH="$(dirname "$FAIRSEQ_PATH")" # parent directory
echo "Fairseq path: $FAIRSEQ_PATH"
export PYTHONPATH=${FAIRSEQ_PATH}:"${PYTHONPATH}"

function train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=1000
UPDATE_FREQ=4
MAX_TOKENS=4096
PRETRAINED_MODEL=mass-base-uncased/mass-base-uncased.pt

fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--user-dir mass --task translation_mass --arch transformer_mass_base \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --min-lr 1e-09 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--update-freq $UPDATE_FREQ --max-tokens $MAX_TOKENS \
--fp16 --ddp-backend=no_c10d \
--max-source-positions 512 --max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--load-from-pretrained-model $PRETRAINED_MODEL \
--save-dir ${DATASET}_checkpoints;

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
LOG_FILE=$3
SAVE_DIR_PREFIX=$4
MODEL=${SAVE_DIR_PREFIX}_checkpoints/checkpoint_best.pt

fairseq-generate $SRCDIR/${DATASET}-bin/ \
--path $MODEL \
--user-dir mass \
--task translation_mass \
--log-format simple \
--batch-size 64 \
--beam 1 \
--min-len 1 \
--no-repeat-ngram-size 3 \
--max-len-b 60 \
--results-path ${SAVE_DIR_PREFIX}_checkpoints \
|& tee $LOG_FILE

}

function evaluate () {

python -W ignore kp_eval.py \
--src_dir $1 \
--file_prefix $2 \
--tgt_dir . \
--log_file $3 \
--k_list 5 M

}

if [[ $2 == 'kp20k' ]]; then
    train "$1" $2
    for dataset in kp20k inspec krapivin nus semeval; do
        decode "$1" $dataset "logs/${dataset}_out.txt" $2
        grep ^S "logs/${dataset}_out.txt" | cut -f1 > "logs/${dataset}_source.txt"
        grep ^T "logs/${dataset}_out.txt" | cut -f2- > "logs/${dataset}_target.txt"
        grep ^H "logs/${dataset}_out.txt" | cut -f3- > "logs/${dataset}_hypotheses.txt"
        evaluate "data/raw/${dataset}" "logs/${dataset}" ${dataset}
    done
elif [[ $2 == 'kptimes' ]]; then
    train "$1" $2
    decode "$1" $2 "logs/${2}_out.txt" $2
    grep ^S "logs/${2}_out.txt" | cut -f1 > "logs/${2}_source.txt"
    grep ^T "logs/${2}_out.txt" | cut -f2- > "logs/${2}_target.txt"
    grep ^H "logs/${2}_out.txt" | cut -f3- > "logs/${2}_hypotheses.txt"
    evaluate "data/raw/${2}" "logs/${2}" ${2}
fi

