#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
CODE_BASE_DIR=`realpath ../..`;

DATASETS=(
    kp20k
    inspec
    nus
    semeval
    krapivin
    kptimes
)

DATASET_NAME=${1:-kp20k};
KEYWORD_TYPE=${2:-present};

if [[ ! " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash retrieve.sh <dataset> <keyword-type>";
    exit;
fi

DATA_DIR=${CODE_BASE_DIR}/retrieval/data;
OUT_DIR=$CURRENT_DIR
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;

FILES=()
if [[ $DATASET_NAME == "kptimes" ]]; then
    FILES+=(${DATA_DIR}/kptimes.train.jsonl)
    FILES+=(${DATA_DIR}/kptimes.valid.jsonl)
    FILES+=(${DATA_DIR}/kptimes.test.jsonl)
    DOMAIN=web
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
else
    FILES+=(${DATA_DIR}/kp20k.train.jsonl)
    FILES+=(${DATA_DIR}/kp20k.valid.jsonl)
    FILES+=(${DATA_DIR}/kp20k.test.jsonl)
    FILES+=(${DATA_DIR}/inspec.test.jsonl)
    FILES+=(${DATA_DIR}/krapivin.test.jsonl)
    FILES+=(${DATA_DIR}/nus.test.jsonl)
    FILES+=(${DATA_DIR}/semeval.test.jsonl)
    DOMAIN=scikp
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
fi

# Create db from preprocessed data.
if [[ ! -f $DB_PATH ]]; then
    python build_db.py --files "${FILES[@]}" --save_path $DB_PATH;
fi

# Index the preprocessed documents.
python build_es.py --db_path $DB_PATH --domain $DOMAIN --config_file_path config.json --port 9200;

# Search documents based on BM25 scores.
OUTFILE=${OUT_DIR}/${DATASET_NAME}.test.${KEYWORD_TYPE}.json
if [[ ! -f $OUTFILE ]]; then
    python es_search.py \
        --index_name ${DOMAIN}_search_test \
        --input_data_file ${DATA_DIR}/${DATASET_NAME}.test.jsonl \
        --output_fp $OUTFILE \
        --keyword $KEYWORD_TYPE \
        --n_docs 100 \
        --port 9200;
fi

python eval_rank.py --input_file $OUTFILE;
