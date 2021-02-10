#!/usr/bin/env bash

DATASETS=(
    KP20k
    inspec
    nus
    semeval
    krapivin
    KPTimes
)

DATASET_NAME=${1:-"KP20k"};
KEYWORD_TYPE=${2:-"present"};

if [[ " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash retrieve.sh <dataset> <keyword-type>";
    exit;
fi

DATA_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/data";
OUT_DIR="/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/bm25";
mkdir -p $OUT_DIR

CODE_BASE_DIR=`realpath ../..`;
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;

FILES=()
if [[ $DATASET_NAME == "KPTimes" ]]; then
    FILES+=(${DATA_DIR}/KPTimes.train.jsonl)
    FILES+=(${DATA_DIR}/KPTimes.valid.jsonl)
    FILES+=(${DATA_DIR}/KPTimes.test.jsonl)
    DOMAIN=web
    DB_PATH="${DATA_DIR}/${DOMAIN}.db";
elif [[ $DATASET_NAME == "KP20k" ]]; then
    FILES+=(${DATA_DIR}/KP20k.train.jsonl)
    FILES+=(${DATA_DIR}/KP20k.valid.jsonl)
    FILES+=(${DATA_DIR}/KP20k.test.jsonl)
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
if [[ ! -f ${OUT_DIR}/${DATASET_NAME}.test.json ]]; then
python es_search.py \
    --index_name ${DOMAIN}_search_test \
    --input_data_file ${DATA_DIR}/${DATASET_NAME}.test.jsonl \
    --output_fp ${OUT_DIR}/${DATASET_NAME}.test.json \
    --keyword $KEYWORD_TYPE \
    --n_docs 100 \
    --port 9200;
fi
