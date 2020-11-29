#!/usr/bin/env bash

mkdir -p download
BASE_URL=https://huggingface.co/microsoft

curl ${BASE_URL}/unilm-base-cased/resolve/main/config.json -o ./download/
curl ${BASE_URL}/unilm-base-cased/resolve/main/pytorch_model.bin -o ./download/
curl ${BASE_URL}/unilm-base-cased/resolve/main/vocab.txt -o ./download/

python convert.py;
