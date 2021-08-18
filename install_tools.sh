#!/usr/bin/env bash

set -eux  # for easier debugging

REPO=$PWD
LIB=$REPO/third_party
mkdir -p $LIB

# install conda env
conda create --name neuralkpgen --file conda-env.txt
conda init bash
conda activate neuralkpgen

pip install sacrebleu==1.2.11

cd $LIB
# install fairseq
rm -rf fairseq
git clone https://github.com/pytorch/fairseq.git --branch v0.9.0 --single-branch
cd fairseq
pip install .
cd ..
# install transformer
git clone https://github.com/huggingface/transformers --branch v3.0.2 --single-branch
cd transformers
pip install .
cd ..
# install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd $LIB