# Fine-tuning BART on Keyphrase Generation
[[https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)]

## Introduction
BART is a sequence-to-sequence model trained with denoising as pretraining objective. See the associated paper for more details. Here, we finetune BART for Keyphrase generation (KP20k, KPTimes) task.

## Prerequisite

- Make sure you have [Fairseq](https://github.com/pytorch/fairseq) (v0.9.0) installed.
- Download and preprocess the datasets: KP20k and KPTimes (if not done). See details [here](https://github.com/wasiahmad/NeuralKpGen/blob/master/data/README.md).
- The run `bash preprocess.sh`.

Once the datasets are processed, we will see a directory named `data` with the following structure.

```
data
  |-kp20k-bin
  |  |-test.source-target.source.bin
  |  |-test.source-target.target.idx
  |  |-dict.source.txt
  |  |-test.source-target.target.bin
  |  |-test.source-target.source.idx
  |  |-dict.target.txt
  |  |-train.source-target.source.bin
  |  |-train.source-target.target.idx
  |  |-valid.source-target.source.bin
  |  |-valid.source-target.target.idx
  |  |-preprocess.log
  |  |-train.source-target.target.bin
  |  |-train.source-target.source.idx
  |  |-valid.source-target.target.bin
  |  |-valid.source-target.source.idx
  |-nus-bin
  |  |-preprocess.log
  |  |-test.source-target.target.bin
  |  |-test.source-target.source.idx
  |  |-dict.target.txt
  |  |-test.source-target.source.bin
  |  |-test.source-target.target.idx
  |  |-dict.source.txt
  |-inspec-bin
  |  |-dict.target.txt
  |  |-dict.source.txt
  |  |-preprocess.log
  |  |-test.source-target.source.idx
  |  |-test.source-target.target.bin
  |  |-test.source-target.target.idx
  |  |-test.source-target.source.bin
  |-nus
  |  |-test.source
  |  |-test.bpe.source
  |  |-test.bpe.target
  |  |-test.target
  |-kptimes
  |  |-test.bpe.source
  |  |-valid.bpe.target
  |  |-train.source
  |  |-test.bpe.target
  |  |-train.target
  |  |-valid.bpe.source
  |  |-test.source
  |  |-valid.target
  |  |-train.bpe.source
  |  |-test.target
  |  |-train.bpe.target
  |  |-valid.source
  |-kp20k
  |  |-test.bpe.source
  |  |-train.target
  |  |-train.bpe.target
  |  |-test.target
  |  |-train.source
  |  |-test.bpe.target
  |  |-test.source
  |  |-train.bpe.source
  |  |-valid.source
  |  |-valid.bpe.source
  |  |-valid.target
  |  |-valid.bpe.target
  |-inspec
  |  |-test.source
  |  |-test.bpe.target
  |  |-test.bpe.source
  |  |-test.target
  |-kptimes-bin
  |  |-dict.target.txt
  |  |-dict.source.txt
  |  |-train.source-target.target.bin
  |  |-test.source-target.source.idx
  |  |-test.source-target.target.bin
  |  |-train.source-target.source.idx
  |  |-valid.source-target.target.bin
  |  |-valid.source-target.source.idx
  |  |-train.source-target.source.bin
  |  |-test.source-target.target.idx
  |  |-test.source-target.source.bin
  |  |-train.source-target.target.idx
  |  |-preprocess.log
  |  |-valid.source-target.source.bin
  |  |-valid.source-target.target.idx
  |-krapivin-bin
  |  |-dict.target.txt
  |  |-test.source-target.target.idx
  |  |-test.source-target.source.bin
  |  |-dict.source.txt
  |  |-preprocess.log
  |  |-test.source-target.source.idx
  |  |-test.source-target.target.bin
  |-krapivin
  |  |-test.bpe.target
  |  |-test.bpe.source
  |  |-test.source
  |  |-test.target
  |-semeval
  |  |-test.bpe.source
  |  |-test.bpe.target
  |  |-test.source
  |  |-test.target
  |-semeval-bin
  |  |-dict.target.txt
  |  |-dict.source.txt
  |  |-test.source-target.source.idx
  |  |-test.source-target.target.bin
  |  |-preprocess.log
  |  |-test.source-target.target.idx
  |  |-test.source-target.source.bin
```


## Model Training and Evaluation

Both training and evaluation can be done by executing the [run.sh](https://github.com/wasiahmad/NeuralKpGen/blob/master/bart/run.sh) script.

### Usage

```
$ bash run.sh -h

Syntax: run.sh GPU_ID DATASET_NAME

GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'
DATASET_NAME   Name of the training dataset. choices: [kp20k|kptimes]
```

For example, if we want to fine-tune and evaluate on KP20k dataset, then execute `run.sh` as follows.

```
bash run.sh 0,1,2,3 kp20k
```

**[Notes]**

- The checkpoints are saved in `{dataset_name}_checkpoints` directory.
- The predictions are stored in `logs/{dataset_name}_test.hypo` file.
- The official evaluation results are written in `results_log_{dataset_name}.txt` file.


## Reference

- [https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)
