# Fine-tuning BART on Keyphrase Generation
[[https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)]

## Introduction
BART is a sequence-to-sequence model trained with denoising as pretraining objective. See the associated paper for more details. Here, we finetune BART for Keyphrase generation (KP20k, KPTimes) task.

## Prerequisite

- Make sure you have [Fairseq](https://github.com/pytorch/fairseq) (v0.9.0) installed.
- Download and preprocess the datasets: KP20k and KPTimes (if not done). See details [here](https://github.com/wasiahmad/NeuralKpGen/blob/master/data/README.md).
- Then run `bash preprocess.sh`.

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

- The checkpoints are saved in `${dataset_name}` directory.
- The predictions are stored in `dataset_name/${eval_dataset}_hypotheses.txt` file.
- The official evaluation results are written in `results_log_{eval_dataset}.txt` file.


## Reference

- [https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)
