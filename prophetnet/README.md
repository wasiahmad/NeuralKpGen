# Fine-tuning ProphetNet on Keyphrase Generation
[ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/pdf/2001.04063.pdf)

## Introduction
ProphetNet is a masked sequence to sequence pre-training method for language generation. 
See the associated paper for more details. Here, we finetune ProphetNet for Keyphrase generation (KP20k, KPTimes) task.

## Model Training and Evaluation

Both training and evaluation can be done by executing the [run.sh](https://github.com/wasiahmad/NeuralKpGen/blob/master/prophetnet/run.sh) script.

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

## Reference

- [https://github.com/microsoft/ProphetNet#how-to-use](https://github.com/microsoft/ProphetNet#how-to-use)
