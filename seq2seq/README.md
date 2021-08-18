# Fine-tuning Seq2Seq Architecture on Keyphrase Generation

## Introduction
We investigate the performance of an RNN-based Sequence-to-Sequence (Seq2Seq) architecture and Transformer on keyphrase
generation. A few points to note.

- We use the BERT Vocabulary and train the word embeddings from scratch.
- There is no copying mechanism involved.

We present the architecture details below.

### RNN-based Seq2seq Architecture

- Embedding size - 512
- Hidden size - 512
- Number of layers - 3

### Transformer

- Number of layers - 6
- Number of attention heads - 12
- Model size - 768

## Model Training and Evaluation

Both training and evaluation can be done by executing the [run.sh](https://github.com/wasiahmad/NeuralKpGen/blob/master/seq2seq/run.sh) script.

### Usage

```
$ bash run.sh -h

Syntax: run.sh GPU_ID DATASET_NAME MODEL_TYPE

GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'
DATASET_NAME   Name of the training dataset. choices: [kp20k|kptimes]
MODEL_TYPE     Model type. Choices: rnn, transformer. 
```

For example, if we want to fine-tune and evaluate on KP20k dataset, then execute `run.sh` as follows.

```
bash run.sh 0,1,2,3 kp20k rnn
```

## Reference

- [https://github.com/microsoft/ProphetNet#how-to-use](https://github.com/microsoft/ProphetNet#how-to-use)
