# Pre-trained Language Models for Keyphrase Generation Task

Here, we finetune Pre-trained NLU and NLG models for Keyphrase generation (KP20k, KPTimes) task. 

## Supported Models

- BERT [[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)]
- SciBERT [[https://arxiv.org/pdf/1903.10676.pdf](https://arxiv.org/pdf/1903.10676.pdf)]
- RoBERTa [[https://arxiv.org/pdf/1907.11692.pdf](https://arxiv.org/pdf/1907.11692.pdf)]
- UniLM [[https://arxiv.org/pdf/1905.03197.pdf](https://arxiv.org/pdf/1905.03197.pdf)]
- UniLMv2 [[https://arxiv.org/pdf/2002.12804.pdf](https://arxiv.org/pdf/2002.12804.pdf)]
- MiniLM [[https://arxiv.org/pdf/2002.10957.pdf](https://arxiv.org/pdf/2002.10957.pdf)]
- MiniLMv2 [[https://arxiv.org/pdf/2012.15828.pdf](https://arxiv.org/pdf/2012.15828.pdf)]
- [bert_uncased_L-X_H-Y_A-Z](https://huggingface.co/google) [[https://arxiv.org/pdf/1908.08962.pdf](https://arxiv.org/pdf/1908.08962.pdf)]
    - Where X = [2, 4, 6, 8, 10, 12], Y = [128, 256, 512, 768], Z = Y // 64
    - E.g., bert_uncased_L-6_H-768_A-12

## Prerequisite

- Make sure you have the [required packages](https://github.com/wasiahmad/NeuralKpGen/blob/master/requirements.txt) installed.
- Download and preprocess the datasets: KP20k, OAGK, and KPTimes (if not done). See details [here](https://github.com/wasiahmad/NeuralKpGen/blob/master/data/README.md).
- Then run `python prepare.py`.


## Model Training and Evaluation

Both training and evaluation can be done by executing the [run.sh](https://github.com/wasiahmad/NeuralKpGen/blob/master/unilm/run.sh) script.

### Usage

```
$ bash run.sh -h

Syntax: run.sh GPU_ID MODEL_NAME DATASET_NAME

GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'
MODEL_NAME     Model name; choices: [unilm1|unilm2|minilm|minilm2-bert-base|minilm2-bert-large|minilm2-roberta|bert-tiny|bert-mini|bert-small|bert-medium|bert-base|bert-large|scibert|roberta]
DATASET_NAME   Name of the training dataset. choices: [kp20k|oagk|kptimes]
```

For example, if we want to fine-tune and evaluate UniLM model on KP20k dataset, then execute `run.sh` as follows.

```
bash run.sh 0,1,2,3 unilm1 kp20k
```

Once training and evaluation is done, we will see a directory `unilm1_kp20k` containing checkpoints, log files, etc. as follows.

```
unilm1_kp20k
  |-cached_features_for_training.pt
  |-ckpt-**
  |  |-config.json
  |  |-optimizer.pt
  |  |-pytorch_model.bin
  |  |-scheduler.pt
  |  |-special_tokens_map.json
  |  |-tokenizer_config.json
  |  |-training_args.bin
  |  |-vocab.txt
  |-ckpt-20000.inspec.test
  |-ckpt-20000.kp20k.test
  |-ckpt-20000.krapivin.test
  |-ckpt-20000.nus.test
  |-ckpt-20000.semeval.test
  |-out.txt
  |-test_log.txt
  |-train_log.txt
  |-train_opt.json
```

**[Notes]**

- When **kp20k|oagk** is used as the training dataset, the models are evaluated on five additional datasets: `[kp20k|oagk, nus, semeval, krapivin, inspec]`.
- The predicted keyphrases are written in `ckpt-20000.[dataset_name].test` files.
- The official evaluation results are written in `results_log_unilm1_[dataset_name].txt` files.

### Training Setup

**[Important]** For training a model on a particular dataset, we need to set the values of 5 hyper-parameters.

```
PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
NUM_WARM_STEPS=1000
NUM_TRAIN_STEPS=20000
```

- We empirically found that training a model for ~5 epochs on KP20k, KPTimes suffices. 
- For example, to iterate over KP20k (~510,000 training examples) 5 times, we need to perform ~20,000 parameter updates with an effective batch size of 128.
- Note that, effective batch size is equal to:

<p align="center">
  <b>PER_GPU_TRAIN_BATCH_SIZE * NUM_GPU * GRADIENT_ACCUMULATION_STEPS</b>
</p>

- In our experiments, we used 4 NVIDIA GeForce RTX 2080 (11gb) GPUs to achieve the effective batch size of 128 (`8*4*4)`. 
- We set `NUM_WARM_STEPS` as `0.05 * NUM_TRAIN_STEPS`. 
- We need to tune `LR` for a given dataset/model. Note that, the default value (1e-4) may not work in a particular setting.

## Reference

- [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm)


