# Present Keyphrase Extraction via Sequence Tagging

## Introduction

Keyphrases are of two types, present and absent. While present keyphrases exactly appear as a contiguous text span in the input document, absent keyphrases do not. Therefore, we can frame the present keyphrase extraction as a sequence tagging task. 

We provide an example as follows.

```
#Input
Real time data aggregation in contention based wireless sensor networks .

#Present_Keyphrases
data aggregation; sensor networks

#Output
"O", "O", "B", "I", "O", "O", "O", "O", "B", "I", "O"
```


### [Supported Models]()

- BERT [[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)]
- SciBERT [[https://arxiv.org/pdf/1903.10676.pdf](https://arxiv.org/pdf/1903.10676.pdf)]
- RoBERTa [[https://arxiv.org/pdf/1907.11692.pdf](https://arxiv.org/pdf/1907.11692.pdf)]
- UniLM [[https://arxiv.org/pdf/1905.03197.pdf](https://arxiv.org/pdf/1905.03197.pdf)]
- MiniLM [[https://arxiv.org/pdf/2002.10957.pdf](https://arxiv.org/pdf/2002.10957.pdf)]
- BART [[https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)]

### Prerequisite

- Make sure you have the [required packages](https://github.com/wasiahmad/NeuralKpGen/blob/master/requirements.txt) installed.
- Run `cd data && bash run.sh`. After finishing, there will be a directory named `processed` inside the `data` directory.
- Run `cd unilm && bash download.sh` to download the `UniLM` model and do necessary conversion to perform training and evaluation.


### Model Training and Evaluation

Both training and evaluation can be done by executing the [run.sh](https://github.com/wasiahmad/NeuralKpGen/blob/master/bart/run.sh) script. Currently, we support training on scientific keyphrase benchmark: KP20k and evaluation on KP20k, NUS, Inspec, Krapivin, and Semeval datasets.


#### Usage

```
$ bash run.sh -h

Syntax: run.sh GPU_ID MODEL_NAME

GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2,3'
MODEL_NAME     Name of the training dataset. choices: [unilm|bert-base|scibert|roberta|bart]
```

For example, if we want to fine-tune and evaluate `scibert`, then execute `run.sh` as follows.

```
bash run.sh 0,1,2,3 scibert
```

**[Notes]**

- The best checkpoint is saved in `kp20k-{MODEL_NAME}/checkpoint-best` directory.
- The predictions are stored in `kp20k-{MODEL_NAME}/{eval_dataset}_hypotheses.txt` file.
- The official evaluation results are written in `results_log_{eval_dataset}.txt` file.

