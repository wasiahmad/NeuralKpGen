# Fine-tuning MASS on Keyphrase Generation
[[https://arxiv.org/pdf/1905.02450.pdf](https://arxiv.org/pdf/1905.02450.pdf)]

## Introduction
MASS is a masked sequence to sequence pre-training method for language generation. See the associated paper for more details. Here, we finetune MASS for 
Keyphrase generation (KP20k, KPTimes) task.

## Dependency

```
git clone https://github.com/pytorch/fairseq.git --branch v0.9.0 --single-branch
cd  fairseq
pip install --editable ./
```

The path of `fairseq` will be automatically added to `PYTHONPATH` (see [here](https://github.com/wasiahmad/NeuralKpGen/blob/master/mass/run.sh#L9)). During our experiment, we found that `pip install fairseq` raises exception. So, to run MASS model, we need to clone the repository and install it in editable mood.


## Reference

- [https://github.com/microsoft/MASS/tree/master/MASS-summarization](https://github.com/microsoft/MASS/tree/master/MASS-summarization)
