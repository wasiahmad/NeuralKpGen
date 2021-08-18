# Pre-trained Language Models for Keyphrase Generation

An Empirical Study on Pre-trained Language Models for Neural Keyphrase Generation.

## Models Trained from Scratch

- RNN-based Seq2seq model
- Transformer

For more details, read [documentation](https://github.com/wasiahmad/NeuralKpGen/tree/master/seq2seq).


## Pre-trained Language Models (PLMs)

#### Autoencoding PLMs

- BERT [[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)]
- SciBERT [[https://arxiv.org/pdf/1903.10676.pdf](https://arxiv.org/pdf/1903.10676.pdf)]
- RoBERTa [[https://arxiv.org/pdf/1907.11692.pdf](https://arxiv.org/pdf/1907.11692.pdf)]

#### Autoregressive PLMs

Also known as Sequence-to-Sequence PLMs.

- MASS [[https://arxiv.org/pdf/1905.02450.pdf](https://arxiv.org/pdf/1905.02450.pdf)]
- BART [[https://arxiv.org/pdf/1910.13461.pdf](https://arxiv.org/pdf/1910.13461.pdf)]
- ProphetNet [[https://arxiv.org/pdf/2001.04063.pdf](https://arxiv.org/pdf/2001.04063.pdf)]

#### Autoencoding and Autoregressive PLMs

- UniLM [[https://arxiv.org/pdf/1905.03197.pdf](https://arxiv.org/pdf/1905.03197.pdf)]
- UniLMv2 [[https://arxiv.org/pdf/2002.12804.pdf](https://arxiv.org/pdf/2002.12804.pdf)]

#### Compressed Pre-trained PLMs

- MiniLM  [[https://arxiv.org/pdf/2002.10957.pdf](https://arxiv.org/pdf/2002.10957.pdf)]
- MiniLMv2 [[https://arxiv.org/pdf/2012.15828.pdf](https://arxiv.org/pdf/2012.15828.pdf)]


## Neural Network (NN) Architecture of PLMs

#### Shallow vs. Deeper Architectures

- [bert_uncased_L-2_H-768_A-12](https://huggingface.co/google/bert_uncased_L-2_H-768_A-12)
- [bert_uncased_L-4_H-768_A-12](https://huggingface.co/google/bert_uncased_L-4_H-768_A-12)
- [bert_uncased_L-6_H-768_A-12](https://huggingface.co/google/bert_uncased_L-6_H-768_A-12)
- [bert_uncased_L-8_H-768_A-12](https://huggingface.co/google/bert_uncased_L-8_H-768_A-12)
- [bert_uncased_L-10_H-768_A-12](https://huggingface.co/google/bert_uncased_L-10_H-768_A-12)
- [bert_uncased_L-12_H-768_A-12](https://huggingface.co/google/bert_uncased_L-12_H-768_A-12)

#### Wide vs. Narrow Architectures

- [bert_uncased_L-6_H-128_A-2](https://huggingface.co/google/bert_uncased_L-6_H-128_A-2)
- [bert_uncased_L-6_H-256_A-4](https://huggingface.co/google/bert_uncased_L-6_H-256_A-4)
- [bert_uncased_L-6_H-512_A-8](https://huggingface.co/google/bert_uncased_L-6_H-512_A-8)
- [bert_uncased_L-6_H-768_A-12](https://huggingface.co/google/bert_uncased_L-6_H-768_A-12)

#### Encoder-only vs. Encoder-Decoder Architectures

[BERT (6L-768H-12A)](https://huggingface.co/google/bert_uncased_L-6_H-768_A-12) vs. [MASS-base-uncased (6L-768H-12A)](https://github.com/microsoft/MASS/tree/master/MASS-summarization#pipeline-for-pre-training)

- Both the models uses the same wordpiece vocabuary (of size 30522).
- Both the models are pre-trained on **Wikipedia + BookCorpus**.
- While BERT has 66M parameters, MASS-base-uncased has 123M parameters.
- If we do not load pre-trained weights, then we can compare the architectures only.
  - Transformer Encoder (6L-768H-12A) vs. Transformer (6L-768H-12A)


## PLMs as Teacher for Knowledge Distillation

[MiniLMv2 Distilled from Different Teacher Models](https://github.com/microsoft/unilm/tree/master/minilm#minilm)

- MiniLMv2-[6L-768H]-distilled-from-**RoBERTa-large**
- MiniLMv2-[6L-768H]-distilled-from-**BERT-large**
- MiniLMv2-[6L-768H]-distilled-from-**BERT-base**
