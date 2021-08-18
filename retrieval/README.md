# Dense Passage Retrieval

Dense Passage Retrieval (`DPR`) - is a set of tools and models for state-of-the-art open-domain Q&A research.
It is based on the following paper:

Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih, [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Preprint 2020.

Please check https://github.com/facebookresearch/DPR.

## Results on Gold Keyphrases

- We consider retrieving top-100 documents.

#### Present Keyphrases

|          | # Ex.   | Acc.  | MAP   |
| -------  | :----:  | :---: | :---: |
| KP20k    | 20,000  |  0.91 |  0.65 |
| Inspec   | 500     |  0.99 |  0.96 |
| Nus      | 211     |  0.96 |  0.83 |
| Krapivin | 400     |  0.90 |  0.60 |
| SemEval  | 100     |  0.77 |  0.43 |
| KPTimes  | 20,000  |  0.67 |  0.27 |

#### Absent Keyphrases

|          | # Ex.   | Acc.   | MAP   |
| -------  | :-----: | :----: | :---: |
| KP20k    | 20,000  |  0.47  |  0.15 |
| Inspec   | 500     |  0.62  |  0.32 |
| Nus      | 211     |  0.63  |  0.26 |
| Krapivin | 400     |  0.46  |  0.14 |
| SemEval  | 100     |  0.48  |  0.09 |
| KPTimes  | 20,000  |  0.48  |  0.18 |
