# BM25 based Retriever

This tool is borrowed from https://github.com/AkariAsai/XORQA/tree/main/baselines/bm25.

## Elastic Search Insolation

To run ElasticSearch in your local environment, you need to install ElasticSearch first. We install ES by running 
the scripts provided by [CLIReval](https://github.com/ssun32/CLIReval) library (Sun et al., ACL demo 2020).

```
git clone https://github.com/ssun32/CLIReval.git
cd CLIReval
pip install -r requirements.txt
bash scripts/install_external_tools.sh
```

Whenever you run ES in your local environment, you need to start an ES instance.

```
bash scripts/server.sh [start|stop]
```

## Index documents and search

Steps involved are:

- Create db from preprocessed data.
- Index the preprocessed documents.
- Search documents based on BM25 scores.

## Results

- We consider retrieving top-100 documents.

#### Present Keyphrases

|          | # Ex.   | Acc.  | MAP   |
| -------  | :----:  | :---: | :---: |
| KP20k    | 20,000  |  0.90 |  ?    |
| Inspec   | 500     |  0.99 |  ?    | 
| Nus      | 211     |  0.95 |  ?    |
| Krapivin | 400     |  0.87 |  ?    |
| SemEval  | 100     |  0.38 |  ?    |
| KPTimes  | 20,000  |  ?    |  ?    |

#### Absent Keyphrases

|          | # Ex.   | Acc.   | MAP   |
| -------  | :-----: | :----: | :---: |
| KP20k    | 20,000  |  ?     |  ?    |
| Inspec   | 500     |  0.05  |  ?    | 
| Nus      | 211     |  0.005 |  ?    |
| Krapivin | 400     |  0.008 |  ?    |
| SemEval  | 100     |  0.01  |  ?    |
| KPTimes  | 20,000  |  ?     |  ?    |
