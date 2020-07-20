# NeuralKpGen

Implementation of the recurrent neural networks (RNNs) and Transformer based neural keyphrase generation methods. `requirements.txt` includes a subset of all the possible required packages.


## Data Preprocessing

Data processing details are provided [here](https://github.com/wasiahmad/NeuralKpGen/tree/master/data). 

- **Training Datasets**: ["kp20k", "kptimes", "stackex", "oagk", "openkp"]
- **Evaluation Datasets**: ["kp20k", "nus", "semeval", "krapivin", "inspec" "kptimes", "stackex", "oagk", "openkp"]


## Training and Evaluation

We can train and evaluate RNN-based Seq2seq model or Transformer on various datasets. 

### Training

```
$ cd  scripts
$ bash train.sh GPU_ID DATASET_NAME MODEL_NAME
```
Where, `DATASET_NAME` should be one of `Training Datasets` listed above and `MODEL_NAME` is a string to name the model and log files. All the models and log files are stored at `tmp/DATASET_NAME` directory.

#### Running experiments on CPU/GPU/Multi-GPU

- If GPU_ID is set to -1, CPU will be used.
- If GPU_ID is set to one specific number, only one GPU will be used.
- If GPU_ID is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.


### Evaluation

```
$ cd  scripts
$ bash test.sh GPU_ID MODEL_DIR MODEL_NAME DATASET_NAME
```
Where, `DATASET_NAME` should be one of `Evaluation Datasets`.

**Evaluation metrics**: Precision@5, Recall@5, F1@5, Precision@M, Recall@M, F1@M


### Generated log files

While training and evaluating EG-Net, a list of files are generated inside a `tmp` directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the experiment.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - Contains the predictions and gold references.


## FAQ

#### How can we set the model parameters?

We can set the hyper-parameters in the [config.yaml](https://github.com/wasiahmad/NeuralKpGen/blob/master/scripts/config.yaml) file.

#### How many GPUs do we need to train the models?

- GPU requirements depend on the `batch_size`, `max_src_len`, and `max_tgt_len`.
- We trained the models using 4 GeForce GTX 1080 GPUs (adjusted the `batch_size` accordingly).

#### Do we use BEAM decoding to generate keyphrases?

No, we only use greedy decoding!


## Notes

- Evaluation scripts are adapted from [keyphrase-generation-rl](https://github.com/kenchan0226/keyphrase-generation-rl).

## Acknowledgement

I borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA) and [OpenNMT](https://github.com/OpenNMT/OpenNMT-py). I thank the authors of these repositeries for their effort.

