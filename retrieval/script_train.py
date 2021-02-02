import os

CHECKPOINT_DIR_PATH = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models/kp20k"
CKPT_FILENAME = 'dpr_biencoder'

pretrained_model = "bert-base-uncased"
DEVICES = [0, 1, 2, 3]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

dataset = 'KP20k'
train_file = '/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.train.jsonl'
dev_file = '/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.valid.jsonl'
script = 'train_encoder.py'

params = [
    ('dataset', 'KP20k'),
    ('output_dir', CHECKPOINT_DIR_PATH),
    ('checkpoint_file_name', CKPT_FILENAME),
    ('fp16', ''),
    ('batch_size', 8),
    ('dev_batch_size', 16),
    ('train_file', train_file),
    ('dev_file', dev_file),
    ('sequence_length', 256),
    ('num_train_epochs', 20),
    ('eval_per_epoch', 1),
    ('learning_rate', 2e-5),
    ('max_grad_norm', 2.0),
    ('encoder_model_type', 'hf_bert'),
    ('pretrained_model_cfg', pretrained_model),
    ('val_av_rank_start_epoch', 1),
    ('warmup_steps', 1237),
    ('seed', 1234)
]

params = ['--{} {}'.format(p[0], str(p[1])).strip() for p in params]
command = \
    "CUDA_VISIBLE_DEVICES={} " \
    "python -m torch.distributed.launch " \
    "--nproc_per_node={} {} {}".format(
        CUDA_VISIBLE_DEVICES,
        str(len(DEVICES)),
        script,
        ' '.join(params)
    )

print(command, flush=True)
os.system(command)
