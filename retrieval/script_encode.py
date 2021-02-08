import os

CHECKPOINT_DIR_PATH = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models/kp20k"
CKPT_FILENAME = 'dpr_biencoder'
CHECKPOINT = os.path.join(CHECKPOINT_DIR_PATH, CKPT_FILENAME)
pretrained_model = "bert-base-uncased"

FILE = "/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.test.jsonl"
OUTPUT_ENCODED_FILE = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs/kp20k_test"
script = 'generate.py'

DEVICES = [1, 2, 3, 4]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

params = [
    ('dataset', 'KP20k'),
    ('model_file', CHECKPOINT),
    ('encoder_model_type', 'hf_roberta'),
    ('pretrained_model_cfg', pretrained_model),
    ('batch_size', 128),
    ('ctx_file', FILE),
    ('shard_id', 0),
    ('num_shards', 1),
    ('out_file', OUTPUT_ENCODED_FILE)
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
