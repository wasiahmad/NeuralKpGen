import os

CHECKPOINT_DIR_PATH = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/models/kp20k"
CKPT_FILENAME = 'dpr_biencoder'
CHECKPOINT = os.path.join(CHECKPOINT_DIR_PATH, CKPT_FILENAME)
OUTPUT_DIR = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs/kp20k"

top_k = 20000
qa_file_suffix = "test"
FILE = "/home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.test.jsonl"
OUTPUT_ENCODED_FILE = "/local/wasiahmad/workspace/projects/NeuralKpGen/retrieval/outputs/kp20k_test.*.pkl"
OUT_FILE = OUTPUT_DIR + str(qa_file_suffix) + "_" + str(top_k) + ".json"
script = 'retriever.py'

DEVICES = [5]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

params = [
    ('dataset', 'KP20k'),
    ('model_file', CHECKPOINT),
    ('ctx_file', FILE),
    ('qa_file', FILE),
    ('encoded_ctx_file', OUTPUT_ENCODED_FILE),
    ('out_file', OUT_FILE),
    ('n-docs', top_k),
    ('batch_size', 64),
    ('match', 'exact'),
    ('sequence_length', 256),
    ('save_or_load_index', '')
]

params = ['--{} {}'.format(p[0], str(p[1])).strip() for p in params]
command = \
    "CUDA_VISIBLE_DEVICES={} " \
    "python {} {}".format(
        CUDA_VISIBLE_DEVICES,
        script,
        ' '.join(params)
    )

print(command, flush=True)
os.system(command)
