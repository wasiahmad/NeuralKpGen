#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib

import argparse
import time
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from retrieval.dpr.models import init_biencoder_components
from retrieval.dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from retrieval.dpr.utils.data_utils import Tensorizer
from retrieval.dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, \
    move_to_device
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def gen_ctx_vectors(
        ctx_rows: List[Tuple[object, str, str]],
        model: nn.Module,
        tensorizer: Tensorizer,
        insert_title: bool = True
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = []
    for j, batch_start in enumerate(range(0, n, bsz)):

        batch_token_tensors = [
            tensorizer.text_to_tensor(
                ctx[1], type='context',
                title=ctx[2] if insert_title else None)
            for ctx in ctx_rows[batch_start:batch_start + bsz]
        ]

        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu()

        ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

        assert len(ctx_ids) == out.size(0)

        total += len(ctx_ids)

        results.extend([
            (ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))
        ])

        if total % 10 == 0:
            logger.info('Encoded passages %d', total)

    return results


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    if args.context_type == 'keywords':
        encoder = encoder.question_model
    else:
        encoder = encoder.ctx_model

    # the following doesn't work for fp16
    # encoder, _ = setup_for_distributed_mode(
    #     encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16, args.fp16_opt_level
    # )
    encoder.to(args.device)
    if args.fp16:
        encoder = encoder.half()
        if args.n_gpu > 1:
            encoder = torch.nn.DataParallel(encoder)

    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')
    logger.debug('saved model keys =%s', saved_state.model_dict.keys())

    if args.context_type == 'keywords':
        prefix_len = len('question_model.')
        model_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items()
            if key.startswith('question_model.')
        }
    else:
        prefix_len = len('ctx_model.')
        model_state = {
            key[prefix_len:]: value for (key, value) in saved_state.model_dict.items()
            if key.startswith('ctx_model.')
        }

    model_to_load.load_state_dict(model_state)

    logger.info('reading data from file=%s', ', '.join(args.ctx_file))

    data = []
    if args.dataset in ["kp20k", "kptimes"]:
        for file in args.ctx_file:
            with open(file) as jsonlfile:
                for line in jsonlfile:
                    ex = json.loads(line)
                    if args.context_type == 'keywords':
                        keywords = ex["present"] + ex["absent"]
                        text = ' ; '.join(keywords)
                    else:
                        text = ex["title"] + \
                               ' {} '.format(tensorizer.tokenizer.sep_token) + \
                               ex["abstract"]
                    data.append((ex['id'], text, None))

    shard_id = 0
    start_time = time.time()
    while True:
        start_idx = shard_id * args.shard_size
        if start_idx >= len(data):
            break
        end_idx = start_idx + args.shard_size
        if end_idx >= len(data):
            end_idx = len(data)
        logger.info(
            'Producing encodings for passages range: %d to %d (out of total %d)',
            start_idx, end_idx, len(data)
        )
        rows = data[start_idx:end_idx]
        vectors = gen_ctx_vectors(rows, encoder, tensorizer, False)
        file = args.out_file + '_' + str(shard_id) + '.pkl'
        pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        logger.info('Writing results to %s' % file)
        with open(file, mode='wb') as f:
            pickle.dump(vectors, f)
        logger.info(
            'Total passages processed %d. Written to %s, time elapsed %f sec.',
            len(vectors), file, time.time() - start_time
        )
        shard_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', required=True, nargs="+", default=["-"], help='Input files to encode')
    parser.add_argument('--out_file', required=True, type=str, help='Output file path to write results to')
    parser.add_argument('--context_type', default='abstract', choices=['abstract', 'keywords'])
    parser.add_argument('--shard_size', type=int, default=50000, help="Total amount of data in one shard")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--dataset', type=str, default=None, help=' to build correct dataset parser ')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)

    main(args)
