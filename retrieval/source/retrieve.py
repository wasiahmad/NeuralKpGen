#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from retrieval.dpr.data.qa_validation import calculate_matches_by_id
from retrieval.dpr.models import init_biencoder_components
from retrieval.dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from retrieval.dpr.utils.data_utils import Tensorizer
from retrieval.dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, \
    move_to_device
from retrieval.dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
            self,
            question_encoder: nn.Module,
            batch_size: int,
            tensorizer: Tensorizer,
            index: DenseIndexer
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):
                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q, type='question')
                    for q in questions[batch_start:batch_start + bsz]
                ]
                q_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), args.device)
                q_seg_batch = move_to_device(torch.zeros_like(q_ids_batch), args.device)
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)
                query_vectors.extend(out.cpu().split(1, dim=0))
                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)
        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
            self,
            query_vectors: np.array,
            top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_jsonl_file(location, pred_loc, keyword_type) -> Iterator[Tuple[str, List[str]]]:
    predictions = []
    if pred_loc is not None:
        with open(pred_loc) as jsonlfile:
            for row in jsonlfile:
                predictions.append(json.loads(row))

    with open(location) as jsonlfile:
        for idx, row in enumerate(jsonlfile):
            ex = json.loads(row)
            if keyword_type == 'query':
                question = ex['query']
                answers = ex['rel_docs']
            else:
                if keyword_type == 'all':
                    keywords = ex["present"] + ex["absent"]
                    if len(keywords) == 0:
                        continue
                    if predictions:
                        keywords = predictions[idx]["present"] + predictions[idx]["absent"]
                else:
                    keywords = ex[keyword_type]
                    if len(keywords) == 0:
                        continue
                    if predictions:
                        keywords = predictions[idx][keyword_type]
                question = ' ; '.join(keywords)
                answers = [ex['id']]

            yield question, answers


def validate(
        answers: List[List[str]],
        result_ctx_ids: List[Tuple[List[object], List[float]]],
        workers_num: int,
        match_type: str
) -> List[List[bool]]:
    match_stats = calculate_matches_by_id(answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
        questions: List[str],
        answers: List[List[str]],
        top_passages_and_scores: List[Tuple[List[object], List[float]]],
        per_question_hits: List[List[bool]],
        out_file: str
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
    encoder = encoder.question_model

    # the following doesn't work for fp16
    # encoder, _ = setup_for_distributed_mode(
    #     encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
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

    prefix_len = len('question_model.')
    question_encoder_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items()
        if key.startswith('question_model.')
    }
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    input_paths = args.encoded_ctx_file
    index_path = "_".join(input_paths[0].split("_")[:-1])
    if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index.index_data(input_paths)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)

    # get questions & answers
    questions = []
    question_answers = []

    for ds_item in parse_jsonl_file(args.qa_file, args.pred_file, args.keyword):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.float().numpy(), args.n_docs)

    questions_doc_hits = validate(
        question_answers, top_ids_and_scores, args.validation_workers, args.match
    )

    if args.out_file:
        save_results(
            questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument("--keyword", type=str, help="Type of the keywords", default="present",
                        choices=["present", "absent", "all", "query"])
    parser.add_argument('--qa_file', required=True, type=str,
                        help="Question and answers file in JSON format")
    parser.add_argument('--pred_file', default=None, type=str,
                        help="Question and answers (predicted) file in JSON format")
    parser.add_argument('--encoded_ctx_file', nargs='+', default='[-]',
                        help='Files of encode passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'exact'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=100, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

    args = parser.parse_args()

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
