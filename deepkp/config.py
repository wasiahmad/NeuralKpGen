""" Implementation of all available options """
from __future__ import print_function

"""Model architecture/optimization options for Seq2seq architecture."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type',
    'emsize',
    'rnn_type',
    'nhid',
    'enc_layers',
    'dec_layers',
    'use_all_enc_layers',
    'bidirection',
    'absolute_pos',
    'max_relative_pos',
    'use_neg_dist',
    'd_ff',
    'd_k',
    'd_v',
    'num_head',
    'trans_drop',
}

SEQ2SEQ_ARCHITECTURE = {
    'attn_type',
    'coverage_attn',
    'copy_attn',
    'force_copy',
    'layer_wise_attn',
    'reuse_copy_attn',
    'share_decoder_embeddings'
}

DATA_OPTIONS = {
    'max_src_len',
    'max_tgt_len',
    'vocab_size',
    'num_train_examples',
    'batch_size',
    'uncase'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'warmup_steps',
    'learning_rate',
    'momentum',
    'weight_decay',
    'rnn_padding',
    'dropout_rnn',
    'dropout',
    'dropout_emb',
    'cuda',
    'grad_clipping',
    'lr_decay',
    'num_epochs',
    'fp16',
    'fp16_opt_level',
    'n_gpu',
    'device'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Data options
    data = parser.add_argument_group('Data parameters')
    data.add_argument('--max_src_len', type=int, default=100,
                      help='Maximum allowed length for the source sequence')
    data.add_argument('--max_tgt_len', type=int, default=50,
                      help='Maximum allowed length for the target sequence')
    data.add_argument('--max_keywords', type=int, default=10,
                      help='Maximum keywords to be considered during training')

    # Model architecture
    model = parser.add_argument_group('Summary Generator')
    model.add_argument('--model_type', type=str, default='rnn',
                       choices=['rnn', 'transformer'],
                       help='Model architecture type')
    model.add_argument('--emsize', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--rnn_type', type=str, default='LSTM',
                       help='RNN type: LSTM, GRU')
    model.add_argument('--nhid', type=int, default=200,
                       help='Hidden size of RNN units')
    model.add_argument('--bidirection', type='bool', default=True,
                       help='use bidirectional recurrent unit')
    model.add_argument('--enc_layers', type=int, default=1,
                       help='Number of encoder layers')
    model.add_argument('--dec_layers', type=int, default=1,
                       help='Number of decoder layers')
    model.add_argument('--use_all_enc_layers', type='bool', default=False,
                       help='Use a weighted average of all encoder layers\' '
                            'representation as the contextual representation')

    # Transformer specific params
    model.add_argument('--absolute_pos', type='bool', default=False,
                       help='Use positional embeddings in encoder')
    model.add_argument('--max_relative_pos', nargs='+', type=int,
                       default=0, help='Max value for relative position representations')
    model.add_argument('--use_neg_dist', type='bool', default=True,
                       help='Use negative Max value for relative position representations')
    model.add_argument('--d_ff', type=int, default=2048,
                       help='Number of units in position-wise FFNN')
    model.add_argument('--d_k', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--d_v', type=int, default=64,
                       help='Hidden size of heads in multi-head attention')
    model.add_argument('--num_head', type=int, default=8,
                       help='Number of heads in Multi-Head Attention')
    model.add_argument('--trans_drop', type=float, default=0.2,
                       help='Dropout for transformer')
    model.add_argument('--layer_wise_attn', type='bool', default=False,
                       help='Use layer-wise attention in Transformer')

    seq2seq = parser.add_argument_group('Seq2seq Model Specific Params')
    seq2seq.add_argument('--attn_type', type=str, default='general',
                         help='Attention type for the seq2seq [dot, general, mlp]')
    seq2seq.add_argument('--coverage_attn', type='bool', default=False,
                         help='Use coverage attention')
    seq2seq.add_argument('--copy_attn', type='bool', default=False,
                         help='Use copy attention')
    seq2seq.add_argument('--force_copy', type='bool', default=False,
                         help='Apply force copying')
    seq2seq.add_argument('--reuse_copy_attn', type='bool', default=False,
                         help='Reuse encoder attention')
    seq2seq.add_argument('--share_decoder_embeddings', type='bool', default=False,
                         help='Share decoder embeddings weight with softmax layer')

    # Optimization details
    optim = parser.add_argument_group('Seq2seq model Optimization')
    optim.add_argument('--dropout_emb', type=float, default=0.2,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout_rnn', type=float, default=0.2,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout for NN layers')
    optim.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=0.99,
                        help='Decay ratio for learning rate')
    optim.add_argument('--grad_clipping', type=float, default=5.0,
                       help='Gradient clipping')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='Stop training if performance doesn\'t improve')
    optim.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of steps for gradient accumulation')
    optim.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of of warmup steps')


def get_model_args(args):
    """Filter args for model ones.
    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER, SEQ2SEQ_ARCHITECTURE, DATA_OPTIONS
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER | SEQ2SEQ_ARCHITECTURE | DATA_OPTIONS

    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))

    return argparse.Namespace(**old_args)


def add_new_model_args(old_args, new_args):
    """Set args to new parameters.
    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.
    We keep the new optimization or RL setting, and leave the model architecture alone.
    """
    global ADVANCED_OPTIONS
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k not in old_args:
            if k in ADVANCED_OPTIONS:
                logger.info('Adding arg %s: %s' % (k, new_args[k]))
                old_args[k] = new_args[k]

    return argparse.Namespace(**old_args)
