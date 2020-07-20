import sys

sys.path.append(".")
sys.path.append("..")

import os
import configargparse
import torch
import logging
import subprocess
import argparse
import numpy as np

import deepkp.config as config
import deepkp.inputters.utils as util
from deepkp.inputters import Vocabulary

from tqdm import tqdm
from deepkp.utils.timer import AverageMeter, Timer
from seq2seq.model import Seq2seqKpGen
from deepkp.eval.kpeval import eval_accuracies, log_result, run_eval

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--fp16', type='bool', default=False,
                         help="Whether to use 16-bit float precision instead of 32-bit")
    runtime.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/',
                       help='Directory of training/validation data')
    files.add_argument('--train_file', type=str,
                       help='Preprocessed train source file')
    files.add_argument('--dev_file', type=str, required=True,
                       help='Preprocessed dev source file')
    files.add_argument('--vocab_file', type=str,
                       help='Preprocessed dev source file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--vocab_size', type=int, default=None,
                            help='Maximum allowed length for dictionary')
    preprocess.add_argument('--disable_extra_one_word_filter', type='bool', default=False,
                            help='Disable extra one word filtering')
    preprocess.add_argument('--no_source_output', type='bool', default=False,
                            help='Do not write Source text in the output log')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--config', is_config_file_arg=True,
                         help='config file path')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')
    general.add_argument('--k_list', nargs='+', default=[5, 'M'],
                         help='K values for evaluation')
    general.add_argument('--pred_file', type=str, default='',
                         help='Name of the output prediction file')
    general.add_argument('--log_file', type=str, default='',
                         help='Name of the output prediction file')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist

    if not args.only_test:
        args.train_file = os.path.join(args.data_dir, args.train_file)
        if not os.path.isfile(args.train_file):
            raise IOError('No such file: %s' % args.train_file)

    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)

    if not args.only_test:
        args.vocab_file = os.path.join(args.data_dir, args.vocab_file)
        if not os.path.isfile(args.vocab_file):
            raise IOError('No such file: %s' % args.vocab_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    if not args.log_file:
        args.log_file = args.model_name + suffix
    args.log_file = os.path.join(args.model_dir, args.log_file + '.txt')
    if not args.pred_file:
        args.pred_file = args.model_name + suffix
    args.pred_file = os.path.join(args.model_dir, args.pred_file + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    return args


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""

    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()
    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)
    pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' %
                         current_epoch)

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        net_loss = model.update(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)

        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    examples = 0
    present_absent_kps = []
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for ex in pbar:
            batch_size = ex['batch_size']
            predictions = model.predict(ex, replace_unk=True)
            for idx in range(batch_size):
                pkp_pred = list(zip(predictions['present_kps'][idx], predictions['present_kp_scores'][idx]))
                akp_pred = list(zip(predictions['absent_kps'][idx], predictions['absent_kp_scores'][idx]))
                pkp_gold = ex['target'][idx].present_keyphrases
                pkp_gold = [kp.text for kp in pkp_gold]
                akp_gold = ex['target'][idx].absent_keyphrases
                akp_gold = [kp.text for kp in akp_gold]
                present_absent_kps.append({
                    'id': ex['ids'][idx],
                    'present': {'pred': pkp_pred, 'gold': pkp_gold},
                    'absent': {'pred': akp_pred, 'gold': akp_gold}
                })

            examples += batch_size
            description = '[validating ... ]'
            if mode == 'dev':
                description = 'Epoch = {} '.format(global_stats['epoch']) + description
            pbar.set_description(description)

    result = eval_accuracies(present_absent_kps,
                             filename=args.pred_file,
                             disable_extra_one_word_filter=args.disable_extra_one_word_filter,
                             k_list=args.k_list)

    result = log_result(result, eval_time.time(), mode, examples)

    if args.only_test:
        # official evaluation
        logger.info('-' * 100)
        logger.info("Running full evaluation...")
        dirname = os.path.dirname(args.pred_file)
        result_file_suffix = os.path.splitext(os.path.basename(args.pred_file))[0]
        run_eval(present_absent_kps,
                 dirname, result_file_suffix,
                 args.disable_extra_one_word_filter, args.k_list)

    return result


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = Seq2seqKpGen.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = Seq2seqKpGen.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = Seq2seqKpGen.load_checkpoint(checkpoint_file)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = Seq2seqKpGen.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                word_dict = Vocabulary()
                word_dict.load(args.vocab_file, vocab_size=args.vocab_size)
                logger.info('Num words in vocabulary = %d' % (len(word_dict)))
                # Initialize model
                model = Seq2seqKpGen(config.get_model_args(args), word_dict)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            # table = model.network.layer_wise_parameters()
            # logger.info('Breakdown of the trainable paramters\n%s' % table)

    model.to(args.device)
    if args.n_gpu > 0 and args.fp16:
        model.activate_fp16()

    if args.n_gpu > 1:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Preparing data loaders')

    if not args.only_test:
        train_dataset = util.KeyphraseDataset(args.train_file, model,
                                              lazy_load=True,
                                              max_src_len=model.args.max_src_len,
                                              max_keywords=args.max_keywords,
                                              max_examples=args.max_examples)
        logger.info('Num train examples = %d' % len(train_dataset))
        args.num_train_examples = len(train_dataset)
        # train_sampler = util.SortedBatchSampler(train_dataset.lengths(),
        #                                         batch_size=args.batch_size,
        #                                         shuffle=True)
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=util.batchify,
            pin_memory=args.n_gpu > 0
        )
    dev_dataset = util.KeyphraseDataset(args.dev_file, model,
                                        lazy_load=True,
                                        max_src_len=model.args.max_src_len,
                                        max_keywords=args.max_keywords,
                                        max_examples=args.max_examples,
                                        test_split=True)
    if args.only_test:
        logger.info('Num test examples = %d' % len(dev_dataset))
    else:
        logger.info('Num dev examples = %d' % len(dev_dataset))
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.data_workers,
        collate_fn=util.batchify,
        pin_memory=args.n_gpu > 0
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    # logger.info('-' * 100)
    # logger.info('CONFIG:\n%s' % json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 1, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, dev_loader, model, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            train(args, train_loader, model, stats)
            result = validate_official(args, dev_loader, model, stats)

            # Save best valid
            if result[args.valid_metric] > stats['best_valid']:
                logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                             stats['epoch'], model.updates))
                model.save(args.model_file)
                stats['best_valid'] = result[args.valid_metric]
                stats['no_improvement'] = 0
            else:
                stats['no_improvement'] += 1
                if stats['no_improvement'] >= args.early_stop:
                    break


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = configargparse.ArgumentParser(
        'Neural Keyphrase Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
