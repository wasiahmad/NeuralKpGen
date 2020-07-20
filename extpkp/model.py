import copy
import math
import logging
import numpy as np
import torch

from torch.nn.utils import clip_grad_norm_
from deepkp.config import override_model_args
from deepkp.models.seq2seq import Sequence2Sequence
from deepkp.utils.copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from deepkp.utils.misc import tens2sen_score
from deepkp.inputters import constants
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class Seq2seqKpGen(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.updates = 0

        self.network = Sequence2Sequence(self.args, self.word_dict)
        if state_dict:
            self.network.load_state_dict(state_dict)

    def activate_fp16(self):
        if not hasattr(self, 'optimizer'):
            self.network.half()  # for testing only
            return
        try:
            global amp
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        # https://github.com/NVIDIA/apex/issues/227
        assert self.optimizer is not None
        self.network, self.optimizer = amp.initialize(self.network,
                                                      self.optimizer,
                                                      opt_level=self.args.fp16_opt_level)

    def init_optimizer(self, optim_state=None, sched_state=None):
        def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1.0, num_warmup_steps))
                return 1.0

            return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.network.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.network.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, self.args.warmup_steps)

        if optim_state:
            self.optimizer.load_state_dict(optim_state)
        if sched_state:
            self.scheduler.load_state_dict(sched_state)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        source_map, alignment = None, None
        if self.args.copy_attn:
            source_map = make_src_map(ex['src_map']).to(self.args.device)
            alignment = align(ex['alignment']).to(self.args.device)

        source_rep = ex['source_rep'].to(self.args.device)
        source_len = ex['source_len'].to(self.args.device)
        target_rep = ex['target_rep'].to(self.args.device)
        target_len = ex['target_len'].to(self.args.device)

        # Run forward
        ml_loss, loss_per_token = self.network(source=source_rep,
                                               source_len=source_len,
                                               target=target_rep,
                                               target_len=target_len,
                                               src_map=source_map,
                                               alignment=alignment)

        loss = ml_loss.mean() if self.args.n_gpu > 1 else ml_loss
        if self.args.fp16:
            global amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args.grad_clipping)
        else:
            loss.backward()
            clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)

        self.updates += 1
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule
        self.optimizer.zero_grad()

        loss_per_token = loss_per_token.mean() if self.args.n_gpu > 1 else loss_per_token
        loss_per_token = loss_per_token.item()
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)

        return {
            'ml_loss': loss.item(),
            'perplexity': perplexity
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, replace_unk=False):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
            src_raw: raw source (passage); required to replace `unk` term
        Output:
            predictions: #batch predicted sequences
        """

        def convert_text_to_string(text):
            """ Converts a sequence of tokens (string) in a single string. """
            out_string = text.replace(" ##", "").strip()
            return out_string

        self.network.eval()

        source_map, alignment = None, None
        blank, fill = None, None
        if self.args.copy_attn:
            source_map = make_src_map(ex['src_map']).to(self.args.device)
            alignment = align(ex['alignment']).to(self.args.device)
            blank, fill = collapse_copy_scores(self.word_dict, ex['src_vocab'])

        source_rep = ex['source_rep'].to(self.args.device)
        source_len = ex['source_len'].to(self.args.device)

        decoder_out = self.network(source=source_rep,
                                   source_len=source_len,
                                   target=None,
                                   target_len=None,
                                   src_map=source_map,
                                   alignment=alignment,
                                   max_len=self.args.max_tgt_len,
                                   tgt_dict=self.word_dict,
                                   blank=blank, fill=fill,
                                   source_vocab=ex['src_vocab'])

        dec_probs = torch.exp(decoder_out['dec_log_probs'])
        predictions, scores = tens2sen_score(decoder_out['predictions'], dec_probs,
                                             self.word_dict, ex['src_vocab'])
        if replace_unk:
            for i in range(len(predictions)):
                enc_dec_attn = decoder_out['attentions'][i]
                if self.args.model_type == 'transformer':
                    # tgt_len x num_heads x src_len
                    assert enc_dec_attn.dim() == 3
                    enc_dec_attn = enc_dec_attn.mean(1)
                predictions[i] = replace_unknown(predictions[i], enc_dec_attn,
                                                 src_raw=ex['source'][i].tokens)

        for bidx in range(ex['batch_size']):
            for i in range(len(predictions[bidx])):
                if predictions[bidx][i] == constants.KP_SEP:
                    scores[bidx][i] = constants.KP_SEP
                elif predictions[bidx][i] == constants.PRESENT_EOS:
                    scores[bidx][i] = constants.PRESENT_EOS
                else:
                    assert isinstance(scores[bidx][i], float)
                    scores[bidx][i] = str(scores[bidx][i])

        predictions = [' '.join(item) for item in predictions]
        scores = [' '.join(item) for item in scores]

        present_kps = []
        absent_kps = []
        present_kp_scores = []
        absent_kp_scores = []
        for bidx in range(ex['batch_size']):
            keyphrases = predictions[bidx].split(constants.PRESENT_EOS)
            kp_scores = scores[bidx].split(constants.PRESENT_EOS)
            pkps = (' %s ' % constants.KP_SEP).join(keyphrases[:-1])
            pkp_scores = (' %s ' % constants.KP_SEP).join(kp_scores[:-1])
            akps = keyphrases[-1]
            akp_scores = kp_scores[-1]

            pre_kps = []
            pre_kp_scores = []
            for pkp, pkp_s in zip(pkps.split(constants.KP_SEP),
                                  pkp_scores.split(constants.KP_SEP)):
                pkp = pkp.strip()
                if pkp:
                    pre_kps.append(convert_text_to_string(pkp))
                    t_scores = [float(i) for i in pkp_s.strip().split()]
                    _score = np.prod(t_scores) / len(t_scores)
                    pre_kp_scores.append(_score)

            present_kps.append(pre_kps)
            present_kp_scores.append(pre_kp_scores)

            abs_kps = []
            abs_kp_scores = []
            for akp, akp_s in zip(akps.split(constants.KP_SEP),
                                  akp_scores.split(constants.KP_SEP)):
                akp = akp.strip()
                if akp:
                    abs_kps.append(convert_text_to_string(akp))
                    t_scores = [float(i) for i in akp_s.strip().split()]
                    _score = np.prod(t_scores) / len(t_scores)
                    abs_kp_scores.append(_score)

            absent_kps.append(abs_kps)
            absent_kp_scores.append(abs_kp_scores)

        return {
            'present_kps': present_kps,
            'absent_kps': absent_kps,
            'present_kp_scores': present_kp_scores,
            'absent_kp_scores': absent_kp_scores
        }

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        network = self.network.module if hasattr(self.network, "module") \
            else self.network
        state_dict = copy.copy(network.state_dict())
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        network = self.network.module if hasattr(self.network, "module") \
            else self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optim_dict': self.optimizer.state_dict(),
            'sched_dict': self.scheduler.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Seq2seqKpGen(args, word_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optim_dict = saved_params['optim_dict']
        sched_dict = saved_params['sched_dict']
        args = saved_params['args']
        model = Seq2seqKpGen(args, word_dict, state_dict)
        model.updates = updates
        model.init_optimizer(optim_dict, sched_dict)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def to(self, device):
        self.network = self.network.to(device)

    def parallelize(self):
        self.network = torch.nn.DataParallel(self.network)
