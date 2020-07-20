import torch
import torch.nn as nn
import torch.nn.functional as f

from deepkp.encoders import RNNEncoder, TransformerEncoder
from deepkp.decoders import RNNDecoder, TransformerDecoder
from deepkp.inputters import constants
from deepkp.modules import GlobalAttention, CopyGenerator, CopyGeneratorCriterion


class Embedder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Embedder, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, args.emsize, constants.PAD)
        self.position_embeddings = None
        if args.model_type == 'transformer' and args.absolute_pos:
            max_position = max(args.max_src_len, args.max_tgt_len) + 2
            self.position_embeddings = nn.Embedding(max_position, args.emsize)
        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self, sequence, sequence_position=None, step=None):
        word_rep = self.word_embeddings(sequence)  # B x P x d
        if self.position_embeddings is not None:
            if sequence_position is None:
                sequence_position = torch.tensor([step]) if step \
                    else torch.arange(start=0, end=sequence.size(1))
                sequence_position = sequence_position.to(sequence)
            position_rep = self.position_embeddings(sequence_position)
            word_rep = word_rep + position_rep
        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()

        if args.model_type == 'rnn':
            self.encoder = RNNEncoder(args.rnn_type,
                                      input_size,
                                      args.bidirection,
                                      args.enc_layers,
                                      args.nhid,
                                      args.dropout_rnn,
                                      use_last=False)
            self.enc_width = args.nhid

        elif args.model_type == 'transformer':
            self.encoder = TransformerEncoder(num_layers=args.enc_layers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
            self.enc_width = input_size

        else:
            raise NotImplementedError

        self.encoder_type = args.model_type
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(self.enc_width, 1, bias=False)

    def count_parameters(self):
        return self.encoder.count_parameters()

    def forward(self, input, input_len):
        out_dict = dict()

        if self.encoder_type == 'rnn':
            # M: batch_size x seq_len x nhid*nlayers
            hidden, M = self.encoder(input, input_len)
            # M: batch_size x seq_len x nhid
            layer_outputs = M.split(self.enc_width, dim=2)
            out_dict['hidden'] = hidden
        elif self.encoder_type == 'transformer':
            layer_outputs, _ = self.encoder(input, input_len)  # B x seq_len x h

        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # batch_size x seq_len x nlayers x nhid
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3), layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]

        out_dict['memory_bank'] = memory_bank
        out_dict['layer_outputs'] = layer_outputs

        return out_dict


class Decoder(nn.Module):
    def __init__(self, args, input_size):
        super(Decoder, self).__init__()

        if args.model_type == 'rnn':
            self.decoder = RNNDecoder(
                args.rnn_type,
                input_size,
                args.bidirection,
                args.dec_layers,
                args.nhid,
                attn_type=args.attn_type,
                copy_attn=args.copy_attn,
                reuse_copy_attn=args.reuse_copy_attn,
                coverage_attn=args.coverage_attn,
                dropout=args.dropout_rnn
            )
            assert args.enc_layers == args.dec_layers

        elif args.model_type == 'transformer':
            self.decoder = TransformerDecoder(
                num_layers=args.dec_layers,
                d_model=input_size,
                heads=args.num_head,
                d_k=args.d_k,
                d_v=args.d_v,
                d_ff=args.d_ff,
                coverage_attn=args.coverage_attn,
                dropout=args.trans_drop
            )
            self.layer_wise_attn = args.layer_wise_attn
            if self.layer_wise_attn:
                assert args.enc_layers == args.dec_layers

        else:
            raise NotImplementedError

        self.decoder_type = args.model_type

    def count_parameters(self):
        return self.decoder.count_parameters()

    def init_decoder(self, **kwargs):
        if self.decoder_type == 'rnn':
            return self.decoder.init_decoder_state(kwargs['hidden'])
        elif self.decoder_type == 'transformer':
            max_mem_len = kwargs['memory_bank'][0].shape[1] \
                if isinstance(kwargs['memory_bank'], list) else kwargs['memory_bank'].shape[1]
            return self.decoder.init_state(kwargs['memory_len'], max_mem_len)

    def decode(self, **kwargs):
        out_dict = dict()

        if self.decoder_type == 'rnn':
            decoder_outputs, _, attns = self.decoder(kwargs['input'],
                                                     kwargs['memory_bank'],
                                                     kwargs['state'],
                                                     kwargs['memory_len'])

        elif self.decoder_type == 'transformer':
            memory_bank = kwargs['layer_outputs'] if self.layer_wise_attn else kwargs['memory_bank']
            decoder_outputs, head_attns = self.decoder(kwargs['input_mask'],
                                                       kwargs['input'],
                                                       memory_bank,
                                                       kwargs['state'],
                                                       step=kwargs.get('step'),
                                                       layer_wise_coverage=kwargs.get('layer_wise_coverage'))
            # std_attn: batch_size x num_heads x tgt_len x src_len
            attns = dict()
            attns["std"] = torch.stack(head_attns["std"], dim=1)

        out_dict['decoder_outputs'] = decoder_outputs
        out_dict['attns'] = attns

        return out_dict

    def forward(self, **kwargs):
        if self.training:
            init_decoder_state = self.init_decoder(**kwargs)
            return self.decode(state=init_decoder_state, **kwargs)
        else:
            raise NotImplementedError


class LMDecoder(nn.Module):
    def __init__(self, args, input_size, dictionary):
        super(LMDecoder, self).__init__()

        self.generator = nn.Linear(input_size, len(dictionary))
        self.use_copy = args.copy_attn
        if self.use_copy:
            self.copy_attn = GlobalAttention(input_size, args.attn_type)
            self.generator = CopyGenerator(input_size, dictionary, self.generator)
            self.criterion = CopyGeneratorCriterion(len(dictionary), args.force_copy)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='none')

    def tie_weights(self, module):
        if isinstance(self.generator, CopyGenerator):
            self.generator.linear.weight = module.weight
        else:
            self.generator.weight = module.weight

    def train_forward(self, decoder_outputs, target, **kwargs):
        if self.use_copy:
            src_map = kwargs['src_map']
            alignment = kwargs['alignment']
            attn_copy = kwargs.get('attns')
            if attn_copy is None:
                _, attn_copy, _ = self.copy_attn(decoder_outputs,
                                                 kwargs['memory_bank'],
                                                 memory_lengths=kwargs['memory_len'],
                                                 softmax_weights=False)
            attn_copy = f.softmax(attn_copy, dim=-1)
            scores = self.generator(decoder_outputs, attn_copy, src_map)
            scores = scores[:, :-1, :].contiguous()
            alignment = alignment[:, :-1].contiguous()
            ml_loss = self.criterion(scores, alignment, target)
        else:
            scores = self.generator(decoder_outputs)  # `batch x tgt_len x vocab_size`
            scores = scores[:, :-1, :].contiguous()  # `batch x tgt_len - 1 x vocab_size`
            ml_loss = self.criterion(scores.view(-1, scores.size(2)), target.view(-1))

        return ml_loss

    def dev_forward(self, decoder_outputs, **kwargs):
        device = decoder_outputs.device
        if self.use_copy:
            src_map = kwargs['src_map']
            attn_copy = kwargs.get('attns')
            if attn_copy is None:
                _, attn_copy, _ = self.copy_attn(decoder_outputs,
                                                 kwargs['memory_bank'],
                                                 memory_lengths=kwargs['memory_len'],
                                                 softmax_weights=False)
            attn_copy = f.softmax(attn_copy, dim=-1)
            scores = self.generator(decoder_outputs, attn_copy, src_map)
            scores = scores.squeeze(1)
            for b in range(scores.size(0)):
                if kwargs['blank'][b]:
                    blank_b = torch.tensor(kwargs['blank'][b], dtype=torch.long, device=device)
                    fill_b = torch.tensor(kwargs['fill'][b], dtype=torch.long, device=device)
                    scores[b].index_add_(0, fill_b, scores[b].index_select(0, blank_b))
                    scores[b].index_fill_(0, blank_b, 1e-10)
        else:
            scores = self.generator(decoder_outputs)
            scores = f.softmax(scores.squeeze(1), dim=-1)

        return scores

    def forward(self, decoder_outputs, target, **kwargs):
        if self.training:
            return self.train_forward(
                decoder_outputs, target, **kwargs
            )
        else:
            return self.dev_forward(
                decoder_outputs, **kwargs
            )
