import torch
import torch.nn as nn
import torch.nn.functional as f

from prettytable import PrettyTable
from deepkp.models.util import Embedder, Encoder, Decoder, LMDecoder
from deepkp.inputters import constants


class Sequence2Sequence(nn.Module):

    def __init__(self, args, vocabulary):
        super(Sequence2Sequence, self).__init__()

        self.embedder = Embedder(args, len(vocabulary))
        self.encoder = Encoder(args, args.emsize)
        self.decoder = Decoder(args, args.emsize)
        output_size = args.nhid if args.model_type == 'rnn' else args.emsize
        self.generator = LMDecoder(args, output_size, vocabulary)

        if args.share_decoder_embeddings:
            assert args.emsize == output_size
            self.generator.tie_weights(self.embedder.word_embeddings)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def count_decoder_parameters(self):
        return self.decoder.count_parameters()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

    def encode(self, source, source_len, source_position=None, **kwargs):
        word_rep = self.embedder(source, source_position)
        return self.encoder(word_rep, source_len)

    def decode(self, target, target_len, **kwargs):
        if self.training:
            word_emb = self.embedder(target)
            input_mask = target.eq(constants.PAD)
            out_dict = self.decoder(input=word_emb,
                                    hidden=kwargs['hidden'],
                                    memory_bank=kwargs['memory_bank'],
                                    layer_outputs=kwargs['layer_outputs'],
                                    memory_len=kwargs['memory_len'],
                                    input_mask=input_mask,
                                    target_len=target_len)

            decoder_outputs = out_dict['decoder_outputs']
            attns = out_dict['attns']
            if isinstance(decoder_outputs, list):  # for transformer
                decoder_outputs = decoder_outputs[-1]

            t = target[:, 1:].contiguous()
            loss = self.generator(decoder_outputs, t,
                                  memory_bank=kwargs['memory_bank'],
                                  memory_len=kwargs['memory_len'],
                                  attns=attns.get('copy'),
                                  src_map=kwargs.get('src_map'),
                                  alignment=kwargs.get('alignment'))
            loss = loss.view(*t.size())
            loss = loss.mul(t.ne(constants.PAD).float())
            return loss
        else:
            return self.generate(**kwargs)

    def generate(self, **kwargs):
        hidden = kwargs.get('hidden')
        memory_bank = kwargs.get('memory_bank')
        memory_len = kwargs.get('memory_len')
        batch_size = memory_bank.size(0)

        init_decoder_state = self.decoder.init_decoder(hidden=hidden,
                                                       memory_bank=memory_bank,
                                                       memory_len=memory_len)

        tgt_words = kwargs.get('tgt_first_word')
        if tgt_words is None:
            tgt_words = torch.empty(batch_size, 1).fill_(constants.BOS)
            tgt_words = tgt_words.to(memory_len)

        dec_preds = []
        copy_info = []
        attentions = []
        dec_log_probs = []

        attns = {"coverage": None}
        for idx in range(kwargs['max_len'] + 1):
            tgt = self.embedder(tgt_words, step=idx)
            input_mask = tgt_words.eq(constants.PAD)
            out_dict = self.decoder.decode(input=tgt,
                                           state=init_decoder_state,
                                           memory_bank=memory_bank,
                                           memory_len=memory_len,
                                           input_mask=input_mask,
                                           layer_outputs=kwargs['layer_outputs'],
                                           step=idx,
                                           layer_wise_coverage=attns.get('coverage'))

            decoder_outputs = out_dict['decoder_outputs']
            attns = out_dict['attns']
            if isinstance(decoder_outputs, list):  # for transformer
                decoder_outputs = decoder_outputs[-1]

            prediction = self.generator(decoder_outputs, None,
                                        attns=attns.get('copy'),
                                        **kwargs)

            tgt_prob, tgt = torch.max(prediction, dim=1, keepdim=True)
            log_prob = torch.log(tgt_prob + 1e-20)
            dec_log_probs.append(log_prob.squeeze(1))
            dec_preds.append(tgt.squeeze(1).clone())

            if "std" in attns:
                std_attn = f.softmax(attns["std"], dim=-1)
                if std_attn.dim() == 3:
                    attentions.append(std_attn.squeeze(1))
                else:  # transformer
                    attentions.append(std_attn.squeeze(2))
            if "copy" in attns:
                mask = tgt.gt(len(kwargs['tgt_dict']) - 1)
                copy_info.append(mask.squeeze(1))

            words = self.__tens2sent(tgt, kwargs['tgt_dict'], kwargs['source_vocab'])

            words = [kwargs['tgt_dict'][w] for w in words]
            words = torch.tensor(words).to(tgt)
            tgt_words = words.unsqueeze(1)

        dec_preds = torch.stack(dec_preds, dim=1)
        dec_log_probs = torch.stack(dec_log_probs, dim=1)
        copy_info = torch.stack(copy_info, dim=1) if copy_info else None
        attentions = torch.stack(attentions, dim=1) if attentions else None

        return {
            'predictions': dec_preds,
            'copy_info': copy_info,
            'attentions': attentions,
            'dec_log_probs': dec_log_probs
        }

    def __tens2sent(self, t, tgt_dict, src_vocabs):
        words = []
        for idx, w in enumerate(t):
            widx = w[0].item()
            if widx < len(tgt_dict):
                words.append(tgt_dict[widx])
            else:
                widx = widx - len(tgt_dict)
                words.append(src_vocabs[idx][widx])
        return words

    def forward(self, source, source_len, target, target_len, **kwargs):
        if self.training:
            encode_dict = self.encode(source, source_len, **kwargs)
            target_loss = self.decode(target, target_len,
                                      hidden=encode_dict.get('hidden'),
                                      memory_bank=encode_dict.get('memory_bank'),
                                      layer_outputs=encode_dict.get('layer_outputs'),
                                      memory_len=source_len,
                                      **kwargs)

            loss = target_loss.mean()
            loss_per_token = target_loss.div((target_len - 1).float().unsqueeze(1)).mean()

            return loss, loss_per_token

        else:
            encode_dict = self.encode(source, source_len, **kwargs)
            return self.decode(None, None,
                               hidden=encode_dict.get('hidden'),
                               memory_bank=encode_dict.get('memory_bank'),
                               layer_outputs=encode_dict.get('layer_outputs'),
                               memory_len=source_len,
                               **kwargs)
