import torch
import torch.nn as nn
from torchcrf import CRF

from transformers.modeling_roberta import RobertaModel
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.use_crf = config.use_crf
        self.ignore_index = config.pad_token_label_id
        if self.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.use_crf:
                loss = -1 * self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs


class RobertaForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.roberta = RobertaModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.use_crf = config.use_crf
        self.ignore_index = config.pad_token_label_id
        if self.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.use_crf:
                loss = -1 * self.crf(logits, labels, mask=attention_mask.byte(), reduction='mean')
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs
