# coding:utf-8

'''
大体的已经搭建了，剩下的自己补足就可以了
'''

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Model(nn.Module):
    def __init__(self, pretrain_model_path, hidden_size):
        super(Model, self).__init__()
        self.pretrain_model_path = pretrain_model_path
        self.bert = BertModel.from_pretrained(self.pretrain_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        self.embed_size = hidden_size
        self.cls = nn.Linear(self.embed_size, 2)

    def forward(self, ids, label, segment):
        context = ids
        types = segment
        mask = torch.ne(context, 0)
        sequence_out, cls_out = self.bert(context, token_type_ids=types, attention_mask=mask, output_all_encoded_layers=False)
        cls_out = self.dropout(cls_out)
        logits = self.cls(cls_out)
        loss = nn.CrossEntropyLoss()(nn.LogSoftmax(dim=-1)(logits), label.view(-1))
        return loss, logits


def read_dataset(path, pretrain_model_path, is_test=False):
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if is_test:
                text_a, text_b = line.split('\t')
            else:
                text_a, text_b, tgt = line.split('\t')
                tgt = int(tgt)
            src_a = tokenizer.convert_tokens_to_ids([CLS_TOKEN] + tokenizer.tokenize(text_a) + [SEP_TOKEN])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_b) + [SEP_TOKEN])
            src = src_a + src_b
            seg = [0] * len(src_a) + [1] * len(src_b)
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
            if is_test:
                dataset.append((src, seg))
            else:
                dataset.append((src, tgt, seg))
    return dataset


pretrain_model_path = 'pretrain/model'
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
'''
可以自行修改params
'''
[CLS_TOKEN] = '[CLS]'
[SEP_TOKEN] = '[SEP]'
seq_length = 32