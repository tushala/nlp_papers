# -*- coding: utf-8 -*-
import torch
from torch import nn
from utils import *


class HS(nn.Module):
    def __init__(self, vocab_size, emb_dim, tensor_d):
        super(HS, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_dim)  # 中心词
        self.embedding_u = nn.Embedding(vocab_size, emb_dim)  # 周围词
        self.tensor_d = tensor_d
        nn.init.xavier_uniform_(self.embedding_v.weight)
        nn.init.xavier_uniform_(self.embedding_u.weight)

    def forward(self, input, target, vocabs):
        input_emb = self.embedding_u(input)  # B*len*h
        target_emb = self.embedding_v(target)
        target_codes = [self.tensor_d[t.item()] for t in target]  # B*1*h

        # if input.size(1) > target.size(1): # CBOW
        #     length = input.size(1)
        input_emb = torch.mean(input_emb, dim=1).unsqueeze(1)
        sig_vector = torch.sigmoid(input_emb.bmm(target_emb.transpose(1, 2))).squeeze(1)
        loss = sum([calc_word_p(target_code, sig_vector) for target_code in target_codes])
        return torch.tensor(loss, dtype=torch.float32, requires_grad=True)
