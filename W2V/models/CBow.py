# -*- coding: utf-8 -*-
from torch import nn
import torch


class CBow(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(CBow, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)  # 中心词
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)  # 周围词

        self.embedding_v.weight.data.uniform_(-1, 1)  # init
        self.embedding_u.weight.data.uniform_(0, 0)  # init

    def forward(self, round_words, target_words, outer_words):
        batch_size = round_words.size(0)
        outer_words = outer_words.expand((batch_size, len(outer_words)))
        round_emb = self.embedding_u(round_words)  # B*len*h
        target_emb = self.embedding_v(target_words)  # B*1*h
        outer_emd = self.embedding_u(outer_words)  # B x V x D
        h = torch.mean(round_emb, dim=1).unsqueeze(1)  # B*1*h

        left = -target_emb.bmm(h.transpose(1, 2)).squeeze(2)  # B*1
        right = outer_emd.bmm(target_emb.transpose(1, 2)).squeeze(2)  # B*V
        right = torch.log(torch.sum(torch.exp(right), dim=1).unsqueeze(1))  # B*1
        return right - left

    def predict(self, word):
        emb = self.embedding_v(word)
        return emb


# c = CBow(20, 20)
# outer_words = torch.tensor(list(range(20)))
# r_w = torch.tensor([[2, 3, 15, 7, 8, 11], [12, 13, 5, 17, 18, 1]])
# t_w = torch.tensor([[9], [6]])
# o = c(r_w, t_w, outer_words)
# print(o.size())
# print(o)
