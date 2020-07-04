# -*- coding: utf-8 -*-
"""结合W2V和SVD的有点，考虑全局且并不是“非常”偏向高频词"""
import torch
import torch.nn as nn


class GloVe(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(GloVe, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)  # 中心词
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)  # 周围词

        self.embedding_v_bias = nn.Embedding(vocab_size, 1)
        self.embedding_u_bias = nn.Embedding(vocab_size, 1)
        nn.init.xavier_uniform_(self.embedding_v.weight)
        nn.init.xavier_uniform_(self.embedding_u.weight)
        nn.init.xavier_uniform_(self.embedding_v_bias.weight)
        nn.init.xavier_uniform_(self.embedding_u_bias.weight)

    def forward(self, center_words, target_words, coocs, weights):
        # coocs -> log(Xij) log(词对ij的个数)

        center_emb = self.embedding_v(center_words)
        target_emb = self.embedding_u(target_words)

        center_b = self.embedding_v_bias(center_words)
        target_b = self.embedding_u_bias(target_words)
        left = target_emb.bmm(center_emb.transpose(1, 2)).squeeze(2)

        loss = weights * torch.pow(left + center_b + target_b - coocs, 2)
        return torch.sum(loss)

    def predict(self, word):
        emb = self.embedding_v(word)
        return emb
