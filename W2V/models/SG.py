# -*- coding: utf-8 -*-
import torch
from torch import nn


class SkipGram(nn.Module):

    def __init__(self, vocab_size, projection_dim):
        super(SkipGram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) # 中心词
        self.embedding_u = nn.Embedding(vocab_size, projection_dim) # 周围词

        self.embedding_v.weight.data.uniform_(-1, 1)  # init
        self.embedding_u.weight.data.uniform_(0, 0)  # init

    def forward(self, center_words, target_words, outer_words):
        batch_size = center_words.size(0)
        if outer_words.size(0) != batch_size:
            outer_words = outer_words.expand((batch_size, len(outer_words)))
        center_embeds = self.embedding_v(center_words)  # B x 1 x D
        target_embeds = self.embedding_u(target_words)  # B x 1 x D
        outer_embeds = self.embedding_u(outer_words)  # B x V x D
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # BxVxD * BxDx1 => BxV
        nll = -torch.mean(
                torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))  # log-softmax

        return nll  # negative log likelihood

    def predict(self, word):
        emb = self.embedding_v(word)
        return emb
