# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from const import *


# 为了省事 就不区分enc_hidden 和dec_hidden

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.enc_emb = nn.Embedding(vocab_size, emb_dim)
        self.enc_gru = nn.GRU(emb_dim, hidden_size, batch_first=True)

    def forward(self, x):
        # hidden = self.init_hidden(input)
        embbed = self.enc_emb(x)
        out, hidden = self.enc_gru(embbed)
        return out, hidden

    # def _init_hidden(self, ):
    #     return torch.zeros()


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.max_length = max_length
        self.dec_emb = nn.Embedding(vocab_size, emb_dim)
        self.dec_gru = nn.GRU(emb_dim, hidden_size, 1, batch_first=True)
        self.dropout_p = nn.Dropout(0.1)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, decoder_inputs, hidden, encoder_outputs, teacher_force=True):
        """

        :param decoder_inputs: b * 1
        :param hidden:  b*L*h
        :param encoder_outputs: 1*b*h
        :return:
        """

        def init_hidden():
            return torch.zeros(hidden.size())

        loss = torch.tensor(0.)
        attns = []
        embedded = self.dec_emb(decoder_inputs)
        hidden = init_hidden()
        decode_input = embedded[:, 0, :]  # <start>
        for i in range(self.max_length - 1):
            attn = self.attn(torch.cat((decode_input, hidden[0]), 1))
            attn_weights = F.softmax(attn)
            attns.append(attn_weights.tolist())
            attn_applied = attn_weights.unsqueeze(1).bmm(encoder_outputs)
            concated = torch.cat((embedded[:, i, :].unsqueeze(1), attn_applied), 2)
            if not teacher_force:
                output = self.attn_combine(concated)
                output = F.relu(output)
            else:
                output = embedded[:, i, :].unsqueeze(1)
            output = self.dropout(output)
            output, hidden = self.dec_gru(output, hidden)
            output = output.squeeze(1)
            output = self.out(output)
            if teacher_force:
                decode_input = self.dec_emb(decoder_inputs[:, i])
            else:
                _, ids = torch.max(output, dim=1)
                decode_input = self.dec_emb(ids)

            loss += self.ce(output, decoder_inputs[:, i])
        return loss, attns


e = Encoder(5586, EMB_DIM, HIDDEN_SIZE)
x = torch.tensor([[0, 5133, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [0, 626, 5, 144, 5583, 4, 5584, 13, 4929, 5585, 4, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 626, 5, 144, 5583, 4, 5584, 13, 4929, 5585, 4, 1,
                                                         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.long)

y = torch.tensor([[0, 2345, 1, 6, 111, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [0, 727, 66, 1234, 602, 44, 558, 133, 929, 5585, 411, 1234,
                   2222, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 126, 511, 1244, 5283, 434, 5584, 1323, 499, 85, 455, 123,
                                                            2213, 2612, 432, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                 dtype=torch.long)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, max_length=MAX_LENGTH):
        super(Seq2Seq, self).__init__()
        self.enc = Encoder(vocab_size, emb_dim, hidden_size)
        self.dec = AttnDecoder(vocab_size, emb_dim, hidden_size)

    def forward(self, que, ans):
        encoder_outputs, hidden = self.enc(x)
        loss, attns = self.dec(ans, hidden, encoder_outputs)
        return loss, attns


# encoder_outputs, hidden = e(x)
# print(x.size())
# print(encoder_outputs.size())  # [3, 24, 100]
# print(hidden.size())  # [1, 3, 100]
#
# ad = AttnDecoder(5586, EMB_DIM, HIDDEN_SIZE, MAX_LENGTH)
# loss, attns = ad(y, hidden, encoder_outputs)
# # print(loss)
# # print(attns)
# # if use_teacher_forcing:
# #         # Teacher forcing
# #         for di in range(target_length):
# #             decoder_output, decoder_hidden, decoder_attention = decoder(
# #                 decoder_input, decoder_hidden, encoder_outputs)
# #             loss += criterion(decoder_output, target_tensor[di])
# #             decoder_input = target_tensor[di]  # 把输入换为目标语句中的单词
# #
# # else:
# #         # Without teacher forcing
# #         for di in range(target_length):
# #             decoder_output, decoder_hidden, decoder_attention = decoder(
# #                 decoder_input, decoder_hidden, encoder_outputs)
# #             topv, topi = decoder_output.topk(1)
# #             decoder_input = topi.squeeze().detach()  # 输入为预测出的单词
