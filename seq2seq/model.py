# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from const import MAX_LENGTH

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
        self.dec_gru = nn.GRU(emb_dim, hidden_size, 1)
        self.dropout_p = nn.Dropout(0.1)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, decoder_inputs, hidden, encoder_outputs):
        """

        :param decoder_inputs: b * 1
        :param hidden:  b*L*h
        :param encoder_outputs: 1*b*h
        :return:
        """
        embedded = self.dec_emb(decoder_inputs)
        embedded = self.dropout(embedded)
        hidden = self._init_hidden()
        # 全teacher_force
        for i in range(self.max_length):
            _, hidden = self.dec_gru(embedded, hidden)
            attn = self.attn(torch.cat((embedded, hidden), 1))
            attn_weights = F.softmax(attn)
            attn_applied = attn_weights.bmm(hidden)
            concated = torch.cat((hidden, attn_applied), 1)
            hidden = self.attn_combine(concated)
            output = F.sigmoid(output)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # output = torch.cat((embedded[0], attn_applied[0]), 1)  # 添加了注意力的序列输出与词向量拼接
        # output = self.attn_combine(output).unsqueeze(0)  # 全连接层
        #
        # # output = F.relu(output)
        # output = F.sigmoid(output)
        # output, hidden = self.gru(output, hidden)  # 当前时刻的输出以及隐层
        #
        # output = F.log_softmax(self.out(output[0]), dim=1)
        # return output, hidden, attn_weights

    def _init_hidden(self):
        return 1
e = Encoder(5586, 100, 100)
x = torch.tensor([[0, 5133, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [0, 626, 5, 144, 5583, 4, 5584, 13, 4929, 5585, 4, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],[0, 626, 5, 144, 5583, 4, 5584, 13, 4929, 5585, 4, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.long)

a, b = e(x)
print(x.size())
print(a.size())
print(b.size())
