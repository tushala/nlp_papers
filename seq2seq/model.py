# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from const import *
from dataproc import word_2_idx


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
    def __init__(self, vocab_size, emb_dim, hidden_size, output_length):
        super(AttnDecoder, self).__init__()
        self.output_length = output_length
        self.dec_emb = nn.Embedding(vocab_size, emb_dim)
        self.dec_gru = nn.GRU(emb_dim, hidden_size, 1, batch_first=True)
        self.dropout_p = nn.Dropout(0.1)
        self.attn = nn.Linear(hidden_size * 2, output_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, decoder_inputs, hidden, encoder_outputs, teacher_force=True, train=True):
        """
        :param decoder_inputs: b * 1
        :param hidden:  b*L*h
        :param encoder_outputs: 1*b*h
        :return:
        """
        def init_hidden():
            return torch.zeros(hidden.size())
        device = decoder_inputs.device
        loss = torch.tensor(0.).to(device)
        attns = []
        embedded = self.dec_emb(decoder_inputs)
        hidden = init_hidden().to(device)
        decode_input = embedded[:, 0, :]  # <start>
        output_length = self.output_length - 1 if train else 1
        for i in range(output_length):
            attn = self.attn(torch.cat((decode_input, hidden[0]), 1))
            attn_weights = F.softmax(attn, dim=1)
            attns.append(attn_weights.squeeze(0).tolist())
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
        return loss, attns, hidden, output


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, output_length):
        super(Seq2Seq, self).__init__()
        self.output_length = output_length
        self.enc = Encoder(vocab_size, emb_dim, hidden_size)
        self.dec = AttnDecoder(vocab_size, emb_dim, hidden_size, output_length)

    def forward(self, que, ans, teacher_force=True):
        encoder_outputs, hidden = self.enc(que)
        loss, attns, _, _ = self.dec(ans, hidden, encoder_outputs, teacher_force)
        loss /= self.output_length
        return loss, attns

    def predict(self, que):
        encoder_outputs, hidden = self.enc(que)
        res = []
        attns = []
        cur_inputs = torch.tensor([word_2_idx["SOS"]]).unsqueeze(0)
        while len(res) < MAX_LENGTH and cur_inputs.squeeze(0).tolist() != word_2_idx["EOS"]:
            _, attn, hidden, nxt_predict = self.dec(cur_inputs, hidden, encoder_outputs, train=False)
            nxt_predict = nxt_predict.squeeze(0)

            _, cur_inputs = nxt_predict.topk(1)
            res.append(cur_inputs.tolist()[0])
            cur_inputs = cur_inputs.unsqueeze(0)
            attns.extend(attn)

        return res, attns
