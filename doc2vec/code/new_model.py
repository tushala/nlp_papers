import torch
from torch import nn
from torch import optim


class model(nn.Module):
    def __init__(self, vocab_size, embed_size, sentence_length):
        super(model, self).__init__()
        self.sentence_embed = nn.Embedding(vocab_size, embed_size)
        self.para_num = 75000
        self.sentence_embed = nn.Embedding(self.para_num, embed_size)
        self.train_lin = nn.Linear(sentence_length * embed_size, vocab_size)
        self.mlp_lin = nn.Linear(embed_size, 50)
        self.relu = nn.ReLU()
        self.mlp_out = nn.Linear(50, 2)
        
    def forward(self, sentence, sentence_label, word_label):
        batch_size = sentence.size(0)
        # label_size = word_label.size(1)
        sentence_embed = self.sentence_embed(sentence)
        para_embed = self.sentence_embed(sentence_label)
        
        input = torch.cat([sentence_embed, para_embed], 1)  # batch_size * 10 * 400
        input = input.view(batch_size, -1)  # batch_size * 4000
        output = self.train_lin(input)
        return output

    def mlp(self, sentence_label):
        batch_size = sentence_label.size(0)
        para_embed = self.sentence_embed(sentence_label)
        para_embed = para_embed.view(batch_size, -1)
        mlp_output = self.relu(self.mlp_lin(para_embed))
        mlp_output = self.mlp_out(mlp_output)
        return mlp_output