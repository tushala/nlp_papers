# -*- coding: utf-8 -*-
from torch.optim import Adam
from utils import *
import numpy as np
from models.SG import SkipGram
from models.CBow import CBow
from models.Glove import GloVe
from models.HS import HS
import torch.nn.functional as F
from torch import nn
import random
import argparse


def train(model: nn.Module, train_loader, vocabs_or_ns, un_table=None, w2i=None):

    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCH):
        losses = []
        for batch_idx, (inputs, target) in enumerate(train_loader):
            model.zero_grad()
            if not isinstance(vocabs_or_ns, torch.Tensor):
                vocabs_or_ns = negative_sampling(target, un_table, w2i)

            loss = model(inputs, target, torch.tensor(vocabs_or_ns, dtype=torch.long))
            # print(999, loss)
            loss.backward()
            optimizer.step()
            # assert 0
            losses.append(loss.tolist())
        print(f"Epoch : {epoch}, mean_loss : {np.mean(losses):.2f}")


def glove_train(model: nn.Module, train_loader):
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCH):
        losses = []
        for batch_idx, (inputs, target, cooc, weight) in enumerate(train_loader):
            model.zero_grad()
            loss = model(inputs, target, cooc, weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.tolist())
        print(f"Epoch : {epoch}, mean_loss : {np.mean(losses):.2f}")


def predict(model: nn.Module, target, vocab, index2word):
    target_V = model.predict(torch.tensor(target).unsqueeze(0))
    target = index2word[target]
    similarities = []
    for i in range(len(vocab)):
        if index2word[i] == target:
            continue

        vector = model.predict(torch.tensor(i).unsqueeze(0))
        cosine_sim = F.cosine_similarity(target_V, vector).tolist()
        similarities.append([index2word[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["SG", "CB", "Glove"], type=str, required=True)
    parser.add_argument("--us", choices=["NS", "HS"], type=str, default=None, help="update_system")
    args = parser.parse_args()

    data_loader, words, w2i, i2w, tensor_code_d_or_neg_func = make_train_data(args)
    vocab = list(words.keys())
    vocab = prepare_word(vocab, w2i)
    vocabs = torch.tensor(vocab, dtype=torch.long)
    if args.us == "HS":
        model = HS(vocabs.size(0), EMBEDDING_DIM, tensor_code_d_or_neg_func)
        train(model, data_loader, vocabs)
    elif args.model == "SG":
        model = SkipGram(vocabs.size(0), EMBEDDING_DIM, args.us)
        if args.us == "NS":
            un_table = get_un_table(words)
            train(model, data_loader, tensor_code_d_or_neg_func, un_table, w2i)
        else:
            train(model, data_loader, vocabs, None, w2i)
    elif args.model == "CB":
        model = CBow(vocabs.size(0), EMBEDDING_DIM, args.us)
        if args.us == "NS":
            un_table = get_un_table(words)
            train(model, data_loader, tensor_code_d_or_neg_func, un_table, w2i)
        else:
            train(model, data_loader, vocabs)
    else:
        model = GloVe(vocabs.size(0), EMBEDDING_DIM)
        glove_train(model, data_loader)

    for i in range(5):
        test = random.choice(list(vocab))
        target = i2w[test]
        print(f"{target}:")
        sim_result = predict(model, test, vocab, i2w)
        print(sim_result)


if __name__ == '__main__':
    main()
