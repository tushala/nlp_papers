# -*- coding: utf-8 -*-
from torch.optim import Adam
from utils import *
import numpy as np
from models.SG import SkipGram
from models.CBow import CBow
from models.Glove import GloVe
import torch.nn.functional as F
from torch import nn
import random
import argparse


def train(model: nn.Module, train_loader, vocabs):
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(EPOCH):
        losses = []
        for batch_idx, (inputs, target) in enumerate(train_loader):
            model.zero_grad()
            loss = model(inputs, target, vocabs)
            loss.backward()
            optimizer.step()
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
    parser.add_argument("--update_system", choices=["NS", "HS"], type=str, required=True)
    args = parser.parse_args()
    data_loader, vocab, w2i, i2w = make_train_data(args)
    vocabs = torch.tensor(vocab, dtype=torch.long)

    if args.model == "SG":
        model = SkipGram(vocabs.size(0) + 1, EMBEDDING_DIM)
        train(model, data_loader, vocabs)
    elif args.model == "CB":
        model = CBow(vocabs.size(0) + 1, EMBEDDING_DIM)
        train(model, data_loader, vocabs)
    else:
        model = GloVe(vocabs.size(0) + 1, EMBEDDING_DIM)
        glove_train(model, data_loader)
    for i in range(5):
        test = random.choice(list(vocab))
        target = i2w[test]
        print(f"{target}:")
        sim_result = predict(model, test, vocab, i2w)
        print(sim_result)


if __name__ == '__main__':
    main()
