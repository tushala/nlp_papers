# -*- coding: utf-8 -*-
from torch.optim import Adam
from utils import *
import numpy as np
from models.SG import SkipGram
from models.CBow import CBow
import torch.nn.functional as F
from torch import nn
import random


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


def predict(model: nn.Module, target, vocab, index2word):
    target_V = model.predict(target)

    similarities = []
    for i in range(len(vocab)):
        if index2word[i] == target:
            continue

        vector = model.predict(torch.tensor(i).unsqueeze(0))
        cosine_sim = F.cosine_similarity(target_V, vector).tolist()
        similarities.append([index2word[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


def main(sg=True):
    data_loader, vocab, w2i, i2w = make_train_data(sg)
    vocabs = torch.tensor(vocab, dtype=torch.long)

    if sg:
        model = SkipGram(vocabs.size(0) + 1, 300)
    else:
        model = CBow(vocabs.size(0) + 1, 300)
    # train(model, data_loader, vocabs)

    for i in range(5):
        test = random.choice(list(vocab))
        target = i2w[test]
        print(f"{target}:")
        sim_result = predict(model, torch.tensor(test).unsqueeze(0), vocab, i2w)
        print(sim_result)


if __name__ == '__main__':
    main()
    main(False)
