# -*- coding: utf-8 -*-

from dataproc import make_dataloader
from model import Seq2Seq
from const import *
from torch.optim import Adam
import torch
import random
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

Save_Path = 's2s.pt'


def train_model(dataloader, model, optim):
    model = model.to(DEVICE)
    for e in range(3):
        for n, (que, ans) in enumerate(dataloader):
            que = que.to(DEVICE)
            ans = ans.to(DEVICE)
            model.train()
            optim.zero_grad()

            loss, attns = model(que, ans)
            loss.backward()
            if (n + 1) % 100 == 0:
                print(f"Epoch: {e} step: {n}/{len(dataloader)} loss: {loss.tolist() :2f}")
                torch.save(model.state_dict(), Save_Path)

                assert 0
            optim.step()
    torch.save(model.state_dict(), Save_Path)


def show_attn(input_words, output_words, attentions):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #     show_plot_visdom()
    plt.show()
    plt.close()


def train():
    dataloader, data = make_dataloader()
    word_2_idx = data["word_2_idx"]
    max_length = data["a_maxlen"]
    model = Seq2Seq(len(word_2_idx), EMB_DIM, HIDDEN_SIZE, max_length)
    # print()
    if not os.path.exists(Save_Path):
        optim = Adam(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-3)
        train_model(dataloader, model, optim)
    else:
        model.load_state_dict(torch.load(Save_Path))
    # predict

    idx_2_word = data["idx_2_word"]
    data_randint = random.randint(0, len(dataloader) - 1)
    idx_randint_list = random.sample(range(0, BATCH_SIZE), 3)

    random_que, random_ans = list(dataloader)[data_randint]
    attns = []
    for i in idx_randint_list:
        que = random_que[i]
        question = [idx_2_word.get(i) for i in que.tolist() if i != word_2_idx["PAD"]]
        que = que.unsqueeze(0)
        length = len(question)
        output, attn = model.predict(que)
        attn = torch.tensor(attn)
        attn = attn[:, :length]
        attn = F.softmax(attn, 0)

        pre_ans = [idx_2_word.get(i) for i in output if i != word_2_idx["PAD"]]
        attns.append(attn)
        print(f"问题: {''.join(question[1:-1])}")
        print(f"预测结果: {''.join(pre_ans)}")
        show_attn(question, pre_ans,attn)
        break


# def predict():
#     model = torch.load(Save_Path, map_location=DEVICE)


if __name__ == '__main__':
    train()
