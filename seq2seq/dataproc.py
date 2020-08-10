# -*- coding: utf-8 -*-


import pickle
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import os
import torch

path = ["data/chatdata_all.txt", "data/xiaohuangji_chatbot_data5.txt"]
from const import *

word_2_idx = {"SOS": 0, "EOS": 1, "PAD": 2}


def load_data(path):
    global word_2_idx
    d = {}
    questions, answers = [], []
    for p in path:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                q, a = line.split("@@")
                q = q[:MAX_LENGTH]
                a = a[:MAX_LENGTH]
                que, ans = [0], [0]
                q_words = q.split(" ")

                for word in q_words:
                    if word not in word_2_idx:
                        word_2_idx[word] = len(word_2_idx)
                        que.append(word_2_idx[word])
                    else:
                        que.append(word_2_idx[word])
                que.append(1)

                a_words = a.split(" ")
                for word in a_words:
                    if word not in word_2_idx:
                        word_2_idx[word] = len(word_2_idx)
                        ans.append(word_2_idx[word])
                    else:
                        ans.append(word_2_idx[word])
                ans.append(1)
                questions.append(que)
                answers.append(ans)
    questions, q_maxlen = padding(questions)
    answers, a_maxlen = padding(answers)
    idx_2_word = {v: k for k, v in word_2_idx.items()}

    d["word_2_idx"] = word_2_idx
    d["idx_2_word"] = idx_2_word
    d["que"] = questions
    d["ans"] = answers
    d["q_maxlen"] = q_maxlen
    d["a_maxlen"] = a_maxlen
    with open("data/data.pkl", "wb") as f:
        pickle.dump(d, f)


def padding(input_list):
    max_len = max(len(i) for i in input_list)
    input_list = [i + [word_2_idx["PAD"]] * (max_len - len(i)) for i in input_list]
    return input_list, max_len


def make_dataloader():
    if not os.path.exists("data/data.pkl"):
        load_data(path)
    with open("data/data.pkl", "rb") as f:
        data = pickle.load(f)
    que_data, ans_data = data["que"], data["ans"]
    que_data = torch.tensor([q for q in que_data], dtype=torch.long)
    ans_data = torch.tensor([a for a in ans_data], dtype=torch.long)
    dataset = TensorDataset(que_data, ans_data)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader, data


if __name__ == '__main__':
    # load_data(path)
    dataloader = make_dataloader()
    for a, b in dataloader:
        print(a)
        print(a.size())
        print(b.size())
        break
