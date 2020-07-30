# -*- coding: utf-8 -*-

word_d = {"SOS": 0, "EOS": 1, "PAD": 2}
import pickle
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import os
import torch
path = ["data/chatdata_all.txt", "data/xiaohuangji_chatbot_data5.txt"]
BATCH_SIZE = 10

def load_data(path):
    global word_d
    questions, answers = [], []
    for p in path:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                q, a = line.split("@@")
                que, ans = [0], [0]
                q_words = q.split(" ")

                for word in q_words:
                    if word not in word_d:
                        word_d[word] = len(word_d)
                        que.append(word_d[word])
                    else:
                        que.append(word_d[word])
                que.append(1)

                a_words = a.split(" ")
                for word in a_words:
                    if word not in word_d:
                        word_d[word] = len(word_d)
                        ans.append(word_d[word])
                    else:
                        ans.append(word_d[word])
                ans.append(1)
                questions.append(que)
                answers.append(ans)
    questions = padding(questions)
    answers = padding(answers)

    d = {}
    d["word_d"] = word_d
    d["que"] = questions
    d["ans"] = answers
    with open("data/data.pkl", "wb") as f:
        pickle.dump(d, f)


def padding(input_list):
    max_len = max(len(i) for i in input_list)
    input_list = [i + [word_d["PAD"]] * (max_len - len(i)) for i in input_list]
    return input_list


def make_data_set():
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
    return dataloader
if __name__ == '__main__':
    # load_data(path)
    dataloader =  make_data_set()
    for a,b in dataloader:
        print(a.size())
        print(b.size())
        break